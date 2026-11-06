"""`python -m lofi_rl.gate1 tally BATCH_DIR --votes FILE [--json OUT]`: Shresth's ear check of a gate1 batch, counted.

- **His standing vote per round:** `ui.votes.active_votes` over the batch's log, so a vote he took back (Backspace
  twice, a `retract` event) no longer counts and the vote he gave after it does.
- **Heard** is D-013 as amended (Shresth, 2026-10-31): a clip counts as heard once 1 bar has played, over every
  attempt of its round, undone votes and abandoned attempts included (`ui.audit.heard_over_attempts`). A tier given
  to a clip that never played is left out of the counts and reported apart, as the shootout tallies left out a mark
  on a clip that never played; so is every clip under 1 bar.
- **The counts:** per tier (Toss, Fine, Good, Great: TASTE.md, Tiers) over the heard, voted rounds; the "fine" count is
  Fine, Good or Great (D-021), against the bar of 15 of 50 (CLAUDE.md §9's 30 %).
- **The split** (D-021): the same counts for the clips that carry a verbatim A or B line of the example their
  prompt showed (the hidden manifest's `verbatim_line`) and for those that do not.
- **Bad input** (exit 2, nothing decided): a folder that is not a gate1 triage batch, a missing votes file, or a log
  that is not this batch's: an event of another batch, a vote of another mode, a round the manifest does not hold,
  or a vote whose clip is not that round's (its opaque id or its real clip id).

Text goes to the terminal; `--json OUT` also writes the tally as JSON (per-round rows included: his tiers are private,
so OUT belongs under data/votes/audit/). Exit 0 when every round has a standing vote and the "fine" count meets the
bar, 3 otherwise. Sound claims: none.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lofi_rl import gate1 as G
from lofi_rl.ui.audit import heard_bars, heard_over_attempts
from lofi_rl.ui.manifest import ManifestError, load_batch
from lofi_rl.ui.votes import EVENT_TYPES, VOTE_TYPES, active_votes, read_events

SCHEMA = "lofi-rl/gate1-tally/1"
MODE = "triage"


def _split_counts(rows: list[dict]) -> dict:
    tiers = {t: sum(1 for r in rows if r["tier"] == t) for t in G.TIERS}
    return {"n": len(rows), "tiers": tiers, "fine_or_better": sum(tiers[t] for t in G.FINE_OR_BETTER)}


def tally(batch_dir: Path | str, votes: Path | str) -> dict:
    """The tally document (module doc). Gate1Error lists every problem of bad input."""
    batch_dir, votes = Path(batch_dir), Path(votes)
    try:
        m, h = load_batch(batch_dir)
    except ManifestError as err:
        raise G.Gate1Error([f"{batch_dir}: {err}"]) from None
    problems: list[str] = []
    if m.mode != MODE:
        problems.append(f"{batch_dir}: a {m.mode} batch, not a {MODE} batch")
    if h.category != G.CATEGORY:
        problems.append(f"{batch_dir}: hidden category {h.category!r}, not {G.CATEGORY!r}: not a gate1 batch")
    infos: dict[str, dict] = {}
    for r in m.rounds:
        hr = h.rounds[r.id]
        (oid,) = r.ids
        info = hr.clips[oid]
        if not isinstance(info.get("verbatim_line"), bool) or not isinstance(info.get("sample"), int):
            problems.append(f"round {r.id}: its hidden clip has no gate1 fields (`sample`, `verbatim_line`)")
        infos[r.id] = info
    if not votes.is_file():
        problems.append(f"{votes} is not a file: the batch's vote log is data/votes/{m.batch}.jsonl")
    if problems:
        raise G.Gate1Error(problems)
    events, unreadable = read_events(votes)
    known = {r.id: r for r in m.rounds}
    for e in events:
        t, eid = e.get("type"), e.get("event_id")
        if e.get("batch") != m.batch:
            problems.append(f"event {eid}: batch {e.get('batch')!r}, not {m.batch!r}")
            continue
        if t in VOTE_TYPES and t != MODE:
            problems.append(f"event {eid}: a {t} vote in a {MODE} batch")
            continue
        rid = e.get("round")
        if t in VOTE_TYPES or t == "abandon":
            if rid not in known:
                problems.append(f"event {eid}: round {rid!r} is not in the batch")
                continue
            r = known[rid]
            for c in e.get("clips") or []:
                if c.get("oid") != r.ids[0] or c.get("clip") != h.rounds[rid].real(r.ids[0]):
                    problems.append(f"event {eid}: clip {c.get('oid')!r} ({c.get('clip')!r}) is not round {rid}'s "
                                    f"clip {r.ids[0]} ({h.rounds[rid].real(r.ids[0])})")
        elif t not in EVENT_TYPES:
            problems.append(f"event {eid}: type {t!r}")
    if problems:
        raise G.Gate1Error(problems)

    act = active_votes(events)
    heard = heard_over_attempts(events)
    rows = []
    for r in m.rounds:
        info, v = infos[r.id], act.get(r.id)
        bars = heard.get((r.id, info["clip"]), "")
        n_bars = heard_bars({"heard_bars": bars})
        tier = v.get("tier") if v else None
        rows.append({"round": r.id, "number": r.number, "clip": r.ids[0], "sample": info["sample"],
                     "verbatim_line": info["verbatim_line"], "tier": tier, "event_id": v["event_id"] if v else None,
                     "heard_bars": bars or None, "bars_heard": n_bars, "heard": n_bars >= G.HEARD_MIN_BARS})
    counted = [r for r in rows if r["tier"] is not None and r["heard"]]
    left_out = [r for r in rows if r["tier"] is not None and not r["heard"]]
    not_voted = [r for r in rows if r["tier"] is None]
    not_heard = [r for r in rows if not r["heard"]]
    whole = _split_counts(counted)
    fine = whole["fine_or_better"]
    complete = not not_voted
    return {
        "schema": SCHEMA, "batch": m.batch, "dir": str(batch_dir), "votes": str(votes), "rounds": len(m.rounds),
        "voted": len(rows) - len(not_voted), "counted": len(counted), "complete": complete,
        "heard_rule": f"at least {G.HEARD_MIN_BARS} bar over every attempt of the round (D-013 as amended)",
        "tiers": whole["tiers"], "fine_or_better": fine,
        "bar": {"need": G.FINE_BAR, "of": G.CLIPS, "met": fine >= G.FINE_BAR},
        "split": {"verbatim_line": _split_counts([r for r in counted if r["verbatim_line"]]),
                  "no_verbatim_line": _split_counts([r for r in counted if not r["verbatim_line"]])},
        "not_heard": [{"number": r["number"], "round": r["round"], "sample": r["sample"], "tier": r["tier"],
                       "bars_heard": r["bars_heard"]} for r in not_heard],
        "left_out": [{"number": r["number"], "round": r["round"], "sample": r["sample"], "tier": r["tier"]}
                     for r in left_out],
        "not_voted": [{"number": r["number"], "round": r["round"], "sample": r["sample"]} for r in not_voted],
        "retracted": sum(1 for e in events if e.get("type") == "retract"),
        "abandoned": sum(1 for e in events if e.get("type") == "abandon"),
        "unreadable_lines": len(unreadable),
        "per_round": rows,
        "exit": 0 if complete and fine >= G.FINE_BAR else 3,
    }


def _tiers(t: dict) -> str:
    return ", ".join(f"{name.capitalize()} {t[name]}" for name in reversed(G.TIERS))


def format_text(doc: dict) -> str:
    s, b = doc["split"], doc["bar"]
    lines = [f"gate1 tally: {doc['batch']} ({doc['votes']})",
             f"  rounds {doc['rounds']}; voted {doc['voted']}; counted {doc['counted']} (voted and heard: "
             f"{doc['heard_rule']})" + ("" if doc["complete"] else "; NOT complete"),
             f"  tiers: {_tiers(doc['tiers'])}",
             f"  \"fine\" (Fine, Good or Great): {doc['fine_or_better']} of {doc['rounds']} rounds; the bar is at least "
             f"{b['need']} of {b['of']}: {'met' if b['met'] else 'NOT met'}"]
    for key, label in (("verbatim_line", "with a verbatim A or B line"), ("no_verbatim_line", "with neither line")):
        x = s[key]
        lines.append(f"  {label}: \"fine\" {x['fine_or_better']} of {x['n']} ({_tiers(x['tiers'])})")
    if doc["not_heard"]:
        lines.append("  not heard (under 1 bar): " + ", ".join(
            f"round {r['number']}" + (f" (a {r['tier']} vote, left out)" if r["tier"] else "")
            for r in doc["not_heard"]))
    if doc["not_voted"]:
        lines.append("  not voted: " + ", ".join(f"round {r['number']}" for r in doc["not_voted"]))
    lines.append(f"  retracted {doc['retracted']}, abandoned {doc['abandoned']}, unreadable lines "
                 f"{doc['unreadable_lines']}")
    lines.append(f"GATE1_TALLY {'complete' if doc['complete'] else 'incomplete'}, \"fine\" {doc['fine_or_better']}, "
                 f"bar {'met' if b['met'] else 'not met'}")
    return "\n".join(lines) + "\n"


def main(a: argparse.Namespace) -> int:
    """The `tally` subcommand (lofi_rl.gate1.cli)."""
    try:
        doc = tally(a.batch, a.votes)
    except G.Gate1Error as err:
        print(f"error: bad input, nothing decided:\n{err}", file=sys.stderr)
        return 2
    print(format_text(doc), end="")
    if a.json is not None:
        text = json.dumps(doc, indent=1) + "\n"
        if str(a.json) == "-":
            print(text, end="")
        else:
            a.json.parent.mkdir(parents=True, exist_ok=True)
            a.json.write_text(text, encoding="utf-8")
    return doc["exit"]
