"""`python -m lofi_rl.gate1 tally BATCH_DIR --votes FILE [--json OUT]`: Shresth's ear check of a gate1 batch, counted.

- **His standing vote per round:** `ui.votes.active_votes` over the batch's log, so a vote he took back (Backspace
  twice, a `retract` event) no longer counts and the vote he gave after it does.
- **Heard** is D-013 as amended (Shresth, 2026-10-31): a clip counts as heard once 1 bar has played, over every
  attempt of its round, undone votes and abandoned attempts included (`ui.audit.heard_over_attempts`). A tier given
  to a clip that never played is left out of the counts and reported apart, as the shootout tallies left out a mark
  on a clip that never played; every clip under 1 bar is listed apart, voted or not.
- **The counts:** per tier (Toss, Fine, Good, Great: TASTE.md, Tiers) over the heard, voted rounds; the "fine" count is
  Fine, Good or Great (D-021), against the bar of 15 of 50 (CLAUDE.md §9's 30 %). The bar is a count: a batch with
  fewer rounds (a clip that did not render) still needs 15.
- **The split** (D-021): the same counts for the clips that carry a verbatim A or B line of the example their
  prompt showed (the hidden manifest's `verbatim_line`) and for those that do not.
- **Bad input** (exit 2, nothing decided): a folder that is not a gate1 triage batch, a missing votes file, or a log
  that is not this batch's: an event of another batch, a vote of another mode or with a tier that is not one of the
  four, a round the manifest does not hold, or a vote whose clip is not that round's (its opaque id or its real clip
  id).

The manifests are read for their labels only (`check_files=False`, as the shootout tally reads them), so a batch
whose FLACs were cleared after voting still tallies. Text goes to the terminal. `--json OUT` writes the tally as JSON
instead (`-` prints it): its per-round rows pair his tiers with sample indices, which are private (D-010), so inside
the repo OUT must be under data/ (gitignored, as the votes are), e.g. data/votes/audit/. Exit 0 when every round has a
standing vote and the "fine" count meets the bar, 3 otherwise. Sound claims: none.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lofi_rl import gate1 as G
from lofi_rl.palette import ROOT
from lofi_rl.ui.audit import heard_bars, heard_over_attempts
from lofi_rl.ui.manifest import ManifestError, load_batch
from lofi_rl.ui.votes import VOTE_TYPES, active_votes, read_events

SCHEMA = "lofi-rl/gate1-tally/1"
MODE = "triage"


def _split_counts(rows: list[dict]) -> dict:
    tiers = {t: sum(1 for r in rows if r["tier"] == t) for t in G.TIERS}
    return {"n": len(rows), "tiers": tiers, "fine_or_better": sum(tiers[t] for t in G.FINE_OR_BETTER)}


def _rounds_of(m, h, batch_dir: Path) -> tuple[dict[str, dict], list[str]]:
    """round id -> its hidden clip, and the problems of a batch that is not a gate1 triage batch."""
    problems: list[str] = []
    if m.mode != MODE:
        problems.append(f"{batch_dir}: a {m.mode} batch, not a {MODE} batch")
    if h.category != G.CATEGORY:
        problems.append(f"{batch_dir}: hidden category {h.category!r}, not {G.CATEGORY!r}: not a gate1 batch")
    infos: dict[str, dict] = {}
    for r in m.rounds:
        hr = h.rounds[r.id]
        if len(r.ids) != 1:
            problems.append(f"round {r.id}: {len(r.ids)} clips; a triage round holds one")
            continue
        info = hr.clips[r.ids[0]]
        if not isinstance(info.get("verbatim_line"), bool) or not G._whole(info.get("sample")):
            problems.append(f"round {r.id}: its hidden clip has no gate1 fields (`sample`, `verbatim_line`)")
        infos[r.id] = info
    return infos, problems


def _log_problems(events: list[dict], m, h) -> list[str]:
    """Why the log is not this batch's (module doc); [] when it is."""
    problems: list[str] = []
    known = {r.id: r for r in m.rounds}
    for e in events:
        t, eid = e.get("type"), e.get("event_id")
        if e.get("batch") != m.batch:
            problems.append(f"event {eid}: batch {e.get('batch')!r}, not {m.batch!r}")
            continue
        if t in VOTE_TYPES and t != MODE:
            problems.append(f"event {eid}: a {t} vote in a {MODE} batch")
            continue
        if t == MODE and e.get("tier") not in G.TIERS:
            problems.append(f"event {eid}: tier {e.get('tier')!r} is not one of {list(G.TIERS)}")
        if t in VOTE_TYPES or t == "abandon":
            rid = e.get("round")
            if rid not in known:
                problems.append(f"event {eid}: round {rid!r} is not in the batch")
                continue
            oid = known[rid].ids[0]
            real = h.rounds[rid].real(oid)
            for c in e.get("clips") or []:
                if not isinstance(c, dict) or c.get("oid") != oid or c.get("clip") != real:
                    got = (c.get("oid"), c.get("clip")) if isinstance(c, dict) else c
                    problems.append(f"event {eid}: clip {got!r} is not round {rid}'s clip {oid} ({real})")
    return problems


def tally(batch_dir: Path | str, votes: Path | str) -> dict:
    """The tally document (module doc). Gate1Error lists every problem of bad input."""
    batch_dir, votes = Path(batch_dir), Path(votes)
    try:
        m, h = load_batch(batch_dir, check_files=False)
    except ManifestError as err:
        raise G.Gate1Error([f"{batch_dir}: {err}"]) from None
    infos, problems = _rounds_of(m, h, batch_dir)
    if not votes.is_file():
        problems.append(f"{votes} is not a file: the batch's vote log is data/votes/{m.batch}.jsonl")
    if problems:
        raise G.Gate1Error(problems)
    events, unreadable = read_events(votes)
    problems = _log_problems(events, m, h)
    if problems:
        raise G.Gate1Error(problems)

    act = active_votes(events)
    heard = heard_over_attempts(events)
    rows = []
    for r in m.rounds:
        info, v = infos[r.id], act.get(r.id)
        bars = heard.get((r.id, info["clip"]), "")
        n_bars = heard_bars({"heard_bars": bars})
        rows.append({"round": r.id, "number": r.number, "clip": r.ids[0], "sample": info["sample"],
                     "verbatim_line": info["verbatim_line"], "tier": v.get("tier") if v else None,
                     "event_id": v["event_id"] if v else None, "heard_bars": bars or None, "bars_heard": n_bars,
                     "heard": n_bars >= G.HEARD_MIN_BARS})
    counted = [r for r in rows if r["tier"] is not None and r["heard"]]
    not_voted = [r for r in rows if r["tier"] is None]
    whole = _split_counts(counted)
    fine = whole["fine_or_better"]
    complete = not not_voted
    met = fine >= G.FINE_BAR
    return {
        "schema": SCHEMA, "batch": m.batch, "dir": str(batch_dir), "votes": str(votes), "rounds": len(m.rounds),
        "voted": len(rows) - len(not_voted), "heard": sum(1 for r in rows if r["heard"]), "counted": len(counted),
        "complete": complete,
        "heard_rule": f"at least {G.HEARD_MIN_BARS} bar over every attempt of its round (D-013 as amended)",
        "tiers": whole["tiers"], "fine_or_better": fine,
        "bar": {"need": G.FINE_BAR, "of": G.CLIPS, "met": met},
        "split": {"verbatim_line": _split_counts([r for r in counted if r["verbatim_line"]]),
                  "no_verbatim_line": _split_counts([r for r in counted if not r["verbatim_line"]])},
        "not_heard": [{"number": r["number"], "round": r["round"], "sample": r["sample"], "voted": r["tier"] is not None,
                       "tier": r["tier"], "bars_heard": r["bars_heard"]} for r in rows if not r["heard"]],
        "not_voted": [{"number": r["number"], "round": r["round"], "sample": r["sample"]} for r in not_voted],
        "retracted": sum(1 for e in events if e.get("type") == "retract"),
        "abandoned": sum(1 for e in events if e.get("type") == "abandon"),
        "unreadable_lines": len(unreadable),
        "per_round": rows,
        "exit": 0 if complete and met else 3,
    }


def tiers_text(t: dict) -> str:
    """Great, Good, Fine, Toss, as the gate files print a tier count."""
    return ", ".join(f"{name.capitalize()} {t[name]}" for name in reversed(G.TIERS))


def format_text(doc: dict) -> str:
    s, b = doc["split"], doc["bar"]
    lines = [f"gate1 tally: {doc['batch']} ({doc['votes']})",
             f"  rounds {doc['rounds']}; voted {doc['voted']}; heard {doc['heard']}; counted {doc['counted']} (voted "
             f"and heard: {doc['heard_rule']})" + ("" if doc["complete"] else "; NOT complete"),
             f"  tiers: {tiers_text(doc['tiers'])}",
             f"  \"fine\" (Fine, Good or Great): {doc['fine_or_better']}; the bar is at least {b['need']} of {b['of']}: "
             f"{'met' if b['met'] else 'NOT met'}"]
    for key, label in (("verbatim_line", "with a verbatim A or B line"), ("no_verbatim_line", "with neither line")):
        x = s[key]
        lines.append(f"  {label}: \"fine\" {x['fine_or_better']} of {x['n']} ({tiers_text(x['tiers'])})")
    if doc["not_heard"]:
        lines.append("  not heard (under 1 bar): " + ", ".join(
            f"round {r['number']}" + (" (its vote is left out)" if r["voted"] else "") for r in doc["not_heard"]))
    if doc["not_voted"]:
        lines.append("  not voted: " + ", ".join(f"round {r['number']}" for r in doc["not_voted"]))
    lines.append(f"  retracted {doc['retracted']}, abandoned {doc['abandoned']}, unreadable lines "
                 f"{doc['unreadable_lines']}")
    lines.append(f"GATE1_TALLY {'complete' if doc['complete'] else 'incomplete'}, \"fine\" {doc['fine_or_better']}, "
                 f"bar {'met' if b['met'] else 'not met'}")
    return "\n".join(lines) + "\n"


def private_path_problem(out: Path) -> str | None:
    """Why `out` may not hold his per-round tiers: inside the repo, only under data/ (gitignored, D-010)."""
    path = out.resolve()
    root = ROOT.resolve()
    if path.is_relative_to(root) and not path.is_relative_to(root / "data"):
        return (f"--json {out}: his per-round tiers are private (D-010); inside the repo they go under data/ "
                "(e.g. data/votes/audit/), never into a committed folder")
    return None


def main(a: argparse.Namespace) -> int:
    """The `tally` subcommand (lofi_rl.gate1.cli)."""
    if a.json is not None and str(a.json) != "-":
        bad = private_path_problem(a.json)
        if bad:
            print(f"error: nothing written: {bad}", file=sys.stderr)
            return 2
    try:
        doc = tally(a.batch, a.votes)
    except G.Gate1Error as err:
        print(f"error: bad input, nothing decided:\n{err}", file=sys.stderr)
        return 2
    if a.json is None:
        print(format_text(doc), end="")
    elif str(a.json) == "-":
        print(json.dumps(doc, indent=1))
    else:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        print(format_text(doc), end="")
        print(f"wrote {a.json}")
    return doc["exit"]


def add_parser(sub) -> None:
    p = sub.add_parser("tally", help="his standing votes on a gate1 batch: the tiers, the \"fine\" count, the split")
    p.add_argument("batch", type=Path, help="the batch folder, inbox/<date>/b<MMDD>-gate1")
    p.add_argument("--votes", type=Path, required=True, help="its vote log, data/votes/<batch>.jsonl")
    p.add_argument("--json", type=Path, default=None, metavar="OUT",
                   help="write the tally as JSON to OUT (under data/ inside the repo), or - to print it")
    p.set_defaults(run_command=main)
