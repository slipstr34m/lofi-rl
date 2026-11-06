"""`python -m lofi_rl.gate1 gate --run RUN_DIR [--batch BATCH_DIR --votes FILE] --write gates/GATE_phase1.md`: the
Phase-1 gate file, in gates/GATE_palette.md's style.

The file, in order:
1. the bar and where it comes from (CLAUDE.md §9; D-019, D-020 as amended, D-021 "What follows" item 1, D-022
   "Consequences"), and what passing takes: both halves, and his line;
2. the run: its job, node and GPU, the stamp generate and score ran from (the job snapshot's), the palette label, the
   prompt and template, the grammar and its backend (recorded, not judged), the briefs and the files read;
3. the validity table (lofi_rl.gate1.check: every condition by name, what junk.json shows, what it needs);
4. the bar's first half: `pass_not_copy` with its CI as junk.json gives it;
5. beside the bar (D-021, never gated): the verbatim A-or-B share among the passing non-copies and among the clips;
6. his ear check, once tallied (lofi_rl.gate1.tally): the batch, how many were heard, the counts per tier, the "fine"
   count against 15 of 50, the split by verbatim line, the clips not heard and the rounds not voted, where his votes
   are; before that, what it waits on;
7. where the gate stands, in one item, from the numbers above;
8. `## How to approve`, kit.gates' current wording, with the line written out for a bare date (`--date`, default
   today in Pacific time), indented so that no reader counts it;
9. `## Approval`, empty: his line is his to write.

**Rewriting** keeps everything under `## Approval`, from the line after its heading to the end of the file, byte for
byte; every line above it is written again. The heading is found as the seal finds it (lofi_rl.freeze.gate.sections:
code fences and HTML comments never read). The writer refuses (exit 2, nothing written) a file with no `## Approval`
heading or more than one (it is not a file this writer made), and a file with a line that starts as an approval
(kit.gates.ATTEMPT) above that heading, which a rewrite would drop. Writing the same text again changes nothing.

`--batch` and `--votes` go together, and the batch must be this run's: every clip's job and palette label are the
run's, and its clips are exactly the run's rendered ones. Sound claims: none.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date as _date
from pathlib import Path

from kit import gates as KG
from lofi_rl import gate1 as G
from lofi_rl.drafts.job import today_pacific
from lofi_rl.freeze.gate import sections
from lofi_rl.gate1 import check as C
from lofi_rl.gate1 import tally as T
from lofi_rl.ui.manifest import ManifestError, load_batch

APPROVAL_HEADING = "## Approval"
TITLE = "# Gate: Phase 1, the base model on the frozen palette"


def _cell(text) -> str:
    return str(text).replace("|", "\\|")


def _value(text: str) -> str:
    """A table cell for a junk.json value: a 64-hex digest shortened as the gate files print them (`09f80a64…`)."""
    return f"`{_cell(G.short(text))}`" if text not in ("", None) else ""


def _thousands(x) -> str:
    return f"{x:,}" if isinstance(x, int) and not isinstance(x, bool) else str(x)


def _method(m) -> str:
    return "Wilson" if m == "wilson" else str(m)


def batch_problems(batch_dir: Path, run: G.Run) -> list[str]:
    """Why a batch is not this run's (module doc); [] when it is."""
    try:
        m, h = load_batch(batch_dir, verify_sha=False)
    except ManifestError as err:
        return [f"{batch_dir}: {err}"]
    out, samples = [], []
    for r in m.rounds:
        for info in h.rounds[r.id].clips.values():
            samples.append(info.get("sample"))
            if str(info.get("job")) != str(run.job) or info.get("palette") != run.palette_label:
                out.append(f"round {r.id}: its clip is from job {info.get('job')} with palette {info.get('palette')}; "
                           f"the run is job {run.job} with {run.palette_label}")
    want = [rec["i"] for rec in run.rendered]
    if sorted(s for s in samples if isinstance(s, int)) != sorted(want) or len(samples) != len(want):
        out.append(f"{batch_dir} holds samples {sorted(map(str, samples))[:10]}…; the run rendered {len(want)} clips, "
                   "and the batch must hold each exactly once")
    return out


def _run_section(doc: dict, run: G.Run) -> list[str]:
    c = doc["constrained"]
    backend = c.get("backend") if isinstance(c.get("backend"), dict) else {}
    stamp = doc["stamp"]
    same = stamp["generate"] == stamp["score"]
    briefs = doc.get("briefs") or {}
    return [
        f"## The run: `lofi-junk-rate` {doc['job']}",
        "",
        f"- **Job:** {doc['job']} on {doc['node']} ({doc['gpu']}).",
        (f"- **Code:** generate and score both read the stamp `{stamp['generate']}`, the job snapshot's."
         if same else f"- **Code:** generate read `{stamp['generate']}`; score read `{stamp['score']}`."),
        f"- **Palette:** `{doc['palette']['label']}` (sha256 `{G.short(doc['palette']['sha256'])}`).",
        f"- **Prompt:** {doc['prompt']['version']}, template `{G.short(doc['prompt']['template_sha256'])}`.",
        f"- **Grammar:** enabled {C.shown(c.get('enabled'))}; regex sha256 `{G.short(c.get('regex_sha256'))}` "
        f"({_thousands(c.get('regex_chars'))} characters); backend `{backend.get('backend')}` (pin taken: "
        f"{C.shown(backend.get('pin_accepted'))}; applied: {C.shown(backend.get('applied'))}), recorded and not part "
        "of validity (D-020).",
        f"- **Briefs:** {doc.get('n')} of `{briefs.get('sampler')}({briefs.get('seed')}, {briefs.get('count')})`, one "
        "completion each.",
        f"- **Files:** `runs/{run.name}/summary/` `junk.json`, `scored.jsonl`, `completions.jsonl` and `clips.json`.",
    ]


def _validity_section(doc: dict) -> list[str]:
    v = doc["validity"]
    lines = ["## Validity (D-020's list as amended, with D-022's two swaps)", "",
             "Every condition reads `summary/junk.json`. D-022 swaps in the frozen palette's sha256 and its "
             "grammar's, and drops the One-mirror clause's 60 frozen files; the stamp condition is the one it keeps.",
             "", "| group | condition | junk.json shows | needs | holds |", "|---|---|---|---|---|"]
    for x in v["conditions"]:
        tag = "" if x["source"] == "D-020" else f" ({x['source']})"
        lines.append(f"| {x['group']} | `{_cell(x['name'])}`{tag} | {_value(x['shows'])} | {_value(x['needs'])} | "
                     f"{'yes' if x['ok'] else '**no**'} |")
    lines.append("")
    if v["valid"]:
        lines.append(f"**Valid:** all {v['of']} conditions hold.")
    else:
        lines.append(f"**Not valid:** {v['of'] - v['held']} of {v['of']} conditions fail ("
                     + ", ".join(f"`{n}`" for n in v["failed"]) + "). An invalid run is reported and decides nothing "
                     "(D-020).")
    return lines


def _bar_section(doc: dict) -> list[str]:
    b = doc["bar"]["pass_not_copy"]
    lo, hi = (b["ci95"] or [None, None])[:2]
    return ["## The bar, first half: pass_not_copy", "",
            f"- `pass_not_copy.count` **{b['count']}** of {b['of']} (rate {b['rate']}; {_method(b['method'])} 95 % CI "
            f"{lo}–{hi}, as junk.json gives it). It needs at least {b['need']}: **{'met' if b['met'] else 'not met'}**."]


def _readout_section(doc: dict) -> list[str]:
    p, k = doc["readouts"]["passing_non_copies"], doc["readouts"]["clips"]
    sampled = "" if k["n"] == k["sampled"] else f" ({k['n']} of the {k['sampled']} sampled rendered)"
    return ["## Beside the bar (D-021: reported, never gated)", "",
            "A verbatim line is the A or B line of the example the loop's own prompt showed, word for word (junk.json "
            "`copy`, scored.jsonl `verbatim`).",
            f"- **Among the passing non-copies:** {p['k']} of {p['n']} carry a verbatim A or B line "
            f"({C.pct(p['k'], p['n'])}); {p['both']} carry both.",
            f"- **Among the clips:** {k['k']} of {k['n']} ({C.pct(k['k'], k['n'])}); {k['both']} carry both{sampled}.",
            "- **His \"fine\" count by verbatim line:** in his ear check, below."]


def _tiers(t: dict) -> str:
    return ", ".join(f"{name.capitalize()} {t[name]}" for name in reversed(G.TIERS))


def _ear_section(tdoc: dict | None, run: G.Run) -> list[str]:
    if tdoc is None:
        return ["## His ear check", "",
                f"- **Not tallied yet.** The batch is built from `runs/{run.name}/summary/clips.json` "
                "(`python -m lofi_rl.gate1 batch`), one clip a round in triage mode, and counted with "
                "`python -m lofi_rl.gate1 tally` once he has voted."]
    s, b = tdoc["split"], tdoc["bar"]
    heard = tdoc["rounds"] - len(tdoc["not_heard"])
    lines = [f"## His ear check: batch {tdoc['batch']}", "",
             f"Triage mode: one clip a round, played through its seam; Toss, Fine, Good or Great (TASTE.md, Tiers). A "
             f"clip counts as heard once {G.HEARD_MIN_BARS} bar has played, over every attempt of its round (D-013 as "
             "amended).",
             f"- **Counts:** {heard} of {tdoc['rounds']} heard; {tdoc['counted']} voted and heard. {_tiers(tdoc['tiers'])}.",
             f"- **\"Fine\" (Fine, Good or Great):** {tdoc['fine_or_better']} of {tdoc['rounds']}. It needs at least "
             f"{b['need']} of {b['of']}: **{'met' if b['met'] else 'not met'}**"
             + ("." if tdoc["complete"] else f", with {len(tdoc['not_voted'])} round(s) still to vote."),
             "- **Split by verbatim line** (D-021):"]
    for key, label in (("verbatim_line", "with a verbatim A or B line"), ("no_verbatim_line", "with neither")):
        x = s[key]
        lines.append(f"  - {label}: \"fine\" {x['fine_or_better']} of {x['n']} ({_tiers(x['tiers'])}).")
    if tdoc["not_heard"]:
        lines.append("- **Not heard (under 1 bar):** " + ", ".join(
            f"round {r['number']}" + (f" (his {r['tier'].capitalize()} there is left out)" if r["tier"] else "")
            for r in tdoc["not_heard"]) + ".")
    else:
        lines.append("- **Not heard (under 1 bar):** none.")
    lines.append("- **Not voted:** " + (", ".join(f"round {r['number']}" for r in tdoc["not_voted"]) + "."
                                         if tdoc["not_voted"] else "none."))
    lines.append(f"- **Audit file** (private): `data/votes/{tdoc['batch']}.jsonl`.")
    return lines


def _standing(doc: dict, tdoc: dict | None) -> list[str]:
    valid, met = doc["validity"]["valid"], doc["bar"]["pass_not_copy"]["met"]
    if not valid:
        text = ("The run is not valid, so it decides nothing (D-020); the ear check waits for a valid run.")
    elif not met:
        text = (f"The first half is not met ({doc['bar']['pass_not_copy']['count']} < "
                f"{doc['bar']['pass_not_copy']['need']}): the gate fails on the frozen palette, and it goes to Shresth "
                "with CLAUDE.md §9's next fallback, D-021's Part B.")
    elif tdoc is None:
        text = "The first half is met. The second half waits on his ear check."
    elif not tdoc["complete"]:
        text = f"The first half is met. His ear check is not complete: {len(tdoc['not_voted'])} round(s) to vote."
    elif tdoc["bar"]["met"]:
        text = ("Both halves of the bar are met. The gate passes with his line under `## Approval`; nothing else "
                "is needed from this file.")
    else:
        text = (f"The first half is met; the ear check is not ({tdoc['fine_or_better']} \"fine\" < "
                f"{tdoc['bar']['need']}): the gate fails on the frozen palette, and it goes to Shresth with CLAUDE.md "
                "§9's next fallback, D-021's Part B.")
    return ["## Where it stands", "", f"- {text}"]


def render(doc: dict, tdoc: dict | None, run: G.Run, day: _date | str) -> str:
    """The file above `## Approval`, and the heading itself (module doc)."""
    head = [TITLE, "",
            "Phase 1 passes when:",
            f"1. a valid `lofi-junk-rate` run of prompt v2 under the grammar on `palettes/lofi_rain_v1.yaml` (frozen "
            f"2026-11-04, sha256 `{G.short(G.FROZEN_SHA256)}`) has `pass_not_copy.count` at least "
            f"{G.PASS_NOT_COPY_BAR} of {G.N};",
            f"2. at least {G.FINE_BAR} of its {G.CLIPS} clips are \"fine\" by Shresth's ear: Fine, Good or Great "
            "(TASTE.md, Tiers);",
            f"3. {G.APPROVER} has written the approval line below.",
            "",
            "**Where the bar comes from:** CLAUDE.md §9 (at least 60 % of 500 base-model samples pass the gates, and "
            "at least 30 % of a 50-clip ear check is \"fine\"); D-019 (the 60 % reads `pass_not_copy`, so copying the "
            "in-prompt example cannot carry it); D-020 as amended (300 of 500 in a valid run, and the validity "
            "list); D-021 \"What follows\" item 1 (this gate, on the frozen palette, and the three readouts beside "
            "it); D-022 \"Consequences\" (the frozen palette's two values in the list). The bar is D-020's, "
            "unchanged.",
            "",
            "Written by `python -m lofi_rl.gate1 gate` (docs/GATE1.md). Writing it again keeps everything under "
            "`## Approval` exactly as it is.",
            ""]
    body = [*_run_section(doc, run), "", *_validity_section(doc), "", *_bar_section(doc), "",
            *_readout_section(doc), "", *_ear_section(tdoc, run), "", *_standing(doc, tdoc), ""]
    how = KG.how_to_approve(G.APPROVER, day, under=APPROVAL_HEADING).rstrip("\n").split("\n")
    return "\n".join([*head, *body, *how, "", APPROVAL_HEADING]) + "\n"


def merge(existing: str | None, generated: str) -> str:
    """The new file: `generated` (which ends with the `## Approval` heading) and, from an existing file, everything
    after its `## Approval` heading, byte for byte (module doc). Gate1Error when the existing file cannot be rewritten
    without dropping something."""
    if existing is None:
        return generated
    found = sections(existing, APPROVAL_HEADING)
    if len(found) != 1:
        raise G.Gate1Error([f"the gate file has {len(found)} `{APPROVAL_HEADING}` headings, not one: it is not a file "
                            "this writer made; move it aside first"])
    first = found[0][0]
    keep = existing.splitlines(keepends=True)
    above = [(k + 1, line.rstrip("\r\n")) for k, line in enumerate(keep[:first]) if KG.ATTEMPT.match(line)]
    if above:
        raise G.Gate1Error([f"line {k} starts as an approval above `{APPROVAL_HEADING}` ({text!r}); a rewrite keeps only "
                            f"what is under `{APPROVAL_HEADING}`, so move it there first" for k, text in above])
    return generated + "".join(keep[first + 1:])


def main(a: argparse.Namespace) -> int:
    """The `gate` subcommand (lofi_rl.gate1.cli)."""
    if (a.batch is None) != (a.votes is None):
        print("error: --batch and --votes go together (the batch folder and its vote log)", file=sys.stderr)
        return 2
    try:
        day = _date.fromisoformat(a.date) if a.date else today_pacific()
        run = G.load_run(a.run)
        doc = C.evaluate(run)
        tdoc = None
        if a.batch is not None:
            bad = batch_problems(Path(a.batch), run)
            if bad:
                raise G.Gate1Error(bad)
            tdoc = T.tally(a.batch, a.votes)
        path = Path(a.write)
        try:
            existing = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            existing = None
        text = merge(existing, render(doc, tdoc, run, day))
    except ValueError as err:          # Gate1Error is a ValueError, as is a bad --date
        print(f"error: nothing written:\n{err}", file=sys.stderr)
        return 2
    if text == existing:
        print(f"{path}: unchanged")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.gate1.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        print(f"wrote {path}")
    ok = KG.gate_approved(path, G.APPROVER)
    print(f"{path}: {'APPROVED by ' + G.APPROVER if ok else 'no approval line by ' + G.APPROVER + ' yet'}")
    for number, _line, why in KG.attempts(path, G.APPROVER):
        print(f"  line {number}: {why}")
    print(f"GATE1_GATE run {doc['verdict']}; ear check "
          + ("not tallied" if tdoc is None else f"\"fine\" {tdoc['fine_or_better']} of {tdoc['rounds']}, "
             f"{'complete' if tdoc['complete'] else 'incomplete'}"))
    return 0
