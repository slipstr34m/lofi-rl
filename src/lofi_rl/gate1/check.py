"""`python -m lofi_rl.gate1 check RUN_DIR [--json]`: is the run valid, and does it meet the bar's first half?

**Validity** is D-020's amended list ("Validity: an arm counts only if its summary/junk.json shows all of the
following"), read from junk.json alone, with D-022's two swaps: `palette.sha256` is the frozen file's B
(`09f80a64…`) instead of dev_gu's, and both regex shas are the frozen grammar's `5c081828…` instead of `1412f11f…`.
D-022 also drops one clause (the One-mirror commit's 60 frozen files of docs/prereg/D-020_junk_path.txt), so the
stamp condition is the one D-022 keeps: `generate.host.stamp` and `score_stamp` show the same `commit=… dirty=0`.
Each condition is reported by name (its junk.json field) with what junk.json shows, in D-020's five groups:
- **Complete:** `n` 500, `requested` 500, `partial` false, `unreadable_lines` 0, `generate.stopped` null;
- **As registered:** `generate.model.revision` `cdbee75f…`, `briefs.seed` 2026, `sampling` equal to
  runs/junk-13632676's, `palette.sha256` B (D-022) with `palette.matches_generate` true, `prompt.version` v2 with
  `prompt.template_sha256` `127e3094…`, `prompt.template_matches_generate` true, `prompt.rows_matching` 500,
  `prompt.example_mismatch` 0;
- **Constrained:** `constrained.enabled` true, `constrained.regex_sha256` and `constrained.in_language.regex_sha256`
  `5c081828…` (D-022), `constrained.in_language.matches_generate` true, `constrained.in_language.stop.out_of_language`
  0, `constrained.in_language.other.n` 0;
- **What the regex guarantees:** `failures.G1`, `G3`, `G4`, `G5` and `crash` 0;
- **One stamp:** above.
Not part of validity (D-020): `constrained.backend`, the render step and G7. They are shown, never judged.
A value must have the registered JSON type too: `partial` 0 is not false, and `n` 500.0 or true is not 500.

**The bar's first half:** `pass_not_copy.count` >= 300 of 500, with `rate` and `ci95` as junk.json gives them.

**Two readouts** (D-021, beside the bar, never gated), from scored.jsonl and clips.json:
- the passing non-copies (`ok` and not `near_copy`) whose loop carries the shown example's A or B line verbatim;
- the same among the clips that rendered (clips.json records with `ok` true): the ones his ear check plays; when
  some did not render, the render step's whole sample is shown beside it. Without clips.json (the render step wrote
  none, or it was not pulled) this readout is reported as absent: validity reads junk.json alone.

Exit 0 when the run is valid and meets the first half, 3 otherwise (invalid, or under 300), 2 on bad input (a file
missing or unreadable, or files that are not one run's: lofi_rl.gate1.load_run). Sound claims: none.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from lofi_rl.gate1 import (BRIEF_SEED, FROZEN_SHA256, MODEL_REVISION, N, PASS_NOT_COPY_BAR, PROMPT_VERSION,
                           REGEX_SHA256, TEMPLATE_SHA256, V0_RUN, V0_SAMPLING, Gate1Error, Run, load_run,
                           verbatim_line)

SCHEMA = "lofi-rl/gate1-check/1"
GROUPS = ("Complete", "As registered", "Constrained", "What the regex guarantees", "One stamp")
STAMP_NAME = "generate.host.stamp = score_stamp"
SWAPPED = ("palette.sha256", "constrained.regex_sha256", "constrained.in_language.regex_sha256")   # D-022
_STAMP = re.compile(r"^commit=(\S+) dirty=(\S+)")
MISSING = object()


@dataclass(frozen=True)
class Condition:
    group: str            # one of GROUPS
    name: str             # the junk.json field it reads
    needs: str            # what the list asks, as text
    shows: str            # what junk.json shows, as text
    ok: bool
    source: str           # "D-020", or "D-022" for a swapped value


def field(doc, path: str):
    """The value at a dotted path of a JSON document, or MISSING."""
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur


def shown(v) -> str:
    """A junk.json value as the reports print it: JSON's spelling (null, true, 1.0), a mapping as k=v pairs."""
    if v is MISSING:
        return "missing"
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return " ".join(f"{k}={shown(x)}" for k, x in v.items())
    return json.dumps(v)


def _int(v, want: int) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v == want


def _str(v, want: str) -> bool:
    return isinstance(v, str) and v == want


def _same_json(v, want) -> bool:
    """Equal as JSON text, so 0 and false, or 1 and 1.0, differ (dict order aside)."""
    return v is not MISSING and json.dumps(v, sort_keys=True) == json.dumps(dict(want), sort_keys=True)


def stamp_fields(stamp) -> tuple[str, str] | None:
    """(commit, dirty) of a `.sync-stamp` line (`commit=… dirty=… diff=… at=…`), or None."""
    m = _STAMP.match(stamp) if isinstance(stamp, str) else None
    return (m.group(1), m.group(2)) if m else None


def conditions(j: dict) -> list[Condition]:
    """Every condition of the validity list (module doc), in D-020's order, on one junk.json."""
    out: list[Condition] = []

    def add(group: str, name: str, needs: str, ok_fn, source: str = "D-020") -> None:
        v = field(j, name)
        out.append(Condition(group, name, needs, shown(v), v is not MISSING and bool(ok_fn(v)), source))

    add("Complete", "n", str(N), lambda v: _int(v, N))
    add("Complete", "requested", str(N), lambda v: _int(v, N))
    add("Complete", "partial", "false", lambda v: v is False)
    add("Complete", "unreadable_lines", "0", lambda v: _int(v, 0))
    add("Complete", "generate.stopped", "null", lambda v: v is None)

    add("As registered", "generate.model.revision", MODEL_REVISION, lambda v: _str(v, MODEL_REVISION))
    add("As registered", "briefs.seed", str(BRIEF_SEED), lambda v: _int(v, BRIEF_SEED))
    add("As registered", "sampling", f"{V0_RUN}'s: {shown(dict(V0_SAMPLING))}", lambda v: _same_json(v, V0_SAMPLING))
    add("As registered", "palette.sha256", FROZEN_SHA256, lambda v: _str(v, FROZEN_SHA256), "D-022")
    add("As registered", "palette.matches_generate", "true", lambda v: v is True)
    add("As registered", "prompt.version", PROMPT_VERSION, lambda v: _str(v, PROMPT_VERSION))
    add("As registered", "prompt.template_sha256", TEMPLATE_SHA256, lambda v: _str(v, TEMPLATE_SHA256))
    add("As registered", "prompt.template_matches_generate", "true", lambda v: v is True)
    add("As registered", "prompt.rows_matching", str(N), lambda v: _int(v, N))
    add("As registered", "prompt.example_mismatch", "0", lambda v: _int(v, 0))

    add("Constrained", "constrained.enabled", "true", lambda v: v is True)
    add("Constrained", "constrained.regex_sha256", REGEX_SHA256, lambda v: _str(v, REGEX_SHA256), "D-022")
    add("Constrained", "constrained.in_language.regex_sha256", REGEX_SHA256, lambda v: _str(v, REGEX_SHA256),
        "D-022")
    add("Constrained", "constrained.in_language.matches_generate", "true", lambda v: v is True)
    add("Constrained", "constrained.in_language.stop.out_of_language", "0", lambda v: _int(v, 0))
    add("Constrained", "constrained.in_language.other.n", "0", lambda v: _int(v, 0))

    for g in ("G1", "G3", "G4", "G5", "crash"):
        add("What the regex guarantees", f"failures.{g}", "0", lambda v: _int(v, 0))

    gen, sco = field(j, "generate.host.stamp"), field(j, "score_stamp")
    a, b = stamp_fields(gen), stamp_fields(sco)
    both = gen if gen == sco and gen is not MISSING else None
    out.append(Condition("One stamp", STAMP_NAME, "the same commit=… dirty=0 in both",
                         f"{shown(both)} (both)" if both is not None else f"generate {shown(gen)}; score {shown(sco)}",
                         a is not None and a == b and a[1] == "0", "D-020; D-022"))
    return out


def readouts(run: Run) -> dict:
    """The two verbatim readouts (module doc), with the A-and-B counts beside them. The clips' readout is over the
    clips that rendered, the ones his ear check plays, with the render step's whole sample beside it; None when
    clips.json is absent."""
    def block(rows: list[dict]) -> dict:
        k = sum(1 for s in rows if verbatim_line(s))
        both = sum(1 for s in rows if (s.get("verbatim") or {}).get("A") is True
                   and (s.get("verbatim") or {}).get("B") is True)
        return {"k": k, "n": len(rows), "share": round(k / len(rows), 4) if rows else None, "both": both}
    clips = None
    if run.clips is not None:
        clips = {**block([run.scored[r["i"]] for r in run.rendered]), "sampled": len(run.sample),
                 "recorded": len(run.clip_records), "rendered": len(run.rendered),
                 "sample": block([run.scored[i] for i in run.sample])}
    return {"passing_non_copies": block([run.scored[i] for i in run.pool]), "clips": clips}


def run_record(run: Run) -> dict:
    """What the run is: its job, stamp, palette, prompt and grammar, as junk.json records them."""
    j, host = run.junk, run.host
    c = j.get("constrained") if isinstance(j.get("constrained"), dict) else {}
    p = j.get("prompt") if isinstance(j.get("prompt"), dict) else {}
    return {"run": run.name, "job": run.job, "node": host.get("node"), "gpu": host.get("gpu"),
            "stamp": {"generate": host.get("stamp"), "score": j.get("score_stamp")},
            "palette": {"id": run.palette.get("id"), "sha256": run.palette.get("sha256"), "label": run.palette_label},
            "prompt": {"version": p.get("version"), "template_sha256": p.get("template_sha256")},
            "constrained": {"enabled": c.get("enabled"), "regex_sha256": c.get("regex_sha256"),
                            "regex_chars": c.get("regex_chars"), "backend": c.get("backend")},
            "briefs": j.get("briefs"), "n": j.get("n"), "requested": j.get("requested")}


def evaluate(run: Run) -> dict:
    """The check's whole report (module doc)."""
    conds = conditions(run.junk)
    failed = [c.name for c in conds if not c.ok]
    pnc = run.junk["pass_not_copy"]
    count, n = pnc.get("count"), run.junk.get("n")
    met = isinstance(count, int) and count >= PASS_NOT_COPY_BAR
    valid = not failed
    if valid:
        verdict = "valid, first half met" if met else "valid, under the bar"
    else:
        verdict = "invalid" if met else "invalid, and under the bar"
    return {"schema": SCHEMA, **run_record(run),
            "validity": {"valid": valid, "held": len(conds) - len(failed), "of": len(conds), "failed": failed,
                         "conditions": [asdict(c) for c in conds]},
            "bar": {"pass_not_copy": {"count": count, "of": n, "need": PASS_NOT_COPY_BAR, "rate": pnc.get("rate"),
                                      "ci95": pnc.get("ci95"), "method": pnc.get("method"), "met": met}},
            "readouts": readouts(run), "render": run.junk.get("render"),
            "verdict": verdict, "exit": 0 if valid and met else 3}


def render_summary(render) -> str:
    """junk.json's `render` block in words: shown, never judged (D-020: the render step and G7 decide nothing)."""
    if not isinstance(render, dict):
        return "junk.json has no render block (render did not run, or ran no clip)"
    keys = ("attempted", "rendered", "G7", "other_gates", "errors", "skipped")
    return (", ".join(f"{k} {shown(render.get(k, MISSING))}" for k in keys)
            + f" (k {shown(render.get('k', MISSING))}, seed {shown(render.get('seed', MISSING))})")


def render_line(render) -> str:
    return f"render step (recorded, not judged): {render_summary(render)}"


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f} %" if n else "-"


def format_text(doc: dict) -> str:
    v, bar, r = doc["validity"], doc["bar"]["pass_not_copy"], doc["readouts"]
    c = doc["constrained"]
    backend = c.get("backend") if isinstance(c.get("backend"), dict) else {}
    lines = [f"gate1 check: {doc['run']} (job {doc['job']}, {doc['node']})",
             f"  stamp: generate {doc['stamp']['generate']}; score {doc['stamp']['score']}",
             f"  palette {doc['palette']['label']}; prompt {doc['prompt']['version']} "
             f"{doc['prompt']['template_sha256']}; constrained {shown(c.get('enabled'))}, regex "
             f"{c.get('regex_sha256')} ({c.get('regex_chars')} characters), backend {backend.get('backend')} "
             f"(pin {shown(backend.get('pin_accepted'))}, applied {shown(backend.get('applied'))}; recorded, not "
             f"judged)",
             "validity (D-020's list as amended; D-022's two swaps):"]
    group = None
    width = max(len(x["name"]) for x in v["conditions"])
    for x in v["conditions"]:
        if x["group"] != group:
            group = x["group"]
            lines.append(f"  {group}:")
        tag = "" if x["source"] == "D-020" else f"  [{x['source']}]"
        lines.append(f"    {'ok  ' if x['ok'] else 'FAIL'}  {x['name']:<{width}}  {x['shows']}"
                     + ("" if x["ok"] else f"   (needs {x['needs']})") + tag)
    lines.append(f"validity: {'VALID' if v['valid'] else 'INVALID'}, {v['held']} of {v['of']} conditions hold"
                 + (f"; failing: {', '.join(v['failed'])}" if v["failed"] else ""))
    lo, hi = (bar["ci95"] or [None, None])[:2]
    lines.append(f"bar, first half: pass_not_copy.count {bar['count']} of {bar['of']}, needs at least {bar['need']}: "
                 f"{'met' if bar['met'] else 'NOT met'} (rate {bar['rate']}, ci95 [{lo}, {hi}], {bar['method']}, as "
                 f"junk.json gives it)")
    p, k = r["passing_non_copies"], r["clips"]
    lines.append(f"readout: a verbatim A or B line in {p['k']} of the {p['n']} passing non-copies "
                 f"({pct(p['k'], p['n'])}; both lines in {p['both']})")
    if k is None:
        lines.append("readout: the clips' share is not there: summary/clips.json is absent (the render step wrote "
                     "none, or it was not pulled)")
    else:
        lines.append(f"readout: a verbatim A or B line in {k['k']} of the {k['n']} clips that rendered "
                     f"({pct(k['k'], k['n'])}; both lines in {k['both']})"
                     + ("" if k["n"] == k["sampled"] else
                        f"; in {k['sample']['k']} of the whole sample of {k['sampled']}, {k['recorded']} recorded"))
    lines.append(render_line(doc.get("render")))
    lines.append(f"GATE1_CHECK {doc['verdict']}")
    return "\n".join(lines) + "\n"


def main(a: argparse.Namespace) -> int:
    """The `check` subcommand (lofi_rl.gate1.cli)."""
    try:
        doc = evaluate(load_run(a.run, need_clips=False))
    except Gate1Error as err:
        print(f"error: bad input, nothing decided:\n{err}", file=sys.stderr)
        return 2
    print(json.dumps(doc, indent=1) if a.json else format_text(doc).rstrip("\n"))
    return doc["exit"]


def add_parser(sub) -> None:
    p = sub.add_parser("check", help="validity (D-020 as amended, D-022's swaps), the bar's first half, two readouts")
    p.add_argument("run", type=Path, help="the run folder, runs/junk-<job> (its summary/ is read)")
    p.add_argument("--json", action="store_true", help="print the report as JSON instead of text")
    p.set_defaults(run_command=main)
