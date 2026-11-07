"""score: junk's gates and readouts on every step-0 draw, the readouts per stream and the candidates
(docs/design/step0_samples.md §2.4), in the policy env.

    python -m lofi_rl.step0 score --run DIR
    python -m lofi_rl.step0 score --from OUT --run REDO      a recovery (§4.2): OUT's records read-only, REDO written

- **Every row** of completions.jsonl goes through junk.score_row with junk.CopyCheck("v2", the frozen palette), as
  junk's score step scores a row (G1-G6 and compile, repairs and P_fix, the copy and knob readouts), and gains stream,
  brief_id, k and text_sha256 (the sha256 of the text as generated): summary/scored.jsonl, in i order.
- **Per stream** (rm_heldout, then train), junk.summarize gives the readout blocks: pass, failures, repairs, tokens,
  near_copy, copy, by_cell, prompt, constrained with in_language, palette and the rest. Each stream is scored with its
  own CopyCheck, so one stream's knob-readout errors never null the other's, and is measured against the requests
  generate.json records for it (`streams`: briefs x draws). They go into step0.json's by_stream, under step0.json's
  own schema: this step never writes a junk.json, which gate1's load_run reads by name and schema.
- **candidates** (select.candidates_block): each brief's draws that pass their records (Q3), in k order; Ref_b's 4 by
  their records (§1.5), the selection this unit fixes for Ref_b; and the held-out render order, the render step's
  round 1.
- step0.json's render, render_code, sets and render_stamp are null here, for the render step to fill; score_stamp is
  this step's .sync-stamp. `partial` is true when fewer rows than generate.json's `requested` were read, or when
  generate.json says it did not finish or stopped.

A partial run is scored as far as it goes. Rows that are not one run's draws (a cut file, a foreign or repeated row)
leave candidates null, with exit 1. Exit 2, with nothing written: no completions.jsonl, no step-0 generate.json, a
refused streams file or palette, or --run and --from that overlap. Sound claims: none.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import lofi_rl.step0 as S
from lofi_rl.brief_streams import ROOT
from lofi_rl.palette import PaletteError, load_palette
from lofi_rl.prompt import PromptError
from lofi_rl.rl import junk as J
from lofi_rl.step0 import Step0Error
from lofi_rl.step0 import requests as R
from lofi_rl.step0 import select as X

EXTRA = ("stream", "brief_id", "k", "text_sha256")       # what a scored row adds to junk.score_row's fields


# --------------------------------------------------------------------------------------------- files
def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)                       # a kill mid-write never leaves a cut file


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def sync_stamp(root: Path | None = None) -> str | None:
    """The code tree's .sync-stamp as this process sees it (each step of the job is its own process)."""
    stamp = Path(ROOT if root is None else root) / ".sync-stamp"
    return stamp.read_text().strip() if stamp.is_file() else None


def separate(out: Path, run: Path) -> list[str]:
    """--from OUT and --run REDO must be two folders, neither inside the other, so a recovery never writes under OUT
    (§4.2)."""
    a, b = Path(out).resolve(), Path(run).resolve()
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        return [f"--run {run} and --from {out} overlap: a recovery writes a folder of its own, never under OUT"]
    return []


def _whole(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# --------------------------------------------------------------------------------------------- score
def score_row(row: dict, palette, check: J.CopyCheck) -> dict:
    """junk.score_row of one completion, plus stream, brief_id, k and text_sha256."""
    out = J.score_row(row, palette, check)
    out.update(stream=row.get("stream"), brief_id=row.get("brief_id"), k=row.get("k"),
               text_sha256=J.sha256_text(row["text"]))
    return out


def stream_requested(meta: Mapping, name, n: int) -> int:
    """The requests generate.json records for a stream (`streams`: briefs x draws), or the rows read when it records
    none."""
    rec = (meta.get("streams") or {}).get(name) if isinstance(meta.get("streams"), Mapping) else None
    if isinstance(rec, Mapping) and _whole(rec.get("briefs")) and _whole(rec.get("draws")):
        return rec["briefs"] * rec["draws"]
    return n


def score_rows(rows: list[dict], meta: dict, palette, run_name: str) -> tuple[list[dict], dict, dict]:
    """(scored rows in i order, by_stream, each stream's CopyCheck): every row scored, then junk.summarize per stream,
    in generate.json's stream order, then any other stream the rows hold."""
    names = list(meta.get("streams") or {}) if isinstance(meta.get("streams"), Mapping) else []
    names += [n for n in dict.fromkeys(r.get("stream") for r in rows) if n not in names]
    blocks, checks, scored = {}, {}, []
    for name in names:
        rs = [r for r in rows if r.get("stream") == name]
        checks[name] = check = J.CopyCheck(S.PROMPT_VERSION, palette)
        ss = [score_row(r, palette, check) for r in rs]
        blocks[name] = J.summarize(rs, ss, {**meta, "requested": stream_requested(meta, name, len(rs))}, palette,
                                   check, run=run_name)
        scored += ss
    scored.sort(key=lambda s: s["i"])
    return scored, blocks, checks


def _headline(name, b: dict) -> str:
    fails = " ".join(f"{g} {c}" for g, c in b["failures"].items() if c) or "no failures"
    rate = f"{100 * b['pass']['rate']:.1f}%" if b["pass"]["rate"] is not None else "-"
    return (f"SCORE {name} n={b['n']}/{b['requested']} pass {b['pass']['count']} = {rate} | near-copies "
            f"{b['near_copy']['count']} | {fails} | truncated {b['tokens']['truncated']}")


def main(a: argparse.Namespace) -> int:
    """The `score` subcommand (lofi_rl.step0.cli)."""
    run = Path(a.run)
    src = run if a.from_ is None else Path(a.from_)
    problems = [] if a.from_ is None else separate(src, run)
    comp, gen = src / "summary" / "completions.jsonl", src / "summary" / "generate.json"
    meta = read_json(gen)
    if not comp.is_file():
        problems.append(f"{comp} is missing (run generate first)")
    if meta is None or meta.get("schema") != S.GENERATE_SCHEMA:
        problems.append(f"{gen} is missing or not a step-0 generate.json ({S.GENERATE_SCHEMA})")
    try:
        streams = R.load()
    except Step0Error as err:
        problems += err.problems
    if problems:
        print("score: nothing scored or written:\n" + "\n".join(f"- {p}" for p in problems), file=sys.stderr)
        return 2
    rows, bad = J.read_jsonl(comp)
    rows.sort(key=lambda r: r["i"])
    try:
        palette = load_palette(ROOT / S.PALETTE, verify=False)       # names only: no audio in the policy env
        scored, blocks, checks = score_rows(rows, meta, palette, run.resolve().name)
    except (OSError, PaletteError, PromptError) as err:
        print(f"score: nothing written: {type(err).__name__}: {err}", file=sys.stderr)
        return 2
    rc = 0
    try:
        candidates = X.candidates_block(scored, streams=streams)
    except Step0Error as err:
        candidates, rc = None, 1
        print("score: the rows are not one run's draws, so candidates is null:\n" + str(err), file=sys.stderr)
    requested = meta["requested"] if _whole(meta.get("requested")) else len(scored)
    doc = {"schema": S.STEP0_SCHEMA, "run": run.resolve().name, "n": len(scored), "requested": requested,
           "partial": len(scored) < requested or meta.get("partial") is not False or meta.get("stopped") is not None,
           "unreadable_lines": bad, "by_stream": blocks, "candidates": candidates, "render": None, "render_code": None,
           "sets": None, "score_stamp": sync_stamp(), "render_stamp": None}
    summ = run / "summary"
    summ.mkdir(parents=True, exist_ok=True)
    write_jsonl(summ / "scored.jsonl", scored)
    write_json(summ / "step0.json", doc)
    for name, check in checks.items():
        for err, count in check.knob_errors.items():
            print(f"score: {name}: the knob readout failed on {count} loop(s), recorded as null: {err}",
                  file=sys.stderr)
    for name, b in blocks.items():
        print(_headline(name, b), flush=True)
    print(f"STEP0_SCORE n={doc['n']}/{requested} partial={doc['partial']} unreadable={bad} "
          f"candidates={'ok' if candidates is not None else 'null'}", flush=True)
    return rc
