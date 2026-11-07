"""`python -m lofi_rl.step0 prereg RUN_DIR [--redo REDO_DIR] [--write]`: the two committed records of a valid step-0
run, docs/prereg/heldout_v1.json and docs/prereg/step0_v1.json (docs/design/step0_samples.md §4.4; D-023).

- **heldout_v1.json** (`lofi-rl/heldout/1`), the file taste_pipeline.md:128 names: the run (job, node, GPU, stamps),
  its render-code key, the settings with their shas and the rule; per tuple, its brief (id, cell, bpm, tonic, twins)
  and the shown example's id, and per clip its k, draw seed, text_sha256 and canon_sha256, clip id, audio_sha256 (the
  float32 render identity), LUFS, true peak, P_fix, near_copy, copy_jaccard and its verbatim A and B flags; the short
  briefs; the texts' pin (completions.jsonl's path and sha256, each clip's text found there by its text_sha256); and
  item 8's readouts beside D-008, reported and never gated, as numbers and flags (lofi_rl.step0.check.item8: each
  tuple's pairwise J5 in junk's copy form and the K/D form, its verbatim-line sharing, and per pair whether the two
  clips differ in A or B, the pairs "agreement on the pairs whose two clips differ in A or B" reads, DECISIONS.md:484).
- **step0_v1.json** (`lofi-rl/step0-run/1`): the run and a redo if any, with every summary file's path, size and
  sha256; the counts per stream, the validity verdict, the probe and the replay; Ref_b per brief (k, draw seed,
  text_sha256 and canon_sha256 of its 4, its spares' k in order); the pool's briefs and their picks; every train
  draw's heldout_text flag, with the train rows that share each held-out clip's canon_sha256, counted per cell; the
  short briefs; and heldout_v1.json's sha256.

Neither holds a LoopScript text (§5): they hold hashes, and the texts stay in runs/ (KEEP and the laptop's runs/, never
committed), pinned by completions.jsonl's sha256. Paths are written as make pull places a run (runs/<run>/summary/...),
so the text depends on the run's files alone. Both are ASCII JSON, one record a line (each tuple, brief or file on one
line; the rest in lofi_rl.brief_streams' layout), and byte-identical when written again from the same run.

The run must pass `check` (lofi_rl.step0.check.evaluate, with git at its stamps' commits): an invalid run is refused
(exit 3), and bad input is exit 2, nothing written either way. Under Q8's (b) (SHORT_RULE "ask") a short held-out or
Ref_b brief is refused too (exit 2), until D-023 records Shresth's answer and the constants carry it. Without --write
the two files' sizes and sha256 are printed and nothing is written; with it each is written unless it is already there
byte for byte. A different existing file is never replaced (exit 2, neither written): there is no --force, and a new
set is a new file version with its own DECISIONS entry (§4.2). Laptop only. Sound claims: none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import lofi_rl.step0 as S
from lofi_rl import brief_streams
from lofi_rl.step0 import Step0Error, short
from lofi_rl.step0 import check as C

WIDTH = brief_streams.WIDTH             # a flat value that fits in WIDTH columns is written on one line
CLIP_FIELDS = ("k", "i", "draw_seed", "text_sha256", "canon_sha256", "clip_id", "audio_sha256", "lufs",
               "true_peak_dbtp", "p_fix", "near_copy", "copy_jaccard", "A", "B")
DRAW_FIELDS = ("k", "i", "draw_seed", "text_sha256", "canon_sha256")     # a Ref_b or pool draw (not rendered here)


# --------------------------------------------------------------------------------------------- the layout
def _scalar(v) -> bool:
    return v is None or isinstance(v, (str, int, float))         # bool is an int


def _flat(v) -> bool:
    """A list or dict whose items are scalars or lists of scalars: what may go on one line."""
    items = list(v.values()) if isinstance(v, dict) else v
    return all(_scalar(x) or (isinstance(x, list) and all(_scalar(y) for y in x)) for x in items)


def _one(v) -> str:
    return json.dumps(v, separators=(", ", ": "), ensure_ascii=True, allow_nan=False)


def _json(v, indent: int = 0, column: int = 0) -> str:
    """JSON text of `v`: a list of records (objects) one record a line, however long; a flat value on one line when
    it fits in WIDTH columns from `column`; otherwise one item a line (lofi_rl.brief_streams' layout)."""
    pad = " " * (indent + 2)
    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
        return "[\n" + ",\n".join(pad + _one(x) for x in v) + "\n" + " " * indent + "]"
    one = _one(v)
    if _scalar(v) or not v or (_flat(v) and column + len(one) <= WIDTH):
        return one
    if isinstance(v, list):
        return "[\n" + ",\n".join(pad + _json(x, indent + 2, indent + 2) for x in v) + "\n" + " " * indent + "]"
    body = []
    for key, x in v.items():
        head = f"{pad}{_one(key)}: "
        body.append(head + _json(x, indent + 2, len(head)))
    return "{\n" + ",\n".join(body) + "\n" + " " * indent + "}"


def text(doc: Mapping) -> str:
    """A file's exact text: ASCII JSON, one record a line, ending in a newline."""
    return _json(dict(doc)) + "\n"


# --------------------------------------------------------------------------------------------- what both hold
def _rel(run_dir: Path, name: str) -> str:
    """A summary file's path as make pull places it: runs/<run folder>/summary/<name>."""
    return f"runs/{Path(run_dir).resolve().name}/summary/{name}"


def answers() -> dict:
    """The answers to the spec's §7 that the run was selected under (lofi_rl.step0's Q1-Q8 constants)."""
    return {"Q1": {"pool_briefs": S.POOL_BRIEFS, "bands": [list(b) for b in S.POOL_BANDS],
                   "fill_band": list(S.POOL_FILL_BAND)},
            "Q2": {"pool_shares_refb": S.POOL_SHARES_REFB}, "Q3": {"passing": S.PASSING},
            "Q4": {"distinct": dict(S.DISTINCT)}, "Q5": {"pool_excludes": S.POOL_EXCLUDES},
            "Q6": {"streams": [list(p) for p in S.STREAMS]}, "Q7": {"committed_before": S.COMMITTED_BEFORE},
            "Q8": {"short_rule": S.SHORT_RULE}}


def run_block(run: C.Run, report: Mapping) -> dict:
    """The run: its folders, job, node, GPU and stamps, and the check's verdict."""
    return {"run": run.name, "redo": None if run.redo is None else run.redo.resolve().name, "job": run.job,
            "node": run.host.get("node"), "gpu": run.host.get("gpu"),
            "stamp": {"generate": run.host.get("stamp"), "score": run.step0.get("score_stamp"),
                      "render": run.step0.get("render_stamp")},
            "validity": {"verdict": report["verdict"], "held": report["validity"]["held"],
                         "of": report["validity"]["of"]}}


def texts_block(run: C.Run) -> dict:
    """The texts' pin: completions.jsonl, where each draw's text is found by its text_sha256."""
    path = run.files["completions.jsonl"]
    return {"file": _rel(run.dir, "completions.jsonl"), "sha256": C.sha256_bytes(path.read_bytes()),
            "by": "text_sha256: sha256 of a row's `text`, UTF-8"}


def settings(run: C.Run) -> dict:
    """What was drawn, rendered and selected under, as the run's files record it (check holds each to its value)."""
    g = run.generate
    prompt = g.get("prompt") or {}
    pal = g.get("palette") or {}
    con = g.get("constrained") or {}
    return {"model": g.get("model"),
            "prompt": {"version": prompt.get("version"), "template_sha256": prompt.get("template_sha256")},
            "palette": {"id": pal.get("id"), "sha256": pal.get("sha256"),
                        "label": f"{pal.get('id')}@{str(pal.get('sha256'))[:8]}"},
            "regex_sha256": con.get("regex_sha256"), "backend": (con.get("backend") or {}).get("backend"),
            "sampling": g.get("sampling"), "per_request_seed": g.get("per_request_seed"),
            "engine_seed": (g.get("engine") or {}).get("seed"), "streams_file": g.get("streams_file"),
            "streams": g.get("streams")}


def rule_block() -> dict:
    return {"name": S.RULE, "mark": S.MARK, "m": dict(S.M), "answers": answers()}


def _entry_head(e: Mapping) -> dict:
    return {k: e.get(k) for k in ("brief_id", "cell", "bpm", "tonic", "twin_of")}


# --------------------------------------------------------------------------------------------- heldout_v1.json
def _clip(t: Mapping, rec: Mapping | None, s: Mapping) -> dict:
    """One held-out clip's record: its draw, its render (clips.json) and its score (scored.jsonl)."""
    rec = rec or {}
    verb = s.get("verbatim") or {}
    got = {**{k: t.get(k) for k in ("k", "i", "draw_seed", "text_sha256", "canon_sha256", "clip_id",
                                     "audio_sha256")},
           "lufs": rec.get("lufs"), "true_peak_dbtp": rec.get("true_peak_dbtp"), "p_fix": rec.get("p_fix"),
           "near_copy": s.get("near_copy"), "copy_jaccard": s.get("copy_jaccard"), "A": verb.get("A"),
           "B": verb.get("B")}
    return {k: got[k] for k in CLIP_FIELDS}


def heldout_doc(run: C.Run, report: Mapping, *, palette=None) -> dict:
    """heldout_v1.json's document (module doc)."""
    rows, scored = run.text_rows(), run.scored_rows()
    final = {r["i"]: r for r in run.records if r.get("final") is True}
    m = S.M["heldout"]
    tuples, shorts = [], []
    for e in run.entries("heldout"):
        clips = [_clip(t, final.get(t["i"]), scored[t["i"]]) for t in e["taken"]]
        example = rows[e["taken"][0]["i"]].get("example") if e["taken"] else None
        rec = {**_entry_head(e), "example": example, "clips": clips}
        if len(clips) == m and not e.get("short"):
            tuples.append(rec)
        else:
            shorts.append({**rec, "not_taken": e.get("not_taken")})
    i8 = C.item8(run, palette)
    twins = sum(1 for e in run.entries("heldout") if e.get("twin_of"))
    return {
        "schema": S.HELDOUT_SCHEMA,
        "registered": [
            f"docs/DECISIONS.md D-023 (rule {S.RULE}): the RM held-out tuples of D-021 item 2 (taste_pipeline.md "
            "lines 126-129), fixed by the first valid run of lofi-step0, or valid pair (spec section 4.2).",
            f"Committed {S.COMMITTED_BEFORE} (Q7). Never trained on, selected on, or tuned on (taste_pipeline.md:129).",
            f"Each brief takes its first {m} draws in k order that pass the hard gates (Q3) and whose canonical texts "
            "differ (Q4); a short brief is reported, never drawn again (Q8)."],
        "written_by": [
            "python -m lofi_rl.step0 prereg RUN [--redo REDO] --write; do not edit this file.",
            "No LoopScript text: each clip's text is in `texts`.file, found by its text_sha256 (spec section 5)."],
        "run": run_block(run, report),
        "render_code": run.render_code,
        "settings": settings(run),
        "rule": rule_block(),
        "texts": texts_block(run),
        "counts": {"briefs": len(run.entries("heldout")), "tuples": len(tuples),
                   "clips": sum(len(t["clips"]) for t in tuples), "short": len(shorts), "twins": twins},
        "tuples": tuples,
        "short": shorts,
        "item8": {
            "note": [
                "D-021 item 8: reported beside D-008, never gated. j5 is the character 5-gram Jaccard (junk.grams,"
                " junk.jaccard) of a pair's copy forms: junk's (the canonical text without sax:) and the K/D form"
                " (DRAFTS.md section 5: keys and kit written K and D).",
                "Pairs are in k order. A and B: whether the two clips' A lines, or B lines, differ; differ: either."
                " verbatim: how many of the tuple's clips carry the shown example's A, B, or both lines verbatim."],
            "summary": i8["summary"], "tuples": i8["tuples"]},
    }


# --------------------------------------------------------------------------------------------- step0_v1.json
def files_block(run: C.Run) -> list[dict]:
    """Every summary file of the run and of its redo: path, size, sha256, and whether check read it."""
    read = {p.resolve() for p in run.files.values()}
    out = []
    for d in (run.dir, run.redo):
        if d is None:
            continue
        for name in S.SUMMARY_FILES:
            p = d / "summary" / name
            if p.is_file():
                data = p.read_bytes()
                out.append({"path": _rel(d, name), "bytes": len(data), "sha256": C.sha256_bytes(data),
                            "read": p.resolve() in read})
    return out


def counts_block(run: C.Run) -> dict:
    out = {}
    for name, _ in S.STREAMS:
        b = (run.step0.get("by_stream") or {}).get(name) or {}
        out[name] = {"n": b.get("n"), "pass": (b.get("pass") or {}).get("count"),
                     "pass_not_copy": (b.get("pass_not_copy") or {}).get("count"),
                     "near_copy": (b.get("near_copy") or {}).get("count"),
                     "truncated": (b.get("tokens") or {}).get("truncated"), "failures": b.get("failures")}
    return out


def _draw(t: Mapping) -> dict:
    return {k: t.get(k) for k in DRAW_FIELDS}


def heldout_text_block(run: C.Run) -> dict:
    """Every train draw's heldout_text flag, by brief in k order; the train rows that share each held-out clip's
    canon_sha256; and the per-cell counts step0.json reads out."""
    flags: dict[str, list] = {}
    for d in run.sets["heldout_text"]:                         # train stream order, then k (select.select)
        flags.setdefault(d["brief_id"], []).append(d.get("heldout_text"))
    train = [s for s in run.scored if s.get("stream") == brief_streams.TRAIN and s.get("ok") is True]
    by_canon: dict[str, list] = {}
    for s in sorted(train, key=lambda s: s["i"]):
        by_canon.setdefault(s.get("canon_sha256"), []).append([s.get("brief_id"), s.get("k")])
    shared = [{"brief_id": e["brief_id"], "k": t.get("k"), "clip_id": t.get("clip_id"),
               "canon_sha256": t.get("canon_sha256"), "train": by_canon[t.get("canon_sha256")]}
              for e in run.entries("heldout") for t in e["taken"] if t.get("canon_sha256") in by_canon]
    return {"per_cell": (run.step0.get("sets") or {}).get("heldout_text"),
            "draws": sum(1 for v in flags.values() for f in v if f is True),
            "flags": [{"brief_id": bid, "flags": v} for bid, v in flags.items()],
            "shared": shared}


def _short(entries: Sequence[Mapping]) -> list[dict]:
    return [{"brief_id": e["brief_id"], "taken": len(e["taken"]), "not_taken": e.get("not_taken")}
            for e in entries if e.get("short")]


def run_doc(run: C.Run, report: Mapping, heldout_sha256: str) -> dict:
    """step0_v1.json's document (module doc)."""
    g = run.generate
    refb = [{"brief_id": e["brief_id"], "cell": e.get("cell"), "taken": [_draw(t) for t in e["taken"]],
             "spares": e.get("spares"), "short": e.get("short")} for e in run.entries("refb")]
    pool = [{**_entry_head(e), "taken": [_draw(t) for t in e["taken"]], "short": e.get("short")}
            for e in run.entries("pool")]
    return {
        "schema": S.RUN_SCHEMA,
        "registered": [
            f"docs/DECISIONS.md D-023 (rule {S.RULE}): the step-0 samples of D-021 item 2, drawn by lofi-step0 under"
            " the grammar with D-021 item 1's streams and draw seeds; this file records the run that fixes them.",
            "Ref_b's and the pool's draws are fixed by their records here; their renders and G7 come later, by D-023's"
            " rules (spec sections 1.5, 1.6 and 3)."],
        "written_by": [
            "python -m lofi_rl.step0 prereg RUN [--redo REDO] --write; do not edit this file.",
            "No LoopScript text: every draw's text is in `texts`.file, found by its text_sha256 (spec section 5)."],
        "run": run_block(run, report),
        "files": files_block(run),
        "rule": rule_block(),
        "validity": {"verdict": report["verdict"], "held": report["validity"]["held"],
                     "of": report["validity"]["of"], "failed": report["validity"]["failed"]},
        "counts": counts_block(run),
        "probe": g.get("probe"),
        "replay": g.get("replay"),
        "render": run.step0.get("render"),
        "render_code": run.render_code,
        "heldout": {"file": S.HELDOUT_FILE, "sha256": heldout_sha256},
        "readouts": {name: (run.step0.get("sets") or {}).get(name) for name in S.SETS},
        "refb": refb,
        "pool": {"rule": S.POOL_BRIEFS, "briefs": len(pool), "picks": sum(len(p["taken"]) for p in pool),
                 "entries": pool},
        "heldout_text": heldout_text_block(run),
        "short": {name: _short(run.entries(name)) for name in S.SETS},
        "texts": texts_block(run),
    }


def documents(run: C.Run, report: Mapping, *, palette=None) -> dict[str, str]:
    """{repo path: exact text} of both files; step0_v1.json pins heldout_v1.json by its sha256."""
    held = text(heldout_doc(run, report, palette=palette))
    sha = hashlib.sha256(held.encode("ascii")).hexdigest()
    return {S.HELDOUT_FILE: held, S.RUN_FILE: text(run_doc(run, report, sha))}


# --------------------------------------------------------------------------------------------- the command
def ask_first(run: C.Run) -> str | None:
    """Under Q8's (b), SHORT_RULE "ask", a short held-out or Ref_b brief waits for Shresth's answer, which D-023
    records and the constants carry before the sets are fixed: the refusal, naming them, or None (finding R1-2). A
    short pool brief is reported under either answer."""
    if S.SHORT_RULE != "ask":
        return None
    short = [f"{name} {e['brief_id']}" for name in ("heldout", "refb") for e in run.entries(name) if e.get("short")]
    if not short:
        return None
    return (f"refused: SHORT_RULE is Q8's (b), ask: {len(short)} held-out or Ref_b briefs are short ("
            f"{', '.join(short[:5])}{', ...' if len(short) > 5 else ''}); Shresth's answer goes into D-023 and the "
            "constants first: nothing written")


def status(path: Path, body: str) -> str:
    """"absent", "identical" (byte for byte) or "differs"."""
    if not path.exists():
        return "absent"
    return "identical" if path.read_bytes() == body.encode("ascii") else "differs"


def prereg(run_dir: Path | str, redo_dir: Path | str | None = None, *, write: bool = False,
           repo: Path | None = None) -> tuple[int, list[str]]:
    """The `prereg` subcommand's work (module doc): (exit code, lines to print). `repo` is the checkout the files go
    into, whose git also reads the stamps' commits (default: this checkout)."""
    repo = Path(C.ROOT if repo is None else repo)
    try:
        run = C.load_run(run_dir, redo_dir)
        report = C.evaluate(run, repo=repo)
    except Step0Error as err:
        return 2, [f"error: bad input, nothing written:\n{err}"]
    if not report["validity"]["valid"]:
        return 3, [f"prereg: {run.name} is not a valid run ({report['validity']['held']} of "
                   f"{report['validity']['of']} conditions hold; failing: {', '.join(report['validity']['failed'])}): "
                   "nothing written. `python -m lofi_rl.step0 check` reports each condition (spec section 4.2 says "
                   "what follows an invalid run)."]
    asked = ask_first(run)
    if asked:
        return 2, [asked]
    try:
        docs = documents(run, report)
    except Step0Error as err:
        return 2, [f"error: nothing written:\n{err}"]
    states = {rel: status(repo / rel, body) for rel, body in docs.items()}
    lines = [f"prereg: {run.name}" + (f" with its redo {run.redo.resolve().name}" if run.redo else "")
             + f" is valid ({report['validity']['held']} of {report['validity']['of']} conditions)"]
    for rel, body in docs.items():
        sha = hashlib.sha256(body.encode("ascii")).hexdigest()
        lines.append(f"  {rel}: {len(body.encode('ascii'))} bytes, sha256 {sha} ({states[rel]})")
    if any(s == "differs" for s in states.values()):
        bad = [rel for rel, s in states.items() if s == "differs"]
        return 2, lines + [f"refused: {', '.join(bad)} already holds another text; a written file is never replaced "
                           "(no --force; a new set is a new file version with its own DECISIONS entry): nothing "
                           "written"]
    if not write:
        return 0, lines + ["nothing written (--write writes the absent files)"]
    for rel, body in docs.items():
        if states[rel] == "absent":
            out = repo / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(out.suffix + ".tmp")
            tmp.write_bytes(body.encode("ascii"))
            tmp.replace(out)
    wrote = [rel for rel, s in states.items() if s == "absent"]
    return 0, lines + [f"wrote {', '.join(wrote)}" if wrote else "unchanged: both files already hold this text",
                       f"next: commit both, and D-023's Outcome with the run, the counts and both sha256 "
                       f"({short(hashlib.sha256(docs[S.HELDOUT_FILE].encode('ascii')).hexdigest())}, "
                       f"{short(hashlib.sha256(docs[S.RUN_FILE].encode('ascii')).hexdigest())})"]


def main(a: argparse.Namespace) -> int:
    """The `prereg` subcommand (lofi_rl.step0.cli)."""
    rc, lines = prereg(a.run, a.redo, write=a.write)
    print("\n".join(lines), file=sys.stderr if rc else sys.stdout)
    return rc

