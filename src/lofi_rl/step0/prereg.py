"""`python -m lofi_rl.step0 prereg RUN_DIR [--redo REDO_DIR] [--write]`: the two committed records of a valid step-0
run, docs/prereg/heldout_v1.json and docs/prereg/step0_v1.json (docs/design/step0_samples.md §4.4; D-023).

- **heldout_v1.json** (`lofi-rl/heldout/1`), the file taste_pipeline.md:128 names: the run (job, node, GPU, stamps),
  its render-code key, the settings with their shas (render_cfg_sha8, the render config the clip ids hash, among them)
  and the rule, in sentences built from the answers; per tuple, its brief (id, cell, bpm, tonic, twins) and the shown
  example's id, and per clip its k, draw seed, text_sha256 and canon_sha256, clip id, audio_sha256 (the float32 render
  identity), LUFS, true peak, P_fix, near_copy, copy_jaccard and its verbatim A and B flags; the short briefs, whose
  clips are held-out data too; the texts' pin (completions.jsonl's path and sha256, each clip's text found there by
  its text_sha256); and item 8's readouts beside D-008, reported and never gated, as numbers and flags
  (lofi_rl.step0.check.item8: each tuple's pairwise J5 in junk's copy form and the K/D form, its verbatim-line
  sharing, and per pair whether the two clips differ in A or B, the pairs "agreement on the pairs whose two clips
  differ in A or B" reads, DECISIONS.md:484).
- **step0_v1.json** (`lofi-rl/step0-run/1`): the run and a redo if any, with every summary file's path, size and
  sha256; the counts per stream, the validity verdict, the probe and the replay (its stop as a code, never a message);
  Ref_b per brief (k, draw seed, text_sha256 and canon_sha256 of its 4, its spares' k in order); the pool's briefs and
  their picks; every train draw's heldout_text flag, with the train rows, passing or not, that share each held-out
  clip's canon_sha256, counted per cell; the short briefs; and heldout_v1.json's sha256.

Neither holds a LoopScript text (§5): they hold hashes, and the texts stay in runs/ (KEEP and the laptop's runs/, never
committed), pinned by completions.jsonl's sha256. Paths are written as make pull places a run (runs/<run>/summary/...),
so the text depends on the run's files alone. Both are ASCII JSON, one record a line (each tuple, brief or file on one
line; the rest in lofi_rl.brief_streams' layout), and byte-identical when written again from the same run.

The run must pass `check` (lofi_rl.step0.check.evaluate, with git at its stamps' commits), at the run's commit
(check.code_problems): an invalid run is refused (exit 3), and bad input is exit 2, nothing written either way. Under
Q8's (b) (SHORT_RULE "ask") a short held-out or Ref_b brief is refused too (exit 2), naming the line to add, until
D-023's entry in the checkout's docs/DECISIONS.md holds Shresth's answer for the run as this line (ask_first; findings
N2, N7):

    D-023 short answer: run step0-<job>; as (a)

It is read only as a line of its own (short_answers, check.d023_lines; findings N7, N11), as kit.gates reads an
approval, but for one list bullet: in D-023's entry, at column 0, starting a block (a bullet of its own, or after a
blank line or a heading), outside code fences and HTML comments, with nothing before its head but that bullet and
what wraps the line (backticks, bold, italics), and nothing after the answer but that wrap's close and a period. So a
sentence, a note or prereg's own refusal that quotes the line, a hard-wrapped sentence's next line, an indented,
nested or fenced line (the form an example takes), a blockquote, a table, a comment or another entry never answers
for him; the refusal names the lines that mention it where no answer is read (finding N12). DECISIONS.md is never
compared with the run's commit, so the line, written after the run, needs no redo and no constant changed; the short
briefs are then handled as Q8(a) handles them, and both files' rule.answers.Q8 records the line. Any other answer for
the run keeps the refusal, which then names those answers and says to replace the run's lines with the one line, since
the line added beside another answer refuses again (finding N8); under SHORT_RULE "report" nothing is read, and Q8
records the rule alone.

**Off the run's commit** (finding N9): once main changes a file check reproduces the run with
(check.reproduction_files: junk's whole closure among them, render's code too), check and prereg refuse there and run
in a git worktree of the run's commit, whose docs/DECISIONS.md predates what D-023 gains after the run. Under Q8(b)
the answer line then goes into D-023 on main and, by hand and uncommitted, into the worktree's docs/DECISIONS.md, the
one prereg reads; prereg writes both files into the worktree, and they are committed on main from there.

Without --write the two files' sizes and sha256 are printed and nothing is written; with it each is written unless it
is already there byte for byte. A different existing file is never replaced (exit 2, neither written): there is no
--force, and a new set is a new file version with its own DECISIONS entry (§4.2). Laptop only. Sound claims: none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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
SHORT_HEAD = "D-023 short answer:"      # Q8(b): the line D-023 gains for a run whose short briefs Shresth answered
SHORT_AS_A = "as (a)"                   # the one answer prereg applies: the short briefs handled as Q8(a) handles them
# his answer is a line of its own (short_answers, findings N7, N11), as check.d023_lines gives it, less one list
# bullet: what wraps the line (backticks, bold, italics), its head and `run <run>; ` come first; the answer is the rest
SHORT = re.compile(r"(?P<wrap>[`*_]*)" + re.escape(SHORT_HEAD) + r" run (?P<run>step0-[\w-]+); (?P<answer>.*)")
WHERE = ("alone on its line in D-023's entry, at column 0 (one list bullet before it at most; not indented, nested, "
         "fenced, in a comment or a sentence's next line)")      # where the line goes, as the refusal says (N12)


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


def answers(short_answer: str | None = None) -> dict:
    """The answers to the spec's §7 that the run was selected under (lofi_rl.step0's Q1-Q8 constants), and under Q8's
    (b) the line of D-023 that answered this run's short briefs, when prereg applied it (ask_first)."""
    return {"Q1": {"pool_briefs": S.POOL_BRIEFS, "bands": [list(b) for b in S.POOL_BANDS],
                   "fill_band": list(S.POOL_FILL_BAND)},
            "Q2": {"pool_shares_refb": S.POOL_SHARES_REFB}, "Q3": {"passing": S.PASSING},
            "Q4": {"distinct": dict(S.DISTINCT)}, "Q5": {"pool_excludes": S.POOL_EXCLUDES},
            "Q6": {"streams": [list(p) for p in S.STREAMS]}, "Q7": {"committed_before": S.COMMITTED_BEFORE},
            "Q8": {"short_rule": S.SHORT_RULE, **({"short_answer": short_answer} if short_answer else {})}}


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
    cfgs = sorted({r.get("render_cfg_sha8") for r in run.records if r.get("ok") is True}, key=str)
    return {"model": g.get("model"),
            "prompt": {"version": prompt.get("version"), "template_sha256": prompt.get("template_sha256")},
            "palette": {"id": pal.get("id"), "sha256": pal.get("sha256"),
                        "label": f"{pal.get('id')}@{str(pal.get('sha256'))[:8]}"},
            "render_cfg_sha8": cfgs[0] if len(cfgs) == 1 else cfgs,     # the clip ids' render config (one, by check)
            "regex_sha256": con.get("regex_sha256"), "backend": (con.get("backend") or {}).get("backend"),
            "sampling": g.get("sampling"), "per_request_seed": g.get("per_request_seed"),
            "engine_seed": (g.get("engine") or {}).get("seed"), "streams_file": g.get("streams_file"),
            "streams": g.get("streams")}


def rule_block(short_answer: str | None = None) -> dict:
    return {"name": S.RULE, "mark": S.MARK, "m": dict(S.M), "answers": answers(short_answer)}


PASSING_WORDS = {"hard": "pass the hard gates", "hard_not_copy": "pass the hard gates and are not near-copies of the "
                 "shown example", "hard_no_repair": "pass the hard gates without a repair"}       # Q3 (a), (b), (c)


def heldout_rule() -> str:
    """heldout_v1.json's sentence for the held-out rule, built from the answers the run was selected under (Q3's
    PASSING, Q4's DISTINCT["heldout"]), so it never states a rule the tuples were not chosen by (finding R1-6)."""
    passing = PASSING_WORDS.get(S.PASSING, f"pass by PASSING {S.PASSING!r}")
    distinct = " and whose canonical texts differ (Q4)" if S.DISTINCT.get("heldout") is True else ", repeats kept (Q4)"
    return (f"Each brief takes its first {S.M['heldout']} draws in k order that {passing} (Q3){distinct}; a short "
            "brief is reported, never drawn again (Q8).")


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


def heldout_doc(run: C.Run, report: Mapping, *, palette=None, short_answer: str | None = None) -> dict:
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
            heldout_rule(),
            "A short brief's clips (`short`) are held-out data too, in no tuple: every rule for held-out clips holds "
            "for them, the RM loader's Q5 rule included (spec section 5)."],
        "written_by": [
            "python -m lofi_rl.step0 prereg RUN [--redo REDO] --write; do not edit this file.",
            "No LoopScript text: each clip's text is in `texts`.file, found by its text_sha256 (spec section 5)."],
        "run": run_block(run, report),
        "render_code": run.render_code,
        "settings": settings(run),
        "rule": rule_block(short_answer),
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
    canon_sha256, passing or not (a draw that parsed carries its canon_sha256 even when it then failed compile), as
    the flags count them (finding R1-4); and the per-cell counts step0.json reads out."""
    flags: dict[str, list] = {}
    for d in run.sets["heldout_text"]:                         # train stream order, then k (select.select)
        flags.setdefault(d["brief_id"], []).append(d.get("heldout_text"))
    train = [s for s in run.scored if s.get("stream") == brief_streams.TRAIN and isinstance(s.get("canon_sha256"), str)]
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


def replay_block(replay) -> dict | None:
    """generate.json's replay readout with its `stopped` cut to its code (the text before its first colon: "deadline",
    "running", "not run", "error"), so an exception message, the one free text a valid run's records carry, never
    reaches a committed file (finding R3-10)."""
    if not isinstance(replay, Mapping):
        return None
    stopped = replay.get("stopped")
    return {**replay, "stopped": stopped.split(":", 1)[0] if isinstance(stopped, str) else stopped}


def _short(entries: Sequence[Mapping]) -> list[dict]:
    return [{"brief_id": e["brief_id"], "taken": len(e["taken"]), "not_taken": e.get("not_taken")}
            for e in entries if e.get("short")]


def run_doc(run: C.Run, report: Mapping, heldout_sha256: str, *, short_answer: str | None = None) -> dict:
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
        "rule": rule_block(short_answer),
        "validity": {"verdict": report["verdict"], "held": report["validity"]["held"],
                     "of": report["validity"]["of"], "failed": report["validity"]["failed"]},
        "counts": counts_block(run),
        "probe": g.get("probe"),
        "replay": replay_block(g.get("replay")),
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


def documents(run: C.Run, report: Mapping, *, palette=None, short_answer: str | None = None) -> dict[str, str]:
    """{repo path: exact text} of both files; step0_v1.json pins heldout_v1.json by its sha256. `short_answer` is the
    D-023 line ask_first applied, recorded in both files' rule.answers.Q8 (None: nothing is added)."""
    held = text(heldout_doc(run, report, palette=palette, short_answer=short_answer))
    sha = hashlib.sha256(held.encode("ascii")).hexdigest()
    return {S.HELDOUT_FILE: held, S.RUN_FILE: text(run_doc(run, report, sha, short_answer=short_answer))}


# --------------------------------------------------------------------------------------------- Q8(b): his answer
def short_line(run_name: str) -> str:
    """The line D-023 gains when Shresth answers Q8(b) for a run's short held-out or Ref_b briefs with (a)'s handling:
    the one answer prereg applies (ask_first)."""
    return f"{SHORT_HEAD} run {run_name}; {SHORT_AS_A}"


def short_answers(decisions: str, run_name: str) -> list[tuple[int, str]]:
    """(line number, answer) of each line of D-023's entry in a DECISIONS.md text that answers the run `run_name`, in
    order (findings N2, N7, N11). Each is a line of its own (check.d023_lines, not shown: at column 0, starting a
    block, outside fences and comments) that, after one list bullet at most and what wraps it (backticks, bold or
    italics), starts with the head and `run <run>; `; its answer is the rest less that wrap's close and a closing
    period. So the line wrapped or with a closing period reads as the bare form, and a note after the answer is part
    of it; a line that quotes it after other text (a sentence, prereg's own refusal, a blockquote, a table), a wrapped
    sentence's next line, an indented, nested or fenced line, one in a comment or in another entry, and a bold head
    (**D-023 short answer:**) are no answer, as kit.gates reads an approval only on a line of its own."""
    out = []
    for number, line in C.d023_lines(decisions):
        m = SHORT.fullmatch(line.rstrip())
        if m is not None and m.group("run") == run_name:
            out.append((number, m.group("answer").removesuffix(".").removesuffix(m.group("wrap")[::-1])
                        .removesuffix(".")))
    return out


def ask_first(run: C.Run, repo: Path) -> tuple[str | None, str | None]:
    """Q8's (b), SHORT_RULE "ask" (findings R1-2, N2, N7, N11, N12): a short held-out or Ref_b brief waits for
    Shresth's answer, which D-023 records as one line (short_line) of its entry in the checkout's docs/DECISIONS.md,
    read by short_answers. That file is never compared with the run's commit (check.code_problems), so the line,
    written after the run, needs no redo and no constant changed. (The refusal, or None; the line applied, or None.)
    Only that line, with no other answer for the run, lifts the refusal: the short briefs are then handled as Q8(a)
    handles them. The refusal names the briefs, the line to add and where it goes (WHERE), with the lines of
    DECISIONS.md that mention it where no answer is read (N12: a line appended after a later entry is in that entry),
    or, when D-023 already answers the run otherwise, those answers (never the line's own) and the line to replace
    them with (finding N8): adding the line beside another answer would only refuse again. A short pool brief is
    reported under either answer, and under "report" nothing is read."""
    if S.SHORT_RULE != "ask":
        return None, None
    short = [f"{name} {e['brief_id']}" for name in ("heldout", "refb") for e in run.entries(name) if e.get("short")]
    if not short:
        return None, None
    try:
        decisions = (Path(repo) / S.DECISIONS).read_text(encoding="utf-8", errors="replace")
    except OSError:
        decisions = ""
    line, read = short_line(run.run_name), short_answers(decisions, run.run_name)
    said = [a for _, a in read]
    if said and all(a == SHORT_AS_A for a in said):
        return None, line
    briefs = (f"refused: SHORT_RULE is Q8's (b), ask: {len(short)} held-out or Ref_b briefs are short ("
              f"{', '.join(short[:5])}{', ...' if len(short) > 5 else ''})")
    why_a = "the held-out set holds fewer tuples, a short Ref_b brief leaves run 1's [0, 400)"
    if not said:
        # the lines that mention the run's answer where none is read: named, so the line is never added there again
        mention = re.compile(re.escape(SHORT_HEAD) + r"[`*_\s]*run " + re.escape(run.run_name) + r"\b")
        unread = [n for n, x in enumerate(decisions.splitlines(), 1) if mention.search(x)]
        where = (f"; {S.DECISIONS} mentions it on line{'s' if len(unread) > 1 else ''} "
                 f"{', '.join(map(str, unread[:5]))}"
                 f"{', ...' if len(unread) > 5 else ''}, where no answer is read" if unread else "")
        return (f"{briefs}; Shresth's answer goes into D-023 first: if he answers (a) for them ({why_a}), add the line "
                f"`{line}` to {S.DECISIONS}, {WHERE}, and run prereg again{where}: nothing written"), None
    others = [a for a in said if a != SHORT_AS_A]
    only = (f"beside the line `{line}`, and only that line" if len(others) < len(said) else
            f"and only the line `{line}`")
    return (f"{briefs}; {S.DECISIONS} answers {run.run_name} {', '.join(repr(a) for a in others)}, which prereg does "
            f"not apply, {only}, alone, lifts the refusal: if his answer for them is (a) ({why_a}), replace the run's "
            "short-answer lines in D-023's entry with that one line, leaving only it, and run prereg again: nothing "
            "written"), None


# --------------------------------------------------------------------------------------------- the command
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
        refused = C.code_problems(run, repo)              # prereg runs at the run's commit, as check does
        if refused:
            raise Step0Error(refused)
        report = C.evaluate(run, repo=repo)
    except Step0Error as err:
        return 2, [f"error: bad input, nothing written:\n{err}"]
    if not report["validity"]["valid"]:
        return 3, [f"prereg: {run.name} is not a valid run ({report['validity']['held']} of "
                   f"{report['validity']['of']} conditions hold; failing: {', '.join(report['validity']['failed'])}): "
                   "nothing written. `python -m lofi_rl.step0 check` reports each condition (spec section 4.2 says "
                   "what follows an invalid run)."]
    asked, answer = ask_first(run, repo)
    if asked:
        return 2, [asked]
    try:
        docs = documents(run, report, short_answer=answer)
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

