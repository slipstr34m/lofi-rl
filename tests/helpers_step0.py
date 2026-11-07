"""Synthetic step-0 runs for tests/test_step0_*.py: a run folder shaped as `lofi-step0` writes it (docs/design/
step0_samples.md §2.6), its summary/ holding generate.json, completions.jsonl, replay.jsonl, scored.jsonl, step0.json,
sets.json and clips.json, every value at what the unit registers, each file's fields exactly lofi_rl.step0.FIELDS'.

    run = make_run(tmp_path)                                    # every requested draw: 3,520 rows, a valid run
    run = make_run(tmp_path, {"5101:0": "a b a/G7 c d e f g"}, briefs=["5101:0", "596:0"])     # a small one

**The table**: brief id -> its 8 draws, k = 0..7, as a string of 8 specs (or a list), each `<label>[/<render>]`:
- a label from LABELS ("h0".."h7", "t0".."t7", "a".."z", "u0".."u7", "v0".."v7"): the draw passes G1-G6 and
  compile with its cell's text of
  that label, so one label in one cell is one canonical text (and one label in two cells two texts). The text is
  prompt v2's shown example of the cell, in the frozen palette's names, with its first kick grid rewritten from the
  label (variant_text): every one is in the grammar's language and passes, so junk.score_row scores it for real (and
  every one is a near-copy of the cell's shown example, J5 >= 0.8, with its A and B lines verbatim);
- G2, a completion cut at the cap (finish_reason "length"), or G6, the example with its vamp's first chord twice:
  both fail that gate for real (G6 inside the language, as the regex cannot see it);
- /G7, /E, /EE or /S (held-out draws only): its render fails G7; errors once and renders when retried; errors twice
  (undetermined); is skipped by the deadline (undetermined). Without one it renders.
A brief the table does not name gets the default draws (default_labels): held-out "h0".."h7", train "t0".."t7", or
"u0".."u7" and "v0".."v7" for the second and third of run 1's train briefs that share one cell, bpm and tonic (twins,
whose draws of one text would be one clip), so the default run holds no gate failure, no repeat, no short brief, no
train draw with a held-out text and no clip in two briefs' sets.

**What is written.** `briefs=None` writes every request (lofi_rl.step0.requests); `briefs` writes those briefs' draws
only, keeping each row's place `i` in the full order, with `requested` the rows written. completions.jsonl holds the
real messages' sha per cell and each cell's PROMPT_TOKENS; scored.jsonl is junk.score_row on each row (memoized per
text and cell: these texts score the same at every tempo and tonic) plus stream, brief_id, k and text_sha256;
replay.jsonl replays every held-out draw, in reverse order, the same; clips.json renders the held-out set round by
round by the lazy rule (lofi_rl.step0.select.pending) as the render step does, with clip ids by
lofi_rl.scoring.protocol.clip_id under the render config of lofi_rl.scoring.render_pool; step0.json holds junk.summarize
per stream and select's candidates, render and readout blocks; sets.json is select.sets_json over scored.jsonl and
clips.json. No FLAC or manifest is written (clips/ is never pulled), and nothing under data/ or the inbox.

Sound claims: none.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import lofi_rl.step0 as S
from lofi_rl import brief_streams, prompt_v2
from lofi_rl.brief import Brief
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.grammar import loopscript_regex
from lofi_rl.palette import ROOT, Palette, load_palette
from lofi_rl.prompt import messages_sha256
from lofi_rl.rl import junk as J
from lofi_rl.step0 import requests as R
from lofi_rl.step0 import select as X

JOB = "13800001"
STAMP = "commit=9ef969e00000 dirty=0 diff=- at=2026-11-07T19:00:00Z"
SOURCE_SHA8 = "5eed5eed"                    # render_code.source_sha8 (features.source_sha8 at the stamp's commit)
NODE, GPU = "b11-13.hpc.usc.edu", "NVIDIA A40, 46068 MiB"
DECISIONS_SHA256 = hashlib.sha256(b"synthetic docs/DECISIONS.md").hexdigest()
PATH_FILE_SHA256 = hashlib.sha256(b"synthetic docs/prereg/D-023_step0_path.txt").hexdigest()
LABELS = (tuple(f"h{k}" for k in range(8)) + tuple(f"t{k}" for k in range(8)) + tuple("abcdefghijklmnopqrstuvwxyz")
          + tuple(f"u{k}" for k in range(8)) + tuple(f"v{k}" for k in range(8)))     # 58 labels: 6 kick bits hold 64
TRAIN_LABELS = ("t", "u", "v")              # a train brief's default labels, by its rank among its twins
PROBE_LABELS = ("a", "b")                   # the probe's text at each of PROBE_SEEDS, as an engine honouring seeds
GATES = ("G2", "G6")                        # the gate failures the writer makes for real
RENDERS = ("", "G7", "E", "EE", "S")        # a held-out draw's render: ok, G7, error once, error twice, deadline skip
KICK_CELLS = (2, 5, 8, 11, 13, 14)          # the kick hits a label's number sets, beside the downbeat
BACKEND_RECORD = {"requested": "xgrammar", "pin_accepted": True, "backend": "xgrammar",
                  "source": "llm_engine.vllm_config.structured_outputs_config.backend", "applied": True}
# stand-ins for what the engine reports (VllmEngine.info): none is a measurement
ENGINE_INFO = {"engine": "vllm", "vllm": "0.32.0", "torch": "2.9.0", "transformers": "5.19.0",
               "path": "/scratch1/hf/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f", "dtype": "bfloat16",
               "enforce_eager": True, "gpu_memory_utilization": S.GPU_MEM, "max_model_len": S.MAX_MODEL_LEN,
               "seed": S.ENGINE_SEED, "mp_method": "fork", "structured_outputs": BACKEND_RECORD}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def palette() -> Palette:
    return load_palette(ROOT / S.PALETTE, verify=False)


@lru_cache(maxsize=1)
def streams():
    return R.load()


@lru_cache(maxsize=1)
def all_requests() -> tuple:
    return R.requests(streams=streams())


@lru_cache(maxsize=1)
def render_cfg() -> dict:
    from lofi_rl.scoring.render_pool import render_config
    return render_config()


@lru_cache(maxsize=1)
def _twin_rank() -> dict[str, int]:
    """Each of run 1's train briefs -> its rank, in stream order, among the train briefs of [0, 400) that share its
    twin key (energy, mood, bpm, tonic): 0 for the first. Ranks go up to 2 (three triples)."""
    st = streams()["train"]
    seen: dict[tuple, int] = {}
    out = {}
    for p in st.picks:
        if S.RUN1[0] <= p.index < S.RUN1[1]:
            key = brief_streams.twin_key(p.brief)
            out[brief_streams.brief_id(st.spec.seed, p.index)] = seen.get(key, 0)
            seen[key] = seen.get(key, 0) + 1
    return out


def default_labels(stream: str, brief_id: str) -> list[str]:
    """A brief's default draws (module doc): held-out h0-h7; train t0-t7, u0-u7 or v0-v7 by its rank among its twins,
    so that no two briefs' default draws are one clip."""
    if stream == "rm_heldout":
        return [f"h{k}" for k in range(S.DRAWS)]
    return [f"{TRAIN_LABELS[_twin_rank().get(brief_id, 0)]}{k}" for k in range(S.DRAWS)]


# --------------------------------------------------------------------------------------------- texts
def _example(cell: str) -> str:
    energy, mood = cell.split("/")
    ((_, text),) = prompt_v2.shown_examples(Brief(energy, mood, 82, 0), palette())
    return text


@lru_cache(maxsize=None)
def variant_text(cell: str, label: str) -> str:
    """The cell's text of a label: its shown example with the first kick grid set from the label's number (module
    doc). Distinct labels give distinct canonical texts within a cell."""
    n = LABELS.index(label)
    lines = _example(cell).splitlines()
    j = next(j for j, line in enumerate(lines) if line.startswith("kick: "))
    cells = ["x"] + ["."] * 15
    for bit, pos in enumerate(KICK_CELLS):
        if n >> bit & 1:
            cells[pos] = "x"
    grids = lines[j].split(": ", 1)[1].split()
    lines[j] = "kick: " + " ".join(["".join(cells), *grids[1:]])
    return "\n".join(lines) + "\n"


@lru_cache(maxsize=None)
def gate_text(cell: str, gate: str) -> tuple[str, str]:
    """(text, finish_reason) of a draw that fails `gate`: G2 cut at 60 % of the cell's h0 text, G6 its vamp's first
    chord twice."""
    base = variant_text(cell, "h0")
    if gate == "G2":
        return base[:int(len(base) * 0.6)], "length"
    lines = base.splitlines()
    j = next(j for j, line in enumerate(lines) if line.startswith("vamp: "))
    words = lines[j].split()
    lines[j] = " ".join([words[0], words[1], words[1], *words[3:]])
    return "\n".join(lines) + "\n", "stop"


@lru_cache(maxsize=None)
def _scored(text: str, energy: str, mood: str, finish_reason: str) -> str:
    """junk.score_row of one text in one cell, as JSON (the cache hands out copies)."""
    row = {"i": 0, "brief": J.brief_dict(Brief(energy, mood, 82, 0)), "text": text, "finish_reason": finish_reason,
           "n_tokens": n_tokens(text, finish_reason)}
    return json.dumps(J.score_row(row, palette(), _check()))


@lru_cache(maxsize=1)
def _check() -> J.CopyCheck:
    return J.CopyCheck(S.PROMPT_VERSION, palette())


def n_tokens(text: str, finish_reason: str) -> int:
    return J.SAMPLING["max_tokens"] if finish_reason == "length" else len(text) // 3


def parse_specs(bid: str, specs: str | Sequence[str], heldout: bool) -> list[tuple[str, str]]:
    """A brief's 8 draw specs as (label or gate, render) pairs (module doc). ValueError on a malformed one."""
    items = specs.split() if isinstance(specs, str) else list(specs)
    if len(items) != S.DRAWS:
        raise ValueError(f"{bid}: {len(items)} draw specs, not {S.DRAWS}")
    out = []
    for spec in items:
        label, _, render = spec.partition("/")
        if label not in LABELS and label not in GATES:
            raise ValueError(f"{bid}: {spec!r}: a draw is a label of LABELS or one of {GATES}")
        if render not in RENDERS or (render and (not heldout or label in GATES)):
            raise ValueError(f"{bid}: {spec!r}: a render outcome is one of {RENDERS[1:]}, on a passing held-out draw")
        out.append((label, render))
    return out


# --------------------------------------------------------------------------------------------- the run
@dataclass
class Run:
    dir: Path
    requests: list                  # the requests written, in order (lofi_rl.step0.requests.Request)
    rows: list[dict]                # completions.jsonl
    scored: list[dict]              # scored.jsonl
    records: list[dict]             # clips.json's records
    files: dict[str, object]        # summary file name -> what was written (dict, or list of rows)

    @property
    def summary(self) -> Path:
        return self.dir / "summary"

    def doc(self, name: str):
        return self.files[name]


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _record(row: dict, scored: dict, rnd: int, outcome: str, final: bool) -> dict:
    """One render record as the render step writes it: junk.render_clip's, plus CLIP_ADDED."""
    i, b = row["i"], Brief(**row["brief"])
    from lofi_rl.scoring import protocol
    cfg = render_cfg()
    if outcome == "ok":
        n = loop_samples(b.bpm)
        rec = {"i": i, "ok": True, "gate": None, "clip": f"clips/{i:04d}.flac", "n": n, "seconds": round(n / 44100, 3),
               "lufs": -14.0, "true_peak_dbtp": -1.5, "audio_sha256": sha(f"pcm {i}"),
               "canon_sha256": scored["canon_sha256"], "p_fix": scored["p_fix"], "render_s": 2.08,
               "flac_bytes": 2562058}
        clip = protocol.clip_id(scored["canon_sha256"], b, palette().id, palette().sha256, cfg)
    elif outcome == "G7":
        rec, clip = {"i": i, "ok": False, "gate": "G7", "detail": "synthetic: the render is out of spec"}, None
    elif outcome == "error":
        rec, clip = {"i": i, "ok": False, "gate": None, "error": "OSError: synthetic: fluidsynth died"}, None
    else:
        rec, clip = {"i": i, "ok": False, "gate": None, "skipped": "render deadline"}, None
    return {**rec, "brief_id": row["brief_id"], "k": row["k"], "round": rnd, "final": final, "clip_id": clip,
            "render_cfg_sha8": protocol.render_cfg_sha8(cfg)}


def render_rounds(rows: list[dict], scored: list[dict], renders: Mapping[int, str]) -> list[dict]:
    """The held-out renders, round by round (select.pending): each draw's outcome is `renders[i]` (RENDERS)."""
    by_i = {r["i"]: r for r in rows}
    sc = {s["i"]: s for s in scored}
    records: list[dict] = []
    tried: dict[int, int] = {}
    rnd = 0
    while True:
        todo = X.pending(scored, records, streams=streams())
        if not todo:
            return records
        rnd += 1
        for i in todo:
            spec = renders.get(i, "")
            tried[i] = tried.get(i, 0) + 1
            if spec == "G7":
                rec = _record(by_i[i], sc[i], rnd, "G7", True)
            elif spec == "S":
                rec = _record(by_i[i], sc[i], rnd, "skipped", True)
            elif spec in ("E", "EE") and tried[i] == 1:
                rec = _record(by_i[i], sc[i], rnd, "error", False)
            elif spec == "EE":
                rec = _record(by_i[i], sc[i], rnd, "error", True)
            else:
                rec = _record(by_i[i], sc[i], rnd, "ok", True)
            records.append(rec)


def make_run(root: Path, table: Mapping[str, str | Sequence[str]] | None = None, *,
             briefs: Iterable[str] | None = None, job: str = JOB, stamp: str = STAMP,
             source_sha8: str = SOURCE_SHA8) -> Run:
    """A synthetic run under `root` (module doc). Returns what it wrote."""
    table = dict(table or {})
    reqs = list(all_requests())
    known = {r.brief_id for r in reqs}
    unknown = sorted(set(table) - known)
    if unknown:
        raise ValueError(f"{unknown}: not briefs the step draws")
    if briefs is not None:
        keep = set(briefs) | set(table)
        reqs = [r for r in reqs if r.brief_id in keep]
    specs: dict[str, list[tuple[str, str]]] = {}
    for r in reqs:
        if r.brief_id not in specs:
            held = r.stream == "rm_heldout"
            specs[r.brief_id] = parse_specs(r.brief_id, table.get(r.brief_id, default_labels(r.stream, r.brief_id)),
                                            held)
    pal = palette()
    cells_msgs = {}
    rows, scored, renders = [], [], {}
    for r in reqs:
        b = r.brief
        if r.cell not in cells_msgs:
            cells_msgs[r.cell] = prompt_v2.build_messages(b, pal)
        label, render = specs[r.brief_id][r.k]
        if label in GATES:
            text, finish = gate_text(r.cell, label)
        else:
            text, finish = variant_text(r.cell, label), "stop"
        row = {"i": r.i, "stream": r.stream, "seed": r.seed, "index": r.index, "brief_id": r.brief_id, "k": r.k,
               "draw_seed": r.draw_seed, "sp_seed": r.draw_seed, "brief": J.brief_dict(b),
               "prompt_sha256": messages_sha256(cells_msgs[r.cell]), "prompt_version": S.PROMPT_VERSION,
               "example": prompt_v2.example_id(b), "text": text, "finish_reason": finish,
               "n_tokens": n_tokens(text, finish), "prompt_tokens": S.PROMPT_TOKENS[r.cell]}
        rows.append(row)
        s = json.loads(_scored(text, b.energy, b.mood, finish))
        s["i"] = r.i
        scored.append({**s, "stream": r.stream, "brief_id": r.brief_id, "k": r.k, "text_sha256": sha(text)})
        if render:
            renders[r.i] = render
    records = render_rounds(rows, scored, renders)

    summ = Path(root) / "summary"
    summ.mkdir(parents=True, exist_ok=True)
    render_code = {"stamp": stamp, "source_sha8": source_sha8}
    held = [r for r in rows if r["stream"] == "rm_heldout"]
    replay = [{"i": r["i"], "text": r["text"], "finish_reason": r["finish_reason"], "n_tokens": r["n_tokens"],
               "same": True} for r in reversed(held)]
    # stand-in timings at the gate run's rate per request (spec §2.9), not measurements
    chunks = [{"start": s, "count": len(rows[s:s + S.CHUNK]), "s": round(0.1002 * len(rows[s:s + S.CHUNK]), 2),
               "tokens": sum(r["n_tokens"] for r in rows[s:s + S.CHUNK])} for s in range(0, len(rows), S.CHUNK)]
    gen_s, tokens = round(sum(c["s"] for c in chunks), 2), sum(c["tokens"] for c in chunks)
    regex = loopscript_regex(pal)
    generate = {
        "schema": S.GENERATE_SCHEMA, "partial": False, "requested": len(rows), "generated": len(rows), "stopped": None,
        "streams": R.stream_counts(reqs),
        "streams_file": {"path": S.STREAMS_FILE, "sha256": S.STREAMS_SHA256, "fresh": True},
        "registration": {"found": True, "mark": S.MARK, "decisions_sha256": DECISIONS_SHA256},
        "path_file": {"sha256": PATH_FILE_SHA256, "blobs_ok": True},
        "model": {"repo": S.MODEL, "revision": S.MODEL_REVISION},
        "prompt": J.prompt_record(S.PROMPT_VERSION, pal, list(cells_msgs.values())),
        "palette": {"id": pal.id, "sha256": pal.sha256, "path": S.PALETTE},
        "sampling": dict(J.SAMPLING), "per_request_seed": True,
        "constrained": {**J.constrained_block(regex), "backend": dict(BACKEND_RECORD)},
        "engine": dict(ENGINE_INFO),
        "sampling_params": None if not rows else (
            "SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, "
            f"top_p=1.0, top_k=0, min_p=0.0, seed={rows[0]['draw_seed']}, max_tokens=512, "
            "structured_outputs=StructuredOutputsParams(regex=...))"),
        "probe": {"seeds": list(S.PROBE_SEEDS), "cell": "/".join(S.PROBE_CELL), "n": S.PROBE_N,
                  "texts_sha256": [sha(variant_text("/".join(S.PROBE_CELL), label)) for label in PROBE_LABELS
                                   for _ in range(S.PROBE_N)], "distinct": len(PROBE_LABELS), "passed": True},
        "replay": {"n": len(replay), "same": len(replay), "differ": 0,
                   "by_chunk": [{"start": s, "count": len(replay[s:s + S.REPLAY_CHUNK]),
                                 "same": len(replay[s:s + S.REPLAY_CHUNK])}
                                for s in range(0, len(replay), S.REPLAY_CHUNK)], "stopped": None},
        "timing": {"load_s": 29.4, "probe_s": 6.1, "chunks": chunks, "gen_s": gen_s, "tokens": tokens,
                   "tok_per_s": round(tokens / gen_s, 1) if gen_s else None, "replay_s": round(0.1002 * len(replay), 2),
                   "total_s": round(72.9 + gen_s + 0.1002 * len(replay), 1)},
        "host": {"node": NODE, "job": job, "stamp": stamp, "gpu": GPU},
    }
    by_stream = {}
    for name in dict.fromkeys(r["stream"] for r in rows):
        rs = [r for r in rows if r["stream"] == name]
        ss = [s for s in scored if s["stream"] == name]
        block = J.summarize(rs, ss, {**generate, "requested": len(rs)}, pal, _check(), run=f"step0-{job}")
        block["score_stamp"] = stamp            # summarize reads this process's .sync-stamp; the job's is the run's
        by_stream[name] = block
    sets = X.sets_json(scored, records, streams=streams(), render_code=render_code)
    step0 = {"schema": S.STEP0_SCHEMA, "run": f"step0-{job}", "n": len(rows), "requested": len(rows),
             "partial": False, "unreadable_lines": 0, "by_stream": by_stream,
             "candidates": X.candidates_block(scored, streams=streams()), "render": X.render_block(records),
             "render_code": render_code, "sets": X.readouts(sets, scored), "score_stamp": stamp, "render_stamp": stamp}
    clips = {"schema": S.CLIPS_SCHEMA, "render_code": render_code, "palette": {"id": pal.id, "sha256": pal.sha256},
             "jobs": 2, "workers": 2, "clips": records}
    _write_json(summ / "generate.json", generate)
    _write_jsonl(summ / "completions.jsonl", rows)
    _write_jsonl(summ / "replay.jsonl", replay)
    _write_jsonl(summ / "scored.jsonl", scored)
    _write_json(summ / "step0.json", step0)
    _write_json(summ / "sets.json", sets)
    _write_json(summ / "clips.json", clips)
    files = {"generate.json": generate, "completions.jsonl": rows, "replay.jsonl": replay, "scored.jsonl": scored,
             "step0.json": step0, "sets.json": sets, "clips.json": clips}
    return Run(Path(root), reqs, rows, scored, records, files)
