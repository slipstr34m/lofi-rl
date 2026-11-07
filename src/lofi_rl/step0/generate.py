"""generate: the step-0 draws (docs/design/step0_samples.md §2.3; D-023), in the policy env (vLLM).

    python -m lofi_rl.step0 generate --run DIR [--deadline-s 960] [--chunk 128] [--hang-s S]

It takes no model, prompt, palette, sampling or stream: they are registered (lofi_rl.step0). It draws the requests
lofi_rl.step0.requests lists for STREAMS, request i as row i of DIR/summary/completions.jsonl.

1. **Pre-checks**, before the model loads, against the code tree the step runs from (the job's snapshot). Any that
   fails stops generate with exit 2, every failure named in `stopped`, and nothing drawn:
   - the streams file: brief_streams.load() accepts it (exactly a fresh build), and its sha256 is STREAMS_SHA256;
   - the palette's sha256 is PALETTE_SHA256, and its grammar's regex sha256 REGEX_SHA256;
   - prompt v2's template sha256 is TEMPLATE_SHA256;
   - every message builds (a PromptError stops it, as it stops junk's generate), once per energy x mood cell: a
     cell's briefs share their messages, since the model sees only the cell;
   - the registration: docs/DECISIONS.md holds D-023's mark (`find_marks`), the mark is MARK, and MARK is the line
     requests.mark_line gives STREAMS and RULE, so nothing is drawn under rules or streams not registered first;
   - the path file: every file docs/prereg/D-023_step0_path.txt lists hashes to its blob in this tree
     (path.check_tree, no git);
   - PROBE_SEEDS are distinct and none of them is one of the streams file's draw seeds.
2. **The engine**: junk.load_engine(MODEL@MODEL_REVISION, seed=ENGINE_SEED, the gate run's gpu_mem and max_model_len,
   xgrammar pinned and read back), in bf16 and eager mode as junk loads it. vLLM's StructuredOutputsParams is imported
   first, so a vLLM without it fails before the model loads.
3. **One SamplingParams per request**: engine.sampling(**junk.SAMPLING, structured_outputs=so, seed=draw_seed), n 1 by
   construction (VllmEngine.sampling), with one StructuredOutputsParams `so` built once and shared, as junk shared one
   SamplingParams across its requests. Each row records its draw_seed and sp_seed, read back from the object it was
   sent with.
4. **The seed probe**, before any draw: one call of PROBE_N identical requests at each of PROBE_SEEDS (the low/hopeful
   cell's messages, the registered sampling and grammar). It passes when, at each seed, at least PROBE_SAME of its
   texts are identical and the seeds' most common texts all differ, so an engine that gives every request one
   randomness, or every request the call's first SamplingParams, misses as one that ignores seeds does (finding
   R2-3); a miss stops generate (`stopped: "seed probe"`, exit 2) with nothing drawn. Only the texts' sha256 are kept.
5. **The draws**: the requests in order, in chunks of --chunk, one engine.chat(convs, [sp, ...]) call each, the
   outputs taken in input order and their count checked. Each chunk's rows are appended to completions.jsonl, flushed
   and fsynced, and generate.json is rewritten, so a kill keeps what was done. No chunk starts that junk's forecast
   puts past --deadline-s (`stopped: "deadline"`).
6. **The replay**, a readout: once every request is drawn, the rm_heldout requests again, in reverse order, in chunks
   of REPLAY_CHUNK, with their own seeds, to replay.jsonl only, each row saying whether its text is the draw's. It
   obeys the deadline, and a cut, failed or killed replay (`replay.stopped` "deadline", "error: ..." or, after a
   kill, "running") leaves the draws complete: generate.json says they are (`partial` false) before the replay
   starts. Its rows are never scored, rendered or selected, and they are held-out data (§5).
7. **The watchdog** (main): faulthandler dumps every thread's stack and exits at deadline + junk.HANG_MARGIN_S, then
   alarm(120) ends a hang in vLLM's teardown, as junk's generate does.

generate.json (lofi-rl/step0-generate/1) holds lofi_rl.step0.FIELDS' fields, rewritten after every chunk, with `trace`
on an error. `engine` is VllmEngine.info as the engine loaded, and `sampling_params` the first draw's SamplingParams
repr, since VllmEngine.sampling overwrites info["sampling_params"] on every call. `partial` is true until the draws
are complete (every request drawn: it turns false before the replay) or generate ends, so a run killed while drawing
reads partial and one killed in the replay does not; a run the deadline cut ends with `stopped` "deadline" and fewer
rows than `requested`.

Exit 0: every request drawn (the replay may be cut); 1: an error, or the deadline cut the draws (a partial run);
2: a pre-check or the probe stopped it, with nothing drawn. Sound claims: none.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import lofi_rl.step0 as S
from lofi_rl import grammar, prompt_v2
from lofi_rl.brief import ENERGIES, MOOD_NAMES
from lofi_rl.brief_streams import ROOT
from lofi_rl.palette import PaletteError, load_palette
from lofi_rl.prompt import PromptError, messages_sha256
from lofi_rl.rl import junk as J
from lofi_rl.step0 import Step0Error
from lofi_rl.step0 import path as P
from lofi_rl.step0 import requests as R

MARK_HEAD, MARK_END, find_marks = R.MARK_HEAD, R.MARK_END, R.find_marks     # one reader of D-023's lines
PROBE = "/".join(S.PROBE_CELL)
STOPPED_CHARS, TRACE_CHARS = 2000, 6000     # as junk keeps them


def _now() -> float:
    return time.monotonic()


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)                       # a kill mid-write never leaves a cut file


def sync_stamp(root: Path) -> str | None:
    """The tree's .sync-stamp as this process sees it (each step of the job is its own process)."""
    stamp = Path(root) / ".sync-stamp"
    return stamp.read_text().strip() if stamp.is_file() else None


def _host(root: Path) -> dict:
    info = {"node": platform.node(), "job": os.environ.get("SLURM_JOB_ID"), "stamp": sync_stamp(root), "gpu": None}
    if shutil.which("nvidia-smi"):              # never torch.cuda here: one CUDA process on the GPU (smoke.py)
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30)
            info["gpu"] = out.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def sp_seed(sp) -> int | None:
    """The seed a SamplingParams object carries, read back from it (a Mapping's "seed" for a stand-in)."""
    return sp.get("seed") if isinstance(sp, Mapping) else getattr(sp, "seed", None)


# --------------------------------------------------------------------------------------------- the pre-checks
def registration(root: Path) -> tuple[dict, list[str]]:
    """generate.json's `registration` and what stops the run: docs/DECISIONS.md must hold a D-023 mark equal to MARK
    (`found`), and MARK must be the line requests.mark_line gives STREAMS and RULE (requests.check_mark)."""
    rec: dict = {"found": False, "mark": None, "decisions_sha256": None}
    problems: list[str] = []
    try:
        R.check_mark()
    except Step0Error as err:
        problems += err.problems
    try:
        raw = (Path(root) / S.DECISIONS).read_bytes()
    except OSError as err:
        return rec, problems + [f"cannot read {S.DECISIONS} ({err.strerror or err})"]
    rec["decisions_sha256"] = hashlib.sha256(raw).hexdigest()
    marks = find_marks(raw.decode("utf-8", errors="replace"))
    rec["found"] = S.MARK in marks
    rec["mark"] = S.MARK if rec["found"] else (marks[0] if marks else None)
    if not marks:
        problems.append(f"{S.DECISIONS} holds no {MARK_HEAD!r} line: D-023 is not registered in this tree")
    elif not rec["found"]:
        problems.append(f"{S.DECISIONS}'s D-023 mark is {marks[0]!r}, not MARK {S.MARK!r}")
    return rec, problems


def path_file(root: Path) -> tuple[dict, list[str]]:
    """generate.json's `path_file` ({sha256, blobs_ok}, path.check_tree in this tree) and what stops the run."""
    try:
        rec = P.check_tree(root)
    except Step0Error as err:
        return {"sha256": None, "blobs_ok": False}, [f"the path file: {p}" for p in err.problems]
    out = {"sha256": rec["sha256"], "blobs_ok": rec["blobs_ok"]}
    if rec["blobs_ok"]:
        return out, []
    parts = [f"{rec['matched']} of {rec['files']} files {S.PATH_FILE} lists have their blob in this tree"]
    if rec["n_changed"]:
        parts.append(f"{rec['n_changed']} differ (first {rec['changed'][0]['path']})")
    if rec["n_missing"]:
        parts.append(f"{rec['n_missing']} are missing (first {rec['missing'][0]})")
    if rec["n_malformed"]:
        parts.append(f"{rec['n_malformed']} of its lines are malformed (first line {rec['malformed'][0]['line']})")
    return out, ["; ".join(parts)]


def streams_file(root: Path) -> tuple[dict, dict | None, list[str]]:
    """generate.json's `streams_file` ({path, sha256, fresh}), the registered streams (None when refused) and what
    stops the run: brief_streams.load() must accept the file, whose sha256 must be STREAMS_SHA256."""
    rel = S.STREAMS_FILE
    rec: dict = {"path": rel, "sha256": None, "fresh": False}
    try:
        rec["sha256"] = hashlib.sha256((Path(root) / rel).read_bytes()).hexdigest()
    except OSError as err:
        return rec, None, [f"cannot read {rel} ({err.strerror or err})"]
    problems, streams = [], None
    try:
        streams = R.load(Path(root) / rel)
        rec["fresh"] = True
    except Step0Error as err:
        problems += err.problems
    if rec["sha256"] != S.STREAMS_SHA256:
        problems.append(f"{rel} has sha256 {S.short(rec['sha256'])}, not the registered {S.short(S.STREAMS_SHA256)}")
    return rec, streams, problems


@dataclass(frozen=True)
class Plan:
    """What the draws need once the pre-checks pass."""
    reqs: tuple                             # requests.Request, in order
    regex: str
    msgs: Mapping[str, list[dict]]          # cell -> its messages (prompt v2)
    shas: Mapping[str, str]                 # cell -> messages_sha256
    examples: Mapping[str, str]             # cell -> the id of the example its prompt shows


def prechecks(root: Path, meta: dict) -> tuple[Plan | None, list[str]]:
    """Step 1 (module doc): fills meta's requested, streams, streams_file, registration, path_file, prompt, palette and
    constrained, and gives the plan, or None and every problem found (each named once)."""
    root = Path(root)
    meta["streams_file"], streams, problems = streams_file(root)
    pal = regex = None
    try:
        pal = load_palette(root / S.PALETTE, verify=False, root=root)   # names only: no audio in the policy env
    except (OSError, PaletteError) as err:
        problems.append(f"{S.PALETTE} cannot be loaded ({err})")
    if pal is not None:
        meta["palette"] = {"id": pal.id, "sha256": pal.sha256, "path": S.PALETTE}
        if pal.sha256 != S.PALETTE_SHA256:
            problems.append(f"{S.PALETTE} has sha256 {S.short(pal.sha256)}, not the frozen "
                            f"{S.short(S.PALETTE_SHA256)}")
        regex = grammar.loopscript_regex(pal)
        meta["constrained"] = J.constrained_block(regex)
        if meta["constrained"]["regex_sha256"] != S.REGEX_SHA256:
            problems.append(f"the palette's regex has sha256 {S.short(meta['constrained']['regex_sha256'])}, not the "
                            f"registered {S.short(S.REGEX_SHA256)}")
    mod = J.PROMPTS[S.PROMPT_VERSION]
    if mod.template_sha256() != S.TEMPLATE_SHA256:
        problems.append(f"prompt {S.PROMPT_VERSION}'s template has sha256 {S.short(mod.template_sha256())}, not the "
                        f"registered {S.short(S.TEMPLATE_SHA256)}")
    msgs, failed = {}, {}
    for e in ENERGIES:                          # every message, once per cell: its briefs share their messages
        for m in MOOD_NAMES:
            if pal is None:
                continue
            try:
                msgs[f"{e}/{m}"] = prompt_v2.build_messages(J.cell_brief(e, m), pal)
            except PromptError as err:
                failed[f"{e}/{m}"] = str(err)
    if failed:
        first = next(iter(failed))
        problems.append(f"PromptError in {len(failed)} cells (first {first}: {failed[first]})")
    meta["registration"], more = registration(root)
    problems += more
    meta["path_file"], more = path_file(root)
    problems += more
    reqs = None
    if streams is not None:
        try:
            reqs = R.requests(root / S.STREAMS_FILE, streams=streams)
        except Step0Error as err:
            problems += err.problems
        seeds = {s for st in streams.values() for p in st.picks for s in p.draw_seeds}
        problems += [f"PROBE_SEEDS {s} is one of {S.STREAMS_FILE}'s draw seeds" for s in S.PROBE_SEEDS if s in seeds]
        if len(set(S.PROBE_SEEDS)) != len(S.PROBE_SEEDS) or len(S.PROBE_SEEDS) < 2:
            problems.append(f"PROBE_SEEDS {S.PROBE_SEEDS!r}: the probe needs two seeds or more, all distinct")
    if reqs is not None:
        meta["requested"], meta["streams"] = len(reqs), R.stream_counts(reqs)
    if pal is not None and not failed:
        drawn = dict.fromkeys(r.cell for r in reqs or ())   # the cells the run sends: prompt_record's longest prompt
        meta["prompt"] = J.prompt_record(S.PROMPT_VERSION, pal, [msgs[c] for c in drawn] or list(msgs.values()))
    else:
        meta["prompt"] = {"version": S.PROMPT_VERSION, "module": mod.__name__, "template_sha256": mod.template_sha256()}
    problems = list(dict.fromkeys(problems))
    if problems:
        return None, problems
    return Plan(reqs, regex, msgs, {c: messages_sha256(v) for c, v in msgs.items()},
                {c: J.example_id(S.PROMPT_VERSION, J.cell_brief(*c.split("/"))) for c in msgs}), []


# --------------------------------------------------------------------------------------------- the engine's calls
def too_late(elapsed: float, deadline: float, timed: Sequence[Mapping], n: int) -> bool:
    """junk's deadline forecast (junk.py:547-554): no chunk starts at or after the deadline, nor once 1.25 x the
    longest chunk timed so far would end past it. A timed chunk of fewer requests than this one is scaled up to its
    n (the replay's chunks are larger than the draws'), never down, so the draws' forecast is junk's own."""
    longest = max((c["s"] * max(1.0, n / c["count"]) for c in timed if c["count"]), default=None)
    return elapsed >= deadline or (longest is not None and elapsed + 1.25 * longest > deadline)


def _sampling(engine, so, seeds: Sequence[int]) -> list:
    return [engine.sampling(**J.SAMPLING, structured_outputs=so, seed=s) for s in seeds]


def _chat(engine, convs: list, sps: list) -> tuple[list, float]:
    t = _now()
    outs = engine.chat(convs, sps)
    dt = _now() - t
    if len(outs) != len(convs):
        raise RuntimeError(f"the engine returned {len(outs)} completions for {len(convs)} requests")
    return outs, dt


def _conv(plan: Plan, cell: str) -> list[dict]:
    return [dict(m) for m in plan.msgs[cell]]       # fresh dicts for every request, as build_messages gives them


def probe_passed(shas: Sequence[str], n: int = S.PROBE_N) -> bool:
    """The probe's verdict on its texts' sha256, PROBE_N a seed in PROBE_SEEDS' order: at each seed at least
    PROBE_SAME identical, and the seeds' most common texts all different."""
    groups = [shas[j:j + n] for j in range(0, len(shas), n)]
    tops = [Counter(g).most_common(1)[0] for g in groups if g]
    return (len(tops) == len(S.PROBE_SEEDS) and all(c >= S.PROBE_SAME for _, c in tops)
            and len({t for t, _ in tops}) == len(tops))


def probe(engine, so, plan: Plan) -> tuple[dict, float]:
    """Step 4 (module doc): generate.json's `probe` and its time."""
    seeds = [s for s in S.PROBE_SEEDS for _ in range(S.PROBE_N)]
    outs, dt = _chat(engine, [_conv(plan, PROBE) for _ in seeds], _sampling(engine, so, seeds))
    shas = [J.sha256_text(c.text) for c in outs]
    return ({"seeds": list(S.PROBE_SEEDS), "cell": PROBE, "n": S.PROBE_N, "texts_sha256": shas,
             "distinct": len(set(shas)), "passed": probe_passed(shas)}, round(dt, 2))


def _row(plan: Plan, r, sp, c: J.Completion) -> dict:
    return {"i": r.i, "stream": r.stream, "seed": r.seed, "index": r.index, "brief_id": r.brief_id, "k": r.k,
            "draw_seed": r.draw_seed, "sp_seed": sp_seed(sp), "brief": J.brief_dict(r.brief),
            "prompt_sha256": plan.shas[r.cell], "prompt_version": S.PROMPT_VERSION, "example": plan.examples[r.cell],
            "text": c.text, "finish_reason": c.finish_reason, "n_tokens": c.n_tokens, "prompt_tokens": c.prompt_tokens}


def draw(engine, so, plan: Plan, chunk: int, deadline: float, meta: dict, summ: Path, t0: float) -> dict[int, str]:
    """Step 5 (module doc): every request, chunk by chunk, into completions.jsonl. Gives the held-out draws' texts
    (i -> text), which the replay compares with."""
    chunks = meta["timing"]["chunks"]
    held: dict[int, str] = {}
    with (summ / "completions.jsonl").open("w", encoding="utf-8") as f:
        for start in range(0, len(plan.reqs), chunk):
            part = plan.reqs[start:start + chunk]
            # chunk 1 also pays the one-time grammar compile: once later chunks exist, predict from them (junk)
            if too_late(_now() - t0, deadline, chunks[1:] or chunks, len(part)):
                meta["stopped"] = "deadline"
                break
            sps = _sampling(engine, so, [r.draw_seed for r in part])
            if start == 0:
                meta["sampling_params"] = repr(sps[0])
            outs, dt = _chat(engine, [_conv(plan, r.cell) for r in part], sps)
            for r, sp, c in zip(part, sps, outs):
                f.write(json.dumps(_row(plan, r, sp, c), ensure_ascii=False) + "\n")
                if r.stream == "rm_heldout":
                    held[r.i] = c.text
            f.flush()
            os.fsync(f.fileno())
            chunks.append({"start": start, "count": len(outs), "s": round(dt, 2),
                           "tokens": sum(c.n_tokens for c in outs)})
            meta["generated"] += len(outs)
            _write_json(summ / "generate.json", meta)
            print(f"[step0:generate] {meta['generated']}/{len(plan.reqs)} in {_now() - t0:.0f} s", flush=True)
    return held


def replay(engine, so, plan: Plan, held: Mapping[int, str], deadline: float, meta: dict, summ: Path,
           t0: float) -> None:
    """Step 6 (module doc). Never raises: an error is the replay's `stopped` (with `trace`), and the draws stand.
    `stopped` reads "running" (generate sets it before the call) until the last chunk is written, so a kill during
    the replay leaves "running" beside draws that are complete."""
    rp, timing = meta["replay"], meta["timing"]
    reqs = [r for r in plan.reqs if r.stream == "rm_heldout"][::-1]
    timed = list(timing["chunks"][1:] or timing["chunks"])
    try:
        with (summ / "replay.jsonl").open("w", encoding="utf-8") as f:
            for start in range(0, len(reqs), S.REPLAY_CHUNK):
                part = reqs[start:start + S.REPLAY_CHUNK]
                if too_late(_now() - t0, deadline, timed, len(part)):
                    rp["stopped"] = "deadline"
                    break
                outs, dt = _chat(engine, [_conv(plan, r.cell) for r in part],
                                 _sampling(engine, so, [r.draw_seed for r in part]))
                same = 0
                for r, c in zip(part, outs):
                    hit = c.text == held[r.i]
                    same += hit
                    f.write(json.dumps({"i": r.i, "text": c.text, "finish_reason": c.finish_reason,
                                        "n_tokens": c.n_tokens, "same": hit}, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                timed.append({"count": len(outs), "s": dt})
                rp["by_chunk"].append({"start": start, "count": len(outs), "same": same})
                rp["n"] += len(outs)
                rp["same"] += same
                rp["differ"] = rp["n"] - rp["same"]
                timing["replay_s"] = round((timing["replay_s"] or 0.0) + dt, 2)
                _write_json(summ / "generate.json", meta)
            else:
                rp["stopped"] = None                    # every chunk replayed
    except Exception as err:  # noqa: BLE001: a readout that fails never costs the draws
        rp["stopped"] = f"error: {type(err).__name__}: {err}"[:STOPPED_CHARS]
        meta["trace"] = traceback.format_exc()[-TRACE_CHARS:]


# --------------------------------------------------------------------------------------------- the step
def _finish(meta: dict, summ: Path, t0: float, rc: int, stopped: str | None = None) -> int:
    if stopped is not None:
        meta["stopped"] = stopped[:STOPPED_CHARS]
    timing, chunks = meta["timing"], meta["timing"]["chunks"]
    gen_s, tokens = round(sum(c["s"] for c in chunks), 2), sum(c["tokens"] for c in chunks)
    timing.update(gen_s=gen_s, tokens=tokens, tok_per_s=round(tokens / gen_s, 1) if gen_s else None,
                  total_s=round(_now() - t0, 1))
    meta["partial"] = False
    _write_json(summ / "generate.json", meta)
    word = {0: "OK", 2: "STOPPED"}.get(rc, "FAILED")
    pr, rp = meta["probe"], meta["replay"]
    probe_line = "not run" if pr["passed"] is None else (
        f"{'passed' if pr['passed'] else 'missed'} ({pr['distinct']} distinct of {len(pr['texts_sha256'])})")
    backend = ((meta.get("constrained") or {}).get("backend") or {}).get("backend")
    print(f"GENERATE_{word} {meta['generated']}/{meta['requested']} stopped={meta['stopped']} probe={probe_line} "
          f"replay={rp['same']}/{rp['n']} same (stopped={rp['stopped']}) tok/s={timing['tok_per_s']} backend={backend}",
          flush=True)
    return rc


def generate(a: argparse.Namespace, root: Path) -> int:
    """Steps 1-6 (module doc) for the code tree `root`, into a.run/summary."""
    t0 = _now()
    summ = Path(a.run) / "summary"
    summ.mkdir(parents=True, exist_ok=True)
    meta: dict = dict.fromkeys(S.FIELDS["generate.json"])
    meta.update(schema=S.GENERATE_SCHEMA, partial=True, generated=0,
                model={"repo": S.MODEL, "revision": S.MODEL_REVISION}, sampling=dict(J.SAMPLING),
                per_request_seed=True, engine={},
                probe={"seeds": list(S.PROBE_SEEDS), "cell": PROBE, "n": S.PROBE_N, "texts_sha256": None,
                       "distinct": None, "passed": None},
                replay={"n": 0, "same": 0, "differ": 0, "by_chunk": [], "stopped": "not run"},
                timing={"load_s": None, "probe_s": None, "chunks": [], "gen_s": None, "tokens": None,
                        "tok_per_s": None, "replay_s": None, "total_s": None},
                host=_host(root))
    plan, problems = prechecks(root, meta)
    if plan is None:
        return _finish(meta, summ, t0, 2, "pre-check: " + "; ".join(problems))
    _write_json(summ / "generate.json", meta)
    rc, held, engine, so = 0, {}, None, None
    try:
        so = J.structured_outputs_params(plan.regex)      # vLLM's class, imported before the model loads
        engine = J.load_engine(f"{S.MODEL}@{S.MODEL_REVISION}", seed=S.ENGINE_SEED, gpu_mem=S.GPU_MEM,
                               max_model_len=S.MAX_MODEL_LEN, backend=S.BACKEND)
        meta["timing"]["load_s"] = engine.load_s
        meta["constrained"]["backend"] = J.backend_record(engine.info)
        meta["engine"] = {k: v for k, v in engine.info.items() if k != "sampling_params"}
        _write_json(summ / "generate.json", meta)
        meta["probe"], meta["timing"]["probe_s"] = probe(engine, so, plan)
        _write_json(summ / "generate.json", meta)
        if not meta["probe"]["passed"]:
            return _finish(meta, summ, t0, 2, "seed probe")
        held = draw(engine, so, plan, a.chunk, a.deadline_s, meta, summ, t0)
    except Exception as err:  # noqa: BLE001: the run keeps every finished chunk and says why it stopped
        meta["stopped"] = f"error: {type(err).__name__}: {err}"[:STOPPED_CHARS]
        meta["trace"] = traceback.format_exc()[-TRACE_CHARS:]
        rc = 1
    if rc == 0 and meta["stopped"] is None and meta["generated"] == len(plan.reqs):
        # the draws are complete: say so before the replay, so a kill there (the watchdog's exit, TERM, scancel)
        # leaves a run that reads complete, and only the readout cut ("running")
        meta["partial"] = False
        meta["replay"]["stopped"] = "running"
        _write_json(summ / "generate.json", meta)
        replay(engine, so, plan, held, a.deadline_s, meta, summ, t0)
    else:
        meta["replay"]["stopped"] = "not run: the draws are not complete"
        rc = 1
    return _finish(meta, summ, t0, rc)


def main(a: argparse.Namespace) -> int:
    """The `generate` subcommand (lofi_rl.step0.cli), under junk's watchdog (module doc, step 7)."""
    hang = a.deadline_s + J.HANG_MARGIN_S if a.hang_s is None else a.hang_s
    if hang > 0:
        faulthandler.enable(all_threads=True)       # a hang dumps every thread's stack to the job log
        faulthandler.dump_traceback_later(hang, exit=True)
    rc = generate(a, ROOT)
    if hang > 0:
        faulthandler.cancel_dump_traceback_later()
        signal.alarm(120)                           # no handler: a hang in vLLM's teardown ends the process
    return rc
