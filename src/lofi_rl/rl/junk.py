"""Base-model junk rate: the Phase 1-2 progress oracle (CLAUDE.md §6) and the data of the Phase 1 gate (§9).

    python -m lofi_rl.rl.junk generate --run DIR [--n 500] [--seed 2026] [--model REPO@SHA]   policy env (vLLM)
    python -m lofi_rl.rl.junk score    --run DIR                                             policy env
    python -m lofi_rl.rl.junk render   --run DIR [--k 50]                                    scorer env

jobs/junk_rate.sbatch runs the three in order on one GPU node, with DIR = $REMOTE_SCR/runs/junk-<job>.

- generate: N briefs from brief.sample_briefs(seed, N), one completion each, with prompt v1 (lofi_rl.prompt),
  sampled as GRPO samples (taste_pipeline §6): temperature 1.0, top_p 1, top_k 0 (every token), max_tokens 512,
  no penalties. The model is Qwen3-4B-Instruct-2507 at the commit smoke.py pins, read from the HF cache
  (kit.hf_cache.resolve_pinned), in bf16 and eager mode as the Phase-0 smoke ran it. Prompts go to vLLM in chunks;
  each finished chunk is appended to completions.jsonl and fsynced, and generate.json is rewritten, so a kill keeps
  what was done. A chunk that would end past --deadline-s is not started.
- score: gates G1-G6 (dsl.parse with the palette's vocabulary and the brief's mode) and compile (compile_loop). No
  audio, no numpy, so it runs in the policy env. Writes scored.jsonl and junk.json.
- render: a seeded sample of --k gate-passing loops rendered with dev_gu (render.pipeline) to one-pass 16-bit
  FLACs under DIR/clips/, for a later ear check. G7 comes from these renders and is merged into junk.json.
  Nothing goes to the inbox.

Files: DIR/summary/{generate.json, completions.jsonl, scored.jsonl, junk.json, clips.json} (make pull brings
summary/ back) and DIR/clips/<i>.flac with <i>.manifest.json (they stay on scratch).

junk.json (schema lofi-rl/junk/1):
- pass: G1-G6 and compile: count, rate, Wilson 95 % CI. pass_not_copy: the same, with near-copies counted as fails;
- failures: each completion's first failing gate (the parser's precedence: G2, G5, G1, G4, G3, G6), with G5 under its
  own label, then compile (compile_loop raised) and crash (the parser raised). G7 is null until render merges the
  count from its sample;
- repairs: by code (loops, count, penalized), repairs per loop (all, passing), P_fix of passing loops (mean, hist);
- tokens: completion-length percentiles, truncations (finish_reason "length", which is G2), loops over 360 tokens
  (instrument.md §1's soft length penalty), and the prompt tokens vLLM counted;
- near_copy: character 5-gram Jaccard with the in-prompt example of at least 0.8. A loop that passes G1-G6 is
  compared by its canonical text without `sax:` (the form the example is printed in), any other by its raw text
  with fences, blank lines and extra spaces dropped;
- prompt: version, template sha, system-message sha, and how many rows were sent exactly today's prompt;
- by_knob: pass counts by energy and by mood. top_details: the commonest gate details, line numbers dropped.

No claim about sound: clips are only measured (length, loudness, hashes).
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import inspect
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
import traceback
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from lofi_rl.brief import ENERGIES, MOOD_NAMES, Brief, sample_briefs
from lofi_rl.compile.compiler import compile_loop
from lofi_rl.dsl import canon, canon_sha, parse
from lofi_rl.dsl.repair import penalty
from lofi_rl.palette import ROOT, Palette, load_palette
from lofi_rl.prng import MASK, splitmix64
from lofi_rl.prompt import (PROMPT_VERSION, build_messages, estimate_tokens, example_text, messages_sha256,
                            system_message, template_sha256)
from lofi_rl.smoke import QWEN

SCHEMA = "lofi-rl/junk/1"
GENERATE_SCHEMA = "lofi-rl/junk-generate/1"
CLIPS_SCHEMA = "lofi-rl/junk-clips/1"
DEFAULT_PALETTE = ROOT / "palettes" / "dev_gu.yaml"
DEFAULT_N, DEFAULT_SEED, DEFAULT_CHUNK, RENDER_K = 500, 2026, 128, 50
# taste_pipeline §6: GRPO's sampling, which make junk-rate shares; top_k 0 is "every token" in vLLM
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0, "max_tokens": 512}
NEAR_COPY, NGRAM = 0.8, 5
HANG_MARGIN_S = 90                      # generate's stack-dump watchdog: --deadline-s plus this
OVER_LENGTH = 360                       # instrument.md §1: the soft length penalty starts above 360 tokens
GATES = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "compile", "crash")
Z95 = 1.959963984540054
_SHA40 = re.compile(r"[0-9a-f]{40}")
_LINE = re.compile(r"^line \d+: ")


def _now() -> float:
    return time.monotonic()


# --------------------------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval for k successes in n trials (0..1 when n is 0)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (0.0 if k == 0 else max(0.0, centre - half), 1.0 if k == n else min(1.0, centre + half))


def percentiles(values: Iterable[float], qs: Iterable[int]) -> dict[str, float | None]:
    """Linear interpolation between order statistics (numpy's default), keyed "p<q>"; None for no values."""
    xs = sorted(values)
    out: dict[str, float | None] = {}
    for q in qs:
        if not xs:
            out[f"p{q}"] = None
            continue
        pos = (len(xs) - 1) * q / 100
        lo, hi = math.floor(pos), math.ceil(pos)
        out[f"p{q}"] = xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
    return out


def grams(text: str, n: int = NGRAM) -> frozenset[str]:
    """Character n-grams of a text; a text shorter than n is one gram, an empty text none."""
    if len(text) < n:
        return frozenset({text}) if text else frozenset()
    return frozenset(text[i:i + n] for i in range(len(text) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """|a & b| / |a | b|; 0 for two empty sets (two empty texts are not copies)."""
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def plain_text(text: str) -> str:
    """Raw text for the near-copy check of a loop that fails the gates: no fences, no blank lines, single spaces."""
    lines = [" ".join(line.split()) for line in text.splitlines() if not line.strip().startswith("```")]
    kept = [line for line in lines if line]
    return "\n".join(kept) + "\n" if kept else ""


def sample_indices(indices: Iterable[int], k: int, seed: int) -> list[int]:
    """k of `indices`, ranked by a keyed SplitMix64 hash: seeded, independent of input order, and a smaller k
    takes a prefix of a larger one."""
    s = splitmix64(seed & MASK)
    return sorted(set(indices), key=lambda i: (splitmix64(s ^ (i & MASK)), i))[:max(0, k)]


# --------------------------------------------------------------------------------------------- files
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def brief_dict(b: Brief) -> dict:
    return {"energy": b.energy, "mood": b.mood, "bpm": b.bpm, "tonic": b.tonic, "scene": b.scene}


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)                       # a kill mid-write never leaves a cut file


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_jsonl(path: Path) -> tuple[list[dict], int]:
    """The rows of a JSONL file, and how many lines could not be read (a line cut by a kill, say)."""
    rows, bad = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(row, dict) and isinstance(row.get("i"), int):
            rows.append(row)
        else:
            bad += 1
    return rows, bad


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _host() -> dict:
    stamp = ROOT / ".sync-stamp"
    info = {"node": platform.node(), "job": os.environ.get("SLURM_JOB_ID"),
            "stamp": stamp.read_text().strip() if stamp.is_file() else None, "gpu": None}
    if shutil.which("nvidia-smi"):              # never torch.cuda here: one CUDA process on the GPU (smoke.py)
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30)
            info["gpu"] = out.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            pass
    return info


# --------------------------------------------------------------------------------------------- generate
@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str | None
    n_tokens: int
    prompt_tokens: int


def split_model(spec: str) -> tuple[str, str]:
    """"org/name@<40-hex commit>" -> (repo, revision). A branch or a short sha is refused: runs pin a commit."""
    repo, sep, rev = spec.rpartition("@")
    if not sep or not repo or not _SHA40.fullmatch(rev):
        raise ValueError(f"model {spec!r}: give REPO@<40-hex commit sha>")
    return repo, rev


def _check_toolchain() -> None:
    """vLLM's model runner JIT-compiles Triton kernels: a C compiler and Python.h (the smoke's checks)."""
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    if not (cc and shutil.which(cc)):
        raise RuntimeError("vLLM's Triton kernels need $CC, gcc or clang")
    python_h = Path(sysconfig.get_paths()["include"]) / "Python.h"
    if not python_h.is_file():
        raise RuntimeError(f"Triton's driver build needs {python_h}")


class VllmEngine:
    """vLLM offline generation as smoke 13628361 ran it: bf16, eager, the engine core forked from a parent that
    never touches CUDA. Policy env only (vllm is imported here, not at module level)."""

    def __init__(self, path: str, *, seed: int, gpu_mem: float, max_model_len: int) -> None:
        _check_toolchain()
        import torch
        import transformers
        import vllm
        from vllm import LLM, SamplingParams
        t = _now()
        self.llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=gpu_mem, max_model_len=max_model_len,
                       enforce_eager=True, seed=seed)
        self.load_s = round(_now() - t, 1)
        self._params = SamplingParams
        self.info = {"engine": "vllm", "vllm": vllm.__version__, "torch": torch.__version__,
                     "transformers": transformers.__version__, "path": path, "dtype": "bfloat16",
                     "enforce_eager": True, "gpu_memory_utilization": gpu_mem, "max_model_len": max_model_len,
                     "seed": seed, "mp_method": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") or "fork"}
        try:
            self.info["max_num_seqs"] = self.llm.llm_engine.vllm_config.scheduler_config.max_num_seqs
        except AttributeError:
            pass
        self._chat_kw = {"use_tqdm": False} if "use_tqdm" in inspect.signature(self.llm.chat).parameters else {}

    def sampling(self, **kw):
        try:
            sp = self._params(n=1, **kw)
        except ValueError:
            if kw.get("top_k") != 0:
                raise
            sp = self._params(n=1, **{**kw, "top_k": -1})      # an older vLLM spells "every token" -1
        self.info["sampling_params"] = repr(sp)
        return sp

    def chat(self, convs: list[list[dict]], sp) -> list[Completion]:
        out = []
        for req in self.llm.chat(convs, sp, **self._chat_kw):
            c = req.outputs[0]
            out.append(Completion(text=c.text, finish_reason=c.finish_reason, n_tokens=len(c.token_ids),
                                  prompt_tokens=len(req.prompt_token_ids or [])))
        return out


def load_engine(model: str, *, seed: int, gpu_mem: float, max_model_len: int) -> VllmEngine:
    """The pinned snapshot from the HF cache (no huggingface_hub call), then vLLM on it."""
    from kit.hf_cache import resolve_pinned
    repo, rev = split_model(model)
    need = QWEN[2] if repo == QWEN[0] else ("config.json", "tokenizer_config.json")
    return VllmEngine(resolve_pinned(repo, rev, need), seed=seed, gpu_mem=gpu_mem, max_model_len=max_model_len)


def cmd_generate(a: argparse.Namespace) -> int:
    t0 = _now()
    summ = Path(a.run) / "summary"
    summ.mkdir(parents=True, exist_ok=True)
    palette = load_palette(a.palette, verify=False)             # names only: no audio in the policy env
    repo, rev = split_model(a.model)
    briefs = sample_briefs(a.seed, a.n)
    convs = [build_messages(b, palette) for b in briefs]
    shas = [messages_sha256(c) for c in convs]
    longest = max((c[0]["content"] + "\n" + c[1]["content"] for c in convs), key=len)
    sampling = dict(SAMPLING, max_tokens=a.max_tokens)
    meta = {"schema": GENERATE_SCHEMA, "partial": True, "requested": a.n, "generated": 0, "stopped": None,
            "briefs": {"seed": a.seed, "count": a.n, "sampler": "lofi_rl.brief.sample_briefs"},
            "model": {"repo": repo, "revision": rev},
            "prompt": {"version": PROMPT_VERSION, "template_sha256": template_sha256(),
                       "system_sha256": sha256_text(system_message(palette)), "chars": len(longest),
                       "tokens_estimate_4cpt": estimate_tokens(longest)},
            "palette": {"id": palette.id, "sha256": palette.sha256, "path": _rel(a.palette)},
            "sampling": sampling, "engine": {}, "timing": {"chunks": []}, "host": _host()}
    _write_json(summ / "generate.json", meta)
    chunks = meta["timing"]["chunks"]
    rc = 0
    with (summ / "completions.jsonl").open("w", encoding="utf-8") as f:
        try:
            engine = load_engine(a.model, seed=a.seed, gpu_mem=a.gpu_mem, max_model_len=a.max_model_len)
            meta["timing"]["load_s"] = engine.load_s
            sp = engine.sampling(**sampling)
            meta["engine"] = dict(engine.info)
            _write_json(summ / "generate.json", meta)
            for start in range(0, a.n, a.chunk):
                elapsed, longest = _now() - t0, max((c["s"] for c in chunks), default=0.0)
                if elapsed >= a.deadline_s or (chunks and elapsed + 1.25 * longest > a.deadline_s):
                    meta["stopped"] = "deadline"
                    break
                part = convs[start:start + a.chunk]
                t = _now()
                outs = engine.chat(part, sp)
                dt = _now() - t
                if len(outs) != len(part):
                    raise RuntimeError(f"the engine returned {len(outs)} completions for {len(part)} prompts")
                for k, c in enumerate(outs):
                    i = start + k
                    f.write(json.dumps({"i": i, "brief": brief_dict(briefs[i]), "prompt_sha256": shas[i],
                                        "text": c.text, "finish_reason": c.finish_reason, "n_tokens": c.n_tokens,
                                        "prompt_tokens": c.prompt_tokens}, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                chunks.append({"start": start, "count": len(outs), "s": round(dt, 2),
                               "tokens": sum(c.n_tokens for c in outs)})
                meta["generated"] += len(outs)
                _write_json(summ / "generate.json", meta)
                print(f"[junk:generate] {meta['generated']}/{a.n} in {_now() - t0:.0f} s", flush=True)
        except Exception as err:  # noqa: BLE001: the run keeps every finished chunk and says why it stopped
            meta["stopped"] = f"error: {type(err).__name__}: {err}"[:2000]
            meta["trace"] = traceback.format_exc()[-6000:]
            rc = 1
    gen_s, tokens = sum(c["s"] for c in chunks), sum(c["tokens"] for c in chunks)
    meta["timing"].update(gen_s=round(gen_s, 2), tokens=tokens, total_s=round(_now() - t0, 1),
                          tok_per_s=round(tokens / gen_s, 1) if gen_s else None)
    meta["partial"] = False
    _write_json(summ / "generate.json", meta)
    print(f"GENERATE_{'OK' if rc == 0 and meta['generated'] else 'FAILED'} {meta['generated']}/{a.n} "
          f"stopped={meta['stopped']} tok/s={meta['timing']['tok_per_s']}", flush=True)
    return rc if meta["generated"] else 1


# --------------------------------------------------------------------------------------------- score
def score_row(row: dict, palette: Palette, example_grams: frozenset[str]) -> dict:
    """One completion through the gates and the compiler, plus its near-copy Jaccard with the in-prompt example."""
    brief = Brief(**row["brief"])
    out = {"i": row["i"], "energy": brief.energy, "mood": brief.mood, "n_tokens": row.get("n_tokens"),
           "finish_reason": row.get("finish_reason"), "truncated": row.get("finish_reason") == "length"}
    repairs, copy_text = [], None
    try:
        res = parse(row["text"], vocab=palette.vocab(), finish_reason=row.get("finish_reason"), mode=brief.mode)
    except Exception as err:  # noqa: BLE001: a parser crash on model output is counted, never fatal
        gate, detail = "crash", f"{type(err).__name__}: {err}"
    else:
        gate, detail, repairs = res.gate.code, res.gate.detail, list(res.repairs)
        if res.gate.ok:
            copy_text = canon(replace(res.ast, sax=None))
            out["canon_sha256"] = canon_sha(res.ast)
            try:
                repairs += list(compile_loop(res.ast, brief, palette).repairs)
            except Exception as err:  # noqa: BLE001: CompileError, or a compiler bug on model output
                gate, detail = "compile", f"{type(err).__name__}: {err}"
    j = jaccard(grams(copy_text if copy_text is not None else plain_text(row["text"])), example_grams)
    out.update(ok=gate is None, gate=gate, detail=detail[:300],
               repairs=dict(sorted(Counter(r.code for r in repairs).items())),
               penalized_codes=sorted({r.code for r in repairs if r.penalized}), p_fix=penalty(repairs),
               copy_jaccard=round(j, 4), near_copy=j >= NEAR_COPY)
    return out


def _rounded(values: dict[str, float | None], digits: int) -> dict[str, float | None]:
    return {k: None if v is None else round(v, digits) for k, v in values.items()}


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"count": k, "rate": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)],
            "method": "wilson"}


def _knobs(scored: list[dict], field: str, values: tuple[str, ...]) -> dict:
    return {v: {"n": sum(1 for s in scored if s[field] == v), "pass": sum(1 for s in scored if s[field] == v and s["ok"])}
            for v in values}


def render_block(clips: dict) -> dict:
    """What the rendered sample says: G7 failures, other gates, infrastructure errors, deadline skips."""
    recs = clips.get("clips", [])
    skipped = sum(1 for r in recs if r.get("skipped"))
    return {"attempted": len(recs) - skipped, "rendered": sum(1 for r in recs if r.get("ok")),
            "G7": sum(1 for r in recs if r.get("gate") == "G7"),
            "other_gates": sum(1 for r in recs if r.get("gate") not in (None, "G7")),
            "errors": sum(1 for r in recs if r.get("error")), "skipped": skipped,
            "seed": clips.get("seed"), "k": clips.get("k")}


def summarize(rows: list[dict], scored: list[dict], meta: dict, palette: Palette, *, unreadable: int = 0,
              run: str = "", clips: dict | None = None) -> dict:
    n, ok = len(scored), [s for s in scored if s["ok"]]
    requested = int(meta.get("requested", n))
    fails = Counter(s["gate"] for s in scored if not s["ok"])
    failures = {g: fails.get(g, 0) for g in GATES}
    render = render_block(clips) if clips else None
    failures["G7"] = render["G7"] if render else None

    by_code: dict[str, dict] = {}
    for s in scored:
        for code, count in s["repairs"].items():
            e = by_code.setdefault(code, {"loops": 0, "count": 0, "penalized": False})
            e["loops"] += 1
            e["count"] += count
            e["penalized"] = e["penalized"] or code in s["penalized_codes"]

    def per_loop(group: list[dict]) -> dict[str, int]:
        hist = Counter(sum(s["repairs"].values()) for s in group)
        return {str(k): hist[k] for k in sorted(hist)}
    p_fix = [s["p_fix"] for s in ok]
    p_hist = Counter(f"{p:.2f}" for p in p_fix)

    n_tok = [s["n_tokens"] for s in scored if isinstance(s.get("n_tokens"), int)]
    p_tok = [r["prompt_tokens"] for r in rows if isinstance(r.get("prompt_tokens"), int)]
    tokens = {"completion": {**_rounded(percentiles(n_tok, (50, 90, 95, 99)), 2), "max": max(n_tok, default=None),
                             "mean": round(sum(n_tok) / len(n_tok), 2) if n_tok else None},
              "truncated": sum(1 for s in scored if s["truncated"]),
              f"over_{OVER_LENGTH}": sum(1 for t in n_tok if t > OVER_LENGTH),
              "max_tokens": meta.get("sampling", {}).get("max_tokens", SAMPLING["max_tokens"]),
              "prompt": {"min": min(p_tok, default=None), "max": max(p_tok, default=None)}}

    copies = sum(1 for s in scored if s["near_copy"])
    copies_ok = sum(1 for s in ok if s["near_copy"])
    current = {}
    matching = 0
    for r in rows:
        key = (r["brief"]["energy"], r["brief"]["mood"])
        if key not in current:
            current[key] = messages_sha256(build_messages(Brief(**r["brief"]), palette))
        matching += r.get("prompt_sha256") == current[key]

    details: dict[str, Counter] = {}
    for s in scored:
        if not s["ok"]:
            details.setdefault(s["gate"], Counter())[_LINE.sub("", s["detail"])] += 1
    gen_palette = meta.get("palette", {}).get("sha256")
    return {
        "schema": SCHEMA, "run": run, "n": n, "requested": requested,
        "partial": n < requested or bool(meta.get("partial")), "unreadable_lines": unreadable,
        "pass": {**_rate(len(ok), n), "gates": "G1-G6 and compile on every completion; G7 on the render sample"},
        "pass_not_copy": _rate(sum(1 for s in ok if not s["near_copy"]), n),
        "failures": failures,
        "render": render,
        "repairs": {"by_code": dict(sorted(by_code.items())),
                    "per_loop": {"all": per_loop(scored), "passing": per_loop(ok)},
                    "p_fix": {"passing": {"mean": round(sum(p_fix) / len(p_fix), 6) if p_fix else None,
                                          "hist": dict(sorted(p_hist.items()))}}},
        "tokens": tokens,
        "near_copy": {"threshold": NEAR_COPY, "ngram": NGRAM, "count": copies,
                      "rate": round(copies / n, 4) if n else None, "count_passing": copies_ok,
                      "rate_of_passing": round(copies_ok / len(ok), 4) if ok else None,
                      "jaccard": {**_rounded(percentiles([s["copy_jaccard"] for s in scored], (50, 90, 99)), 4),
                                  "max": max((s["copy_jaccard"] for s in scored), default=None)}},
        "prompt": {"version": PROMPT_VERSION, "template_sha256": template_sha256(),
                   "system_sha256": sha256_text(system_message(palette)), "rows_matching": matching,
                   "as_generated": meta.get("prompt")},
        "palette": {"id": palette.id, "sha256": palette.sha256,
                    "matches_generate": None if gen_palette is None else gen_palette == palette.sha256},
        "by_knob": {"energy": _knobs(scored, "energy", ENERGIES), "mood": _knobs(scored, "mood", MOOD_NAMES)},
        "top_details": {g: [[d, c] for d, c in counter.most_common(8)] for g, counter in sorted(details.items())},
        "briefs": meta.get("briefs"), "sampling": meta.get("sampling"),
        "generate": {k: meta.get(k) for k in ("partial", "requested", "generated", "stopped", "model", "engine",
                                              "timing", "host")},
    }


def headline(s: dict) -> str:
    p, f = s["pass"], s["failures"]
    lo, hi = p["ci95"]
    rate = f"{100 * p['rate']:.1f}%" if p["rate"] is not None else "-"
    gates = " ".join(f"{g} {f[g]}" for g in GATES if f[g])
    return (f"JUNK n={s['n']}/{s['requested']} pass {p['count']} = {rate} [{100 * lo:.1f}, {100 * hi:.1f}] | "
            f"near-copies {s['near_copy']['count']} (pass, not a copy: {s['pass_not_copy']['count']}) | "
            f"{gates or 'no failures'} | truncated {s['tokens']['truncated']}")


def cmd_score(a: argparse.Namespace) -> int:
    summ = Path(a.run) / "summary"
    comp = summ / "completions.jsonl"
    if not comp.is_file():
        print(f"score: {comp} is missing (run generate first)", file=sys.stderr)
        return 2
    rows, bad = read_jsonl(comp)
    rows.sort(key=lambda r: r["i"])
    meta = _read_json(summ / "generate.json") or {}
    palette = load_palette(a.palette, verify=False)
    example = grams(example_text(palette))
    scored = [score_row(r, palette, example) for r in rows]
    _write_jsonl(summ / "scored.jsonl", scored)
    summary = summarize(rows, scored, meta, palette, unreadable=bad, run=Path(a.run).name,
                        clips=_read_json(summ / "clips.json"))
    _write_json(summ / "junk.json", summary)
    print(headline(summary), flush=True)
    return 0


# --------------------------------------------------------------------------------------------- render
def render_clip(row: dict, palette: Palette, clips_dir: Path, *, cache: Path | None, workers: int) -> dict:
    """Render one loop with the palette to clips/<i>.flac (one pass, 16-bit) and its manifest. Scorer env."""
    from lofi_rl.compile.compiler import CompileError
    from lofi_rl.render.formats import write_flac
    from lofi_rl.render.pipeline import RenderError, manifest_json, render
    name, t = f"{row['i']:04d}", _now()
    try:
        r = render(row["text"], Brief(**row["brief"]), palette, cache=cache, workers=workers,
                   finish_reason=row.get("finish_reason"))
    except RenderError as err:
        return {"i": row["i"], "ok": False, "gate": err.code, "detail": err.detail[:300]}
    except CompileError as err:
        return {"i": row["i"], "ok": False, "gate": "compile", "detail": str(err)[:300]}
    except Exception as err:  # noqa: BLE001: FluidSynth, files: infrastructure, never a gate
        return {"i": row["i"], "ok": False, "gate": None, "error": f"{type(err).__name__}: {err}"[:500]}
    flac = write_flac(clips_dir / f"{name}.flac", r.audio)
    (clips_dir / f"{name}.manifest.json").write_text(manifest_json(r.manifest), encoding="utf-8")
    m = r.manifest
    return {"i": row["i"], "ok": True, "gate": None, "clip": f"clips/{name}.flac", "n": m["n"],
            "seconds": round(m["seconds"], 3), "lufs": round(m["loudness"]["lufs"], 3),
            "true_peak_dbtp": round(m["loudness"]["true_peak_dbtp"], 3), "audio_sha256": m["audio_sha256"],
            "canon_sha256": m["loop"]["canon_sha256"], "p_fix": m["loop"]["p_fix"],
            "render_s": round(_now() - t, 2), "flac_bytes": flac.stat().st_size}


def cmd_render(a: argparse.Namespace) -> int:
    run = Path(a.run)
    summ = run / "summary"
    if not (summ / "scored.jsonl").is_file():
        print(f"render: {summ / 'scored.jsonl'} is missing (run score first)", file=sys.stderr)
        return 2
    scored = {s["i"]: s for s in read_jsonl(summ / "scored.jsonl")[0]}
    rows = {r["i"]: r for r in read_jsonl(summ / "completions.jsonl")[0]}
    meta = _read_json(summ / "generate.json") or {}
    seed = a.seed if a.seed is not None else int((meta.get("briefs") or {}).get("seed", DEFAULT_SEED))
    passing = [i for i, s in scored.items() if s["ok"] and i in rows]
    pick = sample_indices(passing, a.k, seed)
    palette = load_palette(a.palette, verify=not a.no_verify)
    clips_dir = run / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips = {"schema": CLIPS_SCHEMA, "seed": seed, "k": a.k, "passing": len(passing),
             "palette": {"id": palette.id, "sha256": palette.sha256}, "clips": []}
    t0 = _now()
    for i in pick:
        if _now() - t0 >= a.deadline_s:
            rec = {"i": i, "ok": False, "gate": None, "skipped": "render deadline"}
        else:
            rec = render_clip(rows[i], palette, clips_dir, cache=a.cache, workers=a.workers)
        rec.update(brief=rows[i]["brief"], copy_jaccard=scored[i]["copy_jaccard"], near_copy=scored[i]["near_copy"])
        clips["clips"].append(rec)
        _write_json(summ / "clips.json", clips)
        print(f"[junk:render] {len(clips['clips'])}/{len(pick)} i={i} "
              f"{'ok' if rec['ok'] else rec.get('gate') or rec.get('error') or rec.get('skipped')}", flush=True)
    _write_json(summ / "clips.json", clips)
    block = render_block(clips)
    summary = _read_json(summ / "junk.json")
    if summary is not None:
        summary["render"], summary["failures"]["G7"] = block, block["G7"]
        _write_json(summ / "junk.json", summary)
    print(f"RENDER rendered {block['rendered']}/{len(pick)} G7 {block['G7']} other gates {block['other_gates']} "
          f"errors {block['errors']} skipped {block['skipped']}", flush=True)
    return 1 if block["errors"] or (pick and not block["rendered"] and not block["G7"]) else 0


# --------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lofi_rl.rl.junk", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="N completions with vLLM (policy env)")
    g.add_argument("--run", type=Path, required=True)
    g.add_argument("--n", type=int, default=DEFAULT_N)
    g.add_argument("--seed", type=int, default=DEFAULT_SEED, help="brief stream seed, also vLLM's seed")
    g.add_argument("--palette", type=Path, default=DEFAULT_PALETTE)
    g.add_argument("--model", default=f"{QWEN[0]}@{QWEN[1]}", help="REPO@<40-hex commit>, from the HF cache")
    g.add_argument("--max-tokens", type=int, default=SAMPLING["max_tokens"])
    g.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help="prompts per vLLM call")
    g.add_argument("--deadline-s", type=float, default=720.0, help="no chunk starts that would end after this")
    g.add_argument("--hang-s", type=float, default=None,
                   help="dump stacks and exit after this many seconds (default deadline + 90; 0 disables)")
    g.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    g.add_argument("--max-model-len", type=int, default=4096)
    s = sub.add_parser("score", help="gates G1-G6 and compile -> summary/junk.json (policy env)")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--palette", type=Path, default=DEFAULT_PALETTE)
    r = sub.add_parser("render", help="a seeded sample of passing loops to FLAC (scorer env)")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--palette", type=Path, default=DEFAULT_PALETTE)
    r.add_argument("--k", type=int, default=RENDER_K)
    r.add_argument("--seed", type=int, default=None, help="default: the brief seed of generate.json")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--deadline-s", type=float, default=360.0, help="no render starts after this")
    r.add_argument("--cache", type=Path, default=None, help="one-shot cache (default $LOFI_CACHE)")
    r.add_argument("--no-verify", action="store_true", help="skip hashing the palette's source files")
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        if a.n < 1 or a.chunk < 1 or a.max_tokens < 1:
            ap.error("--n, --chunk and --max-tokens must be at least 1")
        hang = a.deadline_s + HANG_MARGIN_S if a.hang_s is None else a.hang_s
        if hang > 0:
            faulthandler.enable(all_threads=True)       # a hang dumps every thread's stack to the job log
            faulthandler.dump_traceback_later(hang, exit=True)
        rc = cmd_generate(a)
        if hang > 0:
            faulthandler.cancel_dump_traceback_later()
            signal.alarm(120)                           # no handler: a hang in vLLM's teardown ends the process
        return rc
    return {"score": cmd_score, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
