"""Base-model junk rate: the Phase 1-2 progress oracle (CLAUDE.md §6) and the data of the Phase 1 gate (§9).

    python -m lofi_rl.rl.junk generate --run DIR [--n 500] [--seed 2026] [--model REPO@SHA] [--constrained]
                                       [--prompt v1|v2]                                      policy env (vLLM)
    python -m lofi_rl.rl.junk score    --run DIR                                             policy env
    python -m lofi_rl.rl.junk render   --run DIR [--k 50] [--workers 4] [--jobs 2]           scorer env

jobs/junk_rate.sbatch runs the three in order on one GPU node, with DIR on the node's own disk, then copies it to
$REMOTE_KEEP/runs/junk-<job> (D-022, amended 2026-11-06: scratch writes stalled the first gate launch).

- generate: N briefs from brief.sample_briefs(seed, N), one completion each, with prompt v1 (lofi_rl.prompt, the
  default) or, with --prompt v2, prompt v2 (lofi_rl.prompt_v2: v1 with its one example chosen per brief, the loop of
  the brief's energy and the next mood). Every message is built before the model loads, so a palette the prompt
  cannot serve (PromptError) stops generate before any GPU work, with the error in generate.json. Each row records
  its prompt_version and the id of the example its prompt showed. The completions are
  sampled as GRPO samples (taste_pipeline §6): temperature 1.0, top_p 1, top_k 0 (every token), max_tokens 512,
  no penalties. The model is Qwen3-4B-Instruct-2507 at the commit smoke.py pins, read from the HF cache
  (kit.hf_cache.resolve_pinned), in bf16 and eager mode as the Phase-0 smoke ran it. Prompts go to vLLM in chunks;
  each finished chunk is appended to completions.jsonl and fsynced, and generate.json is rewritten, so a kill keeps
  what was done. A chunk that would end past --deadline-s is not started.
  --constrained adds grammar-constrained decoding (D-016; docs/GRAMMAR.md): SamplingParams(structured_outputs=
  StructuredOutputsParams(regex=lofi_rl.grammar.loopscript_regex(palette))). vLLM's class is imported inside the run,
  before the model loads, so a vLLM without it fails in seconds with the import named. LLM() is asked for the
  xgrammar backend (structured_outputs_config), so a regex xgrammar refuses stops the run instead of falling back; a
  vLLM whose LLM() refuses the keyword (TypeError) is built without it, and the record says the pin was not taken.
  generate.json and junk.json record under "constrained" the regex's sha256 and length and the backend: the pin, and
  what LLM() was built with, read back from it (BACKEND_PATHS; "unknown" when none resolves). The sampling settings
  stay v0's.
- score: gates G1-G6 (dsl.parse with the palette's vocabulary and the brief's mode) and compile (compile_loop). No
  audio, no numpy, so it runs in the policy env. Writes scored.jsonl and junk.json. The prompt version is
  generate.json's (v1 when it names none; one this file does not know exits 2), and each row is compared with the
  example its own prompt showed. For a constrained run it also counts the completions in the recorded regex's
  language, by finish_reason (stop, length, other).
- render: a seeded sample of --k gate-passing loops that are not near-copies (D-020), rendered with the palette
  (render.pipeline) to one-pass 16-bit FLACs under DIR/clips/, for a later ear check. G7 comes from these renders
  and is merged into junk.json. Nothing goes to the inbox. The renders run --jobs at a time (default 2), each with
  --workers // --jobs workers (the job passes its 4 CPUs, so 2 each), as drafts/render.py renders: the freeze spec
  §5.5 (iii), D-022. A render never depends on its workers (render.pipeline.render), and the records are kept in the
  sample's order, so the clips, their manifests and clips.json's order are those of one-at-a-time rendering (--jobs
  1); only render_s differs. No render starts after --deadline-s, in either order.

The palette: every step reads --palette, by default the frozen palettes/lofi_rain_v1.yaml (D-017; D-022), whose
names, grammar and banks the Phase-1 gate runs on. Generate, score and render must read the same file: score records
its sha256 and whether it matches generate's. D-019's and D-020's runs were made on palettes/dev_gu.yaml (sha256
b2e4a340...), their junk.json records it, and a re-score of one of them passes --palette palettes/dev_gu.yaml.

Files: DIR/summary/{generate.json, completions.jsonl, scored.jsonl, junk.json, clips.json} (make pull brings
summary/ back) and DIR/clips/<i>.flac with <i>.manifest.json (they stay on the cluster, under KEEP).

junk.json (schema lofi-rl/junk/1):
- pass: G1-G6 and compile: count, rate, Wilson 95 % CI. pass_not_copy: the same, with near-copies counted as fails;
- failures: each completion's first failing gate (the parser's precedence: G2, G5, G1, G4, G3, G6), with G5 under its
  own label, then compile (compile_loop raised) and crash (the parser raised). G7 is null until render merges the
  count from its sample;
- repairs: by code (loops, count, penalized), repairs per loop (all, passing), P_fix of passing loops (mean, hist);
- tokens: completion-length percentiles, truncations (finish_reason "length", which is G2), loops over 360 tokens
  (instrument.md §1's soft length penalty), and the prompt tokens vLLM counted;
- near_copy: character 5-gram Jaccard of at least 0.8 with the example shown in the row's own prompt (v1: its one
  example; v2: the brief's, prompt_v2.shown_examples). A loop that passes G1-G6 is compared by its canonical text
  without `sax:` (the form the examples are printed in), any other by its raw text with fences, blank lines and
  extra spaces dropped. `references` counts the rows compared with each example. pass_not_copy is D-019's gated
  number under this rule;
- prompt: version, module, template sha (and whether it is the one generate recorded), the system-message sha (v1)
  or one per energy x mood cell (systems), how many rows were sent exactly today's messages for their brief, and
  how many rows record an example other than the rule's;
- copy: readouts, none of them a gate. any: J5 >= 0.8 with any example prompt v2 can show (all 9, v1's among them).
  quasi_copy: containment, the share of a completion's 5-grams found in the shown example, of at least 0.85, and
  pass_not_quasi_copy (pass, neither a near- nor a quasi-copy, with its CI); then, over the passing loops: the
  containment percentiles, a histogram of copy_jaccard, the vamp/keys/A/B lines equal to the shown example's, the
  share of 16-cell grids equal to a grid of the same lane in it, the forms (span 1, bass lock, snare, ohat and
  shaker bases, edits per loop) and the knob readouts (verify.scoring: the energy band match and the mood
  signature, with the provisional ENERGY_BANDS). A knob readout that raises is recorded as null and never stops
  score;
- by_cell: per energy x mood cell, the example shown and the copy, pass and knob counts;
- constrained: generate.json's record of the decoding grammar (enabled, regex sha256 and length, builder, backend:
  requested, pin_accepted, backend, source, applied), plus in_language: the regex rebuilt from the palette, whether its sha256
  is the recorded one, and for "stop", "length" and "other" completions n, in_language and out_of_language (counted
  only when the sha256 matches; null when unconstrained). A stop completion out of the language means the engine did
  not apply the regex. The block is null for a run whose generate.json predates --constrained (v0, unconstrained);
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
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from lofi_rl import prompt, prompt_v2
from lofi_rl.brief import ENERGIES, MOOD_NAMES, Brief, sample_briefs
from lofi_rl.compile.compiler import compile_loop
from lofi_rl.dsl import canon, canon_sha, parse
from lofi_rl.dsl.repair import penalty
from lofi_rl.dsl.vocab import EDIT_TARGETS
from lofi_rl.grammar import loopscript_regex
from lofi_rl.palette import ROOT, Palette, load_palette
from lofi_rl.prng import MASK, splitmix64
from lofi_rl.prompt import PromptError, estimate_tokens, messages_sha256
from lofi_rl.smoke import QWEN

SCHEMA = "lofi-rl/junk/1"
GENERATE_SCHEMA = "lofi-rl/junk-generate/1"
CLIPS_SCHEMA = "lofi-rl/junk-clips/1"
DEFAULT_PALETTE = ROOT / "palettes" / "lofi_rain_v1.yaml"     # frozen 2026-11-04 (D-017, D-022); D-020 ran on dev_gu
DEFAULT_N, DEFAULT_SEED, DEFAULT_CHUNK, RENDER_K = 500, 2026, 128, 50
# renders at a time in the render step, --workers split between them (the freeze spec §5.5 (iii), D-022): job B's
# slowest one-at-a-time render (17.39 s, runs/drafts-13706355) puts 50 renders at 869.5 s; two at a time with 2
# workers each ran 1.713 times as fast on the laptop on the 42 drafts and 9 references, and 1.765 times on 50 of
# D-020 arm B's loops, with every clip, manifest and FLAC identical (docs/evidence/D-022)
RENDER_JOBS = 2
# taste_pipeline §6: GRPO's sampling, which make junk-rate shares; top_k 0 is "every token" in vLLM
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0, "max_tokens": 512}
PROMPTS = MappingProxyType({"v1": prompt, "v2": prompt_v2})      # the template versions, side by side
DEFAULT_PROMPT = "v1"
NEAR_COPY, NGRAM = 0.8, 5
NEAR_COPY_RULE = "J5 >= 0.8 with the example shown in the row's own prompt"
ANY_RULE = "J5 >= 0.8 with any example prompt v2 can show (prompt_v2.reference_texts)"
# runs/junk-13632676: v0's 122 near-copies hold >= 0.8768 of their 5-grams in v1's example, and no v0 completion
# holds more than 0.5748 in a shootout loop it never saw
QUASI_COPY = 0.85
QUASI_RULE = "containment |G5(x) & G5(shown)| / |G5(x)| >= 0.85"
JACCARD_EDGES = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)                 # left-closed bins of copy_jaccard
JACCARD_BINS = ("<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.75", "0.75-0.8", "0.8-0.9", ">=0.9")
GRID_LANES = EDIT_TARGETS               # comp, bass and the six drum lanes: the lanes a 16-cell grid belongs to
VERBATIM = ("vamp", "keys", "A", "B")
FORMS = ("span1", "lock", "snare", "ohat", "shaker")
CELL_BRIEF = {"bpm": 82, "tonic": 0}    # a cell's stand-in brief: the messages read only scene, energy and mood
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


# --------------------------------------------------------------------------------------------- prompts
def cell_brief(energy: str, mood: str) -> Brief:
    """A stand-in brief for one energy x mood cell: tempo and tonic never reach the messages (tests/test_prompt*.py)."""
    return Brief(energy=energy, mood=mood, **CELL_BRIEF)


def example_id(version: str, brief: Brief) -> str:
    """The id of the example a brief's prompt shows: v1's one example, or v2's rule."""
    return "example" if version == "v1" else PROMPTS[version].example_id(brief)


def example_cell(version: str, brief: Brief) -> tuple[str, str]:
    """(energy, mood) of the brief the shown example was written for."""
    if version == "v1":
        return prompt.EXAMPLE_BRIEF["energy"], prompt.EXAMPLE_BRIEF["mood"]
    return PROMPTS[version].example_cell(brief)


def shown_examples(version: str, brief: Brief, palette: Palette) -> tuple[tuple[str, str], ...]:
    """(id, text) of every complete loop the brief's prompt shows: what its near-copy check compares with."""
    if version == "v1":
        return (("example", prompt.example_text(palette)),)
    return PROMPTS[version].shown_examples(brief, palette)


def system_text(version: str, brief: Brief, palette: Palette) -> str:
    return prompt.system_message(palette) if version == "v1" else PROMPTS[version].system_message(palette, brief)


def prompt_record(version: str, palette: Palette, convs: list[list[dict]]) -> dict:
    """generate.json's "prompt": the version, its module and template sha, v1's one system sha (None for v2), and per
    energy x mood cell the example shown, its own cell and the shas of the example, the system message and the
    messages; chars and the 4-characters estimate are the longest prompt the run sends."""
    mod = PROMPTS[version]
    cells = {}
    for e in ENERGIES:
        for m in MOOD_NAMES:
            b = cell_brief(e, m)
            ((ex_id, ex_text),) = shown_examples(version, b, palette)
            cells[f"{e}/{m}"] = {"example": ex_id, "example_cell": "/".join(example_cell(version, b)),
                                 "example_sha256": sha256_text(ex_text),
                                 "system_sha256": sha256_text(system_text(version, b, palette)),
                                 "messages_sha256": messages_sha256(mod.build_messages(b, palette))}
    longest = max((c[0]["content"] + "\n" + c[1]["content"] for c in convs), key=len)
    return {"version": version, "module": mod.__name__, "template_sha256": mod.template_sha256(),
            "system_sha256": sha256_text(prompt.system_message(palette)) if version == "v1" else None,
            "cells": cells, "chars": len(longest), "tokens_estimate_4cpt": estimate_tokens(longest)}


def _sync_stamp() -> str | None:
    """The mirror's .sync-stamp as this process sees it (each step of the job is its own process)."""
    stamp = ROOT / ".sync-stamp"
    return stamp.read_text().strip() if stamp.is_file() else None


def _host() -> dict:
    info = {"node": platform.node(), "job": os.environ.get("SLURM_JOB_ID"), "stamp": _sync_stamp(), "gpu": None}
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


STRUCTURED_BACKEND = "xgrammar"         # pinned for --constrained, so a regex xgrammar refuses stops the run loudly
# Where a built LLM keeps its structured-outputs backend, newest first: VllmConfig.structured_outputs_config, then
# the older DecodingConfig names, then a V0 engine's own configs. UNVERIFIED for vLLM 0.32.0 (not on the laptop).
BACKEND_PATHS = (
    "llm_engine.vllm_config.structured_outputs_config.backend",
    "llm_engine.vllm_config.decoding_config.backend",
    "llm_engine.vllm_config.decoding_config.guided_decoding_backend",
    "llm_engine.structured_outputs_config.backend",
    "llm_engine.decoding_config.backend",
    "llm_engine.decoding_config.guided_decoding_backend",
)


def read_backend(llm) -> dict:
    """The structured-outputs backend a built LLM is configured with: the first of BACKEND_PATHS that resolves, as
    text, and the path it came from; {"backend": "unknown", "source": None} when none does. Never raises (a record,
    not a check). "auto" means vLLM picks per request: xgrammar, else guidance."""
    for path in BACKEND_PATHS:
        obj = llm
        try:
            for name in path.split("."):
                obj = getattr(obj, name)
        except Exception:  # noqa: BLE001: a missing attribute, or a property that raises
            continue
        if obj is not None:
            return {"backend": str(getattr(obj, "value", obj)), "source": path}     # .value: an enum in older vLLMs
    return {"backend": "unknown", "source": None}


def build_llm(llm_cls, kw: dict, backend: str | None):
    """llm_cls(**kw), asking for the structured-outputs `backend` (structured_outputs_config={"backend": ...}) when
    one is given. On a TypeError, a vLLM whose LLM() does not take that keyword, it is built again without it and the
    pin is recorded as not accepted, so an unknown keyword never costs the run. Returns (llm, pin record)."""
    if backend is None:
        return llm_cls(**kw), {"requested": None, "pin_accepted": None}
    try:
        return llm_cls(**kw, structured_outputs_config={"backend": backend}), {"requested": backend,
                                                                               "pin_accepted": True}
    except TypeError as err:
        return llm_cls(**kw), {"requested": backend, "pin_accepted": False, "pin_error": f"TypeError: {err}"[:300]}


class VllmEngine:
    """vLLM offline generation as smoke 13628361 ran it: bf16, eager, the engine core forked from a parent that
    never touches CUDA. Policy env only (vllm is imported here, not at module level). `backend` pins the
    structured-outputs backend (build_llm); info["structured_outputs"] records the pin, what LLM() was built with
    (read_backend), and `applied`: whether that is the backend asked for (None when nothing was asked for or nothing
    could be read)."""

    def __init__(self, path: str, *, seed: int, gpu_mem: float, max_model_len: int, backend: str | None = None) -> None:
        _check_toolchain()
        import torch
        import transformers
        import vllm
        from vllm import LLM, SamplingParams
        t = _now()
        self.llm, pin = build_llm(LLM, {"model": path, "dtype": "bfloat16", "gpu_memory_utilization": gpu_mem,
                                        "max_model_len": max_model_len, "enforce_eager": True, "seed": seed}, backend)
        self.load_s = round(_now() - t, 1)
        self._params = SamplingParams
        self.info = {"engine": "vllm", "vllm": vllm.__version__, "torch": torch.__version__,
                     "transformers": transformers.__version__, "path": path, "dtype": "bfloat16",
                     "enforce_eager": True, "gpu_memory_utilization": gpu_mem, "max_model_len": max_model_len,
                     "seed": seed, "mp_method": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") or "fork"}
        so = {**pin, **read_backend(self.llm)}
        so["applied"] = None if backend is None or so["backend"] == "unknown" else so["backend"] == backend
        self.info["structured_outputs"] = so
        try:                                    # informational only: never a reason to lose the run
            self.info["max_num_seqs"] = self.llm.llm_engine.vllm_config.scheduler_config.max_num_seqs
        except Exception:  # noqa: BLE001
            pass
        try:
            quiet = "use_tqdm" in inspect.signature(self.llm.chat).parameters
        except (TypeError, ValueError):
            quiet = False
        self._chat_kw = {"use_tqdm": False} if quiet else {}

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


STRUCTURED_IMPORT = "from vllm.sampling_params import StructuredOutputsParams"
REGEX_BUILDER = "lofi_rl.grammar.loopscript_regex"


def structured_outputs_params(regex: str):
    """vLLM's StructuredOutputsParams(regex=...), for SamplingParams(structured_outputs=...): the field v0's
    generate.json shows as `structured_outputs=None`. Imported here, inside the run (policy env only)."""
    try:
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as err:
        raise RuntimeError(f"--constrained: `{STRUCTURED_IMPORT}` failed ({type(err).__name__}: {err}), so the "
                           "model was not loaded and nothing was generated") from err
    return StructuredOutputsParams(regex=regex)


def constrained_block(regex: str | None) -> dict:
    """What generate.json (and junk.json after it) records about grammar-constrained decoding: never the regex
    itself, only its sha256 and length (None when the run is unconstrained). `backend` is filled in once the engine
    is built (backend_record)."""
    if regex is None:
        return {"enabled": False, "regex_sha256": None, "regex_chars": None, "builder": None, "backend": None}
    return {"enabled": True, "regex_sha256": sha256_text(regex), "regex_chars": len(regex), "builder": REGEX_BUILDER,
            "backend": None}


def backend_record(info: dict) -> dict:
    """The structured-outputs backend an engine reports (VllmEngine.info["structured_outputs"]: the pin asked for,
    whether LLM() took it, the backend read after it was built and from where); an engine that reports none gives
    the pin asked for and "unknown"."""
    rec = info.get("structured_outputs")
    if isinstance(rec, dict):
        return dict(rec)
    return {"requested": STRUCTURED_BACKEND, "pin_accepted": None, "backend": "unknown", "source": None,
            "applied": None}


def load_engine(model: str, *, seed: int, gpu_mem: float, max_model_len: int,
                backend: str | None = None) -> VllmEngine:
    """The pinned snapshot from the HF cache (no huggingface_hub call), then vLLM on it."""
    from kit.hf_cache import resolve_pinned
    repo, rev = split_model(model)
    need = QWEN[2] if repo == QWEN[0] else ("config.json", "tokenizer_config.json")
    return VllmEngine(resolve_pinned(repo, rev, need), seed=seed, gpu_mem=gpu_mem, max_model_len=max_model_len,
                      backend=backend)


def cmd_generate(a: argparse.Namespace) -> int:
    t0 = _now()
    summ = Path(a.run) / "summary"
    summ.mkdir(parents=True, exist_ok=True)
    palette = load_palette(a.palette, verify=False)             # names only: no audio in the policy env
    repo, rev = split_model(a.model)
    briefs = sample_briefs(a.seed, a.n)
    mod = PROMPTS[a.prompt]
    head = {"schema": GENERATE_SCHEMA, "partial": True, "requested": a.n, "generated": 0, "stopped": None,
            "briefs": {"seed": a.seed, "count": a.n, "sampler": "lofi_rl.brief.sample_briefs"},
            "model": {"repo": repo, "revision": rev}}
    pal_rec = {"id": palette.id, "sha256": palette.sha256, "path": _rel(a.palette)}
    try:                                # every message is built here, before the model loads
        convs = [mod.build_messages(b, palette) for b in briefs]
        examples = [example_id(a.prompt, b) for b in briefs]
        prompt_rec = prompt_record(a.prompt, palette, convs)
    except PromptError as err:
        stopped = f"error: PromptError: {err}"[:2000]
        _write_json(summ / "generate.json", {
            **head, "partial": False, "stopped": stopped, "palette": pal_rec, "host": _host(),
            "prompt": {"version": a.prompt, "module": mod.__name__, "template_sha256": mod.template_sha256()}})
        print(f"GENERATE_FAILED 0/{a.n} prompt={a.prompt} stopped={stopped}", flush=True)
        return 2
    shas = [messages_sha256(c) for c in convs]
    sampling = dict(SAMPLING, max_tokens=a.max_tokens)
    regex = loopscript_regex(palette) if a.constrained else None
    meta = {**head, "prompt": prompt_rec, "palette": pal_rec,
            "sampling": sampling, "constrained": constrained_block(regex), "engine": {},
            "timing": {"chunks": []}, "host": _host()}
    _write_json(summ / "generate.json", meta)
    chunks = meta["timing"]["chunks"]
    rc = 0
    with (summ / "completions.jsonl").open("w", encoding="utf-8") as f:
        try:
            extra = {}                  # vLLM's StructuredOutputsParams is imported before the model loads
            if regex is not None:
                extra["structured_outputs"] = structured_outputs_params(regex)
            engine = load_engine(a.model, seed=a.seed, gpu_mem=a.gpu_mem, max_model_len=a.max_model_len,
                                 backend=STRUCTURED_BACKEND if regex is not None else None)
            meta["timing"]["load_s"] = engine.load_s
            if regex is not None:
                meta["constrained"]["backend"] = backend_record(engine.info)
            sp = engine.sampling(**sampling, **extra)
            meta["engine"] = dict(engine.info)
            _write_json(summ / "generate.json", meta)
            for start in range(0, a.n, a.chunk):
                # chunk 1 also pays the one-time grammar compile (--constrained): once later chunks exist,
                # predict from them, so a slow compile does not stop a run that would fit (D-020 check, F10)
                timed = chunks[1:] or chunks
                elapsed, longest = _now() - t0, max((c["s"] for c in timed), default=0.0)
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
                                        "prompt_version": a.prompt, "example": examples[i],
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
    regex_sha = meta["constrained"]["regex_sha256"]
    backend = (meta["constrained"]["backend"] or {}).get("backend")
    print(f"GENERATE_{'OK' if rc == 0 and meta['generated'] else 'FAILED'} {meta['generated']}/{a.n} "
          f"prompt={a.prompt} stopped={meta['stopped']} tok/s={meta['timing']['tok_per_s']} "
          f"constrained={f'{regex_sha[:16]} backend={backend}' if regex_sha else 'off'}", flush=True)
    return rc if meta["generated"] else 1


# --------------------------------------------------------------------------------------------- copy readouts
def head_lines(text: str) -> dict[str, str]:
    """A loop text's lines by statement head, the text before the first colon (vamp, keys, A, B, @3 hat, ...)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        head, sep, _ = line.partition(":")
        if sep:
            out[head] = line
    return out


def lane_grids(text: str) -> dict[str, list[str]]:
    """The 16-cell grids of a canonical loop text, by lane: each base lane's 1 or 2 grids (comp; bass unless it is
    `lock`; the drum lanes) and each per-bar edit's grid, under the lane it replaces."""
    out: dict[str, list[str]] = {}
    for line in text.splitlines():
        head, sep, body = line.partition(": ")
        if not sep:
            continue
        lane = head.split()[-1] if head.startswith("@") else head
        if lane in GRID_LANES and body != "lock":
            out.setdefault(lane, []).extend(body.split())
    return out


@dataclass(frozen=True)
class Reference:
    """A complete loop a prompt shows: its id, its canonical text, and what the copy readouts compare with."""
    id: str
    text: str
    grams: frozenset[str]
    lines: Mapping[str, str]                    # head_lines
    grids: Mapping[str, frozenset[str]]         # lane -> every grid of that lane, base or edit

    @classmethod
    def of(cls, ref_id: str, text: str) -> Reference:
        return cls(ref_id, text, grams(text), MappingProxyType(head_lines(text)),
                   MappingProxyType({lane: frozenset(gs) for lane, gs in lane_grids(text).items()}))


def best_match(g: frozenset[str], refs: Sequence[Reference]) -> tuple[float, str | None]:
    """The highest 5-gram Jaccard of a text's grams with `refs`, and the id that gives it (the first, on a tie);
    (0.0, None) when there is nothing to compare with."""
    best, best_id = -1.0, None
    for r in refs:
        j = jaccard(g, r.grams)
        if j > best:
            best, best_id = j, r.id
    return (best, best_id) if best_id is not None else (0.0, None)


def jaccard_bin(j: float) -> str:
    """The jaccard_hist_passing bin of a copy_jaccard value: bins left-closed at JACCARD_EDGES."""
    return JACCARD_BINS[bisect_right(JACCARD_EDGES, j)]


def containment(g: frozenset[str], shown: frozenset[str]) -> float:
    """|g & shown| / |g|: the share of a text's 5-grams that the shown examples hold; 0 for an empty text."""
    return len(g & shown) / len(g) if g else 0.0


def verbatim(text: str, shown: Sequence[Reference]) -> dict[str, bool]:
    """Which of the vamp, keys, A and B lines equal that line of a shown example."""
    lines = head_lines(text)
    return {h: h in lines and any(lines[h] == r.lines.get(h) for r in shown) for h in VERBATIM}


def grid_share(text: str, shown: Sequence[Reference]) -> tuple[int, int]:
    """(k, n): of the text's n grids (base lanes and edits), the k that equal some grid of the same lane, base or
    edit, in a shown example."""
    k = n = 0
    for lane, gs in lane_grids(text).items():
        pool = frozenset().union(*(r.grids.get(lane, frozenset()) for r in shown))
        n += len(gs)
        k += sum(1 for g in gs if g in pool)
    return k, n


def forms(text: str) -> dict:
    """The loop's forms: span 1, bass lock, a snare, ohat or shaker base, and how many per-bar edits."""
    lines = text.splitlines()
    return {"span1": head_lines(text).get("vamp", "").endswith(" span 1"), "lock": "bass: lock" in lines,
            "snare": any(line.startswith("snare:") for line in lines),
            "ohat": any(line.startswith("ohat:") for line in lines),
            "shaker": any(line.startswith("shaker:") for line in lines),
            "edits": sum(1 for line in lines if line.startswith("@"))}


def knob_readout(text: str, brief: Brief, palette: Palette, finish_reason: str | None) -> dict:
    """E, the energy band match and the mood signature (lofi_rl.verify.scoring.score on the raw text, as the reward
    reads it): readouts here, never gates. Imported here, so a verifier that cannot load only loses this readout."""
    from lofi_rl.verify.scoring import score
    k = score(text, brief, palette, finish_reason=finish_reason).knob_parts
    return {"E": round(k["E"], 4), "energy_match": k["energy_match"] == 1, "f_mood": k["f_mood"] == 1}


def energy_bands() -> list[float] | None:
    try:
        from lofi_rl.verify.knobs import ENERGY_BANDS
    except Exception:  # noqa: BLE001: a readout's label, never a reason to stop score
        return None
    return list(ENERGY_BANDS)


class CopyCheck:
    """What one run's completions are compared with: the examples each row's own prompt showed (near_copy, the gated
    rule, plus containment, verbatim lines and grid share), and every example prompt v2 can show (copy.any). It also
    collects the knob readout's errors."""

    def __init__(self, version: str, palette: Palette) -> None:
        self.version, self.palette = version, palette
        refs = prompt_v2.reference_texts(palette, strict=False)
        self.references = tuple(Reference.of(ref_id, text) for ref_id, text in refs.items())
        self.refused = dict(refs.refused)
        self.knob_errors: Counter[str] = Counter()
        self._shown: dict[tuple[str, str], tuple[Reference, ...]] = {}

    def shown(self, brief: Brief) -> tuple[Reference, ...]:
        key = (brief.energy, brief.mood)
        if key not in self._shown:
            self._shown[key] = tuple(Reference.of(ref_id, text)
                                     for ref_id, text in shown_examples(self.version, brief, self.palette))
        return self._shown[key]


# --------------------------------------------------------------------------------------------- score
def score_row(row: dict, palette: Palette, check: CopyCheck) -> dict:
    """One completion through the gates and the compiler, then the copy readouts: its 5-gram Jaccard with the example
    its own prompt showed (near_copy) and with every v2 example (copy_any), its containment in the shown example
    (quasi_copy) and, for a loop that passes, its verbatim lines, shared grids, forms and knob readouts."""
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
    g = grams(copy_text if copy_text is not None else plain_text(row["text"]))
    shown = check.shown(brief)
    j, ref = best_match(g, shown)
    j_any, source = best_match(g, check.references)
    within = containment(g, frozenset().union(*(r.grams for r in shown)))
    out.update(ok=gate is None, gate=gate, detail=detail[:300],
               repairs=dict(sorted(Counter(r.code for r in repairs).items())),
               penalized_codes=sorted({r.code for r in repairs if r.penalized}), p_fix=penalty(repairs),
               copy_jaccard=round(j, 4), near_copy=j >= NEAR_COPY, copy_reference=ref,
               copy_any_jaccard=round(j_any, 4) if source else None, copy_any_source=source,
               near_copy_any=bool(source) and j_any >= NEAR_COPY,
               containment=round(within, 4), quasi_copy=within >= QUASI_COPY,
               verbatim=None, grid_share=None, forms=None, E=None, energy_match=None, f_mood=None)
    if gate is None:
        out.update(verbatim=verbatim(copy_text, shown), grid_share=list(grid_share(copy_text, shown)),
                   forms=forms(copy_text))
        try:
            out.update(knob_readout(row["text"], brief, palette, row.get("finish_reason")))
        except Exception as err:  # noqa: BLE001: a readout that fails is null, never a reason to stop score
            check.knob_errors[f"{type(err).__name__}: {err}"[:300]] += 1
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


FINISH_GROUPS = ("stop", "length", "other")


def in_language(rows: list[dict], recorded: dict | None, palette: Palette) -> dict | None:
    """For a constrained run, how many completions are in the language of the regex generate.json recorded, by
    finish_reason: "stop" ones should all be (one out of it means the engine did not apply the regex), "length" ones
    are cut, so they are counted apart, and "other" is any other reason. The regex is rebuilt from the palette and
    must have the recorded sha256; otherwise nothing is counted (matches_generate false). None when unconstrained."""
    if not (recorded or {}).get("enabled"):
        return None
    regex = loopscript_regex(palette)
    sha = sha256_text(regex)
    out: dict = {"regex_sha256": sha, "matches_generate": sha == recorded.get("regex_sha256")}
    if not out["matches_generate"]:
        return {**out, **dict.fromkeys(FINISH_GROUPS)}
    full = re.compile(regex).fullmatch
    groups: dict[str, list[dict]] = {g: [] for g in FINISH_GROUPS}
    for r in rows:
        reason = r.get("finish_reason")
        groups[reason if reason in ("stop", "length") else "other"].append(r)
    for g, group in groups.items():
        k = sum(1 for r in group if isinstance(r.get("text"), str) and full(r["text"]))
        out[g] = {"n": len(group), "in_language": k, "out_of_language": len(group) - k}
    return out


def constrained_summary(recorded: dict | None, rows: list[dict], palette: Palette) -> dict | None:
    """junk.json's `constrained`: generate.json's block plus the in-language counts; None for a run whose generate.json
    predates --constrained (v0)."""
    if recorded is None:
        return None
    return {**recorded, "in_language": in_language(rows, recorded, palette)}


def prompt_summary(version: str, rows: list[dict], palette: Palette, meta: dict) -> dict:
    """junk.json's "prompt": the run's version and this file's template for it, whether that is the template
    generate recorded, the system shas, the rows sent exactly today's messages for their brief, and the rows whose
    recorded example is not the one the rule gives their brief."""
    mod = PROMPTS[version]
    gen = meta.get("prompt")
    recorded_sha = gen.get("template_sha256") if isinstance(gen, dict) else None
    current: dict[tuple[str, str], tuple[str, str]] = {}
    matching = recorded = mismatch = 0
    for r in rows:
        b = Brief(**r["brief"])
        key = (b.energy, b.mood)
        if key not in current:
            current[key] = (messages_sha256(mod.build_messages(b, palette)), example_id(version, b))
        sha, ex = current[key]
        matching += r.get("prompt_sha256") == sha
        if "example" in r:
            recorded += 1
            mismatch += r["example"] != ex
    return {"version": version, "module": mod.__name__, "template_sha256": mod.template_sha256(),
            "template_matches_generate": None if recorded_sha is None else recorded_sha == mod.template_sha256(),
            "system_sha256": sha256_text(prompt.system_message(palette)) if version == "v1" else None,
            "systems": {f"{e}/{m}": sha256_text(system_text(version, cell_brief(e, m), palette))
                        for e in ENERGIES for m in MOOD_NAMES},
            "rows_matching": matching, "example_recorded": recorded, "example_mismatch": mismatch,
            "as_generated": gen}


def copy_summary(scored: list[dict], ok: list[dict], check: CopyCheck) -> dict:
    """junk.json's "copy": the readouts of copying, none of them a gate (module docstring)."""
    n = len(scored)
    any_rows = [s for s in scored if s["near_copy_any"]]
    within = [s["containment"] for s in ok]
    hist = Counter(jaccard_bin(s["copy_jaccard"]) for s in ok)
    k, g = sum(s["grid_share"][0] for s in ok), sum(s["grid_share"][1] for s in ok)
    edits = Counter(s["forms"]["edits"] for s in ok)
    knobs = None
    if not check.knob_errors and all(s["energy_match"] is not None for s in ok):
        knobs = {"n": len(ok), "energy_match": sum(1 for s in ok if s["energy_match"]),
                 "f_mood": sum(1 for s in ok if s["f_mood"]), "energy_bands": energy_bands()}
    return {
        "note": "readouts, never gates: the Phase 1 bar reads pass_not_copy (D-019)",
        "any": {"rule": ANY_RULE, "threshold": NEAR_COPY, "count": len(any_rows),
                "count_passing": sum(1 for s in any_rows if s["ok"]),
                "sources": dict(sorted(Counter(s["copy_any_source"] for s in any_rows).items())),
                "references": [r.id for r in check.references], "refused": dict(check.refused)},
        "quasi_copy": {"rule": QUASI_RULE, "threshold": QUASI_COPY,
                       "count": sum(1 for s in scored if s["quasi_copy"]),
                       "count_passing": sum(1 for s in ok if s["quasi_copy"])},
        "pass_not_quasi_copy": _rate(sum(1 for s in ok if not s["near_copy"] and not s["quasi_copy"]), n),
        "containment_passing": {**_rounded(percentiles(within, (10, 50, 90)), 4), "max": max(within, default=None)},
        "jaccard_hist_passing": {label: hist.get(label, 0) for label in JACCARD_BINS},
        "verbatim_passing": {**{h: sum(1 for s in ok if s["verbatim"][h]) for h in VERBATIM},
                             "AB": sum(1 for s in ok if s["verbatim"]["A"] and s["verbatim"]["B"])},
        "grid_share_passing": {"verbatim": k, "grids": g, "share": round(k / g, 4) if g else None},
        "forms_passing": {**{f: sum(1 for s in ok if s["forms"][f]) for f in FORMS},
                          "edits": {str(e): edits[e] for e in sorted(edits)}},
        "knobs_passing": knobs,
        "knobs_error": ({"loops": sum(check.knob_errors.values()), "first": next(iter(check.knob_errors))}
                        if check.knob_errors else None),
    }


def cell_summary(scored: list[dict], version: str) -> dict:
    """junk.json's "by_cell": per energy x mood cell, in ENERGIES x MOOD_NAMES order, the example shown and the counts;
    the knob counts are over passing loops, null when their readout failed."""
    out = {}
    for e in ENERGIES:
        for m in MOOD_NAMES:
            group = [s for s in scored if s["energy"] == e and s["mood"] == m]
            ok = [s for s in group if s["ok"]]
            knobs = all(s["energy_match"] is not None for s in ok)
            out[f"{e}/{m}"] = {
                "example": example_id(version, cell_brief(e, m)), "n": len(group),
                "near_copy": sum(1 for s in group if s["near_copy"]),
                "quasi_copy": sum(1 for s in group if s["quasi_copy"]), "pass": len(ok),
                "near_copy_passing": sum(1 for s in ok if s["near_copy"]),
                "quasi_copy_passing": sum(1 for s in ok if s["quasi_copy"]),
                "pass_not_copy": sum(1 for s in ok if not s["near_copy"]),
                "pass_not_quasi_copy": sum(1 for s in ok if not s["near_copy"] and not s["quasi_copy"]),
                "energy_match": sum(1 for s in ok if s["energy_match"]) if knobs else None,
                "f_mood": sum(1 for s in ok if s["f_mood"]) if knobs else None}
    return out


def summarize(rows: list[dict], scored: list[dict], meta: dict, palette: Palette, check: CopyCheck, *,
              unreadable: int = 0, run: str = "", clips: dict | None = None) -> dict:
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
    compared = Counter(s["copy_reference"] for s in scored)

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
                                  "max": max((s["copy_jaccard"] for s in scored), default=None)},
                      "rule": NEAR_COPY_RULE, "references": dict(sorted(compared.items()))},
        "copy": copy_summary(scored, ok, check),
        "by_cell": cell_summary(scored, check.version),
        "prompt": prompt_summary(check.version, rows, palette, meta),
        "constrained": constrained_summary(meta.get("constrained"), rows, palette),
        "palette": {"id": palette.id, "sha256": palette.sha256,
                    "matches_generate": None if gen_palette is None else gen_palette == palette.sha256},
        "by_knob": {"energy": _knobs(scored, "energy", ENERGIES), "mood": _knobs(scored, "mood", MOOD_NAMES)},
        "top_details": {g: [[d, c] for d, c in counter.most_common(8)] for g, counter in sorted(details.items())},
        "briefs": meta.get("briefs"), "sampling": meta.get("sampling"),
        "generate": {k: meta.get(k) for k in ("partial", "requested", "generated", "stopped", "model", "engine",
                                              "timing", "host")},
        "score_stamp": _sync_stamp(),         # equal to generate.host.stamp when one sync served the whole job
    }


def headline(s: dict) -> str:
    p, f = s["pass"], s["failures"]
    lo, hi = p["ci95"]
    rate = f"{100 * p['rate']:.1f}%" if p["rate"] is not None else "-"
    gates = " ".join(f"{g} {f[g]}" for g in GATES if f[g])
    lang = (s.get("constrained") or {}).get("in_language") or {}
    stop, length = lang.get("stop"), lang.get("length")
    in_lang = ("" if not lang else " | in the regex's language: "
               + (f"stop {stop['in_language']}/{stop['n']}, length {length['in_language']}/{length['n']}" if stop
                  else "not counted (the regex is not the one generate recorded)"))
    copy = s.get("copy") or {}
    quasi, not_quasi = copy.get("quasi_copy") or {}, copy.get("pass_not_quasi_copy") or {}
    return (f"JUNK n={s['n']}/{s['requested']} pass {p['count']} = {rate} [{100 * lo:.1f}, {100 * hi:.1f}] | "
            f"near-copies {s['near_copy']['count']} (pass, not a copy: {s['pass_not_copy']['count']}) | "
            f"{gates or 'no failures'} | truncated {s['tokens']['truncated']}{in_lang} | "
            f"quasi-copies {quasi.get('count')} (pass, not a quasi-copy: {not_quasi.get('count')}) | "
            f"prompt {(s.get('prompt') or {}).get('version')}")


def run_version(meta: dict) -> str:
    """The prompt version generate.json records, "v1" when it names none (a run that predates prompt v2)."""
    gen = meta.get("prompt")
    version = gen.get("version") if isinstance(gen, dict) else None
    return version or DEFAULT_PROMPT


def cmd_score(a: argparse.Namespace) -> int:
    summ = Path(a.run) / "summary"
    comp = summ / "completions.jsonl"
    if not comp.is_file():
        print(f"score: {comp} is missing (run generate first)", file=sys.stderr)
        return 2
    meta = _read_json(summ / "generate.json") or {}
    version = run_version(meta)
    if version not in PROMPTS:
        print(f"score: generate.json names prompt version {version!r}, which this junk.py does not know "
              f"({', '.join(PROMPTS)})", file=sys.stderr)
        return 2
    rows, bad = read_jsonl(comp)
    rows.sort(key=lambda r: r["i"])
    palette = load_palette(a.palette, verify=False)
    try:
        check = CopyCheck(version, palette)
        scored = [score_row(r, palette, check) for r in rows]
        summary = summarize(rows, scored, meta, palette, check, unreadable=bad, run=Path(a.run).name,
                            clips=_read_json(summ / "clips.json"))
    except PromptError as err:
        print(f"score: prompt {version} cannot serve palette {palette.id}: {err}", file=sys.stderr)
        return 2
    _write_jsonl(summ / "scored.jsonl", scored)
    _write_json(summ / "junk.json", summary)
    for err, count in check.knob_errors.items():
        print(f"score: the knob readout failed on {count} loop(s), recorded as null: {err}", file=sys.stderr)
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


def render_lanes(jobs: int, workers: int) -> tuple[int, int]:
    """(renders at a time, workers each): `workers` split evenly between at most `workers` renders at a time."""
    jobs = max(1, min(jobs, workers))
    return jobs, max(1, workers // jobs)


def in_order(fn: Callable[[int], dict], items: Sequence[int], jobs: int) -> Iterator[dict]:
    """fn over items, `jobs` at a time, yielded in the items' order whatever finishes first. The pool's threads take
    the items in that order too, so a deadline that fn reads as it starts is passed in the items' order. With one job,
    fn runs here, one item after another, as the render step always did."""
    if jobs <= 1:
        yield from map(fn, items)
        return
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        yield from ex.map(fn, items)


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
    # the ear check reads loops that pass and are not near-copies: 41 of v0's 50 clips were copies (D-020)
    passing = [i for i, s in scored.items() if s["ok"] and not s.get("near_copy") and i in rows]
    pick = sample_indices(passing, a.k, seed)
    palette = load_palette(a.palette, verify=not a.no_verify)
    clips_dir = run / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    jobs, each = render_lanes(a.jobs, a.workers)
    clips = {"schema": CLIPS_SCHEMA, "seed": seed, "k": a.k, "passing": len(passing),
             "palette": {"id": palette.id, "sha256": palette.sha256}, "jobs": jobs, "workers": each, "clips": []}
    t0 = _now()

    def one(i: int) -> dict:
        if _now() - t0 >= a.deadline_s:          # read as the render would start: none starts after the deadline
            rec = {"i": i, "ok": False, "gate": None, "skipped": "render deadline"}
        else:
            rec = render_clip(rows[i], palette, clips_dir, cache=a.cache, workers=each)
        rec.update(brief=rows[i]["brief"], copy_jaccard=scored[i]["copy_jaccard"], near_copy=scored[i]["near_copy"],
                   copy_reference=scored[i].get("copy_reference"), quasi_copy=scored[i].get("quasi_copy"))
        return rec
    for k, rec in enumerate(in_order(one, pick, jobs)):
        i = pick[k]
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
    g.add_argument("--constrained", action="store_true",
                   help="decode under lofi_rl.grammar's LoopScript regex (vLLM structured outputs; docs/GRAMMAR.md)")
    g.add_argument("--prompt", choices=tuple(PROMPTS), default=DEFAULT_PROMPT,
                   help="the prompt template: v1 (lofi_rl.prompt, the default) or v2 (lofi_rl.prompt_v2)")
    s = sub.add_parser("score", help="gates G1-G6 and compile -> summary/junk.json (policy env)")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--palette", type=Path, default=DEFAULT_PALETTE)
    r = sub.add_parser("render", help="a seeded sample of passing loops to FLAC (scorer env)")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--palette", type=Path, default=DEFAULT_PALETTE)
    r.add_argument("--k", type=int, default=RENDER_K)
    r.add_argument("--seed", type=int, default=None, help="default: the brief seed of generate.json")
    r.add_argument("--workers", type=int, default=4,
                   help="worker threads for the whole step, split between the renders at a time (the job: its CPUs)")
    r.add_argument("--jobs", type=int, default=RENDER_JOBS,
                   help="renders at a time, each with --workers // --jobs workers; the clips never depend on it")
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
    if a.cmd == "render" and (a.jobs < 1 or a.workers < 1):
        ap.error("--jobs and --workers must be at least 1")
    return {"score": cmd_score, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
