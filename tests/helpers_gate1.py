"""Synthetic gate runs for tests/test_gate1_*.py: a run folder shaped as `lofi-junk-rate` writes it (summary/ with
junk.json, scored.jsonl, completions.jsonl, generate.json and clips.json; clips/ with one-pass FLACs and their render
manifests), every value set to what the Phase-1 gate registers, and a few knobs to move it off them. No model, no
render: the FLACs are near-silent int16 files of exactly one pass, each with its own bytes. Sound claims: none.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from lofi_rl import gate1 as G
from lofi_rl.brief import sample_briefs
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.rl import junk as J

STAMP = "commit=ac4acfb0d1e2 dirty=0 diff=- at=2026-11-06T19:00:00Z"
JOB = "13783877"
REFS = ("l1_melancholy_low", "l2_melancholy_mid", "l4_wistful_low", "l6_wistful_high", "l7_hopeful_low")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def verbatim_of(i: int) -> dict:
    """A fixed, mixed pattern of verbatim flags: A on every 3rd sample, B on every 5th."""
    return {"vamp": i % 7 == 0, "keys": False, "A": i % 3 == 0, "B": i % 5 == 0}


def junk_doc(n: int, pass_count: int, pnc: int, *, job: str = JOB) -> dict:
    """junk.json as score writes it, at the gate's registered values."""
    return {
        "schema": J.SCHEMA, "run": f"junk-{job}", "n": n, "requested": n, "partial": False, "unreadable_lines": 0,
        "pass": {**J._rate(pass_count, n), "gates": "G1-G6 and compile on every completion; G7 on the render sample"},
        "pass_not_copy": J._rate(pnc, n),
        "failures": {g: 0 for g in J.GATES} | {"G7": 0},
        "render": {"attempted": 50, "rendered": 50, "G7": 0, "other_gates": 0, "errors": 0, "skipped": 0,
                   "seed": G.BRIEF_SEED, "k": 50},
        "near_copy": {"threshold": 0.8, "count": pass_count - pnc, "count_passing": pass_count - pnc},
        "copy": {"verbatim_passing": {}},
        "prompt": {"version": "v2", "module": "lofi_rl.prompt_v2", "template_sha256": G.TEMPLATE_SHA256,
                   "template_matches_generate": True, "system_sha256": None, "rows_matching": n,
                   "example_recorded": n, "example_mismatch": 0},
        "constrained": {"enabled": True, "regex_sha256": G.REGEX_SHA256, "regex_chars": 3408,
                        "builder": "lofi_rl.grammar.loopscript_regex",
                        "backend": {"requested": "xgrammar", "pin_accepted": True, "backend": "xgrammar",
                                    "source": "llm_engine.vllm_config.structured_outputs_config.backend",
                                    "applied": True},
                        "in_language": {"regex_sha256": G.REGEX_SHA256, "matches_generate": True,
                                        "stop": {"n": n, "in_language": n, "out_of_language": 0},
                                        "length": {"n": 0, "in_language": 0, "out_of_language": 0},
                                        "other": {"n": 0, "in_language": 0, "out_of_language": 0}}},
        "palette": {"id": "lofi_rain_v1", "sha256": G.FROZEN_SHA256, "matches_generate": True},
        "briefs": {"seed": G.BRIEF_SEED, "count": n, "sampler": "lofi_rl.brief.sample_briefs"},
        "sampling": dict(J.SAMPLING),
        "generate": {"partial": False, "requested": n, "generated": n, "stopped": None,
                     "model": {"repo": "Qwen/Qwen3-4B-Instruct-2507", "revision": G.MODEL_REVISION},
                     "engine": {"engine": "vllm"}, "timing": {},
                     "host": {"node": "b11-13.hpc.usc.edu", "job": job, "stamp": STAMP,
                              "gpu": "NVIDIA A40, 46068 MiB"}},
        "score_stamp": STAMP,
    }


def write_flac(path: Path, frames: int, mark: int) -> int:
    """A near-silent one-pass FLAC of `frames` stereo frames, its bytes its own (a click at `mark`); its size."""
    x = np.zeros((frames, 2), dtype=np.int16)
    x[mark % 1000: mark % 1000 + 3, :] = 1000 + mark
    x[-1 - mark % 7, 1] = 77
    sf.write(str(path), x, 44100, format="FLAC", subtype="PCM_16")
    return path.stat().st_size


def render_manifest(i: int, brief: dict, canon: str, lufs: float, peak: float) -> dict:
    """The fields of a render manifest (lofi_rl.render.pipeline) that the batch reads and clip_id hashes."""
    from lofi_rl.brief import Brief
    b = Brief(energy=brief["energy"], mood=brief["mood"], bpm=brief["bpm"], tonic=brief["tonic"])
    return {"schema": "lofi-rl/render/1", "loop": {"canon_sha256": canon, "p_fix": 0.0},
            "brief": {"scene": b.scene, "energy": b.energy, "mood": b.mood, "mode": b.mode, "bpm": b.bpm,
                      "tonic": b.tonic, "key": b.key_name},
            "palette": {"id": "lofi_rain_v1", "sha256": G.FROZEN_SHA256, "frozen": "2026-11-04"},
            "n": loop_samples(b.bpm), "sr": 44100,
            "chain": {"cycles": 2, "fluid_preroll_cycles": 1, "midi_cycles": 3, "pedalboard_buffer": 512,
                      "limiter_tile": 3},
            "loudness": {"lufs": lufs, "true_peak_dbtp": peak}, "gates": {"G1-G6": "pass", "G7": "pass"},
            "audio_sha256": sha(f"pcm {i}")}


def make_run(root: Path, *, n: int = 500, copies: int = 54, k: int = 50, render_k: int | None = None,
             flacs: bool = True, job: str = JOB) -> dict:
    """A gate run in `root` (module doc): every row passes, the first `copies` indices are near-copies, and clips.json
    holds junk's sample of `k` (seed 2026) from the rest, the first `render_k` of them recorded (all by default), each
    with its FLAC and manifest under root/clips/ when `flacs`. Returns {"pool", "sample", "briefs"}."""
    summ, clips_dir = root / "summary", root / "clips"
    summ.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(exist_ok=True)
    briefs = sample_briefs(G.BRIEF_SEED, n)
    rows, scored = [], []
    for i, b in enumerate(briefs):
        rows.append({"i": i, "brief": J.brief_dict(b), "prompt_sha256": sha(f"messages {b.energy}/{b.mood}"),
                     "prompt_version": "v2", "example": REFS[i % len(REFS)], "text": f"vamp: loop {i}\n",
                     "finish_reason": "stop", "n_tokens": 250, "prompt_tokens": 1388})
        scored.append({"i": i, "energy": b.energy, "mood": b.mood, "n_tokens": 250, "finish_reason": "stop",
                       "truncated": False, "canon_sha256": sha(f"canon {i}"), "ok": True, "gate": None, "detail": "",
                       "repairs": {}, "penalized_codes": [], "p_fix": 0.0, "copy_jaccard": 0.85 if i < copies else 0.7,
                       "near_copy": i < copies, "copy_reference": REFS[i % len(REFS)], "verbatim": verbatim_of(i),
                       "quasi_copy": False})
    pnc = n - copies
    (summ / "junk.json").write_text(json.dumps(junk_doc(n, n, pnc, job=job), indent=1))
    (summ / "completions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (summ / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scored))
    (summ / "generate.json").write_text(json.dumps({"schema": J.GENERATE_SCHEMA, "requested": n, "generated": n}))
    pool = list(range(copies, n))
    sample = J.sample_indices(pool, k, G.BRIEF_SEED)
    recs = []
    for i in sample[:k if render_k is None else render_k]:
        b = rows[i]["brief"]
        frames = loop_samples(b["bpm"])
        lufs, peak = -14.0 + (i % 9) * 0.0103217, -1.5 - (i % 5) * 0.0511
        size = write_flac(clips_dir / f"{i:04d}.flac", frames, i) if flacs else 1234
        if flacs:
            (clips_dir / f"{i:04d}.manifest.json").write_text(
                json.dumps(render_manifest(i, b, scored[i]["canon_sha256"], lufs, peak), indent=2, sort_keys=True))
        recs.append({"i": i, "ok": True, "gate": None, "clip": f"clips/{i:04d}.flac", "n": frames,
                     "seconds": round(frames / 44100, 3), "lufs": round(lufs, 3), "true_peak_dbtp": round(peak, 3),
                     "audio_sha256": sha(f"pcm {i}"), "canon_sha256": scored[i]["canon_sha256"], "p_fix": 0.0,
                     "render_s": 3.1, "flac_bytes": size, "brief": b, "copy_jaccard": scored[i]["copy_jaccard"],
                     "near_copy": False, "copy_reference": scored[i]["copy_reference"], "quasi_copy": False})
    (summ / "clips.json").write_text(json.dumps(
        {"schema": J.CLIPS_SCHEMA, "seed": G.BRIEF_SEED, "k": k, "passing": len(pool),
         "palette": {"id": "lofi_rain_v1", "sha256": G.FROZEN_SHA256}, "jobs": 2, "workers": 2, "clips": recs},
        indent=1))
    return {"pool": pool, "sample": sample, "briefs": briefs}


def edit_json(path: Path, fn) -> None:
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc, indent=1))


def junk_of(root: Path) -> dict:
    return json.loads((root / "summary" / "junk.json").read_text())


def deepcopy(x):
    return copy.deepcopy(x)
