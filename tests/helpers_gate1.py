"""Synthetic gate runs for tests/test_gate1_*.py: a run folder shaped as `lofi-junk-rate` writes it (summary/ with
junk.json, scored.jsonl, completions.jsonl, generate.json and clips.json; clips/ with one-pass FLACs and, by default,
their render manifests), every value set to what the Phase-1 gate registers, and a few knobs to move it off them. No
model, no render: each FLAC is a sine of exactly one pass at its brief's tempo, with its own pitch, so its bytes and
its loudness are its own; clips.json records the loudness and true peak gate1.batch.measure_flac reads from it, as
the render step records the render's. Votes are written by the rating page's own Ledger (lofi_rl.ui.votes) into
temp folders, never data/votes. Sound claims: none.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from lofi_rl import gate1 as G
from lofi_rl.brief import Brief, sample_briefs
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.gate1 import batch as GB
from lofi_rl.rl import junk as J

STAMP = "commit=ac4acfb0d1e2 dirty=0 diff=- at=2026-11-06T19:00:00Z"
JOB = "13783877"
REFS = ("l1_melancholy_low", "l2_melancholy_mid", "l4_wistful_low", "l6_wistful_high", "l7_hopeful_low")
SEED = 0x5EED5EED5EED5EED


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def verbatim_of(i: int) -> dict:
    """A fixed, mixed pattern of verbatim flags: A on every 3rd sample, B on every 5th."""
    return {"vamp": i % 7 == 0, "keys": False, "A": i % 3 == 0, "B": i % 5 == 0}


def junk_doc(n: int, pass_count: int, pnc: int, *, job: str = JOB, clips: dict | None = None) -> dict:
    """junk.json as score writes it (and render merges its block into), at the gate's registered values."""
    render = J.render_block(clips) if clips else None
    return {
        "schema": J.SCHEMA, "run": f"junk-{job}", "n": n, "requested": n, "partial": False, "unreadable_lines": 0,
        "pass": {**J._rate(pass_count, n), "gates": "G1-G6 and compile on every completion; G7 on the render sample"},
        "pass_not_copy": J._rate(pnc, n),
        "failures": {g: 0 for g in J.GATES} | {"G7": render["G7"] if render else None},
        "render": render,
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
    """A one-pass 16-bit stereo FLAC of `frames` frames: a sine whose pitch and level are the sample's own; its size."""
    t = np.arange(frames, dtype=np.float64) / 44100.0
    x = (0.05 + 0.002 * (mark % 11)) * np.sin(2 * np.pi * (180.0 + 3.0 * (mark % 97)) * t)
    q = np.rint(x * GB.FULL_SCALE).astype(np.int16)
    sf.write(str(path), np.stack([q, q[::-1]], axis=1), 44100, format="FLAC", subtype="PCM_16")
    return path.stat().st_size


def render_manifest(i: int, brief: dict, canon: str, lufs: float, peak: float) -> dict:
    """The fields of a render manifest (lofi_rl.render.pipeline) that the batch reads and clip_id hashes, with the
    render config junk's render step records."""
    b = Brief(energy=brief["energy"], mood=brief["mood"], bpm=brief["bpm"], tonic=brief["tonic"])
    return {**GB.render_config(), "loop": {"canon_sha256": canon, "p_fix": 0.0},
            "brief": {"scene": b.scene, "energy": b.energy, "mood": b.mood, "mode": b.mode, "bpm": b.bpm,
                      "tonic": b.tonic, "key": b.key_name},
            "palette": {"id": "lofi_rain_v1", "sha256": G.FROZEN_SHA256, "frozen": "2026-11-04"},
            "n": loop_samples(b.bpm), "sr": 44100, "loudness": {"lufs": lufs, "true_peak_dbtp": peak},
            "gates": {"G1-G6": "pass", "G7": "pass"}, "audio_sha256": sha(f"pcm {i}")}


def make_run(root: Path, *, n: int = 500, copies: int = 54, k: int = 8, render_k: int | None = None,
             flacs: bool = True, manifests: bool = True, measure: bool = True, job: str = JOB) -> dict:
    """A gate run in `root` (module doc): every row passes, the first `copies` indices are near-copies, and clips.json
    holds junk's sample of `k` (seed 2026) from the rest, the first `render_k` of them recorded (all by default), each
    with its FLAC under root/clips/ when `flacs` and its render manifest beside it when `manifests`. Returns {"pool",
    "sample", "briefs", "measured"}: measured is clip file name -> (LUFS, true peak) as measure_flac read them, or, with
    `measure` False (faster, for tests that build with a stub measure), the -14.0 and -1.5 clips.json then records."""
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
    pool = list(range(copies, n))
    sample = J.sample_indices(pool, k, G.BRIEF_SEED)
    recs, measured = [], {}
    for i in sample[:k if render_k is None else render_k]:
        b = rows[i]["brief"]
        frames = loop_samples(b["bpm"])
        flac = clips_dir / f"{i:04d}.flac"
        if flacs:
            size = write_flac(flac, frames, i)
            lufs, peak = GB.measure_flac(flac) if measure else (-14.0, -1.5)
            measured[flac.name] = (lufs, peak)
            if manifests:
                (clips_dir / f"{i:04d}.manifest.json").write_text(
                    json.dumps(render_manifest(i, b, scored[i]["canon_sha256"], lufs, peak), indent=2, sort_keys=True))
        else:
            size, lufs, peak = 1234, -14.0, -1.5
        recs.append({"i": i, "ok": True, "gate": None, "clip": f"clips/{i:04d}.flac", "n": frames,
                     "seconds": round(frames / 44100, 3), "lufs": round(lufs, 3), "true_peak_dbtp": round(peak, 3),
                     "audio_sha256": sha(f"pcm {i}"), "canon_sha256": scored[i]["canon_sha256"], "p_fix": 0.0,
                     "render_s": 3.1, "flac_bytes": size, "brief": b, "copy_jaccard": scored[i]["copy_jaccard"],
                     "near_copy": False, "copy_reference": scored[i]["copy_reference"], "quasi_copy": False})
    clips = {"schema": J.CLIPS_SCHEMA, "seed": G.BRIEF_SEED, "k": k, "passing": len(pool),
             "palette": {"id": "lofi_rain_v1", "sha256": G.FROZEN_SHA256}, "jobs": 2, "workers": 2, "clips": recs}
    (summ / "junk.json").write_text(json.dumps(junk_doc(n, n, n - copies, job=job, clips=clips), indent=1))
    (summ / "completions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (summ / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scored))
    (summ / "generate.json").write_text(json.dumps({"schema": J.GENERATE_SCHEMA, "requested": n, "generated": n}))
    (summ / "clips.json").write_text(json.dumps(clips, indent=1))
    return {"pool": pool, "sample": sample, "briefs": briefs, "measured": measured}


def copy_run(src: Path, dest: Path) -> Path:
    """A fresh copy of a made run, so a test can change it."""
    shutil.copytree(src, dest)
    return dest


def edit_json(path: Path, fn) -> None:
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc, indent=1))


def edit_rows(path: Path, fn) -> None:
    """Change the rows of a JSONL file in place: fn(rows) edits the list."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    fn(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def junk_of(root: Path) -> dict:
    return json.loads((root / "summary" / "junk.json").read_text())


def deepcopy(x):
    return copy.deepcopy(x)


# --------------------------------------------------------------------------------------------- votes
class Clock:
    """The Ledger's clock, moved by hand (tests/test_ui_votes.py's)."""

    def __init__(self):
        self.t = datetime(2026, 11, 7, 21, 0, 0, tzinfo=timezone(timedelta(hours=-8)))

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


def triage(r, tier: str, bars: str = "11111111") -> dict:
    """A triage vote for round r of a loaded batch, as the page sends it."""
    c = r.clips[0]
    return {"type": "triage", "round": r.id, "tier": tier,
            "clips": [{"clip": c.id, "pos": c.pos, "heard_bars": bars, "heard_ms": 20000}], "last_heard": c.id,
            "switches": 0, "play_order": [1], "autoplay_done": bars.count("1") == 8, "t_ms": {"commit": 25000},
            "device": "airpods", "audio": {"sr": 44100}}


def abandon(r, bars: str) -> dict:
    c = r.clips[0]
    return {"type": "abandon", "round": r.id, "why": "esc",
            "clips": [{"clip": c.id, "pos": c.pos, "heard_bars": bars, "heard_ms": 3000}], "t_ms": {"abandon": 3000}}
