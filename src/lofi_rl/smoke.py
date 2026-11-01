"""Phase-0 smoke checks (CLAUDE.md §5), run by jobs/smoke_env.sbatch, one step per env:

    python -m lofi_rl.smoke facts  --out <dir>    node, CPU, GPU, driver, compute mode, stamp, outbound network
    python -m lofi_rl.smoke scorer --out <dir>    GPU, FluidSynth determinism, loop render, MERT/CLAP/Audiobox timings
    python -m lofi_rl.smoke policy --out <dir>    GPU, vLLM loads Qwen3-4B-Instruct-2507 and writes one LoopScript

Each step rewrites <dir>/<step>.json (atomically) before and after every check, so a crash, kill or hang still
leaves the results so far and names the check that did not finish. Every check is recorded with ok/error/timing,
and the step prints <STEP>_OK or <STEP>_FAILED. A per-step watchdog (STEP_TIMEOUT_S) dumps this process's threads
to the job log and exits 1, so a hang (vLLM start-up has no deadline) still ends in SMOKE_FAILED inside the
walltime; a SIGALRM 30 s later ends a hang in native teardown, after Python has cancelled the watchdog. Fatal
signals print Python stacks (faulthandler), in a forked vLLM engine core too. Heavy imports stay inside the
checks, so every step imports in either env.
No result claims anything about how audio sounds; clips are only measured (length, LUFS, peak, hashes).
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SF2 = ROOT / "assets" / "soundfonts" / "generaluser-gs" / "GeneralUser-GS.sf2"
# (repo, pinned commit, files its loader reads); kit.hf_cache.resolve_pinned also checks every indexed shard
QWEN = ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554",
        ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"))
MERT = ("m-a-p/MERT-v1-95M", "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5",
        ("config.json", "preprocessor_config.json", "configuration_MERT.py", "modeling_MERT.py", "pytorch_model.bin"))
CLAP = ("lukewys/laion_clap", "b3708341862f581175dba5c356a4ebf74a9b6651", ("music_audioset_epoch_15_esc_90.14.pt",))
# fetched by 13624271 (CHANGELOG; docs/design/infra_schedule.md). initialize_predictor() loads it by repo id through
# refs/main, so lofi_rl.scoring.features checks refs/main against this revision before it loads
AUDIOBOX = ("facebook/audiobox-aesthetics", "9b1dd8e5df9af7216e836a98974fe3b82c56ded6",
            ("config.json", "model.safetensors"))
EXAMPLE = ROOT / "tests" / "fixtures" / "loops" / "example.ls"
STEP_TIMEOUT_S = {"facts": 120, "scorer": 480, "policy": 900}   # + 30 s backstops + pytest (~1 min): 27.4 of 30 min
CLAP_MAX = 480_000          # laion_clap's 10 s at 48 kHz; a longer clip gets a random crop ('rand_trunc')


class Recorder:
    def __init__(self, step: str, out_dir: Path):
        self.step, self.out_dir, self.checks, self.critical_failed = step, out_dir, {}, []
        out_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, partial: bool) -> bool:
        ok = not partial and not self.critical_failed
        path = self.out_dir / f"{self.step}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"step": self.step, "ok": ok, "partial": partial, "critical_failed": self.critical_failed,
                                   "checks": self.checks}, indent=1, default=str))
        os.replace(tmp, path)                               # a kill mid-write never leaves a truncated file
        return ok

    def run(self, name: str, fn, critical: bool = True):
        self.checks[name] = {"ok": False, "error": "did not finish (crash, kill or watchdog: see the job log)"}
        self._write(partial=True)
        t0 = time.time()
        try:
            out = fn()
            self.checks[name] = {"ok": True, "s": round(time.time() - t0, 3), **(out or {})}
        except Exception as e:  # noqa: BLE001: a smoke records every failure and continues
            self.checks[name] = {"ok": False, "s": round(time.time() - t0, 3), "error": f"{type(e).__name__}: {e}",
                                 "trace": traceback.format_exc()[-6000:]}
            if critical:
                self.critical_failed.append(name)
        self._write(partial=True)
        print(f"[smoke:{self.step}] {name}: {'ok' if self.checks[name]['ok'] else 'FAILED ' + self.checks[name]['error']}", flush=True)
        return self.checks[name]

    def finish(self) -> int:
        ok = self._write(partial=False)
        soft = [n for n, c in self.checks.items() if not c["ok"] and n not in self.critical_failed]
        tail = (f" {self.critical_failed}" if not ok else "") + (f" (non-critical failed: {soft})" if soft else "")
        print(f"{self.step.upper()}_{'OK' if ok else 'FAILED'}{tail}", flush=True)
        return 0 if ok else 1


def _snapshot(repo: str, rev: str, need: tuple[str, ...] = ()) -> str:
    from kit.hf_cache import resolve_pinned      # a checked path: no huggingface_hub call, the same in both envs
    return resolve_pinned(repo, rev, need)


def _gpu() -> dict:
    import torch
    assert torch.cuda.is_available(), "torch sees no GPU"
    return {"device": torch.cuda.get_device_name(0), "bf16": bool(torch.cuda.is_bf16_supported()),
            "torch": torch.__version__, "cuda": torch.version.cuda}


def _gpu_in_child() -> dict:
    """_gpu() in a child interpreter, so the policy process never initialises CUDA before vLLM starts its
    engine core: one CUDA process on the GPU at a time, which also holds in EXCLUSIVE_PROCESS compute mode."""
    r = subprocess.run([sys.executable, "-c", "import json; from lofi_rl.smoke import _gpu; print(json.dumps(_gpu()))"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"GPU probe exited {r.returncode}: {r.stderr[-3000:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


def clap_windows(x, n: int = 4):
    """A loop's mono mix at 48 kHz as n equal float32 windows (n=4: 2 bars each), for CLAP.

    The window length comes from the resampled signal, not the tempo: N is rounded to whole 64-sample
    blocks at 44.1 kHz, so four tempo-exact 2-bar windows can overrun it (82 BPM: 4 x 280,975 samples
    against 1,123,893). Each window must stay under laion_clap's 10 s, or it gets a random crop and the
    embeddings stop being repeatable.
    """
    import numpy as np
    from scipy.signal import resample_poly
    x48 = resample_poly(np.asarray(x).mean(axis=0), 48000, 44100).astype(np.float32)
    win = x48.shape[0] // n
    assert 0 < win < CLAP_MAX, f"CLAP window of {win} samples"
    return x48[: n * win].reshape(n, win)


# ---------------------------------------------------------------- facts
def step_facts(rec: Recorder) -> None:
    def facts():
        cpu = subprocess.run(["bash", "-c", "lscpu | sed -n 's/^Model name: *//p' | head -1"], capture_output=True, text=True).stdout.strip()
        smi = subprocess.run(["bash", "-c", "nvidia-smi --query-gpu=name,driver_version,memory.total,compute_mode --format=csv,noheader 2>&1 | head -2"],
                             capture_output=True, text=True).stdout.strip()
        stamp = (ROOT / ".sync-stamp").read_text().strip() if (ROOT / ".sync-stamp").exists() else "no-stamp"
        return {"node": platform.node(), "cpu": cpu, "gpu": smi, "glibc": platform.libc_ver()[1], "stamp": stamp,
                "job": os.environ.get("SLURM_JOB_ID"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    rec.run("facts", facts)

    def network():
        import urllib.request
        return {"huggingface.co": urllib.request.urlopen("https://huggingface.co", timeout=5).status}   # per address: 12 x 5 s < the facts watchdog
    rec.run("outbound_network", network, critical=False)


# ---------------------------------------------------------------- scorer env
def step_scorer(rec: Recorder, out_dir: Path) -> None:
    rec.run("gpu", _gpu)

    def fluidsynth_determinism():
        import tempfile
        import mido
        from lofi_rl.render import fluid
        from lofi_rl.render.wavio import pcm_sha256, read_float
        with tempfile.TemporaryDirectory() as d:
            mid = mido.MidiFile(ticks_per_beat=480)
            tr = mido.MidiTrack(); mid.tracks.append(tr)
            for note in (60, 62, 64, 65, 67, 69, 71, 72):
                tr.append(mido.Message("note_on", note=note, velocity=90, time=0))
                tr.append(mido.Message("note_off", note=note, velocity=0, time=240))
            mid.save(Path(d) / "scale.mid")
            a, _ = read_float(fluid.render_midi(SF2, Path(d) / "scale.mid", Path(d) / "a.wav"))
            b, _ = read_float(fluid.render_midi(SF2, Path(d) / "scale.mid", Path(d) / "b.wav"))
        assert pcm_sha256(a) == pcm_sha256(b), "same MIDI, different PCM"
        rms = float((a ** 2).mean() ** 0.5)
        assert rms > 10 ** (-60 / 20), "silent render"
        return {"fluidsynth": fluid.version(), "pcm_sha256": pcm_sha256(a), "samples": int(a.shape[1]), "rms": rms}
    rec.run("fluidsynth_determinism", fluidsynth_determinism)

    clip = {}

    def need_clip(key: str):
        if key not in clip:
            raise RuntimeError("skipped: loop_render produced no clip")
        return clip[key]

    def loop_render():
        from lofi_rl.compile import stub
        from lofi_rl.compile.stub_from_loop import notes_from_loop
        from lofi_rl.dsl import parse
        from lofi_rl.render.loudness import lufs, true_peak_dbtp
        from lofi_rl.render.wavio import pcm_sha256, write_float
        n = stub.loop_samples(82)
        res = parse(EXAMPLE.read_text())
        if res.gate.ok:
            notes, programs, hits = notes_from_loop(res.ast, tonic_pc=2, mode="dorian", n=n)
        else:
            notes, programs, hits = _hello_notes(n)        # still a clip for the scorers; the check fails below
        y, li = stub.mix_and_normalize([(stub.render_stems(notes, programs, n, SF2), 1.0), (stub.render_drums(hits, n), 0.5)])
        y2, _ = stub.mix_and_normalize([(stub.render_stems(notes, programs, n, SF2), 1.0), (stub.render_drums(hits, n), 0.5)])
        clip.update(x=y, path=write_float(out_dir.parent / "hello_loop.wav", y, 44100))   # before the asserts; next to summary/, never pulled
        level, tp = lufs(y, 44100), true_peak_dbtp(y)
        assert res.gate.ok, f"example.ls failed gate {res.gate.code}: {res.gate.detail[:200]} (scorers ran on the stand-in loop)"
        assert y.shape == (2, n), f"length {y.shape[1]} != N {n}"
        assert abs(level + 14) <= 0.1 and tp <= -1.0, (level, tp)
        assert pcm_sha256(y) == pcm_sha256(y2), "two renders of the same loop differ"
        return {"source": "LoopScript tests/fixtures/loops/example.ls (D dorian, 82 BPM, Phase-0 stub voicing)",
                "loopscript_gate": {"ok": res.gate.ok, "code": res.gate.code, "repairs": len(res.repairs)},
                "N": n, "lufs": round(level, 3), "true_peak_dbtp": round(tp, 3), "iterations": li.iterations,
                "pcm_sha256": pcm_sha256(y), "wav": str(clip["path"])}
    rec.run("loop_render", loop_render)

    def mert():
        import numpy as np
        import torch
        from scipy.signal import resample_poly
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        x = need_clip("x")
        path = _snapshot(*MERT)
        model = AutoModel.from_pretrained(path, trust_remote_code=True).to("cuda").eval()
        proc = Wav2Vec2FeatureExtractor.from_pretrained(path, trust_remote_code=True)
        x24 = resample_poly(x.mean(axis=0), 24000, 44100).astype(np.float32)
        inp = proc(x24, sampling_rate=24000, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model(**inp, output_hidden_states=True)          # warm-up
            torch.cuda.synchronize(); t0 = time.time()
            out = model(**inp, output_hidden_states=True)
            torch.cuda.synchronize()
        hs = torch.stack(out.hidden_states)
        assert torch.isfinite(hs).all() and hs.shape[0] == 13 and hs.shape[-1] == 768, f"MERT hidden states {list(hs.shape)}"
        return {"model": path, "hidden_states": list(hs.shape), "ms_per_clip": round(1000 * (time.time() - t0), 1)}
    rec.run("mert", mert)

    def clap():
        import numpy as np
        import torch
        import laion_clap
        windows = clap_windows(need_clip("x"))               # 4 x 2 bars, each under 10 s: no random crop
        ckpt = Path(_snapshot(*CLAP)) / CLAP[2][0]
        m = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        m.load_ckpt(str(ckpt), verbose=False)   # the original file: laion_clap 1.1.7 drops position_ids itself (factory.py:64-65)
        texts = ["city at night, rain on the window", "sunny beach at noon"]
        with torch.no_grad():
            m.get_audio_embedding_from_data(x=windows, use_tensor=False)          # warm-up
            torch.cuda.synchronize(); t0 = time.time()
            a = m.get_audio_embedding_from_data(x=windows, use_tensor=False)
            torch.cuda.synchronize(); dt = time.time() - t0
            b = m.get_audio_embedding_from_data(x=windows, use_tensor=False)
            one = m.get_audio_embedding_from_data(x=windows[:1], use_tensor=False)
            t = m.get_text_embedding(texts, use_tensor=False)
        for e in (a, t):
            assert np.isfinite(e).all() and np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-3), "CLAP embeddings are not finite unit vectors"
        assert t.shape == (len(texts), a.shape[1]), f"text embeddings {t.shape}"
        diff = float(np.abs(a - b).max())
        assert diff <= 1e-4, f"the same windows embedded twice differ by {diff}"
        mean = a.mean(axis=0) / np.linalg.norm(a.mean(axis=0))
        return {"model": str(ckpt), "windows": list(windows.shape), "emb": list(a.shape), "text_emb": list(t.shape),
                "max_abs_diff_two_runs": diff, "max_abs_diff_batch1_vs_batch4": float(np.abs(one[0] - a[0]).max()),
                "ms_per_window_batch4": round(1000 * dt / len(windows), 1),
                "cos_loop_vs_text": {s: round(float(c), 4) for s, c in zip(texts, t @ mean)}}
    rec.run("clap", clap)

    def audiobox():
        from audiobox_aesthetics.infer import initialize_predictor
        path = need_clip("path")
        p = initialize_predictor()
        t0 = time.time()
        pred = p.forward([{"path": str(path)}])
        return {"pred": pred, "ms_per_clip": round(1000 * (time.time() - t0), 1)}
    rec.run("audiobox", audiobox)


def _hello_notes(n: int):
    """A fixed stand-in loop (Dm9 / G13 vamp, 2 bars each; kick, backbeat, 8th hats; two long sax notes)."""
    from lofi_rl.compile import stub
    notes, beat = [], n // 32
    for bar in range(8):
        root = 62 if (bar // 2) % 2 == 0 else 67
        for p in (root + 3, root + 7, root + 10, root + 14):
            notes.append(stub.Note(0, p, bar * 4 * beat, 4 * beat - 200, 70))
        notes.append(stub.Note(1, root - 24, bar * 4 * beat, 2 * beat, 90))
    notes.append(stub.Note(2, 69, 0, 6 * beat, 85))
    notes.append(stub.Note(2, 72, 16 * beat, 6 * beat, 85))
    hits = []
    for bar in range(8):
        o = bar * 16
        hits += [("kick", stub.step_sample(o, n), 1.0), ("kick", stub.step_sample(o + 10, n), 0.8),
                 ("snare", stub.step_sample(o + 4, n), 0.9), ("snare", stub.step_sample(o + 12, n), 0.9)]
        hits += [("hat", stub.step_sample(o + k, n), 0.35 if k % 4 else 0.5) for k in range(0, 16, 2)]
    return notes, {0: 4, 1: 33, 2: 66}, hits


# ---------------------------------------------------------------- policy env
SYSTEM = ("You write 8-bar jazzhop loops in LoopScript, a small text DSL. Answer with LoopScript only, no prose. "
          "Use exactly the statement types of this example, with your own content:\n\n")


def step_policy(rec: Recorder) -> None:
    rec.run("gpu", _gpu_in_child)

    def vllm_one_completion():
        cc = os.environ.get("CC")
        if cc is None:                                      # Triton's own order (triton/runtime/build.py)
            cc = shutil.which("gcc") or shutil.which("clang")
        assert cc and shutil.which(cc), "vLLM's model runner JIT-compiles Triton kernels: it needs $CC, gcc or clang"
        python_h = Path(sysconfig.get_paths()["include"]) / "Python.h"
        assert python_h.is_file(), f"Triton's driver build needs {python_h}"
        from vllm import LLM, SamplingParams
        path = _snapshot(*QWEN)
        t0 = time.time()
        llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=0.45, max_model_len=4096, enforce_eager=True, seed=0)
        load_s = time.time() - t0
        example = EXAMPLE.read_text()
        msgs = [{"role": "system", "content": SYSTEM + example},
                {"role": "user", "content": "Brief: city at night, rain on the window. Energy: mid. Mood: wistful."}]
        t1 = time.time()
        req = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=1024))[0]
        gen_s = time.time() - t1
        out = req.outputs[0]
        distinct = len(set(out.token_ids))                  # an EOS-only answer keeps its 1 token; NaN logits repeat one id
        assert out.finish_reason in ("stop", "length") and out.text.strip() and distinct > 2, (
            f"degenerate completion: finish_reason={out.finish_reason!r} n_tokens={len(out.token_ids)} "
            f"distinct={distinct} text={out.text[:300]!r}")
        import torch
        tok = llm.get_tokenizer()
        info = {"model": path, "cc": cc, "mp_method": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") or "fork",
                "parent_cuda_initialized": torch.cuda.is_initialized(), "load_s": round(load_s, 1), "gen_s": round(gen_s, 2),
                "prompt_tokens": len(req.prompt_token_ids or []), "n_tokens": len(out.token_ids),
                "e2e_tok_per_s": round(len(out.token_ids) / max(gen_s, 1e-6), 1), "finish_reason": out.finish_reason,
                "example_tokens_2507": len(tok.encode(example)), "completion": out.text[:3000]}
        try:
            from lofi_rl.dsl import parse
            res = parse(out.text, finish_reason=out.finish_reason, mode="wistful")
            info["parse_gate"] = {"ok": res.gate.ok, "code": res.gate.code, "detail": res.gate.detail[:300], "repairs": len(res.repairs)}
        except Exception as e:  # noqa: BLE001: a parser crash on real model output is recorded, not fatal to this check
            info["parse_gate"] = f"parser raised {type(e).__name__}: {e}"
        return info
    rec.run("vllm_one_completion", vllm_one_completion)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["facts", "scorer", "policy"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    faulthandler.enable(all_threads=True)               # fatal signals print Python stacks; a forked engine core inherits it
    faulthandler.dump_traceback_later(STEP_TIMEOUT_S[a.step], exit=True)   # never cancelled: also covers Python-level teardown
    signal.alarm(STEP_TIMEOUT_S[a.step] + 30)           # no handler: kills a native-teardown hang (fork children don't inherit it)
    rec = Recorder(a.step, a.out)
    {"facts": lambda: step_facts(rec), "scorer": lambda: step_scorer(rec, a.out), "policy": lambda: step_policy(rec)}[a.step]()
    return rec.finish()


if __name__ == "__main__":
    raise SystemExit(main())
