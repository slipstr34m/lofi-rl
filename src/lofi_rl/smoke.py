"""Phase-0 smoke checks (CLAUDE.md §5), run by jobs/smoke_env.sbatch, one step per env:

    python -m lofi_rl.smoke facts  --out <dir>    node, CPU, GPU, driver, stamp, outbound network
    python -m lofi_rl.smoke scorer --out <dir>    GPU, FluidSynth determinism, loop render, MERT/CLAP/Audiobox timings
    python -m lofi_rl.smoke policy --out <dir>    GPU, vLLM loads Qwen3-4B-Instruct-2507 and writes one LoopScript

Each step writes <dir>/<step>.json. Every check is recorded with ok/error/timing, and the step prints
<STEP>_OK or <STEP>_FAILED. Heavy imports stay inside the checks, so every step imports in either env.
No result claims anything about how audio sounds; clips are only measured (length, LUFS, peak, hashes).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SF2 = ROOT / "assets" / "soundfonts" / "generaluser-gs" / "GeneralUser-GS.sf2"
QWEN = ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554")
MERT = ("m-a-p/MERT-v1-95M", "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5")
CLAP = ("lukewys/laion_clap", "b3708341862f581175dba5c356a4ebf74a9b6651", "music_audioset_epoch_15_esc_90.14.pt")
EXAMPLE = ROOT / "tests" / "fixtures" / "loops" / "example.ls"


class Recorder:
    def __init__(self, step: str):
        self.step, self.checks, self.critical_failed = step, {}, []

    def run(self, name: str, fn, critical: bool = True):
        t0 = time.time()
        try:
            out = fn()
            self.checks[name] = {"ok": True, "s": round(time.time() - t0, 3), **(out or {})}
        except Exception as e:  # noqa: BLE001: a smoke records every failure and continues
            self.checks[name] = {"ok": False, "s": round(time.time() - t0, 3), "error": f"{type(e).__name__}: {e}",
                                 "trace": traceback.format_exc()[-1500:]}
            if critical:
                self.critical_failed.append(name)
        print(f"[smoke:{self.step}] {name}: {'ok' if self.checks[name]['ok'] else 'FAILED ' + self.checks[name]['error']}", flush=True)
        return self.checks[name]

    def finish(self, out_dir: Path) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        ok = not self.critical_failed
        (out_dir / f"{self.step}.json").write_text(json.dumps({"step": self.step, "ok": ok, "critical_failed": self.critical_failed,
                                                              "checks": self.checks}, indent=1, default=str))
        print(f"{self.step.upper()}_{'OK' if ok else 'FAILED'} {self.critical_failed if not ok else ''}".rstrip(), flush=True)
        return 0 if ok else 1


def _snapshot(repo: str, rev: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=rev, local_files_only=True)


def _gpu() -> dict:
    import torch
    assert torch.cuda.is_available(), "torch sees no GPU"
    return {"device": torch.cuda.get_device_name(0), "bf16": bool(torch.cuda.is_bf16_supported()),
            "torch": torch.__version__, "cuda": torch.version.cuda}


# ---------------------------------------------------------------- facts
def step_facts(rec: Recorder) -> None:
    def facts():
        cpu = subprocess.run(["bash", "-c", "lscpu | sed -n 's/^Model name: *//p' | head -1"], capture_output=True, text=True).stdout.strip()
        smi = subprocess.run(["bash", "-c", "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | head -2"],
                             capture_output=True, text=True).stdout.strip()
        stamp = (ROOT / ".sync-stamp").read_text().strip() if (ROOT / ".sync-stamp").exists() else "no-stamp"
        return {"node": platform.node(), "cpu": cpu, "gpu": smi, "glibc": platform.libc_ver()[1], "stamp": stamp,
                "job": os.environ.get("SLURM_JOB_ID")}
    rec.run("facts", facts)

    def network():
        import urllib.request
        return {"huggingface.co": urllib.request.urlopen("https://huggingface.co", timeout=20).status}
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

    def loop_render():
        from lofi_rl.compile import stub
        from lofi_rl.render.loudness import lufs, true_peak_dbtp
        from lofi_rl.render.wavio import pcm_sha256, write_float
        info = {}
        try:
            from lofi_rl.dsl import parse  # the parser lands in Phase 0/1; report its gate on the example loop
            res = parse(EXAMPLE.read_text())
            info["loopscript_gate"] = {"ok": res.gate.ok, "code": res.gate.code, "repairs": len(res.repairs)}
        except Exception as e:  # noqa: BLE001
            info["loopscript_gate"] = f"pending: {type(e).__name__}: {e}"
        n = stub.loop_samples(82)
        notes, programs, hits = _hello_notes(n)
        y, li = stub.mix_and_normalize([(stub.render_stems(notes, programs, n, SF2), 1.0), (stub.render_drums(hits, n), 0.5)])
        y2, _ = stub.mix_and_normalize([(stub.render_stems(notes, programs, n, SF2), 1.0), (stub.render_drums(hits, n), 0.5)])
        assert y.shape == (2, n), f"length {y.shape[1]} != N {n}"
        level, tp = lufs(y, 44100), true_peak_dbtp(y)
        assert abs(level + 14) <= 0.1 and tp <= -1.0, (level, tp)
        assert pcm_sha256(y) == pcm_sha256(y2), "two renders of the same loop differ"
        path = write_float(out_dir.parent / "hello_loop.wav", y, 44100)   # next to summary/, never pulled
        clip.update(x=y, path=path)
        return {**info, "N": n, "lufs": round(level, 3), "true_peak_dbtp": round(tp, 3), "iterations": li.iterations,
                "pcm_sha256": pcm_sha256(y), "wav": str(path)}
    rec.run("loop_render", loop_render)

    def mert():
        import numpy as np
        import torch
        from scipy.signal import resample_poly
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        path = _snapshot(*MERT)
        model = AutoModel.from_pretrained(path, trust_remote_code=True).to("cuda").eval()
        proc = Wav2Vec2FeatureExtractor.from_pretrained(path, trust_remote_code=True)
        x24 = resample_poly(clip["x"].mean(axis=0), 24000, 44100).astype(np.float32)
        inp = proc(x24, sampling_rate=24000, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model(**inp, output_hidden_states=True)          # warm-up
            torch.cuda.synchronize(); t0 = time.time()
            out = model(**inp, output_hidden_states=True)
            torch.cuda.synchronize()
        hs = torch.stack(out.hidden_states)
        return {"hidden_states": list(hs.shape), "ms_per_clip": round(1000 * (time.time() - t0), 1)}
    rec.run("mert", mert, critical=False)

    def clap():
        import numpy as np
        import torch
        import laion_clap
        from huggingface_hub import hf_hub_download
        ckpt = Path(hf_hub_download(CLAP[0], CLAP[2], revision=CLAP[1], local_files_only=True))
        patched = Path(os.environ.get("REMOTE_SCR", "/tmp")) / "cache" / "clap_music_patched.pt"   # re-creatable: scratch, not the shared quota
        if not patched.exists():                              # LAION-AI/CLAP issue #127
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            inner = sd.get("state_dict", sd)
            for k in [k for k in inner if k.endswith("text_branch.embeddings.position_ids")]:
                del inner[k]
            patched.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sd, patched)
        m = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        m.load_ckpt(str(patched))
        from scipy.signal import resample_poly
        x48 = resample_poly(clip["x"].mean(axis=0), 48000, 44100).astype(np.float32)
        win = int(48000 * 8 * 60 / 82)                        # 2 bars at 82 BPM: bar-aligned, under 10 s, no random crop
        chunks = np.stack([x48[i * win:(i + 1) * win] for i in range(4)])
        t0 = time.time()
        a = m.get_audio_embedding_from_data(x=chunks, use_tensor=False)
        dt = time.time() - t0
        b = m.get_audio_embedding_from_data(x=chunks, use_tensor=False)
        return {"emb": list(a.shape), "max_abs_diff_two_runs": float(np.abs(a - b).max()), "ms_per_clip": round(1000 * dt, 1)}
    rec.run("clap", clap, critical=False)

    def audiobox():
        from audiobox_aesthetics.infer import initialize_predictor
        p = initialize_predictor()
        t0 = time.time()
        pred = p.forward([{"path": str(clip["path"])}])
        return {"pred": pred, "ms_per_clip": round(1000 * (time.time() - t0), 1)}
    rec.run("audiobox", audiobox, critical=False)


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
    rec.run("gpu", _gpu)

    def vllm_one_completion():
        from vllm import LLM, SamplingParams
        path = _snapshot(*QWEN)
        t0 = time.time()
        llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=0.45, max_model_len=4096, enforce_eager=True, seed=0)
        load_s = time.time() - t0
        example = EXAMPLE.read_text() if EXAMPLE.exists() else ""
        msgs = [{"role": "system", "content": SYSTEM + example},
                {"role": "user", "content": "Brief: city at night, rain on the window. Energy: mid. Mood: wistful."}]
        t1 = time.time()
        out = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=512))[0].outputs[0]
        gen_s = time.time() - t1
        tok = llm.get_tokenizer()
        info = {"load_s": round(load_s, 1), "gen_s": round(gen_s, 2), "n_tokens": len(out.token_ids),
                "tok_per_s": round(len(out.token_ids) / max(gen_s, 1e-6), 1), "finish_reason": out.finish_reason,
                "example_tokens_2507": len(tok.encode(example)) if example else None, "completion": out.text[:3000]}
        try:
            from lofi_rl.dsl import parse
            res = parse(out.text, finish_reason=out.finish_reason)
            info["parse_gate"] = {"ok": res.gate.ok, "code": res.gate.code, "detail": res.gate.detail[:300], "repairs": len(res.repairs)}
        except Exception as e:  # noqa: BLE001
            info["parse_gate"] = f"pending: {type(e).__name__}: {e}"
        return info
    rec.run("vllm_one_completion", vllm_one_completion)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["facts", "scorer", "policy"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rec = Recorder(a.step)
    a.out.mkdir(parents=True, exist_ok=True)
    {"facts": lambda: step_facts(rec), "scorer": lambda: step_scorer(rec, a.out), "policy": lambda: step_policy(rec)}[a.step]()
    return rec.finish(a.out)


if __name__ == "__main__":
    raise SystemExit(main())
