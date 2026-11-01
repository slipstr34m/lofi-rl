"""Audio features for the reward and the taste RM (taste_pipeline.md §4 "Backbones", §5; docs/SCORER.md §7).

| model | input | output |
|---|---|---|
| MERT-v1-95M (lofi_rl.smoke.MERT pin) | the mix, mono, 24 kHz, four equal 2-bar windows in one batch | per-layer means (13, 768) |
| CLAP music (lofi_rl.smoke.CLAP pin, HTSAT-base) | the mix and the music stem, mono, 48 kHz, four 2-bar windows each (smoke.clap_windows: every window under laion_clap's 10 s, so rand_trunc never fires) | (4, 512) unit window embeddings each |
| Audiobox aesthetics 0.0.4 | the mix (a float WAV, the call the Phase-0 smoke verified) | PQ, CE, PC, CU |

The models run in eval mode under torch.no_grad, in float32 with TF32 off, cuDNN deterministic and no benchmark
autotuning, one clip per batch, so a clip's features never depend on what else is in the request. GPU memory is
capped with torch.cuda.set_per_process_memory_fraction (--gpu-mem-gb). torch, transformers, laion_clap and
audiobox_aesthetics are imported inside Featurizer, so this module imports on the laptop.

feat_version(): the models, the windows, and a sha256 of the code a clip's audio and features come from (the
render pipeline, the DSL, the compiler, this module): a render-code change starts a new cache directory instead
of reusing features of other audio (docs/SCORER.md §8).
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from lofi_rl.palette import ROOT
from lofi_rl.smoke import CLAP, MERT, clap_windows

from .cache import AUDIOBOX_KEYS, ClipFeatures

WINDOWS = 4                     # four 2-bar windows of the 8-bar loop
MERT_SR, CLAP_SR, SR = 24000, 48000, 44100
AUDIOBOX_PKG = "0.0.4"          # requirements/scorer.in
# what a clip's audio and features are a function of, besides the palette (in the clip id)
SOURCE_GLOBS = ("src/lofi_rl/brief.py", "src/lofi_rl/palette.py", "src/lofi_rl/prng.py", "src/lofi_rl/smoke.py",
                "src/lofi_rl/dsl/*.py", "src/lofi_rl/compile/*.py", "src/lofi_rl/theory/*.py",
                "src/lofi_rl/render/*.py", "src/lofi_rl/scoring/features.py", "src/lofi_rl/scoring/render_pool.py")


def source_sha8(root: Path = ROOT, globs: tuple[str, ...] = SOURCE_GLOBS) -> str:
    """sha256 over the listed source files (path and bytes, sorted), first 8 hex."""
    h = hashlib.sha256()
    files = sorted({p for g in globs for p in root.glob(g) if p.is_file()})
    if not files:
        raise FileNotFoundError(f"no source files under {root} for {globs}")
    for p in files:
        h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:8]


def feat_version(kind: str = "real", root: Path = ROOT, *, src: str | None = None) -> str:
    """The feature version (module docstring); `src` is the source_sha8 it names (computed from `root` if None)."""
    base = (f"feat1.mert-{MERT[1][:8]}.clap-{CLAP[1][:8]}.abx-{AUDIOBOX_PKG}.w{WINDOWS}" if kind == "real"
            else f"{kind}1.w{WINDOWS}")
    return f"{base}.src-{source_sha8(root) if src is None else src}"


def mert_windows(mix: np.ndarray, n: int = WINDOWS) -> np.ndarray:
    """The mix, mono at 24 kHz, as n equal float32 windows (the last few samples that do not fill one dropped)."""
    from scipy.signal import resample_poly
    x24 = resample_poly(np.asarray(mix).mean(axis=0), MERT_SR, SR).astype(np.float32)
    win = x24.shape[0] // n
    if win <= 0:
        raise ValueError(f"a clip of {np.asarray(mix).shape[-1]} samples has no MERT windows")
    return x24[: n * win].reshape(n, win)


def check_features(f: ClipFeatures) -> ClipFeatures:
    """Shapes and finiteness; a model that returned NaN is a server fault (HTTP 500), never a score."""
    for name, a, nd in (("mert", f.mert, 2), ("clap_mix", f.clap_mix, 2), ("clap_music", f.clap_music, 2)):
        if a.ndim != nd or not np.all(np.isfinite(a)):
            raise FloatingPointError(f"{f.clip_id}: {name} of shape {a.shape} is not a finite {nd}-D array")
    if f.clap_mix.shape != f.clap_music.shape:
        raise ValueError(f"{f.clip_id}: CLAP mix {f.clap_mix.shape} vs music {f.clap_music.shape}")
    bad = {k: f.audiobox.get(k) for k in AUDIOBOX_KEYS
           if not isinstance(f.audiobox.get(k), float) or not np.isfinite(f.audiobox[k])}
    if bad:
        raise FloatingPointError(f"{f.clip_id}: Audiobox gave {bad}")
    return f


class Featurizer:
    """The scorer env's models (module docstring). featurize() takes one rendered clip; encode_text() gives
    CLAP text embeddings for the theme prompts."""

    name = "real"

    def __init__(self, *, device: str = "cuda", gpu_mem_gb: float = 6.0, tmp_dir: Path | str | None = None) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")     # deterministic cuBLAS
        import torch
        self.torch, self.device = torch, device
        self.cap: dict[str, Any] = {"cap_gb": None}
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda, but torch sees no GPU")
            total = torch.cuda.get_device_properties(0).total_memory
            frac = min(1.0, float(gpu_mem_gb) * 2**30 / total)
            torch.cuda.set_per_process_memory_fraction(frac, 0)
            self.cap = {"cap_gb": float(gpu_mem_gb), "fraction": frac, "total_gb": total / 2**30,
                        "name": torch.cuda.get_device_name(0)}
        elif device != "cpu":
            raise ValueError(f"--device {device!r}: cuda or cpu")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(0)
        self.tmp = Path(tmp_dir or tempfile.mkdtemp(prefix="lofi-scorer-"))
        self.src_sha8 = source_sha8()               # the code feat_ver names: every render worker must run it
        self.feat_ver = feat_version("real", src=self.src_sha8)
        t0 = time.perf_counter()
        self._load()
        self.load_s = round(time.perf_counter() - t0, 2)

    def _load(self) -> None:
        import laion_clap
        from audiobox_aesthetics.infer import initialize_predictor
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        from kit.hf_cache import resolve_pinned
        mert_path = resolve_pinned(*MERT)
        self.mert = AutoModel.from_pretrained(mert_path, trust_remote_code=True).to(self.device).eval()
        self.mert_proc = Wav2Vec2FeatureExtractor.from_pretrained(mert_path, trust_remote_code=True)
        ckpt = Path(resolve_pinned(*CLAP)) / CLAP[2][0]
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.clap.load_ckpt(str(ckpt), verbose=False)  # the original file: laion_clap drops position_ids itself
        self.clap.eval()
        self.abx = initialize_predictor()                # by repo id from the offline cache, as the smoke loads it
        self.models = {"mert": f"{MERT[0]}@{MERT[1]}", "clap": f"{CLAP[0]}@{CLAP[1]}/{CLAP[2][0]}",
                       "audiobox": f"audiobox_aesthetics {AUDIOBOX_PKG} (initialize_predictor)"}

    # -------------------------------------------------------------------------------------- per clip
    def _mert(self, mix: np.ndarray) -> np.ndarray:
        torch = self.torch
        w = mert_windows(mix)
        inp = self.mert_proc(list(w), sampling_rate=MERT_SR, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with torch.no_grad():
            out = self.mert(**inp, output_hidden_states=True)
            hs = torch.stack(out.hidden_states)                     # (13, windows, frames, 768)
            return hs.mean(dim=2).mean(dim=1).float().cpu().numpy()

    def _clap(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            e = self.clap.get_audio_embedding_from_data(x=clap_windows(x, WINDOWS), use_tensor=False)
        return np.asarray(e, dtype=np.float32)

    def _audiobox(self, clip_id: str, mix: np.ndarray) -> dict[str, float]:
        from lofi_rl.render.wavio import write_float
        path = self.tmp / f"{clip_id}.wav"
        write_float(path, mix, SR)
        try:
            with self.torch.no_grad():
                pred = self.abx.forward([{"path": str(path)}])[0]
        finally:
            path.unlink(missing_ok=True)
        return {k: float(pred[k]) for k in AUDIOBOX_KEYS}

    def featurize(self, clip_id: str, canon: str, mix: np.ndarray, music: np.ndarray) -> ClipFeatures:
        t: dict[str, float] = {}
        s = time.perf_counter()
        mert = self._mert(mix)
        t["mert"] = 1000 * (time.perf_counter() - s)
        s = time.perf_counter()
        clap_mix, clap_music = self._clap(mix), self._clap(music)
        t["clap"] = 1000 * (time.perf_counter() - s)
        s = time.perf_counter()
        abx = self._audiobox(clip_id, mix)
        t["audiobox"] = 1000 * (time.perf_counter() - s)
        return check_features(ClipFeatures(clip_id, canon, mert, clap_mix, clap_music, abx, self.device,
                                           {k: round(v, 2) for k, v in t.items()}))

    def encode_text(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            return np.asarray(self.clap.get_text_embedding(list(texts), use_tensor=False), dtype=np.float64)

    # -------------------------------------------------------------------------------------- reports
    def gpu(self) -> dict[str, Any]:
        torch = self.torch
        out = dict(self.cap, device=self.device)
        if self.device == "cuda":
            out.update(peak_allocated_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                       peak_reserved_mb=round(torch.cuda.max_memory_reserved() / 2**20, 1),
                       allocated_mb=round(torch.cuda.memory_allocated() / 2**20, 1))
        return out

    def info(self) -> dict[str, Any]:
        torch = self.torch
        return {"backend": self.name, "models": self.models, "device": self.device, "load_s": self.load_s,
                "torch": torch.__version__, "cuda": torch.version.cuda, "windows": WINDOWS}
