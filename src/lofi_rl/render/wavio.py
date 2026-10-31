"""WAV I/O with a deterministic identity. libsndfile (used by FluidSynth and soundfile) writes a PEAK chunk
with a wall-clock timestamp into float WAVs. Two renders of the same audio a second apart therefore
differ in one header byte (seen 2026-10-30, at offset 60). Identity is defined on the PCM, and our own
outputs are written without a PEAK chunk.
Arrays are (channels, samples) float32, as pedalboard uses them."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import wavfile


def read_float(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(x.T), int(sr)


def write_float(path: Path, x: np.ndarray, sr: int) -> Path:
    """IEEE-float WAV with no PEAK chunk, so the same audio always gives the same bytes."""
    wavfile.write(str(path), sr, np.ascontiguousarray(x.T, dtype=np.float32))
    return path


def pcm_sha256(x: np.ndarray) -> str:
    """sha256 of the float32 little-endian samples, interleaved (samples, channels): the render identity."""
    return hashlib.sha256(np.ascontiguousarray(x.T, dtype="<f4").tobytes()).hexdigest()
