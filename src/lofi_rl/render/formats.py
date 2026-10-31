"""Output files (instrument.md §4 Outputs).

- The canonical render is an IEEE-float32 WAV with no PEAK chunk (wavio.write_float), so the same audio
  always gives the same bytes; its identity is the pcm_sha256 of the samples.
- Listening copies are 16-bit FLAC with TPDF dither. The dither is SplitMix64 noise keyed by the first 8
  bytes of the audio's pcm_sha256, so a given render always gets the same copy. Draws 2m and 2m + 1 of
  that stream dither interleaved sample m: d = u(2m) - u(2m + 1), triangular on (-1, 1) LSB, and the
  sample is rint(32767 x + d), clipped to int16. Everything here is integer or correctly rounded float
  arithmetic, so the copy has the same bits on every machine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from lofi_rl.prng import GOLDEN

from .wavio import pcm_sha256, write_float

SR = 44100
FULL_SCALE = 32767
_GOLDEN = np.uint64(GOLDEN)
_M1, _M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def write_wav(path: Path, x: np.ndarray, sr: int = SR) -> Path:
    """The canonical float32 WAV, (channels, samples) in; byte-deterministic."""
    return write_float(Path(path), x, sr)


def splitmix64_stream(seed: int, count: int, start: int = 0) -> np.ndarray:
    """Outputs start .. start + count - 1 of prng.stream(seed, ...) as uint64, vectorized."""
    k = np.arange(start, start + count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed % (1 << 64)) + k * _GOLDEN + _GOLDEN       # wraps mod 2^64
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def dither_seed(digest: str) -> int:
    """The dither key: the first 8 bytes of a pcm_sha256 hex digest, big-endian."""
    return int(digest[:16], 16)


def tpdf_noise(seed: int, count: int) -> np.ndarray:
    """`count` TPDF values in (-1, 1), exact float64: u(2m) - u(2m + 1) with u = (h >> 11) 2^-53."""
    u = (splitmix64_stream(seed, 2 * count) >> np.uint64(11)).astype(np.float64) * 2.0 ** -53
    return u[0::2] - u[1::2]


def to_int16(x: np.ndarray, seed: int) -> np.ndarray:
    """(channels, samples) float audio to int16 with TPDF dither keyed by `seed`, in interleaved order."""
    inter = np.ascontiguousarray(x.T, dtype=np.float64)
    noise = tpdf_noise(seed, inter.size).reshape(inter.shape)
    q = np.clip(np.rint(inter * FULL_SCALE + noise), -32768, 32767).astype(np.int16)
    return np.ascontiguousarray(q.T)


def write_flac(path: Path, x: np.ndarray, sr: int = SR, digest: str | None = None) -> Path:
    """A 16-bit FLAC listening copy of float audio; the dither is keyed by `digest` (default: its pcm_sha256)."""
    q = to_int16(x, dither_seed(digest or pcm_sha256(x)))
    sf.write(str(path), q.T, sr, format="FLAC", subtype="PCM_16")
    return Path(path)
