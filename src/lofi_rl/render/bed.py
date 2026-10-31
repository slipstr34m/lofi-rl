"""The texture bed (instrument.md §4 Rain bed; §6 `bed`).

- Source: any mono or stereo WAV at 44.1 kHz that the palette names (`bed.src`, one of its sources), read
  from `start_s`, through HighpassFilter 200 Hz. A mono source plays in both channels.
- Seam: an equal-power crossfade over X = xfade_s seconds (1 s). With e the filtered excerpt of N + X samples,
  o[k] = e[X + k] for k < N - X, and o[N - X + j] = cos(θ_j) e[N + j] + sin(θ_j) e[j], θ_j = (π/2) j / X.
  So o[N - X] = e[N] continues o[N - X - 1] = e[N - 1], and the fade ends where o wraps to o[0] = e[X],
  the sample after e[X - 1].
- Level: LUFS_bed = LUFS_music - Δ (24 LU by default), with the music measured after the master chain and
  before the limiter (the pipeline does that measurement).
`bed: null` renders without a bed. The crossfade weights come from Python's math (one libm call each), not
from numpy's vectorized trig, whose last bits can depend on the CPU's SIMD width.
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf

from lofi_rl.palette import ROOT, Palette, Plugin
from lofi_rl.render.loudness import lufs

from .mix import process

SR = 44100
HIGHPASS = (Plugin("HighpassFilter", {"cutoff_frequency_hz": 200.0}),)    # §4: "with HighpassFilter 200"
PREROLL_S = Fraction(1, 2)      # the high-pass warms up on up to 0.5 s of the file before start_s


class BedError(ValueError):
    pass


def samples(seconds: float | Fraction) -> int:
    """Seconds (as written in the palette) to whole samples at 44.1 kHz, rounded half up."""
    return int(Fraction(repr(float(seconds))) * SR + Fraction(1, 2))


def read_excerpt(path: Path, start: int, length: int) -> np.ndarray:
    """`length` stereo float32 samples of a WAV from sample `start`, high-passed: (2, length).

    The filter starts up to PREROLL_S before `start` (as much as the file has), so the excerpt carries
    no filter onset transient."""
    info = sf.info(str(path))
    if info.samplerate != SR:
        raise BedError(f"bed {path.name} is {info.samplerate} Hz; convert it to {SR} Hz once (instrument.md §6)")
    if info.channels not in (1, 2):
        raise BedError(f"bed {path.name} has {info.channels} channels; it must be mono or stereo")
    if start < 0 or start + length > info.frames:
        raise BedError(f"bed {path.name} has {info.frames} samples; this loop needs {start + length} "
                       f"(start {start} + loop + crossfade {length})")
    pre = min(start, samples(PREROLL_S))
    x, _ = sf.read(str(path), start=start - pre, stop=start + length, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x.T)
    if x.shape[0] == 1:
        x = np.concatenate([x, x])
    return process(HIGHPASS, x, SR)[:, pre:]


def seam_loop(e: np.ndarray, n: int, xfade: int) -> np.ndarray:
    """The seamless loop o, (2, n) float32, from an excerpt e of at least n + xfade samples."""
    if not 0 <= xfade <= n:
        raise BedError(f"crossfade {xfade} must lie within the loop's {n} samples")
    if e.shape[1] < n + xfade:
        raise BedError(f"the excerpt has {e.shape[1]} samples; the loop needs {n + xfade}")
    e = e.astype(np.float64)
    o = np.empty((e.shape[0], n), dtype=np.float64)
    o[:, :n - xfade] = e[:, xfade:n]
    theta = [math.pi * j / (2 * xfade) for j in range(xfade)]
    fade_out = np.array([math.cos(t) for t in theta], dtype=np.float64)
    fade_in = np.array([math.sin(t) for t in theta], dtype=np.float64)
    o[:, n - xfade:] = fade_out * e[:, n:n + xfade] + fade_in * e[:, :xfade]
    return o.astype(np.float32)


def bed_loop(palette: Palette, n: int, *, root: Path = ROOT) -> np.ndarray | None:
    """The palette's bed as one seamless cycle of n samples, or None for `bed: null`."""
    if palette.bed is None:
        return None
    xfade = samples(palette.bed.xfade_s)
    e = read_excerpt(palette.source_path(palette.bed.src, root), samples(palette.bed.start_s), n + xfade)
    return seam_loop(e, n, xfade)


def place(bed: np.ndarray, music_lufs: float, delta_lu: float, sr: int = SR) -> tuple[np.ndarray, float, float]:
    """The bed scaled to music_lufs - delta_lu: (scaled bed, gain in dB, the bed's own LUFS before it)."""
    own = lufs(bed, sr)
    if not np.isfinite(own):
        raise BedError("the bed is silent")
    gain = music_lufs - delta_lu - own
    return (bed.astype(np.float64) * 10.0 ** (gain / 20.0)).astype(np.float32), gain, own
