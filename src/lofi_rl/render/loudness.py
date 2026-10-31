"""Loudness for seamless loops (docs/design/instrument.md §4):
- integrated LUFS uses pyloudnorm (BS.1770-4);
- a true-peak BrickwallLimiter (pedalboard) runs on the loop tiled 3 times, and the middle cycle is kept,
  shifted by the limiter's latency, so the seam carries no edge artifacts;
- the gain is iterated until the result is within tolerance of the target.
Arrays are (channels, samples) float32, as pedalboard uses them; pyloudnorm wants (samples, channels).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pyloudnorm as pyln
from pedalboard import BrickwallLimiter
from scipy.signal import resample_poly

TARGET_LUFS = -14.0
TOL_LU = 0.1
CEILING_DBTP = -1.0
LIMIT_DB = -1.5   # pedalboard's true-peak mode can overshoot slightly; aim under the ceiling, then verify


class LoudnessError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoudnessInfo:
    lufs: float
    true_peak_dbtp: float
    gain_db: float
    iterations: int


def lufs(x: np.ndarray, sr: int) -> float:
    return float(pyln.Meter(sr).integrated_loudness(np.ascontiguousarray(x.T, dtype=np.float64)))


def true_peak_dbtp(x: np.ndarray) -> float:
    up = resample_poly(x.astype(np.float64), 4, 1, axis=-1)
    peak = float(np.max(np.abs(up)))
    return -np.inf if peak == 0 else 20 * np.log10(peak)


def _limiter() -> BrickwallLimiter:
    return BrickwallLimiter(ceiling_db=LIMIT_DB, release_ms=100.0, lookahead_ms=5.0, true_peak=True)


@lru_cache(maxsize=None)
def limiter_latency(sr: int) -> int:
    """Samples of delay the limiter adds, measured once with an impulse."""
    imp = np.zeros((1, sr // 2), dtype=np.float32)
    imp[0, 100] = 0.01
    out = _limiter().process(imp, sr)
    return int(np.argmax(np.abs(out[0]))) - 100


def limit_loop(x: np.ndarray, sr: int) -> np.ndarray:
    """Limit one loop cycle as if it repeated forever: tile 3x, limit, keep the middle cycle."""
    n = x.shape[-1]
    lat = limiter_latency(sr)
    y = _limiter().process(np.tile(x.astype(np.float32), 3), sr)
    return y[:, n + lat: 2 * n + lat]


def normalize_loop(x: np.ndarray, sr: int, target: float = TARGET_LUFS, tol: float = TOL_LU,
                   iters: int = 4) -> tuple[np.ndarray, LoudnessInfo]:
    """Gain then limit until the loop sits at `target` LUFS ± `tol`, with true peak at or below CEILING_DBTP."""
    level = lufs(x, sr)
    if not np.isfinite(level):
        raise LoudnessError("silent input")
    g = target - level
    for i in range(1, iters + 1):
        y = limit_loop(x * np.float32(10 ** (g / 20)), sr)
        level = lufs(y, sr)
        if abs(level - target) <= tol:
            tp = true_peak_dbtp(y)
            if tp > CEILING_DBTP:
                raise LoudnessError(f"true peak {tp:.2f} dBTP above {CEILING_DBTP}")
            return y.astype(np.float32), LoudnessInfo(level, tp, g, i)
        g += target - level
    raise LoudnessError(f"not within {tol} LU of {target} after {iters} iterations (last {level:.2f})")
