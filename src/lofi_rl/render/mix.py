"""The bus mix (instrument.md §4, pedalboard 0.9.26 chains; docs/COMPILER.md §4.5).

- Each bus (sax, keys, bass, drums) runs its palette chain, then its static fader `bus_gain_db`. Per-loop
  bus normalization was rejected because it would flatten the energy knob (§4).
- `sax_warm: true` replaces the sax chain's Reverb with the sax_warm chain (high shelf, lowpass, reverb).
- The buses are summed in float64 in BUSES order, then the master chain runs (high-pass, the low-mid cut,
  glue compression).
- Callers pass several cycles of a periodic signal and keep the last one, so reverb tails and compressor
  state carry across the seam ("two cycles, keep the second", §0 amendment 3).
- pedalboard always gets the same block size (BUFFER), with its state reset at the start of each call.
  Every chain in the dev palette gives the same bits at any block size (tests/test_mix.py), so cycle
  boundaries need not fall on block boundaries.
"""
from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pedalboard

from lofi_rl.palette import Mix, Plugin

SR = 44100
BUFFER = 8192                   # pedalboard block size, fixed; pedalboard 0.9.26's default
BUSES = ("sax", "keys", "bass", "drums")      # summing order


class MixError(ValueError):
    pass


def bus_chain(mix: Mix, bus: str) -> tuple[Plugin, ...]:
    """A bus's chain as the palette writes it, with the sax_warm swap applied to the sax."""
    chain = mix.buses[bus]
    if bus != "sax" or not mix.sax_warm:
        return chain
    reverbs = [i for i, p in enumerate(chain) if p.name == "Reverb"]
    if len(reverbs) != 1:
        raise MixError(f"sax_warm replaces the sax chain's one Reverb, but the chain has {len(reverbs)}")
    i = reverbs[0]
    return chain[:i] + mix.buses["sax_warm"] + chain[i + 1:]


def board(chain: Sequence[Plugin]) -> pedalboard.Pedalboard:
    """Fresh pedalboard plugins for a palette chain (class and parameter names are pedalboard's)."""
    return pedalboard.Pedalboard([getattr(pedalboard, p.name)(**dict(p.params)) for p in chain])


def process(chain: Sequence[Plugin], x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Run a chain over (channels, samples) audio from a reset state, in BUFFER-sample blocks."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if not chain:
        return x.copy()
    return board(chain)(x, sr, buffer_size=BUFFER, reset=True)


@dataclass(frozen=True)
class MixOut:
    music: np.ndarray                   # (2, samples) float32: the master chain's output
    buses: Mapping[str, np.ndarray]     # each bus after its chain, before its fader


def mix_buses(stems: Mapping[str, np.ndarray], mix_cfg: Mix, sr: int = SR, workers: int = 4) -> MixOut:
    """Bus chains, faders and the master chain over all the cycles given. The four bus chains run in
    parallel threads; each is independent, so the result never depends on how many."""
    missing = [b for b in BUSES if b not in stems]
    if missing:
        raise MixError(f"missing stems: {missing}")
    shapes = {stems[b].shape for b in BUSES}
    if len(shapes) != 1 or next(iter(shapes))[0] != 2:
        raise MixError(f"stems must all be stereo and the same length, got {sorted(shapes)}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {b: ex.submit(process, bus_chain(mix_cfg, b), stems[b], sr) for b in BUSES}
        buses = {b: futures[b].result() for b in BUSES}
    total = np.zeros(next(iter(shapes)), dtype=np.float64)
    for b in BUSES:
        total += buses[b].astype(np.float64) * 10.0 ** (mix_cfg.bus_gain_db[b] / 20.0)
    return MixOut(process(mix_cfg.master, total.astype(np.float32), sr), buses)


def calibrated_gains(bus_lufs: Sequence[Mapping[str, float]], targets: Mapping[str, float]) -> dict[str, float]:
    """bus_gain_db that puts each bus's median loudness, relative to the sax's, at its target (§4 Bus balance).

    `bus_lufs` holds one {bus: integrated LUFS before the fader} per hand loop (each manifest's
    buses[bus]["lufs"]); silent buses (None) are skipped. The sax fader stays at 0 dB, so a bus's gain is
    (target[bus] - target[sax]) - (median LUFS[bus] - median LUFS[sax]).
    """
    def median(bus: str) -> float:
        vals = [row[bus] for row in bus_lufs if row.get(bus) is not None and np.isfinite(row[bus])]
        if not vals:
            raise MixError(f"no loudness measured for bus {bus}")
        return statistics.median(vals)
    sax = median("sax")
    return {b: round((targets[b] - targets["sax"]) - (median(b) - sax), 2) for b in BUSES}
