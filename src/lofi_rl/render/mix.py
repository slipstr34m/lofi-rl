"""The bus mix (instrument.md §4, pedalboard 0.9.26 chains; docs/COMPILER.md §4.5).

- Each bus (sax, keys, bass, drums) runs its palette chain, then its static fader `bus_gain_db`. Per-loop
  bus normalization was rejected because it would flatten the energy knob (§4).
- `sax_warm: true` replaces the sax chain's Reverb with the sax_warm chain (high shelf, lowpass, reverb).
- The playing instrument's levels (docs/SHOOTOUTS.md §7.1; `Palette.bus_levels`): a sax, keys or bass `trim_db`
  is a pedalboard Gain at the head of its bus chain, after the sax_warm swap, and each bus's `fader_db` adds to
  `bus_gain_db`. A kit's trim is on its lanes' dB (sampler.kit_shots), never a Gain on the drum bus. A trim or fader
  of 0 adds nothing, so a palette without the fields mixes exactly as before them.
- The buses are summed in float64 in BUSES order, then the master chain runs (high-pass, the low-mid cut,
  glue compression).
- Callers pass several cycles of a periodic signal and keep the last one, so reverb tails and compressor
  state carry across the seam ("two cycles, keep the second", §0 amendment 3).
- pedalboard always gets the same block size (BUFFER), with its state reset at the start of each call.
  Every chain in the dev palette gives the same bits at any block size (tests/test_mix.py), so cycle
  boundaries need not fall on block boundaries.
"""
from __future__ import annotations

import math
import numbers
import statistics
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pedalboard

from lofi_rl.palette import Mix, Plugin

SR = 44100
BUFFER = 8192                   # pedalboard block size, fixed; pedalboard 0.9.26's default
BUSES = ("sax", "keys", "bass", "drums")      # summing order
TRIM_BUSES = ("sax", "keys", "bass")          # FluidSynth buses: a trim is a Gain at the chain head


class MixError(ValueError):
    pass


def bus_chain(mix: Mix, bus: str, trim_db: float = 0.0) -> tuple[Plugin, ...]:
    """A bus's chain as the palette writes it, with the sax_warm swap applied to the sax, and the playing
    instrument's trim as a Gain(trim_db) at its head when it is not 0 (SHOOTOUTS.md §7.1). That is the shootout and
    draft carriage's chain-head slot [Gain, {gain_db: t}] (CD-51) plugin for plugin; the drum bus takes no trim."""
    chain = mix.buses[bus]
    if bus == "sax" and mix.sax_warm:
        reverbs = [i for i, p in enumerate(chain) if p.name == "Reverb"]
        if len(reverbs) != 1:
            raise MixError(f"sax_warm replaces the sax chain's one Reverb, but the chain has {len(reverbs)}")
        i = reverbs[0]
        chain = chain[:i] + mix.buses["sax_warm"] + chain[i + 1:]
    if trim_db == 0.0:
        return chain
    if bus not in TRIM_BUSES:
        raise MixError(f"a {bus} trim is not a Gain on its bus: a kit's trim_db is added to every lane's dB "
                       f"(sampler.kit_shots)")
    return (Plugin("Gain", MappingProxyType({"gain_db": float(trim_db)})),) + tuple(chain)


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
    buses: Mapping[str, np.ndarray]     # each bus after its trim and chain, before its fader


def _levels(values: Mapping[str, float] | None, allowed: tuple[str, ...], what: str) -> dict[str, float]:
    """`trims` or `faders` checked: known buses only, each a finite number of dB."""
    out: dict[str, float] = {}
    for b, v in (values or {}).items():
        if b not in allowed:
            if what == "trims" and b == "drums":
                raise MixError("a drums trim is not a Gain on the drum bus: a kit's trim_db is added to every "
                               "lane's dB (sampler.kit_shots)")
            raise MixError(f"{what}: {b!r} is not one of {list(allowed)}")
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v):
            raise MixError(f"{what}[{b!r}] must be a finite number of dB, got {v!r}")
        out[b] = float(v)
    return out


def mix_buses(stems: Mapping[str, np.ndarray], mix_cfg: Mix, sr: int = SR, workers: int = 4, *,
              trims: Mapping[str, float] | None = None, faders: Mapping[str, float] | None = None) -> MixOut:
    """Bus chains, faders and the master chain over all the cycles given. The four bus chains run in
    parallel threads; each is independent, so the result never depends on how many.

    `trims` (sax, keys, bass) and `faders` (any bus) are the playing instruments' trim_db and fader_db
    (`Palette.bus_levels`; SHOOTOUTS.md §7.1): a trim is a Gain at the head of its bus chain (`bus_chain`), and the
    gain after a bus's chain is bus_gain_db + fader_db. A drums trim is a MixError, since a kit's trim is on its
    lanes. None, or a value of 0, runs that bus exactly as without them."""
    missing = [b for b in BUSES if b not in stems]
    if missing:
        raise MixError(f"missing stems: {missing}")
    shapes = {stems[b].shape for b in BUSES}
    if len(shapes) != 1 or next(iter(shapes))[0] != 2:
        raise MixError(f"stems must all be stereo and the same length, got {sorted(shapes)}")
    trims, faders = _levels(trims, TRIM_BUSES, "trims"), _levels(faders, BUSES, "faders")
    chains = {b: bus_chain(mix_cfg, b, trims.get(b, 0.0)) for b in BUSES}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {b: ex.submit(process, chains[b], stems[b], sr) for b in BUSES}
        buses = {b: futures[b].result() for b in BUSES}
    total = np.zeros(next(iter(shapes)), dtype=np.float64)
    for b in BUSES:
        fader = faders.get(b, 0.0)
        gain_db = mix_cfg.bus_gain_db[b] if fader == 0.0 else mix_cfg.bus_gain_db[b] + fader
        total += buses[b].astype(np.float64) * 10.0 ** (gain_db / 20.0)
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
