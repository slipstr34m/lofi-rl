"""Knob verifiers, obeyed at full weight (instrument.md §5; TASTE.md "Obeyed"): the energy band and the mood
signature, which back the 90 % obedience claim. knobs = (f_energy + f_mood) / 2 (Claude's default: §5 gives no
combination).

**Energy.** E = .35 min(1, H/192) + .15 n_X/n_nonghost + .25 min(1, n_keys/48) + .25 min(1, n_bass/48).
- H counts drum hits of all six lanes over the 8 bars, X and x at 1 and g at 0.5; n_X / n_nonghost is the share of
  accents among the non-ghost hits (0 when there are none).
- n_keys and n_bass count strikes (X x g) of the comp and bass grids after edits and `bass: lock`.
- f_energy = exp(-d / 0.04), d the distance from E to the brief's band; obedience is the band match.
- **The bands are provisional** (Claude's default). §5 takes them from the tertiles of E over the seed pool, frozen
  in config/energy_bands.yaml with a DECISIONS.md entry, and the pool does not exist yet. Until then they are
  (0.38, 0.59): with three loops in each energy cell, the 9 shootout reference loops (docs/SHOOTOUTS.md §5; E 0.289-
  0.297 low, 0.462-0.478 mid, 0.700-0.720 high) put a pool's tertiles in the gaps between the cells, and each edge
  is its gap's midpoint (0.3795, 0.589), rounded. instrument.md §1's example (energy mid) has E 0.503 inside them,
  while §5's test bands (.30, .50) would call it high.

**Mood.** score_m = sum over the vamp's chords of bars_c sum over tones w s_m(pc): w = 1 for the root, the 3rd (the
4th of 7sus) and the 7th, 0.5 otherwise; s_m = +1 in mode m, 0 in Alt_m, -1 otherwise, with Alt aeolian {11, 1},
dorian {11}, ionian {8, 10} (semitones above the tonic). f_mood = [the brief's mode beats both others by 2 or more].
"""
from __future__ import annotations

import math

from lofi_rl.dsl.ast import Vamp, chord_bars
from lofi_rl.dsl.vocab import LANES, MODES, NUMERAL_SEMITONES
from lofi_rl.theory.chords import quality

from .view import NON_GHOST, STRIKES, View

ENERGY_BANDS = (0.38, 0.59)     # provisional (see the docstring)
ENERGY_SCALE = 0.04
H_FULL, KEYS_FULL, BASS_FULL = 192, 48, 48
ENERGY_WEIGHTS = (0.35, 0.15, 0.25, 0.25)
ALT = {"aeolian": frozenset({11, 1}), "dorian": frozenset({11}), "ionian": frozenset({8, 10})}
MOOD_MARGIN = 2


def energy(view: View) -> float:
    cells = "".join("".join(view.bars.drums[lane]) for lane in LANES)
    h = sum(c in NON_GHOST for c in cells) + 0.5 * cells.count("g")
    non_ghost = sum(c in NON_GHOST for c in cells)
    accents = cells.count("X") / non_ghost if non_ghost else 0.0
    keys = sum(c in STRIKES for c in "".join(view.bars.comp))
    bass = sum(c in STRIKES for c in "".join(view.bars.bass))
    w_h, w_x, w_k, w_b = ENERGY_WEIGHTS
    return w_h * min(1.0, h / H_FULL) + w_x * accents + w_k * min(1.0, keys / KEYS_FULL) + w_b * min(1.0, bass / BASS_FULL)


def _band(level: str, bands: tuple[float, float]) -> tuple[float, float]:
    lo, hi = bands
    return {"low": (0.0, lo), "mid": (lo, hi), "high": (hi, 1.0)}[level]


def energy_band(e: float, bands: tuple[float, float] = ENERGY_BANDS) -> str:
    """low below the first edge, high from the second, mid between (an edge belongs to the band above it)."""
    lo, hi = bands
    return "low" if e < lo else "high" if e >= hi else "mid"


def energy_fit(e: float, target: str, bands: tuple[float, float] = ENERGY_BANDS) -> float:
    """f_energy = exp(-d / 0.04), d = the distance from E to the target band (0 inside it)."""
    lo, hi = _band(target, bands)
    return math.exp(-max(lo - e, 0.0, e - hi) / ENERGY_SCALE)


def mood_scores(vamp: Vamp) -> dict[str, float]:
    """score_m for aeolian, dorian and ionian (§5's table: Im9 IV13 gives dorian 34, aeolian 26)."""
    bars = chord_bars(vamp)
    out = {}
    for mode in ("aeolian", "dorian", "ionian"):
        scale, alt, total = frozenset(MODES[mode]), ALT[mode], 0.0
        for chord in vamp.chords:
            q, root = quality(chord.quality), NUMERAL_SEMITONES[chord.numeral]
            guide = {0, q.third, q.seventh}
            tones = sum((1.0 if t in guide else 0.5) * (1 if (root + t) % 12 in scale else
                                                       0 if (root + t) % 12 in alt else -1) for t in q.tones)
            total += bars.count(chord) * tones
        out[mode] = total
    return out


def mood_margin(scores: dict[str, float], mode: str) -> float:
    return scores[mode] - max(v for m, v in scores.items() if m != mode)


def mood_pass(scores: dict[str, float], mode: str) -> bool:
    return mood_margin(scores, mode) >= MOOD_MARGIN


def parts(view: View, energy_target: str) -> dict[str, float]:
    e = energy(view)
    scores = mood_scores(view.loop.vamp)
    return {
        "E": e,
        "f_energy": energy_fit(e, energy_target),
        "energy_match": float(energy_band(e) == energy_target),
        "mood_aeolian": scores["aeolian"],
        "mood_dorian": scores["dorian"],
        "mood_ionian": scores["ionian"],
        "mood_margin": mood_margin(scores, view.mode),
        "f_mood": float(mood_pass(scores, view.mode)),
    }


def combine(p: dict[str, float]) -> float:
    return (p["f_energy"] + p["f_mood"]) / 2
