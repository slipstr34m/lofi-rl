"""Modes and sax degrees (docs/design/instrument.md section 3, Pitch mapping).

melancholy = aeolian, wistful = dorian, hopeful = ionian. The tables live in the DSL vocabulary
(stdlib, both envs) so the parser's bare-numeral repair and the compiler read the same numbers.
"""
from __future__ import annotations

from lofi_rl.dsl.vocab import MODES, MOODS, scale_of

BLUE_NOTES = (3, 6, 10)         # semitones above the tonic: b3, b5, b7 (instrument.md section 0)

__all__ = ["BLUE_NOTES", "MODES", "MOODS", "degree_offset", "home_tonic", "mode_of", "scale_of"]


def mode_of(mood: str) -> str:
    """The mode a mood picks: melancholy -> aeolian, wistful -> dorian, hopeful -> ionian."""
    if mood not in MOODS:
        raise ValueError(f"unknown mood {mood!r}; expected one of {sorted(MOODS)}")
    return MOODS[mood]


def degree_offset(mode: str, degree: int, accidental: int = 0, octave: int = 0) -> int:
    """Semitones above the tonic of a mode degree: S[d-1] + accidental + 12 * octave."""
    if not 1 <= degree <= 7:
        raise ValueError(f"degree {degree} is outside 1-7")
    return scale_of(mode)[degree - 1] + accidental + 12 * octave


def home_tonic(tonic: int, home: tuple[int, int]) -> int:
    """T: the brief's tonic pitch class placed inside an instrument's 12-note home band."""
    lo, hi = home
    if hi - lo != 11:
        raise ValueError(f"home band {home} must span exactly 12 notes")
    return lo + (tonic - lo) % 12
