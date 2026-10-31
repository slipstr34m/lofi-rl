"""Closed vocabularies of LoopScript v1 (docs/design/instrument.md sections 1-2).

Grammar-level lists (numerals, qualities, voicings, lanes, durations) are fixed by the spec.
The names the model may write (keys instruments, drum kits, saxes) are placeholders until the
palette freeze: callers build a `Vocab` from palettes/lofi_rain_v1.yaml to override them.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

NUMERAL_SEMITONES: Mapping[str, int] = MappingProxyType({
    "I": 0, "bII": 1, "II": 2, "bIII": 3, "III": 4, "IV": 5,
    "V": 7, "bVI": 8, "VI": 9, "bVII": 10, "VII": 11,
})
NUMERALS: tuple[str, ...] = tuple(NUMERAL_SEMITONES)
QUALITIES: tuple[str, ...] = (
    "maj7", "maj9", "maj7#11", "7", "9", "13", "7sus", "7b9", "7#9",
    "m7", "m9", "m11", "m7b5", "dim7",
)
MINOR_QUALITIES = frozenset({"m7", "m9", "m11", "m7b5", "dim7"})   # a minor third over the root
VOICINGS: tuple[str, ...] = ("rootless", "shell", "spread", "quartal")
LANES: tuple[str, ...] = ("kick", "snare", "rim", "hat", "ohat", "shaker")
BACKBEAT: tuple[str, ...] = ("snare", "rim")
EDIT_TARGETS: tuple[str, ...] = ("comp", "bass") + LANES        # also the canonical edit order
STATEMENT_HEADS = frozenset({"vamp", "keys", "comp", "bass", "drums", "sax", "a", "b", *LANES})

# Free aliases (section 2, leniency step 5). Beyond the spec's list: bare Δ, sus4 and 7sus4, and
# the prefix respellings in `canonical_quality` (M/Δ -> maj, -/min -> m, case-insensitive maj/min).
LANE_ALIASES: Mapping[str, str] = MappingProxyType(
    {"bd": "kick", "sd": "snare", "hh": "hat", "oh": "ohat", "shk": "shaker"})
QUALITY_ALIASES: Mapping[str, str] = MappingProxyType({
    "M7": "maj7", "Δ7": "maj7", "Δ": "maj7", "-7": "m7", "min7": "m7",
    "ø": "m7b5", "ø7": "m7b5", "°7": "dim7",
    "sus": "7sus", "9sus4": "7sus", "sus4": "7sus", "7sus4": "7sus",
})

# Durations in sixteenths. A dot multiplies by 1.5; "s." does not exist.
DUR_STEPS: Mapping[str, int] = MappingProxyType({"s": 1, "e": 2, "q": 4, "h": 8, "w": 16})
STEPS_DUR: Mapping[int, str] = MappingProxyType(
    {1: "s", 2: "e", 3: "e.", 4: "q", 6: "q.", 8: "h", 12: "h.", 16: "w", 24: "w."})

MODES: Mapping[str, tuple[int, ...]] = MappingProxyType({
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "ionian": (0, 2, 4, 5, 7, 9, 11),
})
MOODS: Mapping[str, str] = MappingProxyType(
    {"melancholy": "aeolian", "wistful": "dorian", "hopeful": "ionian"})

# Content the model never writes (G5): tempo/swing, key/mode, effects/levels, the bed.
TEMPO_WORDS = frozenset({"tempo", "bpm", "swing"})
KEY_WORDS = frozenset({"key", "mode", "scale", "major", "minor", "ionian", "dorian", "phrygian",
                       "lydian", "mixolydian", "aeolian", "locrian"})
EFFECT_WORDS = frozenset({"reverb", "delay", "echo", "chorus", "flanger", "phaser", "compressor",
                          "compression", "limiter", "eq", "filter", "lowpass", "highpass",
                          "distortion", "saturation", "bitcrush", "bitcrusher", "wow", "flutter",
                          "sidechain", "fx", "effect", "effects", "volume", "gain", "level",
                          "levels", "db", "velocity", "pan"})
BED_WORDS = frozenset({"bed", "rain", "vinyl", "crackle", "noise", "texture", "ambience", "hiss",
                       "tape"})
FORBIDDEN_WORDS = TEMPO_WORDS | KEY_WORDS | EFFECT_WORDS | BED_WORDS

# Lowercase e, b and g collide with durations, flats and ghost cells, so lowercase note names
# count only for a, c, d, f with an octave digit (c4, f#3); uppercase always does (D, Eb4, qD).
_NOTE_NAME = re.compile(r"(?:[seqhwSEQHW]\.?)?(?:[A-G][b#♭♯]?-?\d?|[acdf][b#♭♯]?-?\d)")
# Root, optional quality, then extensions that may carry quality words (G7sus, C9sus4, E7alt,
# CmMaj7), then an optional slash bass or 6/9. Quality words fold case (CMaj7). No other
# uppercase letter follows the root, so the lane alias BD is not a chord.
_ABS_CHORD = re.compile(
    r"[A-G][b#♭♯]?"
    r"(?:(?i:maj|min|dim|aug)|[mMø°Δ+\-])?"
    r"(?:(?i:maj|sus|add|alt)|[0-9#b♭♯()])*"
    r"(?:/(?:[A-G][b#♭♯]?|[0-9]+))?")
_TRIAD_SEVENTH = {(4, 7, 11): "maj7", (4, 7, 10): "7", (3, 7, 10): "m7", (3, 6, 10): "m7b5",
                  (3, 6, 9): "dim7"}


def _names(items: Iterable[str]) -> frozenset[str]:
    return frozenset(s.strip().lower() for s in items)


@dataclass(frozen=True)
class Vocab:
    """Everything a parse is checked against. Names are case-folded on construction."""

    keys: frozenset[str] = frozenset({"rhodes", "piano", "guitar", "organ", "wurli", "ep"})
    kits: frozenset[str] = frozenset({"dusty", "brushes", "jazz"})
    sax: frozenset[str] = frozenset({"tenor", "alto"})
    voicings: frozenset[str] = frozenset(VOICINGS)
    numerals: frozenset[str] = frozenset(NUMERALS)
    qualities: frozenset[str] = frozenset(QUALITIES)
    forbidden: frozenset[str] = FORBIDDEN_WORDS
    name_aliases: Mapping[str, str] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        for name in ("keys", "kits", "sax", "voicings", "forbidden"):
            object.__setattr__(self, name, _names(getattr(self, name)))
        object.__setattr__(self, "numerals", frozenset(self.numerals))
        object.__setattr__(self, "qualities", frozenset(self.qualities))
        aliases = {k.strip().lower(): v.strip().lower() for k, v in dict(self.name_aliases).items()}
        object.__setattr__(self, "name_aliases", MappingProxyType(aliases))


DEFAULT_VOCAB = Vocab()


def canonical_quality(text: str, qualities: frozenset[str] = frozenset(QUALITIES)) -> str | None:
    """Spelling of a chord quality after aliasing, or None if it is not in `qualities`.

    Only the one-letter `M` (major) vs `m` (minor) prefix is case-sensitive.
    """
    if text in qualities:
        return text
    cand = QUALITY_ALIASES.get(text) or _respell(text)
    return cand if cand in qualities else None


def _respell(text: str) -> str:
    low = text.lower()
    if low.startswith("maj"):
        return "maj" + low[3:]
    if low.startswith("min"):
        return "m" + low[3:]
    if text.startswith(("M", "Δ")):
        return "maj" + text[1:]
    if text.startswith("-"):
        return "m" + text[1:]
    return QUALITY_ALIASES.get(low, low)


def scale_of(mode: str) -> tuple[int, ...]:
    """Semitones of a mode, accepting the mood names too (wistful -> dorian)."""
    name = MOODS.get(mode.lower(), mode.lower())
    if name not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODES) + sorted(MOODS)}")
    return MODES[name]


def diatonic_quality(numeral: str, mode: str) -> str | None:
    """The mode's diatonic 7th chord quality on `numeral`, or None if the root is not in the mode."""
    scale = scale_of(mode)
    root = NUMERAL_SEMITONES.get(numeral)
    if root is None or root not in scale:
        return None
    i = scale.index(root)
    tones = tuple((scale[(i + k) % 7] - root) % 12 for k in (2, 4, 6))
    return _TRIAD_SEVENTH.get(tones)


def forbidden_kind(token: str, words: frozenset[str] = FORBIDDEN_WORDS) -> str | None:
    """Why a failing token is forbidden content (G5), or None if it is an ordinary parse error."""
    if _NOTE_NAME.fullmatch(token) or _ABS_CHORD.fullmatch(token):
        return "a note name or absolute chord"
    low = token.lower().rstrip(":=")
    hits = [w for w in re.findall(r"[a-z]+", low) if w in words and (len(w) >= 3 or w == low)]
    return f"a forbidden word ({hits[0]})" if hits else None
