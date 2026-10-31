"""Chord numerals and qualities -> pitch classes (docs/design/instrument.md section 3, chord tones table).

A quality is stored by role (third, fifth, seventh, extensions) because the voicing rules (V2, docs/COMPILER.md
section 9) pick tones by role: rootless drops the root and, under an extension, a perfect fifth (m7b5 keeps its
b5); shell keeps third, seventh and the top extension (third, b5 and seventh for m7b5 and dim7). Extensions are
compound intervals above the root, in the table's order; the last is the top one.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lofi_rl.dsl.ast import Chord
from lofi_rl.dsl.vocab import NUMERAL_SEMITONES, QUALITIES


@dataclass(frozen=True)
class Quality:
    name: str
    third: int                  # 3 or 4; 5 (the fourth) for sus
    fifth: int                  # 7; 6 for m7b5 and dim7
    seventh: int                # 10 or 11; 9 for dim7
    extensions: tuple[int, ...] = ()

    @property
    def tones(self) -> tuple[int, ...]:
        """Semitones above the root, as in the section 3 table."""
        return (0, self.third, self.fifth, self.seventh) + self.extensions

    @property
    def family(self) -> str:
        """maj, dom, min, sus, half (m7b5) or dim."""
        if self.third == 5:
            return "sus"
        if self.third == 4:
            return "maj" if self.seventh == 11 else "dom"
        if self.fifth == 6:
            return "dim" if self.seventh == 9 else "half"
        return "min"


def _q(name: str, third: int, fifth: int, seventh: int, *ext: int) -> Quality:
    return Quality(name, third, fifth, seventh, tuple(ext))


QUALITY: Mapping[str, Quality] = MappingProxyType({q.name: q for q in (
    _q("maj7", 4, 7, 11), _q("maj9", 4, 7, 11, 14), _q("maj7#11", 4, 7, 11, 14, 18),
    _q("7", 4, 7, 10), _q("9", 4, 7, 10, 14), _q("13", 4, 7, 10, 14, 21),
    _q("7sus", 5, 7, 10, 14),
    _q("7b9", 4, 7, 10, 13), _q("7#9", 4, 7, 10, 15),
    _q("m7", 3, 7, 10), _q("m9", 3, 7, 10, 14), _q("m11", 3, 7, 10, 14, 17),
    _q("m7b5", 3, 6, 10, 17),
    _q("dim7", 3, 6, 9),
)})
assert tuple(QUALITY) == QUALITIES, "the table must cover the DSL vocabulary, in its order"


def quality(name: str) -> Quality:
    if name not in QUALITY:
        raise ValueError(f"unknown chord quality {name!r}")
    return QUALITY[name]


def root_pc(numeral: str, tonic: int) -> int:
    """Chord root = tonic + I 0, bII 1, II 2, bIII 3, III 4, IV 5, V 7, bVI 8, VI 9, bVII 10, VII 11."""
    if numeral not in NUMERAL_SEMITONES:
        raise ValueError(f"unknown numeral {numeral!r}")
    return (tonic + NUMERAL_SEMITONES[numeral]) % 12


def chord_pcs(chord: Chord, tonic: int) -> frozenset[int]:
    """Pitch classes of a chord in the brief's key."""
    root = root_pc(chord.numeral, tonic)
    return frozenset((root + t) % 12 for t in quality(chord.quality).tones)
