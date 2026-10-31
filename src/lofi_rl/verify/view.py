"""The symbolic reading of one loop that every verifier shares (instrument.md §5: bars b = 1..8, steps 1..16).

Everything comes from the parsed Loop and the CompiledLoop, never from audio:
- the 8 concrete bars of `dsl.expand` (bases tiled, edits applied, `bass: lock` resolved), with the bases alone
  kept too, for the seam's "base pattern for bar 8";
- comp and bass "as heard": the expanded grid with each `-` resolved through the compiler's own `segments`, so a
  `-` after a rest is silence ('.'), a hold stays '-', and a hold cut at a chord change is the re-strike it plays
  (its strike's level);
- the sax line as items on the 128-step clock: each note's degree as semitones above the tonic (`rel`), its
  sounding MIDI pitch after the compiler's octave fit and folds (`pitch`, read from the CompiledLoop) and the
  chord of the bar it starts in.
Standard library only (it runs in the policy env).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from lofi_rl.brief import Brief
from lofi_rl.compile.events import CompiledLoop
from lofi_rl.compile.timeline import segments
from lofi_rl.dsl.ast import BARS, PHRASE_STEPS, STEPS, Bars, Chord, Loop, expand, tile, timed
from lofi_rl.dsl.vocab import LANES, NUMERAL_SEMITONES, scale_of
from lofi_rl.theory.chords import quality
from lofi_rl.theory.modes import BLUE_NOTES, degree_offset

TOTAL = BARS * STEPS            # 128 steps a loop
NON_GHOST = frozenset("Xx")
STRIKES = frozenset("Xxg")      # a ghost is a hit, never silence (docs/DSL.md §7)
BACKBEAT = ("snare", "rim")


@dataclass(frozen=True)
class SaxItem:
    """One item of the sax line. A rest has rel 0, pitch 0 and no chord."""

    phrase: str                 # A | B
    step: int                   # onset step 0..127 on the loop clock
    dur: int                    # steps
    rest: bool
    rel: int = 0                # semitones above the tonic as written: mode degree + accidental + 12 * octave marks
    pitch: int = 0              # sounding MIDI pitch, after the compiler's octave fit and folds
    chord: Chord | None = None  # the chord of the bar the item starts in

    @property
    def pc(self) -> int:
        """Pitch class relative to the tonic (0 = the tonic)."""
        return self.rel % 12


@dataclass(frozen=True)
class View:
    loop: Loop
    bars: Bars
    mode: str                                           # aeolian | dorian | ionian
    items: tuple[SaxItem, ...]                          # A then B, rests included, in time order
    base: Mapping[str, tuple[str, ...]] = field(hash=False)   # drum lanes from their bases alone, no edits
    comp: tuple[str, ...] = ()                          # comp bars as heard
    bass: tuple[str, ...] = ()                          # bass bars as heard

    @property
    def notes(self) -> tuple[SaxItem, ...]:
        return tuple(i for i in self.items if not i.rest)

    def phrase(self, name: str) -> tuple[SaxItem, ...]:
        return tuple(i for i in self.items if i.phrase == name)

    def drum(self, lane: str, bar: int) -> str:
        """Bar `bar` (0-7) of a drum lane after edits: 16 cells of X x g ."""
        return self.bars.drums[lane][bar]

    @property
    def scale(self) -> frozenset[int]:
        return frozenset(scale_of(self.mode))


def chord_rel(chord: Chord) -> frozenset[int]:
    """A chord's tones as pitch classes relative to the tonic (extensions included)."""
    root = NUMERAL_SEMITONES[chord.numeral]
    return frozenset((root + t) % 12 for t in quality(chord.quality).tones)


def is_free(item: SaxItem, scale: frozenset[int]) -> bool:
    """instrument.md §0 amendment 1 and TASTE.md (Claude's defaults): a note is free if its pitch class is in the
    mode, is a blue note (3, 6 or 10 semitones above the tonic), or is a tone of the chord sounding at its onset."""
    return item.pc in scale or item.pc in BLUE_NOTES or (item.chord is not None and item.pc in chord_rel(item.chord))


def heard(cells: str, chords: tuple[Chord, ...]) -> tuple[str, ...]:
    """A 128-cell comp or bass grid as it plays, per bar: strikes and re-strikes as their level, holds as '-',
    silence as '.' (timeline.segments, so it is the compiler's own reading of holds and chord changes)."""
    out = ["."] * TOTAL
    for seg in segments(cells, chords):
        out[seg.start % TOTAL] = seg.level
        for k in range(seg.start + 1, seg.end):
            out[k % TOTAL] = "-"
    joined = "".join(out)
    return tuple(joined[b * STEPS:(b + 1) * STEPS] for b in range(BARS))


def make_view(loop: Loop, brief: Brief, compiled: CompiledLoop) -> View:
    bars = expand(loop)
    pitch_at = {note.step: note.pitch for note in compiled.sax.notes}
    items = []
    for phrase, seq, start in (("A", loop.a, 0), ("B", loop.b, PHRASE_STEPS)):
        for k, it in timed(seq, start):
            if it.rest:
                items.append(SaxItem(phrase, k, it.dur, True))
            else:
                rel = degree_offset(brief.mode, it.degree, it.accidental, it.octave)
                items.append(SaxItem(phrase, k, it.dur, False, rel, pitch_at[k], bars.chords[k // STEPS]))
    return View(
        loop=loop, bars=bars, mode=brief.mode, items=tuple(items),
        base=MappingProxyType({lane: tile(loop.lane(lane) or ()) for lane in LANES}),
        comp=heard("".join(bars.comp), bars.chords), bass=heard("".join(bars.bass), bars.chords),
    )
