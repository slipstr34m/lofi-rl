"""The bass: chord roots on the bass grid's rhythm (instrument.md section 3; TASTE.md, Bass holds the roots).

- Pitch = the root of the chord in effect at the strike, one fixed octave per chord for the whole loop:
  (pA, pB) in E1-G2 (28-43) minimizing |pA-36| + |pB-36| + 0.5 |pA-pB|, ties to the lower pair.
- Rhythm = the expanded bass grid (dsl.expand): explicit grids tiled with edits, or `bass: lock`, which
  takes the non-ghost kicks after edits, holds each note to the next onset (at most 8 steps) and forces
  an onset at every chord change (dsl.ast.lock_bass).
- A note held across a chord change is cut there and re-struck if the grid shows `-` (timeline.segments).
"""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

from lofi_rl.dsl.ast import Bars, Chord

from .events import Note
from .humanize import Humanize
from .timeline import BAR, Clock, rnd, segments, wrap

BASS_RANGE = (28, 43)           # E1-G2
BASS_CENTER = 36
SPREAD_WEIGHT = Fraction(1, 2)
CHANNEL = 1


def octave_choices(pc: int, lo: int = BASS_RANGE[0], hi: int = BASS_RANGE[1]) -> tuple[int, ...]:
    return tuple(p for p in range(lo, hi + 1) if p % 12 == pc)


def bass_octaves(root_a: int, root_b: int) -> tuple[int, int]:
    """(pA, pB) for two root pitch classes; ties go to the lower pair (lower sum, then lower pA)."""
    def cost(pair: tuple[int, int]) -> tuple[Fraction, int, int]:
        pa, pb = pair
        return abs(pa - BASS_CENTER) + abs(pb - BASS_CENTER) + SPREAD_WEIGHT * abs(pa - pb), pa + pb, pa
    return min(((pa, pb) for pa in octave_choices(root_a) for pb in octave_choices(root_b)), key=cost)


def held_notes(cells: str, chords: tuple[Chord, ...], pitches: Mapping[Chord, tuple[int, ...]],
               lane: str, clock: Clock, vel: Mapping[str, int], human: Humanize, n: int,
               channel: int) -> tuple[Note, ...]:
    """Notes of a comp or bass grid: each segment plays every pitch of its chord from its start step to
    its end step on the lane's clock (keys and bass share this)."""
    notes = []
    for seg in segments(cells, chords):
        chord = chords[seg.start // BAR]
        on, off = wrap(clock.sample(seg.start), clock.sample(seg.end), n)
        velocity = min(127, max(1, rnd(vel[seg.level] * human.gain(lane, seg.start))))
        for pitch in pitches[chord]:
            notes.append(Note(pitch, on, off, velocity, channel, seg.start))
    return tuple(sorted(notes, key=lambda x: (x.on, x.channel, x.pitch)))


def bass_notes(bars: Bars, pitch: Mapping[Chord, int], clock: Clock, vel: Mapping[str, int],
               human: Humanize, n: int) -> tuple[Note, ...]:
    return held_notes("".join(bars.bass), bars.chords, {c: (p,) for c, p in pitch.items()}, "bass",
                      clock, vel, human, n, CHANNEL)
