"""LoopScript v1 AST (frozen dataclasses) and the 8-bar expansion the verifiers read.

Statement order in the text carries no meaning, so the AST stores lanes in `LANES` order and
edits sorted by (bar, `EDIT_TARGETS`); two texts that differ only in order give equal ASTs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from .vocab import BACKBEAT, EDIT_TARGETS, LANES

BARS = 8
STEPS = 16                      # cells per bar
PHRASE_STEPS = 64               # A = bars 1-4, B = bars 5-8
LOCK_MAX_HOLD = 8               # `bass: lock` notes hold to the next onset, at most 8 steps
REST_BAR = "." * STEPS


@dataclass(frozen=True)
class Chord:
    numeral: str                # "I", "bVII", ... (major-scale reference)
    quality: str                # "m9", "13", ...

    @property
    def text(self) -> str:
        return self.numeral + self.quality


@dataclass(frozen=True)
class Vamp:
    chords: tuple[Chord, Chord]
    span: int = 2               # bars per chord: 2 -> AABB AABB, 1 -> ABAB ABAB


@dataclass(frozen=True)
class Keys:
    instrument: str
    voicing: str


@dataclass(frozen=True)
class Lane:
    name: str                   # one of LANES
    grids: tuple[str, ...]      # 1 or 2 bars of 16 cells from X x g .


@dataclass(frozen=True)
class Edit:
    bar: int                    # 1..8
    target: str                 # a lane, "comp" or "bass"
    grid: str                   # one bar of 16 cells


@dataclass(frozen=True)
class Item:
    """One phrase item. Rests carry degree 0, accidental 0, octave 0."""

    dur: int                    # sixteenths: 1 2 3 4 6 8 12 16 24
    rest: bool = False
    degree: int = 0             # mode degree 1..7
    accidental: int = 0         # -1 flat, 0, +1 sharp
    octave: int = 0             # net count of ' (up) minus , (down)


@dataclass(frozen=True)
class Loop:
    vamp: Vamp
    keys: Keys
    comp: tuple[str, ...]       # 1 or 2 bars of 16 cells from X x g - .
    bass: tuple[str, ...]       # same; the empty tuple means `bass: lock`
    kit: str
    lanes: tuple[Lane, ...]
    a: tuple[Item, ...]
    b: tuple[Item, ...]
    edits: tuple[Edit, ...] = ()
    sax: str | None = None      # optional while the palette has a single sax

    @property
    def bass_lock(self) -> bool:
        return not self.bass

    @property
    def backbeat(self) -> str | None:
        """The lane ("snare" or "rim") that carries the backbeat."""
        return next((lane.name for lane in self.lanes if lane.name in BACKBEAT), None)

    def lane(self, name: str) -> tuple[str, ...] | None:
        return next((lane.grids for lane in self.lanes if lane.name == name), None)


@dataclass(frozen=True)
class Bars:
    """A loop expanded to 8 concrete bars of 16 cells: bases tiled, edits applied, lock resolved."""

    chords: tuple[Chord, ...]
    comp: tuple[str, ...]
    bass: tuple[str, ...]
    drums: Mapping[str, tuple[str, ...]] = field(hash=False)   # every lane in LANES


def lane_key(lane: Lane) -> int:
    return LANES.index(lane.name)


def edit_key(edit: Edit) -> tuple[int, int]:
    return edit.bar, EDIT_TARGETS.index(edit.target)


def chord_bars(vamp: Vamp) -> tuple[Chord, ...]:
    first, second = vamp.chords
    cycle = (first, second) if vamp.span == 1 else (first, first, second, second)
    return tuple(cycle[i % len(cycle)] for i in range(BARS))


def tile(grids: Sequence[str]) -> tuple[str, ...]:
    return tuple(grids[i % len(grids)] for i in range(BARS)) if grids else (REST_BAR,) * BARS


def apply_edits(bars: Sequence[str], edits: Sequence[Edit], target: str) -> tuple[str, ...]:
    out = list(bars)
    for edit in edits:
        if edit.target == target:
            out[edit.bar - 1] = edit.grid
    return tuple(out)


def lock_bass(kick: Sequence[str], chords: Sequence[Chord]) -> tuple[str, ...]:
    """`bass: lock` as 8 bars of cells (instrument.md section 3).

    Onsets are the non-ghost kicks (level copied) plus a forced `x` at every chord change,
    including the seam into bar 1. Each note holds to the next onset, at most 8 steps.
    """
    flat = "".join(kick)
    onsets = {i: c for i, c in enumerate(flat) if c in "Xx"}
    for bar in range(BARS):
        if chords[bar] != chords[bar - 1]:
            onsets.setdefault(bar * STEPS, "x")
    cells = ["."] * len(flat)
    starts = sorted(onsets)
    for j, start in enumerate(starts):
        nxt = starts[j + 1] if j + 1 < len(starts) else starts[0] + len(flat)
        cells[start] = onsets[start]
        for k in range(1, min(LOCK_MAX_HOLD, nxt - start)):
            cells[(start + k) % len(flat)] = "-"
    joined = "".join(cells)
    return tuple(joined[b * STEPS:(b + 1) * STEPS] for b in range(BARS))


def expand(loop: Loop) -> Bars:
    drums = {name: apply_edits(tile(loop.lane(name) or ()), loop.edits, name) for name in LANES}
    chords = chord_bars(loop.vamp)
    bass = lock_bass(drums["kick"], chords) if loop.bass_lock else tile(loop.bass)
    return Bars(
        chords=chords,
        comp=apply_edits(tile(loop.comp), loop.edits, "comp"),
        bass=apply_edits(bass, loop.edits, "bass"),
        drums=MappingProxyType(drums),
    )


def timed(items: Sequence[Item], start: int = 0) -> tuple[tuple[int, Item], ...]:
    """(onset step, item) pairs; use start=64 for phrase B on the 128-step loop clock."""
    out: list[tuple[int, Item]] = []
    for item in items:
        out.append((start, item))
        start += item.dur
    return tuple(out)
