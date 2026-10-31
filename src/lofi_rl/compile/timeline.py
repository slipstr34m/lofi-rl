"""Loop length, grid steps and swing as exact sample positions (instrument.md section 3, Timeline).

- N = 64 * floor(1,323,000 / BPM + 1/2) samples at 44.1 kHz: 8 bars in whole 64-sample blocks, so
  FluidSynth's blocks line up in every cycle. BPM_eff = 84,672,000 / N is what is actually played.
- Step k in [0, 128): t_k = k N/128 + [k odd] (s - 0.5) N/64 + offset_lane, rounded to a sample
  (halves round up). This is sound_sources.md A2's (s - 0.5) 30/BPM seconds of delay per odd 16th.
- Positions are exact fractions until the one rounding, so no float error reaches a sample index, and
  the clock is periodic: t_{k+128} = t_k + N for every lane, offset and nudge.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from lofi_rl.dsl.ast import Chord

SR = 44100
STEPS = 128                     # 8 bars x 16 sixteenths
BAR = 16
BLOCK = 64                      # FluidSynth renders in 64-sample blocks
HALF = Fraction(1, 2)


def loop_samples(bpm: int | Fraction) -> int:
    """N for a tempo: a multiple of 64 within 32 samples of 8 bars of 4/4."""
    return BLOCK * int(Fraction(1_323_000) / Fraction(bpm) + HALF)


def bpm_eff(n: int) -> float:
    """The tempo the loop really plays at: 32 beats in n samples."""
    return 84_672_000 / n


def ms(x: Fraction | int) -> Fraction:
    """Milliseconds -> samples at 44.1 kHz, exactly."""
    return Fraction(x) * SR / 1000


def rnd(x: Fraction) -> int:
    """Round half up: the one rounding rule for every sample position."""
    return int((x + HALF) // 1)


@dataclass(frozen=True)
class Clock:
    """Where step k of one lane falls, in samples from the loop start.

    swing is MPC 16th swing (0.5 straight). swing8, when set, replaces it with 8th-note swing: inside each
    quarter the first 8th gets that share of the time and the 16ths in between are placed linearly
    (used by the sax's `swung` feel). offset is a fixed lane nudge and nudge a per-step humanize offset,
    both in samples.
    """

    n: int
    swing: Fraction = HALF
    offset: Fraction = Fraction(0)
    swing8: Fraction | None = None
    nudge: tuple[Fraction, ...] = ()

    def time(self, k: int) -> Fraction:
        cycle, r = divmod(k, STEPS)
        step = Fraction(self.n, STEPS)
        if self.swing8 is None:
            t = r * step + (r % 2) * (self.swing - HALF) * 2 * step
        else:
            quarter, p = divmod(r, 4)
            warp = (2 * self.swing8 * p if p <= 2 else 4 * self.swing8 + 2 * (1 - self.swing8) * (p - 2))
            t = (4 * quarter + warp) * step
        if self.nudge:
            t += self.nudge[r]
        return t + self.offset + cycle * self.n

    def sample(self, k: int) -> int:
        return rnd(self.time(k))


def wrap(on: int, off: int, n: int) -> tuple[int, int]:
    """An event's onset reduced into [0, n) with its duration kept: the periodic representation."""
    start = on % n
    return start, off - (on - start)


@dataclass(frozen=True)
class Segment:
    """One held stretch of a comp or bass grid: a strike, or a re-strike at a chord change."""

    start: int                  # step 0..127
    end: int                    # exclusive; passes 128 only if a hold wraps the seam with no change
    level: str                  # X | x | g of the strike


def segments(cells: str, chords: Sequence[Chord]) -> tuple[Segment, ...]:
    """Strikes of a 128-cell grid with their holds (`-`), read as a cycle.

    A hold runs over the `-` cells after a strike, across the seam too (bar 8 into bar 1). A note held
    across a chord change is cut at the change and, since the grid shows `-` there, re-struck with the
    new chord at the same level (instrument.md section 3, Chord changes). A `-` with no strike before it
    is silent.
    """
    total = len(cells)
    out = []
    for i, c in enumerate(cells):
        if c not in "Xxg":
            continue
        j = i + 1
        while j < i + total and cells[j % total] == "-":
            j += 1
        start = i
        for b in range(i + 1, j):
            if b % BAR == 0 and chords[(b % total) // BAR] != chords[((b - 1) % total) // BAR]:
                out.append(Segment(start, b, c))
                start = b
        out.append(Segment(start, j, c))
    norm = [Segment(s.start - total, s.end - total, s.level) if s.start >= total else s for s in out]
    return tuple(sorted(norm, key=lambda s: s.start))
