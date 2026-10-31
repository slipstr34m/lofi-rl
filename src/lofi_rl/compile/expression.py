"""Deterministic sax expression: pitch bend and CC11 curves (instrument.md section 3, Sax expression).

Each value is a function of the note list and t mod L only, so the curves repeat exactly every loop.
Per note, from its onset (tau = 0) to its note-off (tau = d, in samples):
- pitch bend every bend_every_ms (10 ms) and CC11 every cc11_every_ms (20 ms), both starting at the onset,
  so the scoop's first value lands with the note-on; CC11 also gets a point at the note-off;
- scoop: notes of 4+ steps after a rest or a leap of 3+ semitones bend from -60 cents to 0 over 80 ms
  along a raised cosine;
- vibrato: notes of 8+ steps start vibrato at 0.35 d, fade it in linearly over 300 ms, depth +-18 cents,
  at f_v = round(5.5 L) / L: a whole number of cycles per loop, phase-locked to the loop clock;
- CC11: short notes 110; long notes (8+ steps) 100 + 27 sin(pi tau / d); the last note of each phrase
  crossfades linearly to 80 over its final 40 %.
Bend values use the RPN range (+-2 semitones): value = round(cents / (100 range) * 8192), clipped.
With expression off there is no bend, and CC11 is the short value at every onset.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

from lofi_rl.palette import SaxExpr

from .events import Control
from .sax import SaxNote
from .timeline import SR, ms, rnd

BEND_MAX = 8191
BEND_MIN = -8192


def vibrato_cycles(n: int, rate_hz: Fraction) -> int:
    """round(rate * L) for a loop of L = n / 44100 seconds: whole vibrato cycles per loop."""
    return rnd(Fraction(rate_hz) * n / SR)


def _grid(d: int, every: Fraction) -> list[int]:
    out, j = [], 0
    while (tau := rnd(j * every)) < d:
        out.append(tau)
        j += 1
    return out


def bend_cents(note: SaxNote, tau: int, expr: SaxExpr, n: int, cycles: int) -> float:
    """Scoop plus vibrato, in cents, tau samples after the note's onset."""
    d = note.off - note.on
    cents = 0.0
    sc = expr.scoop
    span = float(ms(sc.ms))
    if note.scoop and note.dur >= sc.min_steps and tau < span:
        cents += float(sc.cents) * (1.0 + math.cos(math.pi * tau / span)) / 2.0
    vb = expr.vibrato
    start = float(vb.start) * d
    if note.dur >= vb.min_steps and tau >= start:
        fade = min(1.0, (tau - start) / float(ms(vb.fade_ms))) if vb.fade_ms else 1.0
        phase = (cycles * ((note.on + tau) % n)) % n / n       # exact integer reduction, then one division
        cents += float(vb.cents) * fade * math.sin(2.0 * math.pi * phase)
    return cents


def bend_value(cents: float, bend_range: int) -> int:
    return min(BEND_MAX, max(BEND_MIN, math.floor(cents / (100.0 * bend_range) * 8192.0 + 0.5)))


def swell(note: SaxNote, tau: int, expr: SaxExpr) -> int:
    """CC11 tau samples after the note's onset: flat or a sine arch, then the phrase-end decay."""
    c = expr.cc11
    d = note.off - note.on
    value = float(c.short)
    if note.dur >= c.long_steps:
        value = c.long_base + c.long_swell * math.sin(math.pi * min(tau, d) / d)
    if note.last and c.last_share:
        w = min(1.0, max(0.0, (tau - (1 - float(c.last_share)) * d) / (float(c.last_share) * d)))
        value = (1.0 - w) * value + w * c.last
    return min(127, max(0, math.floor(value + 0.5)))


def sax_controls(notes: Sequence[SaxNote], expr: SaxExpr, n: int
                 ) -> tuple[tuple[Control, ...], tuple[Control, ...]]:
    """(pitch-bend points, CC11 points) for the whole line, each sorted by (at, channel)."""
    bend: list[Control] = []
    cc11: list[Control] = []
    cycles = vibrato_cycles(n, expr.vibrato.rate_hz)
    for note in notes:
        if not expr.enabled:
            cc11.append(Control(note.on, note.channel, expr.cc11.short))
            continue
        d = note.off - note.on
        for tau in _grid(d, ms(expr.bend_every_ms)):
            value = bend_value(bend_cents(note, tau, expr, n, cycles), expr.bend_range)
            bend.append(Control((note.on + tau) % n, note.channel, value))
        for tau in _grid(d, ms(expr.cc11_every_ms)) + [d]:
            cc11.append(Control((note.on + tau) % n, note.channel, swell(note, tau, expr)))
    key = lambda c: (c.at, c.channel)                                       # noqa: E731
    return tuple(sorted(bend, key=key)), tuple(sorted(cc11, key=key))
