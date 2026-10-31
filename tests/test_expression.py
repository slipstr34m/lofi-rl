"""Sax expression (instrument.md section 3, Sax expression): whole vibrato cycles per loop, pitch bend in
range, the scoop, the CC11 swells, and the write grids."""
import math
from fractions import Fraction

import pytest

from helpers_compile import (EXAMPLE_BRIEF, FIX, compiled, dev_palette, loop_of, loop_text, palette_with,
                             passing_fixtures)
from lofi_rl.brief import Brief
from lofi_rl.compile.compiler import compile_loop
from lofi_rl.compile.expression import bend_value, vibrato_cycles
from lofi_rl.compile.midi import stem_midi
from lofi_rl.compile.sax import sax_line
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.dsl.ast import timed

SCOOP_FLOOR = math.floor(-60 / 200 * 8192 + 0.5)        # -2458: -60 cents with a +-2 semitone range
VIBRATO_PEAK = math.floor(18 / 200 * 8192 + 0.5)        # 737: 18 cents


@pytest.mark.parametrize("bpm", range(60, 96))
def test_vibrato_has_a_whole_number_of_cycles_per_loop(bpm):
    n = loop_samples(bpm)
    seconds = n / 44100
    cycles = vibrato_cycles(n, Fraction(11, 2))
    assert cycles == math.floor(5.5 * seconds + 0.5)
    rate = cycles / seconds                              # f_v = round(5.5 L) / L
    assert abs(rate - 5.5) <= 0.5 / seconds + 1e-12
    assert math.isclose(rate * seconds, cycles)


def _notes(c):
    """(note, its bend points, its cc11 points), each note's points in time order from its onset."""
    out = []
    for note in sorted(c.sax.notes, key=lambda x: x.on):
        span = lambda p: (p.at - note.on) % c.n                         # noqa: E731
        pts = lambda ps: sorted((p for p in ps if p.channel == note.channel and span(p) <= note.off - note.on),
                                key=span)                                # noqa: E731
        out.append((note, [(span(p), p.value) for p in pts(c.sax.bend)], [(span(p), p.value) for p in pts(c.sax.cc11)]))
    return out


@pytest.mark.parametrize("name", ["example.ls", "span1_dotted.ls", "shaker_edits.ls", "one_bar_base.ls"])
def test_pitch_bend_stays_in_range_and_in_its_cents_bounds(name):
    for brief in (EXAMPLE_BRIEF, Brief("low", "hopeful", 60, 11), Brief("high", "melancholy", 95, 6)):
        c = compiled(name, brief)
        assert c.sax.bend_range == 2
        assert all(-8192 <= p.value <= 8191 for p in c.sax.bend)
        assert all(SCOOP_FLOOR <= p.value <= VIBRATO_PEAK for p in c.sax.bend)
        assert all(0 <= p.at < c.n for p in c.sax.bend + c.sax.cc11)
        assert all(0 <= p.value <= 127 for p in c.sax.cc11)


def test_bend_value_conversion_clips_at_the_range():
    assert bend_value(0.0, 2) == 0 and bend_value(-60.0, 2) == SCOOP_FLOOR and bend_value(18.0, 2) == VIBRATO_PEAK
    assert bend_value(200.0, 2) == 8191 and bend_value(-200.0, 2) == -8192 and bend_value(-500.0, 2) == -8192


def test_points_follow_every_onset_on_10_and_20_ms_grids():
    c = compiled("example.ls")
    for note, bend, cc11 in _notes(c):
        d = note.off - note.on
        assert [t for t, _ in bend] == [math.floor(j * 441 + 0.5) for j in range(len(bend))]
        assert bend[-1][0] < d <= bend[-1][0] + 441
        grid = [t for t, _ in cc11[:-1]]
        assert grid == [math.floor(j * 882 + 0.5) for j in range(len(grid))] and cc11[-1][0] == d


def test_scoop_after_a_rest_or_a_leap_on_notes_of_4_steps_or_more():
    loop = loop_of("example.ls")
    pal = dev_palette()
    line, _ = sax_line(loop, 2, "dorian", pal.sax["tenor"], pal.groove, pal.sax_expr, loop_samples(82))
    c = compiled("example.ls")
    points = {note.step: bend for note, bend, _ in _notes(c)}
    scooped = [x for x in line if x.scoop and x.dur >= 4]
    assert scooped, "the example has scooped notes"
    for x in line:
        bend = points[x.step]
        if x in scooped:
            assert bend[0] == (0, SCOOP_FLOOR), "starts 60 cents flat at the note-on"
            mid = dict(bend)[1764]                                     # 40 ms: half way along the raised cosine
            assert abs(mid - bend_value(-30.0, 2)) <= 1
            d = c.sax.notes[[n.step for n in c.sax.notes].index(x.step)]
            assert all(v == 0 for t, v in bend if 3528 <= t < 0.35 * (d.off - d.on)), "back to pitch at 80 ms"
        elif x.dur < 8:
            assert all(v == 0 for _, v in bend), "short unscooped notes are not bent"


def test_vibrato_starts_after_35_percent_and_fades_in():
    c = compiled("example.ls")
    longs = [(note, bend) for note, bend, _ in _notes(c) if (note.off - note.on) > 0 and _steps(c, note) >= 8]
    assert len(longs) == 4, "4 notes last 8 steps or more (section 1)"
    for note, bend in longs:
        d = note.off - note.on
        start = 0.35 * d
        assert all(v == 0 for t, v in bend if 3528 <= t < start)
        late = [abs(v) for t, v in bend if t >= start + 13230]       # after the 300 ms fade-in
        assert late and max(late) >= VIBRATO_PEAK - 2
        fading = [(t, v) for t, v in bend if start <= t < start + 13230]
        assert all(abs(v) <= VIBRATO_PEAK * (t - start) / 13230 + 1 for t, v in fading)


def _steps(c, note):
    """The written length in steps of an example.ls note."""
    return {k: it.dur for k, it in _timed(loop_of("example.ls"))}[note.step]


def test_vibrato_phase_is_locked_to_the_loop_clock():
    c = compiled("example.ls")
    cycles = vibrato_cycles(c.n, Fraction(11, 2))
    for note, bend, _ in _notes(c):
        if _steps(c, note) < 8:
            continue
        d = note.off - note.on
        for t, v in bend:
            if t >= 0.35 * d + 13230:
                want = 18 * math.sin(2 * math.pi * ((cycles * ((note.on + t) % c.n)) % c.n) / c.n)
                assert v == bend_value(want, 2)


def test_cc11_swells_long_notes_and_decays_each_phrase_ending():
    c = compiled("example.ls")
    items = _timed(loop_of("example.ls"))
    ends = {max(k for k, it in items if not it.rest and (k < 64) == (ph == "A")) for ph in "AB"}
    for note, _, cc11 in _notes(c):
        d, long = note.off - note.on, _steps(c, note) >= 8
        for t, v in cc11:
            want = 100 + 27 * math.sin(math.pi * t / d) if long else 110
            if note.step in ends:
                w = min(1.0, max(0.0, (t - 0.6 * d) / (0.4 * d)))
                want = (1 - w) * want + w * 80
            assert v == math.floor(want + 0.5), (note.step, t)
        if long:
            assert cc11[0][1] == 100 and max(v for _, v in cc11) >= 125
        if note.step in ends:
            assert cc11[-1] == (d, 80), "the last note of a phrase decays to 80"
    assert len(ends) == 2


def _timed(loop):
    return timed(loop.a, 0) + timed(loop.b, 64)


def test_expression_off_writes_no_bend_and_a_flat_swell():
    c = compile_loop(loop_of("example.ls"), EXAMPLE_BRIEF, palette_with(expr_enabled=False))
    assert c.sax.bend == () and len(c.sax.cc11) == len(c.sax.notes)
    assert {p.value for p in c.sax.cc11} == {110} and {p.at for p in c.sax.cc11} == {x.on for x in c.sax.notes}


@pytest.mark.parametrize("text, channels", [((FIX / "example.ls").read_text(), (2, 3)),
                                            ((FIX / "example.ls").read_text().replace("| h.6 qr", "| h.6 q2"),
                                             (2, 3, 4))], ids=["28_notes", "29_notes"])
def test_the_bend_range_is_set_by_rpn_on_every_sax_channel(text, channels):
    c = compile_loop(loop_text(text), EXAMPLE_BRIEF, dev_palette())
    assert c.sax.channels == channels, "an odd line puts B's last note on a third channel"
    head = [m for m in stem_midi(c, "sax").tracks[0][:60] if m.type in ("control_change", "program_change")]
    for ch in channels:
        rpn = [(m.control, m.value) for m in head
               if m.type == "control_change" and m.channel == ch and m.control in (101, 100, 6, 38)]
        assert rpn == [(101, 0), (100, 0), (6, 2), (38, 0), (101, 127), (100, 127)]
        assert [m.program for m in head if m.type == "program_change" and m.channel == ch] == [c.sax.program]


def test_every_fixture_has_bounded_expression():
    for name in passing_fixtures():
        c = compiled(name)
        assert c.sax.bend and c.sax.cc11
        assert {p.channel for p in c.sax.bend + c.sax.cc11} <= set(c.sax.channels)
