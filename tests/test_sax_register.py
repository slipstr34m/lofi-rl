"""The sax line (instrument.md section 3, Pitch mapping, Registers, Sax expression velocity and note-off;
TASTE.md: one sax plays every loop)."""
from dataclasses import replace

import pytest

from helpers_compile import (EXAMPLE_BRIEF, FIX, compiled, dev_palette, loop_of, loop_text, palette_with,
                             passing_fixtures)
from lofi_rl.brief import MOOD_NAMES, Brief
from lofi_rl.compile.compiler import compile_loop
from lofi_rl.compile.sax import fit_shift, sax_clock, sax_pitches
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.dsl.ast import timed
from lofi_rl.dsl.repair import penalty
from lofi_rl.prng import below, stream

BASE = """vamp: Im9 IV13 span 2
keys: rhodes rootless
comp: x---------------
bass: lock
drums: jazz
kick: x.......x.......
snare: ....x.......x...
hat: x.x.x.x.x.x.x.x.
sax: tenor
"""


def test_the_example_lands_where_section_1_says():
    c = compiled("example.ls")
    notes = sorted(c.sax.notes, key=lambda x: x.step)
    assert (notes[0].step, notes[0].on, notes[0].pitch) == (0, 0, 57), "bar 1 lands on A3, the 5th of Dm9"
    assert next(x.pitch for x in notes if x.step == 64) == 62, "bar 5 lands on D4, its root"
    assert (min(x.pitch for x in notes), max(x.pitch for x in notes)) == (50, 62), "sax range D3-D4"
    assert 56 in {x.pitch for x in notes}, "eb5 is Ab, the blue b5"
    assert notes[-1].pitch % 12 == 11, "the answer ends on B, off the tonic"
    assert c.sax_name == "tenor" and c.sax.program == 66 and c.repairs == ()


@pytest.mark.parametrize("tonic", range(12))
def test_tonic_sits_in_the_home_band(tonic):
    loop = loop_text(BASE + "A: w1 | w1 | w1 | w1\nB: w1 | w1 | w1 | h1 hr\n")
    for lead, home in (("tenor", (50, 61)), ("alto", (57, 68))):
        c = compile_loop(loop, Brief("mid", "wistful", 80, tonic), palette_with(sax_lead=lead))
        assert {x.pitch for x in c.sax.notes} == {home[0] + (tonic - home[0]) % 12}


def test_octave_fit_shifts_a_phrase_into_the_comfort_window():
    assert fit_shift([74, 77, 81, 74], (48, 72)) == -1
    assert fit_shift([60, 65], (48, 72)) == 0
    assert fit_shift([26] * 4, (48, 72)) == 0, "no shift helps: ties go to 0"
    assert fit_shift([47, 74, 47, 74], (48, 72)) == -1, "a tie between -1 and +1 goes down"
    loop = loop_text(BASE + "A: w1'' | w3'' | w5'' | w1''\nB: w1 | w3 | w5 | h1 hr\n")
    c = compile_loop(loop, EXAMPLE_BRIEF, dev_palette())
    a = [x.pitch for x in sorted(c.sax.notes, key=lambda x: x.step) if x.step < 64]
    assert a == [62, 65, 69, 62] and c.repairs == (), "written two octaves up, played one octave up, no fold"


def test_out_of_range_notes_fold_with_one_penalized_repair_each():
    loop = loop_text(BASE + "A: w1,, | w3 | w5 | w1,,\nB: w1 | w3 | w5 | h1 hr\n")
    c = compile_loop(loop, EXAMPLE_BRIEF, dev_palette())
    a = [x.pitch for x in sorted(c.sax.notes, key=lambda x: x.step) if x.step < 64]
    assert a == [50, 53, 57, 50]                          # D1 (26) folds up two octaves into 44-75
    assert [r.code for r in c.repairs] == ["sax_fold", "sax_fold"] and all(r.penalized for r in c.repairs)
    assert c.repairs[0].before.startswith("A w1,, in bar 1 step 1") and c.repairs[0].after == "MIDI 50"
    assert penalty(c.repairs) == 0.04


def test_every_fixture_stays_playable_in_every_key_mood_and_sax():
    pal = dev_palette()
    for name in passing_fixtures():
        for mood in MOOD_NAMES:
            loop = loop_of(name, mood)
            for lead in ("tenor", "alto"):
                voice = pal.sax[lead]
                for tonic in range(12):
                    pitched, repairs = sax_pitches(loop, tonic, Brief("mid", mood, 80, tonic).mode, voice)
                    assert all(voice.play[0] <= p <= voice.play[1] for *_, p in pitched), (name, mood, tonic)
                    assert all(r.code == "sax_fold" and r.penalized for r in repairs)


def test_the_fit_minimizes_notes_outside_comfort_with_the_stated_tie_order():
    for seed in range(300):
        hs = stream(seed, 9)
        pitches = [30 + below(h, 60) for h in hs[: 1 + below(hs[8], 8)]]
        k = fit_shift(pitches, (48, 72))
        out = {j: sum(not 48 <= p + 12 * j <= 72 for p in pitches) for j in (-1, 0, 1)}
        assert out[k] == min(out.values())
        assert k == next(j for j in (0, -1, 1) if out[j] == min(out.values()))


def test_one_sax_plays_every_loop_whatever_the_sax_line_says():
    loop = loop_of("shaker_edits.ls", "aeolian")
    assert loop.sax == "alto"
    brief = Brief("mid", "melancholy", 84, 9)
    c = compile_loop(loop, brief, dev_palette())
    assert (c.sax_name, c.sax.program) == ("tenor", 66)
    alto = compile_loop(loop, brief, palette_with(sax_lead="alto"))
    assert (alto.sax_name, alto.sax.program) == ("alto", 65)
    assert all(57 <= x.pitch <= 80 for x in alto.sax.notes)


@pytest.mark.parametrize("name", ["example.ls", "span1_dotted.ls", "shaker_edits.ls", "one_bar_base.ls"])
def test_note_off_15_ms_before_a_note_that_follows_and_at_the_step_end_before_a_rest(name):
    loop = loop_of(name)
    c = compile_loop(loop, EXAMPLE_BRIEF, dev_palette())
    clock = sax_clock(c.n, dev_palette().groove, dev_palette().sax_expr)
    seq = timed(loop.a, 0) + timed(loop.b, 64)
    by_step = {x.step: x for x in c.sax.notes}
    for i, (k, item) in enumerate(seq):
        if item.rest:
            continue
        note = by_step[k]
        nk, nxt = seq[(i + 1) % len(seq)]
        end = clock.sample(k + item.dur)
        if nxt.rest:
            assert note.off == end
        else:
            assert end - note.off in (661, 662), "15 ms is 661.5 samples"


def test_velocity_accents_and_channels():
    loop = loop_of("span1_dotted.ls")
    c = compile_loop(loop, EXAMPLE_BRIEF, dev_palette())
    seq = [(k, it) for k, it in timed(loop.a, 0) + timed(loop.b, 64) if not it.rest]
    firsts = {min(k for k, _ in seq if k < 64), min(k for k, _ in seq if k >= 64)}
    by_step = {x.step: x for x in c.sax.notes}
    for k, it in seq:
        want = 92 + 8 * (k % 16 == 0) + 4 * (k % 16 == 8) - 10 * (it.dur == 1) + 6 * (k in firsts)
        assert by_step[k].velocity == max(60, min(112, want)), k
    chans = [x.channel for x in sorted(c.sax.notes, key=lambda x: x.step)]
    assert chans == [(2, 3)[i % 2] for i in range(len(chans))]
    flat = compile_loop(loop, EXAMPLE_BRIEF, palette_with(expr_enabled=False))
    assert {x.velocity for x in flat.sax.notes} == {92}


EXAMPLE_TEXT = (FIX / "example.ls").read_text()
ODD_LINES = {
    # 9 notes: B's held w5' runs straight into A's first note (instrument.md section 3, the line as a cycle)
    "9_notes": BASE + "A: h5 h3 | w1 | w1 | w5\nB: w3 | w2 | w1 | w5'\n",
    # 29 notes: example.ls with B's closing qr played as q2, which then runs into A's q5 (a scooped leap)
    "29_notes": EXAMPLE_TEXT.replace("| h.6 qr", "| h.6 q2"),
}


def _quarters(count: int) -> str:
    """A 64-step phrase of `count` quarter notes, then quarter rests."""
    return " ".join(["q5", "q3", "q2", "q1"][i % 4] for i in range(count)) + " qr" * (16 - count)


@pytest.mark.parametrize("name", sorted(ODD_LINES))
def test_sax_channels_alternate_around_the_whole_cycle(name):
    # Bend and CC11 are channel-wide, so a note's curves must never share a channel with the note before it,
    # B's last note included: the line is read as a cycle (docs/COMPILER.md section 3.7).
    text = ODD_LINES[name]
    assert "w5'" in text or "h.6 q2" in text
    c = compile_loop(loop_text(text), EXAMPLE_BRIEF, dev_palette())
    line = sorted(c.sax.notes, key=lambda x: x.step)
    assert len(line) % 2 == 1
    clash = [(a.step, b.step) for a, b in zip(line, line[1:] + line[:1]) if a.channel == b.channel]
    assert not clash, f"consecutive notes on one channel: {clash}"
    first, last = line[0], line[-1]
    assert c.n - last.off in (661, 662), "B's last note runs into A's first, 15 ms before it"
    assert {p.channel for p in c.sax.bend + c.sax.cc11 if p.at == first.on} == {first.channel}
    assert set(c.sax.channels) == {x.channel for x in line}, "every channel that plays is set up at t = 0"


@pytest.mark.parametrize("a, b", [(1, 1), (1, 2), (2, 1), (2, 3), (3, 3), (4, 5), (5, 5), (7, 6), (16, 15), (16, 16)])
def test_no_two_cyclically_adjacent_sax_notes_share_a_channel(a, b):
    c = compile_loop(loop_text(BASE + f"A: {_quarters(a)}\nB: {_quarters(b)}\n"), EXAMPLE_BRIEF, dev_palette())
    line = sorted(c.sax.notes, key=lambda x: x.step)
    assert len(line) == a + b
    assert all(x.channel != y.channel for x, y in zip(line, line[1:] + line[:1])), [x.channel for x in line]
    assert {x.channel for x in line} == ({2, 3} if (a + b) % 2 == 0 else {2, 3, 4})
    assert c.sax.channels == tuple(sorted({x.channel for x in line}))


def test_timing_feels():
    loop = loop_of("example.ls")
    n = loop_samples(EXAMPLE_BRIEF.bpm)
    grid = {x.step: x.on for x in compile_loop(loop, EXAMPLE_BRIEF, dev_palette()).sax.notes}
    lag = {x.step: x.on for x in compile_loop(loop, EXAMPLE_BRIEF, palette_with(expr_timing="lag")).sax.notes}
    assert all(lag[k] - grid[k] in (1102, 1103) for k in grid), "lag_ms 25 = 1102.5 samples"
    swung = {x.step: x.on for x in compile_loop(loop, EXAMPLE_BRIEF, palette_with(expr_timing="swung")).sax.notes}
    for k, on in swung.items():
        straight = round(k * n / 128)
        if k % 4 == 0:
            assert on == straight
        elif k % 4 == 2:
            assert abs(on - straight - 0.1 * n / 32) <= 1, "the off-8th gets (0.6 - 0.5) of a beat more"
    assert any(k % 4 == 2 for k in swung)
    odd = [k for k in grid if k % 2]
    assert all(abs(grid[k] - round(k * n / 128) - 0.05 * n / 64) <= 1 for k in odd), "grid: the band's 16th swing"
    assert replace(dev_palette().sax_expr, timing="grid") == dev_palette().sax_expr
