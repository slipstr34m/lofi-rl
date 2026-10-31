"""Timeline and swing (instrument.md section 3, Timeline and swing; Chord changes)."""
import math
from fractions import Fraction

import pytest

from lofi_rl.compile import stub
from lofi_rl.compile.timeline import Clock, Segment, bpm_eff, loop_samples, ms, rnd, segments, wrap
from lofi_rl.dsl.ast import Chord

A, B = Chord("I", "m9"), Chord("IV", "13")
SPAN2 = (A, A, B, B, A, A, B, B)


@pytest.mark.parametrize("bpm", range(60, 96))
def test_n_is_64_aligned_and_matches_the_formula(bpm):
    n = loop_samples(bpm)
    assert n % 64 == 0
    assert n == 64 * ((2 * 1_323_000 + bpm) // (2 * bpm))          # 64 floor(1,323,000/BPM + 1/2), in integers
    assert abs(n - 32 * 60 / bpm * 44100) <= 32
    assert n == stub.loop_samples(bpm)                              # the Phase-0 chain agrees
    assert bpm_eff(n) == 84_672_000 / n and abs(bpm_eff(n) - bpm) < 0.15


def _reference(k, n, s, offset_ms):
    """t_k = k N/128 + [k odd] (s - 0.5) N/64 + offset, in floating point."""
    return k * n / 128 + (k % 2) * (s - 0.5) * n / 64 + offset_ms * 44.1


@pytest.mark.parametrize("bpm", range(60, 96))
def test_swing_sample_positions_match_the_formula(bpm):
    # Every tempo: at 61, 62, 66, 72, 73, 79, 81, 85, 87 and 89 BPM, N/128 is a half sample, so a step's position
    # is not a whole number of samples before the swing is added.
    n = loop_samples(bpm)
    for swing, offset_ms in ((0.5, 0.0), (0.55, 0.0), (0.58, 0.0), (0.6, -4.0), (0.66, 7.5), (0.54, 2.25)):
        clock = Clock(n, swing=Fraction(str(swing)), offset=ms(Fraction(str(offset_ms))))
        for k in range(128):
            ref = _reference(k, n, swing, offset_ms)
            if abs(ref - math.floor(ref) - 0.5) < 1e-6:             # an exact half: round half up
                assert clock.sample(k) == math.floor(ref) + 1, (swing, offset_ms, k, ref)
            else:
                assert clock.sample(k) == math.floor(ref + 0.5), (swing, offset_ms, k, ref)


def test_ten_tempos_put_the_steps_on_half_samples():
    half = [bpm for bpm in range(60, 96) if loop_samples(bpm) % 128]
    assert half == [61, 62, 66, 72, 73, 79, 81, 85, 87, 89] and all(loop_samples(b) % 128 == 64 for b in half)
    n = loop_samples(87)                                            # 973,248 samples
    assert Clock(n).time(127) == Fraction(1_931_289, 2) and Clock(n).sample(127) == 965_645   # 965,644.5 rounds up
    assert Clock(n, Fraction(11, 20)).time(127) == Fraction(1_931_289, 2) + Fraction(n, 1280)  # + 760.35 of swing
    assert Clock(n, Fraction(11, 20)).sample(127) == 966_405


def test_only_odd_steps_are_delayed_by_s_minus_half_times_30_over_bpm():
    n = loop_samples(80)
    straight, swung = Clock(n), Clock(n, swing=Fraction(55, 100))
    for k in range(128):
        delay = swung.sample(k) - straight.sample(k)
        assert delay == 0 if k % 2 == 0 else abs(delay - 0.05 * 30 / 80 * 44100) <= 1
    assert [stub.step_sample(k, n) for k in range(0, 128, 2)] == [swung.sample(k) for k in range(0, 128, 2)]


def test_clock_is_periodic_and_monotonic():
    n = loop_samples(87)
    nudge = tuple(Fraction((k * 37) % 200 - 100, 3) for k in range(128))
    for clock in (Clock(n, Fraction(3, 4), ms(-12)), Clock(n, offset=ms(30), swing8=Fraction(3, 5)),
                  Clock(n, Fraction(11, 20), nudge=nudge)):
        for k in range(-130, 260):
            assert clock.time(k + 128) == clock.time(k) + n
        assert all(clock.time(k) < clock.time(k + 1) for k in range(-1, 129))


def test_eighth_note_swing_places_the_off_eighth_and_the_sixteenths_between():
    n = loop_samples(80)
    q = n // 32                                                     # one beat, exact since 64 | n
    s8 = Fraction(3, 5)
    clock = Clock(n, swing8=s8)
    assert [clock.time(k) for k in range(5)] == [0, s8 * q / 2, s8 * q, s8 * q + (1 - s8) * q / 2, q]
    assert all(Clock(n, swing8=Fraction(1, 2)).time(k) == Clock(n).time(k) for k in range(128))


def test_rounding_and_milliseconds():
    assert ms(15) == Fraction(1323, 2) and rnd(ms(15)) == 662          # 661.5 rounds half up
    assert rnd(Fraction(-1, 2)) == 0 and rnd(Fraction(-3, 2)) == -1 and rnd(Fraction(7, 3)) == 2
    assert wrap(-176, 5000, 1000) == (824, 6000) and wrap(1500, 1700, 1000) == (500, 700)


def _cells(*bars):
    return "".join(bars)


REST = "." * 16


def test_a_hold_across_a_chord_change_is_cut_and_re_struck():
    # bar 2 holds into bar 3, where the chord changes: cut at step 32, re-struck there at the same level.
    cells = _cells(REST, "........X-------", "--------........", REST, REST, REST, REST, REST)
    assert segments(cells, SPAN2) == (Segment(24, 32, "X"), Segment(32, 40, "X"))
    # the same hold inside one chord (bar 1 into bar 2) is one note
    cells = _cells("........x-------", "----....", ) + "." * 8 + REST * 6
    assert segments(cells, SPAN2) == (Segment(8, 20, "x"),)


def test_a_hold_wraps_the_seam_and_is_re_struck_at_bar_1():
    cells = _cells("--..............", REST, REST, REST, REST, REST, REST, "............g---")
    assert segments(cells, SPAN2) == (Segment(0, 2, "g"), Segment(124, 128, "g"))


def test_holds_without_a_strike_are_silent_and_runs_stop_at_the_next_cell():
    cells = _cells("----x--.x-.X----", REST, REST, REST, REST, REST, REST, REST)
    assert segments(cells, SPAN2) == (Segment(4, 7, "x"), Segment(8, 10, "x"), Segment(11, 16, "X"))
    one = "x" + "-" * 127                                           # one strike held round the whole loop
    assert segments(one, (A,) * 8) == (Segment(0, 128, "x"),)
    assert [s.start for s in segments(one, SPAN2)] == [0, 32, 64, 96]
