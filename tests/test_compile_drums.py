"""Drum hits (instrument.md section 3, Velocities and hat/shaker metric offsets; Drum sampler: layers,
round-robin, choke; per-lane swing and offsets from the palette)."""
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType

import pytest

from helpers_compile import EXAMPLE_BRIEF, compiled, dev_palette, loop_of, loop_text, palette_with
from lofi_rl.compile.compiler import compile_loop, lane_clock
from lofi_rl.compile.humanize import Humanize
from lofi_rl.compile.timeline import Clock, ms

GAIN = {"X": 1.0, "x": 0.71, "g": 0.32}
METRIC_DB = [0, -4, -2, -4, -1, -4, -2, -4, 0, -4, -2, -4, -1, -4, -2, -4]   # steps 1..16 of a bar


def test_gains_by_level_with_the_hat_and_shaker_metric_offsets():
    for name in ("example.ls", "shaker_edits.ls"):
        c = compiled(name)
        for d in c.drums:
            metric = 10 ** (METRIC_DB[d.step % 16] / 20) if d.lane in ("hat", "shaker") else 1.0
            assert abs(d.gain - GAIN[d.level] * metric) < 1e-12, (name, d)
    shaker = [d for d in compiled("shaker_edits.ls").drums if d.lane == "shaker"]
    assert shaker and {d.step % 16 for d in shaker} >= {0, 2}


def test_layers_and_round_robin_in_step_order():
    pal = dev_palette()
    kit = pal.kits["jazz"]
    hat = kit.lanes["hat"]
    three = tuple(replace(hat.layers["med"][0], vel=v) for v in (76, 80, 84))
    lanes = dict(kit.lanes)
    lanes["hat"] = replace(hat, layers=MappingProxyType({**hat.layers, "med": three}))
    rr_pal = replace(pal, kits=MappingProxyType({"jazz": replace(kit, lanes=MappingProxyType(lanes))}))
    c = compile_loop(loop_of("example.ls"), EXAMPLE_BRIEF, rr_pal)
    for layer, size in (("med", 3), ("hard", 1), ("soft", 1)):
        hits = sorted((d for d in c.drums if d.lane == "hat" and d.layer == layer), key=lambda d: d.step)
        assert [d.rr for d in hits] == [i % size for i in range(len(hits))]
    assert {d.layer for d in c.drums if d.level == "X"} == {"hard"}


def _next_step(steps, after):
    """The first of `steps` strictly after step `after`, read cyclically (the same step comes last)."""
    return min(steps, key=lambda k: (k - after - 1) % 128)


def _expected_chokes(c, pal):
    """ohat step -> samples from its onset to the onset of the closed hat on the next later grid step."""
    human = Humanize.from_groove(pal.groove, c.seed)
    hat, ohat = (lane_clock(lane, c.n, pal, human) for lane in ("hat", "ohat"))
    steps = [d.step for d in c.drums if d.lane == "hat"]
    out = {}
    for d in (d for d in c.drums if d.lane == "ohat"):
        k = _next_step(steps, d.step)
        out[d.step] = hat.sample(k if k > d.step else k + 128) - ohat.sample(d.step)
    return out


def test_open_hat_is_choked_by_the_next_closed_hat_across_the_seam():
    c = compiled("example.ls")
    hat_steps = [d.step for d in c.drums if d.lane == "hat"]
    hats = {d.step: d.on for d in c.drums if d.lane == "hat"}
    ohats = [d for d in c.drums if d.lane == "ohat"]
    assert len(ohats) == 5 and 30 in hats, "a closed hat shares step 30 with an open hat"
    for d in ohats:
        k = _next_step(hat_steps, d.step)
        assert d.choke_at == hats[k] + (c.n if k <= d.step else 0), d
    assert {d.step: d.choke_at - d.on for d in ohats} == _expected_chokes(c, dev_palette())
    last = max(ohats, key=lambda d: d.on)
    assert last.step == 126 and last.choke_at == c.n + hats[0], "bar 8's open hat rings into bar 1"


@pytest.mark.parametrize("mode", ["bar", "loop"])
def test_a_closed_hat_on_the_same_step_never_chokes_under_humanize(mode):
    # instrument.md §3: humanize only nudges timing (±3 ms); which hat chokes is decided by grid step.
    pal = palette_with(groove_humanize=mode)
    c = compile_loop(loop_of("example.ls"), EXAMPLE_BRIEF, pal)
    rings = {d.step: d.choke_at - d.on for d in c.drums if d.lane == "ohat"}
    assert rings == _expected_chokes(c, pal)
    assert min(rings.values()) > 15_000, f"open hats cut within a step of their onset: {rings}"


@pytest.mark.parametrize("swing, humanize", [({"hat": Fraction(3, 5), "ohat": Fraction(27, 50)}, "off"),
                                             ({}, "bar"), ({}, "loop")])
def test_per_lane_swing_or_humanize_never_lets_a_same_step_hat_choke(swing, humanize):
    # The R3 shootout's "hats 60 with the rest at 54" (instrument.md §6): on an odd step the closed hat lands
    # after the open hat that shares its step, yet only the next step's hat may cut it.
    text = """vamp: Im9 IV13 span 2
keys: rhodes rootless
comp: x---------------
bass: lock
drums: jazz
kick: x.......x.......
snare: ....x.......x...
hat: xxxxxxxxxxxxxxxx
ohat: ...x.......x....
A: w5 | w3 | w1 | w2
B: w5 | w3 | w1 | h2 hr
"""
    pal = palette_with(groove_swing_lanes=MappingProxyType(swing), groove_humanize=humanize)
    c = compile_loop(loop_text(text), EXAMPLE_BRIEF, pal)
    rings = {d.step: d.choke_at - d.on for d in c.drums if d.lane == "ohat"}
    assert len(rings) == 16 and rings == _expected_chokes(c, pal)
    step = c.n // 128
    assert all(0.7 * step < r < 1.3 * step for r in rings.values()), f"each open hat rings about one step: {rings}"


def test_a_later_step_hat_that_sounds_before_the_open_hat_does_not_cut_it():
    # Extreme lane offsets and swing (within the palette's ranges) can put the next step's closed hat ahead of
    # the open hat in time. It cannot cut a hat that has not started, so the one after it does.
    text = """vamp: Im9 IV13 span 2
keys: rhodes rootless
comp: x---------------
bass: lock
drums: jazz
kick: x.......x.......
snare: ....x.......x...
hat: xxxxxxxxxxxxxxxx
ohat: ...x............
A: w5 | w3 | w1 | w2
B: w5 | w3 | w1 | h2 hr
"""
    pal = palette_with(groove_swing_lanes=MappingProxyType({"ohat": Fraction(3, 4)}),
                       groove_offset_ms=MappingProxyType({"hat": Fraction(-45), "ohat": Fraction(45)}))
    brief = replace(EXAMPLE_BRIEF, bpm=95)
    c = compile_loop(loop_text(text), brief, pal)
    hats = {d.step: d.on for d in c.drums if d.lane == "hat"}
    for d in (d for d in c.drums if d.lane == "ohat"):
        assert hats[d.step + 1] < d.on < hats[d.step + 2], "the step-4 hat sounds first"
        assert d.choke_at == hats[d.step + 2], d


def test_no_closed_hat_means_no_choke():
    text = """vamp: Im9 IV13 span 2
keys: rhodes rootless
comp: x---------------
bass: lock
drums: jazz
kick: x.......x.......
snare: ....x.......x...
ohat: ..x...x...x...x.
A: w5 | w3 | w1 | w2
B: w5 | w3 | w1 | h2 hr
"""
    c = compile_loop(loop_text(text), EXAMPLE_BRIEF, dev_palette())
    assert [d.choke_at for d in c.drums if d.lane == "ohat"] == [None] * 32


def test_per_lane_swing_and_offsets_come_from_the_palette():
    pal = palette_with(groove_swing_lanes=MappingProxyType({"hat": Fraction(3, 5)}),
                       groove_offset_ms=MappingProxyType({"kick": Fraction(-4)}))
    c = compile_loop(loop_of("example.ls"), EXAMPLE_BRIEF, pal)
    hat, base, kick = Clock(c.n, Fraction(3, 5)), Clock(c.n, Fraction(11, 20)), Clock(c.n, Fraction(11, 20), ms(-4))
    for d in c.drums:
        clock = {"hat": hat, "kick": kick}.get(d.lane, base)
        assert d.on == clock.sample(d.step) % c.n, d
    first_kick = next(d for d in c.drums if d.lane == "kick" and d.step == 0)
    assert first_kick.on == c.n - 176, "a -4 ms nudge on step 0 wraps to the end of the cycle"
    assert c.drums[-1] == first_kick, "hits stay sorted by onset after the wrap"
    odd_hats = [d for d in c.drums if d.lane == "hat" and d.step % 2]
    assert odd_hats and all(abs(d.on - d.step * c.n / 128 - 0.1 * c.n / 64) <= 0.5 for d in odd_hats)
