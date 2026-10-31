"""Drum hits (instrument.md section 3, Velocities and hat/shaker metric offsets; Drum sampler: layers,
round-robin, choke, as amended by review T7 and T8 in docs/COMPILER.md §3.9; per-lane swing and offsets from the
palette)."""
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
HATS_TEXT = """vamp: Im9 IV13 span 2
keys: rhodes rootless
comp: x---------------
bass: lock
drums: jazz
kick: x.......x.......
snare: ....x.......x...
{hats}
A: w5 | w3 | w1 | w2
B: w5 | w3 | w1 | h2 hr
"""                             # a minimal loop; {hats} takes the hat, ohat or shaker lines under test


def test_gains_by_level_with_the_hat_and_shaker_metric_offsets():
    for name in ("example.ls", "shaker_edits.ls"):
        c = compiled(name)
        for d in c.drums:
            metric = 10 ** (METRIC_DB[d.step % 16] / 20) if d.lane in ("hat", "shaker") else 1.0
            assert abs(d.gain - GAIN[d.level] * metric) < 1e-12, (name, d)
    shaker = [d for d in compiled("shaker_edits.ls").drums if d.lane == "shaker"]
    assert shaker and {d.step % 16 for d in shaker} >= {0, 2}


def _rr_palette(size):
    """The dev palette with `size` one-shots in every layer of every lane (its GU note at `size` velocities)."""
    pal = dev_palette()
    kit = pal.kits["jazz"]
    lanes = {name: replace(drum, layers=MappingProxyType(
        {layer: tuple(replace(shots[0], vel=shots[0].vel - i) for i in range(size)) for layer, shots in
         drum.layers.items()})) for name, drum in kit.lanes.items()}
    return replace(pal, kits=MappingProxyType({"jazz": replace(kit, lanes=MappingProxyType(lanes))}))


def _rr_runs(c):
    """(lane, layer) -> the round-robin index of each of its hits in one cycle, in step order."""
    runs = {}
    for d in sorted(c.drums, key=lambda d: d.step):
        runs.setdefault((d.lane, d.layer), []).append(d.rr)
    return runs


def _check_round_robin(c, size):
    """docs/COMPILER.md §3.9 (review T8): hit h of the `count` hits a lane-layer plays per cycle takes one-shot
    h mod R, except that with count = 1 (mod R), count > 1 and R >= 3 the last takes one-shot 1. So no two hits in
    a row play the same one-shot, across the seam too, except a lone hit and an odd count of two one-shots (two
    cannot alternate around an odd cycle): those keep exactly one repeat."""
    for key, rr in _rr_runs(c).items():
        count = len(rr)
        plain = [h % size for h in range(count)]
        assert rr == (plain[:-1] + [1] if size >= 3 and count > 1 and count % size == 1 else plain), (key, size, rr)
        repeats = sum(a == b for a, b in zip(rr, rr[1:] + rr[:1]))      # the last hit's neighbour: the next first
        assert repeats == int(count == 1 or (size == 2 and count % 2 == 1)), (key, size, rr)


@pytest.mark.parametrize("size", [2, 3, 4])
def test_layers_and_round_robin_in_step_order(size):
    c = compile_loop(loop_of("example.ls"), EXAMPLE_BRIEF, _rr_palette(size))
    runs = _rr_runs(c)
    counts = {key: len(rr) for key, rr in runs.items()}
    assert counts == {("kick", "med"): 28, ("rim", "hard"): 16, ("rim", "soft"): 8, ("hat", "med"): 69,
                      ("ohat", "med"): 5}
    _check_round_robin(c, size)
    # 28 kicks and 16 accented rims are 1 mod 3; 69 closed hats and 5 open hats are 1 mod 4.
    lasts = {key: rr[-1] for key, rr in runs.items()}
    assert lasts == {2: {("kick", "med"): 1, ("rim", "hard"): 1, ("rim", "soft"): 1, ("hat", "med"): 0,
                         ("ohat", "med"): 0},
                     3: {("kick", "med"): 1, ("rim", "hard"): 1, ("rim", "soft"): 1, ("hat", "med"): 2,
                         ("ohat", "med"): 1},
                     4: {("kick", "med"): 3, ("rim", "hard"): 3, ("rim", "soft"): 3, ("hat", "med"): 1,
                         ("ohat", "med"): 1}}[size]
    assert {d.layer for d in c.drums if d.level == "X"} == {"hard"}


@pytest.mark.parametrize("size", [2, 3, 4])
def test_round_robin_does_not_repeat_across_the_seam_where_it_can_avoid_it(size):
    # Review T8: with rr = h mod R, a lane-layer of count = 1 (mod R) hits a cycle plays one-shot 0 last and first
    # again after the seam. 1 to 9 shaker hits in bar 1 cover count = 1 (mod R) and every other count, R = 2, 3, 4.
    pal = _rr_palette(size)
    for count in range(1, 10):
        c = compile_loop(loop_text(HATS_TEXT.format(hats=f"@1 shaker: {'x' * count:.<16}")), EXAMPLE_BRIEF, pal)
        assert len(_rr_runs(c)[("shaker", "med")]) == count
        _check_round_robin(c, size)


def _next_step(steps, after):
    """The first of `steps` strictly after step `after`, read cyclically (the same step comes last)."""
    return min(steps, key=lambda k: (k - after - 1) % 128)


CHOKING = ("hat", "ohat")       # dev_gu: a hi-hat is one instrument, so either hat cuts an open hat (review T7)


def _expected_chokes(c, pal, choking=CHOKING):
    """ohat step -> samples from its onset to where it is cut (docs/COMPILER.md §3.9): for each choking lane, its
    first hit on a strictly later grid step, read cyclically (the same step one cycle on comes last), that sounds
    after the open hat; the earliest of those. None when no choking lane has a hit."""
    human = Humanize.from_groove(pal.groove, c.seed)
    start = lane_clock("ohat", c.n, pal, human)
    out = {}
    for d in (d for d in c.drums if d.lane == "ohat"):
        cuts = []
        for lane in choking:
            clock = lane_clock(lane, c.n, pal, human)
            later = sorted(k if k > d.step else k + 128 for k in {h.step for h in c.drums if h.lane == lane})
            cuts += [t for t in map(clock.sample, later) if t > start.sample(d.step)][:1]
        out[d.step] = min(cuts) - start.sample(d.step) if cuts else None
    return out


def _rings(c):
    """ohat step -> samples from its onset to its choke (None: nothing cuts it)."""
    return {d.step: None if d.choke_at is None else d.choke_at - d.on for d in c.drums if d.lane == "ohat"}


def _same_step_hat_lands_after(c):
    """True when a closed hat on an open hat's grid step sounds after it: the case the step rule exists for."""
    hats = {d.step: d.on for d in c.drums if d.lane == "hat"}
    return any(d.step in hats and hats[d.step] > d.on for d in c.drums if d.lane == "ohat")


def _old_choke():
    """The dev palette with its kit's choke list as it was before review T7: [[ohat, hat]] only."""
    pal = dev_palette()
    kit = pal.kits["jazz"]
    return replace(pal, kits=MappingProxyType({"jazz": replace(kit, choke=(("ohat", "hat"),))}))


def test_open_hat_is_choked_by_the_next_closed_hat_across_the_seam():
    c = compiled("example.ls")
    hat_steps = [d.step for d in c.drums if d.lane == "hat"]
    hats = {d.step: d.on for d in c.drums if d.lane == "hat"}
    ohats = [d for d in c.drums if d.lane == "ohat"]
    assert len(ohats) == 5 and 30 in hats, "a closed hat shares step 30 with an open hat"
    for d in ohats:
        k = _next_step(hat_steps, d.step)
        assert d.choke_at == hats[k] + (c.n if k <= d.step else 0), d
    assert _rings(c) == _expected_chokes(c, dev_palette())
    last = max(ohats, key=lambda d: d.on)
    assert last.step == 126 and last.choke_at == c.n + hats[0], "bar 8's open hat rings into bar 1"


@pytest.mark.parametrize("mode", ["bar", "loop"])
def test_a_closed_hat_on_the_same_step_never_chokes_under_humanize(mode):
    # instrument.md §3: humanize only nudges timing (±3 ms); which hat chokes is decided by grid step.
    pal = palette_with(groove_humanize=mode)
    c = compile_loop(loop_of("example.ls"), EXAMPLE_BRIEF, pal)
    assert _same_step_hat_lands_after(c), "the case under test: a same-step closed hat sounds after the open hat"
    rings = _rings(c)
    assert rings == _expected_chokes(c, pal)
    assert min(rings.values()) > 15_000, f"open hats cut within a step of their onset: {rings}"


@pytest.mark.parametrize("swing, humanize", [({"hat": Fraction(3, 5), "ohat": Fraction(27, 50)}, "off"),
                                             ({}, "bar"), ({}, "loop")])
def test_per_lane_swing_or_humanize_never_lets_a_same_step_hat_choke(swing, humanize):
    # The R3 shootout's "hats 60 with the rest at 54" (instrument.md §6): on an odd step the closed hat lands
    # after the open hat that shares its step, yet only the next step's hat may cut it. Under H1 and H2 the
    # humanize seed of this loop (its B phrase chosen for it) puts some same-step closed hats after theirs too.
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
B: w4 | w3 | w2 | h1 hr
"""
    pal = palette_with(groove_swing_lanes=MappingProxyType(swing), groove_humanize=humanize)
    c = compile_loop(loop_text(text), EXAMPLE_BRIEF, pal)
    assert _same_step_hat_lands_after(c), "the case under test: a same-step closed hat sounds after the open hat"
    rings = _rings(c)
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


def _ohats(c):
    return sorted((d for d in c.drums if d.lane == "ohat"), key=lambda d: d.step)


def test_consecutive_open_hats_cut_each_other():
    # Review T7: a hi-hat is one instrument, so an open hat ends at the next hat hit of either kind. With no closed
    # hat in the loop, each open hat rings until the next one, and bar 8's last until bar 1's first, across the seam.
    loop = loop_text(HATS_TEXT.format(hats="ohat: ..x...x...x...x."))
    c = compile_loop(loop, EXAMPLE_BRIEF, dev_palette())
    ohats = _ohats(c)
    assert [d.step for d in ohats] == list(range(2, 128, 4))
    for d, nxt in zip(ohats, ohats[1:] + ohats[:1]):
        assert d.choke_at == nxt.on + (c.n if nxt.step < d.step else 0), d
    assert _rings(c) == _expected_chokes(c, dev_palette())
    assert set(_rings(compile_loop(loop, EXAMPLE_BRIEF, _old_choke())).values()) == {None}, "[[ohat, hat]] alone"
    # Two open hats, then a closed hat: the first is cut by the second, the second by the closed hat.
    c = compile_loop(loop_text(HATS_TEXT.format(hats="hat: x.......x.......\nohat: ..x.x...........")),
                     EXAMPLE_BRIEF, dev_palette())
    hats = {d.step: d.on for d in c.drums if d.lane == "hat"}
    ohats = {d.step: d for d in c.drums if d.lane == "ohat"}
    assert len(ohats) == 16 and _rings(c) == _expected_chokes(c, dev_palette())
    for bar in range(0, 128, 16):
        assert ohats[bar + 2].choke_at == ohats[bar + 4].on and ohats[bar + 4].choke_at == hats[bar + 8]


def test_a_lone_open_hat_still_rings_to_the_next_closed_hat():
    # An open hat with a closed hat before the next open hat is cut by the closed hat, as before review T7.
    loop = loop_text(HATS_TEXT.format(hats="hat: x.......x.......\nohat: ....x..........."))
    c = compile_loop(loop, EXAMPLE_BRIEF, dev_palette())
    hats = {d.step: d.on for d in c.drums if d.lane == "hat"}
    assert [(d.step, d.choke_at) for d in _ohats(c)] == [(k, hats[k + 4]) for k in range(4, 128, 16)]
    assert compile_loop(loop, EXAMPLE_BRIEF, _old_choke()).drums == c.drums


def test_an_open_hat_with_no_other_hat_rings_until_it_strikes_again():
    # With no other hat hit in the loop, the next hat hit is the open hat itself one cycle on (its own step comes
    # last, as for the closed hat), so it rings for the whole cycle. The sampler leaves a one-shot shorter than the
    # cycle alone (render/sampler.choke). Without the [ohat, ohat] pair nothing cuts it.
    loop = loop_text(HATS_TEXT.format(hats="@7 ohat: ......x........."))
    c = compile_loop(loop, EXAMPLE_BRIEF, dev_palette())
    (d,) = _ohats(c)
    assert d.step == 102 and d.choke_at == d.on + c.n
    assert _rings(c) == _expected_chokes(c, dev_palette()) == {102: c.n}
    assert _rings(compile_loop(loop, EXAMPLE_BRIEF, _old_choke())) == {102: None}


def test_the_earliest_cut_of_either_hat_wins():
    # With two choking lanes the open hat ends at whichever of their next hits (each on a strictly later grid step)
    # sounds first. Extreme lane offsets and swing (within the palette's ranges) let the open hat on step 6 sound
    # before the closed hat on step 5, so the open hat on step 3 ends at step 6's, not at the closed hat.
    loop = loop_text(HATS_TEXT.format(hats="hat: .....x..........\nohat: ...x..x........."))
    pal = palette_with(groove_swing_lanes=MappingProxyType({"hat": Fraction(3, 4)}),
                       groove_offset_ms=MappingProxyType({"hat": Fraction(50), "ohat": Fraction(-50)}))
    c = compile_loop(loop, replace(EXAMPLE_BRIEF, bpm=95), pal)
    hats = {d.step: d.on for d in c.drums if d.lane == "hat"}
    ohats = {d.step: d for d in c.drums if d.lane == "ohat"}
    assert ohats[3].on < ohats[6].on < hats[5], "the open hat on step 6 sounds before the closed hat on step 5"
    assert ohats[3].choke_at == ohats[6].on
    assert _rings(c) == _expected_chokes(c, pal)


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
