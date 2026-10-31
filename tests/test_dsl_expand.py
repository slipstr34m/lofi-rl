"""8-bar expansion: chords per bar, bases tiled, edits applied, `bass: lock` resolved, timed items."""
from pathlib import Path

from lofi_rl.dsl import BARS, STEPS, Chord, chord_bars, expand, lock_bass, parse, timed

FIX = Path(__file__).parent / "fixtures" / "loops"
REST = "." * 16


def load(name: str):
    res = parse((FIX / name).read_text())
    assert res.gate.ok, res.gate
    return res.ast


def onsets(bars, levels="Xxg"):
    return {(b, s) for b, bar in enumerate(bars) for s, c in enumerate(bar) if c in levels}


def test_shape_is_8_bars_of_16_cells_for_every_lane():
    bars = expand(load("example.ls"))
    assert set(bars.drums) == {"kick", "snare", "rim", "hat", "ohat", "shaker"}
    for grid in [bars.comp, bars.bass, *bars.drums.values()]:
        assert len(grid) == BARS and all(len(bar) == STEPS for bar in grid)
    assert bars.drums["snare"] == (REST,) * 8 and bars.drums["shaker"] == (REST,) * 8


def test_example_two_bar_bases_tile_and_edits_replace_one_bar():
    loop = load("example.ls")
    bars = expand(loop)
    kick0, kick1 = loop.lane("kick")
    assert bars.drums["kick"] == (kick0, kick1, kick0, "x......x..x...x.", kick0, kick1, kick0, kick1)
    assert bars.drums["hat"][2] == "x.x.x.x.x.x.x.xx" and bars.drums["hat"][6] == loop.lane("hat")[0]
    assert bars.drums["rim"][5] == "....X..g....X.g."
    assert bars.drums["ohat"][6] == "......x........."
    assert bars.comp[4] == "x-----.x--x-..x-" and bars.comp[7] == "x-----.x----x-x-"
    assert bars.bass == loop.bass * 4


def test_example_chords_span_2():
    bars = expand(load("example.ls"))
    i, iv = Chord("I", "m9"), Chord("IV", "13")
    assert bars.chords == (i, i, iv, iv, i, i, iv, iv)


def test_span_1_alternates_every_bar():
    loop = load("span1_dotted.ls")
    first, second = loop.vamp.chords
    assert chord_bars(loop.vamp) == (first, second) * 4
    assert expand(loop).drums["snare"][4] == "....x..g....x..."


def test_example_claims_from_section_1():
    bars = expand(load("example.ls"))
    bass = onsets(bars.bass)
    kick = onsets(bars.drums["kick"], "Xx")
    assert len(bass) == 24 and len(bass & kick) == 19        # "19 of 24 bass onsets"
    sigs = [(bars.chords[b], tuple(bars.drums[k][b] for k in sorted(bars.drums)),
             bars.bass[b], bars.comp[b]) for b in range(8)]
    assert len(set(sigs)) == 8                               # bars 1-8 all different
    assert all(bars.drums[k][7] == bars.drums[k][1] for k in bars.drums)   # bar 8 = base drums


def test_one_bar_bases_repeat_in_every_bar():
    loop = load("one_bar_base.ls")
    bars = expand(loop)
    for grid in [bars.comp, bars.bass, *bars.drums.values()]:
        assert len(set(grid)) == 1
    assert bars.drums["kick"] == (loop.lane("kick")[0],) * 8


def test_lock_bass_follows_kicks_and_chord_changes_with_edits():
    bars = expand(load("shaker_edits.ls"))
    assert bars.bass == (
        "X-----x---x-----",          # kick X x x, held to the next onset
        "x-------x-------",          # @2 bass replaces the lock
        "X-----x---x-----",
        "X------x--x-----",
        "X-----x---x-----",
        "X------x--x-----",
        "X-----x-..x-----",          # @7 bass
        "X-----x---x-----",          # @8 kick ghost on step 15 is not an onset
    )
    assert bars.drums["shaker"][5] == REST and bars.drums["kick"][7] == "X.....x...x...g."
    assert bars.comp[3] == "x--.x--.x--....."


def test_lock_forces_an_onset_at_each_chord_change_and_holds_at_most_8():
    a, b = Chord("I", "m7"), Chord("IV", "7")
    ghosts = ("g.......g.......",) * 8
    span2 = (a, a, b, b, a, a, b, b)
    assert lock_bass(ghosts, span2) == ("x-------........", REST) * 4
    assert lock_bass(ghosts, (a, b) * 4) == ("x-------........",) * 8


def test_lock_hold_is_cut_by_the_next_onset_and_crosses_bar_lines():
    a, b = Chord("I", "m7"), Chord("IV", "7")
    kick = ("x.X..........x..",) + (REST,) * 7
    bass = lock_bass(kick, (a, a, b, b, a, a, b, b))
    assert bass[0] == "x-X-------...x--"                     # 2 steps, then 8, then 8 more
    assert bass[1] == "-----..........."                     # the last note held over the bar line
    assert bass[2] == "x-------........"                     # forced onset at the chord change
    assert bass[3] == REST


def test_timed_items_give_onsets_on_the_loop_clock():
    loop = load("example.ls")
    a = timed(loop.a)
    assert [t for t, _ in a] == [0, 4, 6, 8, 12, 14, 16, 24, 26, 28, 30, 32, 44, 46, 48, 52, 56]
    b = timed(loop.b, 64)
    assert b[0][0] == 64 and b[-1][0] + b[-1][1].dur == 128
    assert [t for t, item in b if item.rest] == [80, 96, 124]
