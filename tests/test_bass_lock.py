"""The bass (instrument.md section 3, Registers and `bass: lock`; TASTE.md: the bass holds the roots)."""
from fractions import Fraction

from helpers_compile import EXAMPLE_BRIEF, compiled, dev_palette, loop_of, loop_text
from lofi_rl.brief import Brief
from lofi_rl.compile.bass import bass_octaves
from lofi_rl.compile.compiler import compile_loop, lane_clock
from lofi_rl.compile.humanize import Humanize
from lofi_rl.dsl.ast import expand
from lofi_rl.theory.chords import root_pc

LOCK = """vamp: Im9 IV13 span 2
keys: rhodes rootless
comp: x---------------
bass: lock
drums: jazz
kick: X...g...x.....x. x.......x.x.....
snare: ....x.......x...
hat: x.x.x.x.x.x.x.x.
A: w5 | w3 | w1 | w2
B: w5 | w3 | w1 | h2 hr
"""


def test_octave_pair_minimizes_the_cost_with_ties_to_the_lower_pair():
    for a in range(12):
        for b in range(12):
            pairs = [(pa, pb) for pa in range(28, 44) for pb in range(28, 44) if pa % 12 == a and pb % 12 == b]
            cost = lambda p: abs(p[0] - 36) + abs(p[1] - 36) + Fraction(1, 2) * abs(p[0] - p[1])   # noqa: E731
            low = min(cost(p) for p in pairs)
            best = min((p for p in pairs if cost(p) == low), key=lambda p: (p[0] + p[1], p[0]))
            assert bass_octaves(a, b) == best, (a, b)
    assert bass_octaves(2, 7) == (38, 31)                 # D and G: D2, G1
    assert bass_octaves(6, 0) == (30, 36)                 # F#: 30 and 42 tie on |p-36|; the lower pair wins


def _lock_reference(kick_bars, chords):
    """Onsets: non-ghost kicks plus every chord change (bar 8 into bar 1 too); hold to the next, at most 8."""
    flat = "".join(kick_bars)
    onsets = {i: c for i, c in enumerate(flat) if c in "Xx"}
    for bar in range(8):
        if chords[bar] != chords[bar - 1]:
            onsets.setdefault(16 * bar, "x")
    starts = sorted(onsets)
    return [(s, min(starts[(j + 1) % len(starts)] + (128 if j + 1 == len(starts) else 0), s + 8), onsets[s])
            for j, s in enumerate(starts)]


def test_lock_follows_the_non_ghost_kicks_and_forces_chord_changes():
    loop = loop_text(LOCK)
    pal = dev_palette()
    c = compile_loop(loop, EXAMPLE_BRIEF, pal)
    bars = expand(loop)
    ref = _lock_reference(bars.drums["kick"], bars.chords)
    clock = lane_clock("bass", c.n, pal, Humanize.from_groove(pal.groove, c.seed))
    vel = pal.kits["jazz"].bass.vel
    got = [(nt.step, nt.on, nt.off, nt.pitch, nt.velocity) for nt in c.bass.notes]
    want = [(s, clock.sample(s), clock.sample(e), c.bass_pitches[0 if bars.chords[s // 16] == loop.vamp.chords[0]
                                                                  else 1], vel[lv]) for s, e, lv in ref]
    assert got == want
    steps = {s for s, _, _ in ref}
    assert 4 not in steps and 20 not in steps, "ghost kicks give no bass onset"
    assert {0, 32, 64, 96} <= steps, "every chord change has an onset"
    assert {nt.velocity for nt in c.bass.notes if nt.step == 0} == {104}, "an X kick gives an X bass note"


def test_lock_after_edits_on_the_shaker_fixture():
    c = compiled("shaker_edits.ls", Brief("high", "melancholy", 88, 9))
    bars = expand(loop_of("shaker_edits.ls", "aeolian"))
    want = {16 * b + s for b, bar in enumerate(bars.bass) for s, cell in enumerate(bar) if cell in "Xxg"}
    assert {nt.step for nt in c.bass.notes} == want
    assert bars.bass[1] == "x-------x-------" and bars.bass[6] == "X-----x-..x-----", "@2 and @7 bass edits win"
    kick8 = {s for s, cell in enumerate(bars.drums["kick"][7]) if cell in "Xx"}
    assert {nt.step - 112 for nt in c.bass.notes if nt.step >= 112} == kick8, "@8 kick edit drives bar 8"


def test_bass_pitch_is_always_the_chord_root():
    for name, brief in (("example.ls", EXAMPLE_BRIEF), ("one_bar_base.ls", Brief("low", "hopeful", 70, 5)),
                        ("span1_dotted.ls", Brief("mid", "melancholy", 93, 11))):
        c = compiled(name, brief)
        loop = loop_of(name, brief.mode)
        bars = expand(loop)
        for nt in c.bass.notes:
            chord = bars.chords[nt.step // 16]
            assert nt.pitch % 12 == root_pc(chord.numeral, brief.tonic) and 28 <= nt.pitch <= 43
        assert len({nt.pitch for nt in c.bass.notes}) == 2 and set(c.bass_pitches) == {nt.pitch for nt in c.bass.notes}


def test_explicit_bass_holds_are_cut_and_re_struck_at_chord_changes():
    text = LOCK.replace("bass: lock", "bass: ----x-------.... x---------------")
    loop = loop_text(text)
    c = compile_loop(loop, EXAMPLE_BRIEF, dev_palette())
    d2, g1 = c.bass_pitches
    notes = [(nt.step, nt.pitch) for nt in c.bass.notes]
    # bar 2 holds into bar 3 (Im9 -> IV13): re-struck on G at step 32; bar 8 holds into bar 1: re-struck on D
    assert (32, g1) in notes and (0, d2) in notes and (16, d2) in notes
    assert notes.count((0, d2)) == 1
    by_step = {nt.step: nt for nt in c.bass.notes}
    assert by_step[16].off == by_step[32].on, "the held note ends exactly where the re-strike starts"
    assert by_step[32].velocity == by_step[16].velocity == 88


def test_keys_and_bass_play_every_level_at_its_spec_velocity():
    # docs/COMPILER.md §3.5: keys X 96, x 80, g 52 and bass X 104, x 88, g 64, a ghost strike included (no fixture
    # has a g in comp or bass, and `bass: lock` copies only X and x).
    text = LOCK.replace("comp: x---------------", "comp: g---x---X-------").replace("bass: lock",
                                                                                     "bass: g---x---X-------")
    c = compile_loop(loop_text(text), EXAMPLE_BRIEF, dev_palette())
    keys = {(nt.step % 16, nt.velocity) for nt in c.keys.notes}
    bass = {(nt.step % 16, nt.velocity) for nt in c.bass.notes}
    assert keys == {(0, 52), (4, 80), (8, 96)} and bass == {(0, 64), (4, 88), (8, 104)}
    # three strikes a bar: Im9 (b3 b7 9) in bars 1, 2, 5, 6 and IV13 (3 b7 9 13, voicing rules V2) in 3, 4, 7, 8
    assert len(c.keys.notes) == 4 * 3 * 3 + 4 * 3 * 4 and len(c.bass.notes) == 8 * 3
