"""compile_loop over every loop fixture that passes its gate: the interface invariants, the section 1
example's claims, humanize, and determinism (same process, fresh processes, A-B-A order)."""
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest

from helpers_compile import (EXAMPLE_BRIEF, ROOT, compiled, dev_palette, loop_of, loop_text, palette_with,
                             passing_fixtures, spec_sample, spec_sax_pitches)
from helpers_voicing import added
from lofi_rl.brief import Brief
from lofi_rl.compile.compiler import CompileError, compile_loop, lane_clock
from lofi_rl.compile.humanize import Humanize, loop_seed
from lofi_rl.compile.timeline import Clock, loop_samples, ms, segments
from lofi_rl.dsl import canon, parse
from lofi_rl.dsl.ast import expand, timed
from lofi_rl.dsl.vocab import LANES, MODES
from lofi_rl.prng import splitmix64
from lofi_rl.theory.chords import chord_pcs, root_pc
from lofi_rl.theory.voicing import voice_vamp

BRIEFS = (EXAMPLE_BRIEF, Brief("low", "melancholy", 60, 9), Brief("high", "hopeful", 95, 4),
          Brief("mid", "melancholy", 73, 11), Brief("mid", "hopeful", 88, 0))


def test_the_gate_passing_fixtures_are_found():
    names = passing_fixtures()
    assert {"example.ls", "example_leniency.ls", "leniency_lower_x.ls", "one_bar_base.ls", "shaker_edits.ls",
            "span1_dotted.ls"} <= set(names)
    assert not {"g1_kick_words.ls", "g3_no_b.ls", "g4_kit_trap808.ls", "g5_tempo.ls"} & set(names)


EXAMPLE_TEXT = (ROOT / "tests" / "fixtures" / "loops" / "example.ls").read_text()


def test_a_loop_without_a_sax_statement_passes_the_gates():
    # One sax plays every loop (TASTE.md, Lead) and the plan drops `sax:` once one survives, so the line chooses
    # nothing: it is optional, never a G3 requirement (docs/DSL.md). The rest of the loop parses as before.
    assert "sax: tenor\n" in EXAMPLE_TEXT
    vocab = dev_palette().vocab()
    res = parse(EXAMPLE_TEXT.replace("sax: tenor\n", ""), vocab=vocab, mode="dorian")
    assert res.gate.ok, res.gate
    with_sax = parse(EXAMPLE_TEXT, vocab=vocab, mode="dorian")
    assert res.ast.sax is None and with_sax.ast.sax == "tenor"
    assert res.ast == replace(with_sax.ast, sax=None) and res.penalty == with_sax.penalty


def test_a_loop_without_a_sax_statement_compiles_to_the_same_loop_as_with_sax_tenor():
    # The palette's sax_lead plays whatever the line says (docs/COMPILER.md §2), and the humanize seed leaves the
    # line out (§3.6, SR-7), so the line changes nothing the compiler produces, with humanize off or on.
    without = loop_text(EXAMPLE_TEXT.replace("sax: tenor\n", ""))
    tenor, alto = loop_of("example.ls"), loop_text(EXAMPLE_TEXT.replace("sax: tenor\n", "sax: alto\n"))
    for humanize in ("off", "bar", "loop"):
        pal = palette_with(groove_humanize=humanize)
        want = compile_loop(tenor, EXAMPLE_BRIEF, pal).digest()
        assert compile_loop(without, EXAMPLE_BRIEF, pal).digest() == want, humanize
        assert compile_loop(alto, EXAMPLE_BRIEF, pal).digest() == want, humanize


def _check(c, loop, brief):
    pal = dev_palette()
    bars = expand(loop)
    assert c.n == loop_samples(brief.bpm) and c.n % 64 == 0
    assert (c.bpm, c.tonic, c.mode, c.palette_id) == (brief.bpm, brief.tonic, brief.mode, "dev_gu")
    for stem in c.stems:
        assert list(stem.notes) == sorted(stem.notes, key=lambda x: (x.on, x.channel, x.pitch))
        for nt in stem.notes:
            assert 0 <= nt.on < c.n and nt.on < nt.off <= nt.on + c.n and 1 <= nt.velocity <= 127
            assert nt.channel in stem.channels and 0 <= nt.step < 128
    chords = loop.vamp.chords
    # The voicing pair is the section 3.3 choice (voicing rules V2) for the brief's tonic and mode and the loop's
    # own style (test_voicing.py checks voice_vamp against a brute force), and every keys note belongs to the chord
    # it plays under.
    assert c.voicings == voice_vamp(*chords, brief.tonic, brief.mode, loop.keys.voicing), (loop.keys.voicing, brief)
    voicing = dict(zip(chords, c.voicings))
    strikes = segments("".join(bars.comp), bars.chords)
    assert len(c.keys.notes) == sum(len(voicing[bars.chords[s.start // 16]]) for s in strikes)
    for s in strikes:
        chord = bars.chords[s.start // 16]
        assert sorted(nt.pitch for nt in c.keys.notes if nt.step == s.start) == list(voicing[chord])
    mode = {(brief.tonic + s) % 12 for s in MODES[brief.mode]}
    for nt in c.keys.notes:
        chord = bars.chords[nt.step // 16]
        root = root_pc(chord.numeral, brief.tonic)
        # V2.1 R3: quartal may add the quality's available tensions that are in the brief's mode (the voicing
        # oracle's `added`); the other styles take only chord tones, never the root (R4). Nothing outside chord +
        # mode, ever.
        if loop.keys.voicing == "quartal":
            extra = {(root + t) % 12 for t in added(chord.quality, chord.numeral, brief.mode)}
            assert extra <= mode, (chord, extra)
        else:
            extra = set()
            assert nt.pitch % 12 != root, (chord, nt)
        assert nt.pitch % 12 in chord_pcs(chord, brief.tonic) | extra, (chord, nt)
    assert all(48 <= p <= 72 for v in c.voicings for p in v)
    for nt in c.bass.notes:
        assert 28 <= nt.pitch <= 43 and nt.pitch % 12 == root_pc(bars.chords[nt.step // 16].numeral, brief.tonic)
    sax = pal.sax[pal.sax_lead]
    assert len(c.sax.notes) == sum(not it.rest for it in loop.a + loop.b)
    assert all(sax.play[0] <= nt.pitch <= sax.play[1] for nt in c.sax.notes)
    # Section 3.2 from the spec tables, in the brief's mode and tonic; section 3.1 from the formula, at every tempo
    # (73 BPM puts the steps on half samples).
    assert {nt.step: nt.pitch for nt in c.sax.notes} == spec_sax_pitches(loop, brief.tonic, brief.mode), brief
    assert all(nt.on == spec_sample(nt.step, c.n) % c.n for nt in c.sax.notes), "the sax plays on the swung grid"
    assert all(d.on == spec_sample(d.step, c.n) % c.n for d in c.drums), "every hit on its step's time"
    cells = [(lane, 16 * b + s, cell) for lane in LANES for b, bar in enumerate(bars.drums[lane])
             for s, cell in enumerate(bar) if cell in "Xxg"]
    assert sorted((d.lane, d.step, d.level) for d in c.drums) == sorted(cells)
    assert list(c.drums) == sorted(c.drums, key=lambda d: (d.on, LANES.index(d.lane), d.step))
    assert all(0 <= d.on < c.n and d.rr == 0 and d.layer == {"X": "hard", "x": "med", "g": "soft"}[d.level]
               for d in c.drums)
    # dev_gu's open hat is choked by either hat (review T7), so every open hat ends somewhere, at the latest at its own
    # repeat one cycle on, and nothing else is choked.
    assert all((d.choke_at is None) == (d.lane != "ohat") for d in c.drums), "only open hats are choked, all of them"


@pytest.mark.parametrize("name", passing_fixtures())
def test_every_passing_fixture_compiles(name):
    for brief in BRIEFS:
        loop = loop_of(name, brief.mode)
        c = compile_loop(loop, brief, dev_palette())
        _check(c, loop, brief)
        assert len(c.digest()) == 64 and json.loads(c.to_json())["n"] == c.n


def test_the_leniency_fixture_compiles_to_the_example():
    assert compiled("example_leniency.ls").digest() == compiled("example.ls").digest()


def test_section_1_example_claims():
    c = compiled("example.ls")
    bass = {nt.step for nt in c.bass.notes}
    kicks = {d.step for d in c.drums if d.lane == "kick" and d.level in "Xx"}
    assert len(bass) == 24 and len(bass & kicks) == 19, "bass and kick coincide on 19 of 24 bass onsets"
    assert c.n == 1_032_576 and abs(c.bpm_eff - 82) < 0.001
    assert c.voicings == ((53, 60, 64), (53, 57, 59, 64)) and c.bass_pitches == (38, 31)   # F3 C4 E4 | F3 A3 B3 E4
    assert (c.keys_name, c.kit, c.keys.program, c.bass.program) == ("rhodes", "jazz", 4, 32)


def test_compiled_loops_are_frozen():
    c = compiled("example.ls")
    with pytest.raises(FrozenInstanceError):
        c.n = 1
    with pytest.raises(FrozenInstanceError):
        c.sax.notes[0].pitch = 60


def test_same_process_and_aba_order_are_bit_identical():
    pal = dev_palette()
    a, b = loop_of("example.ls"), loop_of("shaker_edits.ls")
    first = compile_loop(a, EXAMPLE_BRIEF, pal)
    other = compile_loop(b, EXAMPLE_BRIEF, pal)
    again = compile_loop(a, EXAMPLE_BRIEF, pal)
    assert first == again and first.digest() == again.digest() != other.digest()
    assert compile_loop(parse(canon(a), vocab=pal.vocab()).ast, EXAMPLE_BRIEF, pal) == first


_CHILD = """
import sys
from pathlib import Path
from lofi_rl.brief import Brief
from lofi_rl.compile.compiler import compile_loop
from lofi_rl.dsl import parse
from lofi_rl.palette import load_palette
pal = load_palette(Path(sys.argv[1]), verify=False)
for name in sys.argv[2:]:
    loop = parse(Path(name).read_text(), vocab=pal.vocab()).ast
    print(compile_loop(loop, Brief("mid", "wistful", 82, 2), pal).digest())
"""


def test_fresh_processes_with_different_hash_seeds_agree():
    files = [str(ROOT / "tests/fixtures/loops" / n) for n in ("example.ls", "shaker_edits.ls")]
    want = [compiled(n).digest() for n in ("example.ls", "shaker_edits.ls")]
    for seed in ("0", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT / "src")}
        out = subprocess.run([sys.executable, "-c", _CHILD, str(ROOT / "palettes/dev_gu.yaml"), *files],
                             capture_output=True, text=True, env=env, check=True)
        assert out.stdout.split() == want


def test_humanize_off_puts_every_hit_on_its_clock():
    c = compiled("example.ls")
    plain = Clock(c.n, swing=dev_palette().groove.swing)
    assert all(d.on == plain.sample(d.step) for d in c.drums)
    # §3.6: the seed hashes the canonical text without its keys:, drums: and sax: lines. example.ls is canonical but
    # for its kit alias, which sits on a dropped line, so that is its own text without lines 2, 5 and 16.
    lines = EXAMPLE_TEXT.splitlines(keepends=True)
    assert [lines[i] for i in (1, 4, 15)] == ["keys: rhodes rootless\n", "drums: dusty\n", "sax: tenor\n"]
    kept = "".join(line for i, line in enumerate(lines) if i not in (1, 4, 15))
    assert c.seed == loop_seed(loop_of("example.ls")) == int(hashlib.sha256(kept.encode()).hexdigest()[:16], 16)


@pytest.mark.parametrize("mode", ["bar", "loop"])
def test_naming_another_keys_instrument_voicing_or_kit_keeps_every_nudge(mode):
    # SR-7: the seed hashes only the vamp, the grids and the phrases (§3.6). So swapping rhodes for organ, or
    # rootless for spread, draws the same humanize nudges: every drum hit and bass note is unchanged, and the keys
    # strike at the same times and velocities (spread changes only their pitches). So does another kit name that
    # resolves to the same kit (parsed without the palette, so canon keeps the name).
    pal = palette_with(groove_humanize=mode)
    base = compile_loop(loop_of("example.ls"), EXAMPLE_BRIEF, pal)

    def strikes(c):
        return sorted({(nt.step, nt.on, nt.off, nt.velocity) for nt in c.keys.notes})

    organ = compile_loop(loop_text(EXAMPLE_TEXT.replace("keys: rhodes", "keys: organ")), EXAMPLE_BRIEF, pal)
    spread = compile_loop(loop_text(EXAMPLE_TEXT.replace("rootless", "spread")), EXAMPLE_BRIEF, pal)
    assert organ.keys_name == "organ" and organ.keys.program == 16 and spread.voicings != base.voicings
    for c in (organ, spread):
        assert c.seed == base.seed
        assert c.drums == base.drums and c.bass.notes == base.bass.notes
        assert strikes(c) == strikes(base)
    assert organ.keys.notes == base.keys.notes
    kits = [parse(EXAMPLE_TEXT.replace("drums: dusty", f"drums: {kit}"), mode="dorian").ast
            for kit in ("dusty", "brushes", "jazz")]
    assert len({canon(loop) for loop in kits}) == 3
    assert {compile_loop(loop, EXAMPLE_BRIEF, pal).digest() for loop in kits} == {base.digest()}


@pytest.mark.parametrize("mode", ["bar", "loop"])
def test_humanize_nudges_stay_within_3_ms_and_8_percent_and_repeat(mode):
    pal = palette_with(groove_humanize=mode)
    loop = loop_of("example.ls")
    c = compile_loop(loop, EXAMPLE_BRIEF, pal)
    plain = Clock(c.n, swing=pal.groove.swing)
    base = {d.lane + str(d.step): d for d in compiled("example.ls").drums}
    for d in c.drums:
        drift = (d.on - plain.sample(d.step) + c.n // 2) % c.n - c.n // 2      # a step-0 hit nudged early wraps
        assert abs(drift) <= 133                                               # 3 ms = 132.3 samples
        assert 0.92 - 1e-9 <= d.gain / base[d.lane + str(d.step)].gain <= 1.08 + 1e-9
    human = Humanize.from_groove(pal.groove, c.seed)
    same_every_bar = all(human.nudge(lane, k) == human.nudge(lane, k + 16) for lane in LANES for k in range(112))
    assert same_every_bar == (mode == "bar")
    assert all(human.nudge(lane, k) == human.nudge(lane, k + 128) for lane in LANES for k in range(128))
    assert c != compiled("example.ls") and c == compile_loop(loop, EXAMPLE_BRIEF, pal)
    keys = lane_clock("keys", c.n, pal, human)
    assert all(nt.on == keys.sample(nt.step) % c.n for nt in c.keys.notes)
    assert all(abs(nt.velocity - {"X": 96, "x": 80, "g": 52}[_level(loop, nt.step)]) <= 8 for nt in c.keys.notes)


LANE_ID = {"kick": 1, "snare": 2, "rim": 3, "hat": 4, "ohat": 5, "shaker": 6, "keys": 7, "bass": 8, "sax": 9}


def _spec_u(seed, draw, lane, k, mode):
    """docs/COMPILER.md §3.6: u = (splitmix64(seed ^ draw << 24 ^ lane << 16 ^ bar << 8 ^ step-in-bar) >> 11) 2^-53,
    with bar 0 in H1 (`bar`) and the bar 0-7 in H2 (`loop`). splitmix64 is pinned by tests/test_prng.py."""
    r = k % 128
    key = seed ^ (draw << 24) ^ (LANE_ID[lane] << 16) ^ ((r // 16 if mode == "loop" else 0) << 8) ^ (r % 16)
    return Fraction(splitmix64(key) >> 11, 2 ** 53)


@pytest.mark.parametrize("mode", ["bar", "loop"])
def test_humanize_draws_follow_the_documented_key(mode):
    # Timing (2u - 1) 3 ms from draw 1, velocity x (1 + (2u - 1) 0.08) from draw 2, each loop from its own seed:
    # computed here from §3.6 for every drum hit and every keys and bass strike of three loops.
    pal = palette_with(groove_humanize=mode)
    patterns = set()
    for name in ("example.ls", "shaker_edits.ls", "one_bar_base.ls"):
        c, plain = compile_loop(loop_of(name), EXAMPLE_BRIEF, pal), compiled(name)
        assert c.seed == plain.seed == loop_seed(loop_of(name))
        timing = lambda lane, k: (2 * _spec_u(c.seed, 1, lane, k, mode) - 1) * Fraction(1323, 10)    # noqa: E731
        factor = lambda lane, k: 1 + (2 * _spec_u(c.seed, 2, lane, k, mode) - 1) * Fraction(2, 25)   # noqa: E731
        gain = {(d.lane, d.step): d.gain for d in plain.drums}
        for d in c.drums:
            assert d.on == spec_sample(d.step, c.n, nudge=timing(d.lane, d.step)) % c.n, (name, d)
            assert d.gain == pytest.approx(gain[d.lane, d.step] * float(factor(d.lane, d.step)), rel=1e-12), d
        for lane, stem, base in (("keys", c.keys, plain.keys), ("bass", c.bass, plain.bass)):
            velocity = {(nt.step, nt.pitch): nt.velocity for nt in base.notes}
            for nt in stem.notes:
                assert nt.on == spec_sample(nt.step, c.n, nudge=timing(lane, nt.step)) % c.n, (name, nt)
                want = math.floor(velocity[nt.step, nt.pitch] * factor(lane, nt.step) + Fraction(1, 2))
                assert nt.velocity == min(127, max(1, want)), (name, lane, nt)
        assert c.sax == plain.sax, "humanize never touches the sax (§3.6 Lanes)"
        human = Humanize.from_groove(pal.groove, c.seed)
        nudges = [human.nudge(lane, k) for lane in LANES + ("keys", "bass") for k in range(128)]
        assert max(nudges) > ms(Fraction(5, 2)) and min(nudges) < -ms(Fraction(5, 2)), "the nudges fill +-3 ms"
        patterns.add(tuple(nudges))
    assert len(patterns) == 3, "different loops draw different nudges"


@pytest.mark.parametrize("mode", ["off", "bar", "loop"])
def test_humanize_keeps_legato_keys_and_bass_strikes_joined(mode):
    # A nudge moves each note-off with the step it ends on (§3.6), so a strike still ends exactly where the next
    # one starts (§3.4), around the seam too.
    text = """vamp: Im9 IV13 span 2
keys: rhodes rootless
comp: x---x---x---x---
bass: x---x---x---x---
drums: jazz
kick: x.......x.......
snare: ....x.......x...
hat: x.x.x.x.x.x.x.x.
A: w5 | w3 | w1 | w2
B: w5 | w3 | w1 | h2 hr
"""
    c = compile_loop(loop_text(text), EXAMPLE_BRIEF, palette_with(groove_humanize=mode))
    for stem in (c.keys, c.bass):
        strikes = sorted({(nt.step, nt.on, nt.off) for nt in stem.notes})
        assert [k for k, _, _ in strikes] == list(range(0, 128, 4))
        assert all((off - nxt) % c.n == 0 for (_, _, off), (_, nxt, _) in zip(strikes, strikes[1:] + strikes[:1]))
        moved = sum(on != spec_sample(k, c.n) % c.n for k, on, _ in strikes)
        assert (moved > 16) == (mode != "off"), (stem.name, moved)


def _level(loop, step):
    for s in segments("".join(expand(loop).comp), expand(loop).chords):
        if s.start == step:
            return s.level
    raise KeyError(step)


def test_unknown_names_and_bad_vamps_are_compile_errors():
    pal = dev_palette()
    loop = loop_of("example.ls")
    with pytest.raises(CompileError, match="keys name"):
        compile_loop(replace(loop, keys=replace(loop.keys, instrument="clavinet")), EXAMPLE_BRIEF, pal)
    with pytest.raises(CompileError, match="kits name"):
        compile_loop(replace(loop, kit="trap808"), EXAMPLE_BRIEF, pal)
    narrow = replace(pal, voicings=("rootless",), qualities=tuple(q for q in pal.qualities if q != "13"))
    with pytest.raises(CompileError, match="chord"):
        compile_loop(loop, EXAMPLE_BRIEF, narrow)
    with pytest.raises(CompileError, match="voicing"):
        compile_loop(replace(loop, keys=replace(loop.keys, voicing="shell")), EXAMPLE_BRIEF, narrow)
    same = replace(loop, vamp=replace(loop.vamp, chords=(loop.vamp.chords[0],) * 2))
    with pytest.raises(CompileError, match="distinct"):
        compile_loop(same, EXAMPLE_BRIEF, pal)
    rests = replace(loop, b=tuple(replace(it, rest=True, degree=0, accidental=0, octave=0) for it in loop.b))
    with pytest.raises(CompileError, match="phrase"):
        compile_loop(rests, EXAMPLE_BRIEF, pal)


def test_sax_items_map_one_to_one_onto_notes():
    loop = loop_of("span1_dotted.ls")
    c = compiled("span1_dotted.ls")
    steps = [k for k, it in timed(loop.a, 0) + timed(loop.b, 64) if not it.rest]
    assert sorted(nt.step for nt in c.sax.notes) == steps
