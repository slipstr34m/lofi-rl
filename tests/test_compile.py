"""compile_loop over every loop fixture that passes its gate: the interface invariants, the section 1
example's claims, humanize, and determinism (same process, fresh processes, A-B-A order)."""
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest

from helpers_compile import EXAMPLE_BRIEF, ROOT, compiled, dev_palette, loop_of, palette_with, passing_fixtures
from lofi_rl.brief import Brief
from lofi_rl.compile.compiler import CompileError, compile_loop, lane_clock
from lofi_rl.compile.humanize import Humanize, loop_seed
from lofi_rl.compile.timeline import Clock, loop_samples, segments
from lofi_rl.dsl import canon, parse
from lofi_rl.dsl.ast import expand, timed
from lofi_rl.dsl.vocab import LANES
from lofi_rl.theory.chords import root_pc

BRIEFS = (EXAMPLE_BRIEF, Brief("low", "melancholy", 60, 9), Brief("high", "hopeful", 95, 4),
          Brief("mid", "melancholy", 73, 11), Brief("mid", "hopeful", 88, 0))


def test_the_gate_passing_fixtures_are_found():
    names = passing_fixtures()
    assert {"example.ls", "example_leniency.ls", "leniency_lower_x.ls", "one_bar_base.ls", "shaker_edits.ls",
            "span1_dotted.ls"} <= set(names)
    assert not {"g1_kick_words.ls", "g3_no_b.ls", "g4_kit_trap808.ls", "g5_tempo.ls"} & set(names)


@pytest.mark.xfail(strict=True, reason="open DSL bug (review finding spec-5): dsl/parse.py REQUIRED omits 'sax', "
                                       "which is outside the compiler's code. Drop this mark once it is fixed.")
def test_a_loop_without_its_sax_statement_fails_g3():
    # instrument.md §1 lists sax: as required and §2 makes a missing one gate G3. The compiler never reads the
    # line (it plays the palette's sax_lead), so today the only effect is on the gate pass rate.
    text = (ROOT / "tests" / "fixtures" / "loops" / "example.ls").read_text()
    assert "sax: tenor\n" in text
    gate = parse(text.replace("sax: tenor\n", ""), vocab=dev_palette().vocab(), mode="dorian").gate
    assert (gate.ok, gate.code, gate.detail) == (False, "G3", "missing: sax")


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
    voicing = dict(zip(chords, c.voicings))
    strikes = segments("".join(bars.comp), bars.chords)
    assert len(c.keys.notes) == sum(len(voicing[bars.chords[s.start // 16]]) for s in strikes)
    for s in strikes:
        chord = bars.chords[s.start // 16]
        assert sorted(nt.pitch for nt in c.keys.notes if nt.step == s.start) == list(voicing[chord])
    assert all(48 <= p <= 72 for v in c.voicings for p in v)
    for nt in c.bass.notes:
        assert 28 <= nt.pitch <= 43 and nt.pitch % 12 == root_pc(bars.chords[nt.step // 16].numeral, brief.tonic)
    sax = pal.sax[pal.sax_lead]
    assert len(c.sax.notes) == sum(not it.rest for it in loop.a + loop.b)
    assert all(sax.play[0] <= nt.pitch <= sax.play[1] for nt in c.sax.notes)
    cells = [(lane, 16 * b + s, cell) for lane in LANES for b, bar in enumerate(bars.drums[lane])
             for s, cell in enumerate(bar) if cell in "Xxg"]
    assert sorted((d.lane, d.step, d.level) for d in c.drums) == sorted(cells)
    assert list(c.drums) == sorted(c.drums, key=lambda d: (d.on, LANES.index(d.lane), d.step))
    assert all(0 <= d.on < c.n and d.rr == 0 and d.layer == {"X": "hard", "x": "med", "g": "soft"}[d.level]
               for d in c.drums)
    if any(d.lane == "hat" for d in c.drums):
        assert all((d.choke_at is None) == (d.lane != "ohat") for d in c.drums), "only open hats are choked"


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
    assert c.voicings == ((53, 60, 64), (57, 59, 65)) and c.bass_pitches == (38, 31)
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
    assert c.seed == loop_seed(loop_of("example.ls"))
    assert c.seed == int(hashlib.sha256(canon(loop_of("example.ls")).encode()).hexdigest()[:16], 16)


@pytest.mark.parametrize("mode", ["bar", "loop"])
def test_humanize_nudges_stay_within_3_ms_and_8_percent_and_repeat(mode):
    pal = palette_with(groove_humanize=mode)
    loop = loop_of("example.ls")
    c = compile_loop(loop, EXAMPLE_BRIEF, pal)
    plain = Clock(c.n, swing=pal.groove.swing)
    base = {d.lane + str(d.step): d for d in compiled("example.ls").drums}
    for d in c.drums:
        assert abs(d.on - plain.sample(d.step)) <= 133                          # 3 ms = 132.3 samples
        assert 0.92 - 1e-9 <= d.gain / base[d.lane + str(d.step)].gain <= 1.08 + 1e-9
    human = Humanize.from_groove(pal.groove, c.seed)
    same_every_bar = all(human.nudge(lane, k) == human.nudge(lane, k + 16) for lane in LANES for k in range(112))
    assert same_every_bar == (mode == "bar")
    assert all(human.nudge(lane, k) == human.nudge(lane, k + 128) for lane in LANES for k in range(128))
    assert c != compiled("example.ls") and c == compile_loop(loop, EXAMPLE_BRIEF, pal)
    keys = lane_clock("keys", c.n, pal, human)
    assert all(nt.on == keys.sample(nt.step) % c.n for nt in c.keys.notes)
    assert all(abs(nt.velocity - {"X": 96, "x": 80, "g": 52}[_level(loop, nt.step)]) <= 8 for nt in c.keys.notes)


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
