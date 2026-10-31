"""lofi_rl.verify on hand-built loops: the symbolic reward terms of docs/design/instrument.md §5 (CLAUDE.md §6 test
oracle; the readings are in docs/VERIFIERS.md).

Every term gets a loop that scores 1, one that scores 0 and an edge case. Expected values are worked out by hand from
§5 in the comments, never read back from the code. Most loops are PLAIN with a line or two changed: PLAIN is
instrument.md §1's example with 1-bar bases and no edits, so its groove and sax lines are the example's.
"""
from __future__ import annotations

import dataclasses
import json
import math
import random
from functools import lru_cache

import pytest

from helpers_compile import FIX, dev_palette
from lofi_rl.brief import Brief
from lofi_rl.compile.compiler import CompileError, compile_loop
from lofi_rl.dsl import parse
from lofi_rl.verify import Score, group_penalties, score
from lofi_rl.verify import knobs as vk
from lofi_rl.verify import penalties as vp
from lofi_rl.verify import sax as vs

BRIEF = Brief(energy="mid", mood="wistful", bpm=82, tonic=2)       # instrument.md §1: D dorian, energy mid
THEORY_KEYS = ("f_bb", "f_k1", "f_lock", "f_tk", "groove", "f_long", "f_land", "f_open", "f_ans", "f_breath",
               "sax", "f_new", "f_seam", "p_chrom")
KNOB_KEYS = ("E", "f_energy", "energy_match", "mood_aeolian", "mood_dorian", "mood_ionian", "mood_margin", "f_mood")
PLAIN = {
    "vamp": "Im9 IV13 span 2",
    "keys": "rhodes rootless",
    "comp": "x-----.x----..x-",
    "bass": "x-------..x-x---",
    "drums": "jazz",
    "kick": "x......x..x.....",
    "snare": "....x.......x...",
    "hat": "x.x.x.x.x.x.x.x.",
    "A": "q5 e6 e5 q3 e2 e1 | h3 er e4 e5 e6 | h.5 er e4 | q3 q2 hr",
    "B": "q1' e7 e1' q5 q3 | er e3 e4 eb5 h5 | qr e4 e3 q2 q1 | h.6 qr",
}
EXAMPLE = (FIX / "example.ls").read_text()
# 16 quarter notes, no rests: used where a term must see no long note and no breath
QUARTERS_A = "q5 q6 q5 q3 | q2 q1 q3 q2 | q5 q6 q5 q3 | q2 q1 q3 q2"
QUARTERS_B = "q1' q7 q1' q5 | q3 q4 q5 q4 | q4 q3 q2 q1 | q2 q3 q6 q6"


def text(over: dict | None = None, **kw: str | None) -> str:
    """PLAIN with lines replaced, added (new heads, edits as "@8 snare") or dropped (None)."""
    lines = dict(PLAIN)
    lines.update(over or {})
    lines.update(kw)
    return "".join(f"{head}: {body}\n" for head, body in lines.items() if body is not None)


@lru_cache(maxsize=None)
def sc(src: str, brief: Brief = BRIEF) -> Score:
    return score(src, brief, dev_palette())


def part(src: str, name: str, brief: Brief = BRIEF) -> float:
    s = sc(src, brief)
    assert s.gate_ok, (s.gate, src)
    return s.theory_parts[name]


# ------------------------------------------------------------------------------------------------ interface
def test_score_has_exactly_the_shared_interface_fields_and_is_frozen():
    names = [f.name for f in dataclasses.fields(Score)]
    assert names == ["gate_ok", "gate", "repairs", "p_fix", "theory", "theory_parts", "knobs", "knob_parts",
                     "p_sax", "penalties", "digest"]
    s = sc(text())
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.theory = 0.0                                              # type: ignore[misc]
    assert tuple(s.theory_parts) == THEORY_KEYS and tuple(s.knob_parts) == KNOB_KEYS
    assert set(s.penalties) == {"bars", "len"}
    row = json.loads(json.dumps(dataclasses.asdict(s)))           # a JSONL log line holds every field
    assert row["theory_parts"]["f_bb"] == 1.0 and row["digest"] == s.digest


def test_plain_scores_by_hand():
    """PLAIN: every groove and sax term is 1 (the example's lines); 1-bar bases with no edits make only bar 3 new
    (f_new = 1/7, the §5 fixture re-specified: the bar signature holds the chord, so it cannot reach 0)."""
    s = sc(text())
    assert s.gate_ok and s.gate is None and s.repairs == 0 and s.p_fix == 0.0
    for name in THEORY_KEYS:
        want = {"f_new": 1 / 7, "p_chrom": 0.0}.get(name, 1.0)
        assert s.theory_parts[name] == pytest.approx(want), name
    assert s.theory == pytest.approx(0.3 + 0.3 + 0.2 / 7 + 0.2)
    # E = .35 * 104/192 + .15 * 0/80 + .25 * 24/48 + .25 * 24/48 (kick 24, snare 16, hat 64; no accents)
    assert s.knob_parts["E"] == pytest.approx(0.35 * 104 / 192 + 0.25 * 0.5 + 0.25 * 0.5)
    assert s.knob_parts["f_energy"] == 1.0 and s.knob_parts["energy_match"] == 1.0
    assert (s.knob_parts["mood_dorian"], s.knob_parts["mood_aeolian"], s.knob_parts["mood_ionian"]) == (34, 26, 14)
    assert s.knob_parts["mood_margin"] == 8 and s.knob_parts["f_mood"] == 1.0 and s.knobs == 1.0
    assert s.p_sax == 0.0 and s.penalties == {"bars": 0.0, "len": 0.0}
    res = parse(text(), vocab=dev_palette().vocab(), mode="wistful")
    assert s.digest == compile_loop(res.ast, BRIEF, dev_palette()).digest()


# ------------------------------------------------------------------------------------------------ gates
@pytest.mark.parametrize("fixture, code", [
    ("g1_kick_words.ls", "G1"), ("g3_no_b.ls", "G3"), ("g4_kit_trap808.ls", "G4"), ("g5_tempo.ls", "G5"),
    ("g6_kick_9_cells.ls", "G6"), ("g6_vamp_one_chord.ls", "G6"),
])
def test_a_failed_gate_zeroes_every_term_and_penalty(fixture, code):
    s = sc((FIX / fixture).read_text())
    assert (s.gate_ok, s.gate, s.digest) == (False, code, None)
    assert s.theory == s.knobs == s.p_sax == s.p_fix == 0.0
    assert set(s.theory_parts.values()) == {0.0} and set(s.knob_parts.values()) == {0.0}
    assert tuple(s.theory_parts) == THEORY_KEYS and tuple(s.knob_parts) == KNOB_KEYS
    assert s.penalties == {"bars": 0.0, "len": 0.0}


def test_g2_truncated_completion():
    s = score(EXAMPLE, BRIEF, dev_palette(), finish_reason="length")
    assert (s.gate_ok, s.gate, s.repairs) == (False, "G2", 0)


def test_gates_run_in_the_briefs_mood():
    """A bare numeral is repaired into the mood's diatonic 7th chord: `I` is Im7 in wistful (dorian) and Imaj7 in
    hopeful (ionian); `bII` has none in dorian, so it is G4 there (docs/DSL.md §6)."""
    bare = text(vamp="I IV span 2")
    assert sc(bare).gate_ok and sc(bare).repairs == 2 and sc(bare).p_fix == pytest.approx(0.04)
    hopeful = Brief(energy="mid", mood="hopeful", bpm=82, tonic=2)
    assert sc(bare, hopeful).gate_ok and sc(bare, hopeful).digest != sc(bare).digest
    assert sc(text(vamp="bII IV span 2")).gate == "G4"


def test_a_compile_error_is_g7_compile(monkeypatch):
    import lofi_rl.verify.scoring as scoring

    def boom(*_args, **_kw):
        raise CompileError("voicing 'quartal' is not in palette x")
    monkeypatch.setattr(scoring, "compile_loop", boom)
    s = score(EXAMPLE, BRIEF, dev_palette())
    assert (s.gate_ok, s.gate, s.repairs, s.theory, s.digest) == (False, "G7-compile", 1, 0.0, None)


def test_repairs_count_every_logged_repair_and_p_fix_only_the_penalized():
    """example_leniency.ls logs two free repairs (strip_prose, the dusty -> jazz alias); a 15-cell kick adds a
    penalized pad_grid (0.02)."""
    assert (sc((FIX / "example_leniency.ls").read_text()).repairs, sc(EXAMPLE).repairs) == (2, 1)
    padded = sc(text(kick="x......x..x...."))
    assert (padded.repairs, padded.p_fix) == (1, pytest.approx(0.02))


def test_mutated_junk_never_raises_and_stays_in_range():
    rng = random.Random(596)
    alphabet = "xX.g-|qeh12345'#b \n"
    for _ in range(30):
        chars = list(EXAMPLE)
        for _ in range(rng.randint(1, 3)):
            chars[rng.randrange(len(chars))] = rng.choice(alphabet)
        s = score("".join(chars), BRIEF, dev_palette())
        assert 0.0 <= s.theory <= 1.0 and 0.0 <= s.knobs <= 1.0 and s.p_sax >= 0.0
        assert all(0.0 <= v <= 1.0 for v in s.theory_parts.values())
        assert all(v >= 0.0 for v in s.penalties.values())
        assert s.gate_ok == (s.digest is not None)


# ------------------------------------------------------------------------------------------------ groove
@pytest.mark.parametrize("over, want", [
    ({}, 1.0),
    ({"snare": "X.......X......."}, 0.0),                           # on 1 and 3, never 2 and 4
    ({"snare": "....x.......g..."}, 0.0),                           # a ghost on 4 is not a backbeat
    ({"snare": "....x.......x... ....x..........."}, 0.5),         # the bracket needs both beats: no half credit
    ({"snare": None, "rim": "............x...", "@1 snare": "....x..........."}, 1 / 8),   # snare 2 + rim 4
])
def test_backbeat_f_bb(over, want):
    assert part(text(over), "f_bb") == pytest.approx(want)


@pytest.mark.parametrize("kick, want", [
    ("x......x..x.....", 1.0), ("X......x..x.....", 1.0),
    (".x.....x..x.....", 0.0),
    ("g......x..x.....", 0.0),                                      # a ghost kick on 1 does not count
    ("x......x..x..... .......x..x.....", 0.5),
])
def test_kick_on_1_f_k1(kick, want):
    assert part(text(kick=kick), "f_k1") == pytest.approx(want)


@pytest.mark.parametrize("bass, want", [
    ("x-------..x-x---", 1.0),          # B {0, 10, 12} on K {0, 7, 10}: 2/3 >= 0.6
    ("lock", 1.0),                      # the kicks, plus chord changes on bar lines, where the kick is
    ("..x...x.....x...", 0.0),          # B {2, 6, 12}: no kick under any
    ("x.x.x.....x.....", 0.5 / 0.6),    # B {0, 2, 4, 10}: 2 of 4 on a kick
    ("................", 0.0),          # no bass at all: nothing locks (Claude's default)
])
def test_kick_bass_lock_f_lock(bass, want):
    assert part(text(bass=bass), "f_lock") == pytest.approx(want)


@pytest.mark.parametrize("over, want", [
    ({}, 1.0),
    ({"hat": "x.......x......."}, 0.0),                             # 2 hits a bar
    ({"hat": "x...x...x...x..."}, 1.0),                             # exactly 4
    ({"hat": "x...x...x......."}, 0.0),                             # 3
    ({"hat": "x.......x.......", "shaker": "..x.......x....."}, 1.0),   # 2 hat + 2 shaker
    ({"hat": "g.g.g.g........."}, 1.0),                             # ghosts are hits
    ({"hat": "x...x...x.......", "ohat": "..............x."}, 0.0),  # the open hat is not hat + shaker
])
def test_timekeeping_f_tk(over, want):
    assert part(text(over), "f_tk") == pytest.approx(want)


# ------------------------------------------------------------------------------------------------ sax shape
@pytest.mark.parametrize("a, b, want", [
    (PLAIN["A"], PLAIN["B"], 1.0),                                  # h3 h.5 + h5 h.6: 4 notes of 8+ steps
    (QUARTERS_A, QUARTERS_B, 0.0),
    ("h5 q6 q5 | " + QUARTERS_A[14:], QUARTERS_B, 0.5),             # 1 long note
    ("h5 h6 | h5 h3 | h2 q1 q3 | q3 q2 qr qr", QUARTERS_B, 0.5),    # 5
    ("h5 h6 | h5 h3 | h2 h1 | q3 q2 qr qr", QUARTERS_B, 0.0),       # 6
    ("q.5 e6 q.5 e3 | " + QUARTERS_A[14:], QUARTERS_B, 0.0),        # 6 steps is not long
])
def test_long_notes_f_long(a, b, want):
    assert part(text(A=a, B=b), "f_long") == pytest.approx(want)


@pytest.mark.parametrize("a, b, want", [
    (PLAIN["A"], PLAIN["B"], 1.0),                                  # A3 = the 5th of Dm9, D4 = its root
    ("qr" + PLAIN["A"][2:], "qr" + PLAIN["B"][3:], 0.0),            # both open on a rest
    ("q4" + PLAIN["A"][2:], PLAIN["B"], 0.75),                      # G over Dm9 is no chord tone: A scores 0.5
    ("q2" + PLAIN["A"][2:], PLAIN["B"], 1.0),                       # E, the 9th of Dm9, is a chord tone
])
def test_landing_f_land(a, b, want):
    assert part(text(A=a, B=b), "f_land") == pytest.approx(want)


@pytest.mark.parametrize("end, want", [("h.6 qr", 1.0), ("h.1 qr", 0.0), ("h.1' qr", 0.0), ("h.2 qr", 1.0)])
def test_open_ending_f_open(end, want):
    assert part(text(B=PLAIN["B"].replace("h.6 qr", end)), "f_open") == pytest.approx(want)


@pytest.mark.parametrize("j, want", [(0.0, 0.0), (0.1, 0.5), (0.2, 1.0), (0.5, 1.0), (0.8, 1.0), (0.9, 0.5),
                                     (1.0, 0.0)])
def test_answer_ramp(j, want):
    assert vs.answer_from_jaccard(j) == pytest.approx(want)


@pytest.mark.parametrize("a, b, want", [
    (PLAIN["A"], PLAIN["B"], 1.0),                                  # J = 6/22
    (PLAIN["A"], PLAIN["A"], 0.0),                                  # B repeats A exactly
    (QUARTERS_A, QUARTERS_B, 0.0),                                  # same rhythm, new notes: J = 1
    ("q1 q2 q3 q5 | q6 q5 h3 | w1 | h2 h3", "q1 q2 q3 q5 | q6 h.5 | w1 | h2 h6", 0.5),   # J = 9/10
])
def test_answer_f_ans(a, b, want):
    assert part(text(A=a, B=b), "f_ans") == pytest.approx(want)


@pytest.mark.parametrize("a, want", [
    (PLAIN["A"], 1.0),                                              # rests 12/64, with an er
    (QUARTERS_A, 0.0),                                              # no rest
    ("q5 q6 q5 q3 | q2 q1 q3 q2 | q5 q6 q5 q3 | q2 q1 hr", 1.0),    # 8/64 = 0.125 >= 0.12
    ("q5 q6 q5 q3 | q2 q1 q3 q2 | q5 q6 q5 q3 | h1 s2 q.r sr", 0.0),    # 7/64 < 0.12
    ("w5 | h3 e2 w.r er", 1.0),                                     # 38/64 = 0.594 <= 0.6
    ("w.5 | w.r wr", 0.0),                                          # 40/64 = 0.625 > 0.6
    ("sr e.5 sr e.6 sr e.5 sr e.3 | sr e.2 sr e.1 sr e.2 sr e.3 | q5 q6 q5 q3 | q2 q1 q3 q2", 0.0),  # no 2-step rest
    ("sr sr q5 sr e.6 sr e.5 e3 | sr e.2 sr e.1 sr e.2 sr e.3 | q5 q6 q5 q3 | q2 q1 q3 q2", 1.0),   # sr sr is one
])
def test_breath_f_breath(a, want):
    """Both phrases get the same line, so f_breath is that phrase's bracket."""
    assert part(text(A=a, B=a), "f_breath") == pytest.approx(want)


# ------------------------------------------------------------------------------------------------ chromatic
@pytest.mark.parametrize("over, want", [
    ({}, 0.0),                                                      # the Ab of B is the blue b5: free
    ({"A": PLAIN["A"].replace("q3 q2 hr", "q3 q#1 hr")}, 0.1),     # Eb over G13: 4 steps, then a rest
    ({"A": PLAIN["A"].replace("q3 e2 e1 |", "q3 e#1 e2 |")}, 0.0),  # 2 steps, then E a semitone up: resolves
    ({"A": PLAIN["A"].replace("q3 e2 e1 |", "q3 e#1 er |")}, 0.1),  # short, but a rest is no resolution
    ({"A": PLAIN["A"].replace("e5 q3 e2 e1 |", "e5 q#1 e2 e1 |")}, 0.1),   # resolves by step, but lasts 4 steps
    ({"A": PLAIN["A"].replace("q3 e2 e1 |", "q3 e#1 e7, |")}, 0.1),  # short, but C3 is 3 semitones under Eb3
    ({"A": "q2" + PLAIN["A"][2:], "B": PLAIN["B"].replace("h.6 qr", "h.6 e5 e#1")}, 0.0),  # B's Eb3 -> A's E3
    ({"A": PLAIN["A"].replace("q3 q2 hr", "h#4 q1 qr")}, 0.0),     # §5's fixture: #4 is the blue b5 (§0, TASTE)
    ({"B": PLAIN["B"].replace("eb5 h5", "hb5 e5")}, 0.0),          # a held blue note is never penalized
    ({"A": "h#1 h#1 | h#1 h#1 | " + PLAIN["A"].split(" | ", 2)[2]}, 0.3),   # 4 bad notes: capped at 0.3
])
def test_chromatic_p_chrom(over, want):
    assert part(text(over), "p_chrom") == pytest.approx(want)


def test_a_borrowed_chords_tones_are_free_only_under_that_chord():
    """bVImaj7 is borrowed in dorian; its 3rd, b6 (Bb in D), is a chord tone in bars 3-4 and is bad in bars 1-2."""
    vamp = "Im9 bVImaj7 span 2"
    over_its_chord = PLAIN["A"].replace("h.5 er e4 |", "h.b6 er e4 |")       # bar 3
    over_im9 = PLAIN["A"].replace("h3 er e4 e5 e6 |", "hb6 er e4 e5 e6 |")   # bar 2
    assert part(text(vamp=vamp, A=over_its_chord), "p_chrom") == 0.0
    assert part(text(vamp=vamp, A=over_im9), "p_chrom") == pytest.approx(0.1)


def test_theory_combines_the_terms_as_section_5_says():
    s = sc(text(A=PLAIN["A"].replace("q3 q2 hr", "q3 q#1 hr"), kick=".x.....x..x....."))
    p = s.theory_parts
    groove = 0.30 * p["f_bb"] + 0.25 * p["f_k1"] + 0.25 * p["f_lock"] + 0.20 * p["f_tk"]
    sax = 0.25 * p["f_long"] + 0.25 * p["f_land"] + 0.20 * p["f_open"] + 0.15 * p["f_ans"] + 0.15 * p["f_breath"]
    assert p["groove"] == pytest.approx(groove) and p["sax"] == pytest.approx(sax)
    want = 0.3 * groove + 0.3 * sax + 0.2 * p["f_new"] + 0.2 * p["f_seam"] - p["p_chrom"]
    assert s.theory == pytest.approx(min(1.0, max(0.0, want)))
    assert (p["f_k1"], p["f_seam"], p["p_chrom"]) == (0.0, 0.0, pytest.approx(0.1))


# ------------------------------------------------------------------------------------------------ new, seam
@pytest.mark.parametrize("src, want", [
    (EXAMPLE, 1.0),                                                 # instrument.md §1: bars 1-8 all differ
    (text(), 1 / 7),                                                # 1-bar bases, no edits: only bar 3 is new
    (text(kick="x......x..x..... x......x.x....x.", hat="x.x.x.x.x.x.x.x. x.x.x.x.x.x.xxx."), 3 / 7),   # 2-bar bases
    (text({"@4 hat": "................"}), 2 / 7),                  # a drop-out bar is something new
    # bar 6 repeats bar 2's written comp, but @5's chord holds into it where bar 2 starts silent: new as heard
    (text({"comp": "x-------........ --------x-------", "@5 comp": "x---------------"}), 5 / 7),
])
def test_something_new_every_bar_f_new(src, want):
    assert part(src, "f_new") == pytest.approx(want)


@pytest.mark.parametrize("over, want", [
    ({}, 1.0),
    ({"@8 snare": "........XXXXXXXX"}, 0.0),                        # §5 fixture: a roll into the seam
    ({"@8 hat": "................", "@8 kick": "................"}, 0.0),   # §5 fixture
    ({"@8 snare": "....x...x.x.x..."}, 0.0),                       # rule 1: 3 non-ghost hits in steps 9-16
    ({"@8 snare": "....x...x...x..."}, 1.0),                       # 2
    ({"@8 hat": "....x.x.xxxxxxxx"}, 1.0),                         # rule 2: 6 cells off the base
    ({"@8 hat": "......x.xxxxxxxx"}, 0.0),                         # 7
    ({"@8 hat": "X.X.X.X.X.X.X.x."}, 0.0),                         # 7 accents are 7 changed cells
    ({"@8 snare": "....x..........."}, 0.0),                       # rule 3: no beat 4 in bar 8
    ({"@1 kick": ".......x..x....."}, 0.0),                        # rule 3: no kick on 1 in bar 1
    ({"@8 hat": "x...x...x......."}, 0.0),                         # rule 4: 3 hats < max(2, 8/2)
    ({"@8 hat": "x...x...x...x..."}, 1.0),                         # 4
    ({"hat": "x.......x.......", "@8 hat": "x..............."}, 0.0),  # 1 < max(2, 2/2)
    ({"hat": "x.......x.......", "@8 hat": "x...x..........."}, 1.0),  # 2 = max(2, 1)
])
def test_no_fill_at_the_seam_f_seam(over, want):
    assert part(text(over), "f_seam") == pytest.approx(want)


# ------------------------------------------------------------------------------------------------ energy
def _e(src: str) -> float:
    return sc(src).knob_parts["E"]


LOW = text(kick="x.......x.......", hat="x...x...x...x...", comp="x---------------", bass="x-------x-------")
HIGH = text(kick="x.......x.......", hat="XXXXXXXXXXXXXXXX", comp="x.x.x.x.x.x.....", bass="x.x.x.x.x.x.....")


def test_energy_formula_and_the_section_5_fixture_bands():
    """§5: quarter hats, an x--- comp and 2 bass hits a bar classify low; 16th accented hats, 6 stabs and 6 bass
    hits a bar classify high, with the test bands .30 / .50."""
    assert _e(LOW) == pytest.approx(0.35 * 64 / 192 + 0.0 + 0.25 * 8 / 48 + 0.25 * 16 / 48)
    assert _e(HIGH) == pytest.approx(0.35 * 160 / 192 + 0.15 * 128 / 160 + 0.25 + 0.25)
    assert vk.energy_band(_e(LOW), (0.30, 0.50)) == "low" and vk.energy_band(_e(HIGH), (0.30, 0.50)) == "high"
    # ghosts weigh 0.5 in H and are not in the accent share: kick 16 + snare 16 + 64 ghost hats at 0.5
    ghosts = text(kick="x.......x.......", hat="g.g.g.g.g.g.g.g.", comp="x---------------", bass="x-------x-------")
    assert _e(ghosts) == pytest.approx(0.35 * (32 + 64 * 0.5) / 192 + 0.25 * 8 / 48 + 0.25 * 16 / 48)


def test_energy_rises_strictly_as_hits_are_added():
    steps = [LOW,
             LOW.replace("hat: x...x...x...x...", "hat: x...x...x...x.x."),        # a hat hit
             LOW.replace("hat: x...x...x...x...", "hat: X...x...x...x.x."),        # an accent too
             LOW.replace("hat: x...x...x...x...", "hat: X...x...x...x.x.").replace("comp: x--------",
                                                                                     "comp: x---x----"),
             LOW.replace("hat: x...x...x...x...", "hat: X...x...x...x.x.").replace("comp: x--------",
                                                                                     "comp: x---x----")
                .replace("bass: x-------x", "bass: x---x---x")]
    values = [_e(s) for s in steps]
    assert all(a < b for a, b in zip(values, values[1:])), values


@pytest.mark.parametrize("e, target, want", [
    (0.45, "mid", 1.0), (0.30, "low", 1.0), (0.80, "high", 1.0),
    (0.42, "low", math.exp(-0.04 / 0.04)),                          # 0.04 above low's top (0.38)
    (0.63, "mid", math.exp(-0.04 / 0.04)),                          # 0.04 above mid's top (0.59)
    (0.30, "high", math.exp(-0.29 / 0.04)),
])
def test_energy_fit_is_exp_of_the_distance_outside_the_band(e, target, want):
    assert vk.ENERGY_BANDS == (0.38, 0.59)
    assert vk.energy_fit(e, target) == pytest.approx(want)


def test_energy_knob_follows_the_brief():
    for energy, match in (("low", 0.0), ("mid", 1.0), ("high", 0.0)):
        s = sc(text(), Brief(energy=energy, mood="wistful", bpm=82, tonic=2))
        assert s.knob_parts["energy_match"] == match
        assert s.knob_parts["f_energy"] == pytest.approx(vk.energy_fit(s.knob_parts["E"], energy))
        assert s.knobs == pytest.approx((s.knob_parts["f_energy"] + s.knob_parts["f_mood"]) / 2)


# ------------------------------------------------------------------------------------------------ mood
def test_mood_section_5_fixtures():
    """Im9 IV13 passes wistful (dorian 34 vs aeolian 26, §5) and fails melancholy; Im7 Vm7 ties aeolian and dorian
    (28 each), so it fails both."""
    melancholy = Brief(energy="mid", mood="melancholy", bpm=82, tonic=2)
    assert sc(text()).knob_parts["f_mood"] == 1.0 and sc(text(), melancholy).knob_parts["f_mood"] == 0.0
    tie = text(vamp="Im7 Vm7 span 2")
    for brief in (BRIEF, melancholy):
        k = sc(tie, brief).knob_parts
        assert (k["mood_aeolian"], k["mood_dorian"], k["mood_margin"], k["f_mood"]) == (28, 28, 0, 0.0)
    # ionian: Im7 = 1 (root) - 1 (b3) + .5 (5th) + 0 (b7, in Alt) = .5; Vm7 = 1 (root) + 0 (its b3 = 10, in Alt)
    # + .5 (5th) + 1 (b7 = 5) = 2.5; 4 bars each: 12
    assert sc(tie, Brief(energy="mid", mood="hopeful", bpm=82, tonic=2)).knob_parts["mood_ionian"] == 12


@pytest.mark.parametrize("vamp, want", [
    # (aeolian, dorian, ionian), worked by hand: 4 bars per chord; w 1 for root, 3rd and 7th, 0.5 otherwise
    (("I", "m9", "IV", "13"), (26, 34, 14)),      # §5: dorian 34, aeolian 26
    (("I", "m7", "V", "m7"), (28, 28, 12)),
    (("II", "m9", "V", "13"), (14, 22, 34)),      # V13's 3rd (11) is in dorian's Alt: 0, not -1
    (("bVI", "maj7", "bVII", "13"), (32, 16, 16)),
    (("I", "m9", "bII", "maj7"), (26, 18, 8)),     # bII's root (1) is in aeolian's Alt: 0, not -1
])
def test_mood_scores_by_hand(vamp, want):
    from lofi_rl.dsl.ast import Chord, Vamp
    a, qa, b, qb = vamp
    got = vk.mood_scores(Vamp((Chord(a, qa), Chord(b, qb)), 2))
    assert (got["aeolian"], got["dorian"], got["ionian"]) == want
    assert vk.mood_scores(Vamp((Chord(a, qa), Chord(b, qb)), 1)) == got      # 4 bars each in either span


def test_mood_margin_of_exactly_2_passes():
    assert vk.mood_pass({"ionian": 30.0, "dorian": 28.0, "aeolian": 1.0}, "ionian")
    assert not vk.mood_pass({"ionian": 29.5, "dorian": 28.0, "aeolian": 1.0}, "ionian")


def test_energy_band_edges_belong_to_the_band_above():
    lo, hi = vk.ENERGY_BANDS
    assert [vk.energy_band(e) for e in (lo - 1e-9, lo, hi - 1e-9, hi)] == ["low", "mid", "mid", "high"]


# ------------------------------------------------------------------------------------------------ P_sax
def test_p_sax_is_zero_on_plain_and_charges_busy_wide_and_chromatic_lines():
    assert sc(text()).p_sax == 0.0
    assert sc(text(A=QUARTERS_A)).p_sax == 0.0                      # 16 notes in a phrase: the band's top edge
    busy = "e5 e6 e5 e3 e2 e1 e2 e3 | e5 e6 e5 e3 e2 e1 e2 e3 | q5 q6 q5 q3 | h2 hr"      # 21 notes in A
    assert sc(text(A=busy)).p_sax == pytest.approx(0.0125 * (21 - 16))
    wide = "q1 q6' " + PLAIN["A"][6:]                               # D3 to B4: 21 semitones, 2 over the band's 19
    assert sc(text(A=wide)).p_sax == pytest.approx(0.02 * 2)
    chromatic = "e#1 e2 e#1 e2 e#1 e2 q3 | e#1 e2 e#1 e2 h3 | h5 hr | w3"   # 5 Eb passing tones that resolve
    s = sc(text(A=chromatic, B=chromatic))
    assert s.theory_parts["p_chrom"] == 0.0                         # every one resolves by step: no P_chrom ...
    assert s.p_sax == pytest.approx(0.5 * (10 / 28 - 0.2))          # ... but 10 of 28 notes are not free
    for parts in (vs.p_sax_parts(sc_view(text(A=busy))),):
        assert parts == {"busy": pytest.approx(0.0625), "wide": 0.0, "chromatic": 0.0}


def sc_view(src: str):
    from lofi_rl.verify.view import make_view
    res = parse(src, vocab=dev_palette().vocab(), mode=BRIEF.mood)
    return make_view(res.ast, BRIEF, compile_loop(res.ast, BRIEF, dev_palette()))


# ------------------------------------------------------------------------------------------------ penalties
def test_repeated_bars_penalty():
    """§5 fixture: static rhythm with the sax only in bars 1 and 5 gives m = 4 (bars 3, 4, 7, 8: chord 2, no sax),
    so P_rep = 0.1 * (4 - 3). A one-cell edit in bar 8 leaves a class of 3: no penalty."""
    sparse = text(A="w5 | wr | wr | wr", B="w5 | wr | wr | wr")
    assert sc(sparse).penalties["bars"] == pytest.approx(0.1)
    assert sc(text(A="w5 | wr | wr | wr", B="w5 | wr | wr | wr", **{"@8 hat": "x.x.x.x.x.x.x.xx"})).penalties[
        "bars"] == 0.0
    assert sc(text()).penalties["bars"] == 0.0 and sc(EXAMPLE).penalties["bars"] == 0.0
    # chord 2's four bars each hold one A3, so m = 4; restriking it mid-bar 8 makes that bar differ, m = 3
    assert sc(text(A="w1 | w2 | w5 | w5", B="w3 | w4 | w5 | w5")).penalties["bars"] == pytest.approx(0.1)
    assert sc(text(A="w1 | w2 | w5 | w5", B="w3 | w4 | w5 | h5 h5")).penalties["bars"] == 0.0


def test_length_penalty_estimates_4_characters_a_token():
    """§5 fixture: padded to 420 tokens -> P_len = .004 * (420 - 360) = .24. The pad is a comment, which the parser
    drops, so only the length changes."""
    def padded(chars: int) -> str:
        base = text()
        return base + "# " + "x" * (chars - len(base) - 3) + "\n"
    assert len(padded(1680)) == 1680
    assert sc(padded(1680)).penalties["len"] == pytest.approx(0.24)
    assert sc(padded(1440)).penalties["len"] == 0.0                 # 360 tokens
    assert sc(padded(1441)).penalties["len"] == pytest.approx(0.004)   # ceil(1441 / 4) = 361
    assert sc(padded(2000)).penalties["len"] == pytest.approx(0.3)     # capped
    assert score(text(), BRIEF, dev_palette(), n_tokens=420).penalties["len"] == pytest.approx(0.24)
    assert sc(padded(1680)).theory == sc(text()).theory and sc(padded(1680)).repairs == 0


# ------------------------------------------------------------------------------------------------ group penalties
def _group(*srcs: str) -> list[float]:
    return group_penalties(list(srcs), [sc(s) for s in srcs])


def test_identical_loops_cost_0_3_per_extra_copy():
    a = text()
    assert _group(a, a) == [0.0, 0.3]
    assert _group(a, a, a) == [0.0, 0.3, 0.3]
    # the same loop with other spacing, a comment and statements in another order is the same compiled loop
    shuffled = "\n".join(reversed(a.splitlines())) + "\n# again\n"
    assert sc(shuffled).digest == sc(a).digest and _group(a, shuffled) == [0.0, 0.3]


def test_near_identical_loops_cost_0_15_and_cousins_pass():
    """The example has 162 events (126 drum hits, 28 sax notes, 8 bar chords). One base hat cell changed adds a hit
    in bars 1, 5 and 7: sim 162/165 >= 0.875, near-identical. Three edits rewritten (bar 3's hats on the off-16ths,
    bar 4's kick on the beats, bar 6's rim accented on 1 and 3) remove 14 events and add 12: sim 148/174 = 0.851,
    a cousin. §5's ramp would charge 0.3 and 0.10; the plan's two levels charge 0.15 and 0. PLAIN, whose 1-bar
    base hat changes in all 8 bars, is near-identical too (140/148)."""
    one_cell = EXAMPLE.replace("hat: x.x.x.x.x.x.x.x. x", "hat: x.x.x.x.x.x.xxx. x")
    cousin = (EXAMPLE.replace("@3 hat: x.x.x.x.x.x.x.xx", "@3 hat: .x.x.x.x.x.x.x..")
              .replace("@4 kick: x......x..x...x.", "@4 kick: x...x...x...x...")
              .replace("@6 rim: ....X..g....X.g.", "@6 rim: X...X...X...X..."))
    assert sc(one_cell).gate_ok and sc(cousin).gate_ok
    assert vp.similarity(vp_events(EXAMPLE), vp_events(one_cell)) == pytest.approx(162 / 165)
    assert vp.similarity(vp_events(EXAMPLE), vp_events(cousin)) == pytest.approx(148 / 174)
    assert _group(EXAMPLE, one_cell) == [0.15, 0.15]
    assert _group(EXAMPLE, cousin) == [0.0, 0.0]
    assert _group(EXAMPLE, EXAMPLE, one_cell) == [0.15, 0.3, 0.15]
    plain_cell = text(hat="x.x.x.x.x.x.xxx.")
    assert vp.similarity(vp_events(text()), vp_events(plain_cell)) == pytest.approx(140 / 148)
    assert _group(text(), plain_cell, cousin) == [0.15, 0.15, 0.0]


def vp_events(src: str) -> frozenset:
    return getattr(sc(src), "_events")


def test_group_penalties_skip_failed_loops_and_check_lengths():
    junk = (FIX / "g1_kick_words.ls").read_text()
    assert _group(junk, junk, text()) == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        group_penalties([text()], [])


def test_a_pickled_score_keeps_its_event_set_and_scores_repeat_across_processes():
    """A reward computed in a worker process comes back whole, and the same completion scores the same in a fresh
    interpreter under another hash seed (no set or dict order reaches a number)."""
    import os
    import pickle
    import subprocess
    import sys
    s = sc(EXAMPLE)
    back = pickle.loads(pickle.dumps(s))
    assert back == s and getattr(back, "_events") == getattr(s, "_events")
    code = ("import dataclasses, json, sys; from pathlib import Path; from lofi_rl.brief import Brief; "
            "from lofi_rl.palette import load_palette; from lofi_rl.verify import score; "
            "s = score(Path(sys.argv[1]).read_text(), Brief('mid', 'wistful', 82, 2), "
            "load_palette(sys.argv[2], verify=False)); print(json.dumps(dataclasses.asdict(s), sort_keys=True))")
    env = {**os.environ, "PYTHONHASHSEED": "12345", "PYTHONPATH": str(FIX.parents[2] / "src")}
    r = subprocess.run([sys.executable, "-c", code, str(FIX / "example.ls"), str(FIX.parents[2] / "palettes" /
                        "dev_gu.yaml")], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(r.stdout) == json.loads(json.dumps(dataclasses.asdict(s), sort_keys=True))


def test_group_penalties_fall_back_to_the_digest_without_the_event_sets():
    """A Score rebuilt by dataclasses.replace or from a JSON log keeps its digest, not its event set: copies still
    cost 0.3, but near-identity cannot be judged, so it costs nothing."""
    a = sc(text())
    bare = dataclasses.replace(a)
    assert not hasattr(bare, "_events") and bare == a
    assert group_penalties([text(), text()], [bare, bare]) == [0.0, 0.3]
    one_cell = sc(EXAMPLE.replace("hat: x.x.x.x.x.x.x.x. x", "hat: x.x.x.x.x.x.xxx. x"))
    assert group_penalties([EXAMPLE, "x"], [dataclasses.replace(sc(EXAMPLE)), one_cell]) == [0.0, 0.0]
