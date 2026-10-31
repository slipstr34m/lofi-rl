"""Hard gates G1-G6 (instrument.md section 2), the section 5 gate fixtures, and gate precedence."""
from pathlib import Path

import pytest

from lofi_rl.dsl import parse

FIX = Path(__file__).parent / "fixtures" / "loops"
EXAMPLE = (FIX / "example.ls").read_text()
A_LINE = "A: q5 e6 e5 q3 e2 e1 | h3 er e4 e5 e6 | h.5 er e4 | q3 q2 hr"


def swap(prefix: str, new: str | None, text: str = EXAMPLE) -> str:
    lines = [new if line.startswith(prefix) else line for line in text.splitlines()]
    return "\n".join(line for line in lines if line is not None) + "\n"


def add(line: str, text: str = EXAMPLE) -> str:
    """Insert a line after `drums:`, inside the loop (prose outside the loop is stripped)."""
    return text.replace("drums: dusty\n", f"drums: dusty\n{line}\n")


def code(text: str, **kw) -> str | None:
    res = parse(text, **kw)
    assert (res.ast is None) == (not res.gate.ok)
    return res.gate.code


@pytest.mark.parametrize("name, expected", [
    ("g5_tempo.ls", "G5"),
    ("g6_vamp_one_chord.ls", "G6"),
    ("g6_kick_9_cells.ls", "G6"),
    ("g4_kit_trap808.ls", "G4"),
    ("g3_no_b.ls", "G3"),
    ("g1_kick_words.ls", "G1"),
])
def test_section5_gate_fixtures(name, expected):
    assert code((FIX / name).read_text()) == expected


def test_g2_finish_length_on_the_pass_fixture():
    res = parse(EXAMPLE, finish_reason="length")
    assert res.gate.code == "G2" and res.ast is None
    assert code(EXAMPLE, finish_reason="stop") is None
    assert code(EXAMPLE, finish_reason=None) is None


def test_gate_details_name_the_line_and_token():
    assert parse((FIX / "g5_tempo.ls").read_text()).gate.detail.startswith("line 6: 'tempo'")
    assert parse((FIX / "g3_no_b.ls").read_text()).gate.detail == "missing: B"
    assert "9 cells" in parse((FIX / "g6_kick_9_cells.ls").read_text()).gate.detail
    assert "trap808" in parse((FIX / "g4_kit_trap808.ls").read_text()).gate.detail


@pytest.mark.parametrize("text", [
    swap("vamp:", "vamp: Im9 IV13 span 3"),
    swap("vamp:", "vamp: Im9 IV13 span"),
    swap("vamp:", "vamp: Im9 IV13 span 2 again"),
    swap("vamp:", "vamp: Im9 foo span 2"),
    swap("A:", A_LINE.replace("q5", "s.5", 1)),                     # no dotted sixteenth
    swap("A:", A_LINE.replace("q5", "q8", 1)),                      # degree 8
    swap("A:", A_LINE.replace("q5", "5q", 1)),                      # degree-first
    swap("A:", A_LINE + " x"),
    swap("@4 kick:", "@9 kick: x......x..x...x."),
    swap("@4 kick:", "@0 kick: x......x..x...x."),
    swap("@4 kick:", "@4 keys: rhodes shell"),
    swap("@4 kick:", "@x kick: x......x..x...x."),
    swap("@3 hat:", "@3 hat: x.x.x.x.x.x.x.xx x.x.x.x.x.x.x.xx"),
    swap("kick:", "kick x......x..x..... x......x.x....x."),
    swap("kick:", "kick: x......x..x..... x......x.x....x. x......x.x....x."),
    swap("kick:", "kick: x...o..x..x..... x......x.x....x."),
    swap("hat:", "hat: x.x.x.x.x.x.x.x. |"),
    swap("comp:", "comp:"),
    swap("comp:", "comp: x-----.x----..x- X-------..X-X--O"),
    swap("bass:", "bass: lock x-------........"),
    swap("keys:", "keys: rhodes"),
    swap("drums:", "drums: dusty brushes"),
    swap("sax:", "sax: tenor alto"),
    swap("@3 hat:", "@٣ hat: x.x.x.x.x.x.x.xx"),                # an Arabic-Indic 3 is not bar 3
    add("## Drums"),
    add("foo: bar"),
    add("Here's the groove:"),
    "",
    "Sorry, I cannot write that loop.\n",
    "# only a comment\n",
])
def test_g1_parse_errors(text):
    assert code(text) == "G1"


@pytest.mark.parametrize("prefix", ["vamp:", "keys:", "comp:", "bass:", "drums:", "kick:", "A:", "B:"])
def test_g3_each_required_statement(prefix):
    res = parse(swap(prefix, None))
    assert res.gate.code == "G3"
    assert res.gate.detail == "missing: " + prefix.rstrip(":")


def test_g3_needs_snare_or_rim_base_not_just_an_edit():
    res = parse(swap("rim:", None))                                  # @6 rim stays
    assert res.gate.code == "G3" and res.gate.detail == "missing: snare or rim"
    assert code(swap("rim:", "snare: ....X.......X... ....X..g....X..g")) is None


def test_sax_is_not_required():
    assert code(swap("sax:", None)) is None


@pytest.mark.parametrize("text, kw", [
    (swap("keys:", "keys: clavinet rootless"), {}),
    (swap("keys:", "keys: rhodes drop2"), {}),
    (swap("sax:", "sax: soprano"), {}),
    (swap("vamp:", "vamp: Im9 IV11 span 2"), {}),                    # 11 is not a quality
    (swap("vamp:", "vamp: Im9 IVm13 span 2"), {}),
    (swap("vamp:", "vamp: #IVm7b5 V7 span 2"), {}),                  # numeral not in the list
    (swap("vamp:", "vamp: bIm7 V7 span 2"), {}),
    (swap("vamp:", "vamp: Im9 Iv13 span 2"), {}),                    # mixed-case numeral
    (swap("vamp:", "vamp: iifoo V7 span 2"), {}),
    (swap("vamp:", "vamp: I IV13 span 2"), {}),                      # bare numeral needs a mode
    (swap("vamp:", "vamp: bII IV13 span 2"), {"mode": "dorian"}),    # not diatonic in dorian
])
def test_g4_not_in_vocab(text, kw):
    assert code(text, **kw) == "G4"


@pytest.mark.parametrize("text", [
    add("tempo: 90"),
    add("Tempo = 90"),
    add("bpm: 82"),
    add("key: D dorian"),
    add("mode: dorian"),
    add("swing: 55"),
    add("reverb: 0.3"),
    add("rain: on"),
    add("D: x......x..x....."),                                     # a note name as a head
    swap("vamp:", "vamp: Dm9 G13 span 2"),
    swap("vamp:", "vamp: C4 IV13 span 2"),
    swap("vamp:", "vamp: Im9 IV13 span 2 minor"),
    swap("A:", A_LINE.replace("q5", "q5 C4", 1)),
    swap("A:", A_LINE.replace("q5", "qD", 1)),
    swap("A:", A_LINE.replace("q5", "f#4", 1)),
    swap("@4 kick:", "@4 vamp: Im9 IV9"),
    swap("@4 kick:", "@4 sax: alto"),
    swap("keys:", "keys: rhodes reverb"),
    swap("keys:", "keys: Ebmaj7 rootless"),
    swap("drums:", "drums: dusty swing"),
    swap("hat:", "hat: x.x.x.x.x.x.x.x. lowpass"),
    swap("sax:", "sax: 90bpm"),
])
def test_g5_forbidden_content(text):
    assert code(text) == "G5"


def test_g5_only_when_the_failing_token_is_forbidden():
    assert code(swap("A:", A_LINE.replace("e6", "E6", 1))) is None   # E6 case-folds to e6
    assert code(add("# 82 bpm in D dorian, reverb on the keys")) is None   # a comment
    assert code("Here's one in D dorian at 82 BPM:\n" + EXAMPLE) is None   # prose before the loop


# A `word:` line with a forbidden word is a statement wherever it sits, not prose. This used to
# be `is None`: the line was stripped as prose for free, so G5 only fired between two statements.
@pytest.mark.parametrize("text", [
    "Key: D dorian, 82 BPM\n" + EXAMPLE,
    "Here's a loop:\nTempo = 82\n```\n" + EXAMPLE + "```\n",
    EXAMPLE + "Reverb: a little on the keys\nEnjoy!\n",
])
def test_g5_forbidden_statement_outside_the_loop(text):
    assert code(text) == "G5"


# Each statement reports its worst failure, with the same order as across lines: G5, G1, G4, G6.
@pytest.mark.parametrize("prefix, line, expected", [
    ("keys:", "keys: clavinet rootless extra", "G1"),
    ("vamp:", "vamp: Im9 IV11 span 3", "G1"),
    ("vamp:", "vamp: Im9 IV13 span 3 minor", "G5"),
    ("keys:", "keys: Dm9", "G5"),
    ("drums:", "drums: dusty Dm9", "G5"),
    ("kick:", "kick: x...o..x..x..... C4", "G5"),
    ("A:", A_LINE.replace("q5", "q8", 1).replace("q3", "C4", 1), "G5"),
    ("kick:", "kick: x......x. x...o..x.x....x.", "G1"),             # G6 grid, G1 grid
])
def test_worst_failure_on_one_line_wins(prefix, line, expected):
    assert code(swap(prefix, line)) == expected


@pytest.mark.parametrize("word", ["A", "a", "B", "b", "BD", "bd", "SD", "Kick", "VAMP", "Sax"])
def test_keyword_without_its_colon_is_g1_in_any_case(word):
    assert code(add(f"{word} x......x..x.....")) == "G1"                  # never a note name


@pytest.mark.parametrize("text", [
    swap("vamp:", "vamp: Im9 Im9 span 2"),
    swap("vamp:", "vamp: Im9 IV13 V7 span 2"),
    swap("vamp:", "vamp: span 2"),
    swap("vamp:", "vamp: IIm9 iim9 span 2"),                         # equal after aliasing
    swap("kick:", "kick: x......x..x x......x.x....x."),             # 11 cells
    swap("hat:", "hat: x.x.x.x.x.x.x.x.x.x.x x.x.x.x.x.x.xxx."),     # 21 cells
    swap("@3 hat:", "@3 hat: x.x.x.x.x."),
    swap("A:", "A: q5 e6 e5 q3 e2 e1 | h3 er e4 e5 e6 | h.5 er s4"),  # 47 steps, 17 short
    swap("A:", A_LINE + " w5 s5"),                                   # 81 steps, 17 long
    swap("A:", "A:"),
    swap("A:", "A: | | |"),
    swap("B:", "B: wr wr wr wr"),                                    # no notes
    swap("B:", "B: wr wr wr wr e1 e1"),                              # notes only past step 64
])
def test_g6_structure(text):
    assert code(text) == "G6"


def test_g6_kick_silent_in_all_bars():
    silent = swap("kick:", "kick: ................ ................")
    assert code(silent) is None                                      # @4 kick still plays
    res = parse(swap("@4 kick:", None, silent))
    assert res.gate.code == "G6" and res.gate.detail == "kick is silent in all 8 bars"
    ghosts = swap("kick:", "kick: g............... ................", swap("@4 kick:", None))
    assert code(ghosts) is None                                      # a ghost is not silence


def test_g6_snare_and_rim_silent_in_all_bars():
    silent = swap("rim:", "rim: ................")
    assert code(silent) is None                                      # @6 rim still plays
    res = parse(swap("@6 rim:", None, silent))
    assert res.gate.code == "G6" and res.gate.detail == "snare and rim are silent in all 8 bars"


def test_precedence_g2_first():
    truncated = EXAMPLE[: EXAMPLE.index("B:") + 9]
    assert code(truncated, finish_reason="length") == "G2"
    assert code(add("tempo: 90"), finish_reason="length") == "G2"
    assert code(EXAMPLE[: EXAMPLE.index("B:")]) == "G3"


def test_precedence_g5_over_g1_over_g4_over_g3_over_g6():
    g1 = (FIX / "g1_kick_words.ls").read_text()
    assert code(add("tempo: 90", g1)) == "G5"
    assert code(swap("keys:", "keys: clavinet rootless", g1)) == "G1"
    assert code(swap("B:", None, swap("drums:", "drums: trap808"))) == "G4"
    assert code(swap("B:", None, swap("vamp:", "vamp: Im9"))) == "G3"
    assert code(swap("vamp:", "vamp: Im9", g1)) == "G1"              # G6 on line 1 hides nothing


def test_repairs_are_kept_when_a_gate_fails():
    res = parse(swap("B:", None, swap("kick:", "bd: x......x..x..... x......x.x....x.")))
    assert res.gate.code == "G3"
    assert [r.code for r in res.repairs] == ["alias_lane"]


@pytest.mark.parametrize("token, expected", [
    ("eq6", "G1"), ("qe3", "G1"), ("e", "G1"), ("b4x", "G1"),        # garbled items stay G1
    ("f#4", "G5"), ("c4", "G5"), ("qD", "G5"), ("Gb4", "G5"), ("eq", "G5"), ("82bpm", "G5"),
    ("Eb4", None),                                                    # case-folds to eb4, a valid item
])
def test_g5_labels_forbidden_tokens_not_typos(token, expected):
    assert code(swap("A:", A_LINE.replace("q5", token, 1))) == expected


def test_vocab_helpers():
    from lofi_rl.dsl.vocab import canonical_quality, diatonic_quality, forbidden_kind
    assert canonical_quality("Maj7") == "maj7" and canonical_quality("MIN7") == "m7"
    assert canonical_quality("7SUS") == "7sus" and canonical_quality("M7B5") is None
    assert canonical_quality("m7") == "m7" and canonical_quality("11") is None
    assert diatonic_quality("#IV", "dorian") is None and diatonic_quality("bVII", "hopeful") is None
    assert forbidden_kind("Dm9") and forbidden_kind("C4") and forbidden_kind("rain")
    assert forbidden_kind("four") is None and forbidden_kind("keys") is None
    for chord in ["G7sus", "C9sus4", "Gsus4", "CMaj7", "E7alt", "C/E", "Dm7/G", "C6/9", "Cm(maj7)"]:
        assert forbidden_kind(chord), chord
    assert forbidden_kind("BD") is None and forbidden_kind("CAFE") is None
