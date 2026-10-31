"""LoopScript v1 parser: the section 1 example, the AST shape, vocab overrides, the public API."""
from pathlib import Path

import pytest

from lofi_rl.dsl import (DEFAULT_VOCAB, Chord, Edit, Gate, Item, Keys, Lane, Loop, ParseResult,
                         Repair, Vamp, Vocab, parse)

FIX = Path(__file__).parent / "fixtures" / "loops"
EXAMPLE = (FIX / "example.ls").read_text()
CLEAN = ["example.ls", "one_bar_base.ls", "shaker_edits.ls", "span1_dotted.ls"]


def swap(prefix: str, new: str | None, text: str = EXAMPLE) -> str:
    lines = [new if line.startswith(prefix) else line for line in text.splitlines()]
    return "\n".join(line for line in lines if line is not None) + "\n"


def n(dur: int, degree: int, acc: int = 0, octave: int = 0) -> Item:
    return Item(dur, False, degree, acc, octave)


def r(dur: int) -> Item:
    return Item(dur, rest=True)


EXPECTED = Loop(
    vamp=Vamp((Chord("I", "m9"), Chord("IV", "13")), 2),
    keys=Keys("rhodes", "rootless"),
    comp=("x-----.x----..x-", "x-------..x-x---"),
    bass=("x-------..x-x---", "x-----.x-x------"),
    kit="dusty",
    lanes=(
        Lane("kick", ("x......x..x.....", "x......x.x....x.")),
        Lane("rim", ("....X.......X...", "....X..g....X..g")),
        Lane("hat", ("x.x.x.x.x.x.x.x.", "x.x.x.x.x.x.xxx.")),
        Lane("ohat", ("................", "..............x.")),
    ),
    edits=(
        Edit(3, "hat", "x.x.x.x.x.x.x.xx"),
        Edit(4, "kick", "x......x..x...x."),
        Edit(5, "comp", "x-----.x--x-..x-"),
        Edit(6, "rim", "....X..g....X.g."),
        Edit(7, "ohat", "......x........."),
        Edit(8, "comp", "x-----.x----x-x-"),
    ),
    sax="tenor",
    a=(n(4, 5), n(2, 6), n(2, 5), n(4, 3), n(2, 2), n(2, 1),
       n(8, 3), r(2), n(2, 4), n(2, 5), n(2, 6),
       n(12, 5), r(2), n(2, 4),
       n(4, 3), n(4, 2), r(8)),
    b=(n(4, 1, octave=1), n(2, 7), n(2, 1, octave=1), n(4, 5), n(4, 3),
       r(2), n(2, 3), n(2, 4), n(2, 5, acc=-1), n(8, 5),
       r(4), n(2, 4), n(2, 3), n(4, 2), n(4, 1),
       n(12, 6), r(4)),
)


def test_example_parses_with_zero_repairs():
    res = parse(EXAMPLE)
    assert isinstance(res, ParseResult)
    assert res.gate == Gate(True, None, "")
    assert res.repairs == []
    assert res.penalty == 0.0


def test_example_ast_is_exact():
    ast = parse(EXAMPLE).ast
    assert ast == EXPECTED
    assert ast.vamp.chords[0].text == "Im9" and ast.vamp.chords[1].text == "IV13"
    assert ast.backbeat == "rim" and not ast.bass_lock
    assert ast.lane("kick") == ("x......x..x.....", "x......x.x....x.") and ast.lane("snare") is None
    assert sum(i.dur for i in ast.a) == 64 and sum(i.dur for i in ast.b) == 64
    assert sum(1 for i in ast.a + ast.b if not i.rest and i.dur >= 8) == 4   # "4 notes last 8+"
    assert ast.b[8] == Item(2, False, 5, -1, 0)                             # eb5, the blue b5


@pytest.mark.parametrize("name", CLEAN)
def test_loop_fixtures_parse_clean(name):
    res = parse((FIX / name).read_text())
    assert res.gate.ok, res.gate
    assert res.repairs == [] and res.penalty == 0.0


def test_dotted_notes_and_octave_marks():
    ast = parse((FIX / "span1_dotted.ls").read_text()).ast
    assert ast.vamp == Vamp((Chord("I", "m11"), Chord("II", "m7")), 1)
    assert ast.a[:5] == (n(6, 5), n(2, 6), n(4, 5, octave=1), n(3, 3), n(1, 2))
    assert ast.a[5] == n(12, 3, octave=-1)                                   # h.3,
    assert ast.a[12] == n(1, 4, acc=1)                                       # s#4
    assert ast.b[:5] == (n(6, 1, octave=1), n(2, 7), n(4, 5), n(3, 3, octave=1), n(1, 2, octave=1))
    assert ast.b[5] == n(24, 3)                                              # w.3 spans bar lines
    assert ast.sax is None


def test_multiple_octave_marks_are_netted():
    text = swap("A:", "A: q5'' q5,, q5', q5,,, | w1 | h1 h1 | w1")
    a = parse(text).ast.a
    assert [i.octave for i in a[:4]] == [2, -2, 0, -3]


def test_sax_statement_is_optional_and_checked_when_present():
    assert parse(swap("sax:", None)).ast.sax is None
    assert parse(swap("sax:", "sax: alto")).ast.sax == "alto"
    assert parse(swap("sax:", "sax: soprano")).gate.code == "G4"


def test_bass_lock_and_rest_items():
    ast = parse((FIX / "shaker_edits.ls").read_text()).ast
    assert ast.bass == () and ast.bass_lock
    assert ast.lane("shaker") == ("g.x.g.x.g.x.g.x.", "g.x.g.x.g.x.ggx.")
    assert [e.target for e in ast.edits] == ["bass", "comp", "shaker", "bass", "kick"]
    assert ast.a[-1] == r(4) and ast.b[5] == n(4, 3, acc=-1)


def test_statement_order_does_not_change_the_ast():
    lines = EXAMPLE.splitlines()
    shuffled = "\n".join(lines[::-1]) + "\n"
    assert parse(shuffled).ast == EXPECTED


def test_failed_gate_has_no_ast():
    res = parse(swap("B:", None))
    assert res.ast is None and not res.gate.ok


def test_vocab_overrides_names():
    vocab = Vocab(keys=["Tine"], kits={"lofi"}, sax=("tenor",))
    assert parse(EXAMPLE, vocab).gate.code == "G4"                            # rhodes, dusty unknown
    text = swap("keys:", "keys: tine rootless", swap("drums:", "drums: LOFI"))
    res = parse(text, vocab)
    assert res.gate.ok and res.ast.keys == Keys("tine", "rootless") and res.ast.kit == "lofi"


def test_vocab_name_aliases_are_free():
    vocab = Vocab(name_aliases={"Fender": "rhodes"})
    res = parse(swap("keys:", "keys: fender rootless"), vocab)
    assert res.gate.ok and res.ast.keys.instrument == "rhodes"
    assert res.repairs == [Repair("alias_name", 2, "fender", "rhodes", False)]
    assert res.penalty == 0.0


def test_name_alias_applies_only_where_it_lands_on_a_legal_name():
    vocab = Vocab(name_aliases={"jazz": "guitar"})
    res = parse(swap("keys:", "keys: jazz rootless"), vocab)
    assert res.ast.keys.instrument == "guitar"
    assert res.repairs == [Repair("alias_name", 2, "jazz", "guitar", False)]
    assert parse(swap("drums:", "drums: jazz"), vocab).ast.kit == "jazz"
    res = parse(swap("sax:", "sax: jazz"), vocab)
    assert res.gate.code == "G4" and res.repairs == []


def test_default_vocab_placeholders():
    assert DEFAULT_VOCAB.keys == {"rhodes", "piano", "guitar", "organ", "wurli", "ep"}
    assert DEFAULT_VOCAB.kits == {"dusty", "brushes", "jazz"}
    assert DEFAULT_VOCAB.sax == {"tenor", "alto"}
    assert DEFAULT_VOCAB.voicings == {"rootless", "shell", "spread", "quartal"}
    assert len(DEFAULT_VOCAB.numerals) == 11 and len(DEFAULT_VOCAB.qualities) == 14
    assert hash(DEFAULT_VOCAB) == hash(Vocab())


def test_unknown_mode_is_a_caller_error():
    with pytest.raises(ValueError):
        parse(EXAMPLE, mode="lydian-ish")
    assert parse(EXAMPLE, mode="wistful").gate.ok and parse(EXAMPLE, mode="dorian").gate.ok


def test_ast_is_frozen_and_hashable():
    ast = parse(EXAMPLE).ast
    with pytest.raises(AttributeError):
        ast.kit = "jazz"
    assert hash(ast) == hash(parse(EXAMPLE).ast)
