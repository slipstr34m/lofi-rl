"""Adversarial review of lofi_rl.dsl against docs/design/instrument.md sections 1, 2 and 5.

Each test is named for a defect and fails against the implementation it was written for.
"""
from pathlib import Path

import pytest

from lofi_rl.dsl import Vocab, expand, parse

FIX = Path(__file__).parent / "fixtures" / "loops"
EXAMPLE = (FIX / "example.ls").read_text()
A_BODY = "q5 e6 e5 q3 e2 e1 | h3 er e4 e5 e6 | h.5 er e4 | q3 q2 hr"
KICK_BODY = "x......x..x..... x......x.x....x."


def swap(prefix: str, new: str | None, text: str = EXAMPLE) -> str:
    lines = [new if line.startswith(prefix) else line for line in text.splitlines()]
    return "\n".join(line for line in lines if line is not None) + "\n"


def add(line: str, text: str = EXAMPLE) -> str:
    """Insert a line after `drums:`, inside the loop."""
    return text.replace("drums: dusty\n", f"drums: dusty\n{line}\n")


def code(text: str, **kw) -> str | None:
    return parse(text, **kw).gate.code


# Grammar: comment = ("#"|"//") (SPACE|EOL) {any}. A line-start "#" or "//" followed by any
# other character is not a comment, so inside the loop it is a parse error. repair.py's
# `_FULL_COMMENT` has no space/EOL lookahead and drops such lines silently: a mistyped edit
# like `#3 hat:` vanishes and the loop passes with zero repairs.
@pytest.mark.parametrize("text", [
    swap("@3 hat:", "#3 hat: x.x.x.x.x.x.x.xx"),
    add("#notes"),
    add("//notes"),
])
def test_line_start_hash_or_slashes_without_a_following_space_is_not_a_comment(text):
    assert code(text) == "G1"


# G5 covers tempo/key/swing words. A `tempo: 90` line placed first or last is a statement with
# a forbidden head, not prose; stripping it as prose (free repair) means G5 only fires when the
# section 5 fixture `tempo: 90` happens to sit between two LoopScript lines.
@pytest.mark.parametrize("text", [
    "tempo: 90\n" + EXAMPLE,
    EXAMPLE + "tempo: 90\n",
    "key: D dorian\n" + EXAMPLE,
    EXAMPLE + "swing: 58\n",
])
def test_forbidden_statement_before_or_after_the_loop_is_g5_not_prose(text):
    assert code(text) == "G5"


# Section 5 seam fixture: example.ls (a rim loop) + `@8 snare: ........XXXXXXXX` -> f_seam 0.
# The snare edit is not a second base lane, yet it is dropped as `snare_and_rim` (0.02), so the
# fill never reaches the verifiers.
def test_snare_edit_on_a_rim_loop_is_kept_for_the_seam_fixture():
    res = parse(EXAMPLE + "@8 snare: ........XXXXXXXX\n")
    assert res.gate.ok, res.gate
    assert res.repairs == []
    assert expand(res.ast).drums["snare"][7] == "........XXXXXXXX"


# Keywords are case-folded, so `A`/`a` and `BD`/`bd` must get the same gate. The uppercase
# forms match the note-name/absolute-chord regexes and are labelled G5 when the colon is missing.
@pytest.mark.parametrize("prefix, upper, lower", [
    ("A:", "A " + A_BODY, "a " + A_BODY),
    ("kick:", "BD " + KICK_BODY, "bd " + KICK_BODY),
])
def test_uppercase_keyword_missing_its_colon_is_g1_like_its_lowercase_form(prefix, upper, lower):
    assert (code(swap(prefix, upper)), code(swap(prefix, lower))) == ("G1", "G1")


def test_edit_on_a_phrase_gets_the_same_gate_whatever_the_keyword_case():
    assert code(add("@5 A: q5 e6 e5 q3")) == code(add("@5 a: q5 e6 e5 q3"))
    assert code(add("@5 B: q5 e6 e5 q3")) == code(add("@5 b: q5 e6 e5 q3"))


# Absolute chords built on the palette's own sus qualities slip past `_ABS_CHORD` (one quality
# prefix, then only digits/accidentals) and are labelled G1 instead of G5.
@pytest.mark.parametrize("vamp", [
    "vamp: G7sus Cmaj9 span 2",
    "vamp: Im9 C9sus4 span 2",
    "vamp: Im9 D7sus4 span 2",
])
def test_absolute_sus_chord_is_labelled_g5(vamp):
    assert code(swap("vamp:", vamp)) == "G5"


# parse.py documents "G5 if any failing token is forbidden content", but the span and arity
# checks run first, so an absolute chord earlier on the same line is never examined.
@pytest.mark.parametrize("prefix, line", [
    ("vamp:", "vamp: Dm9 G13 span 3"),
    ("keys:", "keys: Ebmaj7 rootless extra"),
])
def test_forbidden_token_is_not_masked_by_a_later_span_or_arity_error(prefix, line):
    assert code(swap(prefix, line)) == "G5"


# int() refuses strings over 4300 digits (Python >= 3.11); normalize_space and _edit_head call
# int() on the raw `@<digits>`, so parse raises ValueError instead of returning a gate.
def test_edit_bar_with_thousands_of_digits_is_g1_not_a_crash():
    text = swap("@3 hat:", "@" + "1" * 5000 + " hat: x.x.x.x.x.x.x.xx")
    assert code(text) == "G1"


# Vocab.name_aliases is one flat map applied before the membership check in every name slot,
# so an alias meant for keys ("jazz" guitar) rewrites the legal kit name "jazz" and fails G4.
def test_name_alias_does_not_rewrite_a_legal_name_in_another_slot():
    vocab = Vocab(name_aliases={"jazz": "guitar"})
    res = parse(swap("drums:", "drums: jazz"), vocab)
    assert res.gate.ok, res.gate
    assert res.ast.kit == "jazz"
