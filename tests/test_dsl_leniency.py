"""Leniency steps 1-5 and the penalized repairs of step 6 (instrument.md section 2)."""
from pathlib import Path

import pytest

from lofi_rl.dsl import Chord, Edit, Item, Repair, parse, phrase_text

FIX = Path(__file__).parent / "fixtures" / "loops"
EXAMPLE = (FIX / "example.ls").read_text()
BASE = parse(EXAMPLE).ast
A_BODY = "q5 e6 e5 q3 e2 e1 | h3 er e4 e5 e6 | h.5 er e4 | q3 q2 hr"
KICK0 = "x......x..x....."


def swap(prefix: str, new: str | None, text: str = EXAMPLE) -> str:
    lines = [new if line.startswith(prefix) else line for line in text.splitlines()]
    return "\n".join(line for line in lines if line is not None) + "\n"


def add(line: str, text: str = EXAMPLE) -> str:
    return text.replace("drums: dusty\n", f"drums: dusty\n{line}\n")


def free(code: str, line: int, before: str, after: str) -> Repair:
    return Repair(code, line, before, after, False)


def paid(code: str, line: int, before: str, after: str) -> Repair:
    return Repair(code, line, before, after, True)


# -- steps 1-4: prose, comments, whitespace, case ------------------------------------------
def test_leniency_pass_fixture_same_ast_plus_strip_prose():
    res = parse((FIX / "example_leniency.ls").read_text())
    assert res.gate.ok and res.ast == BASE
    assert res.repairs == [free("strip_prose", 1, "Here's a loop:\n```\n```", "")]
    assert res.penalty == 0.0


def test_leniency_fail_fixture_lowercase_x_changes_the_ast():
    res = parse((FIX / "leniency_lower_x.ls").read_text())
    assert res.gate.ok and res.repairs == []
    assert res.ast != BASE
    assert res.ast.lane("rim") == ("....x.......x...", "....x..g....x..g")


def test_prose_after_the_last_keyword_line_is_stripped():
    res = parse("Sure!\n" + EXAMPLE + "\nThis loop leans on the rim.\n")
    assert res.ast == BASE
    assert res.repairs == [free("strip_prose", 1, "Sure!\nThis loop leans on the rim.", "")]


def test_comment_and_blank_lines_outside_the_loop_are_not_prose():
    assert parse("# my loop\n\n" + EXAMPLE + "\n// end\n").repairs == []


def test_hash_without_a_space_outside_the_loop_is_prose():
    res = parse("#Loop\n" + EXAMPLE + "//end\n")                 # not comments: no space after
    assert res.ast == BASE and res.repairs == [free("strip_prose", 1, "#Loop\n//end", "")]


@pytest.mark.parametrize("text", [
    EXAMPLE.replace("\n", "\r\n"),
    EXAMPLE.replace("\n", "\n\n\n"),
    EXAMPLE.replace(" ", "\t"),
    EXAMPLE.replace(" ", "   "),
    EXAMPLE.replace(": ", " = ").replace("@3 hat = ", "@3 hat="),
    EXAMPLE.replace(": ", " : "),
    "   " + EXAMPLE.replace("\n", "  \n  "),
    EXAMPLE.replace("@3 hat:", "@03 hat:"),                       # leading zeros of an edit bar
    add("# notes"),
    add("// notes"),
    EXAMPLE.replace("\n", "  # notes\n"),
    EXAMPLE.replace("\n", " // notes\n"),
])
def test_silent_normalization_keeps_the_ast(text):
    res = parse(text)
    assert res.gate.ok, res.gate
    assert res.repairs == [] and res.ast == BASE


def test_hash_inside_a_token_is_a_sharp_not_a_comment():
    text = swap("A:", "A: q5 e#4 " + A_BODY[6:] + " # call")
    text = swap("vamp:", "vamp: Im9 IVmaj7#11 span 2  # vamp", text)
    res = parse(text)
    assert res.gate.ok and res.repairs == []
    assert res.ast.a[1] == Item(2, False, 4, 1, 0)
    assert res.ast.vamp.chords[1] == Chord("IV", "maj7#11")


def test_case_folds_keywords_names_durations_but_not_cells_or_chords():
    text = (EXAMPLE.replace("rhodes rootless", "Rhodes ROOTLESS").replace("kick:", "Kick:")
            .replace("A: q5 e6", "a: Q5 E6").replace("span 2", "SPAN 2").replace("eb5", "EB5"))
    res = parse(text)
    assert res.gate.ok and res.repairs == [] and res.ast == BASE
    assert parse(swap("vamp:", "vamp: IM9 IV13 span 2")).ast.vamp.chords[0] == Chord("I", "maj9")


# -- step 5: aliases, logged with no penalty ------------------------------------------------
@pytest.mark.parametrize("prefix, alias, canonical, line", [
    ("kick:", "bd", "kick", 6), ("hat:", "hh", "hat", 8), ("ohat:", "oh", "ohat", 9),
    ("@3 hat:", "@3 HH", "hat", 10), ("@4 kick:", "@4 Bd", "kick", 11),
])
def test_lane_aliases(prefix, alias, canonical, line):
    old = next(x for x in EXAMPLE.splitlines() if x.startswith(prefix))
    res = parse(swap(prefix, alias + old[len(prefix) - 1:]))
    assert res.ast == BASE and res.penalty == 0.0
    assert res.repairs == [free("alias_lane", line, alias.split()[-1], canonical)]


def test_snare_and_shaker_aliases():
    one = (FIX / "one_bar_base.ls").read_text()
    res = parse(one.replace("snare:", "sd:"))
    assert res.ast == parse(one).ast and res.repairs == [free("alias_lane", 7, "sd", "snare")]
    shk = (FIX / "shaker_edits.ls").read_text()
    res = parse(shk.replace("shaker:", "shk:").replace("@6 shk:", "@6 SHK:"))
    assert res.ast == parse(shk).ast
    assert res.repairs == [free("alias_lane", 9, "shk", "shaker"), free("alias_lane", 12, "SHK", "shaker")]


def test_dash_reads_as_rest_in_drum_grids_only():
    res = parse(swap("kick:", "kick: x------x--x----- x......x.x....x."))
    assert res.ast == BASE
    assert res.repairs == [free("alias_dash", 6, "x------x--x-----", KICK0)]
    res = parse(swap("@7 ohat:", "@7 ohat: ------x---------"))
    assert res.ast == BASE and [r.code for r in res.repairs] == ["alias_dash"]
    assert parse(EXAMPLE).ast.comp[0] == "x-----.x----..x-"                 # "-" holds in comp


@pytest.mark.parametrize("token, chord", [
    ("IVM7", Chord("IV", "maj7")), ("IVΔ7", Chord("IV", "maj7")), ("II-7", Chord("II", "m7")),
    ("IImin7", Chord("II", "m7")), ("VIIø", Chord("VII", "m7b5")), ("VIIø7", Chord("VII", "m7b5")),
    ("VII°7", Chord("VII", "dim7")), ("Vsus", Chord("V", "7sus")), ("V9sus4", Chord("V", "7sus")),
    ("IVMaj7", Chord("IV", "maj7")), ("IVM9", Chord("IV", "maj9")), ("II-9", Chord("II", "m9")),
    ("IVΔ", Chord("IV", "maj7")), ("V7SUS", Chord("V", "7sus")), ("II-7b5", Chord("II", "m7b5")),
])
def test_quality_aliases(token, chord):
    res = parse(swap("vamp:", f"vamp: Im9 {token} span 2"))
    assert res.gate.ok and res.ast.vamp.chords[1] == chord
    assert res.repairs == [free("alias_quality", 1, token, chord.text)]


@pytest.mark.parametrize("token, chord", [
    ("ii9", Chord("II", "m9")), ("ii7", Chord("II", "m7")), ("iim7", Chord("II", "m7")),
    ("iv11", Chord("IV", "m11")), ("bvii9", Chord("bVII", "m9")), ("vii°7", Chord("VII", "dim7")),
    ("iiø7", Chord("II", "m7b5")), ("ii-9", Chord("II", "m9")),
])
def test_lowercase_numeral_means_minor(token, chord):
    res = parse(swap("vamp:", f"vamp: Im9 {token} span 2"))
    assert res.gate.ok and res.ast.vamp.chords[1] == chord
    assert res.repairs == [free("alias_minor_numeral", 1, token, chord.text)]


def test_missing_span_becomes_2():
    res = parse(swap("vamp:", "vamp: Im9 IV13"))
    assert res.ast == BASE and res.penalty == 0.0
    assert res.repairs == [free("default_span", 1, "vamp: Im9 IV13", "vamp: Im9 IV13 span 2")]


# -- step 6: repairs, 0.02 each, capped at 0.1 ----------------------------------------------
@pytest.mark.parametrize("cells", [12, 13, 14, 15])
def test_short_grid_is_padded_with_rests(cells):
    res = parse(swap("kick:", f"kick: {KICK0[:cells]} x......x.x....x."))
    assert res.ast == BASE and res.penalty == 0.02
    assert res.repairs == [paid("pad_grid", 6, KICK0[:cells], KICK0)]


@pytest.mark.parametrize("cells", [17, 18, 19, 20])
def test_long_grid_is_truncated(cells):
    long = (KICK0 + "xXgx")[:cells]
    res = parse(swap("kick:", f"kick: {long} x......x.x....x."))
    assert res.ast == BASE and res.repairs == [paid("trunc_grid", 6, long, KICK0)]


def test_short_comp_and_edit_grids_are_padded():
    res = parse(swap("@5 comp:", "@5 comp: x-----.x--x-.", swap("comp:", "comp: x-----.x----..x- x-------..x-x-")))
    assert res.ast.comp == ("x-----.x----..x-", "x-------..x-x-..")
    assert res.ast.edits[2].grid == "x-----.x--x-...."
    assert [(r.code, r.line) for r in res.repairs] == [("pad_grid", 3), ("pad_grid", 12)]


def test_short_phrase_is_padded_with_a_rest():
    short = A_BODY[: -len(" hr")]
    res = parse(swap("A:", "A: " + short))
    assert res.ast == BASE
    assert res.repairs == [paid("pad_phrase", 17, short, A_BODY)]


def test_phrase_16_short_and_odd_gaps_pad_with_writable_rests():
    res = parse(swap("A:", "A: " + A_BODY[: -len(" | q3 q2 hr")]))
    assert res.ast.a[-1] == Item(16, rest=True) and res.penalty == 0.02
    res = parse(swap("A:", "A: " + A_BODY.replace("q2 hr", "q2 e.r")))      # 59 steps
    assert [i.dur for i in res.ast.a[-3:]] == [3, 4, 1] and all(i.rest for i in res.ast.a[-3:])
    assert phrase_text(res.ast.a).endswith("q3 q2 e.r qr sr")


def test_long_phrase_is_truncated():
    res = parse(swap("A:", "A: " + A_BODY + " q5"))
    assert res.ast == BASE
    assert res.repairs == [paid("trunc_phrase", 17, A_BODY + " q5", A_BODY)]
    res = parse(swap("A:", "A: " + A_BODY.replace("hr", "w5")))             # w5 crosses the end
    assert res.ast.a[-1] == Item(8, False, 5) and res.repairs[0].code == "trunc_phrase"
    res = parse(swap("A:", "A: " + A_BODY.replace("q2 hr", "e2 w5")))       # 10 steps of room
    assert res.ast.a[-2:] == (Item(8, False, 5), Item(2, rest=True))
    res = parse(swap("A:", "A: " + A_BODY + " w5"))                         # 16 over: still repaired
    assert res.ast == BASE and res.penalty == 0.02


@pytest.mark.parametrize("mode, vamp, chords", [
    ("dorian", "I IV", (Chord("I", "m7"), Chord("IV", "7"))),
    ("wistful", "ii V", (Chord("II", "m7"), Chord("V", "m7"))),
    ("hopeful", "I IV", (Chord("I", "maj7"), Chord("IV", "maj7"))),
    ("ionian", "VII V", (Chord("VII", "m7b5"), Chord("V", "7"))),
    ("melancholy", "bVI bVII", (Chord("bVI", "maj7"), Chord("bVII", "7"))),
    ("aeolian", "II I", (Chord("II", "m7b5"), Chord("I", "m7"))),
])
def test_bare_numeral_becomes_the_modes_diatonic_seventh(mode, vamp, chords):
    res = parse(swap("vamp:", f"vamp: {vamp} span 2"), mode=mode)
    assert res.gate.ok and res.ast.vamp.chords == chords
    tokens = vamp.split()
    assert res.repairs == [paid("bare_numeral", 1, t, c.text) for t, c in zip(tokens, chords)]
    assert res.penalty == 0.04


@pytest.mark.parametrize("token, chord", [
    ("ivmaj7", Chord("IV", "maj7")), ("ii13", Chord("II", "13")), ("v7b9", Chord("V", "7b9")),
    ("ivM9", Chord("IV", "maj9")), ("vsus", Chord("V", "7sus")),
])
def test_numeral_case_contradicting_the_quality_lets_the_quality_win(token, chord):
    res = parse(swap("vamp:", f"vamp: Im9 {token} span 2"))
    assert res.gate.ok and res.ast.vamp.chords[1] == chord
    assert res.repairs == [paid("case_conflict", 1, token, chord.text)]


def test_duplicate_statement_first_wins():
    res = parse(EXAMPLE + "kick: x............... ................\n")
    assert res.ast == BASE
    assert res.repairs == [paid("duplicate", 19, "kick: x............... ................", "")]
    assert parse(add("kick: x............... ................")).ast.lane("kick")[0] == "x" + "." * 15
    res = parse(EXAMPLE + "a: w1 w1 w1 w1\n@3 hat: ................\n")
    assert res.ast == BASE
    assert [(r.code, r.line) for r in res.repairs] == [("duplicate", 19), ("duplicate", 20)]
    res = parse("vamp: IIm7 V7 span 1\n" + EXAMPLE)
    assert res.ast.vamp.chords == (Chord("II", "m7"), Chord("V", "7"))
    assert res.repairs == [paid("duplicate", 2, "vamp: Im9 IV13 span 2", "")]


def test_both_snare_and_rim_first_wins():
    res = parse(EXAMPLE + "snare: ....x.......x...\n")
    assert res.ast == BASE
    assert res.repairs == [paid("snare_and_rim", 19, "snare: ....x.......x...", "")]
    one = (FIX / "one_bar_base.ls").read_text()
    res = parse(one + "rim: ....X.......X...\n@2 rim: ....X...........\n")
    assert res.ast == parse(one).ast
    assert [(r.code, r.line) for r in res.repairs] == [("snare_and_rim", 11), ("snare_and_rim", 12)]
    assert res.penalty == 0.04


def test_edit_on_the_lane_without_a_base_is_an_ordinary_edit():
    one = (FIX / "one_bar_base.ls").read_text()                  # a snare loop
    res = parse(one + "@2 rim: ....X...........\n")
    assert res.gate.ok and res.repairs == []
    assert res.ast.edits == (Edit(2, "rim", "....X..........."),) and res.ast.backbeat == "snare"
    res = parse(EXAMPLE + "snare: ....x.......x...\n@8 snare: ........XXXXXXXX\n")
    assert res.ast == BASE                                       # both bases: the edit goes too
    assert [(r.code, r.line) for r in res.repairs] == [("snare_and_rim", 19), ("snare_and_rim", 20)]


def test_misplaced_bar_marker_is_logged_only():
    body = "q5 e6 | e5 q3 e2 e1 h3 | er e4 e5 e6 h.5 er e4 q3 q2 hr"
    res = parse(swap("A:", "A: " + body))
    assert res.ast == BASE and res.penalty == 0.0
    assert res.repairs == [free("misplaced_bar", 17, body, A_BODY)]
    assert parse(swap("A:", "A: " + A_BODY.replace("|", ""))).repairs == []  # markers are optional
    assert parse(swap("A:", "A: | " + A_BODY + " |")).repairs == []           # at 0 and 64: fine


def test_sax_octave_fold_is_left_to_the_compiler():
    res = parse(swap("A:", "A: q5''' " + A_BODY[3:]))
    assert res.gate.ok and res.repairs == [] and res.ast.a[0].octave == 3


@pytest.mark.parametrize("pads, penalty", [(1, 0.02), (3, 0.06), (5, 0.1), (6, 0.1), (8, 0.1)])
def test_penalty_is_002_each_capped_at_01(pads, penalty):
    lines = EXAMPLE.splitlines()
    grids = [(i, j) for i in (5, 6, 7, 8) for j in (0, 1)][:pads]
    for i, j in grids:
        head, cells = lines[i].split(": ")
        parts = cells.split()
        parts[j] = parts[j][:15]
        lines[i] = f"{head}: {' '.join(parts)}"
    res = parse("\n".join(lines) + "\n")
    assert res.gate.ok and sum(r.penalized for r in res.repairs) == pads
    assert res.penalty == penalty


def test_byte_order_mark_is_ignored():
    res = parse("﻿" + EXAMPLE)
    assert res.ast == BASE and res.repairs == []
