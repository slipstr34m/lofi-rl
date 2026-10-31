"""Canonical text: parse(canon(x)) == x, canon is idempotent, canon_sha is sha256 of the text."""
import hashlib
from pathlib import Path

import pytest

from lofi_rl.dsl import (Chord, Edit, Item, Keys, Lane, Loop, Vamp, canon, canon_sha, item_text,
                         parse, phrase_text)

FIX = Path(__file__).parent / "fixtures" / "loops"
EXAMPLE = (FIX / "example.ls").read_text()
CANONICAL = ["example.ls", "one_bar_base.ls", "shaker_edits.ls", "span1_dotted.ls"]


def swap(prefix: str, new: str, text: str = EXAMPLE) -> str:
    return "\n".join(new if x.startswith(prefix) else x for x in text.splitlines()) + "\n"


@pytest.mark.parametrize("name", CANONICAL)
def test_fixture_text_is_already_canonical(name):
    text = (FIX / name).read_text()
    assert canon(parse(text).ast) == text


@pytest.mark.parametrize("name", CANONICAL + ["example_leniency.ls", "leniency_lower_x.ls"])
def test_round_trip_and_idempotence(name):
    loop = parse((FIX / name).read_text()).ast
    text = canon(loop)
    again = parse(text)
    assert again.gate.ok and again.repairs == []
    assert again.ast == loop
    assert canon(again.ast) == text


@pytest.mark.parametrize("text", [
    swap("vamp:", "vamp: ii9 V13"),                                   # alias + default span
    swap("kick:", "bd: x------x--x--- x......x.x....x...."),          # alias, dash, pad, trunc
    swap("A:", "A: q5 e6 | e5 q3 e2 e1 h3 er e4 e5 e6 h.5 er e4 q3 q2"),  # misplaced |, padded
    EXAMPLE + "snare: ....x.......x...\nkick: ................\n",    # snare+rim, duplicate
])
def test_repaired_loops_print_text_that_needs_no_repair(text):
    res = parse(text)
    assert res.gate.ok and res.repairs
    clean = parse(canon(res.ast))
    assert clean.repairs == [] and clean.ast == res.ast
    assert canon(clean.ast) == canon(res.ast)


def test_leniency_fixture_canonicalizes_to_the_example():
    loop = parse((FIX / "example_leniency.ls").read_text()).ast
    assert canon(loop) == EXAMPLE
    assert canon_sha(loop) == canon_sha(parse(EXAMPLE).ast)


def test_statement_order_and_comments_do_not_change_canon():
    lines = EXAMPLE.splitlines()
    shuffled = "\n".join(f"{x}   # c" for x in lines[::-1]) + "\n"
    assert canon(parse(shuffled).ast) == EXAMPLE


def test_canon_sha_is_sha256_of_the_canonical_text():
    loop = parse(EXAMPLE).ast
    sha = canon_sha(loop)
    assert sha == hashlib.sha256(EXAMPLE.encode("utf-8")).hexdigest()
    assert len(sha) == 64 and int(sha, 16) >= 0
    one_cell = parse(EXAMPLE.replace("@7 ohat: ......x...", "@7 ohat: ......X...")).ast
    assert canon_sha(one_cell) != sha


def test_canon_sorts_a_hand_built_loop_and_prints_lock():
    loop = Loop(
        vamp=Vamp((Chord("bVI", "maj7"), Chord("bVII", "13")), 1),
        keys=Keys("organ", "shell"),
        comp=("x-..............",),
        bass=(),
        kit="jazz",
        lanes=(Lane("shaker", ("g" * 16,)), Lane("snare", ("....x.......x...",)),
               Lane("kick", ("x.......x.......",))),
        a=(Item(16, False, 1), Item(16, False, 3), Item(16, False, 5), Item(16, rest=True)),
        b=(Item(24, False, 6, -1), Item(4, False, 5, 0, -1), Item(4, rest=True), Item(16, False, 1),
           Item(16, rest=True)),
        edits=(Edit(8, "kick", "x..............."), Edit(2, "shaker", "." * 16), Edit(2, "comp", "x" * 16)),
    )
    text = canon(loop)
    assert text.splitlines() == [
        "vamp: bVImaj7 bVII13 span 1",
        "keys: organ shell",
        "comp: x-..............",
        "bass: lock",
        "drums: jazz",
        "kick: x.......x.......",
        "snare: ....x.......x...",
        "shaker: gggggggggggggggg",
        "@2 comp: xxxxxxxxxxxxxxxx",
        "@2 shaker: ................",
        "@8 kick: x...............",
        "A: w1 | w3 | w5 | wr",
        "B: w.b6 q5, qr | w1 | wr",
    ]
    reparsed = parse(text).ast
    assert reparsed.lanes[0].name == "kick" and reparsed.edits[0] == Edit(2, "comp", "x" * 16)
    assert canon(reparsed) == text


def test_item_and_phrase_text():
    assert item_text(Item(6, False, 3, 1, 2)) == "q.#3''"
    assert item_text(Item(1, False, 7, -1, -1)) == "sb7,"
    assert item_text(Item(12, rest=True)) == "h.r"
    items = [Item(24, False, 3), Item(3, False, 4), Item(1, False, 5), Item(4, False, 6)]
    assert phrase_text(items) == "w.3 e.4 s5 q6"                    # no bar line inside w.3
    assert phrase_text(items + [Item(16, rest=True)]) == "w.3 e.4 s5 q6 | wr"
