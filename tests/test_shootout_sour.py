"""Loop-text substitutions and the anchors' sour copies (docs/SHOOTOUTS.md sections 2.5, 5 and 7.6).

A sour copy raises every sax note's accidental by one (a natural becomes #, a b becomes natural) and changes no
other byte. The builder keeps it only if it passes G1-G6 in its loop's mood and costs P_chrom >= 0.2, with
P_chrom = min(.3, .1 x #bad) as instrument.md section 5 defines it: a note is free if its pitch class is in the
mode, is a blue note (3, 6 or 10 above the tonic) or is a tone of the chord sounding at its onset; a non-free note
is bad unless it lasts 2 steps or fewer and the next note is a free note 1-2 semitones away."""
import pytest

from lofi_rl.brief import Brief
from lofi_rl.dsl import parse
from lofi_rl.shootout.plan import load_plan
from lofi_rl.shootout.sour import (SOUR_MIN_P_CHROM, check_sour, chromatic, raise_accidentals, sax_voice,
                                   spread_text, swap_line)

PLAN = load_plan()
TENOR, ALTO = (sax_voice(r, PLAN.registers[r]) for r in ("tenor", "alto"))
BASE = """vamp: Im9 IV13 span 2
keys: ep rootless
comp: x-----.x----..x- x-------..x-x---
bass: x-------..x-x--- x-----.x-x------
drums: dusty
kick: x......x..x..... x......x.x....x.
rim: ....X.......X... ....X..g....X..g
hat: x.x.x.x.x.x.x.x. x.x.x.x.x.x.xxx.
"""
WISTFUL = Brief(energy="mid", mood="wistful", bpm=82, tonic=2)        # D dorian, the example's brief
HOPEFUL = Brief(energy="mid", mood="hopeful", bpm=82, tonic=0)        # C ionian


def _loop(a: str, b: str, vamp: str = "Im9 IV13 span 2", brief: Brief = WISTFUL):
    text = BASE.replace("vamp: Im9 IV13 span 2", f"vamp: {vamp}") + f"A: {a}\nB: {b}\n"
    res = parse(text, mode=brief.mode)
    assert res.gate.ok, res.gate
    return res.ast


def test_the_sour_copy_raises_every_accidental_by_one_and_changes_nothing_else():
    text = BASE + "A: q5 eb5 e3' hr | h.1 er e4, | w2 | w6\nB: h1' qr q7 | h.b3 qr | w5 | h4 hr\n"
    out = raise_accidentals(text)
    assert out.splitlines()[:8] == text.splitlines()[:8]
    assert out.splitlines()[8:] == ["A: q#5 e5 e#3' hr | h.#1 er e#4, | w#2 | w#6",
                                    "B: h#1' qr q#7 | h.3 qr | w#5 | h#4 hr"]
    with pytest.raises(ValueError, match="sharp"):
        raise_accidentals(out)                          # a # cannot be raised again in LoopScript
    with pytest.raises(ValueError, match="phrase"):
        raise_accidentals(BASE)                         # no A: or B: line


@pytest.mark.parametrize("lid", [f"L{i}" for i in range(1, 10)])
def test_every_reference_loop_has_a_gate_passing_sour_copy(lid):
    # section 2.5: "The 9 loops' sour copies pass G1-G6 in every mood with 4-12 non-free notes each, P_chrom at
    # its 0.3 cap". The 4-12 are the penalized ones under the dev tenor in the loop's own mood (L8 4 ... L3 12);
    # the cap holds under the alto lead too, whose octave fit splits L8's phrases
    lp = PLAN.loops[lid]
    sour = raise_accidentals(lp.text)
    tenor = check_sour(sour, lp.brief, TENOR)
    assert tenor.ok and tenor.gate == "pass" and tenor.p_chrom == 0.3 and 4 <= tenor.bad <= 12, (lid, tenor)
    alto = check_sour(sour, lp.brief, ALTO)
    assert alto.ok and alto.p_chrom == 0.3 and alto.nonfree == tenor.nonfree, (lid, alto)
    for mood in ("melancholy", "wistful", "hopeful"):
        assert parse(sour, mode=mood).gate.ok
    own = parse(lp.text, mode=lp.brief.mode).ast
    assert chromatic(own, lp.brief, TENOR).p_chrom == 0.0       # section 5: P_chrom 0 under the loop's own vamp


def test_free_notes_are_read_by_pitch_class_mode_blue_or_chord_tone():
    # D dorian over Im9 (bars 1-2, 5-6) and IV13 (bars 3-4, 7-8): degree 1 is free; b5 is the blue b5; b6 is
    # not in dorian, not blue, and not a tone of Im9, so it is non-free in bar 1
    assert chromatic(_loop("w1 w1 w1 w1", "w1 w1 w1 w1"), WISTFUL, TENOR).nonfree == 0
    held_blue = chromatic(_loop("wb5 w1 w1 w1", "w1 w1 w1 w1"), WISTFUL, TENOR)
    assert (held_blue.nonfree, held_blue.p_chrom) == (0, 0.0)             # a held blue note is never penalized
    held = chromatic(_loop("hb6 h5 w1 w1 w1", "w1 w1 w1 w1"), WISTFUL, TENOR)
    assert (held.nonfree, held.bad, held.p_chrom) == (1, 1, 0.1)
    passing = chromatic(_loop("eb6 e5 h.5 w1 w1 w1", "w1 w1 w1 w1"), WISTFUL, TENOR)
    assert (passing.nonfree, passing.bad) == (1, 0)                       # 2 steps, a step down to a free 5
    leap = chromatic(_loop("eb6 e1 h.1 w1 w1 w1", "w1 w1 w1 w1"), WISTFUL, TENOR)
    assert leap.bad == 1                                                  # resolves by leap, not by step
    many = chromatic(_loop("qb6 qb6 qb6 qb6 w1 w1 w1", "w1 w1 w1 w1"), WISTFUL, TENOR)
    assert (many.bad, many.p_chrom) == (4, 0.3)                           # capped at .3
    # a chord tone outside the mode is free while its chord sounds: III7's third is 8 above C in C ionian
    tone = chromatic(_loop("w1 w1 wb6 w1", "w1 w1 w1 w1", vamp="IVmaj7 III7 span 2", brief=HOPEFUL), HOPEFUL, TENOR)
    off = chromatic(_loop("wb6 w1 w1 w1", "w1 w1 w1 w1", vamp="IVmaj7 III7 span 2", brief=HOPEFUL), HOPEFUL, TENOR)
    assert (tone.nonfree, off.nonfree) == (0, 1)


def test_the_last_note_resolves_around_the_cycle():
    # B's last note is followed by A's first: a 2-step b6 at the end of B resolving down to A's opening 5
    ast = _loop("w5 w1 w1 w1", "w1 w1 w1 h.1 eb6")
    assert chromatic(ast, WISTFUL, TENOR).bad == 0


def test_a_sour_copy_that_costs_too_little_fails_the_check():
    # in C ionian #3 is the 4th degree, a mode tone, so this copy has no non-free note at all
    text = BASE.replace("Im9 IV13", "Imaj9 IVmaj9") + "A: w3 w3 w3 w3\nB: w3 w3 w3 w3\n"
    chk = check_sour(raise_accidentals(text), HOPEFUL, TENOR)
    assert chk.gate == "pass" and chk.p_chrom == 0.0 < SOUR_MIN_P_CHROM and not chk.ok
    broken = check_sour(raise_accidentals(text).replace("kick:", "kik:"), HOPEFUL, TENOR)
    assert broken.gate == "G1" and not broken.ok


def test_swap_line_replaces_exactly_one_statement_line():
    lp = PLAN.loops["L1"]
    out = swap_line(lp.text, "vamp", "Im9 IVm9 span 2")
    a, b = lp.text.splitlines(), out.splitlines()
    assert [i for i, (x, y) in enumerate(zip(a, b)) if x != y] == [0] and b[0] == "vamp: Im9 IVm9 span 2"
    assert out.endswith("\n") == lp.text.endswith("\n") and len(out.splitlines()) == len(a)
    sp = spread_text(lp.text)
    assert [y for x, y in zip(a, sp.splitlines()) if x != y] == ["keys: ep spread"]
    with pytest.raises(ValueError, match="once"):
        swap_line(lp.text + "vamp: Im9 IV13 span 2\n", "vamp", "Im9 IVm9 span 2")
    with pytest.raises(ValueError, match="once"):
        swap_line(lp.text, "shaker", "x...............")
