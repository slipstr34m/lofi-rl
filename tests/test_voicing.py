"""Keys voicings: instrument.md section 3, Voicings, as amended by voicing rules V2 (docs/COMPILER.md,
"Amendment V2 to instrument.md §3 Voicings"). voicing.py is checked against tests/helpers_voicing.py, an oracle
that never reads voicing.py's templates:
- R1-R6 over every mode x numeral (borrowed ones included) x quality x style at every tonic, with every candidate
  set equal to the oracle's; R7 in rootless, shell and spread, and the quartal coincidences counted;
- the quartal oracle itself against every pitch set in C3-C5, for a sample of chords;
- the voice-leading choice against a brute force over the oracle's candidates, for a sample of vamps;
- the section 3 rules V2 leaves alone: inversions, drop-2, rule 2, the cost and its ties."""
import ast
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import pytest

import helpers_voicing
from helpers_voicing import (added, avoids, colour, cost48, dropped, guide, inverted, is_stack, mode_pcs, named,
                             normal, note_names, oracle_candidates, oracle_vamp, placed, quartal_stacks,
                             register_ok, root_of, rootless_template, sharp9_above_third, shell_template)
from lofi_rl.dsl.ast import Chord
from lofi_rl.dsl.vocab import MODES, NUMERALS, QUALITIES, VOICINGS
from lofi_rl.theory.chords import QUALITY
from lofi_rl.theory.voicing import (allowed, candidates, choose, chord_scale, drop2, inversions, pair_cost, quartal,
                                    rootless, shell, voice_leading, voice_vamp)

# R4's examples and its shell rule, written out (semitones above the root).
ROOTLESS = {"maj7": [4, 7, 11], "maj9": [4, 11, 14], "maj7#11": [4, 11, 14, 18], "7": [4, 7, 10], "9": [4, 10, 14],
            "13": [4, 10, 14, 21], "7sus": [5, 10, 14], "7b9": [4, 10, 13], "7#9": [4, 10, 15], "m7": [3, 7, 10],
            "m9": [3, 10, 14], "m11": [3, 10, 14, 17], "m7b5": [3, 6, 10, 17], "dim7": [3, 6, 9]}
SHELL = {"maj7": [4, 11], "maj9": [4, 11, 14], "maj7#11": [4, 11, 18], "7": [4, 10], "9": [4, 10, 14],
         "13": [4, 10, 21], "7sus": [5, 10, 14], "7b9": [4, 10, 13], "7#9": [4, 10, 15], "m7": [3, 10],
         "m9": [3, 10, 14], "m11": [3, 10, 17], "m7b5": [3, 6, 10], "dim7": [3, 6, 9]}

# The 12 shootout vamps of instrument.md section 6, each in its mood's mode (the /1 and /2 there are spans, not
# tonics), and vamps that reach the other qualities, tried in every mode.
SHOOTOUT = {"Aruarian": ("ionian", ("II", "m9"), ("V", "13")), "Royal-half": ("ionian", ("IV", "maj7"), ("III", "m7")),
            "Two-of-us": ("ionian", ("IV", "maj7"), ("III", "7")), "Lift": ("ionian", ("I", "maj9"), ("IV", "maj9")),
            "i-IV": ("dorian", ("I", "m9"), ("IV", "13")), "Pulse": ("dorian", ("I", "m9"), ("IV", "9")),
            "Step": ("dorian", ("I", "m11"), ("II", "m7")), "Float": ("dorian", ("I", "m9"), ("bVII", "maj7")),
            "Drift": ("aeolian", ("I", "m9"), ("bVI", "maj7")), "Plagal": ("aeolian", ("I", "m9"), ("IV", "m9")),
            "Backdoor": ("aeolian", ("IV", "m9"), ("bVII", "13")),
            "Rise": ("aeolian", ("bVI", "maj7"), ("bVII", "13"))}
OTHERS = [(("II", "m7b5"), ("V", "7b9")), (("I", "dim7"), ("bIII", "maj7#11")), (("I", "7sus"), ("V", "7#9"))]
VAMPS = list(SHOOTOUT.values()) + [(mode, a, b) for a, b in OTHERS for mode in MODES]


@lru_cache(maxsize=None)
def _candidates(root, q, style, numeral, mode):
    return candidates(root, q, style, chord_scale(numeral, mode))


def test_the_oracle_never_imports_voicing_py():
    tree = ast.parse(Path(helpers_voicing.__file__).read_text())
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert modules == {"__future__", "fractions", "functools", "itertools", "lofi_rl.dsl.vocab",
                       "lofi_rl.theory.chords"}


def test_rootless_and_shell_templates_follow_r4():
    for q in QUALITIES:
        assert list(rootless(QUALITY[q])) == ROOTLESS[q] == rootless_template(q), q
        assert list(shell(QUALITY[q])) == SHELL[q] == shell_template(q), q


def test_chord_scale_is_the_brief_mode_above_the_chord_root():
    for mode in MODES:
        for numeral in NUMERALS:
            assert chord_scale(numeral, mode) == mode_pcs(numeral, mode)
    assert chord_scale("I", "dorian") == {0, 2, 3, 5, 7, 9, 10}
    assert chord_scale("V", "aeolian") == {0, 1, 3, 5, 7, 8, 10}            # E phrygian over E7 in A aeolian
    assert chord_scale("bVI", "ionian") == {1, 3, 4, 6, 8, 9, 11}           # C major over a borrowed Ab root
    assert chord_scale("IV", "melancholy") == chord_scale("IV", "aeolian")


def test_quartal_stacks_match_the_oracle_in_every_mode():
    for mode in MODES:
        for numeral in NUMERALS:
            for q in QUALITIES:
                got = quartal(QUALITY[q], chord_scale(numeral, mode))
                assert list(got) == sorted(set(got)) and set(got) == quartal_stacks(q, numeral, mode), \
                    (mode, numeral, q)


@pytest.mark.parametrize("mode", list(MODES))
def test_the_quartal_oracle_matches_every_pitch_set_in_c3_c5(mode):
    # R4 read literally, on tonic 2: every 3- or 4-note set in 48..72 of distinct pitch classes from R3's tone set
    # that holds R1 and R2, keeps the avoid-note rule, is a stack, and passes R5 and R6. A sample: every quality
    # on I, and the chords of the sample vamps.
    chords = {("I", q) for q in QUALITIES} | {c for m, a, b in VAMPS if m == mode for c in (a, b)}
    for numeral, q in sorted(chords):
        root = root_of(numeral, 2)
        tones, need = named(q) | added(q, numeral, mode), guide(q) | colour(q, "quartal")
        pitches = [p for p in range(48, 73) if (p - root) % 12 in tones]
        want = set()
        for size in (3, 4):
            for v in combinations(pitches, size):
                if not is_stack(v):
                    continue
                held = {(p - root) % 12 for p in v}
                if (len(held) == size and need <= held and avoids(held, q) and register_ok(v, q)
                        and sharp9_above_third(v, root)):
                    want.add(v)
        assert placed(root, quartal_stacks(q, numeral, mode), q) == want, (mode, numeral, q)


def _built(root, q, style, numeral, mode):
    """The construction R4 gives this chord: its style, or rootless for a quartal chord with no stack that fits."""
    return "rootless" if style == "quartal" and not placed(root, quartal_stacks(q, numeral, mode), q) else style


def _rule_breaks(v, root, q, built, numeral, mode):
    """The V2 rules a candidate of construction `built` breaks, by name; empty when it keeps all of them."""
    held = {(p - root) % 12 for p in v}
    breaks = []
    if not guide(q) <= held:
        breaks.append("R1 guide tones")
    if not colour(q, built) <= held:
        breaks.append("R2 colour")
    extra = held - named(q)
    if (extra - added(q, numeral, mode) if built == "quartal" else extra) or not avoids(held, q):
        breaks.append("R3 foreign tone")
    if not held <= named(q) | mode_pcs(numeral, mode):
        breaks.append("R3 outside chord + mode")
    shape = normal([p - root for p in v])
    if built == "quartal":
        construction = len(held) == len(v) and len(v) in (3, 4) and is_stack(v)
    else:
        template = shell_template(q) if built == "shell" else rootless_template(q)
        shapes = [dropped(s) if built == "spread" else s for s in inverted(template)]
        construction = 0 not in held and shape in {normal(s) for s in shapes}
    if not construction:
        breaks.append("R4 construction")
    if not sharp9_above_third(v, root):
        breaks.append("R5 #9 under the 3rd")
    if not register_ok(v, q):
        breaks.append("R6 register or clusters")
    return breaks


@pytest.mark.parametrize("style", VOICINGS)
def test_every_chord_in_every_mode_keeps_r1_to_r6(style):
    seen = set()
    for mode in MODES:
        for numeral in NUMERALS:
            for q in QUALITIES:
                for tonic in range(12):
                    root = root_of(numeral, tonic)
                    key = (root, q, chord_scale(numeral, mode) if style == "quartal" else None)
                    if key in seen:                 # the same chord: R3 reads the mode only above the root
                        continue
                    seen.add(key)
                    got = _candidates(root, q, style, numeral, mode)
                    assert got and set(got) == oracle_candidates(root, q, style, numeral, mode), \
                        (style, mode, numeral, q, tonic)
                    assert list(got) == sorted(got) and all(list(v) == sorted(set(v)) for v in got)
                    built = _built(root, q, style, numeral, mode)
                    for v in got:
                        assert not _rule_breaks(v, root, q, built, numeral, mode), \
                            (_rule_breaks(v, root, q, built, numeral, mode), style, mode, numeral, q, note_names(v))
    assert len(seen) == (12 * 14 * 12 if style == "quartal" else 12 * 14)


@pytest.mark.parametrize("style", ["rootless", "shell", "spread"])
def test_r7_two_symbols_never_share_a_voicing_on_one_root(style):
    for root in range(12):
        owner = {}
        for q in QUALITIES:
            for v in candidates(root, q, style):
                assert owner.setdefault(v, q) == q, (style, root, note_names(v), owner[v], q)
    for a, b in (("13", "9"), ("m11", "m9"), ("maj7#11", "maj9"), ("m7b5", "m11")):    # R7's own list
        assert not set(candidates(2, a, style)) & set(candidates(2, b, style)), (style, a, b)


def test_r7_quartal_coincidences_are_counted():
    # Quartal stacks may coincide: an added tension can be another quality's named one (an added 9 on 7 is a 9
    # stack). Counted over every (mode, numeral, tonic, pair of qualities); docs/COMPILER.md quotes the count.
    cases = shared = 0
    pairs = {}
    for mode in MODES:
        for numeral in NUMERALS:
            for tonic in range(12):
                root = root_of(numeral, tonic)
                cands = {q: set(_candidates(root, q, "quartal", numeral, mode)) for q in QUALITIES}
                for a, b in combinations(QUALITIES, 2):
                    cases += 1
                    if cands[a] & cands[b]:
                        shared += 1
                        pairs[a, b] = pairs.get((a, b), 0) + 1
    assert (cases, shared) == (36036, 2430)
    assert pairs[("7", "9")] == pairs[("m7", "m11")] == pairs[("maj7", "maj9")] == 240 and len(pairs) == 13
    same = sum(bool(quartal_stacks(a, n, m)) and quartal_stacks(a, n, m) == quartal_stacks(b, n, m)
               for m in MODES for n in NUMERALS for a, b in combinations(QUALITIES, 2))
    assert same == 0, "no two qualities share every stack in any mode"


def test_quartal_falls_back_to_rootless_when_no_stack_fits():
    # m7b5 has one stack, 11-b7-b3-b5 [5, 10, 15, 18], and its 11 and b5 are 13 semitones apart (R6), so m7b5 is
    # voiced rootless in every mode. Each m9 stack spans 16 semitones, so on roots B, C and C# none fits C3-C5.
    falls = 0
    for mode in MODES:
        for numeral in NUMERALS:
            assert quartal(QUALITY["m7b5"], chord_scale(numeral, mode)) == ((5, 10, 15, 18),)
            for q in QUALITIES:
                for tonic in range(12):
                    root = root_of(numeral, tonic)
                    if not placed(root, quartal_stacks(q, numeral, mode), q):
                        falls += 1
                        assert _candidates(root, q, "quartal", numeral, mode) == candidates(root, q, "rootless"), \
                            (mode, numeral, q, tonic)
                        assert q == "m7b5" or (q == "m9" and root in (11, 0, 1)), (mode, numeral, q, tonic)
    assert falls == 3 * 11 * 12 + 3 * 11 * 3 == 495
    with pytest.raises(ValueError, match="mode"):
        candidates(2, "m9", "quartal")


def test_r5_the_sharp9_of_7sharp9_sounds_above_its_3rd():
    # The second review's T6 case: IIm7 V7#9 in C ionian. G7#9's Bb (the #9) now sits above its B in every style.
    for style in VOICINGS:
        _, g = voice_vamp(Chord("II", "m7"), Chord("V", "7#9"), 0, "ionian", style)
        assert max(p for p in g if p % 12 == 11) < min(p for p in g if p % 12 == 10), (style, note_names(g))


@pytest.mark.parametrize("style", VOICINGS)
def test_the_choice_is_the_brute_force_optimum_over_the_oracle_candidates(style):
    for mode, a, b in VAMPS:
        for tonic in range(12):
            want = oracle_vamp(a, b, tonic, mode, style)
            got = voice_vamp(Chord(*a), Chord(*b), tonic, mode, style)
            assert got == want, (style, mode, a, b, tonic, note_names(got[0]), note_names(got[1]))
            assert 48 * pair_cost(*got) == cost48(*want)


def test_the_example_vamp_voicing_is_pinned():
    # Im9 IV13 in D dorian, rootless (section 1 example): Dm9 = F3 C4 E4 (b3 b7 9), G13 = F3 A3 B3 E4 (b7 9 3 13).
    assert voice_vamp(Chord("I", "m9"), Chord("IV", "13"), 2, "dorian", "rootless") == ((53, 60, 64), (53, 57, 59, 64))


# The shootout vamps at tonic 2 (D), written out from the oracle: rootless, shell, spread, quartal.
SHOOTOUT_AT_D = {
    "Aruarian": ["G3 D4 F#4 | G3 B3 C#4 F#4", "G3 D4 F#4 | G3 C#4 F#4", "D3 G3 F#4 | C#3 G3 B3 F#4",
                 "D3 G3 C#4 F#4 | E3 G3 C#4 F#4"],
    "Royal-half": ["B3 D4 F#4 | A3 C#4 E4", "B3 F#4 | A3 E4", "F#3 D4 B4 | E3 C#4 A4", "F#3 B3 E4 A4 | F#3 B3 E4 A4"],
    "Two-of-us": ["B3 D4 F#4 | Bb3 C#4 E4", "B3 F#4 | Bb3 E4", "F#3 D4 B4 | E3 C#4 Bb4", "F#3 B3 E4 | F#3 Bb3 E4"],
    "Lift": ["F#3 C#4 E4 | A3 B3 F#4", "F#3 C#4 E4 | A3 B3 F#4", "E3 C#4 F#4 | F#3 B3 A4",
             "E3 A3 C#4 F#4 | F#3 B3 E4 A4"],
    "i-IV": ["F3 C4 E4 | F3 A3 B3 E4", "F3 C4 E4 | F3 B3 E4", "F3 E4 C5 | F3 B3 E4 A4", "C3 F3 B3 E4 | D3 F3 B3 E4"],
    "Pulse": ["F3 C4 E4 | A3 B3 F4", "F3 C4 E4 | A3 B3 F4", "F3 E4 C5 | A3 F4 B4", "C3 F3 B3 E4 | F3 B3 E4 A4"],
    "Step": ["E4 F4 G4 C5 | D4 G4 B4", "C4 F4 G4 | D4 G4", "G3 E4 F4 C5 | G3 D4 B4", "D3 G3 C4 F4 | D3 G3 C4 F4"],
    "Float": ["F3 C4 E4 | G3 B3 E4", "C4 E4 F4 | B3 E4", "F3 E4 C5 | G3 E4 B4", "C3 F3 B3 E4 | D3 G3 B3 E4"],
    "Drift": ["F3 C4 E4 | F3 A3 D4", "F3 C4 E4 | A3 D4", "F3 E4 C5 | F3 D4 A4", "C3 F3 Bb3 E4 | C3 F3 A3 D4"],
    "Plagal": ["F3 C4 E4 | A3 Bb3 F4", "F3 C4 E4 | A3 Bb3 F4", "F3 E4 C5 | F3 Bb3 A4", "C3 F3 Bb3 E4 | F3 Bb3 E4 A4"],
    "Backdoor": ["Bb3 F4 A4 | Bb3 D4 E4 A4", "Bb3 F4 A4 | Bb3 E4 A4", "F3 Bb3 A4 | E3 Bb3 D4 A4",
                 "F3 Bb3 E4 A4 | G3 Bb3 E4 A4"],
    "Rise": ["F3 A3 D4 | E3 A3 Bb3 D4", "D3 A3 | E3 A3 Bb3", "F3 D4 A4 | E3 Bb3 D4 A4", "A3 D4 G4 | Bb3 E4 A4"],
}


def test_the_shootout_vamps_at_tonic_2_are_pinned():
    for name, (mode, a, b) in SHOOTOUT.items():
        for style, want in zip(VOICINGS, SHOOTOUT_AT_D[name]):
            va, vb = voice_vamp(Chord(*a), Chord(*b), 2, mode, style)
            assert f"{note_names(va)} | {note_names(vb)}" == want, (name, style)


# ------------------------------------------------------------------- section 3 rules that V2 leaves unchanged
def test_inversions_and_drop2():
    assert inversions((3, 10, 14)) == ((3, 10, 14), (10, 14, 15), (14, 15, 22))
    assert inversions((4, 10, 21)) == ((4, 10, 21), (10, 16, 21), (16, 21, 22))
    assert inversions((4, 10, 14, 21)) == ((4, 10, 14, 21), (10, 14, 16, 21), (14, 16, 21, 22), (16, 21, 22, 26))
    assert drop2((3, 10, 14)) == (-2, 3, 14) and drop2((10, 14, 15, 19)) == (3, 10, 14, 19)


def test_rule_2_rejects_low_clusters_and_minor_ninths():
    m9, b9 = QUALITY["m9"], QUALITY["7b9"]
    assert not allowed((50, 52, 60), m9) and not allowed((54, 55, 62), m9)      # <= 2 semitones below 55
    assert allowed((55, 57, 64), m9) and allowed((60, 61, 70), m9)              # lower note at 55 or above
    assert allowed((53, 56, 62), m9)                                            # 3 semitones is not a cluster
    assert not allowed((48, 58, 61), m9) and allowed((48, 58, 61), b9)          # 13 apart: only 7b9


def test_ties_break_lexicographically():
    a = [(60, 64, 67), (48, 52, 55)]
    b = [(60, 64, 67)]
    assert choose(a, b) == ((60, 64, 67), (60, 64, 67))
    sym = [(55, 59, 62), (58, 62, 65)]                  # mirror images around 60: equal cost against each other
    assert pair_cost(sym[0], sym[1]) == pair_cost(sym[1], sym[0])
    assert choose(sym, sym) == min(((x, y) for x in sym for y in sym), key=lambda p: (pair_cost(*p), p))
    mixed = [(55, 60, 64), (50, 55, 60, 65), (59, 65)]  # 3, 4 and 2 notes: one exact scale for every pair
    assert choose(mixed, mixed) == min(((x, y) for x in mixed for y in mixed), key=lambda p: (pair_cost(*p), p))


def test_voice_leading_pairs_outer_voices_for_unequal_sizes():
    assert voice_leading((60, 64, 67), (59, 62, 67)) == 1 + 2 + 0
    assert voice_leading((60, 64), (59, 62, 67)) == 1 + 3                     # bottom-bottom, top-top
    assert voice_leading((59, 62, 67), (60, 64)) == 1 + 2 + 3                 # the middle voice goes up
    assert pair_cost((60, 64), (59, 62, 67)) == 4 + 6 + Fraction(1, 4) * (2 + Fraction(8, 3))
    assert voice_leading((53, 60, 64), (53, 57, 59, 64)) == 0 + 1 + 0         # 3 into 4: to voices 0, 2, 3 of B
    assert voice_leading((53, 57, 59, 64), (53, 60, 64)) == 0 + 3 + 1 + 0     # 4 into 3: to voices 0, 1, 1, 2 of B
