"""Keys voicings (instrument.md section 3, Voicings): the style templates, rule 2's rejections, and the
voice-leading choice, checked against an independent brute force over every pitch set in C3-C5."""
from fractions import Fraction
from itertools import combinations

import pytest

from lofi_rl.dsl.ast import Chord
from lofi_rl.dsl.vocab import QUALITIES, VOICINGS
from lofi_rl.theory.chords import QUALITY
from lofi_rl.theory.voicing import (allowed, candidates, choose, drop2, inversions, pair_cost, quartal, rootless,
                                    shell, voice_leading, voice_vamp)

# Templates written out from the section 3 rules, one row per quality (semitones above the root).
ROOTLESS = {"maj7": [4, 7, 11], "maj9": [4, 11, 14], "maj7#11": [4, 11, 14], "7": [4, 7, 10], "9": [4, 10, 14],
            "13": [4, 10, 14], "7sus": [5, 10, 14], "7b9": [4, 10, 13], "7#9": [4, 10, 15], "m7": [3, 7, 10],
            "m9": [3, 10, 14], "m11": [3, 10, 14], "m7b5": [3, 10, 17], "dim7": [3, 6, 9]}
SHELL = {"maj7": [4, 11], "maj9": [4, 11, 14], "maj7#11": [4, 11, 18], "7": [4, 10], "9": [4, 10, 14],
         "13": [4, 10, 21], "7sus": [5, 10, 14], "7b9": [4, 10, 13], "7#9": [4, 10, 15], "m7": [3, 10],
         "m9": [3, 10, 14], "m11": [3, 10, 17], "m7b5": [3, 10, 17], "dim7": [3, 9]}
QUARTAL = {"maj7": [4, 9, 14, 19], "maj9": [4, 9, 14, 19], "maj7#11": [4, 9, 14, 18], "7": [10, 16, 21],
           "9": [10, 16, 21, 26], "13": [10, 16, 21, 26], "7sus": [5, 10, 14, 19], "7b9": [10, 16, 21, 25],
           "7#9": [10, 16, 21, 27], "m7": [5, 10, 15, 19], "m9": [5, 10, 15, 19], "m11": [5, 10, 15, 19],
           "m7b5": [3, 10, 17], "dim7": [3, 6, 9]}
# The 12 shootout vamps of section 6, the section 1 example, and vamps that reach the other qualities.
VAMPS = [("II", "m9", "V", "13"), ("IV", "maj7", "III", "m7"), ("IV", "maj7", "III", "7"),
         ("I", "maj9", "IV", "maj9"), ("I", "m9", "IV", "13"), ("I", "m9", "IV", "9"), ("I", "m11", "II", "m7"),
         ("I", "m9", "bVII", "maj7"), ("I", "m9", "bVI", "maj7"), ("I", "m9", "IV", "m9"),
         ("IV", "m9", "bVII", "13"), ("bVI", "maj7", "bVII", "13"), ("II", "m7b5", "V", "7b9"),
         ("I", "dim7", "bIII", "maj7#11"), ("I", "7sus", "V", "7#9")]


def test_templates_follow_the_section_3_rules():
    for q in QUALITIES:
        assert list(rootless(QUALITY[q])) == ROOTLESS[q], q
        assert list(shell(QUALITY[q])) == SHELL[q], q
        assert list(quartal(QUALITY[q])) == QUARTAL[q], q


def test_inversions_and_drop2():
    assert inversions((3, 10, 14)) == ((3, 10, 14), (10, 14, 15), (14, 15, 22))
    assert inversions((4, 10, 21)) == ((4, 10, 21), (10, 16, 21), (16, 21, 22))
    assert drop2((3, 10, 14)) == (-2, 3, 14) and drop2((10, 14, 15, 19)) == (3, 10, 14, 19)


def test_rule_2_rejects_low_clusters_and_minor_ninths():
    m9, b9 = QUALITY["m9"], QUALITY["7b9"]
    assert not allowed((50, 52, 60), m9) and not allowed((54, 55, 62), m9)      # <= 2 semitones below 55
    assert allowed((55, 57, 64), m9) and allowed((60, 61, 70), m9)              # lower note at 55 or above
    assert allowed((53, 56, 62), m9)                                            # 3 semitones is not a cluster
    assert not allowed((48, 58, 61), m9) and allowed((48, 58, 61), b9)          # 13 apart: only 7b9


@pytest.mark.parametrize("style", VOICINGS)
def test_every_candidate_fits_c3_c5_and_passes_rule_2(style):
    for q in QUALITIES:
        for root in range(12):
            cands = candidates(root, q, style)
            assert cands, (q, style, root)
            for v in cands:
                assert list(v) == sorted(set(v)) and 48 <= v[0] and v[-1] <= 72
                assert {(p - root) % 12 for p in v} == {t % 12 for t in _template(q, style)}
                assert all(not (b - a <= 2 and a < 55) for a, b in zip(v, v[1:])), v
                assert q == "7b9" or all(b - a != 13 for i, a in enumerate(v) for b in v[i + 1:]), v


def _template(q, style):
    return {"rootless": ROOTLESS, "shell": SHELL, "quartal": QUARTAL, "spread": ROOTLESS}[style][q]


def _shapes(q, style):
    t = sorted(_template(q, style))
    invs = [sorted(t[j:] + [x + 12 for x in t[:j]]) for j in range(len(t))]
    if style == "spread":
        invs = [sorted(v[:-2] + [v[-2] - 12] + v[-1:]) for v in invs]
    return invs


def _brute_candidates(root, q, style):
    """Every pitch set in 48..72 that is an inversion of the style's template at some octave, minus rule 2."""
    shapes = _shapes(q, style)
    pcs = {(root + s) % 12 for s in shapes[0]}
    pitches = [p for p in range(48, 73) if p % 12 in pcs]
    out = []
    for combo in combinations(pitches, len(shapes[0])):
        if not any((combo[0] - root - s[0]) % 12 == 0 and all(c - combo[0] == x - s[0] for c, x in zip(combo, s))
                   for s in shapes):
            continue
        if any(b - a <= 2 and a < 55 for a, b in zip(combo, combo[1:])):
            continue
        if q != "7b9" and any(b - a == 13 for a, b in combinations(combo, 2)):
            continue
        out.append(combo)
    return out


def _brute_cost(a, b):
    def vl(x, y):
        if len(x) == 1:
            return abs(x[0] - y[0])
        return sum(abs(v - y[int(Fraction(i * (len(y) - 1), len(x) - 1) + Fraction(1, 2))]) for i, v in enumerate(x))
    mean = lambda v: Fraction(sum(v), len(v))                                  # noqa: E731
    return vl(a, b) + vl(b, a) + Fraction(1, 4) * (abs(mean(a) - 60) + abs(mean(b) - 60))


@pytest.mark.parametrize("style", VOICINGS)
def test_choice_matches_the_brute_force_optimum(style):
    for na, qa, nb, qb in VAMPS:
        for tonic in range(12):
            ra, rb = [(tonic + {"I": 0, "II": 2, "bIII": 3, "III": 4, "IV": 5, "V": 7, "bVI": 8, "bVII": 10}[n]) % 12
                      for n in (na, nb)]
            ca, cb = _brute_candidates(ra, qa, style), _brute_candidates(rb, qb, style)
            assert set(ca) == set(candidates(ra, qa, style)) and set(cb) == set(candidates(rb, qb, style))
            best = min(((a, b) for a in ca for b in cb), key=lambda ab: (_brute_cost(*ab), ab))
            got = voice_vamp(Chord(na, qa), Chord(nb, qb), tonic, style)
            assert got == best, (na + qa, nb + qb, tonic, style)
            assert pair_cost(*got) == _brute_cost(*best)


def test_the_example_vamp_voicing_is_pinned():
    # Im9 IV13 in D, rootless (section 1 example): F3 C4 E4 and A3 B3 F4.
    assert voice_vamp(Chord("I", "m9"), Chord("IV", "13"), 2, "rootless") == ((53, 60, 64), (57, 59, 65))


def test_ties_break_lexicographically():
    a = [(60, 64, 67), (48, 52, 55)]
    b = [(60, 64, 67)]
    assert choose(a, b) == ((60, 64, 67), (60, 64, 67))
    sym = [(55, 59, 62), (58, 62, 65)]                  # mirror images around 60: equal cost against each other
    assert pair_cost(sym[0], sym[1]) == pair_cost(sym[1], sym[0])
    assert choose(sym, sym) == min(((x, y) for x in sym for y in sym), key=lambda p: (pair_cost(*p), p))


def test_voice_leading_pairs_outer_voices_for_unequal_sizes():
    assert voice_leading((60, 64, 67), (59, 62, 67)) == 1 + 2 + 0
    assert voice_leading((60, 64), (59, 62, 67)) == 1 + 3                     # bottom-bottom, top-top
    assert voice_leading((59, 62, 67), (60, 64)) == 1 + 2 + 3                 # the middle voice goes up
    assert pair_cost((60, 64), (59, 62, 67)) == 4 + 6 + Fraction(1, 4) * (2 + Fraction(8, 3))
