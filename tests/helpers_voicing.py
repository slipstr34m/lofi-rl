"""The keys-voicing oracle for voicing rules V2 (docs/COMPILER.md, "Amendment V2 to instrument.md §3 Voicings"),
written out from the rules independently of src/lofi_rl/theory/voicing.py, which it never imports.

- Tone sets come from chords.py's interval table (QUALITY: third, fifth, seventh, extensions) and the mode
  tables (dsl.vocab.MODES). Tones are pitch classes above the chord root.
- The rootless, shell and spread templates are rebuilt from R4's wording.
- The quartal stacks come from a brute force over every bottom-to-top ordering of every 3- and 4-tone subset of
  R3's tone set (tests/test_voicing.py checks that against every pitch set in C3-C5 for a sample).
- The voice-leading cost is an exact integer, 48 times the section 3 cost.
Not a test module.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations, permutations

from lofi_rl.dsl.vocab import MODES, NUMERAL_SEMITONES
from lofi_rl.theory.chords import QUALITY

LO, HI = 48, 72                         # R6: C3-C5, sounding MIDI
# V2.1 R3: the tensions a quartal stack may add, per quality (b9 1, 9 2, #9 3, 11 5, #11 6, b13 8, 13 9), before the
# mode: the maj7 family 9 #11 13; the dominants 9 13, plus a b9 or #9 only where the symbol names it; the m7 family
# 9 11 13; m7b5 11 b13; dim7 none; 7sus 9 13.
AVAILABLE = {"maj7": {2, 6, 9}, "maj9": {2, 6, 9}, "maj7#11": {2, 6, 9},
             "7": {2, 9}, "9": {2, 9}, "13": {2, 9}, "7b9": {1, 2, 9}, "7#9": {2, 3, 9},
             "m7": {2, 5, 9}, "m9": {2, 5, 9}, "m11": {2, 5, 9},
             "m7b5": {5, 8}, "dim7": set(), "7sus": {2, 9}}
PERFECT_FIFTH = 7
NOTE = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")    # one fixed name per pitch class


def named(q: str) -> frozenset[int]:
    """The pitch classes the symbol names: root, 3rd, 5th, 7th and the table's extensions."""
    return frozenset(t % 12 for t in QUALITY[q].tones)


def guide(q: str) -> frozenset[int]:
    """R1: the 3rd (the 4th for 7sus) and the 7th (bb7 for dim7)."""
    t = QUALITY[q]
    return frozenset({t.third % 12, t.seventh % 12})


def colour(q: str, style: str) -> frozenset[int]:
    """R2: the top named extension, plus the b5 of m7b5 and dim7. A shell is three notes, so the m7b5 shell
    holds its b5 and not its 11."""
    t = QUALITY[q]
    top = {t.extensions[-1] % 12} if t.extensions else set()
    if t.fifth == PERFECT_FIFTH:
        return frozenset(top)
    return frozenset({t.fifth % 12} | (set() if style == "shell" else top))


def mode_pcs(numeral: str, mode: str) -> frozenset[int]:
    """The brief's mode as pitch classes above the chord root."""
    return frozenset((s - NUMERAL_SEMITONES[numeral]) % 12 for s in MODES[mode])


def added(q: str, numeral: str, mode: str) -> frozenset[int]:
    """R3 (b), V2.1: the tensions a quartal stack may add: the quality's available tensions that are in the brief's
    mode and that the symbol does not name."""
    return frozenset(AVAILABLE[q] & mode_pcs(numeral, mode)) - named(q)


def avoids(held: set[int], q: str) -> bool:
    """R3's avoid-note rule (V2.1): no added tension sits a semitone, in pitch class, above the root, which the bass
    always plays, or above a named tone the voicing also holds (no natural 11 over a major 3rd, no b9 over the
    bass)."""
    names = named(q)
    sounding = (set(held) & names) | {0}
    return not any((t - 1) % 12 in sounding for t in set(held) - names)


def rootless_template(q: str) -> list[int]:
    """R4 rootless: no root. The 3rd, the 7th and the named extensions (at most two, the top one always), plus
    the b5 of m7b5; with no extension, 3rd-5th-7th."""
    t = QUALITY[q]
    if not t.extensions:
        return sorted((t.third, t.fifth, t.seventh))
    altered = (t.fifth,) if t.fifth != PERFECT_FIFTH else ()
    return sorted((t.third, t.seventh, *altered, *t.extensions[-2:]))


def shell_template(q: str) -> list[int]:
    """R4 shell: the 3rd, the 7th and the top extension; b3-b5-b7 for m7b5 and b3-b5-bb7 for dim7."""
    t = QUALITY[q]
    if t.fifth != PERFECT_FIFTH:
        return sorted((t.third, t.fifth, t.seventh))
    return sorted((t.third, t.seventh, *t.extensions[-1:]))


def inverted(template) -> list[list[int]]:
    """Section 3: inversion j raises the template's lowest j tones by an octave."""
    t = sorted(template)
    return [sorted(t[j:] + [x + 12 for x in t[:j]]) for j in range(len(t))]


def dropped(v) -> list[int]:
    """Drop-2: the second-highest note goes down an octave."""
    s = sorted(v)
    return sorted(s[:-2] + [s[-2] - 12] + s[-1:])


def normal(shape) -> tuple[int, ...]:
    """A shape moved by octaves so that its lowest note lies in [0, 12)."""
    s = sorted(shape)
    return tuple(x - 12 * (s[0] // 12) for x in s)


def is_stack(v) -> bool:
    """R4 quartal: every adjacent interval is 5 or 6 semitones, except at most one of 3 or 4."""
    steps = [b - a for a, b in zip(v, v[1:])]
    return all(3 <= s <= 6 for s in steps) and sum(s <= 4 for s in steps) <= 1


@lru_cache(maxsize=None)
def quartal_stacks(q: str, numeral: str, mode: str) -> frozenset[tuple[int, ...]]:
    """R4 quartal: every stack of 3 or 4 tones from R3's tone set (the named tones plus `added`) that holds R1
    and R2 and keeps the avoid-note rule. Brute force over every bottom-to-top ordering of every subset: each
    adjacent interval is under an octave, so an ordering fixes the stack up to octaves. Stacks start in [0, 12)."""
    names, extra = named(q), added(q, numeral, mode)
    need = guide(q) | colour(q, "quartal")
    out = set()
    for size in (3, 4):
        for tones in combinations(sorted(names | extra), size):
            held = set(tones)
            if not need <= held or not avoids(held, q):
                continue
            for order in permutations(tones):
                shape = [order[0]]
                for a, b in zip(order, order[1:]):
                    shape.append(shape[-1] + (b - a) % 12)
                if is_stack(shape):
                    out.add(tuple(shape))
    return frozenset(out)


def sharp9_above_third(v, root: int) -> bool:
    """R5: a #9 (3 semitones above the root, over a major 3rd) sounds above the 3rd, never under it."""
    return all(p > t for p in v if (p - root) % 12 == 3 for t in v if (t - root) % 12 == 4)


def register_ok(v, q: str) -> bool:
    """R6, unchanged: inside C3-C5; no interval of 2 semitones or less whose lower note is below 55; no
    13-semitone pair unless the chord is 7b9."""
    s = sorted(v)
    return (LO <= s[0] and s[-1] <= HI and not any(b - a <= 2 and a < 55 for a, b in zip(s, s[1:]))
            and (q == "7b9" or all(b - a != 13 for a, b in combinations(s, 2))))


def placed(root: int, shapes, q: str) -> frozenset[tuple[int, ...]]:
    """Every octave placement of each shape on the root that lies in C3-C5 and passes R5 and R6."""
    out = set()
    for shape in shapes:
        for k in range(-2, 10):
            v = tuple(root + s + 12 * k for s in sorted(shape))
            if register_ok(v, q) and sharp9_above_third(v, root):
                out.add(v)
    return frozenset(out)


@lru_cache(maxsize=None)
def oracle_candidates(root: int, q: str, style: str, numeral: str, mode: str) -> frozenset[tuple[int, ...]]:
    """Every voicing V2 allows for one chord. Quartal falls back to rootless when no stack has a placement for
    this chord in this mode on this root (R4)."""
    if style == "rootless":
        return placed(root, inverted(rootless_template(q)), q)
    if style == "shell":
        return placed(root, inverted(shell_template(q)), q)
    if style == "spread":
        return placed(root, [dropped(v) for v in inverted(rootless_template(q))], q)
    if style == "quartal":
        return placed(root, quartal_stacks(q, numeral, mode), q) or oracle_candidates(root, q, "rootless",
                                                                                      numeral, mode)
    raise ValueError(style)


@lru_cache(maxsize=None)
def pairing(na: int, nb: int) -> tuple[int, ...]:
    """Section 3 step 3: voice i of A moves to voice round(i (|B| - 1) / (|A| - 1)) of B, halves up."""
    if na == 1:
        return (0,)
    return tuple(int(Fraction(i * (nb - 1), na - 1) + Fraction(1, 2)) for i in range(na))


def cost48(a, b) -> int:
    """48 x (VL(A->B) + VL(B->A) + 0.25 (|muA - 60| + |muB - 60|)), an exact integer for 2, 3 or 4 notes."""
    assert 12 % len(a) == 0 and 12 % len(b) == 0, (a, b)

    def vl(x, y):
        return sum(abs(p - y[j]) for p, j in zip(x, pairing(len(x), len(y))))

    def centre(v):                      # 48 x 0.25 x |sum/n - 60|
        return (12 // len(v)) * abs(sum(v) - 60 * len(v))
    return 48 * (vl(a, b) + vl(b, a)) + centre(a) + centre(b)


def best_pair(cands_a, cands_b) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The minimum-cost pair, ties to the lexicographically smallest (A, B), over two candidate sets."""
    return min(((a, b) for a in cands_a for b in cands_b), key=lambda ab: (cost48(*ab), ab))


def root_of(numeral: str, tonic: int) -> int:
    return (tonic + NUMERAL_SEMITONES[numeral]) % 12


def oracle_vamp(first: tuple[str, str], second: tuple[str, str], tonic: int, mode: str, style: str):
    """The V2 voicing pair of a vamp: chords as (numeral, quality)."""
    (na, qa), (nb, qb) = first, second
    return best_pair(oracle_candidates(root_of(na, tonic), qa, style, na, mode),
                     oracle_candidates(root_of(nb, tonic), qb, style, nb, mode))


def note_names(v) -> str:
    """MIDI pitches as note names, C4 = 60."""
    return " ".join(f"{NOTE[p % 12]}{p // 12 - 1}" for p in v)
