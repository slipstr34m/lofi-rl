"""Keys voicings and the voice-leading choice across the vamp (docs/design/instrument.md section 3, Voicings).

Styles, as semitones above the root:
- rootless: the core tones minus the root, plus the first extension; the fifth goes when there is one;
- shell: the third (the fourth for sus) and the seventh, plus the top extension;
- spread: drop-2 of each inversion of the rootless voicing;
- quartal: maj [4,9,14,18|19] (18 for maj7#11), min [5,10,15,19], dom [10,16,21] plus each extension not
  already in the stack, raised above its top; sus [5,10,14,19]; m7b5 and dim7 fall back to rootless.

The choice (section 3, steps 1-4): every inversion at every octave inside C3-C5 (48-72) is a candidate;
candidates with an interval of 2 semitones or less below 55, or a 13-semitone pair outside 7b9, are
rejected; the pair minimizing VL(A->B) + VL(B->A) + 0.25 (|muA - 60| + |muB - 60|) wins, ties broken by
the lexicographically smallest (A, B). Costs are exact fractions, so ties are exact.
"""
from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from lofi_rl.dsl.ast import Chord
from lofi_rl.dsl.vocab import VOICINGS

from .chords import Quality, quality, root_pc

KEYS_RANGE = (48, 72)           # C3-C5, sounding MIDI
LOW_LIMIT = 55                  # an interval of 2 semitones or less needs its lower note at 55 or above
CLUSTER = 2
MINOR_NINTH = 13                # rejected unless the chord is 7b9
MU_CENTER = 60
MU_WEIGHT = Fraction(1, 4)

Voicing = tuple[int, ...]       # sorted MIDI pitches


def rootless(q: Quality) -> Voicing:
    if q.extensions:
        return (q.third, q.seventh, q.extensions[0])
    return (q.third, q.fifth, q.seventh)


def shell(q: Quality) -> Voicing:
    return (q.third, q.seventh) + q.extensions[-1:]


def quartal(q: Quality) -> Voicing:
    fam = q.family
    if fam == "maj":
        return (4, 9, 14, 18 if 18 in q.extensions else 19)
    if fam == "min":
        return (5, 10, 15, 19)
    if fam == "sus":
        return (5, 10, 14, 19)
    if fam == "dom":
        stack = [10, 16, 21]
        for ext in q.extensions:
            if ext % 12 not in {t % 12 for t in stack}:
                while ext <= stack[-1]:
                    ext += 12
                stack.append(ext)
        return tuple(stack)
    return rootless(q)          # m7b5 and dim7


def inversions(template: Sequence[int]) -> tuple[Voicing, ...]:
    """Inversion j raises the template's lowest j tones by an octave (j = 0 .. len-1)."""
    t = sorted(template)
    return tuple(tuple(sorted(t[j:] + [x + 12 for x in t[:j]])) for j in range(len(t)))


def drop2(v: Sequence[int]) -> Voicing:
    """Drop the second-highest note an octave."""
    s = sorted(v)
    s[-2] -= 12
    return tuple(sorted(s))


def shapes(q: Quality, style: str) -> tuple[Voicing, ...]:
    """Every inversion of a style's template, as semitones above the root (sorted, distinct)."""
    if style == "rootless":
        out = inversions(rootless(q))
    elif style == "shell":
        out = inversions(shell(q))
    elif style == "spread":
        out = tuple(drop2(v) for v in inversions(rootless(q)))
    elif style == "quartal":
        out = inversions(quartal(q))
    else:
        raise ValueError(f"unknown voicing style {style!r}; expected one of {VOICINGS}")
    return tuple(sorted(set(out)))


def allowed(v: Sequence[int], q: Quality) -> bool:
    """Rule 2: no interval of 2 semitones or less whose lower note is below 55; no 13-semitone pair
    unless the chord is 7b9."""
    s = sorted(v)
    if any(b - a <= CLUSTER and a < LOW_LIMIT for a, b in zip(s, s[1:])):
        return False
    if q.name != "7b9" and any(b - a == MINOR_NINTH for i, a in enumerate(s) for b in s[i + 1:]):
        return False
    return True


def candidates(root: int, quality_name: str, style: str,
               lo: int = KEYS_RANGE[0], hi: int = KEYS_RANGE[1]) -> tuple[Voicing, ...]:
    """Rule 1 and 2: every inversion at every octave with all notes in [lo, hi], minus rejections."""
    q = quality(quality_name)
    out = set()
    for shape in shapes(q, style):
        base = [root + s for s in shape]
        first = -((base[0] - lo) // 12)               # the lowest octave shift with every note >= lo
        for k in range(first, (hi - base[-1]) // 12 + 1):
            v = tuple(p + 12 * k for p in base)
            if allowed(v, q):
                out.add(v)
    return tuple(sorted(out))


def voice_leading(a: Sequence[int], b: Sequence[int]) -> int:
    """VL(A->B): summed movement under sorted pairing.

    With equal sizes voice i of A moves to voice i of B. With unequal sizes voice i of A moves to voice
    round(i (|B|-1) / (|A|-1)) of B (halves up), so the outer voices still pair bottom-bottom, top-top.
    """
    sa, sb = sorted(a), sorted(b)
    na, nb = len(sa), len(sb)
    if na == 1:
        return abs(sa[0] - sb[0])
    return sum(abs(x - sb[(2 * i * (nb - 1) + na - 1) // (2 * (na - 1))]) for i, x in enumerate(sa))


def mean_pitch(v: Sequence[int]) -> Fraction:
    return Fraction(sum(v), len(v))


def pair_cost(a: Sequence[int], b: Sequence[int]) -> Fraction:
    """VL(A->B) + VL(B->A) + 0.25 (|muA - 60| + |muB - 60|)."""
    centre = MU_WEIGHT * (abs(mean_pitch(a) - MU_CENTER) + abs(mean_pitch(b) - MU_CENTER))
    return voice_leading(a, b) + voice_leading(b, a) + centre


def choose(cands_a: Sequence[Voicing], cands_b: Sequence[Voicing]) -> tuple[Voicing, Voicing]:
    """The minimum-cost pair; ties go to the lexicographically smallest (A, B)."""
    if not cands_a or not cands_b:
        raise ValueError("a chord has no voicing candidate")
    return min(((a, b) for a in cands_a for b in cands_b), key=lambda ab: (pair_cost(*ab), ab))


def voice_vamp(first: Chord, second: Chord, tonic: int, style: str) -> tuple[Voicing, Voicing]:
    """One voicing per vamp chord, kept for the whole loop (section 3, step 4)."""
    return choose(candidates(root_pc(first.numeral, tonic), first.quality, style),
                  candidates(root_pc(second.numeral, tonic), second.quality, style))
