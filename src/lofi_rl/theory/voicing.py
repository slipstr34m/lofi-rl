"""Keys voicings and the voice-leading choice across the vamp: instrument.md section 3, Voicings, as amended by
voicing rules V2.1 (docs/COMPILER.md §3.3 and §9, "Amendment V2.1 to instrument.md §3 Voicings").

Tones are semitones above the chord root. A quality's named tones are its table entry (theory.chords): root,
third, fifth, seventh and extensions. Every voicing holds the third (the fourth for 7sus) and the seventh (R1),
and the top named extension plus the b5 of m7b5 and dim7 (R2). The styles (R4):
- rootless: no root. The third, the seventh and the named extensions (at most two, the top one always), plus
  m7b5's b5; with no extension, third-fifth-seventh;
- shell: the third and the seventh, plus the top extension; b3-b5-b7 for m7b5 and b3-b5-bb7 for dim7;
- spread: drop-2 of each inversion of the rootless voicing;
- quartal: every stack of 3 or 4 tones drawn from the named tones and the quality's available tensions that are
  in the brief's mode (R3, AVAILABLE_TENSIONS) that holds R1 and R2, adds no tension a semitone above the root
  (which the bass plays) or above a named tone it holds (R3), and has adjacent intervals of 5 or 6 semitones,
  except at most one of 3 or 4. A chord with no quartal candidate is voiced rootless.

The choice (section 3, steps 1-4, plus R8): every inversion at every octave inside C3-C5 (48-72) is a candidate
(for quartal, every stack at every octave); a #9 under the third (R5), an interval of 2 semitones or less below
55, and a 13-semitone pair outside 7b9 are rejected; among the pairs of two different voicings (R8: the vamp
moves; only two equal one-candidate sets leave none), the pair minimizing VL(A->B) + VL(B->A) +
0.25 (|muA - 60| + |muB - 60|) wins, ties broken by the lexicographically smallest (A, B). Costs are exact.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from functools import lru_cache
from math import lcm
from types import MappingProxyType

from lofi_rl.dsl.ast import Chord
from lofi_rl.dsl.vocab import NUMERAL_SEMITONES, VOICINGS, scale_of

from .chords import Quality, quality, root_pc

KEYS_RANGE = (48, 72)           # C3-C5, sounding MIDI
LOW_LIMIT = 55                  # an interval of 2 semitones or less needs its lower note at 55 or above
CLUSTER = 2
MINOR_NINTH = 13                # rejected unless the chord is 7b9
MU_CENTER = 60
MU_WEIGHT = Fraction(1, 4)
PERFECT_FIFTH = 7               # only a perfect fifth goes under an extension (m7b5 keeps its b5)
ROOT = 0                        # the bass always plays it, so R3's avoid-note rule counts it as held (V2.1)
# V2.1 R3: the tensions a quartal stack may add, by family (chords.Quality.family), and only those in the brief's
# mode. A dominant's b9 or #9 comes only with a symbol that names it (7b9, 7#9), as a chord tone.
AVAILABLE_TENSIONS: Mapping[str, frozenset[int]] = MappingProxyType({
    "maj": frozenset({2, 6, 9}),        # 9 #11 13
    "dom": frozenset({2, 9}),           # 9 13
    "min": frozenset({2, 5, 9}),        # 9 11 13
    "half": frozenset({5, 8}),          # 11 b13
    "dim": frozenset(),
    "sus": frozenset({2, 9}),           # 9 13
})
FOURTHS = (5, 6)                # quartal: every adjacent interval ...
THIRDS = (3, 4)                 # ... except at most one of these
STACK_SIZES = (3, 4)
MAJOR_THIRD, SHARP_NINTH = 4, 3     # pitch classes: R5 keeps the #9 above the third

Voicing = tuple[int, ...]       # sorted MIDI pitches, or a sorted shape in semitones above the root


def chord_scale(numeral: str, mode: str) -> frozenset[int]:
    """The brief's mode as pitch classes above the chord's root (a mood names its mode). R3: a quartal stack
    adds only tensions in it."""
    if numeral not in NUMERAL_SEMITONES:
        raise ValueError(f"unknown numeral {numeral!r}")
    root = NUMERAL_SEMITONES[numeral]
    return frozenset((s - root) % 12 for s in scale_of(mode))


def guide_tones(q: Quality) -> frozenset[int]:
    """R1: the third (the fourth for sus) and the seventh (bb7 for dim7), as pitch classes."""
    return frozenset({q.third % 12, q.seventh % 12})


def colour(q: Quality) -> frozenset[int]:
    """R2: the top named extension, plus the b5 of m7b5 and dim7, as pitch classes. The one exception, the
    m7b5 shell, keeps only the b5 (`shell`)."""
    top = {q.extensions[-1] % 12} if q.extensions else set()
    altered = {q.fifth % 12} if q.fifth != PERFECT_FIFTH else set()
    return frozenset(top | altered)


def rootless(q: Quality) -> Voicing:
    """R4 rootless: the third, the seventh and the named extensions (at most two, the top one always), plus an
    altered fifth; with no extension, third-fifth-seventh."""
    if not q.extensions:
        return (q.third, q.fifth, q.seventh)
    altered = (q.fifth,) if q.fifth != PERFECT_FIFTH else ()
    return tuple(sorted((q.third, q.seventh) + altered + q.extensions[-2:]))


def shell(q: Quality) -> Voicing:
    """R4 shell: the third (the fourth for sus) and the seventh, plus the top extension. A shell is three notes,
    so m7b5 and dim7 hold their b5 instead: b3-b5-b7 and b3-b5-bb7."""
    if q.fifth != PERFECT_FIFTH:
        return (q.third, q.fifth, q.seventh)
    return (q.third, q.seventh) + q.extensions[-1:]


def _avoids(held: frozenset[int], named: frozenset[int]) -> bool:
    """R3's avoid-note rule (V2.1): no added tension a semitone above the root, which the bass always plays, or
    above a named tone the stack holds."""
    sounding = (held & named) | {ROOT}
    return not any((t - 1) % 12 in sounding for t in held - named)


@lru_cache(maxsize=None)
def quartal(q: Quality, scale: frozenset[int]) -> tuple[Voicing, ...]:
    """R4 quartal: every stack of 3 or 4 distinct tones from R3's tone set (the named tones, plus the quality's
    available tensions that are in `scale`, the brief's mode above the root) that holds R1 and R2 and keeps the
    avoid-note rule. Adjacent intervals are 5 or 6 semitones, except at most one of 3 or 4. Each stack starts in
    [0, 12); () if none."""
    named = frozenset(t % 12 for t in q.tones)
    tones = named | (AVAILABLE_TENSIONS[q.family] & scale)
    need = guide_tones(q) | colour(q)
    out = []

    def grow(stack: tuple[int, ...], held: frozenset[int], third: bool) -> None:
        if len(stack) in STACK_SIZES and need <= held and _avoids(held, named):
            out.append(stack)
        if len(stack) == STACK_SIZES[-1]:
            return
        for step in THIRDS + FOURTHS:
            top = stack[-1] + step
            if top % 12 in tones and top % 12 not in held and not (third and step in THIRDS):
                grow(stack + (top,), held | {top % 12}, third or step in THIRDS)

    for start in sorted(tones):
        grow((start,), frozenset({start}), False)
    return tuple(sorted(out))


def sharp9_above_third(v: Sequence[int], q: Quality) -> bool:
    """R5: over a major third, a #9 sounds above the third, never a semitone (or a minor 9th) under it. `v` is
    a shape in semitones above the root."""
    if q.third != MAJOR_THIRD:
        return True
    return all(n > t for n in v if n % 12 == SHARP_NINTH for t in v if t % 12 == MAJOR_THIRD)


def inversions(template: Sequence[int]) -> tuple[Voicing, ...]:
    """Inversion j raises the template's lowest j tones by an octave (j = 0 .. len-1)."""
    t = sorted(template)
    return tuple(tuple(sorted(t[j:] + [x + 12 for x in t[:j]])) for j in range(len(t)))


def drop2(v: Sequence[int]) -> Voicing:
    """Drop the second-highest note an octave."""
    s = sorted(v)
    s[-2] -= 12
    return tuple(sorted(s))


def shapes(q: Quality, style: str, scale: Iterable[int] | None = None) -> tuple[Voicing, ...]:
    """Every shape of a style for one chord, as semitones above the root (sorted, distinct), minus R5's
    rejections. Quartal needs `scale`, the brief's mode above the root (chord_scale), and may be empty."""
    if style == "rootless":
        out = inversions(rootless(q))
    elif style == "shell":
        out = inversions(shell(q))
    elif style == "spread":
        out = tuple(drop2(v) for v in inversions(rootless(q)))
    elif style == "quartal":
        if scale is None:
            raise ValueError("a quartal voicing needs the brief's mode (voicing.chord_scale)")
        out = quartal(q, frozenset(scale))
    else:
        raise ValueError(f"unknown voicing style {style!r}; expected one of {VOICINGS}")
    return tuple(sorted({s for s in out if sharp9_above_third(s, q)}))


def allowed(v: Sequence[int], q: Quality) -> bool:
    """Rule 2: no interval of 2 semitones or less whose lower note is below 55; no 13-semitone pair
    unless the chord is 7b9."""
    s = sorted(v)
    if any(b - a <= CLUSTER and a < LOW_LIMIT for a, b in zip(s, s[1:])):
        return False
    if q.name != "7b9" and any(b - a == MINOR_NINTH for i, a in enumerate(s) for b in s[i + 1:]):
        return False
    return True


def _placed(root: int, q: Quality, shapes_: Iterable[Voicing], lo: int, hi: int) -> tuple[Voicing, ...]:
    out = set()
    for shape in shapes_:
        base = [root + s for s in shape]
        first = -((base[0] - lo) // 12)               # the lowest octave shift with every note >= lo
        for k in range(first, (hi - base[-1]) // 12 + 1):
            v = tuple(p + 12 * k for p in base)
            if allowed(v, q):
                out.add(v)
    return tuple(sorted(out))


def candidates(root: int, quality_name: str, style: str, scale: Iterable[int] | None = None,
               lo: int = KEYS_RANGE[0], hi: int = KEYS_RANGE[1]) -> tuple[Voicing, ...]:
    """Rules 1 and 2: every shape at every octave with all notes in [lo, hi], minus rejections. `scale` is the
    brief's mode above the root, needed for quartal. A quartal chord with no candidate (no stack, or none that
    fits) takes the rootless candidates (R4): a voicing, not a repair."""
    q = quality(quality_name)
    out = _placed(root, q, shapes(q, style, scale), lo, hi)
    if not out and style == "quartal":
        out = _placed(root, q, shapes(q, "rootless"), lo, hi)
    return out


def _vl(sa: Sequence[int], sb: Sequence[int]) -> int:
    na, nb = len(sa), len(sb)
    if na == 1:
        return abs(sa[0] - sb[0])
    return sum(abs(x - sb[(2 * i * (nb - 1) + na - 1) // (2 * (na - 1))]) for i, x in enumerate(sa))


def voice_leading(a: Sequence[int], b: Sequence[int]) -> int:
    """VL(A->B): summed movement under sorted pairing.

    With equal sizes voice i of A moves to voice i of B. With unequal sizes voice i of A moves to voice
    round(i (|B|-1) / (|A|-1)) of B (halves up), so the outer voices still pair bottom-bottom, top-top.
    """
    return _vl(sorted(a), sorted(b))


def mean_pitch(v: Sequence[int]) -> Fraction:
    return Fraction(sum(v), len(v))


def pair_cost(a: Sequence[int], b: Sequence[int]) -> Fraction:
    """VL(A->B) + VL(B->A) + 0.25 (|muA - 60| + |muB - 60|)."""
    centre = MU_WEIGHT * (abs(mean_pitch(a) - MU_CENTER) + abs(mean_pitch(b) - MU_CENTER))
    return voice_leading(a, b) + voice_leading(b, a) + centre


def choose(cands_a: Sequence[Voicing], cands_b: Sequence[Voicing]) -> tuple[Voicing, Voicing]:
    """The minimum-cost pair of two different voicings (R8, V2.1: the vamp moves), or the only pair there is when
    both chords have the same single candidate; ties go to the lexicographically smallest (A, B).

    Costs compare as exact integers: pair_cost times S = 4 x lcm(the voicing sizes), 4 being MU_WEIGHT's
    denominator. S is the same for every pair, so the order and the ties are pair_cost's."""
    if not cands_a or not cands_b:
        raise ValueError("a chord has no voicing candidate")
    den = lcm(*{len(v) for v in (*cands_a, *cands_b)})
    factor = MU_WEIGHT.denominator * den

    def prepared(cands: Sequence[Voicing]) -> list[tuple[Voicing, list[int], int]]:
        # S x MU_WEIGHT x |mu - 60| = numerator x (den / n) x |sum - 60 n|
        return [(v, sorted(v), MU_WEIGHT.numerator * (den // len(v)) * abs(sum(v) - MU_CENTER * len(v)))
                for v in cands]

    pa, pb = prepared(cands_a), prepared(cands_b)
    # A pair whose two voicings hold the same pitches sorts after every pair that moves, whatever its cost.
    _, _, a, b = min((sa == sb, factor * (_vl(sa, sb) + _vl(sb, sa)) + ca + cb, a, b)
                     for a, sa, ca in pa for b, sb, cb in pb)
    return a, b


def voice_vamp(first: Chord, second: Chord, tonic: int, mode: str, style: str) -> tuple[Voicing, Voicing]:
    """One voicing per vamp chord, kept for the whole loop (section 3, step 4), the two different whenever they
    can be (R8). The brief's mode sets the tensions a quartal stack may add (R3)."""
    return choose(candidates(root_pc(first.numeral, tonic), first.quality, style, chord_scale(first.numeral, mode)),
                  candidates(root_pc(second.numeral, tonic), second.quality, style,
                             chord_scale(second.numeral, mode)))
