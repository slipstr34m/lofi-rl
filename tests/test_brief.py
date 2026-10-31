"""Briefs (TASTE.md, Briefs): the frozen dataclass and the deterministic brief sampler."""
import math
from collections import Counter
from itertools import combinations

import pytest

from lofi_rl.brief import (BPM_CENTER, BPM_RANGE, CENTER_WEIGHT, ENERGIES, MOOD_NAMES, SCENE, TEMPO_WEIGHTS, Brief,
                           sample_brief, sample_briefs)

FIELDS = ("energy", "mood", "bpm", "tonic")


def test_mood_picks_the_mode():
    modes = {m: Brief("mid", m, 80, 0).mode for m in MOOD_NAMES}
    assert modes == {"melancholy": "aeolian", "wistful": "dorian", "hopeful": "ionian"}
    assert Brief("mid", "wistful", 82, 2).key_name == "D dorian"


@pytest.mark.parametrize("kw", [
    {"energy": "medium"}, {"mood": "happy"}, {"bpm": 59}, {"bpm": 96}, {"bpm": 82.5}, {"bpm": True},
    {"tonic": 12}, {"tonic": -1}, {"tonic": True}, {"scene": "beach"},
])
def test_invalid_briefs_are_refused(kw):
    args = {"energy": "mid", "mood": "wistful", "bpm": 82, "tonic": 2, **kw}
    with pytest.raises(ValueError):
        Brief(**args)


def test_brief_is_frozen_and_hashable():
    b = Brief("low", "hopeful", 75, 9)
    with pytest.raises(AttributeError):
        b.bpm = 90
    assert {b: 1}[Brief("low", "hopeful", 75, 9)] == 1 and b.scene == SCENE


def test_sampler_is_deterministic_and_prefix_stable():
    a = sample_briefs(1234, 50)
    assert a == sample_briefs(1234, 50)
    assert sample_briefs(1234, 20) == a[:20]
    assert sample_briefs(1234, 10, start=40) == a[40:]
    assert sample_brief(1234, 17) == a[17]
    assert sample_briefs(1235, 50) != a


def test_seeds_a_multiple_of_256_apart_do_not_alias():
    # The old key, (seed & MASK) ^ (index << 8) ^ field, gave brief 0 of seed 256 the keys of brief 1 of seed 0,
    # and in general brief i of seed 256k the keys of brief i ^ k of seed 0.
    assert sample_brief(256, 0) != sample_brief(0, 1)
    for k in (1, 2, 8):
        assert [sample_brief(256 * k, i) for i in range(64)] != [sample_brief(0, i ^ k) for i in range(64)], k


def _chance(field: str) -> float:
    """P(two independent briefs agree on `field`)."""
    if field == "bpm":
        total = sum(w for _, w in TEMPO_WEIGHTS)
        return sum((w / total) ** 2 for _, w in TEMPO_WEIGHTS)
    return 1 / {"energy": len(ENERGIES), "mood": len(MOOD_NAMES), "tonic": 12}[field]


def test_streams_from_different_seeds_agree_position_by_position_at_the_chance_rate():
    n = 2000
    streams = {seed: sample_briefs(seed, n) for seed in (0, 256, 512, 2026, 2282)}
    for a, b in combinations(streams, 2):
        # Position by position alone cannot see the old alias: it replayed the same briefs in another order.
        assert Counter(streams[a]) != Counter(streams[b]), f"seed {b} replays the briefs of seed {a}"
        for field in FIELDS:
            p = _chance(field)
            agree = sum(getattr(x, field) == getattr(y, field) for x, y in zip(streams[a], streams[b]))
            assert abs(agree - n * p) <= 4 * math.sqrt(n * p * (1 - p)), (a, b, field, agree, round(n * p, 1))


def _category(brief: Brief, field: str):
    """The field's value; the tempo as one of 3 bands (below, inside, above 75-90), so no table cell is sparse."""
    if field == "bpm":
        return (brief.bpm >= BPM_CENTER[0]) + (brief.bpm > BPM_CENTER[1])
    return getattr(brief, field)


def _chi2_z(xs: list, ys: list) -> float:
    """(chi2 - df) / sqrt(2 df) of the contingency table of xs against ys: about N(0, 1) when independent."""
    n, cx, cy, cxy = len(xs), Counter(xs), Counter(ys), Counter(zip(xs, ys))
    chi2 = sum((cxy[a, b] - cx[a] * cy[b] / n) ** 2 / (cx[a] * cy[b] / n) for a in cx for b in cy)
    df = (len(cx) - 1) * (len(cy) - 1)
    return (chi2 - df) / math.sqrt(2 * df)


def test_adjacent_seeds_do_not_couple_fields():
    # The old key gave the energy of seed 2026 the hash of the mood of seed 2025 (2025 ^ 2026 = 3 = 1 ^ 2), and the
    # mood of seed 2027 the hash of the tempo of seed 2026 (2026 ^ 2027 = 1 = 2 ^ 3).
    n = 2000
    streams = {seed: sample_briefs(seed, n) for seed in (2025, 2026, 2027)}
    for a in (2025, 2026):
        for f in FIELDS:
            for g in FIELDS:
                z = _chi2_z([_category(x, f) for x in streams[a]], [_category(y, g) for y in streams[a + 1]])
                assert z < 6, (f"{f} of seed {a}", f"{g} of seed {a + 1}", round(z, 1))


def test_brief_indices_span_64_bits():
    last = (1 << 64) - 1
    assert sample_brief(0, last) == sample_brief(0, last) and sample_brief(0, last) != sample_brief(1, last)
    for bad in (-1, 1 << 64):
        with pytest.raises(ValueError, match="index"):
            sample_brief(0, bad)


def test_tempo_weights_centre_on_75_90():
    assert [b for b, _ in TEMPO_WEIGHTS] == list(range(BPM_RANGE[0], BPM_RANGE[1] + 1))
    centre = sum(w for b, w in TEMPO_WEIGHTS if BPM_CENTER[0] <= b <= BPM_CENTER[1])
    assert centre == 16 * CENTER_WEIGHT and sum(w for _, w in TEMPO_WEIGHTS) == 84


def test_sampled_distribution():
    briefs = sample_briefs(2026, 12_000)
    bpm = Counter(b.bpm for b in briefs)
    assert min(bpm) >= 60 and max(bpm) <= 95 and len(bpm) == 36
    share = sum(c for t, c in bpm.items() if 75 <= t <= 90) / len(briefs)
    assert abs(share - 64 / 84) < 0.02
    for field, values in (("tonic", range(12)), ("energy", ENERGIES), ("mood", MOOD_NAMES)):
        counts = Counter(getattr(b, field) for b in briefs)
        assert set(counts) == set(values)
        expected = len(briefs) / len(values)
        assert all(abs(c - expected) < 0.12 * expected for c in counts.values()), (field, counts)


def _splitmix64(x: int) -> int:
    """Vigna's SplitMix64, written out here so the brief key is checked against docs/COMPILER.md, not prng.py."""
    z = (x + 0x9E3779B97F4A7C15) % 2 ** 64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) % 2 ** 64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) % 2 ** 64
    return z ^ (z >> 31)


def test_each_field_is_its_own_keyed_draw_as_documented():
    # docs/COMPILER.md §6: h = splitmix64(splitmix64(splitmix64(seed) ^ index) ^ field), with energy 1, mood 2,
    # tempo 3 and tonic 4, and the value ((h >> 11) k) >> 53 for k choices (the tempo over its 84 weight units).
    units = [bpm for bpm, w in TEMPO_WEIGHTS for _ in range(w)]
    for seed in (0, 2026, 2 ** 64 - 1):
        for i in range(300):
            h = [_splitmix64(_splitmix64(_splitmix64(seed) ^ i) ^ field) for field in (1, 2, 3, 4)]
            pick = [((x >> 11) * k) >> 53 for x, k in zip(h, (3, 3, 84, 12))]
            want = Brief(ENERGIES[pick[0]], MOOD_NAMES[pick[1]], units[pick[2]], pick[3])
            assert sample_brief(seed, i) == want, (seed, i)


def test_the_fields_of_one_brief_are_independent():
    # The two knobs must not move together: training and held-out briefs would confound energy and mood, the
    # two knobs the obedience claim measures (CLAUDE.md §9). Equal field ids would draw both from one hash.
    briefs = sample_briefs(2026, 6000)
    assert len({(b.energy, b.mood) for b in briefs}) == 9
    assert len({(b.mood, b.tonic) for b in briefs}) == 36 and len({(b.energy, b.tonic) for b in briefs}) == 36
    for f, g in combinations(FIELDS, 2):
        z = _chi2_z([_category(b, f) for b in briefs], [_category(b, g) for b in briefs])
        assert z < 6, (f, g, round(z, 1))
