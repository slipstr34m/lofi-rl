"""Briefs (TASTE.md, Briefs): the frozen dataclass and the deterministic brief sampler."""
from collections import Counter

import pytest

from lofi_rl.brief import (BPM_CENTER, BPM_RANGE, CENTER_WEIGHT, ENERGIES, MOOD_NAMES, SCENE, TEMPO_WEIGHTS, Brief,
                           sample_brief, sample_briefs)


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
