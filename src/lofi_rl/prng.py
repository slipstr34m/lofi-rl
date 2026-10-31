"""SplitMix64, the project's only random source (docs/design/instrument.md section 3, Humanize).

Every draw is a pure function of a 64-bit key, so a value never depends on how many draws came before,
on dict or set order, or on threads. Test vector: splitmix64(0) == 0xE220A8397B1DCDAF.
"""
from __future__ import annotations

from fractions import Fraction

MASK = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
TEST_VECTOR = (0, 0xE220A8397B1DCDAF)


def splitmix64(x: int) -> int:
    """One SplitMix64 output for state `x` (Steele, Lea and Flood 2014; Vigna's reference code)."""
    z = (x + GOLDEN) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def stream(seed: int, count: int) -> tuple[int, ...]:
    """The first `count` outputs of the SplitMix64 generator seeded with `seed` (state += GOLDEN per draw)."""
    return tuple(splitmix64((seed + i * GOLDEN) & MASK) for i in range(count))


def uniform(h: int) -> Fraction:
    """u = (h >> 11) * 2**-53 in [0, 1), exact: the top 53 bits of a 64-bit hash."""
    return Fraction(h >> 11, 1 << 53)


def below(h: int, k: int) -> int:
    """An index in [0, k) from the top 53 bits of `h`, in exact integer arithmetic."""
    if k < 1:
        raise ValueError("k must be at least 1")
    return ((h >> 11) * k) >> 53
