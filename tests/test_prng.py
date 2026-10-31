"""SplitMix64 (instrument.md section 3, Humanize): the published test vector and the exact draws."""
from fractions import Fraction

import pytest

from lofi_rl.prng import GOLDEN, MASK, TEST_VECTOR, below, splitmix64, stream, uniform


def test_published_test_vector():
    assert splitmix64(0) == 0xE220A8397B1DCDAF == TEST_VECTOR[1]


def test_reference_sequence_for_seed_0():
    # The first outputs of Vigna's splitmix64.c (state x = 0, x += 0x9e3779b97f4a7c15 per call).
    assert stream(0, 4) == (0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F, 0xF88BB8A8724C81EC)
    assert stream(0, 4)[1] == splitmix64(GOLDEN)


def test_outputs_are_64_bit_and_inputs_wrap():
    for x in (0, 1, MASK, 1 << 63, 123456789):
        assert 0 <= splitmix64(x) <= MASK
    assert splitmix64(1 << 64) == splitmix64(0)


def test_uniform_is_the_top_53_bits_exactly():
    assert uniform(0) == 0
    assert uniform(MASK) == 1 - Fraction(1, 1 << 53)
    assert uniform(1 << 11) == Fraction(1, 1 << 53)
    assert uniform((1 << 11) - 1) == 0                       # the low 11 bits never matter


@pytest.mark.parametrize("k", [1, 2, 3, 12, 36, 84])
def test_below_spans_exactly_0_to_k(k):
    assert below(0, k) == 0 and below(MASK, k) == k - 1
    assert {below(h, k) for h in stream(7, 4000)} == set(range(k))


def test_below_rejects_an_empty_range():
    with pytest.raises(ValueError):
        below(5, 0)
