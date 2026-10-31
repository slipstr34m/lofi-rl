"""The texture bed (instrument.md §4 Rain bed): the 1 s equal-power seam crossfade, the 200 Hz high-pass,
mono or stereo sources, and the level at LUFS_music - Δ. The bed sources here are synthetic noise made
inside the tests; the rain file comes later."""
import math

import numpy as np
import pytest

from helpers_compile import dev_palette
from helpers_render import noise, with_bed
from lofi_rl.render.bed import HIGHPASS, BedError, bed_loop, place, read_excerpt, samples, seam_loop
from lofi_rl.render.loudness import lufs
from lofi_rl.render.mix import process
from lofi_rl.render.wavio import write_float

SR = 44100


def test_seam_crossfade_follows_the_formula():
    n, x = 1000, 100
    e = noise(2, n + x + 50, 3)
    o = seam_loop(e, n, x)
    assert o.shape == (2, n) and o.dtype == np.float32
    assert np.array_equal(o[:, :n - x], e[:, x:n]), "o[k] = e[X + k] before the fade"
    assert np.array_equal(o[:, n - x], e[:, n]), "the fade starts on e[N], right after e[N - 1]"
    for j in (1, 37, x - 1):
        th = math.pi * j / (2 * x)
        e64 = e.astype(np.float64)
        want = (math.cos(th) * e64[:, n + j] + math.sin(th) * e64[:, j]).astype(np.float32)
        assert np.array_equal(o[:, n - x + j], want)


def test_the_loop_wraps_without_a_jump():
    n, x = 20_000, 2_000
    t = np.arange(n + x)
    e = np.stack([np.sin(2 * np.pi * t / 1234.5), np.cos(2 * np.pi * t / 987.6)]).astype(np.float32)
    steps = np.abs(np.diff(np.tile(seam_loop(e, n, x), 2), axis=1)).max()
    assert steps <= 1.05 * np.abs(np.diff(e, axis=1)).max()
    plain = np.tile(e[:, :n], 2)
    assert np.abs(np.diff(plain, axis=1)).max() > 5 * steps, "without the fade the wrap jumps"


def test_the_crossfade_keeps_the_power_of_uncorrelated_noise():
    n, x = 200_000, 44_100
    o = seam_loop(noise(2, n + x, 11), n, x).astype(np.float64)
    fade, rest = np.sqrt(np.mean(o[:, n - x:] ** 2)), np.sqrt(np.mean(o[:, :n - x] ** 2))
    assert abs(fade / rest - 1) < 0.02


def test_seam_loop_refuses_impossible_lengths():
    with pytest.raises(BedError):
        seam_loop(noise(2, 150, 1), 100, 100)
    with pytest.raises(BedError):
        seam_loop(noise(2, 500, 1), 100, 200)
    assert np.array_equal(seam_loop(noise(2, 100, 1), 100, 0), noise(2, 100, 1))


def test_excerpts_are_high_passed_from_start_s_and_mono_plays_in_both_channels(tmp_path):
    src = noise(1, 3 * SR, 5)
    write_float(tmp_path / "mono.wav", src, SR)
    start, length = samples(0.25), SR
    assert start == 11025
    e = read_excerpt(tmp_path / "mono.wav", start, length)
    assert e.shape == (2, length) and np.array_equal(e[0], e[1])
    pre = min(start, samples(0.5))
    want = process(HIGHPASS, np.concatenate([src, src])[:, start - pre:start + length])[:, pre:]
    assert np.array_equal(e, want)
    assert [p.name for p in HIGHPASS] == ["HighpassFilter"] and dict(HIGHPASS[0].params) == {"cutoff_frequency_hz": 200.0}
    with pytest.raises(BedError, match="needs"):
        read_excerpt(tmp_path / "mono.wav", start, 3 * SR)
    write_float(tmp_path / "fast.wav", src, 48000)
    with pytest.raises(BedError, match="48000 Hz"):
        read_excerpt(tmp_path / "fast.wav", 0, 100)


def test_place_sets_the_bed_delta_below_the_music():
    bed = noise(2, 4 * SR, 9)
    scaled, gain, own = place(bed, -31.0, 24.0)
    assert own == pytest.approx(lufs(bed, SR)) and gain == pytest.approx(-31.0 - 24.0 - own)
    assert lufs(scaled, SR) == pytest.approx(-55.0, abs=0.01)
    with pytest.raises(BedError, match="silent"):
        place(np.zeros((2, SR), dtype=np.float32), -31.0, 24.0)


def test_bed_null_means_no_bed_and_a_palette_bed_is_one_seamless_cycle(tmp_path):
    assert dev_palette().bed is None and bed_loop(dev_palette(), 64 * 1000) is None
    n = 64 * 2000
    for channels in (1, 2):
        pal = with_bed(tmp_path, n, channels=channels)
        o = bed_loop(pal, n)
        assert o.shape == (2, n) and np.all(np.isfinite(o))
        assert np.array_equal(o[0], o[1]) == (channels == 1)
