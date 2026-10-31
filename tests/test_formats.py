"""Output files (instrument.md §4 Outputs): the float32 WAV with no PEAK chunk, and 16-bit FLAC listening
copies with TPDF dither from SplitMix64 keyed by the audio's pcm_sha256, the same bits every time."""
import numpy as np
import soundfile as sf

from helpers_render import noise
from lofi_rl.prng import stream
from lofi_rl.render.formats import (dither_seed, splitmix64_stream, to_int16, tpdf_noise, write_flac,
                                    write_wav)
from lofi_rl.render.wavio import pcm_sha256, read_float


def test_the_vectorized_stream_is_prng_stream():
    for seed in (0, 1, 0xDEADBEEF, (1 << 64) - 1):
        assert splitmix64_stream(seed, 500).tolist() == list(stream(seed, 500))
    assert splitmix64_stream(5, 10, start=990).tolist() == list(stream(5, 1000))[990:]


def test_tpdf_noise_is_triangular_on_minus_one_to_one():
    d = tpdf_noise(1234, 200_000)
    assert d.dtype == np.float64 and d.min() > -1.0 and d.max() < 1.0
    assert abs(d.mean()) < 0.005 and abs(d.var() - 1 / 6) < 0.003       # var of TPDF on (-1, 1) is 1/6
    q = np.rint(d)
    assert abs(np.mean(q == 0) - 0.75) < 0.005, "|d| < 1/2 three quarters of the time"
    assert np.array_equal(tpdf_noise(1234, 1000), d[:1000])


def test_the_dither_is_a_function_of_the_audio_hash():
    x = 0.3 * noise(2, 50_000, 3)
    seed = dither_seed(pcm_sha256(x))
    assert seed == int(pcm_sha256(x)[:16], 16)
    a, b = to_int16(x, seed), to_int16(x, seed)
    assert a.dtype == np.int16 and a.shape == x.shape and np.array_equal(a, b)
    assert not np.array_equal(a, to_int16(x, seed + 1))
    err = a.astype(np.float64) / 32767 - x
    assert np.abs(err).max() <= 1.5 / 32767, "rint(32767 x + d) stays within 1.5 LSB"
    full = to_int16(np.array([[1.0, -1.0, 0.99999]], dtype=np.float32), 7)
    assert full.min() >= -32768 and full.max() <= 32767


def test_flac_copies_decode_to_the_dithered_samples_and_repeat_byte_for_byte(tmp_path):
    x = 0.3 * noise(2, 44_100, 8)
    p1, p2 = write_flac(tmp_path / "a.flac", x), write_flac(tmp_path / "b.flac", x)
    assert p1.read_bytes() == p2.read_bytes()
    info = sf.info(str(p1))
    assert info.format == "FLAC" and info.subtype == "PCM_16" and info.samplerate == 44100 and info.channels == 2
    got, _ = sf.read(str(p1), dtype="int16", always_2d=True)
    assert np.array_equal(got.T, to_int16(x, dither_seed(pcm_sha256(x))))
    other, _ = sf.read(str(write_flac(tmp_path / "c.flac", x, digest="ab" * 32)), dtype="int16", always_2d=True)
    assert not np.array_equal(other, got), "the digest keys the dither"


def test_the_float_wav_has_no_peak_chunk(tmp_path):
    # Byte identity across a time gap is tests/test_render_basic.py's (write_wav is wavio.write_float).
    x = 0.3 * noise(2, 10_000, 9)
    p1, p2 = write_wav(tmp_path / "1.wav", x), write_wav(tmp_path / "2.wav", x)
    assert p1.read_bytes() == p2.read_bytes() and b"PEAK" not in p1.read_bytes()
    y, sr = read_float(p1)
    assert sr == 44100 and np.array_equal(x, y)
