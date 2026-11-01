"""The one-time 48 -> 44.1 kHz conversion (docs/SHOOTOUTS.md §7.2, CD-36): `to_44k(x, sr)` is
scipy.signal.resample_poly(x, 147, 160, axis=0) in float64 on the whole signal, with scipy's default Kaiser window
(beta 5); `python -m lofi_rl.convert.resample in.wav out.wav` writes it as a float32 WAV with no PEAK chunk. The kit
importer (§7.4) uses it on the Virtuosity one-shots, and the same CLI later converts the rain bed."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from helpers_compile import ROOT
from lofi_rl.convert.resample import ResampleError, main, to_44k
from lofi_rl.render import bed


def ceil_147(n: int) -> int:
    return -(-147 * n // 160)


def test_the_resampled_length_is_ceil_147n_over_160():
    for n in (1, 2, 159, 160, 161, 1000, 48000, 48001, 99991):
        assert to_44k(np.zeros(n), 48000).shape == (ceil_147(n),), n
    assert to_44k(np.zeros((48001, 2)), 48000).shape == (ceil_147(48001), 2)


def test_it_is_resample_poly_147_160_in_float64_on_the_whole_signal():
    x = np.random.default_rng(7).standard_normal((4800, 2)).astype(np.float32) * 0.1
    y = to_44k(x, 48000)
    assert y.dtype == np.float64
    assert np.array_equal(y, resample_poly(x.astype(np.float64), 147, 160, axis=0))


def test_a_1khz_sine_keeps_its_frequency_and_its_level():
    t = np.arange(48000)
    y = to_44k(0.5 * np.sin(2 * np.pi * 1000 * t / 48000), 48000)
    assert y.shape == (44100,)
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    peak_hz = np.argmax(spectrum) * 44100 / len(y)               # bins of 1 Hz
    assert abs(peak_hz - 1000) <= 44100 / len(y)
    mid = y[2000:-2000]                                           # clear of the filter's edges
    rms_db = 20 * np.log10(np.sqrt(np.mean(mid ** 2)) / (0.5 / np.sqrt(2)))
    assert abs(rms_db) < 0.01


def test_only_48khz_input_is_taken(tmp_path):
    for sr in (44100, 96000, 32000):
        with pytest.raises(ResampleError, match=f"{sr} Hz"):
            to_44k(np.zeros(100), sr)
    with pytest.raises(ResampleError, match="samples"):
        to_44k(np.zeros((0,)), 48000)
    src = tmp_path / "already.wav"
    sf.write(str(src), np.zeros((441, 2)), 44100, subtype="PCM_24")
    assert main([str(src), str(tmp_path / "out.wav")]) == 2
    assert not (tmp_path / "out.wav").exists()


def _rain_like(path, seconds=2.0, sr=48000):
    """A stereo 48 kHz 24-bit noise file standing in for the rain recording (the real one is Shresth's download)."""
    x = np.random.default_rng(235827).standard_normal((int(seconds * sr), 2)) * 0.05
    sf.write(str(path), x, sr, subtype="PCM_24")
    return path


def _chunks(data: bytes) -> list[bytes]:
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    out, pos = [], 12
    while pos < len(data):
        cid, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        out.append(cid)
        pos += 8 + size + (size & 1)
    return out


def test_the_cli_writes_the_same_bytes_twice_and_in_a_fresh_process(tmp_path, capsys):
    src = _rain_like(tmp_path / "rain48.wav")
    outs = [tmp_path / f"out{i}.wav" for i in range(3)]
    assert main([str(src), str(outs[0])]) == 0
    assert main([str(src), str(outs[1])]) == 0
    printed = capsys.readouterr().out
    env = dict(os.environ, PYTHONHASHSEED="4242", PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.run([sys.executable, "-m", "lofi_rl.convert.resample", str(src), str(outs[2])],
                          capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    data = [p.read_bytes() for p in outs]
    assert data[0] == data[1] == data[2]
    sha = hashlib.sha256(data[0]).hexdigest()
    assert sha in printed and sha in proc.stdout
    info = sf.info(str(outs[0]))
    assert (info.samplerate, info.channels, info.subtype, info.frames) == (44100, 2, "FLOAT", ceil_147(96000))
    assert b"PEAK" not in _chunks(data[0])
    x, _ = sf.read(str(src), dtype="float64", always_2d=True)
    y, _ = sf.read(str(outs[0]), dtype="float32", always_2d=True)
    assert np.array_equal(y, to_44k(x, 48000).astype(np.float32))


def test_the_bed_loader_takes_the_converted_file_and_refuses_the_original(tmp_path):
    src = _rain_like(tmp_path / "rain48.wav")
    with pytest.raises(bed.BedError, match="48000 Hz"):
        bed.read_excerpt(src, 0, 1000)
    out = tmp_path / "rain44.wav"
    assert main([str(src), str(out)]) == 0
    e = bed.read_excerpt(out, 22050, 44100)
    assert e.shape == (2, 44100) and e.dtype == np.float32 and np.isfinite(e).all() and np.abs(e).max() > 0
