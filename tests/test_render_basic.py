"""Render primitives (docs/design/instrument.md §4):
- FluidSynth is the pinned 2.6.1, and the same MIDI renders to the same PCM in two fresh processes;
- normalize_loop hits -14 ± 0.1 LUFS with true peak at or below -1 dBTP, and keeps the loop length exactly.
Runs wherever FluidSynth is on PATH (the laptop, and the cluster through the tools wrapper). Bit identity is
checked on one machine only: the cluster render is canonical (LESSONS 38)."""
import hashlib
import shutil
import time
from pathlib import Path

import mido
import numpy as np
import pytest

from lofi_rl.render import fluid
from lofi_rl.render.loudness import CEILING_DBTP, lufs, normalize_loop, true_peak_dbtp
from lofi_rl.render.wavio import pcm_sha256, read_float, write_float

ROOT = Path(__file__).resolve().parents[1]
SF2 = ROOT / "assets" / "soundfonts" / "generaluser-gs" / "GeneralUser-GS.sf2"
needs_fluidsynth = pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth not on PATH")


def _scale_midi(path: Path, program: int = 0) -> Path:
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=500000))
    tr.append(mido.Message("program_change", program=program, channel=0, time=0))
    for note in (60, 62, 64, 65, 67, 69, 71, 72):
        tr.append(mido.Message("note_on", note=note, velocity=90, channel=0, time=0))
        tr.append(mido.Message("note_off", note=note, velocity=0, channel=0, time=240))
    mid.save(path)
    return path


@needs_fluidsynth
def test_fluidsynth_is_the_pinned_version():
    assert fluid.version() == fluid.PINNED == "2.6.1"


@needs_fluidsynth
def test_same_midi_renders_to_identical_pcm_in_fresh_processes(tmp_path):
    # Identity is the PCM: libsndfile stamps the wall-clock time into the PEAK chunk of float WAVs, so
    # file bytes differ in one header byte between renders a second apart (found 2026-10-30).
    midi = _scale_midi(tmp_path / "scale.mid")
    a, sr_a = read_float(fluid.render_midi(SF2, midi, tmp_path / "a.wav"))
    time.sleep(1.1)
    b, sr_b = read_float(fluid.render_midi(SF2, midi, tmp_path / "b.wav"))
    assert sr_a == sr_b == 44100 and a.shape[0] == 2 and a.shape[1] > 2 * 44100, "about 2+ s of stereo audio expected"
    assert pcm_sha256(a) == pcm_sha256(b)


def test_our_wav_writer_is_byte_deterministic(tmp_path):
    x = _loop()
    p1, p2 = write_float(tmp_path / "1.wav", x, 44100), write_float(tmp_path / "2.wav", x, 44100)
    time.sleep(1.1)
    p3 = write_float(tmp_path / "3.wav", x, 44100)
    assert hashlib.sha256(p1.read_bytes()).digest() == hashlib.sha256(p2.read_bytes()).digest() == hashlib.sha256(p3.read_bytes()).digest()
    y, sr = read_float(p1)
    assert sr == 44100 and np.array_equal(x, y) and pcm_sha256(x) == pcm_sha256(y)


def _loop(sr: int = 44100, bpm: float = 82.0) -> np.ndarray:
    n = 64 * int(1_323_000 / bpm + 0.5) // 8          # 1 bar here; the compiler uses all 8 (instrument.md §3)
    t = np.arange(n) / sr
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 261.63, 329.63, 392.0)) * 0.05
    kick = np.zeros(n)
    for k in range(4):
        i = k * n // 4
        m = min(n - i, 4000)
        kick[i:i + m] += np.sin(2 * np.pi * 55 * t[:m]) * np.exp(-t[:m] * 30)
    x = (chord + 0.6 * kick).astype(np.float32)
    return np.stack([x, x * 0.98])


def test_normalize_loop_hits_target_with_exact_length_and_true_peak():
    x = _loop()
    y, info = normalize_loop(x, 44100)
    assert y.shape == x.shape and y.dtype == np.float32
    assert abs(lufs(y, 44100) + 14.0) <= 0.1 and abs(info.lufs + 14.0) <= 0.1
    assert true_peak_dbtp(y) <= CEILING_DBTP
    assert 1 <= info.iterations <= 4


def test_normalize_loop_is_deterministic():
    x = _loop()
    a, _ = normalize_loop(x, 44100)
    b, _ = normalize_loop(x, 44100)
    assert np.array_equal(a, b)
