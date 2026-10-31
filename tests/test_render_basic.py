"""Render primitives (docs/design/instrument.md §4):
- FluidSynth is the pinned 2.6.1, and the same MIDI renders to the same bytes in two fresh processes;
- normalize_loop hits -14 ± 0.1 LUFS with true peak at or below -1 dBTP, and keeps the loop length exactly.
Runs wherever FluidSynth is on PATH (the laptop, and the cluster through the tools wrapper). Bit identity is
checked on one machine only: the cluster render is canonical (LESSONS 38)."""
import hashlib
import shutil
from pathlib import Path

import mido
import numpy as np
import pytest

from lofi_rl.render import fluid
from lofi_rl.render.loudness import CEILING_DBTP, lufs, normalize_loop, true_peak_dbtp

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
def test_same_midi_renders_to_identical_bytes_in_fresh_processes(tmp_path):
    midi = _scale_midi(tmp_path / "scale.mid")
    a = fluid.render_midi(SF2, midi, tmp_path / "a.wav")
    b = fluid.render_midi(SF2, midi, tmp_path / "b.wav")
    da, db = a.read_bytes(), b.read_bytes()
    assert len(da) > 44 + 4 * 44100, "about 2 s of float audio expected"
    assert hashlib.sha256(da).hexdigest() == hashlib.sha256(db).hexdigest()


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
