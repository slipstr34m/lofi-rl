"""Phase-0 stub of the render chain, used by the smoke test until the real compiler lands (Phase 1,
docs/design/instrument.md §3). It produces the real timing grid, the two-cycle seamless render and the
loudness chain, but with plain voicings and no expression.
- Timeline: N = 64 * round(1,323,000 / BPM) samples per 8-bar loop, so FluidSynth's 64-sample blocks align.
  MIDI is written at PPQ 22050 with a fixed 120 BPM tempo, so 1 tick = 1 sample at 44.1 kHz.
- Pitched stems render through FluidSynth across two cycles, and the second cycle is kept, so tails wrap.
- Drums are deterministic synthesized one-shots; real kits come from the Phase 1 shootout.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np

from lofi_rl.render import fluid
from lofi_rl.render.loudness import LoudnessInfo, normalize_loop
from lofi_rl.render.wavio import read_float

SR = 44100
PPQ = 22050          # with tempo 500000 us/beat (120 BPM), one tick is one sample at 44.1 kHz
STEPS = 128          # 8 bars x 16 sixteenths


def loop_samples(bpm: float) -> int:
    return 64 * int(1_323_000 / bpm + 0.5)


def step_sample(k: int, n: int, swing: float = 0.55) -> int:
    """Sample index of 16th-step k in [0, 128) with MPC swing (odd steps are delayed)."""
    return int(round(k * n / STEPS + (k % 2) * (swing - 0.5) * n / 64))


@dataclass(frozen=True)
class Note:
    channel: int
    pitch: int
    start: int       # sample index within one cycle
    dur: int         # samples
    vel: int


def write_midi(notes: list[Note], programs: dict[int, int], n: int, path: Path, cycles: int = 2) -> Path:
    events: list[tuple[int, int, mido.Message]] = []
    for c in range(cycles):
        for nt in notes:
            on = c * n + nt.start
            events.append((on, 1, mido.Message("note_on", channel=nt.channel, note=nt.pitch, velocity=nt.vel)))
            events.append((on + max(1, nt.dur), 0, mido.Message("note_off", channel=nt.channel, note=nt.pitch, velocity=0)))
    events.sort(key=lambda e: (e[0], e[1], e[2].channel, e[2].note))
    mid = mido.MidiFile(ticks_per_beat=PPQ)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    for ch, prog in sorted(programs.items()):
        tr.append(mido.Message("program_change", channel=ch, program=prog, time=0))
    t = 0
    for at, _, msg in events:
        tr.append(msg.copy(time=at - t))
        t = at
    tr.append(mido.MetaMessage("end_of_track", time=cycles * n + SR // 10 - t))
    mid.save(path)
    return path


def render_stems(notes: list[Note], programs: dict[int, int], n: int, sf2: Path) -> np.ndarray:
    """Render pitched notes through FluidSynth over two cycles and keep the second (tails wrap)."""
    with tempfile.TemporaryDirectory() as d:
        midi = write_midi(notes, programs, n, Path(d) / "loop.mid")
        x, sr = read_float(fluid.render_midi(sf2, midi, Path(d) / "loop.wav"))
    assert sr == SR
    if x.shape[1] < 2 * n:
        x = np.pad(x, ((0, 0), (0, 2 * n - x.shape[1])))
    return x[:, n:2 * n]


def _kick(sr: int = SR) -> np.ndarray:
    t = np.arange(int(0.25 * sr)) / sr
    f = 50 + 70 * np.exp(-t * 40)
    return (np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 14)).astype(np.float32)


def _noise(m: int, seed: int) -> np.ndarray:
    # an integer-hash noise generator: the same samples on every numpy build and CPU (no RNG streams, no SIMD sin)
    i = np.arange(m, dtype=np.uint64) + np.uint64((seed * 0x9E3779B97F4A7C15) % 2**64)
    z = (i ^ (i >> np.uint64(31))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return ((z >> np.uint64(11)).astype(np.float64) / 2.0 ** 53 * 2 - 1).astype(np.float32)


def _snare(sr: int = SR) -> np.ndarray:
    t = np.arange(int(0.2 * sr)) / sr
    return (0.6 * _noise(len(t), 2) * np.exp(-t * 25) + 0.4 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)).astype(np.float32)


def _hat(sr: int = SR) -> np.ndarray:
    t = np.arange(int(0.05 * sr)) / sr
    x = _noise(len(t), 3)
    x = np.diff(x, prepend=0.0)          # crude high-pass
    return (0.5 * x * np.exp(-t * 90)).astype(np.float32)


def render_drums(hits: list[tuple[str, int, float]], n: int) -> np.ndarray:
    """hits: (lane, start sample, gain) with lane in kick/snare/hat. Wraps around the loop end (seamless)."""
    shots = {"kick": _kick(), "snare": _snare(), "hat": _hat()}
    out = np.zeros(n + SR, dtype=np.float64)
    for lane, start, gain in sorted(hits):
        s = shots[lane]
        out[start:start + len(s)] += gain * s
    y = out[:n].copy()
    y[: SR] += out[n:n + SR]             # tails past the loop end come back at the start
    return np.stack([y, y]).astype(np.float32)


def mix_and_normalize(stems: list[tuple[np.ndarray, float]]) -> tuple[np.ndarray, LoudnessInfo]:
    x = sum(g * s for s, g in stems)
    return normalize_loop(np.asarray(x, dtype=np.float32), SR)
