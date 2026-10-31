"""The Phase-0 render chain (compile/stub.py): exact loop length, the swing grid, the seamless two-cycle
render, deterministic drums and the -14 LUFS target. The real compiler replaces it in Phase 1."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from lofi_rl.compile import stub
from lofi_rl.render.loudness import CEILING_DBTP, lufs, true_peak_dbtp
from lofi_rl.render.wavio import pcm_sha256

ROOT = Path(__file__).resolve().parents[1]
SF2 = ROOT / "assets" / "soundfonts" / "generaluser-gs" / "GeneralUser-GS.sf2"
needs_fluidsynth = pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth not on PATH")


def test_loop_length_is_64_aligned_and_tracks_tempo():
    for bpm in (60, 75, 82, 90, 95):
        n = stub.loop_samples(bpm)
        assert n % 64 == 0 and abs(n - 1920 / bpm * 44100) <= 64


def test_swing_delays_only_odd_steps():
    n = stub.loop_samples(80)
    straight = [stub.step_sample(k, n, 0.5) for k in range(128)]
    swung = [stub.step_sample(k, n, 0.55) for k in range(128)]
    assert all(s == t for k, (s, t) in enumerate(zip(swung, straight)) if k % 2 == 0)
    delay = swung[1] - straight[1]
    assert abs(delay - 0.05 * 30 / 80 * 44100) <= 1      # (s - 0.5) * 30 / BPM seconds (sound_sources.md A2)


def test_drums_are_deterministic_and_wrap_at_the_seam():
    n = stub.loop_samples(82)
    hits = [("kick", 0, 1.0), ("snare", n - 100, 1.0), ("hat", n // 2, 0.5)]
    a, b = stub.render_drums(hits, n), stub.render_drums(hits, n)
    assert a.shape == (2, n) and pcm_sha256(a) == pcm_sha256(b)
    assert np.abs(a[0, :2000]).max() > 0.1, "the snare tail past the loop end wraps to the start"


def _hello(n: int):
    notes, beat = [], n // 32
    for bar in range(8):
        root = 62 if (bar // 2) % 2 == 0 else 67                      # Dm9 / G13 vamp, 2 bars each (stub voicing)
        for p in (root + 3, root + 7, root + 10, root + 14):
            notes.append(stub.Note(0, p, bar * 4 * beat, 4 * beat - 200, 70))
        notes.append(stub.Note(1, root - 24, bar * 4 * beat, 2 * beat, 90))
    notes.append(stub.Note(2, 69, 0, 6 * beat, 85))
    notes.append(stub.Note(2, 72, 16 * beat, 6 * beat, 85))
    hits = []
    for bar in range(8):
        o = bar * 16
        hits += [("kick", stub.step_sample(o, n), 1.0), ("kick", stub.step_sample(o + 10, n), 0.8),
                 ("snare", stub.step_sample(o + 4, n), 0.9), ("snare", stub.step_sample(o + 12, n), 0.9)]
        hits += [("hat", stub.step_sample(o + k, n), 0.35 if k % 4 else 0.5) for k in range(0, 16, 2)]
    return notes, {0: 4, 1: 33, 2: 66}, hits


@needs_fluidsynth
def test_hello_loop_renders_exact_length_at_target_loudness_and_repeats_identically():
    n = stub.loop_samples(82)
    notes, programs, hits = _hello(n)
    keys = stub.render_stems(notes, programs, n, SF2)
    drums = stub.render_drums(hits, n)
    y, info = stub.mix_and_normalize([(keys, 1.0), (drums, 0.5)])
    assert y.shape == (2, n)
    assert abs(lufs(y, 44100) + 14) <= 0.1 and true_peak_dbtp(y) <= CEILING_DBTP
    y2, _ = stub.mix_and_normalize([(stub.render_stems(notes, programs, n, SF2), 1.0), (stub.render_drums(hits, n), 0.5)])
    assert pcm_sha256(y) == pcm_sha256(y2)


# ---------------- the Phase-0 adapter: a parsed LoopScript -> stub notes (compile/stub_from_loop.py) ----------------
EXAMPLE = ROOT / "tests" / "fixtures" / "loops" / "example.ls"


def _example_notes(n):
    from lofi_rl.compile.stub_from_loop import notes_from_loop
    from lofi_rl.dsl import parse
    res = parse(EXAMPLE.read_text())
    assert res.gate.ok and not res.repairs
    return notes_from_loop(res.ast, tonic_pc=2, mode="dorian", n=n)      # D dorian, as in instrument.md §1


def test_example_loopscript_maps_to_the_notes_the_design_describes():
    n = stub.loop_samples(82)
    notes, programs, hits = _example_notes(n)
    sax = sorted((nt.start, nt.pitch) for nt in notes if nt.channel == 2)
    assert sax[0] == (0, 57), "bar 1 lands on A3, the 5th of Dm9"
    assert (min(p for _, p in sax), max(p for _, p in sax)) == (50, 62), "sax range D3-D4"
    assert sum(nt.channel == 1 for nt in notes) == 24, "24 bass onsets"
    assert programs == {0: 4, 1: 33, 2: 66} and len(hits) > 100


@needs_fluidsynth
def test_example_loopscript_renders_exact_length_at_target_loudness():
    n = stub.loop_samples(82)
    notes, programs, hits = _example_notes(n)
    y, _ = stub.mix_and_normalize([(stub.render_stems(notes, programs, n, SF2), 1.0), (stub.render_drums(hits, n), 0.5)])
    assert y.shape == (2, n) and abs(lufs(y, 44100) + 14) <= 0.1 and true_peak_dbtp(y) <= CEILING_DBTP
