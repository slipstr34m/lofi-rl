"""Per-stem MIDI for FluidSynth (compile/midi.py): the dispatch model FluidSynth 2.6.1's file player was
measured to follow, the file layout, and seamless cycles in a real render."""
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from helpers_compile import compiled, dev_palette
from lofi_rl.compile.midi import PPQ, TEMPO, fluid_tick, player_reach, stem_midi, write_stem_midi

needs_fluidsynth = pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth not on PATH")

# Measured on the laptop, FluidSynth 2.6.1, 2026-10-31: one note-on per file at this tick (PPQ 22050,
# 120 BPM), and the 64-sample block in which the output first goes non-zero (first sample 64 b + 1).
MEASURED = {0: 1, 7: 2, 14: 2, 21: 2, 28: 2, 35: 2, 42: 2, 49: 3, 56: 3, 63: 3, 70: 3, 77: 3, 84: 3,
            91: 4, 98: 4, 133: 4, 175: 4, 182: 5, 189: 5, 196: 5, 1000: 17, 44100: 691, 132317: 2069,
            1032576: 16136, 1032626: 16137, 2065152: 32270, 2065202: 32270}


def sounding_block(tick: int) -> int:
    """The model: callback b sends ticks <= player_reach(b); what it sends sounds from block b + 1."""
    b = max(0, tick // 64 - 2)
    while player_reach(b) < tick:
        b += 1
    while b > 0 and player_reach(b - 1) >= tick:
        b -= 1
    return b + 1


def test_the_dispatch_model_reproduces_the_measurements():
    assert {t: sounding_block(t) for t in MEASURED} == MEASURED


def test_fluid_tick_sounds_in_the_block_nearest_each_sample():
    for t in list(range(0, 5000)) + list(range(0, 4_300_000, 997)):
        assert sounding_block(fluid_tick(t)) == max(1, (t + 32) // 64), t
    ticks = [fluid_tick(t) for t in range(0, 300_000, 7)]
    assert ticks == sorted(ticks)


def _absolute(mid):
    now, out = 0, []
    for msg in mid.tracks[0]:
        now += msg.time
        out.append((now, msg))
    return out


@pytest.mark.parametrize("stem", ["keys", "bass", "sax"])
def test_file_layout(stem):
    c = compiled("example.ls")
    s = {x.name: x for x in c.stems}[stem]
    mid = stem_midi(c, stem, cycles=2)
    assert mid.type == 0 and mid.ticks_per_beat == PPQ == 22050
    events = _absolute(mid)
    assert events[0][1].type == "set_tempo" and events[0][1].tempo == TEMPO == 500_000
    first_note = next(i for i, (_, m) in enumerate(events) if m.type == "note_on")
    head = [m for _, m in events[:first_note]]
    for ch in s.channels:
        setup = [(m.type, getattr(m, "control", None), getattr(m, "value", getattr(m, "program", None)))
                 for m in head if getattr(m, "channel", None) == ch]
        assert setup[:3] == [("control_change", 0, s.bank), ("control_change", 32, 0),
                             ("program_change", None, s.program)]
    ons = [(t, m) for t, m in events if m.type == "note_on" and m.velocity > 0]
    offs = [(t, m) for t, m in events if m.type == "note_off"]
    assert len(ons) == len(offs) == 2 * len(s.notes)
    open_notes = {}
    for t, m in events:
        if m.type == "note_on":
            assert (m.channel, m.note) not in open_notes, "a note is never re-struck while it sounds"
            open_notes[(m.channel, m.note)] = t
        elif m.type == "note_off":
            assert open_notes.pop((m.channel, m.note)) < t
    assert not open_notes
    assert events[-1][1].type == "end_of_track" and events[-1][0] >= 2 * c.n + 4410
    for (t, m), note in zip(ons[: len(s.notes)], sorted(s.notes, key=lambda x: (x.on, x.channel, x.pitch))):
        assert sounding_block(t) == max(1, (note.on + 32) // 64)


def test_plain_ticks_are_samples():
    c = compiled("example.ls")
    events = _absolute(stem_midi(c, "sax", cycles=2, fluid=False))
    ons = sorted(t for t, m in events if m.type == "note_on")
    assert ons == sorted([x.on for x in c.sax.notes] + [x.on + c.n for x in c.sax.notes])


def _cycles(c, stem, cycles, fluid):
    from lofi_rl.render import fluid as fs
    from lofi_rl.render.wavio import read_float
    with tempfile.TemporaryDirectory() as d:
        mid = write_stem_midi(c, stem, Path(d) / "s.mid", cycles=cycles, fluid=fluid)
        x, sr = read_float(fs.render_midi(dev_palette().source_path("gu"), mid, Path(d) / "s.wav"))
    assert sr == 44100
    x = np.pad(x, ((0, 0), (0, max(0, cycles * c.n - x.shape[1]))))
    return [x[:, i * c.n:(i + 1) * c.n] for i in range(cycles)]


@pytest.fixture(scope="module")
def three_cycles():
    """(stem, fluid ticks) -> the 3 cycles of example.ls's FluidSynth render. The renders are independent
    processes, so they run side by side: that saves wall time and changes no sample."""
    if shutil.which("fluidsynth") is None:
        pytest.skip("fluidsynth not on PATH")
    c = compiled("example.ls")
    jobs = [("sax", True), ("keys", True), ("sax", False)]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = [ex.submit(_cycles, c, stem, 3, fluid) for stem, fluid in jobs]
        return {job: f.result() for job, f in zip(jobs, futures)}


@needs_fluidsynth
@pytest.mark.parametrize("stem", ["sax", "keys"])
def test_rendered_cycles_repeat_bit_for_bit(stem, three_cycles):
    # The renderer keeps cycle 2 of a 2-cycle render (instrument.md section 3, Seamless); measured on the
    # laptop, cycles 2 and 3 of a 3-cycle render are identical for every stem of four fixtures.
    cyc = three_cycles[(stem, True)]
    assert np.abs(cyc[1]).max() > 0.01
    assert np.abs(cyc[1] - cyc[2]).max() <= 1e-6


@needs_fluidsynth
def test_plain_ticks_would_not_repeat(three_cycles):
    cyc = three_cycles[("sax", False)]
    assert np.abs(cyc[1] - cyc[2]).max() > 1e-3, "why midi.py maps samples to dispatch-safe ticks"
