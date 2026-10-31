"""Per-stem MIDI for FluidSynth 2.6.1 (instrument.md section 3, Timeline; section 4, one process per stem).

The file holds `cycles` copies of one stem's periodic events (an event at sample t plays at t + c N in
cycle c) and ends at cycles N + 0.1 s. PPQ 22050 at a fixed 120 BPM makes one tick one sample.

FluidSynth's file player is not sample-accurate (measured on the laptop with 2.6.1, 2026-10-31, see
docs/COMPILER.md): it renders in 64-sample blocks, and before block b + 1 it sends every event whose tick
is <= cur(b) = round(44.1 * floor(64000 b / 44100)), a whole-millisecond clock, so the event takes effect
from sample 64 (b + 1). With ticks equal to samples the same event lands in different blocks in different
cycles whenever N is not a multiple of 28,224 samples (640 ms), which is every tempo in 60-95 but 60 and
75, and the 2nd and 3rd cycles of a 3-cycle render then differ by up to 0.09 (peak 0.07). `fluid_tick`
instead picks, for each sample t, a tick that this player sends into the block nearest t: the middle of
the tick range (cur(B - 2), cur(B - 1)] for B = round(t / 64), clear of the rounding at either end. Every
event then takes effect at a block boundary within 32 samples of its sample time, the same in every cycle.
"""
from __future__ import annotations

from pathlib import Path

import mido

from .events import CompiledLoop, Stem
from .timeline import BLOCK, SR

PPQ = 22050
TEMPO = 500_000                 # us per beat: 120 BPM, so a tick is a sample at 44.1 kHz
TAIL = SR // 10                 # end of track 0.1 s after the last cycle
RPN_BEND = ((101, 0), (100, 0))         # select RPN 0, pitch-bend sensitivity
RPN_NULL = ((101, 127), (100, 127))
_OFF, _CTRL, _ON = 0, 1, 2              # order of events that share a tick


def player_reach(b: int) -> int:
    """cur(b): the last tick FluidSynth 2.6.1's player sends before rendering block b + 1."""
    msec = 64_000 * b // SR
    return (msec * 441 * 2 + 10) // 20             # round(msec * 44.1), halves up


def fluid_tick(t: int) -> int:
    """A tick FluidSynth 2.6.1 plays from the block boundary nearest sample t (block 1 at the earliest)."""
    block = max(1, (t + BLOCK // 2) // BLOCK)
    b = block - 1
    if b == 0:
        return 0
    return (player_reach(b - 1) + 1 + player_reach(b)) // 2


def stem_events(stem: Stem, n: int, cycles: int = 2,
                fluid: bool = True) -> list[tuple[int, int, mido.Message]]:
    """(tick, order, message) for every event of `cycles` cycles, sorted; fluid=False keeps tick = sample."""
    tick = fluid_tick if fluid else (lambda t: t)
    events = []
    for c in range(cycles):
        base = c * n
        for note in stem.notes:
            on = tick(base + note.on)
            off = max(tick(base + note.off), on + 1)
            if fluid:
                off = max(off, fluid_tick(base + note.on + BLOCK))     # at least one block long
            events.append((on, _ON, mido.Message("note_on", channel=note.channel, note=note.pitch,
                                                 velocity=note.velocity)))
            events.append((off, _OFF, mido.Message("note_off", channel=note.channel, note=note.pitch,
                                                   velocity=0)))
        for p in stem.bend:
            events.append((tick(base + p.at), _CTRL,
                           mido.Message("pitchwheel", channel=p.channel, pitch=p.value)))
        for p in stem.cc11:
            events.append((tick(base + p.at), _CTRL, mido.Message("control_change", channel=p.channel,
                                                                  control=11, value=p.value)))
    events.sort(key=lambda e: (e[0], e[1], e[2].channel, _sort_detail(e[2])))
    return events


def _sort_detail(msg: mido.Message) -> int:
    if msg.type in ("note_on", "note_off"):
        return msg.note
    return msg.control if msg.type == "control_change" else -1


def _setup(stem: Stem) -> list[mido.Message]:
    out = []
    for ch in stem.channels:
        out += [mido.Message("control_change", channel=ch, control=0, value=stem.bank),
                mido.Message("control_change", channel=ch, control=32, value=0),
                mido.Message("program_change", channel=ch, program=stem.program)]
        if stem.bend_range:
            out += [mido.Message("control_change", channel=ch, control=c, value=v) for c, v in RPN_BEND]
            out += [mido.Message("control_change", channel=ch, control=6, value=stem.bend_range),
                    mido.Message("control_change", channel=ch, control=38, value=0)]
            out += [mido.Message("control_change", channel=ch, control=c, value=v) for c, v in RPN_NULL]
    return out


def stem_midi(loop: CompiledLoop, name: str, cycles: int = 2, fluid: bool = True) -> mido.MidiFile:
    """One stem (keys, bass or sax) as a type-0 MIDI file of `cycles` cycles."""
    stem = {s.name: s for s in loop.stems}[name]
    if stem.bank > 127:
        raise ValueError(f"stem {name}: bank {stem.bank} is a drum bank; pitched stems use banks 0-127")
    mid = mido.MidiFile(type=0, ticks_per_beat=PPQ)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    track.extend(_setup(stem))
    now = 0
    for at, _, msg in stem_events(stem, loop.n, cycles, fluid):
        track.append(msg.copy(time=at - now))
        now = at
    end = max(cycles * loop.n + TAIL, now)
    track.append(mido.MetaMessage("end_of_track", time=end - now))
    return mid


def write_stem_midi(loop: CompiledLoop, name: str, path: Path, cycles: int = 2, fluid: bool = True) -> Path:
    stem_midi(loop, name, cycles, fluid).save(str(path))
    return path
