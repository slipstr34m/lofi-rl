"""The compiler's output: a symbolic, sample-exact, periodic loop (docs/COMPILER.md, "The interface").

Every time is a sample index at 44.1 kHz inside one cycle of n samples, and every event repeats in every
cycle: an onset `on` plays at on + c n for each cycle c. Onsets and control points lie in [0, n); a
note's `off` may pass n, and then the note is held into the next cycle. Nothing here is audio.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from lofi_rl.dsl.repair import Repair


@dataclass(frozen=True)
class Note:
    pitch: int                  # sounding MIDI note number
    on: int                     # onset sample in [0, n)
    off: int                    # note-off sample, on < off; may pass n (wraps into the next cycle)
    velocity: int               # MIDI velocity 1..127
    channel: int                # MIDI channel: keys 0, bass 1, sax 2 and 3 in turn (4: B's last, odd lines)
    step: int                   # grid step 0..127 the note starts on


@dataclass(frozen=True)
class Control:
    at: int                     # sample in [0, n) where the value is written, in every cycle
    channel: int
    value: int                  # pitch bend -8192..8191 (0 = centre), or CC11 0..127


@dataclass(frozen=True)
class Stem:
    """One FluidSynth render: a SoundFont preset, its notes and its controller curves."""

    name: str                   # keys | bass | sax
    src: str                    # palette source id of the SoundFont
    bank: int
    program: int
    channels: tuple[int, ...]   # every channel used; each gets bank, program (and bend range) at t = 0
    notes: tuple[Note, ...]     # sorted by (on, channel, pitch)
    bend: tuple[Control, ...] = ()   # pitch-bend points, sorted by (at, channel)
    cc11: tuple[Control, ...] = ()   # expression points, sorted by (at, channel)
    bend_range: int = 0         # semitones, set by RPN 0 at t = 0 on every channel; 0 = none


@dataclass(frozen=True)
class DrumHit:
    lane: str                   # kick | snare | rim | hat | ohat | shaker
    level: str                  # X | x | g, as written
    layer: str                  # hard | med | soft: the kit layer played, after the nearest-layer fallback
    on: int                     # onset sample in [0, n)
    gain: float                 # linear: drum_gain[level] x metric offset (hat, shaker) x humanize
    rr: int                     # round-robin index into the kit's one-shots for (lane, layer)
    step: int                   # grid step 0..127
    choke_at: int | None = None  # sample (may pass n) where a choking lane's next hit cuts this one


@dataclass(frozen=True)
class CompiledLoop:
    n: int                      # samples per cycle, a multiple of 64
    bpm: int                    # the brief's tempo
    bpm_eff: float              # 84,672,000 / n, the tempo actually played
    tonic: int                  # pitch class, C = 0
    mode: str                   # aeolian | dorian | ionian
    palette_id: str
    palette_sha256: str
    kit: str                    # palette kit name (after aliases)
    keys_name: str              # palette keys name (after aliases)
    sax_name: str               # palette sax that plays (the palette's lead)
    voicings: tuple[tuple[int, ...], tuple[int, ...]]   # keys voicing of vamp chord 1 and chord 2
    bass_pitches: tuple[int, int]                       # bass pitch of vamp chord 1 and chord 2
    keys: Stem
    bass: Stem
    sax: Stem
    drums: tuple[DrumHit, ...]  # sorted by (on, lane order, step)
    seed: int                   # humanize seed: SHA256(canonical text)[:8]
    repairs: tuple[Repair, ...] = ()   # compiler repairs (sax octave folds), penalized like the parser's

    @property
    def stems(self) -> tuple[Stem, Stem, Stem]:
        return (self.keys, self.bass, self.sax)

    def to_json(self) -> str:
        """Canonical JSON (sorted keys, no spaces): the basis of `digest`."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """sha256 of the canonical JSON: equal digests mean identical compiled loops."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
