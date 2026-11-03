"""The numpy one-shot drum sampler and its one-shot builder (instrument.md §3 Drum sampler, §4 Assets;
docs/COMPILER.md §3.9 and §4.4).

Sampler (`mix_hits`):
- Every DrumHit plays palette.kits[kit].lanes[lane].layers[layer][rr] from sample `on`, scaled by the
  hit's gain and the lane's dB, with a constant-power pan. The compiler has already chosen the velocity
  layer (X hard, x med, g soft, else the nearest non-empty layer) and the round-robin index.
- A lane's dB is its palette `db` plus the kit's `trim_db` (docs/SHOOTOUTS.md §7.1), added in `kit_shots`, so the
  trim scales the kit's whole drum stem: the float sum the draft palette wrote into its lanes (CD-51). A trim of 0
  leaves every `db` as written.
- A hit with `choke_at` (the open hat, cut by the next closed hat) fades out linearly over 8 ms from that
  sample, which may lie in the next cycle.
- The loop is periodic, so everything past n folds back onto [0, n): the tails of the last hits play at
  the start of the cycle.
- Hits are summed in float64 in the compiler's order (on, lane, step), whatever order they arrive in.

One-shots are mono, a stereo source downmixed to (L + R) / 2, so the palette's lane pan alone places them.
The pan law gL = sqrt((1 - pan) / 2), gR = sqrt((1 + pan) / 2) is constant power and exact at the centre
and the edges, and it uses only correctly rounded operations, so it gives the same bits on every CPU.

Builder (`kit_shots`):
- A SoundFont one-shot {src, bank, prog, note, vel} is rendered once by FluidSynth (a fresh process with
  render/fluid.py's flags, in an empty home) and cached in $LOFI_CACHE (default ~/.cache/lofi-rl). The cache
  key is the sha256 of the source sha256 and every render setting, the builder version and the machine
  included. A cached one-shot is used only if its pcm_sha256 still matches the one recorded when it was
  built; otherwise it is rebuilt. A preset the SoundFont lacks raises fluid.FluidSynthError.
- A file one-shot {src, file} is read from a directory, zip or tar source.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from lofi_rl.compile.events import CompiledLoop, DrumHit
from lofi_rl.compile.midi import PPQ, TEMPO
from lofi_rl.compile.timeline import ms, rnd
from lofi_rl.dsl.vocab import LANES
from lofi_rl.palette import LAYERS, ROOT, OneShot, Palette

from . import fluid
from .machine import fluidsynth_version, machine_key
from .wavio import pcm_sha256, read_float, write_float

log = logging.getLogger(__name__)

SR = 44100
CHOKE_MS = 8                    # instrument.md §3: the open hat is choked "with an 8 ms fade"
CHOKE = rnd(ms(CHOKE_MS))       # 353 samples (352.8, rounded half up like every sample position)
DRUM_CHANNEL = 9                # the GM percussion channel, where FluidSynth plays bank 128
ONSET = 64                      # a note-on at tick 0 sounds from sample 64, block 1 (docs/COMPILER.md §4.3)
HOLD = 4 * SR                   # note-off 4 s after the onset; the GU open hat renders the same with 3 s or 10 s
TAIL = SR // 10                 # end of track 0.1 s after the note-off; FluidSynth renders on until silence
# Bump whenever the builder's output can change for the same settings. 2: FluidSynth runs in an empty home
# (render/fluid.py), so no entry built while a ~/.fluidsynth may have shaped it is ever reused.
BUILDER = "fluid-note/2"
Shots = Mapping[tuple[str, str], Sequence[np.ndarray]]      # (lane, layer) -> mono one-shots, by rr


class SamplerError(ValueError):
    pass


# --------------------------------------------------------------------------------------------- sampler
def pan_gains(pan: float) -> tuple[float, float]:
    """(left, right) gains of a constant-power pan, pan in -1 (left) .. +1 (right)."""
    if not -1.0 <= pan <= 1.0:
        raise SamplerError(f"pan {pan} is outside -1..1")
    return math.sqrt((1.0 - pan) / 2.0), math.sqrt((1.0 + pan) / 2.0)


def choke(shot: np.ndarray, at: int) -> np.ndarray:
    """`shot` cut `at` samples after its onset: unchanged before, a linear fade over CHOKE samples, then gone."""
    if at <= 0:
        raise SamplerError("a choke must come after the onset")
    if at >= len(shot):
        return shot
    end = min(len(shot), at + CHOKE)
    out = shot[:end].astype(np.float64)
    out[at:end] *= (CHOKE - np.arange(end - at, dtype=np.float64)) / CHOKE
    return out


def mix_hits(hits: Sequence[DrumHit], shots: Shots, lanes: Mapping[str, tuple[float, float]],
             n: int) -> np.ndarray:
    """One periodic cycle of the drum stem: (2, n) float32. `lanes` maps a lane to its (pan, dB)."""
    order = sorted(hits, key=lambda h: (h.on, LANES.index(h.lane), h.step))
    played, reach = [], n
    for h in order:
        if not 0 <= h.on < n:
            raise SamplerError(f"{h.lane} hit at {h.on} is outside the cycle [0, {n})")
        layer = shots.get((h.lane, h.layer), ())
        if not 0 <= h.rr < len(layer):
            raise SamplerError(f"{h.lane} {h.layer}: no one-shot {h.rr} (the layer has {len(layer)})")
        x = np.asarray(layer[h.rr], dtype=np.float64)
        if h.choke_at is not None:
            x = choke(x, h.choke_at - h.on)
        played.append((h, x))
        reach = max(reach, h.on + len(x))
    buf = np.zeros((2, reach), dtype=np.float64)
    for h, x in played:
        pan, db = lanes[h.lane]
        left, right = pan_gains(pan)
        g = h.gain * 10.0 ** (db / 20.0)
        end = h.on + len(x)
        buf[0, h.on:end] += (g * left) * x
        buf[1, h.on:end] += (g * right) * x
    out = buf[:, :n].copy()
    for start in range(n, reach, n):              # tails past the cycle come back at its start
        tail = buf[:, start:start + n]
        out[:, :tail.shape[1]] += tail
    return out.astype(np.float32)


def render_drums(compiled: CompiledLoop, kit: KitShots) -> np.ndarray:
    """The compiled loop's drum hits through its kit's one-shots: one cycle, (2, n) float32."""
    return mix_hits(compiled.drums, kit.shots, kit.lanes, compiled.n)


# --------------------------------------------------------------------------------------------- one-shots
@dataclass(frozen=True)
class KitShots:
    """The mono one-shots of one kit by (lane, layer), each lane's (pan, dB) with the kit's trim_db in the dB, and
    each one-shot's identity."""

    kit: str
    shots: Shots
    lanes: Mapping[str, tuple[float, float]]
    pcm: Mapping[tuple[str, str], tuple[str, ...]]     # pcm_sha256 of each one-shot as built or read


def cache_root(cache: Path | str | None = None) -> Path:
    """The cache directory: `cache` if given, else $LOFI_CACHE, else ~/.cache/lofi-rl."""
    base = cache if cache is not None else (os.environ.get("LOFI_CACHE") or "~/.cache/lofi-rl")
    return Path(os.path.expandvars(os.path.expanduser(str(base))))


def channel_of(shot: OneShot) -> int:
    return DRUM_CHANNEL if shot.bank == 128 else 0


def note_settings(shot: OneShot, sf2_sha256: str) -> dict:
    """Everything a SoundFont one-shot's samples depend on: its cache key is the sha256 of this."""
    return {"builder": BUILDER, "sf2_sha256": sf2_sha256, "bank": shot.bank, "prog": shot.prog,
            "note": shot.note, "vel": shot.vel, "channel": channel_of(shot), "onset": ONSET, "hold": HOLD,
            "tail": TAIL, "sr": SR, "gain": fluid.GAIN, "fluidsynth": fluidsynth_version(),
            "machine": machine_key()}


def settings_key(settings: Mapping) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def note_midi(shot: OneShot, path: Path) -> Path:
    """One note at tick 0 (1 tick = 1 sample), released after HOLD. Bank 128 plays on the drum channel,
    where FluidSynth ignores bank select and takes the program from bank 128; other banks use CC0."""
    ch = channel_of(shot)
    mid = mido.MidiFile(type=0, ticks_per_beat=PPQ)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    if ch != DRUM_CHANNEL:
        track.append(mido.Message("control_change", channel=ch, control=0, value=shot.bank))
        track.append(mido.Message("control_change", channel=ch, control=32, value=0))
    track.append(mido.Message("program_change", channel=ch, program=shot.prog))
    track.append(mido.Message("note_on", channel=ch, note=shot.note, velocity=shot.vel))
    track.append(mido.Message("note_off", channel=ch, note=shot.note, velocity=0, time=HOLD))
    track.append(mido.MetaMessage("end_of_track", time=TAIL))
    mid.save(str(path))
    return path


def render_note(shot: OneShot, sf2: Path) -> np.ndarray:
    """FluidSynth's render of one note from its onset block to its last non-zero sample: (2, L) float32."""
    with tempfile.TemporaryDirectory() as d:
        x, sr = read_float(fluid.render_midi(sf2, note_midi(shot, Path(d) / "shot.mid"), Path(d) / "shot.wav"))
    if sr != SR:
        raise SamplerError(f"FluidSynth rendered at {sr} Hz, not {SR}")
    if x.shape[1] <= ONSET or np.any(x[:, :ONSET]):
        raise SamplerError("sound before the onset block: FluidSynth's event dispatch is not the measured one")
    x = x[:, ONSET:]
    live = np.flatnonzero(np.any(x != 0, axis=0))
    if live.size == 0:
        raise SamplerError(f"bank {shot.bank} prog {shot.prog} note {shot.note} vel {shot.vel} renders silent")
    return np.ascontiguousarray(x[:, :live[-1] + 1])


def _atomic(path: Path, write: Callable[[Path], object]) -> None:
    """Write through a temporary file in the same directory, then rename, so readers never see half a file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _cached(wav: Path, meta: Path) -> tuple[np.ndarray, str] | None:
    """A cache entry whose samples still hash to the recorded pcm_sha256, else None."""
    if not (wav.is_file() and meta.is_file()):
        return None
    try:
        want = json.loads(meta.read_text(encoding="utf-8"))["pcm_sha256"]
        x, sr = read_float(wav)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return None
    got = pcm_sha256(x)
    if sr != SR or got != want:
        log.warning("one-shot cache entry %s does not match its pcm_sha256; rebuilding it", wav.name)
        return None
    return x, got


def note_shot(shot: OneShot, palette: Palette, *, cache: Path | str | None = None,
              root: Path = ROOT) -> tuple[np.ndarray, str]:
    """A SoundFont one-shot from the cache, or rendered and cached: ((2, L) float32, pcm_sha256)."""
    settings = note_settings(shot, palette.sources[shot.src].sha256)
    key = settings_key(settings)
    folder = cache_root(cache) / "oneshots"
    wav, meta = folder / f"{key}.wav", folder / f"{key}.json"
    hit = _cached(wav, meta)
    if hit is not None:
        return hit
    x = render_note(shot, palette.source_path(shot.src, root))
    digest = pcm_sha256(x)
    folder.mkdir(parents=True, exist_ok=True)
    _atomic(wav, lambda p: write_float(p, x, SR))
    record = dict(settings, pcm_sha256=digest, samples=int(x.shape[1]))
    _atomic(meta, lambda p: p.write_text(json.dumps(record, sort_keys=True, indent=1) + "\n", encoding="utf-8"))
    return x, digest


def _member(container: Path, member: str) -> bytes:
    if container.is_dir():
        path = (container / member).resolve()
        if container.resolve() not in path.parents:
            raise SamplerError(f"one-shot {member!r} lies outside its source {container}")
        return path.read_bytes()
    if zipfile.is_zipfile(container):
        with zipfile.ZipFile(container) as z:
            return z.read(member)
    if tarfile.is_tarfile(container):
        with tarfile.open(container) as t:
            f = t.extractfile(member)
            if f is None:
                raise SamplerError(f"{member!r} in {container} is not a file")
            return f.read()
    raise SamplerError(f"source {container} is not a directory, zip or tar archive")


def file_shot(shot: OneShot, palette: Palette, *, root: Path = ROOT) -> tuple[np.ndarray, str]:
    """A one-shot read from a file inside its source: ((channels, L) float32, pcm_sha256)."""
    data = _member(palette.source_path(shot.src, root), shot.file or "")
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if sr != SR:
        raise SamplerError(f"one-shot {shot.file} is {sr} Hz; convert it to {SR} Hz once (instrument.md §4)")
    x = np.ascontiguousarray(x.T)
    return x, pcm_sha256(x)


def load_shot(shot: OneShot, palette: Palette, *, cache: Path | str | None = None,
              root: Path = ROOT) -> tuple[np.ndarray, str]:
    if shot.file is not None:
        return file_shot(shot, palette, root=root)
    return note_shot(shot, palette, cache=cache, root=root)


def mono(x: np.ndarray) -> np.ndarray:
    """A one-shot as float64 mono: one channel as is, two as (L + R) / 2."""
    if x.ndim != 2 or x.shape[0] not in (1, 2) or x.shape[1] == 0:
        raise SamplerError(f"a one-shot must be 1 or 2 channels of audio, got shape {x.shape}")
    x = x.astype(np.float64)
    return x[0] if x.shape[0] == 1 else (x[0] + x[1]) * 0.5


def kit_shots(palette: Palette, kit_name: str, *, cache: Path | str | None = None, root: Path = ROOT,
              workers: int = 8) -> KitShots:
    """Every one-shot of a kit, built or loaded once each (in parallel processes; the result never depends
    on how many). Each lane's dB is its `db` plus the kit's `trim_db` (SHOOTOUTS.md §7.1), the sum taken only
    when the trim is not 0."""
    kit = palette.kits[kit_name]
    wanted = list(dict.fromkeys(s for lane in LANES for layer in LAYERS for s in kit.lanes[lane].layers[layer]))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(load_shot, s, palette, cache=cache, root=root) for s in wanted]
        loaded = dict(zip(wanted, (f.result() for f in futures)))
    shots, pcm = {}, {}
    for lane in LANES:
        for layer in LAYERS:
            layer_shots = kit.lanes[lane].layers[layer]
            shots[(lane, layer)] = tuple(mono(loaded[s][0]) for s in layer_shots)
            pcm[(lane, layer)] = tuple(loaded[s][1] for s in layer_shots)
    trim = kit.trim_db
    lanes = {}
    for lane in LANES:
        d = kit.lanes[lane]
        lanes[lane] = (d.pan, d.db + trim if trim else d.db)
    return KitShots(kit_name, shots, lanes, pcm)
