"""render(): LoopScript + brief + palette -> one seamless stereo cycle at -14 LUFS (instrument.md §3-4;
docs/RENDER.md).

Stages, each a pure function of its inputs:
1. parse (gates G1-G6) and compile (docs/COMPILER.md);
2. pitched stems: one fresh FluidSynth 2.6.1 process per stem (keys, bass, sax) on a MIDI file of
   PREROLL + cycles cycles. The pre-roll cycle is dropped, so every kept cycle starts with the previous
   cycle's tails, from voices that have already played the loop once (docs/COMPILER.md §4.3 caveat);
3. drums: the one-shot sampler's periodic cycle, tiled, every lane at its dB plus the kit's trim_db;
4. bus chains, faders and the master chain over the kept cycles; the last cycle is the loop ("two
   cycles, keep the second", §0 amendment 3). Each bus plays its instrument's levels (Palette.bus_levels,
   docs/SHOOTOUTS.md §7.1): a sax, keys or bass trim_db is a Gain at the head of its chain, and the gain after
   the chain is bus_gain_db + fader_db;
5. the bed at LUFS_music - Δ, the music measured after the master chain and before the limiter;
6. loudness: gain into the true-peak limiter on the 3x tile, -14 LUFS within 0.1 LU in at most 4 iterations
   (render/loudness.py), true peak checked on the loop as it repeats;
7. gate G7, and a manifest with no wall clock, paths or cache state in it, so it is as deterministic as the audio.

CLI: python -m lofi_rl.render.pipeline loop.ls --mood wistful --energy mid --bpm 82 --tonic 2
         --palette palettes/dev_gu.yaml --out x.wav [--flac]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from lofi_rl.brief import BPM_RANGE, ENERGIES, MOOD_NAMES, Brief
from lofi_rl.compile.compiler import CompileError, compile_loop
from lofi_rl.compile.events import CompiledLoop
from lofi_rl.compile.midi import write_stem_midi
from lofi_rl.dsl import Loop, canon, canon_sha, parse
from lofi_rl.dsl.repair import penalty
from lofi_rl.palette import ROOT, Palette, PaletteError, load_palette

from . import fluid
from .bed import bed_loop, place, samples
from .formats import write_flac, write_wav
from .loudness import CEILING_DBTP, LoudnessError, limit_loop, lufs, normalize_loop
from .machine import platform_info
from .mix import BUFFER, BUSES, TRIM_BUSES, mix_buses
from .sampler import kit_shots, render_drums
from .wavio import pcm_sha256, read_float

SR = 44100
CYCLES = 2                      # cycles through the bus chains; the last is kept (§3 Seamless)
PREROLL = 1                     # extra FluidSynth cycle rendered first and dropped
LIMITER_TILE = 3                # loudness.limit_loop limits the loop tiled 3 times (§0 amendment 3)
MUSIC_FLOOR_LUFS = -50.0        # G7: music below -50 LUFS
GATE_LU = 0.5                   # G7: final LUFS not within 0.5 LU of the target
GATE_DBTP = -0.9                # G7: true peak above -0.9 dBTP
TP_PAD = 64                     # samples of wrap-around context for the loop's true peak
SCHEMA = "lofi-rl/render/1"


class RenderError(RuntimeError):
    """A loop that fails a gate: G1-G6 from the parser, G7 from the render (instrument.md §2)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# --------------------------------------------------------------------------------------------- stages
def fluid_stems(compiled: CompiledLoop, palette: Palette, cycles: int, *, root: Path = ROOT,
                workers: int = 3) -> dict[str, np.ndarray]:
    """keys, bass and sax over `cycles` cycles: (2, cycles n) float32 each. A stem without notes is silence.

    Every stem MIDI ends a tail after its last cycle, so an output shorter than its cycles, at another rate,
    or silent in a cycle where its notes play is FluidSynth's fault (a full disk, a SoundFont it could not
    load), never padded or passed on: fluid.FluidSynthError, not a gate."""
    n, total = compiled.n, PREROLL + cycles
    out: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as d, ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {}
        for stem in compiled.stems:
            if stem.notes:
                mid = write_stem_midi(compiled, stem.name, Path(d) / f"{stem.name}.mid", cycles=total)
                futures[stem.name] = ex.submit(fluid.render_midi, palette.source_path(stem.src, root), mid,
                                               Path(d) / f"{stem.name}.wav")
        for stem in compiled.stems:
            if stem.name not in futures:
                out[stem.name] = np.zeros((2, cycles * n), dtype=np.float32)
                continue
            x, sr = read_float(futures[stem.name].result())
            if sr != SR or x.shape[0] not in (1, 2):
                raise fluid.FluidSynthError(f"FluidSynth gave {x.shape[0]} channels at {sr} Hz for {stem.name}")
            if x.shape[1] < total * n:
                raise fluid.FluidSynthError(f"FluidSynth's {stem.name} render has {x.shape[1]} samples, fewer than "
                                            f"the {total * n} of its {total} MIDI cycles")
            if x.shape[0] == 1:
                x = np.concatenate([x, x])
            kept = np.ascontiguousarray(x[:, PREROLL * n:total * n])
            silent = [c + 1 for c in range(cycles) if not kept[:, c * n:(c + 1) * n].any()]
            if silent:
                raise fluid.FluidSynthError(f"the {stem.name} stem has {len(stem.notes)} notes but FluidSynth "
                                            f"rendered kept cycle(s) {silent} silent")
            out[stem.name] = kept
    return out


@dataclass(frozen=True, eq=False)
class Music:
    """The music over all rendered cycles, after the master chain and before the bed and the limiter."""

    cycles: np.ndarray                  # (2, cycles n) float32
    n: int
    bus_lufs: dict[str, float | None]   # each bus in the last cycle, after its trim and chain, before its fader
    stems: dict[str, str]               # pcm_sha256 of each bus input's last cycle (drums: with the kit's trim)
    oneshots: dict[str, list[str]]      # "lane/layer" -> pcm_sha256 of each one-shot

    @property
    def last(self) -> np.ndarray:
        return np.ascontiguousarray(self.cycles[:, -self.n:])


def _lufs_or_none(x: np.ndarray) -> float | None:
    level = lufs(x, SR)
    return float(level) if np.isfinite(level) else None


def music_cycles(compiled: CompiledLoop, palette: Palette, cycles: int = CYCLES, *,
                 cache: Path | str | None = None, root: Path = ROOT, workers: int = 4) -> Music:
    """Stages 2-4: FluidSynth stems, the drum sampler, bus chains and the master chain, over `cycles` cycles. The
    pitched stems go into the mix untrimmed, with each instrument's trim_db and fader_db passed to mix_buses; the
    kit's trim is in its lanes (kit_shots)."""
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    n = compiled.n
    levels = palette.bus_levels(compiled.keys_name, compiled.kit, compiled.sax_name)
    stems = fluid_stems(compiled, palette, cycles, root=root, workers=min(workers, 3))
    kit = kit_shots(palette, compiled.kit, cache=cache, root=root, workers=workers)
    stems["drums"] = np.tile(render_drums(compiled, kit), (1, cycles))
    out = mix_buses(stems, palette.mix, SR, workers=workers,
                    trims={b: levels[b].trim_db for b in TRIM_BUSES},
                    faders={b: levels[b].fader_db for b in BUSES})
    return Music(
        cycles=out.music, n=n,
        bus_lufs={b: _lufs_or_none(out.buses[b][:, -n:]) for b in BUSES},
        stems={b: pcm_sha256(stems[b][:, -n:]) for b in BUSES},
        oneshots={f"{lane}/{layer}": list(h) for (lane, layer), h in kit.pcm.items() if h},
    )


def loop_true_peak_dbtp(y: np.ndarray, pad: int = TP_PAD) -> float:
    """True peak of a loop as it repeats: 4x resample_poly with the cycle's other end as context at the seam."""
    n = y.shape[1]
    ext = np.concatenate([y[:, -pad:], y, y[:, :pad]], axis=1).astype(np.float64)
    up = resample_poly(ext, 4, 1, axis=-1)[:, 4 * pad:4 * (pad + n)]
    peak = float(np.max(np.abs(up)))
    return float("-inf") if peak == 0 else 20 * np.log10(peak)


@dataclass(frozen=True, eq=False)
class Finished:
    audio: np.ndarray       # (2, n) float32: music + bed, normalized and limited
    music: np.ndarray       # (2, n) float32: the same gain and limiter without the bed
    report: dict            # loudness and bed numbers for the manifest


def finish(music: np.ndarray, palette: Palette, *, root: Path = ROOT) -> Finished:
    """Stages 5-7 on one cycle of music (after the master chain): bed, loudness, gate G7."""
    n = music.shape[1]
    if not np.all(np.isfinite(music)):
        raise RenderError("G7", "NaN or infinity in the music")
    level = lufs(music, SR)
    if not np.isfinite(level) or level < MUSIC_FLOOR_LUFS:
        raise RenderError("G7", f"music at {level:.2f} LUFS is below {MUSIC_FLOOR_LUFS}")
    loud = palette.loudness
    if loud.ceiling_dbtp != CEILING_DBTP:
        raise ValueError(f"loudness.ceiling_dbtp {loud.ceiling_dbtp}: the limiter in render/loudness.py is "
                         f"fixed to {CEILING_DBTP} dBTP")
    bed = bed_loop(palette, n, root=root)
    mixed, bed_report = music, None
    if bed is not None:
        scaled, gain, own = place(bed, level, palette.bed.delta_lu)
        mixed = (music.astype(np.float64) + scaled).astype(np.float32)
        bed_report = {"src": palette.bed.src, "delta_lu": palette.bed.delta_lu, "start_s": palette.bed.start_s,
                      "xfade_s": palette.bed.xfade_s, "xfade_samples": samples(palette.bed.xfade_s),
                      "lufs_source": float(own), "gain_db": float(gain), "lufs": float(level - palette.bed.delta_lu)}
    try:
        audio, info = normalize_loop(mixed, SR, target=loud.lufs, tol=loud.tol, iters=loud.iters)
    except LoudnessError as err:
        raise RenderError("G7", str(err)) from err
    if bed is None:
        music_out = audio
    else:
        music_out = limit_loop(music * np.float32(10 ** (info.gain_db / 20)), SR).astype(np.float32)
    tp, final = loop_true_peak_dbtp(audio), info.lufs
    if audio.shape != (2, n) or not np.all(np.isfinite(audio)):
        raise RenderError("G7", f"render shape {audio.shape} is not (2, {n}), or it is not finite")
    if abs(final - loud.lufs) > GATE_LU:
        raise RenderError("G7", f"final {final:.2f} LUFS is not within {GATE_LU} LU of {loud.lufs}")
    if tp > GATE_DBTP:
        raise RenderError("G7", f"true peak {tp:.2f} dBTP is above {GATE_DBTP}")
    music_only = ({"lufs": float(lufs(music_out, SR)), "true_peak_dbtp": loop_true_peak_dbtp(music_out)}
                  if bed is not None else {"lufs": float(final), "true_peak_dbtp": tp})
    report = {"music_lufs": float(level), "bed": bed_report,
              "loudness": {"target_lufs": loud.lufs, "tol_lu": loud.tol, "max_iterations": loud.iters,
                           "ceiling_dbtp": loud.ceiling_dbtp, "lufs": float(final), "gain_db": float(info.gain_db),
                           "iterations": info.iterations, "true_peak_dbtp": tp, "music_only": music_only}}
    return Finished(audio, music_out, report)


# --------------------------------------------------------------------------------------------- render
@dataclass(frozen=True, eq=False)
class Render:
    audio: np.ndarray           # (2, n) float32 at 44.1 kHz: render.wav
    music: np.ndarray           # (2, n) float32: music.wav, the same without the bed, for the scorers
    pre_limiter: np.ndarray     # (2, n) float32: the kept cycle after the master chain (stage 4)
    manifest: dict
    compiled: CompiledLoop


def render(loop: str | Loop, brief: Brief, palette: Palette, *, cycles: int = CYCLES,
           cache: Path | str | None = None, root: Path = ROOT, finish_reason: str | None = None,
           workers: int = 4) -> Render:
    """Render one loop. `loop` is LoopScript text (parsed with the palette's vocabulary and gated) or a
    parsed AST. Raises RenderError for a gate failure and CompileError for a loop the palette cannot play.
    Infrastructure faults are never gates: fluid.FluidSynthError, or a file error as it is. The result never
    depends on `workers`."""
    if cycles < 2:
        raise ValueError("a seamless loop needs at least 2 cycles through the bus chains")
    if isinstance(loop, str):
        res = parse(loop, vocab=palette.vocab(), finish_reason=finish_reason, mode=brief.mode)
        if not res.gate.ok:
            raise RenderError(res.gate.code or "G1", res.gate.detail)
        ast, repairs, parse_gate = res.ast, list(res.repairs), "pass"
    else:
        ast, repairs, parse_gate = loop, [], "not run: AST given"
    compiled = compile_loop(ast, brief, palette)
    levels = palette.bus_levels(compiled.keys_name, compiled.kit, compiled.sax_name)
    music = music_cycles(compiled, palette, cycles, cache=cache, root=root, workers=workers)
    done = finish(music.last, palette, root=root)
    all_repairs = repairs + list(compiled.repairs)
    manifest = {
        "schema": SCHEMA,
        "loop": {"canon": canon(ast), "canon_sha256": canon_sha(ast), "compiled_sha256": compiled.digest(),
                 "repairs": [asdict(r) for r in all_repairs], "p_fix": penalty(all_repairs)},
        "brief": {"scene": brief.scene, "energy": brief.energy, "mood": brief.mood, "mode": brief.mode,
                  "bpm": brief.bpm, "tonic": brief.tonic, "key": brief.key_name},
        "palette": {"id": palette.id, "sha256": palette.sha256, "frozen": palette.frozen},
        "platform": dict(platform_info()),         # a copy: platform_info() is cached
        "n": compiled.n, "sr": SR, "bpm_eff": compiled.bpm_eff, "seconds": compiled.n / SR,
        "chain": {"cycles": cycles, "fluid_preroll_cycles": PREROLL, "midi_cycles": PREROLL + cycles,
                  "pedalboard_buffer": BUFFER, "limiter_tile": LIMITER_TILE},
        "instruments": {"keys": compiled.keys_name, "kit": compiled.kit, "sax": compiled.sax_name,
                        "sax_warm": palette.mix.sax_warm},
        "stems": music.stems,
        "oneshots": music.oneshots,
        # gain_db is mix.bus_gain_db; instrument, trim_db and fader_db are the playing instrument's (SHOOTOUTS.md
        # §7.1). The gain after the bus's chain is gain_db + fader_db. Additive keys: the schema stays render/1.
        "buses": {b: {"lufs": music.bus_lufs[b], "gain_db": palette.mix.bus_gain_db[b],
                      "instrument": levels[b].instrument, "trim_db": levels[b].trim_db,
                      "fader_db": levels[b].fader_db} for b in BUSES},
        **done.report,
        "gates": {"G1-G6": parse_gate, "G7": "pass"},
        "audio_sha256": pcm_sha256(done.audio),
        "music_sha256": pcm_sha256(done.music),
    }
    return Render(done.audio, done.music, music.last, manifest, compiled)


def manifest_json(manifest: dict) -> str:
    return json.dumps(manifest, sort_keys=True, indent=2) + "\n"


# --------------------------------------------------------------------------------------------- CLI
def _args(argv: list[str] | None) -> tuple[argparse.Namespace, str, Brief, Palette]:
    """The parsed arguments, the loop text, the brief and the loaded palette. A bad argument (a range, a
    missing or unreadable file, a palette that does not load) exits 2 before anything renders
    (docs/RENDER.md §1)."""
    p = argparse.ArgumentParser(prog="python -m lofi_rl.render.pipeline", description=__doc__.split("\n\n")[0])
    p.add_argument("loop", type=Path, help="a LoopScript file")
    p.add_argument("--mood", required=True, choices=MOOD_NAMES)
    p.add_argument("--energy", required=True, choices=ENERGIES)
    p.add_argument("--bpm", required=True, type=int)
    p.add_argument("--tonic", required=True, type=int, help="pitch class of the key centre, C = 0")
    p.add_argument("--palette", type=Path, default=ROOT / "palettes" / "dev_gu.yaml")
    p.add_argument("--out", required=True, type=Path, help="render.wav; music, manifest and FLAC go next to it")
    p.add_argument("--flac", action="store_true", help="also write a 16-bit dithered FLAC listening copy")
    p.add_argument("--cycles", type=int, default=CYCLES)
    p.add_argument("--cache", type=Path, default=None, help="one-shot cache (default $LOFI_CACHE or ~/.cache/lofi-rl)")
    p.add_argument("--workers", type=int, default=4, help="parallel processes; the output never depends on it")
    p.add_argument("--no-verify", action="store_true", help="skip hashing the palette's source files")
    a = p.parse_args(argv)
    if a.cycles < 2:
        p.error("--cycles must be at least 2 (a seamless loop keeps the last of 2 or more)")
    if not BPM_RANGE[0] <= a.bpm <= BPM_RANGE[1]:
        p.error(f"--bpm {a.bpm} is outside {BPM_RANGE[0]}-{BPM_RANGE[1]} (docs/TASTE.md, Tempo)")
    if not 0 <= a.tonic <= 11:
        p.error(f"--tonic {a.tonic} is not a pitch class 0-11 (C = 0)")
    for name, path in (("loop", a.loop), ("--palette", a.palette)):
        if not path.is_file():
            p.error(f"{name}: {path} is not a file")
    try:
        text = a.loop.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        p.error(f"loop: {a.loop}: {err}")
    try:
        palette = load_palette(a.palette, verify=not a.no_verify)
    except (PaletteError, OSError) as err:
        p.error(f"--palette {a.palette}: {err}")
    return a, text, Brief(energy=a.energy, mood=a.mood, bpm=a.bpm, tonic=a.tonic), palette


def main(argv: list[str] | None = None) -> int:
    a, text, brief, palette = _args(argv)
    try:
        r = render(text, brief, palette, cycles=a.cycles, cache=a.cache, workers=a.workers)
    except RenderError as err:
        print(f"gate {err.code}: {err.detail}", file=sys.stderr)
        return 3
    except CompileError as err:
        print(f"compile: {err}", file=sys.stderr)
        return 3
    out = a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    written = [write_wav(out, r.audio), write_wav(out.with_suffix(".music.wav"), r.music)]
    side = out.with_suffix(".manifest.json")
    side.write_text(manifest_json(r.manifest), encoding="utf-8")
    written.append(side)
    if a.flac:
        written.append(write_flac(out.with_suffix(".flac"), r.audio))
    m = r.manifest
    print(f"N={m['n']} LUFS={m['loudness']['lufs']:.3f} true_peak={m['loudness']['true_peak_dbtp']:.3f} dBTP "
          f"pcm_sha256={m['audio_sha256']}")
    print(manifest_json(m), end="")
    print("wrote " + " ".join(str(p) for p in written), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
