"""FluidSynth 2.6.1 rendering through the CLI fast-render path (docs/research/sound_sources.md, part B5):
- float output, with chorus and reverb off (all effects come from pedalboard);
- one core, and a fresh process per render, so no dither or voice state carries over from an earlier render.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

PINNED = "2.6.1"   # 2.6.0 changed tuning, the default reverb and interpolation: both machines pin 2.6.1
GAIN = 0.2         # FluidSynth's default; float output never clips, and loudness is normalized downstream


def fluidsynth_bin() -> str:
    exe = shutil.which("fluidsynth")
    if not exe:
        raise FileNotFoundError("fluidsynth is not on PATH (laptop: brew install fluid-synth; cluster: jobs/tools.sbatch)")
    return exe


def version(exe: str | None = None) -> str:
    out = subprocess.run([exe or fluidsynth_bin(), "--version"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "runtime version" in line:
            return line.rsplit(" ", 1)[-1].strip()
    raise RuntimeError(f"cannot read the FluidSynth version from {out[:200]!r}")


def render_midi(sf2: Path, midi: Path, out_wav: Path, sr: int = 44100, gain: float = GAIN) -> Path:
    """Render `midi` with `sf2` to a float32 WAV at `sr`. Raises unless FluidSynth is the pinned version."""
    exe = fluidsynth_bin()
    found = version(exe)
    if found != PINNED:
        raise RuntimeError(f"FluidSynth {found} found but {PINNED} is pinned (docs/DECISIONS.md D-005)")
    cmd = [exe, "-q", "-n", "-i", "-R", "0", "-C", "0", "-g", f"{gain}", "-r", f"{sr}",
           "-o", "synth.cpu-cores=1", "-T", "wav", "-O", "float", "-F", str(out_wav), str(sf2), str(midi)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_wav
