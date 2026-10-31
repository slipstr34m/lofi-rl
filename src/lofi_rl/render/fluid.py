"""FluidSynth 2.6.1 rendering through the CLI fast-render path (docs/research/sound_sources.md, part B5):
- float output, with chorus and reverb off (all effects come from pedalboard);
- one core, and a fresh process per render, so no dither or voice state carries over from an earlier render;
- no settings but these flags. FluidSynth 2.6.1 sources /etc/fluidsynth.conf and $HOME/.fluidsynth on every
  start, whatever the flags (-f adds a file; it does not replace them; measured 2026-10-31). So every render
  runs with HOME set to a new empty directory, and no render starts while /etc/fluidsynth.conf holds commands;
- FluidSynth plays what the MIDI asks for or the render fails: a non-zero exit, an error in its log, or a
  preset it substituted for one the SoundFont lacks raises FluidSynthError. These are faults of the machine or
  the palette, never a loop's gate.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

PINNED = "2.6.1"   # 2.6.0 changed tuning, the default reverb and interpolation: both machines pin 2.6.1
GAIN = 0.2         # FluidSynth's default; float output never clips, and loudness is normalized downstream
SYSTEM_CONFIG = Path("/etc/fluidsynth.conf")    # sourced on every start; HOME cannot hide it
# Log lines that mean the render is not what the MIDI asked for. A missing preset with no stand-in leaves
# its channel silent instead, which the callers' silence checks catch.
_FAULT = re.compile(r"^fluidsynth: (?:panic|error):|Instrument not found on channel .*substituted")


class FluidSynthError(RuntimeError):
    """FluidSynth did not render what it was asked to: an infrastructure or palette fault, never a gate."""


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


_PINNED_BINARIES: set[tuple[str, int, int, int]] = set()


def check_version(exe: str) -> None:
    """Raise unless `exe` runs the pinned FluidSynth. `fluidsynth --version` starts a whole FluidSynth (about
    50 ms), so a pass is remembered for that file as it is now (path, size, mtime, inode): a replaced or
    rebuilt binary is checked again."""
    try:
        st = os.stat(exe)
        key = (os.path.realpath(exe), st.st_size, st.st_mtime_ns, st.st_ino)
    except OSError:
        key = None
    if key is not None and key in _PINNED_BINARIES:
        return
    found = version(exe)
    if found != PINNED:
        raise FluidSynthError(f"FluidSynth {found} found but {PINNED} is pinned (docs/DECISIONS.md D-005)")
    if key is not None:
        _PINNED_BINARIES.add(key)


def config_commands(path: Path) -> list[str]:
    """The commands in a FluidSynth config file: every line but blanks and # comments. None when the file is
    absent or unreadable, since FluidSynth, running as this user, cannot read it either."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [s for s in (line.strip() for line in text.splitlines()) if s and not s.startswith("#")]


def render_midi(sf2: Path, midi: Path, out_wav: Path, sr: int = 44100, gain: float = GAIN) -> Path:
    """Render `midi` with `sf2` to a float32 WAV at `sr`. Raises unless FluidSynth is the pinned version, and
    FluidSynthError unless it ran clean in an empty home."""
    exe = fluidsynth_bin()
    check_version(exe)
    commands = config_commands(SYSTEM_CONFIG)
    if commands:
        raise FluidSynthError(f"{SYSTEM_CONFIG} holds commands ({commands[0]!r}, ...): FluidSynth {PINNED} sources "
                              f"it on every start, so renders would depend on it. Remove them to render.")
    cmd = [exe, "-q", "-n", "-i", "-R", "0", "-C", "0", "-g", f"{gain}", "-r", f"{sr}",
           "-o", "synth.cpu-cores=1", "-T", "wav", "-O", "float", "-F", str(out_wav), str(sf2), str(midi)]
    with tempfile.TemporaryDirectory(prefix="lofi-fluid-home-") as home:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                              env=dict(os.environ, HOME=home))
    log = f"{proc.stdout}\n{proc.stderr}".strip()
    if proc.returncode != 0:
        raise FluidSynthError(f"FluidSynth exited {proc.returncode} on {Path(midi).name}: {log[-600:]}")
    faults = [line for line in log.splitlines() if _FAULT.search(line)]
    if faults:
        raise FluidSynthError(f"FluidSynth on {Path(midi).name} with {Path(sf2).name}: " + " | ".join(faults[:4]))
    return out_wav
