"""FluidSynth's command line (instrument.md §4; render/fluid.py): the pinned 2.6.1, the exact flags of every
render, one fresh process per pitched stem with the pre-roll cycle in its MIDI, and the spec's
`-o synth.limiter.active=0`, which FluidSynth 2.6.1 rejects (so the flags leave it out). Also what keeps the
PCM a function of the flags alone (an empty home, no system config) and what makes a FluidSynth fault an error
instead of a different render (substituted presets, errors in its log, short, silent or wrong-rate output)."""
import os
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import mido
import numpy as np
import pytest

from helpers_compile import compiled, dev_palette
from helpers_render import needs_fluidsynth
from lofi_rl.render import fluid
from lofi_rl.render import pipeline as pl
from lofi_rl.render.wavio import pcm_sha256, read_float, write_float

FLAGS = ["-q", "-n", "-i", "-R", "0", "-C", "0", "-g", "0.2", "-r", "44100", "-o", "synth.cpu-cores=1",
         "-T", "wav", "-O", "float", "-F"]


@needs_fluidsynth
def test_the_installed_fluidsynth_is_the_pinned_2_6_1():
    assert fluid.version() == fluid.PINNED == "2.6.1"


def _fake_fluidsynth(monkeypatch, *, returncode=0, stderr=""):
    """Stub FluidSynth's process calls (and the system config, absent); returns (cmd, env, the files in its
    HOME while it runs) for each call."""
    monkeypatch.setattr(fluid, "SYSTEM_CONFIG", Path("/nonexistent/fluidsynth.conf"))
    calls = []

    def fake_run(cmd, **kwargs):
        env = kwargs.get("env")
        calls.append((list(cmd), env and dict(env), env and sorted(os.listdir(env["HOME"]))))
        if "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="FluidSynth runtime version 2.6.1\n", stderr="")
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)
    monkeypatch.setattr(fluid.shutil, "which", lambda name: "/opt/bin/fluidsynth")
    monkeypatch.setattr(fluid.subprocess, "run", fake_run)
    return calls


def test_every_render_uses_the_pinned_flags(monkeypatch):
    calls = _fake_fluidsynth(monkeypatch)
    fluid.render_midi(Path("/sf/gu.sf2"), Path("/m/keys.mid"), Path("/o/keys.wav"))
    assert [cmd for cmd, _, _ in calls] == [
        ["/opt/bin/fluidsynth", "--version"],
        ["/opt/bin/fluidsynth"] + FLAGS + ["/o/keys.wav", "/sf/gu.sf2", "/m/keys.mid"]]
    assert "synth.limiter.active=0" not in calls[-1][0]


def test_every_render_runs_with_an_empty_home_of_its_own(monkeypatch):
    # FluidSynth 2.6.1 sources $HOME/.fluidsynth on every start, even with -f, so the PCM would depend on it.
    calls = _fake_fluidsynth(monkeypatch)
    for _ in range(2):
        fluid.render_midi(Path("/sf/gu.sf2"), Path("/m/keys.mid"), Path("/o/keys.wav"))
    renders = [(env, listing) for cmd, env, listing in calls if "--version" not in cmd]
    homes = [env["HOME"] for env, _ in renders]
    assert all(listing == [] for _, listing in renders), "the home is empty while FluidSynth runs"
    assert len(set(homes)) == 2 and os.environ.get("HOME") not in homes and not any(map(os.path.exists, homes))
    assert all(env["PATH"] == os.environ["PATH"] for env, _ in renders), "the rest of the environment is kept"


def test_a_system_config_with_commands_stops_every_render(monkeypatch, tmp_path):
    # HOME cannot hide /etc/fluidsynth.conf. A file of blanks and comments changes nothing, so it may stay.
    calls = _fake_fluidsynth(monkeypatch)
    conf = tmp_path / "fluidsynth.conf"
    monkeypatch.setattr(fluid, "SYSTEM_CONFIG", conf)
    for text in ("", "# set by the image\n\n   # nothing else\n"):
        conf.write_text(text)
        fluid.render_midi(Path("/sf/gu.sf2"), Path("/m/keys.mid"), Path("/o/keys.wav"))
    renders = len([cmd for cmd, _, _ in calls if "--version" not in cmd])
    conf.write_text("# tuned\ngain 0.6\n")
    with pytest.raises(fluid.FluidSynthError, match="fluidsynth.conf holds commands \\('gain 0.6'"):
        fluid.render_midi(Path("/sf/gu.sf2"), Path("/m/keys.mid"), Path("/o/keys.wav"))
    assert renders == 2 and len([cmd for cmd, _, _ in calls if "--version" not in cmd]) == 2, "it never starts"


@pytest.mark.parametrize("returncode, stderr, message", [
    (1, "fluidsynth: error: out of memory\n", "exited 1"),
    (0, "fluidsynth: error: Failed to load SoundFont\n", "Failed to load SoundFont"),
    (0, "fluidsynth: warning: Instrument not found on channel 9 [bank=128 prog=33], substituted [bank=128 prog=0]\n",
     "substituted")])
def test_a_fluidsynth_error_or_substituted_preset_is_an_infrastructure_fault(monkeypatch, returncode, stderr,
                                                                              message):
    _fake_fluidsynth(monkeypatch, returncode=returncode, stderr=stderr)
    with pytest.raises(fluid.FluidSynthError, match=message):
        fluid.render_midi(Path("/sf/gu.sf2"), Path("/m/keys.mid"), Path("/o/keys.wav"))


def _one_note(path: Path, *, channel: int = 0, bank: int | None = 0, program: int = 4, note: int = 60) -> Path:
    mid = mido.MidiFile(type=0, ticks_per_beat=22050)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    if bank is not None:
        track.append(mido.Message("control_change", channel=channel, control=0, value=bank))
    track.append(mido.Message("program_change", channel=channel, program=program))
    track.append(mido.Message("note_on", channel=channel, note=note, velocity=90))
    track.append(mido.Message("note_off", channel=channel, note=note, velocity=0, time=4410))
    track.append(mido.MetaMessage("end_of_track", time=441))
    mid.save(path)
    return path


@needs_fluidsynth
def test_a_fluidsynth_user_config_does_not_reach_the_pcm(tmp_path, monkeypatch):
    sf2, mid = dev_palette().source_path("gu"), _one_note(tmp_path / "one.mid")
    clean, sr = read_float(fluid.render_midi(sf2, mid, tmp_path / "clean.wav"))
    home = tmp_path / "home"
    home.mkdir()
    (home / ".fluidsynth").write_text("gain 0.6\ninterp 0\nset synth.sample-rate 48000\n")
    cmd = [shutil.which("fluidsynth")] + FLAGS + [str(tmp_path / "seen.wav"), str(sf2), str(mid)]
    subprocess.run(cmd, check=True, capture_output=True, env=dict(os.environ, HOME=str(home)))
    seen, seen_sr = read_float(tmp_path / "seen.wav")
    assert seen_sr != sr or pcm_sha256(seen) != pcm_sha256(clean), "FluidSynth itself reads the file"
    monkeypatch.setenv("HOME", str(home))
    dirty, dirty_sr = read_float(fluid.render_midi(sf2, mid, tmp_path / "dirty.wav"))
    assert (dirty_sr, pcm_sha256(dirty)) == (sr, pcm_sha256(clean)) and sr == 44100


@needs_fluidsynth
def test_fluidsynth_reads_no_config_file_but_the_two_this_module_controls(tmp_path):
    # Pins what FluidSynth 2.6.1 tries (its -v log): $HOME/.fluidsynth, which the empty home hides, and
    # /etc/fluidsynth.conf, which stops the render while it exists. A build that tries more fails here.
    home = tmp_path / "home"
    home.mkdir()
    cmd = [shutil.which("fluidsynth"), "-v"] + FLAGS + [str(tmp_path / "x.wav"), str(dev_palette().source_path("gu")),
                                                       str(_one_note(tmp_path / "one.mid"))]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=dict(os.environ, HOME=str(home)))
    tried = set(re.findall(r"load config file '([^']+)'", proc.stdout + proc.stderr))
    assert tried == {str(fluid.SYSTEM_CONFIG), str(home / ".fluidsynth")}


@needs_fluidsynth
@pytest.mark.parametrize("channel, bank, program", [(9, None, 33), (0, 7, 4)], ids=["drum_kit_33", "bank_7_prog_4"])
def test_a_preset_the_soundfont_lacks_is_never_silently_replaced(tmp_path, channel, bank, program):
    # GeneralUser GS has no drum kit 33 and no bank-7 program 4; FluidSynth would play kit 0 or bank 0 instead.
    mid = _one_note(tmp_path / "one.mid", channel=channel, bank=bank, program=program, note=38)
    with pytest.raises(fluid.FluidSynthError, match="substituted"):
        fluid.render_midi(dev_palette().source_path("gu"), mid, tmp_path / "x.wav")


def test_any_other_version_is_refused(monkeypatch):
    monkeypatch.setattr(fluid.shutil, "which", lambda name: "/opt/bin/fluidsynth")
    monkeypatch.setattr(fluid, "version", lambda exe=None: "2.6.0")
    with pytest.raises(RuntimeError, match="2.6.1 is pinned"):
        fluid.render_midi(Path("a.sf2"), Path("a.mid"), Path("a.wav"))


def test_a_version_pass_is_remembered_only_for_the_same_binary_file(monkeypatch, tmp_path):
    exe = tmp_path / "fluidsynth"
    exe.write_text("#!/bin/sh\n")
    asked = []
    monkeypatch.setattr(fluid, "version", lambda e=None: asked.append(e) or ("2.6.1" if len(asked) < 3 else "2.6.0"))
    fluid.check_version(str(exe))
    fluid.check_version(str(exe))
    assert len(asked) == 1, "one --version per binary"
    exe.write_text("#!/bin/sh\n# rebuilt\n")
    fluid.check_version(str(exe))
    assert len(asked) == 2, "a changed file is checked again"
    exe.write_text("#!/bin/sh\n# downgraded\n")
    with pytest.raises(fluid.FluidSynthError, match="2.6.0 found"):
        fluid.check_version(str(exe))


@needs_fluidsynth
def test_the_spec_limiter_flag_is_not_a_fluidsynth_2_6_1_setting(tmp_path):
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=90, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    mid.save(tmp_path / "one.mid")
    sf2 = dev_palette().source_path("gu")
    cmd = [shutil.which("fluidsynth"), "-q", "-n", "-i", "-o", "synth.limiter.active=0", "-T", "wav", "-O", "float",
           "-F", str(tmp_path / "x.wav"), str(sf2), str(tmp_path / "one.mid")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0 and "synth.limiter.active" in proc.stdout + proc.stderr


def _fake_stems(monkeypatch, c, *, length=None, sr=44100, silent=(), silent_from=0):
    """Stub fluid.render_midi with a constant 1e-3 signal of `length` samples (default: the MIDI's 3 cycles plus
    the tail), zero from sample `silent_from` on for the stems in `silent`; returns the calls."""
    calls = []

    def fake_render(sf2_path, midi, out_wav, *args, **kwargs):
        mid = mido.MidiFile(str(midi))
        calls.append((Path(sf2_path), midi.name, sum(msg.time for msg in mid.tracks[0])))
        x = np.full((2, 3 * c.n + 4410 if length is None else length), 1e-3, dtype=np.float32)
        if midi.stem in silent:
            x[:, silent_from:] = 0.0
        write_float(out_wav, x, sr)
        return out_wav
    monkeypatch.setattr(fluid, "render_midi", fake_render)
    return calls


def _tiny(c):
    """The compiled loop cut to one note per stem in a 640-sample cycle: fluid_stems only measures the stub's
    output against n, so small MIDI files and stub WAVs keep these tests fast."""
    one = {s.name: replace(s, notes=(replace(s.notes[0], on=0, off=100),), bend=(), cc11=()) for s in c.stems}
    return replace(c, n=640, **one)


def test_a_short_fluidsynth_output_is_an_error_not_zero_padding(monkeypatch):
    # Every stem MIDI ends after (1 + C) N samples plus a tail, so a shorter output can only be a fault.
    c = _tiny(compiled("example.ls"))
    _fake_stems(monkeypatch, c, length=3 * c.n - 64)
    with pytest.raises(fluid.FluidSynthError, match="fewer than"):
        pl.fluid_stems(c, dev_palette(), 2)
    _fake_stems(monkeypatch, c, length=3 * c.n)
    assert {k: v.shape for k, v in pl.fluid_stems(c, dev_palette(), 2).items()} == {
        s: (2, 2 * c.n) for s in ("keys", "bass", "sax")}, "exactly the MIDI's cycles is enough"


@pytest.mark.parametrize("silent_cycles", [0, 2], ids=["all", "kept_cycle"])
def test_a_stem_with_notes_that_renders_silent_is_an_error(monkeypatch, silent_cycles):
    c = _tiny(compiled("example.ls"))
    assert c.sax.notes
    _fake_stems(monkeypatch, c, silent={"sax"}, silent_from=silent_cycles * c.n)
    with pytest.raises(fluid.FluidSynthError, match="sax stem .* silent"):
        pl.fluid_stems(c, dev_palette(), 2)


def test_a_wrong_sample_rate_is_an_infrastructure_fault_not_gate_g7(monkeypatch):
    c = _tiny(compiled("example.ls"))
    _fake_stems(monkeypatch, c, sr=48000)
    with pytest.raises(fluid.FluidSynthError, match="48000 Hz") as err:
        pl.fluid_stems(c, dev_palette(), 2)
    assert not isinstance(err.value, pl.RenderError)


def test_one_fresh_fluidsynth_process_per_stem_with_a_pre_roll_cycle(monkeypatch):
    c = compiled("example.ls")
    sf2 = dev_palette().source_path("gu")
    calls = _fake_stems(monkeypatch, c)
    stems = pl.fluid_stems(c, dev_palette(), 2)
    assert sorted(name for _, name, _ in calls) == ["bass.mid", "keys.mid", "sax.mid"]
    assert all(path == sf2 for path, _, _ in calls)
    assert all(3 * c.n + 4410 <= end < 3 * c.n + 44100 for _, _, end in calls), "MIDI = pre-roll + 2 cycles"
    assert {k: v.shape for k, v in stems.items()} == {s: (2, 2 * c.n) for s in ("keys", "bass", "sax")}

    calls.clear()
    no_keys = replace(c, keys=replace(c.keys, notes=()))
    stems = pl.fluid_stems(no_keys, dev_palette(), 2)
    assert sorted(name for _, name, _ in calls) == ["bass.mid", "sax.mid"], "an empty stem is not rendered"
    assert not stems["keys"].any() and stems["keys"].shape == (2, 2 * c.n)
