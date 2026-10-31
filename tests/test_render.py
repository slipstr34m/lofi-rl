"""The whole render (instrument.md §4, §7 test_render): exactly N samples, -14 ± 0.1 LUFS, true peak at or
below -1 dBTP on the loop as it repeats, the bed at -Δ ± 0.5 LU, gate G7, and a manifest that matches the
audio. Every number is measured here, independently of the manifest."""
import json

import numpy as np
import pytest

from helpers_compile import EXAMPLE_BRIEF, FIX, ROOT
from helpers_render import needs_fluidsynth, palette, rendered, with_bed
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.render.formats import write_wav
from lofi_rl.render.loudness import lufs, true_peak_dbtp
from lofi_rl.render.pipeline import RenderError, finish, loop_true_peak_dbtp, main, render
from lofi_rl.render.wavio import pcm_sha256, read_float

SR = 44100


@pytest.fixture(scope="module")
def example():
    return rendered("example.ls")


@needs_fluidsynth
def test_length_loudness_and_true_peak(example):
    n = loop_samples(82)
    assert n == 1_032_576 == example.compiled.n
    for x in (example.audio, example.music, example.pre_limiter):
        assert x.shape == (2, n) and x.dtype == np.float32 and np.all(np.isfinite(x))
    assert abs(lufs(example.audio, SR) + 14.0) <= 0.1
    wrapped = np.concatenate([example.audio[:, -4410:], example.audio, example.audio[:, :4410]], axis=1)
    assert true_peak_dbtp(wrapped) <= -1.0, "measured across the seam, where the loop wraps"
    assert loop_true_peak_dbtp(example.audio) <= -1.0
    assert example.manifest["loudness"]["iterations"] <= 4


@needs_fluidsynth
def test_the_manifest_matches_the_audio(example):
    m = example.manifest
    assert m["n"] == example.compiled.n and m["sr"] == SR and m["bpm_eff"] == example.compiled.bpm_eff
    assert m["audio_sha256"] == pcm_sha256(example.audio) and m["music_sha256"] == pcm_sha256(example.music)
    assert m["loudness"]["lufs"] == pytest.approx(lufs(example.audio, SR), abs=1e-9)
    assert m["loudness"]["true_peak_dbtp"] == pytest.approx(loop_true_peak_dbtp(example.audio), abs=1e-9)
    assert m["music_lufs"] == pytest.approx(lufs(example.pre_limiter, SR), abs=1e-9)
    assert m["palette"]["sha256"] == palette().sha256 and m["loop"]["compiled_sha256"] == example.compiled.digest()
    assert m["brief"]["key"] == "D dorian" and m["gates"] == {"G1-G6": "pass", "G7": "pass"}
    assert m["chain"] == {"cycles": 2, "fluid_preroll_cycles": 1, "midi_cycles": 3, "pedalboard_buffer": 8192,
                          "limiter_tile": 3}
    assert m["bed"] is None and np.array_equal(example.music, example.audio), "bed: null renders without a bed"
    assert set(m["stems"]) == {"sax", "keys", "bass", "drums"} and len(m["oneshots"]) == 18
    assert {"system", "machine", "cpu", "fluidsynth", "numpy", "pedalboard"} <= set(m["platform"])
    text = json.dumps(m)
    assert str(ROOT) not in text and "cache" not in text, "no paths or cache state: the manifest is deterministic"


@needs_fluidsynth
def test_the_render_round_trips_through_its_wav(example, tmp_path):
    path = write_wav(tmp_path / "render.wav", example.audio)
    assert b"PEAK" not in path.read_bytes()[:256]
    x, sr = read_float(path)
    assert sr == SR and pcm_sha256(x) == example.manifest["audio_sha256"]


@needs_fluidsynth
def test_the_bed_sits_delta_below_the_music(example, tmp_path):
    n = example.compiled.n
    pal = with_bed(tmp_path, n, delta_lu=24.0)
    done = finish(example.pre_limiter, pal)
    assert done.audio.shape == done.music.shape == (2, n)
    assert abs(lufs(done.audio, SR) + 14.0) <= 0.1 and loop_true_peak_dbtp(done.audio) <= -1.0
    bed_alone = done.audio.astype(np.float64) - done.music
    delta = lufs(bed_alone, SR) - lufs(done.music, SR)
    assert abs(delta + 24.0) <= 0.5, delta
    rep = done.report["bed"]
    assert rep["delta_lu"] == 24.0 and rep["lufs"] == pytest.approx(done.report["music_lufs"] - 24.0)
    assert rep["xfade_samples"] == 44100


def test_parse_gates_stop_the_render_before_any_audio():
    with pytest.raises(RenderError) as err:
        render((FIX / "g4_kit_trap808.ls").read_text(), EXAMPLE_BRIEF, palette())
    assert err.value.code == "G4"
    with pytest.raises(RenderError) as err:
        render((FIX / "example.ls").read_text(), EXAMPLE_BRIEF, palette(), finish_reason="length")
    assert err.value.code == "G2"


def _cli(tmp_path, **changes):
    args = {"loop": str(FIX / "example.ls"), "--mood": "wistful", "--energy": "mid", "--bpm": "82", "--tonic": "2",
            "--palette": str(ROOT / "palettes" / "dev_gu.yaml"), "--out": str(tmp_path / "x.wav"), **changes}
    return [args.pop("loop")] + [part for flag, value in args.items() for part in (flag, value)] + ["--no-verify"]


@pytest.mark.parametrize("changes, message", [
    ({"--bpm": "100"}, "--bpm 100 is outside 60-95"), ({"--bpm": "59"}, "--bpm 59 is outside 60-95"),
    ({"--bpm": "300"}, "--bpm 300 is outside 60-95"), ({"--tonic": "12"}, "--tonic 12 is not a pitch class"),
    ({"--tonic": "-1"}, "--tonic -1 is not a pitch class"), ({"loop": "missing.ls"}, "missing.ls is not a file"),
    ({"--palette": "nowhere.yaml"}, "nowhere.yaml is not a file"), ({"--cycles": "1"}, "--cycles must be")])
def test_the_cli_exits_2_on_a_bad_argument_before_rendering(tmp_path, capsys, changes, message):
    # docs/RENDER.md §1: exit 2 for bad arguments, 3 for a gate or compile failure, never a traceback.
    with pytest.raises(SystemExit) as err:
        main(_cli(tmp_path, **changes))
    assert err.value.code == 2
    assert message in capsys.readouterr().err
    assert not any(tmp_path.iterdir()), "nothing is written"


def test_the_cli_exits_2_on_a_palette_that_does_not_load(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text((ROOT / "palettes" / "dev_gu.yaml").read_text().replace("sr: 44100", "sr: 48000"))
    with pytest.raises(SystemExit) as err:
        main(_cli(tmp_path, **{"--palette": str(bad), "--out": str(tmp_path / "out" / "x.wav")}))
    assert err.value.code == 2 and "44.1 kHz" in capsys.readouterr().err


def test_gate_g7_on_silent_or_broken_music():
    n = loop_samples(82)
    with pytest.raises(RenderError, match="G7: music at"):
        finish(np.zeros((2, n), dtype=np.float32), palette())
    quiet = np.full((2, n), 1e-5, dtype=np.float32)
    with pytest.raises(RenderError, match="below -50"):
        finish(quiet, palette())
    broken = np.zeros((2, n), dtype=np.float32)
    broken[0, 5] = np.nan
    with pytest.raises(RenderError, match="NaN"):
        finish(broken, palette())
