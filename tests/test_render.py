"""The whole render (instrument.md §4, §7 test_render): exactly N samples, -14 ± 0.1 LUFS, true peak at or
below -1 dBTP on the loop as it repeats, the bed at -Δ ± 0.5 LU, gate G7, and a manifest that matches the
audio. Every number is measured here, independently of the manifest."""
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import resample_poly

from helpers_compile import EXAMPLE_BRIEF, FIX, ROOT
from helpers_render import needs_fluidsynth, palette, rendered, with_bed
from lofi_rl.brief import Brief
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.dsl import parse
from lofi_rl.dsl.repair import penalty
from lofi_rl.render import pipeline
from lofi_rl.render.formats import write_wav
from lofi_rl.render.loudness import LoudnessInfo, lufs, true_peak_dbtp
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


# ------------------------------------------------------------------------------------- synthetic loops
def _tone(seconds: int = 3, bursts: float = 0.0) -> np.ndarray:
    """A periodic stereo loop of whole seconds: a 110 Hz sine (330 cycles in 3 s) plus, scaled by `bursts`, a
    decaying 1 kHz burst every 0.25 s, which the limiter has to catch. (2, n) float32 at 0.1 of full scale."""
    n = seconds * SR
    x = np.sin(2 * np.pi * 110 * np.arange(n) / SR)
    k = np.arange(2000)
    for start in range(0, n, SR // 4):
        x[start:start + 2000] += bursts * np.exp(-k / 300) * np.sin(2 * np.pi * 1000 * k / SR)
    return (0.1 * np.stack([x, x])).astype(np.float32)


def _at_lufs(x: np.ndarray, level: float) -> np.ndarray:
    return (x.astype(np.float64) * 10 ** ((level - lufs(x, SR)) / 20)).astype(np.float32)


def test_normalization_takes_up_to_the_palettes_iterations():
    # RENDER.md §3.5: at most loudness.iters (4) iterations. The bursts make the limiter work, so one pass lands
    # 0.78 LU low (measured 2026-10-31) and a later one converges. No fixture needs a second pass.
    music, pal = _tone(bursts=5.0), palette()
    done = finish(music, pal)
    assert 2 <= done.report["loudness"]["iterations"] <= pal.loudness.iters == 4
    assert abs(done.report["loudness"]["lufs"] + 14.0) <= 0.1 and abs(lufs(done.audio, SR) + 14.0) <= 0.1
    one = replace(pal, loudness=replace(pal.loudness, iters=1))
    with pytest.raises(RenderError, match=r"G7: not within 0.1 LU of -14.0 after 1 iterations"):
        finish(music, one)


@pytest.mark.parametrize("delta", [18.0, 24.0, 30.0])
def test_the_bed_follows_the_palettes_delta(tmp_path, delta):
    # RENDER.md §3.4: LUFS_bed = LUFS_music - Δ, Δ = bed.delta_lu: the rain shootout's 18, 24 and 30
    # (instrument.md §6). The tone never reaches the limiter, so render minus music.wav is the bed alone.
    music = _tone()
    done = finish(music, with_bed(tmp_path, music.shape[1], delta_lu=delta))
    bed_alone = done.audio.astype(np.float64) - done.music
    assert lufs(bed_alone, SR) - lufs(done.music, SR) == pytest.approx(-delta, abs=0.05)
    rep, gain = done.report["bed"], done.report["loudness"]["gain_db"]
    assert rep["delta_lu"] == delta and rep["lufs"] == pytest.approx(done.report["music_lufs"] - delta)
    assert lufs(bed_alone, SR) == pytest.approx(rep["lufs"] + gain, abs=0.05), "the report's level is the bed's"


@pytest.mark.parametrize("level, passes", [(-55.0, False), (-45.0, True)])
def test_gate_g7_on_music_below_minus_50_lufs(level, passes):
    music = _at_lufs(_tone(), level)
    if passes:
        assert abs(finish(music, palette()).report["loudness"]["lufs"] + 14.0) <= 0.1
    else:
        with pytest.raises(RenderError, match=r"G7: music at -55.00 LUFS is below -50.0"):
            finish(music, palette())


@pytest.mark.parametrize("peak_db, passes", [(-0.70, False), (-0.95, True)])
def test_gate_g7_on_a_true_peak_above_minus_0_9_dbtp(monkeypatch, peak_db, passes):
    # The limiter keeps real renders near -1.5 dBTP, so normalize_loop is replaced by one that returns a loop with
    # a chosen true peak; G7 measures it on the loop as it repeats (instrument.md §2: above -0.9 dBTP fails).
    loud = (10 ** (peak_db / 20) * _tone() * 10).astype(np.float32)
    monkeypatch.setattr(pipeline, "normalize_loop", lambda x, sr, **kw: (loud, LoudnessInfo(-14.0, 0.0, 0.0, 1)))
    if passes:
        assert finish(_tone(), palette()).report["loudness"]["true_peak_dbtp"] == pytest.approx(peak_db, abs=0.01)
    else:
        with pytest.raises(RenderError, match=r"G7: true peak -0\.69 dBTP is above -0\.9"):
            finish(_tone(), palette())


def test_the_loop_true_peak_sees_an_overshoot_at_the_seam():
    # RENDER.md §3.5: measured with the other end of the cycle as context, it equals the true peak of the middle
    # cycle of a 3x tile, where the last and first samples meet.
    n = 64_000
    y = np.zeros((2, n), dtype=np.float32)
    y[:, 0] = y[:, -1] = 0.8
    up = resample_poly(np.tile(y.astype(np.float64), 3), 4, 1, axis=-1)[:, 4 * n:8 * n]
    tiled = 20 * np.log10(np.max(np.abs(up)))
    assert tiled == pytest.approx(0.117, abs=0.001) and true_peak_dbtp(y) < -1.9, "the cycle alone hides it"
    assert loop_true_peak_dbtp(y) == pytest.approx(tiled, abs=1e-9)


# ----------------------------------------------------------------- render(): parse, compile and the manifest
@pytest.fixture
def silent_audio(monkeypatch):
    """render() with its audio stages (FluidSynth, sampler, buses, bed, loudness) replaced by silence, so a test
    of its parse call and manifest runs in milliseconds; parse, compile and the manifest stay real."""
    def music_cycles(compiled, palette, cycles=2, **kw):
        n = compiled.n
        return pipeline.Music(np.zeros((2, cycles * n), np.float32), n, {b: None for b in pipeline.BUSES},
                              {b: "0" * 64 for b in pipeline.BUSES}, {})

    def finish_stub(music, palette, **kw):
        return pipeline.Finished(np.zeros_like(music), np.zeros_like(music), {"stub": True})
    monkeypatch.setattr(pipeline, "music_cycles", music_cycles)
    monkeypatch.setattr(pipeline, "finish", finish_stub)


SAX_FOLDS = """vamp: Im9 IV13 span 2
keys: rhodes rootless
comp: x---------------
bass: lock
drums: jazz
kick: x.......x.....
snare: ....x.......x...
hat: x.x.x.x.x.x.x.x.
A: w1,, | w3 | w5 | w1,,
B: w1 | w3 | w5 | h1 hr
"""


def test_the_manifest_carries_parser_and_compiler_repairs_and_their_p_fix(silent_audio):
    # docs/COMPILER.md §3.2: P_fix = dsl.repair.penalty(parse repairs + compiled repairs). Here the 14-cell kick
    # grid is padded by the parser and the two w1,, notes fold in the compiler.
    res = parse(SAX_FOLDS, vocab=palette().vocab(), mode="dorian")
    r = render(SAX_FOLDS, EXAMPLE_BRIEF, palette())
    codes = [rep["code"] for rep in r.manifest["loop"]["repairs"]]
    assert [rep.code for rep in r.compiled.repairs] == ["sax_fold", "sax_fold"] and codes.count("sax_fold") == 2
    assert codes == [rep.code for rep in res.repairs] + ["sax_fold", "sax_fold"] and len(codes) > 2
    assert r.manifest["loop"]["p_fix"] == penalty(list(res.repairs) + list(r.compiled.repairs)) == pytest.approx(0.06)


@pytest.mark.parametrize("mood, chord", [("wistful", "Im7"), ("melancholy", "Im7"), ("hopeful", "Imaj7")])
def test_render_parses_with_the_briefs_mode(silent_audio, mood, chord):
    # docs/COMPILER.md §1: parse(..., mode=brief.mode). A bare numeral becomes the mode's diatonic 7th chord, a
    # penalized repair rather than a gate (instrument.md §2, leniency step 6).
    text = (FIX / "example.ls").read_text().replace("vamp: Im9 IV13 span 2", "vamp: I IV13 span 2")
    r = render(text, Brief("mid", mood, 82, 2), palette())
    assert r.manifest["loop"]["canon"].splitlines()[0] == f"vamp: {chord} IV13 span 2"
    assert r.compiled.mode == Brief("mid", mood, 82, 2).mode
    assert [rep["code"] for rep in r.manifest["loop"]["repairs"] if rep["penalized"]] == ["bare_numeral"]
    assert r.manifest["loop"]["p_fix"] == pytest.approx(0.02) and r.manifest["gates"]["G1-G6"] == "pass"
