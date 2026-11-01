"""The rain excerpt picker (docs/SHOOTOUTS.md section 7.2 acceptance tests; section 4.5, CD-20) and its committed
record, shootouts/rain_excerpts.json.

The picker takes the three pairwise disjoint 33.5 s stretches (0.5 s warm-up, 32 s, 1 s crossfade) with the smallest
range of momentary loudness (400 ms blocks), e1 the smallest. Every file here is synthetic noise written into a temp
folder; the real recording is Shresth's download and is never read here (one test re-picks it when the converted
file is present under $LOFI_ASSETS, and skips otherwise)."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from lofi_rl.convert import rain_excerpts as R
from lofi_rl.palette import ROOT
from lofi_rl.render import bed
from lofi_rl.render.loudness import lufs

SR = 44100
COMMITTED = ROOT / "shootouts" / "rain_excerpts.json"
MANIFEST = {a["path"]: a for a in yaml.safe_load((ROOT / "assets" / "MANIFEST.yaml").read_text())["assets"]}


def noise(seconds: float, seed: int = 235827, steady: tuple[float, float] | None = None) -> np.ndarray:
    """(2, n) stereo noise whose level jumps every second by up to 12 dB, held steady over `steady` (seconds)."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * SR))
    x = rng.standard_normal((2, n)) * 0.05
    gains = 10.0 ** (rng.uniform(-12.0, 0.0, size=int(np.ceil(seconds))) / 20.0)
    env = np.repeat(gains, SR)[:n]
    if steady is not None:
        env[int(steady[0] * SR):int(steady[1] * SR)] = 0.5
    return (x * env).astype(np.float32)


def write(path: Path, x: np.ndarray, sr: int = SR) -> Path:
    sf.write(str(path), x.T, sr, subtype="FLOAT")
    return path


def test_three_disjoint_stretches_inside_the_file_e1_the_steady_one():
    x = noise(140.0, steady=(50.0, 90.0))
    got = R.pick(x, SR, 3)
    assert [e["id"] for e in got] == ["e1", "e2", "e3"]
    spans = [tuple(e["samples"]) for e in got]
    for lo, hi in spans:                                   # 33.5 s each, warm-up included, inside the file
        assert hi - lo == R.STRETCH == 1_477_350 and 0 <= lo and hi <= x.shape[1]
    for i, (a0, a1) in enumerate(spans):                   # pairwise disjoint
        for b0, b1 in spans[i + 1:]:
            assert a1 <= b0 or b1 <= a0
    lo, hi = got[0]["stretch_s"]
    assert 50.0 <= lo and hi <= 90.0, "the one steady stretch is e1"
    assert got[0]["range_lu"] < got[1]["range_lu"] <= got[2]["range_lu"]
    assert bed.samples(got[0]["start_s"]) == got[0]["samples"][0] + SR // 2 and got[0]["blocks"] == 332
    assert all(e["max_lufs"] - e["min_lufs"] == pytest.approx(e["range_lu"]) for e in got)
    # start_s is what the bed reads: its warm-up begins at the stretch, on the 100 ms grid
    assert all(round(e["start_s"] * 10) == e["start_s"] * 10 for e in got)


def test_the_picker_is_deterministic_and_the_cli_writes_the_same_bytes_twice(tmp_path, capsys):
    x = noise(104.0, seed=7)
    assert R.pick(x, SR) == R.pick(x.copy(), SR)
    src = write(tmp_path / "rain44.wav", x)
    outs = [tmp_path / "a.json", tmp_path / "b.json"]
    for out in outs:
        assert R.main([str(src), "--n", "3", "--out", str(out)]) == 0
    assert outs[0].read_bytes() == outs[1].read_bytes()
    doc = json.loads(outs[0].read_text())
    assert doc["schema"] == R.SCHEMA and set(doc["excerpts"]) == {"e1", "e2", "e3"}
    s = doc["source"]
    assert s["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest() and s["frames"] == x.shape[1]
    assert s["asset_path"] == "derived/rain/rain44.wav" and s["sr"] == SR and s["channels"] == 2
    for e in doc["excerpts"].values():                     # each stretch pinned by the hash of its own samples
        lo, hi = e["samples"]
        assert e["pcm_sha256"] == hashlib.sha256(np.ascontiguousarray(x[:, lo:hi].T, "<f4").tobytes()).hexdigest()
    assert "e1: start_s" in capsys.readouterr().out


def test_a_file_shorter_than_three_stretches_or_not_at_44k_is_refused(tmp_path, capsys):
    with pytest.raises(R.ExcerptError, match="100.5 s"):
        R.pick(noise(100.4), SR, 3)
    with pytest.raises(R.ExcerptError, match="44100"):
        R.pick(noise(1.0), 48000, 3)
    src48 = tmp_path / "rain48.wav"
    sf.write(str(src48), noise(2.0).T, 48000, subtype="PCM_24")
    assert R.main([str(src48), "--out", str(tmp_path / "x.json")]) == 2
    assert "48000 Hz" in capsys.readouterr().err and not (tmp_path / "x.json").exists()
    short = write(tmp_path / "short.wav", noise(60.0))
    assert R.main([str(short), "--out", str(tmp_path / "y.json")]) == 2 and not (tmp_path / "y.json").exists()


def test_a_small_range_that_would_leave_too_little_room_is_passed_over():
    # 104 s holds three stretches only near 0, 35 and 70 s: a steady stretch at 20-53.5 s, the smallest range, would
    # leave too little room on both sides, so the picker takes the smallest range that still lets the rest fit
    x = noise(104.0, seed=11, steady=(20.0, 53.5))
    got = R.pick(x, SR, 3)
    spans = sorted(tuple(e["samples"]) for e in got)
    assert spans[0][0] < 2 * SR and spans[1][0] >= spans[0][1] and spans[2][0] >= spans[1][1]
    assert R.pick(x, SR, 1)[0]["stretch_s"][0] >= 20.0, "alone, the steady stretch is the smallest range"


def test_stretches_that_reach_digital_silence_are_never_picked():
    x = noise(110.0)
    x[:, 50 * SR:] = 0.0                     # under -70 LUFS from about 50 s: only stretches that end by then count
    with pytest.raises(R.ExcerptError, match="only 1 disjoint"):
        R.pick(x, SR, 3)
    got = R.pick(x, SR, 1)
    assert got[0]["samples"][1] <= 51 * SR


def test_momentary_blocks_gate_to_the_renderers_integrated_loudness():
    # BS.1770: the momentary blocks are what the integrated loudness gates (pyloudnorm, render/loudness.py)
    x = noise(12.0, seed=3)
    blocks = R.momentary(x, SR)
    assert len(blocks) == 12 * 10 - 3
    power = 10.0 ** ((blocks + 0.691) / 10.0)
    keep = blocks >= -70.0
    rel = -0.691 + 10.0 * np.log10(power[keep].mean()) - 10.0
    keep &= blocks > rel
    gated = -0.691 + 10.0 * np.log10(power[keep].mean())
    assert gated == pytest.approx(lufs(x, SR), abs=1e-3)


# ------------------------------------------------------------------------------- the committed record
def test_the_committed_excerpts_are_three_disjoint_stretches_of_the_pinned_44k_file():
    doc = json.loads(COMMITTED.read_text())
    assert doc["schema"] == R.SCHEMA and doc["tool"] == "python -m lofi_rl.convert.rain_excerpts"
    p = doc["parameters"]
    assert (p["n"], p["warmup_s"], p["loop_s"], p["xfade_s"], p["stretch_s"], p["block_s"], p["hop_s"]) == \
        (3, 0.5, 32.0, 1.0, 33.5, 0.4, 0.1)
    src = doc["source"]
    pin = MANIFEST[f"assets/big/{src['asset_path']}"]
    assert (pin["sha256"], pin["license"], pin["mode"], pin["bytes"]) == \
        (src["sha256"], src["license"], "fetch", src["bytes"])
    original = MANIFEST[src["converted_from"]["path"]]
    assert original["sha256"] == src["converted_from"]["sha256"] and src["converted_from"]["sr"] == 48000
    assert (src["sr"], src["channels"], src["frames"]) == (44100, 2, 8_643_600)     # 196.000 s
    ex = doc["excerpts"]
    assert list(ex) == ["e1", "e2", "e3"]
    assert ex["e1"]["range_lu"] <= ex["e2"]["range_lu"] <= ex["e3"]["range_lu"]       # e1 the smallest range
    spans = [tuple(e["samples"]) for e in ex.values()]
    for i, (a0, a1) in enumerate(spans):
        assert a1 - a0 == R.STRETCH and 0 <= a0 and a1 <= src["frames"]
        assert bed.samples(ex[f"e{i + 1}"]["start_s"]) == a0 + SR // 2             # start_s after the warm-up
        for b0, b1 in spans[i + 1:]:
            assert a1 <= b0 or b1 <= a0


def test_the_real_file_gives_the_committed_record_when_it_is_here():
    assets = Path(os.environ.get("LOFI_ASSETS") or ROOT / "assets" / "big")
    doc = json.loads(COMMITTED.read_text())
    path = assets / doc["source"]["asset_path"]
    if not path.is_file():
        pytest.skip(f"{path} is not on this machine (fetch-only; SHOOTOUTS.md section 7.2)")
    got = R.excerpts_doc(path, n=3, asset_path=doc["source"]["asset_path"])
    assert got["source"]["sha256"] == doc["source"]["sha256"]
    assert got["excerpts"] == doc["excerpts"] and got["parameters"] == doc["parameters"]
