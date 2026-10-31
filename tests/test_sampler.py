"""The drum sampler and its one-shot builder (instrument.md §3 Drum sampler; docs/COMPILER.md §3.9, §4.4):
velocity layers, round-robin, the open-hat choke, the fold at the seam, the pan law, and the cached
FluidSynth renders of the GU Jazz kit."""
import json
import math
import zipfile
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from helpers_compile import EXAMPLE_BRIEF, dev_palette, loop_of
from helpers_render import needs_fluidsynth, palette
from lofi_rl.compile.compiler import compile_loop
from lofi_rl.compile.events import DrumHit
from lofi_rl.palette import OneShot, Source
from lofi_rl.render import fluid, sampler
from lofi_rl.render.sampler import (CHOKE, SamplerError, choke, file_shot, kit_shots, mix_hits, note_settings,
                                    note_shot, pan_gains, render_note, settings_key)
from lofi_rl.render.wavio import pcm_sha256, read_float, write_float

CENTRE = math.sqrt(0.5)
LANES = {lane: (0.0, 0.0) for lane in ("kick", "snare", "rim", "hat", "ohat", "shaker")}


def hit(lane, on, layer="med", rr=0, gain=1.0, step=0, choke_at=None):
    return DrumHit(lane, {"hard": "X", "med": "x", "soft": "g"}[layer], layer, on, gain, rr, step, choke_at)


def impulse(height, length=4):
    x = np.zeros(length)
    x[0] = height
    return x


def test_each_hit_plays_its_layer_and_round_robin_one_shot():
    shots = {("hat", "soft"): (impulse(1.0), impulse(2.0)), ("hat", "med"): (impulse(3.0),),
             ("hat", "hard"): (impulse(4.0), impulse(5.0), impulse(6.0))}
    plays = [("soft", 0, 1.0), ("soft", 1, 2.0), ("med", 0, 3.0), ("hard", 0, 4.0), ("hard", 1, 5.0),
             ("hard", 2, 6.0)]
    hits = [hit("hat", 10 * i, layer, rr, gain=0.5, step=i) for i, (layer, rr, _) in enumerate(plays)]
    y = mix_hits(hits, shots, LANES, 100)
    assert y.shape == (2, 100) and y.dtype == np.float32
    for i, (_, _, height) in enumerate(plays):
        assert y[0, 10 * i] == y[1, 10 * i] == np.float32(height * 0.5 * CENTRE)
    assert np.count_nonzero(y[0]) == len(plays)


def test_compiled_round_robin_reaches_the_sampler():
    # A med hat layer of 3 one-shots: hits take them in step order, 0 1 2 0 1 2 ... (docs/COMPILER.md §3.9).
    pal = dev_palette()
    kit = pal.kits["jazz"]
    hat = kit.lanes["hat"]
    three = tuple(replace(hat.layers["med"][0], vel=v) for v in (76, 80, 84))
    lanes = dict(kit.lanes)
    lanes["hat"] = replace(hat, layers=MappingProxyType({**hat.layers, "med": three}))
    rr_pal = replace(pal, kits=MappingProxyType({"jazz": replace(kit, lanes=MappingProxyType(lanes))}))
    c = compile_loop(loop_of("example.ls"), EXAMPLE_BRIEF, rr_pal)
    shots = {(lane, layer): tuple(impulse(1.0) for _ in rr_pal.kits["jazz"].lanes[lane].layers[layer])
             for lane in LANES for layer in ("soft", "med", "hard")}
    shots[("hat", "med")] = (impulse(10.0), impulse(20.0), impulse(30.0))
    hats = [d for d in c.drums if d.lane == "hat"]
    y = mix_hits(hats, shots, LANES, c.n)
    med = sorted((d for d in hats if d.layer == "med"), key=lambda d: d.step)
    assert len(med) > 6 and [d.rr for d in med] == [i % 3 for i in range(len(med))]
    for d in med:
        assert y[0, d.on] == np.float32(10.0 * (d.rr + 1) * d.gain * CENTRE)


def test_a_layer_with_no_one_shots_falls_back_to_the_softer_neighbour():
    # The hat's x hits want the med layer; with it empty, soft and hard are equally near and soft wins.
    pal = dev_palette()
    kit = pal.kits["jazz"]
    hat = kit.lanes["hat"]
    lanes = dict(kit.lanes)
    lanes["hat"] = replace(hat, layers=MappingProxyType({**hat.layers, "med": ()}))
    gap_pal = replace(pal, kits=MappingProxyType({"jazz": replace(kit, lanes=MappingProxyType(lanes))}))
    c = compile_loop(loop_of("example.ls"), EXAMPLE_BRIEF, gap_pal)
    hats = [d for d in c.drums if d.lane == "hat"]
    assert {d.level for d in hats} == {"x"} and {d.layer for d in hats} == {"soft"}
    shots = {("hat", "soft"): (impulse(1.0),), ("hat", "hard"): (impulse(7.0),)}
    y = mix_hits(hats, shots, LANES, c.n)
    for d in hats:
        assert y[0, d.on] == np.float32(1.0 * d.gain * CENTRE)
    with pytest.raises(SamplerError, match="no one-shot"):
        mix_hits([hit("hat", 0, "med")], shots, LANES, c.n)


def test_the_open_hat_choke_fades_over_8_ms():
    assert CHOKE == 353                                   # 8 ms at 44.1 kHz, rounded half up
    shot = np.ones(2000)
    y = mix_hits([hit("ohat", 100, choke_at=600)], {("ohat", "med"): (shot,)}, LANES, 5000)[0] / CENTRE
    assert np.allclose(y[100:600], 1.0, rtol=0, atol=1e-7)
    fade = (CHOKE - np.arange(CHOKE)) / CHOKE
    assert np.allclose(y[600:600 + CHOKE], fade, rtol=0, atol=1e-7)
    assert not np.any(y[600 + CHOKE:]) and not np.any(y[:100])


def test_a_choke_in_the_next_cycle_wraps_with_the_tail():
    n, shot = 1000, np.ones(2000)
    y = mix_hits([hit("ohat", 900, choke_at=1200)], {("ohat", "med"): (shot,)}, LANES, n)[0] / CENTRE
    assert np.allclose(y[900:], 1.0, atol=1e-7) and np.allclose(y[:200], 1.0, atol=1e-7)
    assert np.allclose(y[200:200 + CHOKE], (CHOKE - np.arange(CHOKE)) / CHOKE, atol=1e-7)
    assert not np.any(y[200 + CHOKE:900])


def test_choke_leaves_a_shot_that_ends_first_alone():
    shot = np.linspace(1, 0, 300)
    assert choke(shot, 300) is shot and choke(shot, 5000) is shot
    with pytest.raises(SamplerError):
        choke(shot, 0)


def test_tails_fold_back_onto_the_cycle():
    n = 1000
    y = mix_hits([hit("kick", 0)], {("kick", "med"): (np.ones(2500),)}, LANES, n)[0] / CENTRE
    assert np.allclose(y[:500], 3.0, atol=1e-6) and np.allclose(y[500:], 2.0, atol=1e-6)
    tail = mix_hits([hit("snare", n - 100)], {("snare", "med"): (np.ones(300),)}, LANES, n)[0] / CENTRE
    assert np.allclose(tail[n - 100:], 1) and np.allclose(tail[:200], 1) and not np.any(tail[200:n - 100])


def test_constant_power_pan_is_exact_at_the_centre_and_edges():
    assert pan_gains(-1.0) == (1.0, 0.0) and pan_gains(1.0) == (0.0, 1.0)
    left, right = pan_gains(0.0)
    assert left == right == CENTRE
    for pan in np.linspace(-1, 1, 41):
        gl, gr = pan_gains(float(pan))
        assert abs(gl * gl + gr * gr - 1.0) < 1e-15
        lo, hi = pan_gains(-float(pan))
        assert (lo, hi) == (gr, gl), "the law is symmetric"
    with pytest.raises(SamplerError):
        pan_gains(1.5)
    lanes = dict(LANES, kick=(-1.0, -6.0))
    y = mix_hits([hit("kick", 0)], {("kick", "med"): (impulse(1.0),)}, lanes, 10)
    assert y[1, 0] == 0.0 and y[0, 0] == np.float32(10 ** (-6 / 20))


def test_the_sum_does_not_depend_on_the_order_hits_arrive_in():
    rng_free = [hit("hat", (37 * k) % 900, step=k, gain=0.1 + 0.01 * k) for k in range(40)]
    shots = {("hat", "med"): (np.sin(np.arange(700) * 0.3) * np.exp(-np.arange(700) / 100.0),)}
    a = mix_hits(rng_free, shots, LANES, 1000)
    b = mix_hits(list(reversed(rng_free)), shots, LANES, 1000)
    assert np.array_equal(a, b)
    with pytest.raises(SamplerError, match="outside the cycle"):
        mix_hits([hit("hat", 1000)], shots, LANES, 1000)


@needs_fluidsynth
def test_note_settings_key_every_input():                 # the settings include FluidSynth's runtime version
    shot = OneShot("gu", 128, 32, 38, 80)
    base = settings_key(note_settings(shot, "a" * 64))
    assert base == settings_key(note_settings(shot, "a" * 64))
    for other in (replace(shot, vel=81), replace(shot, note=37), replace(shot, prog=0)):
        assert settings_key(note_settings(other, "a" * 64)) != base
    assert settings_key(note_settings(shot, "b" * 64)) != base
    keys = set(note_settings(shot, "a" * 64))
    assert {"sf2_sha256", "fluidsynth", "machine", "hold", "gain", "sr", "builder"} <= keys


def test_file_one_shots_come_from_a_directory_or_a_zip(tmp_path):
    x = np.stack([np.linspace(-0.5, 0.5, 300), np.linspace(0.5, -0.5, 300)]).astype(np.float32)
    kit = tmp_path / "kit"
    kit.mkdir()
    write_float(kit / "hat.wav", x, 44100)
    with zipfile.ZipFile(tmp_path / "kit.zip", "w") as z:
        z.write(kit / "hat.wav", "lanes/hat.wav")
    pal = replace(dev_palette(), assets_root=str(tmp_path), sources=MappingProxyType({
        "dir": Source("dir", "kit", "0" * 64, "CC0-1.0", "fetch", None, None, None),
        "zip": Source("zip", "kit.zip", "0" * 64, "CC0-1.0", "fetch", None, None, None)}))
    a, ha = file_shot(OneShot("dir", file="hat.wav"), pal)
    b, hb = file_shot(OneShot("zip", file="lanes/hat.wav"), pal)
    assert np.array_equal(a, x) and np.array_equal(b, x) and ha == hb == pcm_sha256(x)
    assert np.allclose(sampler.mono(a), 0.0, atol=1e-7), "a stereo one-shot plays as (L + R) / 2"
    with pytest.raises(SamplerError, match="outside"):
        file_shot(OneShot("dir", file="../kit.zip"), pal)
    write_float(kit / "slow.wav", x, 48000)
    with pytest.raises(SamplerError, match="48000 Hz"):
        file_shot(OneShot("dir", file="slow.wav"), pal)


@needs_fluidsynth
def test_the_dev_kit_renders_every_one_shot_once():
    kit = kit_shots(palette(), "jazz")
    assert set(kit.shots) == {(lane, layer) for lane in LANES for layer in ("soft", "med", "hard")}
    for key, shots in kit.shots.items():
        (x,) = shots
        assert x.ndim == 1 and x.size > 1000 and np.all(np.isfinite(x)), key
        assert np.flatnonzero(x)[0] <= 2, f"{key}: the one-shot starts at its onset"
        assert len(kit.pcm[key]) == 1 and len(kit.pcm[key][0]) == 64
    for lane in LANES:
        soft, hard = (np.abs(kit.shots[(lane, layer)][0]).max() for layer in ("soft", "hard"))
        assert hard > soft, f"{lane}: velocity 112 peaks above velocity 48"
    assert kit.shots[("ohat", "med")][0].size > kit.shots[("hat", "med")][0].size
    assert kit.lanes == LANES


@needs_fluidsynth
def test_the_builder_caches_verifies_and_rebuilds(tmp_path, monkeypatch):
    shot, pal = OneShot("gu", 128, 32, 38, 80), palette()
    x, digest = note_shot(shot, pal, cache=tmp_path)
    assert digest == pcm_sha256(x) and x.shape[0] == 2
    (meta,) = (tmp_path / "oneshots").glob("*.json")
    record = json.loads(meta.read_text())
    assert record["pcm_sha256"] == digest and record["note"] == 38 and record["sf2_sha256"] == pal.sources["gu"].sha256
    assert meta.stem == settings_key(note_settings(shot, pal.sources["gu"].sha256))

    calls = []
    real = fluid.render_midi

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)
    monkeypatch.setattr(fluid, "render_midi", counting)
    again, d2 = note_shot(shot, pal, cache=tmp_path)
    assert calls == [] and d2 == digest and np.array_equal(again, x), "a verified entry is reused"

    wav = meta.with_suffix(".wav")
    write_float(wav, x * np.float32(0.5), 44100)            # corrupt the cached samples
    fixed, d3 = note_shot(shot, pal, cache=tmp_path)
    assert len(calls) == 1 and d3 == digest and np.array_equal(fixed, x), "a corrupt entry is rebuilt"
    y, _ = read_float(wav)
    assert pcm_sha256(y) == digest


@needs_fluidsynth
def test_bank_128_on_the_drum_channel_selects_the_jazz_kit():
    pal = palette()
    jazz, _ = note_shot(OneShot("gu", 128, 32, 38, 80), pal)
    standard = render_note(OneShot("gu", 128, 0, 38, 80), pal.source_path("gu"))
    assert jazz.shape != standard.shape or not np.array_equal(jazz, standard), "program 32 is not program 0"
    with pytest.raises(fluid.FluidSynthError, match=r"bank=128 prog=33\], substituted"):
        render_note(OneShot("gu", 128, 33, 38, 80), pal.source_path("gu"))    # GU has no drum kit 33


@needs_fluidsynth
def test_a_fluidsynth_user_config_never_reaches_the_one_shot_cache(tmp_path, monkeypatch):
    shot, pal = OneShot("gu", 128, 32, 37, 48), palette()
    clean = render_note(shot, pal.source_path("gu"))
    home = tmp_path / "home"
    home.mkdir()
    (home / ".fluidsynth").write_text("gain 0.6\ninterp 0\n")
    monkeypatch.setenv("HOME", str(home))
    # An entry an earlier builder may have written under such a config is keyed by the old builder, so it is
    # never looked up again.
    stale = dict(note_settings(shot, pal.sources["gu"].sha256), builder="fluid-note/1")
    (tmp_path / "cache" / "oneshots").mkdir(parents=True)
    poisoned = (clean * np.float32(3.0)).astype(np.float32)
    write_float(tmp_path / "cache" / "oneshots" / f"{settings_key(stale)}.wav", poisoned, 44100)
    (tmp_path / "cache" / "oneshots" / f"{settings_key(stale)}.json").write_text(
        json.dumps(dict(stale, pcm_sha256=pcm_sha256(poisoned))))
    x, digest = note_shot(shot, pal, cache=tmp_path / "cache")
    assert digest == pcm_sha256(clean) and np.array_equal(x, clean)
