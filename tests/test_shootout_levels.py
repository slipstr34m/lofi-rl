"""The level-matching tool (docs/SHOOTOUTS.md §3 steps 1-6 and CD-52; §7.1 acceptance tests 4-7; no palette field yet).

Tests 4 and 7 run on tables of made-up loudness values (no audio), the linear-range check (5) on one FluidSynth
stem and one synthetic signal, and the end-to-end match (6) on the real banks with `-m cluster`. Every shootout
clip takes its own fader on the bus its round varies (CD-52), so the tool also measures each (candidate, loop) a
clip plays, the anchor's sour copy included when it is given the job's draw."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import statistics
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import yaml

from helpers_shootout import BASE_PATH, FILE_KITS, PLAN_PATH, ROOT, kit_archive, loops_subset, synthetic_base
from lofi_rl.brief import Brief
from lofi_rl.compile.compiler import compile_loop
from lofi_rl.dsl import parse
from lofi_rl.palette import Plugin, load_palette
from lofi_rl.render.loudness import lufs
from lofi_rl.render.pipeline import fluid_stems
from lofi_rl.shootout.levels import (CLIP_FIX_LU, CLIP_STEPS, LevelsError, chain_lufs, clip_check, clip_fader, faders,
                                     input_lufs, linear_check, main, post_check, run_levels, sax_target, trims,
                                     with_trim)
from lofi_rl.shootout.palettes import generate
from lofi_rl.shootout.plan import load_plan
from lofi_rl.shootout.sour import raise_accidentals

PLAN = load_plan(PLAN_PATH)
needs_fluidsynth = pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth not on PATH")
TWO = ("l1_melancholy_low", "l9_hopeful_high")


# --------------------------------------------------------------------------- §3 steps 2-5 on tables
P = {   # integrated loudness of each bus input, by loop; 4 loops, so every median is the mean of the middle two
    "gu_tine_ep": {"a": -36.0, "b": -35.0, "c": -34.0, "d": -30.0},
    "jrhodes3d": {"a": -20.0, "b": -22.0, "c": -21.0, "d": -25.0},
    "gu_organ": {"a": -28.0, "b": -27.0, "c": -26.5, "d": -29.0},
    "sax_gu_tenor": {"a": -39.0, "b": -38.0, "c": -40.0, "d": -37.0},
    "sax_mtg_alto": {"a": -18.0, "b": -19.5, "c": -17.25, "d": -18.8},
}
GAIN = {"gu_tine_ep": 5.0, "jrhodes3d": 3.0, "gu_organ": 4.0, "sax_gu_tenor": 7.0, "sax_mtg_alto": 9.0}   # chains
BUS = {"gu_tine_ep": "keys", "jrhodes3d": "keys", "gu_organ": "keys", "sax_gu_tenor": "sax", "sax_mtg_alto": "sax"}
ANCHORS = {"keys": "gu_tine_ep", "sax": "sax_gu_tenor"}
GU = {"gu_tine_ep", "gu_organ", "sax_gu_tenor"}


def test_trims_and_faders_from_a_synthetic_table():
    t = trims(P, BUS, ANCHORS, zero=GU)
    med = statistics.median
    assert t["jrhodes3d"] == round(med(P["gu_tine_ep"].values()) - med(P["jrhodes3d"].values()), 2) == -13.0
    assert t["sax_mtg_alto"] == round(med(P["sax_gu_tenor"].values()) - med(P["sax_mtg_alto"].values()), 2) == -20.1
    assert t["gu_tine_ep"] == t["gu_organ"] == t["sax_gu_tenor"] == 0.0, "GU instruments get t = 0"
    assert med([-36.0, -35.0, -34.0, -30.0]) == -34.5 != statistics.median_low([-36.0, -35.0, -34.0, -30.0])
    L = {v: {k: x + GAIN[v] + t[v] for k, x in P[v].items()} for v in P}   # after the chain, with the trim
    s_star = sax_target(L, "sax_gu_tenor")
    assert s_star == med(L["sax_gu_tenor"].values()) == -31.5
    targets = {"sax": 0.0, "keys": -6.0}
    F = faders(L, BUS, s_star, targets)
    for v in P:
        assert F[v] == round(s_star + targets[BUS[v]] - med(L[v].values()), 2), v
    assert F["sax_gu_tenor"] == 0.0, "the sax bus stays where dev_gu puts it"
    assert (F["jrhodes3d"], F["gu_organ"], F["gu_tine_ep"]) == (-6.0, -14.0, -8.0)
    assert sax_target(L, "sax_mtg_alto") == med(L["sax_mtg_alto"].values()) != s_star, "the named anchor decides"
    with pytest.raises(LevelsError, match="anchor"):
        trims({k: v for k, v in P.items() if k != "gu_tine_ep"}, BUS, ANCHORS, zero=GU)
    with pytest.raises(LevelsError, match="jrhodes3d"):
        trims(dict(P, jrhodes3d={"a": float("-inf"), "b": -1.0}), BUS, ANCHORS, zero=GU)


def test_the_median_check_holds_the_median_fader_to_005_lu():
    # §3 step 6b: the median fader F, the one the frozen palette takes, puts the median post-fader loudness at
    # S* + T within 0.05 LU. How far each loop moved from its own L + F through BS.1770's -70 LUFS gate is kept
    # as information: a shootout clip plays its own fader (step 6c), not F.
    L = {"x": {"a": -40.0, "b": -30.0, "c": -35.0}}
    F, s_star, T = {"x": 3.0}, -32.0, 0.0
    exact = {"x": {k: v + 3.0 for k, v in L["x"].items()}}
    assert post_check(exact, L, F, {"x": s_star + T})["x"]["ok"]
    gated = {"x": {"a": -36.93, "b": -27.0, "c": -32.04}}              # a few hundredths through the -70 gate
    r = post_check(gated, L, F, {"x": s_star + T})["x"]
    assert r["ok"] and r["median_lufs"] == -32.04 and r["worst_loop_lu"] == pytest.approx(0.07)
    off = {"x": {"a": -36.8, "b": -27.0, "c": -32.0}}                   # loop a 0.2 LU off its own L + F
    r = post_check(off, L, F, {"x": s_star + T})["x"]
    assert r["ok"] and r["worst_loop_lu"] == pytest.approx(0.2), "no longer a flag: loop a's clips take their own"
    median_off = {"x": {"a": -36.94, "b": -26.94, "c": -31.94}}         # each 0.06 off: the median breaks 0.05
    assert not post_check(median_off, L, F, {"x": s_star + T})["x"]["ok"]


def test_a_clips_fader_is_corrected_by_the_loudness_it_measures_until_the_clip_sits_on_its_target():
    # §3 step 5 per clip (CD-52): F(i, l) = S* + T - L(i, l), rounded to 0.01 dB, then corrected by the loudness
    # the bus measures after that fader. A pure gain moves a quiet bus through BS.1770's -70 LUFS gate: the Day-1
    # dry run's kit_virtuosity drum bus on L1 (-55.0 LUFS before the fader) measured 0.101 LU under its L + F.
    calls = []

    def gated(F):                                       # 0.1 LU under L + F, as that bus measured
        calls.append(F)
        return -55.0 + F - 0.1

    c = clip_fader(-55.0, -20.0, gated)
    assert calls == [35.0, 35.1] and c["F"] == 35.1 and c["tried"] == 2
    assert c["lufs"] == pytest.approx(-20.0, abs=1e-9) and c["off_lu"] == pytest.approx(0.0, abs=1e-9)
    assert (c["L"], c["target_lufs"]) == (-55.0, -20.0)
    loud = clip_fader(-31.234, -26.5, lambda F: -31.234 + F)            # no gate: one measurement
    assert loud["F"] == 4.73 and loud["tried"] == 1 and abs(loud["off_lu"]) <= CLIP_FIX_LU
    growing = clip_fader(-40.0, -20.0, lambda F: -40.0 + F - 0.02 * (F - 10.0))   # the gate grows with the gain
    assert growing["F"] == 20.2 and growing["tried"] == 2 and abs(growing["off_lu"]) <= CLIP_FIX_LU
    # out of corrections, the fader whose measured loudness sat closest wins
    worse = clip_fader(-30.0, -20.0, lambda F: -30.0 + F - (0.1 if F == 10.0 else -0.3), steps=1)
    assert worse["F"] == 10.0 and worse["tried"] == 2 and worse["off_lu"] == pytest.approx(-0.1)
    assert clip_fader(-30.0, -20.0, lambda F: -30.0 + F + 0.3)["tried"] <= 1 + CLIP_STEPS
    with pytest.raises(LevelsError, match="silent"):
        clip_fader(float("-inf"), -20.0, gated)


def test_the_clip_check_holds_every_clip_to_01_lu():
    # §3 step 6c: every clip, at its own fader, sits at S* + T within 0.1 LU.
    clips = {"x": {"a": {"off_lu": 0.004}, "b": {"off_lu": -0.09}},
             "y": {"a": {"off_lu": 0.0}, "a.sour": {"off_lu": 0.12}}}
    out = clip_check(clips, 0.1)
    assert out["x"] == {"worst_off_lu": 0.09, "cases": 2, "off": [], "ok": True}
    assert out["y"] == {"worst_off_lu": 0.12, "cases": 2, "off": ["a.sour"], "ok": False}


# --------------------------------------------------------------------------- §3 step 6a, the linear range
@needs_fluidsynth
def test_the_linear_range_check_passes_a_gu_stem_and_flags_an_input_that_drives_the_compressor():
    base = load_palette(BASE_PATH, verify=False)
    b = yaml.safe_load((ROOT / "shootouts" / "loops" / "briefs.yaml").read_text())["loops"]["l9_hopeful_high"]
    brief = Brief(energy=b["energy"], mood=b["mood"], bpm=b["bpm"], tonic=b["tonic"])
    res = parse((ROOT / "shootouts" / "loops" / "l9_hopeful_high.ls").read_text(), vocab=base.vocab(), mode=brief.mode)
    c = compile_loop(res.ast, brief, base)
    c = replace(c, bass=replace(c.bass, notes=()), sax=replace(c.sax, notes=()))    # one FluidSynth process
    stem = fluid_stems(c, base, 2, workers=1)["keys"]
    chain, n = base.mix.buses["keys"], c.n
    L, _ = chain_lufs(chain, stem, 0.0, n)
    L_low, _ = chain_lufs(chain, stem, -6.02, n)
    gu = linear_check({"l9": L}, {"l9": L_low})
    assert gu["ok"] and abs(gu["median_drop_db"] - 6.02) <= 0.1
    k = np.arange(2 * 44100)                                                        # two 1 s cycles
    hot = np.tile(np.sin(2 * np.pi * 220.0 * k / 44100.0).astype(np.float32), (2, 1))   # 0 dBFS, 18 dB over
    L, _ = chain_lufs(chain, hot, 0.0, 44100)
    L_low, _ = chain_lufs(chain, hot, -6.02, 44100)
    flagged = linear_check({"hot": L}, {"hot": L_low})
    assert not flagged["ok"] and flagged["median_drop_db"] < 5.0, "the compressor takes part of the 6.02 dB"


def test_the_input_loudness_does_not_depend_on_the_minus_70_lufs_gate():
    # A converted bank can render its bus input below BS.1770's -70 LUFS absolute gate (the VSCO 2 pizz bass on L4
    # and L6, 2026-10-31), where the plain integrated loudness is -inf. P is read 40 dB up, then 40 dB is taken off.
    k = np.arange(4 * 44100)
    burst = ((k // 11025) % 2 == 0) * np.sin(2 * np.pi * 110.0 * k / 44100.0)    # 0.25 s on, 0.25 s off
    loud = np.tile((0.05 * burst).astype(np.float32), (2, 1))
    plain = lufs(loud, 44100)
    assert input_lufs(loud) == pytest.approx(plain, abs=0.01)
    quiet = (loud * np.float32(10 ** (-50 / 20))).astype(np.float32)
    assert lufs(quiet, 44100) == float("-inf") or lufs(quiet, 44100) < plain - 50.5
    assert input_lufs(quiet) == pytest.approx(plain - 50.0, abs=0.01)
    with pytest.raises(LevelsError, match="silent"):
        input_lufs(np.zeros((2, 44100), dtype=np.float32))


def test_with_trim_replaces_the_trim_slot_and_refuses_a_chain_without_one():
    base = load_palette(BASE_PATH, verify=False)
    chain = with_trim(base.mix.buses["sax"], -4.5)
    assert chain[0] == Plugin("Gain", MappingProxyType({"gain_db": -4.5})) and chain[1:] == base.mix.buses["sax"][1:]
    with pytest.raises(LevelsError, match="trim slot"):
        with_trim(base.mix.buses["drums"], 1.0)


# ---------------------------------------------------------------------- the tool on a table bench (no audio)
class TableBench:
    """Made-up bus loudness per (variant, loop): the chain is a pure gain (L = L0 + trim), or squashes 3:1 for a
    variant in `squash`; the fader adds its gain plus `gate` (the -70 LUFS gate's few hundredths)."""

    def __init__(self, P, L0, squash=(), gate=0.0):
        self.P, self.L0, self.squash, self.gate = P, L0, set(squash), gate
        self.prepared, self.kept = [], {}

    def prepare(self, run, cases, buses):
        self.prepared.append((run.id, tuple(c.name for c in cases), tuple(buses)))

    def input_lufs(self, run, case, bus):
        return self.P[run.instruments[bus]][case.name]

    def chain_lufs(self, run, case, bus, trim, keep=False):
        v = run.instruments[bus]
        out = self.L0[v][case.name] + (trim / 3 if v in self.squash else trim)
        if keep:
            self.kept[(run.id, case.name, bus)] = out
        return out

    def post_lufs(self, run, case, bus, fader):
        return self.kept[(run.id, case.name, bus)] + fader + self.gate

    def release(self, run):
        pass


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    """Synthetic assets, measuring palettes for Day 1 and a two-loop folder."""
    tmp = tmp_path_factory.mktemp("levels")
    assets = tmp / "assets"
    base = synthetic_base(tmp, assets)
    for kit in FILE_KITS:
        kit_archive(assets, kit)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOFI_ASSETS", str(assets))
        measure = generate(PLAN_PATH, tmp / "palettes", stage="day1", levels=None, base=base)
    variants = sorted(measure.index["variants"])
    P_ = {v: {loop: -45.0 + 0.5 * i + 1.5 * j for j, loop in enumerate(TWO)} for i, v in enumerate(variants)}
    L0 = {v: {loop: -38.0 + 0.25 * i - 0.75 * j for j, loop in enumerate(TWO)} for i, v in enumerate(variants)}
    return {"tmp": tmp, "assets": assets, "base": base, "measure": tmp / "palettes" / "measure",
            "loops": loops_subset(tmp / "loops", TWO), "P": P_, "L0": L0, "variants": variants}


@pytest.fixture(autouse=True)
def rig_env(request, monkeypatch):
    if "rig" in request.fixturenames:
        monkeypatch.setenv("LOFI_ASSETS", str(request.getfixturevalue("rig")["assets"]))


def levels(rig, bench, **kw):
    return run_levels(PLAN_PATH, kw.pop("loops", rig["loops"]), kw.pop("measure", rig["measure"]), "day1",
                      bench=bench, stamp=kw.pop("stamp", "a1b2c3d 0 0"), **kw)


def test_the_tool_measures_each_variant_once_and_writes_t_f_and_the_checks(rig):
    bench = TableBench(rig["P"], rig["L0"], squash={"sax_mtg_alto"}, gate=0.02)
    doc = levels(rig, bench)
    assert [r for r, _, _ in bench.prepared] == [f"m{i}" for i in range(1, 9)], "the anchor run first"
    assert set(doc["variants"]) == set(rig["variants"])
    lv = PLAN.levels
    v = doc["variants"]
    s_star = statistics.median(rig["L0"]["sax_gu_tenor"].values())
    assert doc["S_star"] == s_star and doc["sax_target"] == "sax_gu_tenor"
    for vid, rec in v.items():
        anchor = lv.anchors[rec["bus"]]
        want_t = 0.0 if rec["source"] == "gu" else round(
            statistics.median(rig["P"][anchor].values()) - statistics.median(rig["P"][vid].values()), 2)
        assert rec["t"] == want_t, vid
        assert set(rec["P"]) == set(rec["L"]) == set(TWO)
        assert rec["T"] == {"sax": 0.0, "drums": -1.5, "bass": -3.0, "keys": -6.0}[rec["bus"]]
        assert rec["F"] == round(s_star + rec["T"] - statistics.median(rec["L"].values()), 2), vid
        assert rec["run"] in {r["run"] for r in json.loads((rig["measure"] / "index.json").read_text())["runs"]}
    assert not v["sax_mtg_alto"]["checks"]["linear"]["ok"], "a squashing chain is flagged"
    assert all(rec["checks"]["linear"]["ok"] for k, rec in v.items() if k != "sax_mtg_alto")
    assert all(rec["checks"]["post_fader"]["ok"] for k, rec in v.items() if k != "sax_mtg_alto")
    assert doc["flags"] == ["sax_mtg_alto"]
    # CD-52: each (variant, loop) the tool measured has its own fader, corrected for the bench's +0.02 LU gate
    for vid, rec in v.items():
        target = s_star + rec["T"]
        assert set(rec["clips"]) == set(TWO), vid
        for loop, c in rec["clips"].items():
            assert c["L"] == rec["L"][loop] and c["target_lufs"] == target, (vid, loop)
            assert c["lufs"] == pytest.approx(c["L"] + c["F"] + 0.02) and c["F"] == round(c["F"], 2)
            assert abs(c["off_lu"]) <= CLIP_FIX_LU + 1e-9 and c["off_lu"] == pytest.approx(c["lufs"] - target)
            assert c["lu"] == pytest.approx(c["lufs"] - s_star) and c["tried"] == 2, (vid, loop)
        assert rec["checks"]["clips"]["ok"] and rec["checks"]["clips"]["cases"] == 2
    assert doc["tolerances"]["clip_tol_lu"] == 0.1 and doc["extra_cases"] == {} and doc["draw"] is None
    assert doc["drift"]["stamp"] == "a1b2c3d 0 0"
    assert doc["drift"]["base_sha256"] == hashlib.sha256(rig["base"].read_bytes()).hexdigest()
    assert set(doc["drift"]["digests"]) == {f"m{i}" for i in range(1, 9)}
    assert all(set(d) == set(TWO) for d in doc["drift"]["digests"].values())
    assert doc["prior"] is None and doc["reused"] is False


def tiny_draw(path: Path, victim: str, loop: str, *, renders_on: str | None = None) -> Path:
    """What the level tool reads of the job's draw.json (lofi_rl.shootout.draw): an S1 round's renders on `loop` with
    the GU tenor and `victim`, and the anchor's sour copy of it for `victim`, each with its clip palette's `level`
    (the bus, the option and the case its fader is measured on). `renders_on` puts the GU tenor's clip on another
    loop."""
    lp = PLAN.loop_named(loop)
    renders, palettes = {}, {}
    for opt, sour, where in ((victim, False, loop), (victim, True, loop), ("sax_gu_tenor", False, renders_on or loop)):
        wl = PLAN.loop_named(where)
        text = PLAN.loop_text(wl.id)
        b = wl.brief
        key = f"s1_{opt}_{wl.id.lower()}" + ("_sour" if sour else "")
        renders[f"S1.{where}.{opt}" + (".sour" if sour else "")] = {
            "loop": where, "brief": {"energy": b.energy, "mood": b.mood, "bpm": b.bpm, "tonic": b.tonic},
            "text": raise_accidentals(text) if sour else text, "variant": "sour" if sour else "base",
            "palette": key, "stage": "S1", "option": opt, "what": "sour copy" if sour else "sax source", "bus": "sax"}
        palettes[key] = {"stage": "S1", "day": "day1", "fills": "sax", "option": opt, "base": f"s1_{opt}",
                         "level": {"bus": "sax", "variant": opt, "case": where + (".sour" if sour else "")}}
    assert lp.brief.bpm == renders[f"S1.{loop}.{victim}"]["brief"]["bpm"]
    path.write_text(json.dumps({"schema": "lofi-rl/shootout-draw/1", "stage": "day1",
                                "plan": {"path": str(PLAN_PATH), "sha256": PLAN.sha256},
                                "renders": renders, "palettes": palettes}))
    return path


def test_with_the_draw_the_tool_measures_the_anchors_sour_copy_in_its_candidates_run(rig, tmp_path):
    # CD-52: every clip's varied-bus fader comes from its candidate's bus loudness on the round's own loop. The
    # anchor's sour copy plays its own text, a (candidate, loop) pair the reference loops do not hold, so the draw's
    # sour clip joins the measure plan: once, in its candidate's run, with that candidate's trim. The trim and the
    # median fader (the frozen palette's) still come from the reference loops alone.
    loop, victim = "l9_hopeful_high", "sax_mtg_alto"
    sour = f"{loop}.sour"
    L0 = {v: dict(t) for v, t in rig["L0"].items()}
    L0[victim][sour] = L0[victim][loop] - 1.25
    bench = TableBench(rig["P"], L0, gate=0.02)
    path = tiny_draw(tmp_path / "draw.json", victim, loop)
    doc = levels(rig, bench, draw=path)
    rec = doc["variants"][victim]
    run = rec["run"]
    extra = [p for p in bench.prepared if p[1] != TWO]
    assert extra == [(run, (sour,), ("sax",))], "the sour copy is staged once, on the sax bus of its run"
    c = rec["clips"][sour]
    assert c["L"] == pytest.approx(L0[victim][sour] + rec["t"]) and c["target_lufs"] == doc["S_star"]
    assert abs(c["off_lu"]) <= CLIP_FIX_LU + 1e-9 and c["lufs"] == pytest.approx(c["L"] + c["F"] + 0.02)
    assert set(rec["L"]) == set(rec["P"]) == set(TWO), "the trim and the median fader see the reference loops only"
    assert rec["F"] == round(doc["S_star"] + rec["T"] - statistics.median(rec["L"].values()), 2)
    assert all(sour not in r["clips"] for v, r in doc["variants"].items() if v != victim)
    assert doc["extra_cases"] == {sour: {"loop": loop, "variant": "sour", "variants": [victim],
                                         "text_sha256": hashlib.sha256(raise_accidentals(
                                             PLAN.loop_text(PLAN.loop_named(loop).id)).encode()).hexdigest(),
                                         "brief": doc["loops"][loop]["brief"], "n": doc["loops"][loop]["n"]}}
    assert set(doc["drift"]["digests"][run]) == set(TWO) | {sour}
    assert all(set(d) == set(TWO) for r, d in doc["drift"]["digests"].items() if r != run)
    assert doc["draw"] == {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_a_draw_off_the_tools_loops_or_its_plan_is_refused(rig, tmp_path):
    # Every clip the draw makes needs its fader from this measurement: a render on a loop that --loops does not
    # hold, or a draw of another plan, stops the tool before it measures anything.
    bench = TableBench(rig["P"], rig["L0"])
    with pytest.raises(LevelsError, match="l5_wistful_mid"):
        levels(rig, bench, draw=tiny_draw(tmp_path / "a.json", "sax_mtg_alto", "l9_hopeful_high",
                                         renders_on="l5_wistful_mid"))
    other = json.loads(tiny_draw(tmp_path / "b.json", "sax_mtg_alto", "l9_hopeful_high").read_text())
    other["plan"]["sha256"] = "0" * 64
    (tmp_path / "c.json").write_text(json.dumps(other))
    with pytest.raises(LevelsError, match="another plan"):
        levels(rig, bench, draw=tmp_path / "c.json")
    assert bench.prepared == []


def test_prior_reuses_when_nothing_drifted_and_re_measures_a_changed_base_or_digest_recording_both(rig, tmp_path):
    first = tmp_path / "levels.json"
    first.write_text(json.dumps(levels(rig, TableBench(rig["P"], rig["L0"]))))
    same = TableBench(rig["P"], rig["L0"])
    doc = levels(rig, same, prior=first)
    assert same.prepared == [] and doc["reused"] is True, "equal drift keys: nothing is measured again"
    assert doc["variants"] == json.loads(first.read_text())["variants"]
    assert doc["prior"]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    # A base palette edited after job 1: its sha256 changes, so every instrument is measured again.
    base2 = tmp_path / "base2.yaml"
    base2.write_text(rig["base"].read_text() + "# edited after job 1\n")
    generate(PLAN_PATH, tmp_path / "pal2", stage="day1", levels=None, base=base2)
    moved = {v: {k: x - 0.25 for k, x in t.items()} for v, t in rig["L0"].items()}
    bench = TableBench(rig["P"], moved)
    doc = levels(rig, bench, prior=first, measure=tmp_path / "pal2" / "measure")
    assert len(bench.prepared) == 8 and doc["reused"] is False
    old, new = json.loads(first.read_text())["drift"]["base_sha256"], hashlib.sha256(base2.read_bytes()).hexdigest()
    assert doc["drift"]["base_sha256"] == new and doc["prior"]["drift"]["base_sha256"] == old != new
    assert "base_sha256" in doc["prior"]["changed"]
    assert doc["variants"]["gu_organ"]["prior"] == {"t": 0.0, "F": json.loads(first.read_text())["variants"]
                                                    ["gu_organ"]["F"]}
    assert doc["variants"]["gu_organ"]["moved_db"]["F"] == 0.0, "every L moved together, so no fader moved"
    # A loop that changed (another workflow's edit): its compiled digest changes in every run.
    loops2 = loops_subset(tmp_path / "loops2", TWO)
    l9 = loops2 / "l9_hopeful_high.ls"
    l9.write_text(l9.read_text().replace("@7 ohat: ......x.........", "@7 ohat: ........x......."))
    bench = TableBench(rig["P"], rig["L0"])
    doc = levels(rig, bench, prior=first, loops=loops2)
    assert len(bench.prepared) == 8
    before, after = (json.loads(first.read_text())["drift"]["digests"]["m1"], doc["drift"]["digests"]["m1"])
    assert before["l9_hopeful_high"] != after["l9_hopeful_high"] and before["l1_melancholy_low"] == \
        after["l1_melancholy_low"]
    assert doc["prior"]["drift"]["digests"]["m1"]["l9_hopeful_high"] == before["l9_hopeful_high"]
    assert "digests.m1.l9_hopeful_high" in doc["prior"]["changed"]
    assert "digests.m1.l1_melancholy_low" not in doc["prior"]["changed"]


def test_moves_above_01_db_are_listed(rig, tmp_path):
    first = tmp_path / "levels.json"
    first.write_text(json.dumps(levels(rig, TableBench(rig["P"], rig["L0"]))))
    L0 = {v: dict(t) for v, t in rig["L0"].items()}
    L0["jrhodes3d"] = {k: x + 0.5 for k, x in L0["jrhodes3d"].items()}
    doc = levels(rig, TableBench(rig["P"], L0), prior=first, stamp="e4f5a6b 0 0")
    assert doc["prior"]["changed"] == ["stamp"]
    assert doc["moved"] == ["jrhodes3d"] and doc["variants"]["jrhodes3d"]["moved_db"] == {"t": 0.0, "F": -0.5}


def test_the_tool_refuses_a_measuring_index_that_misses_a_day1_variant(rig, tmp_path):
    folder = tmp_path / "measure"
    shutil.copytree(rig["measure"], folder)
    index = json.loads((folder / "index.json").read_text())
    del index["variants"]["gu_organ"]
    for r in index["runs"]:
        if r["instruments"]["keys"] == "gu_organ":
            r["measure"].remove("keys")
    (folder / "index.json").write_text(json.dumps(index))
    with pytest.raises(LevelsError, match="gu_organ"):
        levels(rig, TableBench(rig["P"], rig["L0"]), measure=folder)


def test_the_cli_writes_levels_json_and_exits_2_on_bad_arguments(rig, tmp_path, capsys):
    out = tmp_path / "summary" / "levels.json"
    args = ["--plan", str(PLAN_PATH), "--loops", str(rig["loops"]), "--palettes", str(rig["measure"]),
            "--stage", "day1", "--out", str(out), "--workers", "2"]
    assert main(args, bench=TableBench(rig["P"], rig["L0"])) == 0
    doc = json.loads(out.read_text())
    assert doc["stage"] == "day1" and len(doc["variants"]) == 20
    assert "S*" in capsys.readouterr().out
    with pytest.raises(SystemExit) as ex:
        main(args[:4] + ["--palettes", str(tmp_path / "none"), "--stage", "day1", "--out", str(out)])
    assert ex.value.code == 2
    with pytest.raises(SystemExit) as ex:
        main(args[:-4] + ["--stage", "day4", "--out", str(out)])
    assert ex.value.code == 2
    with pytest.raises(SystemExit) as ex:
        main(args + ["--draw", str(tmp_path / "no-draw.json")])
    assert ex.value.code == 2
    L0 = {v: dict(t, **{"l9_hopeful_high.sour": t["l9_hopeful_high"]}) for v, t in rig["L0"].items()}
    draw = tiny_draw(tmp_path / "draw.json", "sax_gu_alto", "l9_hopeful_high")
    assert main(args + ["--draw", str(draw)], bench=TableBench(rig["P"], L0)) == 0
    doc = json.loads(out.read_text())
    assert "l9_hopeful_high.sour" in doc["variants"]["sax_gu_alto"]["clips"] and doc["draw"]["file"] == str(draw)
    assert "clips" in capsys.readouterr().out


# ------------------------------------------------------------------------------------ Day 2 (CD-53)
LEADERS2 = "keys=jrhodes3d+gu_tine_ep+fp_fm_ep1/kit=kit_swirly/bass=bass_fp_finger_yr/sax=sax_mtg_tenor+sax_gu_alto"
NINE = tuple(sorted(yaml.safe_load((ROOT / "shootouts" / "loops" / "briefs.yaml").read_text())["loops"]))


class HashBench(TableBench):
    """Made-up loudness for any (palette, instrument, case): a reference run's bus depends only on its instrument and
    the case (as a real bus does), a stage palette's (`config:<key>`) also on the palette, whose option may move it."""

    def __init__(self, gate=0.02):
        super().__init__({}, {}, gate=gate)

    @staticmethod
    def _u(*keys) -> float:
        return int(hashlib.sha256("|".join(keys).encode()).hexdigest()[:8], 16) / 2 ** 32

    def _where(self, run):
        return run.id if run.id.startswith("config:") else ""

    def input_lufs(self, run, case, bus):
        return -45.0 + 6.0 * self._u("P", self._where(run), run.instruments[bus], case.name)

    def chain_lufs(self, run, case, bus, trim, keep=False):
        out = -40.0 + 8.0 * self._u("L", self._where(run), run.instruments[bus], case.name) + trim
        if keep:
            self.kept[(run.id, case.name, bus)] = out
        return out


@pytest.fixture(scope="module")
def rig2(rig, tmp_path_factory):
    """A Day-2 draw from LEADERS2 and its measuring palettes (every instrument it plays, each matched palette)."""
    from lofi_rl.shootout.draw import draw, parse_leaders, write_draw
    tmp = tmp_path_factory.mktemp("levels2")
    doc = draw(PLAN, "day2", seed=0xD2, leaders=parse_leaders(LEADERS2))
    path = write_draw(doc, tmp / "draw.json")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOFI_ASSETS", str(rig["assets"]))
        generate(PLAN_PATH, tmp / "palettes", stage="day2", levels=None, base=rig["base"], draw=path)
    return {"tmp": tmp, "draw": doc, "draw_path": path, "measure": tmp / "palettes" / "measure"}


def levels2(rig2, bench, **kw):
    return run_levels(PLAN_PATH, ROOT / "shootouts" / "loops", kw.pop("measure", rig2["measure"]), "day2",
                      draw=kw.pop("draw", rig2["draw_path"]), bench=bench, stamp="a1b2c3d 0 0", **kw)


def test_day2_measures_its_instruments_again_and_every_clip_on_its_own_palette_and_text(rig2):
    # CD-53: every instrument the day plays is measured as on Day 1 (its trim and median fader, the frozen palette's,
    # section 3 "Drift between jobs"); then every stage palette a matched clip plays is measured on each case its
    # clips play, on each bus its stage matches, with that bus's instrument's trim: the clip's own fader, corrected by
    # the loudness measured after it, and the loudness at the band's median fader.
    bench = HashBench(gate=0.02)
    doc = levels2(rig2, bench)
    draw_doc = rig2["draw"]
    index = json.loads((rig2["measure"] / "index.json").read_text())
    assert set(doc["variants"]) == set(index["variants"]) and set(doc["configs"]) == set(index["configs"])
    assert [p[0] for p in bench.prepared][:len(index["runs"])] == [r["run"] for r in index["runs"]]
    v, s_star = doc["variants"], doc["S_star"]
    for key, cfg in doc["configs"].items():
        st = PLAN.stages[cfg["stage"]]
        assert list(cfg["buses"]) == list(st.match), key
        for bus, e in cfg["buses"].items():
            inst = index["configs"][key]["instruments"][bus]
            assert (e["instrument"], e["t"], e["median_F"]) == (inst, v[inst]["t"], v[inst]["F"]), (key, bus)
            target = s_star + {"sax": 0.0, "drums": -1.5, "bass": -3.0, "keys": -6.0}[bus]
            for case, c in e["cases"].items():
                assert c["target_lufs"] == target and abs(c["off_lu"]) <= CLIP_FIX_LU + 1e-9, (key, bus, case)
                assert c["lufs"] == pytest.approx(c["L"] + c["F"] + 0.02)
                assert c["median_lufs"] == pytest.approx(c["L"] + v[inst]["F"] + 0.02)
            assert e["checks"]["clips"]["ok"]
            assert ("loops" in e) == st.match_loops
            if st.match_loops:
                L9 = e["loops"]["L"]
                assert len(L9) == 9 and e["loops"]["F"] == round(target - statistics.median(L9.values()), 2)
                assert e["loops"]["checks"]["post_fader"]["ok"] and e["loops"]["checks"]["linear"]["ok"]
    # every drawn clip's case is measured on its own palette, and nothing else is but a `loops` config's 9 loops
    want = {}
    for rid, ren in draw_doc["renders"].items():
        spec = draw_doc["palettes"][ren["palette"]]
        want.setdefault(spec["base"], set()).add(spec["level"]["case"])
    for key, cfg in doc["configs"].items():
        got = set(cfg["cases"])
        extra = got - want[key]
        assert want[key] <= got, key
        assert extra == (set() if not cfg["loops"] else
                         {f"{x}.spread" if cfg["stage"] == "K2" else x for x in NINE} - want[key]), key
    assert {c for c in doc["configs"]["v"]["cases"]} == {f"{lp}.{vamp}" for lp in NINE
                                                         for vamp in PLAN.vamps if PLAN.vamps[vamp].mood ==
                                                         PLAN.loop_named(lp).brief.mood}
    # the per-round spreads: one row per drawn round, each bus its stage matches, before and after the own faders
    rounds = [r for b in draw_doc["batches"] for r in b["rounds"]]
    assert len(doc["rounds"]) == len(rounds) == 26
    for row in doc["rounds"]:
        assert list(row["buses"]) == list(PLAN.stages[row["stage"]].match)
        assert row["after_lu"] <= 2 * CLIP_FIX_LU + 1e-9 and row["before_lu"] >= 0
    assert doc["flags"] == []


def test_config_cases_name_each_text_once_and_refuse_a_clip_off_the_measuring_index(rig2):
    from lofi_rl.shootout.levels import config_cases, read_cases
    cases = read_cases(ROOT / "shootouts" / "loops")
    index = json.loads((rig2["measure"] / "index.json").read_text())
    got = config_cases(rig2["draw"], index["configs"], cases)
    k2 = next(k for k in got if k.startswith("k2_"))
    assert [c.name for c in got[k2]] == sorted(f"{x}.spread" for x in NINE)
    assert all(c.text == PLAN.loop_named(c.loop).text.replace("keys: ep rootless", "keys: ep spread") for c in got[k2])
    for c in got["v"]:
        assert c.text.splitlines()[0] == f"vamp: {PLAN.vamps[c.variant].line}" and c.name == f"{c.loop}.{c.variant}"
    with pytest.raises(LevelsError, match="does not measure"):
        config_cases(rig2["draw"], {k: c for k, c in index["configs"].items() if k != "v"}, cases)
    bad = copy.deepcopy(rig2["draw"])
    rid = next(r for r, e in bad["renders"].items() if e["stage"] == "S2b")
    bad["renders"][rid]["text"] += "\n"
    twin = dict(bad["renders"][rid], text=PLAN.loop_named(bad["renders"][rid]["loop"]).text)
    bad["renders"][rid + ".twin"] = twin
    with pytest.raises(LevelsError, match="two different texts"):
        config_cases(bad, index["configs"], cases)


def test_day2_needs_the_draw_its_measuring_palettes_were_made_from(rig2, tmp_path):
    bench = HashBench()
    with pytest.raises(LevelsError, match="needs the job's draw"):
        levels2(rig2, bench, draw=None)
    other = json.loads(rig2["draw_path"].read_text())
    other["seed"] = "0" * 16
    (tmp_path / "other.json").write_text(json.dumps(other))
    with pytest.raises(LevelsError, match="another draw"):
        levels2(rig2, bench, draw=tmp_path / "other.json")
    assert bench.prepared == []


def test_day2_records_each_instruments_move_from_job_1_and_never_reuses_another_days_values(rig, rig2, tmp_path):
    # Section 3 "Drift between jobs": job 2 measures every instrument it plays again and lists any trim or fader
    # that moved more than 0.1 dB from job 1's. A Day-1 levels file is never reused for Day 2, even with equal keys.
    first = tmp_path / "day1.json"
    day1 = levels(rig, TableBench(rig["P"], rig["L0"]))
    for vid in day1["variants"]:
        day1["variants"][vid]["t"] += 0.5 if vid == "kit_swirly" else 0.0
    first.write_text(json.dumps(day1))
    doc = levels2(rig2, HashBench(), prior=first)
    assert doc["reused"] is False and "configs" in doc
    for vid, rec in doc["variants"].items():
        old = day1["variants"][vid]
        assert rec["prior"] == {"t": old["t"], "F": old["F"]}, vid
        assert rec["moved_db"] == {"t": round(rec["t"] - old["t"], 2), "F": round(rec["F"] - old["F"], 2)}, vid
    assert doc["moved"] == sorted(vid for vid, r in doc["variants"].items()
                                  if max(abs(x) for x in r["moved_db"].values()) > 0.1)
    assert "kit_swirly" in doc["moved"]
    same = tmp_path / "same.json"                        # equal drift keys, but another day's file
    same.write_text(json.dumps(dict(levels2(rig2, HashBench()), stage="day1")))
    again = levels2(rig2, HashBench(), prior=same)
    assert again["prior"]["changed"] == [] and again["reused"] is False and "configs" in again


# ------------------------------------------------------------------------ the real banks (§7.1 test 6)
BIG = os.environ.get("LOFI_ASSETS") or os.environ.get("LOFI_ASSETS_BIG")


@pytest.mark.cluster
def test_after_matching_every_clip_sits_at_its_target_on_two_loops(tmp_path, monkeypatch):
    # CD-52 replaced "each loop within 0.1 LU of its own L + F", which kit_virtuosity missed on L1 (0.1007 LU through
    # the -70 LUFS gate, a pure gain on a bus at -55 LUFS): each clip now takes its own fader, measured after it.
    if not BIG or not (Path(BIG) / "converted").is_dir():
        pytest.skip("LOFI_ASSETS (or LOFI_ASSETS_BIG) does not name the fetched assets folder")
    monkeypatch.setenv("LOFI_ASSETS", BIG)
    missing = tuple(k for k in FILE_KITS if not (Path(BIG) / PLAN.candidates[k].archive["path"]).is_file())
    generate(PLAN_PATH, tmp_path / "palettes", stage="day1", levels=None, without=missing)
    doc = run_levels(PLAN_PATH, loops_subset(tmp_path / "loops", TWO), tmp_path / "palettes" / "measure", "day1",
                     workers=8)
    s_star = doc["S_star"]
    for vid, rec in doc["variants"].items():
        target = s_star + rec["T"]
        post = rec["checks"]["post_fader"]
        assert abs(statistics.median(post["lufs"].values()) - target) <= 0.05, vid   # the frozen palette's median F
        for loop in TWO:                                                              # every clip at its own fader
            c = rec["clips"][loop]
            assert c["L"] == rec["L"][loop] and c["F"] == round(c["F"], 2), (vid, loop)
            assert abs(c["lufs"] - target) <= 0.1 and abs(c["off_lu"]) <= CLIP_FIX_LU + 1e-9, (vid, loop)
        assert rec["checks"]["clips"]["ok"], vid
        assert rec["t"] == 0.0 or rec["source"] != "gu", vid
    # A clip's palette carries its own fader on the bus its round varies and the median ones elsewhere; its render
    # reports the tool's L before every fader, so the bus it plays is the one the tool measured after it.
    from lofi_rl.render.pipeline import render
    from lofi_rl.shootout.palettes import Inputs, clip_palette
    (tmp_path / "levels.json").write_text(json.dumps(doc))
    got = generate(PLAN_PATH, tmp_path / "stage", stage="day1", levels=tmp_path / "levels.json", without=missing)
    base = next(p for p in got.index["palettes"] if p["id"] == "k1_vcsl_upright_knight")
    inputs = Inputs(PLAN, ROOT / PLAN.base, 1, missing, ROOT)
    rel, data, entry = clip_palette(inputs, doc, base, "k1_vcsl_upright_knight_l1",
                                    {"bus": "keys", "variant": "vcsl_upright_knight", "case": "l1_melancholy_low"})
    (tmp_path / "stage" / rel).write_bytes(data)
    m = render(PLAN.loop_text("L1"), PLAN.briefs()["l1_melancholy_low"],
               load_palette(tmp_path / "stage" / rel, verify=True)).manifest
    clip = doc["variants"]["vcsl_upright_knight"]["clips"]["l1_melancholy_low"]
    assert entry["fader_db"] == {**base["fader_db"], "keys": clip["F"]} and entry["level"]["lufs"] == clip["lufs"]
    for bus, vid in entry["instruments"].items():
        assert m["buses"][bus]["lufs"] == doc["variants"][vid]["L"]["l1_melancholy_low"], bus
        assert m["buses"][bus]["gain_db"] == entry["fader_db"][bus], bus
    assert m["gates"]["G7"] == "pass"
