"""The level-matching tool (docs/SHOOTOUTS.md §3 steps 1-6; §7.1 acceptance tests 4-7; no palette field yet).

Tests 4 and 7 run on tables of made-up loudness values (no audio), the linear-range check (5) on one FluidSynth
stem and one synthetic signal, and the end-to-end match (6) on the real banks with `-m cluster`."""
from __future__ import annotations

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
from lofi_rl.shootout.levels import (LevelsError, chain_lufs, faders, linear_check, main, post_check, run_levels,
                                     sax_target, trims, with_trim)
from lofi_rl.shootout.palettes import generate
from lofi_rl.shootout.plan import load_plan

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


def test_the_post_fader_check_holds_the_median_to_005_and_each_loop_to_01():
    L = {"x": {"a": -40.0, "b": -30.0, "c": -35.0}}
    F, s_star, T = {"x": 3.0}, -32.0, 0.0
    exact = {"x": {k: v + 3.0 for k, v in L["x"].items()}}
    assert post_check(exact, L, F, {"x": s_star + T})["x"]["ok"]
    gated = {"x": {"a": -36.93, "b": -27.0, "c": -32.04}}              # a few hundredths through the -70 gate
    r = post_check(gated, L, F, {"x": s_star + T})["x"]
    assert r["ok"] and r["median_lufs"] == -32.04 and r["worst_loop_lu"] == pytest.approx(0.07)
    off = {"x": {"a": -36.8, "b": -27.0, "c": -32.0}}                   # loop a 0.2 LU off its own L + F
    assert not post_check(off, L, F, {"x": s_star + T})["x"]["ok"]
    median_off = {"x": {"a": -36.94, "b": -26.94, "c": -31.94}}         # each 0.06 off: the median breaks 0.05
    assert not post_check(median_off, L, F, {"x": s_star + T})["x"]["ok"]


# --------------------------------------------------------------------------- §3 step 6a, the linear range
@needs_fluidsynth
def test_the_linear_range_check_passes_a_gu_stem_and_flags_an_input_that_drives_the_compressor():
    base = load_palette(BASE_PATH, verify=False)
    b = yaml.safe_load((ROOT / "shootouts" / "loops" / "briefs.yaml").read_text())["loops"]["l5_wistful_mid"]
    brief = Brief(energy=b["energy"], mood=b["mood"], bpm=b["bpm"], tonic=b["tonic"])
    res = parse((ROOT / "shootouts" / "loops" / "l5_wistful_mid.ls").read_text(), vocab=base.vocab(), mode=brief.mode)
    c = compile_loop(res.ast, brief, base)
    c = replace(c, bass=replace(c.bass, notes=()), sax=replace(c.sax, notes=()))    # one FluidSynth process
    stem = fluid_stems(c, base, 2, workers=1)["keys"]
    chain, n = base.mix.buses["keys"], c.n
    L, _ = chain_lufs(chain, stem, 0.0, n)
    L_low, _ = chain_lufs(chain, stem, -6.02, n)
    gu = linear_check({"l5": L}, {"l5": L_low})
    assert gu["ok"] and abs(gu["median_drop_db"] - 6.02) <= 0.1
    k = np.arange(2 * n)
    hot = np.tile(np.sin(2 * np.pi * 220.0 * k / 44100.0).astype(np.float32), (2, 1))   # 0 dBFS, 18 dB over
    L, _ = chain_lufs(chain, hot, 0.0, n)
    L_low, _ = chain_lufs(chain, hot, -6.02, n)
    flagged = linear_check({"hot": L}, {"hot": L_low})
    assert not flagged["ok"] and flagged["median_drop_db"] < 5.0, "the compressor takes part of the 6.02 dB"


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
    assert doc["drift"]["stamp"] == "a1b2c3d 0 0"
    assert doc["drift"]["base_sha256"] == hashlib.sha256(rig["base"].read_bytes()).hexdigest()
    assert set(doc["drift"]["digests"]) == {f"m{i}" for i in range(1, 9)}
    assert all(set(d) == set(TWO) for d in doc["drift"]["digests"].values())
    assert doc["prior"] is None and doc["reused"] is False


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


# ------------------------------------------------------------------------ the real banks (§7.1 test 6)
BIG = os.environ.get("LOFI_ASSETS") or os.environ.get("LOFI_ASSETS_BIG")


@pytest.mark.cluster
def test_after_matching_every_bus_sits_at_its_target_on_two_loops(tmp_path, monkeypatch):
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
        assert abs(statistics.median(post["lufs"].values()) - target) <= 0.05, vid
        for loop in TWO:
            assert abs(post["lufs"][loop] - (rec["L"][loop] + rec["F"])) <= 0.1, (vid, loop)
        assert rec["t"] == 0.0 or rec["source"] != "gu"
