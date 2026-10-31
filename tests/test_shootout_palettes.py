"""The shootout palette generator and its base (docs/SHOOTOUTS.md §7.8; §3 "How the values are carried", CD-51).

Default-suite tests run on a synthetic copy of the assets: every SoundFont the base names becomes a tiny file holding
only the preset headers the palettes name (enough for load_palette(verify=True), never for a render), and the two
file kits become synthetic §7.4 archives (a tar of 44.1 kHz mono float32 one-shots plus the importer's printed
`lanes:` and `choke:` blocks), all in tmp_path. The real-asset run is `-m cluster` (LOFI_ASSETS naming assets/big)."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from helpers_shootout import BASE_PATH, FILE_KITS, PLAN_PATH, ROOT, kit_archive, synthetic_base, synthetic_levels
from lofi_rl.dsl import parse
from lofi_rl.dsl.vocab import LANES
from lofi_rl.palette import LAYERS, load_palette
from lofi_rl.shootout.palettes import GenerateError, generate, main
from lofi_rl.shootout.plan import load_plan

DEV = ROOT / "palettes" / "dev_gu.yaml"
PLAN = load_plan(PLAN_PATH)
DAY1 = ("K1", "R1", "R2", "S1")
BUS_OF = {"K1": "keys", "R1": "drums", "R2": "bass", "S1": "sax"}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic assets, a base palette pointing at them, and the measuring palettes and stage palettes made
    from them (with a synthetic levels file)."""
    tmp = tmp_path_factory.mktemp("shootout")
    assets = tmp / "assets"
    base = synthetic_base(tmp, assets)
    tars = {k: kit_archive(assets, k, shaker_db) for k, shaker_db in zip(FILE_KITS, (-3.25, 2.5))}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOFI_ASSETS", str(assets))
        measure = generate(PLAN_PATH, tmp / "palettes", stage="day1", levels=None, base=base)
        levels = synthetic_levels(tmp / "palettes" / "measure" / "index.json", base, tmp / "levels.json")
        stage = generate(PLAN_PATH, tmp / "palettes", stage="day1", levels=levels, base=base)
    return {"tmp": tmp, "assets": assets, "base": base, "tars": tars, "measure": measure, "stage": stage,
            "levels": levels, "out": tmp / "palettes"}


@pytest.fixture(autouse=True)
def assets_env(request, monkeypatch):
    if "env" in request.fixturenames:
        monkeypatch.setenv("LOFI_ASSETS", str(request.getfixturevalue("env")["assets"]))


def stage_palettes(env, stage: str):
    return [p for p in env["stage"].index["palettes"] if p["stage"] == stage]


def flat(node, prefix=""):
    """{dotted path: leaf} of a YAML document; list items by index."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            out.update(flat(v, f"{prefix}{k}."))
        return out or {prefix[:-1]: {}}
    if isinstance(node, list):
        out = {}
        for i, v in enumerate(node):
            out.update(flat(v, f"{prefix}{i}."))
        return out or {prefix[:-1]: []}
    return {prefix[:-1]: node}


def doc_of(env, entry) -> dict:
    return yaml.safe_load((env["out"] / entry["file"]).read_text())


# ------------------------------------------------------------------------------------- the base
def test_the_base_is_dev_gus_band_under_the_shootout_names():
    base, dev = load_palette(BASE_PATH, verify=False), load_palette(DEV, verify=False)
    assert dict(base.aliases) == {} and "ep" not in base.aliases and "dusty" not in base.aliases
    assert set(base.keys) == {"ep"} and set(base.kits) == {"dusty"} and set(base.sax) == {"tenor"}
    assert base.sax_lead == "tenor" and base.sax["tenor"] == dev.sax["tenor"]
    ep, rhodes = base.keys["ep"], dev.keys["rhodes"]
    assert (ep.src, ep.bank, ep.prog, dict(ep.vel)) == (rhodes.src, rhodes.bank, rhodes.prog, dict(rhodes.vel))
    dusty, jazz = base.kits["dusty"], dev.kits["jazz"]
    assert dusty.choke == jazz.choke and dusty.bass == jazz.bass
    assert {k: (dict(v.layers), v.pan, v.db) for k, v in dusty.lanes.items()} == \
        {k: (dict(v.layers), v.pan, v.db) for k, v in jazz.lanes.items()}
    for field in ("groove", "sax_expr", "numerals", "qualities", "voicings", "loudness", "bed", "sr"):
        assert getattr(base, field) == getattr(dev, field), field
    assert dict(base.mix.bus_target_lu) == dict(dev.mix.bus_target_lu)
    assert set(base.mix.bus_gain_db.values()) == {0.0} and base.mix.sax_warm is False
    for bus in ("keys", "bass", "sax"):                 # the trim slot heads the chain, at 0 dB (CD-51)
        head, *rest = base.mix.buses[bus]
        assert (head.name, dict(head.params)) == ("Gain", {"gain_db": 0.0})
        assert tuple(rest) == dev.mix.buses[bus], bus
    for bus in ("drums", "sax_warm"):                   # a kit's trim goes on its lanes' db instead
        assert base.mix.buses[bus] == dev.mix.buses[bus]
    assert base.mix.master == dev.mix.master
    assert base.sources["gu"] == dev.sources["gu"]


def test_every_fetch_source_of_the_base_is_pinned_in_the_asset_manifest():
    entries = yaml.safe_load((ROOT / "assets" / "MANIFEST.yaml").read_text())["assets"]
    pins = {}
    for e in entries:
        pins[e["path"]] = (e["sha256"], e["license"])
        for x in e.get("extracts", []):
            pins[x["path"]] = (x["sha256"], e["license"])
    base = load_palette(BASE_PATH, verify=False)
    used = {c.source for c in PLAN.candidates.values() if c.source}
    assert used == set(base.sources), "the base lists every candidate's source and nothing else"
    for sid, src in base.sources.items():
        if src.mode == "fetch":
            assert (src.sha256, src.license) == pins[f"assets/big/{src.path}"], sid


def test_the_committed_base_loads_its_commit_source_with_its_sha256():
    base = load_palette(BASE_PATH, verify=False)
    gu = base.source_path("gu")
    assert hashlib.sha256(gu.read_bytes()).hexdigest() == base.sources["gu"].sha256


# ---------------------------------------------------------------------------- measuring palettes
def test_the_measuring_runs_measure_every_day1_variant_once_with_the_anchors_first(env):
    index = env["measure"].index
    runs = index["runs"]
    assert [r["run"] for r in runs] == [f"m{i}" for i in range(1, 9)], "8 runs on Day 1 (§3 step 1)"
    assert runs[0]["instruments"] == PLAN.levels.anchors and runs[0]["measure"] == ["sax", "keys", "bass", "drums"]
    measured = [(r["instruments"][b], b) for r in runs for b in r["measure"]]
    want = {(o, BUS_OF[s]) for s in DAY1 for o in PLAN.stages[s].options}
    assert sorted(measured) == sorted(want), "every Day-1 variant once"
    assert set(index["variants"]) == {o for o, _ in want}
    for vid, v in index["variants"].items():
        c = PLAN.candidates[vid]
        assert v["bus"] == c.bus and v["flags"] == {"voicing": "rootless", "sax_warm": False, "sax_expr": True}
        assert (v["kit_map_sha256"] is not None) == (c.bus == "drums")
    for r in runs:
        pal = load_palette(env["out"] / "measure" / r["file"], verify=True)
        assert set(pal.mix.bus_gain_db.values()) == {0.0}
        assert all(pal.mix.buses[b][0].params["gain_db"] == 0.0 for b in ("keys", "bass", "sax"))
        assert hashlib.sha256((env["out"] / "measure" / r["file"]).read_bytes()).hexdigest() == r["sha256"]


# ----------------------------------------------------------------------------- stage palettes
def test_day1_writes_one_palette_per_stage_and_option(env):
    got = sorted((p["stage"], p["option"], p["kit"]) for p in env["stage"].index["palettes"])
    want = [("K1", o, None) for o in PLAN.stages["K1"].options] + \
           [("R1", o, None) for o in PLAN.stages["R1"].options] + \
           [("R2", b, k) for k in PLAN.stages["R1"].options for b in PLAN.stages["R2"].options] + \
           [("S1", o, None) for o in PLAN.stages["S1"].options]
    assert got == sorted(want) and len(got) == 26
    assert env["stage"].index["band"] == PLAN.days[1].band


def test_every_generated_palette_loads_with_verify_and_every_loop_parses_without_repair(env):
    entries = env["stage"].index["palettes"] + [dict(r, file=f"measure/{r['file']}")
                                                for r in env["measure"].index["runs"]]
    vocabs = {}
    for e in entries:
        pal = load_palette(env["out"] / e["file"], verify=True)
        assert pal.sha256 == e["sha256"]
        assert set(pal.keys) == {"ep"} and set(pal.kits) == {"dusty"} and len(pal.sax) == 1
        assert dict(pal.aliases) == {}
        vocabs[pal.vocab()] = e["file"]
    for vocab in vocabs:
        for name in PLAN.loop_names:
            mood = PLAN.briefs()[PLAN.loop_names[name]].mood
            res = parse(PLAN.loop_text(name), vocab=vocab, mode=mood)
            assert res.gate.ok and res.repairs == [], (vocabs[vocab], name, res.gate, res.repairs)


ALLOWED = {   # §2.1 "one thing only": the candidate's block, its trim and its fader (and the palette's name)
    "K1": ("id", "keys.ep.", "mix.buses.keys.0.", "mix.bus_gain_db.keys"),
    "R1": ("id", "kits.dusty.lanes.", "mix.bus_gain_db.drums"),
    "R2": ("id", "kits.dusty.bass.", "mix.buses.bass.0.", "mix.bus_gain_db.bass"),
    "S1": ("id", "sax.", "sax_lead", "mix.buses.sax.0.", "mix.bus_gain_db.sax"),
}


def test_within_each_round_the_palettes_and_loops_differ_only_in_the_candidates_block_trim_and_fader(env):
    for stage in DAY1:
        entries = stage_palettes(env, stage)
        groups = {}
        for e in entries:                                   # R2: a round plays one kit with each bass
            groups.setdefault(e["kit"], []).append(e)
        for group in groups.values():
            docs = [(e["option"], flat(doc_of(env, e))) for e in group]
            for i, (oa, a) in enumerate(docs):
                for ob, b in docs[i + 1:]:
                    diff = {k for k in a.keys() | b.keys() if a.get(k, "<absent>") != b.get(k, "<absent>")}
                    bad = sorted(k for k in diff if not k.startswith(ALLOWED[stage]))
                    assert not bad, (stage, oa, ob, bad[:5])
                    assert diff - {"id"}, (stage, oa, ob, "two options of a round must differ")
    for slot in env["stage"].index["slots"]:
        assert slot["sub"] == {} and slot["text_sha256"] == hashlib.sha256(
            PLAN.loop_text(slot["loop"]).encode()).hexdigest(), "Day 1 plays every loop as written (§5)"


def test_trims_and_faders_are_carried_as_cd51_says(env):
    lv = json.loads(env["levels"].read_text())["variants"]
    blocks = {k: yaml.safe_load((env["assets"] / "derived" / "kits" / f"{k}.yaml").read_text()) for k in FILE_KITS}
    base = yaml.safe_load(env["base"].read_text())
    blocks["kit_gu_jazz"] = {"lanes": base["kits"]["dusty"]["lanes"]}
    for e in env["stage"].index["palettes"]:
        doc = doc_of(env, e)
        for bus, vid in e["instruments"].items():
            assert doc["mix"]["bus_gain_db"][bus] == lv[vid]["F"], (e["file"], bus)
            assert e["fader_db"][bus] == lv[vid]["F"] and e["trim_db"][bus] == lv[vid]["t"]
            if bus == "drums":
                for lane in LANES:
                    want = blocks[vid]["lanes"][lane]["db"] + lv[vid]["t"]
                    assert doc["kits"]["dusty"]["lanes"][lane]["db"] == want, (e["file"], lane)
            else:
                assert doc["mix"]["buses"][bus][0] == ["Gain", {"gain_db": lv[vid]["t"]}], (e["file"], bus)
    alto = next(e for e in stage_palettes(env, "S1") if e["option"] == "sax_mtg_alto")
    doc = doc_of(env, alto)
    assert doc["sax_lead"] == "alto" and list(doc["sax"]) == ["alto"]
    assert doc["sax"]["alto"] == {"src": "mtg_alto", "bank": 0, "prog": 0, **PLAN.registers["alto"]}
    k = next(e for e in stage_palettes(env, "R1") if e["option"] == "kit_swirly")
    assert {s["src"] for lane in doc_of(env, k)["kits"]["dusty"]["lanes"].values()
            for layer in LAYERS for s in lane[layer]} == {"kit_swirly"}
    assert doc_of(env, k)["sources"]["kit_swirly"]["sha256"] == env["tars"]["kit_swirly"]


def test_generation_is_byte_deterministic(env, tmp_path):
    again = generate(PLAN_PATH, tmp_path, stage="day1", levels=env["levels"], base=env["base"])
    assert sorted(p.relative_to(tmp_path) for p in again.files) == \
        sorted(p.relative_to(env["out"]) for p in env["stage"].files)
    for p in again.files:
        assert p.read_bytes() == (env["out"] / p.relative_to(tmp_path)).read_bytes(), p.name


# --------------------------------------------------------------------------------- refusals
def test_a_missing_or_changed_kit_archive_is_refused_and_without_leaves_it_out(env, tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    for src in yaml.safe_load(env["base"].read_text())["sources"].values():
        if src["mode"] == "fetch":
            (assets / src["path"]).parent.mkdir(parents=True, exist_ok=True)
            (assets / src["path"]).write_bytes((env["assets"] / src["path"]).read_bytes())
    monkeypatch.setenv("LOFI_ASSETS", str(assets))
    with pytest.raises(GenerateError, match=r"kit_virtuosity.*derived/kits/kit_virtuosity\.tar"):
        generate(PLAN_PATH, tmp_path / "a", stage="day1", levels=None, base=env["base"])
    got = generate(PLAN_PATH, tmp_path / "b", stage="day1", levels=None, base=env["base"],
                   without=FILE_KITS)
    assert got.index["without"] == list(FILE_KITS)
    assert all(r["instruments"]["drums"] == "kit_gu_jazz" for r in got.index["runs"])
    assert not set(FILE_KITS) & set(got.index["variants"])
    pinned = copy.deepcopy(yaml.safe_load(PLAN_PATH.read_text()))
    pinned["candidates"]["kit_swirly"]["archive"]["sha256"] = "0" * 64
    plan2 = tmp_path / "plan.yaml"
    plan2.write_text(yaml.safe_dump(pinned, sort_keys=False))
    monkeypatch.setenv("LOFI_ASSETS", str(env["assets"]))
    with pytest.raises(GenerateError, match="kit_swirly.*does not match its pinned sha256"):
        generate(plan2, tmp_path / "c", stage="day1", levels=None, base=env["base"])


def test_a_levels_file_from_another_base_or_without_a_variant_is_refused(env, tmp_path):
    doc = json.loads(env["levels"].read_text())
    other = dict(doc, base_sha256="1" * 64)
    (tmp_path / "a.json").write_text(json.dumps(other))
    with pytest.raises(GenerateError, match="base"):
        generate(PLAN_PATH, tmp_path / "a", stage="day1", levels=tmp_path / "a.json", base=env["base"])
    short = copy.deepcopy(doc)
    del short["variants"]["sax_mtg_alto"]
    (tmp_path / "b.json").write_text(json.dumps(short))
    with pytest.raises(GenerateError, match="sax_mtg_alto"):
        generate(PLAN_PATH, tmp_path / "b", stage="day1", levels=tmp_path / "b.json", base=env["base"])
    moved = copy.deepcopy(doc)
    moved["variants"]["kit_swirly"]["kit_map_sha256"] = "2" * 64
    (tmp_path / "c.json").write_text(json.dumps(moved))
    with pytest.raises(GenerateError, match="kit_swirly"):
        generate(PLAN_PATH, tmp_path / "c", stage="day1", levels=tmp_path / "c.json", base=env["base"])
    with pytest.raises(GenerateError, match="day2"):
        generate(PLAN_PATH, tmp_path / "d", stage="day2", levels=None, base=env["base"])


# ------------------------------------------------------------------------------------- the draw
def a_draw() -> dict:
    """A draw that meets every Day-1 constraint: K1's pairs are families, S1's t/a symbols their register."""
    fam = {}
    for o in PLAN.stages["K1"].options:
        fam.setdefault(PLAN.candidates[o].family, []).append(o)
    k1 = {}
    for (pair, (a, b)), members in zip(sorted(PLAN.stages["K1"].draw["pairs"].items()), sorted(fam.values())):
        k1[a], k1[b] = members
    reg = {r: [o for o in PLAN.stages["S1"].options if PLAN.candidates[o].register == r] for r in ("tenor", "alto")}
    s1 = {sym: reg[r][i] for r, syms in PLAN.stages["S1"].draw["groups"].items() for i, sym in enumerate(syms)}
    return {"stage": "day1", "symbols": {
        "K1": k1, "S1": s1,
        "R1": dict(zip(PLAN.stages["R1"].symbols, reversed(PLAN.stages["R1"].options))),
        "R2": dict(zip(PLAN.stages["R2"].symbols, PLAN.stages["R2"].options))}}


def test_a_draw_writes_the_render_list_and_a_draw_off_its_constraints_is_refused(env, tmp_path):
    draw = a_draw()
    (tmp_path / "draw.json").write_text(json.dumps(draw))
    got = generate(PLAN_PATH, tmp_path / "out", stage="day1", levels=env["levels"], base=env["base"],
                   draw=tmp_path / "draw.json")
    clips = json.loads((tmp_path / "out" / "clips.json").read_text())
    assert len(clips["rounds"]) == 18 and sum(len(r["tiles"]) for r in clips["rounds"]) == 24 + 9 + 9 + 24
    files = {(p["stage"], p["option"], p["kit"]): p for p in got.index["palettes"]}
    for r in clips["rounds"]:
        assert r["loop"] in PLAN.loop_names and r["text_sha256"]
        for pos, tile in enumerate(r["tiles"], 1):
            assert tile["tile"] == pos
            kit = draw["symbols"]["R1"][r["kit"]] if r["stage"] == "R2" else None
            assert tile["option"] == draw["symbols"][r["stage"]][tile["symbol"]]
            entry = files[(r["stage"], tile["option"], kit)]
            assert (tile["palette"], tile["palette_sha256"]) == (entry["file"], entry["sha256"])
    bad = copy.deepcopy(draw)
    t1, a1 = bad["symbols"]["S1"]["t1"], bad["symbols"]["S1"]["a1"]
    bad["symbols"]["S1"]["t1"], bad["symbols"]["S1"]["a1"] = a1, t1
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(GenerateError, match="S1.*t1.*tenor"):
        generate(PLAN_PATH, tmp_path / "bad", stage="day1", levels=env["levels"], base=env["base"],
                 draw=tmp_path / "bad.json")
    split = copy.deepcopy(draw)
    k1 = split["symbols"]["K1"]
    k1["P1b"], k1["P2a"] = k1["P2a"], k1["P1b"]
    (tmp_path / "split.json").write_text(json.dumps(split))
    with pytest.raises(GenerateError, match="K1.*P1.*family"):
        generate(PLAN_PATH, tmp_path / "split", stage="day1", levels=env["levels"], base=env["base"],
                 draw=tmp_path / "split.json")


# ---------------------------------------------------------------------------------------- the CLI
def test_the_cli_writes_the_palettes_and_exits_2_or_3_on_bad_inputs(env, tmp_path, capsys):
    out = tmp_path / "cli"
    args = ["--plan", str(PLAN_PATH), "--base", str(env["base"]), "--stage", "day1", "--out", str(out)]
    assert main(args + ["--levels", str(env["levels"])]) == 0
    printed = capsys.readouterr().out
    assert "26 palettes" in printed and (out / "index.json").is_file()
    with pytest.raises(SystemExit) as ex:
        main(["--plan", str(tmp_path / "nope.yaml"), "--stage", "day1", "--levels", "none", "--out", str(out)])
    assert ex.value.code == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dict(json.loads(env["levels"].read_text()), base_sha256="3" * 64)))
    assert main(args + ["--levels", str(bad)]) == 3
    assert "base" in capsys.readouterr().err


# ------------------------------------------------------------------------------------ real assets
BIG = os.environ.get("LOFI_ASSETS") or os.environ.get("LOFI_ASSETS_BIG")


@pytest.mark.cluster
def test_the_real_base_and_day1_palettes_load_with_verify(tmp_path, monkeypatch):
    if not BIG or not (Path(BIG) / "converted").is_dir():
        pytest.skip("LOFI_ASSETS (or LOFI_ASSETS_BIG) does not name the fetched assets folder")
    monkeypatch.setenv("LOFI_ASSETS", BIG)
    base = load_palette(BASE_PATH, verify=True)
    assert base.verify_sources() == []
    missing = [k for k in FILE_KITS if not (Path(BIG) / PLAN.candidates[k].archive["path"]).is_file()]
    got = generate(PLAN_PATH, tmp_path, stage="day1", levels=None, without=tuple(missing))
    for r in got.index["runs"]:
        load_palette(tmp_path / "measure" / r["file"], verify=True)
