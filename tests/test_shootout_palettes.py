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
import re
from pathlib import Path

import pytest
import yaml

from helpers_shootout import (BASE_PATH, FILE_KITS, PLAN_PATH, ROOT, kit_archive, synthetic_base, synthetic_levels,
                              synthetic_levels_day2)
from lofi_rl.dsl import parse
from lofi_rl.dsl.vocab import LANES
from lofi_rl.palette import LAYERS, load_palette
from lofi_rl.shootout.draw import draw, parse_leaders, write_draw
from lofi_rl.shootout.palettes import GenerateError, generate, main, yaml_load, yaml_text
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
    return yaml_load((env["out"] / entry["file"]).read_text())


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


def test_libyaml_and_the_pure_python_dumper_write_the_same_palette_bytes(env):
    # The generator uses libyaml when PyYAML has it; a palette's sha256 must not depend on that.
    for e in env["stage"].index["palettes"][::5] + [dict(env["measure"].index["runs"][1], file="measure/m2.yaml")]:
        text = (env["out"] / e["file"]).read_text()
        body = text.split("\n", 1)[1]
        doc = yaml.safe_load(text)
        assert yaml_text(doc, yaml.SafeDumper) == body, e["file"]
        if hasattr(yaml, "CSafeDumper"):
            assert yaml_text(doc, yaml.CSafeDumper) == body, e["file"]
        assert yaml_load(text) == doc


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


def test_the_importers_printed_document_is_read_and_a_source_for_another_archive_is_refused(env, tmp_path,
                                                                                           monkeypatch):
    # §7.4: `python -m lofi_rl.convert.kit` prints one YAML document, its report in comment lines, then `sources`
    # (the archive's entry, with its sha256), `choke` and `lanes`. Saved as derived/kits/<kit>.yaml, that whole
    # document is what the generator reads (2026-10-31 dry run: the two real kits' printed files were refused).
    assets = tmp_path / "assets"
    for src in yaml.safe_load(env["base"].read_text())["sources"].values():
        if src["mode"] == "fetch":
            (assets / src["path"]).parent.mkdir(parents=True, exist_ok=True)
            (assets / src["path"]).write_bytes((env["assets"] / src["path"]).read_bytes())
    tars = {k: kit_archive(assets, k, printed=True) for k in FILE_KITS}
    monkeypatch.setenv("LOFI_ASSETS", str(assets))
    got = generate(PLAN_PATH, tmp_path / "a", stage="day1", levels=None, base=env["base"])
    assert {r["instruments"]["drums"] for r in got.index["runs"]} == set(PLAN.stages["R1"].options)
    for kit in FILE_KITS:
        assert got.index["variants"][kit]["source_sha256"] == tars[kit]
    blocks = assets / "derived" / "kits" / "kit_swirly.yaml"
    blocks.write_text(blocks.read_text().replace(f"sha256: {tars['kit_swirly']}", "sha256: " + "0" * 64))
    with pytest.raises(GenerateError, match="kit_swirly.*sources.*sha256"):
        generate(PLAN_PATH, tmp_path / "b", stage="day1", levels=None, base=env["base"])
    blocks.write_text(yaml.safe_dump({"lanes": {}, "choke": [], "extra": 1}))
    with pytest.raises(GenerateError, match="kit_swirly"):
        generate(PLAN_PATH, tmp_path / "c", stage="day1", levels=None, base=env["base"])


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
def a_draw(tmp_path: Path, seed: int = 0x5EED) -> tuple[Path, dict]:
    """The job's draw.json for Day 1 (lofi_rl.shootout.draw, the batch builder's draw, §2.3 and §7.6)."""
    doc = draw(PLAN, "day1", seed=seed)
    return write_draw(doc, tmp_path / "draw.json"), doc


def _rewrite(path: Path, edit) -> Path:
    doc = json.loads(path.read_text())
    edit(doc)
    out = path.with_name(f"edited-{len(list(path.parent.glob('edited-*')))}.json")
    out.write_text(json.dumps(doc))
    return out


def test_with_the_jobs_draw_every_palette_it_asks_for_is_written_under_its_key(env, tmp_path):
    # The render driver reads <palettes>/<key>.yaml for each key of the draw's `palettes` table (§7.7). On Day 1
    # every clip plays its own (CD-52): `<base>_<loop>`, or `<base>_<loop>_sour` for the anchor's sour copy, where the
    # base is the stage palette `<stage>_<option>` or R2's `r2_<kit>_<bass>`, written too (its median faders are the
    # frozen palette's).
    path, doc = a_draw(tmp_path)
    got = generate(PLAN_PATH, tmp_path / "out", stage="day1", levels=env["levels"], base=env["base"], draw=path)
    by_id = {p["id"]: p for p in got.index["palettes"]}
    bases = {spec["base"] for spec in doc["palettes"].values()}
    assert len(doc["palettes"]) == len(doc["renders"]) == 67 and len(bases) == 26
    assert set(by_id) == set(doc["palettes"]) | bases and len(by_id) == 67 + 26
    assert got.index["draw"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    slot_of = {"keys": "keys", "drums": "kit", "bass": "bass", "sax": "sax"}
    for key, spec in doc["palettes"].items():
        entry = by_id[key]
        assert entry["file"] == f"{key}.yaml" and (tmp_path / "out" / entry["file"]).is_file()
        assert {slot_of[b]: i for b, i in entry["instruments"].items()} == \
            {s: spec["band"][s] for s in slot_of.values()}, key
        assert load_palette(tmp_path / "out" / entry["file"], verify=False).id == key
    assert {r["palette"] for r in doc["renders"].values()} == set(doc["palettes"])


def test_each_clip_palette_is_its_stage_palette_with_the_clips_own_fader_on_the_bus_its_round_varies(env, tmp_path):
    # CD-52: the varied bus's fader comes from the candidate's measured loudness on the round's own loop (the levels
    # file's clips[case].F); every other bus keeps its instrument's median fader, every bus its trim.
    path, doc = a_draw(tmp_path)
    got = generate(PLAN_PATH, tmp_path / "out", stage="day1", levels=env["levels"], base=env["base"], draw=path)
    lv = json.loads(env["levels"].read_text())
    by_id = {p["id"]: p for p in got.index["palettes"]}
    for key, spec in doc["palettes"].items():
        e, base, want = by_id[key], by_id[spec["base"]], spec["level"]
        clip = lv["variants"][want["variant"]]["clips"][want["case"]]
        assert e["base"] == base["id"] and e["instruments"] == base["instruments"] and e["trim_db"] == base["trim_db"]
        assert e["instruments"][want["bus"]] == want["variant"]
        assert e["fader_db"] == {**base["fader_db"], want["bus"]: clip["F"]}, key
        assert e["level"] == {**want, "L": clip["L"], "F": clip["F"], "lufs": clip["lufs"], "lu": clip["lu"],
                              "target_lufs": clip["target_lufs"], "off_lu": clip["off_lu"], "tol_lu": 0.1}, key
        pal = yaml_load((tmp_path / "out" / e["file"]).read_text())
        assert pal["mix"]["bus_gain_db"] == e["fader_db"] and pal["id"] == key
        assert hashlib.sha256((tmp_path / "out" / e["file"]).read_bytes()).hexdigest() == e["sha256"]
    sour = [r for r in doc["renders"].values() if r["variant"] == "sour"]
    assert len(sour) == 1 and doc["palettes"][sour[0]["palette"]]["level"]["case"] == f"{sour[0]['loop']}.sour"
    bases = {spec["base"] for spec in doc["palettes"].values()}
    assert all("level" not in p and "base" not in p for p in got.index["palettes"] if p["id"] in bases)


def test_within_each_drawn_round_the_clip_palettes_differ_only_in_the_candidates_block_trim_and_fader(env, tmp_path):
    # §2.1 on what the job renders: the palettes of a round's clips, each with its own fader, still differ only in
    # the candidate's block, its trim, its fader and the palette's name; the loop texts only in an anchor's sour copy.
    path, doc = a_draw(tmp_path)
    generate(PLAN_PATH, tmp_path / "out", stage="day1", levels=env["levels"], base=env["base"], draw=path)
    rounds = [r for b in doc["batches"] for r in b["rounds"]]
    assert len(rounds) == 19
    for r in rounds:
        clips = [doc["renders"][c["render"]] for c in r["clips"]]
        docs = [(c["option"], flat(yaml_load((tmp_path / "out" / f"{c['palette']}.yaml").read_text()))) for c in clips]
        for i, (oa, a) in enumerate(docs):
            for ob, b in docs[i + 1:]:
                diff = {k for k in a.keys() | b.keys() if a.get(k, "<absent>") != b.get(k, "<absent>")}
                bad = sorted(k for k in diff if not k.startswith(ALLOWED[r["stage"]]))
                assert not bad, (r["slot"], oa, ob, bad[:5])
                assert diff - {"id"}, (r["slot"], oa, ob, "two clips of a round must differ")
        texts = {c["text"] for c in clips if c["variant"] != "sour"}
        assert len(texts) == 1 and (r["kind"] == "anchor") == any(c["variant"] == "sour" for c in clips)


def test_a_levels_file_without_a_drawn_clip_or_with_a_clip_off_its_target_is_refused(env, tmp_path):
    # Every clip the draw makes needs its own fader from the level tool (run with --draw), and none may sit more
    # than clip_tol_lu from S* + T after matching (§3 step 6c): the generator refuses before any render.
    path, doc = a_draw(tmp_path)
    sour = next(r for r in doc["renders"].values() if r["variant"] == "sour")
    want = doc["palettes"][sour["palette"]]["level"]
    short = json.loads(env["levels"].read_text())
    del short["variants"][want["variant"]]["clips"][want["case"]]
    (tmp_path / "short.json").write_text(json.dumps(short))
    with pytest.raises(GenerateError, match=rf"{want['variant']} on {re.escape(want['case'])}.*--draw"):
        generate(PLAN_PATH, tmp_path / "a", stage="day1", levels=tmp_path / "short.json", base=env["base"], draw=path)
    off = json.loads(env["levels"].read_text())
    off["variants"][want["variant"]]["clips"][want["case"]]["off_lu"] = 0.13
    (tmp_path / "off.json").write_text(json.dumps(off))
    with pytest.raises(GenerateError, match=r"0\.13.*0\.1 LU"):
        generate(PLAN_PATH, tmp_path / "b", stage="day1", levels=tmp_path / "off.json", base=env["base"], draw=path)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists(), "a refused generation writes nothing"


def test_a_draw_off_its_plan_or_its_constraints_is_refused(env, tmp_path):
    path, _ = a_draw(tmp_path)

    def run(p):
        return generate(PLAN_PATH, tmp_path / p.stem, stage="day1", levels=env["levels"], base=env["base"], draw=p)

    def swap_s1(d):                       # t1 (a tenor symbol) and a1 (an alto symbol) trade options
        tenor, alto = d["maps"]["S1"]
        tenor["t1"], alto["a1"] = alto["a1"], tenor["t1"]

    def split_k1(d):                      # twins P1b and P2a trade options: P1 no longer holds one family
        k1 = d["maps"]["K1"][0]
        k1["P1b"], k1["P2a"] = k1["P2a"], k1["P1b"]

    def other_plan(d):
        d["plan"]["sha256"] = "0" * 64

    def extra_palette(d):
        d["palettes"]["k1_nonesuch_l5"] = d["palettes"]["k1_jrhodes3d_l5"]

    def other_base(d):
        d["palettes"]["k1_jrhodes3d_l5"]["base"] = "k1_nonesuch"

    def other_band(d):
        d["palettes"]["k1_jrhodes3d_l5"]["band"]["bass"] = "bass_vsco2_pizz"

    def other_bus(d):
        d["palettes"]["k1_jrhodes3d_l5"]["level"]["variant"] = "gu_organ"

    def other_day(d):
        d["stage"] = "day2"

    for edit, match in ((swap_s1, "S1.*t1.*tenor"), (split_k1, "K1.*P1.*family"), (other_plan, "another plan"),
                        (extra_palette, "k1_nonesuch_l5.*k1_jrhodes3d_l5"), (other_base, "k1_nonesuch"),
                        (other_band, "k1_jrhodes3d_l5.*bass"), (other_bus, "gu_organ.*k1_jrhodes3d plays jrhodes3d"),
                        (other_day, "day2")):
        with pytest.raises(GenerateError, match=match):
            run(_rewrite(path, edit))


# ------------------------------------------------------------------------------------- Day 2 (CD-53)
LEADERS2 = "keys=jrhodes3d+gu_tine_ep+fp_fm_ep1/kit=kit_swirly/bass=bass_fp_finger_yr/sax=sax_mtg_tenor+sax_gu_alto"


@pytest.fixture(scope="module")
def env2(env, tmp_path_factory):
    """A Day-2 draw from LEADERS2, its measuring palettes, a synthetic Day-2 levels file and the stage palettes."""
    tmp = tmp_path_factory.mktemp("day2")
    doc = draw(PLAN, "day2", seed=0xD2, leaders=parse_leaders(LEADERS2))
    path = write_draw(doc, tmp / "draw.json")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOFI_ASSETS", str(env["assets"]))
        measure = generate(PLAN_PATH, tmp / "palettes", stage="day2", levels=None, base=env["base"], draw=path)
        levels = synthetic_levels_day2(tmp / "palettes" / "measure" / "index.json", env["base"], doc,
                                       tmp / "levels.json")
        stage = generate(PLAN_PATH, tmp / "palettes", stage="day2", levels=levels, base=env["base"], draw=path)
    return {"tmp": tmp, "draw": doc, "draw_path": path, "measure": measure, "levels": levels, "stage": stage,
            "out": tmp / "palettes"}


def test_day2_measures_every_instrument_its_draw_plays_and_every_palette_a_matched_clip_plays(env, env2):
    # Section 3 "Drift between jobs": job 2 measures again every instrument it plays (the four level anchors first),
    # each in one run; CD-53: each stage palette a matched clip plays is a measuring palette of its own, at trims
    # and faders 0, with its option's settings (and the vamps' sax mute).
    idx = env2["measure"].index
    doc = env2["draw"]
    played = {p["band"][s] for p in doc["palettes"].values() for s in ("keys", "kit", "bass", "sax")}
    assert set(idx["variants"]) == played | set(PLAN.levels.anchors.values())
    assert idx["runs"][0]["instruments"] == {b: PLAN.levels.anchors[b] for b in ("sax", "keys", "bass", "drums")}
    measured = sorted(r["instruments"][b] for r in idx["runs"] for b in r["measure"])
    assert measured == sorted(idx["variants"]), "every instrument in exactly one run"
    bases = {p["base"] for p in doc["palettes"].values()}
    assert set(idx["configs"]) == bases and len(bases) == 4 + 3 + 2 + 3 + 3 + 3 + 1
    assert idx["draw"]["sha256"] == hashlib.sha256(env2["draw_path"].read_bytes()).hexdigest()
    for key, c in idx["configs"].items():
        st = PLAN.stages[c["stage"]]
        assert c["buses"] == list(st.match) and c["loops"] == st.match_loops, key
        pal = load_palette(env2["out"] / "measure" / c["file"], verify=True)
        assert pal.id == f"measure_day2_{key}" and set(pal.mix.bus_gain_db.values()) == {0.0}
        assert all(pal.mix.buses[b][0].params == {"gain_db": 0.0} for b in ("keys", "bass"))
        d = yaml_load((env2["out"] / "measure" / c["file"]).read_text())
        assert d["groove"]["swing"] == ({"default": 0.58} if key == "r3a_swing_58" else
                                        PLAN.variants["swing_split"].set["groove.swing"] if key == "r3a_swing_split"
                                        else {"default": 0.5} if key == "r3a_swing_50" else {"default": 0.55}), key
        assert d["groove"]["humanize"] == {"r3b_humanize_h1": "bar", "r3b_humanize_h2": "loop"}.get(key, "off")
        assert d["mix"]["sax_warm"] == (key == "s2a_sax_warm_on") and d["sax_expr"]["enabled"] == (key != "s2a_sax_expr_off")
        assert d["sax_expr"]["timing"] == {"s2b_timing_lag": "lag", "s2b_timing_swung": "swung"}.get(key, "grid")
        muted = d["mix"]["buses"]["sax"][1] == ["Gain", {"gain_db": -200.0}]
        assert muted == (key == "v") and (d["loudness"]["iters"] == 16) == (key == "v"), key


def test_day2_stage_palettes_carry_the_band_and_each_clip_its_own_fader_on_every_bus_it_matches(env2):
    lv = json.loads(env2["levels"].read_text())
    by_id = {p["id"]: p for p in env2["stage"].index["palettes"]}
    doc = env2["draw"]
    assert set(by_id) == set(doc["palettes"]) | {p["base"] for p in doc["palettes"].values()}
    assert len(doc["palettes"]) == len(doc["renders"]) == 84
    for key, spec in doc["palettes"].items():
        e, base = by_id[key], by_id[spec["base"]]
        for bus, inst in base["instruments"].items():           # the stage palette: the band's t and median F
            assert base["trim_db"][bus] == lv["variants"][inst]["t"] and base["fader_db"][bus] == lv["variants"][inst]["F"]
        cfg = lv["configs"][spec["base"]]["buses"]
        want = {**base["fader_db"], **{b: cfg[b]["cases"][spec["level"]["case"]]["F"] for b in spec["level"]["buses"]}}
        assert e["fader_db"] == want and e["trim_db"] == base["trim_db"] and e["settings"] == base["settings"], key
        assert e["level"]["case"] == spec["level"]["case"] and list(e["level"]["buses"]) == spec["level"]["buses"]
        for b, x in e["level"]["buses"].items():
            rec = cfg[b]["cases"][spec["level"]["case"]]
            assert x == {"instrument": rec["instrument"], "L": rec["L"], "F": rec["F"], "lufs": rec["lufs"],
                         "lu": rec["lufs"] - lv["S_star"], "target_lufs": rec["target_lufs"], "off_lu": rec["off_lu"],
                         "tol_lu": 0.1}, (key, b)
        pal = yaml_load((env2["out"] / e["file"]).read_text())
        assert pal["mix"]["bus_gain_db"] == e["fader_db"] and pal["id"] == key
    assert all(by_id[k]["sax_muted"] == (by_id[k]["stage"] == "V") for k in by_id)


ALLOWED2 = {   # CD-53 on §2.1: the option (a block or its settings), its trim, and the faders of the matched buses
    "R3a": ("id", "groove.swing.", "mix.bus_gain_db."),
    "R3b": ("id", "groove.humanize", "mix.bus_gain_db.keys", "mix.bus_gain_db.bass", "mix.bus_gain_db.drums"),
    "S1f": ("id", "sax.", "sax_lead", "mix.buses.sax.0.", "mix.bus_gain_db.sax"),
    "S2a": ("id", "mix.sax_warm", "sax_expr.enabled", "mix.bus_gain_db.sax"),
    "S2b": ("id", "sax_expr.timing", "mix.bus_gain_db.sax"),
    "K2": ("id", "keys.ep.", "mix.buses.keys.0.", "mix.bus_gain_db.keys"),
    "V": ("id", "mix.bus_gain_db.keys", "mix.bus_gain_db.bass"),
}


def test_within_each_day2_round_the_clip_palettes_differ_only_in_the_option_and_the_faders_it_matches(env2):
    doc = env2["draw"]
    rounds = [r for b in doc["batches"] for r in b["rounds"]]
    assert len(rounds) == 26
    for r in rounds:
        clips = [doc["renders"][c["render"]] for c in r["clips"]]
        docs = [(c["option"], flat(yaml_load((env2["out"] / f"{c['palette']}.yaml").read_text()))) for c in clips]
        for i, (oa, a) in enumerate(docs):
            for ob, b in docs[i + 1:]:
                diff = {k for k in a.keys() | b.keys() if a.get(k, "<absent>") != b.get(k, "<absent>")}
                bad = sorted(k for k in diff if not k.startswith(ALLOWED2[r["stage"]]))
                assert not bad, (r["slot"], oa, ob, bad[:5])
        texts = {c["text"] for c in clips if c["variant"] != "sour"}
        assert len(texts) == (4 if r["stage"] == "V" else 1), r["slot"]     # a vamp round: one text per vamp


def test_day2_needs_its_draw_and_a_measured_fader_for_every_clip_on_its_instrument(env, env2, tmp_path):
    with pytest.raises(GenerateError, match="day2.*draw"):
        generate(PLAN_PATH, tmp_path / "a", stage="day2", levels=None, base=env["base"])
    lv = json.loads(env2["levels"].read_text())
    spec = next(p for p in env2["draw"]["palettes"].values() if p["stage"] == "V")
    case = spec["level"]["case"]
    for name, edit, said in (
            ("short", lambda d: d["configs"]["v"]["buses"]["bass"]["cases"].pop(case), f"no bass fader for v on {case}"),
            ("off", lambda d: d["configs"]["v"]["buses"]["bass"]["cases"][case].update(off_lu=0.12), r"0\.120 LU"),
            ("other", lambda d: d["configs"]["v"]["buses"]["keys"]["cases"][case].update(instrument="gu_organ"),
             "measured gu_organ on keys"),
            ("unmeasured", lambda d: d["configs"].pop("v"), "did not measure stage palette v")):
        d = copy.deepcopy(lv)
        edit(d)
        (tmp_path / f"{name}.json").write_text(json.dumps(d))
        with pytest.raises(GenerateError, match=said):
            generate(PLAN_PATH, tmp_path / name, stage="day2", levels=tmp_path / f"{name}.json", base=env["base"],
                     draw=env2["draw_path"])
        assert not (tmp_path / name).exists(), "a refused generation writes nothing"
    cut = draw(PLAN, "day2", seed=0xD2, leaders={**parse_leaders(LEADERS2), "sax": ("sax_mtg_tenor",)},
               cuts=["vamps-high", "latin3", "final"])
    got = generate(PLAN_PATH, tmp_path / "cut", stage="day2", levels=None, base=env["base"],
                   draw=write_draw(cut, tmp_path / "cut.json"))
    assert not any(k.startswith("s1f_") for k in got.index["configs"]), "the cut plan has no final"


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
