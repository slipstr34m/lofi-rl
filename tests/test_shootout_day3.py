"""Day 3, job 3 (docs/SHOOTOUTS.md sections 4.5, 7.2, 7.6-7.8): the rain rounds W1 and W2 on the Day-2 picks, the
anchor and the five repeats, through the generator, the level tool, the render record and the batch builder.

Everything runs on synthetic assets in tmp_path (tests/helpers_shootout.py: tiny SoundFonts and kit archives, a
one-second noise file standing in for the converted rain) and on fake renders; the laptop dry run of section 7.7
renders the real thing. Sound claims: none."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from helpers_shootout import FILE_KITS, PLAN_PATH, ROOT, fake_renders, kit_archive, synthetic_base, synthetic_levels
from lofi_rl.palette import load_palette
from lofi_rl.render.loudness import lufs
from lofi_rl.render.wavio import pcm_sha256
from lofi_rl.shootout import render as R
from lofi_rl.shootout.batch import BuildError, build, public_leaks, _stage_options
from lofi_rl.shootout.draw import draw, load_prior, parse_leaders, write_draw
from lofi_rl.shootout.levels import run_levels
from lofi_rl.shootout.palettes import RAIN_SOURCE, GenerateError, generate, yaml_load
from lofi_rl.shootout.plan import load_plan
from lofi_rl.ui.manifest import load_batch

PLAN = load_plan(PLAN_PATH)
# The Day-2 picks as job 3 takes them (section 7.7): one id per instrument slot and every variant slot named
LEADERS3 = ("keys=gu_tine_ep/kit=kit_virtuosity/bass=bass_vsco2_pizz/sax=sax_mtg_alto/swing=swing_55/"
            "humanize=humanize_off/sax_variant=sax_plain/timing=timing_lag")
DAY2 = parse_leaders("keys=gu_tine_ep+gs_wurli_ep200+vcsl_upright_knight/kit=kit_virtuosity+kit_gu_jazz/"
                     "bass=bass_gu_acoustic/sax=sax_mtg_alto+sax_mtg_tenor")
STAGES = ["K1", "R1", "R2", "S1", "R2r", "R3a", "R3b", "S1f", "S2a", "S2b", "K2", "V"]
EXCERPTS = {"e1": 150.4, "e2": 44.3, "e3": 77.8}
DELTA = {"rain_d18": 18.0, "rain_d24": 24.0, "rain_d30": 30.0}


def rain_record(assets: Path, folder: Path) -> Path:
    """A one-second stand-in for the converted rain under `assets`, and an excerpt record pinning it, shaped like
    lofi_rl.convert.rain_excerpts' (the generator reads only its source and each excerpt's start_s)."""
    wav = assets / "derived" / "rain" / "rain44.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), np.random.default_rng(1).standard_normal((44100, 2)).astype(np.float32) * 0.05, 44100,
             subtype="FLOAT")
    doc = {"schema": "lofi-rl/rain-excerpts/1", "tool": "python -m lofi_rl.convert.rain_excerpts",
           "parameters": {"n": 3, "warmup_s": 0.5, "loop_s": 32.0, "xfade_s": 1.0, "stretch_s": 33.5,
                          "block_s": 0.4, "hop_s": 0.1},
           "source": {"asset_path": "derived/rain/rain44.wav", "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                      "license": "CC0 1.0", "url": "https://example.invalid/rain",
                      "converted_from": {"path": "assets/big/rain/rain48.wav", "sha256": "0" * 64, "sr": 48000},
                      "attribution": None},
           "excerpts": {k: {"start_s": v} for k, v in EXCERPTS.items()}}
    path = folder / "rain_excerpts.json"
    path.write_text(json.dumps(doc, indent=1))
    return path


@pytest.fixture(scope="module")
def env3(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("day3")
    assets = tmp / "assets"
    base = synthetic_base(tmp, assets)
    for kit in FILE_KITS:
        kit_archive(assets, kit)
    rain = rain_record(assets, tmp)
    doc = draw(PLAN, "day3", seed=0xD3, leaders=parse_leaders(LEADERS3))
    path = write_draw(doc, tmp / "draw.json")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOFI_ASSETS", str(assets))
        measure = generate(PLAN_PATH, tmp / "palettes", stage="day3", levels=None, base=base, draw=path, rain=rain)
        levels = synthetic_levels(tmp / "palettes" / "measure" / "index.json", base, tmp / "levels.json")
        lv = json.loads(levels.read_text())
        levels.write_text(json.dumps(dict(lv, stage="day3")))
        stage = generate(PLAN_PATH, tmp / "palettes", stage="day3", levels=levels, base=base, draw=path, rain=rain)
    return {"tmp": tmp, "assets": assets, "base": base, "rain": rain, "draw": doc, "draw_path": path,
            "measure": measure, "levels": levels, "stage": stage, "out": tmp / "palettes"}


@pytest.fixture(autouse=True)
def assets_env(request, monkeypatch):
    if "env3" in request.fixturenames:
        monkeypatch.setenv("LOFI_ASSETS", str(request.getfixturevalue("env3")["assets"]))


def flat(node, prefix=""):
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


# ------------------------------------------------------------------------------------- the generator (7.8)
def test_day3_measures_the_day2_picks_again_and_no_rain_palette(env3):
    # section 3 "Drift between jobs": job 3 measures every instrument it plays, the level anchors first; the rain
    # rounds match nothing (section 3 "Not matched"), so there is no measuring config and no bed in a measuring run
    idx = env3["measure"].index
    assert set(idx["variants"]) == {"gu_tine_ep", "kit_virtuosity", "bass_vsco2_pizz", "sax_mtg_alto"} | \
        set(PLAN.levels.anchors.values())
    assert [r["run"] for r in idx["runs"]] == ["m1", "m2"] and idx["configs"] == {}
    for r in idx["runs"]:
        d = yaml_load((env3["out"] / "measure" / r["file"]).read_text())
        assert d["bed"] is None and RAIN_SOURCE not in d["sources"]


def test_each_rain_palette_is_the_day2_band_at_its_median_faders_with_its_own_bed(env3):
    lv = json.loads(env3["levels"].read_text())["variants"]
    by_id = {p["id"]: p for p in env3["stage"].index["palettes"]}
    assert sorted(by_id) == sorted(env3["draw"]["palettes"]) == [
        "w1_excerpt_e1", "w1_excerpt_e2", "w1_excerpt_e3", "w2_rain_d18", "w2_rain_d24", "w2_rain_d30", "w2_rain_none"]
    rain = json.loads(env3["rain"].read_text())
    for key, e in by_id.items():
        assert e["instruments"] == {"sax": "sax_mtg_alto", "keys": "gu_tine_ep", "bass": "bass_vsco2_pizz",
                                    "drums": "kit_virtuosity"}
        for bus, inst in e["instruments"].items():                 # the band's t and median F: the frozen values
            assert (e["trim_db"][bus], e["fader_db"][bus]) == (lv[inst]["t"], lv[inst]["F"]), (key, bus)
        option = key.split("_", 1)[1]
        own = option if key.startswith("w2_") else "rain_d18"
        excerpt = option.removeprefix("excerpt_") if key.startswith("w1_") else "e1"
        want = None if own == "rain_none" else {"src": RAIN_SOURCE, "start_s": EXCERPTS[excerpt], "xfade_s": 1.0,
                                                "delta_lu": DELTA[own]}
        d = yaml_load((env3["out"] / e["file"]).read_text())
        assert d["bed"] == e["bed"] == want, key
        # the Day-2 picks' settings, then the bed variants with the stage's own option last (band_settings)
        stage_band = "rain_d18" if key.startswith("w1_") else "excerpt_e1"
        assert e["settings"] == ["swing_55", "humanize_off", "sax_plain", "timing_lag", stage_band, option], key
        assert d["sax_expr"]["timing"] == "lag" and d["groove"]["swing"] == {"default": 0.55}
        assert d["groove"]["humanize"] == "off" and d["mix"]["sax_warm"] is False and d["sax_expr"]["enabled"]
        if want is None:
            assert RAIN_SOURCE not in d["sources"], "no bed, no rain source"
        else:
            src = d["sources"][RAIN_SOURCE]
            assert (src["path"], src["sha256"], src["mode"], src["license"]) == \
                (rain["source"]["asset_path"], rain["source"]["sha256"], "fetch", "CC0 1.0")
        pal = load_palette(env3["out"] / e["file"], verify=True)       # every source hashed, the rain file's too
        assert (pal.bed is None) == (want is None)
    assert env3["stage"].index["rain"]["sha256"] == hashlib.sha256(env3["rain"].read_bytes()).hexdigest()


def test_within_each_rain_round_the_palettes_differ_only_in_the_bed(env3):
    # section 2.1: one thing only. The music is the same in every clip of a rain round (the band, its faders and the
    # loop text); only the bed, its source and the palette's name differ
    doc = env3["draw"]
    rounds = [r for b in doc["batches"] for r in b["rounds"] if r["kind"] == "none"]
    assert sorted(r["stage"] for r in rounds) == ["W1"] * 3 + ["W2"] * 4
    for r in rounds:
        clips = [doc["renders"][c["render"]] for c in r["clips"]]
        assert len({c["text"] for c in clips}) == 1
        docs = [flat(yaml_load((env3["out"] / f"{c['palette']}.yaml").read_text())) for c in clips]
        for i, a in enumerate(docs):
            for b in docs[i + 1:]:
                diff = {k for k in a.keys() | b.keys() if a.get(k, "<absent>") != b.get(k, "<absent>")}
                assert diff and all(k == "id" or k == "bed" or k.startswith(("bed.", f"sources.{RAIN_SOURCE}"))
                                    for k in diff), (r["slot"], sorted(diff)[:6])


def test_a_missing_or_changed_rain_file_or_an_unknown_excerpt_stops_the_generator(env3, tmp_path, monkeypatch):
    args = dict(stage="day3", levels=None, base=env3["base"], draw=env3["draw_path"])
    gone = tmp_path / "gone.json"
    doc = json.loads(env3["rain"].read_text())
    gone.write_text(json.dumps({**doc, "source": {**doc["source"], "asset_path": "derived/rain/none.wav"}}))
    with pytest.raises(GenerateError, match="is missing"):
        generate(PLAN_PATH, tmp_path / "a", rain=gone, **args)
    other = tmp_path / "other.json"
    other.write_text(json.dumps({**doc, "source": {**doc["source"], "sha256": "1" * 64}}))
    with pytest.raises(GenerateError, match="does not match"):
        generate(PLAN_PATH, tmp_path / "b", rain=other, **args)
    two = tmp_path / "two.json"
    two.write_text(json.dumps({**doc, "excerpts": {"e1": {"start_s": 1.0}, "e2": {"start_s": 2.0}}}))
    with pytest.raises(GenerateError, match="'e3' is not one of the picker's"):
        generate(PLAN_PATH, tmp_path / "c", stage="day3", levels=env3["levels"], base=env3["base"],
                 draw=env3["draw_path"], rain=two)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists() and not (tmp_path / "c" / "index.json").exists()


def test_a_bed_before_day3_is_refused(env3, tmp_path):
    doc = draw(PLAN, "day2", seed=0xD2, leaders=DAY2, adds=["R2r"])
    key = next(iter(doc["palettes"]))
    doc["palettes"][key]["band"]["rain"] = "rain_d24"
    path = write_draw(doc, tmp_path / "draw.json")
    with pytest.raises(GenerateError, match="the rain comes on day 3"):
        generate(PLAN_PATH, tmp_path / "p", stage="day2", levels=None, base=env3["base"], draw=path)


def test_the_s2_rerun_is_measured_and_written_as_on_day2(env3, tmp_path):
    # section 4.4: when the final picks the other finalist, S2a and S2b run again on Day 3 with it, each clip its own
    # sax fader as on Day 2 (CD-53), beside the rain palettes
    from helpers_shootout import synthetic_levels_day2
    doc = draw(PLAN, "day3", seed=0xD3, leaders=parse_leaders(LEADERS3), rerun_s2="sax_mtg_tenor")
    path = write_draw(doc, tmp_path / "draw.json")
    m = generate(PLAN_PATH, tmp_path / "palettes", stage="day3", levels=None, base=env3["base"], draw=path,
                 rain=env3["rain"])
    assert "sax_mtg_tenor" in m.index["variants"]
    assert {c["stage"] for c in m.index["configs"].values()} == {"S2a", "S2b"}
    lv = synthetic_levels_day2(tmp_path / "palettes" / "measure" / "index.json", env3["base"], doc, tmp_path / "lv.json")
    lv.write_text(json.dumps(dict(json.loads(lv.read_text()), stage="day3")))
    st = generate(PLAN_PATH, tmp_path / "palettes", stage="day3", levels=lv, base=env3["base"], draw=path,
                  rain=env3["rain"])
    s2 = [p for p in st.index["palettes"] if p["stage"] in ("S2a", "S2b") and p.get("level")]
    assert len(s2) == 18 and {p["instruments"]["sax"] for p in s2} == {"sax_mtg_tenor"}
    assert all(p.get("bed", None) is None for p in s2)


# ------------------------------------------------------------------------------------- the level tool (7.1)
class HashBench:
    """Made-up loudness for any (instrument, case), as tests/test_shootout_levels.py's HashBench."""

    def __init__(self):
        self.prepared, self.kept = [], {}

    @staticmethod
    def _u(*keys) -> float:
        return int(hashlib.sha256("|".join(keys).encode()).hexdigest()[:8], 16) / 2 ** 32

    def prepare(self, run, cases, buses):
        self.prepared.append(run.id)

    def input_lufs(self, run, case, bus):
        return -45.0 + 6.0 * self._u("P", run.instruments[bus], case.name)

    def chain_lufs(self, run, case, bus, trim, keep=False):
        out = -40.0 + 8.0 * self._u("L", run.instruments[bus], case.name) + trim
        if keep:
            self.kept[(run.id, case.name, bus)] = out
        return out

    def post_lufs(self, run, case, bus, fader):
        return self.kept[(run.id, case.name, bus)] + fader

    def release(self, run):
        pass


def test_the_level_tool_measures_the_day3_band_against_job_1_and_matches_no_rain_round(env3, tmp_path):
    doc = run_levels(PLAN_PATH, ROOT / "shootouts" / "loops", env3["out"] / "measure", "day3", draw=env3["draw_path"],
                     bench=HashBench(), stamp="a1b2c3d 0 0")
    assert set(doc["variants"]) == set(env3["measure"].index["variants"])
    assert doc["configs"] == {} and doc["rounds"] == [] and doc["flags"] == []
    prior = tmp_path / "job1.json"
    prior.write_text(json.dumps(dict(doc, stage="day1", drift={**doc["drift"], "stamp": "job 1"})))
    again = run_levels(PLAN_PATH, ROOT / "shootouts" / "loops", env3["out"] / "measure", "day3",
                       draw=env3["draw_path"], prior=prior, bench=HashBench(), stamp="a1b2c3d 0 0")
    assert again["reused"] is False and again["prior"]["changed"] == ["stamp"]
    assert all(r["moved_db"] == {"t": 0.0, "F": 0.0} for r in again["variants"].values())


# ------------------------------------------------------------------------------------- the render record (7.7)
def test_a_bed_clips_record_carries_its_bed_measured_in_the_render_and_its_music():
    rng = np.random.default_rng(5)
    music = (rng.standard_normal((2, 44100 * 3)) * 0.1).astype(np.float32)
    bed = (rng.standard_normal((2, 44100 * 3)) * 0.1 * 10 ** (-18 / 20)).astype(np.float32)
    report = {"src": "rain", "delta_lu": 18.0, "start_s": 150.4, "xfade_s": 1.0, "xfade_samples": 44100,
              "lufs_source": -30.0, "gain_db": -2.0, "lufs": -32.0}
    got = R.bed_record({"bed": report, "music_lufs": -14.2}, music + bed, music, music)
    assert got["bed"]["bed_lu"] == pytest.approx(lufs(bed, 44100) - lufs(music, 44100), abs=1e-9)
    assert {k: v for k, v in got["bed"].items() if k != "bed_lu"} == report
    assert got["music_lufs"] == -14.2 and got["pre_limiter_sha256"] == pcm_sha256(music)
    assert R.bed_record({"bed": None, "music_lufs": -14.0}, music, music, music)["bed"] is None
    # through the driver's record: a bed stage's clip gets the fields, any other stage's record keeps its keys
    compiled = SimpleNamespace(sax=SimpleNamespace(notes=()), kit="dusty")
    man = {"buses": {b: {"lufs": -20.0, "gain_db": 0.0} for b in ("sax", "keys", "bass", "drums")}, "n": 3, "sr": 44100,
           "loudness": {"lufs": -14.0, "true_peak_dbtp": -1.5, "gain_db": 3.0}, "audio_sha256": "a" * 64,
           "music_sha256": "b" * 64, "loop": {"canon_sha256": "c" * 64, "compiled_sha256": "d" * 64},
           "gates": {"G1-G6": "pass", "G7": "pass"}, "bed": report, "music_lufs": -14.2,
           "schema": "lofi-rl/render/1", "chain": {}, "brief": {"bpm": 76, "tonic": 9, "mode": "aeolian", "energy": "low"},
           "palette": {"id": "w1_excerpt_e1", "sha256": "e" * 64}}
    r = SimpleNamespace(manifest=man, compiled=compiled, audio=music + bed, music=music, pre_limiter=music)
    pal = SimpleNamespace(id="w1_excerpt_e1", sha256="e" * 64, kits={"dusty": SimpleNamespace(
        lanes={"kick": SimpleNamespace(db=0.0)})}, mix=SimpleNamespace(buses={b: () for b in ("sax", "keys", "bass")}))
    entry = {"loop": "l1_melancholy_low", "brief": {"bpm": 76}, "stage": "W1", "option": "excerpt_e1", "what": "x",
             "bus": "bed", "variant": "base", "palette": "w1_excerpt_e1"}
    rec = R._record("W1.l1_melancholy_low.excerpt_e1", entry, pal, "ok", None, r, "f" * 64, 3)
    assert rec["bed"]["delta_lu"] == 18.0 and rec["bed_lufs"] == -32.0 and rec["music_lufs"] == -14.2
    other = R._record("S1.l1.x", {**entry, "bus": "sax", "stage": "S1"}, pal, "ok", None, r, "f" * 64, 3)
    assert set(other) == set(R.RECORD_KEYS)


# ------------------------------------------------------------------------------------- the batches (7.6)
def _days(tmp: Path) -> list:
    """Day 1 and Day 2 built from fake renders (the priors the repeats read), as job 1 and job 2 left them."""
    priors = []
    for name, day, seed, kw in (("shootout-201", "day1", 21, {}), ("shootout-202", "day2", 22,
                                                                   {"leaders": DAY2, "adds": ["R2r"]})):
        run = tmp / "runs" / name
        doc = draw(PLAN, day, seed=seed, **kw)
        write_draw(doc, run / "summary" / "draw.json")
        for key in doc["palettes"]:
            pal = run / "summary" / "palettes" / f"{key}.yaml"
            pal.parent.mkdir(parents=True, exist_ok=True)
            pal.write_text(f"# stand-in for palette {key}\n")
        fake_renders(doc, run / "renders")
        build(doc, run / "renders", tmp / "inbox", summary_dir=run / "summary", plan=PLAN, check=False)
        priors.append(load_prior(run / "summary" / "batches.json"))
    return priors


def test_day3_batches_record_each_rain_clips_bed_and_music_and_hold_the_five_repeats(tmp_path):
    priors = _days(tmp_path)
    doc = draw(PLAN, "day3", seed=23, leaders=parse_leaders(LEADERS3), priors=priors, repeats=STAGES)
    run = tmp_path / "runs" / "shootout-203"
    fake_renders(doc, run / "renders")
    folders = build(doc, run / "renders", tmp_path / "inbox", summary_dir=run / "summary", plan=PLAN)
    assert [f.name for f in folders] == ["b1106-1-mix", "b1106-2-mix"]
    kinds, groups = [], []
    for d in folders:
        m, h = load_batch(d)
        pub = json.loads((d / "manifest.json").read_text())
        hid = json.loads((d / "manifest.hidden.json").read_text())
        assert public_leaks(pub, hid, _stage_options(PLAN, {i["stage"] for r in hid["rounds"].values()
                                                             for i in r["clips"].values()})) == []
        for r in m.rounds:
            hr = h.rounds[r.id]
            kinds.append(hr.kind)
            infos = [hr.clips[c] for c in r.ids]
            if hr.kind == "repeat":
                groups.append(PLAN.stages[infos[0]["stage"]].repeat_group)
                continue
            assert infos[0]["stage"] in ("W1", "W2")
            for x in infos:
                assert {"music_lufs", "bed_lu", "bus_lufs"} <= set(x) and x["fader_db"] is None
                assert (x["bed_lu"] is None) == (x["option"] == "rain_none") == (x["bus_lufs"] is None)
    assert sorted(kinds) == ["anchor"] + ["none"] * 7 + ["repeat"] * 5
    assert sorted(groups) == sorted(PLAN.repeat_groups), "one repeat per group (section 2.5)"


def test_a_rain_round_whose_clips_play_different_music_is_refused(tmp_path):
    doc = draw(PLAN, "day3", seed=24, leaders=parse_leaders(LEADERS3))
    rid = next(c["render"] for b in doc["batches"] for r in b["rounds"] if r["kind"] == "none" for c in r["clips"])
    run = tmp_path / "runs" / "shootout-9"
    fake_renders(doc, run / "renders")
    rec_path = run / "renders" / rid / "record.json"
    rec = json.loads(rec_path.read_text())
    rec_path.write_text(json.dumps(dict(rec, pre_limiter_sha256="0" * 64)))
    with pytest.raises(BuildError, match="must play the same music"):
        build(doc, run / "renders", tmp_path / "inbox", summary_dir=run / "summary", plan=PLAN)
    assert not (tmp_path / "inbox").exists() or not any((tmp_path / "inbox").iterdir())


def test_a_purged_first_showing_made_again_builds_and_its_name_leaks_nothing(tmp_path):
    # A render scratch purged is made again from its first job's palette under `prior.<job>.<render>`; the run
    # folder's word "shootout" (the public manifest's mode) is not in that name, so the leak guard passes
    priors = _days(tmp_path)
    doc = draw(PLAN, "day3", seed=25, leaders=parse_leaders(LEADERS3), priors=priors, repeats=["K2"])
    rep = next(r for b in doc["batches"] for r in b["rounds"] if r["kind"] == "repeat")
    first = rep["clips"][0]
    old = Path(first["run_dir"]) / "renders" / first["render"]
    keep = (old / "render.flac").read_bytes()
    (old / "render.flac").unlink()
    try:
        doc2 = draw(PLAN, "day3", seed=25, leaders=parse_leaders(LEADERS3), priors=priors, repeats=["K2"])
    finally:
        (old / "render.flac").write_bytes(keep)
    rep2 = next(r for b in doc2["batches"] for r in b["rounds"] if r["kind"] == "repeat")
    new = rep2["clips"][0]["render"]
    assert new == f"prior.202.{first['render']}" and "shootout" not in new
    run = tmp_path / "runs" / "shootout-204"
    fake_renders(doc2, run / "renders")
    for c in rep2["clips"]:                     # the render made again equals its first showing (same pcm, same FLAC)
        if c["render"].startswith("prior."):
            src = Path(priors[1].run_dir) / "renders" / c["render"].split(".", 2)[2]
            dst = run / "renders" / c["render"]
            (dst / "render.flac").write_bytes((src / "render.flac").read_bytes())
            rec = json.loads((dst / "record.json").read_text())
            rec.update(clip=c["expect"]["clip"], pcm_sha256=c["expect"]["pcm_sha256"],
                       flac_sha256=c["expect"]["flac_sha256"])
            (dst / "record.json").write_text(json.dumps(rec))
            man = json.loads((dst / "manifest.json").read_text())
            (dst / "manifest.json").write_text(json.dumps(dict(man, audio_sha256=rec["pcm_sha256"])))
    folders = build(doc2, run / "renders", tmp_path / "inbox3", summary_dir=run / "summary", plan=PLAN)
    assert len(folders) == 2
