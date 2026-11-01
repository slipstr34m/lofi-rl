"""The draft palette (lofi_rl.drafts.palette; docs/DRAFTS.md §3).

Offline, so the suite needs no fetched asset: the committed palette parses, holds exactly the Day-2 picks and settings,
its `#| ` block gives every instrument the trim and fader the levels files measured (pinned below from
runs/shootout-13635646 and -13651610, summary/levels.json), each block is the instrument job 2 measured (source sha256,
bank and program; a kit's lanes by the shootout's kit-map hash), every render palette parses and carries its band's
levels and nothing else, and a build from inputs rebuilt from the palette reproduces it byte for byte."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from lofi_rl.compile.compiler import compile_loop
from lofi_rl.dsl import parse
from lofi_rl.drafts import palette as P
from lofi_rl.drafts.briefs import load as load_briefs
from lofi_rl.palette import parse_palette

ROOT = Path(__file__).resolve().parents[1]
BRIEFS = ROOT / "drafts" / "v1" / "briefs.yaml"
# t and median F per instrument (levels.json `variants`, equal in jobs 13635646 and 13651610) and each keys
# instrument's K2 spread median F (job 2, configs.k2_<keys>.buses.keys.loops.F); faders after CD-28's mean
LEVELS = {
    "keys": {"rhodes": ("gu_tine_ep", 0.0, -6.04, -5.84, -5.94), "wurli": ("gs_wurli_ep200", -3.22, -6.65, -6.22, -6.435),
             "piano": ("vcsl_upright_knight", 21.86, -6.06, -6.06, -6.06)},
    "kits": {"studio": ("kit_virtuosity", -14.94, 15.26), "jazz": ("kit_gu_jazz", 0.0, 15.21)},
    "basses": {"studio": ("bass_vsco2_pizz", 34.97, 5.09), "jazz": ("bass_fp_finger_yr", -2.79, 5.13)},
    "sax": ("sax_mtg_alto", 6.63, 2.55),
}
KIT_MAPS = {"studio": "e957b921d80525adc6c0c41f7636cbb35f23a752a109b94509d403733dfb3426",
            "jazz": "940bbb35d060fb7fb39f6ddf474b2abbfdeaaecabbf76402e44cc78c1cbdb67c"}
TAR_SHA = "5ca78cde9f96a1e42e1fd558195125529d40508b53cea1aeb2a34735ca66cd9e"
LOOP = """\
vamp: Im11 IV9 span 1
keys: wurli shell
comp: x-------...x.... ......x-----..x.
bass: x-----x...x..x.. x-------..x-x...
drums: jazz
kick: x.....x...x..... x.......x.x.....
snare: ....X.......x... ....X.......x...
hat: x.x.x.x.x.x.x.x. x.x.x.xxx.x.x.x.
ohat: ................ ..........x.....
@4 snare: ....X.......x..g
A: h5 q4 e3 e2 | q.1 e2 q3 qr | h.5 er e4 | q3 e2 e1 hr
B: h5 q4 e3 e2 | q.1 e7, q1 qr | q6 e5 e4 h3 | q2 e1 e2 hr
"""


@pytest.fixture(scope="module")
def dp():
    return P.load(P.DRAFT_PALETTE)


def test_the_committed_palette_has_no_problem(dp):
    assert dp.id == "draft_day2" and P.problems(dp) == []
    assert P.main(["check"]) == 0


def test_it_holds_the_day2_picks_under_plain_names(dp):
    pal = dp.palette
    assert list(dp.doc["keys"]) == ["rhodes", "wurli", "piano"] and list(dp.doc["kits"]) == ["studio", "jazz"]
    assert {k: v.option for k, v in dp.keys.items()} == {k: v[0] for k, v in LEVELS["keys"].items()}
    assert {k: (dp.kits[k].option, dp.basses[k].option) for k in dp.kits} == {
        k: (LEVELS["kits"][k][0], LEVELS["basses"][k][0]) for k in LEVELS["kits"]}
    assert set(pal.sax) == {"alto"} and pal.sax_lead == "alto" and dp.sax.option == "sax_mtg_alto"
    assert pal.sax["alto"].play == (49, 80) and pal.sax["alto"].home == (57, 68)
    assert str(pal.groove.swing) == "11/20" and not pal.groove.swing_lanes and pal.groove.humanize == "off"
    assert pal.sax_expr.enabled and pal.sax_expr.timing == "lag" and not pal.mix.sax_warm
    assert pal.bed is None and dict(pal.aliases) == {}
    assert pal.vocab().sax == frozenset({"alto"})


def test_only_gu_is_committed_and_every_other_source_is_fetch_only(dp):
    modes = {sid: s.mode for sid, s in dp.palette.sources.items()}
    assert modes.pop("gu") == "commit"
    assert set(modes.values()) == {"fetch"}
    assert dp.doc["sources"]["kit_virtuosity"] == {
        "path": "derived/kits/kit_virtuosity.tar", "sha256": TAR_SHA, "license": "CC0 1.0 (Virtuosity Drums v0.925)",
        "mode": "fetch", "url": None, "converted_from": "shootouts/kits/kit_virtuosity.yaml", "attribution": None}
    assert P.sources_of(dp.doc) == set(dp.doc["sources"])


def test_every_trim_and_fader_is_the_measured_one(dp):
    for name, (option, t, rootless, spread, fader) in LEVELS["keys"].items():
        rec = dp.levels["keys"][name]
        assert (rec["option"], rec["trim_db"], rec["rootless_F"], rec["spread_F"]) == (option, t, rootless, spread)
        assert dp.keys[name].fader_db == fader == P.mean2(rootless, spread)
        assert rec["measured"]["day1"] == rec["measured"]["day2"] == {"t": t, "F": rootless}
    for name, (option, t, f) in LEVELS["kits"].items():
        assert (dp.kits[name].option, dp.kits[name].trim_db, dp.kits[name].fader_db) == (option, t, f)
        bo, bt, bf = LEVELS["basses"][name]
        assert (dp.basses[name].option, dp.basses[name].trim_db, dp.basses[name].fader_db) == (bo, bt, bf)
    assert (dp.sax.option, dp.sax.trim_db, dp.sax.fader_db) == LEVELS["sax"]
    assert dp.levels["S_star"] == -31.924651028769418
    assert dp.levels["from"]["day1"]["sha256"].startswith("9cf78680")
    assert dp.levels["from"]["day2"]["sha256"].startswith("8f4f6e9f")


def test_the_mean_is_exact():
    assert P.mean2(-6.65, -6.22) == -6.435 and P.mean2(-6.04, -5.84) == -5.94 and P.mean2(1.0, 2.0) == 1.5


def test_each_kit_carries_its_trim_on_its_lanes_and_is_the_measured_kit_map(dp):
    for name, kit in dp.doc["kits"].items():
        trim = dp.kits[name].trim_db
        lanes = P.untrimmed_lanes(kit["lanes"], trim)
        assert P.canonical_sha256({"lanes": lanes, "choke": kit["choke"]}) == KIT_MAPS[name]
        assert all(kit["lanes"][lane]["db"] - lanes[lane]["db"] == pytest.approx(trim) for lane in lanes)
        assert kit["choke"] == [["ohat", "hat"], ["ohat", "ohat"]]


def test_as_written_it_plays_the_leaders_band(dp):
    mix = dp.doc["mix"]
    assert mix["bus_gain_db"] == {"sax": 2.55, "drums": 15.26, "bass": 5.09, "keys": -5.94}
    assert [mix["buses"][b][0] for b in ("sax", "keys", "bass")] == [
        ["Gain", {"gain_db": 6.63}], ["Gain", {"gain_db": 0.0}], ["Gain", {"gain_db": 34.97}]]
    lead = P.render_doc(dp, "rhodes", "studio")
    assert lead["mix"] == mix


@pytest.mark.parametrize("keys", ["rhodes", "wurli", "piano"])
@pytest.mark.parametrize("kit", ["studio", "jazz"])
def test_every_render_palette_parses_and_carries_its_band_only(dp, keys, kit):
    doc = P.render_doc(dp, keys, kit)
    text = P.render_text(dp, keys, kit)
    pal = parse_palette(yaml.safe_load(text), "x")
    assert pal.id == f"draft_day2_{keys}_{kit}" and set(pal.keys) == {keys} and set(pal.kits) == {kit}
    assert set(pal.sources) == P.sources_of(doc)
    k, (_, kt, kf), (_, bt, bf) = LEVELS["keys"][keys], LEVELS["kits"][kit], LEVELS["basses"][kit]
    assert dict(pal.mix.bus_gain_db) == {"sax": 2.55, "drums": kf, "bass": bf, "keys": k[4]}
    heads = {b: pal.mix.buses[b][0] for b in ("sax", "keys", "bass")}
    assert {b: (h.name, dict(h.params)) for b, h in heads.items()} == {
        "sax": ("Gain", {"gain_db": 6.63}), "keys": ("Gain", {"gain_db": k[1]}), "bass": ("Gain", {"gain_db": bt})}
    assert pal.kits[kit].lanes["kick"].db == pytest.approx(kt)
    rest = copy.deepcopy(doc)
    mine = copy.deepcopy(dp.doc)
    for d in (rest, mine):            # outside the band's levels, names and sources, the palette as written
        for key in ("id", "keys", "kits", "sources"):
            d.pop(key)
        d["mix"].pop("bus_gain_db")
        for b in ("sax", "keys", "bass"):
            d["mix"]["buses"][b].pop(0)
    assert rest == mine


def test_a_render_palette_changes_no_note_only_levels(dp):
    brief = load_briefs(BRIEFS)["d05"].brief
    res = parse(LOOP, vocab=dp.palette.vocab(), mode=brief.mode)
    assert res.gate.ok and not res.repairs
    a = compile_loop(res.ast, brief, dp.palette)
    b = compile_loop(res.ast, brief, parse_palette(yaml.safe_load(P.render_text(dp, "wurli", "jazz")), "r"))
    assert (a.keys, a.bass, a.sax, a.drums, a.n, a.seed) == (b.keys, b.bass, b.sax, b.drums, b.n, b.seed)


def test_tampering_is_found(dp):
    def broken(edit) -> list[str]:
        text = edit(dp.text)
        assert text != dp.text
        return P.problems(P.parse_draft(text, dp.path))

    assert any("wurli" in x for x in broken(lambda t: t.replace("fader_db: -6.435", "fader_db: -6.43", 1)))
    assert any("kits.studio" in x for x in broken(lambda t: t.replace("db: -14.94", "db: -14.0", 1)))
    assert any("kits.jazz.bass" in x for x in broken(lambda t: t.replace("src: fp_finger_bass_yr\n      bank: 0\n"
                                                                          "      prog: 0", "src: fp_finger_bass_yr\n"
                                                                          "      bank: 0\n      prog: 1", 1)))
    assert any("timing" in x for x in broken(lambda t: t.replace("timing: lag", "timing: grid", 1)))
    assert any("leaders" in x for x in broken(lambda t: t.replace("bus_gain_db: {sax: 2.55, drums: 15.26",
                                                                  "bus_gain_db: {sax: 2.55, drums: 15.2", 1)))
    with pytest.raises(P.DraftPaletteError, match="levels block"):
        P.parse_draft("\n".join(x for x in dp.text.splitlines() if not x.startswith("#|")) + "\n", dp.path)
    with pytest.raises(P.DraftPaletteError, match="levels.keys"):
        P.parse_draft(dp.text.replace("#|     piano:", "#|     upright:", 1), dp.path)
    with pytest.raises(P.DraftPaletteError, match="fader_db"):
        P.parse_draft(dp.text.replace("#|       fader_db: -6.06", "#|       fader_db: loud", 1), dp.path)


def test_vocabulary(dp):
    v = P.vocabulary(dp)
    assert list(v["keys"]) == ["rhodes", "wurli", "piano"] and list(v["kits"]) == ["studio", "jazz"]
    assert v["voicings"] == ["rootless", "shell", "spread", "quartal"]
    assert v["drum_lanes"] == ["kick", "snare", "rim", "hat", "ohat", "shaker"]
    assert len(v["numerals"]) == 11 and len(v["qualities"]) == 14
    assert "Im9" in v["chords_in_mode"]["wistful"] and "bVIImaj7" in v["chords_in_mode"]["wistful"]
    assert "IIm9" in v["chords_in_mode"]["hopeful"] and "V13" in v["chords_in_mode"]["hopeful"]
    assert "IVm9" in v["chords_in_mode"]["melancholy"] and "IVm9" not in v["chords_in_mode"]["wistful"]
    assert [t[0]["vamp"] for t in v["vamp_ranks"]["hopeful"]] == ["aruarian", "royal_half", "lift"]


def test_the_vamp_ranks_quote_the_plan():
    plan = yaml.safe_load((ROOT / "shootouts" / "plan.yaml").read_text(encoding="utf-8"))["vamps"]
    for mood, ranks in P.VAMP_RANKS.items():
        names = [v for tie in ranks for v, _, _ in tie]
        assert sorted(names) == sorted(k for k, v in plan.items() if v["mood"] == mood)
        for tie in ranks:
            for name, chords, span in tie:
                assert (plan[name]["chords"], plan[name]["span"], plan[name]["mood"]) == (chords, span, mood)


# ------------------------------------------------------------------------------------------- build round trip
def _inputs(dp, tmp: Path) -> tuple[Path, Path, dict]:
    """Levels files and a Virtuosity importer output rebuilt from the palette's own record (a stand-in tar, since
    the real one is fetch-only), so `build` can run without any fetched asset."""
    lv = dp.levels
    base = (ROOT / "shootouts" / "palettes" / "base.yaml").read_bytes()
    fake_tar = b"a stand-in for the Virtuosity tar, never a real archive\n"
    assets = tmp / "assets"
    (assets / "derived" / "kits").mkdir(parents=True)
    (assets / "derived" / "kits" / "kit_virtuosity.tar").write_bytes(fake_tar)
    studio = dp.doc["kits"]["studio"]
    printed = {"choke": studio["choke"], "lanes": P.untrimmed_lanes(studio["lanes"], dp.kits["studio"].trim_db)}
    (assets / "derived" / "kits" / "kit_virtuosity.yaml").write_text(yaml.safe_dump(printed, sort_keys=False),
                                                                      encoding="utf-8")
    fake_sha = P.sha256_bytes(fake_tar)
    recs = list(lv["keys"].values()) + list(lv["sax"].values())
    for kit in lv["kits"].values():
        recs += [kit, kit["bass"]]
    day1 = {"stage": "day1", "S_star": lv["S_star"], "targets": lv["targets"], "variants": {}}
    day2 = {"stage": "day2", "S_star": lv["S_star"], "targets": lv["targets"], "base_sha256": P.sha256_bytes(base),
            "variants": {}, "configs": {}}
    for r in recs:
        m = r["measured"]
        what = dict(m["as"])
        if what.get("source_sha256") == TAR_SHA:
            what["source_sha256"] = fake_sha
        day1["variants"][r["option"]] = {**m["day1"], **what}
        day2["variants"][r["option"]] = {**m["day2"], **what}
    for r in lv["keys"].values():
        day2["configs"][f"k2_{r['option']}"] = {"buses": {"keys": {"instrument": r["option"],
                                                                   "loops": {"F": r["spread_F"]}}}}
    p1, p2 = tmp / "levels1.json", tmp / "levels2.json"
    p1.write_text(json.dumps(day1), encoding="utf-8")
    day2["prior"] = {"sha256": P.file_sha256(p1)}
    p2.write_text(json.dumps(day2), encoding="utf-8")
    subs = {fake_sha: TAR_SHA, P.file_sha256(p1): lv["from"]["day1"]["sha256"],
            P.file_sha256(p2): lv["from"]["day2"]["sha256"]}
    return p1, p2, {"assets": assets, "subs": subs}


def test_a_build_from_the_same_inputs_is_byte_identical(dp, tmp_path, monkeypatch):
    p1, p2, extra = _inputs(dp, tmp_path)
    monkeypatch.setenv("LOFI_ASSETS", str(extra["assets"]))
    text = P.build(p1, p2)
    for fake, real in extra["subs"].items():
        text = text.replace(fake, real)
    assert text == dp.text


def test_a_build_refuses_inputs_that_disagree_with_the_measurements(dp, tmp_path, monkeypatch):
    p1, p2, extra = _inputs(dp, tmp_path)
    monkeypatch.setenv("LOFI_ASSETS", str(extra["assets"]))
    doc = json.loads(p2.read_text(encoding="utf-8"))
    doc["variants"]["gs_wurli_ep200"]["prog"] = 3
    p2.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(P.DraftPaletteError, match="gs_wurli_ep200"):
        P.build(p1, p2)
    doc["variants"]["gs_wurli_ep200"]["prog"] = 0
    doc["prior"]["sha256"] = "0" * 64
    p2.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(P.DraftPaletteError, match="prior"):
        P.build(p1, p2)
    p1b, p2b, extra = _inputs(dp, tmp_path / "again")
    (extra["assets"] / "derived" / "kits" / "kit_virtuosity.tar").write_bytes(b"another archive\n")
    monkeypatch.setenv("LOFI_ASSETS", str(extra["assets"]))
    with pytest.raises(P.DraftPaletteError, match="archive job 2 measured"):
        P.build(p1b, p2b)


def test_write_render_palettes_indexes_each_band(dp, tmp_path):
    index = P.write_render_palettes(dp, [("piano", "jazz"), ("rhodes", "studio"), ("piano", "jazz")], tmp_path)
    assert [e["id"] for e in index["palettes"]] == ["draft_day2_piano_jazz", "draft_day2_rhodes_studio"]
    e = index["palettes"][0]
    assert e["options"] == {"sax": "sax_mtg_alto", "drums": "kit_gu_jazz", "bass": "bass_fp_finger_yr",
                            "keys": "vcsl_upright_knight"}
    assert e["trim_db"] == {"sax": 6.63, "drums": 0.0, "bass": -2.79, "keys": 21.86}
    assert e["fader_db"] == {"sax": 2.55, "drums": 15.21, "bass": 5.13, "keys": -6.06}
    assert P.sha256_bytes((tmp_path / e["file"]).read_bytes()) == e["sha256"]
    assert json.loads((tmp_path / "index.json").read_text())["palette"]["sha256"] == dp.sha256
