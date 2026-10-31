"""The batch builder (docs/SHOOTOUTS.md section 7.6, acceptance tests 1, 3, 5, 6, 7 and 8), on fake renders.

Renders go in, and out come `inbox/<date>/<batch>/` folders holding only opaque FLACs and the public and hidden
manifests in docs/RATING.md Part 2's format, plus `summary/batches.json`, the id map the day-3 repeats read. The
renders here are fake (tests/helpers_shootout.py: a FLAC header, a record and a manifest per render), so a
whole day builds in well under a second; tests/test_shootout_e2e.py builds from real renders."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from helpers_shootout import fake_renders
from lofi_rl.shootout.batch import BuildError, build, hidden_names, main, public_leaks
from lofi_rl.shootout.draw import draw, load_prior, parse_leaders, write_draw
from lofi_rl.shootout.plan import load_plan
from lofi_rl.ui.manifest import HIDDEN_NAME, PUBLIC_NAME, _words, load_batch

ROOT = Path(__file__).resolve().parents[1]
PLAN = load_plan()
DAY2 = parse_leaders("keys=gu_tine_ep+fp_fm_ep1+gu_organ/kit=kit_swirly/bass=bass_fp_finger_yr/"
                     "sax=sax_mtg_tenor+sax_gu_alto")
DAY3 = parse_leaders("keys=fp_fm_ep1/kit=kit_swirly/bass=bass_fp_finger_yr/sax=sax_mtg_tenor")
ALL_STAGES = ["K1", "R1", "R2", "S1", "R3a", "R3b", "S1f", "S2a", "S2b", "K2", "V"]


def run_day(tmp: Path, name: str, day: str, seed: int, *, inbox: Path | None = None, overrides=None,
            purge=(), check=True, **kw):
    run = tmp / "runs" / name
    doc = draw(PLAN, day, seed=seed, **kw)
    write_draw(doc, run / "summary" / "draw.json")
    for key in doc["palettes"]:                       # the generator's output (section 7.8), a stand-in here
        pal = run / "summary" / "palettes" / f"{key}.yaml"
        pal.parent.mkdir(parents=True, exist_ok=True)
        pal.write_text(f"# stand-in for palette {key}\n")
    fake_renders(doc, run / "renders", overrides)
    for rid in purge:
        (run / "renders" / rid / "render.flac").unlink()
    out = build(doc, run / "renders", inbox or tmp / "inbox", summary_dir=run / "summary", plan=PLAN, check=check)
    return run, doc, out


@pytest.fixture(scope="module")
def days(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("shootout")
    r1, d1, o1 = run_day(tmp, "shootout-101", "day1", 11)
    r2, d2, o2 = run_day(tmp, "shootout-102", "day2", 12, leaders=DAY2)
    priors = [load_prior(r / "summary" / "batches.json") for r in (r1, r2)]
    r3, d3, o3 = run_day(tmp, "shootout-103", "day3", 13, leaders=DAY3, priors=priors, repeats=ALL_STAGES)
    return {"tmp": tmp, "day1": (r1, d1, o1), "day2": (r2, d2, o2), "day3": (r3, d3, o3)}


def _folders(days):
    return [p for day in ("day1", "day2", "day3") for p in days[day][2]]


def test_each_batch_folder_holds_only_its_flacs_and_both_manifests_and_passes_the_check(days):
    folders = _folders(days)
    planned = [(name, b.date) for b in PLAN.batches for name in b.names]
    assert [p.name for p in folders] == [name for name, _ in planned]
    assert all(p.parent.name == date and p.parent.parent == days["tmp"] / "inbox"
               for p, (_, date) in zip(folders, planned))
    for d in folders:
        m, h = load_batch(d)
        files = {p.name for p in d.iterdir()}
        assert files == {PUBLIC_NAME, HIDDEN_NAME} | {c.file for r in m.rounds for c in r.clips}, d.name
        assert all(re.fullmatch(r"k[0-9a-f]{10}\.flac", f) for f in files - {PUBLIC_NAME, HIDDEN_NAME})
        assert all(re.fullmatch(r"r[0-9a-f]{8}", r.id) for r in m.rounds)
        assert m.mode == "shootout" and m.bars == 8 and m.sr == 44100 and m.scene == "city at night"
    # the builder ran the CLI check on every folder before moving it into place; once more from here
    p = subprocess.run([sys.executable, "-m", "lofi_rl.ui.manifest", "check", str(folders[2])], capture_output=True,
                       text=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert p.returncode == 0 and p.stdout.startswith("ok: b1103-3-sax (shootout), 7 rounds, 28 clips"), p.stderr
    assert not list((days["tmp"] / "inbox").glob(".staging*"))


def test_the_batches_hold_their_planned_rounds_in_the_drawn_order(days):
    kinds = {}
    for d in _folders(days):
        m, h = load_batch(d)
        kinds[d.name] = sorted(h.rounds[r.id].kind for r in m.rounds)
        assert h.rounds[m.rounds[0].id].kind == "none"
        loops = [h.rounds[r.id].brief_id for r in m.rounds]
        assert all(a != b for a, b in zip(loops, loops[1:]))
    assert kinds["b1103-1-keys"] == ["none"] * 6 and kinds["b1103-2-groove"] == ["none"] * 6
    assert kinds["b1103-3-sax"] == ["anchor"] + ["none"] * 6
    assert kinds["b1104-2-sax"] == ["anchor"] + ["none"] * 9 and kinds["b1104-4-vamps"] == ["none"] * 9
    mix = kinds["b1106-1-mix"] + kinds["b1106-2-mix"]
    assert sorted(mix) == ["anchor"] + ["none"] * 7 + ["repeat"] * 5
    for d in _folders(days):
        doc = json.loads((d / PUBLIC_NAME).read_text())
        drawn = next(b for day in ("day1", "day2", "day3") for b in days[day][1]["batches"] if b["name"] == d.name)
        assert doc["est_min"] == drawn["est_min"] and doc["note"] == drawn["note"]
        assert [r["bpm"] for r in doc["rounds"]] == [r["bpm"] for r in drawn["rounds"]]


def test_hidden_manifests_say_what_each_clip_really_is(days):
    run, doc, folders = days["day1"]
    batches = json.loads((run / "summary" / "batches.json").read_text())
    for d in folders:
        m, h = load_batch(d)
        for r in m.rounds:
            hr = h.rounds[r.id]
            for c in r.clips:
                info = hr.clips[c.id]
                rec = json.loads((run / "renders" / info["render"] / "record.json").read_text())
                assert info["clip"] == rec["clip"] and info["pcm_sha256"] == rec["pcm_sha256"]
                assert info["option"] == rec["option"] and info["loop"] == hr.brief_id == rec["loop"]
                assert info["palette"] == rec["palette"]["label"] and info["sax_range"] == "51-72"
                assert {"what", "stage", "slot", "trim_db", "fader_db", "bus_lufs", "render"} <= set(info)
                assert c.sha256 == rec["flac_sha256"]
            entry = next(x for x in batches["batches"][d.name]["rounds"] if x["id"] == r.id)
            assert [x["id"] for x in entry["clips"]] == list(r.ids)
            assert [x["render"] for x in entry["clips"]] == [hr.clips[c].get("render") for c in r.ids]
    assert batches["run"] == "shootout-101" and batches["stage"] == "day1" and batches["seed"] == doc["seed"]


def test_one_seed_builds_byte_identical_batches(tmp_path):
    outs = []
    for name in ("a", "b"):
        run, doc, folders = run_day(tmp_path / name, "shootout-1", "day1", 77, inbox=tmp_path / name / "inbox",
                                    check=False)
        files = sorted(p for p in (tmp_path / name).rglob("*") if p.is_file() and "renders" not in p.parts)
        outs.append({str(p.relative_to(tmp_path / name)): p.read_bytes() for p in files})
    assert outs[0] == outs[1] and len(outs[0]) > 60
    run_day(tmp_path / "c", "shootout-1", "day1", 78, inbox=tmp_path / "c" / "inbox", check=False)
    a = {k: v for k, v in outs[0].items() if k.endswith(PUBLIC_NAME)}
    c = {str(p.relative_to(tmp_path / "c")): p.read_bytes() for p in (tmp_path / "c" / "inbox").rglob(PUBLIC_NAME)}
    assert a.keys() == c.keys() and all(a[k] != c[k] for k in a)


def test_an_anchor_round_names_its_sour_clip_and_its_copy_passes_the_gates(days):
    run, doc, folders = days["day1"]
    m, h = load_batch(next(d for d in folders if d.name == "b1103-3-sax"))
    anchors = [h.rounds[r.id] for r in m.rounds if h.rounds[r.id].kind == "anchor"]
    assert len(anchors) == 1
    a = anchors[0]
    sour = a.clips[a.anchor_clip]
    assert sour["what"] == "sour copy" and sour["render"].endswith(".sour")
    assert doc["sour"][sour["render"]]["ok"] and doc["sour"][sour["render"]]["p_chrom"] >= 0.2
    others = [i for oid, i in a.clips.items() if oid != a.anchor_clip]
    design = [h.rounds[r.id] for r in m.rounds if h.rounds[r.id].kind == "none" and h.rounds[r.id].brief_id == a.brief_id]
    assert len(design) == 1
    assert {i["clip"] for i in others} < {i["clip"] for i in design[0].clips.values()}
    assert sour["option"] in {i["option"] for i in design[0].clips.values()}
    assert sour["clip"] not in {i["clip"] for i in design[0].clips.values()}


def _names(hid):
    """The batch's own hidden names, plus every candidate of the lists its stages draw from."""
    names = hidden_names(hid)
    for hr in hid["rounds"].values():
        for info in hr["clips"].values():
            for g in PLAN.stages[info["stage"]].groups:
                names |= set(PLAN.band_options[g.leaders] if g.leaders else g.options)
    return names


def test_no_hidden_value_appears_in_the_public_manifests(days):
    # section 7.6 test 7: every option, loop, palette and render name, with ui/manifest.py's category-word
    # exemption (b1103-3-sax and its note may say "sax"); and no stretch of a real clip id or pcm hash
    for d in _folders(days):
        pub = json.loads((d / PUBLIC_NAME).read_text())
        hid = json.loads((d / HIDDEN_NAME).read_text())
        names = _names(hid)
        assert len(names) > 10
        assert public_leaks(pub, hid, names) == [], d.name
        cats = _words(hid["category"]) | {w for r in hid["rounds"].values() for w in _words(r["category"])}
        text = " ".join([pub["batch"], pub["scene"], pub["note"], pub["mode"]] +
                        [r["id"] for r in pub["rounds"]] + [x for r in pub["rounds"] for c in r["clips"]
                                                            for x in (c["id"], c["file"])])
        words = _words(text)
        for n in names:
            assert not (_words(n) - cats) & words, (d.name, n)
        ids = [x for r in pub["rounds"] for x in [r["id"], *(c["id"] for c in r["clips"])]]
        for r in hid["rounds"].values():
            for info in r["clips"].values():
                for secret in (info["clip"][1:], info["pcm_sha256"]):
                    assert not any(secret[k:k + 10] in i for i in ids for k in range(len(secret) - 9))
    sax = json.loads((days["day1"][2][2] / PUBLIC_NAME).read_text())
    assert "sax" in sax["note"] and "sax" in sax["batch"]


def test_the_leak_guard_names_a_planted_option(days):
    d = days["day1"][2][0]
    pub = json.loads((d / PUBLIC_NAME).read_text())
    hid = json.loads((d / HIDDEN_NAME).read_text())
    assert public_leaks(pub, hid) == []
    pub["note"] = "Which keys do you like best? Maybe the organ."
    assert any("organ" in e for e in public_leaks(pub, hid))
    pub["note"] = "Which keys do you like best?"
    real = next(iter(next(iter(hid["rounds"].values()))["clips"].values()))["clip"]
    pub["rounds"][0]["clips"][0]["id"] = "k" + real[3:13]                 # 10 hex digits of a real clip id
    assert any("real" in e for e in public_leaks(pub, hid))


def test_repeats_show_a_prior_round_again_with_its_tiles_reversed(days):
    run, doc, folders = days["day3"]
    prior = {}
    for day in ("day1", "day2"):
        pb = json.loads((days[day][0] / "summary" / "batches.json").read_text())
        for name, b in pb["batches"].items():
            for r in b["rounds"]:
                prior[r["id"]] = (name, r)
    firsts = {}
    for rid, (name, r) in sorted(prior.items(), key=lambda x: (x[1][0], x[1][1]["index"])):
        if r["kind"] == "none":
            firsts.setdefault(r["stage"], rid)
    groups = []
    for d in folders:
        m, h = load_batch(d)
        for r in m.rounds:
            hr = h.rounds[r.id]
            if hr.kind != "repeat":
                continue
            assert hr.of in prior and hr.of != r.id                            # resolves to a --prior round
            name, src = prior[hr.of]
            assert src["kind"] == "none" and hr.of not in firsts.values()       # never a stage's first round
            assert [hr.clips[c]["clip"] for c in r.ids] == [c["clip"] for c in reversed(src["clips"])]
            assert [hr.clips[c]["option"] for c in r.ids] == [c["option"] for c in reversed(src["clips"])]
            assert hr.brief_id == src["loop"] and hr.category == src["category"]
            assert all(x not in {c["id"] for c in src["clips"]} for x in r.ids)    # new opaque ids
            groups.append(PLAN.stages[src["stage"]].repeat_group)
    assert sorted(groups) == sorted(PLAN.repeat_groups)                        # one per group


def test_no_repeat_comes_from_a_stage_that_repeats_does_not_name(days):
    priors = [load_prior(days[d][0] / "summary" / "batches.json") for d in ("day1", "day2")]
    for seed in range(20, 26):
        doc = draw(PLAN, "day3", seed=seed, leaders=DAY3, priors=priors, repeats=["K2", "R1", "S1"])
        reps = [r for b in doc["batches"] for r in b["rounds"] if r["kind"] == "repeat"]
        assert sorted(r["stage"] for r in reps) == ["K2", "R1", "S1"]
    assert not [r for b in draw(PLAN, "day3", seed=1, leaders=DAY3, priors=priors)["batches"]
                for r in b["rounds"] if r["kind"] == "repeat"]


def test_a_purged_prior_render_is_made_again_and_must_match_its_first_showing(days, tmp_path):
    r1 = days["day1"][0]
    priors = [load_prior(r1 / "summary" / "batches.json")]
    doc = draw(PLAN, "day3", seed=31, leaders=DAY3, priors=priors, repeats=["K1"])
    rep = next(r for b in doc["batches"] for r in b["rounds"] if r["kind"] == "repeat")
    rid = rep["clips"][0]["render"]
    assert "run_dir" in rep["clips"][0] and not rid.startswith("prior.")
    flac = r1 / "renders" / rid / "render.flac"
    keep = flac.read_bytes()
    flac.unlink()
    try:
        doc2 = draw(PLAN, "day3", seed=31, leaders=DAY3, priors=priors, repeats=["K1"])
    finally:
        flac.write_bytes(keep)
    rep2 = next(r for b in doc2["batches"] for r in b["rounds"] if r["kind"] == "repeat")
    new = rep2["clips"][0]["render"]
    assert new == f"prior.shootout-101.{rid}" and "run_dir" not in rep2["clips"][0]
    entry = doc2["renders"][new]
    assert entry["expect_pcm_sha256"] == rep2["clips"][0]["expect"]["pcm_sha256"]
    assert entry["palette_path"].endswith(f"shootout-101/summary/palettes/{entry['palette']}.yaml")
    assert entry["text"] == days["day1"][1]["renders"][rid]["text"]


@pytest.mark.parametrize("case", ["missing", "gate", "length", "flac_changed", "repeat_mismatch"])
def test_a_missing_render_a_failed_gate_or_a_clip_that_is_not_one_pass_stops_the_build(tmp_path, case):
    doc = draw(PLAN, "day1", seed=5)
    run = tmp_path / "runs" / "shootout-9"
    rid = sorted(doc["renders"])[3]
    over = {"missing": {rid: None}, "gate": {rid: {"status": "gate"}},
            "length": {rid: {"n": 1000}}, "flac_changed": {}, "repeat_mismatch": {}}[case]
    fake_renders(doc, run / "renders", over)
    if case == "flac_changed":
        (run / "renders" / rid / "render.flac").write_bytes(b"fLaC" + bytes(38))
    if case == "repeat_mismatch":
        clip = next(c for b in doc["batches"] for r in b["rounds"] for c in r["clips"] if c["render"] == rid)
        clip["expect"] = {"clip": "c" + "0" * 16, "pcm_sha256": "0" * 64, "flac_sha256": "0" * 64}
    with pytest.raises(BuildError) as err:
        build(doc, run / "renders", tmp_path / "inbox", summary_dir=run / "summary", plan=PLAN)
    assert rid in str(err.value)
    assert not (tmp_path / "inbox").exists() or not any((tmp_path / "inbox").iterdir())
    assert not (run / "summary" / "batches.json").exists()


def test_an_existing_batch_folder_is_never_overwritten(tmp_path):
    doc = draw(PLAN, "day1", seed=5)
    run = tmp_path / "runs" / "shootout-9"
    fake_renders(doc, run / "renders")
    (tmp_path / "inbox" / "2026-11-03" / "b1103-2-groove").mkdir(parents=True)
    with pytest.raises(BuildError, match="exists"):
        build(doc, run / "renders", tmp_path / "inbox", summary_dir=run / "summary", plan=PLAN)
    assert sorted(p.name for p in (tmp_path / "inbox" / "2026-11-03").iterdir()) == ["b1103-2-groove"]


def test_the_build_refuses_a_draw_of_another_plan(tmp_path):
    doc = draw(PLAN, "day1", seed=5)
    run = tmp_path / "runs" / "shootout-9"
    fake_renders(doc, run / "renders")
    doc["plan"]["sha256"] = "0" * 64
    with pytest.raises(BuildError, match="plan"):
        build(doc, run / "renders", tmp_path / "inbox", summary_dir=run / "summary", plan=PLAN)


def test_the_cli_draws_and_then_builds(tmp_path, capsys):
    run = tmp_path / "runs" / "shootout-5"
    assert main(["draw", "--plan", str(PLAN.path), "--stage", "day1", "--summary", str(run / "summary"),
                 "--seed", "00000000000000ff"]) == 0
    doc = json.loads((run / "summary" / "draw.json").read_text())
    assert doc["seed"] == "00000000000000ff" and len(doc["renders"]) == 67
    fake_renders(doc, run / "renders")
    assert main(["build", "--plan", str(PLAN.path), "--stage", "day1", "--renders", str(run / "renders"),
                 "--out", str(tmp_path / "inbox")]) == 0
    assert len(list((tmp_path / "inbox").glob("*/*/manifest.json"))) == 3
    out = capsys.readouterr().out
    assert "b1103-3-sax" in out and "draw.json" in out
    assert main(["draw", "--plan", str(PLAN.path), "--stage", "day1", "--summary", str(tmp_path / "s2")]) == 0
    assert re.fullmatch(r"[0-9a-f]{16}", json.loads((tmp_path / "s2" / "draw.json").read_text())["seed"])
    assert main(["draw", "--plan", str(PLAN.path), "--stage", "day2", "--summary", str(tmp_path / "s3")]) == 2
    assert "needs --leaders" in capsys.readouterr().err
    assert main(["build", "--plan", str(PLAN.path), "--stage", "day2", "--renders", str(run / "renders"),
                 "--out", str(tmp_path / "inbox2")]) == 2
