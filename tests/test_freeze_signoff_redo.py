"""The sign-off job (lofi_rl.freeze.signoff; `lofi-drafts --signoff`; docs/DRAFTS.md §7.1) with a redo round.

The stand-in world of tests/test_freeze_signoff.py (its GeneralUser draft palette, frozen palette, rain and renderer),
on the synthetic redo round of tests/helpers_redo.py: drafts/v1's 30 drafts as committed and a redo1 folder of 12
sharing their stems; KEPT is keep3's 10 at the tree's paths, the sign-off listen b1103-signoff tossed four of them,
and the one REDOS round replaces those four with redo1 files, d06_horn among them (the stem of a tossed keep3 pick).
keep3 (job 7: its 10 kept drafts) and redo1 (job 8: all 12) are this world's renderer's renders with the draft
stand-in's render palettes; redo1's summary and renders sit under the job's --runs-root (runs/drafts-8), keep3's do
not. The job runs once, with REDOS, SIGNOFFS, KEPT and JOBS on lofi_rl.freeze for its module. Nothing is listened to
and nothing is claimed about sound."""
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import helpers_redo as H
import test_freeze_signoff as TS
from lofi_rl import freeze as FZ
from lofi_rl.brief import Brief
from lofi_rl.drafts import briefs as DB
from lofi_rl.drafts import check as DC
from lofi_rl.drafts import job as J
from lofi_rl.drafts import palette as P
from lofi_rl.drafts.render import render_all
from lofi_rl.freeze import batch as SB
from lofi_rl.freeze import checks as C
from lofi_rl.freeze import signoff as S
from lofi_rl.palette import load_palette
from lofi_rl.render.pipeline import manifest_json
from lofi_rl.ui.manifest import TEXT_MAX, load_batch
from test_freeze_signoff import (PICKS, RAIN_START_S, VARIANTS, argv, draft_standin, failing,  # noqa: F401
                                 fake_renderer, fast, frozen_doc_from_draft, renders_source, state_of, summary,
                                 write_rain)

ROOT = Path(__file__).resolve().parents[1]
REDO_CELLS = {("low", "melancholy"), ("mid", "hopeful")}      # two of prompt v2's references are enough here
# The stand-in rain must hold 0.5 s + the slowest loop + 1 s: the real briefs reach down to 66 BPM (d10), whose
# 8 bars at 44.1 kHz are loop_samples(66) = 1,282,880 frames (test_freeze_signoff's 1,200,000 fits its 81 and 82 BPM).
RAIN_FRAMES = 1_500_000


def draft_tasks(folder: Path, files: list[Path], dp: P.DraftPalette, pal_file: dict) -> list[dict]:
    """lofi-drafts' render tasks for `files` of `folder` (lofi_rl.drafts.job: the render id is the stem, the draft
    palette's render palette of its keys and kit, S* the draft palette's)."""
    briefs = DB.load(DB.briefs_file_for(folder))
    drafts = {d.path.name: d for d in DC.read_drafts(folder, briefs)}
    out = []
    for f in files:
        d = drafts[f.name]
        DC.check_one(d, dp.palette)
        b = briefs[d.brief_id].brief
        out.append({"render": d.path.stem, "file": J._label(d.path), "text": d.text, "brief_id": d.brief_id,
                    "brief": {"energy": b.energy, "mood": b.mood, "bpm": b.bpm, "tonic": b.tonic},
                    "persona": d.persona, "keys": d.loop.keys.instrument, "kit": d.loop.kit,
                    "palette": str(pal_file[(d.loop.keys.instrument, d.loop.kit)]), "s_star": dp.levels["S_star"]})
    return out


def build_redo_world(tmp: Path, mp: pytest.MonkeyPatch) -> SimpleNamespace:
    renderer = fake_renderer()
    assets = tmp / "assets"
    mp.setattr(TS, "RAIN_FRAMES", RAIN_FRAMES)
    rain_file, rain = write_rain(assets)
    (tmp / "rain_excerpts.json").write_text(json.dumps(rain), encoding="utf-8")
    dp = draft_standin(tmp / "draft_gu.yaml")
    source = {"path": rain["source"]["asset_path"], "sha256": rain["source"]["sha256"], "license": "CC0 1.0",
              "mode": "fetch", "url": None, "converted_from": None, "attribution": None}
    frozen = tmp / "lofi_rain_v1.yaml"
    frozen.write_text("# tests only: a stand-in of the frozen palette\n" + P.yaml_text(frozen_doc_from_draft(
        dp, bed={"src": "rain", "start_s": RAIN_START_S, "xfade_s": 1.0, "delta_lu": 24.0}, rain=source)),
        encoding="utf-8")
    v1, redo1 = H.write_tree(tmp / "drafts")
    consts = H.constants(v1, redo1, keep3_job=7, redo_job=8)
    H.patch(mp, consts)
    H.lift_near_copy_limit(mp)
    gate = tmp / "GATE_palette.md"
    gate.write_text(H.gate_text(consts), encoding="utf-8")
    mp.setenv("LOFI_ASSETS", str(assets))
    for name, value in (("GATE", gate), ("RAIN", tmp / "rain_excerpts.json"), ("DRAFT_PALETTE", dp.path)):
        mp.setattr(S, name, value)
    calls: list[list[str]] = []
    mp.setattr(S, "freeze_check", lambda args, root=ROOT: (calls.append([str(a) for a in args]) or 0, "ok: stand-in"))
    mp.setattr(S, "plan_variants", lambda path: copy.deepcopy(VARIANTS))
    mp.setattr(S, "BED", dict(PICKS))
    mp.setattr(S, "JOBS", {"day1": 1, "day2": 2, "day3": 3, "keep3": 7})
    real_refs = S.reference_loops
    mp.setattr(S, "reference_loops", lambda pal: {k: v for k, v in real_refs(pal).items() if v[0] in REDO_CELLS})

    def fake_repeat(task, out_wav, palette_path, cache):
        r = renderer(task["text"], Brief(**task["brief"]), load_palette(palette_path, verify=False))
        text = manifest_json(r.manifest)
        Path(out_wav).with_suffix(".manifest.json").write_text(text, encoding="utf-8")
        return {"exit": 0, "process_s": 0.01, "manifest_sha256": C.sha256_bytes(text.encode()),
                "audio_sha256": r.manifest["audio_sha256"], "music_sha256": r.manifest["music_sha256"],
                "n": r.manifest["n"]}
    mp.setattr(S, "run_repeat", fake_repeat)
    kept_v1 = [Path(f) for f in consts["KEPT"].values()]
    every = [*kept_v1, *sorted(redo1.glob("*.ls"))]
    combos = sorted({(re.search(r"^keys: (\w+) ", f.read_text(), re.M).group(1),
                      re.search(r"^drums: (\w+)", f.read_text(), re.M).group(1)) for f in every})
    index = P.write_render_palettes(dp, combos, tmp / "palettes")
    pal_file = {(e["keys"], e["kit"]): tmp / "palettes" / e["file"] for e in index["palettes"]}
    render_all(draft_tasks(v1, kept_v1, dp, pal_file), tmp / "k3" / "renders", jobs=4, workers=1,
               summary_dir=tmp / "k3" / "summary", renderer=renderer)
    redo_run = tmp / "runs" / "drafts-8"
    render_all(draft_tasks(redo1, sorted(redo1.glob("*.ls")), dp, pal_file), redo_run / "renders", jobs=4, workers=1,
               summary_dir=redo_run / "summary", renderer=renderer)
    k3, r1 = tmp / "k3" / "summary" / "renders.json", redo_run / "summary" / "renders.json"
    expect = C.expectations([renders_source("keep3", 7, k3), renders_source("keep3_redo1", 8, r1)],
                            kept=FZ.kept_rounds(), tossed=FZ.tossed_kept(), draft=dp)
    (v1 / C.EXPECT_NAME).write_text(C.expect_text(expect), encoding="utf-8")
    return SimpleNamespace(tmp=tmp, v1=v1, redo1=redo1, frozen=frozen, dp=dp, gate=gate, k3=k3, r1=r1, expect=expect,
                           calls=calls, renderer=renderer, consts=consts, rain_file=rain_file)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        w = build_redo_world(tmp_path_factory.mktemp("signoff_redo"), mp)
        w.rc = J.main(argv(w), renderer=w.renderer)
        yield w
    finally:
        mp.undo()


def kept_ids() -> dict[str, str]:
    """The kept loops' render ids here: a drafts/v1 pick's is its stem, a redo1 pick's carries its folder."""
    return {**{s: s for s in ("d02_chords", "d03_chords", "d04_horn", "d05_groove", "d08_horn", "d09_chords")},
            **{f"redo1_{b}_{H.PICKS[b]}": f"{b}_{H.PICKS[b]}" for b in H.REOPENED}}


def test_the_job_signs_off_the_new_kept_ten(world):
    assert world.rc == 0 and [s["exit"] for s in summary(world, "stages.json")["stages"]] == [0, 0, 0, 0, 0]
    assert [s["key"] for s in world.expect["sources"]] == ["keep3", "keep3_redo1"] and world.expect["tossed"] == []
    renders = summary(world, "renders.json")
    assert renders["kept_loops"] == kept_ids() and sorted(renders["kept"]) == sorted(kept_ids())
    k3 = json.loads(world.k3.read_text())["renders"]
    r1 = json.loads(world.r1.read_text())["renders"]
    for rid, stem in kept_ids().items():
        row, e = renders["kept"][rid], world.expect["loops"][stem]
        own = (r1 if rid.startswith("redo1_") else k3)[stem]
        assert e["pcm_sha256"] == own["pcm_sha256"] and row["twin0"]["pcm_sha256"] == own["pcm_sha256"]
        assert row["frozen"]["file"] == own["file"] and row["frozen"]["render"] == rid
    # the d06 pick shares its stem with the keep3 pick he tossed: its twin is redo1's render, never keep3's d06_horn
    assert renders["kept"]["redo1_d06_horn"]["twin0"]["pcm_sha256"] == r1["d06_horn"]["pcm_sha256"] != \
        k3["d06_horn"]["pcm_sha256"]
    assert renders["kept"]["redo1_d06_horn"]["frozen"]["file"] == str(world.redo1 / "d06_horn.ls")
    verify = summary(world, "verify.json")
    assert verify["ok"] and verify["checks"]["twin_is_kept"]["sources"] == {
        rid: "keep3_redo1" if rid.startswith("redo1_") else "keep3" for rid in kept_ids()}


def test_render_b_holds_every_other_draft_once_with_unique_ids(world):
    others = summary(world, "renders.json")["others"]
    drafts = {rid: r for rid, r in others.items() if r["kind"] == "draft"}
    assert len(drafts) == 30 + 12 - 10 == 32 and all(r["status"] == "ok" for r in others.values())
    assert len({r["file"] for r in drafts.values()}) == 32                       # 32 files under 32 ids
    assert drafts["d06_horn"]["file"] == str(world.v1 / "d06_horn.ls")          # the tossed one, now not kept
    assert drafts["redo1_d06_groove"]["file"] == str(world.redo1 / "d06_groove.ls")
    assert not set(drafts) & set(kept_ids()) and {r["kind"] for r in others.values()} == {"draft", "reference"}
    check = summary(world, "check.json")
    assert check["problems"] == []
    assert [(r["folder"], r["round"], r["drafts"], r["ok"]) for r in check["a"]["folders"]] == [
        (str(world.v1), "keep3", 30, 30), (str(world.redo1), "keep3_redo1", 12, 12)]
    assert (check["c"]["drafts"], check["c"]["fullmatch"], check["c"]["kept_fullmatch"]) == (42, 42, 10)
    assert check["sources_scratch"]["absent"] == ["keep3"]                      # keep3's run is not on this scratch
    assert check["sources_scratch"]["sources"]["keep3_redo1"]["equal"] is True   # redo1's is, and gives its records


def test_a_twin_mismatch_reads_its_own_sources_manifest(world, fast):
    st = state_of(world)
    st["records"]["kept"]["redo1_d01_horn"]["twin0"]["pcm_sha256"] = "0" * 64
    bad = failing(st)
    assert list(bad) == ["twin_is_kept"] and "is not keep3_redo1's" in bad["twin_is_kept"][0]
    # redo1's job left its render under the loop's stem in runs/drafts-8, on this scratch: no bus input differs
    assert S.verify_doc(st)["checks"]["twin_is_kept"]["stem_diffs"] == {
        "redo1_d01_horn": ["no bus input differs: the difference comes after the bus inputs"]}
    st = state_of(world)
    st["records"]["kept"]["d02_chords"]["twin0"]["pcm_sha256"] = "0" * 64
    assert S.verify_doc(st)["checks"]["twin_is_kept"]["stem_diffs"] == {"d02_chords": [
        f"keep3's manifest {world.tmp / 'runs' / 'drafts-7' / 'renders' / 'd02_chords' / 'manifest.json'} is not on "
        f"scratch"]}


def test_the_batch_says_again(world):
    m, h = load_batch(world.tmp / "inbox" / "2026-11-05" / "b1105-signoff")
    lo, hi = summary(world, "verify.json")["checks"]["bed_heard"]["kept_below"]
    assert m.note == SB.note_text(10, summary(world, "palettes/index.json")["bed"]["want"], (lo, hi), again=True)
    assert m.note.startswith("Sign-off again: your 10 kept loops") and "new to your ears" not in m.note
    assert len(m.note) <= TEXT_MAX and len(m.rounds) == 10
    infos = [next(iter(hr.clips.values())) for hr in h.rounds.values()]
    assert {i["render"] for i in infos} == set(kept_ids())
    assert {i["draft"] for i in infos} == set(FZ.kept_now().values())
    assert summary(world, "batch.json")["push"].startswith("Needs ear check: b1105-signoff. Your 10 kept loops again")


def test_the_job_refuses_while_a_tossed_file_is_still_kept(world, monkeypatch, tmp_path):
    """The same tree with the redo round left out of REDOS: the four he tossed are kept again, so the check names each
    with its brief and batch, and nothing is rendered."""
    monkeypatch.setattr(FZ, "REDOS", ())
    monkeypatch.setattr(FZ, "JOBS", {k: v for k, v in FZ.JOBS.items() if k != "keep3_redo1"})
    run = argv(world, run="drafts-tossed", inbox="inbox-tossed")
    assert J.main(run, renderer=world.renderer) == 3
    check = summary(world, "check.json", "drafts-tossed")
    assert check["tossed"] == C.tossed_problems() and len(check["tossed"]) == 4
    assert [t.split(":")[0] for t in check["tossed"]] == list(H.REOPENED)
    assert all("b1103-signoff" in t for t in check["tossed"])
    assert [s["stage"] for s in summary(world, "stages.json", "drafts-tossed")["stages"]] == ["check"]
    assert not (world.tmp / "runs" / "drafts-tossed" / "renders").exists()
