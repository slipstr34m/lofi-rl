"""The `lofi-drafts` job (lofi_rl.drafts.job and .render; jobs/drafts.sbatch; docs/DRAFTS.md §7) on fixture drafts.

Most tests run the job with a stand-in renderer (the real compiler, synthetic audio, stated loudness numbers), so they
cover the stages, the refusals and the batch without FluidSynth. One test renders three fixture drafts for real through
render.pipeline.render with a GeneralUser-only stand-in of the draft palette (committed SoundFont, no fetched asset) and
checks the batch the rating page would load. Nothing is listened to and nothing is claimed about sound."""
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from helpers_render import needs_fluidsynth
from lofi_rl.compile.compiler import compile_loop
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.drafts import job as J
from lofi_rl.drafts import palette as P
from lofi_rl.drafts.render import clip_id
from lofi_rl.dsl import canon, canon_sha, parse
from lofi_rl.render.pipeline import RenderError
from lofi_rl.render.wavio import pcm_sha256
from lofi_rl.ui.manifest import load_batch
from test_drafts_check import FIXTURES, write_folder

ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "jobs" / "drafts.sbatch"


def fake_renderer(lufs: float = -14.02, peak: float = -1.5, boom: bool = False):
    """render.pipeline.render's shape: the real parse and compile, then a synthetic cycle of n samples and a manifest
    with the stated loudness."""
    def render(text, brief, palette, *, cache=None, workers=1):
        if boom:
            raise RuntimeError("a stand-in infrastructure fault")
        res = parse(text, vocab=palette.vocab(), mode=brief.mode)
        if not res.gate.ok:
            raise RenderError(res.gate.code, res.gate.detail)
        compiled = compile_loop(res.ast, brief, palette)
        n = compiled.n
        seed = int(canon_sha(res.ast)[:8], 16)
        x = (0.05 * np.sin(np.arange(n) * (0.01 + (seed % 97) * 1e-4))).astype(np.float32)
        audio = np.stack([x, x])
        manifest = {"schema": "lofi-rl/render/1", "chain": {"cycles": 2, "fluid_preroll_cycles": 1},
                    "loop": {"canon": canon(res.ast), "canon_sha256": canon_sha(res.ast),
                             "compiled_sha256": compiled.digest(), "repairs": [], "p_fix": 0.0},
                    "brief": {"scene": brief.scene, "energy": brief.energy, "mood": brief.mood, "mode": brief.mode,
                              "bpm": brief.bpm, "tonic": brief.tonic, "key": brief.key_name},
                    "palette": {"id": palette.id, "sha256": palette.sha256, "frozen": palette.frozen},
                    "n": n, "sr": 44100,
                    "buses": {b: {"lufs": -33.0, "gain_db": palette.mix.bus_gain_db[b]} for b in
                              ("sax", "keys", "bass", "drums")},
                    "loudness": {"lufs": lufs, "true_peak_dbtp": peak, "gain_db": 4.0},
                    "gates": {"G1-G6": "pass", "G7": "pass"}, "audio_sha256": pcm_sha256(audio),
                    "music_sha256": pcm_sha256(audio)}
        return SimpleNamespace(audio=audio, music=audio, manifest=manifest, compiled=compiled)
    return render


def run(tmp_path, *extra, renderer=None, drafts=None, palette=None):
    """The job on `drafts` (the fixtures by default) with a stand-in renderer and, unless `palette` names another, the
    GeneralUser stand-in of the draft palette, whose one source is committed, so the palette stage verifies it."""
    drafts = drafts or write_folder(tmp_path / "v1")
    if palette is None:
        palette = tmp_path / "draft_gu.yaml"
        if not palette.exists():
            gu_palette(palette)
    argv = ["--drafts", str(drafts), "--run", str(tmp_path / "runs" / "drafts-1"), "--inbox", str(tmp_path / "inbox"),
            "--date", "2026-11-05", "--jobs", "4", "--seed", "00000000000000aa", "--cache", str(tmp_path / "cache"),
            "--palette", str(palette), *extra]
    return J.main(argv, renderer=renderer or fake_renderer())


def summary(tmp_path, name: str):
    return json.loads((tmp_path / "runs" / "drafts-1" / "summary" / name).read_text(encoding="utf-8"))


def test_the_job_renders_every_draft_and_builds_one_keep3_batch(tmp_path):
    assert run(tmp_path) == 0
    folder = tmp_path / "inbox" / "2026-11-05" / "b1105-keep3"
    m, h = load_batch(folder)
    assert m.mode == "keep3" and len(m.rounds) == 2 and all(len(r.clips) == 3 for r in m.rounds)
    renders = summary(tmp_path, "renders.json")["renders"]
    assert sorted(renders) == sorted(Path(f).stem for f in FIXTURES) and all(r["status"] == "ok" for r in renders.values())
    for rid, rec in renders.items():
        man = json.loads((tmp_path / "runs" / "drafts-1" / "renders" / rid / "manifest.json").read_text())
        assert rec["clip"] == clip_id(man) and rec["palette"]["id"] == f"draft_gu_test_{rec['keys']}_{rec['kit']}"
        assert (tmp_path / "runs" / "drafts-1" / "renders" / rid / "loop.ls").read_text() == FIXTURES[f"{rid}.ls"]
    combos = {(r["keys"], r["kit"]) for r in renders.values()}
    index = summary(tmp_path, "palettes/index.json")
    assert {(e["keys"], e["kit"]) for e in index["palettes"]} == combos
    stages = summary(tmp_path, "stages.json")
    assert [s["stage"] for s in stages["stages"]] == ["check", "palettes", "render", "build"]
    assert all(s["exit"] == 0 for s in stages["stages"])
    args = summary(tmp_path, "args.json")
    assert args["batch"] == "b1105-keep3" and args["scope"] == ["d01", "d05"] and summary(tmp_path, "check.json")["ok"]
    for hr in h.rounds.values():
        assert sorted(i["persona"] for i in hr.clips.values()) == ["chords", "groove", "horn"]
        assert all(i["brief"] == hr.brief_id and i["draft"].endswith(f"{hr.brief_id}_{i['persona']}.ls")
                   for i in hr.clips.values())


# Three fresh d05 drafts for the redraft test: they pass every check, also against FIXTURES.
REDO = {
    "d05_groove.ls": """\
vamp: Im9 IV9 span 1
keys: rhodes spread
comp: ..x-....x-....x- ..x-......x-x-..
bass: x--x....x--x.... x-----x.x---....
drums: jazz
kick: x..x....x.x..... x..x......x.....
snare: ....x..g....x... ....x..g....x..g
hat: xxx.xxx.xxx.xxx. xxx.xxx.xxx.x.x.
@2 kick: x..x....x.x...x.
@3 snare: ....x.......x.g.
@4 hat: xxx.xxx.xxx.xxxx
@5 bass: x--x....x--x..x.
@6 comp: ..x-....x-..x-x-
@7 kick: x..x....x.......
@8 hat: xxx.xxx.xxx.xx..
A: e1 e2 q3 h5 | qr e6 e5 h4 | q.3 e2 h1 | qr q2 hr
B: e1 e2 q3 h5 | qr e7 e1' h2' | q.7 e6 h5 | qr q4 hr
""",
    "d05_horn.ls": """\
vamp: Im9 bVIImaj7 span 2
keys: piano rootless
comp: x-----x---...... x-------x---x---
bass: x-------x-x..... x-----x-------..
drums: studio
kick: x.......x.x..... x.......x...x...
rim: ....x.......x... ....x.......x...
hat: x.x.x.x.x.x.x.x. x.x.x.x.x.x.xxx.
shaker: x.x.x.x.x.x.x.x. x.x.x.x.x.x.x.x.
@3 rim: ....x..g....x...
@4 kick: x.......x.x...x.
@6 hat: x.x.x.x.x.xxx.x.
@7 comp: x-------x-------
@8 kick: x.......x.......
A: h.5' e4' e3' | h2' qr q1' | w7 | e6 e5 q4 hr
B: h.5' e4' e3' | h2' qr q3' | h.4' q3' | e2' e1' q7 hr
""",
    "d05_chords.ls": """\
vamp: Im11 IV13 span 2
keys: wurli quartal
comp: x---x---x---x--- x-------x-x-x---
bass: x-----x---x-x--- x---x-----x-----
drums: jazz
kick: x.....x...x.x... x.....x.....x...
snare: ....X.......X... ....X.......X...
hat: x.g.x.g.x.g.x.g. x.g.x.g.x.g.xxg.
ohat: ................ ..............x.
@2 comp: x---x---x-x-x---
@3 kick: x.....x.x...x...
@5 snare: ....X..g....X...
@6 comp: x-------x---x-x-
@7 hat: x.g.x.g.xxg.x.g.
@8 kick: x.....x.....x...
A: q3 e4 e5 h6 | q5 e4 e3 h2 | q.1 e2 h3 | h5 hr
B: q3 e4 e5 h6 | q7 e1' e2' h3' | q.2' e1' h7 | h6 hr
""",
}


def test_with_the_draft_palette_each_draft_plays_its_bands_trims_and_faders(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "verify_files", lambda files: [])   # the fetch-only sources are not in the suite's tree
    assert run(tmp_path, palette=P.DRAFT_PALETTE) == 0
    dp = P.load()
    for rec in summary(tmp_path, "renders.json")["renders"].values():
        band = dp.band(rec["keys"], rec["kit"])
        assert rec["palette"]["id"] == f"draft_day2_{rec['keys']}_{rec['kit']}"
        assert {b: (v["trim_db"], v["fader_db"]) for b, v in rec["buses"].items()} == {
            b: (pytest.approx(band[b].trim_db), band[b].fader_db) for b in band}
    files = {e["file"] for e in summary(tmp_path, "palettes/index.json")["palettes"]}
    assert files == {f"draft_day2_{k}_{d}.yaml" for k, d in (("rhodes", "studio"), ("wurli", "jazz"),
                                                              ("piano", "studio"), ("piano", "jazz"))}


def test_a_redraft_renders_only_its_briefs_and_takes_the_next_name(tmp_path):
    import shutil
    v1 = write_folder(tmp_path / "v1")
    assert run(tmp_path) == 0
    redo = tmp_path / "v1" / "redo1"
    redo.mkdir()
    for name, text in {**REDO, "d05_chords.ls": FIXTURES["d05_chords.ls"]}.items():
        (redo / name).write_text(text, encoding="utf-8")
    shutil.rmtree(tmp_path / "runs")
    assert run(tmp_path, "--briefs", "d05", "--against", str(v1), drafts=redo) == 3
    rows = {d["file"]: d for d in summary(tmp_path, "check.json")["drafts"]}
    assert rows["d05_chords.ls"]["drafts_nearest"] == "v1/d05_chords.ls" and "drafts" in rows["d05_chords.ls"]["problems"]
    assert rows["d05_horn.ls"]["problems"] == {} and rows["d05_groove.ls"]["problems"] == {}
    assert not (tmp_path / "inbox" / "2026-11-05" / "b1105-keep3-redo").exists()
    (redo / "d05_chords.ls").write_text(REDO["d05_chords.ls"], encoding="utf-8")
    shutil.rmtree(tmp_path / "runs")
    assert run(tmp_path, "--briefs", "d05", "--against", str(v1), drafts=redo) == 0
    check = summary(tmp_path, "check.json")
    assert check["scope"] == ["d05"] and len(check["against"]) == 6 and check["ok"]
    m, _ = load_batch(tmp_path / "inbox" / "2026-11-05" / "b1105-keep3-redo")
    assert len(m.rounds) == 1 and summary(tmp_path, "args.json")["batch"] == "b1105-keep3-redo"
    assert sorted(summary(tmp_path, "renders.json")["renders"]) == ["d05_chords", "d05_groove", "d05_horn"]
    assert J.batch_name(J._date(2026, 11, 5), tmp_path / "inbox", True, None) == "b1105-keep3-redo2"


def test_a_failed_check_renders_nothing(tmp_path):
    drafts = write_folder(tmp_path / "v1", {**FIXTURES, "d05_horn.ls": FIXTURES["d05_horn.ls"].replace(
        "drums: jazz", "drums: trap")})
    assert run(tmp_path, drafts=drafts) == 3
    assert not (tmp_path / "runs" / "drafts-1" / "renders").exists()
    assert not (tmp_path / "inbox").exists()
    assert [s["stage"] for s in summary(tmp_path, "stages.json")["stages"]] == ["check"]


def test_a_render_off_the_loudness_spec_stops_before_the_batch(tmp_path):
    assert run(tmp_path, renderer=fake_renderer(lufs=-13.85)) == 3
    recs = summary(tmp_path, "renders.json")["renders"]
    assert all(r["status"] == "level" and "LUFS" in r["error"] for r in recs.values())
    assert not (tmp_path / "inbox" / "2026-11-05" / "b1105-keep3").exists()
    shutil_rm(tmp_path / "runs")
    assert run(tmp_path, renderer=fake_renderer(peak=-0.95)) == 3
    assert all("true peak" in r["error"] for r in summary(tmp_path, "renders.json")["renders"].values())


def shutil_rm(p: Path) -> None:
    import shutil
    shutil.rmtree(p)


def test_an_infrastructure_fault_is_exit_1(tmp_path):
    assert run(tmp_path, renderer=fake_renderer(boom=True)) == 1
    recs = summary(tmp_path, "renders.json")["renders"]
    assert all(r["status"] == "error" and "infrastructure" in r["error"] for r in recs.values())


def test_bad_arguments_and_an_existing_batch_exit_2(tmp_path):
    assert run(tmp_path, "--briefs", "d09") == 2
    assert run(tmp_path, "--name", "b 1105") == 2
    assert run(tmp_path) == 0
    assert run(tmp_path) == 2                                 # the batch exists: never overwritten


def test_a_dry_run_renders_and_builds_nothing(tmp_path, capsys):
    assert run(tmp_path, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "would render d05_horn" in out and "would build" in out
    assert not (tmp_path / "runs" / "drafts-1" / "renders").exists() and not (tmp_path / "inbox").exists()


def test_one_seed_gives_the_same_batch(tmp_path):
    assert run(tmp_path / "a") == 0
    assert run(tmp_path / "b") == 0
    a = tmp_path / "a" / "inbox" / "2026-11-05" / "b1105-keep3"
    b = tmp_path / "b" / "inbox" / "2026-11-05" / "b1105-keep3"
    files = sorted(p.name for p in a.iterdir())
    assert files == sorted(p.name for p in b.iterdir())
    assert (a / "manifest.json").read_text() == (b / "manifest.json").read_text()


def test_the_clip_id_is_the_shootouts():
    from lofi_rl.shootout.render import clip_id as shootout_clip_id
    man = {"schema": "lofi-rl/render/1", "chain": {"cycles": 2}, "loop": {"canon_sha256": "a" * 64},
           "brief": {"bpm": 82, "tonic": 9, "mode": "dorian", "energy": "mid"},
           "palette": {"id": "draft_day2_rhodes_jazz", "sha256": "b" * 64}}
    assert clip_id(man) == shootout_clip_id(man)


def test_the_job_script():
    s = SBATCH.read_text()
    assert re.search(r"^#SBATCH --job-name=lofi-drafts$", s, re.M)
    assert re.search(r"^#SBATCH --partition=main$", s, re.M)
    cpus = int(re.search(r"^#SBATCH --cpus-per-task=(\d+)$", s, re.M).group(1))
    mem = int(re.search(r"^#SBATCH --mem=(\d+)G$", s, re.M).group(1))
    h, mi, sec = map(int, re.search(r"^#SBATCH --time=(\d+):(\d+):(\d+)$", s, re.M).group(1, 2, 3))
    worst = (cpus + mem / 4) * (60 * h + mi + sec / 60)
    stated = int(re.search(r"Cost: .*-> at most (\d+) SU", s).group(1))
    assert worst == stated == 180 and stated <= 500
    assert "MODE=scorer source scripts/job_env.sh" in s and "lofi_rl.drafts.job" in s and "make submit" in s


# --------------------------------------------------------------------------------------------- real renders
def gu_palette(path: Path) -> Path:
    """A GeneralUser-only stand-in of palettes/draft_day2.yaml: the same names, settings and chains, every instrument a
    GU preset (keys GU Tine EP 4, both kits the GU Jazz kit with GU Acoustic Bass 32, the alto GU Alto Sax 65) at the GU
    instruments' measured trims and faders (levels.json: gu_tine_ep -6.04, kit_gu_jazz 15.21, bass_gu_acoustic 5.18,
    sax_gu_alto 0.89, all trims 0). Tests only: it is not the Day-2 band."""
    dp = P.load()
    doc = copy.deepcopy(dp.doc)
    doc["id"] = "draft_gu_test"
    doc["sources"] = {"gu": doc["sources"]["gu"]}
    for k in doc["keys"]:
        doc["keys"][k] = {"src": "gu", "bank": 0, "prog": 4, "vel": doc["keys"][k]["vel"]}
    jazz = dp.doc["kits"]["jazz"]
    for k in doc["kits"]:
        doc["kits"][k] = {"bass": {"src": "gu", "bank": 0, "prog": 32, "vel": jazz["bass"]["vel"]},
                          "choke": copy.deepcopy(jazz["choke"]), "lanes": copy.deepcopy(jazz["lanes"])}
    doc["sax"]["alto"].update(src="gu", bank=0, prog=65)
    lv = copy.deepcopy(dp.levels)
    for rec in lv["keys"].values():
        rec.update(trim_db=0.0, fader_db=-6.04)
    for rec in lv["kits"].values():
        rec.update(trim_db=0.0, fader_db=15.21)
        rec["bass"].update(trim_db=0.0, fader_db=5.18)
    lv["sax"]["alto"].update(trim_db=0.0, fader_db=0.89)
    for b in ("sax", "keys", "bass"):
        doc["mix"]["buses"][b][0] = ["Gain", {"gain_db": 0.0}]
    doc["mix"]["bus_gain_db"] = {"sax": 0.89, "drums": 15.21, "bass": 5.18, "keys": -6.04}
    block = "".join(f"{P.BLOCK}{line}".rstrip() + "\n" for line in P.yaml_text({"draft_levels": lv}).splitlines())
    path.write_text("# tests only: a GeneralUser stand-in for palettes/draft_day2.yaml\n" + block + P.yaml_text(doc),
                    encoding="utf-8")
    assert P.problems(P.load(path)) == []
    return path


@pytest.mark.laptop
@needs_fluidsynth
def test_three_fixture_drafts_render_for_real_into_a_batch_the_page_loads(tmp_path):
    briefs = yaml.safe_load((ROOT / "drafts" / "v1" / "briefs.yaml").read_text(encoding="utf-8"))
    briefs["briefs"] = {"d01": {"energy": "low", "mood": "melancholy", "bpm": 95, "tonic": 3, "key": "Eb aeolian"}}
    briefs["count"] = 1
    folder = tmp_path / "v1"
    folder.mkdir()
    (folder / "briefs.yaml").write_text(yaml.safe_dump(briefs, sort_keys=False), encoding="utf-8")
    for name in ("d01_groove.ls", "d01_horn.ls", "d01_chords.ls"):
        (folder / name).write_text(FIXTURES[name], encoding="utf-8")
    pal = gu_palette(tmp_path / "draft_gu.yaml")
    argv = ["--drafts", str(folder), "--run", str(tmp_path / "runs" / "drafts-1"), "--inbox", str(tmp_path / "inbox"),
            "--date", "2026-11-05", "--jobs", "3", "--seed", "0000000000000001", "--palette", str(pal),
            "--cache", str(tmp_path / "cache")]
    assert J.main(argv) == 0
    m, h = load_batch(tmp_path / "inbox" / "2026-11-05" / "b1105-keep3")
    assert len(m.rounds) == 1 and len(m.rounds[0].clips) == 3
    recs = summary(tmp_path, "renders.json")["renders"]
    for rec in recs.values():
        assert rec["status"] == "ok" and abs(rec["lufs"] + 14) <= 0.1 and rec["true_peak_dbtp"] <= -1.0
        assert rec["n"] == loop_samples(95) and rec["flac_frames"] == rec["n"]
        assert rec["palette"]["id"].startswith("draft_gu_test_")
