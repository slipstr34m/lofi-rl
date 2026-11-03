"""The sign-off job (lofi_rl.freeze.signoff and .batch; `lofi-drafts --signoff`; docs/DRAFTS.md §7) on a stand-in world.

The world is small and committed-only: the six fixture drafts of tests/test_drafts_check.py (briefs d01 and d05), a
GeneralUser stand-in of the draft palette with non-zero trims (so the twin's fields-for-slots carriage is exercised), a
stand-in of the frozen palette made from it the way the generator makes the real one (the freeze spec §3.3: the
levels as fields, the kit lanes untrimmed, no chain-head slots, bus_gain_db 0, a rain bed), a synthetic noise file as
the rain, and a stand-in renderer: the real parse, compile and bed placement (render.bed), synthetic music that depends
on the loop and on each bus's effective levels, stated loudness numbers. keep3's records are that renderer's renders
with the draft palette's render palettes, so `twin0 = keep3` holds exactly when the twin carries keep3's levels.
The job runs once per world; its refusals are checked on the verify stage's own function. Nothing is listened to and
nothing is claimed about sound."""
import copy
import hashlib
import json
import shutil
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import yaml

from lofi_rl.brief import Brief
from lofi_rl.compile.compiler import compile_loop
from lofi_rl.drafts import job as J
from lofi_rl.drafts import palette as P
from lofi_rl.drafts.render import render_all
from lofi_rl.dsl import canon, canon_sha, parse
from lofi_rl.freeze import batch as SB
from lofi_rl.freeze import checks as C
from lofi_rl.freeze import signoff as S
from lofi_rl.palette import load_palette
from lofi_rl.render.pipeline import RenderError, manifest_json
from lofi_rl.render.wavio import pcm_sha256, write_float
from lofi_rl.ui.manifest import TEXT_MAX, load_batch
from test_drafts_check import FIXTURES, write_folder
from test_drafts_job import gu_palette

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import approve_job  # noqa: E402

KEPT = ("d01_chords", "d05_groove")          # wurli on the jazz kit; rhodes on the studio kit
CELLS = {("low", "melancholy"), ("mid", "wistful")}
BUSES = ("sax", "keys", "bass", "drums")
RAIN_START_S, RAIN_FRAMES = 0.5, 1_200_000   # the stand-in rain: 0.5 s + the slowest loop + 1 s fits in it
E2 = [22_050, 66_150]                         # the stand-in excerpt's stretch (frames)
VARIANTS = {"excerpt_e2": {"bed.excerpt": "e2", "bed.delta_lu": 18},     # shootouts/plan.yaml's shape
            "rain_d24": {"bed.excerpt": "e1", "bed.delta_lu": 24}}
PICKS = {"excerpt_from": "excerpt_e2", "delta_from": "rain_d24"}


# --------------------------------------------------------------------------------------------- the stand-in world
def effective(pal, compiled, lv) -> dict:
    """Each bus's effective (trim, fader) as drafts.render reports them: slot + field, bus_gain_db + field."""
    out = {}
    for b in BUSES:
        if b == "drums":
            slot = float(pal.kits[compiled.kit].lanes["kick"].db)
        else:
            ch = pal.mix.buses[b]
            slot = float(ch[0].params.get("gain_db", 0.0)) if ch and ch[0].name == "Gain" else 0.0
        trim = slot + lv[b].trim_db if lv[b].trim_db else slot
        fader = pal.mix.bus_gain_db[b] + lv[b].fader_db if lv[b].fader_db else pal.mix.bus_gain_db[b]
        out[b] = (trim, fader)
    return out


def fake_renderer(*, lufs: float = -14.02, peak: float = -1.5, g7_texts=()):
    """render.pipeline.render's shape (module doc)."""
    from lofi_rl.render.bed import bed_loop, place, samples
    from lofi_rl.render.loudness import lufs as measure

    def render(text, brief, palette, *, cache=None, workers=1):
        if text in g7_texts:
            raise RenderError("G7", "a stand-in G7 failure")
        res = parse(text, vocab=palette.vocab(), mode=brief.mode)
        if not res.gate.ok:
            raise RenderError(res.gate.code, res.gate.detail)
        compiled = compile_loop(res.ast, brief, palette)
        n, seed = compiled.n, int(canon_sha(res.ast)[:8], 16)
        lv = palette.bus_levels(compiled.keys_name, compiled.kit, compiled.sax_name)
        eff = effective(palette, compiled, lv)
        scale = 10 ** (sum(t + f for t, f in eff.values()) / 400)
        x = (0.05 * scale * np.sin(np.arange(n) * (0.01 + (seed % 97) * 1e-4))).astype(np.float32)
        pre = np.stack([x, x])
        music_lufs = float(measure(pre, 44100))
        mixed, bed = pre, None
        if palette.bed is not None:
            scaled, gain, own = place(bed_loop(palette, n), music_lufs, palette.bed.delta_lu)
            mixed = (pre.astype(np.float64) + scaled).astype(np.float32)
            bed = {"src": palette.bed.src, "delta_lu": palette.bed.delta_lu, "start_s": palette.bed.start_s,
                   "xfade_s": palette.bed.xfade_s, "xfade_samples": samples(palette.bed.xfade_s),
                   "lufs_source": float(own), "gain_db": float(gain), "lufs": music_lufs - palette.bed.delta_lu}
        audio, music = (mixed * np.float32(2.0)).astype(np.float32), (pre * np.float32(2.0)).astype(np.float32)
        manifest = {"schema": "lofi-rl/render/1", "chain": {"cycles": 2, "fluid_preroll_cycles": 1},
                    "loop": {"canon": canon(res.ast), "canon_sha256": canon_sha(res.ast),
                             "compiled_sha256": compiled.digest(), "repairs": [], "p_fix": 0.0},
                    "brief": {"scene": brief.scene, "energy": brief.energy, "mood": brief.mood, "mode": brief.mode,
                              "bpm": brief.bpm, "tonic": brief.tonic, "key": brief.key_name},
                    "palette": {"id": palette.id, "sha256": palette.sha256, "frozen": palette.frozen},
                    "platform": {"system": "stand-in"}, "n": n, "sr": 44100,
                    "instruments": {"keys": compiled.keys_name, "kit": compiled.kit, "sax": compiled.sax_name,
                                    "sax_warm": palette.mix.sax_warm},
                    "stems": {b: f"{seed:08x}{b}" for b in BUSES}, "oneshots": {},
                    "buses": {b: {"lufs": -40.0 + eff[b][0] / 10 + (seed % 7) / 100, "gain_db": palette.mix.bus_gain_db[b],
                                  "instrument": lv[b].instrument, "trim_db": lv[b].trim_db, "fader_db": lv[b].fader_db}
                              for b in BUSES},
                    "music_lufs": music_lufs, "bed": bed,
                    "loudness": {"lufs": lufs, "true_peak_dbtp": peak, "gain_db": 6.0, "iterations": 2},
                    "gates": {"G1-G6": "pass", "G7": "pass"}, "audio_sha256": pcm_sha256(audio),
                    "music_sha256": pcm_sha256(music)}
        return SimpleNamespace(audio=audio, music=music, pre_limiter=pre, manifest=manifest, compiled=compiled)
    return render


def draft_standin(path: Path) -> P.DraftPalette:
    """test_drafts_job.gu_palette with non-zero trims: on the band the palette plays as written (rhodes, studio: its
    slots and bus_gain_db), on wurli and on the studio kit (in its lanes' db, the draft carriage)."""
    dp = P.load(gu_palette(path.with_name("gu_base.yaml")))
    doc, lv = copy.deepcopy(dp.doc), copy.deepcopy(dp.levels)
    lv["keys"]["rhodes"].update(trim_db=2.5, fader_db=-6.04)
    lv["keys"]["wurli"].update(trim_db=-3.22, fader_db=-6.435)
    lv["kits"]["studio"].update(trim_db=-14.94, fader_db=15.26)
    lv["kits"]["studio"]["bass"].update(trim_db=1.5, fader_db=5.09)
    lv["sax"]["alto"].update(trim_db=6.63, fader_db=2.55)
    for lane in doc["kits"]["studio"]["lanes"].values():
        lane["db"] = lane["db"] + -14.94
    for bus, t in (("sax", 6.63), ("keys", 2.5), ("bass", 1.5)):
        doc["mix"]["buses"][bus][0] = ["Gain", {"gain_db": t}]
    doc["mix"]["bus_gain_db"] = {"sax": 2.55, "drums": 15.26, "bass": 5.09, "keys": -6.04}
    block = "".join(f"{P.BLOCK}{line}".rstrip() + "\n" for line in P.yaml_text({"draft_levels": lv}).splitlines())
    path.write_text("# tests only: a GeneralUser stand-in for palettes/draft_day2.yaml\n" + block + P.yaml_text(doc),
                    encoding="utf-8")
    out = P.load(path)
    assert P.problems(out) == []
    return out


def frozen_doc_from_draft(dp: P.DraftPalette, *, bed: dict | None = None, rain: dict | None = None,
                          keys_offset_db: dict | None = None) -> dict:
    """The frozen palette's document from a draft palette, as the freeze spec §3.3 builds it: every level a field,
    each kit's lanes untrimmed, no chain-head trim slots, bus_gain_db 0; with `bed`, the bed and its `rain` source;
    with `keys_offset_db`, keys faders moved by those decimals (a level change of his, spec §4.2)."""
    d = copy.deepcopy(dp.doc)
    d["id"], d["frozen"] = "lofi_rain_v1", None
    lv = dp.levels
    for name, rec in lv["keys"].items():
        d["keys"][name].update(trim_db=rec["trim_db"], fader_db=rec["fader_db"])
    for name, rec in lv["sax"].items():
        d["sax"][name].update(trim_db=rec["trim_db"], fader_db=rec["fader_db"])
    for name, rec in lv["kits"].items():
        kit = d["kits"][name]
        kit["lanes"] = P.untrimmed_lanes(kit["lanes"], float(rec["trim_db"]))
        kit["bass"].update(trim_db=rec["bass"]["trim_db"], fader_db=rec["bass"]["fader_db"])
        d["kits"][name] = {"trim_db": rec["trim_db"], "fader_db": rec["fader_db"], **kit}
    for bus in P.TRIM_BUSES:
        assert d["mix"]["buses"][bus][0][0] == "Gain"
        d["mix"]["buses"][bus] = d["mix"]["buses"][bus][1:]
    d["mix"]["bus_gain_db"] = {b: 0.0 for b in P.BUS_ORDER}
    if bed is not None:
        d["sources"]["rain"], d["bed"] = rain, bed
    for name, db in (keys_offset_db or {}).items():
        d["keys"][name]["fader_db"] = float(Decimal(repr(d["keys"][name]["fader_db"])) + Decimal(repr(db)))
    return d


def write_rain(assets: Path) -> tuple[Path, dict]:
    """The stand-in rain (mono float32 noise) under assets/, and its excerpt record (shootouts/rain_excerpts.json's
    shape) with e1 and e2."""
    f = assets / "derived" / "rain" / "standin_rain.wav"
    f.parent.mkdir(parents=True, exist_ok=True)
    x = (np.random.default_rng(3).standard_normal(RAIN_FRAMES) * 0.05).astype(np.float32)
    write_float(f, x[None, :], 44100)                 # no PEAK chunk, so the bytes (and the sha256) never vary
    raw, _ = sf.read(str(f), dtype="float32", always_2d=True)
    stretch = lambda lo, hi: pcm_sha256(np.ascontiguousarray(raw.T[:, lo:hi]))   # noqa: E731
    rec = {"schema": "lofi-rl/rain-excerpts/1", "parameters": {"xfade_s": 1.0},
           "source": {"asset_path": "derived/rain/standin_rain.wav",
                      "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "license": "CC0 1.0 (a test stand-in)",
                      "url": None, "attribution": None},
           "excerpts": {"e1": {"start_s": 1.0, "samples": [0, 44_100], "pcm_sha256": stretch(0, 44_100)},
                        "e2": {"start_s": RAIN_START_S, "samples": E2, "pcm_sha256": stretch(*E2)}}}
    return f, rec


def gate_text(kept_files: list[str]) -> str:
    rows = "".join(f"| {Path(f).name[:3]} | stand-in | `{f}` |\n" for f in kept_files)
    return ("# GATE stand-in\n\n## Kept drafts (keep3): voted (a test stand-in)\n\n| Brief | Energy, mood | Kept |\n"
            f"|---|---|---|\n{rows}\n## How to approve\n")


def build_world(tmp: Path, mp: pytest.MonkeyPatch, *, keys_offset_db: dict | None = None) -> SimpleNamespace:
    renderer = fake_renderer()
    assets = tmp / "assets"
    rain_file, rain = write_rain(assets)
    (tmp / "rain_excerpts.json").write_text(json.dumps(rain), encoding="utf-8")
    dp = draft_standin(tmp / "draft_gu.yaml")
    source = {"path": rain["source"]["asset_path"], "sha256": rain["source"]["sha256"], "license": "CC0 1.0",
              "mode": "fetch", "url": None, "converted_from": None, "attribution": None}
    bed = {"src": "rain", "start_s": RAIN_START_S, "xfade_s": 1.0, "delta_lu": 24.0}
    frozen = tmp / "lofi_rain_v1.yaml"
    frozen.write_text("# tests only: a stand-in of the frozen palette\n" + P.yaml_text(
        frozen_doc_from_draft(dp, bed=bed, rain=source, keys_offset_db=keys_offset_db)), encoding="utf-8")
    v1 = write_folder(tmp / "v1")
    files = {Path(f).stem: J._label(v1 / f) for f in FIXTURES}
    gate = tmp / "GATE_palette.md"
    gate.write_text(gate_text([files[k] for k in KEPT]), encoding="utf-8")
    mp.setenv("LOFI_ASSETS", str(assets))
    for name, value in (("GATE", gate), ("RAIN", tmp / "rain_excerpts.json"), ("DRAFT_PALETTE", dp.path)):
        mp.setattr(S, name, value)
    calls: list[list[str]] = []
    mp.setattr(S, "freeze_check", lambda args, root=ROOT: (calls.append([str(a) for a in args]) or 0, "ok: stand-in"))
    mp.setattr(S, "plan_variants", lambda path: copy.deepcopy(VARIANTS))
    mp.setattr(S, "BED", dict(PICKS))
    mp.setattr(S, "JOBS", {"day1": 1, "day2": 2, "day3": 3, "keep3": 7})
    real_refs = S.reference_loops
    mp.setattr(S, "reference_loops", lambda pal: {k: v for k, v in real_refs(pal).items() if v[0] in CELLS})

    def fake_repeat(task, out_wav, palette_path, cache):
        r = renderer(task["text"], Brief(**task["brief"]), load_palette(palette_path, verify=False))
        text = manifest_json(r.manifest)
        Path(out_wav).with_suffix(".manifest.json").write_text(text, encoding="utf-8")
        return {"exit": 0, "process_s": 0.01, "manifest_sha256": C.sha256_bytes(text.encode()),
                "audio_sha256": r.manifest["audio_sha256"], "music_sha256": r.manifest["music_sha256"],
                "n": r.manifest["n"]}
    mp.setattr(S, "run_repeat", fake_repeat)
    # keep3: the kept drafts through the draft palette's render palettes, as lofi-drafts rendered them
    briefs = {"d01": {"energy": "low", "mood": "melancholy", "bpm": 81, "tonic": 3},
              "d05": {"energy": "mid", "mood": "wistful", "bpm": 82, "tonic": 9}}
    combos = {"d01_chords": ("wurli", "jazz"), "d05_groove": ("rhodes", "studio")}
    index = P.write_render_palettes(dp, combos.values(), tmp / "k3" / "palettes")
    pal_file = {(e["keys"], e["kit"]): tmp / "k3" / "palettes" / e["file"] for e in index["palettes"]}
    tasks = [{"render": rid, "file": files[rid], "text": FIXTURES[f"{rid}.ls"], "brief_id": rid[:3],
              "brief": briefs[rid[:3]], "persona": rid[4:], "keys": combos[rid][0], "kit": combos[rid][1],
              "palette": str(pal_file[combos[rid]]), "s_star": dp.levels["S_star"]} for rid in KEPT]
    render_all(tasks, tmp / "k3" / "renders", jobs=2, workers=1, summary_dir=tmp / "k3" / "summary", renderer=renderer)
    k3 = tmp / "k3" / "summary" / "renders.json"
    expect = C.expectations(json.loads(k3.read_text()), renders_sha256=C.file_sha256(k3),
                            renders_path="runs/drafts-7/summary/renders.json", job=7, kept=C.gate_kept(gate), draft=dp)
    (v1 / C.EXPECT_NAME).write_text(C.expect_text(expect), encoding="utf-8")
    return SimpleNamespace(tmp=tmp, v1=v1, frozen=frozen, dp=dp, gate=gate, k3=k3, expect=expect, calls=calls,
                           renderer=renderer, files=files, rain_file=rain_file)


def argv(w, *extra, run: str = "drafts-1", inbox: str = "inbox") -> list[str]:
    return ["--drafts", str(w.v1), "--signoff", str(w.frozen), "--run", str(w.tmp / "runs" / run), "--inbox",
            str(w.tmp / inbox), "--date", "2026-11-05", "--jobs", "4", "--seed", "00000000000000bb", "--cache",
            str(w.tmp / "cache"), *extra]


def summary(w, name: str, run: str = "drafts-1"):
    return json.loads((w.tmp / "runs" / run / "summary" / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        w = build_world(tmp_path_factory.mktemp("signoff"), mp)
        w.rc = J.main(argv(w), renderer=w.renderer)
        yield w
    finally:
        mp.undo()


def state_of(w, run: str = "drafts-1") -> dict:
    """The verify stage's input, read back from a run's summary."""
    renders, index = summary(w, "renders.json", run), summary(w, "palettes/index.json", run)
    return {"pal": load_palette(w.frozen, verify=False), "expect": C.load_expectations(w.v1 / C.EXPECT_NAME),
            "bed": index["bed"]["want"], "changes": index["changes"], "run": w.tmp / "runs" / run,
            "runs_root": w.tmp / "runs", "palette_record": renders["palette"],
            "records": {"kept": renders["kept"], "others": renders["others"]}}


# --------------------------------------------------------------------------------------------- the job
def test_the_job_runs_every_stage_and_builds_one_triage_batch(world, capsys):
    assert world.rc == 0
    stages = summary(world, "stages.json")
    assert [s["stage"] for s in stages["stages"]] == ["check", "palettes", "render", "verify", "build"]
    assert all(s["exit"] == 0 for s in stages["stages"])
    args = summary(world, "args.json")
    pal = load_palette(world.frozen, verify=False)
    assert args["mode"] == "signoff" and args["batch"] == "b1105-signoff" and args["palette_sha256"] == pal.sha256
    m, h = load_batch(world.tmp / "inbox" / "2026-11-05" / "b1105-signoff")
    assert m.mode == "triage" and len(m.rounds) == len(KEPT) and all(len(r.clips) == 1 for r in m.rounds)
    assert h.category == "signoff" and all(hr.kind == "none" for hr in h.rounds.values())
    infos = [next(iter(hr.clips.values())) for hr in h.rounds.values()]
    assert {i["render"] for i in infos} == set(KEPT)
    assert all(i["palette"] == f"lofi_rain_v1@{pal.sha256[:8]}" and i["bed_excerpt"] == "e2" and i["bed_delta_lu"] == 24.0
               for i in infos)
    verify = summary(world, "verify.json")
    lo, hi = verify["checks"]["bed_heard"]["kept_below"]
    assert m.note == SB.note_text(2, summary(world, "palettes/index.json")["bed"]["want"], (lo, hi))
    assert f"{lo:.1f}-{hi:.1f} below" in m.note and "24 below" in m.note and "18 below" in m.note
    assert len(m.note) <= TEXT_MAX
    batch = summary(world, "batch.json")
    assert batch["push"].startswith("Needs ear check: b1105-signoff.") and batch["mode"] == "triage"
    assert [x[0] for x in world.calls] == ["--palette", "--palette"]           # the gate check, then --verify
    assert "--gate" in world.calls[0] and "--verify" in world.calls[1] and "--levels-day1" not in world.calls[1]


def test_verify_passes_every_check_and_records_what_it_measured(world):
    v = summary(world, "verify.json")
    assert v["ok"] and v["problems"] == [] and v["twin"] == "twin0"
    c = v["checks"]
    assert all(ch["ok"] for ch in c.values())
    beds = c["bed_placed"]["beds"]
    assert set(beds) == set(KEPT) | {"d01_groove", "d01_horn", "d05_horn", "d05_chords", "ref_l1_melancholy_low",
                                     "ref_example"}
    assert all(abs(b["placed_lu"] + 24.0) <= S.PLACED_TOL_LU for b in beds.values())
    assert set(c["iterations"]["iterations"]) == set(beds) and c["iterations"]["most"] == 2
    assert c["render_time"]["slowest_s"] is not None and c["render_time"]["within"] is True
    renders = summary(world, "renders.json")
    for rid in KEPT:
        row, e = renders["kept"][rid], world.expect["loops"][rid]
        assert row["twin0"]["pcm_sha256"] == e["pcm_sha256"]                    # twin0 is what keep3 rendered
        assert row["frozen"]["pre_limiter_sha256"] == row["twin0"]["pre_limiter_sha256"]
        assert row["frozen"]["music_lufs"] == row["twin0"]["music_lufs"]
        assert row["frozen"]["pcm_sha256"] != e["pcm_sha256"]                   # the bed is the one change
        assert {b: (v["lufs"], v["trim_db"], v["fader_db"]) for b, v in row["frozen"]["buses"].items()} == \
            {b: (v["lufs"], v["trim_db"], v["fader_db"]) for b, v in e["buses"].items()}
        assert row["repeat"]["manifest_sha256"] == row["frozen"]["manifest_sha256"]
    assert renders["kept"]["d05_groove"]["frozen"]["buses"]["drums"]["trim_db"] == -14.94   # studio's kit trim
    assert renders["kept"]["d01_chords"]["frozen"]["buses"]["keys"]["trim_db"] == -3.22     # wurli's
    assert all(r["kind"] in ("draft", "reference") for r in renders["others"].values())
    assert renders["others"]["ref_example"]["brief_id"] == "d05" and renders["others"]["ref_example"]["status"] == "ok"
    assert (world.tmp / "runs" / "drafts-1" / "renders" / "others" / "ref_example.flac").is_file()


def test_the_check_and_palettes_record_what_they_found(world):
    check = summary(world, "check.json")
    assert check["problems"] == [] and check["keep3_scratch"]["present"] is False
    assert check["a"]["folder"] == {"drafts": 6, "ok": 6, "missing": [], "references": check["a"]["folder"]["references"],
                                    "failing": {}}
    assert check["b"]["count"] == 9 and check["b"]["identical"] is True
    assert check["c"]["fullmatch"] == check["c"]["drafts"] == 6
    pal = summary(world, "palettes/index.json")
    assert pal["problems"] == [] and pal["changes"] == {} and set(pal["twins"]) == {"twin0"}
    assert all(not x["present"] and x["note"] == "not on scratch" for x in pal["levels_scratch"].values())
    assert pal["bed"]["stretch_sha256"] == pal["bed"]["want"]["pcm_sha256"]
    twin = yaml.safe_load((world.tmp / "runs" / "drafts-1" / "summary" / "palettes" / pal["twins"]["twin0"]["file"])
                          .read_text())
    assert twin["id"] == "lofi_rain_v1_nobed" and twin["bed"] is None and "rain" not in twin["sources"]
    assert twin["keys"]["wurli"]["fader_db"] == -6.435 and twin["kits"]["studio"]["trim_db"] == -14.94


def test_a_dry_run_checks_then_renders_and_builds_nothing(world, capsys):
    capsys.readouterr()
    assert J.main(argv(world, "--dry-run", run="dry", inbox="inbox-dry"), renderer=world.renderer) == 0
    out = capsys.readouterr().out
    assert "would render d01_chords" in out and "[twin0]" in out and "would repeat d05_groove" in out
    assert "would render ref_example" in out and "would build" in out and S.OK_LINE not in out
    assert not (world.tmp / "runs" / "dry" / "renders").exists() and not (world.tmp / "inbox-dry").exists()
    assert [s["stage"] for s in summary(world, "stages.json", "dry")["stages"]] == ["check", "palettes"]


def test_a_level_change_renders_twin1_and_moves_only_its_fader(world, tmp_path_factory):
    """The same world with a synthetic level change: the wurli fader 1.0 dB up in the frozen palette."""
    with pytest.MonkeyPatch.context() as mp:
        changed = build_world(tmp_path_factory.mktemp("signoff_change"), mp, keys_offset_db={"wurli": 1.0})
        assert J.main(argv(changed), renderer=changed.renderer) == 0
        check_the_change(changed)


def check_the_change(changed):
    pal = summary(changed, "palettes/index.json")
    assert set(pal["twins"]) == {"twin0", "twin1"} and list(pal["changes"]) == ["keys.wurli"]
    assert pal["changes"]["keys.wurli"]["keep3"] == [-3.22, -6.435] and pal["changes"]["keys.wurli"]["frozen"] == [-3.22, -5.435]
    v = summary(changed, "verify.json")
    assert v["ok"] and v["twin"] == "twin1"
    renders = summary(changed, "renders.json")
    row, e = renders["kept"]["d01_chords"], changed.expect["loops"]["d01_chords"]
    assert row["twin0"]["pcm_sha256"] == e["pcm_sha256"]                         # twin0 still is keep3
    assert row["frozen"]["pre_limiter_sha256"] == row["twin1"]["pre_limiter_sha256"] != row["twin0"]["pre_limiter_sha256"]
    assert row["frozen"]["buses"]["keys"]["fader_db"] == -5.435 and e["buses"]["keys"]["fader_db"] == -6.435
    assert row["frozen"]["buses"]["keys"]["lufs"] == e["buses"]["keys"]["lufs"]
    other = renders["kept"]["d05_groove"]                                         # rhodes: nothing of it moved
    assert other["frozen"]["pre_limiter_sha256"] == other["twin0"]["pre_limiter_sha256"]
    m, _ = load_batch(changed.tmp / "inbox" / "2026-11-05" / "b1105-signoff")
    assert SB.CHANGE_WORDS in m.note and len(m.note) <= TEXT_MAX
    assert "The keys play at the levels you gave." in summary(changed, "batch.json")["push"]


# --------------------------------------------------------------------------------------------- refusals
@pytest.fixture
def fast(monkeypatch):
    """The bed rebuild memoized per (palette, n, music_lufs, delta): the full runs above measure it for real."""
    real, memo = S.placed_bed, {}

    def placed(pal, n, music_lufs, delta, cache):
        key = (pal.sha256, n, music_lufs, delta)
        if key not in memo:
            memo[key] = real(pal, n, music_lufs, delta, cache)
        return memo[key]
    monkeypatch.setattr(S, "placed_bed", placed)
    return placed


def failing(st: dict) -> dict:
    v = S.verify_doc(st)
    assert not v["ok"]
    return {k: c["problems"] for k, c in v["checks"].items() if c["problems"]}


def test_a_twin_that_is_not_keep3_fails_and_names_the_stems(world, fast, tmp_path):
    st = state_of(world)
    st["records"]["kept"]["d01_chords"]["twin0"]["pcm_sha256"] = "0" * 64
    bad = failing(st)
    assert list(bad) == ["twin_is_kept"] and "d01_chords" in bad["twin_is_kept"][0]

    def named() -> list[str]:
        diffs = S.verify_doc(st)["checks"]["twin_is_kept"]["stem_diffs"]
        assert list(diffs) == ["d01_chords"]
        return diffs["d01_chords"]

    def theirs(runs_root: Path) -> Path:            # where keep3's job (7 here) left the render, as on scratch
        return runs_root / "drafts-7" / "renders" / "d01_chords" / "manifest.json"
    assert named() == [f"keep3's manifest {theirs(st['runs_root'])} is not on scratch"]
    st["runs_root"] = tmp_path / "runs"                                   # a scratch that still holds keep3's renders
    theirs(st["runs_root"]).parent.mkdir(parents=True)
    shutil.copyfile(world.tmp / "k3" / "renders" / "d01_chords" / "manifest.json", theirs(st["runs_root"]))
    assert named() == ["no bus input differs: the difference comes after the bus inputs"]      # twin0 played keep3's
    mine = json.loads((st["run"] / "renders" / "twin0" / "d01_chords" / "manifest.json").read_text(encoding="utf-8"))
    k3 = json.loads(theirs(st["runs_root"]).read_text(encoding="utf-8"))
    k3["stems"]["keys"], k3["oneshots"] = "f" * 64, {"kick": "e" * 64}   # keep3 played other keys and one-shots
    theirs(st["runs_root"]).write_text(json.dumps(k3), encoding="utf-8")
    assert named() == [f"stem keys: {mine['stems']['keys']} here, {'f' * 64} in keep3", "the one-shots differ"]
    st["run"] = tmp_path / "elsewhere"                                    # twin0's own manifest is not there
    assert named() == [f"{tmp_path / 'elsewhere' / 'renders' / 'twin0' / 'd01_chords' / 'manifest.json'} is missing"]


def test_a_bed_placed_off_by_more_than_the_tolerance_fails(world, fast, monkeypatch):
    monkeypatch.setattr(S, "placed_bed", lambda *a: (fast(*a)[0] - 0.011, *fast(*a)[1:]))
    assert list(failing(state_of(world))) == ["bed_placed"]
    monkeypatch.setattr(S, "placed_bed", fast)
    st = state_of(world)
    st["records"]["kept"]["d05_groove"]["frozen"]["bed"]["gain_db"] += 1e-9    # another bed than the rebuilt one
    bad = failing(st)
    assert list(bad) == ["bed_placed"] and "d05_groove" in bad["bed_placed"][0]


def test_an_n_mismatch_fails(world, fast):
    st = state_of(world)
    st["expect"]["loops"]["d05_groove"]["n"] += 64
    assert list(failing(st)) == ["n"]
    st = state_of(world)
    st["records"]["kept"]["d01_chords"]["repeat"]["n"] = 1
    assert list(failing(st)) == ["n"]


def test_a_repeat_that_differs_fails(world, fast):
    st = state_of(world)
    st["records"]["kept"]["d01_chords"]["repeat"]["manifest_sha256"] = "1" * 64
    bad = failing(st)
    assert list(bad) == ["repeat"] and "manifest_sha256" in bad["repeat"][0]
    st = state_of(world)
    st["records"]["kept"]["d05_groove"]["repeat"] = {"exit": 3, "process_s": 1.0, "error": "gate G7: stand-in"}
    assert list(failing(st)) == ["repeat"]


def test_a_g7_failure_among_the_one_at_a_time_renders_fails(world, fast):
    st = state_of(world)
    st["records"]["others"]["d01_groove"].update(status="gate", gate="G7", error="G7: loudness did not converge")
    bad = failing(st)
    assert list(bad) == ["iterations"] and bad["iterations"] == ["d01_groove: G7 failure: G7: loudness did not converge"]
    st = state_of(world)
    st["records"]["others"]["ref_example"].update(status="compile", error="stand-in")
    assert list(failing(st)) == ["status"]


def test_loudness_and_a_bus_level_off_keep3_fail(world, fast):
    st = state_of(world)
    st["records"]["others"]["d05_horn"]["lufs"] = -13.85
    assert list(failing(st)) == ["loudness"]
    st = state_of(world)
    st["records"]["kept"]["d05_groove"]["frozen"]["buses"]["keys"]["fader_db"] = -6.0
    bad = failing(st)
    assert list(bad) == ["bus_levels"] and "d05_groove.frozen: the keys bus's fader_db" in bad["bus_levels"][0]
    st = state_of(world)
    st["records"]["kept"]["d05_groove"]["frozen"]["music_lufs"] += 1e-9
    assert "music_under_bed" in failing(st)


def test_the_g7_render_fails_the_whole_job_before_the_batch(world):
    rc = J.main(argv(world, run="drafts-g7", inbox="inbox-g7"), renderer=fake_renderer(g7_texts={FIXTURES["d05_horn.ls"]}))
    assert rc == 3
    assert [s["stage"] for s in summary(world, "stages.json", "drafts-g7")["stages"]] == ["check", "palettes", "render",
                                                                                         "verify"]
    assert summary(world, "verify.json", "drafts-g7")["checks"]["iterations"]["problems"][0].startswith(
        "d05_horn: G7 failure")
    assert not (world.tmp / "inbox-g7").exists()


def test_a_kept_file_that_is_not_the_text_keep3_rendered_stops_at_the_check(world, tmp_path):
    other = tmp_path / "v1"
    shutil.copytree(world.v1, other)
    text = FIXTURES["d01_chords.ls"]
    edited = text.replace("| q4 e3 e2 h1 |", "| q4 e3 e1 h1 |", 1)           # one sax note of the call
    assert edited != text
    (other / "d01_chords.ls").write_text(edited, encoding="utf-8")
    run = ["--drafts", str(other), "--signoff", str(world.frozen), "--run", str(tmp_path / "runs" / "drafts-2"),
           "--inbox", str(tmp_path / "inbox"), "--date", "2026-11-05", "--seed", "00000000000000bb"]
    assert J.main(run, renderer=world.renderer) == 3
    check = json.loads((tmp_path / "runs" / "drafts-2" / "summary" / "check.json").read_text())
    assert any("d01_chords.ls: sha256" in x and "not the text keep3 rendered" in x for x in check["problems"])
    assert not (tmp_path / "runs" / "drafts-2" / "renders").exists() and not (tmp_path / "inbox").exists()


def test_scratch_cross_checks_absent_recorded_present_with_another_sha_refused(world, tmp_path, monkeypatch):
    expect = world.expect
    absent = S.keep3_cross_check(expect, tmp_path)
    assert absent["present"] is False and absent["problems"] == []
    k3 = tmp_path / "drafts-7" / "summary" / "renders.json"
    k3.parent.mkdir(parents=True)
    shutil.copyfile(world.k3, k3)
    same = S.keep3_cross_check(expect, tmp_path)
    assert same["present"] and same["equal"] and same["problems"] == []
    k3.write_text(k3.read_text() + " ", encoding="utf-8")
    assert "keep3_kept10.json pins" in S.keep3_cross_check(expect, tmp_path)["problems"][0]
    # the shootout levels files: with all three present the freeze check gets them; with one missing it does not
    calls = []
    monkeypatch.setattr(S, "freeze_check", lambda args, root=ROOT: (calls.append([str(a) for a in args]) or 0, "ok"))
    for job in (1, 2, 3):
        f = tmp_path / f"shootout-{job}" / "summary" / "levels.json"
        f.parent.mkdir(parents=True)
        f.write_text("{}", encoding="utf-8")
    pal = load_palette(world.frozen, verify=False)
    st = {"palette_record": {"id": pal.id}}
    doc = S.palettes_stage(pal, world.frozen, world.frozen.read_text(), tmp_path / "pal", tmp_path, st)
    assert doc["problems"] == [] and all(x["present"] for x in doc["levels_scratch"].values())
    assert calls[-1][calls[-1].index("--levels-day3") + 1].endswith("shootout-3/summary/levels.json")
    (tmp_path / "shootout-2" / "summary" / "levels.json").unlink()
    doc = S.palettes_stage(pal, world.frozen, world.frozen.read_text(), tmp_path / "pal2", tmp_path, st)
    assert doc["levels_scratch"]["day2"] == {"path": str(tmp_path / "shootout-2" / "summary" / "levels.json"),
                                             "present": False, "note": "not on scratch"}
    assert "--levels-day1" not in calls[-1]
    monkeypatch.setattr(S, "freeze_check", lambda args, root=ROOT: (3, "problem: levels-day1 sha256 differs"))
    doc = S.palettes_stage(pal, world.frozen, world.frozen.read_text(), tmp_path / "pal3", tmp_path, st)
    assert any("exited 3" in x for x in doc["problems"])


def test_a_bed_that_is_not_the_picks_is_refused(world, tmp_path):
    pal = load_palette(world.frozen, verify=False)
    rain = json.loads((world.tmp / "rain_excerpts.json").read_text())
    want = S.expected_bed(VARIANTS, rain, PICKS)
    assert (want["excerpt"], want["delta_lu"], want["w1_delta_lu"], want["w2_excerpt"]) == ("e2", 24.0, 18.0, "e1")
    assert S.bed_problems(pal, want) == []
    e1 = S.expected_bed(VARIANTS, rain, {"excerpt_from": "rain_d24", "delta_from": "rain_d24"})
    assert any("bed.start_s" in x for x in S.bed_problems(pal, e1))           # rain_d24's own excerpt is e1
    assert any("new to your ears" in x for x in S.bed_problems(pal, e1))
    assert S.stretch_sha256(world.rain_file, E2) == want["pcm_sha256"]
    assert S.stretch_sha256(world.rain_file, [E2[0] + 1, E2[1] + 1]) != want["pcm_sha256"]


# --------------------------------------------------------------------------------------------- batch, args, job script
def test_the_order_never_puts_two_rounds_of_one_cell_together():
    briefs = yaml.safe_load((ROOT / "drafts" / "v1" / "briefs.yaml").read_text())["briefs"]
    cells = {f"{b}_x": (v["energy"], v["mood"]) for b, v in briefs.items()}
    assert sum(1 for c in cells.values() if c == ("low", "wistful")) == 2       # d02 and d10
    for seed in range(20):
        d = SB.draw_order(cells, seed)
        assert d["order_constraint_met"] and sorted(d["order"]) == sorted(cells)
        assert all(cells[a] != cells[b] for a, b in zip(d["order"], d["order"][1:]))
    assert SB.draw_order(cells, 5) == SB.draw_order(cells, 5) != SB.draw_order(cells, 6)


def test_the_note_fits_the_page_for_the_kept_ten():
    bed = {"delta_lu": 24.0, "w1_delta_lu": 18.0}
    for change in (False, True):
        note = SB.note_text(10, bed, (22.64, 23.99), change)
        assert len(note) <= TEXT_MAX and "22.6-24.0 below" in note and note.endswith("Toss, Fine, Good or Great?")
    assert SB.note_text(10, bed, (22.64, 23.99)) == (
        "Sign-off: the same 10 loops you kept, now with the rain you picked in Day 3, at 24 below the music. This rain "
        "at this level is new to your ears: Day 3 played it at 18 below, and 24 below on another rain. After the final "
        "limiter it measures 22.6-24.0 below in these loops. Toss, Fine, Good or Great?")


def test_batch_names_count_the_earlier_sign_offs(tmp_path):
    day = J._date(2026, 11, 5)
    assert SB.batch_name(day, tmp_path) == "b1105-signoff"
    (tmp_path / "2026-11-05" / "b1105-signoff").mkdir(parents=True)
    assert SB.batch_name(day, tmp_path) == "b1105-signoff-2"
    (tmp_path / "2026-11-04" / "b1104-keep3").mkdir(parents=True)
    assert SB.batch_name(J._date(2026, 11, 7), tmp_path) == "b1107-signoff-2"
    assert SB.batch_name(day, tmp_path, "b1105-mine") == "b1105-mine"


def test_signoff_arguments_are_refused_where_they_do_not_belong(world, tmp_path, capsys):
    base = ["--drafts", str(world.v1), "--run", str(tmp_path / "r"), "--inbox", str(tmp_path / "i")]
    assert J.main(base + ["--signoff", str(world.frozen), "--briefs", "d01"]) == 2
    assert J.main(base + ["--signoff", str(world.frozen), "--palette", str(world.dp.path)]) == 2
    assert J.main(base + ["--signoff", str(world.frozen), "--against", str(world.v1)]) == 2
    assert J.main(base + ["--runs-root", str(tmp_path)]) == 2
    assert J.main(base + ["--signoff", str(tmp_path / "nowhere.yaml")]) == 2
    (tmp_path / "i" / "2026-11-05" / "b1105-x").mkdir(parents=True)
    assert J.main(base + ["--signoff", str(world.frozen), "--date", "2026-11-05", "--name", "b1105-x"]) == 2
    assert not (tmp_path / "r").exists()
    assert "--runs-root belongs to --signoff" in capsys.readouterr().err


def test_the_job_script_names_the_mode_and_keeps_its_registered_sha():
    s = (ROOT / "jobs" / "drafts.sbatch").read_text()
    assert approve_job.job_sha(ROOT / "jobs" / "drafts.sbatch") == "85a1ae8e085fd255"
    assert "--signoff palettes/lofi_rain_v1.yaml" in s and "SIGNOFF_OK" in s
