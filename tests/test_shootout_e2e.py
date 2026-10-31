"""End to end on the laptop: two tiny rounds through the real pipeline with dev_gu, built into a batch and checked.

A two-round plan (one stage, two options, two tiny loops) is drawn, two palettes stand in for the generator's
(section 7.8: dev_gu with `sax_expr.enabled` true or false), the job's render driver renders the four clips four
at a time through `render.pipeline.render`, and the batch builder writes the batch, which
`python -m lofi_rl.ui.manifest check` then passes. The loops are tiny (95 BPM, the shortest pass, and few notes) so
the four renders fit the default suite's time; the reference loops' texts are covered by the draw tests and are
rendered by the cluster job. Nothing is listened to and nothing is claimed about sound."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf
import yaml

from helpers_render import needs_fluidsynth
from helpers_shootout import fake_render
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.shootout.batch import main as batch_main
from lofi_rl.shootout.draw import draw, write_draw
from lofi_rl.shootout.plan import load_plan
from lofi_rl.shootout.render import RECORD_KEYS
from lofi_rl.shootout.render import main as render_main
from lofi_rl.ui.manifest import HIDDEN_NAME, PUBLIC_NAME, load_batch

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "palettes" / "dev_gu.yaml"
# Two sparse loops that pass G1-G6 (dev_gu aliases ep and dusty). Their comp and bass are rests, so each render
# runs one FluidSynth process (the sax) instead of three (RENDER.md 3.1: a stem with no notes runs no process);
# the candidate here is the sax expression, and the keys and bass stems are rendered by tests/test_render*.py.
LOOPS = {
    "t1": "vamp: Im9 IV13 span 2\nkeys: ep rootless\ncomp: ................\nbass: ................\n"
          "drums: dusty\nkick: x............... x...............\nrim: ....x.......x... ....x.......x...\n"
          "A: w5 w3 w1 w2\nB: w5 w3 w1 w7\n",
    "t2": "vamp: Imaj9 IVmaj9 span 2\nkeys: ep rootless\ncomp: ................\nbass: ................\n"
          "drums: dusty\nkick: x............... x...............\nrim: ....x.......x... ....x.......x...\n"
          "A: w3 w2 w1 w5\nB: w3 w2 w1 w6\n",
}
BRIEFS = {"loops": {"t1": {"mood": "wistful", "energy": "low", "bpm": 95, "tonic": 2},
                    "t2": {"mood": "hopeful", "energy": "low", "bpm": 95, "tonic": 7}}}


def tiny_plan(loops: Path) -> dict:
    """A one-stage, two-round plan in shootouts/plan.yaml's schema, on the two tiny loops in `loops`."""
    reg = {"tenor": {"play": [44, 75], "comfort": [48, 72], "home": [50, 61]},
           "alto": {"play": [49, 80], "comfort": [55, 77], "home": [57, 68]}}
    gu = {"gu_tine_ep": {"bus": "keys", "source": "gu", "bank": 0, "prog": 4, "family": "rhodes"},
          "kit_gu_jazz": {"bus": "drums", "base": True},
          "bass_gu_acoustic": {"bus": "bass", "source": "gu", "bank": 0, "prog": 32},
          "sax_gu_tenor": {"bus": "sax", "source": "gu", "bank": 0, "prog": 66, "register": "tenor"}}
    band = {"keys": "gu_tine_ep", "drums": "kit_gu_jazz", "bass": "bass_gu_acoustic", "sax": "sax_gu_tenor"}
    return {
        "version": 1, "scene": "city at night", "base": "shootouts/palettes/base.yaml", "loops": str(loops),
        "loop_names": {"L6": "t1", "L9": "t2"}, "voices": {"keys": "ep", "kit": "dusty"}, "registers": reg,
        "candidates": gu,
        "variants": {"sax_plain": {"stage": "T", "set": {}, "what": "expression on"},
                     "sax_expr_off": {"stage": "T", "set": {"sax_expr.enabled": False}, "what": "expression off"}},
        "vamps": {},
        "stages": {"T": {"day": 1, "bus": "sax", "fills": "sax_variant", "category": "sax_warmth",
                         "what": "sax variant", "symbols": ["m1", "m2"], "options": ["sax_plain", "sax_expr_off"],
                         "draw": {"rule": "any"},
                         "slots": [{"id": "T.1", "loop": "L9", "tiles": ["m1", "m2"]},
                                   {"id": "T.2", "loop": "L6", "tiles": ["m2", "m1"]}]}},
        "days": {1: {"date": "2026-10-31", "stages": ["T"], "band": band,
                     "fixed": {"swing": None, "humanize": None, "sax_variant": "sax_plain", "timing": None,
                               "bed": None, "rain": None}}},
        "batches": [{"names": ["b1031-1-tiny"], "date": "2026-10-31", "day": 1, "rounds": {"T": 2}, "anchors": 0,
                     "category": "sax_warmth", "note": "Which sax do you like best?", "est_min": [1.0, 1.0]}],
        "controls": {"repeats": {"count": 0, "groups": {}, "day": 3, "min_days_after": 2, "tiles": "reversed",
                                 "never": "-"},
                     "anchors": {"count": 0, "per_day": 0, "sour": "-", "gate": "-"}},
        "cuts": [],
        "levels": {"anchors": {"keys": "gu_tine_ep", "drums": "kit_gu_jazz", "bass": "bass_gu_acoustic",
                               "sax": "sax_gu_tenor"}, "sax_target": "sax_gu_tenor", "zero_trim_source": "gu",
                   "linear_drop_db": 6.02, "linear_tol_db": 0.1, "median_tol_lu": 0.05, "loop_tol_lu": 0.1},
    }


def _palettes(doc: dict, folder: Path) -> None:
    """Stand-ins for the generator: dev_gu under the palette's key, the sax expression on or off."""
    folder.mkdir(parents=True)
    text = DEV.read_text()
    assert text.count("id: dev_gu\n") == 1 and text.count("  enabled: true ") == 1
    for key, spec in doc["palettes"].items():
        out = text.replace("id: dev_gu\n", f"id: {key}\n")
        if spec["band"]["sax_variant"] == "sax_expr_off":
            out = out.replace("  enabled: true ", "  enabled: false")
        (folder / f"{key}.yaml").write_text(out)


@pytest.mark.laptop
@needs_fluidsynth
def test_two_tiny_rounds_render_through_the_pipeline_build_and_pass_the_check(tmp_path):
    (tmp_path / "loops").mkdir()
    for name, text in LOOPS.items():
        (tmp_path / "loops" / f"{name}.ls").write_text(text)
    (tmp_path / "loops" / "briefs.yaml").write_text(yaml.safe_dump(BRIEFS))
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(yaml.safe_dump(tiny_plan(tmp_path / "loops"), sort_keys=False))
    plan = load_plan(plan_file)
    run = tmp_path / "runs" / "shootout-e2e"
    doc = draw(plan, "day1", seed=0xE2E)
    draw_json = write_draw(doc, run / "summary" / "draw.json")
    assert sorted(doc["palettes"]) == ["t_sax_expr_off", "t_sax_plain"] and len(doc["renders"]) == 4
    _palettes(doc, run / "summary" / "palettes")

    assert render_main(["--draw", str(draw_json), "--palettes", str(run / "summary" / "palettes"),
                        "--out", str(run / "renders"), "--jobs", "4", "--workers", "1"]) == 0
    assert batch_main(["build", "--plan", str(plan_file), "--stage", "day1", "--renders", str(run / "renders"),
                       "--out", str(tmp_path / "inbox")]) == 0

    d = tmp_path / "inbox" / "2026-10-31" / "b1031-1-tiny"
    p = subprocess.run([sys.executable, "-m", "lofi_rl.ui.manifest", "check", str(d)], capture_output=True,
                       text=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert p.returncode == 0 and p.stdout.startswith("ok: b1031-1-tiny (shootout), 2 rounds, 4 clips"), p.stderr
    m, h = load_batch(d)
    assert {f.name for f in d.iterdir()} == {PUBLIC_NAME, HIDDEN_NAME} | {c.file for r in m.rounds for c in r.clips}
    renders = json.loads((run / "summary" / "renders.json").read_text())["renders"]
    for r in m.rounds:
        hr = h.rounds[r.id]
        n = loop_samples(r.bpm)
        assert hr.kind == "none" and r.bpm == plan.loop_named(hr.brief_id).brief.bpm
        assert {hr.clips[c]["option"] for c in r.ids} == {"sax_plain", "sax_expr_off"}
        pcm = set()
        for c in r.clips:
            info = hr.clips[c.id]
            rec = renders[info["render"]]
            man = json.loads((run / "renders" / info["render"] / "manifest.json").read_text())
            assert info["pcm_sha256"] == man["audio_sha256"] == rec["pcm_sha256"]
            assert info["palette"] == f"{man['palette']['id']}@{man['palette']['sha256'][:8]}"
            assert man["gates"] == {"G1-G6": "pass", "G7": "pass"} and abs(man["loudness"]["lufs"] + 14) <= 0.1
            assert man["loudness"]["true_peak_dbtp"] <= -1.0
            assert sf.info(str(d / c.file)).frames == n == man["n"] == c.frames     # one pass, decodable
            assert info["bus_lufs"] == pytest.approx(man["buses"]["sax"]["lufs"] + man["buses"]["sax"]["gain_db"])
            lo, hi = map(int, info["sax_range"].split("-"))
            assert 44 <= lo <= hi <= 75                                     # the tenor's playable range
            assert (run / "renders" / info["render"] / "loop.ls").read_text() == plan.loop_named(hr.brief_id).text
            pcm.add(info["pcm_sha256"])
        assert len(pcm) == 2                                                # the two options render differently
    real = next(iter(renders.values()))
    fake = fake_render(tmp_path / "fake", "x", doc["renders"][real["render"]])
    assert set(real) == set(fake) == set(RECORD_KEYS) and set(real["palette"]) == set(fake["palette"])
    assert set(real["buses"]["sax"]) == set(fake["buses"]["sax"])
