"""The Phase-1 gate's ear-check batch (lofi_rl.gate1.batch; docs/GATE1.md): one triage round per clip that rendered,
the order with no two consecutive rounds in one cell, opaque ids under the leak rules, the two manifests (the public
one holds the scene and each round's bpm only; the hidden one, category gate1, no hidden controls), each FLAC held to
what clips.json records, and every refusal with nothing written. Synthetic runs and one-pass sine FLACs come from
tests/helpers_gate1.py; one test renders two committed drafts through junk's own render step where the frozen
palette's sources are here. Nothing here listens."""
import importlib.util
import json
import os
import re
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from helpers_gate1 import JOB, SEED, copy_run, edit_clips, edit_json, junk_doc, make_run, sha, verbatim_of
from lofi_rl import gate1 as G
from lofi_rl.brief import Brief, sample_brief
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.drafts.batch import BuildError
from lofi_rl.drafts.render import clip_id
from lofi_rl.gate1 import batch as GB
from lofi_rl.gate1 import cli
from lofi_rl.palette import load_palette
from lofi_rl.rl import junk as J
from lofi_rl.ui.manifest import HIDDEN_NAME, PUBLIC_NAME, load_batch, sha256_file

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "palettes" / "lofi_rain_v1.yaml"
DATE, NAME = "2026-11-07", "b1107-gate1"
HEX64 = re.compile(r"[0-9a-f]{64}")


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    root = tmp_path_factory.mktemp("gate1-batch") / "junk-13783877"
    info = make_run(root, render_k=8)
    return root, info


@pytest.fixture
def run_dir(made, tmp_path) -> Path:
    return copy_run(made[0], tmp_path / "junk-13783877")


@pytest.fixture
def fast(made):
    """measure_flac's values for the made FLACs, read when the run was made: for tests about everything else."""
    measured = made[1]["measured"]
    return lambda path: measured[Path(path).name]


def build(run_dir: Path, tmp_path: Path, **kw) -> Path:
    args = dict(name=NAME, date=DATE, out_root=tmp_path / "inbox", seed=SEED)
    args.update(kw)
    return GB.build(G.load_run(run_dir), run_dir / "clips", **args)


def nothing_written(run_dir: Path, tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    assert not (inbox / DATE / NAME).exists()
    assert not inbox.exists() or not any(p.name.startswith(".staging-") for p in inbox.iterdir())
    assert not (run_dir / "summary" / "draw.json").exists() and not (run_dir / "summary" / "batch.json").exists()


# --------------------------------------------------------------------------------------------- the batch
def test_a_gate_batch_passes_the_rating_pages_own_check(run_dir, tmp_path, made):
    folder = build(run_dir, tmp_path)                             # the real measure, as the command runs it
    assert folder == tmp_path / "inbox" / DATE / NAME
    m, h = load_batch(folder)                                      # every file and sha checked, the leak check run
    run = G.load_run(run_dir)
    assert m.mode == "triage" and m.bars == 8 and m.sr == 44100 and len(m.rounds) == 8
    assert all(len(r.clips) == 1 for r in m.rounds)
    pub = json.loads((folder / PUBLIC_NAME).read_text())
    assert set(pub) == {"v", "batch", "mode", "scene", "note", "bars", "sr", "est_min", "rounds"}
    assert pub["scene"] == "city at night"
    assert pub["note"] == "Phase-1 gate: 8 loops, one a round. Toss, Fine, Good or Great?"
    assert all(set(r) == {"id", "bpm", "clips"} and set(r["clips"][0]) == {"id", "file", "sha256"}
               for r in pub["rounds"])
    hid = json.loads((folder / HIDDEN_NAME).read_text())
    assert hid["category"] == G.CATEGORY == h.category
    batch = json.loads((run_dir / "summary" / "batch.json").read_text())
    assert [x["sample"] for x in batch["rounds"]] != [] and sorted(x["sample"] for x in batch["rounds"]) == sorted(
        made[1]["recorded"])
    by_i = {rec["i"]: rec for rec in run.rendered}
    for r, row, ledger in zip(m.rounds, pub["rounds"], batch["rounds"]):
        hr = h.rounds[r.id]
        assert (hr.kind, hr.of, hr.anchor_clip, hr.category) == ("none", None, None, "gate1")   # no hidden controls
        (oid,) = r.ids
        info, i = hr.clips[oid], ledger["sample"]
        rec, s, brief = by_i[i], run.scored[i], sample_brief(2026, i)
        assert row["bpm"] == brief.bpm == info["bpm"] and hr.brief_id == info["brief"] == f"s2026-{i:04d}"
        assert (info["energy"], info["mood"], info["tonic"], info["key"]) == (brief.energy, brief.mood, brief.tonic,
                                                                              brief.key_name)
        assert info["sample"] == i and info["text_sha256"] == sha(f"vamp: loop {i}\n")
        assert info["canon_sha256"] == rec["canon_sha256"] == s["canon_sha256"]
        assert info["pcm_sha256"] == rec["audio_sha256"] == sha(f"pcm {i}")
        assert (info["lufs"], info["true_peak_dbtp"]) == (rec["lufs"], rec["true_peak_dbtp"])
        v = verbatim_of(i)
        assert [info[f"verbatim_{x}"] for x in ("vamp", "keys", "A", "B")] == [v["vamp"], v["keys"], v["A"], v["B"]]
        assert info["verbatim_line"] is (v["A"] or v["B"])
        assert (info["copy_jaccard"], info["copy_reference"]) == (s["copy_jaccard"], s["copy_reference"])
        assert (info["run"], info["job"]) == ("junk-13783877", JOB)
        assert info["palette"] == f"lofi_rain_v1@{G.FROZEN_SHA256[:8]}"
        src = run_dir / "clips" / f"{i:04d}.flac"
        assert info["flac_sha256"] == row["clips"][0]["sha256"] == sha256_file(folder / f"{oid}.flac")
        assert info["flac_sha256"] == sha256_file(src)
        side = json.loads(src.with_suffix(".manifest.json").read_text())
        assert info["clip"] == clip_id(side) == ledger["clip"]["real"]
        assert ledger["clip"] == {"id": oid, "real": info["clip"], "pcm_sha256": info["pcm_sha256"],
                                  "flac_sha256": info["flac_sha256"], "verbatim_line": info["verbatim_line"],
                                  "manifest": True}
    assert (batch["sampled"], batch["recorded"], batch["rendered"]) == (50, 8, 8)       # the step ended after 8
    assert batch["left_out"] == [{"i": i, "why": "not recorded: the render step ended before it"}
                                 for i in made[1]["sample"][8:]]
    assert (batch["batch"], batch["dir"]) == (NAME, f"{DATE}/{NAME}")
    assert (batch["mode"], batch["category"]) == ("triage", "gate1")
    assert batch["without_manifest"] == [] and batch["same_real_clip"] == []
    assert batch["est_min"] == pub["est_min"] == round(sum(10 * 240 / r["bpm"] + 8 for r in pub["rounds"]) / 60, 1)
    assert batch["push"].startswith(f"Needs ear check: {NAME}. 8 loops from the Phase-1 gate run (job {JOB})")
    draw = json.loads((run_dir / "summary" / "draw.json").read_text())
    assert draw["seed"] == f"{SEED:016x}" and draw["order"] == [f"{x['sample']:04d}" for x in batch["rounds"]]
    for doc in (pub, hid):                                        # the seed is only in draw.json
        assert f"{SEED:016x}" not in json.dumps(doc)
    assert sorted(p.name for p in folder.iterdir() if not p.name.endswith(".flac")) == [HIDDEN_NAME, PUBLIC_NAME]


def test_the_public_manifest_holds_the_scene_and_bpm_and_nothing_hidden(run_dir, tmp_path, fast):
    folder = build(run_dir, tmp_path, measure=fast)
    text = (folder / PUBLIC_NAME).read_text()
    hid = json.loads((folder / HIDDEN_NAME).read_text())
    for hr in hid["rounds"].values():
        for info in hr["clips"].values():
            for key in ("clip", "brief", "text_sha256", "canon_sha256", "pcm_sha256", "energy", "mood", "key",
                        "copy_reference", "run", "palette"):
                assert str(info[key]) not in text, key
    for word in ("melancholy", "wistful", "hopeful", "lofi_rain", "junk", "verbatim", "sample"):
        assert word not in text


def test_ids_never_hold_a_stretch_of_a_hidden_digest(run_dir, tmp_path, fast):
    folder = build(run_dir, tmp_path, measure=fast)
    m, h = load_batch(folder)
    ids = [r.id for r in m.rounds] + [c.id for r in m.rounds for c in r.clips]
    assert all(re.fullmatch(r"r[0-9a-f]{8}", r.id) and re.fullmatch(r"k[0-9a-f]{10}", r.ids[0]) for r in m.rounds)
    stretches = set()
    for hr in h.rounds.values():
        for info in hr.clips.values():
            for v in info.values():
                for digest in HEX64.findall(str(v)) + ([str(v)[1:]] if str(v).startswith("c") else []):
                    stretches |= {digest[k:k + 10] for k in range(len(digest) - 9)}
    assert stretches and not any(s in x for s in stretches for x in ids)
    infos = {k: info for hr in h.rounds.values() for k, info in hr.clips.items()}
    avoid = GB._avoid(infos)
    for info in infos.values():                    # drafts.batch's rule, and the other digests the batch keeps
        for key in ("pcm_sha256", "text_sha256", "canon_sha256", "flac_sha256"):
            assert info[key][:10] in avoid and info[key][-10:] in avoid
        assert info["clip"][1:11] in avoid and {info["mood"], "junk"} <= avoid


def test_one_seed_gives_one_batch_and_another_seed_another(made, tmp_path, fast):
    a = build(copy_run(made[0], tmp_path / "a" / "junk-13783877"), tmp_path / "a", measure=fast)
    b = build(copy_run(made[0], tmp_path / "b" / "junk-13783877"), tmp_path / "b", measure=fast)
    c = build(copy_run(made[0], tmp_path / "c" / "junk-13783877"), tmp_path / "c", measure=fast, seed=SEED ^ 1)
    for name in (PUBLIC_NAME, HIDDEN_NAME):
        assert (a / name).read_bytes() == (b / name).read_bytes()
        assert (a / name).read_bytes() != (c / name).read_bytes()


# --------------------------------------------------------------------------------------------- the order
CELLS = [(e, m) for e in ("low", "mid", "high") for m in ("melancholy", "wistful", "hopeful")]


def test_the_order_puts_no_two_consecutive_rounds_in_one_cell():
    cells = {f"{i:04d}": CELLS[i % 9] for i in range(50)}
    for seed in (0, 1, SEED, (1 << 64) - 1):
        d = GB.draw_order(cells, seed)
        assert sorted(d["order"]) == sorted(cells) and d["order_constraint_met"] and d["same_cell_neighbours"] == 0
        assert all(cells[a] != cells[b] for a, b in zip(d["order"], d["order"][1:]))
        assert d["seed"] == f"{seed:016x}" and d["cells"] == {k: list(cells[k]) for k in d["order"]}
    assert len({tuple(GB.draw_order(cells, s)["order"]) for s in range(20)}) == 20


def test_when_no_order_keeps_the_cells_apart_the_draw_says_so():
    cells = {"0001": CELLS[0], "0002": CELLS[0], "0003": CELLS[0], "0004": CELLS[1]}
    d = GB.draw_order(cells, 7)                                     # three of one cell in four rounds: one pair at best
    assert d["order_constraint_met"] is False and d["same_cell_neighbours"] == 1
    assert [cells[k] for k in d["order"]] in ([CELLS[0], CELLS[1], CELLS[0], CELLS[0]],
                                              [CELLS[0], CELLS[0], CELLS[1], CELLS[0]])


def test_the_built_batch_keeps_its_cells_apart(run_dir, tmp_path, fast):
    build(run_dir, tmp_path, measure=fast)
    batch = json.loads((run_dir / "summary" / "batch.json").read_text())
    cells = [(sample_brief(2026, x["sample"]).energy, sample_brief(2026, x["sample"]).mood) for x in batch["rounds"]]
    assert batch["order_constraint_met"] and all(a != b for a, b in zip(cells, cells[1:]))


# --------------------------------------------------------------------------------------------- left out
def test_clips_that_did_not_render_are_left_out_and_named(run_dir, tmp_path, fast):
    def change(d):
        d["clips"][2] = {"i": d["clips"][2]["i"], "ok": False, "gate": "G7", "detail": "true peak -0.80 dBTP"}
        d["clips"][4] = {"i": d["clips"][4]["i"], "ok": False, "gate": None, "error": "FluidSynthError: killed"}
        del d["clips"][6:]                                         # the render step ended two clips earlier
    edit_clips(run_dir, change)
    sample = G.load_run(run_dir).sample
    build(run_dir, tmp_path, measure=fast)
    batch = json.loads((run_dir / "summary" / "batch.json").read_text())
    assert (batch["sampled"], batch["recorded"], batch["rendered"]) == (50, 6, 4)
    later = [{"i": i, "why": "not recorded: the render step ended before it"} for i in sample[6:]]
    assert batch["left_out"] == [{"i": sample[2], "why": "gate G7: true peak -0.80 dBTP"},
                                 {"i": sample[4], "why": "error: FluidSynthError: killed"}, *later]


def test_without_the_render_manifests_the_real_clip_ids_are_the_same(run_dir, tmp_path, made, fast):
    with_ = build(copy_run(made[0], tmp_path / "w" / "junk-13783877"), tmp_path / "w", measure=fast)
    for p in (run_dir / "clips").glob("*.manifest.json"):
        p.unlink()
    without = build(run_dir, tmp_path, measure=fast)
    reals = [sorted(i["clip"] for hr in json.loads((f / HIDDEN_NAME).read_text())["rounds"].values()
                    for i in hr["clips"].values()) for f in (with_, without)]
    assert reals[0] == reals[1]
    batch = json.loads((run_dir / "summary" / "batch.json").read_text())
    assert sorted(batch["without_manifest"]) == sorted(made[1]["recorded"])
    assert all(x["clip"]["manifest"] is False for x in batch["rounds"])


# --------------------------------------------------------------------------------------------- the FLACs
def _first(run_dir: Path) -> Path:
    i = json.loads((run_dir / "summary" / "clips.json").read_text())["clips"][0]["i"]
    return run_dir / "clips" / f"{i:04d}.flac"


def _refused(run_dir, tmp_path, message, **kw):
    with pytest.raises(BuildError) as err:
        build(run_dir, tmp_path, **kw)
    assert any(message in e for e in err.value.errors), err.value.errors
    nothing_written(run_dir, tmp_path)
    return err.value.errors


def test_a_missing_flac_refuses(run_dir, tmp_path, fast):
    _first(run_dir).unlink()
    _refused(run_dir, tmp_path, "is missing", measure=fast)


def test_a_flac_that_is_a_link_refuses(run_dir, tmp_path, fast):
    flac = _first(run_dir)
    real = flac.with_name("elsewhere.flac")
    flac.rename(real)
    flac.symlink_to(real)
    _refused(run_dir, tmp_path, "is not a plain file", measure=fast)


def test_a_flac_of_other_bytes_refuses(run_dir, tmp_path, fast):
    with _first(run_dir).open("ab") as f:
        f.write(b"\0")
    _refused(run_dir, tmp_path, "bytes; clips.json records", measure=fast)


def test_a_flac_of_another_length_refuses(run_dir, tmp_path, fast):
    import soundfile as sf
    flac = _first(run_dir)
    x, sr = sf.read(str(flac), dtype="int16", always_2d=True)
    sf.write(str(flac), x[:-64], sr, format="FLAC", subtype="PCM_16")
    errs = _refused(run_dir, tmp_path, "frames, 2 channels at 44100 Hz; clips.json records", measure=fast)
    assert any("bytes; clips.json records" in e for e in errs)


def test_two_flacs_swapped_refuse(run_dir, tmp_path):
    """Two clips' files under each other's names: the bytes, the frames or the loudness give each away."""
    recs = json.loads((run_dir / "summary" / "clips.json").read_text())["clips"]
    a, b = (run_dir / "clips" / f"{recs[k]['i']:04d}.flac" for k in (0, 1))
    tmp = a.with_name("swap.tmp")
    a.rename(tmp)
    b.rename(a)
    tmp.rename(b)
    errs = _refused(run_dir, tmp_path, f"sample {recs[0]['i']:04d}")
    assert any(f"sample {recs[1]['i']:04d}" in e for e in errs)


def test_the_same_bytes_count_with_other_samples_refuses_by_their_loudness(run_dir, tmp_path):
    """A copy whose size and length are the record's but whose samples are not: only the measure can tell, and does
    (the manifest is removed, so the measure is the only check on the samples)."""
    flac = _first(run_dir)
    flac.with_suffix(".manifest.json").unlink()
    clips = run_dir / "summary" / "clips.json"
    edit_json(clips, lambda d: d["clips"][0].update(lufs=round(d["clips"][0]["lufs"] + 0.01, 3)))
    _refused(run_dir, tmp_path, "measures lufs")
    edit_json(clips, lambda d: d["clips"][0].update(lufs=round(d["clips"][0]["lufs"] - 0.01, 3), true_peak_dbtp=-1.0))
    _refused(run_dir, tmp_path, "measures true_peak_dbtp")


def test_a_flac_sha256_clips_json_records_must_match(run_dir, tmp_path, fast):
    ok = copy_run(run_dir, tmp_path / "ok" / "junk-13783877")          # a recorded sha256 that matches is taken
    flac = _first(ok)
    edit_json(ok / "summary" / "clips.json", lambda d: d["clips"][0].update(flac_sha256=sha256_file(flac)))
    build(ok, tmp_path / "ok", measure=fast)
    edit_json(run_dir / "summary" / "clips.json", lambda d: d["clips"][0].update(flac_sha256="0" * 64))
    _refused(run_dir, tmp_path, "sha256 is not the", measure=fast)


def test_a_flac_that_does_not_decode_refuses(run_dir, tmp_path):
    def broken(path):
        raise RuntimeError("Error : unexpected end of file")
    _refused(run_dir, tmp_path, "does not decode (RuntimeError: Error : unexpected end of file)", measure=broken)


def test_measure_flac_reads_what_render_formats_wrote(tmp_path):
    """The copy's samples at render.formats' full scale (32767) measure as the float render does, within the
    tolerance: a dithered 16-bit FLAC of a float signal peaking near a render's (-1.5 dBTP), as junk.render_clip
    writes it. Read at libsndfile's own scale (32768) the loudness would sit 0.00027 LU low on every clip."""
    from lofi_rl.render.formats import write_flac
    from lofi_rl.render.loudness import lufs
    from lofi_rl.render.pipeline import loop_true_peak_dbtp
    n = loop_samples(82)
    t = np.arange(n) / 44100.0
    x = np.stack([0.84 * np.sin(2 * np.pi * 220 * t), 0.6 * np.sin(2 * np.pi * 331 * t + 1)]).astype(np.float32)
    path = write_flac(tmp_path / "x.flac", x)
    got = GB.measure_flac(path)
    assert abs(got[0] - lufs(x, 44100)) < 1e-5 and abs(got[1] - loop_true_peak_dbtp(x)) < GB.MEASURE_TOL / 2


# --------------------------------------------------------------------------------------------- the records
def test_a_manifest_that_disagrees_with_clips_json_refuses(run_dir, tmp_path, fast):
    side = _first(run_dir).with_suffix(".manifest.json")
    edit_json(side, lambda d: d.update(audio_sha256="f" * 64))
    _refused(run_dir, tmp_path, "manifest.json audio_sha256 is 'ffff", measure=fast)
    edit_json(side, lambda d: d.update(audio_sha256=None, gates={"G1-G6": "pass", "G7": "fail"}))
    errs = _refused(run_dir, tmp_path, "manifest.json gates is", measure=fast)
    assert any("audio_sha256 is None" in e for e in errs)
    side.write_text("{")
    _refused(run_dir, tmp_path, "manifest.json is not readable JSON", measure=fast)


@pytest.mark.parametrize("field, value, message", [
    ("clip", "clips/9999.flac", "not clips/"),
    ("n", 12345, "clips.json records n 12345"),
    ("audio_sha256", "abc", "audio_sha256 'abc' is not a sha256"),
    ("canon_sha256", "e" * 64, "is not the render's 'eeee"),    # score's canonical loop is not the render's
    ("brief", {"energy": "low", "mood": "hopeful", "bpm": 80, "tonic": 0, "scene": "city at night"},
     "is not brief"),
])
def test_a_clips_json_record_off_the_run_refuses(run_dir, tmp_path, fast, field, value, message):
    edit_json(run_dir / "summary" / "clips.json", lambda d: d["clips"][0].update({field: value}))
    _refused(run_dir, tmp_path, message, measure=fast)


def test_every_problem_is_listed_before_anything_is_written(run_dir, tmp_path, fast):
    recs = json.loads((run_dir / "summary" / "clips.json").read_text())["clips"]
    for rec in recs[:3]:
        (run_dir / "clips" / f"{rec['i']:04d}.flac").unlink()
    with pytest.raises(BuildError) as err:
        build(run_dir, tmp_path, measure=fast)
    assert len(err.value.errors) == 3 and "3 problem(s), nothing written" in str(err.value)
    nothing_written(run_dir, tmp_path)


# --------------------------------------------------------------------------------------------- never overwritten
def test_an_existing_batch_is_never_overwritten(run_dir, tmp_path, fast):
    (tmp_path / "inbox" / DATE / NAME).mkdir(parents=True)
    with pytest.raises(BuildError, match="exists; a built batch is never overwritten"):
        build(run_dir, tmp_path, measure=fast)
    assert not any((tmp_path / "inbox" / DATE / NAME).iterdir())


def test_a_runs_batch_record_is_never_overwritten(run_dir, tmp_path, fast):
    build(run_dir, tmp_path, measure=fast)
    before = {p: (run_dir / "summary" / p).read_bytes() for p in ("draw.json", "batch.json")}
    with pytest.raises(BuildError) as err:
        build(run_dir, tmp_path, measure=fast, name="b1107-gate1-2")
    errs = err.value.errors
    assert sum("this run's ear check was built already" in e for e in errs) == 2
    assert any(f"(batch {NAME})" in e for e in errs)
    assert any(f"holds this run's ear check already ({DATE}/{NAME})" in e for e in errs)
    assert not any("move it aside" in e for e in errs)                  # nothing here suggests a second ear check
    assert {p: (run_dir / "summary" / p).read_bytes() for p in before} == before
    assert not (tmp_path / "inbox" / DATE / "b1107-gate1-2").exists()


def test_a_second_ear_check_of_one_run_is_never_built(run_dir, tmp_path, fast):
    """Review F4's case: the run's draw.json and batch.json moved aside, its ear check is still in the inbox (under
    any name or date: found by its hidden manifest), so a second batch refuses and names it. Another run's batch in
    that inbox is not this run's."""
    build(run_dir, tmp_path, measure=fast, name="b1107-mine")
    for f in ("draw.json", "batch.json"):
        (run_dir / "summary" / f).rename(run_dir / "summary" / f"{f}.aside")
    with pytest.raises(BuildError) as err:
        build(run_dir, tmp_path, measure=fast, name="b1108-gate1-2", date="2026-11-08")
    assert err.value.errors == [f"{tmp_path / 'inbox'} holds this run's ear check already ({DATE}/b1107-mine): the "
                                "gate counts one ear check per run, so a second one is Shresth's call"]
    assert not (tmp_path / "inbox" / "2026-11-08").exists() and not (run_dir / "summary" / "batch.json").exists()
    other = copy_run(run_dir, tmp_path / "o" / "junk-13783878")
    edit_json(other / "summary" / "junk.json", lambda d: d["generate"]["host"].update(job="13783878"))
    assert GB.run_batches(tmp_path / "inbox", G.load_run(other)) == []
    build(other, tmp_path, measure=fast, name="b1108-gate1-2", date="2026-11-08")
    assert GB.run_batches(tmp_path / "inbox", G.load_run(other)) == ["2026-11-08/b1108-gate1-2"]
    nojob = copy_run(run_dir, tmp_path / "n" / "junk-13783877")         # a run with no job id: matched by its name
    edit_json(nojob / "summary" / "junk.json", lambda d: d["generate"]["host"].pop("job"))
    assert GB.run_batches(tmp_path / "inbox", G.load_run(nojob)) == [f"{DATE}/b1107-mine"]


def test_a_bad_name_or_clips_folder_refuses(run_dir, tmp_path, fast):
    with pytest.raises(BuildError, match="must match"):
        build(run_dir, tmp_path, measure=fast, name="../escape")
    with pytest.raises(BuildError, match="is not a folder"):
        GB.build(G.load_run(run_dir), tmp_path / "nowhere", name=NAME, date=DATE, out_root=tmp_path / "inbox",
                 seed=SEED, measure=fast)
    nothing_written(run_dir, tmp_path)


def test_no_clip_rendered_refuses(run_dir, tmp_path, fast):
    edit_clips(run_dir,
               lambda d: d.update(clips=[{"i": r["i"], "ok": False, "gate": "G7", "detail": "x"} for r in d["clips"]]))
    _refused(run_dir, tmp_path, "no clip rendered", measure=fast)


def test_a_failed_page_check_moves_nothing_into_place(run_dir, tmp_path, fast, monkeypatch):
    monkeypatch.setattr(GB, "check_folders", lambda folders: [(1, "manifest.json: something")])
    _refused(run_dir, tmp_path, "ui.manifest check failed: manifest.json: something", measure=fast)


# --------------------------------------------------------------------------------------------- the command
def test_the_command_builds_into_the_inbox_and_names_the_next_batch(made, tmp_path, capsys):
    inbox = tmp_path / "inbox"
    one = copy_run(made[0], tmp_path / "one" / "junk-13783877")
    assert cli.main(["batch", str(one), "--clips", str(one / "clips"), "--inbox", str(inbox), "--date", DATE,
                     "--seed", f"{SEED:016x}"]) == 0
    out = capsys.readouterr().out
    assert f"batch {NAME}: 8 triage rounds of 50 sampled clips" in out and f"push: Needs ear check: {NAME}." in out
    assert out.count("not recorded: the render step ended before it") == 42
    two = copy_run(made[0], tmp_path / "two" / "junk-13783878")       # another run: one ear check each
    edit_json(two / "summary" / "junk.json", lambda d: d["generate"]["host"].update(job="13783878"))
    assert cli.main(["batch", str(two), "--clips", str(two / "clips"), "--inbox", str(inbox),
                     "--date", "2026-11-09"]) == 0
    again = copy_run(made[0], tmp_path / "again" / "junk-13783877")    # the first run again, its records elsewhere
    assert cli.main(["batch", str(again), "--clips", str(again / "clips"), "--inbox", str(inbox),
                     "--date", "2026-11-09"]) == 2
    assert f"holds this run's ear check already ({DATE}/{NAME})" in capsys.readouterr().err
    assert (inbox / "2026-11-09" / "b1109-gate1-2").is_dir() and (inbox / DATE / NAME).is_dir()
    assert GB.batch_name(date(2026, 11, 9), inbox) == "b1109-gate1-3"
    assert GB.batch_name(date(2026, 11, 9), inbox, "b1109-mine") == "b1109-mine"
    seeds = {json.loads((r / "summary" / "draw.json").read_text())["seed"] for r in (one, two)}
    assert len(seeds) == 2 and f"{SEED:016x}" in seeds             # a fresh seed when none is given


def test_the_command_refuses_a_run_that_does_not_pass_check(made, tmp_path, capsys):
    bad = copy_run(made[0], tmp_path / "bad" / "junk-13783877")
    edit_json(bad / "summary" / "junk.json", lambda d: d["prompt"].update(rows_matching=499))
    assert cli.main(["batch", str(bad), "--clips", str(bad / "clips"), "--inbox", str(tmp_path / "inbox")]) == 3
    assert "does not pass check (invalid)" in capsys.readouterr().err
    assert not (tmp_path / "inbox").exists() and not (bad / "summary" / "batch.json").exists()
    under = tmp_path / "under" / "junk-13783878"
    make_run(under, copies=201, render_k=2, flacs=False)
    assert cli.main(["batch", str(under), "--clips", str(under / "clips"), "--inbox", str(tmp_path / "inbox")]) == 3
    assert "(valid, under the bar)" in capsys.readouterr().err


def test_the_command_says_bad_input_with_exit_2(made, tmp_path, capsys):
    r = copy_run(made[0], tmp_path / "r" / "junk-13783877")
    (r / "summary" / "clips.json").unlink()
    assert cli.main(["batch", str(r), "--clips", str(r / "clips"), "--inbox", str(tmp_path / "inbox")]) == 2
    assert "clips.json is missing" in capsys.readouterr().err
    r2 = copy_run(made[0], tmp_path / "r2" / "junk-13783877")
    assert cli.main(["batch", str(r2), "--clips", str(r2 / "clips"), "--inbox", str(tmp_path / "inbox"),
                     "--date", "2026-13-01"]) == 2
    assert cli.main(["batch", str(r2), "--clips", str(r2 / "clips"), "--inbox", str(tmp_path / "inbox"),
                     "--seed", "zz"]) == 2
    (_first(r2)).unlink()
    assert cli.main(["batch", str(r2), "--clips", str(r2 / "clips"), "--inbox", str(tmp_path / "inbox")]) == 2
    assert "problem(s), nothing written" in capsys.readouterr().err
    assert not (tmp_path / "inbox").exists() or not any((tmp_path / "inbox").rglob("*.flac"))


@pytest.mark.parametrize("off, message", [
    ({"k": 100}, "clips.json: `k` is 100; the ear check is junk's sample of 50"),
    ({"draw_seed": 7}, "clips.json: `seed` is 7; the ear check's sample is drawn with the run's brief seed 2026"),
])
def test_a_sample_that_is_not_the_registered_draw_is_never_batched(tmp_path, capsys, off, message):
    """A hand re-render of another k or seed (its clips.json and junk.json one render step's) is another draw: the
    batch refuses it as bad input, so the ear check never holds more than the 50 clips, or other clips."""
    run = tmp_path / "junk-13783877"
    make_run(run, render_k=3, measure=False, **off)
    inbox = tmp_path / "inbox"
    assert cli.main(["batch", str(run), "--clips", str(run / "clips"), "--inbox", str(inbox), "--date", DATE]) == 2
    err = capsys.readouterr().err
    assert "error: bad input, nothing written" in err and message in err
    assert not inbox.exists() and not (run / "summary" / "batch.json").exists()


# --------------------------------------------------------------------------------------------- junk's own render step
needs_audio = pytest.mark.skipif(
    shutil.which("fluidsynth") is None or any(importlib.util.find_spec(m) is None for m in ("pedalboard", "soundfile")),
    reason="renders for real: needs fluidsynth on PATH, pedalboard and soundfile")


def _frozen_sources_missing() -> list[str]:
    frozen = load_palette(FROZEN, verify=False)
    missing = []
    for src in sorted(frozen.sources):
        try:
            path = frozen.source_path(src)
        except Exception as err:  # noqa: BLE001: an unset assets variable, say
            return [f"{src}: {err}"]
        if not path.is_file():
            missing.append(str(path))
    return missing


@needs_audio
def test_a_batch_from_junks_own_render_step_on_the_frozen_palette(tmp_path):
    """Two committed drafts placed at measurement-stream indices of their own cells, rendered by `junk render` with its
    defaults (the frozen palette, two at a time): the batch takes its FLACs and manifests as they come, measures them
    within the tolerance, and gives each clip the real clip id of its render manifest. Skipped where the frozen
    palette's fetch-only sources are not here."""
    missing = _frozen_sources_missing()
    if missing:
        pytest.skip(f"the frozen palette's fetch-only sources are not here ({len(missing)}, e.g. {missing[0]})")
    run = tmp_path / "junk-13783877"
    summ = run / "summary"
    summ.mkdir(parents=True)
    picks = {20: "d01_groove", 7: "redo1/d06_chords"}             # sample_brief(2026, i): low/melancholy, mid/hopeful
    rows, scored = [], []
    for i in range(500):
        b = sample_brief(2026, i)
        text = (ROOT / "drafts" / "v1" / f"{picks[i]}.ls").read_text(encoding="utf-8") if i in picks else "x\n"
        rows.append({"i": i, "brief": J.brief_dict(b), "text": text, "finish_reason": "stop", "n_tokens": 250})
        scored.append({"i": i, "ok": i in picks, "gate": None if i in picks else "G1", "near_copy": False,
                       "canon_sha256": sha(f"canon {i}"), "verbatim": verbatim_of(i), "copy_jaccard": 0.5,
                       "copy_reference": "l1_melancholy_low"})
    (summ / "completions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (summ / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scored))
    (summ / "generate.json").write_text(json.dumps({"schema": J.GENERATE_SCHEMA, "briefs": {"seed": 2026}}))
    (summ / "junk.json").write_text(json.dumps(junk_doc(500, 2, 2)))
    assert J.main(["render", "--run", str(run), "--k", "2", "--workers", "4", "--no-verify",
                   "--cache", str(tmp_path / "cache")]) == 0
    clips = json.loads((summ / "clips.json").read_text())
    assert [c["i"] for c in clips["clips"]] == J.sample_indices(picks, 2, 2026) and all(c["ok"] for c in clips["clips"])
    from lofi_rl.dsl import canon_sha, parse
    pal = load_palette(FROZEN, verify=False)
    for i in picks:                                   # scored.jsonl's sha is the render's, as score writes it
        res = parse(rows[i]["text"], vocab=pal.vocab(), finish_reason="stop", mode=sample_brief(2026, i).mode)
        scored[i]["canon_sha256"] = canon_sha(res.ast)
    (summ / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scored))
    folder = GB.build(G.load_run(run), run / "clips", name=NAME, date=DATE, out_root=tmp_path / "inbox", seed=SEED)
    m, h = load_batch(folder)
    assert len(m.rounds) == 2
    for r in m.rounds:
        info = h.rounds[r.id].clips[r.ids[0]]
        side = json.loads((run / "clips" / f"{info['sample']:04d}.manifest.json").read_text())
        assert info["clip"] == clip_id(side) and info["pcm_sha256"] == side["audio_sha256"]
        got = GB.measure_flac(run / "clips" / f"{info['sample']:04d}.flac")
        assert abs(got[0] - side["loudness"]["lufs"]) < GB.MEASURE_TOL / 2
        assert abs(got[1] - side["loudness"]["true_peak_dbtp"]) < GB.MEASURE_TOL / 2
