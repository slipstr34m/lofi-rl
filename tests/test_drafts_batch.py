"""The keep3 batch of the drafts (lofi_rl.drafts.batch; docs/DRAFTS.md §7): the Latin-square tiles, the round order,
the opaque ids and the refusals, on records and FLACs made here (no render, no listening)."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from lofi_rl.compile.timeline import loop_samples
from lofi_rl.drafts import batch as BT
from lofi_rl.drafts.briefs import load as load_briefs
from lofi_rl.drafts.personas import PERSONAS
from lofi_rl.render.formats import write_flac
from lofi_rl.ui.manifest import HIDDEN_NAME, PUBLIC_NAME, load_batch, sha256_file

ROOT = Path(__file__).resolve().parents[1]
BRIEFS = {b: x.as_dict() for b, x in load_briefs(ROOT / "drafts" / "v1" / "briefs.yaml").items()}


def test_ten_rounds_balance_every_persona_over_the_tiles():
    for seed in (0, 1, 2026, 0xFFFFFFFFFFFFFFFF):
        draw = BT.draw_rounds(BRIEFS, seed)
        assert [r["brief"] for r in draw["rounds"]] != [] and sorted(r["brief"] for r in draw["rounds"]) == sorted(BRIEFS)
        counts = BT.tile_counts(draw)
        assert all(sorted(c) == [3, 3, 4] for c in counts.values())
        assert sorted(c.index(4) for c in counts.values()) == [0, 1, 2]       # each tile's 4 is another persona's
        for r in draw["rounds"]:
            assert sorted(r["tiles"]) == sorted(PERSONAS)
            assert r["tiles"] == [draw["symbols"][BT.SQUARE[r["row"]][t]] for t in range(3)]
        cells = [(BRIEFS[r["brief"]]["energy"], BRIEFS[r["brief"]]["mood"]) for r in draw["rounds"]]
        assert all(a != b for a, b in zip(cells, cells[1:]))                  # d10 never follows its own cell
        assert Counter(r["row"] for r in draw["rounds"]).most_common(1)[0][1] == 4


def test_nine_rounds_are_three_full_squares_and_a_redraft_uses_distinct_rows():
    nine = {b: v for b, v in BRIEFS.items() if b != "d10"}
    assert all(c == [3, 3, 3] for c in BT.tile_counts(BT.draw_rounds(nine, 7)).values())
    two = {b: BRIEFS[b] for b in ("d04", "d07")}
    draw = BT.draw_rounds(two, 7)
    assert len({r["row"] for r in draw["rounds"]}) == 2
    assert all(sorted(c) == [0, 1, 1] for c in BT.tile_counts(draw).values())   # distinct rows: two tiles each


def test_one_seed_one_draw_and_seeds_differ():
    assert BT.draw_rounds(BRIEFS, 42) == BT.draw_rounds(BRIEFS, 42)
    draws = {json.dumps(BT.draw_rounds(BRIEFS, s)["rounds"]) for s in range(20)}
    assert len(draws) == 20
    assert len({tuple(BT.draw_rounds(BRIEFS, s)["symbols"]) for s in range(20)}) == 6     # every persona map occurs


def test_ids_avoid_every_hidden_word_and_stretch():
    rng, used = BT.Rng(1, "x"), set()
    avoid = {"0", "1", "2", "3", "4", "5", "6", "7"}                          # leaves only 8-f
    ids = [BT._draw_id(rng, "k", 10, avoid, used) for _ in range(50)]
    assert len(set(ids)) == 50 and all(set(i[1:]) <= set("89abcdef") for i in ids)
    real = "c" + "0123456789abcdef0"
    words = BT._secrets([{"brief_id": "d01", "clips": {"x": {"clip": real, "persona": "groove", "pcm_sha256": "f" * 64}}}])
    assert "0123456789" in words and "groove" in words and "f" * 10 in words and "d01" not in words


# --------------------------------------------------------------------------------------------- build
def fake_records(renders: Path, briefs=("d01", "d05"), *, bpm_of=None) -> dict:
    """One ok record per (brief, persona) with a real one-pass FLAC, as lofi_rl.drafts.render writes them."""
    out = {}
    for i, b in enumerate(briefs):
        brief = BRIEFS[b]
        bpm = bpm_of or brief["bpm"]
        n = loop_samples(bpm)
        for j, p in enumerate(PERSONAS):
            rid = f"{b}_{p}"
            d = renders / rid
            d.mkdir(parents=True)
            x = (0.01 * (i * 3 + j + 1) * np.sin(np.arange(n) / (40.0 + j))).astype(np.float32)
            flac = write_flac(d / "render.flac", np.stack([x, x]))
            out[rid] = {"schema": "lofi-rl/draft-render/1", "render": rid, "status": "ok", "error": None,
                        "file": f"drafts/v1/{rid}.ls", "draft_sha256": f"{i * 3 + j:064x}", "brief_id": b,
                        "brief": {k: brief[k] for k in ("energy", "mood", "bpm", "tonic")} | {"bpm": bpm},
                        "persona": p, "keys": "rhodes", "kit": "jazz",
                        "palette": {"id": "draft_day2_rhodes_jazz", "sha256": "e" * 64,
                                    "label": "draft_day2_rhodes_jazz@eeeeeeee", "file": "x.yaml"},
                        "clip": f"c{i * 3 + j + 1:016x}", "n": n, "sr": 44100, "bpm": bpm, "lufs": -14.02,
                        "true_peak_dbtp": -1.5, "pcm_sha256": f"{i * 3 + j + 7:064x}", "flac_sha256": sha256_file(flac),
                        "gates": {"G1-G6": "pass", "G7": "pass"}}
    return out


def _build(tmp_path, recs, briefs=("d01", "d05"), **kw):
    args = dict(briefs={b: BRIEFS[b] for b in briefs}, name="b1105-keep3", date="2026-11-05",
                out_root=tmp_path / "inbox", seed=99, summary_dir=tmp_path / "summary")
    args.update(kw)
    return BT.build(recs, tmp_path / "renders", **args)


def test_a_batch_passes_the_rating_pages_own_check(tmp_path):
    recs = fake_records(tmp_path / "renders")
    folder = _build(tmp_path, recs)
    assert folder == tmp_path / "inbox" / "2026-11-05" / "b1105-keep3"
    assert sorted(p.name for p in folder.iterdir() if not p.name.endswith(".flac")) == [HIDDEN_NAME, PUBLIC_NAME]
    m, h = load_batch(folder)
    assert m.mode == "keep3" and len(m.rounds) == 2 and all(len(r.clips) == 3 for r in m.rounds)
    pub = json.loads((folder / PUBLIC_NAME).read_text())
    assert pub["scene"] == "city at night" and pub["note"] == BT.NOTE and "created" not in pub
    assert {r["bpm"] for r in pub["rounds"]} == {BRIEFS["d01"]["bpm"], BRIEFS["d05"]["bpm"]}
    batch = json.loads((tmp_path / "summary" / "batch.json").read_text())
    for r, hr in zip(batch["rounds"], (h.rounds[x.id] for x in m.rounds)):
        assert hr.brief_id == r["brief"] and hr.kind == "none"
        for c in r["clips"]:
            info = hr.clips[c["id"]]
            assert (info["persona"], info["draft"], info["clip"], info["brief"]) == (
                c["persona"], c["draft"], c["clip"], r["brief"])
            assert info["draft"] == f"drafts/v1/{r['brief']}_{c['persona']}.ls"
    text = (folder / PUBLIC_NAME).read_text()
    assert not any(w in text for w in ("groove", "horn", "chords", "drafts/", "d01", "d05", "rhodes"))
    draw = json.loads((tmp_path / "summary" / "draw.json").read_text())
    assert draw["seed"] == f"{99:016x}" and [r["brief"] for r in draw["rounds"]] == [r["brief"] for r in batch["rounds"]]


def test_one_seed_gives_byte_identical_batches(tmp_path):
    recs = fake_records(tmp_path / "renders")
    a = _build(tmp_path, recs, out_root=tmp_path / "a", summary_dir=tmp_path / "sa")
    b = _build(tmp_path, recs, out_root=tmp_path / "b", summary_dir=tmp_path / "sb")
    c = _build(tmp_path, recs, out_root=tmp_path / "c", summary_dir=tmp_path / "sc", seed=100)
    files = sorted(p.name for p in a.iterdir())
    assert files == sorted(p.name for p in b.iterdir())
    assert all((a / f).read_bytes() == (b / f).read_bytes() for f in files)
    assert (a / PUBLIC_NAME).read_text() != (c / PUBLIC_NAME).read_text()


def test_the_build_refuses_and_writes_nothing(tmp_path):
    recs = fake_records(tmp_path / "renders")
    broken = {**recs, "d05_horn": {**recs["d05_horn"], "status": "level", "error": "-13.7 LUFS"}}
    with pytest.raises(BT.BuildError, match="level"):
        _build(tmp_path, broken)
    with pytest.raises(BT.BuildError, match="a round needs one of each"):
        _build(tmp_path, {k: v for k, v in recs.items() if k != "d01_horn"})
    with pytest.raises(BT.BuildError, match="does not hold"):
        _build(tmp_path, recs, briefs=("d05",))
    twin = {**recs, "d05_horn": {**recs["d05_horn"], "clip": recs["d05_groove"]["clip"]}}
    with pytest.raises(BT.BuildError, match="one real clip"):
        _build(tmp_path, twin)
    short = {**recs, "d05_horn": {**recs["d05_horn"], "bpm": 70}}
    with pytest.raises(BT.BuildError, match="one pass"):
        _build(tmp_path, short)
    with pytest.raises(BT.BuildError, match="must match"):
        _build(tmp_path, recs, name="keep 3")
    assert not (tmp_path / "inbox").exists() or not any((tmp_path / "inbox").rglob("*.flac"))
    _build(tmp_path, recs)
    with pytest.raises(BT.BuildError, match="never overwritten"):
        _build(tmp_path, recs)


def test_the_estimate_follows_the_plan():
    assert BT.round_seconds(3, 80) == pytest.approx(0.75 * 3 * 24 + 8)
