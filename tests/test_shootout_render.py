"""The render driver's per-clip level record (docs/SHOOTOUTS.md §3 and CD-52; §7.7), with no audio.

Each clip of an instrument stage plays its own palette, whose varied bus carries the clip's own fader; the
generator's index.json beside the palettes gives that palette's `level` (the level tool's L before the fader, the
fader, and the loudness it measured after it). A render copies the level into its record and must play the bus the
tool measured: its pre-fader loudness within AGREE_LU of L, at the same fader, with the measured loudness within
the clip tolerance of S* + T. A render that does not gets status `level`, and the batch builder refuses it. The
renders themselves are covered by tests/test_shootout_e2e.py and the `-m cluster` level test."""
import json

import pytest

from lofi_rl.shootout.render import AGREE_LU, RECORD_KEYS, palette_levels, with_level

LEVEL = {"bus": "keys", "variant": "jrhodes3d", "case": "l5_wistful_mid", "L": -31.25, "F": 3.71, "lufs": -27.541,
         "lu": -6.001, "target_lufs": -27.54, "off_lu": -0.001, "tol_lu": 0.1}


def record(lufs=-31.25, fader=3.71) -> dict:
    rec = {k: None for k in RECORD_KEYS}
    rec.update(status="ok", buses={"keys": {"lufs": lufs, "fader_db": fader, "trim_db": -4.5,
                                            "post_lufs": None if lufs is None else lufs + fader}})
    return rec


def test_a_record_carries_its_clips_level_when_the_render_plays_the_bus_the_tool_measured():
    out = with_level(record(), LEVEL)
    assert out["status"] == "ok" and out["error"] is None and "level" in RECORD_KEYS
    assert out["level"] == {**LEVEL, "render_lufs": -31.25, "agree_lu": 0.0, "ok": True}
    near = with_level(record(lufs=-31.25 + AGREE_LU / 2), LEVEL)
    assert near["status"] == "ok" and near["level"]["agree_lu"] == pytest.approx(AGREE_LU / 2)


@pytest.mark.parametrize("rec,level,said", [
    (record(lufs=-31.25 + 2 * AGREE_LU), LEVEL, "before its fader"),       # not the bus the tool measured
    (record(fader=3.72), LEVEL, "fader"),                                   # not the clip's fader
    (record(lufs=None), LEVEL, "silent"),
    (record(), dict(LEVEL, off_lu=0.13), "0.1 LU"),                          # matched, but off its target
])
def test_a_render_that_does_not_play_its_clips_level_gets_status_level(rec, level, said):
    out = with_level(rec, level)
    assert out["status"] == "level" and said in out["error"] and out["level"]["ok"] is False
    assert {k: out["level"][k] for k in level} == level


DAY2 = {"case": "l7_hopeful_low.drift", "buses": {   # CD-53: a vamp clip matches its keys and its bass
    "keys": {"instrument": "gu_tine_ep", "L": -31.25, "F": 3.71, "lufs": -27.541, "lu": -6.001, "target_lufs": -27.54,
             "off_lu": -0.001, "tol_lu": 0.1},
    "bass": {"instrument": "bass_gu_acoustic", "L": -40.0, "F": 15.46, "lufs": -24.54, "lu": -3.0,
             "target_lufs": -24.54, "off_lu": 0.0, "tol_lu": 0.1}}}


def record2(keys=(-31.25, 3.71), bass=(-40.0, 15.46)) -> dict:
    rec = record(*keys)
    rec["buses"]["bass"] = {"lufs": bass[0], "fader_db": bass[1], "trim_db": 0.0,
                            "post_lufs": None if bass[0] is None else bass[0] + bass[1]}
    return rec


def test_a_day2_clip_must_play_every_bus_it_matches_at_its_own_fader():
    # CD-53: from Day 2 on a clip's level holds one entry per bus its round's varied thing moves; the render must
    # play each of them as the level tool measured it, or the record gets status `level`.
    out = with_level(record2(), DAY2)
    assert out["status"] == "ok" and out["level"]["ok"] is True and out["level"]["case"] == DAY2["case"]
    assert {b: (v["render_lufs"], v["agree_lu"]) for b, v in out["level"]["buses"].items()} == \
        {"keys": (-31.25, 0.0), "bass": (-40.0, 0.0)}
    for rec, said in ((record2(bass=(-40.0 + 2 * AGREE_LU, 15.46)), "bass bus reads"),
                      (record2(bass=(-40.0, 15.45)), "bass fader"), (record2(keys=(None, 3.71)), "keys bus is silent")):
        bad = with_level(rec, DAY2)
        assert bad["status"] == "level" and said in bad["error"] and bad["level"]["ok"] is False
    off = with_level(record2(), {**DAY2, "buses": {**DAY2["buses"], "bass": {**DAY2["buses"]["bass"], "off_lu": 0.2}}})
    assert off["status"] == "level" and "bass bus +0.200 LU from S* + T" in off["error"]


def test_each_palette_takes_its_level_from_the_generators_index_beside_it(tmp_path):
    d, stand = tmp_path / "palettes", tmp_path / "stand-ins"
    d.mkdir()
    stand.mkdir()
    (d / "index.json").write_text(json.dumps({"schema": "lofi-rl/shootout-palettes/1", "palettes": [
        {"file": "k1_jrhodes3d_l5.yaml", "id": "k1_jrhodes3d_l5", "base": "k1_jrhodes3d", "level": LEVEL},
        {"file": "k1_jrhodes3d.yaml", "id": "k1_jrhodes3d"}]}))
    files = [d / "k1_jrhodes3d_l5.yaml", d / "k1_jrhodes3d.yaml", stand / "t_sax_plain.yaml"]
    assert palette_levels(files) == {files[0]: LEVEL, files[1]: None, files[2]: None}
    (d / "index.json").write_text("{not json")
    with pytest.raises(ValueError, match="index.json"):
        palette_levels(files[:1])
