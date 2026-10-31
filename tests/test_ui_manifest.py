"""Batch manifests for the rating UI: the demo builder, schema validation and the leak check
(docs/RATING.md; src/lofi_rl/ui/manifest.py). All audio is synthetic and lives in temp dirs."""
import json
import os
import shutil
import tempfile
import wave
from pathlib import Path

import pytest

from lofi_rl.ui.manifest import (HIDDEN_NAME, MODES, PUBLIC_NAME, ManifestError, audio_info, build_demo,
                                 default_plays, load_batch, parse_hidden, parse_public)

FAST = dict(bpm=240.0, sr=8000)          # 8 bars of 1 s: small clips, quick tests


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    root = tmp_path_factory.mktemp("inbox") / "2026-11-03"
    return {m: build_demo(root / f"demo-{m}", m, **FAST) for m in MODES}


def _copy(src, tmp_path, name=None):
    """A fresh copy of a demo batch (each in its own parent, so one test can make several)."""
    dst = Path(tempfile.mkdtemp(dir=tmp_path)) / (name or src.name)
    shutil.copytree(src, dst)
    return dst


def _edit(path, fn):
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def _errors(batch_dir):
    with pytest.raises(ManifestError) as e:
        load_batch(batch_dir)
    return "\n".join(e.value.errors)


@pytest.mark.parametrize("mode", sorted(MODES))
def test_demo_batches_validate(demos, mode):
    m, h = load_batch(demos[mode])
    assert m.mode == mode and m.batch == f"demo-{mode}" and m.bars == 8
    lo, hi = MODES[mode]
    assert all(lo <= len(r.clips) <= hi for r in m.rounds)
    assert all(c.frames == 8 * 8000 and c.sr == 8000 for r in m.rounds for c in r.clips)
    kinds = [h.rounds[r.id].kind for r in m.rounds]
    assert kinds[1] == "anchor" and kinds[-1] == "repeat" and h.rounds[m.rounds[-1].id].of == m.rounds[0].id
    first, rep = m.rounds[0], m.rounds[-1]           # the repeat shows the same clips, positions reversed
    assert [h.rounds[first.id].real(c.id) for c in first.clips][::-1] == \
           [h.rounds[rep.id].real(c.id) for c in rep.clips]
    assert not set(first.ids) & set(rep.ids)        # ...under fresh opaque ids


def test_public_copy_has_nothing_hidden(demos):
    m, h = load_batch(demos["shootout"])
    served = json.dumps(m.public())
    assert set(m.public()) == {"v", "batch", "mode", "bars", "sr", "note", "est_min", "rounds"}
    for hr in h.rounds.values():
        for info in hr.clips.values():
            assert info["clip"] not in served and info["option"] not in served
    assert "bpm" not in served and "sha256" not in served       # the scene is the only brief information


def test_playlist_plays_are_fixed_per_round(demos):
    m, _ = load_batch(demos["playlist"])
    assert [r["plays"] for r in m.public()["rounds"]] == [2, 3, 4, 2, 3]
    assert default_plays("r0123abcd") == default_plays("r0123abcd") and 2 <= default_plays("r0123abcd") <= 4


def test_audio_headers_match_soundfile(tmp_path):
    import numpy as np
    import soundfile as sf
    y = np.zeros((12345, 2), dtype=np.float32)
    sf.write(tmp_path / "a.flac", y, 22050, format="FLAC", subtype="PCM_16")
    sf.write(tmp_path / "a.wav", y, 22050, subtype="PCM_16")
    assert audio_info(tmp_path / "a.flac") == (22050, 2, 12345)
    assert audio_info(tmp_path / "a.wav") == (22050, 2, 12345)
    with wave.open(str(tmp_path / "b.wav"), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(8000), w.writeframes(b"\0\0" * 100)
    assert audio_info(tmp_path / "b.wav") == (8000, 1, 100)
    (tmp_path / "c.flac").write_bytes(b"not audio")
    with pytest.raises(ValueError):
        audio_info(tmp_path / "c.flac")


@pytest.mark.parametrize("change, expect", [
    (lambda d: d.update(variant="rhodes"), "unknown key(s) ['variant']"),
    (lambda d: d.update(mode="vote"), "mode: must be one of"),
    (lambda d: d.update(v=2), "v: must be 1"),
    (lambda d: d.pop("bars"), "bars: required"),
    (lambda d: d.update(bpm=500), "bpm: must be a number from 30 to 300"),
    (lambda d: d["rounds"][0].update(id="round-1"), "rounds[0].id: must match"),
    (lambda d: d["rounds"][1].update(id=d["rounds"][0]["id"]), "appears twice"),
    (lambda d: d["rounds"][0]["clips"].pop(), "a bws4 round has 4 clips, this one has 3"),
    (lambda d: d["rounds"][0]["clips"][0].update(id="rhodes-take-1"), "clips[0].id: must be opaque"),
    (lambda d: d["rounds"][0]["clips"][0].update(file="../manifest.hidden.json"), "must be a bare opaque file name"),
    (lambda d: d["rounds"][0]["clips"][0].update(file="sub/k0123456789.flac"), "must be a bare opaque file name"),
    (lambda d: d["rounds"][0]["clips"][0].update(option="piano"), "unknown key(s) ['option']"),
    (lambda d: d["rounds"][0]["clips"][0].update(sha256="0" * 64), "sha256 does not match"),
    (lambda d: d["rounds"][0].update(plays=3), "only playlist rounds have plays"),
    (lambda d: d.update(batch="other"), "does not match its folder name"),
])
def test_public_schema_errors_are_reported(demos, tmp_path, change, expect):
    d = _copy(demos["bws4"], tmp_path)
    _edit(d / PUBLIC_NAME, change)
    assert expect in _errors(d)


def test_file_problems_are_reported(demos, tmp_path):
    d = _copy(demos["bws4"], tmp_path)
    pub = json.loads((d / PUBLIC_NAME).read_text())
    files = [c["file"] for r in pub["rounds"] for c in r["clips"]]
    os.remove(d / files[0])
    (d / files[1]).unlink()
    os.symlink(d / files[2], d / files[1])                  # a symlink, even to a clip, is refused
    errs = _errors(d)
    assert f"{files[0]}: listed but missing" in errs
    assert f"{files[1]}: is a symlink" in errs


def test_round_clips_must_be_equally_long(demos, tmp_path):
    import numpy as np
    import soundfile as sf
    d = _copy(demos["bws4"], tmp_path)
    pub = json.loads((d / PUBLIC_NAME).read_text())
    first = pub["rounds"][0]["clips"][0]
    sf.write(d / first["file"], np.zeros(8 * 8000 - 1000, dtype=np.float32), 8000, format="FLAC", subtype="PCM_16")
    first.pop("sha256")
    (d / PUBLIC_NAME).write_text(json.dumps(pub))
    errs = _errors(d)
    assert "its clips differ in length" in errs


def test_hidden_manifest_is_required_and_checked(demos, tmp_path):
    d = _copy(demos["shootout"], tmp_path)
    (d / HIDDEN_NAME).unlink()
    assert "manifest.hidden.json (it says which rounds are repeats or anchors) is missing" in _errors(d)

    d = _copy(demos["shootout"], tmp_path, "demo-shootout-2")
    for name in (PUBLIC_NAME, HIDDEN_NAME):
        _edit(d / name, lambda x: x.update(batch="demo-shootout-2"))
    hid = json.loads((d / HIDDEN_NAME).read_text())
    r0, r1, r2 = list(hid["rounds"])[:3]
    hid["rounds"][r0]["kind"] = "repeat"                    # a repeat without `of`
    hid["rounds"][r1].pop("anchor_clip")                    # an anchor without its broken copy
    oid = next(iter(hid["rounds"][r2]["clips"]))
    hid["rounds"][r2]["clips"][oid].pop("option")           # a shootout clip without its option
    hid["rounds"].pop(list(hid["rounds"])[-1])              # a public round with no hidden entry
    (d / HIDDEN_NAME).write_text(json.dumps(hid))
    errs = _errors(d)
    assert f"rounds.{r0}.of: a repeat names the round id" in errs
    assert f"rounds.{r1}.anchor_clip: an anchor names" in errs
    assert f"rounds.{r2}.clips.{oid}.option: shootout clips name the option" in errs
    assert "missing (every public round needs an entry)" in errs


def _leak(demos, tmp_path, fn):
    """Apply fn(public, hidden) to a copy of the shootout demo and return the hidden-manifest errors."""
    d = _copy(demos["shootout"], tmp_path)
    pub = json.loads((d / PUBLIC_NAME).read_text())
    hid = json.loads((d / HIDDEN_NAME).read_text())
    fn(pub, hid)
    m = parse_public(pub, d, check_files=False)
    with pytest.raises(ManifestError) as e:
        parse_hidden(hid, m)
    return "\n".join(e.value.errors)


def _rename(pub, hid, c_index, new):
    rnd = pub["rounds"][0]
    old = rnd["clips"][c_index]["id"]
    rnd["clips"][c_index]["id"] = new
    clips = hid["rounds"][rnd["id"]]["clips"]
    clips[new] = clips.pop(old)
    return clips[new]


def test_opaque_ids_must_not_leak_variant_names(demos, tmp_path):
    def spelled(pub, hid):
        _rename(pub, hid, 0, "kab0123456")["option"] = "kab"       # an id that spells the option
    assert "clip id kab0123456 contains 'kab' (option of clip kab0123456)" in _leak(demos, tmp_path, spelled)

    def long_hex_word(pub, hid):
        _rename(pub, hid, 1, "kbeefcafe00ab")["source"] = "beefcafe00ab"   # all hex, but long enough to check
    assert "contains 'beefcafe00ab' (source of clip kbeefcafe00ab)" in _leak(demos, tmp_path, long_hex_word)

    def real_id(pub, hid):
        rnd = pub["rounds"][0]
        info = hid["rounds"][rnd["id"]]["clips"][rnd["clips"][0]["id"]]
        rnd["clips"][1]["file"] = "k" + info["clip"][1:12] + ".flac"   # a file named after a real clip id
    errs = _leak(demos, tmp_path, real_id)
    assert "file k" in errs and "(the real id of clip" in errs

    def note(pub, hid):
        pub["note"] = "Keys shootout: is the piano better than the organ?"
    errs = _leak(demos, tmp_path, note)
    assert "the note names 'piano'" in errs and "the note names 'organ'" in errs
    assert "'keys'" not in errs                                    # naming the category is fine


def test_short_hex_words_are_not_flagged(demos, tmp_path):
    """Random hex ids contain short hex words like 'fade' by chance; flagging them would block real batches."""
    d = _copy(demos["shootout"], tmp_path)
    pub = json.loads((d / PUBLIC_NAME).read_text())
    hid = json.loads((d / HIDDEN_NAME).read_text())
    _rename(pub, hid, 0, "kfade012345")["what"] = "fade take"
    parse_hidden(hid, parse_public(pub, d, check_files=False))


def test_broken_of_may_name_an_opaque_id(demos):
    _, h = load_batch(demos["bws4"])
    anchor = next(hr for hr in h.rounds.values() if hr.kind == "anchor")
    assert anchor.clips[anchor.anchor_clip]["broken_of"] in anchor.clips      # and the batch still validates


def test_demo_refuses_to_overwrite(demos):
    with pytest.raises(FileExistsError):
        build_demo(demos["ab"], "ab", **FAST)


def test_cli_demo_and_check(tmp_path, capsys):
    from lofi_rl.ui.manifest import _main
    d = tmp_path / "2026-11-03" / "cli-demo"
    assert _main(["demo", str(d), "--mode", "triage", "--rounds", "3", "--bpm", "240", "--sr", "8000",
                  "--format", "wav"]) == 0
    assert (d / PUBLIC_NAME).exists() and len(list(d.glob("*.wav"))) == 3
    assert _main(["check", str(d)]) == 0
    assert "ok: cli-demo (triage), 3 rounds" in capsys.readouterr().out
    _edit(d / PUBLIC_NAME, lambda x: x.update(mode="nope"))
    assert _main(["check", str(d)]) == 2


def test_manifest_text_must_be_valid_unicode(demos, tmp_path):
    """A lone surrogate ("\\ud800") in a manifest would pass as text and then fail every vote write for that round
    (integrity-4, the same crash by another way in): refuse it when the batch loads."""
    pub = _copy(demos["bws4"], tmp_path)
    raw = (pub / PUBLIC_NAME).read_text()
    (pub / PUBLIC_NAME).write_text(raw.replace('"scene": "demo batch: synthetic test tones"',
                                               '"scene": "rain \\ud800 at night"', 1))
    assert "not valid Unicode" in _errors(pub)
    hid = _copy(demos["bws4"], tmp_path)
    _edit(hid / HIDDEN_NAME, lambda d: next(iter(d["rounds"].values())).update(brief_id="b-x"))
    raw = (hid / HIDDEN_NAME).read_text()
    (hid / HIDDEN_NAME).write_text(raw.replace('"brief_id": "b-x"', '"brief_id": "b-\\udfff"', 1))
    assert "not valid Unicode" in _errors(hid)
