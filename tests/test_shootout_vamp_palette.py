"""The vamp rounds' palette (docs/SHOOTOUTS.md §7.3 acceptance tests), rendered through the real pipeline.

The generator mutes the sax of every vamp palette with a -200 dB `Gain` right after the sax chain's trim slot
(`lofi_rl.shootout.palettes.mute_sax`), so the sax bus is silent while its stem still renders, and lets the loudness
loop take MUTED_LOUDNESS_ITERS iterations, since the master gain is higher with no sax. Here the base palette's GU
band (the committed GU SoundFont only) renders one short loop muted and unmuted: the muted palette loads with
verify=True, passes G7 at -14 +- 0.1 LUFS with a null sax bus loudness, every stem (the sax's too) has the unmuted
palette's hash, and two renders are identical. Nothing is listened to and nothing is claimed about sound."""
import copy

import pytest
import yaml

from helpers_render import needs_fluidsynth
from helpers_shootout import BASE_PATH
from lofi_rl.brief import Brief
from lofi_rl.palette import parse_palette
from lofi_rl.render.pipeline import render
from lofi_rl.shootout.palettes import MUTE_DB, MUTED_LOUDNESS_ITERS, GenerateError, mute_sax

# A short loop (95 BPM, the shortest pass) with keys, bass, drums and a sax line, passing G1-G6 in wistful.
LOOP = ("vamp: Im9 IV13 span 2\nkeys: ep rootless\ncomp: x-------x------- x---------x-----\n"
        "bass: x-------..x----- x-----------....\ndrums: dusty\nkick: x.........x..... x.......x.......\n"
        "rim: ....x.......x... ....x.......x...\nA: h5 q.4 e3 | q1 hr q3 | h.6 er e5 | q4 e3 e1 hr\n"
        "B: h5 q.4 e3 | q1 e7 e1 hr | h.3 er e5 | q4 e3 q.2 qr\n")
BRIEF = Brief(energy="low", mood="wistful", bpm=95, tonic=2)


def gu_base() -> dict:
    """The committed base palette with only its GU source: the dev band under the shootout names, trim slots in."""
    doc = yaml.safe_load(BASE_PATH.read_text())
    doc["sources"] = {"gu": doc["sources"]["gu"]}
    return doc


def test_mute_sax_puts_the_gain_after_the_trim_slot_and_raises_the_loudness_iterations():
    doc = gu_base()
    out = mute_sax(copy.deepcopy(doc))
    assert out["mix"]["buses"]["sax"][0] == doc["mix"]["buses"]["sax"][0] == ["Gain", {"gain_db": 0.0}]
    assert out["mix"]["buses"]["sax"][1] == ["Gain", {"gain_db": MUTE_DB}] and MUTE_DB == -200.0
    assert out["mix"]["buses"]["sax"][2:] == doc["mix"]["buses"]["sax"][1:]
    assert out["loudness"] == {**doc["loudness"], "iters": MUTED_LOUDNESS_ITERS}
    assert {b: c for b, c in out["mix"]["buses"].items() if b != "sax"} == \
        {b: c for b, c in doc["mix"]["buses"].items() if b != "sax"}
    bad = copy.deepcopy(doc)
    bad["mix"]["buses"]["sax"] = bad["mix"]["buses"]["sax"][1:]
    with pytest.raises(GenerateError, match="trim slot"):
        mute_sax(bad)


@pytest.mark.laptop
@needs_fluidsynth
def test_a_vamp_palette_renders_with_a_silent_sax_bus_and_the_unmuted_stems(tmp_path):
    plain = gu_base()
    muted = mute_sax(copy.deepcopy(plain))
    pals = {}
    for name, doc in (("plain", plain), ("muted", muted)):
        doc["id"] = f"vamp_test_{name}"
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        pals[name] = parse_palette(yaml.safe_load(path.read_text()), str(path))
    from lofi_rl.palette import load_palette
    assert load_palette(tmp_path / "muted.yaml", verify=True).mix.buses["sax"][1].params == {"gain_db": MUTE_DB}
    a = render(LOOP, BRIEF, pals["muted"], workers=2)
    b = render(LOOP, BRIEF, pals["muted"], workers=2)
    p = render(LOOP, BRIEF, pals["plain"], workers=2)
    m = a.manifest
    assert m["gates"] == {"G1-G6": "pass", "G7": "pass"} and abs(m["loudness"]["lufs"] + 14.0) <= 0.1
    assert m["buses"]["sax"]["lufs"] is None and p.manifest["buses"]["sax"]["lufs"] is not None
    assert all(m["buses"][x]["lufs"] == p.manifest["buses"][x]["lufs"] for x in ("keys", "bass", "drums"))
    assert m["stems"] == p.manifest["stems"], "the mute is in the sax chain: every stem renders as unmuted"
    assert a.manifest["audio_sha256"] == b.manifest["audio_sha256"] != p.manifest["audio_sha256"]
