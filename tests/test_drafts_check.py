"""The draft validator (lofi_rl.drafts.check; docs/DRAFTS.md §5), on fixture drafts written here for two briefs of
drafts/v1/briefs.yaml (d01 low/melancholy, d05 mid/wistful). They are test fixtures, never the real drafts, which the
writers produce; nothing here is a claim about how any of them sounds."""
import json
from pathlib import Path

import pytest
import yaml

from lofi_rl import prompt
from lofi_rl.drafts import check as C
from lofi_rl.drafts.palette import load as load_palette
from lofi_rl.dsl import parse

ROOT = Path(__file__).resolve().parents[1]
BRIEFS = ROOT / "drafts" / "v1" / "briefs.yaml"
FIXTURES = {
    "d05_groove.ls": """\
vamp: Im11 IV9 span 1
keys: rhodes shell
comp: x-------...x.... ......x-----..x.
bass: x-----x...x..x.. x-------..x-x...
drums: studio
kick: x.....x...x..... x.......x.x.....
snare: ....X.......x... ....X.......x...
hat: x.x.x.x.x.x.x.x. x.x.x.xxx.x.x.x.
@2 kick: x.....x...x...x.
@3 hat: x.x.x.x.x.x.x.xx
@4 snare: ....X.......x..g
@5 bass: x-----x...x.x...
@6 kick: x.........x.....
@7 comp: x---....x-----..
@8 hat: x.x.x.x.x.x.xxx.
A: h5 q4 e3 e2 | q.1 e2 q3 qr | h.5 er e4 | q3 e2 e1 hr
B: h5 q4 e3 e2 | q.1 e7, q1 qr | q6 e5 e4 h3 | q2 e1 e2 hr
""",
    "d05_horn.ls": """\
vamp: Im9 IV13 span 2
keys: wurli rootless
comp: x-----x-----.... ..x-------x-----
bass: lock
drums: jazz
kick: x.......x.x..... x.....x...x.....
rim: ....x.......x... ....x.......x.g.
hat: x.x.x.x.x.x.x.x. x.x.x.x.x.x.x.x.
shaker: ..x...x...x...x. ..x...x...x...x.
@3 kick: x.......x..x....
@4 hat: x.x.x.x.x.x.x.xx
@5 comp: x-------x-----..
@6 rim: ....x..g....x...
@7 kick: x.....x...x.....
@8 hat: x.x.x.x.x.xxx.x.
A: w1' | qr e7 e6 h5 | h.4 qr | e3 e4 q5 hr
B: w1' | qr e2' e1' h7 | h.6 qr | e5 e4 q3 hr
""",
    "d05_chords.ls": """\
vamp: IIm7 Im11 span 2
keys: piano quartal
comp: x-----x---x---x- x-----x---x-----
bass: x-----x---x----- x-------x-x-----
drums: studio
kick: x.........x..... x.......x.......
snare: ....x.......x... ....x.......x...
hat: x.g.x.g.x.g.x.g. x.g.x.g.x.g.x.gx
ohat: ................ ..............x.
@2 comp: x-----x---x-x---
@4 kick: x.........x...x.
@5 hat: x.g.x.g.x.g.xxg.
@6 comp: x---x-----x---x-
@7 snare: ....x..g....x...
@8 kick: x.........x.....
A: q2 e3 e4 h5 | q6 e5 e4 q3 qr | h.2 e1 e2 | q3 qr hr
B: q2 e3 e4 h5 | q6 e7 e1' q2' qr | h.7 e6 e5 | q4 qr hr
""",
    "d01_groove.ls": """\
vamp: Im9 bVImaj7 span 2
keys: piano rootless
comp: x-------........ x-----------....
bass: x-------..x..... x-----....x.....
drums: jazz
kick: x.........x..... x.......x.......
rim: ....x.......x... ....x.......x...
hat: x...x...x...x... x...x...x...x.x.
@2 bass: x-------..x...x.
@3 kick: x.........x.x...
@5 hat: x...x...x.x.x...
@6 comp: x-----------x---
@7 kick: x.......x.......
@8 rim: ....x.......x..g
A: h.5 e4 e3 | h1 qr q2 | q3 e4 e5 h6 | q5 e4 e3 hr
B: h.5 e4 e3 | h1 qr e7, e1 | q2 e3 e4 h3 | q2 e1 e7, hr
""",
    "d01_horn.ls": """\
vamp: IVm9 Im9 span 2
keys: rhodes spread
comp: x--------------- ........x-------
bass: x-------........ x-------....x...
drums: studio
kick: x.........x..... x.....x.........
snare: ....x.......x... ....x.......x...
hat: x.g.x...x.g.x... x.g.x...x.g.x.x.
@2 kick: x.........x...x.
@4 comp: x-------x-------
@5 hat: x.g.x...x.g.x.g.
@6 bass: x-------..x.....
@7 kick: x.......x.......
@8 snare: ....x.......x.g.
A: q.3 e2 h1 | qr q2 h3 | w5 | e4 e3 q2 hr
B: q.3 e2 h1 | qr q7, h1 | h.b5 e4 e3 | q2 e1 e2 hr
""",
    "d01_chords.ls": """\
vamp: bVImaj7 Im11 span 1
keys: wurli shell
comp: x-----......x--- x---------......
bass: x-----......x... x---------......
drums: jazz
kick: x...........x... x.........x.....
rim: ....x.......x... ....x.......x...
hat: x...x.x.x...x.x. x...x.x.x...x...
@3 comp: x---------x-----
@4 kick: x.....x.....x...
@5 hat: x...x.x.x...x.xx
@6 comp: x-----x-----....
@7 bass: x---------x-----
@8 kick: x.........x.....
A: e5 e6 q5 h3 | q2 e1 e2 h3 | q4 e3 e2 h1 | h.2 qr
B: e5 e6 q5 h3 | q2 e3 e4 h5 | q6 e5 e4 h3 | h.4 qr
""",
}


def briefs_doc(ids=("d01", "d05")) -> dict:
    doc = yaml.safe_load(BRIEFS.read_text(encoding="utf-8"))
    doc["briefs"] = {k: v for k, v in doc["briefs"].items() if k in ids}
    doc["count"] = len(doc["briefs"])
    return doc


def write_folder(folder: Path, drafts: dict[str, str] = FIXTURES, ids=("d01", "d05")) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "briefs.yaml").write_text(yaml.safe_dump(briefs_doc(ids), sort_keys=False), encoding="utf-8")
    for name, text in drafts.items():
        (folder / name).write_text(text, encoding="utf-8")
    return folder


def by_file(rep: C.Report) -> dict[str, C.Draft]:
    return {d.file: d for d in rep.drafts}


def test_the_fixtures_pass_every_check(tmp_path, capsys):
    folder = write_folder(tmp_path / "v1")
    rep = C.check_folder(folder)
    assert rep.ok and rep.missing == [] and len(rep.drafts) == 6
    assert rep.n_refs == 10 and rep.n_pool == 6
    for d in rep.drafts:
        assert d.ok, (d.file, d.problems)
        assert d.info["band"] == d.brief.energy and d.info["mood_margin"] >= 2
        assert d.info["refs_j5"] < 0.6 and d.info["drafts_j5"] < 0.6 and d.info["repairs"] == []
    assert C.main([str(folder), "--json", str(tmp_path / "out.json")]) == 0
    out = capsys.readouterr().out
    assert "6 of 6 drafts pass; 0 missing" in out and "d05_horn.ls" in out
    doc = json.loads((tmp_path / "out.json").read_text())
    assert doc["ok"] and doc["schema"] == C.SCHEMA and {d["file"] for d in doc["drafts"]} == set(FIXTURES)


def test_every_reference_is_compared(tmp_path):
    pal = load_palette().palette
    refs = C.references(pal)
    ids = [r.id for r in refs]
    assert ids[0] == "prompt_v1:example" and len(ids) == 10
    assert sum(i.startswith("prompt_v2:") for i in ids) == 8      # v2's ninth is v1's example, listed once
    assert ids[-1] == "shootout:l5_wistful_mid"                    # the one shootout loop no prompt shows
    assert len({r.form for r in refs}) == 10


def test_one_draft_per_brief_per_persona(tmp_path, capsys):
    drafts = {k: v for k, v in FIXTURES.items() if k != "d01_horn.ls"}
    folder = write_folder(tmp_path / "v1", drafts)
    rep = C.check_folder(folder)
    assert not rep.ok and rep.missing == ["d01_horn.ls"]
    assert C.main([str(folder)]) == 1
    assert "d01_horn.ls: missing" in capsys.readouterr().out
    assert C.check_folder(folder, scope=["d05"]).ok                # out of scope, d01 is not asked for


@pytest.mark.parametrize("name,why", [("d05_trumpet.ls", "persona"), ("d09_horn.ls", "brief d09"),
                                      ("notes.ls", "not <brief id>")])
def test_names(tmp_path, name, why):
    folder = write_folder(tmp_path / "v1", {**FIXTURES, name: FIXTURES["d05_horn.ls"]})
    d = by_file(C.check_folder(folder))[name]
    assert why in d.problems["name"]


def _one(tmp_path, name: str, text: str) -> C.Draft:
    folder = write_folder(tmp_path / "v1", {**FIXTURES, name: text})
    return by_file(C.check_folder(folder))[name]


def test_a_gate_failure(tmp_path):
    d = _one(tmp_path, "d05_groove.ls", FIXTURES["d05_groove.ls"].replace("drums: studio", "drums: dusty"))
    assert d.problems["gate"].startswith("G4")


def test_any_repair_fails(tmp_path):
    d = _one(tmp_path, "d05_groove.ls", FIXTURES["d05_groove.ls"].replace("@6 kick: x.........x.....",
                                                                          "@6 kick: x.........x..."))
    assert "pad_grid" in d.problems["gate"] and d.info["repairs"] == ["pad_grid"]
    d = _one(tmp_path, "d05_groove.ls", FIXTURES["d05_groove.ls"].replace("vamp: Im11 IV9 span 1", "vamp: Im11 IV9"))
    assert "default_span" in d.problems["gate"]          # a free repair fails too: a draft is written in the language


def test_a_sax_line_fails(tmp_path):
    d = _one(tmp_path, "d05_groove.ls", FIXTURES["d05_groove.ls"].replace("A: ", "sax: alto\nA: ", 1))
    assert "sax: line" in d.problems["gate"]


def test_a_sax_octave_fold_fails(tmp_path):
    text = FIXTURES["d05_groove.ls"].replace("A: h5 q4", "A: h5'' q4")
    d = _one(tmp_path, "d05_groove.ls", text)
    assert "sax_fold" in d.problems["compile"]


def test_energy_must_be_the_briefs(tmp_path):
    text = FIXTURES["d01_groove.ls"].replace("vamp: Im9 bVImaj7 span 2", "vamp: Im9 IV13 span 2")
    d = _one(tmp_path, "d05_groove.ls", text)             # a low loop under a mid brief
    assert set(d.problems) == {"energy", "drafts"} and "is low; the brief is mid" in d.problems["energy"]


def test_the_mood_signature_must_be_the_briefs(tmp_path):
    d = _one(tmp_path, "d05_groove.ls", FIXTURES["d05_groove.ls"].replace("vamp: Im11 IV9 span 1",
                                                                          "vamp: IIm9 V13 span 1"))
    assert set(d.problems) == {"mood"} and "dorian" in d.problems["mood"]


def test_a_near_copy_of_a_prompt_example_or_a_shootout_loop_fails(tmp_path):
    example = prompt.EXAMPLE_TEMPLATE.format(keys="piano", kit="jazz")
    d = _one(tmp_path, "d05_chords.ls", example)
    assert d.info["refs_nearest"] == "prompt_v1:example" and d.info["refs_j5"] == 1.0 and "refs" in d.problems
    l5 = (ROOT / "shootouts" / "loops" / "l5_wistful_mid.ls").read_text().replace("keys: ep", "keys: wurli") \
        .replace("drums: dusty", "drums: studio").replace("@8 comp: x-------x--x--x-", "@8 comp: x-------x--x----")
    d = _one(tmp_path, "d05_chords.ls", l5)
    assert d.info["refs_nearest"] == "shootout:l5_wistful_mid" and d.info["refs_j5"] > 0.9 and "refs" in d.problems


def test_a_near_copy_of_another_draft_fails_both(tmp_path):
    twin = FIXTURES["d05_groove.ls"].replace("keys: rhodes shell", "keys: piano shell").replace("drums: studio",
                                                                                               "drums: jazz")
    folder = write_folder(tmp_path / "v1", {**FIXTURES, "d05_horn.ls": twin})
    got = by_file(C.check_folder(folder))
    for a, b in (("d05_horn.ls", "d05_groove.ls"), ("d05_groove.ls", "d05_horn.ls")):
        assert got[a].info["drafts_j5"] == 1.0 and got[a].info["drafts_nearest"] == b and "drafts" in got[a].problems


def test_a_redraft_is_compared_with_the_drafts_it_replaces(tmp_path):
    v1 = write_folder(tmp_path / "v1")
    redo = tmp_path / "v1" / "redo1"
    redo.mkdir()
    fresh = {"d05_groove.ls": FIXTURES["d01_chords.ls"].replace("vamp: bVImaj7 Im11 span 1", "vamp: Im9 IV13 span 1"),
             "d05_horn.ls": FIXTURES["d05_horn.ls"], "d05_chords.ls": FIXTURES["d05_chords.ls"]}
    for name, text in fresh.items():
        (redo / name).write_text(text, encoding="utf-8")
    rep = C.check_folder(redo, scope=["d05"], against=[v1])
    got = by_file(rep)
    assert len(rep.against) == 6 and rep.missing == []
    assert got["d05_horn.ls"].info["drafts_nearest"] == "v1/d05_horn.ls" and "drafts" in got["d05_horn.ls"].problems
    assert "drafts" in got["d05_chords.ls"].problems
    assert got["d05_groove.ls"].info["drafts_nearest"] == "v1/d01_chords.ls"


def test_copy_form_ignores_the_instrument_names_and_the_sax_line():
    pal = load_palette().palette
    a = parse(FIXTURES["d05_groove.ls"], vocab=pal.vocab(), mode="dorian").ast
    b = parse(FIXTURES["d05_groove.ls"].replace("rhodes", "piano").replace("studio", "jazz")
              .replace("A: ", "sax: alto\nA: ", 1), vocab=pal.vocab(), mode="dorian").ast
    assert C.copy_form(a) == C.copy_form(b) and "keys: K shell" in C.copy_form(a) and "drums: D" in C.copy_form(a)


def test_bad_arguments_exit_2(tmp_path, capsys):
    assert C.main([str(tmp_path / "none")]) == 2
    (tmp_path / "empty").mkdir()
    assert C.main([str(tmp_path / "empty")]) == 2
    folder = write_folder(tmp_path / "v1")
    assert C.main([str(folder), "--briefs", "d02"]) == 2
    assert "not in the briefs file" in capsys.readouterr().err
    assert C.main([str(folder), "--against", str(folder)]) == 2
    assert "itself" in capsys.readouterr().err
    assert C.main([str(folder), "--against", str(tmp_path / "nowhere")]) == 2
