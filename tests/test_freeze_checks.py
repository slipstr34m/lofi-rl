"""The checks before the sign-off (lofi_rl.freeze.checks; docs/DRAFTS.md §7.1; docs/FREEZE.md §5).

drafts/v1/keep3_kept10.json (schema lofi-rl/keep3-kept10/2) is the record of the kept loops' renders, each from its own
source round, rebuilt here from those rounds' summaries when runs/ holds them. Today it holds keep3's 10 with
`--allow-tossed`: the four loops Shresth tossed at the sign-off listen b1103-signoff are still in the kept list, since
no redo pick has replaced them yet. So the real report fails on exactly those four (its check (i)), and every other
check passes; a synthetic redo round that replaces them (tests/helpers_redo.py: a drafts tree in a temporary folder,
REDOS, SIGNOFFS, KEPT and JOBS monkeypatched) passes. The laptop report runs on a stand-in of the frozen palette made
from palettes/draft_day2.yaml the way the generator makes the real file (tests/test_freeze_signoff.py), and on
palettes/lofi_rain_v1.yaml itself, as it stands and as the seal leaves it (copies). The freeze check's command
(`python -m lofi_rl.freeze check`) is replaced by a recorder, except in the last two tests. The tests on the committed
file and the real gate file read them by what they record (the kept list, its sources, the round that supplies a
brief), so recording a redo round and running `expect` again leaves them valid; keep3's own bus table (the freeze
spec's §4.1 and Q1) is checked on keep3's ten. Nothing is rendered and nothing is claimed about sound."""
import copy
import hashlib
import itertools
import json
import re
import statistics
from pathlib import Path

import pytest
import yaml

import helpers_redo as H
from lofi_rl import freeze as FZ
from lofi_rl.drafts import palette as P
from lofi_rl.freeze import checks as C
from lofi_rl.freeze import gate as G
from lofi_rl.freeze import palette as F
from lofi_rl.palette import load_palette
from test_freeze_signoff import frozen_doc_from_draft

ROOT = Path(__file__).resolve().parents[1]
K3R = ROOT / "runs" / "drafts-13656049" / "summary" / "renders.json"
EXPECT = ROOT / "drafts" / "v1" / C.EXPECT_NAME
GATE = ROOT / "gates" / "GATE_palette.md"
FROZEN = ROOT / "palettes" / "lofi_rain_v1.yaml"
KEPT = ["d01_chords", "d02_chords", "d03_chords", "d04_horn", "d05_groove", "d06_horn", "d07_chords", "d08_horn",
        "d09_chords", "d10_groove"]                     # gates/GATE_palette.md, Kept drafts (keep3)
TOSSED = ["d01_chords", "d06_horn", "d07_chords", "d10_groove"]  # gates/GATE_palette.md, Sign-off listen b1103-signoff
DAY, OTHER_DAY = "2026-11-05", "2026-11-06"             # his APPROVED date in these tests, and another


def standin(path: Path, **edit) -> Path:
    """The frozen palette as the generator builds it from the draft palette, with the rain bed of the Day-3 picks."""
    rain = json.loads((ROOT / "shootouts" / "rain_excerpts.json").read_text())
    src = rain["source"]
    source = {"path": src["asset_path"], "sha256": src["sha256"], "license": src["license"], "mode": "fetch",
              "url": src["url"], "converted_from": None, "attribution": None}
    bed = {"src": "rain", "start_s": rain["excerpts"]["e2"]["start_s"], "xfade_s": rain["parameters"]["xfade_s"],
           "delta_lu": 24.0}
    doc = frozen_doc_from_draft(P.load(), bed=bed, rain=source)
    for k, v in edit.items():
        doc[k] = v(doc[k])
    path.write_text("# tests only: a stand-in of palettes/lofi_rain_v1.yaml\n" + P.yaml_text(doc), encoding="utf-8")
    return path


class Recorder:
    def __init__(self, rc: int = 0, out: str = "ok: recorded"):
        self.calls, self.rc, self.out = [], rc, out

    def __call__(self, args, root=ROOT):
        self.calls.append([str(a) for a in args])
        return self.rc, self.out


def his_line(text: str, day: str) -> str:
    """The gate file's text with only `APPROVED by Shresth Rana <day>` under its `## Approval` heading (once he has
    signed, the real file holds his own line there, and it is replaced); the rest of the file as it is."""
    lines = text.splitlines(keepends=True)
    (head,) = [i for i, line in enumerate(lines) if line.rstrip() == G.APPROVAL_HEADING]
    end = next((i for i in range(head + 1, len(lines)) if re.match(r"#{1,2}\s", lines[i])), len(lines))
    rest = "".join(lines[end:])
    out = "".join(lines[:head + 1]).rstrip("\n") + f"\n\nAPPROVED by Shresth Rana {day}\n"
    out += "\n" + rest if rest else ""
    assert G.approval_date(out) == day                  # read where the seal and `check --sealed` read it
    return out


def real_drafts() -> int:
    """How many drafts the real folders hold: drafts/v1's and every drafts/v1/redo<k>/'s present."""
    return len(list((ROOT / "drafts" / "v1").glob("*.ls"))) + len(list((ROOT / "drafts" / "v1").glob("redo*/*.ls")))


def committed_renders() -> dict[str, Path]:
    """{round key: its summary/renders.json} for every source of the committed file; skip unless runs/ holds them."""
    doc = json.loads(EXPECT.read_text(encoding="utf-8"))
    out = {s["key"]: ROOT / s["path"] for s in doc["sources"]}
    gone = [str(p.relative_to(ROOT)) for p in out.values() if not p.is_file()]
    if gone:
        pytest.skip(f"{gone} not here (runs/ is never committed or synced)")
    return out


def renders_args(renders: dict[str, Path]) -> list[str]:
    return [x for k, p in renders.items() for x in ("--renders", f"{k}={p}")]


# --------------------------------------------------------------------------------------------- keep3_kept10.json
def test_the_committed_expectations_are_what_expect_writes_from_their_sources(tmp_path, capsys):
    renders = committed_renders()
    allow = C.load_expectations(EXPECT)["allow_tossed"]
    doc = C.expect_from_files(renders, draft=P.load(), allow_tossed=allow)
    assert C.expect_text(doc) == EXPECT.read_text(encoding="utf-8")
    out = tmp_path / "k.json"
    assert C.main(["expect", *renders_args(renders), *(["--allow-tossed"] if allow else []), "--out", str(out)]) == 0
    assert out.read_bytes() == EXPECT.read_bytes()
    printed = capsys.readouterr().out
    assert f"{len(doc['kept'])} kept loops from keep3 runs/drafts-13656049/summary/renders.json" in printed


def test_expect_refuses_while_a_tossed_loop_is_kept(tmp_path, capsys):
    if not FZ.tossed_kept():
        pytest.skip("no kept loop is tossed now")
    renders = committed_renders()
    assert C.main(["expect", *renders_args(renders), "--out", str(tmp_path / "k.json")]) == 2
    err = capsys.readouterr().err.splitlines()
    assert err == [f"error: {x}" for x in C.tossed_problems()] and not (tmp_path / "k.json").exists()
    # the --keep3 alias is --renders keep3=
    assert C.main(["expect", "--keep3", str(renders["keep3"]), "--renders", f"keep3={renders['keep3']}"]) == 2
    assert "is given twice" in capsys.readouterr().err


def test_the_committed_expectations_hold_the_kept_texts_and_the_tossed_record():
    doc = C.load_expectations(EXPECT)
    assert doc["schema"] == C.EXPECT_SCHEMA == "lofi-rl/keep3-kept10/2"
    assert doc["kept"] == list(FZ.kept_now().values()) == list(C.gate_kept_now(GATE).values())
    keep3 = [s for s in doc["sources"] if s["key"] == "keep3"]
    assert keep3 == [{"key": "keep3", "path": "runs/drafts-13656049/summary/renders.json", "job": 13656049,
                      "sha256": "94ee7e1ec732173e25aded99966f1177bd91df35a47cdec8c0271875e9d8163d"}]
    assert [s["key"] for s in doc["sources"]] == [k for k in C.round_order() if k in {s for _f, s in
                                                                                       FZ.kept_rounds().values()}]
    assert doc["s_star"] == P.load().levels["S_star"]
    rounds = FZ.kept_rounds()
    for rid, e in doc["loops"].items():
        assert C.file_sha256(ROOT / e["file"]) == e["draft_sha256"] and Path(e["file"]).stem == rid
        assert (e["file"], e["source"]) == rounds[e["brief_id"]]
        for b in C.BUSES:                           # its source's lu was post_lufs - S*, as drafts/render.py has it
            assert e["buses"][b]["post_lufs"] - doc["s_star"] == e["buses"][b]["lu"]
    assert C.kept_problems(doc, GATE) == [] and C.constants_problems(doc) == []
    assert [(t["brief"], t["file"], t["signoffs"]) for t in doc["tossed"]] == [
        (b, f, list(s)) for b, f, s in FZ.tossed_kept()]
    assert doc["allow_tossed"] is True or doc["tossed"] == []


def test_today_the_tossed_loops_are_keep3s_four_of_b1103_signoff():
    """The state of 2026-11-03: no redo round recorded, so the four he tossed are still kept (and recorded so)."""
    if FZ.REDOS:
        pytest.skip("a redo round is recorded: the four are replaced")
    doc = C.load_expectations(EXPECT)
    assert doc["kept"] == [f"drafts/v1/{k}.ls" for k in KEPT] and doc["allow_tossed"] is True
    assert [t["file"] for t in doc["tossed"]] == [f"drafts/v1/{k}.ls" for k in TOSSED]
    assert {t["signoffs"][0] for t in doc["tossed"]} == {"b1103-signoff"}
    assert [s["key"] for s in doc["sources"]] == ["keep3"] and {e["source"] for e in doc["loops"].values()} == {"keep3"}


def k3_doc(tmp_path) -> tuple[dict, dict]:
    """A keep3 renders document for two drafts/v1 drafts copied to tmp_path, d01_chords and d05_groove, and its kept
    list as lofi_rl.freeze.kept_rounds gives one: {brief: (file, "keep3")}. Each record has the committed records'
    shape: the values of the committed loop of its brief (whichever round supplies that brief now), with its own file,
    text sha256 and persona, so the document does not depend on which loops are kept."""
    by_brief = {e["brief_id"]: e for e in C.load_expectations(EXPECT)["loops"].values()}
    renders = {}
    for rid in ("d01_chords", "d05_groove"):
        f = tmp_path / f"{rid}.ls"
        f.write_text((ROOT / "drafts" / "v1" / f"{rid}.ls").read_text(), encoding="utf-8")
        e = by_brief[rid[:3]]
        renders[rid] = {**{k: e[k] for k in C.LOOP_KEYS}, "file": str(f), "draft_sha256": C.file_sha256(f),
                        "persona": rid[4:], "status": "ok", "palette": {"label": e["palette"]},
                        "buses": copy.deepcopy(e["buses"])}
    return {"renders": renders}, {rid[:3]: (renders[rid]["file"], "keep3") for rid in renders}


def source(doc, key="keep3", job=9) -> C.Source:
    return C.Source(key, f"x/drafts-{job}/summary/renders.json", job, "a" * 64, doc)


def test_expectations_refuse_what_its_source_did_not_render(tmp_path):
    doc, kept = k3_doc(tmp_path)
    dp = P.load()
    ok = C.expectations([source(doc)], kept=kept, draft=dp, root=tmp_path)
    assert sorted(ok["loops"]) == ["d01_chords", "d05_groove"] and ok["s_star"] == dp.levels["S_star"]
    assert ok["sources"] == [{"key": "keep3", "path": "x/drafts-9/summary/renders.json", "job": 9, "sha256": "a" * 64}]
    assert {e["source"] for e in ok["loops"].values()} == {"keep3"} and ok["tossed"] == [] and not ok["allow_tossed"]

    def refused(d, kept=kept, sources=None):
        with pytest.raises(C.ChecksError) as err:
            C.expectations(sources or [source(d)], kept=kept, draft=dp, root=tmp_path)
        return err.value.errors
    assert refused(doc, kept={**kept, "d02": (str(tmp_path / "d02_chords.ls"), "keep3")})[0].endswith(
        "0 records in keep3's renders.json, not one")
    bad = copy.deepcopy(doc)
    bad["renders"]["d05_groove"]["status"] = "level"
    assert "keep3's render is level" in refused(bad)[0]
    bad = copy.deepcopy(doc)
    bad["renders"]["d05_x"] = bad["renders"].pop("d05_groove")              # rendered under another id
    assert "rendered it as d05_x, not under its stem d05_groove" in refused(bad)[0]
    (tmp_path / "d01_chords.ls").write_text("vamp: edited\n", encoding="utf-8")
    assert "not the one keep3 rendered" in refused(doc)[0]
    (tmp_path / "d01_chords.ls").write_text((ROOT / "drafts" / "v1" / "d01_chords.ls").read_text(), encoding="utf-8")
    bad = copy.deepcopy(doc)
    bad["renders"]["d01_chords"]["buses"]["keys"]["lu"] += 1e-9                     # another S* than the draft's
    assert "was not computed with S*" in refused(bad)[0]
    # a round that supplies no kept loop, and a supplier with no renders document
    assert refused(doc, sources=[source(doc), source(doc, "keep3_redo1", 10)]) == [
        "keep3_redo1 (x/drafts-10/summary/renders.json) supplies no kept loop now; leave its --renders out"]
    assert refused(doc, kept={**kept, "d05": (kept["d05"][0], "keep3_redo1")}) == [
        f"no renders.json for keep3_redo1, which supplies {[kept['d05'][0]]}"]


def test_a_renders_path_must_name_its_rounds_job(tmp_path):
    doc, _ = k3_doc(tmp_path)
    p = tmp_path / "runs" / "drafts-9" / "summary" / "renders.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(doc), encoding="utf-8")
    got = C.read_source("keep3", p, job=9, root=tmp_path)
    assert (got.path, got.job, got.sha256) == ("runs/drafts-9/summary/renders.json", 9, C.file_sha256(p))
    with pytest.raises(C.ChecksError, match=r"names the runs \[9\], not drafts-8"):
        C.read_source("keep3", p, job=8, root=tmp_path)
    with pytest.raises(C.ChecksError, match=r"names the runs \[\], not drafts-9"):
        C.read_source("keep3", tmp_path / "renders.json", job=9, root=tmp_path)
    assert C.renders_map(["keep3=a/b.json", "keep3_redo1=c.json"]) == {"keep3": Path("a/b.json"),
                                                                     "keep3_redo1": Path("c.json")}
    assert C.renders_map([], "k.json") == {"keep3": Path("k.json")}
    for bad in (["keep3"], ["=a.json"], ["keep3=a.json", "keep3=b.json"]):
        with pytest.raises(C.ChecksError):
            C.renders_map(bad)


def test_a_tossed_loop_still_kept_refuses_in_expect(tmp_path):
    """The message names the brief, the file and the sign-off batch; --allow-tossed writes it and records so."""
    doc, kept = k3_doc(tmp_path)
    tossed = [("d05", kept["d05"][0], ("b1103-signoff",))]
    with pytest.raises(C.ChecksError) as err:
        C.expectations([source(doc)], kept=kept, draft=P.load(), tossed=tossed, root=tmp_path)
    assert err.value.errors == [
        f"d05: {kept['d05'][0]} is still kept, but Shresth tossed it at the sign-off listen b1103-signoff "
        "(lofi_rl.freeze.SIGNOFFS); a redo round's pick replaces it first (docs/DRAFTS.md §6)"]
    got = C.expectations([source(doc)], kept=kept, draft=P.load(), tossed=tossed, allow_tossed=True, root=tmp_path)
    assert got["allow_tossed"] is True and got["tossed"] == [{"brief": "d05", "file": kept["d05"][0],
                                                              "signoffs": ["b1103-signoff"]}]


def test_load_expectations_refuses_a_file_that_is_not_one(tmp_path):
    """Each edit of the committed file names its own problem. The loops edited are the file's first two kept loops and
    the source added is a round it does not name, whichever rounds supply the kept list."""
    good = json.loads(EXPECT.read_text())
    one, two = (Path(f).stem for f in good["kept"][:2])
    keys = [s["key"] for s in good["sources"]]
    other = next(k for k in (f"keep3_redo{n}" for n in itertools.count(1)) if k not in keys)
    cases = {"schema": (lambda d: d.update(schema="lofi-rl/keep3-kept10/1"), "not a lofi-rl/keep3-kept10/2 document"),
             "loops": (lambda d: d["loops"].pop(two), "its loops are not exactly its kept files"),
             "bus": (lambda d: d["loops"][one]["buses"]["keys"].pop("off_lu"), f"loop {one} needs each bus's"),
             "source sha": (lambda d: d["sources"][0].update(sha256="nope"), "sources needs a list of"),
             "no sources": (lambda d: d.update(sources=[]), "sources needs a list of"),
             "loop source": (lambda d: d["loops"][two].update(source=other), f"loop {two} comes from {other}, not one"),
             "source unused": (lambda d: d["sources"].append({**d["sources"][0], "key": other}),
                               f"source {other} supplies no loop"),
             "loop field": (lambda d: d["loops"][two].pop("source"), f"loop {two} lacks a field"),
             "tossed": (lambda d: d.update(allow_tossed=False, tossed=[{"brief": d["loops"][two]["brief_id"],
                                                                       "file": d["kept"][1],
                                                                       "signoffs": ["b1103-signoff"]}]),
                        "it records tossed loops without allow_tossed"),
             "tossed shape": (lambda d: d.update(tossed=[{"file": d["kept"][1]}]), "needs allow_tossed and a tossed")}
    for name, (edit, what) in cases.items():
        d = copy.deepcopy(good)
        edit(d)
        f = tmp_path / f"{name}.json"
        f.write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises(C.ChecksError) as err:
            C.load_expectations(f)
        assert any(what in x for x in err.value.errors), (name, err.value.errors)
    with pytest.raises(C.ChecksError, match="not a lofi-rl/keep3-kept10/2 document"):   # main bd33ce2's file
        C.load_expectations(tmp_path / "schema.json")


def test_the_gate_files_keep3_table_is_read_by_its_heading(tmp_path):
    head = "## Kept drafts (keep3): voted 2026-11-01 (batch b1101-keep3; job 13656049)\n\n"
    table = "| Brief | Energy, mood | Kept |\n|---|---|---|\n| d01 | low, melancholy | `drafts/v1/d01_chords.ls` |\n" \
            "| d02 | low, wistful | `drafts/v1/d02_chords.ls` |\n"
    f = tmp_path / "g.md"
    f.write_text("# Gate\n\n## Approval\n\nAPPROVED by nobody\n\n" + head + "Some words first.\n\n" + table +
                 "\n- **Kept per persona**: `drafts/v1/d09_horn.ls` is named in prose, not in the table\n\n## Next\n"
                 "| d03 | low, hopeful | `drafts/v1/d03_chords.ls` |\n", encoding="utf-8")
    assert C.gate_kept(f) == ["drafts/v1/d01_chords.ls", "drafts/v1/d02_chords.ls"]
    for bad in ("# Gate\n" + table, head + head + table, head + "no table\n",
                head + table.replace("d02_chords", "d03_chords")):
        f.write_text(bad, encoding="utf-8")
        with pytest.raises(C.ChecksError):
            C.gate_kept(f)


# --------------------------------------------------------------------------------------------- the synthetic redo round
@pytest.fixture
def tree(tmp_path, monkeypatch):
    """The synthetic redo round of tests/helpers_redo.py with its constants on lofi_rl.freeze, synthetic renders.json
    files for keep3 (job 9, every drafts/v1 file) and for redo1 (job 10, every redo1 file) under tmp/runs, and a
    gate file recording it. keep3's records are the committed file's values (pcm keyed by the stem for the drafts it
    did not keep); redo1's are its own: another pcm_sha256 and music_sha256 for each file."""
    v1, redo1 = H.write_tree(tmp_path / "drafts")
    consts = H.constants(v1, redo1, keep3_job=9, redo_job=10)
    H.patch(monkeypatch, consts)
    loops = C.load_expectations(EXPECT)["loops"]
    by_brief = {e["brief_id"]: e for e in loops.values()}

    def record(f: Path, tag: str) -> dict:
        e = loops.get(f.stem) if tag == "keep3" else None
        e = e or {**by_brief[f.name[:3]], "persona": f.stem[4:],
                  "pcm_sha256": hashlib.sha256(f"{tag}:{f.stem}".encode()).hexdigest(),
                  "music_sha256": hashlib.sha256(f"{tag}:{f.stem}:music".encode()).hexdigest()}
        return {**{k: e[k] for k in C.LOOP_KEYS}, "file": str(f), "status": "ok", "palette": {"label": e["palette"]},
                "buses": copy.deepcopy(e["buses"]), "draft_sha256": C.file_sha256(f)}
    paths = {}
    for key, job, folder in (("keep3", 9, v1), ("keep3_redo1", 10, redo1)):
        p = tmp_path / "runs" / f"drafts-{job}" / "summary" / "renders.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"renders": {f.stem: record(f, key) for f in sorted(folder.glob("*.ls"))}}),
                     encoding="utf-8")
        paths[key] = p
    gate = tmp_path / "GATE_palette.md"
    gate.write_text(H.gate_text(consts), encoding="utf-8")
    H.lift_near_copy_limit(monkeypatch)
    return {"v1": v1, "redo1": redo1, "consts": consts, "renders": paths, "gate": gate, "tmp": tmp_path}


def test_expect_writes_schema_2_with_two_sources_and_the_redo_files_own_values(tree):
    doc = C.expect_from_files(tree["renders"], draft=P.load())
    redo1, v1 = tree["redo1"], tree["v1"]
    assert [s["key"] for s in doc["sources"]] == ["keep3", "keep3_redo1"]
    assert [s["job"] for s in doc["sources"]] == [9, 10] and doc["tossed"] == [] and doc["allow_tossed"] is False
    assert doc["kept"] == list(FZ.kept_now().values()) and len(doc["loops"]) == 10
    redo_doc = json.loads(tree["renders"]["keep3_redo1"].read_text())["renders"]
    keep3_doc = json.loads(tree["renders"]["keep3"].read_text())["renders"]
    for b in H.REOPENED:
        stem = f"{b}_{H.PICKS[b]}"
        e = doc["loops"][stem]
        assert (e["file"], e["source"]) == (str(redo1 / f"{stem}.ls"), "keep3_redo1")
        assert e["pcm_sha256"] == redo_doc[stem]["pcm_sha256"] != keep3_doc[stem]["pcm_sha256"]   # its own render
        assert e["draft_sha256"] == C.file_sha256(redo1 / f"{stem}.ls") != C.file_sha256(v1 / f"{stem}.ls")
    for stem in ("d02_chords", "d03_chords", "d04_horn", "d05_groove", "d08_horn", "d09_chords"):
        assert doc["loops"][stem]["source"] == "keep3" and doc["loops"][stem]["file"] == str(v1 / f"{stem}.ls")
    assert "d06_horn" in doc["loops"] and doc["loops"]["d06_horn"]["source"] == "keep3_redo1"   # not the tossed one
    with pytest.raises(C.ChecksError, match="no renders.json for keep3_redo1"):
        C.expect_from_files({"keep3": tree["renders"]["keep3"]}, draft=P.load())
    with pytest.raises(C.ChecksError, match="not keep3 or a round"):
        C.expect_from_files({**tree["renders"], "keep3_redo2": tree["renders"]["keep3"]}, draft=P.load())


def expect_args(tree, out: Path, *extra: str, rounds=("keep3", "keep3_redo1")) -> list[str]:
    return ["expect", *[x for k in rounds for x in ("--renders", f"{k}={tree['renders'][k]}")], "--gate",
            str(tree["gate"]), "--out", str(out), *extra]


def test_the_expect_command_checks_the_record_before_it_writes(tree, capsys, monkeypatch):
    """`checks expect` on the tree (no runs/ needed): it writes the file from both rounds. It refuses (exit 2) and
    writes nothing without --renders, with keep3's renders given twice, when the gate file's kept list is not
    kept_now(), and when REDOS breaks its rules. With the redo round left out of REDOS the four he tossed are kept
    again: refused unless --allow-tossed, which writes them under `tossed` and says so."""
    out = tree["tmp"] / "k.json"
    assert C.main(expect_args(tree, out)) == 0
    assert C.load_expectations(out)["kept"] == list(FZ.kept_now().values())
    out.unlink()
    capsys.readouterr()

    def refused(args: list[str], what: str) -> str:
        assert C.main(args) == 2
        err = capsys.readouterr().err
        assert what in err and not out.exists(), err
        return err
    refused(["expect", "--gate", str(tree["gate"]), "--out", str(out)], "error: give --renders keep3=")
    refused(expect_args(tree, out, "--keep3", str(tree["renders"]["keep3"])), "--renders keep3 is given twice")
    gate, text = tree["gate"], tree["gate"].read_text(encoding="utf-8")
    gate.write_text(text.replace(f"`{tree['redo1'] / 'd06_horn.ls'}`", f"`{tree['redo1'] / 'd06_groove.ls'}`"),
                    encoding="utf-8")
    refused(expect_args(tree, out), "is not lofi_rl.freeze.kept_now()")
    gate.write_text(text, encoding="utf-8")
    (redo,) = FZ.REDOS
    monkeypatch.setattr(FZ, "REDOS", ({**redo, "heading": "## Kept drafts (keep3 redo2)"},))
    refused(expect_args(tree, out), "error: REDOS[0] (keep3_redo1): its heading '## Kept drafts (keep3 redo2)'")
    monkeypatch.setattr(FZ, "REDOS", ())
    monkeypatch.setattr(FZ, "JOBS", {k: v for k, v in FZ.JOBS.items() if k != "keep3_redo1"})
    err = refused(expect_args(tree, out, rounds=("keep3",)), "is still kept, but Shresth tossed it")
    assert err.splitlines() == [f"error: {x}" for x in C.tossed_problems()] and len(C.tossed_problems()) == 4
    assert C.main(expect_args(tree, out, "--allow-tossed", rounds=("keep3",))) == 0
    doc = C.load_expectations(out)
    assert doc["allow_tossed"] is True and [t["brief"] for t in doc["tossed"]] == list(H.REOPENED)
    assert "--allow-tossed: 4 of them tossed at a sign-off listen" in capsys.readouterr().out


def test_a_kept_redo_file_is_read_by_its_path_never_the_same_stem_v1_file(tree):
    doc = C.expect_from_files(tree["renders"], draft=P.load())
    got = C.kept_drafts(doc)
    for b in H.REOPENED:
        stem = f"{b}_{H.PICKS[b]}"
        assert got[stem].path == (tree["redo1"] / f"{stem}.ls").resolve()
        assert got[stem].text == (tree["redo1"] / f"{stem}.ls").read_text() != (tree["v1"] / f"{stem}.ls").read_text()
    pal = load_palette(standin(tree["tmp"] / "lofi_rain_v1.yaml"), verify=False)
    va = C.vocabulary_check(pal, tree["v1"], doc)
    assert va["problems"] == [] and all(r["rendered"] for r in va["kept"].values())
    assert va["kept"]["d06_horn"]["sha256"] == C.file_sha256(tree["redo1"] / "d06_horn.ls")
    # the same document read by stem from drafts/v1 would be the tossed d06_horn: its sha256 is not the one rendered
    by_stem = {**doc, "loops": {**doc["loops"], "d06_horn": {**doc["loops"]["d06_horn"],
                                                             "file": str(tree["v1"] / "d06_horn.ls")}}}
    assert any("d06_horn.ls: sha256" in x and "is not the text keep3_redo1 rendered" in x
               for x in C.vocabulary_check(pal, tree["v1"], by_stem)["problems"])


def test_check_i_names_each_way_the_file_and_the_constants_differ(tree):
    """constants_problems (the report's (i), the sign-off check stage's kept_vs_constants) on copies of the tree's
    schema-2 document, each edited one way: each edit names its own problem, and only it."""
    doc = C.expect_from_files(tree["renders"], draft=P.load())
    assert C.constants_problems(doc) == []
    d01, d02 = str(tree["redo1"] / "d01_horn.ls"), doc["kept"][1]
    now = list(FZ.kept_now().values())

    def edited(edit) -> list[str]:
        d = copy.deepcopy(doc)
        edit(d)
        return C.constants_problems(d)
    assert edited(lambda d: d["loops"]["d01_horn"].update(source="keep3")) == [
        f"{d01}: keep3_kept10.json has it from keep3, but keep3_redo1 picked it"]
    assert edited(lambda d: d["sources"][1].update(job=11)) == [
        "keep3_kept10.json's source keep3_redo1 is job 11, not lofi_rl.freeze.JOBS['keep3_redo1'] (10)"]
    assert edited(lambda d: d["sources"].reverse()) == [
        "keep3_kept10.json's sources are ['keep3_redo1', 'keep3']; the rounds that supply the kept loops are "
        "['keep3', 'keep3_redo1']"]
    assert edited(lambda d: d["sources"].pop()) == [
        "keep3_kept10.json's sources are ['keep3']; the rounds that supply the kept loops are ['keep3', 'keep3_redo1']"]
    assert edited(lambda d: d.update(tossed=[{"brief": "d02", "file": d02, "signoffs": ["b1103-signoff"]}])) == [
        f"keep3_kept10.json records the tossed loops [('d02', '{d02}')]; the sign-off listens toss [] of the kept list "
        f"now: run `checks expect` again"]
    assert edited(lambda d: d["kept"].reverse()) == [
        f"keep3_kept10.json keeps {now[::-1]}, but lofi_rl.freeze.kept_now() keeps {now}: run `python -m "
        f"lofi_rl.freeze.checks expect` again"]


def test_the_report_with_the_synthetic_redo_round_passes(tree, monkeypatch):
    expect = tree["v1"] / C.EXPECT_NAME
    expect.write_text(C.expect_text(C.expect_from_files(tree["renders"], draft=P.load())), encoding="utf-8")
    pal = standin(tree["tmp"] / "lofi_rain_v1.yaml")
    rec = Recorder()
    doc = C.report(pal, drafts=tree["v1"], gate=tree["gate"], renders=tree["renders"], runner=rec)
    assert doc["ok"], doc["problems"]
    a, c, i = doc["a"], doc["c"], doc["i"]
    assert [(r["folder"], r["round"], r["drafts"], r["ok"]) for r in a["folders"]] == [
        (str(tree["v1"]), "keep3", 30, 30), (str(tree["redo1"]), "keep3_redo1", 12, 12)]
    assert a["folders"][1]["scope"] == list(H.REOPENED) and a["folders"][1]["against"] == [str(tree["v1"])]
    assert (c["drafts"], c["fullmatch"], c["kept"], c["kept_fullmatch"]) == (42, 42, 10, 10)
    assert doc["h"]["equal"] is True and doc["g"]["kept_table_matches"] is True and i["tossed_kept"] == []
    assert i["sources"] == ["keep3", "keep3_redo1"] and i["kept_now"] == FZ.kept_now()
    assert "(i) the kept list: 10 kept now (keep3 6, keep3_redo1 4)" in C.report_text(doc)
    # the same tree with the round forgotten: the four he tossed are kept again, and the report names exactly them
    monkeypatch.setattr(FZ, "REDOS", ())
    monkeypatch.setattr(FZ, "JOBS", {k: v for k, v in FZ.JOBS.items() if k != "keep3_redo1"})
    again = C.report(pal, drafts=tree["v1"], runner=rec)["i"]["problems"]
    assert [x for x in again if "still kept" in x] == C.tossed_problems() and len(C.tossed_problems()) == 4
    assert any("but lofi_rl.freeze.kept_now() keeps" in x for x in again)       # and the file is not that list


def test_a_redo_gate_table_that_disagrees_with_redos_fails_the_report(tree):
    expect = tree["v1"] / C.EXPECT_NAME
    expect.write_text(C.expect_text(C.expect_from_files(tree["renders"], draft=P.load())), encoding="utf-8")
    gate = tree["gate"]
    gate.write_text(gate.read_text().replace(f"`{tree['redo1'] / 'd07_groove.ls'}`",
                                             f"`{tree['redo1'] / 'd07_horn.ls'}`"), encoding="utf-8")
    doc = C.report(standin(tree["tmp"] / "lofi_rain_v1.yaml"), drafts=tree["v1"], gate=gate, runner=Recorder())
    assert doc["g"]["kept_table_matches"] is False and any("redo rounds' tables over it" in x for x in doc["problems"])
    assert any("the kept table differs from REDOS keep3_redo1's picks" in x
               for x in G.picks_problems(gate.read_text(), picks={}))


@pytest.mark.skipif(not FROZEN.is_file(), reason="palettes/lofi_rain_v1.yaml is the generator's (S2), not written yet")
def test_freeze_check_gate_compares_a_redo_rounds_table_with_redos(tmp_path, monkeypatch):
    """`python -m lofi_rl.freeze check --gate` (lofi_rl.freeze.palette.check, in this process so REDOS can hold a
    synthetic round) on a copy of the real gate file with a redo round's section added: the round after the last one
    REDOS records (keep3_redo1 while none is), so it never meets a recorded round's section."""
    k = 1 + max((int(r["key"].removeprefix("keep3_redo")) for r in FZ.REDOS), default=0)
    key, folder, job, rounds, listens = f"keep3_redo{k}", f"drafts/v1/redo{k}", 13710000 + k, len(FZ.REDOS), \
        len(FZ.SIGNOFFS)
    head = f"## Kept drafts (keep3 redo{k})"
    redo = {"key": key, "folder": folder, "batch": "b1105-keep3-redo", "job": job, "heading": head,
            "kept": {"d01": f"{folder}/d01_horn.ls", "d06": f"{folder}/d06_horn.ls"}}
    heading = f"{head}: voted 2026-11-05 (batch b1105-keep3-redo; job {job})"
    section = (f"{heading}\n\n| Brief | Energy, mood | Kept |\n|---|---|---|\n"
               f"| d01 | low, melancholy | `{folder}/d01_horn.ls` |\n"
               f"| d06 | mid, hopeful | `{folder}/d06_horn.ls` |\n\n")
    real = GATE.read_text(encoding="utf-8")
    assert real.count("\n## How to approve\n") == 1
    gate = tmp_path / "GATE_palette.md"
    gate.write_text(real.replace("\n## How to approve\n", "\n" + section + "## How to approve\n"), encoding="utf-8")
    assert F.check(FROZEN, gate=gate)[1] == [f"{heading}: a redo round the gate file records and lofi_rl.freeze.REDOS "
                                             f"lacks"]
    monkeypatch.setattr(FZ, "REDOS", (*FZ.REDOS, redo))
    monkeypatch.setitem(FZ.JOBS, key, job)
    _fp, bad, done = F.check(FROZEN, gate=gate)
    assert bad == [] and "(c) picks = " in done[-1]
    assert f"{rounds + 1} redo rounds, {listens} sign-off listens" in done[-1]
    gate.write_text(gate.read_text().replace(f"`{folder}/d06_horn.ls`", f"`{folder}/d06_groove.ls`"), encoding="utf-8")
    assert F.check(FROZEN, gate=gate)[1] == [
        f"{head}: the kept table differs from REDOS {key}'s picks: "
        f"[('d06', '{folder}/d06_groove.ls'), ('d06', '{folder}/d06_horn.ls')]"]


def test_drafts_folders_render_ids_and_a_folder_of_another_name(tree, tmp_path):
    folders, problems = C.drafts_folders(tree["v1"])
    assert problems == [] and [(f.path, f.scope, f.key) for f in folders] == [
        (tree["v1"], None, "keep3"), (tree["redo1"], tuple(H.REOPENED), "keep3_redo1")]
    ids = [C.render_id(f, tree["v1"]) for f in C.draft_files(folders)]
    assert len(ids) == len(set(ids)) == 42 and "d06_horn" in ids and "redo1_d06_horn" in ids
    (tree["v1"] / "redo2").mkdir()                                       # a later folder with nothing in it yet
    (tree["v1"] / "redo_old").mkdir()
    folders, problems = C.drafts_folders(tree["v1"])
    assert len(folders) == 2 and any("redo_old: not a redo folder's name" in p for p in problems)
    assert any("redo2: no <brief>_<persona>.ls draft" in p for p in problems)
    (tree["v1"] / "redo2" / "d07_horn.ls").write_text((tree["redo1"] / "d07_groove.ls").read_text(), encoding="utf-8")
    folders, _ = C.drafts_folders(tree["v1"])
    assert folders[2].scope == ("d07",) and folders[2].against == (tree["v1"], tree["redo1"]) and folders[2].key is None
    gone = dict(tree["consts"]["REDOS"][0], folder=str(tmp_path / "elsewhere" / "redo1"))
    assert any("is not a redo folder of" in p for p in C.drafts_folders(tree["v1"], redos=(gone,))[1])


# --------------------------------------------------------------------------------------------- the report
def test_the_report_on_a_standin_fails_on_exactly_the_tossed_loops(tmp_path):
    """The real tree and the real constants: every check passes but (i), which names the four loops tossed at
    b1103-signoff that no redo pick has replaced yet."""
    pal = standin(tmp_path / "lofi_rain_v1.yaml")
    rec = Recorder()
    renders = {s["key"]: ROOT / s["path"] for s in C.load_expectations(EXPECT)["sources"]
               if (ROOT / s["path"]).is_file()}
    doc = C.report(pal, gate=GATE, renders=renders, verify=True, runner=rec)
    assert doc["problems"] == doc["i"]["problems"] == C.tossed_problems()
    assert [p.split(":")[0] for p in doc["problems"]] == [b for b, _f, _s in FZ.tossed_kept()]
    assert [c[2:] for c in rec.calls] == [["--verify"], ["--gate", str(GATE)]]       # not sealed: no --sealed
    a, b, c = doc["a"], doc["b"], doc["c"]
    assert a["folder"]["ok"] == a["folder"]["drafts"] == real_drafts() and len(a["kept"]) == 10
    assert all(r["rendered"] and not r["problems"] and r["repairs"] == [] for r in a["kept"].values())
    assert b["count"] == 9 and b["identical"] is True and b["identical_to"] == "dev_gu"
    assert (c["chars"], c["sha256"]) == (C.REGEX_CHARS, C.REGEX_SHA256) == (3408, "5c081828ac50718edc4db82c2a855ea52210"
                                                                                  "debb375280ba01e8442de60f8558")
    assert c["fullmatch"] == c["drafts"] == real_drafts() and c["kept_fullmatch"] == 10
    assert doc["g"]["kept_table_matches"] is True
    assert doc["h"].get("equal", True) is True
    text = C.report_text(doc)
    assert "median off target (LU)" in text
    assert text.splitlines()[-1].startswith("ok:" if doc["ok"] else f"{len(doc['problems'])} problem(s):")


def test_the_report_refuses_constants_that_break_their_own_rules(tmp_path, monkeypatch, capsys):
    """REDOS and SIGNOFFS are read by every check of the report, so a malformed entry is bad input (exit 2)."""
    pal = standin(tmp_path / "lofi_rain_v1.yaml")
    monkeypatch.setattr(FZ, "REDOS", ({"key": "keep3_redo1", "folder": "drafts/v1/redo1"},))
    with pytest.raises(C.ChecksError, match="a redo round holds exactly"):
        C.report(pal, runner=Recorder())
    monkeypatch.setattr(C, "freeze_check", Recorder())
    assert C.main(["report", "--palette", str(pal)]) == 2
    assert "error: REDOS[0]: a redo round holds exactly" in capsys.readouterr().err
    monkeypatch.setattr(FZ, "REDOS", ())
    monkeypatch.setattr(FZ, "SIGNOFFS", ({"batch": "b1103-signoff", "job": 13701026, "tossed": ("drafts/v1/x.ls",)},))
    with pytest.raises(C.ChecksError, match="which neither keep3 nor a redo round kept"):
        C.report(pal, runner=Recorder())


def test_the_bus_table_is_the_kept_loops_and_names_a_bus_over_one_lu():
    """(e) of the committed file, whichever rounds supply its loops: each bus's median, range and over flag, the keys
    per instrument and against each loop's own sax, and the instruments played."""
    expect = C.load_expectations(EXPECT)
    loops = expect["loops"]
    e = C.bus_table(expect)
    for b in C.BUSES:
        offs = [x["buses"][b]["off_lu"] for x in loops.values()]
        assert e["buses"][b]["median_off_lu"] == statistics.median(offs)
        assert (e["buses"][b]["min_off_lu"], e["buses"][b]["max_off_lu"]) == (min(offs), max(offs))
        assert e["buses"][b]["over"] is (abs(statistics.median(offs)) > C.OVER_LU)
    keys = sorted({x["keys"] for x in loops.values()})
    assert {i: v["median"] for i, v in e["keys_by_instrument"].items()} == {
        i: statistics.median(x["buses"]["keys"]["off_lu"] for x in loops.values() if x["keys"] == i) for i in keys}
    k, s = ({r: x["buses"][b] for r, x in loops.items()} for b in ("keys", "sax"))
    assert e["keys_vs_own_sax"]["median"] == statistics.median(
        (k[r]["lu"] - s[r]["lu"]) - (k[r]["target_lu"] - s[r]["target_lu"]) for r in loops)
    assert e["uses"]["keys"] == {i: sorted(r for r, x in loops.items() if x["keys"] == i) for i in keys}


def keep3_ten() -> dict:
    """keep3's ten loops as a keep3_kept10.json document: the committed file while its kept list is keep3's
    (2026-11-03), else rebuilt from keep3's renders.json where runs/ holds it (skip otherwise)."""
    doc = C.load_expectations(EXPECT)
    if [s["key"] for s in doc["sources"]] == ["keep3"] and doc["kept"] == list(FZ.KEPT.values()):
        return doc
    if not K3R.is_file():
        pytest.skip("the kept list is no longer keep3's ten, and runs/ does not hold keep3's renders.json")
    return C.expectations([C.read_source("keep3", K3R, job=13656049)],
                          kept={b: (f, "keep3") for b, f in FZ.KEPT.items()}, draft=P.load())


def test_the_bus_table_of_keep3s_ten_puts_the_keys_to_shresth():
    """The freeze spec's §4.1 table and Q1 (2026-11-03), read from keep3's renders
    (runs/drafts-13656049/summary/renders.json): the keys bus is the one more than OVER_LU off."""
    expect = keep3_ten()
    assert sorted(expect["loops"]) == sorted(KEPT)
    e = C.bus_table(expect)
    assert {b: round(e["buses"][b]["median_off_lu"], 3) for b in C.BUSES} == {"sax": 0.137, "keys": -1.220,
                                                                             "bass": 0.207, "drums": 0.499}
    assert [b for b in C.BUSES if e["buses"][b]["over"]] == ["keys"]
    assert {i: round(v["median"], 2) for i, v in e["keys_by_instrument"].items()} == {"wurli": -2.42, "rhodes": -1.41,
                                                                                       "piano": 0.62}
    assert round(e["keys_vs_own_sax"]["median"], 2) == -0.94
    assert {k: len(v) for k, v in e["uses"]["keys"].items()} == {"piano": 4, "rhodes": 3, "wurli": 3}
    assert e["uses"]["kits"]["studio"] == ["d01_chords", "d10_groove"] and len(e["uses"]["kits"]["jazz"]) == 8


def test_the_report_names_a_regex_that_is_not_the_frozen_vocabulary(tmp_path):
    def more_keys(keys):
        return {**keys, "organ": copy.deepcopy(keys["piano"])}
    pal = standin(tmp_path / "lofi_rain_v1.yaml", keys=more_keys)
    doc = C.report(pal, runner=Recorder())
    assert not doc["ok"] and len(doc["c"]["problems"]) == 1 and "characters with sha256" in doc["c"]["problems"][0]
    assert doc["a"]["problems"] == [] and doc["b"]["problems"] == []


def test_check_folder_checks_the_drafts_against_the_palette_it_is_given(tmp_path):
    from lofi_rl.drafts.check import check_folder
    pal = load_palette(standin(tmp_path / "lofi_rain_v1.yaml", keys=lambda k: {n: v for n, v in k.items()
                                                                              if n != "piano"}), verify=False)
    rep = check_folder(ROOT / "drafts" / "v1", palette=pal)
    failing = sorted(d.file for d in rep.drafts if not d.ok)
    uses_piano = sorted(f.name for f in (ROOT / "drafts" / "v1").glob("*.ls") if "keys: piano" in f.read_text())
    assert failing == uses_piano and uses_piano
    assert all("gate" in d.problems for d in rep.drafts if not d.ok)
    assert check_folder(ROOT / "drafts" / "v1").ok                      # the draft palette, as before


def test_the_report_names_a_freeze_check_that_fails_and_a_stale_kept_list(tmp_path):
    """A copy of the real gate file with one kept loop's row naming another draft of its brief: the row of the round
    that supplies d04 now (keep3's table, or the redo round's that picked it)."""
    pal = standin(tmp_path / "lofi_rain_v1.yaml")
    kept = FZ.kept_now()["d04"]
    other = str(Path(kept).with_name("d04_chords.ls" if Path(kept).name != "d04_chords.ls" else "d04_horn.ls"))
    lines = GATE.read_text(encoding="utf-8").splitlines(keepends=True)
    (row,) = [i for i, x in enumerate(lines) if x.startswith("| d04 |") and f"`{kept}`" in x]
    lines[row] = lines[row].replace(f"`{kept}`", f"`{other}`")
    gate = tmp_path / "gate.md"
    gate.write_text("".join(lines), encoding="utf-8")
    doc = C.report(pal, gate=gate, runner=Recorder(3, "problem: a pick"))
    assert not doc["ok"] and doc["g"]["kept_table_matches"] is False
    assert any("freeze check --gate exited 3" in x for x in doc["problems"])
    assert any(x.startswith("keep3_kept10.json keeps") and "but the gate file's keep3 table" in x and other in x
               for x in doc["problems"])


def test_the_cli_prints_the_report_and_refuses_bad_input(tmp_path, capsys, monkeypatch):
    pal = standin(tmp_path / "lofi_rain_v1.yaml")
    monkeypatch.setattr(C, "freeze_check", Recorder())
    assert C.main(["report", "--palette", str(pal)]) == (3 if FZ.tossed_kept() else 0)
    out = capsys.readouterr().out
    keys = C.bus_table(C.load_expectations(EXPECT))["buses"]["keys"]["median_off_lu"]     # -1.220 for keep3's ten
    assert "(e) bus levels of the kept loops" in out and f"keys   {keys:+.3f}" in out and "(g) picks: not run" in out
    assert [x for x in out.splitlines() if x.startswith("problem: ")] == [f"problem: {x}" for x in C.tossed_problems()]
    assert C.main(["report", "--palette", str(tmp_path / "nowhere.yaml")]) == 2
    assert C.main(["report", "--palette", str(pal), "--drafts", str(tmp_path)]) == 2      # no keep3_kept10.json
    assert C.main(["report", "--palette", str(pal), "--renders", "keep3"]) == 2           # not KEY=PATH


def test_the_report_checks_a_sealed_file_as_sealed(tmp_path):
    """After the seal, `frozen:` is his date and the gate file holds his line: both freeze checks get --sealed, so the
    check holds the date to his line instead of refusing it. The recorder stands in for the check here, so nothing
    reads the line; the last test runs the check for real on copies of the real files."""
    pal = standin(tmp_path / "lofi_rain_v1.yaml", frozen=lambda _: DAY)
    gate = tmp_path / "GATE_palette.md"
    gate.write_text(his_line(GATE.read_text(encoding="utf-8"), DAY), encoding="utf-8")
    rec = Recorder()
    doc = C.report(pal, gate=gate, keep3=K3R if K3R.is_file() else None, verify=True, runner=rec)
    assert doc["problems"] == C.tossed_problems()                       # the tossed loops, as on the file unsealed
    assert doc["palette"]["frozen"] == load_palette(pal, verify=False).frozen == DAY
    assert rec.calls == [["--palette", str(pal), "--verify", "--sealed"],
                         ["--palette", str(pal), "--gate", str(gate), "--sealed"]]
    assert (doc["d"]["cmd"], doc["g"]["cmd"]) == ("python -m lofi_rl.freeze check --verify --sealed",
                                                  "python -m lofi_rl.freeze check --gate --sealed")
    assert doc["g"]["kept_table_matches"] is True


@pytest.mark.skipif(not FROZEN.is_file(), reason="palettes/lofi_rain_v1.yaml is the generator's (S2), not written yet")
def test_the_report_on_the_frozen_palette_itself():
    doc = C.report(FROZEN, gate=GATE)                       # the freeze check runs for real here
    assert doc["problems"] == C.tossed_problems() and doc["g"]["exit"] == 0
    assert load_palette(FROZEN, verify=False).id == "lofi_rain_v1"
    assert yaml.safe_load(FROZEN.read_text())["frozen"] in (None, load_palette(FROZEN, verify=False).frozen)


@pytest.mark.skipif(not FROZEN.is_file(), reason="palettes/lofi_rain_v1.yaml is the generator's (S2), not written yet")
def test_the_report_on_the_frozen_palette_as_the_seal_leaves_it(tmp_path):
    """Copies of the real files as the seal leaves them: the palette rebuilt with his date (the generator's rebuild,
    which changes only the `frozen:` line) and his line under `## Approval`. The freeze check runs for real, so the
    report's (g) passes only when it passes --sealed; the same call against a line of another date fails. The kept
    loops tossed at a sign-off listen are (i)'s, as on the file unsealed."""
    sealed = tmp_path / "lofi_rain_v1.yaml"
    sealed.write_text(F.rebuild_text(F.load(FROZEN), frozen=DAY), encoding="utf-8")
    old, new = FROZEN.read_text(encoding="utf-8").splitlines(), sealed.read_text(encoding="utf-8").splitlines()
    assert len(old) == len(new) and [b for a, b in zip(old, new) if a != b] in ([], [f"frozen: '{DAY}'"])
    gate, other = tmp_path / "GATE_palette.md", tmp_path / "GATE_other.md"
    gate.write_text(his_line(GATE.read_text(encoding="utf-8"), DAY), encoding="utf-8")
    other.write_text(his_line(GATE.read_text(encoding="utf-8"), OTHER_DAY), encoding="utf-8")
    calls = []

    def runner(args, root=ROOT):
        calls.append([str(a) for a in args])
        return C.freeze_check(args, root)
    doc = C.report(sealed, gate=gate, runner=runner)
    assert doc["problems"] == C.tossed_problems()
    assert doc["palette"]["frozen"] == DAY and doc["g"]["exit"] == 0
    assert calls == [["--palette", str(sealed), "--gate", str(gate), "--sealed"]]
    rc, out = C.freeze_check(["--palette", sealed, "--gate", other, "--sealed"])
    assert rc == 3 and f"frozen is '{DAY}', the APPROVED line's date is {OTHER_DAY}" in out
