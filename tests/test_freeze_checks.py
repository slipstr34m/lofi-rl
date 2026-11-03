"""The checks before the sign-off (lofi_rl.freeze.checks; docs/DRAFTS.md §7; the freeze spec §4.1).

drafts/v1/keep3_kept10.json is keep3's record of the 10 kept loops, rebuilt here from keep3's summary when runs/ holds
it. The laptop report runs on a stand-in of the frozen palette made from palettes/draft_day2.yaml the way the generator
makes the real file (tests/test_freeze_signoff.py), and on palettes/lofi_rain_v1.yaml itself once it is written, as it
stands and as the seal leaves it (copies). The freeze check's command (`python -m lofi_rl.freeze check`) is replaced by
a recorder, except in the last two tests. Nothing is rendered and nothing is claimed about sound."""
import copy
import json
import re
import statistics
from pathlib import Path

import pytest
import yaml

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


# --------------------------------------------------------------------------------------------- keep3_kept10.json
@pytest.mark.skipif(not K3R.is_file(), reason="runs/ is not here (it is never committed or synced)")
def test_the_committed_expectations_are_what_expect_writes_from_keep3(tmp_path, capsys):
    doc = C.expect_from_file(K3R, job=13656049, kept=C.gate_kept(GATE), draft=P.load())
    assert C.expect_text(doc) == EXPECT.read_text(encoding="utf-8")
    out = tmp_path / "k.json"
    assert C.main(["expect", "--keep3", str(K3R), "--job", "13656049", "--out", str(out)]) == 0
    assert out.read_bytes() == EXPECT.read_bytes()
    assert "10 kept loops of runs/drafts-13656049/summary/renders.json" in capsys.readouterr().out


def test_the_committed_expectations_hold_the_ten_texts_he_kept():
    doc = C.load_expectations(EXPECT)
    assert doc["kept"] == [f"drafts/v1/{k}.ls" for k in KEPT] == C.gate_kept(GATE)
    assert doc["keep3"] == {"path": "runs/drafts-13656049/summary/renders.json", "job": 13656049,
                            "sha256": doc["keep3"]["sha256"]} and len(doc["keep3"]["sha256"]) == 64
    assert doc["s_star"] == P.load().levels["S_star"]
    for rid, e in doc["loops"].items():
        assert C.file_sha256(ROOT / e["file"]) == e["draft_sha256"]
        for b in C.BUSES:                                   # keep3's lu was post_lufs - S*, as drafts/render.py has it
            assert e["buses"][b]["post_lufs"] - doc["s_star"] == e["buses"][b]["lu"]
    assert C.kept_problems(doc, GATE) == []


def k3_doc(tmp_path) -> tuple[dict, list[str]]:
    """A keep3 renders document for two fixture drafts in tmp_path (the committed records' shape)."""
    loops = C.load_expectations(EXPECT)["loops"]
    renders = {}
    for rid in ("d01_chords", "d05_groove"):
        f = tmp_path / f"{rid}.ls"
        f.write_text((ROOT / "drafts" / "v1" / f"{rid}.ls").read_text(), encoding="utf-8")
        e = loops[rid]
        renders[rid] = {**{k: e[k] for k in C.LOOP_KEYS}, "file": str(f), "status": "ok",
                        "palette": {"label": e["palette"]}, "buses": copy.deepcopy(e["buses"])}
    return {"renders": renders}, [renders[r]["file"] for r in renders]


def test_expectations_refuse_what_keep3_did_not_render(tmp_path):
    doc, kept = k3_doc(tmp_path)
    dp = P.load()
    ok = C.expectations(doc, renders_sha256="a" * 64, renders_path="x/drafts-9/summary/renders.json", job=9, kept=kept,
                        draft=dp, root=tmp_path)
    assert sorted(ok["loops"]) == ["d01_chords", "d05_groove"] and ok["s_star"] == dp.levels["S_star"]

    def refused(d, kept=kept, job=9, path="x/drafts-9/summary/renders.json"):
        with pytest.raises(C.ChecksError) as err:
            C.expectations(d, renders_sha256="a" * 64, renders_path=path, job=job, kept=kept, draft=dp, root=tmp_path)
        return err.value.errors
    assert refused(doc, kept=kept + [str(tmp_path / "d02_chords.ls")])[0].endswith("0 keep3 records, not one")
    bad = copy.deepcopy(doc)
    bad["renders"]["d05_groove"]["status"] = "level"
    assert "keep3's render is level" in refused(bad)[0]
    (tmp_path / "d01_chords.ls").write_text("vamp: edited\n", encoding="utf-8")
    assert "not the one keep3 rendered" in refused(doc)[0]
    (tmp_path / "d01_chords.ls").write_text((ROOT / "drafts" / "v1" / "d01_chords.ls").read_text(), encoding="utf-8")
    bad = copy.deepcopy(doc)
    bad["renders"]["d01_chords"]["buses"]["keys"]["lu"] += 1e-9                     # another S* than the draft's
    assert "was not computed with S*" in refused(bad)[0]
    assert "is not job 8's" in refused(doc, job=8)[0]


def test_load_expectations_refuses_a_file_that_is_not_one(tmp_path):
    good = json.loads(EXPECT.read_text())
    cases = {"schema": lambda d: d.update(schema="lofi-rl/other/1"),
             "loops": lambda d: d["loops"].pop("d05_groove"),
             "bus": lambda d: d["loops"]["d01_chords"]["buses"]["keys"].pop("off_lu"),
             "keep3": lambda d: d["keep3"].update(sha256="nope")}
    for name, edit in cases.items():
        d = copy.deepcopy(good)
        edit(d)
        f = tmp_path / f"{name}.json"
        f.write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises(C.ChecksError):
            C.load_expectations(f)


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


# --------------------------------------------------------------------------------------------- the report
def test_the_report_passes_on_a_standin_of_the_frozen_palette(tmp_path):
    pal = standin(tmp_path / "lofi_rain_v1.yaml")
    rec = Recorder()
    doc = C.report(pal, gate=GATE, keep3=K3R, verify=True, runner=rec)
    assert doc["ok"], doc["problems"]
    assert [c[2:] for c in rec.calls] == [["--verify"], ["--gate", str(GATE)]]       # not sealed: no --sealed
    a, b, c = doc["a"], doc["b"], doc["c"]
    assert a["folder"]["ok"] == a["folder"]["drafts"] == 30 and len(a["kept"]) == 10
    assert all(r["keep3"] and not r["problems"] and r["repairs"] == [] for r in a["kept"].values())
    assert b["count"] == 9 and b["identical"] is True and b["identical_to"] == "dev_gu"
    assert (c["chars"], c["sha256"]) == (C.REGEX_CHARS, C.REGEX_SHA256) == (3408, "5c081828ac50718edc4db82c2a855ea52210"
                                                                                  "debb375280ba01e8442de60f8558")
    assert c["fullmatch"] == c["drafts"] == 30 and c["kept_fullmatch"] == 10
    assert doc["g"]["kept_table_matches"] is True
    assert doc["h"].get("equal", True) is True
    text = C.report_text(doc)
    assert "median off target (LU)" in text and text.splitlines()[-1].startswith("ok:")


def test_the_bus_table_is_keep3s_and_puts_the_keys_to_shresth():
    expect = C.load_expectations(EXPECT)
    e = C.bus_table(expect)
    for b in C.BUSES:
        offs = [expect["loops"][r]["buses"][b]["off_lu"] for r in KEPT]
        assert e["buses"][b]["median_off_lu"] == statistics.median(offs)
        assert (e["buses"][b]["min_off_lu"], e["buses"][b]["max_off_lu"]) == (min(offs), max(offs))
    # the freeze spec's §4.1 table and Q1, read from keep3's renders (runs/drafts-13656049/summary/renders.json)
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
    pal = standin(tmp_path / "lofi_rain_v1.yaml")
    gate = tmp_path / "gate.md"
    gate.write_text(GATE.read_text().replace("`drafts/v1/d04_horn.ls`", "`drafts/v1/d04_chords.ls`"), encoding="utf-8")
    doc = C.report(pal, gate=gate, runner=Recorder(3, "problem: a pick"))
    assert not doc["ok"] and doc["g"]["kept_table_matches"] is False
    assert any("freeze check --gate exited 3" in x for x in doc["problems"])
    assert any("but the gate file's keep3 table keeps" in x for x in doc["problems"])


def test_the_cli_prints_the_report_and_refuses_bad_input(tmp_path, capsys, monkeypatch):
    pal = standin(tmp_path / "lofi_rain_v1.yaml")
    monkeypatch.setattr(C, "freeze_check", Recorder())
    assert C.main(["report", "--palette", str(pal)]) == 0
    out = capsys.readouterr().out
    assert "(e) bus levels of the kept loops" in out and "keys   -1.220" in out and "(g) picks: not run" in out
    assert C.main(["report", "--palette", str(tmp_path / "nowhere.yaml")]) == 2
    assert C.main(["report", "--palette", str(pal), "--drafts", str(tmp_path)]) == 2      # no keep3_kept10.json


def test_the_report_checks_a_sealed_file_as_sealed(tmp_path):
    """After the seal, `frozen:` is his date and the gate file holds his line: both freeze checks get --sealed, so the
    check holds the date to his line instead of refusing it. The recorder stands in for the check here, so nothing
    reads the line; the last test runs the check for real on copies of the real files."""
    pal = standin(tmp_path / "lofi_rain_v1.yaml", frozen=lambda _: DAY)
    gate = tmp_path / "GATE_palette.md"
    gate.write_text(his_line(GATE.read_text(encoding="utf-8"), DAY), encoding="utf-8")
    rec = Recorder()
    doc = C.report(pal, gate=gate, keep3=K3R, verify=True, runner=rec)
    assert doc["ok"], doc["problems"]
    assert doc["palette"]["frozen"] == load_palette(pal, verify=False).frozen == DAY
    assert rec.calls == [["--palette", str(pal), "--verify", "--sealed"],
                         ["--palette", str(pal), "--gate", str(gate), "--sealed"]]
    assert (doc["d"]["cmd"], doc["g"]["cmd"]) == ("python -m lofi_rl.freeze check --verify --sealed",
                                                  "python -m lofi_rl.freeze check --gate --sealed")
    assert doc["g"]["kept_table_matches"] is True


@pytest.mark.skipif(not FROZEN.is_file(), reason="palettes/lofi_rain_v1.yaml is the generator's (S2), not written yet")
def test_the_report_on_the_frozen_palette_itself():
    doc = C.report(FROZEN, gate=GATE)                       # the freeze check runs for real here
    assert doc["ok"], doc["problems"]
    assert load_palette(FROZEN, verify=False).id == "lofi_rain_v1"
    assert yaml.safe_load(FROZEN.read_text())["frozen"] in (None, load_palette(FROZEN, verify=False).frozen)


@pytest.mark.skipif(not FROZEN.is_file(), reason="palettes/lofi_rain_v1.yaml is the generator's (S2), not written yet")
def test_the_report_on_the_frozen_palette_as_the_seal_leaves_it(tmp_path):
    """Copies of the real files as the seal leaves them: the palette rebuilt with his date (the generator's rebuild,
    which changes only the `frozen:` line) and his line under `## Approval`. The freeze check runs for real, so the
    report passes only when it passes --sealed; the same call against a line of another date fails."""
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
    assert doc["ok"], doc["problems"]
    assert doc["palette"]["frozen"] == DAY and doc["g"]["exit"] == 0
    assert calls == [["--palette", str(sealed), "--gate", str(gate), "--sealed"]]
    rc, out = C.freeze_check(["--palette", sealed, "--gate", other, "--sealed"])
    assert rc == 3 and f"frozen is '{DAY}', the APPROVED line's date is {OTHER_DAY}" in out
