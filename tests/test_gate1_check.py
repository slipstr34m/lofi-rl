"""The Phase-1 gate's check (lofi_rl.gate1.check; docs/GATE1.md): D-020's amended validity list with D-022's two
swaps, condition by condition, the bar's first half, the two verbatim readouts, the refusals of files that are not
one run's own, and arm B (runs/junk-13640325, dev_gu), which must fail exactly the swapped conditions. Synthetic runs
come from tests/helpers_gate1.py; nothing here renders or listens."""
import hashlib
import json
from pathlib import Path

import pytest

from helpers_gate1 import STAMP, copy_run, edit_json, edit_rows, make_run, verbatim_of
from lofi_rl import gate1 as G
from lofi_rl import prompt_v2, smoke
from lofi_rl.gate1 import check as C
from lofi_rl.gate1 import cli
from lofi_rl.grammar import loopscript_regex
from lofi_rl.palette import load_palette
from lofi_rl.rl import junk as J
from lofi_rl.ui import votes as V

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "palettes" / "lofi_rain_v1.yaml"
ARM_B = ROOT / "runs" / "junk-13640325"
V0 = ROOT / "runs" / "junk-13632676" / "summary" / "junk.json"
DEV_GU_SHA = "b2e4a340b9387e49f896e235c0f1f7f3c9339308ddc307b343c2d96e9eb8fab9"
DEV_GU_REGEX = "1412f11f9ebdc520b50c263eda14be12829951357c17f43496ab2ddf5235e64e"
V1_TEMPLATE = "7aceb2bc0d7339a7d3c9af7bf0d808ee156d4f7a8684e0393d923940fa5c1e68"
SWAPPED = ["palette.sha256", "constrained.regex_sha256", "constrained.in_language.regex_sha256"]


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("gate1-check") / "junk-13783877"
    make_run(root, k=50, flacs=False)
    return root


@pytest.fixture
def run_dir(base, tmp_path) -> Path:
    return copy_run(base, tmp_path / "junk-13783877")


def failed(run_dir: Path) -> list[str]:
    return C.evaluate(G.load_run(run_dir))["validity"]["failed"]


def set_path(doc: dict, path: str, value) -> None:
    *head, last = path.split(".")
    for part in head:
        doc = doc[part]
    doc[last] = value


def del_path(doc: dict, path: str) -> None:
    *head, last = path.split(".")
    for part in head:
        doc = doc[part]
    del doc[last]


# --------------------------------------------------------------------------------------------- the registered values
def test_the_registered_values_are_what_the_code_makes_today():
    """Each value is written out as its decision states it; a drift in the code shows up here, never as a moved bar."""
    pal = load_palette(FROZEN, verify=False)
    assert G.FROZEN_SHA256 == pal.sha256 == hashlib.sha256(FROZEN.read_bytes()).hexdigest()       # D-022, B
    assert G.REGEX_SHA256 == hashlib.sha256(loopscript_regex(pal).encode("utf-8")).hexdigest()    # D-022
    assert G.TEMPLATE_SHA256 == prompt_v2.template_sha256()                                       # D-020, arm B
    assert G.MODEL_REVISION == smoke.QWEN[1]                                                      # D-019
    assert dict(G.V0_SAMPLING) == J.SAMPLING and list(G.V0_SAMPLING) == list(J.SAMPLING)          # D-019, D-020
    assert (G.N, G.BRIEF_SEED, G.CLIPS) == (J.DEFAULT_N, J.DEFAULT_SEED, J.RENDER_K) == (500, 2026, 50)
    assert G.PASS_NOT_COPY_BAR == 300 == 0.6 * G.N and G.FINE_BAR == 15 == 0.3 * G.CLIPS        # CLAUDE.md §9
    assert G.TIERS == V.TIERS and G.FINE_OR_BETTER == ("fine", "good", "great")
    assert J.DEFAULT_PALETTE == FROZEN                                    # the gate job's default palette (D-022)


def test_v0_sampling_is_the_v0_runs_own():
    """D-020: `sampling` equals runs/junk-13632676's, read from its junk.json when make pull has brought it."""
    if not V0.is_file():
        pytest.skip(f"{V0.relative_to(ROOT)} is not on this machine (make pull brings it to the laptop)")
    v0 = json.loads(V0.read_text())["sampling"]
    assert json.dumps(v0) == json.dumps(dict(G.V0_SAMPLING))


# --------------------------------------------------------------------------------------------- validity
def test_a_run_at_every_registered_value_is_valid_and_meets_the_first_half(run_dir):
    doc = C.evaluate(G.load_run(run_dir))
    v = doc["validity"]
    assert v["valid"] and v["failed"] == [] and v["held"] == v["of"] == 27
    assert [c["group"] for c in v["conditions"]] == (["Complete"] * 5 + ["As registered"] * 10 + ["Constrained"] * 6
                                                     + ["What the regex guarantees"] * 5 + ["One stamp"])
    assert {c["name"] for c in v["conditions"] if c["source"] == "D-022"} == set(SWAPPED)
    assert doc["bar"]["pass_not_copy"] == {"count": 446, "of": 500, "need": 300, "rate": 0.892,
                                           "ci95": [0.8617, 0.9163], "method": "wilson", "met": True}
    assert (doc["verdict"], doc["exit"]) == ("valid, first half met", 0)
    assert (doc["job"], doc["palette"]["label"]) == ("13783877", f"lofi_rain_v1@{G.FROZEN_SHA256[:8]}")


# every condition, moved off its registered value: exactly that condition fails, by its name
OFF = [
    ("n", "n", 500.0),                                            # the count is right, the JSON type is not
    ("requested", "requested", 501),
    ("partial", "partial", True),
    ("partial", "partial", 0),                                    # 0 is not false
    ("unreadable_lines", "unreadable_lines", 2),
    ("generate.stopped", "generate.stopped", "deadline"),
    ("generate.model.revision", "generate.model.revision", "0" * 40),
    ("briefs.seed", "briefs.seed", 2027),
    ("briefs.seed", "briefs.seed", True),
    ("sampling", "sampling.temperature", 0.9),
    ("sampling", "sampling.max_tokens", 512.0),
    ("palette.sha256", "palette.sha256", DEV_GU_SHA),
    ("palette.matches_generate", "palette.matches_generate", None),
    ("prompt.version", "prompt.version", "v1"),
    ("prompt.template_sha256", "prompt.template_sha256", V1_TEMPLATE),
    ("prompt.template_matches_generate", "prompt.template_matches_generate", False),
    ("prompt.rows_matching", "prompt.rows_matching", 499),
    ("prompt.example_mismatch", "prompt.example_mismatch", 1),
    ("constrained.enabled", "constrained.enabled", False),
    ("constrained.regex_sha256", "constrained.regex_sha256", DEV_GU_REGEX),
    ("constrained.in_language.regex_sha256", "constrained.in_language.regex_sha256", DEV_GU_REGEX),
    ("constrained.in_language.matches_generate", "constrained.in_language.matches_generate", False),
    ("constrained.in_language.stop.out_of_language", "constrained.in_language.stop.out_of_language", 1),
    ("constrained.in_language.other.n", "constrained.in_language.other.n", 1),
    ("failures.G1", "failures.G1", 1),
    ("failures.G3", "failures.G3", 2),
    ("failures.G4", "failures.G4", 1),
    ("failures.G5", "failures.G5", 1),
    ("failures.crash", "failures.crash", 1),
    (C.STAMP_NAME, "score_stamp", "commit=0123456789ab dirty=0 diff=- at=2026-11-06T19:00:00Z"),
    (C.STAMP_NAME, "score_stamp", None),
]


@pytest.mark.parametrize("name, path, value", OFF, ids=[f"{p}={v!r}" for _, p, v in OFF])
def test_each_condition_fails_by_its_own_name(run_dir, name, path, value):
    edit_json(run_dir / "summary" / "junk.json", lambda d: set_path(d, path, value))
    if path == "palette.sha256":           # render read the palette score read, so clips.json records it too
        edit_json(run_dir / "summary" / "clips.json", lambda d: set_path(d, path, value))
    doc = C.evaluate(G.load_run(run_dir))
    assert doc["validity"]["failed"] == [name] and doc["validity"]["held"] == 26
    assert (doc["verdict"], doc["exit"]) == ("invalid", 3)
    cond = next(c for c in doc["validity"]["conditions"] if c["name"] == name)
    assert cond["ok"] is False and cond["shows"] != cond["needs"]


def test_the_stamp_needs_dirty_0_and_one_commit_in_both(run_dir):
    dirty = "commit=ac4acfb0d1e2 dirty=1 diff=1a2b3c4d5e6f at=2026-11-06T19:00:00Z"
    edit_json(run_dir / "summary" / "junk.json", lambda d: (set_path(d, "generate.host.stamp", dirty),
                                                             set_path(d, "score_stamp", dirty)))
    assert failed(run_dir) == [C.STAMP_NAME]
    later = "commit=ac4acfb0d1e2 dirty=0 diff=- at=2026-11-06T19:20:00Z"     # another sync time, the same commit
    edit_json(run_dir / "summary" / "junk.json", lambda d: (set_path(d, "generate.host.stamp", STAMP),
                                                             set_path(d, "score_stamp", later)))
    doc = C.evaluate(G.load_run(run_dir))
    assert doc["validity"]["failed"] == []                    # D-020 compares the commit and dirty fields only
    cond = next(c for c in doc["validity"]["conditions"] if c["name"] == C.STAMP_NAME)
    assert "generate commit=ac4acfb0d1e2" in cond["shows"] and "score commit=ac4acfb0d1e2" in cond["shows"]


@pytest.mark.parametrize("path, names", [
    ("generate.stopped", ["generate.stopped"]),                # absent is not null
    ("constrained", ["constrained.enabled", "constrained.regex_sha256", "constrained.in_language.regex_sha256",
                     "constrained.in_language.matches_generate", "constrained.in_language.stop.out_of_language",
                     "constrained.in_language.other.n"]),
    ("sampling", ["sampling"]),
    ("score_stamp", [C.STAMP_NAME]),
])
def test_a_missing_field_fails_and_shows_missing(run_dir, path, names):
    edit_json(run_dir / "summary" / "junk.json", lambda d: del_path(d, path))
    doc = C.evaluate(G.load_run(run_dir))
    assert doc["validity"]["failed"] == names
    first = next(c for c in doc["validity"]["conditions"] if c["name"] == names[0])
    assert "missing" in first["shows"]


def test_an_unconstrained_run_fails_the_constrained_group(run_dir):
    """A run without --constrained has constrained null (junk.constrained_summary): nothing in that group holds."""
    edit_json(run_dir / "summary" / "junk.json", lambda d: set_path(d, "constrained", None))
    assert failed(run_dir) == ["constrained.enabled", "constrained.regex_sha256", "constrained.in_language.regex_sha256",
                               "constrained.in_language.matches_generate",
                               "constrained.in_language.stop.out_of_language", "constrained.in_language.other.n"]


def test_the_backend_the_render_step_and_g7_are_shown_never_judged(run_dir):
    def off(d):
        d["constrained"]["backend"] = {"requested": "xgrammar", "pin_accepted": False, "backend": "unknown",
                                       "source": None, "applied": None}
        d["failures"]["G7"] = 3
        d["render"] = {"attempted": 50, "rendered": 47, "G7": 3, "other_gates": 0, "errors": 0, "skipped": 0,
                       "seed": 2026, "k": 50}
    edit_json(run_dir / "summary" / "junk.json", off)
    doc = C.evaluate(G.load_run(run_dir))
    assert doc["validity"]["valid"] and doc["exit"] == 0
    text = C.format_text(doc)
    assert "backend unknown (pin false, applied null; recorded, not judged)" in text
    assert "render step (recorded, not judged): attempted 50, rendered 47, G7 3" in text


# --------------------------------------------------------------------------------------------- the bar's first half
def test_under_300_is_valid_and_under_the_bar(tmp_path):
    root = tmp_path / "junk-13783878"
    make_run(root, copies=201, k=50, flacs=False)               # 299 passing non-copies
    doc = C.evaluate(G.load_run(root))
    assert doc["validity"]["valid"] and doc["bar"]["pass_not_copy"]["count"] == 299
    assert doc["bar"]["pass_not_copy"]["met"] is False and (doc["verdict"], doc["exit"]) == ("valid, under the bar", 3)
    root = tmp_path / "junk-13783879"
    make_run(root, copies=200, k=50, flacs=False)               # exactly 300
    doc = C.evaluate(G.load_run(root))
    assert doc["bar"]["pass_not_copy"]["count"] == 300 and doc["exit"] == 0


def test_invalid_and_under_the_bar_says_both(tmp_path):
    root = tmp_path / "junk-1"
    make_run(root, copies=250, k=50, flacs=False)
    edit_json(root / "summary" / "junk.json", lambda d: set_path(d, "partial", True))
    doc = C.evaluate(G.load_run(root))
    assert (doc["verdict"], doc["exit"]) == ("invalid, and under the bar", 3)


# --------------------------------------------------------------------------------------------- the two readouts
def test_the_readouts_count_a_verbatim_a_or_b_line(run_dir):
    run = G.load_run(run_dir)
    r = C.readouts(run)
    pool = list(range(54, 500))
    ab = [i for i in pool if verbatim_of(i)["A"] or verbatim_of(i)["B"]]
    both = [i for i in pool if verbatim_of(i)["A"] and verbatim_of(i)["B"]]
    assert r["passing_non_copies"] == {"k": len(ab), "n": 446, "share": round(len(ab) / 446, 4), "both": len(both)}
    sample = run.sample
    assert len(sample) == 50 and r["clips"]["n"] == r["clips"]["rendered"] == 50
    assert r["clips"]["k"] == sum(1 for i in sample if verbatim_of(i)["A"] or verbatim_of(i)["B"])
    assert r["clips"]["both"] == sum(1 for i in sample if verbatim_of(i)["A"] and verbatim_of(i)["B"])
    # the vamp and keys lines alone are not a verbatim line (D-021: "a verbatim A or B line")
    assert G.verbatim_line({"verbatim": {"vamp": True, "keys": True, "A": False, "B": False}}) is False
    assert G.verbatim_line({"verbatim": {"vamp": False, "keys": False, "A": False, "B": True}}) is True


def test_the_clips_readout_covers_the_clips_that_rendered_with_the_sample_beside_it(run_dir):
    def two_fail(d):
        d["clips"][3] = {"i": d["clips"][3]["i"], "ok": False, "gate": "G7", "detail": "true peak -0.80 dBTP"}
        d["clips"][7] = {"i": d["clips"][7]["i"], "ok": False, "gate": None, "skipped": "render deadline"}
    edit_json(run_dir / "summary" / "clips.json", two_fail)
    run = G.load_run(run_dir)
    r = C.readouts(run)["clips"]
    assert (r["n"], r["rendered"], r["recorded"], r["sampled"], r["sample"]["n"]) == (48, 48, 50, 50, 50)
    rendered = [x["i"] for x in run.rendered]
    assert r["k"] == sum(1 for i in rendered if verbatim_of(i)["A"] or verbatim_of(i)["B"])
    assert "of the whole sample of 50, 50 recorded" in C.format_text(C.evaluate(run))


def test_without_clips_json_the_check_still_reports_validity(run_dir, capsys):
    (run_dir / "summary" / "clips.json").unlink()
    assert cli.main(["check", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "readout: the clips' share is not there: summary/clips.json is absent" in out
    assert out.rstrip().endswith("GATE1_CHECK valid, first half met")
    with pytest.raises(G.Gate1Error, match="clips.json is missing"):
        G.load_run(run_dir)                                     # the batch needs it (need_clips defaults to True)


# --------------------------------------------------------------------------------------------- bad input
def _drop_row(rows):
    rows.pop(100)


BAD = [
    ("no junk.json", lambda r: (r / "summary" / "junk.json").unlink(), "junk.json is missing"),
    ("junk schema", lambda r: edit_json(r / "summary" / "junk.json", lambda d: d.update(schema="lofi-rl/junk/0")),
     "schema 'lofi-rl/junk/0'"),
    ("junk not json", lambda r: (r / "summary" / "junk.json").write_text("{"), "not readable JSON"),
    ("no scored", lambda r: (r / "summary" / "scored.jsonl").unlink(), "scored.jsonl is missing"),
    ("no completions", lambda r: (r / "summary" / "completions.jsonl").unlink(), "completions.jsonl is missing"),
    ("scored short", lambda r: edit_rows(r / "summary" / "scored.jsonl", _drop_row), "not one score"),
    ("scored cut line", lambda r: (r / "summary" / "scored.jsonl").open("a").write('{"i": 5'), "unreadable line"),
    ("scored repeat", lambda r: edit_rows(r / "summary" / "scored.jsonl", lambda rows: rows.__setitem__(1, rows[0])),
     "repeats sample index"),
    ("pass count", lambda r: edit_json(r / "summary" / "junk.json", lambda d: d["pass"].update(count=499)),
     "`pass.count` is 499"),
    ("pnc count", lambda r: edit_json(r / "summary" / "junk.json", lambda d: d["pass_not_copy"].update(count=447)),
     "`pass_not_copy.count` is 447"),
    ("pnc not whole", lambda r: edit_json(r / "summary" / "junk.json", lambda d: d["pass_not_copy"].update(count=446.0)),
     "no whole-number `pass_not_copy.count`"),
    ("no verbatim", lambda r: edit_rows(r / "summary" / "scored.jsonl", lambda rows: rows[300].update(verbatim=None)),
     "have no `verbatim` readout"),
    ("unscored row", lambda r: (r / "summary" / "completions.jsonl").open("a").write(json.dumps(
        {"i": 500, "brief": {}, "text": "x"}) + "\n"), "are not in scored.jsonl"),
    ("unwritten row", lambda r: edit_rows(r / "summary" / "completions.jsonl", _drop_row), "have no completions.jsonl"),
    ("clips order", lambda r: edit_json(r / "summary" / "clips.json", lambda d: d["clips"].reverse()),
     "are not junk's sample of 50"),
    ("clips copy", lambda r: edit_json(r / "summary" / "clips.json", lambda d: d["clips"][0].update(i=3)),
     "are not junk's sample of 50"),
    ("clips seed", lambda r: edit_json(r / "summary" / "clips.json", lambda d: d.update(seed=2027)),
     "are not junk's sample of 50 (seed 2027)"),
    ("clips palette", lambda r: edit_json(r / "summary" / "clips.json", lambda d: d["palette"].update(sha256=DEV_GU_SHA)),
     "rendered with palette b2e4a340…"),
    ("clips passing", lambda r: edit_json(r / "summary" / "clips.json", lambda d: d.update(passing=445)),
     "clips.json counts 445 passing non-copies"),
    ("clips schema", lambda r: edit_json(r / "summary" / "clips.json", lambda d: d.update(schema="x")), "schema 'x'"),
    ("clips k", lambda r: edit_json(r / "summary" / "clips.json", lambda d: d.update(k="50")), "must be whole numbers"),
]


@pytest.mark.parametrize("what, change, message", BAD, ids=[b[0] for b in BAD])
def test_files_that_are_not_one_runs_own_are_bad_input(run_dir, capsys, what, change, message):
    change(run_dir)
    with pytest.raises(G.Gate1Error) as err:
        G.load_run(run_dir)
    assert any(message in p for p in err.value.problems), err.value.problems
    if what not in ("clips order", "clips copy", "clips seed", "clips palette", "clips passing", "clips schema",
                    "clips k"):
        assert cli.main(["check", str(run_dir)]) == 2
        assert "error: bad input, nothing decided" in capsys.readouterr().err


def test_every_problem_is_listed_at_once(run_dir):
    edit_json(run_dir / "summary" / "junk.json", lambda d: (d["pass"].update(count=1), d["pass_not_copy"].update(count=2)))
    with pytest.raises(G.Gate1Error) as err:
        G.load_run(run_dir)
    assert len(err.value.problems) == 2


def test_a_folder_without_summary_is_bad_input(tmp_path, capsys):
    assert cli.main(["check", str(tmp_path)]) == 2
    assert "is not a folder: a run folder holds summary/" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------- the command line
def test_the_text_names_every_condition_and_the_json_is_the_report(run_dir, capsys):
    assert cli.main(["check", str(run_dir)]) == 0
    out = capsys.readouterr().out
    for c in C.conditions(json.loads((run_dir / "summary" / "junk.json").read_text())):
        assert f"ok    {c.name}" in out
    assert "validity: VALID, 27 of 27 conditions hold" in out
    assert "bar, first half: pass_not_copy.count 446 of 500, needs at least 300: met (rate 0.892, ci95 [0.8617, " \
           "0.9163], wilson, as junk.json gives it)" in out
    assert out.rstrip().endswith("GATE1_CHECK valid, first half met")
    assert cli.main(["check", str(run_dir), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == C.SCHEMA and doc == json.loads(json.dumps(C.evaluate(G.load_run(run_dir))))


def test_python_dash_m_runs_the_check(run_dir):
    import subprocess
    import sys
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    p = subprocess.run([sys.executable, "-m", "lofi_rl.gate1", "check", str(run_dir)], capture_output=True, text=True,
                       env=env)
    assert p.returncode == 0 and "GATE1_CHECK valid, first half met" in p.stdout, p.stderr


# --------------------------------------------------------------------------------------------- arm B
def test_arm_b_on_dev_gu_fails_exactly_the_swapped_conditions(capsys):
    """D-020's arm B (v2 with the grammar on dev_gu) met every condition of D-020's own list; under the gate's list it
    fails D-022's swapped values and nothing else, and its first half is met (446)."""
    if not (ARM_B / "summary" / "junk.json").is_file():
        pytest.skip(f"{ARM_B.relative_to(ROOT)}/summary is not on this machine (make pull brings it to the laptop)")
    doc = C.evaluate(G.load_run(ARM_B))
    v = doc["validity"]
    assert v["failed"] == SWAPPED and (v["held"], v["of"]) == (24, 27)
    shows = {c["name"]: c["shows"] for c in v["conditions"]}
    assert shows["palette.sha256"] == DEV_GU_SHA
    assert shows["constrained.regex_sha256"] == shows["constrained.in_language.regex_sha256"] == DEV_GU_REGEX
    assert doc["bar"]["pass_not_copy"]["count"] == 446 and doc["bar"]["pass_not_copy"]["met"] is True
    assert (doc["verdict"], doc["exit"]) == ("invalid", 3)
    r = doc["readouts"]
    assert (r["passing_non_copies"]["k"], r["passing_non_copies"]["n"], r["passing_non_copies"]["both"]) == (367, 446, 339)
    assert (r["clips"]["k"], r["clips"]["n"], r["clips"]["both"]) == (38, 50, 35)
    assert cli.main(["check", str(ARM_B)]) == 3
    assert capsys.readouterr().out.rstrip().endswith("GATE1_CHECK invalid")
