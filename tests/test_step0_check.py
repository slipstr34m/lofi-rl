"""The step-0 check (lofi_rl.step0.check; docs/design/step0_samples.md §4.1-4.2, §6.1): the validity list condition by
condition on a valid synthetic run, every tamper §6.1 lists caught by the condition that reads it, a retried render
error that left no brief undetermined, a redo pair on a temporary git repo that passes only when its changed files are
the ones its amendment names, the refusals of files that are not one run's own, and the shown-never-judged readouts
(item 8's on hand-built tuples, --rescore).

Synthetic runs come from tests/helpers_step0.py, stamped with the commits of a temporary git repo by
tests/helpers_step0_laptop.py; nothing here reads a pulled run, renders or listens.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import helpers_step0 as H
import helpers_step0_laptop as L
import lofi_rl.step0 as S
from lofi_rl import prompt_v2
from lofi_rl.brief import Brief
from lofi_rl.rl import junk as J
from lofi_rl.step0 import Step0Error, cli
from lofi_rl.step0 import check as C
from lofi_rl.step0 import requests as R

pytestmark = pytest.mark.laptop   # git reads the commits: never on a compute node (CLAUDE.md §1)

DEV_GU_SHA = "b2e4a340b9387e49f896e235c0f1f7f3c9339308ddc307b343c2d96e9eb8fab9"
DEV_GU_REGEX = "1412f11f9ebdc520b50c263eda14be12829951357c17f43496ab2ddf5235e64e"
RUN = "step0-13800001"
Q6B_MARK = R.mark_line((("rm_heldout", "all"), ("train", "pool")))


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """(the repo, its code commit, a valid stamped run): built once, copied by the tests that change them."""
    root = tmp_path_factory.mktemp("step0-check")
    repo, c1 = L.make_repo(root / "repo")
    L.stamped_run(root / RUN, repo, c1)
    return repo, c1, root / RUN


@pytest.fixture
def run_dir(base, tmp_path) -> Path:
    return L.copy_run(base[2], tmp_path / RUN)


@pytest.fixture
def repo_copy(base, tmp_path) -> Path:
    """A copy of the repo, history included, for a test that commits."""
    return L.copy_run(base[0], tmp_path / "repo")


def report(run_dir: Path, repo: Path, redo: Path | None = None) -> dict:
    return C.evaluate(C.load_run(run_dir, redo), repo=repo)


def failed(run_dir: Path, repo: Path, redo: Path | None = None) -> list[str]:
    return report(run_dir, repo, redo)["validity"]["failed"]


def cond(doc: dict, name: str) -> dict:
    (c,) = [c for c in doc["validity"]["conditions"] if c["name"] == name]
    return c


def set_path(doc: dict, path: str, value) -> None:
    *head, last = path.split(".")
    for part in head:
        doc = doc[part]
    doc[last] = value


# --------------------------------------------------------------------------------------------- a valid run
def test_a_valid_synthetic_run_passes_condition_by_condition(base, capsys, monkeypatch):
    repo, c1, run_dir = base
    doc = report(run_dir, repo)
    v = doc["validity"]
    assert v["failed"] == [] and v["valid"] and v["held"] == v["of"] == 74
    assert (doc["verdict"], doc["exit"]) == ("valid", 0)
    groups = [c["group"] for c in v["conditions"]]
    assert groups == sorted(groups, key=C.GROUPS.index), "every group in the list's order"
    assert [groups.count(g) for g in C.GROUPS] == [9, 3, 22, 3, 12, 10, 3, 4, 5, 3]
    names = {c["name"] for c in v["conditions"]}
    for want in ("generate.stopped", "generate.partial", "n", "requested", "unreadable_lines", "requests.missing",
                 "requests.duplicate", "requests.foreign", "requests.seed_mismatch", "generate.probe.passed",
                 "requests.prompt_tokens_mismatch", "generate.model.revision", "generate.streams_file.sha256",
                 "generate.streams_file.fresh", "generate.streams", "requests.brief_mismatch",
                 "by_stream.rm_heldout.prompt.rows_matching", "by_stream.train.prompt.rows_matching",
                 "generate.palette.sha256", "by_stream.train.palette.matches_generate", "clips.palette.sha256",
                 "generate.sampling", "generate.registration.found", "generate.registration.mark",
                 "by_stream.rm_heldout.constrained.in_language.stop.out_of_language",
                 "by_stream.train.constrained.in_language.other.n", "by_stream.train.failures.G5",
                 "by_stream.rm_heldout.failures.crash", C.STAMP_NAME, "generate.path_file.blobs_ok",
                 "the path at the generate commit", "render.errors_final", "render.skipped", "render.other_gates",
                 "held-out briefs undetermined", "sets.heldout streams", "sets.refb streams", "sets.pool streams",
                 "sets.pool taken", "sets.pool held-out", "sets.json = step0.select", "clip ids",
                 "render_code.source_sha8", "docs/DECISIONS.md at the generate commit"):
        assert want in names, want
    assert cond(doc, "n")["shows"] == "3520"
    assert cond(doc, "by_stream.rm_heldout.prompt.rows_matching")["needs"] == "320"
    assert cond(doc, "clip ids")["shows"].startswith("160 clip ids equal the formula (lofi_rain_v1@09f80a64")
    assert cond(doc, "sets.pool taken")["shows"] == "660 picks of 220 briefs"
    assert f"at {c1[:12]}: holds MARK" in cond(doc, "docs/DECISIONS.md at the generate commit")["shows"]
    assert cond(doc, "render_code.source_sha8")["shows"] == f"at {c1[:12]}: {L.source_sha8(repo)}; " \
                                                            f"render_code.source_sha8 {L.source_sha8(repo)}"
    assert doc["requests"]["rows"] == doc["requests"]["requested"] == 3520
    text = C.format_text(doc)
    assert text.rstrip().endswith("STEP0_CHECK valid") and "validity: VALID, 74 of 74 conditions hold" in text
    assert "rescore: not run (--rescore)" in text
    monkeypatch.setattr(C, "ROOT", repo)                       # the command line reads the laptop's own repo
    assert cli.main(["check", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert out.rstrip().splitlines()[-1] == "STEP0_CHECK valid"
    assert cli.main(["check", str(run_dir), "--json"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["schema"] == C.SCHEMA and got["verdict"] == "valid" and got["validity"]["of"] == 74


def test_the_shown_readouts_are_the_runs_and_never_judged(base):
    repo, _, run_dir = base
    sh = report(run_dir, repo)["shown"]
    assert sh["backend"]["backend"] == "xgrammar" and sh["probe"]["passed"] is True
    assert sh["replay"]["same"] == sh["replay_file"]["same"] == sh["replay_file"]["rows"] == 320
    assert sh["render"]["errors_retried"] == 0 and sh["host"]["gpu"] == H.GPU
    assert sh["sets"]["pool"]["taken"] == 660 and set(sh["timing"]) == set(C.TIMINGS)
    i8 = sh["item8"]
    assert (i8["tuples"], i8["pairs"]) == (40, 240) and i8["all_carry"] == {"A": 40, "B": 40, "AB": 40}
    assert i8["differ"] == 0, "every synthetic loop is its cell's example with another first kick grid"


# --------------------------------------------------------------------------------------------- the tampers (§6.1)
def _drop_row(i: int):
    def fn(run_dir: Path) -> None:
        L.edit_rows(run_dir / "summary" / "completions.jsonl", lambda rows: rows.remove(next(r for r in rows
                                                                                              if r["i"] == i)))
        L.edit_rows(run_dir / "summary" / "scored.jsonl", lambda rows: rows.remove(next(r for r in rows
                                                                                         if r["i"] == i)))
        L.edit_json(run_dir / "summary" / "step0.json", lambda d: d.update(n=3519))
    return fn


def _comp(i: int, **fields):
    def fn(run_dir: Path) -> None:
        L.edit_rows(run_dir / "summary" / "completions.jsonl",
                    lambda rows: next(r for r in rows if r["i"] == i).update(fields))
    return fn


def _doc(name: str, path: str, value):
    return lambda run_dir: L.edit_json(run_dir / "summary" / name, lambda d: set_path(d, path, value))


def _repeat(i: int):
    def fn(run_dir: Path) -> None:
        f = run_dir / "summary" / "completions.jsonl"
        line = next(x for x in f.read_text().splitlines() if json.loads(x)["i"] == i)
        f.write_text(f.read_text() + line + "\n")
    return fn


def _wrong_clip_id(run_dir: Path) -> None:
    """clips.json's first record and sets.json's tuple entry agree on a clip id the formula does not give."""
    L.edit_json(run_dir / "summary" / "clips.json", lambda d: d["clips"][0].update(clip_id="c0123456789abcdef"))
    L.edit_json(run_dir / "summary" / "sets.json", lambda d: d["heldout"][0]["taken"][0].update(
        clip_id="c0123456789abcdef"))


def _pool_row_from_rm_heldout(run_dir: Path) -> None:
    L.edit_json(run_dir / "summary" / "sets.json", lambda d: d["pool"][0]["taken"][0].update(i=5))


def _edited_sets(run_dir: Path) -> None:
    L.edit_json(run_dir / "summary" / "sets.json", lambda d: d["refb"][3].update(spares=[]))


def _dirty(run_dir: Path) -> None:
    """generate and score ran from a tree with uncommitted changes (stamp.sh's dirty=1); render's stamp is clean."""
    st = json.loads((run_dir / "summary" / "generate.json").read_text())["host"]["stamp"].replace("dirty=0", "dirty=1")
    _doc("generate.json", "host.stamp", st)(run_dir)
    _doc("step0.json", "score_stamp", st)(run_dir)


TAMPERS = [
    ("a seed (draw_seed)", _comp(5, draw_seed=12345), ["requests.seed_mismatch"]),
    ("a seed (sp_seed)", _comp(400, sp_seed=1), ["requests.seed_mismatch"]),
    ("a repeated request", _repeat(17), ["requests.duplicate"]),
    ("a foreign row", _comp(9, k=2), ["requests.missing", "requests.foreign"]),        # row 9 is 5101:1's k1
    ("a misaligned prompt_tokens", _comp(400, prompt_tokens=S.PROMPT_TOKENS["low/hopeful"] + 1),
     ["requests.prompt_tokens_mismatch"]),
    ("another brief", _comp(3, brief=J.brief_dict(Brief("low", "hopeful", 82, 0))), ["requests.brief_mismatch"]),
    ("an out-of-language stop completion",
     _doc("step0.json", "by_stream.rm_heldout.constrained.in_language.stop.out_of_language", 1),
     ["by_stream.rm_heldout.constrained.in_language.stop.out_of_language"]),
    ("another palette sha", _doc("generate.json", "palette.sha256", DEV_GU_SHA), ["generate.palette.sha256"]),
    ("another palette at score", _doc("step0.json", "by_stream.train.palette.sha256", DEV_GU_SHA),
     ["by_stream.train.palette.sha256"]),
    ("another regex sha", _doc("step0.json", "by_stream.train.constrained.regex_sha256", DEV_GU_REGEX),
     ["by_stream.train.constrained.regex_sha256"]),
    ("two stamps", _doc("step0.json", "score_stamp", "commit=0123456789ab dirty=0 diff=- at=2026-11-07T19:00:00Z"),
     [C.STAMP_NAME]),
    ("a dirty stamp", _dirty, [C.STAMP_NAME]),
    ("no D-023 mark", _doc("generate.json", "registration.found", False), ["generate.registration.found"]),
    ("streams that differ from the mark (its mark)", _doc("generate.json", "registration.mark", Q6B_MARK),
     ["generate.registration.mark"]),
    ("streams that differ from the mark (drawn)", _doc("generate.json", "streams.train.briefs", 220),
     ["generate.streams"]),
    ("an edited sets.json", _edited_sets, ["sets.json = step0.select"]),
    ("a wrong clip id", _wrong_clip_id, ["clip ids"]),
    ("a pool row from rm_heldout", _pool_row_from_rm_heldout, ["sets.pool streams", "sets.json = step0.select"]),
    ("a stopped generate", _doc("generate.json", "stopped", "deadline"), ["generate.stopped"]),
    ("the probe missed", _doc("generate.json", "probe.passed", False), ["generate.probe.passed"]),
    ("a cut line", _doc("step0.json", "unreadable_lines", 1), ["unreadable_lines"]),
    ("a G3 under the grammar", _doc("step0.json", "by_stream.train.failures.G3", 1), ["by_stream.train.failures.G3"]),
    ("no render block", _doc("step0.json", "render", None), ["render.errors_final", "render.skipped",
                                                              "render.other_gates"]),
    ("another model", _doc("generate.json", "model.revision", "0" * 40), ["generate.model.revision"]),
    ("a stale streams file", _doc("generate.json", "streams_file.fresh", False), ["generate.streams_file.fresh"]),
    ("other sampling", _doc("generate.json", "sampling.top_p", 0.95), ["generate.sampling"]),
    ("a path-file blob that differs", _doc("generate.json", "path_file.blobs_ok", False),
     ["generate.path_file.blobs_ok"]),
]


@pytest.mark.parametrize("why, tamper, names", TAMPERS, ids=[t[0] for t in TAMPERS])
def test_each_tamper_is_caught_by_the_condition_that_reads_it(base, run_dir, why, tamper, names):
    repo = base[0]
    tamper(run_dir)
    doc = report(run_dir, repo)
    assert doc["validity"]["failed"] == names and (doc["verdict"], doc["exit"]) == ("invalid", 3)
    for name in names:
        c = cond(doc, name)
        assert c["ok"] is False and c["shows"] != c["needs"]
    assert C.format_text(doc).rstrip().endswith("STEP0_CHECK invalid")


def test_a_missing_request_is_named_with_the_rest_it_breaks(base, run_dir):
    _drop_row(7)(run_dir)
    doc = report(run_dir, base[0])
    assert doc["validity"]["failed"] == ["n", "requests.missing", "sets.json = step0.select"]
    assert cond(doc, "requests.missing")["shows"] == "1 (first i 7)"
    shows = cond(doc, "sets.json = step0.select")["shows"]
    assert "select refuses the files: 5101:0 holds draws [0, 1, 2, 3, 4, 5, 6]" in shows


@pytest.mark.parametrize("path, value, name", [
    ("partial", 0, "generate.partial"),                         # 0 is not false
    ("requested", 3520.0, "generate.requested"),                # the count is right, the JSON type is not
    ("probe.passed", 1, "generate.probe.passed"),
    ("registration.found", "true", "generate.registration.found"),
])
def test_a_value_needs_the_registered_json_type(base, run_dir, path, value, name):
    L.edit_json(run_dir / "summary" / "generate.json", lambda d: set_path(d, path, value))
    assert failed(run_dir, base[0]) == [name]


def test_n_needs_the_registered_json_type_too(base, run_dir):
    L.edit_json(run_dir / "summary" / "step0.json", lambda d: d.update(n=3520.0))
    assert failed(run_dir, base[0]) == ["n"]


def test_a_commit_whose_files_differ_from_the_path_file_is_caught(base, run_dir, repo_copy):
    c3 = L.commit(repo_copy, "kit.env drifts from the path file", {"kit.env": "# not the listed kit.env\n"})
    L.restamp(run_dir, repo_copy, c3)
    doc = report(run_dir, repo_copy)
    assert doc["validity"]["failed"] == ["the path at the generate commit"]
    assert "1 differ (kit.env)" in cond(doc, "the path at the generate commit")["shows"]


def test_a_path_file_other_than_the_one_generate_checked_is_caught(base, run_dir):
    L.edit_json(run_dir / "summary" / "generate.json", lambda d: set_path(d, "path_file.sha256", "0" * 64))
    doc = report(run_dir, base[0])
    assert doc["validity"]["failed"] == ["the path at the generate commit"]
    assert "is not generate's" in cond(doc, "the path at the generate commit")["shows"]


def test_a_commit_without_d023s_mark_is_caught(base, run_dir, repo_copy):
    c4 = L.commit(repo_copy, "D-023 without its mark", {S.DECISIONS: L.decisions_text(mark=False)})
    L.restamp(run_dir, repo_copy, c4)
    doc = report(run_dir, repo_copy)
    assert doc["validity"]["failed"] == ["docs/DECISIONS.md at the generate commit"]
    assert "does not hold MARK" in cond(doc, "docs/DECISIONS.md at the generate commit")["shows"]


def test_a_commit_whose_mark_generate_would_not_read_is_caught(base, run_dir, repo_copy):
    """The git condition reads the mark as generate's pre-check does (requests.find_marks, finding R2-2): MARK with
    text after it on its line is no mark, though the text holds MARK as a substring."""
    text = L.decisions_text(mark=False) + f"- **The mark:** {S.MARK} (checked by generate)\n"
    c5 = L.commit(repo_copy, "D-023's mark with text after it", {S.DECISIONS: text})
    L.restamp(run_dir, repo_copy, c5)
    doc = report(run_dir, repo_copy)
    assert S.MARK in text and doc["validity"]["failed"] == ["docs/DECISIONS.md at the generate commit"]
    assert "does not hold MARK" in cond(doc, "docs/DECISIONS.md at the generate commit")["shows"]


def test_a_stamp_the_repo_does_not_hold_fails_every_git_condition(base, run_dir):
    L.restamp(run_dir, base[0], "0123456789ab" + "0" * 28, steps=("score", "render"))
    for name in ("score_stamp", "render_stamp"):
        assert json.loads((run_dir / "summary" / "step0.json").read_text())[name].startswith("commit=0123456789ab")
    doc = report(run_dir, base[0])
    assert doc["validity"]["failed"] == [C.STAMP_NAME, "render_code.source_sha8"]
    assert "commit 0123456789ab is not in" in cond(doc, "render_code.source_sha8")["shows"]


def test_render_code_other_than_the_commits_is_caught(base, run_dir):
    L.restamp(run_dir, base[0], base[1], steps=("render",), source="deadbeef")
    doc = report(run_dir, base[0])
    assert doc["validity"]["failed"] == ["render_code.source_sha8"]


def test_under_q5b_a_pool_pick_may_share_a_text_but_not_be_a_held_out_clip(base, run_dir, monkeypatch):
    monkeypatch.setattr(S, "POOL_EXCLUDES", "clip")
    doc = report(run_dir, base[0])
    assert "no pool pick is a held-out clip" in cond(doc, "sets.pool held-out")["needs"]
    assert doc["validity"]["failed"] == [], "the default run holds neither"


# --------------------------------------------------------------------------------------------- the render step
@pytest.fixture(scope="module")
def retried(base, tmp_path_factory):
    repo, c1, _ = base
    root = tmp_path_factory.mktemp("step0-check-retried") / RUN
    L.stamped_run(root, repo, c1, {"5101:0": "h0 h1/E h2 h3 h4 h5 h6 h7", "5101:1": "h0 h1 h2/G7 h3 h4 h5 h6 h7"})
    return root


def test_a_retried_error_that_left_no_brief_undetermined_still_passes(base, retried):
    doc = report(retried, base[0])
    assert doc["validity"]["failed"] == [] and doc["exit"] == 0
    r = doc["shown"]["render"]
    assert (r["errors_retried"], r["errors_final"], r["G7"], r["rounds"]) == (1, 0, 1, 2)
    assert "render: attempted 162, rendered 160, G7 1, errors_retried 1, rounds 2" in C.format_text(doc)


def test_an_error_twice_leaves_the_brief_undetermined_and_the_run_invalid(base, tmp_path):
    repo, c1, _ = base
    run_dir = tmp_path / RUN
    L.stamped_run(run_dir, repo, c1, {"5101:2": "h0 h1 h2 h3/EE h4 h5 h6 h7"})
    doc = report(run_dir, repo)
    assert doc["validity"]["failed"] == ["render.errors_final", "held-out briefs undetermined"]
    assert cond(doc, "held-out briefs undetermined")["shows"] == "1 of 40 (5101:2)"


# --------------------------------------------------------------------------------------------- a redo pair (§4.2)
def _redo(out: Path, dest: Path, repo: Path, commit: str, *, files=("scored.jsonl", "step0.json", "sets.json",
                                                                     "clips.json")) -> Path:
    """A recovery's folder: OUT's score and render records, written again at `commit` (its stamps and key)."""
    (dest / "summary").mkdir(parents=True)
    for name in files:
        (dest / "summary" / name).write_bytes((out / "summary" / name).read_bytes())
    L.restamp(dest, repo, commit, steps=("score", "render"))
    return dest


AMENDED = "src/lofi_rl/step0/select.py"


def _fix(repo: Path, amendment: str | None) -> str:
    text = (repo / AMENDED).read_text() + "# the redo's fix\n"
    extra = "" if amendment is None else f"\n- {amendment}\n"
    return L.commit(repo, "the redo's fix", {AMENDED: text, S.DECISIONS: L.decisions_text(extra=extra)})


@pytest.mark.parametrize("amendment", [f"D-023 redo amendment: run {RUN}; changed {AMENDED}",
                                       f"D-023 redo amendment: run {RUN}; changed `{AMENDED}`.",
                                       f"**D-023 redo amendment: run {RUN}; changed {AMENDED} (the fix to select).**"],
                         ids=["the bare form", "a backticked path and a closing period", "bold, with a note"])
def test_a_redo_pair_passes_when_its_changed_files_are_the_amendments(base, run_dir, repo_copy, tmp_path, amendment):
    c2 = _fix(repo_copy, amendment)
    redo = _redo(run_dir, tmp_path / "step0-redo-13800002", repo_copy, c2)
    doc = report(run_dir, repo_copy, redo)
    assert doc["validity"]["failed"] == [] and doc["exit"] == 0
    assert doc["validity"]["held"] == doc["validity"]["of"] == 75, "a redo pair: one condition more than a run"
    names = [c["name"] for c in doc["validity"]["conditions"]]
    assert C.REDO_STAMP_NAME in names and C.STAMP_NAME not in names and "the path at the redo commit" in names
    shows = cond(doc, "the path at the redo commit")["shows"]
    assert f"at {c2[:12]}:" in shows and f"1 differ ({AMENDED})" in shows and f"names {AMENDED}" in shows
    assert doc["files"] == {"generate.json": "run", "completions.jsonl": "run", "replay.jsonl": "run",
                            "scored.jsonl": "redo", "step0.json": "redo", "sets.json": "redo", "clips.json": "redo"}
    assert doc["redo"] == "step0-redo-13800002" and f"redo step0-redo-13800002" in C.format_text(doc)
    rec = doc["git"]["redo_commits"][c2[:12]]
    assert rec["amendment"] == [AMENDED] and rec["path"]["excepted"] == [AMENDED]


@pytest.mark.parametrize("amendment", [None, f"D-023 redo amendment: run {RUN}; changed src/lofi_rl/step0/score.py",
                                       f"D-023 redo amendment: run step0-13800009; changed {AMENDED}"],
                         ids=["no amendment", "another file", "another run"])
def test_a_redo_pair_fails_when_a_changed_file_is_not_the_amendments(base, run_dir, repo_copy, tmp_path, amendment):
    c2 = _fix(repo_copy, amendment)
    redo = _redo(run_dir, tmp_path / "step0-redo-13800002", repo_copy, c2)
    doc = report(run_dir, repo_copy, redo)
    assert doc["validity"]["failed"] == ["the path at the redo commit"] and doc["exit"] == 3
    assert f"not named: {AMENDED}" in cond(doc, "the path at the redo commit")["shows"]


def test_a_render_only_redo_reads_scores_from_the_run(base, run_dir, repo_copy, tmp_path):
    c2 = _fix(repo_copy, f"D-023 redo amendment: run {RUN}; changed {AMENDED}, src/lofi_rl/step0/render.py")
    redo = _redo(run_dir, tmp_path / "step0-redo-13800003", repo_copy, c2, files=("step0.json", "sets.json",
                                                                                   "clips.json"))
    L.edit_json(redo / "summary" / "step0.json", lambda d: d.update(score_stamp=L.stamp(base[1])))
    doc = report(run_dir, repo_copy, redo)
    assert doc["validity"]["failed"] == [] and doc["files"]["scored.jsonl"] == "run"
    assert doc["stamp"]["score"] == L.stamp(base[1]) and doc["stamp"]["render"] == L.stamp(c2)


def test_a_score_redo_without_its_render_redo_reads_invalid_and_says_so(base, run_dir, repo_copy, tmp_path):
    """Finding E3: score alone writes step0.json with no render block and no sets.json or clips.json, so the pair
    reads REDO's scores beside OUT's sets: invalid, with the render redo named as what is missing."""
    c2 = _fix(repo_copy, f"D-023 redo amendment: run {RUN}; changed {AMENDED}")
    redo = tmp_path / "step0-redo-13800005"
    (redo / "summary").mkdir(parents=True)
    for name in ("scored.jsonl", "step0.json"):
        (redo / "summary" / name).write_bytes((run_dir / "summary" / name).read_bytes())
    L.restamp(redo, repo_copy, c2, steps=("score",))
    L.edit_json(redo / "summary" / "step0.json", lambda d: d.update(render=None, render_code=None, sets=None,
                                                                     render_stamp=None))
    doc = report(run_dir, repo_copy, redo)
    assert doc["exit"] == 3 and doc["validity"]["failed"] == [C.REDO_STAMP_NAME, "render.errors_final",
                                                              "render.skipped", "render.other_gates"]
    assert "a score redo without its render redo" in cond(doc, C.REDO_STAMP_NAME)["shows"]
    assert "without its render redo" not in cond(report(run_dir, repo_copy, _redo(
        run_dir, tmp_path / "step0-redo-13800006", repo_copy, c2)), C.REDO_STAMP_NAME)["shows"]


def test_a_redo_at_the_runs_own_commit_and_a_dirty_redo(base, run_dir, tmp_path):
    repo, c1, _ = base
    redo = _redo(run_dir, tmp_path / "step0-redo-13800004", repo, c1)
    doc = report(run_dir, repo, redo)
    assert doc["validity"]["failed"] == []
    assert cond(doc, "the path at the redo commit")["shows"] == f"the redo ran at the generate commit {c1[:12]}"
    L.edit_json(redo / "summary" / "step0.json", lambda d: d.update(score_stamp=L.stamp(c1, dirty=1)))
    assert failed(run_dir, repo, redo) == [C.REDO_STAMP_NAME]


def test_a_redo_pair_reads_generates_files_from_the_run_and_the_rest_from_the_redo():
    assert set(C.GENERATE_FILES) | set(C.LATER_FILES) == set(S.SUMMARY_FILES)
    assert not set(C.GENERATE_FILES) & set(C.LATER_FILES)
    assert set(C.MAY_BE_ABSENT) < set(S.SUMMARY_FILES) and "scored.jsonl" not in C.MAY_BE_ABSENT


def test_amendment_lines_are_read_per_run():
    text = ("- D-023 redo amendment: run step0-1; changed a.py, b/c.py\n"
            "`D-023 redo amendment: run step0-1; changed d.py`\n"
            "D-023 redo amendment: run step0-2; changed e.py\n"
            "D-023 redo amendment: run step0-1; changed nothing\n")
    assert C.amendment_files(text, "step0-1") == ["a.py", "b/c.py", "d.py"]
    assert C.amendment_files(text, "step0-2") == ["e.py"] and C.amendment_files(text, "step0-3") == []
    assert C.amendment_files(text, None) == ["a.py", "b/c.py", "d.py", "e.py"], "every run's"


@pytest.mark.parametrize("line, files", [
    ("D-023 redo amendment: run step0-7; changed src/a.py", ["src/a.py"]),
    ("- D-023 redo amendment: run step0-7; changed src/a.py.", ["src/a.py"]),
    ("D-023 redo amendment: run step0-7; changed `src/a.py`.", ["src/a.py"]),
    ("D-023 redo amendment: run step0-7; changed `src/a.py`, `src/b.py`.", ["src/a.py", "src/b.py"]),
    ("D-023 redo amendment: run step0-7; changed src/a.py and src/b.py", ["src/a.py", "src/b.py"]),
    ("**D-023 redo amendment: run step0-7; changed src/a.py**", ["src/a.py"]),
    ("D-023 redo amendment: run step0-7; changed src/a.py, src/b.py (the deadline fix).", ["src/a.py", "src/b.py"]),
    ("`D-023 redo amendment: run step0-7; changed src/a.py`", ["src/a.py"]),
    ("D-023 redo amendment: run step0-7; changed nothing.", []),
    ("D-023 redo amendment: run step0-7; changed `nothing`", []),
    ("- **D-023 redo amendment:** run step0-7; changed src/a.py", []),
], ids=["bare", "a closing period", "backticks", "two in backticks", "and", "bold", "a note", "inline code",
        "nothing.", "`nothing`", "a bold head is not the form"])
def test_an_amendment_line_is_read_as_house_style_writes_it(line, files):
    """Findings R3-3, E2 and R2-6: DECISIONS.md backticks its paths and ends its bullets with a period, so the line
    is read leniently; only a bold head, which breaks the line's form, is not read (D-023 gives the form verbatim)."""
    assert C.amendment_files(f"# D-023\n{line}\nmore text\n", "step0-7") == files
    assert C.amendment_files(f"# D-023\n{line}\n", "step0-8") == []


def test_an_amended_file_off_the_path_is_shown_and_excepts_nothing(base, run_dir, repo_copy, tmp_path):
    c2 = _fix(repo_copy, f"D-023 redo amendment: run {RUN}; changed {AMENDED}, CHANGELOG.md")
    redo = _redo(run_dir, tmp_path / "step0-redo-13800002", repo_copy, c2)
    doc = report(run_dir, repo_copy, redo)
    assert doc["validity"]["failed"] == []
    assert "named but not on the path: CHANGELOG.md" in cond(doc, "the path at the redo commit")["shows"]


# --------------------------------------------------------------------------------------------- bad input
def _replace_text(name: str, old: str, new: str):
    def fn(run_dir: Path) -> None:
        f = run_dir / "summary" / name
        f.write_text(f.read_text().replace(old, new, 1))
    return fn


BAD = [
    ("no summary", lambda r: shutil.rmtree(r / "summary"), "is not a folder"),
    ("no generate.json", lambda r: (r / "summary" / "generate.json").unlink(), "generate.json is missing"),
    ("no scored.jsonl", lambda r: (r / "summary" / "scored.jsonl").unlink(), "scored.jsonl is missing"),
    ("generate schema", _doc("generate.json", "schema", "lofi-rl/junk-generate/1"),
     "schema 'lofi-rl/junk-generate/1', not 'lofi-rl/step0-generate/1'"),
    ("sets schema", _doc("sets.json", "schema", "lofi-rl/step0-sets/0"), "not 'lofi-rl/step0-sets/1'"),
    ("not json", lambda r: (r / "summary" / "step0.json").write_text("{"), "not readable JSON"),
    ("scored cut line", lambda r: (r / "summary" / "scored.jsonl").open("a").write('{"i": 5'), "unreadable line"),
    ("scored text", lambda r: L.edit_rows(r / "summary" / "scored.jsonl", lambda rows: rows[4].update(
        text_sha256="0" * 64)), "not a score of these completions"),
    ("unscored row", lambda r: L.edit_rows(r / "summary" / "scored.jsonl", lambda rows: rows.pop(9)),
     "are not in scored.jsonl"),
    ("n", _doc("step0.json", "n", 3519), "step0.json `n` is 3519: not one score"),
    ("render block", _doc("step0.json", "render.rendered", 159), "is not the one clips.json gives"),
    ("render code", _doc("sets.json", "render_code.source_sha8", "deadbeef"), "the render-code key differs"),
    ("render stamp", _doc("step0.json", "render_stamp", "commit=0123456789ab dirty=0"), "is not render_code.stamp"),
    ("sets shape", _doc("sets.json", "pool", {"596:0": []}), "`pool` must be a list of entries"),
    ("clips shape", _doc("clips.json", "clips", None), "`clips` must be a list of records"),
]


@pytest.mark.parametrize("why, tamper, message", BAD, ids=[b[0] for b in BAD])
def test_files_that_are_not_one_runs_own_are_bad_input(run_dir, capsys, why, tamper, message):
    tamper(run_dir)
    with pytest.raises(Step0Error, match=re.escape(message)):
        C.load_run(run_dir)
    assert cli.main(["check", str(run_dir)]) == 2
    captured = capsys.readouterr()
    assert "error: bad input, nothing decided" in captured.err
    assert captured.out.rstrip().splitlines()[-1] == "STEP0_CHECK bad input"


def _stopped(run_dir: Path, why: str) -> None:
    """generate.json as generate leaves a run it stopped before its first draw, and nothing else in summary/."""
    summ = run_dir / "summary"
    gen = json.loads((summ / "generate.json").read_text())
    gen.update(stopped=why, generated=0, partial=False, timing={**gen["timing"], "chunks": []},
               replay={"n": 0, "same": 0, "differ": 0, "by_chunk": [], "stopped": "not run"})
    if why == "seed probe":
        gen["probe"]["passed"] = False
    else:
        gen.update(engine={}, probe={**gen["probe"], "texts_sha256": None, "distinct": None, "passed": None})
    for p in summ.iterdir():
        p.unlink()
    (summ / "generate.json").write_text(json.dumps(gen, indent=1))


@pytest.mark.parametrize("why", ["seed probe", "pre-check: docs/DECISIONS.md holds no 'D-023 mark:' line"])
def test_a_run_stopped_before_its_first_draw_reads_invalid_not_bad_input(base, run_dir, capsys, why):
    """Finding E1: generate.json alone, stopped with nothing drawn, reads invalid with the stop named; anything else
    beside it is not one run's files."""
    _stopped(run_dir, why)
    doc = report(run_dir, base[0])
    v = doc["validity"]
    assert doc["exit"] == 3 and v["of"] == 74 and {"generate.stopped", "generate.probe.passed", "n",
                                                   "requests.missing"} <= set(v["failed"])
    assert cond(doc, "generate.stopped")["shows"] == why and "generate.partial" not in v["failed"]
    assert C.format_text(doc).rstrip().endswith("STEP0_CHECK invalid")
    (run_dir / "summary" / "step0.json").write_text("{}")
    with pytest.raises(Step0Error, match="beside a generate.json that drew nothing"):
        C.load_run(run_dir)


def test_a_score_that_left_no_records_is_named_a_scoring_defect(run_dir):
    """Finding R2-7: generate's records complete, score killed or crashed before its end (no scored.jsonl, no
    step0.json): bad input, named as §4.2 routes it."""
    for name in ("scored.jsonl", "step0.json", "sets.json", "clips.json"):
        (run_dir / "summary" / name).unlink()
    with pytest.raises(Step0Error) as err:
        C.load_run(run_dir)
    assert "scored.jsonl is missing" in str(err.value) and "a scoring defect, D-020's 'a crash'" in str(err.value)


def test_a_redo_folder_must_be_another_folder(run_dir):
    with pytest.raises(Step0Error, match="is the run folder itself"):
        C.load_run(run_dir, run_dir)


def test_without_sets_and_clips_the_run_is_invalid_not_bad_input(base, run_dir):
    for name in ("sets.json", "clips.json"):
        (run_dir / "summary" / name).unlink()
    L.edit_json(run_dir / "summary" / "step0.json", lambda d: d.update(render=None, render_code=None,
                                                                        render_stamp=None))
    doc = report(run_dir, base[0])
    assert doc["exit"] == 3 and "sets.json = step0.select" in doc["validity"]["failed"]
    assert cond(doc, "clip ids")["shows"] == "clips.json is missing"
    assert cond(doc, "render.errors_final")["shows"] == "missing"
    assert cond(doc, C.STAMP_NAME)["ok"] is False, "no render stamp"


def test_check_imports_with_the_policy_envs_absent_modules_and_yaml_blocked():
    """The policy env's build imports every lofi_rl module outside its scorer-side list (jobs/env_policy.sbatch), so
    check imports git, the palette, junk and the features only inside the functions that use them."""
    blocked = ("yaml", "numpy", "mido", "soundfile", "scipy", "pedalboard", "torch", "trl", "vllm", "transformers")
    code = ("import sys\n" + "".join(f"sys.modules[{m!r}] = None\n" for m in blocked)
            + "import lofi_rl.step0.check\nprint('IMPORTS_OK')\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PYTHONPATH": str(L.ROOT / "src")}, cwd=L.ROOT)
    assert p.returncode == 0 and "IMPORTS_OK" in p.stdout, p.stderr[-2000:]


# --------------------------------------------------------------------------------------------- the formulas
def test_the_clip_id_formula_is_the_scorers(tmp_path):
    from lofi_rl.scoring import protocol
    from lofi_rl.scoring.render_pool import render_config
    pal = H.palette()
    cfg = render_config()
    for b in (Brief("low", "hopeful", 60, 0), Brief("high", "melancholy", 95, 11), Brief("mid", "wistful", 82, 6)):
        canon = H.sha(f"canon {b}")
        assert C.clip_id(canon, b, f"{pal.id}@{pal.sha256[:8]}", protocol.render_cfg_sha8(cfg)) == \
            protocol.clip_id(canon, b, pal.id, pal.sha256, cfg)


def test_the_copy_forms_are_junks_and_the_drafts_k_d_form():
    from lofi_rl.dsl import canon, parse
    from lofi_rl.drafts.check import copy_form
    pal = H.palette()
    for cell in ("low/melancholy", "mid/wistful", "high/hopeful"):
        b = Brief(*cell.split("/"), 82, 0)
        ((_, text),) = prompt_v2.shown_examples(b, pal)
        res = parse(text, vocab=pal.vocab(), mode=b.mode)
        junk_form, kd = C.copy_forms(text, b, pal)
        assert kd == copy_form(res.ast)
        row = J.score_row({"i": 0, "brief": J.brief_dict(b), "text": text, "finish_reason": "stop", "n_tokens": 1},
                          pal, J.CopyCheck("v2", pal))
        assert J.verbatim(junk_form, J.CopyCheck("v2", pal).shown(b)) == row["verbatim"], "junk's own copy form"
    with pytest.raises(Step0Error, match="fails G"):
        C.copy_forms("vamp: nonsense\n", Brief("low", "hopeful", 82, 0), pal)


# --------------------------------------------------------------------------------------------- item 8's readouts
def _with_line(text: str, head: str, line: str) -> str:
    return "\n".join(line if x.startswith(f"{head}: ") else x for x in text.splitlines()) + "\n"


def test_item8_readouts_on_hand_built_tuples():
    from lofi_rl.drafts.check import copy_form
    from lofi_rl.dsl import canon, parse
    pal = H.palette()
    b = Brief("mid", "wistful", 82, 0)
    cell = "mid/wistful"
    other = {h: next(x for x in H._example("high/hopeful").splitlines() if x.startswith(f"{h}: ")) for h in "AB"}
    texts = [H.variant_text(cell, "a"), H.variant_text(cell, "b"),
             _with_line(H.variant_text(cell, "c"), "A", other["A"]),
             _with_line(_with_line(H.variant_text(cell, "d"), "A", other["A"]), "B", other["B"])]
    check = J.CopyCheck("v2", pal)
    rows = [J.score_row({"i": k, "brief": J.brief_dict(b), "text": t, "finish_reason": "stop", "n_tokens": 1}, pal,
                        check) for k, t in enumerate(texts)]
    assert all(r["ok"] for r in rows)
    got = C.tuple_readout([(k, t, r) for k, (t, r) in enumerate(zip(texts, rows))], b, pal)
    assert [p["k"] for p in got["pairs"]] == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], "6 pairs, k order"
    assert [(p["A"], p["B"], p["differ"]) for p in got["pairs"]] == [
        (False, False, False), (True, False, True), (True, True, True), (True, False, True), (True, True, True),
        (False, True, True)]
    assert got["verbatim"] == {"A": sum(r["verbatim"]["A"] for r in rows),
                               "B": sum(r["verbatim"]["B"] for r in rows),
                               "AB": sum(r["verbatim"]["A"] and r["verbatim"]["B"] for r in rows)}
    assert got["verbatim"] == {"A": 2, "B": 3, "AB": 2}, "the example's A in clips 0 and 1, its B in 0-2"
    asts = [parse(t, vocab=pal.vocab(), mode=b.mode).ast for t in texts]
    for p in got["pairs"]:
        x, y = (asts[k] for k in p["k"])
        junk_x, junk_y = canon(replace(x, sax=None)), canon(replace(y, sax=None))
        assert p["j5"] == round(J.jaccard(J.grams(junk_x), J.grams(junk_y)), 4)
        assert p["j5_kd"] == round(J.jaccard(J.grams(copy_form(x)), J.grams(copy_form(y))), 4)
    # 0 and 1 differ in one kick cell whose 5-grams the text holds elsewhere, so their 5-gram sets are equal
    assert got["pairs"][0]["j5"] == 1.0 and got["pairs"][1]["j5"] < got["pairs"][0]["j5"]


# --------------------------------------------------------------------------------------------- --rescore
def test_rescore_lists_the_rows_that_score_differently(tmp_path):
    run = H.make_run(tmp_path / RUN, briefs=["5101:0", "596:0"])
    loaded = C.load_run(run.dir)
    rs = C.rescore(loaded)
    assert (rs["rows"], rs["differ"], rs["fields"]) == (16, 0, {}), "the synthetic rows are junk.score_row's"
    L.edit_rows(run.summary / "scored.jsonl", lambda rows: rows[3].update(p_fix=0.5, near_copy=False))
    rs = C.rescore(C.load_run(run.dir))
    assert rs["differ"] == 1 and rs["fields"] == {"near_copy": 1, "p_fix": 1} and rs["first"] == [
        {"i": 3, "fields": ["near_copy", "p_fix"]}]
    doc = C.evaluate(C.load_run(run.dir), rescore_rows=True)
    assert doc["rescore"]["differ"] == 1 and doc["exit"] == 3, "a run of 2 briefs is not complete"
    assert "rescore with this checkout: 1 of 16 rows differ from scored.jsonl" in C.format_text(doc)
