"""The step-0 prereg files (lofi_rl.step0.prereg; docs/design/step0_samples.md §4.4, §5, §6.1): docs/prereg/
heldout_v1.json and docs/prereg/step0_v1.json from a valid synthetic run, written only into a temporary git repo.

What these tests pin: what each file holds, as §4.4 lists it; ASCII JSON, one record a line; byte-identical when
written again, and from a copy of the run elsewhere; an invalid run refused and bad input refused, nothing written; a
different existing file never replaced, neither file written; item 8's readouts, with the A/B-differ flags of every
pair, on a run whose held-out tuple is hand-built; a redo pair recorded with its files; no LoopScript text in either
file; and nothing written but the two files: no .ls file, nothing under data/ or inbox/.

Synthetic runs come from tests/helpers_step0.py, stamped by tests/helpers_step0_laptop.py; nothing here reads a
pulled run, renders or listens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import helpers_step0 as H
import helpers_step0_laptop as L
import lofi_rl.step0 as S
from lofi_rl.brief import Brief
from lofi_rl.rl import junk as J
from lofi_rl.step0 import cli
from lofi_rl.step0 import check as C
from lofi_rl.step0 import prereg as PR
from lofi_rl.step0 import select as X

pytestmark = pytest.mark.laptop   # git reads the stamps' commits: never on a compute node (CLAUDE.md §1)

RUN = "step0-13800001"
FILES = (S.HELDOUT_FILE, S.RUN_FILE)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("step0-prereg")
    repo, c1 = L.make_repo(root / "repo")
    L.stamped_run(root / RUN, repo, c1)
    return repo, c1, root / RUN


@pytest.fixture
def repo(base, tmp_path) -> Path:
    return L.copy_run(base[0], tmp_path / "repo")


@pytest.fixture
def run_dir(base, tmp_path) -> Path:
    return L.copy_run(base[2], tmp_path / RUN)


@pytest.fixture(scope="module")
def docs(base) -> dict:
    """Both files' texts and documents, from the base run."""
    repo, _, run_dir = base
    run = C.load_run(run_dir)
    texts = PR.documents(run, C.evaluate(run, repo=repo))
    return {"texts": texts, "held": json.loads(texts[S.HELDOUT_FILE]), "step0": json.loads(texts[S.RUN_FILE]),
            "run": run}


def tree_files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file() and ".git" not in p.parts}


# --------------------------------------------------------------------------------------------- what they hold
def test_heldout_v1_holds_what_section_4_4_lists(docs):
    held, run = docs["held"], docs["run"]
    assert held["schema"] == S.HELDOUT_SCHEMA == "lofi-rl/heldout/1"
    assert list(held) == ["schema", "registered", "written_by", "run", "render_code", "settings", "rule", "texts",
                          "counts", "tuples", "short", "item8"]
    r = held["run"]
    assert (r["run"], r["job"], r["node"], r["gpu"], r["redo"]) == (RUN, H.JOB, H.NODE, H.GPU, None)
    assert set(r["stamp"]) == {"generate", "score", "render"} and r["validity"]["verdict"] == "valid"
    assert held["render_code"] == run.render_code and held["rule"]["name"] == S.RULE and held["rule"]["mark"] == S.MARK
    st = held["settings"]
    assert st["model"]["revision"] == S.MODEL_REVISION and st["prompt"]["template_sha256"] == S.TEMPLATE_SHA256
    assert st["palette"]["sha256"] == S.PALETTE_SHA256 and st["regex_sha256"] == S.REGEX_SHA256
    assert st["streams_file"]["sha256"] == S.STREAMS_SHA256 and st["sampling"] == dict(S.V0_SAMPLING)
    assert held["rule"]["answers"]["Q4"] == {"distinct": {"heldout": True, "refb": False}}
    assert held["counts"] == {"briefs": 40, "tuples": 40, "clips": 160, "short": 0, "twins": 8}
    comp = run.files["completions.jsonl"]
    assert held["texts"]["sha256"] == hashlib.sha256(comp.read_bytes()).hexdigest()
    assert held["texts"]["file"] == f"runs/{RUN}/summary/completions.jsonl"
    sets = json.loads((run.dir / "summary" / "sets.json").read_text())
    scored = run.scored_rows()
    recs = {r["i"]: r for r in run.records if r["final"]}
    rows = run.text_rows()
    for t, e in zip(held["tuples"], sets["heldout"]):
        assert t["brief_id"] == e["brief_id"] and (t["cell"], t["bpm"], t["tonic"]) == (e["cell"], e["bpm"], e["tonic"])
        assert t["twin_of"] == e["twin_of"] and t["example"] == rows[e["taken"][0]["i"]]["example"]
        assert [c["k"] for c in t["clips"]] == [x["k"] for x in e["taken"]]
        for c, x in zip(t["clips"], e["taken"]):
            assert list(c) == list(PR.CLIP_FIELDS)
            assert {k: c[k] for k in S.TAKEN_FIELDS} == x, "the draw and its render, as sets.json takes them"
            s, rec = scored[x["i"]], recs[x["i"]]
            assert (c["lufs"], c["true_peak_dbtp"], c["p_fix"]) == (rec["lufs"], rec["true_peak_dbtp"], rec["p_fix"])
            assert (c["near_copy"], c["copy_jaccard"]) == (s["near_copy"], s["copy_jaccard"])
            assert (c["A"], c["B"]) == (s["verbatim"]["A"], s["verbatim"]["B"])
            assert hashlib.sha256(rows[x["i"]]["text"].encode()).hexdigest() == c["text_sha256"], "found by its sha"
    assert sum(1 for t in held["tuples"] if t["twin_of"]) == 8
    i8 = held["item8"]
    assert i8["tuples"] == C.item8(run)["tuples"] and i8["summary"]["tuples"] == 40
    assert all(len(t["pairs"]) == 6 and all(set(p) == {"k", "j5", "j5_kd", "A", "B", "differ"} for p in t["pairs"])
               for t in i8["tuples"]), "6 pairs a tuple, each with its J5 in both forms and its A/B-differ flags"


def test_step0_v1_holds_what_section_4_4_lists(docs):
    st, run, held_text = docs["step0"], docs["run"], docs["texts"][S.HELDOUT_FILE]
    assert st["schema"] == S.RUN_SCHEMA == "lofi-rl/step0-run/1"
    assert st["heldout"] == {"file": S.HELDOUT_FILE, "sha256": hashlib.sha256(held_text.encode()).hexdigest()}
    files = {f["path"]: f for f in st["files"]}
    assert set(files) == {f"runs/{RUN}/summary/{n}" for n in S.SUMMARY_FILES}
    for name in S.SUMMARY_FILES:
        data = (run.dir / "summary" / name).read_bytes()
        assert files[f"runs/{RUN}/summary/{name}"] == {"path": f"runs/{RUN}/summary/{name}", "bytes": len(data),
                                                       "sha256": hashlib.sha256(data).hexdigest(), "read": True}
    assert st["validity"] == {"verdict": "valid", "held": 74, "of": 74, "failed": []}
    assert st["counts"]["rm_heldout"]["n"] == 320 and st["counts"]["train"]["pass"] == 3200
    assert st["probe"]["passed"] is True and st["replay"]["same"] == 320
    sets = json.loads((run.dir / "summary" / "sets.json").read_text())
    assert len(st["refb"]) == 400 and [b["brief_id"] for b in st["refb"]] == [e["brief_id"] for e in sets["refb"]]
    for b, e in zip(st["refb"], sets["refb"]):
        assert b["spares"] == e["spares"] and len(b["taken"]) == 4
        assert b["taken"] == [{k: t[k] for k in PR.DRAW_FIELDS} for t in e["taken"]]
    pool = st["pool"]
    assert (pool["briefs"], pool["picks"], pool["rule"]) == (220, 660, S.POOL_BRIEFS)
    assert [p["brief_id"] for p in pool["entries"]] == [e["brief_id"] for e in sets["pool"]]
    ht = st["heldout_text"]
    assert len(ht["flags"]) == 400 and all(len(f["flags"]) == 8 for f in ht["flags"])
    assert ht["draws"] == 0 and ht["shared"] == [] and len(ht["per_cell"]) == 9
    assert st["short"] == {"heldout": [], "refb": [], "pool": []}
    assert st["readouts"]["pool"]["taken"] == 660 and st["texts"] == docs["held"]["texts"]


def test_both_are_ascii_json_one_record_a_line(docs):
    for rel, body in docs["texts"].items():
        assert body.isascii() and body.endswith("}\n") and json.loads(body)
        lines = body.splitlines()
        assert all(json.loads(line.strip().rstrip(",")) for line in lines if line.strip().startswith('{"brief_id"'))
    held = docs["texts"][S.HELDOUT_FILE].splitlines()
    assert sum(1 for x in held if x.startswith('    {"brief_id": "5101:') and '"clips": [' in x) == 40, "a tuple a line"
    assert sum(1 for x in held if x.startswith('      {"brief_id": "5101:') and '"pairs": [' in x) == 40
    step0 = docs["texts"][S.RUN_FILE].splitlines()
    assert sum(1 for x in step0 if x.startswith('    {"brief_id": "596:') and '"spares"' in x) == 400, "a brief a line"
    assert sum(1 for x in step0 if x.startswith('      {"brief_id": "596:') and '"short"' in x) == 220
    assert sum(1 for x in step0 if x.startswith('      {"brief_id": "596:') and '"flags"' in x) == 400
    assert sum(1 for x in step0 if x.startswith('    {"path": "runs/')) == 7


def test_no_loopscript_text_in_either_file(docs):
    run = docs["run"]
    lines = {x.strip() for r in run.rows for x in r["text"].splitlines() if x.strip()}
    assert len(lines) > 100 and any(x.startswith("kick: ") for x in lines)
    for rel, body in docs["texts"].items():
        assert not [x for x in lines if x in body], rel
        for head in ("vamp: ", "keys: ", "kick: ", "A: ", "B: ", "drums: "):     # a text inside a JSON string
            assert f'"{head}' not in body and f"\\n{head}" not in body, (rel, head)


# --------------------------------------------------------------------------------------------- writing them
def test_written_again_they_are_byte_identical_and_nothing_else_is_written(base, repo, run_dir, tmp_path):
    before = tree_files(tmp_path)
    rc, lines = PR.prereg(run_dir, write=True, repo=repo)
    assert rc == 0 and lines[-2] == f"wrote {S.HELDOUT_FILE}, {S.RUN_FILE}"
    new = tree_files(tmp_path) - before
    assert new == {f"repo/{rel}" for rel in FILES}, "the two files and nothing else"
    assert not [p for p in new if p.endswith(".ls") or "/data/" in f"/{p}" or "/inbox/" in f"/{p}"]
    first = {rel: (repo / rel).read_bytes() for rel in FILES}
    rc, lines = PR.prereg(run_dir, write=True, repo=repo)
    assert rc == 0 and lines[-2] == "unchanged: both files already hold this text"
    assert {rel: (repo / rel).read_bytes() for rel in FILES} == first and tree_files(tmp_path) - before == new
    elsewhere = L.copy_run(run_dir, tmp_path / "pulled" / RUN)          # the same run, pulled to another folder
    run = C.load_run(elsewhere)
    again = PR.documents(run, C.evaluate(run, repo=repo))
    assert {rel: body.encode("ascii") for rel, body in again.items()} == first
    held = first[S.HELDOUT_FILE]
    assert json.loads(first[S.RUN_FILE])["heldout"]["sha256"] == hashlib.sha256(held).hexdigest()


def test_without_write_nothing_is_written(base, repo, run_dir, tmp_path):
    before = tree_files(tmp_path)
    rc, lines = PR.prereg(run_dir, repo=repo)
    assert rc == 0 and lines[-1] == "nothing written (--write writes the absent files)"
    assert all("(absent)" in x for x in lines[1:3]) and tree_files(tmp_path) == before


def test_a_different_existing_file_is_never_replaced(base, repo, run_dir):
    (repo / S.RUN_FILE).write_text('{"schema": "lofi-rl/step0-run/1", "an": "earlier run"}\n')
    rc, lines = PR.prereg(run_dir, write=True, repo=repo)
    assert rc == 2 and "refused: docs/prereg/step0_v1.json already holds another text" in lines[-1]
    assert not (repo / S.HELDOUT_FILE).exists(), "neither file is written"
    assert (repo / S.RUN_FILE).read_text() == '{"schema": "lofi-rl/step0-run/1", "an": "earlier run"}\n'
    sub = next(a for a in cli.parser()._actions if isinstance(a, argparse._SubParsersAction))
    assert set(sub.choices["prereg"]._option_string_actions) == {"-h", "--help", "--redo", "--write"}, "no --force"
    with pytest.raises(SystemExit):
        cli.main(["prereg", str(run_dir), "--write", "--force"])


def test_an_invalid_run_is_refused_and_nothing_written(base, repo, run_dir, tmp_path):
    L.edit_json(run_dir / "summary" / "generate.json", lambda d: d["host"].update(stamp=L.stamp(base[1], dirty=1)))
    before = tree_files(tmp_path)
    rc, lines = PR.prereg(run_dir, write=True, repo=repo)
    assert rc == 3 and "is not a valid run (73 of 74 conditions hold; failing: " + C.STAMP_NAME in lines[0]
    assert tree_files(tmp_path) == before


def test_bad_input_is_refused_and_nothing_written(base, repo, run_dir, tmp_path):
    (run_dir / "summary" / "step0.json").unlink()
    before = tree_files(tmp_path)
    rc, lines = PR.prereg(run_dir, write=True, repo=repo)
    assert rc == 2 and "step0.json is missing" in lines[0] and tree_files(tmp_path) == before


def test_the_command_line_writes_into_the_checkout(base, repo, run_dir, monkeypatch, capsys):
    monkeypatch.setattr(C, "ROOT", repo)
    assert cli.main(["prereg", str(run_dir)]) == 0 and not (repo / S.HELDOUT_FILE).exists()
    assert cli.main(["prereg", str(run_dir), "--write"]) == 0
    out = capsys.readouterr().out
    assert f"wrote {S.HELDOUT_FILE}, {S.RUN_FILE}" in out and all((repo / rel).is_file() for rel in FILES)
    (repo / S.HELDOUT_FILE).write_text((repo / S.HELDOUT_FILE).read_text().replace('"twins": 8', '"twins": 7'))
    assert cli.main(["prereg", str(run_dir), "--write"]) == 2
    assert "already holds another text" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------- item 8, a redo pair
def _rewrite_texts(run_dir: Path, texts: dict[int, str]) -> None:
    """Give some held-out rows other texts, as the steps would have written them: completions.jsonl's text,
    junk.score_row's row (with stream, brief_id, k and text_sha256), the render records' canon_sha256 and clip ids,
    and sets.json by select over the files."""
    summ = run_dir / "summary"
    pal = H.palette()
    check = J.CopyCheck(S.PROMPT_VERSION, pal)
    rows = {r["i"]: r for r in map(json.loads, (summ / "completions.jsonl").read_text().splitlines())}
    for i, t in texts.items():
        rows[i]["text"] = t
    (summ / "completions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows.values()))

    def rescore(scored: list[dict]) -> None:
        for j, s in enumerate(scored):
            if s["i"] in texts:
                r = rows[s["i"]]
                scored[j] = {**J.score_row(r, pal, check), "stream": r["stream"], "brief_id": r["brief_id"],
                             "k": r["k"], "text_sha256": H.sha(r["text"])}
    L.edit_rows(summ / "scored.jsonl", rescore)
    scored = [json.loads(x) for x in (summ / "scored.jsonl").read_text().splitlines()]
    by_i = {s["i"]: s for s in scored}

    def recut(d: dict) -> None:
        for rec in d["clips"]:
            if rec["i"] in texts and rec["ok"]:
                rec["canon_sha256"] = by_i[rec["i"]]["canon_sha256"]
                rec["clip_id"] = C.clip_id(rec["canon_sha256"], Brief(**rows[rec["i"]]["brief"]),
                                           f"{pal.id}@{pal.sha256[:8]}", rec["render_cfg_sha8"])
    L.edit_json(summ / "clips.json", recut)
    clips = json.loads((summ / "clips.json").read_text())
    sets = X.sets_json(scored, clips["clips"], streams=H.streams(), render_code=clips["render_code"])
    (summ / "sets.json").write_text(json.dumps(sets, indent=1))


def test_item8_flags_the_pairs_that_differ_in_a_or_b(base, repo, run_dir):
    cell = "high/melancholy"                                    # 5101:0's cell; its rows are i 0-7
    other = {h: next(x for x in H._example("low/wistful").splitlines() if x.startswith(f"{h}: ")) for h in "AB"}

    def swap(text: str, head: str) -> str:
        return "\n".join(other[head] if x.startswith(f"{head}: ") else x for x in text.splitlines()) + "\n"
    _rewrite_texts(run_dir, {2: swap(H.variant_text(cell, "h2"), "A"),
                             3: swap(swap(H.variant_text(cell, "h3"), "A"), "B")})
    run = C.load_run(run_dir)
    report = C.evaluate(run, repo=repo)
    assert report["validity"]["failed"] == []
    held = json.loads(PR.documents(run, report)[S.HELDOUT_FILE])
    (t,) = [t for t in held["item8"]["tuples"] if t["brief_id"] == "5101:0"]
    assert [(p["k"], p["A"], p["B"], p["differ"]) for p in t["pairs"]] == [
        ([0, 1], False, False, False), ([0, 2], True, False, True), ([0, 3], True, True, True),
        ([1, 2], True, False, True), ([1, 3], True, True, True), ([2, 3], False, True, True)]
    assert t["verbatim"] == {"A": 2, "B": 3, "AB": 2}
    assert held["item8"]["summary"]["differ"] == 5 and held["item8"]["summary"]["all_carry"]["AB"] == 39
    (clips,) = [x["clips"] for x in held["tuples"] if x["brief_id"] == "5101:0"]
    assert [(c["A"], c["B"]) for c in clips] == [(True, True), (True, True), (False, True), (False, False)]


def test_a_redo_pair_is_recorded_with_its_files(base, repo, run_dir, tmp_path):
    fix = "src/lofi_rl/step0/select.py"
    c2 = L.commit(repo, "the redo's fix", {
        fix: (repo / fix).read_text() + "# the redo's fix\n",
        S.DECISIONS: L.decisions_text(extra=f"\n- D-023 redo amendment: run {RUN}; changed {fix}\n")})
    redo = tmp_path / "step0-redo-13800002"
    (redo / "summary").mkdir(parents=True)
    for name in ("scored.jsonl", "step0.json", "sets.json", "clips.json"):
        (redo / "summary" / name).write_bytes((run_dir / "summary" / name).read_bytes())
    L.restamp(redo, repo, c2, steps=("score", "render"))
    rc, lines = PR.prereg(run_dir, redo, write=True, repo=repo)
    assert rc == 0 and "with its redo step0-redo-13800002" in lines[0]
    st = json.loads((repo / S.RUN_FILE).read_text())
    assert st["run"]["redo"] == "step0-redo-13800002" and st["run"]["stamp"]["render"] == L.stamp(c2)
    read = {f["path"]: f["read"] for f in st["files"]}
    assert read == {**{f"runs/{RUN}/summary/{n}": n in C.GENERATE_FILES for n in S.SUMMARY_FILES},
                    **{f"runs/step0-redo-13800002/summary/{n}": True for n in C.LATER_FILES}}
    assert json.loads((repo / S.HELDOUT_FILE).read_text())["render_code"]["stamp"] == L.stamp(c2)


# --------------------------------------------------------------------------------------------- the module
def test_prereg_imports_with_the_policy_envs_absent_modules_and_yaml_blocked():
    blocked = ("yaml", "numpy", "mido", "soundfile", "scipy", "pedalboard", "torch", "trl", "vllm", "transformers")
    code = ("import sys\n" + "".join(f"sys.modules[{m!r}] = None\n" for m in blocked)
            + "import lofi_rl.step0.prereg\nprint('IMPORTS_OK')\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PYTHONPATH": str(L.ROOT / "src")}, cwd=L.ROOT)
    assert p.returncode == 0 and "IMPORTS_OK" in p.stdout, p.stderr[-2000:]


def test_the_layout_is_deterministic_and_refuses_nan():
    doc = {"b": [{"x": 1, "y": [1, 2]}, {"x": 2, "y": []}], "a": {"k": 1.5, "s": "\u00e9"}, "c": []}
    assert PR.text(doc) == ('{\n  "b": [\n    {"x": 1, "y": [1, 2]},\n    {"x": 2, "y": []}\n  ],\n'
                            '  "a": {"k": 1.5, "s": "\\u00e9"},\n  "c": []\n}\n')
    with pytest.raises(ValueError):
        PR.text({"x": float("nan")})
    rc, lines = PR.prereg(Path("/nonexistent/step0-1"))
    assert rc == 2 and "is not a folder" in lines[0]
