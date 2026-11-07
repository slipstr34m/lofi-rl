"""The step-0 score step (lofi_rl.step0.score; docs/design/step0_samples.md §2.4, §6.1) on synthetic runs
(tests/helpers_step0.py), whose completions.jsonl it scores for real with junk's gates and readouts.

What these tests pin (§6.1): junk.score_row's fields plus stream, brief_id, k and text_sha256, row for row the helper's
scored.jsonl; the per-stream readouts, junk.summarize over each stream's rows against that stream's requests, each
stream with its own CopyCheck; the candidates; no junk.json; score running with the policy env's absent modules blocked
(as tests/test_junk.py runs junk's score); --from writing only under --run; the refusals; a partial run scored as far as
it goes; rows that are not one run's draws.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import helpers_step0 as H
import lofi_rl.step0 as S
from lofi_rl.brief import Brief
from lofi_rl.rl import junk as J
from lofi_rl.step0 import cli
from lofi_rl.step0 import score as Sc
from lofi_rl.step0 import select as X

ROOT = Path(__file__).resolve().parents[1]
# two held-out briefs (a repeat, a G7 and an error in 5101:0; a G2 in 5101:1), a held-out twin's train brief with a
# held-out text and a G6, and a train brief with a G2
TABLE = {"5101:0": "a b a/G7 c d/E e f g", "5101:1": "h0 G2 h1 h2 h3 h4 h5 h6", "596:177": "t0 t1 t2 t3 a b c G6",
         "596:0": "t0 G2 t1 t2 t3 t4 t5 t6"}
NAME = f"step0-{H.JOB}"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rows_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def doc_of(path: Path) -> dict:
    return json.loads(path.read_text())


def hashes(folder: Path) -> dict[str, str]:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


@pytest.fixture
def run(tmp_path):
    """The helper's run of TABLE, with score's two files taken away (kept as `want`)."""
    r = H.make_run(tmp_path / NAME, TABLE, briefs=list(TABLE))
    r.want = {"scored.jsonl": rows_of(r.summary / "scored.jsonl"), "step0.json": doc_of(r.summary / "step0.json")}
    (r.summary / "scored.jsonl").unlink()
    (r.summary / "step0.json").unlink()
    return r


def test_scored_rows_are_junks_plus_the_four_fields(run):
    assert cli.main(["score", "--run", str(run.dir)]) == 0
    got = rows_of(run.summary / "scored.jsonl")
    assert got == run.want["scored.jsonl"], "row for row the helper's: junk.score_row plus the four fields"
    assert [r["i"] for r in got] == [r["i"] for r in run.rows], "in i order"
    for s, c in zip(got, run.rows):
        fields = [f for f in S.SCORE_ROW_FIELDS if f != "canon_sha256" or "canon_sha256" in s]
        assert list(s) == fields + list(Sc.EXTRA) == [f for f in S.FIELDS["scored.jsonl"] if f in s]
        assert (s["stream"], s["brief_id"], s["k"], s["text_sha256"]) == (c["stream"], c["brief_id"], c["k"],
                                                                          sha(c["text"]))
        assert {k: v for k, v in s.items() if k not in Sc.EXTRA} == J.score_row(c, H.palette(),
                                                                                  J.CopyCheck("v2", H.palette()))
    assert {s["gate"] for s in got if not s["ok"]} == {"G2", "G6"}


def test_step0_json_holds_the_readouts_per_stream_and_the_candidates(run):
    assert cli.main(["score", "--run", str(run.dir)]) == 0
    doc, want = doc_of(run.summary / "step0.json"), run.want["step0.json"]
    assert list(doc) == list(S.FIELDS["step0.json"]) and doc["schema"] == S.STEP0_SCHEMA
    assert (doc["run"], doc["n"], doc["requested"], doc["partial"], doc["unreadable_lines"]) == (NAME, 32, 32, False, 0)
    assert all(doc[k] is None for k in ("render", "render_code", "sets", "render_stamp")), "the render step's"
    assert doc["score_stamp"] == Sc.sync_stamp()
    assert list(doc["by_stream"]) == ["rm_heldout", "train"]
    for name, n in (("rm_heldout", 16), ("train", 16)):
        got, helper = doc["by_stream"][name], want["by_stream"][name]
        assert {k: v for k, v in got.items() if k != "score_stamp"} == {k: v for k, v in helper.items()
                                                                       if k != "score_stamp"}, name
        assert (got["n"], got["requested"], got["partial"], got["schema"]) == (n, n, False, J.SCHEMA)
    assert doc["by_stream"]["rm_heldout"]["failures"]["G2"] == 1 and doc["by_stream"]["train"]["failures"]["G6"] == 1
    scored = rows_of(run.summary / "scored.jsonl")
    assert doc["candidates"] == want["candidates"] == X.candidates_block(scored, streams=H.streams())
    assert doc["candidates"]["refb"] == {"596:0": [0, 2, 3, 4], "596:177": [0, 1, 2, 3]}, "Ref_b by its records"
    assert doc["candidates"]["heldout_order"] == [r["i"] for r in run.rows if r["brief_id"] == "5101:0" and
                                                  r["k"] in (0, 1, 3, 4)] + [r["i"] for r in run.rows
                                                                              if r["brief_id"] == "5101:1" and
                                                                              r["k"] in (0, 2, 3, 4)]


def test_score_never_writes_a_junk_json(run):
    assert cli.main(["score", "--run", str(run.dir)]) == 0
    assert sorted(p.name for p in run.summary.iterdir()) == sorted(S.SUMMARY_FILES)
    assert not list(run.dir.rglob("junk.json"))


def test_each_stream_is_scored_with_its_own_copy_check(run, monkeypatch, capsys):
    """A knob readout that fails on one train brief's rows nulls train's knob counts only (596:0, which no held-out
    brief equals; 596:177 is 5101:0's twin, so its brief is a held-out brief's too)."""
    brief = {Brief(**r["brief"]) for r in run.rows if r["brief_id"] == "596:0"}
    assert not brief & {Brief(**r["brief"]) for r in run.rows if r["stream"] == "rm_heldout"}
    real = J.knob_readout

    def knobs(text, b, palette, finish_reason):
        if b in brief:
            raise RuntimeError("verifier exploded")
        return real(text, b, palette, finish_reason)
    monkeypatch.setattr(J, "knob_readout", knobs)
    assert cli.main(["score", "--run", str(run.dir)]) == 0
    by = doc_of(run.summary / "step0.json")["by_stream"]
    assert by["train"]["copy"]["knobs_passing"] is None and by["train"]["copy"]["knobs_error"]["loops"] == 7
    assert by["rm_heldout"]["copy"]["knobs_passing"]["n"] == 15 and by["rm_heldout"]["copy"]["knobs_error"] is None
    assert "score: train: the knob readout failed on 7 loop(s)" in capsys.readouterr().err


def test_score_runs_with_the_policy_envs_missing_modules_blocked(run):
    """mido, soundfile, scipy, pedalboard and numpy are blocked (the policy env has none of the audio stack, and its
    numpy is not the laptop's), and vLLM, torch and transformers too: score never needs them, the knob readout
    included."""
    code = f"""
import sys
BLOCKED = ("mido", "soundfile", "scipy", "pedalboard", "numpy", "torch", "vllm", "transformers", "trl")
for m in BLOCKED:
    sys.modules[m] = None
from lofi_rl.step0 import cli
rc = cli.main(["score", "--run", {str(run.dir)!r}])
leaked = [m for m in BLOCKED if sys.modules.get(m) is not None]
print("RC", rc, "LEAKED", leaked)
"""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300, cwd=ROOT)
    assert out.returncode == 0 and "RC 0 LEAKED []" in out.stdout, out.stderr[-3000:]
    scored = rows_of(run.summary / "scored.jsonl")
    assert scored == run.want["scored.jsonl"]
    assert all(s["energy_match"] is not None for s in scored if s["ok"]), "the knob readout ran with numpy blocked"


def test_from_scores_outs_records_into_redo_and_never_writes_under_out(run, tmp_path):
    assert cli.main(["score", "--run", str(run.dir)]) == 0
    before = hashes(run.dir)
    redo = tmp_path / "step0-redo-13800002"
    assert cli.main(["score", "--from", str(run.dir), "--run", str(redo)]) == 0
    assert hashes(run.dir) == before, "OUT is read only"
    assert sorted(p.relative_to(redo).as_posix() for p in redo.rglob("*") if p.is_file()) == [
        "summary/scored.jsonl", "summary/step0.json"]
    assert rows_of(redo / "summary" / "scored.jsonl") == rows_of(run.summary / "scored.jsonl")
    a, b = doc_of(redo / "summary" / "step0.json"), doc_of(run.summary / "step0.json")
    assert a["run"] == "step0-redo-13800002" and {blk["run"] for blk in a["by_stream"].values()} == {a["run"]}

    def named(doc: dict, name: str) -> dict:
        return {**doc, "run": name, "by_stream": {s: {**blk, "run": name} for s, blk in doc["by_stream"].items()}}
    assert named(a, NAME) == b, "the same records score the same, under the folder's own name"


@pytest.mark.parametrize("where", ["same", "inside", "around"])
def test_from_and_run_that_overlap_are_refused(run, tmp_path, where, capsys):
    target = {"same": run.dir, "inside": run.dir / "redo", "around": run.dir.parent}[where]
    before = hashes(run.dir)
    assert cli.main(["score", "--from", str(run.dir), "--run", str(target)]) == 2
    assert "overlap" in capsys.readouterr().err and hashes(run.dir) == before
    assert not (run.dir / "redo").exists() and not (run.summary / "scored.jsonl").exists()


def test_bad_input_is_refused_with_nothing_written(run, capsys):
    gen = run.summary / "generate.json"
    meta = doc_of(gen)
    gen.write_text(json.dumps({**meta, "schema": "lofi-rl/junk-generate/1"}))
    assert cli.main(["score", "--run", str(run.dir)]) == 2 and "not a step-0 generate.json" in capsys.readouterr().err
    gen.unlink()
    assert cli.main(["score", "--run", str(run.dir)]) == 2
    (run.summary / "completions.jsonl").unlink()
    assert cli.main(["score", "--run", str(run.dir)]) == 2 and "run generate first" in capsys.readouterr().err
    assert not (run.summary / "scored.jsonl").exists() and not (run.summary / "step0.json").exists()


def test_a_partial_run_is_scored_as_far_as_it_goes(run):
    comp = run.summary / "completions.jsonl"
    rows = rows_of(comp)
    comp.write_text("".join(json.dumps(r) + "\n" for r in rows if r["brief_id"] != "596:0") + '{"i": 3')
    gen = doc_of(run.summary / "generate.json")
    (run.summary / "generate.json").write_text(json.dumps({**gen, "stopped": "deadline", "generated": 24}))
    assert cli.main(["score", "--run", str(run.dir)]) == 0
    doc = doc_of(run.summary / "step0.json")
    assert (doc["n"], doc["requested"], doc["partial"], doc["unreadable_lines"]) == (24, 32, True, 1)
    assert doc["by_stream"]["train"]["n"] == 8 and doc["by_stream"]["train"]["partial"] is True
    assert set(doc["candidates"]["passing"]) == {"5101:0", "5101:1", "596:177"}, "the briefs it holds"


def test_rows_that_are_not_one_runs_draws_leave_candidates_null(run, capsys):
    comp = run.summary / "completions.jsonl"
    rows = rows_of(comp)
    comp.write_text("".join(json.dumps(r) + "\n" for r in rows + [rows[3]]))
    assert cli.main(["score", "--run", str(run.dir)]) == 1
    doc = doc_of(run.summary / "step0.json")
    assert doc["candidates"] is None and doc["n"] == 33 and "repeated" in capsys.readouterr().err
    assert len(rows_of(run.summary / "scored.jsonl")) == 33, "every row is still scored, for the record"


def test_stream_requested_reads_generate_jsons_streams():
    meta = {"streams": {"rm_heldout": {"seed": 5101, "briefs": 40, "draws": 8}}}
    assert Sc.stream_requested(meta, "rm_heldout", 3) == 320 and Sc.stream_requested(meta, "train", 7) == 7
    assert Sc.stream_requested({"streams": None}, "train", 5) == 5
