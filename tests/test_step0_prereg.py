"""The step-0 prereg files (lofi_rl.step0.prereg; docs/design/step0_samples.md §4.4, §5, §6.1): docs/prereg/
heldout_v1.json and docs/prereg/step0_v1.json from a valid synthetic run, written only into a temporary git repo.

What these tests pin: what each file holds, as §4.4 lists it; ASCII JSON, one record a line; byte-identical when
written again, and from a copy of the run elsewhere; an invalid run refused and bad input refused, nothing written; a
different existing file never replaced, neither file written; item 8's readouts, with the A/B-differ flags of every
pair, on a run whose held-out tuple is hand-built; a redo pair recorded with its files; no LoopScript text in either
file; and nothing written but the two files: no .ls file, nothing under data/ or inbox/. Under Q8(b), a short
held-out or Ref_b brief refused until D-023's entry in DECISIONS.md holds the run's short-answer line on a line of its
own (never a sentence, a comment or another entry that quotes it), and with it the files Q8(a) writes, the line
recorded; under "report", the bytes the code wrote before that line existed.

Synthetic runs come from tests/helpers_step0.py, stamped by tests/helpers_step0_laptop.py; nothing here reads a
pulled run, renders or listens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
    cfg8 = {r["render_cfg_sha8"] for r in run.records if r["ok"]}
    assert st["render_cfg_sha8"] in cfg8 and len(cfg8) == 1, "the clip ids' render config (finding R3-6)"
    c, t = held["tuples"][0]["clips"][0], held["tuples"][0]
    brief = Brief(*t["cell"].split("/"), t["bpm"], t["tonic"])
    assert C.clip_id(c["canon_sha256"], brief, st["palette"]["label"], st["render_cfg_sha8"]) == c["clip_id"], (
        "a reader of this file alone recomputes a clip id")
    assert held["registered"][2] == PR.heldout_rule() and "`short`" in held["registered"][3], "R1-6, R3-4"
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
    assert {name: r["same_clip"] for name, r in st["readouts"].items()} == {
        name: {"heldout_clip": 0, "refb_clip": 0, "pool_clip": 0} for name in S.SETS}, "step0.json's, carried over"


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


@pytest.mark.parametrize("passing, distinct, words", [
    ("hard", True, "that pass the hard gates (Q3) and whose canonical texts differ (Q4);"),
    ("hard_not_copy", True, "that pass the hard gates and are not near-copies of the shown example (Q3) and whose "
                            "canonical texts differ (Q4);"),
    ("hard_no_repair", True, "that pass the hard gates without a repair (Q3) and whose canonical texts differ (Q4);"),
    ("hard", False, "that pass the hard gates (Q3), repeats kept (Q4);")])
def test_the_held_out_rule_is_stated_as_the_answers_set_it(monkeypatch, passing, distinct, words):
    """Finding R1-6: heldout_v1.json's sentence follows Q3 and Q4, so under Q4(c) it never says the texts differ."""
    monkeypatch.setattr(S, "PASSING", passing)
    monkeypatch.setattr(S, "DISTINCT", {"heldout": distinct, "refb": False})
    assert PR.heldout_rule() == f"Each brief takes its first 4 draws in k order {words} a short brief is reported, " \
                                "never drawn again (Q8)."


def test_shared_lists_every_train_row_with_a_held_out_clips_text(base, run_dir):
    """Finding R1-4: a train draw that parsed as a held-out clip's text and then failed compile carries that text's
    canon_sha256 and select flags it, so `shared` lists it too: the flags and the rows agree in the one file."""
    summ = run_dir / "summary"
    held = json.loads((summ / "sets.json").read_text())["heldout"][0]["taken"][0]["canon_sha256"]   # 5101:0 k0's

    def fail_compile(rows: list[dict]) -> None:
        next(r for r in rows if (r["brief_id"], r["k"]) == ("596:41", 5)).update(
            canon_sha256=held, ok=False, gate="compile", detail="synthetic: it parsed, then failed compile")
    L.edit_rows(summ / "scored.jsonl", fail_compile)
    scored = [json.loads(x) for x in (summ / "scored.jsonl").read_text().splitlines()]
    clips = json.loads((summ / "clips.json").read_text())
    doc = X.sets_json(scored, clips["clips"], streams=H.streams(), render_code=clips["render_code"])
    (summ / "sets.json").write_text(json.dumps(doc, indent=1))
    L.edit_json(summ / "step0.json", lambda d: d.update(sets=X.readouts(doc, scored)))
    block = PR.heldout_text_block(C.load_run(run_dir))
    shared = [x["train"] for x in block["shared"] if x["canon_sha256"] == held]   # each held-out clip of that text
    assert shared and shared == [[["596:41", 5]]] * len(shared) and block["draws"] == 1
    assert block["draws"] == len({tuple(t) for x in block["shared"] for t in x["train"]}), "one count, both ways"


HEADS = re.compile(r"(?:^|\n)\s*(?:vamp|keys|kick|A|B|sax|drums):\s")      # a LoopScript statement's head


def _strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for k, x in v.items():
            yield k
            yield from _strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from _strings(x)


def text_problems(body: str) -> list[str]:
    """What would put a step-0 text in a committed prereg file (spec §5; finding R3-10): bytes that are not ASCII or
    not the writer's own layout (a hand edit), a string value holding a newline or a LoopScript statement head, or a
    texts pin outside runs/, which git ignores."""
    out = [] if body.isascii() else ["not ASCII"]
    doc = json.loads(body)
    if PR.text(doc) != body:
        out.append("not prereg's own layout")
    out += [f"a string holds a text: {x[:60]!r}" for x in _strings(doc) if "\n" in x or HEADS.search(x)]
    pin = (doc.get("texts") or {}).get("file", "")
    ignored = subprocess.run(["git", "-C", str(L.ROOT), "check-ignore", "-q", pin], capture_output=True).returncode == 0
    if not (pin.startswith("runs/") and ignored):
        out.append(f"the texts pin {pin!r} is not under the ignored runs/")
    return out


def test_neither_file_can_carry_a_text(docs, base, run_dir):
    for rel, body in docs["texts"].items():
        assert text_problems(body) == [], rel
    bad = json.loads(docs["texts"][S.RUN_FILE])
    bad["registered"].append("vamp: Im9 IV13 span 2")
    assert text_problems(PR.text(bad)) == ["a string holds a text: 'vamp: Im9 IV13 span 2'"]
    assert "not prereg's own layout" in text_problems(PR.text(bad).replace('\n  "run"', ' "run"'))
    L.edit_json(run_dir / "summary" / "generate.json", lambda d: d["replay"].update(
        stopped="error: RuntimeError: the engine quoted its prompt\nvamp: Im9 IV13 span 2"))
    run = C.load_run(run_dir)
    report = C.evaluate(run, repo=base[0])
    assert report["validity"]["failed"] == [], "the replay is shown, never judged"
    st = json.loads(PR.documents(run, report)[S.RUN_FILE])
    assert st["replay"]["stopped"] == "error" and text_problems(PR.text(st)) == [], "its stop as a code"


def test_the_committed_prereg_files_carry_no_text():
    have = [rel for rel in FILES if (L.ROOT / rel).is_file()]
    if not have:
        pytest.skip("neither docs/prereg/heldout_v1.json nor step0_v1.json is committed yet (spec §4.4)")
    for rel in have:
        assert text_problems((L.ROOT / rel).read_text(encoding="ascii")) == [], rel


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


@pytest.mark.parametrize("rule", ["report", "ask"])
def test_under_q8b_a_short_held_out_brief_waits_for_shresth(base, repo, tmp_path, monkeypatch, rule):
    """Finding R1-2: under SHORT_RULE "ask" (Q8(b)) prereg refuses, naming the brief, while a held-out or Ref_b brief
    is short, so the sets are not fixed before his answer; under "report" (Q8(a)) the short tuple is reported."""
    run_dir = tmp_path / RUN
    L.stamped_run(run_dir, base[0], base[1], {"5101:2": "a a a b b c G2 G6"})      # three distinct texts: short
    monkeypatch.setattr(S, "SHORT_RULE", rule)
    before = tree_files(tmp_path)
    rc, lines = PR.prereg(run_dir, write=True, repo=repo)
    if rule == "ask":
        assert rc == 2 and lines[-1].startswith("refused: SHORT_RULE is Q8's (b), ask: 1 held-out or Ref_b")
        assert "heldout 5101:2" in lines[-1] and tree_files(tmp_path) == before, "nothing written"
    else:
        assert rc == 0 and json.loads((repo / S.HELDOUT_FILE).read_text())["counts"]["short"] == 1


# --------------------------------------------------------------------------------------------- Q8(b): his answer
SHORT = {"5101:2": "a a a b b c G2 G6", "596:5": "w x y G2 G6 G2 G6 G2"}   # three distinct texts; three passing
ASKED = "2 held-out or Ref_b briefs are short (heldout 5101:2, refb 596:5)"
ANSWER = f"D-023 short answer: run {RUN}; as (a)"
BEFORE = "57b89f7"          # wt/step0 before the answer line (finding N2): what prereg wrote under SHORT_RULE "report"


@pytest.fixture(scope="module")
def short_run(base, tmp_path_factory) -> Path:
    """A valid run with a short held-out brief and a short Ref_b brief (SHORT), stamped with the base repo's commit."""
    run_dir = tmp_path_factory.mktemp("step0-prereg-short") / RUN
    L.stamped_run(run_dir, base[0], base[1], SHORT)
    return run_dir


def answered(repo: Path, *lines: str) -> str:
    """A later commit whose docs/DECISIONS.md also holds `lines`, as D-023 gains his answer after the run: only
    DECISIONS.md changes, which check never compares with the run's commit. Its full sha."""
    return L.commit(repo, "his answer to Q8(b)", {S.DECISIONS: L.decisions_text(extra="\n" + "\n".join(lines) + "\n")})


WHERE = "alone on its line in D-023's entry, at column 0"     # where the refusal says the line goes (finding N12)


def mentions(repo: Path, out: str) -> None:
    """Finding N12: a refusal that asks for the line names every line of DECISIONS.md that mentions the run's answer
    where none is read (here, each line holding RUN), and only those."""
    lines = [n for n, x in enumerate((repo / S.DECISIONS).read_text().splitlines(), 1) if RUN in x]
    named = f"mentions it on line{'s' if len(lines) > 1 else ''} {', '.join(map(str, lines))}, where no answer is read"
    assert (named in out) if lines else ("mentions it on line" not in out), (lines, out)


CASES = [
    ("no line", (), False),
    ("this run, as (a)", (f"- {ANSWER}",), True),
    ("in a fenced block", ("  ```", f"  {ANSWER}", "  ```"), False),       # finding N11: the form an example takes
    ("indented", (f"      {ANSWER}",), False),
    ("in backticks, with a period", (f"- `{ANSWER}`.",), True),
    ("in bold", (f"- **{ANSWER}**",), True),
    ("another run", ("- D-023 short answer: run step0-13800009; as (a)",), False),
    ("another answer", (f"- D-023 short answer: run {RUN}; as (b)",), False),
    ("a note after the answer", (f"- {ANSWER}, in his words",), False),
    ("a bold head", (f"- **D-023 short answer:** run {RUN}; as (a)",), False),
    ("two answers for the run", (f"- {ANSWER}", f"- D-023 short answer: run {RUN}; ask again"), False),
]


@pytest.mark.parametrize("why, lines, writes", CASES, ids=[c[0] for c in CASES])
def test_under_q8b_his_answer_line_lifts_the_refusal(short_run, repo, tmp_path, monkeypatch, why, lines, writes):
    """Finding N2: under SHORT_RULE "ask", prereg reads Shresth's answer for this run from one line of D-023's entry in
    the checkout's docs/DECISIONS.md (prereg.short_answers), at a later commit that changes nothing check compares
    with the run's commit, so no redo. Only that run's line saying "as (a)", with no other answer for it, writes the
    files; anything else refuses, with nothing written: naming the line to add, where it goes and the lines that
    mention it unread (finding N12), or, when D-023 answers the run otherwise, that answer and the line to replace it
    with (finding N8). A fenced or indented line is no answer (finding N11): it is the form an example takes."""
    monkeypatch.setattr(S, "SHORT_RULE", "ask")
    if lines:
        answered(repo, *lines)
    before = tree_files(tmp_path)
    rc, out = PR.prereg(short_run, write=True, repo=repo)
    if writes:
        assert rc == 0 and out[-2] == f"wrote {S.HELDOUT_FILE}, {S.RUN_FILE}", out
        held, st = (json.loads((repo / rel).read_text()) for rel in FILES)
        assert held["counts"]["tuples"] == 39 and [e["brief_id"] for e in held["short"]] == ["5101:2"]
        assert [e["brief_id"] for e in st["short"]["refb"]] == ["596:5"]
        for doc in (held, st):
            assert doc["rule"]["answers"]["Q8"] == {"short_rule": "ask", "short_answer": ANSWER}, "the line applied"
    else:
        assert rc == 2 and out[-1].startswith(f"refused: SHORT_RULE is Q8's (b), ask: {ASKED}"), out
        assert out[-1].endswith("nothing written") and tree_files(tmp_path) == before, "nothing written"
        if why in ("another answer", "a note after the answer", "two answers for the run"):
            assert f"{S.DECISIONS} answers {RUN} " in out[-1], "what D-023 says is named"
            assert f"`{ANSWER}`" in out[-1] and "replace the run's short-answer lines" in out[-1], "N8: the fix"
            assert "add the line" not in out[-1], "N8: adding the line beside another answer refuses again"
        else:
            assert f"add the line `{ANSWER}` to {S.DECISIONS}, {WHERE}" in out[-1], out[-1]
            mentions(repo, out[-1])


QUOTED = [      # finding N7: text that quotes the line, or holds it where no answer stands
    ("prereg's own refusal, pasted", None),         # the refusal itself, as a session records a pending block
    ("a not-yet note", (f"- Not yet: Shresth has not answered. Once he does, add {ANSWER}",)),
    ("a sentence declining it, in backticks", (f"- Shresth declined `{ANSWER}` for this run; he wants (b) again.",)),
    ("a sentence, in double quotes", (f'- He said no to "{ANSWER}".',)),
    ("an HTML comment block", ("<!--", ANSWER, "-->")),
    ("a one-line HTML comment", (f"<!-- {ANSWER} -->",)),
    ("a blockquote", (f"> {ANSWER}",)),
    ("struck through", (f"- ~~{ANSWER}~~",)),
    ("a table cell", ("| question | answer |", "|---|---|", f"| Q8 | {ANSWER} |")),
    ("under the next entry's heading", ("## D-024 (synthetic): the next entry", "", f"- {ANSWER}")),
    # finding N11: a hard-wrapped sentence whose quote lands alone on its next line, as D-019's Outcome wraps
    ("a pending note, wrapped", ("- **Outcome (pending his answer to Q8(b)):** the run is valid; prereg refused its 2 "
                                 "short briefs and waits for", f"  `{ANSWER}`.")),
    ("a not-yet note, wrapped", ("- Not yet: Shresth has not answered Q8(b) for this run. Once he does, and only if he "
                                 "answers (a), add the line", f"  {ANSWER}")),
    ("a declining sentence, wrapped", ("- Shresth declined (a)'s handling for this run's short briefs, so it is not "
                                       "added to this entry:", f"  `{ANSWER}`")),
    ("a nested sentence, wrapped", ("- Pending:", "  - until then the line", f"    `{ANSWER}`",
                                    "    stays out of this entry.")),
    ("a paragraph's next line", ("Not yet: once he answers (a), D-023 gains", ANSWER)),
    ("a fenced example", ("- Once he answers (a), D-023 gains:", "", "  ```", f"  {ANSWER}", "  ```")),
    ("an indented example", ("Once he answers (a), D-023 gains:", "", f"    {ANSWER}")),
    ("a nested bullet", ("- Once he answers (a), D-023 gains:", f"  - {ANSWER}")),
    ("a comment opened after one closes", ("<!-- a --> <!--", f"- {ANSWER}", "-->")),
]
READ = [        # ... and the line of its own, as house style writes it, still reads
    ("in italics", (f"_{ANSWER}_",)),
    ("a numbered item", (f"1. {ANSWER}",)),
    ("after a comment that closes", ("<!-- a note", "on two lines -->", f"- {ANSWER}")),
    ("before the next entry", (f"- {ANSWER}", "", "## D-024 (synthetic): the next entry", "- something else")),
    ("after a fence that holds a comment's start", ("  ```", "  <!--", "  ```", "", f"- {ANSWER}")),
    ("a bullet after a paragraph", ("He answered Q8(b) on 2026-11-20:", f"- {ANSWER}")),
]


@pytest.mark.parametrize("why, lines, writes", [(*c, False) for c in QUOTED] + [(*c, True) for c in READ],
                         ids=[c[0] for c in QUOTED + READ])
def test_under_q8b_his_answer_is_only_a_line_of_its_own_in_d023(short_run, repo, tmp_path, monkeypatch, why, lines,
                                                                 writes):
    """Findings N7, N11: the answer is read only as a line of its own in D-023's entry, as kit.gates reads an approval
    but for one list bullet: at column 0, starting a block, outside code fences and HTML comments. A sentence, a note
    or prereg's own refusal that quotes the line, a hard-wrapped sentence's next line, an indented, nested or fenced
    example, a comment, a blockquote, a table or another entry is no answer at all (nothing is named as one, and the
    refusal still asks for the line, says where it goes and names the lines that mention it, finding N12), so no text
    that only mentions the line writes the files before Shresth answered. The line in italics, as a numbered item, or
    as a bullet right after a paragraph (a list interrupts one) reads."""
    monkeypatch.setattr(S, "SHORT_RULE", "ask")
    if lines is None:
        rc, out = PR.prereg(short_run, repo=repo)
        assert rc == 2 and f"`{ANSWER}`" in out[-1], out
        lines = (f"- **Outcome (pending his answer to Q8(b)):** prereg said: {out[-1]}",)
    answered(repo, *lines)
    before = tree_files(tmp_path)
    rc, out = PR.prereg(short_run, write=True, repo=repo)
    if writes:
        assert rc == 0 and out[-2] == f"wrote {S.HELDOUT_FILE}, {S.RUN_FILE}", out
        for rel in FILES:
            assert json.loads((repo / rel).read_text())["rule"]["answers"]["Q8"]["short_answer"] == ANSWER
    else:
        assert rc == 2 and out[-1].startswith(f"refused: SHORT_RULE is Q8's (b), ask: {ASKED}"), out
        assert f"add the line `{ANSWER}` to {S.DECISIONS}, {WHERE}" in out[-1], out
        assert f" answers {RUN} " not in out[-1], "nothing is named as an answer"
        mentions(repo, out[-1])
        assert tree_files(tmp_path) == before, "nothing written"


def test_under_q8b_the_line_added_after_a_later_entry_is_named_and_moving_it_into_d023_writes(short_run, repo,
                                                                                              tmp_path, monkeypatch):
    """Finding N12: once another entry follows D-023 (D-024, from Phase 2), the line appended at the end of the file
    stands in that entry and is no answer. The refusal says the line goes in D-023's entry and names the line where it
    stands unread, so following it ends the refusal: moved into D-023's entry, it writes."""
    monkeypatch.setattr(S, "SHORT_RULE", "ask")
    later = ("## D-024 (synthetic): the seed pool", "", "- something else", "")
    answered(repo, *later, f"- {ANSWER}")
    rc, out = PR.prereg(short_run, write=True, repo=repo)
    assert rc == 2 and f"add the line `{ANSWER}` to {S.DECISIONS}, {WHERE}" in out[-1], out
    n = (repo / S.DECISIONS).read_text().splitlines().index(f"- {ANSWER}") + 1
    assert f"mentions it on line {n}, where no answer is read: nothing written" in out[-1], out[-1]
    answered(repo, f"- {ANSWER}", "", *later)
    rc, out = PR.prereg(short_run, write=True, repo=repo)
    assert rc == 0 and out[-2] == f"wrote {S.HELDOUT_FILE}, {S.RUN_FILE}", out


NOTED = f"- {ANSWER} (Shresth, 2026-11-20)"         # his answer in house style, attributed after the line
OTHERWISE = [   # finding N8: D-023 answers the run otherwise, beside the line or not
    ("the line with a note", (NOTED,), ["as (a) (Shresth, 2026-11-20)"]),
    ("the line with a note, and the line", (NOTED, f"- {ANSWER}"), ["as (a) (Shresth, 2026-11-20)"]),
    ("the line, and another answer", (f"- {ANSWER}", f"- D-023 short answer: run {RUN}; ask again"), ["ask again"]),
]


@pytest.mark.parametrize("why, lines, others", OTHERWISE, ids=[c[0] for c in OTHERWISE])
def test_under_q8b_the_refusal_says_to_replace_another_answer_not_to_add_the_line(short_run, repo, tmp_path,
                                                                                   monkeypatch, why, lines, others):
    """Finding N8: when D-023 answers the run otherwise, the refusal names only the answers prereg does not apply
    (never the line's own "as (a)") and says to replace the run's lines with the one line, not to add it: added beside
    another answer, the line refused again, naming itself among the answers not applied. Done as it says, prereg
    writes."""
    monkeypatch.setattr(S, "SHORT_RULE", "ask")
    answered(repo, *lines)
    before = tree_files(tmp_path)
    rc, out = PR.prereg(short_run, write=True, repo=repo)
    assert rc == 2 and out[-1].startswith(f"refused: SHORT_RULE is Q8's (b), ask: {ASKED}; "), out
    named = f"; {S.DECISIONS} answers {RUN} {', '.join(repr(a) for a in others)}, which prereg does not apply, "
    assert named in out[-1] and f"`{ANSWER}`" in out[-1] and "add the line" not in out[-1], out[-1]
    assert ("replace the run's short-answer lines in D-023's entry with that one line, leaving only it, and run "
            "prereg again: nothing written") in out[-1] and tree_files(tmp_path) == before
    assert ("beside the line" in out[-1]) == (f"- {ANSWER}" in lines), "the line's own answer is not named"
    answered(repo, f"- {ANSWER}")                       # the run's lines replaced with that one line
    rc, out = PR.prereg(short_run, write=True, repo=repo)
    assert rc == 0 and out[-2] == f"wrote {S.HELDOUT_FILE}, {S.RUN_FILE}", out


def test_under_q8b_off_the_runs_commit_the_line_also_goes_into_a_worktree_of_it(base, short_run, repo, tmp_path,
                                                                                monkeypatch):
    """Finding N9: once main changes a file check reproduces the run with (here one only junk's imports reach, which
    check never runs), prereg refuses there and runs in a git worktree of the run's commit, whose docs/DECISIONS.md
    predates his answer line on main: it refuses there until the line is added to that checkout's DECISIONS.md too,
    then writes the bytes the line writes at the run's commit, to be committed on main. prereg's module doc and spec
    section 10 give that route."""
    monkeypatch.setattr(S, "SHORT_RULE", "ask")
    direct = L.copy_run(base[0], tmp_path / "direct")
    answered(direct, f"- {ANSWER}")
    assert PR.prereg(short_run, write=True, repo=direct)[0] == 0
    want = {rel: (direct / rel).read_bytes() for rel in FILES}
    edit = "src/lofi_rl/render/pipeline.py"
    assert edit in C.reproduction_files(repo)
    answered(repo, f"- {ANSWER}")
    L.commit(repo, "main moves on: an edit of the render pipeline", {edit: (repo / edit).read_text() + "# later\n"})
    rc, out = PR.prereg(short_run, write=True, repo=repo)
    assert rc == 2 and f"check out {base[1][:12]} (a git worktree)" in out[0] and edit in out[0], out
    worktree = tmp_path / "at-the-runs-commit"
    L.git(repo, "worktree", "add", "-q", "--detach", str(worktree), base[1])
    rc, out = PR.prereg(short_run, write=True, repo=worktree)
    assert rc == 2 and f"add the line `{ANSWER}` to {S.DECISIONS}" in out[-1], "its DECISIONS.md predates the line"
    decisions = worktree / S.DECISIONS
    decisions.write_text(decisions.read_text() + f"\n- {ANSWER}\n")      # by hand, uncommitted
    rc, out = PR.prereg(short_run, write=True, repo=worktree)
    assert rc == 0 and {rel: (worktree / rel).read_bytes() for rel in FILES} == want, out
    assert not any((repo / rel).exists() for rel in FILES), "main gets them by a copy and a commit"
    spec = (L.ROOT / "docs" / "design" / "step0_samples.md").read_text(encoding="utf-8")
    texts = {"spec section 10": next(x for x in spec.splitlines() if x.startswith("- **Q8(b), ask**")),
             "prereg's module doc": PR.__doc__.split("**Off the run's commit**", 1)[1].split("\n\n", 1)[0]}
    for where, text in texts.items():
        text = re.sub(r"\s+", " ", text)
        assert "git worktree of the run's commit" in text and "goes into D-023 on main" in text, where
        assert "the worktree's docs/DECISIONS.md" in text and "committed on main" in text, where


def test_under_q8b_his_answer_handles_the_short_briefs_as_q8a_does(short_run, base, repo, tmp_path, monkeypatch):
    """Finding N2: with the answer applied, both files are the ones Q8(a) writes for the run but for rule.answers.Q8,
    which records the line (and step0_v1.json's pin of heldout_v1.json, which that changes): the held-out set holds a
    tuple fewer, and the short Ref_b brief is recorded short, to leave run 1's [0, 400)."""
    report_repo = L.copy_run(base[0], tmp_path / "repo-report")
    assert PR.prereg(short_run, write=True, repo=report_repo)[0] == 0
    want = {rel: json.loads((report_repo / rel).read_text()) for rel in FILES}
    monkeypatch.setattr(S, "SHORT_RULE", "ask")
    answered(repo, f"- {ANSWER}")
    assert PR.prereg(short_run, write=True, repo=repo)[0] == 0
    got = {rel: json.loads((repo / rel).read_text()) for rel in FILES}
    for rel in FILES:
        assert got[rel]["rule"]["answers"]["Q8"] == {"short_rule": "ask", "short_answer": ANSWER}
        got[rel]["rule"]["answers"]["Q8"] = {"short_rule": "report"}
    got[S.RUN_FILE]["heldout"]["sha256"] = want[S.RUN_FILE]["heldout"]["sha256"]
    assert got == want
    refb = {e["brief_id"]: e for e in got[S.RUN_FILE]["refb"]}
    assert refb["596:5"]["short"] is True and len(refb["596:5"]["taken"]) == 3


def _written_at(commit: str, run_dir: Path, repo: Path, tmp_path: Path) -> dict[str, bytes]:
    """Both files as the code of `commit` writes them (prereg --write) for `run_dir` into `repo`, in a subprocess
    whose lofi_rl, streams file and palette are that commit's, from this checkout's history (git archive)."""
    code_dir = tmp_path / f"code-{commit}"
    code_dir.mkdir()
    archive = subprocess.run(["git", "-C", str(L.ROOT), "archive", commit, "src", S.STREAMS_FILE, S.PALETTE],
                             capture_output=True, check=True, timeout=120)
    subprocess.run(["tar", "-x", "-C", str(code_dir)], input=archive.stdout, check=True, timeout=120)
    code = ("import sys\nfrom pathlib import Path\nfrom lofi_rl.step0 import prereg as PR\n"
            "rc, lines = PR.prereg(Path(sys.argv[1]), write=True, repo=Path(sys.argv[2]))\n"
            "print('\\n'.join(lines))\nsys.exit(rc)\n")
    p = subprocess.run([sys.executable, "-B", "-c", code, str(run_dir), str(repo)], capture_output=True, text=True,
                       timeout=600, env={**os.environ, "PYTHONPATH": str(code_dir / "src")}, cwd=code_dir)
    assert p.returncode == 0, (p.stdout + p.stderr)[-3000:]
    return {rel: (repo / rel).read_bytes() for rel in FILES}


def test_under_report_prereg_writes_the_bytes_it_wrote_before_the_answer_line(short_run, base, tmp_path):
    """Finding N2: under SHORT_RULE "report" (Q8(a)) the answer line changes nothing. For a run with a short held-out
    and a short Ref_b brief, and a short-answer line in DECISIONS.md, prereg writes byte for byte what the code at
    BEFORE writes for it."""
    if subprocess.run(["git", "-C", str(L.ROOT), "cat-file", "-e", f"{BEFORE}^{{commit}}"],
                      capture_output=True).returncode:
        pytest.skip(f"commit {BEFORE} is not in this clone")
    assert S.SHORT_RULE == "report"
    repo_now = L.copy_run(base[0], tmp_path / "repo-now")
    answered(repo_now, f"- {ANSWER}")
    repo_then = L.copy_run(repo_now, tmp_path / "repo-then")
    then = _written_at(BEFORE, short_run, repo_then, tmp_path)
    rc, out = PR.prereg(short_run, write=True, repo=repo_now)
    assert rc == 0, out
    assert {rel: (repo_now / rel).read_bytes() for rel in FILES} == then
    assert json.loads(then[S.HELDOUT_FILE])["counts"]["short"] == 1 and b"short_answer" not in then[S.RUN_FILE]


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
