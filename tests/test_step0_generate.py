"""The step-0 draws (lofi_rl.step0.generate; docs/design/step0_samples.md §2.3, §6.1) on a fake engine, as
tests/test_junk.py runs junk's generate: the laptop has no vLLM.

The pre-checks read a code tree (tests/helpers_step0_engine.py's code_tree): a temporary copy of what the path file
lists, with a docs/DECISIONS.md that holds D-023's mark and a path file written into that tree. The repo's own
docs/prereg/D-023_step0_path.txt and D-023 are written at the code commit, with Shresth's answers, so no test reads or
writes them. The tree is only read: the code that runs is this checkout's.

What these tests pin (§6.1): one SamplingParams per request, n 1, its seed, one list per call, with one shared
StructuredOutputsParams; the chunks, their fsync and the deadline (`stopped: deadline`, a partial run); the probe
passing with seeds honoured and stopping with seeds ignored (exit 2, nothing drawn); each pre-check refusing before the
engine loads (a mark that differs, a path-file blob that differs, PROBE_SEED among the draw seeds, and the rest); the
replay in replay.jsonl only, in reverse order, with its counts; generate.json's fields; the watchdog.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter

import pytest

import helpers_step0 as H
import helpers_step0_engine as E
import lofi_rl.step0 as S
from lofi_rl import grammar, prompt_v2
from lofi_rl.prompt import PromptError, messages_sha256
from lofi_rl.rl import junk as J
from lofi_rl.step0 import cli
from lofi_rl.step0 import generate as G
from lofi_rl.step0 import requests as R

ROOT = G.ROOT
SMALL = ("5101:0", "5101:1", "596:0", "596:177")      # two held-out briefs, a train brief and a held-out twin's
DECISIONS_TEXT, FakeEngine, INFO, PINNED = E.DECISIONS_TEXT, E.FakeEngine, E.INFO, E.PINNED


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return E.code_tree(tmp_path_factory.mktemp("code") / "tree")


@pytest.fixture
def own_tree(tmp_path, tree):
    """A copy of the tree this test may change."""
    dst = tmp_path / "tree"
    shutil.copytree(tree, dst)
    return dst


@pytest.fixture
def subset(monkeypatch):
    """Draw only SMALL's 32 requests (each keeping its place i in the full order), not all 3,520."""
    return E.draw_only(monkeypatch, SMALL)


def run_generate(monkeypatch, tree, run, engine=None, *, args=(), made=None):
    """generate in `tree` through the command line; the engine and the StructuredOutputsParams are fakes."""
    loads = []
    engine = FakeEngine() if engine is None else engine
    monkeypatch.setattr(G, "ROOT", tree)
    monkeypatch.setattr(J, "load_engine", lambda model, **kw: loads.append((model, kw)) or engine)
    built = [] if made is None else made
    monkeypatch.setattr(J, "structured_outputs_params",
                        lambda regex: built.append(regex) or {"regex_sha256": sha(regex)})
    rc = cli.main(["generate", "--run", str(run), "--hang-s", "0", *args])
    summ = run / "summary"
    meta = json.loads((summ / "generate.json").read_text())
    return rc, meta, loads, engine


def rows_of(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else None


# --------------------------------------------------------------------------------------------- the draws
def test_every_request_is_drawn_with_its_own_seed_one_list_per_call(monkeypatch, tree, tmp_path):
    made: list = []
    rc, meta, loads, engine = run_generate(monkeypatch, tree, tmp_path / "run", made=made)
    assert rc == 0 and meta["stopped"] is None and meta["partial"] is False
    reqs = R.requests(streams=H.streams())
    assert meta["requested"] == meta["generated"] == len(reqs) == 3520
    assert loads == [(f"{S.MODEL}@{S.MODEL_REVISION}", {"seed": 0, "gpu_mem": 0.85, "max_model_len": 4096,
                                                         "backend": "xgrammar"})], "the gate run's engine, seed 0"
    rx = grammar.loopscript_regex(H.palette())
    assert made == [rx], "one StructuredOutputsParams, built once from the frozen palette's regex"
    shared = {"regex_sha256": sha(rx)}
    draws = engine.calls[1:-2]                          # the probe first, the replay's two chunks last
    assert [len(c) for c, _ in draws] == [128] * 27 + [64]
    sent = [sp for _, sps in draws for sp in sps]
    assert [sp.seed for sp in sent] == [r.draw_seed for r in reqs], "each request its registered seed, in order"
    for sp in sent:
        assert sp.n == 1 and sp.structured_outputs == shared
        assert {k: v for k, v in vars(sp).items() if k not in ("n", "seed", "structured_outputs")} == J.SAMPLING
    assert len({id(sp) for sp in sent}) == 3520, "one object per request"
    assert all(sp.structured_outputs is sent[0].structured_outputs for sp in sent), "the grammar object is shared"
    rows = rows_of(tmp_path / "run" / "summary" / "completions.jsonl")
    assert [r["i"] for r in rows] == list(range(3520)) and all(list(r) == list(S.FIELDS["completions.jsonl"])
                                                               for r in rows)
    for r, q in zip(rows, reqs):
        assert (r["stream"], r["seed"], r["index"], r["brief_id"], r["k"]) == (q.stream, q.seed, q.index,
                                                                               q.brief_id, q.k)
        assert r["draw_seed"] == r["sp_seed"] == q.draw_seed and r["brief"] == J.brief_dict(q.brief)
    convs = [c for c, _ in draws]
    flat = [conv for call in convs for conv in call]
    for r, q, conv in zip(rows, reqs, flat):
        own = prompt_v2.build_messages(q.brief, H.palette())
        assert conv == own and r["prompt_sha256"] == messages_sha256(own), "a cell's briefs share their messages"
        assert r["example"] == prompt_v2.example_id(q.brief) and r["prompt_version"] == "v2"
        assert r["prompt_tokens"] == S.PROMPT_TOKENS[q.cell] and r["text"] == f"loop {q.cell} {q.draw_seed}\n"
    assert meta["sampling_params"] == repr(sent[0]) == f"FakeSP(n=1, seed={reqs[0].draw_seed})"
    assert [c["count"] for c in meta["timing"]["chunks"]] == [128] * 27 + [64]
    assert meta["streams"] == {"rm_heldout": {"seed": 5101, "briefs": 40, "draws": 8},
                               "train": {"seed": 596, "briefs": 400, "draws": 8}}


def test_generate_json_holds_the_registered_fields_and_values(monkeypatch, tree, tmp_path, subset):
    rc, meta, _, engine = run_generate(monkeypatch, tree, tmp_path / "run")
    assert rc == 0 and list(meta) == list(S.FIELDS["generate.json"]), "no trace on a clean run"
    for key, fields in S.BLOCKS.items():
        name, _, dotted = key.partition(":")
        if name != "generate.json":
            continue
        node = meta
        for part in dotted.split("."):
            node = node[part] if part != "*" else next(iter(node.values()))
        assert list(node) == list(fields), key
    assert meta["schema"] == S.GENERATE_SCHEMA and meta["requested"] == meta["generated"] == 32
    assert meta["streams_file"] == {"path": S.STREAMS_FILE, "sha256": S.STREAMS_SHA256, "fresh": True}
    assert meta["registration"] == {"found": True, "mark": S.MARK,
                                    "decisions_sha256": sha((tree / S.DECISIONS).read_text())}
    assert meta["path_file"] == {"sha256": hashlib.sha256((tree / S.PATH_FILE).read_bytes()).hexdigest(),
                                 "blobs_ok": True}
    assert meta["model"] == {"repo": S.MODEL, "revision": S.MODEL_REVISION}
    assert meta["palette"] == {"id": "lofi_rain_v1", "sha256": S.PALETTE_SHA256, "path": S.PALETTE}
    assert meta["sampling"] == J.SAMPLING and list(meta["sampling"]) == list(S.V0_SAMPLING)
    assert meta["per_request_seed"] is True
    rx = grammar.loopscript_regex(H.palette())
    assert meta["constrained"] == {"enabled": True, "regex_sha256": S.REGEX_SHA256, "regex_chars": len(rx),
                                   "builder": "lofi_rl.grammar.loopscript_regex", "backend": PINNED}
    assert meta["engine"] == INFO, "VllmEngine.info as it loaded, without the per-request sampling_params"
    cells = dict.fromkeys(r.cell for r in subset)
    assert meta["prompt"] == J.prompt_record("v2", H.palette(), [prompt_v2.build_messages(J.cell_brief(
        *c.split("/")), H.palette()) for c in cells])
    assert meta["prompt"]["template_sha256"] == S.TEMPLATE_SHA256
    assert meta["streams"] == R.stream_counts(subset)
    assert meta["host"]["stamp"] is None and set(meta["host"]) == {"node", "job", "stamp", "gpu"}
    assert meta["timing"]["load_s"] == 1.5 and meta["timing"]["tokens"] == 32 * 100


def test_the_host_stamp_is_the_trees_sync_stamp(monkeypatch, own_tree, tmp_path, subset):
    (own_tree / ".sync-stamp").write_text("commit=9ef969e00000 dirty=0 diff=- at=2026-11-07T19:00:00Z\n")
    rc, meta, *_ = run_generate(monkeypatch, own_tree, tmp_path / "run")
    assert rc == 0 and meta["host"]["stamp"] == "commit=9ef969e00000 dirty=0 diff=- at=2026-11-07T19:00:00Z"


def test_each_chunk_is_fsynced_and_generate_json_rewritten(monkeypatch, tree, tmp_path, subset):
    synced, written = [], []
    real_fsync, real_write = os.fsync, G._write_json
    monkeypatch.setattr(os, "fsync", lambda fd: synced.append(fd) or real_fsync(fd))
    monkeypatch.setattr(G, "_write_json", lambda path, obj: written.append(obj["generated"]) or real_write(path, obj))
    rc, meta, *_ = run_generate(monkeypatch, tree, tmp_path / "run", args=("--chunk", "8"))
    assert rc == 0 and [c["count"] for c in meta["timing"]["chunks"]] == [8, 8, 8, 8]
    assert len(synced) == 4 + 1, "each of the 4 draw chunks and the replay's one chunk of 16"
    assert [w for w in written if w] == [8, 16, 24, 32, 32, 32], "after each chunk, the replay and the end"


def test_a_chunk_the_forecast_puts_past_the_deadline_is_not_started(monkeypatch, tree, tmp_path, subset):
    """The probe's 4 requests take 4 s and each chunk of 8 takes 8 s. Against a 30-s deadline, chunks start at 4, 12
    and 20 s (20 + 1.25 x 8 = 30 is not past it) and the fourth, at 28 s, is not started: 24 of 32 drawn."""
    clock = [0.0]
    monkeypatch.setattr(G, "_now", lambda: clock[0])
    engine = FakeEngine(clock=clock, per_request_s=1.0)
    rc, meta, _, engine = run_generate(monkeypatch, tree, tmp_path / "run", engine,
                                       args=("--chunk", "8", "--deadline-s", "30"))
    assert rc == 1 and meta["stopped"] == "deadline" and meta["partial"] is False
    assert (meta["generated"], meta["requested"]) == (24, 32) and len(engine.calls) == 1 + 3
    assert meta["replay"]["stopped"] == "not run: the draws are not complete" and meta["replay"]["n"] == 0
    assert not (tmp_path / "run" / "summary" / "replay.jsonl").exists()
    assert len(rows_of(tmp_path / "run" / "summary" / "completions.jsonl")) == 24, "every finished chunk is kept"


def test_an_engine_error_keeps_the_finished_chunks(monkeypatch, tree, tmp_path, subset):
    engine = FakeEngine(fail_at=3)                      # the probe, one chunk, then the error
    rc, meta, *_ = run_generate(monkeypatch, tree, tmp_path / "run", engine, args=("--chunk", "8"))
    assert rc == 1 and meta["stopped"] == "error: RuntimeError: the fake engine died" and meta["generated"] == 8
    assert "RuntimeError" in meta["trace"] and list(meta)[-1] == "trace"
    assert len(rows_of(tmp_path / "run" / "summary" / "completions.jsonl")) == 8


def test_a_short_engine_output_is_an_error(monkeypatch, tree, tmp_path, subset):
    class Short(FakeEngine):
        def chat(self, convs, sps):
            out = super().chat(convs, sps)
            return out if len(self.calls) == 1 else out[:-1]
    rc, meta, *_ = run_generate(monkeypatch, tree, tmp_path / "run", Short())
    assert rc == 1 and "the engine returned 31 completions for 32 requests" in meta["stopped"]
    assert rows_of(tmp_path / "run" / "summary" / "completions.jsonl") == [], "a misaligned chunk is not written"


# --------------------------------------------------------------------------------------------- the probe
def test_the_probe_passes_with_seeds_honoured(monkeypatch, tree, tmp_path, subset):
    rc, meta, _, engine = run_generate(monkeypatch, tree, tmp_path / "run")
    assert rc == 0
    convs, sps = engine.calls[0]
    want = prompt_v2.build_messages(J.cell_brief("low", "hopeful"), H.palette())
    assert len(convs) == S.PROBE_N == 4 and all(c == want for c in convs), "the low/hopeful cell's messages"
    assert [sp.seed for sp in sps] == [S.PROBE_SEED] * 4 and len({id(sp) for sp in sps}) == 4
    text = f"loop low/hopeful {S.PROBE_SEED}\n"
    assert meta["probe"] == {"seed": 1, "cell": "low/hopeful", "n": 4, "texts_sha256": [sha(text)] * 4,
                             "distinct": 1, "passed": True}
    assert meta["timing"]["probe_s"] is not None


def test_the_probe_stops_generate_when_seeds_are_ignored(monkeypatch, tree, tmp_path, subset):
    rc, meta, loads, engine = run_generate(monkeypatch, tree, tmp_path / "run", FakeEngine(honour_seeds=False))
    assert rc == 2 and meta["stopped"] == "seed probe" and meta["generated"] == 0 and len(loads) == 1
    assert len(engine.calls) == 1, "nothing is drawn after a missed probe"
    assert meta["probe"]["distinct"] == 4 and meta["probe"]["passed"] is False
    assert not (tmp_path / "run" / "summary" / "completions.jsonl").exists()
    assert not (tmp_path / "run" / "summary" / "replay.jsonl").exists()


def test_the_probe_passes_with_two_texts_of_four_identical(monkeypatch, tree, tmp_path, subset):
    texts = iter(["x\n", "y\n", "x\n", "z\n"])

    def text_of(cell, key):
        return (next(texts) if key == S.PROBE_SEED else f"loop {key}\n"), "stop", 10, S.PROMPT_TOKENS[cell]
    rc, meta, *_ = run_generate(monkeypatch, tree, tmp_path / "run", FakeEngine(text_of))
    assert rc == 0 and (meta["probe"]["distinct"], meta["probe"]["passed"]) == (3, True), "PROBE_SAME is 2"


# --------------------------------------------------------------------------------------------- the pre-checks
def refused(monkeypatch, tree, run, *needles):
    """generate in `tree` stops at its pre-checks: exit 2, each needle in `stopped`, the engine never loaded and
    nothing drawn."""
    rc, meta, loads, _ = run_generate(monkeypatch, tree, run)
    assert rc == 2 and loads == [], "no engine for a refused run"
    assert meta["stopped"].startswith("pre-check: ") and meta["partial"] is False and meta["generated"] == 0
    for n in needles:
        assert n in meta["stopped"], (n, meta["stopped"])
    assert not (run / "summary" / "completions.jsonl").exists()
    assert list(meta) == list(S.FIELDS["generate.json"])
    return meta


def test_a_mark_that_differs_is_refused(monkeypatch, own_tree, tmp_path):
    other = S.MARK.replace("step0-select/1", "step0-select/2")
    (own_tree / S.DECISIONS).write_text(DECISIONS_TEXT.replace(S.MARK, other))
    meta = refused(monkeypatch, own_tree, tmp_path / "run", "D-023 mark is", "not MARK")
    assert meta["registration"]["found"] is False and meta["registration"]["mark"] == other
    (own_tree / S.DECISIONS).write_text("# Decisions\n\nno entry yet\n")
    meta = refused(monkeypatch, own_tree, tmp_path / "run2", "holds no 'D-023 mark:' line")
    assert meta["registration"] == {"found": False, "mark": None,
                                    "decisions_sha256": sha("# Decisions\n\nno entry yet\n")}
    (own_tree / S.DECISIONS).unlink()
    refused(monkeypatch, own_tree, tmp_path / "run3", f"cannot read {S.DECISIONS}")


def test_a_mark_that_is_not_the_streams_line_is_refused(monkeypatch, own_tree, tmp_path):
    """MARK must be the line STREAMS and RULE give: a MARK edited without STREAMS is refused, even where DECISIONS.md
    holds it."""
    edited = S.MARK.replace("3520 requests", "3521 requests")
    monkeypatch.setattr(S, "MARK", edited)
    (own_tree / S.DECISIONS).write_text(DECISIONS_TEXT.replace(R.mark_line(), edited))
    meta = refused(monkeypatch, own_tree, tmp_path / "run", "draw only the streams D-023's mark names")
    assert meta["registration"]["found"] is True, "the tree holds this MARK; the streams do not give it"


@pytest.mark.parametrize("line", [f"- The mark: `{S.MARK}`", f"{S.MARK}", f"**{S.MARK}.**", f'"{S.MARK}"',
                                  f"Mark (one line): {S.MARK}", f"- `{S.MARK}` (checked by generate)"])
def test_find_marks_reads_the_mark_as_a_decisions_line_may_hold_it(line):
    assert G.find_marks(f"# D-023\n{line}\nmore text\n") == [S.MARK]


@pytest.mark.parametrize("line", [f"{S.MARK} and more", f"**{S.MARK}; and more**", f"{S.MARK[:-1]}"])
def test_text_after_an_unquoted_mark_is_part_of_it(line):
    assert G.find_marks(f"# D-023\n{line}\n") != [S.MARK], "only a backtick or a double quote closes a mark early"


def test_a_path_file_blob_that_differs_is_refused(monkeypatch, own_tree, tmp_path):
    sel = own_tree / "src/lofi_rl/step0/select.py"
    sel.write_text(sel.read_text() + "# an edit after the path file\n")
    meta = refused(monkeypatch, own_tree, tmp_path / "run", "1 differ (first src/lofi_rl/step0/select.py)")
    assert meta["path_file"]["blobs_ok"] is False and meta["path_file"]["sha256"] is not None
    (own_tree / "kit.env").unlink()
    refused(monkeypatch, own_tree, tmp_path / "run2", "1 are missing (first kit.env)")
    (own_tree / S.PATH_FILE).unlink()
    meta = refused(monkeypatch, own_tree, tmp_path / "run3", "the path file: cannot read the path file")
    assert meta["path_file"] == {"sha256": None, "blobs_ok": False}


def test_probe_seed_among_the_draw_seeds_is_refused(monkeypatch, tree, tmp_path):
    monkeypatch.setattr(S, "PROBE_SEED", 379233320)     # 596:0 k0's draw seed (brief_streams_v1.json)
    refused(monkeypatch, tree, tmp_path / "run", "PROBE_SEED 379233320 is one of docs/prereg/brief_streams_v1.json's")


def test_an_edited_streams_file_is_refused(monkeypatch, own_tree, tmp_path):
    f = own_tree / S.STREAMS_FILE
    f.write_bytes(f.read_bytes().replace(b'"draw_seed": 379233320', b'"draw_seed": 379233321', 1))
    meta = refused(monkeypatch, own_tree, tmp_path / "run", "the streams file is refused",
                   "not the registered bb2daee5", "differ (first docs/prereg/brief_streams_v1.json)")
    assert meta["streams_file"]["fresh"] is False and meta["streams_file"]["sha256"] == hashlib.sha256(
        f.read_bytes()).hexdigest()
    assert meta["requested"] is None and meta["streams"] is None, "no request is known from a refused file"


@pytest.mark.parametrize("name, needle", [("PALETTE_SHA256", "not the frozen"), ("REGEX_SHA256", "the palette's regex"),
                                          ("TEMPLATE_SHA256", "prompt v2's template")])
def test_another_palette_regex_or_template_sha_is_refused(monkeypatch, tree, tmp_path, name, needle):
    monkeypatch.setattr(S, name, "0" * 64)
    refused(monkeypatch, tree, tmp_path / "run", needle)


def test_a_message_that_does_not_build_is_refused(monkeypatch, tree, tmp_path):
    real = prompt_v2.build_messages

    def build(brief, palette):
        if brief.energy == "high":
            raise PromptError("example l9_hopeful_high needs maj9")
        return real(brief, palette)
    monkeypatch.setattr(prompt_v2, "build_messages", build)
    meta = refused(monkeypatch, tree, tmp_path / "run", "PromptError in 3 cells (first high/melancholy: example")
    assert meta["prompt"] == {"version": "v2", "module": "lofi_rl.prompt_v2", "template_sha256": S.TEMPLATE_SHA256}


def test_every_failing_pre_check_is_named_at_once(monkeypatch, own_tree, tmp_path):
    (own_tree / S.DECISIONS).write_text("# no mark\n")
    (own_tree / "kit.env").write_text("changed\n")
    monkeypatch.setattr(S, "TEMPLATE_SHA256", "0" * 64)
    refused(monkeypatch, own_tree, tmp_path / "run", "prompt v2's template", "holds no 'D-023 mark:' line",
            "1 differ (first kit.env)")


# --------------------------------------------------------------------------------------------- the replay
def test_the_replay_redraws_the_held_out_requests_in_reverse_into_replay_jsonl_only(monkeypatch, tree, tmp_path,
                                                                                    subset):
    rc, meta, _, engine = run_generate(monkeypatch, tree, tmp_path / "run")
    assert rc == 0
    held = [r for r in subset if r.stream == "rm_heldout"]
    assert len(held) == 16
    replay_call = engine.calls[-1]
    assert [sp.seed for sp in replay_call[1]] == [r.draw_seed for r in reversed(held)], "reverse order, own seeds"
    rows = rows_of(tmp_path / "run" / "summary" / "replay.jsonl")
    assert [r["i"] for r in rows] == [r.i for r in reversed(held)] and all(r["same"] for r in rows)
    assert all(list(r) == list(S.FIELDS["replay.jsonl"]) for r in rows)
    drawn = rows_of(tmp_path / "run" / "summary" / "completions.jsonl")
    assert len(drawn) == 32 and Counter(r["i"] for r in drawn) == Counter(r.i for r in subset), "never in completions"
    assert meta["replay"] == {"n": 16, "same": 16, "differ": 0, "by_chunk": [{"start": 0, "count": 16, "same": 16}],
                              "stopped": None}
    assert meta["timing"]["replay_s"] is not None


def test_the_full_replay_is_two_chunks_of_160(monkeypatch, tree, tmp_path):
    rc, meta, _, engine = run_generate(monkeypatch, tree, tmp_path / "run")
    assert rc == 0 and [len(c) for c, _ in engine.calls[-2:]] == [160, 160]
    assert meta["replay"]["by_chunk"] == [{"start": 0, "count": 160, "same": 160},
                                          {"start": 160, "count": 160, "same": 160}]
    rows = rows_of(tmp_path / "run" / "summary" / "replay.jsonl")
    assert [r["i"] for r in rows] == list(range(319, -1, -1))


def test_replay_counts_texts_that_differ(monkeypatch, tree, tmp_path, subset):
    seen = Counter()

    def text_of(cell, key):
        seen[key] += 1
        return f"loop {key} take {seen[key] if key != S.PROBE_SEED else 1}\n", "stop", 10, S.PROMPT_TOKENS[cell]
    rc, meta, *_ = run_generate(monkeypatch, tree, tmp_path / "run", FakeEngine(text_of))
    assert rc == 0 and (meta["replay"]["n"], meta["replay"]["same"], meta["replay"]["differ"]) == (16, 0, 16)
    rows = rows_of(tmp_path / "run" / "summary" / "replay.jsonl")
    assert all(r["same"] is False and r["text"].endswith("take 2\n") for r in rows)


def test_a_replay_cut_by_the_deadline_or_an_error_leaves_the_draws_complete(monkeypatch, tree, tmp_path, subset):
    clock = [0.0]
    monkeypatch.setattr(G, "_now", lambda: clock[0])
    engine = FakeEngine(clock=clock, per_request_s=1.0)
    # the probe ends at 4 s and the one chunk of 32 at 36 s; the replay's 16 are forecast at 1.25 x 32 = 40 s: past 70
    rc, meta, *_ = run_generate(monkeypatch, tree, tmp_path / "run", engine, args=("--deadline-s", "70"))
    assert rc == 0 and meta["stopped"] is None and meta["generated"] == 32
    assert meta["replay"]["stopped"] == "deadline" and meta["replay"]["n"] == 0
    assert rows_of(tmp_path / "run" / "summary" / "replay.jsonl") == []
    rc, meta, *_ = run_generate(monkeypatch, tree, tmp_path / "run2", FakeEngine(fail_at=3))
    assert rc == 0 and meta["generated"] == 32 and meta["stopped"] is None
    assert meta["replay"]["stopped"] == "error: RuntimeError: the fake engine died" and "trace" in meta


# --------------------------------------------------------------------------------------------- the watchdog
def test_the_watchdog_dumps_stacks_at_the_deadline_plus_junks_margin(monkeypatch, tree, tmp_path, subset):
    calls = []
    monkeypatch.setattr(G.faulthandler, "enable", lambda **kw: calls.append(("enable", kw)))
    monkeypatch.setattr(G.faulthandler, "dump_traceback_later", lambda s, **kw: calls.append(("dump", s, kw)))
    monkeypatch.setattr(G.faulthandler, "cancel_dump_traceback_later", lambda: calls.append(("cancel",)))
    monkeypatch.setattr(G.signal, "alarm", lambda s: calls.append(("alarm", s)))
    monkeypatch.setattr(G, "ROOT", tree)
    monkeypatch.setattr(J, "load_engine", lambda model, **kw: FakeEngine())
    monkeypatch.setattr(J, "structured_outputs_params", lambda regex: {})
    assert cli.main(["generate", "--run", str(tmp_path / "run")]) == 0
    assert calls == [("enable", {"all_threads": True}), ("dump", 960.0 + J.HANG_MARGIN_S, {"exit": True}),
                     ("cancel",), ("alarm", 120)]
    calls.clear()
    assert cli.main(["generate", "--run", str(tmp_path / "run2"), "--hang-s", "0"]) == 0 and calls == []


def test_the_forecast_is_junks_for_the_draws_and_scaled_up_for_larger_chunks():
    timed = [{"s": 10.0, "count": 128}]
    assert G.too_late(0.0, 100.0, [], 128) is False and G.too_late(100.0, 100.0, [], 128) is True
    assert G.too_late(87.0, 100.0, timed, 128) is False and G.too_late(87.6, 100.0, timed, 128) is True
    assert G.too_late(87.6, 100.0, timed, 64) is True, "a smaller chunk is never forecast as faster (junk's rule)"
    assert G.too_late(84.0, 100.0, timed, 160) is False and G.too_late(84.5, 100.0, timed, 160) is True, (
        "160 requests at 10 s per 128: 12.5 s, x 1.25")
