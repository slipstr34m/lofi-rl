"""The base-model junk rate (lofi_rl.rl.junk; CLAUDE.md §6, the Phase 1-2 progress oracle).

The laptop has no vLLM, so `generate` runs here on a fake engine with the real prompt and brief sampler; the
real one runs in jobs/junk_rate.sbatch (tests/test_junk_job.py). `score` runs for real on a synthetic completions file whose every number
is worked out by hand below (gates, CI, repairs, P_fix, token percentiles, near-copies). `render` runs here with
the audio stubbed out; a `cluster`-marked test renders two loops for real (the node suites run it).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lofi_rl import grammar, prompt
from lofi_rl.brief import Brief, sample_briefs
from lofi_rl.compile.compiler import CompileError
from lofi_rl.palette import load_palette
from lofi_rl.rl import junk

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "palettes" / "dev_gu.yaml"

ORIG = """vamp: IVm9 bVII13 span 2
keys: organ shell
comp: x---..x-x---.... x-----..x-x-....
bass: lock
drums: jazz
kick: X.....x...x..... X.....x.x.....x.
snare: ....x.......x... ....x.......x.g.
hat: x.xxx.x.x.xxx.x. x.x.x.xxx.x.x.x.
shaker: ..x...x...x...x. ................
@2 kick: X.....x...x...x.
@4 hat: x.x.x.x.x.x.x.x.
@6 comp: x---..x-x---x-x-
@7 snare: ....x..g....x...
A: h1 q3 q5 | w7 | qr e6 e5 q4 q3 | h2 hr
B: h1 q3 q5 | w6 | qr e5 e4 q3 q2 | h.4 qr
"""
# an alias lane name (free) and a 14-cell snare grid (pad_grid, 0.02)
R2 = ORIG.replace("kick: X.....x...x..... X", "bd: X.....x...x..... X").replace(
    "snare: ....x.......x... ....", "snare: ....x.......x. ....")
# a second hat base (duplicate, 0.02) and a B: 4 steps short (pad_phrase, 0.02)
R3 = """vamp: Imaj9 IVmaj7 span 1
keys: rhodes spread
comp: x-------x---x---
bass: x.....x...x.....
drums: jazz
kick: x.........x.....
rim: ....x.......x...
hat: x.x.x.x.x.x.x.x.
hat: xxxxxxxxxxxxxxxx
@3 rim: ....x.......x..x
@5 kick: x.....x...x.....
@8 hat: x.x.x.x.x.x.xxx.
A: q1 q2 q3 q5 | h6 h5 | q3 q2 q1 qr | w3
B: q1 q2 q3 q5 | h6 h5 | q4 q3 q2 qr | h2 qr
"""


@pytest.fixture(scope="module")
def pal():
    return load_palette(DEV, verify=False)


def _cases(pal) -> list[tuple[str, str, str, str, int]]:
    """(text, finish_reason, energy, mood, n_tokens) of the 12 synthetic completions."""
    ex = prompt.example_text(pal)
    return [
        (ex, "stop", "mid", "wistful", 263),                                            # 0 pass, a copy
        ("Here's a loop:\n```\n" + ex + "```\nEnjoy!\n", "stop", "low", "wistful", 280),  # 1 pass, a copy
        (R2, "stop", "high", "melancholy", 300),                                       # 2 pass, P_fix 0.02
        (R3, "stop", "mid", "hopeful", 310),                                           # 3 pass, P_fix 0.04
        (ex.replace("vamp: Im9 IV13 span 2", "vamp: Dm7 G7 span 2"), "stop", "mid", "wistful", 270),  # 4 G5, a copy
        ("kick: four on the floor\n", "stop", "low", "hopeful", 10),                   # 5 G1
        ("vamp: Im9 IV13 span 2\nkeys: rhodes roo", "length", "high", "hopeful", 512),  # 6 G2
        (ORIG.replace("B: h1 q3 q5 | w6 | qr e5 e4 q3 q2 | h.4 qr\n", ""), "stop", "low", "melancholy", 200),  # 7 G3
        (ORIG.replace("drums: jazz", "drums: trap808"), "stop", "mid", "melancholy", 240),  # 8 G4
        (ORIG.replace("vamp: IVm9 bVII13 span 2", "vamp: IVm9"), "stop", "high", "wistful", 250),  # 9 G6
        ("", "stop", "low", "melancholy", 0),                                           # 10 G1
        ("I cannot write music, sorry.\n", "stop", "high", "hopeful", 8),              # 11 G1
    ]


def _write_run(run: Path, pal, cases=None, *, drift: int | None = 11) -> list[dict]:
    """A synthetic run directory: generate.json and completions.jsonl, as `generate` writes them."""
    cases = _cases(pal) if cases is None else cases
    summ = run / "summary"
    summ.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (text, fin, energy, mood, n_tok) in enumerate(cases):
        b = Brief(energy=energy, mood=mood, bpm=82, tonic=2)
        sha = prompt.messages_sha256(prompt.build_messages(b, pal))
        rows.append({"i": i, "brief": junk.brief_dict(b), "prompt_sha256": "0" * 64 if i == drift else sha,
                     "text": text, "finish_reason": fin, "n_tokens": n_tok, "prompt_tokens": 1388 + i % 3})
    with (summ / "completions.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    meta = {"schema": junk.GENERATE_SCHEMA, "partial": False, "requested": len(rows), "generated": len(rows),
            "stopped": None, "briefs": {"seed": 1, "count": len(rows)},
            "model": {"repo": "fake/model", "revision": "f" * 40},
            "prompt": {"version": prompt.PROMPT_VERSION, "template_sha256": prompt.template_sha256(),
                       "system_sha256": junk.sha256_text(prompt.system_message(pal))},
            "palette": {"id": pal.id, "sha256": pal.sha256}, "sampling": dict(junk.SAMPLING)}
    (summ / "generate.json").write_text(json.dumps(meta))
    return rows


def _percentile(xs, q):     # numpy's default ("linear"), written out
    xs = sorted(xs)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


# ------------------------------------------------------------------------------------------- statistics
def test_wilson_interval_matches_known_values():
    lo, hi = junk.wilson(4, 10)
    assert lo == pytest.approx(0.1682, abs=1e-4) and hi == pytest.approx(0.6873, abs=1e-4)
    assert junk.wilson(0, 10) == (0.0, pytest.approx(0.2775, abs=1e-4))
    assert junk.wilson(10, 10) == (pytest.approx(0.7225, abs=1e-4), 1.0)
    assert junk.wilson(0, 0) == (0.0, 1.0)
    lo, hi = junk.wilson(300, 500)                  # 60 %: the Phase 1 gate's bar
    assert lo == pytest.approx(0.5565, abs=1e-4) and hi == pytest.approx(0.6420, abs=1e-4)


def test_percentiles_interpolate_linearly_like_numpy():
    assert junk.percentiles([4, 1, 3, 2], (0, 25, 50, 100)) == {"p0": 1, "p25": 1.75, "p50": 2.5, "p100": 4}
    assert junk.percentiles([7], (50, 99)) == {"p50": 7, "p99": 7}
    assert junk.percentiles([], (50,)) == {"p50": None}


def test_character_5gram_jaccard():
    g = junk.grams
    assert junk.jaccard(g("abcdef"), g("abcdefg")) == pytest.approx(2 / 3)
    assert junk.jaccard(g("abcdef"), g("abcdef")) == 1.0
    assert junk.jaccard(g("abcdef"), g("uvwxyz")) == 0.0
    assert junk.jaccard(g(""), g("")) == 0.0, "two empty texts are not copies"
    assert g("abc") == frozenset({"abc"}), "a text shorter than n is one gram"


def test_plain_text_drops_fences_blank_lines_and_extra_spaces():
    assert junk.plain_text("```\n  kick:   x...  \n\n\tB: q1\n```\n") == "kick: x...\nB: q1\n"


def test_sample_indices_is_a_seeded_subset_independent_of_input_order():
    pick = junk.sample_indices(range(100), 50, seed=7)
    assert len(pick) == len(set(pick)) == 50 and set(pick) <= set(range(100))
    assert pick == junk.sample_indices(list(reversed(range(100))), 50, seed=7)
    assert pick != junk.sample_indices(range(100), 50, seed=8)
    assert pick != sorted(pick), "rank order, not index order"
    assert sorted(junk.sample_indices([3, 9, 4], 50, seed=7)) == [3, 4, 9]
    assert junk.sample_indices(range(100), 10, seed=7) == pick[:10], "a smaller sample is a prefix"


# ------------------------------------------------------------------------------------------- score
def test_score_summary_on_a_synthetic_completions_file(tmp_path, pal):
    _write_run(tmp_path, pal)
    assert junk.main(["score", "--run", str(tmp_path), "--palette", str(DEV)]) == 0
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    scored = [json.loads(l) for l in (tmp_path / "summary" / "scored.jsonl").read_text().splitlines()]
    assert [r["i"] for r in scored] == list(range(12))

    assert s["schema"] == junk.SCHEMA and s["n"] == 12 and s["requested"] == 12 and s["partial"] is False
    assert s["pass"]["count"] == 4 and s["pass"]["rate"] == pytest.approx(4 / 12, abs=1e-4)
    assert s["pass"]["ci95"] == pytest.approx(list(junk.wilson(4, 12)), abs=1e-4)
    assert s["failures"] == {"G1": 3, "G2": 1, "G3": 1, "G4": 1, "G5": 1, "G6": 1, "G7": None,
                             "compile": 0, "crash": 0}
    assert [r["gate"] for r in scored] == [None, None, None, None, "G5", "G1", "G2", "G3", "G4", "G6", "G1", "G1"]

    rep = s["repairs"]
    assert rep["by_code"] == {
        "alias_lane": {"loops": 1, "count": 1, "penalized": False},
        "default_span": {"loops": 1, "count": 1, "penalized": False},
        "duplicate": {"loops": 1, "count": 1, "penalized": True},
        "pad_grid": {"loops": 1, "count": 1, "penalized": True},
        "pad_phrase": {"loops": 1, "count": 1, "penalized": True},
        "strip_prose": {"loops": 2, "count": 2, "penalized": False}}
    assert rep["per_loop"] == {"all": {"0": 7, "1": 3, "2": 2}, "passing": {"0": 1, "1": 1, "2": 2}}
    assert rep["p_fix"]["passing"] == {"mean": pytest.approx(0.015), "hist": {"0.00": 2, "0.02": 1, "0.04": 1}}
    assert [r["p_fix"] for r in scored[:4]] == [0.0, 0.0, 0.02, 0.04]

    tok = s["tokens"]
    n_tok = [c[4] for c in _cases(pal)]
    for q in (50, 90, 95, 99):
        assert tok["completion"][f"p{q}"] == pytest.approx(_percentile(n_tok, q))
    assert tok["completion"]["max"] == 512 and tok["completion"]["mean"] == pytest.approx(sum(n_tok) / 12)
    assert tok["truncated"] == 1 and tok["over_360"] == 1 and tok["max_tokens"] == 512
    assert tok["prompt"] == {"min": 1388, "max": 1390}

    nc = s["near_copy"]
    assert nc["threshold"] == 0.8 and nc["ngram"] == 5
    assert nc["count"] == 3 and nc["rate"] == pytest.approx(3 / 12, abs=1e-4)
    assert nc["count_passing"] == 2 and nc["rate_of_passing"] == pytest.approx(0.5)
    assert [r["near_copy"] for r in scored] == [True, True, False, False, True] + [False] * 7
    assert scored[0]["copy_jaccard"] == 1.0 and scored[1]["copy_jaccard"] == 1.0, "fences and prose do not hide a copy"
    assert s["pass_not_copy"]["count"] == 2 and s["pass_not_copy"]["ci95"] == pytest.approx(list(junk.wilson(2, 12)), abs=1e-4)

    assert s["prompt"]["version"] == "v1" and s["prompt"]["template_sha256"] == prompt.template_sha256()
    assert s["prompt"]["system_sha256"] == junk.sha256_text(prompt.system_message(pal))
    assert s["prompt"]["rows_matching"] == 11, "row 11 was sent another prompt"
    assert s["palette"] == {"id": "dev_gu", "sha256": pal.sha256, "matches_generate": True}

    assert s["by_knob"]["energy"] == {"low": {"n": 4, "pass": 1}, "mid": {"n": 4, "pass": 2}, "high": {"n": 4, "pass": 1}}
    assert s["by_knob"]["mood"] == {"melancholy": {"n": 4, "pass": 1}, "wistful": {"n": 4, "pass": 2},
                                    "hopeful": {"n": 4, "pass": 1}}
    assert s["top_details"]["G1"][0] == ["no LoopScript statements", 2]
    assert s["top_details"]["G5"] == [["'Dm7' is a note name or absolute chord", 1]]


def test_score_counts_compile_errors_and_parser_crashes_without_stopping(tmp_path, pal, monkeypatch):
    ex = prompt.example_text(pal)
    _write_run(tmp_path, pal, [(ex, "stop", "mid", "wistful", 263), (ORIG, "stop", "low", "melancholy", 250),
                               ("boom", "stop", "mid", "hopeful", 5)], drift=None)
    real_compile, real_parse = junk.compile_loop, junk.parse

    def compile_loop(loop, brief, palette):
        if brief.mood == "melancholy":
            raise CompileError("the palette cannot play this")
        return real_compile(loop, brief, palette)

    def parse(text, **kw):
        if text == "boom":
            raise RuntimeError("parser bug")
        return real_parse(text, **kw)
    monkeypatch.setattr(junk, "compile_loop", compile_loop)
    monkeypatch.setattr(junk, "parse", parse)
    assert junk.main(["score", "--run", str(tmp_path), "--palette", str(DEV)]) == 0
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    assert s["pass"]["count"] == 1 and s["failures"]["compile"] == 1 and s["failures"]["crash"] == 1
    assert s["top_details"]["crash"] == [["RuntimeError: parser bug", 1]]


def test_score_skips_a_cut_last_line_and_reports_a_partial_run(tmp_path, pal):
    _write_run(tmp_path, pal, drift=None)
    comp = tmp_path / "summary" / "completions.jsonl"
    comp.write_text(comp.read_text() + '{"i": 12, "brief": {"ener')       # killed mid-write
    meta = json.loads((tmp_path / "summary" / "generate.json").read_text())
    meta.update(requested=20, partial=True, stopped="deadline")
    (tmp_path / "summary" / "generate.json").write_text(json.dumps(meta))
    assert junk.main(["score", "--run", str(tmp_path), "--palette", str(DEV)]) == 0
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    assert s["n"] == 12 and s["requested"] == 20 and s["partial"] is True and s["unreadable_lines"] == 1
    assert s["generate"]["stopped"] == "deadline"


def test_score_without_completions_fails_loudly(tmp_path):
    assert junk.main(["score", "--run", str(tmp_path), "--palette", str(DEV)]) == 2


# ------------------------------------------------------------------------------------------- generate (fake engine)
class FakeEngine:
    """The engine interface junk.generate uses: .info, .load_s, .sampling(**kw), .chat(convs, sp)."""

    def __init__(self, texts, clock=None, step_s=0.0):
        self.texts, self.calls, self.clock, self.step_s = texts, [], clock, step_s
        self.info, self.load_s = {"engine": "fake"}, 1.5

    def sampling(self, **kw):
        self.sp = dict(kw)
        return self.sp

    def chat(self, convs, sp):
        assert sp is self.sp
        self.calls.append(convs)
        if self.clock is not None:
            self.clock[0] += self.step_s
        base = sum(len(c) for c in self.calls[:-1])
        return [junk.Completion(text=self.texts[(base + k) % len(self.texts)], finish_reason="stop",
                                n_tokens=100 + base + k, prompt_tokens=1388) for k in range(len(convs))]


def test_generate_writes_one_row_per_brief_with_the_v1_prompt(tmp_path, pal, monkeypatch):
    engine = FakeEngine([prompt.example_text(pal), ORIG])
    seen = {}

    def load_engine(model, **kw):
        seen.update(model=model, **kw)
        return engine
    monkeypatch.setattr(junk, "load_engine", load_engine)
    rc = junk.main(["generate", "--run", str(tmp_path), "--n", "10", "--seed", "33", "--chunk", "4",
                    "--palette", str(DEV), "--hang-s", "0"])
    assert rc == 0
    assert seen["model"] == f"{junk.QWEN[0]}@{junk.QWEN[1]}" and seen["seed"] == 33
    assert engine.sp == {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
                         "presence_penalty": 0.0, "frequency_penalty": 0.0, "max_tokens": 512}
    assert [len(c) for c in engine.calls] == [4, 4, 2]
    briefs = sample_briefs(33, 10)
    sent = [conv for call in engine.calls for conv in call]
    assert sent == [prompt.build_messages(b, pal) for b in briefs]
    rows = [json.loads(l) for l in (tmp_path / "summary" / "completions.jsonl").read_text().splitlines()]
    assert [r["i"] for r in rows] == list(range(10))
    assert [Brief(**r["brief"]) for r in rows] == list(briefs)
    assert [r["prompt_sha256"] for r in rows] == [prompt.messages_sha256(c) for c in sent]
    assert rows[1]["text"] == ORIG and rows[9]["n_tokens"] == 109 and rows[0]["finish_reason"] == "stop"
    meta = json.loads((tmp_path / "summary" / "generate.json").read_text())
    assert meta["partial"] is False and meta["generated"] == meta["requested"] == 10 and meta["stopped"] is None
    assert meta["prompt"]["version"] == "v1" and meta["prompt"]["template_sha256"] == prompt.template_sha256()
    assert meta["briefs"] == {"seed": 33, "count": 10, "sampler": "lofi_rl.brief.sample_briefs"}
    assert meta["model"] == {"repo": junk.QWEN[0], "revision": junk.QWEN[1]}
    assert [c["count"] for c in meta["timing"]["chunks"]] == [4, 4, 2] and meta["timing"]["load_s"] == 1.5
    assert meta["constrained"] == UNCONSTRAINED, "v0's settings: no structured outputs unless --constrained"
    assert junk.main(["score", "--run", str(tmp_path), "--palette", str(DEV)]) == 0
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    assert s["n"] == 10 and s["prompt"]["rows_matching"] == 10 and s["near_copy"]["count"] == 5
    assert s["constrained"] == UNCONSTRAINED


UNCONSTRAINED = {"enabled": False, "regex_sha256": None, "regex_chars": None, "builder": None}


def test_generate_constrained_sends_the_loopscript_regex_and_records_its_sha(tmp_path, pal, monkeypatch):
    engine = FakeEngine([prompt.example_text(pal), ORIG])
    monkeypatch.setattr(junk, "load_engine", lambda model, **kw: engine)
    made = []

    def params(regex):
        made.append(regex)
        return {"StructuredOutputsParams": {"regex": regex}}
    monkeypatch.setattr(junk, "structured_outputs_params", params)
    rc = junk.main(["generate", "--run", str(tmp_path), "--n", "4", "--chunk", "4", "--palette", str(DEV),
                    "--hang-s", "0", "--constrained"])
    assert rc == 0
    rx = grammar.loopscript_regex(pal)
    assert made == [rx], "built once, from the run's palette"
    assert engine.sp == {**junk.SAMPLING, "structured_outputs": {"StructuredOutputsParams": {"regex": rx}}}
    meta = json.loads((tmp_path / "summary" / "generate.json").read_text())
    want = {"enabled": True, "regex_sha256": grammar.regex_sha256(pal), "regex_chars": len(rx),
            "builder": "lofi_rl.grammar.loopscript_regex"}
    assert meta["constrained"] == want
    assert meta["sampling"] == junk.SAMPLING, "the sampling settings are v0's; the regex is recorded by its sha"
    assert junk.main(["score", "--run", str(tmp_path), "--palette", str(DEV)]) == 0
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    assert s["constrained"] == want
    assert s["pass"]["count"] == 4 and s["failures"]["G1"] == 0, "the score step is the same"


def test_generate_constrained_fails_before_the_model_loads_naming_the_import(tmp_path, pal, monkeypatch):
    import types
    loads = []
    monkeypatch.setattr(junk, "load_engine", lambda model, **kw: loads.append(model) or FakeEngine([ORIG]))
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", None)          # a vLLM without StructuredOutputsParams
    rc = junk.main(["generate", "--run", str(tmp_path), "--n", "2", "--palette", str(DEV), "--hang-s", "0",
                    "--constrained"])
    assert rc == 1 and loads == [], "no GPU time is spent on a run that cannot constrain"
    meta = json.loads((tmp_path / "summary" / "generate.json").read_text())
    assert "from vllm.sampling_params import StructuredOutputsParams" in meta["stopped"]
    assert meta["generated"] == 0 and meta["constrained"]["regex_sha256"] == grammar.regex_sha256(pal)
    assert (tmp_path / "summary" / "completions.jsonl").read_text() == ""


def test_structured_outputs_params_is_vllms_regex_params_and_reaches_sampling_params(monkeypatch):
    import types
    seen = _fake_vllm(monkeypatch, refuse_top_k_zero=False)

    class StructuredOutputsParams:
        def __init__(self, **kw):
            self.kw = kw
    mod = types.ModuleType("vllm.sampling_params")
    mod.StructuredOutputsParams = StructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", mod)
    so = junk.structured_outputs_params("vamp: (Im9|IV13)")
    assert isinstance(so, StructuredOutputsParams) and so.kw == {"regex": "vamp: (Im9|IV13)"}
    eng = junk.VllmEngine("/hf/snap", seed=5, gpu_mem=0.85, max_model_len=4096)
    sp = eng.sampling(**junk.SAMPLING, structured_outputs=so)
    assert sp.kw == {**junk.SAMPLING, "n": 1, "structured_outputs": so}
    assert "init" in seen


def test_generate_stops_at_its_deadline_and_keeps_every_finished_chunk(tmp_path, pal, monkeypatch):
    clock = [0.0]
    engine = FakeEngine([ORIG], clock=clock, step_s=100.0)
    monkeypatch.setattr(junk, "load_engine", lambda model, **kw: engine)
    monkeypatch.setattr(junk, "_now", lambda: clock[0])
    rc = junk.main(["generate", "--run", str(tmp_path), "--n", "10", "--chunk", "4", "--deadline-s", "250",
                    "--palette", str(DEV), "--hang-s", "0"])
    assert rc == 0
    rows = (tmp_path / "summary" / "completions.jsonl").read_text().splitlines()
    assert len(rows) == 8, "two chunks of 4: a third would end past the deadline (200 + 1.25 x 100 > 250)"
    meta = json.loads((tmp_path / "summary" / "generate.json").read_text())
    assert meta["generated"] == 8 and meta["requested"] == 10 and meta["stopped"] == "deadline"


def test_generate_records_an_engine_failure_and_exits_nonzero(tmp_path, pal, monkeypatch):
    class Broken(FakeEngine):
        def chat(self, convs, sp):
            if self.calls:
                raise RuntimeError("engine core died")
            return super().chat(convs, sp)
    monkeypatch.setattr(junk, "load_engine", lambda model, **kw: Broken([ORIG]))
    rc = junk.main(["generate", "--run", str(tmp_path), "--n", "6", "--chunk", "3", "--palette", str(DEV),
                    "--hang-s", "0"])
    assert rc == 1
    meta = json.loads((tmp_path / "summary" / "generate.json").read_text())
    assert meta["generated"] == 3 and meta["stopped"].startswith("error: RuntimeError: engine core died")
    assert len((tmp_path / "summary" / "completions.jsonl").read_text().splitlines()) == 3


def _fake_vllm(monkeypatch, *, refuse_top_k_zero: bool) -> dict:
    """vllm, torch and transformers stand-ins with the attributes VllmEngine reads (the laptop has none of them)."""
    import types
    seen: dict = {}

    class SamplingParams:
        def __init__(self, **kw):
            if refuse_top_k_zero and kw.get("top_k") == 0:
                raise ValueError("top_k must be -1 (disable), or at least 1, got 0.")
            self.kw = kw

        def __repr__(self):
            return f"SamplingParams(top_k={self.kw['top_k']})"

    class Output:
        def __init__(self, text, reason, ids):
            self.text, self.finish_reason, self.token_ids = text, reason, ids

    class Request:
        def __init__(self, k):
            self.outputs = [Output(f"text {k}", "length" if k else "stop", list(range(3 + k)))]
            self.prompt_token_ids = list(range(1388))

    class LLM:
        def __init__(self, **kw):
            seen["init"] = kw

        def chat(self, messages, sampling_params=None, use_tqdm=True):
            seen["chat"] = (messages, sampling_params, use_tqdm)
            return [Request(k) for k in range(len(messages))]
    for name, attrs in (("vllm", {"LLM": LLM, "SamplingParams": SamplingParams, "__version__": "0.32.0"}),
                        ("torch", {"__version__": "2.13.0"}), ("transformers", {"__version__": "5.19.0"})):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(junk, "_check_toolchain", lambda: None)
    return seen


@pytest.mark.parametrize("refuse", [False, True], ids=["top_k 0", "older vLLM: top_k -1"])
def test_vllm_engine_runs_vllm_as_the_smoke_did_and_maps_its_outputs(monkeypatch, refuse):
    seen = _fake_vllm(monkeypatch, refuse_top_k_zero=refuse)
    eng = junk.VllmEngine("/hf/snap", seed=5, gpu_mem=0.85, max_model_len=4096)
    assert seen["init"] == {"model": "/hf/snap", "dtype": "bfloat16", "gpu_memory_utilization": 0.85,
                            "max_model_len": 4096, "enforce_eager": True, "seed": 5}
    sp = eng.sampling(**junk.SAMPLING)
    assert sp.kw == {**junk.SAMPLING, "n": 1, "top_k": -1 if refuse else 0}
    assert eng.info["sampling_params"] == repr(sp) and eng.info["vllm"] == "0.32.0" and "max_num_seqs" not in eng.info
    convs = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    out = eng.chat(convs, sp)
    assert seen["chat"] == (convs, sp, False), "batched, with the progress bar off"
    assert out == [junk.Completion("text 0", "stop", 3, 1388), junk.Completion("text 1", "length", 4, 1388)]


def test_load_engine_takes_the_pinned_snapshot_from_the_hf_cache(tmp_path, monkeypatch):
    repo, rev = junk.QWEN[0], junk.QWEN[1]
    snap = tmp_path / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
    snap.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setattr(junk, "VllmEngine", lambda path, **kw: (path, kw))
    with pytest.raises(FileNotFoundError, match="config.json"):       # smoke.QWEN's files must be there
        junk.load_engine(f"{repo}@{rev}", seed=1, gpu_mem=0.8, max_model_len=4096)
    for f in junk.QWEN[2]:
        (snap / f).write_text("{}")
    path, kw = junk.load_engine(f"{repo}@{rev}", seed=1, gpu_mem=0.8, max_model_len=4096)
    assert path == str(snap) and kw == {"seed": 1, "gpu_mem": 0.8, "max_model_len": 4096}


def test_model_spec_needs_a_pinned_commit():
    assert junk.split_model("org/name@" + "a" * 40) == ("org/name", "a" * 40)
    for bad in ("org/name", "org/name@main", "org/name@abc123"):
        with pytest.raises(ValueError):
            junk.split_model(bad)


# ------------------------------------------------------------------------------------------- render (stubbed audio)
def test_render_samples_passing_loops_and_merges_g7_into_the_summary(tmp_path, pal, monkeypatch):
    _write_run(tmp_path, pal, drift=None)
    assert junk.main(["score", "--run", str(tmp_path), "--palette", str(DEV)]) == 0
    calls = []

    def fake_clip(row, palette, clips_dir, *, cache, workers):
        calls.append(row["i"])
        if row["i"] == 3:
            return {"i": 3, "ok": False, "gate": "G7", "detail": "true peak -0.5 dBTP is above -0.9"}
        (clips_dir / f"{row['i']:04d}.flac").write_bytes(b"fLaC")
        return {"i": row["i"], "ok": True, "gate": None, "clip": f"clips/{row['i']:04d}.flac"}
    monkeypatch.setattr(junk, "render_clip", fake_clip)
    assert junk.main(["render", "--run", str(tmp_path), "--k", "3", "--palette", str(DEV), "--no-verify"]) == 0
    assert calls == junk.sample_indices([0, 1, 2, 3], 3, seed=1), "the brief seed from generate.json"
    clips = json.loads((tmp_path / "summary" / "clips.json").read_text())
    assert [c["i"] for c in clips["clips"]] == calls and clips["k"] == 3 and clips["seed"] == 1
    assert all("copy_jaccard" in c and "brief" in c for c in clips["clips"])
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    want_fail = int(3 in calls)
    assert s["failures"]["G7"] == want_fail
    assert s["render"] == {"attempted": 3, "rendered": 3 - want_fail, "G7": want_fail, "other_gates": 0,
                           "errors": 0, "skipped": 0, "seed": 1, "k": 3}
    assert not (tmp_path / "inbox").exists()


def test_render_reports_infrastructure_errors_and_deadline_skips(tmp_path, pal, monkeypatch):
    _write_run(tmp_path, pal, drift=None)
    assert junk.main(["score", "--run", str(tmp_path), "--palette", str(DEV)]) == 0
    clock = [0.0]

    def fake_clip(row, palette, clips_dir, *, cache, workers):
        clock[0] += 100.0
        return {"i": row["i"], "ok": False, "gate": None, "error": "FluidSynthError: no SoundFont"}
    monkeypatch.setattr(junk, "render_clip", fake_clip)
    monkeypatch.setattr(junk, "_now", lambda: clock[0])
    rc = junk.main(["render", "--run", str(tmp_path), "--k", "4", "--deadline-s", "150", "--palette", str(DEV),
                    "--no-verify"])
    assert rc == 1, "an infrastructure error is not a gate"
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    assert s["render"]["errors"] == 2 and s["render"]["skipped"] == 2 and s["failures"]["G7"] == 0


@pytest.mark.cluster
def test_render_writes_one_pass_flacs_for_real(tmp_path, pal):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("pedalboard")
    if not shutil.which("fluidsynth"):
        pytest.skip("no fluidsynth")
    ex = prompt.example_text(pal)
    _write_run(tmp_path, pal, [(ex, "stop", "mid", "wistful", 263), (ORIG, "stop", "low", "melancholy", 250)],
               drift=None)
    assert junk.main(["score", "--run", str(tmp_path), "--palette", str(DEV)]) == 0
    assert junk.main(["render", "--run", str(tmp_path), "--k", "2", "--palette", str(DEV), "--workers", "2"]) == 0
    clips = json.loads((tmp_path / "summary" / "clips.json").read_text())["clips"]
    assert len(clips) == 2 and all(c["ok"] for c in clips)
    for c in clips:
        x, sr = sf.read(str(tmp_path / c["clip"]), dtype="int16")
        assert sr == 44100 and x.shape == (c["n"], 2), "one pass of the loop: the rating page loops it"
        assert (tmp_path / c["clip"]).with_suffix(".manifest.json").is_file()
    assert json.loads((tmp_path / "summary" / "junk.json").read_text())["render"]["rendered"] == 2


# ------------------------------------------------------------------------------------------- the policy env
def test_score_runs_with_the_policy_envs_missing_modules_blocked(tmp_path, pal):
    """lofi_rl.{dsl, compile, theory, brief, palette, prompt, rl.junk} import, and `score` runs, with mido,
    soundfile, scipy and pedalboard blocked (requirements/policy.lock.txt has none of them) and without numpy
    (the policy env's numpy 2.3.5 is not the laptop's 1.26.4, so the score path does not touch it).
    lofi_rl.verify and lofi_rl.rl.reward are checked too once they exist."""
    _write_run(tmp_path, pal, drift=None)
    code = f"""
import importlib, importlib.util, sys
for m in ("mido", "soundfile", "scipy", "pedalboard", "numpy"):
    sys.modules[m] = None
for m in ("lofi_rl.dsl", "lofi_rl.compile", "lofi_rl.compile.compiler", "lofi_rl.theory", "lofi_rl.theory.voicing",
          "lofi_rl.brief", "lofi_rl.palette", "lofi_rl.prompt", "lofi_rl.rl", "lofi_rl.rl.junk"):
    importlib.import_module(m)
from lofi_rl.rl import junk
rc = junk.main(["score", "--run", {str(tmp_path)!r}, "--palette", {str(DEV)!r}])
del sys.modules["numpy"]
for m in ("lofi_rl.verify", "lofi_rl.rl.reward"):
    if importlib.util.find_spec(m) is not None:
        importlib.import_module(m)
print("RC", rc)
"""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0 and "RC 0" in out.stdout, out.stderr[-3000:]
    assert json.loads((tmp_path / "summary" / "junk.json").read_text())["pass"]["count"] == 4
