"""The base-model junk rate (lofi_rl.rl.junk; CLAUDE.md §6, the Phase 1-2 progress oracle).

The laptop has no vLLM, so `generate` runs here on a fake engine with the real prompt and brief sampler; the
real one runs in jobs/junk_rate.sbatch (tests/test_junk_job.py). `score` runs for real on a synthetic completions file whose every number
is worked out by hand below (gates, CI, repairs, P_fix, token percentiles, near-copies). `render` runs here with
the audio stubbed out; a `cluster`-marked test renders two loops for real (the node suites run it).

Prompt versions: generate takes --prompt v1 (the default) or v2 and records it; score reads the version from
generate.json and compares each row with the example its own prompt showed (a v2 synthetic run below), and the copy
readouts (copy.any, containment, verbatim lines, grid share, forms, knobs, by_cell) are checked on hand-built loops.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import types
from dataclasses import replace
from pathlib import Path

import pytest

from lofi_rl import grammar, prompt, prompt_v2
from lofi_rl.brief import ENERGIES, MOOD_NAMES, Brief, sample_briefs
from lofi_rl.compile.compiler import CompileError
from lofi_rl.palette import load_palette
from lofi_rl.rl import junk

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "palettes" / "dev_gu.yaml"
needs_audio = pytest.mark.skipif(
    shutil.which("fluidsynth") is None or any(importlib.util.find_spec(m) is None for m in ("pedalboard", "soundfile")),
    reason="renders for real: needs fluidsynth on PATH, pedalboard and soundfile")

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


def _write_run(run: Path, pal, cases=None, *, drift: int | None = 11, version: str = "v1") -> list[dict]:
    """A synthetic run directory: generate.json and completions.jsonl, as `generate` writes them. A v1 run's rows
    are v0's (no prompt_version or example field); a v2 run's rows carry both, as generate writes them now."""
    cases = _cases(pal) if cases is None else cases
    mod = {"v1": prompt, "v2": prompt_v2}[version]
    summ = run / "summary"
    summ.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (text, fin, energy, mood, n_tok) in enumerate(cases):
        b = Brief(energy=energy, mood=mood, bpm=82, tonic=2)
        sha = mod.messages_sha256(mod.build_messages(b, pal))
        row = {"i": i, "brief": junk.brief_dict(b), "prompt_sha256": "0" * 64 if i == drift else sha}
        if version != "v1":
            row.update(prompt_version=version, example=prompt_v2.example_id(b))
        rows.append({**row, "text": text, "finish_reason": fin, "n_tokens": n_tok, "prompt_tokens": 1388 + i % 3})
    with (summ / "completions.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    system = junk.sha256_text(prompt.system_message(pal)) if version == "v1" else None
    meta = {"schema": junk.GENERATE_SCHEMA, "partial": False, "requested": len(rows), "generated": len(rows),
            "stopped": None, "briefs": {"seed": 1, "count": len(rows)},
            "model": {"repo": "fake/model", "revision": "f" * 40},
            "prompt": {"version": mod.PROMPT_VERSION, "template_sha256": mod.template_sha256(), "system_sha256": system},
            "palette": {"id": pal.id, "sha256": pal.sha256}, "sampling": dict(junk.SAMPLING)}
    (summ / "generate.json").write_text(json.dumps(meta))
    return rows


def _score(run: Path) -> tuple[dict, list[dict]]:
    """junk.json and scored.jsonl after `score` on a run directory."""
    assert junk.main(["score", "--run", str(run), "--palette", str(DEV)]) == 0
    summ = run / "summary"
    return (json.loads((summ / "junk.json").read_text()),
            [json.loads(line) for line in (summ / "scored.jsonl").read_text().splitlines()])


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

    # the near-copy rule compares each row with the example its own prompt showed: v1's one example
    assert [r["copy_reference"] for r in scored] == ["example"] * 12
    assert nc["rule"] == "J5 >= 0.8 with the example shown in the row's own prompt"
    assert nc["references"] == {"example": 12}
    p = s["prompt"]
    assert p["module"] == "lofi_rl.prompt" and p["template_matches_generate"] is True
    assert p["systems"] == {f"{e}/{m}": p["system_sha256"] for e in ENERGIES for m in MOOD_NAMES}
    assert (p["example_recorded"], p["example_mismatch"]) == (0, 0), "v0's rows record no example"
    # readouts: copy.any (all 9 v2 examples) is near_copy here, as at step 0
    c = s["copy"]
    assert c["any"]["count"] == nc["count"] == 3 and c["any"]["count_passing"] == nc["count_passing"] == 2
    assert c["any"]["sources"] == {"example": 3} and c["any"]["refused"] == {}
    assert c["any"]["references"] == [ex_id for ex_id, _ in prompt_v2.EXAMPLES.values()]
    assert [r["near_copy_any"] for r in scored] == [r["near_copy"] for r in scored]
    assert [r["quasi_copy"] for r in scored] == [True, True, False, False, True, False, True] + [False] * 5, (
        "row 6 is the start of v1's example, cut by max_tokens: a quasi-copy, not a near-copy")
    assert scored[6]["containment"] == pytest.approx(34 / 35, abs=1e-4) and scored[6]["copy_jaccard"] < 0.8
    assert c["quasi_copy"]["count"] == 4 and c["quasi_copy"]["count_passing"] == 2
    assert c["pass_not_quasi_copy"] == {**junk._rate(2, 12)}
    assert c["containment_passing"] == {"p10": pytest.approx(0.4429, abs=1e-4), "p50": pytest.approx(0.7573, abs=1e-4),
                                        "p90": 1.0, "max": 1.0}
    assert c["jaccard_hist_passing"] == {"<0.5": 2, "0.5-0.6": 0, "0.6-0.7": 0, "0.7-0.75": 0, "0.75-0.8": 0,
                                         "0.8-0.9": 0, ">=0.9": 2}
    assert c["verbatim_passing"] == {"vamp": 2, "keys": 2, "A": 2, "B": 2, "AB": 2}
    assert scored[0]["grid_share"] == scored[1]["grid_share"] == [18, 18]
    k, n = (sum(scored[i]["grid_share"][j] for i in range(4)) for j in (0, 1))
    assert c["grid_share_passing"] == {"verbatim": k, "grids": n, "share": round(k / n, 4)} and n == 58
    assert c["forms_passing"] == {"span1": 1, "lock": 1, "snare": 1, "ohat": 2, "shaker": 1,
                                  "edits": {"3": 1, "4": 1, "6": 2}}
    assert c["knobs_passing"] == {"n": 4, "energy_match": 2, "f_mood": 4, "energy_bands": [0.38, 0.59]}
    assert c["knobs_error"] is None
    assert [r["energy_match"] for r in scored[:4]] == [True, False, False, True] and scored[3]["E"] == pytest.approx(0.4305)
    assert all(r[k] is None for r in scored[4:] for k in ("verbatim", "grid_share", "forms", "E", "energy_match", "f_mood"))
    assert list(s["by_cell"]) == [f"{e}/{m}" for e in ENERGIES for m in MOOD_NAMES]
    assert {cell["example"] for cell in s["by_cell"].values()} == {"example"}
    assert s["by_cell"]["mid/wistful"] == {"example": "example", "n": 2, "near_copy": 2, "quasi_copy": 2, "pass": 1,
                                           "near_copy_passing": 1, "quasi_copy_passing": 1, "pass_not_copy": 0,
                                           "pass_not_quasi_copy": 0, "energy_match": 1, "f_mood": 1}
    assert junk.headline(s).endswith(" | quasi-copies 4 (pass, not a quasi-copy: 2) | prompt v1")

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


def _v2_cases(pal) -> list[tuple[str, str, str, str, int]]:
    """(text, finish_reason, energy, mood, n_tokens) of a synthetic prompt-v2 run."""
    refs = prompt_v2.reference_texts(pal)
    return [
        (refs["l4_wistful_low"], "stop", "low", "melancholy", 280),     # 0 shown l4, writes l4: a copy
        (prompt.example_text(pal), "stop", "high", "hopeful", 263),    # 1 shown l3, writes v1's example: no copy
        (ORIG, "stop", "mid", "wistful", 300),                          # 2 shown l8: neither
        (prompt.example_text(pal), "stop", "mid", "melancholy", 263),  # 3 the control cell: shown v1's, a copy
        ("kick: four on the floor\n", "stop", "low", "hopeful", 10),   # 4 G1
    ]


def test_score_v2_compares_each_row_with_the_example_its_own_prompt_showed(tmp_path, pal):
    _write_run(tmp_path, pal, _v2_cases(pal), drift=None, version="v2")
    s, scored = _score(tmp_path)
    assert [r["copy_reference"] for r in scored] == ["l4_wistful_low", "l3_melancholy_high", "l8_hopeful_mid",
                                                    "example", "l1_melancholy_low"]
    assert [r["ok"] for r in scored] == [True, True, True, True, False]
    assert [r["near_copy"] for r in scored] == [True, False, False, True, False]
    assert [r["quasi_copy"] for r in scored] == [True, False, False, True, False]
    assert scored[0]["copy_jaccard"] == 1.0 and scored[0]["containment"] == 1.0
    assert scored[1]["copy_jaccard"] < 0.5 and scored[1]["containment"] < 0.85, "v1's example is not l3"
    # copy.any compares with all 9: v1's example in a high/hopeful row is counted there, with its source
    assert [r["copy_any_source"] for r in scored[:4]] == ["l4_wistful_low", "example", "l8_hopeful_mid", "example"]
    assert [r["near_copy_any"] for r in scored] == [True, True, False, True, False]
    c = s["copy"]
    assert (c["any"]["count"], c["any"]["count_passing"]) == (3, 3)
    assert c["any"]["sources"] == {"example": 2, "l4_wistful_low": 1}
    nc = s["near_copy"]
    assert (nc["count"], nc["count_passing"]) == (2, 2)
    assert nc["references"] == {"example": 1, "l1_melancholy_low": 1, "l3_melancholy_high": 1, "l4_wistful_low": 1,
                                "l8_hopeful_mid": 1}
    assert s["pass"]["count"] == 4
    assert s["pass_not_copy"] == junk._rate(2, 5), "rows 1 and 2"
    assert c["pass_not_quasi_copy"] == junk._rate(2, 5)
    assert c["quasi_copy"]["count"] == 2 and c["quasi_copy"]["count_passing"] == 2
    assert scored[0]["verbatim"] == {"vamp": True, "keys": True, "A": True, "B": True}
    assert scored[1]["verbatim"] == {"vamp": False, "keys": True, "A": False, "B": False}, "keys: rhodes rootless"
    assert scored[0]["grid_share"] == [14, 14] and scored[1]["grid_share"] == [0, 18]
    assert scored[0]["forms"] == {"span1": False, "lock": True, "snare": False, "ohat": False, "shaker": False,
                                  "edits": 6}
    assert (scored[0]["energy_match"], scored[0]["f_mood"]) == (True, False), "l4: low energy, wistful chords"
    cells = s["by_cell"]
    assert [cells[k]["example"] for k in cells] == ["l4_wistful_low", "l7_hopeful_low", "l1_melancholy_low", "example",
                                                    "l8_hopeful_mid", "l2_melancholy_mid", "l6_wistful_high",
                                                    "l9_hopeful_high", "l3_melancholy_high"]
    zero = {"n": 0, "near_copy": 0, "quasi_copy": 0, "pass": 0, "near_copy_passing": 0, "quasi_copy_passing": 0,
            "pass_not_copy": 0, "pass_not_quasi_copy": 0, "energy_match": 0, "f_mood": 0}
    assert cells["low/wistful"] == {"example": "l7_hopeful_low", **zero}
    assert cells["low/melancholy"] == {"example": "l4_wistful_low", "n": 1, "near_copy": 1, "quasi_copy": 1, "pass": 1,
                                       "near_copy_passing": 1, "quasi_copy_passing": 1, "pass_not_copy": 0,
                                       "pass_not_quasi_copy": 0, "energy_match": 1, "f_mood": 0}
    assert cells["high/hopeful"] == {"example": "l3_melancholy_high", "n": 1, "near_copy": 0, "quasi_copy": 0,
                                     "pass": 1, "near_copy_passing": 0, "quasi_copy_passing": 0, "pass_not_copy": 1,
                                     "pass_not_quasi_copy": 1, "energy_match": 0, "f_mood": 0}
    assert cells["mid/wistful"]["pass_not_quasi_copy"] == 1 and cells["mid/melancholy"]["near_copy"] == 1
    assert cells["low/hopeful"]["n"] == 1 and cells["low/hopeful"]["pass"] == 0
    assert sum(cell["n"] for cell in cells.values()) == 5
    p = s["prompt"]
    assert (p["version"], p["module"]) == ("v2", "lofi_rl.prompt_v2")
    assert p["template_sha256"] == prompt_v2.template_sha256() and p["template_matches_generate"] is True
    assert p["system_sha256"] is None and len(set(p["systems"].values())) == 9
    assert p["systems"]["mid/melancholy"] == junk.sha256_text(prompt.system_message(pal)), "the control cell"
    assert (p["rows_matching"], p["example_recorded"], p["example_mismatch"]) == (5, 5, 0)
    assert junk.headline(s).endswith(" | quasi-copies 2 (pass, not a quasi-copy: 2) | prompt v2")


def test_score_reads_the_prompt_version_from_generate_json(tmp_path, pal, capsys):
    gen = tmp_path / "summary" / "generate.json"
    _write_run(tmp_path, pal, drift=None)
    meta = json.loads(gen.read_text())
    del meta["prompt"]["version"]
    gen.write_text(json.dumps(meta))
    s, _ = _score(tmp_path)
    assert s["prompt"]["version"] == "v1" and s["near_copy"]["count"] == 3, "no version: v1, as v0's record reads"
    del meta["prompt"]
    gen.write_text(json.dumps(meta))
    s, _ = _score(tmp_path)
    assert s["prompt"]["version"] == "v1" and s["prompt"]["template_matches_generate"] is None
    meta["prompt"] = {"version": "v9"}
    gen.write_text(json.dumps(meta))
    (tmp_path / "summary" / "junk.json").unlink()
    assert junk.main(["score", "--run", str(tmp_path), "--palette", str(DEV)]) == 2
    assert "'v9'" in capsys.readouterr().err
    assert not (tmp_path / "summary" / "junk.json").exists(), "nothing is scored under a prompt it does not know"


def test_score_counts_rows_whose_recorded_example_is_not_the_rules(tmp_path, pal):
    _write_run(tmp_path, pal, _v2_cases(pal), drift=None, version="v2")
    comp = tmp_path / "summary" / "completions.jsonl"
    rows = [json.loads(line) for line in comp.read_text().splitlines()]
    rows[2]["example"] = "l5_wistful_mid"
    comp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    s, scored = _score(tmp_path)
    assert (s["prompt"]["example_recorded"], s["prompt"]["example_mismatch"]) == (5, 1)
    assert scored[2]["copy_reference"] == "l8_hopeful_mid", "the rule decides what a row is compared with"


def test_rows_matching_rebuilds_the_messages_with_the_runs_version(tmp_path, pal):
    _write_run(tmp_path, pal, _v2_cases(pal), drift=None, version="v2")
    s, _ = _score(tmp_path)
    assert s["prompt"]["rows_matching"] == 5
    gen = tmp_path / "summary" / "generate.json"
    meta = json.loads(gen.read_text())
    meta["prompt"]["version"] = "v1"                 # the same rows, read as a v1 run
    gen.write_text(json.dumps(meta))
    s, scored = _score(tmp_path)
    assert s["prompt"]["rows_matching"] == 1, "only the control cell's v2 messages are v1's"
    assert s["prompt"]["template_matches_generate"] is False, "the recorded sha is v2's"
    assert [r["copy_reference"] for r in scored] == ["example"] * 5


def test_copy_readouts_on_hand_built_loops():
    g = junk.grams
    assert junk.containment(g("abcdefg"), g("xabcdefx")) == pytest.approx(2 / 3), "abcde and bcdef of 3 grams"
    assert junk.containment(g(""), g("abcdef")) == 0.0 and junk.containment(g("abcdef"), frozenset()) == 0.0
    assert junk.best_match(g("abcdef"), ()) == (0.0, None)
    ref = junk.Reference.of("ref", "vamp: Im9 IV13 span 1\nkeys: rhodes rootless\ncomp: x-------x------- x---x-----------\n"
                                   "bass: lock\ndrums: jazz\nkick: x.......x.......\nrim: ....x.......x...\n"
                                   "hat: x.x.x.x.x.x.x.x.\n@3 hat: xxxxxxxxxxxxxxxx\n@5 comp: x---------------\n"
                                   "A: w1 | w2 | w3 | w4\nB: w1 | w2 | w3 | h5 hr\n")
    loop = ("vamp: Im9 IV13 span 2\nkeys: rhodes rootless\n"
            "comp: x--------------- x---x-----------\n"     # 2: the ref's @5 comp edit grid and its base comp grid
            "bass: x.......x.......\n"                       # 0: the ref's bass is lock (no grid)
            "drums: jazz\nkick: x.......x.......\n"           # 1
            "snare: ....x.......x...\n"                      # 0: no snare lane in the ref
            "hat: xxxxxxxxxxxxxxxx\n"                         # 1: the ref's @3 hat edit grid
            "ohat: ..............x.\n"                       # 0
            "@2 hat: x.x.x.x.x.x.x.x.\n"                     # 1: the ref's base hat grid
            "@4 bass: x---------------\n"                    # 0: the ref's @5 comp grid, but another lane
            "A: w1 | w2 | w3 | w4\nB: w1 | w2 | w3 | w5\n")
    assert junk.grid_share(loop, [ref]) == (5, 9)
    assert junk.grid_share(ref.text, [ref]) == (7, 7), "bass: lock holds no grid"
    assert junk.verbatim(loop, [ref]) == {"vamp": False, "keys": True, "A": True, "B": False}
    assert junk.verbatim(ref.text, [ref]) == dict.fromkeys(("vamp", "keys", "A", "B"), True)
    assert junk.forms(loop) == {"span1": False, "lock": False, "snare": True, "ohat": True, "shaker": False, "edits": 2}
    assert junk.forms(ref.text) == {"span1": True, "lock": True, "snare": False, "ohat": False, "shaker": False,
                                    "edits": 2}
    assert dict(ref.grids) == {"comp": frozenset({"x-------x-------", "x---x-----------", "x---------------"}),
                               "kick": frozenset({"x.......x......."}), "rim": frozenset({"....x.......x..."}),
                               "hat": frozenset({"x.x.x.x.x.x.x.x.", "xxxxxxxxxxxxxxxx"})}
    bins = {0.0: "<0.5", 0.4999: "<0.5", 0.5: "0.5-0.6", 0.6: "0.6-0.7", 0.7: "0.7-0.75", 0.7499: "0.7-0.75",
            0.75: "0.75-0.8", 0.7999: "0.75-0.8", 0.8: "0.8-0.9", 0.8999: "0.8-0.9", 0.9: ">=0.9", 1.0: ">=0.9"}
    assert {j: junk.jaccard_bin(j) for j in bins} == bins, "left-closed bins"


def test_a_knob_readout_that_raises_is_null_and_never_stops_score(tmp_path, pal, monkeypatch, capsys):
    _write_run(tmp_path, pal, drift=None)

    def broken(*args, **kw):
        raise RuntimeError("verifier exploded")
    monkeypatch.setattr(junk, "knob_readout", broken)
    s, scored = _score(tmp_path)
    assert all(r[k] is None for r in scored for k in ("E", "energy_match", "f_mood"))
    assert scored[0]["verbatim"] is not None, "the other readouts stand"
    c = s["copy"]
    assert c["knobs_passing"] is None
    assert c["knobs_error"] == {"loops": 4, "first": "RuntimeError: verifier exploded"}
    assert s["by_cell"]["mid/wistful"]["energy_match"] is None and s["by_cell"]["mid/wistful"]["f_mood"] is None
    assert s["by_cell"]["high/hopeful"]["energy_match"] == 0, "a cell with no passing loop counts 0"
    assert s["pass"]["count"] == 4 and s["near_copy"]["count"] == 3, "the score itself is unchanged"
    assert "knob readout failed on 4 loop(s)" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------- generate (fake engine)
class FakeEngine:
    """The engine interface junk.generate uses: .info, .load_s, .sampling(**kw), .chat(convs, sp)."""

    def __init__(self, texts, clock=None, step_s=0.0, info=None):
        self.texts, self.calls, self.clock, self.step_s = texts, [], clock, step_s
        self.info, self.load_s = dict(info or {"engine": "fake"}), 1.5

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
    assert all(r["prompt_version"] == "v1" and r["example"] == "example" for r in rows)
    meta = json.loads((tmp_path / "summary" / "generate.json").read_text())
    assert meta["partial"] is False and meta["generated"] == meta["requested"] == 10 and meta["stopped"] is None
    assert meta["prompt"]["version"] == "v1" and meta["prompt"]["template_sha256"] == prompt.template_sha256()
    longest = max((c[0]["content"] + "\n" + c[1]["content"] for c in sent), key=len)
    system = junk.sha256_text(prompt.system_message(pal))
    cells = {f"{e}/{m}": {"example": "example", "example_cell": "mid/wistful",
                          "example_sha256": junk.sha256_text(prompt.example_text(pal)), "system_sha256": system,
                          "messages_sha256": prompt.messages_sha256(prompt.build_messages(junk.cell_brief(e, m), pal))}
             for e in ENERGIES for m in MOOD_NAMES}
    assert meta["prompt"] == {"version": "v1", "module": "lofi_rl.prompt", "template_sha256": V1_TEMPLATE_SHA256,
                              "system_sha256": system, "cells": cells, "chars": len(longest),
                              "tokens_estimate_4cpt": prompt.estimate_tokens(longest)}, "v1's record, plus module and cells"
    assert meta["briefs"] == {"seed": 33, "count": 10, "sampler": "lofi_rl.brief.sample_briefs"}
    assert meta["model"] == {"repo": junk.QWEN[0], "revision": junk.QWEN[1]}
    assert [c["count"] for c in meta["timing"]["chunks"]] == [4, 4, 2] and meta["timing"]["load_s"] == 1.5
    assert meta["constrained"] == UNCONSTRAINED, "v0's settings: no structured outputs unless --constrained"
    assert seen["backend"] is None, "no structured-outputs backend is pinned for an unconstrained run"
    assert junk.main(["score", "--run", str(tmp_path), "--palette", str(DEV)]) == 0
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    assert s["n"] == 10 and s["prompt"]["rows_matching"] == 10 and s["near_copy"]["count"] == 5
    assert s["constrained"] == {**UNCONSTRAINED, "in_language": None}


V1_TEMPLATE_SHA256 = "7aceb2bc0d7339a7d3c9af7bf0d808ee156d4f7a8684e0393d923940fa5c1e68"     # tests/test_prompt.py's pin
# brief cell -> the example prompt v2 shows it (tests/test_prompt_v2.py's table)
V2_TABLE = {"low/melancholy": "l4_wistful_low", "low/wistful": "l7_hopeful_low", "low/hopeful": "l1_melancholy_low",
            "mid/melancholy": "example", "mid/wistful": "l8_hopeful_mid", "mid/hopeful": "l2_melancholy_mid",
            "high/melancholy": "l6_wistful_high", "high/wistful": "l9_hopeful_high",
            "high/hopeful": "l3_melancholy_high"}


def test_generate_prompt_v2_sends_v2s_messages_and_records_its_nine_cells(tmp_path, pal, monkeypatch):
    engine = FakeEngine([prompt.example_text(pal), ORIG])
    monkeypatch.setattr(junk, "load_engine", lambda model, **kw: engine)
    rc = junk.main(["generate", "--run", str(tmp_path), "--n", "30", "--seed", "7", "--chunk", "16",
                    "--palette", str(DEV), "--hang-s", "0", "--prompt", "v2"])
    assert rc == 0
    briefs = sample_briefs(7, 30)
    sent = [conv for call in engine.calls for conv in call]
    assert sent == [prompt_v2.build_messages(b, pal) for b in briefs]
    assert any(c != prompt.build_messages(b, pal) for c, b in zip(sent, briefs)), "not v1's messages"
    rows = [json.loads(l) for l in (tmp_path / "summary" / "completions.jsonl").read_text().splitlines()]
    assert [r["prompt_sha256"] for r in rows] == [prompt_v2.messages_sha256(c) for c in sent]
    assert [r["example"] for r in rows] == [V2_TABLE[f"{b.energy}/{b.mood}"] for b in briefs]
    assert all(r["prompt_version"] == "v2" for r in rows)
    meta = json.loads((tmp_path / "summary" / "generate.json").read_text())
    rec = meta["prompt"]
    longest = max((c[0]["content"] + "\n" + c[1]["content"] for c in sent), key=len)
    assert {k: v for k, v in rec.items() if k != "cells"} == {
        "version": "v2", "module": "lofi_rl.prompt_v2", "template_sha256": prompt_v2.template_sha256(),
        "system_sha256": None, "chars": len(longest), "tokens_estimate_4cpt": prompt.estimate_tokens(longest)}
    assert list(rec["cells"]) == list(V2_TABLE)
    for cell, want in V2_TABLE.items():
        b = junk.cell_brief(*cell.split("/"))
        e, m = prompt_v2.example_cell(b)
        assert rec["cells"][cell] == {
            "example": want, "example_cell": f"{e}/{m}",
            "example_sha256": junk.sha256_text(prompt_v2.example_text(pal, b)),
            "system_sha256": junk.sha256_text(prompt_v2.system_message(pal, b)),
            "messages_sha256": prompt_v2.messages_sha256(prompt_v2.build_messages(b, pal))}, cell
    assert rec["cells"]["mid/melancholy"]["system_sha256"] == junk.sha256_text(prompt.system_message(pal))
    assert len({c["system_sha256"] for c in rec["cells"].values()}) == 9
    s, scored = _score(tmp_path)
    assert s["prompt"]["version"] == "v2" and s["prompt"]["rows_matching"] == 30
    assert (s["prompt"]["example_recorded"], s["prompt"]["example_mismatch"]) == (30, 0)
    assert [r["copy_reference"] for r in scored] == [r["example"] for r in rows]
    assert s["near_copy"]["count"] == 1 == sum(1 for r in rows if r["example"] == "example" and r["i"] % 2 == 0), (
        "v1's example (every even row) is a near-copy only where it was shown: the mid/melancholy cell (row 18)")
    assert s["copy"]["any"]["count"] == 15 and s["copy"]["any"]["sources"] == {"example": 15}, (
        "every even row is a copy of v1's example, which only copy.any sees outside its cell")


def test_generate_refuses_an_unknown_prompt_version(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(junk, "load_engine", lambda model, **kw: pytest.fail("no engine for a refused argument"))
    with pytest.raises(SystemExit) as stop:
        junk.main(["generate", "--run", str(tmp_path), "--n", "2", "--prompt", "v3", "--hang-s", "0"])
    assert stop.value.code == 2 and "invalid choice: 'v3'" in capsys.readouterr().err
    assert not (tmp_path / "summary").exists()


def test_generate_v2_stops_before_the_engine_on_a_palette_that_cannot_play_an_example(tmp_path, pal, monkeypatch):
    no_maj9 = replace(pal, qualities=tuple(q for q in pal.qualities if q != "maj9"), sha256="3" * 64)
    monkeypatch.setattr(junk, "load_palette", lambda path, verify=False: no_maj9)
    loads = []
    monkeypatch.setattr(junk, "load_engine", lambda model, **kw: loads.append(model) or FakeEngine([ORIG]))
    rc = junk.main(["generate", "--run", str(tmp_path), "--n", "4", "--palette", str(DEV), "--hang-s", "0",
                    "--prompt", "v2"])
    assert rc != 0 and loads == [], "no GPU time is spent on a prompt the palette cannot serve"
    assert not (tmp_path / "summary" / "completions.jsonl").exists()
    meta = json.loads((tmp_path / "summary" / "generate.json").read_text())
    assert meta["generated"] == 0 and meta["stopped"].startswith("error: PromptError: ")
    assert "l7_hopeful_low" in meta["stopped"] and "maj9" in meta["stopped"]
    assert meta["prompt"] == {"version": "v2", "module": "lofi_rl.prompt_v2",
                              "template_sha256": prompt_v2.template_sha256()}
    monkeypatch.setattr(junk, "load_engine", lambda model, **kw: FakeEngine([ORIG]))
    assert junk.main(["generate", "--run", str(tmp_path / "v1"), "--n", "4", "--palette", str(DEV),
                      "--hang-s", "0"]) == 0, "v1's example needs no maj9"


UNCONSTRAINED = {"enabled": False, "regex_sha256": None, "regex_chars": None, "builder": None, "backend": None}
# what VllmEngine.info["structured_outputs"] holds when vLLM takes the pin (test_vllm_engine_pins_the_xgrammar_backend)
PINNED = {"requested": "xgrammar", "pin_accepted": True, "backend": "xgrammar",
          "source": "llm_engine.vllm_config.structured_outputs_config.backend", "applied": True}


def test_generate_constrained_sends_the_loopscript_regex_and_records_its_sha(tmp_path, pal, monkeypatch):
    engine = FakeEngine([prompt.example_text(pal), ORIG], info={"engine": "fake", "structured_outputs": PINNED})
    asked = []
    monkeypatch.setattr(junk, "load_engine", lambda model, **kw: asked.append(kw["backend"]) or engine)
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
    assert asked == ["xgrammar"], "the backend is pinned, so a regex xgrammar refuses fails the run"
    assert engine.sp == {**junk.SAMPLING, "structured_outputs": {"StructuredOutputsParams": {"regex": rx}}}
    meta = json.loads((tmp_path / "summary" / "generate.json").read_text())
    want = {"enabled": True, "regex_sha256": grammar.regex_sha256(pal), "regex_chars": len(rx),
            "builder": "lofi_rl.grammar.loopscript_regex", "backend": PINNED}
    assert meta["constrained"] == want
    assert meta["sampling"] == junk.SAMPLING, "the sampling settings are v0's; the regex is recorded by its sha"
    assert junk.main(["score", "--run", str(tmp_path), "--palette", str(DEV)]) == 0
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    full = re.compile(rx).fullmatch
    assert full(prompt.example_text(pal)) and full(ORIG), "both fake completions are in the language"
    assert s["constrained"] == {**want, "in_language": {
        "regex_sha256": grammar.regex_sha256(pal), "matches_generate": True,
        "stop": {"n": 4, "in_language": 4, "out_of_language": 0},
        "length": {"n": 0, "in_language": 0, "out_of_language": 0},
        "other": {"n": 0, "in_language": 0, "out_of_language": 0}}}
    assert s["pass"]["count"] == 4 and s["failures"]["G1"] == 0, "the score step is the same"


def _v0_like(pal) -> dict[str, str]:
    """Completions as v0 wrote them (runs/junk-13632676): its line ends ("  \\n", none after B) on the example, and
    the same with v0's commonest faults, which a regex-constrained engine cannot write."""
    ex = prompt.example_text(pal)
    v0 = ex[:-1].replace("\n", "  \n")
    return {"in": v0,
            "two grids on an edit": v0.replace("@4 kick: x......x..x...x.", "@4 kick: x......x..x...x. x......x..x...x."),
            "a 15-cell edit": v0.replace("@7 ohat: ......x.........", "@7 ohat: ......x........"),
            "a sax line": v0.replace("  \nA: ", "  \nsax: tenor  \nA: "),
            "cut": v0[:200]}


class Ignoring(FakeEngine):
    """An engine that takes the regex and ignores it: it returns what v0's unconstrained model wrote."""

    def __init__(self, texts):
        super().__init__([t for _, t in texts], info={"engine": "fake", "structured_outputs": PINNED})
        self.reasons = [r for r, _ in texts]

    def chat(self, convs, sp):
        out = super().chat(convs, sp)
        return [replace(c, finish_reason=self.reasons[k % len(self.reasons)]) for k, c in enumerate(out)]


def test_score_counts_constrained_completions_in_and_out_of_the_regexs_language(tmp_path, pal, monkeypatch):
    v = _v0_like(pal)
    engine = Ignoring([("stop", v["in"]), ("stop", v["two grids on an edit"]), ("stop", v["a 15-cell edit"]),
                       ("stop", v["a sax line"]), ("length", v["cut"]), (None, v["in"])])
    monkeypatch.setattr(junk, "load_engine", lambda model, **kw: engine)
    monkeypatch.setattr(junk, "structured_outputs_params", lambda regex: {"regex_sha256": junk.sha256_text(regex)})
    assert junk.main(["generate", "--run", str(tmp_path), "--n", "6", "--palette", str(DEV), "--hang-s", "0",
                      "--constrained"]) == 0
    assert junk.main(["score", "--run", str(tmp_path), "--palette", str(DEV)]) == 0
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    lang = s["constrained"]["in_language"]
    assert lang["matches_generate"] is True and lang["regex_sha256"] == grammar.regex_sha256(pal)
    assert lang["stop"] == {"n": 4, "in_language": 1, "out_of_language": 3}, "the engine did not apply the regex"
    assert lang["length"] == {"n": 1, "in_language": 0, "out_of_language": 1}, "cut completions are counted apart"
    assert lang["other"] == {"n": 1, "in_language": 1, "out_of_language": 0}
    assert s["constrained"]["backend"] == PINNED
    assert "in the regex's language: stop 1/4, length 0/1" in junk.headline(s)
    assert s["failures"]["G1"] == 1 and s["failures"]["G2"] == 1, "the gates are scored as before"


def test_in_language_is_not_counted_against_a_regex_generate_did_not_record(tmp_path, pal, monkeypatch):
    engine = FakeEngine([prompt.example_text(pal)], info={"engine": "fake", "structured_outputs": PINNED})
    monkeypatch.setattr(junk, "load_engine", lambda model, **kw: engine)
    monkeypatch.setattr(junk, "structured_outputs_params", lambda regex: {})
    assert junk.main(["generate", "--run", str(tmp_path), "--n", "2", "--palette", str(DEV), "--hang-s", "0",
                      "--constrained"]) == 0
    gen = tmp_path / "summary" / "generate.json"
    meta = json.loads(gen.read_text())
    meta["constrained"]["regex_sha256"] = "0" * 64          # a regex grammar.py no longer builds
    gen.write_text(json.dumps(meta))
    assert junk.main(["score", "--run", str(tmp_path), "--palette", str(DEV)]) == 0
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    assert s["constrained"]["in_language"] == {"regex_sha256": grammar.regex_sha256(pal), "matches_generate": False,
                                               "stop": None, "length": None, "other": None}
    assert "not counted" in junk.headline(s)


def test_an_engine_that_reports_no_backend_is_recorded_as_unknown(tmp_path, pal, monkeypatch):
    monkeypatch.setattr(junk, "load_engine", lambda model, **kw: FakeEngine([ORIG]))
    monkeypatch.setattr(junk, "structured_outputs_params", lambda regex: {})
    assert junk.main(["generate", "--run", str(tmp_path), "--n", "1", "--palette", str(DEV), "--hang-s", "0",
                      "--constrained"]) == 0
    meta = json.loads((tmp_path / "summary" / "generate.json").read_text())
    assert meta["constrained"]["backend"] == {"requested": "xgrammar", "pin_accepted": None, "backend": "unknown",
                                              "source": None, "applied": None}


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


def test_a_slow_first_chunk_does_not_stop_a_run_that_fits(tmp_path, pal, monkeypatch):
    """D-020 check F10: chunk 1 also pays the one-time grammar compile, so once later chunks exist the deadline
    forecast reads them. Chunk 1 takes 150 s and the others 50 s, against a 400 s deadline: the old forecast
    (longest chunk x 1.25) stopped after 12 of 16 rows (250 + 187.5 > 400); all 16 now finish by 300 s."""
    clock = [0.0]

    class Compiling(FakeEngine):
        def chat(self, convs, sp):
            self.step_s = 150.0 if not self.calls else 50.0
            return super().chat(convs, sp)
    monkeypatch.setattr(junk, "load_engine", lambda model, **kw: Compiling([ORIG], clock=clock))
    monkeypatch.setattr(junk, "_now", lambda: clock[0])
    rc = junk.main(["generate", "--run", str(tmp_path), "--n", "16", "--chunk", "4", "--deadline-s", "400",
                    "--palette", str(DEV), "--hang-s", "0"])
    assert rc == 0
    meta = json.loads((tmp_path / "summary" / "generate.json").read_text())
    assert meta["generated"] == 16 and meta["stopped"] is None and clock[0] == 300.0


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


def _fake_vllm(monkeypatch, *, refuse_top_k_zero: bool, pin: str = "takes") -> dict:
    """vllm, torch and transformers stand-ins with the attributes VllmEngine reads (the laptop has none of them).
    pin "takes": LLM() takes structured_outputs_config and keeps its backend where vLLM's VllmConfig does ("auto"
    without it); "ignores": it takes the keyword and keeps "auto"; "refuses": LLM() raises TypeError on that
    keyword, as a vLLM without it would, and has no engine."""
    import types
    seen: dict = {"inits": []}

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
            seen["inits"].append(kw)
            if pin == "refuses" and "structured_outputs_config" in kw:
                raise TypeError("LLM.__init__() got an unexpected keyword argument 'structured_outputs_config'")
            seen["init"] = kw
            if pin in ("takes", "ignores"):
                backend = (kw.get("structured_outputs_config") or {}).get("backend", "auto")
                backend = "auto" if pin == "ignores" else backend
                self.llm_engine = types.SimpleNamespace(vllm_config=types.SimpleNamespace(
                    structured_outputs_config=types.SimpleNamespace(backend=backend)))

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


BASE_KW = {"model": "/hf/snap", "dtype": "bfloat16", "gpu_memory_utilization": 0.85, "max_model_len": 4096,
           "enforce_eager": True, "seed": 5}


def test_vllm_engine_pins_the_xgrammar_backend_and_reads_it_back_after_the_build(monkeypatch):
    seen = _fake_vllm(monkeypatch, refuse_top_k_zero=False)
    eng = junk.VllmEngine("/hf/snap", seed=5, gpu_mem=0.85, max_model_len=4096, backend="xgrammar")
    assert seen["inits"] == [{**BASE_KW, "structured_outputs_config": {"backend": "xgrammar"}}], "built once"
    assert eng.info["structured_outputs"] == PINNED


def test_vllm_engine_builds_without_the_pin_when_llm_refuses_the_keyword(monkeypatch):
    seen = _fake_vllm(monkeypatch, refuse_top_k_zero=False, pin="refuses")
    eng = junk.VllmEngine("/hf/snap", seed=5, gpu_mem=0.85, max_model_len=4096, backend="xgrammar")
    assert seen["inits"] == [{**BASE_KW, "structured_outputs_config": {"backend": "xgrammar"}}, BASE_KW]
    rec = eng.info["structured_outputs"]
    assert rec["requested"] == "xgrammar" and rec["pin_accepted"] is False
    assert "structured_outputs_config" in rec["pin_error"] and rec["pin_error"].startswith("TypeError")
    assert (rec["backend"], rec["source"], rec["applied"]) == ("unknown", None, None), "nothing to read: not fatal"
    assert eng.chat([[{"role": "user", "content": "a"}]], eng.sampling(**junk.SAMPLING))[0].text == "text 0"


def test_vllm_engine_records_a_pin_that_was_taken_but_not_applied(monkeypatch):
    seen = _fake_vllm(monkeypatch, refuse_top_k_zero=False, pin="ignores")
    eng = junk.VllmEngine("/hf/snap", seed=5, gpu_mem=0.85, max_model_len=4096, backend="xgrammar")
    assert len(seen["inits"]) == 1
    rec = eng.info["structured_outputs"]
    assert rec["pin_accepted"] is True and rec["backend"] == "auto" and rec["applied"] is False


def test_vllm_engine_unconstrained_asks_for_no_backend_and_records_vllms_default(monkeypatch):
    seen = _fake_vllm(monkeypatch, refuse_top_k_zero=False)
    eng = junk.VllmEngine("/hf/snap", seed=5, gpu_mem=0.85, max_model_len=4096)
    assert seen["inits"] == [BASE_KW]
    assert eng.info["structured_outputs"] == {"requested": None, "pin_accepted": None, "backend": "auto",
                                              "source": "llm_engine.vllm_config.structured_outputs_config.backend",
                                              "applied": None}


def test_read_backend_follows_the_attribute_chains_and_never_raises():
    ns = types.SimpleNamespace

    class Raising:
        @property
        def vllm_config(self):
            raise RuntimeError("engine core not reachable")

    class Mode:                                    # an enum member, as older vLLMs typed the backend
        value = "outlines"
    newest = ns(llm_engine=ns(vllm_config=ns(structured_outputs_config=ns(backend="xgrammar"))))
    older = ns(llm_engine=ns(vllm_config=ns(structured_outputs_config=ns(backend=None),
                                            decoding_config=ns(backend="guidance"))))
    v0_engine = ns(llm_engine=ns(decoding_config=ns(guided_decoding_backend=Mode())))
    assert junk.read_backend(newest) == {"backend": "xgrammar",
                                         "source": "llm_engine.vllm_config.structured_outputs_config.backend"}
    assert junk.read_backend(older) == {"backend": "guidance", "source": "llm_engine.vllm_config.decoding_config.backend"}
    assert junk.read_backend(v0_engine) == {"backend": "outlines",
                                            "source": "llm_engine.decoding_config.guided_decoding_backend"}
    for nothing in (object(), ns(llm_engine=Raising()), ns(llm_engine=None)):
        assert junk.read_backend(nothing) == {"backend": "unknown", "source": None}


def test_build_llm_retries_only_on_a_typeerror(monkeypatch):
    calls = []

    class Strict:
        def __init__(self, **kw):
            calls.append(kw)
            if "structured_outputs_config" in kw:
                raise ValueError("the regex is not supported by xgrammar")
    with pytest.raises(ValueError, match="xgrammar"):
        junk.build_llm(Strict, {"model": "m"}, "xgrammar")
    assert calls == [{"model": "m", "structured_outputs_config": {"backend": "xgrammar"}}], "a refusal stays loud"
    llm, pin = junk.build_llm(Strict, {"model": "m"}, None)
    assert isinstance(llm, Strict) and pin == {"requested": None, "pin_accepted": None}


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
    assert path == str(snap) and kw == {"seed": 1, "gpu_mem": 0.8, "max_model_len": 4096, "backend": None}
    path, kw = junk.load_engine(f"{repo}@{rev}", seed=1, gpu_mem=0.8, max_model_len=4096, backend="xgrammar")
    assert kw["backend"] == "xgrammar"


def test_model_spec_needs_a_pinned_commit():
    assert junk.split_model("org/name@" + "a" * 40) == ("org/name", "a" * 40)
    for bad in ("org/name", "org/name@main", "org/name@abc123"):
        with pytest.raises(ValueError):
            junk.split_model(bad)


FROZEN = ROOT / "palettes" / "lofi_rain_v1.yaml"
# tests/test_palette_frozen.py's pins: the sealed file (B) and its LoopScript regex, the gate's grammar (D-022)
FROZEN_SHA256 = "09f80a644476abe4d14517d0ec4129d69e1a4bc91eb0935c9ffd152411872183"
FROZEN_REGEX_SHA256 = "5c081828ac50718edc4db82c2a855ea52210debb375280ba01e8442de60f8558"


def test_every_step_reads_the_frozen_palette_by_default(tmp_path, monkeypatch):
    """The Phase-1 gate runs on the frozen palette (the freeze spec §5.5; D-022). jobs/junk_rate.sbatch passes no
    --palette, so generate, score and render all read junk.DEFAULT_PALETTE: palettes/lofi_rain_v1.yaml, the file
    tests/test_palette_frozen.py pins. D-020's runs used dev_gu, which their junk.json records."""
    assert junk.DEFAULT_PALETTE == FROZEN
    frozen = load_palette(FROZEN, verify=False)
    assert (frozen.id, str(frozen.frozen)) == ("lofi_rain_v1", "2026-11-04")
    assert frozen.sha256 == FROZEN_SHA256
    seen = {}
    for cmd in ("generate", "score", "render"):
        monkeypatch.setattr(junk, f"cmd_{cmd}", lambda a, cmd=cmd: seen.__setitem__(cmd, a.palette) or 0)
    assert junk.main(["generate", "--run", str(tmp_path), "--hang-s", "0"]) == 0
    assert junk.main(["score", "--run", str(tmp_path)]) == 0
    assert junk.main(["render", "--run", str(tmp_path)]) == 0
    assert seen == {"generate": FROZEN, "score": FROZEN, "render": FROZEN}
    job = (ROOT / "jobs" / "junk_rate.sbatch").read_text()
    code = [ln for ln in job.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert sum("-m lofi_rl.rl.junk" in ln for ln in code) == 3 and not any("--palette" in ln for ln in code)


def test_the_gates_generate_and_score_record_the_frozen_palette_and_its_grammar(tmp_path, monkeypatch):
    """The Phase-1 gate's own arguments (ARGS='500 2026 1 v2': prompt v2 and --constrained) with no --palette, as the
    job passes, through the fake engine: junk.json holds the two values D-022 swaps into D-020's validity list,
    palette.sha256 = B with matches_generate true and both regex shas = the frozen grammar's, and the v2 and grammar
    conditions the list keeps hold too. The completions are prompt v2's references under the frozen palette."""
    frozen = load_palette(FROZEN, verify=False)
    refs = prompt_v2.reference_texts(frozen, strict=True)
    engine = FakeEngine(list(refs.values())[:3], info={"engine": "fake", "structured_outputs": PINNED})
    monkeypatch.setattr(junk, "load_engine", lambda model, **kw: engine)
    monkeypatch.setattr(junk, "structured_outputs_params", lambda regex: {"regex_sha256": junk.sha256_text(regex)})
    n = 18
    assert junk.main(["generate", "--run", str(tmp_path), "--n", str(n), "--chunk", "8", "--hang-s", "0",
                      "--prompt", "v2", "--constrained"]) == 0
    assert junk.main(["score", "--run", str(tmp_path)]) == 0
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    assert s["palette"] == {"id": "lofi_rain_v1", "sha256": FROZEN_SHA256, "matches_generate": True}
    assert grammar.regex_sha256(frozen) == FROZEN_REGEX_SHA256
    c = s["constrained"]
    assert c["enabled"] is True and c["regex_sha256"] == c["in_language"]["regex_sha256"] == FROZEN_REGEX_SHA256
    assert c["in_language"]["matches_generate"] is True and c["in_language"]["other"]["n"] == 0
    assert c["in_language"]["stop"] == {"n": n, "in_language": n, "out_of_language": 0}
    p = s["prompt"]
    assert (p["version"], p["template_matches_generate"], p["rows_matching"], p["example_mismatch"]) == (
        "v2", True, n, 0)
    assert s["n"] == n and {g: s["failures"][g] for g in ("G1", "G3", "G4", "G5", "crash")} == {
        "G1": 0, "G3": 0, "G4": 0, "G5": 0, "crash": 0}


# ------------------------------------------------------------------------------------------- render (stubbed audio)
@pytest.mark.parametrize("jobs", ["1", "2"])
def test_render_samples_passing_loops_and_merges_g7_into_the_summary(tmp_path, pal, monkeypatch, jobs):
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
    assert junk.main(["render", "--run", str(tmp_path), "--k", "3", "--palette", str(DEV), "--no-verify",
                      "--jobs", jobs]) == 0
    scored = {r["i"]: r for r in map(json.loads, (tmp_path / "summary" / "scored.jsonl").read_text().splitlines())}
    pool = [i for i, r in scored.items() if r["ok"] and not r["near_copy"]]       # D-020: copies are not ear-checked
    assert pool and len(pool) < sum(r["ok"] for r in scored.values()), "the fixture has a passing near-copy to skip"
    pick = junk.sample_indices(pool, 3, seed=1)
    if jobs == "1":
        assert calls == pick, "the brief seed from generate.json"
    assert sorted(calls) == sorted(pick), "two lanes may start in either order; each loop renders once"
    clips = json.loads((tmp_path / "summary" / "clips.json").read_text())
    assert [c["i"] for c in clips["clips"]] == pick and clips["k"] == 3 and clips["seed"] == 1
    assert clips["passing"] == len(pool)
    assert all("copy_jaccard" in c and "brief" in c for c in clips["clips"])
    for c in clips["clips"]:
        assert (c["copy_reference"], c["quasi_copy"]) == (scored[c["i"]]["copy_reference"], scored[c["i"]]["quasi_copy"])
        assert c["copy_reference"] == "example"
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    want_fail = int(3 in calls)
    assert s["failures"]["G7"] == want_fail
    assert s["render"] == {"attempted": len(calls), "rendered": len(calls) - want_fail, "G7": want_fail,
                           "other_gates": 0, "errors": 0, "skipped": 0, "seed": 1, "k": 3}
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
    rc = junk.main(["render", "--run", str(tmp_path), "--k", "4", "--deadline-s", "50", "--palette", str(DEV),
                    "--no-verify", "--jobs", "1"])
    assert rc == 1, "an infrastructure error is not a gate"
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    assert s["render"]["errors"] == 1 and s["render"]["skipped"] == 1 and s["failures"]["G7"] == 0   # 2 non-copies


# R3 in six briefs and R2: seven passing loops, none a near-copy of the example (the 50-clip sample's pool, D-020)
POOL7 = [(R3, "stop", e, m, 310) for e, m in (("low", "melancholy"), ("low", "hopeful"), ("mid", "wistful"),
                                               ("mid", "hopeful"), ("high", "melancholy"), ("high", "wistful"))]
POOL7.append((R2, "stop", "high", "hopeful", 300))


def _pool7_run(run: Path, pal) -> list[int]:
    """A synthetic run whose render pool is POOL7, scored; the seeded sample of 6 that render takes."""
    _write_run(run, pal, POOL7, drift=None)
    assert junk.main(["score", "--run", str(run), "--palette", str(DEV)]) == 0
    scored = [json.loads(line) for line in (run / "summary" / "scored.jsonl").read_text().splitlines()]
    pool = [r["i"] for r in scored if r["ok"] and not r["near_copy"]]
    assert len(pool) == 7
    return junk.sample_indices(pool, 6, seed=1)


def test_render_lanes_split_the_workers_between_the_renders_at_a_time():
    """The freeze spec §5.5 (iii): the job passes its 4 CPUs as --workers, so the default 2 renders get 2 each."""
    assert junk.RENDER_JOBS == 2
    assert [junk.render_lanes(j, w) for j, w in ((2, 4), (1, 4), (2, 2), (2, 1), (2, 3), (4, 8), (8, 4))] == [
        (2, 2), (1, 4), (2, 1), (1, 1), (2, 1), (4, 2), (4, 1)]


def test_render_two_at_a_time_keeps_the_samples_order_and_splits_the_workers(tmp_path, pal, monkeypatch):
    """The first loop of the sample renders until the last one has started, so the other five finish first, on the
    other lane: clips.json still lists the six in the sample's order, two renders at a time, 2 workers each."""
    pick = _pool7_run(tmp_path, pal)
    lock, release = threading.Lock(), threading.Event()
    state = {"active": 0, "peak": 0, "finished": [], "workers": set()}

    def fake_clip(row, palette, clips_dir, *, cache, workers):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            state["workers"].add(workers)
        if row["i"] == pick[0]:
            assert release.wait(timeout=30), "the other lane never reached the last loop"
        elif row["i"] == pick[-1]:
            release.set()
        with lock:
            state["active"] -= 1
            state["finished"].append(row["i"])
        return {"i": row["i"], "ok": True, "gate": None, "clip": f"clips/{row['i']:04d}.flac"}
    monkeypatch.setattr(junk, "render_clip", fake_clip)
    assert junk.main(["render", "--run", str(tmp_path), "--k", "6", "--palette", str(DEV), "--no-verify",
                      "--workers", "4"]) == 0
    clips = json.loads((tmp_path / "summary" / "clips.json").read_text())
    assert [c["i"] for c in clips["clips"]] == pick and state["finished"][0] != pick[0]
    assert (clips["jobs"], clips["workers"]) == (2, 2) and state["workers"] == {2} and state["peak"] == 2
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    assert s["render"] == {"attempted": 6, "rendered": 6, "G7": 0, "other_gates": 0, "errors": 0, "skipped": 0,
                           "seed": 1, "k": 6}


def test_render_two_at_a_time_starts_no_render_after_the_deadline(tmp_path, pal, monkeypatch):
    """One lane holds the sample's first loop while the other renders the next three, each moving the clock 100 s
    on; against a 250-s deadline the fifth and sixth find the clock at 300 and are skipped, whichever lane takes
    them. The first loop started at 0, so it renders, though it ends after the deadline."""
    pick = _pool7_run(tmp_path, pal)
    clock, lock, release = [0.0], threading.Lock(), threading.Event()

    def fake_clip(row, palette, clips_dir, *, cache, workers):
        if row["i"] == pick[0]:
            assert release.wait(timeout=30), "the other lane never moved the clock to 300"
        else:
            with lock:
                clock[0] += 100.0
                if clock[0] >= 300.0:
                    release.set()
        return {"i": row["i"], "ok": True, "gate": None, "clip": f"clips/{row['i']:04d}.flac"}
    monkeypatch.setattr(junk, "render_clip", fake_clip)
    monkeypatch.setattr(junk, "_now", lambda: clock[0])
    assert junk.main(["render", "--run", str(tmp_path), "--k", "6", "--deadline-s", "250", "--palette", str(DEV),
                      "--no-verify"]) == 0
    clips = json.loads((tmp_path / "summary" / "clips.json").read_text())["clips"]
    assert [c["i"] for c in clips] == pick
    assert [bool(c["ok"]) for c in clips] == [True] * 4 + [False] * 2
    assert [c.get("skipped") for c in clips[4:]] == ["render deadline"] * 2
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    assert (s["render"]["attempted"], s["render"]["rendered"], s["render"]["skipped"]) == (4, 4, 2)


def test_render_refuses_no_jobs_or_no_workers(tmp_path):
    for bad in (["--jobs", "0"], ["--workers", "0"]):
        with pytest.raises(SystemExit) as err:
            junk.main(["render", "--run", str(tmp_path), *bad])
        assert err.value.code == 2


@needs_audio
def test_render_two_at_a_time_gives_one_at_a_times_clips_for_real(tmp_path, pal):
    """The freeze spec §5.5 (iii) on real audio: the same three loops rendered one at a time with 4 workers and two
    at a time with 2 workers each give the same records in the same order (render_s aside), and byte-identical FLACs
    and manifests. A render never depends on its workers (render.pipeline.render; tests/test_determinism.py)."""
    out = {}
    for jobs in ("1", "2"):
        run = tmp_path / f"jobs{jobs}"
        _write_run(run, pal, POOL7[:3], drift=None)
        assert junk.main(["score", "--run", str(run), "--palette", str(DEV)]) == 0
        assert junk.main(["render", "--run", str(run), "--k", "3", "--palette", str(DEV), "--workers", "4",
                          "--jobs", jobs]) == 0
        doc = json.loads((run / "summary" / "clips.json").read_text())
        files = {c["clip"]: ((run / c["clip"]).read_bytes(),
                             (run / c["clip"]).with_suffix(".manifest.json").read_bytes()) for c in doc["clips"]}
        out[jobs] = (doc, files)
    (one, one_files), (two, two_files) = out["1"], out["2"]
    assert (one["jobs"], one["workers"], two["jobs"], two["workers"]) == (1, 4, 2, 2)
    assert len(one["clips"]) == 3 and all(c["ok"] for c in one["clips"])
    strip = [{k: v for k, v in c.items() if k != "render_s"} for c in one["clips"]]
    assert [{k: v for k, v in c.items() if k != "render_s"} for c in two["clips"]] == strip
    assert two_files == one_files


# three committed drafts of brief d01's cell that play the frozen palette's three keys and both kits (their keys: and
# drums: lines): piano and jazz, wurli and studio, rhodes and jazz. The sample takes them as [0, 2, 1] (the brief seed
# 1), so the two renders that start together both play the jazz kit, whose one-shots are SoundFont notes rendered into
# the one-shot cache (lofi_rl.render.sampler.note_shot); the studio kit's are files, read from their source
FROZEN_DRAFTS = ("d01_groove", "d01_chords", "redo1/d01_chords")


def _kit(text: str) -> str:
    return re.search(r"^drums: (\w+)$", text, re.M)[1]


def _cached_shots(pal, kit: str) -> set[str]:
    """The one-shot cache keys of a kit's SoundFont shots, as kit_shots loads them (each distinct shot once)."""
    from lofi_rl.render import sampler        # here, not at the top: it needs the audio libraries (needs_audio)
    lanes = pal.kits[kit].lanes
    wanted = dict.fromkeys(s for lane in sampler.LANES for layer in sampler.LAYERS for s in lanes[lane].layers[layer])
    return {sampler.settings_key(sampler.note_settings(s, pal.sources[s.src].sha256)) for s in wanted if s.file is None}


def _frozen_sources_missing() -> list[str]:
    """The frozen palette's source files that are not on this machine: the fetch-only ones live under $LOFI_ASSETS
    (by default assets/big, never committed), so a fresh checkout has none of them."""
    frozen = load_palette(FROZEN, verify=False)
    missing = []
    for src in sorted(frozen.sources):
        try:
            path = frozen.source_path(src)
        except Exception as err:  # noqa: BLE001: an unset assets variable, say
            return [f"{src}: {err}"]
        if not path.is_file():
            missing.append(str(path))
    return missing


@needs_audio
def test_render_two_at_a_time_gives_one_at_a_times_clips_with_the_frozen_palette(tmp_path):
    """The Phase-1 gate's own configuration (the freeze spec §5.5 (iii); D-022): the frozen palette with its rain bed
    and banks. Two at a time with 2 workers each on a cold one-shot cache, where the two renders that start together
    both play the jazz kit and so both fill its one-shots at once, then one at a time with 4 workers on the filled
    cache: the same records in the same order (render_s aside), byte-identical FLACs and manifests, and each one-shot
    cached once. Skipped where the fetch-only sources are not here."""
    missing = _frozen_sources_missing()
    if missing:
        pytest.skip(f"the frozen palette's fetch-only sources are not here ({len(missing)}, e.g. {missing[0]})")
    frozen = load_palette(FROZEN, verify=False)
    cases = [((ROOT / "drafts" / "v1" / f"{d}.ls").read_text(encoding="utf-8"), "stop", "low", "melancholy", 300)
             for d in FROZEN_DRAFTS]
    assert sorted({_kit(text) for text, *_ in cases}) == ["jazz", "studio"]
    oneshots, jazz = tmp_path / "cache" / "oneshots", _cached_shots(frozen, "jazz")
    assert jazz and not _cached_shots(frozen, "studio"), "only the jazz kit's one-shots go through the cache"
    out = {}
    for jobs in ("2", "1"):
        run = tmp_path / f"jobs{jobs}"
        _write_run(run, frozen, cases, drift=None, version="v2")
        assert junk.main(["score", "--run", str(run), "--palette", str(FROZEN)]) == 0
        if jobs == "2":
            assert not oneshots.exists(), "the two-lane run starts on a cold cache"
        assert junk.main(["render", "--run", str(run), "--k", "3", "--palette", str(FROZEN), "--no-verify",
                          "--workers", "4", "--jobs", jobs, "--cache", str(tmp_path / "cache")]) == 0
        doc = json.loads((run / "summary" / "clips.json").read_text())
        files = {c["clip"]: ((run / c["clip"]).read_bytes(),
                             (run / c["clip"]).with_suffix(".manifest.json").read_bytes()) for c in doc["clips"]}
        out[jobs] = (doc, files)
        cached = sorted(p.name for p in oneshots.iterdir())
        assert cached == sorted(f"{k}.{ext}" for k in jazz for ext in ("json", "wav")), (
            "each jazz one-shot cached once, with its record, and no temporary file left")
    (one, one_files), (two, two_files) = out["1"], out["2"]
    pick = [c["i"] for c in two["clips"]]
    assert pick == junk.sample_indices([0, 1, 2], 3, seed=1) == [0, 2, 1]
    assert [_kit(cases[i][0]) for i in pick[:2]] == ["jazz", "jazz"], "the two lanes' first renders fill one kit"
    assert (one["jobs"], one["workers"], two["jobs"], two["workers"]) == (1, 4, 2, 2)
    assert one["palette"] == {"id": "lofi_rain_v1", "sha256": frozen.sha256}
    assert len(one["clips"]) == 3 and all(c["ok"] for c in one["clips"]), one["clips"]
    strip = [{k: v for k, v in c.items() if k != "render_s"} for c in one["clips"]]
    assert [{k: v for k, v in c.items() if k != "render_s"} for c in two["clips"]] == strip
    assert two_files == one_files
    manifest = json.loads(next(iter(one_files.values()))[1])
    assert manifest["bed"]["src"] == "rain" and manifest["bed"]["delta_lu"] == 24.0, "the frozen bed renders"


@pytest.mark.cluster
def test_render_writes_one_pass_flacs_for_real(tmp_path, pal):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("pedalboard")
    if not shutil.which("fluidsynth"):
        pytest.skip("no fluidsynth")
    # two loops in the render pool: since D-020 a near-copy of the example (as prompt.example_text is) never is
    _write_run(tmp_path, pal, [(R3, "stop", "mid", "wistful", 310), (ORIG, "stop", "low", "melancholy", 250)],
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
    """lofi_rl.{dsl, compile, theory, brief, palette, prompt, prompt_v2, rl.junk} import, and `score` runs, with mido,
    soundfile, scipy and pedalboard blocked (requirements/policy.lock.txt has none of them) and without numpy
    (the policy env's numpy 2.3.5 is not the laptop's 1.26.4, so the score path does not touch it). The knob readout
    (lofi_rl.verify.scoring) runs there too. lofi_rl.verify and lofi_rl.rl.reward are checked too once they exist."""
    _write_run(tmp_path, pal, drift=None)
    code = f"""
import importlib, importlib.util, sys
for m in ("mido", "soundfile", "scipy", "pedalboard", "numpy"):
    sys.modules[m] = None
for m in ("lofi_rl.dsl", "lofi_rl.compile", "lofi_rl.compile.compiler", "lofi_rl.theory", "lofi_rl.theory.voicing",
          "lofi_rl.brief", "lofi_rl.palette", "lofi_rl.prompt", "lofi_rl.prompt_v2", "lofi_rl.rl", "lofi_rl.rl.junk"):
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
    s = json.loads((tmp_path / "summary" / "junk.json").read_text())
    assert s["pass"]["count"] == 4
    assert s["copy"]["knobs_passing"] == {"n": 4, "energy_match": 2, "f_mood": 4, "energy_bands": [0.38, 0.59]}, (
        "the knob readout ran with numpy blocked")
