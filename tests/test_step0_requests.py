"""The step-0 requests (lofi_rl.step0.requests; docs/design/step0_samples.md §1.1, §1.2, §2.2 and §6.1) and the
registered values the steps share (lofi_rl.step0).

What these tests pin:
- the requests: 320 and 3,200 under Q6(a), or 320 and 1,760 under Q6(b), in stream, brief and k order, each seed
  brief_streams.draw_seed(seed, index, k) and the streams file's own, with the file's test vectors;
- the refusals: an edited streams file, a STREAMS whose mark is not MARK, a malformed STREAMS;
- MARK: the line mark_line gives for STREAMS and RULE;
- the registered values against what the code and the committed files make today (smoke.QWEN, prompt v2's template,
  the frozen palette and its grammar, the streams file, junk's sampling and engine names), and, where the run is on
  this machine, the values read from past runs (skipped elsewhere: runs/ is never committed).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

import lofi_rl.step0 as S
from lofi_rl import brief_streams, grammar, prompt_v2, smoke
from lofi_rl.palette import load_palette
from lofi_rl.rl import junk
from lofi_rl.step0 import Step0Error
from lofi_rl.step0 import requests as R
from lofi_rl.step0 import select as X

ROOT = Path(__file__).resolve().parents[1]
STREAMS_FILE = ROOT / "docs" / "prereg" / "brief_streams_v1.json"
# the 40 registered held-out indices of seed 5101 (brief_streams_v1.json, streams.rm_heldout; spec §1.2)
HELDOUT = (*range(0, 12), 13, 14, 15, *range(17, 27), 28, 29, 30, 31, 36, 37, 38, 39, 40, 47, 49, 50, 52, 54, 58)
Q6B = (("rm_heldout", "all"), ("train", "pool"))


@pytest.fixture(scope="module")
def reqs():
    return R.requests()


def test_q6a_draws_320_held_out_and_3200_train_requests_in_stream_brief_and_k_order(reqs):
    assert S.STREAMS == (("rm_heldout", "all"), ("train", "all")), "Q6(a), the recommended answer"
    assert len(reqs) == 3520 and Counter(r.stream for r in reqs) == {"rm_heldout": 320, "train": 3200}
    assert [r.i for r in reqs] == list(range(3520))
    assert [r.stream for r in reqs] == ["rm_heldout"] * 320 + ["train"] * 3200, "rm_heldout first, then train"
    held = [r.index for r in reqs if r.stream == "rm_heldout"][::8]
    assert tuple(held) == HELDOUT, "the stream's own order: its 40 registered indices"
    assert [r.index for r in reqs if r.stream == "train"][::8] == list(range(400)), "run 1's [0, 400)"
    assert [r.k for r in reqs] == list(range(8)) * 440, "k = 0..7 within each brief"
    assert all(r.brief_id == f"{r.seed}:{r.index}" for r in reqs)
    assert {r.seed for r in reqs if r.stream == "rm_heldout"} == {5101}
    assert {r.seed for r in reqs if r.stream == "train"} == {596}
    assert R.stream_counts(reqs) == {"rm_heldout": {"seed": 5101, "briefs": 40, "draws": 8},
                                     "train": {"seed": 596, "briefs": 400, "draws": 8}}


def test_every_request_carries_its_registered_draw_seed_and_the_files_vectors_hold(reqs):
    streams = R.load()
    picks = {(name, p.index): p for name in ("rm_heldout", "train") for p in streams[name].picks}
    for r in reqs:
        pick = picks[(r.stream, r.index)]
        assert r.draw_seed == brief_streams.draw_seed(r.seed, r.index, r.k) == pick.draw_seeds[r.k]
        assert r.brief == brief_streams.sample_brief(r.seed, r.index) == pick.brief
    by = {(r.seed, r.index, r.k): r.draw_seed for r in reqs}
    assert by[(596, 0, 0)] == 379233320 and by[(5101, 0, 0)] == 1702876545     # brief_streams_v1.json:50-57
    vectors = json.loads(STREAMS_FILE.read_text())["draw_seeds"]["test_vectors"]
    drawn = [v for v in vectors if (v["seed"], v["index"], v["k"]) in by]
    assert len(drawn) == 6, "596:0 k0-k3, 596:399 k7 and 5101:0 k0 are drawn; seeds 0 and 5104 are not"
    assert all(by[(v["seed"], v["index"], v["k"])] == v["draw_seed"] for v in drawn)
    assert len({r.draw_seed for r in reqs}) == 3520, "no seed repeats (brief_streams.build refuses one)"


def test_q6b_draws_only_the_pools_220_train_briefs(monkeypatch):
    mark = R.mark_line(Q6B)
    assert mark == ("D-023 mark: rule step0-select/1; streams rm_heldout (seed 5101, all 40 briefs) and train "
                    "(seed 596, the seed pool's 220 briefs by Q1 band); 8 draws a brief, 2080 requests")
    monkeypatch.setattr(S, "STREAMS", Q6B)
    monkeypatch.setattr(S, "MARK", mark)
    reqs = R.requests()
    assert Counter(r.stream for r in reqs) == {"rm_heldout": 320, "train": 1760}
    pool = X.pool_briefs(R.load()["train"])
    assert [r.index for r in reqs if r.stream == "train"][::8] == sorted(pool) == list(pool)
    assert [r.i for r in reqs] == list(range(2080))


def test_mark_is_the_line_for_streams_and_rule():
    assert S.MARK == R.mark_line(S.STREAMS, S.RULE) == R.mark_line()
    assert S.RULE in S.MARK and "rm_heldout (seed 5101" in S.MARK and "train (seed 596" in S.MARK
    assert S.MARK.endswith("3520 requests") and "\n" not in S.MARK, "one line"
    assert S.MARK.isascii()


def test_a_streams_that_differs_from_the_mark_is_refused(monkeypatch):
    monkeypatch.setattr(S, "STREAMS", Q6B)          # MARK still names every train brief
    with pytest.raises(Step0Error, match="draw only the streams D-023's mark names"):
        R.requests()
    monkeypatch.undo()
    with pytest.raises(Step0Error, match="but MARK is"):
        R.requests(mark=S.MARK.replace("step0-select/1", "step0-select/2"))
    with pytest.raises(Step0Error):
        R.requests(drawn=(("rm_heldout", "all"),))      # a mark for held-out alone is not MARK either


@pytest.mark.parametrize("drawn, why", [
    ((("measurement", "all"),), "needs a stream of"),
    ((("rm_heldout", "pool"),), "only train holds"),
    ((("train", "all"), ("rm_heldout", "all")), "in the order"),
    ((("train", "all"), ("train", "pool")), "each stream once"),
    ((("train", "some"),), "needs a stream of"),
    ((), "names no stream"),
])
def test_a_malformed_streams_is_refused(drawn, why):
    with pytest.raises(Step0Error, match=why):
        R.mark_line(drawn)


def test_an_edited_streams_file_is_refused(tmp_path):
    edited = tmp_path / "brief_streams_v1.json"
    data = STREAMS_FILE.read_bytes()
    edited.write_bytes(data.replace(b'"draw_seed": 379233320', b'"draw_seed": 379233321', 1))
    assert edited.read_bytes() != data
    with pytest.raises(Step0Error, match="the streams file is refused: .*not a fresh build"):
        R.requests(edited)
    shutil.copy(STREAMS_FILE, edited)
    assert len(R.requests(edited)) == 3520, "the committed bytes are accepted wherever they are"
    with pytest.raises(Step0Error, match="cannot|No such file"):
        R.requests(tmp_path / "missing.json")


def test_the_registered_values_are_what_the_code_and_files_make_today():
    assert (S.MODEL, S.MODEL_REVISION) == smoke.QWEN[:2]
    assert S.PROMPT_VERSION == prompt_v2.PROMPT_VERSION and S.TEMPLATE_SHA256 == prompt_v2.template_sha256()
    pal = load_palette(ROOT / S.PALETTE, verify=False)
    assert S.PALETTE_SHA256 == pal.sha256 == hashlib.sha256((ROOT / S.PALETTE).read_bytes()).hexdigest()
    assert S.REGEX_SHA256 == grammar.regex_sha256(pal)
    assert S.STREAMS_SHA256 == hashlib.sha256(STREAMS_FILE.read_bytes()).hexdigest()
    assert (ROOT / S.STREAMS_FILE) == brief_streams.STREAMS_FILE
    assert dict(S.V0_SAMPLING) == junk.SAMPLING and list(S.V0_SAMPLING) == list(junk.SAMPLING)
    assert S.BACKEND == junk.STRUCTURED_BACKEND and S.CHUNK == junk.DEFAULT_CHUNK
    assert S.DRAWS == brief_streams.DRAWS and dict(S.M) == {"heldout": 4, "refb": 4, "pool": 3}
    assert S.RULE == "step0-select/1" and S.ENGINE_SEED == 0 and S.PROBE_SEEDS == (1, 2)
    assert "/".join(S.PROBE_CELL) in S.PROMPT_TOKENS and S.PROBE_SAME <= S.PROBE_N
    assert S.GENERATE_DEADLINE_S == 960 and S.RENDER_DEADLINE_S == 900 and S.REPLAY_CHUNK == 160


def test_the_probe_seeds_are_none_of_the_files_6800_draw_seeds():
    seeds = [s for st in R.load().values() for p in st.picks for s in p.draw_seeds]
    assert len(seeds) == len(set(seeds)) == 6800
    assert not set(S.PROBE_SEEDS) & set(seeds) and len(set(S.PROBE_SEEDS)) == len(S.PROBE_SEEDS) == 2


def test_prompt_tokens_hold_one_count_per_cell():
    assert list(S.PROMPT_TOKENS) == [brief_streams.cell_name(c) for c in brief_streams.CELLS]
    assert min(S.PROMPT_TOKENS.values()) == 1325 and max(S.PROMPT_TOKENS.values()) == 1411
    assert len(set(S.PROMPT_TOKENS.values())) == 9, "each cell's own length"


def _pulled(rel: str) -> Path:
    p = ROOT / rel
    if not p.is_file():
        pytest.skip(f"{rel} is not on this machine (make pull brings it to the laptop)")
    return p


def test_the_values_read_from_the_gate_run_are_its_files():
    rows = [json.loads(line) for line in _pulled(f"{S.GATE_RUN}/summary/completions.jsonl").read_text().splitlines()]
    got: dict[str, set[int]] = {}
    for r in rows:
        got.setdefault(f"{r['brief']['energy']}/{r['brief']['mood']}", set()).add(r["prompt_tokens"])
    assert len(rows) == 500 and {c: {n} for c, n in S.PROMPT_TOKENS.items()} == got


def test_the_sampling_read_from_v0_is_its_generate_json():
    gen = json.loads(_pulled(f"{S.V0_RUN}/summary/generate.json").read_text())
    assert gen["sampling"] == dict(S.V0_SAMPLING) and list(gen["sampling"]) == list(S.V0_SAMPLING)
