"""The brief streams of D-021 item 1 (lofi_rl.brief_streams; docs/prereg/brief_streams_v1.json): the committed file is
a fresh build; each evaluation stream holds the first briefs of each energy x mood cell in stream order, up to the
stated per-cell counts; every draw seed is the documented hash (re-derived here with its own SplitMix64), in 31 bits,
and no seed repeats across streams, briefs or draws; the twin lists recompute from seed 596's [0, 400), and D-021's own
twin counts hold in stream order."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from lofi_rl import brief_streams as S
from lofi_rl.brief import ENERGIES, MOOD_NAMES, sample_brief, sample_briefs
from lofi_rl.prng import below

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "docs" / "prereg" / "brief_streams_v1.json"
MASK = (1 << 64) - 1
CELLS = [(e, m) for e in ENERGIES for m in MOOD_NAMES]
# D-021 item 1 as registered: stream -> (seed, briefs, stratified by cell).
D021 = {"train": (596, 400, False), "measurement": (2026, 500, False), "rm_heldout": (5101, 40, True),
        "dev": (5102, 40, True), "final_ab": (5103, 100, True), "partb_dev": (5104, 270, True)}
# D-021 item 1's laptop count, "in stream order": 8 of seed 5101's first 40, 4 of 5102's first 40, 15 of 5103's
# first 100.
D021_TWINS_IN_STREAM_ORDER = {5101: (40, 8), 5102: (40, 4), 5103: (100, 15)}


def sm64(x: int) -> int:                       # Vigna's reference SplitMix64, written out apart from lofi_rl.prng
    z = (x + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def want_key(seed: int, index: int) -> int:     # the file's formula, field b"draw" = 0x64726177
    return sm64(sm64(sm64(seed % 2 ** 64) ^ index) ^ 0x64726177)


def want_seed(seed: int, index: int, k: int) -> int:
    return sm64(want_key(seed, index) ^ k) >> 33


def key4(b) -> tuple:
    return b.energy, b.mood, b.bpm, b.tonic


def row4(r: dict) -> tuple:
    return r["energy"], r["mood"], r["bpm"], r["tonic"]


@pytest.fixture(scope="module")
def doc() -> dict:
    return json.loads(FILE.read_text(encoding="ascii"))


def test_the_committed_file_is_a_fresh_build(doc):
    assert S.STREAMS_FILE == FILE
    assert FILE.read_bytes() == S.text().encode("ascii")
    assert S.main(["--check", str(FILE)]) == 0 and S.main(["--check"]) == 0
    assert doc == S.document() and doc["schema"] == "lofi-rl/brief-streams/1"
    assert set(S.load(FILE)) == set(D021)


def test_check_and_load_refuse_any_edit(tmp_path):
    good = FILE.read_text(encoding="ascii")
    seed0 = str(json.loads(good)["streams"]["dev"]["briefs"][0]["draw_seeds"][0])
    edits = {"a seed": good.replace(seed0, str(int(seed0) + 1), 1), "a space": good.replace('": ', '":  ', 1),
             "no final newline": good.rstrip("\n"), "CRLF": good.replace("\n", "\r\n")}
    for name, bad in edits.items():
        assert bad != good, name
        p = tmp_path / "edited.json"
        p.write_bytes(bad.encode("ascii"))
        assert S.main(["--check", str(p)]) == 1, name
        with pytest.raises(S.BriefStreamsError, match="not a fresh build"):
            S.load(p)
    assert S.main(["--check", str(tmp_path / "absent.json")]) == 2
    with pytest.raises(S.BriefStreamsError):
        S.load(tmp_path / "absent.json")


def test_the_writer_never_replaces_a_different_file_without_force(tmp_path):
    out = tmp_path / "prereg" / "streams.json"
    assert S.main(["--out", str(out)]) == 0 and out.read_bytes() == FILE.read_bytes()
    assert S.main(["--out", str(out)]) == 0                                 # unchanged: nothing to write
    out.write_text("{}\n", encoding="ascii")
    assert S.main(["--out", str(out)]) == 1 and out.read_text(encoding="ascii") == "{}\n"
    assert S.main(["--out", str(out), "--force"]) == 0 and out.read_bytes() == FILE.read_bytes()


def test_the_streams_are_d021s_and_every_brief_is_the_samplers(doc):
    streams = doc["streams"]
    assert list(streams) == list(D021)
    assert {n: (s["seed"], s["count"], s["selection"] == "stratified") for n, s in streams.items()} == D021
    for name in ("train", "measurement"):                                  # ranges: [0, count), in order
        s = streams[name]
        assert (s["selection"], s["start"], s["stop"], s["read"]) == ("range", 0, s["count"], s["count"])
        assert [r["index"] for r in s["briefs"]] == list(range(s["count"]))
        got = Counter(f"{r['energy']}/{r['mood']}" for r in s["briefs"])
        assert s["cell_counts"] == {f"{e}/{m}": got[f"{e}/{m}"] for e, m in CELLS}
    for name, s in streams.items():
        for r in s["briefs"]:
            assert row4(r) == key4(sample_brief(s["seed"], r["index"])), (name, r["index"])
    assert [key4(b) for b in sample_briefs(596, 400)] == [row4(r) for r in streams["train"]["briefs"]]


def test_per_cell_counts_follow_the_stated_rule(doc):
    assert list(S.per_cell_counts(40)) == CELLS
    assert list(S.per_cell_counts(40).values()) == [5, 5, 5, 5, 4, 4, 4, 4, 4]
    assert list(S.per_cell_counts(100).values()) == [12, 11, 11, 11, 11, 11, 11, 11, 11]
    assert list(S.per_cell_counts(270).values()) == [30] * 9
    # D-021's sizes: "4 or 5 a cell for 40, 11 or 12 for 100, and 30 for 270".
    assert [set(S.per_cell_counts(n).values()) for n in (40, 100, 270)] == [{4, 5}, {11, 12}, {30}]
    for n in range(120):
        q, r = list(S.per_cell_counts(n).values()), n % 9
        assert q[:r] == [-(-n // 9)] * r and q[r:] == [n // 9] * (9 - r) and sum(q) == n, n
    for bad in (-1, True, 2.0, "40"):
        with pytest.raises(S.BriefStreamsError):
            S.per_cell_counts(bad)
    for name in S.EVALUATION:
        s = doc["streams"][name]
        assert s["per_cell"] == {f"{e}/{m}": k for (e, m), k in S.per_cell_counts(s["count"]).items()}, name
        got = Counter(f"{r['energy']}/{r['mood']}" for r in s["briefs"])
        assert dict(got) == {c: k for c, k in s["per_cell"].items() if k}, name


def test_each_evaluation_stream_takes_the_first_briefs_of_each_cell_in_stream_order(doc):
    assert S.EVALUATION == ("rm_heldout", "dev", "final_ab", "partb_dev")
    for name in S.EVALUATION:
        s = doc["streams"][name]
        quota = {tuple(c.split("/")): k for c, k in s["per_cell"].items()}
        idx = [r["index"] for r in s["briefs"]]
        assert idx == sorted(set(idx)) and len(idx) == s["count"] == sum(quota.values()), name
        assert s["read"] == idx[-1] + 1, name
        seen: dict[tuple, list[int]] = {c: [] for c in CELLS}
        for i in range(s["read"]):                                        # the test's own walk of the stream
            b = sample_brief(s["seed"], i)
            seen[(b.energy, b.mood)].append(i)
        assert idx == sorted(i for c in CELLS for i in seen[c][:quota[c]]), name
        last = sample_brief(s["seed"], s["read"] - 1)                      # the walk stops at the brief that fills
        assert len(seen[(last.energy, last.mood)]) == quota[(last.energy, last.mood)], name   # the last open cell
        assert S.stratified_indices(s["seed"], s["count"]) == (tuple(idx), s["read"]), name
    with pytest.raises(S.BriefStreamsError, match="still have room"):
        S.stratified_indices(5101, 40, limit=20)                           # fails closed, never short


def test_draw_seeds_are_the_documented_hash_in_31_bits(doc):
    d = doc["draw_seeds"]
    assert d["field"] == S.FIELD_DRAW == 0x64726177 == int.from_bytes(b"draw", "big")
    assert S.FIELD_DRAW not in (1, 2, 3, 4)                                # brief.py's field ids
    assert d["bits"] == 31 and S.DRAWS == 8
    assert d["draws"] == {n: (0 if n == "measurement" else 8) for n in doc["streams"]}
    assert all(s["draws"] == d["draws"][n] for n, s in doc["streams"].items())
    assert "0xe220a8397b1dcdaf" in " ".join(d["formula"])                  # splitmix64(0), prng.TEST_VECTOR
    by_seed = {s["seed"]: {r["index"]: r["draw_seeds"] for r in s["briefs"]} for s in doc["streams"].values()}
    for v in d["test_vectors"]:
        assert v["key"] == f"0x{want_key(v['seed'], v['index']):016x}", v
        assert v["draw_seed"] == want_seed(v["seed"], v["index"], v["k"]) == S.draw_seed(v["seed"], v["index"], v["k"])
        if v["seed"]:
            assert by_seed[v["seed"]][v["index"]][v["k"]] == v["draw_seed"], v   # a draw the table holds
    for name, s in doc["streams"].items():
        for r in s["briefs"]:
            assert r["draw_seeds"] == [want_seed(s["seed"], r["index"], k) for k in range(s["draws"])], (name, r["index"])
            assert all(0 <= x < 2 ** 31 for x in r["draw_seeds"]), (name, r["index"])
    # Edges: the seed is read mod 2**64; the index and k are 64-bit words; prng.below(h, 2**31) is h >> 33.
    assert S.draw_seed(MASK, MASK, 3) == want_seed(MASK, MASK, 3)
    assert S.draw_seed(1 << 64, 7, 1) == S.draw_seed(0, 7, 1) and S.draw_seed(-1, 7, 1) == S.draw_seed(MASK, 7, 1)
    for bad in (-1, 1 << 64, True):
        with pytest.raises(S.BriefStreamsError):
            S.draw_seed(0, bad, 0)
        with pytest.raises(S.BriefStreamsError):
            S.draw_seed(0, 0, bad)
    for h in (0, 1, MASK, 1 << 63, 0xE220A8397B1DCDAF):
        assert below(h, 1 << 31) == h >> 33


def test_a_draw_key_never_reuses_a_hash_that_drew_its_briefs_fields():
    for seed, index in ((596, 0), (5101, 58), (MASK, 12345)):
        fields = {sm64(sm64(sm64(seed) ^ index) ^ f) for f in (1, 2, 3, 4)}   # brief.py's keys (COMPILER.md §6)
        assert S.draw_key(seed, index) not in fields


def test_no_draw_seed_repeats_across_streams_briefs_or_draws(doc, monkeypatch):
    seeds = [x for s in doc["streams"].values() for r in s["briefs"] for x in r["draw_seeds"]]
    assert len(seeds) == 8 * sum(n for name, (_, n, _) in D021.items() if name != "measurement") == 6800
    assert len(set(seeds)) == len(seeds)
    assert len({s["seed"] for s in doc["streams"].values()}) == len(D021)
    monkeypatch.setattr(S, "draw_seed", lambda seed, index, k: index * 4 + k)    # brief 0 of every stream alike
    with pytest.raises(S.BriefStreamsError, match="repeats"):
        S.build()


def test_twins_recompute_from_train_and_none_is_excluded(doc):
    by: dict[tuple, list[int]] = {}
    for i, b in enumerate(sample_briefs(596, 400)):
        by.setdefault(key4(b), []).append(i)
    tw = doc["twins"]
    assert tw["against"] == {"stream": "train", "seed": 596, "start": 0, "stop": 400}
    assert list(tw["streams"]) == list(S.EVALUATION)
    for name in S.EVALUATION:
        s = doc["streams"][name]
        want = [{"id": f"{s['seed']}:{r['index']}", "train": [f"596:{j}" for j in by[row4(r)]]}
                for r in s["briefs"] if row4(r) in by]
        assert tw["streams"][name] == {"count": len(want), "of": s["count"], "briefs": want}, name
        assert len(s["briefs"]) == s["count"], name                       # every twin stays in its stream
    assert {n: tw["streams"][n]["count"] for n in S.EVALUATION} == {"rm_heldout": 8, "dev": 4, "final_ab": 13,
                                                                   "partb_dev": 40}


def test_d021s_twin_counts_hold_in_stream_order(doc):
    train = {key4(b) for b in sample_briefs(596, 400)}
    for seed, (n, count) in D021_TWINS_IN_STREAM_ORDER.items():
        got = [i for i, b in enumerate(sample_briefs(seed, n)) if key4(b) in train]
        assert len(got) == count, (seed, n, got)
    order = doc["twins"]["in_stream_order"]["streams"]
    assert list(order) == list(S.EVALUATION)
    for name in S.EVALUATION:
        s = doc["streams"][name]
        ids = [f"{s['seed']}:{i}" for i, b in enumerate(sample_briefs(s["seed"], s["count"])) if key4(b) in train]
        assert order[name] == {"first": s["count"], "count": len(ids), "ids": ids}, name


def test_the_file_says_what_held_out_means_and_that_nothing_is_excluded(doc):
    held = " ".join(doc["held_out"])
    assert "fresh samples of the same 9 prompts" in held and "never reads" in held
    assert doc["excluded"][0].startswith("None") and "twins included" in doc["excluded"][0]
    assert doc["cells"] == [list(c) for c in CELLS]


def test_the_file_is_ascii_json_with_one_brief_a_line():
    raw = FILE.read_bytes()
    assert raw.isascii() and b"\\u" not in raw and b"\r" not in raw and raw.endswith(b"}\n")
    rows = [ln for ln in raw.decode("ascii").splitlines() if ln.lstrip().startswith('{"index": ')]
    assert len(rows) == sum(n for _, n, _ in D021.values()) == 1350


def test_the_module_needs_only_the_standard_library_and_the_brief_sampler():
    probe = ("import sys\n"
             "for m in ('yaml', 'numpy', 'torch'):\n"
             "    sys.modules[m] = None\n"
             "import lofi_rl.brief_streams as S\n"
             "raise SystemExit(S.main(['--check']))\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
