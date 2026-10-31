"""kit.bench + scripts/scaling_study.py: throughput points, the nodes axis, multi-node points kept as their
own points, and no stats file ever dropped silently. (The kit's 2-node job and its test were dropped;
Phase 4 writes the render-scaling jobs.)"""
import csv
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from kit.bench import load_task, run

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import scaling_study  # noqa: E402
from scaling_study import merge_node_stats, scaling_table  # noqa: E402


def test_bench_runs_a_task_and_writes_stats(tmp_path):
    s = run("kit.bench:demo_task", 8, 2, tmp_path / "w2")
    assert s["n"] == 8 and s["workers"] == 2 and s["throughput_per_min"] > 0
    assert json.loads((tmp_path / "w2" / "stats.json").read_text())["task"] == "kit.bench:demo_task"
    assert load_task("kit.bench:demo_task")(1) == load_task("kit.bench:demo_task")(1)


def test_scaling_table_uses_throughput_so_workload_size_does_not_matter(tmp_path):
    rows = [{"n": 100, "workers": 1, "wall_s": 120.0}, {"n": 1000, "workers": 8, "wall_s": 150.0},
            {"n": 1000, "workers": 64, "wall_s": 30.0, "throughput_per_min": 2000.0}]
    t = scaling_study.scaling_table(rows)
    assert [r["workers"] for r in t] == [1, 8, 64] and t[0]["throughput"] == 50.0
    assert t[1]["speedup"] == 8.0 and t[1]["efficiency"] == 1.0 and t[2]["speedup"] == 40.0
    g = scaling_study.scaling_table([{"gpus": 1, "calls_per_min": 1025.3}, {"gpus": 2, "calls_per_min": 2040.2}],
                                    x="gpus", metric="calls_per_min")
    assert g[1]["speedup"] == round(2040.2 / 1025.3, 3)
    scaling_study.write_csv(t, tmp_path / "s.csv")
    scaling_study.plot(t, tmp_path / "s.png", "t")
    assert (tmp_path / "s.png").stat().st_size > 1000


def _bench(job, workers, n=1000, wall=10.0, nodes="1", host="h1", **kw):
    """A stats row shaped like kit.bench's (slurm_nodes is a string: it comes from the environment)."""
    r = {"task": "kit.bench:demo_task", "n": n, "workers": workers, "wall_s": wall,
         "throughput_per_min": round(60 * n / wall, 2), "mean_task_s": 0.5, "host": host,
         "slurm_job": str(job), "slurm_nodes": nodes, "file": f"bench_{job}_w{workers}_{host}.json"}
    r.update(kw)
    return r


def test_nodes_axis_reads_slurm_nodes_and_keeps_the_full_node_point():
    sweep = [_bench(100, w, wall=600.0 / w) for w in (1, 2, 4)]                  # one node: 100, 200, 400 /min
    shares = [_bench(200, 4, n=500, wall=80.0, nodes="2", host="a"), _bench(200, 4, n=500, wall=78.0, nodes="2", host="b")]
    msgs = []
    t = scaling_table(sweep + shares, x="nodes", warn=msgs.append)
    assert [r["nodes"] for r in t] == [1, 2]
    assert t[0]["file"] == "bench_100_w4_h1.json"                                 # the full node, not the 1-worker run
    assert any("nodes=1" in m and "ignored" in m and "bench_100_w1_h1.json" in m and "bench_100_w2_h1.json" in m
               for m in msgs), msgs
    assert t[1]["n"] == 1000 and t[1]["wall_s"] == 80.0 and t[1]["throughput"] == 750.0   # slowest node's wall
    assert t[1]["speedup"] == 1.875 and t[1]["efficiency"] == round(1.875 / 2, 3)


def test_per_node_files_become_their_own_point_instead_of_being_swallowed():
    # the review's reproduction: a 1/2/4-worker sweep (job 100) + kit.bench run on each of 2 nodes (job 200)
    rows = [_bench(100, w, n=100, wall=60.0 / w) for w in (1, 2, 4)]
    rows += [_bench(200, 4, n=100, wall=8.0, nodes="2", host="a"), _bench(200, 4, n=100, wall=8.4, nodes="2", host="b")]
    msgs = []
    t = scaling_table(rows, warn=msgs.append)
    assert [(r["workers"], r["nodes"]) for r in t] == [(1, 1), (2, 1), (4, 1), (8, 2)]
    assert t[3]["file"] == "bench_200_w4_a.json;bench_200_w4_b.json" and t[3]["host"] == "a,b"
    assert any("merged 2 per-node files" in m for m in msgs), msgs


def test_incomplete_or_duplicated_node_files_are_skipped_loudly():
    msgs = []
    t = scaling_table([_bench(100, 4), _bench(200, 4, nodes="2", host="a")], warn=msgs.append)
    assert [(r["workers"], r["nodes"]) for r in t] == [(4, 1)]
    assert any("1 of 2 per-node files" in m and "bench_200_w4_a.json" in m for m in msgs), msgs
    a = _bench(400, 4, nodes="2", host="a")
    msgs = []
    assert scaling_table([a, dict(a, file="copy/" + a["file"])], warn=msgs.append) == []
    assert any("repeats another file" in m for m in msgs) and any("1 of 2" in m for m in msgs), msgs


def test_a_whole_run_file_covers_its_own_node_files():
    whole = merge_node_stats([_bench(300, 4, nodes="2", host="a"), _bench(300, 4, nodes="2", host="b")], wall_s=11.0)
    whole["file"] = "bench_300_w8.json"
    msgs = []
    t = scaling_table([whole, _bench(300, 4, nodes="2", host="a")], warn=msgs.append)
    assert [(r["file"], r["nodes"], r["workers"]) for r in t] == [("bench_300_w8.json", 2, 8)]
    assert any("covers them" in m for m in msgs), msgs


def test_duplicate_points_keep_the_latest_job_and_name_the_others():
    msgs = []
    t = scaling_table([_bench(100, 8, wall=10.0), _bench(300, 8, wall=9.0), _bench(100, 1, n=100, wall=10.0)],
                      warn=msgs.append)
    assert [(r["workers"], r["job"]) for r in t] == [(1, "100"), (8, "300")]
    assert any("workers=8 nodes=1" in m and "kept bench_300_w8_h1.json" in m and "bench_100_w8_h1.json" in m
               for m in msgs), msgs


def test_files_without_the_axis_are_named_and_main_fails_when_nothing_is_left(tmp_path):
    msgs = []
    assert scaling_table([_bench(100, 4)], x="gpus", warn=msgs.append) == []
    assert any("skipped" in m and "'gpus'" in m for m in msgs), msgs
    f = tmp_path / "bench_100_w4.json"
    f.write_text(json.dumps(_bench(100, 4)))
    assert scaling_study.main(["--glob", str(f), "--x", "gpus", "--outdir", str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out").exists()


def test_merge_node_stats_adds_up_and_records_each_node():
    p = [_bench(5, 64, wall=40.0, nodes="2", host="a", cpu_model="EPYC 7513"),
         _bench(5, 64, wall=41.5, nodes="2", host="b", cpu_model="EPYC 7542")]
    m = merge_node_stats(p, wall_s=42.0)
    assert (m["n"], m["workers"], m["nodes"], m["wall_s"]) == (2000, 128, 2, 42.0)
    assert m["throughput_per_min"] == round(60 * 2000 / 42.0, 2) and m["per_node_wall_s"] == [40.0, 41.5]
    assert m["cpu_models"] == ["EPYC 7513", "EPYC 7542"] and m["host"] == "a,b" and "whole" in m["wall_measured"]
    assert merge_node_stats(p)["wall_s"] == 41.5 and "slowest" in merge_node_stats(p)["wall_measured"]
    with pytest.raises(ValueError):
        merge_node_stats([p[0], dict(p[1], task="other:task")])
    with pytest.raises(ValueError):
        merge_node_stats([])


def test_real_kit_bench_files_from_a_two_node_allocation(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    run("kit.bench:demo_task", 8, 2, tmp_path / "one")
    monkeypatch.setenv("SLURM_JOB_ID", "78")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    for k in (0, 1):                                      # what running kit.bench on each node writes
        run("kit.bench:demo_task", 8, 2, tmp_path / f"node{k}")
    rows = scaling_study.load(str(tmp_path / "*" / "stats.json"))
    for r in rows:
        r["host"] = Path(r["file"]).parent.name            # one laptop plays both nodes
    t = scaling_table(rows, x="nodes", warn=[].append)
    assert [(r["nodes"], r["n"], r["job"]) for r in t] == [(1, 8, "77"), (2, 16, "78")]


def test_cli_combines_globs_keeps_the_csv_columns_and_draws_both_series(tmp_path, capsys):
    a, b = tmp_path / "sweep", tmp_path / "two"
    a.mkdir()
    b.mkdir()
    for w in (1, 2, 4):
        (a / f"bench_100_w{w}.json").write_text(json.dumps(_bench(100, w, wall=600.0 / w)))
    whole = merge_node_stats([_bench(200, 4, nodes="2", host="x"), _bench(200, 4, nodes="2", host="y")], wall_s=12.0)
    (b / "bench_200_w8.json").write_text(json.dumps(whole))
    out = tmp_path / "out"
    assert scaling_study.main(["--glob", f"{a}/*.json", "--glob", f"{b}/*.json", "--tag", "w", "--outdir", str(out)]) == 0
    with open(out / "scaling_w.csv") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["workers", "nodes", "n", "wall_s", "throughput", "host", "job", "file", "speedup", "efficiency"]
    assert [(r["workers"], r["nodes"]) for r in rows] == [("1", "1"), ("2", "1"), ("4", "1"), ("8", "2")]
    assert (out / "scaling_w.png").stat().st_size > 1000
    assert "nodes=2" in capsys.readouterr().out
    assert scaling_study.main(["--glob", f"{a}/*.json", "--glob", f"{b}/*.json", "--x", "nodes", "--tag", "n",
                               "--outdir", str(out)]) == 0
    with open(out / "scaling_n.csv") as f:
        assert [r["nodes"] for r in csv.DictReader(f)] == ["1", "2"]
    assert "WARNING" in capsys.readouterr().err
