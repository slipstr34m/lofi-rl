#!/usr/bin/env python
"""Strong-scaling table + plot from benchmark stats files (kit.bench, kit.bench_vllm or anything with the same keys).

    python scripts/scaling_study.py --glob 'artifacts/scaling/bench_<job>_w*.json' --tag cpu_v1
    python scripts/scaling_study.py --glob 'artifacts/scaling/bench_<job2>_w*.json' --x nodes --tag nodes_v1
    python scripts/scaling_study.py --glob 'artifacts/scaling/vllm_<job>_g*.json' --x gpus --metric calls_per_min --tag gpu_v1

`--glob` may be repeated (e.g. the one-node sweep plus the two-node job). Each point is a THROUGHPUT
(items per minute; 60*n/wall_s when the file has no explicit metric), so points measured on different
workload sizes stay comparable. Speed-up and efficiency are relative to the smallest x. Writes
docs/scaling/scaling_<tag>.csv/.png.

Nothing is dropped silently: every file that is not a point of the table is named in a WARNING on stderr.
- One point per (x, node count). A two-node run is its own point, never folded into a one-node point
  with the same x. A file's node count is its `nodes`, else `slurm_nodes` (what kit.bench writes),
  else 1; `--x nodes` uses the same rule.
- A kit.bench file from a multi-node allocation (`slurm_nodes` > 1 and no `nodes` key) is ONE node's
  share of the work. The files of one job (same task, workers and n) are merged into one point: n and
  workers add up, and the wall time is the slowest node's, so launch skew is not counted.
  jobs/bench_scaling_2node.sbatch measures the whole step instead and writes one file with `nodes`.
  A group that is missing a node's file is skipped.
- Several files at one point: the file with the most workers is kept (on the nodes axis that is the
  full-node run), then the latest SLURM job (a rerun supersedes), then the last file name.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path


def _warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def _int(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _name(r: dict) -> str:
    return str(r.get("file") or "<row without a file>")


def load(patterns: str | list[str]) -> list[dict]:
    """Every stats file matched by one or more glob patterns (each file once, sorted by path)."""
    if isinstance(patterns, str):
        patterns = [patterns]
    rows = []
    for f in sorted({f for p in patterns for f in glob.glob(p)}):
        s = json.loads(Path(f).read_text())
        s["file"] = f
        rows.append(s)
    return rows


def node_count(r: dict) -> int:
    """`nodes` (a whole-run file), else `slurm_nodes` (kit.bench: SLURM_JOB_NUM_NODES), else 1."""
    return _int(r.get("nodes")) or _int(r.get("slurm_nodes")) or 1


def x_value(r: dict, x: str) -> int:
    if x == "nodes":
        return node_count(r)
    v = _int(r.get(x))
    if v is None:
        keys = ", ".join(sorted(k for k in r if k != "file"))
        raise KeyError(f"no integer {x!r} (keys: {keys})")
    return v


def throughput(r: dict, metric: str | None) -> float:
    if metric:
        if r.get(metric) is None:
            raise KeyError(f"no metric {metric!r}")
        return float(r[metric])
    if r.get("throughput_per_min") is not None:
        return float(r["throughput_per_min"])
    return 60.0 * float(r["n"]) / float(r["wall_s"])


def merge_node_stats(parts: list[dict], wall_s: float | None = None) -> dict:
    """One whole-run point from the per-node kit.bench stats of a multi-node run (one dict per node).

    n and workers add up. wall_s is the run's wall time when the job measured it around the whole
    multi-node step (jobs/bench_scaling_2node.sbatch); otherwise it is the slowest node's own wall,
    which leaves out launch skew between nodes."""
    if not parts:
        raise ValueError("no per-node stats to merge")
    tasks = {p.get("task") for p in parts}
    if len(tasks) > 1:
        raise ValueError(f"per-node stats ran different tasks: {sorted(map(str, tasks))}")
    walls = [float(p["wall_s"]) for p in parts]
    wall = float(wall_s) if wall_s is not None else max(walls)
    n = sum(int(p["n"]) for p in parts)
    means = [float(p["mean_task_s"]) for p in parts if p.get("mean_task_s") is not None]
    out = {
        "task": parts[0].get("task"), "n": n, "workers": sum(int(p["workers"]) for p in parts),
        "nodes": len(parts), "slurm_nodes": len(parts), "wall_s": round(wall, 3),
        "throughput_per_min": round(60 * n / wall, 2) if wall > 0 else None,
        "wall_measured": "around the whole step (launch, start-up and the slowest node included)" if wall_s is not None
                         else "slowest node's own wall (launch skew not counted)",
        "per_node_workers": [int(p["workers"]) for p in parts], "per_node_n": [int(p["n"]) for p in parts],
        "per_node_wall_s": walls, "max_node_wall_s": max(walls),
        "mean_task_s": round(sum(means) / len(means), 4) if means else None,
        "host": ",".join(str(p.get("host")) for p in parts), "slurm_job": parts[0].get("slurm_job"),
    }
    if any(p.get("cpu_model") for p in parts):
        out["cpu_models"] = [p.get("cpu_model") for p in parts]
    return out


def _is_node_share(r: dict) -> bool:
    return "nodes" not in r and (_int(r.get("slurm_nodes")) or 1) > 1


def combine_node_shares(rows: list[dict], warn=_warn) -> list[dict]:
    """Replace the per-node kit.bench files of each multi-node run by one merged point (module docstring)."""
    covered = {str(r.get("slurm_job")) for r in rows if not _is_node_share(r) and node_count(r) > 1}
    out, groups = [], {}
    for r in rows:
        if _is_node_share(r):
            key = (str(r.get("slurm_job")), str(r.get("task")), str(r.get("workers")), str(r.get("n")))
            groups.setdefault(key, []).append(r)
        else:
            out.append(r)
    for (job, _task, workers, n), parts in sorted(groups.items()):
        names = ", ".join(_name(p) for p in parts)
        if job in covered:
            warn(f"job {job}: per-node files ignored, the job's whole-run file covers them: {names}")
            continue
        uniq, seen = [], set()
        for p in parts:
            key = json.dumps({k: v for k, v in p.items() if k != "file"}, sort_keys=True, default=str)
            if key in seen:
                warn(f"job {job}: {_name(p)} repeats another file's numbers exactly (a copy of the same node?); ignored")
                continue
            seen.add(key)
            uniq.append(p)
        want = _int(parts[0].get("slurm_nodes"))
        if len(uniq) != want:
            warn(f"job {job}: {len(uniq)} of {want} per-node files (workers={workers}, n={n}): "
                 f"incomplete multi-node run, skipped: {names}")
            continue
        merged = merge_node_stats(uniq)
        merged["file"] = ";".join(_name(p) for p in uniq)
        warn(f"job {job}: merged {want} per-node files into one {want}-node point (wall = slowest node, launch "
             f"skew not counted; jobs/bench_scaling_2node.sbatch measures it): {names}")
        out.append(merged)
    return out


def scaling_table(rows: list[dict], x: str = "workers", metric: str | None = None, warn=None) -> list[dict]:
    warn = warn or _warn
    cands: dict[tuple[int, int], list] = {}
    for r in combine_node_shares(rows, warn):
        try:
            k, thr = x_value(r, x), throughput(r, metric)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            warn(f"{_name(r)}: skipped, no usable {x!r}/{metric or 'throughput'} ({e})")
            continue
        nodes = node_count(r)
        rec = {x: k, "nodes": nodes, "n": r.get("n"), "wall_s": r.get("wall_s"), "throughput": round(thr, 2),
               "host": r.get("host"), "job": r.get("slurm_job"), "file": r.get("file")}
        rank = (_int(r.get("workers"), 0), _int(r.get("slurm_job"), -1), _name(r))
        cands.setdefault((k, nodes), []).append((rank, rec))
    table = []
    for key in sorted(cands):
        ranked = sorted(cands[key], key=lambda c: c[0])
        keep = ranked[-1][1]
        if len(ranked) > 1:
            where = f"{x}={key[0]}" + ("" if x == "nodes" else f" nodes={key[1]}")
            warn(f"{where}: {len(ranked)} files, kept {keep['file']} (most workers, then latest job); ignored "
                 + ", ".join(str(c[1]["file"]) for c in ranked[:-1]))
        table.append(keep)
    if not table:
        return table
    x0, t0 = table[0][x], table[0]["throughput"]
    for t in table:
        t["speedup"] = round(t["throughput"] / t0, 3)
        t["efficiency"] = round(t["speedup"] / (t[x] / x0), 3)
    return table


def write_csv(table: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)


def plot(table: list[dict], path: Path, title: str, x: str = "workers") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[int, list[dict]] = {}          # one line per node count: a 2-node point is its own series
    for t in table:
        series.setdefault(1 if x == "nodes" else int(t.get("nodes") or 1), []).append(t)
    labelled = len(series) > 1
    xs = sorted({t[x] for t in table})
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, (nodes, pts) in enumerate(sorted(series.items())):
        fmt, lab = ("o-" if i == 0 else "s--"), f"{nodes} node{'' if nodes == 1 else 's'}"
        px = [p[x] for p in pts]
        ax[0].plot(px, [p["throughput"] for p in pts], fmt, label=lab)
        ax[1].plot(px, [p["speedup"] for p in pts], fmt, label=f"measured ({lab})" if labelled else "measured")
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log", base=2)
    ax[0].set_xlabel(x); ax[0].set_ylabel("throughput (items / min)"); ax[0].set_title("throughput")
    ax[0].grid(True, which="both", alpha=0.3)
    if labelled:
        ax[0].legend()
    ax[1].plot(xs, [xi / xs[0] for xi in xs], "--", color="grey", label="ideal")
    for t in table:
        ax[1].annotate(f"{t['efficiency']:.0%}", (t[x], t["speedup"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log", base=2)
    ax[1].set_xlabel(x); ax[1].set_ylabel(f"speed-up vs {xs[0]}"); ax[1].set_title("strong scaling (labels = efficiency)")
    ax[1].legend(); ax[1].grid(True, which="both", alpha=0.3)
    fig.suptitle(title); fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Strong-scaling CSV + plot from benchmark stats files.")
    ap.add_argument("--glob", required=True, action="append", help="stats files; repeat to combine jobs")
    ap.add_argument("--x", default="workers",
                    help="the scaled resource: workers, gpus, nodes (= nodes, else slurm_nodes, else 1), ...")
    ap.add_argument("--metric", default=None, help="throughput key in the files (default: 60*n/wall_s)")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--outdir", type=Path, default=Path("docs/scaling"))
    a = ap.parse_args(argv)
    rows = load(a.glob)
    if not rows:
        print("no stats files matched", " ".join(a.glob))
        return 1
    table = scaling_table(rows, a.x, a.metric)
    if not table:
        print(f"no stats file has a usable {a.x!r}" + (f" and {a.metric!r}" if a.metric else ""))
        return 1
    write_csv(table, a.outdir / f"scaling_{a.tag}.csv")
    plot(table, a.outdir / f"scaling_{a.tag}.png", f"strong scaling ({a.tag})", a.x)
    multi = a.x != "nodes" and len({t["nodes"] for t in table}) > 1
    for t in table:
        nodes = f" nodes={t['nodes']}" if multi else ""
        print(f"{a.x}={t[a.x]:4d}{nodes} throughput={t['throughput']:10.2f}/min speedup={t['speedup']:.2f} eff={t['efficiency']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
