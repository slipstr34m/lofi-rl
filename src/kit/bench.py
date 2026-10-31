"""Strong-scaling benchmark harness for embarrassingly parallel work (the CSCI 596 "concepts applied" core).

    python -m kit.bench --task mypkg.tasks:work --n 1000 --workers 64 --out $KIT_CLUSTER_SCRATCH/bench/<job>/w64

`--task module:function` is called as function(i) for i in range(n) by a multiprocessing Pool of
`--workers` processes (the function is imported once per worker). stats.json records n, workers,
wall_s, throughput_per_min, mean_task_s and the SLURM job/node info; scripts/scaling_study.py turns a
set of them into a CSV + plot. Run the sweep inside ONE allocation (jobs/bench_scaling.sbatch) so every
point sees the same node. On Discovery "CPUs" are hardware threads: efficiency drops past the physical
core count (measured 2026-10-25: 94 % at 8 workers, 74 % at 32, 49 % at 64 threads on an EPYC 7542).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

_FN = None


def load_task(spec: str):
    mod, _, fn = spec.partition(":")
    if not fn:
        raise ValueError(f"--task must be module:function, got {spec!r}")
    return getattr(importlib.import_module(mod), fn)


def _init(spec: str) -> None:
    global _FN
    _FN = load_task(spec)


def _one(i: int) -> float:
    t0 = time.time()
    _FN(i)
    return time.time() - t0


def demo_task(i: int) -> int:
    """A small CPU-bound task for smoke tests and for checking a node's scaling."""
    s = 0
    for k in range(200_000 + (i % 7) * 1000):
        s = (s + k * k) % 1_000_003
    return s


def run(task: str, n: int, workers: int, out: Path, chunksize: int = 4) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(processes=workers, initializer=_init, initargs=(task,)) as pool:
        per = pool.map(_one, range(n), chunksize=chunksize)
    wall = time.time() - t0
    stats = {
        "task": task, "n": n, "workers": workers, "wall_s": round(wall, 3),
        "throughput_per_min": round(60 * n / wall, 2) if wall > 0 else None,
        "mean_task_s": round(sum(per) / len(per), 4) if per else None,
        "host": os.uname().nodename, "cpu_count": os.cpu_count(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="kit.bench:demo_task", help="module:function called as f(i)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunksize", type=int, default=4)
    a = ap.parse_args(argv)
    print(json.dumps(run(a.task, a.n, a.workers, a.out, a.chunksize)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
