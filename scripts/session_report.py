#!/usr/bin/env python
"""Print the SU ledger for the CHANGELOG from sacct output (run ON the cluster via remote.sh):

    scripts/remote.sh python scripts/session_report.py            # since project_start_date
    scripts/remote.sh python scripts/session_report.py --since 2026-11-01
    scripts/remote.sh python scripts/session_report.py --all      # every job of $USER on the account

Uses the same formula and the same scope as sbatch_guard.sh: SU = max(billing, cpu + GB/4 + 8*GPU) x
minutes, over jobs whose name starts with budget.yaml's job_name_prefix (this project's jobs; other
projects of the same user on the same account are left out). sacct is the ledger: myaccount drifts
between calls, sshare decays, sreport lags.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

W = {"a100": 8, "a40": 8, "l40s": 8, "v100": 4, "p100": 4}


def load_budget(path: Path) -> dict:
    """budget.yaml is flat 'key: value' lines; parsed with the stdlib so this runs on the login node's python3."""
    out: dict = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().strip('"').strip("'")
        out[k.strip()] = int(v) if v.isdigit() else v
    return out


def su_per_min(tres: str) -> float:
    f = dict(kv.split("=", 1) for kv in tres.split(",") if "=" in kv)
    billing = float(f.get("billing") or 0)
    cpu = float(f.get("cpu") or 0)
    mem_gb = 0.0
    if f.get("mem"):
        m = re.match(r"([\d.]+)([KMGT]?)", f["mem"])
        mem_gb = float(m.group(1)) * {"K": 1 / 1048576, "M": 1 / 1024, "G": 1, "T": 1024, "": 1 / 1073741824}[m.group(2)]
    gpu = sum(float(v) * W.get(k.split(":", 1)[1].lower(), 8) for k, v in f.items() if k.startswith("gres/gpu:"))
    return max(billing, cpu + mem_gb / 4 + gpu)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    ap.add_argument("--all", action="store_true", help="ignore job_name_prefix: every job of $USER on the account")
    ap.add_argument("--budget", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "budget.yaml")
    a = ap.parse_args(argv)
    b = load_budget(a.budget)
    since = a.since or str(b["project_start_date"])
    prefix = "" if a.all else str(b.get("job_name_prefix") or "")
    out = subprocess.run(["sacct", "-u", os.environ["USER"], "-A", b["account"], "-S", since, "-X", "-n", "-P",
                          "--format=JobID,JobName,Partition,State,ElapsedRaw,AllocTRES,NodeList"], capture_output=True, text=True, check=True).stdout
    total = 0.0
    print(f"{'JobID':>9} {'name':18} {'part':6} {'state':10} {'min':>7} {'SU/min':>7} {'SU':>9}  node")
    for line in out.splitlines():
        jid, name, part, state, el, tres, node = line.split("|")
        if prefix and not name.startswith(prefix):
            continue
        rate = su_per_min(tres)
        su = rate * int(el or 0) / 60
        total += su
        print(f"{jid:>9} {name[:18]:18} {part:6} {state[:10]:10} {int(el or 0)/60:7.1f} {rate:7.1f} {su:9.1f}  {node}")
    scope = f"jobs named {prefix}*" if prefix else "all jobs of $USER on the account"
    print(f"TOTAL {total:.1f} SU since {since} for {scope}  (cap {b['max_project_sus']}, {100*total/b['max_project_sus']:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
