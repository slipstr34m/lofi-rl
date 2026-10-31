#!/usr/bin/env python3
"""Register or update a job type in jobs/APPROVED.yaml. The budget guard refuses any job type that is
unregistered, has changed since approval, or is projected above its max_su (CLAUDE.md §4).

    scripts/approve_job.py jobs/x.sbatch --max-su 720 --by Shresth [--date 2026-10-30]
    scripts/approve_job.py --check        # entries whose script changed since approval (re-approve them)

Claude may register a type itself (--by Claude) only up to budget.yaml's self_approve_max_su (500 SU).
Anything larger is written with --by Shresth, and only after Shresth said yes in chat.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "jobs" / "APPROVED.yaml"
BUDGET = ROOT / "config" / "budget.yaml"
ENTRY = re.compile(r"^(\S+):\s*\{(.*)\}\s*$")


def job_sha(path: Path) -> str:
    """sha256[:16] over #SBATCH lines and non-comment, non-blank lines (same rule as sbatch_guard.sh)."""
    keep = [l.rstrip() for l in path.read_text().splitlines(keepends=True)
            if l.startswith("#SBATCH") or (l.strip() and not l.lstrip().startswith("#"))]
    return hashlib.sha256("\n".join(keep).encode()).hexdigest()[:16]


def load(registry: Path = REGISTRY) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if registry.exists():
        for line in registry.read_text().splitlines():
            m = ENTRY.match(line.strip())
            if m:
                out[m.group(1)] = {k.strip(): v.strip() for k, v in (kv.split(":", 1) for kv in m.group(2).split(",") if ":" in kv)}
    return out


def self_approve_max(budget: Path = BUDGET) -> float:
    for line in budget.read_text().splitlines():
        if line.startswith("self_approve_max_su:"):
            return float(line.split(":", 1)[1].split("#")[0])
    return 0.0


def write(entries: dict[str, dict[str, str]], registry: Path = REGISTRY) -> None:
    head = ("# Job-type registry read by scripts/sbatch_guard.sh. One entry per line:\n"
            "#   <job path>: {sha: <sha256[:16] of #SBATCH + non-comment lines>, max_su: <n>, by: <who>, date: <yyyy-mm-dd>}\n"
            "# Entries above budget.yaml self_approve_max_su must say by: Shresth (his yes in chat).\n"
            "# Write entries with scripts/approve_job.py (or `make approve`), never by hand.\n")
    lines = [f"{k}: {{sha: {v['sha']}, max_su: {v['max_su']}, by: {v['by']}, date: {v['date']}}}" for k, v in sorted(entries.items())]
    registry.write_text(head + "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", nargs="?")
    ap.add_argument("--max-su", type=float)
    ap.add_argument("--by", choices=["Shresth", "Claude"])
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    entries = load()
    if a.check:
        stale = [k for k, v in entries.items() if not (ROOT / k).exists() or job_sha(ROOT / k) != v["sha"]]
        for k in stale:
            print(f"stale: {k} (missing or changed since approval)")
        return 1 if stale else 0
    if not (a.job and a.max_su is not None and a.by):
        ap.error("job, --max-su and --by are required (or use --check)")
    path = (ROOT / a.job).resolve()
    rel = str(path.relative_to(ROOT))
    if a.by == "Claude" and a.max_su > self_approve_max(BUDGET):
        print(f"refused: {rel} max_su={a.max_su:g} is above self_approve_max_su; it needs --by Shresth after his yes in chat",
              file=sys.stderr)
        return 2
    entries[rel] = {"sha": job_sha(path), "max_su": f"{a.max_su:g}", "by": a.by, "date": a.date}
    write(entries)
    print(f"{rel}: {entries[rel]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
