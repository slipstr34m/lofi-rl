"""scripts/session_report.py: the CHANGELOG's SU ledger uses the guard's formula and scope (jobs named
<job_name_prefix>*), so another project of the same user on the same account is not billed here."""
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, rows, *args, prefix='"demo-"'):
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    (tmp_path / "rows.txt").write_text("".join(r + "\n" for r in rows))
    sacct = b / "sacct"
    sacct.write_text(f'#!/bin/sh\necho "$@" > "{tmp_path}/args.txt"\ncat "{tmp_path}/rows.txt"\n')
    sacct.chmod(sacct.stat().st_mode | stat.S_IXUSR)
    budget = tmp_path / "budget.yaml"
    budget.write_text(f'max_project_sus: 1000\nproject_start_date: "2026-10-26"\naccount: "acct"\njob_name_prefix: {prefix}\n')
    env = dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", USER="tester")
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "session_report.py"), "--budget", str(budget), *args],
                          capture_output=True, text=True, env=env)


ROWS = ["101|demo-smoke|debug|COMPLETED|600|billing=2,cpu=1,mem=4G,node=1|d05-01",      # 2 SU/min x 10 min = 20
        "102|old-fetch|main|COMPLETED|6000|billing=12,cpu=4,mem=32G,node=1|d05-02"]     # another project: 1,200


def test_ledger_counts_only_this_projects_jobs(tmp_path):
    p = _run(tmp_path, ROWS)
    assert p.returncode == 0, p.stderr
    assert "TOTAL 20.0 SU since 2026-10-26 for jobs named demo-*" in p.stdout
    assert "old-fetch" not in p.stdout
    assert "-u tester -A acct -S 2026-10-26" in (tmp_path / "args.txt").read_text()


def test_ledger_all_and_no_prefix_count_everything(tmp_path):
    assert "TOTAL 1220.0 SU" in _run(tmp_path, ROWS, "--all").stdout
    assert "TOTAL 1220.0 SU" in _run(tmp_path, ROWS, prefix='""').stdout
