"""Static hygiene checks on the SLURM job scripts and shell helpers. Each rule is a lesson from a failed job
(docs/kit/LESSONS.md) or a project rule (CLAUDE.md §1, §4): logs go to scratch, every job goes through the
budget guard, there is no git and no credential on the cluster, and GPU jobs name the env they run in."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOBS = sorted((ROOT / "jobs").glob("*.sbatch"))
SHELL = sorted((ROOT / "scripts").glob("*.sh"))


def _code(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


def _kit_env() -> dict:
    return dict(re.findall(r"^([A-Z_]+)=(\S*)", (ROOT / "kit.env").read_text(), re.M))


@pytest.mark.parametrize("job", JOBS, ids=[j.name for j in JOBS])
def test_job_script_hygiene(job):
    s = job.read_text()
    assert s.startswith("#!/bin/bash")
    assert "set -euo pipefail" in s
    assert re.search(r"^#SBATCH --job-name=lofi-[a-z0-9-]+$", s, re.M), "named for the ledger (budget.yaml job_name_prefix)"
    assert re.search(r"Cost: .*SU", s), "the header states the job's worst-case SU cost"
    assert "make submit" in s, "the header shows how to submit (through the budget guard)"
    assert "scripts/job_env.sh" in s or "scripts/_kit_env.sh" in s, "paths come from kit.env"
    assert not re.search(r"\bgit\b", _code(s)), "no git on the cluster (CLAUDE.md §1)"
    assert not re.search(r"(HF_TOKEN|hf auth login|OPENAI_API_KEY|GEMINI_API_KEY|ANTHROPIC_API_KEY)", s), "no credentials on the cluster"
    if re.search(r"^#SBATCH --(gres|gpus)", s, re.M):
        assert re.search(r"MODE=(policy|scorer) (source|\.) scripts/job_env.sh", s), "GPU jobs name their env"


def test_job_log_dirs_agree_with_kit_env():
    # #SBATCH lines cannot expand variables, so each job spells out the scratch log dir; it must be the one
    # kit.env describes (cluster_setup.sh init creates it; watch_job.sh reads <name>_<id>.out from it)
    v = _kit_env()
    want = f"{v['KIT_CLUSTER_SCRATCH_ROOT'].replace(v['KIT_CLUSTER_USER'], '%u')}/{v['KIT_PROJECT']}/logs/%x_%j.out"
    for job in JOBS:
        m = re.search(r"^#SBATCH --output=(\S+)", job.read_text(), re.M)
        assert m and m.group(1) == want, f"{job.name}: --output={m.group(1) if m else None}, kit.env implies {want}"


def test_guard_is_the_only_sbatch_caller():
    offenders = []
    for p in list((ROOT / "scripts").glob("*.sh")) + JOBS:
        if p.name == "sbatch_guard.sh":
            continue
        for line in p.read_text().splitlines():
            # a bare `sbatch ` command; "x.sbatch " (a job file name in a usage string) is not one
            if re.search(r"(^|[^_a-zA-Z/.])sbatch\s", line) and "sbatch_guard" not in line and not line.strip().startswith("#"):
                offenders.append(f"{p.name}: {line.strip()}")
    assert not offenders, offenders


@pytest.mark.parametrize("sh", SHELL, ids=[p.name for p in SHELL])
def test_shell_syntax(sh):
    assert subprocess.run(["bash", "-n", str(sh)], capture_output=True).returncode == 0


def test_job_env_has_offline_cache_explicit_interpreter_and_no_git():
    s = (ROOT / "scripts" / "job_env.sh").read_text()
    assert "HF_HUB_OFFLINE" in s and 'PY="$_ENV/bin/python"' in s and "env -u LD_LIBRARY_PATH" in s
    assert "PYTHONPATH=" in s, "the project is imported from src/, never installed into an env"
    assert not re.search(r"\bgit\b", _code(s)), "no git on the cluster"


def test_kit_env_is_literal_so_make_and_bash_read_it_the_same():
    for line in (ROOT / "kit.env").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            assert re.fullmatch(r"[A-Z_]+=[^\s$#\"']+", line), f"not a literal KEY=value line: {line!r}"


def test_env_builds_sync_a_lock_and_end_with_an_import_check():
    for name, lock in (("env_policy.sbatch", "policy"), ("env_scorer.sbatch", "scorer")):
        code = _code((ROOT / "jobs" / name).read_text())
        assert f"requirements/{lock}.lock.txt" in code and "pip sync" in code, name
        assert "walk_packages" in code and "IMPORTS_OK" in code, name
        assert "timeout" in code, f"{name}: every build step has a timeout (LESSONS 4)"
        assert (ROOT / "requirements" / f"{lock}.lock.txt").exists(), lock


def test_fluidsynth_is_pinned_and_wrapped():
    s = (ROOT / "jobs" / "tools.sbatch").read_text()
    assert "fluidsynth=2.6.1" in s and "env -u LD_LIBRARY_PATH" in s and '!= "2.6.1"' not in s
    assert '[ "$FS_VER" = "2.6.1" ]' in s
