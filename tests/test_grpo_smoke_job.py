"""jobs/grpo_smoke.sbatch (lofi-grpo-smoke): its optional first argument names the config, one of the
configs/grpo_smoke*.yaml files (default configs/grpo_smoke.yaml, so a bare submit runs the smoke 13632679 ran), and
anything else stops it before any work (bash runs the job's own argument lines here, from the repo root as the job
does). The config reaches every step; the worst case stays at 500 SU. tests/test_job_scripts.py applies the hygiene
rules every job script follows."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from lofi_rl.rl.config import load_config

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs" / "grpo_smoke.sbatch"
CONFIGS = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs").glob("grpo_smoke*.yaml"))
SU_PER_MIN = {"cpu": 1, "gb": 0.25, "gpu": 8}      # CARC: 1 CPU, 4 GB and 1 GPU per minute


def _sbatch(text: str) -> dict[str, str]:
    return dict(re.findall(r"^#SBATCH --([a-z-]+)=(\S+)$", text, re.M))


def _code(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _args_block() -> str:
    """The job's argument lines: from CFG= to the end of the config check, before the env is sourced."""
    lines = JOB.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("CFG="))
    env = next(i for i, line in enumerate(lines) if "scripts/job_env.sh" in line and not line.startswith("#"))
    return "\n".join(lines[start:env])


def _run(*args: str) -> subprocess.CompletedProcess:
    script = "set -euo pipefail\n" + _args_block() + '\nprintf "%s" "$CFG"\n'
    return subprocess.run(["bash", "-c", script, "job", *args], capture_output=True, text=True, timeout=30, cwd=ROOT)


def test_the_job_is_sized_at_500_su_and_states_its_cost():
    s = JOB.read_text()
    sb = _sbatch(s)
    assert sb["job-name"] == "lofi-grpo-smoke" and sb["gres"] == "gpu:a100:1" and sb["time"] == "00:25:00"
    h, m, sec = (int(x) for x in sb["time"].split(":"))
    rate = SU_PER_MIN["cpu"] * int(sb["cpus-per-task"]) + SU_PER_MIN["gb"] * int(sb["mem"].removesuffix("G")) + 8
    assert rate == 20 and rate * (60 * h + m + sec / 60) == 500, "Claude registers a type up to 500 SU"
    assert "Cost: 4 CPU + 8 (32 GB) + 8 (1 A100) = 20 SU/min -> at most 500 SU (25 min)" in s


def test_a_bare_submit_runs_todays_smoke_config():
    out = _run()
    assert out.returncode == 0 and out.stdout == "configs/grpo_smoke.yaml", out.stderr
    assert re.search(r"^CFG=\$\{1:-configs/grpo_smoke\.yaml\}$", JOB.read_text(), re.M)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_every_smoke_config_is_taken_and_loads(cfg):
    out = _run(cfg)
    assert out.returncode == 0 and out.stdout == cfg, out.stderr
    load_config(ROOT / cfg)


def test_the_v2_grammar_config_is_one_of_them():
    assert {"configs/grpo_smoke.yaml", "configs/grpo_smoke_constrained.yaml",
            "configs/grpo_smoke_v2_grammar.yaml"} <= set(CONFIGS)


@pytest.mark.parametrize("bad", ["configs/grpo_smoke_missing.yaml", "configs/rating.yaml", "grpo_smoke.yaml",
                                 "configs/../configs/grpo_smoke.yaml", str(ROOT / "configs" / "grpo_smoke.yaml"),
                                 "configs/grpo_smoke*.yaml", "configs/grpo_smoke.yml", "v2"])
def test_any_other_config_stops_the_job_before_any_work(bad):
    out = _run(bad)
    assert out.returncode == 2 and out.stdout == "" and f"not '{bad}'" in out.stderr
    assert "configs/grpo_smoke_v2_grammar.yaml" in out.stderr, "the message lists the configs it takes"


def test_a_second_word_stops_the_job():
    out = _run("configs/grpo_smoke.yaml", "extra")
    assert out.returncode == 2 and "at most one argument" in out.stderr


def test_the_config_reaches_the_dry_run_both_launches_and_both_checks():
    code = _code(JOB.read_text())
    uses = [line for line in code if '--config "$CFG"' in line]
    assert len(uses) == 5, uses
    assert sum("lofi_rl.rl.train" in line for line in uses) == 3 and sum("smoke_check" in line for line in uses) == 2
    assert not any("configs/grpo_smoke.yaml" in line for line in code if not line.startswith("CFG=")), \
        "no step names a config of its own"
    assert any(line.startswith('echo "grpo-smoke: CONFIG=$CFG"') for line in code), "the log names the config"


def test_the_header_states_the_argument_and_the_submit_lines():
    s = JOB.read_text()
    assert "Args: [CONFIG=configs/grpo_smoke.yaml]" in s
    assert "make submit JOB=jobs/grpo_smoke.sbatch\n" in s
    assert "make submit JOB=jobs/grpo_smoke.sbatch ARGS='configs/grpo_smoke_v2_grammar.yaml'" in s
