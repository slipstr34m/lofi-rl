"""jobs/junk_rate.sbatch (lofi-junk-rate): sized under 500 SU worst case, its stated cost matches its resources,
and its three steps run in the right env, each under a timeout that fits the walltime. tests/test_job_scripts.py
applies the hygiene rules every job script follows (name, log path, Cost line, make submit, job_env, no git)."""
from __future__ import annotations

import re
from pathlib import Path

from lofi_rl.rl import junk

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs" / "junk_rate.sbatch"
SU_PER_MIN = {"cpu": 1, "gb": 0.25, "gpu": 8}      # CARC: 1 CPU, 4 GB and 1 A40/A100/L40S GPU per minute


def _sbatch(text: str) -> dict[str, str]:
    return dict(re.findall(r"^#SBATCH --([a-z-]+)=(\S+)$", text, re.M))


def _minutes(t: str) -> float:
    h, m, s = (int(x) for x in t.split(":"))
    return 60 * h + m + s / 60


def _code(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_junk_job_is_sized_under_500_su_and_states_its_cost():
    s = JOB.read_text()
    sb = _sbatch(s)
    assert sb["job-name"] == "lofi-junk-rate" and sb["partition"] in ("debug", "gpu")
    assert re.fullmatch(r"gpu(:[a-z0-9]+)?:1", sb["gres"]), "one GPU"
    cpus, mem = int(sb["cpus-per-task"]), int(sb["mem"].removesuffix("G"))
    rate = SU_PER_MIN["cpu"] * cpus + SU_PER_MIN["gb"] * mem + SU_PER_MIN["gpu"]
    worst = rate * _minutes(sb["time"])
    assert worst <= 500, "Claude registers a type up to 500 SU itself (budget.yaml self_approve_max_su)"
    m = re.search(r"Cost: (\d+) CPU \+ (\d+) \((\d+) GB\) \+ 8 \(1 GPU\) = (\d+) SU/min x (\d+) min -> at most (\d+) SU", s)
    assert m, "the header states the cost in the kit's form"
    assert (int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6])) == (
        cpus, mem // 4, mem, rate, _minutes(sb["time"]), worst)
    assert "make submit JOB=jobs/junk_rate.sbatch" in s
    assert "OVR='-p gpu --gres=gpu:a100:1 --constraint=a100-80gb'" in s


def test_junk_job_runs_generate_and_score_in_policy_then_render_in_scorer_under_timeouts():
    s = JOB.read_text()
    code = _code(s)
    steps = [(i, re.search(r"-m lofi_rl\.rl\.junk (\w+)", line).group(1)) for i, line in enumerate(code)
             if "-m lofi_rl.rl.junk" in line]
    assert [name for _, name in steps] == ["generate", "score", "render"]
    modes = [(i, re.search(r"MODE=(\w+)", line).group(1)) for i, line in enumerate(code) if "scripts/job_env.sh" in line]

    def mode_at(at: int) -> str:
        return [m for i, m in modes if i < at][-1]
    assert [mode_at(i) for i, _ in steps] == ["policy", "policy", "scorer"]
    budget = 0.0
    for i, name in steps:
        m = re.search(r"timeout -k (\d+) (\d+) ", code[i])
        assert m, f"{name} runs under a timeout (vLLM start-up has no deadline of its own)"
        budget += int(m[1]) + int(m[2])
        d = re.search(r"--deadline-s (\d+)", code[i])
        if name != "score":
            assert d and int(d[1]) < int(m[2]), f"{name} stops itself before its timeout"
        if name == "generate":
            assert int(d[1]) + junk.HANG_MARGIN_S < int(m[2]), "a hang dumps its stacks before the timeout kills it"
    assert budget <= 60 * _minutes(_sbatch(s)["time"]) - 60, "the steps' timeouts fit the walltime with a minute spare"


def test_junk_job_writes_one_run_dir_and_never_the_inbox():
    s = JOB.read_text()
    code = "\n".join(_code(s))
    assert 'RUN="$REMOTE_SCR/runs/junk-${SLURM_JOB_ID:-local}"' in code
    assert code.count('--run "$RUN"') == 3 and "inbox" not in code
    assert re.search(r"^N=\$\{1:-500\}$", s, re.M) and re.search(rf"^SEED=\$\{{2:-{junk.DEFAULT_SEED}\}}$", s, re.M)
    assert f"--k {junk.RENDER_K}" in code and "JUNK_RATE_OK" in code
