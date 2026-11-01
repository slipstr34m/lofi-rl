"""jobs/junk_rate.sbatch (lofi-junk-rate): sized under 500 SU worst case, its stated cost matches its resources,
and its three steps run in the right env, each under a timeout that fits the walltime. Its arguments, N SEED
CONSTRAINED PROMPT, reach generate as given (bash runs the job's own argument lines here), and a bad CONSTRAINED or
PROMPT stops it before any work. tests/test_job_scripts.py applies the hygiene rules every job script follows (name,
log path, Cost line, make submit, job_env, no git)."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

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


def _args_block(s: str) -> str:
    """The job's argument lines, from N= to the end of the last argument check (PROMPT's) before the env is sourced,
    to run in bash on the laptop."""
    lines = s.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("N="))
    env = next(i for i, line in enumerate(lines) if "scripts/job_env.sh" in line and not line.startswith("#"))
    end = max(i for i, line in enumerate(lines[:env]) if line.strip() == "esac")
    return "\n".join(lines[start:end + 1])


@pytest.mark.parametrize("args, flag", [((), ""), (("500", "2026"), ""), (("500", "2026", "0"), ""),
                                        (("500", "2026", "1"), "--constrained")])
def test_the_third_arg_constrained_reaches_generate_as_its_flag(args, flag):
    s = JOB.read_text()
    assert re.search(r"^CONSTRAINED=\$\{3:-0\}$", s, re.M), "default 0: v0's unconstrained run"
    gen = next(line for line in _code(s) if "-m lofi_rl.rl.junk generate" in line)
    assert gen.rstrip().endswith("$CONSTRAIN_FLAG || FAIL=1"), "passed through to generate"
    script = "set -euo pipefail\n" + _args_block(s) + '\nprintf "%s|%s|%s" "$N" "$SEED" "$CONSTRAIN_FLAG"\n'
    out = subprocess.run(["bash", "-c", script, "job", *args], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert out.stdout == f"{args[0] if args else 500}|{args[1] if args else junk.DEFAULT_SEED}|{flag}"


@pytest.mark.parametrize("bad", ["2", "yes", "true", "-1", "01"])
def test_a_constrained_arg_other_than_0_or_1_stops_the_job_before_any_work(bad):
    script = "set -euo pipefail\n" + _args_block(JOB.read_text()) + '\necho REACHED\n'
    out = subprocess.run(["bash", "-c", script, "job", "500", "2026", bad], capture_output=True, text=True, timeout=30)
    assert out.returncode == 2 and "REACHED" not in out.stdout and "CONSTRAINED must be 0 or 1" in out.stderr


def test_the_header_states_the_third_arg_and_how_to_submit_a_constrained_run():
    s = JOB.read_text()
    assert "Args: [N=500] [SEED=2026] [CONSTRAINED=0]" in s
    assert "make submit JOB=jobs/junk_rate.sbatch ARGS='500 2026 1'" in s


@pytest.mark.parametrize("args, version", [((), "v1"), (("500", "2026"), "v1"), (("500", "2026", "1"), "v1"),
                                           (("500", "2026", "0", "v1"), "v1"), (("500", "2026", "1", "v2"), "v2"),
                                           (("20", "7", "0", "v2"), "v2")])
def test_the_fourth_arg_prompt_reaches_generate_as_its_flag(args, version):
    s = JOB.read_text()
    assert re.search(r"^PROMPT=\$\{4:-v1\}$", s, re.M), "default v1: the prompt v0 ran"
    gen = next(line for line in _code(s) if "-m lofi_rl.rl.junk generate" in line)
    assert '--prompt "$PROMPT" $CONSTRAIN_FLAG || FAIL=1' in gen, "passed through to generate"
    assert sum("--prompt" in line for line in _code(s)) == 1, "only generate takes it; score reads generate.json"
    script = "set -euo pipefail\n" + _args_block(s) + '\nprintf "%s|%s|%s|%s" "$N" "$SEED" "$CONSTRAIN_FLAG" "$PROMPT"\n'
    out = subprocess.run(["bash", "-c", script, "job", *args], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    flag = "--constrained" if len(args) > 2 and args[2] == "1" else ""
    assert out.stdout == f"{args[0] if args else 500}|{args[1] if args else junk.DEFAULT_SEED}|{flag}|{version}"
    assert version in junk.PROMPTS and junk.DEFAULT_PROMPT == "v1"
    echo = next(line for line in _code(s) if line.startswith('echo "junk-rate: N='))
    assert "PROMPT=$PROMPT" in echo, "the job log shows the prompt version"


@pytest.mark.parametrize("bad", ["v3", "V2", "2", "v0", "prompt_v2", "v1,v2"])
def test_a_prompt_arg_other_than_v1_or_v2_stops_the_job_before_any_work(bad):
    script = "set -euo pipefail\n" + _args_block(JOB.read_text()) + '\necho REACHED\n'
    out = subprocess.run(["bash", "-c", script, "job", "500", "2026", "1", bad], capture_output=True, text=True,
                         timeout=30)
    assert out.returncode == 2 and "REACHED" not in out.stdout
    assert f"PROMPT must be v1 or v2, not '{bad}'" in out.stderr


def test_the_jobs_prompt_versions_are_junks():
    s = JOB.read_text()
    m = re.search(r'^case "\$PROMPT" in\n  ([a-z0-9|]+)\) ;;$', s, re.M)
    assert m and tuple(m[1].split("|")) == tuple(junk.PROMPTS), "the job takes exactly the versions generate takes"


def test_the_header_states_the_fourth_arg_and_how_to_submit_a_prompt_v2_run():
    s = JOB.read_text()
    assert "Args: [N=500] [SEED=2026] [CONSTRAINED=0] [PROMPT=v1]" in s
    assert "make submit JOB=jobs/junk_rate.sbatch ARGS='500 2026 1 v2'" in s
    assert "src/lofi_rl/prompt_v2.py" in s
