"""jobs/step0.sbatch (lofi-step0; docs/design/step0_samples.md §2.1, §2.7, §2.8, §6.1), as tests/test_junk_job.py
checks junk's job: its stated cost matches its resources and its registration covers it; it never requeues; its three
steps run in the policy, policy and scorer envs, each under a timeout, with deadlines under the timeouts and the stack
dump under generate's; the timeouts fit the walltime with a minute spare; the run is written on the node's disk and
copied to KEEP once, records first, after the render and from an EXIT trap that TERM reaches; a used OUT is refused
before the traps are set; OUT and the one-shot cache are on KEEP; nothing goes to the inbox; any argument is refused.

The script also runs here, end to end, in bash, with stand-ins for scripts/job_env.sh, `timeout` and the three steps
(each logs its env and arguments and writes a file of the run), so its argument check, its OUT check, its copies and
its traps are seen working, not only read. tests/test_job_scripts.py applies the rules every job script follows.

The registry check (test_the_registration_covers_the_worst_case) fails until the job is registered: the registration
waits for Shresth's answers to the spec's §7 and comes after the script's last change (§6.2 step 5).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import lofi_rl.step0 as S
from lofi_rl.rl import junk as J

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs" / "step0.sbatch"
SU_PER_MIN = {"cpu": 1, "gb": 0.25, "gpu": 8}      # CARC: 1 CPU, 4 GB and 1 A40/A100/L40S GPU per minute
COST = "Cost: 4 CPU + 6 (24 GB) + 8 (1 GPU) = 18 SU/min x 45 min -> at most 810 SU."
STEPS = ("generate", "score", "render")


def _sbatch(text: str) -> dict[str, str]:
    return dict(re.findall(r"^#SBATCH --([a-z-]+)=(\S+)$", text, re.M))


def _minutes(t: str) -> float:
    h, m, s = (int(x) for x in t.split(":"))
    return 60 * h + m + s / 60


def _code(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _worst() -> float:
    sb = _sbatch(JOB.read_text())
    rate = SU_PER_MIN["cpu"] * int(sb["cpus-per-task"]) + SU_PER_MIN["gb"] * int(sb["mem"].removesuffix("G")) \
        + SU_PER_MIN["gpu"]
    return rate * _minutes(sb["time"])


# --------------------------------------------------------------------------------------------- shape and cost
def test_the_job_states_its_cost_and_its_shape():
    s = JOB.read_text()
    sb = _sbatch(s)
    assert sb == {"job-name": "lofi-step0", "partition": "gpu", "nodes": "1", "ntasks": "1", "cpus-per-task": "4",
                  "mem": "24G", "gres": "gpu:a40:1", "time": "00:45:00",
                  "output": "/scratch1/%u/lofi-rl/logs/%x_%j.out"}, "junk's shape on gpu, 45 min (spec §2.1)"
    assert re.search(r"^#SBATCH --no-requeue$", s, re.M), "a node failure never reruns the job into its own OUT"
    assert [line for line in s.splitlines() if line.startswith("#SBATCH")][-1] == "#SBATCH --no-requeue"
    assert COST in s and _worst() == 810
    m = re.search(r"Cost: (\d+) CPU \+ (\d+) \((\d+) GB\) \+ 8 \(1 GPU\) = (\d+) SU/min x (\d+) min "
                  r"-> at most (\d+) SU", s)
    cpus, mem = int(sb["cpus-per-task"]), int(sb["mem"].removesuffix("G"))
    assert (int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6])) == (
        cpus, mem // 4, mem, 18, _minutes(sb["time"]), _worst())
    assert "make submit JOB=jobs/step0.sbatch" in s and "OVR=" not in s and "ARGS=" not in s
    assert sb["job-name"] == S.JOB_NAME and JOB == ROOT / S.JOB


def test_the_worst_case_is_one_claude_may_register():
    limit = float(re.search(r"^self_approve_max_su: (\d+)", (ROOT / "config" / "budget.yaml").read_text(), re.M)[1])
    assert _worst() <= limit == 1000, "at most 1,000 SU: Claude registers it (D-003 as amended 2026-11-06)"


def test_the_registration_covers_the_worst_case():
    """Fails until `make approve JOB=jobs/step0.sbatch MAX=810 BY=Claude` (spec §6.2 step 5), after Shresth's
    answers to the spec's §7."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import approve_job
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    reg = re.search(r"^jobs/step0\.sbatch: \{sha: (\w+), max_su: (\d+), by: (\w+), date: [\d-]+\}$",
                    (ROOT / "jobs" / "APPROVED.yaml").read_text(), re.M)
    assert reg, "jobs/step0.sbatch is not registered in jobs/APPROVED.yaml yet"
    assert reg[1] == approve_job.job_sha(JOB), "the script changed since its registration: re-register it"
    assert _worst() <= int(reg[2]) <= 1000 and reg[3] == "Claude"


# --------------------------------------------------------------------------------------------- the steps
def _steps(code: list[str]) -> list[tuple[int, str]]:
    return [(i, re.search(r"-m lofi_rl\.step0 (\w+)", line).group(1)) for i, line in enumerate(code)
            if "-m lofi_rl.step0" in line]


def test_generate_score_and_render_run_in_policy_policy_and_scorer_under_timeouts():
    s = JOB.read_text()
    code = _code(s)
    steps = _steps(code)
    assert [name for _, name in steps] == list(STEPS)
    modes = [(i, re.search(r"MODE=(\w+)", line).group(1)) for i, line in enumerate(code)
             if "scripts/job_env.sh" in line]
    assert [[m for i, m in modes if i < at][-1] for at, _ in steps] == ["policy", "policy", "scorer"]
    budget, seen = 0, {}
    for i, name in steps:
        m = re.search(r"^timeout -k (\d+) (\d+) \"\$PY\" -m lofi_rl\.step0 ", code[i])
        assert m and code[i].rstrip().endswith("|| FAIL=1"), f"{name} runs under a timeout"
        budget += int(m[1]) + int(m[2])
        d = re.search(r"--deadline-s (\d+)", code[i])
        seen[name] = (int(m[1]), int(m[2]), int(d[1]) if d else None)
    assert seen == {"generate": (15, 1080, 960), "score": (10, 420, None), "render": (10, 960, 900)}, "spec §2.8"
    assert seen["generate"][2] == S.GENERATE_DEADLINE_S and seen["render"][2] == S.RENDER_DEADLINE_S
    assert seen["generate"][2] + J.HANG_MARGIN_S < seen["generate"][1], "a hang dumps its stacks before the kill"
    assert seen["render"][1] - seen["render"][2] == 60, "a render started at the deadline ends before the timeout"
    copies = [line for line in code if ' cp -a "$RUN/' in line]
    for line in copies:
        m = re.search(r"timeout -k (\d+) (\d+) cp -a ", line)
        assert m, "each copy out runs under a timeout"
        budget += int(m[1]) + int(m[2])
    assert budget == 2589 and budget <= 60 * _minutes(_sbatch(s)["time"]) - 60, \
        "the steps' timeouts fit the walltime with at least a minute spare (2,589 of 2,700 s)"


def test_the_steps_take_only_the_run_and_their_deadlines():
    """The model, prompt, palette, streams and sampling are registered (lofi_rl.step0): no step is given one."""
    code = _code(JOB.read_text())
    lines = {name: code[i] for i, name in _steps(code)}
    assert lines["generate"].split(" -m lofi_rl.step0 ")[1] == 'generate --run "$RUN" --deadline-s 960 || FAIL=1'
    assert lines["score"].split(" -m lofi_rl.step0 ")[1] == 'score --run "$RUN" || FAIL=1'
    assert lines["render"].split(" -m lofi_rl.step0 ")[1] == (
        'render --run "$RUN" --workers "${SLURM_CPUS_PER_TASK:-4}" --deadline-s 900 || FAIL=1')
    joined = "\n".join(code)
    for flag in ("--model", "--prompt", "--palette", "--seed", "--constrained", "--n ", "--chunk", "--from", "--k "):
        assert flag not in joined, flag


def test_the_render_step_runs_two_lanes_on_the_jobs_cpus():
    cpus = int(_sbatch(JOB.read_text())["cpus-per-task"])
    assert cpus == S.RENDER_WORKERS == 4 and J.render_lanes(J.RENDER_JOBS, cpus) == (2, 2)
    assert not any("--jobs" in line for line in _code(JOB.read_text()))


def test_the_run_is_copied_once_records_first_after_the_render_and_from_the_traps():
    code = _code(JOB.read_text())
    steps = _steps(code)
    copies = [line for line in code if ' cp -a "$RUN/' in line]
    assert len(copies) == 2 and '"$RUN/summary"' in copies[0] and '"$RUN/clips"' in copies[1], "records first"
    assert "timeout -k 2 30 cp -a" in copies[0] and "timeout -k 2 60 cp -a" in copies[1]
    calls = [i for i, line in enumerate(code) if line.startswith("copy_out ||")]
    assert len(calls) == 1 and calls[0] > steps[-1][0], "the copy out runs after the render"
    assert any(line.startswith('[ "$COPIED" = 1 ] && return 0') for line in map(str.strip, code)), "once"
    traps = [i for i, line in enumerate(code) if line.startswith("trap ")]
    assert any("copy_out" in code[i] and code[i].rstrip().endswith("EXIT") for i in traps)
    assert any(re.fullmatch(r"trap 'exit 143' TERM(\s+#.*)?", code[i]) for i in traps)
    assert max(traps) < steps[0][0], "the traps are set before any step runs"


def test_a_used_out_is_refused_before_the_traps_are_set():
    code = _code(JOB.read_text())
    check = next(i for i, line in enumerate(code) if line.startswith('if [ -e "$OUT" ]'))
    assert code[check + 2].strip() == "exit 2"
    first_trap = min(i for i, line in enumerate(code) if line.startswith("trap "))
    mkdir = next(i for i, line in enumerate(code) if line.startswith("mkdir -p"))
    assert check < mkdir < first_trap, "refused before OUT is made and before an EXIT trap could copy into it"


def test_the_run_is_on_the_nodes_disk_and_out_and_the_cache_on_keep():
    code = "\n".join(_code(JOB.read_text()))
    assert 'RUN="${TMPDIR:-/tmp}/step0-${SLURM_JOB_ID:-local}"' in code
    assert 'OUT="$REMOTE_KEEP/runs/step0-${SLURM_JOB_ID:-local}"' in code
    assert 'export LOFI_CACHE="$REMOTE_KEEP/cache/lofi-rl"' in code
    assert "$REMOTE_SCR" not in code and "inbox" not in code, "nothing of the run on scratch, nothing to the inbox"
    assert code.count('--run "$RUN"') == 3 and "STEP0_OK" in code


# --------------------------------------------------------------------------------------------- the script, run
JOB_ENV = """# a stand-in for scripts/job_env.sh: KEEP and the interpreter are the test's
REMOTE_KEEP="$TEST_KEEP"
PY="$TEST_BIN/fakepy"
export STEP_MODE="${MODE:-scorer}"
model_rev() { echo cdbee75f; }
"""
FAKE_TIMEOUT = """#!/bin/bash
# a stand-in for coreutils timeout: logs its limits, then runs the command in its place
echo "timeout $1 $2 $3" >> "$TEST_LOG.timeouts"
shift 3
exec "$@"
"""
FAKE_PY = """#!/bin/bash
# a stand-in for the env's python: `-m lofi_rl.step0 <step> --run RUN ...` logs and writes a file of the run
step="$3"; run="$5"
echo "$STEP_MODE $step ${*:4}" >> "$TEST_LOG"
case "$step" in
  generate) echo '{"step": "generate"}' > "$run/summary/generate.json" ;;
  score) echo '{"step": "score"}' > "$run/summary/step0.json" ;;
  render) mkdir -p "$run/clips"; echo fLaC > "$run/clips/0000.flac"; echo '{}' > "$run/summary/clips.json" ;;
esac
if [ "${FAKE_TERM:-}" = "$step" ]; then kill -TERM "$PPID"; fi
if [ "${FAKE_FAIL:-}" = "$step" ]; then exit 1; fi
exit 0
"""


@pytest.fixture
def world(tmp_path):
    """The job's code tree (the real script, a stand-in job_env.sh), KEEP, the node's disk and the stand-in tools."""
    code, keep, node, bin_ = (tmp_path / d for d in ("code", "keep", "node", "bin"))
    for d in (code / "jobs", code / "scripts", keep, node, bin_):
        d.mkdir(parents=True)
    shutil.copy(JOB, code / "jobs" / "step0.sbatch")
    (code / "scripts" / "job_env.sh").write_text(JOB_ENV)
    for name, body in (("timeout", FAKE_TIMEOUT), ("fakepy", FAKE_PY)):
        (bin_ / name).write_text(body)
        (bin_ / name).chmod(0o755)
    env = {"PATH": f"{bin_}:{os.environ['PATH']}", "HOME": str(tmp_path), "SLURM_SUBMIT_DIR": str(code),
           "SLURM_JOB_ID": "4242", "SLURM_CPUS_PER_TASK": "4", "TMPDIR": str(node), "TEST_KEEP": str(keep),
           "TEST_BIN": str(bin_), "TEST_LOG": str(tmp_path / "steps.log")}

    def run(*args: str, **extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(code / "jobs" / "step0.sbatch"), *args], capture_output=True, text=True,
                              env={**env, **extra}, timeout=60, cwd=tmp_path)
    run.keep, run.node, run.log = keep, node, tmp_path / "steps.log"
    run.out, run.run = keep / "runs" / "step0-4242", node / "step0-4242"
    return run


def _log(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.is_file() else []


def _files(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def test_the_script_runs_its_three_steps_and_copies_the_run_out(world):
    p = world()
    assert p.returncode == 0 and p.stdout.rstrip().endswith("STEP0_OK"), p.stderr
    assert _log(world.log) == [f"policy generate --run {world.run} --deadline-s 960", f"policy score --run {world.run}",
                               f"scorer render --run {world.run} --workers 4 --deadline-s 900"]
    assert _log(Path(f"{world.log}.timeouts")) == ["timeout -k 15 1080", "timeout -k 10 420", "timeout -k 10 960",
                                                   "timeout -k 2 30", "timeout -k 2 60"]
    assert _files(world.out) == ["clips/0000.flac", "summary/clips.json", "summary/generate.json",
                                 "summary/step0.json"]
    assert f"-> {world.out} (written at {world.run})" in p.stdout and "refs/main=cdbee75f" in p.stdout


@pytest.mark.parametrize("args", [("1",), ("500", "2026"), ("--dry-run",), ("",)])
def test_any_argument_stops_the_job_before_any_work(world, args):
    p = world(*args)
    assert p.returncode == 2 and "the job takes no argument" in p.stderr
    assert _log(world.log) == [] and not world.out.exists() and not world.run.exists()


def test_a_used_out_is_refused_and_never_merged_into(world):
    world.out.mkdir(parents=True)
    (world.out / "earlier.txt").write_text("an earlier attempt's file\n")
    p = world()
    assert p.returncode == 2 and "exists and is not empty" in p.stderr
    assert _log(world.log) == [] and _files(world.out) == ["earlier.txt"], "nothing copied in: no EXIT trap was set"
    shutil.rmtree(world.out)
    world.out.write_text("a file where OUT should be\n")
    assert world().returncode == 2 and _log(world.log) == []
    world.out.unlink()
    world.out.mkdir()                                   # an empty OUT is a fresh one
    assert world().returncode == 0 and "summary/generate.json" in _files(world.out)


def test_a_failed_step_still_copies_the_records_out(world):
    p = world(FAKE_FAIL="render")
    assert p.returncode == 1 and "STEP0_FAILED" in p.stdout
    assert [line.split()[1] for line in _log(world.log)] == list(STEPS), "every step runs; each fails on its own"
    assert "summary/step0.json" in _files(world.out) and "clips/0000.flac" in _files(world.out)


def test_term_exits_143_through_the_exit_trap_which_copies_the_run_out(world):
    p = world(FAKE_TERM="score")
    assert p.returncode == 143, p.stderr
    assert [line.split()[1] for line in _log(world.log)] == ["generate", "score"], "no step after the TERM"
    assert _files(world.out) == ["summary/generate.json", "summary/step0.json"], "the records reach KEEP"
    assert "STEP0_OK" not in p.stdout
