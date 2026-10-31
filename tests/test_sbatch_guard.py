"""Tests for scripts/sbatch_guard.sh with stubbed sacct / squeue / sbatch (CLAUDE.md §3.5).

The SU formula under test is CARC's: SUs = (CPUs + GB/4 + 8*GPUs[A100/A40/L40S] + 4*GPUs[V100/P100]) x minutes.
Reference examples from CARC's Running Jobs guide:
  1 CPU + 8 GB for 1 h = 180 SU;   8 CPU + 320 GB + 8 A100 for 1 h = 9120 SU.
The guard must FAIL CLOSED (refuse) when sacct or squeue fails, and must count the remaining
reservation of every pending/running job against the cap.
"""
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / "scripts" / "sbatch_guard.sh"


def _write_exec(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def harness(tmp_path):
    """run(script_text, sacct_rows=(), active=(), budget=None, *cli, fail=None) -> CompletedProcess

    sacct_rows: (AllocTRES, ElapsedRaw) for the 'used' query.
    active: (State, ReqTRES, AllocTRES, TimelimitRaw, ElapsedRaw) for each pending/running job.
    fail: 'sacct_used' | 'sacct_active' | 'squeue' makes that stub exit 1.
    """
    bins = tmp_path / "bin"
    bins.mkdir()
    submitted = tmp_path / "submitted.txt"
    sacct_args = tmp_path / "sacct_args.txt"
    _write_exec(bins / "sacct", f'''#!/bin/sh
echo "$@" >> "{sacct_args}"
case " $* " in
  *" -j "*) [ "$STUB_FAIL" = sacct_active ] && exit 1; cat "$STUB_SACCT_ACTIVE";;
  *)        [ "$STUB_FAIL" = sacct_used ] && exit 1; cat "$STUB_SACCT";;
esac
''')
    _write_exec(bins / "squeue", '#!/bin/sh\n[ "$STUB_FAIL" = squeue ] && exit 1\ncat "$STUB_SQUEUE"\n')
    _write_exec(bins / "sbatch", f'#!/bin/sh\necho "$@" > "{submitted}"\necho "Submitted batch job 424242"\n')

    def run(script_text, sacct_rows=(), active=(), budget=None, *cli, fail="", env_extra=None):
        b = dict(max_project_sus=250000, max_concurrent_jobs=8, max_gpu_nodes_per_job=2,
                 max_job_hours=47.5, warn_at_fraction=0.7, project_start_date="2026-10-25",
                 account="anakano_429")
        b.update(budget or {})
        budget_file = tmp_path / "budget.yaml"
        budget_file.write_text("".join(f"{k}: {v}\n" for k, v in b.items()))
        (tmp_path / "sacct.txt").write_text("".join(f"{tres}|{elapsed}\n" for tres, elapsed in sacct_rows))
        (tmp_path / "squeue.txt").write_text("".join(f"{1000 + i}\n" for i in range(len(active))))
        (tmp_path / "sacct_active.txt").write_text("".join(
            f"{1000 + i}|{st}|{req}|{alloc}|{lim}|{el}\n" for i, (st, req, alloc, lim, el) in enumerate(active)))
        script = tmp_path / "job.sbatch"
        script.write_text(script_text)
        for f in (submitted, sacct_args):
            if f.exists():
                f.unlink()
        env = dict(os.environ, BUDGET_FILE=str(budget_file), GUARD_LOG=str(tmp_path / "submissions.log"),
                   SACCT_BIN=str(bins / "sacct"), SQUEUE_BIN=str(bins / "squeue"), SBATCH_BIN=str(bins / "sbatch"),
                   STUB_SACCT=str(tmp_path / "sacct.txt"), STUB_SACCT_ACTIVE=str(tmp_path / "sacct_active.txt"),
                   STUB_SQUEUE=str(tmp_path / "squeue.txt"), STUB_FAIL=fail, USER="tester", TMPDIR=str(tmp_path), **(env_extra or {}))
        proc = subprocess.run(["bash", str(GUARD), *cli, str(script)], env=env, capture_output=True, text=True,
                              cwd=tmp_path)
        proc.submitted = submitted.read_text().strip() if submitted.exists() else None
        proc.sacct_args = sacct_args.read_text() if sacct_args.exists() else ""
        return proc

    return run


GPU_NODE_JOB = """#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=0
#SBATCH --gres=gpu:a100:2 --constraint=a100-80gb
#SBATCH --time=01:00:00
echo hi
"""
SMALL = "#!/bin/bash\n#SBATCH -t 00:01:00\n"


def test_carc_worked_examples_for_used_sus(harness):
    rows = [("billing=3,cpu=1,mem=8G,node=1", 3600),                      # 180 SU
            ("billing=152,cpu=8,mem=320G,node=1,gres/gpu:a100=8", 3600)]  # 9120 SU
    p = harness(SMALL, rows)
    assert p.returncode == 0, p.stderr
    assert "used=9300.0 SU" in p.stdout
    assert "-S 2026-10-25" in p.sacct_args and "-A anakano_429" in p.sacct_args and "-u tester" in p.sacct_args


def test_recompute_is_conservative_when_billing_missing(harness):
    # No billing= field: 64 cpu + 64 (256 GB) + 16 (2 A100) = 144 SU/min for 60 min = 8640
    p = harness(SMALL, [("cpu=64,mem=256G,node=1,gres/gpu:a100=2", 3600)])
    assert "used=8640.0 SU" in p.stdout


def test_projection_full_a100_node_one_hour(harness):
    p = harness(GPU_NODE_JOB)
    assert p.returncode == 0, p.stderr
    assert "projected_max=8640.0 SU" in p.stdout
    assert p.submitted.startswith("--account=anakano_429")


def test_refuses_when_cap_would_be_exceeded(harness):
    p = harness(GPU_NODE_JOB, [("billing=144,cpu=64,mem=256G,node=1,gres/gpu:a100=2", 3600 * 28)])
    assert p.returncode == 1
    assert "REFUSED" in p.stderr and "max_project_sus" in p.stderr
    assert p.submitted is None


@pytest.mark.parametrize("fail", ["sacct_used", "sacct_active", "squeue"])
def test_fails_closed_when_accounting_commands_fail(harness, fail):
    active = [("PENDING", "billing=6,cpu=4,mem=8G,node=1", "", "10", "0")]
    p = harness(SMALL, (), active, None, fail=fail)
    assert p.returncode == 1, p.stdout
    assert "REFUSED" in p.stderr and ("sacct failed" in p.stderr or "squeue failed" in p.stderr)
    assert p.submitted is None


def test_pending_and_running_reservations_count_against_the_cap(harness):
    # each queued full-A100-node job holds 144 SU/min x 1500 min = 216,000 SU of the 250,000 cap
    big = ("PENDING", "billing=144,cpu=64,gres/gpu=2,mem=256G,node=1", "", "1500", "0")
    p = harness(GPU_NODE_JOB, (), [big])
    assert p.returncode == 0, p.stderr
    assert "reserved_by_active=216000.0 SU (1 jobs)" in p.stdout
    p = harness(GPU_NODE_JOB, (), [big, big])
    assert p.returncode == 1 and "max_project_sus" in p.stderr and p.submitted is None
    # a RUNNING job only reserves what is left of its limit: 10 min left of 60 at 6 SU/min = 60
    run = ("RUNNING", "billing=6,cpu=4,mem=8G,node=1", "billing=6,cpu=4,mem=8G,node=1", "60", "3000")
    p = harness(SMALL, (), [run])
    assert "reserved_by_active=60.0 SU" in p.stdout and p.returncode == 0


def test_refuses_at_concurrency_limit(harness):
    active = [("PENDING", "billing=1,cpu=1,node=1", "", "1", "0")] * 8
    p = harness("#!/bin/bash\n#SBATCH -t 00:10:00\n", (), active)
    assert p.returncode == 1 and "max_concurrent_jobs" in p.stderr and p.submitted is None


def test_refuses_too_many_gpu_nodes_including_ranges(harness):
    p = harness(GPU_NODE_JOB.replace("--nodes=1", "--nodes=3"))
    assert p.returncode == 1 and "max_gpu_nodes_per_job" in p.stderr and p.submitted is None
    p = harness(GPU_NODE_JOB.replace("--nodes=1", "--nodes=1-3"))
    assert p.returncode == 1 and "3 GPU nodes" in p.stderr and p.submitted is None
    p = harness(GPU_NODE_JOB.replace("--nodes=1", "--nodes=1-2"))
    assert p.returncode == 0 and "projected_max=17280.0 SU (2 node" in p.stdout


@pytest.mark.parametrize("spec, hours, ok", [
    ("30", 0.5, True), ("30:00", 0.5, True), ("2:00:00", 2.0, True), ("1-12", 36.0, True),
    ("1-12:30", 36.5, True), ("1-23:59", 47.9833, False), ("2-00:00:00", 48.0, False), ("47:30:00", 47.5, True),
])
def test_slurm_time_formats(harness, spec, hours, ok):
    p = harness(f"#!/bin/bash\n#SBATCH -t {spec}\n")
    assert f"x {hours:.4f} h" in p.stdout, p.stdout + p.stderr
    assert (p.returncode == 0) == ok
    if not ok:
        assert "max_job_hours" in p.stderr


def test_unlimited_time_is_refused(harness):
    p = harness("#!/bin/bash\n#SBATCH -t UNLIMITED\n")
    assert p.returncode == 1 and "unlimited" in p.stderr


def test_cli_override_wins_and_is_forwarded(harness):
    p = harness(GPU_NODE_JOB, (), (), None, "--time=3-00:00:00")
    assert p.returncode == 1 and "max_job_hours" in p.stderr
    p = harness(GPU_NODE_JOB, (), (), None, "-t", "00:30:00")
    assert p.returncode == 0, p.stderr
    assert "-t 00:30:00" in p.submitted
    assert "projected_max=4320.0 SU" in p.stdout


def test_unknown_value_option_is_not_taken_for_the_script(harness, tmp_path):
    decoy = tmp_path / "decoy.txt"
    decoy.write_text("not a job")
    p = harness(SMALL, (), (), None, "--some-new-option", str(decoy), "--output", "x.log")
    assert p.returncode == 0, p.stderr
    assert p.submitted.endswith("job.sbatch") and str(decoy) in p.submitted and "--output x.log" in p.submitted


def test_gpus_per_task_syntax_and_v100_weight(harness):
    s = "#!/bin/bash\n#SBATCH -p gpu -N 1 -n 1 -c 8 --mem=32G --gpus-per-task=v100:1 -t 01:00:00\n"
    p = harness(s)
    assert p.returncode == 0, p.stderr
    # 8 cpu + 8 (32 GB) + 4 (one V100) = 20 SU/min -> 1200 SU/h
    assert "projected_max=1200.0 SU" in p.stdout


@pytest.mark.parametrize("opts, mem_gb", [
    ("-c 4 --mem=0G", 256), ("-c 4 --mem=0", 256), ("-c 4 --mem=0M", 256),
    ("-c 16 --mem=0 --gres=gpu:l40s:4", 768), ("-c 8 --mem-per-cpu=4G", 32), ("-c 4 --mem=8192", 8),
    ("-c 4 --mem=1T", 1024), ("-c 4", 8),
])
def test_memory_specs(harness, opts, mem_gb):
    p = harness(f"#!/bin/bash\n#SBATCH {opts} -t 00:10:00\n")
    assert f"mem/node={mem_gb}" in p.stdout, p.stdout + p.stderr


def test_warns_past_fraction(harness):
    p = harness(SMALL, [("billing=1,cpu=1,node=1", 60 * 200000)])
    assert p.returncode == 0 and "WARNING" in p.stderr


def test_logs_submission(harness, tmp_path):
    p = harness("#!/bin/bash\n#SBATCH -t 00:05:00\n")
    assert p.returncode == 0
    log = (tmp_path / "submissions.log").read_text()
    assert "424242" in log and "projected=" in log and "reserved=" in log


def test_no_script_is_refused(harness, tmp_path):
    proc = subprocess.run(["bash", str(GUARD), "-t", "00:05:00"], capture_output=True, text=True,
                          env=dict(os.environ, BUDGET_FILE=str(REPO / "config" / "budget.yaml"),
                                   SQUEUE_BIN="true", SACCT_BIN="true", SBATCH_BIN="false"))
    assert proc.returncode == 1 and "no job script" in proc.stderr


# ---------- arrays, --exclusive, the per-project ledger, log directories (kit review 2026-10-26) ----------
SMALL_HOUR = "#!/bin/bash\n#SBATCH -p main -c 1 --mem=4G -t 01:00:00\n"   # 1 + 1 = 2 SU/min -> 120 SU


@pytest.mark.parametrize("spec, tasks", [
    ("0-9", 10), ("0-9%2", 10), ("1,3,5-7", 5), ("0-15:4", 4), ("0-9:2%3", 5), ("7", 1),
])
def test_array_projection_counts_every_task(harness, spec, tasks):
    p = harness(SMALL_HOUR, (), (), {"max_concurrent_jobs": 64}, f"--array={spec}")
    assert p.returncode == 0, p.stderr
    assert f"projected_max={120.0 * tasks:.1f} SU" in p.stdout and f"x {tasks} task(s)" in p.stdout
    p = harness(SMALL_HOUR, (), (), {"max_concurrent_jobs": 64}, "-a", spec)
    assert f"projected_max={120.0 * tasks:.1f} SU" in p.stdout


def test_array_directive_and_cap(harness):
    # ten 47-hour full-A100-node tasks are ~4M SU: refused even though one task fits the cap
    job = GPU_NODE_JOB.replace("--time=01:00:00", "--time=47:00:00 --array=0-9%2")
    p = harness(job, (), (), {"max_concurrent_jobs": 64})
    assert p.returncode == 1 and "max_project_sus" in p.stderr and p.submitted is None
    assert "x 10 task(s)" in p.stdout


def test_array_tasks_count_toward_concurrency(harness):
    p = harness(SMALL_HOUR, (), (), None, "--array=0-9")           # 10 tasks > 8
    assert p.returncode == 1 and "max_concurrent_jobs" in p.stderr and p.submitted is None
    p = harness(SMALL_HOUR, (), (), None, "--array=0-7")           # 8 tasks fit exactly
    assert p.returncode == 0, p.stderr


def test_bad_array_spec_is_refused(harness):
    for spec in ("9-0", "a-b", "0-9:0", ""):
        p = harness(SMALL_HOUR, (), (), None, f"--array={spec}") if spec else harness(SMALL_HOUR, (), (), None, "--array", "%4")
        assert p.returncode == 1 and "REFUSED" in p.stderr and p.submitted is None, spec


def _rerun(tmp_path):
    """Re-run the guard on the harness's last job script after a test rewrote the stub files."""
    b = tmp_path / "bin"
    env = dict(os.environ, BUDGET_FILE=str(tmp_path / "budget.yaml"), GUARD_LOG=str(tmp_path / "s.log"),
               SACCT_BIN=str(b / "sacct"), SQUEUE_BIN=str(b / "squeue"), SBATCH_BIN=str(b / "sbatch"),
               STUB_SACCT=str(tmp_path / "sacct.txt"), STUB_SACCT_ACTIVE=str(tmp_path / "sacct_active.txt"),
               STUB_SQUEUE=str(tmp_path / "squeue.txt"), STUB_FAIL="", USER="tester", TMPDIR=str(tmp_path))
    return subprocess.run(["bash", str(GUARD), str(tmp_path / "job.sbatch")], capture_output=True, text=True,
                          cwd=tmp_path, env=env)


def test_queued_array_range_is_expanded(harness, tmp_path):
    # squeue folds pending tasks into one line; sacct -X returns one PENDING row for the range
    harness(SMALL_HOUR)
    (tmp_path / "squeue.txt").write_text("5000_[2-99%8]|x\n5000_0|x\n5000_1|x\n")
    (tmp_path / "sacct_active.txt").write_text(
        "5000_[2-99%8]|PENDING|billing=2,cpu=1,mem=4G,node=1||60|0\n"
        "5000_0|RUNNING|billing=2,cpu=1,mem=4G,node=1|billing=2,cpu=1,mem=4G,node=1|60|0\n"
        "5000_1|RUNNING|billing=2,cpu=1,mem=4G,node=1|billing=2,cpu=1,mem=4G,node=1|60|0\n")
    run = _rerun(tmp_path)
    # 100 tasks x 2 SU/min x 60 min = 12,000 SU reserved; 100 active jobs -> refused on concurrency
    assert "reserved_by_active=12000.0 SU (100 jobs)" in run.stdout, run.stdout + run.stderr
    assert run.returncode == 1 and "max_concurrent_jobs" in run.stderr


@pytest.mark.parametrize("opts, cpus, mem, gpus", [
    ("-p main -c 8 --mem=16G --exclusive", 64, 256, 0),            # 64 + 64 = 128 SU/min
    ("-p gpu -c 8 --mem=32G --gres=gpu:a100:1 --exclusive", 64, 256, 2),
    ("-p gpu -c 8 --gres=gpu:l40s:1 --exclusive=user", 64, 768, 4),
    ("-p gpu -c 8 --gres=gpu:1 --exclusive", 64, 768, 4),           # open GPU type: the biggest node
    ("-p largemem -c 8 --exclusive", 64, 1536, 0),
])
def test_exclusive_costs_a_whole_node(harness, opts, cpus, mem, gpus):
    p = harness(f"#!/bin/bash\n#SBATCH {opts} -t 01:00:00\n")
    assert p.returncode == 0, p.stderr
    assert f"cpus/node={cpus} mem/node={mem}" in p.stdout and f"gpus/node={gpus}" in p.stdout and "exclusive" in p.stdout
    rate = cpus + mem * 0.25 + gpus * 8
    assert f"projected_max={rate * 60:.1f} SU" in p.stdout


def test_job_name_prefix_scopes_the_ledger_and_is_enforced(harness):
    budget = {"job_name_prefix": "demo-"}
    # rows are TRES|ElapsedRaw|JobName: only demo-* counts (10 SU); the other project's 100 SU does not
    used = [("billing=1,cpu=1,node=1", "6000|old-proj-job"), ("billing=1,cpu=1,node=1", "600|demo-smoke")]
    p = harness("#!/bin/bash\n#SBATCH -J demo-x -t 00:10:00\n", used, (), budget)
    assert p.returncode == 0, p.stderr
    assert "used=10.0 SU" in p.stdout
    assert "--format=AllocTRES,ElapsedRaw,JobName" in p.sacct_args
    p = harness("#!/bin/bash\n#SBATCH -J demo-x -t 00:10:00\n", used, (), None)     # no prefix: everything
    assert "used=110.0 SU" in p.stdout
    p = harness("#!/bin/bash\n#SBATCH -t 00:10:00\n", (), (), budget)              # no name -> script name
    assert p.returncode == 1 and "job_name_prefix" in p.stderr and p.submitted is None
    p = harness("#!/bin/bash\n#SBATCH --job-name=other-x -t 00:10:00\n", (), (), budget)
    assert p.returncode == 1 and "does not start with 'demo-'" in p.stderr
    p = harness("#!/bin/bash\n#SBATCH -t 00:10:00\n", (), (), budget, "--job-name=demo-cli")
    assert p.returncode == 0, p.stderr


def test_active_jobs_of_other_projects_are_not_ours(harness, tmp_path):
    harness("#!/bin/bash\n#SBATCH -J demo-x -t 00:10:00\n", (), (), {"job_name_prefix": "demo-", "max_concurrent_jobs": 1})
    (tmp_path / "squeue.txt").write_text("1000|other-job\n1001|demo-run\n")
    (tmp_path / "sacct_active.txt").write_text("1001|PENDING|billing=1,cpu=1,node=1||10|0\n")
    run = _rerun(tmp_path)
    assert "reserved_by_active=10.0 SU (1 jobs)" in run.stdout, run.stdout + run.stderr
    assert run.returncode == 1 and "max_concurrent_jobs=1" in run.stderr      # demo-run counts, other-job does not
    (tmp_path / "squeue.txt").write_text("1000|other-job\n")
    run = _rerun(tmp_path)
    assert run.returncode == 0 and "(0 jobs)" in run.stdout, run.stdout + run.stderr


def test_missing_log_directory_is_created_or_refused(harness, tmp_path):
    logdir = tmp_path / "scratch" / "tester" / "demo" / "logs"
    p = harness(f"#!/bin/bash\n#SBATCH -t 00:10:00\n#SBATCH --output={tmp_path}/scratch/%u/demo/logs/%x-%j.out\n")
    assert p.returncode == 0, p.stderr
    assert logdir.is_dir(), "the guard creates the --output directory (with %u expanded) before submitting"
    blocker = tmp_path / "file"
    blocker.write_text("x")
    p = harness(f"#!/bin/bash\n#SBATCH -t 00:10:00\n#SBATCH --output={blocker}/logs/%x-%j.out\n")
    assert p.returncode == 1 and "die without a log" in p.stderr and p.submitted is None


# ---------------- job-type registry (jobs/APPROVED.yaml) and the soft SU target (CLAUDE.md §4) ----------------
import sys  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
import approve_job  # noqa: E402

REG_JOB = "#!/bin/bash\n#SBATCH --job-name=lofi-x\n#SBATCH -t 00:10:00 -c 1 --mem=4G\n# a comment\necho hi\n"   # 2 SU/min x 10 = 20 SU


def _registry(tmp_path, sha=None, max_su=100, by="Claude"):
    job = tmp_path / "job.sbatch"
    job.write_text(REG_JOB)
    rel = os.path.relpath(os.path.realpath(job), os.path.realpath(REPO))
    reg = tmp_path / "APPROVED.yaml"
    reg.write_text(f"# header\n{rel}: {{sha: {sha or approve_job.job_sha(job)}, max_su: {max_su}, by: {by}, date: 2026-10-30}}\n")
    return {"approvals_file": str(reg), "job_name_prefix": "lofi-"}


def test_unregistered_job_type_needs_approval(harness, tmp_path):
    reg = tmp_path / "APPROVED.yaml"
    reg.write_text("# empty registry\n")
    p = harness(REG_JOB, (), (), {"approvals_file": str(reg), "job_name_prefix": "lofi-"})
    assert p.returncode == 4 and "NEEDS_APPROVAL new" in p.stderr and p.submitted is None, p.stderr


def test_changed_job_needs_reapproval(harness, tmp_path):
    p = harness(REG_JOB, (), (), _registry(tmp_path, sha="0" * 16))
    assert p.returncode == 4 and "NEEDS_APPROVAL changed" in p.stderr and p.submitted is None, p.stderr


def test_projection_above_max_su_needs_approval(harness, tmp_path):
    p = harness(REG_JOB, (), (), _registry(tmp_path, max_su=10))
    assert p.returncode == 4 and "NEEDS_APPROVAL over-estimate" in p.stderr and p.submitted is None, p.stderr


def test_registered_job_is_submitted_and_comments_keep_the_approval(harness, tmp_path):
    budget = _registry(tmp_path)
    p = harness(REG_JOB, (), (), budget)
    assert p.returncode == 0 and "registry" in p.stdout and p.submitted, p.stderr
    p = harness(REG_JOB.replace("# a comment", "# a different comment"), (), (), budget)
    assert p.returncode == 0 and p.submitted, "editing comments must not void the approval"


def test_soft_target_needs_an_ok_but_the_hard_cap_never_moves(harness, tmp_path):
    rows = [("billing=1,cpu=1,mem=4G,node=1", "36000|lofi-old")]     # a lofi- job: 2 SU/min x 600 min = 1,200 SU used
    budget = dict(_registry(tmp_path), target_sus=500, max_project_sus=10000)
    p = harness(REG_JOB, rows, (), budget)
    assert p.returncode == 4 and "NEEDS_APPROVAL past-target" in p.stderr and p.submitted is None, p.stderr
    p = harness(REG_JOB, rows, (), budget, env_extra={"PAST_TARGET_OK": "2026-11-20"})
    assert p.returncode == 0 and p.submitted, p.stderr
    p = harness(REG_JOB, rows, (), dict(budget, max_project_sus=610), env_extra={"PAST_TARGET_OK": "2026-11-20"})
    assert p.returncode == 1 and "exceed max_project_sus" in p.stderr and p.submitted is None
