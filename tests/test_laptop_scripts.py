"""Laptop-side scripts against a stub ssh that runs the remote command locally:
watch_job.sh never reports a failed squeue as "left the queue", shows the estimated start of a pending
job and reads <name>_<id>.out logs; remote.sh runs in the mirror, prints the mirror's stamp, handles
commands that end in a heredoc, a comment or '&', and runs nothing when the mirror is missing."""
import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _exec(p: Path, body: str):
    p.write_text(body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)


def _kit_env(proj: Path, mirror: Path):
    (proj / "kit.env").write_text(
        "KIT_PROJECT=demo\nKIT_BRANCH=main\nKIT_CLUSTER_HOST=fakehost\nKIT_CLUSTER_FQDN=localhost\nKIT_CLUSTER_USER=u\n"
        f"KIT_ACCOUNT=acct\nKIT_CLUSTER_PATH={mirror}\nREMOTE_PROJ={mirror}\nKIT_CLUSTER_SCRATCH={proj.parent}/scratch\n")


def _bins(tmp_path):
    bins = tmp_path / "bin"
    bins.mkdir(exist_ok=True)
    _exec(bins / "ssh", '#!/bin/bash\nfor last; do :; done\nexec bash -c "$last"\n')
    return bins


# ---------------- watch_job.sh ----------------
def _watch(tmp_path, squeue_body, *args):
    proj = tmp_path / "proj"
    (proj / "scripts").mkdir(parents=True)
    for f in ("_kit_env.sh", "_cluster_ssh.sh", "watch_job.sh"):
        shutil.copy(ROOT / "scripts" / f, proj / "scripts" / f)
    _kit_env(proj, tmp_path / "mirror")
    bins = _bins(tmp_path)
    _exec(bins / "squeue", "#!/bin/bash\n" + squeue_body)
    _exec(bins / "sacct", '#!/bin/bash\ncase " $* " in *" --format=JobName "*) echo demo-x;;\n'
                          '  *) echo "4242|demo-x|RUNNING|60|billing=2|d05-01|0:0";; esac\n')
    env = dict(os.environ, PATH=f"{bins}:{os.environ['PATH']}", COUNT=str(tmp_path / "count"))
    return subprocess.run(["bash", str(proj / "scripts" / "watch_job.sh"), "4242", "0", *args],
                          capture_output=True, text=True, env=env, timeout=60)


def test_watch_squeue_error_is_a_failed_poll_not_completion(tmp_path):
    p = _watch(tmp_path, 'echo "slurm_load_jobs error: Socket timed out on send/recv operation" >&2; exit 1\n', "5")
    assert "left the queue" not in p.stdout, p.stdout
    assert "squeue failed (exit 1): slurm_load_jobs error: Socket timed out" in p.stdout
    assert p.returncode == 2 and "may still be running" in p.stdout


def test_watch_invalid_job_id_means_it_left(tmp_path):
    p = _watch(tmp_path, 'echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1\n', "5")
    assert "left the queue (squeue no longer knows it)" in p.stdout and p.returncode == 0, p.stdout


def test_watch_shows_estimated_start_then_completion(tmp_path):
    body = ('n=$(cat "$COUNT" 2>/dev/null || echo 0); echo $((n+1)) > "$COUNT"\n'
            '[ "$n" -lt 2 ] && echo "PENDING 0:00 (Priority) start=2026-10-26T21:56:00"\nexit 0\n')
    (tmp_path / "scratch" / "logs").mkdir(parents=True)
    (tmp_path / "scratch" / "logs" / "demo-x_4242.out").write_text("step 1\n\nRESULT_OK\n")
    p = _watch(tmp_path, body, "10")
    assert "PENDING 0:00 (Priority) start=2026-10-26T21:56:00" in p.stdout
    assert "job 4242 left the queue" in p.stdout and "still in the queue" not in p.stdout
    assert "### " in p.stdout and p.stdout.rstrip().endswith("RESULT_OK") and p.returncode == 0, p.stdout


def test_watch_says_when_it_ran_out_of_polls(tmp_path):
    p = _watch(tmp_path, 'echo "RUNNING 5:00 d05-01 start=2026-10-26T13:00:00"\n', "2")
    assert "still in the queue after 2 polls" in p.stdout and "left the queue" not in p.stdout
    assert p.returncode == 1 and "4242  demo-x  RUNNING" in p.stdout and "no log at" in p.stdout


# ---------------- remote.sh ----------------
def _remote_setup(tmp_path):
    laptop = tmp_path / "laptop"
    (laptop / "scripts").mkdir(parents=True)
    for f in ("_kit_env.sh", "_cluster_ssh.sh", "remote.sh", "stamp.sh"):
        shutil.copy(ROOT / "scripts" / f, laptop / "scripts" / f)
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    (mirror / ".sync-stamp").write_text("commit=abc123def456 dirty=0 diff=- at=2026-10-30T00:00:00Z\n")
    _kit_env(laptop, mirror)
    env = dict(os.environ, PATH=f"{_bins(tmp_path)}:{os.environ['PATH']}")
    return laptop, mirror, env


def _remote(laptop, env, cmd):
    return subprocess.run(["bash", str(laptop / "scripts" / "remote.sh"), cmd], capture_output=True, text=True,
                          env=env, cwd=laptop, timeout=60)


def test_remote_runs_heredocs_comments_and_background_jobs(tmp_path):
    laptop, mirror, env = _remote_setup(tmp_path)
    p = _remote(laptop, env, 'python3 - <<"PYEOF"\nprint("from heredoc")\nPYEOF')
    assert p.returncode == 0 and "from heredoc" in p.stdout, p.stdout + p.stderr
    assert "[remote] mirror=commit=abc123def456 dirty=0" in p.stdout
    p = _remote(laptop, env, "echo with-comment   # a trailing note")
    assert p.returncode == 0 and "with-comment" in p.stdout, p.stderr
    p = _remote(laptop, env, "sleep 0 &")
    assert p.returncode == 0, p.stderr
    p = _remote(laptop, env, "echo a; echo b")
    assert p.stdout.count("\n") >= 3 and "a\nb" in p.stdout


def test_remote_runs_in_the_mirror_and_runs_nothing_without_it(tmp_path):
    laptop, mirror, env = _remote_setup(tmp_path)
    (mirror / "here.txt").write_text("x")
    p = _remote(laptop, env, "cat here.txt")
    assert p.returncode == 0 and p.stdout.strip().endswith("x"), p.stdout + p.stderr
    shutil.rmtree(mirror)
    p = _remote(laptop, env, "echo should-not-run; echo also")
    assert p.returncode != 0 and "should-not-run" not in p.stdout and "also" not in p.stdout


def test_remote_needs_a_command(tmp_path):
    laptop, _, env = _remote_setup(tmp_path)
    p = subprocess.run(["bash", str(laptop / "scripts" / "remote.sh")], capture_output=True, text=True, env=env)
    assert p.returncode == 2 and "usage" in p.stderr
