"""`make sync` and `make submit` safety nets: check_deletes.sh refuses protected or mass deletions, and
submit.sh refuses unless the mirror holds exactly the laptop's committed code."""
import os
import shutil
import stat
import subprocess
from pathlib import Path
import pytest

pytestmark = pytest.mark.laptop   # creates git repos: never on a compute node (CLAUDE.md §1)

ROOT = Path(__file__).resolve().parents[1]


def _deletes(lines, **env):
    return subprocess.run(["bash", str(ROOT / "scripts" / "check_deletes.sh")], input="".join(l + "\n" for l in lines),
                          capture_output=True, text=True, env=dict(os.environ, **env))


def test_ordinary_deletions_pass():
    p = _deletes([">f+++++++++ src/new.py", "*deleting   src/old.py", "*deleting   tests/old_test.py"])
    assert p.returncode == 0 and "2 deletion(s)" in p.stdout, p.stderr


def test_protected_paths_are_refused():
    for path in ("data/votes/2026-11-10.jsonl", ".venv-policy/bin/python", "logs/x_1.out", "jobs/submissions.log"):
        p = _deletes([f"*deleting   {path}"])
        assert p.returncode == 4 and path in p.stderr, path


def test_mass_deletion_is_refused_unless_overridden():
    lines = [f"*deleting   src/f{i}.py" for i in range(201)]
    assert _deletes(lines).returncode == 4
    assert _deletes(lines, MAX_DELETE="500").returncode == 0


def _git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=True,
                          env=dict(os.environ, GIT_AUTHOR_NAME="T", GIT_AUTHOR_EMAIL="t@e", GIT_COMMITTER_NAME="T",
                                   GIT_COMMITTER_EMAIL="t@e"))


def _submit_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for f in ("_kit_env.sh", "stamp.sh", "submit.sh"):
        shutil.copy(ROOT / "scripts" / f, repo / "scripts" / f)
    (repo / "kit.env").write_text("KIT_PROJECT=demo\nKIT_CLUSTER_USER=u\nKIT_ACCOUNT=a\nKIT_CLUSTER_PATH=/m\nREMOTE_KEEP=/keep\n")
    remote = repo / "scripts" / "remote.sh"   # stub: records what would run on the cluster
    remote.write_text('#!/bin/bash\necho "REMOTE: $*"\n')
    remote.chmod(remote.stat().st_mode | stat.S_IXUSR)
    (repo / "jobs").mkdir()
    (repo / "jobs" / "x.sbatch").write_text("#!/bin/bash\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _submit(repo, *args, **env):
    return subprocess.run(["bash", str(repo / "scripts" / "submit.sh"), *args], cwd=repo, capture_output=True, text=True,
                          env=dict(os.environ, **env))


def test_submit_needs_a_matching_sync_and_clean_tree(tmp_path):
    repo = _submit_repo(tmp_path)
    p = _submit(repo, "jobs/x.sbatch")
    assert p.returncode == 3 and "run make sync first" in p.stderr
    stamp = subprocess.run(["bash", str(repo / "scripts" / "stamp.sh")], capture_output=True, text=True).stdout
    (repo / ".sync-stamp").write_text(stamp)
    p = _submit(repo, "-t", "00:20:00", "jobs/x.sbatch", "arg one")
    assert p.returncode == 0, p.stderr
    assert "REMOTE: GUARD_LOG=/keep/submissions.log scripts/sbatch_guard.sh -t 00:20:00 jobs/x.sbatch arg\\ one" in p.stdout
    (repo / "jobs" / "x.sbatch").write_text("#!/bin/bash\necho changed\n")     # uncommitted edit
    p = _submit(repo, "jobs/x.sbatch")
    assert p.returncode == 3, "a dirty tree no longer matches the synced stamp"
    (repo / ".sync-stamp").write_text(subprocess.run(["bash", str(repo / "scripts" / "stamp.sh")], capture_output=True,
                                                     text=True).stdout)
    assert _submit(repo, "jobs/x.sbatch").returncode == 3 and "commit first" in _submit(repo, "jobs/x.sbatch").stderr
    assert _submit(repo, "jobs/x.sbatch", ALLOW_DIRTY="1").returncode == 0
