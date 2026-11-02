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
    (repo / ".gitignore").write_text(".sync-stamp\n.sync-manifest\n")   # as in the repo: else every stamp is dirty
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _submit(repo, *args, **env):
    clean = {k: v for k, v in os.environ.items() if k not in ("PAST_TARGET_OK", "ALLOW_DIRTY")}
    return subprocess.run(["bash", str(repo / "scripts" / "submit.sh"), *args], cwd=repo, capture_output=True, text=True,
                          env=dict(clean, **env))


def test_submit_needs_a_matching_sync_and_clean_tree(tmp_path):
    repo = _submit_repo(tmp_path)
    p = _submit(repo, "jobs/x.sbatch")
    assert p.returncode == 3 and "run make sync first" in p.stderr
    stamp = subprocess.run(["bash", str(repo / "scripts" / "stamp.sh")], capture_output=True, text=True).stdout
    (repo / ".sync-stamp").write_text(stamp)
    p = _submit(repo, "-t", "00:20:00", "jobs/x.sbatch", "arg one")
    assert p.returncode == 0, p.stderr
    guard = 'cd "$snap" && GUARD_LOG=/keep/submissions.log scripts/sbatch_guard.sh -t 00:20:00 jobs/x.sbatch arg\\ one'
    assert guard in p.stdout
    assert "[ \"$m\" = " in p.stdout and "exit 3; }" in p.stdout, "the mirror's own stamp is checked on the cluster"
    now = (repo / ".sync-stamp").read_text().split()[:3]
    snap = "snap=$(SNAP_ROOT=/keep/snap python3 scripts/snapshot.py make --expect " + "\\ ".join(now) + ") || exit 3;"
    assert p.stdout.index("[ \"$m\" = ") < p.stdout.index(snap) < p.stdout.index(guard), "stamp, snapshot, then guard"
    assert 'case "$snap" in /keep/snap/?*) ;;' in p.stdout, "only a path under the snapshot root is submitted from"
    assert "PAST_TARGET_OK" not in p.stdout
    p = _submit(repo, "jobs/x.sbatch", PAST_TARGET_OK="2026-12-01 Shresth")
    assert "&& PAST_TARGET_OK=2026-12-01\\ Shresth GUARD_LOG=/keep/submissions.log scripts/sbatch_guard.sh" in p.stdout
    (repo / "jobs" / "x.sbatch").write_text("#!/bin/bash\necho changed\n")     # uncommitted edit
    p = _submit(repo, "jobs/x.sbatch")
    assert p.returncode == 3, "a dirty tree no longer matches the synced stamp"
    (repo / ".sync-stamp").write_text(subprocess.run(["bash", str(repo / "scripts" / "stamp.sh")], capture_output=True,
                                                     text=True).stdout)
    assert _submit(repo, "jobs/x.sbatch").returncode == 3 and "commit first" in _submit(repo, "jobs/x.sbatch").stderr
    assert _submit(repo, "jobs/x.sbatch", ALLOW_DIRTY="1").returncode == 0


def test_stamp_counts_untracked_files_as_dirty(tmp_path):
    """make sync ships untracked files too, so a job could import one: they make the stamp dirty, like an edit."""
    repo = _submit_repo(tmp_path)
    stamp = lambda: subprocess.run(["bash", str(repo / "scripts" / "stamp.sh")], capture_output=True, text=True).stdout
    assert " dirty=0 diff=- " in stamp()
    (repo / ".sync-stamp").write_text("ignored by .gitignore\n")
    assert " dirty=0 diff=- " in stamp()
    (repo / "new.py").write_text("X = 1\n")
    first = stamp()
    assert " dirty=1 " in first
    (repo / "new.py").write_text("X = 2\n")
    assert " dirty=1 " in stamp() and stamp().split()[2] != first.split()[2], "the diff hash covers untracked content"


def test_stamp_survives_a_nested_repo_a_dangling_symlink_and_a_symlink_to_a_dir(tmp_path):
    repo = _submit_repo(tmp_path)
    stamp = lambda: subprocess.run(["bash", str(repo / "scripts" / "stamp.sh")], capture_output=True, text=True)
    (repo / "vendored").mkdir()
    _git(repo / "vendored", "init", "-q")                              # git lists an untracked nested repo as 'vendored/'
    (repo / "dangling.py").symlink_to("missing.py")
    (repo / "dirlink").symlink_to("jobs")
    p = stamp()
    assert p.returncode == 0 and " dirty=1 " in p.stdout and len(p.stdout.split()[2]) == len("diff=") + 12, p.stderr


def _frozen(repo, *args):
    return subprocess.run(["bash", str(repo / "scripts" / "check_frozen.sh"), *args], cwd=repo, capture_output=True,
                          text=True)


def test_check_frozen_refuses_a_sync_that_changes_a_frozen_file(tmp_path):
    """D-020's second amendment: while its jobs wait or run, make sync must not change a file on their code path."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "check_frozen.sh", repo / "scripts" / "check_frozen.sh")
    (repo / "src").mkdir()
    (repo / "src" / "a.py").write_text("A = 1\n")
    (repo / "src" / "b.py").write_text("B = 2\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    tree = _git(repo, "ls-tree", "-r", "HEAD", "--", "src/a.py", "src/b.py").stdout
    (repo / "frozen.txt").write_text("# a comment\n" + "".join(f"{l.split()[2]}  {l.split()[3]}\n"
                                                               for l in tree.splitlines()))
    p = _frozen(repo)
    assert p.returncode == 0 and p.stdout == "", "no .sync-freeze: nothing is frozen"
    (repo / ".sync-freeze").write_text("frozen.txt\n")
    p = _frozen(repo)
    assert p.returncode == 0 and "2 frozen files unchanged" in p.stdout, p.stderr
    (repo / "src" / "a.py").write_text("A = 3\n")          # an uncommitted edit counts: rsync ships the tree
    p = _frozen(repo)
    assert p.returncode == 1 and "src/a.py changed" in p.stderr
    (repo / "src" / "a.py").write_text("A = 1\n")
    (repo / "src" / "b.py").unlink()
    p = _frozen(repo)
    assert p.returncode == 1 and "src/b.py is gone" in p.stderr
    assert _frozen(repo, "missing.txt").returncode == 2


def test_make_sync_runs_the_freeze_check_first():
    text = (ROOT / "Makefile").read_text()
    sync = text[text.index("\nsync:"):text.index("\n\n", text.index("\nsync:"))]
    assert sync.index("scripts/check_frozen.sh") < sync.index("scripts/stamp.sh > .sync-stamp") \
        < sync.index("$(SYNC) $(CTRL) ./")
