"""scripts/snapshot.py: `make sync` writes .sync-manifest (rsync's own file list, sha256 each), and submit.sh freezes
the mirror into a read-only, hash-checked snapshot that the guard and sbatch run from, so a later sync cannot change
the code of a queued or running job (job_env.sh imports the snapshot's src/)."""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
import pytest

pytestmark = pytest.mark.laptop   # Homebrew rsync and git: never on a compute node (CLAUDE.md §1)

ROOT = Path(__file__).resolve().parents[1]
RSYNC = "/opt/homebrew/bin/rsync"
if not Path(RSYNC).exists():
    pytest.skip("GNU rsync 3.x (Homebrew) is needed, as for make sync", allow_module_level=True)
STAMP = "commit=abc123def456 dirty=0 diff=- at=2026-11-01T00:00:00Z\n"
FIELDS = "commit=abc123def456 dirty=0 diff=-"
MARKER = ".lofi-snapshot"


def _exec(p: Path, body: str):
    p.write_text(body)
    p.chmod(0o755)


def _repo(tmp_path, scripts=("snapshot.py",)):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for f in scripts:
        shutil.copy(ROOT / "scripts" / f, repo / "scripts" / f)
    (repo / ".rsyncignore").write_text("data/\n*.wav\n")
    (repo / "kit.env").write_text(f"KIT_PROJECT=demo\nKIT_CLUSTER_USER=u\nKIT_ACCOUNT=a\nKIT_CLUSTER_PATH=/m\n"
                                  f"REMOTE_KEEP={tmp_path / 'keep'}\n")
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "src" / "pkg" / "a.py").write_text("A = 1\n")
    (repo / "src" / "pkg" / "same.py").write_text("A = 1\n")      # the same content as a.py: one stored object
    _exec(repo / "scripts" / "run.sh", "#!/bin/bash\necho ran\n")
    (repo / "data").mkdir()
    (repo / "data" / "votes.jsonl").write_text("{}\n")              # excluded by .rsyncignore
    (repo / "x.wav").write_bytes(b"RIFF")                            # excluded by .rsyncignore
    (repo / ".sync-stamp").write_text(STAMP)
    return repo


def _run_manifest(repo):
    return subprocess.run([sys.executable, str(repo / "scripts" / "snapshot.py"), "manifest"], cwd=repo,
                          capture_output=True, text=True, env=dict(os.environ, RSYNC=RSYNC))


def _manifest(repo):
    p = _run_manifest(repo)
    assert p.returncode == 0, p.stderr
    return [line.split("  ", 1) for line in (repo / ".sync-manifest").read_text().splitlines()]


def _sync(repo, mirror):
    """What make sync does after writing .sync-stamp and .sync-manifest: the tree (-c), then the two control files."""
    mirror.mkdir(exist_ok=True)
    subprocess.run([RSYNC, "-a", "-c", "--delete", "--exclude-from=.rsyncignore", "--exclude=/.sync-stamp",
                    "--exclude=/.sync-manifest", "./", f"{mirror}/"], cwd=repo, check=True)
    subprocess.run([RSYNC, "-a", "-I", ".sync-stamp", ".sync-manifest", f"{mirror}/"], cwd=repo, check=True)


def _make(tree, *args, **env):
    e = {k: v for k, v in os.environ.items() if k not in ("SNAP_ROOT", "REMOTE_KEEP")}
    return subprocess.run([sys.executable, str(tree / "scripts" / "snapshot.py"), "make", *args], cwd=tree,
                          capture_output=True, text=True, env=dict(e, **env))


def _snap(tree, *args, **env):
    p = _make(tree, *args, **env)
    assert p.returncode == 0, p.stderr
    return Path(p.stdout.strip())


def _rm(tree: Path):
    """Remove a tree that holds read-only snapshot directories (directories only: files are shared objects)."""
    for d, _, _ in os.walk(tree):
        os.chmod(d, 0o700)
    shutil.rmtree(tree)


def _synced(tmp_path):
    repo = _repo(tmp_path)
    _manifest(repo)
    _sync(repo, tmp_path / "mirror")
    return repo, tmp_path / "mirror"


def _resync(repo, mirror, path, text):
    (repo / path).write_text(text)
    _manifest(repo)
    _sync(repo, mirror)


# ---------------------------------------------------------------- the manifest
def test_manifest_lists_exactly_what_make_sync_sends(tmp_path):
    repo = _repo(tmp_path)
    entries = _manifest(repo)
    names = [n for _, n in entries]
    assert names == sorted(names)
    assert {".sync-stamp", ".rsyncignore", "kit.env", "scripts/run.sh", "src/pkg/a.py"} <= set(names)
    assert not {"data/votes.jsonl", "x.wav", ".sync-manifest"} & set(names), "excluded files and itself are not listed"
    for h, n in entries:
        assert h == hashlib.sha256((repo / n).read_bytes()).hexdigest(), n
    if shutil.which("sha256sum"):
        assert subprocess.run(["sha256sum", "-c", ".sync-manifest"], cwd=repo, capture_output=True).returncode == 0


def test_manifest_refuses_a_symlink(tmp_path):
    repo = _repo(tmp_path)
    (repo / "link.py").symlink_to("src/pkg/a.py")
    p = _run_manifest(repo)
    assert p.returncode != 0 and "link.py is not a regular file" in p.stderr


@pytest.mark.parametrize("name", [MARKER, ".objects/x", "src/.snapshot", ".snapshot"])
def test_manifest_refuses_reserved_names(tmp_path, name):
    """The marker and the object store are the snapshot's; VAST shows a hidden .snapshot in every /project2 dir."""
    repo = _repo(tmp_path)
    (repo / name).parent.mkdir(parents=True, exist_ok=True)
    (repo / name).write_text("x\n")
    p = _run_manifest(repo)
    assert p.returncode != 0 and "the name is reserved" in p.stderr, p.stderr


def _git_repo(tmp_path):
    repo = _repo(tmp_path, scripts=("snapshot.py", "stamp.sh"))
    (repo / ".gitignore").write_text(".sync-stamp\n.sync-manifest\ndata/\n*.wav\n*.parquet\n")
    git = dict(os.environ, GIT_AUTHOR_NAME="T", GIT_AUTHOR_EMAIL="t@e", GIT_COMMITTER_NAME="T", GIT_COMMITTER_EMAIL="t@e")
    for a in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-qm", "init"]):
        subprocess.run(["git", *a], cwd=repo, check=True, env=git, capture_output=True)
    (repo / ".sync-stamp").write_text(subprocess.run(["bash", "scripts/stamp.sh"], cwd=repo, capture_output=True,
                                                     text=True, check=True).stdout)
    return repo


def test_manifest_refuses_synced_files_git_ignores(tmp_path):
    """No commit holds them, yet rsync ships them and a dirty=0 stamp would vouch for them."""
    repo = _git_repo(tmp_path)
    assert _run_manifest(repo).returncode == 0
    (repo / "src" / "pkg" / "table.parquet").write_bytes(b"PAR1")
    p = _run_manifest(repo)
    assert p.returncode != 0 and "ignored by git" in p.stderr and "src/pkg/table.parquet" in p.stderr


def test_manifest_refuses_a_tree_that_changed_after_its_stamp(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "src" / "pkg" / "a.py").write_text("A = 666\n")          # an edit landing while make sync runs
    p = _run_manifest(repo)
    assert p.returncode != 0 and "an edit landed while make sync ran" in p.stderr


def test_make_sync_sends_the_control_files_after_the_tree():
    text = (ROOT / "Makefile").read_text()
    sync = text[text.index("\nsync:"):text.index("\n\n", text.index("\nsync:"))]
    assert sync.index("scripts/stamp.sh > .sync-stamp") < sync.index("python3 scripts/snapshot.py manifest") \
        < sync.index("$(SYNC) $(CTRL) -n -i") < sync.index("$(SYNC) $(CTRL) ./") < sync.index("-I .sync-stamp .sync-manifest")
    assert "CTRL := --exclude=/.sync-stamp --exclude=/.sync-manifest" in text
    assert "SYNC := $(RS) -c --delete" in text, "checksums: rsync's quick check misses a same-size edit in the same second"
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert ".sync-stamp" in ignored and ".sync-manifest" in ignored, "untracked control files would make every stamp dirty"


# ---------------------------------------------------------------- make
def test_snapshot_is_complete_read_only_hash_checked_and_deduplicated(tmp_path):
    repo, mirror = _synced(tmp_path)
    p = _make(mirror)                                   # SNAP_ROOT from kit.env's REMOTE_KEEP
    assert p.returncode == 0, p.stderr
    snap = Path(p.stdout.strip())
    assert snap.parent == tmp_path / "keep" / "snap" and snap.name.startswith("abc123def456-") and len(snap.name) == 25
    assert (snap / MARKER).is_file() and "commit=abc123def456" in (snap / MARKER).read_text()
    for _, n in _manifest(repo):
        assert (snap / n).read_bytes() == (mirror / n).read_bytes(), n
    assert not (snap / "data").exists() and not (snap / "x.wav").exists()
    assert os.access(snap / "scripts" / "run.sh", os.X_OK) and not os.access(snap / "src" / "pkg" / "a.py", os.X_OK)
    for d, dirs, files in os.walk(snap):
        for x in [d] + [os.path.join(d, f) for f in files]:
            assert not os.stat(x).st_mode & 0o222, f"{x} is writable"
    a, same = os.stat(snap / "src" / "pkg" / "a.py"), os.stat(snap / "src" / "pkg" / "same.py")
    assert (a.st_ino, a.st_dev) == (same.st_ino, same.st_dev), "equal files share one stored object"
    objects = sorted(os.listdir(tmp_path / "keep" / "snap" / ".objects"))
    assert len(objects) == len(_manifest(repo)) - 1 and sum(o.endswith(".x") for o in objects) == 2   # run.sh, snapshot.py
    assert _make(mirror).stdout == p.stdout, "a complete snapshot is reused"
    assert _make(mirror, "--expect", FIELDS).stdout == p.stdout
    assert sorted(os.listdir(tmp_path / "keep" / "snap" / ".objects")) == objects


def test_a_later_sync_does_not_change_an_earlier_snapshot(tmp_path):
    repo, mirror = _synced(tmp_path)
    first = _snap(mirror)
    _resync(repo, mirror, "src/pkg/a.py", "A = 22\n")
    second = _snap(mirror)
    assert second != first
    assert (first / "src" / "pkg" / "a.py").read_text() == "A = 1\n"
    assert (second / "src" / "pkg" / "a.py").read_text() == "A = 22\n"
    assert os.stat(first / "kit.env").st_ino == os.stat(second / "kit.env").st_ino, "unchanged files are not copied"


def test_a_same_size_edit_in_the_same_mtime_second_still_reaches_the_snapshot(tmp_path):
    """rsync's quick check would skip it (same size and mtime second), and keep skipping it: make sync uses -c."""
    repo, mirror = _synced(tmp_path)
    a = repo / "src" / "pkg" / "a.py"
    t = os.stat(a).st_mtime
    a.write_text("A = 3\n")
    os.utime(a, (t, t))
    _manifest(repo)
    _sync(repo, mirror)
    assert (_snap(mirror) / "src" / "pkg" / "a.py").read_text() == "A = 3\n"


def test_a_mirror_that_does_not_match_its_manifest_is_refused(tmp_path):
    repo, mirror = _synced(tmp_path)
    (mirror / "src" / "pkg" / "a.py").write_text("A = 99\n")          # an edit on the cluster, or a partial sync
    p = _make(mirror, SNAP_ROOT=str(tmp_path / "s1"))
    assert p.returncode == 3 and "src/pkg/a.py in the mirror does not match .sync-manifest" in p.stderr
    assert "run make sync" in p.stderr
    assert sorted(os.listdir(tmp_path / "s1")) == [".objects"], "nothing half-made is left behind"
    assert not [o for o in os.listdir(tmp_path / "s1" / ".objects") if o.startswith(".tmp-")]
    (mirror / "src" / "pkg" / "a.py").write_text("A = 1\n")
    (mirror / "scripts" / "run.sh").unlink()                          # a file the sync never delivered
    p = _make(mirror, SNAP_ROOT=str(tmp_path / "s2"))
    assert p.returncode == 3 and "scripts/run.sh is listed in .sync-manifest but is not a regular file" in p.stderr
    os.remove(mirror / ".sync-manifest")
    p = _make(mirror, SNAP_ROOT=str(tmp_path / "s3"))
    assert p.returncode == 3 and "no .sync-manifest" in p.stderr


def test_the_stamp_must_be_the_one_the_manifest_lists(tmp_path):
    """Another sync's control files landing mid-read would pair one sync's stamp with another's manifest."""
    _, mirror = _synced(tmp_path)
    (mirror / ".sync-stamp").write_text("commit=999999999999 dirty=0 diff=- at=2026-11-01T00:00:01Z\n")
    p = _make(mirror)
    assert p.returncode == 3 and "is not the one its .sync-manifest lists" in p.stderr
    assert not (tmp_path / "keep" / "snap").exists()


def test_expect_must_match_on_build_and_on_reuse(tmp_path):
    """submit.sh passes the laptop's fields: a sync from elsewhere that landed after its stamp check is refused."""
    _, mirror = _synced(tmp_path)
    p = _make(mirror, "--expect", "commit=ffffffffffff dirty=0 diff=-")
    assert p.returncode == 3 and "a sync from elsewhere landed" in p.stderr and not (tmp_path / "keep" / "snap").exists()
    snap = _snap(mirror, "--expect", FIELDS)
    p = _make(mirror, "--expect", "commit=abc123def456 dirty=1 diff=0123456789ab")
    assert p.returncode == 3 and "a sync from elsewhere landed" in p.stderr
    p = _make(snap, "--expect", "commit=ffffffffffff dirty=0 diff=-")
    assert p.returncode == 3 and f"snapshot {snap.name} holds" in p.stderr


def test_reuse_rechecks_the_files_and_repairs_the_modes(tmp_path):
    _, mirror = _synced(tmp_path)
    snap = _snap(mirror)
    os.chmod(snap / "src", 0o700)                                      # a kill between rename and chmod, or a hand
    assert _snap(mirror) == snap and os.stat(snap / "src").st_mode & 0o777 == 0o500
    f = snap / "src" / "pkg" / "a.py"
    os.chmod(snap / "src", 0o700)
    os.chmod(snap / "src" / "pkg", 0o700)
    os.chmod(f, 0o600)
    assert _snap(mirror) == snap and os.stat(f).st_mode & 0o777 == 0o400
    os.chmod(f, 0o600)
    f.write_text("A = 7\n")                                            # the shared inode: a damaged snapshot
    p = _make(mirror)
    assert p.returncode == 3 and f"snapshot {snap.name} no longer matches its manifest (src/pkg/a.py)" in p.stderr
    assert f"snapshot.py prune {snap.name}" in p.stderr


def test_reuse_refuses_a_file_the_manifest_does_not_list(tmp_path):
    """A file planted while a killed make left the tree writable would be imported (src/sitecustomize.py)."""
    _, mirror = _synced(tmp_path)
    snap = _snap(mirror)
    os.chmod(snap / "src", 0o700)
    (snap / "src" / "sitecustomize.py").write_text("print('planted')\n")
    p = _make(mirror)
    assert p.returncode == 3 and "holds src/sitecustomize.py, which its manifest does not list" in p.stderr


def test_a_stored_object_wins_over_a_drifted_mirror_with_a_warning(tmp_path):
    repo, mirror = _synced(tmp_path)
    first = _snap(mirror)
    _resync(repo, mirror, "src/pkg/a.py", "A = 25\n")                # same.py's object is stored by the first make
    (mirror / "src" / "pkg" / "same.py").write_text("drifted\n")     # ...then the mirror's copy drifts
    p = _make(mirror)
    assert p.returncode == 0, p.stderr
    assert "the mirror's src/pkg/same.py differs from .sync-manifest" in p.stderr
    assert (Path(p.stdout.strip()) / "src" / "pkg" / "same.py").read_text() == "A = 1\n"
    assert first.exists()


def test_os_errors_are_refusals_not_tracebacks(tmp_path):
    _, mirror = _synced(tmp_path)
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, 0o500)
    try:
        p = _make(mirror, SNAP_ROOT=str(ro / "snap"))
    finally:
        os.chmod(ro, 0o700)
    assert p.returncode == 3 and "REFUSED: PermissionError" in p.stderr and "Traceback" not in p.stderr


def test_a_damaged_object_is_replaced_not_linked(tmp_path):
    repo, mirror = _synced(tmp_path)
    first = _snap(mirror)
    kit = first / "kit.env"
    os.chmod(kit, 0o600)
    kit.write_text("damaged\n")                                        # also damages the stored object (one inode)
    _resync(repo, mirror, "src/pkg/a.py", "A = 23\n")
    p = _make(mirror)
    assert p.returncode == 0, p.stderr
    second = Path(p.stdout.strip())
    assert "stored object for kit.env is damaged; replacing it" in p.stderr
    assert (second / "kit.env").read_text() == (repo / "kit.env").read_text()
    assert os.stat(second / "kit.env").st_ino != os.stat(first / "kit.env").st_ino


def test_inside_a_snapshot_make_checks_and_returns_it(tmp_path):
    _, mirror = _synced(tmp_path)
    snap = _snap(mirror)
    assert _snap(snap) == snap                                         # a chain link resubmitting from its snapshot
    assert len(os.listdir(tmp_path / "keep" / "snap")) == 2            # the snapshot and .objects, nothing new


def test_a_marker_outside_the_snapshot_root_is_not_trusted(tmp_path):
    _, mirror = _synced(tmp_path)
    (mirror / MARKER).write_text("not a snapshot\n")
    snap = _snap(mirror)
    assert snap.parent == tmp_path / "keep" / "snap" and snap != mirror


def test_manifest_paths_cannot_leave_or_collide_in_the_snapshot(tmp_path):
    _, mirror = _synced(tmp_path)
    good = (mirror / ".sync-manifest").read_text()
    for bad in ("../evil.py", "/etc/passwd", "src/../../evil.py", MARKER, ".objects/x", "src/.snapshot", "src//a.py"):
        (mirror / ".sync-manifest").write_text(good + f"{'0' * 64}  {bad}\n")
        p = _make(mirror, SNAP_ROOT=str(tmp_path / "s"))
        assert p.returncode == 3 and "is not '<sha256>  <relative path>'" in p.stderr, bad
    first = good.splitlines()[0]
    (mirror / ".sync-manifest").write_text(good + first + "\n")
    p = _make(mirror, SNAP_ROOT=str(tmp_path / "s"))
    assert p.returncode == 3 and "twice" in p.stderr
    (mirror / ".sync-manifest").write_text(good + f"{'0' * 64}  src/pkg/a.py/x\n")
    p = _make(mirror, SNAP_ROOT=str(tmp_path / "s"))
    assert p.returncode == 3 and "both as a file and as a directory" in p.stderr


# ---------------------------------------------------------------- prune
def test_prune_refuses_a_snapshot_in_use_and_deletes_only_orphan_objects(tmp_path):
    repo, mirror = _synced(tmp_path)
    old = _snap(mirror)
    (repo / "src" / "pkg" / "same.py").write_text("B = 1\n")         # then "A = 1" is only in the old snapshot
    _resync(repo, mirror, "src/pkg/a.py", "A = 24\n")
    new = _snap(mirror)
    bins = tmp_path / "bin"
    bins.mkdir()
    _exec(bins / "squeue", f'#!/bin/bash\n[ -n "$FAIL" ] && exit 1\necho "{new}"\n')
    objects = tmp_path / "keep" / "snap" / ".objects"
    before = set(os.listdir(objects))

    def prune(*keys, **env):
        e = {k: v for k, v in os.environ.items() if k not in ("SNAP_ROOT", "REMOTE_KEEP", "FAIL")}
        return subprocess.run([sys.executable, str(mirror / "scripts" / "snapshot.py"), "prune", *keys],
                              capture_output=True, text=True, env=dict(e, SQUEUE_BIN=str(bins / "squeue"), **env))
    p = prune(old.name, new.name)
    assert p.returncode == 3 and "working directory of a queued or running job" in p.stderr
    assert old.is_dir() and new.is_dir(), "nothing is pruned when one key is refused"
    assert prune(old.name, FAIL="1").returncode == 3 and old.is_dir(), "fails closed without squeue"
    assert prune("../mirror").returncode == 3 and mirror.is_dir()
    p = prune(old.name, old.name)
    assert p.returncode == 0 and not old.exists() and p.stdout.count(f"pruned {old.name}") == 1, p.stderr
    assert not [x for x in os.listdir(tmp_path / "keep" / "snap") if x.startswith(".tmp-prune-")]
    gone = before - set(os.listdir(objects))
    assert gone == {hashlib.sha256(b"A = 1\n").hexdigest()}, "only the object no snapshot links any more"
    assert _snap(mirror) == new, "the other snapshot is intact and passes its re-check"


# ---------------------------------------------------------------- jobs, the guard, submit
def test_job_env_imports_the_snapshot_and_says_when_it_runs_the_live_mirror(tmp_path):
    repo = _repo(tmp_path, scripts=("snapshot.py", "_kit_env.sh", "job_env.sh"))
    env_py = tmp_path / "keep" / "envs" / "scorer" / "bin" / "python"
    env_py.parent.mkdir(parents=True)
    env_py.symlink_to(sys.executable)
    with open(repo / "kit.env", "a") as f:
        f.write(f"REMOTE_PROJ={tmp_path / 'mirror'}\nREMOTE_SCR={tmp_path / 'scr'}\nKIT_CLUSTER_HF_HOME={tmp_path / 'hf'}\n"
                f"KIT_ENV_SCORER={env_py.parent.parent}\nKIT_ENV_POLICY={tmp_path / 'none'}\nKIT_TOOLS={tmp_path / 'none'}\n")
    _manifest(repo)
    mirror = tmp_path / "mirror"
    _sync(repo, mirror)
    snap = _snap(mirror)
    bins = tmp_path / "bin"
    bins.mkdir()
    _exec(bins / "lscpu", "#!/bin/bash\necho 'Model name: Test CPU'\n")    # a compute node has lscpu; macOS does not
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "GUARD_LOG")}
    env["PATH"] = f"{bins}:{env['PATH']}"

    def job(tree):
        return subprocess.run(["bash", "-c", 'set -euo pipefail; cd "$1"; MODE=scorer source scripts/job_env.sh; '
                               'echo "PP=$PYTHONPATH CODE=$LOFI_CODE LOG=$GUARD_LOG"', "_", str(tree)],
                              capture_output=True, text=True, env=env)
    p = job(snap)
    assert p.returncode == 0, p.stderr
    assert f"PP={snap}/src CODE={snap} LOG={tmp_path / 'keep'}/submissions.log" in p.stdout
    assert f"| code={snap} (snapshot) | commit=abc123def456" in p.stdout
    p = job(mirror)
    assert p.returncode == 0 and f"PP={mirror}/src" in p.stdout and "(NO SNAPSHOT: the live mirror" in p.stdout


def test_the_guard_in_a_snapshot_logs_to_keep_and_refuses_logs_it_could_not_write(tmp_path):
    repo = _repo(tmp_path, scripts=("snapshot.py", "sbatch_guard.sh"))
    (repo / "config").mkdir()
    (repo / "config" / "budget.yaml").write_text(
        "max_project_sus: 250000\nmax_concurrent_jobs: 8\nmax_gpu_nodes_per_job: 2\nmax_job_hours: 47.5\n"
        "warn_at_fraction: 0.7\nproject_start_date: 2026-10-25\naccount: acct\njob_name_prefix: lofi-\n")
    (repo / "jobs").mkdir()
    head = "#!/bin/bash\n#SBATCH --job-name=lofi-x --time=00:10:00 --cpus-per-task=1\n"
    (repo / "jobs" / "abs.sbatch").write_text(head + f"#SBATCH --output={tmp_path}/logs/%x_%j.out\n")
    (repo / "jobs" / "rel.sbatch").write_text(head + "#SBATCH --output=logs/%x_%j.out\n")
    (repo / "jobs" / "none.sbatch").write_text(head)
    (repo / "jobs" / "err.sbatch").write_text(head + f"#SBATCH --output={tmp_path}/logs/%j.out --error=e_%j.err\n")
    _manifest(repo)
    mirror = tmp_path / "mirror"
    _sync(repo, mirror)
    snap = _snap(mirror)
    bins = tmp_path / "bin"
    bins.mkdir()
    _exec(bins / "sacct", "#!/bin/sh\nexit 0\n")
    _exec(bins / "squeue", "#!/bin/sh\nexit 0\n")
    _exec(bins / "sbatch", f'#!/bin/sh\necho "$PWD $@" > "{tmp_path}/submitted"\necho "Submitted batch job 4242"\n')
    env = {k: v for k, v in os.environ.items() if k != "GUARD_LOG"}
    env.update(SACCT_BIN=str(bins / "sacct"), SQUEUE_BIN=str(bins / "squeue"), SBATCH_BIN=str(bins / "sbatch"),
               USER="tester", TMPDIR=str(tmp_path))

    def guard(job):
        return subprocess.run(["bash", "scripts/sbatch_guard.sh", f"jobs/{job}"], cwd=snap, capture_output=True,
                              text=True, env=env)
    p = guard("abs.sbatch")
    assert p.returncode == 0, p.stderr
    assert (tmp_path / "submitted").read_text().startswith(f"{snap} "), "sbatch runs in the snapshot"
    assert "4242" in (tmp_path / "keep" / "submissions.log").read_text(), "the ledger is KEEP's, not the snapshot's"
    for job, why in (("rel.sbatch", "--output 'logs/%x_%j.out' is not an absolute path"),
                     ("none.sbatch", "--output '<unset>' is not an absolute path"),
                     ("err.sbatch", "--error 'e_%j.err' is not an absolute path")):
        p = guard(job)
        assert p.returncode == 1 and why in p.stderr, (job, p.stderr)
    p = subprocess.run(["bash", "scripts/sbatch_guard.sh", "-D", str(tmp_path), "jobs/abs.sbatch"], cwd=snap,
                       capture_output=True, text=True, env=env)
    assert p.returncode == 1 and "moves the job out of its code snapshot" in p.stderr
    work = tmp_path / "work"                           # the guard's own tests run it from a snapshot (render_check):
    work.mkdir()                                       # sbatch's cwd is theirs, writable, so the rule does not apply
    shutil.copy(snap / "jobs" / "none.sbatch", work / "none.sbatch")
    p = subprocess.run(["bash", str(snap / "scripts" / "sbatch_guard.sh"), "none.sbatch"], cwd=work,
                       capture_output=True, text=True, env=dict(env, GUARD_LOG=str(work / "ledger")))
    assert p.returncode == 0 and "4242" in (work / "ledger").read_text(), p.stderr


def test_submit_runs_the_guard_inside_a_fresh_snapshot(tmp_path):
    """End to end through submit.sh's real remote command: stamp check, snapshot, then the guard in the snapshot."""
    repo = _repo(tmp_path, scripts=("snapshot.py", "_kit_env.sh", "stamp.sh", "submit.sh"))
    mirror = tmp_path / "mirror"
    _exec(repo / "scripts" / "remote.sh", '#!/bin/bash\ncd "$MIRROR" && exec bash -c "$*"\n')
    _exec(repo / "scripts" / "sbatch_guard.sh",
          '#!/bin/bash\necho "GUARD in $PWD log=$GUARD_LOG pto=${PAST_TARGET_OK:-none}: $*"\n')
    (repo / "jobs").mkdir()
    (repo / "jobs" / "x.sbatch").write_text("#!/bin/bash\n")
    (repo / ".gitignore").write_text(".sync-stamp\n.sync-manifest\n")
    git = dict(os.environ, GIT_AUTHOR_NAME="T", GIT_AUTHOR_EMAIL="t@e", GIT_COMMITTER_NAME="T", GIT_COMMITTER_EMAIL="t@e")
    for a in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-qm", "init"]):
        subprocess.run(["git", *a], cwd=repo, check=True, env=git, capture_output=True)
    (repo / ".sync-stamp").write_text(subprocess.run(["bash", "scripts/stamp.sh"], cwd=repo, capture_output=True,
                                                     text=True, check=True).stdout)
    assert " dirty=0 " in (repo / ".sync-stamp").read_text()
    _manifest(repo)
    _sync(repo, mirror)

    def submit(*args, **env):
        e = {k: v for k, v in os.environ.items() if k != "PAST_TARGET_OK"}
        return subprocess.run(["bash", "scripts/submit.sh", *args], cwd=repo, capture_output=True, text=True,
                              env=dict(e, MIRROR=str(mirror), **env))
    p = submit("-t", "00:20:00", "jobs/x.sbatch", "arg one")
    assert p.returncode == 0, p.stderr
    snap = next(l for l in p.stdout.splitlines() if l.startswith("code snapshot: ")).split(": ", 1)[1]
    assert Path(snap).parent == tmp_path / "keep" / "snap" and (Path(snap) / MARKER).is_file()
    assert f"GUARD in {snap} log={tmp_path / 'keep'}/submissions.log pto=none: -t 00:20:00 jobs/x.sbatch arg one" \
        in p.stdout
    p = submit("jobs/x.sbatch", PAST_TARGET_OK="2026-11-01 (chat)")
    assert p.returncode == 0 and "pto=2026-11-01 (chat): jobs/x.sbatch" in p.stdout, p.stderr
    (mirror / "src" / "pkg" / "a.py").write_text("A = 99\n")          # the mirror drifts: a fresh snapshot is refused
    _rm(tmp_path / "keep" / "snap")                                   # objects included: nothing to rebuild from
    p = submit("jobs/x.sbatch")
    assert p.returncode == 3 and "does not match .sync-manifest" in p.stderr and "GUARD" not in p.stdout
