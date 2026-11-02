#!/usr/bin/env python3
"""snapshot.py: a job runs the code it was submitted with, not whatever the mirror holds when it starts.

    python3 scripts/snapshot.py manifest                      # laptop, in `make sync`: write .sync-manifest
    python3 scripts/snapshot.py make [--expect 'c= d= diff=']  # cluster, in the mirror, from submit.sh: print the path
    python3 scripts/snapshot.py prune <key>...                # cluster: delete snapshots no queued or running job uses

manifest  lists every file `make sync` sends (rsync's own dry-run list under .rsyncignore) with its sha256, in
          `sha256sum -c` format, sorted by path. It lists .sync-stamp but not itself. It refuses symlinks, the
          names the snapshot reserves (.lofi-snapshot and .objects at the top), any `.snapshot` (VAST reserves it
          in every directory of /project2), and in a git checkout any synced file git ignores (no commit holds
          it; .sync-stamp aside). After hashing it runs stamp.sh again and refuses when the tree no longer matches
          .sync-stamp (an edit landed while make sync ran).
make      copies exactly the files .sync-manifest lists into $SNAP_ROOT/<commit>-<manifest sha12>/ (SNAP_ROOT
          defaults to $REMOTE_KEEP/snap, from the environment or kit.env), checks every hash there and makes the
          tree read-only. The manifest and the stamp are each read once. The stamp must be the one the manifest
          lists, and with --expect its first three fields must be the laptop's (submit.sh passes them), so a sync
          landing mid-submission is refused rather than snapshotted.
          submit.sh then runs the guard from that directory, so SLURM_SUBMIT_DIR is the snapshot and job_env.sh
          imports its src/. A later `make sync` cannot change the code of a queued or running job. A chain link
          that resubmits itself from inside its snapshot keeps the same code, and the budget.yaml and APPROVED.yaml
          it was submitted under.
          A file the snapshot needs that the mirror lacks is refused with exit 3 (run make sync), and so is one the
          mirror holds with other content (a failed or partial sync, an edit on the cluster) while its object is
          not stored yet. Once the object is stored, the snapshot takes the manifest's content from it, with a
          warning that the mirror differs.
          Each file is stored once, by content and exec bit, under $SNAP_ROOT/.objects/ and hard-linked into every
          snapshot that has it; it is copied, with a warning, where the file system refuses a link. A stored
          object is re-checked (hash, no write bits) before it is linked and replaced when it is wrong. A complete
          snapshot is reused once its files are re-checked against its own manifest (a file it does not list is
          refused) and its modes repaired.
          Run inside a snapshot (a <commit>-<sha12> directory directly under SNAP_ROOT), `make` checks that
          snapshot and prints it.
prune     is the only way to delete snapshots. It refuses a snapshot that is the working directory of one of
          our queued or running jobs (squeue %Z; the guard refuses --chdir from a snapshot, so a job's working
          directory is its snapshot). It moves each snapshot aside first, so no submission can reuse it, checks
          squeue again, and only then deletes it: it makes only directories writable (never the shared read-only
          files, so the other snapshots stay intact), then deletes the objects no snapshot links to.

Standard library only: the login node's python3 runs `make` and `prune` (light: about 400 small files, seconds).
"""
import errno
import hashlib
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST, STAMP, MARKER, OBJECTS = ".sync-manifest", ".sync-stamp", ".lofi-snapshot", ".objects"
RESERVED_TOP = (MARKER, OBJECTS)
VAST_RESERVED = ".snapshot"        # a hidden directory in every directory of /project2 (VAST); never a file of ours
HEX = re.compile(r"[0-9a-f]{64}")
KEY = re.compile(r"(?:[0-9a-f]{7,40}|none)-[0-9a-f]{12}")
LINK_FALLBACK = {errno.EXDEV, errno.EMLINK, errno.EPERM, errno.EOPNOTSUPP, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)}
STALE_S = 3600                     # a .tmp-* entry this old is a killed make's leftover


class Refused(Exception):
    """The snapshot cannot be made, reused or pruned (exit 3)."""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stamp_fields(stamp: str) -> str:
    """commit=… dirty=… diff=…, the part of a stamp that submit.sh compares."""
    return " ".join(stamp.split()[:3])


# ---------------------------------------------------------------- laptop: the manifest
def sync_files(root: Path, rsync: str) -> list:
    """The regular files `make sync` sends, from rsync's own dry run with the same filters."""
    with tempfile.TemporaryDirectory() as d:
        p = subprocess.run([rsync, "-an", "--exclude-from=.rsyncignore", f"--exclude=/{MANIFEST}", "--out-format=%n",
                            "./", f"{d}/dest/"], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise SystemExit(f"snapshot.py: rsync dry run failed ({p.returncode}): {p.stderr.strip()}")
    names = [n for n in p.stdout.splitlines() if n and not n.endswith("/")]
    for n in names:
        if os.path.islink(root / n) or not (root / n).is_file():
            raise SystemExit(f"snapshot.py: {n} is not a regular file (symlink?): a job snapshot cannot hold it")
        if "\\" in n:
            raise SystemExit(f"snapshot.py: {n!r} has a backslash, which sha256sum -c would misread")
        if n.split("/")[0] in RESERVED_TOP or VAST_RESERVED in n.split("/"):
            raise SystemExit(f"snapshot.py: {n}: the name is reserved ({', '.join(RESERVED_TOP)} by the snapshot, "
                             f"{VAST_RESERVED} by VAST on /project2): rename it")
    if STAMP not in names:
        raise SystemExit(f"snapshot.py: {STAMP} is not in the sync: make sync writes it before the manifest")
    if _git_tree(root):
        q = subprocess.run(["git", "check-ignore", "--stdin"], cwd=root, input="\n".join(names) + "\n",
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ignored = sorted(set(q.stdout.split("\n")) - {"", STAMP})
        if q.returncode not in (0, 1) or ignored:
            raise SystemExit(f"snapshot.py: synced but ignored by git, so no commit holds them: {ignored[:5]} "
                             f"{q.stderr.strip()}: add them to .rsyncignore (and tests/test_privacy.py) or delete them")
    return sorted(names)


def _git_tree(root: Path) -> bool:
    p = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=root, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
    return p.returncode == 0 and p.stdout.strip() == "true"


def write_manifest(root: Path, rsync: str) -> Path:
    names = sync_files(root, rsync)
    text = "".join(f"{sha256(root / n)}  {n}\n" for n in names)
    if _git_tree(root) and (root / "scripts" / "stamp.sh").is_file():
        now = subprocess.run(["bash", str(root / "scripts" / "stamp.sh")], stdout=subprocess.PIPE, text=True).stdout
        if stamp_fields(now) != stamp_fields((root / STAMP).read_text()):
            raise SystemExit(f"snapshot.py: the tree is now '{stamp_fields(now)}', not the stamp's "
                             f"'{stamp_fields((root / STAMP).read_text())}': an edit landed while make sync ran; "
                             "run make sync again")
    out = root / MANIFEST
    tmp = out.with_name(f".{MANIFEST}.tmp-{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, out)
    size = sum((root / n).stat().st_size for n in names)
    print(f"manifest: {len(names)} files, {size / 1e6:.1f} MB -> {MANIFEST}")
    return out


# ---------------------------------------------------------------- cluster: make, reuse, prune
def parse_manifest(text: str, label: str = MANIFEST) -> list:
    """[(sha256, relative path)]; refuses anything that could point outside the snapshot or collide inside it."""
    entries, names, dirs = [], set(), set()
    for i, line in enumerate(text.splitlines(), 1):
        h, sep, name = line.partition("  ")
        parts = name.split("/")
        if not sep or not HEX.fullmatch(h) or not name or name.startswith("/") or ".." in parts or "" in parts \
                or "." in parts or name == MANIFEST or parts[0] in RESERVED_TOP or VAST_RESERVED in parts:
            raise Refused(f"{label} line {i} is not '<sha256>  <relative path>': {line!r}")
        if name in names:
            raise Refused(f"{label} lists {name} twice")
        names.add(name)
        dirs.update("/".join(parts[:k]) for k in range(1, len(parts)))
        entries.append((h, name))
    clash = sorted(names & dirs)
    if clash:
        raise Refused(f"{label} lists {clash[0]} both as a file and as a directory")
    return entries


def snap_root() -> Path:
    v = os.environ.get("SNAP_ROOT")
    if not v:
        keep = os.environ.get("REMOTE_KEEP")
        if not keep and (ROOT / "kit.env").exists():
            m = re.search(r"^REMOTE_KEEP=(\S+)$", (ROOT / "kit.env").read_text(), re.M)
            keep = m.group(1) if m else None
        if not keep:
            raise Refused("neither SNAP_ROOT nor REMOTE_KEEP is set")
        v = f"{keep}/snap"
    return Path(v)


def is_snapshot(tree: Path, root: Path) -> bool:
    """A complete snapshot: a <commit>-<sha12> directory directly under the snapshot root, with its marker."""
    return tree.parent.resolve() == root.resolve() and bool(KEY.fullmatch(tree.name)) and (tree / MARKER).is_file()


def _mode(x: bool) -> int:
    return 0o500 if x else 0o400


def _expect(stamp: str, expect, where: str) -> None:
    if expect is not None and stamp_fields(stamp) != expect:
        raise Refused(f"{where} holds '{stamp_fields(stamp)}' but the laptop submitted '{expect}': a sync from "
                      "elsewhere landed meanwhile; run make sync and submit again")


def check(dest: Path, expect=None) -> Path:
    """Re-check a complete snapshot before it is reused: every file against its own manifest, its stamp against
    --expect, and its modes (repaired: read-only files, r-x directories)."""
    try:
        entries = parse_manifest((dest / MANIFEST).read_text(), f"{dest.name}/{MANIFEST}")
        stamp = (dest / STAMP).read_text()
    except OSError as e:
        raise Refused(f"snapshot {dest.name} is incomplete ({e}): python3 scripts/snapshot.py prune {dest.name}") from e
    for h, name in entries:
        p = dest / name
        if not p.is_file() or p.is_symlink() or sha256(p) != h:
            raise Refused(f"snapshot {dest.name} no longer matches its manifest ({name}): "
                          f"python3 scripts/snapshot.py prune {dest.name}, then submit again")
    listed = {name for _, name in entries} | {MANIFEST, MARKER}
    for d, dirs, files in os.walk(dest):
        dirs[:] = [x for x in dirs if x != VAST_RESERVED]
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), dest)
            if rel not in listed:
                raise Refused(f"snapshot {dest.name} holds {rel}, which its manifest does not list: "
                              f"python3 scripts/snapshot.py prune {dest.name}, then submit again")
    _expect(stamp, expect, f"snapshot {dest.name}")
    for h, name in entries:
        st = os.stat(dest / name)
        if st.st_mode & 0o222:
            os.chmod(dest / name, _mode(bool(st.st_mode & stat.S_IXUSR)))
    for d, _, _ in os.walk(dest, topdown=False):
        if os.stat(d).st_mode & 0o777 != 0o500:
            os.chmod(d, 0o500)
    return dest


def _object(objects: Path, mirror: Path, name: str, h: str) -> Path:
    """The content-addressed copy of the mirror's `name`: re-checked when it exists, created from a verified copy
    of the mirror file when it is missing or wrong."""
    src = mirror / name
    st = os.lstat(src) if os.path.lexists(src) else None
    if st is None or not stat.S_ISREG(st.st_mode):
        raise Refused(f"{name} is listed in {MANIFEST} but is not a regular file in the mirror: run make sync")
    x = bool(st.st_mode & stat.S_IXUSR)
    obj = objects / (h + (".x" if x else ""))
    if obj.is_file() and not obj.is_symlink():
        if sha256(obj) == h:
            if os.stat(obj).st_mode & 0o222:
                os.chmod(obj, _mode(x))
            if sha256(src) != h:
                print(f"snapshot.py: WARNING: the mirror's {name} differs from {MANIFEST}; the snapshot takes the "
                      "manifest's content from the stored object (a partial sync, or an edit on the cluster?)",
                      file=sys.stderr)
            return obj
        print(f"snapshot.py: WARNING: stored object for {name} is damaged; replacing it", file=sys.stderr)
    tmp = objects / f".tmp-{h}-{os.getpid()}"
    try:
        shutil.copyfile(src, tmp)
        if sha256(tmp) != h:
            raise Refused(f"{name} in the mirror does not match {MANIFEST} (a failed or partial sync, or an edit "
                          "on the cluster): run make sync")
        os.chmod(tmp, _mode(x))
        os.replace(tmp, obj)
    except OSError as e:
        raise Refused(f"cannot store {name}: {e}") from e
    finally:
        if os.path.lexists(tmp):
            os.unlink(tmp)
    return obj


def _writable(tree: Path) -> None:
    """Directories only: the files are shared read-only objects, and other snapshots link them."""
    for d, dirs, _ in os.walk(tree):
        os.chmod(d, 0o700)
        for s in dirs:
            os.chmod(os.path.join(d, s), 0o700)


def _sweep(root: Path) -> None:
    """Remove .tmp-* leftovers of killed makes (older than an hour)."""
    now = time.time()
    for parent in (root, root / OBJECTS):
        for p in parent.glob(".tmp-*"):
            try:
                if now - os.lstat(p).st_mtime < STALE_S:
                    continue
                if p.is_dir() and not p.is_symlink():
                    _writable(p)
                    shutil.rmtree(p)
                else:
                    os.unlink(p)
            except OSError:
                pass


def make(mirror: Path = ROOT, expect=None) -> Path:
    root = snap_root()
    if is_snapshot(mirror, root):
        return check(mirror, expect)           # already a snapshot: a chain link resubmitting itself
    try:
        mbytes = (mirror / MANIFEST).read_bytes()
        sbytes = (mirror / STAMP).read_bytes()
    except FileNotFoundError as e:
        raise Refused(f"no {Path(e.filename).name} in {mirror}: run make sync") from e
    entries = parse_manifest(mbytes.decode())
    if dict((n, h) for h, n in entries).get(STAMP) != hashlib.sha256(sbytes).hexdigest():
        raise Refused(f"the mirror's {STAMP} is not the one its {MANIFEST} lists (a sync in progress or a failed "
                      "one): retry, or run make sync")
    stamp = sbytes.decode().strip()
    _expect(stamp, expect, "the mirror")
    m = re.search(r"\bcommit=(\S+)", stamp)
    key = f"{m.group(1) if m else 'none'}-{hashlib.sha256(mbytes).hexdigest()[:12]}"
    dest = root / key
    if (dest / MARKER).is_file():
        return check(dest, expect)
    os.umask(0o077)
    objects = root / OBJECTS
    objects.mkdir(parents=True, exist_ok=True)
    _sweep(root)
    tmp = Path(tempfile.mkdtemp(prefix=f".tmp-{key}-", dir=root))
    copies = 0
    try:
        if os.path.lexists(tmp / MARKER):
            raise Refused(f"the file system shows a {MARKER} in a new directory: the marker name is reserved here")
        for h, name in entries:
            obj = _object(objects, mirror, name, h)
            dst = tmp / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(obj, dst)
            except OSError as e:
                if e.errno not in LINK_FALLBACK:
                    raise Refused(f"cannot link {name} into the snapshot: {e}") from e
                shutil.copy2(obj, dst)
                copies += 1
        bad = [name for h, name in entries if sha256(tmp / name) != h]
        if bad:
            raise Refused(f"{len(bad)} file(s) differ from {MANIFEST} after the copy, e.g. {bad[0]}")
        (tmp / MANIFEST).write_bytes(mbytes)
        (tmp / MARKER).write_text(f"snapshot of {mirror} made {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                                  f"files={len(entries)} copied={copies} | {stamp}\n")
        for f in (MANIFEST, MARKER):
            os.chmod(tmp / f, 0o400)
        try:
            os.rename(tmp, dest)
        except OSError:
            if not (dest / MARKER).is_file():  # not another submission that finished first
                raise
            _writable(tmp)
            shutil.rmtree(tmp)
            return check(dest, expect)
        for d, _, _ in os.walk(dest, topdown=False):
            os.chmod(d, 0o500)
        if copies:
            print(f"snapshot.py: WARNING: {copies} file(s) were copied, not hard-linked: each snapshot costs their "
                  "full size", file=sys.stderr)
        return dest
    except BaseException:
        if tmp.exists():
            _writable(tmp)
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def in_use() -> set:
    """Working directories of our queued and running jobs (fails closed when squeue fails)."""
    p = subprocess.run([os.environ.get("SQUEUE_BIN", "squeue"), "-u", os.environ.get("USER", ""), "-h", "-o", "%Z"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise Refused(f"squeue failed ({p.returncode}): not pruning without knowing which snapshots jobs use")
    return {os.path.realpath(line.strip()) for line in p.stdout.splitlines() if line.strip()}


def prune(keys: list) -> int:
    root = snap_root()
    keys = list(dict.fromkeys(keys))
    used = in_use()
    for key in keys:
        dest = root / key
        if not KEY.fullmatch(key) or not dest.is_dir() or dest.is_symlink():
            raise Refused(f"{key} is not a snapshot under {root}")
        if os.path.realpath(dest) in used:
            raise Refused(f"{key} is the working directory of a queued or running job: not pruned")
    for key in keys:
        dest, aside = root / key, root / f".tmp-prune-{key}-{os.getpid()}"
        os.chmod(dest, 0o700)
        os.rename(dest, aside)                 # a submission arriving now builds a fresh snapshot instead
        if os.path.realpath(dest) in in_use():  # one was submitted from it between the first check and the move
            os.rename(aside, dest)
            os.chmod(dest, 0o500)
            raise Refused(f"{key} became the working directory of a job while pruning: not pruned")
        _writable(aside)
        shutil.rmtree(aside)
        print(f"pruned {key}")
    n = 0
    for obj in (root / OBJECTS).iterdir():
        if obj.is_file() and not obj.name.startswith(".tmp-") and os.stat(obj).st_nlink == 1:
            obj.unlink()
            n += 1
    print(f"removed {n} object(s) no snapshot links to")
    return 0


def _stop(signum, frame):
    raise SystemExit(128 + signum)             # so the cleanup in make() runs on a hangup or a kill


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    for s in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, _stop)
    try:
        if argv[:1] == ["manifest"] and len(argv) == 1:
            write_manifest(ROOT, os.environ.get("RSYNC") or shutil.which("rsync") or "rsync")
            return 0
        if argv[:1] == ["make"] and len(argv) in (1, 3) and (len(argv) == 1 or argv[1] == "--expect"):
            print(make(expect=argv[2] if len(argv) == 3 else None))
            return 0
        if argv[:1] == ["prune"] and len(argv) > 1:
            return prune(argv[1:])
    except Refused as e:
        print(f"snapshot.py: REFUSED: {e}", file=sys.stderr)
        return 3
    except OSError as e:                       # a full quota, a permission: nothing half-made is left (make cleans up)
        print(f"snapshot.py: REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    print("\n".join(__doc__.strip().splitlines()[2:5]), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
