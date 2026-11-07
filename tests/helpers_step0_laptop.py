"""A temporary git repo, and synthetic step-0 runs stamped with its commits, for the laptop-side step-0 tests
(tests/test_step0_check.py, tests/test_step0_prereg.py): what lofi_rl.step0.check reads with git, so that a synthetic
run (tests/helpers_step0.py) can be valid condition by condition.

    repo, c1 = make_repo(tmp / "repo")                         # the code commit: D-023's mark and the path file
    run = stamped_run(tmp / "step0-13800001", repo, c1)        # helpers_step0.make_run, stamped with c1

**The repo** holds what the path file lists (as tests/test_step0_path.py's tree does): src/, the data files (the
SoundFont as a small stand-in: the palette is loaded without its sources), scripts/job_env.sh, scripts/_kit_env.sh,
kit.env and a stand-in jobs/step0.sbatch; a stand-in docs/DECISIONS.md holding D-023's mark (or not); and the path
file, written by lofi_rl.step0.path over that tree, committed with it. `commit` adds later commits (a redo's fix, a
file that drifts from the path file, an amendment of D-023).

**A stamped run** is make_run's run with every stamp (generate.json host.stamp, step0.json score_stamp and
render_stamp, the render-code key in step0.json, sets.json and clips.json) naming a commit, render_code.source_sha8
features.source_sha8 of that commit's tree (the repo's checkout, which is that commit), and generate.json's records of
what its pre-checks read there: path_file.sha256 and registration.decisions_sha256, read with git.

Nothing here writes outside the temporary folders the tests give it. Sound claims: none.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

import helpers_step0 as H
import lofi_rl.step0 as S
from lofi_rl.step0 import path as P

ROOT = P.ROOT
SF2 = "assets/soundfonts/generaluser-gs/GeneralUser-GS.sf2"
JOB_TEXT = "#!/bin/bash\n#SBATCH --job-name=lofi-step0\n# a stand-in for the job script, for these tests\n"
AT = "2026-11-07T19:00:00Z"
GIT_ENV = dict(os.environ, GIT_AUTHOR_NAME="T", GIT_AUTHOR_EMAIL="t@e", GIT_COMMITTER_NAME="T",
               GIT_COMMITTER_EMAIL="t@e")


def decisions_text(mark: bool = True, extra: str = "") -> str:
    """A stand-in docs/DECISIONS.md: D-023's heading and, when `mark`, its mark on a line of its own."""
    body = "# Decisions (a stand-in for the step-0 tests)\n\n## D-023 (synthetic): the step-0 samples\n\n"
    if mark:
        body += f"- **The mark** (generate checks it):\n\n      {S.MARK}\n"
    return body + extra


def git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=GIT_ENV, timeout=60)
    assert p.returncode == 0, p.stderr
    return p.stdout


def blob(repo: Path, commit: str, rel: str) -> bytes:
    p = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{rel}"], capture_output=True,
                       timeout=60)
    assert p.returncode == 0, p.stderr.decode()
    return p.stdout


def make_repo(dest: Path, *, mark: bool = True) -> tuple[Path, str]:
    """The repo (module doc), committed once: (its folder, the commit's full sha)."""
    root = Path(dest)
    shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for rel in [*P.data_files(ROOT), "scripts/job_env.sh", "scripts/_kit_env.sh", "kit.env"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        if rel == SF2:
            (root / rel).write_bytes(b"a stand-in for the SoundFont\n")
        else:
            shutil.copy(ROOT / rel, root / rel)
    (root / "jobs").mkdir()
    (root / S.JOB).write_text(JOB_TEXT)
    (root / S.DECISIONS).parent.mkdir(parents=True, exist_ok=True)
    (root / S.DECISIONS).write_text(decisions_text(mark))
    P.write(root)
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "the code commit, with D-023's mark and the path file")
    return root, git(root, "rev-parse", "HEAD").strip()


def commit(repo: Path, message: str, edits: Mapping[str, str | None]) -> str:
    """A later commit: each path's new text (None deletes it). Its full sha."""
    for rel, text in edits.items():
        p = Path(repo) / rel
        if text is None:
            p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD").strip()


def stamp(commit: str, dirty: int = 0) -> str:
    """A .sync-stamp line naming a commit (scripts/stamp.sh's form, the commit as its first 12 hex)."""
    return f"commit={commit[:12]} dirty={dirty} diff=- at={AT}"


def source_sha8(repo: Path) -> str:
    """features.source_sha8 of the repo's checkout (the commit a test stamps with is its HEAD)."""
    from lofi_rl.scoring.features import source_sha8 as sha8
    return sha8(Path(repo))


def edit_json(path: Path, fn: Callable[[dict], object]) -> None:
    doc = json.loads(Path(path).read_text())
    fn(doc)
    Path(path).write_text(json.dumps(doc, indent=1))


def edit_rows(path: Path, fn: Callable[[list[dict]], object]) -> None:
    """Change the rows of a JSONL file in place: fn(rows) edits the list."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    fn(rows)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def restamp(run_dir: Path, repo: Path, commit: str, *, steps: tuple[str, ...] = ("generate", "score", "render"),
            source: str | None = None) -> None:
    """Stamp a run's steps with a commit (module doc): generate's stamp and what its pre-checks read there, score's
    stamp, render's stamp and render-code key (source_sha8 `source`, default the repo checkout's)."""
    summ = Path(run_dir) / "summary"
    st = stamp(commit)
    if "generate" in steps:
        pf, dec = blob(repo, commit, S.PATH_FILE), blob(repo, commit, S.DECISIONS)

        def gen(d: dict) -> None:
            d["host"]["stamp"] = st
            d["path_file"]["sha256"] = hashlib.sha256(pf).hexdigest()
            d["registration"]["decisions_sha256"] = hashlib.sha256(dec).hexdigest()
        edit_json(summ / "generate.json", gen)
    if "score" in steps:
        edit_json(summ / "step0.json", lambda d: d.update(score_stamp=st))
    if "render" in steps:
        code = {"stamp": st, "source_sha8": source_sha8(repo) if source is None else source}
        edit_json(summ / "step0.json", lambda d: d.update(render_stamp=st, render_code=code))
        for name in ("sets.json", "clips.json"):
            edit_json(summ / name, lambda d: d.update(render_code=code))


def stamped_run(dest: Path, repo: Path, commit: str, table: Mapping | None = None, **kw) -> H.Run:
    """helpers_step0.make_run at `dest`, stamped with `commit` of `repo` (module doc)."""
    run = H.make_run(Path(dest), table, stamp=stamp(commit), source_sha8=source_sha8(repo), **kw)
    restamp(dest, repo, commit, steps=("generate",))
    return run


def copy_run(src: Path, dest: Path) -> Path:
    shutil.copytree(src, dest)
    return Path(dest)
