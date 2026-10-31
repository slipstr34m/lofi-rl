"""Votes, audio and heavy or non-commercial assets never enter git, and votes never go to the rsync mirror
(CLAUDE.md §7, §11; docs/DECISIONS.md D-010, D-012)."""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


needs_git = pytest.mark.skipif(_git("rev-parse", "--git-dir").returncode != 0, reason="not a git checkout")


@needs_git
@pytest.mark.parametrize("path", ["data/votes/2026-11-10.jsonl", "data/briefs/x.jsonl", "inbox/2026-11-03/a.flac",
                                  "runs/r1/summary/gate.json", "assets/big/jrhodes3d.sf2", "x.wav", "ckpt/step-25/a.safetensors",
                                  ".sync-stamp", ".env"])
def test_private_and_heavy_paths_are_ignored(path):
    assert _git("check-ignore", "-q", path).returncode == 0, f"{path} is not git-ignored"


@needs_git
def test_nothing_private_is_tracked():
    tracked = _git("ls-files").stdout.splitlines()
    bad = [f for f in tracked if f.startswith(("data/", "inbox/", "runs/", "ckpt/", "assets/big/"))
           or f.endswith((".wav", ".flac", ".mp3", ".pt", ".safetensors"))]
    assert not bad, bad


def test_votes_and_audio_never_reach_the_mirror():
    rules = (ROOT / ".rsyncignore").read_text().split()
    for pattern in ("data/", "inbox/", "runs/", "*.wav", "*.flac", ".env", ".git/", "jobs/submissions.log"):
        assert pattern in rules, pattern
