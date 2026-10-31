"""Resolve Hugging Face repo ids to local snapshot directories (offline jobs, HF_HUB_OFFLINE=1).

Weights are fetched by commit sha (jobs/fetch_weights.sbatch also writes refs/main). Loading by
repo id offline needs refs/main; loading by the snapshot path never does, so every loader in the
project passes `resolve(repo_id)`.
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve(repo_id: str, hf_home: str | None = None) -> str:
    """Local snapshot path for `repo_id` if one is cached (refs/main preferred), else `repo_id`."""
    if os.path.isdir(repo_id):
        return repo_id
    home = Path(hf_home or os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    base = home / "hub" / ("models--" + repo_id.replace("/", "--"))
    ref = base / "refs" / "main"
    if ref.exists():
        snap = base / "snapshots" / ref.read_text().strip()
        if snap.is_dir():
            return str(snap)
    snaps = sorted((base / "snapshots").glob("*")) if (base / "snapshots").is_dir() else []
    return str(snaps[0]) if len(snaps) == 1 else repo_id
