"""Resolve Hugging Face repo ids to local snapshot directories (offline jobs, HF_HUB_OFFLINE=1).

Weights are fetched by commit sha (jobs/fetch.sbatch also writes refs/main). Loaders pass
`resolve_pinned(repo_id, revision)`: the snapshot path at the pinned commit, checked on disk. A path never
goes through huggingface_hub, so hub 2.x's offline completeness check (trees/<sha>.json against the files
on disk) cannot reject a snapshot fetched with allow_patterns (smoke 13624734). Not pinned yet: the
Audiobox predictor and laion_clap's roberta-base, which their libraries load by repo id.
`resolve(repo_id)` is the unpinned kit helper, kept for ad-hoc use.
Stdlib only, so it imports the same in every env.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path

_SHA = re.compile(r"[0-9a-f]{40}")


def hub_cache(hf_home: str | None = None) -> Path:
    """The hub cache dir as huggingface_hub resolves it (constants.py in 0.36 and 2.4): $HF_HUB_CACHE, else
    $HUGGINGFACE_HUB_CACHE, else $HF_HOME/hub, else $XDG_CACHE_HOME/huggingface/hub, else
    ~/.cache/huggingface/hub, with ~ and $VARS expanded. An empty variable counts as unset; an explicit
    `hf_home` wins (tests)."""
    def expand(p: str) -> Path:
        return Path(os.path.expandvars(os.path.expanduser(p)))
    if hf_home:
        return expand(hf_home) / "hub"
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return expand(env)
    home = os.environ.get("HF_HOME") or os.path.join(os.environ.get("XDG_CACHE_HOME") or "~/.cache", "huggingface")
    return expand(home) / "hub"


def resolve_pinned(repo_id: str, revision: str, need: Iterable[str] = (), hf_home: str | None = None) -> str:
    """The cached snapshot dir of `repo_id` at commit `revision`, as a str (vLLM rejects a Path).

    Fails closed with FileNotFoundError, naming what is missing, if the snapshot dir is absent, if any
    entry in it is a dangling blob link (scratch purges), if a file in `need` is absent, or if a shard
    named in model.safetensors.index.json is absent. It never falls back to refs/main, another snapshot
    or the repo id.
    """
    if not _SHA.fullmatch(revision):
        raise ValueError(f"{repo_id}: pin a 40-hex commit sha, not {revision!r}")
    snap = hub_cache(hf_home) / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
    if not snap.is_dir():
        raise FileNotFoundError(f"{repo_id}@{revision[:8]}: no cached snapshot at {snap} (fetch it with jobs/fetch.sbatch)")
    missing = {str(p.relative_to(snap)) for p in snap.rglob("*") if not p.exists()}   # exists() follows links
    missing |= {f for f in need if not (snap / f).is_file()}
    index = snap / "model.safetensors.index.json"
    if index.is_file():
        missing |= {f for f in json.loads(index.read_text())["weight_map"].values() if not (snap / f).is_file()}
    if missing:
        raise FileNotFoundError(f"{repo_id}@{revision[:8]}: missing or dangling in {snap}: {sorted(missing)}")
    return str(snap)


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
