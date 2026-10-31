"""kit.hf_cache.resolve_pinned: the pinned, checked snapshot path every offline loader uses (smoke 13624734:
huggingface_hub 2.4.0 rejected a snapshot that fetch.sbatch had fetched with allow_patterns)."""
import json
import os

import pytest

from kit.hf_cache import hub_cache, resolve_pinned

A, B = "a" * 40, "b" * 40


def _snapshot(tmp_path, rev, files):
    """A hub-cache snapshot laid out like the real one: relative symlinks into the repo's blobs/."""
    base = tmp_path / "hub" / "models--org--name"
    snap, blobs = base / "snapshots" / rev, base / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True)
    for name, text in files.items():
        blob = blobs / f"{rev[:4]}-{name.replace('/', '_')}"
        blob.write_text(text)
        link = snap / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
    return snap


def test_returns_the_pinned_snapshot_as_a_str_and_ignores_refs_main(tmp_path):
    a = _snapshot(tmp_path, A, {"config.json": "{}", "sub/extra.txt": "x"})
    _snapshot(tmp_path, B, {"config.json": "{}"})
    (tmp_path / "hub" / "models--org--name" / "refs").mkdir()
    (tmp_path / "hub" / "models--org--name" / "refs" / "main").write_text(B)
    got = resolve_pinned("org/name", A, need=["config.json"], hf_home=str(tmp_path))
    assert got == str(a) and isinstance(got, str), "vLLM's EngineArgs calls .lower() on the model: a Path breaks it"


def test_a_missing_snapshot_or_a_branch_name_fails_closed(tmp_path):
    _snapshot(tmp_path, A, {"config.json": "{}"})
    with pytest.raises(FileNotFoundError, match="no cached snapshot"):
        resolve_pinned("org/name", "c" * 40, hf_home=str(tmp_path))
    with pytest.raises(ValueError, match="40-hex"):
        resolve_pinned("org/name", "main", hf_home=str(tmp_path))


def test_dangling_links_missing_needs_and_missing_shards_are_all_named(tmp_path):
    index = {"weight_map": {"w1": "model-1.safetensors", "w2": "model-2.safetensors"}}
    snap = _snapshot(tmp_path, A, {"config.json": "{}", "model.safetensors.index.json": json.dumps(index),
                                   "model-1.safetensors": "x", "sub/extra.txt": "y"})
    (snap / "config.json").resolve().unlink()                 # a scratch purge took the blob, the link stays
    with pytest.raises(FileNotFoundError) as e:
        resolve_pinned("org/name", A, need=["config.json", "tokenizer.json"], hf_home=str(tmp_path))
    assert str(e.value).endswith("['config.json', 'model-2.safetensors', 'tokenizer.json']")


def test_a_purged_shared_store_blob_is_caught_through_both_link_levels(tmp_path):
    """hub 2.4.0 links large files twice: snapshots/<sha>/f -> blobs/<etag> -> hub/blobs/<2 hex>/<hash>."""
    snap = _snapshot(tmp_path, A, {"config.json": "{}"})
    store = tmp_path / "hub" / "blobs" / "ab" / ("ab" + "0" * 62)
    store.parent.mkdir(parents=True)
    store.write_text("weights")
    etag = tmp_path / "hub" / "models--org--name" / "blobs" / ("e" * 40)
    etag.symlink_to(os.path.relpath(store, etag.parent))
    (snap / "pytorch_model.bin").symlink_to(os.path.relpath(etag, snap))
    assert resolve_pinned("org/name", A, need=["pytorch_model.bin"], hf_home=str(tmp_path)) == str(snap)
    store.unlink()                                            # scratch purged the shared store; both links remain
    with pytest.raises(FileNotFoundError) as e:
        resolve_pinned("org/name", A, need=["pytorch_model.bin"], hf_home=str(tmp_path))
    assert str(e.value).endswith("['pytorch_model.bin']")


def test_hub_cache_expands_like_huggingface_hub(monkeypatch, tmp_path):
    for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert hub_cache() == tmp_path / ".cache" / "huggingface" / "hub"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert hub_cache() == tmp_path / "xdg" / "huggingface" / "hub"
    monkeypatch.setenv("HF_HOME", "~/hf")
    assert hub_cache() == tmp_path / "hf" / "hub"
    monkeypatch.setenv("SCR", str(tmp_path / "scr"))
    monkeypatch.setenv("HF_HUB_CACHE", "$SCR/cache")
    assert hub_cache() == tmp_path / "scr" / "cache"


def test_hub_cache_follows_huggingface_hubs_order(monkeypatch, tmp_path):
    for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert hub_cache() == tmp_path / "home" / "hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "legacy"))
    assert hub_cache() == tmp_path / "legacy"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    assert hub_cache() == tmp_path / "cache"
    assert hub_cache(str(tmp_path / "x")) == tmp_path / "x" / "hub"
