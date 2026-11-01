"""Clip records and the feature cache (taste_pipeline.md §5, "Caching"; docs/SCORER.md §7).

A clip record is everything the reward needs from a clip's audio: the render's outcome and numbers, and its
features (MERT, CLAP on the mix and on the music stem, Audiobox). It is a function of the clip id and feat_ver
alone, so it is cached:

    <cache>/feat/<feat_ver>/hw-<sha8>/<clip_id>.npz     arrays mert, clap_mix, clap_music, plus meta (JSON text)
    an in-memory LRU of 4,096 records in front of it

feat_ver names the models, windows, the render code and the library stack the features were taken on
(features.feat_version), so a record from other code is never read. `hw` names the hardware they were taken on,
lofi_rl.render.machine.machine_key() (system, architecture, CPU model) and the GPU's name, the project's render
rule ("two CPU models never share renders"): the server keys the directory by its sha8, and every record stores
both, so an A100 node and an L40S node, or two CPU models, never serve each other's records (PQ against tau_PQ
and the CLAP margin would otherwise depend on which node saw a clip first). Without `hw` (tests) the records sit
in <cache>/feat/<feat_ver>/. Gate failures of the render (G7, a compile error) are deterministic and
cached; infrastructure failures (a FluidSynth fault, a timeout) are not. Writes go to a temporary file and are
renamed into place, so a reader never sees half a record. A file that does not load, or holds another clip or
version, is a miss and is overwritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "lofi-scorer/clip/1"
LRU_SIZE = 4096
AUDIOBOX_KEYS = ("PQ", "CE", "PC", "CU")
ARRAYS = ("mert", "clap_mix", "clap_music")


@dataclass(frozen=True, eq=False)
class ClipFeatures:
    """A clip's features, on the mix as heard unless named otherwise (taste_pipeline.md §4, §5)."""

    clip_id: str
    canon: str                      # the canonical LoopScript: what a symbolic RM backbone reads
    mert: np.ndarray                # (13, 768) float32: MERT-v1-95M per-layer means over four 2-bar windows
    clap_mix: np.ndarray            # (4, 512) float32: CLAP music window embeddings of the mix (unit rows)
    clap_music: np.ndarray          # (4, 512) float32: the same on the music-only stem (theme)
    audiobox: dict[str, float]      # PQ, CE, PC, CU on the mix
    device: str = ""                # the --device: "cuda" or "cpu" ("fake" for the fakes)
    t_ms: dict[str, float] = field(default_factory=dict)
    gpu: str = ""                   # torch.cuda.get_device_name(0), or "cpu" ("fake")
    machine: str = ""               # lofi_rl.render.machine.machine_key(): system, architecture, CPU model


@dataclass(frozen=True, eq=False)
class ClipRecord:
    clip_id: str
    canon: str | None               # None when the text did not parse
    ok: bool                        # parse, compile and render passed, in spec: the item's `hard`
    gate: str | None = None         # why not: G1-G6 (parser), "G7" (render), "compile", "render-error", "render-timeout"
    error: str | None = None
    transient: bool = False         # an infrastructure failure: never cached
    render: dict[str, Any] = field(default_factory=dict)
    feats: ClipFeatures | None = None


def hw_key(machine: str, gpu: str) -> str:
    """The hardware a record was made on: lofi_rl.render.machine.machine_key() and the GPU's name."""
    return f"{machine}|{gpu}"


def record_meta(rec: ClipRecord, feat_ver: str, hw: str | None = None) -> dict[str, Any]:
    f = rec.feats
    return {"schema": SCHEMA, "clip_id": rec.clip_id, "feat_ver": feat_ver, "hw": hw, "canon": rec.canon,
            "ok": rec.ok, "gate": rec.gate, "error": rec.error, "render": rec.render,
            "audiobox": None if f is None else {k: float(f.audiobox[k]) for k in AUDIOBOX_KEYS},
            "device": None if f is None else f.device, "gpu": None if f is None else f.gpu,
            "machine": None if f is None else f.machine, "t_ms": None if f is None else f.t_ms}


class FeatureCache:
    """The two-level cache. `root` None keeps only the in-memory LRU; `hw` (hw_key: the server passes its
    featurizer's) keys the directory and is checked in every record. Counters: hits_mem, hits_disk, misses,
    writes, bad (files that did not load)."""

    def __init__(self, root: Path | str | None, feat_ver: str, size: int = LRU_SIZE, hw: str | None = None) -> None:
        if not feat_ver or any(c in feat_ver for c in "/\\\0") or feat_ver.startswith("."):
            raise ValueError(f"feat_ver {feat_ver!r} cannot name a directory")
        self.feat_ver, self.hw = feat_ver, hw
        sub = () if hw is None else (f"hw-{hashlib.sha256(hw.encode()).hexdigest()[:8]}",)
        self.dir = None if root is None else Path(root, "feat", feat_ver, *sub)
        self.size = int(size)
        self._lru: OrderedDict[str, ClipRecord] = OrderedDict()
        self._lock = threading.Lock()
        self.hits_mem = self.hits_disk = self.misses = self.writes = self.bad = 0
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, clip_id: str) -> Path | None:
        return None if self.dir is None else self.dir / f"{clip_id}.npz"

    def __len__(self) -> int:
        return len(self._lru)

    def _remember(self, rec: ClipRecord) -> None:
        self._lru[rec.clip_id] = rec
        self._lru.move_to_end(rec.clip_id)
        while len(self._lru) > self.size:
            self._lru.popitem(last=False)

    def get(self, clip_id: str) -> tuple[ClipRecord, str] | None:
        """(record, "mem" | "disk"), or None on a miss."""
        with self._lock:
            rec = self._lru.get(clip_id)
            if rec is not None:
                self._lru.move_to_end(clip_id)
                self.hits_mem += 1
                return rec, "mem"
        rec = self._load(clip_id)
        with self._lock:
            if rec is None:
                self.misses += 1
                return None
            self.hits_disk += 1
            self._remember(rec)
            return rec, "disk"

    def put(self, rec: ClipRecord) -> None:
        if rec.transient:
            return
        with self._lock:
            self._remember(rec)
        path = self.path(rec.clip_id)
        if path is None:
            return
        arrays: dict[str, np.ndarray] = {"meta": np.array(json.dumps(record_meta(rec, self.feat_ver, self.hw),
                                                                     sort_keys=True))}
        if rec.feats is not None:
            for k in ARRAYS:
                arrays[k] = np.ascontiguousarray(getattr(rec.feats, k), dtype=np.float32)
        fd, tmp = tempfile.mkstemp(prefix=f".{rec.clip_id}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, **arrays)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        with self._lock:
            self.writes += 1

    def _load(self, clip_id: str) -> ClipRecord | None:
        path = self.path(clip_id)
        if path is None or not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                if (meta.get("schema"), meta.get("clip_id"), meta.get("feat_ver")) != (SCHEMA, clip_id, self.feat_ver):
                    raise ValueError("another schema, clip or feat_ver")
                if self.hw is not None and meta.get("hw") != self.hw:
                    raise ValueError("made on other hardware")
                feats = None
                if meta["ok"]:
                    arr = {k: np.array(z[k], dtype=np.float32) for k in ARRAYS}
                    if any(a.ndim != 2 or not np.all(np.isfinite(a)) for a in arr.values()):
                        raise ValueError("bad arrays")
                    ab = meta["audiobox"]
                    feats = ClipFeatures(clip_id, meta["canon"], arr["mert"], arr["clap_mix"], arr["clap_music"],
                                         {k: float(ab[k]) for k in AUDIOBOX_KEYS}, meta.get("device") or "",
                                         dict(meta.get("t_ms") or {}), gpu=meta.get("gpu") or "",
                                         machine=meta.get("machine") or "")
            return ClipRecord(clip_id, meta["canon"], bool(meta["ok"]), meta["gate"], meta["error"], False,
                              dict(meta["render"] or {}), feats)
        except Exception:  # noqa: BLE001: an unreadable record is a miss, rewritten by the next put
            with self._lock:
                self.bad += 1
            return None

    def counters(self) -> dict[str, Any]:
        with self._lock:
            return {"dir": None if self.dir is None else str(self.dir), "hw": self.hw, "lru": len(self._lru),
                    "lru_size": self.size,
                    "hits_mem": self.hits_mem, "hits_disk": self.hits_disk, "misses": self.misses,
                    "writes": self.writes, "bad": self.bad}
