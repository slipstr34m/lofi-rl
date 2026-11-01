"""Deterministic stand-in backends for the scorer (docs/SCORER.md §9): laptop tests, the trainer rehearsal, and the
fake RM of the cluster smoke. Nothing here renders or hears audio, and no value here describes how a clip sounds.

- FakeRenderer: no FluidSynth; a clip "renders" to a record whose numbers come from its canonical text's sha256.
  `fail` maps a canonical sha256 to a (gate, detail) render failure; `infra` names ones that hit an
  infrastructure fault. `calls` lists every canonical text it was asked to render (the dedup tests count them).
- FakeFeaturizer: features drawn from a generator seeded by the clip's canonical sha256, in the real shapes
  (13 x 768 MERT, 4 x 512 unit CLAP rows); Audiobox scores from the same seed, or from `audiobox` overrides.
  encode_text gives unit vectors seeded by each text. `raise_for` names clips whose featurization raises.
- TableRM: utilities and P_acc looked up by canonical sha256 (hand-built cases); FakeRM: computed from the
  features (the smoke, on real features). load_fake_rm(dir) is an rm.json factory for FakeRM.
The server runs them with --backend fake; /health then says "backend": "fake", which a real client refuses.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .cache import ClipFeatures
from .features import WINDOWS, feat_version, source_sha8
from .render_pool import RenderJob, failure

DIM = 512


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rng(*parts: str) -> np.random.Generator:
    return np.random.default_rng(int(sha("|".join(parts))[:16], 16))


def _unit(rows: np.ndarray) -> np.ndarray:
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


class FakeRenderer:
    name = "fake"

    def __init__(self, *, fail: Mapping[str, tuple[str, str]] | None = None, infra: Iterable[str] = (),
                 workers: int = 1) -> None:
        self.fail, self.infra, self.workers = dict(fail or {}), set(infra), workers
        self.calls: list[str] = []
        self.rebuilds, self.timeout_s = 0, 0.0

    def warm(self, timeout_s: float = 0.0) -> list[int]:
        return []

    def close(self) -> None:
        pass

    def render_many(self, jobs: Sequence[RenderJob], on_done: Callable[[int, dict[str, Any]], None]) -> None:
        for k, j in enumerate(jobs):
            self.calls.append(j.canon)
            s = sha(j.canon)
            if s in self.fail:
                on_done(k, failure(*self.fail[s], transient=False))
                continue
            if s in self.infra:
                on_done(k, failure("render-error", "FluidSynthError: fake infrastructure fault", transient=True))
                continue
            b = j.brief
            info = {"n": 1_032_576, "sr": 44100, "seconds": 23.415, "bpm_eff": float(b["bpm"]), "lufs": -14.0,
                    "true_peak_dbtp": -1.5, "music_lufs": -14.0, "music_true_peak_dbtp": -1.5, "iterations": 1,
                    "bed": False, "audio_sha256": sha("fake-audio|" + j.canon + json.dumps(dict(b), sort_keys=True)),
                    "music_sha256": sha("fake-music|" + j.canon), "canon_sha256": s, "compiled_sha256": sha("c|" + s),
                    "centroid_hz": 800.0 + int(s[:6], 16) % 2400, "render_ms": 0.0}
            on_done(k, {"ok": True, "gate": None, "error": None, "transient": False, "info": info, "manifest": None,
                        "mix": None, "music": None, "render_ms": 0.0})


class FakeFeaturizer:
    name = "fake"

    def __init__(self, *, audiobox: Mapping[str, Mapping[str, float]] | None = None,
                 raise_for: Iterable[str] = ()) -> None:
        self.src_sha8 = source_sha8()
        self.feat_ver = feat_version("fake", src=self.src_sha8)
        self.audiobox, self.raise_for = dict(audiobox or {}), set(raise_for)
        self.calls: list[str] = []
        self.device = "fake"

    def featurize(self, clip_id: str, canon: str, mix: Any, music: Any) -> ClipFeatures:
        s = sha(canon)
        self.calls.append(clip_id)
        if s in self.raise_for:
            raise RuntimeError(f"fake featurizer failure on {clip_id}")
        g = _rng("feat", s)
        mert = g.standard_normal((13, 768)).astype(np.float32)
        clap_mix = _unit(g.standard_normal((WINDOWS, DIM)))
        clap_music = _unit(clap_mix + 0.1 * g.standard_normal((WINDOWS, DIM)).astype(np.float32))
        u = g.uniform(size=4)
        abx = {"PQ": 5.0 + 3.0 * u[0], "CE": 4.0 + 3.0 * u[1], "PC": 3.0 + 3.0 * u[2], "CU": 5.0 + 3.0 * u[3]}
        abx = {k: float(v) for k, v in {**abx, **self.audiobox.get(s, {})}.items()}
        return ClipFeatures(clip_id, canon, mert, clap_mix, clap_music, abx, self.device, {"fake": 0.0})

    def encode_text(self, texts: list[str]) -> np.ndarray:
        return np.stack([_unit(_rng("text", t).standard_normal((1, DIM)))[0] for t in texts]).astype(np.float64)

    def gpu(self) -> dict[str, Any]:
        return {"device": "fake", "cap_gb": None}

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "models": {}, "device": "fake", "load_s": 0.0, "windows": WINDOWS}


class TableRM:
    """Hand-built utilities: table[canonical sha256] = (utilities, p_acc); `default` for the rest."""

    def __init__(self, table: Mapping[str, tuple[Sequence[float], float]], *, rm_ver: str = "table-rm/1",
                 members: int = 5, default: tuple[Sequence[float], float] | None = None) -> None:
        self.table, self.rm_ver, self.members = dict(table), rm_ver, members
        self.default = default or ([0.0] * members, 0.9)

    def _row(self, feats: ClipFeatures) -> tuple[Sequence[float], float]:
        return self.table.get(sha(feats.canon), self.default)

    def utilities(self, feats: ClipFeatures) -> Sequence[float]:
        return [float(x) for x in self._row(feats)[0]]

    def p_acc(self, feats: ClipFeatures) -> float:
        return float(self._row(feats)[1])


class FakeRM:
    """Utilities from the features themselves, deterministic for given features: member m reads the mean of
    MERT layer 1 + 2m and the CLAP mix embedding's first coordinates. Not a taste model."""

    def __init__(self, rm_ver: str = "fake-rm/1", members: int = 5) -> None:
        self.rm_ver, self.members = rm_ver, members

    def utilities(self, feats: ClipFeatures) -> Sequence[float]:
        return [float(np.mean(feats.mert[1 + 2 * m]) * 10.0 + np.mean(feats.clap_mix[:, m])) for m in range(self.members)]

    def p_acc(self, feats: ClipFeatures) -> float:
        z = float(np.mean(feats.clap_mix[:, : self.members]) * 20.0)
        return float(1.0 / (1.0 + np.exp(-z)))


def load_fake_rm(directory: Path) -> FakeRM:
    """rm.json factory "lofi_rl.scoring.fakes:load_fake_rm": reads rm_ver and members from rm.json."""
    doc = json.loads((Path(directory) / "rm.json").read_text(encoding="utf-8"))
    return FakeRM(doc["rm_ver"], int(doc["members"]))
