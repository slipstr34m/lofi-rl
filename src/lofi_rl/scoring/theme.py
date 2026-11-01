"""The CLAP theme term (taste_pipeline.md §5, "CLAP theme"; docs/SCORER.md §5.2; configs/clap_theme_v1.yaml).

    e        = normalize(mean of the music stem's four 2-bar CLAP window embeddings)     (before the rain bed)
    m        = mean_j cos(e, p_j) - max_k mean_i cos(e, c_k,i)
               p: the 8 city-at-night prompts plus the brief's scene; c_k: contrast scene k's 4 prompts
    r_theme  = sigma((m - med) / (1.4826 MAD)),  med and MAD of m over the pool (calib theme.json)

The prompt embeddings come from CLAP's text tower (features.ClapText, or a fake in tests), embedded once: the
config's texts in one batch at start, and any other scene in a batch of two (laion_clap 1.1.7 cannot embed a
batch of one text). Every number here is float64 dot products of unit vectors: deterministic.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from lofi_rl.brief import SCENE
from lofi_rl.palette import ROOT

from .calib import ThemeCalib, robust_sigmoid

DEFAULT_CONFIG = ROOT / "configs" / "clap_theme_v1.yaml"
N_POSITIVE, N_SCENES, PER_SCENE = 8, 5, 4       # §5: "8 city-at-night prompts ... 5 contrast scenes x 4 prompts"
_KEYS = {"version", "include_brief_scene", "positive", "contrast"}


class ThemeError(ValueError):
    pass


@dataclass(frozen=True)
class ThemeConfig:
    version: str
    include_brief_scene: bool
    positive: tuple[str, ...]
    contrast: tuple[tuple[str, tuple[str, ...]], ...]       # (scene, its prompts), in file order
    path: str
    sha256: str

    def texts(self) -> list[str]:
        return list(self.positive) + [p for _, ps in self.contrast for p in ps]


def _prompts(node, where: str, n: int) -> tuple[str, ...]:
    if not isinstance(node, list) or len(node) != n or not all(isinstance(p, str) and p.strip() for p in node):
        raise ThemeError(f"{where}: expected {n} non-empty prompt strings")
    if len({p.strip().lower() for p in node}) != n:
        raise ThemeError(f"{where}: prompts repeat")
    return tuple(p.strip() for p in node)


def load_config(path: Path | str = DEFAULT_CONFIG) -> ThemeConfig:
    path = Path(path)
    try:                            # an unreadable file or a YAML syntax error is a bad config (exit 2), never a traceback
        data = path.read_bytes()
        doc = yaml.safe_load(data)
    except (OSError, yaml.YAMLError) as err:
        raise ThemeError(f"{path}: cannot read the prompt config ({type(err).__name__}: {err})") from None
    if not isinstance(doc, dict) or set(doc) != _KEYS:
        raise ThemeError(f"{path}: expected exactly the keys {sorted(_KEYS)}")
    if not isinstance(doc["version"], str) or not doc["version"]:
        raise ThemeError(f"{path}: version must be a string")
    if doc["include_brief_scene"] is not True:
        raise ThemeError(f"{path}: include_brief_scene must be true (§5: the positives include the brief's scene)")
    positive = _prompts(doc["positive"], f"{path}: positive", N_POSITIVE)
    con = doc["contrast"]
    if not isinstance(con, dict) or len(con) != N_SCENES:
        raise ThemeError(f"{path}: contrast must hold {N_SCENES} scenes")
    contrast = tuple((str(k), _prompts(v, f"{path}: contrast[{k!r}]", PER_SCENE)) for k, v in con.items())
    every = [p.lower() for p in positive] + [p.lower() for _, ps in contrast for p in ps]
    if len(set(every)) != len(every):
        raise ThemeError(f"{path}: a prompt appears twice")
    return ThemeConfig(doc["version"], True, positive, contrast, str(path), hashlib.sha256(data).hexdigest())


def unit_mean(windows: np.ndarray) -> np.ndarray:
    """The clip's embedding: the mean of its window embeddings, normalized, in float64."""
    w = np.asarray(windows, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] < 1:
        raise ValueError(f"window embeddings of shape {w.shape}")
    e = w.mean(axis=0)
    n = float(np.linalg.norm(e))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("the window embeddings average to zero")
    return e / n


class Theme:
    """m and r_theme for one prompt config, with the prompt embeddings computed once by `encode_text`
    (a function: list of texts -> (k, d) unit rows)."""

    def __init__(self, config: ThemeConfig, encode_text: Callable[[list[str]], np.ndarray],
                 scenes: Sequence[str] = (SCENE,)) -> None:
        self.config = config
        self._encode = encode_text
        fixed = config.texts() + [s for s in scenes if s not in config.texts()]
        emb = self._embed(fixed)
        self._text = dict(zip(fixed, emb))
        self.pos = np.stack([self._text[p] for p in config.positive])
        self.contrast = [(scene, np.stack([self._text[p] for p in ps])) for scene, ps in config.contrast]

    def _embed(self, texts: list[str]) -> np.ndarray:
        e = np.asarray(self._encode(list(texts)), dtype=np.float64)
        if e.shape[0] != len(texts) or e.ndim != 2 or not np.all(np.isfinite(e)):
            raise ThemeError(f"the text encoder returned {e.shape} for {len(texts)} texts")
        n = np.linalg.norm(e, axis=1, keepdims=True)
        if not np.all(n > 0):
            raise ThemeError("the text encoder returned a zero vector")
        return e / n

    def scene_embedding(self, scene: str) -> np.ndarray:
        if scene not in self._text:
            self._text[scene] = self._embed([scene, scene])[0]
        return self._text[scene]

    def margin(self, windows: np.ndarray, scene: str = SCENE) -> dict:
        """m and its parts for a clip's music-stem window embeddings (module docstring)."""
        e = unit_mean(windows)
        pos = np.vstack([self.pos, self.scene_embedding(scene)[None, :]])
        pos_mean = float((pos @ e).mean())
        per = {name: float((c @ e).mean()) for name, c in self.contrast}
        top = max(per, key=lambda k: (per[k], k))
        return {"m": pos_mean - per[top], "pos_mean": pos_mean, "contrast_max": per[top], "nearest_contrast": top,
                "contrast": per}

    @staticmethod
    def r_theme(m: float, calib: ThemeCalib | None) -> float:
        return 0.0 if calib is None else robust_sigmoid(m, calib.med, calib.mad)
