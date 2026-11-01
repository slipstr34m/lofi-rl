"""Calibrations of the audio reward, each a JSON file in the --calib directory (docs/SCORER.md §5).

| file       | holds                                                           | missing means           |
|------------|-----------------------------------------------------------------|-------------------------|
| pq.json    | tau_pq: the 5th percentile of Audiobox PQ over Good-or-better clips | PQ factor 1 (no floor) |
| theme.json | med, mad: median and MAD of the CLAP margin m over the pool, for one prompt file (theme_sha256) | r_theme 0 |
| ce.json    | the CE check on Shresth's strict pairs: agree, lower95, n_pairs; med and mad of CE when it passes | r_ce 0 |

taste_pipeline.md §5: tau_PQ "the 5th percentile of PQ over Good-or-better clips"; r_theme = sigma((m - med) /
(1.4826 MAD)) "with median and MAD taken over the pool"; CE earns its weight "only if the sign of dCE agrees with
>= 55 % of strict pairs in the first 400-600 votes and the bootstrap lower bound is > 50 %". The server decides
`validated` from agree and lower95 itself; a file that also states `validated` must agree with that.
r_ce, once validated, is the same robust sigmoid as r_theme, over CE: sigma((CE - med) / (1.4826 MAD)) (§5 gives
CE no scale; this is the server's choice, docs/SCORER.md §8).

A missing file disables its term and says so (`flags()`, /health, every item's diag). A file that exists but is
malformed, or was made for another prompt file, stops the server at start: a broken calibration is never read as
a missing one. So does a --calib directory that is given but does not exist: a mistyped or purged path would
otherwise turn every calibration off for a whole run. Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PQ_FILE, THEME_FILE, CE_FILE = "pq.json", "theme.json", "ce.json"
CE_AGREE_MIN = 0.55             # taste_pipeline.md §5
CE_LOWER_MIN = 0.50
MAD_SCALE = 1.4826              # MAD -> sigma for a normal distribution (§5's r_theme)
_INFO = frozenset({"n", "source", "date", "note", "feat_ver"})     # informational keys any file may carry


class CalibError(ValueError):
    """A calibration file that exists but cannot be used: the server refuses to start."""


@dataclass(frozen=True)
class PQCalib:
    tau_pq: float
    path: str
    sha256: str
    info: dict


@dataclass(frozen=True)
class ThemeCalib:
    med: float
    mad: float
    theme_sha256: str
    path: str
    sha256: str
    info: dict


@dataclass(frozen=True)
class CECalib:
    agree: float
    lower95: float
    n_pairs: int
    validated: bool
    med: float | None
    mad: float | None
    path: str
    sha256: str
    info: dict


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def robust_sigmoid(x: float, med: float, mad: float) -> float:
    """sigma((x - med) / (1.4826 MAD)): §5's r_theme normalization."""
    return sigmoid((x - med) / (MAD_SCALE * mad))


def _read(path: Path) -> tuple[dict, str]:
    try:
        data = path.read_bytes()
    except OSError as err:          # it exists (is_file) but cannot be read: a broken calibration, not a missing one
        raise CalibError(f"{path}: cannot read it ({type(err).__name__}: {err})") from None
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as err:
        raise CalibError(f"{path}: not UTF-8 JSON ({err})") from None
    if not isinstance(doc, dict):
        raise CalibError(f"{path}: expected a JSON object")
    return doc, hashlib.sha256(data).hexdigest()


def _check_keys(doc: dict, path: Path, required: set[str], optional: frozenset[str] = frozenset()) -> None:
    missing = sorted(required - set(doc))
    unknown = sorted(set(doc) - required - optional - _INFO)
    if missing:
        raise CalibError(f"{path}: lacks {missing}")
    if unknown:
        raise CalibError(f"{path}: unknown keys {unknown}")


def _num(doc: dict, key: str, path: Path, lo: float | None = None, hi: float | None = None,
         positive: bool = False) -> float:
    v = doc[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise CalibError(f"{path}: {key} = {v!r} is not a finite number")
    v = float(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi) or (positive and v <= 0):
        raise CalibError(f"{path}: {key} = {v} is out of range")
    return v


def _info(doc: dict) -> dict:
    return {k: doc[k] for k in sorted(_INFO) if k in doc}


def load_pq(path: Path) -> PQCalib:
    doc, sha = _read(path)
    _check_keys(doc, path, {"tau_pq"})
    return PQCalib(_num(doc, "tau_pq", path, 0.0, 10.0), str(path), sha, _info(doc))


def load_theme(path: Path, theme_sha256: str) -> ThemeCalib:
    doc, sha = _read(path)
    _check_keys(doc, path, {"med", "mad", "theme_sha256"})
    if doc["theme_sha256"] != theme_sha256:
        raise CalibError(f"{path}: made for the prompt file with sha256 {str(doc['theme_sha256'])[:12]}..., but the "
                         f"server's is {theme_sha256[:12]}...: recalibrate on the pool with these prompts")
    return ThemeCalib(_num(doc, "med", path, -2.0, 2.0), _num(doc, "mad", path, positive=True), theme_sha256,
                      str(path), sha, _info(doc))


def load_ce(path: Path) -> CECalib:
    doc, sha = _read(path)
    _check_keys(doc, path, {"agree", "lower95", "n_pairs"}, frozenset({"validated", "med", "mad"}))
    agree, lower = _num(doc, "agree", path, 0.0, 1.0), _num(doc, "lower95", path, 0.0, 1.0)
    n = doc["n_pairs"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise CalibError(f"{path}: n_pairs = {n!r} is not a count")
    validated = agree >= CE_AGREE_MIN and lower > CE_LOWER_MIN
    if "validated" in doc and doc["validated"] is not validated:
        raise CalibError(f"{path}: says validated={doc['validated']!r}, but agree {agree} and lower95 {lower} give "
                         f"{validated} (>= {CE_AGREE_MIN} and > {CE_LOWER_MIN})")
    med = mad = None
    if validated:
        if "med" not in doc or "mad" not in doc:
            raise CalibError(f"{path}: CE passed its check, so r_ce needs med and mad of CE over the pool")
        med, mad = _num(doc, "med", path, 0.0, 10.0), _num(doc, "mad", path, positive=True)
    return CECalib(agree, lower, n, validated, med, mad, str(path), sha, _info(doc))


@dataclass(frozen=True)
class Calibration:
    """What the --calib directory holds. None means the file is missing and its term or factor is disabled."""

    dir: str | None
    pq: PQCalib | None
    theme: ThemeCalib | None
    ce: CECalib | None

    @property
    def ce_validated(self) -> bool:
        return self.ce is not None and self.ce.validated

    def flags(self) -> dict[str, dict[str, Any]]:
        """For /health: each calibration's status, file and values."""
        def entry(c: Any, status: str, **values: Any) -> dict[str, Any]:
            if c is None:
                return {"status": "missing", "effect": status}
            return {"status": "live", "file": c.path, "sha256": c.sha256, **values, **c.info}
        out = {"pq": entry(self.pq, "PQ factor 1", **({"tau_pq": self.pq.tau_pq} if self.pq else {})),
               "theme": entry(self.theme, "r_theme 0",
                              **({"med": self.theme.med, "mad": self.theme.mad} if self.theme else {}))}
        if self.ce is None:
            out["ce"] = {"status": "missing", "effect": "r_ce 0; the reward moves CE's weight to knobs"}
        else:
            out["ce"] = {"status": "validated" if self.ce.validated else "not_validated", "file": self.ce.path,
                         "sha256": self.ce.sha256, "agree": self.ce.agree, "lower95": self.ce.lower95,
                         "n_pairs": self.ce.n_pairs, "med": self.ce.med, "mad": self.ce.mad, **self.ce.info}
            if not self.ce.validated:
                out["ce"]["effect"] = "r_ce 0; the reward moves CE's weight to knobs"
        return out


def calib_version(c: Calibration, *, refs_sha256: str | None, theme_sha256: str) -> str:
    """calib_ver (lofi_rl.scoring.protocol): "calib-" + the first 12 hex of sha256 over the sha256 of pq.json,
    theme.json, ce.json and the RM's refs.json ("missing" for an absent one) and of the theme config."""
    parts = [x.sha256 if x is not None else "missing" for x in (c.pq, c.theme, c.ce)]
    parts += [refs_sha256 or "missing", theme_sha256]
    return "calib-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def load(directory: Path | str | None, *, theme_sha256: str) -> Calibration:
    """Read the calibration directory. None (no --calib) or a missing file disables; a bad file, or a directory
    that is given but does not exist, raises CalibError."""
    if directory is None:
        return Calibration(None, None, None, None)
    d = Path(directory)
    if not d.exists():
        raise CalibError(f"--calib {d} does not exist: a calibration directory that is given must exist (a "
                         f"mistyped or purged path would turn every calibration off for the whole run)")
    if not d.is_dir():
        raise CalibError(f"--calib {d} is not a directory")

    def present(name: str) -> Path | None:
        p = d / name
        return p if p.is_file() else None
    pq, th, ce = present(PQ_FILE), present(THEME_FILE), present(CE_FILE)
    return Calibration(str(d), load_pq(pq) if pq else None, load_theme(th, theme_sha256) if th else None,
                       load_ce(ce) if ce else None)
