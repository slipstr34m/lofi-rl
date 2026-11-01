"""The taste reward model as the scorer sees it, its loader, and PAR (docs/SCORER.md §6; taste_pipeline.md §4).

The RM itself (lofi_rl.taste: MERT/CLAP/symbolic backbones, MaxDiff heads, the ensemble) is not built yet. The
server needs only this protocol, evaluated on a clip's cached features:

    rm.rm_ver                 a version string, pinned per run
    rm.members                M, the ensemble size
    rm.utilities(feats)       [u_1(x), ..., u_M(x)]: each member's utility of the clip
    rm.p_acc(feats)           P(tier >= Fine): the ensemble's acceptability, in [0, 1]

An RM directory (--rm DIR) holds:
- rm.json:   {"rm_ver": ..., "members": M, "factory": "package.module:callable"}; the callable is called with the
             directory and returns the RM. A directory without rm.json is no RM (NullRM).
- refs.json: {"rm_ver": ..., "members": M, "refs": {brief_id: [[u_m(ref_1), ..., u_m(ref_4)] for m in 1..M]}}:
             each member's utilities of the brief's 4 step-0 reference samples.

PAR (taste_pipeline.md §4, "Ensemble and PAR"), exactly:

    d_m = u_m(x) - mean_j u_m(ref_b,j)          over the brief's 4 step-0 references
    r_taste = sigma(mean_m d_m - 1.0 * std_m d_m)

std_m is the population standard deviation over the M members (the design does not say; docs/SCORER.md §8).
Missing pieces disable what they feed: no RM (no --rm, or a directory without rm.json) -> r_taste 0 and P_acc
factor 1; an RM without refs.json -> r_taste 0, P_acc live. An --rm directory that is given but does not exist is
an error, not "no RM": a mistyped or purged path would train a whole run with taste off. A brief_id that refs.json lacks, while taste is live, is a request error (HTTP 422): the trainer
crashes rather than train on a reward with its taste term silently off for that brief. Standard library only.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .calib import sigmoid

RM_FILE, REFS_FILE = "rm.json", "refs.json"
N_REFS = 4                      # "Ref_b is 4 step-0 samples of brief b" (§4)
PAR_K = 1.0                     # r_taste = sigma(mean - 1.0 std)
NULL_RM_VER = "null"


class RMError(ValueError):
    """An RM directory that exists but cannot be used: the server refuses to start."""


@runtime_checkable
class RewardModel(Protocol):
    rm_ver: str
    members: int

    def utilities(self, feats: Any) -> Sequence[float]: ...

    def p_acc(self, feats: Any) -> float: ...


class NullRM:
    """No reward model: r_taste 0 and the P_acc factor 1 (module docstring)."""

    rm_ver = NULL_RM_VER
    members = 0

    def utilities(self, feats: Any) -> Sequence[float]:
        return ()

    def p_acc(self, feats: Any) -> float:
        raise RuntimeError("the null RM has no P_acc")


@dataclass(frozen=True)
class Refs:
    rm_ver: str
    members: int
    refs: Mapping[str, tuple[tuple[float, ...], ...]]     # brief_id -> M rows of N_REFS utilities
    path: str
    sha256: str

    def means(self, brief_id: str) -> tuple[float, ...]:
        """mean_j u_m(ref_b,j) for each member m."""
        return tuple(statistics.fmean(row) for row in self.refs[brief_id])


@dataclass(frozen=True)
class RMState:
    """The server's RM: the model (None without one), its references (None without refs.json), and why."""

    model: Any
    refs: Refs | None
    dir: str | None
    status: str                 # "live" | "missing" | "no_refs"
    rm_ver: str
    sha256: str | None = None   # of rm.json

    @property
    def taste_live(self) -> bool:
        return self.model is not None and self.refs is not None

    @property
    def pacc_live(self) -> bool:
        return self.model is not None

    def flags(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "rm_ver": self.rm_ver, "dir": self.dir}
        if self.model is None:
            out["effect"] = "r_taste 0 and P_acc factor 1"
        else:
            out.update(members=int(self.model.members), rm_json_sha256=self.sha256)
            if self.refs is None:
                out["effect"] = f"r_taste 0 ({REFS_FILE} missing); P_acc live"
            else:
                out.update(ref_briefs=len(self.refs.refs), refs_sha256=self.refs.sha256)
        return out


def par(u: Sequence[float], ref_means: Sequence[float], k: float = PAR_K) -> tuple[float, float, float]:
    """(r_taste, mean_m d_m, std_m d_m) with d_m = u_m - ref_means[m] (module docstring)."""
    if len(u) != len(ref_means) or len(u) < 1:
        raise ValueError(f"{len(u)} member utilities against {len(ref_means)} reference means")
    d = [float(a) - float(b) for a, b in zip(u, ref_means)]
    if not all(math.isfinite(x) for x in d):
        raise ValueError(f"non-finite centred utilities {d}")
    mean, std = statistics.fmean(d), statistics.pstdev(d)
    return sigmoid(mean - k * std), mean, std


def _json(path: Path) -> tuple[dict, str]:
    try:
        data = path.read_bytes()
    except OSError as err:
        raise RMError(f"{path}: cannot read it ({type(err).__name__}: {err})") from None
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as err:
        raise RMError(f"{path}: not UTF-8 JSON ({err})") from None
    if not isinstance(doc, dict):
        raise RMError(f"{path}: expected a JSON object")
    return doc, hashlib.sha256(data).hexdigest()


def _members(doc: dict, path: Path) -> int:
    m = doc.get("members")
    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        raise RMError(f"{path}: members = {m!r} is not a positive count")
    return m


def load_refs(path: Path, rm_ver: str, members: int) -> Refs:
    doc, sha = _json(path)
    unknown = sorted(set(doc) - {"rm_ver", "members", "refs", "note", "source", "date"})
    if unknown:
        raise RMError(f"{path}: unknown keys {unknown}")
    if doc.get("rm_ver") != rm_ver:
        raise RMError(f"{path}: references for rm_ver {doc.get('rm_ver')!r}, but the RM is {rm_ver!r}")
    if _members(doc, path) != members:
        raise RMError(f"{path}: {doc['members']} members, but the RM has {members}")
    raw = doc.get("refs")
    if not isinstance(raw, dict) or not raw:
        raise RMError(f"{path}: refs must map brief ids to utilities")
    refs: dict[str, tuple[tuple[float, ...], ...]] = {}
    for bid, rows in raw.items():
        ok = isinstance(rows, list) and len(rows) == members and all(
            isinstance(r, list) and len(r) == N_REFS and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in r) for r in rows)
        if not ok:
            raise RMError(f"{path}: refs[{bid!r}] must be {members} lists of {N_REFS} finite utilities")
        refs[str(bid)] = tuple(tuple(float(v) for v in r) for r in rows)
    return Refs(rm_ver, members, refs, str(path), sha)


def _factory(spec: Any, path: Path):
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise RMError(f"{path}: factory must be 'package.module:callable', not {spec!r}")
    mod, name = spec.split(":")
    try:
        module = importlib.import_module(mod)
    except Exception as err:  # noqa: BLE001: whatever the module raises at import (SyntaxError too) stops the server
        raise RMError(f"{path}: cannot load the factory {spec!r}: {type(err).__name__}: {err}") from err
    try:
        fn = getattr(module, name)
    except AttributeError as err:
        raise RMError(f"{path}: cannot load the factory {spec!r}: {err}") from err
    if not callable(fn):
        raise RMError(f"{path}: {spec!r} is not callable")
    return fn


def load_rm(directory: Path | str | None) -> RMState:
    """The --rm directory (module docstring): NullRM when none is given or it has no rm.json; RMError when the
    given directory does not exist or is broken."""
    if directory is None:
        return RMState(None, None, None, "missing", NULL_RM_VER)
    d = Path(directory)
    if not d.exists():
        raise RMError(f"--rm {d} does not exist: an RM directory that is given must exist (a mistyped or purged "
                      f"path would train the whole run with taste off)")
    if not d.is_dir():
        raise RMError(f"--rm {d} is not a directory")
    if not (d / RM_FILE).is_file():
        return RMState(None, None, str(d), "missing", NULL_RM_VER)
    path = d / RM_FILE
    doc, sha = _json(path)
    unknown = sorted(set(doc) - {"rm_ver", "members", "factory", "note", "source", "date"})
    if unknown:
        raise RMError(f"{path}: unknown keys {unknown}")
    rm_ver = doc.get("rm_ver")
    if not isinstance(rm_ver, str) or not rm_ver or rm_ver == NULL_RM_VER:
        raise RMError(f"{path}: rm_ver must be a version string other than {NULL_RM_VER!r}")
    members = _members(doc, path)
    factory = _factory(doc.get("factory"), path)
    try:
        model = factory(d)
    except Exception as err:  # noqa: BLE001: a broken RM stops the server with its reason
        raise RMError(f"{path}: the factory {doc['factory']!r} raised {type(err).__name__}: {err}") from err
    if not isinstance(model, RewardModel):
        raise RMError(f"{path}: the factory returned {type(model).__name__}, which lacks the RewardModel members")
    if model.rm_ver != rm_ver or int(model.members) != members:
        raise RMError(f"{path}: the factory's RM says rm_ver {model.rm_ver!r} with {model.members} members, "
                      f"rm.json says {rm_ver!r} with {members}")
    refs = load_refs(d / REFS_FILE, rm_ver, members) if (d / REFS_FILE).is_file() else None
    return RMState(model, refs, str(d), "live" if refs is not None else "no_refs", rm_ver, sha)
