"""The scorer server's wire protocol (docs/SCORER.md §3): versions, the request and response schema, validation
and clip ids. Standard library only: the policy env imports it through lofi_rl.scoring.client.

Every response names four versions, and a client that pinned them at /health refuses a response that differs:
- **reward_ver** (REWARD_VER): the server's formulas: the audio gate, r_ce, r_theme and PAR;
- **feat_ver**: the features (MERT, CLAP and Audiobox revisions, windows) and the render code they were taken on;
- **rm_ver**: the reward model, "null" without one (taste_pipeline.md §4: pinned per run);
- **render_ver**: the render configuration in the clip id (taste_pipeline.md §2's render_cfg_sha8).

One item per completion. Items are keyed by clip_id (taste_pipeline.md §2, as lofi_rl.shootout.render.clip_id
computes it from a manifest):

    clip_id = "c" + sha256(sha256(canonical LoopScript) | bpm,key,mode,energy | palette_id@sha8 | render_cfg_sha8)[:16]

so two texts with one canonical form, for one brief and palette, are one clip, rendered and scored once. A text
that does not parse has no canonical form: its id hashes "raw:" + sha256(text) in the first field instead.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from lofi_rl.brief import PITCH_NAMES, Brief

PROTOCOL = "lofi-scorer/1"
REWARD_VER = "audio-reward/1"
FLAC_FORMAT = "pcm16/tpdf-splitmix64"         # = lofi_rl.shootout.render.FLAC_FORMAT, part of §2's render config
MAX_ITEMS = 1024
MAX_TEXT_CHARS = 32_768
MAX_BRIEF_ID = 200
MAX_BODY_BYTES = 16 * 2**20
AUDIO_TERMS: tuple[str, ...] = ("ce", "theme", "taste")
SIG_KEYS: tuple[str, ...] = ("pq", "pc", "ce", "cu", "clap_margin", "centroid_hz")       # taste_pipeline.md §7
DIAG_KEYS: tuple[str, ...] = ("rm_mean", "rm_std", "p_acc", "pq", "pc", "cu")             # taste_pipeline.md §5
TERM_FLAGS: tuple[str, ...] = ("pq_gate", "pacc_gate", "ce", "theme", "taste")
VERSION_KEYS: tuple[str, ...] = ("reward_ver", "feat_ver", "rm_ver", "render_ver")
CACHE_KINDS: tuple[str, ...] = ("miss", "mem", "disk", "dup", "none")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_CLIP = re.compile(r"c[0-9a-f]{16}")
_ITEM_KEYS = frozenset({"i", "text", "brief", "brief_id"})
_BRIEF_KEYS = frozenset({"energy", "mood", "bpm", "tonic", "scene"})


class ProtocolError(ValueError):
    """A request or a response that breaks the schema: HTTP 400 on the server, a ScorerError in the client."""


# ------------------------------------------------------------------------------------------ ids
def render_cfg_sha8(cfg: Mapping[str, Any]) -> str:
    """taste_pipeline.md §2's render_cfg_sha8: the first 8 hex of sha256 of the render config's sorted JSON
    ({"schema", "chain", "flac"}, as lofi_rl.shootout.render.clip_id builds it from a manifest)."""
    return hashlib.sha256(json.dumps(dict(cfg), sort_keys=True).encode()).hexdigest()[:8]


def render_version(cfg: Mapping[str, Any]) -> str:
    return f"render-cfg-{render_cfg_sha8(cfg)}"


def _id(first: str, brief: Brief, palette_id: str, palette_sha256: str, cfg: Mapping[str, Any]) -> str:
    key = (f"{first}|{brief.bpm},{PITCH_NAMES[brief.tonic]},{brief.mode},{brief.energy}|"
           f"{palette_id}@{palette_sha256[:8]}|{render_cfg_sha8(cfg)}")
    return "c" + hashlib.sha256(key.encode()).hexdigest()[:16]


def clip_id(canon_sha256: str, brief: Brief, palette_id: str, palette_sha256: str, cfg: Mapping[str, Any]) -> str:
    """The clip id of a parsed loop (module docstring); canon_sha256 is lofi_rl.dsl.canon_sha of its AST."""
    return _id(canon_sha256, brief, palette_id, palette_sha256, cfg)


def raw_clip_id(text: str, brief: Brief, palette_id: str, palette_sha256: str, cfg: Mapping[str, Any]) -> str:
    """The clip id of a text that does not parse (it has no canonical form)."""
    return _id("raw:" + hashlib.sha256(text.encode("utf-8")).hexdigest(), brief, palette_id, palette_sha256, cfg)


# ------------------------------------------------------------------------------------------ requests
@dataclass(frozen=True)
class Item:
    i: int                  # the completion's index in the reward's batch
    text: str               # the completion's LoopScript text, as generated
    brief: Brief
    brief_id: str | None    # the dataset row's brief id: the key of the RM's step-0 references (PAR)


@dataclass(frozen=True)
class ScoreRequest:
    items: tuple[Item, ...]
    palette_sha256: str
    step: int | None
    expect: Mapping[str, str]      # versions the client pinned at /health; empty for none


@dataclass(frozen=True)
class RenderRequest:
    items: tuple[Item, ...]
    palette_sha256: str
    out_dir: str | None            # absolute: write <clip_id>.flac and <clip_id>.manifest.json there
    flac: bool


def brief_dict(b: Brief) -> dict[str, Any]:
    return asdict(b)


def item_dict(i: int, text: str, brief: Brief | Mapping[str, Any], brief_id: Any = None) -> dict[str, Any]:
    """One request item, as the client sends it."""
    b = brief_dict(brief) if isinstance(brief, Brief) else dict(brief)
    return {"i": int(i), "text": text, "brief": b, "brief_id": None if brief_id is None else str(brief_id)}


def score_request(items: Sequence[Mapping[str, Any]], *, palette_sha256: str, step: int | None = None,
                  expect: Mapping[str, str] | None = None) -> dict[str, Any]:
    body = {"protocol": PROTOCOL, "palette_sha256": palette_sha256, "step": step, "items": [dict(x) for x in items]}
    if expect:
        body["expect"] = dict(expect)
    parse_score_request(body)                      # the client never sends what the server would refuse
    return body


def render_request(items: Sequence[Mapping[str, Any]], *, palette_sha256: str, out_dir: str | None = None,
                   flac: bool = False) -> dict[str, Any]:
    body = {"protocol": PROTOCOL, "palette_sha256": palette_sha256, "items": [dict(x) for x in items],
            "out_dir": out_dir, "flac": flac}
    parse_render_request(body)
    return body


def _int(v: Any, where: str, lo: int | None = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ProtocolError(f"{where} must be an integer, not {type(v).__name__}")
    if lo is not None and v < lo:
        raise ProtocolError(f"{where} must be >= {lo}, not {v}")
    return v


def _keys(obj: Any, where: str, required: frozenset[str] | set[str], optional: frozenset[str] | set[str] = frozenset()) -> dict:
    if not isinstance(obj, Mapping):
        raise ProtocolError(f"{where} must be a JSON object, not {type(obj).__name__}")
    missing = sorted(set(required) - set(obj))
    unknown = sorted(set(obj) - set(required) - set(optional))
    if missing:
        raise ProtocolError(f"{where} lacks {missing}")
    if unknown:
        raise ProtocolError(f"{where} has unknown keys {unknown}")
    return dict(obj)


def _brief(obj: Any, where: str) -> Brief:
    d = _keys(obj, where, {"energy", "mood", "bpm", "tonic"}, {"scene"})
    try:
        return Brief(**d)
    except (TypeError, ValueError) as err:
        raise ProtocolError(f"{where}: {err}") from None


def _items(obj: Any) -> tuple[Item, ...]:
    if not isinstance(obj, list):
        raise ProtocolError("items must be a list")
    if len(obj) > MAX_ITEMS:
        raise ProtocolError(f"{len(obj)} items: at most {MAX_ITEMS} per request")
    out, seen = [], set()
    for n, raw in enumerate(obj):
        where = f"items[{n}]"
        d = _keys(raw, where, {"i", "text", "brief"}, {"brief_id"})
        i = _int(d["i"], f"{where}.i", 0)
        if i in seen:
            raise ProtocolError(f"{where}.i = {i} is repeated: one item per completion")
        seen.add(i)
        text = d["text"]
        if not isinstance(text, str):
            raise ProtocolError(f"{where}.text must be a string")
        if len(text) > MAX_TEXT_CHARS:
            raise ProtocolError(f"{where}.text has {len(text)} characters: at most {MAX_TEXT_CHARS}")
        bid = d.get("brief_id")
        if bid is not None and (not isinstance(bid, str) or not bid or len(bid) > MAX_BRIEF_ID):
            raise ProtocolError(f"{where}.brief_id must be null or a string of 1-{MAX_BRIEF_ID} characters")
        out.append(Item(i, text, _brief(d["brief"], f"{where}.brief"), bid))
    return tuple(out)


def _head(obj: Any, required: set[str], optional: set[str]) -> dict:
    d = _keys(obj, "the request", required | {"protocol", "palette_sha256", "items"}, optional)
    if d["protocol"] != PROTOCOL:
        raise ProtocolError(f"protocol {d['protocol']!r}: this server speaks {PROTOCOL!r}")
    if not isinstance(d["palette_sha256"], str) or not _HEX64.fullmatch(d["palette_sha256"]):
        raise ProtocolError("palette_sha256 must be 64 lowercase hex characters")
    return d


def parse_score_request(obj: Any) -> ScoreRequest:
    d = _head(obj, set(), {"step", "expect"})
    step = d.get("step")
    if step is not None:
        step = _int(step, "step")
    expect = d.get("expect") or {}
    expect = _keys(expect, "expect", set(), set(VERSION_KEYS))
    if not all(isinstance(v, str) for v in expect.values()):
        raise ProtocolError("expect holds version strings")
    return ScoreRequest(_items(d["items"]), d["palette_sha256"], step, expect)


def parse_render_request(obj: Any) -> RenderRequest:
    d = _head(obj, set(), {"out_dir", "flac"})
    out_dir, flac = d.get("out_dir"), d.get("flac", False)
    if out_dir is not None and (not isinstance(out_dir, str) or not out_dir.startswith("/")):
        raise ProtocolError("out_dir must be null or an absolute path")
    if not isinstance(flac, bool):
        raise ProtocolError("flac must be true or false")
    return RenderRequest(_items(d["items"]), d["palette_sha256"], out_dir, flac)


def decode_json(body: bytes) -> Any:
    """A request body: UTF-8 JSON with no NaN or infinity."""
    def bad_constant(name: str) -> Any:
        raise ProtocolError(f"{name} is not JSON")
    try:
        return json.loads(body.decode("utf-8"), parse_constant=bad_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ProtocolError(f"the body is not UTF-8 JSON: {err}") from None


def encode_json(obj: Any) -> bytes:
    """A response body. allow_nan=False: a NaN that reached a response is a server bug, never sent."""
    return json.dumps(obj, allow_nan=False, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


# ------------------------------------------------------------------------------------------ responses
def _finite(v: Any, where: str, *, null: bool = False, lo: float | None = None, hi: float | None = None) -> float | None:
    if v is None and null:
        return None
    if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)):
        raise ProtocolError(f"{where} = {v!r} is not a finite number")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ProtocolError(f"{where} = {v!r} is outside [{lo}, {hi}]")
    return float(v)


def check_versions(obj: Mapping[str, Any], expect: Mapping[str, str] | None, where: str) -> dict[str, str]:
    """The four versions of a response, checked against the pinned ones (when given)."""
    got = {}
    for k in VERSION_KEYS:
        v = obj.get(k)
        if not isinstance(v, str) or not v:
            raise ProtocolError(f"{where} lacks the version {k}")
        got[k] = v
    for k, v in (expect or {}).items():
        if got.get(k) != v:
            raise ProtocolError(f"{where}: {k} is {got.get(k)!r}, but {v!r} was pinned at /health")
    return got


def check_terms(obj: Any, where: str) -> dict[str, bool]:
    if not isinstance(obj, Mapping) or set(obj) != set(TERM_FLAGS) or not all(isinstance(v, bool) for v in obj.values()):
        raise ProtocolError(f"{where}.terms must map exactly {list(TERM_FLAGS)} to booleans")
    return dict(obj)


def check_item(obj: Any, where: str) -> dict[str, Any]:
    """One response item (docs/SCORER.md §3.3). Extra keys are allowed; the documented ones must be valid."""
    if not isinstance(obj, Mapping):
        raise ProtocolError(f"{where} must be an object")
    for k in ("i", "clip_id", "hard", "G", "gate", "r", "sig", "diag", "t_ms"):
        if k not in obj:
            raise ProtocolError(f"{where} lacks {k!r}")
    _int(obj["i"], f"{where}.i", 0)
    if not isinstance(obj["clip_id"], str) or not _CLIP.fullmatch(obj["clip_id"]):
        raise ProtocolError(f"{where}.clip_id {obj['clip_id']!r} is not c + 16 hex")
    if obj["hard"] not in (0, 1) or isinstance(obj["hard"], bool):
        raise ProtocolError(f"{where}.hard must be 0 or 1")
    g = _finite(obj["G"], f"{where}.G", lo=0.0, hi=1.0)
    if obj["hard"] == 0 and g != 0.0:
        raise ProtocolError(f"{where}: G = {g} with hard = 0")
    if obj["gate"] is not None and not isinstance(obj["gate"], str):
        raise ProtocolError(f"{where}.gate must be null or a string")
    if (obj["gate"] is None) != (obj["hard"] == 1):
        raise ProtocolError(f"{where}: gate {obj['gate']!r} with hard = {obj['hard']}")
    r = obj["r"]
    if not isinstance(r, Mapping) or set(r) != set(AUDIO_TERMS):
        raise ProtocolError(f"{where}.r must hold exactly {list(AUDIO_TERMS)}")
    for k in AUDIO_TERMS:
        _finite(r[k], f"{where}.r.{k}", lo=0.0, hi=1.0)
    sig = obj["sig"]
    if not isinstance(sig, Mapping) or not set(SIG_KEYS) <= set(sig):
        raise ProtocolError(f"{where}.sig must hold {list(SIG_KEYS)}")
    for k in SIG_KEYS:
        _finite(sig[k], f"{where}.sig.{k}", null=True)
    diag = obj["diag"]
    if not isinstance(diag, Mapping) or not set(DIAG_KEYS) <= set(diag):
        raise ProtocolError(f"{where}.diag must hold {list(DIAG_KEYS)}")
    for k in DIAG_KEYS:
        _finite(diag[k], f"{where}.diag.{k}", null=True)
    if not isinstance(diag.get("disabled"), list) or not all(isinstance(x, str) for x in diag["disabled"]):
        raise ProtocolError(f"{where}.diag.disabled must be a list of term names")
    if diag.get("cache") not in CACHE_KINDS:
        raise ProtocolError(f"{where}.diag.cache must be one of {list(CACHE_KINDS)}")
    if not isinstance(obj["t_ms"], Mapping):
        raise ProtocolError(f"{where}.t_ms must be an object")
    return dict(obj)


def check_score_response(obj: Any, sent: Sequence[Mapping[str, Any]],
                         expect: Mapping[str, str] | None = None) -> dict[str, Any]:
    """A /score response for the items `sent`, in their order, with the pinned versions."""
    if not isinstance(obj, Mapping):
        raise ProtocolError("the response must be a JSON object")
    if obj.get("protocol") != PROTOCOL:
        raise ProtocolError(f"the response speaks {obj.get('protocol')!r}, not {PROTOCOL!r}")
    check_versions(obj, expect, "the response")
    check_terms(obj.get("terms"), "the response")
    items = obj.get("items")
    if not isinstance(items, list) or len(items) != len(sent):
        raise ProtocolError(f"the response has {len(items) if isinstance(items, list) else 'no'} items for "
                            f"{len(sent)} sent")
    for n, (it, s) in enumerate(zip(items, sent)):
        check_item(it, f"items[{n}]")
        if it["i"] != s["i"]:
            raise ProtocolError(f"items[{n}].i is {it['i']}, but item {s['i']} was sent there")
    return dict(obj)


def check_health(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, Mapping) or obj.get("protocol") != PROTOCOL:
        raise ProtocolError(f"/health speaks {obj.get('protocol') if isinstance(obj, Mapping) else obj!r}, "
                            f"not {PROTOCOL!r}")
    if obj.get("ok") is not True:
        raise ProtocolError(f"/health says ok={obj.get('ok')!r}")
    check_versions(obj.get("versions") or {}, None, "/health")
    check_terms(obj.get("terms"), "/health")
    if not isinstance(obj.get("palette_sha256"), str) or not _HEX64.fullmatch(obj["palette_sha256"]):
        raise ProtocolError("/health lacks the palette's sha256")
    return dict(obj)
