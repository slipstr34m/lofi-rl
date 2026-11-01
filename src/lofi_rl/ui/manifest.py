"""Batch manifests for the rating UI (docs/RATING.md; docs/design/taste_pipeline.md §1-§3).

A batch is one folder, `inbox/<date>/<batch>/`, holding opaque audio files and two manifests:

- `manifest.json` (public): the batch id, mode, scene text, bpm and bars, and rounds of opaque clip ids with
  their files. The page gets a trimmed copy of it (`Manifest.public()`).
- `manifest.hidden.json` (never served): which rounds are repeats or anchors, and what each clip really is
  (its real clip id, option or variant name, source). The server reads it only to label the votes it writes, for
  the audit aggregates once voting is done, and for `serve_from` when it names one: the first day the batch may be
  served (a Day-3 shootout batch, whose repeats come at least 2 days after their first showing's vote, D-013).

Both are validated on load; every problem is reported at once (`ManifestError.errors`). Opaque ids and file
names must not leak what a clip is: the check compares them with every word in the hidden manifest.

Stdlib only, because the server imports this module. `demo` writes synthetic clips (sine or noise) for tests
and for a browser check; it imports numpy and soundfile lazily and falls back to stdlib `wave` without them.

    python -m lofi_rl.ui.manifest demo <dir> --mode bws4      # a synthetic batch in <dir>
    python -m lofi_rl.ui.manifest check <dir>                  # validate a batch folder
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import stat
import struct
import sys
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path

SCHEMA_VERSION = 1
PUBLIC_NAME = "manifest.json"
HIDDEN_NAME = "manifest.hidden.json"

# mode -> (fewest, most) clips per round
MODES: dict[str, tuple[int, int]] = {
    "shootout": (2, 4),   # <=4 blind variants of one LoopScript -> best and worst
    "keep3": (2, 3),      # 3 drafts -> keep 1, or "none acceptable"
    "bws4": (4, 4),       # 4 clips from one brief -> best, worst, escape hatches, one tag
    "triage": (1, 1),     # 1 clip -> Toss / Fine / Good / Great
    "ab": (2, 2),         # 2 clips -> forced pick (the final A/B)
    "playlist": (1, 1),   # background: each loop plays 2-4 times; a heart or an x only when one stands out
}
KINDS = ("none", "repeat", "anchor", "heldout")

BATCH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
ROUND_RE = re.compile(r"r[0-9a-f]{6,32}")
CLIP_RE = re.compile(r"[a-z][0-9a-f]{8,32}")
FILE_RE = re.compile(r"[a-z][0-9a-f]{8,32}\.(?:flac|wav)")
REAL_RE = re.compile(r"c[0-9a-f]{8,64}")       # real clip ids: "c" + sha256(...)[:16] (taste_pipeline §2)
SHA_RE = re.compile(r"[0-9a-f]{64}")
DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

PUBLIC_KEYS = frozenset({"v", "batch", "mode", "scene", "note", "bpm", "bars", "sr", "created", "est_min", "rounds"})
ROUND_KEYS = frozenset({"id", "scene", "bpm", "plays", "clips"})
CLIP_KEYS = frozenset({"id", "file", "sha256"})

MAX_FILE_BYTES = 256 * 1024 * 1024
DURATION_TOL = 0.02      # a clip's length may differ from bars x 240 / bpm by 2 % (N is rounded to whole blocks)
LEAK_WINDOW = 10         # this many consecutive hex digits of a real clip id count as a leak
TEXT_MAX = 300


class ManifestError(ValueError):
    """A manifest failed validation. `errors` lists every problem found."""

    def __init__(self, errors, where: str = ""):
        self.errors = [str(e) for e in errors] or ["invalid manifest"]
        self.where = str(where)
        head = f"{self.where}: " if self.where else ""
        if len(self.errors) == 1:
            msg = head + self.errors[0]
        else:
            msg = head + f"{len(self.errors)} problems:\n  - " + "\n  - ".join(self.errors)
        super().__init__(msg)


@dataclass(frozen=True)
class Clip:
    id: str                       # opaque, per batch
    file: str                     # bare file name inside the batch folder
    pos: int                      # 1-based tile position on the page
    sha256: str | None = None
    frames: int | None = None     # filled in when the files are checked
    sr: int | None = None
    channels: int | None = None


@dataclass(frozen=True)
class Round:
    id: str
    number: int                   # 1-based position in the manifest
    scene: str
    bpm: float
    plays: int | None             # playlist only: how many times the loop plays
    clips: tuple[Clip, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.clips)

    def clip(self, cid: str) -> Clip | None:
        for c in self.clips:
            if c.id == cid:
                return c
        return None


@dataclass(frozen=True)
class Manifest:
    dir: Path
    batch: str
    mode: str
    bars: int
    rounds: tuple[Round, ...]
    scene: str | None = None
    note: str | None = None
    bpm: float | None = None
    sr: int | None = None
    created: str | None = None
    est_min: float | None = None
    _by_id: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_by_id", {r.id: r for r in self.rounds})

    def round(self, rid) -> Round | None:
        return self._by_id.get(rid) if isinstance(rid, str) else None

    @property
    def files(self) -> dict[str, Path]:
        """Every file the page may fetch: bare name -> path. Nothing else in the folder is ever served."""
        return {c.file: self.dir / c.file for r in self.rounds for c in r.clips}

    def public(self, plays_range: tuple[int, int] = (2, 4)) -> dict:
        """What the page gets: the scene is the only brief information (no bpm, no hashes, nothing hidden)."""
        out = {"v": SCHEMA_VERSION, "batch": self.batch, "mode": self.mode, "bars": self.bars, "sr": self.sr,
               "note": self.note, "est_min": self.est_min, "rounds": []}
        for r in self.rounds:
            row = {"id": r.id, "scene": r.scene, "clips": [{"id": c.id, "file": c.file} for c in r.clips]}
            if self.mode == "playlist":
                row["plays"] = r.plays if r.plays else default_plays(r.id, plays_range)
            out["rounds"].append(row)
        return out


@dataclass(frozen=True)
class HiddenRound:
    id: str
    kind: str                     # none | repeat | anchor | heldout
    of: str | None                # repeat: the round id of the first presentation (often another batch)
    anchor_clip: str | None       # anchor: the opaque id of the broken copy
    brief_id: str | None
    category: str | None          # shootout: what is being compared ("keys", "drums+bass", ...)
    clips: dict                   # opaque id -> {"clip": real id, ...what it really is}

    def real(self, oid) -> str | None:
        info = self.clips.get(oid) if oid else None
        return info["clip"] if info else None


@dataclass(frozen=True)
class Hidden:
    batch: str
    category: str | None
    rounds: dict                  # round id -> HiddenRound
    serve_from: date | None = None    # the first day the batch may be served (ui.serve), or None


def serve_from_date(value) -> date | None:
    """A hidden manifest's `serve_from` as a date: YYYY-MM-DD, the first day `make rate` may serve the batch (a
    Day-3 shootout batch, since its repeats come at least 2 days after their first showing's vote, D-013), or None
    when there is none. ValueError for anything else."""
    if value is None:
        return None
    if not isinstance(value, str) or not DATE_RE.fullmatch(value):
        raise ValueError(f"serve_from {value!r} is not a date YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"serve_from {value!r} is not a date YYYY-MM-DD") from None


def default_plays(round_id: str, plays_range: tuple[int, int] = (2, 4)) -> int:
    """Playlist rounds without `plays`: a fixed pick in [lo, hi] from the round id, so a reload never changes it."""
    lo, hi = plays_range
    h = int(hashlib.sha256(round_id.encode()).hexdigest()[:8], 16)
    return lo + h % (hi - lo + 1)


# ----------------------------------------------------------------------------- audio headers (stdlib only)

def audio_info(path) -> tuple[int, int, int]:
    """(sample rate, channels, frames) from a FLAC STREAMINFO block or a RIFF/WAVE header."""
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:3] == b"ID3":                       # an ID3v2 tag in front of a FLAC stream
            size = (head[6] << 21) | (head[7] << 14) | (head[8] << 7) | head[9]
            f.seek(10 + size)
            head = f.read(4)
            if head != b"fLaC":
                raise ValueError("not a FLAC file after its ID3 tag")
            return _flac_info(f)
        if head[:4] == b"fLaC":
            f.seek(4)
            return _flac_info(f)
        if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
            return _wav_info(f)
    raise ValueError("not a FLAC or WAV file")


def _flac_info(f) -> tuple[int, int, int]:
    block = f.read(4)
    if len(block) < 4 or (block[0] & 0x7F) != 0:
        raise ValueError("FLAC stream does not start with STREAMINFO")
    length = int.from_bytes(block[1:4], "big")
    data = f.read(length)
    if length < 34 or len(data) < 34:
        raise ValueError("FLAC STREAMINFO is truncated")
    x = int.from_bytes(data[10:18], "big")
    sr = x >> 44
    channels = ((x >> 41) & 0x7) + 1
    frames = x & ((1 << 36) - 1)
    if sr <= 0:
        raise ValueError("FLAC has no sample rate")
    if frames <= 0:
        raise ValueError("FLAC does not state its length (total samples is 0)")
    return sr, channels, frames


def _wav_info(f) -> tuple[int, int, int]:
    f.seek(12)
    fmt = None
    while True:
        hdr = f.read(8)
        if len(hdr) < 8:
            raise ValueError("WAV has no data chunk")
        cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
        if cid == b"fmt ":
            body = f.read(size)
            if len(body) < 16:
                raise ValueError("WAV fmt chunk is truncated")
            _tag, channels, sr, _rate, align, _bits = struct.unpack("<HHIIHH", body[:16])
            fmt = (channels, sr, align)
            if size % 2:
                f.seek(1, 1)
        elif cid == b"data":
            if fmt is None:
                raise ValueError("WAV data chunk comes before its fmt chunk")
            channels, sr, align = fmt
            if not channels or not sr or not align:
                raise ValueError("WAV fmt chunk is invalid")
            return sr, channels, size // align
        else:
            f.seek(size + (size % 2), 1)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- validation helpers

def _match(rx: re.Pattern, value) -> bool:
    return isinstance(value, str) and rx.fullmatch(value) is not None


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _text(value, where: str, errs: list, *, required: bool = False, limit: int = TEXT_MAX):
    if value is None:
        if required:
            errs.append(f"{where}: required (plain text)")
        return None
    if not isinstance(value, str) or not value.strip():
        errs.append(f"{where}: must be non-empty text")
        return None
    if len(value) > limit:
        errs.append(f"{where}: longer than {limit} characters")
        return None
    if any(ord(ch) < 32 for ch in value):
        errs.append(f"{where}: contains control characters")
        return None
    return value.strip()


def _num(value, where: str, errs: list, lo: float, hi: float, *, required: bool = False):
    if value is None:
        if required:
            errs.append(f"{where}: required (a number from {lo:g} to {hi:g})")
        return None
    if not _is_num(value) or not lo <= value <= hi:
        errs.append(f"{where}: must be a number from {lo:g} to {hi:g}, got {value!r}")
        return None
    return value


def _int(value, where: str, errs: list, lo: int, hi: int, *, required: bool = False):
    if value is None:
        if required:
            errs.append(f"{where}: required (a whole number from {lo} to {hi})")
        return None
    if not _is_int(value) or not lo <= value <= hi:
        errs.append(f"{where}: must be a whole number from {lo} to {hi}, got {value!r}")
        return None
    return value


def _read_json(path: Path, what: str):
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise ManifestError([f"{what} is missing"], path) from None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ManifestError([f"{what} is not valid JSON ({e})"], path) from None
    try:            # an escape such as "\ud800" decodes to a lone surrogate, which no vote line could be written with
        json.dumps(data, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        raise ManifestError([f"{what} holds text that is not valid Unicode (a lone surrogate escape such as "
                             "\\ud800)"], path) from None
    return data


# ----------------------------------------------------------------------------- the public manifest

def parse_public(data, batch_dir, *, check_files: bool = True, verify_sha: bool = True) -> Manifest:
    """Validate the public manifest of the batch in `batch_dir`. Raises ManifestError with every problem."""
    batch_dir = Path(batch_dir)
    where = batch_dir / PUBLIC_NAME
    if not isinstance(data, dict):
        raise ManifestError(["the manifest must be a JSON object"], where)
    errs: list[str] = []
    extra = sorted(set(data) - PUBLIC_KEYS)
    if extra:
        errs.append(f"unknown key(s) {extra}: the public manifest may hold only {sorted(PUBLIC_KEYS)} "
                    f"(what a clip really is belongs in {HIDDEN_NAME})")
    if data.get("v") != SCHEMA_VERSION:
        errs.append(f"v: must be {SCHEMA_VERSION}")
    batch = data.get("batch")
    if not _match(BATCH_RE, batch):
        errs.append(f"batch: must match {BATCH_RE.pattern}, got {batch!r}")
    elif batch != batch_dir.name:
        errs.append(f"batch {batch!r} does not match its folder name {batch_dir.name!r}")
    mode = data.get("mode")
    if mode not in MODES:
        errs.append(f"mode: must be one of {sorted(MODES)}, got {mode!r}")
    scene = _text(data.get("scene"), "scene", errs)
    note = _text(data.get("note"), "note", errs)
    bpm = _num(data.get("bpm"), "bpm", errs, 30, 300)
    bars = _int(data.get("bars"), "bars", errs, 1, 64, required=True)
    sr = _int(data.get("sr"), "sr", errs, 8000, 192000)
    created = _text(data.get("created"), "created", errs, limit=64)
    est_min = _num(data.get("est_min"), "est_min", errs, 0, 24 * 60)

    rounds_raw = data.get("rounds")
    if not isinstance(rounds_raw, list) or not rounds_raw:
        errs.append("rounds: must be a non-empty list")
        rounds_raw = []
    lo, hi = MODES.get(mode, (1, 4))
    seen_rounds: set[str] = set()
    seen_clips: set[str] = set()
    seen_files: set[str] = set()
    rounds: list[Round] = []
    for i, r in enumerate(rounds_raw):
        w = f"rounds[{i}]"
        if not isinstance(r, dict):
            errs.append(f"{w}: must be an object")
            continue
        extra = sorted(set(r) - ROUND_KEYS)
        if extra:
            errs.append(f"{w}: unknown key(s) {extra} (a round may hold only {sorted(ROUND_KEYS)})")
        rid = r.get("id")
        if not _match(ROUND_RE, rid):
            errs.append(f"{w}.id: must match {ROUND_RE.pattern}, got {rid!r}")
        elif rid in seen_rounds:
            errs.append(f"{w}.id: {rid} appears twice")
        else:
            seen_rounds.add(rid)
        rscene = _text(r.get("scene"), f"{w}.scene", errs) or scene
        if not rscene:
            errs.append(f"{w}: no scene text (set it on the batch or on the round)")
        rbpm = _num(r.get("bpm"), f"{w}.bpm", errs, 30, 300) or bpm
        if rbpm is None:
            errs.append(f"{w}: no bpm (set it on the batch or on the round)")
        plays = r.get("plays")
        if plays is not None:
            if mode != "playlist":
                errs.append(f"{w}.plays: only playlist rounds have plays")
                plays = None
            else:
                plays = _int(plays, f"{w}.plays", errs, 1, 16)
        clips_raw = r.get("clips")
        if not isinstance(clips_raw, list):
            errs.append(f"{w}.clips: must be a list")
            clips_raw = []
        elif not lo <= len(clips_raw) <= hi:
            want = f"{lo}" if lo == hi else f"{lo} to {hi}"
            errs.append(f"{w}.clips: a {mode} round has {want} clips, this one has {len(clips_raw)}")
        clips: list[Clip] = []
        for j, c in enumerate(clips_raw):
            cw = f"{w}.clips[{j}]"
            if not isinstance(c, dict):
                errs.append(f"{cw}: must be an object")
                continue
            extra = sorted(set(c) - CLIP_KEYS)
            if extra:
                errs.append(f"{cw}: unknown key(s) {extra} (a clip may hold only {sorted(CLIP_KEYS)})")
            cid, fname, sha = c.get("id"), c.get("file"), c.get("sha256")
            ok = True
            if not _match(CLIP_RE, cid):
                errs.append(f"{cw}.id: must be opaque, matching {CLIP_RE.pattern}, got {cid!r}")
                ok = False
            elif cid in seen_clips:
                errs.append(f"{cw}.id: {cid} appears twice in the batch (each appearance gets its own id)")
                ok = False
            else:
                seen_clips.add(cid)
            if not _match(FILE_RE, fname):
                errs.append(f"{cw}.file: must be a bare opaque file name matching {FILE_RE.pattern}, got {fname!r}")
                ok = False
            elif fname in seen_files:
                errs.append(f"{cw}.file: {fname} is listed twice")
                ok = False
            else:
                seen_files.add(fname)
            if sha is not None and not _match(SHA_RE, sha):
                errs.append(f"{cw}.sha256: must be 64 lowercase hex digits")
                sha = None
            if ok:
                clips.append(Clip(id=cid, file=fname, pos=j + 1, sha256=sha))
        if _match(ROUND_RE, rid) and rscene and rbpm is not None:
            rounds.append(Round(id=rid, number=i + 1, scene=rscene, bpm=float(rbpm), plays=plays, clips=tuple(clips)))
    if errs:
        raise ManifestError(errs, where)
    m = Manifest(dir=batch_dir, batch=batch, mode=mode, bars=bars, rounds=tuple(rounds), scene=scene, note=note,
                 bpm=None if bpm is None else float(bpm), sr=sr, created=created,
                 est_min=None if est_min is None else float(est_min))
    return check_files_of(m, verify_sha=verify_sha) if check_files else m


def check_files_of(m: Manifest, *, verify_sha: bool = True) -> Manifest:
    """Every listed file exists in the batch folder (no symlinks), parses, matches its hash, and the clips of a
    round are equally long and as long as `bars` at the round's bpm. Returns the manifest with lengths filled."""
    errs: list[str] = []
    root = m.dir.resolve()
    infos: dict[str, tuple[int, int, int]] = {}
    for r in m.rounds:
        for c in r.clips:
            p = m.dir / c.file
            try:
                st = os.lstat(p)
            except FileNotFoundError:
                errs.append(f"{c.file}: listed but missing")
                continue
            if stat.S_ISLNK(st.st_mode):
                errs.append(f"{c.file}: is a symlink; clips must be plain files inside the batch folder")
                continue
            if not stat.S_ISREG(st.st_mode):
                errs.append(f"{c.file}: is not a regular file")
                continue
            if p.resolve().parent != root:
                errs.append(f"{c.file}: resolves outside the batch folder")
                continue
            if not 0 < st.st_size <= MAX_FILE_BYTES:
                errs.append(f"{c.file}: size {st.st_size} bytes is out of range")
                continue
            try:
                infos[c.id] = audio_info(p)
            except (OSError, ValueError, struct.error) as e:
                errs.append(f"{c.file}: {e}")
                continue
            if verify_sha and c.sha256 and sha256_file(p) != c.sha256:
                errs.append(f"{c.file}: sha256 does not match the manifest")
    rates = sorted({v[0] for v in infos.values()})
    if len(rates) > 1:
        errs.append(f"clips use different sample rates {rates}; a batch uses one rate")
    if m.sr is not None and rates and rates != [m.sr]:
        errs.append(f"sr is {m.sr} but the clips are at {rates}")
    for r in m.rounds:
        lengths = sorted({infos[c.id][2] for c in r.clips if c.id in infos})
        if len(lengths) > 1:
            errs.append(f"round {r.id}: its clips differ in length ({lengths} samples); "
                        "a round's clips must be equally long so they stay in sync")
        expected = m.bars * 240.0 / r.bpm
        for c in r.clips:
            if c.id not in infos:
                continue
            sr_c, _, frames = infos[c.id]
            dur = frames / sr_c
            if abs(dur - expected) > DURATION_TOL * expected:
                errs.append(f"{c.file}: {dur:.2f} s long, but {m.bars} bars at {r.bpm:g} BPM is {expected:.2f} s "
                            "(a clip is exactly one pass of the loop)")
    if errs:
        raise ManifestError(errs, m.dir / PUBLIC_NAME)
    rounds = tuple(replace(r, clips=tuple(replace(c, sr=infos[c.id][0], channels=infos[c.id][1], frames=infos[c.id][2])
                                           for c in r.clips)) for r in m.rounds)
    return replace(m, rounds=rounds, sr=m.sr or (rates[0] if rates else None))


# ----------------------------------------------------------------------------- the hidden manifest

def _scalar(x) -> bool:
    return x is None or isinstance(x, (str, bool)) or _is_num(x)


def parse_hidden(data, m: Manifest) -> Hidden:
    """Validate manifest.hidden.json against the public manifest, then check that nothing leaks."""
    where = m.dir / HIDDEN_NAME
    if not isinstance(data, dict):
        raise ManifestError(["the hidden manifest must be a JSON object"], where)
    errs: list[str] = []
    if data.get("v") != SCHEMA_VERSION:
        errs.append(f"v: must be {SCHEMA_VERSION}")
    if data.get("batch") != m.batch:
        errs.append(f"batch: {data.get('batch')!r} does not match the public manifest's {m.batch!r}")
    category = _text(data.get("category"), "category", errs, limit=80)
    try:
        serve_from = serve_from_date(data.get("serve_from"))
    except ValueError as err:
        errs.append(str(err))
        serve_from = None
    rounds_raw = data.get("rounds")
    if not isinstance(rounds_raw, dict):
        errs.append("rounds: must be an object mapping each round id to what it really is")
        rounds_raw = {}
    extra = sorted(set(rounds_raw) - {r.id for r in m.rounds})
    if extra:
        errs.append(f"rounds: {extra} are not in the public manifest")
    out: dict[str, HiddenRound] = {}
    for r in m.rounds:
        w = f"rounds.{r.id}"
        hr = rounds_raw.get(r.id)
        if not isinstance(hr, dict):
            errs.append(f"{w}: missing (every public round needs an entry)")
            continue
        kind = hr.get("kind", "none")
        if kind not in KINDS:
            errs.append(f"{w}.kind: must be one of {list(KINDS)}, got {kind!r}")
            kind = "none"
        of = hr.get("of")
        if kind == "repeat":
            if not _match(ROUND_RE, of):
                errs.append(f"{w}.of: a repeat names the round id of its first presentation")
            elif of == r.id:
                errs.append(f"{w}.of: a repeat must not reuse the id of the round it repeats")
        elif of is not None and not _match(ROUND_RE, of):
            errs.append(f"{w}.of: must be a round id or null")
        anchor = hr.get("anchor_clip")
        if kind == "anchor":
            if anchor not in r.ids:
                errs.append(f"{w}.anchor_clip: an anchor names the opaque id of the broken copy in this round")
        elif anchor is not None:
            errs.append(f"{w}.anchor_clip: only anchor rounds have one")
            anchor = None
        brief_id = _text(hr.get("brief_id"), f"{w}.brief_id", errs, limit=80)
        rcat = _text(hr.get("category"), f"{w}.category", errs, limit=80)
        clips = hr.get("clips")
        if not isinstance(clips, dict):
            errs.append(f"{w}.clips: must map each opaque clip id to what it really is")
            clips = {}
        if set(clips) != set(r.ids):
            errs.append(f"{w}.clips: must cover exactly the round's clips {list(r.ids)}")
        reals = []
        for oid, info in clips.items():
            cw = f"{w}.clips.{oid}"
            if not isinstance(info, dict):
                errs.append(f"{cw}: must be an object")
                continue
            if not _match(REAL_RE, info.get("clip")):
                errs.append(f"{cw}.clip: the real clip id must match {REAL_RE.pattern}")
            else:
                reals.append(info["clip"])
            bad = sorted(k for k, v in info.items() if not _scalar(v))
            if bad:
                errs.append(f"{cw}: {bad} must be plain values (text, numbers, true/false or null)")
            if m.mode == "shootout" and not (isinstance(info.get("option"), str) and info["option"].strip()):
                errs.append(f"{cw}.option: shootout clips name the option they stand for")
        if len(reals) != len(set(reals)):
            errs.append(f"{w}: the same real clip appears twice in one round")
        if m.mode == "shootout" and not (rcat or category):
            errs.append(f"{w}: a shootout round needs a category (on the round or the batch)")
        out[r.id] = HiddenRound(id=r.id, kind=kind, of=of, anchor_clip=anchor, brief_id=brief_id,
                                category=rcat or category, clips={k: dict(v) for k, v in clips.items()
                                                                  if isinstance(v, dict)})
    if not errs:
        errs.extend(leak_errors(m, out, category))
    if errs:
        raise ManifestError(errs, where)
    return Hidden(batch=m.batch, category=category, rounds=out, serve_from=serve_from)


_WORD = re.compile(r"[a-z0-9]+")
_HEX = frozenset("0123456789abcdef")


def _words(value) -> set[str]:
    """Words of a hidden value that must never show up in an opaque id. Short all-hex words are skipped,
    because random hex ids contain them by chance."""
    out = set()
    for w in _WORD.findall(str(value).lower()):
        if len(w) < 3 or (set(w) <= _HEX and len(w) < LEAK_WINDOW):
            continue
        out.add(w)
    return out


def leak_errors(m: Manifest, hidden_rounds: dict, category: str | None) -> list[str]:
    """Opaque ids and file names must not contain what a clip is: any stretch of its real id, or a word of its
    option, variant or source names, or of the brief id. The batch note (shown on the page) must not name a
    shootout option. Hidden values that are themselves this batch's opaque ids (e.g. `broken_of`) are fine."""
    public_ids = {r.id for r in m.rounds} | {c.id for r in m.rounds for c in r.clips}
    secrets: dict[str, str] = {}
    options: dict[str, str] = {}
    cat_words: set[str] = _words(category or "")
    for hr in hidden_rounds.values():
        cat_words |= _words(hr.category or "")
        for w in _words(hr.brief_id or ""):
            secrets.setdefault(w, f"the brief id of round {hr.id}")
        for oid, info in hr.clips.items():
            real = str(info.get("clip", ""))[1:]
            for k in range(0, max(0, len(real) - LEAK_WINDOW + 1)):
                secrets.setdefault(real[k:k + LEAK_WINDOW], f"the real id of clip {oid}")
            for key, val in info.items():
                if key == "clip":
                    continue
                for w in _words(val):
                    secrets.setdefault(w, f"{key} of clip {oid}")
                    if key == "option":
                        options.setdefault(w, f"the option of clip {oid}")
    for w in public_ids:
        secrets.pop(w, None)
    errs = []
    labelled = []
    for r in m.rounds:
        labelled.append((f"round id {r.id}", r.id))
        for c in r.clips:
            labelled.append((f"clip id {c.id}", c.id))
            labelled.append((f"file {c.file}", c.file))
    for label, text in labelled:
        low = text.lower()
        for w, src in secrets.items():
            if w in low:
                errs.append(f"{label} contains {w!r} ({src}); opaque ids and file names must not leak what a "
                            "clip is")
    if m.note:
        low = m.note.lower()
        for w, src in options.items():
            if w not in cat_words and re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", low):
                errs.append(f"the note names {w!r} ({src}); the page must not name the options it compares")
    return errs


# ----------------------------------------------------------------------------- loading

def load_batch(batch_dir, *, check_files: bool = True, verify_sha: bool = True) -> tuple[Manifest, Hidden]:
    """Load and validate both manifests of a batch folder."""
    d = Path(batch_dir)
    if not d.is_dir():
        raise ManifestError([f"{d} is not a batch folder"])
    pub = _read_json(d / PUBLIC_NAME, PUBLIC_NAME)
    m = parse_public(pub, d, check_files=check_files, verify_sha=verify_sha)
    hid = _read_json(d / HIDDEN_NAME, f"{HIDDEN_NAME} (it says which rounds are repeats or anchors)")
    return m, parse_hidden(hid, m)


def load_public(batch_dir) -> Manifest:
    """The public manifest alone, without file checks (for listing batches)."""
    d = Path(batch_dir)
    return parse_public(_read_json(d / PUBLIC_NAME, PUBLIC_NAME), d, check_files=False)


# ----------------------------------------------------------------------------- the synthetic demo batch

DEMO_K = {"shootout": 4, "keep3": 3, "bws4": 4, "triage": 1, "ab": 2, "playlist": 1}
DEMO_ROUNDS = {"shootout": 4, "keep3": 3, "bws4": 4, "triage": 6, "ab": 4, "playlist": 5}
DEMO_OPTIONS = ("rhodes", "piano", "guitar", "organ")
DEMO_FREQS = (220.0, 277.18, 329.63, 392.0, 246.94, 293.66)


def _hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def _synth(frames: int, sr: int, bpm: float, beats: int, freq: float | None, seed: int):
    """One loop: a short decaying burst on every beat, louder on each downbeat. freq=None gives noise bursts."""
    beat = 60.0 / bpm
    burst_len = max(1, int(min(beat, 0.35) * sr))
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        t = np.arange(burst_len) / sr
        env = np.exp(-t / 0.09)
        burst = (np.random.default_rng(seed).standard_normal(burst_len) * 0.5 if freq is None
                 else np.sin(2 * np.pi * freq * t)) * env
        y = np.zeros(frames)
        for b in range(beats):
            s = int(round(b * beat * sr))
            if s >= frames:
                break
            e = min(frames, s + burst_len)
            y[s:e] += (0.5 if b % 4 == 0 else 0.3) * burst[:e - s]
        return np.clip(y, -1.0, 1.0)
    rng = random.Random(seed)
    burst = [(rng.gauss(0, 0.5) if freq is None else math.sin(2 * math.pi * freq * i / sr)) * math.exp(-i / sr / 0.09)
             for i in range(burst_len)]
    y = [0.0] * frames
    for b in range(beats):
        s = int(round(b * beat * sr))
        amp = 0.5 if b % 4 == 0 else 0.3
        for i in range(burst_len):
            if s + i >= frames:
                break
            y[s + i] += amp * burst[i]
    return [max(-1.0, min(1.0, v)) for v in y]


def _write_audio(path: Path, samples, sr: int, fmt: str) -> None:
    if fmt == "flac":
        import soundfile as sf
        import numpy as np
        sf.write(str(path), np.asarray(samples, dtype=np.float32), sr, format="FLAC", subtype="PCM_16")
        return
    import wave
    try:
        import numpy as np
        pcm = (np.asarray(samples) * 32767.0).round().astype("<i2").tobytes()
    except ImportError:
        pcm = b"".join(struct.pack("<h", int(round(v * 32767.0))) for v in samples)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build_demo(batch_dir, mode: str, *, rounds: int | None = None, bpm: float = 90.0, bars: int = 8,
               sr: int = 44100, seed: int = 596, fmt: str = "flac", force: bool = False) -> Path:
    """Write a synthetic batch (sine or noise clips plus both manifests) into `batch_dir`, whose name is the
    batch id. Round 2 is a broken anchor and the last round repeats round 1 with positions reversed."""
    d = Path(batch_dir)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    if not _match(BATCH_RE, d.name):
        raise ValueError(f"the folder name {d.name!r} is the batch id and must match {BATCH_RE.pattern}")
    if fmt not in ("flac", "wav"):
        raise ValueError("fmt must be flac or wav")
    if (d / PUBLIC_NAME).exists() and not force:
        raise FileExistsError(f"{d / PUBLIC_NAME} exists (pass force=True / --force to overwrite)")
    if fmt == "flac":
        try:
            import soundfile  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            fmt = "wav"
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"{seed}|{d.name}|{mode}")
    n_rounds = rounds or DEMO_ROUNDS[mode]
    if n_rounds < 1:
        raise ValueError("rounds must be at least 1")
    k = DEMO_K[mode]
    frames = int(round(bars * 240.0 / bpm * sr))
    anchor_at = 1 if n_rounds >= 2 else None
    repeat_at = n_rounds - 1 if n_rounds >= 3 else None

    plan: list[list[tuple[int, int | None]]] = []      # per round: (variant, position it is a broken copy of)
    for i in range(n_rounds):
        if i == repeat_at:
            plan.append(list(reversed(plan[0])))
            continue
        variants = [i % len(DEMO_FREQS)] if k == 1 else rng.sample(range(len(DEMO_OPTIONS)), k)
        row: list[tuple[int, int | None]] = [(v, None) for v in variants]
        if i == anchor_at:
            if k >= 2:
                victim = rng.randrange(k)
                src = (victim + 1) % k
                row[victim] = (row[src][0], src)
            else:
                row[0] = (row[0][0], -1)
        plan.append(row)

    audio_cache: dict[tuple[int, bool], object] = {}

    def audio(v: int, broken: bool):
        key = (v, broken)
        if key not in audio_cache:
            audio_cache[key] = _synth(frames, sr, bpm, bars * 4, None if broken else DEMO_FREQS[v], seed + v)
        return audio_cache[key]

    def real_id(v: int, broken: bool) -> str:
        return "c" + hashlib.sha256(f"demo|{mode}|{v}|{'broken' if broken else 'ok'}".encode()).hexdigest()[:16]

    pub_rounds, hid_rounds, round_ids = [], {}, []
    for i, row in enumerate(plan):
        rid = "r" + _hex(rng, 8)
        round_ids.append(rid)
        clips_pub, clips_hid = [], {}
        oids = ["k" + _hex(rng, 10) for _ in row]
        for pos, ((v, broken_of), oid) in enumerate(zip(row, oids)):
            broken = broken_of is not None
            fname = f"{oid}.{fmt}"
            _write_audio(d / fname, audio(v, broken), sr, fmt)
            clips_pub.append({"id": oid, "file": fname, "sha256": sha256_file(d / fname)})
            info = {"clip": real_id(v, broken), "what": "demo broken copy" if broken else f"demo tone {v + 1}",
                    "source": "demo"}
            if mode == "shootout":
                info["option"] = DEMO_OPTIONS[v % len(DEMO_OPTIONS)]
            if broken and broken_of is not None and broken_of >= 0:
                info["broken_of"] = oids[broken_of]
            clips_hid[oid] = info
        entry = {"kind": "none", "brief_id": f"demo-brief-{i % max(1, n_rounds - 1) + 1}", "clips": clips_hid}
        if i == anchor_at:
            entry["kind"] = "anchor"
            entry["anchor_clip"] = next(oid for (v, b), oid in zip(row, oids) if b is not None)
        if i == repeat_at:
            entry["kind"] = "repeat"
            entry["of"] = round_ids[0]
            entry["brief_id"] = hid_rounds[round_ids[0]]["brief_id"]
        if mode == "shootout":
            entry["category"] = "keys"
        hid_rounds[rid] = entry
        pr = {"id": rid, "clips": clips_pub}
        if mode == "playlist":
            pr["plays"] = 2 + i % 3
        pub_rounds.append(pr)

    loop_s = bars * 240.0 / bpm                    # one pass of the loop, in seconds
    secs_per_round = (k * loop_s * 0.75 if k > 1 else loop_s * 1.25) + 8     # taste_pipeline §8 listening budget
    if mode == "playlist":
        secs_per_round = 3 * loop_s
    public = {"v": SCHEMA_VERSION, "batch": d.name, "mode": mode, "scene": "demo batch: synthetic test tones",
              "note": f"Demo {mode} batch for checking the page. The clips are test tones.",
              "bpm": bpm, "bars": bars, "sr": sr, "created": datetime.now().astimezone().isoformat(timespec="seconds"),
              "est_min": round(n_rounds * secs_per_round / 60.0, 1), "rounds": pub_rounds}
    hidden = {"v": SCHEMA_VERSION, "batch": d.name, "rounds": hid_rounds}
    if mode == "shootout":
        hidden["category"] = "keys"
    _atomic_json(d / HIDDEN_NAME, hidden)
    _atomic_json(d / PUBLIC_NAME, public)           # written last: a batch is complete once this exists
    load_batch(d)                                   # the demo must pass its own validation
    return d


# ----------------------------------------------------------------------------- CLI

def _main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.ui.manifest", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    dp = sub.add_parser("demo", help="write a synthetic batch (sine or noise clips plus both manifests)")
    dp.add_argument("dir", help="the batch folder to write; its name is the batch id")
    dp.add_argument("--mode", required=True, choices=sorted(MODES))
    dp.add_argument("--rounds", type=int)
    dp.add_argument("--bpm", type=float, default=90.0)
    dp.add_argument("--bars", type=int, default=8)
    dp.add_argument("--sr", type=int, default=44100)
    dp.add_argument("--seed", type=int, default=596)
    dp.add_argument("--format", choices=("flac", "wav"), default="flac")
    dp.add_argument("--force", action="store_true", help="overwrite an existing demo batch")
    cp = sub.add_parser("check", help="validate a batch folder")
    cp.add_argument("dir")
    cp.add_argument("--no-sha", action="store_true", help="skip the sha256 check of each clip")
    a = p.parse_args(argv)
    try:
        if a.cmd == "demo":
            d = build_demo(a.dir, a.mode, rounds=a.rounds, bpm=a.bpm, bars=a.bars, sr=a.sr, seed=a.seed,
                           fmt=a.format, force=a.force)
            m, _ = load_batch(d)
            n = sum(len(r.clips) for r in m.rounds)
            print(f"demo {m.mode} batch {m.batch}: {len(m.rounds)} rounds, {n} clips in {d}")
            print("serve it with votes in a scratch folder (never data/votes):")
            print(f"  LOFI_VOTES=\"$(mktemp -d)\" make rate BATCH={d}")
            return 0
        m, h = load_batch(a.dir, verify_sha=not a.no_sha)
        kinds = {}
        for hr in h.rounds.values():
            kinds[hr.kind] = kinds.get(hr.kind, 0) + 1
        print(f"ok: {m.batch} ({m.mode}), {len(m.rounds)} rounds, "
              f"{sum(len(r.clips) for r in m.rounds)} clips, {kinds}")
        return 0
    except (ManifestError, ValueError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(_main())
