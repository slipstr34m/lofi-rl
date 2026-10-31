"""Vote events for the rating UI (docs/RATING.md; docs/design/taste_pipeline.md §1-§2).

- `check_event`: what the page may send. The page knows only opaque ids; every field is checked per event type
  and batch mode, and unknown fields are refused.
- `Ledger`: one batch's log, `$LOFI_VOTES/<batch>.jsonl`. The server labels each event (its own clock, the
  session and micro-session, the real clip ids, the hidden round kind), appends it as one JSON line and fsyncs
  it. `$LOFI_VOTES/state.json` holds the cursor so a session resumes; the log stays the source of truth, and
  the queue (which round comes next) is derived from it whenever the server starts.
- Several servers may share one votes folder (one batch each). They share state.json and the session clock,
  and each reads, updates and rewrites them under `folder_lock`, so no server undoes another's update and no
  session id is used twice.
- `read_events`, `active_votes`, `read_all`: the reader the audit uses. A retract never deletes a line: it is
  a later event that takes an earlier vote back.

Stdlib only.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import re
import secrets
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lofi_rl.ui.manifest import Hidden, Manifest

SCHEMA_VERSION = 1
VOTE_TYPES = ("shootout", "keep3", "bws4", "triage", "ab", "playlist")   # one per batch mode
EVENT_TYPES = VOTE_TYPES + ("session", "abandon", "retract")
TIERS = ("toss", "fine", "good", "great")
NEG_TAGS = ("fake", "no_groove", "sour", "boring")     # a negative tag goes with the worst pick
POS_TAGS = ("nodding", "sax_line", "chords")           # a positive tag goes with the best pick
TAGS = NEG_TAGS + POS_TAGS
MARKS = ("heart", "cross", "none")                     # playlist
# `playing`: the page sends it after 5 minutes of uninterrupted playback with no other event, so a round left
# looping is not mistaken for a gap (a micro-session ends after 10 minutes without events).
SESSION_WHAT = ("start", "round_start", "pause", "resume", "devicechange", "nudge", "playing", "end")
ABANDON_WHY = ("esc", "pagehide", "undo", "no_commit", "error", "other")
T_MS_KEYS = ("first_play", "best", "worst", "no_best", "no_worst", "all_bad", "tag", "tier", "pick", "mark",
             "commit", "abandon")
AUDIO_KEYS = ("sr", "latency_ms", "base_latency_ms")

EVENT_ID_RE = re.compile(r"e[0-9a-f]{12,32}")
DEVICE_RE = re.compile(r"[a-z0-9_-]{1,32}")
STATE_NAME = "state.json"
MAX_MS = 10 ** 9

LISTEN_KEYS = frozenset({"clips", "last_heard", "switches", "play_order", "autoplay_done", "t_ms", "device", "audio"})
PICK_KEYS = {
    "bws4": frozenset({"best", "worst", "no_best", "no_worst", "all_bad", "tag"}),
    "shootout": frozenset({"best", "worst", "no_best", "no_worst", "all_bad"}),
    "keep3": frozenset({"best", "all_bad"}),            # best = the loop he keeps; all_bad = "none acceptable"
    "triage": frozenset({"tier"}),
    "ab": frozenset({"pick"}),
    "playlist": frozenset({"mark", "plays"}),
}
ALLOWED = {
    **{t: frozenset({"type", "event_id", "round"}) | LISTEN_KEYS | PICK_KEYS[t] for t in VOTE_TYPES},
    "abandon": frozenset({"type", "event_id", "round", "why"}) | LISTEN_KEYS,
    "retract": frozenset({"type", "event_id", "target", "why"}),
    "session": frozenset({"type", "event_id", "what", "round", "detail", "device", "audio"}),
}


class EventError(ValueError):
    """The page sent something malformed (HTTP 400)."""


class Conflict(EventError):
    """Well formed, but it does not fit the batch's current state (HTTP 409)."""


class LogBusy(RuntimeError):
    """Another rating server already holds this batch's log."""


def new_event_id() -> str:
    return "e" + secrets.token_hex(12)


# ----------------------------------------------------------------------------- checking what the page sends

def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _bool(raw: dict, key: str) -> bool:
    v = raw.get(key, False)
    if v is None:
        return False
    if not isinstance(v, bool):
        raise EventError(f"{key}: must be true or false")
    return v


def plain_text(s: str) -> bool:
    """No control characters, and encodable as UTF-8: json.loads turns "\\ud800" into a lone surrogate, which
    would pass every other check and then fail when the line is written."""
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return not any(ord(ch) < 32 for ch in s)


def _short_text(v, key: str, limit: int = 200):
    if v is None:
        return None
    if not isinstance(v, str) or len(v) > limit or not plain_text(v):
        raise EventError(f"{key}: must be plain text of at most {limit} characters")
    return v


def _check_clips(clips, r, bars: int) -> list[dict]:
    if not isinstance(clips, list) or len(clips) != len(r.clips):
        raise EventError(f"clips: must list the round's {len(r.clips)} clip(s) in order")
    out = []
    for i, (got, want) in enumerate(zip(clips, r.clips)):
        w = f"clips[{i}]"
        if not isinstance(got, dict):
            raise EventError(f"{w}: must be an object")
        extra = sorted(set(got) - {"clip", "pos", "heard_bars", "heard_ms"})
        if extra:
            raise EventError(f"{w}: unknown field(s) {extra}")
        if got.get("clip") != want.id:
            raise EventError(f"{w}.clip: expected {want.id} (the clip at position {want.pos})")
        if got.get("pos") != want.pos:
            raise EventError(f"{w}.pos: expected {want.pos}")
        hb = got.get("heard_bars")
        if not isinstance(hb, str) or len(hb) != bars or set(hb) - {"0", "1"}:
            raise EventError(f"{w}.heard_bars: must be {bars} characters of 0 and 1 (bar 1 first)")
        hm = got.get("heard_ms", 0)
        if not _is_num(hm) or not 0 <= hm <= MAX_MS:
            raise EventError(f"{w}.heard_ms: must be a number of milliseconds")
        out.append({"clip": want.id, "pos": want.pos, "heard_bars": hb, "heard_ms": int(round(hm))})
    return out


def _check_device(raw: dict, out: dict) -> None:
    dev = raw.get("device")
    if dev is not None and not (isinstance(dev, str) and DEVICE_RE.fullmatch(dev)):
        raise EventError(f"device: must match {DEVICE_RE.pattern}")
    out["device"] = dev
    audio = raw.get("audio")
    if audio is None:
        out["audio"] = None
        return
    if not isinstance(audio, dict) or set(audio) - set(AUDIO_KEYS):
        raise EventError(f"audio: an object with at most {list(AUDIO_KEYS)}")
    for k, v in audio.items():
        if v is not None and (not _is_num(v) or not 0 <= v <= 10 ** 6):
            raise EventError(f"audio.{k}: must be a number")
    out["audio"] = {k: audio[k] for k in AUDIO_KEYS if audio.get(k) is not None}


def _check_listening(raw: dict, r, m: Manifest, out: dict, *, need_clips: bool) -> None:
    clips = raw.get("clips")
    if clips is None:
        if need_clips:
            raise EventError("clips: a vote lists every clip of its round with what was heard")
    else:
        out["clips"] = _check_clips(clips, r, m.bars)
    lh = raw.get("last_heard")
    if lh is not None and lh not in r.ids:
        raise EventError("last_heard: not a clip of this round")
    out["last_heard"] = lh
    sw = raw.get("switches", 0)
    if not _is_int(sw) or not 0 <= sw <= 100_000:
        raise EventError("switches: must be a whole number")
    out["switches"] = sw
    po = raw.get("play_order", [])
    if (not isinstance(po, list) or any(not _is_int(x) or not 1 <= x <= len(r.clips) for x in po)
            or len(set(po)) != len(po)):
        raise EventError(f"play_order: distinct positions 1 to {len(r.clips)}")
    out["play_order"] = po
    out["autoplay_done"] = _bool(raw, "autoplay_done")
    t = raw.get("t_ms") or {}
    if not isinstance(t, dict) or set(t) - set(T_MS_KEYS):
        raise EventError(f"t_ms: an object with keys from {list(T_MS_KEYS)}")
    tm = {}
    for k, v in t.items():
        if not _is_num(v) or not 0 <= v <= MAX_MS:
            raise EventError(f"t_ms.{k}: must be a number of milliseconds")
        tm[k] = int(round(v))
    out["t_ms"] = tm
    _check_device(raw, out)


def _check_picks(t: str, raw: dict, r, out: dict) -> None:
    ids = r.ids

    def clip_or_none(key):
        v = raw.get(key)
        if v is not None and v not in ids:
            raise EventError(f"{key}: not a clip of this round")
        return v

    if t in ("bws4", "shootout"):
        best, worst = clip_or_none("best"), clip_or_none("worst")
        nb, nw, ab = _bool(raw, "no_best"), _bool(raw, "no_worst"), _bool(raw, "all_bad")
        if best is not None and best == worst:
            raise EventError("best and worst must be different clips")
        if best is not None and nb:
            raise EventError("a best pick and 'no clear best' exclude each other")
        if worst is not None and nw:
            raise EventError("a worst pick and 'no clear worst' exclude each other")
        if ab and best is not None:
            raise EventError("'all bad' and a best pick exclude each other")
        if not ab and (best is None and not nb or worst is None and not nw):
            raise EventError("incomplete: pick a best (or 'no clear best') and a worst (or 'no clear worst'), "
                             "or say 'all bad'")
        out.update(best=best, worst=worst, no_best=nb, no_worst=nw, all_bad=ab)
        if t == "bws4":
            tag = raw.get("tag")
            if tag is not None:
                if tag not in TAGS:
                    raise EventError(f"tag: one of {list(TAGS)}")
                if tag in NEG_TAGS and worst is None:
                    raise EventError(f"the tag {tag!r} goes with the worst pick, and there is none")
                if tag in POS_TAGS and best is None:
                    raise EventError(f"the tag {tag!r} goes with the best pick, and there is none")
            out["tag"] = tag
    elif t == "keep3":
        best, ab = clip_or_none("best"), _bool(raw, "all_bad")
        if (best is None) == (not ab):
            raise EventError("keep one loop, or say 'none acceptable' (not both)")
        out.update(best=best, all_bad=ab)
    elif t == "triage":
        tier = raw.get("tier")
        if tier not in TIERS:
            raise EventError(f"tier: one of {list(TIERS)}")
        out["tier"] = tier
    elif t == "ab":
        pick = raw.get("pick")
        if pick not in ids:
            raise EventError("pick: the clip picked (forced choice)")
        out["pick"] = pick
    elif t == "playlist":
        mark = raw.get("mark", "none")
        if mark not in MARKS:
            raise EventError(f"mark: one of {list(MARKS)}")
        plays = raw.get("plays", 0)
        if not _is_int(plays) or not 0 <= plays <= 64:
            raise EventError("plays: how many full passes played (a whole number)")
        out.update(mark=mark, plays=plays)


def check_event(raw, m: Manifest) -> dict:
    """Validate one event from the page against the batch's public manifest. Returns a clean copy."""
    if not isinstance(raw, dict):
        raise EventError("an event must be a JSON object")
    t = raw.get("type")
    if t not in EVENT_TYPES:
        raise EventError(f"type: one of {list(EVENT_TYPES)}, got {t!r}")
    if t in VOTE_TYPES and t != m.mode:
        raise EventError(f"this is a {m.mode} batch, so it takes {m.mode} votes, not {t}")
    extra = sorted(set(raw) - ALLOWED[t])
    if extra:
        raise EventError(f"{t}: unknown field(s) {extra}")
    out: dict = {"type": t}
    eid = raw.get("event_id")
    if eid is not None:
        if not (isinstance(eid, str) and EVENT_ID_RE.fullmatch(eid)):
            raise EventError(f"event_id: must match {EVENT_ID_RE.pattern}")
        out["event_id"] = eid
    if t == "retract":
        target = raw.get("target")
        if not (isinstance(target, str) and EVENT_ID_RE.fullmatch(target)):
            raise EventError("target: the event_id of the vote to take back")
        out["target"] = target
        out["why"] = _short_text(raw.get("why"), "why")
        return out
    if t == "session":
        what = raw.get("what")
        if what not in SESSION_WHAT:
            raise EventError(f"what: one of {list(SESSION_WHAT)}")
        rid = raw.get("round")
        if rid is not None and m.round(rid) is None:
            raise EventError("round: not a round of this batch")
        if what == "round_start" and rid is None:
            raise EventError("round_start needs the round")
        out.update(what=what, round=rid, detail=_short_text(raw.get("detail"), "detail"))
        _check_device(raw, out)
        return out
    r = m.round(raw.get("round"))
    if r is None:
        raise EventError("round: not a round of this batch")
    out["round"] = r.id
    if t == "abandon":
        why = raw.get("why", "other")
        if why not in ABANDON_WHY:
            raise EventError(f"why: one of {list(ABANDON_WHY)}")
        out["why"] = why
        _check_listening(raw, r, m, out, need_clips=False)
        return out
    _check_picks(t, raw, r, out)
    _check_listening(raw, r, m, out, need_clips=True)
    out["t_ms"] = {k: v for k, v in out["t_ms"].items() if k in _marks_made(out)}
    return out


def _marks_made(vote: dict) -> set[str]:
    """The t_ms keys a vote may carry: the time to each mark it holds. A mark that was set and then cleared
    (T after a best, a tag pressed twice) leaves no time behind."""
    keys = {"first_play", "commit"}
    for k in ("best", "worst", "tag", "tier", "pick"):
        if vote.get(k) is not None:
            keys.add(k)
    for k in ("no_best", "no_worst", "all_bad"):
        if vote.get(k):
            keys.add(k)
    if vote.get("mark") not in (None, "none"):
        keys.add("mark")
    return keys


# ----------------------------------------------------------------------------- the queue, derived from the log

@dataclass
class Derived:
    order: list           # round ids in presentation order (abandoned rounds move to the end)
    active: dict          # round id -> event_id of its standing vote
    pending: list         # rounds without a standing vote, in order
    current: str | None
    done: bool
    last_vote: str | None         # event_id of the most recent standing vote (Backspace takes it back)
    last_vote_round: str | None
    open_round: str | None        # started (round_start) but neither voted nor abandoned


def derive(m: Manifest, events: list) -> Derived:
    order = [r.id for r in m.rounds]
    known = set(order)
    active: dict[str, str] = {}
    votes: dict[str, dict] = {}
    seq: list[str] = []
    open_round = None
    for e in events:
        t = e.get("type")
        rid = e.get("round")
        if t in VOTE_TYPES and rid in known:
            active[rid] = e["event_id"]
            votes[e["event_id"]] = e
            seq.append(e["event_id"])
            if open_round == rid:
                open_round = None
        elif t == "retract":
            v = votes.get(e.get("target"))
            if v is not None and active.get(v["round"]) == v["event_id"]:
                del active[v["round"]]
        elif t == "abandon" and rid in known:
            if rid not in active:
                order.remove(rid)
                order.append(rid)
            if open_round == rid:
                open_round = None
        elif t == "session" and e.get("what") == "round_start" and rid in known and rid not in active:
            open_round = rid
    pending = [rid for rid in order if rid not in active]
    standing = set(active.values())
    last = next((eid for eid in reversed(seq) if eid in standing), None)
    return Derived(order=order, active=active, pending=pending, current=pending[0] if pending else None,
                   done=not pending, last_vote=last, last_vote_round=votes[last]["round"] if last else None,
                   open_round=open_round if open_round in pending else None)


# ----------------------------------------------------------------------------- files

def fsync_dir(path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write(path: Path, data: bytes) -> None:
    """Write via a temp file in the same folder, fsync, then rename over the target."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    fsync_dir(path.parent)


@contextmanager
def folder_lock(path):
    """An exclusive flock on a folder (released when its descriptor closes). Every server sharing $LOFI_VOTES
    holds it on that folder while it reads, updates and rewrites state.json and logs an event, so two servers
    never undo each other's update or start the same session id."""
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def log_line(record: dict) -> bytes:
    """One event as one line of the log. Raises before anything is written if the record cannot be encoded."""
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


class VoteLog:
    """Append-only JSONL: one line per event, fsynced before append() returns. Holds an exclusive lock on the
    file for its lifetime, so two servers never write one batch.

    A write that fails part-way (a full disk, a quota) is cut back to where the file ended, so a retry starts on
    a clean line. If even that fails, or the file ends mid-line after a crash, the next write starts on a fresh
    line: the fragment stays on a line of its own and is skipped when read."""

    def __init__(self, path):
        self.path = Path(path)
        existed = self.path.exists()
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise LogBusy(f"{self.path} is in use by another rating server") from None
        self.fd: int | None = fd
        try:
            if not existed:
                fsync_dir(self.path.parent)
            self._torn = not self._ends_clean()        # a crash mid-line: the next write starts a new line
        except BaseException:
            self.close()
            raise

    def _ends_clean(self) -> bool:
        """True when the log is empty or its last byte ends a line."""
        size = os.fstat(self.fd).st_size
        if not size:
            return True
        with open(self.path, "rb") as f:
            f.seek(size - 1)
            return f.read(1) == b"\n"

    def _write(self, data: bytes) -> None:
        if self.fd is None:
            raise RuntimeError("the vote log is closed")
        if self._torn and not self._ends_clean():
            data = b"\n" + data
        size = os.fstat(self.fd).st_size
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(self.fd, view):]
            os.fsync(self.fd)
        except BaseException:
            self._cut_back(size)
            raise
        self._torn = False

    def _cut_back(self, size: int) -> None:
        """Undo a failed write: truncate the log to `size` bytes again. If that fails too, remember that the file
        may end mid-line, so the next write starts on a line of its own."""
        try:
            os.ftruncate(self.fd, size)
            os.fsync(self.fd)
        except OSError:
            self._torn = True

    def append(self, record: dict) -> None:
        self.append_many([record])

    def append_many(self, records) -> None:
        """Several events in one write: all of them land, or none (each is encoded before anything is written)."""
        self._write(b"".join(log_line(r) for r in records))

    def close(self) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None


def read_events(path) -> tuple[list[dict], list[str]]:
    """All events of one log in file order, plus a note for each unreadable line (skipped, never fatal)."""
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return [], []
    events, problems = [], []
    for n, line in enumerate(raw.split(b"\n"), 1):
        if not line.strip():
            continue
        try:
            e = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            problems.append(f"line {n}: unreadable, skipped")
            continue
        if not isinstance(e, dict) or e.get("type") not in EVENT_TYPES or not isinstance(e.get("event_id"), str):
            problems.append(f"line {n}: not an event, skipped")
            continue
        events.append(e)
    return events, problems


def active_votes(events: list) -> dict[str, dict]:
    """round id -> its standing vote (retracted votes removed)."""
    act: dict[str, dict] = {}
    by_id: dict[str, dict] = {}
    for e in events:
        t = e.get("type")
        if t in VOTE_TYPES and isinstance(e.get("round"), str):
            act[e["round"]] = e
            by_id[e["event_id"]] = e
        elif t == "retract":
            v = by_id.get(e.get("target"))
            if v is not None and act.get(v["round"]) is v:
                del act[v["round"]]
    return act


def read_all(votes_dir) -> dict[str, list[dict]]:
    """batch id -> events, for every <batch>.jsonl in the votes folder."""
    out = {}
    for p in sorted(Path(votes_dir).glob("*.jsonl")):
        out[p.stem] = read_events(p)[0]
    return out


def read_state(path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _letters(n: int) -> str:
    s = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        s = chr(97 + rem) + s
    return s


def _parse_ts(s):
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- the ledger

def _session_of(st: dict) -> dict | None:
    s = st.get("session")
    return dict(s) if isinstance(s, dict) and isinstance(s.get("id"), str) else None


class Ledger:
    """One batch's vote log plus the cursor. Thread-safe; every write goes through submit().

    Other servers may run on other batches with the same votes folder. Each event's session and micro-session
    come from the session in state.json (shared by all of them), and the event is logged and state.json
    rewritten (this batch's entry merged into what the file holds) under folder_lock."""

    def __init__(self, votes_dir, manifest: Manifest, hidden: Hidden, *, micro_gap_s: float = 600.0,
                 ui: str = "", clock=None):
        self.dir = Path(votes_dir)
        self.m, self.h = manifest, hidden
        self.micro_gap = float(micro_gap_s)
        self.ui = ui
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.path = self.dir / f"{manifest.batch}.jsonl"
        self.state_path = self.dir / STATE_NAME
        self._lock = threading.Lock()
        self._log = VoteLog(self.path)
        try:
            self.events, self.problems = read_events(self.path)
            self._by_id = {e["event_id"]: e for e in self.events}
            self._d = derive(manifest, self.events)
            known = {r.id for r in manifest.rounds}
            stray = sum(1 for e in self.events if e.get("round") is not None and e.get("round") not in known)
            if stray:
                self.problems.append(f"{stray} event(s) name rounds that are not in this manifest")
            self._fresh = True                       # this server has logged nothing yet
            with folder_lock(self.dir):
                st = read_state(self.state_path)
                self._sess = _session_of(st)         # the session as of this server's last event (for the page)
                saved = (st.get("batches") or {}).get(manifest.batch) if isinstance(st.get("batches"), dict) else None
                self.rebuilt = isinstance(saved, dict) and (saved.get("current") != self._d.current
                                                            or saved.get("settled") != len(self._d.active))
                self._put_state(st, self.clock())
        except BaseException:
            self._log.close()
            raise

    def close(self) -> None:
        self._log.close()

    # -- what the page sees: the cursor, never anything hidden
    def state(self) -> dict:
        with self._lock:
            return self._state()

    def _state(self) -> dict:
        d, m, s = self._d, self.m, self._sess or {}
        total = len(m.rounds)
        nudges = s.get("nudges", 0)
        return {"batch": m.batch, "mode": m.mode, "total": total, "settled": len(d.active),
                "index": min(len(d.active) + 1, total), "current": d.current, "upcoming": d.pending[1:3],
                "done": d.done,
                "last_vote": {"event_id": d.last_vote, "round": d.last_vote_round} if d.last_vote else None,
                "session": s.get("id"), "micro": s.get("micro"),
                # when the micro-session began and how many break nudges it has had: a reloaded page keeps
                # counting toward the 25-minute nudge from there (taste_pipeline §1 Resume)
                "micro_start": s.get("micro_start"), "micro_nudges": nudges if _is_int(nudges) else 0}

    @property
    def done(self) -> bool:
        return self._d.done

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.events)

    def submit(self, raw) -> dict:
        """Validate, label, append (fsynced) and update the cursor. Raises EventError or Conflict."""
        with self._lock:
            ev = check_event(raw, self.m)
            eid = ev.get("event_id")
            if eid is not None and eid in self._by_id:          # a retry of something already written
                if self._by_id[eid].get("type") != ev["type"]:
                    raise Conflict("that event_id was already used by another event")
                return {"ok": True, "duplicate": True, "event_id": eid, "type": ev["type"], "state": self._state()}
            d, t = self._d, ev["type"]
            if t in VOTE_TYPES and ev["round"] in d.active:
                raise Conflict("this round already has a vote (take it back first to vote again)")
            if t == "abandon" and ev["round"] in d.active:
                raise Conflict("this round already has a vote, so it cannot be abandoned")
            if t == "session" and ev["what"] == "round_start" and ev["round"] in d.active:
                raise Conflict("this round already has a vote")
            if t == "retract":
                target = self._by_id.get(ev["target"])
                if target is None or target.get("type") not in VOTE_TYPES:
                    raise Conflict("nothing to take back: no vote with that id in this batch")
                if d.active.get(target["round"]) != target["event_id"]:
                    raise Conflict("that vote was already taken back")
            now = self.clock()
            todo = []
            if t == "session" and ev["what"] == "start" and d.open_round:
                todo.append({"type": "abandon", "round": d.open_round, "why": "no_commit"})
            todo.append(ev)
            with folder_lock(self.dir):
                st = read_state(self.state_path)
                sess = self._next_session(_session_of(st), now, ev)
                recs = [self._record(e, now, sess["id"], sess["micro"]) for e in todo]
                self._log.append_many(recs)                 # fsynced; on failure nothing is kept or changed
                for rec in recs:
                    self.events.append(rec)
                    self._by_id[rec["event_id"]] = rec
                self._d = derive(self.m, self.events)
                self._sess, self._fresh = sess, False
                self._put_state(st, now, sess)
            return {"ok": True, "event_id": recs[-1]["event_id"], "type": t, "state": self._state()}

    # -- session and micro-session (taste_pipeline §1 Resume: a gap under 10 min continues the micro-session)
    def _next_session(self, s: dict | None, now: datetime, ev: dict) -> dict:
        """The session for an event at `now`, from the shared one in state.json (`s`, not changed). After a gap
        of micro_gap or more, this server's first event (or the first of a new day) begins a new session; any
        other event begins the next micro-session. The page's `playing` heartbeats keep a long listen gapless."""
        stamp = now.isoformat(timespec="seconds")
        last = _parse_ts(s.get("last")) if s else None
        gap = max(0.0, (now - last).total_seconds()) if last is not None else None
        today = now.date().isoformat()
        long_gap = gap is None or gap >= self.micro_gap
        if s is None or last is None or (long_gap and (self._fresh or s.get("date") != today)):
            n = (s.get("n", 0) + 1) if s and s.get("date") == today and _is_int(s.get("n")) else 1
            s = {"id": f"s{now:%m%d}{_letters(n)}", "date": today, "n": n, "micro": 1, "micro_start": stamp,
                 "nudges": 0}
        else:
            s = dict(s)
            if long_gap:
                s.update(micro=int(s.get("micro", 0)) + 1, micro_start=stamp, nudges=0)
        if ev["type"] == "session" and ev["what"] == "nudge":
            s["nudges"] = (s["nudges"] if _is_int(s.get("nudges")) else 0) + 1
        s["last"] = stamp
        return s

    def _record(self, ev: dict, now: datetime, sess: str, micro: int) -> dict:
        m, t = self.m, ev["type"]
        rec = {"v": SCHEMA_VERSION, "type": t, "event_id": ev.get("event_id") or new_event_id(),
               "ts": now.isoformat(timespec="seconds"), "session": sess, "micro": micro, "batch": m.batch}
        if t == "retract":
            target = self._by_id[ev["target"]]
            rec.update(target=target["event_id"], round=target["round"], why=ev.get("why"))
            return rec
        if t == "session":
            rec.update(what=ev["what"], round=ev.get("round"), detail=ev.get("detail"), device=ev.get("device"),
                       audio=ev.get("audio"), ui=self.ui)
            return rec
        r = m.round(ev["round"])
        hr = self.h.rounds[r.id]
        real = hr.real
        rec.update(round=r.id, index=len(self._d.active) + 1, brief_id=hr.brief_id, scene=r.scene)
        if "clips" in ev:
            rec["clips"] = [{"clip": real(c["clip"]), "oid": c["clip"], "pos": c["pos"],
                             "heard_bars": c["heard_bars"], "heard_ms": c["heard_ms"]} for c in ev["clips"]]
        if t in ("bws4", "shootout"):
            rec.update(best=real(ev["best"]), worst=real(ev["worst"]), no_best=ev["no_best"],
                       no_worst=ev["no_worst"], all_bad=ev["all_bad"])
            if t == "bws4":
                tag = ev.get("tag")
                rec["tag"] = None if tag is None else {
                    "name": tag, "clip": real(ev["worst"] if tag in NEG_TAGS else ev["best"])}
        elif t == "keep3":
            rec.update(best=real(ev["best"]), all_bad=ev["all_bad"])
        elif t == "triage":
            rec["tier"] = ev["tier"]
        elif t == "ab":
            rec.update(left=real(r.clips[0].id), right=real(r.clips[1].id), pick=real(ev["pick"]))
        elif t == "playlist":
            rec.update(mark=ev["mark"], plays=ev["plays"])
        elif t == "abandon":
            rec["why"] = ev["why"]
        rec.update(last_heard=real(ev.get("last_heard")), switches=ev.get("switches", 0),
                   play_order=ev.get("play_order", []), autoplay_done=ev.get("autoplay_done", False),
                   t_ms=ev.get("t_ms", {}), device=ev.get("device"), audio=ev.get("audio"), ui=self.ui)
        hidden = {"kind": hr.kind, "of": hr.of}
        if hr.kind == "anchor":
            hidden["anchor"] = real(hr.anchor_clip)
        rec["hidden"] = hidden
        if t == "shootout":
            rec["category"] = hr.category or self.h.category
            rec["options"] = [hr.clips[c.id].get("option") for c in r.clips]
        return rec

    def _put_state(self, st: dict, now: datetime, session: dict | None = None) -> None:
        """Rewrite state.json from `st`, its content as read under folder_lock: this batch's cursor replaces its
        own entry (other servers' entries are kept as read) and `session`, when given, the shared session.
        A failure is noted, not raised: the event is already in the log, which rebuilds the cursor at start."""
        batches = st.get("batches") if isinstance(st.get("batches"), dict) else {}
        d = self._d
        batches[self.m.batch] = {"current": d.current, "settled": len(d.active), "total": len(self.m.rounds),
                                 "done": d.done, "events": len(self.events),
                                 "updated": now.isoformat(timespec="seconds")}
        out = {"v": SCHEMA_VERSION, "updated": now.isoformat(timespec="seconds"),
               "session": session if session is not None else _session_of(st), "batches": batches}
        try:
            atomic_write(self.state_path, (json.dumps(out, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
        except OSError as e:
            note = f"state.json could not be updated ({e.strerror or e}); the vote log holds every event"
            if note not in self.problems:
                self.problems.append(note)
            sys.stderr.write(f"rate: {note}\n")
