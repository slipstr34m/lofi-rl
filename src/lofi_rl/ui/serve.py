"""The rating server: `make rate` (docs/RATING.md; docs/design/taste_pipeline.md §1-§2).

A stdlib ThreadingHTTPServer bound to 127.0.0.1 only (port 8596 by default). It serves the static page, one
batch's clips (only the files its public manifest lists) and a trimmed copy of that manifest. It never serves
manifest.hidden.json. POST /api/event appends each event, fsynced, to $LOFI_VOTES/<batch>.jsonl and updates
$LOFI_VOTES/state.json, so a session resumes where it stopped. GET /api/audit stays locked (423) until every
round of the batch has a vote.

Routes
  GET  /, /index.html, /static/app.js, /static/audio.js     the page
  GET  /manifest.json                                       the trimmed public manifest
  GET  /clips/<file>                                        a listed clip, nothing else
  GET  /api/config                                          configs/rating.yaml values the page needs
  GET  /api/state                                           the cursor: which round, how many are done
  GET  /api/audit                                           aggregates, once voting is done
  POST /api/event                                           one event (vote, session, abandon or retract)

Loads no model: the only imports beyond the standard library are yaml and lofi_rl.ui.

    python -m lofi_rl.ui.serve [inbox/<date>/<batch>] [--port 8596] [--no-browser]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socketserver
import stat
import sys
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

from lofi_rl.ui.audit import aggregate
from lofi_rl.ui.manifest import BATCH_RE, ManifestError, load_batch, load_public
from lofi_rl.ui.votes import (DEVICE_RE, Conflict, EventError, LogBusy, Ledger, derive, read_all,
                              read_events)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "rating.yaml"
STATIC_DIR = Path(__file__).resolve().with_name("static")
STATIC_ROUTES = {"/": "index.html", "/index.html": "index.html", "/static/app.js": "app.js",
                 "/static/audio.js": "audio.js"}
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".flac": "audio/flac", ".wav": "audio/wav"}
JSON_TYPE = "application/json; charset=utf-8"
LOOPBACK = ("127.0.0.1", "localhost")
MAX_BODY = 64 * 1024
SECURITY_HEADERS = (
    ("Cache-Control", "no-store"),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Frame-Options", "DENY"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                                "connect-src 'self'; media-src 'self'; img-src 'self' data:; base-uri 'none'; "
                                "form-action 'none'; frame-ancestors 'none'"),
)

DEFAULT_CONFIG = {
    "port": 8596,
    "heard": {"bar_fraction": 0.75, "min_bars": 4, "toss_min_bars": 2},
    "autoplay": {"bars_per_clip": 8, "wrap_bars": 2, "start_next_round": True},
    "switch_ramp_ms": 5,
    "nudge_minutes": 25,
    "micro_gap_minutes": 10,
    "playlist_plays": [2, 4],
    "daily_budget_minutes": 60,
    "devices": ["airpods", "headphones", "laptop", "speakers"],
    "default_device": "airpods",
}


class ConfigError(ValueError):
    pass


class VotesDirError(ValueError):
    pass


# ----------------------------------------------------------------------------- config

def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _intv(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def load_config(path=None) -> dict:
    """configs/rating.yaml merged over the defaults; unknown keys and out-of-range values are errors."""
    path = Path(path) if path else CONFIG_PATH
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{path} is missing") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: must be a mapping")
    errs = []
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    for key, val in data.items():
        if key not in DEFAULT_CONFIG:
            errs.append(f"unknown key {key!r}")
        elif isinstance(DEFAULT_CONFIG[key], dict):
            if not isinstance(val, dict):
                errs.append(f"{key}: must be a mapping")
                continue
            for sub, sval in val.items():
                if sub not in DEFAULT_CONFIG[key]:
                    errs.append(f"unknown key {key}.{sub}")
                else:
                    cfg[key][sub] = sval
        else:
            cfg[key] = val
    h, a = cfg["heard"], cfg["autoplay"]
    checks = [
        (_intv(cfg["port"]) and 1024 <= cfg["port"] <= 65535, "port: a whole number from 1024 to 65535"),
        (_num(h["bar_fraction"]) and 0.5 <= h["bar_fraction"] <= 1.0, "heard.bar_fraction: from 0.5 to 1"),
        (_intv(h["min_bars"]) and 1 <= h["min_bars"] <= 64, "heard.min_bars: a whole number from 1 to 64"),
        (_intv(h["toss_min_bars"]) and 1 <= h["toss_min_bars"] <= 64, "heard.toss_min_bars: from 1 to 64"),
        (_intv(a["bars_per_clip"]) and 1 <= a["bars_per_clip"] <= 64, "autoplay.bars_per_clip: from 1 to 64"),
        (_intv(a["wrap_bars"]) and 0 <= a["wrap_bars"] <= 16, "autoplay.wrap_bars: from 0 to 16"),
        (isinstance(a["start_next_round"], bool), "autoplay.start_next_round: true or false"),
        (_num(cfg["switch_ramp_ms"]) and 1 <= cfg["switch_ramp_ms"] <= 50, "switch_ramp_ms: from 1 to 50"),
        (_num(cfg["nudge_minutes"]) and 1 <= cfg["nudge_minutes"] <= 240, "nudge_minutes: from 1 to 240"),
        (_num(cfg["micro_gap_minutes"]) and 1 <= cfg["micro_gap_minutes"] <= 240, "micro_gap_minutes: 1 to 240"),
        (_num(cfg["daily_budget_minutes"]) and 1 <= cfg["daily_budget_minutes"] <= 1440,
         "daily_budget_minutes: from 1 to 1440"),
    ]
    errs += [msg for ok, msg in checks if not ok]
    pp = cfg["playlist_plays"]
    if not (isinstance(pp, list) and len(pp) == 2 and all(_intv(x) and 1 <= x <= 16 for x in pp) and pp[0] <= pp[1]):
        errs.append("playlist_plays: [fewest, most], whole numbers from 1 to 16")
    devs = cfg["devices"]
    if not (isinstance(devs, list) and devs and all(isinstance(d, str) and DEVICE_RE.fullmatch(d) for d in devs)):
        errs.append("devices: a list of short lowercase names")
    elif cfg["default_device"] not in devs:
        errs.append("default_device: must be one of devices")
    if _intv(h["toss_min_bars"]) and _intv(h["min_bars"]) and h["toss_min_bars"] > h["min_bars"]:
        errs.append("heard.toss_min_bars: at most heard.min_bars")
    if errs:
        raise ConfigError(f"{path}: " + "; ".join(errs))
    return cfg


# ----------------------------------------------------------------------------- folders

def votes_dir_from_env() -> Path:
    v = os.environ.get("LOFI_VOTES")
    return Path(v).expanduser() if v else ROOT / "data" / "votes"


def inbox_dir_from_env() -> Path:
    v = os.environ.get("LOFI_INBOX")
    return Path(v).expanduser() if v else ROOT / "inbox"


def _ident(p: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(p)
    except OSError:
        return None
    return st.st_dev, st.st_ino


def _inside(child, parent) -> bool:
    """Is `child` the folder `parent` or somewhere under it? Decided by file identity (device and inode), not by
    spelling: on this laptop's case-insensitive disk "CSCI596-Project" is the repo, and so is
    /System/Volumes/Data/Users/... (a firmlink that resolve() keeps). `child` need not exist yet. When `parent`
    does not exist (data/ in a fresh clone), its nearest existing ancestor is matched by identity and the missing
    names must follow by name."""
    child = Path(child).expanduser().resolve()
    parent = Path(parent).expanduser().resolve()
    missing: list[str] = []
    anchor = parent
    while _ident(anchor) is None and anchor != anchor.parent:
        missing.insert(0, anchor.name)
        anchor = anchor.parent
    want = _ident(anchor)
    if want is None:
        return False
    chain = [child, *child.parents]                     # child first, the file-system root last
    for i, p in enumerate(chain):
        if _ident(p) == want:
            below = [q.name for q in reversed(chain[:i])]   # the names from just below the match down to child
            return below[:len(missing)] == missing
    return False


def prepare_votes_dir(votes_dir, batch_dir) -> tuple[Path, bool]:
    """Votes are private: inside the repo they may live only under data/ (gitignored), never in a batch folder.
    Creates the folder (mode 700) if it is missing. Returns (path, created)."""
    v = Path(votes_dir).expanduser().resolve()
    b = Path(batch_dir).expanduser().resolve()
    if _inside(v, b):
        raise VotesDirError(f"votes folder {v} is inside the batch folder {b}; votes never sit next to clips")
    if _inside(v, ROOT) and not _inside(v, ROOT / "data"):
        raise VotesDirError(f"votes folder {v} is inside the repo but not under data/ (gitignored); "
                            "set LOFI_VOTES to data/votes or to a folder outside the repo")
    if v.exists() and not v.is_dir():
        raise VotesDirError(f"{v} exists and is not a folder")
    created = not v.exists()
    if created:
        os.makedirs(v, mode=0o700)
        os.chmod(v, 0o700)
    return v, created


# ----------------------------------------------------------------------------- the app

def load_static() -> tuple[dict, str]:
    """route -> (bytes, content type), plus the ui hash written into every event."""
    files, h = {}, hashlib.sha256()
    for name in ("index.html", "app.js", "audio.js"):
        data = (STATIC_DIR / name).read_bytes()
        h.update(name.encode() + b"\0" + data + b"\0")
        files[name] = (data, CONTENT_TYPES[Path(name).suffix])
    return {route: files[name] for route, name in STATIC_ROUTES.items()}, h.hexdigest()[:12]


class RatingApp:
    """One batch: its manifests, its ledger, and the bytes the server may send."""

    def __init__(self, batch_dir, votes_dir, config: dict, *, clock=None, quiet: bool = False,
                 verify_sha: bool = True):
        self.config = config
        self.quiet = quiet
        self.batch_dir = Path(batch_dir).expanduser()
        self.manifest, self.hidden = load_batch(self.batch_dir, verify_sha=verify_sha)
        self.votes_dir, self.created_votes_dir = prepare_votes_dir(votes_dir, self.batch_dir)
        self.static, self.ui = load_static()
        self.ledger = Ledger(self.votes_dir, self.manifest, self.hidden,
                             micro_gap_s=config["micro_gap_minutes"] * 60.0, ui=self.ui, clock=clock)
        root = self.batch_dir.resolve()
        self.files = {name: root / name for name in self.manifest.files}
        public = self.manifest.public(tuple(config["playlist_plays"]))
        self.public_json = json.dumps(public, ensure_ascii=False).encode("utf-8")
        self.config_json = json.dumps(self.page_config(), ensure_ascii=False).encode("utf-8")
        self.port = None
        self.allowed_hosts: set[str] = set()
        self.allowed_origins: set[str] = set()

    def page_config(self) -> dict:
        c = self.config
        return {"heard": c["heard"], "autoplay": c["autoplay"], "switch_ramp_ms": c["switch_ramp_ms"],
                "nudge_minutes": c["nudge_minutes"], "micro_gap_minutes": c["micro_gap_minutes"],
                "devices": c["devices"], "default_device": c["default_device"], "ui": self.ui}

    def bind(self, port: int) -> None:
        self.port = port
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.allowed_origins = {f"http://{h}" for h in self.allowed_hosts}

    def notes(self) -> list[str]:
        out = []
        if self.created_votes_dir:
            out.append(f"created the votes folder {self.votes_dir}")
        if _inside(self.votes_dir, (ROOT / "data").resolve()) and not (self.votes_dir / ".git").exists():
            out.append(f"{self.votes_dir} is not the private votes repo yet, so make backup-votes will refuse "
                       "(docs/RATING.md, 'Where votes go')")
        if self.ledger.rebuilt:
            out.append("state.json did not match the vote log; the cursor was rebuilt from the log")
        out += [f"vote log: {p}" for p in self.ledger.problems]
        return out

    def submit(self, payload) -> dict:
        """Append one event; announce `batch_done` once, when the last round gets its vote (taste_pipeline §1:
        the push-notification loop waits for it; state.json's batches.<batch>.done flips at the same time)."""
        was_done = self.ledger.done
        reply = self.ledger.submit(payload)
        if reply["state"]["done"] and not was_done and not self.quiet:
            print(f"rate: batch_done {self.manifest.batch} ({reply['state']['total']} rounds)", flush=True)
        return reply

    def audit(self) -> tuple[int, dict]:
        events = self.ledger.snapshot()
        d = derive(self.manifest, events)
        if not d.done:
            return 423, {"locked": True, "remaining": len(d.pending),
                         "error": "The summary opens once every round of this batch has a vote."}
        all_events = read_all(self.votes_dir)
        all_events[self.manifest.batch] = events
        c = self.config
        out = aggregate(self.manifest, self.hidden, events, all_events=all_events,
                        heard_min_bars=c["heard"]["min_bars"], toss_min_bars=c["heard"]["toss_min_bars"],
                        daily_budget_min=c["daily_budget_minutes"])
        out["problems"] = list(self.ledger.problems)
        return 200, out

    def close(self) -> None:
        self.ledger.close()


# ----------------------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "lofi-rate/1"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if not self.server.app.quiet:
            sys.stderr.write("rate: " + (fmt % args) + "\n")

    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("HEAD")

    def do_POST(self):
        self._dispatch("POST")

    def _refuse_method(self):
        self.close_connection = True
        self._json(405, {"error": "method not allowed"}, extra=(("Allow", "GET, HEAD, POST"),))

    do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_TRACE = do_CONNECT = _refuse_method

    # -- responses
    def _send(self, status: int, body: bytes, ctype: str, *, head: bool = False, extra=()) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in SECURITY_HEADERS + tuple(extra):
            self.send_header(k, v)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _json(self, status: int, obj, *, head: bool = False, extra=()) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), JSON_TYPE, head=head, extra=extra)

    # -- routing
    def _clean_path(self) -> str | None:
        """The percent-decoded path, or None for anything malformed (NUL, backslash, dot segments, bad UTF-8)."""
        try:
            path = unquote(urlsplit(self.path).path, errors="strict")
        except (UnicodeDecodeError, ValueError):
            return None
        if not path.startswith("/") or "\x00" in path or "\\" in path:
            return None
        if any(seg in (".", "..") for seg in path.split("/")):
            return None
        return path

    def _dispatch(self, method: str) -> None:
        app = self.server.app
        try:
            if self.headers.get("Host") not in app.allowed_hosts:
                self.close_connection = True
                return self._json(421, {"error": "this server answers only to 127.0.0.1"})
            path = self._clean_path()
            if path is None:
                self.close_connection = True
                return self._json(400, {"error": "refused: malformed path"})
            head = method == "HEAD"
            if method == "POST":
                if path != "/api/event":
                    self.close_connection = True
                    return self._json(404, {"error": "not found"})
                return self._post_event()
            if path == "/api/event":
                return self._json(405, {"error": "POST events here"}, head=head, extra=(("Allow", "POST"),))
            if path in app.static:
                body, ctype = app.static[path]
                return self._send(200, body, ctype, head=head)
            if path == "/manifest.json":
                return self._send(200, app.public_json, JSON_TYPE, head=head)
            if path == "/api/config":
                return self._send(200, app.config_json, JSON_TYPE, head=head)
            if path == "/api/state":
                return self._json(200, app.ledger.state(), head=head)
            if path == "/api/audit":
                status, body = app.audit()
                return self._json(status, body, head=head)
            if path.startswith("/clips/"):
                return self._clip(path[len("/clips/"):], head)
            return self._json(404, {"error": "not found"}, head=head)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            traceback.print_exc()
            self.close_connection = True
            try:
                self._json(500, {"error": "server error (see the terminal running make rate)"})
            except OSError:
                pass

    def _clip(self, name: str, head: bool) -> None:
        app = self.server.app
        path = app.files.get(name)            # only names the public manifest lists, matched exactly
        if path is None:
            return self._json(404, {"error": "not found"}, head=head)
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError:
            return self._json(404, {"error": "not found"}, head=head)
        with os.fdopen(fd, "rb") as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode):
                return self._json(404, {"error": "not found"}, head=head)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPES[Path(name).suffix])
            self.send_header("Content-Length", str(st.st_size))
            for k, v in SECURITY_HEADERS:
                self.send_header(k, v)
            self.end_headers()
            if not head:
                while True:
                    chunk = f.read(1 << 16)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

    def _post_event(self) -> None:
        app = self.server.app
        origin = self.headers.get("Origin")
        if origin is not None and origin not in app.allowed_origins:
            self.close_connection = True
            return self._json(403, {"error": "refused: requests from other sites are not accepted"})
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            return self._json(411, {"error": "send a Content-Length"})
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype not in ("application/json", "text/plain"):
            self.close_connection = True
            return self._json(415, {"error": "send the event as JSON"})
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True
            return self._json(411, {"error": "send a Content-Length"})
        if n <= 0:
            self.close_connection = True
            return self._json(400, {"error": "the body is empty"})
        if n > MAX_BODY:
            self.close_connection = True
            return self._json(413, {"error": f"an event is at most {MAX_BODY} bytes"})
        raw = self.rfile.read(n)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            return self._json(400, {"error": "the body is not JSON"})
        try:
            reply = app.submit(payload)
        except Conflict as e:
            return self._json(409, {"error": str(e), "state": app.ledger.state()})
        except EventError as e:
            return self._json(400, {"error": str(e)})
        return self._json(200, reply)


class RatingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        # HTTPServer.server_bind calls socket.getfqdn(), a reverse DNS lookup that took 35 s on this laptop
        # (measured 2026-10-31). The name is never used, so bind without it.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def make_server(app: RatingApp, port: int | None = None, host: str = "127.0.0.1") -> RatingHTTPServer:
    """Bind 127.0.0.1 (never another interface). port=0 picks a free port (tests)."""
    if host not in LOOPBACK:
        raise ValueError(f"the rating UI serves 127.0.0.1 only, not {host!r}")
    srv = RatingHTTPServer(("127.0.0.1", app.config["port"] if port is None else port), Handler)
    srv.app = app
    app.bind(srv.server_address[1])
    return srv


# ----------------------------------------------------------------------------- choosing a batch

def find_batches(inbox) -> list[Path]:
    """Batch folders in the inbox (inbox/<date>/<batch>/manifest.json), oldest date first."""
    return sorted(p.parent for p in Path(inbox).glob("*/*/manifest.json"))


def batch_done(batch_dir, votes_dir) -> bool | None:
    """True once every round has a vote; None if the manifest does not load."""
    try:
        m = load_public(batch_dir)
    except ManifestError:
        return None
    events, _ = read_events(Path(votes_dir) / f"{m.batch}.jsonl")
    return derive(m, events).done


def resolve_batch(arg, inbox, votes) -> Path:
    if arg:
        p = Path(arg).expanduser()
        if p.is_dir():
            return p
        if BATCH_RE.fullmatch(str(arg)):
            hits = sorted(q.parent for q in Path(inbox).glob(f"*/{arg}/manifest.json"))
            if len(hits) == 1:
                return hits[0]
        raise ManifestError([f"no batch folder {arg!r} (give inbox/<date>/<batch>)"])
    for b in find_batches(inbox):
        if batch_done(b, votes) is False:
            return b
    raise ManifestError([f"nothing to rate: no unfinished batch in {inbox}"])


# ----------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lofi_rl.ui.serve",
                                 description="Serve the rating page for one batch on 127.0.0.1.")
    ap.add_argument("batch", nargs="?", help="inbox/<date>/<batch> (default: the oldest unfinished batch)")
    ap.add_argument("--port", type=int, help="default: port in configs/rating.yaml (8596)")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--no-browser", action="store_true", help="do not open the page in a browser")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.config)
        votes, inbox = votes_dir_from_env(), inbox_dir_from_env()
        batch = resolve_batch(a.batch, inbox, votes)
        app = RatingApp(batch, votes, cfg)
    except (ConfigError, ManifestError, VotesDirError, LogBusy) as e:
        print(f"rate: {e}", file=sys.stderr)
        return 2
    port = cfg["port"] if a.port is None else a.port
    try:
        srv = make_server(app, port)
    except OSError as e:
        app.close()
        print(f"rate: cannot listen on 127.0.0.1:{port} ({e}). Is make rate already running? "
              f"Try: make rate ... (or python -m lofi_rl.ui.serve --port {port + 1})", file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    st = app.ledger.state()
    m = app.manifest
    print(f"rate: {m.batch} ({m.mode}): {st['settled']} of {st['total']} rounds have a vote. "
          f"Votes go to {app.votes_dir}", flush=True)
    for note in app.notes():
        print(f"rate: note: {note}", flush=True)
    print(f"rate: open {url}  (Ctrl-C stops the server; your place is saved)", flush=True)
    if not a.no_browser and not os.environ.get("LOFI_NO_BROWSER"):
        webbrowser.open(url)
    try:
        srv.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nrate: stopped; your place is saved.")
    finally:
        srv.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
