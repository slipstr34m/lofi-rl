"""ScorerClient: HTTP/1.1 + JSON to the scorer server over a node-local Unix socket (docs/SCORER.md §4).

    client = ScorerClient("/tmp/lofi-scorer.sock")
    client.wait_healthy(600)                  # polls GET /health; pins the five versions and the live terms
    out = client.score(items, palette_sha256=palette.sha256, step=12)     # POST /score, 300 s timeout

Any non-200 answer, an unreachable socket, a timeout, a malformed response, or a version or live terms that
differ from the pinned ones raise ScorerError (taste_pipeline.md §5: "Any error crashes the trainer, which then
resumes from its checkpoint"); nothing is ever scored silently on the client side. One connection per request,
so a restarted server is simply reached again, and refused (409) if it was restarted with another RM,
calibration, references, features or formulas.

A pin lasts one client; a run outlives its processes. read_pin and write_pin keep it in a file that every link
of a chained run shares (lofi_rl.rl.reward's scorer_pin): a link seeds its client from the file, so a server
with another version than the run's first link pinned is refused before the link trains. Standard library
only: lofi_rl.rl.reward imports it in the policy env.
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import protocol
from .protocol import ProtocolError

SCORE_TIMEOUT_S = 300.0         # taste_pipeline.md §5
HEALTH_TIMEOUT_S = 10.0
ENV_SOCK = "LOFI_SCORER_SOCK"   # the reward hook's switch (lofi_rl.rl.reward.make_reward)
PIN_SCHEMA = "lofi-scorer/pin/1"


class ScorerError(RuntimeError):
    """The scorer could not score: a non-200 status (`status`), a timeout, an unreachable socket or a bad answer."""

    def __init__(self, message: str, *, status: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class ScorerUnavailable(ScorerError):
    """The socket is missing, refuses connections, or /health never answered ok in time."""


class ScorerTimeout(ScorerError):
    """No answer within the timeout."""


class UnixHTTPConnection(http.client.HTTPConnection):
    """http.client.HTTPConnection over an AF_UNIX stream socket. The Host header says "localhost"."""

    def __init__(self, path: str | Path, timeout: float = SCORE_TIMEOUT_S) -> None:
        super().__init__("localhost", timeout=timeout)
        self.unix_path = str(path)

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.settimeout(self.timeout)
            s.connect(self.unix_path)
        except BaseException:
            s.close()
            raise
        self.sock = s


class ScorerClient:
    """The trainer's side of the scorer (module docstring). `allow_fake` lets it talk to a server running the
    fake backends (lofi_rl.scoring.fakes: tests and rehearsals); a real run refuses one."""

    def __init__(self, sock: str | Path, *, timeout: float = SCORE_TIMEOUT_S, allow_fake: bool = False) -> None:
        self.sock = str(sock)
        self.timeout = float(timeout)
        self.allow_fake = allow_fake
        self.pinned: dict[str, str] | None = None
        self.pinned_terms: dict[str, bool] | None = None
        self.health_doc: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"ScorerClient({self.sock!r})"

    # -------------------------------------------------------------------------------------- transport
    def request(self, method: str, path: str, body: Any = None, *, timeout: float | None = None) -> dict[str, Any]:
        """One request on a fresh connection: the decoded JSON of a 200 answer, else ScorerError."""
        t = self.timeout if timeout is None else float(timeout)
        conn = UnixHTTPConnection(self.sock, timeout=t)
        payload = None if body is None else json.dumps(body, allow_nan=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Connection": "close"}
        try:
            conn.request(method, path, body=payload, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            status = resp.status
        except (FileNotFoundError, ConnectionRefusedError) as err:
            raise ScorerUnavailable(f"scorer at {self.sock} is not reachable: {err}") from err
        except (socket.timeout, TimeoutError) as err:
            raise ScorerTimeout(f"{method} {path}: no answer from {self.sock} within {t:g} s") from err
        except (OSError, http.client.HTTPException) as err:
            raise ScorerError(f"{method} {path} to {self.sock} failed: {type(err).__name__}: {err}") from err
        finally:
            conn.close()
        try:
            doc = json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, ValueError):
            doc = raw[:2000].decode("utf-8", "replace")
        if status != 200:
            msg = doc.get("error") if isinstance(doc, Mapping) else doc
            raise ScorerError(f"{method} {path} answered {status}: {msg}", status=status, body=doc)
        if not isinstance(doc, dict):
            raise ScorerError(f"{method} {path} answered 200 with no JSON object", status=status, body=doc)
        return doc

    # -------------------------------------------------------------------------------------- endpoints
    def health(self, timeout: float = HEALTH_TIMEOUT_S) -> dict[str, Any]:
        doc = self.request("GET", "/health", timeout=timeout)
        try:
            protocol.check_health(doc)
        except ProtocolError as err:
            raise ScorerError(f"/health: {err}", status=200, body=doc) from err
        if doc.get("backend") != "real" and not self.allow_fake:
            raise ScorerError(f"the scorer at {self.sock} runs the {doc.get('backend')!r} backend: a real run "
                              f"needs the real one (pass allow_fake=True only in tests and rehearsals)",
                              status=200, body=doc)
        return doc

    def wait_healthy(self, timeout: float = 600.0, poll: float = 0.5,
                     alive: Callable[[], bool] | None = None) -> dict[str, Any]:
        """Poll /health until it answers ok, then pin its versions and live terms (later answers must carry the
        same ones; a client seeded from a pin file must find them here already). ScorerUnavailable after
        `timeout` seconds; a server that answers but is wrong raises at once. `alive`, when given, is asked on
        every poll whether the server's process still runs: once it says no, ScorerUnavailable at once, since a
        server that exited during startup (exit 1 or 2) looks like a slow start until the deadline."""
        deadline = time.monotonic() + float(timeout)
        last: Exception | None = None
        while True:
            try:
                doc = self.health(timeout=min(HEALTH_TIMEOUT_S, max(1.0, deadline - time.monotonic())))
                break
            except (ScorerUnavailable, ScorerTimeout) as err:
                last = err
            except ScorerError as err:
                if err.status is None or err.status >= 500:
                    last = err          # starting up or briefly failing: keep polling until the deadline
                else:
                    raise
            if alive is not None and not alive():
                raise ScorerUnavailable(f"the scorer at {self.sock} exited before /health answered ok: {last}")
            if time.monotonic() >= deadline:
                raise ScorerUnavailable(f"the scorer at {self.sock} was not healthy within {timeout:g} s: {last}")
            time.sleep(poll)
        try:
            versions = protocol.check_versions(doc["versions"], self.pinned, "/health")
        except ProtocolError as err:            # a restarted server with another RM, features or formulas
            raise ScorerError(str(err), status=200, body=doc) from err
        terms = dict(doc["terms"])
        if self.pinned_terms is not None and terms != self.pinned_terms:
            raise ScorerError(f"/health: the live terms {terms} are not the pinned {self.pinned_terms}", status=200,
                              body=doc)
        self.pinned, self.pinned_terms, self.health_doc = versions, terms, doc
        return doc

    def score(self, items: Sequence[Mapping[str, Any]], *, palette_sha256: str, step: int | None = None,
              timeout: float | None = None) -> dict[str, Any]:
        """POST /score: one result per item, in order (protocol.check_score_response)."""
        body = protocol.score_request(items, palette_sha256=palette_sha256, step=step, expect=self.pinned)
        doc = self.request("POST", "/score", body, timeout=timeout)
        try:
            out = protocol.check_score_response(doc, body["items"], self.pinned)
        except ProtocolError as err:
            raise ScorerError(f"/score: {err}", status=200, body=doc) from err
        if self.pinned_terms is not None and dict(out["terms"]) != self.pinned_terms:
            # the reward took its CE-rule weights from the pinned terms: a response computed with others is refused
            raise ScorerError(f"/score: the response's terms {out['terms']} are not the pinned {self.pinned_terms}",
                              status=200, body=doc)
        return out

    def render(self, items: Sequence[Mapping[str, Any]], *, palette_sha256: str, out_dir: str | None = None,
               flac: bool = False, timeout: float | None = None) -> dict[str, Any]:
        body = protocol.render_request(items, palette_sha256=palette_sha256, out_dir=out_dir, flac=flac)
        return self.request("POST", "/render", body, timeout=timeout)

    def stats(self, timeout: float = HEALTH_TIMEOUT_S) -> dict[str, Any]:
        return self.request("GET", "/stats", timeout=timeout)

    # -------------------------------------------------------------------------------------- pinned state
    def _health(self) -> dict[str, Any]:
        if self.health_doc is None:
            raise ScorerError("wait_healthy() first: the client has no pinned /health")
        return self.health_doc

    @property
    def terms(self) -> dict[str, bool]:
        return dict(self._health()["terms"])

    @property
    def ce_validated(self) -> bool:
        """Whether the server's CE term is live: Audiobox CE validated on Shresth's votes (calib ce.json)."""
        return bool(self._health()["terms"]["ce"])


# ------------------------------------------------------------------------------------------ the run's pin
def read_pin(path: str | Path) -> dict[str, Any] | None:
    """A run's scorer pin (write_pin), or None when there is none yet. ScorerError for a file that is not one."""
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as err:
        raise ScorerError(f"{p}: cannot read the run's scorer pin: {err}") from err
    try:
        doc = json.loads(raw)
        if not isinstance(doc, dict) or doc.get("schema") != PIN_SCHEMA:
            raise ProtocolError(f"not a {PIN_SCHEMA} document")
        protocol.check_versions(doc.get("versions") or {}, None, "the pin")
        protocol.check_terms(doc.get("terms"), "the pin")
    except (ValueError, ProtocolError) as err:
        raise ScorerError(f"{p}: the run's scorer pin is malformed: {err}") from None
    return doc


def write_pin(path: str | Path, client: ScorerClient, **extra: Any) -> dict[str, Any]:
    """Write `client`'s pinned versions and terms (after wait_healthy) to `path`, atomically."""
    if client.pinned is None or client.pinned_terms is None:
        raise ScorerError("wait_healthy() first: the client has nothing pinned to write")
    doc = {"schema": PIN_SCHEMA, "versions": dict(client.pinned), "terms": dict(client.pinned_terms), **extra}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return doc


def from_env(environ: Mapping[str, str] | None = None) -> ScorerClient | None:
    """A client on $LOFI_SCORER_SOCK, or None when it is unset or empty."""
    sock = (os.environ if environ is None else environ).get(ENV_SOCK, "")
    return ScorerClient(sock) if sock else None
