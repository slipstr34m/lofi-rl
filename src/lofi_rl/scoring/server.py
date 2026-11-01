"""The scorer server (docs/SCORER.md §4; taste_pipeline.md §5, "Scorer server").

    python -m lofi_rl.scoring.server --sock PATH [--workers 8] [--cache DIR] [--calib DIR] [--rm DIR]
        [--gpu-mem-gb 6] [--device cuda|cpu] [--palette palettes/dev_gu.yaml] [--theme configs/clap_theme_v1.yaml]
        [--backend real|fake] [--render auto|real|fake] [--render-workers 1] [--render-timeout-s 150] [--lru 4096]

HTTP/1.1 + JSON on a node-local Unix socket, created mode 0600 (owner only), so it cannot clash with another
job's ports. The server refuses to start if PATH exists and is not a stale socket this user owns (a live one
means another server; anything else is not ours to delete). It loads the models before it binds the socket, so
the socket appears only when /health can answer ok.

    GET  /health   versions, live terms, calibration flags, palette, backend, GPU
    POST /score    one item per completion: {clip_id, hard, G, r{ce,theme,taste}, sig{...}, diag{...}, t_ms}
    POST /render   render only (optionally FLAC and manifest into out_dir)
    GET  /stats    per-stage timings, cache hits, peak GPU memory, pool rebuilds

Status codes: 400 a malformed request, 404 an unknown path, 409 another palette or pinned version, 413 a body
over 16 MiB, 422 a brief without PAR references or an unusable out_dir, 500 a backend exception (with its
message and the end of its traceback). The client raises on anything but 200.

SIGTERM or SIGINT stops the server, removes its socket (if it is still the one it created) and ends the render
workers; exit 0. SIGUSR1 prints every thread's stack (faulthandler). Exit 2 for a bad argument, calibration, RM
directory, theme config or socket path, before anything heavy loads where possible, and for a renderer that
cannot render: before it binds, the server renders a canary (core.Scorer.check_renderer), so FluidSynth missing
or not 2.6.1, or an /etc/fluidsynth.conf with commands, stops it instead of answering every item with G = 0.
"""
from __future__ import annotations

import argparse
import faulthandler
import http.server
import json
import os
import signal
import socket
import socketserver
import stat
import sys
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lofi_rl.palette import ROOT, PaletteError, load_palette

from . import calib as calib_mod
from . import protocol, rm_api, theme
from .cache import LRU_SIZE, FeatureCache
from .core import RendererError, Scorer, request_error
from .protocol import MAX_BODY_BYTES, ProtocolError, encode_json
from .render_pool import RenderPoolDrift

MAX_SUN_PATH = 103 if sys.platform == "darwin" else 107     # sockaddr_un.sun_path, less its NUL
DEFAULT_PALETTE = ROOT / "palettes" / "dev_gu.yaml"          # until palettes/lofi_rain_v1.yaml is frozen (D-017)


class SocketPathError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[scorer {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def check_sock_path(path: Path) -> str:
    """Make `path` free for a new socket: "free", or "stale" after removing a dead socket this user owns.
    Raises SocketPathError for anything else (module docstring)."""
    if len(os.fsencode(path)) > MAX_SUN_PATH:
        raise SocketPathError(f"--sock {path}: {len(os.fsencode(path))} bytes, over the {MAX_SUN_PATH} a Unix "
                              f"socket path may have here")
    if not path.parent.is_dir():
        raise SocketPathError(f"--sock {path}: {path.parent} is not a directory")
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return "free"
    if not stat.S_ISSOCK(st.st_mode):
        raise SocketPathError(f"--sock {path} exists and is not a socket: refusing to replace it")
    if st.st_uid != os.getuid():
        raise SocketPathError(f"--sock {path} is a socket of uid {st.st_uid}, not this user's ({os.getuid()})")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        s.connect(str(path))
    except (ConnectionRefusedError, FileNotFoundError):
        os.unlink(path)
        return "stale"
    except OSError as err:
        raise SocketPathError(f"--sock {path}: cannot tell whether it is live ({err})") from err
    finally:
        s.close()
    raise SocketPathError(f"--sock {path} is a live socket: another server is listening there")


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True               # an in-flight request never holds up a shutdown
    block_on_close = False

    def __init__(self, path: str, scorer: Scorer, logf: Callable[[str], None]) -> None:
        self.scorer, self.logf = scorer, logf
        super().__init__(path, _Handler)


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "lofi-scorer/1"
    sys_version = ""
    server: _Server

    def address_string(self) -> str:    # a Unix peer has no (host, port); the base class would index ''
        return "unix"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002: the base class's name
        self.server.logf(f"{self.command or '-'} {self.path}: " + (format % args))

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        pass                            # _reply logs one line per request

    def _reply(self, status: int, doc: Any, t0: float, note: str = "") -> None:
        try:
            body = encode_json(doc)
        except (TypeError, ValueError) as err:
            status, body = 500, encode_json({"error": f"the response could not be encoded: {err}", "status": 500})
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True
        self.server.logf(f"{self.command} {self.path} {status} {1000 * (time.perf_counter() - t0):.0f} ms {note}".rstrip())

    def _fail(self, err: Exception, t0: float) -> None:
        status, msg = request_error(err)
        doc: dict[str, Any] = {"error": msg, "status": status, "type": type(err).__name__}
        if status >= 500:
            doc["trace"] = traceback.format_exc()[-4000:]
        self._reply(status, doc, t0, note=msg[:300])

    def _body(self) -> Any:
        n = self.headers.get("Content-Length")
        if n is None:
            raise ProtocolError("a POST needs a Content-Length (no chunked bodies)")
        try:
            n = int(n)
        except ValueError:
            raise ProtocolError(f"Content-Length {n!r} is not a number") from None
        if n < 0 or n > MAX_BODY_BYTES:
            raise _TooLarge(f"a body of {n} bytes: at most {MAX_BODY_BYTES}")
        return protocol.decode_json(self.rfile.read(n))

    def do_GET(self) -> None:  # noqa: N802: the base class's name
        t0 = time.perf_counter()
        try:
            if self.path == "/health":
                return self._reply(200, self.server.scorer.health(), t0)
            if self.path == "/stats":
                return self._reply(200, self.server.scorer.stats(), t0)
            return self._reply(404, {"error": f"no GET {self.path}", "status": 404}, t0)
        except Exception as err:  # noqa: BLE001: answered as a 500, never a dropped connection
            self._fail(err, t0)

    def do_POST(self) -> None:  # noqa: N802
        t0 = time.perf_counter()
        try:
            if self.path not in ("/score", "/render"):
                self.close_connection = True
                return self._reply(404, {"error": f"no POST {self.path}", "status": 404}, t0)
            body = self._body()
            if self.path == "/score":
                out = self.server.scorer.score(protocol.parse_score_request(body))
                b = out["batch"]
                note = f"n={b['n']} unique={b['unique']} rendered={b['rendered']} cache={b['cache']}"
            else:
                out = self.server.scorer.render(protocol.parse_render_request(body))
                note = f"n={len(out['items'])}"
            self._reply(200, out, t0, note)
        except _TooLarge as err:
            self._reply(413, {"error": str(err), "status": 413}, t0)
        except Exception as err:  # noqa: BLE001
            self._fail(err, t0)

    def do_PUT(self) -> None:  # noqa: N802
        self._reply(405, {"error": "only GET and POST", "status": 405}, time.perf_counter())

    do_DELETE = do_PATCH = do_PUT  # noqa: N815


class _TooLarge(Exception):
    pass


class ScorerServer:
    """The HTTP server around a Scorer: bind() makes the 0600 socket, serve_forever() answers, shutdown() (from
    another thread or a signal) stops it, close() removes the socket if it is still the one bind() made."""

    def __init__(self, scorer: Scorer, sock: Path | str, logf: Callable[[str], None] = log) -> None:
        self.scorer, self.sock, self.logf = scorer, Path(sock), logf
        self._srv: _Server | None = None
        self._ident: tuple[int, int] | None = None
        self._stopping = threading.Event()

    def bind(self) -> None:
        state = check_sock_path(self.sock)
        if state == "stale":
            self.logf(f"removed a stale socket at {self.sock}")
        old = os.umask(0o177)               # the socket is born 0600: no window with wider permissions
        try:
            self._srv = _Server(str(self.sock), self.scorer, self.logf)
        finally:
            os.umask(old)
        os.chmod(self.sock, 0o600)
        st = os.lstat(self.sock)
        self._ident = (st.st_dev, st.st_ino)

    def serve_forever(self) -> None:
        assert self._srv is not None, "bind() first"
        self._srv.serve_forever(poll_interval=0.2)

    def shutdown(self) -> None:
        if self._srv is not None and not self._stopping.is_set():
            self._stopping.set()
            self._srv.shutdown()

    def close(self) -> None:
        if self._srv is not None:
            self._srv.server_close()
        try:
            st = os.lstat(self.sock)
            if self._ident == (st.st_dev, st.st_ino):
                os.unlink(self.sock)
        except FileNotFoundError:
            pass

    def run(self) -> int:
        """bind, answer until SIGTERM or SIGINT, then close (module docstring)."""
        self.bind()

        def on_signal(signum: int, _frame: Any) -> None:
            self.logf(f"signal {signal.Signals(signum).name}: shutting down")
            threading.Thread(target=self.shutdown, daemon=True).start()

        previous = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        h = self.scorer.health()
        print(json.dumps({"event": "ready", "sock": str(self.sock), "pid": os.getpid(), "versions": h["versions"],
                          "terms": h["terms"], "backend": h["backend"], "canary": h["renderer"]["canary"]}),
              flush=True)
        try:
            self.serve_forever()
        finally:
            self.close()
            for s, prev in previous.items():
                signal.signal(s, prev)
        self.logf(f"stopped; {self.sock} removed")
        return 0


# ------------------------------------------------------------------------------------------ building
def build_scorer(a: argparse.Namespace) -> Scorer:
    """The Scorer the arguments describe. Cheap checks first (configs, calibration, RM), then the models, the
    render pool and the canary (RendererError when it does not render in spec)."""
    from .render_pool import RenderPool, render_config
    render = a.render if a.render != "auto" else ("real" if a.backend == "real" else "fake")
    palette = load_palette(a.palette, verify=render == "real")
    theme_cfg = theme.load_config(a.theme)
    cal = calib_mod.load(a.calib, theme_sha256=theme_cfg.sha256)
    rm = rm_api.load_rm(a.rm)
    if a.backend == "real":
        from .features import Featurizer
        featurizer: Any = Featurizer(device=a.device, gpu_mem_gb=a.gpu_mem_gb)
    else:
        from .fakes import FakeFeaturizer
        featurizer = FakeFeaturizer(render=render)      # its feat_ver names the renderer (the cache key)
    if render == "real":
        renderer: Any = RenderPool(a.palette, workers=a.workers, render_workers=a.render_workers,
                                   timeout_s=a.render_timeout_s,
                                   expect={"src_sha8": featurizer.src_sha8, "palette_sha256": palette.sha256})
    else:
        from .fakes import FakeRenderer
        renderer = FakeRenderer(workers=a.workers)
    try:
        renderer.warm()                     # RenderPoolDrift if a worker loads other code than feat_ver names
        scorer = Scorer(palette=palette, renderer=renderer, featurizer=featurizer,
                        theme=theme.Theme(theme_cfg, featurizer.encode_text), calib=cal, rm=rm,
                        cache=FeatureCache(a.cache, featurizer.feat_ver, a.lru, hw=featurizer.hw),
                        render_cfg=render_config(), backend=a.backend)
        scorer.check_renderer("at start, before the socket")
    except BaseException:
        renderer.close()                    # no render workers left behind by a server that never binds
        raise
    return scorer


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.scoring.server", description=__doc__.split("\n\n")[0])
    p.add_argument("--sock", type=Path, required=True, help="the Unix socket to create (mode 0600)")
    p.add_argument("--workers", type=int, default=8, help="render processes")
    p.add_argument("--cache", type=Path, default=None,
                   help="cache root: features go to <cache>/feat/<feat_ver>/hw-<sha8 of the machine and GPU>/")
    p.add_argument("--calib", type=Path, default=None, help="pq.json, theme.json, ce.json; missing ones disable")
    p.add_argument("--rm", type=Path, default=None, help="rm.json and refs.json; missing: r_taste 0, P_acc factor 1")
    p.add_argument("--gpu-mem-gb", type=float, default=6.0, help="cap on this process's CUDA allocator")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--palette", type=Path, default=DEFAULT_PALETTE)
    p.add_argument("--theme", type=Path, default=theme.DEFAULT_CONFIG)
    p.add_argument("--backend", choices=("real", "fake"), default="real", help="fake: lofi_rl.scoring.fakes (tests)")
    p.add_argument("--render", choices=("auto", "real", "fake"), default="auto", help="auto follows --backend")
    p.add_argument("--render-workers", type=int, default=1, help="FluidSynth processes per render")
    p.add_argument("--render-timeout-s", type=float, default=150.0)
    p.add_argument("--lru", type=int, default=LRU_SIZE)
    a = p.parse_args(argv)
    if a.workers < 1 or a.render_workers < 1 or a.lru < 1 or a.gpu_mem_gb <= 0 or a.render_timeout_s <= 0:
        p.error("--workers, --render-workers and --lru must be >= 1; --gpu-mem-gb and --render-timeout-s > 0")
    for name in ("palette", "theme"):
        if not getattr(a, name).is_file():
            p.error(f"--{name} {getattr(a, name)} is not a file")
    if a.backend == "real" and a.render == "fake":
        p.error("--backend real needs real renders (the models read the audio): --render auto or real")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:                                                 # stacks on a crash, and on SIGUSR1 for a hang
        faulthandler.enable(all_threads=True)
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, all_threads=True)
    except (OSError, ValueError):                        # stderr without a file descriptor (a captured test run)
        pass
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # Audiobox and laion_clap pick CUDA when they see it
    try:
        check_sock_path(a.sock)                          # fail before the models load (checked again at bind)
        scorer = build_scorer(a)
    except (SocketPathError, PaletteError, calib_mod.CalibError, rm_api.RMError, theme.ThemeError, RendererError,
            RenderPoolDrift) as err:
        print(f"scorer: {err}", file=sys.stderr)
        return 2
    try:
        return ScorerServer(scorer, a.sock).run()
    except SocketPathError as err:
        print(f"scorer: {err}", file=sys.stderr)
        return 2
    finally:
        close = getattr(scorer.renderer, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    raise SystemExit(main())
