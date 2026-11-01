"""The scorer server over a real Unix socket (src/lofi_rl/scoring/server.py, client.py; docs/SCORER.md §4), with the
fake backends (lofi_rl.scoring.fakes): /health, /score, /render, /stats, status codes and the client's errors,
the 0600 socket, the stale/live/foreign path rules, and SIGTERM/SIGINT removing the socket (a subprocess)."""
import contextlib
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

import pytest

from helpers_scoring import BRIEF_A, BRIEF_B, PALETTE, ROOT, canon_sha_of, fixture_text, make_scorer, short_sock_dir
from lofi_rl.palette import load_palette
from lofi_rl.scoring import fakes, protocol, server
from lofi_rl.scoring.client import ScorerClient, ScorerError, ScorerUnavailable, UnixHTTPConnection
from lofi_rl.scoring.render_pool import failure
from lofi_rl.scoring.server import ScorerServer, SocketPathError, check_sock_path

PAL = load_palette(PALETTE, verify=False)
EX = fixture_text("example.ls")


@contextlib.contextmanager
def served(tmp_path, **kw):
    scorer = make_scorer(tmp_path, **kw)
    with short_sock_dir() as d:
        sock = d / "s.sock"
        srv = ScorerServer(scorer, sock, logf=lambda m: None)
        srv.bind()
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            client = ScorerClient(sock, allow_fake=True, timeout=30)
            client.wait_healthy(10)
            yield client, scorer, sock
        finally:
            srv.shutdown()
            t.join(5)
            srv.close()
    assert not sock.exists()


def _items(texts, brief=BRIEF_A, brief_id="596:0"):
    return [protocol.item_dict(i, t, brief, brief_id) for i, t in enumerate(texts)]


def _raw(sock, method, path, body=b"", headers=None):
    conn = UnixHTTPConnection(sock, timeout=10)
    try:
        conn.request(method, path, body=body, headers=headers or {"Content-Type": "application/json"})
        r = conn.getresponse()
        return r.status, json.loads(r.read() or b"null")
    finally:
        conn.close()


def _raw_once(sock, request: bytes):
    """One request written in a single sendall, and the status and JSON body of the answer. http.client writes
    the headers and the body separately, so a server that answers before reading the body (a 413) and closes can
    break the second write (BrokenPipeError: a flaky test under load)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10)
    try:
        s.connect(str(sock))
        s.sendall(request)
        raw = b""
        while chunk := s.recv(65536):
            raw += chunk
    finally:
        s.close()
    head, _, body = raw.partition(b"\r\n\r\n")
    return int(head.split()[1]), json.loads(body or b"null")


# ------------------------------------------------------------------------------------------ endpoints
def test_health_reports_versions_terms_and_every_missing_calibration(tmp_path):
    with served(tmp_path) as (client, scorer, _):
        h = client.health()
    assert h["ok"] is True and h["protocol"] == protocol.PROTOCOL and h["backend"] == "fake"
    assert set(h["versions"]) == set(protocol.VERSION_KEYS) and h["versions"]["rm_ver"] == "null"
    assert h["versions"]["feat_ver"].startswith("fake1.render-fake.") and h["palette_sha256"] == PAL.sha256
    assert h["terms"] == {k: False for k in protocol.TERM_FLAGS}
    assert sorted(h["disabled"]) == sorted(protocol.TERM_FLAGS)
    cal = h["calibration"]
    assert cal["pq"] == {"status": "missing", "effect": "PQ factor 1"}
    assert cal["theme"]["status"] == "missing" and cal["ce"]["status"] == "missing"
    assert cal["rm"]["status"] == "missing" and "P_acc factor 1" in cal["rm"]["effect"]
    assert cal["theme_config"]["sha256"] == scorer.theme.config.sha256
    assert client.pinned == h["versions"] and client.ce_validated is False


def test_score_returns_one_documented_item_per_completion_in_order_and_renders_each_clip_once(tmp_path):
    renderer = fakes.FakeRenderer()
    texts = [EX, fixture_text("shaker_edits.ls"), EX, fixture_text("example_leniency.ls"), "not loopscript"]
    with served(tmp_path, renderer=renderer) as (client, _, _sock):
        out = client.score(_items(texts), palette_sha256=PAL.sha256, step=7)
        again = client.score(_items(texts), palette_sha256=PAL.sha256, step=8)
        stats = client.stats()
    items = out["items"]
    assert [it["i"] for it in items] == [0, 1, 2, 3, 4] and out["step"] == 7
    for it in items:
        assert set(it) >= {"i", "clip_id", "hard", "G", "gate", "r", "sig", "diag", "t_ms"}
        assert set(it["r"]) == set(protocol.AUDIO_TERMS) and set(it["sig"]) >= set(protocol.SIG_KEYS)
        assert set(it["diag"]) >= set(protocol.DIAG_KEYS) | {"disabled", "cache", "dup_of", "error", "render"}
    a, shaker, dup, len_, junk = items
    assert dup["clip_id"] == a["clip_id"] == len_["clip_id"], "one canonical form, one clip"
    assert (dup["diag"]["cache"], dup["diag"]["dup_of"], len_["diag"]["dup_of"]) == ("dup", 0, 0)
    assert (a["diag"]["cache"], a["hard"], a["G"], a["gate"]) == ("miss", 1, 1.0, None)
    assert a["sig"]["pq"] is not None and a["sig"]["clap_margin"] is not None and a["sig"]["centroid_hz"] > 0
    assert (junk["hard"], junk["G"], junk["gate"], junk["diag"]["cache"]) == (0, 0.0, "G1", "none")
    assert junk["sig"]["pq"] is None and junk["diag"]["error"]
    assert len(renderer.calls) == 2, "example (x3 texts) and shaker: two renders"
    assert out["batch"]["unique"] == 3 and out["batch"]["rendered"] == 2 and out["batch"]["cache"]["dup"] == 2
    assert [it["diag"]["cache"] for it in again["items"]] == ["mem", "mem", "dup", "dup", "none"]
    assert len(renderer.calls) == 2, "the second request is all cache hits"
    assert [x["r"] for x in again["items"]] == [x["r"] for x in items]
    assert stats["counts"]["requests_score"] == 2 and stats["cache"]["hits_mem"] == 2
    assert stats["t_ms"]["request"]["n"] == 2 and stats["t_ms"]["render"]["n"] == 2
    assert set(stats["t_ms"]) >= {"parse", "render", "featurize", "score", "request"}


def test_render_endpoint_renders_without_features(tmp_path):
    feat = fakes.FakeFeaturizer()
    with served(tmp_path, featurizer=feat) as (client, _, _sock):
        out = client.render(_items([EX, "junk", EX]), palette_sha256=PAL.sha256, out_dir=str(tmp_path))
    a, junk, dup = out["items"]
    assert a["ok"] and a["render"]["audio_sha256"] and a["files"] == {}, "the fake renderer has no audio to write"
    assert (junk["ok"], junk["gate"]) == (False, "G1") and dup["dup_of"] == 0 and feat.calls == []
    assert set(out) >= set(protocol.VERSION_KEYS)


# ------------------------------------------------------------------------------------------ status codes
def test_malformed_requests_are_400_and_the_client_raises(tmp_path):
    with served(tmp_path) as (client, _, sock):
        status, doc = _raw(sock, "POST", "/score", b"{not json")
        assert status == 400 and "JSON" in doc["error"] and "trace" not in doc
        status, doc = _raw(sock, "POST", "/score", json.dumps({"protocol": protocol.PROTOCOL}).encode())
        assert status == 400 and "lacks" in doc["error"]
        bad = protocol.score_request(_items([EX]), palette_sha256=PAL.sha256)
        bad["items"][0]["brief"]["mood"] = "furious"
        status, doc = _raw(sock, "POST", "/score", json.dumps(bad).encode())
        assert status == 400 and "mood" in doc["error"]
        with pytest.raises(ScorerError) as err:
            client.request("POST", "/score", {"protocol": "lofi-scorer/0"})
        assert err.value.status == 400
        assert _raw(sock, "GET", "/nope")[0] == 404 and _raw(sock, "POST", "/nope")[0] == 404
        assert _raw(sock, "PUT", "/score")[0] == 405
        status, doc = _raw_once(sock, b"POST /score HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                                      b"Content-Length: %d\r\n\r\n{}" % (protocol.MAX_BODY_BYTES + 1))
        assert status == 413 and "at most" in doc["error"]
        assert client.health()["ok"], "the server keeps answering"


def test_a_backend_exception_is_500_with_its_trace_and_the_client_raises(tmp_path):
    feat = fakes.FakeFeaturizer(raise_for={canon_sha_of(EX)})
    with served(tmp_path, featurizer=feat) as (client, _, _sock):
        with pytest.raises(ScorerError) as err:
            client.score(_items([EX]), palette_sha256=PAL.sha256)
        assert err.value.status == 500 and "fake featurizer failure" in str(err.value)
        assert "Traceback" in err.value.body["trace"] and err.value.body["type"] == "RuntimeError"
        ok = client.score(_items([fixture_text("shaker_edits.ls")]), palette_sha256=PAL.sha256)
        assert ok["items"][0]["hard"] == 1, "one failed request does not break the next"


class Drifting(fakes.FakeRenderer):
    """The fake renderer, with `extra(job)` merged into every outcome (a worker's hashes, a manifest)."""

    def __init__(self, extra):
        super().__init__()
        self.extra = extra

    def render_many(self, jobs, on_done):
        super().render_many(jobs, lambda k, out: on_done(k, {**out, **self.extra(jobs[k])}))


def _manifest(canon_sha256):
    from lofi_rl.scoring.render_pool import render_config
    cfg = render_config()
    b = BRIEF_A
    return {"schema": cfg["schema"], "chain": cfg["chain"], "loop": {"canon_sha256": canon_sha256},
            "brief": {"bpm": b.bpm, "tonic": b.tonic, "mode": b.mode, "energy": b.energy},
            "palette": {"id": PAL.id, "sha256": PAL.sha256}}


@pytest.mark.parametrize("extra, match", [
    (lambda j: {"worker": {"pid": 1, "src_sha8": "deadbeef", "palette_sha256": PAL.sha256}}, "src_sha8"),
    (lambda j: {"worker": {"pid": 1, "src_sha8": None, "palette_sha256": "f" * 64}}, "palette_sha256"),
    (lambda j: {"manifest": _manifest("0" * 64)}, "manifest of clip"),
])
def test_a_render_from_a_drifted_worker_or_of_another_clip_is_a_500_and_never_cached(tmp_path, extra, match):
    with served(tmp_path, renderer=Drifting(extra)) as (client, scorer, _sock):
        with pytest.raises(ScorerError) as err:
            client.score(_items([EX]), palette_sha256=PAL.sha256)
        assert len(scorer.cache) == 0
    assert err.value.status == 500 and err.value.body["type"] == "RenderPoolDrift" and match in str(err.value)


def test_a_render_whose_worker_and_manifest_agree_is_scored(tmp_path):
    from lofi_rl.dsl import canon_sha, parse
    want = {"pid": 1, "src_sha8": fakes.FakeFeaturizer().src_sha8, "palette_sha256": PAL.sha256}
    sha = canon_sha(parse(EX, vocab=PAL.vocab(), mode=BRIEF_A.mode).ast)
    with served(tmp_path, renderer=Drifting(lambda j: {"worker": want, "manifest": _manifest(sha)})) as (client, _, _s):
        assert client.score(_items([EX]), palette_sha256=PAL.sha256)["items"][0]["hard"] == 1


def test_another_palette_or_pinned_version_is_409(tmp_path):
    with served(tmp_path) as (client, _, _sock):
        with pytest.raises(ScorerError) as err:
            client.score(_items([EX]), palette_sha256="0" * 64)
        assert err.value.status == 409 and "palette" in str(err.value)
        client.pinned = {**client.pinned, "rm_ver": "rm-from-yesterday"}
        with pytest.raises(ScorerError) as err:
            client.score(_items([EX]), palette_sha256=PAL.sha256)
        assert err.value.status == 409 and "rm_ver" in str(err.value)


def test_render_failures_zero_one_item_and_infrastructure_faults_are_not_cached(tmp_path):
    shaker = fixture_text("shaker_edits.ls")
    renderer = fakes.FakeRenderer(fail={canon_sha_of(EX): ("G7", "final -15.2 LUFS is not within 0.5 LU")},
                                  infra={canon_sha_of(shaker)})
    span = fixture_text("span1_dotted.ls")
    with served(tmp_path, renderer=renderer) as (client, scorer, _sock):
        out = client.score(_items([EX, shaker, span]), palette_sha256=PAL.sha256)
        again = client.score(_items([EX, shaker, span]), palette_sha256=PAL.sha256)
        canary = scorer.canary_job().canon
    g7, infra, ok = out["items"]
    assert (g7["hard"], g7["G"], g7["gate"]) == (0, 0.0, "G7") and "LUFS" in g7["diag"]["error"]
    assert (infra["hard"], infra["gate"]) == (0, "render-error") and "FluidSynthError" in infra["diag"]["error"]
    assert ok["hard"] == 1 and ok["G"] == 1.0
    assert [it["diag"]["cache"] for it in again["items"]] == ["mem", "miss", "mem"], "the fault is retried"
    assert renderer.calls.count(renderer.calls[1]) == 2
    assert renderer.calls.count(canary) == 2 and scorer.canary["ok"], "a lone fault, a healthy canary each time"


class Faulty(fakes.FakeRenderer):
    """The fake renderer, every render of which is an infrastructure fault while `broken` (FluidSynth gone)."""

    def __init__(self, broken=False, **kw):
        super().__init__(**kw)
        self.broken = broken

    def render_many(self, jobs, on_done):
        if not self.broken:
            return super().render_many(jobs, on_done)
        for k, j in enumerate(jobs):
            self.calls.append(j.canon)
            on_done(k, failure("render-error", "FileNotFoundError: fluidsynth is not on PATH", transient=True))


def test_a_renderer_that_cannot_render_stops_the_server_before_it_binds(monkeypatch, capsys):
    monkeypatch.setattr(fakes, "FakeRenderer", lambda **kw: Faulty(broken=True, **kw))
    with short_sock_dir() as d:
        rc = server.main(["--sock", str(d / "s.sock"), "--backend", "fake", "--workers", "1"])
        assert rc == 2 and not (d / "s.sock").exists()
    err = capsys.readouterr().err
    assert "the renderer cannot render (at start" in err and "fluidsynth is not on PATH" in err


def test_a_renderer_that_breaks_after_start_makes_score_a_500_not_a_batch_of_zeros(tmp_path):
    r = Faulty()
    with served(tmp_path, renderer=r) as (client, scorer, _sock):
        assert scorer.check_renderer("at start")["ok"]
        assert client.health()["renderer"]["canary"]["audio_sha256"], "/health reports the canary's pcm hash"
        r.broken = True
        with pytest.raises(ScorerError) as err:
            client.score(_items([EX, fixture_text("shaker_edits.ls")]), palette_sha256=PAL.sha256)
        h = client.health()
        assert len(scorer.cache) == 0
    assert err.value.status == 500 and err.value.body["type"] == "RendererError" and "cannot render" in str(err.value)
    assert (h["renderer"]["canary"]["ok"], h["renderer"]["canary"]["gate"]) == (False, "render-error")


# ------------------------------------------------------------------------------------------ the socket
def test_the_socket_is_mode_0600_and_owned_by_this_user(tmp_path):
    with served(tmp_path) as (_, _s, sock):
        st = os.stat(sock)
        assert stat.S_ISSOCK(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o600 and st.st_uid == os.getuid()


def test_the_path_rules_free_stale_live_and_not_a_socket(tmp_path):
    with short_sock_dir() as d:
        assert check_sock_path(d / "free.sock") == "free"
        stale = d / "stale.sock"
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(stale))
        s.close()                                       # bound and closed: the file stays, nobody listens
        assert stale.exists() and check_sock_path(stale) == "stale" and not stale.exists()
        live = d / "live.sock"
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(live))
        srv.listen(1)
        try:
            with pytest.raises(SocketPathError, match="live"):
                check_sock_path(live)
        finally:
            srv.close()
        plain = d / "plain.sock"
        plain.write_text("not a socket")
        with pytest.raises(SocketPathError, match="not a socket"):
            check_sock_path(plain)
        assert plain.read_text() == "not a socket", "never deleted"
        link = d / "link.sock"
        link.symlink_to(stale)
        with pytest.raises(SocketPathError, match="not a socket"):
            check_sock_path(link)
        with pytest.raises(SocketPathError, match="bytes"):
            check_sock_path(d / ("x" * 200))
        with pytest.raises(SocketPathError, match="not a directory"):
            check_sock_path(d / "missing-dir" / "s.sock")


def test_close_never_removes_a_socket_that_replaced_its_own(tmp_path):
    # Another server took the path over (the socket removed by hand, a second server started on the same --sock):
    # the first one's shutdown must leave the live socket alone, or the trainer's next /score finds nothing.
    with short_sock_dir() as d:
        sock = d / "s.sock"
        srv = ScorerServer(make_scorer(tmp_path), sock, logf=lambda m: None)
        srv.bind()
        other = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        other.bind(str(d / "r.sock"))                   # bound while sock still exists: the inodes differ
        try:
            os.replace(d / "r.sock", sock)
            assert os.lstat(sock).st_ino != srv._ident[1]
            srv.close()
            assert stat.S_ISSOCK(os.lstat(sock).st_mode), "the replacement is still there"
            os.unlink(sock)
            srv.close()                                 # a second close, with the path gone: no error
        finally:
            other.close()
        assert not sock.exists()


def test_the_client_raises_when_nothing_listens(tmp_path):
    with short_sock_dir() as d:
        c = ScorerClient(d / "none.sock", timeout=2)
        with pytest.raises(ScorerUnavailable):
            c.health()
        t = time.monotonic()
        with pytest.raises(ScorerUnavailable, match="not healthy within"):
            c.wait_healthy(timeout=1.0, poll=0.1)
        assert time.monotonic() - t < 5
        with pytest.raises(ScorerError, match="wait_healthy"):
            _ = c.ce_validated


def test_a_real_client_refuses_a_fake_backend(tmp_path):
    with served(tmp_path) as (_, _s, sock):
        with pytest.raises(ScorerError, match="fake") as err:
            ScorerClient(sock).wait_healthy(5)
        assert err.value.status == 200


def _server_proc(sock, *extra):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.Popen([sys.executable, "-m", "lofi_rl.scoring.server", "--sock", str(sock), "--backend",
                             "fake", "--workers", "1", *extra], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_stops_the_server_process_and_removes_its_socket(sig):
    with short_sock_dir() as d:
        sock = d / "s.sock"
        p = _server_proc(sock)
        try:
            ScorerClient(sock, allow_fake=True).wait_healthy(60, poll=0.1)
            assert stat.S_IMODE(os.stat(sock).st_mode) == 0o600
            out = ScorerClient(sock, allow_fake=True, timeout=30)
            out.wait_healthy(5)
            assert out.score(_items([EX]), palette_sha256=PAL.sha256)["items"][0]["hard"] == 1
            p.send_signal(sig)
            stdout, stderr = p.communicate(timeout=30)
        finally:
            if p.poll() is None:
                p.kill()
                p.communicate()
        assert p.returncode == 0, stderr[-2000:]
        assert not sock.exists(), "the socket is removed on shutdown"
        ready = json.loads(stdout.splitlines()[0])
        assert ready["event"] == "ready" and ready["backend"] == "fake" and ready["sock"] == str(sock)
        assert ready["canary"]["ok"] is True, "the canary rendered before the socket appeared"
        assert "shutting down" in stderr


def test_the_server_refuses_a_path_that_is_not_its_to_replace():
    with short_sock_dir() as d:
        plain = d / "s.sock"
        plain.write_text("keep me")
        p = _server_proc(plain)
        _, stderr = p.communicate(timeout=60)
        assert p.returncode == 2 and "not a socket" in stderr and plain.read_text() == "keep me"


def test_two_briefs_with_one_loop_are_two_clips(tmp_path):
    renderer = fakes.FakeRenderer()
    with served(tmp_path, renderer=renderer) as (client, _, _sock):
        items = _items([EX]) + [protocol.item_dict(1, EX, BRIEF_B, "596:1")]
        out = client.score(items, palette_sha256=PAL.sha256)
    a, b = out["items"]
    assert a["clip_id"] != b["clip_id"] and len(renderer.calls) == 2
