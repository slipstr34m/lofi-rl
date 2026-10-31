"""The rating server over HTTP (src/lofi_rl/ui/serve.py), with stdlib http.client against a server on an
ephemeral port over temp dirs: endpoints, 127.0.0.1 only, path traversal, the hidden manifest, appends and
fsyncs, resume, retract, the audit lock, schema errors, no model imports, and votes only under $LOFI_VOTES."""
import builtins
import http.client
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from lofi_rl.ui.manifest import HIDDEN_NAME, PUBLIC_NAME, build_demo, load_batch
from lofi_rl.ui.serve import (ROOT, ConfigError, RatingApp, VotesDirError, load_config, main, make_server,
                              prepare_votes_dir, resolve_batch)

FAST = dict(bpm=240.0, sr=8000)


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    return build_demo(tmp_path_factory.mktemp("inbox") / "2026-11-03" / "demo-bws4", "bws4", **FAST)


class Server:
    """A RatingApp on an ephemeral port, served from a thread."""

    def __init__(self, batch_dir, votes_dir, cfg=None):
        self.app = RatingApp(batch_dir, votes_dir, cfg or load_config(), quiet=True)
        self.srv = make_server(self.app, 0)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()
        self.m, self.h = self.app.manifest, self.app.hidden
        self.log = Path(self.app.ledger.path)

    def req(self, method, path, body=None, headers=None, ctype="application/json"):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = dict(headers or {})
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            h.setdefault("Content-Type", ctype)
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        out = (r.status, dict(r.getheaders()), r.read())
        c.close()
        return out

    def get_json(self, path):
        st, _, body = self.req("GET", path)
        return st, json.loads(body)

    def post(self, ev, **kw):
        st, _, body = self.req("POST", "/api/event", ev, **kw)
        return st, json.loads(body)

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.app.close()


@pytest.fixture
def server(batch, tmp_path):
    s = Server(batch, tmp_path / "votes")
    yield s
    s.close()


def _vote(s, r, best=0, worst=3, **extra):
    ids = r.ids
    return {"type": "bws4", "round": r.id, "best": ids[best], "worst": ids[worst],
            "clips": [{"clip": c, "pos": i + 1, "heard_bars": "11111111", "heard_ms": 8000} for i, c in enumerate(ids)],
            "last_heard": ids[-1], "switches": 0, "play_order": [1, 2, 3, 4], "autoplay_done": True,
            "t_ms": {"best": 1000, "worst": 2000, "commit": 3000}, "device": "airpods", **extra}


def _lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


# ----------------------------------------------------------------------------- binding and endpoints

def test_binds_127_0_0_1_only(server):
    assert server.srv.server_address[0] == "127.0.0.1"
    assert server.srv.socket.family == socket.AF_INET and server.srv.socket.getsockname()[0] == "127.0.0.1"
    for host in ("0.0.0.0", "", "::", "192.168.1.5"):
        with pytest.raises(ValueError):
            make_server(server.app, 0, host=host)


def test_page_and_api_endpoints(server):
    for path, ctype in [("/", "text/html"), ("/index.html", "text/html"), ("/static/app.js", "text/javascript"),
                        ("/static/audio.js", "text/javascript"), ("/manifest.json", "application/json"),
                        ("/api/config", "application/json"), ("/api/state", "application/json")]:
        st, hdrs, body = server.req("GET", path)
        assert st == 200 and hdrs["Content-Type"].startswith(ctype) and body, path
        assert hdrs["X-Content-Type-Options"] == "nosniff" and hdrs["Cache-Control"] == "no-store"
        assert "frame-ancestors 'none'" in hdrs["Content-Security-Policy"]
    st, cfg = server.get_json("/api/config")
    assert cfg["heard"] == {"bar_fraction": 0.75, "min_bars": 4, "toss_min_bars": 2}
    assert cfg["autoplay"]["bars_per_clip"] == 8 and cfg["nudge_minutes"] == 25 and cfg["ui"] == server.app.ui
    st, man = server.get_json("/manifest.json")
    assert [r["id"] for r in man["rounds"]] == [r.id for r in server.m.rounds]
    assert all(set(c) == {"id", "file"} for r in man["rounds"] for c in r["clips"])
    st, state = server.get_json("/api/state")
    assert state["current"] == server.m.rounds[0].id and state["total"] == 4 and not state["done"]
    c = server.m.rounds[0].clips[0]
    st, hdrs, body = server.req("GET", f"/clips/{c.file}")
    assert st == 200 and hdrs["Content-Type"] == "audio/flac" and body == (server.m.dir / c.file).read_bytes()
    st, hdrs, body = server.req("HEAD", "/")
    assert st == 200 and body == b"" and int(hdrs["Content-Length"]) > 0
    assert server.req("GET", "/nope")[0] == 404
    assert server.req("POST", "/api/state", {"a": 1})[0] == 404
    assert server.req("GET", "/api/event")[0] == 405
    assert server.req("PUT", "/", b"x")[0] == 405
    assert server.req("OPTIONS", "/api/event")[0] == 405


# ----------------------------------------------------------------------------- what is never served

TRAVERSAL = [
    "/manifest.hidden.json", "/clips/manifest.hidden.json", "/clips/manifest.json",
    "/clips/../manifest.hidden.json", "/clips/..%2fmanifest.hidden.json", "/clips/%2e%2e%2fmanifest.hidden.json",
    "/clips/%2E%2E/manifest.hidden.json", "/clips/%252e%252e%252fmanifest.hidden.json",
    "/clips/..%5cmanifest.hidden.json", "/clips/..\\manifest.hidden.json", "/clips/%2fetc%2fpasswd",
    "/clips//etc/passwd", "/clips/./manifest.json", "/static/../serve.py", "/static/..%2f..%2fserve.py",
    "/static/%2e%2e/votes.py", "/clips/k0%00.flac", "/..%2f..%2f..%2fetc%2fpasswd", "/clips/%c0%ae%c0%ae/x",
    "/%2e%2e/%2e%2e/configs/rating.yaml", "/clips/?file=manifest.hidden.json",
]


@pytest.mark.parametrize("path", TRAVERSAL)
def test_path_traversal_is_refused(server, path):
    st, _, body = server.req("GET", path)
    assert st in (400, 404), (path, st)
    assert set(json.loads(body)) == {"error"}


def test_the_hidden_manifest_is_never_served(server):
    hidden = (server.m.dir / HIDDEN_NAME).read_bytes()
    secrets = {info["clip"] for hr in server.h.rounds.values() for info in hr.clips.values()}
    secrets |= {hr.brief_id for hr in server.h.rounds.values() if hr.brief_id}
    paths = ["/", "/index.html", "/static/app.js", "/static/audio.js", "/manifest.json", "/api/config",
             "/api/state", "/api/audit"] + [f"/clips/{c.file}" for r in server.m.rounds for c in r.clips] + TRAVERSAL
    st, reply = server.post({"type": "session", "what": "start"})
    for _, _, body in [(0, 0, json.dumps(reply).encode())] + [server.req("GET", p) for p in paths]:
        assert hidden not in body
        for s in secrets:
            assert s.encode() not in body, s


def test_unlisted_files_are_404(batch, tmp_path):
    d = tmp_path / "2026-11-03" / batch.name
    shutil.copytree(batch, d)
    listed = next(iter(load_batch(d)[0].files))
    shutil.copy(d / listed, d / "k0123456789.flac")         # a real clip under a valid opaque name, unlisted
    (d / "notes.txt").write_text("private")
    s = Server(d, tmp_path / "votes")
    try:
        assert s.req("GET", f"/clips/{listed}")[0] == 200
        for name in ("k0123456789.flac", "notes.txt", PUBLIC_NAME, HIDDEN_NAME, ""):
            assert s.req("GET", f"/clips/{name}")[0] == 404, name
    finally:
        s.close()


def test_host_origin_and_body_checks(server):
    assert server.req("GET", "/", headers={"Host": "evil.example"})[0] == 421          # DNS rebinding
    assert server.req("GET", "/", headers={"Host": f"127.0.0.1:{server.port + 1}"})[0] == 421
    assert server.req("GET", "/", headers={"Host": f"localhost:{server.port}"})[0] == 200
    start = {"type": "session", "what": "start"}
    assert server.req("POST", "/api/event", start, headers={"Origin": "http://evil.example"})[0] == 403
    assert server.req("POST", "/api/event", start, headers={"Origin": "null"})[0] == 403
    assert server.req("POST", "/api/event", start, ctype="application/x-www-form-urlencoded")[0] == 415
    assert server.req("POST", "/api/event", b"{nope")[0] == 400
    assert server.req("POST", "/api/event", b" " * (64 * 1024 + 1))[0] == 413
    assert server.req("POST", "/api/event", b"")[0] == 400
    assert not _lines(server.log)                                   # nothing refused was written
    assert server.req("POST", "/api/event", start, headers={"Origin": f"http://127.0.0.1:{server.port}"})[0] == 200
    assert server.req("POST", "/api/event", start, ctype="text/plain;charset=UTF-8")[0] == 200   # sendBeacon
    assert len(_lines(server.log)) == 2


# ----------------------------------------------------------------------------- writing votes

def test_events_are_appended_and_fsynced(server, monkeypatch):
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
    r0 = server.m.rounds[0]
    for i, ev in enumerate([{"type": "session", "what": "start", "device": "airpods"},
                            {"type": "session", "what": "round_start", "round": r0.id}, _vote(server, r0)]):
        n = synced.count(server.app.ledger._log.fd)
        st, reply = server.post(ev)
        assert st == 200 and reply["ok"]
        assert synced.count(server.app.ledger._log.fd) == n + 1     # fsynced before the reply came back
        lines = _lines(server.log)                                  # re-read after the write
        assert len(lines) == i + 1 and lines[-1]["event_id"] == reply["event_id"]
    rec = lines[-1]
    hr = server.h.rounds[r0.id]
    assert rec["best"] == hr.real(r0.ids[0]) and rec["clips"][0]["oid"] == r0.ids[0]
    assert reply["state"]["current"] == server.m.rounds[1].id and reply["state"]["last_vote"]["round"] == r0.id
    assert "hidden" not in json.dumps(reply)                        # the reply never echoes hidden labels


def test_schema_errors_and_conflicts_over_http(server):
    r0 = server.m.rounds[0]
    st, body = server.post(_vote(server, r0, best=1, worst=1))
    assert st == 400 and "best and worst must be different" in body["error"]
    st, body = server.post({"type": "triage", "round": r0.id, "tier": "fine"})
    assert st == 400 and "takes bws4 votes" in body["error"]
    st, body = server.post(dict(_vote(server, r0), rm_score=0.8))
    assert st == 400 and "unknown field(s) ['rm_score']" in body["error"]
    assert server.post(_vote(server, r0))[0] == 200
    st, body = server.post(_vote(server, r0))
    assert st == 409 and "already has a vote" in body["error"] and body["state"]["settled"] == 1


def test_state_resumes_after_a_restart(batch, tmp_path):
    votes = tmp_path / "votes"
    s = Server(batch, votes)
    try:
        s.post({"type": "session", "what": "start"})
        s.post(_vote(s, s.m.rounds[0]))
        s.post(_vote(s, s.m.rounds[1], best=1, worst=2))
    finally:
        s.close()
    state = json.loads((votes / "state.json").read_text())
    assert state["batches"]["demo-bws4"]["current"] == s.m.rounds[2].id and state["session"]["id"]
    s2 = Server(batch, votes)
    try:
        st, cur = s2.get_json("/api/state")
        assert cur["current"] == s2.m.rounds[2].id and cur["settled"] == 2 and cur["index"] == 3
        st, reply = s2.post({"type": "session", "what": "start"})
        assert reply["state"]["current"] == s2.m.rounds[2].id
        assert reply["state"]["session"] == state["session"]["id"]       # restarted within 10 min: same session
    finally:
        s2.close()


def test_retract_over_http(server):
    r0 = server.m.rounds[0]
    st, first = server.post(_vote(server, r0))
    st, back = server.post({"type": "retract", "target": first["event_id"], "why": "undo"})
    assert st == 200 and back["state"]["current"] == r0.id and back["state"]["settled"] == 0
    st, body = server.post({"type": "retract", "target": first["event_id"]})
    assert st == 409 and "already taken back" in body["error"]
    assert server.post(_vote(server, r0, best=2, worst=1))[0] == 200
    assert [e["type"] for e in _lines(server.log)] == ["bws4", "retract", "bws4"]


def test_audit_is_locked_until_the_batch_is_done(server):
    st, body = server.get_json("/api/audit")
    assert st == 423 and body["locked"] and body["remaining"] == 4 and set(body) == {"locked", "remaining", "error"}
    anchor = next(r for r in server.m.rounds if server.h.rounds[r.id].kind == "anchor")
    broken = anchor.ids.index(server.h.rounds[anchor.id].anchor_clip)
    for r in server.m.rounds[:3]:
        if r is anchor:
            server.post(_vote(server, r, best=(broken + 1) % 4, worst=broken))
        else:
            server.post(_vote(server, r))
        assert server.get_json("/api/audit")[0] == 423
    st, last = server.post(_vote(server, server.m.rounds[3]))
    assert last["state"]["done"]
    st, audit = server.get_json("/api/audit")
    assert st == 200 and audit["voted"] == 4 and audit["anchors"] == {
        "rounds": 1, "caught": 1, "rate": 1.0, "micro": audit["anchors"]["micro"]}
    assert audit["repeats"]["rounds"] == 1 and audit["repeats"]["found"] == 1
    assert len(audit["list"]) == 4 and all(row["event_id"] for row in audit["list"])
    st, _ = server.post({"type": "retract", "target": audit["list"][0]["event_id"], "why": "audit"})
    assert server.get_json("/api/audit")[0] == 423                   # taken back: asked again, locked again


# ----------------------------------------------------------------------------- privacy and imports

def _tree(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in Path(root).rglob("*") if p.is_file())


def test_votes_are_written_only_under_the_votes_dir(batch, tmp_path, monkeypatch):
    votes = tmp_path / "votes"
    writes = []
    real_open, real_os_open, real_replace, real_mkdir = builtins.open, os.open, os.replace, os.mkdir
    wflags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def note(p):
        p = Path(os.fsdecode(p))
        writes.append(Path(os.path.realpath(p.parent)) / p.name)

    def rec_open(file, mode="r", *a, **k):
        if isinstance(file, (str, bytes, os.PathLike)) and set(mode) & set("wax+"):
            note(file)
        return real_open(file, mode, *a, **k)

    def rec_os_open(path, flags, *a, **k):
        if flags & wflags:
            note(path)
        return real_os_open(path, flags, *a, **k)

    monkeypatch.setattr(builtins, "open", rec_open)
    monkeypatch.setattr(io, "open", rec_open)
    monkeypatch.setattr(os, "open", rec_os_open)
    monkeypatch.setattr(os, "replace", lambda s, d, *a, **k: (note(d), real_replace(s, d, *a, **k))[1])
    monkeypatch.setattr(os, "mkdir", lambda p, *a, **k: (note(p), real_mkdir(p, *a, **k))[1])
    before = _tree(batch.parent.parent)
    s = Server(batch, votes)
    try:
        s.post({"type": "session", "what": "start"})
        s.post({"type": "session", "what": "round_start", "round": s.m.rounds[0].id})
        s.post({"type": "abandon", "round": s.m.rounds[0].id, "why": "esc"})
        for r in list(s.m.rounds[1:]) + [s.m.rounds[0]]:
            st, reply = s.post(_vote(s, r))
            assert st == 200
        assert s.get_json("/api/audit")[0] == 200
        s.post({"type": "retract", "target": reply["event_id"]})
        s.post({"type": "session", "what": "end"})
    finally:
        s.close()
    assert _tree(batch.parent.parent) == before                      # the batch folder is untouched
    root = Path(os.path.realpath(votes))
    assert writes and {w.name for w in writes} >= {"demo-bws4.jsonl", "state.json"}
    for w in writes:
        assert w == root or root in w.parents, w
        assert not str(w).startswith((str(ROOT / "data"), str(ROOT / "inbox")))


def test_votes_dir_must_not_sit_in_the_batch_or_the_public_repo(batch, tmp_path):
    with pytest.raises(VotesDirError, match="inside the batch folder"):
        prepare_votes_dir(batch / "votes", batch)
    stray = ROOT / "votes-not-allowed-here"
    with pytest.raises(VotesDirError, match="inside the repo but not under data/"):
        prepare_votes_dir(stray, batch)
    assert not stray.exists()
    made, created = prepare_votes_dir(tmp_path / "v", batch)
    assert created and oct(made.stat().st_mode & 0o777) == "0o700"


def _case_variant(p: Path):
    """The same folder with the case of its last name flipped, or None on a case-sensitive file system."""
    alt = p.with_name(p.name.swapcase())
    try:
        return alt if alt.name != p.name and os.path.samefile(alt, p) else None
    except OSError:
        return None


def test_votes_dir_guard_goes_by_the_folder_not_its_spelling(batch, tmp_path, monkeypatch):
    """The laptop's disk ignores case and /System/Volumes/Data/... is another name for /Users/...: any spelling of
    the repo (or of the batch folder) is still that folder (integrity-3)."""
    import lofi_rl.ui.serve as serve
    alias = _case_variant(ROOT)                                       # read-only checks on the real repo
    if alias is not None:
        assert serve._inside(alias / "votes", ROOT) and not serve._inside(alias / "votes", ROOT / "data")
        assert serve._inside(alias / "data" / "votes", ROOT / "data")
    firm = Path("/System/Volumes/Data") / str(ROOT).lstrip("/")
    if firm.is_dir() and os.path.samefile(firm, ROOT):
        assert serve._inside(firm / "votes", ROOT) and not serve._inside(firm / "votes", ROOT / "data")
    fake = tmp_path / "fakerepo"                                      # a stand-in repo: nothing lands in the real one
    (fake / "data").mkdir(parents=True)
    monkeypatch.setattr(serve, "ROOT", fake)
    falias = _case_variant(fake)
    if falias is not None:
        with pytest.raises(VotesDirError, match="inside the repo but not under data/"):
            prepare_votes_dir(falias / "votes", batch)
        assert sorted(p.name for p in fake.iterdir()) == ["data"]
        made, created = prepare_votes_dir(falias / "data" / "votes", batch)    # data/ spelled another way is fine
        assert created and os.path.samefile(made, fake / "data" / "votes")
    with pytest.raises(VotesDirError, match="inside the repo but not under data/"):
        prepare_votes_dir(tmp_path / "elsewhere" / ".." / "fakerepo" / "votes", batch)
    balias = _case_variant(batch)
    if balias is not None:
        with pytest.raises(VotesDirError, match="inside the batch folder"):
            prepare_votes_dir(balias / "votes", batch)
    if alias is None and falias is None and balias is None:
        pytest.skip("this file system is case-sensitive; only the plain spellings were checked")


def test_a_lone_surrogate_is_refused_with_400_and_nothing_is_written(server):
    """integrity-4: json.loads turns "\\ud800" into text UTF-8 cannot encode; the server refuses it up front."""
    r0 = server.m.rounds[0]
    assert server.post({"type": "session", "what": "round_start", "round": r0.id})[0] == 200
    for raw in (b'{"type": "session", "what": "start", "detail": "\\ud800"}',
                b'{"type": "session", "what": "pause", "detail": "ok \\udfff"}'):
        st, _, body = server.req("POST", "/api/event", raw)
        assert st == 400 and "plain text" in json.loads(body)["error"]
    assert [e.get("what") for e in _lines(server.log)] == ["round_start"]     # no abandon(no_commit), no start
    assert server.get_json("/api/state")[1]["current"] == r0.id


def test_the_server_imports_no_model():
    code = ("import sys, json; before = set(sys.modules); "
            "import lofi_rl.ui.serve, lofi_rl.ui.manifest, lofi_rl.ui.votes, lofi_rl.ui.audit; "
            "print(json.dumps(sorted(set(sys.modules) - before)))")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    mods = json.loads(out.stdout)
    tops = {m.split(".")[0] for m in mods}
    heavy = {"torch", "transformers", "numpy", "scipy", "soundfile", "laion_clap", "audiobox_aesthetics", "pedalboard",
             "vllm", "trl", "peft", "sklearn", "librosa", "torchaudio", "pyloudnorm", "mido", "pretty_midi"}
    assert not tops & heavy
    allowed = set(sys.stdlib_module_names) | {"yaml", "_yaml", "lofi_rl", "cython_runtime"}   # PyYAML's C parser
    assert {t for t in tops if not t.startswith("_cython_")} <= allowed, tops - allowed
    assert {m for m in mods if m.startswith("lofi_rl")} <= {"lofi_rl", "lofi_rl.ui", "lofi_rl.ui.serve",
                                                           "lofi_rl.ui.manifest", "lofi_rl.ui.votes",
                                                           "lofi_rl.ui.audit"}


# ----------------------------------------------------------------------------- config, batch choice, CLI

def test_config_file_and_errors(tmp_path):
    cfg = load_config()
    assert cfg["port"] == 8596 and cfg["heard"]["bar_fraction"] == 0.75 and cfg["heard"]["min_bars"] == 4
    assert cfg["autoplay"]["bars_per_clip"] == 8 and cfg["nudge_minutes"] == 25 and cfg["micro_gap_minutes"] == 10
    for text, msg in [("colour: blue\n", "unknown key 'colour'"), ("heard: {min_bars: 0}\n", "heard.min_bars"),
                      ("port: 80\n", "port"), ("heard: {fraction: 0.5}\n", "unknown key heard.fraction"),
                      ("playlist_plays: [4, 2]\n", "playlist_plays"), ("default_device: car\n", "default_device")]:
        p = tmp_path / "bad.yaml"
        p.write_text(text)
        with pytest.raises(ConfigError, match=msg):
            load_config(p)


def test_the_oldest_unfinished_batch_is_served_by_default(tmp_path):
    inbox, votes = tmp_path / "inbox", tmp_path / "votes"
    votes.mkdir()
    a = build_demo(inbox / "2026-11-03" / "b-old", "triage", rounds=2, **FAST)
    b = build_demo(inbox / "2026-11-04" / "b-new", "triage", rounds=2, **FAST)
    assert resolve_batch(None, inbox, votes) == a
    m, h = load_batch(a)
    from lofi_rl.ui.votes import Ledger
    led = Ledger(votes, m, h)
    for r in m.rounds:
        led.submit({"type": "triage", "round": r.id, "tier": "fine",
                    "clips": [{"clip": r.ids[0], "pos": 1, "heard_bars": "11111111"}]})
    led.close()
    assert resolve_batch(None, inbox, votes) == b
    assert resolve_batch("b-old", inbox, votes) == a
    assert resolve_batch(str(b), inbox, votes) == b


def test_cli_errors(batch, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOFI_VOTES", str(tmp_path / "votes"))
    monkeypatch.setenv("LOFI_INBOX", str(tmp_path / "empty-inbox"))
    assert main(["--no-browser"]) == 2
    assert "nothing to rate" in capsys.readouterr().err
    assert main([str(tmp_path / "missing"), "--no-browser"]) == 2
    busy = Server(batch, tmp_path / "other-votes")
    try:
        assert main([str(batch), "--no-browser", "--port", str(busy.port)]) == 1
        assert "cannot listen on 127.0.0.1" in capsys.readouterr().err
    finally:
        busy.close()


def test_batch_done_is_announced_once(batch, tmp_path, capsys):
    app = RatingApp(batch, tmp_path / "votes", load_config(), quiet=False)
    try:
        replies = [app.submit(_vote(app, r)) for r in app.manifest.rounds]
        assert [r["state"]["done"] for r in replies] == [False, False, False, True]
        assert json.loads((tmp_path / "votes" / "state.json").read_text())["batches"]["demo-bws4"]["done"]
        out = capsys.readouterr().out
        assert out.count("rate: batch_done demo-bws4 (4 rounds)") == 1
        app.submit({"type": "retract", "target": replies[-1]["event_id"]})
        app.submit(_vote(app, app.manifest.rounds[-1]))                 # done again: announced again
        assert capsys.readouterr().out.count("batch_done") == 1
    finally:
        app.close()
