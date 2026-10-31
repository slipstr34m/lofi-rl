"""The page half of app.js (startRound, auto-play, mark and skip, devicechange, the break nudge, listenInfo), run
unchanged in JavaScriptCore against a real rating server.

The stand-in browser is small: DOM stubs, a fake Web Audio whose clock runs only while the context runs, timers
on a fake wall clock, and fetch() bridged to a real RatingApp over HTTP on 127.0.0.1 (each request is one stdout
line; its reply comes back as one stdin line). The server's clock follows the page's wall clock, so sessions,
micro-sessions and the vote lines are the real ones. Votes go to temp dirs. Skipped where no jsc is installed."""
import http.client
import json
import re
import shutil
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lofi_rl.ui.manifest import build_demo
from lofi_rl.ui.serve import RatingApp, load_config, make_server

STATIC = Path(__file__).resolve().parents[1] / "src" / "lofi_rl" / "ui" / "static"
JSC_PATHS = ["/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc",
             "/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc"]
JSC = next((p for p in JSC_PATHS if Path(p).exists()), None) or shutil.which("jsc")
pytestmark = pytest.mark.skipif(JSC is None, reason="no JavaScriptCore shell on this machine")

TZ = timezone(timedelta(hours=-8))
START = datetime(2026, 11, 10, 21, 0, 0, tzinfo=TZ)
FAST = dict(bpm=240.0, sr=8000)          # 8 bars of 1 s

BROWSER = r"""
// ---- clocks: the wall clock (Date.now, performance.now) and the audio clock (runs only while the context runs)
var WALL = { t: START_MS }, CLOCK = { t: 0 };
Date.now = function () { return WALL.t; };
var performance = { now: function () { return WALL.t; } };

// ---- timers on the wall clock (an interval fires at most once per step, like a busy page)
var TIMERS = [], NEXT_ID = 1;
function setTimeout(fn, ms) { var id = NEXT_ID++; TIMERS.push({ id: id, at: WALL.t + Math.max(0, +ms || 0), fn: fn, every: 0 }); return id; }
function setInterval(fn, ms) { var e = Math.max(1, +ms || 0), id = NEXT_ID++; TIMERS.push({ id: id, at: WALL.t + e, fn: fn, every: e }); return id; }
function clearTimeout(id) { TIMERS = TIMERS.filter(function (x) { return x.id !== id; }); }
var clearInterval = clearTimeout;
function fireDue() {
  for (var n = 0; n < 100000; n++) {
    var due = null;
    for (var i = 0; i < TIMERS.length; i++) if (TIMERS[i].at <= WALL.t && (!due || TIMERS[i].at < due.at)) due = TIMERS[i];
    if (!due) return;
    if (due.every) { due.at += due.every; if (due.at <= WALL.t) due.at = WALL.t + due.every; }
    else TIMERS.splice(TIMERS.indexOf(due), 1);
    due.fn();
    drainMicrotasks();
  }
  throw new Error('timers never settle');
}
function settle() { for (var i = 0; i < 4; i++) { drainMicrotasks(); fireDue(); } drainMicrotasks(); }
function advance(sec, step) {                  // wall time passes; audio time too while the context runs
  step = step || 0.1;
  for (var i = 0, n = Math.round(sec / step); i < n; i++) {
    WALL.t += step * 1000;
    if (CTX && CTX.state === 'running') CLOCK.t += step;
    fireDue(); drainMicrotasks();
  }
}
function advanceAudio(to, step) {              // until the audio clock reaches `to` (the context must run)
  step = step || 0.1;
  while (CLOCK.t < to - 1e-9) {
    if (!CTX || CTX.state !== 'running') throw new Error('the audio clock is stopped at ' + CLOCK.t);
    var dt = Math.min(step, to - CLOCK.t);
    WALL.t += dt * 1000; CLOCK.t += dt;
    fireDue(); drainMicrotasks();
  }
}

// ---- Web Audio: params record their automation and evaluate it (curves and ramps interpolate)
function Param(v) { this.v0 = v; this.events = []; }
Object.defineProperty(Param.prototype, 'value', {
  get: function () { return this.at(CLOCK.t); },
  set: function (v) { if (!this.events.length) this.v0 = v; this.events.push({ k: 'set', v: v, t: CLOCK.t }); }
});
Param.prototype.setValueAtTime = function (v, t) { this.events.push({ k: 'set', v: v, t: t }); return this; };
Param.prototype.linearRampToValueAtTime = function (v, t) { this.events.push({ k: 'lin', v: v, t: t }); return this; };
Param.prototype.setValueCurveAtTime = function (c, t, d) {
  this.events.forEach(function (e) {
    if (e.k === 'curve' && t < e.t + e.d && e.t < t + d) throw new Error('NotSupportedError: curves overlap');
  });
  this.events.push({ k: 'curve', c: Array.prototype.slice.call(c), t: t, d: d }); return this;
};
Param.prototype.cancelScheduledValues = function (t) { this.events = this.events.filter(function (e) { return e.t < t; }); return this; };
Param.prototype.at = function (t) {
  var evs = this.events.slice().sort(function (a, b) { return a.t - b.t; }), pv = this.v0, pt = -Infinity;
  for (var i = 0; i < evs.length; i++) {
    var e = evs[i];
    if (e.k === 'lin') {
      if (t < e.t) return pt === -Infinity || e.t <= pt ? pv : pv + (e.v - pv) * (t - pt) / (e.t - pt);
      pv = e.v; pt = e.t; continue;
    }
    if (t < e.t) return pv;
    if (e.k === 'set') { pv = e.v; pt = e.t; continue; }
    if (t >= e.t + e.d) { pv = e.c[e.c.length - 1]; pt = e.t + e.d; continue; }
    var x = (t - e.t) / e.d * (e.c.length - 1), k = Math.floor(x);
    return e.c[k] + (e.c[Math.min(k + 1, e.c.length - 1)] - e.c[k]) * (x - k);
  }
  return pv;
};
function ANode() { this.outs = []; }
ANode.prototype.connect = function (n) { this.outs.push(n); if (n.ins) n.ins.push(this); return n; };
ANode.prototype.disconnect = function () { this.outs = []; };
function GainNode() { ANode.call(this); this.gain = new Param(1); }
GainNode.prototype = Object.create(ANode.prototype);
var SOURCES = [];
function SourceNode(ctx) { ANode.call(this); this.ctx = ctx; this.loop = false; this.buffer = null; this.startedAt = null; this.offset = null; this.stoppedAt = null; }
SourceNode.prototype = Object.create(ANode.prototype);
SourceNode.prototype.start = function (t, off) { this.startedAt = t; this.offset = off || 0; };
SourceNode.prototype.stop = function (t) { this.stoppedAt = t === undefined ? CLOCK.t : t; };
var CTX = null;
function FakeAudioContext(opts) {
  this.sampleRate = (opts && opts.sampleRate) || 48000; this.state = 'suspended'; this.onstatechange = null;
  this.destination = new ANode(); this.destination.ins = []; this.outputLatency = 0; this.baseLatency = 0;
  CTX = this;
}
Object.defineProperty(FakeAudioContext.prototype, 'currentTime', { get: function () { return CLOCK.t; } });
FakeAudioContext.prototype.createGain = function () { return new GainNode(); };
FakeAudioContext.prototype.createBufferSource = function () { var s = new SourceNode(this); SOURCES.push(s); return s; };
FakeAudioContext.prototype.setState = function (s) {
  if (this.state === s) return;
  this.state = s;
  if (typeof this.onstatechange === 'function') this.onstatechange();
};
FakeAudioContext.prototype.resume = function () { this.setState('running'); return Promise.resolve(); };
FakeAudioContext.prototype.suspend = function () { this.setState('suspended'); return Promise.resolve(); };
FakeAudioContext.prototype.decodeAudioData = function () { return Promise.resolve({ duration: LOOP_S, sampleRate: this.sampleRate }); };
var AudioContext = FakeAudioContext;

// ---- the DOM and the rest of the window
function El(tag, id) {
  this.tagName = String(tag || 'div').toUpperCase(); this.id = id || null; this.textContent = '';
  this.hidden = !!id && HIDDEN_IDS.indexOf(id) >= 0;        // as index.html has them
  this.className = ''; this.dataset = {}; this.children = []; this.value = ''; this.listeners = {};
}
Object.defineProperty(El.prototype, 'innerHTML', { get: function () { return ''; }, set: function () { this.children = []; this.textContent = ''; } });
El.prototype.appendChild = function (c) { this.children.push(c); return c; };
El.prototype.addEventListener = function (k, fn) { (this.listeners[k] = this.listeners[k] || []).push(fn); };
El.prototype.querySelector = function () { return null; };
El.prototype.blur = function () {};
El.prototype.closest = function () { return null; };
var ELS = {}, DOC_LISTENERS = {}, WIN_LISTENERS = {}, DEVICE_LISTENERS = [];
var document = {
  readyState: 'complete',
  getElementById: function (id) { return ELS[id] || (ELS[id] = new El('div', id)); },
  createElement: function (tag) { return new El(tag); },
  addEventListener: function (k, fn) { (DOC_LISTENERS[k] = DOC_LISTENERS[k] || []).push(fn); }
};
function addEventListener(k, fn) { (WIN_LISTENERS[k] = WIN_LISTENERS[k] || []).push(fn); }
var location = { reload: function () { throw new Error('the page asked to reload'); } };
var navigator = { mediaDevices: { addEventListener: function (k, fn) { if (k === 'devicechange') DEVICE_LISTENERS.push(fn); } } };
var localStorage = { d: {}, getItem: function (k) { return k in this.d ? this.d[k] : null; }, setItem: function (k, v) { this.d[k] = String(v); } };
function confirm() { return true; }
function Blob(parts, opts) { this.parts = parts; this.type = opts && opts.type; }
var crypto = { getRandomValues: function (b) { for (var i = 0; i < b.length; i++) b[i] = Math.floor(Math.random() * 256); return b; } };
var console = { log: function () { print('log: ' + Array.prototype.join.call(arguments, ' ')); },
                warn: function () { print('warn: ' + Array.prototype.join.call(arguments, ' ')); } };

// ---- fetch(), bridged to the real server
function fetch(url, opts) {
  opts = opts || {};
  var req = { method: opts.method || 'GET', url: String(url), wall: WALL.t };
  if (opts.body !== undefined && opts.body !== null) req.body = String(opts.body);
  if (opts.headers && opts.headers['Content-Type']) req.ctype = opts.headers['Content-Type'];
  print('@@REQ ' + JSON.stringify(req));
  var res = JSON.parse(readline());
  return Promise.resolve({
    ok: res.status >= 200 && res.status < 300, status: res.status,
    json: function () { try { return Promise.resolve(JSON.parse(res.body)); } catch (e) { return Promise.reject(e); } },
    arrayBuffer: function () { return Promise.resolve(new ArrayBuffer(8)); }
  });
}

// ---- what a scenario does and sees
function out(name, value) { print('@@OUT ' + JSON.stringify({ name: name, value: value })); }
function key(code, ch) {
  var ev = { code: code, key: ch === undefined ? '' : ch, repeat: false, metaKey: false, ctrlKey: false, altKey: false,
             target: document.getElementById('page-body'), preventDefault: function () {} };
  (DOC_LISTENERS.keydown || []).slice().forEach(function (fn) { fn(ev); });
  settle();
}
function deviceChange() { DEVICE_LISTENERS.slice().forEach(function (fn) { fn({}); }); settle(); }
function reloadPage() {                        // the tab reloads: pagehide, then a fresh copy of the page boots
  (WIN_LISTENERS.pagehide || []).slice().forEach(function (fn) { fn({ persisted: false }); });
  settle();
  DOC_LISTENERS = {}; WIN_LISTENERS = {}; DEVICE_LISTENERS = []; TIMERS = []; ELS = {};
  if (CTX) CTX.state = 'closed';
  CTX = null;
  load(APP_JS);
  settle();
}
function status() { return document.getElementById('status').textContent; }
function banner() { var b = document.getElementById('banner'); return { shown: !b.hidden, kind: b.dataset.kind || null, text: b.textContent }; }
function masterGain() { return CTX && CTX.destination.ins.length ? CTX.destination.ins[0].gain.at(CLOCK.t) : null; }
function level(t) {                            // how loud the output is: 0 while nothing can be heard
  if (t === undefined) t = CLOCK.t;
  if (!CTX || CTX.state !== 'running') return 0;
  var sum = 0;
  SOURCES.forEach(function (s) {
    if (s.ctx !== CTX || s.startedAt === null || s.startedAt > t || (s.stoppedAt !== null && s.stoppedAt <= t)) return;
    var g = 1, n = s.outs[0];
    while (n && n !== CTX.destination) { g *= n.gain.at(t); n = n.outs[0]; }
    if (n === CTX.destination) sum += g;
  });
  return Math.round(sum * 1000) / 1000;
}
function roundSources(k) { return SOURCES.filter(function (s) { return s.ctx === CTX; }).slice(-k); }
function upAt(k, t) {                          // the clip (0-based) whose gain is up at t among the newest k sources
  var src = roundSources(k);
  for (var i = 0; i < src.length; i++) if (src[i].outs[0].gain.at(t) >= 0.5) return i;
  return -1;
}
"""


class WallClock:
    """The server's clock, set from the page's wall clock with every request."""

    def __init__(self):
        self.ms = START.timestamp() * 1000

    def __call__(self):
        return datetime.fromtimestamp(self.ms / 1000, TZ)


class PageRun:
    def __init__(self, outs, log, events, app):
        self.outs, self.log, self.events, self.m, self.h = outs, log, events, app.manifest, app.hidden

    def warnings(self):
        return [line for line in self.log if line.startswith(("warn:", "Exception", "ERROR"))]

    def votes(self):
        return [e for e in self.events if e["type"] == self.m.mode]

    def sessions(self, what):
        return [e for e in self.events if e["type"] == "session" and e["what"] == what]

    def real(self, round_index, pos):
        r = self.m.rounds[round_index]
        return self.h.rounds[r.id].real(r.ids[pos - 1])


def _forward(port, req):
    method, url = req.get("method") or "GET", req["url"]
    if url.startswith("/clips/") and method == "GET":
        method = "HEAD"                                   # the fake decoder needs no bytes, only that it is served
    headers = {"Host": f"127.0.0.1:{port}"}
    body = None
    if req.get("body") is not None:
        body = req["body"].encode("utf-8")
        headers["Content-Type"] = req.get("ctype") or "text/plain;charset=UTF-8"
        headers["Origin"] = f"http://127.0.0.1:{port}"
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        c.request(method, url, body=body, headers=headers)
        r = c.getresponse()
        data = r.read()
    finally:
        c.close()
    return {"status": r.status, "body": data.decode("utf-8", "replace")}


def run_page(tmp_path, mode, scenario, *, timeout=90):
    batch = build_demo(tmp_path / "inbox" / "2026-11-10" / f"demo-{mode}", mode, **FAST)
    clock = WallClock()
    app = RatingApp(batch, tmp_path / "votes", load_config(), clock=clock, quiet=True)
    srv = make_server(app, 0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    loop_s = app.manifest.bars * 240.0 / app.manifest.rounds[0].bpm
    files = []
    hidden_ids = re.findall(r'<\w+ id="([a-z-]+)"[^>]*\bhidden\b', (STATIC / "index.html").read_text())
    for i, text in enumerate([f"var START_MS = {clock.ms!r}; var LOOP_S = {loop_s!r}; "
                              f"var APP_JS = {json.dumps(str(STATIC / 'app.js'))}; "
                              f"var HIDDEN_IDS = {json.dumps(hidden_ids)};\n" + BROWSER, None, None, scenario]):
        if text is None:
            files.append(str(STATIC / ("audio.js" if i == 1 else "app.js")))
        else:
            f = tmp_path / f"page{i}.js"
            f.write_text(text)
            files.append(str(f))
    proc = subprocess.Popen([JSC, *files], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    killer = threading.Timer(timeout, proc.kill)
    killer.start()
    outs, log = {}, []
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("@@REQ "):
                req = json.loads(line[6:])
                clock.ms = req["wall"]
                proc.stdin.write(json.dumps(_forward(port, req)) + "\n")
                proc.stdin.flush()
            elif line.startswith("@@OUT "):
                o = json.loads(line[6:])
                outs[o["name"]] = o["value"]
            else:
                log.append(line)
        rc = proc.wait(timeout=10)
    finally:
        killer.cancel()
        if proc.poll() is None:                        # an error above: never leave jsc waiting on its stdin
            proc.kill()
            proc.wait(timeout=10)
        srv.shutdown()
        srv.server_close()
        app.close()
    assert rc == 0, "jsc failed:\n" + "\n".join(log)
    events = [json.loads(x) for x in app.ledger.path.read_text().splitlines() if x.strip()]
    return PageRun(outs, log, events, app)


def _ts(e):
    return datetime.fromisoformat(e["ts"])


# ----------------------------------------------------------------------------- auto-play, skip, what the vote logs

AUTOPLAY = r"""
settle();
out('loaded', status());
key('Space', ' ');
var src = roundSources(4), t0 = src[0].startedAt;
out('together', src.every(function (s) { return s.startedAt === t0 && s.loop && s.offset === 0; }));
out('plan', [4.5, 12.5, 20.5, 28.5].map(function (b) { return upAt(4, t0 + b); }));
out('lines', [8, 16, 24].map(function (b) { return [upAt(4, t0 + b - 0.003), upAt(4, t0 + b + 0.003)]; }));
advanceAudio(t0 + 10.5);
key('KeyW', 'w');                    // best = clip 2, the clip that is playing: on to the next unheard clip
advance(0.1);
out('skip', [11.5, 18.5, 19.5, 26.5].map(function (t) { return upAt(4, t0 + t); }));
advanceAudio(t0 + 22);
key('KeyZ', 'z');                    // worst = clip 1
key('Enter', 'Enter');
advance(0.5);
out('next', { status: status(), started: roundSources(4).every(function (s) { return s.startedAt > t0 + 22; }),
              level: level() });
"""


def test_autoplay_runs_1_to_k_and_marking_the_playing_clip_skips_on(tmp_path):
    """taste_pipeline §1 Round mechanics: auto-play 1 -> k, 8 bars each, switching on bar lines; marking the clip
    that is playing hands the next unheard clip the rest of the bar plus 8 bars; the vote logs each clip's own
    heard bars; Enter starts the next round."""
    run = run_page(tmp_path, "bws4", AUTOPLAY)
    o = run.outs
    assert o["loaded"] == "Press Space to play." and o["together"]
    assert o["plan"] == [0, 1, 2, 3]                                 # clip 1, 2, 3, 4 in bars 1-8, 9-16, ...
    assert o["lines"] == [[0, 1], [1, 2], [2, 3]]                    # each switch on its bar line
    assert o["skip"] == [2, 2, 3, 3]                                 # clip 3 to bar 19, then clip 4
    (v,) = run.votes()
    assert v["best"] == run.real(0, 2) and v["worst"] == run.real(0, 1)
    assert [c["heard_bars"] for c in v["clips"]] == ["11111111", "11000000", "11111111", "00011100"]
    for got, want in zip([c["heard_ms"] for c in v["clips"]], [8000, 2532.5, 8467.5, 3000]):
        assert abs(got - want) <= 1, v["clips"]                      # the skip ramp is centred at t0 + 10.5325
    assert v["play_order"] == [1, 2, 3, 4] and v["last_heard"] == run.real(0, 4) and not v["autoplay_done"]
    assert set(v["t_ms"]) == {"best", "worst", "commit"} and v["micro"] == 1
    assert [e.get("round") for e in run.sessions("round_start")] == [run.m.rounds[0].id, run.m.rounds[1].id]
    assert o["next"]["started"] and o["next"]["level"] == 1 and o["next"]["status"].startswith("Playing 1")
    assert not run.warnings(), run.log


# ----------------------------------------------------------------------------- triage: the wrap, heartbeats, the nudge

TRIAGE_LONG = r"""
settle();
key('Space', ' ');
var t0 = roundSources(1)[0].startedAt;
advanceAudio(t0 + 9.9);
out('s9_9', status());
advanceAudio(t0 + 10.1);
out('s10_1', status());
advance(1499 - (WALL.t - START_MS) / 1000, 1);      // listen on to 24:59 after the page loaded
out('b24_59', banner());
advance(2, 1);
out('b25_01', banner());
key('KeyA', 'a');
key('Enter', 'Enter');
"""


def test_triage_wraps_into_bars_1_2_and_a_long_listen_stays_one_micro_session(tmp_path):
    """Q4: a single clip plays 8 bars, then bars 1-2 again (auto-play ends at bar 10). Playback heartbeats keep 25
    minutes of listening in one micro-session (spec-8), and the break nudge comes at 25 minutes (BS.1116)."""
    run = run_page(tmp_path, "triage", TRIAGE_LONG)
    o = run.outs
    assert "auto-play" in o["s9_9"] and "auto-play" not in o["s10_1"]
    assert not (o["b24_59"]["shown"] and o["b24_59"]["kind"] == "nudge")
    assert o["b25_01"]["shown"] and o["b25_01"]["kind"] == "nudge" and "25 minutes" in o["b25_01"]["text"]
    (nudge,) = run.sessions("nudge")
    assert nudge["ts"] == "2026-11-10T21:25:00-08:00"
    beats = run.sessions("playing")
    assert len(beats) >= 4
    gaps = [(_ts(b) - _ts(a)).total_seconds() for a, b in zip(run.events, run.events[1:])]
    assert max(gaps) <= 5 * 60 + 2, gaps                                # never a 10-minute gap while it plays
    (v,) = run.votes()
    assert v["tier"] == "toss" and v["autoplay_done"] and v["clips"][0]["heard_bars"] == "11111111"
    assert (v["session"], v["micro"]) == ("s1110a", 1)
    assert not run.warnings(), run.log


# ----------------------------------------------------------------------------- the output changes (spec-1, M4)

DEVICE = r"""
settle();
key('Space', ' ');
var t0 = roundSources(4)[0].startedAt;
advanceAudio(t0 + 10);
key('KeyQ', 'q');                                   // best = 1
key('KeyV', 'v');                                   // worst = 4
deviceChange();                                     // the AirPods disconnect
advance(0.5);
function look() { return { ctx: CTX.state, master: masterGain(), level: level(), banner: banner(), status: status() }; }
out('held', look());
key('Enter', 'Enter');                              // he saves: the next round loads, but waits
advance(2);
out('enter', look());
key('Digit5', '5');
advance(1);
out('five', look());
key('Digit2', '2');
advance(1);
out('two', look());
key('Backspace', 'Backspace');                      // take the vote back: round 1 comes back, still waiting
key('Backspace', 'Backspace');
advance(1);
out('undo', look());
key('Space', ' ');                                  // he checks "Listening on" and presses Space
advance(1);
out('space', look());
"""


def test_a_devicechange_holds_playback_until_space(tmp_path):
    """taste_pipeline §1 Logging: a devicechange pauses playback. Nothing plays again (not the next round after
    Enter, not 5, not 1-4, not the round reopened by Backspace) until he presses Space."""
    run = run_page(tmp_path, "bws4", DEVICE)
    o = run.outs
    for step in ("held", "enter", "five", "two", "undo"):
        assert o[step]["level"] == 0 and o[step]["ctx"] != "running", (step, o[step])
        assert o[step]["banner"]["shown"] and o[step]["banner"]["kind"] == "device", (step, o[step])
        assert "Space" in o[step]["status"], (step, o[step])
    assert o["held"]["master"] == 0
    assert o["space"]["ctx"] == "running" and o["space"]["master"] == 1 and o["space"]["level"] == 1
    assert not o["space"]["banner"]["shown"]
    kinds = [(e["type"], e.get("what")) for e in run.events]
    assert kinds == [("session", "start"), ("session", "round_start"), ("session", "devicechange"), ("bws4", None),
                     ("retract", None), ("session", "round_start")]
    assert run.events[-1]["round"] == run.m.rounds[0].id               # round 1 again, started by Space
    assert not run.warnings(), run.log


DEVICE_THEN_ESC = r"""
settle();
key('Space', ' ');
var t0 = roundSources(4)[0].startedAt;
advanceAudio(t0 + 3);
deviceChange();
advance(0.5);
key('Escape', 'Escape');                            // he stops
advance(1);
out('ended', { level: level(), banner: banner(), ended: !document.getElementById('ended').hidden });
key('Space', ' ');                                  // Space goes on, and it is the Space that lifts the hold
advance(1);
out('space', { level: level(), banner: banner(), ctx: CTX.state });
"""


def test_space_after_esc_goes_on_past_a_hold(tmp_path):
    run = run_page(tmp_path, "bws4", DEVICE_THEN_ESC)
    o = run.outs
    assert o["ended"]["ended"] and o["ended"]["level"] == 0 and o["ended"]["banner"]["kind"] == "device"
    assert o["space"]["level"] == 1 and o["space"]["ctx"] == "running" and not o["space"]["banner"]["shown"]
    assert [(e["type"], e.get("what") or e.get("why")) for e in run.events] == [
        ("session", "start"), ("session", "round_start"), ("session", "devicechange"), ("abandon", "esc"),
        ("session", "end"), ("session", "round_start")]
    assert run.events[-1]["round"] == run.m.rounds[1].id               # round 1 went to the end of the batch
    assert not run.warnings(), run.log


# ----------------------------------------------------------------------------- the system stops the audio (spec-4)

INTERRUPT = r"""
settle();
key('Space', ' ');
var t0 = roundSources(1)[0].startedAt;
advanceAudio(t0 + 1);
CTX.setState('interrupted');                        // e.g. Safari: another app takes the audio
settle();
out('interrupted', { master: masterGain(), banner: banner(), status: status() });
advance(0.2);
CTX.setState('running');                            // the interruption ends; the browser runs the context again
settle();
advance(6);
out('after', { level: level(), status: status() });
key('Space', ' ');
advance(3);
out('space', { level: level(), banner: banner() });
key('KeyS', 's');
key('Enter', 'Enter');
"""


def test_after_a_system_interruption_nothing_plays_until_space(tmp_path):
    """What is audible and what is logged agree: while the page says paused, the output is silent and nothing
    counts as heard; Space goes on."""
    run = run_page(tmp_path, "triage", INTERRUPT)
    o = run.outs
    assert o["interrupted"]["master"] == 0 and o["interrupted"]["banner"]["shown"]
    assert o["after"]["level"] == 0 and o["after"]["status"].startswith("Paused")
    assert o["space"]["level"] == 1 and not o["space"]["banner"]["shown"]
    (v,) = run.votes()
    assert v["clips"][0]["heard_bars"] == "11110000" and abs(v["clips"][0]["heard_ms"] - 4000) <= 50, v["clips"]
    pauses = run.sessions("pause")
    assert len(pauses) == 1 and pauses[0]["detail"] == "system: interrupted"
    assert len(run.sessions("resume")) == 1
    assert not run.warnings(), run.log


# ----------------------------------------------------------------------------- a reload keeps the nudge clock (spec-9)

RELOAD = r"""
settle();
key('Space', ' ');
advance(20 * 60, 1);
out('b20', banner());
reloadPage();                                       // the micro-session goes on: the gap is well under 10 minutes
key('Space', ' ');
advance(1490 - (WALL.t - START_MS) / 1000, 1);
out('b24_50', banner());
advance(20, 1);
out('b25_10', banner());
"""


def test_a_reload_keeps_counting_toward_the_break_nudge(tmp_path):
    """Resume: the nudge comes 25 minutes into the micro-session, not 25 minutes after the last page load."""
    run = run_page(tmp_path, "triage", RELOAD)
    o = run.outs
    assert not o["b20"]["shown"] and not o["b24_50"]["shown"]
    assert o["b25_10"]["shown"] and o["b25_10"]["kind"] == "nudge"
    (nudge,) = run.sessions("nudge")
    assert nudge["ts"] == "2026-11-10T21:25:00-08:00" and nudge["micro"] == 1
    starts = run.sessions("start")
    assert len(starts) == 2 and {(e["session"], e["micro"]) for e in starts} == {("s1110a", 1)}
    assert [e["why"] for e in run.events if e["type"] == "abandon"] == ["pagehide"]
    assert not run.warnings(), run.log


RELOAD_AFTER_NUDGE = r"""
settle();
key('Space', ' ');
advance(26 * 60, 1);                                // the nudge came at 25 minutes
out('b26', banner());
reloadPage();
key('Space', ' ');
advance(2990 - (WALL.t - START_MS) / 1000, 1);      // 49:50 into the micro-session
out('b49_50', banner());
advance(20, 1);
out('b50_10', banner());
"""


def test_a_reload_does_not_repeat_a_nudge_already_shown(tmp_path):
    """The server counts the micro-session's nudges, so a reloaded page waits for the next 25 minutes."""
    run = run_page(tmp_path, "triage", RELOAD_AFTER_NUDGE)
    o = run.outs
    assert o["b26"]["shown"] and o["b26"]["kind"] == "nudge"
    assert not o["b49_50"]["shown"] and o["b50_10"]["shown"] and o["b50_10"]["kind"] == "nudge"
    assert [e["ts"] for e in run.sessions("nudge")] == ["2026-11-10T21:25:00-08:00", "2026-11-10T21:50:00-08:00"]
    assert not run.warnings(), run.log
