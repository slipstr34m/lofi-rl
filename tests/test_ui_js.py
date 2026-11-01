"""The page's JavaScript, without a browser: syntax, the audio engine against a fake AudioContext, and the
mark rules and event payloads, which are then posted to a real Ledger so the page and the server agree.

Runs in JavaScriptCore (`jsc`, part of macOS); skipped where no engine is installed (nothing is installed)."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from lofi_rl.ui.manifest import MODES, build_demo, load_batch
from lofi_rl.ui.votes import Ledger

STATIC = Path(__file__).resolve().parents[1] / "src" / "lofi_rl" / "ui" / "static"
JSC_PATHS = ["/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc",
             "/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc"]
JSC = next((p for p in JSC_PATHS if Path(p).exists()), None) or shutil.which("jsc")
needs_jsc = pytest.mark.skipif(JSC is None, reason="no JavaScriptCore shell on this machine")

FAKE_WEB_AUDIO = r"""
// A fake Web Audio API: a clock moved by hand, and params that record and evaluate their automation.
var CLOCK = { t: 0 };
function Param(v) { this.value = v; this.v0 = v; this.events = []; }
Param.prototype.setValueAtTime = function (v, t) { this.events.push({ k: 'set', v: v, t: t }); };
Param.prototype.linearRampToValueAtTime = function (v, t) { this.events.push({ k: 'lin', v: v, t: t }); };
Param.prototype.setValueCurveAtTime = function (c, t, d) {
  for (var i = 0; i < this.events.length; i++) {
    var e = this.events[i];
    if (e.k === 'curve' && t < e.t + e.d && e.t < t + d) throw new Error('NotSupportedError: curves overlap');
  }
  this.events.push({ k: 'curve', c: c, t: t, d: d });
};
Param.prototype.cancelScheduledValues = function (t) { this.events = this.events.filter(function (e) { return e.t < t; }); };
// Evaluate the automation at t as Web Audio does: linear ramps and value curves interpolate.
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
function Node() { this.out = []; }
Node.prototype.connect = function (n) { this.out.push(n); return n; };
Node.prototype.disconnect = function () { this.out = []; };
function Gain() { Node.call(this); this.gain = new Param(1); }
Gain.prototype = Object.create(Node.prototype);
function Source() { Node.call(this); this.loop = false; this.buffer = null; this.startedAt = null; this.stoppedAt = null; }
Source.prototype = Object.create(Node.prototype);
Source.prototype.start = function (t, off) { this.startedAt = t; this.offset = off; };
Source.prototype.stop = function (t) { this.stoppedAt = t; };
function FakeContext(opts) {
  this.sampleRate = (opts && opts.sampleRate) || 48000; this.state = 'suspended';
  this.destination = new Node(); this.outputLatency = 0.012; this.baseLatency = 0.005;
}
Object.defineProperty(FakeContext.prototype, 'currentTime', { get: function () { return CLOCK.t; } });
FakeContext.prototype.createGain = function () { return new Gain(); };
FakeContext.prototype.createBufferSource = function () { return new Source(); };
FakeContext.prototype.resume = function () { this.state = 'running'; return Promise.resolve(); };
FakeContext.prototype.suspend = function () { this.state = 'suspended'; return Promise.resolve(); };
var AudioContext = FakeContext;
var TIMERS = [];
function setTimeout(fn, ms) { TIMERS.push(fn); return TIMERS.length; }
function clearTimeout(id) { if (TIMERS[id - 1]) TIMERS[id - 1] = null; }
function runTimers() { var t = TIMERS; TIMERS = []; t.forEach(function (fn) { if (fn) fn(); }); }
function assert(c, msg) { if (!c) throw new Error('assert: ' + msg); }
function near(a, b, msg) { if (Math.abs(a - b) > 1e-6) throw new Error('near: ' + msg + ': got ' + a + ', want ' + b); }
"""

AUDIO_CHECKS = r"""
var A = LofiAudio;
var eng = new A.Engine({ sampleRate: 44100, rampMs: 5 });
var ctx = eng.context();
ctx.state = 'running';
var bufs = [0, 1, 2, 3].map(function () { return { duration: 8 }; });      // 8 bars of 1 s each
var p = new A.RoundPlayer(eng, bufs, { bars: 8, barFraction: 0.75 });
var t0 = p.start(0);
near(t0, 0.06, 'every clip starts at one time');
assert(p.sources.every(function (s) { return s.startedAt === t0 && s.offset === 0 && s.loop === true; }), 'one start, looping');
var end = p.planAuto([0, 1, 2, 3], 8, t0, 2);
near(end, t0 + 32, 'auto-play: 8 bars each, 1 -> 4');
function ups(t) { return p.gains.map(function (g) { return g.gain.at(t); }); }
var u = ups(t0 + 4); assert(u.join() === '1,0,0,0', 'clip 1 up in bars 1-8: ' + u);
u = ups(t0 + 12); assert(u.join() === '0,1,0,0', 'clip 2 up in its 8 bars: ' + u);
u = ups(t0 + 8 + 0.0026); assert(u.join() === '0,1,0,0', 'the 5 ms ramp is centred on the bar line: ' + u);
var mid = p.gains[1].gain.at(t0 + 8); assert(mid > 0.3 && mid < 0.7, 'mid-ramp gain ' + mid);
// the ramps are 5 ms raised cosines (BS.1534), complementary, centred on the bar line: a linear ramp is a quarter
// of the way up a quarter of the way in, a raised cosine only 0.146 of the way
var rp = p.ramps[0], rs = rp.start;
near(rp.end - rs, 0.005, 'ramp length = switch_ramp_ms'); near(rs + 0.0025, t0 + 8, 'centred on the bar line');
var upQ = p.gains[1].gain.at(rs + 0.00125), downQ = p.gains[0].gain.at(rs + 0.00125);
assert(Math.abs(upQ - (0.5 - 0.5 * Math.cos(Math.PI / 4))) < 0.002, 'raised cosine a quarter in: ' + upQ);
near(upQ + downQ, 1, 'the two gains are complementary');
var rc = A.raisedCosine(0, 1, 5);
assert([0, 0.1464, 0.5, 0.8536, 1].every(function (x, i) { return Math.abs(rc[i] - x) < 1e-4; }), 'raisedCosine: ' + Array.prototype.join.call(rc));
assert(p.upAt(t0 + 8 - 0.001) === 0 && p.upAt(t0 + 8 + 0.001) === 1, 'the up clip changes on the bar line');
CLOCK.t = t0 + 8.5; p.update();
assert(p.heardString(0) === '11111111', 'clip 1 heard all 8 bars: ' + p.heardString(0));
assert(p.heardString(1) === '00000000', 'half a bar is not heard: ' + p.heardString(1));
CLOCK.t = t0 + 8.75 + 1e-4; p.update();
assert(p.heardString(1) === '10000000', '75 % of a bar is heard: ' + p.heardString(1));
near(Math.round(p.heardMs(0)), 8000, 'heard ms');
assert(p.playOrder().join() === '0,1', 'play order');
// by hand: listen to 4 mid-bar; the rest of the auto plan is dropped
CLOCK.t = t0 + 10.3;
var ran = false;
p.change(function (t) { ran = true; near(t, CLOCK.t + A.LEAD, 'lead'); p.cancelFrom(t); p.switchTo(3, t); });
assert(ran && p.lastPlanned() === 3, 'a switch by hand runs at once');
u = ups(t0 + 20); assert(u.join() === '0,0,0,1', 'only clip 4 stays up: ' + u);
assert(p.timeline.length === 3, 'timeline 1, 2, 4');
CLOCK.t = t0 + 13.0; p.update();
assert(p.heardString(3) === '00011000', 'clip 4: bar 3 partly, bars 4-5 whole: ' + p.heardString(3));
assert(p.heardString(1) === '11000000', 'clip 2: bar 3 was only 30 %: ' + p.heardString(1));
// change() waits while a ramp runs, so a ramp is never cut
p.switchTo(0, CLOCK.t + 0.001);
var waited = false;
p.change(function () { waited = true; });
assert(!waited && TIMERS.length === 1, 'waits for the ramp');
CLOCK.t += 0.02; runTimers();
assert(waited, 'then runs');
// marking the playing clip mid-bar: the next clip gets the rest of the bar plus 8 whole bars
CLOCK.t = t0 + 14.4;
p.change(function (t) { p.cancelFrom(t); p.switchTo(2, t); near(p.planAuto([2, 3], 8, t, 0), t0 + 31, 'skip end'); });
near(p.timeline[p.timeline.length - 1].from, t0 + 23, 'the next auto switch is on a bar line');
// paused: a switch sets the gains at once (nothing is audible, nothing can click)
eng.paused = true; ctx.state = 'suspended';
p.change(function (t) { p.cancelFrom(t); p.switchTo(1, t); });
u = ups(CLOCK.t + 0.1); assert(u.join() === '0,1,0,0', 'paused switch: ' + u);
eng.paused = false; ctx.state = 'running';
// replay from bar 1: heard bars kept, new sources start together
var before = p.heardString(3), old = p.sources, t1 = p.restart(1);
assert(old.every(function (s) { return s.stoppedAt !== null; }), 'old sources stop');
assert(p.sources.every(function (s) { return s.startedAt === t1 && s.loop; }), 'new ones start at one time');
assert(p.heardString(3) === before && p.barAt(t1 + 2.5) === 2, 'heard bars kept; bar 3');
// 75 % must fall inside one bar: 60 % made of two stretches is not heard
var q = new A.RoundPlayer(eng, bufs.slice(0, 2), { bars: 8, barFraction: 0.75 });
CLOCK.t = 100; var s0 = q.start(0);
CLOCK.t = s0 + 0.2; q.change(function (t) { q.cancelFrom(t); q.switchTo(1, t); });
CLOCK.t = s0 + 0.6; q.change(function (t) { q.cancelFrom(t); q.switchTo(0, t); });
CLOCK.t = s0 + 1.0; q.update();
assert(q.heardString(0) === '00000000', 'two stretches of ~60 %: ' + q.heardString(0));
// playlist: plays its passes, fades and stops
var r = new A.RoundPlayer(eng, [{ duration: 8 }], { bars: 8 });
CLOCK.t = 200; var s1 = r.start(0); r.scheduleStop(s1 + 16);
CLOCK.t = s1 + 30; r.update();
near(r.passesDone(), 2, 'passes stop at the scheduled end');
near(Math.round(r.heardMs(0)), 16000, 'heard time stops too');
// paused stretches never count as heard, even while the clock runs on before the context suspends
var w = new A.RoundPlayer(eng, [{ duration: 8 }], { bars: 8, barFraction: 0.75 });
CLOCK.t = 300; var s2 = w.start(0);
CLOCK.t = s2 + 0.5; eng.pause();
CLOCK.t = s2 + 2.5; w.update();
near(Math.round(w.heardMs(0)), 500, 'no heard time while paused');
eng.unlock();
CLOCK.t = s2 + 5.0; w.update();
near(Math.round(w.heardMs(0)), 3000, 'heard time again after resuming');
assert(w.heardString(0) === '00011000', 'bars 4-5 heard, bars 1-3 not: ' + w.heardString(0));
assert(w.heardCount(0) === 2 && w.heardBar(0, 3) && !w.heardBar(0, 2), 'heard count and per-bar view');
// a hold (the output changed): pause, and only resume() (Space) lifts it; unlock() (auto-start, 5) does not
var mg = eng.master.gain;
CLOCK.t = s2 + 6; eng.hold('device');
assert(eng.paused && eng.held === 'device', 'held and paused');
CLOCK.t = s2 + 6.1; runTimers();
assert(ctx.state === 'suspended' && mg.at(CLOCK.t) === 0, 'silent while held');
eng.unlock();
assert(eng.paused && eng.held === 'device' && ctx.state === 'suspended' && mg.at(CLOCK.t + 0.1) === 0, 'unlock does not lift a hold');
eng.resume();
assert(!eng.paused && eng.held === null && ctx.state === 'running' && mg.at(CLOCK.t + 0.01) === 1, 'Space lifts it');
// the system stops the audio (Safari "interrupted"): muted at once and held; if the browser later runs the
// context again by itself, it is stopped again, so nothing plays (or counts as heard) until Space
var said = null; eng.onInterrupt = function (st) { said = st; };
CLOCK.t = s2 + 7; w.update(); var before7 = w.heardMs(0);
ctx.state = 'interrupted'; ctx.onstatechange();
assert(said === 'interrupted' && eng.paused && eng.held === 'system' && mg.at(CLOCK.t) === 0, 'interrupted: muted, held');
ctx.state = 'running'; ctx.onstatechange();
assert(ctx.state === 'suspended' && mg.at(CLOCK.t + 1) === 0, 'stays stopped when the browser resumes it: ' + ctx.state);
CLOCK.t = s2 + 9; w.update(); near(w.heardMs(0), before7, 'nothing heard while interrupted');
eng.resume();
assert(ctx.state === 'running' && !eng.paused && eng.held === null && mg.at(CLOCK.t + 0.01) === 1, 'Space goes on');
print('audio ok');
"""

LONG_LOOP_CHECKS = r"""
// A 40-bar loop (manifests allow 1-64 bars): heard bars past bar 32 must land on their own bars.
var A = LofiAudio, eng = new A.Engine({ rampMs: 5 }), ctx = eng.context();
ctx.state = 'running';
var p = new A.RoundPlayer(eng, [{ duration: 40 }, { duration: 40 }], { bars: 40, barFraction: 0.75 });
var t0 = p.start(1);                                  // clip 2 up, 1 s bars
p.switchTo(0, t0 + 33 - 0.0025);                      // clip 1 up for bar 34 only
p.switchTo(1, t0 + 34 - 0.0025);
CLOCK.t = t0 + 40 - 0.001; p.update();
var one = p.heardString(0), two = p.heardString(1);
assert(one === new Array(34).join('0') + '1' + new Array(7).join('0'), 'clip 1 heard bar 34 only: ' + one);
assert(two === new Array(34).join('1') + '0' + new Array(7).join('1'), 'clip 2 heard bars 1-33 and 35-40: ' + two);
assert(p.heardCount(0) === 1 && p.heardCount(1) === 39 && p.heardBar(0, 33) && !p.heardBar(0, 1), 'counts');
print('long ok');
"""

RULES_CHECKS = r"""
var R = LofiRules, out = {};
function play(mode, round, acts) {
  var m = R.newMarks();
  for (var i = 0; i < acts.length; i++) {
    var res = R.apply(mode, m, acts[i], round.clips.length, 1000 * (i + 1));
    if (res.error) throw new Error(mode + ' ' + JSON.stringify(acts[i]) + ': ' + res.error);
    m = res.marks;
  }
  return m;
}
function info(round) {
  var k = round.clips.length, clips = [], order = [];
  for (var i = 0; i < k; i++) { clips.push({ heard_bars: '11110000', heard_ms: 12000.4 }); order.push(i); }
  return { eventId: R.newId(), clips: clips, playOrder: order, lastHeard: k - 1, switches: 1, autoplayDone: true,
           tCommit: 9000.7, device: 'airpods', audio: { sr: 8000, latency_ms: 12 }, plays: 2 };
}
function chk(c, msg) { if (!c) throw new Error('rule: ' + msg); }
var plans = { bws4: [['best', 0], ['worst', 3], ['tag', 'sour']], shootout: [['noBest'], ['worst', 1]],
              keep3: [['best', 2]], triage: [['tier', 'toss'], ['tier', 'good']], ab: [['best', 1]],
              playlist: [['best', 0]] };
Object.keys(plans).forEach(function (mode) {
  var round = MANIFESTS[mode].rounds[0], m = play(mode, round, plans[mode]);
  chk(R.problem(mode, m) === null, mode + ' is complete: ' + R.problem(mode, m));
  out[mode] = R.votePayload(mode, round, m, info(round));
});
var r0 = MANIFESTS.bws4.rounds[0], r1 = MANIFESTS.bws4.rounds[1], r2 = MANIFESTS.bws4.rounds[2];
out.session = R.sessionPayload('round_start', r1.id, 'airpods', { sr: 8000 });
out.abandon = R.abandonPayload(r1, info(r1), 'esc');
out.abandon_bare = R.abandonPayload(r2, null, 'error');
// t_ms holds times only for marks that are in the vote (spec-5): Q, then T clears the best; L on, then off
var r3 = MANIFESTS.bws4.rounds[3];
out.tms_cleared = R.votePayload('bws4', r3, play('bws4', r3, [['best', 0], ['noBest'], ['worst', 3], ['tag', 'sour'],
                                                               ['tag', 'sour']]), info(r3));
chk(JSON.stringify(out.tms_cleared.t_ms) === '{"no_best":2000,"worst":3000,"commit":9001}', 'cleared marks leave no time: ' + JSON.stringify(out.tms_cleared.t_ms));
var tg = R.votePayload('bws4', r0, play('bws4', r0, [['best', 0], ['allBad']]), info(r0)).t_ms;
chk(Object.keys(tg).sort().join() === 'all_bad,commit', 'G after a best: ' + JSON.stringify(tg));
var pl = MANIFESTS.playlist.rounds[0];
tg = R.votePayload('playlist', pl, play('playlist', pl, [['best', 0], ['best', 0]]), info(pl)).t_ms;
chk(Object.keys(tg).join() === 'commit', 'a heart toggled off: ' + JSON.stringify(tg));
var m = play('bws4', r0, [['best', 0]]);
chk(R.problem('bws4', m).indexOf('Pick a worst') === 0, 'a best alone needs a worst');
m = play('bws4', r0, [['best', 0], ['worst', 0]]); chk(m.best === null && m.worst === 0, 'worst on the best clip moves it');
m = play('bws4', r0, [['best', 0], ['noBest']]); chk(m.best === null && m.noBest, 'T clears the best');
m = play('bws4', r0, [['noBest'], ['noBest']]); chk(!m.noBest, 'T toggles');
m = play('bws4', r0, [['best', 1], ['allBad']]); chk(m.allBad && m.best === null && R.problem('bws4', m) === null, 'G alone saves');
m = play('bws4', r0, [['best', 1], ['worst', 2], ['tag', 'nodding'], ['tag', 'nodding']]); chk(m.tag === null, 'a tag toggles');
m = play('bws4', r0, [['best', 1], ['noWorst'], ['tag', 'boring']]); chk(R.problem('bws4', m).indexOf('worst pick') > 0, 'negative tag needs a worst');
// a 2-clip shootout round (Day 2's sax final): its hints name only Q W and Z X, and best + worst saves
var s2 = { id: 'r2clips0', scene: 'city at night', clips: [{ id: 'kaaaaaaaaaa', file: 'kaaaaaaaaaa.flac' },
                                                           { id: 'kbbbbbbbbbb', file: 'kbbbbbbbbbb.flac' }] };
m = play('shootout', s2, [['best', 0]]);
chk(R.problem('shootout', m, 2) === 'Pick a worst (Z X), or press B for "no clear worst".', 'k=2 worst hint: ' + R.problem('shootout', m, 2));
chk(R.problem('shootout', R.newMarks(), 2).indexOf('Pick a best (Q W),') === 0, 'k=2 best hint');
chk(R.problem('shootout', m).indexOf('(Z X C V)') > 0, 'no k: the 4 keys, as before');
chk(R.apply('shootout', m, ['worst', 2], 2, 0).error, 'no clip 3 in a 2-clip round');
m = play('shootout', s2, [['best', 0], ['worst', 1]]); chk(R.problem('shootout', m, 2) === null, 'k=2 best + worst saves');
var p2 = R.votePayload('shootout', s2, m, info(s2));
chk(p2.best === 'kaaaaaaaaaa' && p2.worst === 'kbbbbbbbbbb' && p2.clips.length === 2, 'k=2 payload: ' + JSON.stringify(p2));
chk(R.apply('bws4', R.newMarks(), ['tier', 'fine'], 4, 0).error, 'no tiers in bws4');
chk(R.apply('ab', R.newMarks(), ['worst', 0], 2, 0).error, 'no worst in ab');
chk(R.apply('ab', R.newMarks(), ['best', 2], 2, 0).error, 'ab has two clips');
chk(R.apply('ab', R.newMarks(), ['allBad'], 2, 0).error, 'ab is forced choice');
chk(R.apply('shootout', R.newMarks(), ['tag', 'sour'], 4, 0).error, 'no tags in shootouts');
chk(R.apply('keep3', R.newMarks(), ['noBest'], 3, 0).error, 'no hatches in keep3');
m = play('keep3', MANIFESTS.keep3.rounds[0], [['best', 1], ['allBad']]); chk(m.allBad && m.best === null, 'G drops the kept draft');
chk(R.apply('triage', R.newMarks(), ['best', 0], 1, 0).error.indexOf('tier') > 0, 'triage asks for a tier');
m = play('playlist', MANIFESTS.playlist.rounds[0], [['best', 0], ['worst', 0]]); chk(m.mark === 'cross', 'an x replaces a heart');
m = play('playlist', MANIFESTS.playlist.rounds[0], [['best', 0], ['best', 0]]); chk(m.mark === null, 'a heart toggles');
var K = R.KEYMAP, want = { Digit1: 'listen', Digit4: 'listen', Digit5: 'replay', KeyQ: 'best', KeyR: 'best',
  KeyT: 'noBest', KeyA: 'tier', KeyF: 'tier', KeyG: 'allBad', KeyZ: 'worst', KeyV: 'worst', KeyB: 'noWorst',
  KeyJ: 'tag', Semicolon: 'tag', KeyU: 'tag', KeyO: 'tag', Space: 'play', Enter: 'commit', Backspace: 'undo', Escape: 'end' };
Object.keys(want).forEach(function (c) { chk(K[c] && K[c][0] === want[c], 'key ' + c); });
chk(K.KeyQ[1] === 0 && K.KeyW[1] === 1 && K.KeyE[1] === 2 && K.KeyR[1] === 3, 'Q W E R are best 1-4');
chk(K.KeyZ[1] === 0 && K.KeyX[1] === 1 && K.KeyC[1] === 2 && K.KeyV[1] === 3, 'Z X C V are worst 1-4');
chk(K.KeyJ[1] === 'fake' && K.KeyK[1] === 'no_groove' && K.KeyL[1] === 'sour' && K.Semicolon[1] === 'boring', 'negative tags');
chk(K.KeyU[1] === 'nodding' && K.KeyI[1] === 'sax_line' && K.KeyO[1] === 'chords', 'positive tags');
chk(K.KeyA[1] === 'toss' && K.KeyS[1] === 'fine' && K.KeyD[1] === 'good' && K.KeyF[1] === 'great', 'tiers');
chk(/^e[0-9a-f]{24}$/.test(R.newId()), 'event ids');
print(JSON.stringify(out));
"""


def _run_jsc(tmp_path, *parts):
    files = []
    for i, part in enumerate(parts):
        if isinstance(part, Path):
            files.append(str(part))
        else:
            f = tmp_path / f"part{i}.js"
            f.write_text(part)
            files.append(str(f))
    out = subprocess.run([JSC, *files], capture_output=True, text=True, timeout=20)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout.strip()


@needs_jsc
@pytest.mark.parametrize("name", ["audio.js", "app.js"])
def test_javascript_parses(name, tmp_path):
    src = json.dumps(str(STATIC / name))
    assert _run_jsc(tmp_path, f"checkSyntax({src}); print('ok');") == "ok"


@needs_jsc
def test_audio_engine_with_a_fake_audio_context(tmp_path):
    assert _run_jsc(tmp_path, FAKE_WEB_AUDIO, STATIC / "audio.js", AUDIO_CHECKS) == "audio ok"


@needs_jsc
def test_heard_bars_past_bar_32_land_on_their_own_bars(tmp_path):
    """JavaScript's 32-bit shifts alias bar 33 onto bar 1, so a bitmask cannot hold the 64 bars a manifest may
    have (integrity-6, spec-11)."""
    assert _run_jsc(tmp_path, FAKE_WEB_AUDIO, STATIC / "audio.js", LONG_LOOP_CHECKS) == "long ok"


@needs_jsc
def test_page_payloads_are_accepted_by_the_server(tmp_path):
    root = tmp_path / "inbox" / "2026-11-03"
    batches = {m: load_batch(build_demo(root / f"demo-{m}", m, bpm=240.0, sr=8000)) for m in MODES}
    manifests = {m: b[0].public() for m, b in batches.items()}
    out = json.loads(_run_jsc(tmp_path, FAKE_WEB_AUDIO, STATIC / "audio.js", STATIC / "app.js",
                              "var MANIFESTS = " + json.dumps(manifests) + ";", RULES_CHECKS))
    for mode, (m, h) in batches.items():
        votes = tmp_path / f"votes-{mode}"
        votes.mkdir()
        led = Ledger(votes, m, h, ui="jsc")
        try:
            reply = led.submit(out[mode])                      # exactly what the page would POST
            assert reply["ok"] and reply["state"]["settled"] == 1, mode
            rec = json.loads(led.path.read_text().splitlines()[-1])
            r0, hr = m.rounds[0], h.rounds[m.rounds[0].id]
            assert [c["heard_bars"] for c in rec["clips"]] == ["11110000"] * len(r0.clips)
            assert rec["clips"][0]["heard_ms"] == 12000 and rec["t_ms"]["commit"] == 9001
            assert rec["play_order"] == list(range(1, len(r0.clips) + 1))
            assert rec["last_heard"] == hr.real(r0.ids[-1]) and rec["device"] == "airpods"
            if mode == "bws4":
                assert rec["best"] == hr.real(r0.ids[0]) and rec["tag"] == {"name": "sour", "clip": hr.real(r0.ids[3])}
                assert set(rec["t_ms"]) == {"best", "worst", "tag", "commit"}
                for ev in (out["session"], out["abandon"], out["abandon_bare"]):
                    assert led.submit(ev)["ok"]
                tail = [json.loads(x) for x in led.path.read_text().splitlines()[-3:]]
                assert [(e["type"], e.get("what") or e.get("why")) for e in tail] == [
                    ("session", "round_start"), ("abandon", "esc"), ("abandon", "error")]
                assert tail[1]["clips"][0]["clip"] == h.rounds[m.rounds[1].id].real(m.rounds[1].ids[0])
                assert led.state()["current"] == m.rounds[3].id     # both abandoned rounds went to the end
                assert led.submit(out["tms_cleared"])["ok"]
                rec3 = json.loads(led.path.read_text().splitlines()[-1])
                assert rec3["t_ms"] == {"no_best": 2000, "worst": 3000, "commit": 9001} and rec3["tag"] is None
            elif mode == "shootout":
                assert rec["no_best"] and rec["best"] is None and rec["worst"] == hr.real(r0.ids[1])
            elif mode == "keep3":
                assert rec["best"] == hr.real(r0.ids[2]) and not rec["all_bad"]
            elif mode == "triage":
                assert rec["tier"] == "good"
            elif mode == "ab":
                assert rec["pick"] == hr.real(r0.ids[1]) and rec["t_ms"]["pick"] == 1000
            elif mode == "playlist":
                assert rec["mark"] == "heart" and rec["plays"] == 2
        finally:
            led.close()


def test_page_wiring():
    """Every element app.js looks up exists in index.html, scripts load from this server only, audio.js first."""
    html = (STATIC / "index.html").read_text()
    app = (STATIC / "app.js").read_text()
    ids = set(re.findall(r"\$\('([a-z-]+)'\)", app))
    assert ids and not {i for i in ids if f'id="{i}"' not in html}
    scripts = re.findall(r'<script src="([^"]+)"', html)
    assert scripts == ["/static/audio.js", "/static/app.js"]
    for text in (html, app, (STATIC / "audio.js").read_text()):
        assert not re.search(r"https?://", text.replace("http://' + h", "")), "no outside URLs"
