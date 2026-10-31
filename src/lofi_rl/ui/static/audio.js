/* audio.js: Web Audio playback for the rating page (docs/RATING.md; docs/design/taste_pipeline.md §1).

   One AudioContext for the page. A round's clips are all decoded first, then every clip starts at one
   context time with loop = true, so they stay on the same bar. Switching only moves gains, with short
   raised-cosine ramps (5 ms by default, no longer crossfade, BS.1534), so the bar position is kept.

   Heard bars: a bar counts for a clip when its gain was up for at least `barFraction` of that bar. The page
   calls a clip heard once `min_bars` distinct bars count (both in configs/rating.yaml). They are kept as one
   flag per bar, not a bitmask: JavaScript shifts wrap at 32 bits, and a manifest may have up to 64 bars.

   Pausing: pause() is the listener's Space. hold(why) also pauses, and then nothing but resume() (Space again)
   plays: not the next round's auto-start, not a replay. The page holds when the audio output changes; the engine
   holds by itself when the system stops the audio (Safari's "interrupted"), and keeps it stopped if the browser
   later runs the context again on its own, so what plays is always what the page counts as heard.

   No DOM in this file: it also runs in a bare JavaScript engine for tests (tests/test_ui_js.py). */
(function (global) {
  'use strict';

  // Seconds between deciding a change and its ramps, so a ramp is never cancelled while it runs.
  var LEAD = 0.03;

  function raisedCosine(from, to, n) {
    var out = new Float32Array(n);
    for (var i = 0; i < n; i++) {
      var w = 0.5 - 0.5 * Math.cos(Math.PI * i / (n - 1));
      out[i] = from + (to - from) * w;
    }
    return out;
  }

  function quiet(p) {
    if (p && typeof p.then === 'function') return p.then(function () {}, function () {});
    return Promise.resolve();
  }

  // decodeAudioData has a promise form and an older callback form (Safari); accept either.
  function decode(ctx, data) {
    return new Promise(function (resolve, reject) {
      var done = false;
      function ok(b) { if (!done) { done = true; resolve(b); } }
      function bad(e) { if (!done) { done = true; reject(e || new Error('the browser could not decode this clip')); } }
      try {
        var p = ctx.decodeAudioData(data, ok, bad);
        if (p && typeof p.then === 'function') p.then(ok, bad);
      } catch (e) { bad(e); }
    });
  }

  // ---------------------------------------------------------------------------------------- Engine

  function Engine(opts) {
    opts = opts || {};
    this.sampleRate = opts.sampleRate || 0;
    this.rampSec = (opts.rampMs || 5) / 1000;
    this.ctx = null;
    this.master = null;
    this.cache = new Map();
    this.paused = false;          // what the listener asked for
    this.held = null;             // why playback waits for Space: 'device' (output changed) or 'system'
    this.silences = [];           // [start, end] context times while paused (end = Infinity until resumed)
    this.onInterrupt = null;      // called when the system stops the audio (e.g. Safari's "interrupted")
    this._suspendTimer = null;
    this._fetch = opts.fetch || (typeof global.fetch === 'function' ? global.fetch.bind(global) : null);
  }

  Engine.prototype.context = function () {
    if (this.ctx) return this.ctx;
    var AC = global.AudioContext || global.webkitAudioContext;
    if (!AC) throw new Error('This browser has no Web Audio, so it cannot play the clips.');
    var ctx = null;
    if (this.sampleRate) {
      try { ctx = new AC({ sampleRate: this.sampleRate }); } catch (e) { ctx = null; }  // keeps the loop sample-exact
    }
    if (!ctx) ctx = new AC();
    this.ctx = ctx;
    this.master = ctx.createGain();
    this.master.gain.value = 1;
    this.master.connect(ctx.destination);
    var self = this;
    ctx.onstatechange = function () {
      if (!self._started) return;
      if (ctx.state === 'running') {
        // The browser ran the context again by itself (an interruption ended). The page still says paused,
        // so stop it again: nothing may play that the page does not count.
        if (self.paused && typeof ctx.suspend === 'function') quiet(ctx.suspend());
        return;
      }
      if (!self.paused) {                       // the system stopped the audio: mute at once, wait for Space
        self.held = self.held || 'system';
        self.paused = true;
        self._silenceFrom(ctx.currentTime);
        self._muteNow();
        if (typeof self.onInterrupt === 'function') self.onInterrupt(ctx.state);
      }
    };
    return ctx;
  };

  // Call from a key press or a click: browsers start audio only after a gesture. While playback is held it does
  // nothing (only resume() goes on), so an auto-start or a replay never plays past a hold.
  Engine.prototype.unlock = function () {
    var ctx = this.context();
    if (this.held) return Promise.resolve();
    var last = this.silences[this.silences.length - 1];
    if (last && last[1] === Infinity) last[1] = ctx.currentTime;
    this.paused = false;
    this._started = true;
    this._cancelSuspend();
    this._rampMaster(1);
    if (ctx.state !== 'running' && typeof ctx.resume === 'function') return quiet(ctx.resume());
    return Promise.resolve();
  };

  // Space: lifts a hold, then plays.
  Engine.prototype.resume = function () {
    this.held = null;
    return this.unlock();
  };

  // Pause until Space (the audio output changed): nothing else plays meanwhile.
  Engine.prototype.hold = function (why) {
    this.held = why || 'hold';
    this.pause();
  };

  Engine.prototype.pause = function () {
    if (!this.ctx || this.paused) return;
    this.paused = true;
    this._silenceFrom(this.ctx.currentTime);
    this._rampMaster(0);
    var ctx = this.ctx, self = this;
    this._cancelSuspend();
    this._suspendTimer = setTimeout(function () {
      self._suspendTimer = null;
      if (self.paused && typeof ctx.suspend === 'function') quiet(ctx.suspend());
    }, Math.ceil(this.rampSec * 1000) + 20);
  };

  // The clock keeps running until the context is suspended (and longer in a throttled background tab);
  // none of that counts as heard.
  Engine.prototype._silenceFrom = function (t) {
    this.silences.push([t, Infinity]);
    if (this.silences.length > 64) this.silences.shift();
  };

  // [a, b] minus the paused stretches, as a list of [x, y].
  Engine.prototype.audible = function (a, b) {
    var out = [[a, b]];
    for (var i = 0; i < this.silences.length; i++) {
      var s = this.silences[i], next = [];
      for (var j = 0; j < out.length; j++) {
        var x = out[j][0], y = out[j][1];
        if (s[1] <= x || s[0] >= y) { next.push([x, y]); continue; }
        if (s[0] > x) next.push([x, s[0]]);
        if (s[1] < y) next.push([s[1], y]);
      }
      out = next;
    }
    return out;
  };

  Engine.prototype._cancelSuspend = function () {
    if (this._suspendTimer !== null) { clearTimeout(this._suspendTimer); this._suspendTimer = null; }
  };

  Engine.prototype._rampMaster = function (to) {
    if (!this.master) return;
    var g = this.master.gain, t = this.ctx.currentTime;
    try {
      g.cancelScheduledValues(t);
      g.setValueAtTime(g.value, t);
      g.linearRampToValueAtTime(to, t + this.rampSec);
    } catch (e) { g.value = to; }
  };

  // Silence now, with no ramp: the context is not rendering (it was stopped), so nothing can click, and it
  // comes back silent if the browser runs it again.
  Engine.prototype._muteNow = function () {
    if (!this.master) return;
    var g = this.master.gain;
    try {
      g.cancelScheduledValues(0);
      g.setValueAtTime(0, this.ctx.currentTime);
    } catch (e) { g.value = 0; }
  };

  // Fetch and decode one clip (cached by URL, so a round can be prefetched while another plays).
  Engine.prototype.load = function (url) {
    if (!this.cache.has(url)) {
      var self = this;
      if (!this._fetch) return Promise.reject(new Error('no fetch() in this browser'));
      var p = this._fetch(url, { cache: 'no-store' }).then(function (r) {
        if (!r.ok) throw new Error('could not fetch ' + url + ' (HTTP ' + r.status + ')');
        return r.arrayBuffer();
      }).then(function (data) { return decode(self.context(), data); });
      p.catch(function () { self.cache.delete(url); });
      this.cache.set(url, p);
    }
    return this.cache.get(url);
  };

  // Forget decoded clips that are not listed (rounds that are done), to keep memory small.
  Engine.prototype.keepOnly = function (urls) {
    var keep = new Set(urls), self = this;
    Array.from(this.cache.keys()).forEach(function (u) { if (!keep.has(u)) self.cache.delete(u); });
  };

  Engine.prototype.now = function () { return this.ctx ? this.ctx.currentTime : 0; };

  Engine.prototype.running = function () { return !!this.ctx && this.ctx.state === 'running' && !this.paused; };

  // Seconds between the context clock and the ear (large on Bluetooth); used only to draw the bar ruler.
  Engine.prototype.latency = function () {
    var c = this.ctx;
    if (!c) return 0;
    var v = (c.outputLatency || 0) + (c.baseLatency || 0);
    return isFinite(v) && v > 0 && v < 1 ? v : 0;
  };

  Engine.prototype.info = function () {
    var c = this.ctx;
    if (!c) return null;
    var out = { sr: c.sampleRate };
    if (isFinite(c.outputLatency) && c.outputLatency >= 0) out.latency_ms = Math.round(c.outputLatency * 1000);
    if (isFinite(c.baseLatency) && c.baseLatency >= 0) out.base_latency_ms = Math.round(c.baseLatency * 1000);
    return out;
  };

  // ---------------------------------------------------------------------------------------- RoundPlayer

  // The clips of one round, looping together. Clip indices are 0-based (the page shows 1..k).
  function RoundPlayer(engine, buffers, opts) {
    if (!buffers || !buffers.length) throw new Error('a round needs at least one clip');
    opts = opts || {};
    this.e = engine;
    this.ctx = engine.context();
    this.buffers = buffers;
    this.k = buffers.length;
    this.bars = opts.bars || 8;
    this.barFraction = opts.barFraction || 0.75;
    this.rampSec = engine.rampSec;
    this.loopDur = buffers[0].duration;
    this.barDur = this.loopDur / this.bars;
    var d0 = this.loopDur;
    this.mismatch = buffers.some(function (b) { return Math.abs(b.duration - d0) > 0.002; });
    this.heard = [];              // per clip: one flag per bar of the loop, 1 once that bar was heard (bar 1 first)
    this.heardMsArr = [];         // per clip: milliseconds its gain was up while playing
    this.order = [];              // clips in the order they first came up
    for (var i = 0; i < this.k; i++) { this.heard.push(new Uint8Array(this.bars)); this.heardMsArr.push(0); }
    this.started = false;
    this.stopped = false;
    this._reset();
  }

  RoundPlayer.prototype._reset = function () {
    this.sources = [];
    this.gains = [];
    this.out = null;
    this.t0 = 0;
    this.timeline = [];           // [{clip, from, to}]: when each clip's gain is up (context seconds; to null = open)
    this.ramps = [];              // [{start, end}] gain ramps scheduled
    this.coverage = [];           // per clip: Map(bar instance since t0 -> seconds up)
    for (var i = 0; i < this.k; i++) this.coverage.push(new Map());
    this.accrued = 0;             // heard time is counted up to this context time
    this.stopAt = Infinity;
  };

  RoundPlayer.prototype._build = function (clip, t0) {
    var ctx = this.ctx;
    this._reset();
    this.out = ctx.createGain();
    this.out.gain.value = 1;
    this.out.connect(this.e.master);
    for (var i = 0; i < this.k; i++) {
      var src = ctx.createBufferSource();
      src.buffer = this.buffers[i];
      src.loop = true;
      var g = ctx.createGain();
      var v = i === clip ? 1 : 0;
      g.gain.value = v;
      g.gain.setValueAtTime(v, ctx.currentTime);
      src.connect(g);
      g.connect(this.out);
      src.start(t0, 0);
      this.sources.push(src);
      this.gains.push(g);
    }
    this.t0 = t0;
    this.timeline = [{ clip: clip, from: t0, to: null }];
    this.accrued = t0;
  };

  // Start every clip at one time, with `clip` up. Returns that start time (bar 1 of the loop).
  RoundPlayer.prototype.start = function (clip, when) {
    if (this.started) throw new Error('this round is already playing');
    this._build(clip || 0, Math.max(when || 0, this.ctx.currentTime + 0.06));
    this.started = true;
    return this.t0;
  };

  RoundPlayer.prototype.now = function () { return this.ctx.currentTime; };

  // The clip whose gain is up at context time t (default now), counting planned switches.
  RoundPlayer.prototype.upAt = function (t) {
    if (t === undefined) t = this.ctx.currentTime;
    var c = this.timeline.length ? this.timeline[0].clip : 0;
    for (var i = 0; i < this.timeline.length; i++) {
      if (this.timeline[i].from <= t) c = this.timeline[i].clip; else break;
    }
    return c;
  };

  // The clip that is up once every planned switch has happened.
  RoundPlayer.prototype.lastPlanned = function () {
    return this.timeline.length ? this.timeline[this.timeline.length - 1].clip : 0;
  };

  // 0-based bar of the loop at context time t.
  RoundPlayer.prototype.barAt = function (t) {
    if (t === undefined) t = this.ctx.currentTime;
    if (!this.started || t < this.t0) return 0;
    var n = Math.floor((t - this.t0) / this.barDur + 1e-9);
    return ((n % this.bars) + this.bars) % this.bars;
  };

  // The first bar line at or after t.
  RoundPlayer.prototype.nextBarTime = function (t) {
    var n = Math.ceil((t - this.t0) / this.barDur - 1e-6);
    return this.t0 + Math.max(0, n) * this.barDur;
  };

  // Whole passes of the loop played since the (re)start.
  RoundPlayer.prototype.passesDone = function (t) {
    if (t === undefined) t = Math.min(this.ctx.currentTime, this.stopAt);
    if (!this.started || t < this.t0) return 0;
    return Math.floor((t - this.t0) / this.loopDur + 1e-9);
  };

  RoundPlayer.prototype._rampFreeAt = function () {
    var e = 0;
    for (var i = 0; i < this.ramps.length; i++) if (this.ramps[i].end > e) e = this.ramps[i].end;
    return e;
  };

  RoundPlayer.prototype._ramp = function (param, from, to, t) {
    try {
      param.setValueCurveAtTime(raisedCosine(from, to, 64), t, this.rampSec);
    } catch (e) {
      try {
        param.setValueAtTime(from, t);
        param.linearRampToValueAtTime(to, t + this.rampSec);
      } catch (e2) { param.value = to; }
    }
  };

  // Schedule a switch at `at` (never overlapping an earlier ramp): the clip that is up ramps down while
  // `clip` ramps up. Returns the time the ramps start, or null when nothing changes.
  RoundPlayer.prototype.switchTo = function (clip, at) {
    if (!this.started || this.stopped || clip < 0 || clip >= this.k) return null;
    var from = this.lastPlanned();
    if (from === clip) return null;
    var t = Math.max(at || 0, this.ctx.currentTime + 0.005, this._rampFreeAt() + 0.0005);
    this._ramp(this.gains[from].gain, 1, 0, t);
    this._ramp(this.gains[clip].gain, 0, 1, t);
    this.ramps.push({ start: t, end: t + this.rampSec });
    var mid = t + this.rampSec / 2;           // the moment the up clip changes, for heard bars
    this.timeline[this.timeline.length - 1].to = mid;
    this.timeline.push({ clip: clip, from: mid, to: null });
    return t;
  };

  // Drop every planned ramp that starts at or after t (auto-play switches not begun yet).
  RoundPlayer.prototype.cancelFrom = function (t) {
    for (var i = 0; i < this.gains.length; i++) {
      try { this.gains[i].gain.cancelScheduledValues(t); } catch (e) { /* ignore */ }
    }
    this.ramps = this.ramps.filter(function (r) { return r.start < t; });
    var half = this.rampSec / 2;
    while (this.timeline.length > 1 && this.timeline[this.timeline.length - 1].from >= t + half - 1e-9) this.timeline.pop();
    this.timeline[this.timeline.length - 1].to = null;
  };

  // While the audio is stopped (paused, or the system suspended it) nothing can be mid-ramp audibly:
  // set every gain to what the timeline says at the current time and forget the plan.
  RoundPlayer.prototype._hardReset = function () {
    var t = this.ctx.currentTime, up = this.upAt(t);
    for (var i = 0; i < this.gains.length; i++) {
      var g = this.gains[i].gain, v = i === up ? 1 : 0;
      try { g.cancelScheduledValues(0); g.setValueAtTime(v, t); } catch (e) { g.value = v; }
    }
    this.ramps = [];
    while (this.timeline.length > 1 && this.timeline[this.timeline.length - 1].from > t) this.timeline.pop();
    this.timeline[this.timeline.length - 1].to = null;
    if (this.timeline[this.timeline.length - 1].clip !== up) this.timeline.push({ clip: up, from: t, to: null });
  };

  // Run fn(t) at a safe point: t is when its ramps may start. While audio runs, wait until no ramp is
  // running or about to start (cancelling one mid-way would click); while stopped, reset the gains first.
  RoundPlayer.prototype.change = function (fn) {
    if (!this.started || this.stopped) return;
    var self = this;
    if (this.e.paused || this.ctx.state !== 'running') {
      this._hardReset();
      fn(this.ctx.currentTime + 0.001);
      return;
    }
    var now = this.ctx.currentTime, wait = 0;
    for (var i = 0; i < this.ramps.length; i++) {
      var r = this.ramps[i];
      if (r.start <= now + LEAD && r.end >= now - 0.002) wait = Math.max(wait, r.end - now);
    }
    if (wait > 0) {
      setTimeout(function () { self.change(fn); }, Math.ceil(wait * 1000) + 3);
      return;
    }
    fn(now + LEAD);
  };

  // Auto-play: seq[0] is up at `from`; each clip gets `barsEach` whole bars, then the next comes up on a
  // bar line. A single clip plays `extraBars` more (the seam). Returns when the pass ends.
  RoundPlayer.prototype.planAuto = function (seq, barsEach, from, extraBars) {
    var at = this.nextBarTime(from) + barsEach * this.barDur;
    if (seq.length === 1) at += (extraBars || 0) * this.barDur;
    for (var j = 1; j < seq.length; j++) {
      this.switchTo(seq[j], at - this.rampSec / 2);
      at += barsEach * this.barDur;
    }
    return at;
  };

  // Count heard time up to now (or tNow): each bar instance collects the seconds its clip was up.
  RoundPlayer.prototype.update = function (tNow) {
    if (!this.started) return;
    var now = tNow === undefined ? this.ctx.currentTime : tNow;
    if (now > this.stopAt) now = this.stopAt;
    if (now <= this.accrued) return;
    for (var i = 0; i < this.timeline.length; i++) {
      var s = this.timeline[i];
      if (s.from > now) break;
      if (this.order.indexOf(s.clip) < 0) this.order.push(s.clip);
      var a = Math.max(s.from, this.accrued), b = Math.min(s.to === null ? Infinity : s.to, now);
      if (b > a) {
        var parts = this.e.audible(a, b);
        for (var j = 0; j < parts.length; j++) if (parts[j][1] > parts[j][0]) this._cover(s.clip, parts[j][0], parts[j][1]);
      }
    }
    this.accrued = now;
  };

  RoundPlayer.prototype._cover = function (clip, a, b) {
    this.heardMsArr[clip] += (b - a) * 1000;
    var bd = this.barDur, m = this.coverage[clip];
    var n = Math.floor((a - this.t0) / bd + 1e-9);
    for (;;) {
      var s = this.t0 + n * bd, e = s + bd;
      var ov = Math.min(b, e) - Math.max(a, s);
      if (ov > 0) {
        var v = (m.get(n) || 0) + ov;
        m.set(n, v);
        if (v >= this.barFraction * bd - 1e-6) this.heard[clip][((n % this.bars) + this.bars) % this.bars] = 1;
      }
      if (e >= b) break;
      n += 1;
    }
  };

  // Bar b (0-based) of clip i was heard.
  RoundPlayer.prototype.heardBar = function (i, b) { return this.heard[i][b] === 1; };
  RoundPlayer.prototype.heardString = function (i) { return Array.prototype.join.call(this.heard[i], ''); };
  RoundPlayer.prototype.heardCount = function (i) {
    var n = 0, h = this.heard[i];
    for (var b = 0; b < h.length; b++) n += h[b];
    return n;
  };
  RoundPlayer.prototype.heardMs = function (i) { return this.heardMsArr[i]; };
  RoundPlayer.prototype.playOrder = function () { return this.order.slice(); };

  // Replay from bar 1 with `clip` up: the old sources fade out, new ones start. Heard bars are kept.
  RoundPlayer.prototype.restart = function (clip) {
    if (!this.started) return this.start(clip);
    this.update();
    var ctx = this.ctx, r = this.rampSec, t = ctx.currentTime + 0.02;
    var oldOut = this.out, oldSources = this.sources;
    try {
      oldOut.gain.cancelScheduledValues(0);
      oldOut.gain.setValueAtTime(1, ctx.currentTime);
      oldOut.gain.linearRampToValueAtTime(0, t + r);
    } catch (e) { oldOut.gain.value = 0; }
    oldSources.forEach(function (s) { try { s.stop(t + r + 0.01); } catch (e) { /* already stopped */ } });
    setTimeout(function () { try { oldOut.disconnect(); } catch (e) { /* gone */ } }, 2000);
    this.stopped = false;
    this._build(clip, t + r + 0.01);
    return this.t0;
  };

  // Fade out just before t and stop there (playlist: after its passes).
  RoundPlayer.prototype.scheduleStop = function (t) {
    var g = this.out.gain, r = this.rampSec;
    try {
      g.setValueAtTime(1, Math.max(this.ctx.currentTime, t - r));
      g.linearRampToValueAtTime(0, t);
    } catch (e) { /* the stop below still ends it */ }
    this.sources.forEach(function (s) { try { s.stop(t + 0.01); } catch (e) { /* ignore */ } });
    this.stopAt = t;
  };

  // Stop now with a short fade.
  RoundPlayer.prototype.stop = function () {
    if (!this.started || this.stopped) { this.stopped = true; return; }
    this.update();
    this.stopped = true;
    var ctx = this.ctx, t = ctx.currentTime, out = this.out, r = this.rampSec;
    try {
      out.gain.cancelScheduledValues(0);
      out.gain.setValueAtTime(out.gain.value, t);
      out.gain.linearRampToValueAtTime(0, t + r);
    } catch (e) { out.gain.value = 0; }
    this.sources.forEach(function (s) { try { s.stop(t + r + 0.01); } catch (e) { /* ignore */ } });
    setTimeout(function () { try { out.disconnect(); } catch (e) { /* gone */ } }, 1000);
    if (t < this.stopAt) this.stopAt = t;
  };

  global.LofiAudio = {
    Engine: Engine,
    RoundPlayer: RoundPlayer,
    raisedCosine: raisedCosine,
    LEAD: LEAD
  };
})(typeof window !== 'undefined' ? window : this);
