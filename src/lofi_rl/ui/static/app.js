/* app.js: the rating page (docs/RATING.md; docs/design/taste_pipeline.md §1). Vanilla JS, no build step.
   Audio lives in audio.js (window.LofiAudio).

   `Rules` holds the marks, the commit checks and the event payloads, with no DOM, so tests run it in a bare
   JavaScript engine and post what it builds to a real server (tests/test_ui_js.py).

   The page shows the scene text and nothing else about the brief: never a score, never model output. */
(function (global) {
  'use strict';

  // ------------------------------------------------------------------ the key map (taste_pipeline.md §1)
  //   1 2 3 4 | 5  listen k (bar-synced)   | 5 replay from bar 1
  //   Q W E R | T  best = k (ab: Q/W)      | T no clear best
  //   A S D F | G  Toss/Fine/Good/Great    | G all four bad / none acceptable
  //   Z X C V | B  worst = k               | B no clear worst
  //   J fake  K no groove  L sour  ; boring  |  U nodding  I sax line  O chords
  //   (negative tag -> worst, positive -> best) . Space play . Enter commit . Bksp undo . Esc end
  // Keys are matched by position (event.code), so the map holds on any keyboard layout.
  var KEYMAP = {
    Digit1: ['listen', 0], Digit2: ['listen', 1], Digit3: ['listen', 2], Digit4: ['listen', 3], Digit5: ['replay'],
    Numpad1: ['listen', 0], Numpad2: ['listen', 1], Numpad3: ['listen', 2], Numpad4: ['listen', 3], Numpad5: ['replay'],
    KeyQ: ['best', 0], KeyW: ['best', 1], KeyE: ['best', 2], KeyR: ['best', 3], KeyT: ['noBest'],
    KeyA: ['tier', 'toss'], KeyS: ['tier', 'fine'], KeyD: ['tier', 'good'], KeyF: ['tier', 'great'], KeyG: ['allBad'],
    KeyZ: ['worst', 0], KeyX: ['worst', 1], KeyC: ['worst', 2], KeyV: ['worst', 3], KeyB: ['noWorst'],
    KeyJ: ['tag', 'fake'], KeyK: ['tag', 'no_groove'], KeyL: ['tag', 'sour'], Semicolon: ['tag', 'boring'],
    KeyU: ['tag', 'nodding'], KeyI: ['tag', 'sax_line'], KeyO: ['tag', 'chords'],
    Space: ['play'], Enter: ['commit'], NumpadEnter: ['commit'], Backspace: ['undo'], Escape: ['end']
  };
  // When a browser gives no event.code, fall back to the character.
  var KEY_BY_CHAR = {
    '1': 'Digit1', '2': 'Digit2', '3': 'Digit3', '4': 'Digit4', '5': 'Digit5',
    q: 'KeyQ', w: 'KeyW', e: 'KeyE', r: 'KeyR', t: 'KeyT', a: 'KeyA', s: 'KeyS', d: 'KeyD', f: 'KeyF', g: 'KeyG',
    z: 'KeyZ', x: 'KeyX', c: 'KeyC', v: 'KeyV', b: 'KeyB', j: 'KeyJ', k: 'KeyK', l: 'KeyL', ';': 'Semicolon',
    u: 'KeyU', i: 'KeyI', o: 'KeyO', ' ': 'Space', Spacebar: 'Space', Enter: 'Enter', Backspace: 'Backspace',
    Escape: 'Escape', Esc: 'Escape'
  };

  var NEG_TAGS = ['fake', 'no_groove', 'sour', 'boring'];
  var POS_TAGS = ['nodding', 'sax_line', 'chords'];
  var TAG_WORDS = { fake: 'fake', no_groove: 'no groove', sour: 'sour', boring: 'boring',
                    nodding: 'nodding', sax_line: 'sax line', chords: 'chords' };
  var TAG_KEYS = { fake: 'J', no_groove: 'K', sour: 'L', boring: ';', nodding: 'U', sax_line: 'I', chords: 'O' };
  var TIERS = ['toss', 'fine', 'good', 'great'];
  var TIER_WORDS = { toss: 'Toss', fine: 'Fine', good: 'Good', great: 'Great' };
  var TIER_KEYS = { toss: 'A', fine: 'S', good: 'D', great: 'F' };
  var BEST_KEYS = ['Q', 'W', 'E', 'R'];
  var WORST_KEYS = ['Z', 'X', 'C', 'V'];

  // What each mode asks, in plain words, and which marks it takes.
  var MODES = {
    shootout: { best: true, worst: true, hatches: true, allBad: 'all bad', tags: false, tier: false,
                title: 'Shootout', ask: 'Which one is best, and which is worst?' },
    keep3: { best: true, allBad: 'none acceptable', title: 'Keep one',
             ask: 'Which draft do you keep? Press G if none is acceptable.' },
    bws4: { best: true, worst: true, hatches: true, allBad: 'all four bad', tags: true, title: 'Best and worst',
            ask: 'Which one is best, and which is worst?' },
    triage: { tier: true, title: 'Triage', ask: 'Toss, Fine, Good or Great?' },
    ab: { best: true, title: 'A or B', ask: 'Which one do you like more?' },
    playlist: { playlist: true, title: 'Playlist',
                ask: 'Just listen. Press Q for a heart or Z for an x, only when one stands out.' }
  };

  function keyHelp(mode, k) {
    var listen = k > 1 ? '1-' + k + ' listen' : '1 listen';
    var common = ['5 from bar 1', 'Space play or pause', 'Enter save', 'Backspace undo', 'Esc stop'];
    switch (mode) {
      case 'shootout':
        return [listen, BEST_KEYS.slice(0, k).join(' ') + ' best', WORST_KEYS.slice(0, k).join(' ') + ' worst',
                'T no clear best', 'B no clear worst', 'G all bad'].concat(common);
      case 'bws4':
        return [listen, 'Q W E R best', 'Z X C V worst', 'T no clear best', 'B no clear worst', 'G all four bad',
                'tags: J fake, K no groove, L sour, ; boring (go with the worst)',
                'U nodding, I sax line, O chords (go with the best)'].concat(common);
      case 'keep3':
        return [listen, BEST_KEYS.slice(0, k).join(' ') + ' keep', 'G none acceptable'].concat(common);
      case 'triage':
        return [listen, 'A Toss', 'S Fine', 'D Good', 'F Great'].concat(common);
      case 'ab':
        return [listen, 'Q pick 1', 'W pick 2'].concat(common);
      case 'playlist':
        return ['Q heart', 'Z x', 'Enter next loop', '5 from bar 1', 'Space play or pause',
                'Backspace undo', 'Esc stop'];
    }
    return common;
  }

  // ------------------------------------------------------------------ Rules (no DOM)

  function newId() {
    var b = new Uint8Array(12), c = global.crypto, s = 'e', i;
    if (c && typeof c.getRandomValues === 'function') c.getRandomValues(b);
    else for (i = 0; i < 12; i++) b[i] = Math.floor(Math.random() * 256);
    for (i = 0; i < 12; i++) s += (b[i] < 16 ? '0' : '') + b[i].toString(16);
    return s;
  }

  function newMarks() {
    return { best: null, worst: null, noBest: false, noWorst: false, allBad: false, tier: null, tag: null,
             mark: null, t: {} };
  }

  function cloneMarks(m) {
    var t = {};
    Object.keys(m.t).forEach(function (key) { t[key] = m.t[key]; });
    return { best: m.best, worst: m.worst, noBest: m.noBest, noWorst: m.noWorst, allBad: m.allBad,
             tier: m.tier, tag: m.tag, mark: m.mark, t: t };
  }

  // One key's effect on the marks of a round with k clips. Returns {marks} or {marks, error} (unchanged).
  // Best, worst and tier keys set; T, B, G and the tag keys toggle; Backspace (in the page) undoes.
  function apply(mode, m, act, k, tMs) {
    var spec = MODES[mode] || {};
    var n = cloneMarks(m);
    var kind = act[0], arg = act[1];
    var t = Math.max(0, Math.round(tMs || 0));
    function no(msg) { return { marks: m, error: msg }; }
    if (spec.playlist) {
      if (kind === 'best' && arg === 0) { n.mark = m.mark === 'heart' ? null : 'heart'; if (n.mark) n.t.mark = t; return { marks: n }; }
      if (kind === 'worst' && arg === 0) { n.mark = m.mark === 'cross' ? null : 'cross'; if (n.mark) n.t.mark = t; return { marks: n }; }
      return no('In playlist mode: Q for a heart, Z for an x, Enter for the next loop.');
    }
    switch (kind) {
      case 'best':
        if (!spec.best) return no(mode === 'triage' ? 'Triage takes a tier: A Toss, S Fine, D Good, F Great.' : 'That key does nothing here.');
        if (arg >= k) return no('There is no clip ' + (arg + 1) + ' in this round.');
        n.best = arg;
        if (n.worst === arg) n.worst = null;
        n.noBest = false;
        n.allBad = false;
        n.t[mode === 'ab' ? 'pick' : 'best'] = t;
        return { marks: n };
      case 'worst':
        if (!spec.worst) return no(mode === 'keep3' ? 'Keep one (' + BEST_KEYS.slice(0, k).join(' ') + '), or press G if none is acceptable.'
                                   : mode === 'ab' ? 'Pick one: Q for 1 or W for 2.' : 'That key does nothing here.');
        if (arg >= k) return no('There is no clip ' + (arg + 1) + ' in this round.');
        n.worst = arg;
        if (n.best === arg) n.best = null;
        n.noWorst = false;
        n.t.worst = t;
        return { marks: n };
      case 'noBest':
        if (!spec.hatches) return no('"No clear best" is not used in this mode.');
        n.noBest = !m.noBest;
        if (n.noBest) { n.best = null; n.t.no_best = t; }
        return { marks: n };
      case 'noWorst':
        if (!spec.hatches) return no('"No clear worst" is not used in this mode.');
        n.noWorst = !m.noWorst;
        if (n.noWorst) { n.worst = null; n.t.no_worst = t; }
        return { marks: n };
      case 'allBad':
        if (!spec.allBad) return no(mode === 'ab' ? 'The A or B rounds always need a pick.' : 'That key does nothing here.');
        n.allBad = !m.allBad;
        if (n.allBad) { n.best = null; n.t.all_bad = t; }
        return { marks: n };
      case 'tier':
        if (!spec.tier) return no('Tiers (A S D F) are for triage.');
        if (TIERS.indexOf(arg) < 0) return no('Unknown tier.');
        n.tier = arg;
        n.t.tier = t;
        return { marks: n };
      case 'tag':
        if (!spec.tags) return no('Tags are only used in best-and-worst rounds.');
        n.tag = m.tag === arg ? null : arg;
        if (n.tag) n.t.tag = t;
        return { marks: n };
    }
    return no('That key does nothing here.');
  }

  // Why the round cannot be saved yet, in plain words (null when it can).
  function problem(mode, m) {
    if (mode === 'bws4' || mode === 'shootout') {
      if (m.tag) {
        var neg = NEG_TAGS.indexOf(m.tag) >= 0, word = '"' + TAG_WORDS[m.tag] + '"';
        if (neg && m.worst === null) return 'The tag ' + word + ' goes with the worst pick. Pick a worst (Z X C V), or press ' + TAG_KEYS[m.tag] + ' again to drop it.';
        if (!neg && m.best === null) return 'The tag ' + word + ' goes with the best pick. Pick a best (Q W E R), or press ' + TAG_KEYS[m.tag] + ' again to drop it.';
      }
      if (m.allBad) return null;
      if (m.best === null && !m.noBest) return 'Pick a best (Q W E R), or press T for "no clear best".';
      if (m.worst === null && !m.noWorst) return 'Pick a worst (Z X C V), or press B for "no clear worst".';
      return null;
    }
    if (mode === 'keep3') return m.best === null && !m.allBad ? 'Keep one (Q W E), or press G if none is acceptable.' : null;
    if (mode === 'triage') return m.tier ? null : 'Pick a tier: A Toss, S Fine, D Good, F Great.';
    if (mode === 'ab') return m.best === null ? 'Pick one: Q for 1 or W for 2.' : null;
    return null;
  }

  // The time to each mark the vote holds. A mark set and later cleared (T after a best, a tag pressed twice, G
  // after a best) leaves no time behind.
  function times(mode, m, extra) {
    var held = { first_play: true };
    if (m.best !== null) held[mode === 'ab' ? 'pick' : 'best'] = true;
    if (m.worst !== null) held.worst = true;
    if (m.noBest) held.no_best = true;
    if (m.noWorst) held.no_worst = true;
    if (m.allBad) held.all_bad = true;
    if (m.tag) held.tag = true;
    if (m.tier) held.tier = true;
    if (m.mark) held.mark = true;
    var t = {};
    Object.keys(m.t).forEach(function (key) { if (held[key]) t[key] = Math.max(0, Math.round(m.t[key])); });
    Object.keys(extra).forEach(function (key) { t[key] = Math.max(0, Math.round(extra[key] || 0)); });
    return t;
  }

  function listening(round, info) {
    var ids = round.clips.map(function (c) { return c.id; });
    var clips = round.clips.map(function (c, i) {
      var h = (info.clips && info.clips[i]) || {};
      return { clip: c.id, pos: i + 1, heard_bars: h.heard_bars, heard_ms: Math.max(0, Math.round(h.heard_ms || 0)) };
    });
    var order = (info.playOrder || []).map(function (i) { return i + 1; });
    var last = info.lastHeard === null || info.lastHeard === undefined ? null : ids[info.lastHeard];
    return { clips: clips, last_heard: last === undefined ? null : last, play_order: order };
  }

  // The vote event for a round. `info` carries what the audio saw (see listenInfo in the page).
  function votePayload(mode, round, m, info) {
    var ids = round.clips.map(function (c) { return c.id; });
    function at(i) { return i === null || i === undefined ? null : ids[i]; }
    var l = listening(round, info);
    var p = { type: mode, event_id: info.eventId || newId(), round: round.id, clips: l.clips,
              last_heard: l.last_heard, switches: info.switches || 0, play_order: l.play_order,
              autoplay_done: !!info.autoplayDone, t_ms: times(mode, m, { commit: info.tCommit }),
              device: info.device || null, audio: info.audio || null };
    if (mode === 'bws4' || mode === 'shootout') {
      p.best = at(m.best); p.worst = at(m.worst);
      p.no_best = !!m.noBest; p.no_worst = !!m.noWorst; p.all_bad = !!m.allBad;
      if (mode === 'bws4') p.tag = m.tag || null;
    } else if (mode === 'keep3') {
      p.best = at(m.best); p.all_bad = !!m.allBad;
    } else if (mode === 'triage') {
      p.tier = m.tier;
    } else if (mode === 'ab') {
      p.pick = at(m.best);
    } else if (mode === 'playlist') {
      p.mark = m.mark || 'none';
      p.plays = Math.max(0, Math.min(64, info.plays || 0));
    }
    return p;
  }

  // A round that started but was not saved (Esc, closing the tab, taking back the previous vote).
  function abandonPayload(round, info, why) {
    var p = { type: 'abandon', event_id: newId(), round: round.id, why: why };
    if (info && info.clips && info.clips.length === round.clips.length) {
      var l = listening(round, info);
      p.clips = l.clips;
      p.last_heard = l.last_heard;
      p.play_order = l.play_order;
      p.switches = info.switches || 0;
      p.autoplay_done = !!info.autoplayDone;
      p.t_ms = { abandon: Math.max(0, Math.round(info.tCommit || 0)) };
    }
    if (info && info.device) p.device = info.device;
    if (info && info.audio) p.audio = info.audio;
    return p;
  }

  function sessionPayload(what, roundId, device, audio, detail) {
    var p = { type: 'session', event_id: newId(), what: what };
    if (roundId) p.round = roundId;
    if (device) p.device = device;
    if (audio) p.audio = audio;
    if (detail) p.detail = String(detail).slice(0, 200);
    return p;
  }

  var Rules = { MODES: MODES, KEYMAP: KEYMAP, KEY_BY_CHAR: KEY_BY_CHAR, NEG_TAGS: NEG_TAGS, POS_TAGS: POS_TAGS,
                TIERS: TIERS, newId: newId, newMarks: newMarks, cloneMarks: cloneMarks, apply: apply,
                problem: problem, votePayload: votePayload, abandonPayload: abandonPayload,
                sessionPayload: sessionPayload, keyHelp: keyHelp };
  global.LofiRules = Rules;

  // ------------------------------------------------------------------ the page
  if (typeof document === 'undefined' || !document.getElementById) return;

  var S = {
    cfg: null, man: null, st: null, mode: null, rounds: new Map(), engine: null,
    cur: null,                 // the round on screen (vote view)
    view: 'loading',           // loading | vote | ended | audit | error
    device: 'airpods',
    lastActive: 0, microStart: 0, nudges: 0,
    lastPost: 0,               // when the page last sent an event (playback heartbeats fill longer silences)
    armedUndo: 0,
    audit: null, replay: null,
    dom: { ruler: [], tiles: [], cells: [] },
    flashTimer: null
  };

  function $(id) { return document.getElementById(id); }

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }

  function clipUrl(c) { return '/clips/' + encodeURIComponent(c.file); }

  // -- talking to the server
  function request(url, opts) {
    return fetch(url, opts).then(function (r) {
      return r.json().then(function (b) { return b; }, function () { return null; }).then(function (body) {
        if (!r.ok) {
          var err = new Error((body && body.error) || ('HTTP ' + r.status));
          err.status = r.status;
          err.body = body;
          throw err;
        }
        return body;
      });
    }, function () {
      var err = new Error('The rating server is not answering. Is make rate still running? Start it again, then reload this page.');
      err.network = true;
      throw err;
    });
  }

  function getJSON(url) { return request(url, { cache: 'no-store' }); }

  function post(payload) {
    S.lastPost = Date.now();
    return request('/api/event', { method: 'POST', cache: 'no-store', headers: { 'Content-Type': 'application/json' },
                                   body: JSON.stringify(payload) });
  }

  // For the moment the page closes: keepalive fetch, else sendBeacon.
  function beacon(payload) {
    var body = JSON.stringify(payload);
    S.lastPost = Date.now();
    try {
      fetch('/api/event', { method: 'POST', keepalive: true, headers: { 'Content-Type': 'application/json' }, body: body })
        .catch(function () {});
      return;
    } catch (e) { /* fall through */ }
    try {
      if (global.navigator && global.navigator.sendBeacon) global.navigator.sendBeacon('/api/event', new Blob([body], { type: 'text/plain' }));
    } catch (e) { /* nothing more to try */ }
  }

  function softError(err) { if (global.console) global.console.warn('rate:', err); }

  function errText(err) { return (err && err.message) || String(err || 'Something went wrong.'); }

  function audioInfo() { return S.engine ? S.engine.info() : null; }

  function session(what, roundId, detail) {
    return post(Rules.sessionPayload(what, roundId, S.device, audioInfo(), detail)).catch(softError);
  }

  // -- start
  function boot() {
    document.addEventListener('keydown', onKey);
    global.addEventListener('pagehide', onPageHide);
    global.addEventListener('pageshow', function (ev) { if (ev.persisted) global.location.reload(); });
    $('device').addEventListener('change', onDeviceChange);
    $('tiles').addEventListener('click', onTileClick);
    $('buttons').addEventListener('click', onButtonClick);
    $('flags').addEventListener('click', onButtonClick);
    $('audit-list').addEventListener('click', onAuditClick);
    load().catch(fatal);
  }

  function load() {
    return Promise.all([getJSON('/api/config'), getJSON('/manifest.json')]).then(function (got) {
      S.cfg = got[0];
      S.man = got[1];
      S.mode = S.man.mode;
      S.man.rounds.forEach(function (r) { S.rounds.set(r.id, r); });
      if (!global.LofiAudio) throw new Error('audio.js did not load.');
      S.engine = new global.LofiAudio.Engine({ sampleRate: S.man.sr || 0, rampMs: S.cfg.switch_ramp_ms });
      S.engine.onInterrupt = function (state) {          // the engine has muted and holds until Space
        showBanner('device', 'The system paused the audio. Press Space to go on.');
        session('pause', S.cur ? S.cur.round.id : null, 'system: ' + state);
        render();
      };
      initDevice();
      $('ui').textContent = 'page ' + S.cfg.ui;
      return post(Rules.sessionPayload('start', null, S.device, null));
    }).then(function (reply) {
      S.st = reply.state;
      watchDevices();
      // A reload within the micro-session keeps its clock: the break nudge counts from when it began.
      var t = Date.now(), began = Date.parse(S.st.micro_start || '');
      S.lastActive = t;
      S.microStart = isFinite(began) && began <= t ? began : t;
      S.nudges = Math.max(0, S.st.micro_nudges | 0);
      setInterval(tick, 100);
      if (S.st.done) return showAudit(false);
      S.view = 'vote';
      return openRound(S.st.current, false);
    });
  }

  // -- rounds
  // `autostart`: play once loaded (after a save, a take-back, Space in the stopped view). While playback is held
  // (the output changed, or the system stopped the audio) it waits for Space; `explicit` is that Space.
  function openRound(id, autostart, explicit) {
    if (S.cur && S.cur.player) S.cur.player.stop();
    var round = S.rounds.get(id);
    if (!round) return Promise.reject(new Error('The server named a round this page does not know. Reload the page.'));
    var cur = { round: round, k: round.clips.length, marks: Rules.newMarks(), undo: [], player: null,
                buffers: null, started: false, startedAt: 0, switches: 0, auto: null, autoplayDone: false,
                committing: false, eventId: null, loadError: null, endAt: Infinity };
    S.cur = cur;
    S.view = 'vote';
    render();
    return Promise.all(round.clips.map(function (c) { return S.engine.load(clipUrl(c)); })).then(function (bufs) {
      if (S.cur !== cur) return null;
      cur.buffers = bufs;
      prefetch();
      if (autostart && S.cfg.autoplay.start_next_round && S.view === 'vote') return startRound(cur, undefined, explicit);
      render();
      return null;
    }, function (err) {
      if (S.cur !== cur) return;
      cur.loadError = 'Could not load this round: ' + errText(err) +
        (/decode/i.test(errText(err)) ? ' This browser may not play FLAC; try Chrome, or ask Claude for WAV copies.' : '') +
        ' Press Esc to skip it for now.';
      render();
    });
  }

  function prefetch() {
    var keep = [];
    function add(id) { var r = S.rounds.get(id); if (r) r.clips.forEach(function (c) { keep.push(clipUrl(c)); }); }
    if (S.cur) add(S.cur.round.id);
    ((S.st && S.st.upcoming) || []).slice(0, 1).forEach(add);
    S.engine.keepOnly(keep);
    keep.forEach(function (u) { S.engine.load(u).catch(function () {}); });
  }

  // `explicit` is Space (or the Play button): the one gesture that lifts a hold. Anything else (an auto-start,
  // 1-4 or 5 before the round has started) waits while playback is held.
  function startRound(cur, clip, explicit) {
    if (!cur || cur.started || !cur.buffers || cur.loadError) return Promise.resolve();
    if (S.engine.held && !explicit) { render(); return Promise.resolve(); }
    return S.engine.resume().then(function () {
      if (S.cur !== cur || cur.started) return;
      if (S.engine.ctx && S.engine.ctx.state !== 'running') {    // the browser wants a key press first
        setMsg('Press Space to play.');
        render();
        return;
      }
      var p = new global.LofiAudio.RoundPlayer(S.engine, cur.buffers,
                                               { bars: S.man.bars, barFraction: S.cfg.heard.bar_fraction });
      var t0 = p.start(clip === undefined ? 0 : clip);
      cur.player = p;
      cur.started = true;
      cur.startedAt = now() + Math.max(0, (t0 - p.now()) * 1000);
      if (S.mode === 'playlist') {
        cur.endAt = t0 + (cur.round.plays || 3) * p.loopDur;
        p.scheduleStop(cur.endAt);
      } else if (clip === undefined) {
        var seq = [];
        for (var i = 0; i < cur.k; i++) seq.push(i);
        cur.auto = { endsAt: p.planAuto(seq, S.cfg.autoplay.bars_per_clip, t0, S.cfg.autoplay.wrap_bars) };
      }
      if (p.mismatch) setMsg('These clips are not the same length, so they may drift apart. Tell Claude.');
      hideBanner('device');
      noteActive();
      session('round_start', cur.round.id);
      render();
    });
  }

  function now() { return (global.performance && global.performance.now) ? global.performance.now() : Date.now(); }

  function tMs(cur) { return cur.started ? Math.max(0, now() - cur.startedAt) : 0; }

  // What the audio saw, for the event.
  function listenInfo(cur) {
    var p = cur.player, info = { eventId: cur.eventId, device: S.device, audio: audioInfo(), switches: cur.switches,
                                 tCommit: tMs(cur), clips: [], playOrder: [], lastHeard: null,
                                 autoplayDone: cur.autoplayDone, plays: 0 };
    var i;
    if (p) {
      p.update();
      for (i = 0; i < cur.k; i++) info.clips.push({ heard_bars: p.heardString(i), heard_ms: p.heardMs(i) });
      info.playOrder = p.playOrder();
      info.lastHeard = p.upAt();
      info.plays = p.passesDone();
    } else {
      var zero = new Array(S.man.bars + 1).join('0');
      for (i = 0; i < cur.k; i++) info.clips.push({ heard_bars: zero, heard_ms: 0 });
    }
    return info;
  }

  // -- keys
  function onKey(ev) {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    var tag = ev.target && ev.target.tagName;
    if (tag === 'SELECT' || tag === 'INPUT' || tag === 'TEXTAREA') {
      if (ev.key === 'Escape' && ev.target.blur) ev.target.blur();
      return;
    }
    var code = ev.code && KEYMAP[ev.code] ? ev.code : (KEY_BY_CHAR[ev.key] || KEY_BY_CHAR[String(ev.key || '').toLowerCase()]);
    var act = code ? KEYMAP[code] : null;
    if (!act) return;
    ev.preventDefault();
    if (ev.repeat) return;
    noteActive();
    try { dispatch(act); } catch (err) { fatal(err); }
  }

  function dispatch(act) {
    var kind = act[0];
    if (S.view === 'audit') return auditKey(act);
    if (S.view === 'error' || S.view === 'loading') return;
    if (S.view === 'ended') {
      if (kind === 'play' || kind === 'commit') return goOn(kind === 'play');
      return setMsg('Stopped. Press Space to go on.');
    }
    var cur = S.cur;
    if (!cur) return;
    if (kind === 'play') return togglePlay();
    if (kind === 'end') return endSession();
    if (kind === 'undo') return undo();
    if (kind === 'commit') return commit();
    if (kind === 'listen') return listen(act[1]);
    if (kind === 'replay') return replayRound();
    return mark(act);
  }

  var HELD_TEXT = 'Paused: check "Listening on" below, then press Space to go on.';

  function togglePlay() {
    var cur = S.cur;
    if (cur.loadError) return setMsg(cur.loadError);
    if (!cur.buffers) return setMsg('Still loading the clips...');
    if (!cur.started) return startRound(cur, undefined, true).catch(fatal);
    if (S.engine.paused) {
      S.engine.resume();                          // Space also lifts a hold
      hideBanner('device');
      session('resume', cur.round.id);
    } else {
      S.engine.pause();
      session('pause', cur.round.id);
    }
    render();
  }

  function listen(i) {
    var cur = S.cur;
    if (i >= cur.k) return setMsg('There is no clip ' + (i + 1) + ' in this round.');
    if (!cur.started) {
      if (!cur.buffers) return setMsg('Still loading the clips...');
      if (S.engine.held) return setMsg(HELD_TEXT);
      return startRound(cur, i).catch(fatal);     // starting on a chosen clip: no auto-play
    }
    var p = cur.player;
    if (!cur.auto && p.lastPlanned() === i) return;
    p.change(function (t) {
      p.cancelFrom(t);
      cur.auto = null;                            // listening by hand ends auto-play
      if (p.lastPlanned() !== i) { p.switchTo(i, t); cur.switches += 1; }
      render();
    });
  }

  function replayRound() {
    var cur = S.cur;
    if (!cur.started) return S.engine.held ? setMsg(HELD_TEXT) : togglePlay();
    var p = cur.player;
    S.engine.unlock();                            // while held this does nothing: back to bar 1, still paused
    if (S.engine.held) setMsg('Back to bar 1. ' + HELD_TEXT);
    p.restart(p.upAt());
    cur.auto = null;
    if (S.mode === 'playlist') {
      cur.endAt = p.t0 + (cur.round.plays || 3) * p.loopDur;
      p.scheduleStop(cur.endAt);
    }
    render();
  }

  function mark(act) {
    var cur = S.cur;
    if (!cur.started) return setMsg('Press Space to start listening first.');
    if (cur.committing) return;
    var res = Rules.apply(S.mode, cur.marks, act, cur.k, tMs(cur));
    if (res.error) return setMsg(res.error);
    cur.undo.push(cur.marks);
    cur.marks = res.marks;
    cur.eventId = null;
    S.armedUndo = 0;
    setMsg('');
    // Marking the clip that is playing moves on to the next unheard clip.
    if ((act[0] === 'best' || act[0] === 'worst') && S.mode !== 'playlist' && cur.player) {
      cur.player.update();
      if (act[1] === cur.player.upAt()) skipFrom(cur, act[1]);
    }
    render();
  }

  function skipFrom(cur, from) {
    var p = cur.player, min = S.cfg.heard.min_bars, next = -1, j;
    for (j = 1; j < cur.k; j++) {
      var c = (from + j) % cur.k;
      if (p.heardCount(c) < min) { next = c; break; }
    }
    if (next < 0) return;
    var wasAuto = !!cur.auto;
    p.change(function (t) {
      p.cancelFrom(t);
      p.switchTo(next, t);
      if (wasAuto) {
        var seq = [next];
        for (var c2 = next + 1; c2 < cur.k; c2++) if (p.heardCount(c2) < min) seq.push(c2);
        cur.auto = { endsAt: p.planAuto(seq, S.cfg.autoplay.bars_per_clip, t, 0) };
      } else {
        cur.auto = null;
      }
      render();
    });
  }

  function commit() {
    var cur = S.cur;
    if (cur.committing) return;
    if (S.mode === 'playlist') return playlistCommit(cur, true);
    if (!cur.started) return setMsg('Press Space to listen first.');
    var why = Rules.problem(S.mode, cur.marks);
    if (why) return setMsg(why);
    if (!cur.eventId) cur.eventId = Rules.newId();
    var payload = Rules.votePayload(S.mode, cur.round, cur.marks, listenInfo(cur));
    cur.committing = true;
    render();
    post(payload).then(function (reply) {
      S.st = reply.state;
      afterCommit(cur);
    }, function (err) {
      cur.committing = false;
      if (err.status === 409 && err.body && err.body.state) {
        S.st = err.body.state;
        setMsg('This round already had a vote; moving on.');
        return afterCommit(cur);
      }
      setMsg('Not saved: ' + errText(err) + ' Press Enter to try again.');
      render();
    });
  }

  function afterCommit(cur) {
    if (cur.player) cur.player.stop();
    flash('Saved.');
    S.armedUndo = 0;
    if (S.view === 'ended') return render();          // Esc was pressed while saving: stay stopped
    if (S.st.done) return showAudit(true);
    return openRound(S.st.current, true).catch(fatal);
  }

  // Playlist: a loop's event is saved when its passes end, or on Enter (next loop now), or on Esc if marked.
  function playlistCommit(cur, advance) {
    if (cur.committing) return Promise.resolve();
    if (!cur.started) return advance ? startRound(cur) : Promise.resolve();
    cur.committing = true;
    if (!cur.eventId) cur.eventId = Rules.newId();
    var payload = Rules.votePayload('playlist', cur.round, cur.marks, listenInfo(cur));
    return post(payload).then(function (reply) {
      S.st = reply.state;
      return true;
    }, function (err) {
      if (err.status === 409 && err.body && err.body.state) { S.st = err.body.state; return true; }
      cur.committing = false;
      setMsg('Not saved: ' + errText(err) + ' Press Enter to try again.');
      render();
      return false;
    }).then(function (saved) {
      if (!saved) return null;
      if (cur.player) cur.player.stop();
      if (!advance || S.view === 'ended') return null;
      if (S.st.done) return showAudit(true);
      return openRound(S.st.current, true);
    }).catch(fatal);
  }

  function undo() {
    var cur = S.cur;
    if (cur && cur.undo.length && !cur.committing) {
      cur.marks = cur.undo.pop();
      cur.eventId = null;
      setMsg('Undone.');
      render();
      return;
    }
    var last = S.st && S.st.last_vote;
    if (!last) return setMsg('Nothing to undo.');
    var t = Date.now();
    if (t - S.armedUndo > 3000) {
      S.armedUndo = t;
      return setMsg('Press Backspace again to take back your last vote. That round will be asked again.');
    }
    S.armedUndo = 0;
    retract(last.event_id, 'undo').catch(fatal);
  }

  // Take a saved vote back (append-only: the server logs a retract) and ask that round again.
  function retract(target, why) {
    var cur = S.cur, first = Promise.resolve();
    if (S.view === 'vote' && cur && cur.started && !cur.committing) {
      first = post(Rules.abandonPayload(cur.round, listenInfo(cur), 'undo')).catch(softError);
    }
    if (cur && cur.player) cur.player.stop();
    return first.then(function () {
      return post({ type: 'retract', event_id: Rules.newId(), target: target, why: why });
    }).then(function (reply) {
      S.st = reply.state;
      stopReplay();
      return openRound(S.st.current, true);
    }).then(function () {
      setMsg('Vote taken back. Here is that round again.');
    });
  }

  function endSession() {
    var cur = S.cur, first = Promise.resolve();
    if (cur && cur.loadError && !cur.started && !cur.committing) {
      first = post(Rules.abandonPayload(cur.round, null, 'error')).catch(softError);   // try it again at the end
      cur.committing = true;
    } else if (cur && cur.started && !cur.committing) {
      if (S.mode === 'playlist' && cur.marks.mark) first = playlistCommit(cur, false);
      else first = post(Rules.abandonPayload(cur.round, listenInfo(cur), 'esc')).catch(softError);
      cur.committing = true;
    }
    if (cur && cur.player) cur.player.stop();
    S.view = 'ended';
    hideBanner();
    render();
    first.then(function () { return session('end', null); }).then(function () {
      return getJSON('/api/state');
    }).then(function (st) { S.st = st; render(); }).catch(softError);
  }

  // Space (explicit) or Enter in the stopped view.
  function goOn(explicit) {
    hideBanner();
    getJSON('/api/state').then(function (st) {
      S.st = st;
      if (st.done) return showAudit(false);
      return openRound(st.current, true, !!explicit);
    }).catch(fatal);
  }

  function onPageHide() {
    var cur = S.cur;
    if (S.view !== 'vote' || !cur || !cur.started || cur.committing) return;
    var info = listenInfo(cur);
    beacon(S.mode === 'playlist' && cur.marks.mark ? Rules.votePayload('playlist', cur.round, cur.marks, info)
                                                   : Rules.abandonPayload(cur.round, info, 'pagehide'));
    cur.committing = true;
  }

  // -- the clock: heard bars, auto-play's end, playlist passes, the break nudge, playback heartbeats
  function tick() {
    var cur = S.cur, ctxNow = S.engine ? S.engine.now() : 0;
    if (S.view === 'vote' && cur && cur.player && cur.started) {
      cur.player.update();
      if (cur.auto && ctxNow >= cur.auto.endsAt) { cur.auto = null; cur.autoplayDone = true; }
      if (S.mode === 'playlist' && !cur.committing && ctxNow >= cur.endAt) {
        cur.autoplayDone = true;
        playlistCommit(cur, true);
      }
    }
    if (S.replay && S.replay.player) S.replay.player.update();
    var playing = isPlaying();
    if (playing) { noteActive(); heartbeat(); }
    checkNudge(playing);
    renderLive();
  }

  // The server ends a micro-session after micro_gap_minutes without events. Audio that keeps playing (a round
  // left looping) is not a gap, so after half that time with no other event the page says it is still playing.
  function heartbeat() {
    var every = (S.cfg ? S.cfg.micro_gap_minutes : 10) * 60000 / 2;
    if (Date.now() - S.lastPost < every) return;
    session('playing', S.view === 'vote' && S.cur ? S.cur.round.id : null);
  }

  function isPlaying() {
    if (!S.engine || !S.engine.running()) return false;
    var p = S.view === 'vote' ? (S.cur && S.cur.player) : (S.view === 'audit' && S.replay ? S.replay.player : null);
    return !!(p && p.started && !p.stopped);
  }

  function noteActive() {
    var t = Date.now(), gap = (S.cfg ? S.cfg.micro_gap_minutes : 10) * 60000;
    if (t - S.lastActive >= gap) { S.microStart = t; S.nudges = 0; }
    S.lastActive = t;
  }

  function checkNudge(playing) {
    if (!S.cfg || !playing) return;
    var mins = (Date.now() - S.microStart) / 60000;
    if (mins >= S.cfg.nudge_minutes * (S.nudges + 1)) {
      S.nudges += 1;
      showBanner('nudge', 'You have been listening for ' + Math.round(mins) + ' minutes. A short break helps your ears. ' +
                 'Press Esc to stop (your place is saved), or carry on.');
      session('nudge', S.cur ? S.cur.round.id : null);
    }
  }

  // -- devices
  var DEVICE_WORDS = { airpods: 'AirPods', headphones: 'other headphones', laptop: 'laptop speakers',
                       speakers: 'other speakers' };

  function initDevice() {
    var sel = $('device'), saved = null;
    sel.innerHTML = '';
    S.cfg.devices.forEach(function (d) {
      var o = el('option', null, DEVICE_WORDS[d] || d);
      o.value = d;
      sel.appendChild(o);
    });
    try { saved = global.localStorage.getItem('lofi-rate-device'); } catch (e) { saved = null; }
    S.device = S.cfg.devices.indexOf(saved) >= 0 ? saved : S.cfg.default_device;
    sel.value = S.device;
  }

  function onDeviceChange() {
    S.device = $('device').value;
    try { global.localStorage.setItem('lofi-rate-device', S.device); } catch (e) { /* not kept */ }
    $('device').blur();
  }

  // A devicechange pauses playback (taste_pipeline §1 Logging) and holds it: nothing plays again (not the next
  // round after Enter, not 5 or 1-4, not a round reopened by Backspace) until he checks the output and presses Space.
  function watchDevices() {
    var md = global.navigator && global.navigator.mediaDevices;
    if (!md || typeof md.addEventListener !== 'function') return;
    md.addEventListener('devicechange', function () {
      var was = isPlaying();
      S.engine.hold('device');
      showBanner('device', 'Your audio output changed' + (was ? ', so playback is paused' : '') +
                 '. Check the "Listening on" choice below, then press Space.');
      session('devicechange', S.cur ? S.cur.round.id : null);
      render();
    });
  }

  // -- messages
  function setMsg(text) { var m = $('msg'); if (m) m.textContent = text || ''; }

  function flash(text) {
    setMsg(text);
    if (S.flashTimer) clearTimeout(S.flashTimer);
    S.flashTimer = setTimeout(function () { if ($('msg').textContent === text) setMsg(''); }, 1500);
  }

  function showBanner(kind, text) {
    var b = $('banner');
    b.textContent = text;
    b.dataset.kind = kind;
    b.hidden = false;
  }

  // Without a kind, hides any banner except the output-changed one while playback is still held.
  function hideBanner(kind) {
    var b = $('banner');
    if (kind ? b.dataset.kind !== kind : (b.dataset.kind === 'device' && S.engine && S.engine.held)) return;
    b.hidden = true;
  }

  function fatal(err) {
    softError(err);
    if (S.cur && S.cur.player) S.cur.player.stop();
    S.view = 'error';
    $('error-text').textContent = errText(err);
    render();
  }

  // -- drawing
  function render() {
    $('loading').hidden = S.view !== 'loading';
    $('vote').hidden = !(S.view === 'vote' || S.view === 'ended');
    $('audit').hidden = S.view !== 'audit';
    $('error').hidden = S.view !== 'error';
    if (S.man) {
      $('batch').textContent = S.man.batch;
      $('mode').textContent = (MODES[S.mode] || {}).title || S.mode;
      $('est').textContent = S.man.est_min ? 'about ' + Math.round(S.man.est_min) + ' min in all' : '';
    }
    if (S.st && S.view !== 'audit') {
      $('progress').textContent = S.st.done ? 'all ' + S.st.total + ' rounds done'
                                            : 'round ' + S.st.index + ' of ' + S.st.total;
    }
    if (S.view === 'vote' || S.view === 'ended') renderVote();
    renderLive();
  }

  function renderVote() {
    var cur = S.cur, spec = MODES[S.mode] || {};
    $('note').textContent = (S.man && S.man.note) || '';
    $('ask').textContent = spec.ask || '';
    $('scene').textContent = cur ? cur.round.scene : '';
    $('ended').hidden = S.view !== 'ended';
    buildRuler();
    buildTiles(cur);
    buildFlags(cur);
    var keys = $('keys');
    keys.innerHTML = '';
    keyHelp(S.mode, cur ? cur.k : 1).forEach(function (line) { keys.appendChild(el('span', 'key', line)); });
    var playLabel = $('buttons').querySelector('[data-act="play"] span');
    if (playLabel) playLabel.textContent = cur && cur.started && !S.engine.paused ? 'Pause' : 'Play';
  }

  function buildRuler() {
    var r = $('ruler'), bars = S.man ? S.man.bars : 8;
    if (S.dom.ruler.length === bars) return;
    r.innerHTML = '';
    S.dom.ruler = [];
    for (var b = 0; b < bars; b++) {
      var c = el('span', 'bar', String(b + 1));
      r.appendChild(c);
      S.dom.ruler.push(c);
    }
  }

  function tileBadges(cur, i) {
    var m = cur.marks, out = [];
    if (S.mode === 'keep3') { if (m.best === i) out.push(['keep', 'Keep']); }
    else if (S.mode === 'ab') { if (m.best === i) out.push(['best', 'Pick']); }
    else if (S.mode === 'triage') { if (m.tier) out.push(['tier', TIER_WORDS[m.tier]]); }
    else if (S.mode === 'playlist') {
      if (m.mark === 'heart') out.push(['best', '♥']);
      if (m.mark === 'cross') out.push(['worst', '✗']);
    } else {
      if (m.best === i) out.push(['best', 'Best']);
      if (m.worst === i) out.push(['worst', 'Worst']);
    }
    if (m.tag && S.mode === 'bws4') {
      var neg = NEG_TAGS.indexOf(m.tag) >= 0;
      if ((neg && m.worst === i) || (!neg && m.best === i)) out.push(['tag', TAG_WORDS[m.tag]]);
    }
    return out;
  }

  function tileButtons(i) {
    switch (S.mode) {
      case 'shootout': case 'bws4':
        return [['best', i, 'Best ' + BEST_KEYS[i]], ['worst', i, 'Worst ' + WORST_KEYS[i]]];
      case 'keep3': return [['best', i, 'Keep ' + BEST_KEYS[i]]];
      case 'ab': return [['best', i, 'Pick ' + BEST_KEYS[i]]];
      case 'triage': return TIERS.map(function (t) { return ['tier', t, TIER_WORDS[t] + ' ' + TIER_KEYS[t]]; });
      case 'playlist': return [['best', 0, '♥ Q'], ['worst', 0, '✗ Z']];
    }
    return [];
  }

  function buildTiles(cur) {
    var box = $('tiles'), bars = S.man ? S.man.bars : 8;
    box.innerHTML = '';
    S.dom.tiles = [];
    S.dom.cells = [];
    if (!cur) return;
    for (var i = 0; i < cur.k; i++) {
      var t = el('div', 'tile');
      t.dataset.i = String(i);
      t.appendChild(el('div', 'num', String(i + 1)));
      var heard = el('div', 'heard');
      var cells = [];
      for (var b = 0; b < bars; b++) { var c = el('span', 'cell'); heard.appendChild(c); cells.push(c); }
      t.appendChild(heard);
      var badges = el('div', 'badges');
      tileBadges(cur, i).forEach(function (bd) { badges.appendChild(el('span', 'badge ' + bd[0], bd[1])); });
      t.appendChild(badges);
      var btns = el('div', 'tile-btns');
      tileButtons(i).forEach(function (b2) {
        var btn = el('button', null, b2[2]);
        btn.dataset.act = b2[0];
        btn.dataset.arg = String(b2[1]);
        btns.appendChild(btn);
      });
      t.appendChild(btns);
      t.appendChild(el('div', 'hint', cur.k > 1 ? 'press ' + (i + 1) + ' to listen' : ''));
      box.appendChild(t);
      S.dom.tiles.push(t);
      S.dom.cells.push(cells);
    }
  }

  function buildFlags(cur) {
    var box = $('flags'), spec = MODES[S.mode] || {}, m = cur ? cur.marks : Rules.newMarks();
    box.innerHTML = '';
    function chip(act, arg, label, on) {
      var b = el('button', 'chip' + (on ? ' on' : ''), label);
      b.dataset.act = act;
      if (arg !== null) b.dataset.arg = arg;
      box.appendChild(b);
    }
    if (spec.hatches) {
      chip('noBest', null, 'T no clear best', m.noBest);
      chip('noWorst', null, 'B no clear worst', m.noWorst);
    }
    if (spec.allBad) chip('allBad', null, 'G ' + spec.allBad, m.allBad);
    if (spec.tags) {
      NEG_TAGS.concat(POS_TAGS).forEach(function (tg) { chip('tag', tg, TAG_KEYS[tg] + ' ' + TAG_WORDS[tg], m.tag === tg); });
    }
  }

  // Cheap redraw, ten times a second: the bar ruler, heard shading, which clip is up.
  function renderLive() {
    var p = null;
    if (S.view === 'vote' && S.cur && S.cur.player) p = S.cur.player;
    var bar = -1, up = -1;
    if (p && p.started && !p.stopped) {
      var t = Math.max(0, p.now() - S.engine.latency());
      bar = p.barAt(t);
      up = p.upAt(t);
    }
    S.dom.ruler.forEach(function (c, b) { c.className = 'bar' + (b === bar ? ' now' : ''); });
    var min = S.cfg ? S.cfg.heard.min_bars : 4;
    for (var i = 0; i < S.dom.tiles.length; i++) {
      var tile = S.dom.tiles[i];
      var heardN = p ? p.heardCount(i) : 0;
      tile.className = 'tile' + (i === up ? ' up' : '') + (heardN >= min ? ' heard-ok' : '');
      var cells = S.dom.cells[i];
      for (var b2 = 0; b2 < cells.length; b2++) {
        var on = p ? p.heardBar(i, b2) : false;
        cells[b2].className = 'cell' + (on ? ' on' : '') + (i === up && b2 === bar ? ' now' : '');
      }
    }
    var st = $('status');
    if (st && (S.view === 'vote' || S.view === 'ended')) st.textContent = statusText(S.cur, up);
    if (S.view === 'audit') renderReplayLive();
  }

  function statusText(cur, up) {
    if (S.view === 'ended') return 'Stopped.';
    if (!cur) return '';
    if (cur.loadError) return cur.loadError;
    if (!cur.buffers) return 'Loading the clips...';
    if (cur.committing) return 'Saving...';
    if (!cur.started) return S.engine.held ? HELD_TEXT : 'Press Space to play.';
    if (S.engine.paused) return 'Paused. Press Space to go on.';
    var which = up >= 0 && cur.k > 1 ? 'Playing ' + (up + 1) : 'Playing';
    if (S.mode === 'playlist' && cur.player) {
      return which + ' (pass ' + Math.min(cur.round.plays || 3, cur.player.passesDone() + 1) + ' of ' + (cur.round.plays || 3) + ')';
    }
    return which + (cur.auto ? ' (auto-play: each clip in turn)' : '');
  }

  // -- clicks
  function onTileClick(ev) {
    var btn = ev.target.closest ? ev.target.closest('button') : null;
    if (btn) return onButtonClick(ev);     // Best / Worst / tier buttons inside a tile
    var tile = ev.target.closest ? ev.target.closest('.tile') : null;
    if (!tile || !S.cur) return;
    noteActive();
    try { dispatch(['listen', parseInt(tile.dataset.i, 10)]); } catch (err) { fatal(err); }
  }

  function onButtonClick(ev) {
    var btn = ev.target.closest ? ev.target.closest('button') : null;
    if (!btn || !btn.dataset.act) return;
    btn.blur();                            // keys go back to the page, not the button
    noteActive();
    var act = btn.dataset.act, arg = btn.dataset.arg;
    var a = [act];
    if (arg !== undefined) a.push(act === 'tier' || act === 'tag' ? arg : parseInt(arg, 10));
    try { dispatch(a); } catch (err) { fatal(err); }
  }

  // ------------------------------------------------------------------ the audit view (after voting)
  function showAudit(justFinished) {
    if (S.cur && S.cur.player) S.cur.player.stop();
    S.cur = null;
    S.view = 'audit';
    hideBanner();
    render();
    $('audit-summary').textContent = 'Loading the summary...';
    $('audit-list').innerHTML = '';
    return getJSON('/api/audit').then(function (data) {
      S.audit = data;
      renderAudit(data, justFinished);
    }, function (err) {
      if (err.status === 423) return goOn();       // a vote was taken back meanwhile
      throw err;
    }).catch(fatal);
  }

  function pct(x) { return x === null || x === undefined ? '-' : Math.round(x * 100) + '%'; }

  function line(box, text, cls) { box.appendChild(el('p', cls || null, text)); }

  function renderAudit(a, justFinished) {
    var box = $('audit-summary');
    box.innerHTML = '';
    $('audit-title').textContent = justFinished ? 'Batch done. Thank you!' : 'This batch is done';
    line(box, 'All ' + a.rounds + ' rounds have a vote' + (a.retracted ? ' (' + a.retracted + ' taken back and asked again)' : '') + '.');
    if (a.heard && a.heard.clips) {
      line(box, 'You heard ' + a.heard.heard + ' of ' + a.heard.clips + ' clips for at least ' + a.heard.min_bars + ' bars (' + pct(a.heard.rate) + ').');
    }
    if (a.anchors && a.anchors.rounds) {
      line(box, 'Broken copies caught: ' + a.anchors.caught + ' of ' + a.anchors.rounds + '.');
      (a.anchors.micro || []).forEach(function (g) {
        if (g.flag) line(box, 'In ' + g.session + ', part ' + g.micro + ', fewer than half of the broken copies were caught, so those votes may be set aside or checked again.', 'warn');
      });
    }
    if (a.repeats && a.repeats.rounds) {
      if (a.repeats.compared) {
        line(box, 'Repeated rounds: your two answers agreed on ' + a.repeats.agree + ' of ' + a.repeats.compared +
             ' comparisons (' + pct(a.repeats.p) + ', likely between ' + pct(a.repeats.ci95[0]) + ' and ' + pct(a.repeats.ci95[1]) + ').');
      }
      if (a.repeats.missing) line(box, a.repeats.missing + ' repeated round(s) have no first answer to compare with yet.');
    }
    (a.position_of_best || []).forEach(function (row) {
      var parts = row.counts.map(function (c, i) { return (i + 1) + ': ' + c; }).join(', ');
      var sway = row.p !== null && row.p < 0.05 ? 'position may have swayed your picks' : 'no sign that position mattered';
      line(box, 'Picks by position (' + parts + '): ' + sway + (row.chi2 !== null ? ' (chi-square ' + row.chi2.toFixed(2) + ', p ' + row.p.toFixed(2) + ').' : '.'));
    });
    if (a.time) {
      line(box, 'Time on this batch: ' + a.time.minutes + ' min' + (a.time.est_min ? ' (planned about ' + Math.round(a.time.est_min) + ')' : '') +
           '. Today so far: ' + a.time.today_minutes + ' of ' + a.time.daily_budget_minutes + ' min.');
    }
    renderModeTally(box, a);
    (a.problems || []).forEach(function (p) { line(box, 'Note: ' + p, 'warn'); });
    renderAuditRows();
  }

  function renderModeTally(box, a) {
    if (a.mode === 'shootout' && a.shootout) {
      var s = a.shootout;
      line(box, (s.category ? s.category.charAt(0).toUpperCase() + s.category.slice(1) : 'Options') + ': how often each was picked best and worst.');
      var tbl = el('table', 'tally'), head = el('tr');
      ['option', 'best', 'worst', 'best minus worst', 'times played'].forEach(function (h) { head.appendChild(el('th', null, h)); });
      tbl.appendChild(head);
      (s.options || []).forEach(function (o) {
        var tr = el('tr');
        [o.option, o.best, o.worst, o.net, o.shown].forEach(function (v) { tr.appendChild(el('td', null, String(v))); });
        tbl.appendChild(tr);
      });
      box.appendChild(tbl);
      line(box, '"No clear best" ' + s.no_best + ' times, "no clear worst" ' + s.no_worst + ', "all bad" ' + s.all_bad + '.');
      line(box, 'Tell Claude which 2 or 3 to keep; Claude writes them into gates/GATE_palette.md for your APPROVED line.');
    } else if (a.mode === 'bws4' && a.bws4) {
      var b = a.bws4, tags = Object.keys(b.tags || {}).map(function (k) { return (TAG_WORDS[k] || k) + ' ' + b.tags[k]; });
      line(box, '"No clear best" ' + b.no_best + ' times, "no clear worst" ' + b.no_worst + ', "all four bad" ' + b.all_bad + '.');
      if (tags.length) line(box, 'Tags: ' + tags.join(', ') + '.');
    } else if (a.mode === 'keep3' && a.keep3) {
      line(box, 'Kept a draft in ' + a.keep3.kept + ' rounds; none acceptable in ' + a.keep3.none_acceptable + '.');
    } else if (a.mode === 'triage' && a.triage) {
      var t = a.triage.tiers;
      line(box, 'Toss ' + t.toss + ', Fine ' + t.fine + ', Good ' + t.good + ', Great ' + t.great + '. Fine or better: ' + pct(a.triage.fine_or_better) + '.');
    } else if (a.mode === 'ab' && a.ab) {
      line(box, 'Picked the first ' + a.ab.picks[0] + ' times and the second ' + a.ab.picks[1] + ' times.');
    } else if (a.mode === 'playlist' && a.playlist) {
      var p = a.playlist;
      line(box, 'Played ' + p.played + ' loops: ' + p.hearts + ' hearts, ' + p.crosses + ' x (heart rate ' + pct(p.heart_rate) + ').');
    }
  }

  function picksText(row) {
    var p = row.picks || {};
    if (!row.voted) return 'no vote';
    switch (S.mode) {
      case 'bws4': case 'shootout':
        return [p.all_bad ? (S.mode === 'bws4' ? 'all four bad' : 'all bad') : (p.no_best ? 'no clear best' : 'best ' + p.best),
                p.no_worst ? 'no clear worst' : (p.worst ? 'worst ' + p.worst : null),
                p.tag ? 'tag ' + (TAG_WORDS[p.tag] || p.tag) : null].filter(Boolean).join(', ');
      case 'keep3': return p.none_acceptable ? 'none acceptable' : 'kept ' + p.kept;
      case 'triage': return TIER_WORDS[p.tier] || '-';
      case 'ab': return 'picked ' + p.pick;
      case 'playlist': return p.mark === 'heart' ? 'heart' : p.mark === 'cross' ? 'x' : 'no mark';
    }
    return '';
  }

  function renderAuditRows() {
    var box = $('audit-list'), a = S.audit;
    box.innerHTML = '';
    if (!a) return;
    box.appendChild(el('h2', null, 'Rounds'));
    box.appendChild(el('p', 'small', 'Listen again with the button, then 1-4 to switch, 5 from bar 1, Space to pause, Esc to stop. "Take back" asks that round again.'));
    a.list.forEach(function (row, idx) {
      var r = el('div', 'row' + (S.replay && S.replay.idx === idx ? ' playing' : ''));
      r.appendChild(el('span', 'rn', 'Round ' + row.number));
      r.appendChild(el('span', 'rp', picksText(row)));
      var heardCells = el('span', 'rh');
      row.clips.forEach(function (c) {
        var mini = el('span', 'mini');
        String(c.heard_bars || '').split('').forEach(function (ch) { mini.appendChild(el('i', ch === '1' ? 'on' : null)); });
        heardCells.appendChild(mini);
      });
      r.appendChild(heardCells);
      var listenBtn = el('button', null, S.replay && S.replay.idx === idx ? 'Stop' : 'Listen');
      listenBtn.dataset.act = S.replay && S.replay.idx === idx ? 'stop' : 'listen';
      listenBtn.dataset.idx = String(idx);
      r.appendChild(listenBtn);
      if (row.event_id) {
        var back = el('button', null, 'Take back');
        back.dataset.act = 'retract';
        back.dataset.idx = String(idx);
        r.appendChild(back);
      }
      box.appendChild(r);
    });
  }

  function renderReplayLive() {
    var rep = S.replay, ind = $('replay-status');
    if (!ind) return;
    if (!rep || !rep.player || !rep.player.started) { ind.textContent = ''; return; }
    var p = rep.player, t = Math.max(0, p.now() - S.engine.latency());
    ind.textContent = 'Round ' + rep.row.number + ': playing ' + (p.upAt(t) + 1) + ' of ' + rep.k + ', bar ' + (p.barAt(t) + 1) +
                      (S.engine.paused ? ' (paused)' : '');
  }

  function onAuditClick(ev) {
    var btn = ev.target.closest ? ev.target.closest('button') : null;
    if (!btn) return;
    btn.blur();
    var idx = parseInt(btn.dataset.idx, 10);
    if (btn.dataset.act === 'listen') auditListen(idx).catch(fatal);
    else if (btn.dataset.act === 'stop') { stopReplay(); renderAuditRows(); }
    else if (btn.dataset.act === 'retract') auditRetract(idx);
  }

  function auditListen(idx) {
    stopReplay();
    var row = S.audit.list[idx], round = S.rounds.get(row.round);
    if (!round) return Promise.resolve();
    var rep = { idx: idx, row: row, round: round, k: round.clips.length, player: null };
    S.replay = rep;
    S.engine.keepOnly(round.clips.map(clipUrl));       // keep only this round's decoded audio
    renderAuditRows();
    return Promise.all(round.clips.map(function (c) { return S.engine.load(clipUrl(c)); })).then(function (bufs) {
      if (S.replay !== rep) return null;
      return S.engine.unlock().then(function () {
        if (S.replay !== rep) return;
        rep.player = new global.LofiAudio.RoundPlayer(S.engine, bufs, { bars: S.man.bars, barFraction: S.cfg.heard.bar_fraction });
        rep.player.start(0);
      });
    });
  }

  function stopReplay() {
    if (S.replay && S.replay.player) S.replay.player.stop();
    S.replay = null;
  }

  function auditKey(act) {
    var rep = S.replay, kind = act[0];
    if (kind === 'end') { stopReplay(); renderAuditRows(); return; }
    if (!rep || !rep.player) return;
    var p = rep.player;
    if (kind === 'listen' && act[1] < rep.k) {
      p.change(function (t) { p.cancelFrom(t); p.switchTo(act[1], t); });
    } else if (kind === 'replay') {
      S.engine.unlock();
      p.restart(p.upAt());
    } else if (kind === 'play') {
      if (S.engine.paused) { S.engine.resume(); hideBanner('device'); } else S.engine.pause();
    }
  }

  function auditRetract(idx) {
    var row = S.audit.list[idx];
    if (!row || !row.event_id) return;
    if (!global.confirm('Take back your vote on round ' + row.number + '? It will be asked again, and this summary closes until it has a vote.')) return;
    stopReplay();
    S.view = 'vote';
    retract(row.event_id, 'audit').catch(fatal);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})(typeof window !== 'undefined' ? window : this);
