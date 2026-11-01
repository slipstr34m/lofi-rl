# Rating page

The rating page is where you vote. It plays the loops of one batch, you pick with the keyboard, and every
answer is saved on this laptop the moment you give it. It runs only on this laptop and nobody else can
reach it. It shows the scene text and nothing else about a loop: no scores, no model guesses, ever.

Part 1 is for you (how to use it). Part 2 is for Claude and the batch jobs (file formats).

## Part 1: using it

### Start

```bash
make rate                                   # the oldest batch that still needs votes
make rate BATCH=inbox/2026-11-03/b1103-keys # a given batch
```

Your browser opens `http://127.0.0.1:8596`. If it doesn't, open that address yourself. Press **Space** to
start the first round. Ctrl-C in the terminal stops the server; your place is saved.

At the bottom of the page, "Listening on" says which headphones or speakers you are using (AirPods by
default). Change it if you switch; it is saved with every answer.

### A round

- The big text is the scene. Below it, the bar ruler shows which of the 8 bars is playing.
- Each tile is one loop. All loops in a round play in step, so when you switch with 1–4 you land on the same
  bar of the other loop.
- The page plays each loop for 8 bars in turn, 1 → 2 → 3 → 4. You don't have to wait: you can pick at any
  time. If you mark the loop that is playing, the page moves on to the next one you haven't heard yet.
- The small squares on a tile fill in as you hear its bars. A bar counts once you've heard most of it (75 %).
  A loop counts as heard once 1 bar has played (a tick appears on the tile), and bars heard before an undo or a
  reload still count. Only picks involving a loop you never played are set aside, so skipping a bad loop after a
  bar or two still counts (D-013 as amended, 2026-10-31).
- Press **Enter** to save. The next round starts by itself.

### The keys

```
1 2 3 4 | 5  listen k (bar-synced)   | 5 replay from bar 1
Q W E R | T  best = k (ab: Q/W)      | T no clear best
A S D F | G  Toss/Fine/Good/Great    | G all four bad / none acceptable
Z X C V | B  worst = k               | B no clear worst
J fake  K no groove  L sour  ; boring  |  U nodding  I sax line  O chords
(negative tag → worst, positive → best) · Space play · Enter commit · Bksp undo · Esc end
```

The keys go by position, so they work the same on any keyboard layout. Every key also has a button on the page.

### What each kind of batch asks

| Kind | You hear | You answer |
|---|---|---|
| **shootout** | 2–4 versions of the same loop, one thing changed (keys, drums and bass, vamp, sax...) | the best (Q W E R) and the worst (Z X C V). T or B if there is no clear best or worst, G if all are bad. |
| **keep3** | 3 drafts | the one to keep (Q W E), or G if none is acceptable |
| **bws4** | 4 loops for one scene | the best and the worst. T, B, G as above. One optional reason tag (below). |
| **triage** | 1 loop (it plays 8 bars, then bars 1–2 again so you hear it loop around) | A **Toss** (a dealbreaker), S **Fine** (not embarrassing), D **Good** (I'd play it while working), F **Great** (can't stop nodding) |
| **ab** | 2 loops | the one you like more: Q for 1, W for 2. There is no "can't decide". |
| **playlist** | loops one after another, each 2–4 times | nothing, unless one stands out: Q for a heart, Z for an x. Enter skips to the next loop. |

**Reason tags** (best-and-worst rounds only, at most one): J fake, K no groove, L sour, ; boring go with
your worst pick; U nodding, I sax line, O chords go with your best pick. Press the same key again to drop it.

### Changing your mind

- **Backspace** undoes your last key in this round, one step at a time.
- After saving, press **Backspace twice** to take back the vote you just gave. That round comes back and you
  answer it again. Nothing is erased: the log keeps both, and the old answer stops counting.
- T, B, G and the tags switch off if you press them again.

### Stopping, breaks and coming back

- **Esc** stops. The round you were in moves to the end of the batch, and you start fresh next time. Closing the
  tab does the same.
- Coming back (Space, or `make rate` later) carries on where you stopped. Within 10 minutes it counts as the
  same short listening stretch; after a longer gap a new stretch starts. A loop left playing is not a gap.
- After 25 minutes of listening the page suggests a short break, counted from the start of the stretch even if
  the page reloaded in between. Esc is always safe.
- If your audio output changes (AirPods disconnect, say), playback pauses so nothing plays out loud, and nothing
  plays again until you press Space: not the next round after Enter, not 5 or 1–4. Check "Listening on", then
  press Space. The same happens if the system stops the audio (another app takes it over, say).

### Checks you won't see

A few rounds repeat an earlier round a couple of days later (in a new order), and a few contain a deliberately
broken copy of a loop. That is how the project measures how consistent the votes are. You are never told
which rounds those are. Just answer as usual: a broken copy is simply one to mark worst (or Toss).

### When a batch is done

The page switches to a summary of your own answers, in counts only: how many loops you heard, how many broken
copies you caught, how often you agreed with yourself on repeats, whether position seemed to sway your picks,
and the time spent against your hour a day. A shootout also shows how often each option was picked best and
worst; tell Claude which 2–3 to keep, and Claude writes them into `gates/GATE_palette.md` for your
`APPROVED by Shresth Rana <date>` line.

Each round has **Listen** (hear it again; 1–4 switch, 5 from bar 1, Space pauses, Esc stops) and **Take back**
(that round is asked again, and the summary closes until it has a new answer).

### Where votes go

Votes are private. They are written only to `data/votes/` on this laptop: one file per batch plus a small
`state.json` that remembers your place. `data/` is never committed to the public repo; `make backup-votes`
copies the votes to the private votes repo and to the cluster. If `data/votes/` doesn't exist yet, `make rate`
creates it and says it is not the private repo yet; then ask Claude to connect it before the first backup.

---

## Part 2: formats (for Claude and the batch jobs)

Code: `src/lofi_rl/ui/{serve,manifest,votes,audit}.py` (stdlib plus PyYAML; no model, numpy or audio library is
imported) and `src/lofi_rl/ui/static/{index.html,app.js,audio.js}` (vanilla JS, no build step). Settings:
`configs/rating.yaml`. Design: `docs/design/taste_pipeline.md` §1–§3.

### Folders and environment

| | Default | Override |
|---|---|---|
| batches | `inbox/<date>/<batch>/` | `LOFI_INBOX` |
| votes | `data/votes/` | `LOFI_VOTES` (inside the repo, only under `data/`; never inside a batch folder; checked by folder identity, so another spelling of the same path, such as a different case or `/System/Volumes/Data/...`, counts as the same folder) |
| config | `configs/rating.yaml` | `--config` |

`python -m lofi_rl.ui.serve [batch] [--port N] [--no-browser]` is what `make rate` runs (`LOFI_NO_BROWSER=1`
also skips the browser). Without a batch it serves the oldest `inbox/*/*/manifest.json` whose rounds are not all
voted. Tests and demos always pass temp folders; nothing under `data/` or `inbox/` is touched by them.

### A batch folder

```
inbox/2026-11-03/b1103-keys/
  manifest.json            public: the page may see a trimmed copy of it
  manifest.hidden.json     never served: what each clip really is
  k3f9a1c2d7e.flac ...     one file per clip appearance, opaque names, flat (no subfolders)
```

`manifest.json` (v1). Unknown keys are errors, so nothing private can slip in by accident:

```json
{"v": 1, "batch": "b1103-keys", "mode": "shootout",
 "scene": "city at night, rain on the window", "note": "Which keys do you like best?",
 "bpm": 82, "bars": 8, "sr": 44100, "created": "2026-11-03T08:00:00-08:00", "est_min": 15,
 "rounds": [{"id": "r3f9c2a1b", "scene": "...", "bpm": 82, "plays": 3,
             "clips": [{"id": "k3f9a1c2d7e", "file": "k3f9a1c2d7e.flac", "sha256": "<64 hex>"}]}]}
```

- `batch` matches `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` and equals the folder name. `mode` is one of shootout
  (2–4 clips a round), keep3 (2–3), bws4 (4), triage (1), ab (2), playlist (1).
- `scene` and `bpm` may be set per round (triage, bws4 and playlist rounds come from different briefs). Every
  round needs both. `bars` is required (8 for LoopScript). `sr`, `created`, `est_min`, `note` are optional.
  `plays` (1–16) is for playlist rounds only; without it the server picks 2–4 from the round id.
- Round ids match `r[0-9a-f]{6,32}`; clip ids match `[a-z][0-9a-f]{8,32}` and are unique in the batch (a clip
  shown twice gets two ids); files match `[a-z][0-9a-f]{8,32}\.(flac|wav)`, are unique, and are plain files in
  the folder (no symlinks).
- On load every file is checked: it parses (FLAC STREAMINFO or WAV header, stdlib only), its `sha256` matches
  if given, the batch has one sample rate, a round's clips are equally long, and each clip is one pass of the
  loop: `bars × 240 / bpm` seconds within 2 %. All problems are reported together.
- The page gets only `v, batch, mode, bars, sr, note, est_min` and per round `id, scene, plays, clips[id, file]`.
  No bpm, no hashes: the scene is the only brief information it shows.

`manifest.hidden.json` (v1):

```json
{"v": 1, "batch": "b1103-keys", "category": "keys",
 "rounds": {"r3f9c2a1b": {"kind": "anchor", "of": null, "anchor_clip": "k3f9a1c2d7e", "brief_id": "b0412",
                          "category": "keys",
                          "clips": {"k3f9a1c2d7e": {"clip": "c1a2b3c4d5e6f7a8b", "option": "rhodes",
                                                    "what": "broken copy", "broken_of": "k5e6..."}}}}}
```

- Every public round has an entry, and its `clips` cover exactly the round's clips. `clip` is the real clip id
  (`c` + hex, taste_pipeline §2); the other clip fields are free plain values (option, what, source, step,
  broken_of, ...). A real clip appears at most once per round.
- `kind`: `none`, `repeat` (needs `of`, the round id of the first showing, which must differ from this round's
  id), `anchor` (needs `anchor_clip`, the opaque id of the broken copy), or `heldout`.
- Shootouts need a `category` (batch or round) and an `option` per clip.
- Leak check: no round id, clip id or file name may contain 10 hex digits in a row of a real clip id, or any
  word of 3+ letters from the hidden values or brief id (short all-hex words are skipped, since random ids
  contain them by chance). The note may name the category but not an option.

Synthetic batches for tests and browser checks (sine or noise bursts, never music, never committed):

```bash
python -m lofi_rl.ui.manifest demo <dir> --mode bws4 [--rounds N] [--bpm 90] [--bars 8] [--sr 44100] [--format flac|wav]
python -m lofi_rl.ui.manifest check <dir>
```

In a demo, round 2 is an anchor and the last round repeats round 1 with positions reversed.

### What the page sends (`POST /api/event`)

JSON with opaque ids only. Unknown fields are refused (400); a vote for a round that already has one is a
conflict (409). Optional `event_id` (`e` + 12–32 hex) makes retries safe: the same id is written once.

| type | fields |
|---|---|
| vote (`type` = the batch mode) | `round`, `clips[{clip, pos, heard_bars, heard_ms}]` in position order, `last_heard`, `switches`, `play_order` (positions), `autoplay_done`, `t_ms{best, worst, no_best, no_worst, all_bad, tag, tier, pick, mark, commit}`, `device`, `audio{sr, latency_ms, base_latency_ms}`, plus the picks below |
| picks: bws4 | `best`, `worst`, `no_best`, `no_worst`, `all_bad`, `tag` (one of fake, no_groove, sour, boring → worst; nodding, sax_line, chords → best). Complete when `all_bad`, or a best (or `no_best`) and a worst (or `no_worst`). |
| picks: shootout | as bws4, without `tag` |
| picks: keep3 | `best` (the kept draft) or `all_bad` (none acceptable), exactly one |
| picks: triage | `tier`: toss, fine, good or great |
| picks: ab | `pick` (forced) |
| picks: playlist | `mark`: heart, cross or none; `plays` (whole passes heard) |
| `abandon` | `round`, `why` (esc, pagehide, undo, no_commit, error, other), optional listening fields |
| `retract` | `target` (the event_id of a standing vote), optional `why` |
| `session` | `what` (start, round_start, pause, resume, devicechange, nudge, playing, end), optional `round`, `detail`, `device`, `audio` |

`heard_bars` is a string of `bars` characters, bar 1 first, `1` where the clip's gain was up for at least
`heard.bar_fraction` of that bar. `heard_ms` is the time its gain was up while playing. `t_ms` holds the time
to each mark the vote holds: the server drops times of marks that were set and then cleared (T after a best, a
tag pressed twice). `detail` and `why` are plain text: no control characters, valid Unicode (a lone surrogate
escape such as `\ud800` gets 400; so does such text in a manifest, when the batch loads).

The page sends `session` `playing` after half of `micro_gap_minutes` (5 minutes) of uninterrupted playback with no
other event, so a loop left playing is not a gap. A `pause` with `detail` `system: <state>` means the system
stopped the audio (Safari's `interrupted`, say); the page then holds playback until Space, as after a
`devicechange`.

### What the server writes

One line per event in `$LOFI_VOTES/<batch>.jsonl` (mode 600), appended and fsynced before the reply. The server
adds its own clock, the session and micro-session, the real clip ids, the round's position in the presentation
and the hidden labels, so a vote stays meaningful if the inbox is purged:

```json
{"v":1,"type":"bws4","event_id":"e…","ts":"2026-11-10T21:14:03-08:00","session":"s1110a","micro":2,
 "batch":"b1110-01","round":"r…","index":12,"brief_id":"…","scene":"…",
 "clips":[{"clip":"c…","oid":"k…","pos":1,"heard_bars":"11110000","heard_ms":11800}],
 "best":"c…","worst":"c…","no_best":false,"no_worst":false,"all_bad":false,"tag":{"name":"sax_line","clip":"c…"},
 "last_heard":"c…","switches":3,"play_order":[1,2,3,4],"autoplay_done":true,
 "t_ms":{"best":41000,"worst":52000,"commit":55000},"device":"airpods","audio":{"sr":44100,"latency_ms":120},
 "ui":"<sha12 of the page files>","hidden":{"kind":"none","of":null}}
```

- triage adds `tier`; ab writes `left`, `right`, `pick`; keep3 `best`, `all_bad`; playlist `mark`, `plays`;
  shootout adds `category` and `options` (each clip's option, in position order); anchors add
  `hidden.anchor` (the broken copy's real id).
- `retract` lines carry `target` and its `round`; `abandon` lines carry `why`; `session` lines carry `what`.
- **Session ids**: `s` + month and day + a letter for each new session that day. A server start after a gap of
  10 minutes or more begins a new session; within a session, a gap of 10 minutes or more begins the next
  micro-session (`micro` counts from 1). Gaps are measured between events from any server on the same votes
  folder (the session lives in `state.json`), so an id is never used for two sessions.
- **The queue** is derived from the log every time the server starts: rounds in manifest order, an abandoned
  round moves to the end, a retracted vote reopens its round. A batch is done when every round has a standing
  vote. A `round_start` with no vote or abandon after it (the tab crashed) is logged as `abandon`
  (`no_commit`) at the next page load.
- `$LOFI_VOTES/state.json` (written atomically) holds the session (with `micro_start` and the micro-session's
  `nudges`) and each batch's cursor (`current, settled, total, done, events`). If it disagrees with the log, the
  log wins and the server says so. If it cannot be written (a full disk), the event is still saved and the
  server notes it in the terminal and the audit.
- **Batch done**: when the last round gets its vote, `batches.<batch>.done` in `state.json` turns true and the
  server prints `rate: batch_done <batch> (<n> rounds)`. Either is the signal for the push-notification loop
  (taste_pipeline §1). A retract afterwards reopens the batch.
- The log is locked while a server runs, so two servers never write one batch. Two servers may share one votes
  folder (one batch each, each on its own port): each logs its event and rewrites `state.json` (its own batch's
  entry merged into what the file holds) under an exclusive lock on the folder, so neither reverts the other's
  cursor or `done` flag.
- A write that fails part-way (a full disk, a quota) is cut back to where the log ended, so a retry with the same
  `event_id` lands on a line of its own. A session start and the `abandon` it logs for an open round are written
  together or not at all. A torn last line from a crash (or a failed cut-back) is kept apart and skipped when read.

### Other endpoints

- `GET /api/state`: the cursor (`current`, `upcoming`, `index`, `total`, `settled`, `done`, `last_vote`,
  `session`, `micro`, `micro_start`, `micro_nudges`). Never a hidden label. A reloaded page counts toward the
  break nudge from `micro_start`.
- `GET /api/config`: the `configs/rating.yaml` values the page uses, plus the page hash.
- `GET /api/audit`: 423 until every round has a standing vote, then the aggregates: counts, heard rate,
  anchors caught (with micro-sessions that caught fewer than half of 2 or more anchors flagged), repeat agreement
  on implied pairs with a 95 % Wilson interval (the first showing is looked up in every log in `$LOFI_VOTES`),
  best position counts with a chi-square test, minutes against `daily_budget_minutes`, the per-mode tallies, and
  per-round rows with positions and picks only (no real ids, no hidden labels). The shootout's best and worst
  counts take each block once: hidden repeats are left out of them (they count only for agreement).
- Only `127.0.0.1` is bound. Requests whose `Host` is not `127.0.0.1:<port>` or `localhost:<port>` get 421; a
  POST from another `Origin` gets 403; only listed clip files are served; dot segments, encoded or not, get 400.

### Settings (`configs/rating.yaml`)

`port` 8596 · `heard.bar_fraction` 0.75 · `heard.min_bars` 1 · `heard.toss_min_bars` 1 ·
`autoplay.bars_per_clip` 8 · `autoplay.wrap_bars` 2 (single-clip rounds) · `autoplay.start_next_round` ·
`switch_ramp_ms` 5 · `nudge_minutes` 25 · `micro_gap_minutes` 10 · `playlist_plays` [2, 4] ·
`daily_budget_minutes` 60 · `devices`, `default_device`.

### Audio, for the record

All clips of a round are decoded first, then started at one AudioContext time with `loop = true`. A switch
moves only gains: the old clip's gain and the new one's follow complementary 5 ms raised-cosine ramps, centred on
the switch time; there is no longer crossfade. Auto-play switches land on bar lines. The context runs at the
batch's sample rate when the browser allows it, so the loop stays sample-exact. A `devicechange` pauses playback
and holds it until Space. If the system stops the audio, the output is muted at once and held the same way; if the
browser later runs the context again by itself, the page stops it again, so nothing plays that is not counted.
Heard bars are kept as one flag per bar (not a 32-bit mask), so loops of up to 64 bars are counted correctly.
