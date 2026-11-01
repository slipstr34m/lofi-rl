# DRAFTS.md: the keep3 drafts

Three writers each draft one loop for each of 10 briefs; Shresth hears each brief's three drafts in one keep3 round
and keeps one, or says "none acceptable" (docs/design/taste_pipeline.md §8 "30 drafts → keep 10"; docs/TASTE.md,
Labeling: "Claude drafts 30 loops and he keeps 10; the rejects become labeled negatives"; docs/SHOOTOUTS.md §1 step 3).
A kept loop is labelled at least Fine and beats each of its two rejects; "none acceptable" labels all three Toss and
asks for a redraft of that brief (§6). The drafts play the Day-2 picks (gates/GATE_palette.md, Shresth 2026-11-01:
"Yes, use these"), with no rain bed yet: Day 3 decides it.

- **Code:** `src/lofi_rl/drafts/` (`briefs`, `personas`, `palette`, `check`, `render`, `batch`, `job`), the job
  `jobs/drafts.sbatch` (`lofi-drafts`), tests `tests/test_drafts_*.py`.
- **Files:** `drafts/v1/briefs.yaml` (the briefs), `drafts/v1/<brief>_<persona>.ls` (the 30 drafts, written next),
  `palettes/draft_day2.yaml` (the draft palette).
- **Sound claims:** none. A draft that passes every check below is only a draft; Shresth's ear decides.

## 1. The briefs

`drafts/v1/briefs.yaml`, written by `python -m lofi_rl.drafts.briefs` (seed 1105, the keep3 render date) and checked by
`python -m lofi_rl.drafts.briefs --check`. The scene is always city at night (TASTE.md, Briefs).

| id | energy | mood (mode) | BPM | key |
|---|---|---|---|---|
| d01 | low | melancholy (aeolian) | 81 | Eb aeolian |
| d02 | low | wistful (dorian) | 85 | Bb dorian |
| d03 | low | hopeful (ionian) | 71 | Db ionian |
| d04 | mid | melancholy (aeolian) | 89 | B aeolian |
| d05 | mid | wistful (dorian) | 82 | A dorian |
| d06 | mid | hopeful (ionian) | 90 | E ionian |
| d07 | high | melancholy (aeolian) | 86 | Ab aeolian |
| d08 | high | wistful (dorian) | 77 | D dorian |
| d09 | high | hopeful (ionian) | 76 | Gb ionian |
| d10 | low | wistful (dorian) | 66 | C dorian |

**The rule** (`lofi_rl.drafts.briefs.RULE`; every draw a keyed SplitMix64 hash
h(field, i) = splitmix64(splitmix64(splitmix64(seed) ⊕ field) ⊕ i)):
- **Cells:** d01–d09 are the 9 energy × mood cells in order; d10 is one more, drawn uniformly (low/wistful came up).
- **Tempo:** a stratified draw of 10 from the project's own brief sampler weights (`lofi_rl.brief.TEMPO_WEIGHTS`:
  60–95 BPM, each tempo in 75–90 four times as likely as one outside it, TASTE.md Tempo/feel). Stratum k of the 84
  weight units gives one tempo, and a keyed permutation deals them out. So the 10 spread over the distribution as its
  tenths do: here 66–90 BPM, 8 of 10 in 75–90.
- **Tonic:** the first 10 of the 12 pitch classes in a keyed order: each tonic equally likely (TASTE.md, Claude's
  defaults: 12 tonics, uniform), and 10 different keys (TASTE.md Variation: different briefs, different songs).

**What the writers see:** the scene and the two knobs, energy and mood, as the policy does (instrument.md §1). Tempo,
tonic and mode are the compiler's; the writers write degrees, numerals and grids.

## 2. The writers

Each writer is one persona below and drafts one loop for every brief: `drafts/v1/<brief>_<persona>.ls`, e.g.
`d04_horn.ls`. The personas differ in what they lead with, never in what they may write. Everything below traces to a
row of docs/TASTE.md (quoted in *italics*) or to a Claude's default it lists.

### 2.1 House rules (all three writers)

- **The language.** LoopScript only (docs/DSL.md is the grammar), in the draft palette's names (§4). No prose, no
  comments, no code fence, no `sax:` line (the palette's one sax plays every loop, DSL.md §4). The text must pass the
  gates with no repair at all, free ones included (§5): spell `span`, write every grid with exactly 16 cells.
- **Harmony.** A 2-chord vamp; every chord is at least a 7th and 9ths are common (*Harmony*). Mood picks the mode
  (*Briefs*): the vamp's mood signature must favour the brief's mode by 2 or more (§5). Borrowed chords are allowed
  as long as it does (*Harmony*, Claude's default). The vamp ranks of §4 are seed data you may lean on, never rules.
- **Drums and bass carry it** (*Job: head-nod: drums and bass carry it*). Aim, in every bar, for a non-ghost kick on
  step 1, a non-ghost snare or rim on steps 5 and 13 and at least 4 hat or shaker hits (the guided verifiers,
  TASTE.md "Guided, then faded"; theory in the validator's table shows how close a draft is); a mid-loop drop-out may
  break them on purpose (Claude's default: welcome, but not in bar 8). The bass holds the roots (*Bass*): you write
  only its rhythm, and most bass strikes should land on kicks (60 % or more), or write `bass: lock`.
- **Energy** (*Energy: more energy means busier drums, harder accents, busier keys and a busier bass rhythm*). The
  validator computes E = .35·min(1, H/192) + .15·(X share of the non-ghost hits) + .25·min(1, comp strikes/48) +
  .25·min(1, bass strikes/48), H being the drum hits of all six lanes over 8 bars (a ghost counts half). The brief's
  band is required: low below 0.38, mid 0.38 to 0.59, high from 0.59 (provisional bands, VERIFIERS.md §7 item 9).
  The reference loops sit at 0.29–0.30 (low), 0.46–0.48 (mid) and 0.70–0.72 (high); X accents rise with energy.
  The vamp's `span` is no energy cue (the row names drums, accents, keys and bass): pick it for the vamp and the line
  at any energy.
- **Something new every bar** (*Variation*): with per-bar edits, no bar repeats an earlier one in chord, drums, bass
  and comp together. Bar 5 plays bar 1's chord under the answer's opening, which restates the call's, so give bar 5
  more than a ghost or a level (a ghost plays at 0.32 of an accent): a hit, a strike or a push. **No fill at the
  seam** (*Drums*): bar 8 keeps the groove: at most 2 non-ghost snare or rim hits in its steps 9–16, at most 6 drum
  cells changed from the base bar, the backbeat kept, and enough hats. Drop-outs are welcome mid-loop but not in bar 8
  (TASTE.md, Claude's defaults).
- **The sax** (*Sax*; one alto, the palette's lead): *a few long notes* (2–4 notes of 8 steps or more over the whole
  line); *phrase in bars 1–4, answer in bars 5–8; each lands on a downbeat* (A and B each open on step 1 with a tone
  of that bar's chord); *the answer hangs open* (B ends off degree 1); the answer *starts like the call, ends
  differently* (Claude's default): on another note than the call's last, in any octave; rests in each phrase (12–60 %
  of its 64 steps, one rest of 2 steps or more), and *up front like a vocal*, never more than 3 bars (48 steps)
  without one, the call's end running into the answer included. *Blue notes are welcome*: the ♭3, ♭5 (`#4` or `b5`)
  and ♭7 above the tonic are always free. Each phrase is exactly 64 steps.
- **The sax with the keys** (*Dealbreakers: sour notes*): no sax note sounds a beat (4 steps) or more a semitone or a
  minor 9th from a keys or bass tone while the keys leave its own pitch class out: the root over a voiced maj7, the 5th
  over a voiced #11, a held blue #4 over a voiced 11. The compiler keeps a minor 9th out of the keys voicing itself
  (theory/voicing.py) but never reads the line, and the theory terms count every mode tone as free, so only the
  validator's `rubs` column (§5) shows one: it should read 0. A long note on a tone the voicing plays never rubs.
- **The sax guardrail** (*Soft penalties on busy, wide or chromatic lines that never fade*): at most 16 notes a phrase,
  a range of at most 19 semitones over the line, and at most 1 note in 5 outside the mode, the blue notes and the
  chord sounding (a short one that steps to a free note is fine). Keep the line where the alto is comfortable: the
  compiler puts the tonic in its home band (MIDI 57–68), shifts each phrase by an octave or not to keep the most notes
  in 55–77, and folds a note still outside 49–80 by octaves, which is a repair and fails the draft (COMPILER.md §3.2).
- **Never written:** tempo, key, mode, swing, effects, levels, the rain (*Mix*, DSL.md §5 forbidden words). Swing is
  the palette's 0.55 and humanize is off (Day 2).
- **Different songs** (*Variation: different briefs should give different songs*): no draft copies a prompt example,
  a shootout reference loop or another draft (5-gram Jaccard below 0.6, §5). Across your 10 drafts, use each keys
  instrument and both kits, and more than one voicing style (Claude's default: the votes, not the writer, decide
  between the survivors, so none should go unheard).
- **Run the validator** on your drafts (§5) until every line says `ok`.

### 2.2 Card 1: `groove` (groove first)

*Leads with the drums and the bass rhythm, then the vamp and the keys, then the sax.*

1. **The pocket first** (*Job*; *Great: can't stop nodding*; *Dealbreaker: no groove*). Write kick, the backbeat lane
   (`snare` or `rim`), hats and maybe shaker or open hat, then the bass rhythm against the kick (`bass: lock`, or a
   grid whose strikes sit on the kicks with a push or two). Use the three levels the grids have (TASTE.md, Claude's
   defaults: 16-step grids with X/x/g levels): ghosts (`g`) between the main hits, accents (`X`) rising with energy.
2. **Energy lives in the groove** (*Energy*): low is sparse kick and hats with few accents; high is busier kick and
   hats, more ghosts, more `X`, a busier bass rhythm. The keys follow.
3. **Bar by bar** (*Variation*; *Drums*): most per-bar edits are on drum lanes and the bass; bar 8 keeps the groove.
4. **Then the harmony**: a vamp that fits the mood; comp strikes that sit between the kicks and leave the backbeat
   clear.
5. **Then the sax**: simple and spacious, a phrase and its answer over the pocket, the house rules' long notes and
   rests.

### 2.3 Card 2: `horn` (sax line first)

*Leads with the sax phrase and its answer, then a band that leaves the line room.*

1. **The line first** (*Sax*; *Great: a sax line he'd replay*). Write A, then B: a few long notes, the phrase landing
   on the downbeat of bar 1 and the answer on bar 5, the answer starting like the call and ending differently, open
   (off degree 1). Blue notes as color (*Blue notes are welcome; long-note color is left to votes*).
2. **Up front like a vocal** (*Sax*): rests that let it breathe; the guardrail's bands (*Sax guardrail*).
3. **Then the vamp under it**: chords whose tones hold the long notes, or a long note that leans on a chord tone of
   the next chord.
4. **The keys under a held note** are left to votes (*Keys: what the keys do under a held sax note is left to votes*):
   across your drafts, sometimes hold the chord, sometimes rest, sometimes a short stab, so the votes can tell them
   apart.
5. **Then the groove**: steady drums and bass that carry the head-nod under the line (*Job*), at the brief's energy.

### 2.4 Card 3: `chords` (chords and keys first)

*Leads with the vamp, the keys instrument, the voicing and the comp, then the groove, then the sax.*

1. **The chords first** (*Harmony*; *Great: chords that give a feeling*; *Dealbreaker: sour notes*). Pick the vamp
   from the mood (the ranks of §4 are seed data), the span (`span 2` plays each chord two bars, `span 1` alternates
   them), the keys instrument and the voicing style (rootless, shell, spread or quartal: *the model picks the keys
   voicing style*, Claude's default).
2. **The comp rhythm**: strikes and holds (`-`) that move with the chord changes; busier comp at higher energy
   (*Energy: busier keys*).
3. **Then the groove** that supports the comp (*Job*: drums and bass still carry it), with per-bar edits.
4. **Then the sax** drawn from the chord tones: each phrase opens on a tone of its bar's chord, long notes on chord
   tones or blue notes.

## 3. The draft palette

`palettes/draft_day2.yaml`, id `draft_day2`: the Day-2 picks as one palette, under the plain-word names the drafts
write (instrument.md §6). Built by `python -m lofi_rl.drafts.palette build` with the shootout palette generator's
mechanism (SHOOTOUTS.md §7.8, CD-51) from `shootouts/plan.yaml`, `shootouts/palettes/base.yaml`, the Virtuosity
importer's outputs under `$LOFI_ASSETS` and the `levels.json` of `lofi-shootout` 13635646 (job 1) and 13651610
(job 2); never edited by hand.

| name | instrument (shootout option) | trim dB | fader dB | rule |
|---|---|---|---|---|
| keys `rhodes` | `gu_tine_ep` (GU Tine EP, prog 4) | 0.00 | −5.94 | CD-28: mean of K1 rootless −6.04 and K2 spread −5.84 |
| keys `wurli` | `gs_wurli_ep200` | −3.22 | −6.435 | CD-28: mean of −6.65 and −6.22 |
| keys `piano` | `vcsl_upright_knight` | +21.86 | −6.06 | CD-28: mean of −6.06 and −6.06 |
| kit `studio` | `kit_virtuosity` | −14.94 (in its lanes' db) | +15.26 | median F |
| its bass | `bass_vsco2_pizz` | +34.97 | +5.09 | median F |
| kit `jazz` | `kit_gu_jazz` (GU Jazz kit 32) | 0.00 (in its lanes' db) | +15.21 | median F |
| its bass | `bass_fp_finger_yr` | −2.79 | +5.13 | median F |
| sax `alto` (lead) | `sax_mtg_alto`, warmth off, expression on, timing lag | +6.63 | +2.55 | median F; lag keeps the instrument's (CD-26) |

- **Every number** is job 2's re-measure of job 1's (every move 0.00 dB; S\* −31.9247 LUFS in both), as each
  instrument's t and median F (SHOOTOUTS.md §3, "Frozen values"); the keys faders take CD-28's mean because the drafts
  pick their voicing.
- **Settings:** swing 0.55 on every lane, humanize off, `sax_warm` false, `sax_expr.enabled` true and `timing: lag`
  (the plan variants swing_55, humanize_off, sax_plain and timing_lag), `bed: null`, no aliases, the chord list and
  voicings as they stand (the grammar's full lists), loudness −14 LUFS.
- **Sources:** GeneralUser GS is the one committed source; every other (the converted SF2s, the FreePats bass, the
  Virtuosity tar) is mode `fetch` under `$LOFI_ASSETS`, as in every shootout palette (D-012).
- **Per-instrument levels.** palette.py holds one trim slot and one fader per bus until the freeze adds per-instrument
  fields (SHOOTOUTS.md §7.1), so each instrument's trim and fader, with where each number came from (both jobs' t and
  F, the K2 spread F, the source sha256 and bank and program or the kit map's sha256 that job 2 measured), is the `#| `
  block at the top of the file: YAML in comments, which palette.py does not read and the file's sha256 pins. A kit's
  trim is in its lanes' db, as in the shootout palettes.
- **As written** the palette plays the leaders' band (`rhodes`, `studio`): it is a palette the renderer accepts as it
  stands. **Each draft renders with its own render palette**, `draft_day2_<keys>_<kit>`: the palette narrowed to the
  draft's keys instrument and kit, with that band's trims in the chain-head `Gain` slots and its faders in
  `bus_gain_db`, and only the sources it plays (`lofi_rl.drafts.palette.render_doc`). It changes no note: a draft
  compiles to the same events with either.
- **Checks:** `python -m lofi_rl.drafts.palette check` (offline: the names, options and settings are the picks', each
  block is the instrument job 2 measured, the levels are the measured ones, the palette as written plays the leaders'
  band, every render palette parses); with `--levels-day1 … --levels-day2 …` a fresh build must be byte-identical; with
  `--verify`, the shootout's own palette validation (`lofi_rl.shootout.render.verify_palettes`: each source hashed once
  against its sha256, each SoundFont preset read) and the SF2 coverage check, on the palette and all 6 render
  palettes. Run on the laptop on 2026-11-01 with `LOFI_ASSETS` at the fetched assets: `ok … 6 render palettes parse;
  sources, presets and coverage verified`, and the build was byte-identical.

## 4. The vocabulary a draft may use

`python -m lofi_rl.drafts.palette vocab` prints it from the palette.

| | |
|---|---|
| keys instruments | `rhodes`, `wurli`, `piano` |
| voicing styles | `rootless`, `shell`, `spread`, `quartal` |
| kits (each with its bass) | `studio`, `jazz` |
| sax | `alto` is the only one; leave the `sax:` line out |
| numerals | I bII II bIII III IV V bVI VI bVII VII |
| qualities | maj7 maj9 maj7#11 7 9 13 7sus 7b9 7#9 m7 m9 m11 m7b5 dim7 |
| drum lanes | kick snare rim hat ohat shaker (a snare or rim base is required; one of them only) |
| per-bar edits (`@N lane:`) | comp bass kick snare rim hat ohat shaker |
| durations | s e q h w = 1 2 4 8 16 steps; a dot adds half: e. q. h. w. (no s.) |
| degrees | 1–7 of the mode, `b`/`#` before, `'`/`,` after for an octave up or down; `r` is a rest |

**Chords whose every tone lies in the mood's mode** (`lofi_rl.prompt.mood_chords`; others are allowed while the mood
signature still wins by 2):
- **melancholy** (aeolian): I7sus Im7 Im9 Im11 IIm7b5 bIIImaj7 bIIImaj9 IV7sus IVm7 IVm9 IVm11 Vm7 bVImaj7 bVImaj9
  bVImaj7#11 bVII7 bVII9 bVII13 bVII7sus
- **wistful** (dorian): I7sus Im7 Im9 Im11 IIm7 bIIImaj7 bIIImaj9 bIIImaj7#11 IV7 IV9 IV13 IV7sus V7sus Vm7 Vm9 Vm11
  VIm7b5 bVIImaj7 bVIImaj9
- **hopeful** (ionian): Imaj7 Imaj9 II7sus IIm7 IIm9 IIm11 IIIm7 IVmaj7 IVmaj9 IVmaj7#11 V7 V9 V13 V7sus VI7sus VIm7
  VIm9 VIm11 VIIm7b5

**Vamp ranks, seed data only** (gates/GATE_palette.md, Day 2; ties on one line):
- hopeful: aruarian `IIm9 V13 span 2`; royal_half `IVmaj7 IIIm7 span 1`;
  lift `Imaj9 IVmaj9 span 2` and two_of_us `IVmaj7 III7 span 2` (III7 is borrowed: it still wins ionian by 22)
- melancholy: plagal `Im9 IVm9 span 2`; rise `bVImaj7 bVII13 span 2`; backdoor `IVm9 bVII13 span 2`;
  drift `Im9 bVImaj7 span 2`
- wistful: float `Im9 bVIImaj7 span 2`; i_iv `Im9 IV13 span 2` and pulse `Im9 IV9 span 1`; step `Im11 IIm7 span 2`

## 5. The validator

```bash
python -m lofi_rl.drafts.check drafts/v1                       # every brief
python -m lofi_rl.drafts.check drafts/v1 --briefs d04+d07      # some briefs
python -m lofi_rl.drafts.check drafts/v1/redo1 --briefs d04 --against drafts/v1   # a redraft (§6)
```

For every draft of the briefs in scope it checks, in order, and prints one table row (then one line per problem):
1. **name**: `<brief id>_<persona id>.ls` with a brief of the briefs file and a persona of `groove`, `horn`, `chords`;
2. **gate**: G1–G6 with the draft palette's vocabulary in the brief's mood, no repair at all, no `sax:` line;
3. **compile**: against the draft palette and the brief, with no sax octave fold;
4. **energy**: E's band is the brief's (§2.1);
5. **mood**: the brief's mode wins the mood signature by 2 or more (VERIFIERS.md §4);
6. **refs**: character 5-gram Jaccard below 0.6 against every prompt example (prompt v1's, and the 9 of
   `prompt_v2.reference_texts`) and every shootout reference loop; these 19 are 10 distinct texts, since v2's nine are
   eight of the shootout loops (all but L5) and v1's example;
7. **drafts**: the same below 0.6 against every other draft of the folder and of each `--against` folder;

and, over the scope, **one draft per brief per persona**: exactly one file for each (brief, persona).

The Jaccard is the junk-rate near-copy measure (`lofi_rl.rl.junk`: character 5-grams) on each loop's copy form: its
canonical text with no `sax:` line and the keys instrument and kit written `K` and `D`, so naming another instrument
never lowers it. For scale, the 10 references sit at 0.22–0.44 from each other (median 0.32). E, the mood margin,
theory, P_sax and `rubs` are printed as information, never failures. `rubs` counts the sax notes that sound 4 steps or
more (as written, with the comp and the bass as heard, each sounding step playing its bar's voicing or root) a
semitone or a minor 9th from a keys or bass tone while the keys play the note's own pitch class under it for fewer
than 2 steps; each one is listed after the table (`d01_horn.ls: rub (information): bar 3 step 1: sax B4 +1 from keys
Bb4 for 12 of its 12 steps`, from the draft before the review of 2026-11-01; house rules, §2.1). `--json` writes the
whole report. Exit 0 when every check passes, 1 when one fails, 2 for bad arguments.

## 6. Redrafts

"None acceptable" (G) for a brief labels its three drafts Toss and asks for new ones. The three writers redraft that
brief into a new folder beside the old drafts, which stay as the labelled negatives: `drafts/v1/redo1/` (then
`redo2/`, …), holding only `<brief>_<persona>.ls` for the named briefs; its briefs come from `drafts/v1/briefs.yaml`.
Check it with `--briefs` and `--against drafts/v1` (and every earlier redo folder), so no redraft repeats a draft it
replaces, and render it with the same arguments (§7). A redraft batch is named `b<MMDD>-keep3-redo`, then `-redo2`, ….

## 7. The render job

```bash
make submit JOB=jobs/drafts.sbatch ARGS='--drafts drafts/v1'
make submit JOB=jobs/drafts.sbatch ARGS='--drafts drafts/v1/redo1 --briefs d04+d07 --against drafts/v1'
```

`jobs/drafts.sbatch` (`lofi-drafts`; `main`, 8 CPU, 16 GB, 15 min; 12 SU/min, **at most 180 SU**; `MODE=scorer`) runs
`python -m lofi_rl.drafts.job`, whose stages stop at the first failure, so nothing reaches the inbox unless every draft
passed every check and rendered in spec:
1. **check**: the validator over the scope (summary/check.json);
2. **palettes**: the draft palette's checks, one render palette per (keys, kit) the drafts name (summary/palettes/ with
   index.json), each source hashed once, every preset read, SF2 coverage;
3. **render**: every draft through `render.pipeline.render` (a fresh FluidSynth process per pitched stem, the drum
   sampler, the bus chains, the master and the loudness loop), 8 at a time with 1 worker each, into
   `$REMOTE_SCR/runs/drafts-<job>/renders/<brief>_<persona>/`; each must sit at −14 LUFS within 0.1 LU with a true peak
   at or below −1 dBTP (CLAUDE.md §5) and pass G1–G7; summary/renders.json records each bus's trim, fader and loudness
   (also relative to S\* and to its target, for the freeze's check of the kept 10, SHOOTOUTS.md §3);
4. **build**: one keep3 batch into `$REMOTE_SCR/inbox/<date>/b<MMDD>-keep3/`: one round per brief, its three drafts on
   tiles from a 3×3 Latin square (each persona on each tile 3 times over 9 rounds; over 10, each persona's tile counts
   4, 3 and 3, each tile's 4 another persona's), rounds in a drawn order with no two consecutive in one energy × mood
   cell (a redraft whose briefs share one cell cannot avoid that: its order is still drawn, the first with the fewest
   such neighbours, and draw.json says `order_constraint_met: false`), all drawn from a fresh 64-bit seed recorded only
   in summary/draw.json; opaque ids (`r` + 8 hex, `k` + 10 hex)
   that hold no hidden word; `manifest.json` (the scene, each round's bpm, opaque ids) and `manifest.hidden.json` (per
   clip the real clip id, persona, draft file and sha256, the brief's knobs, tempo and key, instruments, render palette,
   render id, pcm_sha256, LUFS, true peak); checked by `python -m lofi_rl.ui.manifest check` before it is moved into
   place; summary/batch.json maps every tile.

`make pull` brings back the summary and the inbox. **Laptop dry run** (2026-11-01, Apple M4, the real draft palette
and banks under a scratch run, inbox and cache, 6 fixture drafts for d01 and d05, `--jobs 8`): every stage in 8.9 s
(check 0.2, palettes 0.9, render 7.7, build 0.1); 6 of 6 renders ok at −14.094 … −14.028 LUFS, true peak at most
−1.494 dBTP, a render 4.8–7.7 s; the batch passed `ui.manifest check`. Laptop renders are not canonical (LESSONS 38),
and none is queued for listening.

## 8. Claude's defaults here

| default | reason |
|---|---|
| Names `rhodes`, `wurli`, `piano`, `studio`, `jazz`, `alto` | plain words (instrument.md §6); `rhodes` is gu_tine_ep's plan family, `jazz` the GU kit's own name, `studio` a neutral word for Virtuosity; the freeze may rename them, a mechanical change of two lines |
| Keys faders by CD-28's mean | the drafts pick their voicing, as the model will |
| Per-instrument levels in a `#| ` comment block, one render palette per (keys, kit) | palette.py, the renderer and the shootout tools are frozen or being edited elsewhere; this is CD-51's own carriage, and the file's sha256 still pins every value |
| The palette as written plays the leaders' band | it renders as it stands, and the one band it can carry is the leaders' |
| Stratified tempos and distinct tonics | 10 briefs from the sampler alone gave 5 of 10 outside 75–90 at seed 1105 and repeated tonics; the strata keep TASTE's centre and the keys stay different |
| No repair at all, no `sax:` line | a draft that needed a repair would teach the slip; the line chooses nothing and prompt v1 shows none |
| Jaccard on the copy form, the 10 distinct references | renaming an instrument must not hide a copy; v2's examples are the shootout loops |
| No hidden controls in the keep3 batch | D-013's repeats need 2 days and its anchors a sour copy; the keep3 rounds label drafts, and the RM's consistency comes from the bws4 repeats (D-008) |
| A redraft goes to a new folder | the rejects stay as labelled negatives (TASTE.md, Labeling) |
| Loudness held to −14 ± 0.1 LUFS and −1 dBTP, not G7's 0.5 LU and −0.9 dBTP | CLAUDE.md §5's spec for a render |
| A rub: 4 written steps, a semitone or a minor 9th (or two octaves and a semitone) from a keys or bass tone, unless the keys play the note's pitch class 2 steps under it; information, never a failure | the review of 2026-11-01: on the 30 drafts as first written it lists exactly the 12 a scan of the compiled notes found, and no 2-step passing note; a b9 the symbol does not name may still be meant, so the column informs and the house rule asks for 0 |
| An answer ends on another pitch class than its call; never 3 bars without a rest; bar 5 more than a ghost or a level | TASTE.md's *ends differently*, *up front like a vocal* and *something new every bar*, read where the verifiers (f_ans, f_breath, f_new) pass by the letter: six answers ended on the call's pitch, two lines ran 56 steps, and in 11 drafts bar 5 changed only a ghost or a level; tests/test_drafts_v1.py holds drafts/v1 to them |
