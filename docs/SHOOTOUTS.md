# SHOOTOUTS.md: the Phase-1 shootouts

Shresth hears the shootouts in the rating page (`make rate`, docs/RATING.md) on **Nov 3–4** (51 minutes planned,
inside the plan's "about 55 min") and on **Nov 6** (about 15 minutes: rain, and the hidden checks his vote rules
call for). They prune each sound choice to its survivors; the survivors go into `palettes/lofi_rain_v1.yaml`,
which freezes on Nov 6 and never changes after (docs/DECISIONS.md D-017).

- **Status (2026-10-31, revised after two reviews):** the job-1 tooling of §7.11 is built and integrated on one
  plan (§7.1–§7.8, below), and so is the tally tool that job 2 waits for (§7.9, branch `wt/tally`). A laptop dry run
  of job 1 rendered all 67 Day-1 clips and built the three Day-1 batches (§7.7, "Dry run"); those laptop clips are
  not canonical (LESSONS 38) and none is queued for listening. Since CD-52 each clip takes its own fader on the bus
  its round varies, and every Day-1 round's candidates sit within 0.01 LU of each other on its loop in the dry run
  (§3 step 8). Job 1 then ran on the cluster (`lofi-shootout` 13635646, 25.8 SU; CHANGELOG 2026-10-31 17:45): its
  three Day-1 batches are in `inbox/2026-11-03/`, and the tally reads per-round spreads of 0.0012–0.0091 LU from
  their hidden manifests (§7.9). The 9 reference loops in `shootouts/loops/` exist and pass the gates (§5).
- **Sound claims:** none. Every candidate, variant and level here is a starting point for Shresth's ear. Claude
  records counts and picks, never descriptions of sound (TASTE.md: "Shootout prunes, votes decide").
- **Blind by construction.** This file gives every round in symbols (`P1a`, `k2`, `v3`, ...). Which option sits
  on which tile, the order of the rounds and the hidden controls are drawn by the batch builder at job time from
  a seed that is never committed (§2.3). The drawn tables are published in `gates/GATE_palette.md` after the
  freeze.
- **Sources**, and the tags that cite them:

  | Tag | Source |
  |---|---|
  | T·*row* | a row of `docs/TASTE.md` (Shresth's interview, the spec) |
  | P§*n* | the project plan, `~/.claude/plans/sequential-skipping-cookie.md` (§3 palette and shootouts, §4 listening budget, §7 schedule, §9 pushes) |
  | I6 | `docs/design/instrument.md` §6 (shootouts and palette schema) |
  | A | `docs/SHOOTOUT_ASSETS.md` (the 14 fetched sources) |
  | R | `docs/RATING.md` (shootout mode, batch format, hidden manifest, keys) |
  | D-*n* | an entry of `docs/DECISIONS.md` |
  | CD-*n* | a Claude's default; each has its one-line reason in §10 |

## 1. Purpose and outputs

| Shootout | Candidates | Outcome | Goes into `lofi_rain_v1.yaml` |
|---|---|---|---|
| Sax | 6 sources | **exactly 1 source** (T·Sax "one sax"), plus its variants: warmth on/off, expression kept or not, timing on-grid / lag / swung | `sax` (one voice), `sax_lead`, `mix.sax_warm`, `sax_expr.enabled`, `sax_expr.timing`, the sax trim and fader |
| Keys | 8 instruments | **2–3 instruments** (T·Keys) | `keys` (2–3 voices), a trim and a fader each |
| Drums + bass | 3 kits, 3 basses, 4 swings, 3 humanize settings | **2–3 kits**, each carrying its paired bass (I6); **one swing**, which stays 0.55 unless Shresth names another (T·Tempo/feel) [CD-13]; **one humanize** setting (T·Drums, T·Bass, T·Humanize) | `kits` (lanes from the importer, §7.4, and the paired `bass`), `groove.swing`, `groove.humanize`, a trim and fader per kit and per bass |
| Vamps | 12 vamps, 4 per mood | **harmony data**: a ranking of the 4 vamps in each mood, input to the seed workflow (P§5); and **a report on borrowed chords** for Shresth to decide (T·Harmony; P§3) [CD-45] | `chords.numerals` changes only on his words |
| Rain | 3 excerpts; Δ ∈ {none, 30, 24, 18} | one excerpt and one Δ (T·Mix "Rain barely there"; A: the excerpt needs his ear) | `bed.src`, `bed.start_s`, `bed.delta_lu`; `bed: null` only on his words |

Drum **patterns** are not compared with the kit fixed: R1 plays each kit on three patterns, but the palette has no
pattern field and the model writes every pattern [CD-46; §11 question 3].

**From votes to the freeze** (gates/GATE_palette.md; P§3 "Freeze"):
1. After each day's batches Claude tallies the votes (§7.9, §8), proposes the survivors and pushes the tally with
   whatever needs Shresth's word (swing, borrowed chords, `rain_none`, unresolved ties). He says which to keep
   (R: "tell Claude which 2–3 to keep"); his word decides.
2. Claude writes the survivors, the tally tables, flagged rounds and the audit file paths into
   `gates/GATE_palette.md`, and a CHANGELOG entry. Shootout votes never train the RM (taste_pipeline.md §2).
3. Nov 5: the 30 drafted loops are rendered with the Day-2 picks and Shresth keeps 10 (keep3; P§7). The drafts use
   the chord list as it stands after his borrowed-chord answer (§8).
4. Nov 6, after the last batches: Claude assembles `palettes/lofi_rain_v1.yaml` from the survivors, their trims and
   faders (§3) and the rain pick, with plain-word names the model sees (I6), and checks the kept 10 against it
   (chord list, bus levels). Shresth signs off with `APPROVED by Shresth Rana <yyyy-mm-dd>` in
   `gates/GATE_palette.md`; Claude sets `frozen:`, adds the DECISIONS entry and copies the drawn round tables into
   `gates/GATE_palette.md`. Nothing changes after (D-017).

## 2. Common protocol

### 2.1 Rounds
- A round is **one loop text and one brief**, with **2–4 clips that differ in one thing only**: the candidate (a
  source, a variant, a vamp line or a level). Everything else is byte-identical: loop text (apart from the line
  under test), brief, palette (apart from the candidate's block, its trim and its fader), the other buses' trims
  and faders. A test enforces this (§7.8). Anchors (§2.5) are the one exception, by design.
- Shootout mode (R): best with `Q W E R`, worst with `Z X C V`, `T` no clear best, `B` no clear worst, `G` all bad.
  No reason tags. The page shows only the scene, "city at night".
- Each batch's `note` names its category, never an option (R), for example "Which keys do you like best?".

### 2.2 Rendering
- **Every clip is a cluster render** (CLAUDE.md §6; LESSONS 38), made by the `lofi-shootout` job (§7.7).
- **One pass each:** a clip is one cycle of the loop, exactly N samples, as RATING.md requires (bars × 240 / bpm
  within 2 %). The page loops each clip, but auto-play hands every clip except the last to the next one at its
  bar 8 (`audio.js` `planAuto`), so a clip's own seam is heard only on the last tile or when he stays on a tile
  (taste_pipeline.md §9 issue 4).
- **Listening copy:** 16-bit FLAC with the renderer's TPDF dither (RENDER.md §3.6), named by an opaque id (`k` +
  hex, R). Renders, their WAVs and their manifests stay in `$REMOTE_SCR/runs/shootout-<job>/` on the cluster; only
  the FLACs and the two batch manifests go to the inbox (§6).
- **Hidden manifest** (R): batch `category`; per round `kind` (`none`, `repeat` with `of`, `anchor` with
  `anchor_clip`), `brief_id` (the loop's name) and `category`; per clip `clip` (the real id, taste_pipeline.md §2),
  `option`, and the free fields `what` (variant), `loop`, `palette` (id@sha8), `trim_db`, `fader_db` (the clip's
  own fader on the bus its round varies, CD-52), `bus_lufs` (post-fader loudness of the candidate's bus, as the level
  tool measured it after that fader; keys and bass in vamp rounds), `bus_lu` (the same relative to S\*, §3; clips of
  instrument stages), `sax_range` (lowest and highest MIDI note of the lead), `render` (render id) and `pcm_sha256`.

### 2.3 Blinding (the draw)
The repo is public (T·Assets), so nothing committed may say which option is on which tile, or which round is a
hidden control (R: "You are never told which rounds those are") [CD-21].
- **Public, here and in `shootouts/plan.yaml`:** the candidate lists (TASTE and I6 name them anyway), each stage's
  design in symbols with its constraints, the loop of each round slot, and each batch's contents.
- **Drawn per job by the batch builder (§7.6)** from a 64-bit seed it creates at job time:
  1. which option each symbol stands for, within the stage's constraints (§4: for example, K1's twin pairs are
     always families, and plagal and float always get the central tile sums);
  2. the order of the rounds in each batch (§2.4);
  3. the hidden controls (§2.5): which round each anchor imitates, which clip is broken, which round each repeat
     shows again, and where each sits.
- **Recorded only** in `runs/shootout-<job>/summary/draw.json` and `batches.json` (on the cluster, and in the
  laptop's gitignored `runs/` after `make pull`; CLAUDE.md §7) and in the hidden manifests.
- **Published after the freeze:** Claude copies the drawn tables into `gates/GATE_palette.md` with the tallies.
- **Tests** (§7.6, §7.8): `plan.yaml` holds only symbols in its tiles; two seeds give different option-to-tile maps;
  one seed always gives the same batches; every draw meets its stage's constraints.

### 2.4 Balance
- **Each candidate is heard in at least 2 different loops** (3 or 4 for most), spread over moods and energies.
- **Tiles.** Auto-play runs 1 → 4 (R), so position is the main nuisance:

  | Stage | Design | Tile sums |
  |---|---|---|
  | K1 | 6 rounds of 4; each candidate on 3 different tiles | 6–9 each (the minimum spread for 8 candidates); each family's two members add to 15 [CD-8] |
  | K2 | 2 rounds of 3 (4 on a tie) | 3, 4, 5 (5 each on a tie) |
  | R1, R2, R3b, S2a, S2b, W1, W2 | Latin squares: each option on every tile once [CD-14] | equal |
  | R3a | 2 rounds of 4, the second reversed | 5 each |
  | S1 | 6 rounds of 4; each source on every tile once [CD-17] | 10 each |
  | S1 final | 3 rounds of 2 | 4 and 5 |
  | Vamps | per mood, 3 rows of a 4×4 Latin square | 6, 7, 8, 9; plagal and float always 7 or 8 [CD-16] |

- **Order** [CD-22]: consecutive rounds of a batch never share a loop, and a batch opens with a design round (not a
  repeat or anchor). A scratch planner (2026-10-31) found such an order in 200 of 200 draws for every Nov-3 and
  Nov-4 batch, and in 500 of 500 draws for the two Nov-6 batches with their repeats (redrawing the split between
  them when needed); each Nov-6 batch stayed under 8.6 min planned, or 11.6 with the S2 re-run.
- **Position checks:** the tally tool pools position-of-best counts by clip count across all shootout batches,
  labelled descriptive (§7.9).

### 2.5 Hidden controls (D-013)
D-013 (decided by Shresth, pre-registered): about 10 % of rounds are repeats, **at least 2 days later**, with tiles
reversed; about 5 % are broken anchors; a session that misses its anchors counts at half weight. The shootouts
follow it as written. Whether to keep these checks in the shootouts at all is put to him (§11, question 1).
- **Repeats: 5**, one per group: keys (K1, K2); drums + bass (R1, R2, R3a, R3b); vamps; sax sources (S1 and its
  final); sax variants (S2a, S2b). Each shows one design round of its group again, drawn from every round except the
  first one served of each stage [CD-24], tiles reversed, in a Nov-6 batch: Day-1 rounds come back 3 days later and
  Day-2 rounds 2 days later. Before job 3 Claude reads the vote dates and names the stages whose rounds were first
  heard at least 2 days before Nov 6; repeats are drawn only from those, and a group with none gets no repeat. No
  repeat comes after the freeze, so rain has none. 5 of 58 rounds (8.6 %).
- **Anchors: 3**, one per day [CD-23]. An anchor is an extra round of its batch's stage (loop, candidates and tiles
  drawn) in which one clip is a **sour copy**: the same loop with every sax note's accidental raised by one (a
  natural becomes `#`, a `b` becomes natural), rendered with that clip's candidate. The 9 loops' sour copies pass
  G1–G6 in every mood with 4–12 non-free notes each, P_chrom at its 0.3 cap (§5). Anchors sit only in batches that
  have a sax. 3 of 58 rounds (5.2 %).
- **In the tallies:** the audit leaves repeats out of the counts and skips the anchor clip of an anchor round
  (`ui/audit.py`); Claude's recount drops anchor rounds entirely and gives half weight to a micro-session that
  missed its anchor (D-013; §8).
- **Not in D-008's p:** shootout repeats measure consistency on instrument and level comparisons; the p of D-008's
  gate comes only from bws4 repeats.
- **What could give a repeat away** (reported with the agreement figure): on Nov 6 the keys, drums + bass, vamp and
  sax rounds are otherwise over, so a round that changes keys, drums or bass, chords or the sax is a repeat; a vamp
  repeat is the only round that day with no sax; a repeat has no rain in any clip, unlike the rain rounds; a Day-1
  repeat plays the dev band in its fixed slots.

### 2.6 Order and time per round
**Order** [CD-48]. Every stage that needs an earlier result comes a day later, so each day's batches render the
night before:

| Day (date) | Batches, in the order `make rate` serves them [CD-49] | Needs |
|---|---|---|
| 1 (Nov 3) | keys K1 → groove R1, R2 → sax S1 (+ anchor) | the tooling of §7 and the SFZ → SF2 converter; job 1 on Nov 2 evening |
| 2 (Nov 4) | groove R3a, R3b → sax final, S2a, S2b (+ anchor) → keys K2 → vamps | Day-1 tallies; job 2 on Nov 3 evening |
| 3 (Nov 6) | two mixed batches: rain W1, W2 (+ anchor) and the 5 repeats (+ the S2 re-run, §4.4) | the rain file by Nov 5 noon (§11); job 3 on Nov 5 evening; before the sign-off |

Days are relative: if the tooling slips, every day moves back by the same amount and the freeze with it, and the
repeats keep their 2-day gap.

**Time per round** from RATING.md's numbers: one pass is 8 bars × 240 / bpm = 1920 / bpm seconds (25.3 s at
76 BPM, 21.3 s at 90); auto-play plays each clip's 8 bars in turn; a clip counts as heard at 4 bars. The planning
figure (taste_pipeline.md §8) is 0.75 of auto-play plus 8 s to mark and save:

| Clips | Full auto-play | Heard minimum (4 bars each) | Plan estimate |
|---|---|---|---|
| 2 | 42.7–50.5 s | 21.3–25.3 s | 40.0–45.9 s |
| 3 | 64.0–75.8 s | 32.0–37.9 s | 56.0–64.8 s |
| 4 | 85.3–101.1 s | 42.7–50.5 s | 72.0–83.8 s |

(90 BPM to 76 BPM.)

### 2.7 Budget, pushes and cut list
Per round at each loop's tempo, from the round slots of §4 (scratch planner, 2026-10-31):

| Shootout | Rounds | Plan estimate | Full auto-play |
|---|---|---|---|
| Keys (K1 6, K2 2) | 8 | 9.7 min | 11.5 min |
| Drums + bass (R1 3, R2 3, R3a 2, R3b 3) | 11 | 11.6 min | 13.5 min |
| Vamps | 9 | 11.6 min | 13.9 min |
| Sax (S1 6, final 3, S2a 3, S2b 3) | 15 | 16.0 min | 18.7 min |
| Rain (W1 3, W2 4) | 7 | 8.3 min | 9.8 min |
| Anchors (D-013) | 3 | 3.8 min | 4.5 min |
| Repeats (D-013), drawn | 5 | 4.7–6.6 min (5.7 expected) | 5.4–8.0 min (6.7) |
| **Total** | **58** | **66.7 min** (65.7–67.6) | **78.6 min** (77.3–79.8) |

- **By day:** Day 1 23.0 min (19 rounds); Day 2 28.3 min (26); Day 3 15.3 min expected (13 rounds; 14.4–16.3).
  Nov 3–4 is 51.3 min, inside P§7's "about 55 min"; the total is inside P§4's 1.1 h for Phase-1 shootouts. If the
  sax final changes the winner, the S2 re-run adds 6.1 min (6 rounds) on Day 3.
- **Pushes** (P§9 and TASTE's defaults: at most 2 a day, each worth 15 min of listening or less, by the plan
  estimate): Nov 3: keys + groove 13.8 min, then sax 9.2. Nov 4: groove + sax 14.7, then keys + vamps 13.6.
  Nov 6: the two mixed batches, about 7.7 each and never above 8.6 (11.6 with the S2 re-run). Each day stays
  under T·Listening's hour.
- **At full auto-play** Day 1 is 27.3 min, Day 2 33.1 and Day 3 18.1; the larger pushes then run 16–17 min.
- **Cut list** [CD-50]. After Day 1, Claude divides the audit's minutes for Day 1 (`time.minutes`, summed over its
  batches) by the plan's 23.0, giving r. If r × 28.3 would pass 30 (Day 2's two pushes) or r × 51.3 would pass 55
  (P§7), cut in this order until both fit:

  | # | Cut | Saves (plan / full) | What is lost |
  |---|---|---|---|
  | 1 | Vamp rounds on the high loops (L3, L6, L9) | 3.7 / 4.3 min | each vamp heard in 2 loops; the 2 rounds per mood play the second reversed (every tile sum 5) |
  | 2 | The third round of R3b, S2a and S2b | 3.0 / 3.4 min | those stages fall back to 2 rounds, the default on the middle tile sum |
  | 3 | The S1 final | 2.1 / 2.3 min | S1 alone picks the sax (§8) |

  Day 3 is cut only if r × its plan would pass 30: first W1 down to 2 rounds (e1 on the middle tile sum), then W2
  down to 2 rounds, the second reversed.

## 3. Level matching [CD-25, CD-52]

A candidate must never win by being louder, and each candidate should meet its bus chain the way the dev instrument
does. Every instrument in every clip plays with a **trim** before its bus chain and a **fader** after it, so that
its bus sits at the palette's `bus_target_lu` relative to the sax bus (sax 0, drums −1.5, bass −3, keys −6 LU:
instrument.md §4, `dev_gu.yaml`). Today every `dev_gu` gain is 0 dB, and on the fixtures the GU drums sit 15.42 LU
under the sax (RENDER.md §3.3), so unmatched clips are not an option.

**Per clip** [CD-52, a Claude's default amending this section, 2026-10-31]. Each shootout clip's fader on the bus its
round varies comes from that candidate's measured bus loudness on the round's own loop, so every candidate in a
round sits at the same bus loudness: the palette's `bus_target_lu` relative to the sax (S\* + T below, as before),
within 0.1 LU, checked clip by clip. *Why:* the first Day-1 dry run matched each candidate's bus to its target by the
median over the 9 reference loops, which left 10 of the 18 design rounds, and the anchor, with a candidate loudness
spread above 1.0 LU on the round's own loop, the largest 4.20 LU (R1.3 on L8); a candidate must never win by being
louder. The frozen palette still takes each candidate's median fader, as before, recorded in `levels.json`. The
level tool already measures every (candidate, loop) pair on the 9 reference loops; the one text a clip plays that
they do not hold, an anchor's sour copy, joins its measure plan (§7.1). The hidden manifest records each clip's
fader and its measured loudness (§2.2, §6).

**Why a trim before the chain.** During review (2026-10-31) the GU Jazz kit's one-shots peaked at −22.8 dBFS (hard
kick) and −21.8 (hard snare), while Virtuosity's kick-mic kick peaked at −1.7 dBFS and Swirly's kicks and snares
reached −0.2 and −0.1. Untrimmed, the file kits would reach the drum compressor (threshold −16 dB) about 20 dB
hotter than the GU kit. Scaling the GU stems on L1, L3 and L8, every chain stayed within 0.3 dB of
level-independent up to about +14 dB, but at +21 dB the drum chain took 1.7 dB more off L3, against at most
0.06 dB more off L1 and L8. A fader after the chain matches loudness, not what the compressor did.

**Procedure** (the levels tool, §7.1; job 1 before any clip renders, job 2 for the Day-2 variants):
1. **Stems.** For every instrument variant *i* (Day 1: the 8 keys instruments in rootless voicing, 3 kits, 3
   basses, 6 saxes; Day 2: the K2 three in spread voicing, the S1 leader with warm on and with expression off) and
   each of the 9 reference loops ℓ with its own brief, build *i*'s bus input with trim 0 (RENDER.md §2 stages 2–3:
   the FluidSynth stem or the drum sampler). One run measures one variant of each bus at once: 8 runs per loop on
   Day 1, 3 on Day 2.
2. **Trim.** P(*i*, ℓ) is the integrated loudness of *i*'s bus input over the kept cycle. t(*i*) = median over ℓ of
   P(*g*, ℓ) minus median over ℓ of P(*i*, ℓ), where *g* is the GU instrument of *i*'s bus: GU Tine EP 4 (keys),
   GU Jazz kit 32 (drums), GU Acoustic Bass 32 (bass), GU Tenor 66 (sax). GU instruments get t = 0.
3. **Chain.** L(*i*, ℓ) is the loudness of *i*'s bus after its chain, with the trim, before the fader.
4. **One sax target.** S\* = the median of L(GU tenor, ℓ): the sax bus stays where `dev_gu` puts it.
5. **Fader.** F(*i*) = S\* + T(bus) − median over ℓ of L(*i*, ℓ), with T the bus target above: the median fader,
   which the frozen palette takes. **A clip's own fader** [CD-52] on the bus its round varies is F(*i*, ℓ) =
   S\* + T(bus) − L(*i*, ℓ) on the round's loop ℓ, then corrected by the loudness that bus measures after it until
   it is within 0.005 LU of S\* + T, at most 3 times: a pure gain moves a quiet bus through BS.1770's −70 LUFS gate
   (`kit_virtuosity` on L1, at −55.0 LUFS before its fader, measures 0.101 LU under L + F). In the Day-1 dry run the
   median of each variant's 9 clip faders equals its F(*i*) (all 20 variants), so the frozen fader is the middle of
   those Shresth hears. The fixed instruments of each round keep their median F, so they are the same in
   every clip of the round (§2.1). Trims and faders are rounded to 0.01 dB (as `mix.calibrated_gains` rounds).
6. **Checks.** (a) Linear range: feeding *i*'s trimmed input 6.02 dB lower must lower L by 6.02 ± 0.1 dB (median
   over the loops); a candidate that fails is flagged, since its chain then treats it differently from the GU
   instrument. (b) The median: at its median F, each candidate's median post-fader bus loudness must equal S\* + T
   within 0.05 LU (the frozen palette's fader). (c) Per clip: at its own fader, every clip's bus must sit at S\* + T
   within 0.1 LU. How far each loop's bus moves from its own L + F through the −70 LUFS gate at the median F is
   recorded (`worst_loop_lu`) but flags nothing now, since each clip's own fader is measured after it [CD-52].
7. **Render** every clip with those trims and its own faders; the master chain, limiter and −14 LUFS normalisation
   run as usual (RENDER.md §2), so every clip ends at −14 ± 0.1 LUFS. The drums end up about 14 dB above today's
   renders (RENDER.md §3.3 proposes +13.92 dB on the fixtures), so the master chain sees more drums than it does
   today. A render must play the bus the tool measured: its pre-fader loudness within 0.001 LU of the tool's L, at
   the clip's fader (§7.7, render status `level` otherwise).
8. **Per round:** each candidate's post-fader bus loudness on that round's loop is recorded (the hidden manifest's
   `bus_lufs`, as the level tool measured it after the clip's own fader, and `bus_lu`, the same relative to S\*); a
   spread above 1.0 LU flags the round in the tally report (§8). Step 6c bounds every spread at 0.2 LU.
   - **Measured in the first Day-1 dry run, median faders** (laptop, real banks and the two §7.4 kits, 2026-10-31;
     §7.7): every variant's median post-fader bus loudness sat within 0.005 LU of S\* + T, and the renders carried
     the tool's L and F exactly (264 render-bus pairs, largest difference 0). But matching medians over the 9 loops
     leaves the candidates of a round apart on that round's loop: 10 of the 18 Day-1 design rounds, and the anchor,
     have a spread above 1.0 LU (K1 0.66–2.95, R1 0.89–4.20, R2 0.95–3.85, S1 0.49–2.73 LU; the largest R1.3 on
     L8). The old per-loop check (each loop within 0.1 LU of its own L + F) flagged `kit_virtuosity`: on L1 its drum
     bus, at −55.0 LUFS before the fader, sat 0.101 LU under its own L + F after it (the −70 LUFS gate).
   - **Measured with per-clip faders** [CD-52] (the same seed and inputs, three runs, 2026-10-31; §7.7): the tool
     measured 181 clip faders (20 variants on the 9 loops, and the anchor's sour copy of L9 for its candidate); 169
     took one measurement and 12 one correction, and the worst sits 0.0050 LU from S\* + T. Every Day-1 round's
     candidate spread on its loop, read from the hidden manifests, is 0.0012–0.0091 LU (K1 0.0050–0.0069, R1
     0.0063–0.0072, R2 0.0037–0.0057, S1 0.0012–0.0091, the anchor 0.0078); as pre-fader loudness + fader from the
     render records it is at most 0.0691 LU (R2.2 on L7, the gate on a quiet bus). All 67 renders read the tool's
     L before their own fader exactly (difference 0). No variant is flagged; `kit_virtuosity`'s L1 clip takes
     21.65 dB (its median F is 15.26) and sits 0.0031 LU from its target. The trims and median faders are those of
     the median-fader run, and its 26 stage palettes are byte for byte the same.

**Not matched:**
- **Variants that move notes in time** (swing, humanize, sax lag and swung timing) keep the instrument's trim and
  fader [CD-26]. Their bus loudness is recorded per clip; a spread above 0.5 LU across a round's clips is flagged.
- **Vamps** keep the band's trims and faders; the harmony is the candidate [CD-27]. A vamp changes the keys and bass
  notes, so each clip's keys and bass bus loudness is recorded and a spread above 1.0 LU across a round's clips is
  flagged.
- **Rain:** the level is the candidate.

**How the values are carried** [CD-51]. Each shootout palette holds one instrument per bus, so no palette code is
needed before the freeze: a kit's trim is added to every lane's `db`, a FluidSynth instrument's trim is a `Gain`
plugin at the head of its bus chain, and faders go into `mix.bus_gain_db` (−60 … +24). Each clip of an instrument
stage plays its own palette: its stage palette with the varied bus's fader set to the clip's own [CD-52; §7.8]. The
frozen palette holds 2–3 keys and kits on one bus, so it needs per-instrument `trim_db` and `fader_db` fields (§7.1,
by Nov 6).

**Frozen values:** each survivor's t and median F (step 5), not a clip's own fader [CD-52]; a keys instrument's F is
the mean of its rootless (K1) and spread (K2) values [CD-28]. These replace instrument.md §4's calibration on the
hand loops [CD-29]. At the freeze Claude records the kept 10's median loudness per bus; a bus more than 1.0 LU from
its target is put to Shresth before the sign-off.

**Recorded per instrument variant** in `runs/shootout-<job>/summary/levels.json`: id, bus, source id and sha256,
bank and program (or kit map sha256), variant flags (voicing, `sax_warm`, `sax_expr`), the 9 values of P and of L,
t, F, S\*, T, `clips` (each clip case's L, own fader, the loudness measured after it, that loudness relative to S\*,
the miss from S\* + T and the measurements it took: the 9 loops and every drawn text such as an anchor's sour copy),
the three checks, and the drift keys: the mirror stamp (`.sync-stamp`), the base palette's sha256 and each (run,
case)'s compiled digest. **Per clip** (hidden manifest and `renders.json`): every bus's t and F (the varied bus its
own), the candidate's bus loudness as the level tool measured it after that fader (`bus_lufs`, and `bus_lu` relative
to S\*), the render's `level` record (the tool's L against the render's own pre-fader loudness), the master gain,
final LUFS, true peak and `pcm_sha256`.

**Drift between jobs.** Voicing, choke and humanize code are still changing in other workflows. Jobs 2 and 3 compare
their drift keys with job 1's; on any difference they re-measure every instrument they play (stage-level, cheap)
and use the new values, and a trim or fader that moves by more than 0.1 dB is listed in the tally report and the
next push. The bus chains are pinned in `shootouts/palettes/base.yaml` from job 1 to the freeze: instrument.md
§4's chain tuning on the hand loops does not happen before the freeze, since it would change what Shresth heard.

## 4. The shootouts

**Fixed in every round** unless the round tests it: the loop's brief (mood, tempo, tonic; §5); swing 0.55 on every
lane (T·Tempo/feel); humanize off (the palette default, COMPILER.md §3.6); `sax_warm` off, sax expression on, timing
`grid` (palette defaults, COMPILER.md §3.7); voicing as written, rootless (I6) [CD-4]; no bed [CD-6]; trims and
faders per §3.

**The fixed band** [CD-7]:
- **Day 1** (K1, R1, R2, S1): the dev palette's instrument in every slot not under test: keys GU Tine EP 4, kit GU
  Jazz 32, bass GU Acoustic Bass 32, sax GU Tenor 66.
- **Day 2:** the Day-1 leaders (§8): keys the K1 leader, the R1 leader kit with its R2 bass, sax the S1 leader.
- **Day 3:** the Day-2 picks: keys the leader over K1 + K2, the R3 picks, the sax after the final with its S2 picks.

Option ids below are the `option` values of the hidden manifests. Tiles read left to right as 1–4. Symbols are
mapped to options by the draw (§2.3).

### 4.1 Keys (T·Keys; I6; A)
| Id | Source (licence) | Preset | Family |
|---|---|---|---|
| `jrhodes3d` | jRhodes3d mono, converted SFZ → SF2 (CC BY-NC 4.0, fetch-only) | converter bank `jrhodes3d-mono`, (0, 0) | Rhodes |
| `gu_tine_ep` | GeneralUser GS (permissive) | (0, 4) Tine Electric Piano | Rhodes |
| `fp_fm_ep1` | FreePats FM Piano #1 (CC0) | (0, 0) [CD-37] | other EP |
| `gs_wurli_ep200` | Greg Sullivan Wurlitzer EP200, converted (CC BY 3.0) | `gsullivan-wurlitzer`, (0, 0) | other EP |
| `fp_upright_kw` | FreePats Upright Piano KW (CC0) | (0, 0) | piano |
| `vcsl_upright_knight` | VCSL Upright Piano, Knight, converted (CC0) | `vcsl-upright-knight`, (0, 0) | piano |
| `fp_fsbs_jazz_guitar` | FreePats FSBS jazz guitar, bridge (CC0) | (0, 0) | guitar/organ |
| `gu_organ` | GeneralUser GS | (0, 16) Tonewheel Organ | guitar/organ |

**K1, tournament** (Day 1, rootless) [CD-8]. Twin pairs P1–P4 are the four families; the draw picks which family is
which pair and which member is *a* or *b* (384 maps). Three parallel classes of 2 rounds, each class on one loop:

| Slot | Loop | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| K1.1 | L5 wistful mid 83 | P1a | P1b | P2a | P2b |
| K1.2 | L1 melancholy low 76 | P1b | P3a | P3b | P1a |
| K1.3 | L9 hopeful high 90 | P4a | P4b | P1a | P1b |
| K1.4 | L5 | P3a | P3b | P4a | P4b |
| K1.5 | L1 | P2b | P4a | P4b | P2a |
| K1.6 | L9 | P2a | P2b | P3a | P3b |

Twins meet 3 times and every other pair once; each candidate is heard on L1, L5 and L9 on three different tiles;
the two members of every family have tile sums that add to 15.

**K2, spread re-run** (Day 2; I6 "top 3 re-run with the spread voicing"). The K1 top 3 (§8) as u1–u3, ranks drawn
to symbols, with `keys: ep spread`. Their clips keep their K1 ids as `option`, with `what: spread`, so K1 and K2
add up per instrument. A tie at third place takes all tied up to 4 (u4, the second round then reversed) [CD-41]:

| Slot | Loop | 1 | 2 | 3 |
|---|---|---|---|---|
| K2.1 | L3 melancholy high 87 | u1 | u2 | u3 |
| K2.2 | L8 hopeful mid 85 | u2 | u3 | u1 |

### 4.2 Drums + bass (T·Drums, T·Bass, T·Humanize; I6; A)
**Kits** (§7.4 has each lane map; every kit takes `dev_gu`'s choke list):

| Id | Source (licence) | Mics and lanes |
|---|---|---|
| `kit_virtuosity` | Virtuosity Drums v0.925 (CC0), 48 kHz, converted once to 44.1 kHz (§7.2) | premix **kickmic + snaremic + oh** at 0 dB; room, mid and lofi left out [CD-31]; its own perc shaker (close + oh), level set as a borrowed lane [CD-30] |
| `kit_swirly` | Karoryfer Swirly Drums v1.104, brushes, dry (CC0) | kick beater + reso, snare top + bottom at 0 dB; rim = snare edge (A); **shaker = VCSL Shaker, Large** (the kit has none), level set as a borrowed lane [CD-30] |
| `kit_gu_jazz` | GeneralUser GS Jazz kit 32 as one-shots (`dev_gu`) | its own lanes, shaker note 82 |

**Basses:** `bass_vsco2_pizz` (VSCO 2 CE contrabass pizzicato, converted, CC0), `bass_fp_finger_yr` (FreePats
Finger Bass YR, CC0; it covers MIDI 26–45, the bass register is 28–43), `bass_gu_acoustic` (GU 32).

**R1, kits × patterns** (Day 1; I6 "kits × 3 patterns at 55 %, no humanize"). Kits as k1–k3; every clip plays the
GU acoustic bass, so the kit is the only difference. Latin square. Across the three loops every lane is played
(snare L3 and L8, rim L4, ohat L8, shaker L3):

| Slot | Loop | 1 | 2 | 3 |
|---|---|---|---|---|
| R1.1 | L3 melancholy high 87 | k1 | k2 | k3 |
| R1.2 | L4 wistful low 78 | k2 | k3 | k1 |
| R1.3 | L8 hopeful mid 85 | k3 | k1 | k2 |

No round compares patterns with the kit fixed [CD-46; §11 question 3].

**R2, basses on the kits** (Day 1). One round per kit, all three kits [CD-9], so each kit's best bass becomes its
pairing (I6 "each kit carries its paired bass"). Basses as b1–b3, Latin square:

| Slot | Kit | Loop | 1 | 2 | 3 |
|---|---|---|---|---|---|
| R2.1 | k1 | L6 wistful high 89 | b1 | b2 | b3 |
| R2.2 | k2 | L7 hopeful low 79 | b2 | b3 | b1 |
| R2.3 | k3 | L2 melancholy mid 81 | b3 | b1 | b2 |

**R3, swing and humanize** (Day 2) on the R1 leader with its R2 bass, in separate rounds [CD-10]: swing rounds
with humanize off, humanize rounds at swing 0.55. Swing options `swing_50`, `swing_55`, `swing_58`, `swing_split`
(hat and ohat at 0.60, every other lane at 0.54: I6 "hats 60 with the rest at 54") [CD-11]. Every swing variant
moves every timing lane, the sax included [CD-12]: the sax's `grid` timing is its lane's swing (COMPILER.md §3.7),
so R3a also moves the sax, and S2b is measured at 0.55.

R3a (swing, w1–w4; advisory, §8) [CD-13]:

| Slot | Loop | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| R3a.1 | L9 hopeful high 90 | w1 | w2 | w3 | w4 |
| R3a.2 | L4 wistful low 78 | w4 | w3 | w2 | w1 |

R3b (humanize; n1–n3 drawn among `humanize_off`, `humanize_h1` = `bar` and `humanize_h2` = `loop`; COMPILER.md
§3.6: drums, bass and keys, never the sax), Latin square [CD-14]:

| Slot | Loop | 1 | 2 | 3 |
|---|---|---|---|---|
| R3b.1 | L3 melancholy high 87 | n1 | n2 | n3 |
| R3b.2 | L7 hopeful low 79 | n2 | n3 | n1 |
| R3b.3 | L5 wistful mid 83 | n3 | n1 | n2 |

### 4.3 Vamps (T·Harmony; T·Labeling "a vamp shootout of about 12 vamps takes about 15 min"; I6)
**The sax is muted**, as I6 says ("one groove, no sax"), by a −200 dB `Gain` at the head of the sax chain (§7.3).
A round has one brief, so the 4 vamps of one mood share a round [CD-15]; the line `vamp:` is the only change to
the loop. Chords and spans are I6's; the margin is the mood check of instrument.md §5 (target mode's score minus the
best other mode; pass ≥ 2), recomputed for this plan: all 12 pass. "Borrowed" marks the vamps with a bVI or bVII
chord (P§3 "including borrowed bVI and bVII") [CD-45].

| Mood | Id | Vamp | Margin | Borrowed |
|---|---|---|---|---|
| hopeful | `aruarian` | IIm9 V13 span 2 | 12 | |
| hopeful | `royal_half` | IVmaj7 IIIm7 span 1 | 18 | |
| hopeful | `two_of_us` | IVmaj7 III7 span 2 | 22 | |
| hopeful | `lift` | Imaj9 IVmaj9 span 2 | 20 | |
| wistful | `i_iv` | Im9 IV13 span 2 | 8 | |
| wistful | `pulse` | Im9 IV9 span 1 | 8 | |
| wistful | `step` | Im11 IIm7 span 2 | 4 | |
| wistful | `float` | Im9 bVIImaj7 span 2 | 8 | yes |
| melancholy | `drift` | Im9 bVImaj7 span 2 | 8 | yes |
| melancholy | `plagal` | Im9 IVm9 span 2 | 8 | |
| melancholy | `backdoor` | IVm9 bVII13 span 2 | 16 | yes |
| melancholy | `rise` | bVImaj7 bVII13 span 2 | 16 | yes |

Facts the borrowed-chord report depends on:
- In their own moods, the bVI and bVII chords above have no tone outside the mood's mode (bVII is part of dorian,
  bVI and bVII of aeolian). The only chord in the 12 with a tone outside its mode is `two_of_us`'s III7 (its third,
  8 semitones above the tonic), and it is not a bVI or bVII chord.
- No hopeful vamp has a bVI or bVII chord, so the one mood where they would be borrowed is never auditioned.
- `chords.numerals` is one list for every mood: removing bVI and bVII would make `drift`, `backdoor`, `rise` and
  `float` unwritable.

**Rounds** (Day 2; fixed: the Day-1 leaders on keys, kit and bass; no sax). Per mood, three rows of a 4×4 Latin
square on the mood's three loops, vamps as v1–v4 per mood. The draw maps `plagal` (melancholy) and `float`
(wistful) to v2 or v3 and the rest at random [CD-16]:

| Row | Loops (melancholy · wistful · hopeful) | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 1 | L1 · L4 · L7 (low) | v1 | v2 | v3 | v4 |
| 2 | L2 · L5 · L8 (mid) | v2 | v1 | v4 | v3 |
| 3 | L3 · L6 · L9 (high) | v3 | v4 | v1 | v2 |

Tile sums: v1 6, v2 7, v3 8, v4 9. Every loop keeps "something new every bar" (f_new = 1) under all 4 vamps of its
mood, span 1 included (§5).

### 4.4 Sax (T·Sax, T·Sax realism; I6; A)
| Id | Source (licence) | Preset | Register |
|---|---|---|---|
| `sax_fp_tenor` | FreePats Tenor Saxophone, full SF2 (CC0) | (0, 0) | tenor |
| `sax_weresax_alto` | Karoryfer Weresax v1.003 alto, converted (CC0) | `weresax-alto`, (0, 0) | alto |
| `sax_mtg_tenor` | MTG Solo Saxophones tenor, converted (CC BY 4.0 / 3.0) | `mtg-tenor`, (0, 0) | tenor |
| `sax_mtg_alto` | MTG Solo Saxophones alto, converted | `mtg-alto`, (0, 0) | alto |
| `sax_gu_tenor` | GeneralUser GS | (0, 66) | tenor |
| `sax_gu_alto` | GeneralUser GS | (0, 65) | alto |

**Registers** are instrument.md §3's (tenor play 44–75, comfort 48–72, home 50–61; alto 49–80, 55–77, 57–68). With
the same degrees, the compiler's octave fit puts the alto either on the tenor's notes or an octave above them,
depending on the tonic and the phrase: on **L2, L4 and L8** both phrases sit 12 above the tenor's; on the other six
loops the alto plays the tenor's MIDI notes (54–72, the bottom of its comfort window). §5 lists each loop's range.

**S1, sources** (Day 1) [CD-17]. Tenors as t1–t3, altos as a1–a3. Each round leaves out two sources of one
register, so each source plays 4 rounds, once on every tile; same-register pairs meet 3 times, cross pairs twice.
The three rounds that hold every alto include the one alto-above loop (L4), so every alto is heard once an octave
above the tenor register; every other S1 round is on a loop where the two registers share notes:

| Slot | Loop | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| S1.1 | L4 wistful low 78 | t3 | a1 | a2 | a3 |
| S1.2 | L3 melancholy high 87 | a3 | t1 | t2 | t3 |
| S1.3 | L1 melancholy low 76 | t1 | a2 | a3 | a1 |
| S1.4 | L5 wistful mid 83 | a1 | t2 | t3 | t1 |
| S1.5 | L9 hopeful high 90 | t2 | a3 | a1 | a2 |
| S1.6 | L7 hopeful low 79 | a2 | t3 | t1 | t2 |

Each source covers all three moods.

**S1 final** (Day 2) [CD-18]: the S1 top 2 (§8) as f1, f2, on the three loops S1 did not use (two of them
alto-above). If three or more tie for the two places, the final takes three, as a 3×3 Latin square on the same
loops (+1.0 min):

| Slot | Loop | 1 | 2 |
|---|---|---|---|
| S1f.1 | L2 melancholy mid 81 | f1 | f2 |
| S1f.2 | L6 wistful high 89 | f2 | f1 |
| S1f.3 | L8 hopeful mid 85 | f1 | f2 |

**S2, the leader's variants** (Day 2; P§3 "warm off/on, expression off as a control, timing on-grid / slight lag /
swung") on the S1 leader; if the final picks the other finalist, S2 re-runs on Day 3 with it (+6.1 min, 18
renders). Two kinds of round, each a Latin square [CD-14]: warmth with the control, and timing at warm off [CD-19].
`sax_plain` = warm off, expression on, grid; `sax_warm_on` = `mix.sax_warm: true`; `sax_expr_off` =
`sax_expr.enabled: false`; `timing_lag` = 25 ms; `timing_swung` = 8th swing 0.6 (`dev_gu` values, COMPILER.md
§3.7).

| Slot | Loop | 1 | 2 | 3 |
|---|---|---|---|---|
| S2a.1 (m1–m3 drawn among plain, warm_on, expr_off) | L1 melancholy low 76 | m1 | m2 | m3 |
| S2a.2 | L4 wistful low 78 | m2 | m3 | m1 |
| S2a.3 | L9 hopeful high 90 | m3 | m1 | m2 |
| S2b.1 (g1–g3 drawn among grid, lag, swung) | L3 melancholy high 87 | g1 | g2 | g3 |
| S2b.2 | L5 wistful mid 83 | g2 | g3 | g1 |
| S2b.3 | L7 hopeful low 79 | g3 | g1 | g2 |

### 4.5 Rain (T·Mix; P§3; A)
- **Source:** Freesound 235827 by Metadex (CC0), 48 kHz 24-bit stereo, which Shresth downloads to
  `assets/big/rain/235827__metadex__soft-rain-on-roof-window.wav` (A). Claude converts it once to 44.1 kHz (§7.2).
- **Excerpts** [CD-20]: a bed needs 0.5 s of high-pass warm-up before `start_s`, then N + 1 s, and N is 32 s at
  60 BPM (RENDER.md §3.4), so each candidate is a 33.5 s stretch. Claude's picker (§7.2) takes the three
  non-overlapping stretches with the smallest range of momentary loudness (400 ms blocks), with no claim about what
  they hold; `excerpt_e1` is the one with the smallest range. The page says the recording also has hum, birds and
  distant thunder, which only his ear can place.
- **Fixed:** the Day-2 picks everywhere, with the sax.

W1 (excerpts at Δ 18, the loudest bed under test; z1–z3 drawn among `excerpt_e1`, `excerpt_e2`, `excerpt_e3`)
and W2 (levels on `excerpt_e1`; y1–y4 drawn among `rain_none`, `rain_d30`, `rain_d24`, `rain_d18`), Latin
squares [CD-14]:

| Slot | Loop | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| W1.1 | L1 melancholy low 76 | z1 | z2 | z3 | |
| W1.2 | L7 hopeful low 79 | z2 | z3 | z1 | |
| W1.3 | L5 wistful mid 83 | z3 | z1 | z2 | |
| W2.1 | L2 melancholy mid 81 | y1 | y2 | y3 | y4 |
| W2.2 | L4 wistful low 78 | y2 | y3 | y4 | y1 |
| W2.3 | L6 wistful high 89 | y3 | y4 | y1 | y2 |
| W2.4 | L8 hopeful mid 85 | y4 | y1 | y2 | y3 |

W1 and W2 render the same night, so W2 cannot know W1's pick; the bed is set by loudness (LUFS_bed = LUFS_music − Δ,
RENDER.md §3.4), so a Δ keeps its meaning on another excerpt.

## 5. Reference loops

Nine loops in `shootouts/loops/`, one per mood × energy cell [CD-3], each with one brief in
`shootouts/loops/briefs.yaml`. They follow TASTE.md's structure: 4/4, 8 bars, a 2-chord vamp from I6's 12 with
every chord a 7th or more (T·Harmony); a sax phrase in bars 1–4 and its answer in bars 5–8, each starting on the
downbeat of bar 1 or 5 on a chord tone, with 4 notes of 8 steps or more over the line, rests in each phrase, and an
answer that starts like the call and ends off the tonic (T·Sax; TASTE.md defaults); kick on 1 and a backbeat on 2
and 4 in every bar; something new every bar through per-bar edits, under every vamp of the loop's mood; bar 8
keeps the base drums (T·Drums "no fill at the seam"); energy as drum density, accents, keys and bass density
(T·Energy). The sax lines follow the same plan at every energy (10–14 notes a phrase), since energy belongs to the
band (instrument.md §8 issue 4, default a).

| File | Mood | Energy | BPM | Key | Vamp | Bass | Base lanes | Sax notes A / B | B ends on | Accidentals | Accents | E | Tenor MIDI | Alto MIDI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `l1_melancholy_low.ls` | melancholy | low | 76 | A aeolian | drift | grid | kick rim hat | 10 / 11 | 2 | b5 | 0 | 0.289 | 55–65 | 55–65 |
| `l2_melancholy_mid.ls` | melancholy | mid | 81 | E aeolian | backdoor | grid | kick snare hat | 13 / 14 | 3 | b5 | 3 | 0.462 | 52–64 | 64–76 |
| `l3_melancholy_high.ls` | melancholy | high | 87 | C aeolian | rise | grid | kick snare hat shaker | 14 / 14 | 4 | b5 | 19 | 0.720 | 58–70 | 58–70 |
| `l4_wistful_low.ls` | wistful | low | 78 | G dorian | float | lock | kick rim hat | 13 / 11 | 7 | | 0 | 0.297 | 55–65 | 67–77 |
| `l5_wistful_mid.ls` | wistful | mid | 83 | D dorian | step | grid | kick snare hat ohat | 12 / 14 | 7 | b5 | 4 | 0.478 | 59–68 | 59–68 |
| `l6_wistful_high.ls` | wistful | high | 89 | B dorian | pulse | grid | kick snare hat shaker | 13 / 13 | 4 | | 16 | 0.717 | 54–71 | 54–71 |
| `l7_hopeful_low.ls` | hopeful | low | 79 | F ionian | lift | grid | kick rim hat shaker | 13 / 14 | 6 | b3 b7 | 0 | 0.294 | 55–72 | 55–72 |
| `l8_hopeful_mid.ls` | hopeful | mid | 85 | Eb ionian | aruarian | lock | kick snare hat | 14 / 13 | 2 | b5 | 3 | 0.465 | 51–65 | 63–77 |
| `l9_hopeful_high.ls` | hopeful | high | 90 | Bb ionian | royal_half | grid | kick snare hat shaker | 14 / 13 | 3 | b7 | 16 | 0.700 | 58–70 | 58–70 |

- **E** is instrument.md §5's energy score, computed from the text. The real bands are tertiles of the seed pool and
  are not frozen; against the §5 fixture's test bands (.30 / .50) the low, mid and high loops fall in their bands
  [CD-5]. Accents (`X` cells) rise with energy: 0 in the low loops, 3–4 in the mid, 16–19 in the high (T·Energy
  "harder accents").
- **Sax register**, after the compiler's per-phrase octave fit (COMPILER.md §3.2): the tenor lead (`dev_gu`) plays
  MIDI 51–72 over all nine, inside its comfort window; the alto lead (a scratch copy of `dev_gu` with
  `sax_lead: alto`) plays 54–77. Under both leads, A and B take the same octave shift in every loop, and nothing
  folds.

**Changed in review (2026-10-31):**
- **Span-1 vamps** (`pulse`, `royal_half`) used to repeat bars 1–2 in bars 3–4 of L4 and L7 (f_new 0.714) and bar 2
  in bar 4 of L5 and L8 (0.857). New edits, each keeping its lane's hit count: L4 `@3 hat`, `@4 kick`; L7 `@3 hat`,
  `@4 kick`; L5 `@4 hat`; L8 `@4 kick`.
- **Alto split:** L4's `q1'` and L8's `q.3'` in bar 8 reached MIDI 79 under the alto, above its comfort top of 77,
  so the octave fit put the call an octave above the answer. They are now `q5` and `q.7`; B still ends off the
  tonic.
- **Accents in the mid loops:** L2 snare `X` on beat 4 of its second base bar, L5 snare `X` on beat 2 of its first,
  L8 kick `X` on beat 1 of its first.
- **Bar-8 changes:** L2 and L8 now shift a comp strike in bar 8 instead of adding one on step 15, so three loops
  (L1, L3, L5) keep a step-15 strike before the seam.

**Names** [CD-1]. Every loop says `keys: ep rootless` and `drums: dusty`, and none has a `sax:` line [CD-2]. Each
shootout palette names its single keys voice `ep` and its single kit `dusty` and puts the candidate (or the fixed
instrument) under those names, with no aliases; the sax entry is named by its register and set as `sax_lead`. The
DSL has no bass name: the kit carries its bass (I6), so the bass candidate goes into `kits.dusty.bass`. The loop text
is then the same in every clip of a round.

**Substitutions** a round may make, and nothing else: `keys: ep spread` (K2) and the `vamp:` line (vamps). Anchors
add the sour copy (§2.5).

**Checks** (2026-10-31, laptop; `dev_gu` at sha256 `b2e4a340…`, with other workflows' uncommitted compiler and
voicing changes in the tree, which the parse results do not depend on):

```bash
PYTHONPATH=src .venv/bin/python -c "
import yaml, pathlib
from lofi_rl.dsl.parse import parse
from lofi_rl.palette import load_palette
v = load_palette('palettes/dev_gu.yaml').vocab()
for name, b in yaml.safe_load(open('shootouts/loops/briefs.yaml'))['loops'].items():
    t = pathlib.Path(f'shootouts/loops/{name}.ls').read_text()
    for mood in ('melancholy', 'wistful', 'hopeful'):
        r = parse(t, vocab=v, mode=mood)
        print(name, mood, 'ok' if r.gate.ok else 'FAIL', r.gate.code, [x.code for x in r.repairs], r.penalty)
"
```

- **Gates:** all 27 lines print `ok None ['alias_name', 'alias_name'] 0.0`: every loop passes G1–G6 in every mood,
  with two free aliases (`ep` → rhodes, `dusty` → jazz in `dev_gu`) and P_fix 0. Under the default vocabulary,
  where `ep` and `dusty` are names, each parses with no repair and prints back unchanged from `canon`.
- **Under every vamp of its mood** (36 texts): G1–G6 pass in all three moods; f_new 1, f_seam 1, no bar repeated
  (P_rep's largest class is 1), no chromatic penalty; each compiles with no fold under both leads, A and B on the
  same octave shift. With a substituted vamp the sax's first note is not always a chord tone (S 0.875 for L8 under
  `royal_half` and `two_of_us`, L9 under `lift`); the sax is muted in vamp rounds.
- **Under every voicing** (36 texts): G1–G6 pass in all three moods; each compiles with no repair.
- **Own vamp:** groove G 1 (kick–bass lock 20/24 in L5, full elsewhere), sax score S 1, f_new 1, f_seam 1,
  P_chrom 0, theory 1 in every loop.
- **Sour copies** (anchors): all 9 pass G1–G6 in all three moods, with 4–12 non-free notes each (P_chrom 0.3) and no
  folds; the octave fit then splits the phrases of L4 and L8 under the alto and of L7 under the tenor.
- **Not checked:** gate G7. Nothing is rendered yet; job 1's measuring pass builds every loop's stems and its clip
  renders check G7.

## 6. Batches and pushes

Contents, not order: the order inside each batch is drawn (§2.3).

| Date | Batch (`inbox/<date>/<batch>/`) | Rounds | Category | Note | est_min (plan / full) |
|---|---|---|---|---|---|
| 2026-11-03 | `b1103-1-keys` | K1 ×6 | keys | Which keys do you like best? | 7.8 / 9.3 |
| 2026-11-03 | `b1103-2-groove` | R1 ×3, R2 ×3 | kits, bass (per round) | Which drums and bass do you like best? | 6.0 / 6.9 |
| 2026-11-03 | `b1103-3-sax` | S1 ×6, anchor ×1 | sax | Which sax do you like best? | 9.2 / 11.1 |
| 2026-11-04 | `b1104-1-groove` | R3a ×2, R3b ×3 | swing, humanize | Which groove do you like best? | 5.6 / 6.5 |
| 2026-11-04 | `b1104-2-sax` | final ×3, S2a ×3, S2b ×3, anchor ×1 | sax, sax_warmth, sax_timing | Which sax do you like best? | 9.2 / 10.5 |
| 2026-11-04 | `b1104-3-keys` | K2 ×2 | keys | Which keys do you like best? | 1.9 / 2.2 |
| 2026-11-04 | `b1104-4-vamps` | V ×9 | vamps | Which chords do you like best? There is no sax in this batch. | 11.6 / 13.9 |
| 2026-11-06 | `b1106-1-mix`, `b1106-2-mix` | W1 ×3, W2 ×4, anchor ×1 and the 5 repeats, split into two batches of about equal length (+ the S2 re-run) | per round | Pick the best and the worst in each round. | about 7.7 / 9.1 each |

- **Pushes:** Nov 3: `b1103-1` + `b1103-2` (13.8 min), then `b1103-3` (9.2). Nov 4: `b1104-1` + `b1104-2` (14.7),
  then `b1104-3` + `b1104-4` (13.6). Nov 6: one push per batch.
- **In the inbox:** `manifest.json` (public, RATING.md v1: `mode: shootout`, `scene: "city at night"` and `bpm`
  per round, `bars: 8`, `sr: 44100`, `note`, `est_min`, opaque round and clip ids, each FLAC's sha256),
  `manifest.hidden.json` (§2.2 fields: per clip its own fader, `fader_db`, and its bus's measured loudness,
  `bus_lufs` and `bus_lu`, CD-52) and one FLAC per clip appearance. A repeat round gets new opaque ids for the same
  renders (R: "a clip shown twice gets two ids"). Nothing else.
- **On the cluster only:** `$REMOTE_SCR/runs/shootout-<job>/renders/<render id>/` (render WAV, music WAV, render
  manifest, FLAC) and `.../summary/` (`levels.json`, `draw.json`, `leaders.json`, the generated palettes,
  `renders.json` mapping render id → loop, brief, palette, variant, trims, faders and hashes, and `batches.json`
  mapping opaque id → render id). `make pull` brings back `summary/` and the inbox, never the renders; `runs/` is
  never committed.
- **Options carry their stage's prefix** (`kit_`, `bass_`, `swing_`, `humanize_`, `timing_`, `rain_`, ...), so the
  audit's single option list per batch splits cleanly by category when one batch holds several stages.

## 7. Tooling to build

What the pipeline lacks today, from the second review's shootout-readiness findings (CHANGELOG 2026-10-31 11:30) and
this plan's reviews. Each item is the next build's spec: an interface and acceptance tests. Tests follow the repo's
rules: `pytest -q` on the laptop under 60 s, cluster-only cases marked `cluster`.

**Dependency, not on this list:** the SFZ → SF2 converter, being built on branch `converter`
(`python -m lofi_rl.convert <bank-id> --big assets/big --out <file.sf2>`; banks `weresax-alto`, `mtg-tenor`,
`mtg-alto`, `jrhodes3d-mono`, `vcsl-upright-knight`, `vsco2-contrabass-pizz`, `gsullivan-wurlitzer`, one preset each
at (0, 0)). Its outputs become fetch-mode sources with `converted_from` set, under `assets/big/derived/`.

### 7.1 Level matching (§3)
- **Tool:** `python -m lofi_rl.shootout.levels --plan shootouts/plan.yaml --loops shootouts/loops --palettes <run>/summary/palettes/measure --stage day1|day2|day3 [--draw <run>/summary/draw.json] [--prior <levels.json>] --out <run>/summary/levels.json [--workers 8]`.
  It runs §3 steps 1–6 at stage level (stems and bus chains; no master chain, bed or loudness loop), several buses
  per run, and writes each variant's t and median F, every clip's own fader [CD-52], and the drift keys. With
  `--draw` (the job passes it) it also measures every text a drawn clip plays that the 9 loops do not hold. With
  `--prior` it compares drift keys and re-measures on any difference.
- **Palette fields** (needed only for the frozen palette, by Nov 6): optional `trim_db` and `fader_db` (−40 … +40,
  default 0) on every `keys.<name>`, `kits.<name>` (the drums bus), `kits.<name>.bass` and `sax.<name>`. `trim_db`
  scales the instrument's bus input before its chain (a kit's whole drum stem); the gain after a bus's chain becomes
  `mix.bus_gain_db[bus]` + the playing instrument's `fader_db`. The render manifest's `buses[bus]` gains
  `instrument`, `trim_db` and `fader_db`.
- **Acceptance tests:**
  1. With every field at 0, all golden render hashes are unchanged.
  2. `trim_db: t` on a FluidSynth voice gives the same bus output (within 1e-7) as a `[Gain, {gain_db: t}]` at the
     head of that bus's chain; on a kit, the same as adding t to every lane's `db`.
  3. `fader_db: 6` on the keys voice raises only the keys bus after its chain, by 6.000 dB (pre-fader
     `buses.keys.lufs` unchanged to 1e-9); the other stems' hashes are unchanged.
  4. t and F from a synthetic table of loudness values, with `statistics.median` for even counts; GU instruments
     get t = 0; the sax target comes from the named anchor instrument.
  5. The linear-range check passes a trimmed GU stem and flags a synthetic input that drives the compressor.
  6. After matching, every instrument's median post-fader bus loudness at its median F equals S\* + T within
     0.05 LU, and every clip at its own fader sits at S\* + T within 0.1 LU; a clip palette's render reports the
     tool's L before every fader and the clip's own fader on its varied bus (`-m cluster`, two loops). Until
     CD-52 the second half read "every loop within 0.1 LU of its own L + F", which `kit_virtuosity` missed on L1
     by 0.0007 LU (0.1007, the −70 LUFS gate).
  7. A changed base-palette sha or compiled digest makes `--prior` re-measure, and both values are recorded.
  8. A value outside −40 … +40, or a field on an unknown key, is a `PaletteError`.
- **Built (2026-10-31, the tool only; the palette fields and tests 1–3 and 8 wait for the frozen palette):**
  `src/lofi_rl/shootout/levels.py`, tests 4–7 in `tests/test_shootout_levels.py` (6 is `-m cluster`). Choices the
  spec left open:
  - **P is read 40 dB up** (`input_lufs`). BS.1770's −70 LUFS absolute gate is its only level-dependent part. Plain,
    the VSCO 2 pizz bass input reads −inf on L4 and L6 and up to 7.37 LU high on the other loops (the gate keeps only
    its loudest blocks), and the sparse GU drums 0.38 LU high on L1. L and the post-fader levels stay plain, as the
    render manifest measures them.
  - **Check (b)** holds only the median (0.05 LU at the median F). The per-loop distance from L + F through the gate
    (`worst_loop_lu`) is kept as information since CD-52; check (c) holds every clip at its own fader to
    `levels.clip_tol_lu`, 0.1 LU (plan.yaml; it replaced `loop_tol_lu`).
  - **GU instruments get t = 0** for every instrument of the anchor's source (`levels.zero_trim_source: gu`), so
    the GU organ and the GU alto too.
  - **One run per variant.** Run m1 measures the four anchors, and each later run one new variant per bus. A
    variant is measured only in its first run: a bus whose variant repeats is not rendered.
  - **`--prior`** compares the stamp, the base sha256, every (run, case) compiled digest and each variant's source
    and kit-map sha256. Equal keys reuse every value. Any difference re-measures everything and records the prior's
    keys, the changed ones, and each variant's prior t and F and its move; moves above 0.1 dB are listed.
  - **Clip faders (CD-52, 2026-10-31):** `clip_fader` sets F(*i*, ℓ) = S\* + T − L(*i*, ℓ), measures the bus after
    it (the kept chain output scaled in float64, as `mix.mix_buses` sums it into the master), and corrects F by the
    miss until it is within `CLIP_FIX_LU` (0.005 LU, half a 0.01 dB step), at most `CLIP_STEPS` (3) times, keeping
    the fader measured closest. Every (variant, loop) of the 9 gets one, so `levels.json`'s `clips` has them all;
    `clip_check` is check (c). The measurements run in the tool's thread pool beside the median ones.
  - **Drawn texts (`--draw`):** every render whose draw palette names a `level` (§7.6) must play a loop that
    `--loops` holds, with its brief. A render of the loop's own text is measured already; any other text becomes an
    extra case `<loop>.<variant>`: on Day 1 the anchor's sour copy, `<loop>.sour`, measured once in its candidate's
    run, on that bus, with the candidate's trim, and kept out of P, t, the median F and check (a). A draw of another
    plan or stage, or a render on a loop `--loops` does not hold, exits 3 before anything is measured. `levels.json`
    records the draw's sha256 and each extra case (its loop, variant, candidates, text sha256, brief and n).
  - **Tests** (`tests/test_shootout_levels.py`): the median check, `clip_fader` on a gated, a plain and a growing
    gate and out of corrections, `clip_check`, the tool's clips on a table bench with a +0.02 LU gate, a draw's sour
    copy staged once in its candidate's run, the refusals, and test 6 (`-m cluster`).
  - **Loops** are those that `--loops`'s briefs.yaml lists. The stamp is `<repo>/.sync-stamp` (null on the laptop).
  - **Checked against the real banks** (laptop, 9 loops, dev band and every converted and FreePats bank, the two
    file kits left out until §7.4 lands): every linear-range and post-fader check passes. A render of a generated
    palette reports the tool's L on every bus exactly (`-m cluster`). With the two §7.4 kits and per-clip faders,
    the Day-1 dry runs' numbers are in §3 step 8 and §7.7.

### 7.2 One-time 48 → 44.1 kHz conversion, and the rain excerpt picker
- **For:** the Virtuosity one-shots (720 files, 48 kHz 24-bit: A) via the importer, and the rain file. The sampler
  and the bed loader take 44.1 kHz only (RENDER.md §3.2, §9). MTG's 48 kHz samples go into an SF2, whose sample
  headers carry their rate, so they need no conversion.
- **Resampler:** `lofi_rl.convert.resample.to_44k(x: np.ndarray, sr: int) -> np.ndarray` and
  `python -m lofi_rl.convert.resample <in.wav> <out.wav>`: `scipy.signal.resample_poly(x, 147, 160, axis=0)`
  (scipy's default Kaiser window, β = 5) in float64 on the whole signal, written as float32 WAV with no PEAK chunk
  (`wavio.write_float`) [CD-36]. Run once on the laptop; outputs go to `assets/big/derived/`, their sha256 into a
  lock file and into the palettes that use them, and to the cluster with `make push-assets`.
- **Rain excerpt picker:** `python -m lofi_rl.convert.rain_excerpts <rain at 44.1 kHz> --n 3 --out shootouts/rain_excerpts.json`
  writes the `start_s` of the three non-overlapping 33.5 s stretches (0.5 s warm-up, 32 s, 1 s crossfade) with the
  smallest range of momentary loudness in 400 ms blocks, e1 first (§4.5). Committed before job 3.
- **Acceptance tests:** resampled length ⌈147·n/160⌉; a 1 kHz sine at 48 kHz comes out at 1 kHz (FFT peak within
  one bin) and within 0.01 dB of its level; two runs give identical bytes; 44.1 kHz input is refused; the bed
  loader accepts the converted rain file and refuses the original. The picker returns three stretches that are
  pairwise disjoint (warm-ups included), each at least 33.5 s and inside the file; it is deterministic; on a
  synthetic file with one steady stretch it returns that stretch as e1; a file shorter than 3 × 33.5 s is refused.
- **Built 2026-10-31, the resampler only** (`src/lofi_rl/convert/resample.py`, `tests/test_resample.py`; docs/KITS.md
  §2): every resampler test above passes. The rain picker is not built.

### 7.3 Sax mute for the vamp rounds (no code)
- The palette generator (§7.8) writes the vamp palettes with `[Gain, {gain_db: -200.0}]` at the head of the sax
  chain, so the sax bus is silent while the sax stem still renders; `what` says "sax muted". During review a palette
  built this way rendered L1 at −14.00 LUFS with G7 passing and a null sax bus loudness. A `render(..., mute=)`
  option is not needed.
- **Acceptance tests:** a vamp palette loads with `verify=True`; its render passes G7 at −14 ± 0.1 LUFS with
  `buses.sax.lufs` null; the keys, bass and drums stem hashes equal the unmuted palette's; two renders are
  identical.

### 7.4 One-shot kit importer
- **Interface:** `python -m lofi_rl.convert.kit shootouts/kits/<kit>.yaml --big assets/big --out assets/big/derived/kits/<kit>.tar`
  writes a tar of 44.1 kHz mono float32 WAV one-shots and prints the palette's `lanes:` block (one `{src, file}`
  per layer and round robin) and its `choke:` list, which is `dev_gu`'s `[[ohat, hat], [ohat, ohat]]` for every
  kit. A tar, because `verify=True` cannot hash a directory and `.rsyncignore` drops loose WAVs (RENDER.md §10).
- **Per one-shot** [CD-35]: read each chosen mic's file; stereo → (L + R)/2, as the sampler does anyway (RENDER.md
  §3.2); sum the mics with their dB gains, keeping their recorded timing; resample if 48 kHz (§7.2); trim the
  lead-in at the first sample within −30 dB of the one-shot's peak (−20 dB for shaker lanes), keeping 1 ms before
  it with a raised-cosine fade-in; drop trailing digital silence. Tar members sorted, with mtime 0, uid/gid 0, mode
  0644 and no pax headers, so the archive is byte-deterministic.
- **What the files hold** (read 2026-10-31 from the folders under `assets/big/`): Virtuosity's kickmic, snaremic
  and lofi files are mono, its oh, mid and room files stereo, all 48 kHz 24-bit, and its perc shakers 44.1 kHz;
  Swirly is mono 16-bit 44.1 kHz; the VCSL shakers are stereo 44.1 kHz. A file first comes within −20 dB of its
  peak up to 699 samples in for Virtuosity (48 kHz, the room mic latest), 416 for Swirly and 2,497 for the VCSL
  shakers: the lead-in the trim removes.
- **Kit maps** [CD-32–CD-34]. Layers soft / med / hard take velocity layers ⌊0.25N + ½⌋, ⌊0.6N + ½⌋ and
  ⌊0.9N + ½⌋ of a set's N [CD-33]; a set with no round robins uses the layer and its two neighbours:

  | Lane | Virtuosity (kickmic + snaremic + oh) | Swirly |
  |---|---|---|
  | kick | `kick_snon` vl 1 / 2 / 4, rr 1–4 | `marching_kick` beater + reso, vl 6 / 13 / 20, rr 1–4 |
  | snare | `snare_center` vl 9 / 22 / 32 (± 1 as rr) | `snare_main` (`snare_hit`) top + btm, vl 3 / 7 / 11, rr 1–4 |
  | rim | `snare_crossstick` vl 4 / 10 / 14 (± 1) | `snare_edge` top + btm, vl 3 / 6 / 9, rr 1–4 (A) |
  | hat | `hh_closed` vl 1 / 2 / 4, rr 1–4 | `hat_closed` vl 3 / 7 / 11, rr 1–4 |
  | ohat | `hh_open` vl 1 / 2 / 4, rr 1–3 | `hat_open` vl 4 / 8 / 13, rr 1–4 |
  | shaker | its perc shaker, close + oh mics, Shake1D/1U rr 1–7, D and U alternating, all layers | VCSL Shaker, Large: Shake1D rr2, Shake1U rr1, Shake1D rr3, Shake1U rr4, all layers |

  Not used: Virtuosity's ride and rimshot; Swirly's `snare_stir` (a 6–13 s sustained stir no step lane can play),
  `hat_tight` and `hat_foot`. Lane `pan` stays 0.
- **Lane levels.** The kit's trim (§3) goes on every lane's `db`. A borrowed lane (Swirly's VCSL shaker,
  Virtuosity's separately recorded perc shaker) also gets an offset o = (GU shaker − GU snare) − (kit shaker − kit
  snare), each term the loudness of that lane alone playing `x` on every 8th note for 2 bars at 85 BPM through the
  sampler [CD-30]. The kit's own lanes keep their recorded balance.
- **Acceptance tests:** two runs give identical tar bytes; every member is 44.1 kHz mono float32; every member's
  first sample within −30 dB of its peak (−20 dB for shakers) sits at sample 44 ± 1; a synthetic two-mic set with
  impulses 100 samples apart at 48 kHz comes out 92 ± 1 samples apart (the timing kept); stereo input becomes
  (L + R)/2; the printed choke list equals `dev_gu`'s; a borrowed lane's offset reproduces the GU shaker-to-snare
  difference within 0.1 LU on the test grid; a palette with the printed blocks loads with `verify=True`, with every
  lane of all six non-empty.
- **Built 2026-10-31** (`src/lofi_rl/convert/kit.py`, `shootouts/kits/kit_virtuosity.yaml` and `kit_swirly.yaml`,
  `tests/test_kit_import.py`; docs/KITS.md §3): every test above passes, on synthetic kits and on the real banks.
  Swirly's VCSL shaker lane is inside `kit_swirly.tar`. The two archives' sha256s, counts and lead-in trims are in
  docs/KITS.md §3.2; they are not yet pinned in `assets/MANIFEST.yaml` or under `assets/big/derived/kits/`.

### 7.5 SF2 coverage checks at palette load
- **Interface:** `palette.sf2_coverage(path, bank, prog) -> frozenset[(key, vel)]` from the SoundFont's preset and
  instrument zones (key and velocity ranges, global zones included); `load_palette(verify=True)` refuses a palette
  in which a voice cannot sound a note the compiler may send: keys 48–72 at the voice's `vel` values; kit basses
  28–43 at theirs; each sax's `play` range at every velocity 60–112 (the sax's velocity clip, COMPILER.md §3.5);
  each SoundFont one-shot's note at its `vel`. The error lists the missing (key, velocity) cells.
- **Known today** (read from the files, 2026-10-31): FreePats tenor 43–89, FM Piano #1 27–98 (all 4 presets),
  Upright KW 21–108, FSBS jazz guitar 35–86, Finger Bass YR 26–45, at every velocity tested; GU presets 4, 16, 32,
  65 and 66 cover 0–108 or more. The converted banks are checked when they exist.
- **Acceptance tests:** the GU presets and the five archive SF2s above pass; a synthetic SF2 with no zone at key 60
  fails naming 60; a preset whose zones stop at velocity 63 fails at 104; a sax whose zones cover 48–72 fails for
  `play: [44, 75]` with 44–47 and 73–75 listed.
- **Built 2026-10-31** (`src/lofi_rl/palette.py`, `tests/test_sf2_coverage.py`; docs/KITS.md §1): every test above
  passes, the seven converted banks pass in their roles, and the reader agrees with FluidSynth 2.6.1 on global zones
  and generator order. With humanize on, the keys and bass velocities checked widen by the humanize range.

### 7.6 Batch builder
- **Interface:** `python -m lofi_rl.shootout.batch --plan shootouts/plan.yaml --stage day1|day2|day3 --renders <run>/renders --out <inbox root> [--prior <run>/summary/batches.json ...] [--seed <hex>]`
  builds every batch of the stage. It draws (§2.3) from `secrets.randbits(64)` unless `--seed` is given (tests
  only), copies each clip's FLAC under an opaque name, writes `manifest.json` and `manifest.hidden.json`, records
  the draw in `<run>/summary/draw.json` and the id map in `batches.json`, then runs
  `python -m lofi_rl.ui.manifest check` on each folder. Opaque ids are random per batch (`k` + 10 hex for clips,
  `r` + 8 hex for rounds).
- **Anchors:** the builder writes the sour copy of the anchor round's loop (§2.5), gate-checks it (G1–G6 in its
  mood, P_chrom ≥ 0.2) and has the job render it.
- **Repeats** (day3): read from the `--prior` maps. Each reuses the first showing's render (same real clip ids) under
  new opaque ids, `kind: repeat`, `of` = the first showing's opaque round id, tiles reversed. A render that scratch
  has purged is rendered again and must match its recorded `pcm_sha256`.
- **Acceptance tests:**
  1. The folder holds only the FLACs and the two manifests; `ui.manifest check` passes (leak check, one pass per
     clip, equal lengths per round, one sample rate).
  2. `plan.yaml` places only symbols on tiles; no option id appears in a round's tiles.
  3. Two seeds give different option-to-tile maps in every stage; one seed gives byte-identical batches.
  4. Every draw meets its stage's constraints: K1's family sums of 15; one alto-above S1 round per alto; plagal and
     float on v2 or v3; the Latin squares; R3a reversed; no two consecutive rounds on one loop; a design round
     first.
  5. Every repeat's `of` resolves to a round of a `--prior` batch, its real ids equal the first showing's, and its
     tiles are reversed; no repeat is drawn from a stage that `--repeats` does not name.
  6. An anchor round has `kind: anchor`, an `anchor_clip`, and a gate-passing sour copy.
  7. No hidden value appears in the public manifest: a test searches it for every option, loop and palette name,
     with `ui/manifest.py`'s category-word exemption (`b1103-3-sax` and its note may say "sax").
  8. A missing render, a failed gate or a clip that is not one pass stops the build with nothing written.
- **Built (2026-10-31):** `src/lofi_rl/shootout/{draw,batch,sour,render}.py` and their tests
  (`tests/test_shootout_{draw,batch,sour,e2e}.py`); every test above passes. The interface is two steps,
  `python -m lofi_rl.shootout.batch draw --plan … --stage … --summary <run>/summary [--leaders …] [--prior …]
  [--repeats …] [--rerun-s2 …] [--cut …] [--seed <hex>]` and `… batch build --plan … --stage … --renders
  <run>/renders --out <inbox root> [--draw …]`, because the palettes and the renders depend on the draw (R2 plays
  each bass on its round's kit, an anchor's sour copy must be rendered). The draw keys every random choice by the
  seed and a label (stage, batch, group), so a seed gives the same draw anywhere; its `palettes` table names each
  palette the renders need by key (`<stage>_<option>`, R2 `r2_<kit>_<bass>`) and the band it plays.
- **One palette per clip (CD-52, 2026-10-31):** a render of a stage whose option is an instrument (it fills the
  keys, kit, bass or sax slot: K1, R1, R2, S1, and K2 and the S1 final on Day 2) plays its own palette,
  `<base>_<loop id>` or, for a changed text, `<base>_<loop id>_<variant>` (an anchor's sour copy: `…_sour`), where
  `base` is the stage palette above; its entry adds `base` and `level` (the bus, the option, and the case its fader
  is measured on: the loop's name, or `<loop>.<variant>`). Day 1 asks for 67 palettes, one per render, on 26
  bases. Other stages keep one palette per option. The builder refuses a clip whose palette names a `level` and
  whose record carries none for that case, or one whose level check failed (§7.7), and writes each clip's own
  `fader_db`, its measured `bus_lufs` and `bus_lu` into the hidden manifest (§2.2).

### 7.7 Cluster job type `lofi-shootout`
- **Script:** `jobs/shootout.sbatch`, job name `lofi-shootout`, partition `main`, 8 CPU, 16 GB, 20 min (was 30;
  set from the dry run below), logs to `/scratch1/%u/lofi-rl/logs/%x_%j.out`, `MODE=scorer source
  scripts/job_env.sh` (so `LOFI_ASSETS` and `LOFI_CACHE` are set). Header cost line: 8 CPU + 4 (16 GB) =
  12 SU/min, **at most 240 SU**. Claude registers it (`make approve JOB=jobs/shootout.sbatch MAX=240 BY=Claude`;
  ≤ 500 SU).
- **Submitted only by** `make submit JOB=jobs/shootout.sbatch ARGS='--stage day2 --leaders keys=<id>+<id>+<id>/kit=<id>/bass=<id>/sax=<id>+<id>'`.
  Day 2 takes the K1 top 3 as keys (4 on a tie at third place, §4.1) and the S1 top 2 as sax (3 on a tie, §4.4),
  leader first, and one kit and one bass; with the `final` cut, sax takes one id. Day 3 takes one id per slot,
  `ARGS='--stage day3 --leaders keys=<id>/kit=<id>/bass=<id>/sax=<id>'`, and also names `swing`, `humanize`,
  `sax_variant` and `timing` when a Day-2 pick is not the plan's default (`swing_55`, `humanize_off`, `sax_plain`,
  `timing_grid`). `ARGS` reach the script as arguments, so they need no comma-safe `--export`; they hold no spaces.
  Day 3 adds `--prior <job>+<job>`, `--repeats <stage>+...` (the stages first heard at least 2 days before Nov 6),
  and, if needed, `--rerun-s2 <sax id>`. A cut adds `--cut vamps-high+latin3+final`. Before any stage the job checks
  the leaders as the draw does (slots, ids and counts, after the cuts), so a wrong form exits 2, under `--dry-run`
  too; then it writes all of it into `summary/leaders.json`. `scripts/submit.sh` refuses a dirty or unsynced tree
  (exit 3; `ALLOW_DIRTY=1` would ship other workflows' edits), so each job waits for a clean, synced tree.
- **Stages** (`python -m lofi_rl.shootout.job`): the draw (§7.6) → generator `--levels none` (§7.8) → levels
  (§7.1) with the draw, so every clip's own fader is measured, the anchor's sour copy included [CD-52] → generator
  with the levels and the draw (a palette per clip) → clip renders, 8 at a time with 1 worker each → FLACs →
  batches (§7.6) → `SHOOTOUT_OK`. Every palette source is hashed once per stage process, not once per palette. Any
  gate or compile failure fails the job before any batch is written, and so does a clip off its target after
  matching (the generator refuses it) or a render whose bus is not the one the tool measured (render status
  `level`: its pre-fader loudness more than 0.001 LU from the tool's L, or another fader). Each stage's exit code
  and wall time go into `summary/stages.json` and the log.
- **SU estimate:**

  | Run | Measuring runs (stage level) | Clip renders | Wall at 8 in parallel, + 2 min | Expected SU (12/min) |
  |---|---|---|---|---|
  | day1 (Nov 2) | 72 (8 per loop) | 67 | 3.9–10.9 min (from the dry runs below; first estimate 3.0–4.1) | 46–131 |
  | day2 (Nov 3) | 27 (3 per loop) | 84 | 2.9–3.8 min | 34–45 |
  | day3 (Nov 5) | re-measures only on drift | 26 (+18 for an S2 re-run) | 2.2–2.5 min | 27–30 (+2–4) |

  About 107–206 SU expected for the three runs (days 2 and 3 still on the first basis); at most 720 SU (3 × 240)
  if each used its full walltime; a re-run costs about one day2 run. **First basis:** the cluster render check took
  2.86–3.66 s per fixture on EPYC and 4.94–5.95 s on Xeon 4116 (CLI, 4 workers, 82 BPM, start-up included;
  `runs/render-check-{13630298,13630299,13630300,13630522}/summary/fixtures.json`); × 1.35 for 1 worker (RENDER.md
  §7: 1.54 against 1.14 s) and × 1.08 for 76 BPM gives 4.2–8.7 s a render; 94 % efficiency at 8 workers
  (LESSONS 37); a measuring run counts as 0.6 of a render on Day 1 and 0.3 on Day 2 (one stem); 2 min of fixed cost
  per run.
- **Dry run, median faders** (2026-10-31, laptop, Apple M4 with 10 cores; `python -m lofi_rl.shootout.job --stage
  day1 --jobs 8` with a scratch seed, the real banks and the two §7.4 archives under a scratch `LOFI_ASSETS`;
  twice, the first with an empty one-shot cache): every stage passed. The draw gave 3 batches, 19 rounds (6 + 6 + 7
  with the anchor), 67 renders, 26 palettes and 1 gate-passing sour copy (P_chrom 0.3); the levels tool measured 20
  variants in 8 runs; all 67 renders passed G1–G7 at −14.09 … −13.99 LUFS, true peak at most −1.468 dBTP; each
  batch passed `python -m lofi_rl.ui.manifest check`. The second run gave byte-identical batches (76 files),
  draw.json, palettes (all 34; only the index, which names each run's own files, differs), level values and render
  records. Stage seconds, run 1 / run 2: draw 0.1 / 0.1, measure 0.4 / 0.4, levels 44.8 / 43.8, palettes 0.5 / 0.5,
  render (67) 43.3 / 42.8, build 0.3 / 0.3, all 89.4 / 87.9; a render took 2.98–7.31 s (median 4.71).
- **Dry run, per-clip faders** [CD-52] (2026-10-31, the same laptop, seed, banks, kits and command, the job's own
  runner; three runs, the first with an empty one-shot cache, each into its own scratch inbox and runs folder):
  every stage passed. The draw gave the same 3 batches, 19 rounds and 67 renders, now with 67 palettes, one per
  clip, on 26 bases, and the same gate-passing sour copy (S1 on L9 for an alto, P_chrom 0.3). The levels tool
  measured the 20 variants in 8 runs and 181 clip faders, the sour copy the one extra case; the generator wrote 93
  palettes (26 stage palettes, 67 clips' own). All 67 renders passed G1–G7 at −14.099 … −13.996 LUFS, true peak at
  most −1.473 dBTP, and each read the tool's L before its own fader exactly. Each batch passed `python -m
  lofi_rl.ui.manifest check` (b1103-1-keys 6 rounds and 24 clips, b1103-2-groove 6 and 18, b1103-3-sax 7 and 28).
  Runs 2 and 3 each gave run 1's bytes: the batches (76 files), draw.json, batches.json, the palettes (all 101 YAMLs;
  the index differs only in the paths it names), the level values and the render records (and every render file,
  the 134 WAVs included: 402 files; `wavio.write_float` writes no PEAK chunk, so the WAVs carry no timestamp). The per-round spreads are in §3 step 8.

  | Stage | draw | measure | levels | palettes | render (67) | build | all |
  |---|---|---|---|---|---|---|---|
  | seconds, run 1 / 2 / 3 | 0.1 / 0.2 / 0.1 | 0.5 / 0.5 / 0.5 | 46.0 / 44.5 / 44.3 | 0.7 / 0.7 / 0.7 | 49.1 / 47.7 / 44.8 | 0.3 / 0.4 / 0.3 | 96.7 / 94.0 / 90.7 |

  A render took 2.76–8.15 s (median 5.38) in run 1 and 3.41–7.09 s (4.91) in run 3, with 8 at a time; the peak
  memory over every stage was at most 3.0 GB (3,020,521,472 bytes, the independent check's run a). The change costs no measurable time:
  just before run 3, the median-fader code (ae828ae) took 93.6 s on the same seed (levels 45.3, render 47.0),
  against run 3's 90.7, and it reproduced the first dry run's render records exactly; the laptop's own spread over
  the seven Day-1 runs, 86.4–98.7 s, is what moved the estimate (two of them the independent check's own, 86.4 s
  with an empty one-shot cache and 98.7 s warm). The renders took 167 loudness-loop iterations
  against 163 with median faders. On the cluster (× 1.3–2.7 a render, × 2 if the 8 CPUs are 4 cores' hyperthreads,
  + 2 min of start-up) Day 1 comes to about 3.9–10.9 min, 46–131 SU, inside the 20-min walltime, which is 1.8 times
  the slowest.

### 7.8 Shootout plan and palette generator
- **Plan:** `shootouts/plan.yaml` transcribes §4 and §6 in symbols: candidates (source, preset, register, family,
  kit map), each stage's symbols and constraints, round slots (loop, tiles in symbols, substitutions), the batches
  and their contents, the fixed band per day and the cut options. No option is placed on a tile.
- **Generator:** `python -m lofi_rl.shootout.palettes --plan shootouts/plan.yaml --levels <levels.json>|none --draw <draw.json> --out <run>/summary/palettes/`
  writes one palette per (stage, option) from a committed base, `shootouts/palettes/base.yaml`: `dev_gu`'s groove,
  sax expression, chains and loudness blocks (chains pinned to the freeze, §3), every candidate's source, and
  `aliases: {}`, since `palette.py` refuses an alias that is also a palette name ("an alias cannot also be a palette
  name"), and the shootout voices are named `ep` and `dusty`. `--levels none` writes the measuring palettes (trims
  and faders 0). Vamp palettes carry the sax mute (§7.3).
- **Acceptance tests:** every generated palette loads with `verify=True`; `base.yaml` has no `ep` or `dusty`
  alias; every loop parses with no repair under each generated vocabulary; within each round the palettes and loop
  texts differ only in the candidate's block, trim and fader (a test diffs them), which is §2.1's "one thing only";
  every substitution a round makes keeps G1–G6 and f_new = 1 (the `vamp:` lines of its mood); every candidate is
  heard in ≥ 2 distinct loops on distinct tiles within its stage.
- **Built (2026-10-31, Day 1 / job 1):** `shootouts/plan.yaml` (all of §4 and §6, every day), its loader
  `src/lofi_rl/shootout/plan.py`, `shootouts/palettes/base.yaml` and `src/lofi_rl/shootout/palettes.py`. Tests:
  `tests/test_shootout_plan.py` (the doc's tables written out a second time, and the design properties) and
  `tests/test_shootout_palettes.py`. Choices the spec left open:
  - **CLI:** `--stage day1|day2|day3` was added, and `--draw` is optional. The stage palettes do not depend on the
    draw: Day 1 writes all 26, one per (stage, option) and for R2 one per (kit, bass), since each R2 round plays one
    kit, each as `<id>.yaml` (`k1_jrhodes3d`, `r2_kit_swirly_bass_gu_acoustic`), with each instrument's median F.
    With `--draw` (the job passes the batch builder's draw.json) the draw must be this day's draw of this plan, each
    stage's maps must meet its rule (K1 family pairs, S1 registers, `central` for the vamps), and every palette its
    `palettes` table asks for must be one written here with the same band. Only day1 is built; day2 and day3 exit 2.
  - **Clip palettes (CD-52, 2026-10-31):** with `--draw`, each draw palette that names a `level` (§7.6) is written
    as `<key>.yaml`, the file the render driver reads: its base's palette with the varied bus's fader set to the
    levels file's `clips[case].F` for its option (`clip_palette`), every other trim and fader as the base's. Its key
    must be `<base>_<loop id>[_<variant>]` from its base and case. A clip the levels file has no fader for (the
    tool ran without `--draw`), or one more than `clip_tol_lu` from S\* + T, stops the generation before anything is
    written. The index entry adds `base` and `level` (bus, variant, case, the tool's L, F, the loudness measured
    after F, `lu` relative to S\*, the target, the miss and the tolerance), which the render driver copies into its
    record. A test diffs the clip palettes of every drawn round: they differ only in the candidate's block, trim,
    fader and name (§2.1). Day 1 writes 93 palettes: 26 stage palettes and 67 clips' own.
  - **The base** is a loadable palette: the dev band under `ep`, `dusty` and `tenor`, and every SoundFont
    candidate's source, pinned as in assets/MANIFEST.yaml. Each pitched chain starts with a trim slot,
    `[Gain, {gain_db: 0.0}]`, which leaves the output bit-identical at 0 dB. A palette's id names its stage and
    option, the one difference the round diff allows besides the candidate's block, trim and fader.
  - **The §7.4 kits** are read from `$LOFI_ASSETS`: `derived/kits/<kit>.tar` and the importer's printed document
    (`sources`, `choke` and `lanes`) saved as `derived/kits/<kit>.yaml` (plan.yaml `archive`); a printed sources
    entry whose sha256 is not the tar's is refused. Every one-shot must name one source, the tar, which is added to
    every palette with its sha256; the plan's `archive.sha256` pin is checked once it is set (both are null until
    the archives are pinned in assets/MANIFEST.yaml: KITS.md §4). `--without` leaves an unbuilt kit out and records
    it.
  - **Measuring palettes** (`--levels none`): `measure/m1.yaml` … `m8.yaml` and `measure/index.json`.
  - **YAML** is written by libyaml when PyYAML has it; a test checks that it gives the same bytes as the
    pure-Python dumper. Two runs give identical bytes.
- **One plan for every tool (2026-10-31).** The level side and the batch side each wrote a plan; they are now one
  `plan.yaml` and one loader, with `substitute` and `apply_cuts`. Where the two read this doc differently, the plan
  follows the doc: the vamps are one stage `V` (§6 "V ×9", §2.5's one vamps group, CD-24's one warm-up round) with a
  draw per mood on its three loops (`V.1`–`V.9`); the repeat group of R1, R2, R3a and R3b is `drums_bass` (§2.5
  "drums + bass"); the batch categories are §6's words ("kits, bass", "swing, humanize", "sax, sax_warmth,
  sax_timing", and `per_round` for the Nov-6 pair); K2's substitution is the line `keys: ep spread` (§5); R2's kits
  are k1–k3 (§4.2). The day-3 cuts, which this doc does not name, are `w1-two` and `w2-two`.

### 7.9 Tally tool (laptop)
- **Interface:** `python -m lofi_rl.shootout.tally --votes data/votes --inbox inbox [--batch <name> ...]` reads the
  vote logs and the hidden manifests, reruns `ui.audit.aggregate` per batch, and writes `data/votes/audit/<batch>.json`
  and `data/votes/audit/shootouts.json` (private, never committed) with §8's figures per stage: the recount (BW over
  heard appearances, anchor rounds dropped, half weight for a micro-session that missed its anchor), the pooled
  repeat agreement with a Wilson 95 % interval, position-of-best counts pooled by clip count across all shootout
  batches (descriptive), and the flagged rounds of §3.
- **Acceptance tests:** on a synthetic batch its counts equal the audit's, anchor rounds excepted; a best or worst
  mark on a clip heard under 4 bars is left out; repeats are scored for agreement and left out of the counts; a
  missed anchor halves its micro-session's weight.
- **Built (2026-10-31, branch `wt/tally`):** `src/lofi_rl/shootout/tally.py` and `tests/test_shootout_tally.py`. As
  built: `python -m lofi_rl.shootout.tally --votes <dir> --inbox <dir> [--batch <name> ...] [--out <dir>] [--report
  <file.md>] [--plan shootouts/plan.yaml] [--config configs/rating.yaml] [--levels <levels.json> ...]`. `--out`
  defaults to `<votes>/audit/`; inside the repo `--out` and `--report` are written only under `data/`, never in the
  inbox (exit 2 otherwise), since a tally is as private as the votes until Shresth confirms it (§1 step 2).
  - **What it writes.** `<out>/<batch>.json`: the batch's status (`no_votes`, `partial`, `complete`), the audit's
    own aggregate, its flags and every standing vote as the recount read it (options by tile, picks, weight).
    `<out>/shootouts.json`: per stage the recount table, the escape hatches, the anchor rounds dropped, the votes at
    half weight and the marks left out; the K1 + K2 and S1 + final pools; the micro-sessions with anchors; the
    pooled repeat agreement with §2.5's caveat; position of best; the flags and each stage's largest spread; the
    drift moves of `--levels`; each stage's first vote date (job 3's `--repeats`); the rules' outcomes; job 2's
    proposal; and everything that needs Shresth's word. `--report` writes it as markdown for
    `gates/GATE_palette.md`: per stage a table of option, shown, heard, best, worst, BW, net and all-bad rounds,
    per option only (never per tile, so it gives no draw away); a flagged round is named only once its batch has
    every vote; and nothing says which rounds come back or when (RATING.md: he is never told), so the first vote
    dates stay in the JSON and §2.5's caveat appears only once repeats have votes. A batch with no votes is
    reported as such, never as zeros.
  - **Rules as functions**, each applied once every round of its stage's batches has a vote: after Day 1
    `keys_to_k2` (CD-41), `kits_after_r1` (CD-42), `bass_pairings` (CD-43), `sax_to_final` (CD-18, with D-012's
    no-free-sax check) and `day2_cut` (CD-50); after Day 2 `keys_survivors` (CD-40), `sax_final` (with the S2
    re-run note), `warmth` (with the `sax_expr_off` question), `timing`, `swing` (advisory, always put to
    Shresth), `humanize` and `vamp_ranks` (CD-45's borrowed-chord condition); after Day 3 `rain_excerpt` and
    `rain_level` (a `rain_none` lead goes to Shresth). S2a and S2b rounds in a Day-3 batch are tallied apart, as
    the S2 re-run.
  - **Job 2's proposal:** `keys=<K2, the K1 leader first>/kit=<the R1 leader>/bass=<its R2 pairing>/sax=<the
    finalists, the S1 leader first>` (one sax id with the `final` cut), and `--cut` from the cut list. The tool
    stops unless `draw.parse_leaders` reads the string back and `draw.day_band` builds the Day-2 band from it after
    the cuts; the report gives the `make submit` line.
  - **Raised** (`RuleGap`, naming the section; the tally records it, blocks what depends on it and puts it to
    Shresth): after CD-44, fewer than 3 keys left for K2, fewer than 2 kits, or fewer than 2 saxes for the final;
    no option left on a single-winner choice; a kit without exactly one R2 round; keys survivors fully tied across
    the line of the top 2 (unless all of them fit as the third with BW above 0) or for the third place, or fewer
    than 2 of them; every finalist all-bad. The spec gives no rule for these.
  - **Choices the spec left open:** BW = (best − worst) / heard, its three terms over heard appearances and
    weighted, in exact fractions; `shown` counts every appearance in a design round; an option never heard has no
    BW and ranks last. A micro-session missed its anchor when it caught fewer than half of its anchors (with one
    anchor, when it did not catch it), read with `ui.audit.anchor_caught` over every log in `--votes`; all its votes
    count half, in every term. G marks every clip of its round, heard or not, and a worst given with G still counts;
    CD-44 compares weighted all-bad rounds with weighted rounds. A full tie without the default leaves a palette
    value at its default meanwhile (swing, humanize, warmth, timing, rain); where the leader must be one of the
    options (the K1, R1, pooled R2 and S1 leaders, the keys leader over K1 + K2) a seeded draw names a provisional
    one. Every seeded draw is keyed by the sha256 of the stage's standing design-round vote ids (`draw.Rng`), so it
    is fixed once the votes exist and the private logs reproduce it. S1's places: two fully tied for one place go to
    a seeded draw; three or more fully tied at the line give a three-source final, a seeded draw filling it when
    more are tied than places are left. An all-bad key is not sent to K2; with one kit out, the other two stay.
    D-012's check reads the S1 leader before CD-44 removes anything, and "most rounds" as more than half. The cut
    list's limits are 30 and 55 minutes, on the audits' `time.minutes`. CD-45's tie is a BW equal to another vamp's
    of its mood. `sax_expr_off` "has the top BW" when it shares it. Flags: over 1.0 LU for the instrument stages and
    S2a's warmth variants, over 0.5 LU for R3a, R3b and S2b, over 1.0 LU on keys or bass for the vamps, none for
    rain; a level missing where one is checked is a flag too. The rules wait for every round of their batches, and
    job 2's proposal for every round of the Day-1 batches, because the anchor in the last batch can halve a
    micro-session that began in an earlier one. The S2 re-run note compares the final's winner with the S1 leader
    this tally proposes (job 2's own is in its `summary/leaders.json`). The recount's audit-style counts must equal
    `ui.audit.aggregate`'s per option, or the tool stops.
  - **Not built:** Day 3's `--leaders` (it needs job 2's `leaders.json`, which the inbox does not hold; the report
    lists the Day-2 picks it is made of) and Day 3's cut (plan.yaml gives no saving for `w1-two` or `w2-two`).
  - **Tests** (17, on batches the real draw and batch builder make from fake renders, every vote written by the
    rating page's own `Ledger` on a fake clock): the four acceptance tests above; a retracted vote; the Day-2
    proposal waiting for the anchor, and a late missed anchor halving the groove votes of its micro-session; the
    leaders string through `parse_leaders`, `day_band`, the draw and the job's own `--dry-run` check; the cut list
    and the `final` cut's one sax; ties at third place and the S1 places; the all-bad rule on keys and on kits (a
    `RuleGap` that blocks the proposal); a batch with no votes and a partial one; keys survivors, the sax final, the
    variant, vamp and rain rules; Day-2 and Day-3 votes through every later rule; level flags on the instrument,
    timing and vamp limits; the CLI's files and its refusal of a public path. Seven mutations of the recount
    (heard filter, half weight, repeats, anchors, the tie at third, CD-44, G) each fail the test meant for it.
  - **Checked on the real Day-1 batches** (laptop, 2026-10-31; the inbox read-only, synthetic votes in a scratch
    folder, options ranked alphabetically so nothing depends on the draw): 3 batches and 19 rounds read, the S1
    anchor found, no round flagged (per-round spread 0.0012–0.0091 LU, read independently from the hidden
    manifests too), and a proposal of 3 keys, 1 kit, 1 bass and 2 saxes that `draw.day_band` accepts and the job's
    `--dry-run` check passes (exit 0).

### 7.10 Only if Shresth asks for them
- **Per-mood chord list** (if bVI and bVII should leave the hopeful mood only, §11 question 2): a palette field
  `chords.numerals_by_mood` and a G4 check in `parse(..., mode=)`. Test: `vamp: Im9 bVImaj7 span 2` passes in
  melancholy and is G4 in hopeful.
- **A pattern round** (§11 question 3): 2 rounds of 3 patterns (the drum lanes and drum edits of three loops of one
  energy, put into one host loop with `bass: lock`), GU kit, on Day 1 in its own batch before the sax (+2.5 min;
  Nov 3's second push becomes 11.7 min). The generator writes the host-loop variants and gate-checks them.

### 7.11 Order of building
- **By Nov 2 evening (job 1):** the converter, §7.5, §7.2's resampler (Virtuosity), §7.4, §7.8, §7.1's tool (no
  palette field yet), §7.6 for day1 with anchors, §7.7.
- **By Nov 3 evening (job 2):** §7.9 (the first tally is due), `--leaders`, the S1 final, the vamp palettes (§7.3).
- **By Nov 5 afternoon (job 3):** §7.2's rain picker and the converted rain file, §7.6's `--prior` repeats,
  `--rerun-s2`.
- **By Nov 6 (the freeze):** §7.1's palette fields and the frozen palette.

## 8. Tallies and decisions

**The audit's shootout tally** (`ui/audit.py`, `GET /api/audit`, served once every round of a batch has a vote): per
option `shown`, `best`, `worst` and `net` = best − worst, sorted by net, then best; hidden repeats are left out, and
an anchor round's broken clip is skipped while its other clips count; every shown clip counts, heard or not. It
also gives the no-best, no-worst and all-bad counts, the position-of-best chi-square, and repeat agreement with a
Wilson 95 % interval.

**Claude's recount** [CD-38], per stage across its batches:
- A best or worst mark counts only if that clip was heard for 4 bars or more (T·Vote unit: pairs with unheard clips
  are dropped). **BW = (best − worst) / shown**, over heard appearances. Both the audit's net and BW are reported;
  decisions use BW.
- Anchor rounds are dropped; a micro-session that missed its anchor counts at half weight (D-013).
- `T` (no clear best) or `B` (no clear worst) adds no mark. `G` (all bad) marks every clip in the round; an option
  marked all-bad in half of its rounds or more cannot survive [CD-44].
- **Ties** [CD-39]: BW, then more bests, then fewer worsts (which matters only when shown counts differ); then, on a
  single-winner choice, the current default if it is among the tied; a tie without the default is put to Shresth in
  the next push, and the default stays meanwhile.

**Rules per shootout** (Shresth confirms every survivor, §1):

| Shootout | Rule |
|---|---|
| Keys | K1's BW ranks all 8; the top 3 go to K2 (ties at third: up to 4; beyond 4, a seeded draw among the fully tied) [CD-41]. Survivors come only from the K2 participants: the top 2 by BW over K1 + K2, and the third if its BW is above 0 [CD-40]. |
| Kits | All 3 survive unless the last by R1 BW has BW below 0, which leaves 2; a tie for last keeps all 3 [CD-42] (TASTE keeps 2–3). |
| Basses | Each surviving kit takes the best bass of its own R2 round (I6: each kit carries its paired bass); a round with no best (T or G) takes the pooled R2 leader [CD-43]. `gates/GATE_palette.md` states that each pairing rests on one round. |
| Swing | Advisory [CD-13]: swing stays 0.55 (T·Tempo/feel "about 55 % MPC") unless Shresth names another option after seeing the R3a tally, which is reported with the note that two rounds cannot separate four options reliably. |
| Humanize | The R3b BW leader (T·Humanize "decided by the shootout"); ties as above, the default being off. |
| Vamps | Per mood, the 4 vamps ranked by BW: seed data. **Borrowed chords** (P§3: they go "only if all of them rank last"): if `float` ranks 4th in wistful **and** `plagal` 1st in melancholy, Claude reports it and asks Shresth (§11 question 2); nothing leaves `chords.numerals` without his words [CD-45]. A tie on either condition counts as not met (TASTE's default keeps the chords). The keep3 drafts follow his answer; if it comes after they are written, a kept draft with a removed chord is redrafted and heard again before the freeze. |
| Sax | **Exactly 1.** The S1 BW top 2 go to the final [CD-18] (ties for the two places by the tie rules, then a seeded draw; three or more tied give a three-source final). The final's winner (more bests over its rounds) is the sax; a tied final goes to the higher combined S1 + final BW, then to Shresth. S2 runs on the S1 leader and re-runs on Day 3 if the final picks the other. If the S1 leader is all-bad in half its rounds, or most S1 rounds are all-bad, no free sax passes, and D-012's fallback (the lead becomes Rhodes) is Shresth's call. |
| Sax variants | Warmth: the S2a BW leader of `sax_plain` and `sax_warm_on` (T·Sax realism: "if it still reads fake, warm the sax only"). If `sax_expr_off` has the top BW in S2a, Claude reports it and asks whether expression stays on. Timing: the S2b BW leader; ties as above, the default being grid. |
| Rain | Excerpt: the W1 BW leader; ties go to e1. Level: the W2 BW leader (Δ 24 is a Claude's default in TASTE); ties go to 24. A `rain_none` lead goes to Shresth, since no bed changes T·Mix's "Rain barely there". |

**Consistency** [CD-47]. The repeat agreement (implied pairs, the audit's figure) is pooled over the shootout
repeats and reported with its Wilson 95 % interval and §2.5's caveat; one repeat per shootout gives 3–5 pairs, too
few to flag a single shootout. Position-of-best counts are pooled by clip count and reported as descriptive.

**What is recorded:**
- `data/votes/audit/` (private): each batch's audit JSON and `shootouts.json`.
- `gates/GATE_palette.md`, per shootout: the survivors, a table of `shown`, `heard`, `best`, `worst`, `net` and BW
  per option, the escape-hatch counts, the pooled repeat agreement, flagged rounds (§3), the audit files the numbers
  come from, and Shresth's confirmation in his words with the date. The `APPROVED by Shresth Rana <yyyy-mm-dd>` line
  comes on Nov 6, after the kept 10; the drawn round tables follow it.
- CHANGELOG: the same numbers with their file paths, and the SU of each `lofi-shootout` run (`sacct`).

## 9. Departures from the source docs

- **Loops.** I6 runs the sax on "2 hand loops", the keys on "a fixed groove", the vamps on "one groove" and the rain
  on "one hand loop". This plan uses the 9 reference loops throughout [CD-3]: the sax on all 9 (S1 6, final 3), the
  keys on 5, the vamps on 9, the rain on 7. Clip appearances, anchors and repeats aside: keys 30 (I6: 11),
  drums + bass 35 (27), vamps 36 (12), sax 48 (11), rain 25 (4).
- **R2 on all three kits** (I6: the top 2) [CD-9]; **swing and humanize in separate rounds** (I6: the 12-clip grid)
  [CD-10]; **swing advisory** around Shresth's 55 % [CD-13].
- **Levels.** Per-instrument trims and faders matched on the reference loops replace instrument.md §4's
  `bus_gain_db` calibration on the hand loops [CD-25, CD-29], each shootout clip takes its own fader on its round's
  loop [CD-52], and the chains are not tuned on the hand loops before the freeze (§3, drift).
- **Rain** comes on Nov 6, with the file asked for by Nov 5 noon (P§7 has Shresth download it on Nov 6).
- **Borrowed chords:** P§3's rule is reported, not applied; removal needs Shresth's words [CD-45], since the
  palette's chord list is shared by every mood.
- **Patterns:** TASTE's drums row has the shootout prune patterns too; there is no pattern round by default [CD-46;
  §11].
- **Beyond the plan:** the S1 final [CD-18], the Latin squares for R3b, S2a, S2b, W1 and W2 [CD-14], the per-instrument
  trims [CD-25], and D-013's anchors and 2-day repeats in the shootouts (§2.5).
- **Listening:** Nov 6 holds about 15 minutes of listening besides P§7's 10-minute sign-off.

**Found for other docs** (not changed here): COMPILER.md §3.2's per-phrase octave fit can put a phrase and its answer
an octave apart, as it did in L4 and L8 under the alto before this review. The reference loops now avoid it; loops
the model writes will not.

## 10. Claude's defaults in this plan

Each is overridable by Shresth; the reason is one line.

| # | Default | Reason |
|---|---|---|
| CD-1 | Loops use `ep` and `dusty`; each shootout palette names its one keys voice and kit that way, with no aliases | the loop text is then byte-identical across a round's clips, so the palette is the only difference |
| CD-2 | No `sax:` line in any loop | the compiler plays the palette's `sax_lead` whatever the line says (DSL.md §4) |
| CD-3 | 9 reference loops, one per mood × energy, 76–90 BPM, 9 tonics, for every shootout | they cover every knob cell and several registers; the hand loops I6 names do not exist yet |
| CD-4 | Rootless voicing in every loop file | I6 fixes the keys shootout at rootless; one voicing keeps the other shootouts comparable |
| CD-5 | Energy checked against the §5 fixture's test bands .30 / .50 | the real bands come from the seed pool, which does not exist yet |
| CD-6 | No bed before the rain shootout | the file is not here, and its level is itself under test |
| CD-7 | Untested slots: the dev band on Day 1, the Day-1 leaders on Day 2, the Day-2 picks on Day 3 | the only instruments known to render before any tally, then the nearest to the final palette |
| CD-8 | K1 as three parallel classes of 2 rounds; twins = one family; family tile sums of 15 | every pair meets at least once; families meet 3 times on equal positions |
| CD-9 | R2 on all three kits instead of the top 2 | Day 1 renders in one job, and every kit gets a pairing |
| CD-10 | Swing and humanize in separate rounds | the 12-clip grid does not fit; main effects need fewer rounds |
| CD-11 | `split` swing puts ohat with hat | they are one instrument |
| CD-12 | Swing variants move every timing lane, the sax included | the sax's `grid` timing is defined as the band's grid (COMPILER.md §3.7) |
| CD-13 | R3a is advisory: swing stays 0.55 unless Shresth names another | his row says about 55 %; Day 2 has no room for the four rounds that four options need |
| CD-14 | Latin squares for R3b, S2a, S2b, W1 (3×3) and W2 (4×4) | these single values freeze with no later votes, so each option gets every tile once |
| CD-15 | Vamp rounds within one mood | a round has one brief, and the mood picks the mode |
| CD-16 | `plagal` and `float` always on the central tile sums (7 or 8) | they decide P§3's borrowed-chord condition, so position cannot push it either way |
| CD-17 | S1 as 6 rounds of 4, each source on every tile once; one alto-above round per alto | the sax is the signature sound; every alto is heard once in its own octave |
| CD-18 | An S1 final of the top 2, 3 rounds of 2 | exactly one sax freezes, and S1 alone often leaves the top two close |
| CD-19 | S2 timing rounds at warm off | both kinds of S2 round render the same night |
| CD-20 | Rain excerpts by smallest loudness range; W1 at Δ 18; W2 on e1 | the page lists sounds only his ear can place; Δ 18 is the loudest bed under test; W1 and W2 render the same night |
| CD-21 | Option-to-tile maps, round order and hidden controls drawn at job time, never committed | the repo is public, and a committed map would name the option on every tile |
| CD-22 | Consecutive rounds of a batch never share a loop; a batch opens with a design round | back-to-back rounds on one loop invite comparisons across rounds, which the tally cannot see |
| CD-23 | Anchors are extra rounds with a sour copy (every sax note's accidental raised by one) | one mechanical rule breaks every loop the same way and keeps every design round whole |
| CD-24 | A repeat never shows the first-served round of a stage | that round is the stage's warm-up |
| CD-25 | Level matching with trims before the chains and faders after, anchored on the GU instruments | no candidate wins by loudness, and each meets its compressor as the dev instrument does |
| CD-26 | Timing variants keep the instrument's trim and fader; spreads above 0.5 LU flagged | they move notes in time, not in level |
| CD-27 | Vamp clips not level-matched; keys and bass loudness recorded, spreads above 1.0 LU flagged | the harmony is the candidate; matching would change the band between clips |
| CD-28 | A keys instrument's frozen fader is the mean of its rootless and spread values | the frozen palette has one fader per instrument, and the model picks the voicing |
| CD-29 | The faders he heard go into the frozen palette; a kept-10 bus off target by 1.0 LU goes to Shresth | recalibrating on the kept 10 would change the mix he voted on |
| CD-30 | A borrowed lane's level copies the GU kit's shaker-to-snare difference | a lane recorded apart from the kit has no recorded balance with it |
| CD-31 | Virtuosity premix kickmic + snaremic + oh; room, mid, lofi out | Swirly was fetched dry and the GU kit is a SoundFont kit, so all three are compared on close and overhead mics; the lofi mic is left out by its name (T·Mix: clean; T·Lineage: the dust is not a target), unheard |
| CD-32 | Virtuosity rim = cross-stick; hats closed and open; ride, rimshot, stir, tight and foot unused | the GU kit's rim lane plays note 37, Side Stick in the General MIDI map; each DSL lane takes one articulation |
| CD-33 | Velocity layers at 0.25 / 0.6 / 0.9 of the range; neighbours as round robins | ghost, normal and accent fall inside the sampled range without its extremes |
| CD-34 | Swirly's shaker is VCSL Large Shake1D/1U; Virtuosity uses its own perc shaker | single down and up shakes fit a 16-step lane; the Small set's file names say double, faster, roll and slap |
| CD-35 | Trim at −30 dB re peak (−20 for shakers), 1 ms kept | every one-shot starts at the same offset; a shake builds up over tens of ms |
| CD-36 | `resample_poly` 147/160, one time, hashes pinned | exact rational ratio; one run on one machine avoids cross-CPU drift |
| CD-37 | FM Piano #1 preset (0, 0) | its other presets are named for single velocities (v60, v80, v100) |
| CD-38 | Heard-filtered BW; anchor rounds dropped from the recount | candidates are shown unequal numbers of times, and an anchor round is not a design round |
| CD-39 | Remaining single-winner ties go to the default; a tie without it goes to Shresth | a tie shows no preference over what the palette already does |
| CD-40 | Keys survivors only from the K2 participants; the third if its BW is above 0 | only they were heard with the spread voicing; above 0 means more bests than worsts |
| CD-41 | K2 takes a tie at third place up to 4 clips, then a seeded draw | 4 clips is the page's limit |
| CD-42 | Kits: the last goes only if its BW is below 0; a tie for last keeps all 3 | TASTE keeps 2–3, and dropping two would leave one |
| CD-43 | Each kit's bass from its own R2 round; the pooled leader if that round has no best | I6 pairs each kit with the bass heard with it |
| CD-44 | An option all-bad in half its rounds cannot survive | it failed as often as it was heard (T·Tiers: Toss = a dealbreaker fired) |
| CD-45 | "Borrowed" = a bVI or bVII chord; nothing removed without his words | P§3 names bVI and bVII; the chord list is shared by every mood, and these chords belong to two of the three modes |
| CD-46 | No pattern round; patterns left to the seeds and votes | the palette has no pattern field, and the model writes every pattern |
| CD-47 | Repeat agreement pooled over the shootouts; no per-shootout flag | one repeat gives 3–5 pairs, too few to flag a shootout |
| CD-48 | Day 1 first stages, Day 2 dependent stages, Day 3 rain and repeats | each stage renders the night after the tally it needs; D-013 needs a 2-day gap |
| CD-49 | Batch names sort in serving order | `make rate` serves the oldest unfinished batch by path |
| CD-50 | The cut list's order | it cuts a stage's third loop before any second one, and the S1 final last |
| CD-51 | Shootout palettes carry trims as a chain-head `Gain` or lane `db` and faders in `bus_gain_db` | no palette code is needed before the freeze, while other workflows edit `palette.py` and `pipeline.py` |
| CD-52 | Each shootout clip's fader on the bus its round varies comes from its candidate's measured loudness on the round's own loop (0.1 LU per clip); the frozen palette keeps the median fader (amends §3, 2026-10-31) | matching the median over the 9 loops left 10 of the 18 Day-1 design rounds and the anchor over 1.0 LU apart on their own loop (up to 4.20 LU, R1.3 on L8), and a candidate must never win by being louder |

## 11. Waiting on Shresth

Questions in plain words, each with its deadline and what happens without an answer:

1. **Hidden checks in the shootouts. ANSWERED 2026-10-31: keep them** (Shresth: "Defaults are fine."). Was due by Nov 2 evening, before job 1. "Your vote rules bring a few rounds back at
   least two days later and slip in a deliberately broken loop now and then. In the shootouts that means 5 repeated
   rounds on Nov 6, when the keys, drums, chords and sax rounds are otherwise over, so you might recognise them, and
   3 rounds where one clip has a deliberately sour sax line. About 9 minutes in all. Keep them, or leave them out of
   the shootouts?" Without an answer they stay, as D-013 says.
2. **Borrowed chords** (by Nov 4 morning, before the chords batch). "Four of the twelve chord pairs use a chord that
   comes from outside the key in the hopeful mood but belongs to the key in the sad and wistful moods. No hopeful
   pair in the shootout uses one, so the shootout cannot test them there. The plan drops these chords if every pair
   that has one comes last. If that happens, should Claude drop them everywhere, only from hopeful loops, or keep
   them?" Without an answer they stay (TASTE's default).
3. **A beat round. ANSWERED 2026-10-31: no beat round** (Shresth: "Defaults are fine."). Was due by Nov 2 evening. "Your drums answer says the shootout should narrow down kits and beats. The
   plan compares kits, each on 3 beats, but has no round that compares beats with the same kit, because the model
   writes every beat and your votes judge them later. Want a short beat round, about 3 minutes on Nov 3?" Without an
   answer there is none.
4. **The rain file. DONE 2026-10-31:** in `assets/big/rain/`, pinned in assets/MANIFEST.yaml (sha256 `60f7398e…`) and on the
   cluster.

And the listening itself. No job type needs his approval (`lofi-shootout` is at most 240 SU).
