# Gate: palette_v1 freeze (Phase 1)

The palette freezes when Shresth has:
1. picked the shootout survivors by ear: sax (1), keys (2–3), drums+bass kits (2–3), vamps, rain level, humanize on/off;
2. kept 10 of the 30 drafted loops (keep3 rounds);
3. written the approval line below.

After the freeze, `palettes/lofi_rain_v1.yaml` never changes (docs/DECISIONS.md D-017).

## Shootout results

### Day 1: voted 2026-10-31 (batches b1103-1-keys, b1103-2-groove, b1103-3-sax; job 13635646)

**Shresth's confirmation, 2026-10-31:** "Yes, use these". This answered the proposal of the survivors below. The proposal also covered:
- **The heard rule:** a pick counts once its clip has played 1 bar, over every attempt of its round. That is his rule: "Imagine it sounded so bad that I did not even let it play more than 4 bars." (D-013 as amended).
- **A 3-round bass re-listen** in Day 2.
- **No Day-2 cuts.**

**Survivors**
- **Keys to K2:** `gs_wurli_ep200`, `gu_tine_ep` and `vcsl_upright_knight`, in a full tie. `gu_tine_ep`, the default, leads (CD-39). Out: `gu_organ`, `jrhodes3d`, `fp_fm_ep1`, `fp_fsbs_jazz_guitar` and `fp_upright_kw`.
- **Kits:** `kit_virtuosity` and `kit_gu_jazz` survive. Virtuosity leads: it ties on BW and has more bests. `kit_swirly` is dropped, since its BW is −0.667, below 0 (CD-42).
- **Bass:** undecided. No R2 round has a best, and the pooled leaders `bass_fp_finger_yr` and `bass_vsco2_pizz` tie without the default.
  - R2r decides it on Day 2 (CD-54). The leader kit gets two R2r rounds, the other kit one, and each kit takes the best bass of its own rounds (CD-43).
  - The Day-2 band keeps `bass_gu_acoustic` meanwhile (CD-39).
- **Sax final:** `sax_mtg_alto`, the S1 leader, against `sax_mtg_tenor` (CD-18).

**Day 2's job:** `--stage day2 --leaders keys=gu_tine_ep+gs_wurli_ep200+vcsl_upright_knight/kit=kit_virtuosity+kit_gu_jazz/bass=bass_gu_acoustic/sax=sax_mtg_alto+sax_mtg_tenor --prior 13635646 --add R2r`, with no cuts.
- On the audit's 30.7 Day-1 minutes, the CD-50 list would have cut `vamps-high+latin3+final`.
- Those minutes include 8.1 minutes of an attempt with no sound, which was the Mac's output device, not the page. Without them Day 1 took 22.7 minutes against the plan's 23.
- Shresth's word: no cuts.

**Tables**
- Heard: played for at least 1 bar, over every attempt of the round.
- BW = (best − worst) / heard; this is what decides.
- net: the audit's best − worst over every shown clip.
- all-bad: rounds marked G.

K1 (keys):

| place | option | shown | heard | best | worst | BW | net | all-bad |
|---|---|---|---|---|---|---|---|---|
| 1= | `gs_wurli_ep200` | 3 | 3 | 1 | 0 | +0.333 | +1 | 1 of 3 |
| 1= | `gu_tine_ep` | 3 | 3 | 1 | 0 | +0.333 | +1 | 0 of 3 |
| 1= | `vcsl_upright_knight` | 3 | 3 | 1 | 0 | +0.333 | +1 | 1 of 3 |
| 4= | `gu_organ` | 3 | 3 | 1 | 1 | 0.000 | 0 | 0 of 3 |
| 4= | `jrhodes3d` | 3 | 3 | 1 | 1 | 0.000 | 0 | 0 of 3 |
| 6= | `fp_fm_ep1` | 3 | 3 | 0 | 1 | −0.333 | −1 | 1 of 3 |
| 6= | `fp_fsbs_jazz_guitar` | 3 | 3 | 0 | 1 | −0.333 | −1 | 0 of 3 |
| 6= | `fp_upright_kw` | 3 | 3 | 0 | 1 | −0.333 | −1 | 1 of 3 |

R1 (kits):

| place | option | shown | heard | best | worst | BW | net | all-bad |
|---|---|---|---|---|---|---|---|---|
| 1 | `kit_virtuosity` | 3 | 3 | 2 | 1 | +0.333 | +1 | 0 of 3 |
| 2 | `kit_gu_jazz` | 3 | 3 | 1 | 0 | +0.333 | +1 | 0 of 3 |
| 3 | `kit_swirly` | 3 | 3 | 0 | 2 | −0.667 | −2 | 0 of 3 |

R2 (basses):

| place | option | shown | heard | best | worst | BW | net | all-bad |
|---|---|---|---|---|---|---|---|---|
| 1= | `bass_fp_finger_yr` | 3 | 3 | 0 | 0 | 0.000 | 0 | 1 of 3 |
| 1= | `bass_vsco2_pizz` | 3 | 3 | 0 | 0 | 0.000 | 0 | 1 of 3 |
| 3 | `bass_gu_acoustic` | 3 | 3 | 0 | 1 | −0.333 | −1 | 1 of 3 |

S1 (sax):

| place | option | shown | heard | best | worst | BW | net | all-bad |
|---|---|---|---|---|---|---|---|---|
| 1 | `sax_mtg_alto` | 4 | 4 | 3 | 0 | +0.750 | +4 | 0 of 4 |
| 2 | `sax_mtg_tenor` | 4 | 3 | 1 | 0 | +0.333 | +1 | 0 of 4 |
| 3= | `sax_gu_alto` | 4 | 4 | 1 | 1 | 0.000 | 0 | 0 of 4 |
| 3= | `sax_weresax_alto` | 4 | 4 | 1 | 1 | 0.000 | 0 | 0 of 4 |
| 5 | `sax_fp_tenor` | 4 | 3 | 0 | 1 | −0.333 | −1 | 0 of 4 |
| 6 | `sax_gu_tenor` | 4 | 3 | 0 | 2 | −0.667 | −3 | 0 of 4 |

**Escape hatches:** K1 had one G. R1 had none. R2 had T 2, B 2 and G 1. S1 had none.

**Dropped and left out:** S1's anchor round was dropped; it was caught, so every vote keeps full weight. 1 mark was left out: a sax "worst" on a clip that never played.

**Flags (§3):** none. The largest per-round spreads were K1 0.0081, R1 0.0072, R2 0.0057 and S1 0.0091 LU, against a limit of 1.0.

**Repeats (CD-47):** none yet; they come on Day 3.

**Position of best** (descriptive, pooled):
- 3-clip rounds: tiles [1, 2, 0], p 0.368;
- 4-clip rounds: [7, 2, 2, 1], chi-square 7.33 on 3 df, p 0.062.

**Audit files** (private, never committed): `data/votes/audit/{b1103-1-keys,b1103-2-groove,b1103-3-sax,shootouts}.json` and `day1-report.md`. They come from `python -m lofi_rl.shootout.tally` at main `e0ebc8f`, and an independent recount over the raw logs gives the same figures for every option.

### Day 2: voted 2026-11-01 (batches b1104-1-groove, b1104-2-sax, b1104-3-keys, b1104-4-vamps; job 13651610)

**Shresth's confirmation, 2026-11-01:** "Yes, use these". This answered the proposal of the results below, which kept swing at 0.55 because he named no other.

**Results**
- **Sax:** `sax_mtg_alto` wins the final, with more bests (CD-18). S2 ran on it, so nothing re-runs on Day 3.
  - Warmth: `sax_plain`. It ties `sax_warm_on` on BW, so the default stays (CD-39).
  - Expression stays on: `sax_expr_off` came last in S2a.
  - Timing: `timing_lag`, the S2b BW leader.
- **Keys:** `gu_tine_ep` (leads), `gs_wurli_ep200` and `vcsl_upright_knight` (CD-40, BW over K1 + K2). K2 had no marks: T and B in both rounds.
- **Kits and their basses** (CD-43 over R2r, CD-54):
  - `kit_virtuosity` with `bass_vsco2_pizz`, from its two R2r rounds;
  - `kit_gu_jazz` with `bass_fp_finger_yr`, from its one R2r round.
- **Swing:** `swing_55`. R3a had no marks; this is advisory (CD-13), and Shresth named no other.
- **Humanize:** `humanize_off`. R3b had no marks, so the default stays.
- **Vamps** (seed data, CD-45): hopeful `aruarian`, then `royal_half`, with `lift` and `two_of_us` tied third. Melancholy `plagal` (tied with `rise` on BW, ahead on bests), then `rise`, `backdoor`, `drift`. Wistful `float`, then `i_iv` and `pulse` tied, then `step`.
- **Borrowed chords:** the condition is not met, since `float` is first in wistful. `chords.numerals` stays as it is (SHOOTOUTS.md §11 question 2 does not arise).
- **Weight:** the sax batch's anchor was marked "no clear best" and "no clear worst", so micro-session `s1101a`/1 counts at half weight (D-013). That session holds every Day-2 vote, so no outcome changes.

**Tables** (the same columns as Day 1; weighted marks):

R2r (basses, pooled over its three rounds):

| place | option | shown | heard | best | worst | BW | net |
|---|---|---|---|---|---|---|---|
| 1= | `bass_fp_finger_yr` | 3 | 1.5 | 0.5 | 0.5 | 0.000 | 0 |
| 1= | `bass_vsco2_pizz` | 3 | 1.5 | 0.5 | 0.5 | 0.000 | 0 |
| 3 | `bass_gu_acoustic` | 3 | 1.5 | 0 | 0.5 | −0.333 | −1 |

R2r per kit:
- `kit_virtuosity` (2 rounds): `bass_vsco2_pizz` +0.500, `bass_gu_acoustic` −0.500, `bass_fp_finger_yr` −0.500.
- `kit_gu_jazz` (1 round): `bass_fp_finger_yr` +1.000, `bass_gu_acoustic` 0.000, `bass_vsco2_pizz` −1.000.

S1f (sax final), and S1 + S1f pooled:

| stage | option | shown | heard | best | worst | BW |
|---|---|---|---|---|---|---|
| S1f | `sax_mtg_alto` | 3 | 1.5 | 1 | 0.5 | +0.333 |
| S1f | `sax_mtg_tenor` | 3 | 1.5 | 0.5 | 0.5 | 0.000 |
| S1 + S1f | `sax_mtg_alto` | 7 | 5.5 | 4 | 0.5 | +0.636 |
| S1 + S1f | `sax_mtg_tenor` | 7 | 4.5 | 1.5 | 0.5 | +0.222 |

S2a (warmth) and S2b (timing):

| stage | option | shown | heard | best | worst | BW |
|---|---|---|---|---|---|---|
| S2a | `sax_plain` | 3 | 1.5 | 0.5 | 0 | +0.333 |
| S2a | `sax_warm_on` | 3 | 1.5 | 0.5 | 0 | +0.333 |
| S2a | `sax_expr_off` | 3 | 1.5 | 0 | 0.5 | −0.333 |
| S2b | `timing_lag` | 3 | 1.5 | 0.5 | 0 | +0.333 |
| S2b | `timing_swung` | 3 | 1 | 0 | 0 | 0.000 |
| S2b | `timing_grid` | 3 | 1.5 | 0 | 0.5 | −0.333 |

**Rounds without marks:** K2 (3 keys, 2 rounds), R3a (4 swings, 2 rounds) and R3b (3 humanize settings, 3 rounds) had no best or worst marks. Every option there sits at BW 0.000.

**K1 + K2 pooled:** `gs_wurli_ep200`, `gu_tine_ep` and `vcsl_upright_knight` each have 5 shown, 4 heard, 1 best, 0 worst and BW +0.250.

**Escape hatches:**
- R2r: T 1.
- R3a: T 2, B 2.
- R3b: T 3, B 3.
- S1f: B 1.
- S2a: T 1, B 2.
- S2b: T 2, B 2.
- K2: T 2, B 2.
- V: T 1, B 5.
- None used G.

**Flags (§3):** none. Every Day-2 stage's largest spread is under 0.01 LU; the limit is 1.0, or 0.5 for the timing variants.

**Hidden checks:** Day 2's one anchor was missed (above). There are no repeats yet; they come on Day 3.

**Audit files** (private): `data/votes/audit/{b1104-1-groove,b1104-2-sax,b1104-3-keys,b1104-4-vamps,shootouts}.json` and `day2-report.md`, from the tally at main `7a0dfd0`. An independent recount over the raw logs gives the same figures for every stage, for R2r per kit and for V per mood.

### Day 3: voted 2026-11-03 (batches b1106-1-mix, b1106-2-mix; job 13656048)

**Shresth's confirmation, 2026-11-03:** "Yes, use these", with "24 below (Recommended)" as his word on the rain level.
He was asked which rain level the palette should use, since W2's three quietest options tied with no best given. A tie keeps the default, Δ 24, but a `rain_none` lead is his call (SHOOTOUTS.md §8, Rain). He was then asked whether to use the rest of Day 3's results.

**Results**
- **Rain excerpt:** `excerpt_e2`, the W1 BW leader, holding the only best.
- **Rain level:** `rain_d24` (Δ 24 LU under the music), by his word.
  - W2 had no best in any of its 4 rounds.
  - `rain_d24`, `rain_d30` and `rain_none` tied at BW 0.000.
  - `rain_d18` came last with 2 worsts.
- **The sax:** no S2 re-run, since the final's winner is the sax S2 ran on.

**Tables** (weighted marks; every Day-3 vote is at full weight):

| stage | option | shown | heard | best | worst | BW | net |
|---|---|---|---|---|---|---|---|
| W1 | `excerpt_e2` | 3 | 3 | 1 | 0 | +0.333 | +1 |
| W1 | `excerpt_e1` | 3 | 3 | 0 | 0 | 0.000 | 0 |
| W1 | `excerpt_e3` | 3 | 3 | 0 | 0 | 0.000 | 0 |
| W2 | `rain_d24` | 4 | 4 | 0 | 0 | 0.000 | 0 |
| W2 | `rain_d30` | 4 | 4 | 0 | 0 | 0.000 | 0 |
| W2 | `rain_none` | 4 | 4 | 0 | 0 | 0.000 | 0 |
| W2 | `rain_d18` | 4 | 4 | 0 | 2 | −0.500 | −2 |

**Escape hatches:** W1: T 2, B 3. W2: T 4, B 2. None used G.

**Flags (§3):** none. W1's largest bed spread is 0.0274 LU and its largest sax spread 0.0015 LU; the limit is 1.0. W2 has no level flag.

**Hidden checks**
- **The anchor:** micro-session `s1103a`/1 caught its one anchor, so every Day-3 vote counts at full weight (D-013).
- **Repeats** (CD-47): 3 of 7 implied pairs agree (p 0.429, Wilson 95 % 0.158–0.750) over 5 repeats, the first showings voted on Days 1 and 2. With one repeat per shootout, that is too few pairs to read any one shootout from (SHOOTOUTS.md §8, Consistency).
- **Position of best** (descriptive, pooled over every shootout batch): 2 clips [1, 2], p 0.564; 3 clips [3, 2, 4], p 0.717; 4 clips [9, 7, 4, 2], chi-square 5.27 on 3 df, p 0.153.

**Audit files** (private): `data/votes/audit/{b1106-1-mix,b1106-2-mix,shootouts}.json` and `day3-report.md`, from the tally at main `dcc0d9c` (2026-11-03 09:50).
- An independent recount of W1 and W2 over the raw logs and the hidden manifests gives the same figures: shown, heard, best, worst, BW and escape hatches.
- The standing votes are 7 design rounds, 5 repeats and 1 anchor.

## Kept drafts (keep3): voted 2026-11-01 (batch b1101-keep3; job 13656049)

The palette's gate item 2: Shresth kept 10 of the 30 drafted loops (docs/DRAFTS.md; taste_pipeline.md §8). That is one per brief, from the three personas' drafts of that brief, rendered with `palettes/draft_day2.yaml`. No round was marked "none acceptable", so there are no redrafts.

| Brief | Energy, mood | Kept |
|---|---|---|
| d01 | low, melancholy | `drafts/v1/d01_chords.ls` |
| d02 | low, wistful | `drafts/v1/d02_chords.ls` |
| d03 | low, hopeful | `drafts/v1/d03_chords.ls` |
| d04 | mid, melancholy | `drafts/v1/d04_horn.ls` |
| d05 | mid, wistful | `drafts/v1/d05_groove.ls` |
| d06 | mid, hopeful | `drafts/v1/d06_horn.ls` |
| d07 | high, melancholy | `drafts/v1/d07_chords.ls` |
| d08 | high, wistful | `drafts/v1/d08_horn.ls` |
| d09 | high, hopeful | `drafts/v1/d09_chords.ls` |
| d10 | low, wistful | `drafts/v1/d10_groove.ls` |

- **Kept per persona** (descriptive): chords 5, horn 3, groove 2.
- **Labels** (taste_pipeline §8, for the RM in Phase 2): every kept loop is at least Fine, and beats each of its round's rejects. In round d08 one reject never played, so under D-013 as amended that one pair is dropped.
- **Audit file** (private): `data/votes/b1101-keep3.jsonl`, backed up as votes commit `085f8d2`.

## Sign-off listen: voted 2026-11-03 (batch b1103-signoff; job 13701026)

The kept 10, rendered on the cluster with `palettes/lofi_rain_v1.yaml` (sha256 `d3590c8d…`, `frozen: null`), the rain
bed included. Triage mode: one loop per round, played through its seam.
- **Job 13701026:** `SIGNOFF_OK`. Every check in `runs/drafts-13701026/summary/verify.json` passed.
  - The no-bed twin's audio equals keep3's for all 10, so the music under the rain is what he kept.
  - The repeats equal their renders.
  - The bed is placed at −24.0 LU. After the limiter it sits 22.6–24.0 LU under the music.
- **Counts:** 10 of 10 heard. Great 2, Good 1, Fine 3, Toss 4.
- **Tossed:** `drafts/v1/d01_chords.ls`, `drafts/v1/d06_horn.ls`, `drafts/v1/d07_chords.ls` and `drafts/v1/d10_groove.ls`.
- **Shresth, 2026-11-03,** asked what made him toss them (the rain, a part, or the loops themselves): "The loops
  themselves".
- **Outcome: the sign-off has not passed** (docs/FREEZE.md, the sign-off loop).
  - His words name the loops, not the palette, so the palette does not change.
  - Briefs d01, d06, d07 and d10 reopen through the drafts redo path.
  - The kept list changes only by his words, after he hears the redrafts.
  - There is no APPROVED line yet.
- **Audit file** (private): `data/votes/b1103-signoff.jsonl`, backed up as votes commit `9dec1cc`.

## Kept drafts (keep3 redo1): voted 2026-11-03 (batch b1103-keep3-redo; job 13703830)

The four briefs the sign-off reopened (d01, d06, d07 and d10), redrafted by the three writers into `drafts/v1/redo1/`
(docs/DRAFTS.md §6). They were rendered with `palettes/draft_day2.yaml`, as keep3 was, and heard as one keep3 round per
brief. Shresth kept one loop in each round; no round was marked "none acceptable".

| Brief | Energy, mood | Kept |
|---|---|---|
| d01 | low, melancholy | `drafts/v1/redo1/d01_groove.ls` |
| d06 | mid, hopeful | `drafts/v1/redo1/d06_chords.ls` |
| d07 | high, melancholy | `drafts/v1/redo1/d07_groove.ls` |
| d10 | low, wistful | `drafts/v1/redo1/d10_chords.ls` |

- **Job 13703830:** `DRAFTS_OK`. Its 12 renders sit at −14.082 … −13.994 LUFS, with true peak at most −1.456 dBTP;
  5.4 SU.
- **Kept per persona** (descriptive): groove 2, chords 2, horn 0.
- **Labels** (taste_pipeline §8, for the RM in Phase 2): each kept loop is at least Fine and beats each of its round's
  two rejects. Every clip was heard for at least one bar.
- **The kept list now** is keep3's table with these four picks over it (docs/FREEZE.md §5). The four tossed loops
  stay out of it.
- **Audit file** (private): `data/votes/b1103-keep3-redo.jsonl`, backed up as votes commit `5e804de`.

## Sign-off listen: voted 2026-11-04 (batch b1103-signoff-2; job 13706355)

The new kept 10, keep3's six that passed the first sign-off and his four redo picks, rendered on the cluster with
`palettes/lofi_rain_v1.yaml` (sha256 `d3590c8d…`, `frozen: null`, unchanged since the first sign-off), the rain bed
included. Triage mode: one loop per round, played through its seam.
- **Job 13706355:** `SIGNOFF_OK`. Every check in `runs/drafts-13706355/summary/verify.json` passed.
  - The no-bed twin's audio equals each loop's own source render (keep3's for six, the redo round's for four), so the
    music under the rain is what he kept.
  - The repeats equal their renders.
  - The bed is placed at −24.0 LU. After the limiter it sits 22.6–23.9 LU under the music on the kept loops.
- **Before it** (Q2, SHOOTOUTS.md §3): on the new kept 10 the drums sit +1.149 LU and the keys −1.040 LU off their
  targets (median). Shresth, 2026-11-03: "Keep as heard (Recommended)". Nothing changed.
- **Counts:** 10 of 10 heard. Great 3, Good 4, Fine 3, Toss 0.
- **Tossed:** none.
- **Outcome: the sign-off passed.** His APPROVED line below seals the palette.
- **Audit file** (private): `data/votes/b1103-signoff-2.jsonl`, backed up as votes commit `a1eb3c0`.

## How to approve
Add one line of your own, exactly in this form:

    APPROVED by Shresth Rana <yyyy-mm-dd>

## Approval

APPROVED by Shresth Rana 2026-11-04

## Drawn round tables (published after the freeze)

What the batch builders drew at job time, published now that the freeze is approved (SHOOTOUTS.md §2.3; his approval line above is dated 2026-11-04). Written by `python -m lofi_rl.freeze.tables` from each job's `summary/draw.json` and `summary/batches.json` (keep3 and its redo rounds: `summary/batch.json`), and checked against every batch's `manifest.json` and `manifest.hidden.json`. Left out: the 64-bit seeds, the opaque round and clip ids, the real clip ids and the audio hashes. Each job's files are named by their sha256.

- Rounds: each shootout batch's design rounds by slot, and keep3's rounds (and each redo round's) in serving order. Tiles: left to right as served, each `option` (what it played).
- Left out: 8 rounds, the hidden checks (RATING.md, "Checks you won't see"). Phase 2's votes and the final A/B use the same checks, and the rater is never told which rounds those are. The full tables are kept privately, never committed.

### Drawn, Day 1 (job 13635646)

- `runs/shootout-13635646/summary/draw.json` sha256 `50465af89860bb0d879ebcbec6b65e2c78e779a29c4ae545a44792a3dbb20cc7`
- `runs/shootout-13635646/summary/batches.json` sha256 `9c6307dcfa122a2bd1f91c0d3052faa6756fa9bf11a11b050e2b23e0e634c3ec`

Symbol to option, per stage:

| stage | symbol | option |
|---|---|---|
| K1 | P1a | `jrhodes3d` |
| K1 | P1b | `gu_tine_ep` |
| K1 | P2a | `fp_fsbs_jazz_guitar` |
| K1 | P2b | `gu_organ` |
| K1 | P3a | `vcsl_upright_knight` |
| K1 | P3b | `fp_upright_kw` |
| K1 | P4a | `gs_wurli_ep200` |
| K1 | P4b | `fp_fm_ep1` |
| R1 | k1 | `kit_swirly` |
| R1 | k2 | `kit_gu_jazz` |
| R1 | k3 | `kit_virtuosity` |
| R2 | b1 | `bass_gu_acoustic` |
| R2 | b2 | `bass_vsco2_pizz` |
| R2 | b3 | `bass_fp_finger_yr` |
| S1 | t1 | `sax_gu_tenor` |
| S1 | t2 | `sax_mtg_tenor` |
| S1 | t3 | `sax_fp_tenor` |
| S1 | a1 | `sax_mtg_alto` |
| S1 | a2 | `sax_gu_alto` |
| S1 | a3 | `sax_weresax_alto` |

#### b1103-1-keys (`inbox/2026-11-03/b1103-1-keys/`)

| slot | loop | tile 1 | tile 2 | tile 3 | tile 4 |
|---|---|---|---|---|---|
| K1.1 | l5_wistful_mid | `jrhodes3d` (rootless) | `gu_tine_ep` (rootless) | `fp_fsbs_jazz_guitar` (rootless) | `gu_organ` (rootless) |
| K1.2 | l1_melancholy_low | `gu_tine_ep` (rootless) | `vcsl_upright_knight` (rootless) | `fp_upright_kw` (rootless) | `jrhodes3d` (rootless) |
| K1.3 | l9_hopeful_high | `gs_wurli_ep200` (rootless) | `fp_fm_ep1` (rootless) | `jrhodes3d` (rootless) | `gu_tine_ep` (rootless) |
| K1.4 | l5_wistful_mid | `vcsl_upright_knight` (rootless) | `fp_upright_kw` (rootless) | `gs_wurli_ep200` (rootless) | `fp_fm_ep1` (rootless) |
| K1.5 | l1_melancholy_low | `gu_organ` (rootless) | `gs_wurli_ep200` (rootless) | `fp_fm_ep1` (rootless) | `fp_fsbs_jazz_guitar` (rootless) |
| K1.6 | l9_hopeful_high | `fp_fsbs_jazz_guitar` (rootless) | `gu_organ` (rootless) | `vcsl_upright_knight` (rootless) | `fp_upright_kw` (rootless) |

#### b1103-2-groove (`inbox/2026-11-03/b1103-2-groove/`)

| slot | loop | tile 1 | tile 2 | tile 3 |
|---|---|---|---|---|
| R1.1 | l3_melancholy_high | `kit_swirly` (kit, GU acoustic bass, 3 patterns at 55 %, no humanize) | `kit_gu_jazz` (kit, GU acoustic bass, 3 patterns at 55 %, no humanize) | `kit_virtuosity` (kit, GU acoustic bass, 3 patterns at 55 %, no humanize) |
| R1.2 | l4_wistful_low | `kit_gu_jazz` (kit, GU acoustic bass, 3 patterns at 55 %, no humanize) | `kit_virtuosity` (kit, GU acoustic bass, 3 patterns at 55 %, no humanize) | `kit_swirly` (kit, GU acoustic bass, 3 patterns at 55 %, no humanize) |
| R1.3 | l8_hopeful_mid | `kit_virtuosity` (kit, GU acoustic bass, 3 patterns at 55 %, no humanize) | `kit_swirly` (kit, GU acoustic bass, 3 patterns at 55 %, no humanize) | `kit_gu_jazz` (kit, GU acoustic bass, 3 patterns at 55 %, no humanize) |
| R2.1 | l6_wistful_high | `bass_gu_acoustic` (bass on the round's kit) | `bass_vsco2_pizz` (bass on the round's kit) | `bass_fp_finger_yr` (bass on the round's kit) |
| R2.2 | l7_hopeful_low | `bass_vsco2_pizz` (bass on the round's kit) | `bass_fp_finger_yr` (bass on the round's kit) | `bass_gu_acoustic` (bass on the round's kit) |
| R2.3 | l2_melancholy_mid | `bass_fp_finger_yr` (bass on the round's kit) | `bass_gu_acoustic` (bass on the round's kit) | `bass_vsco2_pizz` (bass on the round's kit) |

#### b1103-3-sax (`inbox/2026-11-03/b1103-3-sax/`)

| slot | loop | tile 1 | tile 2 | tile 3 | tile 4 |
|---|---|---|---|---|---|
| S1.1 | l4_wistful_low | `sax_fp_tenor` (sax source) | `sax_mtg_alto` (sax source) | `sax_gu_alto` (sax source) | `sax_weresax_alto` (sax source) |
| S1.2 | l3_melancholy_high | `sax_weresax_alto` (sax source) | `sax_gu_tenor` (sax source) | `sax_mtg_tenor` (sax source) | `sax_fp_tenor` (sax source) |
| S1.3 | l1_melancholy_low | `sax_gu_tenor` (sax source) | `sax_gu_alto` (sax source) | `sax_weresax_alto` (sax source) | `sax_mtg_alto` (sax source) |
| S1.4 | l5_wistful_mid | `sax_mtg_alto` (sax source) | `sax_mtg_tenor` (sax source) | `sax_fp_tenor` (sax source) | `sax_gu_tenor` (sax source) |
| S1.5 | l9_hopeful_high | `sax_mtg_tenor` (sax source) | `sax_weresax_alto` (sax source) | `sax_mtg_alto` (sax source) | `sax_gu_alto` (sax source) |
| S1.6 | l7_hopeful_low | `sax_gu_alto` (sax source) | `sax_fp_tenor` (sax source) | `sax_gu_tenor` (sax source) | `sax_mtg_tenor` (sax source) |

### Drawn, Day 2 (job 13651610)

- `runs/shootout-13651610/summary/draw.json` sha256 `e09fab10a3a48e969169ea013ddfc60010bed98a31a2ecf0907e2e7ad7998e7a`
- `runs/shootout-13651610/summary/batches.json` sha256 `2526161c00c77f1d7a37d22bb46e3f025514202a6ac0062fb8f05cc8ccce539b`

Symbol to option, per stage:

| stage | symbol | option |
|---|---|---|
| K2 | u1 | `vcsl_upright_knight` |
| K2 | u2 | `gu_tine_ep` |
| K2 | u3 | `gs_wurli_ep200` |
| R2r | b1 | `bass_vsco2_pizz` |
| R2r | b2 | `bass_fp_finger_yr` |
| R2r | b3 | `bass_gu_acoustic` |
| R3a | w1 | `swing_50` |
| R3a | w2 | `swing_split` |
| R3a | w3 | `swing_55` |
| R3a | w4 | `swing_58` |
| R3b | n1 | `humanize_h1` |
| R3b | n2 | `humanize_h2` |
| R3b | n3 | `humanize_off` |
| S1f | f1 | `sax_mtg_alto` |
| S1f | f2 | `sax_mtg_tenor` |
| S2a | m1 | `sax_plain` |
| S2a | m2 | `sax_expr_off` |
| S2a | m3 | `sax_warm_on` |
| S2b | g1 | `timing_swung` |
| S2b | g2 | `timing_lag` |
| S2b | g3 | `timing_grid` |
| V (V.1, V.2, V.3) | v1 | `rise` |
| V (V.1, V.2, V.3) | v2 | `plagal` |
| V (V.1, V.2, V.3) | v3 | `backdoor` |
| V (V.1, V.2, V.3) | v4 | `drift` |
| V (V.4, V.5, V.6) | v1 | `step` |
| V (V.4, V.5, V.6) | v2 | `i_iv` |
| V (V.4, V.5, V.6) | v3 | `float` |
| V (V.4, V.5, V.6) | v4 | `pulse` |
| V (V.7, V.8, V.9) | v1 | `aruarian` |
| V (V.7, V.8, V.9) | v2 | `lift` |
| V (V.7, V.8, V.9) | v3 | `two_of_us` |
| V (V.7, V.8, V.9) | v4 | `royal_half` |

#### b1104-1-groove (`inbox/2026-11-04/b1104-1-groove/`)

| slot | loop | tile 1 | tile 2 | tile 3 | tile 4 |
|---|---|---|---|---|---|
| R2r.1 | l1_melancholy_low | `bass_vsco2_pizz` (bass re-listen on the round's kit, swing 0.55, no humanize) | `bass_fp_finger_yr` (bass re-listen on the round's kit, swing 0.55, no humanize) | `bass_gu_acoustic` (bass re-listen on the round's kit, swing 0.55, no humanize) |  |
| R2r.2 | l5_wistful_mid | `bass_fp_finger_yr` (bass re-listen on the round's kit, swing 0.55, no humanize) | `bass_gu_acoustic` (bass re-listen on the round's kit, swing 0.55, no humanize) | `bass_vsco2_pizz` (bass re-listen on the round's kit, swing 0.55, no humanize) |  |
| R2r.3 | l9_hopeful_high | `bass_gu_acoustic` (bass re-listen on the round's kit, swing 0.55, no humanize) | `bass_vsco2_pizz` (bass re-listen on the round's kit, swing 0.55, no humanize) | `bass_fp_finger_yr` (bass re-listen on the round's kit, swing 0.55, no humanize) |  |
| R3a.1 | l9_hopeful_high | `swing_50` (swing, humanize off (advisory, CD-13)) | `swing_split` (swing, humanize off (advisory, CD-13)) | `swing_55` (swing, humanize off (advisory, CD-13)) | `swing_58` (swing, humanize off (advisory, CD-13)) |
| R3a.2 | l4_wistful_low | `swing_58` (swing, humanize off (advisory, CD-13)) | `swing_55` (swing, humanize off (advisory, CD-13)) | `swing_split` (swing, humanize off (advisory, CD-13)) | `swing_50` (swing, humanize off (advisory, CD-13)) |
| R3b.1 | l3_melancholy_high | `humanize_h1` (humanize at swing 0.55) | `humanize_h2` (humanize at swing 0.55) | `humanize_off` (humanize at swing 0.55) |  |
| R3b.2 | l7_hopeful_low | `humanize_h2` (humanize at swing 0.55) | `humanize_off` (humanize at swing 0.55) | `humanize_h1` (humanize at swing 0.55) |  |
| R3b.3 | l5_wistful_mid | `humanize_off` (humanize at swing 0.55) | `humanize_h1` (humanize at swing 0.55) | `humanize_h2` (humanize at swing 0.55) |  |

#### b1104-2-sax (`inbox/2026-11-04/b1104-2-sax/`)

| slot | loop | tile 1 | tile 2 | tile 3 |
|---|---|---|---|---|
| S1f.1 | l2_melancholy_mid | `sax_mtg_alto` (sax final) | `sax_mtg_tenor` (sax final) |  |
| S1f.2 | l6_wistful_high | `sax_mtg_tenor` (sax final) | `sax_mtg_alto` (sax final) |  |
| S1f.3 | l8_hopeful_mid | `sax_mtg_alto` (sax final) | `sax_mtg_tenor` (sax final) |  |
| S2a.1 | l1_melancholy_low | `sax_plain` (the S1 leader's warmth, with the expression-off control) | `sax_expr_off` (the S1 leader's warmth, with the expression-off control) | `sax_warm_on` (the S1 leader's warmth, with the expression-off control) |
| S2a.2 | l4_wistful_low | `sax_expr_off` (the S1 leader's warmth, with the expression-off control) | `sax_warm_on` (the S1 leader's warmth, with the expression-off control) | `sax_plain` (the S1 leader's warmth, with the expression-off control) |
| S2a.3 | l9_hopeful_high | `sax_warm_on` (the S1 leader's warmth, with the expression-off control) | `sax_plain` (the S1 leader's warmth, with the expression-off control) | `sax_expr_off` (the S1 leader's warmth, with the expression-off control) |
| S2b.1 | l3_melancholy_high | `timing_swung` (the S1 leader's timing at warm off (CD-19)) | `timing_lag` (the S1 leader's timing at warm off (CD-19)) | `timing_grid` (the S1 leader's timing at warm off (CD-19)) |
| S2b.2 | l5_wistful_mid | `timing_lag` (the S1 leader's timing at warm off (CD-19)) | `timing_grid` (the S1 leader's timing at warm off (CD-19)) | `timing_swung` (the S1 leader's timing at warm off (CD-19)) |
| S2b.3 | l7_hopeful_low | `timing_grid` (the S1 leader's timing at warm off (CD-19)) | `timing_swung` (the S1 leader's timing at warm off (CD-19)) | `timing_lag` (the S1 leader's timing at warm off (CD-19)) |

#### b1104-3-keys (`inbox/2026-11-04/b1104-3-keys/`)

| slot | loop | tile 1 | tile 2 | tile 3 |
|---|---|---|---|---|
| K2.1 | l3_melancholy_high | `vcsl_upright_knight` (spread) | `gu_tine_ep` (spread) | `gs_wurli_ep200` (spread) |
| K2.2 | l8_hopeful_mid | `gu_tine_ep` (spread) | `gs_wurli_ep200` (spread) | `vcsl_upright_knight` (spread) |

#### b1104-4-vamps (`inbox/2026-11-04/b1104-4-vamps/`)

| slot | loop | tile 1 | tile 2 | tile 3 | tile 4 |
|---|---|---|---|---|---|
| V.1 | l1_melancholy_low | `rise` (sax muted) | `plagal` (sax muted) | `backdoor` (sax muted) | `drift` (sax muted) |
| V.2 | l2_melancholy_mid | `plagal` (sax muted) | `rise` (sax muted) | `drift` (sax muted) | `backdoor` (sax muted) |
| V.3 | l3_melancholy_high | `backdoor` (sax muted) | `drift` (sax muted) | `rise` (sax muted) | `plagal` (sax muted) |
| V.4 | l4_wistful_low | `step` (sax muted) | `i_iv` (sax muted) | `float` (sax muted) | `pulse` (sax muted) |
| V.5 | l5_wistful_mid | `i_iv` (sax muted) | `step` (sax muted) | `pulse` (sax muted) | `float` (sax muted) |
| V.6 | l6_wistful_high | `float` (sax muted) | `pulse` (sax muted) | `step` (sax muted) | `i_iv` (sax muted) |
| V.7 | l7_hopeful_low | `aruarian` (sax muted) | `lift` (sax muted) | `two_of_us` (sax muted) | `royal_half` (sax muted) |
| V.8 | l8_hopeful_mid | `lift` (sax muted) | `aruarian` (sax muted) | `royal_half` (sax muted) | `two_of_us` (sax muted) |
| V.9 | l9_hopeful_high | `two_of_us` (sax muted) | `royal_half` (sax muted) | `aruarian` (sax muted) | `lift` (sax muted) |

### Drawn, Day 3 (job 13656048)

- `runs/shootout-13656048/summary/draw.json` sha256 `834b370537ee41013a1a545653522c13828f830eb3d6709e38ea65d85fca9977`
- `runs/shootout-13656048/summary/batches.json` sha256 `ee3b2008af7303b02e6103e5d6eada22be97d79a7e6da3e04208ebd8d7aa022c`

Symbol to option, per stage:

| stage | symbol | option |
|---|---|---|
| W1 | z1 | `excerpt_e1` |
| W1 | z2 | `excerpt_e2` |
| W1 | z3 | `excerpt_e3` |
| W2 | y1 | `rain_none` |
| W2 | y2 | `rain_d18` |
| W2 | y3 | `rain_d24` |
| W2 | y4 | `rain_d30` |

#### b1106-1-mix (`inbox/2026-11-06/b1106-1-mix/`)

| slot | loop | tile 1 | tile 2 | tile 3 | tile 4 |
|---|---|---|---|---|---|
| W1.1 | l1_melancholy_low | `excerpt_e1` (rain excerpt at delta 18 (CD-20)) | `excerpt_e2` (rain excerpt at delta 18 (CD-20)) | `excerpt_e3` (rain excerpt at delta 18 (CD-20)) |  |
| W1.2 | l7_hopeful_low | `excerpt_e2` (rain excerpt at delta 18 (CD-20)) | `excerpt_e3` (rain excerpt at delta 18 (CD-20)) | `excerpt_e1` (rain excerpt at delta 18 (CD-20)) |  |
| W1.3 | l5_wistful_mid | `excerpt_e3` (rain excerpt at delta 18 (CD-20)) | `excerpt_e1` (rain excerpt at delta 18 (CD-20)) | `excerpt_e2` (rain excerpt at delta 18 (CD-20)) |  |
| W2.1 | l2_melancholy_mid | `rain_none` (rain level on excerpt e1) | `rain_d18` (rain level on excerpt e1) | `rain_d24` (rain level on excerpt e1) | `rain_d30` (rain level on excerpt e1) |
| W2.2 | l4_wistful_low | `rain_d18` (rain level on excerpt e1) | `rain_d24` (rain level on excerpt e1) | `rain_d30` (rain level on excerpt e1) | `rain_none` (rain level on excerpt e1) |

#### b1106-2-mix (`inbox/2026-11-06/b1106-2-mix/`)

| slot | loop | tile 1 | tile 2 | tile 3 | tile 4 |
|---|---|---|---|---|---|
| W2.3 | l6_wistful_high | `rain_d24` (rain level on excerpt e1) | `rain_d30` (rain level on excerpt e1) | `rain_none` (rain level on excerpt e1) | `rain_d18` (rain level on excerpt e1) |
| W2.4 | l8_hopeful_mid | `rain_d30` (rain level on excerpt e1) | `rain_none` (rain level on excerpt e1) | `rain_d18` (rain level on excerpt e1) | `rain_d24` (rain level on excerpt e1) |

### Drawn, keep3 (job 13656049)

- `runs/drafts-13656049/summary/draw.json` sha256 `57148c55d4606676c147e456e1f4ac15c14d2a23bd56ea4bb7ec4a9344ca323a`
- `runs/drafts-13656049/summary/batch.json` sha256 `7ab93c2b5e11b5623665ce13863f042b369cf7ba174db869d4b2cdc10a6a196f`

The persona on each tile of each row of the Latin square:

| row | tile 1 | tile 2 | tile 3 |
|---|---|---|---|
| 0 | `chords` | `horn` | `groove` |
| 1 | `horn` | `groove` | `chords` |
| 2 | `groove` | `chords` | `horn` |

Round order: drawn with no two consecutive rounds in one energy and mood cell.

#### b1101-keep3 (`inbox/2026-11-01/b1101-keep3/`)

| # | brief | energy, mood | row | tile 1 | tile 2 | tile 3 |
|---|---|---|---|---|---|---|
| 1 | d06 | mid, hopeful | 2 | `groove` (`drafts/v1/d06_groove.ls`) | `chords` (`drafts/v1/d06_chords.ls`) | `horn` (`drafts/v1/d06_horn.ls`) |
| 2 | d09 | high, hopeful | 0 | `chords` (`drafts/v1/d09_chords.ls`) | `horn` (`drafts/v1/d09_horn.ls`) | `groove` (`drafts/v1/d09_groove.ls`) |
| 3 | d01 | low, melancholy | 2 | `groove` (`drafts/v1/d01_groove.ls`) | `chords` (`drafts/v1/d01_chords.ls`) | `horn` (`drafts/v1/d01_horn.ls`) |
| 4 | d05 | mid, wistful | 0 | `chords` (`drafts/v1/d05_chords.ls`) | `horn` (`drafts/v1/d05_horn.ls`) | `groove` (`drafts/v1/d05_groove.ls`) |
| 5 | d04 | mid, melancholy | 0 | `chords` (`drafts/v1/d04_chords.ls`) | `horn` (`drafts/v1/d04_horn.ls`) | `groove` (`drafts/v1/d04_groove.ls`) |
| 6 | d07 | high, melancholy | 1 | `horn` (`drafts/v1/d07_horn.ls`) | `groove` (`drafts/v1/d07_groove.ls`) | `chords` (`drafts/v1/d07_chords.ls`) |
| 7 | d08 | high, wistful | 1 | `horn` (`drafts/v1/d08_horn.ls`) | `groove` (`drafts/v1/d08_groove.ls`) | `chords` (`drafts/v1/d08_chords.ls`) |
| 8 | d02 | low, wistful | 1 | `horn` (`drafts/v1/d02_horn.ls`) | `groove` (`drafts/v1/d02_groove.ls`) | `chords` (`drafts/v1/d02_chords.ls`) |
| 9 | d03 | low, hopeful | 2 | `groove` (`drafts/v1/d03_groove.ls`) | `chords` (`drafts/v1/d03_chords.ls`) | `horn` (`drafts/v1/d03_horn.ls`) |
| 10 | d10 | low, wistful | 2 | `groove` (`drafts/v1/d10_groove.ls`) | `chords` (`drafts/v1/d10_chords.ls`) | `horn` (`drafts/v1/d10_horn.ls`) |

### Drawn, keep3 redo1 (job 13703830)

- `runs/drafts-13703830/summary/draw.json` sha256 `fc834ea9982d37ec1f6eb800eb1ebd9045bcda3120217b467cdb289feb584192`
- `runs/drafts-13703830/summary/batch.json` sha256 `5f08a612beaa53ccfd8e8792e96daac596890d2dce170026208cba0b3de55e3b`

The persona on each tile of each row of the Latin square:

| row | tile 1 | tile 2 | tile 3 |
|---|---|---|---|
| 0 | `groove` | `horn` | `chords` |
| 1 | `horn` | `chords` | `groove` |
| 2 | `chords` | `groove` | `horn` |

Round order: drawn with no two consecutive rounds in one energy and mood cell.

#### b1103-keep3-redo (`inbox/2026-11-03/b1103-keep3-redo/`)

| # | brief | energy, mood | row | tile 1 | tile 2 | tile 3 |
|---|---|---|---|---|---|---|
| 1 | d01 | low, melancholy | 2 | `chords` (`drafts/v1/redo1/d01_chords.ls`) | `groove` (`drafts/v1/redo1/d01_groove.ls`) | `horn` (`drafts/v1/redo1/d01_horn.ls`) |
| 2 | d06 | mid, hopeful | 1 | `horn` (`drafts/v1/redo1/d06_horn.ls`) | `chords` (`drafts/v1/redo1/d06_chords.ls`) | `groove` (`drafts/v1/redo1/d06_groove.ls`) |
| 3 | d07 | high, melancholy | 0 | `groove` (`drafts/v1/redo1/d07_groove.ls`) | `horn` (`drafts/v1/redo1/d07_horn.ls`) | `chords` (`drafts/v1/redo1/d07_chords.ls`) |
| 4 | d10 | low, wistful | 1 | `horn` (`drafts/v1/redo1/d10_horn.ls`) | `chords` (`drafts/v1/redo1/d10_chords.ls`) | `groove` (`drafts/v1/redo1/d10_groove.ls`) |
