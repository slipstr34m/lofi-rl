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
