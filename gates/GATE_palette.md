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

## How to approve
Add one line of your own, exactly in this form:

    APPROVED by Shresth Rana <yyyy-mm-dd>

## Approval
