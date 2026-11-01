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

## How to approve
Add one line of your own, exactly in this form:

    APPROVED by Shresth Rana <yyyy-mm-dd>

## Approval
