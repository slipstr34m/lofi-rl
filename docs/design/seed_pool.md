# The seed pool (taste_pipeline.md §8; CLAUDE.md §9, Phase 2): build spec

**Status:** a build spec, waiting on Shresth's answers to §9 (or his OK on its defaults) and on the step-0 unit (`docs/design/step0_samples.md`, built on `wt/step0`), whose 220 briefs and 660 loops it uses. It was written on 2026-11-07 at main `9ef969e` and revised the same day. `wt/step0` kept moving while this was written: at the revision it is `57b89f7`, 35 commits ahead of main, with a clean tree, and its spec has a new §10, "As built" (97 lines added and 22 removed against main; `git diff --stat`). Every `step0_samples.md` line cited below is at `57b89f7`. There `pool_briefs` still gives 220 briefs under both step-0 Q1 rules (35/169/16 and 30/176/14, 206 shared; `pool_briefs_check.py`), and 191 of the 220 positions keep their brief id (`design_sim.py`). Nothing in this spec has run, and no code for it exists.
- Its rules go into a new entry, **D-024**, after D-023 (§10). That covers the rules taste_pipeline.md §8 leaves open and the registered text they depart from, listed under "What this entry supersedes", as D-021 and D-023 do (`docs/DECISIONS.md:309-312`; `step0_samples.md:604-616`). Entries are never edited; a later one supersedes (`docs/DECISIONS.md:3`).
- D-024 carries Shresth's answers to §9 and every rule, and is committed with the first code (Build A, §8.2), before the pilot writes a loop. Each job's registration is an amendment to it, made before that job's first run. From then on D-024 is the record, and amendments go there, never here.
- This file is committed first on its own, for his answers, and again beside the code.

**How it was drafted and reviewed** (design workflow, 2026-11-07, read-only):
- Four scouts read the repo: the plan of record, the reusable code, what the pool must cover for his taste, and cost and schedule. One designer wrote the spec from their reports and its own reading.
- Three adversarial reviewers (V1 the registered text, V2 engineering and schedule, V3 downstream use and his taste) raised 61 findings: 2 blockers, 28 major and 31 minor. A reviser applied all 61 at their root, three of them in part:
  - V2-1: his app's usage reading stays out of git, so Q5 weighs the forecast against last week's workflows instead;
  - V2-26: render-pool gets its headroom, but he is not asked again whether Claude may register it, since D-003's amendment already says so;
  - V3-8: his measured pace is read from his votes just before Q4 is sent, which a read-only design may not do.
- Both blockers were the blind arm's checker printing the taste profile's verifiers (§2.5).

**Re-checked by hand** (laptop, read-only; scripts in the design session's scratchpad, not committed: `seed-pool/` and its `scout-taste/` and `revise/`, which print counts only and never a LoopScript text):
- step 0's `pool_briefs` under both Q1 rules (`pool_briefs_check.py`) and their twin pairs (`revise/twins.py`); the plan rule (`revise/plan16.py`, `revise/design_sim16.py`, `revise/vamp_sim16.py`, `revise/cards2.py`); the 5-word sequences DSL.md shares with TASTE.md and DRAFTS.md (`revise/shingles.py`);
- `features.source_sha8()` at main and on `wt/step0` (`source_sha8.py`: `41700aa5` in both), and `render_cfg_sha8` (`a6e3a12f`, `lofi_rl.scoring.render_pool.render_config()`);
- the 42 drafts: copy distances (`drafts_j5.py`), styles (`drafts_stats.py`), every tonic (`tonics.py`), the broken-copy transforms (`broken.py`, `broken2.py`, `broken3.py`, and the reviewers' `review-V3/fake_alt.py`, `fake_parts.py`), the mutation operators (`mutate.py`, `mutate2.py`), the long-note colour (`review-V2/colour.py`) and three house rules (`revise/house3.py`), all on the frozen palette;
- hand-built loops written for this check: a held ♭5 in the minor moods (`revise/blue_minor.py`, `blue_minor_im7.py`) and wide lines under 12 tonics (`revise/wide_fold.py`);
- the Phase-1 gate run's 500 passing loops (`runs/junk-13788483/summary/completions.jsonl`, read in memory): instruments and vamps (`base_counts.py`), verifier spreads (`base_dist.py`), P_sax parts (`review-V3/psax_parts.py`), same-cell similarity (`base_sim.py`);
- the workflow state files under `~/.claude/projects/*/*/workflows/` (tokens, durations, models and concurrency only; `revise/wf_stats.py`, `revise/wf_window.py`);
- laptop timings (`time_parse.py`), job costs (`cost_model.py`, `spec_cost.py`, `revise/cost2.py`) and listening time (`revise/listen.py`), over the run summaries they name.

Every number below names its source, and every estimate is labelled.

**The unit.** taste_pipeline.md §8 (`:327-360`): the seed pool, its writer workflow, its render and feature jobs, and the triage of 300. CLAUDE.md §9 puts it first in Phase 2 (Nov 9–15): "Seed pool of ≈2,000 loops rendered. Triage of 300 plus best/worst votes" (`CLAUDE.md:151`).
- None of it is built: there is no `src/lofi_rl/seeds/`, no `workflows/`, no `seeds/`, no `lofi_rl.dsl.check`, and no render-pool or embed job; `jobs/APPROVED.yaml` holds 14 types and none of these (`ls`, `grep`, this session).
- The plan's schedule had the seed workflow and "render about 2k" on Fri Nov 6 and the triage on Nov 7–8 (`docs/design/infra_schedule.md:336-338`). Votes now start Nov 9 at the earliest (§8.3).

- **Code:** a new package, `src/lofi_rl/seeds/`, run as `python -m lofi_rl.seeds <command>`; two workflow scripts, `workflows/seed_pool_v1.js` and `workflows/seed_pool_blind_v1.js`; a Makefile target `pull-clips`. Tests are `tests/test_seeds_*.py` (§8.1).
- **Jobs:** `jobs/render_pool.sbatch` (`lofi-render-pool`, at most 800 SU a run) and `jobs/embed.sbatch` (`lofi-embed`, at most 1,000 SU a run), registered by Claude (§4).
- **Unchanged, byte for byte:**
  - every file `features.source_sha8()` hashes: `src/lofi_rl/{brief,palette,prng,smoke}.py`, `dsl/`, `compile/`, `theory/`, `render/`, `scoring/features.py` and `scoring/render_pool.py` (`src/lofi_rl/scoring/features.py:47-49`). So this unit never changes `feat_ver`, and its checker does not live in `lofi_rl.dsl`;
  - `palettes/lofi_rain_v1.yaml`, `drafts/v1/**`, `src/lofi_rl/drafts/`, `src/lofi_rl/shootout/`, `src/lofi_rl/freeze/`, `src/lofi_rl/step0/` and `docs/prereg/*` (imported or read, never edited).
- **Writes:**
  - committed: `seeds/**` (§5.3), the workflow scripts, the two job scripts and their registry lines, `configs/calib/v1/theme.json`, D-024 and this file;
  - laptop only: `runs/render-pool-<job>/` and `runs/embed-<job>/` summaries (by `make pull`), the triage's FLACs and draw, `inbox/<date>/b<MMDD>-triage-<k>/`, the writer agents' worktrees under `.claude/worktrees/` (git and rsync ignore them, `tests/test_privacy.py:43-48`), and one folder outside the repo, `/Users/shresthrana/lofi-seed-blind/` (§2.4);
  - cluster: `$REMOTE_KEEP/runs/render-pool-<job>/`, `$REMOTE_KEEP/runs/embed-<job>/` and `$REMOTE_KEEP/cache/scorer/`.
- **Never:** no step-0 text, base-model text or vote in any committed file; nothing under `data/` except what the triage tally and builder read and the tally's private counts file (§7.4); no `manifest.hidden.json` read by any writer; his app's usage figures in no committed file (§2.9).
- **Sound claims:** none. The tools print counts, hashes and loudness measurements only.

## 1. What the pool holds

### 1.1 Sources and counts

| source | count | briefs | made by | checked by | rendered | copy pool | triage |
|---|---|---|---|---|---|---|---|
| Claude-written, informed arm | 550 (a planning count) | the 220 pool briefs (`596:<i>`) | the writer workflow (§2) | `seeds check --arm informed`, the barrier | r01 90, r02 460 | yes | 80 |
| Claude-written, blind arm | 550 (a planning count) | the same | the blind workflow (§2) | `seeds check --arm blind`, the barrier | r01 90, r02 460 | yes | 80 |
| base-model step-0 picks | 660 (a planning count) | the same | `lofi-step0` (its own unit) | `step0 check` | r01 | never | 100 |
| mutations of the kept 10 | 150 | the draft briefs d01–d10 | `seeds mutate` (§3.1) | its family rules | r01 | yes | 25 |
| broken copies | 100 | their source's pool brief | `seeds broken` (§3.2) | its kind's signature | r01 15, r02 85 | yes | 15 |
| drafts | 42 (10 kept, 32 not) | d01–d10 | committed already | `drafts.check`: 42 of 42 pass on the frozen palette (`docs/DECISIONS.md:861`) | r01, with the 28 rejects' no-bed twins as checks (§5.4) | yes | 0 |
| **pool** | **2,052** | | | | | | **300** |
| held-out (not pool) | 160 | `5101:<i>` | `lofi-step0` | `step0 check` | step 0 | never | never |
| Phase-1 ear check (not pool) | 50 | measurement stream (2026) | the gate run | `gate1 check` | the gate run | never | never |

- **2,052 against the plan's 2,040** (`taste_pipeline.md:346-353`): the plan counted 30 drafts ("Rejects / kept 20 / 10", `:352`). There are 42 now: 30 in `drafts/v1` and 12 in `drafts/v1/redo1` (`ls`, this session). The kept 10 are keep3's 6 and the redo's 4 (`drafts/v1/keep3_kept10.json` "kept"); the 32 others are 20 keep3 rejects, the 4 he tossed at the first sign-off and 8 redo rejects (`gates/GATE_palette.md:204-281`).
- **The counts are planning counts.** A step-0 pick that fails G7 at its render is dropped and never replaced, and some briefs come up short: about 2.4 by simulation, 11.9 under step-0 Q5(a) (`step0_samples.md:145, 160`). A Claude-written slot still short after its backfill rounds is reported, never forced (§2.8).
- **Per pool brief:** 5 Claude-written loops (2 or 3 of each arm) and 3 base-model loops, 8 clips. **Per draft brief:** its 3 drafts (6 for the 4 redrafted briefs) and the kept loop's mutations.
- **Not pool, but featurized** (§4.2): the 160 held-out clips (the RM gate needs them on the training clips' `feat_ver` and node type, `step0_samples.md:383`) and the Phase-1 ear check's 50, whose tiers are labels (`step0_samples.md:580`).
- **Not anywhere here:** PAR's Ref_b (1,600), rendered with the lofi-grpo ask (`step0_samples.md:384`).

### 1.2 The briefs
- **Pool briefs: step 0's 220.** D-023 records that they stand in for the planner's 220 (`taste_pipeline.md:329`; `step0_samples.md:137, 613`), and the planner here imports `lofi_rl.step0.select.pool_briefs`. Under step-0 Q1(a) they lie within `596:[0, 380)` with tempo bands 35/169/16; under (b) within `[0, 260)`, 30/176/14; 206 are shared (`pool_briefs_check.py`, this session, as `step0_samples.md:129-136`). Either way each cell holds 25 (low/*, mid/melancholy) or 24.
- **Under step-0 Q1(c)** (a new brief stream for the pool, `step0_samples.md:560`) this spec needs an amendment before the plan is built: plan.json, §6.1 and render-pool's check would name that stream.
- **A Claude-written loop belongs to a cell, not a brief.** Writers see only the scene, the energy and the mood, as the policy does (`docs/DRAFTS.md:44-45`). So the plan lays its slots on 220 positions, each a cell and its rank there, and binds positions to brief ids by D-023's Q1 answer.
  - Under either answer all 220 positions keep their cell, and 191 keep their brief id (`design_sim.py`).
  - No text is rewritten when a binding moves: every loop must compile with no sax octave fold under all 12 tonics (§2.5). 0 of the 42 drafts fold under any tonic (`tonics.py`).
  - Rubs depend on the tonic in 5 of the 42 drafts (`tonics.py`), and a line's sounding range does too (§2.5), so the rules read under the bound brief are checked again at binding.
- **Twin pool briefs.** 7 pairs of pool briefs share cell, bpm and tonic under step-0 Q1(a), and 8 under (b), so one text would be one clip in both (`step0_samples.md:152`; `revise/twins.py`). Step 0 already skips an earlier pool brief's pick (`:143`), and Claude-side loops have unique digests, so no pool clip is in two briefs; render-check asserts it (§5.1).
- **Draft briefs:** d01–d10 (seed 1105, `drafts/v1/briefs.yaml`), for the drafts and their mutations.

### 1.3 What the pool is for, and why its shape
- **What uses it:** the RM's best/worst rounds, which come "by brief" (`taste_pipeline.md:110`); the triage of 300 (`:360`); the copy pool, which holds only loops the policy did not write (D-021 item 2, `docs/DECISIONS.md:449-450`); and the calibrations of §5.5.
- **What it must cover:** what TASTE.md leaves to his ear: "Kits, pocket, keys instrument and voicing, sax source and warmth, vamps, what the keys do under the sax, long-note color, borrowed chords, everything else" (`docs/TASTE.md:95`). CD-46 left drum patterns "to the seeds and votes" (`docs/SHOOTOUTS.md:1799`).
- **What exists covers a narrow corner** (counts, this session):
  - every one of the 42 drafts plays a non-ghost backbeat on steps 5 and 13 in all 8 bars, and 39 of the kept 10's 40 long sax notes sit on a chord tone (`drafts_stats.py`);
  - 9 of the kept 10 use a vamp outside the 12 ranked in the vamp shootout; only d05_groove uses one, float (`drafts_stats.py`);
  - the base model writes piano in 498 of 500 passing loops, the jazz kit in 500, no quartal voicing (spread 391, shell 70, rootless 39) and none of the 12 ranked vamps (`base_counts.py`). By the scouts' rules (scratchpad `common.py`), none of its 2,008 long sax notes sits on a blue note outside the mode or has the keys silent under it, and 457 of its 500 loops have a rub (`base_counts.py`);
  - P_sax: the drafts sit inside every provisional band (0 of 42 outside), and the base model goes past only the wide band, 44 of 500 (`review-V3/psax_parts.py`).
- So the Claude-written 1,100 carry the coverage, through style cards that vary what TASTE.md leaves to his ear (§2.2).

## 2. The writer workflow

### 2.1 Slots, shards, arms and waves
- **Slots.** 220 positions × 5 = 1,100 slots, `p0000`–`p1099`. Positions run cell by cell in rank order: position 9r + c is the r-th brief of cell c, cells in ENERGIES × MOOD_NAMES order, skipping a cell that has run out. So any prefix of positions is balanced over the 9 cells.
- **Shards.** 16 shards: slot j of position b goes to shard (5b + j) mod 16 (5 and 16 share no factor).
  - Each shard holds 68–69 slots of 68–69 different briefs, 6 to 9 of each cell; each brief's 5 loops come from 5 consecutive shards (`revise/plan16.py`).
  - This departs from 10 briefs × 5 cards a shard, which taste_pipeline.md:336-342 implies, and from the first draft's 22 shards: 16 make each arm's wave one batch of 8 runs that all run at once (below). With 5 writers a brief, his within-brief votes compare writers and arms, as keep3's 3 personas a brief did (`docs/DRAFTS.md:3-4`).
- **Arms** (D-018, `docs/DECISIONS.md:118`, read as Q1 answers): even shards informed, odd shards blind. A brief's 5 consecutive shards hold 3 of one parity and 2 of the other, by the parity of b: 550 and 550, and 110 briefs each way (`revise/plan16.py`).
- **Lead orders** ("personas vary by shard index", `taste_pipeline.md:336`): shard mod 3 gives 0 drums first, 1 sax line first, 2 chords first. Every brief gets all three (220 of 220). The informed arm has 3, 2 and 3 shards of each (206, 138 and 206 slots), the blind arm 3, 3 and 2 (`revise/plan16.py`).
- **Waves** (`revise/plan16.py`):

  | wave | positions | slots | per shard | informed / blind | briefs per cell |
  |---|---|---|---|---|---|
  | W1 (the triage's 32 briefs and 4 spares) | [0, 36) | 180 | 11–12 | 90 / 90 | 4 |
  | W2 | [36, 97) | 305 | 19–20 | 153 / 152 | 6–7 |
  | W3 | [97, 158) | 305 | 19–20 | 152 / 153 | 6–7 |
  | W4 | [158, 220) | 310 | 19–20 | 155 / 155 | 6–7 |

- **Under Q5(c)** (3 loops a brief, 660): slot j of position b goes to shard (3b + j) mod 16, so 41–42 slots a shard; each brief gets 2 of one arm and 1 of the other (330 and 330); 192 of 220 briefs get all three lead orders and 28 get two; W1 becomes positions [0, 60), 180 slots; and each brief's cards hold 3 levels of each axis (`revise/plan16.py`).
- **A run** is one agent writing one shard's slots of one wave: 11–12 loops in W1 and 19–20 in W2–W4. The drafts precedent was 10 a writer (`docs/DRAFTS.md:3`) and the redo round's 4. The pilot measures whether a writer's tokens a loop grow along its run (§2.9).
- **Agents at a time.** At most 10 a batch, the plan's limit unless he raises it (`~/.claude/plans/sequential-skipping-cookie.md:282`). The Workflow tool has run at most 8 agents at once on this laptop: the peak of `wf_723f863d-8a1` (106 agents) and `wf_184d980d-ce2` (73), from each agent's `startedAt` and `durationMs` in their state files (`revise/wf_stats.py`). So each arm's wave is one batch of 8 runs, all at once, and a wave takes 2 run-lengths: the informed batch, then the blind one. The two sessions never run writer batches at the same time (Claude's default).
- **Pace (estimates, from the two precedents of §2.9):**
  - a W1 run-length of 11–12 loops takes 23–37 min at the first drafts' pace (127–185 s a loop) or 1.3–2.0 h at the redo round's (7.3–10.2 min a loop), so W1's two take 0.8–1.2 h or 2.7–4.1 h;
  - a W2–W4 run-length of 19–20 loops takes 40–62 min or 2.3–3.4 h, so W2–W4's six take 4.0–6.2 h or 13.9–20.4 h;
  - both before barriers, reviews and backfill. §8.3's dates are recomputed from W1's measured pace.
- **Agent count (estimate):** 16 writer runs a wave, 64 in all, plus the pilot's 2, the backfill rounds' runs, one reviewer a wave and, under Q7(a), one probe: about 75–90 agent runs.

### 2.2 The style cards

Each slot gets a card of nine levels. The axes are the plan's (`taste_pipeline.md:329-334`), changed where noted, plus four items of TASTE.md:95 that the plan's axes leave out. A fifth item, borrowed chords, gets a coverage target instead of an axis (§2.8).

| axis | levels | trace, and the change from the plan |
|---|---|---|
| vamp | informed: a ranked vamp of the mood, or "free"; blind: any vamp that fits the mood | the 12 ranked vamps are seed data, 4 a mood (`docs/DRAFTS.md:60, 220-225`). Each passes the mood check only in its own mood (`vamps.py`). The plan's axis held only the 12; 9 of the kept 10 and all 500 base-model loops use others (§1.3). The ranks come from his votes, so no blind card or blind check names one, and none is excluded for the blind arm (below) |
| pocket | straight, ghost shuffle, half-time, drag | "late snare" (`:330`) cannot be written: grids have 16 steps and no nudges (`docs/TASTE.md:100`), and the frozen palette has no per-lane timing offsets (`offset_ms: {}`, `palettes/lofi_rain_v1.yaml:420`). "Drag" (a soft hit on the 16th note after each backbeat) stands in; whether it reads late is for his ear. Half-time (the backbeat on step 9) breaks the guided backbeat check f_bb on purpose (`docs/VERIFIERS.md:67`): theory falls to 0.681–0.71 on the kept 10 (`mutate.py`). That check fades (`docs/TASTE.md:91`). The informed half-time card says it overrides DRAFTS.md's aim of a backbeat on steps 5 and 13 (`docs/DRAFTS.md:61-62`) |
| bass | lock (`bass: lock`), syncopated, sparse, push | sparse means few strikes for the brief's energy, inside its band (`docs/TASTE.md:37`) |
| voicing | rootless, shell, spread, quartal | the card names the word. The compiled voicing is classified too: a chord with no quartal candidate is voiced rootless (`src/lofi_rl/theory/voicing.py:14`) |
| sax approach | long tones, blues riff, motif + sequence, call and response with rests | the plan's (`:333`) |
| keys | rhodes, wurli, piano | added: `docs/TASTE.md:95`; `docs/DRAFTS.md:99-101` |
| kit | studio or jazz, at Q2's share (a third studio by default) | added, for the same reasons |
| under the sax's long notes | the keys hold, rest, or stab | added: "What the keys do under a held sax note is left to votes" (`docs/TASTE.md:39`) |
| long-note colour | chord tone, tension (a mode tone not in the chord), blue (a blue note outside the mode) | added: "long-note color is left to votes" (`docs/TASTE.md:33`). In aeolian and dorian the ♭3 and ♭7 are mode and chord tones, so there "blue" is the ♭5 (§2.6) |

- **The rule** (Claude's default; a seeded, byte-identical rebuild):
  - **Levels.** Within a cell, card i = 5r + j (r the brief's rank in the cell) takes level (i + offset) mod m on an axis with m levels. The offset is (cell index + axis index) mod m. So every brief gets 4 different levels of each 4-level axis and all 3 of each 3-level axis.
  - **Shuffle.** Each brief's 5 levels are then permuted per axis by a keyed SplitMix64 draw (`lofi_rl.drafts.batch.Rng`, `src/lofi_rl/drafts/batch.py:69-91`).
  - **Kit.** Studio goes on i mod 3 = 0 for a third, i mod 10 = 0 for a tenth, or nowhere (Q2).
  - **Vamps.** A brief's informed slots take a ranked vamp, except its last informed slot, which is free. Ranked vamps cycle through the mood's four within the cell. Blind slots take any vamp of the mood.
  - **Sax-line flags** (review V3-6). Every 9th slot of each arm in each mood, in plan order, carries a flag, alternating "a busier sax line: 17 to 22 notes in each phrase" and "a wider sax line: 20 to 24 semitones from its lowest note to its highest". They cross the provisional P_sax bands, at most 16 notes a phrase and 19 semitones (`docs/VERIFIERS.md:113-114`), which are to come from his Good-or-better loops (`:117`); without loops beyond them his votes cannot move the bands (§5.5). The flags are fixed in plan.json and no coverage figure changes them.
  - **Balancing.** A seeded swap search exchanges levels between two slots of one brief (so no brief's set of levels changes) until every arm × level count is within ±5 % of even. It also stops a pair of one brief from sharing more than 4 of the 8 non-vamp axes. It never moves a flag, and plan.json records what it reached.
- **What the rule gives** (`revise/design_sim16.py`, `revise/vamp_sim16.py`, `revise/cards2.py`, before the swap search; kit at a third):
  - every 4-level axis has 275 cards a level, and every 3-level axis 366–367;
  - studio gets 368 of 1,100 at a third (55 of the triage's 160) and 112 at a tenth (18 of 160);
  - 330 cards name a ranked vamp, 27–29 for each ranked vamp in its mood, with none repeated within a brief, and 770 are free (60 % of the informed arm is ranked; 48 of the triage's 160);
  - 126 slots carry a flag: each part covers 5.4–6.0 % of each arm's loops in each mood, and 20 of the triage's 160;
  - every pair of a brief's 5 cards differs in vamp and on at least one more of the plan's axes (0 of 2,200 pairs fail);
  - the two-way cells of the 4-level axes hold 50–88 against 68.8 even;
  - arm × level counts run 117–158 against 137.5 even, which the search narrows.
- **A free vamp** (the informed arm's free slot) is any two chords of the vocabulary whose mood signature favours the brief's mode by 2 or more, and that is none of the 12 ranked vamps in either order. **A blind vamp** is any such two chords, ranked ones included (reviews V1-2, V3-3). A blind writer who lands on a ranked vamp is never told so; coverage.json counts it. Each brief's 5 vamp lines must differ (§2.6).

### 2.3 What each arm sees and is held to

| | informed (8 shards) | blind (8 shards), under Q1(a) |
|---|---|---|
| where | agents of this repo's session, each in a fresh git worktree of the wave's commit (§2.4). They load the project CLAUDE.md, which the informed arm may | a separate Claude Code session that Shresth opens in `/Users/shresthrana/lofi-seed-blind/`, outside the repo (Q6). The project CLAUDE.md summarizes the taste and tells every session to read TASTE.md (`CLAUDE.md:3, 8, 141`), so the blind writers must not load it. No CLAUDE.md sits above that folder (`ls`, this session) |
| reads | its reading pack: `docs/TASTE.md` and `docs/DSL.md` as committed, and `seeds/claude/v1/pack/`, which holds `docs/DRAFTS.md` §2 (house rules, personas) and §4 (vocabulary, mood chords, vamp ranks) excerpted and the vocabulary; and its run's cards file. Nothing else (§6.3) | the folder's `DSL.md`, cut to its §3–§9 (grammar, statements, vocabulary, leniency, gates, canonical text and meaning) with no line that names TASTE.md or the shootouts (`docs/DSL.md:10, 133, 175, 363` do). DSL.md shares 5 five-word sequences with TASTE.md, all on line 133, and 4 with DRAFTS.md §2, all on line 4, and the cut leaves both out (`revise/shingles.py`); the frozen palette's vocabulary and mood chords (as prompt v1 shows them, `docs/DECISIONS.md:130-133`); its cards file in plain musical words; a one-line lead order; and `./check` |
| persona | DRAFTS.md §2.2–2.4 as written (groove, horn, chords): every card traces to a TASTE.md row (`src/lofi_rl/drafts/personas.py:1-5`). Cards decide keys, kit and voicing | drums first, sax line first, or chords first: one line each, with no word from TASTE.md or DRAFTS.md (a test checks, §8.1) |
| held to | the common rules and the house rules (§2.5) | the common rules only (§2.5) |
| shown by `./check` | every rule's verdict, plus theory and its parts, P_sax's parts, rubs and the classified axes as information | only the common rules' verdicts, E with its band, the mood margin, the card axis that missed, and a copy's J5 against "an existing loop" (§2.5) |

- **Under Q1(b)** the blind arm gets no card, no flag and no lead order: only the cut DSL.md, the vocabulary, the brief and `./check`. Rule 6 and every card-based target, the flags' included, then apply to the informed arm only, and its slots' cards stay in plan.json as a record.
- **Under Q1(c)** the blind arm gets the cards, the persona cards, the house rules and the informed check; only TASTE.md is withheld.
- **Blind cards** say "when the sax holds a note 8 steps or longer, …" for the last two axes, since "a few long notes" is a TASTE.md row (`docs/TASTE.md:33`). A loop with no such note matches them (§2.6).
- **Under Q6(b)** (no blind session), the blind arm runs in this repo's session with the project CLAUDE.md loaded. D-024 records that departure, and the audit (§6.3) still shows whether any blind writer opened TASTE.md.

### 2.4 The workflows and their files
- **Inputs.**
  - `python -m lofi_rl.seeds plan --write` writes `seeds/claude/v1/plan.json` (§5.3), a pure function of the seed and the §9 answers.
  - `seeds cards --wave W` writes one cards file a run under `seeds/claude/v1/cards/` and the pack's DRAFTS.md excerpt and vocabulary, applying plan.json and `seeds/claude/v1/card_changes.jsonl` (§2.8). They are committed before the wave starts, so each writer's worktree holds them and every loop accepted so far.
  - `seeds export-blind --wave W` copies the blind runs' cards, the cut DSL.md, the vocabulary, the folder's own CLAUDE.md (read only this folder; run `./check` after each loop) and `BRIEF.md` into the blind folder. Its `./check` wrapper runs the checker from a worktree of the wave's commit that the export makes, so the checker cannot change during a wave.
  - `seeds handoff --wave W` writes `runs/READY-<W>` there: the wave, its run ids, the plan commit and the folder's file hashes (review V3-15).
- **Scripts** (review V2-12):
  - `workflows/seed_pool_v1.js` runs the informed runs in this repo's session, each writer agent with `isolation: 'worktree'`, as the drafts' setup agent ran (`wf_6930ad87-bf9`), so its tree holds no `runs/`, `data/` or `inbox/` (all git-ignored).
  - `workflows/seed_pool_blind_v1.js` (copied into the blind folder) runs the blind runs in the blind session.
  - Every agent of both is pinned to `claude-opus-5-5`, the model of every agent of the drafts and the redrafts (`wf_6930ad87-bf9`, `wf_7f517813-71f`; `revise/wf_stats.py`). This project's workflow agents have run on two models, 246 on `claude-opus-5-5` and 179 on `claude-fable-5-1` (`revise/wf_stats.py`), so an unpinned arm could differ from the other by model.
  - Their args are the wave's runs (`seeds runs --wave W1 --arm informed --json`). Each run is one writer agent returning `{shard, wave, slots: [{slot, ok, rewrites}], verdict, stage}` through a schema.
  - Batches hold at most 10 runs (8 a wave). No agent commits, renders or edits another run's files ("Agents only parse", `taste_pipeline.md:344`).
- **Files.**
  - An informed writer writes `p<NNNN>.ls` into `stage/<run>/` in its own worktree, under `.claude/worktrees/`, which `.gitignore` and `.rsyncignore` already hold (`tests/test_privacy.py:43-48`). A blind writer writes into `/Users/shresthrana/lofi-seed-blind/runs/<run>/`, outside the repo.
  - Only the barrier moves loops into the repo (§2.7).
  - The informed prompt says: read only the pack, the run's cards and the check's output, and skip CLAUDE.md's start ritual (no CHANGELOG, no `make pull` or `make status`; `CLAUDE.md:8, 166`).
  - The informed check is one command, `PYTHONPATH=<worktree>/src /Users/shresthrana/csci596-project/.venv/bin/python -m lofi_rl.seeds check stage/<run> --arm informed [--slot pNNNN]`, the one Bash call the audit allows besides writes into the stage folder (§6.3).
- **Rewrites.** A writer runs `check` until each slot reads ok. After a slot's second failed check past its first, the writer leaves it and the slot goes to backfill ("rewritten, at most twice", `taste_pipeline.md:340`).
- **Redoing a run.** A run with more than 20 % of its slots still failing is redone once by a fresh agent, which gets the failure log. This is the plan's 20 % rule (`:341`), read for rejections, since nothing is repaired.
- **A dead agent's run** is redone once.
- **Opening the blind session** (review V2-10; Q6). The session tool takes another folder only for a session he asks for in words naming it, so Shresth opens it himself in `/Users/shresthrana/lofi-seed-blind/`, in this session's permission mode and model. His first message is one line: "Read BRIEF.md and follow it."
  - `BRIEF.md` (committed as `seeds/blind_v1/BRIEF.md`) authorizes W1–W4 and the backfill rounds up front and fixes the model.
  - It tells the session to wait for `runs/READY-<wave>` with a background until-loop, run that wave's workflow with exactly the runs the file lists, write `runs/DONE-<wave>` when its batch ends, and act on nothing else.
- **The handoff is by files** (reviews V2-10, V3-15). The main session writes READY files and waits for DONE files with a background until-loop. A message between the sessions is only a notice: a session may hold another session's message for its user's approval. Every READY and DONE file goes into `log.jsonl`.

### 2.5 The checker (`python -m lofi_rl.seeds check <run dir> --arm informed|blind [--slot pNNNN] [--json]`)
It generalizes `lofi_rl.drafts.check`: `check_one`, `references`, `copy_form` and `rubs` (`src/lofi_rl/drafts/check.py:71-302`). That validator accepts only `d\d{2}_<persona>.ls` names (`personas.py:17`) and the draft palette by default, and already takes a palette argument (`check.py:246-260`). Its table prints theory, P_sax, rubs and the nearest reference and draft by name (`check.py:305-327`), which no blind output may.

**Common rules.** Each is a failure, in both arms (rule version `seeds-check/1`):
1. **name:** `p<NNNN>.ls`, a slot of this run in plan.json;
2. **gate:** G1–G6 under the frozen palette's vocabulary in the slot's mood, with no repair at all, free ones included, and no `sax:` line (`docs/DRAFTS.md:55-57`; `check.py:192-196`);
3. **compile:** no sax octave fold under any of the 12 tonics (§1.2);
4. **energy:** E in the cell's provisional band, 0.38 / 0.59 (`src/lofi_rl/verify/knobs.py:31`);
5. **mood:** the brief's mode wins the mood signature by 2 or more;
6. **card** (the informed arm, and the blind arm unless Q1(b)):
   - the vamp is the card's ranked line exactly, or on a free card a free vamp (informed), or any vamp of the mood (blind; §2.2);
   - the keys, kit and voicing word are the card's;
   - pocket, bass, under and colour, as classified (§2.6), are the card's;
   - a flagged slot's flagged P_sax part is above 0 under the bound brief. A line's sounding range moves with the tonic, since the compiler shifts each phrase by an octave to keep most notes in 55–77: one hand-built line read 19 semitones under some tonics and 24 under others (`revise/wide_fold.py`);
   - the last two clauses are failures for a slot's first writer and information for its backfill writer, whose miss is accepted and counted as `card_miss` (review V2-5);
7. **copies**, all on the K/D copy form (`check.py:71-73`):
   - J5 below 0.6 against the 10 reference texts, the 42 drafts and every accepted pool loop of any brief. The drafts' largest values are 0.395 within a brief, 0.410 across briefs and 0.419 to a reference (`drafts_j5.py`);
   - no A or B line equal to a reference's, a draft's or an accepted loop's (D-021 Part B's rule, `docs/DECISIONS.md:553`);
   - a compiled digest that no accepted loop has;
   - a failure says "too close to an existing loop", with its J5, and names no file in either arm, so no output points a writer at a path it may not read.

**Informed arm only, also failures.** These are DRAFTS.md §2.1's house rules that a check can read (review V1-11). Until now the drafts met them only through tests/test_drafts_v1.py and the reviewers:
8. rubs 0 under the bound brief;
9. P_sax 0, so inside the provisional bands; a flagged slot is held to the other parts only;
10. B ends on another pitch class than A;
11. never more than 48 steps without a rest, with the call running into the answer and the answer into the call;
12. bar 5 brings more than a ghost or a level (`tests/test_drafts_v1.py:56-79`);
13. 2–4 sax notes of 8 steps or more;
14. A and B each open on step 1 with a tone of their bar's chord, and B ends off degree 1 (`docs/DRAFTS.md:79-85`);
15. no fill at the seam: f_seam's rules 1, 2 and 4, and rule 3 with the card's backbeat and the kick on 1 in bar 1 (`docs/VERIFIERS.md:84-88`; `docs/DRAFTS.md:75-77`). f_seam's own rule 3 asks for the backbeat on steps 5 and 13 (`src/lofi_rl/verify/variation.py:46-60`; `groove.py:15, 22-23`), which a half-time loop has not, so the checker reads it as step 9 under a half-time card;
16. something new every bar: f_new = 1 (`docs/VERIFIERS.md:82`; `docs/DRAFTS.md:73-74`);
17. rests in each phrase: f_breath = 1, a rest share of 12–60 % with one rest of 2 steps or more (`docs/VERIFIERS.md:80`; `docs/DRAFTS.md:82-83`).

All 42 drafts meet rules 15–17 (`revise/house3.py`).

**What `./check` prints** (reviews V1-1, V3-1, V2-11):
- **Informed:** each rule's verdict and the values it judges; theory and its parts, P_sax's parts, rubs, the classified axes and the characters as information.
- **Blind:** only the common rules' verdicts, E with its band, the mood margin, the card axis that missed, and a copy's J5 against "an existing loop". Nothing else, in the table or in `--json`: no theory or theory part, no P_sax or P_sax part, no rubs, no house-rule reading, no classified axis beyond the card's, no repo path. test_seeds_blind.py pins its key list (§8.1).
- **Not shown to any writer:** the backstops (§2.7).
- **Versions** (review V2-13). manifest.jsonl records each loop's rule version. A new version is a D-024 amendment and applies to later waves only, and test_seeds_v1.py checks each loop under its own version.
- **Output and timing.** The output is a table plus `--json`, with exit 0, 1 or 2. On the laptop that is about 0.13 s a loop (12 compiles at 9.46 ms plus a 12.68 ms score, `time_parse.py`), plus about 22 ms of J5 against about 1,400 references at 15.49 µs a pair (an estimate).

### 2.6 The diversity critic (deterministic code, at the barrier)
**Classifiers.** These read the parsed and compiled loop. The thresholds are Claude's defaults, pinned by tests. Each axis takes the first level that holds, in this order, so every loop gets exactly one (review V2-15):
- **pocket:**
  1. half-time: a non-ghost backbeat on step 9 and none on 5 or 13, in 6 or more of the 8 bars;
  2. drag: a ghost on steps 6 and 14 in 6 or more bars;
  3. ghost shuffle: ghosts are at least 25 % of the drum hits;
  4. straight: otherwise;
- **bass:**
  1. lock: `bass: lock`;
  2. push: 2 or more strikes on the 16th before a chord change;
  3. syncopated: 30 % or more of the strikes off both the kicks and the beats;
  4. sparse: at most 1.5, 2.5 or 3.5 strikes a bar at low, mid or high energy. The drafts' bass sits at about 2–3, 3–4 and 4–5 strikes a bar (`drafts_stats.py`);
  5. otherwise lock (a grid that follows the kicks);
- **voicing:** the compiled voicing;
- **keys, kit:** the names;
- **under the long notes:** for each sax note of 8 steps or more, f is the share of its steps the keys sound under it. Hold when f ≥ 0.75 and the keys strike nothing new under it; rest when f ≤ 0.25; stab otherwise. The loop takes the majority, a tie going to the longest note's level (then the earliest). With no such note, the level is "none";
- **long-note colour:** blue if a long note's pitch class is a blue note outside the brief's mode (the ♭5 in aeolian and dorian; ♭3, ♭5 or ♭7 in ionian; `src/lofi_rl/theory/modes.py:10`; `docs/DRAFTS.md:84`); else tension if one sits on a mode tone outside its chord; else chord tone; with no long note, "none".
  - Why outside the mode (review V2-7): read as written (any long note on ♭3, ♭5 or ♭7), 16 of the 30 minor-mood drafts are "blue"; read outside the mode, 1 is. 22 of their 113 long notes sit on the ♭3 or ♭7, the third and seventh of Im7 and Im9 (`review-V2/colour.py`).
  - The level can be written in the minor moods with no rub: on hand-built loops, a held ♭5 over Im9, in any of the four voicings, or over Im7 with a shell voicing or the keys silent under it, has no rub under any of the 12 tonics (`revise/blue_minor.py`, `blue_minor_im7.py`).
- **sax approach, information only** (first that holds):
  1. long tones: 3 or more notes of 8 steps or more;
  2. blues riff: 2 or more notes on a blue note outside the mode;
  3. motif + sequence: a 3-note interval pattern that recurs at another pitch;
  4. call and response with rests: rests make up 35 % or more of each phrase;
  5. otherwise none.
- "None" for under and colour matches a blind card's conditional wording and is never a card_miss (review V3-10). coverage.json counts it per arm. The informed arm cannot reach it (rule 13).
- The counts in §1.3 and §2.2 that came from the scouts' classifier (scratchpad `common.py`, which differs in "under") are measured again with the final code at build time.

**Rules.** Each failure sends one slot to backfill:
- every pair of a brief's 5 loops differs in its vamp line and on at least one more classified axis (pocket, bass, compiled voicing, keys, kit, under, colour). That is the plan's "differ on ≥2 axes", read pairwise (`taste_pipeline.md:342`);
- a brief's 5 loops use 5 different vamp lines and at least 3 different numeral pairs. royal_half and two_of_us share IV–III, and i_iv and pulse share I–IV (`docs/DRAFTS.md:221-225`). That is the plan's "≥3 vamps";
- **which slot goes** (review V2-4): a staged loop never displaces an accepted one. Of a failing pair, the staged slot goes, or the higher slot index when both are staged. A blind loop sent back for its vamp is told "another loop of this brief uses this vamp line", never a rank.

### 2.7 The barrier, dedupe and backstops (`python -m lofi_rl.seeds barrier --wave W`; Claude, after each batch)
1. **It runs the checker again** on every staged loop, under the loop's rule version. The writers' own runs are not trusted.
2. **Dedupe,** within the barrier and against every accepted loop:
   - an equal canonical text or compiled digest is identical;
   - J5 of 0.6 or more, or a shared A or B line, is too close;
   - a staged loop that collides with an accepted loop goes to backfill, whatever its slot index; between two staged loops the higher slot index goes (review V2-4).

   Mutations and broken copies are exempt only within their own family (§3).
3. **The critic** (§2.6) runs on every brief whose 5 slots are all accepted or staged.
4. **Backstops.** These are Claude-side, and no writer sees them:
   - no canonical sha256 in `docs/prereg/heldout_v1.json`. This runs again over every accepted loop once that file is committed;
   - J5 below 0.6 against every completion text in the laptop's `runs/*/summary/completions.jsonl`, read in memory, with only ids and values printed. A Claude-written loop must not be model output (`docs/DECISIONS.md:449-450`).
5. **The transcript audit** (§6.3).
6. **The model** (review V2-12): it reads each run's agents' `model` from the workflow state file and refuses a run whose agents ran another model than `claude-opus-5-5`.
7. **Accepting.** It moves accepted loops into `seeds/claude/v1/`, appends their manifest rows, writes the coverage report and the backfill list, and commits after `pytest -q`.
- **Cost:** all pairs of 1,100 copy forms take about 9.4 s on the laptop (`time_parse.py`).

### 2.8 Coverage and backfill
- **The report** (`seeds/claude/v1/coverage.json`, counts only), per cell and arm:
  - each card level, as carded and as classified, with "none" for under and colour;
  - keys × kit, the compiled voicing, span, snare or rim, bass lock or grid;
  - edits a loop, mid-loop drop-outs, and B opening like A;
  - P_sax's parts beyond the provisional bands, and the flagged slots as classified;
  - theory, rubs, E, characters, and free-vamp reuse;
  - ranked-vamp use per arm, the blind arm's included (review V3-3);
  - borrowed chords per mood and arm (review V1-10): vamps with a bVI or bVII chord in ionian (CD-45, `docs/SHOOTOUTS.md:1798`), and vamps with a chord outside the mood's in-mode list in every mood (`docs/DRAFTS.md:211-218`).
- **Targets** (Claude's defaults):
  - every card level within ±10 % of its planned count in each cell, as classified;
  - span 1 and span 2 each in at least 20 % of a cell's loops, and snare and rim each in at least 20 %. Cards fix neither, though a ranked vamp fixes its span;
  - a mid-loop drop-out in at least 10 % of a cell's loops ("welcome mid-loop but not in bar 8", `docs/TASTE.md:109`);
  - busy and wide each beyond its provisional band in at least 5 % of each arm's loops in each mood, through the flags (§2.2). Chromatic lines come from the sour copies (§3.2);
  - a chord outside the mood's in-mode list in at least 10 % of each mood's free vamps, per arm. TASTE.md allows borrowed chords (`docs/TASTE.md:25`), and D-022 kept them (`docs/DECISIONS.md:811`);
  - no vamp line more than 4 times in its mood among the free slots of the informed arm and all of the blind arm's.
- **Backfill** (reviews V1-6, V2-5, V2-6):
  - A shortfall first changes the cards of later waves' unwritten slots. Each change goes into the append-only `seeds/claude/v1/card_changes.jsonl` (wave, slot, old card, new card, and the counts that triggered it), so plan.json stays a pure function of the seed (review V2-14). No P_sax figure ever changes a blind card.
  - After each wave's barrier, a backfill round of fresh agents with the same cards rewrites every slot that holds no accepted loop. Its loops go through the same checker and barrier. W1's round comes before r01's list is written.
  - An accepted loop is never rewritten for coverage. An over-filled level is reported.
  - Whatever is still short after W4's backfill round is reported, never forced, so 1,100 is a planning count (D-024 supersedes `taste_pipeline.md:344`'s "backfills until the counts are met").

### 2.9 The pilot, the pace and the review
- **The pilot,** before the rest of W1: two runs, one an arm, of 6 W1 slots each, counted as W1 slots.
  - It records each agent's `tokens`, `durationMs` and `model` from the workflow state files (exact), rewrites a loop, and failures by check, in `seeds/claude/v1/log.jsonl` and the CHANGELOG.
  - It confirms where the transcripts are (§6.3), runs the audit on both, and, under Q7(a), runs the probe that must be refused.
  - From these it forecasts the 1,100. Then the pace follows Q5, whose stop check comes at the end of W1 (180 slots, the pilot's 12 among them).
  - His app's usage percent is read in session only. The commit records the outcome ("the other 920 fit in half of what is left of the week: yes or no") beside the measured tokens and minutes (review V3-16).
- **Measured precedents** (`revise/wf_stats.py`, `revise/wf_window.py`, over the workflow state files):
  - `wf_6930ad87-bf9` (the 30 drafts) started at 17:31Z on Nov 1 and ran 228.9 min, 3,828,952 tokens in all. Its three writers used 279,623 + 340,450 + 376,248 = 996,321 tokens for 30 drafts, about 33,211 a loop, and took 1,268.6–1,848.8 s for 10 each (127–185 s a loop).
  - `wf_7f517813-71f` (the 12 redrafts, written under the house rules this spec turns into checks) used 474,049 + 357,183 + 396,109 = 1,227,341 tokens for its three writers, about 102,278 a loop, and took 1,755.1–2,453.9 s for 4 each (7.3–10.2 min a loop).
  - **So the 1,100 would take about 36.5 to 112.5 million tokens of writing alone** (estimate), before checks, reviews, backfill and barriers.
  - For scale, the 41 workflows started from 23:42Z on Oct 30 to 19:55Z on Nov 4 used 79,295,351 tokens. Their last two agents (`wf_d0e59516-024`'s fix and verify) ended in error at 22:54Z on Nov 4, when they stopped at the weekly usage limit (`CHANGELOG.md:188`). No workflow started again until 19:44Z on Nov 6.
  - The same week must also carry this unit's builds, step 0's review and merge, the rounds unit, which must serve by about Nov 11, and the RM unit.
- **Review.** One adversarial reviewer agent a wave, in the informed session.
  - It reads 30 accepted loops drawn with the plan seed (15 from each arm), their cards and the coverage report.
  - It reports systematic slips of the kinds the drafts' reviews found: rubs, answers ending on the call's pitch, bar 5 changing only a ghost, span tracking energy (`tests/test_drafts_v1.py:1-6`; `docs/DRAFTS.md:465`).
  - Its findings change the checker (a new rule version, for later waves only) or later cards (card_changes.jsonl), never an accepted loop by hand. A rule drawn from TASTE.md changes the informed arm only.

## 3. Mutations and broken copies (deterministic laptop code)

### 3.1 Mutations of the kept 10 (150)
- **Parents:** `lofi_rl.freeze.kept_now()`: keep3's 6 and redo1's 4 (`drafts/v1/keep3_kept10.json`). All 10 play the jazz kit, with keys piano 4, wurli 4 and rhodes 2 (`docs/DECISIONS.md:860`).
- **15 each:** 5 families of 3 variants. Each variant changes one thing. Measured on the kept 10 as K/D J5 to the source (`mutate.py`):

  | family | its 3 variants | the lines it may change | measured on the kept 10 |
  |---|---|---|---|
  | keys | 2 other voicing words whose compiled voicings differ from the source's; 1 other keys instrument | `keys:` | voicing 0.926–0.949, and 2 of 30 word swaps compile to the source's own loop, so they are skipped; keys 1.0, since the K/D form hides the instrument |
  | groove | the studio kit (under Q2(c), a drag pocket instead); one pocket change (ghost shuffle, or straight without ghosts if the source is a ghost shuffle); one bass change (`lock` and the kick's grid swapped) | `drums:`, the lane lines, the edits and `bass:` | kit 1.0; ghost shuffle 0.899–0.975, band kept 9 of 10; straight 0.821–0.974 |
  | answer | B's bars 3–4 a scale step up; B's first long note moved to a blue note outside the mode; B's two shortest notes made rests | `B:` | step up 0.872–0.929, adding a rub in 6 of 10; blue 0.971–0.985 |
  | vamp | 3 other vamps of the mood that keep rubs 0 and P_chrom 0: the ranked ones in rank order first, then the mood's in-mode chords (`docs/DRAFTS.md:211-218`) in sorted pairs | `vamp:` | 0.887–1.0. 14 of 40 ranked swaps add rubs, and only 5 of the 10 have 3 clean ranked swaps (`mutate2.py`) |
  | energy | each other band, by a density rule repeated until E is in band; one mid-loop drop-out bar (bars 3–6, never bar 8) | the lane, comp, bass and edit lines | down 0.847–0.957, up 0.914–0.971; one pass reached the band in 8 of 10 |

  Two changes at once sit at 0.766–0.927 (`mutate2.py`), so a mutation makes one change.
- **Each mutation must** (review V2-16):
  - change no canonical line outside its family's; its classified axes are recorded, not required, since a vamp swap also moves the compiled voicing and can move the colour;
  - compile to a digest unlike its source's and every sibling's;
  - pass the gate with zero repairs, compile with no fold, keep its brief's energy band and mood margin ≥ 2, and have rubs 0. Mutations are cousins of loops he kept, not sour ones.

  A variant that fails takes its family's next candidate in a fixed order, and a family left short is reported.
- **Briefs.** A mutation takes its source's draft brief. An energy shift takes the draft brief of its new cell in the same mood: d01–d09 hold one cell each (`docs/DRAFTS.md:21-31`).
- **Copies.** A mutation is exempt from J5 and shared lines against its parent and siblings, and is checked like any pool loop against everything else. Both copy forms' J5 to the parent, and the event-set similarity, are recorded. Mutations go in the copy pool (D-021 item 2).
- **Commands:** `python -m lofi_rl.seeds mutate --write` and `--check` (a byte-identical rebuild).

### 3.2 Broken copies (100)
- **Sources:** 100 different accepted Claude-written loops: 15 from the triage's 160, drawn after W1's backfill round, and 85 from W2–W4, drawn after W4's backfill round, so no source is rewritten under its copy (review V2-6).
  - Never a draft, a mutation or a step-0 text: a copy of a step-0 text would be model output and stay out of the copy pool (`step0_samples.md:628`), and D-021 item 2 puts broken copies in it (`docs/DECISIONS.md:449`).
  - At most one per brief, balanced over cells and arms, drawn in a fixed order with the plan seed.
- **Kinds:** his four dealbreakers (`docs/TASTE.md:27`). Each is a pure function of the loop, confirmed by a signature that the base model's loops almost never show:

  | kind | transform | applies, of the 42 drafts | P_sax | confirmed by | base-model loops with the signature (of 500) |
  |---|---|---|---|---|---|
  | sour | every sax note's accidental raised by one (`shootout.sour.raise_accidentals`, the shootouts' anchor, `docs/SHOOTOUTS.md:182-186`); a source with a `#` note is refused | 36, all with P_chrom 0.3 | 0.0–0.14 (its chromatic part) | P_chrom ≥ 0.2 | 1 |
  | no groove | each drum lane rotated (kick 3 steps, snare or rim 1, hats 1, open hat 5, shaker 3, the bass grid 5, per-bar edits likewise) | 42, band kept in 41; f_bb 0 in all, groove 0.20–0.544 | 0, the source's, in 42 | f_bb = 0 and groove ≤ 0.55 | 0 with groove ≤ 0.55; 6 with f_bb 0 |
  | boring | every lane, the comp and the bass reduced to their first bar, every per-bar edit removed, and B set to A | 42, band kept in 42; f_new 1/7 and f_ans 0 in all | 0, the source's, in 42 | f_new ≤ 1/7 and f_ans = 0 | 2 with f_ans 0; 4 with f_new ≤ 0.43 |
  | fake | every drum hit at one level (X and x become x, ghosts removed); every sax note of 4 steps or more cut to 3 steps, the rest of its length a rest, so no scoop, vibrato or swell can fire (`src/lofi_rl/compile/expression.py:6-11`) and every onset stays | 42, band kept in 40, mood in 42; the note count the source's in 42 | 0, the source's, in 42 | no sax note over 3 steps, and no ghost or accent | 0: every one holds a note of 8 steps or more |

  Sources: `broken.py`, `broken2.py`, `broken3.py`, `review-V3/fake_alt.py` and `base_dist.py` (re-run this session).
- **Why the fake transform keeps the onsets** (review V3-4). The first draft cut each long note into 2-step notes. On the 42 drafts that gave 40–54 sax notes against their sources' 19–30, with P_sax's busy part at its 0.1 cap in 42 of 42 (`review-V3/fake_parts.py`). Busy lines would then exist in the pool almost only as fake copies, and their Toss label would teach "busy = bad" on top of P_sax.
- **Each copy must also** pass the gate with zero repairs, compile under its source's brief, keep its source's energy band and mood margin ≥ 2, keep its source's P_sax (every kind but sour), and later pass G7. Then it can stand in a round of its source's brief beside its source (`taste_pipeline.md:46`). A source that fails takes the next in the draw.
- **The vocabulary.** `check_sour` parses with the default vocabulary, which has no `studio` kit (`src/lofi_rl/shootout/sour.py:155-160`; `src/lofi_rl/dsl/vocab.py:89-91`). So the pool's own check uses the frozen palette's vocabulary, and `shootout/sour.py` stays untouched.
- **Under Q3(b)** no fake copies are made: the 100 are 34, 33 and 33 of the other three kinds, and the triage's 15 are 5, 5 and 5.
- **Commands:** `python -m lofi_rl.seeds broken --write` and `--check`.

### 3.3 Calibrating the kinds (reviews V3-5, V1-3, V3-14)
The rule below is fixed in D-024 before his first triage vote.
- **The triage holds 15 copies:** 4 sour, 4 no groove, 4 boring and 3 fake, each made from one of the triage's Claude-written loops (§7.1), as hidden anchors (kind `anchor`, expected Toss).
- **An informative showing** is a triage copy whose source he rated Fine or better, or, later, a best/worst round holding the copy and its source in which he did not answer "all four bad". A showing is **caught** when the copy is Toss, or is picked worst.
- **A kind is caught** when at least 3 of its informative showings are caught, and at least two-thirds are. It is **not caught** once it has 6 or more informative showings and fewer than two-thirds are caught. Otherwise it is **undecided**.
- **Why a floor of 3, and why the statuses keep moving.** The triage's copies decide little. Under the first draft's two-thirds rule, a kind he Tosses half the time would read caught (3 of 4) with probability 0.31, and one he Tosses 80 % of the time would miss with probability 0.18; three fake copies needing 2 give 0.50; and Q4(b)'s 2 copies a kind, needing both, give 0.25 at p = 0.5 and only 0.81 at p = 0.9 (binomial arithmetic, this session). So the statuses are recomputed from every informative showing at each RM fit, the rounds' included, and a kind read caught on the triage can turn not caught.
- **A caught kind** makes later hidden anchors, and its copies keep the plan's synthetic label, Toss at weight 0.25 (`taste_pipeline.md:173`).
- **A kind not caught** makes no more anchors, and its synthetic labels are dropped. His own tiers and votes on its copies still count, like any other.
- **An undecided kind** carries no synthetic label and makes no anchor. Its later copies are served as ordinary clips (kind `none`) in rounds of their source's brief, beside the source, until it is decided.
- **Sittings.** A sitting's weight counts only its missed anchors of caught kinds, as the statuses stand at each RM fit. A sitting whose only missed anchors are of a kind not caught, or undecided, keeps full weight, and its held-out rounds still count. D-013 says such a sitting counts at half weight and never enters the held-out set (`docs/DECISIONS.md:88`), so D-024 lists this under what it supersedes, and Q3 tells him.
- **The 5 % anchors** (`taste_pipeline.md:46`) come from caught kinds only. Until one is caught, rounds carry no anchors and the 5 % waits; D-024 records it.
- **Prior art:** sour copies were the shootouts' 3 anchors (`docs/SHOOTOUTS.md:182-186`). He caught Day 1's and Day 3's, and marked Day 2's "no clear best" and "no clear worst", so that micro-session counted at half weight (`gates/GATE_palette.md:80, 109, 196`).

## 4. The jobs

### 4.1 `lofi-render-pool` (`jobs/render_pool.sbatch`)
- **Shape.** main, 1 node, 32 CPU, 32 GB, 20 min, `--no-requeue`, `--output=/scratch1/%u/lofi-rl/logs/%x_%j.out`. The cost line: `Cost: 32 CPU + 8 (32 GB) = 40 SU/min x 20 min -> at most 800 SU.` The 200 SU under the cap leave room for a later change, such as more memory after the first MaxRSS, without his yes (review V2-26).
- **Registered by Claude:** `make approve JOB=jobs/render_pool.sbatch MAX=800 BY=Claude`, then `python3 scripts/approve_job.py --check`, under D-003 as amended on 2026-11-06 (`docs/DECISIONS.md:26`; `config/budget.yaml:17` `self_approve_max_su: 1000`), with a CHANGELOG line. It comes after the script's last change and before the commit that adds its test (§8.2).
  - This corrects "`lofi-render-pool` (1,440 SU in the plan) … stay his" (`CHANGELOG.md:172`), and "render-pool is Shresth's" in step 0's §3, which cites it (`step0_samples.md:385`). The 1,440 is a 30-minute planning row at 48 SU/min, sized for an assumed 3 s a loop (`docs/design/infra_schedule.md:239, 248`).
- **The argument:** one committed list, `seeds/render/r<NN>.json`. Anything else stops the job before any work (exit 2).
- **The steps** (`MODE=scorer`):
  1. **check** (`timeout -k 10 120`):
     - the snapshot's DECISIONS.md holds D-024;
     - each listed `.ls` file has its listed sha256;
     - each step-0 row's text_sha256 is found in `$REMOTE_KEEP/runs/step0-<job>/summary/completions.jsonl`, whose sha256 equals the pin in `docs/prereg/step0_v1.json`;
     - every brief is `brief_streams.sample_brief(seed, index)` or its `drafts/v1/briefs.yaml` entry;
     - the palette's sha256 is `09f80a64…`.
  2. **render** (`timeout -k 10 846 … --deadline-s 786`):
     - 4 processes of 8 lanes each, over quarters of the list, each lane one render with 1 worker: the shape `lofi-drafts` measured, 8 at a time with 1 worker each (`docs/DRAFTS.md:297`). 32 threads in one interpreter are unmeasured (review V2-23);
     - each render through `junk.render_clip` (FLAC and manifest, `src/lofi_rl/rl/junk.py:1038-1060`), which names files `clips/<i:04d>` by list row (`junk.py:1043`), as step 0's do (`step0_samples.md:306`); records.jsonl maps each row to its clip id (review V2-17);
     - the 28 rejects' no-bed twins: `lofi_rl.freeze.signoff.twin_doc` of the frozen palette at the levels keep3 played, as the sign-off job builds twin0 (`src/lofi_rl/freeze/signoff.py:254, 804`), imported unchanged. They are checks, never pool clips, and are not featurized (§5.4);
     - each clip's record appended and fsynced as it finishes, as junk does with clips.json;
     - no render starts after the deadline;
     - every outcome is recorded with the render-code key (the stamp and `features.source_sha8()`), as step 0 records its own (`step0_samples.md:387`).
  3. **copy-out**, D-022's pattern (`docs/DECISIONS.md:961`):
     - the run is written at `RUN=${TMPDIR:-/tmp}/render-pool-<job>` and copied to `OUT=$REMOTE_KEEP/runs/render-pool-<job>`: the summary under `timeout -k 2 30`, the clips under `timeout -k 2 120`;
     - the copy runs after the render, and from EXIT and TERM traps;
     - a used OUT is refused (`step0_samples.md:338`);
     - `LOFI_CACHE` is on KEEP.
- **Budget:** 130 + 856 + 32 + 122 = 1,140 s, which is 60 s under 1,200 s (junk's 60 s rule, `tests/test_junk_job.py:91`).
- **The lists:**
  - r01 holds W1's accepted loops (up to 180), their 15 broken copies, the 150 mutations, the 42 drafts, the 660 step-0 picks and the 28 twins: 1,075 renders, 1,047 of them pool clips (`revise/cost2.py`);
  - r02 holds W2–W4's 920, every backfill round's loops, W1's G7 rewrites, r01's skips and the other 85 broken copies: 1,005 at planning counts;
  - r03 is a contingency for what r02 leaves. Each list is made on the laptop from the pulled records. Nothing on the cluster reads an earlier run.
- **A G7 failure:**
  - a Claude-written loop's slot is rewritten (backfill) and rendered in the next list;
  - a mutation or broken copy takes its next candidate;
  - a step-0 pick is dropped and never replaced (`step0_samples.md:145`).
- **Cost** (estimates, `revise/cost2.py`):
  - **The rate.** A render costs 7.15–10.46 billed CPU-s a clip in measured runs: junk-13788483 8.32, drafts-13656049 7.15, the three shootout jobs 7.98–10.46. The central planning value is 8. The xeon-4116 nodes render 1.59 times slower than epyc-7513 (render-check fixtures; `spec_cost.py`).
  - **r01:** about 7.5 min and 299 SU (376 high, 566 worst). **r02:** about 7.2 min and 286 SU (358 high, 535 worst). About 585 SU for both, at most 800 a run.
  - **By the 786 s deadline** a run finishes about 2,829 clips at the central rate, 2,043 high and 1,209 worst.
- **Memory and storage.**
  - **Memory:** laptop runs of 8 renders at a time peaked at 2.63–3.30 GB (`docs/SHOOTOUTS.md:1353`; `CHANGELOG.md:1331`), so 32 at a time need about 10.5–13.2 GB of the 32 (an estimate). Linux MaxRSS is unmeasured, so the job records its largest child's, and the CHANGELOG adds `sacct` MaxRSS.
  - **Storage:** about 5.26 GB of FLACs on KEEP (2,052 × the gate run's mean 2,562,058 B, `runs/junk-13788483/summary/clips.json`). Check `myquota` first: `/project2/anakano_429` is 1 TiB shared by the class, with 88 GiB used on 2026-10-30 (`docs/DECISIONS.md:14`).
- **Records** (`summary/`): `args.json`, `check.json`, `records.jsonl` (per clip: list row i, source, clip id, canon_sha256, brief id, outcome, n, LUFS, true peak, audio_sha256, FLAC sha256 and bytes, render_s, iterations, the render-code key and the palette label), `render.json` (counts, timings, node, the lscpu CPU model, nproc, and `platform_info()`, the audio stack) and `stages.json`.
  - `clips/<i:04d>.flac` and their manifests stay on KEEP. Each manifest holds the loop's canonical text (`src/lofi_rl/render/pipeline.py:246`), so the step-0 picks' manifests are model texts on KEEP (§6.2).
  - Step-0 rows carry hashes only.
- **The driver** `src/lofi_rl/seeds/render.py` imports its audio code inside functions, since the policy env's build imports every module outside its scorer-side list (`jobs/env_policy.sbatch:32-37`; `step0_samples.md:228`).

### 4.2 `lofi-embed` (`jobs/embed.sbatch`)
- **Why it renders again.**
  - The features need the float mix and the music-only stem (`src/lofi_rl/scoring/features.py:3-8, 214-253`), and the FLAC is "the 16-bit dithered listening copy" of the mix (`docs/SCORER.md:100-102`).
  - Storing float stems would take about 37.6 GB on the class's shared KEEP, which only a job could delete (`cost_model.py`).
  - So embed renders and featurizes through the scorer's own path (`POST /score`), which also returns G7 and the audio hash: a determinism check against render-pool (§5.2).
- **Shape.** gpu, 1 L40S (`--gres=gpu:l40s:1`), 24 CPU, 32 GB, 25 min, `--no-requeue`. The cost line: `Cost: 24 CPU + 8 (32 GB) + 8 (1 GPU) = 40 SU/min x 25 min -> at most 1,000 SU.` Registered by Claude: `make approve JOB=jobs/embed.sbatch MAX=1000 BY=Claude`, after the script's last change and before the commit that adds its test. It sits at the cap, so a change that raises its worst case needs his yes or a shorter walltime.
  - The L40S is the GPU of the GRPO smokes and the scorer smoke (`docs/DECISIONS.md:597-598`; `CHANGELOG.md:597, 662`). Feature records never cross hardware: the cache keys on the CPU model and the GPU name (`docs/SCORER.md:286-290`).
- **The steps** (`MODE=scorer`):
  1. check the list (`timeout -k 10 60`);
  2. `export LOFI_CACHE=$REMOTE_KEEP/cache/lofi-rl` before any server starts: `scripts/job_env.sh:22` defaults it to scratch, and the server's render workers call `pipeline.render` with no cache argument (`src/lofi_rl/scoring/render_pool.py:125`) (review V2-19);
  3. start `python -m lofi_rl.scoring.server --sock $TMPDIR/… --palette palettes/lofi_rain_v1.yaml --workers 22 --cache $REMOTE_KEEP/cache/scorer --theme configs/clap_theme_v1.yaml`, with no `--rm` and no `--calib`, under `timeout -k 30` of the time left, as the scorer smoke runs its servers (`jobs/scorer_smoke.sbatch:47`). Phase-2 jobs pass the frozen palette, since the server's default stays dev_gu (`docs/DECISIONS.md:923`);
  4. wait for `/health` (at most 180 s; the featurizer loaded in 37.32 s cold, `runs/scorer-smoke-13651611/summary/scorer_smoke.json`);
  5. `python -m lofi_rl.seeds embed --list … --deadline-s 1020 --chunk 128` (`timeout -k 10 1080`) sends `/score` in chunks of 128. A chunk takes about 72 s at the high rate plus one slow render's drain, inside the 150 s render timeout a batch and the client's 300 s (`render_pool.py:49`; `client.py:32`); the cap is 1,024 items a request (`protocol.py:37`). Transient items (`render-error`, `render-timeout`, which the cache does not keep, `src/lofi_rl/scoring/cache.py:1-17`) are retried once at the end, and whatever is left goes into a laptop-made e1b list;
  6. a trap stops the server, and the summary is copied to KEEP (`timeout -k 2 30`).

  Budget for e1: 70 + 190 + 1,090 + 32 = 1,382 s, which is 118 s under 1,500 s.
- **e2's overlap first** (reviews V1-14, V2-8). A record "is a function of the clip id and feat_ver alone" and a hit is served without a render (`cache.py:1-17`; `src/lofi_rl/scoring/core.py:293-305`; `docs/SCORER.md:284-285`), so overlap clips sent through KEEP's cache on e1's hardware would compare a record with itself.
  - So e2 first starts a server with `--cache $TMPDIR/overlap`, empty, scores the 20 overlap clips (`timeout -k 10 120`), and stops it. Then it starts the server on KEEP's cache for the rest.
  - On the node, the driver compares the 20 fresh records with e1's records in KEEP's cache (each array, Audiobox and audio_sha256). It writes `summary/overlap.json`: the largest absolute and relative difference per array, audio equality, and the cache each clip came from (`diag.cache`, `docs/SCORER.md:116`).
  - The precedent: the scorer smoke's two cold caches gave max_abs 0.0 on every array (`scorer_smoke.json` `feature_diff_a_b`).
  - Budget for e2: 70 + 190 + 130 + 190 + 828 + 32 = 1,440 s, 60 s under 1,500 s, with the main embed under `timeout -k 10 818 … --deadline-s 758`.
- **The held-out and ear-check stacks.** For §5.2, the driver also reads the `platform` block (less `cpu`) of each held-out and ear-check clip's manifest on KEEP (`step0_samples.md:684`), records it, and copies no text.
- **Per clip** (`summary/records.jsonl`): clip id, source, brief id, hard and gate, `sig` {pq, pc, ce, cu, clap_margin, centroid_hz}, the render's audio sha256 and LUFS, the hardware key, the cache it came from and feat_ver. `summary/embed.json` adds the node's lscpu CPU model and `platform_info()`. The features themselves stay in the cache on KEEP.
- **The runs:**
  - e1 after r01: r01's 1,047 pool clips, the 160 held-out clips and the Phase-1 ear check's 50, 1,257 in all;
  - e2 after r02: r02's pool clips (1,005 at planning counts) and 20 of e1's clips again through the fresh cache, 1,025 in all;
  - **e3, a contingency** (review V2-20): when e1's and e2's hardware keys differ, the union list (2,262 clips) is sent again before the RM unit reads any feature. Clips already cached under the node's key are disk hits that cost no render, so e3 renders only what its key lacks; its deadline leaves the rest to e3b. Each run records its node's CPU model, since a pinned CPU model is not reliable (`docs/kit/LESSONS.md:245-249`).
- **Texts.** Claude-side rows come from the job's snapshot. Step-0 rows and the ear check's come from KEEP by text_sha256: `runs/step0-<job>/` and `runs/junk-13788483/summary/completions.jsonl`, pinned by their sha256. Every cache record stores the canonical text (`cache.py:74-76`), so e1 adds the 660 picks', the 160 held-out and the 50 ear-check texts to KEEP's cache (§6.2).
- **Cost** (estimates, `revise/cost2.py`, at 8 CPU-s a render on 22 workers and 0.2235 s of features a clip on an L40S, from `scorer_smoke.json` pass a, plus each chunk's drain on its slowest render, job B's 17.39 s less its 7.707 s mean, `docs/DECISIONS.md:866`):
  - e1: about 12.6 min, 503 SU (633 high);
  - e2: about 11.9 min, 474 SU (578 high), the overlap server's start included;
  - about 977 SU for both, at most 1,000 a run;
  - e3, if needed: about 422 SU (526 high) on e1's key, 503 (633) on e2's, and about 825 SU on a third key, which its deadline would cut at the high rate.
- **Before e1:** `ls $HF_HOME/hub` for the three pinned snapshots, since scratch is purged (`CLAUDE.md` §1). If they are gone, the registered `jobs/fetch.sbatch` fetches them again.
- **feat_ver.** This unit changes no file that `source_sha8` hashes, and `source_sha8` is `41700aa5` at main and on `wt/step0` (`source_sha8.py`). So e1, e2 and run 1 share it unless another unit edits those files. Then the features are taken again, at up to 1,000 SU a run.

### 4.3 SU

| job | runs | SU/min | at most a run | expected (estimate) |
|---|---|---|---|---|
| `lofi-render-pool` | r01, r02 (r03 if needed) | 40 | 800 | about 299 + 286 = 585 |
| `lofi-embed` | e1, e2 (e1b, e3, e3b if needed) | 40 | 1,000 | about 503 + 474 = 977 |
| this unit | 4 runs | | 3,600 | **about 1,562** |
| `lofi-step0` (its own unit) | 1 | 18 | 810 | about 301 (`step0_samples.md:372`) |

- SU used so far: 2,583.8 of the 1,000,000 target (`CHANGELOG.md:6`).
- Every run's SU goes in the CHANGELOG from `sacct` or `python3 scripts/session_report.py` (`CLAUDE.md` §4).

## 5. Records, validity, and what is committed

### 5.1 Render-pool validity (`python -m lofi_rl.seeds render-check runs/render-pool-<job>`, laptop, after `make pull`)
- **Complete:** every listed clip has an outcome: ok, G7, another gate, an error, or skipped at the deadline. A skip is no failure; the clip goes into the next list.
- **As listed:** the list's sha256 is the committed file's; every check of step 1 holds; and every step-0 row's clip id equals `sets.json`'s.
- **Step 0's picks:** `step0.select`, given every scored row, step 0's clips.json records and this run's G7 outcomes, gives step0_v1.json's picks less the G7 drops. A recomputed pick that differs stops the registry (step 0's ask, `step0_samples.md:659, 717`).
- **One stamp:** the snapshot's commit, with dirty 0, holds the list.
- **Render code:** the recorded `source_sha8` equals `features.source_sha8()` at the stamp's commit, read with git.
- **In spec:** every ok render sits at −14 LUFS within 0.1 LU, with a true peak at or below −1 dBTP (`CLAUDE.md` §5) and n = `loop_samples(bpm)`.
- **Ids:** every clip id equals the formula over its recorded inputs (`taste_pipeline.md:76`; `render_cfg_sha8` `a6e3a12f`), and no pool clip id is under two brief ids (§1.2).
- **An invalid run** decides nothing. It is fixed and run again into its own folder, and nothing under `runs/` on the cluster is deleted (`CLAUDE.md:175`).

### 5.2 Embed validity (`embed-check runs/embed-<job>`)
- **One version and one machine:** one feat_ver and one hardware key in the run, the palette `09f80a64…`, the theme file's sha256, and no RM and no calibration loaded.
- **The rule for audio** is step 0's (`step0_samples.md:393-396, 684`). The render-code key holds each job's stamp, so two jobs never share it. Under the same `source_sha8`, palette, `render_cfg_sha8` and audio stack (`platform` less `cpu`), a different `audio_sha256` is a determinism violation. Otherwise it is logged as render drift.
- **Pool clips:** each clip's audio_sha256 against render-pool's. A violation: the job's records are reported and not used.
- **The held-out 160 and the ear check's 50** (reviews V1-7, V2-21), against their first renders: `heldout_v1.json`'s audio_sha256 and render-code key (`step0_samples.md:461-463`), and `runs/junk-13788483/summary/clips.json`, with the gate run's `source_sha8` read with git at its stamp. The stacks come from the manifests on KEEP (§4.2).
  - A violation: e1's held-out features are not used, and the run is reported.
  - A held-out clip that fails G7 in e1 is reported, and its tuple is dropped from the gate (step-0 Q8(a), `step0_samples.md:396`).
- **The overlap:** e2's 20 fresh records (§4.2). On e1's hardware key they must be identical. On another key the largest difference is recorded, and e3 follows before the RM unit reads any feature.

### 5.3 Files

| file | what | committed |
|---|---|---|
| `seeds/claude/v1/plan.json` | `lofi-rl/seed-plan/1`: the seed (596), the rule, the 220 positions (cell, rank), their binding to brief ids by D-023's Q1, and per slot its shard, arm, lead order, wave, card and flag. A pure function of the seed and the answers: byte-identical on rebuild (`seeds plan --check`) | yes |
| `seeds/claude/v1/card_changes.jsonl` | the append-only card changes of §2.8 | yes |
| `seeds/claude/v1/cards/<run>.md`, `seeds/claude/v1/pack/` | each run's cards, as the writer read them, and the informed pack's DRAFTS.md §2 and §4 excerpt and vocabulary | yes |
| `seeds/claude/v1/p<NNNN>.ls` | the accepted loops, one flat folder that the copy pool lists non-recursively (`src/lofi_rl/rl/config.py:136-143`) | yes |
| `seeds/claude/v1/manifest.jsonl` | per loop: slot, file sha256, canon_sha256, compiled digest, cell, brief id, card, flag, arm, lead order, run, the writer's model, the rule version, the checks, the classified axes, nearest J5s, rewrites, and wave | yes |
| `seeds/claude/v1/log.jsonl` | per run and barrier: times, each agent's tokens, minutes and model (from the workflow state files), slots, rewrites, failures by check, collisions, backfills, the READY and DONE files, the audit's result, and Q5's yes or no. Never the app's usage readings | yes |
| `seeds/claude/v1/coverage.json` | §2.8's report (counts) | yes |
| `seeds/blind_v1/` | the blind folder's templates: its CLAUDE.md, BRIEF.md, the lead orders, the DSL.md cut and `./check` | yes |
| `seeds/mutations/v1/*.ls`, `manifest.jsonl` | the 150, with parent, family, variant, brief, both J5s and the event-set similarity | yes |
| `seeds/broken/v1/*.ls`, `manifest.jsonl` | the 100, with kind, source, brief and signature values | yes |
| `seeds/render/r<NN>.json`, `seeds/embed/e<N>.json` | the job lists. Step-0, held-out and ear-check rows give only run, brief, k and the text and canonical sha256s | yes |
| `seeds/registry_v1.jsonl` | the pool registry (§5.4) | yes |
| `seeds/drafts_map_v1.json` | each draft's clip ids under draft_day2 (keep3), palette A (sign-off) and B (the pool), with the audio checks (§5.4) | yes |
| `seeds/readouts_v1.json` | §5.5's readouts (numbers only) | yes |
| `configs/calib/v1/theme.json` | med, mad and theme_sha256 (`src/lofi_rl/scoring/calib.py:50-56`) | yes |
| `runs/render-pool-*`, `runs/embed-*` | the summaries, by `make pull` | never (`.gitignore`) |
| triage FLACs, `draw.json`, inbox batches, hidden manifests | §7 | never |

### 5.4 The registry and the drafts map
- **`seeds/registry_v1.jsonl`** (`lofi-rl/seed-registry/1`): one row a pool clip.
  - clip id, source, and the brief id with its energy, mood, bpm and tonic;
  - for Claude-written loops: arm, lead order, card and flag; for mutations and broken copies: parent and operator or kind;
  - the text: the file and its sha256 for Claude-side rows, and for step-0 rows only {run, brief id, k, text_sha256}, never the text;
  - canon_sha256 and the compiled digest;
  - render {job, row, outcome, audio_sha256, FLAC sha256, n, LUFS, true peak, render-code key, palette label} and embed {job, feat_ver, hw, sig};
  - the symbolic readouts: E, band, mood margin, theory, P_sax's parts, rubs and the classified axes. For step-0 rows they are computed from the laptop's copy of the texts, in memory.
- **`seeds/drafts_map_v1.json`** links each draft's keep3 clip id (palette draft_day2, no rain bed), its sign-off clip ids (palette A, `lofi_rain_v1@d3590c8d`) and its pool clip id (palette B), by canon_sha256 and brief.
  - It checks that the sign-off render's audio sha256 equals the pool render's: A and B differ only in `frozen:`, which no render reads (`docs/DECISIONS.md:789-790`).
  - **Which keep3 renders the frozen palette's music is known for** (review V1-17). The kept loops' no-bed twins equalled their source renders 10 of 10 in both sign-off jobs: keep3's kept 10, the 4 later tossed among them, in 13701026 (`docs/DECISIONS.md:846`), and the current kept 10, keep3's 6 and redo1's 4, in 13706355 (`:836, 844`). The 20 keep3 rejects and the 8 redo rejects were never twinned.
  - So r01 renders those 28 rejects' no-bed twins (§4.1), and the map checks each twin's audio_sha256 against the pcm_sha256 in keep3's or redo1's renders.json. A reject whose twin differs keeps no keep3 label, and the count is reported.
  - Its inputs are the renders.json files of drafts-13656049, -13703830, -13701026 and -13706355 on the laptop.
  - The RM loader uses it (§7.5).

### 5.5 Calibrations read from the pool

| calibration | the plan's rule | this unit | why |
|---|---|---|---|
| energy bands | tertiles of E over the pool (`docs/design/instrument.md:294`; `docs/VERIFIERS.md:167-172`) | reports the tertiles of the Claude-written and base-model loops, each source alone and together. The bands stay 0.38 / 0.59 | circular: every Claude-written loop must land in the provisional bands (rule 4). The base model's E already falls low 0.278–0.43, mid 0.447–0.623 and high 0.686–0.748 (`base_counts.py`), with the band met in 498 of 500 (`runs/junk-13788483/summary/junk.json`) |
| near-identity threshold | the p99.5 of same-brief Claude pairs, "so close cousins pass" (`taste_pipeline.md:252`) | reports that value and the base model's same-cell distribution. NEAR stays 0.875 (`src/lofi_rl/verify/penalties.py:39`) | the critic pushes a brief's Claude loops apart: the drafts' same-brief event-set similarity reaches 0.490 at most (`drafts_j5.py`). Meanwhile 55.5 % of same-cell base-model pairs reach 0.875 and 98.3 % reach 0.49 (`base_sim.py`), so a p99.5 near 0.5 would charge almost every pair of step-0-like cousins |
| `theme.json` | median and MAD of the CLAP margin "over the pool" (`taste_pipeline.md:245`) | written after e2 from the Claude-written, base-model and draft clips. Mutations, broken copies, held-out clips and the ear-check 50 are left out | it needs no votes. 150 mutations are 15 cousins of each kept loop and would weight the kept 10 fifteen-fold; broken copies are broken on purpose; held-out clips are never "tuned on" (`:129`) |
| `pq.json` (τ_PQ) | the 5th percentile of PQ over Good-or-better clips (`:235`) | gives each clip's PQ (embed) | needs his tiers: the RM unit writes it |
| `ce.json` | CE against his strict pairs (`:240`) | gives each clip's CE | needs his votes: the RM unit writes it |
| P_sax bands | from his Good-or-better loops (`docs/TASTE.md:92`; `docs/VERIFIERS.md:117`) | gives each clip's P_sax parts; the flagged slots supply loops beyond the bands (§2.2) | circular as built: rule 9 holds the informed arm inside, the drafts are inside and the base model goes past only the wide band (§1.3). So the pool's reading is reported, not adopted, until the pool holds out-of-band loops he has tiered (review V3-6) |
| hack-watch bands | from Good-or-better clips (`taste_pipeline.md:306`) | gives each clip's symbolic readouts | need his tiers: their unit |
| `max_completion_length`, P_len | "512, then 1.3× the pool's p99" (`:265`); P_len above the pool's p95 (`:254`) | reports characters, and Qwen tokens if the laptop's cached tokenizer is present | run 1 keeps 512 (`docs/DECISIONS.md:422`), and P_len keeps its 360-token free length (`docs/VERIFIERS.md:128`; `penalties.py:36`). The gate run's p99 is 305 tokens (`junk.json`) |
| P_copy at step 0 | none | reports P_copy of the gate run's 500 completions against the copy pool, computed in memory | shows whether the pool moves step 0's copy penalty before run 1 |

### 5.6 What is never committed
- step-0, held-out, Ref_b or gate-run texts. They stay in `runs/` on the laptop and on KEEP (`step0_samples.md:506-508`);
- votes, tiers, the triage's draw seed, hidden manifests, inbox batches (D-010, `docs/DECISIONS.md:66-69`);
- FLAC or WAV files, render manifests, feature caches, RM weights. MERT-v1-95M is CC-BY-NC (`docs/design/infra_schedule.md:208`), so its features stay on KEEP;
- workflow transcripts, the agents' worktrees and the blind folder's runs;
- his app's usage readings (review V3-16).

## 6. Separation and privacy

### 6.1 Held-out
- The held-out clips (`5101:*`) are never pool, triage or round clips other than their own held-out rounds (`step0_samples.md:489-490`). The pool's briefs are training briefs (`596:*`).
- No Claude-written loop, mutation or broken copy may have a held-out canonical text. The barrier checks this once `heldout_v1.json` is committed (§2.7).
- No calibration reads a held-out clip (§5.5).

### 6.2 Model texts and the copy pool
- **On the cluster only** (review V2-18). Render-pool and embed read the 660 picks', the held-out clips' and the ear-check clips' texts on the cluster from KEEP, by sha256, and their summaries hold hashes and clip ids only. Two things on KEEP do hold those texts:
  - each render's manifest, which carries the canonical text (`pipeline.py:246`);
  - the scorer cache, whose every record stores it (`cache.py:74-76`; `docs/SCORER.md:282-285`).

  So no laptop tool or writer reads the cache, and `make pull-clips` pulls FLACs only, never a manifest.
- **Committed loops are never model output.** Every `.ls` under `seeds/` is Claude-written or made from one. Mutations come only from the kept 10, and broken copies only from accepted Claude-written loops. A test checks that no committed `.ls` has the canonical sha256 of a step-0 row or of any completion in the laptop's `runs/` (§8.1).
- **The copy pool for run 1** is `seeds/claude/v1`, `seeds/mutations/v1`, `seeds/broken/v1`, `drafts/v1` and `drafts/v1/redo1`. That is 1,392 files plus prompt v2's 9 references, 1,401 in all. Run 1's config registers it (`docs/DECISIONS.md:927`), and its builder refuses step-0 texts (D-023's ask, `step0_samples.md:628`).

### 6.3 What the writers may read, and the audit (reviews V1-18, V2-9, V2-24, V3-2)
- **Informed writers** work in a fresh worktree of the wave's commit, which holds no `runs/`, `data/` or `inbox/`. They may read the pack, their run's cards file and their stage folder, and run the one check command (§2.4). They never read:
  - the main checkout or anything outside their worktree. After `make pull` the main checkout's `runs/` holds step-0, held-out and base-model texts (`Makefile:33-37`), and `../../../runs/` from a worktree reaches it;
  - `CHANGELOG.md`, `docs/DECISIONS.md`, `gates/`, or `docs/DRAFTS.md` outside the pack's §2 and §4. These record his picks and tosses, for example `docs/DRAFTS.md:462` and `docs/DECISIONS.md:776-785`;
  - `drafts/`, `shootouts/loops/`, the prompt modules' examples (`src/lofi_rl/prompt*.py`), `tests/fixtures/loops/`, another agent's worktree or another run's folder.
- **Blind writers** read only their folder, and run `./check`.
- **The audit** (`python -m lofi_rl.seeds audit <transcript dir> --arm …`) checks against a list of what is allowed, not a list of what is forbidden. It is deterministic.
  - **Its inputs:** this session's agent transcripts, `~/.claude/projects/-Users-shresthrana-csci596-project/<session id>/subagents/workflows/<wf id>/agent-<id>.jsonl` (present for `wf_6930ad87-bf9`, `ls`, this session). The blind session's sit under its own project key, `~/.claude/projects/-Users-shresthrana-lofi-seed-blind/…`; the pilot confirms the path.
  - **A violation** is any of these:
    - a Read, Grep or Glob path outside the allowed files;
    - a Bash call that is not the check command or a write into the run's own folder;
    - any path that resolves into `/Users/shresthrana/csci596-project` outside the agent's worktree;
    - any recursive search (`grep -r`, `rg`, `find`, `ls -R`, `**` globs) or interpreter one-liner (`python -c`).
  - It records counts and the offending calls in `log.jsonl`. A run with a violation loses all its loops, a fresh agent rewrites them, and the CHANGELOG says so. A blind run's violation also enters D-024's arm counts.
  - It finds a read only after the text has reached the agent. That is why the worktrees leave the texts out, and why Q7 asks for rules that refuse the read.
- **Settings rules (Q7).** Under Q7(a), Claude adds deny rules for `runs/`, `data/` and `inbox/` to the writers' settings. The pilot's probe, an agent that tries to read `runs/io-probe-13787408/summary/io.json` by its absolute path (the I/O probe's timings, with no loop text in it), must be refused before W1 goes on. If the rules can only be set for the whole project, they would also block the main session's pulls and tallies, so Claude asks again rather than set them.

### 6.4 Votes
- Triage tiers go to `data/votes/` through the rating page, as every vote does. The tally and the builder of batches 5–7 read them only on the laptop and publish counts (§7.3, §7.4).

### 6.5 Sound
- Nothing in this unit says how any clip sounds. His ear decides, through the triage and the rounds.

## 7. The triage of 300 and the first best/worst rounds

### 7.1 What the triage holds (`python -m lofi_rl.seeds triage`)
- **Claude-written, 160 (a planning count):** all 5 loops of the first 32 of W1's 36 positions, in plan order, whose 5 loops are all accepted and rendered (80 informed, 80 blind, 3–4 briefs a cell when the first 32 are complete; `revise/plan16.py`). The builder reports the arm split and the cells it took.
- **Base-model, 100** (review V3-9): 3 step-0 picks of each of positions [0, 33) and 1 of position 33. So 96 share a brief with the triage's Claude-written loops, and there are up to 99 within-brief base-model pairs at weight 0.5 (`taste_pipeline.md:157`), against 50 in a layout of 2 a brief. A brief short of picks is filled from positions [34, …).
  - These are never Ref_b and never held-out, by step 0's rules (`step0_samples.md:139-143`), and are step0_v1.json's pool picks, as step 0 asks (`:717`).
- **Mutations, 25:** 5 of each family, at most 3 of one kept loop, drawn.
- **Broken, 15:** §3.3's, each made from one of the 160.
- **Not random** (review V1-4). TASTE.md says "Triage covers a random 300 seeds" (`docs/TASTE.md:57`). This triage is the plan's strata (`taste_pipeline.md:360`), taken from the loops written first, because W2–W4 do not exist when it is built. D-024 lists it under what it supersedes, and Q4 tells him and offers a random second half.
- **Under Q4(b)** each stratum halves: 80 Claude-written (all 5 of the first 16 complete positions), 50 base-model (3 of each of positions [0, 16) and 2 of position 16), 12 mutations and 8 broken copies (2 of each kind; 3, 3 and 2 under Q3(b)), with about 15 repeats. A kind with 2 copies cannot be decided by the triage, and one with 3 only if all 3 are informative and caught (§3.3).
- **Under Q4(c)** the Claude-written 160 are all 5 loops of the first 16 complete positions (80) and 80 drawn at random from every other accepted Claude-written loop once W4's barrier is done. Then 48 of the base-model 100 share a brief with them. The random 80 go into batches 5–7.
- **Under Q5(c)** (3 loops a brief) the Claude-written 160 are 160 of the first 54 complete positions' 162 loops, the other 2 left out by the draw.

### 7.2 Batches, order and hidden controls
- **Batches 1–4,** `b<MMDD>-triage-<k>`, hold 50 first showings each and are built at §8.2 step 6. Each is dealt from every stratum and holds 2–3 broken copies as hidden anchors (kind `anchor`, expected Toss, `taste_pipeline.md:46`).
- **Batches 5–7** (reviews V1-16, V2-22, V3-7) are built only after batches 1–3 are voted. Each holds about 33 first showings and 10 repeats, about 43 rounds.
  - Repeats are 10 % (D-013, `docs/DECISIONS.md:88`), kind `repeat`. Each is drawn from rounds first heard at least 2 days before its batch's `serve_from`, which the builder sets from the vote log's dates. No batch is more than a third repeats.
  - The tally scores a repeat only when its vote falls at least 2 calendar days after its first showing's, and marks it `too_soon` otherwise, as the shootouts did (`docs/SHOOTOUTS.md:180-181`).
  - **Not in D-008's p:** triage repeats measure tier consistency and are reported, as the shootouts' were (`docs/SHOOTOUTS.md:190`).
  - **What could give a repeat away** (reported with the agreement, as `docs/SHOOTOUTS.md:192-195` does): a repeat is a loop he has heard before, in a batch whose other loops are new to him.
  - He is never told which rounds those are (`docs/RATING.md:85-87`).
- **Order:**
  - no two consecutive rounds in one cell (the sign-off's and gate1's rule, `src/lofi_rl/gate1/batch.py:37-40`);
  - a broken copy at least 10 rounds from its source when they share a batch;
  - two mutations of one kept loop at least 10 rounds apart.
- **Each round** plays one loop: the 8 bars, then bars 1–2 again (`configs/rating.yaml` `wrap_bars: 2`). He answers Toss, Fine, Good or Great, and may mark it as soon as he is sure.
- **Time** (estimates, `revise/listen.py`). A round played to the end is 10 bars × 240 / bpm + 8 s (`src/lofi_rl/freeze/batch.py:55-56, 134-136`). Over the strata's own briefs that is 36.84 s a round: about 31 min a batch of 50, 3.07 h for the 300 and 3.38 h with the 30 repeats.
  - The plan says about 1.2 h (`taste_pipeline.md:360`), which is 14.4 s a clip.
  - His one pace on new loops is his Phase-1 ear check, b1106-gate1: 50 triage rounds on 2026-11-06 (`CHANGELOG.md:66`). `lofi_rl.ui.audit` reports a batch's `time.minutes` (`src/lofi_rl/ui/audit.py:264`), a count, from the votes this design does not read. So Claude reads it before Q4 goes to him and puts it in Q4's options (§9).
- **Public manifest:** the scene ("city at night", `lofi_rl.brief.SCENE`), each round's bpm and opaque ids (`docs/RATING.md:151-161`).
- **Hidden manifest:** category `seed_triage`; per clip the real clip id, source, brief id, slot and file sha256, arm, lead order, card, flag, parent, kind, pcm_sha256, FLAC sha256, LUFS, true peak, render job and palette label.

### 7.3 The builder and its refusals
- **It refuses,** writing nothing:
  - before `docs/prereg/heldout_v1.json` is committed with its sha256 in D-023's Outcome. That is step-0 Q7(a): "before the first Phase-2 vote (the triage of 300 or the first best/worst round)" (`step0_samples.md:589`). Another answer to step-0 Q7 moves this refusal with it;
  - without D-024, or without §3.3's rule in it;
  - for a clip whose G7 is not ok in the registry;
  - for a FLAC that is not its record: its size, sha256, STREAMINFO (n frames = `loop_samples(bpm)`), and LUFS and true peak within tolerance, as `gate1 batch` checks (`src/lofi_rl/gate1/batch.py:11-24`);
  - for any held-out clip id or held-out canonical text (step-0 Q5(a)) or any Ref_b draw among the 300;
  - for batches 5–7 before batches 1–3 have their votes.
- **Its FLACs** come from KEEP by `make pull-clips RUN=render-pool-<job> LIST=<file>`: rsync `--files-from` of `clips/<i:04d>.flac` names, mapped from clip ids through records.jsonl, into `runs/<run>/clips/`, never deleting and never a manifest. The list sits under `runs/`, which git never sees. That is about 770 MB for 300 (an estimate at 2,562,058 B a clip). Only listening copies are pulled (`CLAUDE.md:66`).
- **It writes** into `inbox/<date>/.staging-*`, checked by `python -m lofi_rl.ui.manifest check` in a fresh interpreter, then moved into place, and never over an existing batch. This is gate1's procedure.
  - Ids are `r` + 8 hex and `k` + 10 hex, drawn again on a leak (`src/lofi_rl/drafts/batch.py`).
  - The draw's 64-bit seed is kept only in `runs/seeds-triage-<date>/draw.json` (CD-21).

### 7.4 The tally (`python -m lofi_rl.seeds triage tally <batch> --votes <log>`)
- It reads his votes on the laptop, as `gate1 tally` does.
- Its private JSON goes to `data/votes/audit/`, and only counts reach the CHANGELOG: tiers per source and arm, each kind's informative and caught showings and its status (§3.3), the repeats' agreement with `too_soon` left out, and the heard counts. Per-loop tiers stay private.

### 7.5 Left to their own units, and what this unit asks of them
- **The rounds unit** (the bws4 builder and `jobs/lofi-queue.sbatch`, `taste_pipeline.md:105-121`; not built). It must serve by about Nov 11, so that the 40 held-out first showings are heard by about Nov 13 (`step0_samples.md:549`). Asked of it:
  - rounds by brief, from `seeds/registry_v1.jsonl`. It may serve the pool's FLACs on KEEP rather than re-render them (`taste_pipeline.md:105-108`), re-rendering 1 % of the clips it serves and comparing their audio as step 0's §3 says (`taste_pipeline.md:78`; `step0_samples.md:393-396`);
  - anchors only from caught kinds, each broken copy in a round of its source's brief with its source; an undecided kind's copies as kind `none` beside their source (§3.3);
  - at least a third of the Phase-2 training rounds before Nov 15 made of a pool brief's 3 base-model clips and 1 Claude-written clip (review V3-9). Every held-out tuple is 4 base-model clips, so D-008 scores only base-against-base pairs. A random quad of a pool brief's 8 clips holds all 3 base clips with probability 5/70, about 7 % (combinatorics, this session);
  - draft-brief rounds (a kept loop, its rejects and its mutations) allowed;
  - a readout beside D-008 of agreement on base-only pairs against mixed pairs, at every nightly re-check. Base-model and Claude loops differ by source in ways the RM can see (§1.3).
- **The RM loader** (`lofi_rl.taste`; not built). Asked of it:
  - the plan's label table (`taste_pipeline.md:165-173`), with the synthetic broken-copy labels only for caught kinds, and §3.3's sitting weights;
  - keep3's labels through `seeds/drafts_map_v1.json`, only for drafts whose no-bed twin matched (§5.4);
  - for the 4 loops kept in keep3 and tossed at the sign-off: the sign-off tier, heard with the rain, replaces keep3's "at least Fine", while keep3's "kept beats its played rejects" pairs stay (Claude's default);
  - one hardware key a fit (§5.2);
  - D-023's held-out rules;
  - `pq.json` and `ce.json` (§5.5).
- **Run 1's config:**
  - the copy pool of §6.2. P_copy's exact Jaccard against 1,401 references costs about 0.69 s of laptop time a step of 32 completions, at 15.49 µs a pair (an estimate, `spec_cost.py`; `time_parse.py`). GRPO.md suggests an LSH index if a large pool shows in the step time (`docs/GRPO.md:85-88`);
  - D-023's refusals;
  - `max_completion_length` 512.
- **TASTE.md:82** still says Claude registers types up to 500 SU. His words of 2026-11-06, "increase your limit for job to 1000 SU now" (`docs/DECISIONS.md:26`), bring it up to date at this unit's commit, with the date. TASTE.md changes only with his words (`docs/TASTE.md:9`).

## 8. Tests and the order of work

### 8.1 Tests (laptop; synthetic where a cluster file is needed)

| file | checks |
|---|---|
| `test_seeds_plan.py` | 1,100 slots; 16 shards of 68–69 slots over 68–69 briefs; arms 550/550 with 3+2 or 2+3 a brief; lead orders, all 3 a brief under Q5(a) or (b), 192 of 220 under Q5(c); wave counts 180/305/305/310; level counts and the vamp rule (330 ranked, none repeated in a brief, blind never carded ranked); the 126 flags; no within-brief pair failing the plan axes; the swap search's bound, never moving a flag; binding under step-0 Q1 (a) and (b) keeps every cell (191 brief ids the same); `pool_briefs`' 220 under both rules pinned; `--check` byte-identical; card_changes.jsonl applied by `seeds cards` |
| `test_seeds_check.py` | hand-built loops for every common and informed rule: a free repair refused, a fold under one tonic of 12, a band, a mood, a card axis, a flag under its bound brief, J5 0.6 against a reference, a draft and a pool loop, a shared A line, an equal digest; a blind loop on plagal passing rule 6 (review V1-2); half-time passing rule 15; the blind arm passing what only the informed arm fails; rule versions; `--slot`; no file named in a copy failure; the table and exit codes |
| `test_seeds_blind.py` | `seeds check --arm blind`, table and `--json`, on hand-built loops: its keys are exactly the allowed list, with none of `THEORY_KEYS` (`src/lofi_rl/verify/scoring.py:27-28`), p_sax, busy, wide, chromatic or rubs, no repo path, and no 5-word sequence of docs/TASTE.md, DRAFTS.md §2 or docs/VERIFIERS.md. The exported DSL.md, the folder's CLAUDE.md and BRIEF.md, the cards and the handoff file share no 5-word sequence with docs/TASTE.md or DRAFTS.md §2 and hold no "TASTE" or "shootout" token; no blind card names a ranked vamp |
| `test_seeds_classify.py` | each classifier on hand-built grids and lines, each overlap under its precedence (drag with many ghosts, half-time with ghosts, a push that also syncopates), hold, rest and stab at f of 0.75 and 0.25 and a tie, "none" with no long note, and colour and blues riff in each mood (♭3 and ♭7 chord tones in aeolian and dorian, blue in ionian) |
| `test_seeds_barrier.py` | collisions: a staged loop against an accepted one with a higher slot index goes (review V2-4), and between staged loops the higher index goes; the critic's rules; the held-out and model-output backstops on synthetic `runs/` and `heldout_v1.json`; coverage targets; a backfill list that never names an accepted slot; a run of another model refused; nothing written on a failure |
| `test_seeds_audit.py` | the record shape (tool, path or command) of the pilot's real transcripts, with no message text, and synthetic cases: a Read of `runs/…`, a Bash `cat` of `data/…`, a blind Read of TASTE.md, a `grep -r` from the repo root, an absolute read of the main checkout's `runs/`, a `../../../runs/` read, a `python -c` open, a `**` glob, each caught; clean runs pass |
| `test_seeds_mutate.py` | on the kept 10: 15 each, each mutation's canonical diff only in its family's lines, digests distinct, every admission rule; Q2(c)'s substitute; `--check` byte-identical |
| `test_seeds_broken.py` | each kind's signature and band on hand-built sources; P_sax kept for no groove, boring and fake; sources only accepted Claude-written loops, drawn after their wave's backfill round; Q3(b)'s split (34/33/33, and 5/5/5 in the triage); deterministic |
| `test_seeds_v1.py` | the committed pool, as tests/test_drafts_v1.py holds the drafts: every loop passes its arm's checks under its own rule version, with results cached by (file sha256, version) in `.pytest_cache`; the counts; the coverage file is the code's; no committed `.ls` equals a step-0 or `runs/` completion's canonical sha256 (skipped without `runs/`) |
| `test_seeds_render.py` | the driver with `render_clip` stubbed: 4 processes of 8 lanes over quarters, records appended and fsynced, `<i:04d>` names mapped to clip ids, the deadline, outcomes and the render-code key, step-0 rows read by sha256 with no text written, the 28 twins through `twin_doc`; with FluidSynth and LOFI_ASSETS (skipped otherwise), two real renders whose clip ids equal `scoring.protocol.clip_id` |
| `test_render_pool_job.py`, `test_embed_job.py` | as `tests/test_junk_job.py`: the Cost line against `#SBATCH` and the registry (render-pool 800 and embed 1,000, by Claude); `--no-requeue`; each step under a timeout, the servers under `timeout -k 30`, deadlines under timeouts, the sum at least 60 s under the walltime; the run on the node's disk; OUT and caches on KEEP, `LOFI_CACHE` exported before a server starts; e2's two servers in order; a used OUT refused; traps; the argument refused unless it is a committed list |
| `test_seeds_embed.py` | the driver against the scorer's fake backend (`--backend fake`): chunking, record fields, one feat_ver and hardware key, the retry of transient items and the e1b list, the audio rule against render records, the held-out and ear-check comparison against heldout_v1.json and the gate run's clips.json (violation and drift), a held-out G7 failure dropping its tuple, and the overlap through a fresh cache |
| `test_seeds_registry.py` | the registry and drafts map from synthetic summaries; render-check and embed-check catching each tamper; step-0's recomputed picks against step0_v1.json; a twin pool clip under two briefs refused; a reject's twin that differs losing its label |
| `test_seeds_triage.py` | the strata, the base-model layout included; the order rules; batches 1–4 first showings only; in batches 5–7, no batch more than a third repeats and each repeat at least 2 days after its first showing's vote; the tally's `too_soon`; §3.3's statuses on synthetic tiers; the leak rule; every refusal of §7.3; `ui.manifest check` passing; the tally's counts on a synthetic log |
| `test_seeds_docs.py` | this file and D-024 state the registered values: counts, axes, rules, jobs and SU |

- No existing test changes: the informed stage lives in the agents' worktrees, which `tests/test_privacy.py:43-48` already covers. `tests/test_approvals.py` keeps checking every registered sha.

### 8.2 The order of work (review V2-3)
1. Shresth's answers to §9, in the order of §9's opening, and to step-0's §7, or his OK on the defaults.
2. **Build A,** from `wt/step0`'s tip, since the planner needs only `lofi_rl.step0.select.pool_briefs`: plan, cards and the pack, check, classify, barrier, audit, export-blind, handoff, the two workflow scripts and their tests, with a test pinning `pool_briefs`' 220 under both Q1 rules. D-024 drafted with every rule. An independent review with its findings applied; `pytest -q`; commit. It is rebased after step 0's merge.
3. Shresth opens the blind session (Q6(a)). The pilot (12 W1 slots), then the rest of W1 (168), its barrier and its backfill round; the 150 mutations and the 15 triage copies. Each barrier commits after `pytest -q`.
4. **Build B,** during W1: mutate, broken, the render driver and `jobs/render_pool.sbatch`, render-check, the registry, `pull-clips` and the triage builder. Register render-pool right after its script's last change (`make approve JOB=jobs/render_pool.sbatch MAX=800 BY=Claude`, then `scripts/approve_job.py --check`) and before the commit that adds test_render_pool_job.py, which checks the registry (as `tests/test_junk_job.py:34-46`; `step0_samples.md:541-542`). CHANGELOG lines; review; `pytest -q`; commit.
5. Step 0, its own unit: review, D-023, registration and merge. Then on USC's VPN: `make sync`; the step-0 job; `make pull`; `step0 check`; `prereg --write`; commit (D-023's Outcome).
6. `myquota`; r01; `make pull`; `render-check`; the registry; `make pull-clips`; triage batches 1–4; push him the inbox note.
7. W2–W4 with their barriers and backfill rounds. **Build C,** after his first triage sitting: embed and `jobs/embed.sbatch` (registered before the commit that adds its test), embed-check, theme.json, the readouts, the tally and the drafts map.
8. W4's backfill round; the 85 broken copies; r02; e1 (after r01 and Build C) and e2; `embed-check`; e3 if the keys differ; theme.json; the readouts; the registry complete; triage batches 5–7 once batches 1–3 are voted. Commit.
9. The CHANGELOG at each step, with SU from `session_report.py`.

### 8.3 The critical path to his first votes

| date | step | waits on |
|---|---|---|
| Sat Nov 7 | his answers: this spec's Q1, Q2, Q5, Q6 and Q7 and step-0's Q1–Q8 first; Q3 and Q4 by Nov 8 | Shresth |
| Sat Nov 7 – Sun Nov 8 | Build A, reviewed and committed; he opens the blind session (Q6(a)) | his pilot answers |
| Sun Nov 8 | the pilot, then W1 (0.8–1.2 h or 2.7–4.1 h of writing, §2.1), its barrier and backfill round; the mutations and the 15 triage copies; Build B during W1, render-pool registered | Q1, Q2, Q5, Q6, Q7 |
| Sun Nov 8 | step 0 merged; sync; `lofi-step0` (about 301 SU; A40 waits have run 72 min to 16.5 h, `CHANGELOG.md:141, 615, 782`, though the gate's second launch started about 6 min after its sync, `runs/junk-13788483/summary/generate.json` host stamp 22:27:18Z and `CHANGELOG.md:85`); pull, check, both prereg files committed | step-0's answers; USC's VPN (`CHANGELOG.md:59`) |
| Sun Nov 8 – Mon Nov 9 | r01 (1,075 renders, about 7.5 min and 299 SU on main); render-check; registry; 300 FLACs pulled; triage batches 1–4 | the prereg commit; Q3, Q4 |
| **Mon Nov 9** | **his first triage sitting: the first Phase-2 votes** (Nov 10 if the step-0 job queues long) | the batches |
| Nov 9–11 | W2–W4 (920; 4.0–6.2 h or 13.9–20.4 h of writing) with barriers and backfill rounds; Build C; the 85 copies; r02 (about 286 SU); e1 (about 503 SU; L40S waits 2.9–7 h, `CHANGELOG.md:597, 662`) and e2 (about 474 SU) | the writing's pace |
| Nov 10–11 | triage batches 5–7; `theme.json`; the registry complete; the rounds unit serves its first best/worst rounds | batches 1–3 voted; the rounds unit |
| by Nov 13 | the 40 held-out first showings heard | the rounds unit |
| Nov 15 | the RM gate, re-checked nightly to Nov 17 (D-008, `docs/DECISIONS.md:52-58`) | — |

- **The binding risks:** USC's VPN and the step-0 job's GPU queue; the writing's pace and his weekly allowance (Q5); the rounds unit, which is not built; and his listening hours (Q4).

## 9. Open questions for Shresth
Nothing is written before his answers or his OK on the defaults. Each option says what he would hear or do and what it costs. An "I don't know" takes (a); where the choice is about sound, his triage and votes then settle it.
- **Send order, by what each blocks** (review V3-13): Q1, Q2, Q5, Q6 and Q7 block the pilot; Q3 and Q4 block the triage batches (§8.2 step 6). Step-0's Q1 also binds this plan (§1.2).
- **Before Q4 is sent,** Claude reads his minutes for b1106-gate1's 50 rounds (`lofi_rl.ui.audit`, a count; his tiers stay private) and puts the measured figure in Q4's options in place of the two estimates. If it leaves fewer than 200 best-and-worst rounds by Nov 15 under the full triage, (a) and (b) change places, so the recommendation becomes the half triage (review V3-8).

- **Q1. Half of Claude's writers won't see your taste profile; on Oct 30 you said they "see only the DSL spec and the brief". Should they also get a short card of plain musical choices to try, such as "laid-back drums with soft extra hits; piano; the sax builds on one short idea", with nothing from your profile?**
  - (a) **Recommended.** Yes. Both halves then try the same range, so your votes compare like with like. Their loops must match the card, and their checker tells them only whether a loop plays, fits the energy and mood, matches the card, and copies nothing.
  - (b) No card: only the loop language and the brief, word for word, with the same checker minus the card. Their loops vary less, and Claude's informed half covers the rest.
  - (c) Everything but the profile itself, Claude's rules drawn from it included.
- **Q2. One of your two drum kits, "studio", has never stayed in a loop you kept: you tossed both kept loops that used it at the sign-off, and you picked the other kit in all 4 redrafts (`docs/DECISIONS.md:939-940`). How much more of it should you hear in Claude's new loops?**
  - (a) **Recommended.** About 1 in 3 of Claude's 1,100 (about 55 of the 160 in your first triage), so your votes can confirm it either way.
  - (b) About 1 in 10 (about 18 in the triage).
  - (c) None in the new loops or the cousins of your kept 10. The 19 studio drafts stay in the pool and can come up in rounds of their own briefs, and your keep3 and sign-off votes on them still count.
- **Q3. Some hidden checks are deliberately broken copies of a loop, one for each of your four dealbreakers: sour, no groove, boring, and fake or cheap. Claude can change only the notes and rhythms, not how the instruments sound, since the sound set is frozen. When a loop strikes you as fake or cheap, what is it closest to?**
  - (a) **Recommended.** It plays too evenly, like a machine: every drum hit at one level with the soft in-between hits removed, and each held sax note cut short so it never swells or wavers. Claude builds that, and your sittings show whether you catch it.
  - (b) The instruments themselves sound synthetic. That cannot be built now, so the 25 fake copies become extra copies of the other three kinds.
  - (c) I don't know. Same as (a).
  - Under any answer, a kind you don't catch stops being a hidden check, and a sitting where you missed only that kind keeps its full weight and its held-out rounds; the registered rule would halve it and drop them.
- **Q4. Before the best-and-worst rounds, you rate 300 single loops (Toss / Fine / Good / Great), plus about 30 repeats mixed into later sittings, as usual.** They are not a random 300: Claude's 160 are every loop of the first 32 training briefs, written first; the base model's 100 are 3 from each of the first 33 briefs and 1 more, so 96 share a brief with Claude's; then 25 cousins of your kept 10 and 15 broken copies. Played to the end that is about 3.4 hours, though you can mark a loop as soon as you are sure. An hour a day from Nov 9 to Nov 15 is 7 hours (estimates).
  - (a) **Recommended.** All of it, in sittings of 50 (about 31 minutes each at full length), from about Nov 9, alternating with best-and-worst sittings from about Nov 11. Played to the end, that leaves about 160 best-and-worst rounds by Nov 15 (the plan counted on about 250), 40 of them the held-out sets the check needs; marking early, as the plan assumed, about 255.
  - (b) Half now: 150 loops and about 15 repeats, the rest only if the reward model needs them. That leaves about 240 best-and-worst rounds played to the end, or 285 marking early. The junk floor and the sax limits rest on fewer of your loops, and the broken copies are judged in the rounds instead.
  - (c) As (a), but half of Claude's 160 drawn at random from all of its loops once they are written (about Nov 10), so the junk floor and the sax limits rest on a random sample. The same time as (a); your last three triage sittings wait for the writing.
- **Q5. Writing Claude's 1,100 loops is the largest piece of agent work in the project so far.** The first 30 drafts took about 33,000 tokens a loop and the 12 redrafts about 102,000, so the 1,100 would take about 37 to 113 million tokens before checks and reviews. For scale, all of Claude's workflows from Oct 30 to Nov 4 used about 79 million, and the agents stopped at your weekly limit on Nov 4. The same week has to carry the rounds and the reward model too (estimates; §2.9).
  - (a) **Recommended.** The 180 loops of the first wave first (your first triage needs 160 of them), measured exactly. Claude then stops and asks you before the other 920 if the measured rate says they would take more than half of what is left of the week, after keeping enough for the rounds and the reward model.
  - (b) All 1,100 straight through. If the week runs out, all Claude work stops until it resets, the rounds and the reward model included.
  - (c) Fewer: 3 a brief, 660 in all, about 40 % less writing. Each training brief then has 6 loops for your rounds instead of 8, and the pool holds about 1,612 loops against the plan's about 2,000.
- **Q6. The writers who must not see your profile need their own Claude session in a separate folder, which only you can start.**
  - (a) **Recommended.** You start it once, by Sun Nov 8: a new Claude Code session in `/Users/shresthrana/lofi-seed-blind/`, with this session's model and mode, and one line, "Read BRIEF.md and follow it." It then writes each wave by itself when Claude leaves the next wave's files there.
  - (b) You can't: they write in this project's session, where Claude's project notes summarize your taste. D-024 records it, and the audit shows whether any of them opened your profile, but their loops are no longer blind.
- **Q7. Claude's loop writers must never open your votes, the inbox or the base model's loops, the held-out ones above all. May Claude add settings rules that refuse those folders to the writers?**
  - (a) **Recommended.** Yes, for the writers only, checked in the pilot by a test agent that must be refused, and removed after the last wave. The writers also work in copies of the project that leave those folders out, and an audit reads every step they took.
  - (b) No settings change: the copies and the audit only. A read would then be caught only afterwards, and that run's loops thrown away.

## 10. D-024: what it records, supersedes and asks
D-024 is written at §8.2 step 2, with Shresth's answers in his words, and committed with Build A before the pilot. Each job's registration (render-pool at step 4, embed at step 7) is an amendment, made before that job's first run, as D-003 and D-022 carry theirs (`docs/DECISIONS.md:25-26, 957`).
- **Decided by:** Shresth (his answers to §9, or his OK on the defaults), with Claude's defaults (§11).
- **It records:**
  - the pool: its sources and counts (§1.1), and its briefs, which are D-023's 220 and the draft briefs;
  - the plan rule: positions, slots, 16 shards, arms, lead orders, waves, the cards, the flags and their rule, the swap search, the binding, and the seed 596 (§2.1–2.2);
  - the arms as Q1 answers (§2.3), and the blind arm's information set: the cut DSL.md, the vocabulary and mood chords, the brief's scene, energy and mood, its cards and lead order (Q1(a)), and `./check`'s restricted output, with nothing of TASTE.md, DRAFTS.md or the verifiers;
  - the workflows, at most 10 agents a batch, the writers' model (`claude-opus-5-5`), the worktrees and the blind folder, the handoff files, the audit and the settings rules (§2.4, §6.3; Q6, Q7);
  - the checker's rules by arm and their versions, the critic, dedupe, the backstops, coverage, card_changes.jsonl and the backfill rounds (§2.5–2.8);
  - the mutation families and the broken kinds, their sources, signatures, and §3.3's calibration rule, fixed before his first triage vote;
  - the jobs `lofi-render-pool` (at most 800 SU) and `lofi-embed` (at most 1,000 SU): their shapes, registry shas, Claude's registration, their validity rules, and the lists r01/r02 and e1/e2 with their contingencies (§4–5);
  - the triage: its strata, batches, hidden controls, refusals and tally, and that its repeats are not in D-008's p (§7);
  - the calibration rules of §5.5;
  - how this unit meets step 0's asks of the pool's render (`step0_samples.md:627, 659, 717`; §5.1);
  - after the runs, its Outcome: the counts, the SU, and the registry's sha256.
- **What this entry supersedes** (each with the question or Claude's default that settles it):
  1. `taste_pipeline.md:329-334`: the planner's briefs, now D-023's 220 (step-0 Q1). Also the card axes: "late snare" becomes "drag"; "12 vamps" becomes ranked vamps in the informed arm and any vamp of the mood in the blind arm (Q1); keys, kit, what the keys do under long notes, and long-note colour are added, with the sax-line flags (Claude's default).
  2. `taste_pipeline.md:336-342`: shards of 10 briefs × 5 cards become 16 shards of 68–69 slots; "personas vary by shard index" becomes lead orders by shard mod 3 and arms by parity; the critic becomes §2.6's rules (Claude's default).
  3. `taste_pipeline.md:338-341`: `lofi_rl.dsl.check --repair` with "deterministic repairs only" (instrument aliases, which `aliases: {}` makes impossible, `palettes/lofi_rain_v1.yaml:413`; and "chord symbols at edit distance 1", which no parser repair does, `docs/DSL.md:225-259`). This becomes `lofi_rl.seeds check` with zero repairs, DRAFTS.md's rule (`docs/DRAFTS.md:55-57`). The 20 % rule is read for rejections.
  4. `taste_pipeline.md:344`: render-pool at about 400 SU and embed at about 250 become §4's shapes and costs, and embed renders through the scorer, so it reads no stored audio. "Backfills until the counts are met" becomes backfill rounds whose shortfall is reported, never forced (§2.8).
  5. `CHANGELOG.md:172`, "`lofi-render-pool` (1,440 SU in the plan) … stay his", and `step0_samples.md:385`, "render-pool is Shresth's": Claude registers render-pool at 800 SU and embed at 1,000 (D-003 as amended).
  6. `taste_pipeline.md:352` and `docs/DECISIONS.md:449`: "Rejects / kept 20 / 10" and "the 30 drafts" become the 42 drafts, which D-022 left open (`docs/DECISIONS.md:927`).
  7. `taste_pipeline.md:360` and `:364-369`: the triage's 1.2 h becomes about 3.1 h at full length plus D-013's 30 repeats, and the Phase-2 row's "bws4 5.5 h ≈ 250 rounds" becomes what Q4's answer leaves (§9). `docs/TASTE.md:57`'s "a random 300" becomes the strata of §7.1, taken from the loops written first (Q4).
  8. `taste_pipeline.md:252`, `docs/VERIFIERS.md:117, 167-172, 181-183` and `docs/design/instrument.md:294`: NEAR, the energy bands and the P_sax bands from the pool are reported, not adopted (§5.5).
  9. `taste_pipeline.md:265` and `:254`: "1.3× the pool's p99" and P_len at the pool's p95 become readouts; run 1 keeps 512, and P_len its 360-token free length (`docs/VERIFIERS.md:128`).
  10. `taste_pipeline.md:5, 77, 96, 213`: `$REMOTE_DATA` and scratch renders and caches become KEEP and the node's disk (D-022's amendment, `docs/DECISIONS.md:957-961`).
  11. `docs/DECISIONS.md:118` (D-018): read as his Q1 answer.
  12. `docs/DECISIONS.md:88` (D-013) and `taste_pipeline.md:46, 173`: a sitting whose only missed anchors are of a kind not caught, or undecided, keeps full weight and its held-out rounds; synthetic broken-copy labels go only to caught kinds; anchors come only from caught kinds, and while none is caught the 5 % waits (§3.3; Q3).
  13. `taste_pipeline.md:245`: the theme's median and MAD over the Claude-written, base-model and draft clips, without mutations and broken copies (§5.5; Claude's default).
  14. `taste_pipeline.md:105-108`: the rounds unit may serve the pool's FLACs on KEEP instead of re-rendering them, with 1 % re-rendered and compared (§7.5; Claude's default).
  15. `CLAUDE.md:151`, "Seed pool of ≈2,000 loops": under Q5(c), about 1,612.
- **Readings, which supersede nothing:**
  - the plan's "≥2 axes" is pairwise.
- **Asked of later units:** §7.5.

## 11. Claude's defaults here (each overridable; no answer needed unless he objects)
- **The plan:**
  - positions cell by cell; 16 shards by (5b + j) mod 16; arms by parity; lead orders by shard mod 3; waves W1–W4, W1 with 4 spare positions for the triage;
  - runs of 11–12 and 19–20 loops; at most 10 agents a batch, one batch of 8 an arm a wave, and the two sessions never at once;
  - the cards' nine axes and their levels, the rule, the swap search and the seed 596; the sax-line flags on every 9th slot of each arm in each mood;
  - "drag" in place of "late snare", half-time kept, its card overriding DRAFTS.md's backbeat aim;
  - ranked vamps for the informed arm only, 3 in 5 of its slots; any vamp of the mood for the blind arm; 5 vamp lines a brief;
  - sparse bass inside the brief's band.
- **The arms:**
  - the blind session outside the repo, opened by him with BRIEF.md, handoffs by READY and DONE files, and the fallback; its cards in plain words, with conditional long-note axes;
  - the informed arm in fresh worktrees with a reading pack, held to the house rules as checks (rules 8–17), the blind arm to the common rules only, and the blind check's restricted output;
  - every writer on `claude-opus-5-5`; the allow-list audit, and a run that broke a read rule rewritten.
- **The checker:**
  - zero repairs; no fold under any of 12 tonics; J5 below 0.6 against everything outside a loop's own family, same brief included; no shared A or B line; unique digests; no file named in a copy failure;
  - the classifiers, their precedence and thresholds, and blue as a blue note outside the mode; rewrites at most twice; the 20 % rule read for rejections; card misses as failures for a first writer only;
  - rule versions; the coverage targets; card changes for later waves first; a backfill round after each wave; an accepted loop never rewritten for coverage.
- **Mutations and copies:**
  - mutations: the five families of three, one change each within its family's lines, rubs 0, briefs as §3.1 gives them;
  - broken copies: the four transforms and signatures, the fake one keeping every onset, sources only accepted Claude-written loops drawn after their wave's backfill, one a brief;
  - §3.3's calibration: three caught showings and two-thirds, decided over the triage and the rounds; undecided kinds served as ordinary clips.
- **The jobs:**
  - render-pool at 20 min and 800 SU, in 4 processes of 8 lanes; embed's shape, deadlines and timeouts; lists made on the laptop, with no cross-run reads on the cluster;
  - embed in two runs on an L40S, e2's overlap through a fresh cache, e3 if the keys differ; FLACs and features on KEEP, never stems; the 28 rejects' twins in r01.
- **Records:**
  - every committed file of §5.3; hashes only for model texts; tokens and minutes from the workflow files, never the app's usage readings;
  - the registry committed, and the drafts map;
  - the calibrations reported rather than adopted, except `theme.json`.
- **The triage:**
  - the first 32 complete W1 positions' Claude loops; 3 base-model picks of each of the first 33 briefs and 1 more; 5 mutations a family;
  - batches 1–4 of first showings, then 5–7 with the repeats mixed in; the order rules; FLACs pulled by `make pull-clips`;
  - the refusal before `heldout_v1.json` is committed; Q4's recommendation set by his measured pace.
- **Asked of later units:** §7.5's defaults, among them the sign-off tier replacing keep3's label for the 4 tossed loops, the 3-plus-1 quads in a third of the rounds, and serving from KEEP with 1 % re-rendered.
