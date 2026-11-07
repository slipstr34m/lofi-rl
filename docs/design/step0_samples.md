# The step-0 samples (D-021 "What follows" item 2): build spec

**Status:** a build spec, written and revised on 2026-11-06 at repo HEAD `cfc24df` after an independent review whose findings are applied below, and built on 2026-11-07 on branch `wt/step0`, with the recommended (a) of each §7 question as a named constant in `src/lofi_rl/step0/__init__.py`. §10 lists every accepted deviation of the code from sections 1–9; a sentence the build's fixes made wrong is corrected in place and named there. It waits on Shresth's answers to §7 (or his OK on its defaults). Nothing in it has run on the cluster.
- Its rules go into a new entry, **D-023**, in `docs/DECISIONS.md` (§8). That covers the rules D-021 leaves open and also the registered text they depart from, listed under "What this entry supersedes", as D-021 listed its own (`docs/DECISIONS.md:309-312`). Entries are never edited; a later one supersedes (`:3`).
- D-023 carries Shresth's answers to §7 and is committed with the code, before the job, as D-021 was (`:306`). From then on D-023 is the record, and amendments go there, never here.
- This file is committed first on its own, for his answers. It is updated with them and committed again beside the code.

**How it was drafted** (workflow `wf_72e78bac-bc8`, 2026-11-06, read-only, 9 agents):
- Four scouts read the repo: the pre-registration, the downstream uses, the reusable code, and cost and infrastructure.
- One designer wrote the spec. Three adversarial reviewers, one per lens (fidelity to the registered text, engineering, downstream use), raised 30 findings: 0 blockers, 10 major, 20 minor.
- The designer applied all 30. The major ones: the PQ floor and the tempo strata, which the first draft had read as unregistered (§1.3, Q1, Q3); a held-out-only draw that would have left the 660 undrawn (Q6); held-out texts reaching the pool under other clip ids (Q5); recoveries that could never read valid (§4.2); and a supersession list for D-023 (§8).
- Re-checked by hand after the revision (laptop, 2026-11-06): the same-text pairs per cell over `runs/junk-13788483/summary/scored.jsonl` (75 of 13,836), both Q1 walks (35/169/16 within [0, 380); 30/176/14 within [0, 260); 206 shared), the 40 held-out indices and their cell counts, the 6,800 distinct draw seeds without PROBE_SEED, the gate run's timings (`gen_s` 50.1, `load_s` 29.4, `render_s` 199.01 over 50) and the cited registered text.

**The unit.** D-021 "What follows", item 2: "The step-0 samples, all drawn under the grammar, with item 1's streams and draw seeds: the RM held-out tuples, PAR's Ref_b and the seed pool's 660" (`docs/DECISIONS.md:537`).
- It comes after the Phase-1 gate, which Shresth approved on 2026-11-06 (`CHANGELOG.md:26-33`).
- It comes before Phase 2's GRPO smoke (`docs/DECISIONS.md:538`).
- Its cluster job waits for the USC VPN (`CHANGELOG.md:37`).

- **Code:** a new package, `src/lofi_rl/step0/`, run as `python -m lofi_rl.step0 <generate|score|render|check|prereg|path>`. Tests are `tests/test_step0_*.py`, on synthetic runs (§6.1).
- **Job:** `jobs/step0.sbatch` (`lofi-step0`): one A40, 45 min, at most 810 SU, registered by Claude (§2.1). It takes no argument.
- **Unchanged, byte for byte:**
  - `src/lofi_rl/rl/junk.py`;
  - `jobs/junk_rate.sbatch`, whose registry sha is `0eb12128013576e3` (`jobs/APPROVED.yaml:11`; recomputed with `scripts/approve_job.py`'s `job_sha`, this session);
  - `tests/test_junk.py` and `tests/test_junk_job.py`;
  - `docs/prereg/brief_streams_v1.json` and `palettes/lofi_rain_v1.yaml`.
- **Writes:**
  - on the cluster, `$REMOTE_KEEP/runs/step0-<job>/` (§2.7), and a recovery's own `step0-redo-<job>/` (§4.2);
  - on the laptop, committed: `docs/prereg/heldout_v1.json` and `docs/prereg/step0_v1.json` (§4.4), `docs/prereg/D-023_step0_path.txt` (§4.1), D-023 and this file.
- **Never:** it reads nothing under `data/`, opens no `manifest.hidden.json`, writes nothing to `inbox/` and asks for no ear check.
- **Sound claims:** none. The tools print counts, hashes and loudness measurements only, as junk does (`junk.py:82`).

**Re-read from the files this session** (laptop, read-only):
- the gate run `runs/junk-13788483/summary/`: `generate.json`, `junk.json`, `clips.json`, `scored.jsonl`, `completions.jsonl`, and their file times (`stat`);
- `runs/junk-13783877/summary/` (`completions.jsonl`, `generate.json`), every junk run's `generate.json` `host.gpu`, and `runs/io-probe-13787408/summary/io.json`;
- `python -m lofi_rl.brief_streams --check` (ok, a fresh build), and `shasum -a 256` of the streams file and of the palette;
- `stratified_indices(596, 220)` and a tempo-band walk over [0, 400) (§1.6);
- per-cell same-text pair counts over the gate run's `scored.jsonl`, and a simulation of the selection rules (§1.4);
- `git log` over the audio path, and `lofi_rl.scoring.features.source_sha8()` at HEAD (`41700aa5`);
- `grep` for `requeue`, `lofi-gen` and the held-out seeds.

Every laptop count below names its source, and every estimate is labelled with its arithmetic.

## 1. What is drawn

### 1.1 The registered draw rule (all three sets)
- **One request per draw.** "Draw k of a brief (k = 0..7) is one vLLM request with n = 1 and SamplingParams(seed=draw_seeds[k])" (`brief_streams_v1.json:39`). With n > 1, vLLM gives child i the seed + i, which is not the table's (`:43`; `src/lofi_rl/brief_streams.py:26-27`).
- **Selection.** "A use that needs m samples passing the gates takes the first m passing draws in k order; a brief still short after its draws is reported short, never drawn again under other seeds" (`brief_streams_v1.json:40`; `DECISIONS.md:739-740`). D-023 states where this unit reads that rule further (§8).
- **Draws per brief.** Train and every evaluation stream have 8 draws a brief; the measurement stream has none (`brief_streams_v1.json:48`; `DECISIONS.md:739-741`). A brief's seeds are the same under every checkpoint (`brief_streams_v1.json:42`).
- **Seeds come only from `lofi_rl.brief_streams.load()`.** It refuses a file that is not exactly a fresh build (`brief_streams.py:382-392`). The file's sha256 is `bb2daee53e22974c…` (this session).
- **Brief ids** are `<seed>:<index>` in every record (`brief_streams.py:119-121`; `src/lofi_rl/rl/train.py:159-160`).
- **Step 0** (`DECISIONS.md:583, 425, 811`):
  - Qwen3-4B-Instruct-2507 @`cdbee75f`, with no adapter;
  - prompt v2 (template `127e3094…`);
  - the frozen palette's grammar (regex `5c081828…`);
  - T 1, top_p 1, top_k 0, 512 tokens.

  The sampling is junk's `SAMPLING` (`junk.py:133-134`), which the gate run recorded (`generate.json` `sampling`).

### 1.2 The three sets

| set | stream (seed) | briefs | draws a brief | m | rule | settled by | traced to |
|---|---|---|---|---|---|---|---|
| RM held-out | `rm_heldout` (5101) | the 40 registered: indices 0–11, 13–15, 17–26, 28–31, 36–40, 47, 49, 50, 52, 54, 58; 5/5/5/5/4/4/4/4/4 by cell | 8 | 4 | the first 4 passing draws in k order whose canonical texts differ | Q3, Q4 | `brief_streams_v1.json:987-1037`; `taste_pipeline.md:126-127`; `DECISIONS.md:436, 441` |
| Ref_b | `train` (596) | run 1's [0, 400) | 8 | 4 | the first 4 draws in k order that pass G1–G6 and compile, repeats kept; a G7 failure later takes the first spare | Q2, Q3, Q4 | `brief_streams_v1.json:61-70`; `DECISIONS.md:433, 603`; `taste_pipeline.md:177`; `SCORER.md:229, 308` |
| seed pool's 660 | `train` (596) | 220 of [0, 400), by Q1's rule | the same 8 as Ref_b | 3 | the first 3 passing draws after the brief's 4 Ref_b draws, with texts unlike Ref_b's, each other's and (Q5(a)) every held-out clip's | Q1, Q2, Q5 | `taste_pipeline.md:329, 348-349`; `DECISIONS.md:537` |

The requests are rm_heldout 40 × 8 = 320 and train 400 × 8 = 3,200: 3,520 in all under Q6(a). Under Q6(b) the train stream is drawn only for the pool's 220 briefs: 320 + 1,760 = 2,080. Every draw of every drawn brief is made (§1.7).

### 1.3 Passing (Q3)
- **The registered gate.** "G = hard × 1[PQ ≥ τ_PQ] × clip(P_acc / 0.5, 0.25, 1)", with "hard = parse ∧ compile ∧ render ∧ the hard theory checks" (`taste_pipeline.md:233-234`). The P_acc factor is at least 0.25, so only `hard` and the PQ floor can make G zero. "Pass the gate" (`:127`, `:349`) and "passing the gates" (`brief_streams_v1.json:40`) name nothing else.
- **The reading (Q3(a)).** A draw passes when it passes `hard`:
  - G1–G6 and compile, as junk's score counts them (`junk.py:51-54, 726-737`);
  - G7, the render in spec (`src/lofi_rl/render/pipeline.py:56-58, 169-203`).

  That is the scorer's `hard` (`SCORER.md:62-63, 111`) and TASTE.md's hard gates, "parser, compiler, renderer" (`docs/TASTE.md:94`). The hard theory checks sit in the symbolic gate (`SCORER.md:62-63`).
- **The PQ floor is not applied** to any of the three sets, and D-023 records it (§8).
  - τ_PQ is "the 5th percentile of PQ over Good-or-better clips" (`taste_pipeline.md:235`). It is set from the triage of 300 (`:360`), a Phase-2 vote whose 100 base-model clips come from these 660, after the held-out set must be committed (Q7).
  - PQ itself needs Audiobox features, which this unit does not compute (§3).
  - The scorer reads a missing τ_PQ as "PQ factor 1" (`SCORER.md:207`).
- **Why G7 counts.** Held-out clips are played in bws4 rounds (`taste_pipeline.md:30`). Ref_b's clips need features, and a clip outside spec has G = 0 (`SCORER.md:112`).
- **G7 is applied lazily, in k order.** A draw is rendered only while the draws before it in its set have not filled m.
  - The held-out set is rendered in this unit (§2.5).
  - Ref_b and the pool get their G7 at their own renders (§3), by the rules in §1.5–1.6.
  - G7 is a function of the text, the brief, the palette and the render code. Cluster renders are canonical (`DECISIONS.md:35-39`) and the same across CPU models (`CHANGELOG.md:11`). But the render code can change under one clip id (`SCORER.md:299-300`): 16 commits touched render/, compile/, dsl/, theory/ or palette.py since 2026-10-29, the last `e271619` on 2026-11-03 (`git log`, this session).
  - So the rule does not fix every outcome now. Each G7 outcome is recorded with the code that rendered it, and D-023 says which render decides each set (§3).
  - In the gate run, 0 of 50 renders failed G7 (`junk.json` `render`).
- **Near-copies pass.** J5 ≥ 0.8 with the shown example is the Phase-1 bar's subtraction (`DECISIONS.md:141-142`), not a gate.
  - D-021 item 8 expects tuples that carry the example's lines (`DECISIONS.md:483-486`).
  - "Held-out" means fresh samples of the 9 prompts (`DECISIONS.md:443`), and Ref_b stands for "a typical step-0 loop" (`taste_pipeline.md:178`).
  - The gate run had 23 near-copies of 500, 13 of them among high/wistful's 57 (`junk.json` `near_copy`, `by_cell`).
  - They are reported per set (§4.4).
- **Repaired loops pass,** as junk counts them; the reward charges them P_fix (`docs/VERIFIERS.md:35`).
  - In the gate run, 443 of 500 passing loops had no repair (`junk.json` `repairs.per_loop.passing`).
  - Part B's "zero repairs" was a rule for SFT targets only (`DECISIONS.md:551`).

### 1.4 The RM held-out tuples (rm_heldout, seed 5101)
- **m = 4.** "Each brief contributes its first 4 step-0 samples that pass the gate" (`taste_pipeline.md:127`). D-021 replaced only that file's seed and its `data/` split (`DECISIONS.md:312, 436`).
- **The rule (Q4(a)).** Walk the brief's draws k = 0..7. Take a draw that passes (§1.3) and whose canonical text differs from every draw already taken for the brief. The canonical text is `canon_sha256` (`junk.py:733`). Stop at 4.
- **Why the texts must differ:**
  - two draws of one brief with one canonical text are one clip (`taste_pipeline.md:76`; `SCORER.md:128-131`);
  - a round may hold a real clip only once (`RATING.md:174-175`; `src/lofi_rl/ui/manifest.py:596-597`);
  - so a tuple with a repeat could not be the bws4 round it is for.
- **Same-text rates.** In the gate run, 75 of 13,836 same-cell pairs of passing loops share a canonical text: 0.00542 (laptop count over `scored.jsonl`, this session). By cell:
  - 0 in three cells: low/hopeful 0/1,326, mid/wistful 0/1,540, high/melancholy 0/1,225;
  - low/melancholy 1/990, low/wistful 2/2,415, mid/hopeful 2/1,378;
  - mid/melancholy 29/1,596 (0.01817), high/hopeful 23/1,770 (0.01299), high/wistful 18/1,596 (0.01128).
- **How often a later draw is needed (estimate).** Assuming independent draws, a brief's first 4 hold a repeat with probability 1 − (1 − r)⁶. Summed over the 40 briefs' cells, that is about 1.2 briefs. The simulation below gives 1.1.
- **The simulation (laptop, this session; an estimate).** It ran 4,000 runs of §1.4–1.6's rules, with every draw passing (the gate run: 500 of 500 pass, 0 of 50 G7).
  - Each cell's texts are modelled on its passing loops in the gate run's `scored.jsonl`. A text seen k ≥ 2 times among the cell's n gets probability √(k(k−1)/(n(n−1))), which keeps the cell's same-text pair rate. Every other draw is a new text.
  - Per cell, briefs as the sets hold them: held-out 5/5/5/5/4/4/4/4/4, train as `brief_streams_v1.json:69`, pool 25/25/25/25/24/24/24/24/24.
- **Twins stay** (`brief_streams_v1.json:16`). There are 8 (`:1490-1502`), carried into heldout_v1.json for the RM's report (`DECISIONS.md:448`).
- **This unit renders and fixes the set:** 160 clips if no brief is short (§2.5, §4.4).

### 1.5 PAR's Ref_b (train, seed 596, [0, 400))
- **m = 4.** "Ref_b is 4 step-0 samples of brief b" (`taste_pipeline.md:177`). There are exactly 4 per brief and member (`SCORER.md:229, 308`; `src/lofi_rl/scoring/rm_api.py:43, 166-170`), so 1,600 in all (`DECISIONS.md:603`).
- **Passing applies (Claude's default).** The text does not say so. But the samples are "rendered and featurized" (`DECISIONS.md:603`), and utilities come from a clip's cached features (`SCORER.md:224-226`). So Ref_b is "a use that needs m samples passing the gates".
- **The rule.** Take the brief's first 4 draws that pass G1–G6 and compile, in k order, **with repeats kept** (Q4(a)). Each one is an honest sample of step 0, and PAR averages their utilities (`SCORER.md:231-232`).
  - About 13 of the 400 briefs hold a repeat in their first 4 (estimate, computed as in §1.4 with train's cell counts).
- **G7 comes at Ref_b's render** (§3), under that render's recorded code. A reference that fails G7 is replaced by the brief's first **spare** that passes it, in k order (Q2(a)).
  - The spares are the brief's other passing draws, less the pool's draws, any draw with a pool draw's text, and any draw that is another pool brief's pick (a twin's clip, §1.6).
  - This departs from the registered words, which take the next passing draw even when it is a pool draw. Example: Ref_b k0–3, pool k4–6, k2 fails G7. The rule here gives Ref_b {k0, k1, k3, k7}; the words would give {k0, k1, k3, k4}, a pool loop. It is in D-023's list (§8).
- **This unit fixes the selection by its records (§4.4).** Rendering and features come later (§3).

### 1.6 The seed pool's 660 (Q1, Q2, Q5)
- **What is registered.**
  - The count: "Base-model step-0 samples that pass the gate (220×3, from `lofi-gen` at T 1.0)" (`taste_pipeline.md:349`). `lofi-gen` appears nowhere else in docs, src, jobs, configs or scripts (grep, this session). `lofi-step0` stands in for it, recorded in D-023.
  - "Item 1's streams and draw seeds" (`DECISIONS.md:537`). Item 1 registers no seed-pool stream (`brief_streams.py:86-93`).
  - The briefs: "`python -m lofi_rl.seeds.plan --seed 596` produces 220 briefs stratified by energy × mood × tempo" (`taste_pipeline.md:329`). The planner is not in the repo (there is no `src/lofi_rl/seeds/`), and no tempo bands or counts are written anywhere.
- **The briefs (Q1).** Both options read seed 596 within [0, 400), so every brief has registered seeds, and both give the cells 25 or 24 (the first 220 mod 9 = 4 cells take 25; `brief_streams.py:140-161`; `brief_streams_v1.json:20-27`).
  - **(a) Energy × mood × tempo band (recommended; the bands and counts are Claude's default).**
    - The bands are 60–74, 75–90 and 91–95 BPM, around TASTE.md's centre (`docs/TASTE.md:24`; `src/lofi_rl/brief.py:16-18`).
    - Each cell's bands take the sampler's own shares, 15/84, 64/84 and 5/84 (`brief.py:57-63`), by largest remainder: 4, 19, 2 for a cell of 25, and 4, 18, 2 for 24.
    - Walk [0, 400) and take each brief whose cell and band have room. A band that [0, 400) cannot fill gives its room to its cell's 75–90 band: low/melancholy and high/wistful hold 1 fast brief there, high/melancholy 3 slow.
    - Result (this session): 220 briefs within [0, 380), 35 slow, 169 in 75–90, 16 fast.
  - **(b) Energy × mood only**, the rule of every other balanced set: `stratified_indices(596, 220)` reads [0, 260), with 30 slow, 176 in 75–90 and 14 fast (this session).
  - 206 of the 220 are the same under (a) and (b). Both hold 7 of the 9 twin counterparts (13, 79, 81, 95, 137, 177, 225) and all of 596:[0, 40) (this session).
  - The 1,100 Claude-written loops share these 220 briefs (`taste_pipeline.md:348`), because Phase-2 rounds come "by brief" (`:110`). D-023 records that these 220 stand in for the planner's, so a later `lofi_rl.seeds.plan` takes them.
  - Under Q6(a), every train brief is drawn, so this choice changes no draw.
- **The draws (Q2(a), Q5).** Walk the brief's draws after its 4 Ref_b draws, k ascending. Take a draw that:
  - passes G1–G6 and compile;
  - has a canonical text unlike the brief's Ref_b texts and its earlier picks;
  - under Q5(a), has a canonical text that no held-out clip has. Under Q5(b) or (c), it must only not be a held-out clip (the same clip id, which only a twin can give);
  - is not another train brief's Ref_b clip (Q2(a)) and not an earlier pool brief's pick: two train briefs of one cell, bpm and tonic give one clip for one text (below).

  Stop at 3. G7 comes at the pool's render, under that render's recorded code. A pool draw that fails G7 is dropped and never replaced, so every leftover passing draw stays a Ref_b spare: Ref_b must hold exactly 4 (`rm_api.py:166-170`), and 660 is a planning count.
- **Why after Ref_b (Q2).**
  - Read literally, the pool's "first 3 passing in k order" are the first 3 of Ref_b's 4: the same seeds, prompt and model.
  - Then 220 of run 1's 400 briefs would be centred on clips Shresth votes on, which train the RM (`taste_pipeline.md:110-113, 360`), and the other 180 would not.
  - PAR's bound, the std over members (`SCORER.md:231-232`), would then differ between the two groups.
  - It costs 660 later renders and nothing here: the train draws serve either reading.
  - **It separates clips, not texts.** Exact texts recur across a cell's briefs. In the simulation (§1.4), about 176 of the 1,600 Ref_b draws share their exact text with another brief's pool draw (about 496 such pairs). Under (a), a 220 brief and an 180 brief are about equally likely to have one, so the two groups are treated alike.
  - **Twin train briefs are where a text is one clip.** The clip id has no brief id, and run 1's [0, 400) holds 26 pairs of train briefs of one cell, bpm and tonic, 7 of them both pool briefs (2026-11-07). So every Ref_b by its records comes first; each pool brief, in stream order, then skips another brief's Ref_b clip and an earlier pool brief's pick; and the spares leave out every pool pick. Walking each brief alone would put a yardstick clip in the pool in about half of runs (an estimate: 201 of 400 simulated runs under Q5(a), the build's review).
- **Held-out texts (Q5).** The clip id hashes the canonical text with bpm, tonic, mode and energy (`SCORER.md:123`; `src/lofi_rl/scoring/protocol.py:68-71`). So one text in two briefs of a cell is two clips. But the RM's symbolic backbone reads the canonical LoopScript (`SCORER.md:226`; `CLAUDE.md:142`).
  - In the simulation, with the pool skipping only held-out clips: about 30 of the 660 pool draws share the text of one of about 11 of the 160 held-out clips.
  - In the gate run, 8 of the 50 ear-check clips Shresth voted carry a text that repeats in their cell (`clips.json` over `scored.jsonl`, this session). So labels from other sources can match a held-out text too, and Q5(a)'s RM rule covers every source (§8).
  - Every train draw records `heldout_text` (true when its canon_sha256 equals a held-out clip's), counted per cell (§4.4).
- **Shortness (estimates).**
  - Closed form: for a cell with repeat rate r, let q = 1 − 5r be the chance that a later draw is new (about five texts are taken before it). Then P(short) = 1 − q⁴ − 4q³(1 − q).
  - That gives 25 × 0.044 + 24 × 0.023 + 24 × 0.018 ≈ 2.1 briefs in mid/melancholy, high/hopeful and high/wistful, each typically one clip short. Low/melancholy, low/wistful and mid/hopeful add 25 × 1.5e-4 + 25 × 1.0e-4 + 24 × 3.2e-4 ≈ 0.014.
  - The simulation gives 2.4 short briefs, and 11.9 under Q5(a).

### 1.7 Every draw is made (Claude's default)
"The first m passing draws in k order" allows stopping early (`brief_streams_v1.json:40`). The job draws all 8 of every drawn brief anyway, because that gives:
- one generate pass;
- a selection that is a pure function of complete records;
- spares for G7;
- the pool's draws from the same pass.

Stopping at m would need at least 160 + 1,600 + 660 = 2,420 requests. The other 1,100 cost about 1,100 × (0.1002 + 0.056) s ≈ 172 s, or 52 SU (estimate, at §2.9's rates).

### 1.8 Short briefs
- **The rule:** a short brief is reported, never drawn again under other seeds (`brief_streams_v1.json:40`).
- **The record:** `short: true` in sets.json, with each draw and why it was not taken: a gate code, `repeat_of:<k>`, `refb`, `heldout_text` or `heldout_clip`. Short briefs are counted in step0.json and in both prereg files.
- **Held-out:** the set holds fewer tuples (Q8(a)). A tuple whose clip later fails G7 under newer render code is dropped the same way and reported (§3).
- **Ref_b** (Q8(a)):
  - a run-1 brief without 4 references is refused while taste is live (HTTP 422, `SCORER.md:236-238`);
  - a refs.json row must hold exactly 4 (`rm_api.py:166-170`);
  - so a short brief leaves run 1's [0, 400) (`DECISIONS.md:433`), is reported, and goes to Shresth with the lofi-grpo ask.
- **Pool:** short by the missing clips. 660 is a planning count.
- **Likelihood (estimates).**
  - Passing only (Ref_b, repeats kept): at the gate run's lower pass bound, p = 0.9924 (`junk.json` `pass.ci95`). Then P(fewer than 4 passing of 8) ≈ 56 p³ (1 − p)⁵ = 56 × 0.9774 × 2.54e-11 ≈ 1.4e-9 per brief.
  - Held-out, under the distinct-text rule: repeats dominate. A mid/melancholy brief is short with probability about 6.6e-5 (1,000,000 simulated briefs under §1.4's model). Resampling 8 of that cell's 57 passing loops gives 1.5e-5 without replacement and 1.5e-4 with it.
  - Over all 40 held-out briefs, that is about 4e-4 (1 in 2,500) under the model, and about 1.3e-3 (1 in 800) with replacement.
  - Pool: §1.6.

## 2. The job

### 2.1 Shape and cost
- **`#SBATCH` lines:**
  - `--job-name=lofi-step0`, `--partition=gpu`, 1 node, 1 task;
  - `--cpus-per-task=4`, `--mem=24G`, `--gres=gpu:a40:1`, `--time=00:45:00`;
  - `--output=/scratch1/%u/lofi-rl/logs/%x_%j.out`;
  - `--no-requeue`.

  That is junk's shape (`jobs/junk_rate.sbatch:2-10`) with three changes:
  - the partition is written in (the gate ran with `OVR='-p gpu'`, `DECISIONS.md:920`);
  - the walltime is §2.8's;
  - `--no-requeue`, so a node failure never reruns the job under its own id into its own OUT (§2.7). Nothing in `jobs/` or `scripts/` sets requeue today (grep, this session).
- **The A40.** All six junk runs generated on A40s (each run's `generate.json` `host.gpu`, this session). Other GPU types are unmeasured for generation.
- **Cost:** `Cost: 4 CPU + 6 (24 GB) + 8 (1 GPU) = 18 SU/min x 45 min -> at most 810 SU.`, at CLAUDE.md's rates (`CLAUDE.md:79`). 810 ≤ 1,000, so Claude registers it: `make approve JOB=jobs/step0.sbatch MAX=810 BY=Claude` (D-003 as amended 2026-11-06, `DECISIONS.md:26`; `config/budget.yaml:17`), with a CHANGELOG line.
- **No argument.**
  - The streams the job draws are a registered constant, `lofi_rl.step0.STREAMS`, set from Shresth's answer to Q6. It is written into D-023's mark, which generate checks before any work (§2.3).
  - Model, prompt, palette, grammar and sampling are registered constants too.
  - Any argument stops the job before any work, with exit 2, as junk's argument checks do (`junk_rate.sbatch:74-82`).
- **Submit:** `make submit JOB=jobs/step0.sbatch`.
- **Hygiene:** the script meets `tests/test_job_scripts.py:24-45`: the name, the log path, the Cost line, a `make submit` line, `MODE=policy source scripts/job_env.sh`, and no git or credentials.

### 2.2 Code: a new package, with junk.py untouched
- **Why new code, not flags on junk.**
  - Junk is the Phase 1–2 progress oracle and D-020's registered measurement (`CLAUDE.md:116`; `DECISIONS.md:208-213`).
  - The measurement stream has no draws precisely because junk seeds its engine (`brief_streams.py:31-32`).
  - A flag would ride in every later junk-rate snapshot.
  - Other modules already reuse junk by import (`gate1/__init__.py:46`, `rl/copy_penalty.py:38`, `rl/trl_probe.py:102`, `rl/train.py:88`, `drafts/check.py:52`).
- **From junk, public names only:**
  - `SAMPLING`, `STRUCTURED_BACKEND`, `PROMPTS`, `split_model`;
  - `load_engine` with its `VllmEngine` (`sampling`, `chat`, `info`, `load_s`);
  - `structured_outputs_params`, `constrained_block`, `backend_record`, `prompt_record`, `example_id`;
  - `brief_dict`, `sha256_text`, `read_jsonl`, `Completion`;
  - `CopyCheck`, `score_row`, `summarize`;
  - `render_clip`, `render_lanes`, `in_order`;
  - `grams`, `jaccard`, `HANG_MARGIN_S`, `DEFAULT_CHUNK`.

  These are at `junk.py:126-150, 188-198, 216-339, 342-499, 695-977, 1038-1077`. The private helpers (`_write_json`, `_host`, `_now`) are written again in step0, a few lines each.
- **Copied, not refactored.** Generate's chunk, deadline and fsync loop (`junk.py:547-573`) is written again with per-request seeds. A refactor would change the measurement's code.
- **Imports.**
  - The policy-side modules import with tests/test_rl_imports.py's blocked list (`tests/test_rl_imports.py:13-14`).
  - vLLM is imported inside functions.
  - `step0/render.py` imports its audio code inside functions, as `junk.py:1040-1042` does, because the policy env's build imports every `lofi_rl` module outside its scorer-side list (`jobs/env_policy.sbatch:32-37`).

| module | env | does |
|---|---|---|
| `step0/__init__.py` | any | the registered values: model revision; template, palette, regex and streams-file shas; `STREAMS`; m per set; rule `step0-select/1`; PROBE_SEEDS 1 and 2; ENGINE_SEED 0; schemas; `MARK`, the D-023 mark |
| `step0/requests.py` | any | the requests from `brief_streams.load()` for `STREAMS`: (i, stream, seed, index, brief_id, k, draw_seed, brief), in stream order (rm_heldout, then train), then brief, then k |
| `step0/generate.py` | policy | the pre-checks, the engine, the probe, the draws, the replay |
| `step0/score.py` | policy | `junk.score_row` per row, readouts per stream, candidates, Ref_b's selection |
| `step0/select.py` | any (stdlib) | the selection rule, pure: held-out, Ref_b, the pool's 220 (Q1) and its draws (Q2, Q5) |
| `step0/render.py` | scorer | the held-out renders and their render-code key; the final held-out set and the pool |
| `step0/path.py` | any (stdlib) | the path file: the blob shas of its files (`sha1("blob <n>\0" + bytes)`, no git needed); on the laptop, git reads a commit's blobs |
| `step0/check.py`, `prereg.py` | laptop | validity, a redo pair included; the committed files |
| `step0/cli.py`, `__main__.py` | any | `python -m lofi_rl.step0 …` |

### 2.3 generate (policy env)
`python -m lofi_rl.step0 generate --run DIR [--deadline-s 960] [--chunk 128] [--hang-s S]`

1. **Pre-checks, before the model loads.** Each one stops with exit 2, `stopped` recorded and nothing drawn:
   - `brief_streams.load()` accepts the file, and its sha256 is `bb2daee5…`;
   - the palette's sha256 is `09f80a64…`, and its regex's is `5c081828…` (`tests/test_palette_frozen.py:19, 22`);
   - v2's template sha is `127e3094…` (`tests/test_prompt_v2.py:38`);
   - every message builds (a PromptError stops it, as at `junk.py:514-524`), once per cell, since a cell's briefs share their messages (`DECISIONS.md:362-363`);
   - the code tree's `docs/DECISIONS.md` holds D-023's mark, and the mark equals `step0.MARK`: rule `step0-select/1` and the streams of `STREAMS`. So no job draws under rules, or streams, not registered first. The precedent is M5, checked in the job's own snapshot (`DECISIONS.md:713-714`);
   - every file that `docs/prereg/D-023_step0_path.txt` lists hashes to its blob in this tree, as M5 checked D-021's list (`DECISIONS.md:713-714`);
   - PROBE_SEEDS, 1 and 2, are not among the file's 6,800 draw seeds (neither is, and the 6,800 are distinct; 2026-11-07).
2. **The engine:** `junk.load_engine(model, seed=0, gpu_mem=0.85, max_model_len=4096, backend="xgrammar")`.
   - These are the gate run's engine settings, bf16, eager, xgrammar pinned and read back (`junk.py:402-436, 492-499`; `generate.json` `engine`), with seed 0 in place of its 2026 (`generate.json` `engine.seed`).
   - Seed 0 is the smoke's (`src/lofi_rl/smoke.py:296`). With every request seeded, it should not reach sampling. That is UNVERIFIED for vLLM 0.32.0: no vLLM source is on the laptop. It is recorded.
3. **One SamplingParams per request:** `engine.sampling(**junk.SAMPLING, structured_outputs=S, seed=draw_seed)`.
   - n is 1 by construction (`junk.py:437-445`).
   - S is one `StructuredOutputsParams(regex=…)` (`junk.py:460-468`), built once and shared, as junk shared one SamplingParams across 500 requests (`junk.py:544, 557`). Recompiling the grammar per request would show in chunk times.
   - Each row records `draw_seed`, and `sp_seed`, read back from the object it was sent with.
4. **The probe,** before any draw (Claude's default):
   - One call of 6 requests: 3 identical requests at each of seeds 1 and 2, the low/hopeful cell's messages, the registered sampling.
   - It passes when, at each seed, at least 2 of its 3 texts are identical, and the two seeds' texts differ.
   - With seeds applied, each seed's 3 should match. A numeric difference between batch rows could split one off (UNVERIFIED); 2 of 3 still pass.
   - **An engine that does not use each request's own seed misses.** One randomness for the whole call, or every request sampled with the call's first SamplingParams, gives one text at both seeds; randomness per request that ignores the seed gives 3 different texts at a seed.
   - **A pass by chance is at most about 0.005 % (estimate).** Low/hopeful's 1,326 same-cell pairs held no identical canonical text in the gate run (laptop count, 2026-11-06). By the rule of three, the per-pair rate is then under about 3/1,326 at 95 %. Three texts make 3 pairs, about 0.68 % at a seed, and both seeds about 0.0046 %.
   - **A miss by chance with seeds applied:** the two seeds give one text, at most about 3/1,326 ≈ 0.23 % by the same rule. The seeds are fixed, so it would repeat at every launch, and PROBE_SEEDS would change, recorded in D-023 (estimate).
   - A miss stops generate (`stopped: "seed probe"`, exit 2) before any draw.
   - It is also the first call of `LLM.chat` with a list of SamplingParams, which is UNVERIFIED for vLLM 0.32.0 on the laptop. A vLLM that refuses the list stops here.
   - **Its decoding (estimate):** 233 tokens (low/hopeful's longest gate-run completion, this session) to 512 (the cap) at the single-stream 42.9 tok/s (`junk_rate.sbatch:34`): about 5.4 to 11.9 s, the 6 requests decoding together.
   - Only the texts' sha256 are kept.
5. **The draws.**
   - The requests are sent in order, in chunks of 128 (junk's measured `DEFAULT_CHUNK`), one `engine.chat(convs, [sp, …])` call each.
   - Each chunk's rows are appended, flushed and fsynced, and generate.json is rewritten (`junk.py:561-572`). The output count is checked (`:559-560`). No chunk starts past the deadline, by junk's forecast (`:547-554`).
   - Outputs are taken in input order, as junk takes them (`:557-566`).
   - **Each cell's prompt length is fixed:** the gate run had 9 values, 1,325–1,411 tokens, one per cell (laptop count over its completions.jsonl, this session). So a row with another cell's `prompt_tokens` shows a misaligned output (§4.1).
6. **The replay,** a readout (Claude's default):
   - After the draws, the 320 rm_heldout requests are sent again, in reverse order, in chunks of 160, with their own seeds.
   - Each replay row records whether its text equals the draw's. It goes to `replay.jsonl` only, and is never scored, rendered or selected. It is the same seeds, not a draw under other seeds. Its rows are held-out data (§5).
   - It obeys the deadline. A cut replay leaves the draws complete.
   - **Why:** the gate's two launches, both seeded at the engine with 2026, matched on 273 of their 384 shared rows (119, 76 and 78 by chunk; laptop count over `runs/junk-13783877` and `runs/junk-13788483`, this session). No record says whether per-request seeds replay under another batch shape.
   - The sets are fixed by their records either way (§4.4). The readout says whether a re-draw could stand in for a record.
7. **The watchdog:** faulthandler at deadline + 90 s, then `alarm(120)` for vLLM's teardown (`junk.py:150, 1167-1174`).

### 2.4 score (policy env)
`python -m lofi_rl.step0 score --run DIR` (or `--from OUT --run REDO` in a recovery, §4.2)
- **Every row** goes through `junk.score_row` with `CopyCheck("v2", palette)` (`junk.py:695-758`), and gains `stream`, `brief_id`, `k` and `text_sha256`.
- **Per stream,** `junk.summarize` (`junk.py:906-977`) gives the readout blocks: pass, failures, repairs, tokens, near_copy, copy, by_cell, prompt, constrained with in_language, and palette.
  - They are written under step0.json's own schema, never as `junk.json`, which gate1's `load_run` reads by name and schema (`gate1/__init__.py:212-220`).
- **Candidates:** each brief's draws that pass G1–G6 and compile, in k order. This step also writes Ref_b's selection (§1.5) and the held-out render order.
- **Time (estimate):** the gate run's 500 rows scored in 28 s (generate.json 14:35:33, scored.jsonl 14:36:01, `stat`, this session). `cp -a` and `rsync -az` keep the file times (`junk_rate.sbatch:91`, `Makefile:6-8`). That is 0.056 s a row, about 197 s for 3,520.

### 2.5 render (scorer env): the held-out set only
`python -m lofi_rl.step0 render --run DIR [--workers 4] [--deadline-s 900] [--cache DIR]` (or `--from OUT --run REDO`, §4.2)
- **A partial run is refused** (exit 2): its sets could not count.
- **The render-code key.** The step records its own stamp and `lofi_rl.scoring.features.source_sha8()` (`features.py:46-60`: the render and feature code, `41700aa5` at HEAD, this session) as `render_code` in clips.json, sets.json and step0.json (§3).
- **Rounds.**
  - Round 1 renders each held-out brief's first 4 distinct candidates: 160 if no brief is short.
  - Each later round renders, for each brief still short of 4, its next distinct candidates.
  - Briefs go in stream order and draws in k order, through `junk.render_clip`, `render_lanes` and `in_order` (`junk.py:1038-1077`): two at a time, 2 workers each, on the 4 CPUs. The lanes never change a clip (`DECISIONS.md:916`).
- **Outcomes.**
  - A G7 failure: the draw is not taken.
  - An infrastructure error (never a gate, `junk.py:1051-1052`): retried once in the next round. A retry that renders counts in `errors_retried`. A second error leaves the brief undetermined and counts in `errors_final`.
  - No render starts after the deadline (`junk.py:1101-1103`). A skipped render leaves its brief undetermined.
  - Any undetermined brief makes the step exit 1 (§4.2).
- **Files:**
  - `clips/<i:04d>.flac`, one pass in 16 bits, with its `.manifest.json` (`junk.py:1053-1054`);
  - the clip id, from the manifest (`src/lofi_rl/shootout/render.py:93-99`; `SCORER.md:122-127`);
  - `render_cfg_sha8` in clips.json, so the laptop can check every clip id from recorded inputs.
- **At the end:** the final held-out set; then the pool's 220 (Q1) and their picks, with Q5's held-out-text (or held-out-clip) exclusion against the final held-out set. The step writes sets.json, clips.json and step0.json's render block.
- **Time (estimates).**
  - The gate run's 50 renders at 2 × 2 took 104 s of step wall (14:36:01 → 14:37:45; `render_s` summing 199.01 s, `clips.json`). That is 2.08 s a clip, about 333 s for 160.
  - At the gate run's slowest render on a lane, 5.79 s (`clips.json`, this session): 160 × 5.79 / 2 ≈ 463 s.
  - At job B's mean one-at-a-time render, 7.707 s, divided by the 2 × 2 laptop speedup of 1.713 (`DECISIONS.md:866, 869`): 160 × 4.50 ≈ 720 s.
  - At job B's slowest, 17.39 s: 160 × 10.15 ≈ 1,624 s, which the deadline would cut (§4.2). The gate run came in at about a fifth of that projection (99.5 s against 507.6 s for 50, `DECISIONS.md:888`).

### 2.6 Files and schemas (`RUN/summary/`, all pulled)

| file | schema | content |
|---|---|---|
| `generate.json` | `lofi-rl/step0-generate/1` | partial, requested, generated, stopped; streams {name: seed, briefs, draws}; streams_file {path, sha256, fresh}; registration {found, mark, decisions_sha256}; path_file {sha256, blobs_ok}; model; prompt (`junk.prompt_record`); palette; sampling (`junk.SAMPLING`), per_request_seed true; constrained (`junk.constrained_block`, with the backend); engine (`VllmEngine.info`) and the first draw's SamplingParams repr; probe {seeds, cell, n, texts_sha256, distinct, passed}; replay {n, same, differ, by_chunk, stopped}; timing; host {node, job, stamp, gpu}; trace on an error |
| `completions.jsonl` | one row a draw | i, stream, seed, index, brief_id, k, draw_seed, sp_seed, brief, prompt_sha256, prompt_version, example, text, finish_reason, n_tokens, prompt_tokens |
| `replay.jsonl` | one row a replay | i (the draw's), text, finish_reason, n_tokens, same |
| `scored.jsonl` | one row a draw | `junk.score_row`'s fields (`junk.py:722-750`), plus stream, brief_id, k, text_sha256 |
| `step0.json` | `lofi-rl/step0/1` | run, n, requested, partial, unreadable_lines; by_stream (summarize's blocks); candidates; render {attempted, rendered, G7, other_gates, errors_final, errors_retried, skipped, rounds}; render_code {stamp, source_sha8}; per-set readouts (near-copies, repairs, verbatim A/B, heldout_text per cell); score_stamp, render_stamp |
| `sets.json` | `lofi-rl/step0-sets/1` | rule `step0-select/1`, the mark, render_code; heldout, refb, pool: per brief {brief_id, stream, cell, bpm, tonic, twin_of, taken [{k, i, draw_seed, text_sha256, canon_sha256, clip_id, audio_sha256}], not_taken [{k, why}], spares (Ref_b), short}; per train draw `heldout_text` |
| `clips.json` | `lofi-rl/step0-clips/1` | render_code; per render: `junk.render_clip`'s record (`junk.py:1056-1060`), plus brief_id, k, round, final, clip_id, render_cfg_sha8 |

- `RUN/clips/` holds the held-out FLACs and their manifests. They are not pulled.
- **Sizes (estimate):** the gate run's completions.jsonl and scored.jsonl are 456,936 and 373,358 B for 500 rows (`wc -c`), so about 3.2 and 2.6 MB at 3,520 rows. 160 FLACs at the gate run's 2,562,058 B mean (`clips.json`) come to about 410 MB.

### 2.7 Staging and copy-out (D-022)
- **Where the steps write:**
  - `RUN=${TMPDIR:-/tmp}/step0-<job>`, on the node's disk;
  - `OUT=$REMOTE_KEEP/runs/step0-<job>`;
  - `export LOFI_CACHE=$REMOTE_KEEP/cache/lofi-rl`, since job_env.sh still defaults it to scratch (`scripts/job_env.sh:22`).

  That is junk's pattern (`junk_rate.sbatch:84-96`; `DECISIONS.md:961`). Scratch writes took 2.209–132.108 s a round, and a 2.6 MB clip up to 171.923 s. KEEP took 0.006–0.008 s (`runs/io-probe-13787408/summary/io.json`; `DECISIONS.md:960`).
- **A used OUT is refused.** Before its traps are set, the script stops with exit 2 if `$OUT` exists and is not empty. Junk makes OUT with `mkdir -p` and copies into it with `cp -a` (`junk_rate.sbatch:87-92`), which would merge two attempts under one job id.
- **`copy_out`, once** (and, after a copy that fails, the records once more):
  - `timeout -k 2 30 cp -a "$RUN/summary" "$OUT/"`, then `timeout -k 2 60 cp -a "$RUN/clips" "$OUT/"`;
  - after the render, and from an EXIT trap with `trap 'exit 143' TERM`;
  - a copy that fails is followed by one copy of the records only, from the EXIT trap, under `timeout -k 2 20`, and no more. §2.8 counts it.

  Junk's 10 s copies were sized for 50 clips. 410 MB at KEEP's 0.008 s per 2.6 MB clip is about 1.3 s of writes (estimate). The node disk's reads are unmeasured, hence 60 s.
- **What stays on scratch:** only the Slurm log, plus job_env.sh's mkdirs (`scripts/job_env.sh:34`). Weights are read from `HF_HOME` on scratch (`kit.env:31`), as every junk run read them (the gate run loaded in 29.4 s). Scratch reads were never probed.

### 2.8 Deadlines, timeouts and the walltime

| step | command | kill grace | budget |
|---|---|---|---|
| generate | `timeout -k 15 1080 … generate --deadline-s 960` (stack dump at 960 + 90 = 1,050) | 15 | 1,095 s |
| score | `timeout -k 10 420 … score` | 10 | 430 s |
| render | `timeout -k 10 960 … render --deadline-s 900` | 10 | 970 s |
| copy-out | `timeout -k 2 30`, `timeout -k 2 60`; after a copy that failed, `timeout -k 2 20` (the records, from the EXIT trap) | 2 + 2 + 2 | 116 s |
| sum | | | 2,611 s: 89 s under 2,700 s (45 min) |

- **Generate's margin (estimate).** Its draws may take about 2.3 times the gate run's time per request before the deadline refuses a chunk: (960 − 72.9 − 12) / (3,840 × 0.1002) = 875.1 / 384.8 = 2.27. Under Q6(b), 3.64. Throughput with per-request seeds is unmeasured.
- **Render's deadline** sits 60 s under its timeout, as junk's does (`DECISIONS.md:879`).
- **The minute spare.** The sum leaves at least 60 s under the walltime, junk's rule (`tests/test_junk_job.py:91`).

### 2.9 SU (estimates, on an A40, at the gate run's rates)

| part | seconds | arithmetic |
|---|---|---|
| start to the first chunk | 72.9 | 123 s from job start 14:33:30 (`CHANGELOG.md:63`) to generate.json at 14:35:33, less 50.1 s of chunks (`generate.json` `timing`) |
| probe | ≤ 12 | 6 requests in one call (3 at each of 2 seeds), up to 512 tokens at 42.9 tok/s (§2.3) |
| draws | 352.7 | 3,520 × 0.1002 s (50.1 s / 500) |
| replay | 32.1 | 320 × 0.1002 s |
| score | 197.1 | 3,520 × 0.056 s |
| render | 332.8 | 160 × 2.08 s |
| copy-out | ≈ 5 | an assumption |
| **total, Q6(a)** | **≈ 1,005 s ≈ 16.7 min** | **≈ 301 SU at 18 SU/min** |

- **Under Q6(b)** (2,080 requests): 72.9 + 12 + 208.4 + 32.1 + 116.5 + 332.8 + 5 ≈ 780 s ≈ 13.0 min ≈ 234 SU. So the other 180 briefs add about 67 SU.
- **The train stream's draws** (Ref_b's and the pool's together) add about 150 SU. Held-out alone would be 72.9 + 10 + 32.1 + 32.1 + 17.9 + 332.8 + 5 ≈ 503 s ≈ 151 SU. That option is not offered, because the pool comes from the same draws (Q6).
- **At most:** 810 SU under either answer.
- **Used so far:** 2,583.8 of the 1,000,000 SU target (`CHANGELOG.md:6`).

## 3. Rendering and features: what is rendered here, what waits, and which code decides

| set | rendered here | features here | instead | why |
|---|---|---|---|---|
| held-out 160 | yes | no | features from the job that featurizes the seed pool (`lofi-embed`, `taste_pipeline.md:344`; not built), with the same feat_ver and node type | the tuples are fixed now, with G7, their audio hashes and their render code; the RM's features must share one feat_ver and hardware key with its training clips (`SCORER.md:89, 282-290, 299-303`) |
| Ref_b 1,600 | no | no | rendered and featurized with the lofi-grpo ask (`DECISIONS.md:603, 633`), on run 1's GPU type and feat_ver; refs.json once an RM exists (`SCORER.md:227-230`) | cache records never cross hardware, any code edit re-renders, and refs.json records neither (`rm_api.py:152-172`); a render now buys only G7, at about 1,600 × 8.32 CPU-s ≈ 3.7 CPU-h (estimate: the gate run's 104 s × 4 CPUs / 50) |
| pool 660 | no | no | `lofi-render-pool`, then `lofi-embed`, with the rest of the pool (`taste_pipeline.md:344`; neither built; render-pool is Shresth's, `CHANGELOG.md:150`) | one render and feature pass for the whole pool |

- **The render-code key.** Every G7 outcome is recorded with the stamp of the step that rendered it and `features.source_sha8()`, the hash of the render and feature code (`features.py:46-60`; `SCORER.md:89`). The clip id does not change when the render output does (`SCORER.md:299-300`).
- **Which code decides each set's G7** (registered in D-023):
  - held-out: this run's render step;
  - Ref_b and the pool: the render jobs named in the table, under their own recorded keys, by §1.5–1.6's rules.

  No later outcome re-selects anything else.
- **The audio hash.** `audio_sha256` is the float32 render identity: `pcm_sha256` of the mix (`pipeline.py:266`; `src/lofi_rl/render/wavio.py:27-29`). The registry's `pcm_sha` is "the sha256 of the canonical int16 PCM" (`taste_pipeline.md:77`), which is the queue job's to record. A later re-render of a held-out clip is compared on `audio_sha256`, like with like:
  - under the same `source_sha8`, palette and `render_cfg_sha8`, and the same audio stack (the manifest's `platform`, less `cpu`), a mismatch is a determinism violation (`taste_pipeline.md:78`). The render-code key holds each step's stamp, so a later job never shares it;
  - otherwise it is logged as render drift and never changes the set;
  - a held-out clip that fails G7 at its round's render is not served. Its tuple is dropped from the gate and reported (Q8(a)).
- **No freeze (Claude's default, §9).** The render path is not frozen with `.sync-freeze` (`DECISIONS.md:261-276`).
  - The project's own answer to render drift is detection: feat_ver hashes the render code, and a change re-renders (`SCORER.md:299-303`).
  - A freeze would block, through submit.sh's mirror check (`scripts/submit.sh:21-29`), every cluster job whose commit edits one of those files during Phase 2.
- **Serving.**
  - The held-out FLACs stay on KEEP.
  - The bws4 rounds come from the queue job, which re-renders picked clips from their LoopScripts (`taste_pipeline.md:105-108`; not built).
  - Nothing goes to the inbox.

## 4. Records and validity

### 4.1 Validity
`python -m lofi_rl.step0 check RUN_DIR [--redo REDO_DIR] [--json]` runs on the laptop after `make pull`. Its list is D-020's (`DECISIONS.md:208-223`), with D-022's values (`:900-913`), adapted to per-request seeds.
- Each condition is reported by name, with what the files show.
- Exit 0 is valid, 3 invalid and 2 bad input. The last line is `STEP0_CHECK <verdict>`.
- Types are the registered ones, as gate1 reads them (`docs/GATE1.md:53-54`).

| group | conditions |
|---|---|
| Complete | `generate.stopped` null; `generate.partial` false; `n` = `requested` = the marked streams' requests (3,520 under Q6(a), 2,080 under (b)); `unreadable_lines` 0; `requests.missing`, `requests.duplicate` and `requests.foreign` all 0 |
| Seeded | `requests.seed_mismatch` 0: each row's `draw_seed` and `sp_seed` equal `brief_streams.draw_seed(seed, index, k)`; `probe.passed` true; each cell's rows share one `prompt_tokens`, equal to runs/junk-13788483's for that cell (1,325–1,411) |
| As registered | `generate.model.revision` `cdbee75f…`; `streams_file.sha256` `bb2daee5…` and `fresh` true; the streams drawn equal the mark's; every row's brief equals `sample_brief(seed, index)` (requests.brief_mismatch 0); per stream, `prompt.version` v2, `template_sha256` `127e3094…`, `template_matches_generate` true, `rows_matching` the stream's requests (320 and 3,200 under Q6(a)), `example_mismatch` 0; `palette.sha256` `09f80a64…` in generate, in score per stream with `matches_generate` true, and in render (clips.json); `sampling` equal to runs/junk-13632676's |
| Registered first | `registration.found` true: the job's code tree held D-023's mark, equal to `step0.MARK` |
| Constrained, per stream | `constrained.enabled` true; `regex_sha256` and `in_language.regex_sha256` `5c081828…`; `in_language.matches_generate` true; `stop.out_of_language` 0; `other.n` 0 |
| What the regex guarantees | `failures.G1`, `G3`, `G4`, `G5` and `crash` all 0 (`DECISIONS.md:221`) |
| One stamp, one path | generate.host.stamp, score_stamp and render_stamp show the same `commit=… dirty=0`; `path_file.blobs_ok` true; at that commit, every file `docs/prereg/D-023_step0_path.txt` lists has its listed blob (git on the laptop). A redo pair follows §4.2's rule instead |
| Held-out render | `render.errors_final` 0, `render.skipped` 0, `render.other_gates` 0; no held-out brief undetermined |
| Sets | every heldout row is stream `rm_heldout`, and every refb and pool row stream `train`; the pool is non-empty; under Q5(a) no pool pick has `heldout_text` true, and under (b) or (c) none is a held-out clip |
| Reproduced on the laptop | sets.json equals `step0.select` over scored.jsonl and clips.json; every clip id equals the formula (`taste_pipeline.md:76`) over its recorded canon_sha256, brief, palette label and render_cfg_sha8; `render_code.source_sha8` equals `features.source_sha8` over the render stamp's commit, read with git |

- **The path file,** `docs/prereg/D-023_step0_path.txt`, is written by `python -m lofi_rl.step0 path --write` at the code commit, last, after every file it lists is final. It is the precedent's form (`docs/prereg/D-020_junk_path.txt`, `D-021_masked_smoke_path.txt`): "<git blob sha>  <path>" lines, as `git ls-tree` prints them.
  - It lists the import closure of `lofi_rl.step0` (the lofi_rl and kit modules reached by static import analysis, imports inside functions included).
  - It also lists `jobs/step0.sbatch`, `scripts/job_env.sh`, `scripts/_kit_env.sh` and `kit.env`.
  - And the committed data files those modules read: the palette and its committed sources, prompt v2's example files, and the streams file, as D-020's list did.
  - The job runs from its snapshot, so no `.sync-freeze` is set (§3).
- **Shown, never judged:**
  - `constrained.backend` (as D-020 treats it, `DECISIONS.md:223`);
  - the replay readout, the timings and the GPU;
  - `render.errors_retried`;
  - near-copy, repair and `heldout_text` counts;
  - item 8's readouts;
  - `check --rescore`, which re-scores completions.jsonl with the laptop's code and lists any differences from scored.jsonl. It is meaningful at the run's commit.

### 4.2 An invalid run, and recovery
D-020's rule applies (`DECISIONS.md:224-228`).
- **It decides nothing.** It is reported, and its rows never enter any set, pool or file.
- **A scoring or selection defect, or a held-out render cut short or failed by infrastructure:** the run is redone from its own records, with no new generation.
  - **The fix comes first:** any code fix, and every file it changes, is recorded in an amendment of D-023 before the redo.
  - **The redo:** `python -m lofi_rl.step0 render --from OUT --run REDO`, after `score --from OUT --run REDO` when the scores are redone (a score redo is always followed by its render redo: new scores need new sets, which only render writes), in `lofi-step0-redo`. That is a CPU job on `main`, a new type of at most 1,000 SU, which Claude builds and registers only when it is needed.
  - **Its folder:** the redo writes `$REMOTE_KEEP/runs/step0-redo-<job>/`, reads OUT's summary read-only, and never writes under OUT. Nothing under runs/ on the cluster is deleted (`CLAUDE.md:175`).
  - **Its stamp:** the redo runs at a later commit, since `make submit` needs the mirror at the committed laptop code (`scripts/submit.sh:21-29`) and each step stamps from its own tree (`junk.py:315-318`; `scripts/job_env.sh:35`). It is valid only if, at that commit, every file the path file lists has its listed blob, except the files the amendment names. That is D-020's second amendment's rule (`DECISIONS.md:261-276`) with this list.
  - **Its check:** `step0 check OUT --redo REDO` reads generate's records from OUT and score's and render's from REDO, and checks both stamps.
- **Anything else** (incomplete, a setting, the seeds, the grammar, the stamp, the registration, the streams): fixed, with the fix recorded in D-023 before the re-run. Then the run is drawn again whole, once, into its own folder, and never merged with the invalid rows.
- **Invalid twice:** it goes to Shresth with the records (`DECISIONS.md:540`).
- **The first valid run, or valid pair, fixes the sets.** No later run replaces them. A new set is a new file version, with its own DECISIONS entry.

### 4.3 What make pull brings back
- **The summary:** `runs/step0-<job>/summary/**`, and a redo's, from `$REMOTE_KEEP/runs/` (`Makefile:37`; `CLAUDE.md:58`): §2.6's seven files, about 7 MB (estimate). Pull before reading any result (`CLAUDE.md:64`).
- **Not the clips:** the FLACs and manifests stay on KEEP (`CLAUDE.md:66`).
- **Not the Slurm log,** which stays on scratch.

### 4.4 Fixing the outputs
Once check reads VALID:
1. **`python -m lofi_rl.step0 prereg runs/step0-<job> [--redo runs/step0-redo-<job>] --write` writes two files.** Neither holds a LoopScript text (§5).
   - **`docs/prereg/heldout_v1.json`** (`lofi-rl/heldout/1`), the file `taste_pipeline.md:128` names. It holds:
     - the run (job, node, GPU, stamp), its render-code key, the settings and their shas, and the rule;
     - per tuple, the brief (id, cell, bpm, tonic, twins) and the shown example's id;
     - per clip: k, draw seed, text_sha256 and canon_sha256, the clip id, `audio_sha256` (the float32 render identity), LUFS, true peak, P_fix, near_copy, copy_jaccard and its verbatim A and B flags;
     - the short briefs;
     - the texts' pin: completions.jsonl's path and sha256, where each clip's text is found by its text_sha256;
     - item 8's readouts beside D-008 (`DECISIONS.md:482-486`), reported and never gated, as numbers and flags:
       - each tuple's pairwise J5 in both copy forms (junk's, the canonical text without `sax:`, and the K/D form, `src/lofi_rl/drafts/check.py:71-73`);
       - its verbatim-line sharing (how many of its 4 carry A, B, and both);
       - for each of its 6 pairs, whether the two differ in A or B. Those flags fix, before any vote, the pairs that the readout "agreement on the pairs whose two clips differ in A or B" reads (`DECISIONS.md:484`).
   - **`docs/prereg/step0_v1.json`** (`lofi-rl/step0-run/1`). It holds:
     - the run, a redo if any, and every summary file's path, size and sha256;
     - the counts per stream, the validity verdict, the probe and the replay;
     - Ref_b per brief: k, draw seed, text and canonical sha256 of its 4, and its spares' k in order;
     - the pool's 220 and their picks;
     - every train draw's `heldout_text` flag, with the train rows that share each held-out clip's canon_sha256, counted per cell;
     - the short briefs, and heldout_v1.json's sha256.

     The texts are not repeated: they are in completions.jsonl, pinned by its sha256.
   - **Both files** are ASCII JSON, one record a line, and byte-identical when written again from the same run, as `brief_streams.text` is (`brief_streams.py:361-379`). The writer never replaces a different file and has no `--force`.
2. **Commit both.** D-023 gains its Outcome: the run, the counts and both sha256s ("with its sha recorded in DECISIONS.md", `taste_pipeline.md:128`). It is committed before the first Phase-2 vote (Q7).
3. **The KEEP run folder is kept.** Nothing under runs/ on the cluster is deleted (`CLAUDE.md:175`). Two copies hold the texts: KEEP and the laptop's runs/, which is never committed (`CLAUDE.md:126`) and so is in no git worktree.
4. **The CHANGELOG** records the job's SU (`python3 scripts/session_report.py`).

## 5. Separation and privacy
- **Held-out is never trained on:** "Never trained on, selected on, or tuned on" (`taste_pipeline.md:129`).
  - **Its briefs** (5101:*) are never in a GRPO dataset. Run 1 is 596:[0, 400) (`DECISIONS.md:433`).
    - Today only a comment guards that (`configs/grpo_smoke_v2_grammar_masked.yaml:30`): `rl/config.py:106, 262` and `train.py:159-160` take `briefs.seed` as written.
    - D-023 asks for a refusal of the evaluation seeds in `lofi_rl.rl.config`, built with Phase 2's GRPO smoke config (§8).
  - **Every rm_heldout row is held-out data:** taken or not, in completions.jsonl, scored.jsonl and replay.jsonl. It is used only through heldout_v1.json, never as a base-model, triage, control or pool clip.
  - **Held-out clips** enter no seed pool, triage or round other than their own held-out rounds and those rounds' hidden repeats.
    - Held-out tuples are repeated first (`taste_pipeline.md:45`), and a round has one kind, so a repeat is kind `repeat` (`RATING.md:176-177`; `src/lofi_rl/ui/manifest.py:156`).
    - Labels from both go only to the gate: acc from first showings (`taste_pipeline.md:183`), p from repeats (`:184`).
  - **Clip ids and texts.** The clip id hashes the canonical text with bpm, tonic, mode and energy, and no brief id (`SCORER.md:123, 130-131`; `protocol.py:68-71`).
    - So a twin's draw with its counterpart's text is the same clip, and the pool skips it.
    - The same text in another brief of the cell is another clip with the same LoopScript features (`SCORER.md:226`). Q5 settles that case. Under (a), the pool skips such draws, and the RM loader drops every label on a clip whose canon_sha256 equals a held-out clip's, from any source (§8).
    - Near-identical texts stay, since held-out means "fresh samples" of the same 9 prompts (`DECISIONS.md:443`).
  - **Twin counts.** 7 of the 9 twin counterparts are among the 220 under either Q1 option (this session). Two estimates, at the per-cell rates:
    - about 0.16 pool draws would be a held-out twin's clip (3 × 4 draw pairs per counterpart in the pool);
    - about 0.51 Ref_b draws would be (4 × 4 pairs per counterpart). Those are reported only, since Ref_b trains nothing.
- **No row of this run enters the copy pool.** The pool "never holds the 660 step-0 samples or any other model output" (`DECISIONS.md:449-450`).
  - The copy pool reads `.ls` files, or directories of them, non-recursively, with no provenance check (`src/lofi_rl/rl/config.py:136-143, 335-338`).
  - The step-0 tools write no `.ls` file anywhere (tested, §6.1).
  - D-023 names the key the later unit that builds run 1's pool checks against (§8).
- **Ref_b is never voted on, served or put in the seed pool** (Q2(a)).
- **The 660** are never in the copy pool, and never Ref_b's (§1.6).
- **No committed file holds a step-0 text.**
  - Workflow agents run in full checkouts (`tests/test_privacy.py:43`), and they write the seed pool (D-018, `DECISIONS.md:115-119`; `taste_pipeline.md:336-344`).
  - So both prereg files hold hashes, and the texts stay in runs/ (§4.4).
- **Nothing under `data/`:** the tools read and write nothing there, and no output here is a vote (D-010, `DECISIONS.md:66-69`). No `manifest.hidden.json` is read.
- **Shresth is shown nothing in this unit.**
  - Later, held-out tuples reach him only as bws4 rounds of kind `heldout`, listed only in manifest.hidden.json, which is never served (`RATING.md:135, 176-177`; `taste_pipeline.md:44`).
  - The page gets only the scene and opaque ids (`RATING.md:160-161`).
  - heldout_v1.json lists clips, not rounds, and is committed by design (`taste_pipeline.md:128`).
- **Nothing about sound,** anywhere in this unit.

## 6. Tests and the order of work

### 6.1 Tests (laptop, synthetic: no vLLM, no GPU)

| file | checks |
|---|---|
| `test_step0_requests.py` | 320 and 3,200 requests (Q6(a)) or 320 and 1,760 (Q6(b)) in stream, brief and k order; every seed is `draw_seed(seed, index, k)`, and the file's vectors hold (596:0 k0 → 379233320; 5101:0 k0 → 1702876545; `brief_streams_v1.json:50-57`); an edited streams file is refused; a `STREAMS` that differs from `MARK` is refused |
| `test_step0_select.py` | hand-built tables: everything passes; a gate failure; a repeat skipped for held-out and kept for Ref_b; a G7 failure taking the next draw; the pool after Ref_b, skipping Ref_b's texts, a held-out text (Q5(a)) and a twin's clip; Ref_b's spares skipping pool draws and pool texts, with §1.5's {k0, k1, k3, k7} example; a pool G7 failure not replaced; short briefs and their reasons; the same result whatever the row order; both Q1 walks reproduce this session's counts (220 each; 35/169/16 and 30/176/14; 206 shared) |
| `test_step0_generate.py` | on a fake engine (as `tests/test_junk.py:484-502`): one SamplingParams per request, n 1, its seed, one list per call; chunks, fsync and the deadline (`stopped: deadline`, partial); the probe, 3 requests at each of PROBE_SEEDS (1, 2) in one call, passing with seeds honoured (2 of a seed's 3 alike are enough) and stopping (exit 2, nothing drawn) on an engine that ignores the seeds, on one that gives a call one randomness (`OneNoise`), on one that samples every request with the call's first SamplingParams (`FirstParams`) and on one that refuses a list of them; each pre-check refusing before the engine loads (a mark that differs, a path-file blob that differs, a PROBE_SEEDS value among the draw seeds, or two equal); the replay never in completions.jsonl, its counts right, and a cut, failed or killed replay leaving the draws complete |
| `test_step0_score.py` | score_row's fields plus the four new ones; the per-stream readouts; the candidates; no `junk.json`; it runs with the policy env's absent modules blocked (as `tests/test_junk.py:1311-1338`) |
| `test_step0_render.py` | with render_clip stubbed (as `tests/test_junk.py:1043-1185`): round 1's 160; G7 leading to a next round; an error retried once and rendered (`errors_retried` 1, `errors_final` 0), and an error twice (undetermined, exit 1); the deadline; a partial run refused; `render_code` recorded; the pool's selection with the held-out-text exclusion; `--from` writes only under `--run`. With FluidSynth and LOFI_ASSETS (skipped otherwise): two real renders whose clip ids equal `shootout.render.clip_id` and `scoring.protocol.clip_id` (`protocol.py:74-76`) |
| `test_step0_check.py` | a valid synthetic run passes, condition by condition; each tamper is caught: a seed, a missing or repeated request, a misaligned prompt_tokens, an out-of-language stop completion, another palette or regex sha, two stamps, a commit whose files differ from the path file, no D-023 mark, streams that differ from the mark, an edited sets.json, a wrong clip id, a pool row from rm_heldout; a retried error that left no brief undetermined still passes; a redo pair on a temporary git repo passes only when its changed files are the amendment's |
| `test_step0_path.py` | the path file lists the import closure (imports inside functions included), the job, the env scripts and the data files; written twice it is byte-identical; a changed file is caught both by blob in a tree and by git at a commit |
| `test_step0_prereg.py` | an invalid run is refused; writing again gives byte-identical files; a different existing file is never replaced; item 8's readouts on hand-built tuples; no LoopScript text in either file; no `.ls` file and nothing under `data/` or `inbox/` is written |
| `test_step0_job.py` | as `tests/test_junk_job.py`: the Cost line against `#SBATCH` and the registry (≤ 1,000, by Claude); `--no-requeue`; generate, score, render in policy, policy, scorer; each under a timeout, with deadlines under the timeouts and the stack dump under generate's; the copies after the render, summary first; the EXIT and TERM traps; a non-empty OUT refused before the traps are set; the sum ≥ 60 s under the walltime; the run on the node's disk; OUT and LOFI_CACHE on KEEP; no inbox; any argument refused |
| `test_step0_imports.py` | the policy-side step0 modules import with the blocked list; the junk names step0 uses exist with the values it expects (SAMPLING, `STRUCTURED_BACKEND == "xgrammar"`, `DEFAULT_CHUNK == 128`, `HANG_MARGIN_S == 90`, render_clip's parameters) |
| `test_step0_docs.py` | this file and D-023 state the registered values: the counts, m per set, the rule name and the mark, the walltime and the cost |

No existing test changes. `tests/test_approvals.py:12-20` keeps checking every registered sha, junk's included.

### 6.2 The order of work
1. Shresth's answers to §7, or his OK on the defaults.
2. Build `src/lofi_rl/step0/` and `jobs/step0.sbatch` with their tests. junk.py, junk_rate.sbatch and their tests stay unchanged (`git diff --stat cfc24df` over them is empty).
3. An independent review, with its findings applied, as the gate tool's was (`CHANGELOG.md:70-72`).
4. Finalize `jobs/step0.sbatch`. Write D-023 (§8: his answers, rule `step0-select/1`, the mark, the job and its SU, the validity list, the invalid-run and redo rules, the supersessions and the asks of later units) and this file. Then write the path file last (`python -m lofi_rl.step0 path --write`).
5. Register: `make approve JOB=jobs/step0.sbatch MAX=810 BY=Claude`, then `python3 scripts/approve_job.py --check`. The registry sha covers the script's `#SBATCH` and non-comment lines (`scripts/approve_job.py:26-30`), so this comes after the last change to the script. Add the CHANGELOG lines: the registration, as D-003's amendment asks, and the unit's entry.
6. Run `pytest -q`; it now includes test_step0_job's registry check (as `tests/test_junk_job.py:44-46`), which needs step 5 (`CLAUDE.md:110`). Make one commit holding the code, the tests, this file, D-023, the path file, the registry line and the CHANGELOG, and push. If anything on the path changes after step 4, the path file (and, for the script, the registration) is redone first.
7. On the laptop, the pre-flight: `PYTHONPATH=src .venv/bin/python -m lofi_rl.step0 path --check` (`STEP0_PATH ok`: every file the path file lists has its blob in this checkout) and `.venv/bin/python -m pytest -q tests/test_step0_path.py::test_the_committed_path_file_and_d023_pass_generates_pre_checks_in_this_checkout` (the same blobs, a redo's amended files aside, and D-023's mark read as generate reads it). Then, on USC's VPN: `make sync`, then `make submit JOB=jobs/step0.sbatch`. Run `scripts/watch_job.sh <id>` in the background.
8. `make pull`, then `python -m lofi_rl.step0 check runs/step0-<job>`.
9. If valid: `prereg --write`; commit the two files; add D-023's Outcome with their sha256s and the CHANGELOG entry with the SU; run `pytest -q`; push. If invalid: §4.2.
10. **Left to later units:**
    - the 660's render and features (render-pool, embed), by Q2's G7 rule, recording their render code;
    - Ref_b's render, features and refs.json (the lofi-grpo ask), by Q2's G7 rule, recording their render code. If Phase 2's GRPO smoke runs with taste live, its briefs need theirs first (`SCORER.md:236-238`). No Phase-2 smoke config exists yet (`ls configs/`; `DECISIONS.md:926`). That its briefs are 596:[0, 40) is an inference from the existing smokes' briefs block (seed 596, count 40, start 0; `configs/grpo_smoke_v2_grammar_masked.yaml:29-32`). All of 596:[0, 40) are among the 220 under either Q1 option (this session);
    - the held-out bws4 rounds (the queue job). First showings are needed by about Nov 13. That is an inference: repeats come at least 2 days later, and held-out rounds are "all done by Nov 15" (`taste_pipeline.md:45, 114`; `DECISIONS.md:88`);
    - the planner taking §1.6's 220 briefs;
    - the RM loader's held-out rules, the copy-pool key and the seed guard (§8);
    - the other step-0 draws, which item 2 does not list: the dev quads and the final A/B's arm (`taste_pipeline.md:115, 131-133`).

## 7. Open questions for Shresth
Nothing runs before his answers or his OK on the defaults. "The yardstick" below is PAR's Ref_b: the 4 base-model loops per training brief that run 1 scores new loops against.

- **Q1. Which 220 training briefs give the seed pool its 660 base-model loops?** The seed-pool plan says "220 briefs stratified by energy × mood × tempo" (`taste_pipeline.md:329`), but its planner was never built and no tempo rule was written.
  - (a) **Recommended.** Balance by energy × mood (25 or 24 a cell) and, inside each cell, by tempo band: 60–74, 75–90 and 91–95 BPM, in the shares the brief sampler uses (4, 19 or 18, and 2 a cell). Where run 1's 400 briefs are short of a band in a cell, the cell fills up from 75–90. That gives 35 slow, 169 middle and 16 fast.
  - (b) Energy × mood only, the rule of the other balanced sets: 30 slow, 176 middle and 14 fast.
  - (c) A new brief stream just for the pool. That needs new draws and a new pre-registration.

  206 of the 220 are the same under (a) and (b). The same briefs also serve the 1,100 Claude-written loops.
- **Q2. May a seed-pool loop also be one of its brief's 4 yardstick loops?**
  - (a) **Recommended.** No. The pool takes the 3 draws after the yardstick's 4 whose LoopScripts differ from them. So you never vote on a yardstick clip, and the 220 pool briefs and the other 180 are treated alike.
    - For the rare loop that later fails the render check: a yardstick loop is replaced by the brief's next unused draw, never a pool loop, and a pool loop is dropped, not replaced, since a yardstick must hold exactly 4.
    - Example: yardstick draws 1–4, pool 5–7. If draw 3 fails, the yardstick becomes 1, 2, 4 and 8, where the registered wording would take draw 5.
    - The same LoopScript also recurs across a cell's briefs: about 176 of the 1,600 yardstick loops share their text with another brief's pool loop (estimate). So (a) separates clips, not texts.
  - (b) Yes: reuse 3 of the 4 yardstick loops, as the registered wording reads. It saves 660 renders later, but 220 of the 400 briefs would then be scored against clips you voted on, and the other 180 not.
- **Q3. Which base-model loops count as passing?** The reward's gate is hard × 1[PQ ≥ τ_PQ] × clip(P_acc / 0.5, 0.25, 1) (`taste_pipeline.md:233`), so only the hard checks and the PQ junk floor can zero it.
  - (a) **Recommended.** The hard checks only: it parses, compiles and renders in spec.
    - The PQ floor is left out, because its threshold comes from your Good-or-better votes in the triage of 300. That triage draws its base-model clips from these 660 and comes after the held-out set must be fixed (`:235`, `:360`). The scorer itself reads a missing threshold as no floor (`SCORER.md:207`).
    - Loops that copy the shown example, and loops the parser repaired, still count: they are real step-0 output, and the readouts report them.
  - (b) As (a), but near-copies of the shown example are left out, as in the Phase-1 ear check.
  - (c) As (a), but repaired loops are left out.
- **Q4. Two identical loops from one brief.**
  - (a) **Recommended.** A held-out set of 4 skips the repeat and takes the next draw, since one round cannot play the same clip twice. The yardstick keeps repeats, since each is an honest sample. About 1 held-out brief and 13 yardstick briefs are affected (estimates).
  - (b) Both skip repeats.
  - (c) Neither skips; a held-out set with a repeat could then not be played.
- **Q5. A held-out loop's exact LoopScript turning up elsewhere.** The base model sometimes writes the same LoopScript for two briefs of one energy and mood: 75 of 13,836 same-cell pairs in the gate run. Such a loop is a different clip at another tempo and key, but the reward model's LoopScript features see the same text. By estimate, about 30 of the 660 pool loops would share the text of about 11 of the 160 held-out loops.
  - (a) **Recommended.** The pool skips those loops, and the reward model never trains on any vote involving a loop with a held-out loop's exact text, whatever its source, the Phase-1 ear check included.
    - About 12 pool briefs would then come up a loop short, instead of about 2 (estimate).
    - Near-identical loops still count as different, since held-out means fresh samples of the same 9 prompts.
  - (b) Keep the pool as drawn; the reward model just never trains on votes involving those ~30 loops.
  - (c) Neither; report the gate's agreement with and without the ~11 affected held-out loops.
- **Q6. The yardstick's 1,600 loops: draw them now, or with the run-1 request?** D-021 lists them in this step (`DECISIONS.md:537`), but put them in the run-1 (lofi-grpo) request (`:603`) when Claude's job limit was 500 SU. The seed pool's 660 come from the same training-brief draws. So drawing only the held-out loops now would leave Phase 2's triage without base-model loops, and that is not offered.
  - (a) **Recommended.** Draw all 400 training briefs now: about 301 SU expected, at most 810 for the job (estimates). Render and measure the yardstick with the run-1 request, on run 1's GPU type.
  - (b) Draw the 220 pool briefs now (their yardstick is fixed too) and the other 180 with the run-1 request: about 234 SU now (estimate).
- **Q7. "Committed before the first vote" was written Oct 30, before the shootout votes. What should it mean now?**
  - (a) **Recommended.** Before the first Phase-2 vote (the triage of 300 or the first best/worst round), and before any round with a held-out loop is built.
  - (b) You name another point.
- **Q8. A brief with too few usable loops in its 8 draws.** Not expected: about 1 in 700 million per yardstick brief, and about 1 in 2,500 that any of the 40 held-out briefs falls short (about 1 in 800 under a cruder model; estimates).
  - (a) **Recommended.** The held-out set has one set fewer, and a held-out set whose loop later fails the render check under newer render code is dropped the same way, reported. A yardstick brief left short leaves run 1's 400 and comes to you with the run-1 request.
  - (b) Ask you when it happens.

## 8. D-023: what it records, supersedes, reads and asks of later units
D-023 is written at §6.2 step 4, with Shresth's answers in his words, and committed with the code before the job.
- **Decided by:** Shresth (his answers to §7, or his OK on the defaults), with Claude's defaults (§9).
- **It records:**
  - rule `step0-select/1`: §1.3–1.8 as answered;
  - the mark the job checks (§2.3): one line naming the rule and the streams drawn, equal to `lofi_rl.step0.MARK`;
  - the job `lofi-step0`, its registry sha, at most 810 SU, registered by Claude under D-003's amendment (`DECISIONS.md:26`);
  - the path file and the stamp rule, the validity list, and the invalid-run and redo rules (§4.1–4.2);
  - after the run, its Outcome: the run, the counts and both prereg files' sha256 (`taste_pipeline.md:128`).
- **What this entry supersedes** (each with the question that settles it):
  1. `docs/prereg/brief_streams_v1.json:40`, "takes the first m passing draws in k order", for these three sets. The file itself stays unchanged ("do not edit this file", `:4`). For these uses:
     - the held-out set takes the first 4 passing draws whose canonical texts differ (Q4);
     - "passing" is the hard gates without the PQ floor (Q3);
     - Ref_b's G7 replacement skips the pool's draws, and a pool draw that fails G7 is not replaced (Q2);
     - the pool's draws start after Ref_b's 4, with texts unlike Ref_b's (Q2), and, under Q5(a), unlike every held-out clip's.
  2. `taste_pipeline.md:127`, "its first 4 step-0 samples that pass the gate": as item 1 (Q3, Q4).
  3. `taste_pipeline.md:233-235`, the gate's PQ factor, for the three sets (Q3).
  4. `taste_pipeline.md:128`, "committed before the first vote" (Q7).
  5. `taste_pipeline.md:329`, the planner's 220 briefs. This unit's 220, by Q1's rule, stand in for them, and a later `lofi_rl.seeds.plan` takes them.
  6. `taste_pipeline.md:349`, "from `lofi-gen`": `lofi-step0` draws the 660.
  7. `DECISIONS.md:603`, Ref_b's 1,600 samples in the lofi-grpo ask: their draws move here, in full or for the pool's 220 (Q6). Their render, features and refs.json stay with the ask (`:603`, `:633`).
  8. `DECISIONS.md:433` and `:436`, run 1's [0, 400) and the 40 held-out briefs: a short brief, or a held-out tuple whose clip later fails G7, leaves them, reported (Q8(a)).
- **Readings, which supersede nothing:**
  - "Never trained on, selected on, or tuned on" (`taste_pipeline.md:129`) holds at the text level under Q5(a) or (b). Near-identical texts remain, since held-out means "fresh samples" of the same 9 prompts (`DECISIONS.md:443`).
  - D-021 item 2's "the seed pool's 660" is a planning count. Short briefs make it fewer, and are reported.
- **Asked of later units:**
  - **The RM loader** (`lofi_rl.taste`, not built; `SCORER.md:229-230`):
    - labels on held-out clips go only to the gate: acc from first showings, p from repeats (`taste_pipeline.md:183-184`);
    - under Q5(a) or (b), every label on a clip whose canon_sha256 equals a held-out clip's (heldout_v1.json) is dropped from training, whatever its source, and the count is reported beside D-008.
  - **The queue job** (`taste_pipeline.md:105-108`; not built):
    - held-out clips appear only in their held-out rounds and those rounds' repeats;
    - a re-render is compared on `audio_sha256` as §3 says, and a clip that fails G7 there drops its tuple (Q8(a)).
  - **The pool's and Ref_b's renders** (render-pool; the lofi-grpo ask): G7 by Q2's rules, under their own recorded render-code key; their outcome re-selects nothing else.
  - **Run 1's copy-pool builder:** it refuses any entry whose canon_sha256 is one of this run's step-0 rows' (scored.jsonl, pinned by step0_v1.json), unless the entry is one of prompt v2's 9 reference texts. A mutation or broken copy made from a step-0 text counts as model output (`DECISIONS.md:449-450`). `taste_pipeline.md:351` does not say what broken copies are made from.
  - **`lofi_rl.rl.config`:** it refuses a `briefs.seed` equal to an evaluation stream's seed (5101–5104, `brief_streams.EVALUATION`), with a test. This is built with Phase 2's GRPO smoke config, before that smoke runs.

## 9. Claude's defaults here (each overridable; no answer needed unless he objects)
- **Draws:**
  - every drawn brief's 8 draws are made (§1.7);
  - G7 is applied lazily, in k order (§1.3);
  - Ref_b uses the passing rule and keeps repeats, with spares for G7 (§1.5);
  - under Q1(a), the bands 60–74, 75–90 and 91–95 BPM, their shares and the fill-from-centre rule (§1.6).
- **Separation:**
  - twins stay;
  - a pool draw that is a held-out clip is always skipped;
  - the yardstick is never played (§5);
  - no committed file holds a step-0 text (§4.4, §5).
- **The engine and the job:**
  - the seed probe on low/hopeful at seeds 1 and 2, and the replay readout (§2.3);
  - engine seed 0, chunks of 128, an A40, 45 min and at most 810 SU (§2.1, §2.8);
  - no argument, `--no-requeue`, and a used OUT refused (§2.1, §2.7);
  - the redo job is built only if needed (§4.2).
- **Records:**
  - only the held-out set is rendered here (§3);
  - the render code is recorded with every G7 outcome, and the render path is not frozen (§3);
  - both prereg files hold hashes only, and pin the summary files (§4.4);
  - a new entry, D-023, carries the rules, rather than an amendment inside D-021.

## 10. As built (2026-11-07)
The code on branch `wt/step0` (`src/lofi_rl/step0/`, `jobs/step0.sbatch`, `tests/test_step0_*.py` and their helpers) departs from sections 1–9 where these bullets say. Each is accepted: the three builders' 45 stated deviations ("deviation n" below), then the fixes from an end-to-end test (E1–E3) and three reviewers (R1–R3), each finding checked by an adversarial verifier, from the recheck of those fixes (N1–N6), and from a second recheck, of the N fixes (N7–N10). A sentence above that a fix made wrong is corrected in place, and its bullet says so. §7's questions are unchanged; the code holds the recommended (a) of each until D-023 records Shresth's answers.

**§1, what is drawn**
- **The held-out walk under lazy G7** (deviation 9): a draw whose text equals a draw still to render counts as its repeat. If that draw then fails G7, the next round may take the repeat, which renders then; the taken set is the one full information gives.
- **Ref_b and the pool before their renders** (deviation 8): a train draw with no render record stands by its records, so sets.json's Ref_b 4, spares and pool picks are their records'. A pool pick later dropped at G7 stays out of the spares, a Ref_b replacement never moves the pool, and a Ref_b brief short by its records leaves its pool brief no draw.
- **The later renders of Ref_b and the pool** (finding R3-1): `select.select` with `pending_train` and `select.pending` with refb or pool refuse to run before the held-out set is decided by the records they are given, since Q5's exclusion reads the final held-out set. A later unit passes every scored row and this run's clips.json records with its own.
- **Clips across twin train briefs** (finding R1-1; §1.5 and §1.6 corrected in place): every Ref_b by its records first; then each pool brief, in stream order, skips another brief's Ref_b clip (`refb_clip`, under Q2(a) only) and an earlier pool brief's pick (`pool_clip`, under any answer); then the spares and a G7 replacement leave out every pool pick. The same result comes whatever the rows' order. The skips are rare, about 0.8 draws a run (the review's simulation), so §1.6's shortness estimates stand, a little low.
- **Q5(a) across cells** (deviation 10): a held-out clip's text is compared in every cell, a short held-out brief's clips included.
- **The record of a short brief** (deviations 1, 5): each sets.json entry also carries `undetermined` (a draw its walk needs has no decided render). A draw not taken gives a gate code (score's, or a render's G7 or other gate), `near_copy` or `repaired` (Q3(b), (c) only), `repeat_of:<k>` (any text equality inside the brief), `refb`, `pool` (Ref_b's walk reaching a pool draw after a G7 replacement), `heldout_text`, `heldout_clip`, `refb_clip` or `pool_clip`.
- **Q8(b), ask** (findings R1-2, N2, N7, N8, N9): only a short held-out or Ref_b brief, the ones Q8 is about, stops anything. Render exits 1, and prereg refuses, naming the line to add, until D-023 records his answer for the run as one line of its entry in the checkout's docs/DECISIONS.md, `D-023 short answer: run step0-<job>; as (a)`. It is read only as a line of its own (`prereg.short_answers`), as kit.gates reads an approval: in D-023's entry, outside HTML comments, with nothing before its head but the indent, one list bullet and what wraps the line (backticks, bold, italics), and nothing after the answer but that wrap's close and a period; a fenced or indented line reads too. A sentence, a note or prereg's own refusal that quotes the line, a blockquote, a table, a comment or another entry is no answer, so no text that only mentions the line answers for him. DECISIONS.md is no file check compares with the run's commit, so the line, written after the run, needs no redo and no constant changed. Prereg reads it in the checkout it runs in, the run's commit (finding N9): once main changes a file check reproduces the run with (`check.reproduction_files`, junk's whole closure among them, render's code too), check and prereg refuse there and run in a git worktree of the run's commit (`git worktree add --detach <dir> <commit>`, the commit the refusal names), whose DECISIONS.md predates the line. The line then goes into D-023 on main and, by hand and uncommitted, into the worktree's docs/DECISIONS.md; prereg writes both files into the worktree, and they are committed on main from there. Only that line, with no other answer for the run, lifts the refusal: the short briefs are then handled as Q8(a) handles them (the held-out set holds fewer tuples, a short Ref_b brief leaves run 1's [0, 400)), and both prereg files' `rule.answers.Q8` records the line. Any other answer for the run keeps the refusal, which then names those answers, never the line's own "as (a)", and says to replace the run's short-answer lines with the one line, leaving only it: the line added beside another answer (the line with a note after it among them) would refuse again. A short pool brief is reported under either answer: about 12 a run under Q5(a) (§1.6).

**§2, the job**
- **The pre-checks** (deviation 14): every pre-check whose input loads runs, and every failure is named in one `stopped` ("pre-check: " and the problems joined by "; "). `requested` and `streams` are null when the streams file is refused.
- **How D-023 is read** (deviation 15; finding R2-2): `requests.find_marks` reads each mark from its head to the next backtick or double quote, or its line's end, less trailing spaces, tabs, periods, asterisks, underscores and quotes; one must equal MARK. Generate's pre-check, check's git condition and the docs test all read it so, and a mark wrapped across lines or with text after it fails pytest rather than the job.
- **The answers line** (finding R1-3): `requests.answers_line()` writes Q1–Q5 and Q8 from their constants as one line, which D-023 holds verbatim beside his answers in words. A recorded answer the constants do not carry fails `tests/test_step0_docs.py`. MARK still names the rule and the streams only.
- **The probe** (finding R2-3; §2.3, §2.6, §2.9 and §9 corrected in place, and §6.1's test_step0_generate row, finding N5): 3 requests at each of seeds 1 and 2 in one call, passing when each seed's texts agree at least twice and the two seeds' differ. An engine that gives a call one randomness, or the call's first SamplingParams, stops there.
- **generate's exits and files** (deviations 17, 18): exit 1 when the deadline cuts the draws. completions.jsonl is opened when the draws start, so a pre-check stop, a missed probe or an engine that fails to load leaves generate.json alone.
- **The replay** (deviation 16; finding R2-1): it runs only once every draw is made; otherwise `replay.stopped` is "not run: the draws are not complete". Its forecast scales the longest timed chunk up to 160 requests, and an error is `replay.stopped` with a `trace`, the exit staying 0.
- **`partial`** (finding R2-1): generate.json turns `partial` false, with `replay.stopped` "running", before the replay starts. A kill in the replay (the watchdog's exit, TERM, scancel) then leaves a complete run that score and render take, not a full re-draw.
- **score** (deviations 4, 19): each stream gets its own CopyCheck, and its `requested` from generate.json; step0.json's `partial` is also true when generate.json is partial or stopped. `candidates` is {passing, refb, heldout_order}; rows that are not one run's draws are still scored and written, with `candidates` null and exit 1.
- **render's clock, errors and exits** (deviations 20–22): the deadline counts from the step's start. An exception out of `junk.render_clip` is an infrastructure error, retried once; `errors_retried` counts a retry that then rendered or failed a gate. Render also exits 1 on `other_gates` above 0, and exits 2, writing nothing, on an unreadable palette or render-code key or a `--run` and `--from` that overlap.
- **The clip id** (deviation 24): from the manifest's own fields through `lofi_rl.scoring.protocol.clip_id` (stdlib), tested equal to `lofi_rl.shootout.render.clip_id`; a record with no render carries the render pool's `render_cfg_sha8`.
- **`render --from`** (deviation 23): it reads score's records from REDO when a score recovery wrote them there, else from OUT, copying OUT's scored.jsonl into REDO byte for byte and keeping OUT's `run` and `score_stamp` in REDO's step0.json.
- **The modules** (deviations 7, 12, 25): `__init__.py` also holds the §7 answers, one named constant per choice (POOL_BRIEFS, POOL_BANDS, POOL_FILL_BAND, POOL_SHARES_REFB, PASSING, DISTINCT, POOL_EXCLUDES, STREAMS, COMMITTED_BEFORE, SHORT_RULE), the values read once from past runs (PROMPT_TOKENS, V0_SAMPLING) and each file's fields (FIELDS, BLOCKS, ENTRY_FIELDS). ROOT comes from `brief_streams`, so the stdlib modules import with yaml blocked. step0 also reads junk's public `cell_brief`, `RENDER_JOBS` and `head_lines`. `path` prints the list; `--write` writes it, replacing one that differs; `--check` checks a tree's blobs without git; `--commit REV` checks a commit's with git; `--root DIR` names the tree.
- **The files' fields** (deviations 1–4, 11; findings R1-7, R2-3): generate.json's SamplingParams repr is its own `sampling_params`, since `VllmEngine.sampling` rewrites engine.info's on every call; its timing is {load_s, probe_s, chunks, gen_s, tokens, tok_per_s, replay_s, total_s}, and its probe records `seeds`. step0.json's per-set readouts are `sets`: per set {briefs, taken, short, undetermined, near_copy, repaired, verbatim, same_clip}, with `heldout_text` per cell. sets.json's `twin_of` lists the rm_heldout–train twins, and its `heldout_text` is one record a train draw. clips.json also holds palette {id, sha256}, jobs and workers.
- **`same_clip`** (finding R1-7): per set, the taken draws that are the same clip (one canonical text at one cell, bpm and tonic) as another brief's taken draw in the held-out set, Ref_b or the pool. It tells a held-out twin's clip among Ref_b's from the same text elsewhere in its cell; reported only.
- **A used OUT** (deviation 26): the job also refuses an OUT that is a file, and copies clips/ only when `$RUN/clips` exists.
- **The copy-out** (finding R2-5; §2.7 and §2.8 corrected in place): a copy that fails is followed by one copy of the records only, from the EXIT trap, under `timeout -k 2 20`. The timeouts come to 2,611 s, 89 s under the walltime.

**§3, rendering**
- **A later re-render** (finding R3-5; corrected in place): the render-code key holds each step's stamp, so no later job shares it. A mismatch on `audio_sha256` is a determinism violation under the same `source_sha8`, palette, `render_cfg_sha8` and audio stack; the stack is in each manifest on KEEP, and clips.json does not copy it.

**§4, records and validity**
- **The command and its last line** (deviations 33, 34): `check RUN_DIR [--redo REDO_DIR] [--json] [--rescore]`. `--json` prints the report alone, its verdict inside; on bad input the last line is `STEP0_CHECK bad input`, the problems on stderr.
- **Bad input** (deviation 33; findings R2-7, N4): a summary/ folder or file missing or unreadable, a schema that is not the step's, or files that are not one run's own, as `lofi_rl.gate1.load_run` refuses them. A scored.jsonl or step0.json missing beside generate's records is named as the scoring defect §4.2 redoes. Three files may be absent (`check.MAY_BE_ABSENT`). Without replay.jsonl no condition fails: the replay is shown, never judged, and only its readout is absent. Without sets.json the 5 Sets conditions, `held-out briefs undetermined` and `sets.json = step0.select` fail, and item 8's readout is absent. Without clips.json `clips.palette.sha256`, `clip ids` and `sets.json = step0.select` (select reads its records) fail; beside a step0.json with a render block it is bad input, since render writes clips.json before that block.
- **A run stopped before its first draw** (findings E1, N1): generate.json alone, `generated` 0 and no completions.jsonl, is read with no rows: `stopped` set (a pre-check, the probe, a failed engine load), or `stopped` null with `partial` true (a kill during the engine's load or the probe's call: the watchdog's exit or timeout's TERM records no stop). It reads invalid, naming generate.stopped, or for a kill generate.partial, shown as killed before its first draw, and generate.probe.passed when the probe missed or never ran; §4.2's whole re-draw follows. generate.json alone in any other state is bad input.
- **The conditions** (deviations 28–32): 74 under either Q6 answer, the prompt, palette, constrained and regex conditions counted per stream. A redo pair has 75: its stamp condition is `stamps of the redo pair`, and `the path at the redo commit` is added.
- **What each group adds** (deviations 28–32): Complete holds step0.json's n and requested, and generate.json's requested, to the marked requests. Seeded adds requests.prompt_tokens_mismatch (`lofi_rl.step0.PROMPT_TOKENS`); As registered adds requests.brief_mismatch and palette.sha256 per stream and in clips.json (§4.1's row corrected in place, finding R3-36). Registered first adds registration.mark and, with git, D-023's mark at the generate commit, read as generate reads it, in a DECISIONS.md whose sha256 is registration.decisions_sha256. One stamp reads the path file at the stamp's commit and requires it to be the one generate checked. Reproduced requires each record's canon_sha256 to be score's for its row, and one render_cfg_sha8 a run.
- **At the run's commit** (findings R3-11, N3): check and prereg refuse (exit 2) a checkout where a file that can change a verdict of check or a byte prereg writes differs from its blob at the run's latest stamp: every module check and prereg import, by static analysis (select, requests, the answer constants, the DSL, junk and what it reaches, the features' `SOURCE_GLOBS` and the palette code), and the streams file and the palette, 62 files at this branch on 2026-11-07 (`check.reproduction_files`). The checker's own machinery is not among them, so a fix to it runs anywhere: check.py, prereg.py, path.py (the path file, git's blobs and this list), cli.py and __main__.py. Nor is docs/DECISIONS.md, read with git at the stamps' commits and, for Q8(b)'s answer line, from the checkout. `tests/test_step0_check.py` traces what check and prereg import and read on a valid run, so the list cannot fall behind the code; a closure that cannot be read is bad input, never a pass. Once main has changed one of these files, check and prereg run in a git worktree of the run's commit, and prereg's two files are committed on main from there (finding N9; the Q8(b) bullet above gives the answer line's route).
- **The path file** (deviations 6, 44): it lists the palette's one mode-commit source (the SoundFont, not the LICENSE, README and UPSTREAM beside it that D-020's list held), prompt v2's 9 example sources and the streams file: 86 files, 70 of them the closure, at this branch on 2026-11-07. The closure is the whole package, check.py and prereg.py included, so a laptop fix to either is a listed file that a redo's amendment names.
- **Shown, never judged** (finding R3-43): also the probe's seeds, cell, requests and distinct texts (its `passed` is judged, under Seeded), the engine and `per_request_seed`, and the verbatim and same-clip counts.
- **The amendment line** (deviation 35; findings R3-3, E2, R2-6): one line per redo, `D-023 redo amendment: run step0-<job>; changed <path>, <path>` (or `changed nothing`), read at each later commit. It is read as house style writes it: a path in backticks, a closing period, a bold line or a note after a path read as the bare form, and a bold head is not read. A named file that did not change is fine; one off the path is shown.
- **The redo pair's folders** (deviation 36): generate.json, completions.jsonl and replay.jsonl from OUT; scored.jsonl, step0.json, sets.json and clips.json from REDO, each from OUT when REDO lacks it. Every stamp shows dirty=0, and each later commit holds every blob of the generate commit's path file but the amended files.
- **A score redo** (finding E3; §4.2 corrected in place): score writes step0.json with no render block and no sets.json or clips.json, so a score redo is always followed by its render redo. A pair holding score's records alone reads invalid, its stamp condition saying why.
- **A score that left nothing** (finding R2-7): when generate's records are complete but score or render left no scored.jsonl or step0.json, the run is a scoring defect, D-020's "a crash", redone from its records with score and render `--from OUT`.
- **prereg** (deviation 38; findings R1-2, R3-11, N2): exit 3 for an invalid run, and 2 for bad input, a refusal (a short held-out or Ref_b brief under Q8(b), until D-023's answer line for the run; a checkout off the run's commit) or a differing existing file, neither then written. Without `--write` it prints both files' size and sha256.
- **heldout_v1.json** (deviations 37, 39, 40; findings R1-6, R3-4, R3-6, N2): `tuples` holds the briefs with all 4 clips, and `short` a short brief's partial clips with its reasons: held-out data, in no tuple, as a registered line says. It adds `rule` (name, mark, m, the answers, Q8(b)'s answer line when applied), `counts` and `settings.render_cfg_sha8`; its rule sentence is built from Q3 and Q4 as answered; paths read runs/<run>/summary/<file>.
- **Item 8** (deviation 37): on tuples only. "Differ in A or B" is read as the two clips' A lines, or B lines, differing as text in junk's copy form. J5 is a set measure: two clips one kick cell apart can read 1.0.
- **step0_v1.json** (deviation 41; findings R1-4, R3-10): each file record carries `read`; `readouts` is step0.json's per-set block; each Ref_b record carries its brief_id and cell; pool entries carry no reasons; the flags are one record a brief. `shared` lists every train row with a held-out clip's text, passing or not, as the flags count them, and the replay's `stopped` is cut to its code.
- **The layout** (deviation 42): `brief_streams.text`'s, plus one rule: a list of objects puts each object on one line. Written again from the same run at the same code it is byte-identical; check and prereg run at the run's commit.

**§5, separation and privacy**
- **The RM loader's ask** (findings R3-4, R3-50, R3-51): under Q5(a) or (b), every label on a clip whose canon_sha256 equals that of any clip heldout_v1.json lists, in `tuples` or in `short`, in any cell, is dropped from training, from any source.
- **No text in a committed file** (finding R3-10): `tests/test_step0_prereg.py` reads each committed prereg file, once there is one, for ASCII in prereg's own layout, no newline or LoopScript statement head in any string, and a texts pin under the ignored runs/. `tests/test_privacy.py` predates this unit and is unchanged.

**§6, tests and the order of work**
- **The helpers** (deviations 13, 27, 45): `tests/helpers_step0.py` writes synthetic runs: real G2 and G6 failures only; every passing loop a near-copy of its cell's example, with A and B verbatim; stand-in timings and engine info; the second and third twin train briefs given their own default draws, so no clip is in two briefs. `tests/helpers_step0_engine.py` holds the fake engine and the pre-check tree, `tests/helpers_step0_laptop.py` a temporary git repo and runs stamped with its commits.
- **The end-to-end test**: `tests/test_step0_e2e.py` runs generate, score and render as the job does, from a git-archive snapshot of a temporary repo, then check and prereg on the files they wrote.
- **test_step0_docs** (deviation 43; findings R3-8, R2-2, R1-3): the spec half also maps §4.1's vocabulary to check's conditions, and reads the job's #SBATCH lines, timeouts, walltime and Cost line. The D-023 half is one test, the expected failure until D-023 is written: the unfolded entry's mark and answers line as generate reads them, m in each set's bullet, 74 and 75 conditions, the amendment form, the separation rules, the RM loader's `short`, and no placeholder or comment. A stand-in entry passes and nine faulty forms fail.
- **The pre-flight** (findings R2-4, R3-9, N5): from `path --write` until step0_v1.json exists, `pytest -q` checks in this checkout that every listed file has its blob (a redo's amended files aside) and that D-023's mark reads as MARK. `PYTHONPATH=src .venv/bin/python -m lofi_rl.step0 path --check` is the same blob check, run by hand with that test before `make sync`, as §6.2 step 7 says (corrected in place).

**§8, D-023**
- **What the D-023 draft carries beyond §8's list** (findings R1-3, R3-2, R3-11, E3, R2-7, N2): the answers line beside his answers; §5's separation rules as a bullet of their own; the amendment line's form; under Q8(b), the short-answer line's form; check and prereg run at the run's commit; a score redo is followed by its render redo; and a score that left no records is a scoring defect.
- **Asks added** (findings R3-1, R3-52, R3-5): the pool's and Ref_b's later renders pass every scored row and this run's clips.json records with their own (select refuses otherwise), and stop on a recomputed pick that differs from step0_v1.json's. Outside the held-out set's own rounds and their repeats, base-model clips reach a round or the triage of 300 only as step0_v1.json's pool picks. The queue job compares a re-render as §3 says.
- **The CHANGELOG** (finding R3-12; not on this branch, which leaves the CHANGELOG to the unit's commit): that entry is to carry a Next list naming the asks of later units (the RM loader, the queue, the later renders, the copy-pool builder, the seed guard) and D-023, since a session reads the CHANGELOG first.

**§9, Claude's defaults added**
- The pool's and the spares' rule by clip across twin train briefs (§1.6); the probe at two seeds (§2.3); one bounded retry of the records' copy (§2.7).
