# Run 1's start, registered again: draft D-021 (PROPOSED, awaiting Shresth)

**Status:** a proposal. Nothing in it runs before Shresth answers. On his yes it moves to `docs/DECISIONS.md`
as D-021 with his words, and its laptop work starts. On his no, Part B (the SFT warm-start) is registered in full
and comes to him first.

**How it was drafted** (workflow `wf_76128e23-f1d`, 2026-11-01):
- Three independent proposals, one per angle: a self-distilled SFT warm-start (`rft`), a curated SFT warm-start
  (`curated`) and a masked trainer (`masked`). They are in `docs/design/run1_start_proposals.md`.
- A judge scored them 6 / 6.2 / 6.7 (0-10, mean of six criteria) and
  synthesized: the masked trainer first, with the SFT warm-start as its registered fallback.
- A pre-registration referee listed 8 major and 13 minor problems with the judge's draft and revised it.
  The text below is the revised entry.

**Numbers Claude re-read from the files before sending this** (2026-11-01):
- `runs/grpo-smoke-13657200/summary/*/steps.jsonl` (grammar): step-1 gap 0.1146 at learning rate 0.0; per-step gap
  0.1050-0.2571; per-step mean IS weight 0.0010-0.0814.
- `runs/grpo-smoke-13632679/summary/*/steps.jsonl` (no grammar): step-1 gap 0.0087; per-step gap 0.0044-0.0091;
  per-step mean IS weight 0.6981-1.0272.
- `runs/junk-13656050/summary/junk.json` (v2, no grammar): G1 397, G2 0, G5 0; pass 78, pass_not_copy 64.
- `runs/junk-13640325/summary/junk.json` (arm B, v2 + grammar): `verbatim_passing` A 415, B 398, AB 392 (so A or B
  in 421); `grid_share_passing` 7,287 of 8,319; `knobs_passing` energy_match 498, f_mood 333.
- Arm B's nearest same-cell neighbour (5-gram Jaccard of the canonical text without `sax:`, recomputed from
  `completions.jsonl`): median 0.9377, with 448 of 500 at 0.8 or more.
- The entry's other laptop counts are the drafting agents'; they are re-read before the entry is decided.

---

## D-021 (2026-11-01, PROPOSED, awaiting Shresth): run 1's start, registered again: keep the grammar by training the grammar-masked policy, decided by one smoke; the SFT warm-start only if that fails

**Decided by:** Shresth, pending; proposed by Claude.
- Nothing here runs before his answer. The entry is committed, with his words, before its first job.
- D-020's IS verdict sent run 1's start to a later entry. The CHANGELOG (2026-11-01, 19:30) promised him the SFT warm-start draft before any job.
- This entry puts one cheaper step in front of that draft. It keeps the draft as the fallback (Part B), and keeps D-016's order after it.
- **What this entry supersedes, once he says yes.** Entries are never edited; a later one supersedes.
  - D-020's IS-half verdict ("the grammar goes off for GRPO run 1"), but only if Part A passes. If Part A misses, that verdict stands.
  - D-020's Before-GRPO clause that scores the copy penalty "against the seed pool". Item 2 narrows the pool.
  - taste_pipeline §3's held-out seed (596) and its `data/briefs/briefs_v1.jsonl` split. Item 1 replaces both. `data/` is never committed, so that file could not hold a pre-registration.

- **Why**
  - **D-020's two readings for v2.**
    - Without the grammar, G1 + G2 + G5 = 397 of 500 (`runs/junk-13656050/summary/junk.json`). That keeps the grammar on.
    - With it, the GRPO smoke's mean |log IS ratio| was 0.1538 (`runs/grpo-smoke-13657200/summary/smoke.json`), above 0.05. That turned it off.
  - **The gap comes from the mask alone.**
    - Step 1's rollouts come before any update, and `learning_rate` is 0.0 at step 1 (`a/steps.jsonl`). So vLLM and the trainer held the same weights.
    - The gap there was 0.1146. At step 1 of the unconstrained smoke 13632679, it was 0.0087.
    - So vLLM's log-probs carry the grammar's mask, and the trainer's do not. This conclusion rests on those two step-1 gaps.
    - 13657200's `run.json` `vllm_logprobs` only lists lines. `apply_grammar_bitmask` is at `gpu_model_runner.py:4335`. `_sample` is at 3437, with its sampler call at 3448. `processed_logprobs` is at `sampler.py:270–273`, and TRL asks for it at `vllm_generation.py:370`.
    - The record says `checked: false` and does not show the call order. V1 and M10 check it.
  - **What that did to training.** TRL 1.17.1 weights each sequence by exp(Σ old − sampling) and masks weights above 3.0 (`a/run.json`: `vllm_importance_sampling_correction` true, `sequence_mask`, `clip_max` 3.0). The exact form is UNVERIFIED until check T3. From `{a,b}/steps.jsonl` of both smokes:

    | per step | with the grammar (13657200) | without (13632679) |
    |---|---|---|
    | mean IS weight | 0.0010–0.0814 | 0.698–1.027 |
    | max IS weight ÷ (32 × mean) | 0.319–1.000, and ≥ 0.9 in 7 of 15 steps | 0.061–0.112 |
    | grad norm | 0.00088–0.062 | 0.059–0.221 |
    | IS gap | 0.105–0.257 | 0.0044–0.0091 |

    - The second row is the largest weight's share of the step's 32, if TRL's logged mean is per sequence. T3 checks this; no value exceeds 1.000.
    - Read that way, one completion carried at least 90 % of the weight in 7 of the 15 grammar steps, and 32–87 % in the other 8.
  - **The cause can be removed.**
    - If the trainer masks its own logits with the same grammar, π_trainer is the masked distribution vLLM samples from. GRPO is then on-policy for the policy that is played.
    - taste_pipeline §6's rule exists to catch exactly this ("the masked sampler mismatches the trainer").
    - This entry changes the trainer, not the rule:
      - D-020's definition and its 0.05 threshold are kept;
      - the reading is registered before it is taken;
      - a measured miss leaves D-020's verdict in force.
    - What differs from 13657200 (CLAUDE.md §6): the trainer masks its logits in all three passes.
  - **The order (D-016):** grammar first, then the SFT warm-start, then edit-mode rollouts, then stop.
    - This entry stays on lever 1, registers lever 2 as the fallback, and names lever 3 after it.
    - Trying lever 1 first costs the SFT path no time.
      - An SFT checkpoint that can start run 1 must be built with the frozen palette's names. So none can exist before the freeze, which is planned for Nov 6 (docs/SHOOTOUTS.md).
      - The 30 drafts fullmatch `draft_day2`'s regex 30 of 30, and dev_gu's only 3 of 30 (laptop count).
    - The masked smoke can run now on dev_gu.
  - **Why not an SFT first.**
    - **A self-distilled corpus would teach copying at the part level.**
      - Arm B of D-020 has 445 candidates: passing, no sax fold, J_any < 0.8.
      - Of these, 367 carry the shown example's A or B line verbatim (339 both), and 86.6 % of their grids are the example's (laptop count over `runs/junk-13640325/summary/scored.jsonl`).
      - Only 78 of the 500 copy neither line and sit under J_any 0.8, and 50 of those are in one cell, mid/wistful.
    - **A Claude-written corpus avoids that.** But it needs about 180 new loops under global constraints, and the model risks memorizing about 21 targets per prompt.
    - **Both redefine step 0** for D-009 and D-014, which he registered when step 0 was the base model.
  - **What this step 0 is.** These are laptop counts over arm B (`runs/junk-13640325/summary/`). He reads them before he answers.
    - **It copies the shown example's sax lines.**
      - A or B is verbatim in 421 of 500 loops, and both in 392.
      - 7,287 of 8,319 grids are the example's.
      - pass_not_copy (446) counts only whole-loop near-copies, so these loops pass.
    - **It is already near-collapsed inside a cell.** The nearest same-cell neighbour sits at J5 median 0.9377, with 448 of 500 at 0.8 or more.
    - **A cell is one prompt.**
      - Every brief of a cell gets the same messages. So "different briefs should give different songs" (TASTE.md, Variation) rests on sampling alone.
      - Nothing in the reward sees two briefs of one cell together: P_group sees one brief's 8, and item 2 keeps model output out of the pool.
    - **D-009 cannot see a collapse inside a cell.**
      - Suppose every cell's loops were replaced by that cell's step-0 medoid. The mean 1 − J5 across 100 random arm-B briefs would still keep 0.938–0.963 of its step-0 value (200 draws).
      - Only 11.1 % of the pairs share a cell, and D-009's bar is 0.70.
      - This is a symbolic proxy for D-009's embedding distance (an estimate). Audio also differs by tempo and tonic, which would hide a collapse further.
    - **What this entry does about it:** items 3–8 add the readouts and alarms this calls for. D-009, D-014 and the obedience claim are his, and stay as registered.
  - **Data** (laptop counts, 2026-11-01):
    - `runs/junk-13640325/summary` (arm B: v2 + grammar, dev_gu), `runs/junk-13656050/summary` (v2 without the grammar);
    - `runs/grpo-smoke-13657200/summary` and `runs/grpo-smoke-13632679/summary`;
    - `drafts/v1/`.
    - Raw completions in the regex's language: 26 of 500 without the grammar, 500 of 500 with it.

- **Part A: the masked trainer** (`src/lofi_rl/rl/`; docs/GRPO.md gains a section)
  - **The config key:** `trainer_logprobs: grammar_masked`. It is refused without `structured_outputs`, and the dry run and `run.json` record it.
  - **The override:** `LofiGRPOTrainer(GRPOTrainer)` overrides one method, `_get_per_token_logps_and_entropies`.
    - In every forward of the policy, the logits at completion positions get a large finite negative outside the grammar's allowed set, written out of place. That covers three passes:
      - the old pass, which feeds the IS weight and D-020's metric;
      - the reference pass, with the adapter off;
      - the loss pass.
    - Prompt rows, rows after the first EOS or pad, and the extra last row are left alone.
    - The fill is finite, not −inf, so TRL's entropy (Σ p·log p) stays finite.
    - **T2's alternative.** T2 may show that TRL's own `sampling_mask` is a keep-mask applied before the log-softmax in all three passes. If so, the grammar mask is ANDed into it instead.
      - `run.json` names the path that ran.
      - M1's tests cover both paths.
    - It also logs the step's 32 per-sequence IS weights, from the old and sampling log-probs (`is/seq_weights`), for M11.
  - **The mask is xgrammar's own** (0.2.8, `requirements/policy.lock.txt`).
    - A `GrammarMatcher` is compiled from the same regex string. Its `TokenizerInfo` is built the way vLLM 0.32's backend builds it: vocabulary size and stop ids, read at run time from the installed `v1/structured_output/backend_xgrammar.py`.
    - The matcher is replayed along each completion's sampled ids, so the trainer applies the bitmask vLLM applied, by construction.
    - The API names are UNVERIFIED until that run-time read.
  - **An independent check:** `src/lofi_rl/grammar_mask.py`, standard library only. It is a Thompson NFA of the regex over the byte-level vocabulary, with allowed sets memoized by state set.
    - **On the cluster**, it is built from the pinned snapshot's `tokenizer.json`, with the vocabulary size and stop ids that V2 reads. So it and xgrammar see one vocabulary.
    - **On the laptop** (2026-11-01, with the cached `mlx-community/Qwen3-4B-4bit` `tokenizer.json`, as in GRAMMAR.md §3.3), it accepts every token of the canonical re-encodings of all 980 completions of arm B and smoke 13657200. There are 0 violations.
    - These laptop numbers estimate xgrammar's bitmask; they are not it.
      - At most 52 tokens are allowed at any position (median 20–21), and 7.3–7.5 % of positions are forced.
      - A cold 32-completion step takes 0.13 s on the laptop, and a warm one 0.013 s.
    - **Canonical re-encoding** gives vLLM's token count for only 361 of the 500 arm-B completions, and for 227 of 320 and 107 of 160 in the smoke's two launches. So the sampled ids themselves are checked on the cluster (M8).
  - **A sampled token outside the mask** stops the launch before the loss sees it.
  - **Source checks before the model loads** (`trl_probe`, AST reads). They fail closed with exit 2, after about 2 minutes.
    - T1: the old, reference and loss log-probs come only from `self._get_per_token_logps_and_entropies` (`use_liger_kernel` false).
    - T2: that method's body. It runs one forward per micro-batch, keeps rows in order, takes `logits_to_keep + 1` and drops the last row, then applies the temperature and the log-softmax. T2 also reads what `sampling_mask` is, and checks that it is None at top_k 0 and top_p 1.
    - T3: the sequence_mask weight, and whether `sampling/importance_sampling_ratio/mean` is a mean over sequences or over tokens.
    - T4: the KL comes from the reference and current log-probs; `use_bias_correction_kl` is recorded.
    - T5: lm_head is not a LoRA target.
    - V1: in the method that holds `gpu_model_runner.py:4335`, `apply_grammar_bitmask` changes the logits before `self._sample` runs. Also, `_sample` reaches the `processed_logprobs` branch at `sampler.py:270–273`, which TRL asks for. V1 is recorded, and M10 is its test.
    - V2: how `backend_xgrammar.py` builds its `TokenizerInfo` (vocabulary size and stop ids) and compiles the regex.
  - **Per-step records:**
    - `mask/violations` and `mask/rows`;
    - `mask/neglogz`, the token-weighted −log Z in the old pass;
    - `mask/forced_share` and `mask/nfa_mismatch`;
    - `is/seq_weights`.
  - **The smoke:** `configs/grpo_smoke_v2_grammar_masked.yaml`, which is `configs/grpo_smoke_v2_grammar.yaml` plus that one key (`tests/test_rl_train.py` pins it).
    - It runs on the unchanged `jobs/grpo_smoke.sbatch`. That job takes any `configs/grpo_smoke*.yaml`, so its registered sha `1dded12d386c7fab` stands.
    - It runs on an L40S, as 13657200 did, so the numerics compare.
    - Its code path is frozen in `docs/prereg/D-021_masked_smoke_path.txt`, with `.sync-freeze` set until it ends.

- **Settings:** D-020's run-1 settings, plus `trainer_logprobs: grammar_masked`.
  - Model: Qwen3-4B-Instruct-2507 @`cdbee75f`; LoRA r 16, α 32, dropout 0, all-linear.
  - Optimizer: lr 1e-5, constant after 3 warmup steps.
  - Batch: 4 prompts × 8 completions; `num_iterations` 1; `max_completion_length` 512 with `mask_truncated_completions`.
  - Loss: β 0.04 to the adapter-off base, under the same mask; dapo; `scale_rewards` batch.
  - Sampling: T 1, top_p 1, top_k 0; TRL's IS correction at its default.
  - Prompt: v2 (template `127e3094…`).
    - D-020's rule holds: if the frozen palette refuses a v2 example, the prompt is registered first.
    - `draft_day2` refuses none of the 9 (laptop check).
  - Grammar: `loopscript_regex(palette)` with xgrammar pinned. The smoke uses dev_gu's `1412f11f…`; run 1 uses the frozen `lofi_rain_v1`'s own sha.
  - P_copy: against the 9 v2 references, plus the pool in item 2.

- **For whichever part runs** (grafted from the two SFT drafts)
  1. **Brief streams, fixed now** in `docs/prereg/brief_streams_v1.json`, committed with this entry before any step-0 sample is drawn.
     - **Training:** seed 596 from index 0. Run 1 takes [0, 400).
     - **Measurement:** seed 2026, [0, 500) (junk-rate).
     - **Evaluation:**
       - RM held-out: 40 briefs from seed 5101;
       - dev: 40 from seed 5102;
       - the final A/B: 100 from seed 5103;
       - Part B's dev set: 270 from seed 5104.
       - Seeds 5101–5104 appear nowhere in the repo today (laptop grep).
     - **Stratified by cell.** Each stream takes the first briefs of each cell in stream order, up to per-cell counts the file fixes: 4 or 5 a cell for 40, 11 or 12 for 100, and 30 for 270.
     - **Draw seeds.** The file also fixes the vLLM seed of the k-th draw for brief i of each stream. That covers the step-0 samples, the checkpoint quads and the A/B's ≤ 4 draws.
     - **What "held-out" means.** The model sees only the cell, so all 9 prompts are the same in every stream. Held-out briefs are kept apart only by fresh samples, rendered at tempi and tonics the model never reads.
     - **Twins.** In stream order, some evaluation briefs have the same energy, mood, bpm and tonic as a brief in seed 596's [0, 400) (laptop count):
       - 8 of seed 5101's first 40;
       - 4 of 5102's first 40;
       - 15 of 5103's first 100.
       - The file lists them for the RM's report and excludes none. The policy never reads bpm or tonic, and an exclusion would thin the 75–90 BPM centre.
  2. **The copy penalty's pool holds only loops the policy did not write:** the 9 v2 references, the 30 drafts, and the seed pool's Claude-written loops, mutations and broken copies.
     - It never holds the 660 step-0 samples or any other model output.
     - Why: at step 0, the nearest same-cell neighbour has J at median 0.9377, and 448 of 500 sit at J ≥ 0.8 (laptop count over arm B). A pool holding step-0 samples would put P_copy at its 0.3 cap on nearly every completion.
     - This supersedes D-020's "scored against the seed pool", and narrows taste_pipeline §5's "vs pool".
  3. **Part-level copy readouts, per step,** on the GRPO path: the share of completions whose A or B line equals that line of the shown example, and the share of grids that are the example's (junk.py's `verbatim` and `grid_share`).
     - At step 0 under the grammar (arm B): A or B verbatim in 421 of 500, and 7,287 of 8,319 grids (`scored.jsonl`, `junk.json`).
     - Whole-loop J cannot see a copied sax line over new drums, and that path is already in step 0: 8 of the 52 loops at J5 ≤ 0.6 (P_copy 0) carry a verbatim A or B line (laptop count).
     - **Alarm** (hack-watch, taste_pipeline §7). It fires when, for 3 consecutive 5-step windows, both hold:
       - the verbatim A-or-B share stays at or above its step-0 value;
       - the mean J5 with the shown example falls at least 0.1 below its step-0 value.
       - That is P_copy evaded with the lines kept. For scale, arm B's values are 0.842 and 0.7115.
  4. **Diversity readouts,** in junk.json, in smoke.json, and per 5-step window in run 1. There are two copy forms, named once here and used everywhere in this entry. Arm B's values (laptop count):
     - **junk's form** (canonical text without `sax:`):
       - cell-balanced within-cell mean 1 − J5: 0.2263;
       - pair-weighted mean J5: 0.7743;
       - extra exact duplicates within cells: 34 (466 distinct of 500).
     - **the K/D form** (DRAFTS.md §5: keys and kit written K and D):
       - cell-balanced within-cell mean 1 − J5: 0.2158;
       - pair-weighted mean J5: 0.7839;
       - extra exact duplicates within cells: 44 (456 distinct, 0.912).
     - The K/D form ignores instrument names, so it is the one compared across palettes: in B4, and in run 1 against step 0.
     - In run 1, a window's value compares completions of different briefs of one cell.
  5. **Step-0 values.** Items 3, 4 and 6 read their step-0 values from the official Phase-1 gate run on the frozen palette (Part A) or from B2's run (Part B), with the same seed and settings. Arm B's dev_gu values are for scale only.
  6. **Within-cell collapse alarm** (hack-watch, taste_pipeline §7): the window's within-cell mean J5 across briefs (K/D form) stays above its step-0 value for 3 consecutive 5-step windows.
  7. **Run-1 STOP rules, and what follows a STOP.**
     - **The rules:** any mask violation; a per-step gap above 0.05; an IS-weight mean below 0.3 for 3 steps in a row.
     - **A STOP** ends run 1 at its last checkpoint (one every 25 steps). Nothing resumes until an amendment here names the cause and the fix.
     - **Any of these rules marks a trainer defect.**
       - The masked smoke runs on the fix (at most 500 SU) and must meet M5–M11.
       - Run 1 then resumes from the last checkpoint before the first bad step.
       - If the smoke misses, D-020's verdict applies to run 1, and Part B goes to Shresth.
     - **An alarm of items 3 or 6** follows taste_pipeline §7's On-alarm list.
  8. **Readouts beside the registered gates and claims.** These are reported and never gated. D-008, D-009, D-014 and the obedience claim stay as he registered them.
     - **Beside D-008's gate:**
       - each held-out tuple's pairwise J5 and verbatim-line sharing;
       - agreement on the pairs whose two clips differ in A or B;
       - under Part B, agreement without the clips at J5 ≥ 0.8 to a training target.
       - Under this step 0, about 56 % of held-out tuples would carry both of the shown example's sax lines in all four clips (20,000 random within-cell 4-tuples of arm B; laptop count, an estimate).
     - **Beside D-009's number:** the within-cell mean 1 − J5 (K/D form) and the distinct forms per cell, at step N and at step 0, on the 100 final briefs.
     - **Beside D-014's A/B:**
       - each trial's copy readouts for both arms: J5 to the shown example, to the pool and to any training target, and verbatim A/B;
       - the win rate on the trials whose step-0 clip carries no verbatim line of its shown example;
       - a trial order in which no two consecutive trials share a cell.
     - **Beside the obedience claim:** energy and mood obedience separately, with step 0's values on the same briefs. For scale, arm B on dev_gu matched the energy band in 498 of 500 loops and the mood signature in 333 of 500.

- **Pre-registered gates for Part A**
  - **Two kinds of miss.**
    - **An invalid launch** (M5–M8, or M10's first clause) measured nothing. Its defect is fixed, the fix is recorded here, and the smoke runs again. A launch invalid twice goes to Shresth, as D-020's invalid arms did.
    - **A measured miss** (M9, M10's ratio or M11) on a valid launch gets at most one re-run, with the one fix named now.
      - That fix is the masked log-softmax computed in fp32 inside the override, in all three passes. vLLM's sampler takes its log-probs in fp32.
      - It applies only when M8 holds, so the mask matched and the miss is numerical.
    - **Not named:** `cast_lm_head_to_fp32`. TRL documents it for untied embeddings only (UNVERIFIED here: there is no TRL source on the laptop), and the model's config has `tie_word_embeddings: true` (the cached configs).
  - **M1 (laptop):** `pytest -q` passes on the commit that holds this entry. That includes the new `tests/test_grammar_mask.py` and `tests/test_rl_masked_logps.py`, which cover both T2 paths.
  - **M2 (laptop):** the NFA accepts every token of the 980 re-encoded completions above, with EOS at each end: 0 violations. This checks the NFA on re-encodings only; M8 checks the sampled ids.
  - **M3 (laptop):** at GRAMMAR.md §3.3's four points, the allowed sets are 20 with EOS refused, 20 with EOS allowed, 25, and 8.
  - **M4 (laptop):** the dry run reports regex `1412f11f…`, template `127e3094…` and reward check 0.2784511276657195, as `runs/grpo-smoke-13657200/summary/dry_run.json` does. It also reports `trainer_logprobs: grammar_masked`.
  - **M5 (registration, validity):** this entry and the frozen path are committed and synced before the submit. Both launches' stamps show that commit with dirty=0.
  - **M6 (validity,** `smoke.json`):
    - GRPO_SMOKE_OK;
    - in `in_language`: stop out_of_language 0, other.n 0, and at least one stop completion per launch;
    - xgrammar pinned and applied in both launches.
  - **M7 (validity, sources):** T1–T5 read `match` in both launches' `run.json`; otherwise the launch stops before the model loads. V1 and V2 are recorded.
  - **M8 (validity, fidelity):**
    - 0 sampled tokens outside the trainer's mask, over every completion token of both launches;
    - 96 masked rows per step: 32 old, 32 reference, 32 loss;
    - 0 positions where the NFA's allowed set differs from xgrammar's bitmask, over one vocabulary and one set of stop ids;
    - every logged loss, KL and entropy finite;
    - `run.json` names the T2 path that ran.
  - **M9 (measured): D-020's quantity, with its threshold unchanged.**
    - The token-weighted mean, over every step of both launches, of TRL's `sampling/sampling_logp_difference/mean` (source verdict `match`) is at most 0.05.
    - For reference, 13657200 read 0.1538, and 13632679 read 0.0044–0.0091 per step.
  - **M10: removed rather than reduced.**
    - **Validity:** the trainer's own token-weighted −log Z (`mask/neglogz`) is at least 0.05. 13657200's gap, 0.1538, implies a −log Z of about that size, so a value under 0.05 means the records disagree with that run.
    - **Measured:** M9's value is at most 0.25 times `mask/neglogz`. If vLLM's log-probs came before the mask, the gap would equal −log Z and this would fail. So M10 also settles D-020's caveat.
  - **M11 (measured): the gradient survives TRL's correction** (from `is/seq_weights`).
    - At every step, the mean of the 32 weights is at least 0.3, and the largest is at most 0.25 of their sum.
    - TRL's `sampling/importance_sampling_ratio/{mean,max}` stands in only if T3 shows its mean is over sequences.
  - **M12 (cost; it decides only the run-1 estimate):** mean `step_s` is at most 43.3 s (1.15 × 13657200's 37.63 s). Otherwise the lofi-grpo estimate is redone before the ask. The job's SU is read from `sacct`.

- **What follows**
  - **M5–M11 all pass:** this entry supersedes D-020's IS-half verdict. The grammar is on for run 1 with the masked trainer, and D-020's Outcome (v2 with the grammar) is run 1's setting again. Then, in order:
    1. **D-020's official Phase-1 gate** on the frozen palette: `lofi-junk-rate` with `ARGS='500 2026 1 v2'`.
       - It needs `pass_not_copy.count` ≥ 300 of 500 in a valid run, and at least 15 of 50 clips "fine" by his ear, with his APPROVED line in `gates/GATE_phase1.md`. The bar is D-020's and unchanged.
       - Before he signs, the gate file reports three things beside the bar:
         - the verbatim A-or-B share among the passing non-copies;
         - the same share among the 50 clips;
         - his "fine" count, split by whether a clip carries a verbatim line.
       - This run's readouts become the step-0 values of items 3, 4 and 6.
    2. **The step-0 samples,** all drawn under the grammar, with item 1's streams and draw seeds: the RM held-out tuples, PAR's Ref_b and the seed pool's 660.
    3. **Phase 2's GRPO smoke** (CLAUDE.md §9) meets M5–M11 under the same two-kind rule. It runs with the scorer on, the frozen palette's grammar and run 1's pool.
    4. **Run 1's code path** is frozen as the smokes' paths were, or snapshotted per job (CHANGELOG Status). Then `lofi-grpo` goes to him for his yes.
  - **An invalid launch, twice:** it goes to Shresth with the records. Nothing else on this path runs until he answers.
  - **A measured miss after its one re-run:** D-020's verdict stands, the grammar is off for run 1, and Part B goes to Shresth.
  - **The Phase-1 gate fails on the frozen palette:** it goes to Shresth with CLAUDE.md §9's next fallback, Part B.
  - **Part B fails:** see Part B.

- **Part B: the SFT warm-start** (D-016's lever 2). It runs only if Part A or the Phase-1 gate fails, or if he declines Part A. It is registered in full as an amendment here before any of its jobs.
  - **The corpus is Claude-written. It never holds model output, nor a loop he voted down.**
    - It holds the 10 kept drafts (d01, d02, d03, d07, d09, d04, d06, d08, d05 and d10; CHANGELOG 2026-11-01 13:38), plus 180 new loops, 20 per cell.
    - The 20 rejected drafts stay out of it. They are labeled negatives (TASTE.md, Labeling; taste_pipeline §8: each kept loop ≻ its rejects). They stay in the copy pool and in the RM's labels.
    - The new loops are written as the first seed-pool shards, which Phase 2 needs either way (taste_pipeline §8, D-018; half the writers see TASTE.md).
    - Admission rules:
      - zero repairs;
      - J5 < 0.6 against the 10 references (the 9 v2 examples plus L5, DRAFTS.md §5) and against every other target;
      - no A or B line equal to a reference's or another target's;
      - written in the model's own line-end layout (GRAMMAR.md §3.3).
    - At least 21 targets per cell (22 in low/wistful, which has two draft briefs).
  - **Training:**
    - LoRA r 16, α 32, all-linear, on the exact v2 messages of each target's cell. The loss is on the completion only, up to the end-of-turn token.
    - lr 1e-4, constant after 5 warmup steps; up to 3 epochs, with the adapter saved after each.
    - **The dev bars** are read inside `lofi-sft`, with vLLM and each saved adapter, on 270 completions without the grammar from seed 5104:
      - G1 + G2 + G5 ≤ 27;
      - pass_not_copy ≥ 189;
      - near-copies of a training target ≤ 27.
    - The earliest save that passes the dev bars is merged: an fp32 merge, one bf16 copy and a sha256 manifest. If no save passes, nothing is merged and Part B fails.
  - **Gates on the merged model** (seed 2026 × 500, no grammar, the frozen palette):
    - B1: G1 + G2 + G5 ≤ 100, so the grammar stays off.
    - B2: pass_not_copy ≥ 300, the Phase-1 bar on run 1's own settings.
    - B3, copying:
      - near-copies of any target ≤ 50;
      - completions with an A or B line equal to a target's or a reference's ≤ 125;
      - mean P_copy ≤ 0.0877, half of arm B's 0.1754.
    - B4: diversity no worse than arm B's in the K/D form (item 4): within-cell pair-weighted mean J5 ≤ 0.7839, and distinct forms ÷ passing ≥ 0.912.
    - B5: a GRPO smoke without the grammar meets M9 and M11.
    - B6: at least 15 of 50 clips "fine". The 50 are drawn from loops that pass and are near-copies (J5 ≥ 0.8) of neither the shown example nor any training target.
  - **A miss:** each job gets at most one re-run, after a fix recorded here. Part B fails if no save passes the dev bars, or if any of B1–B6 still fails.
  - **Part B fails:** D-016's lever 3, edit-mode rollouts, is drafted and taken to Shresth before any job. If that fails too, D-016's last step, stop, goes to him.
  - **Step 0** becomes the merged checkpoint, sampled without the grammar. That needs his word at that point (D-009, D-014).
  - **Its jobs** are each at most 500 SU, so Claude registers them:
    - `lofi-sft`: A100, 4 CPU, 32 GB, 25 min (20 SU/min, at most 500 SU), covering training, the dev bars and the merge;
    - `lofi-junk-rate` with a MODEL argument, re-registered at most 450;
    - the registered smoke.

- **Step 0 under Part A**
  - It is the base model (Qwen3-4B-Instruct-2507 @`cdbee75f`, no adapter) with prompt v2. vLLM decodes it under the frozen palette's grammar at T 1, top_p 1, top_k 0 and 512 tokens.
  - That is D-020's own Outcome. Arm B measured it on dev_gu: 500 of 500 pass, and 446 of 500 pass_not_copy.
  - What it copies, and what that means for D-009 and D-014, is under "What this step 0 is".
  - Under the mask, it is also the KL reference.
  - Every checkpoint is decoded through that same grammar. That covers:
    - the dev briefs that pick step N;
    - the 100 final briefs;
    - the obedience check;
    - the across-brief no-collapse distance;
    - the ♥ secondary.
    - Nothing unmasked is ever played to him.

- **Jobs and SU** (CLAUDE.md §4 rates; 1,481.9 SU of the 1,000,000 target used so far, CHANGELOG Status)
  - **The masked smoke:** `lofi-grpo-smoke`, already registered (sha `1dded12d386c7fab`, max 500).
    - Command: `make submit JOB=jobs/grpo_smoke.sbatch OVR='--gres=gpu:l40s:1 --constraint=l40s' ARGS='configs/grpo_smoke_v2_grammar_masked.yaml'`.
    - Shape: L40S, 4 CPU, 32 GB, 25 min, at 20 SU/min, so at most 500 SU a launch.
    - Expected: about 250. 13657200 cost 243.0 SU in 12:09, and the mask adds under 1 s a step (an estimate; the laptop NFA takes 0.13 s for a cold step).
    - At most three launches: the first, one after an invalid launch, and one after a measured miss. That is 1,500 SU at worst.
  - **The Phase-1 gate:** `lofi-junk-rate`, registered (sha `4a9d8685231fd573`, max 450). Its measured runs cost 80.7–99.0 SU, and D-020 already requires it.
  - **Phase 2's GRPO smoke,** with the scorer on: taste_pipeline §6 puts it at about 550 SU on an A100, or 670 on an A40 (UNVERIFIED). That is over 500, so its shape goes to him with the lofi-grpo ask.
  - **PAR's Ref_b** for run 1's 400 briefs: 1,600 step-0 samples under the grammar, rendered and featurized. No job type or estimate exists yet; it goes into the lofi-grpo ask.
  - **Run 1:** `lofi-grpo`, a new type over 500 SU, for his yes later.
    - taste_pipeline §6's estimates (UNVERIFIED):
      - A100: about 2,300 ± 35 %, worst case 3,780 at 1:45;
      - L40S: about 3,200 ± 35 %, worst case 5,400 at 2:30.
    - **From measured parts** (estimates):
      - 13657200's L40S step was 37.63 s, with the symbolic reward.
      - The CHANGELOG (2026-11-01, 15:05) sizes the scorer for 32 passing completions at about 12 s of rendering on 7 workers, plus about 7 s of features. §6 budgets 9 s.
      - That makes about 57 s a step. At 36 SU/min, 100 steps plus 7 min of startup come to about 3,700 SU on an L40S, and about 2,900 on an A100 by the same arithmetic.
      - Both are inside §6's worst cases. M12 and Phase 2's smoke replace these numbers before the ask.
  - **Part B, only if needed:** at most 1,450 SU, or 2,900 with one re-run of each job; about 590–720 expected (an estimate).
  - **New SU this entry adds before run 1 under Part A:** about 250 expected, 1,500 at most.

- **What Shresth decides**
  - **Now, one yes or no:** Part A first, with Part B only if it fails. Part A means the masked smoke, the grammar kept, and step 0 the base model under the grammar.
  - **What a yes accepts.** He can override each item in his own words.
    - **Step 0 copies the example's lines.** It copies the shown example's sax lines in most loops (A or B in 421 of 500, both in 392; arm B).
      - So the final A/B compares step N with loops that mostly carry Claude's example lines, and the two arms may be told apart by ear.
      - Item 8 reports this. D-014 is unchanged.
    - **"Held-out" means fresh samples.** In D-009, D-014 and the obedience claim, it means fresh samples of the same 9 prompts.
    - **D-009 cannot see a collapse inside a cell.**
      - A total per-cell collapse would keep 0.94–0.96 of step 0's across-brief distance (the proxy above).
      - So items 4, 6 and 8 watch it. Making a within-cell readout a claim would need his word.
    - **The defaults this entry sets:**
      - the narrowed copy pool;
      - the brief streams;
      - the alarms and readouts;
      - Part B's corpus, without the drafts he rejected.
  - **If no:** Part B is registered in full as the next entry and comes to him. Nothing runs.
  - **Later:**
    - `lofi-grpo`'s SU, which is over 500, together with Phase 2's smoke shape and Ref_b's job;
    - only if Part B runs, step 0 as the SFT checkpoint.
