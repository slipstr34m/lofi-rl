# Run 1's start: the three source proposals (workflow `wf_76128e23-f1d`, 2026-11-01)
Read-only drafts, each from one angle, written before the judge saw the others. `docs/design/run1_start.md` holds the synthesized, refereed draft; these are its sources. Their numbers are the drafting agents' laptop counts.

## `masked`: Masked trainer (D-021): keep v2 + the grammar, and train the grammar-masked policy

**Judge's score:** 6.7. Sub-scores (0-10): syntax without the grammar 3; no copying or collapse 6; final claims 9; SU and Shresth's time 9; implementation risk 5; fallback order 8. Mean 6.7.
- Syntax: it does not teach syntax at all. Without the mask, the base model still fails 397/500. Syntax holds by construction instead (arm B passes 500/500 under the grammar), so every checkpoint must be decoded with the grammar forever.
- Copying: it teaches nothing at step 0, so nothing is sharpened. Restoring on-policy weights is what lets P_copy work at all. In 13657200 the per-step mean IS weight was 0.0010-0.0814, and one completion held at least 0.9 of a step's weight in 7 of 15 steps (recomputed from {a,b}/steps.jsonl). So the grammar run mostly learned from one completion per step. Step 0's part-level copying (A or B line verbatim in 421/500) is left to a reward that cannot see it.
- Claims: step 0 = base + v2 + grammar is D-020's own reading, and the closest to D-009 and D-014 as he wrote them. Both arms of every claim are decoded through one grammar.
- Cost: one smoke on an already-registered job type (at most 500 SU, about 250 expected), no new type, and a one-line confirmation from him.
- Risk: new code against unverified TRL 1.17.1 internals (the override, sampling_mask, the three passes). Its mask must equal xgrammar's at every position, including padded vocabulary rows and stop ids. The laptop prototype reproduces: 0 violations over 980 re-encoded completions, at most 52 allowed tokens, 7.3-7.5 % of positions forced. Failure is cheap and fails closed. One gap: an in-place -inf fill can break autograd or make TRL's entropy NaN (0 * -inf); a finite, out-of-place fill avoids both.
- Order: stays on D-016's lever 1, keeps D-020's definition and threshold, is registered before the measurement, and leaves D-020's verdict standing on failure. But it supersedes a recorded verdict and the CHANGELOG's stated next step, and the entry has to say why.

### Summary

Run 1 keeps prompt v2 and the LoopScript regex. D-020's IS mismatch is removed by construction: a GRPOTrainer subclass applies the same regex mask to the trainer's logits at every completion position, in the old, reference and loss passes. That makes pi_trainer the masked distribution vLLM actually samples from.

The files show the mismatch was worse than a 0.15-nat bias. Under TRL 1.17.1's default IS correction (vllm_importance_sampling_correction true, sequence_mask, clip_max 3.0; read from runs/grpo-smoke-13657200/summary/a/run.json), smoke 13657200 shows:
- a per-step IS-weight mean of 0.0010-0.0814, against 0.698-1.027 without the grammar (13632679);
- one completion carrying 32-100 % of each step's weight, and at least 90 % in 7 of 15 steps;
- grad norms of 0.0009-0.062, against 0.059-0.221.
So the grammar run, as built, effectively trained on about one completion per step. Masking the trainer restores on-policy GRPO for the policy that is deployed. A laptop prototype shows the grammar still leaves real choices: at most 52 tokens are allowed at any position (median 20-21), only 7.3-7.5 % of positions are forced, and every one of the 30 drafts, the 10 kept included, is in the language of its own palette's regex.

D-020 does not allow this without a new entry, for four reasons:
- its verdict is recorded ("the grammar goes off for GRPO run 1 ... run 1's start is registered again, in a later entry");
- entries are never edited;
- the masked smoke changes files on D-020's frozen path (train.py, trl_probe.py, smoke_check.py, config.py);
- changing pi_trainer changes the algorithm, not only a reading.

So D-021 registers, before any job:
- the masked trainer;
- one masked smoke on the already-registered lofi-grpo-smoke type (worst case 500 SU, expected about 250). Its gates keep D-020's definition and its 0.05 threshold, and add fidelity, removal and gradient checks;
- step 0 = the base model decoded with v2 under the grammar;
- a fixed failure branch: if any gate fails, D-020's verdict stands and the SFT warm-start goes to Shresth.

Feasibility with TRL 1.17.1 and vLLM 0.32 is likely but not proven on the laptop. It rests on five TRL facts and two vLLM facts that only the cluster's installed source can confirm, and each is a fail-closed check made before the model loads. The smoke's own IS numbers already indicate that vLLM's reported log-probs come after the mask.

### Method

1. What run 1 trains. Everything is as D-020's run-1 settings, except item 2.
- Model: Qwen3-4B-Instruct-2507 @cdbee75f17c01a7cc42f958dc650907174af0554, bf16, with LoRA r 16, alpha 32, dropout 0, all-linear.
- Trainer: TRL 1.17.1 GRPOTrainer, with vLLM 0.32 in colocate mode.
- Settings from configs/grpo_smoke_v2_grammar.yaml:
  - lr 1e-5, constant_with_warmup, 3 warmup steps;
  - 4 prompts x 8 = 32 completions per step; num_iterations 1;
  - max_completion_length 512; mask_truncated_completions;
  - beta 0.04 to the adapter-off base; dapo loss; scale_rewards batch;
  - T 1.0, top_p 1, top_k 0;
  - TRL's IS correction left at its default: sequence_mask, clip_max 3.0.
- Run 1 changes only: max_steps 100, save_steps 25.
- Prompt: v2 (lofi_rl.prompt_v2, template 127e3094...).
- Grammar: vllm_structured_outputs_regex = loopscript_regex(palette), with xgrammar pinned. The smoke uses dev_gu (1412f11f...); run 1 uses the frozen lofi_rain_v1's own sha.
- Reward: D-020's run-1 reward, G x sum w(t)r - penalties. P_copy is scored against all 9 v2 references plus the seed pool. Taste, CE and theme come from the scorer server.
- Briefs: the training stream, seed 596, start 0, count 400.

2. The one change. A new optional config key, `trainer_logprobs: grammar_masked`. It is refused without `structured_outputs`, and the dry run and run.json record it.
- With it, train.py builds LofiGRPOTrainer(GRPOTrainer). Its only override is `_get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep, *a, **kw)`, which returns super()'s result computed inside a context that does three things:
  - (a) a forward pre-hook on `model` reads each micro-batch's input_ids;
  - (b) a forward hook on `model.get_output_embeddings()` fills the head's logits in place with -inf outside the regex's allowed set at completion position j. That is head row L-1-C+j, with C = logits_to_keep. Prompt rows, rows after the first EOS or pad, and the last row are left untouched;
  - (c) a sampled token outside its own mask raises, so the launch stops before a -inf ever reaches the loss.
- So every log-prob TRL uses is log q(y_t) = logit(y_t) - logsumexp over A_t: the old one (its IS ratio and logged gap), the reference one (the KL) and the loss's. That is exactly the distribution vLLM samples from.
- TRL's logged kl and entropy become the masked policy's.
- If check T2 shows that TRL's own `sampling_mask` is a keep-mask applied to the logits in all three passes, the override ANDs the grammar mask into it instead of using hooks. The math is the same.

3. The mask: src/lofi_rl/grammar_mask.py, standard library only, like grammar.py.
- A Thompson NFA of the exact regex string.
- The tokenizer's vocabulary, decoded byte-level to text.
  - Non-ASCII tokens, special tokens and rows past the tokenizer's ids are never allowed.
  - EOS means xgrammar's stop ids, allowed exactly at accepting prefixes.
- Allowed sets are memoized by NFA state set.
- The prototype's numbers are under data.

4. Cluster source checks, added to trl_probe as AST reads. They run before the model loads and fail closed; a refusal costs about 2 minutes.
- T1: the old, reference and loss per-token log-probs are each assigned only from `self._get_per_token_logps_and_entropies`. No other log-softmax of model logits exists on this path (use_liger_kernel is false).
- T2: that method's body:
  - one model call per micro-batch, rows in order, with logits_to_keep + 1;
  - the last row dropped, then the temperature division and the log-softmax;
  - what `sampling_mask` is, and that it is None at top_k 0 and top_p 1.
- T3: the sequence_mask code. The weight is exp(sum over the completion mask of old - sampling), zeroed above 3.0, and it multiplies the per-token loss. The logged mean is over the 32 sequences; the files agree, since max/(32 x mean) never exceeds 0.999998.
- T4: the KL comes from the reference and current log-probs; record what use_bias_correction_kl (true) does.
- T5: lm_head is not a LoRA target.
- V1: in v1/worker/gpu_model_runner.py's sampling function, apply_grammar_bitmask (line 4335 in 13657200's record) runs on the logits before self._sample (lines 3437 and 3448). The sampler then computes processed_logprobs from those logits (sampler.py 270-273), and TRL's engine sets logprobs_mode="processed_logprobs" (generation/vllm_generation.py:370).
- V2: in v1/structured_output/backend_xgrammar.py, how TokenizerInfo is built (vocab size, stop ids) and how compile_regex is called. They are copied to build M8's comparison bitmask. The xgrammar 0.2.8 API names are recorded too.

5. Records.
- run.json `trainer_logprobs`: the mode, the regex sha, the vocabulary sha, the EOS ids and the source verdicts.
- Per-step metrics appended to TRL's _metrics: mask/violations, mask/rows, mask/neglogz_mean (logsumexp over all minus logsumexp over allowed, in the old pass), mask/forced_share and mask/xgrammar_mismatch.
- smoke.json gets a `mask` block and D-021's rule.

6. The masked smoke.
- configs/grpo_smoke_v2_grammar_masked.yaml is configs/grpo_smoke_v2_grammar.yaml plus that one key.
- It runs on the unchanged jobs/grpo_smoke.sbatch, on an L40S like 13657200, so the numerics are comparable.
- Its code path is frozen in docs/prereg/D-021_masked_smoke_path.txt, with .sync-freeze set until the smoke ends.

7. D-021, written before the smoke.
- It supersedes D-020's run-1 verdict only through the gates M6-M11.
- It keeps D-020's definition: the mean, over every completion token of every step, of |log pi_vLLM - log pi_trainer|, where pi_trainer is now the masked trainer. It keeps the 0.05 threshold.
- The parse-failure half stays read: 397 of 500 is over 100.
- If the gates pass, D-020's Outcome (v2 with the grammar) is the run-1 setting again. The official Phase-1 gate then runs it on the frozen palette.
- If any gate fails, the grammar goes off (D-020) and the SFT warm-start (CLAUDE.md §9) goes to Shresth.
- One re-run is allowed, only after a fix recorded in D-021.
- D-021 also fixes the dev and held-out brief seeds, and that the seed pool's base-model samples are drawn under the grammar.

8. In run 1: the same mask metrics every step. A STOP, as in hack-watch, on any violation, on a per-step gap above 0.05, or on an IS-weight mean below 0.3 for 3 steps in a row.

9. Alternatives rejected.
- vLLM pre-mask (raw) log-probs, or turning TRL's IS correction off: both hide the mismatch, since the sampler stays masked and the trainer does not.
- Generating in the trainer without vLLM: too slow.
- The SFT warm-start: it is the fallback, not this angle.

### Data from the files

The scratch scripts are in <session scratchpad>/run1design/masked-trainer/: inlang.py, tokmask.py, tokmask_points.py, tokmask_timing.py and drafts_inlang.py.

No SFT data is used, so there is no example count. What follows is what the design rests on.

Raw completions that fullmatch the dev_gu regex (sha 1412f11f, 3,395 characters; Python re, my counts):
- runs/junk-13656050 (v2 without the grammar): 26 of 500. Its junk.json has G1 397 of 500.
- runs/junk-13640325 (arm B): 500 of 500. junk.json: pass 500/500, pass_not_copy 446/500, ci95 [0.8617, 0.9163].
- runs/junk-13632676 (v0): 0 of 500.

runs/grpo-smoke-13657200/summary (243.0 SU, L40S):
- smoke.json:
  - 480 of 480 stop completions in the language;
  - IS readout token-weighted mean 0.1538, per step 0.105-0.257;
  - mean step_s 37.63 (35.47-39.92); startup 115.0 s.
- {a,b}/steps.jsonl:
  - sampling/importance_sampling_ratio/mean 0.0010-0.0814 per step (mean of the 15 steps 0.0219);
  - the largest weight's share of the step's 32-weight sum 0.319-1.000, at least 0.9 in 7 of 15 steps;
  - sampling_logp_difference/max 20.57-41.48;
  - grad_norm 0.00088-0.0618.

runs/grpo-smoke-13632679 (no grammar):
- IS-weight mean 0.698-1.027 (mean 0.874);
- largest share 0.061-0.112;
- gap 0.0044-0.0091 per step, max 0.73-2.50;
- grad_norm 0.059-0.221;
- step_s 40.65.

Cross-check: E_q[product of Z] equals the unmasked model's in-language rate. The 0.0219 mean weight has the same order as 26/500 = 0.052. This is an inference that vLLM's log-probs are taken after the mask, not a verified fact.

13657200's a/run.json:
- GRPOConfig: vllm_importance_sampling_correction true, mode sequence_mask, clip_max 3.0, steps_per_generation 8, generation_batch_size 32, use_bias_correction_kl true, cast_lm_head_to_fp32 false.
- TRL source lines 2762-2772: old log-probs come from self._get_per_token_logps_and_entropies(self.model, ..., sampling_mask=sampling_mask) and return 3 values.
- The vLLM evidence lines are listed under method 4.

Token-mask estimate on the laptop:
- How it was built: my own NFA of the regex (61,206 states); the Qwen3 tokenizer.json from ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit; canonical BPE re-encoding, which equals vLLM's token count for 227/320, 107/160 and 361/500 completions.
- Forced positions (1 token allowed, EOS counted):
  - 5,998 of 81,359 = 0.074 (launch a);
  - 2,940 of 39,233 = 0.075 (launch b);
  - 9,521 of 130,347 = 0.073 (arm B).
- Allowed-set size: median 20-21, p90 36, p99 52, max 52.
- 0 violations in 980 completions.
- GRAMMAR.md §3.3's 4 points reproduced: 20 with EOS refused, 20 + EOS, 25, 8.
- Cost: NFA 0.02 s, vocabulary trie 0.13 s, one 32-completion step 0.136 s cold and 0.013 s warm.

drafts/v1: all 30 drafts, and all 10 kept per gates/GATE_palette.md, fullmatch loopscript_regex(palettes/draft_day2.yaml) (sha 5c081828). Under dev_gu, only 3 of 30 do (other names).

docs/prereg holds no brief splits, only D-020's two frozen-path lists. Seeds in use: 596 (GRPO), 2026 (junk-rate), 1105 (drafts).

SU used so far: 1,481.9 (CHANGELOG Status).

### Step 0

Step 0 is the policy before any training:
- the model: Qwen3-4B-Instruct-2507 @cdbee75f with no adapter;
- the prompt: v2 (template 127e3094...);
- decoding: vLLM under the grammar built from the frozen palette (its own regex sha), at T 1.0, top_p 1, top_k 0, max_tokens 512.

This is the masked policy at theta_0. D-020's arm B measured it on dev_gu: 500/500 pass and 446/500 pass_not_copy. The official Phase-1 gate re-measures it on the frozen palette. Step N is the same decoder with checkpoint N's LoRA.

What gets trained is the masked policy; the trainer never learns syntax for the unmasked distribution. So everything is drawn through the identical grammar:
- every checkpoint;
- the dev briefs that pick step N;
- the 100 held-out briefs of the blind A/B (D-014);
- the obedience check (at least 90 %);
- the across-brief no-collapse distance (at least 70 % of step 0, D-009);
- the playlist heart rate.

Both arms of every claim go through that grammar, and an unmasked sample is never shown.

This is the reading D-020 itself made for "both arms pass": run 1 starts from the base model with v2 and the grammar. So D-021 can record it. It still needs Shresth's word, a one-line confirmation, for two reasons:
- D-014 and D-009 are his pre-registered claims;
- D-020's IS verdict moved the plan of record to the SFT warm-start, which would define a different step 0.

### Jobs and SU

Rates are CLAUDE.md §4's: GPU 8 SU/min, CPU 1 SU/core-min, memory 1 SU per 4 GB-min.

(1) Laptop build, tests, dry run and D-021: 0 SU.

(2) The masked smoke, `lofi-grpo-smoke`:
- Already registered: sha 1dded12d386c7fab, max 500 SU. The type does not change, because the new config is passed as an argument.
- Command: make submit JOB=jobs/grpo_smoke.sbatch OVR='--gres=gpu:l40s:1 --constraint=l40s' ARGS='configs/grpo_smoke_v2_grammar_masked.yaml'
- Shape: gpu partition, 1 L40S, 4 CPU, 32 GB, 25 min, so 20 SU/min.
- Worst case 500 SU. Expected about 250 SU, an estimate: 13657200 cost 243.0 SU in 12:09, and the mask adds under 1 s per step.
- No approval and no re-registration needed.
- At most one re-run after a recorded fix: up to 500 SU more.

(3) Carried from D-020, not new: the official Phase-1 gate on the frozen palette.
- lofi-junk-rate, sha 4a9d8685231fd573, max 450 SU; an A40 on gpu, ARGS='500 2026 1 v2'.
- Up to 450 SU. The two v2/v1 arms cost 93.0 and 99.0 SU.
- Later, the step-0 samples for the dev and held-out streams use the same type, up to 450 SU each.

(4) Run 1: a new type, jobs/grpo.sbatch (`lofi-grpo`).
- Shape: 1 GPU + 12 CPU + 64 GB = 36 SU/min (taste_pipeline §6). 100 steps, a save every 25, and a code snapshot inside the job.
- On an A100-80 at --time 1:45: worst case 3,780 SU, expected about 2,300 ±35 % (§6, UNVERIFIED).
- On an L40S at --time 2:30: worst case 5,400 SU, expected about 3,000, an estimate. It is built from 100 steps of (37.6 s symbolic step from 13657200 + under 1 s of mask + 8.9 s cold scorer batch from 13651611), plus about 5 min of startup and saves: about 83 min.
- It exceeds 500 SU, so it needs Shresth's yes in chat, registered BY=Shresth.

Only run 1 exceeds 500 SU. New SU before run 1 under this design: about 250 expected, at most 500 (at most 1,000 with the one re-run). 1,481.9 of the 1,000,000 target is used so far.

### Validation gates

- M1 (laptop): pytest -q passes on the tree that holds D-021, with new tests tests/test_grammar_mask.py and tests/test_rl_masked_logps.py (the alignment logic, with stand-ins). Read from the pytest run. The merged tree had 2503 passed and 3 skipped on 2026-11-01 (CHANGELOG).
- M2 (laptop): lofi_rl.grammar_mask accepts every token of all 500 completions in runs/junk-13640325/summary/completions.jsonl and all 480 in runs/grpo-smoke-13657200/summary/{a,b}/rewards.jsonl, re-encoded with the Qwen3 tokenizer.json, with EOS allowed at each end. Required: 0 violations of 980. The prototype had 0.
- M3 (laptop): its allowed sets at the four points of docs/GRAMMAR.md §3.3's table, on the in-prompt example, equal that table: end of A 20 tokens with EOS refused, end of B 20 with EOS allowed, after kick's one grid 25, after the last edit 8. The prototype matched all four.
- M4 (laptop): the dry run of configs/grpo_smoke_v2_grammar_masked.yaml reports regex sha 1412f11f..., v2 template 127e3094... and reward check 0.2784511276657195, the same as runs/grpo-smoke-13657200/summary/dry_run.json. tests/test_rl_train.py pins that the config differs from configs/grpo_smoke_v2_grammar.yaml only by trainer_logprobs.
- M5 (registration): D-021 in docs/DECISIONS.md and docs/prereg/D-021_masked_smoke_path.txt are committed and synced before the submit. .sync-freeze names the list until the smoke ends. The stamps in {a,b}/run.json show that commit with dirty=0.
- M6 (cluster, validity as for 13657200), read from runs/grpo-smoke-<job>/summary/smoke.json: ok is true and the log ends GRPO_SMOKE_OK; in_language stop.out_of_language is 0 and other.n is 0, with at least 1 stop completion per launch; the xgrammar backend is pinned and applied in both launches.
- M7 (cluster, sources), read from {a,b}/run.json trainer_logprobs.sources: the TRL checks T1-T5 must read match, or the launch stops before the model loads (exit 2) and D-020's verdict stands unless a fix is registered. The vLLM checks V1-V2 are recorded, and M10 decides them.
- M8 (cluster, fidelity), read from smoke.json mask: 0 sampled tokens outside the trainer's mask, over every completion token of both launches; 96 rows masked per step (32 old + 32 reference + 32 loss); 0 positions, over every position of every completion, where the trainer's allowed set differs from xgrammar 0.2.8's bitmask built as V2 reads vLLM's backend.
- M9 (cluster, D-020's quantity and threshold, unchanged), read from smoke.json is_ratio: the token-weighted mean, over every step of both launches, of TRL's sampling/sampling_logp_difference/mean (source verdict match) must be at most 0.05. 13657200 read 0.1538.
- M10 (cluster, removed rather than reduced), read from smoke.json mask.neglogz and is_ratio: the trainer's own token-weighted mean -log Z must be at least 0.05, and M9's value at most 0.25 times it. If vLLM's log-probs came before the mask, the gap would equal -log Z and this would fail.
- M11 (cluster, the gradient survives TRL's IS correction), read from {a,b}/steps.jsonl: at every step, sampling/importance_sampling_ratio/mean is at least 0.3 and max/(32 x mean) is at most 0.25. For reference, 13632679 had 0.698-1.027 and 0.061-0.112; 13657200 had 0.0010-0.0814 and 0.319-1.000.
- M12 (cluster, cost; it decides only the run-1 estimate), read from smoke.json steps and sacct: mean step_s at most 43.3 s (1.15 x 13657200's 37.63 s), and the job at most 500 SU. If it is over, the lofi-grpo estimate is redone before the ask.
- M13 (carried from D-020): the official Phase-1 gate of v2 with the grammar on the frozen palette. runs/junk-<job>/summary/junk.json must show pass_not_copy.count of at least 300 of 500 in a valid run, and the 50-clip 'fine' ear check, at least 30 %, is recorded in gates/ with Shresth's line.
- Rule: if M6-M11 all pass, the grammar stays on for run 1 with the masked trainer. If any fails, the grammar goes off (D-020) and the SFT warm-start goes to Shresth. Only one re-run is allowed, after a fix recorded in D-021. The only fix named in advance for a miss caused by noise alone is cast_lm_head_to_fp32: true.

### Risks

- TRL 1.17.1's internals may not be what T1-T5 assume. There could be another path that computes log-probs, different micro-batching or row alignment, or a sampling_mask with other semantics. Then the launch stops at the source check (about 2 min, at most about 40 SU) or fails M8, and D-020's verdict stands unless a registered fix follows.
- The trainer's mask may differ from vLLM's xgrammar mask. Likely places: the EOS/stop ids, the model's padded vocabulary (151,936 in Qwen3's config, to be read from the snapshot) against the tokenizer's 151,669 ids, special tokens, or xgrammar's regex conversion. The result would be violations, which stop the launch, or a residual gap. M8's per-position equality check against xgrammar finds both.
- vLLM's processed log-probs may come before the mask after all (D-020's caveat). Then the gap stays about -log Z and M9/M10 fail. The files make this unlikely, but that is an inference: IS weights of 0.001-0.081 and per-token gaps up to 20.6-41.5 nats would both sit at the no-grammar level if both operands were unmasked.
- The trained policy is valid only under the grammar. p_theta is never taught syntax, and its mass outside the grammar can drift; mask/neglogz logs it, and nothing gates it. Any checkpoint sampled without the grammar, or base-model seed-pool samples drawn without it, is not the trained policy.
- Learning happens only at choice points. About 7.4 % of tokens are forced (laptop estimate) and carry no gradient and no KL. DAPO's per-token normalization dilutes the loss by that share.
- Numerics. Where the grammar removes the dominant token, the masked log-prob is computed from tail logits, so bf16 noise there may exceed the no-grammar 0.0044-0.0091. M9's 0.05 leaves room and M10 is relative. M11 could still miss on biased noise alone; the pre-named fix is cast_lm_head_to_fp32.
- How it reads as pre-registration. A trainer change after D-020's reading could look like moving the bar. D-021 counters that: it keeps D-020's definition and threshold, adds stricter checks, is registered before the smoke, allows one smoke plus one re-run after a recorded fix, and leaves D-020's verdict in force if any gate fails.
- TRL's default sequence-level IS correction still drops sequences whose weight is above 3: 13632679's per-step minimum was 0 at 10 of 15 steps. That behavior is unchanged, and benign once pi_trainer matches the sampler.
- Memory. Each forward adds a boolean mask of micro-batch x length x vocabulary, about 180-190 MB at 4 x 300 (estimate), on a 48 GB card that already gives vLLM 0.30 of it. Peak memory is recorded.
- The smoke validates dev_gu's regex, but run 1 uses the frozen palette's regex (a new sha). The in-run STOP rule and M8's counters on run 1's first steps cover that gap.
- Calendar. 13657200 queued 2.9 h, and D-020's junk arms queued 11.5 h or more. The masked smoke should end by about Nov 9-10, so the SFT fallback (planned for Nov 9-10 in infra_schedule risk 4) still fits before run 1 on Nov 16.

### What Shresth decides

- Which path run 1 takes. Either D-021's masked trainer: the masked smoke first (at most 500 SU, an already-registered type), with the SFT warm-start only if a gate fails. Or D-020's verdict as it stands: grammar off, and the SFT warm-start draft the CHANGELOG announced for his OK. His D-016 order holds either way, since D-021 keeps lever 1 and D-020's verdict moves to lever 2. Claude's pick: the masked smoke first.
- Confirm step 0 = the base model decoded with v2 under the grammar. Every checkpoint, the dev briefs that pick step N, and the held-out A/B, obedience and no-collapse samples are all decoded through that same grammar. This is a reading of his D-014 and D-009.
- The run-1 job type `lofi-grpo` (over 500 SU, estimates). On an A100-80: about 2,300 SU, worst case 3,780 at 1:45. On an L40S: about 3,000 SU, worst case 5,400 at 2:30. It needs his yes once, registered BY=Shresth.

## `rft`: sft1: a self-distilled grammar warm-start (rejection-sampled LoRA SFT on the model's own constrained v2 loops), then GRPO run 1 without the grammar

**Judge's score:** 6. Sub-scores (0-10): syntax without the grammar 7; no copying or collapse 3; final claims 6; SU and Shresth's time 5; implementation risk 6; fallback order 9. Mean 6.0.
- Syntax: its targets are exactly the grammar's language, and the habit it must cure is narrow. All 397 G1 failures without the grammar are a second grid on an @N edit: 1,658 of 2,990 edit lines (recounted from runs/junk-13656050/summary). One epoch on about 1,750 targets plausibly cures it, but no SFT has been measured here; V5a and a second-epoch branch cover a miss.
- Copying: it filters only on whole-loop J_any < 0.8. Of its 445 arm-B candidates, 367 carry the shown example's A or B line verbatim, and 86.6 % of their grids (6,385 of 7,376) are the example's (my count over runs/junk-13640325/summary/scored.jsonl). So it distills part-level copying. V6 and V7 only ask for no worse than base + grammar, and that baseline is already concentrated: within each cell, the nearest neighbour's J has median 0.9377, with 34 exact duplicates (recomputed). The KL to sft1 then anchors that. Its rule that P_copy's pool never holds model-written loops is right, and I grafted it.
- Claims: step 0 = sft1, a sharpened base + grammar. Reward-free selection keeps his taste out of it, but it still redefines step 0 for D-009 and D-014, which needs his word.
- Cost: at most 2,186 SU before run 1 (about 1,250 expected), including a 1,200-SU job type that needs his yes. Three jobs run in sequence after the freeze, and it puts nine decisions to him.
- Risk: mostly existing junk machinery. The 9,000-sample yield is a power-law extrapolation from 37-67 candidates per cell, and TRL's SFT field names are unverified until the job runs.
- Order: exactly D-016's lever 2, as D-020's verdict and the CHANGELOG direct.

### Summary

GRPO run 1 starts from sft1. sft1 is Qwen3-4B-Instruct-2507 after one epoch of LoRA SFT on its own grammar-constrained prompt-v2 loops, with the adapter merged into the weights. Run 1 then trains without the grammar, so the IS gap that turned the grammar off (0.1538 in smoke 13657200) does not arise.

A new data job samples 9,000 constrained completions on the frozen palette, using training-stream briefs (seed 596, indices from 1,000,000). A loop is kept only if all of these hold:
- it passes G1–G6 and compile;
- it is not a near-copy of any v2 example (J_any < 0.8, D-019's near-copy line and the top of the copy penalty's ramp);
- it is not within J 0.9 of a loop already kept in its energy × mood cell (at most 250 per cell).

Parser repairs are written out in canonical form. No reward term ranks or selects beyond the near-copy cut.

On D-020 arm B's 500 completions this keeps 445 before de-dup and about 243 after. Per-cell fits predict about 1,750 from 9,000 (an estimate). I rejected a J ≤ 0.6 cut: it keeps only 52 of 500, with 0 or 1 in four cells, and that tail scores lower on theory (0.908 vs 0.973) and on mood (f_mood 0.442 vs 0.713).

Before run 1, sft1 must pass these checks, on D-019's 500 briefs without the grammar:
- G1+G2+G5 ≤ 100, the lever rule, so the grammar stays off;
- pass_not_copy ≥ 300, the Phase 1 bar;
- copying and within-cell diversity no worse than base + grammar on the same briefs;
- mean |log IS| ≤ 0.05 in a GRPO smoke;
- ≥ 30 % Fine in Shresth's 50-clip ear check.

Step 0 becomes sft1 for every claim, which needs Shresth's word. Cost before run 1 is at most 2,186 SU worst case, about 1,250 expected. Only the lofi-sft job type (worst 1,200 SU) needs his yes.

### Method

1. REGISTER FIRST
A new entry, D-021 ("run 1's start registered again", which D-020's verdict requires), is committed before any job. It records everything below.
- Brief-stream registry, in docs/prereg/brief_streams_v1.json:
  - GRPO rows: seed 596, indices [0, 1e6);
  - SFT data: seed 596, [1e6, 2e6); the pilot uses [2e6, 3e6);
  - measurement only: seed 2026, [0, 500);
  - RM held-out 40, dev 40 and final 100: three new seeds, fixed in the same commit, never trained or selected on.
- Also recorded: the filters, the SFT settings, gates V1–V9 and their branches.
- A frozen code path (docs/prereg/D-021_sft_path.txt plus .sync-freeze), held while the jobs wait or run.
- Note on streams: the v2 prompt depends only on energy × mood, so every stream sends the same 9 prompts. Training and evaluation are kept apart only because evaluation loops are fresh samples, never SFT targets or GRPO rollouts.

2. DATA: job lofi-sft-data (policy env, then the scorer env for a 50-clip G7 sample)
- Model: Qwen3-4B-Instruct-2507 @cdbee75f, on vLLM 0.32 in bf16 eager mode, with xgrammar pinned.
- Decoding: constrained by loopscript_regex(palettes/lofi_rain_v1.yaml). The regex's new sha is recorded.
- Prompt: v2 (template 127e3094…), with messages built from the frozen palette.
- Sampling: D-019's (T 1.0, top_p 1, top_k 0, max_tokens 512), one completion per brief.
- Briefs: N = 9,000 from sample_briefs(596, 9000, start=1,000,000), about 1,000 per cell. They run in chunks of 128 under a deadline that leaves time to score.
- Scoring: junk.score_row (G1–G6, compile, repairs, copy readouts against the 9 v2 references filled for the frozen palette, knob readouts). It writes a standard junk.json, which doubles as the frozen-palette version of D-020 arm B.

3. FILTER AND TARGET (new module lofi_rl.rl.sft_data, standard library only)
A completion is kept only if all of these hold:
- finish reason stop, and in the regex's language;
- G1–G6 and compile pass;
- no compiler repair (sax_fold);
- J_any < 0.8, the largest character 5-gram Jaccard with any of the 9 v2 examples. This is junk's measure, the same J that P_copy reads.

The target text:
- If the parser logged no repair, the target is the raw completion.
- Otherwise it is canon(AST without sax:), laid out with the completion's own line ends. pad_phrase, trunc_phrase and misplaced_bar are therefore written out corrected.

De-dup and cap:
- Inside each energy × mood cell, survivors are walked in a seeded order (seed 596). Any loop whose J with a target already kept in that cell is ≥ 0.9 is dropped; this also drops exact duplicates.
- Each cell is capped at 250, filled at random (seeded), never ranked.
- Theory, knobs, P_sax, taste and audio are readouts only, never filters. The SFT therefore distills the grammar and the no-copy rule, not the reward.

Outputs:
- sft_set.jsonl: brief id, cell, messages sha, target, J_any, nearest reference.
- sft_data.json: counts after each filter by cell, the discovery curve, readouts of kept vs all, and the rows whose re-encoded token count differs from vLLM's.

4. SFT: job lofi-sft (policy env; TRL 1.17.1 SFTTrainer and PEFT 0.21.2 from requirements/policy.lock.txt)
- Base: Qwen3-4B-Instruct-2507 @cdbee75f, bf16, gradient checkpointing.
- LoRA: r 16, α 32, dropout 0, all-linear (taste_pipeline §6's shape).
- Examples: the prompt is the exact v2 chat messages for the brief. The completion is the target plus the chat template's end-of-turn token. Loss is on completion tokens only.
- Split: 95/5 train/eval, stratified by cell (seed 596).
- Schedule: LR 1e-4 peak, 5 % warmup, cosine to 0, 1 epoch, AdamW, weight decay 0, seed 596, max length 2,048.
- Effective batch 16: 8 × 2 on an A100-80; 4 × 4 on a 48 GB card.
- Before loading, the job checks the SFTConfig fields, the completion mask and the end-of-turn handling against the installed TRL, as fit_fields does.
- Then merge_and_unload into bf16 → $REMOTE_KEEP/ckpt/sft1-<job>/merged, with a sha256 manifest. The adapter is kept beside it.

5. EVALUATION, IN THE SAME JOB
- (a) eval_sft: junk generate + score of sft1 without the grammar. Prompt v2, frozen palette, D-019's 500 briefs (seed 2026), D-019's sampling. The model is identified by its manifest sha; junk.py gains a --model-path option for this.
- (b) eval_base_grammar: the base model with the grammar on the same 500 briefs and palette.
- (c) A render of 50 of (a)'s gate-passing loops that are not near-copies, for the ear check.
- junk.json gains two readouts:
  - a diversity block: cell-balanced within-cell mean of 1 − J5 over gate-passing loops, and the count of extra exact duplicates;
  - an in-language count for unconstrained runs.

6. SMOKE
The registered lofi-grpo-smoke job, with a new config file, configs/grpo_smoke_v2_sft.yaml:
- It is configs/grpo_smoke_v2_grammar.yaml minus structured_outputs.
- The model is sft1, given by path plus manifest sha; the palette is lofi_rain_v1.
- It runs 10 steps, then resumes from step 5.

7. RUN 1 START (configs/grpo_run1.yaml, only after V1–V9 pass)
- Model sft1, with no structured_outputs. The regex stays only as a per-step in-language readout for drift.
- Prompt v2.
- KL reference: PEFT's adapter-off model, which is sft1, at β 0.04. A fresh r-16 LoRA trains on top.
- Briefs: seed 596, indices [0, 400).
- copy_penalty pool: the 9 v2 examples, plus Claude-written seeds, mutations and the kept drafts once they exist. It never holds sft_set targets or any other model-generated loop.
- Scorer and taste RM: as in taste_pipeline §5 (RM gate Nov 15). All other §6 settings are unchanged.
- The seed pool's 660 step-0 samples, the RM held-out tuples and PAR's Ref_b are all drawn from sft1.

8. LAPTOP WORK BEFORE THE FIRST JOB (pytest -q)
- sft_data.py, with a test that reproduces the arm-B funnel: 445 candidates, 445/445 valid targets, about 243 kept after the J ≥ 0.9 de-dup.
- junk.py: --start, --model-path, and the diversity and in-language readouts.
- sft.py.
- Config and trainer support for a local merged model, with its manifest verified before load.
- smoke_check's IS rule applied to the unconstrained config.
- The two new jobs and the two configs.

9. OPTIONAL PILOT (Shresth decides)
Run the same data job and SFT job on palettes/dev_gu.yaml now, on indices from 2e6. That gives V5–V7 days before the freeze. Its checkpoint can never start run 1: dev_gu's keys are organ and rhodes and its only kit is jazz, while draft_day2's keys are rhodes, wurli and piano and its kits studio and jazz.

### Data from the files

SOURCES
- D-020 arm B: runs/junk-13640325/summary/{completions.jsonl, scored.jsonl, junk.json}. That is base + grammar + prompt v2, on dev_gu, seed 2026.
- Measured on the laptop with the real verify.score, copy_penalty and grammar modules. My J equals scored.jsonl's copy_any_jaccard on 500 of 500.
- Scratch scripts: survive.py, targets.py, dedup.py, yield.py and div.py in <session scratchpad>/run1design/sft-selfdistill/

FUNNEL ON ARM B (500 completions)
- 500 pass G1–G6 and compile.
- 499 have no compiler repair (sax_fold).
- 445 also have J_any < 0.8 (89.0 %).
- All 445 give valid targets: each is in regex 1412f11f and re-parses with no repair. 404 are the raw text; 41 are rewritten canonically.
- About 243 remain after the J ≥ 0.9 de-dup (mean of 20 seeded orders). Among the 445 there are only 420 distinct canonical texts, so 25 are exact duplicates.

OTHER COPY THRESHOLDS (gates only)

| Threshold | Kept of 500 |
|---|---|
| J_any ≤ 0.6 | 52 |
| J_any < 0.7 | 181 |
| J_any < 0.75 | 327 |
| J_any < 0.8 | 446 |

- J ≤ 0.6 by cell: high/hopeful 1/60, high/melancholy 0/50, high/wistful 1/57, low/hopeful 1/52, low/melancholy 5/45, low/wistful 8/70, mid/hopeful 9/53, mid/melancholy 3/57, mid/wistful 24/56.
- J < 0.7 is thinnest in high/wistful (4/57) and mid/melancholy (5/57).

WHY REWRITE REPAIRS RATHER THAN DROP THEM
- Dropping repaired loops, and requiring P_sax 0, cuts the J ≤ 0.6 count to 16 and the J < 0.8 count to 397.
- Within each cell, the less-copying half has p_fix > 0 in 40 of 220 loops; the other half in 0 of 226.

QUALITY BY COPY BAND (verify.score)

| Band | n | Theory | f_mood | Energy match |
|---|---|---|---|---|
| J ≤ 0.6 | 52 | 0.908 | 0.442 | 0.962 |
| 0.6 < J < 0.8 | 394 | 0.973 | 0.713 | 1.0 |
| J ≥ 0.8 | 54 | 0.973 | 0.537 | 1.0 |

HOW CONCENTRATED STEP 0 ALREADY IS
- There are only 9 distinct prompts, one per energy × mood cell.
- Mean pairwise J is 0.7747 within a cell and 0.2894 across cells (20,000 sampled pairs).
- Within a cell, the nearest-neighbour J has median 0.9377; 448 of 500 have a neighbour at J ≥ 0.8.
- Cell-balanced within-cell mean of 1 − J: 0.2263.
- 34 extra exact duplicates among the 500.

DE-DUP OF THE 446 NON-COPIES (5 seeded orders)

| Drop at J ≥ | Kept |
|---|---|
| 0.95 | 327 |
| 0.9 | 244.6 |
| 0.85 | 161.2 |
| 0.8 | 113.0 |

At 0.9, the kept fraction falls from 0.83 (after 5 seen per cell) to 0.58 (after 40).

YIELD FORECAST (estimate: power-law fits per cell on 37–67 candidates)

| Generated per cell | Kept at 0.9 | high/wistful at 0.9 | Kept at 0.95 | high/wistful at 0.95 |
|---|---|---|---|---|
| 500 | ≈1,492 | ≈37 | ≈2,303 | 93 |
| 1,000 | ≈2,652 | ≈56 | ≈4,280 | 153 |
| 2,000 | ≈4,730 | ≈84 | ≈7,978 | 253 |

- With the 250-per-cell cap at 1,000 generated per cell, the forecast is about 1,757 kept.
- So N = 9,000.

TRAINING SIGNAL PER TARGET (junk.json)
- Each loop has 5.87 per-bar edits and 16.6 grids on average.
- About 1,750 targets therefore show about 10,000 one-grid edits and about 29,000 16-cell grids.

STEP-0 COPY PENALTY
P_copy has mean 0.1754 over all 500, which matches GRPO.md, and 0.1606 over the 445 candidates.

WITHOUT THE GRAMMAR (runs/junk-13656050/summary/junk.json)
- G1 397 of 500; pass 78; pass_not_copy 64; 22 pass with no repair.
- Raw text in the regex's language: 26 of 500. Arm B: 500 of 500; v0: 0 of 500.
- The top 8 G1 details, all "takes at most 1 grid(s)", cover 311 of the 397.
- Mean completion length is 283.03 tokens, against 261.19 with the grammar.

IS GAP (runs/grpo-smoke-13657200/summary/smoke.json and {a,b}/steps.jsonl)
- Token-weighted mean 0.1538.
- At step 1 the adapter is still at its initial state, so vLLM and the trainer hold the same weights. The gap there is 0.1146, against 0.0087 at step 1 of the unconstrained smoke 13632679, which ran 0.0044–0.0091 per step. The gap comes with the grammar.
- In 13632679 (prompt v1, no grammar), the gate rate was 0.094 at step 1 and 0.19–0.38 at steps 7–10.

THROUGHPUT
Arm B generated 130,595 tokens in 50.62 s, or 2,579.9 tok/s, on an A40. Prompts are 1,322–1,408 tokens.

BRIEF SEEDS ON RECORD
- docs/prereg holds only the two D-020 frozen-path lists.
- No held-out, dev or final seeds are recorded yet (GRPO.md §6.5).
- The configs fix 596 as the training seed; junk-rate uses 2026.

### Step 0

WHAT STEP 0 IS
Step 0 = sft1: the merged SFT checkpoint, identified by its manifest sha, sampled without the grammar under prompt v2 on the frozen palette at T 1.0, top_p 1, top_k 0. It is GRPO run 1's starting policy, and also its KL reference: the adapter is merged, so PEFT's adapter-off model is sft1.

WHERE IT IS USED
- the preference claim (D-014): step N vs step 0 on 100 held-out briefs, each arm using its first gate-passing sample of up to 4 draws;
- the no-collapse claim (D-009): step N keeps ≥ 70 % of step 0's across-brief distance;
- the ♥ secondary;
- the step-0 vs step-100 pair for the Nakano deck;
- every "step-0 sample" in taste_pipeline: the RM held-out tuples, PAR's Ref_b, the seed pool's 660 step-0 samples and the checkpoint quads.

BASE + GRAMMAR STAYS AS A RECORDED REFERENCE, NOT A COMPARATOR
- V6 and V7 compare sft1 with it before run 1, so an SFT that collapses diversity or copies more cannot become step 0 and hide that.
- An optional, ungated 30-trial blind A/B of sft1 vs base + grammar would show the SFT's own effect.

WHY NOT THE UNCONSTRAINED BASE MODEL
v2 without the grammar passes 78 of 500 per draw (runs/junk-13656050). With 4 draws, (1 − 0.156)^4 ≈ 0.51 of the A/B briefs would have no gate-passing step-0 loop and would forfeit.

WHY NOT BASE + GRAMMAR AS STEP 0
It is not a checkpoint of the run and not its KL reference, and it would credit the SFT's effect to RL.

THIS NEEDS SHRESTH'S WORD
- D-009 and D-014 are his pre-registrations, written when step 0 was the base model (D-019: v0 decides whether run 1 starts from the plain base model).
- D-020 (c) requires the SFT warm-start be taken to him before any job.

### Jobs and SU

Rates (CLAUDE.md §4): GPU 8 SU/min each, CPU 1 SU per core-minute, memory 1 SU per 4 GB-minute.

| Job | Status | Shape | SU/min | Worst SU | Expected (estimate) |
|---|---|---|---|---|---|
| J1 lofi-sft-data | new type, ≤ 500: Claude registers it | gpu partition, 1 GPU (A40 like the junk runs, any bf16 card), 4 CPU, 24 GB, 27 min | 18 | 486 | about 22 min, 400 SU |
| J2 lofi-sft | new type, over 500: needs Shresth's yes | gpu, 1 A100-80GB (an L40S by OVR at the same walltime), 4 CPU, 32 GB, 60 min | 20 | 1,200 | 25–35 min, 500–700 SU |
| J3 lofi-grpo-smoke | already registered (sha 1dded12d386c7fab, max 500); only a new config file | A100, 25 min | 20 | 500 | about 270 SU |

Expected-time breakdowns:
- J1, from arm B: load 23.1 s; 9,000 × 261.19 tokens at 2,579.9 tok/s ≈ 15.2 min; scoring ≈ 37 ms a loop on 3 workers ≈ 2 min; de-dup under 1 min; a 50-clip G7 render ≈ 2.5 min.
- J2, UNVERIFIED: about 1,670 training examples × about 1,630 tokens ≈ 2.7M tokens of training; merge and save ≈ 3 min; two 500-brief evals ≈ 5 min; 50 renders ≈ 2.5 min.
- J3: its own header's estimate. Smoke 13657200 took 243.0 SU on an L40S.

Only if a gate fails:
- J2 again: ≤ 1,200 SU;
- a J1 refill: ≤ 486 SU;
- a grammar smoke on sft1: ≤ 500 SU.

Optional dev_gu pilot now: J1 ≤ 486 + J2 ≤ 1,200, so ≤ 1,686 SU worst case, about 900–1,100 expected.

Total before run 1, with no branches and no pilot: worst case 2,186 SU, expected about 1,150–1,370 SU. So far 1,481.9 SU of the 1,000,000 target are used (CHANGELOG Status).

Run 1 itself is a new lofi-grpo type over 500, so it needs Shresth's yes. taste_pipeline §6's estimate stands (UNVERIFIED):
- A100: about 2,300 SU ± 35 %; worst case 36 SU/min × 1:45 = 3,780.
- L40S: about 3,200 SU ± 35 %; worst case 36 SU/min × 2:30 = 5,400.

Over 500 SU: J2 (and the pilot's J2), and run 1.

### Validation gates

- V1. Data validity. Read from runs/sft-data-<job>/summary/junk.json and sft_data.json. All of these must hold:
- model cdbee75f;
- prompt v2 (template 127e3094), with every row's messages sha equal to its cell's;
- palette sha equal to the frozen lofi_rain_v1's, with matches_generate true;
- constrained with the frozen palette's regex: the recorded sha equals the rebuilt one, xgrammar applied, stop out_of_language 0 and other.n 0 (D-020's Validity);
- sampling equal to runs/junk-13632676's;
- every brief index in seed 596's [1,000,000, 2,000,000);
- generate and score stamps equal, and the frozen files equal docs/prereg/D-021_sft_path.txt.

If it fails: fix and re-run once. A second failure goes to Shresth.
- V2. Enough data. Read from sft_data.json's kept counts: total ≥ 1,000 and every cell ≥ 40. The arm-B forecast is about 1,750 in total, with the thinnest cell about 56.

If it fails, once:
- a cell under 40 is refilled from its own candidates with the de-dup at J ≥ 0.95 (recorded);
- if the total is still short, one more lofi-sft-data job runs on the next 9,000 indices and the two pools merge.
- V3. Purity. An independent laptop script (not the job's code) re-checks the pulled runs/sft-data-<job>/summary/sft_set.jsonl. All of these must hold:
- every target fullmatches the frozen regex;
- every target parses with gates passed and no repair, to the recorded canonical text;
- J_any < 0.8 against all 9 v2 references, filled for the frozen palette;
- no pair inside a cell has J ≥ 0.9 (0.95 in a refilled cell);
- every brief index is in the SFT range;
- the count equals sft_data.json's.

If any check fails, the set is rebuilt. No SFT job runs before V3 passes.
- V4. Training sanity. Read from runs/sft-<job>/summary/sft.json. All of these must hold:
- all losses finite;
- on the held-out 5 %, completion-token NLL after SFT is below the base model's on the same targets;
- the trainable parameter count equals the GRPO LoRA's (r 16, all-linear; run.json of smoke 13657200);
- the merged checkpoint's sha256 manifest is written and re-verified.
- V5. Grammar absorbed: the Phase 1 bar on the run-1 settings. Read from runs/sft-<job>/summary/eval_sft/junk.json: sft1 without the grammar, prompt v2, seed 2026 × 500, D-019's sampling, the frozen palette. It must be complete and as registered under D-020's list, with the model identified by sft1's manifest sha. Two tests:
- (a) failures.G1 + G2 + G5 ≤ 100. This is the 20 % in the taste_pipeline §6 / D-020 lever rule, so the grammar stays off. Without the grammar the base model had 397 (runs/junk-13656050).
- (b) pass_not_copy.count ≥ 300 (D-019/D-020). Readouts: 64 without the grammar; 446 with it (runs/junk-13640325, dev_gu).
- Ungated readout: the raw in-language count (26 of 500 without the grammar, 500 of 500 with it).

If (a) fails:
1. One more epoch from the epoch-1 adapter, measured once.
2. If it still fails, a grammar smoke on sft1. If its mean |log IS| ≤ 0.05, run 1 = sft1 with the grammar; otherwise it goes to Shresth, and the next fallback is edit-mode rollouts.

If (b) fails while (a) passes: it goes to Shresth.
- V6. No more copying than base + grammar. Read from eval_sft/junk.json against eval_base_grammar/junk.json (same 500 briefs, same palette): copy.any.count for sft1 ≤ max(copy.any.count for base + grammar, 25). Arm B on dev_gu had 54.

If it fails: the SFT job re-runs once with the de-dup at J ≥ 0.85. A second failure goes to Shresth.
- V7. No collapse from the SFT. Read from the same two files' diversity block. Both must hold:
- cell-balanced within-cell mean of 1 − J5 over gate-passing loops: sft1 ≥ 0.9 × base + grammar's (arm B on dev_gu: 0.2263);
- extra exact duplicates within cells: sft1 ≤ max(base + grammar's, 10) (arm B: 34).

If it fails: handled as V6.
- V8. The GRPO path on sft1. Read from runs/grpo-smoke-<job>/summary/smoke.json (configs/grpo_smoke_v2_sft.yaml, no grammar). All of these must hold:
- GRPO_SMOKE_OK;
- is_ratio token-weighted mean ≤ 0.05. This shows vLLM and the trainer run the same merged weights; the unconstrained smoke 13632679 ran 0.0044–0.0091 per step;
- gate rate ≥ 0.8 at step 1;
- 0 reward errors.

If it fails, it is a wiring defect: fix it and re-smoke. It decides nothing about the SFT.
- V9. Phase 1 ear check (needs ear check). Shresth tiers the 50 clips listed in runs/sft-<job>/summary/eval_sft/clips.json (gate-passing, not near-copies). The bar is ≥ 15 Fine or better, CLAUDE.md §9's ≥ 30 %. His result and an APPROVED line go in a gate file (gates/GATE_phase1.md).

If it fails, it goes to Shresth, with CLAUDE.md §9's next fallback.

### Risks

- The SFT can sharpen a distribution that is already concentrated. In arm B the within-cell nearest-neighbour J has median 0.9377, and 34 of 500 are exact duplicates, so "different briefs give different songs" is already weak at step 0.

Guards:
- the J ≥ 0.9 de-dup;
- a single epoch;
- V7 (≥ 0.9 × base + grammar's within-cell distance);
- β 0.04 toward sft1.

Later collapse shows in the D-009 readout and in the hack-watch's within-group distance.
- One epoch over about 1,750 targets may not absorb the grammar (V5a fails). No SFT of this kind has been measured here; the pilot would measure it days earlier.

Branches, in order: a second epoch; then sft1 with the grammar if its IS gap is ≤ 0.05; then edit-mode rollouts.
- The yield is extrapolated from only 37–67 candidates per cell (power-law fits). The thinnest cell, high/wistful, is forecast at about 56 for 1,000 generated. If it lands under 40, V2's refill at J ≥ 0.95, or a second data job, covers it.
- The SFT is palette-specific, so the real data job waits for the freeze. Day 3 is served from Nov 3; infra_schedule plans the freeze for Nov 6 and the Phase 1 gate for Nov 8.

Several Phase 2 inputs must come from sft1: the seed pool's 660 step-0 samples, the RM held-out tuples and PAR's Ref_b. A late freeze or a failed gate therefore delays Phase 2's on-policy coverage. The pilot takes the method risk off that path.
- The copy pool can be contaminated. If model-generated loops (sft_set targets, or the 660 step-0 samples) enter the copy penalty's pool, nearly every completion will sit at J ≥ 0.8 to some pool loop (arm B's within-cell nearest-neighbour median is 0.9377), and P_copy becomes a flat 0.3.

The pool must hold only loops the policy did not write. This departs from taste_pipeline §5's wording.
- Redefining step 0 moves pre-registered claims. If the SFT made loops worse, step N vs sft1 becomes easier to win. V9 (≥ 30 % Fine) and the optional sft1 vs base + grammar A/B stop a weak start from being read as RL progress.
- The near-copy cut is not neutral on content. Dropping J ≥ 0.8 raises the set's f_mood (0.537 in the near-copies against 0.713 in the 0.6–0.8 band), so step-0 knob obedience shifts slightly before any RL. It is reported, not gated.
- The targets come from the token-masked sampler, which renormalizes each step locally. That is not the base model's distribution conditioned on the language; it is the distribution D-020 measured (446 of 500 pass_not_copy).

Also, re-tokenizing a target's text can give different ids from the ones the masked sampler emitted. J1 counts such rows as a readout.
- Merging the LoRA into bf16 weights rounds part of the update away. V5 evaluates the merged weights, not the adapter, and V8 checks that vLLM and the trainer load identical weights. The 8 GB checkpoint lives in $REMOTE_KEEP, on /project2's 1 TB shared by the class.
- TRL 1.17.1's SFTTrainer field names, completion-only masking and end-of-turn handling are unverified, since the laptop has no TRL. J2 checks them against the installed source before loading the model, so a mismatch fails within minutes.
- Without the mask, GRPO can drift back into syntax errors. Three things watch for it: G = 0 on failed loops, the KL term toward sft1, and a per-step in-language readout that keeps the regex as a monitor.
- Three jobs run in sequence between the freeze and run 1, and the CHANGELOG shows queue waits of 2.7–16.5 h. J1 and J2 should be submitted as soon as the freeze lands.
- sft1 bakes in prompt v2 and the frozen palette: any later prompt change needs a new SFT.
- All 9 prompts are identical across brief streams, since the model never sees tempo or tonic. Held-out briefs are therefore separated from training only by fresh samples. GRPO already has this property.

### What Shresth decides

- Whether run 1 starts from an SFT warm-start at all. CLAUDE.md §9 and D-016 make it the next fallback, and D-020 (c) says it is "drafted and taken to Shresth before any job".
- Whether step 0 becomes sft1 for D-009, D-014, the ♥ secondary and every step-0 sample in taste_pipeline (recommended), or stays base + grammar.
- Whether to approve the lofi-sft job type: worst case 1,200 SU (A100, 60 min), expected about 500–700 SU.
- The copy rule. Recommended: a hard cut at J_any < 0.8, the J ≥ 0.9 within-cell de-dup and the 250-per-cell cap, with no reward-term filter. The alternative is a stricter cut; in arm B, J ≤ 0.6 keeps 52 of 500, with 0 or 1 in four cells.
- Whether to run the dev_gu pilot now (≤ 1,686 SU worst case, about 1,000 expected) or wait for the palette freeze.
- Whether the copy penalty's pool should never hold model-generated loops (sft_set targets, step-0 samples), only the 9 v2 examples and Claude-written loops. This changes taste_pipeline §5's pool as written.
- Optional and ungated: a 30-trial blind forced-choice A/B of sft1 vs base + grammar in Phase 4's ablation slot, about 30 minutes of listening.
- Whether to fix the brief-stream registry now, including the RM held-out, dev and final seeds, instead of later before Nov 30.
- Later: approval of run 1's lofi-grpo job type, which is over 500 SU.

## `curated`: SFT warm-start v1: a 210-loop, part-diverse corpus on prompt v2, grammar off; step 0 = the merged SFT checkpoint

**Judge's score:** 6.2. Sub-scores (0-10): syntax without the grammar 7; no copying or collapse 6; final claims 6; SU and Shresth's time 5; implementation risk 4; fallback order 9. Mean 6.2.
- Syntax: its targets have zero repairs and use the model's own layout. The 30 drafts fullmatch their palette's regex (30/30 under draft_day2, recounted). The habit is narrow: keeping only each edit's first grid would lift v2's no-grammar pass from 78 to 363 of 500 (recounted). So 192 examples with an earliest-epoch dev bar should be enough, but that is unmeasured.
- Copying: the only design that keeps copying out of its targets (J5 < 0.6 to everything, no shared A or B line) and gates part-level copying (V2g, at most 125/500). But about 23 targets per prompt for up to 3 epochs invites memorizing the corpus, and the KL to the SFT model would anchor that. V1, V2c and V2e guard it. Its K/D-form diversity bars recompute exactly: within-cell J5 0.7839, 456 distinct of 500.
- Claims: step 0 becomes Claude's style plus his own kept and rejected drafts, which are the RM's positives and negatives. That blurs what RL adds, and the redefinition needs his word.
- Cost: about 590-720 SU, with no job type over 500. But it needs 180 new loops under global pairwise constraints, six times the 30-draft batch that took a 12-agent workflow, written on his Claude plan's usage (D-018) and on the critical path.
- Risk: the corpus itself is the hard, slow part, and it adds memorization risk and a dependency on renaming to the frozen palette.
- Order: D-016's lever 2, consistent with D-018.

### Summary

The grammar can't stay on in GRPO: the IS gap was 0.1538 > 0.05 (runs/grpo-smoke-13657200/summary/smoke.json). Without it, base model + prompt v2 fails syntax on 397 of 500 (runs/junk-13656050/summary/junk.json). Those failures come from one narrow cause, so a small SFT can fix them:
- All 397 G1 failures break one rule: an @N edit line given two grids. 1,658 of 2,990 edit lines carry two or more grids, concentrated in 398 completions, so the error is a habit set early in a loop.
- If each edit line kept only its first grid, the 500 completions would pass 363 times instead of 78.
- What's left is also small: 101 vamp lines without exactly 2 chords, 24 edit lines that write `lock`, and repairable grid lengths.

The real danger for an SFT is copying, not syntax.
- The base model's own grammar-valid loops copy a prompt example's sax line verbatim most of the time (arm B: A 415 and B 398 of 500).
- Only 4 of 2,000 base-model completions survive a strict filter: no copy of any reference, knobs correct, no penalized repair.
- So filtered model samples can't be the corpus; they would teach copying at the part level. The same goes for recombining parts of drafts, and for the 9 prompt examples plus L5 (they are the copy penalty's references).

The proposal:
1. LoRA-SFT Qwen3-4B on 210 targets: the 30 keep3 drafts plus 180 new Claude-written loops, 20 per energy × mood cell.
2. A target is admitted only if it passes the gates with zero repairs, sits in its cell's energy band with mood margin ≥ 2, has J5 < 0.6 against every reference and every other target, and shares no A or B line with any of them.
3. Train on prompt v2's exact messages, in the model's own whitespace layout.
4. Keep the earliest of 3 epoch checkpoints that clears a dev bar for syntax and memorization, and merge it.
5. Re-run the Phase 1 gate on the merged model (no grammar, frozen palette), then a GRPO smoke from it.
6. Run 1 then starts from the merged checkpoint: fresh LoRA, β 0.04 KL to the SFT model, prompt v2, no grammar, and the copy penalty pooled over the corpus.

Why about 20 per cell: syntax needs few examples, but memorization sets the size. With ≥ 23 targets per cell, no single target is more than about 5 % of what that cell's prompt is trained toward. An optional dose ladder measures whether fewer would do.

Step 0 becomes the SFT checkpoint, which needs Shresth's word. The SFT path is 3 jobs, each ≤ 500 SU at worst (about 590–720 SU expected), so none needs his SU approval; run 1 itself still does.

### Method

**What changes**
Only the weights GRPO starts from:
- prompt v2 stays as is (template sha 127e3094…);
- the grammar stays off (D-020 outcome);
- the reward stays as is, except that the copy penalty's pool grows.

**1. Corpus sft/v1: 210 targets**

(a) The 30 drafts in drafts/v1, as written: 10 kept and 20 rejects (gates/GATE_palette.md). No round was marked "none acceptable". So a reject is a draft he heard and liked less than a sibling, which says more about it than about any unvoted new loop.

(b) 180 new loops, 20 per energy × mood cell.
- They are written on the laptop by a Claude Code workflow (D-018: no API key; half the writers see docs/TASTE.md, half only the DSL card, the brief and the admission rules).
- The workflow reuses src/lofi_rl/drafts: personas groove, horn and chords, and DRAFTS.md §2.1's house rules.
- Each loop gets a style card from a Latin hypercube over taste_pipeline §8's axes: drum pocket (4), bass (4), voicing (4), sax approach (4), and the mood's vamps.
- Writer briefs come from a new seed, 1201 (unused in src, tests, docs and configs), taking the first 20 per cell.

(c) Excluded, and why:
- **The 9 prompt-v2 examples (they include v1's example.ls) and l5_wistful_mid.** They are the references P_copy charges, so training on them trains toward exactly what the penalty charges.
- **All 2,000 base-model completions.** Only 4 survive the strict filter (data_from_files), and most of the rest copy a reference's sax line.
- **Mutations or recombinations of the drafts.** They reuse sax lines and grids, which teaches copying at the part level, and whole-loop J cannot see it.

(d) Admission. Every target must meet all of these; the drafts validator is extended to check them, with tests in tests/test_sft_corpus.py:
- G1–G6 pass with zero repairs, free ones included, under the run palette in the cell's mood;
- it compiles with no sax fold;
- E falls in the cell's energy band, and the mood margin is ≥ 2;
- P_sax 0 and rubs 0;
- J5 < 0.6 against the 10 reference loops and against every other target, where J5 is the character 5-gram Jaccard of the K/D copy form (DRAFTS.md §5);
- no A or B line is equal to another target's or a reference's;
- at most 660 characters in the model's layout (about 300 tokens).

(e) Coverage per cell:
- at least 6 distinct vamps;
- every keys instrument and both kits;
- at least 3 voicings;
- span 1 and span 2;
- bass lock and a grid bass;
- snare and rim;
- 4–7 edits.

(f) Layout: each target is written the way the base model writes. That means two spaces before every newline, no blank lines, no final newline and no `sax:` line. SFT then spends nothing on moving whitespace tokens, and the parser reads this layout with no repair (GRAMMAR.md §3.3).

(g) Pairing:
- **Prompt:** prompt_v2.build_messages(cell brief, run palette). These are the exact messages run 1 sends for that cell, including the example from the same energy and the next mood.
- **Completion:** the target text plus the chat template's end-of-turn token.
- That gives 9 prompts, with 23 targets each (26 for low/wistful).

(h) Split: 2 new loops per cell are held out (18 in all), chosen by a keyed hash on seed 1201. Training uses 192, including all 30 drafts.

(i) Build: `python -m lofi_rl.sft.build`, a new module. It writes sft/v1/{targets/*.ls, corpus.jsonl, manifest.json}, which is LoopScript text only and gets committed. The manifest holds each target's sha256, the palette sha, the template sha, the 9 messages_sha256 and the coverage table.

(j) Palette:
- The loops are written now in draft_day2's names (rhodes, wurli, piano; studio, jazz).
- They are built against the frozen palettes/lofi_rain_v1.yaml, with a declared rename table if the freeze renames anything.
- Training waits until the 9 messages_sha256 under the frozen palette match the manifest.

**2. Training job: jobs/sft.sbatch** (lofi-sft, MODE=policy)

Model and loss:
- Base: Qwen3-4B-Instruct-2507 @cdbee75f, bf16, gradient checkpointing.
- LoRA: r 16, α 32, dropout 0.05, all-linear (the same modules as GRPO).
- Loss on the completion only, with our own tokenization:
  - input ids = apply_chat_template(prompt, add_generation_prompt=True), checked equal to what GRPO's trainer builds;
  - then the target's ids and <|im_end|>;
  - labels −100 on the prompt.

Schedule:
- lr 1e-4, constant after 5 warmup steps. It is constant so that the epoch-k checkpoints are prefixes of one schedule.
- Effective batch 8, 3 epochs = 72 steps, about 945k tokens (estimate).
- Max length 2048, seed 596.
- Adapters are saved after epochs 1, 2 and 3.

Pre-flight checks, which fail closed before any training:
- every config field exists in the installed TRL and transformers;
- for one example per cell: the prompt ids equal GRPO's, the mask covers exactly the target plus <|im_end|>, and there is no <think>;
- the dataset sha equals the manifest's.

Dev eval, for each checkpoint:
- merge in fp32 and save as bf16 to scratch;
- vLLM (bf16, eager, no grammar) generates 270 completions: the first 30 briefs per cell of brief.sample_briefs(3001, ·), a new seed;
- sampling: T 1.0, top_p 1, top_k 0, 512 tokens, prompt v2;
- scoring: junk.score_row, imported, plus readouts of copying from the corpus: J5 against the train and held-out targets, and verbatim A and B lines.

Selection, pre-registered:
- Take the earliest epoch with dev G1+G2+G5 ≤ 27/270, pass_not_copy ≥ 189/270 and near-copies of a train target (J5 ≥ 0.8) ≤ 27/270. The earliest is the one that fixes syntax with the least drift from the base.
- It goes to $REMOTE_KEEP/ckpt/sft-v1/ with its tokenizer, its config and a sha256 manifest.
- If no epoch qualifies, nothing is merged.

Outputs: summary/sft.json holds the data sha, the settings, the versions, the train and held-out loss every 6 steps, each epoch's dev readouts and the selection.

**3. Code** (laptop, tests first)
- src/lofi_rl/sft/{build,train,eval}.py;
- the drafts briefs and check modules generalized to N briefs per cell, with the rule on shared lines;
- a local `model: {path, manifest_sha256}` option in rl/config.py, rl/train.py and rl/junk.py, failing closed on any sha mismatch;
- a corpus-copy block (`--pool`) in junk's score step;
- configs/grpo_smoke_sft_v1.yaml and configs/grpo_run1.yaml.

**4. Phase 1 gate, re-run on the merged model**
- This follows D-020: the gate is re-run with the settings run 1 actually uses.
- junk-rate: 500 completions on brief.sample_briefs(2026, 500), prompt v2, no grammar, D-019's sampling, the frozen palette.
- It renders 50 passing loops that are not copies, for his ear check.

**5. GRPO smoke from the checkpoint**
It uses grpo_smoke.yaml plus:
- the model's path and manifest sha;
- the frozen palette;
- prompt v2;
- copy_penalty pool [drafts/v1, sft/v1/targets];
- no structured_outputs.

**6. Run 1**
- Model: the merged checkpoint, with a fresh LoRA r 16.
- KL: β 0.04 to the adapter-off model, which is the SFT model. Continuing the SFT adapter instead would put the KL reference on the base model and pull the syntax back.
- Prompt v2 and no grammar. taste_pipeline §6 turns the grammar on only above 20 % parse failures.
- P_copy against 219 references: the 9 v2 examples, the 30 drafts and the 180 new loops.
- Briefs: seed 596, 400 rows; 100 steps, saved every 25; §6's settings otherwise.
- It is registered in D-021 with the checkpoint's sha before any run-1 job.

### Data from the files

Counts measured on the laptop on 2026-11-01. The scratch outputs are in <session scratchpad>/run1design/sft-warmstart/, as out/analysis.json, out/analysis2.json, out/eligible_pool.jsonl and drafts_check.json.

**drafts/v1**
- 30 drafts: 3 per cell, 6 in low/wistful (d02 and d10).
- `lofi_rl.drafts.check`: 30 of 30 ok.
  - J5 against the references: 0.271–0.419.
  - J5 to the nearest other draft: 0.305–0.410.
  - theory 0.99–1.00; P_sax 0; rubs 0.
- Kept 10: chords 5, horn 3, groove 2 (gates/GATE_palette.md).
- Pairwise J5 over 435 pairs: p10 0.216, p50 0.266, p90 0.325, max 0.410.
- 30 distinct A lines and 30 distinct B lines; none equals a reference's.
- 28 distinct vamps. Voicings: spread 7, shell 9, rootless 8, quartal 6. Snare 22, rim 8. Bass lock 3.
- Grids: 91 of 485 also appear in a reference, and 166 in another draft.
- Under dev_gu, 15 of 30 fail G4 (`studio`), and 12 of the other 15 need alias repairs. So the corpus needs the frozen palette's names.
- In the model's layout the drafts are 465–638 characters, about 211–290 tokens at arm B's 2.204 characters per token.

**References**
- 10 distinct loops: the 9 v2 examples plus L5.
- Pairwise J5: p50 0.321, max 0.440.
- All 9 v2 examples build under palettes/draft_day2.yaml (0 refused).

**Base-model samples**
Source: the 2,000 completions in runs/junk-13632676, junk-13640324, junk-13640325 and junk-13656050.
- Passing: 98, 494, 500 and 78.
- The strict filter keeps 0, 3, 1 and 0, which is 4 unique loops. It requires all of: p_fix 0, energy_match and f_mood both true, K/D J5 < 0.6 against all 10 references, and no A or B line equal to a reference's.
- Without the knob condition: arm B 40, arm A 68.
- A or B line equal to a reference's:
  - arm B: A 415, B 398 of 500 (junk.json verbatim_passing: A 415, B 398, AB 392);
  - arm A: A 347, B 399 of 494.

**Without the grammar** (junk-13656050)
- Failures: G1 397, G6 23, G4 2. All 397 G1 details are "@N … takes at most 1 grid(s)".
- 1,658 of 2,990 edit lines have two or more grids, spread over 398 completions.
- 101 of 500 vamp lines don't have exactly 2 chords. 24 edit lines write `lock`.
- Repairs, counted in loops: snare_and_rim 176, pad_grid 141, trunc_grid 74, misplaced_bar 60, pad_phrase 50, trunc_phrase 18.

**Counterfactual: keep only the first grid of each edit line**
- pass 363. Remaining failures: G6 105, mostly the vamp; G1 24 (`lock`); G4 8 (m7sus).
- 46 near-copies of the shown example.
- 308 of 363 have an A or B line equal to a reference's.

**Line endings**
- arm B: 7,717 of 7,792 newlines are two-space; 65 blank lines before A; 500 of 500 end with no newline; 0 `sax:` lines.
- junk-13656050: 7,791 of 7,828; 37; 500 of 500; 0.

**Diversity**
- arm B, exact over all pairs, K/D form: all-pairs mean J5 0.3434, within-cell 0.7839, across-cell 0.2884; 456 distinct of 500.
- arm A, from sampled pairs: within-cell 0.736, across-cell 0.657.

**Knobs** (arm B, from junk.json): energy_match 498 of 500, f_mood 333 of 500.

**Tokens** (arm B): completions p50 257.5, p99 312, max 316; prompts 1,322–1,408.

**GRPO**
- Smoke 13657200: IS 0.1538, per step 0.105–0.257; P_copy mean 0.172, with 289 of 320 completions penalized; steps 35.5–39.9 s; 243.0 SU.
- Smoke 13632679, no grammar: IS 0.0044–0.0091; 280.7 SU.
- arm B's step-0 P_copy mean is 0.1754 (docs/GRPO.md §2).

**Generation speed and cost**
- arm B: vLLM loaded in 23.1 s and generated 500 completions in 50.6 s (generate.json).
- junk-rate runs cost 80.7–99.0 SU.
- SU used so far: 1,481.9 (CHANGELOG).

### Step 0

**What step 0 is under this design.** It is the selected SFT checkpoint, merged into the base weights. It is sampled exactly as step N will be: prompt v2, no grammar, T 1.0, top_p 1, top_k 0, 512 tokens.

It is three things at once:
- GRPO's initial policy;
- its KL reference (β 0.04, adapter off);
- the arm the three final claims compare against.

**The three claims, read this way:**
- **Preference:** he picks step N over step 0 at least 70 % of the time. This measures what GRPO adds on top of SFT.
- **Obedience:** at least 90 % at step N.
- **No collapse:** step N keeps at least 70 % of step 0's distance across briefs. Gate V2e keeps step 0 itself from being a collapsed baseline: it must be at least as diverse as base + grammar.

SFT's own effect is reported separately against the base model, from the junk readouts: base + v2 without the grammar has pass_not_copy 64/500, and base + grammar has 446/500, but the grammar is off.

**Other places "step 0" is used.** taste_pipeline §3–4 uses it in four places, and all must come from this checkpoint:
- the RM held-out set ("first 4 step-0 samples that pass the gate");
- PAR's Ref_b, 4 step-0 samples per brief;
- the checkpoint quads;
- the seed pool's 660 step-0 samples.

So SFT has to land before the first bws4 vote, planned for Nov 9.

**The Phase 1 gate** is read on this checkpoint too (V2, V3). It is "the chosen settings re-run on the frozen palette" (D-020), now that the fallback order (D-016) has moved to SFT.

**This needs Shresth's word.**
- D-014 and D-009 are his decisions, made while run 1 was to start from the base model (D-019, D-020).
- The corpus contains his kept drafts and Claude's loops, so the meaning of the claim changes.
- The alternative, step 0 = base model (+ grammar), would credit SFT's gains to RL. I recommend against it.
- Optional: a blind A/B of about 30 trials, SFT step 0 against base + grammar, to document what SFT did by ear.

### Jobs and SU

Rates (CLAUDE.md §4): GPU 8 SU/min, CPU 1 SU per core-minute, memory 1 SU per 4 GB-minute.

**1. lofi-sft** (new type, jobs/sft.sbatch, MODE=policy)
- Resources: gpu partition, 1 A100-80GB (--gres=gpu:a100:1 --constraint=a100-80gb), 4 CPU, 32 GB, --time 00:25:00.
- Rate: 8 + 4 + 8 = 20 SU/min, so the worst case is 500 SU.
- Expected: about 13–17 min, about 260–340 SU. This is an estimate: about 945k training tokens at an assumed 2,000–4,500 tok/s is 3.5–8 min; three merge + vLLM evals take about 2 min each (vLLM load 23.1 s and 500 completions in 50.6 s, from arm B's generate.json); startup takes 2–3 min.
- At ≤ 500 SU, Claude registers it (make approve MAX=500 BY=Claude).
- A 48 GB card would need 35 min, which is 700 SU, so that route needs Shresth's OK.

**2. lofi-junk-rate with a MODEL argument**
- This is a changed type, so Claude re-registers it (≤ 500 SU).
- Resources: an A40 with -p gpu as in D-020, 4 CPU, 24 GB, 25 min.
- Rate: 4 + 6 + 8 = 18 SU/min, so the worst case is 450 SU.
- Expected: 80–100 SU (measured 80.7, 86.4, 93.0 and 99.0).

**3. lofi-grpo-smoke**
- The script is unchanged (sha 1dded12d386c7fab, max 500); only a new configs/grpo_smoke_sft_v1.yaml is added.
- Resources: A100 for 25 min, or L40S through OVR, both at 20 SU/min, so the worst case is 500 SU.
- Expected: about 245–280 SU (measured 243.0 and 280.7).

**Optional**
- A dose-ladder lofi-sft: adapters trained on 30 and on 120 targets, each evaluated on the same 270 dev prompts. At most 500 SU, about 200 expected (estimate).
- A contingent second round: at most 500 + 450 SU, about 430 expected.

**Totals before run 1**
- Expected: about 590–720 SU. Worst case: 1,450 SU (plus 500 for the ladder and 950 if a second round is needed).
- No SFT-path job exceeds 500 SU.

**Run 1** (lofi-grpo, 100 steps) does exceed 500 SU and needs Shresth's yes; this design doesn't change that.
- About 1,420 SU on an L40S with the symbolic reward only (CHANGELOG 2026-10-31).
- About 2,300 ±35 % SU on an A100, or about 3,200 ±35 % on an L40S, with the scorer server (taste_pipeline §6; these are estimates).
- About 1,481.9 SU of the 1,000,000 target have been used so far.

### Validation gates

- V0 Corpus. Read from sft/v1/manifest.json; checked by tests/test_sft_corpus.py.
- 210 targets admitted: 30 drafts and 180 new, with ≥ 23 per cell.
- Every target: zero repairs, compiles, in its cell's energy band, mood margin ≥ 2, P_sax 0, rubs 0.
- Max pairwise J5 < 0.6, and max J5 against the 10 references < 0.6.
- 0 shared A or B lines; the coverage rules are met in every cell.
- The 9 messages_sha256 equal those built under the frozen palette.
- V1 SFT job. Read from runs/sft-<job>/summary/sft.json.
- The pre-flight passed: the prompt ids equal GRPO's for all 9 cells, and the loss mask is exactly the target plus <|im_end|>.
- A checkpoint was selected by the dev rule (270 completions on seed 3001): G1+G2+G5 ≤ 27, pass_not_copy ≥ 189, and near-copies of a train target ≤ 27.
- The checkpoint's manifest sha is recorded.
- V2 Validity. Read from runs/junk-<job>/summary/junk.json.
- n = requested = 500; partial false; unreadable 0; stopped null.
- The model is V1's manifest sha.
- prompt v2 template 127e3094…; rows_matching 500; example_mismatch 0.
- The palette sha is the frozen palette's.
- constrained.enabled is false, and the sampling is D-019's.
- V2a. pass_not_copy.count ≥ 300, from junk.json. This is D-019's bar and the first half of the Phase 1 gate.
- V2b. failures.G1 + G2 + G5 ≤ 100, from junk.json. That is the 20 % of taste_pipeline §6, so the grammar is not needed.
- V2c. Near-copies of any corpus target (J5 ≥ 0.8, K/D form) ≤ 50 of 500. Read from junk.json's new sft_copy block.
- V2d. Mean P_copy over the 500, against run 1's 219 references, ≤ 0.0877. That is half of arm B's 0.1754 (docs/GRPO.md §2). Read from junk.json's sft_copy block.
- V2e. Diversity no worse than base + grammar (runs/junk-13640325). Read from junk.json's sft_copy block.
- All-pairs mean J5 ≤ 0.3434.
- Within-cell mean J5 ≤ 0.7839.
- Distinct copy forms ÷ passing ≥ 0.912.
- The committed tool re-derives these thresholds into D-021 before the run.
- V2f. Knobs among passing loops: energy_match ≥ 0.946 and f_mood ≥ 0.616. These are arm B's 0.996 and 0.666, each minus 0.05. Read from junk.json knobs_passing.
- V2g. At most 125 of 500 completions have an A or B line equal to a corpus target's or a reference's. This is the part-level copy gate; arm B had 415 for A. Read from junk.json's sft_copy block.
- V3 Ear check, the second half of the Phase 1 gate (CLAUDE.md §9). At least 15 of 50 rendered clips (passing, not copies) are "fine". Shresth records it in the Phase 1 gate file in gates/ with his APPROVED line.
- V4 GRPO smoke from the checkpoint. Read from runs/grpo-smoke-<job>/summary/smoke.json.
- ok true (GRPO_SMOKE_OK).
- is_ratio token-weighted mean ≤ 0.05.
- Both launches' run.json show V1's manifest sha and no structured_outputs.
- In launch a: gate rate ≥ 0.75 over its 320 completions, and p_copy_mean ≤ 0.0877.
- Outcome rule, pre-registered in D-021.
- **All pass:** run 1 starts from the checkpoint.
- **V1, V2a or V2b fails:** one second round, registered first as an amendment. It adds 90 Claude loops aimed at top_details, plus the SFT model's own samples that pass V0's admission rules. If that fails too, edit-mode rollouts come next (D-016).
- **V2c, V2d, V2e or V2g fails while syntax passes:** take the earlier epoch if it met V1's syntax bars; otherwise one retry at lr 5e-5 for 2 epochs; then Shresth decides.
- **V2f or V3 fails:** Shresth decides.
- **V4 fails:** no run 1 until the merged weights are fixed.

### Risks

- The corpus may be hard to finish. 180 new loops that each pass with zero repairs, keep J5 < 0.6 against everything else and share no sax line with anything are about 6× the drafts batch. The global constraints get harder as the corpus grows, and Shresth's Claude plan limits may bind (D-018). Mitigation: write in shards checked with --against. The dose ladder shows whether fewer would do.
- The model may memorize the corpus and collapse onto a few targets per cell. The KL to the SFT model would then anchor that collapse through GRPO. This is why the V1 selection and V2c, V2d, V2e and V2g come before run 1.
- Copying at the part level is invisible to whole-loop J; for example, a corpus sax line placed over new drums. V2g gates it. GRPO's P_copy remains whole-loop only.
- Style shift: step 0 will follow Claude's writing in the corpus rather than the base model's own habits. Only Shresth's ear can judge it (V3). The RM's PAR baseline moves with it.
- Redefining step 0 has knock-on effects. The RM held-out set, PAR's Ref_b, the checkpoint quads and the seed pool's 660 step-0 samples must all come from the SFT checkpoint. If SFT slips past the first bws4 vote (planned Nov 9), those artifacts have to be rebuilt.
- The palette dependency: training waits for the freeze. If the freeze renames something, a declared rename table is applied. If it drops an instrument, the loops that use it are remapped and validated again.
- Nothing about TRL, transformers or the Qwen3 chat template can be checked on the laptop (no torch or TRL there): the <think> and <|im_end|> handling, and the loss mask. Mitigation: we tokenize ourselves, and the pre-flight checks compare the prompt ids with GRPO's and fail closed.
- Merged weights could differ between vLLM and the trainer through dtype or merge rounding, which would show up as an IS gap. Mitigation: merge in fp32 and save one bf16 copy that both load; V4 checks it.
- Draft labels and the reward could interact. The 20 rejects are RM negatives, and the 10 kept drafts are RM positives that the reward might later favor near-copies of. All 30 drafts are in the P_copy pool.
- Prompt v2 still shows an example, so the copy channel stays open. SFT teaches against using it, but GRPO could relearn copying if other rewards outweigh P_copy. Hack-watch logs the copy readouts.
- The policy sees only the cell: 9 distinct prompts, all of them trained on. "Held-out" briefs differ only in tempo and tonic, which the model never sees. MERT distance across briefs will include tempo and key effects, and only the symbolic V2e looks at the LoopScript itself.
- Queue waits: 2.7–11.5 h each were observed, and three jobs run in sequence, so about 1–2 days pass before run 1.
- Storage: an 8 GB merged checkpoint goes on /project2, which is 1 TB shared by the whole class.
- The energy bands are provisional (VERIFIERS.md §7 item 9). If they are replaced by the pool's tertiles, some corpus targets may no longer sit in their cell's band and must be validated again.
- Housekeeping: my first analysis run did not use -B, so it probably refreshed 9 bytecode caches under src/lofi_rl/**/__pycache__. Those are gitignored and excluded from rsync. Nothing else was written in the repo.

### What Shresth decides

- Whether to approve the SFT warm-start as run 1's start. It is D-016's next lever, and D-020 requires his OK before any job. If he approves, it is registered as D-021.
- Whether step 0 is the merged SFT checkpoint, for all three final claims and for the RM's step-0 artifacts (recommended), with the base model's readouts reported separately; or whether step 0 stays the base model (+ grammar).
- Whether the corpus includes the 20 rejected drafts (recommended) or only the kept 10. The 9 reference loops and L5 stay out as targets (recommended).
- Whether 180 new Claude-written loops fit his plan's usage, with D-018's half-and-half TASTE.md split; or whether to start with 90 and the dose ladder.
- When to train: after the palette freeze (recommended), or now on draft_day2's names if he confirms the frozen palette keeps those names and lists.
- Listening: the 50-clip "fine" check on the SFT model, about 0.3 h, which is the second half of the Phase 1 gate. Optionally, a blind A/B of about 30 trials, SFT step 0 against base + grammar.
- Whether to run the optional dose-ladder job (about 200 SU expected, at most 500).
- Approval of run 1's lofi-grpo SU, which is over 500. This design doesn't change that.

## The referee's problems with the judge's draft (all applied in run1_start.md)

- **[major]** This step 0 makes D-009's no-collapse claim close to automatic, and nothing guards TASTE.md's 'different briefs should give different songs' inside a cell. A v2 prompt depends only on the cell, so every brief of a cell gets the same messages. Step 0 is already near-collapsed there: the nearest same-cell neighbour sits at J5 median 0.9377, with 448 of 500 at 0.8 or more (referee recount, <session scratchpad>/run1design/referee/verify1.py over runs/junk-13640325). D-009 averages over all pairs of 100 briefs, and only 11.1 % of those pairs share a cell. Replacing every cell's loops with that cell's step-0 medoid keeps 0.938–0.963 of the across-brief mean 1 − J5 over 200 random 100-brief draws, against a 0.70 bar (verify2.py in the same folder; a symbolic proxy for the embedding distance, an estimate). No reward term sees two briefs of one cell together: P_group sees one brief's 8 completions, and item 2 keeps model output out of the pool. The entry notes that 'the model sees only the cell' but draws no consequence from it, and 'What Shresth decides' does not tell him.
  - **Fix:** Leave D-009 as he registered it, and add three things. (a) A within-cell, across-brief diversity readout: per 5-step window in run 1, and at the final evaluation beside D-009's number (K/D form; mean 1 − J5 and distinct forms per cell; step N vs step 0). (b) A hack-watch alarm when the window's within-cell mean J5 across briefs stays above its step-0 value for 3 consecutive windows. (c) A plain line in 'What Shresth decides': D-009 cannot see a collapse inside a cell, and turning the readout into a claim needs his word.
- **[major]** Item 3's part-copy alarm ('the share above its step-0 value for 3 consecutive 5-step windows') can hardly fire. The step-0 share is already 421 of 500 = 0.842 (junk.json verbatim_passing: A 415, B 398, both 392). The hack it is meant to catch is P_copy evaded by rewriting grids while keeping the example's sax lines, and that path is already in step 0: 8 of the 52 loops at J5 ≤ 0.6 (P_copy 0) carry a verbatim A or B line, 7 of them both (laptop count over scored.jsonl). The alarm's step-0 value is also arm B's, on dev_gu, while run 1 uses the frozen palette.
  - **Fix:** Alarm on the evasion signature instead: for 3 consecutive 5-step windows, the verbatim A-or-B share stays at or above its step-0 value while the mean J5 with the shown example falls at least 0.1 below its step-0 value (arm B, for scale: 0.842 and 0.7115). Take every step-0 value from the official Phase-1 gate run on the frozen palette (same seed and settings), not from arm B.
- **[major]** Part A's path rests on D-020's official Phase-1 gate, and part-level copying can carry that gate. D-019 set pass_not_copy so that 'copying the example cannot carry it', but its near-copy test is whole-loop J5 ≥ 0.8. Of arm B's 445 candidates under J_any 0.8, 367 carry the shown example's A or B line verbatim (339 both), and 86.6 % of their grids are the example's (6,385 of 7,376; laptop recount). The 50 ear-check clips are drawn from these passing non-copies, so his 'fine' count mostly judges Claude's example sax lines over base-model variations.
  - **Fix:** Keep D-020's bar unchanged: moving it now would move a registered bar. Before he writes the APPROVED line in gates/GATE_phase1.md, report three things beside it: the verbatim A-or-B share among the passing non-copies, the same share among the 50 clips, and his 'fine' count split by whether a clip carries a verbatim line.
- **[major]** With this step 0, the final A/B (D-014) is partly a comparison with Claude's example lines, and its blinding is weaker. The step-0 arm will mostly replay the 9 shown examples' sax lines (A or B in 421 of 500, both in 392; arm B). With about 11 final briefs per cell, he hears the same phrase and answer many times from one arm, so the arms may be told apart by ear. Eight of the 9 examples are shootout loops (DRAFTS.md §5). P_copy trains step N away from exactly these loops. The entry records no copy status per trial and does not limit repeats, and 'What Shresth decides' does not mention any of this.
  - **Fix:** Keep D-014 as registered and add reported readouts. For each trial, log both arms' copy readouts: J5 to the shown example, to the pool and to any SFT target, and verbatim A/B. Report the win rate on the trials whose step-0 clip carries no verbatim line of its shown example. Order the trials so that no two consecutive trials share a cell. State the issue in 'What Shresth decides'.
- **[major]** The miss rule lumps invalid launches together with measured misses: 'Any of M6–M11 fails (after at most one recorded re-run): D-020's verdict stands'. The xgrammar API names are UNVERIFIED, and T2 may find a different TRL method body, so a crash or a failed source check on the first launch is likely. Under this rule, a second implementation defect would end lever 1 for a reason unrelated to the IS question. D-020's own rule kept the two apart: an invalid arm was fixed and re-run, and an arm invalid twice went to Shresth. M10's first clause (mask/neglogz ≥ 0.05) is also a check that the records are consistent, not a reading: 13657200's 0.1538 already implies a token-weighted −log Z near 0.15.
  - **Fix:** Split the gates into two kinds. An invalid launch (M5–M8, or M10's first clause) measured nothing: its defect is fixed, the fix is recorded here, and the smoke runs again. A launch invalid twice goes to Shresth. A measured miss (M9, M10's ratio or M11) on a valid launch gets at most one re-run, with the fix named now; after that, D-020's verdict stands. Count the SU: at most three launches, 1,500 SU worst case.
- **[major]** Item 5's run-1 STOP rules (any mask violation; a per-step gap above 0.05; an IS-weight mean below 0.3 for 3 steps in a row) say when run 1 stops, but not what follows. They do not say whether it resumes, from which checkpoint, after what check, or when D-020's verdict takes over.
  - **Fix:** A STOP ends run 1 at its last checkpoint (one every 25 steps), and nothing resumes until an amendment here names the cause and the fix. Any of the three rules marks a trainer defect: the masked smoke runs on the fix (at most 500 SU) and must meet M5–M11, and run 1 then resumes from the last checkpoint before the first bad step. If the smoke misses, D-020's verdict applies to run 1 and Part B goes to Shresth. A hack-watch alarm follows taste_pipeline §7's On-alarm list.
- **[major]** Part B has no failure outcome. D-016's order is grammar, SFT warm-start, edit-mode rollouts, then stop. The entry says nothing about B1–B6 failing, or about no saved adapter passing the dev bars ('the earliest save that passes the dev bars is merged'). It also does not say what happens if he says no now.
  - **Fix:** Part B fails if no save passes the dev bars, or if any of B1–B6 still fails after at most one recorded re-run. D-016's lever 3, edit-mode rollouts, is then drafted and taken to him before any job; if that fails too, D-016's last step, stop, goes to him. If he declines Part A now, Part B is registered in full as the next entry and brought to him, and nothing runs.
- **[major]** Part B's corpus 'holds the 30 drafts', which includes the 20 he rejected in keep3. The kept 10 are d01, d02, d03, d07 and d09 (chords), d04, d06 and d08 (horn), and d05 and d10 (groove) (CHANGELOG 2026-11-01 13:38). TASTE.md (Labeling) makes the rejects labeled negatives, and taste_pipeline §8 labels each kept loop ≻ its rejects. Training the SFT model to reproduce the rejects goes against his own votes.
  - **Fix:** Use only the 10 kept drafts, plus the 180 new loops. That gives at least 21 targets per cell (22 in low/wistful, which has two draft briefs). The 20 rejects stay in the copy pool and the RM labels, but never in the SFT corpus. Change 'about 23 targets per prompt' to 'about 21'.
- **[minor]** The only fix named for a noise-only miss, cast_lm_head_to_fp32: true, may be refused. TRL documents that flag for untied embeddings only (UNVERIFIED here: there is no TRL source on the laptop). Qwen3-4B-Instruct-2507's config has tie_word_embeddings: true, both in the cluster copy read on 2026-10-30 (<session scratchpad>/remote_qwen/config.json) and in the cached ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25/config.json. A re-run built on that flag could stop at startup and use up the only re-run.
  - **Fix:** Name a fix the override controls: compute the masked log-softmax in fp32 inside _get_per_token_logps_and_entropies, in all three passes. It applies only when M8 holds, so the mask matched and the miss is numerical. Use cast_lm_head_to_fp32 only if a run-time read of TRL's source shows it accepts tied embeddings.
- **[minor]** The draft says '13657200's record shows gpu_model_runner.py line 4335 before 3437 and 3448'. That misreads the record. run.json's vllm_logprobs block lists line numbers (4335 comes after 3437 in the file), says checked: false, and notes that it did not check whether the log-probs include the mask. Line numbers do not show call order. The evidence for the claim is the step-1 comparison: 0.1146 against 0.0087.
  - **Fix:** Say that the record lists the lines and does not show their order, and rest the claim on the two step-1 gaps. Have V1 read, inside the method that holds line 4335, that apply_grammar_bitmask runs before self._sample. M10 is the test.
- **[minor]** Two numbers equal 'the largest weight's share of the step's 32' only if TRL's logged IS mean is a plain mean over the 32 sequences: the Why table's row and M11's 'max ÷ (32 × mean)'. T3 has not confirmed this yet. Every value is at most 1.000, which fits but does not prove it. The sentence 'learned mostly from one completion per step' also overstates the data: the share was at least 0.9 in 7 of 15 steps, and 0.319–0.868 in the other 8 (referee recount of both smokes' steps.jsonl).
  - **Fix:** Have the override log the step's 32 per-sequence weights (is/seq_weights), and read M11 from them: mean ≥ 0.3, and largest ÷ sum ≤ 0.25. Label the table row 'max ÷ (32 × mean)', and say 'in 7 of 15 steps one completion carried at least 90 % of the weight'.
- **[minor]** The entry overrides registered text without saying so. Item 2 narrows the copy pool, but D-020's Before-GRPO list registered the copy penalty as 'scored against the seed pool', and that pool holds the 660 step-0 samples (taste_pipeline §8); the entry names only taste_pipeline §5. A pass would also overturn D-020's IS-half verdict, which the entry implies but never states. 'Settings: D-020's run-1 settings, unchanged' sits next to a new trainer key.
  - **Fix:** Add a 'supersedes' list under Decided-by: D-020's IS-half verdict (only if Part A passes), D-020's 'against the seed pool' clause, and taste_pipeline §3's held-out seed and briefs file. Write 'D-020's run-1 settings, plus trainer_logprobs: grammar_masked'.
- **[minor]** Diversity is measured two ways under one name. Item 4 uses junk's copy form (canonical text without sax), balanced by cell: 1 − J5 = 0.2263, with 34 extra duplicates. B4 uses the K/D copy form, weighted by pair: mean J5 0.7839, with distinct ÷ passing 0.912 (44 duplicates). Both are right for arm B (referee recount, verify1.json in the referee scratch folder), but a reader cannot tell B4's bar from item 4's readout.
  - **Fix:** Name both forms once in item 4 and give arm B's values in each. Canonical form: cell-balanced 1 − J5 0.2263, pair-weighted mean J5 0.7743, 34 duplicates, 466 distinct. K/D form: 0.2158, 0.7839, 44 duplicates, 456 distinct (0.912). Use the K/D form, which ignores instrument names, wherever palettes differ (B4, and run 1 against step 0). B4 cites item 4.
- **[minor]** Nothing re-checks the masked trainer under run 1's own regex (the frozen palette's), reward (with the scorer on) and copy pool before lofi-grpo. CLAUDE.md §9 already plans a Phase-2 GRPO smoke, but the entry does not hold it to M5–M11. The SU list leaves out that smoke: taste_pipeline §6 puts it at about 550 SU on an A100 (UNVERIFIED), which is over 500 and needs his yes. It also leaves out PAR's Ref_b, 1,600 step-0 samples for run 1's 400 briefs, which has no job type yet. Run 1's code path has no freeze or per-job snapshot either; CHANGELOG Status lists the snapshot as open.
  - **Fix:** Add to What follows: Phase 2's GRPO smoke (frozen palette, scorer on, run 1's pool) must meet M5–M11 under the same two-kind rule before lofi-grpo is asked for. Run 1's code path is frozen, or snapshotted per job. Put the smoke's shape and Ref_b's job in the lofi-grpo ask.
- **[minor]** The expected run-1 SU is understated. The entry quotes taste_pipeline §6, which budgets 9 s of reward per step. The CHANGELOG's own scorer sizing for a grammar step (2026-11-01 15:05) is about 12 s of rendering on 7 workers plus about 7 s of features for 32 passing completions. Adding that to 13657200's 37.63 s L40S step (symbolic reward only) gives about 57 s a step. At 36 SU/min, with 100 steps and 7 min of startup, that is about 3,700 SU on an L40S, and about 2,900 on an A100 by the same arithmetic on §6's A100 step (estimates). §6 says 3,200 and 2,300. Both new figures still sit inside §6's worst cases (5,400 at 2:30; 3,780 at 1:45).
  - **Fix:** Quote the estimate from measured parts beside §6's in Jobs and SU. Have M12's step time and Phase 2's smoke replace both figures before the ask.
- **[minor]** The independent check's laptop numbers come from the cached mlx-community/Qwen3-4B-4bit tokenizer.json, not the pinned snapshot's: 980 completions with 0 violations, at most 52 allowed tokens, 7.3–7.5 % forced, and 361 of 500 token counts matched. The same holds for GRAMMAR.md §3.3. The referee re-ran the masked-trainer scratch scripts and got the same results (<session scratchpad>/run1design/masked-trainer/tokmask.py and tokmask_points.py). These are estimates of xgrammar's bitmask, not the bitmask itself. M8's NFA-against-bitmask comparison also needs the same stop ids as xgrammar's TokenizerInfo and the same vocabulary size (151,936 rows against 151,643 regular tokens).
  - **Fix:** Build the cluster NFA from the pinned snapshot's tokenizer.json, with the vocabulary size and stop ids V2 reads, and label the laptop numbers as estimates. On cost, the check is no walltime risk: the laptop NFA took 0.134 s for a cold 32-completion step and 0.013 s for a warm one (tokmask_timing.py, re-run).
- **[minor]** If T2 finds TRL's sampling_mask usable, the grammar mask is ANDed into it 'instead'. So the code that runs is chosen at run time from two paths, and neither the records nor the gates say which one ran.
  - **Fix:** run.json names the path that ran, M1's tests cover both paths, and M8 requires the record.
- **[minor]** Item 1 has four gaps. It replaces taste_pipeline §3's held-out seed (596) and its data/briefs/briefs_v1.jsonl without saying so, and data/ is never committed, so that file could not hold a pre-registration. It fixes the briefs but not the vLLM seeds of the draws that §3 relies on ('first 4 step-0 samples', '≤4 seeded draws'). It does not record exact twins: in stream order, 8 of seed 5101's first 40, 4 of 5102's first 40 and 15 of 5103's first 100 have the same energy, mood, bpm and tonic as a brief in seed 596's [0, 400) (laptop count with brief.sample_briefs). And its per-cell stratification counts are left unstated.
  - **Fix:** State the supersession. In the file, fix the vLLM seed of the k-th draw for brief i of each stream, and the per-cell counts and order (4 or 5 per cell for 40, 11 or 12 for 100, 30 for 270). List the twins but exclude none: the policy never reads bpm or tonic, and an exclusion would thin the 75–90 BPM centre. Spell out that 'held-out' means fresh samples of the same 9 prompts.
- **[minor]** This step 0 makes most of the RM's held-out tuples near-identical (taste_pipeline §3: 4 step-0 samples per brief). In 56.1 % of 20,000 random within-cell 4-tuples of arm B, all four clips carry both of the shown example's sax lines, and the median of the tuples' mean pairwise J5 is 0.79 (verify3.py in the referee scratch folder; an estimate). His repeat agreement p and the RM's agreement both drift toward chance, so D-008's gate says little. Under Part B, held-out clips can also be near-copies of SFT targets that are themselves seed-pool items the RM trains on.
  - **Fix:** Keep D-008 and §3's source as registered, and report three things beside the gate: each held-out tuple's pairwise J5 and verbatim-line sharing; agreement on the pairs whose two clips differ in A or B; and, under Part B, agreement without the clips at J5 ≥ 0.8 to a training target.
- **[minor]** Part B does not say where the dev bars are read: 270 completions per saved adapter, for up to 3 saves. In separate junk-rate runs they would add up to 3 × 450 SU to the stated 1,450 maximum, which also counts no re-run. B6's 50 ear-check clips exclude near-copies of the shown example only, so near-copies of SFT targets (B3 allows up to 50) can get into them.
  - **Fix:** Read the dev bars inside lofi-sft (vLLM with each saved adapter, within its 25 min). State Part B's worst case with one re-run of each job: 2,900 SU. Draw B6's clips only from loops that are near-copies of neither the shown example nor any training target.
- **[minor]** At this step 0, copying carries part of the obedience claim (≥ 90 % on held-out briefs). v2 shows each brief an example at its own energy, so step 0 matches the energy band in 498 of 500 loops (arm B energy_match), while the mood signature holds in only 333 of 500. The entry lists the obedience check among the masked decodes but records no baseline for it.
  - **Fix:** Report energy and mood obedience separately at step N, with step 0's values on the same briefs beside them. The claim stays as registered.
