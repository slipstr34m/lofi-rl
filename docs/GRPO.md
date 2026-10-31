# GRPO.md: the trainer, the interim reward and the smoke

The GRPO trainer, the reward it optimizes until the scorer server exists, and the smoke job that tests both on one
GPU.
- **Spec:** `docs/design/taste_pipeline.md` §5 (reward architecture) and §6 (GRPO settings, smoke protocol, SU
  estimates); `docs/design/instrument.md` §2 (gates) and §5 (verifiers); `CLAUDE.md` §8.
- **Code:** `src/lofi_rl/rl/`: `curriculum.py`, `reward.py`, `config.py`, `train.py`, `smoke_check.py`.
- **Config:** `configs/grpo_smoke.yaml`. **Job:** `jobs/grpo_smoke.sbatch` (`lofi-grpo-smoke`).
- **Tests:** `tests/test_rl_{reward,train,smoke_check,imports,rehearsal}.py`, with the stand-ins in
  `tests/helpers_rl.py`. `test_rl_rehearsal.py` runs `train()` twice (10 steps, then a resume from checkpoint-5)
  against stand-in torch, transformers, TRL, PEFT and datasets modules that call the reward the way GRPOTrainer
  does. `smoke_check` then judges what those two runs wrote. That tests the wiring; real TRL is the cluster's to test.
- **Interfaces used:** `lofi_rl.prompt` (`PROMPT_VERSION`, `build_messages`, `template_sha256`) and
  `lofi_rl.verify` (`score`, `group_penalties`) are written by other builders. The tests use fakes that follow
  their shared interfaces exactly.
- **Sound claims:** none. Nothing here renders audio.

## 1. Entry points

```bash
python -m lofi_rl.rl.train --config configs/grpo_smoke.yaml --dry-run          # laptop or cluster: no torch
python -m lofi_rl.rl.train --config configs/grpo_smoke.yaml [--resume-from <run>/a/checkpoint-5] \
    [--run-dir DIR] [--summary-dir DIR] [--gpu-profile a100-80gb|48gb]        # policy env, one GPU
python -m lofi_rl.rl.smoke_check --run <run> --stage first|final [--rc a=0 --rc b=0]
make submit JOB=jobs/grpo_smoke.sbatch                                          # once the job type is registered
```

- **Exit codes of `train`:** 0 done; 2 a config problem (including a GRPOConfig field TRL lacks, or a prompt too
  long for the limits); 3 no policy env (torch, transformers, TRL, PEFT, datasets or vLLM missing: use `--dry-run`).
  Any other failure is a traceback and 1.
- **Defaults:** `--run-dir` is `$REMOTE_SCR/runs/<run>-<SLURM_JOB_ID>` (or `./runs/<run>-local`), `--summary-dir`
  is `<run-dir>/summary`, and the GPU profile follows the card's memory (§3).

## 2. The interim reward (`lofi_rl.rl.reward`)

```
R = G × Σ_k w_k(t)·r_k − (P_sax + Σ penalties + P_group + P_fix)
```

`make_reward(palette_path, log_path)` returns the reward function TRL calls once per generation batch, as
`reward(prompts=…, completions=…, completion_ids=…, **columns) -> list[float]`.

| term | source | in the smoke |
|---|---|---|
| G | `verify.score(text, brief, palette).gate_ok`, else 0; a completion cut at the token limit is G2 | live |
| r_theory, r_knobs | `verify.score` | live |
| r_taste, r_ce, r_theme | the scorer server (taste RM, Audiobox CE, CLAP theme), taste_pipeline §5 | **0**, weights logged |
| P_sax, penalties (bars, len), P_fix | `verify.score` | live |
| P_group | `verify.group_penalties(texts, scores)` over each prompt's group | live |

- **Weights** (`lofi_rl.rl.curriculum`): w(t) = (1 − s)·W0 + s·W1 with s = min(t, 30)/30.
  - W0 = theory .50, taste .20, knobs .15, CE .05, theme .10; W1 = theory .10, taste .60, knobs .15, CE .05, theme .10.
  - So theory falls linearly from .50 to .10 over steps 0–30 and knobs stay at .15. The endpoints, and a weight
    that does not move, are returned exactly.
  - §5's note "ce → 0, knobs → .20 unless CE validated" belongs to the scorer-server reward, after the CE check on
    Shresth's votes. It is not applied here.
- **The step t** is `trainer_state.global_step` when TRL passes it, which is 0 for the first step's rollouts.
  Otherwise it is the reward's own call counter (one call per optimizer step). A resumed run starts that counter at
  its checkpoint's step (`start_step`).
- **G2.** TRL cuts `completion_ids` after the first EOS, so a completion with at least `max_completion_length` ids
  and no EOS as its last id hit the limit. It is gate G2 whatever its text: G2 comes first (DSL.md §7) and parses
  nothing (DSL.md §1), so its P_sax, penalties and P_fix are 0, and only P_group applies. Without the limit, the
  reward never guesses: `truncated` is `null`.
- **A failed gate** keeps its penalties (the formula as written). TRL's `mask_truncated_completions` keeps G2
  completions out of the loss; their rewards still enter the batch statistics.
- **Groups.** A group is the completions of one dataset row: the trainer's `brief_id` column, or the prompt's hash
  when that column is absent. TRL repeats each row `num_generations` times, so a group is one GRPO group however the
  batch is ordered. `group_penalties` sees a G2 completion's score with `gate_ok` False and gate "G2".
- **Failures.** A verifier exception, or a value that is not a finite number, scores 0 with gate `"error"`, and its
  message is logged. A `group_penalties` failure does the same to the whole group, rather than skip its penalty. The
  smoke fails on any error. `strict=True` re-raises instead (the dry run uses it).
- **The log** (`<summary>/rewards.jsonl`) has one line per completion:
  - `step`, `step_source`, `call`, `i`, `group`, `group_key`, `brief_id`, `brief`, `n_tokens`, `truncated`;
  - `gate_ok`, `gate`, `G`, `w` and `r` (all five terms), `theory_parts`, `knob_parts`;
  - `p_sax`, `penalties`, `p_group`, `p_fix`, `repairs`, `positive` (= G·Σ w·r), `penalty`, `reward`;
  - `digest`, `text_sha256`, `text`; plus `raw_gate` (the verifier's view of a cut text) or `error`.
- **Per-step summary.** `reward.stats[step]` holds the means of every term, the gate rate and counts, truncations,
  errors, the reward's mean, std, min and max, and the call time. The trainer's step log copies it (§4).
- **TRL's logging hooks.** taste_pipeline §5 names `log_metric` and `log_extra`. When TRL passes them the reward logs
  `lofi/reward`, `lofi/G`, `lofi/r_<term>` and `lofi/p_group`. A hook that raises is recorded under `hook_errors`
  and never stops the reward.
- **TODO (taste_pipeline §5, "Scorer server"):** `lofi_rl.scoring.server` (scorer env, `POST /score` over a node-local
  Unix socket) will supply r_taste, r_ce, r_theme and the audio part of the gate: render, PQ floor and P_acc.

## 3. The run config (`configs/grpo_smoke.yaml`, `lofi_rl.rl.config`)

| §6 setting | field(s) | smoke value |
|---|---|---|
| Model | `model.repo`, `model.revision` | Qwen/Qwen3-4B-Instruct-2507 @ `cdbee75f…` (= `lofi_rl.smoke.QWEN`) |
| LoRA | `lora` | r 16, α 32, dropout 0.0, all-linear |
| Learning rate | `learning_rate`, `lr_scheduler_type`, `warmup_steps` | 1.0e-5, constant_with_warmup, 3 |
| Batch | `prompts_per_step` × `num_generations`; per profile `per_device_train_batch_size` × `gradient_accumulation_steps`; `num_iterations` | 4 × 8 = 32; 8 × 4 (A100-80) or 4 × 8 (48 GB); 1 |
| Completions | `max_completion_length`, `mask_truncated_completions` | 512, true |
| β, loss, scaling | `beta`, `loss_type`, `scale_rewards` | 0.04, dapo, batch |
| Sampling | `temperature`, `top_p`, `top_k` | 1.0, 1.0, 0 |
| vLLM | `use_vllm`, `vllm_mode`; per profile `vllm_gpu_memory_utilization`, `vllm_enable_sleep_mode` | true, colocate; 0.25 and off (A100-80), 0.30 and on (48 GB) |
| Smoke | `max_steps`, `save_steps`, `smoke.resume_from_step` | 10, 5, 5 |

- **Also set,** so that nothing falls to a TRL default:
  - `max_prompt_length` 3072: TRL 0.2x defaulted to 512 and cut longer prompts on the left;
  - `vllm_max_model_length` 4096: otherwise vLLM may size its KV cache for the model's 262k context;
  - gradient checkpointing (non-reentrant) and bf16;
  - `save_only_model` false, because the relaunch needs the optimizer, scheduler and RNG state;
  - `remove_unused_columns` false, because the reward reads the brief columns;
  - `log_completions` false (the reward's log already has every completion), `report_to` none, seed 596,
    `logging_steps` 1.
- **Validation** is strict, like `palette.py`:
  - unknown top-level keys are errors;
  - every §6 setting must be stated, in `trainer` or in each profile;
  - numbers, bools and choices are typed: write `1.0e-5`, because YAML reads `1e-5` as text;
  - `output_dir`, `resume_from_checkpoint` and hub settings are refused;
  - each profile must give 32 completions per step;
  - `save_steps` must divide `max_steps`;
  - the resume step must be a saved step before the end;
  - `briefs.count` must cover `max_steps × prompts_per_step`, so no brief repeats.
- **GPU profiles.** The run takes the profile with the largest `min_gib` the card meets: 70 GiB → `a100-80gb`,
  40 GiB → `48gb` (L40S or A40). `--gpu-profile` overrides it.
- **TRL field names.** No TRL source was on the laptop, so every name in `trainer` and `profiles` is unverified here
  (§7). On the cluster, `fit_fields` checks each one against the installed `GRPOConfig` before anything heavy loads:
  - a name it lacks stops the run with exit 2, at a cost of about 2 minutes;
  - the four names in `optional_fields` (`max_prompt_length`, `vllm_max_model_length`, `vllm_enable_sleep_mode`,
    `log_completions`) are dropped and reported instead;
  - `run.json` lists the names it verified and the ones it dropped, and saves the full resolved `GRPOConfig.to_dict()`.
- **Grammar-constrained decoding** (D-016; docs/GRAMMAR.md). The top-level key `structured_outputs: loopscript_regex`
  is optional; `configs/grpo_smoke_constrained.yaml` is the smoke with that key added.
  - `load_config` builds `lofi_rl.grammar.loopscript_regex` from the config's palette, and `resolve` sets it as
    `vllm_structured_outputs_regex` in every profile.
  - The regex is never in the YAML: writing that field in `trainer` or a profile is refused, and so is listing it in
    `optional_fields`.
  - The dry run and `run.json` record its builder, field, sha256 and length under `structured_outputs`.

## 4. The trainer (`lofi_rl.rl.train`)

- **The model.** `kit.hf_cache.resolve_pinned` finds the pinned snapshot in the offline HF cache and fails closed.
  `AutoModelForCausalLM` loads it in bf16, and `GRPOTrainer` gets the model, the tokenizer (left padding and
  truncation, as TRL loads it) and the LoRA config. With PEFT, the KL reference is the adapter-off base.
- **The dataset.** `brief.sample_briefs(seed 596, count 40)` gives 40 briefs, one row each: `prompt` (from
  `prompt.build_messages`, checked to be system/user messages ending with the user's), `brief` (JSON, sorted keys)
  and `brief_id` (`"596:<index>"`). That is 40 rows = 10 steps × 4 prompts.
- **Before the GPU work** the run:
  - tokenizes every prompt with the chat template and refuses one longer than `max_prompt_length`, or one that
    with `max_completion_length` overflows `vllm_max_model_length`;
  - collects the EOS ids (tokenizer and generation config) for the reward's G2 check.
- **Outputs** go to the summary dir, which `make pull` brings back. Checkpoints stay in the run dir.
  - `run.json`: the settings, verified and dropped TRL fields, `GRPOConfig.to_dict()`, versions, GPU, snapshot,
    palette and prompt identity, prompt token min, mean and max, trainable parameters, `startup_s`, `wall_s`, the
    status and any error with its trace. It is rewritten at each stage, so a crash leaves what happened so far.
  - `steps.jsonl`: one `step` record per optimizer step, with `trl` (TRL's logged metrics: reward, reward_std,
    completion lengths, KL, loss, …), `lofi` (the reward's summary of that step's rollouts) and `step_s`. There is
    one `save` record per checkpoint, and a `train_end` record. TRL scores step k's rollouts at global step k and
    logs after the update, at k + 1, so record k + 1 carries the reward's stats[k].
  - `rewards.jsonl`: the reward's per-completion log (§2).
- **Resume.** `--resume-from <ckpt>` passes the checkpoint to `trainer.train`. It reads the checkpoint's global step,
  so the reward's fallback counter starts there.
- **Teardown.** After a GPU run the process ends with `os._exit`, which skips native teardown (vLLM, NCCL) that can
  hang past the walltime. faulthandler prints every thread's stack on a crash or a SIGTERM from `timeout`.
- **`--dry-run`** validates the config, builds the 40 prompts and prints JSON:
  - the resolved settings of every profile, the prompt version, template sha, brief mix, lengths and a preview;
  - the unverified TRL names;
  - `tests/fixtures/loops/example.ls` scored through the real reward path, when `lofi_rl.verify` exists.

  It never imports torch, TRL, vLLM, transformers, PEFT or datasets
  (`tests/test_rl_train.py::test_dry_run_runs_with_torch_trl_and_vllm_blocked`).

## 5. The smoke (`jobs/grpo_smoke.sbatch`)

1. `--dry-run` in the policy env checks the config, the prompts and the reward on `example.ls`. If it fails, no GPU
   work starts.
2. **Launch a** (`<run>/a`): 10 steps of 4 prompts × 8 completions, saved at steps 5 and 10.
3. `smoke_check --stage first`: launch a exited 0, and `a/checkpoint-5` and `a/checkpoint-10` each have a trainer
   state at their step, adapter weights and an optimizer state, so they can resume. Missing scheduler or RNG state
   is a warning. It prints `FIRST_RUN_OK`.
4. **Launch b** (`<run>/b`, so `a/checkpoint-10` is never overwritten): `--resume-from a/checkpoint-5`.
5. `smoke_check --stage final` writes `<run>/summary/smoke.json` and ends the log with `GRPO_SMOKE_OK` only if:
   - both launches exited 0;
   - launch a trained steps 1–10;
   - launch b resumed from `a/checkpoint-5`, started at step 5, trained exactly steps 6–10 (the resumed step count),
     ended at global step 10 and saved `b/checkpoint-10`;
   - the reward scored 32 completions per step in 4 groups of 8, at curriculum steps 0–9 (a) and 5–9 (b);
   - there were no reward errors.

   Warnings list a curriculum step that did not come from TRL's `trainer_state`, dropped optional fields, failed
   logging hooks, and duplicate step records. `smoke.json` also has, for each step, the reward mean and std (ours
   and TRL's), KL, mean completion length, gate rate, loss and step time, and each launch's startup and wall time.

**Timeouts.** Each launch runs under `timeout -k 30s` with the job's remaining time minus 3 minutes, so the final
check always runs inside the walltime.

**Sizing** (the §6 estimates are UNVERIFIED; this smoke measures them):

| | A100-80GB (default) | 48 GB card (OVR) |
|---|---|---|
| Shape | gpu, 1 A100-80GB, 4 CPU, 32 GB | L40S on gpu, or A40 on debug; 4 CPU, 32 GB |
| Rate | 4 + 32/4 + 8 = **20 SU/min** | 20 SU/min |
| Step | §6's 34 s − its 9 s render reward + ~0.5 s symbolic reward ≈ 25.5 s | L40S 49 − 9 = 40 s; A40 ≈ 50 s with sleep mode |
| Expected | 15 steps 6.4 min + startup 3.5 + 2.5 min + saves 0.25 + job 0.5 ≈ **13.2 min ≈ 270 SU** | 12.5 + 8 + 0.75 ≈ 21 min ≈ 420 SU |
| Pessimistic | steps +35 % (34.4 s): 8.6 + startups 2 × 5 + 0.75 ≈ 19.4 min | 16.9 + 10 + 0.75 ≈ 27.6 min |
| `--time` | **25 min**: 5.6 min over the pessimistic case, 1.9 × the expected | 40 min |
| Worst case | 20 × 25 = **500 SU** | 20 × 40 = 800 SU (Shresth's OK) |

- **Startup** covers the imports, the transformers load and vLLM's load, compile and CUDA-graph capture, plus the
  first step's warm-up. Phase-0's eager vLLM load alone took 37 s on an A40 (CHANGELOG, smoke 13628361). The second
  launch reuses the compile caches on scratch.
- **Against §6's 550 SU:** the trimmed figure drops the render reward (9 s a step), 8 CPUs (12 → 4, since the
  symbolic reward needs no render pool) and 32 GB (64 → 32, since no scorer server runs).
- **The 48 GB route** passes `OVR='--gres=gpu:l40s:1 --constraint=l40s --time=00:40:00'`, or
  `OVR='-p debug --gres=gpu:a40:1 --constraint=a40 --time=00:40:00'`. Its projected 800 SU is over the 500 SU a
  Claude registration allows, so the guard refuses it until Shresth approves.

## 6. Choices the brief left open

1. `make_reward` takes optional keyword-only extras:
   - `verify`, to inject a module (the tests);
   - `max_completion_length` and `eos_token_ids`, to see G2;
   - `start_step`, for a resumed run when TRL passes no `trainer_state`;
   - `strict`.

   The two positional arguments are the shared interface. The returned object is a callable `LofiReward`, with
   `__name__` `lofi_reward` (TRL's metric name) and `.stats`.
2. G2 is detected from token ids (§2), and it zeroes P_sax, penalties and P_fix but not P_group. A failed gate keeps
   its penalties. A verifier error scores 0 and fails the smoke.
3. A group is a dataset row (`brief_id`), else a prompt hash. The prompt shows only the scene and the knobs, so two
   rows with different tempi or tonics can share a prompt text and are still two groups.
4. The curriculum keeps §5's W0 and W1 for taste, CE and theme, and their r is 0. The step is TRL's `global_step`,
   counted from 0.
5. The dataset columns are `prompt`, `brief` (JSON) and `brief_id`. The brief stream is seed 596 with 40 rows.
   Held-out and dev briefs must use other seeds (to be recorded before Nov 30).
6. The config schema: `trainer` plus GPU `profiles` picked by memory, `optional_fields` for names TRL may lack, and
   `smoke.resume_from_step`.
7. LoRA dropout is 0.0 (PEFT's default: no dropout noise between generation and the loss), with bias none and task
   CAUSAL_LM. The settings beyond §6 are listed in §3.
8. The trainer loads the model and tokenizer itself and hands them to `GRPOTrainer`.
9. The output layout: `run.json`, `steps.jsonl` and `rewards.jsonl` per launch. The relaunch writes to its own
   `b/` and `summary/b/`.
10. After a GPU run the process ends with a hard exit. Each launch is capped by `timeout`, and the smoke verdict
    includes both exit codes.
11. Stage `first` gates the relaunch only on what resuming needs. Reward errors and step counts are judged at the
    final stage, so a verifier bug does not hide the resume test.
12. The 4 CPUs and 32 GB are trimmed from §6's 12 CPUs and 64 GB, and the walltime is 25 min (§5).
13. The import test blocks the ML stack as well as mido, soundfile, scipy and pedalboard, because the laptop has
    none of it.

## 7. Unverified until the cluster smoke (no TRL source on the laptop)

- **GRPOConfig / TrainingArguments names.** These are checked by `fit_fields`, recorded in `run.json`, and also by
  `tests/test_rl_train.py::test_trl_knows_every_setting_of_the_smoke` in the policy env:
  - learning_rate, lr_scheduler_type, warmup_steps, max_steps, per_device_train_batch_size, gradient_accumulation_steps;
  - num_generations, num_iterations, max_completion_length, mask_truncated_completions, beta, loss_type, scale_rewards;
  - temperature, top_p, top_k, use_vllm, vllm_mode, vllm_gpu_memory_utilization;
  - save_strategy, save_steps, save_only_model, logging_steps, bf16, gradient_checkpointing,
    gradient_checkpointing_kwargs, remove_unused_columns, report_to, seed;
  - optional: max_prompt_length, vllm_max_model_length, vllm_enable_sleep_mode, log_completions;
  - constrained config only: vllm_structured_outputs_regex (required there, never optional).
- **Values:**
  - `top_k: 0` means "off" (vLLM 0.32 reads 0 or −1 as all tokens);
  - `scale_rewards: batch` and `loss_type: dapo` exist in this TRL;
  - `warmup_steps` is still an integer step count in transformers 5.19.
- **TRL behaviour:**
  - the reward kwargs include `trainer_state` (the smoke warns if the step came from the counter);
  - `completion_ids` end at, and include, the EOS;
  - conversational completions arrive as `[{"role": "assistant", "content": …}]`;
  - the signatures of `log_metric` and `log_extra`;
  - one reward call per optimizer step with these batch settings (the smoke checks 32 completions per step);
  - colocate mode sets vLLM's `max_model_len` from `vllm_max_model_length`.
- **Transformers, PEFT and checkpoints:**
  - `from_pretrained(dtype=torch.bfloat16)`;
  - `apply_chat_template(tokenize=False)` returns text;
  - checkpoint file names: `trainer_state.json`, `adapter_model.safetensors`, `optimizer.pt`, `scheduler.pt`,
    `rng_state.pth`;
  - `PeftModel.get_nb_trainable_parameters`.
