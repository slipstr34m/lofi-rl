# GRPO.md: the trainer, the interim reward and the smoke

The GRPO trainer, the reward it optimizes until the scorer server exists, and the smoke job that tests both on one
GPU.
- **Spec:** `docs/design/taste_pipeline.md` §5 (reward architecture) and §6 (GRPO settings, smoke protocol, SU
  estimates); `docs/design/instrument.md` §2 (gates) and §5 (verifiers); `CLAUDE.md` §8.
- **Code:** `src/lofi_rl/rl/`: `curriculum.py`, `reward.py`, `copy_penalty.py`, `config.py`, `train.py`,
  `trl_probe.py`, `smoke_check.py`, `masked.py` (the masked trainer, §8), and `src/lofi_rl/grammar_mask.py` (its
  independent NFA check).
- **Configs:** `configs/grpo_smoke.yaml` (the smoke 13632679 ran), `configs/grpo_smoke_constrained.yaml` (plus the
  grammar), `configs/grpo_smoke_v2_grammar.yaml` (D-020's run-1 settings: prompt v2, the grammar, the copy
  penalty) and `configs/grpo_smoke_v2_grammar_masked.yaml` (that plus D-021's `trainer_logprobs: grammar_masked`,
  §8). **Job:** `jobs/grpo_smoke.sbatch` (`lofi-grpo-smoke`), which takes one of them as its argument (§5).
- **Tests:** `tests/test_rl_{reward,copy_penalty,train,trl_probe,smoke_check,imports,rehearsal,masked_logps}.py`,
  `tests/test_grammar_mask.py` and `tests/test_grpo_smoke_job.py`, with the stand-ins in `tests/helpers_rl.py`. `test_rl_rehearsal.py` runs
  `train()` twice (10 steps, then a resume from checkpoint-5) against stand-in torch, transformers, TRL, PEFT,
  datasets and vLLM modules that call the reward the way GRPOTrainer does, on today's smoke and on the v2 grammar
  smoke. `smoke_check` then judges what those two runs wrote. That tests the wiring; real TRL is the cluster's to
  test.
- **Interfaces used:** `lofi_rl.prompt` and `lofi_rl.prompt_v2` (`PROMPT_VERSION`, `build_messages`,
  `template_sha256`; v2 also `example_id` and `reference_texts`) and `lofi_rl.verify` (`score`, `group_penalties`)
  are written by other builders. The tests use fakes that follow their shared interfaces exactly, or the real
  modules.
- **Prompt version:** the config's `prompt` key, v1 (the default) or v2 (§3). D-020's outcome picks v2 with the
  grammar for run 1, and its Before-GRPO-run-1 list is built here: the prompt key, the copy penalty (§2), the
  backend pin and the in-language count (§4, §5), and the IS readout for the grammar lever (§5).
  `prompt_v2.py`'s docstring still says GRPO builds v1 and that P_pool must exist first; that file is on D-020's
  frozen path (`docs/prereg/D-020_junk_path.txt`), so this page supersedes that sentence.
- **Sound claims:** none. Nothing here renders audio.

## 1. Entry points

```bash
python -m lofi_rl.rl.train --config configs/grpo_smoke.yaml --dry-run          # laptop or cluster: no torch
python -m lofi_rl.rl.train --config configs/grpo_smoke.yaml [--resume-from <run>/a/checkpoint-5] \
    [--run-dir DIR] [--summary-dir DIR] [--gpu-profile a100-80gb|48gb] [--scorer-pin FILE]   # policy env, one GPU
python -m lofi_rl.rl.smoke_check --run <run> --stage first|final [--config CFG] [--rc a=0 --rc b=0]
make submit JOB=jobs/grpo_smoke.sbatch                                          # today's smoke (configs/grpo_smoke.yaml)
make submit JOB=jobs/grpo_smoke.sbatch ARGS='configs/grpo_smoke_v2_grammar.yaml'   # D-020's run-1 settings
make submit JOB=jobs/grpo_smoke.sbatch OVR='--gres=gpu:l40s:1 --constraint=l40s' \
    ARGS='configs/grpo_smoke_v2_grammar_masked.yaml'                            # D-021 Part A's smoke (§8)
```

- **The job type** changed with its config argument: its sha is `1dded12d386c7fab` (it was `07203168742f7411`), at
  most 500 SU, so it is re-registered (`make approve JOB=jobs/grpo_smoke.sbatch MAX=500 BY=Claude`) before the next
  submit; the guard refuses it until then.

- **Exit codes of `train`:** 0 done; 2 a config problem (including a GRPOConfig field TRL lacks, or a prompt too
  long for the limits), or a masked launch refused by its checks (run.json `status: refused`, §8); 3 no policy env
  (torch, transformers, TRL, PEFT, datasets or vLLM missing: use `--dry-run`). Any other failure is a traceback and 1,
  a sampled token outside the mask (`MaskViolation`) among them.
- **Defaults:** `--run-dir` is `$REMOTE_SCR/runs/<run>-<SLURM_JOB_ID>` (or `./runs/<run>-local`), `--summary-dir`
  is `<run-dir>/summary`, and the GPU profile follows the card's memory (§3). `--scorer-pin` (used only with a
  scorer, docs/SCORER.md §10) is `<run-dir>/scorer_pin.json`, which a resumed link first copies from the run dir
  of its `--resume-from` checkpoint, so every link of a run pins what its first link pinned.

## 2. The interim reward (`lofi_rl.rl.reward`)

```
R = G × Σ_k w_k(t)·r_k − (P_sax + Σ penalties + P_group + P_fix + P_copy)
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
| P_copy | `lofi_rl.rl.copy_penalty` (below), only when the config sets `copy_penalty` | v2 grammar smoke only |

- **P_copy** (taste_pipeline §5's copy row; D-020's Before-GRPO-run-1 list).
  - **The ramp:** 0 at J ≤ 0.6, rising linearly to 0.3 at J ≥ 0.8 (`copy_penalty.ramp`; the endpoints are exact).
  - **J** is the largest character 5-gram Jaccard of the completion with any reference, by junk.py's measure: its
    `grams`, `jaccard`, `Reference` and `best_match` are imported, never copied. The completion's copy form is
    junk.score_row's: a text that passes the parser's gates (dsl.parse under the palette's vocabulary in the brief's
    mode; a cut completion is G2) is compared by its canonical text without `sax:`, any other by `junk.plain_text`.
    `tests/test_rl_copy_penalty.py` checks J against junk's own `copy_any_jaccard` (v2) and `copy_jaccard` (v1)
    on 114 texts: every repo loop and 54 edited copies of the examples.
  - **The references** are every example the run's prompt version can show, since across its prompts the policy
    sees all of them: v1 its one (`prompt.example_text`), v2 all 9 of `prompt_v2.reference_texts` (strict). The
    config's `copy_penalty.pool` adds one reference per `.ls` file (the seed pool, later), in its own copy form
    with no brief's mode, keyed by its repo path. taste_pipeline §5 names MinHash; this is the exact Jaccard it
    estimates, compared with every reference (an LSH index is the change to make if a large pool shows in the step
    time).
  - **Where it applies:** every scored completion. A failed gate and a G2 keep it, as they keep P_group: it reads
    the text, not a parse. A verifier error still scores 0, with `p_copy` 0 and no J.
  - **Opt-in, decided here:** the reward adds P_copy only when built with `copy=` (`train` does it for a config with
    `copy_penalty`). Without it the record has no `p_copy` field, not even 0.0, and today's smoke keeps its rewards
    and log byte for byte (`tests/test_scoring_reward_hook.py`'s golden, unchanged).
  - **What it does to v2 under the grammar.** D-020 arm B's 500 completions
    (`runs/junk-13640325/summary/completions.jsonl`) went through the real reward at step 0, one group per brief, on
    the laptop on 2026-11-01.
    - **The measure:** J and the nearest reference equal junk's `copy_any_jaccard` and `copy_any_source`
      (`scored.jsonl`) on 500 of 500.
    - **How many it charges:** 52 have J ≤ 0.6, 394 sit in the ramp and 54 have J ≥ 0.8 (junk's `near_copy_any`
      count is also 54). So P_copy is above 0 for 448 of the 500.
    - **Its size:** mean 0.1754 (median 0.1911), against a positive term of mean 0.6078 (median 0.6312). It is never
      larger than the positive term. The mean reward is 0.4285 with it and 0.6038 without.
    - **What follows:** in run 1 it is the largest penalty, and it reaches most completions, not only the near-copies
      that pass_not_copy counts.
- **The in-language count** (a constrained config; D-020's Validity, on the GRPO path). With `language=` (the
  config's `structured_outputs`) every completion gets `finish` and `in_language`: whether its text fullmatches the
  decoding regex under Python's `re`.
  - **The finish reason** comes from the token ids, as G2 does: `length` when the completion hit the limit with no
    EOS last, `stop` when it ended before the limit or on an EOS, `other` when the ids or the limit are unknown. TRL
    passes no vLLM finish reason to a reward.
  - Under a working constraint every `stop` completion is in the language; a `length` one is cut, so it is counted
    apart.
- **The shown example:** a prompt-v2 row carries `example_id`, and the log copies it per completion. A v1 row has no
  such column, so v1's records are unchanged.

- **Weights** (`lofi_rl.rl.curriculum`): w(t) = (1 − s)·W0 + s·W1 with s = min(t, 30)/30.
  - W0 = theory .50, taste .20, knobs .15, CE .05, theme .10; W1 = theory .10, taste .60, knobs .15, CE .05, theme .10.
  - So theory falls linearly from .50 to .10 over steps 0–30 and knobs stay at .15. The endpoints, and a weight
    that does not move, are returned exactly.
  - §5's note "ce → 0, knobs → .20 unless CE validated" belongs to the scorer-server reward: the reward applies it
    only when a scorer reports CE as not validated (SCORER.md §10), never in the smoke.
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
  - `digest`, `text_sha256`, `text`; plus `raw_gate` (the verifier's view of a cut text) or `error`;
  - with P_copy: `p_copy`, `copy_j` (the largest J) and `copy_nearest` (its reference: the first on a tie, J 0
    included, as junk's best_match); with the language count: `finish` and `in_language`; on a v2 row:
    `example_id`.
- **Per-step summary.** `reward.stats[step]` holds the means of every term, the gate rate and counts, truncations,
  errors, the reward's mean, std, min and max, and the call time. The trainer's step log copies it (§4). With P_copy
  it adds `copy` (p_copy_mean, penalized, j_mean, j_max, at_least 0.6 and 0.8, the nearest references of the
  penalized); with the language count `in_language` (the regex sha256, and n, in_language, out_of_language for
  stop, length and other, junk.json's shape).
- **TRL's logging hooks.** taste_pipeline §5 names `log_metric` and `log_extra`. When TRL passes them the reward logs
  `lofi/reward`, `lofi/G`, `lofi/r_<term>` and `lofi/p_group`, plus `lofi/p_copy` and `lofi/stop_out_of_language`
  when those terms are on. A hook that raises is recorded under `hook_errors` and never stops the reward.
- **The scorer server** (taste_pipeline §5, "Scorer server") is built: `lofi_rl.scoring`, specified in
  [docs/SCORER.md](SCORER.md). With `$LOFI_SCORER_SOCK` set (or `make_reward(..., scorer=...)`), the reward sends the
  completions that passed the symbolic gate to it in one `POST /score` and takes r_taste, r_ce, r_theme and the
  audio part of the gate (render, PQ floor, P_acc factor) from it (SCORER.md §2, §10). The smoke runs without it,
  so its reward and logs are unchanged byte for byte (`tests/test_scoring_reward_hook.py`).

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
  - The dry run and `run.json` record its builder, field, sha256 and length under `structured_outputs`, and
    run.json adds the backend record (§4).
  - **The field name exists** in TRL 1.17.1's GRPOConfig: `runs/grpo-smoke-13632679/summary/a/run.json` lists
    `vllm_structured_outputs_regex` (as null) in `grpo_config`. Whether TRL hands it to vLLM is what the in-language
    count checks.
- **The prompt version:** the optional top-level `prompt: v1|v2` (default v1, so a config without it is unchanged).
  It picks the template module every row's messages come from: v1 `lofi_rl.prompt`, v2 `lofi_rl.prompt_v2`. Any
  other value is refused at load, and the trainer refuses a module whose `PROMPT_VERSION` is another.
- **The copy penalty:** the optional top-level `copy_penalty: {pool: [...]}` turns P_copy on (§2). `pool` is
  required inside it: a list of repo-relative (or absolute) `.ls` files or directories of them, `[]` for the
  prompt's examples only. A missing path, another suffix, an empty directory or a path listed twice is refused at
  load. The thresholds are taste_pipeline §5's and live in code, never in a config.
- **`configs/grpo_smoke_v2_grammar.yaml`** is `configs/grpo_smoke.yaml` plus `prompt: v2`,
  `structured_outputs: loopscript_regex` and `copy_penalty: {pool: []}`, and every other key the smoke's
  (`tests/test_rl_train.py`). `configs/grpo_smoke.yaml` itself is unchanged: sha256 `d0ed1702…`, the file smoke
  13632679 ran.

## 4. The trainer (`lofi_rl.rl.train`)

- **The model.** `kit.hf_cache.resolve_pinned` finds the pinned snapshot in the offline HF cache and fails closed.
  `AutoModelForCausalLM` loads it in bf16, and `GRPOTrainer` gets the model, the tokenizer (left padding and
  truncation, as TRL loads it) and the LoRA config. With PEFT, the KL reference is the adapter-off base.
- **The dataset.** `brief.sample_briefs(seed 596, count 40)` gives 40 briefs, one row each: `prompt` (from the
  config's prompt module's `build_messages`, checked to be system/user messages ending with the user's), `brief`
  (JSON, sorted keys) and `brief_id` (`"596:<index>"`). That is 40 rows = 10 steps × 4 prompts.
  - **Prompt v2 rows** are v2's per-cell messages, and carry `example_id` (`prompt_v2.example_id`), the example
    their prompt shows. For the smoke's 40 briefs that is 8 cells (no high/melancholy brief) and 8 of the 9 examples.
  - **v1 rows** have no new column: their one example is `"example"`.
- **The backend pin** (a constrained config). Before `GRPOTrainer` is built, `trl_probe.BackendPin` wraps
  `vllm.LLM.__init__` so the LLM that TRL builds in colocate mode asks for
  `structured_outputs_config={"backend": "xgrammar"}`, as junk.py's own engine does (`junk.STRUCTURED_BACKEND`).
  - **No keyword:** a TypeError on the keyword builds the LLM again without it, and `pin_accepted` is false.
  - **TRL's own choice:** a caller that passes its own `structured_outputs_config` keeps it, recorded as
    `set_by_caller`.
  - **The read-back:** after construction `find_engine` walks the trainer's attributes (at most 3 links; never into
    the model, tokenizer, accelerator or datasets) to the LLM. `junk.read_backend` reads its backend.
  - **The record:** run.json `structured_outputs.backend` has `requested`, `pin_accepted`, `backend`, `source`,
    `applied`, `engine` (the path it was found at) and `calls` (LLMs built). If no engine is on the trainer when it
    is built, the read is made again after training.
  - The class is restored when the launch ends. None of this raises; an unconstrained config pins nothing.
- **Before the GPU work** the run:
  - tokenizes every prompt with the chat template and refuses one longer than `max_prompt_length`, or one that
    with `max_completion_length` overflows `vllm_max_model_length`;
  - collects the EOS ids (tokenizer and generation config) for the reward's G2 check.
- **Outputs** go to the summary dir, which `make pull` brings back. Checkpoints stay in the run dir.
  - `run.json`: the settings, verified and dropped TRL fields, `GRPOConfig.to_dict()`, versions, GPU, snapshot,
    palette and prompt identity, prompt token min, mean and max, trainable parameters, `startup_s`, `wall_s`, the
    status and any error with its trace. It is rewritten at each stage, so a crash leaves what happened so far.
    - **The prompt:** `version`, `module`, `template_sha256` (v1 `7aceb2bc…`, v2 `127e3094…`), and per row
      `brief_ids`, `example_ids` and `messages_sha256`, with `example_counts`.
    - **`copy_penalty`:** the rule, its numbers, the reference ids with each prompt example's sha256, and the pool.
    - **`structured_outputs.backend`:** the pin's record, above.
    - **`trl_logp_metric`:** where the installed TRL computes `sampling/sampling_logp_difference/mean`, and whether
      that is D-020's definition (§5).
    - **`vllm_logprobs`:** evidence only, unchecked: lines of the installed vLLM showing where the grammar bitmask
      meets the logits and which log-probs the sampler returns.
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
  - the resolved settings of every profile, the prompt version, module, template sha, every row's example id and
    messages sha, brief mix, lengths and a preview;
  - the unverified TRL names;
  - the copy penalty's summary, the language count's regex sha and the backend the run will pin;
  - `trl_logp_metric`: the installed TRL's metric source, read as text (on the laptop: `not_found`);
  - `tests/fixtures/loops/example.ls` scored through the real reward path, when `lofi_rl.verify` exists: the
    symbolic reward, with P_copy and the language count as the config sets them, and the scorer off even when
    `$LOFI_SCORER_SOCK` is exported (the report's `reward.scorer` says whether it is set; the scorer is never
    contacted).
  - **On the laptop, 2026-11-01:**
    - `configs/grpo_smoke.yaml` reports the reward check smoke 13632679's dry run reported: 0.5784511276657195, the
      same profiles, prompts and preview (`runs/grpo-smoke-13632679/summary/dry_run.json`).
    - `configs/grpo_smoke_v2_grammar.yaml` scores example.ls as a copy of `example`: J 1.0, P_copy 0.3, reward
      0.2784511276657195. example.ls is v1's example with a `sax:` line.

  It never imports torch, TRL, vLLM, transformers, PEFT or datasets
  (`tests/test_rl_train.py::test_dry_run_runs_with_torch_trl_and_vllm_blocked`).

## 5. The smoke (`jobs/grpo_smoke.sbatch`)

**The config** is the job's one optional argument: one of the `configs/grpo_smoke*.yaml` files in the mirror, default
`configs/grpo_smoke.yaml`. Any other value, or a second word, stops the job (exit 2) before the env is sourced. The
config reaches the dry run, both launches and both checks, and the log names it.
- `make submit JOB=jobs/grpo_smoke.sbatch` runs today's smoke, the config smoke 13632679 ran.
- `make submit JOB=jobs/grpo_smoke.sbatch ARGS='configs/grpo_smoke_v2_grammar.yaml'` runs D-020's run-1
  settings on the default A100. Smoke 13632679 ran on an L40S at the same 25 min:
  `OVR='--gres=gpu:l40s:1 --constraint=l40s'` with the same ARGS, still at most 500 SU.

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
   - there were no reward errors;
   - with a constrained config, every `stop` completion of both launches is in the regex's language, every
     completion has an in-language record, none has the finish reason `other` (no finish reason from the token
     ids, so the check cannot judge it; D-020's Validity: `other.n` is 0), each launch has at least one `stop`
     completion (else nothing shows the grammar applied), and each launch's run.json names the config's regex
     (sha256);
   - with `copy_penalty`, every completion has a `p_copy`, and each launch's run.json `copy_penalty.pool.files` is
     the number of `.ls` files the config's pool holds (the trainer handed the pool to the copy penalty).

   Warnings list a curriculum step that did not come from TRL's `trainer_state`, dropped optional fields, failed
   logging hooks, duplicate step records, and an IS readout that is unavailable or whose rule reads the mean of the
   steps. `smoke.json` also has, for each step, the reward mean and std (ours and TRL's), KL, mean completion
   length, gate rate, loss, step time and TRL's `sampling/sampling_logp_difference/mean` (as TRL logged it, read or
   not), and each launch's startup and wall time. A launch a in-language failure is judged at the final stage, like
   a reward error, so it never blocks the resume test.

**What the D-020 settings add to `smoke.json`:**
- **`in_language`** (constrained): the regex's sha256 and the counts by finish reason (stop, length, other: n,
  in_language, out_of_language), in total and per launch, with per-step stop counts, the first out-of-language
  stop completions and each launch's `shows_grammar` (below).
- **`backend`** (constrained): each launch's run.json record of the pin: requested, pin_accepted, backend, source,
  applied, engine. It decides nothing, as in junk.json.
- **`copy`** (copy_penalty): per launch, the pool files run.json records, P_copy's mean, the completions it
  penalized, J ≥ 0.8, the largest J, the nearest references, and how many of the penalized were nearest to the
  example their own prompt showed.
- **`is_ratio`**: D-020's IS readout for the grammar lever. D-020 defines it as the mean, over every completion
  token of every step, of |log π_vLLM(token) − log π_trainer(token)|.
  - **When TRL's metric is read:** only when every launch's run.json `trl_logp_metric.verdict` is `match`. That is,
    the installed TRL's source was parsed at run time and its `sampling/sampling_logp_difference/mean` is
    `mean(|old_per_token_logps − sampling_per_token_logps|[completion mask])`. The checks are in
    `lofi_rl.rl.trl_probe`: a plain mean of an absolute difference, masked by `completion_mask`, one operand
    vLLM's sampling log-probs, the other computed on `self.model`, aggregated only by gather/mean/item.
  - **Otherwise** `value_source` is `unavailable`, with the verdicts and the reason, and a warning. No value or rule
    is given, though TRL's numbers stay in `steps`. Computing it from what the trainer exposes would mean hooking
    TRL internals no laptop test can check, so this build does not.
  - **What the `match` check leaves open:** it is TRL's half of D-020's definition. Whether vLLM's reported
    log-probs (`sampling_per_token_logps`) are the masked sampler's, D-020's π_vLLM under the grammar, is UNVERIFIED
    (§7). `vllm_logprobs_checked` records each launch's run.json `vllm_logprobs.checked`, which this build always
    writes as false (`trl_probe.vllm_logprobs_evidence` records vLLM's lines, never a verdict). With the grammar, a
    rule whose launches are not all true carries that caveat.
  - **Every step of both launches counts:** the resumed steps 6–10 are steps the smoke ran, generated again. Per
    step, `value` is TRL's logged mean and `tokens` its completion tokens from rewards.jsonl (step k's record
    carries reward step k − 1). Truncated completions are left out under `mask_truncated_completions`, as TRL's
    mask leaves them out.
  - **The means:** `mean_of_steps` and `token_weighted_mean`, which is D-020's "over every completion token", are
    reported, overall and `by_launch`.
  - **The rule** reads the token-weighted mean when every step has its tokens, else the mean of the steps:
    `exceeds` is value > 0.05. `caveats` name steps with truncated completions, whose tokens TRL's value leaves
    out, the two means falling on two sides of 0.05, a rule that reads the mean of the steps (the steps that
    have no token count), which is not D-020's mean over every completion token, and vLLM's log-probs not checked
    to be the masked sampler's.
  - **When it is D-020's reading** (`applies`): only for the grammar + v2 config (`configured`), and only when both
    launches showed the grammar applied (`grammar_shown`). A launch shows it (its `in_language` `shows_grammar`)
    when it scored completions, its run.json names the config's regex, and none of the in-language failures above
    was found. By D-020's Validity a constrained run that fails that check decides nothing, so its rule says so,
    with a warning, even when the value is under 0.05. An engine without the grammar can read under it: the
    unconstrained smoke 13632679 logged 0.0044–0.0091 at its 15 steps
    (`runs/grpo-smoke-13632679/summary/{a,b}/steps.jsonl`).

**Timeouts.** Each launch runs under `timeout -k 30s` with the job's remaining time minus 3 minutes, so the final
check always runs inside the walltime.

**With the D-020 settings** the walltime and the cost should stay; the smoke measures them.
- **The grammar** compiles the regex once per launch and masks each decode step. D-020's constrained junk runs
  generated 2,615.3 and 2,579.9 tok/s, against v0's 2,230.1 (`runs/junk-1364032{4,5}/summary/generate.json`,
  `runs/junk-13632676/summary/generate.json`).
- **P_copy and the language check** added 0.43 ms a completion to a reward of about 12 ms, in a laptop run of 20
  completions.

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

**The D-020 build (2026-11-01).** These are choices made for the Before-GRPO-run-1 list. They are readings of D-020
for its smoke, so the merge records them in D-020 before that smoke runs.

14. **Opting in:**
    - P_copy is opt-in by a config key, `copy_penalty`. Today's v1 smoke config has none, so its rewards stay byte
      for byte; the v2 grammar smoke sets it.
    - The in-language count is on for every constrained config.
    - Neither adds a field, not even a 0, when off.
15. **What P_copy covers:** every scored completion, a G2 and a failed gate included (it reads the text, like
    P_group). The pool's files are compared in their own copy form, with no brief's mode.
16. **The finish reason** comes from the token ids (G2's test), since TRL passes no vLLM finish reason to a reward.
    `stop` is "ended before the limit or on an EOS".
17. **A v2 row** carries `example_id`; a v1 row gets no new column, so v1's dataset and log are what they were.
18. **The backend pin** wraps `vllm.LLM.__init__` for the launch rather than assuming where TRL builds its engine.
    The engine is then found by walking the trainer, not read from an assumed attribute.
19. **The IS readout:**
    - It counts every step of both launches.
    - It weights the per-step values by completion tokens, truncated ones out as TRL's mask leaves them out.
    - The 0.05 rule reads the token-weighted mean, which is D-020's "over every completion token".
    - It is D-020's reading only when both launches showed the grammar applied (`grammar_shown`); otherwise the smoke
      fails and the rule decides nothing (D-020, Validity).
    - Without a checked TRL source it is unavailable, never computed from guessed internals.
    - An unavailable readout is a warning, not a smoke failure: D-020's reading then stays open.

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

**The D-020 build: UNVERIFIED until the v2 grammar smoke runs.** Each item is recorded by the run, and the smoke
fails closed where it can.
- **Recorded, not yet seen working:**
  - **Where TRL builds its engine:** TRL 1.17.1 builds its colocate engine through `vllm.LLM`, so the pin reaches
    it. Recorded: `structured_outputs.backend.calls` and `pin_accepted`; `calls` 0 means the pin never ran.
  - **Where TRL keeps it:** the engine sits on the trainer within 3 attribute links. Recorded: `engine`, null when
    not found.
  - **The read-back:** `junk.read_backend`'s paths hold for TRL's engine as they did for junk's. Recorded: `source`;
    `unknown` when none resolves.
- **Checked by the in-language count:**
  - TRL passes `vllm_structured_outputs_regex` into the SamplingParams of every generation;
  - the text TRL decodes for a reward is the text vLLM's grammar constrained;
  - a completion shorter than the limit ended on an EOS (so `stop` is vLLM's stop).

  A `stop` completion out of the language fails the smoke, as do an `other` completion and a launch with no
  `stop` completion. Any of them also keeps the IS rule from being D-020's reading (§5).
- **Read from the installed TRL, and possibly not read at all:** whether TRL 1.17.1's
  `sampling/sampling_logp_difference/mean` is D-020's definition.
  - The check reads its source at run time (`trl_logp_metric`) and fails closed: a definition it does not
    recognise makes the readout unavailable.
  - The check is on TRL's text. Two things it does not see:
    - that one value is appended per optimizer step (the smoke's `grpo_config` shows `generation_batch_size` 32 =
      32 completions per step: `runs/grpo-smoke-13632679/summary/a/run.json`);
    - that TRL's mask matches the reward's token counts.
- **Not checked by any step:** which log-probs vLLM 0.32 reports, before or after the grammar bitmask.
  - D-020's π_vLLM is the masked sampler. TRL's metric shows the mask's effect only if vLLM computes its log-probs
    from the masked logits.
  - `vllm_logprobs` records the lines that say so, and TRL's `logprobs_mode` lines are in `trl_logp_metric`, for a
    reader. Neither decides anything.
  - So a `match` verdict establishes TRL's half of D-020's quantity, not all of it. The constrained smoke's IS rule
    carries the caveat (`smoke.json` `is_ratio.rule.caveats`) unless every launch's `vllm_logprobs.checked` is true.
- **The dataset column:** TRL passes the dataset's `example_id` column to the reward as it passes `brief_id`; a v2
  record without it is visible in rewards.jsonl.

**Since the D-021 build (2026-11-02).** The installed sources of the cluster's policy env (TRL 1.17.1, vLLM 0.32.0,
xgrammar 0.2.8, PEFT, transformers) were read on the laptop from copies that are not in the repo (licenses). §8 cites
their lines. That changes two items above:
- **Which log-probs vLLM 0.32 reports** is now read, not left unchecked. In the method that samples,
  `apply_grammar_bitmask` changes the logits (`v1/worker/gpu_model_runner.py:4335`) before `self._sample` runs
  (`:4340`), and TRL asks for `processed_logprobs`. A masked launch records this as V1 (not gating). The masked
  smoke's M10 ratio tests it at run time, so for that smoke D-021 settles D-020's caveat. The unmasked smokes keep the
  caveat as written.
- **TRL's mask against the reward's token counts** is now recorded per step for a masked smoke, in `smoke.json`
  `d021.diagnostics.token_counts` and `replay_positions`. It is not gated.

**The D-021 build: UNVERIFIED until the masked smoke runs.** Each item is recorded by the run, and the launch fails
closed where it can.
- **The run-time source checks:** T1-T5, V1 and V2 read the installed packages again at run time, and those readings
  gate (exit 2). On the laptop, against the copies, every check matched for both modes with smoke 13657200's
  recorded `grpo_config` as the settings (§8).
- **The masked subclass on real TRL:** `masked.make_trainer_class(GRPOTrainer)` built with `mask_setup=`, its three
  overrides called where §8 says TRL calls them, the pass labels against PEFT's adapter state, and 32 rows in each
  pass of a step. Recorded in each step's `mask` record. M8 fails if anything is missing.
- **xgrammar's API as `XgrammarReplay` calls it:** `TokenizerInfo.from_huggingface`, `GrammarCompiler`,
  `compile_regex`, `GrammarMatcher` and its bitmask calls. These are read from xgrammar 0.2.8's source and run only
  on the cluster. A failure stops the launch before the model loads (exit 1).
- **One vocabulary:** the NFA and the replay agree at every position (M8: `nfa_mismatch` 0). They are built over
  the snapshot's `config.json` vocab_size and the EOS set that vLLM's `all_stop_token_ids` holds.
- **The cost of the mask** (M12). Each step adds three parts:
  - the replay, the NFA and the support. D-021 estimates under 1 s a step, which covers only this part. On the laptop
    the NFA's build over the cached Qwen3 tokenizer took 0.295 s (2026-11-02).
  - one more no-grad forward of the step's 32 rows in the old pass: TRL's own path without the support, for M10's
    −log Z (`masked.py`, MaskedPasses.logps). It is of the order of the old pass itself (an estimate, unmeasured).
  - the 2-row fill self-check, once at each launch's first old pass.

  Each step's `mask.timing` measures them (`unmasked_s`, `selfcheck_s`, `walk_s`, `nfa_s`), and smoke.json's
  `gates.M12.breakdown` averages them. Run 1's STOP rules do not read −log Z, so whether run 1 keeps that forward is
  decided when its code path is registered.

## 8. The masked trainer (D-021 Part A)

D-021 (Shresth, 2026-11-02: "A: fix first") keeps the grammar for run 1 if one smoke shows the trainer can be made
on-policy for the masked sampler. This section is the wiring: the config key, what a launch does before it trains,
what it records, and the smoke's verdict. The trainer itself is `lofi_rl.rl.masked` (its docstring has the passes and
the alignment), the independent check is `lofi_rl.grammar_mask`, and the source checks are
`lofi_rl.rl.trl_probe.source_checks` and `lm_head_record`.
- **Why.** vLLM applies the grammar's bitmask to the logits (`vllm/v1/worker/gpu_model_runner.py:4335`) before it
  samples (`:4340`). TRL asks it for the processed log-probs, so `sampling_per_token_logps` are normalized over the
  allowed set. TRL 1.17.1's trainer normalized its own log-probs over the whole vocabulary.
- D-020's IS gap is that difference (D-021, Why). Smoke 13657200 read it at 0.1538, token-weighted
  (`runs/grpo-smoke-13657200/summary/smoke.json`).
- The masked trainer takes its old, reference and loss log-probs over the same allowed sets vLLM sampled from.
- **What TRL's logs then mean.** TRL's `kl` is between the masked policy and the masked reference, since both come
  through the support. TRL's `entropy` is still the whole vocabulary's, not the masked policy's: the kernel keeps it
  so ("The entropy stays that of the whole vocabulary", `trl/kernels/chunked_logprob.py:483`, computed at `:598`), and
  the fill form logs the same quantity. A hack-watch that reads entropy reads the unmasked distribution's.
- **Both forms normalize in fp32.** Under TRL 1.17.1 the support path is already an fp32 masked log-softmax of bf16
  logits: autocast at `grpo_trainer.py:1566`, a bf16 logit tile (`chunked_logprob.py:510-512`, `:548`), fp32
  statistics (`:160`, `:353`, `:589-597`). The fill form (`grammar_masked_fp32`) takes bf16 logits too, so it is an
  independent re-implementation, not a change of numerics. After a measured miss, read launch a's
  `mask.selfcheck.masked` before spending the re-run.

### 8.1 The config

- **`trainer_logprobs`** (top level, optional) takes the values of `lofi_rl.rl.masked.TRAINER_LOGPROBS`:
  `grammar_masked` and `grammar_masked_fp32`.
  - `load_config` imports them only for a config that sets the key, so every other config never imports that module.
  - It refuses the key when the module cannot be imported, when the value is not one of them, and without
    `structured_outputs`, since the mask is the grammar's.
- **The path** (`RunConfig.mask_path`, from `masked.path_for`):
  - **`support_ids`** (`grammar_masked`). TRL 1.17.1's `_get_per_token_logps_and_entropies` writes a `sampling_mask`
    (B, C, K, padded with −1) into the completion columns of `support_ids` (`trl/trainer/grpo_trainer.py:1557-1561`).
    It then runs the fused LM head (`:1566-1567`), whose kernel normalizes each label's log-prob over that support.
  - TRL builds its own sampling mask only under filtered sampling: top_p < 1, top_k > 0 or min_p > 0 (`:1152-1161`).
    These settings are none of those, so the grammar's allowed sets go in as that mask.
  - The old pass (`:2762-2772`), the reference pass (`:2846-2856`) and the loss pass (`:3151-3169`) all pass it.
  - **`fill`** (`grammar_masked_fp32`) is D-021's registered logits form: the override's own fp32 log-softmax, with
    a finite fill outside the set. It is D-021's one named fix for a measured miss, unused until an amendment says so.
- **`config.MASK_T2_PATHS`:** which form of TRL's method (source_checks' T2 `path`) each mode can run under.
  - `grammar_masked` needs `support_ids`, since it hands TRL the support.
  - `grammar_masked_fp32` computes its masked log-probs itself, so it runs under either form.
- **`configs/grpo_smoke_v2_grammar_masked.yaml`** is `configs/grpo_smoke_v2_grammar.yaml` plus
  `trainer_logprobs: grammar_masked`, and every profile resolves to the same trainer settings
  (`tests/test_rl_train.py` pins both).
  - The job is unchanged: sha `1dded12d386c7fab`, at most 500 SU.
  - It runs on an L40S, as 13657200 did, so the numerics compare (the command is in §1).

### 8.2 A masked launch, in order (`lofi_rl.rl.train`)

1. **As before:** the GPU profile; the settings fitted to the installed GRPOConfig; the snapshot, the prompts, the
   tokenizer and the prompt lengths; the EOS ids. Then `GRPOConfig` is built.
2. **The source checks**, before the model loads: `trl_probe.source_checks(args=GRPOConfig.to_dict())`, into
   run.json `trl_source_checks`. `train.mask_refusals` refuses the launch (status `refused`, exit 2) unless:
   - every gating check reads `match` (T1-T5, and any other the record names);
   - the record's own verdict is `match`;
   - T2's path is one `MASK_T2_PATHS` allows for the mode.

   A source check that raises, or is missing, becomes a record that does not match.
3. **The mask's two halves**, over one vocabulary and one set of stop ids:
   - **vocab_size** comes from the snapshot's `config.json`. vLLM builds its xgrammar backend with
     `model_config.get_vocab_size()` (`vllm/v1/structured_output/__init__.py:144`), which is the model config's
     vocab_size (`vllm/config/model.py:1616-1617`).
   - **The stop ids** are the tokenizer's eos plus the generation config's `eos_token_id`: the set train.py already
     builds for G2. That is vLLM's `all_stop_token_ids` (`vllm/sampling_params.py:772-800`), which each request's
     grammar is compiled with (`vllm/v1/structured_output/__init__.py:199-205`). For Qwen3-4B-Instruct-2507 they are
     151643 and 151645 (`runs/grpo-smoke-13657200/summary/a/run.json`, `eos_token_ids`).
   - **`masked.XgrammarReplay(regex, tokenizer, vocab_size=, stop_ids=)`** is built as
     `vllm/v1/structured_output/backend_xgrammar.py:69-78` builds vLLM's. It takes the very regex string GRPOConfig
     hands vLLM, and the trainer's tokenizer.
   - **`grammar_mask.GrammarMask.from_tokenizer_json(<snapshot>/tokenizer.json, ...)`** is the independent NFA.
   - **`masked.MaskSetup`** holds both, and refuses a replay or an NFA built over another regex, vocabulary or
     stop-id set.
4. **The trainer class:** `masked.make_trainer_class(GRPOTrainer)`.
   - train.py refuses (exit 2) a class that is not a subclass of GRPOTrainer, or that inherits any of `_generate`,
     `_generate_and_score_completions` and `_get_per_token_logps_and_entropies` unchanged.
   - GRPOTrainer is then built from that class, with `mask_setup=`.
5. **T5's run-time half:** `trl_probe.lm_head_record(trainer.model)` on the PEFT model just built, into run.json
   `lm_head`. Anything but `match` refuses the launch (exit 2).
6. **Training.** A sampled token outside the mask raises `MaskViolation` in `_generate`, before any pass sees the
   rows: status `failed`, exit 1.

**run.json adds:**
- **For every config:** `trainer_logprobs` (null without the key) and `code`. `code` holds `$LOFI_CODE`, whether it
  and the imported tree hold scripts/snapshot.py's `.lofi-snapshot`, and whether they are one tree.
- **For a masked launch:** `trl_source_checks`, `lm_head` and `masked`.
  - `masked` is `MaskSetup.record()`: mode, path, regex sha, vocab size, stop ids, self-check rows, and the replay's
    and the NFA's own records.
  - It adds where the vocab size, the stop ids and the tokenizer came from, and the trainer class.
  - At the end it adds `stats_steps` and `last_stats`, the last step's record. A launch stopped by a MaskViolation
    never logs that record.

**steps.jsonl:** each step record of a masked launch has `mask`, the masked trainer's record of the rollouts that step
scored. It sits at the reward's offset: record k + 1 carries generation step k's. Its fields are
`masked.new_step_record`'s. An unmasked launch's lines are unchanged.

**The dry run** adds `trainer_logprobs` (M4) and `masked`.
- `masked` holds the mode, `path_expected`, the T2 paths the mode can run under, the source checks against the
  settings every profile shares, and `refusals`: what a launch would refuse on.
- On the laptop (2026-11-02) it reports regex `1412f11f…`, template `127e3094…`, reward check 0.2784511276657195 and
  `trainer_logprobs: grammar_masked`, as D-021's M4 asks.
- There every check reads `not_found` (no TRL, vLLM or PEFT is installed), so `refusals` lists them.
- It runs with torch, TRL, vLLM, PEFT and xgrammar blocked (`tests/test_rl_train.py`).

**Laptop evidence, 2026-11-02.** `trl_probe.source_checks` was run over copies of the cluster's installed TRL, vLLM
and PEFT, with 13657200's recorded `grpo_config` as the settings.
- T1-T5, V1 and V2 all read `match`; the path is `support_ids`; T3's `ratio_mean_over` is `sequences`.
- `train.mask_refusals` returns nothing for either mode.
- The dry run's shared settings give the same result. TRL's defaults are then read from `trainer/grpo_config.py`, for
  example `use_bias_correction_kl` true at `:977`.
- These are readings of copies. The launch reads the installed packages again, and that reading gates.

### 8.3 The smoke's verdict (`smoke.json` `d021`)

At the final stage of a masked config, `smoke_check` adds `d021`.
- **Each gate** of D-021 has its kind, value, bar, pass, source and reasons. `pass` is true, false, or null when the
  gate cannot be decided.
- **The verdict**, and what D-021 says follows it.

| gate | kind | read from | passes when |
|---|---|---|---|
| M5 | validity | both run.json `stamp`; the code tree smoke_check reads (the job's snapshot) | one commit, dirty=0, in both launches (`code` recorded beside); every file `docs/prereg/D-021_masked_smoke_path.txt` lists hashes to its blob, and `docs/DECISIONS.md` holds D-021 |
| M6 | validity | this report | GRPO_SMOKE_OK; stop out_of_language 0, other.n 0 and ≥ 1 stop completion a launch; xgrammar requested, pin accepted, read back and applied in both launches |
| M7 | validity | run.json `trl_source_checks`, `lm_head` | T1-T5 `match` in both launches, T5's run-time half too; V1 and V2 recorded |
| M8 | validity | run.json `masked`; each step's `mask` and TRL logs | see below |
| M9 | measured | `is_ratio` | the token-weighted mean of TRL's `sampling/sampling_logp_difference/mean` over every step ≤ 0.05 |
| M10 | validity | each step's `mask` | `neglogz` weighted by `neglogz_tokens`, over every step, ≥ 0.05 (M10's first clause) |
| M10_ratio | measured | M9 and M10 | M9 ≤ 0.25 × that −log Z (M10's ratio) |
| M11 | measured | each step's `mask.seq_weights` | at every step, the mean of the 32 weights ≥ 0.3 and the largest ≤ 0.25 × their sum |
| M12 | estimate-only | steps.jsonl `step_s` | the mean step_s ≤ 43.3 s |

- **M8, at every step:**
  - the replay checked all 32 completions, and found 0 violations;
  - 32 rows in each of the old, reference and loss passes;
  - 0 NFA mismatches, over more than 0 compared positions;
  - the mode's path, and TRL's `loss`, `kl` and `entropy` finite.

  **M8, per launch:** run.json names the path, and one vocabulary and stop-id set, which the NFA's and the replay's
  records repeat; no launch stopped on a MaskViolation.
- **M5's registration half** reads the tree smoke_check imports (`config.ROOT`): in the job that is the read-only
  snapshot the launches ran from, so the list and the entry are checked in the very code they froze. A missing,
  malformed or unreadable list, a file whose bytes no longer hash to its blob, or a DECISIONS.md without D-021 fails
  M5. The record keeps the list's sha256, its file count and up to 10 examples of each problem.
- **M9** needs every launch's TRL metric source to read `match` (as `is_ratio` does), and every step its value and
  its tokens.
- **M10** skips a step whose old pass had no valid token (every completion truncated and zeroed by
  `mask_truncated_completions`): it weighs 0 in a token-weighted mean, and `gates.M10.zero_weight` lists it. A missing
  or ill-typed record still leaves M10 undecided.
- **M12** compares with 1.15 × 13657200's mean step, 37.63 s over its 15 steps. Beside it, `breakdown` averages each
  `mask.timing` field and the step net of `unmasked_s` and `selfcheck_s`. It decides nothing.
- **`selfcheck_flags`** (diagnostics, not gates): a launch's first self-check is flagged when it is missing or errored,
  or its fill-vs-support difference has a mean over 0.05 or a max over 1.0. Each flag prints as a warning. On a
  `measured_miss`, a flag adds a note to what follows: D-021's fp32 fix presupposes a numerical miss, and a self-check
  that disagrees points at the GPU's support path instead.
- **The IS readout's caveat** for a masked config says the masked-trainer reading: had vLLM taken its log-probs before
  the bitmask, TRL's value would approach the token-weighted −log Z, and M10's ratio settles it. Unmasked configs keep
  D-020's caveat.
- **What TRL computes** for these readings:
  - The IS metric is the mean of |old − sampling| over the completion mask, NaN dropped
    (`trl/trainer/grpo_trainer.py:2979-2989`).
  - The applied weight is exp(Σ (old − sampling) · mask) per sequence in the sequence modes, set to 0 outside
    [clip_min, clip_max] under `sequence_mask` (`:2777-2820`). It multiplies the loss (`:3268-3269`).
  - The logged ratio mean is over the flattened (B, 1) tensor, so over sequences (`:2993-3012`).
  - The KL comes from the reference and loss-pass log-probs (`:3220-3227`). `kl` is logged under β ≠ 0 (`:3390`),
    `entropy` always (`:3392`), and `loss` by transformers (`trainer.py:2237`).
  - TRL's log averages each metric and drops NaN (`:3424-3443`).
- **The verdict:**
  - `invalid`: a validity gate did not pass, or a measured gate cannot be decided. The launch measured nothing
    (D-021: its defect is fixed, the fix recorded, and the smoke runs again).
  - `measured_miss`: every validity gate passed and a measured gate missed. That launch gets D-021's one re-run, with
    `grammar_masked_fp32`.
  - `pass`: every validity and measured gate passed.
  - M12 never changes the verdict: it decides only the run-1 estimate.
- **The log** prints one line per gate. Then `D021_PART_A <verdict>`, with the gates that did not pass, comes just
  before the last line, `GRPO_SMOKE_OK` or `GRPO_SMOKE_FAILED`. That last line keeps its meaning: the D-021 gates
  never add to the smoke's failures (M6 reads them), and the exit code follows it.
- **`diagnostics`**, recorded and never gated:
  - per step, the reward's token counts against the trainer's: the old pass's valid positions, and the tokens the
    replay checked;
  - each launch's first self-check;
  - the mask's shape: forced share, largest allowed set, cache misses and non-finite masked log-probs.
- **The IS readout's `says`**, for a masked config, names D-021's M9 rather than D-020's rule.

### 8.4 Choices made in this build

These are readings of D-021 for its smoke. D-021's amendment of 2026-11-02 records them, in the commit the smoke's
stamps name. Readings 1 and 6 are interpretations; the others restate the entry or are implementation, and decide
nothing.
1. **A measured gate that cannot be decided makes the launch `invalid`, not a measured miss.** D-021 calls a launch
   that measured nothing invalid. Its named fix (fp32) is for a numerical miss, and it applies only when M8 holds.
2. **M9 reads every step:** a step without TRL's value or its token count leaves M9 undecided. It does not fall back
   to the mean of the steps.
3. **M11's stand-in:** for a step without `seq_weights`, TRL's logged ratio mean and max stand in only when both
   launches' T3 reads that mean as over sequences (D-021's rule). Then the share is max ÷ (32 × mean), D-021's table.
4. **M10's two clauses** are two gates, `M10` (validity) and `M10_ratio` (measured), since D-021 gives them two kinds.
5. **The D-021 verdict stands beside the smoke's own.** `GRPO_SMOKE_OK` still means the smoke ran end to end, so a
   measured miss on a smoke that ran end to end exits 0, and its `D021_PART_A` line says the miss.
6. **`MASK_T2_PATHS`:** `grammar_masked_fp32` runs under either TRL form, or D-021's fix could never run on TRL
   1.17.1, whose form is `support_ids`.
7. **The source checks' settings:** a launch passes `GRPOConfig.to_dict()`: its resolved settings plus TRL's own
   defaults, which T1, T3 and T4 read. The dry run passes the settings every profile shares.
8. **M8's coverage:** "over every completion token" is read as the replay checking all 32 completions of every step,
   with TRL's own sampling mask read (not left `unread`).
9. **`trainer_logprobs` and `code`** are in every run.json. An unmasked launch's steps.jsonl is unchanged.
10. **Record names.** D-021's per-step records sit in each steps.jsonl line's `mask`: `mask/rows` is `mask.rows`
    ({old, ref, loss}), and `is/seq_weights` is `mask.seq_weights` (TRL's applied weights), with
    `mask.seq_weights_raw` beside it. TRL's own logs carry only scalar summaries: `mask/violations`, `mask/neglogz`,
    `mask/forced_share`, `mask/nfa_mismatch`, `is/seq_weight_mean` and `is/seq_weight_max_share`.
11. **What T2 checks.** T2 checks that TRL's statements are present: the support built from `sampling_mask`, placed in
    the completion columns, shifted with the labels and handed to the kernel. It does not trace the data flow. The
    alignment itself is checked numerically: by the preflight (§8.5) on the installed TRL, and at run time by M9, M10
    and the self-check.

### 8.5 The preflight (`jobs/mask_preflight.sbatch`, `lofi_rl.rl.mask_preflight`)

A CPU job on `main` (8 CPU, 32 GB, 20 min: at most 320 SU) that runs before the smoke, on the policy env, the parts of
the masked trainer the laptop cannot run. It decides nothing D-021 registered; a failure there is fixed before the
smoke is submitted. Its module docstring has the details; in short:
1. **The source checks** a launch makes before the model loads, with the settings every profile shares.
2. **The mask's two builders**, built exactly as `train.build_mask` builds them, walked along the same ids and
   compared position by position:
   - the tokenizer's ids of every in-language text: prompt v2's 9 references and D-020's 980 completions;
   - the same texts one character per token, so every character boundary is a token boundary;
   - 400 seeded random walks, 300 drawn from the NFA's allowed sets and 100 from xgrammar's own matcher.

   The job stops if any of D-020's three completion files is missing from scratch, and the verdict needs all 980.
3. **TRL's support path on a tiny random Qwen3.** TRL's own `_get_per_token_logps_and_entropies`, with the support
   `masked.build_support` makes, is checked against the fill form, a float64 masked log-softmax and their gradients.
   - The vocabulary spans three of the kernel's 32768-wide tiles, and the supports reach the last one.
   - A support one column off must not agree.
   - One call takes TRL's compact packing, the GPU's, through a patched `trl.trainer.utils._use_triton`; it must
     equal the non-compact call.
   - The CPU runs the kernel's torch fallback, so the Triton kernels are first exercised by the smoke's self-check.
4. **The trainer's own glue on real objects:**
   - `adapter_state_of` outside and inside `disable_adapter()`;
   - `lm_head_record` (T5's run-time half);
   - one step of `MaskedPasses` with `TorchOps` through TRL's real method: old, reference, two loss micro-batches,
     then `after_score`.

It ends its log with `MASK_PREFLIGHT_OK` or `MASK_PREFLIGHT_FAILED <reasons>` and writes
`$REMOTE_SCR/runs/mask-preflight-<job>/summary/preflight.json`.
