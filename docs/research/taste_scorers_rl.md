# lofi-rl research report: capturing one listener's taste, and what the scorers and RL stack allow (as of 2026-10-30)

Local context: `/Users/shresthrana/csci596-project/CLAUDE.md` is the only file in the project directory. There is no `CHANGELOG.md` and no `docs/` yet. Arithmetic marked "(mine)" is my own derivation and is not cited.

---

## A. Preference elicitation from a single listener

### A1. Pairwise vs best-worst vs k-way ranking

**Information per trial (mine; noise-free upper bounds):**

| Protocol | Clips per trial | Bits per trial | Pairs implied |
|---|---|---|---|
| Pairwise A/B | 2 | 1 | 1 |
| Best-worst of 4 (BWS-4) | 4 | log2(12) = 3.58 | 5 of 6 ([Kiritchenko & Mohammad 2017](https://arxiv.org/abs/1712.01765)) |
| Full rank of 4 | 4 | log2(24) = 4.58 | 6 |

- Per clip heard, BWS-4 and rank-4 carry roughly 1.8x and 2.3x the bits of A/B (mine). This only holds if each clip is heard about once. Memory load grows with k.
- MUSHRA caps a trial at 12 signals and relies on free switching between them ([BS.1534-3](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1534-3-201510-I!!PDF-E.pdf)).
- BWS was significantly more reliable than rating scales at equal annotation counts, and the gap was largest at small budgets (text-sentiment study, not audio) ([1712.01765](https://arxiv.org/abs/1712.01765)).

**Turning rankings into Bradley-Terry (BT) pairs:**
- Expand a k-way ranking into all C(k,2) pairs. Train each ranking's pairs as one batch element. InstructGPT found that shuffling these correlated pairs as independent examples overfit the reward model ([2203.02155](https://arxiv.org/abs/2203.02155)).
- For BWS, either use the 5 implied pairs (down-weighted), or fit the MaxDiff likelihood P(best=i, worst=j) ∝ exp(u_i − u_j) directly ([BWS overview](https://en.wikipedia.org/wiki/Best%E2%80%93worst_scaling)).
- Naive rank-breaking can give inconsistent estimates; weighted breaking fixes this ([Khetan & Oh](https://arxiv.org/abs/1601.05495)).

**Fit to GRPO (mine):** GRPO advantages are computed within one prompt's group. Ranking or BWS over 4 completions of the same brief yields exactly the within-group labels the reward model must get right.

### A2. Preference strength, ties, "both bad", reason tags

- **Margins.** Llama 2 subtracts a rating-dependent margin in the reward-model loss. It helped on clearly separable pairs and hurt on similar ones ([2307.09288](https://arxiv.org/abs/2307.09288)). Llama 3 dropped it after scaling data ([2407.21783](https://arxiv.org/abs/2407.21783)).
- **Thurstone vs BT.** Thurstone Case V (probit) is close to BT (logit). The pwcmp toolbox handles ties, incomplete designs, outliers, confidence intervals and priors for few observers ([1712.03686](https://arxiv.org/abs/1712.03686)).
- **Ties.** BT-with-ties (Rao-Kupper) beat plain BT when the data contains ties ([2410.05328](https://arxiv.org/abs/2410.05328)). Rao-Kupper and Davidson tie models performed similarly inside DPO ([2409.17431](https://arxiv.org/abs/2409.17431)).
- **Graded labels.** An ordinal cumulative-link model (Agresti 1992) can treat slight / clear / tie as one scale. UNVERIFIED (not fetched).
- **Low-confidence votes.** Stiennon et al. found them less useful than high-confidence votes, but still better to keep than to drop. Filtering out weaker labelers did not help ([2009.01325](https://arxiv.org/abs/2009.01325)).
- **"Both bad" is not a tie (design suggestion).** It is absolute information. Options:
  - store it as its own label;
  - train an acceptability head that feeds the gate;
  - or add a fixed-score virtual "acceptable" anchor item, so "both bad" becomes anchor > A and anchor > B.
- **Reason tags.** Fine-grained labels improve reward models ([2306.01693](https://arxiv.org/abs/2306.01693), not re-fetched). For about 1.5k votes, multi-label tags work as an auxiliary head that regularizes the small taste head and helps catalog hacks.

### A3. Presentation controls

- **Order.** Randomize presentation order within and between sessions ([BS.1116-3](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1116-3-201502-I!!PDF-E.pdf)).
- **Switching.**
  - BS.1116-3 prefers near-instant switching, about 40 ms in total for fade-down, change-over and fade-up. Switches must be click-free and usable with eyes closed.
  - BS.1534-3 requires 5 ms raised-cosine fades on every switch, never cross-fades, and loops of at least 500 ms.
  - [webMUSHRA](https://github.com/audiolabs/webMUSHRA) implements this: Web Audio, sample-accurate switching, looping, AB/ABX/paired modes, YAML config.
  - Because the palette fixes tempo, every loop has the same length and bar grid. Switching between A and B at the same bar position is therefore trivial.
- **Clip length.**
  - MUSHRA recommends about 10 s, preferably no more than 12 s. Longer clips invite primacy and recency effects and must be justified in the report (BS.1534-3).
  - Counterpoint: in MusicRL users compared 20 s clips, and reward-model accuracy fell from 60% to 56% when inputs were cut to 3-5 s ([2402.04229](https://arxiv.org/abs/2402.04229)).
- **Loudness.**
  - Both standards require excerpt loudness to be aligned before the test. MUSHRA limits the listener's own level adjustment to within 4 dB of reference.
  - Integrated −14 LUFS over a full loop does not equalize a 10 s excerpt; re-normalize each excerpt (mine).
  - How much small level mismatches bias preference is UNVERIFIED (only anecdotal evidence found).
- **Position bias.** Time-order errors change direction with stimuli and interval, so counterbalance rather than correct for a fixed direction. Log which side played first, which was heard last before the vote, and how many switches. Add a position term to the BT model, and present repeats with sides reversed.

### A4. Self-consistency and the noise ceiling

**Ceiling formula (mine).** If each vote matches the latent preference with probability q, two votes on the same pair agree with probability p = q² + (1−q)². So the best possible model agreement is q = (1 + √(2p−1)) / 2.

| Repeat agreement p | 0.58 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 |
|---|---|---|---|---|---|---|
| Ceiling q | 0.70 | 0.72 | 0.77 | 0.82 | 0.85 | 0.89 |

- If repeat agreement is below 58%, the 70% target cannot be reached. The formula is approximate when pair difficulty varies.

**Evidence on self- and inter-rater agreement:**
- Music similarity, same raters two weeks apart: intra-rater ρ = 0.80 on 5-genre material but only 0.38 on single-genre material. Mood stability raised intra-rater agreement ([Flexer et al., TISMIR 2021](https://transactions.ismir.net/articles/10.5334/tismir.107)).
- Music preference ratings, test-retest r = .65 to .88 (LeBlanc 1998). UNVERIFIED (search snippet only).
- Text summaries: researcher-researcher agreement was about 65% on two samples from the same RL policy and about 80% on the easiest pairs ([2009.01325](https://arxiv.org/abs/2009.01325)).
- MusicRL authors agreed with crowd labels 60% of the time ([2402.04229](https://arxiv.org/abs/2402.04229)).
- TuneJury: two authors agreed with each other on 26 of 30 pairs, and with Arena votes 63-70% ([2606.17006](https://arxiv.org/abs/2606.17006)).
- An "80% self-agreement on duplicated pairs" target is practitioner lore only. UNVERIFIED.

**Statistical precision (mine):**
- 200 held-out votes at 70% gives a 95% CI of about ±6.4 pp.
- 100 briefs (the final blind A/B) gives about ±9 pp.
- 100-150 repeated pairs estimate p to about ±7-9 pp.
- Spread repeats across days, since mood matters.

### A5. Session length and fatigue

- **Standards.** BS.1116-3: a session should last no more than 20-30 min, with about 10-15 trials, and rest between sessions at least as long as the session. The older BS.562-3 said 15-20 min.
- **Hidden anchors.** BS.1116 plants known anchors to check the listener. MUSHRA post-screening drops assessors who rate the hidden reference below 90 on more than 15% of items. An analog here: plant obviously broken loops at about 5% of trials to detect fatigue.
- **Time budget (mine; assumes 80 BPM, so an 8-bar loop is 24 s):**
  - Full loops played A then B: about 55 s per vote, so 1,500 votes is about 23 h, or about 55 sessions of 25 min.
  - 10 s excerpts with switching: about 25 s per vote, so about 10 h, or about 25 sessions.
  - BWS-4: about 65 s per tuple, so 300 tuples (about 1,500 implied pairs, which are correlated) is about 5.4 h.

### A6. Active learning and how many pairs are needed

**Methods:**
- ASAP (expected information gain plus approximate message passing) gave the most accurate scores for a fixed item set ([2004.05691](https://arxiv.org/abs/2004.05691)).
- Randomly chosen BT pairs are inefficient under tight budgets ([2511.12796](https://arxiv.org/abs/2511.12796)).
- Uncertainty sampling with instance-dependent guarantees ([2602.01581](https://arxiv.org/abs/2602.01581)).

**Caveats:**
- Uncertain pairs are near-ties, which are the least self-consistent votes. Mix active and random selection.
- Keep a frozen, random, pre-registered held-out set, built from same-brief, on-policy pairs as GRPO sees them.

**Sample size: no published single-listener learning curve was found.** Population-level evidence:
- TuneJury (frozen CLAP + MERT-330M, 2.8M-parameter MLP): 17.5k pairs reached 70.9% held-out; 5.1k pairs reached 66.8% ([2606.17006](https://arxiv.org/abs/2606.17006)).
- MusicRL: 285k crowd pairs reached 60%.
- Gao et al.: text reward models trained on fewer than about 2,000 comparisons stayed near chance ([2210.10760](https://arxiv.org/abs/2210.10760)).
- Implication (mine): with about 1,300 training votes, use a low-capacity head (PCA to 32-64 dims plus L2 logistic BT, or a Gaussian-process preference model). Fit learning curves at 200 / 400 / 800 votes.

**Workflow constraint.** CLAUDE.md forbids training on the laptop. Active pair selection therefore needs either a policy exception for a CPU-only refit in seconds, or a pair queue built nightly on the cluster.

### A7. Reward-model over-optimization and mitigations for GRPO

- **Gao et al. 2022.** With d = √KL, gold score follows d(α − βd) for best-of-n and d(α − β log d) for RL. The KL penalty raised the proxy score but not the gold-score frontier, so it acted like early stopping. More reward-model data reduced Goodharting ([2210.10760](https://arxiv.org/abs/2210.10760)).
- **Ensembles.**
  - Worst-case and uncertainty-weighted ensembles nearly eliminated over-optimization in best-of-n (up to 70% gain) and reduced it in PPO ([2310.02743](https://arxiv.org/abs/2310.02743)).
  - Ensembles with different pretraining beat ensembles with different fine-tuning seeds, but shared error patterns were still exploited ([2312.09244](https://arxiv.org/abs/2312.09244)). The analog here is an ensemble over different backbones (MERT / CLAP / symbolic features).
- **Bounded, centered reward.** PAR, sigmoid(r − r_ref), substantially reduced hacking ([2502.18770](https://arxiv.org/abs/2502.18770)).
- **On-policy refresh.** Llama 2 collected new preferences from the latest policy each round because reward-model accuracy degrades off-distribution ([2307.09288](https://arxiv.org/abs/2307.09288)).
- **Music-specific evidence.**
  - MusicRL over-optimized its user-preference reward model in under 2,000 steps, even with KL 0.001.
  - SMART (TRL GRPO, Audiobox CE reward, β 0.04, 8 prompts x 8 completions, 200 iterations) produced more notes, more polyphony, wider pitch range, more chromaticism and fewer empty beats. A long β = 0 run collapsed output diversity ([2504.16839](https://arxiv.org/abs/2504.16839)).
- **Length and duplication.** TRL's default `loss_type="dapo"` removes length bias. `mask_truncated_completions` follows DAPO. Within-group de-duplication targets the diversity collapse SMART reported.

---

## B. The audio scorers

### B1. Meta Audiobox Aesthetics

**Axes** (each scored 1-10; definitions from [2502.05139](https://arxiv.org/abs/2502.05139)):
- PQ (Production Quality): technical quality only — clarity, fidelity, dynamics, spatialization.
- PC (Production Complexity): how many audio components are mixed together.
- CE (Content Enjoyment): subjective, artistic enjoyment.
- CU (Content Usefulness): how usable the audio would be as source material for a creator.

**Model and inputs:**
- WavLM-style encoder, 12 layers, 768-d, about 0.1B parameters.
- Input is resampled to 16 kHz mono.
- Inference splits audio into non-overlapping 10 s windows and returns a length-weighted average.
- Training used about 97k clips, about 500 h. Per the summary, no synthetic audio was in training, so SoundFont renders may be out of distribution.

**Package and API:**
- PyPI `audiobox_aesthetics` 0.0.4 (May 2025), Python ≥3.9, torch ≥2.2. Mostly CC-BY 4.0 ([GitHub](https://github.com/facebookresearch/audiobox-aesthetics), [PyPI](https://pypi.org/project/audiobox-aesthetics/)).
- Python: `initialize_predictor().forward([{"path": ...}])`, or pass `{"path": tensor, "sample_rate": sr}`.
- CLI: `audio-aes in.jsonl --batch-size 100`; supports `start_time`/`end_time` per line and has Slurm options.

**Behaviour on music:**
- On PAM-music, utterance-level Pearson was 0.661 for CE, 0.587 for PQ and 0.710 for PC. The authors say it is slightly worse than PAM on music.
- Rendering choice moved mean CE by more than 1 point (5.42 vs 6.60 across two renderers) ([SMART](https://arxiv.org/abs/2504.16839)).
- Optimizing PQ alone raised all Audiobox scores about 10%, while an independent alignment check dropped ("verifier hacking") ([SCORE](https://arxiv.org/abs/2509.19831)).

**Runtime:** no published figures. Estimate (mine): about 350 GFLOP per 24 s clip, i.e. about 10-40 ms on an A100, and roughly 10 s per CPU core. UNVERIFIED; measure in `smoke_env`.

### B2. LAION-CLAP

**laion_clap package** (1.1.7, [GitHub](https://github.com/LAION-AI/CLAP)):
- Music checkpoint `music_audioset_epoch_15_esc_90.14.pt` (ESC-50 90.14%, GTZAN 71%) needs `amodel='HTSAT-base'` and `enable_fusion=False`.
- Input must be 48 kHz. Non-fusion models target clips of 10 s or less.
- The README example round-trips audio through int16 before embedding.
- Loading may fail with an unexpected `position_ids` key; the workaround is to strip that key from the checkpoint ([issue #127](https://github.com/LAION-AI/CLAP/issues/127)).

**Hugging Face `laion/larger_clap_music`** (Apache-2.0 per its card):
- Used via `ClapModel` / `ClapProcessor`.
- Its preprocessor config ([config](https://huggingface.co/laion/larger_clap_music/raw/main/preprocessor_config.json)) is 48 kHz, 10 s maximum, `truncation="rand_trunc"`, `padding="repeatpad"`, 64 mel bins, 50-14,000 Hz.
- `rand_trunc` takes a random crop of longer audio. The class default `"fusion"` also uses random crops ([docs](https://huggingface.co/docs/transformers/model_doc/clap)).
- So a 21-27 s loop gets a nondeterministic embedding unless you chunk it yourself into fixed, bar-aligned windows of 10 s or less.
- Whether this checkpoint is the same as the 90.14 one is UNVERIFIED.

**Prompt practice:**
- The original CLAP template is "This is a sound of [label]". LAION evaluated GTZAN with "This audio is a song.".
- Bare labels often beat templates; no universal template exists ([2409.13676](https://arxiv.org/abs/2409.13676)). Use prompt ensembles, and score relative to a fixed set of other themes rather than raw cosine.

**Known failure modes:**
- Negation is barely encoded ([2607.12290](https://arxiv.org/abs/2607.12290)).
- Event order is often confused (T-CLAP literature).
- CLAPScore correlates weakly with human ratings; Human-CLAP raised Spearman correlation by more than 0.25 ([2506.23553](https://arxiv.org/abs/2506.23553)).
- MuLan text-adherence predicted MusicRL user preferences only 51.6% of the time.
- The fixed rain bed is in every mix and may saturate rain-related themes (hypothesis). Consider scoring CLAP on a render without the bed.

### B3. MERT

**Variants:**

| Model | Layers x dim | Hidden states | Training data | License |
|---|---|---|---|---|
| v1-95M | 12 x 768 | 13 | 20k h | CC-BY-NC-4.0 |
| v1-330M | 24 x 1024 | 25 | 160k h | CC-BY-NC-4.0 |

- Both take 24 kHz input and produce 75 frames per second. Pretraining context was 5 s, but the convolutional relative position embedding allows longer inputs. Downstream evaluations average over 5 s windows ([MERT paper](https://arxiv.org/abs/2306.00107); model cards for [95M](https://huggingface.co/m-a-p/MERT-v1-95M) and [330M](https://huggingface.co/m-a-p/MERT-v1-330M)).

**Compatibility:** requires `trust_remote_code`. It breaks on transformers ≥4.44 because the config lacks `conv_pos_batch_norm`. The posted workaround sets `config.conv_pos_batch_norm = False` before loading, but it is untested on the 95M model ([discussion #4](https://huggingface.co/m-a-p/MERT-v1-95M/discussions/4)). Whether `nnAudio` is needed (CQT path only) is UNVERIFIED.

**Which layers carry what:**
- The best layer depends on the task; the cards suggest choosing empirically or learning weights over all layers.
- Top layers lean toward the pretraining objective ([MERT paper](https://arxiv.org/abs/2306.00107), App. C).
- Pitch and chord-root information strengthens in deeper layers, and timbre is retained less ([NeurIPS 2024 workshop](https://neurips.cc/virtual/2024/105747)).
- On MERT-330M, melodic information sits at layer 23 and rhythm at layers 3-6 ([MERIT](https://arxiv.org/abs/2605.27346); suggestive, different setup).
- A clean timbre-to-layer mapping for MERT is UNVERIFIED.

**Runtime:**
- Community Apple M1 Pro figures for the 95M model: 0.34-0.78 s per 30 s window ([community port](https://hf-cn0.pages.dev/khersh/MERT-v1-95M-apple-silicon)).
- Estimate (mine): A100 about 20-50 ms per 24 s clip for 95M and about 3.5x that for 330M. UNVERIFIED.

### B4. Which embedding for the personal taste head

- **Concatenation did not help at population scale.** TuneJury ablation (single seed, 17.5k pairs): CLAP audio only 0.705, MERT only 0.695, CLAP + MERT 0.701, all three 0.705 ([2606.17006](https://arxiv.org/abs/2606.17006), App. C). Every audio variant fell inside a 0.013 band. MERT was weaker on MusicPrefs (0.650 vs 0.733).
- **Generic quality and semantics predict individual preference poorly.** In MusicRL, quality predicted user preference 53.3% of the time and text-adherence 51.6%. CLAP's semantic advantage shrinks on user-preference ranking ([2505.23298](https://arxiv.org/abs/2505.23298)).
- **Reasoning for lo-fi (inference):** timbre, effects, tempo and the bed are fixed, so loops differ mainly in pitch, harmony and rhythm content. That favours a layer-weighted MERT-95M over CLAP. Also test a symbolic baseline built from LoopScript features (free and deterministic).
- **Recommendation:** with about 1.3k votes, avoid concatenation unless it is regularized hard (PCA). Choose by cross-validated log-loss on the first 400-600 votes.

---

## C. The RL stack (late 2026)

### C1. Versions

- **TRL** 1.17.1, released 2026-10-30 ([PyPI](https://pypi.org/project/trl/)). Requires Python ≥3.11, transformers ≥4.56.2, accelerate ≥1.11, datasets ≥4.7, peft ≥0.17, and vLLM ≥0.22.1 and <0.33 ([pyproject](https://raw.githubusercontent.com/huggingface/trl/main/pyproject.toml)).
- **vLLM** 0.32.0, released 2026-10-18 ([PyPI](https://pypi.org/project/vllm/)).
  - It pins `transformers>=5.16.1,<5.20.0` and `tokenizers>=0.23.2` ([release notes](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)).
  - PyPI wheels default to CUDA 13.0; CUDA 12.9 wheels are also published. Source builds need GCC 13.
  - It probably targets PyTorch 2.15. UNVERIFIED.
- **transformers** latest is 5.22.1, which is outside vLLM's pin ([PyPI](https://pypi.org/project/transformers/)).
- **Working set:** TRL 1.17.x + vLLM 0.32.0 + transformers 5.19.x. Check that the CARC driver supports CUDA 13; otherwise use the cu129 wheels.

### C2. GRPOConfig defaults (main docs, [grpo_trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer))

- **vLLM:** `use_vllm=False` (must be turned on), `vllm_mode="colocate"`, `vllm_gpu_memory_utilization=0.3`, `vllm_tensor_parallel_size=1`, `vllm_enable_sleep_mode=False`, `vllm_structured_outputs_regex=None`. The regex option can constrain LoopScript syntax during generation.
- **Batching:** `num_generations=8`. The effective batch (processes x per-device batch x grad-accum) must be divisible by it. `generation_batch_size` and `steps_per_generation` are mutually exclusive. `max_completion_length=512`.
- **Loss and KL:**
  - `beta=0.0`, so no reference model is loaded and no KL is applied.
  - `loss_type="dapo"`. Other options: grpo, dr_grpo, bnpo, cispo, sapo, luspo, vespo.
  - `scale_rewards="group"`. Alternatives are "batch" or "none"; Dr. GRPO recommends no scaling.
  - `multi_objective_aggregation="sum_then_normalize"`, or "normalize_then_sum" (GDPO).
- **Sampling:** `temperature=1.0`, `top_p=1.0`, `top_k=0`, `num_iterations=1`, `epsilon=0.2`. Also available: `chat_template_kwargs` (e.g. to set `enable_thinking=False`), `generation_kwargs`, `remove_unused_columns=False`.
- **Importance-sampling correction:** on by default with vLLM (`sequence_mask` mode, clip max 3.0). If you sample with top_p < 1, top_k > 0 or min_p > 0, the ratios are biased unless you run colocate mode on vLLM ≥0.28 with top_k > 0.
- **Different from transformers TrainingArguments:** `learning_rate=1e-6`, `gradient_checkpointing=True`, `bf16=True`, `logging_steps=10`.

### C3. Reward functions

**Signature.** `f(prompts, completions, completion_ids, trainer_state, log_extra, log_metric, [environments], **dataset_cols) -> list[float | None]`.
- Every dataset column except `prompt` arrives as a keyword argument. Functions may be sync or `async`.
- With conversational datasets, completions are lists of messages, so read `c[0]["content"]`.
- Returning `None` for a sample excludes that function from that sample's reward.

**Combining rewards.**
- Multiple functions are summed, weighted by `reward_weights`. The weights are static, so implement the curriculum inside the functions using `trainer_state.global_step`.
- The gate is multiplicative, so each component function must apply it, or all components can read from a shared per-completion render/score cache (design suggestion).

**Logging.** Each function gets `rewards/<name>/mean` and `rewards/<name>/std`. Also logged: `reward`, `reward_std`, `frac_reward_zero_std`, `completions/clipped_ratio`, `entropy`, and `kl` (only when beta ≠ 0).

### C4. LoRA and vLLM

- **Setup.** Pass `peft_config=LoraConfig(...)`. With PEFT, the reference model is the base with the adapter disabled, so no extra copy is held.
- **Weight sync.** Parameters are gathered, `merge_adapter()` runs, merged base weights are pushed to vLLM (colocate: `load_weights`; server: NCCL `update_named_params`), then `unmerge_adapter()`, then `reset_prefix_cache()` ([vllm_generation.py](https://raw.githubusercontent.com/huggingface/trl/main/trl/generation/vllm_generation.py)). vLLM's own LoRA support is not used.
- **Async alternative.** AsyncGRPOTrainer (TRL ≥1.14) can sync only the adapter, via disk and `/v1/load_lora_adapter`. It needs a shared filesystem and `--enable-lora` ([blog](https://huggingface.co/blog/asyncgrpo-lora-hfjobs)).
- **Hyperparameters.** For RL, rank 1-32 on all linear layers is enough. Use about 10x the full fine-tune learning rate (TRL suggests 1e-5; one example uses 5e-5 with r=1, α=32). LoRA tolerates large effective batches poorly, so stay below about 32 ([LoRA Without Regret](https://huggingface.co/docs/trl/main/en/lora_without_regret)).

### C5. Multi-GPU on Discovery

- **Server mode** starts the server with `VLLM_SERVER_DEV_MODE=1 vllm serve <model> --weight-transfer-config '{"backend":"nccl"}' --logprobs-mode processed_logprobs --max-logprobs -1`, on GPUs separate from the trainer. `trl vllm-serve` is deprecated. Since vLLM 0.14, data parallelism no longer helps dense models; scale with tensor parallelism ([vllm_integration](https://huggingface.co/docs/trl/main/en/vllm_integration)).
- **Discovery GPU nodes:**

| GPU | Per node | Node CPU / RAM | Notes |
|---|---|---|---|
| A100 | 2 | 64 cores, 248 GB | |
| L40S (48 GB) | 3 | 64 cores, 748 GB | |
| A40 | 2 | | also on `debug` |

  - At most 36 GPUs per user. CARC's docs use `--gpus-per-task=a100:1 --constraint=a100-80gb` ([CARC GPU guide](https://carc.usc.edu/user-guides/advanced-hpc-programming/gpu-programming), [resource overview](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/resource-overview-discovery)).

### C6. 4B-8B open instruct models (late 2026)

| Model | License | Architecture notes | Instruction-following | Integration risk |
|---|---|---|---|---|
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Apache-2.0 | Dense, 36 layers, GQA, non-thinking only, 262K context | IFEval 83.4, BFCL-v3 61.9 | Lowest |
| [Granite 4.1 8B](https://huggingface.co/ibm-granite/granite-4.1-8b) (2026-04-29) | Apache-2.0 | Dense, GQA, 128K context | IFEval 87.06, BFCL v3 68.27 | Low |
| [Qwen3.5-4B / 9B](https://huggingface.co/Qwen/Qwen3.5-4B) (2026-03-02) | Apache-2.0 | Hybrid: 3 Gated-DeltaNet layers per gated-attention layer, vision encoder, thinks by default | IFEval 89.8 / 91.5 (mode unstated) | Higher; training kernels for the hybrid layers UNVERIFIED |
| [Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it) (Apr 2026) | Apache-2.0 | 4.5B effective / 8B with per-layer embeddings; text, image and audio in | IFEval not reported | Medium |
| [Ministral 3 8B Instruct 2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) | Apache-2.0 | 8.4B language model + 0.4B vision | Not reported | Recommends temperature below 0.1; GRPO samples at 1.0 |
| [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Llama 3.1 Community License | Dense | IFEval 80.4 (UNVERIFIED) | Low |
| OLMo 3 7B Instruct ([listing](https://openrouter.ai/allenai/olmo-3-7b-instruct-20251121)) | Apache-2.0 | Fully open data | IFEval not found | Low |

- No new Qwen small text model appears on the [Qwen org page](https://huggingface.co/Qwen); the smallest Qwen3.8 text model is 27B. No 2026 Llama small model was found (UNVERIFIED).
- No benchmark covers custom-DSL strictness. The Phase 1 junk-rate gate on 2-3 candidates is the real test.

---

## The 10 facts most likely to change design decisions

1. **The 70% target may be at or above Shresth's own consistency ceiling.** The ceiling is about (1+√(2p−1))/2 for repeat agreement p, so p must be at least 58%. Single-genre material cut intra-rater ρ from 0.80 to 0.38 ([Flexer](https://transactions.ismir.net/articles/10.5334/tismir.107)). Two samples from the same policy (GRPO's case) were the hardest pairs, at about 65% ([Stiennon](https://arxiv.org/abs/2009.01325)). 200 held-out votes give ±6.4 pp. Measure repeat agreement in week one.
2. **1,500 votes is thin by published standards.** Population models needed 17.5k pairs for 70.9% and got 66.8% with 5k ([TuneJury](https://arxiv.org/abs/2606.17006)); MusicRL reached 60% with 285k ([2402.04229](https://arxiv.org/abs/2402.04229)); text reward models were near chance below about 2k ([Gao](https://arxiv.org/abs/2210.10760)). CLAP+MERT concatenation gave no gain. Use low-dimensional heads, symbolic features and learning curves.
3. **The listening budget breaks the Nov 9-15 window as planned.** BS.1116 allows 20-30 min sessions with equal rest. 1,500 full-length sequential A/Bs is about 23 h (about 55 sessions). 10 s bar-aligned excerpts with synced switching cut this to about 10 h; BWS-4 (5 of 6 pairs per trial) to about 5 h.
4. **Audiobox may push against lo-fi sparseness.** A CE reward under GRPO produced denser, more polyphonic, more chromatic output and diversity collapse ([SMART](https://arxiv.org/abs/2504.16839)). PC literally scores the number of components. The renderer shifts CE by more than 1 point, and the model saw no synthetic audio in training. Validate each axis against votes before giving it weight, and don't reward PC.
5. **CLAP rewards are nondeterministic by default.** HF `larger_clap_music` randomly crops audio longer than 10 s (`rand_trunc`), and 8-bar loops are longer. Chunk deterministically. CLAP also fails on negation and correlates weakly with human ratings.
6. **The scorer environment conflicts with the RL environment.** MERT remote code breaks on transformers ≥4.44 (config override needed), while vLLM 0.32 needs transformers 5.16-5.19. laion_clap needs a checkpoint patch. MERT is CC-BY-NC. vLLM 0.32's default wheels are CUDA 13. Plan the scorer environment separately in `smoke_env`.
7. **TRL defaults have changed.** No KL or reference model (beta 0), DAPO loss, and group-std reward scaling, which amplifies reward-model noise when completions are similar ("batch"/"none" and GDPO exist). `reward_weights` are static, so the curriculum must go through `trainer_state`. Sampling with top_p/top_k biases the IS correction unless run in colocate mode on vLLM ≥0.28 with top_k > 0.
8. **Discovery A100 nodes hold only 2 GPUs (L40S nodes 3).** The "1→4 GPU step-time" scaling study needs multi-node or L40S nodes. Colocate vs server mode is a real trade-off on 2 GPUs. The 64 cores on a GPU node can render rollouts in-node.
9. **Model choice risk.** Qwen3.5-4B is a hybrid Gated-DeltaNet + vision model that thinks by default. Dense, non-thinking, Apache-2.0 options (Qwen3-4B-Instruct-2507, Granite 4.1 8B) are lower-risk for TRL + vLLM + LoRA. `vllm_structured_outputs_regex` can enforce LoopScript syntax; how it interacts with the IS correction is UNVERIFIED.
10. **Over-optimization arrives fast and KL won't fix it.** A music user-preference reward model was over-optimized in under 2,000 steps; KL acted only like early stopping. Plan for:
    - early stopping on held-out human votes;
    - an ensemble over different backbones (mean − λ·std);
    - a bounded, centered reward (PAR);
    - nightly votes on on-policy samples (Llama-2-style refresh);
    - logging LoopScript hack signatures (note density, polyphony, chromaticism, rests).