"""The RL loop for lofi-rl (docs/design/taste_pipeline.md §5-6; docs/GRPO.md).

  junk         the base-model test (junk-rate): generate with vLLM, score the gates, render a sample
  curriculum   the reward weights w(t): theory 0.5 -> 0.1 over steps 0-30, knobs 0.15
  reward       make_reward(palette_path, log_path): the TRL reward function, one JSONL line per completion
  config       configs/grpo_*.yaml: load, validate, pick a GPU profile, fit TRL's GRPOConfig fields
  train        python -m lofi_rl.rl.train --config configs/grpo_smoke.yaml [--resume-from <ckpt>] [--dry-run]
  smoke_check  the verdict of jobs/grpo_smoke.sbatch: checkpoints, the resumed step count, smoke.json

Nothing here imports torch, TRL, vLLM or the audio stack at module level, so every module imports on the
laptop and in the policy env (tests/test_rl_imports.py); train.py and junk.py import them inside a run.
"""
