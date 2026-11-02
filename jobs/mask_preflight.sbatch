#!/bin/bash
#SBATCH --job-name=lofi-mask-preflight
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# The masked trainer's preflight (D-021 Part A; docs/GRPO.md §8), on a CPU node of the policy env, before the GPU
# smoke spends a launch: python -m lofi_rl.rl.mask_preflight (its docstring has the four checks):
#   1. the source checks train.py makes before the model loads, over the installed TRL, vLLM and PEFT;
#   2. xgrammar's replay and the NFA, built as train.build_mask builds them for a launch, walked along the real
#      tokenizer's ids of prompt v2's references and of D-020's 980 completions (junk 13640325's completions.jsonl,
#      smoke 13657200's a and b rewards.jsonl), along the same texts one character per token, and along seeded random
#      walks drawn from each builder's allowed sets: their allowed sets must agree everywhere. The three files must be
#      on scratch: the job stops, before the preflight runs, if one is missing;
#   3. TRL's own support path on a tiny random Qwen3 whose vocabulary spans three of the kernel's vocabulary tiles,
#      against the fill form, a float64 masked log-softmax and their gradients, a support one column off that must
#      not agree, and TRL's compact packing (the GPU's) against its non-compact one;
#   4. the masked trainer's glue on the same PEFT model: the adapter state, T5's run-time half, and one step of
#      masked.MaskedPasses through TRL's real method.
# It decides nothing D-021 registered (the gates are read from the smoke); it ends the log with MASK_PREFLIGHT_OK or
# MASK_PREFLIGHT_FAILED <reasons>.
#   make submit JOB=jobs/mask_preflight.sbatch
# Results: $REMOTE_SCR/runs/mask-preflight-<job>/summary/preflight.json (make pull).
# Cost: 8 CPU + 8 (32 GB) = 16 SU/min -> at most 320 SU (20 min); expected a few minutes (an estimate).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
MODE=policy source scripts/job_env.sh
TEXTS=()
for f in "$REMOTE_SCR/runs/junk-13640325/summary/completions.jsonl" \
         "$REMOTE_SCR/runs/grpo-smoke-13657200/summary/a/rewards.jsonl" \
         "$REMOTE_SCR/runs/grpo-smoke-13657200/summary/b/rewards.jsonl"; do
  if [ ! -f "$f" ]; then
    echo "MASK_PREFLIGHT_FAILED $f is not on scratch: the preflight walks all of D-020's 980 completions"
    exit 1
  fi
  TEXTS+=(--texts "$f")
done
OUT="$REMOTE_SCR/runs/mask-preflight-${SLURM_JOB_ID:-local}/summary"
mkdir -p "$OUT"
export TOKENIZERS_PARALLELISM=false
timeout -k 30s 18m "$PY" -m lofi_rl.rl.mask_preflight --config configs/grpo_smoke_v2_grammar_masked.yaml \
  --out "$OUT" "${TEXTS[@]}"
