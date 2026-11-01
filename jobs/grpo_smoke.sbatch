#!/bin/bash
#SBATCH --job-name=lofi-grpo-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --time=00:25:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# GRPO smoke (docs/design/taste_pipeline.md §6, smoke row; docs/GRPO.md §5): does GRPO run end to end on one GPU?
#   0. --dry-run in the policy env: the config, the prompts, and the reward on tests/fixtures/loops/example.ls
#   1. launch a: python -m lofi_rl.rl.train, 10 steps of 4 prompts x 8 completions, saved at steps 5 and 10
#   2. smoke_check --stage first: a/checkpoint-5 and a/checkpoint-10 exist and can resume (FIRST_RUN_OK)
#   3. launch b: the same config with --resume-from a/checkpoint-5; it must train exactly steps 6-10
#   4. smoke_check --stage final: the resumed step count, the reward's steps and groups, no reward errors, both
#      exit codes; writes summary/smoke.json (per-step reward and time) and ends the log with GRPO_SMOKE_OK
# Args: [CONFIG=configs/grpo_smoke.yaml], one of the configs/grpo_smoke*.yaml files in the mirror; anything else, or
# a second word, stops the job before any work. A bare submit runs configs/grpo_smoke.yaml, the smoke 13632679 ran.
#   configs/grpo_smoke_v2_grammar.yaml is D-020's run-1 setting: prompt v2, the grammar (vllm_structured_outputs_regex,
#   with xgrammar pinned on TRL's vLLM engine and read back), the copy penalty. Its smoke.json adds the completions
#   in the regex's language by finish reason (a stop completion out of it fails the smoke), the backend record, the
#   copy readout and D-020's IS readout (TRL's sampling/sampling_logp_difference/mean per step, read only when
#   run.json says the installed TRL's source was checked to compute mean |old_per_token_logps -
#   sampling_per_token_logps| over completion_mask: TRL's half of D-020's definition; whether vLLM's reported
#   log-probs include the grammar mask is unchecked (UNVERIFIED), run.json vllm_logprobs).
# The reward is the interim symbolic one (lofi_rl.rl.reward: gates, verifiers, knobs, penalties; no render, no
# scorer server), so the §6 step estimate drops its 9 s reward. The trainer checks every GRPOConfig name against
# the installed TRL before it loads the model, so a wrong name costs about 2 minutes, not the walltime.
# Each launch may use the job's remaining time minus 3 min (timeout -k 30s), so the final check always runs.
# Results: $REMOTE_SCR/runs/grpo-smoke-<job>/summary/ (make pull). Checkpoints stay on scratch under a/ and b/.
# Cost: 4 CPU + 8 (32 GB) + 8 (1 A100) = 20 SU/min -> at most 500 SU (25 min); expected about 270 SU (13-14 min).
#   The grammar adds a regex compile per launch and a token mask per decode step: D-020's constrained junk runs
#   generated at 2,580-2,615 tok/s against v0's 2,230 (runs/junk-1364032{4,5}/summary/generate.json), and the copy
#   penalty and the language check add about 0.4 ms a completion to a reward of about 12 ms (laptop, 20
#   completions), so the walltime and the cost should stay (an estimate: this smoke measures it).
#   make submit JOB=jobs/grpo_smoke.sbatch
#   make submit JOB=jobs/grpo_smoke.sbatch ARGS='configs/grpo_smoke_v2_grammar.yaml'
# An L40S at the same 25 min is still at most 500 SU; smoke 13632679 took 14.0 min on one (48gb profile):
#   make submit JOB=jobs/grpo_smoke.sbatch OVR='--gres=gpu:l40s:1 --constraint=l40s' ARGS='configs/grpo_smoke_v2_grammar.yaml'
# A 48 GB card with 40 min is at most 800 SU, expected about 420 SU, so it needs Shresth's OK first (the 500 SU
# registration would refuse it):
#   make submit JOB=jobs/grpo_smoke.sbatch OVR='--gres=gpu:l40s:1 --constraint=l40s --time=00:40:00'
#   make submit JOB=jobs/grpo_smoke.sbatch OVR='-p debug --gres=gpu:a40:1 --constraint=a40 --time=00:40:00'
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
CFG=${1:-configs/grpo_smoke.yaml}
[ $# -le 1 ] || { echo "grpo-smoke: takes at most one argument, a config; got $#: $*" >&2; exit 2; }
KNOWN=0
for c in configs/grpo_smoke*.yaml; do [ "$c" = "$CFG" ] && KNOWN=1; done
if [ "$KNOWN" != 1 ]; then
  echo "grpo-smoke: CONFIG must be one of $(echo configs/grpo_smoke*.yaml), not '$CFG'" >&2; exit 2
fi
MODE=policy source scripts/job_env.sh
echo "grpo-smoke: CONFIG=$CFG"
RUN="$REMOTE_SCR/runs/grpo-smoke-${SLURM_JOB_ID:-local}"
OUT="$RUN/summary"
mkdir -p "$OUT" "$REMOTE_SCR/cache/vllm" "$REMOTE_SCR/cache/triton" "$REMOTE_SCR/cache/inductor"
# compile caches on scratch, never the small home; the second launch reuses the first one's
export VLLM_CACHE_ROOT="$REMOTE_SCR/cache/vllm" TRITON_CACHE_DIR="$REMOTE_SCR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$REMOTE_SCR/cache/inductor" TOKENIZERS_PARALLELISM=false MASTER_ADDR=127.0.0.1
left() { "$PY" -c 'import time; from kit.chain import end_time; e = end_time(); print(max(60, int(e - time.time()) - 180) if e else 1320)'; }
port() { "$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_mode --format=csv,noheader || true
RC_A=skipped; RC_B=skipped
if "$PY" -m lofi_rl.rl.train --config "$CFG" --dry-run > "$OUT/dry_run.json"; then
  RC_A=0
  MASTER_PORT=$(port) timeout -k 30s "$(left)s" "$PY" -m lofi_rl.rl.train --config "$CFG" \
    --run-dir "$RUN/a" --summary-dir "$OUT/a" || RC_A=$?
  if "$PY" -m lofi_rl.rl.smoke_check --run "$RUN" --stage first --config "$CFG" --rc "a=$RC_A"; then
    RC_B=0
    MASTER_PORT=$(port) timeout -k 30s "$(left)s" "$PY" -m lofi_rl.rl.train --config "$CFG" \
      --run-dir "$RUN/b" --summary-dir "$OUT/b" --resume-from "$RUN/a/checkpoint-5" || RC_B=$?
  fi
else
  echo "dry run failed: no GPU work was started"
fi
du -sh "$RUN/a" "$RUN/b" 2>/dev/null || true
"$PY" -m lofi_rl.rl.smoke_check --run "$RUN" --stage final --config "$CFG" --rc "a=$RC_A" --rc "b=$RC_B"
