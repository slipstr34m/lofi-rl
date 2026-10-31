#!/bin/bash
#SBATCH --job-name=lofi-junk-rate
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:25:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# The base model's junk rate (CLAUDE.md §6: the Phase 1-2 progress oracle; §9: the Phase 1 gate's data). The logic
# lives in src/lofi_rl/rl/junk.py, so this script (and its registry sha) stays stable. Steps, each under a timeout:
#   1. generate (policy env, vLLM): N completions of prompt v1 (src/lofi_rl/prompt.py), one per brief of
#      brief.sample_briefs(SEED, N), at T 1.0, top_p 1, top_k 0, max_tokens 512 (taste_pipeline §6), from
#      Qwen3-4B-Instruct-2507 @cdbee75f in the HF cache; chunks of 128, and none starts that would end after 12 min
#   2. score (policy env): gates G1-G6 and compile -> summary/junk.json: pass rate and CI, the count per gate,
#      repairs and P_fix, token lengths and truncations, near-copies of the in-prompt example, prompt version and sha
#   3. render (scorer env): a seeded sample of 50 gate-passing loops with dev_gu to one-pass FLACs under clips/, for
#      a later ear check (nothing goes to the inbox); their G7 count is merged into junk.json
# Args: [N=500] [SEED=2026]. Results: $REMOTE_SCR/runs/junk-<job>/summary/ (make pull); the clips stay on scratch.
# Sizing, from smoke 13628361 on an A40 (CHANGELOG 2026-10-31: vLLM load 37.0 s, 42.9 tok/s single-stream, eager):
#   generate: imports and load about 1 min; at most 500 x 512 = 256,000 tokens, which at 16 single streams' worth
#   (686 tok/s; batched decode is not measured yet) is 6.2 min; the shared system prompt is prefilled once (prefix
#   cache). 12 min deadline, 14 min timeout. score: about 10 ms a loop, 1.5 min timeout. render: about 3 s a loop
#   (render-check 13632135: 3.05-3.66 s per CLI render), so 50 loops in about 2.5 min; 6 min deadline, 7 min timeout.
#   Expected: about 9 min, 160 SU.
# Cost: 4 CPU + 6 (24 GB) + 8 (1 GPU) = 18 SU/min x 25 min -> at most 450 SU.
#   make submit JOB=jobs/junk_rate.sbatch                       (debug A40)
#   make submit JOB=jobs/junk_rate.sbatch OVR='-p gpu --gres=gpu:a100:1 --constraint=a100-80gb' ARGS='500 2026'
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
N=${1:-500}
SEED=${2:-2026}
MODE=policy source scripts/job_env.sh
RUN="$REMOTE_SCR/runs/junk-${SLURM_JOB_ID:-local}"
mkdir -p "$RUN/summary"
echo "junk-rate: N=$N SEED=$SEED Qwen3-4B-Instruct-2507 refs/main=$(model_rev Qwen/Qwen3-4B-Instruct-2507) -> $RUN"
FAIL=0
timeout -k 15 840 "$PY" -m lofi_rl.rl.junk generate --run "$RUN" --n "$N" --seed "$SEED" --deadline-s 720 || FAIL=1
timeout -k 10 90 "$PY" -m lofi_rl.rl.junk score --run "$RUN" || FAIL=1
MODE=scorer source scripts/job_env.sh
timeout -k 10 420 "$PY" -m lofi_rl.rl.junk render --run "$RUN" --k 50 --workers "${SLURM_CPUS_PER_TASK:-4}" --deadline-s 360 || FAIL=1
if [ "$FAIL" = 0 ]; then echo "JUNK_RATE_OK"; else echo "JUNK_RATE_FAILED (see $RUN/summary/)"; exit 1; fi
