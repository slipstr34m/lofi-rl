#!/bin/bash
#SBATCH --job-name=lofi-smoke-env
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# Phase-0 oracle (CLAUDE.md §5): it must pass before anything else. Checks, in order:
#   1. node facts and outbound network
#   2. scorer env: GPU visible; FluidSynth renders the same PCM twice; a loop renders to exactly N samples
#      at -14 +/- 0.1 LUFS with true peak <= -1 dBTP; MERT, CLAP and Audiobox run (timings; CLAP is
#      deterministic on bar-aligned windows)
#   3. the test suite on a compute node: pytest -m "not laptop", so no git
#   4. policy env: GPU visible; vLLM loads Qwen3-4B-Instruct-2507 and writes one LoopScript at T=0
# Results go to $REMOTE_SCR/runs/smoke-<job>/summary/*.json (make pull brings them back). The job ends
# with SMOKE_OK only if every critical check passed. The logic lives in src/lofi_rl/smoke.py, so this
# script (and its registry sha) stays stable.
# Cost: 8 CPU + 8 (32 GB) + 8 (1 GPU) = 24 SU/min -> at most 720 SU.
#   make submit JOB=jobs/smoke_env.sbatch      (if debug is drained: OVR='-p gpu --gres=gpu:1 --constraint=a40|a100|l40s')
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
MODE=scorer source scripts/job_env.sh
OUT="$REMOTE_SCR/runs/smoke-${SLURM_JOB_ID:-local}/summary"
mkdir -p "$OUT"
FAIL=0
"$PY" -m lofi_rl.smoke facts --out "$OUT" || FAIL=1
"$PY" -m lofi_rl.smoke scorer --out "$OUT" || FAIL=1
"$PY" -m pytest -q -m "not laptop" -p no:cacheprovider tests/ 2>&1 | tail -n 15 | tee "$OUT/pytest.txt" || true
grep -qE "^[0-9]+ passed" "$OUT/pytest.txt" && ! grep -qE "[0-9]+ (failed|error)" "$OUT/pytest.txt" || FAIL=1
MODE=policy source scripts/job_env.sh
"$PY" -m lofi_rl.smoke policy --out "$OUT" || FAIL=1
if [ "$FAIL" = 0 ]; then echo "SMOKE_OK"; else echo "SMOKE_FAILED (see $OUT/*.json)"; exit 1; fi
