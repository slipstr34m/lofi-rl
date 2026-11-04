#!/bin/bash
#SBATCH --job-name=lofi-junk-rate
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:27:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# The base model's junk rate (CLAUDE.md §6: the Phase 1-2 progress oracle; §9: the Phase 1 gate's data). The logic
# lives in src/lofi_rl/rl/junk.py, so this script and its registry sha change only with the job's shape (re-registered
# 2026-11-04 for the frozen palette's render step, D-022). Steps, each under a timeout:
#   1. generate (policy env, vLLM): N completions of prompt PROMPT (v1: src/lofi_rl/prompt.py; v2:
#      src/lofi_rl/prompt_v2.py), one per brief of brief.sample_briefs(SEED, N), at T 1.0, top_p 1, top_k 0,
#      max_tokens 512 (taste_pipeline §6), from Qwen3-4B-Instruct-2507 @cdbee75f in the HF cache; chunks of 128, and
#      none starts that would end after 12 min
#   2. score (policy env): gates G1-G6 and compile -> summary/junk.json: pass rate and CI, the count per gate,
#      repairs and P_fix, token lengths and truncations, near-copies of the example each row's prompt showed and the
#      copy readouts, the prompt version (read from generate.json) and its sha
#   3. render (scorer env): a seeded sample of 50 gate-passing loops that are not near-copies (D-020) with the
#      frozen palette (junk.py's default palettes/lofi_rain_v1.yaml, D-022; D-019's and D-020's runs used dev_gu)
#      to one-pass FLACs under clips/, for a later ear check (nothing goes to the inbox); their G7 count is merged
#      into junk.json. Every step reads that default: this script passes no --palette. Two renders at a time, each
#      with half of --workers (junk.py's default --jobs 2: the freeze spec 5.5 (iii), D-022), and none starts after
#      the deadline; the clips and their order are those of one-at-a-time rendering
# Args: [N=500] [SEED=2026] [CONSTRAINED=0] [PROMPT=v1]. Results: $REMOTE_SCR/runs/junk-<job>/summary/ (make pull);
# the clips stay on scratch. CONSTRAINED=1 passes --constrained to generate: grammar-constrained decoding under
# lofi_rl.grammar's LoopScript regex (D-016, docs/GRAMMAR.md), its sha256 recorded in generate.json and junk.json.
# PROMPT passes --prompt to generate: v1 (the default) or v2 (each brief shown the example of its energy and the next
# mood), recorded in generate.json and junk.json with its template sha. A CONSTRAINED other than 0 or 1, or a PROMPT
# other than v1 or v2, stops the job before any work.
# Sizing, from smoke 13628361 on an A40 (CHANGELOG 2026-10-31: vLLM load 37.0 s, 42.9 tok/s single-stream, eager):
#   generate: imports and load about 1 min; at most 500 x 512 = 256,000 tokens, which at 16 single streams' worth
#   (686 tok/s; batched decode is not measured yet) is 6.2 min; the shared system prompt is prefilled once (prefix
#   cache). 12 min deadline, 14 min timeout. score: about 10 ms a loop, plus about 27 ms a passing loop for the knob
#   readout (laptop: 500 passing loops in 19.0 s), 1.5 min timeout. render, with the frozen palette (the freeze spec
#   5.5; D-022): job B (lofi-drafts 13706355) rendered it one at a time with 4 workers in 2.90-17.39 s (mean 7.71 s,
#   runs/drafts-13706355/summary/verify.json), so 50 renders take at most 50 x 17.39 = 869.5 s that way; two at a
#   time with 2 workers each ran 1.713 times as fast on the laptop (the 42 drafts and 9 references; 1.765 on 50 of
#   D-020 arm B's loops), so at most 869.5 / 1.713 = 507.6 s, and about 50 x 7.71 / 1.713 = 225 s at job B's mean.
#   535 s deadline (no render starts after it), 595 s timeout: the three timeouts and their kill grace are 1,560 s,
#   a minute under the 27 min walltime (25 min, a 360 s deadline and 420 s timeout until 2026-11-04).
#   Expected: about 6.2 min, 112 SU (about 11 min, 197 SU, at the 507.6 s worst): D-020's arms took 5.2 and 5.5 min
#   (93.0 and 99.0 SU), of which their dev_gu renders took 160 and 184 s (runs/junk-1364032{4,5}/summary/clips.json
#   render_s), and this render step takes about 225 s. v0 (13632676) took 4.8 min, 86.4 SU. CONSTRAINED=1 adds the
#   regex compile and a token mask per decode step inside the same 12 min generate deadline, so the walltime and the
#   cost stay.
#   PROMPT=v2 changes only the prompt text: 1,322-1,408 Qwen3 tokens against v1's 1,389-1,391, in 9 system messages
#   (one per energy x mood cell) where v1 has 1, so the walltime and the cost stay.
# Cost: 4 CPU + 6 (24 GB) + 8 (1 GPU) = 18 SU/min x 27 min -> at most 486 SU.
#   make submit JOB=jobs/junk_rate.sbatch                       (debug A40)
#   make submit JOB=jobs/junk_rate.sbatch ARGS='500 2026 1'     (debug A40, constrained)
#   make submit JOB=jobs/junk_rate.sbatch ARGS='500 2026 1 v2'  (debug A40, constrained, prompt v2)
#   make submit JOB=jobs/junk_rate.sbatch OVR='-p gpu --gres=gpu:a100:1 --constraint=a100-80gb' ARGS='500 2026'
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
N=${1:-500}
SEED=${2:-2026}
CONSTRAINED=${3:-0}
PROMPT=${4:-v1}
case "$CONSTRAINED" in
  0) CONSTRAIN_FLAG="" ;;
  1) CONSTRAIN_FLAG="--constrained" ;;
  *) echo "junk-rate: CONSTRAINED must be 0 or 1, not '$CONSTRAINED'" >&2; exit 2 ;;
esac
case "$PROMPT" in
  v1|v2) ;;
  *) echo "junk-rate: PROMPT must be v1 or v2, not '$PROMPT'" >&2; exit 2 ;;
esac
MODE=policy source scripts/job_env.sh
RUN="$REMOTE_SCR/runs/junk-${SLURM_JOB_ID:-local}"
mkdir -p "$RUN/summary"
echo "junk-rate: N=$N SEED=$SEED CONSTRAINED=$CONSTRAINED PROMPT=$PROMPT Qwen3-4B-Instruct-2507 refs/main=$(model_rev Qwen/Qwen3-4B-Instruct-2507) -> $RUN"
FAIL=0
# $CONSTRAIN_FLAG is unquoted on purpose: empty, it adds no argument
timeout -k 15 840 "$PY" -m lofi_rl.rl.junk generate --run "$RUN" --n "$N" --seed "$SEED" --deadline-s 720 --prompt "$PROMPT" $CONSTRAIN_FLAG || FAIL=1
timeout -k 10 90 "$PY" -m lofi_rl.rl.junk score --run "$RUN" || FAIL=1
MODE=scorer source scripts/job_env.sh
timeout -k 10 595 "$PY" -m lofi_rl.rl.junk render --run "$RUN" --k 50 --workers "${SLURM_CPUS_PER_TASK:-4}" --deadline-s 535 || FAIL=1
if [ "$FAIL" = 0 ]; then echo "JUNK_RATE_OK"; else echo "JUNK_RATE_FAILED (see $RUN/summary/)"; exit 1; fi
