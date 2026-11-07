#!/bin/bash
#SBATCH --job-name=lofi-step0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
#SBATCH --no-requeue
# The step-0 samples (docs/DECISIONS.md D-021 "What follows" item 2; the rules D-023 registers; the build spec
# docs/design/step0_samples.md): the RM held-out tuples, PAR's Ref_b and the seed pool's 660, drawn from the base model
# under the grammar with D-021 item 1's streams and draw seeds. The logic lives in src/lofi_rl/step0/, so this script
# and its registry sha change only with the job's shape. Steps, each under a timeout with its own deadline:
#   1. generate (policy env, vLLM): the pre-checks before the model loads (the streams file, the palette, its regex and
#      prompt v2's template at their registered shas, every cell's messages, D-023's mark in docs/DECISIONS.md equal to
#      lofi_rl.step0.MARK, every file docs/prereg/D-023_step0_path.txt lists at its blob in this snapshot, the probe
#      seed none of the draw seeds), then Qwen3-4B-Instruct-2507 @cdbee75f from the HF cache with engine seed 0, the
#      seed probe, and one request per draw with its registered seed under the frozen palette's grammar (T 1, top_p 1,
#      top_k 0, 512 tokens): the requests of lofi_rl.step0.STREAMS, 3,520 under Q6(a), in chunks of 128, none started
#      that would end after 960 s; then the 320 held-out requests again in reverse, a readout to replay.jsonl only
#   2. score (policy env): junk's gates and readouts on every draw, per stream, into summary/step0.json (never a
#      junk.json), with the candidates and Ref_b's selection by its records
#   3. render (scorer env): the held-out set only, in rounds of select.pending (round 1: each brief's first 4 distinct
#      candidates) through junk.render_clip, two at a time with 2 workers each on the job's 4 CPUs, none started after
#      900 s; then the sets (summary/sets.json), the clip records (summary/clips.json) and step0.json's render block.
#      Ref_b and the pool are rendered later, by the jobs the spec's section 3 names. Nothing goes to the inbox
# No argument: the streams, the model, the prompt, the palette and the sampling are registered (lofi_rl.step0), and
# any argument stops the job before any work (exit 2). It never requeues (--no-requeue): a node failure must not rerun
# the job under its own id into its own OUT. Results: $REMOTE_KEEP/runs/step0-<job>/ (summary/ by make pull; clips/,
# the held-out FLACs and manifests, stay on KEEP); `python -m lofi_rl.step0 check runs/step0-<job>` reads them.
# Sizing (the spec's sections 2.8 and 2.9; estimates at the gate run's rates on an A40, which ran 13788483's 500
# requests in 50.1 s of chunks and its 50 renders in 104 s of step wall):
#   generate: about 72.9 s to the first chunk, at most about 12 s of probe, 3,520 x 0.1002 = 352.7 s of draws and
#   320 x 0.1002 = 32.1 s of replay; deadline 960 s (no chunk starts that would end after it), stacks dumped at
#   960 + 90 = 1,050 s, timeout 1,080 s. score: 3,520 x 0.056 = 197.1 s, timeout 420 s. render: 160 x 2.08 = 332.8 s at
#   the gate run's rate (about 720 s at job B's mean one-at-a-time render over 1.713, its 2 x 2 speedup); deadline
#   900 s, 60 s under its 960 s timeout, as junk's render step. Timeouts and kill graces: generate 1,080 + 15 s,
#   score 420 + 10 s, render 960 + 10 s and the two copies out (30 + 2) + (60 + 2) s come to 2,589 s, 111 s under
#   the 45 min walltime, at least the minute spare junk keeps.
#   Expected: about 1,005 s, 16.7 min, about 301 SU (an estimate); at most 810 SU.
# Files (D-022): the steps write on the node's own disk ($TMPDIR), and the run is copied to
#   $REMOTE_KEEP/runs/step0-<job>/ once, after the render or from an EXIT trap if a step stops the script early or the
#   job is killed (TERM: the time limit, scancel), records first and then the clips; the one-shot cache is on KEEP too.
#   An OUT that exists and is not empty stops the job before its traps are set, so two attempts never merge under one
#   job id. Scratch writes stalled the first gate launch (13783877; lofi-io-probe 13787408, D-022); only the Slurm log
#   and job_env.sh's own folders stay on scratch, and the weights are read from the HF cache there.
# Cost: 4 CPU + 6 (24 GB) + 8 (1 GPU) = 18 SU/min x 45 min -> at most 810 SU.
#   make submit JOB=jobs/step0.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
if [ "$#" -ne 0 ]; then
  echo "step0: the job takes no argument (got $#: $*): its streams and settings are registered (lofi_rl.step0)" >&2
  exit 2
fi
MODE=policy source scripts/job_env.sh
RUN="${TMPDIR:-/tmp}/step0-${SLURM_JOB_ID:-local}"      # the steps write here, on the node's own disk
OUT="$REMOTE_KEEP/runs/step0-${SLURM_JOB_ID:-local}"     # the run is copied here after the render (make pull)
export LOFI_CACHE="$REMOTE_KEEP/cache/lofi-rl"          # the one-shot cache, off scratch (job_env keeps it)
if [ -e "$OUT" ] && { [ ! -d "$OUT" ] || [ -n "$(ls -A "$OUT")" ]; }; then
  echo "step0: $OUT exists and is not empty: a used OUT is never written again, so two attempts never merge" >&2
  exit 2
fi
mkdir -p "$RUN/summary" "$OUT"
COPIED=0
copy_out() {   # the run to KEEP, once: its records first, then the held-out clips
  [ "$COPIED" = 1 ] && return 0
  timeout -k 2 30 cp -a "$RUN/summary" "$OUT/" || return 1
  if [ -d "$RUN/clips" ]; then timeout -k 2 60 cp -a "$RUN/clips" "$OUT/" || return 1; fi
  COPIED=1
}
trap 'copy_out || echo "step0: copying $RUN to $OUT failed" >&2' EXIT
trap 'exit 143' TERM   # Slurm's TERM (time limit, scancel) exits through the EXIT trap, which copies the run out
echo "step0: Qwen3-4B-Instruct-2507 refs/main=$(model_rev Qwen/Qwen3-4B-Instruct-2507) -> $OUT (written at $RUN)"
FAIL=0
timeout -k 15 1080 "$PY" -m lofi_rl.step0 generate --run "$RUN" --deadline-s 960 || FAIL=1
timeout -k 10 420 "$PY" -m lofi_rl.step0 score --run "$RUN" || FAIL=1
MODE=scorer source scripts/job_env.sh
timeout -k 10 960 "$PY" -m lofi_rl.step0 render --run "$RUN" --workers "${SLURM_CPUS_PER_TASK:-4}" --deadline-s 900 || FAIL=1
copy_out || { echo "step0: copying $RUN to $OUT failed" >&2; FAIL=1; }
if [ "$FAIL" = 0 ]; then echo "STEP0_OK"; else echo "STEP0_FAILED (see $OUT/summary/)"; exit 1; fi
