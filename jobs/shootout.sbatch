#!/bin/bash
#SBATCH --job-name=lofi-shootout
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# Shootout batches (docs/SHOOTOUTS.md section 7.7): one day's shootout clips rendered on the cluster (cluster
# renders are canonical, LESSONS 38) and built into blind batches for the rating page. python -m
# lofi_rl.shootout.job runs the stages in order and stops at the first that fails, so a gate or compile failure
# ends the job before any batch is written:
#   1. the draw (lofi_rl.shootout.batch draw): option maps, round order, anchors and repeats from a fresh 64-bit
#      seed, recorded only in summary/draw.json (CD-21), with the renders and palettes the day needs
#   2. the palette generator with --levels none (section 7.8), 3. the levels tool (section 7.1),
#   4. the generator again with the measured trims and faders
#   5. the clip renders, 8 at a time with 1 worker each (lofi_rl.shootout.render); every palette source is
#      hashed once per job
#   6. the batches (lofi_rl.shootout.batch build): opaque one-pass FLACs plus manifest.json and
#      manifest.hidden.json per batch, each folder checked by python -m lofi_rl.ui.manifest check
# Submit (ARGS hold no spaces; every ARG also goes into summary/leaders.json):
#   make submit JOB=jobs/shootout.sbatch ARGS='--stage day1'
#   make submit JOB=jobs/shootout.sbatch ARGS='--stage day2 --leaders keys=<id>+<id>+<id>/kit=<id>/bass=<id>/sax=<id>+<id> --prior <job1>'
#   make submit JOB=jobs/shootout.sbatch ARGS='--stage day3 --leaders keys=<id>/kit=<id>/bass=<id>/sax=<id> --prior <job1>+<job2> --repeats <stage>+... [--rerun-s2 <sax id>] [--cut <name>+...]'
# Results: $REMOTE_SCR/runs/shootout-<job>/summary/ (draw.json, leaders.json, levels.json, palettes/, renders.json,
# batches.json) and $REMOTE_SCR/inbox/<date>/<batch>/; make pull brings back both, never the renders.
# Cost: 8 CPU + 4 (16 GB) = 12 SU/min -> at most 360 SU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
MODE=scorer source scripts/job_env.sh
RUN="$REMOTE_SCR/runs/shootout-${SLURM_JOB_ID:-local}"
mkdir -p "$RUN/summary" "$RUN/renders" "$REMOTE_SCR/inbox"
"$PY" -m lofi_rl.shootout.job --run "$RUN" --runs "$REMOTE_SCR/runs" --inbox "$REMOTE_SCR/inbox" \
  --jobs "${SLURM_CPUS_PER_TASK:-8}" "$@"
echo "SHOOTOUT_OK"
