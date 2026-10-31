#!/bin/bash
#SBATCH --job-name=lofi-shootout
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# Shootout batches (docs/SHOOTOUTS.md section 7.7): one day's shootout clips rendered on the cluster (cluster
# renders are canonical, LESSONS 38) and built into blind batches for the rating page. python -m
# lofi_rl.shootout.job runs the stages in order and stops at the first that fails, so a gate or compile failure
# ends the job before any batch is written:
#   1. the draw (lofi_rl.shootout.batch draw): option maps, round order, anchors and repeats from a fresh 64-bit
#      seed, recorded only in summary/draw.json (CD-21), with the renders and palettes the day needs
#   2. the palette generator with --levels none: the measuring palettes (section 7.8)
#   3. the levels tool (section 7.1): trims and faders on the 9 reference loops
#   4. the generator again with those trims and faders: one palette per draw key, checked against the draw
#   5. the clip renders, 8 at a time with 1 worker each (lofi_rl.shootout.render); every palette source is
#      hashed once per job
#   6. the batches (lofi_rl.shootout.batch build): opaque one-pass FLACs plus manifest.json and
#      manifest.hidden.json per batch, each folder checked by python -m lofi_rl.ui.manifest check
# Submit (ARGS hold no spaces; every ARG also goes into summary/leaders.json):
#   make submit JOB=jobs/shootout.sbatch ARGS='--stage day1'
#   make submit JOB=jobs/shootout.sbatch ARGS='--stage day2 --leaders keys=<id>+<id>+<id>/kit=<id>/bass=<id>/sax=<id>+<id> --prior <job1>'
#   make submit JOB=jobs/shootout.sbatch ARGS='--stage day3 --leaders keys=<id>/kit=<id>/bass=<id>/sax=<id> --prior <job1>+<job2> --repeats <stage>+... [--rerun-s2 <sax id>] [--cut <name>+...]'
# Results: $REMOTE_SCR/runs/shootout-<job>/summary/ (draw.json, leaders.json, levels.json, palettes/, renders.json,
# batches.json, and summary/stages.json: each stage's exit code and wall time) and $REMOTE_SCR/inbox/<date>/<batch>/;
# make pull brings back both, never the renders.
# Walltime: the Day-1 dry run on the laptop (Apple M4, --jobs 8, 2026-10-31) took 87.9-89.4 s for every stage
# (levels 43.8-44.8 s, 67 renders 42.8-43.3 s, the rest 1.3 s) at a peak of 3.0 GB. The cluster renders a fixture
# 1.3-2.7 x slower (SHOOTOUTS.md section 7.7); with 8 CPUs that may be 4 cores' hyperthreads (x 2) and 2 min of
# start-up that is 4-10 min, about 50-120 SU. 20 min leaves twice the slowest of those.
# Cost: 8 CPU + 4 (16 GB) = 12 SU/min -> at most 240 SU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
MODE=scorer source scripts/job_env.sh
RUN="$REMOTE_SCR/runs/shootout-${SLURM_JOB_ID:-local}"
mkdir -p "$RUN/summary" "$RUN/renders" "$REMOTE_SCR/inbox"
"$PY" -m lofi_rl.shootout.job --run "$RUN" --runs "$REMOTE_SCR/runs" --inbox "$REMOTE_SCR/inbox" \
  --jobs "${SLURM_CPUS_PER_TASK:-8}" "$@"
echo "SHOOTOUT_OK"
