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
#   3. the levels tool (section 7.1): trims and median faders on the 9 reference loops, and every clip's own
#      fader on the bus its round varies, the anchor's sour copy included (CD-52)
#   4. the generator again with those trims and faders: one palette per draw key (each clip of an instrument
#      stage its own), checked against the draw
#   5. the clip renders, 8 at a time with 1 worker each (lofi_rl.shootout.render); every palette source is
#      hashed once per job
#   6. the batches (lofi_rl.shootout.batch build): opaque one-pass FLACs plus manifest.json and
#      manifest.hidden.json per batch, each folder checked by python -m lofi_rl.ui.manifest check
# Submit (ARGS hold no spaces; every ARG also goes into summary/leaders.json):
#   make submit JOB=jobs/shootout.sbatch ARGS='--stage day1'
#   make submit JOB=jobs/shootout.sbatch ARGS='--stage day2 --leaders keys=<id>+<id>+<id>/kit=<id>/bass=<id>/sax=<id>+<id> --prior <job1>'
#   with the bass undecided (SHOOTOUTS.md CD-54: R2r, the bass re-listen, on the two surviving kits, leader first):
#   make submit JOB=jobs/shootout.sbatch ARGS='--stage day2 --leaders keys=<id>+<id>+<id>/kit=<id>+<id>/bass=bass_gu_acoustic/sax=<id>+<id> --prior <job1> --add R2r'
#   Day 3, the Day-2 picks with every variant slot named, both prior jobs and the repeats' stages (the tally's line):
#   make submit JOB=jobs/shootout.sbatch ARGS='--stage day3 --leaders keys=<id>/kit=<id>/bass=<id>/sax=<id>/swing=<id>/humanize=<id>/sax_variant=<id>/timing=<id> --prior <job1>+<job2> --repeats <stage>+... [--rerun-s2 <sax id>] [--cut <name>+...]'
#   Day 3 needs the rain at 44.1 kHz under $LOFI_ASSETS (derived/rain/, pinned in shootouts/rain_excerpts.json).
# Results: $REMOTE_SCR/runs/shootout-<job>/summary/ (draw.json, leaders.json, levels.json, palettes/, renders.json,
# batches.json, and summary/stages.json: each stage's exit code and wall time) and $REMOTE_SCR/inbox/<date>/<batch>/;
# make pull brings back both, never the renders.
# Walltime: the Day-1 dry runs on the laptop (Apple M4, --jobs 8, 2026-10-31) took 86.4-98.7 s for every stage
# over seven runs: five with per-clip faders (CD-52; two of them an independent check's), levels 44.3-46.0 s and
# 67 renders 44.8-49.1 s in the builder's three; two with median faders, 87.9-89.4 s. Peak memory 3.0 GB. The
# cluster renders a fixture 1.3-2.7 x slower (2.86-5.95 s, SHOOTOUTS.md 7.7, against 2.2 s, RENDER.md 7); with 8
# CPUs that may be 4 cores' hyperthreads (x 2) and 2 min of start-up that is 3.9-10.9 min, about 46-131 SU.
# 20 min is 1.8 times the slowest.
# Cost: 8 CPU + 4 (16 GB) = 12 SU/min -> at most 240 SU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
MODE=scorer source scripts/job_env.sh
RUN="$REMOTE_SCR/runs/shootout-${SLURM_JOB_ID:-local}"
mkdir -p "$RUN/summary" "$RUN/renders" "$REMOTE_SCR/inbox"
"$PY" -m lofi_rl.shootout.job --run "$RUN" --runs "$REMOTE_SCR/runs" --inbox "$REMOTE_SCR/inbox" \
  --jobs "${SLURM_CPUS_PER_TASK:-8}" "$@"
echo "SHOOTOUT_OK"
