#!/bin/bash
#SBATCH --job-name=lofi-drafts
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# Keep3 drafts (docs/DRAFTS.md section 7): the drafted loops rendered on the cluster with the draft palette
# (palettes/draft_day2.yaml, one render palette per keys and kit; cluster renders are canonical, LESSONS 38) and built
# into one keep3 batch for the rating page. python -m lofi_rl.drafts.job runs the stages in order and stops at the
# first that fails, so a draft that fails a check, a render off -14 LUFS or above -1 dBTP, or a gate ends the job
# before any batch is written:
#   1. check: python -m lofi_rl.drafts.check over the drafts (gates, knobs, 5-gram Jaccard, one per brief and persona)
#   2. palettes: the draft palette's checks, a render palette per (keys, kit), each source hashed once, presets, coverage
#   3. render: every draft, 8 at a time with 1 worker each (lofi_rl.drafts.render; FluidSynth a fresh process per stem)
#   4. build: the keep3 batch (one round per brief, persona tiles by a Latin square drawn from a fresh seed, opaque ids,
#      manifest.json and manifest.hidden.json), checked by python -m lofi_rl.ui.manifest check
# Submit (ARGS hold no spaces; every ARG also goes into summary/args.json):
#   make submit JOB=jobs/drafts.sbatch ARGS='--drafts drafts/v1'
#   a redraft of the briefs marked "none acceptable", compared with the drafts they replace:
#   make submit JOB=jobs/drafts.sbatch ARGS='--drafts drafts/v1/redo1 --briefs d04+d07 --against drafts/v1'
#   the sign-off render of the kept loops with the frozen palette (lofi_rl.freeze.signoff; docs/DRAFTS.md section 7):
#   make submit JOB=jobs/drafts.sbatch ARGS='--drafts drafts/v1 --signoff palettes/lofi_rain_v1.yaml'
#   its stages are check, palettes, render (the kept loops with the palette and with its no-bed twin, 8 at a time; the
#   other 20 drafts and prompt v2's 9 references one at a time with 4 workers; each kept render again in a fresh
#   process), verify (summary/verify.json) and build (a triage batch b<MMDD>-signoff[-<k>]); it prints SIGNOFF_OK
#   before DRAFTS_OK. Its laptop run (Apple M4, 2026-11-03) took 152.3 s of stages, 146.5 s of them renders; at
#   1.3-2.7 x on main (below) that is about 3-7 min of renders plus start-up, within the 15 min below.
# Results: $REMOTE_SCR/runs/drafts-<job>/summary/ (args.json, check.json, palettes/, renders.json, draw.json,
# batch.json, stages.json; the sign-off mode adds verify.json) and $REMOTE_SCR/inbox/<date>/b<MMDD>-keep3[-redo<k>]/
# (in the sign-off mode, b<MMDD>-signoff[-<k>]/); make pull brings back both, never the renders.
# Walltime: the laptop dry run (Apple M4, 6 fixture drafts with the real draft palette and banks, --jobs 8,
# 2026-11-01) took 8.9 s for every stage, a render 4.8-7.7 s; the Day-2 shootout job rendered 93 clips in 162 s of
# stages on main (13651610). The cluster renders 1.3-2.7 x slower (SHOOTOUTS.md 7.7), so 30 renders, 8 at a time, take
# about 1-3 min, plus about 2 min of start-up: 15 min is 3 times the slowest.
# Cost: 8 CPU + 4 (16 GB) = 12 SU/min -> at most 180 SU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
MODE=scorer source scripts/job_env.sh
RUN="$REMOTE_SCR/runs/drafts-${SLURM_JOB_ID:-local}"
mkdir -p "$RUN/summary" "$RUN/renders" "$REMOTE_SCR/inbox"
"$PY" -m lofi_rl.drafts.job --run "$RUN" --inbox "$REMOTE_SCR/inbox" --jobs "${SLURM_CPUS_PER_TASK:-8}" "$@"
echo "DRAFTS_OK"
