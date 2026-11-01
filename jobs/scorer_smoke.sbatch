#!/bin/bash
#SBATCH --job-name=lofi-scorer-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=00:20:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# Scorer-server smoke (docs/SCORER.md §11; docs/design/taste_pipeline.md §5): does the scorer score a GRPO batch on
# one GPU, repeatably, within its memory cap, and how long does each stage take?
#   0. setup: 32 items (4 briefs x 8, from tests/fixtures/loops: 24 distinct clips, 20 renders), smoke-only
#      calibrations and the fake RM (lofi_rl.scoring.fakes; not a taste model), so every term runs live
#   1. server a on $TMPDIR (cold cache a): wait for /health, score the batch twice (the second is all memory hits)
#   2. server b, a fresh process with a cold cache b: the same batch; every audio term is compared with a's
#   3. server c, a fresh process on a's disk cache: all disk hits, identical to a
#   4. verdict: runs/scorer-smoke-<job>/summary/scorer_smoke.json under $REMOTE_SCR with per-stage timings, peak GPU
#      memory, cache hits, the max abs difference of every audio term between the passes, and ok; the log ends
#      with SCORER_SMOKE_OK or SCORER_SMOKE_FAILED
# Each server is stopped with SIGTERM and must exit 0 and remove its socket. Every step runs under timeout
# (docs/kit/LESSONS.md 4), sized from the job's remaining time so the verdict always runs. Each pass gets its
# server's pid and stops waiting once that process exits (a server that dies during startup would otherwise look
# like a slow start for the whole --wait-s); if pass a fails, servers b and c are skipped and the verdict runs at
# once, so a server that cannot start ends the job instead of holding the GPU through three waits (about
# 390 + 330 + 210 s by left()'s sizing). Features stay on scratch (cache_a, cache_b); only summary/ is pulled.
# Cost: 8 CPU + 8 (32 GB) + 8 (1 L40S) = 24 SU/min -> at most 480 SU (20 min); expected about 6-8 min,
# 150-190 SU (UNVERIFIED: three model loads and 40 renders on the cluster have not been timed).
#   make submit JOB=jobs/scorer_smoke.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
MODE=scorer source scripts/job_env.sh
RUN="$REMOTE_SCR/runs/scorer-smoke-${SLURM_JOB_ID:-local}"
OUT="$RUN/summary"
SOCKS="${TMPDIR:-/tmp}/lofi-scorer-${SLURM_JOB_ID:-local}"
[ "${#SOCKS}" -le 90 ] || SOCKS="/tmp/lofi-scorer-${SLURM_JOB_ID:-local}"   # a socket path has at most 107 bytes
mkdir -p "$OUT" "$RUN/cache_a" "$RUN/cache_b" "$SOCKS"
chmod 700 "$SOCKS"
export TOKENIZERS_PARALLELISM=false
timeout 30s nvidia-smi --query-gpu=name,memory.total,driver_version,compute_mode --format=csv,noheader || true
# seconds a step may run: the job's remaining time less $1 s still needed after it, at most $2 s
left() { timeout 60s "$PY" -c 'import sys, time; from kit.chain import end_time; e = end_time(); r, c = int(sys.argv[1]), int(sys.argv[2]); print(max(60, min(c, int(e - time.time()) - r)) if e else c)' "$1" "$2"; }
PID=""
RCS=()
trap 'if [ -n "$PID" ]; then kill -TERM "$PID" 2>/dev/null || true; fi' EXIT
start() {   # NAME CACHE SECONDS: a server in the background on $SOCKS/NAME.sock
  timeout -k 30s "$3s" "$PY" -m lofi_rl.scoring.server --sock "$SOCKS/$1.sock" --workers 7 --cache "$2" \
    --calib "$RUN/calib" --rm "$RUN/rm" --gpu-mem-gb 6 --device cuda > "$OUT/server_$1.log" 2>&1 &
  PID=$!
}
stop() {    # NAME: SIGTERM, up to 60 s to exit, then SIGKILL to the wrapper's process group (timeout leads one, so
            # the server dies with it, and its render workers end themselves once it is gone); rc to the verdict
  local rc=0 n=0
  kill -TERM "$PID" 2>/dev/null || true
  while kill -0 "$PID" 2>/dev/null && [ "$n" -lt 60 ]; do sleep 1; n=$((n + 1)); done
  if kill -0 "$PID" 2>/dev/null; then kill -KILL -- "-$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true; fi
  wait "$PID" || rc=$?
  RCS+=(--rc "$1=$rc")
  PID=""
  tail -n 5 "$OUT/server_$1.log" || true
}
FAIL=0
timeout 2m "$PY" -m lofi_rl.scoring.smoke setup --run "$RUN" || FAIL=1

T=$(left 480 420); start a "$RUN/cache_a" "$((T + 60))"
A_OK=1
timeout -k 10s "${T}s" "$PY" -m lofi_rl.scoring.smoke pass --sock "$SOCKS/a.sock" --run "$RUN" --pid "$PID" \
  --out "$OUT/pass_a.json" --repeat 2 --wait-s "$((T - 30))" || { FAIL=1; A_OK=0; }
timeout 30s nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$OUT/nvidia_smi_a.txt" 2>&1 || true
stop a

if [ "$A_OK" = 1 ]; then
  T=$(left 240 360); start b "$RUN/cache_b" "$((T + 60))"
  timeout -k 10s "${T}s" "$PY" -m lofi_rl.scoring.smoke pass --sock "$SOCKS/b.sock" --run "$RUN" --pid "$PID" \
    --out "$OUT/pass_b.json" --wait-s "$((T - 30))" || FAIL=1
  stop b

  T=$(left 120 240); start c "$RUN/cache_a" "$((T + 60))"
  timeout -k 10s "${T}s" "$PY" -m lofi_rl.scoring.smoke pass --sock "$SOCKS/c.sock" --run "$RUN" --pid "$PID" \
    --out "$OUT/pass_c.json" --wait-s "$((T - 30))" || FAIL=1
  stop c
else
  echo "scorer smoke: pass a failed, so servers b and c are skipped (see $OUT/pass_a.json and server_a.log)"
fi

timeout 60s du -sh "$RUN/cache_a" "$RUN/cache_b" 2>/dev/null || true
T=$(left 10 180)
timeout -k 10s "${T}s" "$PY" -m lofi_rl.scoring.smoke verdict --run "$RUN" --out "$OUT/scorer_smoke.json" \
  --pass "a=$OUT/pass_a.json" --pass "b=$OUT/pass_b.json" --pass "c=$OUT/pass_c.json" \
  --cache "a=$RUN/cache_a" --cache "b=$RUN/cache_b" "${RCS[@]}" \
  --sock "a=$SOCKS/a.sock" --sock "b=$SOCKS/b.sock" --sock "c=$SOCKS/c.sock" \
  --gpu-log "$OUT/nvidia_smi_a.txt" || FAIL=1
rmdir "$SOCKS" 2>/dev/null || true
if [ "$FAIL" != 0 ]; then echo "scorer smoke: a step failed (see $OUT)"; exit 1; fi
