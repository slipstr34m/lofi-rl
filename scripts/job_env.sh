# job_env.sh: source at the top of EVERY job, after `set -euo pipefail` and `cd` to its submit directory
# (the job's code snapshot, scripts/snapshot.py; the mirror only for a job submitted outside make submit):
#
#   cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
#   MODE=scorer source scripts/job_env.sh   # MODE=policy | scorer | tools
#
# What the job gets:
#   - $PY: that env's own interpreter (never a bare `python`)
#   - PYTHONPATH=<code>/src, where <code> ($LOFI_CODE) is the tree this script lives in: the job's submit-time
#     snapshot (scripts/snapshot.py, made by submit.sh), so a later `make sync` cannot change a queued or running
#     job's code. The project is not installed into the envs, so two envs never shadow it
#   - an offline Hugging Face cache, unbuffered logs, bounded BLAS/OMP threads
#   - uv and FluidSynth from $REMOTE_KEEP/opt/bin
#   - one header line naming the node, the CPU model, the code tree and its .sync-stamp
# No git: the mirror and its snapshots have none (CLAUDE.md §1). Helpers:
#   model_rev <repo>   the cached revision of a model (log it next to every result)
#   slurm <cmd...>     run SLURM/system tools without a conda LD_LIBRARY_PATH (LESSONS 8)
. "$(dirname "${BASH_SOURCE[0]}")/_kit_env.sh"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export HF_HOME=$KIT_CLUSTER_HF_HOME HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_HUB_DISABLE_TELEMETRY=1
export UV_CACHE_DIR=$REMOTE_SCR/cache/uv UV_LINK_MODE=copy PIP_CACHE_DIR=$REMOTE_SCR/cache/pip
export LOFI_CACHE=${LOFI_CACHE:-$REMOTE_SCR/cache/lofi-rl}   # rendered one-shots: scratch, never the small home
export LOFI_ASSETS=${LOFI_ASSETS:-$REMOTE_KEEP/assets}   # fetch-only sources (make push-assets), never the mirror
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1   # no vLLM usage telemetry from compute nodes
export LOFI_CODE="$_KIT_ROOT" PATH="$REMOTE_KEEP/opt/bin:$PATH" PYTHONPATH="$_KIT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GUARD_LOG="${GUARD_LOG:-$REMOTE_KEEP/submissions.log}"   # a chain link resubmitting itself logs to KEEP too
case "${MODE:-scorer}" in
  policy) _ENV=$KIT_ENV_POLICY;;
  scorer) _ENV=$KIT_ENV_SCORER;;
  tools)  _ENV=$KIT_TOOLS;;
  *) echo "job_env.sh: MODE must be policy, scorer or tools (got '${MODE}')"; exit 2;;
esac
PY="$_ENV/bin/python"; [ -x "$PY" ] || { echo "env python missing at $PY (build the env first)"; exit 3; }
mkdir -p "$REMOTE_SCR/logs" "$REMOTE_SCR/runs" "$REMOTE_SCR/cache"
_STAMP=$(cat "$LOFI_CODE/.sync-stamp" 2>/dev/null || echo "no-stamp")
_KIND=snapshot; [ -f "$LOFI_CODE/.lofi-snapshot" ] || _KIND="NO SNAPSHOT: the live mirror, which a make sync changes"
_CPU=$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)
echo "interpreter: $PY ($("$PY" -c "import sys; print(sys.version.split()[0])")) | mode=${MODE:-scorer} job=${SLURM_JOB_ID:-} node=$(hostname) cpu=${_CPU:-?} | code=$LOFI_CODE ($_KIND) | ${_STAMP}"
model_rev() { local b="$HF_HOME/hub/models--${1//\//--}"; [ -f "$b/refs/main" ] && cat "$b/refs/main" || basename "$(ls -d "$b"/snapshots/* 2>/dev/null | head -1)"; }
slurm() { env -u LD_LIBRARY_PATH "$@"; }
