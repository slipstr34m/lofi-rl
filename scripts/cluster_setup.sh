#!/usr/bin/env bash
# cluster_setup.sh: one-time cluster setup, run FROM THE LAPTOP (BatchMode ssh; login node only).
#
#   scripts/cluster_setup.sh facts   read-only: partitions, GPU and CPU node types and constraint names,
#                                    per-user QOS limits, queue priority factors, storage and quotas, SU
#                                    usage, modules (paste the output into the CHANGELOG)
#   scripts/cluster_setup.sh init    create the kit.env directories (idempotent mkdir; logs must exist
#                                    BEFORE any job is submitted, or SLURM cannot open --output) and keep
#                                    the per-user project dir private (votes are backed up under it)
# The kit's netcheck (an srun from the login node) is gone: compute-node network is checked inside
# jobs/smoke_env.sbatch, which goes through the budget guard like every job.
set -euo pipefail
. "$(dirname "$0")/_cluster_ssh.sh"
run() { ssh $CLUSTER_SSH_OPTS "$CLUSTER_HOST" "bash -lc $(printf '%q' "$1")" 2> >(grep -v 'module: command not found' >&2); }

case "${1:-}" in
  facts)
    run '
      echo "### partitions"; sinfo -s
      echo; echo "### gpu partition limits + SU billing weights"; scontrol show partition gpu | grep -E "MaxTime|TRES=|TRESBillingWeights"
      echo; echo "### GPU node types (count, gres, cpus, mem MB, features = --constraint names)"; sinfo -p gpu -h -N -o "%G %c %m %f" | sort | uniq -c
      echo; echo "### CPU node types on main (count, cpus, mem MB, features)"; sinfo -p main -h -N -o "%c %m %f" | sort | uniq -c | sort -rn
      echo; echo "### debug partition nodes"; sinfo -p debug -N -o "%N %G %t %C %E"
      echo; echo "### per-user QOS limits"; sacctmgr -n -P show qos format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU 2>/dev/null | head -12
      echo; echo "### storage"; myquota 2>/dev/null || true
      echo; echo "### allocation SUs (myaccount drifts; scripts/session_report.py is the ledger)"; myaccount 2>/dev/null || true
      echo; echo "### queue priority weights, our pending jobs and estimated starts"
      sprio -w 2>/dev/null | head -4; sprio -u "$USER" 2>/dev/null | head -6; squeue --start -u "$USER" 2>/dev/null | head -6
      echo; echo "### modules"; for m in conda cuda ffmpeg fluidsynth; do echo "-- $m"; module spider $m 2>&1 | grep -E "^\s+$m/" | tail -4; done
      echo; echo "### login-node python and rsync"; python3 -V; rsync --version | head -1'
    ;;
  init)
    run "set -e
      mkdir -p '$REMOTE_PROJ' '$REMOTE_KEEP/envs' '$REMOTE_KEEP/opt/bin' '$REMOTE_KEEP/votes' '$REMOTE_KEEP/ckpt' '$REMOTE_KEEP/assets' '$REMOTE_KEEP/runs' '$REMOTE_KEEP/cache' \
               '$REMOTE_SCR/logs' '$REMOTE_SCR/runs' '$REMOTE_SCR/renders' '$REMOTE_SCR/inbox' '$REMOTE_SCR/cache' '$KIT_CLUSTER_HF_HOME'
      chmod 700 '$KIT_CLUSTER_PROJECT_ROOT' '$REMOTE_KEEP/votes'
      ls -ld '$KIT_CLUSTER_PROJECT_ROOT' '$REMOTE_PROJ' '$REMOTE_KEEP' '$REMOTE_KEEP/votes' '$REMOTE_SCR' '$REMOTE_SCR/logs' '$KIT_CLUSTER_HF_HOME'"
    ;;
  *)
    echo "usage: scripts/cluster_setup.sh facts | init" >&2; exit 2;;
esac
