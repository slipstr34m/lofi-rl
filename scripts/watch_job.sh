#!/usr/bin/env bash
# watch_job.sh — poll a SLURM job from the laptop (BatchMode ssh) until it leaves the queue, then print
# its accounting line and the tail of its log ($KIT_CLUSTER_SCRATCH/logs/<jobname>-<jobid>.out, the
# --output convention of every job template). One short ssh per poll; never blocks the cluster.
# A failed poll — ssh (Duo expired, network) or squeue itself (slurmctld timeouts) — is reported as
# such, never as "job left the queue"; only an empty squeue answer or "Invalid job id" means that.
# PENDING lines carry SLURM's estimated start (start=...; N/A until the backfill scheduler plans it).
#
#   scripts/watch_job.sh <jobid> [poll_seconds=60] [max_polls=30] [tail_lines=40]
#   Run it in the background and do other work meanwhile.
#   Exit status: 0 the job left the queue, 1 still queued after max_polls, 2 polling failed 3x in a row.
set -uo pipefail
JOB="${1:?jobid}"; EVERY="${2:-60}"; MAX="${3:-30}"; TAIL="${4:-40}"
. "$(dirname "$0")/_cluster_ssh.sh"
fails=0; left=0
for i in $(seq 1 "$MAX"); do
  out=$(ssh $CLUSTER_SSH_OPTS "$CLUSTER_HOST" "squeue -j $JOB -h -o '%T %M %R start=%S' 2>&1; echo __rc=\$?" 2>/dev/null)
  rc=$?
  if [ $rc -ne 0 ] || ! echo "$out" | grep -q "__rc="; then
    fails=$((fails + 1)); echo "$(date +%T) ssh failed (exit $rc) — Duo session or network? ($fails in a row)"
    [ $fails -ge 3 ] && { echo "giving up: run 'ssh -fN $CLUSTER_HOST' to re-authenticate"; exit 2; }
    sleep "$EVERY"; continue
  fi
  qrc=$(echo "$out" | sed -n 's/^__rc=//p' | tail -1)
  st=$(echo "$out" | grep -v "__rc=" | grep -v 'module: command')
  if [ "$qrc" != "0" ]; then
    if echo "$st" | grep -q "Invalid job id"; then echo "$(date +%T) job $JOB left the queue (squeue no longer knows it)"; left=1; break; fi
    fails=$((fails + 1)); echo "$(date +%T) squeue failed (exit $qrc): $(echo "$st" | head -1) ($fails in a row)"
    [ $fails -ge 3 ] && { echo "giving up after $fails failed polls; the job may still be running (sacct -j $JOB)"; exit 2; }
    sleep "$EVERY"; continue
  fi
  fails=0
  if [ -z "$st" ]; then echo "$(date +%T) job $JOB left the queue"; left=1; break; fi
  echo "$(date +%T) $st"
  [ "$i" -lt "$MAX" ] && sleep "$EVERY"
done
[ "$left" = 1 ] || echo "$(date +%T) job $JOB is still in the queue after $MAX polls (state above); accounting so far:"
# accounting line + log tail, as one remote script (printf %q quoting, as in remote.sh)
REMOTE=$(cat <<EOS
sacct -j $JOB -X -n -P --format=JobID,JobName,State,ElapsedRaw,AllocTRES,NodeList,ExitCode | sed 's/|/  /g'
f=\$(sacct -j $JOB -X -n -P --format=JobName | head -1)
log="$KIT_CLUSTER_SCRATCH/logs/\${f}_$JOB.out"
if [ -f "\$log" ]; then echo "### \$log"; grep -vE '^\s*\$' "\$log" | tail -$TAIL | cut -c1-400; else echo "no log at \$log"; fi
EOS
)
ssh $CLUSTER_SSH_OPTS "$CLUSTER_HOST" "bash -lc $(printf '%q' "$REMOTE")" 2>&1 | grep -v 'module: command' || true
[ "$left" = 1 ]   # exit 0: left the queue; 1: still queued after max_polls; 2: polling failed (above)
