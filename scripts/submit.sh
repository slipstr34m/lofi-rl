#!/usr/bin/env bash
# submit.sh: the laptop's only way to submit a job (used by `make submit`).
#
#   scripts/submit.sh [sbatch options...] jobs/x.sbatch [script args...]
#
# It refuses unless the cluster mirror holds exactly the laptop's committed code: the first three fields
# of the current stamp must equal those in .sync-stamp, and dirty=0 unless ALLOW_DIRTY=1 (stamp.sh counts
# untracked files as dirty). The same fields are then checked on the cluster against the mirror's own
# .sync-stamp, which `make sync` sends after the tree, so a failed rsync leaves the mirror's old stamp and the
# check refuses (D-020 check, F3). Then, on the cluster, scripts/snapshot.py copies the files of the mirror's
# .sync-manifest into a read-only, hash-checked snapshot under $REMOTE_KEEP/snap, refusing (exit 3) when the
# mirror does not match its manifest or the snapshot's stamp is not the laptop's (--expect: a sync from
# elsewhere that landed in between). The budget guard runs from that snapshot, with its log kept outside the
# --delete mirror ($REMOTE_KEEP), so sbatch runs there: it is the job's SLURM_SUBMIT_DIR, and a later make sync
# cannot change the job's code. PAST_TARGET_OK (Shresth's OK to pass the SU target) is passed to the guard when
# set; it also reaches the job's environment, so a chain's later links inherit it.
# Guard exit 4 means NEEDS_APPROVAL: ask Shresth in chat, then `make approve`.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
. "$REPO/scripts/_kit_env.sh"
now=$("$REPO/scripts/stamp.sh" | cut -d' ' -f1-3)
synced=$(cut -d' ' -f1-3 "$REPO/.sync-stamp" 2>/dev/null || echo none)
if [ "$now" != "$synced" ]; then
  echo "submit.sh: laptop is at '$now' but the last sync was '$synced': run make sync first" >&2
  exit 3
fi
case "$now" in
  *dirty=1*) [ "${ALLOW_DIRTY:-0}" = 1 ] || { echo "submit.sh: uncommitted changes: commit first (or ALLOW_DIRTY=1)" >&2; exit 3; };;
esac
[ $# -gt 0 ] || { echo "usage: scripts/submit.sh [job options...] jobs/x.sbatch [args...]" >&2; exit 2; }
keep=$(printf '%q' "$REMOTE_KEEP")
pto=""; [ -z "${PAST_TARGET_OK:-}" ] || pto="PAST_TARGET_OK=$(printf '%q' "$PAST_TARGET_OK") "
exec "$REPO/scripts/remote.sh" "m=\$(cut -d' ' -f1-3 .sync-stamp 2>/dev/null || echo none); [ \"\$m\" = $(printf '%q' "$now") ] || { echo \"submit.sh: the mirror is at '\$m', the laptop at '$now': run make sync\" >&2; exit 3; }; snap=\$(SNAP_ROOT=$keep/snap python3 scripts/snapshot.py make --expect $(printf '%q' "$now")) || exit 3; case \"\$snap\" in $keep/snap/?*) ;; *) echo \"submit.sh: '\$snap' is not a snapshot under $keep/snap\" >&2; exit 3;; esac; echo \"code snapshot: \$snap\"; cd \"\$snap\" && ${pto}GUARD_LOG=$keep/submissions.log scripts/sbatch_guard.sh$(printf ' %q' "$@")"
