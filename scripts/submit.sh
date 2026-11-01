#!/usr/bin/env bash
# submit.sh: the laptop's only way to submit a job (used by `make submit`).
#
#   scripts/submit.sh [sbatch options...] jobs/x.sbatch [script args...]
#
# It refuses unless the cluster mirror holds exactly the laptop's committed code: the first three fields
# of the current stamp must equal those in .sync-stamp, and dirty=0 unless ALLOW_DIRTY=1. The same fields are
# then checked on the cluster against the mirror's own .sync-stamp, because `make sync` writes the laptop's
# file before its rsync, so a failed rsync would leave a stamp the mirror never got (D-020 check, F3). Then
# it runs the budget guard on the cluster, with the guard's log kept outside the --delete mirror ($REMOTE_KEEP).
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
exec "$REPO/scripts/remote.sh" "m=\$(cut -d' ' -f1-3 .sync-stamp 2>/dev/null || echo none); [ \"\$m\" = $(printf '%q' "$now") ] || { echo \"submit.sh: the mirror is at '\$m', the laptop at '$now': run make sync\" >&2; exit 3; }; GUARD_LOG=$REMOTE_KEEP/submissions.log scripts/sbatch_guard.sh$(printf ' %q' "$@")"
