#!/usr/bin/env bash
# remote.sh: run a command in the cluster mirror ($REMOTE_PROJ), in a login shell, over BatchMode ssh.
#
#   scripts/remote.sh '<command...>'
#   scripts/remote.sh 'GUARD_LOG=... scripts/sbatch_guard.sh jobs/x.sbatch arg'   # env vars go INSIDE the quotes
#
# There is no git on the cluster (CLAUDE.md §1). The mirror is pushed by `make sync`, which writes
# .sync-stamp. This script prints the mirror's stamp next to the laptop's, so every remote result records
# which code it ran. The command runs inside a { } group whose closing brace is on its OWN line, so a
# command may end in a heredoc, a '# comment' or a '&' (LESSONS 14). Never put credentials in the command.
set -euo pipefail
. "$(dirname "$0")/_cluster_ssh.sh"

if [ $# -eq 0 ]; then
  echo "usage: scripts/remote.sh '<command...>'" >&2
  exit 2
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LAPTOP_STAMP=$("$REPO/scripts/stamp.sh" 2>/dev/null | cut -d' ' -f1-2 || echo "commit=none")

REMOTE_CMD="cd '$REMOTE_PROJ' && echo \"[remote] mirror=\$(cut -d' ' -f1-2 .sync-stamp 2>/dev/null || echo none) laptop=$LAPTOP_STAMP host=\$(hostname -s) \$(date '+%F %T')\" && {
$*
}"

# Some cluster ~/.bashrc files call `module` before it is defined in non-login shells; drop only that noise.
exec ssh $CLUSTER_SSH_OPTS "$CLUSTER_HOST" "bash -lc $(printf '%q' "$REMOTE_CMD")" \
  2> >(grep -v 'module: command not found' >&2)
