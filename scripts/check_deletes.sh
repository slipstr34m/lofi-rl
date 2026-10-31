#!/usr/bin/env bash
# check_deletes.sh: the safety net for `make sync`. It reads `rsync -n -i --delete` output on stdin and
# refuses (exit 4) when the real sync would delete a protected path, or more than MAX_DELETE files.
# Cluster state lives outside the mirror (kit.env REMOTE_KEEP, REMOTE_SCR); this catches a mistake anyway.
set -euo pipefail
MAX=${MAX_DELETE:-200}
n=0
bad=()
while IFS= read -r line; do
  case "$line" in
    '*deleting '*)
      p=${line#\*deleting}
      p=${p#"${p%%[![:space:]]*}"}
      n=$((n + 1))
      case "$p" in
        data/*|.venv*|opt/*|ckpt/*|logs/*|hf/*|assets/big/*|votes/*|.sync-stamp|jobs/submissions.log) bad+=("$p");;
      esac;;
  esac
done
if [ ${#bad[@]} -gt 0 ]; then
  printf 'check_deletes: refusing, the sync would delete protected path %s\n' "${bad[@]}" >&2
  exit 4
fi
if [ "$n" -gt "$MAX" ]; then
  echo "check_deletes: refusing, $n deletions > $MAX (set MAX_DELETE to override)" >&2
  exit 4
fi
echo "check_deletes: $n deletion(s), none protected"
