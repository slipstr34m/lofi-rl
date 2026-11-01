#!/usr/bin/env bash
# check_frozen.sh: refuse a sync that would change a frozen file (D-020's second amendment, docs/DECISIONS.md).
#
#   scripts/check_frozen.sh [list...]      # default: the lists named in .sync-freeze, one per line
#
# A list holds "<git blob sha>  <path>" lines (# starts a comment), as `git ls-tree` prints them. Every path's
# file on disk (what rsync would ship, committed or not) must hash to its recorded blob. `make sync` runs this
# first whenever .sync-freeze exists (gitignored); delete .sync-freeze when the jobs it protects have ended.
set -euo pipefail
cd "$(dirname "$0")/.."
lists=("$@")
if [ ${#lists[@]} -eq 0 ]; then
  [ -f .sync-freeze ] || exit 0
  while IFS= read -r l; do [ -n "${l%%#*}" ] && lists+=("$l"); done < .sync-freeze
fi
bad=0
for list in "${lists[@]}"; do
  [ -f "$list" ] || { echo "check_frozen: no list $list" >&2; exit 2; }
  while read -r sha path; do
    case "$sha" in ''|'#'*) continue;; esac
    if [ ! -f "$path" ]; then
      echo "check_frozen: $path is gone (frozen by $list)" >&2; bad=1; continue
    fi
    now=$(git hash-object "$path")
    [ "$now" = "$sha" ] || { echo "check_frozen: $path changed (frozen by $list: $sha, now $now)" >&2; bad=1; }
  done < "$list"
done
[ "$bad" = 0 ] || { echo "check_frozen: sync refused; restore the files or end the freeze (.sync-freeze)" >&2; exit 1; }
echo "check_frozen: $(cat "${lists[@]}" | grep -vc '^#') frozen files unchanged"
