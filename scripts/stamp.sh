#!/usr/bin/env bash
# stamp.sh: one line naming the laptop's code state, written to .sync-stamp by `make sync`:
#   commit=<sha12> dirty=<0|1> diff=<sha256(git diff HEAD + untracked files' sha256)[:12] or -> at=<UTC time>
# Untracked files that git does not ignore count as dirty: rsync sends them, so a job could import them.
# submit.sh compares the first three fields with .sync-stamp, so a job never runs code the cluster lacks.
set -euo pipefail
cd "$(dirname "$0")/.."
c=$(git rev-parse --short=12 HEAD 2>/dev/null || echo none)
u=$(git ls-files --others --exclude-standard 2>/dev/null)   # untracked, not ignored: make sync ships these too
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] || [ -n "$u" ]; then
  d=1; h=$({ git diff HEAD
              # regular files by content; a symlink by its target; anything else (a nested repo, a vanished file) by name
              git ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do
                if [ -L "$f" ]; then printf 'link %s -> %s\n' "$f" "$(readlink "$f")"
                elif [ -f "$f" ]; then shasum -a 256 "$f"
                else printf 'other %s\n' "$f"; fi
              done; } | shasum -a 256 | cut -c1-12)
else
  d=0; h=-
fi
echo "commit=$c dirty=$d diff=$h at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
