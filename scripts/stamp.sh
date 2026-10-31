#!/usr/bin/env bash
# stamp.sh: one line naming the laptop's code state, written to .sync-stamp by `make sync`:
#   commit=<sha12> dirty=<0|1> diff=<sha256(git diff HEAD)[:12] or -> at=<UTC time>
# submit.sh compares the first three fields with .sync-stamp, so a job never runs code the cluster lacks.
set -euo pipefail
cd "$(dirname "$0")/.."
c=$(git rev-parse --short=12 HEAD 2>/dev/null || echo none)
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
  d=1; h=$(git diff HEAD | shasum -a 256 | cut -c1-12)
else
  d=0; h=-
fi
echo "commit=$c dirty=$d diff=$h at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
