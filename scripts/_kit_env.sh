# _kit_env.sh — sourced by every kit script and job: loads <repo>/kit.env and refuses placeholders.
# Works when sourced from bash on the laptop (scripts/*.sh) and inside SLURM jobs (scripts/job_env.sh).
_KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f "$_KIT_ROOT/kit.env" ]; then
  echo "missing $_KIT_ROOT/kit.env — copy kit.env.example and fill it in (bootstrap.sh does this)" >&2
  return 1 2>/dev/null || exit 1
fi
set -a
. "$_KIT_ROOT/kit.env"
set +a
case "${KIT_PROJECT}${KIT_CLUSTER_USER}${KIT_ACCOUNT}${KIT_CLUSTER_PATH}" in
  *__*) echo "kit.env still contains __PLACEHOLDERS__ (run bootstrap.sh or edit it)" >&2; return 1 2>/dev/null || exit 1;;
esac
