#!/bin/bash
#SBATCH --job-name=lofi-tools
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# Builds $KIT_TOOLS outside the rsync mirror. It holds:
#   - python 3.12, the base interpreter of the uv venvs
#   - uv
#   - FluidSynth 2.6.1. Discovery has no module for it, and the version is pinned to match the laptop,
#     because 2.6.0 changed tuning and interpolation (docs/research/sound_sources.md).
# All three come from conda-forge, via the conda module, with a timeout on every step (LESSONS 4).
# Jobs never activate this env. They call $REMOTE_KEEP/opt/bin/fluidsynth, a wrapper that drops
# LD_LIBRARY_PATH (LESSONS 8).
# Cost: 4 CPU + 2 (8 GB) = 6 SU/min -> at most 270 SU.
#   make submit JOB=jobs/tools.sbatch          (runs scripts/sbatch_guard.sh on the cluster)
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
. scripts/_kit_env.sh
export CONDA_PKGS_DIRS=$REMOTE_SCR/cache/conda_pkgs
module purge; module load usc >/dev/null 2>&1 || true; module load "$KIT_CONDA_MODULE"
mkdir -p "$REMOTE_KEEP/opt/bin" "$REMOTE_SCR/logs" "$CONDA_PKGS_DIRS"
echo "job=${SLURM_JOB_ID:-} node=$(hostname) $(cat .sync-stamp 2>/dev/null || echo no-stamp)"
if [ ! -x "$KIT_TOOLS/bin/python3.12" ] || [ ! -x "$KIT_TOOLS/bin/fluidsynth" ] || [ ! -x "$KIT_TOOLS/bin/uv" ]; then
  timeout 30m conda create -y -p "$KIT_TOOLS" -c conda-forge --override-channels "python=3.12" uv "fluidsynth=2.6.1"
fi
cat > "$REMOTE_KEEP/opt/bin/fluidsynth" <<EOF
#!/bin/bash
# FluidSynth 2.6.1 from $KIT_TOOLS, without a conda LD_LIBRARY_PATH leaking into the caller (LESSONS 8).
exec env -u LD_LIBRARY_PATH "$KIT_TOOLS/bin/fluidsynth" "\$@"
EOF
chmod +x "$REMOTE_KEEP/opt/bin/fluidsynth"
ln -sfn "$KIT_TOOLS/bin/uv" "$REMOTE_KEEP/opt/bin/uv"
module purge
FS_VER=$("$REMOTE_KEEP/opt/bin/fluidsynth" --version | sed -n 's/.*runtime version \([0-9.]*\).*/\1/p' | head -1)
echo "PINS fluidsynth $FS_VER | $("$REMOTE_KEEP/opt/bin/uv" --version) | $("$KIT_TOOLS/bin/python3.12" -V)"
[ "$FS_VER" = "2.6.1" ] || { echo "expected FluidSynth 2.6.1, got '$FS_VER'"; exit 1; }
echo "TOOLS_OK"
