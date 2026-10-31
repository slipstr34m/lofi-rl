#!/bin/bash
#SBATCH --job-name=lofi-env-scorer
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# Builds the SCORER venv ($KIT_ENV_SCORER) from requirements/scorer.lock.txt, compiled on the laptop for
# x86_64-manylinux_2_28. The env holds:
#   - render: pedalboard 0.9.26, pyloudnorm 0.2.0, mido, pretty_midi, soundfile
#   - scorers: MERT via transformers 4.43.4, laion_clap 1.1.7, audiobox_aesthetics 0.0.4, torch 2.8
# It is kept apart from the policy env because MERT's remote code breaks on transformers>=4.44, while vLLM
# 0.32 needs transformers 5.16-5.19 (docs/research/taste_scorers_rl.md). FluidSynth is the binary from
# jobs/tools.sbatch.
# The job ends by importing every scorer-side module (LESSONS 43). Every step has a timeout (LESSONS 4).
# Cost: 8 CPU + 8 (32 GB) = 16 SU/min -> at most 960 SU.
#   make submit JOB=jobs/env_scorer.sbatch     (needs jobs/tools.sbatch first)
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
. scripts/_kit_env.sh
export UV_CACHE_DIR=$REMOTE_SCR/cache/uv UV_LINK_MODE=copy
UV=$REMOTE_KEEP/opt/bin/uv; [ -x "$UV" ] || { echo "no uv at $UV: run jobs/tools.sbatch first"; exit 3; }
mkdir -p "$(dirname "$KIT_ENV_SCORER")" "$UV_CACHE_DIR"
[ -x "$KIT_ENV_SCORER/bin/python" ] || timeout 10m "$UV" venv --python "$KIT_TOOLS/bin/python3.12" "$KIT_ENV_SCORER"
timeout 45m "$UV" pip sync --python "$KIT_ENV_SCORER/bin/python" requirements/scorer.lock.txt
MODE=scorer . scripts/job_env.sh
"$PY" -c "import torch, torchaudio, transformers, numpy, pedalboard, pyloudnorm; print('PINS torch', torch.__version__, 'cuda', torch.version.cuda, 'torchaudio', torchaudio.__version__, 'transformers', transformers.__version__, 'numpy', numpy.__version__, 'pedalboard', pedalboard.__version__)"
"$PY" -c "import laion_clap, audiobox_aesthetics; print('PINS laion_clap + audiobox_aesthetics import OK')"
"$PY" - lofi_rl kit <<'PYCHK'
import importlib, pkgutil, sys
bad, n = [], 0
for top in sys.argv[1:]:
    pkg = importlib.import_module(top); n += 1
    for info in pkgutil.walk_packages(getattr(pkg, "__path__", []), top + ".", onerror=lambda name: bad.append(f"{name}: walk failed")):
        if info.name.startswith("lofi_rl.rl"):
            continue  # the RL side imports trl/vllm: checked by jobs/env_policy.sbatch
        try:
            importlib.import_module(info.name); n += 1
        except Exception as e:  # noqa: BLE001: report every failure, not just the first
            bad.append(f"{info.name}: {type(e).__name__}: {e}")
if bad:
    print("IMPORTS_FAILED", *bad, sep="\n  "); raise SystemExit(1)
print(f"IMPORTS_OK {n} modules of", *sys.argv[1:])
PYCHK
"$UV" pip freeze --python "$PY" | sort > "$REMOTE_SCR/logs/pins-scorer-${SLURM_JOB_ID:-local}.txt"
echo "pins: $REMOTE_SCR/logs/pins-scorer-${SLURM_JOB_ID:-local}.txt"; du -sh "$KIT_ENV_SCORER"
echo "SCORER_ENV_OK"
