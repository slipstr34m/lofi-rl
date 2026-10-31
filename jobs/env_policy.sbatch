#!/bin/bash
#SBATCH --job-name=lofi-env-policy
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# Builds the POLICY venv ($KIT_ENV_POLICY: vLLM 0.32, TRL 1.17, transformers 5.19, torch as vLLM pins it).
# It is built from requirements/policy.lock.txt, which is compiled on the laptop for x86_64-manylinux_2_28
# (Rocky 8.10, glibc 2.28). Wheels bundle their CUDA runtime; driver 580 is CUDA 13 (docs/kit/PLAYBOOK.md §11).
# The project is not installed: jobs get PYTHONPATH=src, so the policy and scorer envs never shadow it.
# The job ends by importing every policy-side module, so a missing dependency fails here, in a CPU job,
# not in a GPU job after hours in the queue (LESSONS 43). Every step has a timeout (LESSONS 4).
# Cost: 8 CPU + 8 (32 GB) = 16 SU/min -> at most 960 SU.
#   make submit JOB=jobs/env_policy.sbatch     (needs jobs/tools.sbatch first)
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
. scripts/_kit_env.sh
export UV_CACHE_DIR=$REMOTE_SCR/cache/uv UV_LINK_MODE=copy
UV=$REMOTE_KEEP/opt/bin/uv; [ -x "$UV" ] || { echo "no uv at $UV: run jobs/tools.sbatch first"; exit 3; }
mkdir -p "$(dirname "$KIT_ENV_POLICY")" "$UV_CACHE_DIR"
[ -x "$KIT_ENV_POLICY/bin/python" ] || timeout 10m "$UV" venv --python "$KIT_TOOLS/bin/python3.12" "$KIT_ENV_POLICY"
timeout 45m "$UV" pip sync --python "$KIT_ENV_POLICY/bin/python" requirements/policy.lock.txt
MODE=policy . scripts/job_env.sh
"$PY" -c "import torch, vllm, trl, transformers, peft; print('PINS vllm', vllm.__version__, 'trl', trl.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
# Import check: every policy-side module (render/compile/verify/taste/scoring/ui need the scorer env).
"$PY" - lofi_rl <<'PYCHK'
import importlib, pkgutil, sys
SCORER_SIDE = ("lofi_rl.render", "lofi_rl.compile", "lofi_rl.verify", "lofi_rl.taste", "lofi_rl.scoring", "lofi_rl.ui")
bad, n = [], 0
for top in sys.argv[1:]:
    pkg = importlib.import_module(top); n += 1
    for info in pkgutil.walk_packages(getattr(pkg, "__path__", []), top + ".", onerror=lambda name: bad.append(f"{name}: walk failed")):
        if info.name.startswith(SCORER_SIDE):
            continue
        try:
            importlib.import_module(info.name); n += 1
        except Exception as e:  # noqa: BLE001: report every failure, not just the first
            bad.append(f"{info.name}: {type(e).__name__}: {e}")
if bad:
    print("IMPORTS_FAILED", *bad, sep="\n  "); raise SystemExit(1)
print(f"IMPORTS_OK {n} modules of", *sys.argv[1:])
PYCHK
"$UV" pip freeze --python "$PY" | sort > "$REMOTE_SCR/logs/pins-policy-${SLURM_JOB_ID:-local}.txt"
echo "pins: $REMOTE_SCR/logs/pins-policy-${SLURM_JOB_ID:-local}.txt"; du -sh "$KIT_ENV_POLICY"
echo "POLICY_ENV_OK"
