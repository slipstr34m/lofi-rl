#!/bin/bash
#SBATCH --job-name=lofi-fetch
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# Downloads ungated model repos ANONYMOUSLY into $KIT_CLUSTER_HF_HOME (the cluster never holds a token).
# Download only, no installs. It pins the fetched commit (refs/main -> sha), so offline jobs can load by
# repo id, and prints revision, licence and size per repo for the CHANGELOG. A rerun resumes partial
# downloads. It needs 32 GB because the xet downloader was OOM-killed at 8 GB on a big model (LESSONS 25).
# Spec forms:
#   <org/repo>[@rev]                    configs/tokenizer/code plus *.safetensors (else *.bin)
#   <org/repo>[@rev]#configs            only json/txt/tokenizer files
#   <org/repo>[@rev]#files=a.pt,b.json  exactly these files. The CLAP music checkpoint is a .pt, which the
#                                       kit's default mode skipped.
# Cost: 4 CPU + 8 (32 GB) = 12 SU/min -> at most 720 SU.
#   make submit JOB=jobs/fetch.sbatch ARGS='Qwen/Qwen3-4B-Instruct-2507@<sha> ...'   (needs the policy env)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
MODE=policy source scripts/job_env.sh
unset HF_HUB_OFFLINE
mkdir -p "$HF_HOME"
"$PY" - "$@" <<'PYEOF'
import os, sys, time
from pathlib import Path
from huggingface_hub import get_token, model_info, snapshot_download

assert get_token() is None, "a Hugging Face token exists on the cluster: forbidden"
hub = Path(os.environ["HF_HOME"]) / "hub"
failed = []
for spec in sys.argv[1:]:
    spec, _, mode = spec.partition("#")
    repo, _, rev = spec.partition("@")
    info = model_info(repo, revision=rev or None, files_metadata=True)
    if info.gated:
        print(f"SKIP {repo}: gated repos are not allowed"); failed.append(repo); continue
    names = [s.rfilename for s in info.siblings]
    if mode == "configs":
        allow = ["*.json", "*.txt", "*.model", "*.jinja"]
    elif mode.startswith("files="):
        allow = [f for f in mode[len("files="):].split(",") if f]
        missing = [f for f in allow if f not in names]
        if missing:
            print(f"MISSING {repo}@{info.sha}: {missing} (repo has {sorted(names)[:20]})"); failed.append(repo)
            allow = [f for f in allow if f in names]
    else:
        has_st = any(n.endswith(".safetensors") for n in names)
        allow = ["*.json", "*.txt", "*.model", "*.jinja", "*.py"] + (["*.safetensors"] if has_st else ["*.bin"])
    t0 = time.time()
    path = snapshot_download(repo, revision=info.sha, allow_patterns=allow, max_workers=4)
    ref = hub / ("models--" + repo.replace("/", "--")) / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(info.sha)
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    lic = (info.card_data or {}).get("license") if info.card_data else None
    print(f"MODEL {repo} revision {info.sha} gated {info.gated} license {lic} size_gb {size/1e9:.2f} "
          f"time_s {time.time()-t0:.0f} path {path}", flush=True)
if failed:
    raise SystemExit(f"FETCH_INCOMPLETE {failed}")
PYEOF
du -sh "$HF_HOME"
echo "FETCH_DONE"
