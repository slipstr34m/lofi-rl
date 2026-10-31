#!/bin/bash
#SBATCH --job-name=lofi-render-check
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# Renderer check on one CPU model. Cluster renders are canonical (docs/kit/LESSONS.md 38), so the canonical
# render hashes come from here (docs/RENDER.md, tests/test_determinism.py). Steps:
#   1. the test suite on a compute node: pytest -m "not laptop", the cluster-marked seam sweep included
#   2. this machine's golden hashes (every render case and the one-shots), recorded into the run's summary
#      (LOFI_RECORD_GOLDEN), never into the mirror
#   3. every fixture loop through the render CLI, a fresh process each, with a cold one-shot cache: N, LUFS,
#      true peak, pcm_sha256 and the manifest. Gate fixtures (g*.ls) must exit 3; the others must render.
# main has three CPU models (epyc-7513, epyc-7542, xeon-4116). Run once per model and compare the summaries on
# the laptop: equal hashes mean one golden key serves the whole partition.
#   make submit JOB=jobs/render_check.sbatch OVR='--constraint=epyc-7513'
# Results: $REMOTE_SCR/runs/render-check-<job>/summary/ (make pull). Renders stay on scratch; no audio is pulled.
# Cost: 8 CPU + 4 (16 GB) = 12 SU/min -> at most 360 SU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
MODE=scorer source scripts/job_env.sh
RUN="$REMOTE_SCR/runs/render-check-${SLURM_JOB_ID:-local}"
OUT="$RUN/summary"
mkdir -p "$OUT" "$RUN/renders"
FAIL=0
export LOFI_CACHE="$RUN/cache"
"$PY" -m pytest -q -m "not laptop" -p no:cacheprovider tests/ 2>&1 | tail -n 25 | tee "$OUT/pytest.txt" || true
grep -qE "^[0-9]+ passed" "$OUT/pytest.txt" && ! grep -qE "[0-9]+ (failed|error)" "$OUT/pytest.txt" || FAIL=1
LOFI_RECORD_GOLDEN="$OUT/render_hashes.json" "$PY" -m pytest -q -m "cluster or not cluster" -p no:cacheprovider \
  tests/test_determinism.py 2>&1 | tail -n 5 | tee "$OUT/golden.txt" || true
grep -qE "^[0-9]+ passed" "$OUT/golden.txt" && ! grep -qE "[0-9]+ (failed|error)" "$OUT/golden.txt" || FAIL=1
export LOFI_CACHE="$RUN/cache-cli"
"$PY" - "$RUN" "$OUT" <<'EOF' || FAIL=1
import json, re, shutil, subprocess, sys, time
from pathlib import Path
run, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "manifests").mkdir(exist_ok=True)
rows, bad = [], []
for f in sorted(Path("tests/fixtures/loops").glob("*.ls")):
    dst = run / "renders" / f.stem / "x.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "lofi_rl.render.pipeline", str(f), "--mood", "wistful", "--energy",
                        "mid", "--bpm", "82", "--tonic", "2", "--palette", "palettes/dev_gu.yaml", "--out", str(dst)],
                       capture_output=True, text=True)
    row = {"fixture": f.name, "exit": p.returncode, "seconds": round(time.perf_counter() - t0, 2)}
    m = re.search(r"^N=(\d+) LUFS=(\S+) true_peak=(\S+) dBTP pcm_sha256=([0-9a-f]{64})", p.stdout, re.M)
    if m:
        row.update(n=int(m[1]), lufs=float(m[2]), true_peak_dbtp=float(m[3]), pcm_sha256=m[4])
        shutil.copy(dst.parent / "x.manifest.json", out / "manifests" / f"{f.stem}.json")
    ok = p.returncode == 3 if f.name.startswith("g") else (p.returncode == 0 and m is not None)
    if not ok:
        row["stderr_tail"] = p.stderr[-800:]
        bad.append(f.name)
    rows.append(row)
    print(json.dumps(row))
(out / "fixtures.json").write_text(json.dumps({"rows": rows, "bad": bad}, indent=1) + "\n")
sys.exit(1 if bad else 0)
EOF
if [ "$FAIL" = 0 ]; then echo "RENDER_CHECK_OK"; else echo "RENDER_CHECK_FAILED (see $OUT)"; exit 1; fi
