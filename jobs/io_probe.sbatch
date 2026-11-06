#!/bin/bash
#SBATCH --job-name=lofi-io-probe
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# File-system probe. The Phase-1 gate launch lofi-junk-rate 13783877 (a02-15, 2026-11-06) spent about 200 s between
# its 13-s generate chunks, where junk.py only appends 128 JSON lines, fsyncs them and rewrites generate.json on
# scratch; the D-020 runs (b-nodes, Nov 1-3) spent 12-13 s there. This times the same writes where a job can put them:
# scratch ($REMOTE_SCR), the node's local disk ($TMPDIR or /tmp) and the KEEP area on /project2 ($REMOTE_KEEP):
#   - 3 rounds of 128 JSON lines of about 4 KB each, appended, flushed and fsynced, then a small JSON file written
#     to a temporary name, fsynced and renamed (junk.py's generate pattern);
#   - 10 files of 2.6 MB, each written and fsynced (one FLAC clip's size);
#   - one read of the largest file under $LOFI_ASSETS (a SoundFont bank).
# Every probe file is removed afterwards. Results: $REMOTE_SCR/runs/io-probe-<job>/summary/io.json (make pull).
#   make submit JOB=jobs/io_probe.sbatch                       (main)
#   make submit JOB=jobs/io_probe.sbatch OVR='-p gpu --gres=gpu:a40:1 -w a02-15'   (the gate's node, with its GPU)
# Cost: 2 CPU + 1 (4 GB) = 3 SU/min -> at most 30 SU; on a GPU node with one A40, 11 SU/min -> at most 110 SU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
MODE=scorer source scripts/job_env.sh
RUN="$REMOTE_SCR/runs/io-probe-${SLURM_JOB_ID:-local}"
mkdir -p "$RUN/summary"
LOCAL_ROOT="${TMPDIR:-/tmp}"
timeout -k 10 480 "$PY" - "$RUN/summary/io.json" "$RUN/work" "$LOCAL_ROOT/lofi-io-probe-${SLURM_JOB_ID:-local}" \
  "$REMOTE_KEEP/tmp/io-probe-${SLURM_JOB_ID:-local}" <<'EOF'
import json, os, shutil, socket, sys, time
from pathlib import Path

out, targets = Path(sys.argv[1]), {"scratch": Path(sys.argv[2]), "local": Path(sys.argv[3]), "keep": Path(sys.argv[4])}
line = json.dumps({"i": 0, "text": "x" * 4000}) + "\n"
blob = os.urandom(2_600_000)
res = {"node": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"), "tmpdir": os.environ.get("TMPDIR"),
       "targets": {k: str(v) for k, v in targets.items()}, "results": {}}
for name, d in targets.items():
    r = {}
    try:
        d.mkdir(parents=True, exist_ok=True)
        rounds = []
        with open(d / "completions.jsonl", "a") as f:
            for _ in range(3):
                t = time.perf_counter()
                for _ in range(128):
                    f.write(line)
                f.flush()
                os.fsync(f.fileno())
                tmp = d / ".generate.json.tmp"
                with open(tmp, "w") as g:
                    g.write(json.dumps({"rounds": len(rounds)}))
                    g.flush()
                    os.fsync(g.fileno())
                os.replace(tmp, d / "generate.json")
                rounds.append(round(time.perf_counter() - t, 3))
        r["jsonl_round_s"] = rounds
        flac = []
        for k in range(10):
            t = time.perf_counter()
            with open(d / f"clip{k}.bin", "wb") as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            flac.append(round(time.perf_counter() - t, 3))
        r["clip_write_s"] = flac
    except OSError as err:
        r["error"] = f"{type(err).__name__}: {err}"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    res["results"][name] = r
assets = Path(os.environ.get("LOFI_ASSETS", ""))
banks = sorted((p for p in assets.rglob("*.sf2") if p.is_file()), key=lambda p: p.stat().st_size)[-1:] if assets.is_dir() else []
if banks:
    t = time.perf_counter()
    with open(banks[0], "rb") as f:
        n = 0
        while chunk := f.read(1 << 20):
            n += len(chunk)
    res["bank_read"] = {"file": banks[0].name, "bytes": n, "s": round(time.perf_counter() - t, 3)}
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res))
EOF
echo IO_PROBE_OK
