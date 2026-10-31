#!/bin/bash
#SBATCH --job-name=lofi-seam-diag
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch1/%u/lofi-rl/logs/%x_%j.out
# Seam diagnostic. lofi-render-check 13630298-300 found that on Linux cycles 2 and 3 of a 3-cycle render differ
# by more than 1e-6 for 6 of the 20 cases in tests/test_seam.py's tempo sweep, on all three CPU models, while
# the laptop gives 0.0. For every sweep case (5 fixtures x 61/71/87/95 BPM) and example.ls at 82 BPM this
# measures where the cycles differ:
#   - each FluidSynth stem over MIDI cycles 1-5 with no pre-roll (consecutive cycles, and cycles 2 vs 4);
#   - the bus chains alone, fed an exactly periodic input;
#   - the production chain over 3 and 4 cycles, as test_seam checks it.
# Each difference is reported as its max, where it peaks, and its max in the first and last second of the cycle
# (the seam). Results: $REMOTE_SCR/runs/seam-diag-<job>/summary/seam.json (make pull). No audio is kept.
#   make submit JOB=jobs/seam_diag.sbatch OVR='--constraint=epyc-7513'
# Cost: 4 CPU + 2 (8 GB) = 6 SU/min -> at most 120 SU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
MODE=scorer source scripts/job_env.sh
RUN="$REMOTE_SCR/runs/seam-diag-${SLURM_JOB_ID:-local}"
mkdir -p "$RUN/summary"
export LOFI_CACHE="$RUN/cache"
CASES=example.ls@82
for b in 61 71 87 95; do for f in example.ls leniency_lower_x.ls one_bar_base.ls shaker_edits.ls span1_dotted.ls; do
  CASES="$CASES,$f@$b"; done; done
"$PY" - "$RUN/summary/seam.json" "$CASES" <<'EOF'
import json, sys, time
from pathlib import Path
import numpy as np
from lofi_rl.brief import Brief
from lofi_rl.compile.compiler import compile_loop
from lofi_rl.dsl.parse import parse
from lofi_rl.palette import load_palette
from lofi_rl.render import pipeline
from lofi_rl.render.mix import mix_buses
from lofi_rl.render.pipeline import SR, music_cycles

out = Path(sys.argv[1])
cases = [c.split("@") for c in sys.argv[2].split(",")]          # name@bpm,...
pal = load_palette(Path("palettes/dev_gu.yaml"), verify=True)
FIX = Path("tests/fixtures/loops")

def where(d, n, tol=1e-6):
    """Where two cycles differ: max, sample of the max, count over tol, and max in the first and last second."""
    a = np.abs(d).max(axis=0)
    idx = np.flatnonzero(a > tol)
    return {"max": float(a.max()), "at_s": round(int(a.argmax()) / SR, 4), "n_over": int(idx.size),
            "first_s": round(int(idx[0]) / SR, 4) if idx.size else None,
            "last_s": round(int(idx[-1]) / SR, 4) if idx.size else None,
            "max_first_1s": float(a[:SR].max()), "max_last_1s": float(a[-SR:].max())}

rows = []
for name, bpm in cases:
    t0 = time.perf_counter()
    bpm = int(bpm)
    brief = Brief(energy="mid", mood="melancholy", bpm=bpm, tonic=bpm % 12)     # as tests/test_seam.py
    res = parse((FIX / name).read_text(), vocab=pal.vocab(), mode=brief.mode)
    c = compile_loop(res.ast, brief, pal)
    n = c.n
    row = {"case": f"{name}@{bpm}", "n": n, "stems": {}, "bus_only": {}, "music": {}}
    old = pipeline.PREROLL
    pipeline.PREROLL = 0
    try:
        st = pipeline.fluid_stems(c, pal, 5)                    # MIDI cycles 1-5 exactly as rendered
    finally:
        pipeline.PREROLL = old
    for s, x in st.items():
        cyc = [x[:, i * n:(i + 1) * n] for i in range(5)]
        row["stems"][s] = {"peak": float(np.abs(cyc[3]).max()),
                           **{f"d{k + 1}{k + 2}": where(cyc[k] - cyc[k + 1], n) for k in range(4)},
                           "d24": where(cyc[1] - cyc[3], n)}
    # bus chains alone, on an exactly periodic input (MIDI cycle 4 of each stem tiled 3x, drums as production)
    per = {s: np.tile(x[:, 3 * n:4 * n], (1, 3)) for s, x in st.items()}
    m = music_cycles(c, pal, 1)
    from lofi_rl.render.pipeline import kit_shots, render_drums
    per["drums"] = np.tile(render_drums(c, kit_shots(pal, c.kit)), (1, 3))
    bus = mix_buses(per, pal.mix, SR).music
    bc = [bus[:, i * n:(i + 1) * n] for i in range(3)]
    row["bus_only"] = {"d12": where(bc[0] - bc[1], n), "d23": where(bc[1] - bc[2], n)}
    # the production chain over 3 and 4 cycles (what test_seam checks)
    for count in (3, 4):
        mc = music_cycles(c, pal, count).cycles
        cy = [mc[:, i * n:(i + 1) * n] for i in range(count)]
        row["music"][f"C{count}"] = {f"d{k + 1}{k + 2}": where(cy[k] - cy[k + 1], n) for k in range(1, count - 1)}
        row["music"][f"C{count}"]["peak"] = float(np.abs(cy[-1]).max())
    row["seconds"] = round(time.perf_counter() - t0, 1)
    rows.append(row)
    s = {k: (v["d34"]["max"], v["d45"]["max"]) for k, v in row["stems"].items()}
    print(json.dumps({"case": row["case"], "stems_d34_d45": s, "bus_only_d23": row["bus_only"]["d23"]["max"],
                      "music_C3_d23": row["music"]["C3"]["d23"], "music_C4_d34": row["music"]["C4"]["d34"]["max"],
                      "s": row["seconds"]}), flush=True)
out.write_text(json.dumps(rows, indent=1) + "\n")
EOF
echo "SEAM_DIAG_DONE"
