"""Time junk's render step (lofi_rl.rl.junk.render_clip) on the laptop with the frozen palette, in two orders, and
check that both give identical audio.

  order "1x4": one render at a time with 4 workers (junk.cmd_render as it stands; jobs/junk_rate.sbatch --workers 4)
  order "2x2": two renders at a time with 2 workers each (option iii; drafts/render.py's ThreadPoolExecutor)

usage: python render_orders.py <worktree> <set> <out dir> [--orders 1x4,2x2,1x4,2x2]
  set "drafts": the 42 drafts and 9 references, with the briefs job B (13706355) rendered them with
                (runs/drafts-13706355/summary/renders.json: 41 one-at-a-time renders + the kept 10)
  set "armb":   junk's own sample: <out>/run (a copy of runs/junk-13640325/summary's generate.json and completions.jsonl)
                re-scored with `python -m lofi_rl.rl.junk score --palette palettes/lofi_rain_v1.yaml`, then
                sample_indices(passing non-copies, 50, brief seed) exactly as cmd_render picks them

Writes <out>/orders.json. Laptop renders are not canonical (LESSONS 38): only the ratio and the timing shape count.
"""
import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("worktree", type=Path)
ap.add_argument("set", choices=("drafts", "armb"))
ap.add_argument("out", type=Path)
ap.add_argument("--orders", default="1x4,2x2,1x4,2x2")
ap.add_argument("--main", type=Path, default=Path("/Users/shresthrana/csci596-project"))
a = ap.parse_args()
WT = a.worktree
sys.path.insert(0, str(WT / "src"))

from lofi_rl import prompt_v2  # noqa: E402
from lofi_rl.palette import load_palette  # noqa: E402
from lofi_rl.rl import junk  # noqa: E402

FROZEN = WT / "palettes" / "lofi_rain_v1.yaml"
a.out.mkdir(parents=True, exist_ok=True)
pal = load_palette(FROZEN, verify=True)
cache = a.out / "cache"


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def drafts_rows() -> tuple[list[dict], dict]:
    src = a.main / "runs" / "drafts-13706355" / "summary" / "renders.json"
    doc = json.loads(src.read_text())
    refs = prompt_v2.reference_texts(pal, strict=True)
    recs = {rid: r for rid, r in doc["others"].items()}
    for rid, row in doc["kept"].items():
        recs[rid] = row["frozen"]
    rows = []
    for k, rid in enumerate(sorted(recs)):
        r = recs[rid]
        f = r["file"]
        text = refs[f.split(":", 1)[1]] if f.startswith("prompt_v2:") else (WT / f).read_text(encoding="utf-8")
        b = r["brief"]
        rows.append({"i": k, "rid": rid, "file": f, "text": text, "finish_reason": "stop",
                     "brief": {"energy": b["energy"], "mood": b["mood"], "bpm": b["bpm"], "tonic": b["tonic"]},
                     "cluster_render_s": r.get("render_s"), "cluster_pcm_sha256": r.get("pcm_sha256")})
    return rows, {"source": str(src), "source_sha256": sha_file(src), "n": len(rows),
                  "kinds": {"draft": sum(1 for r in rows if not r["file"].startswith("prompt_v2:")),
                            "reference": sum(1 for r in rows if r["file"].startswith("prompt_v2:"))}}


def armb_rows() -> tuple[list[dict], dict]:
    src = a.main / "runs" / "junk-13640325" / "summary"
    run = a.out / "run"
    (run / "summary").mkdir(parents=True, exist_ok=True)
    for f in ("generate.json", "completions.jsonl"):
        shutil.copyfile(src / f, run / "summary" / f)
    env = {**os.environ, "PYTHONPATH": str(WT / "src")}
    p = subprocess.run([sys.executable, "-m", "lofi_rl.rl.junk", "score", "--run", str(run), "--palette", str(FROZEN)],
                       capture_output=True, text=True, env=env, cwd=WT, timeout=600)
    if p.returncode != 0:
        raise SystemExit(f"score failed: {p.stderr[-2000:]}")
    scored = {s["i"]: s for s in junk.read_jsonl(run / "summary" / "scored.jsonl")[0]}
    comp = {r["i"]: r for r in junk.read_jsonl(run / "summary" / "completions.jsonl")[0]}
    meta = json.loads((run / "summary" / "generate.json").read_text())
    seed = int((meta.get("briefs") or {}).get("seed", junk.DEFAULT_SEED))
    passing = [i for i, s in scored.items() if s["ok"] and not s.get("near_copy") and i in comp]
    pick = junk.sample_indices(passing, 50, seed)
    rows = []
    for i in pick:
        r = dict(comp[i])
        r["rid"] = f"{i:04d}"
        rows.append(r)
    js = json.loads((run / "summary" / "junk.json").read_text())
    return rows, {"source": str(src), "completions_sha256": sha_file(src / "completions.jsonl"),
                  "rescored_with": str(FROZEN), "score_headline": p.stdout.strip().splitlines()[-1],
                  "pass_count": js["pass"]["count"], "pass_not_copy": js["pass_not_copy"]["count"],
                  "failures": {g: c for g, c in js["failures"].items() if c}, "passing_not_copy": len(passing),
                  "seed": seed, "pick": pick,
                  "keys_kits": sorted({(t.split("keys: ")[1].split()[0], t.split("drums: ")[1].split()[0])
                                       for t in (r["text"] for r in rows)})}


def cpu_now() -> float:
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def run_order(rows: list[dict], order: str, rep: int) -> dict:
    jobs, workers = {"1x4": (1, 4), "2x2": (2, 2)}[order]
    clips_dir = a.out / f"clips_{order}_{rep}"
    shutil.rmtree(clips_dir, ignore_errors=True)
    clips_dir.mkdir(parents=True)
    c0, t0 = cpu_now(), time.perf_counter()

    def one(row):
        t = time.perf_counter()
        rec = junk.render_clip(row, pal, clips_dir, cache=cache, workers=workers)
        rec["wall_s"] = round(time.perf_counter() - t, 3)
        return rec
    if jobs == 1:
        recs = [one(r) for r in rows]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            recs = list(ex.map(one, rows))
    wall, cpu = time.perf_counter() - t0, cpu_now() - c0
    out = {"order": order, "rep": rep, "jobs": jobs, "workers": workers, "wall_s": round(wall, 2),
           "cpu_s": round(cpu, 2), "cpus_busy": round(cpu / wall, 2), "n": len(recs),
           "ok": sum(1 for r in recs if r.get("ok")), "renders": {}}
    for row, rec in zip(rows, recs):
        name = f"{row['i']:04d}"
        flac = clips_dir / f"{name}.flac"
        man = clips_dir / f"{name}.manifest.json"
        out["renders"][row["rid"]] = {
            "ok": rec.get("ok"), "gate": rec.get("gate"), "error": rec.get("error"), "render_s": rec.get("render_s"),
            "wall_s": rec["wall_s"], "audio_sha256": rec.get("audio_sha256"),
            "flac_sha256": sha_file(flac) if flac.is_file() else None,
            "manifest_sha256": sha_file(man) if man.is_file() else None}
    rs = sorted(v["wall_s"] for v in out["renders"].values())
    out["render_wall_s"] = {"min": rs[0], "median": rs[len(rs) // 2], "max": rs[-1], "sum": round(sum(rs), 2)}
    print(f"{order} rep {rep}: {out['ok']}/{out['n']} ok, wall {out['wall_s']} s, cpu {out['cpu_s']} s "
          f"({out['cpus_busy']} busy), render median {out['render_wall_s']['median']} max {out['render_wall_s']['max']}",
          flush=True)
    shutil.rmtree(clips_dir, ignore_errors=True)    # keep the hashes, not the audio
    return out


rows, meta = drafts_rows() if a.set == "drafts" else armb_rows()
# warm the one-shot cache (both kits) so no order pays for building it
kits = {}
for r in rows:
    kit = r["text"].split("drums: ")[1].split()[0]
    kits.setdefault(kit, r)
warm_dir = a.out / "warm"
warm_dir.mkdir(exist_ok=True)
tw = time.perf_counter()
for kit, r in sorted(kits.items()):
    junk.render_clip(r, pal, warm_dir, cache=cache, workers=4)
shutil.rmtree(warm_dir, ignore_errors=True)
warm_s = round(time.perf_counter() - tw, 2)

results = []
reps: dict[str, int] = {}
for order in a.orders.split(","):
    reps[order] = reps.get(order, 0) + 1
    results.append(run_order(rows, order, reps[order]))

# identical audio: every order and rep against the first 1x4 run
base = next(r for r in results if r["order"] == "1x4")
identical = {}
for r in results:
    diffs = [rid for rid, v in r["renders"].items()
             if (v["audio_sha256"], v["flac_sha256"], v["manifest_sha256"]) !=
             (base["renders"][rid]["audio_sha256"], base["renders"][rid]["flac_sha256"],
              base["renders"][rid]["manifest_sha256"])]
    identical[f"{r['order']}#{r['rep']}"] = {"differs": diffs, "same": len(r["renders"]) - len(diffs)}

by = {}
for r in results:
    by.setdefault(r["order"], []).append(r["wall_s"])
mean = {k: round(sum(v) / len(v), 2) for k, v in by.items()}
summary = {"set": a.set, "meta": meta, "palette": {"path": str(FROZEN), "sha256": pal.sha256},
           "machine": {"cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                             text=True).stdout.strip(), "ncpu": os.cpu_count()},
           "warm_s": warm_s, "orders": [{k: v for k, v in r.items() if k != "renders"} for r in results],
           "wall_by_order": by, "mean_wall_by_order": mean,
           "speedup_2x2": round(mean["1x4"] / mean["2x2"], 3) if "2x2" in mean and "1x4" in mean else None,
           "identical_to_first_1x4": identical,
           "per_render": {rid: {f"{r['order']}#{r['rep']}": r["renders"][rid]["wall_s"] for r in results}
                          for rid in base["renders"]},
           "rows": [{k: v for k, v in r.items() if k not in ("text",)} for r in rows]}
(a.out / "orders.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
print(json.dumps({k: summary[k] for k in ("set", "warm_s", "mean_wall_by_order", "speedup_2x2")}, indent=1))
print(json.dumps(identical, indent=1))
