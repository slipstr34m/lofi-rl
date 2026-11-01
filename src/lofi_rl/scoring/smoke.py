"""The steps of jobs/scorer_smoke.sbatch (docs/SCORER.md §11): does the scorer server score a GRPO-sized batch on a
GPU, repeatably, and how long does each stage take?

    python -m lofi_rl.scoring.smoke setup   --run RUN
    python -m lofi_rl.scoring.smoke pass    --sock SOCK --run RUN --out PASS.json [--repeat 2] [--allow-fake]
    python -m lofi_rl.scoring.smoke verdict --run RUN --out scorer_smoke.json --pass a=A.json --pass b=B.json
                                            [--pass c=C.json] [--cache a=DIR --cache b=DIR] [--rc a=0 ...]
                                            [--sock a=PATH ...] [--gpu-log FILE]

- **setup** writes RUN/batch.json: 32 items, 4 briefs x 8 completions in GRPO order, built from
  tests/fixtures/loops: the five distinct gate-passing loops, example_leniency.ls (example.ls's canonical form),
  example.ls again (the same text) and g1_kick_words.ls (gate G1: nothing to render). So 24 distinct clips, 20
  renders. It also writes RUN/calib/ and RUN/rm/: smoke-only calibrations (tau_PQ 1.0, a theme med/MAD, a CE
  "validation") and the fake RM (lofi_rl.scoring.fakes:load_fake_rm, not a taste model) with zero references, so
  every term and factor runs live on real features. None of these numbers is a calibration of anything.
- **pass** waits for /health, scores the batch `repeat` times on one server, reads /stats, and writes the
  health, every response, the wall times and the stats.
- **verdict** checks the passes and writes scorer_smoke.json: every pass has 32 items with the expected gates
  and no render error; the same server's second pass is all memory hits and identical; a fresh server with a cold
  cache (b) agrees with the first (a) within TOL on every audio term (the max abs difference of each is recorded);
  a fresh server on a's disk cache (c) is all disk hits and identical; the features in a's and b's caches are
  compared array by array; servers exited 0 and removed their sockets. It prints SCORER_SMOKE_OK or
  SCORER_SMOKE_FAILED <reasons> last and exits 0 or 1.
No result says anything about how a clip sounds: the smoke measures agreement, timing and memory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from lofi_rl.brief import Brief
from lofi_rl.palette import ROOT, load_palette

from . import protocol, theme
from .client import ScorerClient

FIX = ROOT / "tests" / "fixtures" / "loops"
BRIEFS = (Brief("mid", "wistful", 82, 2), Brief("low", "melancholy", 70, 9), Brief("high", "hopeful", 95, 7),
          Brief("mid", "melancholy", 61, 0))
GROUP = ("example.ls", "example_leniency.ls", "leniency_lower_x.ls", "one_bar_base.ls", "shaker_edits.ls",
         "span1_dotted.ls", "example.ls", "g1_kick_words.ls")
FAILS_PARSE = {"g1_kick_words.ls": "G1"}
RM_VER = "smoke-fake-rm/1"
MEMBERS = 5
# the largest |a - b| allowed between two cold passes; centroid_hz is CPU-side and expected to be exact
TOL = {"G": 1e-3, "r.ce": 1e-3, "r.theme": 1e-3, "r.taste": 1e-3, "sig.pq": 1e-3, "sig.pc": 1e-3, "sig.ce": 1e-3,
       "sig.cu": 1e-3, "sig.clap_margin": 1e-4, "sig.centroid_hz": 1e-6, "diag.p_acc": 1e-3, "diag.rm_mean": 1e-3,
       "diag.rm_std": 1e-3}
NOTE = "scorer smoke only: not a calibration of anything (lofi_rl.scoring.smoke)"


def _write(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------ setup
def batch() -> list[dict[str, Any]]:
    items = []
    for b, brief in enumerate(BRIEFS):
        for name in GROUP:
            it = protocol.item_dict(len(items), (FIX / name).read_text(encoding="utf-8"), brief, f"smoke:{b}")
            items.append({**it, "fixture": name, "expect_gate": FAILS_PARSE.get(name)})
    return items


def cmd_setup(a: argparse.Namespace) -> int:
    run = Path(a.run)
    items = batch()
    pal = load_palette(a.palette, verify=False)
    cfg = theme.load_config(a.theme)
    _write(run / "batch.json", {"palette_sha256": pal.sha256, "palette_id": pal.id, "items": items})
    _write(run / "calib" / "pq.json", {"tau_pq": 1.0, "note": NOTE})
    _write(run / "calib" / "theme.json", {"med": 0.0, "mad": 0.05, "theme_sha256": cfg.sha256, "note": NOTE})
    _write(run / "calib" / "ce.json", {"agree": 0.6, "lower95": 0.55, "n_pairs": 0, "med": 5.5, "mad": 0.5,
                                       "note": NOTE})
    _write(run / "rm" / "rm.json", {"rm_ver": RM_VER, "members": MEMBERS, "note": NOTE,
                                    "factory": "lofi_rl.scoring.fakes:load_fake_rm"})
    _write(run / "rm" / "refs.json", {"rm_ver": RM_VER, "members": MEMBERS, "note": NOTE,
                                      "refs": {f"smoke:{b}": [[0.0] * 4] * MEMBERS for b in range(len(BRIEFS))}})
    print(f"[scorer-smoke] setup: {len(items)} items, {len(BRIEFS)} briefs, palette {pal.id} {pal.sha256[:12]}",
          flush=True)
    return 0


# ------------------------------------------------------------------------------------------ pass
def cmd_pass(a: argparse.Namespace) -> int:
    doc = _read(Path(a.run) / "batch.json")
    items = [{k: it[k] for k in ("i", "text", "brief", "brief_id")} for it in doc["items"]]
    client = ScorerClient(a.sock, allow_fake=a.allow_fake, timeout=a.timeout_s)
    t0 = time.perf_counter()
    out: dict[str, Any] = {"sock": str(a.sock), "runs": []}
    try:
        out["health"] = client.wait_healthy(a.wait_s)
        out["wait_s"] = round(time.perf_counter() - t0, 2)
        for _ in range(a.repeat):
            t = time.perf_counter()
            resp = client.score(items, palette_sha256=doc["palette_sha256"], step=0)
            out["runs"].append({"wall_s": round(time.perf_counter() - t, 3), "response": resp})
            print(f"[scorer-smoke] pass {a.out.name}: {len(resp['items'])} items in "
                  f"{out['runs'][-1]['wall_s']} s, cache {resp['batch']['cache']}", flush=True)
        out["stats"] = client.stats()
        out["ok"] = True
    except Exception as err:  # noqa: BLE001: recorded, and the verdict fails on it
        out.update(ok=False, error=f"{type(err).__name__}: {err}", status=getattr(err, "status", None),
                   body=getattr(err, "body", None))
        print(f"[scorer-smoke] pass {a.out.name} FAILED: {out['error'][:500]}", flush=True)
    _write(a.out, out)
    return 0 if out["ok"] else 1


# ------------------------------------------------------------------------------------------ verdict
def _get(item: dict, key: str) -> Any:
    node: Any = item
    for k in key.split("."):
        node = node.get(k) if isinstance(node, dict) else None
    return node


def compare(xs: list[dict], ys: list[dict]) -> dict[str, float | None]:
    """The max |x - y| of every audio term over items (None when no item has the term in both)."""
    out: dict[str, float | None] = {}
    for key in TOL:
        diffs = []
        for x, y in zip(xs, ys):
            a, b = _get(x, key), _get(y, key)
            if a is None and b is None:
                continue
            diffs.append(math.inf if a is None or b is None else abs(float(a) - float(b)))
        out[key] = max(diffs) if diffs else None
    return out


def _identical(xs: list[dict], ys: list[dict]) -> bool:
    keys = ("clip_id", "hard", "G", "gate", "r", "sig")
    return all({k: x[k] for k in keys} == {k: y[k] for k in keys} for x, y in zip(xs, ys))


def feature_diffs(dir_a: Path, dir_b: Path) -> dict[str, Any]:
    """Array-by-array max |a - b| of the clips both caches hold (<cache>/feat/<feat_ver>/*.npz)."""
    import numpy as np
    fa = {p.name: p for p in Path(dir_a).glob("feat/*/*.npz")}
    fb = {p.name: p for p in Path(dir_b).glob("feat/*/*.npz")}
    common = sorted(set(fa) & set(fb))
    out: dict[str, Any] = {"clips_a": len(fa), "clips_b": len(fb), "common": len(common)}
    worst: dict[str, float] = {}
    for name in common:
        with np.load(fa[name], allow_pickle=False) as za, np.load(fb[name], allow_pickle=False) as zb:
            ma, mb = json.loads(str(za["meta"])), json.loads(str(zb["meta"]))
            if ma["ok"] != mb["ok"]:
                worst["ok_mismatch"] = worst.get("ok_mismatch", 0) + 1
                continue
            if not ma["ok"]:
                continue
            for k in ("mert", "clap_mix", "clap_music"):
                worst[k] = max(worst.get(k, 0.0), float(np.max(np.abs(za[k] - zb[k]))))
            for k, v in ma["audiobox"].items():
                worst[f"audiobox.{k}"] = max(worst.get(f"audiobox.{k}", 0.0), abs(v - mb["audiobox"][k]))
            same_pcm = ma["render"]["audio_sha256"] == mb["render"]["audio_sha256"]
            worst["pcm_mismatch"] = worst.get("pcm_mismatch", 0) + (0 if same_pcm else 1)
    out["max_abs"] = worst
    return out


def _pairs(values: list[str] | None) -> dict[str, str]:
    out = {}
    for v in values or []:
        k, _, x = v.partition("=")
        out[k] = x
    return out


def _check_pass(name: str, doc: dict, expect: list[dict], fails: list[str]) -> list[list[dict]]:
    if not doc.get("ok"):
        fails.append(f"pass {name}: {doc.get('error', 'no result')[:300]}")
        return []
    runs = [r["response"]["items"] for r in doc["runs"]]
    for k, items in enumerate(runs):
        if len(items) != len(expect):
            fails.append(f"pass {name}.{k}: {len(items)} items for {len(expect)}")
            continue
        for it, ex in zip(items, expect):
            if ex["expect_gate"] is not None and (it["hard"], it["gate"]) != (0, ex["expect_gate"]):
                fails.append(f"pass {name}.{k} item {it['i']} ({ex['fixture']}): gate {it['gate']}, "
                             f"expected {ex['expect_gate']}")
            if it["gate"] in ("render-error", "render-timeout", "parse-error"):
                fails.append(f"pass {name}.{k} item {it['i']} ({ex['fixture']}): {it['gate']} "
                             f"{(it['diag'].get('error') or '')[:200]}")
    return runs


def cmd_verdict(a: argparse.Namespace) -> int:
    run = Path(a.run)
    expect = _read(run / "batch.json")["items"]
    passes = {}
    for k, v in _pairs(a.passes).items():
        try:
            passes[k] = _read(Path(v))
        except (OSError, ValueError) as err:          # killed by its timeout before it wrote, or cut short
            passes[k] = {"ok": False, "error": f"no result in {v}: {type(err).__name__}: {err}"}
    fails: list[str] = []
    runs = {k: _check_pass(k, d, expect, fails) for k, d in passes.items()}
    out: dict[str, Any] = {"job": os.environ.get("SLURM_JOB_ID"), "node": os.uname().nodename,
                           "batch": {"n": len(expect), "briefs": len(BRIEFS),
                                     "distinct_clips": len({(it["fixture"].replace("example_leniency", "example"),
                                                             it["brief_id"]) for it in expect})},
                           "tolerance": TOL, "passes": {}}
    for k, d in passes.items():
        st = d.get("stats") or {}
        out["passes"][k] = {"ok": d.get("ok"), "wait_s": d.get("wait_s"),
                            "wall_s": [r["wall_s"] for r in d.get("runs", [])],
                            "cache": [r["response"]["batch"]["cache"] for r in d.get("runs", [])],
                            "gates": [r["response"]["batch"]["gates"] for r in d.get("runs", [])],
                            "hard": [sum(it["hard"] for it in r["response"]["items"]) for r in d.get("runs", [])],
                            "t_ms": st.get("t_ms"), "gpu": st.get("gpu"), "counts": st.get("counts"),
                            "cache_counters": st.get("cache"),
                            "terms": (d.get("health") or {}).get("terms"),
                            "versions": (d.get("health") or {}).get("versions"),
                            "featurizer": (d.get("health") or {}).get("featurizer")}
    ra, rb, rc = runs.get("a") or [], runs.get("b") or [], runs.get("c") or []
    if ra:
        terms = (passes["a"].get("health") or {}).get("terms") or {}
        off = [k for k, v in terms.items() if not v]
        if off:
            fails.append(f"pass a: terms off {off}: the smoke's calibration and RM should turn every one on")
        renders = Counter(it["diag"]["cache"] for it in ra[0])
        out["cache_hits"] = {"a.0": dict(renders)}
        if len(ra) > 1:
            out["cache_hits"]["a.1"] = dict(Counter(it["diag"]["cache"] for it in ra[1]))
            if any(it["diag"]["cache"] in ("miss",) for it in ra[1]):
                fails.append("pass a.1: a clip was rendered again on the same server (memory cache missed)")
            if not _identical(ra[0], ra[1]):
                fails.append("pass a.1 differs from a.0 on the same server")
            out["diff_a0_a1"] = compare(ra[0], ra[1])
    if ra and rb:
        out["diff_a_b"] = diff = compare(ra[0], rb[0])
        over = {k: v for k, v in diff.items() if v is not None and v > TOL[k]}
        if over:
            fails.append(f"a fresh server (b) differs from a beyond tolerance: {over}")
        if any(x["clip_id"] != y["clip_id"] or x["hard"] != y["hard"] for x, y in zip(ra[0], rb[0])):
            fails.append("b disagrees with a on clip ids or gates")
    if ra and rc:
        out["diff_a_c"] = compare(ra[0], rc[0])
        out["cache_hits"]["c.0"] = dict(Counter(it["diag"]["cache"] for it in rc[0]))
        if any(it["diag"]["cache"] == "miss" for it in rc[0]):
            fails.append("pass c: a clip was rendered although a's disk cache held it")
        if not _identical(ra[0], rc[0]):
            fails.append("pass c (a's disk cache) differs from a")
    caches = _pairs(a.cache)
    if "a" in caches and "b" in caches:
        out["feature_diff_a_b"] = feature_diffs(Path(caches["a"]), Path(caches["b"]))
    for k, rc_ in _pairs(a.rc).items():
        out.setdefault("server_rc", {})[k] = rc_
        if rc_ != "0":
            fails.append(f"server {k} exited {rc_}, not 0, after SIGTERM")
    for k, path in _pairs(a.sock).items():
        gone = not Path(path).exists()
        out.setdefault("socket_removed", {})[k] = gone
        if not gone:
            fails.append(f"server {k} left its socket {path}")
    if a.gpu_log and Path(a.gpu_log).is_file():
        out["nvidia_smi"] = Path(a.gpu_log).read_text(encoding="utf-8", errors="replace").strip()[-2000:]
    peak = [((p.get("gpu") or {}).get("peak_allocated_mb"), (p.get("gpu") or {}).get("cap_gb"))
            for p in out["passes"].values()]
    out["peak_gpu_allocated_mb"] = max((x for x, _ in peak if x is not None), default=None)
    caps = [c for _, c in peak if c is not None]
    if out["peak_gpu_allocated_mb"] is not None and caps and out["peak_gpu_allocated_mb"] > 1024 * min(caps):
        fails.append(f"peak GPU allocation {out['peak_gpu_allocated_mb']} MB is over the {min(caps)} GB cap")
    for k in ("a", "b"):
        if k not in passes:
            fails.append(f"pass {k} is missing")
    out["ok"], out["failures"] = not fails, fails
    _write(Path(a.out), out)
    print(json.dumps({k: out.get(k) for k in ("diff_a_b", "cache_hits", "peak_gpu_allocated_mb")}, default=str))
    print("SCORER_SMOKE_OK" if not fails else "SCORER_SMOKE_FAILED " + " | ".join(fails), flush=True)
    return 0 if not fails else 1


# ------------------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lofi_rl.scoring.smoke", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="write the batch, the smoke calibrations and the fake RM under --run")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--palette", type=Path, default=ROOT / "palettes" / "dev_gu.yaml")
    s.add_argument("--theme", type=Path, default=theme.DEFAULT_CONFIG)
    p = sub.add_parser("pass", help="score the batch on one server")
    p.add_argument("--sock", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--wait-s", type=float, default=600.0)
    p.add_argument("--timeout-s", type=float, default=300.0)
    p.add_argument("--allow-fake", action="store_true", help="accept a --backend fake server (laptop rehearsal)")
    v = sub.add_parser("verdict", help="compare the passes and write scorer_smoke.json")
    v.add_argument("--run", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--pass", dest="passes", action="append", metavar="NAME=FILE", required=True)
    v.add_argument("--cache", action="append", metavar="NAME=DIR")
    v.add_argument("--rc", action="append", metavar="NAME=CODE")
    v.add_argument("--sock", action="append", metavar="NAME=PATH")
    v.add_argument("--gpu-log", type=Path)
    a = ap.parse_args(argv)
    return {"setup": cmd_setup, "pass": cmd_pass, "verdict": cmd_verdict}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
