"""Do arm B's (runs/junk-13640325) passing non-copy loops parse and compile under the frozen palette?

usage: python armb_compile.py <worktree> <run dir>
"""
import json
import sys
from collections import Counter
from pathlib import Path

WT = Path(sys.argv[1])
RUN = Path(sys.argv[2])
sys.path.insert(0, str(WT / "src"))

from lofi_rl.brief import Brief  # noqa: E402
from lofi_rl.compile.compiler import CompileError, compile_loop  # noqa: E402
from lofi_rl.dsl import parse  # noqa: E402
from lofi_rl.palette import load_palette  # noqa: E402
from lofi_rl.rl.junk import read_jsonl, sample_indices  # noqa: E402

frozen = load_palette(WT / "palettes/lofi_rain_v1.yaml", verify=False)
dev = load_palette(WT / "palettes/dev_gu.yaml", verify=False)
scored = {s["i"]: s for s in read_jsonl(RUN / "summary/scored.jsonl")[0]}
rows = {r["i"]: r for r in read_jsonl(RUN / "summary/completions.jsonl")[0]}
passing = [i for i, s in scored.items() if s["ok"] and not s.get("near_copy") and i in rows]
pick = sample_indices(passing, 50, 2026)
out = {"run": str(RUN), "passing_not_copy": len(passing), "pick_50_seed_2026": pick}


def check(i, pal):
    r = rows[i]
    b = Brief(**r["brief"])
    res = parse(r["text"], vocab=pal.vocab(), finish_reason=r.get("finish_reason"), mode=b.mode)
    if not res.gate.ok:
        return f"{res.gate.code}: {res.gate.detail[:90]}"
    try:
        compile_loop(res.ast, b, pal)
    except CompileError as e:
        return f"compile: {str(e)[:90]}"
    return "ok"


for name, pal in (("dev_gu", dev), ("lofi_rain_v1", frozen)):
    res = {i: check(i, pal) for i in passing}
    cnt = Counter(v if v == "ok" else v.split(":")[0] for v in res.values())
    det = Counter(v for v in res.values() if v != "ok")
    out[name] = {"sha256": pal.sha256, "ok_of_passing": sum(v == "ok" for v in res.values()),
                 "by_result": dict(cnt), "top_details": det.most_common(8),
                 "ok_of_pick50": sum(res[i] == "ok" for i in pick),
                 "ok_indices": [i for i in passing if res[i] == "ok"]}
print(json.dumps(out, indent=1))
