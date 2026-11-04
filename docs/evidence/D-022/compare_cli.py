"""Compare two junk render runs of the same scored run (CLI --jobs 1 against --jobs 2): records minus render_s,
FLAC bytes and manifest bytes, clip by clip; and the summed render_s.

usage: python compare_cli.py <run one-at-a-time> <run two-at-a-time> <out json>
"""
import hashlib
import json
import sys
from pathlib import Path

one, two, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
a = json.loads((one / "summary" / "clips.json").read_text())
b = json.loads((two / "summary" / "clips.json").read_text())


def strip(c):
    return {k: v for k, v in c.items() if k != "render_s"}


def h(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


diff = []
for x, y in zip(a["clips"], b["clips"]):
    same = strip(x) == strip(y)
    fx, fy = one / x["clip"], two / y["clip"]
    same = same and h(fx) == h(fy) and h(fx.with_suffix(".manifest.json")) == h(fy.with_suffix(".manifest.json"))
    if not same:
        diff.append(x["i"])
res = {"one": {"run": str(one), "jobs": a["jobs"], "workers": a["workers"], "n": len(a["clips"]),
               "render_s_sum": round(sum(c["render_s"] for c in a["clips"]), 2)},
       "two": {"run": str(two), "jobs": b["jobs"], "workers": b["workers"], "n": len(b["clips"]),
               "render_s_sum": round(sum(c["render_s"] for c in b["clips"]), 2)},
       "same_order": [c["i"] for c in a["clips"]] == [c["i"] for c in b["clips"]],
       "clips_differ": diff, "clips_same": len(a["clips"]) - len(diff),
       "palette": [a["palette"], b["palette"]]}
out.write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
