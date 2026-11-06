"""Probe: in tests/test_junk.py's frozen-palette two-lane run (as the test now orders its drafts), which renders write
the one-shot cache while both lanes run. Usage: python concurrent_fill_probe.py <worktree> <out_dir> (writes only there)."""
import json
import sys
import threading
import time
from pathlib import Path

wt, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(wt / "src"), str(wt / "tests")]
import test_junk as T                       # noqa: E402
from lofi_rl.palette import load_palette     # noqa: E402
from lofi_rl.render import sampler           # noqa: E402
from lofi_rl.rl import junk                  # noqa: E402

frozen = load_palette(T.FROZEN, verify=False)
cases = [((wt / "drafts" / "v1" / f"{d}.ls").read_text(encoding="utf-8"), "stop", "low", "melancholy", 300)
         for d in T.FROZEN_DRAFTS]
run = out / "jobs2"
T._write_run(run, frozen, cases, drift=None, version="v2")
assert junk.main(["score", "--run", str(run), "--palette", str(T.FROZEN)]) == 0
t0, lock, events = time.monotonic(), threading.Lock(), []
orig_clip, orig_atomic = junk.render_clip, sampler._atomic


def clip(row, palette, clips_dir, *, cache, workers):
    with lock:
        events.append(("start", row["i"], time.monotonic() - t0))
    rec = orig_clip(row, palette, clips_dir, cache=cache, workers=workers)
    with lock:
        events.append(("end", row["i"], time.monotonic() - t0))
    return rec


def atomic(path, write):
    with lock:
        events.append(("write", path.name, time.monotonic() - t0))
    return orig_atomic(path, write)


junk.render_clip, sampler._atomic = clip, atomic
cache = out / "cache"
assert not cache.exists()
rc = junk.main(["render", "--run", str(run), "--k", "3", "--palette", str(T.FROZEN), "--no-verify", "--workers", "4",
                "--jobs", "2", "--cache", str(cache)])
spans = {}
for kind, what, t in events:
    if kind in ("start", "end"):
        spans.setdefault(what, {})[kind] = round(t, 3)
writes = [(t, name) for kind, name, t in events if kind == "write"]
both = [t for t, _ in writes if sum(1 for s in spans.values() if s["start"] <= t <= s["end"]) == 2]
result = {"rc": rc, "drafts": list(T.FROZEN_DRAFTS),
          "pick": [c["i"] for c in json.loads((run / "summary" / "clips.json").read_text())["clips"]],
          "spans": spans, "writes": len(writes), "distinct_written": len({n for _, n in writes}),
          "writes_while_two_renders_ran": len(both),
          "files_left": len(list((cache / "oneshots").iterdir()))}
(out / "concurrent_fill_probe.json").write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
