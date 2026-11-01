"""jobs/scorer_smoke.sbatch (lofi-scorer-smoke) and its steps (src/lofi_rl/scoring/smoke.py; docs/SCORER.md §11):
the job is sized under 500 SU with CLAUDE.md §4's rates and says so; it runs in the scorer env on one L40S; every
step is under a timeout; it starts three servers on sockets under $TMPDIR and writes scorer_smoke.json under
$REMOTE_SCR. The smoke's own steps are rehearsed on the laptop against in-process servers with the fake backends:
the batch, the passes, and a verdict that passes on good passes and fails on each defect it exists to catch.
tests/test_job_scripts.py applies the hygiene rules every job script follows."""
from __future__ import annotations

import contextlib
import json
import re
import subprocess
import threading
from pathlib import Path

import pytest

from helpers_scoring import make_scorer, short_sock_dir
from lofi_rl.scoring import smoke
from lofi_rl.scoring.server import ScorerServer

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs" / "scorer_smoke.sbatch"
SU_PER_MIN = {"cpu": 1, "gb": 0.25, "gpu": 8}      # CLAUDE.md §4: CPU 1/core-min, 1 per 4 GB-min, GPU 8/min


def _sbatch(text):
    return dict(re.findall(r"^#SBATCH --([a-z-]+)=(\S+)$", text, re.M))


def _code(text):
    return [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]


def test_the_job_is_sized_under_500_su_and_states_its_worst_case():
    s = JOB.read_text()
    sb = _sbatch(s)
    assert (sb["job-name"], sb["partition"], sb["gres"], sb["cpus-per-task"], sb["mem"]) == (
        "lofi-scorer-smoke", "gpu", "gpu:l40s:1", "8", "32G")
    h, m, sec = (int(x) for x in sb["time"].split(":"))
    minutes = 60 * h + m + sec / 60
    assert minutes <= 20
    rate = SU_PER_MIN["cpu"] * 8 + SU_PER_MIN["gb"] * 32 + SU_PER_MIN["gpu"]
    assert rate == 24 and rate * minutes == 480 <= 500
    assert "Cost: 8 CPU + 8 (32 GB) + 8 (1 L40S) = 24 SU/min -> at most 480 SU (20 min)" in s
    assert "make submit JOB=jobs/scorer_smoke.sbatch" in s
    assert re.search(r"^MODE=scorer source scripts/job_env\.sh$", s, re.M)


def test_every_step_runs_under_a_timeout():
    for line in _code(JOB.read_text()):
        if '"$PY"' in line or line.lstrip().startswith("nvidia-smi"):
            assert "timeout" in line, f"no timeout: {line.strip()}"


def test_the_steps_three_servers_and_the_outputs():
    code = "\n".join(_code(JOB.read_text()))
    steps = re.findall(r"-m lofi_rl\.scoring\.smoke (\w+)", code)
    assert steps == ["setup", "pass", "pass", "pass", "verdict"]
    assert re.findall(r"(?:^|; )start (\w) \"\$RUN/(cache_\w)\"", code, re.M) == [
        ("a", "cache_a"), ("b", "cache_b"), ("c", "cache_a")]
    assert re.findall(r"^stop (\w)$", code, re.M) == ["a", "b", "c"]
    assert 'SOCKS="${TMPDIR:-/tmp}/lofi-scorer-${SLURM_JOB_ID:-local}"' in code and '--sock "$SOCKS/$1.sock"' in code
    assert 'RUN="$REMOTE_SCR/runs/scorer-smoke-${SLURM_JOB_ID:-local}"' in code and 'OUT="$RUN/summary"' in code
    assert '--out "$OUT/scorer_smoke.json"' in code and "--repeat 2" in code
    assert "--gpu-mem-gb 6 --device cuda" in code and "kill -TERM" in code
    for flag in ('--pass "a=', '--pass "b=', '--pass "c=', '--cache "a=', '--cache "b=', '"${RCS[@]}"', '--sock "a='):
        assert flag in code, flag
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0


def test_the_smoke_batch_is_four_briefs_of_eight_with_24_distinct_clips():
    items = smoke.batch()
    assert len(items) == 32 and [it["i"] for it in items] == list(range(32))
    assert [it["brief_id"] for it in items] == [f"smoke:{k // 8}" for k in range(32)]
    assert len({json.dumps(it["brief"], sort_keys=True) for it in items}) == 4
    assert sum(it["expect_gate"] == "G1" for it in items) == 4


# ------------------------------------------------------------------------------------------ rehearsal
@contextlib.contextmanager
def _server(tmp_path, run, cache):
    scorer = make_scorer(tmp_path, calib_dir=run / "calib", rm_dir=run / "rm", cache_root=cache)
    with short_sock_dir() as d:
        srv = ScorerServer(scorer, d / "s.sock", logf=lambda m: None)
        srv.bind()
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            yield d / "s.sock"
        finally:
            srv.shutdown()
            t.join(5)
            srv.close()


def _rehearse(tmp_path):
    run, out = tmp_path / "run", tmp_path / "run" / "summary"
    assert smoke.main(["setup", "--run", str(run)]) == 0
    socks = {}
    for name, cache, repeat in (("a", "cache_a", "2"), ("b", "cache_b", "1"), ("c", "cache_a", "1")):
        with _server(tmp_path, run, run / cache) as sock:
            socks[name] = sock
            assert smoke.main(["pass", "--sock", str(sock), "--run", str(run), "--out", str(out / f"pass_{name}.json"),
                               "--repeat", repeat, "--wait-s", "10", "--allow-fake"]) == 0
    return run, out, socks


def _verdict(run, out, socks, *extra):
    args = ["verdict", "--run", str(run), "--out", str(out / "scorer_smoke.json")]
    for k in "abc":
        args += ["--pass", f"{k}={out / f'pass_{k}.json'}", "--rc", f"{k}=0", "--sock", f"{k}={socks[k]}"]
    args += ["--cache", f"a={run / 'cache_a'}", "--cache", f"b={run / 'cache_b'}", *extra]
    rc = smoke.main(args)
    return rc, json.loads((out / "scorer_smoke.json").read_text())


def test_the_smoke_steps_pass_on_the_fake_backends(tmp_path, capsys):
    run, out, socks = _rehearse(tmp_path)
    rc, v = _verdict(run, out, socks)
    assert rc == 0 and v["ok"] and v["failures"] == [], v["failures"]
    assert "SCORER_SMOKE_OK" in capsys.readouterr().out.splitlines()[-1]
    assert v["batch"] == {"n": 32, "briefs": 4, "distinct_clips": 24}
    assert v["cache_hits"]["a.0"] == {"miss": 20, "none": 4, "dup": 8}
    assert v["cache_hits"]["a.1"] == {"mem": 20, "none": 4, "dup": 8}
    assert v["cache_hits"]["c.0"] == {"disk": 20, "none": 4, "dup": 8}
    assert all(x == 0.0 for x in v["diff_a_b"].values() if x is not None)
    assert v["feature_diff_a_b"]["common"] == 20 and v["feature_diff_a_b"]["max_abs"]["mert"] == 0.0
    assert v["socket_removed"] == {"a": True, "b": True, "c": True}
    assert v["passes"]["a"]["terms"] == {k: True for k in v["passes"]["a"]["terms"]}, "every term live"
    assert set(v["passes"]["a"]["t_ms"]) >= {"render", "featurize", "score", "request"}


def test_the_verdict_fails_on_one_clip_rendered_to_other_pcm_whatever_tol_says(tmp_path, capsys):
    # A PCM drift small enough to move G, r, sig and P_acc by less than TOL still breaks CLAUDE.md §6.
    import numpy as np
    run, out, socks = _rehearse(tmp_path)
    b = json.loads((out / "pass_b.json").read_text())
    it = next(x for x in b["runs"][0]["response"]["items"] if x["hard"] == 1 and x["diag"]["cache"] == "miss")
    it["diag"]["render"]["audio_sha256"] = "0" * 64
    (out / "pass_b.json").write_text(json.dumps(b))
    [rec] = (run / "cache_b").glob(f"feat/**/{it['clip_id']}.npz")
    with np.load(rec, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["meta"]))
    meta["render"]["audio_sha256"] = "0" * 64
    arrays["meta"] = np.array(json.dumps(meta))
    with rec.open("wb") as f:
        np.savez(f, **arrays)
    rc, v = _verdict(run, out, socks)
    assert rc == 1 and capsys.readouterr().out.splitlines()[-1].startswith("SCORER_SMOKE_FAILED")
    joined = " | ".join(v["failures"])
    assert "not a's bit for bit" in joined and f"item {it['i']} audio_sha256" in joined
    assert "other PCM" in joined and v["feature_diff_a_b"]["max_abs"]["pcm_mismatch"] == 1
    assert v["diff_a_b"] and not any("beyond tolerance" in f for f in v["failures"]), "within TOL on every term"


def test_the_verdict_fails_when_a_or_b_was_not_cold_or_the_caches_miss_clips(tmp_path):
    run, out, socks = _rehearse(tmp_path)
    a = json.loads((out / "pass_a.json").read_text())
    a["runs"][0]["response"]["batch"]["cache"]["mem"] = 1
    (out / "pass_a.json").write_text(json.dumps(a))
    next((run / "cache_b").glob("feat/**/*.npz")).unlink()
    rc, v = _verdict(run, out, socks)
    joined = " | ".join(v["failures"])
    assert rc == 1 and "pass a.0 did not start from a cold cache" in joined
    assert "share 19 clips, fewer than the 20 a.0 rendered" in joined


def test_the_verdict_fails_on_drift_a_missing_pass_a_bad_exit_or_a_left_socket(tmp_path, capsys):
    run, out, socks = _rehearse(tmp_path)
    b = json.loads((out / "pass_b.json").read_text())
    b["runs"][0]["response"]["items"][0]["r"]["theme"] += 0.01
    (out / "pass_b.json").write_text(json.dumps(b))
    rc, v = _verdict(run, out, socks)
    assert rc == 1 and any("beyond tolerance" in f and "r.theme" in f for f in v["failures"])
    assert capsys.readouterr().out.splitlines()[-1].startswith("SCORER_SMOKE_FAILED")
    (out / "pass_c.json").unlink()
    socks["a"].parent.mkdir(parents=True, exist_ok=True)
    socks["a"].write_text("left behind")
    rc, v = _verdict(run, out, socks, "--rc", "b=137")
    assert rc == 1
    joined = " | ".join(v["failures"])
    assert "pass c" in joined and "exited 137" in joined and "left its socket" in joined
    socks["a"].unlink()
