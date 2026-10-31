"""The parts of the Phase-0 smoke (src/lofi_rl/smoke.py) that run without a GPU: CLAP windowing, the
Recorder's step files and the hang guards. The smoke itself runs only on a compute node (jobs/smoke_env.sbatch)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

from lofi_rl.compile import stub
from lofi_rl.smoke import CLAP_MAX, STEP_TIMEOUT_S, Recorder, clap_windows

SRC = Path(__file__).resolve().parents[1] / "src"


def _run_main(tmp_path, body: str):
    """Run lofi_rl.smoke.main() in a child interpreter after `body` patches the module (imported as s)."""
    code = f"import json, signal, time, lofi_rl.smoke as s\n{body}\n"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "PYTHONPATH": str(SRC), "OUT": str(tmp_path)})


@pytest.mark.parametrize("bpm", [60, 75, 82, 90, 95])
def test_clap_windows_cover_the_resampled_loop_and_stay_under_10_s(bpm):
    n = stub.loop_samples(bpm)
    x = (0.1 * np.random.default_rng(bpm).standard_normal((2, n))).astype(np.float32)
    w = clap_windows(x)
    x48 = resample_poly(x.mean(axis=0), 48000, 44100).astype(np.float32)
    assert w.shape == (4, x48.shape[0] // 4) and w.shape[1] < CLAP_MAX
    assert w.dtype == np.float32 and w.flags.c_contiguous
    assert np.array_equal(w.reshape(-1), x48[: w.size]) and x48.shape[0] - w.size < 4


def test_recorder_rewrites_the_step_file_around_every_check(tmp_path):
    rec, seen = Recorder("scorer", tmp_path), {}

    def peek():
        seen.update(json.loads((tmp_path / "scorer.json").read_text()))

    rec.run("first", lambda: {"n": 0})
    rec.run("second", peek)
    assert seen["partial"] and not seen["ok"] and seen["checks"]["first"] == {"ok": True, "s": seen["checks"]["first"]["s"], "n": 0}
    assert seen["checks"]["second"]["error"].startswith("did not finish"), "a crash mid-check still names the check"
    assert json.loads((tmp_path / "scorer.json").read_text())["checks"]["second"]["ok"]
    assert not list(tmp_path.glob("*.tmp")), "writes go through a temp file and os.replace"


def test_the_watchdog_ends_a_hung_step_with_exit_1_and_the_step_file_names_the_check(tmp_path):
    t0 = time.time()
    r = _run_main(tmp_path, 's.STEP_TIMEOUT_S["facts"] = 2\n'
                            's.step_facts = lambda rec: rec.run("hang", lambda: time.sleep(30))\n'
                            'raise SystemExit(s.main(["facts", "--out", __import__("os").environ["OUT"]]))')
    assert r.returncode == 1 and time.time() - t0 < 20, (r.returncode, r.stderr[-500:])
    assert "Timeout (0:00:02)!" in r.stderr, "faulthandler dumps the stacks to the job log"
    d = json.loads((tmp_path / "facts.json").read_text())
    assert d["partial"] and d["checks"]["hang"]["error"].startswith("did not finish")


def test_main_arms_the_native_teardown_backstop_alarm(tmp_path):
    r = _run_main(tmp_path, 's.step_facts = lambda rec: None\n'
                            'rc = s.main(["facts", "--out", __import__("os").environ["OUT"]])\n'
                            'print(json.dumps([rc, signal.alarm(0)]))')      # alarm(0) returns the seconds left
    rc, left = json.loads(r.stdout.strip().splitlines()[-1])
    assert rc == 0 and STEP_TIMEOUT_S["facts"] < left <= STEP_TIMEOUT_S["facts"] + 30


def test_recorder_fails_the_step_only_on_critical_checks(tmp_path, capsys):
    rec = Recorder("policy", tmp_path)
    rec.run("soft", lambda: 1 / 0, critical=False)
    assert rec.finish() == 0 and "POLICY_OK (non-critical failed: ['soft'])" in capsys.readouterr().out
    rec.run("hard", lambda: 1 / 0)
    assert rec.finish() == 1
    d = json.loads((tmp_path / "policy.json").read_text())
    assert d["critical_failed"] == ["hard"] and not d["ok"] and not d["partial"]
    assert "ZeroDivisionError" in d["checks"]["hard"]["trace"]
