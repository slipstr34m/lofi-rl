"""kit.chain (the chained-job template and its tests return with jobs/grpo.sbatch in Phase 3): the deadline comes from SLURM and adapts to the last
unit's wall time; only COMPLETE units are resumed; the latest pointer moves atomically; a bounded run
writes DONE so it never chains; a chain that cannot progress stops itself; and the job template checks
DONE/STOP before loading a model and resubmits through the guard without conda's libraries."""
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from kit import chain
from kit.chain import (MARKER, deadline, end_time, is_complete, latest_complete, mark_complete, mark_done,
                       publish_latest, request_stop, run_units, should_stop, stop_file, unit_dir)

ROOT = Path(__file__).resolve().parents[1]
QUIET = {"log": lambda msg: None}


@pytest.fixture(autouse=True)
def no_slurm(monkeypatch):
    """The suite may run inside a SLURM job (jobs/smoke_cpu.sbatch): never read that job's end time."""
    for k in list(os.environ):
        if k.startswith("SLURM_"):
            monkeypatch.delenv(k)


def _stub(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class Clock:
    def __init__(self, t: float):
        self.t = t

    def __call__(self) -> float:
        return self.t


# ---------------------------------------------------------------- the deadline
def test_fixed_margin_regression_a_long_unit_is_not_started():
    """HIGH (first project's review): a fixed 30-minute margin let SLURM kill a long unit mid-way."""
    now = 1_000_000.0
    assert not should_stop(600.0, now + 1800, now)      # 600 s x 1.25 = 750 s fits in 30 min
    assert should_stop(1500.0, now + 1800, now)         # 1500 s x 1.25 = 1875 s does not


@pytest.mark.parametrize("last, left, stop", [
    (0.0, 121.0, False), (0.0, 119.0, True),            # no wall time known yet: at least 120 s
    (96.0, 121.0, False), (96.0, 119.0, True),          # 1.25 x 96 = 120 s: the floor still rules
    (100.0, 126.0, False), (100.0, 124.0, True),        # 1.25 x 100 = 125 s
    (3600.0, 4501.0, False), (3600.0, 4499.0, True),    # an hour-long unit needs 75 minutes
    (-5.0, 121.0, False),                               # a clock glitch counts as 0
])
def test_adaptive_margin(last, left, stop):
    now = 5000.0
    assert should_stop(last, now + left, now) is stop


def test_no_deadline_never_stops():
    assert should_stop(1e9, None, 0.0) is False


def test_deadline_keeps_a_reserve_for_teardown_and_resubmission():
    assert deadline(600, end=10_000.0) == 9_400.0
    assert deadline(600, end=None) is None
    assert deadline() is None                            # outside SLURM


def test_end_time_prefers_slurm_job_end_time(monkeypatch):
    t = int(time.time()) + 3600
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(t))
    assert end_time() == float(t)


@pytest.fixture
def scontrol(tmp_path, monkeypatch):
    """make(end_field) installs a stub scontrol for job 4242; returns the file where it records its
    arguments and whether it saw conda's LD_LIBRARY_PATH."""
    bins = tmp_path / "bin"
    bins.mkdir()
    seen = tmp_path / "scontrol_seen.txt"

    def make(end_field: str) -> Path:
        _stub(bins / "scontrol", f'#!/bin/sh\necho "args=$* LD=${{LD_LIBRARY_PATH-unset}}" > "{seen}"\n'
                                 f'echo "JobId=$4 JobName=p-chain JobState=RUNNING StartTime=2026-10-27T07:35:12 '
                                 f'{end_field} Deadline=N/A"\n')
        monkeypatch.setenv("PATH", f"{bins}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("SLURM_JOB_ID", "4242")
        monkeypatch.setenv("LD_LIBRARY_PATH", "/conda/env/lib")
        return seen

    return make


def test_end_time_falls_back_to_scontrol_without_condas_libraries(scontrol):
    seen = scontrol("EndTime=2026-10-27T11:35:12")
    assert end_time() == time.mktime(time.strptime("2026-10-27T11:35:12", "%Y-%m-%dT%H:%M:%S"))
    assert seen.read_text().strip() == "args=show job -o 4242 LD=unset"


def test_a_stale_slurm_job_end_time_is_ignored(scontrol, monkeypatch):
    scontrol("EndTime=2026-10-27T11:35:12")
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time()) - 60))     # in the past: not this job's
    assert end_time() == time.mktime(time.strptime("2026-10-27T11:35:12", "%Y-%m-%dT%H:%M:%S"))


def test_end_time_unknown_or_unavailable_is_none(scontrol, monkeypatch, tmp_path):
    scontrol("EndTime=Unknown")
    assert end_time() is None
    monkeypatch.setenv("PATH", str(tmp_path / "no_such_dir"))                 # no scontrol at all
    assert end_time() is None
    monkeypatch.delenv("SLURM_JOB_ID")
    assert end_time() is None


# ---------------------------------------------------------------- markers, resume, the pointer
def _units(run: Path, n: int, walls: dict | None = None) -> None:
    for k in range(n):
        d = unit_dir(run, k)
        d.mkdir(parents=True)
        (d / "data.txt").write_text(str(k))
        mark_complete(d, {"wall_s": (walls or {}).get(k, 1.0)})


def test_mark_complete_is_atomic_and_records_stats(tmp_path):
    d = tmp_path / "unit_0000"
    d.mkdir()
    assert is_complete(d) is None
    rec = mark_complete(d, {"wall_s": 12.5, "items": 3})
    assert rec["complete"] is True and is_complete(d)["items"] == 3 and is_complete(d)["wall_s"] == 12.5
    assert [p.name for p in d.iterdir()] == [MARKER]               # no temporary file left behind
    with pytest.raises(FileNotFoundError):
        mark_complete(tmp_path / "unit_0001")


@pytest.mark.parametrize("text", ['{"complete": tr', '{"complete": false}', "[1, 2]", ""])
def test_half_written_or_foreign_markers_do_not_count(tmp_path, text):
    d = tmp_path / "unit_0000"
    d.mkdir()
    (d / MARKER).write_text(text)
    assert is_complete(d) is None and latest_complete(tmp_path) is None


def test_resume_skips_an_incomplete_newest_unit(tmp_path):
    _units(tmp_path, 3)
    (tmp_path / "unit_0003").mkdir()                                # a job was killed in unit 3
    (tmp_path / "unit_0003" / "data.txt").write_text("partial")
    (tmp_path / "unit_0007.partial-20261027T000000-1").mkdir()      # set aside earlier: ignored
    (tmp_path / "unit_x").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    os.symlink("unit_0002", tmp_path / "unit_0009")                 # a symlink is never a unit
    assert latest_complete(tmp_path) == 2
    assert latest_complete(tmp_path, prefix="gen") is None
    assert latest_complete(tmp_path / "missing") is None


def test_latest_pointer_is_never_missing_while_it_moves(tmp_path):
    _units(tmp_path, 3)
    publish_latest(tmp_path, unit_dir(tmp_path, 0))
    stop, misses = threading.Event(), []

    def reader():
        while not stop.is_set():
            try:
                os.readlink(tmp_path / "latest")
            except FileNotFoundError:
                misses.append(1)

    t = threading.Thread(target=reader)
    t.start()
    try:
        for i in range(300):
            publish_latest(tmp_path, unit_dir(tmp_path, i % 3))
    finally:
        stop.set()
        t.join()
    assert not misses, f"the pointer was missing {len(misses)} times"
    assert os.readlink(tmp_path / "latest") == "unit_0002"         # relative: the run directory can move
    assert (tmp_path / "latest" / "data.txt").read_text() == "2"
    assert not list(tmp_path.glob(".latest.tmp*"))


def test_a_failed_pointer_update_leaves_the_old_pointer(tmp_path, monkeypatch):
    _units(tmp_path, 2)
    publish_latest(tmp_path, unit_dir(tmp_path, 0))

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(chain.os, "replace", boom)
    with pytest.raises(OSError):
        publish_latest(tmp_path, unit_dir(tmp_path, 1))
    monkeypatch.undo()
    assert os.readlink(tmp_path / "latest") == "unit_0000"
    assert not list(tmp_path.glob(".latest.tmp*"))


def test_the_pointer_never_names_an_incomplete_unit(tmp_path):
    (tmp_path / "unit_0000").mkdir()
    with pytest.raises(ValueError):
        publish_latest(tmp_path, tmp_path / "unit_0000")
    assert not (tmp_path / "latest").is_symlink()


def test_done_and_stop_files(tmp_path):
    assert stop_file(tmp_path) is None
    request_stop(tmp_path, "pause for review")
    assert stop_file(tmp_path) == "STOP" and (tmp_path / "STOP").read_text() == "pause for review\n"
    mark_done(tmp_path, "max_units=3 reached")
    assert stop_file(tmp_path) == "DONE"                                # DONE wins


# ---------------------------------------------------------------- run_units
def _recording_work(seen: list):
    def work(k, out, prev):
        assert out.is_dir() and not list(out.iterdir())                 # every unit gets a fresh directory
        (out / "data.txt").write_text(f"{k} after {prev.name if prev else None}")
        seen.append(k)
        return {"items": k * 10}
    return work


def test_a_run_resumes_and_ends_with_done(tmp_path):
    seen = []
    assert run_units(tmp_path, _recording_work(seen), 3, end=None, **QUIET) == "max_units"
    assert seen == [0, 1, 2] and latest_complete(tmp_path) == 2
    assert (tmp_path / "DONE").read_text().startswith("max_units=3")
    assert os.readlink(tmp_path / "latest") == "unit_0002"
    assert (tmp_path / "unit_0002" / "data.txt").read_text() == "2 after unit_0001"
    rec = is_complete(tmp_path / "unit_0001")
    assert rec["items"] == 10 and rec["unit"] == 1 and "wall_s" in rec
    again = []
    assert run_units(tmp_path, _recording_work(again), 3, end=None, **QUIET) == "DONE" and again == []


def test_a_bounded_smoke_run_writes_done_so_it_never_chains(tmp_path):
    """HIGH (first project's review): a 5-generation smoke run never wrote DONE and would have
    chained 23.5-hour GPU jobs all the way to the full run's length."""
    assert run_units(tmp_path, lambda k, o, p: None, 2, end=time.time() + 7200, **QUIET) == "max_units"
    assert stop_file(tmp_path) == "DONE"
    with pytest.raises(ValueError):
        run_units(tmp_path / "unbounded", lambda k, o, p: None, 0, end=None, **QUIET)


def test_a_unit_killed_midway_is_redone_and_its_partial_files_are_kept(tmp_path):
    def crashy(k, out, prev):
        (out / "data.txt").write_text("partial")
        if k == 2:
            raise RuntimeError("killed")

    with pytest.raises(RuntimeError):
        run_units(tmp_path, crashy, 4, end=None, **QUIET)
    assert latest_complete(tmp_path) == 1 and os.readlink(tmp_path / "latest") == "unit_0001"
    assert is_complete(tmp_path / "unit_0002") is None and stop_file(tmp_path) is None
    seen = []
    assert run_units(tmp_path, _recording_work(seen), 4, end=None, **QUIET) == "max_units"
    assert seen == [2, 3]
    aside = list(tmp_path.glob("unit_0002.partial-*"))
    assert len(aside) == 1 and (aside[0] / "data.txt").read_text() == "partial"
    assert (tmp_path / "unit_0002" / "data.txt").read_text() == "2 after unit_0001"


def test_time_budget_returns_for_the_next_link_without_done(tmp_path):
    clk = Clock(1_000_000.0)

    def work(k, out, prev):
        clk.t += 600.0                                   # every unit takes 10 minutes

    end = clk.t + 600.0 + 2000.0                         # 2,000 s usable after the 600 s reserve
    why = run_units(tmp_path, work, 100, end=end, reserve_s=600.0, clock=clk, **QUIET)
    # unit 0 needs 120 s; units 1 and 2 need 750 s (1.25 x 600) and fit; unit 3 would end past the deadline
    assert why == "time_budget" and latest_complete(tmp_path) == 2
    assert stop_file(tmp_path) is None                   # the job script resubmits
    assert is_complete(unit_dir(tmp_path, 2))["wall_s"] == 600.0


def test_a_resumed_link_sizes_its_first_unit_from_the_marker(tmp_path):
    """1,250 s needed (1.25 x the marker's 1,000 s), 1,300 s left: unit 1 runs, unit 2 does not.
    With 1,000 s left the same marker stops the link before any unit (the no_progress test)."""
    _units(tmp_path, 1, walls={0: 1000.0})
    clk, called = Clock(2_000_000.0), []

    def work(k, out, prev):
        called.append(k)
        clk.t += 1000.0

    why = run_units(tmp_path, work, 10, end=clk.t + 600.0 + 1300.0, clock=clk, **QUIET)
    assert why == "time_budget" and called == [1] and latest_complete(tmp_path) == 1


def test_no_progress_writes_stop_so_a_chain_cannot_loop_forever(tmp_path):
    _units(tmp_path, 1, walls={0: 1000.0})                # this run's units take ~1,000 s
    clk, called = Clock(2_000_000.0), []
    why = run_units(tmp_path, lambda k, o, p: called.append(k), 10, end=clk.t + 600.0 + 1000.0, clock=clk, **QUIET)
    assert why == "no_progress" and not called
    assert stop_file(tmp_path) == "STOP" and "no_progress" in (tmp_path / "STOP").read_text()
    assert latest_complete(tmp_path) == 0


def test_stop_pauses_before_the_next_unit(tmp_path):
    request_stop(tmp_path, "human review")
    called = []
    assert run_units(tmp_path, lambda k, o, p: called.append(k), 5, end=None, **QUIET) == "STOP"
    assert not called and not (tmp_path / "DONE").exists()          # STOP is a pause, not DONE


def test_a_stop_touched_during_a_unit_lets_that_unit_finish(tmp_path):
    def work(k, out, prev):
        if k == 1:
            (tmp_path / "STOP").write_text("human\n")

    assert run_units(tmp_path, work, 5, end=None, **QUIET) == "STOP"
    assert latest_complete(tmp_path) == 1 and not (tmp_path / "DONE").exists()


def test_units_can_be_named_for_the_project(tmp_path):
    run_units(tmp_path, lambda k, o, p: None, 2, end=None, prefix="gen", **QUIET)
    assert (tmp_path / "gen_0001" / MARKER).exists() and latest_complete(tmp_path, prefix="gen") == 1
    assert latest_complete(tmp_path) is None


def test_demo_controller(tmp_path, capsys):
    assert chain.main(["--run-dir", str(tmp_path / "r"), "--max-units", "2", "--unit-seconds", "0",
                       "--server", "http://127.0.0.1:9", "--model", "m"]) == 0
    assert "chain demo stopped: max_units" in capsys.readouterr().out
    assert json.loads((tmp_path / "r" / "unit_0001" / "unit.json").read_text()) == {"unit": 1, "after": "unit_0000"}
