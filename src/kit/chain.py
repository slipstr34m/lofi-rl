"""Chained long runs: a SLURM job works in units (generations, epochs, batches), checkpoints every
unit, returns before its time limit, and jobs/chained_template.sbatch resubmits it as the next link.

    from kit import chain

    def work(k, out_dir, prev_dir):      # unit k: read prev_dir's checkpoint (None for unit 0),
        ...                              # write unit k's files into out_dir, return a dict of stats
        return {"items": n}

    why = chain.run_units(run_dir, work, max_units=100)

run_units resumes after the newest COMPLETE unit, never starts a unit that might not finish before
the deadline, marks each unit complete LAST and then moves the `latest` pointer atomically, and
writes DONE at max_units, so the job script stops resubmitting. A smoke test is simply a small
max_units: it ends with DONE and can never chain. The pieces are public for controllers that need
their own loop; `python -m kit.chain` is a demo controller for testing a chain end to end.

A run directory:
    <run>/unit_0000/ ... COMPLETE.json   one directory per unit; COMPLETE.json, written last, marks it complete
    <run>/latest -> unit_XXXX            the newest complete unit (a relative symlink, replaced atomically)
    <run>/DONE                           the run is finished: never resubmit
    <run>/STOP                           pause: the current unit finishes, nothing resubmits (delete it to resume)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable

MARKER = "COMPLETE.json"
DONE, STOP = "DONE", "STOP"
LATEST = "latest"
FACTOR, FLOOR_S = 1.25, 120.0     # the next unit needs max(1.25 x the last unit's wall time, 120 s)
RESERVE_S = 600.0                 # kept free after the controller returns: stop the server, resubmit
SLURM = "slurm"                   # end=SLURM: read this job's end time from SLURM


def end_time() -> float | None:
    """This job's end as a Unix time: SLURM_JOB_END_TIME, else `scontrol show job -o $SLURM_JOB_ID`;
    None outside SLURM. scontrol runs without conda's LD_LIBRARY_PATH (conda's libs broke system tools)."""
    v = os.environ.get("SLURM_JOB_END_TIME", "")
    if v.isdigit() and int(v) > time.time():          # a stale value (inherited, or in the past) is ignored
        return float(v)
    jid = os.environ.get("SLURM_JOB_ID")
    if not jid:
        return None
    env = {k: val for k, val in os.environ.items() if k != "LD_LIBRARY_PATH"}
    try:
        out = subprocess.run(["scontrol", "show", "job", "-o", jid], capture_output=True, text=True,
                             timeout=30, env=env).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"\bEndTime=(\S+)", out)
    if not m:
        return None
    try:
        return time.mktime(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S"))   # scontrol prints local time
    except ValueError:                                                      # Unknown, N/A
        return None


def deadline(reserve_s: float = RESERVE_S, end: float | str | None = SLURM) -> float | None:
    """When the controller must have returned: the job's end minus `reserve_s` for stopping the server
    and the guard's resubmission. A link that SLURM kills before it resubmits ends the whole chain."""
    if end == SLURM:
        end = end_time()
    return None if end is None else float(end) - reserve_s


def should_stop(last_unit_wall_s: float, deadline: float | None, now: float | None = None) -> bool:
    """True when the next unit might not finish before `deadline`: it needs max(1.25 x the last unit's
    wall time, 120 s). A fixed margin let SLURM kill a long unit mid-way and waste its GPU time."""
    if deadline is None:
        return False
    now = time.time() if now is None else now
    return now + max(FACTOR * max(float(last_unit_wall_s), 0.0), FLOOR_S) > deadline


def unit_dir(run_dir: str | os.PathLike, k: int, prefix: str = "unit") -> Path:
    return Path(run_dir) / f"{prefix}_{k:04d}"


def _atomic_write(path: Path, text: str) -> None:
    """tmp file + fsync + os.replace: readers see the old file or the whole new one, never a torn one."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def mark_complete(unit_dir: str | os.PathLike, stats: dict | None = None) -> dict:
    """Write the unit's completeness marker. Call it LAST, after every file of the unit is closed:
    resume trusts only marked units, and the atomic write means a kill leaves no half-written marker."""
    d = Path(unit_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"unit directory {d} does not exist")
    rec = dict(stats or {}, complete=True, unit_dir=d.name, written_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"))
    _atomic_write(d / MARKER, json.dumps(rec, indent=1, default=str) + "\n")
    return rec


def is_complete(unit_dir: str | os.PathLike) -> dict | None:
    """The marker's contents when `unit_dir` is a complete unit; None when the marker is missing,
    unreadable or not a completeness record (a job killed mid-unit leaves a directory without one)."""
    try:
        rec = json.loads((Path(unit_dir) / MARKER).read_text())
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) and rec.get("complete") is True else None


def latest_complete(run_dir: str | os.PathLike, prefix: str = "unit") -> int | None:
    """Index of the newest COMPLETE unit (resume after it), or None for a fresh run. Scans the unit
    directories themselves: never the `latest` pointer, never a directory without its marker."""
    run = Path(run_dir)
    if not run.is_dir():
        return None
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    found = [int(m.group(1)) for p in run.iterdir()
             if (m := pat.match(p.name)) and p.is_dir() and not p.is_symlink() and is_complete(p) is not None]
    return max(found) if found else None


def publish_latest(run_dir: str | os.PathLike, unit_dir: str | os.PathLike, name: str = LATEST) -> Path:
    """Point <run>/<name> at a complete unit atomically: a new symlink under a temporary name, then
    os.replace over the old one. Readers (a pull, a resumed job) never find it missing or half-made."""
    run, d = Path(run_dir), Path(unit_dir)
    if is_complete(d) is None:
        raise ValueError(f"{d} is not a complete unit (call mark_complete first)")
    link, tmp = run / name, run / f".{name}.tmp-{os.getpid()}"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(os.path.relpath(d, run), tmp)          # relative: the run directory can be moved
    try:
        os.replace(tmp, link)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return link


def stop_file(run_dir: str | os.PathLike) -> str | None:
    """'DONE' or 'STOP' when that file exists in the run directory (DONE first), else None.
    jobs/chained_template.sbatch checks the same two files before it loads a model."""
    for f in (DONE, STOP):
        if (Path(run_dir) / f).exists():
            return f
    return None


def mark_done(run_dir: str | os.PathLike, reason: str) -> None:
    """The run is finished: the job script never resubmits a run with DONE."""
    _atomic_write(Path(run_dir) / DONE, reason.rstrip() + "\n")


def request_stop(run_dir: str | os.PathLike, reason: str = "requested") -> None:
    """Pause the chain: the current unit finishes, nothing resubmits; delete STOP to resume."""
    _atomic_write(Path(run_dir) / STOP, reason.rstrip() + "\n")


def _set_aside(d: Path) -> Path:
    """Move an incomplete unit (a job was killed in it) out of the way; it is kept, never deleted."""
    dest = d.with_name(f"{d.name}.partial-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}")
    os.replace(d, dest)
    return dest


def run_units(run_dir: str | os.PathLike, work: Callable[[int, Path, Path | None], dict | None], max_units: int,
              *, reserve_s: float = RESERVE_S, end: float | str | None = SLURM, prefix: str = "unit",
              log: Callable[[str], None] = print, clock: Callable[[], float] = time.time) -> str:
    """Run units of a chained run until one of these, and return it:
      "DONE"         the run was already finished
      "STOP"         a STOP file: pause (nothing is resubmitted)
      "max_units"    unit max_units-1 is complete: DONE is written, so the chain ends
      "time_budget"  the next unit might not finish before the deadline: return; the job resubmits
      "no_progress"  not even one unit fits in this job's time: STOP is written, because a chain that
                     never progresses would resubmit itself forever (raise --time or shorten a unit)
    work(k, out_dir, prev_dir) does unit k into the fresh directory out_dir (prev_dir is unit k-1, or
    None for k == 0) and returns stats for the marker. An exception propagates and nothing is marked:
    the next job redoes unit k, after moving the partial directory aside. end=None means no deadline."""
    if max_units < 1:
        raise ValueError("max_units must be >= 1: every run is bounded")
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    dl = deadline(reserve_s, end)
    last = latest_complete(run, prefix)
    k = 0 if last is None else last + 1
    last_wall = float((is_complete(unit_dir(run, last, prefix)) or {}).get("wall_s", 0.0)) if last is not None else 0.0
    log(f"chain: run={run} resume_at={k} max_units={max_units} last_wall_s={last_wall:.1f} "
        f"deadline={time.strftime('%F %T', time.localtime(dl)) if dl else None}")
    done_here = 0
    while True:
        f = stop_file(run)
        if f:
            log(f"chain: {f} file present; stopping at unit {k}")
            return f
        if k >= max_units:
            mark_done(run, f"max_units={max_units} reached (last unit {k - 1})")
            log(f"chain: max_units={max_units} reached; DONE written")
            return "max_units"
        if should_stop(last_wall, dl, clock()):
            if done_here == 0:
                need = max(FACTOR * last_wall, FLOOR_S)
                request_stop(run, f"no_progress: unit {k} needs ~{need:.0f} s but only {dl - clock():.0f} s "
                                  f"were left before the deadline; raise the job's --time or shorten a unit")
                log(f"chain: no unit fits in this job (need ~{need:.0f} s); STOP written")
                return "no_progress"
            log(f"chain: time budget: unit {k} might not finish; returning for the next link")
            return "time_budget"
        out = unit_dir(run, k, prefix)
        if out.exists() or out.is_symlink():
            log(f"chain: unit {k} is incomplete (a job was killed in it); moved to {_set_aside(out).name}")
        out.mkdir()
        prev = unit_dir(run, k - 1, prefix) if k > 0 else None
        t0 = clock()
        stats = work(k, out, prev) or {}
        wall = clock() - t0
        mark_complete(out, dict(stats, unit=k, wall_s=round(wall, 3)))
        publish_latest(run, out)
        log(f"chain: unit {k} complete in {wall:.1f} s")
        last_wall, k, done_here = wall, k + 1, done_here + 1


def main(argv=None) -> int:
    """Demo controller (MODULE=kit.chain in jobs/chained_template.sbatch): --max-units units of
    --unit-seconds each, so the chain machinery can be tried before the project's controller exists."""
    ap = argparse.ArgumentParser(description="demo controller for a chained run")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--max-units", type=int, required=True)
    ap.add_argument("--unit-seconds", type=float, default=5.0)
    ap.add_argument("--reserve-s", type=float, default=RESERVE_S)
    ap.add_argument("--server", default=None, help="passed by the job template; unused by the demo")
    ap.add_argument("--model", default=None, help="passed by the job template; unused by the demo")
    a = ap.parse_args(argv)

    def work(k: int, out: Path, prev: Path | None) -> dict:
        time.sleep(a.unit_seconds)
        (out / "unit.json").write_text(json.dumps({"unit": k, "after": prev.name if prev else None}) + "\n")
        return {"slept_s": a.unit_seconds}

    why = run_units(a.run_dir, work, a.max_units, reserve_s=a.reserve_s)
    print(f"chain demo stopped: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
