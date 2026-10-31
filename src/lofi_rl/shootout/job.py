"""The `lofi-shootout` job's stages, in order (docs/SHOOTOUTS.md section 7.7). `jobs/shootout.sbatch` runs

    python -m lofi_rl.shootout.job --run $REMOTE_SCR/runs/shootout-<job> --runs $REMOTE_SCR/runs \
        --inbox $REMOTE_SCR/inbox --jobs 8 <the job's ARGS>

with the ARGS `make submit JOB=jobs/shootout.sbatch ARGS='...'` passes: `--stage day1|day2|day3`, and
`--leaders keys=<id>+<id>+<id>/kit=<id>/bass=<id>/sax=<id>+<id>` from day 2 on; day 3 adds
`--prior <job>+<job>`, `--repeats <stage>+...` and, if needed, `--rerun-s2 <sax id>`; a cut adds
`--cut <name>+<name>`. ARGS hold no spaces, so they need no comma-safe `--export`.

1. Everything the job was given goes into `summary/leaders.json` first.
2. `draw`: `python -m lofi_rl.shootout.batch draw` writes `summary/draw.json` (the maps, the order, the anchors and
   repeats, and the renders and palettes the job must make), from a seed drawn there and recorded only there.
3. `measure`: the palette generator (section 7.8) with `--levels none`: the measuring palettes.
4. `levels`: the levels tool (section 7.1), with `--prior` the first prior job's levels.json (its drift keys).
5. `palettes`: the generator again, with the measured trims and faders.
6. `render`: every clip, `--jobs` at a time with 1 worker each (`python -m lofi_rl.shootout.render`).
7. `build`: `python -m lofi_rl.shootout.batch build` writes the batches into the inbox and checks each folder.

The first stage that fails ends the job with its exit code, so a gate or compile failure stops it before any
batch is written. Before any stage runs, the runner checks its arguments, that the generator and the levels
tool are present in this mirror (they are built in their own workflows, sections 7.1 and 7.8), that every
`--prior` run has its batches.json, and that none of the day's batch folders is already in the inbox (the build
never overwrites one, so a re-run fails here instead of after its renders).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

from lofi_rl.palette import ROOT

from .draw import DrawError, parse_leaders
from .plan import PLAN, PlanError, day_name, day_number, load_plan

LOOPS = ROOT / "shootouts" / "loops"
STEPS = ("draw", "measure", "levels", "palettes", "render", "build")
TOOLS = ("lofi_rl.shootout.palettes", "lofi_rl.shootout.levels")      # sections 7.8 and 7.1
LEADERS_SCHEMA = "lofi-rl/shootout-leaders/1"
_JOBS = re.compile(r"[0-9]+(\+[0-9]+)*")
_NAMES = re.compile(r"[A-Za-z0-9_-]+(\+[A-Za-z0-9_-]+)*")


class JobError(ValueError):
    pass


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.shootout.job", description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True, type=Path, help="this job's run folder, $REMOTE_SCR/runs/shootout-<job>")
    p.add_argument("--runs", type=Path, default=None, help="the folder holding every run (default: the run's parent)")
    p.add_argument("--inbox", required=True, type=Path, help="the inbox root the batches go into")
    p.add_argument("--stage", required=True, choices=("day1", "day2", "day3"))
    p.add_argument("--leaders", default=None)
    p.add_argument("--prior", default="", help="earlier lofi-shootout job ids, joined by +")
    p.add_argument("--repeats", default="", help="stages whose rounds may come back, joined by +")
    p.add_argument("--rerun-s2", default=None)
    p.add_argument("--cut", default="", help="cut-list names joined by +")
    p.add_argument("--jobs", type=int, default=8, help="renders at a time (one per CPU)")
    p.add_argument("--plan", type=Path, default=PLAN)
    p.add_argument("--loops", type=Path, default=LOOPS)
    p.add_argument("--seed", default=None, help="tests only: the draw's seed in hex")
    p.add_argument("--dry-run", action="store_true", help="print the stages and write leaders.json, run nothing")
    return p.parse_args(argv)


def _split(text: str, rx: re.Pattern, what: str) -> list[str]:
    if not text:
        return []
    if not rx.fullmatch(text):
        raise JobError(f"--{what} {text!r}: expected names joined by + ({what})")
    return text.split("+")


def check(a: argparse.Namespace) -> dict:
    """Every argument checked against the plan; the leaders.json document."""
    if a.prior and not _JOBS.fullmatch(a.prior):
        raise JobError(f"--prior {a.prior!r}: expected lofi-shootout job ids joined by +")
    prior = a.prior.split("+") if a.prior else []
    repeats = _split(a.repeats, _NAMES, "repeats")
    cut = _split(a.cut, _NAMES, "cut")
    try:
        leaders = parse_leaders(a.leaders)
        plan = load_plan(a.plan)
    except (DrawError, PlanError) as err:
        raise JobError(str(err)) from err
    if repeats and day_number(a.stage) != plan.repeat_day:
        raise JobError(f"--repeats: repeats come only on {day_name(plan.repeat_day)}")
    if a.rerun_s2 and day_number(a.stage) != plan.rerun_day:
        raise JobError("--rerun-s2: the S2 re-run comes only on "
                       + (day_name(plan.rerun_day) if plan.rerun_day else "no day"))
    cuts = sorted(c.id for c in plan.cuts)
    unknown = [c for c in cut if c not in cuts]
    if unknown:
        raise JobError(f"--cut: {unknown} are not cuts of the plan ({cuts})")
    runs = a.runs or a.run.parent
    return {"schema": LEADERS_SCHEMA, "stage": a.stage, "leaders": {k: list(v) for k, v in leaders.items()},
            "prior": prior, "prior_runs": [str(runs / f"shootout-{j}") for j in prior], "repeats": repeats,
            "rerun_s2": a.rerun_s2, "cut": cut, "jobs": a.jobs, "plan": str(a.plan), "plan_sha256": plan.sha256}


def commands(a: argparse.Namespace, doc: dict) -> list[tuple[str, list[str]]]:
    py, s = sys.executable, a.run / "summary"
    draw = [py, "-m", "lofi_rl.shootout.batch", "draw", "--plan", str(a.plan), "--stage", a.stage,
            "--summary", str(s)]
    if a.leaders:
        draw += ["--leaders", a.leaders]
    if doc["prior_runs"]:
        draw += ["--prior", *(str(Path(r) / "summary" / "batches.json") for r in doc["prior_runs"])]
    if doc["repeats"]:
        draw += ["--repeats", "+".join(doc["repeats"])]
    if a.rerun_s2:
        draw += ["--rerun-s2", a.rerun_s2]
    if doc["cut"]:
        draw += ["--cut", "+".join(doc["cut"])]
    if a.seed:
        draw += ["--seed", a.seed]
    measure = [py, "-m", "lofi_rl.shootout.palettes", "--plan", str(a.plan), "--levels", "none",
               "--draw", str(s / "draw.json"), "--out", str(s / "palettes" / "measure")]
    levels = [py, "-m", "lofi_rl.shootout.levels", "--plan", str(a.plan), "--loops", str(a.loops),
              "--palettes", str(s / "palettes" / "measure"), "--stage", a.stage]
    if doc["prior_runs"]:                       # drift keys against job 1's (section 3, "Drift between jobs")
        levels += ["--prior", str(Path(doc["prior_runs"][0]) / "summary" / "levels.json")]
    levels += ["--out", str(s / "levels.json"), "--workers", str(a.jobs)]
    palettes = [py, "-m", "lofi_rl.shootout.palettes", "--plan", str(a.plan), "--levels", str(s / "levels.json"),
                "--draw", str(s / "draw.json"), "--out", str(s / "palettes")]
    render = [py, "-m", "lofi_rl.shootout.render", "--draw", str(s / "draw.json"), "--palettes", str(s / "palettes"),
              "--out", str(a.run / "renders"), "--summary", str(s), "--jobs", str(a.jobs), "--workers", "1"]
    build = [py, "-m", "lofi_rl.shootout.batch", "build", "--plan", str(a.plan), "--stage", a.stage,
             "--renders", str(a.run / "renders"), "--out", str(a.inbox), "--draw", str(s / "draw.json")]
    return list(zip(STEPS, [draw, measure, levels, palettes, render, build]))


def run_step(name: str, argv: list[str]) -> int:
    print(f"shootout: {name}: {shlex.join(argv)}", flush=True)
    return subprocess.run(argv).returncode


def main(argv: list[str] | None = None) -> int:
    try:
        a = _args(argv)
    except SystemExit as err:                       # argparse: a bad argument (2) or --help (0)
        return int(err.code or 0)
    try:
        doc = check(a)
    except JobError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if not a.dry_run:
        missing = [m for m in TOOLS if find_spec(m) is None]
        if missing:
            print(f"error: {missing} are not in this mirror; the job needs the palette generator (SHOOTOUTS.md "
                  "section 7.8) and the levels tool (section 7.1)", file=sys.stderr)
            return 2
        for r in doc["prior_runs"]:
            if not (Path(r) / "summary" / "batches.json").is_file():
                print(f"error: --prior: {r}/summary/batches.json is missing", file=sys.stderr)
                return 2
        there = [str(a.inbox / b.date / name) for b in load_plan(a.plan).batches_of(a.stage) for name in b.names
                 if (a.inbox / b.date / name).exists()]
        if there:                                   # the build would refuse them after every render: fail now
            print(f"error: {there} exist; a built batch is never overwritten, so move them aside to re-run "
                  f"{a.stage}", file=sys.stderr)
            return 2
    summary = a.run / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    (summary / "leaders.json").write_text(json.dumps({**doc, "argv": list(argv or sys.argv[1:])}, indent=1) + "\n",
                                          encoding="utf-8")
    steps = commands(a, doc)
    if a.dry_run:
        for name, cmd in steps:
            print(f"{name}: {shlex.join(cmd)}")
        return 0
    for name, cmd in steps:
        rc = run_step(name, cmd)
        if rc != 0:
            print(f"shootout: stopped at {name} (exit {rc}); no batch is written before the build stage",
                  file=sys.stderr)
            return rc
    print("shootout: every stage done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
