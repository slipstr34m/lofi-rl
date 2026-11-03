"""The `lofi-drafts` job (docs/DRAFTS.md §7). `jobs/drafts.sbatch` runs

    python -m lofi_rl.drafts.job --run $REMOTE_SCR/runs/drafts-<job> --inbox $REMOTE_SCR/inbox --jobs 8 <ARGS>

with the ARGS `make submit JOB=jobs/drafts.sbatch ARGS='...'` passes (no spaces, so no comma-safe --export):

    --drafts drafts/v1                                               # the 30 drafts: one batch of 10 rounds
    --drafts drafts/v1/redo1 --briefs d04+d07 --against drafts/v1    # a redraft: only the named briefs

and optionally `--date YYYY-MM-DD` (the inbox's date folder; default today, Pacific) and `--name <batch>` (default
`b<MMDD>-keep3`, or for a redraft the first free of `b<MMDD>-keep3-redo`, `-redo2`, ...).

    --drafts drafts/v1 --signoff palettes/lofi_rain_v1.yaml          # the sign-off render of the kept loops

is the sign-off mode (docs/DRAFTS.md §7): the kept loops with the frozen palette, checked against what keep3 rendered
(drafts/v1/keep3_kept10.json), then one triage batch for Shresth's sign-off listen. This module only parses it and
hands the run to lofi_rl.freeze.signoff, whose docstring has its stages; `--runs-root` (where earlier jobs' run folders
are, for its cross-checks) belongs to that mode alone, and `--briefs`, `--against` and `--palette` do not.

Stages, in order; each one's exit code and wall time go into summary/stages.json and the log, and the first that fails
ends the job, so nothing reaches the inbox unless every draft passed every check and rendered in spec:
1. `check`: the validator (lofi_rl.drafts.check) over the briefs in scope, `--against` folders included in its
   Jaccard pool; summary/check.json. Any failure: exit 3.
2. `palettes`: the draft palette's own checks (lofi_rl.drafts.palette.problems), one render palette per (keys, kit) the
   drafts name into summary/palettes/ with index.json, then the shootout's validation of every palette file (each
   source hashed once against its sha256, each SoundFont preset read) and the SF2 coverage check. Any problem: exit 3.
3. `render`: every draft in scope (lofi_rl.drafts.render), `--jobs` at a time with one worker each, into
   <run>/renders/<brief>_<persona>/; summary/renders.json. A gate, compile or level failure: exit 3; an
   infrastructure fault: exit 1.
4. `build`: the keep3 batch (lofi_rl.drafts.batch) into <inbox>/<date>/<name>/, checked by `python -m
   lofi_rl.ui.manifest check`; summary/draw.json and summary/batch.json.
Bad arguments, or a batch folder that already exists, exit 2 before any stage. `--dry-run` runs the check and the
palettes, then prints what the render and the build would do, and writes nothing outside the run folder.
"""
from __future__ import annotations

import argparse
import json
import secrets as _secrets
import sys
import time
from datetime import date as _date
from datetime import datetime
from pathlib import Path

from lofi_rl.palette import ROOT, PaletteError

from . import batch as BT
from . import briefs as B
from .check import check_folder, parse_scope, table
from .palette import DRAFT_PALETTE, DraftPaletteError, load, problems, verify_files, write_render_palettes

STAGES_SCHEMA = "lofi-rl/draft-stages/1"
ARGS_SCHEMA = "lofi-rl/draft-args/1"


class JobError(ValueError):
    pass


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.drafts.job", description=__doc__.split("\n\n")[0])
    p.add_argument("--drafts", required=True, type=Path, help="the drafts folder (repo-relative or absolute)")
    p.add_argument("--run", required=True, type=Path, help="this job's run folder, $REMOTE_SCR/runs/drafts-<job>")
    p.add_argument("--inbox", required=True, type=Path, help="the inbox root the batch goes into")
    p.add_argument("--briefs", default=None, help="a redraft: the briefs to render, joined by +")
    p.add_argument("--against", type=Path, action="append", default=[], help="earlier drafts the check compares with")
    p.add_argument("--palette", type=Path, default=None, help=f"the draft palette (default {_label(DRAFT_PALETTE)})")
    p.add_argument("--signoff", type=Path, default=None, metavar="PALETTE",
                   help="the sign-off render of the kept loops with this frozen palette (lofi_rl.freeze.signoff)")
    p.add_argument("--runs-root", type=Path, default=None,
                   help="--signoff only: the folder of earlier jobs' runs, read for cross-checks (default: --run's parent)")
    p.add_argument("--date", default=None, help="the inbox date folder, YYYY-MM-DD (default: today, Pacific)")
    p.add_argument("--name", default=None, help="the batch name (default b<MMDD>-keep3[-redo<k>], b<MMDD>-signoff[-<k>])")
    p.add_argument("--jobs", type=int, default=8, help="renders at a time (one per CPU)")
    p.add_argument("--cache", type=Path, default=None, help="the one-shot cache (default $LOFI_CACHE)")
    p.add_argument("--seed", default=None, help="tests only: the batch draw's seed, 16 hex digits")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _repo_path(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def today_pacific() -> _date:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).date()
    except Exception:                                       # no tz database: the node's own date
        return _date.today()


def batch_name(day: _date, inbox: Path, redraft: bool, name: str | None) -> str:
    if name:
        return name
    base = f"b{day:%m%d}-keep3"
    if not redraft:
        return base
    k = 1
    while (inbox / day.isoformat() / (f"{base}-redo" if k == 1 else f"{base}-redo{k}")).exists():
        k += 1
    return f"{base}-redo" if k == 1 else f"{base}-redo{k}"


def _seed(text: str | None) -> int:
    if text is None:
        return _secrets.randbits(64)
    value = int(text, 16)
    if not 0 <= value < 1 << 64:
        raise ValueError("--seed must fit in 64 bits")
    return value


def _label(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


class Stages:
    """summary/stages.json, rewritten as each stage ends."""

    def __init__(self, summary: Path) -> None:
        self.summary, self.done = summary, []

    def run(self, name: str, fn) -> int:
        t0 = time.perf_counter()
        print(f"drafts: {name}", flush=True)
        try:
            rc = fn()
        except (DraftPaletteError, PaletteError, B.BriefsError, BT.BuildError, ValueError) as err:
            print(f"drafts: {name}: {err}", file=sys.stderr, flush=True)
            rc = 3
        self.done.append({"stage": name, "exit": rc, "seconds": round(time.perf_counter() - t0, 1)})
        doc = {"schema": STAGES_SCHEMA, "stages": self.done, "seconds": round(sum(s["seconds"] for s in self.done), 1)}
        tmp = self.summary / ".stages.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        tmp.replace(self.summary / "stages.json")
        print(f"drafts: {name}: done in {self.done[-1]['seconds']:.1f} s (exit {rc})", flush=True)
        return rc


def main(argv: list[str] | None = None, *, renderer=None) -> int:
    try:
        a = _args(argv)
    except SystemExit as err:
        return int(err.code or 0)
    if a.signoff is not None:
        if a.briefs is not None or a.against or a.palette is not None:
            print("error: --signoff renders every kept loop with its own palette: it takes no --briefs, --against or "
                  "--palette", file=sys.stderr)
            return 2
        from lofi_rl.freeze import signoff
        return signoff.run(a, list(argv if argv is not None else sys.argv[1:]), renderer=renderer)
    if a.runs_root is not None:
        print("error: --runs-root belongs to --signoff", file=sys.stderr)
        return 2
    drafts = _repo_path(a.drafts)
    against = [_repo_path(x) for x in a.against]
    palette_path = _repo_path(a.palette or DRAFT_PALETTE)
    try:
        if a.jobs < 1:
            raise JobError("--jobs must be at least 1")
        briefs = B.load(B.briefs_file_for(drafts))
        scope = parse_scope(a.briefs) or list(briefs)
        unknown = [s for s in scope if s not in briefs]
        if unknown:
            raise JobError(f"--briefs {unknown} are not in {B.briefs_file_for(drafts)}")
        day = _date.fromisoformat(a.date) if a.date else today_pacific()
        name = batch_name(day, a.inbox, a.briefs is not None, a.name)
        if not BT.BATCH_RE.fullmatch(name):
            raise JobError(f"--name {name!r} must match {BT.BATCH_RE.pattern}")
        if (a.inbox / day.isoformat() / name).exists():
            raise JobError(f"{a.inbox / day.isoformat() / name} exists; a built batch is never overwritten")
        seed = _seed(a.seed)
        for x in against:
            if not x.is_dir():
                raise JobError(f"--against {x} is not a folder")
            if x.resolve() == drafts.resolve():
                raise JobError(f"--against {x} is the drafts folder itself (each draft would match itself)")
    except (JobError, B.BriefsError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    summary = a.run / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    (summary / "args.json").write_text(json.dumps({
        "schema": ARGS_SCHEMA, "argv": list(argv if argv is not None else sys.argv[1:]), "drafts": _label(drafts),
        "scope": scope, "against": [_label(x) for x in against], "palette": _label(palette_path),
        "date": day.isoformat(), "batch": name, "jobs": a.jobs, "dry_run": a.dry_run}, indent=1) + "\n",
        encoding="utf-8")
    stages = Stages(summary)
    state: dict = {}

    def check() -> int:
        rep = check_folder(drafts, scope=scope, against=against, palette_path=palette_path)
        (summary / "check.json").write_text(json.dumps(rep.as_dict(), indent=1) + "\n", encoding="utf-8")
        print(table(rep), flush=True)
        for d in rep.drafts:
            for c, why in d.problems.items():
                print(f"{d.file}: {c}: {why}", file=sys.stderr)
        for m in rep.missing:
            print(f"{m}: missing", file=sys.stderr)
        state["drafts"] = [d for d in rep.drafts if d.brief_id in scope]
        return 0 if rep.ok else 3

    def palettes() -> int:
        dp = load(palette_path)
        bad = problems(dp)
        combos = sorted({(d.loop.keys.instrument, d.loop.kit) for d in state["drafts"]})
        index = write_render_palettes(dp, combos, summary / "palettes")
        files = [summary / "palettes" / e["file"] for e in index["palettes"]]
        if not bad:
            bad = verify_files(files)
        for x in bad:
            print(f"palette: {x}", file=sys.stderr)
        state.update(dp=dp, index={(e["keys"], e["kit"]): e for e in index["palettes"]})
        return 3 if bad else 0

    def tasks() -> list[dict]:
        out = []
        for d in state["drafts"]:
            e = state["index"][(d.loop.keys.instrument, d.loop.kit)]
            b = briefs[d.brief_id].brief
            out.append({"render": d.path.stem, "file": _label(d.path), "text": d.text, "brief_id": d.brief_id,
                        "brief": {"energy": b.energy, "mood": b.mood, "bpm": b.bpm, "tonic": b.tonic},
                        "persona": d.persona, "keys": d.loop.keys.instrument, "kit": d.loop.kit,
                        "palette": str(summary / "palettes" / e["file"]),
                        "s_star": state["dp"].levels.get("S_star")})
        return out

    def render() -> int:
        from .render import render_all
        recs = render_all(tasks(), a.run / "renders", jobs=a.jobs, workers=1, cache=a.cache, summary_dir=summary,
                          renderer=renderer)
        state["records"] = recs
        bad = {rid: r for rid, r in recs.items() if r["status"] != "ok"}
        for rid, r in sorted(bad.items()):
            print(f"render {rid}: {r['status']}: {str(r['error'] or '').strip().splitlines()[-1:]}", file=sys.stderr)
        ok = [r for r in recs.values() if r["status"] == "ok"]
        if ok:
            lufs = [r["lufs"] for r in ok]
            peaks = [r["true_peak_dbtp"] for r in ok]
            print(f"render: {len(ok)} of {len(recs)} ok, {min(lufs):.3f} ... {max(lufs):.3f} LUFS, true peak at most "
                  f"{max(peaks):.3f} dBTP", flush=True)
        if any(r["status"] == "error" for r in bad.values()):
            return 1
        return 3 if bad else 0

    def build() -> int:
        folder = BT.build(state["records"], a.run / "renders", briefs={b: briefs[b].as_dict() for b in scope},
                          name=name, date=day.isoformat(), out_root=a.inbox, seed=seed, summary_dir=summary)
        doc = json.loads((summary / "batch.json").read_text(encoding="utf-8"))
        clips = sum(len(r["clips"]) for r in doc["rounds"])
        print(f"batch {name}: {len(doc['rounds'])} rounds, {clips} clips, est {doc['est_min']} min -> {folder}")
        return 0

    for stage_name, fn in (("check", check), ("palettes", palettes)):
        rc = stages.run(stage_name, fn)
        if rc:
            print(f"drafts: stopped at {stage_name} (exit {rc}); nothing was rendered or built", file=sys.stderr)
            return rc
    if a.dry_run:
        for t in tasks():
            print(f"would render {t['render']} ({t['file']}) with {Path(t['palette']).name}")
        print(f"would build {a.inbox / day.isoformat() / name} ({len(scope)} rounds)")
        return 0
    for stage_name, fn in (("render", render), ("build", build)):
        rc = stages.run(stage_name, fn)
        if rc:
            print(f"drafts: stopped at {stage_name} (exit {rc}); no batch is written before the build",
                  file=sys.stderr)
            return rc
    print(f"drafts: every stage done in {sum(s['seconds'] for s in stages.done):.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
