"""The command line of lofi_rl.step0 (docs/design/step0_samples.md §2.3-2.5, §4.1, §4.2, §4.4): one subcommand per
module, each module's main(args) taking the parsed arguments.

    python -m lofi_rl.step0 generate --run DIR [--deadline-s 960] [--chunk 128] [--hang-s S]
    python -m lofi_rl.step0 score --run DIR [--from OUT]
    python -m lofi_rl.step0 render --run DIR [--from OUT] [--workers 4] [--deadline-s 900] [--cache DIR]
    python -m lofi_rl.step0 check RUN_DIR [--redo REDO_DIR] [--json] [--rescore]
    python -m lofi_rl.step0 prereg RUN_DIR [--redo REDO_DIR] [--write]
    python -m lofi_rl.step0 path [--write | --check | --commit REV] [--root DIR]

generate takes no model, prompt, palette or sampling: they are registered (lofi_rl.step0), as are the streams it draws.
`--from OUT --run REDO` redoes score or render from a run's own records into a new folder, never writing under OUT
(§4.2). Each module is imported only when its subcommand runs, so a step's env never imports another step's stack, and
a module not built yet is a clear error (exit 2).

Exit codes: 0 done (check: valid), 1 a step failed (render: a held-out brief left undetermined), 3 the verdict is no
(check: invalid; path: a listed file differs), 2 bad input or a refusal, with nothing drawn or written.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import lofi_rl.step0 as S
from lofi_rl.brief_streams import ROOT

PACKAGE = "lofi_rl.step0"
# subcommand -> its module in lofi_rl.step0, each with a main(args) -> exit code
COMMANDS = {"generate": "generate", "score": "score", "render": "render", "check": "check", "prereg": "prereg",
            "path": "path"}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m lofi_rl.step0", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="the draws, with the pre-checks, the probe and the replay (policy env)")
    g.add_argument("--run", type=Path, required=True, help="the run folder (the node's own disk in the job)")
    g.add_argument("--deadline-s", type=float, default=S.GENERATE_DEADLINE_S,
                   help="no chunk starts that would end after this many seconds")
    g.add_argument("--chunk", type=int, default=S.CHUNK, help="requests per engine call")
    g.add_argument("--hang-s", type=float, default=None,
                   help="dump stacks and exit after this many seconds (default deadline + junk.HANG_MARGIN_S; 0 off)")

    s = sub.add_parser("score", help="junk.score_row on every draw, the readouts per stream, the candidates (policy)")
    s.add_argument("--run", type=Path, required=True, help="the run folder it writes (REDO with --from)")
    s.add_argument("--from", dest="from_", type=Path, default=None, metavar="OUT",
                   help="a recovery (§4.2): read OUT's records, read-only, and write under --run only")

    r = sub.add_parser("render", help="the held-out renders in rounds, then the sets (scorer env)")
    r.add_argument("--run", type=Path, required=True, help="the run folder it writes (REDO with --from)")
    r.add_argument("--from", dest="from_", type=Path, default=None, metavar="OUT",
                   help="a recovery (§4.2): read OUT's records, read-only, and write under --run only")
    r.add_argument("--workers", type=int, default=S.RENDER_WORKERS,
                   help="worker threads for the step, two renders at a time (junk.render_lanes)")
    r.add_argument("--deadline-s", type=float, default=S.RENDER_DEADLINE_S, help="no render starts after this")
    r.add_argument("--cache", type=Path, default=None, help="one-shot cache (default $LOFI_CACHE)")

    c = sub.add_parser("check", help="the validity list (§4.1), on the laptop after make pull")
    c.add_argument("run", type=Path, help="the run folder, runs/step0-<job> (its summary/ is read)")
    c.add_argument("--redo", type=Path, default=None, metavar="REDO_DIR",
                   help="a recovery's folder: generate's records from RUN, score's and render's from it (§4.2)")
    c.add_argument("--json", action="store_true", help="print the report as JSON")
    c.add_argument("--rescore", action="store_true",
                   help="re-score completions.jsonl with this code and list differences (shown, never judged)")

    p = sub.add_parser("prereg", help="docs/prereg/heldout_v1.json and step0_v1.json from a valid run (§4.4)")
    p.add_argument("run", type=Path, help="the run folder, runs/step0-<job>")
    p.add_argument("--redo", type=Path, default=None, metavar="REDO_DIR", help="its recovery's folder, if any")
    p.add_argument("--write", action="store_true", help="write both files (never over a different file)")

    t = sub.add_parser("path", help=f"{S.PATH_FILE}: the blob shas of every file the job runs or reads (§4.1)")
    mode = t.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help=f"write {S.PATH_FILE} under --root")
    mode.add_argument("--check", action="store_true", help="every listed file has its blob under --root (no git)")
    mode.add_argument("--commit", default=None, metavar="REV",
                      help="every listed file has its blob at commit REV of the repo --root (git; the laptop)")
    t.add_argument("--root", type=Path, default=ROOT, help="the tree (default: this checkout)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = parser()
    a = ap.parse_args(argv)
    if a.cmd == "generate" and (a.chunk < 1 or a.deadline_s <= 0):
        ap.error("--chunk must be at least 1 and --deadline-s above 0")
    if a.cmd == "render" and (a.workers < 1 or a.deadline_s <= 0):
        ap.error("--workers must be at least 1 and --deadline-s above 0")
    name = f"{PACKAGE}.{COMMANDS[a.cmd]}"
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as err:
        if err.name != name:
            raise
        print(f"error: {name} is not built yet (src/lofi_rl/step0/{COMMANDS[a.cmd]}.py), so `{a.cmd}` cannot run",
              file=sys.stderr)
        return 2
    return module.main(a)
