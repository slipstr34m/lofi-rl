"""The command line of lofi_rl.gate1 (docs/GATE1.md): one subcommand per module, each adding its own parser.

    python -m lofi_rl.gate1 check RUN_DIR [--json]
    python -m lofi_rl.gate1 batch RUN_DIR --clips DIR [--inbox DIR] [--date YYYY-MM-DD]
    python -m lofi_rl.gate1 tally BATCH_DIR --votes FILE [--json OUT]
    python -m lofi_rl.gate1 gate --run RUN_DIR [--batch BATCH_DIR --votes FILE] --write gates/GATE_phase1.md

Exit codes: 0 done (check: valid and the first half met; tally: every round voted and the "fine" bar met), 3 the
verdict is no or the run does not pass check (batch refuses such a run), 2 bad input or a refusal, with nothing
written.
"""
from __future__ import annotations

import argparse
import importlib

# subcommand -> its module in lofi_rl.gate1, each with add_parser(sub) and a main(args) it sets as `run_command`
COMMANDS = {"check": "check", "batch": "batch", "tally": "tally", "gate": "gatefile"}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m lofi_rl.gate1", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for module in COMMANDS.values():
        importlib.import_module(f"lofi_rl.gate1.{module}").add_parser(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    return a.run_command(a)
