"""The command line of lofi_rl.gate1 (docs/GATE1.md): one subcommand per module, each adding its own parser.

    python -m lofi_rl.gate1 check RUN_DIR [--json]

Exit codes: 0 done (check: valid and the first half met), 3 the verdict is no (check: invalid or under 300), 2 bad
input or a refusal, with nothing written.
"""
from __future__ import annotations

import argparse
import importlib

COMMANDS = ("check",)           # modules of lofi_rl.gate1, each with add_parser(sub) and main(args)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m lofi_rl.gate1", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in COMMANDS:
        importlib.import_module(f"lofi_rl.gate1.{name}").add_parser(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    return a.run_command(a)
