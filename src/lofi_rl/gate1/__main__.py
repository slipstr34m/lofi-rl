"""python -m lofi_rl.gate1 check|batch|tally|gate: the Phase-1 gate's four steps (lofi_rl.gate1; docs/GATE1.md)."""
from __future__ import annotations

import sys

from lofi_rl.gate1.cli import main

if __name__ == "__main__":
    sys.exit(main())
