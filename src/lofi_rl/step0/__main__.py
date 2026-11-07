"""python -m lofi_rl.step0 generate|score|render|check|prereg|path: the step-0 samples (lofi_rl.step0)."""
from __future__ import annotations

import sys

from lofi_rl.step0.cli import main

if __name__ == "__main__":
    sys.exit(main())
