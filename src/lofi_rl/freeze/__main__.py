"""python -m lofi_rl.freeze build|check|vocab: the frozen palette's generator and check (lofi_rl.freeze.palette;
docs/FREEZE.md §3). The other freeze tools have their own entry points (python -m lofi_rl.freeze.<module>)."""
from __future__ import annotations

import sys

from lofi_rl.freeze.palette import main

if __name__ == "__main__":
    sys.exit(main())
