"""Human gates: a gate passes only on an explicit, dated approval line written by the human.

    from kit.gates import gate_approved
    gate_approved("gates/GATE_2.md")                          # any approver
    gate_approved("gates/GATE_6_publish.md", "Ada Lovelace")  # that approver only

    python -m kit.gates gates/GATE_2.md [--name "Ada Lovelace"]   # exit 0 approved, 1 not

An approval is a line of its own, `APPROVED by <Name> <YYYY-MM-DD>`: at column 0, nothing after the
date, a real calendar date, outside code fences and HTML comments. The gate file's own instructions
("append `APPROVED by <Name> <YYYY-MM-DD>`"), quoted, indented, bulleted or fenced examples, unfilled
placeholders and "NOT APPROVED" never count. A substring check once accepted a publish gate's own
instructions as the human's go. The agent records a chat approval by appending such a line itself.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
from pathlib import Path

# the name may not start with a placeholder or markup character (_ < ` * [ #) and holds no < > ` * [ ]
APPROVAL_LINE = re.compile(r"^APPROVED by (?P<who>[^\s<>`*_\[\]#][^<>`*\[\]]*?) (?P<date>\d{4}-\d{2}-\d{2})[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def approvals(path: str | os.PathLike) -> list[tuple[str, str]]:
    """Every valid (name, date) approval in the gate file, in file order; [] if it is missing."""
    try:
        text = Path(path).read_text(encoding="utf-8").lstrip("﻿")
    except (OSError, UnicodeDecodeError):
        return []
    found, fence, in_comment = [], None, False
    for line in text.splitlines():
        if in_comment:
            in_comment = "-->" not in line
            continue
        m = _FENCE.match(line)
        if m:
            mark = m.group(1)
            if fence is None:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        if "<!--" in line and "-->" not in line.split("<!--", 1)[1]:
            in_comment = True
            continue
        m = APPROVAL_LINE.match(line)
        if not m:
            continue
        try:
            dt.date.fromisoformat(m.group("date"))
        except ValueError:
            continue
        found.append((" ".join(m.group("who").split()), m.group("date")))
    return found


def gate_approved(path: str | os.PathLike, name: str | None = None) -> bool:
    """True when the gate file holds an approval line (by `name`, when given)."""
    found = approvals(path)
    if name is None:
        return bool(found)
    want = " ".join(name.split())
    return any(who == want for who, _ in found)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="exit 0 when a gate file holds an approval line")
    ap.add_argument("gate", type=Path)
    ap.add_argument("--name", default=None, help="require this approver")
    a = ap.parse_args(argv)
    found = approvals(a.gate)
    ok = gate_approved(a.gate, a.name)
    seen = "; ".join(f"{who} {date}" for who, date in found)
    print(f"{a.gate}: {'APPROVED' if ok else 'NOT approved'}" + (f" (approval lines: {seen})" if seen else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
