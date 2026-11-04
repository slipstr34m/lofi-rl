"""The freeze record as literals, for the freeze tests (docs/FREEZE.md §5.2: lofi_rl.freeze JOBS, REDOS, SIGNOFFS).

The live constants grow: each redo round Shresth picks in goes into REDOS (its job into JOBS), and each sign-off listen
into SIGNOFFS. A synthetic world that read them would break at the next step the docs prescribe, though the code and
the record agree. So every synthetic world pins the record it was written for:
- **TODAY** is the record of 2026-11-03: the lofi-shootout jobs of Days 1-3 and keep3's lofi-drafts job (JOBS), no redo
  round, and the one sign-off listen b1103-signoff (job 13701026) with its four Toss (LISTEN, TOSSED), as
  gates/GATE_palette.md records them. These never change: a later record only adds to them.
- **RECORDED** is a recorded-shaped future, for tests only (nobody's picks): redo1 keeps d01, d06 and d10 (d07 "none
  acceptable"), redo2 keeps d07, and a second listen passes. It keeps every rule of gate.records_problems.

`pin(monkeypatch, record)` puts a record on lofi_rl.freeze for one test. `put(record)` puts it there for a whole process,
JOBS updated in place (every `from lofi_rl.freeze import JOBS` holds that one dict), as a recorded state would be:
tests/test_freeze_record.py reruns representative tests of each freeze test module under RECORDED that way.
`live()` reads the live constants (a module that pins TODAY reads them at import, for its real-record test). Tests
only; nothing is claimed about sound."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lofi_rl import freeze as F

TOSSED = ("drafts/v1/d01_chords.ls", "drafts/v1/d06_horn.ls", "drafts/v1/d07_chords.ls", "drafts/v1/d10_groove.ls")
LISTEN = {"batch": "b1103-signoff", "job": 13701026, "tossed": TOSSED}
TODAY: dict[str, Any] = {
    "JOBS": {"day1": 13635646, "day2": 13651610, "day3": 13656048, "keep3": 13656049},
    "REDOS": (),
    "SIGNOFFS": (LISTEN,),
}

REDO1 = {"key": "keep3_redo1", "folder": "drafts/v1/redo1", "batch": "b1105-keep3-redo", "job": 13710001,
         "heading": "## Kept drafts (keep3 redo1)",
         "kept": {"d01": "drafts/v1/redo1/d01_horn.ls", "d06": "drafts/v1/redo1/d06_horn.ls",
                  "d10": "drafts/v1/redo1/d10_chords.ls"}}
REDO2 = {"key": "keep3_redo2", "folder": "drafts/v1/redo2", "batch": "b1107-keep3-redo", "job": 13720001,
         "heading": "## Kept drafts (keep3 redo2)", "kept": {"d07": "drafts/v1/redo2/d07_horn.ls"}}
RECORDED: dict[str, Any] = {
    "JOBS": {**TODAY["JOBS"], "keep3_redo1": REDO1["job"], "keep3_redo2": REDO2["job"]},
    "REDOS": (REDO1, REDO2),
    "SIGNOFFS": (LISTEN, {"batch": "b1110-signoff", "job": 13800000, "tossed": ()}),
}


def pin(monkeypatch, record: Mapping[str, Any] = TODAY) -> None:
    """`record` on lofi_rl.freeze for one test: REDOS and SIGNOFFS replaced, JOBS a fresh copy of its jobs."""
    monkeypatch.setattr(F, "REDOS", tuple(record["REDOS"]))
    monkeypatch.setattr(F, "SIGNOFFS", tuple(record["SIGNOFFS"]))
    monkeypatch.setattr(F, "JOBS", dict(record["JOBS"]))


def put(record: Mapping[str, Any] = RECORDED) -> None:
    """`record` on lofi_rl.freeze for the whole process: REDOS and SIGNOFFS replaced, its jobs added to JOBS in place."""
    F.REDOS, F.SIGNOFFS = tuple(record["REDOS"]), tuple(record["SIGNOFFS"])
    F.JOBS.update(record["JOBS"])


def live() -> dict[str, Any]:
    """The live constants as they stand: KEPT, REDOS, SIGNOFFS and a copy of JOBS."""
    return {"KEPT": dict(F.KEPT), "REDOS": F.REDOS, "SIGNOFFS": F.SIGNOFFS, "JOBS": dict(F.JOBS)}
