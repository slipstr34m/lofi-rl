"""A synthetic redo round for the freeze tests (docs/DRAFTS.md §6; docs/FREEZE.md §5): a drafts tree in a temporary
folder holding drafts/v1 as committed (its 30 drafts and briefs.yaml) and a `redo1/` folder of 12 drafts for the four
briefs the sign-off listen b1103-signoff reopened (d01, d06, d07, d10), each sharing its stem with a drafts/v1 file.

Each redo draft is the drafts/v1 draft of its name with the next keys instrument (rhodes -> wurli -> piano -> rhodes):
its text, and so its sha256 and its renders, differ from the drafts/v1 file of its stem, while it passes the same
gates, compile, energy and mood checks. The validator's near-copy check compares the copy form, which writes every keys
instrument `K`, so these edits sit at 5-gram Jaccard 1.0 from their drafts/v1 file by construction: a test that runs
the validator over the tree lifts lofi_rl.drafts.check.J5_MAX (`lift_near_copy_limit`). The picks: d01 horn, d06 horn
(the stem of the tossed keep3 pick drafts/v1/d06_horn.ls), d07 groove and d10 chords.

`constants(...)` gives KEPT, REDOS, SIGNOFFS and JOBS for the tree (keep3's picks as in the real gate file, at the
tree's paths; one redo round; b1103-signoff with its four Toss at the tree's paths), `patch` puts them on lofi_rl.freeze
with monkeypatch, and `gate_text` writes a gate file that records them. Tests only: nothing here is a draft anyone
wrote, and nothing is claimed about sound."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from lofi_rl import freeze as F

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "drafts" / "v1"
REOPENED = ("d01", "d06", "d07", "d10")
PERSONAS = ("groove", "horn", "chords")
PICKS = {"d01": "horn", "d06": "horn", "d07": "groove", "d10": "chords"}
KEYS = ("rhodes", "wurli", "piano")
REDO_BATCH = "b1105-keep3-redo"
LISTEN = {"batch": "b1103-signoff", "job": 13701026}
EMS = {"d01": "low, melancholy", "d02": "low, wistful", "d03": "low, hopeful", "d04": "mid, melancholy",
       "d05": "mid, wistful", "d06": "mid, hopeful", "d07": "high, melancholy", "d08": "high, wistful",
       "d09": "high, hopeful", "d10": "low, wistful"}


def rotate_keys(text: str) -> str:
    """The draft with its keys instrument moved to the next of KEYS."""
    m = re.search(r"^keys: (\w+) ", text, re.M)
    assert m and m.group(1) in KEYS, text
    new = KEYS[(KEYS.index(m.group(1)) + 1) % len(KEYS)]
    return text[:m.start(1)] + new + text[m.end(1):]


def write_tree(root: Path) -> tuple[Path, Path]:
    """(v1, redo1) under root: drafts/v1's 30 drafts and briefs.yaml, and the 12 redo drafts."""
    v1, redo1 = root / "v1", root / "v1" / "redo1"
    redo1.mkdir(parents=True)
    shutil.copyfile(V1 / "briefs.yaml", v1 / "briefs.yaml")
    for f in sorted(V1.glob("*.ls")):
        shutil.copyfile(f, v1 / f.name)
    for b in REOPENED:
        for p in PERSONAS:
            name = f"{b}_{p}.ls"
            (redo1 / name).write_text(rotate_keys((V1 / name).read_text(encoding="utf-8")), encoding="utf-8")
    return v1, redo1


def constants(v1: Path, redo1: Path, *, keep3_job: int, redo_job: int) -> dict:
    """KEPT, REDOS, SIGNOFFS and JOBS for the tree (module doc). The files are absolute paths, as the tools label a
    file outside the repo."""
    kept = {b: str(v1 / Path(f).name) for b, f in F.KEPT.items()}
    redo = {"key": "keep3_redo1", "folder": str(redo1), "batch": REDO_BATCH, "job": redo_job,
            "heading": "## Kept drafts (keep3 redo1)",
            "kept": {b: str(redo1 / f"{b}_{PICKS[b]}.ls") for b in REOPENED}}
    listen = {**LISTEN, "tossed": tuple(str(v1 / Path(f).name) for f in F.SIGNOFFS[0]["tossed"])}
    return {"KEPT": kept, "REDOS": (redo,), "SIGNOFFS": (listen,),
            "JOBS": {**F.JOBS, "keep3": keep3_job, "keep3_redo1": redo_job}}


def patch(monkeypatch, consts: dict) -> None:
    for name, value in consts.items():
        monkeypatch.setattr(F, name, value)


def lift_near_copy_limit(monkeypatch) -> None:
    """The redo stand-ins are edits of drafts/v1 drafts (module doc): the validator's 5-gram limit is lifted."""
    from lofi_rl.drafts import check as DC
    monkeypatch.setattr(DC, "J5_MAX", 1.01)


def gate_text(consts: dict) -> str:
    """A gate file recording the tree's keep3 table, its redo round and its sign-off listen, read as the real one is."""
    def table(kept: dict) -> str:
        rows = "".join(f"| {b} | {EMS[b]} | `{f}` |\n" for b, f in kept.items())
        return f"| Brief | Energy, mood | Kept |\n|---|---|---|\n{rows}"
    (redo,), (listen,) = consts["REDOS"], consts["SIGNOFFS"]
    tossed = ", ".join(f"`{f}`" for f in listen["tossed"])
    return (f"# GATE stand-in\n\n## Kept drafts (keep3): voted (a test stand-in; job {consts['JOBS']['keep3']})\n\n"
            f"{table(consts['KEPT'])}\n"
            f"## Sign-off listen: voted (batch {listen['batch']}; job {listen['job']})\n\n- **Tossed:** {tossed}.\n\n"
            f"## Kept drafts (keep3 redo1): voted (batch {redo['batch']}; job {redo['job']})\n\n{table(redo['kept'])}\n"
            f"## How to approve\n\n## Approval\n")
