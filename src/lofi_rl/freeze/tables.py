"""The drawn round tables, published in gates/GATE_palette.md after the freeze (docs/SHOOTOUTS.md §2.3, "Published
after the freeze"; §1, "From votes to the freeze", step 4; the freeze spec §5.2), and kept in full in a private file.

    python -m lofi_rl.freeze.tables [--runs runs] [--inbox inbox] [--gate gates/GATE_palette.md]   # print the section
    python -m lofi_rl.freeze.tables --write gates/GATE_palette.md --private [--runs runs] [--inbox inbox]
                                    # the full tables into the private file, then the section into the gate file
    python -m lofi_rl.freeze.tables --private [data/votes/audit/drawn_round_tables.md] [...]        # the full tables

**What it publishes** in the gate file, for the jobs in lofi_rl.freeze.JOBS (`render`): the shootout days, keep3 and
each redo round of lofi_rl.freeze.REDOS, whose job is JOBS[its key]; the design rounds only.
Phase 2's votes and the final A/B use the same hidden checks as the shootouts (taste_pipeline.md, "Hidden controls"),
and the rater is never told which rounds those are (RATING.md, "Checks you won't see"). So the published section
neither says how those rounds are built nor lets anyone tell which of his rounds they were: it holds no such round,
no serving position of a shootout round, and no legend of their kinds. One line gives how many rounds it leaves out
and why. (A Claude's default from review finding S4-6, to be recorded in D-022.)
- **Each lofi-shootout job (Days 1-3):**
  - each stage's symbol-to-option maps (`summary/draw.json`, `maps`). A stage with several maps under the same symbols
    (the vamps, one map per mood) names the design slots each one was drawn for;
  - then each batch's design rounds by slot (`summary/batches.json`), not in serving order: the slot and loop, and
    each tile's option with what it played, left to right as served.
- **The keep3 job, and each redo round's job** (`lofi-drafts` too, no hidden checks), each under its own heading
  (`Drawn, keep3`, `Drawn, keep3 redo1`, ...): which persona each tile of each row of its Latin square holds
  (`summary/draw.json`), then each round in serving order (`summary/batch.json`): the brief, energy and mood, the row,
  and each tile's persona and draft.

**What the private file holds** (`render_private`, written only with `--private`, only under `data/`, which git and
the sync never carry; by default `data/votes/audit/drawn_round_tables.md`, beside the votes' audit files): the full
tables, every round of every batch in serving order with its position and kind. The kind is `design`, or
`repeat of <slot>` with where the slot's round was first shown (its tiles reversed), or `anchor of <slot>`, whose
broken tile (the sour copy) is marked. It is never printed.

**Left out of both:** the 64-bit seeds, which are never committed (SHOOTOUTS.md §2.3, CD-21); the opaque round and
clip ids; the real clip ids; and the audio hashes. Each job's two files are named by their sha256 instead, under their
repo-relative paths, so the bytes do not depend on where the files were read.

**Cross-checks.** Any failure refuses (exit 3) and lists every one:
- the draw against the batches: schemas, stage, run, batch names, dates and categories, and each round's slot, loop,
  kind, stage and tiles (option, what, render);
- each batch against its `inbox/<date>/<batch>/manifest.json`: the round order, each tile's opaque id and FLAC
  sha256, and each round's bpm against the draw's;
- each batch against its `manifest.hidden.json`: its rounds, each round's kind, `of`, `anchor_clip`, brief and
  category, and each clip's real id, option, what, stage, slot, loop, render and pcm_sha256;
- every design or anchor round's options are among its stage's map values;
- an anchor's broken tile is the draw's `anchor_pos`;
- a repeat's `of` resolves to a design round of an earlier job's batch (the draw's `of_batch` and `of_run`), on the
  same slot and loop, with the same real clips in reverse order;
- for keep3, every round's tiles are its row of the square.

**Before the freeze it refuses** (exit 3) unless the gate file holds his APPROVED line, read as the seal reads it
(lofi_rl.freeze.seal.approval). `--write` puts the section `## Drawn round tables (published after the freeze)` right
after the `## Approval` section, found as lofi_rl.freeze.gate finds it (`gate.sections`: code fences and HTML comments
are never read), and leaves every other byte of the file as it was. `--private` writes the full tables to its path,
which must lie under the repo's `data/` folder and whose folder must exist (it is never created). `--write` publishes
only once the full tables are recorded privately: give `--private` with it, or run `--private` first (without it,
`--write` checks the default file). Either file: writing the same tables again changes nothing, and writing different
tables refuses. Both flags may be given; a refusal of either writes neither, and the private file is written first.
The same input files give the same bytes.

Sound claims: none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from lofi_rl import freeze as F
from lofi_rl.drafts.batch import BATCH_SCHEMA as KEEP3_BATCH_SCHEMA
from lofi_rl.drafts.batch import CATEGORY as KEEP3_CATEGORY
from lofi_rl.drafts.batch import DRAW_SCHEMA as KEEP3_DRAW_SCHEMA
from lofi_rl.palette import ROOT
from lofi_rl.shootout.batch import BATCHES_SCHEMA
from lofi_rl.shootout.draw import SCHEMA as SHOOTOUT_DRAW_SCHEMA
from lofi_rl.ui.manifest import HIDDEN_NAME, PUBLIC_NAME
from lofi_rl.ui.votes import atomic_write

from . import JOBS
from . import gate as G
from .gate import APPROVAL_HEADING
from .seal import GATE_PATH, SealError, approval

SECTION = "## Drawn round tables (published after the freeze)"
PRIVATE_TITLE = "# Drawn round tables, in full (private)"
PRIVATE_ROOT = ROOT / "data"                    # gitignored and never synced (.gitignore, .rsyncignore; CLAUDE.md §7)
DAYS = ("day1", "day2", "day3")
DAY_NAMES = {"day1": "Day 1", "day2": "Day 2", "day3": "Day 3"}
KINDS = {"none": "design", "repeat": "repeat", "anchor": "anchor"}
CLIP_FIELDS = ("clip", "option", "what", "render", "pcm_sha256")                # batches.json clip = hidden clip
KEEP3_FIELDS = (("clip", "clip"), ("persona", "persona"), ("draft", "draft"), ("render", "render"),
                ("pcm_sha256", "pcm_sha256"))


class TablesError(ValueError):
    """The tables refused. `problems` lists every problem found."""

    def __init__(self, problems: Sequence[str]):
        self.problems = [str(p) for p in problems] or ["refused"]
        super().__init__("; ".join(self.problems))


class BadInput(TablesError):
    """An input file is missing or unreadable, or an output path is not allowed (exit 2)."""


@dataclass(frozen=True)
class Tile:
    option: str          # the option (keep3: the persona)
    what: str            # what the clip played (keep3: the draft file)
    broken: bool = False  # an anchor's sour copy


@dataclass(frozen=True)
class Row:
    index: int
    slot: str            # keep3: the brief
    loop: str            # keep3: "energy, mood"
    kind: str
    tiles: tuple[Tile, ...]
    row: int | None = None   # keep3: the square's row


@dataclass(frozen=True)
class BatchTable:
    name: str
    folder: str          # <date>/<batch> under the inbox
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class JobTables:
    key: str             # day1 | day2 | day3 | keep3 | keep3_redo<k> (a REDOS key)
    job: int
    files: tuple[tuple[str, str], ...]               # (repo-relative path, sha256 of its bytes)
    maps: tuple[tuple[str, str, str], ...] = ()      # (stage label, symbol, option): shootout jobs; the label of a
    #                                                  shared-symbol map names its design and anchor slots
    square: tuple[tuple[str, ...], ...] = ()         # the persona on each tile of each row: keep3
    order: str = ""                                  # keep3: whether the order kept cells apart
    batches: tuple[BatchTable, ...] = ()
    public_maps: tuple[tuple[str, str, str], ...] = ()   # `maps`, each label naming its design slots only


# --------------------------------------------------------------------------------------------- reading
def _json(path: Path) -> tuple[Any, str]:
    try:
        data = Path(path).read_bytes()
        return json.loads(data.decode("utf-8")), hashlib.sha256(data).hexdigest()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise BadInput([f"{path}: {err}"]) from None


def _slot_key(slot: str) -> tuple:
    stage, _, n = slot.rpartition(".")
    return (stage, int(n)) if n.isdigit() else (slot, 0)


def _tiles(rnd: Mapping) -> list[Mapping]:
    return sorted(rnd["clips"], key=lambda c: c["pos"])


def _check_public_hidden(where: str, folder: Path, name: str, rounds: Sequence[Mapping], category: str,
                         problems: list[str]) -> tuple[dict, dict]:
    """The batch's two manifests against its rounds (each with `id` and tiles carrying `id` and `flac_sha256`): the
    order, the opaque ids and FLAC hashes, and the hidden rounds' ids, batch and category. Returns (public rounds by
    id, hidden rounds by id) for the caller's field checks."""
    pub, _ = _json(folder / PUBLIC_NAME)
    hid, _ = _json(folder / HIDDEN_NAME)
    if pub.get("batch") != name or hid.get("batch") != name:
        problems.append(f"{where}: its manifests name batch {pub.get('batch')!r} and {hid.get('batch')!r}")
    if hid.get("category") != category:
        problems.append(f"{where}: hidden category {hid.get('category')!r}, the batch's is {category!r}")
    ids = [r["id"] for r in rounds]
    pub_rounds = {r.get("id"): r for r in pub.get("rounds") or []}
    if [r.get("id") for r in pub.get("rounds") or []] != ids:
        problems.append(f"{where}: {PUBLIC_NAME} serves its rounds in another order than the batch's positions")
    hid_rounds = hid.get("rounds") if isinstance(hid.get("rounds"), dict) else {}
    if sorted(hid_rounds) != sorted(ids):
        problems.append(f"{where}: {HIDDEN_NAME} holds {len(hid_rounds)} rounds, not the batch's {len(ids)}")
    for r in rounds:
        pr = pub_rounds.get(r["id"]) or {}
        tiles = _tiles(r)
        if [c.get("id") for c in pr.get("clips") or []] != [c["id"] for c in tiles]:
            problems.append(f"{where} #{r['index']}: {PUBLIC_NAME} plays other clips or another tile order")
        elif [c.get("sha256") for c in pr["clips"]] != [c["flac_sha256"] for c in tiles]:
            problems.append(f"{where} #{r['index']}: a tile's FLAC sha256 is not the batch's")
    return pub_rounds, hid_rounds


def _shootout(day: str, job: int, draw: Mapping, batches: Mapping, inbox: Path, index: Mapping[str, tuple],
              problems: list[str]) -> tuple[tuple, tuple, tuple]:
    """(maps, public maps, batch tables) of one shootout job, every cross-check failure appended to `problems`."""
    tag = f"{DAY_NAMES[day]} (job {job})"
    if draw.get("schema") != SHOOTOUT_DRAW_SCHEMA or batches.get("schema") != BATCHES_SCHEMA:
        problems.append(f"{tag}: schemas {draw.get('schema')!r} and {batches.get('schema')!r}, not "
                        f"{SHOOTOUT_DRAW_SCHEMA} and {BATCHES_SCHEMA}")
    if draw.get("stage") != day or batches.get("stage") != day or batches.get("run") != f"shootout-{job}":
        problems.append(f"{tag}: draw stage {draw.get('stage')!r}, batches stage {batches.get('stage')!r} and run "
                        f"{batches.get('run')!r}, not {day} and shootout-{job}")
    maps: Mapping[str, list[Mapping[str, str]]] = draw["maps"]
    drawn = {b["name"]: b for b in draw["batches"]}
    if list(drawn) != list(batches["batches"]):
        problems.append(f"{tag}: the draw holds batches {list(drawn)}, the batches file {list(batches['batches'])}")
    served: dict[str, list[tuple[str, set[str], bool]]] = {}   # stage -> [(slot, options, design?)]: design and anchors
    tables = []
    for name, bb in batches["batches"].items():
        db = drawn.get(name, {"rounds": []})
        where = f"{tag} {name}"
        if (db.get("date"), db.get("category")) != (bb["date"], bb["category"]) or bb["dir"] != f"{bb['date']}/{name}":
            problems.append(f"{where}: the draw's date and category {db.get('date')!r}, {db.get('category')!r}, the "
                            f"batches file's {bb['date']!r}, {bb['category']!r} in {bb['dir']!r}")
        rounds = sorted(bb["rounds"], key=lambda r: r["index"])
        if [r["index"] for r in rounds] != list(range(1, len(rounds) + 1)):
            problems.append(f"{where}: positions {[r['index'] for r in rounds]} are not 1..{len(rounds)}")
        if len(db["rounds"]) != len(rounds):
            problems.append(f"{where}: the draw has {len(db['rounds'])} rounds, the batches file {len(rounds)}")
        pub_rounds, hid_rounds = _check_public_hidden(where, inbox / bb["dir"], name, rounds, bb["category"], problems)
        rows = []
        for dr, r in zip(db["rounds"], rounds):
            at = f"{where} #{r['index']} ({r['slot']})"
            tiles = _tiles(r)
            if [c["pos"] for c in tiles] != list(range(1, len(tiles) + 1)):
                problems.append(f"{at}: tiles {[c['pos'] for c in tiles]} are not 1..{len(tiles)}")
            for k in ("slot", "loop", "kind", "stage", "category"):
                if dr.get(k) != r.get(k):
                    problems.append(f"{at}: the draw's {k} {dr.get(k)!r}, the batches file's {r.get(k)!r}")
            if [(c.get("option"), c.get("what"), c.get("render")) for c in dr["clips"]] != \
                    [(c["option"], c["what"], c["render"]) for c in tiles]:
                problems.append(f"{at}: the draw's tiles are not the batches file's")
            if (pub_rounds.get(r["id"]) or {}).get("bpm") != dr.get("bpm"):
                problems.append(f"{at}: {PUBLIC_NAME} bpm {(pub_rounds.get(r['id']) or {}).get('bpm')!r}, the draw's "
                                f"{dr.get('bpm')!r}")
            hr = hid_rounds.get(r["id"]) or {}
            for hk, rk in (("kind", "kind"), ("of", "of"), ("anchor_clip", "anchor_clip"), ("brief_id", "loop"),
                           ("category", "category")):
                if hr.get(hk) != r.get(rk):
                    problems.append(f"{at}: hidden {hk} {hr.get(hk)!r}, the batches file's {rk} {r.get(rk)!r}")
            hclips = hr.get("clips") if isinstance(hr.get("clips"), dict) else {}
            for c in tiles:
                h = hclips.get(c["id"])
                if not isinstance(h, Mapping):
                    problems.append(f"{at} tile {c['pos']}: not in {HIDDEN_NAME}")
                    continue
                bad = [k for k in CLIP_FIELDS if h.get(k) != c.get(k)]
                bad += [k for k in ("stage", "slot", "loop") if h.get(k) != r.get(k)]
                if bad:
                    problems.append(f"{at} tile {c['pos']}: hidden {bad} differ from the batches file")
            if (r["kind"] == "anchor") != (r["anchor_clip"] is not None):
                problems.append(f"{at}: kind {r['kind']!r} with anchor clip {r['anchor_clip']!r}")
            if (r["kind"] == "repeat") != (r["of"] is not None):
                problems.append(f"{at}: kind {r['kind']!r} with first showing {r['of']!r}")
            broken = {c["pos"] for c in tiles if r["kind"] == "anchor" and c["id"] == r["anchor_clip"]}
            if r["kind"] == "none":
                kind = "design"
            elif r["kind"] == "anchor":
                if len(broken) != 1 or broken != {dr.get("anchor_pos")} or dr.get("imitates") != r["slot"]:
                    problems.append(f"{at}: the anchor's broken tile {sorted(broken)} is not the draw's anchor_pos "
                                    f"{dr.get('anchor_pos')!r} of {dr.get('imitates')!r}")
                kind = f"anchor of {r['slot']}"
            elif r["kind"] == "repeat":
                kind = _repeat_kind(at, r, dr, tiles, index, job, problems)
            else:
                problems.append(f"{at}: kind {r['kind']!r} is none of {sorted(KINDS)}")
                kind = str(r["kind"])
            if r["kind"] in ("none", "anchor"):
                served.setdefault(r["stage"], []).append((r["slot"], {c["option"] for c in tiles}, r["kind"] == "none"))
            rows.append(Row(r["index"], r["slot"], r["loop"], kind,
                            tuple(Tile(c["option"], c["what"], c["pos"] in broken) for c in tiles)))
        tables.append(BatchTable(name, bb["dir"], tuple(rows)))
    return (*_map_rows(tag, maps, served, problems), tuple(tables))


def _repeat_kind(at: str, r: Mapping, dr: Mapping, tiles: Sequence[Mapping], index: Mapping[str, tuple], job: int,
                 problems: list[str]) -> str:
    """`repeat of <slot>, first shown <batch> #<position>`, after checking the first showing."""
    first = index.get(r["of"])
    if r["of"] != dr.get("of") or first is None:
        problems.append(f"{at}: its first showing does not resolve to a round of the jobs' batches")
        return f"repeat of {r['slot']}"
    _day, fjob, fname, fr = first
    if (fname, f"shootout-{fjob}") != (dr.get("of_batch"), dr.get("of_run")) or fjob >= job:
        problems.append(f"{at}: its first showing is {fname} of shootout-{fjob}, the draw says {dr.get('of_batch')!r} "
                        f"of {dr.get('of_run')!r}")
    if (fr["slot"], fr["loop"], fr["kind"]) != (r["slot"], r["loop"], "none"):
        problems.append(f"{at}: its first showing is a {fr['kind']} round of {fr['slot']} on {fr['loop']}")
    if [c["clip"] for c in tiles] != [c["clip"] for c in reversed(_tiles(fr))]:
        problems.append(f"{at}: its clips are not its first showing's in reverse order")
    return f"repeat of {r['slot']}, first shown {fname} #{fr['index']}"


def _slots(slots: Sequence[str]) -> str:
    return ", ".join(sorted(set(slots), key=_slot_key)) or "no round"


def _map_rows(tag: str, maps: Mapping[str, list[Mapping[str, str]]],
              served: Mapping[str, list[tuple[str, set[str], bool]]],
              problems: list[str]) -> tuple[tuple[tuple[str, str, str], ...], tuple[tuple[str, str, str], ...]]:
    """(stage label, symbol, option) for every map of every stage, twice: labelled for the private file and for the
    published section. The cross-check: each design or anchor round's options are among its stage's map values. A
    stage whose maps share symbols labels each map with the slots whose rounds it was drawn for (each such round must
    fit exactly one map): the design and anchor slots in the private file, the design slots alone in the section."""
    out, public = [], []
    for stage, rounds in served.items():
        values = {o for m in maps.get(stage, []) for o in m.values()}
        for slot, opts, _design in rounds:
            if not opts <= values:
                problems.append(f"{tag} {slot}: options {sorted(opts - values)} are in none of {stage}'s maps")
    for stage, group in maps.items():
        symbols = [s for m in group for s in m]
        shared = len(symbols) != len(set(symbols))
        slots: list[list[str]] = [[] for _ in group]
        design: list[list[str]] = [[] for _ in group]
        if shared:
            for slot, opts, is_design in served.get(stage, []):
                fits = [i for i, m in enumerate(group) if opts <= set(m.values())]
                if len(fits) != 1:
                    problems.append(f"{tag} {slot}: its options fit {len(fits)} of {stage}'s maps, not one")
                else:
                    slots[fits[0]].append(slot)
                    if is_design:
                        design[fits[0]].append(slot)
        for i, m in enumerate(group):
            label = f"{stage} ({_slots(slots[i])})" if shared else stage
            public_label = f"{stage} ({_slots(design[i])})" if shared else stage
            out += [(label, sym, opt) for sym, opt in m.items()]
            public += [(public_label, sym, opt) for sym, opt in m.items()]
    return tuple(out), tuple(public)


def is_drafts(key: str) -> bool:
    """Whether a job's tables are a lofi-drafts job's (keep3, or a redo round's), not a shootout day's."""
    return key == "keep3" or key.startswith("keep3_")


def job_name(key: str) -> str:
    """A job's name in the headings: Day 1-3, keep3, keep3 redo<k>."""
    return DAY_NAMES.get(key, key.replace("_", " "))


def _keep3(runs: Path, inbox: Path, job: int, problems: list[str], key: str = "keep3") -> JobTables:
    """A lofi-drafts job's tables (keep3's, or a redo round's under its REDOS key)."""
    summary = runs / f"drafts-{job}" / "summary"
    draw, draw_sha = _json(summary / "draw.json")
    batch, batch_sha = _json(summary / "batch.json")
    tag = f"{job_name(key)} (job {job})"
    try:
        if draw.get("schema") != KEEP3_DRAW_SCHEMA or batch.get("schema") != KEEP3_BATCH_SCHEMA:
            problems.append(f"{tag}: schemas {draw.get('schema')!r} and {batch.get('schema')!r}, not "
                            f"{KEEP3_DRAW_SCHEMA} and {KEEP3_BATCH_SCHEMA}")
        symbols, square = list(draw["symbols"]), [list(r) for r in draw["square"]]
        name = batch["batch"]
        where = f"{tag} {name}"
        if batch["dir"] != f"{batch['date']}/{name}":
            problems.append(f"{where}: dir {batch['dir']!r} is not {batch['date']}/{name}")
        rounds = sorted(batch["rounds"], key=lambda r: r["index"])
        if [r["index"] for r in rounds] != list(range(1, len(rounds) + 1)):
            problems.append(f"{where}: positions {[r['index'] for r in rounds]} are not 1..{len(rounds)}")
        if len(draw["rounds"]) != len(rounds):
            problems.append(f"{where}: the draw has {len(draw['rounds'])} rounds, the batch file {len(rounds)}")
        _pub, hid_rounds = _check_public_hidden(where, inbox / batch["dir"], name, rounds, KEEP3_CATEGORY, problems)
        rows = []
        for dr, r in zip(draw["rounds"], rounds):
            at = f"{where} #{r['index']} ({r['brief']})"
            tiles = _tiles(r)
            personas = [c["persona"] for c in tiles]
            if (dr.get("brief"), dr.get("row"), dr.get("tiles")) != (r["brief"], r["row"], personas):
                problems.append(f"{at}: the draw's brief, row and tiles are not the batch file's")
            if personas != [symbols[i] for i in square[r["row"]]]:
                problems.append(f"{at}: its tiles {personas} are not row {r['row']} of the square")
            hr = hid_rounds.get(r["id"]) or {}
            got = (hr.get("kind"), hr.get("of"), hr.get("anchor_clip"), hr.get("brief_id"), hr.get("category"))
            if got != ("none", None, None, r["brief"], KEEP3_CATEGORY):
                problems.append(f"{at}: hidden kind, of, anchor clip, brief and category {got}")
            hclips = hr.get("clips") if isinstance(hr.get("clips"), dict) else {}
            cells = set()
            for c in tiles:
                h = hclips.get(c["id"])
                if not isinstance(h, Mapping):
                    problems.append(f"{at} tile {c['pos']}: not in {HIDDEN_NAME}")
                    continue
                bad = [hk for hk, bk in KEEP3_FIELDS if h.get(hk) != c.get(bk)]
                if h.get("brief") != r["brief"]:
                    bad.append("brief")
                if bad:
                    problems.append(f"{at} tile {c['pos']}: hidden {bad} differ from the batch file")
                cells.add((h.get("energy"), h.get("mood")))
            if len(cells) != 1:
                problems.append(f"{at}: its clips name {len(cells)} energy and mood cells")
            energy, mood = next(iter(cells)) if len(cells) == 1 else ("?", "?")
            rows.append(Row(r["index"], r["brief"], f"{energy}, {mood}", "design",
                            tuple(Tile(c["persona"], c["draft"]) for c in tiles), r["row"]))
        met, k = draw.get("order_constraint_met"), draw.get("same_cell_neighbours")
        order = ("drawn with no two consecutive rounds in one energy and mood cell" if met is True and k == 0
                 else f"drawn; {k} pairs of consecutive rounds share an energy and mood cell (no drawn order had none)")
        return JobTables(key, job, ((f"runs/drafts-{job}/summary/draw.json", draw_sha),
                                    (f"runs/drafts-{job}/summary/batch.json", batch_sha)),
                         square=tuple(tuple(symbols[i] for i in row) for row in square), order=order,
                         batches=(BatchTable(name, batch["dir"], tuple(rows)),))
    except (KeyError, TypeError, IndexError, AttributeError) as err:
        raise BadInput([f"{tag}: {summary}: a file is not the drafts job's shape ({type(err).__name__}: {err})"]) from None


def collect(runs: Path, inbox: Path, jobs: Mapping[str, int] = JOBS,
            redos: Sequence[Mapping[str, Any]] | None = None) -> list[JobTables]:
    """Every job's tables, cross-checked (module doc): the three shootout days, keep3, then each redo round of
    `redos` (lofi_rl.freeze.REDOS, read now, unless given), whose job is jobs[its key]. TablesError listing every
    problem; BadInput for a missing or unreadable file."""
    runs, inbox = Path(runs), Path(inbox)
    redos = F.REDOS if redos is None else redos
    raw = {}
    for day in DAYS:
        job = int(jobs[day])
        summary = runs / f"shootout-{job}" / "summary"
        raw[day] = (job, *_json(summary / "draw.json"), *_json(summary / "batches.json"))
    problems: list[str] = []
    index: dict[str, tuple] = {}
    for day, (job, _d, _ds, batches, _bs) in raw.items():
        for name, b in (batches.get("batches") or {}).items():
            for r in b.get("rounds") or []:
                if r.get("id") in index:
                    problems.append(f"round id of {name} #{r.get('index')} also names a round of {index[r['id']][2]}")
                index[r.get("id")] = (day, job, name, r)
    out = []
    for day in DAYS:
        job, draw, draw_sha, batches, batches_sha = raw[day]
        try:
            maps, public_maps, tables = _shootout(day, job, draw, batches, inbox, index, problems)
        except (KeyError, TypeError, IndexError, AttributeError) as err:
            raise BadInput([f"{DAY_NAMES[day]} (job {job}): a file is not the shootout job's shape "
                            f"({type(err).__name__}: {err})"]) from None
        out.append(JobTables(day, job, ((f"runs/shootout-{job}/summary/draw.json", draw_sha),
                                        (f"runs/shootout-{job}/summary/batches.json", batches_sha)),
                             maps=maps, batches=tables, public_maps=public_maps))
    out.append(_keep3(runs, inbox, int(jobs["keep3"]), problems))
    for r in redos:
        r = r if isinstance(r, Mapping) else {}
        key = r.get("key") if isinstance(r.get("key"), str) else "keep3_redo?"
        job = jobs.get(key)
        if job is None or job != r.get("job"):
            problems.append(f"{job_name(key)}: REDOS names job {r.get('job')}, the jobs {job}")
            continue
        out.append(_keep3(runs, inbox, int(job), problems, key=key))
    if problems:
        raise TablesError(problems)
    return out


# --------------------------------------------------------------------------------------------- writing
def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("\\", "\\\\").replace("|", "\\|")


def _tile(t: Tile, keep3: bool) -> str:
    what = f"`{_cell(t.what)}`" if keep3 else _cell(t.what)
    return f"`{_cell(t.option)}` ({what})" + (" **broken**" if t.broken else "")


def hidden_checks(tables: Sequence[JobTables]) -> int:
    """How many rounds of the batches are hidden checks (any kind but `design`): the rounds the section leaves out."""
    return sum(1 for t in tables for b in t.batches for r in b.rows if r.kind != "design")


def _sources() -> str:
    return (f"Written by `python -m lofi_rl.freeze.tables` from each job's `summary/draw.json` and "
            f"`summary/batches.json` (keep3 and its redo rounds: `summary/batch.json`), and checked against every "
            f"batch's `{PUBLIC_NAME}` "
            f"and `{HIDDEN_NAME}`. Left out: the 64-bit seeds, the opaque round and clip ids, the real clip ids and "
            f"the audio hashes. Each job's files are named by their sha256.")


def _left_out(n: int) -> str:
    """The section's one line on the rounds it leaves out: how many, and why (module doc), never how they are built."""
    if n == 0:
        return "- Left out: no round; none of these batches held a hidden check."
    rounds = "1 round, a hidden check" if n == 1 else f"{n} rounds, the hidden checks"
    return (f"- Left out: {rounds} (RATING.md, \"Checks you won't see\"). Phase 2's votes and the final A/B use the "
            f"same checks, and the rater is never told which rounds those are. The full tables are kept privately, "
            f"never committed.")


def _job(t: JobTables, depth: str, maps: Sequence[tuple[str, str, str]]) -> list[str]:
    """A job's heading, its files, and its maps (a shootout job) or its Latin square (keep3 and its redo rounds)."""
    L = [f"{depth} Drawn, {job_name(t.key)} (job {t.job})", ""]
    L += [f"- `{path}` sha256 `{sha}`" for path, sha in t.files] + [""]
    if is_drafts(t.key):
        L += ["The persona on each tile of each row of the Latin square:", "",
              "| row | " + " | ".join(f"tile {i + 1}" for i in range(len(t.square[0]))) + " |",
              "|---|" + "---|" * len(t.square[0])]
        L += [f"| {r} | " + " | ".join(f"`{_cell(p)}`" for p in row) + " |" for r, row in enumerate(t.square)]
        return L + ["", f"Round order: {t.order}.", ""]
    L += ["Symbol to option, per stage:", "", "| stage | symbol | option |", "|---|---|---|"]
    return L + [f"| {_cell(stage)} | {_cell(sym)} | `{_cell(opt)}` |" for stage, sym, opt in maps] + [""]


def _table(columns: Sequence[str], rows: Sequence[tuple[list[str], tuple[Tile, ...]]], keep3: bool) -> list[str]:
    """A Markdown table: the `columns`, then one tile column per tile of the widest row; each row its lead cells and
    its tiles, a shorter row's missing tiles left empty."""
    width = max((len(tiles) for _, tiles in rows), default=0)
    L = ["| " + " | ".join([*columns, *(f"tile {i + 1}" for i in range(width))]) + " |",
         "|" + "---|" * (len(columns) + width)]
    for lead, tiles in rows:
        L.append("| " + " | ".join(lead + [_tile(x, keep3) for x in tiles] + [""] * (width - len(tiles))) + " |")
    return L + [""]


def _design_order(r: Row) -> tuple:
    """A design round's place in the section: by slot, never by serving position."""
    return _slot_key(r.slot), r.loop, tuple((x.option, x.what) for x in r.tiles)


def render(tables: Sequence[JobTables], approved: date) -> str:
    """The section for the gate file, ending in one newline: the design rounds only (module doc, "What it
    publishes"), with one line on the hidden checks it leaves out."""
    L = [SECTION, "",
         f"What the batch builders drew at job time, published now that the freeze is approved (SHOOTOUTS.md §2.3; "
         f"his approval line above is dated {approved.isoformat()}). " + _sources(),
         "",
         "- Rounds: each shootout batch's design rounds by slot, and keep3's rounds (and each redo round's) in serving "
         "order. Tiles: left to right as served, each `option` (what it played).",
         _left_out(hidden_checks(tables)),
         ""]
    for t in tables:
        keep3 = is_drafts(t.key)
        L += _job(t, "###", t.public_maps)
        for b in t.batches:
            L += [f"#### {b.name} (`inbox/{b.folder}/`)", ""]
            if keep3:
                L += _table(["#", "brief", "energy, mood", "row"],
                            [([str(r.index), _cell(r.slot), _cell(r.loop), str(r.row)], r.tiles) for r in b.rows], True)
                continue
            design = sorted((r for r in b.rows if r.kind == "design"), key=_design_order)
            L += _table(["slot", "loop"], [([_cell(r.slot), _cell(r.loop)], r.tiles) for r in design], False) \
                if design else ["No design round.", ""]
    while L and L[-1] == "":
        L.pop()
    return "\n".join(L) + "\n"


def render_private(tables: Sequence[JobTables], approved: date) -> str:
    """The private file's Markdown, ending in one newline: every round of every batch in serving order, with its
    position and kind, the broken tile of an anchor marked (module doc, "What the private file holds")."""
    L = [PRIVATE_TITLE, "",
         f"Private: every round of the drawn batches in serving order, the hidden checks included. It is kept under "
         f"`data/`, which git and the sync never carry, and is never printed. `{SECTION}` in gates/GATE_palette.md "
         f"publishes the design rounds only: Phase 2's votes and the final A/B use the same checks, and the rater is "
         f"never told which rounds those are (RATING.md, \"Checks you won't see\"). His approval line is dated "
         f"{approved.isoformat()}. " + _sources(),
         "",
         "- Kind: `design` is a round of its stage. `repeat of <slot>` is that slot's round shown again, tiles "
         "reversed, with where it was first shown. `anchor of <slot>` is that slot's round with one clip replaced by "
         "a sour copy, the tile marked **broken**.",
         "- Tiles: left to right as served, each `option` (what it played).",
         ""]
    for t in tables:
        keep3 = is_drafts(t.key)
        L += _job(t, "##", t.maps)
        for b in t.batches:
            L += [f"### {b.name} (`inbox/{b.folder}/`, {len(b.rows)} rounds)", ""]
            if keep3:
                L += _table(["#", "brief", "energy, mood", "kind", "row"],
                            [([str(r.index), _cell(r.slot), _cell(r.loop), _cell(r.kind), str(r.row)], r.tiles)
                             for r in b.rows], True)
            else:
                L += _table(["#", "slot", "loop", "kind"],
                            [([str(r.index), _cell(r.slot), _cell(r.loop), _cell(r.kind)], r.tiles) for r in b.rows],
                            False)
    while L and L[-1] == "":
        L.pop()
    return "\n".join(L) + "\n"


def sections(text: str, heading: str) -> list[tuple[int, int]]:
    """(heading line, end line) indices of every section whose heading line is `heading` (trailing space aside), found
    through lofi_rl.freeze.gate's own reader (gate.sections: code fences and HTML comments never read), so the
    section goes where the seal's reader sees `## Approval` end: a section runs to the next heading of its level or
    above, or to the end of the file."""
    return G.sections(text, heading)


def insert(text: str, section: str) -> tuple[str, bool]:
    """The gate file's text with `section` right after its `## Approval` section, every other byte kept; and whether
    it changed. A section already there with the same text changes nothing; one with other text refuses."""
    lines = text.splitlines()
    have = sections(text, SECTION)
    if len(have) > 1:
        raise TablesError([f"the gate file holds {len(have)} `{SECTION}` sections"])
    if have:
        first, end = have[0]
        old = "\n".join(lines[first:end]).rstrip("\n") + "\n"
        if old != section:
            raise TablesError([f"the gate file already holds a `{SECTION}` section with other tables; it is never "
                               f"rewritten"])
        return text, False
    found = sections(text, APPROVAL_HEADING)
    if len(found) != 1:
        raise TablesError([f"the gate file has {len(found)} `{APPROVAL_HEADING}` headings, not one"])
    end = found[0][1]
    keep = text.splitlines(keepends=True)
    before, after = "".join(keep[:end]), "".join(keep[end:])
    gap = "" if before.endswith("\n\n") else "\n" if before.endswith("\n") else "\n\n"
    return before + gap + section + ("\n" + after if after else ""), True


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tables.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def private_default() -> Path:
    """Where `--private` writes when it names no file: beside the votes' audit files (SHOOTOUTS.md §8)."""
    return PRIVATE_ROOT / "votes" / "audit" / "drawn_round_tables.md"


def private_path(path: Path | str) -> Path:
    """`path`, resolved (symlinks followed), when it lies inside PRIVATE_ROOT (the repo's `data/`, which git and the
    sync never carry) and its folder exists. BadInput otherwise: the full tables never go anywhere else, and no folder
    is created for them (the votes repo is nested at data/votes/, CLAUDE.md §7)."""
    root, got = PRIVATE_ROOT.resolve(), Path(path).resolve()
    if got == root or not got.is_relative_to(root):
        raise BadInput([f"--private {path}: the full tables go only under {PRIVATE_ROOT}/ (never committed or synced), "
                        f"not to {got}"])
    if not got.parent.is_dir():
        raise BadInput([f"--private {path}: its folder {got.parent} does not exist; it is never created here"])
    if got.exists() and not got.is_file():
        raise BadInput([f"--private {path}: {got} is not a file"])
    return got


def private_changes(path: Path, full: str, write: bool) -> bool:
    """Whether the private file must be written: False when it already holds exactly `full`. TablesError when it
    holds other tables (a record is never rewritten), or, with `write` false (`--write` without `--private`), when it
    does not exist: the full tables are recorded privately before the section is published."""
    try:
        have = path.read_bytes() if path.exists() else None
    except OSError as err:
        raise BadInput([f"the private file {path}: {err}"]) from None
    if have is not None:
        if have != full.encode("utf-8"):
            raise TablesError([f"{path} already holds other full tables; a record is never rewritten"])
        return False
    if not write:
        raise TablesError([f"the full tables are not recorded privately in {path} yet: give --private too, so they "
                           f"exist before the section is published"])
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.freeze.tables", description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=ROOT / "runs", help="the pulled run folders")
    p.add_argument("--inbox", type=Path, default=ROOT / "inbox", help="the pulled batches")
    p.add_argument("--gate", type=Path, default=None, help="the gate file read for his APPROVED line")
    p.add_argument("--write", type=Path, default=None, metavar="GATE",
                   help="add the section to this gate file (the full tables must be in the private file, or --private)")
    p.add_argument("--private", nargs="?", const="", default=None, metavar="PATH",
                   help="write the full tables, hidden checks included, to this file under data/ (default "
                        "data/votes/audit/drawn_round_tables.md); never printed")
    a = p.parse_args(argv)
    if a.write and a.gate and a.write.resolve() != a.gate.resolve():
        print("error: --gate and --write name two files; the section goes into the gate file it is approved by",
              file=sys.stderr)
        return 2
    gate = a.write or a.gate or GATE_PATH
    try:
        text = gate.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        print(f"error: the gate file {gate}: {err}", file=sys.stderr)
        return 2
    try:
        when = approval(text)
    except SealError as err:
        for x in err.problems:
            print(f"refused: {x}", file=sys.stderr)
        print("refused: the drawn tables are published after the freeze, once his APPROVED line is in", file=sys.stderr)
        return 3
    try:
        private = None
        if a.private is not None or a.write is not None:
            private = private_path(Path(a.private) if a.private else private_default())
        tabs = collect(a.runs, a.inbox, JOBS)
        section = render(tabs, when)
        if private is None:
            sys.stdout.write(section)
            return 0
        full = render_private(tabs, when)
        write_private = private_changes(private, full, a.private is not None)
        new, changed = insert(text, section) if a.write is not None else (text, False)
    except BadInput as err:
        for x in err.problems:
            print(f"error: {x}", file=sys.stderr)
        return 2
    except TablesError as err:
        for x in err.problems:
            print(f"problem: {x}", file=sys.stderr)
        return 3
    if write_private:
        atomic_write(private, full.encode("utf-8"))             # as the votes are written: 0600, fsynced
        n = hidden_checks(tabs)
        print(f"wrote the full tables ({n} hidden-check round{'' if n == 1 else 's'} included) to {private} (sha256 "
              f"{hashlib.sha256(full.encode('utf-8')).hexdigest()})")
    else:
        print(f"{private} already holds these full tables; unchanged")
    if a.write is None:
        return 0
    if changed:
        _write(gate, new)
        print(f"wrote `{SECTION}` into {gate} (sha256 {hashlib.sha256(section.encode('utf-8')).hexdigest()} of the "
              f"section)")
    else:
        print(f"{gate} already holds these tables; unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
