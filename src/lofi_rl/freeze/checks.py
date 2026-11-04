"""The checks before the sign-off (docs/DRAFTS.md §7.1; docs/FREEZE.md §5): the kept loops against the frozen palette.

    python -m lofi_rl.freeze.checks report [--palette palettes/lofi_rain_v1.yaml] [--drafts drafts/v1]
        [--gate gates/GATE_palette.md] [--renders KEY=runs/drafts-<job>/summary/renders.json ...] [--keep3 PATH]
        [--verify] [--json OUT]
    python -m lofi_rl.freeze.checks expect --renders keep3=runs/drafts-13656049/summary/renders.json
        [--renders keep3_redo<k>=runs/drafts-<job>/summary/renders.json ...] [--keep3 PATH] [--allow-tossed]
        [--gate gates/GATE_palette.md] [--draft-palette palettes/draft_day2.yaml] [--out drafts/v1/keep3_kept10.json]

**The kept list** is lofi_rl.freeze.kept_now(): keep3's table (KEPT) with each redo round's picks (REDOS) applied in
order, one loop per brief. Each kept loop comes from one round, its *source*: keep3 (`lofi-drafts` 13656049), or the
redo round that picked it, whose folder (drafts/v1/redo<k>/) holds its file. A redo file shares its stem with the draft
of its brief and persona in drafts/v1 (drafts/v1/redo1/d01_horn.ls and drafts/v1/d01_horn.ls), so every kept draft is
read by its file path, never by its stem.

**`expect`** writes `drafts/v1/keep3_kept10.json` (schema lofi-rl/keep3-kept10/2): what each source rendered for the
loops it supplies, copied from that round's summary/renders.json, given as `--renders KEY=PATH` for keep3 and for each
redo round that supplies a kept loop (`--keep3 PATH` is `--renders keep3=PATH`). Each round's job is
lofi_rl.freeze.JOBS[KEY], and its path must name drafts-<that job>. The document:
- `sources`: [{key, path, job, sha256}], keep3 first, then each redo round that supplies a kept loop, in round order;
- `draft_palette` (path and sha256) and `s_star`, the draft palette's S* (`#| draft_levels.S_star`): every bus record
  of every source was computed with it, so each bus's `post_lufs - S*` must give its `lu` exactly;
- `kept`: kept_now() as a list, in brief order; `allow_tossed` and `tossed` (below);
- `loops`, keyed by the kept file's stem (one kept loop per brief, so no two share a stem): its `file` (repo path), its
  `source` (its round's key) and every field schema 1 had: draft_sha256 (which must still be the committed file's),
  the brief, persona, keys and kit, the render palette label, n, pcm_sha256, music_sha256, LUFS, true peak, master gain
  and each bus's record (pre-fader LUFS, effective trim and fader, after the fader, relative to S* and off its target),
  as its source wrote them. Each kept file has exactly one record in its own source's renders.json, under its stem,
  and that record is ok.

`expect` refuses while a kept file is one a sign-off listen tossed (lofi_rl.freeze.tossed_kept; the message names the
brief, the file and the sign-off batch). With `--allow-tossed` it writes the file anyway and records so
(`allow_tossed: true`, and each such loop under `tossed`): the document then still holds those loops' renders, and the
report (i) and the sign-off job still refuse them. The gate file's kept list (keep3's table with each redo round's
table over it) must be kept_now() too. The sign-off job reads these values from its own code snapshot, so a scratch
purge (CLAUDE.md §1) cannot remove them; runs/ is never committed or synced.

**`report`** runs the laptop checks and prints them:
- (a) each kept file, read by its path, is the text its source rendered (its sha256 is keep3_kept10.json's) and
  passes gate, compile, energy and mood under the frozen palette; and `drafts.check.check_folder` with the frozen
  palette passes every draft of every drafts folder (`drafts_folders`): drafts/v1 for every brief, then each redo
  folder in round order (REDOS's, and every drafts/v1/redo<k>/ present) for the briefs that have files there, against
  drafts/v1 and the earlier redo folders (gates G1-G6 in the brief's mood with no repair and no `sax:` line, compile
  with no repair, the energy band, the mood margin, the 5-gram Jaccard);
- (b) prompt v2 gives its 9 references under the palette with strict=True, identical to dev_gu's texts (so the copy
  penalty's references do not move);
- (c) the palette's regex (lofi_rl.grammar) is REGEX_CHARS characters with sha256 REGEX_SHA256, draft_day2's, and every
  draft of every folder fullmatches it;
- (d) with `--verify`: `python -m lofi_rl.freeze check --verify` (sources hashed, presets read, SF2 coverage);
- (e) the bus levels of the kept loops, from keep3_kept10.json: each bus's median off its target relative to S*, its
  per-loop range and whether the median is more than OVER_LU off (SHOOTOUTS.md §3: such a bus goes to Shresth), the keys
  per instrument, the keys against each loop's own sax, and which instruments the kept loops play;
- (f) the laptop loudness and bed dry runs are evidence only (not canonical, LESSONS 38): not run here;
- (g) with `--gate`: `python -m lofi_rl.freeze check --gate` (the picks, the redo rounds and the sign-off listens
  against the gate file), and keep3_kept10.json's kept list against the gate file's (`gate.kept_now_table`: keep3's
  table with each redo round's table over it, in order);
- (h) keep3_kept10.json's schema, and with `--renders` for every source it names (when the files are there) a fresh
  `expect` from them, with the `allow_tossed` it records, must be the committed file byte for byte;
- (i) the kept list against the constants: REDOS and SIGNOFFS keep their own rules (gate.records_problems);
  keep3_kept10.json keeps kept_now(), each loop from the round that picked it, its sources are the rounds that supply
  them, each with its JOBS job, and the tossed loops it records are the ones tossed now; and no kept file is one a
  sign-off listen tossed. Since b1103-signoff (2026-11-03) its four tossed loops are still kept, so (i) fails on
  exactly them until a redo round's picks replace them.
On a sealed file (`frozen:` set) both freeze checks add `--sealed`, so `frozen` must be the date of the APPROVED line
(docs/FREEZE.md §3.6 (i)): the line in the --gate file for (g), in gates/GATE_palette.md (the check's default) for (d).
Exit 0 when every check passes, 3 when one fails, 2 for bad input (and for a refused `expect`). Sound claims: none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lofi_rl import freeze as F
from lofi_rl import prompt_v2
from lofi_rl.drafts import briefs as B
from lofi_rl.drafts.check import Draft, Report, check_folder, check_one, read_drafts
from lofi_rl.drafts.palette import DRAFT_PALETTE, DraftPalette, DraftPaletteError
from lofi_rl.drafts.palette import load as load_draft
from lofi_rl.drafts.personas import DRAFT_RE
from lofi_rl.freeze import FROZEN
from lofi_rl.freeze import gate as G
from lofi_rl.grammar import loopscript_regex
from lofi_rl.palette import ROOT, Palette, PaletteError, load_palette

EXPECT_SCHEMA = "lofi-rl/keep3-kept10/2"
REPORT_SCHEMA = "lofi-rl/freeze-checks/2"
EXPECT_NAME = "keep3_kept10.json"                 # in the drafts folder: drafts/v1/keep3_kept10.json
DRAFTS = ROOT / "drafts" / "v1"
GATE = ROOT / "gates" / "GATE_palette.md"
DEV_PALETTE = ROOT / "palettes" / "dev_gu.yaml"   # the palette D-020's copy penalty was computed under
# The regex run 1 and the Phase-1 gate decode under: draft_day2's, since the regex reads only the vocabulary
# (grammar.py). Computed this session from palettes/draft_day2.yaml: 3,408 characters, this sha256.
REGEX_CHARS = 3408
REGEX_SHA256 = "5c081828ac50718edc4db82c2a855ea52210debb375280ba01e8442de60f8558"
OVER_LU = 1.0                 # SHOOTOUTS.md §3: a bus more than 1.0 LU off its target goes to Shresth before the freeze
BUSES = ("sax", "keys", "bass", "drums")          # render/mix.py's summing order
LOOP_KEYS = ("file", "draft_sha256", "brief_id", "brief", "persona", "keys", "kit", "n", "bpm", "pcm_sha256",
             "music_sha256", "lufs", "true_peak_dbtp", "master_gain_db")
BUS_KEYS = ("lufs", "trim_db", "fader_db", "post_lufs", "lu", "target_lu", "off_lu")
SOURCE_KEYS = ("key", "path", "job", "sha256")
REDO_DIR = re.compile(r"redo([1-9]\d*)")          # a redo folder's name (docs/DRAFTS.md §6)


class ChecksError(ValueError):
    """Bad input, or a check that cannot even start: `errors` lists every problem found."""

    def __init__(self, errors: Sequence[str] | str):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("; ".join(self.errors))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path | str) -> str:
    return sha256_bytes(Path(path).read_bytes())


def label(path: Path | str, root: Path = ROOT) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------------- the kept list
def _gate_rows(path: Path | str, table: Mapping[str, str]) -> list[str]:
    return [f"{path}: brief {b}'s row keeps {f}, a draft of another brief" for b, f in table.items()
            if not Path(f).name.startswith(f"{b}_")]


def gate_kept(path: Path | str = GATE) -> list[str]:
    """The gate file's keep3 table as a list of the kept files (repo-relative, in table order), read by
    lofi_rl.freeze.gate (the table under its `## Kept drafts (keep3)` heading). ChecksError when the gate reader
    refuses the file, or when a row keeps a draft of another brief."""
    try:
        table = G.kept(G.read(path))
    except G.GateError as err:
        raise ChecksError(f"{path}: {err}") from err
    bad = _gate_rows(path, table)
    if bad:
        raise ChecksError(bad)
    return list(table.values())


def gate_kept_now(path: Path | str = GATE, *, redos: Sequence[Mapping] | None = None) -> dict[str, str]:
    """The gate file's kept list: keep3's table with each redo round's table (REDOS, unless given) over it, in order
    (lofi_rl.freeze.gate.kept_now_table); {brief: file}. ChecksError as gate_kept."""
    try:
        table = G.kept_now_table(G.read(path), redos=redos)
    except G.GateError as err:
        raise ChecksError(f"{path}: {err}") from err
    bad = _gate_rows(path, table)
    if bad:
        raise ChecksError(bad)
    return table


def tossed_problems(tossed: Sequence[tuple[str, str, Sequence[str]]] | None = None) -> list[str]:
    """One line per kept file a sign-off listen tossed (lofi_rl.freeze.tossed_kept, unless given), naming the brief,
    the file and the sign-off batch."""
    tossed = F.tossed_kept() if tossed is None else tossed
    return [f"{b}: {f} is still kept, but Shresth tossed it at the sign-off listen {' and '.join(batches)} "
            f"(lofi_rl.freeze.SIGNOFFS); a redo round's pick replaces it first (docs/DRAFTS.md §6)"
            for b, f, batches in tossed]


def round_order(redos: Sequence[Mapping] | None = None) -> list[str]:
    """The rounds' keys in order: keep3, then each REDOS round."""
    return ["keep3"] + [r["key"] for r in (F.REDOS if redos is None else redos)]


# --------------------------------------------------------------------------------------------- the drafts folders
@dataclass(frozen=True)
class Folder:
    """A drafts folder the checks run over: drafts/v1, or a redo folder (docs/DRAFTS.md §6)."""

    path: Path
    scope: tuple[str, ...] | None       # None: every brief of the briefs file (drafts/v1)
    against: tuple[Path, ...]           # the earlier folders its drafts are compared with (the 5-gram Jaccard)
    key: str | None                     # its round: keep3, a REDOS key, or None (a redo folder with no picks yet)


def drafts_folders(drafts: Path | str, *, redos: Sequence[Mapping] | None = None,
                   root: Path = ROOT) -> tuple[list[Folder], list[str]]:
    """(the folders, the problems): `drafts` (drafts/v1, every brief), then its redo folders in k order: each REDOS
    round's folder (which must be `<drafts>/redo<k>` and exist) and every `<drafts>/redo*/` present (named redo<k>).
    A redo folder's scope is the briefs of the briefs file that have a file there (a file of another name is the
    validator's to refuse), and its drafts are compared with `drafts` and the earlier redo folders."""
    drafts = Path(drafts)
    redos = F.REDOS if redos is None else redos
    problems: list[str] = []
    found: dict[int, Path] = {}
    keys: dict[int, str] = {}
    for p in sorted(drafts.glob("redo*")):
        if not p.is_dir():
            continue
        m = REDO_DIR.fullmatch(p.name)
        if not m:
            problems.append(f"{label(p, root)}: not a redo folder's name (redo<k>, docs/DRAFTS.md §6)")
            continue
        found[int(m.group(1))] = p
    for r in redos:
        p = root / r["folder"]
        m = REDO_DIR.fullmatch(p.name)
        if not m or p.resolve().parent != drafts.resolve():
            problems.append(f"{r['key']}: its folder {r['folder']} is not a redo folder of {label(drafts, root)}")
        elif not p.is_dir():
            problems.append(f"{r['key']}: its folder {r['folder']} is not there")
        else:
            keys[int(m.group(1))] = r["key"]
    out = [Folder(drafts, None, (), "keep3")]
    earlier = [drafts]
    for k in sorted(found):
        p = found[k]
        try:
            briefs = B.load(B.briefs_file_for(p))
        except B.BriefsError as err:
            problems.append(f"{label(p, root)}: {err}")
            continue
        scope = tuple(sorted({m["brief"] for f in p.glob("*.ls") if (m := DRAFT_RE.fullmatch(f.name))
                              and m["brief"] in briefs}))
        if not scope:
            problems.append(f"{label(p, root)}: no <brief>_<persona>.ls draft of a brief of its briefs file")
            continue
        out.append(Folder(p, scope, tuple(earlier), keys.get(k)))
        earlier.append(p)
    return out, problems


def draft_files(folders: Sequence[Folder]) -> list[Path]:
    """Every *.ls file of the folders, folder by folder."""
    return [f for fo in folders for f in sorted(fo.path.glob("*.ls"))]


def render_id(path: Path | str, drafts: Path | str) -> str:
    """A draft's render id across the folders: its stem in `drafts` (d01_horn), and `<folder>_<stem>` in a redo
    folder (redo1_d01_horn), so the ids of one job never repeat."""
    p = Path(path)
    return p.stem if p.resolve().parent == Path(drafts).resolve() else f"{p.parent.name}_{p.stem}"


# --------------------------------------------------------------------------------------------- keep3_kept10.json
@dataclass(frozen=True)
class Source:
    """One round's summary/renders.json as `expect` reads it."""

    key: str                            # keep3 or a REDOS key
    path: str                           # its repo-relative path (as `make pull` puts it)
    job: int                            # JOBS[key]
    sha256: str                         # of its bytes
    doc: Mapping                        # the document

    def record(self) -> dict:
        return {"key": self.key, "path": self.path, "job": int(self.job), "sha256": self.sha256}


def read_source(key: str, path: Path | str, *, job: int, root: Path = ROOT) -> Source:
    """A round's summary/renders.json at `path`, which must name its job's run folder drafts-<job>."""
    p = Path(path)
    named = [int(m.group(1)) for part in p.parts if (m := re.fullmatch(r"drafts-(\d+)", part))]
    if named != [int(job)]:
        raise ChecksError(f"--renders {key}={p}: the path names the runs {named}, not drafts-{job} "
                          f"(lofi_rl.freeze.JOBS[{key!r}])")
    try:
        data = p.read_bytes()
        doc = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ChecksError(f"{p}: {err}") from err
    return Source(key, label(p, root), int(job), sha256_bytes(data), doc)


def source_loops(src: Source, files: Sequence[str], *, s_star: float,
                 root: Path = ROOT) -> tuple[dict[str, dict], list[str]]:
    """({stem: loop record}, problems) for each of `files` (repo paths) from its source's renders document: exactly one
    record of that file, under the file's stem, and ok; the committed text the one it rendered; every field there;
    every bus's lu computed with `s_star`."""
    recs = src.doc.get("renders") if isinstance(src.doc, Mapping) else None
    if not isinstance(recs, Mapping) or not recs:
        return {}, [f"{src.key}'s renders document {src.path} has no `renders` mapping"]
    by_file: dict[str | None, list[str]] = {}
    for rid, r in recs.items():
        by_file.setdefault(r.get("file") if isinstance(r, Mapping) else None, []).append(rid)
    errors, loops = [], {}
    for f in files:
        rids = by_file.get(f, [])
        if len(rids) != 1:
            errors.append(f"{f}: {len(rids)} records in {src.key}'s renders.json, not one")
            continue
        rid, r = rids[0], recs[rids[0]]
        if rid != Path(f).stem:
            errors.append(f"{f}: {src.key} rendered it as {rid}, not under its stem {Path(f).stem}")
            continue
        if r.get("status") != "ok":
            errors.append(f"{rid}: {src.key}'s render is {r.get('status')}, not ok")
            continue
        try:
            text_sha = file_sha256(root / f)
        except OSError as err:
            errors.append(f"{f}: {err}")
            continue
        if text_sha != r.get("draft_sha256"):
            errors.append(f"{f}: the committed text (sha256 {text_sha[:12]}...) is not the one {src.key} rendered "
                          f"({str(r.get('draft_sha256'))[:12]}...)")
        missing = [k for k in LOOP_KEYS if k not in r] + [f"buses.{b}" for b in BUSES if b not in (r.get("buses") or {})]
        if missing or not isinstance((r.get("palette") or {}).get("label"), str):
            errors.append(f"{rid}: {src.key}'s record lacks {missing or ['palette.label']}")
            continue
        buses = {}
        for b in BUSES:
            got = r["buses"][b]
            if any(k not in got for k in BUS_KEYS):
                errors.append(f"{rid}: {src.key}'s {b} bus record lacks {[k for k in BUS_KEYS if k not in got]}")
                continue
            if got["post_lufs"] - s_star != got["lu"]:
                errors.append(f"{rid}: the {b} bus's lu {got['lu']} was not computed with S* {s_star} "
                              f"(post_lufs - S* is {got['post_lufs'] - s_star})")
            buses[b] = {k: got[k] for k in BUS_KEYS}
        loops[rid] = {**{k: r[k] for k in LOOP_KEYS}, "palette": r["palette"]["label"], "buses": buses,
                      "source": src.key}
    return loops, errors


def expectations(sources: Sequence[Source], *, kept: Mapping[str, tuple[str, str]], draft: DraftPalette,
                 tossed: Sequence[tuple[str, str, Sequence[str]]] = (), allow_tossed: bool = False,
                 root: Path = ROOT) -> dict:
    """keep3_kept10.json's document (module doc). `sources`: the rounds' renders documents in round order; `kept`:
    {brief: (kept file, the key of the round that picked it)} (lofi_rl.freeze.kept_rounds); `tossed`: the kept files a
    sign-off listen tossed (lofi_rl.freeze.tossed_kept), refused unless `allow_tossed`. ChecksError listing every
    problem: a tossed loop, a source with no renders or a renders file with no kept loop, a kept file with no record
    or two, a record not ok, a committed file whose text is not the one its source rendered, or a bus whose lu was not
    computed with the draft palette's S*."""
    s_star = draft.levels.get("S_star")
    if isinstance(s_star, bool) or not isinstance(s_star, (int, float)):
        raise ChecksError(f"{draft.path}: the levels block has no S_star")
    errors = [] if allow_tossed else tossed_problems(tossed)
    by_key: dict[str, Source] = {}
    for s in sources:
        if s.key in by_key:
            errors.append(f"two renders documents for {s.key}")
        by_key[s.key] = s
    supplies: dict[str, list[str]] = {}
    for _b, (f, key) in kept.items():
        supplies.setdefault(key, []).append(f)
    errors += [f"no renders.json for {key}, which supplies {files}" for key, files in supplies.items()
               if key not in by_key]
    errors += [f"{s.key} ({s.path}) supplies no kept loop now; leave its --renders out" for s in sources
               if s.key not in supplies]
    stems = [Path(f).stem for f, _ in kept.values()]
    if len(set(stems)) != len(stems):
        errors.append(f"two kept files share a stem: {sorted(s for s in stems if stems.count(s) > 1)}")
    loops: dict[str, dict] = {}
    for s in sources:
        got, errs = source_loops(s, supplies.get(s.key, []), s_star=s_star, root=root)
        loops.update(got)
        errors += errs
    if errors:
        raise ChecksError(errors)
    return {"schema": EXPECT_SCHEMA, "tool": "python -m lofi_rl.freeze.checks expect",
            "sources": [s.record() for s in sources],
            "draft_palette": {"path": label(draft.path, root), "sha256": draft.sha256}, "s_star": s_star,
            "kept": [f for f, _ in kept.values()], "allow_tossed": bool(allow_tossed),
            "tossed": [{"brief": b, "file": f, "signoffs": list(batches)} for b, f, batches in tossed],
            "loops": {Path(f).stem: loops[Path(f).stem] for f, _ in kept.values()}}


def expect_text(doc: Mapping) -> str:
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def expect_from_files(renders: Mapping[str, Path | str], *, draft: DraftPalette, allow_tossed: bool = False,
                      root: Path = ROOT) -> dict:
    """`expectations` of the rounds' renders files ({round key: path}) for the constants' kept list (kept_rounds,
    tossed_kept), each round's job JOBS[key], the sources in round order."""
    order = round_order()
    unknown = [k for k in renders if k not in order]
    if unknown:
        raise ChecksError([f"--renders {k}: not keep3 or a round of lofi_rl.freeze.REDOS {order[1:]}" for k in unknown])
    missing_jobs = [k for k in renders if k not in F.JOBS]
    if missing_jobs:
        raise ChecksError([f"--renders {k}: lofi_rl.freeze.JOBS has no job for it" for k in missing_jobs])
    sources = [read_source(k, renders[k], job=F.JOBS[k], root=root) for k in order if k in renders]
    return expectations(sources, kept=F.kept_rounds(), tossed=F.tossed_kept(), draft=draft,
                        allow_tossed=allow_tossed, root=root)


def load_expectations(path: Path | str) -> dict:
    """keep3_kept10.json, its structure checked: the schema; its sources (key, path, job, sha256, each key once); S*;
    the kept files, each with one loop record (with all four buses) under its stem, from one of its sources; each
    source supplying a loop; and the tossed record. ChecksError listing every problem."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ChecksError(f"{p}: {err}") from err
    if not isinstance(doc, dict) or doc.get("schema") != EXPECT_SCHEMA:
        raise ChecksError(f"{p}: not a {EXPECT_SCHEMA} document")
    errors = []
    sources = doc.get("sources")
    if not isinstance(sources, list) or not sources or not all(
            isinstance(s, dict) and sorted(s) == sorted(SOURCE_KEYS) and isinstance(s["key"], str)
            and isinstance(s["path"], str) and isinstance(s["job"], int) and not isinstance(s["job"], bool)
            and re.fullmatch(r"[0-9a-f]{64}", str(s["sha256"])) for s in sources):
        raise ChecksError(f"{p}: sources needs a list of {{{', '.join(SOURCE_KEYS)}}} records")
    keys = [s["key"] for s in sources]
    if len(set(keys)) != len(keys):
        errors.append(f"{p}: a source is named twice in {keys}")
    if isinstance(doc.get("s_star"), bool) or not isinstance(doc.get("s_star"), (int, float)):
        errors.append(f"{p}: s_star must be a number")
    kept, loops = doc.get("kept"), doc.get("loops")
    if not isinstance(kept, list) or not kept or not isinstance(loops, dict):
        raise ChecksError(errors + [f"{p}: needs a kept list and a loops mapping"])
    files = sorted(str(e.get("file")) for e in loops.values() if isinstance(e, dict))
    if files != sorted(str(f) for f in kept) or len(set(kept)) != len(kept):
        errors.append(f"{p}: its loops are not exactly its kept files")
    for rid, e in loops.items():
        if not isinstance(e, dict) or any(k not in e for k in LOOP_KEYS + ("palette", "buses", "source")):
            errors.append(f"{p}: loop {rid} lacks a field")
        elif Path(e["file"]).stem != rid:
            errors.append(f"{p}: loop {rid} names the file {e['file']}")
        elif e["source"] not in keys:
            errors.append(f"{p}: loop {rid} comes from {e['source']}, not one of its sources {keys}")
        elif sorted(e["buses"]) != sorted(BUSES) or any(any(k not in v for k in BUS_KEYS) for v in e["buses"].values()):
            errors.append(f"{p}: loop {rid} needs each bus's {list(BUS_KEYS)}")
    used = {e.get("source") for e in loops.values() if isinstance(e, dict)}
    errors += [f"{p}: source {k} supplies no loop" for k in keys if k not in used]
    tossed = doc.get("tossed")
    if not isinstance(doc.get("allow_tossed"), bool) or not isinstance(tossed, list) or not all(
            isinstance(t, dict) and sorted(t) == ["brief", "file", "signoffs"] and t["file"] in kept for t in tossed):
        errors.append(f"{p}: needs allow_tossed and a tossed list of {{brief, file, signoffs}} of its kept files")
    elif tossed and not doc["allow_tossed"]:
        errors.append(f"{p}: it records tossed loops without allow_tossed")
    if errors:
        raise ChecksError(errors)
    return doc


def kept_problems(expect: Mapping, gate: Path | str, *, redos: Sequence[Mapping] | None = None) -> list[str]:
    """keep3_kept10.json's kept list against the gate file's: keep3's table with each redo round's table over it, in
    order (gate_kept_now)."""
    redos = F.REDOS if redos is None else redos
    try:
        table = gate_kept_now(gate, redos=redos)
    except ChecksError as err:
        return err.errors
    if list(expect["kept"]) != list(table.values()):
        over = " with its redo rounds' tables over it" if redos else ""
        return [f"keep3_kept10.json keeps {expect['kept']}, but the gate file's keep3 table{over} keeps "
                f"{list(table.values())}"]
    return []


def constants_problems(expect: Mapping) -> list[str]:
    """keep3_kept10.json against the constants (lofi_rl.freeze, read now): REDOS and SIGNOFFS keep their own rules
    (gate.records_problems); it keeps kept_now(), each loop from the round that picked it; its sources are the rounds
    that supply the kept loops, in round order, each with its JOBS job; and the tossed loops it records are the ones
    tossed_kept() gives now. The tossed loops themselves are tossed_problems'."""
    out = G.records_problems()
    if out:
        return out
    rounds = F.kept_rounds()
    want = [f for f, _ in rounds.values()]
    if list(expect["kept"]) != want:
        out.append(f"keep3_kept10.json keeps {expect['kept']}, but lofi_rl.freeze.kept_now() keeps {want}: run "
                   f"`python -m lofi_rl.freeze.checks expect` again")
    loops = {e["file"]: e for e in expect["loops"].values()}
    out += [f"{f}: keep3_kept10.json has it from {loops[f]['source']}, but {key} picked it"
            for f, key in rounds.values() if f in loops and loops[f]["source"] != key]
    supply = [k for k in round_order() if any(key == k for _, key in rounds.values())]
    have = [s["key"] for s in expect["sources"]]
    if have != supply:
        out.append(f"keep3_kept10.json's sources are {have}; the rounds that supply the kept loops are {supply}")
    out += [f"keep3_kept10.json's source {s['key']} is job {s['job']}, not lofi_rl.freeze.JOBS[{s['key']!r}] "
            f"({F.JOBS.get(s['key'])})" for s in expect["sources"] if F.JOBS.get(s["key"]) != s["job"]]
    now = sorted((b, f) for b, f, _ in F.tossed_kept())
    recorded = sorted((t["brief"], t["file"]) for t in expect["tossed"])
    if recorded != now:
        out.append(f"keep3_kept10.json records the tossed loops {recorded}; the sign-off listens toss {now} of the "
                   f"kept list now: run `checks expect` again")
    return out


# --------------------------------------------------------------------------------------------- the checks
def kept_drafts(expect: Mapping, root: Path = ROOT) -> dict[str, Draft]:
    """The kept drafts by loop id, each read from its own file path (keep3_kept10.json's `file`) and named as the
    validator names it (drafts.check.read_drafts over that file's folder with that folder's briefs), so a redo file is
    never mistaken for the drafts/v1 file of the same stem."""
    folders: dict[Path, dict[Path, Draft]] = {}
    out = {}
    for rid, e in expect["loops"].items():
        p = Path(root / e["file"]).resolve()
        if p.parent not in folders:
            try:
                briefs = B.load(B.briefs_file_for(p.parent))
                folders[p.parent] = {d.path.resolve(): d for d in read_drafts(p.parent, briefs)}
            except (B.BriefsError, OSError):
                folders[p.parent] = {}
        d = folders[p.parent].get(p)
        if d is not None:
            out[rid] = d
    return out


def vocabulary_check(palette: Palette, drafts: Path, expect: Mapping, *, with_report: bool = False,
                     root: Path = ROOT):
    """(a): each kept file, read by its path, is its source's text and passes check_one under the palette; every
    folder of `drafts_folders` passes check_folder under it. Returns the record with its `problems`, and with
    `with_report` also [(folder, check_folder's Report)] (their drafts parsed under the palette)."""
    problems: list[str] = []
    kept = kept_drafts(expect, root)
    rows = {}
    for rid, e in expect["loops"].items():
        d = kept.get(rid)
        if d is None:
            problems.append(f"{e['file']}: not there (a kept draft is read by its own path)")
            continue
        sha = sha256_bytes(d.text.encode("utf-8"))
        check_one(d, palette)
        rows[rid] = {"file": e["file"], "source": e["source"], "sha256": sha, "rendered": sha == e["draft_sha256"],
                     "problems": dict(d.problems), "E": d.info.get("E"), "mood_margin": d.info.get("mood_margin"),
                     "repairs": d.info.get("repairs")}
        if sha != e["draft_sha256"]:
            problems.append(f"{e['file']}: sha256 {sha[:12]}... is not the text {e['source']} rendered "
                            f"({e['draft_sha256'][:12]}..., keep3_kept10.json)")
        problems += [f"{e['file']}: {c}: {why}" for c, why in d.problems.items()]
    folders, bad = drafts_folders(drafts, root=root)
    problems += bad
    reports: list[tuple[Folder, Report]] = []
    per = []
    for fo in folders:
        rep = check_folder(fo.path, scope=None if fo.scope is None else list(fo.scope), against=list(fo.against),
                           palette=palette)
        reports.append((fo, rep))
        rec = {"folder": label(fo.path, root), "round": fo.key, "scope": rep.scope,
               "against": [label(x, root) for x in fo.against], "drafts": len(rep.drafts),
               "ok": sum(d.ok for d in rep.drafts), "missing": [label(fo.path / m, root) for m in rep.missing],
               "failing": {label(d.path, root): d.problems for d in rep.drafts if not d.ok}}
        per.append(rec)
        if not rep.ok:
            problems.append(f"check_folder({rec['folder']}, palette={palette.id}): {rec['ok']} of {rec['drafts']} "
                            f"drafts pass, missing {rec['missing']}")
    folder = {"drafts": sum(r["drafts"] for r in per), "ok": sum(r["ok"] for r in per),
              "missing": [m for r in per for m in r["missing"]],
              "references": reports[0][1].n_refs if reports else 0,
              "failing": {k: v for r in per for k, v in r["failing"].items()}}
    doc = {"kept": rows, "folder": folder, "folders": per, "problems": problems}
    return (doc, reports) if with_report else doc


def reference_check(palette: Palette, dev: Palette | None) -> dict:
    """(b): prompt v2's references under the palette, strict, and whether they are dev_gu's texts."""
    problems: list[str] = []
    try:
        refs = prompt_v2.reference_texts(palette, strict=True)
    except Exception as err:            # PromptError: the palette refuses an example
        return {"count": 0, "problems": [f"prompt v2 refuses an example under {palette.id}: {err}"]}
    if len(refs) != len(prompt_v2.EXAMPLES):
        problems.append(f"prompt v2 gives {len(refs)} references, not {len(prompt_v2.EXAMPLES)}")
    same = None
    if dev is not None:
        theirs = prompt_v2.reference_texts(dev, strict=True)
        same = dict(refs) == dict(theirs)
        if not same:
            problems.append(f"prompt v2's references under {palette.id} differ from {dev.id}'s (the copy penalty's)")
    return {"count": len(refs), "of": len(prompt_v2.EXAMPLES), "identical_to": None if dev is None else dev.id,
            "identical": same, "problems": problems}


def regex_check(palette: Palette, files: Sequence[Path], kept: Sequence[str], *, root: Path = ROOT) -> dict:
    """(c): the regex's length and sha256, and which of `files` (every draft of every folder) and of the kept files
    (repo paths) fullmatch it."""
    rx = loopscript_regex(palette)
    sha = sha256_bytes(rx.encode("utf-8"))
    pat = re.compile(rx)
    match = {Path(f).resolve() for f in files if pat.fullmatch(Path(f).read_text(encoding="utf-8"))}
    problems = []
    if (len(rx), sha) != (REGEX_CHARS, REGEX_SHA256):
        problems.append(f"the regex is {len(rx)} characters with sha256 {sha}, not {REGEX_CHARS} with {REGEX_SHA256}")
    if len(match) != len(files):
        problems.append(f"{len(files) - len(match)} of {len(files)} drafts do not fullmatch the regex: "
                        f"{sorted(label(f, root) for f in files if Path(f).resolve() not in match)}")
    return {"chars": len(rx), "sha256": sha, "drafts": len(files), "fullmatch": len(match), "kept": len(kept),
            "kept_fullmatch": sum((root / k).resolve() in match for k in kept), "problems": problems}


def freeze_check(args: Sequence[str], root: Path = ROOT) -> tuple[int, str]:
    """`python -m lofi_rl.freeze check <args>` in a fresh interpreter (the generator's own check, docs/FREEZE.md §3.6):
    (exit code, its output)."""
    import os
    src = str(root / "src")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    p = subprocess.run([sys.executable, "-m", "lofi_rl.freeze", "check", *map(str, args)], capture_output=True,
                       text=True, env=env, cwd=root)
    return p.returncode, (p.stdout + p.stderr).strip()


def bus_table(expect: Mapping) -> dict:
    """(e): per bus the median off target, its per-loop range and whether the median is over OVER_LU; the keys per
    instrument; the keys against each loop's own sax; the instruments the kept loops play."""
    loops = expect["loops"]
    buses = {}
    for b in BUSES:
        offs = {rid: e["buses"][b]["off_lu"] for rid, e in loops.items()}
        med = statistics.median(offs.values())
        buses[b] = {"median_off_lu": med, "min_off_lu": min(offs.values()), "max_off_lu": max(offs.values()),
                    "over": abs(med) > OVER_LU, "loops": offs}
    by_keys: dict[str, dict[str, float]] = {}
    for rid, e in loops.items():
        by_keys.setdefault(e["keys"], {})[rid] = e["buses"]["keys"]["off_lu"]
    own = {}
    for rid, e in loops.items():
        k, s = e["buses"]["keys"], e["buses"]["sax"]
        own[rid] = (k["lu"] - s["lu"]) - (k["target_lu"] - s["target_lu"])
    uses: dict[str, dict[str, list[str]]] = {"keys": {}, "kits": {}}
    for rid, e in loops.items():
        uses["keys"].setdefault(e["keys"], []).append(rid)
        uses["kits"].setdefault(e["kit"], []).append(rid)
    return {"over_lu": OVER_LU, "buses": buses,
            "keys_by_instrument": {i: {"loops": v, "median": statistics.median(v.values())} for i, v in by_keys.items()},
            "keys_vs_own_sax": {"loops": own, "median": statistics.median(own.values())},
            "uses": {k: {n: sorted(v) for n, v in sorted(u.items())} for k, u in uses.items()}}


def renders_map(renders: Sequence[str] = (), keep3: Path | str | None = None) -> dict[str, Path]:
    """{round key: path} from `--renders KEY=PATH` arguments and the `--keep3 PATH` alias. ChecksError for a malformed
    argument or a round given twice."""
    out: dict[str, Path] = {}
    errors = []
    pairs = [("keep3", str(keep3))] if keep3 is not None else []
    for arg in renders:
        key, sep, path = str(arg).partition("=")
        if not sep or not key or not path:
            errors.append(f"--renders {arg}: not KEY=PATH")
            continue
        pairs.append((key, path))
    for key, path in pairs:
        if key in out:
            errors.append(f"--renders {key} is given twice (--keep3 PATH is --renders keep3=PATH)")
        out[key] = Path(path)
    if errors:
        raise ChecksError(errors)
    return out


def report(palette_path: Path, *, drafts: Path = DRAFTS, gate: Path | None = None, keep3: Path | None = None,
           renders: Mapping[str, Path] | None = None, verify: bool = False, draft_palette: Path = DRAFT_PALETTE,
           dev_palette: Path | None = DEV_PALETTE, root: Path = ROOT, runner=None) -> dict:
    """Every check of the module doc, as one document with `problems` (empty when all pass). `renders`: {round key:
    its renders.json} for (h), `keep3` its alias for keep3. `runner` is the freeze check's runner (default
    `freeze_check`). ChecksError for bad input."""
    runner = runner or freeze_check
    given = dict(renders or {})
    if keep3 is not None:
        if "keep3" in given:
            raise ChecksError("--keep3 and --renders keep3= name keep3's renders twice")
        given["keep3"] = Path(keep3)
    try:
        pal = load_palette(palette_path, verify=False, root=root)
        dev = None if dev_palette is None else load_palette(dev_palette, verify=False, root=root)
    except (PaletteError, OSError) as err:
        raise ChecksError(f"{err}") from err
    expect_path = drafts / EXPECT_NAME
    expect = load_expectations(expect_path)
    doc: dict = {"schema": REPORT_SCHEMA, "palette": {"path": label(palette_path, root), "id": pal.id,
                                                       "sha256": pal.sha256, "frozen": pal.frozen},
                 "expect": {"path": label(expect_path, root), "sha256": file_sha256(expect_path),
                            "sources": expect["sources"]}}
    doc["a"] = vocabulary_check(pal, drafts, expect, root=root)
    doc["b"] = reference_check(pal, dev)
    folders, _ = drafts_folders(drafts, root=root)
    doc["c"] = regex_check(pal, draft_files(folders), expect["kept"], root=root)
    # Without --sealed the freeze check refuses any `frozen` date, so a sealed file is checked as sealed (module doc).
    sealed = ["--sealed"] if pal.frozen is not None else []
    if verify:
        cmd = " ".join(["freeze check --verify", *sealed])
        rc, out = runner(["--palette", palette_path, "--verify", *sealed], root)
        doc["d"] = {"cmd": f"python -m lofi_rl.{cmd}", "exit": rc, "output": out,
                    "problems": [] if rc == 0 else [f"{cmd} exited {rc}: {out[-2000:]}"]}
    else:
        doc["d"] = {"cmd": None, "problems": [], "note": "not run: pass --verify (it needs the fetched assets)"}
    # A bus over OVER_LU is a finding that goes to Shresth (SHOOTOUTS.md §3; the freeze spec's Q1), never a failure.
    doc["e"] = {**bus_table(expect), "problems": []}
    doc["f"] = {"note": "the laptop loudness and bed dry runs are evidence only (not canonical, LESSONS 38); not run "
                        "here", "problems": []}
    g: dict = {"problems": []}
    if gate is not None:
        cmd = " ".join(["freeze check --gate", *sealed])
        rc, out = runner(["--palette", palette_path, "--gate", gate, *sealed], root)
        g.update(cmd=f"python -m lofi_rl.{cmd}", exit=rc, output=out)
        if rc != 0:
            g["problems"].append(f"{cmd} exited {rc}: {out[-2000:]}")
        kp = kept_problems(expect, gate)
        g.update(kept_table_matches=not kp)
        g["problems"] += kp
    else:
        g["note"] = "not run: pass --gate gates/GATE_palette.md"
    doc["g"] = g
    h: dict = {"schema": EXPECT_SCHEMA, "problems": []}
    need = [s["key"] for s in expect["sources"]]
    if given:
        missing = [k for k in need if k not in given]
        absent = [str(given[k]) for k in need if k in given and not Path(given[k]).is_file()]
        if missing:
            h["note"] = f"no --renders for {missing}: the committed file stands"
        elif absent:
            h["note"] = f"{absent} not here: the committed file stands"
        else:
            h["rebuilt_from"] = {k: label(given[k], root) for k in need}
            try:
                fresh = expect_text(expect_from_files({k: given[k] for k in need}, draft=load_draft(draft_palette),
                                                      allow_tossed=expect["allow_tossed"], root=root))
                h["equal"] = fresh == expect_path.read_text(encoding="utf-8")
                if not h["equal"]:
                    h["problems"].append(f"{label(expect_path, root)} is not what `expect` writes from "
                                         f"{list(h['rebuilt_from'].values())}")
            except ChecksError as err:
                h["equal"] = False
                h["problems"] += [f"`expect` from {list(h['rebuilt_from'].values())} refuses: {x}" for x in err.errors]
    doc["h"] = h
    tossed = F.tossed_kept()
    doc["i"] = {"kept_now": F.kept_now(), "sources": need,
                "signoffs": [{"batch": s["batch"], "job": s["job"], "tossed": list(s["tossed"])} for s in F.SIGNOFFS],
                "tossed_kept": [{"brief": b, "file": f, "signoffs": list(bs)} for b, f, bs in tossed],
                "problems": constants_problems(expect) + tossed_problems(tossed)}
    doc["problems"] = [x for k in "abcdefghi" for x in doc[k]["problems"]]
    doc["ok"] = not doc["problems"]
    return doc


def _signed(x: float, nd: int = 3) -> str:
    return f"{x:+.{nd}f}"


def report_text(doc: Mapping) -> str:
    """The report as plain lines, the bus levels as a table."""
    p, a, b, c, e = doc["palette"], doc["a"], doc["b"], doc["c"], doc["e"]
    n_kept = len(a["kept"])
    sources = ", ".join(f"{s['key']} {s['path']}" for s in doc["expect"]["sources"])
    lines = [f"palette {p['path']} ({p['id']}, sha256 {p['sha256']}, frozen {p['frozen']})",
             f"(a) the kept loops: {sum(r['rendered'] for r in a['kept'].values())} of {n_kept} files are the texts "
             f"their sources rendered, {sum(not r['problems'] for r in a['kept'].values())} of {n_kept} pass gate, "
             f"compile, energy and mood; check_folder with this palette: {a['folder']['ok']} of "
             f"{a['folder']['drafts']} drafts pass in {len(a['folders'])} folders "
             f"({', '.join(r['folder'] for r in a['folders'])}), {len(a['folder']['missing'])} missing",
             f"(b) prompt v2: {b['count']} of {b.get('of', 9)} references under strict=True"
             + (f", identical to {b['identical_to']}'s: {b['identical']}" if b.get("identical_to") else ""),
             f"(c) regex: {c['chars']} characters, sha256 {c['sha256']}; the kept loops fullmatch {c['kept_fullmatch']} "
             f"of {c['kept']}, every draft {c['fullmatch']} of {c['drafts']}",
             "(d) SF2 coverage: " + (f"{doc['d']['cmd']}: exit {doc['d']['exit']}" if doc["d"].get("cmd")
                                     else doc["d"]["note"]),
             f"(e) bus levels of the kept loops (their sources' renders: {sources}), relative to S* and the bus "
             f"targets:"]
    rows = [("bus", "median off target (LU)", "per-loop range", f"over {e['over_lu']:.1f}?")]
    for bus, r in e["buses"].items():
        rows.append((bus, _signed(r["median_off_lu"]), f"{_signed(r['min_off_lu'])} ... {_signed(r['max_off_lu'])}",
                     "yes" if r["over"] else "no"))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines += ["    " + "  ".join(x.ljust(w) for x, w in zip(r, widths)).rstrip() for r in rows]
    lines.append("    keys per instrument:")
    for inst, r in e["keys_by_instrument"].items():
        loops = ", ".join(f"{rid} {_signed(v)}" for rid, v in r["loops"].items())
        lines.append(f"      {inst}: {loops}; median {_signed(r['median'])}")
    lines.append(f"    keys against each loop's own sax (keys lu - sax lu - their targets' difference): median "
                 f"{_signed(e['keys_vs_own_sax']['median'])}")
    lines.append("    the kept loops play: " + "; ".join(
        f"{kind} " + ", ".join(f"{n} in {len(v)} ({', '.join(v)})" for n, v in u.items()) for kind, u in e["uses"].items()))
    lines.append(f"(f) {doc['f']['note']}")
    g = doc["g"]
    lines.append("(g) picks: " + (f"{g['cmd']}: exit {g['exit']}; the gate file's kept list is keep3_kept10.json's: "
                                  f"{g['kept_table_matches']}" if g.get("cmd") else g["note"]))
    h = doc["h"]
    lines.append(f"(h) {doc['expect']['path']} (sha256 {doc['expect']['sha256'][:12]}...): schema {h['schema']}"
                 + (f"; rebuilt from {', '.join(h['rebuilt_from'].values())}: equal {h['equal']}" if "equal" in h else
                    f"; {h['note']}" if "note" in h else ""))
    i = doc["i"]
    by_round = {k: sum(r["source"] == k for r in doc["a"]["kept"].values()) for k in i["sources"]}
    listens = "; ".join(f"{s['batch']} (job {s['job']}) tossed {len(s['tossed'])}" for s in i["signoffs"]) or "none"
    lines.append(f"(i) the kept list: {len(i['kept_now'])} kept now ("
                 + ", ".join(f"{k} {n}" for k, n in by_round.items()) + f"); sign-off listens: {listens}; kept and "
                 f"tossed: {len(i['tossed_kept'])}")
    lines += [f"problem: {x}" for x in doc["problems"]]
    lines.append(("ok" if doc["ok"] else f"{len(doc['problems'])} problem(s)") + f": {p['path']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- CLI
def _repo(p: Path | None) -> Path | None:
    return None if p is None else (p if p.is_absolute() else ROOT / p)


def expect_main(a: argparse.Namespace) -> int:
    """`expect` (module doc): every renders file given, the constants' records and the gate file's kept list checked
    first."""
    renders = {k: _repo(v) for k, v in renders_map(a.renders, a.keep3).items()}
    if not renders:
        raise ChecksError("give --renders keep3=runs/drafts-<job>/summary/renders.json (and one --renders per redo "
                          "round that supplies a kept loop)")
    bad = G.records_problems()
    if bad:
        raise ChecksError(bad)
    table, now = gate_kept_now(_repo(a.gate)), F.kept_now()
    if table != now:
        raise ChecksError(f"the gate file's kept list {table} is not lofi_rl.freeze.kept_now() {now}")
    doc = expect_from_files(renders, draft=load_draft(_repo(a.draft_palette)), allow_tossed=a.allow_tossed)
    text = expect_text(doc)
    out = _repo(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    src = "; ".join(f"{s['key']} {s['path']} (job {s['job']}, sha256 {s['sha256']})" for s in doc["sources"])
    print(f"wrote {label(out)} (sha256 {sha256_bytes(text.encode('utf-8'))}): {len(doc['kept'])} kept loops from {src}")
    if doc["tossed"]:
        print(f"--allow-tossed: {len(doc['tossed'])} of them tossed at a sign-off listen, recorded under `tossed`: "
              + ", ".join(f"{t['brief']} {t['file']} ({', '.join(t['signoffs'])})" for t in doc["tossed"]))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.freeze.checks", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="the checks before the sign-off")
    r.add_argument("--palette", type=Path, default=None, help="default: the frozen palette, palettes/lofi_rain_v1.yaml")
    r.add_argument("--drafts", type=Path, default=DRAFTS)
    r.add_argument("--gate", type=Path, default=None)
    r.add_argument("--renders", action="append", default=[], metavar="KEY=PATH",
                   help="a round's summary/renders.json (keep3, keep3_redo<k>), to rebuild the expectations")
    r.add_argument("--keep3", type=Path, default=None, help="--renders keep3=PATH")
    r.add_argument("--verify", action="store_true", help="also run the freeze check with --verify (fetched assets)")
    r.add_argument("--draft-palette", type=Path, default=DRAFT_PALETTE)
    r.add_argument("--json", type=Path, default=None)
    x = sub.add_parser("expect", help="write keep3_kept10.json from the rounds' summary/renders.json")
    x.add_argument("--renders", action="append", default=[], metavar="KEY=PATH",
                   help="keep3=PATH, and keep3_redo<k>=PATH for each redo round that supplies a kept loop")
    x.add_argument("--keep3", type=Path, default=None, help="--renders keep3=PATH")
    x.add_argument("--allow-tossed", action="store_true",
                   help="write it although a kept loop was tossed at a sign-off listen (recorded in the file)")
    x.add_argument("--gate", type=Path, default=GATE)
    x.add_argument("--draft-palette", type=Path, default=DRAFT_PALETTE)
    x.add_argument("--out", type=Path, default=DRAFTS / EXPECT_NAME)
    a = p.parse_args(argv)
    try:
        if a.cmd == "expect":
            return expect_main(a)
        renders = {k: _repo(v) for k, v in renders_map(a.renders, a.keep3).items()}
        doc = report(_repo(a.palette) or Path(FROZEN), drafts=_repo(a.drafts), gate=_repo(a.gate), renders=renders,
                     verify=a.verify, draft_palette=_repo(a.draft_palette))
    except (ChecksError, DraftPaletteError, B.BriefsError, OSError) as err:
        for x in (err.errors if isinstance(err, ChecksError) else [err]):
            print(f"error: {x}", file=sys.stderr)
        return 2
    print(report_text(doc))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if doc["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
