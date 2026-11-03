"""The checks before the sign-off (docs/DRAFTS.md §7; the freeze spec §4.1): the kept 10 against the frozen palette.

    python -m lofi_rl.freeze.checks report [--palette palettes/lofi_rain_v1.yaml] [--drafts drafts/v1]
        [--gate gates/GATE_palette.md] [--keep3 runs/drafts-13656049/summary/renders.json] [--verify] [--json OUT]
    python -m lofi_rl.freeze.checks expect --keep3 runs/drafts-13656049/summary/renders.json
        [--gate gates/GATE_palette.md] [--draft-palette palettes/draft_day2.yaml] [--job 13656049]
        [--out drafts/v1/keep3_kept10.json]

**`expect`** writes `drafts/v1/keep3_kept10.json` (schema lofi-rl/keep3-kept10/1): what keep3 (`lofi-drafts` 13656049)
rendered for each loop Shresth kept, copied from that job's summary/renders.json (K3R), with K3R's path, job and
sha256. The kept list is the gate file's keep3 table (`## Kept drafts (keep3)`, read by lofi_rl.freeze.gate: by its
heading and its Brief and Kept columns, never by line number). Per loop: the file and its draft_sha256 (which must still be the committed file's), the
brief, persona, keys and kit, the render palette label, n, pcm_sha256, music_sha256, LUFS, true peak, master gain and
each bus's record as K3R wrote it (pre-fader LUFS, effective trim and fader, after the fader, relative to S* and off its
target). S* is the draft palette's (`#| draft_levels.S_star`, the one keep3's records were computed with: every
bus's `post_lufs - S*` must give its `lu` exactly). The sign-off job reads these values from its own code snapshot, so
a scratch purge (CLAUDE.md §1) cannot remove them; runs/ is never committed or synced.

**`report`** runs §4.1's checks on the laptop and prints them:
- (a) each kept file is the text keep3 rendered (its sha256 is keep3_kept10.json's), and `drafts.check.check_folder`
  over the drafts folder with the frozen palette passes every draft (gates G1-G6 in the brief's mood with no repair and
  no `sax:` line, compile with no repair, the energy band, the mood margin, the 5-gram Jaccard);
- (b) prompt v2 gives its 9 references under the palette with strict=True, identical to dev_gu's texts (so the copy
  penalty's references do not move);
- (c) the palette's regex (lofi_rl.grammar) is REGEX_CHARS characters with sha256 REGEX_SHA256, draft_day2's, and every
  draft fullmatches it;
- (d) with `--verify`: `python -m lofi_rl.freeze check --verify` (sources hashed, presets read, SF2 coverage);
- (e) the bus levels of the kept 10, from keep3_kept10.json: each bus's median off its target relative to S*, its
  per-loop range and whether the median is more than OVER_LU off (SHOOTOUTS.md §3: such a bus goes to Shresth), the keys
  per instrument, the keys against each loop's own sax, and which instruments the kept 10 play;
- (f) the laptop loudness and bed dry runs are evidence only (not canonical, LESSONS 38): not run here;
- (g) with `--gate`: `python -m lofi_rl.freeze check --gate` (the picks against the gate file), and keep3_kept10.json's
  kept list against the gate file's keep3 table;
- (h) keep3_kept10.json's schema, and with `--keep3` (when the file is there) a fresh `expect` from it must be the
  committed file byte for byte.
On a sealed file (`frozen:` set) both freeze checks add `--sealed`, so `frozen` must be the date of the APPROVED line
(docs/FREEZE.md §3.6 (i)): the line in the --gate file for (g), in gates/GATE_palette.md (the check's default) for (d).
Exit 0 when every check passes, 3 when one fails, 2 for bad input. Sound claims: none.
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
from pathlib import Path

from lofi_rl import prompt_v2
from lofi_rl.drafts import briefs as B
from lofi_rl.drafts.check import Draft, check_folder, check_one, read_drafts
from lofi_rl.drafts.palette import DRAFT_PALETTE, DraftPalette, DraftPaletteError
from lofi_rl.drafts.palette import load as load_draft
from lofi_rl.freeze import FROZEN, JOBS
from lofi_rl.freeze import gate as G
from lofi_rl.grammar import loopscript_regex
from lofi_rl.palette import ROOT, Palette, PaletteError, load_palette

EXPECT_SCHEMA = "lofi-rl/keep3-kept10/1"
REPORT_SCHEMA = "lofi-rl/freeze-checks/1"
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


# --------------------------------------------------------------------------------------------- the gate file
def gate_kept(path: Path | str = GATE) -> list[str]:
    """The gate file's keep3 table as a list of the kept files (repo-relative, in table order), read by
    lofi_rl.freeze.gate (the table under its `## Kept drafts (keep3)` heading). ChecksError when the gate reader
    refuses the file, or when a row keeps a draft of another brief."""
    try:
        table = G.kept(G.read(path))
    except G.GateError as err:
        raise ChecksError(f"{path}: {err}") from err
    bad = [f"{path}: keep3 row {b} keeps {f}, a draft of another brief" for b, f in table.items()
           if not Path(f).name.startswith(f"{b}_")]
    if bad:
        raise ChecksError(bad)
    return list(table.values())


# --------------------------------------------------------------------------------------------- keep3_kept10.json
def expectations(renders: Mapping, *, renders_sha256: str, renders_path: str, job: int, kept: Sequence[str],
                 draft: DraftPalette, root: Path = ROOT) -> dict:
    """keep3_kept10.json's document (module doc) from keep3's summary/renders.json document `renders`. ChecksError
    listing every problem: a kept file with no record or two, a record not ok, a committed file whose text is not the
    one keep3 rendered, or a bus whose lu was not computed with the draft palette's S*."""
    recs = renders.get("renders") if isinstance(renders, Mapping) else None
    if not isinstance(recs, Mapping) or not recs:
        raise ChecksError("the keep3 renders document has no `renders` mapping")
    m = re.search(r"drafts-(\d+)", str(renders_path))
    if m and int(m.group(1)) != int(job):
        raise ChecksError(f"{renders_path} is not job {job}'s renders.json")
    s_star = draft.levels.get("S_star")
    if isinstance(s_star, bool) or not isinstance(s_star, (int, float)):
        raise ChecksError(f"{draft.path}: the levels block has no S_star")
    by_file: dict[str, list[str]] = {}
    for rid, r in recs.items():
        by_file.setdefault(r.get("file"), []).append(rid)
    errors, loops = [], {}
    for f in kept:
        rids = by_file.get(f, [])
        if len(rids) != 1:
            errors.append(f"{f}: {len(rids)} keep3 records, not one")
            continue
        rid, r = rids[0], recs[rids[0]]
        if r.get("status") != "ok":
            errors.append(f"{rid}: keep3's render is {r.get('status')}, not ok")
            continue
        try:
            text_sha = file_sha256(root / f)
        except OSError as err:
            errors.append(f"{f}: {err}")
            continue
        if text_sha != r.get("draft_sha256"):
            errors.append(f"{f}: the committed text (sha256 {text_sha[:12]}...) is not the one keep3 rendered "
                          f"({str(r.get('draft_sha256'))[:12]}...)")
        missing = [k for k in LOOP_KEYS if k not in r] + [f"buses.{b}" for b in BUSES if b not in (r.get("buses") or {})]
        if missing or not isinstance((r.get("palette") or {}).get("label"), str):
            errors.append(f"{rid}: keep3's record lacks {missing or ['palette.label']}")
            continue
        buses = {}
        for b in BUSES:
            got = r["buses"][b]
            if any(k not in got for k in BUS_KEYS):
                errors.append(f"{rid}: keep3's {b} bus record lacks {[k for k in BUS_KEYS if k not in got]}")
                continue
            if got["post_lufs"] - s_star != got["lu"]:
                errors.append(f"{rid}: the {b} bus's lu {got['lu']} was not computed with S* {s_star} "
                              f"(post_lufs - S* is {got['post_lufs'] - s_star})")
            buses[b] = {k: got[k] for k in BUS_KEYS}
        loops[rid] = {**{k: r[k] for k in LOOP_KEYS}, "palette": r["palette"]["label"], "buses": buses}
    if errors:
        raise ChecksError(errors)
    return {"schema": EXPECT_SCHEMA, "tool": "python -m lofi_rl.freeze.checks expect",
            "keep3": {"path": renders_path, "job": int(job), "sha256": renders_sha256},
            "draft_palette": {"path": label(draft.path, root), "sha256": draft.sha256}, "s_star": s_star,
            "kept": list(kept), "loops": loops}


def expect_text(doc: Mapping) -> str:
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def expect_from_file(keep3: Path | str, *, job: int, kept: Sequence[str], draft: DraftPalette,
                     root: Path = ROOT) -> dict:
    """`expectations` of keep3's summary/renders.json at `keep3`."""
    p = Path(keep3)
    try:
        data = p.read_bytes()
        doc = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ChecksError(f"{p}: {err}") from err
    return expectations(doc, renders_sha256=sha256_bytes(data), renders_path=label(p, root), job=job, kept=kept,
                        draft=draft, root=root)


def load_expectations(path: Path | str) -> dict:
    """keep3_kept10.json, its structure checked: the schema, keep3's path, job and sha256, S*, the kept files and one
    loop record (with all four buses) per kept file. ChecksError listing every problem."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ChecksError(f"{p}: {err}") from err
    errors = []
    if not isinstance(doc, dict) or doc.get("schema") != EXPECT_SCHEMA:
        raise ChecksError(f"{p}: not a {EXPECT_SCHEMA} document")
    k3 = doc.get("keep3") or {}
    if not (isinstance(k3.get("path"), str) and isinstance(k3.get("job"), int)
            and re.fullmatch(r"[0-9a-f]{64}", str(k3.get("sha256")))):
        errors.append(f"{p}: keep3 needs its path, job and sha256")
    if isinstance(doc.get("s_star"), bool) or not isinstance(doc.get("s_star"), (int, float)):
        errors.append(f"{p}: s_star must be a number")
    kept, loops = doc.get("kept"), doc.get("loops")
    if not isinstance(kept, list) or not kept or not isinstance(loops, dict):
        raise ChecksError(errors + [f"{p}: needs a kept list and a loops mapping"])
    if sorted(e.get("file") for e in loops.values() if isinstance(e, dict)) != sorted(kept):
        errors.append(f"{p}: its loops are not exactly its kept files")
    for rid, e in loops.items():
        if not isinstance(e, dict) or any(k not in e for k in LOOP_KEYS + ("palette", "buses")):
            errors.append(f"{p}: loop {rid} lacks a field")
        elif Path(e["file"]).stem != rid:
            errors.append(f"{p}: loop {rid} names the file {e['file']}")
        elif sorted(e["buses"]) != sorted(BUSES) or any(any(k not in v for k in BUS_KEYS) for v in e["buses"].values()):
            errors.append(f"{p}: loop {rid} needs each bus's {list(BUS_KEYS)}")
    if errors:
        raise ChecksError(errors)
    return doc


def kept_problems(expect: Mapping, gate: Path | str) -> list[str]:
    """keep3_kept10.json's kept list against the gate file's keep3 table."""
    try:
        table = gate_kept(gate)
    except ChecksError as err:
        return err.errors
    if list(expect["kept"]) != table:
        return [f"keep3_kept10.json keeps {expect['kept']}, but the gate file's keep3 table keeps {table}"]
    return []


# --------------------------------------------------------------------------------------------- the checks
def kept_drafts(drafts: Path, expect: Mapping) -> dict[str, Draft]:
    """The kept drafts, read and named as the validator reads them (drafts.check.read_drafts), by render id."""
    briefs = B.load(B.briefs_file_for(drafts))
    every = {d.path.stem: d for d in read_drafts(drafts, briefs)}
    return {rid: every[rid] for rid in expect["loops"] if rid in every}


def vocabulary_check(palette: Palette, drafts: Path, expect: Mapping, *, with_report: bool = False):
    """(a): each kept file is keep3's text and passes check_one under the palette; every draft of the folder passes
    check_folder under it. Returns the record with its `problems`, and with `with_report` also check_folder's Report
    (its drafts parsed under the palette)."""
    problems: list[str] = []
    kept = kept_drafts(drafts, expect)
    rows = {}
    for rid, e in expect["loops"].items():
        d = kept.get(rid)
        if d is None:
            problems.append(f"{e['file']}: not in {drafts}")
            continue
        sha = sha256_bytes(d.text.encode("utf-8"))
        check_one(d, palette)
        rows[rid] = {"file": e["file"], "sha256": sha, "keep3": sha == e["draft_sha256"],
                     "problems": dict(d.problems), "E": d.info.get("E"), "mood_margin": d.info.get("mood_margin"),
                     "repairs": d.info.get("repairs")}
        if sha != e["draft_sha256"]:
            problems.append(f"{e['file']}: sha256 {sha[:12]}... is not the text keep3 rendered "
                            f"({e['draft_sha256'][:12]}..., keep3_kept10.json)")
        problems += [f"{e['file']}: {c}: {why}" for c, why in d.problems.items()]
    rep = check_folder(drafts, palette=palette)
    folder = {"drafts": len(rep.drafts), "ok": sum(d.ok for d in rep.drafts), "missing": rep.missing,
              "references": rep.n_refs, "failing": {d.file: d.problems for d in rep.drafts if not d.ok}}
    if not rep.ok:
        problems.append(f"check_folder({label(drafts)}, palette={palette.id}): {folder['ok']} of {folder['drafts']} "
                        f"drafts pass, missing {rep.missing}")
    doc = {"kept": rows, "folder": folder, "problems": problems}
    return (doc, rep) if with_report else doc


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


def regex_check(palette: Palette, drafts: Path, kept: Sequence[str]) -> dict:
    """(c): the regex's length and sha256, and which drafts fullmatch it."""
    rx = loopscript_regex(palette)
    sha = sha256_bytes(rx.encode("utf-8"))
    pat = re.compile(rx)
    files = sorted(drafts.glob("*.ls"))
    match = {f.stem for f in files if pat.fullmatch(f.read_text(encoding="utf-8"))}
    kept_ids = [Path(f).stem for f in kept]
    problems = []
    if (len(rx), sha) != (REGEX_CHARS, REGEX_SHA256):
        problems.append(f"the regex is {len(rx)} characters with sha256 {sha}, not {REGEX_CHARS} with {REGEX_SHA256}")
    if len(match) != len(files):
        problems.append(f"{len(files) - len(match)} of {len(files)} drafts do not fullmatch the regex: "
                        f"{sorted(f.stem for f in files if f.stem not in match)}")
    return {"chars": len(rx), "sha256": sha, "drafts": len(files), "fullmatch": len(match),
            "kept": len(kept_ids), "kept_fullmatch": sum(k in match for k in kept_ids), "problems": problems}


def freeze_check(args: Sequence[str], root: Path = ROOT) -> tuple[int, str]:
    """`python -m lofi_rl.freeze check <args>` in a fresh interpreter (the generator's own check, the freeze spec §3.4):
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


def report(palette_path: Path, *, drafts: Path = DRAFTS, gate: Path | None = None, keep3: Path | None = None,
           verify: bool = False, draft_palette: Path = DRAFT_PALETTE, dev_palette: Path | None = DEV_PALETTE,
           root: Path = ROOT, runner=None) -> dict:
    """Every check of the module doc, as one document with `problems` (empty when all pass). `runner` is the freeze
    check's runner (default `freeze_check`). ChecksError for bad input."""
    runner = runner or freeze_check
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
                            "keep3": expect["keep3"]}}
    doc["a"] = vocabulary_check(pal, drafts, expect)
    doc["b"] = reference_check(pal, dev)
    doc["c"] = regex_check(pal, drafts, expect["kept"])
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
    if keep3 is not None and Path(keep3).is_file():
        fresh = expect_text(expect_from_file(keep3, job=expect["keep3"]["job"], kept=expect["kept"],
                                             draft=load_draft(draft_palette), root=root))
        h["rebuilt_from"] = label(keep3, root)
        h["equal"] = fresh == expect_path.read_text(encoding="utf-8")
        if not h["equal"]:
            h["problems"].append(f"{label(expect_path, root)} is not what `expect` writes from {label(keep3, root)}")
    elif keep3 is not None:
        h["note"] = f"{keep3} is not here: the committed file stands"
    doc["h"] = h
    doc["problems"] = [x for k in "abcdefgh" for x in doc[k]["problems"]]
    doc["ok"] = not doc["problems"]
    return doc


def _signed(x: float, nd: int = 3) -> str:
    return f"{x:+.{nd}f}"


def report_text(doc: Mapping) -> str:
    """The report as plain lines, the bus levels as §4.1's table."""
    p, a, b, c, e = doc["palette"], doc["a"], doc["b"], doc["c"], doc["e"]
    n_kept = len(a["kept"])
    lines = [f"palette {p['path']} ({p['id']}, sha256 {p['sha256']}, frozen {p['frozen']})",
             f"(a) the kept loops: {sum(r['keep3'] for r in a['kept'].values())} of {n_kept} files are the texts keep3 "
             f"rendered, {sum(not r['problems'] for r in a['kept'].values())} of {n_kept} pass gate, compile, energy "
             f"and mood; check_folder with this palette: {a['folder']['ok']} of {a['folder']['drafts']} drafts pass, "
             f"{len(a['folder']['missing'])} missing",
             f"(b) prompt v2: {b['count']} of {b.get('of', 9)} references under strict=True"
             + (f", identical to {b['identical_to']}'s: {b['identical']}" if b.get("identical_to") else ""),
             f"(c) regex: {c['chars']} characters, sha256 {c['sha256']}; the kept loops fullmatch {c['kept_fullmatch']} "
             f"of {c['kept']}, every draft {c['fullmatch']} of {c['drafts']}",
             "(d) SF2 coverage: " + (f"{doc['d']['cmd']}: exit {doc['d']['exit']}" if doc["d"].get("cmd")
                                     else doc["d"]["note"]),
             f"(e) bus levels of the kept loops (keep3's renders, {doc['expect']['keep3']['path']}), relative to S* and "
             "the bus targets:"]
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
    lines.append("(g) picks: " + (f"{g['cmd']}: exit {g['exit']}; the gate file's keep3 table is keep3_kept10.json's "
                                  f"kept list: {g['kept_table_matches']}" if g.get("cmd") else g["note"]))
    h = doc["h"]
    lines.append(f"(h) {doc['expect']['path']} (sha256 {doc['expect']['sha256'][:12]}...): schema {h['schema']}"
                 + (f"; rebuilt from {h['rebuilt_from']}: equal {h['equal']}" if "equal" in h else
                    f"; {h['note']}" if "note" in h else ""))
    lines += [f"problem: {x}" for x in doc["problems"]]
    lines.append(("ok" if doc["ok"] else f"{len(doc['problems'])} problem(s)") + f": {p['path']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- CLI
def _repo(p: Path | None) -> Path | None:
    return None if p is None else (p if p.is_absolute() else ROOT / p)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.freeze.checks", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="the checks before the sign-off (§4.1)")
    r.add_argument("--palette", type=Path, default=None, help="default: the frozen palette, palettes/lofi_rain_v1.yaml")
    r.add_argument("--drafts", type=Path, default=DRAFTS)
    r.add_argument("--gate", type=Path, default=None)
    r.add_argument("--keep3", type=Path, default=None, help="keep3's summary/renders.json, to rebuild the expectations")
    r.add_argument("--verify", action="store_true", help="also run the freeze check with --verify (fetched assets)")
    r.add_argument("--draft-palette", type=Path, default=DRAFT_PALETTE)
    r.add_argument("--json", type=Path, default=None)
    x = sub.add_parser("expect", help="write keep3_kept10.json from keep3's summary/renders.json")
    x.add_argument("--keep3", type=Path, required=True)
    x.add_argument("--gate", type=Path, default=GATE)
    x.add_argument("--draft-palette", type=Path, default=DRAFT_PALETTE)
    x.add_argument("--job", type=int, default=None, help="keep3's job id (default: lofi_rl.freeze.JOBS['keep3'])")
    x.add_argument("--out", type=Path, default=DRAFTS / EXPECT_NAME)
    a = p.parse_args(argv)
    try:
        if a.cmd == "expect":
            job = a.job if a.job is not None else int(JOBS["keep3"])
            doc = expect_from_file(_repo(a.keep3), job=job, kept=gate_kept(_repo(a.gate)),
                                   draft=load_draft(_repo(a.draft_palette)))
            text = expect_text(doc)
            out = _repo(a.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"wrote {label(out)} (sha256 {sha256_bytes(text.encode('utf-8'))}): {len(doc['kept'])} kept loops of "
                  f"{doc['keep3']['path']} (job {job}, sha256 {doc['keep3']['sha256']})")
            return 0
        doc = report(_repo(a.palette) or Path(FROZEN), drafts=_repo(a.drafts), gate=_repo(a.gate), keep3=_repo(a.keep3),
                     verify=a.verify, draft_palette=_repo(a.draft_palette))
    except (ChecksError, DraftPaletteError, B.BriefsError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(report_text(doc))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if doc["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
