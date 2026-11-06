"""Job B, the sign-off render of the frozen palette (docs/DRAFTS.md §7; the freeze spec §4.3): the `--signoff` mode of
the `lofi-drafts` job. `jobs/drafts.sbatch` runs

    python -m lofi_rl.drafts.job --run $REMOTE_SCR/runs/drafts-<job> --inbox $REMOTE_SCR/inbox --jobs 8 \
        --drafts drafts/v1 --signoff palettes/lofi_rain_v1.yaml

from `make submit JOB=jobs/drafts.sbatch ARGS='--drafts drafts/v1 --signoff palettes/lofi_rain_v1.yaml'`, and
lofi_rl.drafts.job hands the run to `run` here. Optional: `--date` (the inbox's date folder, default today, Pacific),
`--name` (default b<MMDD>-signoff, the k-th sign-off batch b<MMDD>-signoff-<k>), `--cache`, `--runs-root` (the folder
holding earlier jobs' runs, read for cross-checks only; default --run's parent, $REMOTE_SCR/runs), `--dry-run`, and
`--seed` (tests only).

**Inputs** come from the job's code snapshot: the frozen palette, the drafts folder (drafts/v1) with its briefs.yaml,
its redo folders (drafts/v1/redo<k>/) and keep3_kept10.json (`python -m lofi_rl.freeze.checks expect`),
gates/GATE_palette.md, shootouts/plan.yaml and shootouts/rain_excerpts.json, the draft palette (whose `#| ` levels
keep3 and the redo rounds played) and prompt v2's references (code); the sources come from $LOFI_ASSETS. Scratch is
read only for cross-checks, so a purge cannot stop the job.

**The kept loops** are lofi_rl.freeze.kept_now() (keep3's picks with each redo round's over them), each from its own
source round (keep3_kept10.json's `sources`). Each is read by its file path, never by its stem: a redo file shares its
stem with a drafts/v1 file. **Render ids** are unique across the job: a drafts/v1 draft's id is its stem (d02_chords),
a redo folder's draft's id carries its folder (redo1_d01_horn, lofi_rl.freeze.checks.render_id), so every kept
render maps to its one kept loop, and that loop to its one source record.

**Stages**, each one's exit code and wall time in summary/stages.json; the first that fails ends the job, and nothing
reaches the inbox:
1. `check` (summary/check.json): the laptop checks again in the snapshot (lofi_rl.freeze.checks): keep3_kept10.json's
   schema, its kept list against the constants (kept_now(), each loop's source, the sources' jobs) and against the
   gate file's (keep3's table with each redo round's over it); no kept file tossed at a sign-off listen
   (lofi_rl.freeze.SIGNOFFS: the problem names the brief, the file and the sign-off batch); each kept file is the text
   its source rendered and passes check_one under the frozen palette, and check_folder passes every draft of drafts/v1
   and of each redo folder (a); prompt v2's 9 references, strict and identical to dev_gu's (b); the regex's length and
   sha256, and every draft of every folder fullmatches it (c); `python -m lofi_rl.freeze check --gate` (g); every
   kept loop's brief is its briefs.yaml brief, and every reference's cell has a brief. Each source's
   summary/renders.json that is still under --runs-root (`drafts-<its job>`) must have keep3_kept10.json's sha256 and
   give its records; each one that is not there is recorded.
2. `palettes` (summary/palettes/index.json): `python -m lofi_rl.freeze check --verify` (sources hashed, presets,
   coverage), with `--levels-day1/2/3` when all three shootout levels files are under --runs-root (absent ones are
   recorded as not on scratch); the bed is the picks' (the excerpt of the `excerpt_from` variant and the level of the
   `delta_from` variant of lofi_rl.freeze.BED) and the rain file's stretch of that excerpt hashes to the record's
   pcm_sha256; the twins, verified like the draft render palettes (drafts.palette.verify_files): `twin0`, the frozen
   document with no bed at the levels keep3 played (the draft palette's `#| ` levels), and, when the frozen levels
   differ from those (a level change of Shresth's, spec §4.2), `twin1`, the frozen document with no bed.
3. `render` (summary/renders.json): (a) the kept loops with the frozen palette and with each twin, `--jobs` at a time
   with one worker each (lofi_rl.drafts.render.render_one, FLACs kept); (b) every draft of every folder that is not
   kept and prompt v2's 9 references with the frozen palette, in this process, one at a time with 4 workers, as junk's
   render step ran until D-022, when it took two at a time with 2 workers each (lofi_rl.rl.junk.RENDER_JOBS); a
   reference renders in its own cell, with the brief of that cell's first brief id; (c) each frozen kept render again
   in a fresh process (`python -m lofi_rl.render.pipeline ... --no-verify --workers 4`,
   PYTHONHASHSEED=4093), one at a time, its wall time `process_s`. A record holds n, LUFS, true peak, the loudness
   iterations, master gain, music_lufs, the bed report with `bed_lu` and pre_limiter_sha256
   (lofi_rl.shootout.render.bed_record), pcm_sha256, music_sha256 and `render_s`; a kept loop's records also hold
   each bus's effective trim, fader and loudness.
4. `verify` (summary/verify.json): N; -14 LUFS within 0.1 LU and a true peak at or below -1 dBTP; G1-G7 with no repair;
   the iterations of every render (a G7 failure fails the job); the bed as placed, measured on the rebuilt stem
   (bed.place(bed.bed_loop(palette, n), music_lufs, delta) within PLACED_TOL_LU of -delta, its gain and own loudness
   the manifest's exactly); the bed as heard, recorded and flagged above HEARD_FLAG_LU closer than -delta; the frozen
   render equals its fresh-process repeat (the whole manifest); twin0 is what he kept (its own source's pcm_sha256:
   keep3's, or its redo round's); the music under the bed is the twin's (pre_limiter_sha256 and music_lufs); each
   bus's pre-fader loudness, effective trim and fader are its source's (a changed level moves only by its change); and
   the render times, with the Phase-1 gate's projection (PROJECTION_CLIPS x the slowest one-at-a-time render against
   PROJECTION_LIMIT_S). On a twin mismatch it names the stems that differ from its source's manifest
   (<runs-root>/drafts-<its source's job>/renders/<its stem>/manifest.json), when scratch still holds it.
5. `build` (summary/draw.json, summary/batch.json): the triage batch (lofi_rl.freeze.batch), checked by `python -m
   lofi_rl.ui.manifest check` before it is moved into place.
Then SIGNOFF_OK (and the job script's DRAFTS_OK). Exit 0; 2 for bad arguments or a batch that exists; 3 for a failed
check; 1 for an infrastructure fault. `--dry-run` runs the check and the palettes, prints what the render and the
build would do, and writes nothing outside the run folder. Sound claims: none.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date as _date
from pathlib import Path

import yaml

from lofi_rl import freeze as F
from lofi_rl import prompt_v2
from lofi_rl.brief import Brief
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.drafts import briefs as B
from lofi_rl.drafts import job as J
from lofi_rl.drafts import palette as DP
from lofi_rl.drafts.render import default_renderer, level_problems, render_one
from lofi_rl.freeze import BED, JOBS
from lofi_rl.palette import ROOT, Palette, PaletteError, load_palette, parse_palette
from lofi_rl.ui.manifest import BATCH_RE

from . import batch as SB
from . import checks as C
from . import gate as G

ARGS_SCHEMA = "lofi-rl/signoff-args/1"
CHECK_SCHEMA = "lofi-rl/signoff-check/2"           # /2: the kept loops' sources, the tossed check, every folder
PALETTES_SCHEMA = "lofi-rl/signoff-palettes/1"
RENDERS_SCHEMA = "lofi-rl/signoff-renders/2"       # /2: kept records under render ids, `kept_loops` maps them
VERIFY_SCHEMA = "lofi-rl/signoff-verify/1"
OK_LINE = "SIGNOFF_OK"
GATE = C.GATE
PLAN = ROOT / "shootouts" / "plan.yaml"
RAIN = ROOT / "shootouts" / "rain_excerpts.json"
DRAFT_PALETTE = DP.DRAFT_PALETTE            # its `#| ` levels are what keep3 played: twin0's levels
DEV_PALETTE = C.DEV_PALETTE
PLACED_TOL_LU = 0.01        # the bed as placed: lufs(scaled stem) - music_lufs within 0.01 LU of -delta (spec §4.3)
# The bed as heard is flagged (information, never a failure) when it sits more than this closer than -delta: the
# largest such offset among the W2 renders at delta 24 that Shresth judged (runs/shootout-13656048/summary/
# renders.json, W2.l6_wistful_high.rain_d24: 2.0147364132581025, read this session), rounded up to 0.001 as the spec has it.
HEARD_FLAG_LU = 2.015
PROJECTION_CLIPS, PROJECTION_LIMIT_S = 50, 300.0     # spec §5.5: nothing changes if 50 x the slowest render <= 300 s
REPEAT_HASHSEED = "4093"    # the fresh-process repeat's hash seed, as tests/test_determinism.py's CLI render
REPEAT_WORKERS = 4          # (c) and (b) run with 4 workers, as junk's render step did until D-022 (now 2 x 2)
ONE_AT_A_TIME_WORKERS = 4
# A fresh-process repeat that runs longer than this is a fault: on the laptop one took 3.01-4.81 s (docs/DRAFTS.md
# §7.1), and the whole job's walltime is 15 min (jobs/drafts.sbatch).
REPEAT_TIMEOUT_S = 300
GOOD_GATES = {"G1-G6": "pass", "G7": "pass"}
KINDS = ("frozen", "twin0", "twin1")


class SignoffError(ValueError):
    pass


# --------------------------------------------------------------------------------------------- hooks (tests replace)
freeze_check = C.freeze_check               # `python -m lofi_rl.freeze check ...` in a fresh interpreter
verify_files = DP.verify_files              # the shootout's source and preset validation plus SF2 coverage


def plan_variants(path: Path) -> dict[str, dict]:
    """Each plan variant's `set` mapping (shootouts/plan.yaml, read by the shootout's own loader)."""
    from lofi_rl.shootout.plan import load_plan
    return {vid: dict(v.set) for vid, v in load_plan(path).variants.items()}


def reference_loops(palette: Palette) -> dict[str, tuple[tuple[str, str], str]]:
    """Prompt v2's references under the palette, strict: {example id: ((energy, mood) of its cell, text)}."""
    cells = {ex_id: cell for cell, (ex_id, _) in prompt_v2.EXAMPLES.items()}
    return {ex_id: (cells[ex_id], text) for ex_id, text in prompt_v2.reference_texts(palette, strict=True).items()}


def run_repeat(task: Mapping, out_wav: Path, palette_path: Path, cache: Path | None) -> dict:
    """Render (c): the loop again in a fresh process through the pipeline's CLI, with another hash seed and 4 workers;
    its exit code, wall time (`process_s`) and the manifest it wrote."""
    b = task["brief"]
    cmd = [sys.executable, "-m", "lofi_rl.render.pipeline", str(task["loop_path"]), "--mood", b["mood"],
           "--energy", b["energy"], "--bpm", str(b["bpm"]), "--tonic", str(b["tonic"]), "--palette", str(palette_path),
           "--out", str(out_wav), "--no-verify", "--workers", str(REPEAT_WORKERS)]
    if cache is not None:
        cmd += ["--cache", str(cache)]
    src = str(ROOT / "src")
    env = {**os.environ, "PYTHONHASHSEED": REPEAT_HASHSEED,
           "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=REPEAT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"exit": None, "process_s": round(time.perf_counter() - t0, 2),
                "error": f"timed out after {REPEAT_TIMEOUT_S} s"}
    out = {"exit": p.returncode, "process_s": round(time.perf_counter() - t0, 2)}
    side = Path(out_wav).with_suffix(".manifest.json")
    if p.returncode == 0 and side.is_file():
        text = side.read_text(encoding="utf-8")
        m = json.loads(text)
        out.update(manifest_sha256=C.sha256_bytes(text.encode("utf-8")), audio_sha256=m["audio_sha256"],
                   music_sha256=m["music_sha256"], n=m["n"])
    else:
        out["error"] = (p.stderr or p.stdout)[-2000:]
    return out


# --------------------------------------------------------------------------------------------- the bed and the twins
def expected_bed(variants: Mapping[str, Mapping], rain: Mapping, picks: Mapping[str, str]) -> dict:
    """The bed the picks name (the freeze spec §2.9): the excerpt of the `excerpt_from` variant and only the level of
    the `delta_from` variant (rain_d24 also names e1, the excerpt W2 played its levels on), with the excerpt's start,
    stretch and pcm_sha256, the crossfade and the rain file from the rain record. Also what Day 3 played of it, which the
    note names: the level W1 played the excerpt at (`w1_delta_lu`) and the excerpt W2 played the level on."""
    ex, lv = variants[picks["excerpt_from"]], variants[picks["delta_from"]]
    excerpt = ex["bed.excerpt"]
    e = rain["excerpts"][excerpt]
    return {"excerpt": excerpt, "start_s": float(e["start_s"]), "xfade_s": float(rain["parameters"]["xfade_s"]),
            "delta_lu": float(lv["bed.delta_lu"]), "samples": [int(e["samples"][0]), int(e["samples"][1])],
            "pcm_sha256": e["pcm_sha256"], "asset_path": rain["source"]["asset_path"],
            "file_sha256": rain["source"]["sha256"], "w1_delta_lu": float(ex["bed.delta_lu"]),
            "w2_excerpt": lv.get("bed.excerpt"), "from": dict(picks)}


def bed_problems(pal: Palette, want: Mapping) -> list[str]:
    """The palette's bed against the picks' (expected_bed), and the note's claim that Day 3 never played the pair."""
    out = []
    if pal.bed is None:
        return [f"{pal.id} has no bed; the picks' is {want['excerpt']} at {want['delta_lu']:g} LU"]
    for k in ("start_s", "xfade_s", "delta_lu"):
        if getattr(pal.bed, k) != want[k]:
            out.append(f"bed.{k} is {getattr(pal.bed, k)!r}, the picks' ({want['excerpt']}) is {want[k]!r}")
    s = pal.sources.get(pal.bed.src)
    if s is None or (s.path, s.sha256, s.mode) != (want["asset_path"], want["file_sha256"], "fetch"):
        out.append(f"the bed's source {pal.bed.src} is not the rain record's file {want['asset_path']} "
                   f"({want['file_sha256'][:12]}..., mode fetch)")
    if want["w1_delta_lu"] == want["delta_lu"] or want["w2_excerpt"] == want["excerpt"]:
        out.append("Day 3 played this excerpt at this level: the note's 'new to your ears' does not hold")
    return out


def stretch_sha256(path: Path, samples: Sequence[int]) -> str:
    """pcm_sha256 of the file's frames [lo, hi), as lofi_rl.convert.rain_excerpts hashes an excerpt's stretch."""
    import numpy as np
    import soundfile as sf

    from lofi_rl.render.wavio import pcm_sha256
    lo, hi = samples
    x, _ = sf.read(str(path), start=lo, stop=hi, dtype="float32", always_2d=True)
    if x.shape[0] != hi - lo:
        raise SignoffError(f"{path} holds {x.shape[0]} of the excerpt's {hi - lo} frames")
    return pcm_sha256(np.ascontiguousarray(x.T))


def palette_levels(pal: Palette) -> dict[str, tuple[float, float]]:
    """Every instrument's (trim_db, fader_db), by palette path."""
    out = {f"sax.{n}": (v.trim_db, v.fader_db) for n, v in pal.sax.items()}
    out.update({f"keys.{n}": (v.trim_db, v.fader_db) for n, v in pal.keys.items()})
    for n, k in pal.kits.items():
        out[f"kits.{n}"] = (k.trim_db, k.fader_db)
        out[f"kits.{n}.bass"] = (k.bass.trim_db, k.bass.fader_db)
    return out


def keep3_levels(dp: DP.DraftPalette) -> dict[str, tuple[float, float]]:
    """The levels keep3 played (the draft palette's `#| ` record), by palette path."""
    out = {f"sax.{dp.palette.sax_lead}": (dp.sax.trim_db, dp.sax.fader_db)}
    out.update({f"keys.{n}": (v.trim_db, v.fader_db) for n, v in dp.keys.items()})
    for n in dp.kits:
        out[f"kits.{n}"] = (dp.kits[n].trim_db, dp.kits[n].fader_db)
        out[f"kits.{n}.bass"] = (dp.basses[n].trim_db, dp.basses[n].fader_db)
    return out


def _node(doc: dict, path: str) -> dict:
    node = doc
    for k in path.split("."):
        node = node[k]
    return node


def twin_doc(doc: Mapping, twin_id: str, levels: Mapping[str, tuple[float, float]] | None = None) -> dict:
    """The frozen document with no bed (and without a source only the bed played) under `twin_id`; with `levels`, each
    instrument's trim_db and fader_db set from it, by palette path."""
    d = copy.deepcopy(dict(doc))
    d["id"], bed, d["bed"] = twin_id, d.get("bed"), None
    if bed and bed["src"] not in DP.sources_of(d):
        d["sources"].pop(bed["src"], None)
    for path, (t, f) in (levels or {}).items():
        node = _node(d, path)
        node["trim_db"], node["fader_db"] = t, f
    return d


def twin_text(doc: Mapping, frozen: Palette, what: str) -> str:
    head = (f"# Sign-off twin {doc['id']} ({what}): written by lofi_rl.freeze.signoff from {frozen.id} (sha256 "
            f"{frozen.sha256[:12]}...); do not edit. Sound claims: none.\n")
    return head + DP.yaml_text(doc)


# --------------------------------------------------------------------------------------------- renders
def _brief(b: Mapping) -> Brief:
    return Brief(energy=b["energy"], mood=b["mood"], bpm=b["bpm"], tonic=b["tonic"])


def extras(r) -> dict:
    """What a sign-off record adds to a drafts record: the bed with bed_lu, music_lufs and pre_limiter_sha256
    (bed_record), the loudness iterations, the repairs, the sax that played and the manifest's own sha256."""
    from lofi_rl.render.pipeline import manifest_json
    from lofi_rl.shootout.render import bed_record
    m = r.manifest
    return {**bed_record(m, r.audio, r.music, r.pre_limiter), "iterations": m["loudness"]["iterations"],
            "repairs": list(m["loop"]["repairs"]), "sax": m["instruments"]["sax"],
            "manifest_sha256": C.sha256_bytes(manifest_json(m).encode("utf-8"))}


def _capture(renderer, box: dict):
    base = renderer or default_renderer()

    def render(text, brief, palette, *, cache=None, workers=1):
        r = base(text, brief, palette, cache=cache, workers=workers)
        box["extra"] = extras(r)
        return r
    return render


def render_kept(items: list[tuple[str, dict, Path]], *, jobs: int, cache: Path | None, renderer) -> list[tuple[str, dict]]:
    """Render (a): each (kind, task, out dir) through drafts.render.render_one, `jobs` at a time with one worker each;
    [(kind, record with the sign-off fields and render_s)], in order. Each record.json is rewritten with them."""
    def one(item):
        kind, task, out = item
        box: dict = {}
        rec, s = render_one(task, str(out), 1, None if cache is None else str(cache), renderer=_capture(renderer, box))
        rec = {**rec, **box.get("extra", {}), "kind": kind, "render_s": round(s, 2)}
        (out / task["render"] / "record.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n",
                                                          encoding="utf-8")
        return kind, rec
    for _, _, out in items:
        out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        return list(ex.map(one, items))


def render_other(task: Mapping, pal: Palette, out: Path, *, cache: Path | None, renderer) -> dict:
    """Render (b): one loop with `pal` in this process with 4 workers, its FLAC and manifest written as junk's render
    step writes them, `render_s` taken over the same work (lofi_rl.rl.junk.render_clip). A record, never an exception
    for a gate or compile failure."""
    from lofi_rl.compile.compiler import CompileError
    from lofi_rl.render.formats import write_flac
    from lofi_rl.render.pipeline import RenderError, manifest_json
    out.mkdir(parents=True, exist_ok=True)
    rid, b = task["render"], task["brief"]
    rec = {"render": rid, "kind": task["kind"], "file": task["file"], "brief_id": task["brief_id"], "brief": dict(b),
           "bpm": b["bpm"], "status": None, "error": None, "gate": None}
    t0 = time.perf_counter()
    r = None
    try:
        r = (renderer or default_renderer())(task["text"], _brief(b), pal, cache=cache, workers=ONE_AT_A_TIME_WORKERS)
        write_flac(out / f"{rid}.flac", r.audio)
        (out / f"{rid}.manifest.json").write_text(manifest_json(r.manifest), encoding="utf-8")
        rec["status"] = "ok"
    except RenderError as err:
        rec.update(status="gate", gate=err.code, error=f"{err.code}: {err.detail}")
    except CompileError as err:
        rec.update(status="compile", error=str(err))
    except Exception:                                   # an infrastructure fault: never a gate, never hidden
        rec.update(status="error", error=traceback.format_exc(limit=8))
    rec["render_s"] = round(time.perf_counter() - t0, 2)
    if rec["status"] != "ok":
        return rec
    try:
        m = r.manifest
        rec.update(keys=r.compiled.keys_name, kit=r.compiled.kit, n=m["n"], lufs=m["loudness"]["lufs"],
                   true_peak_dbtp=m["loudness"]["true_peak_dbtp"], master_gain_db=m["loudness"]["gain_db"],
                   pcm_sha256=m["audio_sha256"], music_sha256=m["music_sha256"], gates=m["gates"], **extras(r))
    except Exception:                                   # a manifest without a field this record reads: a fault
        rec.update(status="error", error=traceback.format_exc(limit=8))
        return rec
    bad = level_problems(rec)
    if bad:
        rec.update(status="level", error="; ".join(bad))
    return rec


# --------------------------------------------------------------------------------------------- verify
def placed_bed(pal: Palette, n: int, music_lufs: float, delta: float, cache: dict) -> tuple[float, float, float]:
    """The bed as placed, rebuilt as the pipeline places it: (lufs(scaled stem) - music_lufs, the gain, the bed's own
    loudness), with one bed loop per n."""
    from lofi_rl.render.bed import SR, bed_loop, place
    from lofi_rl.render.loudness import lufs
    if n not in cache:
        cache[n] = bed_loop(pal, n)
    scaled, gain, own = place(cache[n], music_lufs, delta)
    return float(lufs(scaled, SR) - music_lufs), float(gain), float(own)


def stem_diff(mine: Path, theirs: Path, source: str = "keep3") -> list[str]:
    """Which bus inputs (`stems`) and one-shots of two render manifests differ, or why they cannot be compared:
    `theirs` is the kept loop's render in its source round (`source`: keep3 or a redo round's key)."""
    if not mine.is_file():
        return [f"{mine} is missing"]
    if not theirs.is_file():
        return [f"{source}'s manifest {theirs} is not on scratch"]
    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (mine, theirs))
    sa, sb = a.get("stems") or {}, b.get("stems") or {}
    out = [f"stem {k}: {sa.get(k)} here, {sb.get(k)} in {source}" for k in sorted(set(sa) | set(sb))
           if sa.get(k) != sb.get(k)]
    if a.get("oneshots") != b.get("oneshots"):
        out.append("the one-shots differ")
    return out or ["no bus input differs: the difference comes after the bus inputs"]


def instrument_path(rec: Mapping, bus: str) -> str:
    """The palette path of the instrument a render record's bus plays."""
    return {"sax": f"sax.{rec.get('sax')}", "keys": f"keys.{rec.get('keys')}", "bass": f"kits.{rec.get('kit')}.bass",
            "drums": f"kits.{rec.get('kit')}"}[bus]


def bus_expectation(want: Mapping, change: Mapping | None) -> dict:
    """What a bus of a frozen or twin1 render must read, given its source's record of it (`want`, keep3's or a redo
    round's, which played keep3's levels): that pre-fader loudness, effective trim and fader; under a level change of
    his (spec §4.2) the frozen fader instead (bus_gain_db is 0, so the effective fader is the instrument's field), and,
    when the trim moved, no loudness or trim to compare (recorded only)."""
    if change is None:
        return {k: want[k] for k in ("lufs", "trim_db", "fader_db")}
    (k3_trim, k3_fader), (trim, fader) = change["keep3"], change["frozen"]
    out = {"fader_db": want["fader_db"] if fader == k3_fader else fader}
    if trim == k3_trim:
        out.update(lufs=want["lufs"], trim_db=want["trim_db"])
    return out


def _g7(rec: Mapping) -> bool:
    return rec.get("status") == "gate" and str(rec.get("error") or "").startswith("G7")


def verify_doc(st: Mapping) -> dict:
    """Stage 4 (module doc) over the render stage's records (`st`: the job state): one document, its `problems`
    empty when every check passes. Each kept render id maps to its kept loop (`st["kept_ids"]`; a render id that is
    the loop's own stem when the job state has no map), and each kept loop is checked against its own source."""
    pal, expect, bed, changes = st["pal"], st["expect"], st["bed"], st["changes"]
    delta = bed["delta_lu"]
    kept, others = st["records"]["kept"], st["records"]["others"]
    loop_of = st.get("kept_ids") or {rid: rid for rid in kept}
    jobs = {s["key"]: s["job"] for s in expect["sources"]}
    twin = "twin1" if changes else "twin0"
    every = [(f"{rid}.{k}", row[k]) for rid, row in kept.items() for k in KINDS if k in row] + list(others.items())
    frozen = [(rid, row["frozen"]) for rid, row in kept.items() if "frozen" in row] + list(others.items())
    rendered = ("ok", "level")
    checks: dict[str, dict] = {}

    def add(name: str, rule: str, problems: list[str], **info) -> None:
        checks[name] = {"rule": rule, "ok": not problems, "problems": problems, **info}

    add("status", "every render compiled and passed G1-G6 and G7",
        [f"{w}: {r.get('status')}: {str(r.get('error') or '').strip()[-300:]}" for w, r in every
         if r.get("status") not in rendered and not _g7(r)])
    iters = {w: r.get("iterations") for w, r in frozen}
    add("iterations", f"the loudness iterations of every frozen render, recorded; a G7 failure fails the job (the "
        f"palette's cap is {pal.loudness.iters})", [f"{w}: G7 failure: {r.get('error')}" for w, r in frozen if _g7(r)],
        iterations=iters, most=max((v for v in iters.values() if v is not None), default=None))
    ns = [f"{w}: n {r['n']} is not loop_samples({r['bpm']}) = {loop_samples(r['bpm'])}" for w, r in every
          if r.get("n") is not None and r["n"] != loop_samples(r["bpm"])]
    for rid, row in kept.items():
        e = expect["loops"][loop_of[rid]]
        want = e["n"]
        got = {**{k: row[k].get("n") for k in KINDS if k in row}, "repeat": (row.get("repeat") or {}).get("n")}
        ns += [f"{rid}.{k}: n {v} is not {e['source']}'s {want}" for k, v in got.items() if v is not None and v != want]
    add("n", "N = its source's n (keep3's or its redo round's) = loop_samples(bpm), exactly", ns)
    add("loudness", "-14 LUFS within 0.1 LU and a true peak at or below -1 dBTP (every render, the twins included)",
        [f"{w}: {x}" for w, r in every if r.get("status") in rendered for x in level_problems(r)])
    add("gates", "G1-G6 and G7 pass with no repair",
        [f"{w}: gates {r.get('gates')}, repairs {r.get('repairs')}" for w, r in every
         if r.get("status") in rendered and (r.get("gates") != GOOD_GATES or r.get("repairs"))])

    placed, heard, flags, beds, cache = [], {}, [], {}, {}
    for w, r in frozen:
        if r.get("status") not in rendered:
            continue
        b = r.get("bed")
        if not b:
            placed.append(f"{w}: the render has no bed")
            continue
        got, gain, own = placed_bed(pal, r["n"], r["music_lufs"], delta, cache)
        beds[w] = {"excerpt": bed["excerpt"], "delta_lu": delta, "placed_lu": got, "heard_lu": b["bed_lu"],
                   "closer_lu": b["bed_lu"] + delta}
        if not abs(got + delta) <= PLACED_TOL_LU:
            placed.append(f"{w}: the rebuilt bed sits {got:.4f} LU from the music, not -{delta:g} within {PLACED_TOL_LU}")
        if gain != b["gain_db"] or own != b["lufs_source"]:
            placed.append(f"{w}: the rebuilt bed's gain {gain!r} and own loudness {own!r} are not the manifest's "
                          f"{b['gain_db']!r} and {b['lufs_source']!r}")
        heard[w] = b["bed_lu"]
        if b["bed_lu"] + delta > HEARD_FLAG_LU:
            flags.append(f"{w}: the bed as heard sits {b['bed_lu'] + delta:.3f} LU closer than -{delta:g} (information, "
                         f"above {HEARD_FLAG_LU})")
    add("bed_placed", f"the rebuilt bed stem at music_lufs - {delta:g} within {PLACED_TOL_LU} LU, its gain and own "
        "loudness the manifest's exactly", placed, beds=beds)
    below = [-heard[rid] for rid in kept if rid in heard]
    add("bed_heard", f"the bed as heard, lufs(render - music) - lufs(music): recorded, not gated; flagged when more "
        f"than {HEARD_FLAG_LU} LU closer than -delta", [], heard_lu=heard, flags=flags,
        kept_below=[min(below), max(below)] if below else None)

    rep_bad, twin_bad, music_bad, bus_bad, diffs = [], [], [], [], {}
    for rid, row in kept.items():
        key = loop_of[rid]
        e, fz, rep = expect["loops"][key], row.get("frozen") or {}, row.get("repeat")
        if fz.get("status") == "ok":
            if rep is None:
                rep_bad.append(f"{rid}: no fresh-process repeat")
            elif rep.get("exit") != 0:
                rep_bad.append(f"{rid}: the fresh-process repeat exited {rep.get('exit')}: "
                               f"{str(rep.get('error') or '').strip()[-300:]}")
            else:
                for k, mine in (("manifest_sha256", fz.get("manifest_sha256")), ("audio_sha256", fz.get("pcm_sha256")),
                                ("music_sha256", fz.get("music_sha256"))):
                    if rep.get(k) != mine:
                        rep_bad.append(f"{rid}: the repeat's {k} {str(rep.get(k))[:12]}... is not the render's "
                                       f"{str(mine)[:12]}...")
        t0 = row.get("twin0") or {}
        if t0.get("pcm_sha256") != e["pcm_sha256"]:
            twin_bad.append(f"{rid}: twin0's pcm_sha256 {str(t0.get('pcm_sha256'))[:12]}... is not {e['source']}'s "
                            f"{e['pcm_sha256'][:12]}...")
            # the source job rendered the loop under its stem (lofi_rl.drafts.job), the loop's key here
            diffs[rid] = stem_diff(st["run"] / "renders" / "twin0" / rid / "manifest.json",
                                   st["runs_root"] / f"drafts-{jobs[e['source']]}" / "renders" / key / "manifest.json",
                                   e["source"])
        tw = row.get(twin) or {}
        for k in ("pre_limiter_sha256", "music_lufs"):
            if fz.get(k) is None or fz.get(k) != tw.get(k):
                music_bad.append(f"{rid}: the frozen render's {k} {fz.get(k)!r} is not {twin}'s {tw.get(k)!r}")
        for kind in KINDS:
            rec = row.get(kind) or {}
            if rec.get("buses") is None:
                continue
            for b in C.BUSES:
                change = None if kind == "twin0" else changes.get(instrument_path(rec, b))
                for k, v in bus_expectation(e["buses"][b], change).items():
                    if rec["buses"][b][k] != v:
                        bus_bad.append(f"{rid}.{kind}: the {b} bus's {k} is {rec['buses'][b][k]!r}, not "
                                       f"{e['source'] if change is None else 'the changed level'}'s {v!r}")
    add("repeat", "the frozen render = its fresh-process repeat: audio_sha256, music_sha256 and the whole manifest",
        rep_bad, process_s={rid: (row.get("repeat") or {}).get("process_s") for rid, row in kept.items()})
    add("twin_is_kept", "twin0's audio_sha256 = its source's pcm_sha256 (keep3_kept10.json: keep3's or its redo "
        "round's)", twin_bad, stem_diffs=diffs, sources={rid: expect["loops"][loop_of[rid]]["source"] for rid in kept})
    add("music_under_bed", f"pre_limiter_sha256 and music_lufs: the frozen render = {twin}", music_bad)
    add("bus_levels", "each bus's pre-fader LUFS, effective trim and fader = its source's; a level change of his moves "
        "only its own fader (spec §4.2)", bus_bad, changes=changes)
    one = {rid: r.get("render_s") for rid, r in others.items()}
    slow = max((v for v in one.values() if v is not None), default=None)
    proj = None if slow is None else round(PROJECTION_CLIPS * slow, 2)
    add("render_time", f"recorded (information): render_s of each one-at-a-time render and process_s of each repeat; "
        f"the Phase-1 gate's projection, {PROJECTION_CLIPS} x the slowest render, against {PROJECTION_LIMIT_S:g} s", [],
        render_s=one, slowest_s=slow, projection_s=proj, within=None if proj is None else proj <= PROJECTION_LIMIT_S)
    problems = [p for c in checks.values() for p in c["problems"]]
    return {"schema": VERIFY_SCHEMA, "ok": not problems, "palette": st["palette_record"], "twin": twin,
            "checks": checks, "problems": problems}


# --------------------------------------------------------------------------------------------- the job
def _write(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def run(a: argparse.Namespace, argv: Sequence[str], *, renderer=None) -> int:
    """The `--signoff` run of lofi_rl.drafts.job (module doc), from its parsed arguments."""
    drafts = J._repo_path(a.drafts)
    frozen_path = J._repo_path(a.signoff)
    runs_root = J._repo_path(a.runs_root) if a.runs_root else Path(a.run).parent
    try:
        if a.jobs < 1:
            raise SignoffError("--jobs must be at least 1")
        briefs = B.load(B.briefs_file_for(drafts))
        day = _date.fromisoformat(a.date) if a.date else J.today_pacific()
        name = SB.batch_name(day, a.inbox, a.name)
        if not BATCH_RE.fullmatch(name):
            raise SignoffError(f"--name {name!r} must match {BATCH_RE.pattern}")
        if (a.inbox / day.isoformat() / name).exists():
            raise SignoffError(f"{a.inbox / day.isoformat() / name} exists; a built batch is never overwritten")
        seed = J._seed(a.seed)
        pal = load_palette(frozen_path, verify=False)
        frozen_text = frozen_path.read_text(encoding="utf-8")
    except (SignoffError, B.BriefsError, PaletteError, OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    summary = a.run / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    pal_rec = {"path": J._label(frozen_path), "id": pal.id, "sha256": pal.sha256, "label": f"{pal.id}@{pal.sha256[:8]}",
               "frozen": pal.frozen}
    _write(summary / "args.json", {
        "schema": ARGS_SCHEMA, "argv": list(argv), "mode": "signoff", "drafts": J._label(drafts),
        "palette": pal_rec["path"], "palette_id": pal.id, "palette_sha256": pal.sha256, "date": day.isoformat(),
        "batch": name, "jobs": a.jobs, "runs_root": str(runs_root), "dry_run": a.dry_run})
    stages = J.Stages(summary)
    st: dict = {"pal": pal, "palette_record": pal_rec, "run": Path(a.run), "runs_root": runs_root, "changes": {}}

    def check() -> int:
        doc = check_stage(pal, frozen_path, drafts, briefs, runs_root, st)
        _write(summary / "check.json", doc)
        a_ = doc.get("a") or {}
        print(f"check: kept {len(st.get('expect', {}).get('loops', {}))}; drafts {a_.get('folder', {}).get('ok')} of "
              f"{a_.get('folder', {}).get('drafts')} pass; regex {doc.get('c', {}).get('sha256')}; references "
              f"{doc.get('b', {}).get('count')}", flush=True)
        for x in doc["problems"]:
            print(f"check: {x}", file=sys.stderr)
        return 0 if not doc["problems"] else 3

    def palettes() -> int:
        doc = palettes_stage(pal, frozen_path, frozen_text, summary / "palettes", runs_root, st)
        _write(summary / "palettes" / "index.json", doc)
        for x in doc["problems"]:
            print(f"palettes: {x}", file=sys.stderr)
        return 0 if not doc["problems"] else 3

    def render() -> int:
        recs = render_stage(st, frozen_path, a.run / "renders", jobs=a.jobs, cache=a.cache, renderer=renderer)
        _write(summary / "renders.json", recs)
        every = [r for row in recs["kept"].values() for k, r in row.items() if k in KINDS] + list(recs["others"].values())
        ok = [r for r in every if r.get("status") == "ok"]
        print(f"render: {len(ok)} of {len(every)} ok in {recs['seconds']}", flush=True)
        for r in every:
            if r.get("status") != "ok":
                print(f"render {r.get('kind')} {r.get('render')}: {r.get('status')}: "
                      f"{str(r.get('error') or '').strip().splitlines()[-1:]}", file=sys.stderr)
        return 1 if any(r.get("status") == "error" for r in every) else 0

    def verify() -> int:
        doc = verify_doc(st)
        st["verify"] = doc
        _write(summary / "verify.json", doc)
        for x in doc["problems"]:
            print(f"verify: {x}", file=sys.stderr)
        heard = doc["checks"]["bed_heard"]
        print(f"verify: {'ok' if doc['ok'] else 'FAIL'}; the bed as heard {heard['kept_below']} LU below the music on "
              f"the kept loops; projection {doc['checks']['render_time']['projection_s']} s", flush=True)
        return 0 if doc["ok"] else 3

    def build() -> int:
        kept = st["records"]["kept"]
        beds = st["verify"]["checks"]["bed_placed"]["beds"]
        lo_hi = tuple(st["verify"]["checks"]["bed_heard"]["kept_below"])
        change = bool(st["changes"])
        again = bool(F.SIGNOFFS)                # an earlier sign-off listen played this rain at this level
        note = SB.note_text(len(kept), st["bed"], lo_hi, change, again=again)
        push = SB.push_text(name, len(kept), st["bed"], lo_hi, change, again=again)
        folder = SB.build({rid: row["frozen"] for rid, row in kept.items()}, a.run / "renders" / "signoff",
                          beds=beds, note=note, push=push, name=name, date=day.isoformat(), out_root=a.inbox,
                          seed=seed, summary_dir=summary)
        doc = json.loads((summary / "batch.json").read_text(encoding="utf-8"))
        print(f"batch {name}: {len(doc['rounds'])} triage rounds, est {doc['est_min']} min -> {folder}")
        print(f"push: {push}")
        return 0

    for stage_name, fn in (("check", check), ("palettes", palettes)):
        rc = stages.run(stage_name, fn)
        if rc:
            print(f"signoff: stopped at {stage_name} (exit {rc}); nothing was rendered or built", file=sys.stderr)
            return rc
    if a.dry_run:
        for kind, task, out in kept_items(st, frozen_path, a.run / "renders"):
            print(f"would render {task['render']} ({task['file']}) with {Path(task['palette']).name} [{kind}]")
        for task in other_tasks(st):
            print(f"would render {task['render']} ({task['file']}, brief {task['brief_id']}) with {frozen_path.name}, "
                  f"one at a time with {ONE_AT_A_TIME_WORKERS} workers")
        for rid in st["kept_ids"]:
            print(f"would repeat {rid} with {frozen_path.name} in a fresh process (PYTHONHASHSEED={REPEAT_HASHSEED}, "
                  f"--workers {REPEAT_WORKERS})")
        print(f"would build {a.inbox / day.isoformat() / name} ({len(st['kept_ids'])} triage rounds)")
        return 0
    for stage_name, fn in (("render", render), ("verify", verify), ("build", build)):
        rc = stages.run(stage_name, fn)
        if rc:
            print(f"signoff: stopped at {stage_name} (exit {rc}); no batch is written before the build",
                  file=sys.stderr)
            return rc
    print(f"signoff: every stage done in {sum(s['seconds'] for s in stages.done):.1f} s", flush=True)
    print(OK_LINE, flush=True)
    return 0


def check_stage(pal: Palette, frozen_path: Path, drafts: Path, briefs: Mapping[str, B.DraftBrief], runs_root: Path,
                st: dict) -> dict:
    """Stage 1 (module doc): its document, with `problems`; the kept loops and the drafts go into `st`."""
    doc: dict = {"schema": CHECK_SCHEMA, "palette": st["palette_record"], "problems": []}
    expect_path = drafts / C.EXPECT_NAME
    try:
        expect = C.load_expectations(expect_path)
    except C.ChecksError as err:
        doc["problems"] += err.errors
        return doc
    st["expect"] = expect
    doc["expect"] = {"path": J._label(expect_path), "sha256": C.file_sha256(expect_path), "sources": expect["sources"],
                     "kept": expect["kept"]}
    doc["kept_vs_constants"] = C.constants_problems(expect)
    if G.records_problems():                # REDOS or SIGNOFFS break their own rules: nothing below can read them
        doc["problems"] += doc["kept_vs_constants"]
        return doc
    doc["kept_vs_gate"] = C.kept_problems(expect, GATE)
    doc["tossed"] = C.tossed_problems()
    doc["problems"] += doc["kept_vs_constants"] + doc["kept_vs_gate"] + doc["tossed"]
    va, reports = C.vocabulary_check(pal, drafts, expect, with_report=True)
    every = [d for _fo, rep in reports for d in rep.drafts]
    st["drafts"] = {C.render_id(d.path, drafts): d for d in every}
    if len(st["drafts"]) != len(every):
        ids = [C.render_id(d.path, drafts) for d in every]
        twice = sorted({i for i in ids if ids.count(i) > 1})
        doc["problems"].append(f"render ids repeat across the drafts folders: {twice}")
    # each kept render id -> its kept loop (keep3_kept10.json's key, the kept file's stem), in brief order
    st["kept_ids"] = {C.render_id(ROOT / f, drafts): Path(f).stem for f in expect["kept"]}
    lost = [f for f in expect["kept"] if C.render_id(ROOT / f, drafts) not in st["drafts"]]
    if lost:
        doc["problems"].append(f"kept files outside the drafts folders {[r['folder'] for r in va['folders']]}: {lost}")
    doc["a"] = va
    dev = load_palette(DEV_PALETTE, verify=False)
    doc["b"] = C.reference_check(pal, dev)
    folders, _ = C.drafts_folders(drafts)
    doc["c"] = C.regex_check(pal, C.draft_files(folders), expect["kept"])
    doc["problems"] += va["problems"] + doc["b"]["problems"] + doc["c"]["problems"]
    rc, out = freeze_check(["--palette", frozen_path, "--gate", GATE])
    doc["g"] = {"cmd": "python -m lofi_rl.freeze check --gate", "exit": rc, "output": out[-4000:]}
    if rc != 0:
        doc["problems"].append(f"freeze check --gate exited {rc}: {out[-2000:]}")
    briefs_bad = []
    for rid, e in expect["loops"].items():
        bid = e["brief_id"]
        b = briefs.get(bid)
        want = None if b is None else {"energy": b.brief.energy, "mood": b.brief.mood, "bpm": b.brief.bpm,
                                       "tonic": b.brief.tonic}
        if want != e["brief"]:
            briefs_bad.append(f"{rid}: {e['source']} played brief {e['brief']}, briefs.yaml's {bid} is {want}")
    cells: dict[tuple[str, str], str] = {}
    for bid in sorted(briefs):
        cells.setdefault(briefs[bid].cell, bid)
    st["cell_brief"] = cells
    try:
        refs = reference_loops(pal)
    except Exception as err:            # PromptError: (b) already names it
        refs = {}
        briefs_bad.append(f"prompt v2's references: {err}")
    briefs_bad += [f"reference {ex_id}: no brief of its cell {list(cell)} in briefs.yaml" for ex_id, (cell, _) in
                   refs.items() if cell not in cells]
    st["refs"] = refs
    doc["briefs"] = {"problems": briefs_bad, "reference_briefs": {ex_id: cells.get(cell) for ex_id, (cell, _) in
                                                                   refs.items()}}
    doc["problems"] += briefs_bad
    doc["sources_scratch"] = sources_cross_check(expect, runs_root)
    doc["problems"] += doc["sources_scratch"]["problems"]
    return doc


def source_cross_check(expect: Mapping, source: Mapping, runs_root: Path, s_star: float) -> dict:
    """One source's summary/renders.json under runs_root (`drafts-<its job>`), if scratch still holds it: its sha256
    must be the one keep3_kept10.json pins, and it must give the records of the loops it supplies."""
    p = runs_root / f"drafts-{source['job']}" / "summary" / "renders.json"
    if not p.is_file():
        return {"path": str(p), "present": False, "note": "not on scratch: keep3_kept10.json stands", "problems": []}
    data = p.read_bytes()
    sha = C.sha256_bytes(data)
    out = {"path": str(p), "present": True, "sha256": sha, "problems": []}
    if sha != source["sha256"]:
        out["problems"].append(f"{p} has sha256 {sha[:12]}..., keep3_kept10.json pins {source['sha256'][:12]}... for "
                               f"{source['key']}")
        return out
    mine = {rid: e for rid, e in expect["loops"].items() if e["source"] == source["key"]}
    src = C.Source(source["key"], source["path"], source["job"], sha, json.loads(data.decode("utf-8")))
    fresh, errors = C.source_loops(src, [e["file"] for e in mine.values()], s_star=s_star)
    out["equal"] = not errors and fresh == mine and s_star == expect["s_star"]
    if not out["equal"]:
        out["problems"].append(f"{p}'s records are not keep3_kept10.json's for {source['key']}" +
                               (f": {errors}" if errors else ""))
    return out


def sources_cross_check(expect: Mapping, runs_root: Path) -> dict:
    """Each source of keep3_kept10.json against its summary/renders.json under runs_root (source_cross_check), with
    the draft palette's S*; each one scratch no longer holds is recorded."""
    s_star = DP.load(DRAFT_PALETTE).levels.get("S_star")
    rows = {s["key"]: source_cross_check(expect, s, runs_root, s_star) for s in expect["sources"]}
    return {"sources": rows, "absent": [k for k, r in rows.items() if not r["present"]],
            "problems": [x for r in rows.values() for x in r["problems"]]}


def palettes_stage(pal: Palette, frozen_path: Path, frozen_text: str, out: Path, runs_root: Path, st: dict) -> dict:
    """Stage 2 (module doc): its document, with `problems`; the bed, the twins and the changes go into `st`."""
    doc: dict = {"schema": PALETTES_SCHEMA, "frozen": st["palette_record"], "problems": []}
    lv = {}
    for day in ("day1", "day2", "day3"):                # the shootout jobs the palette's levels come from (JOBS)
        p = runs_root / f"shootout-{JOBS[day]}" / "summary" / "levels.json"
        lv[day] = ({"path": str(p), "present": True, "sha256": C.file_sha256(p)} if p.is_file()
                   else {"path": str(p), "present": False, "note": "not on scratch"})
    doc["levels_scratch"] = lv
    args = ["--palette", frozen_path, "--verify"]
    if all(x["present"] for x in lv.values()):
        args += [x for day in ("day1", "day2", "day3") for x in (f"--levels-{day}", lv[day]["path"])]
    rc, text = freeze_check(args)
    doc["freeze_check"] = {"args": [str(x) for x in args], "exit": rc, "output": text[-4000:]}
    if rc != 0:
        doc["problems"].append(f"freeze check {' '.join(str(x) for x in args)} exited {rc}: {text[-2000:]}")
    try:
        rain = json.loads(RAIN.read_text(encoding="utf-8"))
        want = expected_bed(plan_variants(PLAN), rain, BED)
    except (OSError, KeyError, TypeError, ValueError) as err:
        doc["problems"].append(f"the picks' bed: {err}")
        return doc
    st["bed"] = want
    bed_doc = {"want": want, "problems": bed_problems(pal, want)}
    if not bed_doc["problems"]:
        try:
            got = stretch_sha256(pal.source_path(pal.bed.src), want["samples"])
            bed_doc["stretch_sha256"] = got
            if got != want["pcm_sha256"]:
                bed_doc["problems"].append(f"the rain file's frames {want['samples'][0]}-{want['samples'][1]} hash to "
                                           f"{got[:12]}..., not excerpt {want['excerpt']}'s {want['pcm_sha256'][:12]}...")
        except (OSError, RuntimeError, SignoffError, PaletteError) as err:
            bed_doc["problems"].append(f"the rain file: {err}")
    doc["bed"] = bed_doc
    doc["problems"] += bed_doc["problems"]
    try:
        k3 = keep3_levels(DP.load(DRAFT_PALETTE))
    except DP.DraftPaletteError as err:
        doc["problems"].append(f"keep3's levels: {err}")
        return doc
    fz = palette_levels(pal)
    if set(fz) != set(k3):
        doc["problems"].append(f"the frozen instruments {sorted(fz)} are not the ones keep3 played {sorted(k3)}")
        return doc
    changes = {p: {"keep3": list(k3[p]), "frozen": list(fz[p]), "fader_offset_db": fz[p][1] - k3[p][1],
                   "trim_offset_db": fz[p][0] - k3[p][0]} for p in sorted(fz) if fz[p] != k3[p]}
    st["changes"] = changes
    doc["changes"] = changes
    base = yaml.safe_load(frozen_text)
    twins = {"twin0": (twin_doc(base, f"{pal.id}_nobed", k3), "no bed, at the levels keep3 played")}
    if changes:
        twins["twin1"] = (twin_doc(base, f"{pal.id}_nobed_adjusted"), "no bed, at the frozen levels")
    out.mkdir(parents=True, exist_ok=True)
    doc["twins"], files = {}, []
    for kind, (d, what) in twins.items():
        text = twin_text(d, pal, what)
        path = out / f"{d['id']}.yaml"
        path.write_text(text, encoding="utf-8")
        try:
            tp = parse_palette(yaml.safe_load(text), str(path), C.sha256_bytes(text.encode("utf-8")))
        except PaletteError as err:
            doc["problems"].append(f"{kind}: {err}")
            continue
        files.append(path)
        doc["twins"][kind] = {"file": path.name, "id": tp.id, "sha256": tp.sha256, "label": f"{tp.id}@{tp.sha256[:8]}",
                              "what": what}
    st["twins"] = {k: out / v["file"] for k, v in doc["twins"].items()}
    if not doc["problems"]:
        bad = verify_files(files)
        doc["problems"] += [f"twin: {x}" for x in bad]
    return doc


def kept_items(st: Mapping, frozen_path: Path, renders: Path) -> list[tuple[str, dict, Path]]:
    """Render (a)'s (kind, task, out dir): each kept loop with the frozen palette and with each twin, under its render
    id (st["kept_ids"], in brief order), its draft the one read from its own file."""
    out = []
    pals = {"frozen": frozen_path, **st.get("twins", {})}
    for rid, key in st["kept_ids"].items():
        d = st["drafts"][rid]
        b = d.brief
        task = {"render": rid, "loop": key, "source": st["expect"]["loops"][key]["source"], "file": J._label(d.path),
                "loop_path": str(d.path), "text": d.text, "brief_id": d.brief_id,
                "brief": {"energy": b.energy, "mood": b.mood, "bpm": b.bpm, "tonic": b.tonic},
                "persona": d.persona, "keys": d.loop.keys.instrument, "kit": d.loop.kit, "s_star": st["expect"]["s_star"]}
        for kind in KINDS:
            if kind in pals:
                out.append((kind, {**task, "palette": str(pals[kind])}, renders / ("signoff" if kind == "frozen" else kind)))
    return out


def other_tasks(st: Mapping) -> list[dict]:
    """Render (b)'s tasks: every draft of every folder that is not kept (by its path), then prompt v2's references,
    each with its cell's brief."""
    out = []
    for rid, d in sorted(st["drafts"].items()):
        if rid in st["kept_ids"] or d.brief is None:
            continue
        b = d.brief
        out.append({"render": rid, "kind": "draft", "file": J._label(d.path), "text": d.text, "brief_id": d.brief_id,
                    "brief": {"energy": b.energy, "mood": b.mood, "bpm": b.bpm, "tonic": b.tonic}})
    briefs = {d.brief_id: d.brief for d in st["drafts"].values() if d.brief is not None}
    for ex_id, (cell, text) in st["refs"].items():
        bid = st["cell_brief"][cell]
        b = briefs.get(bid)
        out.append({"render": f"ref_{ex_id}", "kind": "reference", "file": f"prompt_v2:{ex_id}", "text": text,
                    "brief_id": bid, "brief": {"energy": b.energy, "mood": b.mood, "bpm": b.bpm, "tonic": b.tonic}})
    return out


def render_stage(st: dict, frozen_path: Path, renders: Path, *, jobs: int, cache: Path | None, renderer) -> dict:
    """Stage 3 (module doc): renders (a), (b) and (c); the records go into `st` and the returned document."""
    t0 = time.perf_counter()
    kept: dict[str, dict] = {rid: {} for rid in st["kept_ids"]}
    items = kept_items(st, frozen_path, renders)
    for (kind, task, _), (_, rec) in zip(items, render_kept(items, jobs=jobs, cache=cache, renderer=renderer)):
        kept[task["render"]][kind] = rec
    t1 = time.perf_counter()
    others = {}
    for task in other_tasks(st):
        others[task["render"]] = render_other(task, st["pal"], renders / "others", cache=cache, renderer=renderer)
        print(f"render (b) {task['render']}: {others[task['render']]['status']} in "
              f"{others[task['render']]['render_s']} s", flush=True)
    t2 = time.perf_counter()
    tasks = {task["render"]: task for kind, task, _ in items if kind == "frozen"}
    for rid, row in kept.items():
        if row.get("frozen", {}).get("status") == "ok":
            row["repeat"] = run_repeat(tasks[rid], renders / "signoff" / rid / "repeat.wav", frozen_path, cache)
            print(f"render (c) {rid}: exit {row['repeat']['exit']} in {row['repeat']['process_s']} s", flush=True)
    t3 = time.perf_counter()
    st["records"] = {"kept": kept, "others": others}
    return {"schema": RENDERS_SCHEMA, "palette": st["palette_record"],
            "twins": {k: Path(v).name for k, v in st.get("twins", {}).items()}, "kept_loops": dict(st["kept_ids"]),
            "kept": kept, "others": others,
            "seconds": {"kept": round(t1 - t0, 1), "others": round(t2 - t1, 1), "repeats": round(t3 - t2, 1)}}
