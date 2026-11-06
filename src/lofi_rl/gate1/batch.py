"""`python -m lofi_rl.gate1 batch RUN_DIR --clips DIR [--inbox DIR] [--date YYYY-MM-DD]`: the gate run's clips as one
triage batch for Shresth's ear check (D-021 "What follows" item 1: at least 15 of 50 "fine").

- **Only for a run that passes check** (valid, and `pass_not_copy.count` >= 300). Otherwise it refuses with exit 3
  and writes nothing: an invalid run decides nothing (D-020), and under 300 the gate fails whatever the ear says
  (D-021), so the rounds would cost his listening and decide nothing.
- **The clips** are clips.json's records that rendered, in the sample's order; one that did not (G7, another gate, an
  infrastructure error, a render-deadline skip, or a sample the render step never reached) is left out and named in
  summary/batch.json and on the terminal. lofi_rl.gate1.load_run has already checked that clips.json is the run's
  own sample of the passing non-copies (D-020).
- **Each FLAC must be what clips.json records,** or the batch refuses, every problem listed before anything is
  written. clips.json keeps no sha256 of the FLAC file: junk.render_clip records the float render's pcm_sha256
  (`audio_sha256`) and the FLAC's size (`flac_bytes`), and the 16-bit copy is dithered, so its samples never hash to
  the float render's. So `<i>.flac` in --clips (the name clips.json's `clip` gives) is held to every field clips.json
  does record of it:
  - a plain file of exactly `flac_bytes`;
  - its STREAMINFO one pass of the loop: 44,100 Hz, 2 channels, clips.json's `n` frames, which is `loop_samples(bpm)`
    of the brief;
  - its samples, decoded and measured as the render measured them (render.loudness.lufs and
    render.pipeline.loop_true_peak_dbtp, at full scale 32767 as render.formats wrote them): within MEASURE_TOL of
    clips.json's `lufs` and `true_peak_dbtp`, which keep 3 decimals (on three laptop renders through
    junk.render_clip the copy measured within 1e-6 LU and 3.2e-4 dB of its render);
  - and, should clips.json ever record a `flac_sha256`, that sha256 exactly.
  From here on each copy has its own sha256: the public manifest lists it, `ui.manifest check` verifies it in the
  batch folder, and the hidden manifest and batch.json keep it.
- **The render's manifest** (`<i>.manifest.json`, which junk's render step writes beside each FLAC) is read when it
  was copied with the FLAC, and must then agree with clips.json and the run: `audio_sha256`, `n`, `sr`, the loop's
  `canon_sha256`, the palette, the brief, LUFS and true peak to clips.json's 3 decimals, and its gates passed. The
  real clip id (taste_pipeline.md §2, lofi_rl.drafts.render.clip_id) hashes its schema and chain; without it, the
  render's schema and chain come from lofi_rl.render.pipeline, as junk.render_clip renders (cycles 2), and
  batch.json says which clips had no manifest.
- **The brief** is the measurement stream's brief i (`brief.sample_brief(2026, i)`, D-021 item 1), in clips.json and
  in completions.jsonl alike, and clips.json's `canon_sha256` (the render's) is scored.jsonl's (score's): one text,
  parsed with one palette.
- **Mode triage** (docs/RATING.md): one loop a round, answered A Toss, S Fine, D Good or F Great; the page plays the
  8 bars, then bars 1-2 again, so the seam is heard.
- **Order:** drawn from a fresh 64-bit seed with no two consecutive rounds in one energy x mood cell, as
  lofi_rl.freeze.batch draws the sign-off (its procedure under this batch's own labels): the first drawn order with
  the fewest such neighbours, up to ORDER_TRIES, and the draw says whether the rule held. The seed is recorded only
  in the run's summary/draw.json, which runs/ keeps out of git (CD-21).
- **Ids:** `r` + 8 hex per round, `k` + 10 hex per clip, drawn again while one holds a word of a hidden value or a
  10-hex stretch of a real clip id or a pcm_sha256 (lofi_rl.drafts.batch's leak rule, RATING.md), and here of the
  other digests the hidden manifest holds too.
- **Public manifest:** the scene, a plain note, and each round's bpm and clip (opaque id, file, sha256). Nothing else.
- **Hidden manifest:** category `gate1`, every round kind `none` (no hidden controls, as keep3 and the sign-off); per
  clip the real clip id, the sample index and its brief, the loop text's sha256 (the completion as the model wrote
  it) and the canonical sha, the verbatim flags (vamp, keys, A, B, and `verbatim_line`: A or B), the copy Jaccard and
  the example it was measured against, the render's pcm_sha256, LUFS and true peak, the FLAC's sha256, the run, its
  job id and the palette label.
- **Files:** written into <inbox>/.staging-*, checked by `python -m lofi_rl.ui.manifest check` in a fresh
  interpreter, then moved into place. An existing batch folder is never overwritten, and neither is a run's
  summary/draw.json or summary/batch.json.
- **One ear check per run:** the gate counts the batch the run's summary/batch.json records (lofi_rl.gate1.gatefile).
  So the batch refuses a run whose summary records one already (draw.json or batch.json), and a run whose ear check
  the inbox already holds: a gate1 batch, under any name or date, whose hidden manifest names this run's job
  (`run_batches`). A second ear check of one run is Shresth's call, and nothing here builds one.
- **Name:** `b<MMDD>-gate1`, then `b<MMDD>-gate1-<k>` for the inbox's k-th gate1 batch over every date.

Exit 0 built, 3 refused because the run does not pass check, 2 bad input or a refusal of the clips, with nothing
written. Sound claims: none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
from collections.abc import Callable, Mapping
from datetime import date as _date
from pathlib import Path

from lofi_rl import gate1 as G
from lofi_rl.brief import Brief, sample_brief
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.drafts.batch import ORDER_TRIES, SCENE, BuildError, Rng, _draw_id, _secrets, check_folders
from lofi_rl.drafts.job import _seed, today_pacific
from lofi_rl.drafts.render import clip_id
from lofi_rl.freeze.batch import HEARD_BARS, round_seconds
from lofi_rl.gate1 import check as C
from lofi_rl.prng import MASK
from lofi_rl.rl.junk import brief_dict
from lofi_rl.ui.manifest import (BATCH_RE, HIDDEN_NAME, LEAK_WINDOW, PUBLIC_NAME, SCHEMA_VERSION, TEXT_MAX, audio_info,
                                 sha256_file)

MODE = "triage"
DRAW_SCHEMA = "lofi-rl/gate1-draw/1"
BATCH_SCHEMA = "lofi-rl/gate1-batch/1"
BARS, SR, CHANNELS = 8, 44100, 2
FULL_SCALE = 32767               # render.formats.to_int16: sample = rint(32767 x + dither)
MEASURE_TOL = 0.002              # LU and dB: clips.json's 3 decimals (0.0005) plus the 16-bit copy (<= 3.2e-4 seen)
NAME_RE = re.compile(r"b\d{4}-gate1(?:-(?P<k>\d+))?")
GATES_PASSED = {"G1-G6": "pass", "G7": "pass"}
Measure = Callable[[Path], tuple[float, float]]


def batch_name(day: _date, inbox: Path | str, name: str | None = None) -> str:
    """`name` if given; else `b<MMDD>-gate1` for the inbox's first gate1 batch and `b<MMDD>-gate1-<k>` for its k-th
    (counting every date's), skipping a name that exists (lofi_rl.freeze.batch.batch_name's rule)."""
    if name:
        return name
    inbox = Path(inbox)
    earlier = [p for p in inbox.glob("*/b*-gate1*") if p.is_dir() and NAME_RE.fullmatch(p.name)]
    k = len(earlier) + 1
    while True:
        cand = f"b{day:%m%d}-gate1" if k == 1 else f"b{day:%m%d}-gate1-{k}"
        if not (inbox / day.isoformat() / cand).exists():
            return cand
        k += 1


def run_batches(inbox: Path | str, run: G.Run) -> list[str]:
    """`<date>/<batch>` of every gate1 batch in the inbox that holds this run's clips, in path order: its hidden
    manifest (category gate1) names the run's job, or the run's folder name when the run records no job. Whatever the
    batch's name, so a batch built under --name is found too. A hidden manifest that does not read is not this
    module's to judge, and is passed over."""
    out = []
    for hid in sorted(Path(inbox).glob(f"*/*/{HIDDEN_NAME}")):
        try:
            doc = json.loads(hid.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        rounds = doc.get("rounds") if isinstance(doc, dict) and doc.get("category") == G.CATEGORY else None
        infos = [info for hr in (rounds.values() if isinstance(rounds, dict) else ()) if isinstance(hr, dict)
                 and isinstance(hr.get("clips"), dict) for info in hr["clips"].values() if isinstance(info, dict)]
        if any((str(info.get("job")) == run.job) if run.job is not None else (info.get("run") == run.name)
               for info in infos):
            out.append(f"{hid.parent.parent.name}/{hid.parent.name}")
    return out


def draw_order(cells: Mapping[str, tuple[str, str]], seed: int) -> dict:
    """The rounds' serving order (module doc). `cells`: clip key -> its brief's (energy, mood), in sample order."""
    ids = list(cells)
    rng = Rng(seed, "gate1", "order")
    order, neighbours = list(ids), None
    for _ in range(ORDER_TRIES):
        cand = rng.shuffle(list(ids))
        k = sum(cells[a] == cells[b] for a, b in zip(cand, cand[1:]))
        if neighbours is None or k < neighbours:            # the first drawn order with the fewest same-cell pairs
            order, neighbours = cand, k
        if k == 0:
            break
    return {"schema": DRAW_SCHEMA, "seed": f"{seed & MASK:016x}", "order": order,
            "cells": {key: list(cells[key]) for key in order}, "order_constraint_met": not neighbours,
            "same_cell_neighbours": neighbours or 0}


def note_text(k: int) -> str:
    """The page's note: plain words, no option or hidden value, no claim about sound."""
    return f"Phase-1 gate: {k} loops, one a round. Toss, Fine, Good or Great?"


def push_text(name: str, k: int, job: str | None, est_min: float) -> str:
    return (f"Needs ear check: {name}. {k} loops from the Phase-1 gate run (job {job}), one a round in triage, about "
            f"{est_min:g} min.")


def run_seed(run: G.Run) -> int:
    """The brief stream's seed junk.json records (check holds it to 2026): lofi_rl.gate1.brief_seed."""
    return G.brief_seed(run.junk)


def why_not(rec: Mapping) -> str:
    """Why a clips.json record has no clip to hear."""
    if rec.get("skipped"):
        return f"skipped: {rec['skipped']}"
    if rec.get("gate"):
        return f"gate {rec['gate']}: {str(rec.get('detail') or '').strip()[:200]}"
    if rec.get("error"):
        return f"error: {str(rec['error']).strip()[:200]}"
    return "not rendered"


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# --------------------------------------------------------------------------------------------- the render's values
def measure_flac(path: Path) -> tuple[float, float]:
    """(integrated LUFS, true peak in dBTP) of a listening copy, decoded at the full scale render.formats wrote it at
    and measured with the render's own functions. Scorer-side imports, here and not at the top: the policy env's
    import walk (jobs/env_policy.sbatch) imports this module without numpy's audio stack."""
    import numpy as np
    import soundfile as sf

    from lofi_rl.render.loudness import lufs
    from lofi_rl.render.pipeline import loop_true_peak_dbtp
    q, sr = sf.read(str(path), dtype="int16", always_2d=True)
    x = np.ascontiguousarray(q.T, dtype=np.float64) / FULL_SCALE
    return float(lufs(x, int(sr))), float(loop_true_peak_dbtp(x))


def render_config() -> dict:
    """The schema and chain a render manifest records for junk's render step (render.pipeline.render at its default
    cycles), for the real clip id of a clip copied without its manifest."""
    from lofi_rl.render import pipeline as P
    return {"schema": P.SCHEMA, "chain": {"cycles": P.CYCLES, "fluid_preroll_cycles": P.PREROLL,
                                          "midi_cycles": P.PREROLL + P.CYCLES, "pedalboard_buffer": P.BUFFER,
                                          "limiter_tile": P.LIMITER_TILE}}


def flac_problems(flac: Path, rec: Mapping, at: str, measure: Measure) -> tuple[int | None, list[str]]:
    """The FLAC against what clips.json records of it (module doc): (its size, []) or (size or None, its problems)."""
    errs: list[str] = []
    try:
        st = os.lstat(flac)
    except FileNotFoundError:
        return None, [f"{at}: {flac} is missing"]
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        return None, [f"{at}: {flac} is not a plain file"]
    size = st.st_size
    if size != rec.get("flac_bytes"):
        errs.append(f"{at}: {flac.name} is {size} bytes; clips.json records {rec.get('flac_bytes')!r}")
    try:
        sr, ch, frames = audio_info(flac)
    except (OSError, ValueError, struct.error) as err:
        return size, errs + [f"{at}: {flac.name}: {err}"]
    if (sr, ch, frames) != (SR, CHANNELS, rec.get("n")):
        errs.append(f"{at}: {flac.name} holds {frames} frames, {ch} channels at {sr} Hz; clips.json records "
                    f"{rec.get('n')!r} frames, and a clip is {CHANNELS} channels at {SR} Hz")
    if "flac_sha256" in rec and sha256_file(flac) != rec.get("flac_sha256"):
        errs.append(f"{at}: {flac.name}'s sha256 is not the {rec.get('flac_sha256')!r} clips.json records")
    if errs:
        return size, errs
    try:
        got_lufs, got_peak = measure(flac)
    except Exception as err:  # noqa: BLE001: a FLAC that does not decode is a refusal, never a crash
        return size, [f"{at}: {flac.name} does not decode ({type(err).__name__}: {err})"]
    for what, got, want in (("lufs", got_lufs, rec.get("lufs")),
                            ("true_peak_dbtp", got_peak, rec.get("true_peak_dbtp"))):
        if not (_num(want) and _num(got) and abs(got - want) <= MEASURE_TOL):
            errs.append(f"{at}: {flac.name} measures {what} {got:.4f}; clips.json records {want!r} (within "
                        f"{MEASURE_TOL}): not the clip the render step wrote")
    return size, errs


def manifest_problems(side: Path, rec: Mapping, run: G.Run, want: Mapping, at: str) -> tuple[dict | None, list[str]]:
    """The render's manifest beside the FLAC, if it was copied (module doc): (it or None, its problems)."""
    try:
        m = json.loads(side.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, []
    except (OSError, UnicodeDecodeError, ValueError) as err:
        return None, [f"{at}: {side.name} is not readable JSON ({err})"]
    if not isinstance(m, dict):
        return None, [f"{at}: {side.name} is not a JSON object"]
    loud = m.get("loudness") if isinstance(m.get("loudness"), dict) else {}
    mb = m.get("brief") if isinstance(m.get("brief"), dict) else {}
    pal = m.get("palette") if isinstance(m.get("palette"), dict) else {}
    loop = m.get("loop") if isinstance(m.get("loop"), dict) else {}
    checks = (
        ("audio_sha256", m.get("audio_sha256"), rec.get("audio_sha256")),
        ("n", m.get("n"), rec.get("n")),
        ("sr", m.get("sr"), SR),
        ("loop.canon_sha256", loop.get("canon_sha256"), rec.get("canon_sha256")),
        ("palette", [pal.get("id"), pal.get("sha256")], [run.palette.get("id"), run.palette.get("sha256")]),
        ("brief", {k: mb.get(k) for k in ("energy", "mood", "bpm", "tonic")},
         {k: want[k] for k in ("energy", "mood", "bpm", "tonic")}),
        ("loudness.lufs", round(loud["lufs"], 3) if _num(loud.get("lufs")) else None, rec.get("lufs")),
        ("loudness.true_peak_dbtp", round(loud["true_peak_dbtp"], 3) if _num(loud.get("true_peak_dbtp")) else None,
         rec.get("true_peak_dbtp")),
        ("gates", m.get("gates"), GATES_PASSED),
    )
    errs = [f"{at}: {side.name} {what} is {got!r}, the run records {need!r}" for what, got, need in checks
            if not _same(got, need)]
    return m, errs


def clip_item(run: G.Run, rec: Mapping, clips_dir: Path, measure: Measure) -> tuple[dict | None, list[str]]:
    """One rendered clip, checked against the run's records (module doc): (its item, []) or (None, its problems)."""
    i = rec["i"]
    at = f"sample {i:04d}"
    errs: list[str] = []
    s, row = run.scored[i], run.rows[i]
    if Path(str(rec.get("clip") or "")).name != f"{i:04d}.flac":
        errs.append(f"{at}: clips.json names {rec.get('clip')!r}, not clips/{i:04d}.flac")
    seed = run_seed(run)
    want = brief_dict(sample_brief(seed, i))
    if not _same(rec.get("brief"), want) or not _same(row.get("brief"), want):
        return None, errs + [f"{at}: its brief is not brief {i} of the measurement stream (seed {seed}): clips.json "
                             f"{rec.get('brief')}, completions.jsonl {row.get('brief')}"]
    n = loop_samples(want["bpm"])
    if rec.get("n") != n:
        errs.append(f"{at}: clips.json records n {rec.get('n')!r}; one pass at {want['bpm']} BPM is {n} samples")
    for key in ("audio_sha256", "canon_sha256"):
        if not (isinstance(rec.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", rec[key])):
            errs.append(f"{at}: clips.json's {key} {rec.get(key)!r} is not a sha256")
    if s.get("canon_sha256") != rec.get("canon_sha256"):        # score and render parse one text with one palette
        errs.append(f"{at}: scored.jsonl's canon_sha256 {s.get('canon_sha256')!r} is not the render's "
                    f"{rec.get('canon_sha256')!r} (clips.json)")
    if errs:
        return None, errs
    flac = clips_dir / f"{i:04d}.flac"
    size, errs = flac_problems(flac, rec, at, measure)
    m, more = manifest_problems(clips_dir / f"{i:04d}.manifest.json", rec, run, want, at)
    errs += more
    if errs:
        return None, errs
    b = Brief(energy=want["energy"], mood=want["mood"], bpm=want["bpm"], tonic=want["tonic"])
    cfg = m if m is not None else render_config()
    try:
        real = clip_id({"schema": cfg["schema"], "chain": cfg["chain"], "loop": {"canon_sha256": rec["canon_sha256"]},
                        "brief": {"bpm": b.bpm, "tonic": b.tonic, "mode": b.mode, "energy": b.energy},
                        "palette": {"id": run.palette.get("id"), "sha256": str(run.palette.get("sha256"))}})
    except (KeyError, TypeError) as err:
        return None, [f"{at}: {flac.name}'s manifest lacks what the real clip id needs ({type(err).__name__}: {err})"]
    return {"i": i, "brief": want, "brief_id": f"s{seed}-{i:04d}", "key": b.key_name,
            "text_sha256": hashlib.sha256(str(row.get("text", "")).encode("utf-8")).hexdigest(),
            "canon_sha256": rec["canon_sha256"], "verbatim": dict(s["verbatim"]), "verbatim_line": G.verbatim_line(s),
            "copy_jaccard": s.get("copy_jaccard"), "copy_reference": s.get("copy_reference"),
            "pcm_sha256": rec["audio_sha256"], "lufs": rec["lufs"], "true_peak_dbtp": rec["true_peak_dbtp"],
            "n": n, "flac": flac, "flac_bytes": size, "flac_sha256": sha256_file(flac), "clip": real,
            "manifest": m is not None}, []


def hidden_clip(item: Mapping, run: G.Run) -> dict:
    """What a clip really is (module doc): plain values only, as ui.manifest requires."""
    b, v = item["brief"], item["verbatim"]
    return {"clip": item["clip"], "sample": item["i"], "brief": item["brief_id"], "energy": b["energy"],
            "mood": b["mood"], "bpm": b["bpm"], "tonic": b["tonic"], "key": item["key"],
            "text_sha256": item["text_sha256"], "canon_sha256": item["canon_sha256"],
            **{f"verbatim_{h}": v.get(h) is True for h in ("vamp", "keys", "A", "B")},
            "verbatim_line": item["verbatim_line"], "copy_jaccard": item["copy_jaccard"],
            "copy_reference": item["copy_reference"], "pcm_sha256": item["pcm_sha256"],
            "flac_sha256": item["flac_sha256"], "lufs": item["lufs"], "true_peak_dbtp": item["true_peak_dbtp"],
            "run": run.name, "job": run.job, "palette": run.palette_label}


def _avoid(hidden: Mapping[str, Mapping]) -> set[str]:
    """What no opaque id may hold: drafts.batch's leak rule over every hidden clip, plus every 10-hex stretch of the
    other digests (text, canonical loop, FLAC)."""
    out = _secrets({"brief_id": info["brief"], "clips": {key: info}} for key, info in hidden.items())
    for info in hidden.values():
        for key in ("text_sha256", "canon_sha256", "flac_sha256"):
            raw = str(info[key])
            out |= {raw[k:k + LEAK_WINDOW] for k in range(len(raw) - LEAK_WINDOW + 1)}
    return out


def build(run: G.Run, clips_dir: Path | str, *, name: str, date: str, out_root: Path | str, seed: int,
          summary_dir: Path | str | None = None, check: bool = True, measure: Measure = measure_flac) -> Path:
    """The batch of `run`'s rendered clips into <out_root>/<date>/<name>/ (module doc); the batch folder. BuildError,
    before anything is written, lists every problem."""
    clips_dir, out_root = Path(clips_dir), Path(out_root)
    summary_dir = Path(summary_dir) if summary_dir is not None else run.dir / "summary"
    errors: list[str] = []
    if not BATCH_RE.fullmatch(name):
        errors.append(f"batch name {name!r} must match {BATCH_RE.pattern}")
    target = out_root / date / name
    if target.exists():
        errors.append(f"{target} exists; a built batch is never overwritten")
    for fname in ("draw.json", "batch.json"):
        p = summary_dir / fname
        if p.exists():
            try:
                was = json.loads(p.read_text(encoding="utf-8")).get("batch")
            except (OSError, ValueError, AttributeError):
                was = None
            errors.append(f"{p} exists{f' (batch {was})' if was else ''}: this run's ear check was built already, "
                          "and its record is never overwritten. The gate counts one ear check per run, the batch "
                          "this record names; a second one is Shresth's call")
    earlier = run_batches(out_root, run)
    if earlier:
        errors.append(f"{out_root} holds this run's ear check already ({', '.join(earlier)}): the gate counts one ear "
                      "check per run, so a second one is Shresth's call")
    if run.clips is None:
        errors.append(f"{run.dir / 'summary' / 'clips.json'} is missing: the batch is built from it")
    if not clips_dir.is_dir():
        errors.append(f"--clips {clips_dir} is not a folder")
    if errors:
        raise BuildError(errors)
    items, left_out = [], []
    for rec in run.clip_records:
        if rec.get("ok") is not True:
            left_out.append({"i": rec["i"], "why": why_not(rec)})
            continue
        item, errs = clip_item(run, rec, clips_dir, measure)
        errors += errs
        if item is not None:
            items.append(item)
    recorded = {r["i"] for r in run.clip_records}
    left_out += [{"i": i, "why": "not recorded: the render step ended before it"} for i in run.sample
                 if i not in recorded]
    if not items and not errors:
        errors.append("no clip rendered: there is nothing to hear")
    note = note_text(len(items))
    if len(note) > TEXT_MAX:
        errors.append(f"the note is {len(note)} characters; the page shows at most {TEXT_MAX}")
    if errors:
        raise BuildError(errors)

    by_key = {f"{it['i']:04d}": it for it in items}
    draw = draw_order({key: (it["brief"]["energy"], it["brief"]["mood"]) for key, it in by_key.items()}, seed)
    hidden_info = {key: hidden_clip(it, run) for key, it in by_key.items()}
    avoid = _avoid(hidden_info)
    rng, used = Rng(seed, "gate1", "ids", name), set()
    pub_rounds, hid_rounds, ledger, files = [], {}, [], []
    for idx, key in enumerate(draw["order"]):
        it = by_key[key]
        rid = _draw_id(rng, "r", 8, avoid, used)
        oid = _draw_id(rng, "k", 10, avoid, used)
        pub_rounds.append({"id": rid, "bpm": it["brief"]["bpm"],
                           "clips": [{"id": oid, "file": f"{oid}.flac", "sha256": it["flac_sha256"]}]})
        hid_rounds[rid] = {"kind": "none", "of": None, "anchor_clip": None, "brief_id": it["brief_id"],
                           "category": G.CATEGORY, "clips": {oid: hidden_info[key]}}
        ledger.append({"id": rid, "index": idx + 1, "sample": it["i"], "brief": it["brief_id"],
                       "clip": {"id": oid, "real": it["clip"], "pcm_sha256": it["pcm_sha256"],
                                "flac_sha256": it["flac_sha256"], "verbatim_line": it["verbatim_line"],
                                "manifest": it["manifest"]}})
        files.append((it["flac"], f"{oid}.flac", it["flac_sha256"]))
    est = round(sum(round_seconds(r["bpm"]) for r in pub_rounds) / 60.0, 1)
    public = {"v": SCHEMA_VERSION, "batch": name, "mode": MODE, "scene": SCENE, "note": note, "bars": BARS, "sr": SR,
              "est_min": est, "rounds": pub_rounds}
    hidden = {"v": SCHEMA_VERSION, "batch": name, "category": G.CATEGORY, "rounds": hid_rounds}

    out_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=out_root))
    try:
        d = staging / date / name
        d.mkdir(parents=True)
        for src, fname, sha in files:
            shutil.copyfile(src, d / fname)
            if sha256_file(d / fname) != sha:
                raise BuildError([f"{src} changed while the batch was built"])
        (d / HIDDEN_NAME).write_text(json.dumps(hidden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (d / PUBLIC_NAME).write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if check:
            rc, text = check_folders([d])[0]
            if rc != 0:
                raise BuildError([f"ui.manifest check failed: {text}"])
        if target.exists():
            raise BuildError([f"{target} appeared while building"])
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(d, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "draw.json").write_text(json.dumps(draw, indent=1) + "\n", encoding="utf-8")
    reals: dict[str, list[int]] = {}
    for it in items:
        reals.setdefault(it["clip"], []).append(it["i"])
    audio_s = round(sum(HEARD_BARS * 240.0 / r["bpm"] for r in pub_rounds), 1)
    doc = {"schema": BATCH_SCHEMA, "batch": name, "date": date, "dir": f"{date}/{name}", "mode": MODE,
           "category": G.CATEGORY, "run": run.name, "job": run.job, "palette": run.palette_label,
           "sampled": len(run.sample), "recorded": len(run.clip_records), "rendered": len(items),
           "left_out": left_out, "without_manifest": [it["i"] for it in items if not it["manifest"]],
           "same_real_clip": [v for v in reals.values() if len(v) > 1],
           "order_constraint_met": draw["order_constraint_met"], "same_cell_neighbours": draw["same_cell_neighbours"],
           "est_min": est, "audio_s": audio_s, "note": note, "push": push_text(name, len(items), run.job, est),
           "rounds": ledger}
    (summary_dir / "batch.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return target


def main(a: argparse.Namespace) -> int:
    """The `batch` subcommand (lofi_rl.gate1.cli)."""
    from lofi_rl.ui.serve import inbox_dir_from_env
    try:
        run = G.load_run(a.run)
    except G.Gate1Error as err:
        print(f"error: bad input, nothing written:\n{err}", file=sys.stderr)
        return 2
    doc = C.evaluate(run)
    if doc["exit"] != 0:
        print(f"batch: refused, nothing written: {run.name} does not pass check ({doc['verdict']}). An invalid run "
              "decides nothing (D-020), and under 300 the gate fails whatever the ear says (D-021), so an ear check "
              f"would cost his listening and decide nothing. See: python -m lofi_rl.gate1 check {a.run}",
              file=sys.stderr)
        return 3
    try:
        day = _date.fromisoformat(a.date) if a.date else today_pacific()
        seed = _seed(a.seed)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    inbox = a.inbox if a.inbox is not None else inbox_dir_from_env()
    name = batch_name(day, inbox, a.name)
    try:
        folder = build(run, a.clips, name=name, date=day.isoformat(), out_root=inbox, seed=seed)
    except BuildError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    rec = json.loads((run.dir / "summary" / "batch.json").read_text(encoding="utf-8"))
    print(f"batch {name}: {rec['rendered']} triage rounds of {rec['sampled']} sampled clips, est {rec['est_min']} min "
          f"-> {folder}")
    for x in rec["left_out"]:
        print(f"  left out: sample {x['i']:04d} ({x['why']})")
    if rec["without_manifest"]:
        print(f"  {len(rec['without_manifest'])} clip(s) came without their render manifest; their real clip ids use "
              "lofi_rl.render.pipeline's schema and chain")
    if rec["same_real_clip"]:
        print(f"  the same real clip in more than one round: samples {rec['same_real_clip']}")
    if not rec["order_constraint_met"]:
        print(f"  order: {rec['same_cell_neighbours']} pair(s) of consecutive rounds share a cell (no drawn order "
              "avoided it)")
    print(f"push: {rec['push']}")
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("batch", help="his triage batch of the run's clips, for the ear check (only for a run that "
                                     "passes check)")
    p.add_argument("run", type=Path, help="the run folder, runs/junk-<job> (its summary/ is read, and draw.json and "
                                          "batch.json are written there)")
    p.add_argument("--clips", type=Path, required=True,
                   help="the folder holding the run's <i>.flac (and, when copied, <i>.manifest.json)")
    p.add_argument("--inbox", type=Path, default=None, help="the inbox root (default $LOFI_INBOX or inbox/)")
    p.add_argument("--date", default=None, help="the inbox date folder, YYYY-MM-DD (default: today, Pacific)")
    p.add_argument("--name", default=None, help="the batch name (default b<MMDD>-gate1, then b<MMDD>-gate1-<k>)")
    p.add_argument("--seed", default=None, help="tests only: the draw's seed, 16 hex digits (default: a fresh one)")
    p.set_defaults(run_command=main)
