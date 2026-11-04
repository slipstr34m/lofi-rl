"""The sign-off batch (docs/DRAFTS.md §7; the freeze spec §4.4): the kept loops with the frozen palette, one triage round
each, for Shresth's sign-off listen.

- **Mode triage** (docs/RATING.md): one loop per round, answered A Toss, S Fine, D Good or F Great (TASTE.md, Tiers; a
  Toss is "a dealbreaker fired"). The page plays the 8 bars, then bars 1-2 again, so he hears the loop wrap (TASTE.md:
  ear checks play two passes, so the seam is audible). Each clip is one pass of exactly N samples: the frozen render,
  with the bed, as its 16-bit TPDF-dithered FLAC (RENDER.md §3.6).
- **Order**: drawn from a fresh 64-bit seed, with no two consecutive rounds in the same energy x mood cell: keep3's rule
  and its keyed SplitMix64 stream (lofi_rl.drafts.batch.Rng), under this batch's own labels. When no drawn order manages
  it, the order is the first drawn one with the fewest such neighbours, and the draw says so. The seed is recorded only
  in the run's summary/draw.json, never committed (CD-21).
- **Ids**: `r` + 8 hex per round and `k` + 10 hex per clip, drawn again whenever one would hold a word of a hidden
  value or a 10-hex stretch of a real clip id or a pcm_sha256 (the rating page's leak check, RATING.md).
- **Note**: plain words naming the one change from keep3, where the same loops played with no bed: the rain he picked
  in Day 3 at its level, a pairing Day 3 never played (W1 played the excerpt at another level and W2 the level on another
  excerpt, shootouts/plan.yaml), and the bed as measured after the final limiter in these loops. It names no option, so
  the leak check passes it. A level change of his (the freeze spec §4.2) is named too. A later sign-off (an earlier
  listen is in lofi_rl.freeze.SIGNOFFS: the loops reopened, docs/FREEZE.md §5) says "again" instead: that listen
  played this rain at this level, and redo picks may be among the kept loops, so neither "new to your ears" nor "the
  same loops" holds.
- **Hidden manifest**: category `signoff`, every round kind `none` (no hidden controls, as keep3: DRAFTS.md §8); per clip
  the real clip id, the draft file and its sha256, the persona and brief id, the knobs, tempo and key, the instruments,
  the palette label (`<id>@<sha8>` of the signed-off file), the render id, pcm_sha256, LUFS, true peak, and the bed as
  placed and as heard.
- **Files**: as keep3's (lofi_rl.drafts.batch): written into <out>/.staging-*, checked by `python -m
  lofi_rl.ui.manifest check` in a fresh interpreter, and only then moved into place; an existing batch folder is never
  overwritten. summary/draw.json and summary/batch.json (each round's id, brief and clip, the note and the push text)
  go to the run's summary.
- **Names**: `b<MMDD>-signoff`, then `b<MMDD>-signoff-<k>` for the k-th sign-off batch in the inbox (the freeze spec
  §4.5: a new sign-off after a change; the old batch stays as it was).

Sound claims: none.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from datetime import date as _date
from pathlib import Path

from lofi_rl.brief import Brief
from lofi_rl.drafts.batch import (ORDER_TRIES, SCENE, BuildError, Rng, _draw_id, _secrets, check_folders,
                                  check_record)
from lofi_rl.prng import MASK
from lofi_rl.ui.manifest import BATCH_RE, HIDDEN_NAME, PUBLIC_NAME, SCHEMA_VERSION, TEXT_MAX

MODE = "triage"
CATEGORY = "signoff"
DRAW_SCHEMA = "lofi-rl/signoff-draw/1"
BATCH_SCHEMA = "lofi-rl/signoff-batch/1"
BARS, SR = 8, 44100
HEARD_BARS = 10          # a triage round plays the 8 bars, then bars 1-2 again (RATING.md, "What each kind ... asks")
MARK_S = 8.0             # seconds to mark and save a round: keep3's estimate (lofi_rl.drafts.batch.round_seconds)
NAME_RE = re.compile(r"b\d{4}-signoff(?:-(?P<k>\d+))?")
CHANGE_WORDS = "the keys at the levels you gave"     # the note's words for a level change of his (spec §4.2)


def batch_name(day: _date, inbox: Path | str, name: str | None = None) -> str:
    """`name` if given; else `b<MMDD>-signoff` for the inbox's first sign-off batch and `b<MMDD>-signoff-<k>` for its
    k-th (counting every date's), skipping a name that exists."""
    if name:
        return name
    inbox = Path(inbox)
    earlier = [p for p in inbox.glob("*/b*-signoff*") if p.is_dir() and NAME_RE.fullmatch(p.name)]
    k = len(earlier) + 1
    while True:
        cand = f"b{day:%m%d}-signoff" if k == 1 else f"b{day:%m%d}-signoff-{k}"
        if not (inbox / day.isoformat() / cand).exists():
            return cand
        k += 1


def draw_order(cells: Mapping[str, tuple[str, str]], seed: int) -> dict:
    """The rounds' serving order (module doc). `cells`: render id -> its brief's (energy, mood), in brief order."""
    ids = list(cells)
    rng = Rng(seed, "signoff", "order")
    order, neighbours = list(ids), None
    for _ in range(ORDER_TRIES):
        cand = rng.shuffle(list(ids))
        k = sum(cells[a] == cells[b] for a, b in zip(cand, cand[1:]))
        if neighbours is None or k < neighbours:            # the first drawn order with the fewest same-cell pairs
            order, neighbours = cand, k
        if k == 0:
            break
    return {"schema": DRAW_SCHEMA, "seed": f"{seed & MASK:016x}", "order": order,
            "cells": {rid: list(cells[rid]) for rid in order}, "order_constraint_met": not neighbours,
            "same_cell_neighbours": neighbours or 0}


def _below(heard: tuple[float, float]) -> str:
    lo, hi = heard
    return f"{lo:.1f}-{hi:.1f}"


def note_text(kept: int, bed: Mapping, heard: tuple[float, float], change: bool = False, *, again: bool = False) -> str:
    """The page's note. `bed`: the frozen bed's `delta_lu`, and `w1_delta_lu`, the level Day 3 played its excerpt at;
    `heard`: the range, in LU below the music, of the bed as measured after the final limiter in these loops. `again`:
    an earlier sign-off listen played this rain at this level (lofi_rl.freeze.SIGNOFFS is not empty), so the note
    neither calls it new to his ears nor the loops the same ones (redo picks may be among them)."""
    d, w1 = bed["delta_lu"], bed["w1_delta_lu"]
    if again:
        what = f"and {CHANGE_WORDS}" if change else "as in your last sign-off listen"
        return (f"Sign-off again: your {kept} kept loops, with the rain you picked in Day 3 at {d:g} below the music, "
                f"{what}. After the final limiter it measures {_below(heard)} below in these loops. Toss, Fine, Good "
                f"or Great?")
    tail = (f"This rain at this level is new to your ears: Day 3 played it at {w1:g} below, and {d:g} below on another "
            f"rain. After the final limiter it measures {_below(heard)} below")
    if not change:
        return (f"Sign-off: the same {kept} loops you kept, now with the rain you picked in Day 3, at {d:g} below the "
                f"music. {tail} in these loops. Toss, Fine, Good or Great?")
    return (f"Sign-off: your {kept} kept loops, now with the Day-3 rain at {d:g} below the music and {CHANGE_WORDS}. "
            f"{tail}. Toss, Fine, Good or Great?")


def push_text(name: str, kept: int, bed: Mapping, heard: tuple[float, float], change: bool = False, *,
              again: bool = False) -> str:
    """The push notification after the job (the freeze spec §4.3), in plain words; `again` as in note_text."""
    d, w1 = bed["delta_lu"], bed["w1_delta_lu"]
    extra = " The keys play at the levels you gave." if change else ""
    if again:
        return (f"Needs ear check: {name}. Your {kept} kept loops again, with the rain you picked in Day 3 at {d:g} "
                f"below the music, as in your last sign-off listen.{extra} After the final limiter it measures "
                f"{_below(heard)} below in these {kept}.")
    return (f"Needs ear check: {name}. The same {kept} loops you kept, now with the rain you picked in Day 3, at {d:g} "
            f"below the music.{extra} You have not heard that rain at that level together: Day 3 played it at {w1:g} "
            f"below, and {d:g} below on another rain. After the final limiter it measures {_below(heard)} below in "
            f"these {kept}.")


def round_seconds(bpm: float) -> float:
    """A triage round's listening estimate: the 10 bars it plays, plus the time to mark and save."""
    return HEARD_BARS * 240.0 / bpm + MARK_S


def hidden_clip(rec: Mapping, bed: Mapping) -> dict:
    """What a clip really is (module doc). `bed`: its `excerpt`, `delta_lu`, `placed_lu` and `heard_lu`."""
    b = rec["brief"]
    key = Brief(energy=b["energy"], mood=b["mood"], bpm=b["bpm"], tonic=b["tonic"]).key_name
    return {"clip": rec["clip"], "draft": rec["file"], "draft_sha256": rec["draft_sha256"], "persona": rec["persona"],
            "brief": rec["brief_id"], "energy": b["energy"], "mood": b["mood"], "bpm": b["bpm"], "tonic": b["tonic"],
            "key": key, "keys": rec["keys"], "kit": rec["kit"], "sax": rec.get("sax"),
            "palette": rec["palette"]["label"], "render": rec["render"], "pcm_sha256": rec["pcm_sha256"],
            "lufs": rec["lufs"], "true_peak_dbtp": rec["true_peak_dbtp"], "bed_excerpt": bed["excerpt"],
            "bed_delta_lu": bed["delta_lu"], "bed_placed_lu": bed["placed_lu"], "bed_lu": bed["heard_lu"]}


def build(records: Mapping[str, Mapping], renders_dir: Path | str, *, beds: Mapping[str, Mapping], note: str,
          push: str, name: str, date: str, out_root: Path | str, seed: int, summary_dir: Path | str,
          check: bool = True) -> Path:
    """The sign-off batch of `records` (render id -> the frozen render's record, in brief order) into
    <out_root>/<date>/<name>/, each clip from <renders_dir>/<render id>/render.flac. `beds`: render id -> its bed as
    placed and heard (hidden_clip). Returns the batch folder; BuildError, before anything is written, listing every
    problem."""
    renders_dir, out_root, summary_dir = Path(renders_dir), Path(out_root), Path(summary_dir)
    errors: list[str] = []
    if not BATCH_RE.fullmatch(name):
        errors.append(f"batch name {name!r} must match {BATCH_RE.pattern}")
    target = out_root / date / name
    if target.exists():
        errors.append(f"{target} exists; a built batch is never overwritten")
    if not records:
        errors.append("no records: a sign-off batch holds the kept loops")
    for rid, rec in records.items():
        errors += check_record(renders_dir / rid, rec)
        if rid not in beds:
            errors.append(f"{rid}: no bed measurement")
    reals = [r.get("clip") for r in records.values()]
    if len(set(reals)) != len(reals):
        errors.append("two loops render to one real clip")
    if len(note) > TEXT_MAX:
        errors.append(f"the note is {len(note)} characters; the page shows at most {TEXT_MAX}")
    if errors:
        raise BuildError(errors)

    draw = draw_order({rid: (r["brief"]["energy"], r["brief"]["mood"]) for rid, r in records.items()}, seed)
    hidden_info = {rid: hidden_clip(rec, beds[rid]) for rid, rec in records.items()}
    avoid = _secrets({"brief_id": records[rid]["brief_id"], "clips": {rid: hidden_info[rid]}} for rid in records)
    rng, used = Rng(seed, "signoff", "ids", name), set()
    pub_rounds, hid_rounds, ledger, files = [], {}, [], []
    for i, rid in enumerate(draw["order"]):
        rec = records[rid]
        round_id = _draw_id(rng, "r", 8, avoid, used)
        oid = _draw_id(rng, "k", 10, avoid, used)
        pub_rounds.append({"id": round_id, "bpm": rec["bpm"],
                           "clips": [{"id": oid, "file": f"{oid}.flac", "sha256": rec["flac_sha256"]}]})
        hid_rounds[round_id] = {"kind": "none", "of": None, "anchor_clip": None, "brief_id": rec["brief_id"],
                                "category": CATEGORY, "clips": {oid: hidden_info[rid]}}
        ledger.append({"id": round_id, "index": i + 1, "brief": rec["brief_id"],
                       "clip": {"id": oid, "draft": rec["file"], "render": rid, "real": rec["clip"],
                                "pcm_sha256": rec["pcm_sha256"], "flac_sha256": rec["flac_sha256"]}})
        files.append((renders_dir / rid / "render.flac", f"{oid}.flac"))
    est = round(sum(round_seconds(r["bpm"]) for r in pub_rounds) / 60.0, 1)
    public = {"v": SCHEMA_VERSION, "batch": name, "mode": MODE, "scene": SCENE, "note": note, "bars": BARS, "sr": SR,
              "est_min": est, "rounds": pub_rounds}
    hidden = {"v": SCHEMA_VERSION, "batch": name, "category": CATEGORY, "rounds": hid_rounds}

    out_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=out_root))
    try:
        d = staging / date / name
        d.mkdir(parents=True)
        for src, fname in files:
            shutil.copyfile(src, d / fname)
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
    audio_s = round(sum(HEARD_BARS * 240.0 / r["bpm"] for r in pub_rounds), 1)
    doc = {"schema": BATCH_SCHEMA, "batch": name, "date": date, "dir": f"{date}/{name}", "mode": MODE,
           "est_min": est, "audio_s": audio_s, "note": note, "push": push, "rounds": ledger}
    (summary_dir / "batch.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return target
