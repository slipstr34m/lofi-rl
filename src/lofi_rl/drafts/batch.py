"""The keep3 batch of the drafts (docs/RATING.md, keep3; docs/DRAFTS.md §7).

One round per brief, holding that brief's three drafts (one per persona); Shresth keeps one with Q, W or E, or says
"none acceptable" with G, which labels all three Toss and asks for a redraft of that brief (taste_pipeline.md §8).

**Tiles** (`draw_rounds`): a 3x3 Latin square, square[row][tile] = (row + tile) mod 3, whose three symbols stand for the
three personas in a drawn order. The rounds take its rows in turn, each row as often as the round count allows and the
rest drawn without repeats, the rows dealt to the rounds at random: over 9 rounds each persona sits on each tile 3
times; over 10, each persona's tile counts are 4, 3 and 3, and each tile's 4 belongs to a different persona. The
round order is drawn too, with no two consecutive rounds in the same energy x mood cell. Every draw is keyed SplitMix64
(lofi_rl.prng) from a 64-bit seed (`secrets.randbits(64)` at job time, `--seed` in tests), recorded only in the run's
summary/draw.json, never committed (the repo is public: SHOOTOUTS.md §2.3, CD-21).

**Ids**: `r` + 8 hex per round and `k` + 10 hex per clip, drawn per batch from the seed; an id that would hold a word of
any hidden value or a 10-hex stretch of a real clip id or a pcm_sha256 is drawn again (the rating page's leak check,
docs/RATING.md).

**Files**: `<out>/<date>/<batch>/` holds the FLACs under their opaque names, `manifest.json` (public: the scene, each
round's bpm, opaque ids, file hashes) and `manifest.hidden.json` (per round the brief id; per clip the real clip id, the
persona, the draft file and its sha256, the brief's knobs, tempo and key, the instruments, the render palette, the
render id, pcm_sha256, LUFS and true peak). The batch is written into `<out>/.staging-*`, checked by
`python -m lofi_rl.ui.manifest check` in a fresh interpreter, and only then moved into place; an existing batch folder
is never overwritten. `summary/draw.json` and `summary/batch.json` (each round's id, brief, and per tile the opaque id,
persona, draft, render and real clip id) go to the run's summary.

Sound claims: none.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from lofi_rl.brief import Brief
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.prng import GOLDEN, MASK, below, splitmix64
from lofi_rl.ui.manifest import (BATCH_RE, HIDDEN_NAME, LEAK_WINDOW, PUBLIC_NAME, SCHEMA_VERSION, _words, audio_info,
                                 sha256_file)

from .personas import PERSONAS

SCENE = "city at night"
BARS, SR = 8, 44100
SQUARE = tuple(tuple((row + tile) % 3 for tile in range(3)) for row in range(3))
NOTE = "Keep the loop you like best of the three (Q W E), or G if none is acceptable."
CATEGORY = "drafts"
DRAW_SCHEMA = "lofi-rl/draft-draw/1"
BATCH_SCHEMA = "lofi-rl/draft-batch/1"
ORDER_TRIES = 2000
DURATION_TOL = 0.02


class BuildError(ValueError):
    """The build stopped before writing anything; `errors` lists every problem found."""

    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        super().__init__(f"{len(self.errors)} problem(s), nothing written:\n  - " + "\n  - ".join(self.errors))


class Rng:
    """A SplitMix64 stream keyed by the seed and labels (the shootout draw's construction)."""

    def __init__(self, seed: int, *labels: str) -> None:
        h = hashlib.sha256("|".join([f"{seed & MASK:016x}", *labels]).encode()).digest()
        self.state = int.from_bytes(h[:8], "big")

    def next64(self) -> int:
        out = splitmix64(self.state)
        self.state = (self.state + GOLDEN) & MASK
        return out

    def below(self, k: int) -> int:
        return below(self.next64(), k)

    def shuffle(self, xs: list) -> list:
        for i in range(len(xs) - 1, 0, -1):
            j = self.below(i + 1)
            xs[i], xs[j] = xs[j], xs[i]
        return xs

    def hex(self, n: int) -> str:
        return "".join("0123456789abcdef"[self.below(16)] for _ in range(n))


def round_seconds(clips: int, bpm: float) -> float:
    """The plan's listening estimate (taste_pipeline.md §8): 0.75 of auto-play plus 8 s to mark and save."""
    return 0.75 * clips * BARS * 240.0 / bpm + 8.0


def draw_rounds(briefs: Mapping[str, Mapping], seed: int, personas: Sequence[str] = PERSONAS) -> dict:
    """The tiles and the order (module doc). `briefs`: brief id -> {energy, mood, ...}, in brief order. Returns the
    draw document: seed, symbol -> persona, and the rounds in serving order, each {brief, row, tiles: personas}."""
    if len(personas) != 3:
        raise ValueError("keep3 rounds hold three drafts")
    ids = list(briefs)
    rng = Rng(seed, "keep3", "tiles")
    symbols = rng.shuffle(list(personas))
    full, extra = divmod(len(ids), 3)
    rows = [r for _ in range(full) for r in range(3)] + rng.shuffle([0, 1, 2])[:extra]
    rng.shuffle(rows)
    order_rng = Rng(seed, "keep3", "order")
    cell = {b: (briefs[b]["energy"], briefs[b]["mood"]) for b in ids}
    order = list(ids)
    for _ in range(ORDER_TRIES):
        cand = order_rng.shuffle(list(ids))
        if all(cell[a] != cell[b] for a, b in zip(cand, cand[1:])):
            order = cand
            break
    rounds = [{"brief": b, "row": rows[i], "tiles": [symbols[SQUARE[rows[i]][t]] for t in range(3)]}
              for i, b in enumerate(order)]
    return {"schema": DRAW_SCHEMA, "seed": f"{seed & MASK:016x}", "square": [list(r) for r in SQUARE],
            "symbols": symbols, "rounds": rounds}


def tile_counts(draw: Mapping) -> dict[str, list[int]]:
    """How often each persona sits on each tile over the draw's rounds."""
    out = {p: [0, 0, 0] for p in draw["symbols"]}
    for r in draw["rounds"]:
        for t, p in enumerate(r["tiles"]):
            out[p][t] += 1
    return out


def _secrets(hidden_rounds: Iterable[Mapping]) -> set[str]:
    """What an opaque id must not hold, as ui.manifest's leak check reads a hidden manifest: every word of every hidden
    value and brief id, and every 10-hex stretch of every real clip id and pcm_sha256."""
    out: set[str] = set()
    for hr in hidden_rounds:
        out |= _words(hr.get("brief_id") or "")
        for info in hr["clips"].values():
            for key, val in info.items():
                text = str(val or "")
                if key in ("clip", "pcm_sha256"):
                    raw = text[1:] if key == "clip" else text
                    out |= {raw[k:k + LEAK_WINDOW] for k in range(len(raw) - LEAK_WINDOW + 1)}
                if key != "clip":
                    out |= _words(val)
    return out


def _draw_id(rng: Rng, prefix: str, n: int, avoid: set[str], used: set[str]) -> str:
    while True:
        x = prefix + rng.hex(n)
        if x not in used and not any(w in x for w in avoid):
            used.add(x)
            return x


def _hidden_clip(rec: Mapping) -> dict:
    b = rec["brief"]
    key = Brief(energy=b["energy"], mood=b["mood"], bpm=b["bpm"], tonic=b["tonic"]).key_name
    return {"clip": rec["clip"], "persona": rec["persona"], "draft": rec["file"], "draft_sha256": rec["draft_sha256"],
            "brief": rec["brief_id"], "energy": b["energy"], "mood": b["mood"], "bpm": b["bpm"], "tonic": b["tonic"],
            "key": key, "keys": rec["keys"], "kit": rec["kit"], "palette": rec["palette"]["label"],
            "render": rec["render"], "pcm_sha256": rec["pcm_sha256"], "lufs": rec["lufs"],
            "true_peak_dbtp": rec["true_peak_dbtp"]}


def check_record(folder: Path, rec: Mapping) -> list[str]:
    """A render folder fit for the batch: status ok, gates passed, one pass of the loop, its FLAC unchanged."""
    rid = rec.get("render")
    if rec.get("status") != "ok":
        return [f"{rid}: the render is {rec.get('status')}: {str(rec.get('error') or '').strip()[-300:]}"]
    errs = []
    if (rec.get("gates") or {}) != {"G1-G6": "pass", "G7": "pass"}:
        errs.append(f"{rid}: gates {rec.get('gates')}")
    n = loop_samples(rec["bpm"])
    flac = folder / "render.flac"
    try:
        sr, _ch, frames = audio_info(flac)
    except (OSError, ValueError) as err:
        return errs + [f"{rid}: no readable listening copy ({err})"]
    one_pass = BARS * 240.0 / rec["bpm"]
    if rec.get("n") != n or sr != SR or frames != n or abs(frames / sr - one_pass) > DURATION_TOL * one_pass:
        errs.append(f"{rid}: {frames} samples at {sr} Hz, not one pass of {n} at {SR}")
    if sha256_file(flac) != rec.get("flac_sha256"):
        errs.append(f"{rid}: {flac} changed since it was rendered")
    return errs


def check_folders(folders: list[Path]) -> list[tuple[int, str]]:
    """`python -m lofi_rl.ui.manifest check <folder>` for each folder in a fresh interpreter, as the rating page's
    server would load it."""
    src = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    out = []
    for d in folders:
        p = subprocess.run([sys.executable, "-m", "lofi_rl.ui.manifest", "check", str(d)], capture_output=True,
                           text=True, env=env)
        out.append((p.returncode, (p.stdout + p.stderr).strip()))
    return out


def build(records: Mapping[str, Mapping], renders_dir: Path | str, *, briefs: Mapping[str, Mapping], name: str,
          date: str, out_root: Path | str, seed: int, summary_dir: Path | str, check: bool = True) -> Path:
    """The keep3 batch of `records` (render id -> record) into <out_root>/<date>/<name>/. `briefs` names the rounds
    (brief id -> its fields), each of which needs one ok record per persona. Returns the batch folder."""
    renders_dir, out_root, summary_dir = Path(renders_dir), Path(out_root), Path(summary_dir)
    errors: list[str] = []
    if not BATCH_RE.fullmatch(name):
        errors.append(f"batch name {name!r} must match {BATCH_RE.pattern}")
    target = out_root / date / name
    if target.exists():
        errors.append(f"{target} exists; a built batch is never overwritten")
    by_brief: dict[str, dict[str, Mapping]] = {}
    for rid, rec in sorted(records.items()):
        errors += check_record(renders_dir / rid, rec)
        by_brief.setdefault(rec.get("brief_id"), {})
        if rec.get("persona") in by_brief[rec.get("brief_id")]:
            errors.append(f"brief {rec.get('brief_id')}: two drafts of persona {rec.get('persona')}")
        by_brief[rec.get("brief_id")][rec.get("persona")] = rec
    for b in briefs:
        have = sorted(by_brief.get(b, {}))
        if have != sorted(PERSONAS):
            errors.append(f"brief {b}: drafts of {have}, a round needs one of each of {sorted(PERSONAS)}")
    extra = sorted(set(by_brief) - set(briefs))
    if extra:
        errors.append(f"records of briefs {extra} that the batch does not hold")
    if errors:
        raise BuildError(errors)
    for b in briefs:
        reals = [r["clip"] for r in by_brief[b].values()]
        if len(set(reals)) != len(reals):
            errors.append(f"brief {b}: two drafts render to one real clip")
    if errors:
        raise BuildError(errors)

    draw = draw_rounds(briefs, seed)
    hidden_info = {b: {p: _hidden_clip(by_brief[b][p]) for p in PERSONAS} for b in briefs}
    avoid = _secrets({"brief_id": b, "clips": hidden_info[b]} for b in briefs)
    rng, used = Rng(seed, "keep3", "ids", name), set()
    pub_rounds, hid_rounds, ledger, files = [], {}, [], []
    for i, r in enumerate(draw["rounds"]):
        b = r["brief"]
        rid = _draw_id(rng, "r", 8, avoid, used)
        oids = [_draw_id(rng, "k", 10, avoid, used) for _ in r["tiles"]]
        recs = [by_brief[b][p] for p in r["tiles"]]
        pub_rounds.append({"id": rid, "bpm": recs[0]["bpm"],
                           "clips": [{"id": o, "file": f"{o}.flac", "sha256": rec["flac_sha256"]}
                                     for o, rec in zip(oids, recs)]})
        hid_rounds[rid] = {"kind": "none", "of": None, "anchor_clip": None, "brief_id": b, "category": CATEGORY,
                           "clips": {o: hidden_info[b][p] for o, p in zip(oids, r["tiles"])}}
        ledger.append({"id": rid, "index": i + 1, "brief": b, "row": r["row"],
                       "clips": [{"id": o, "pos": t + 1, "persona": p, "draft": rec["file"], "render": rec["render"],
                                  "clip": rec["clip"], "pcm_sha256": rec["pcm_sha256"], "flac_sha256": rec["flac_sha256"]}
                                 for t, (o, p, rec) in enumerate(zip(oids, r["tiles"], recs))]})
        files += [(renders_dir / rec["render"] / "render.flac", f"{o}.flac") for o, rec in zip(oids, recs)]
    est = round(sum(round_seconds(len(r["clips"]), r["bpm"]) for r in pub_rounds) / 60.0, 1)
    public = {"v": SCHEMA_VERSION, "batch": name, "mode": "keep3", "scene": SCENE, "note": NOTE, "bars": BARS, "sr": SR,
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
    doc = {"schema": BATCH_SCHEMA, "batch": name, "date": date, "dir": f"{date}/{name}", "est_min": est,
           "tile_counts": tile_counts(draw), "rounds": ledger}
    (summary_dir / "batch.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return target
