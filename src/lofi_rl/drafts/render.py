"""The drafts job's renders (docs/DRAFTS.md §7): every draft through the real pipeline, with its own render palette.

`render_all(tasks, out_dir, jobs=8, workers=1)` renders each task (a draft: its text, brief and render palette) with
`lofi_rl.render.pipeline.render`: one fresh FluidSynth process per pitched stem, the drum sampler, the bus chains, the
master chain and the loudness loop to -14 LUFS (docs/RENDER.md). The renders run `jobs` at a time in a thread pool with
`workers` each, as the shootout's render stage does (lofi_rl.shootout.render: the pipeline's FluidSynth stems are
subprocesses and its one-shot cache writes through unique temporary files, so concurrent renders share nothing that
races, and the output never depends on `jobs` or `workers`).

Each render gets `<out>/<render id>/`: `loop.ls` (the exact text), `render.wav`, `music.wav`, `manifest.json` (the
render manifest), `render.flac` (the 16-bit TPDF-dithered listening copy) and `record.json`, the line the batch
builder reads. A record's status:
- `ok`: G1-G7 passed, and the loop sits at -14 LUFS within 0.1 LU with its true peak at or below -1 dBTP
  (CLAUDE.md §5, the smoke spec; G7 itself allows 0.5 LU and -0.9 dBTP);
- `level`: it rendered but missed those two numbers;
- `gate` or `compile`: the draft failed a gate or did not compile, with no audio kept;
- `error`: an infrastructure fault (never a gate), with its traceback.

The record also holds the real clip id (taste_pipeline.md §2, the shootout's formula), every bus's trim, fader and
loudness before and after its fader, that loudness relative to the sax target S* and its miss from the bus target
(the freeze compares the kept 10 with the targets, SHOOTOUTS.md §3), the master gain, and the lead's MIDI range.
Sound claims: none.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from lofi_rl.brief import PITCH_NAMES, Brief
from lofi_rl.palette import Palette, load_palette

RECORD_SCHEMA = "lofi-rl/draft-render/1"
RENDERS_SCHEMA = "lofi-rl/draft-renders/1"
FLAC_FORMAT = "pcm16/tpdf-splitmix64"          # the shootout's listening copy (lofi_rl.shootout.render.FLAC_FORMAT)
TARGET_LUFS, LUFS_TOL, PEAK_MAX_DBTP = -14.0, 0.1, -1.0
BUSES = ("sax", "keys", "bass", "drums")        # render/mix.py's summing order
RECORD_KEYS = ("schema", "render", "status", "error", "file", "draft_sha256", "brief_id", "brief", "persona",
               "keys", "kit", "palette", "clip", "n", "sr", "bpm", "lufs", "true_peak_dbtp", "master_gain_db",
               "pcm_sha256", "music_sha256", "flac_sha256", "flac_frames", "canon_sha256", "compiled_sha256", "gates",
               "buses", "sax_range")
Renderer = Callable[..., object]                 # render.pipeline.render's signature


def clip_id(manifest: dict) -> str:
    """The real clip id (taste_pipeline.md §2): "c" + sha256(canonical loop sha | bpm,key,mode,energy | palette
    id@sha8 | render cfg sha8)[:16], the render cfg being the manifest's schema and chain plus the FLAC format."""
    cfg = json.dumps({"schema": manifest["schema"], "chain": manifest["chain"], "flac": FLAC_FORMAT}, sort_keys=True)
    b, p = manifest["brief"], manifest["palette"]
    key = (f"{manifest['loop']['canon_sha256']}|{b['bpm']},{PITCH_NAMES[b['tonic']]},{b['mode']},{b['energy']}|"
           f"{p['id']}@{p['sha256'][:8]}|{hashlib.sha256(cfg.encode()).hexdigest()[:8]}")
    return "c" + hashlib.sha256(key.encode()).hexdigest()[:16]


@lru_cache(maxsize=None)
def _palette(path: str) -> Palette:
    return load_palette(path, verify=False)      # verified once before the renders (job stage `palettes`)


def _trim(pal: Palette, bus: str, kit: str) -> float:
    if bus == "drums":
        return float(pal.kits[kit].lanes["kick"].db)
    chain = pal.mix.buses[bus]
    return float(chain[0].params.get("gain_db", 0.0)) if chain and chain[0].name == "Gain" else 0.0


def _record(task: dict, status: str, error: str | None, pal: Palette | None = None, r=None,
            flac_sha: str | None = None, flac_frames: int | None = None) -> dict:
    rec = {k: None for k in RECORD_KEYS}
    rec.update({"schema": RECORD_SCHEMA, "render": task["render"], "status": status, "error": error,
                "file": task["file"], "draft_sha256": hashlib.sha256(task["text"].encode("utf-8")).hexdigest(),
                "brief_id": task["brief_id"], "brief": dict(task["brief"]), "persona": task["persona"],
                "keys": task.get("keys"), "kit": task.get("kit"), "bpm": task["brief"]["bpm"]})
    if pal is not None:
        rec["palette"] = {"id": pal.id, "sha256": pal.sha256, "label": f"{pal.id}@{pal.sha256[:8]}",
                          "file": Path(task["palette"]).name}
    if r is None:
        return rec
    m = r.manifest
    s_star, targets = task.get("s_star"), dict(pal.mix.bus_target_lu)
    buses = {}
    for b in BUSES:
        pre, fader = m["buses"][b]["lufs"], m["buses"][b]["gain_db"]
        post = None if pre is None else pre + fader
        lu = None if post is None or s_star is None else post - s_star
        buses[b] = {"lufs": pre, "fader_db": fader, "trim_db": _trim(pal, b, r.compiled.kit), "post_lufs": post,
                    "lu": lu, "target_lu": targets.get(b), "off_lu": None if lu is None else lu - targets[b]}
    pitches = [n.pitch for n in r.compiled.sax.notes]
    rec.update({"clip": clip_id(m), "n": m["n"], "sr": m["sr"], "lufs": m["loudness"]["lufs"],
                "true_peak_dbtp": m["loudness"]["true_peak_dbtp"], "master_gain_db": m["loudness"]["gain_db"],
                "pcm_sha256": m["audio_sha256"], "music_sha256": m["music_sha256"], "flac_sha256": flac_sha,
                "flac_frames": flac_frames, "canon_sha256": m["loop"]["canon_sha256"],
                "compiled_sha256": m["loop"]["compiled_sha256"], "gates": m["gates"], "buses": buses,
                "sax_range": [min(pitches), max(pitches)] if pitches else None})
    return rec


def level_problems(rec: dict) -> list[str]:
    """Why a rendered record misses the loudness spec: -14 LUFS within 0.1 LU, true peak at most -1 dBTP."""
    out = []
    if rec["lufs"] is None or abs(rec["lufs"] - TARGET_LUFS) > LUFS_TOL:
        out.append(f"{rec['lufs']} LUFS is not within {LUFS_TOL} LU of {TARGET_LUFS}")
    if rec["true_peak_dbtp"] is None or rec["true_peak_dbtp"] > PEAK_MAX_DBTP:
        out.append(f"true peak {rec['true_peak_dbtp']} dBTP is above {PEAK_MAX_DBTP}")
    return out


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def default_renderer() -> Renderer:
    from lofi_rl.render.pipeline import render     # numpy, scipy, pedalboard: the scorer env and the laptop
    return render


def render_one(task: dict, out_dir: str, workers: int = 1, cache: str | None = None,
               renderer: Renderer | None = None) -> tuple[dict, float]:
    """Render one draft into <out_dir>/<render id>/; (its record, wall seconds). A gate or compile failure is a
    record, never an exception; anything else is status `error`."""
    from lofi_rl.compile.compiler import CompileError
    from lofi_rl.render.formats import write_flac, write_wav
    from lofi_rl.render.pipeline import RenderError, manifest_json
    from lofi_rl.ui.manifest import audio_info, sha256_file
    t0 = time.perf_counter()
    rid = task["render"]
    final, part = Path(out_dir) / rid, Path(out_dir) / f".{rid}.part"
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    (part / "loop.ls").write_text(task["text"], encoding="utf-8")
    pal = None
    try:
        pal = _palette(str(task["palette"]))
        b = task["brief"]
        brief = Brief(energy=b["energy"], mood=b["mood"], bpm=b["bpm"], tonic=b["tonic"])
        r = (renderer or default_renderer())(task["text"], brief, pal, cache=cache, workers=workers)
        write_wav(part / "render.wav", r.audio)
        write_wav(part / "music.wav", r.music)
        (part / "manifest.json").write_text(manifest_json(r.manifest), encoding="utf-8")
        flac = write_flac(part / "render.flac", r.audio)
        _sr, _ch, frames = audio_info(flac)
        rec = _record(task, "ok", None, pal, r, sha256_file(flac), frames)
        bad = level_problems(rec)
        if bad:
            rec.update(status="level", error="; ".join(bad))
    except RenderError as err:
        rec = _record(task, "gate", f"{err.code}: {err.detail}", pal)
    except CompileError as err:
        rec = _record(task, "compile", str(err), pal)
    except Exception:                                   # an infrastructure fault: never a gate, never hidden
        rec = _record(task, "error", traceback.format_exc(limit=8), pal)
    _write_json(part / "record.json", rec)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(part, final)
    return rec, time.perf_counter() - t0


def render_all(tasks: list[dict], out_dir: Path | str, *, jobs: int = 8, workers: int = 1,
               cache: Path | str | None = None, summary_dir: Path | str | None = None,
               renderer: Renderer | None = None) -> dict[str, dict]:
    """Every task, `jobs` at a time; {render id: record}, and summary/renders.json when `summary_dir` is given."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = [t["render"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"render ids repeat: {sorted(i for i in ids if ids.count(i) > 1)}")
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        results = list(ex.map(lambda t: render_one(t, str(out), workers, None if cache is None else str(cache),
                                                   renderer), tasks))
    records = {t["render"]: rec for t, (rec, _) in zip(tasks, results)}
    seconds = {t["render"]: round(s, 2) for t, (_, s) in zip(tasks, results)}
    if summary_dir is not None:
        s = Path(summary_dir)
        s.mkdir(parents=True, exist_ok=True)
        _write_json(s / "renders.json", {"schema": RENDERS_SCHEMA, "renders": records, "seconds": seconds,
                                         "failed": sorted(r for r, rec in records.items() if rec["status"] != "ok")})
    return records
