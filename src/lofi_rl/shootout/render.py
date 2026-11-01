"""The `lofi-shootout` job's clip renders (docs/SHOOTOUTS.md sections 2.2, 3 and 7.7).

    python -m lofi_rl.shootout.render --draw <run>/summary/draw.json --palettes <run>/summary/palettes \
        --out <run>/renders [--summary <run>/summary] [--jobs 8] [--workers 1] [--cache DIR]

Renders every render the draw lists, `--jobs` at a time with `--workers` each (8 and 1 on the job's 8 CPUs;
the output never depends on either), through the real pipeline (`render.pipeline.render`, two cycles, the master
chain and -14 LUFS as usual). The renders run in a thread pool: the pipeline already runs its FluidSynth stems and
bus chains in threads, FluidSynth is a subprocess with its own empty HOME, and the one-shot cache writes through
unique temporary files, so concurrent renders share nothing that races; four renders gave the same pcm_sha256 in
threads as in separate processes (laptop, 2026-10-31), and finished in 2.6-3.0 s against 4.1 s. Each render gets its own folder `<out>/<render id>/`:
`loop.ls` (the exact text), `render.wav`, `music.wav`, `manifest.json` (the render manifest), `render.flac` (the
16-bit TPDF-dithered listening copy, RENDER.md section 3.6) and `record.json`, the line the batch builder reads.
A loop that fails a gate or does not compile gets a record with status `gate` or `compile` and no audio, so the
builder stops before writing any batch; an infrastructure fault gets status `error`.

- **Palettes** come from the generator (section 7.8): `<palettes>/<palette key>.yaml`, one per key in the draw's
  `palettes` table, or a render's own `palette_path` (a purged day-3 repeat, made again from its first job's
  palette). Every source file is hashed once per job, not once per palette ("Every palette source is hashed
  once per job"), and every SoundFont preset is checked against its file's headers once per file; then the
  workers load the palettes with `verify=False`.
- **The real clip id** (taste_pipeline.md section 2): "c" + sha256(canonical loop sha | bpm,key,mode,energy |
  palette id@sha8 | render cfg sha8)[:16], the render cfg being the manifest's schema and chain plus the FLAC
  format.
- **Per clip** (section 3, "Recorded per instrument variant ... Per clip"): every bus's trim (a chain-head `Gain`
  for keys, bass and sax; the kit's kick-lane dB for drums, CD-51), fader (`mix.bus_gain_db`), loudness before
  the fader and after it, the master gain, the final LUFS and true peak, and the lead's MIDI range.
- **The clip's own level** (CD-52): a clip palette's entry in the generator's index.json beside it carries a
  `level` (the bus its round varies, the level tool's L before the clip's fader, the fader, and the loudness the
  tool measured after it). The record copies it, and the render must play that very bus: its pre-fader loudness
  within AGREE_LU of L, at the same fader, with the measured loudness within the clip tolerance of S* + T. A render
  that does not gets status `level` (the batch builder refuses it). A palette with no index beside it (a test's
  stand-in) gets no level.
- `summary/renders.json` collects every record, plus the wall time of each render (kept out of the records, so
  a record is as deterministic as its render).

Exit 0 when every render succeeded, 3 when one failed a gate, did not compile or missed its level, 1 on an
infrastructure fault, 2 for bad arguments or palettes. Sound claims: none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from lofi_rl.brief import PITCH_NAMES, Brief
from lofi_rl.compile.compiler import CompileError
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.palette import ROOT, Palette, PaletteError, _file_sha256, load_palette, sf2_presets
from lofi_rl.render.formats import write_flac, write_wav
from lofi_rl.render.mix import BUSES
from lofi_rl.render.pipeline import RenderError, manifest_json, render
from lofi_rl.ui.manifest import audio_info, sha256_file

RECORD_SCHEMA = "lofi-rl/shootout-render/1"
RENDERS_SCHEMA = "lofi-rl/shootout-renders/1"
FLAC_FORMAT = "pcm16/tpdf-splitmix64"
RECORD_KEYS = ("schema", "render", "status", "error", "loop", "brief", "stage", "option", "what", "bus", "variant",
               "palette", "clip", "n", "sr", "bpm", "lufs", "true_peak_dbtp", "master_gain_db", "pcm_sha256",
               "music_sha256", "flac_sha256", "flac_frames", "canon_sha256", "compiled_sha256", "gates", "buses",
               "bed_lufs", "sax_range", "expect_pcm_sha256", "level")
AGREE_LU = 0.001        # a clip's bus before its fader against the level tool's L (both read the same signal)
__all__ = ["AGREE_LU", "RECORD_KEYS", "RECORD_SCHEMA", "clip_id", "loop_samples", "main", "palette_levels",
           "render_all", "verify_palettes", "with_level"]


def clip_id(manifest: dict) -> str:
    """The real clip id of a render (taste_pipeline.md section 2), from its render manifest."""
    cfg = json.dumps({"schema": manifest["schema"], "chain": manifest["chain"], "flac": FLAC_FORMAT}, sort_keys=True)
    b, p = manifest["brief"], manifest["palette"]
    key = (f"{manifest['loop']['canon_sha256']}|{b['bpm']},{PITCH_NAMES[b['tonic']]},{b['mode']},{b['energy']}|"
           f"{p['id']}@{p['sha256'][:8]}|{hashlib.sha256(cfg.encode()).hexdigest()[:8]}")
    return "c" + hashlib.sha256(key.encode()).hexdigest()[:16]


def palette_file(entry: dict, palettes_dir: Path) -> Path:
    return Path(entry["palette_path"]) if entry.get("palette_path") else Path(palettes_dir) / f"{entry['palette']}.yaml"


def verify_palettes(paths: list[Path], root: Path = ROOT) -> list[str]:
    """Load each palette and check its sources: every distinct file hashed once, every SoundFont's presets read
    once. The problems found, empty when all is well."""
    problems: list[str] = []
    want: dict[Path, tuple[str, str]] = {}            # file -> (sha256, first palette naming it)
    pals: dict[Path, Palette] = {}
    for p in paths:
        try:
            pals[p] = pal = load_palette(p, verify=False, root=root)
        except (PaletteError, OSError) as err:
            problems.append(f"{p}: {err}")
            continue
        for src in sorted(pal.sources):
            try:
                f = pal.source_path(src, root)
            except PaletteError as err:
                problems.append(f"{p}: {err}")
                continue
            sha = pal.sources[src].sha256
            if want.setdefault(f, (sha, p.name))[0] != sha:
                problems.append(f"{p}: source {src} ({f}) has sha256 {sha[:12]}..., but {want[f][1]} says "
                                f"{want[f][0][:12]}...")
    for f, (sha, who) in sorted(want.items()):
        if not f.is_file():
            problems.append(f"{f} (named by {who}) is missing")
        elif _file_sha256(f) != sha:
            problems.append(f"{f} (named by {who}) does not match its sha256")
    if problems:
        return problems
    presets: dict[Path, frozenset] = {}
    for p, pal in pals.items():
        for where, src, bank, prog in pal.presets():
            f = pal.source_path(src, root)
            if f not in presets:
                try:
                    presets[f] = sf2_presets(f)
                except (OSError, PaletteError) as err:
                    problems.append(f"{p}: source {src}: {err}")
                    presets[f] = frozenset()
                    continue
            if presets[f] and (bank, prog) not in presets[f]:
                problems.append(f"{p}: {where}: bank {bank} prog {prog} is not a preset of source {src}")
    return problems


@lru_cache(maxsize=None)
def _palette(path: str) -> Palette:
    return load_palette(path, verify=False)          # verified once by the parent (verify_palettes)


def _trim(pal: Palette, bus: str, kit: str) -> float:
    if bus == "drums":
        return float(pal.kits[kit].lanes["kick"].db)
    chain = pal.mix.buses[bus]
    return float(chain[0].params.get("gain_db", 0.0)) if chain and chain[0].name == "Gain" else 0.0


def _record(rid: str, entry: dict, pal: Palette | None, status: str, error: str | None, r=None,
            flac_sha: str | None = None, flac_frames: int | None = None) -> dict:
    rec = {k: None for k in RECORD_KEYS}
    rec.update({"schema": RECORD_SCHEMA, "render": rid, "status": status, "error": error,
                "loop": entry["loop"], "brief": entry["brief"], "stage": entry["stage"], "option": entry["option"],
                "what": entry["what"], "bus": entry["bus"], "variant": entry["variant"],
                "bpm": entry["brief"]["bpm"], "expect_pcm_sha256": entry.get("expect_pcm_sha256")})
    if pal is not None:
        rec["palette"] = {"key": entry["palette"], "id": pal.id, "sha256": pal.sha256,
                          "label": f"{pal.id}@{pal.sha256[:8]}"}
    if r is None:
        return rec
    m = r.manifest
    pitches = [n.pitch for n in r.compiled.sax.notes]
    buses = {}
    for b in BUSES:
        pre, fader = m["buses"][b]["lufs"], m["buses"][b]["gain_db"]
        buses[b] = {"lufs": pre, "fader_db": fader, "trim_db": _trim(pal, b, r.compiled.kit),
                    "post_lufs": None if pre is None else pre + fader}
    rec.update({"clip": clip_id(m), "n": m["n"], "sr": m["sr"], "lufs": m["loudness"]["lufs"],
                "true_peak_dbtp": m["loudness"]["true_peak_dbtp"], "master_gain_db": m["loudness"]["gain_db"],
                "pcm_sha256": m["audio_sha256"], "music_sha256": m["music_sha256"], "flac_sha256": flac_sha,
                "flac_frames": flac_frames, "canon_sha256": m["loop"]["canon_sha256"],
                "compiled_sha256": m["loop"]["compiled_sha256"], "gates": m["gates"], "buses": buses,
                "bed_lufs": m["bed"]["lufs"] if m["bed"] else None,
                "sax_range": [min(pitches), max(pitches)] if pitches else None})
    return rec


def palette_levels(files: list[Path]) -> dict[Path, dict | None]:
    """Each palette's `level` (CD-52) from the generator's index.json in its folder, None for a palette the index
    gives none or with no index beside it (a test's stand-in). Each index is read once. ValueError for an index that
    is not readable JSON."""
    by_dir: dict[Path, dict[str, dict | None]] = {}
    out: dict[Path, dict | None] = {}
    for f in files:
        f = Path(f)
        if f.parent not in by_dir:
            index = f.parent / "index.json"
            try:
                doc = json.loads(index.read_text(encoding="utf-8"))
            except FileNotFoundError:
                doc = {}
            except (OSError, ValueError) as err:
                raise ValueError(f"{index}: {err}") from err
            by_dir[f.parent] = {e["file"]: e.get("level") for e in (doc.get("palettes") or [])}
        out[f] = by_dir[f.parent].get(f.name)
    return out


def with_level(rec: dict, level: dict) -> dict:
    """The record of a clip whose palette names a level (CD-52), with that level, the render's own pre-fader
    loudness of its bus (`render_lufs`) and their difference (`agree_lu`). Status `level` unless the render plays
    the bus the level tool measured (within AGREE_LU), at the clip's fader, and the tool's loudness after that fader
    sits within its tolerance of S* + T."""
    got = rec["buses"][level["bus"]]
    pre, problems = got["lufs"], []
    agree = None if pre is None else pre - level["L"]
    if pre is None:
        problems.append(f"the {level['bus']} bus is silent")
    elif abs(agree) > AGREE_LU:
        problems.append(f"the {level['bus']} bus reads {pre:.4f} LUFS before its fader, the level tool's L is "
                        f"{level['L']:.4f} ({agree:+.4f} LU, more than {AGREE_LU})")
    if got["fader_db"] != level["F"]:
        problems.append(f"the {level['bus']} fader is {got['fader_db']} dB, the clip's own is {level['F']} dB")
    if abs(level["off_lu"]) > level["tol_lu"]:
        problems.append(f"the level tool measured the bus {level['off_lu']:+.3f} LU from S* + T, more than "
                        f"{level['tol_lu']} LU")
    out = dict(rec, level={**level, "render_lufs": pre, "agree_lu": agree, "ok": not problems})
    if problems and out["status"] == "ok":
        out.update(status="level", error="; ".join(problems))
    return out


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def render_one(rid: str, entry: dict, palette_path: str, out_dir: str, workers: int = 1,
               cache: str | None = None, level: dict | None = None) -> tuple[dict, float]:
    """Render one clip into <out_dir>/<rid>/ and return (its record, wall seconds). Never raises for a gate or
    compile failure: the record says so. `level` is the clip palette's level (CD-52), checked by `with_level`."""
    t0 = time.perf_counter()
    final = Path(out_dir) / rid
    part = Path(out_dir) / f".{rid}.part"
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    (part / "loop.ls").write_text(entry["text"], encoding="utf-8")
    pal = None
    try:
        pal = _palette(palette_path)
        b = entry["brief"]
        brief = Brief(energy=b["energy"], mood=b["mood"], bpm=b["bpm"], tonic=b["tonic"])
        r = render(entry["text"], brief, pal, cache=cache, workers=workers)
        write_wav(part / "render.wav", r.audio)
        write_wav(part / "music.wav", r.music)
        (part / "manifest.json").write_text(manifest_json(r.manifest), encoding="utf-8")
        flac = write_flac(part / "render.flac", r.audio)
        _sr, _ch, frames = audio_info(flac)
        rec = _record(rid, entry, pal, "ok", None, r, sha256_file(flac), frames)
        if level is not None:
            rec = with_level(rec, level)
        want = entry.get("expect_pcm_sha256")
        if want and rec["pcm_sha256"] != want:
            rec.update(status="error", error=f"pcm_sha256 {rec['pcm_sha256']} differs from the first showing's {want}")
    except RenderError as err:
        rec = _record(rid, entry, pal, "gate", f"{err.code}: {err.detail}")
    except CompileError as err:
        rec = _record(rid, entry, pal, "compile", str(err))
    except Exception:                                   # an infrastructure fault: never a gate, never hidden
        rec = _record(rid, entry, pal, "error", traceback.format_exc(limit=8))
    _write_json(part / "record.json", rec)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(part, final)
    return rec, time.perf_counter() - t0


def render_all(doc: dict, palettes_dir: Path | str, out_dir: Path | str, *, jobs: int = 8, workers: int = 1,
               cache: Path | str | None = None, summary_dir: Path | str | None = None,
               only: list[str] | None = None) -> dict[str, dict]:
    """Every render of the draw (or `only` those), `jobs` at a time. Returns {render id: record} and writes
    summary/renders.json when `summary_dir` is given."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = {rid: e for rid, e in sorted(doc["renders"].items()) if not only or rid in only}
    files = {rid: palette_file(e, Path(palettes_dir)) for rid, e in todo.items()}
    levels = palette_levels(list(files.values()))
    tasks = [(rid, e, str(files[rid]), str(out), workers, None if cache is None else str(cache), levels[files[rid]])
             for rid, e in todo.items()]
    records: dict[str, dict] = {}
    seconds: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        results = list(ex.map(lambda t: render_one(*t), tasks))
    for (rid, *_), (rec, s) in zip(tasks, results):
        records[rid], seconds[rid] = rec, round(s, 2)
    if summary_dir is not None:
        s = Path(summary_dir)
        s.mkdir(parents=True, exist_ok=True)
        _write_json(s / "renders.json", {"schema": RENDERS_SCHEMA, "renders": records, "seconds": seconds,
                                         "failed": sorted(r for r, rec in records.items() if rec["status"] != "ok")})
    return records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.shootout.render", description=__doc__.split("\n\n")[0])
    p.add_argument("--draw", required=True, type=Path)
    p.add_argument("--palettes", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="the run's renders folder")
    p.add_argument("--summary", type=Path, help="where renders.json goes (default: the draw's folder)")
    p.add_argument("--jobs", type=int, default=8, help="renders at a time")
    p.add_argument("--workers", type=int, default=1, help="workers inside each render")
    p.add_argument("--cache", type=Path, default=None, help="one-shot cache (default $LOFI_CACHE)")
    p.add_argument("--only", default="", help="comma-separated render ids (debugging)")
    a = p.parse_args(argv)
    try:
        doc = json.loads(a.draw.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        print(f"error: --draw {a.draw}: {err}", file=sys.stderr)
        return 2
    only = [x for x in a.only.split(",") if x]
    files = sorted({palette_file(e, a.palettes) for rid, e in doc["renders"].items() if not only or rid in only})
    t0 = time.perf_counter()
    problems = verify_palettes(files)
    if problems:
        print("error: the palettes do not verify:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 2
    try:
        own = sum(x is not None for x in palette_levels(files).values())
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if own:
        print(f"render: {own} of the {len(files)} palettes carry their clip's own level (CD-52)", flush=True)
    print(f"render: {len(files)} palettes verified ({time.perf_counter() - t0:.1f} s); "
          f"{len(only) or len(doc['renders'])} renders, {a.jobs} at a time", flush=True)
    recs = render_all(doc, a.palettes, a.out, jobs=a.jobs, workers=a.workers, cache=a.cache,
                      summary_dir=a.summary or a.draw.parent, only=only or None)
    bad = {rid: r for rid, r in recs.items() if r["status"] != "ok"}
    for rid, r in sorted(bad.items()):
        print(f"render {rid}: {r['status']}: {(r['error'] or '').strip().splitlines()[-1:]}", file=sys.stderr)
    print(f"render: {len(recs) - len(bad)} ok, {len(bad)} failed", flush=True)
    if any(r["status"] == "error" for r in bad.values()):
        return 1
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
