"""The rain excerpt picker (docs/SHOOTOUTS.md section 7.2, section 4.5, CD-20).

    python -m lofi_rl.convert.rain_excerpts <rain at 44.1 kHz> --n 3 --out shootouts/rain_excerpts.json
        [--from <the 48 kHz original>] [--asset-path derived/rain/<file>] [--manifest assets/MANIFEST.yaml]

A bed needs 0.5 s of high-pass warm-up before `start_s`, then N + 1 s, and N is 32 s at 60 BPM (RENDER.md section
3.4), so each candidate is a 33.5 s stretch: [start_s - 0.5, start_s + 33). The picker takes the `n` pairwise
disjoint stretches (warm-ups included) with the smallest range of momentary loudness, e1 the smallest; it makes no
claim about what they hold (the page lists hum, birds and distant thunder, which only Shresth's ear can place).

- **Momentary loudness** is BS.1770's: the K-weighting of pyloudnorm's meter (the one render/loudness.py measures
  with), then the mean square of each 400 ms block, stepped by 100 ms, summed over the two channels, as
  -0.691 + 10 log10. Blocks start on exact sample multiples of 4,410 (100 ms at 44.1 kHz), so a stretch of 33.5 s
  holds 332 whole blocks and its range is their largest minus their smallest loudness.
- **Measured as the bed plays** (a choice the spec left open): through the bed's own HighpassFilter 200 Hz
  (render/bed.py's HIGHPASS), run once over the whole file, since the renderer plays every excerpt through it.
- **Candidates** start on the 100 ms grid, anywhere the whole stretch fits in the file; a stretch that holds a
  silent block (under BS.1770's -70 LUFS absolute gate) has no range and is never taken. The order is (range,
  start): e1 is the smallest range, then each next one the smallest range disjoint from every one already taken,
  each taken only if the rest can still be placed (so a small range in the middle of a short file cannot leave too
  little room on both sides). A file
  shorter than n x 33.5 s, or one that holds fewer than n disjoint stretches with a range, is refused; so is any
  rate but 44.1 kHz (convert a 48 kHz original once with `python -m lofi_rl.convert.resample` first, section 7.2).
- **Recorded with hashes**: the file's sha256 and pcm_sha256, frames, channels and the asset path the palettes name
  (relative to $LOFI_ASSETS), the original's path and sha256 (`--from`), the licence and url from assets/MANIFEST.yaml's
  entry for that original, the parameters, and per excerpt its start_s, its stretch in seconds and samples, its range,
  its quietest and loudest block and the pcm_sha256 of the stretch's samples. Deterministic: the same file gives the
  same bytes.

Exit 0, 2 for a refused input (nothing written). Sound claims: none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import yaml

from lofi_rl.palette import ROOT
from lofi_rl.render.bed import HIGHPASS
from lofi_rl.render.mix import process
from lofi_rl.render.wavio import pcm_sha256, read_float

SCHEMA = "lofi-rl/rain-excerpts/1"
SR = 44100
HOP = 4410                    # 100 ms: the momentary blocks' step and the candidates' grid
BLOCK = 4 * HOP               # 400 ms
WARMUP_S = Fraction(1, 2)     # the bed's high-pass warm-up before start_s (render/bed.py PREROLL_S)
LOOP_S = Fraction(32)         # N at 60 BPM, the longest loop a brief can ask for (8 bars x 240 / 60)
XFADE_S = Fraction(1)         # the bed's seam crossfade (xfade_s)
STRETCH_S = WARMUP_S + LOOP_S + XFADE_S          # 33.5 s
STRETCH = int(STRETCH_S * SR)                    # 1,477,350 samples = 335 hops
SILENT_LUFS = -70.0           # BS.1770's absolute gate: a block under it is silence
MANIFEST = ROOT / "assets" / "MANIFEST.yaml"


class ExcerptError(ValueError):
    pass


def momentary(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """BS.1770 momentary loudness of (channels, samples) audio: one value per 400 ms block, stepped by 100 ms,
    the first block at sample 0 (-inf for a silent block)."""
    if sr != SR:
        raise ExcerptError(f"{sr} Hz: the picker reads the 44.1 kHz file the bed plays")
    y = np.array(x, dtype=np.float64)
    for f in pyln.Meter(sr)._filters.values():          # K-weighting: pyloudnorm's high shelf, then its high pass
        for ch in range(y.shape[0]):
            y[ch] = f.apply_filter(y[ch])
    hops = y.shape[1] // HOP
    if hops < 4:
        return np.zeros(0)
    sq = (y[:, :hops * HOP] ** 2).reshape(y.shape[0], hops, HOP).sum(axis=2).sum(axis=0)   # per 100 ms, channels summed
    blocks = sq[:-3] + sq[1:-2] + sq[2:-1] + sq[3:]                  # four hops make one 400 ms block
    with np.errstate(divide="ignore"):
        return -0.691 + 10.0 * np.log10(blocks / BLOCK)


def ranges(blocks: np.ndarray, hops_per_stretch: int = STRETCH // HOP) -> np.ndarray:
    """The range (max - min) of the whole blocks inside each stretch that starts on hop k, for every k whose
    stretch fits: a stretch of H hops holds H - 3 blocks. A stretch holding a block under BS.1770's -70 LUFS absolute
    gate (silence, which the gate leaves out of any loudness) has no range: inf."""
    inside = hops_per_stretch - 3
    if len(blocks) < inside:
        return np.zeros(0)
    blocks = np.where(blocks >= SILENT_LUFS, blocks, -np.inf)
    win = np.lib.stride_tricks.sliding_window_view(blocks, inside)
    with np.errstate(invalid="ignore"):
        r = win.max(axis=1) - win.min(axis=1)
    return np.where(np.isfinite(r), r, np.inf)


def disjoint_count(starts: list[int], taken: list[int], hops: int) -> int:
    """How many more stretches of `hops` hops fit among the candidate `starts` (in time order), disjoint from each
    other and from `taken`: equal lengths, so taking the earliest start each time places the most."""
    count, last = 0, None
    for k in starts:
        if (last is None or k - last >= hops) and all(abs(k - j) >= hops for j in taken):
            count, last = count + 1, k
    return count


def pick(x: np.ndarray, sr: int = SR, n: int = 3) -> list[dict]:
    """The n disjoint stretches with the smallest momentary-loudness range, smallest first (see the module
    docstring). x is (channels, samples) at 44.1 kHz, as the bed reads the file."""
    if sr != SR:
        raise ExcerptError(f"the file is {sr} Hz; convert it to {SR} Hz once (python -m lofi_rl.convert.resample)")
    if x.ndim != 2 or x.shape[0] not in (1, 2):
        raise ExcerptError(f"expected mono or stereo (channels, samples) audio, got shape {x.shape}")
    frames = x.shape[1]
    if frames < n * STRETCH:
        raise ExcerptError(f"the file holds {frames / SR:.3f} s; {n} disjoint stretches of {float(STRETCH_S)} s need "
                           f"{n * float(STRETCH_S)} s")
    heard = process(HIGHPASS, x, sr)                     # as the bed plays it (render/bed.py)
    if heard.shape[0] == 1:
        heard = np.concatenate([heard, heard])           # a mono bed plays in both channels
    blocks = momentary(heard, sr)
    hops = STRETCH // HOP
    r = ranges(blocks, hops)
    finite = [k for k in range(len(r)) if np.isfinite(r[k])]          # starts in time order
    most = disjoint_count(finite, [], hops)
    if most < n:
        raise ExcerptError(f"only {most} disjoint stretches of {float(STRETCH_S)} s with a loudness range (no block "
                           f"under {SILENT_LUFS:g} LUFS) fit in the file; {n} are needed")
    order = sorted(finite, key=lambda k: (float(r[k]), k))
    taken: list[int] = []
    while len(taken) < n:                                # the smallest range that still lets the rest fit
        taken.append(next(k for k in order if all(abs(k - j) >= hops for j in taken)
                          and disjoint_count(finite, [*taken, k], hops) >= n - len(taken) - 1))
    out = []
    for i, k in enumerate(taken, 1):
        lo = k * HOP
        inside = blocks[k:k + hops - 3]
        start = Fraction(lo, SR) + WARMUP_S
        out.append({"id": f"e{i}", "start_s": float(start), "stretch_s": [float(Fraction(lo, SR)),
                                                                          float(Fraction(lo + STRETCH, SR))],
                    "samples": [lo, lo + STRETCH], "range_lu": float(r[k]), "min_lufs": float(inside.min()),
                    "max_lufs": float(inside.max()), "blocks": int(len(inside)),
                    "pcm_sha256": pcm_sha256(np.ascontiguousarray(x[:, lo:lo + STRETCH]))})
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_entry(sha256: str, manifest: Path = MANIFEST) -> dict | None:
    """assets/MANIFEST.yaml's entry pinning this sha256, or None."""
    try:
        doc = yaml.safe_load(Path(manifest).read_text(encoding="utf-8")) or {}
    except OSError:
        return None
    return next((a for a in doc.get("assets") or [] if a.get("sha256") == sha256), None)


def excerpts_doc(path: Path, *, n: int = 3, asset_path: str | None = None, original: Path | None = None,
                 manifest: Path = MANIFEST) -> dict:
    """The picker's record of one file (see the module docstring)."""
    path = Path(path)
    info = sf.info(str(path))
    if info.samplerate != SR:
        raise ExcerptError(f"{path} is {info.samplerate} Hz; convert it to {SR} Hz once "
                           "(python -m lofi_rl.convert.resample)")
    x, sr = read_float(path)
    picked = pick(x, sr, n)
    src = {"asset_path": asset_path or f"derived/rain/{path.name}", "sha256": _sha256(path),
           "pcm_sha256": pcm_sha256(x), "frames": int(x.shape[1]), "sr": SR, "channels": int(x.shape[0]),
           "subtype": info.subtype, "bytes": path.stat().st_size, "converted_from": None, "license": None,
           "url": None, "attribution": None}
    if original is not None:
        o_sha = _sha256(Path(original))
        entry = manifest_entry(o_sha, manifest) or {}
        o_path = entry.get("path") or str(original)
        src["converted_from"] = {"path": o_path, "sha256": o_sha, "sr": sf.info(str(original)).samplerate,
                                 "by": "python -m lofi_rl.convert.resample (resample_poly 147/160, float32 WAV)"}
        src.update(license=entry.get("license"), url=entry.get("url"))
    return {"schema": SCHEMA, "tool": "python -m lofi_rl.convert.rain_excerpts",
            "parameters": {"n": n, "warmup_s": float(WARMUP_S), "loop_s": float(LOOP_S), "xfade_s": float(XFADE_S),
                           "stretch_s": float(STRETCH_S), "block_s": BLOCK / SR, "hop_s": HOP / SR,
                           "measure": "BS.1770 momentary loudness (pyloudnorm K-weighting) after the bed's "
                                      "HighpassFilter 200 Hz; range = max - min of the stretch's whole blocks",
                           "order": "range, then start; each next excerpt disjoint from those taken"},
            "source": src, "excerpts": {e["id"]: {k: v for k, v in e.items() if k != "id"} for e in picked}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lofi_rl.convert.rain_excerpts", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="the rain file at 44.1 kHz (the converted one the bed reads)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from", dest="original", type=Path, default=None, help="the 48 kHz original it was made from")
    ap.add_argument("--asset-path", default=None, help="its path under $LOFI_ASSETS (default derived/rain/<name>)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    a = ap.parse_args(argv)
    try:
        if not a.src.is_file():
            raise ExcerptError(f"{a.src} is missing")
        if a.original is not None and not a.original.is_file():
            raise ExcerptError(f"--from {a.original} is missing")
        if a.n < 1:
            raise ExcerptError("--n must be 1 or more")
        doc = excerpts_doc(a.src, n=a.n, asset_path=a.asset_path, original=a.original, manifest=a.manifest)
    except (ExcerptError, RuntimeError) as err:             # soundfile raises RuntimeError on unreadable files
        print(f"rain_excerpts: {err}", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".part")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    tmp.replace(a.out)
    for eid, e in doc["excerpts"].items():
        print(f"{eid}: start_s {e['start_s']} (stretch {e['stretch_s'][0]}-{e['stretch_s'][1]} s), momentary range "
              f"{e['range_lu']:.3f} LU ({e['min_lufs']:.2f} to {e['max_lufs']:.2f} LUFS over {e['blocks']} blocks)")
    print(f"wrote {a.out} (source sha256 {doc['source']['sha256']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
