"""The one-time 48 -> 44.1 kHz conversion (docs/SHOOTOUTS.md §7.2, CD-36).

The sampler and the bed loader take 44.1 kHz only (RENDER.md §3.2, §9), so a 48 kHz source is converted once and
the result is pinned by its sha256: the Virtuosity one-shots (by the kit importer, `lofi_rl.convert.kit`) and the
rain recording (by this CLI). An SF2's samples carry their own rate, so the MTG banks need no conversion.

`to_44k(x, sr)` is scipy.signal.resample_poly(x, 147, 160, axis=0) in float64 on the whole signal: the exact
rational ratio 44100/48000, scipy's default Kaiser window (beta 5) and zero padding at both ends. Time runs along
axis 0, so x is (samples,) or (samples, channels); the result has ceil(147 n / 160) samples. Any rate other than
48 kHz is refused, 44.1 kHz included: nothing is resampled twice.

    python -m lofi_rl.convert.resample <in.wav> <out.wav>

reads any file libsndfile reads, writes a float32 WAV with no PEAK chunk (`wavio.write_float`, so the same audio
always gives the same bytes) and prints the output's sha256, its pcm_sha256 and the frame counts. Exit 2 for an
input that is not 48 kHz, missing or the output itself, with nothing written. One run on one machine gives the
pinned bytes; scipy's filter and upfirdn are deterministic there, and a different CPU could differ in the last
bits, which the pin would show.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from lofi_rl.render.wavio import pcm_sha256, write_float

UP, DOWN = 147, 160                 # 44100 / 48000 in lowest terms
SR_IN, SR_OUT = 48000, 44100


class ResampleError(ValueError):
    pass


def out_length(n: int) -> int:
    """Samples out for n samples in: ceil(147 n / 160)."""
    return -(-UP * n // DOWN)


def to_44k(x: np.ndarray, sr: int) -> np.ndarray:
    """x at 48 kHz, time on axis 0, resampled to 44.1 kHz in float64."""
    if sr != SR_IN:
        raise ResampleError(f"input is {sr} Hz; only {SR_IN} Hz is converted (to {SR_OUT} Hz)")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim not in (1, 2) or x.shape[0] == 0:
        raise ResampleError(f"expected (samples,) or (samples, channels) with samples > 0, got shape {x.shape}")
    return resample_poly(x, UP, DOWN, axis=0)


def convert_file(src: Path, out: Path) -> dict:
    """Resample one file and write it (through a temporary file and a rename); returns what the CLI prints."""
    info = sf.info(str(src))
    x, sr = sf.read(str(src), dtype="float64", always_2d=True)
    y = to_44k(x, int(sr)).astype(np.float32)
    tmp = out.with_name(out.name + ".part")
    write_float(tmp, y.T, SR_OUT)
    tmp.replace(out)
    data = out.read_bytes()
    return {"in": str(src), "in_rate": int(sr), "in_frames": int(x.shape[0]), "in_subtype": info.subtype,
            "channels": int(x.shape[1]), "out": str(out), "out_frames": int(y.shape[0]),
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "pcm_sha256": pcm_sha256(y.T)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lofi_rl.convert.resample", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="a 48 kHz audio file")
    ap.add_argument("out", type=Path, help="the 44.1 kHz float32 WAV to write")
    a = ap.parse_args(argv)
    try:
        if not a.src.is_file():
            raise ResampleError(f"{a.src} is missing")
        if a.out.exists() and a.out.resolve() == a.src.resolve():
            raise ResampleError("the output would overwrite the input")
        rate = sf.info(str(a.src)).samplerate
        if rate != SR_IN:
            raise ResampleError(f"{a.src} is {rate} Hz; only {SR_IN} Hz is converted (to {SR_OUT} Hz)")
        r = convert_file(a.src, a.out)
    except (ResampleError, RuntimeError) as err:      # soundfile raises RuntimeError on unreadable files
        print(f"resample: {err}", file=sys.stderr)
        return 2
    print(f"{r['out']}: sha256 {r['sha256']} ({r['bytes']:,} bytes); {r['in_frames']:,} frames at {r['in_rate']} Hz "
          f"({r['in_subtype']}) -> {r['out_frames']:,} at {SR_OUT} Hz float32, {r['channels']} channel(s); "
          f"pcm_sha256 {r['pcm_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
