"""Shared, cached renders for the renderer tests: the verified dev palette and one render per (fixture, brief).

Cached per test session (lru_cache, like helpers_compile), so the renders that several modules check are
made once. They use the default one-shot cache ($LOFI_CACHE, else ~/.cache/lofi-rl): an entry is reused only
while its samples hash to the pcm_sha256 recorded when it was built. Not a test module.
"""
from __future__ import annotations

import shutil
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from helpers_compile import DEV, EXAMPLE_BRIEF, FIX, ROOT
from lofi_rl.brief import Brief
from lofi_rl.palette import Bed, Palette, Source, load_palette
from lofi_rl.render.formats import splitmix64_stream
from lofi_rl.render.pipeline import Render, render
from lofi_rl.render.wavio import write_float

needs_fluidsynth = pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth not on PATH")
OTHER_BRIEF = Brief(energy="high", mood="hopeful", bpm=95, tonic=7)     # G ionian: B of the A-B-A order test
__all__ = ["DEV", "EXAMPLE_BRIEF", "FIX", "OTHER_BRIEF", "ROOT", "needs_fluidsynth", "noise", "palette",
           "rendered", "with_bed"]


@lru_cache(maxsize=None)
def palette() -> Palette:
    """The dev palette, with its SoundFont checked against its sha256 (what the CLI loads)."""
    return load_palette(DEV, verify=True)


def rendered(name: str, brief: Brief = EXAMPLE_BRIEF) -> Render:
    return _rendered(name, brief)


@lru_cache(maxsize=None)
def _rendered(name: str, brief: Brief) -> Render:      # positional, so every call shape shares one entry
    return render((FIX / name).read_text(), brief, palette())


def noise(channels: int, count: int, seed: int) -> np.ndarray:
    """Deterministic white noise in [-0.5, 0.5): SplitMix64, (channels, count) float32. Made inside the
    tests, never committed (instrument.md §4: the rain file comes later)."""
    h = splitmix64_stream(seed, channels * count) >> np.uint64(11)
    return (h.astype(np.float64) * 2.0 ** -53 - 0.5).astype(np.float32).reshape(channels, count)


def with_bed(folder: Path, n: int, *, channels: int = 2, delta_lu: float = 24.0, start_s: float = 0.5,
             xfade_s: float = 1.0) -> Palette:
    """The dev palette plus a synthetic noise bed written to `folder`/bed.wav (a fetch-mode source)."""
    start, xfade = round(start_s * 44100), round(xfade_s * 44100)
    write_float(folder / "bed.wav", 0.1 * noise(channels, start + n + xfade + 4410, 2026), 44100)
    src = Source("bed", "bed.wav", "0" * 64, "CC0-1.0 (synthetic test noise)", "fetch", None, None, None)
    pal = palette()
    return replace(pal, assets_root=str(folder), sources=MappingProxyType({**pal.sources, "bed": src}),
                   bed=Bed("bed", start_s, xfade_s, delta_lu))
