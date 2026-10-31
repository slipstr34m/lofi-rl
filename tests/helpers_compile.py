"""Shared, cached inputs for the compiler tests: the dev palette, parsed fixtures and compiled loops, plus
spec oracles (docs/COMPILER.md section 3 written out independently of src/).

Cached per test session (lru_cache), so each fixture is parsed and compiled once however many test
modules use it. Not a test module.
"""
from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from lofi_rl.brief import Brief
from lofi_rl.compile.compiler import compile_loop
from lofi_rl.compile.events import CompiledLoop
from lofi_rl.dsl import parse
from lofi_rl.dsl.ast import Loop
from lofi_rl.palette import Palette, load_palette

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "loops"
DEV = ROOT / "palettes" / "dev_gu.yaml"
EXAMPLE_BRIEF = Brief(energy="mid", mood="wistful", bpm=82, tonic=2)      # instrument.md section 1: D dorian


@lru_cache(maxsize=None)
def dev_palette() -> Palette:
    return load_palette(DEV, verify=False)


def palette_with(**changes) -> Palette:
    """The dev palette with top-level fields, or groove./sax_expr. fields, replaced."""
    pal = dev_palette()
    groove = {k[7:]: v for k, v in changes.items() if k.startswith("groove_")}
    expr = {k[5:]: v for k, v in changes.items() if k.startswith("expr_")}
    top = {k: v for k, v in changes.items() if not k.startswith(("groove_", "expr_"))}
    if groove:
        top["groove"] = replace(pal.groove, **groove)
    if expr:
        top["sax_expr"] = replace(pal.sax_expr, **expr)
    return replace(pal, **top)


@lru_cache(maxsize=None)
def loop_of(name: str, mode: str = "wistful") -> Loop:
    res = parse((FIX / name).read_text(), vocab=dev_palette().vocab(), mode=mode)
    assert res.gate.ok, (name, res.gate)
    return res.ast


def loop_text(text: str, mode: str = "wistful") -> Loop:
    res = parse(text, vocab=dev_palette().vocab(), mode=mode)
    assert res.gate.ok, res.gate
    return res.ast


@lru_cache(maxsize=None)
def compiled(name: str, brief: Brief = EXAMPLE_BRIEF) -> CompiledLoop:
    return compile_loop(loop_of(name, brief.mode), brief, dev_palette())


def passing_fixtures() -> list[str]:
    """Every fixture that passes its gate under the dev palette's vocabulary (with aliases)."""
    out = []
    for p in sorted(FIX.glob("*.ls")):
        if parse(p.read_text(), vocab=dev_palette().vocab(), mode="wistful").gate.ok:
            out.append(p.name)
    return out


# ------------------------------------------------------------------------------------- spec oracles
# docs/COMPILER.md section 3 written out from its tables, independently of src/lofi_rl/compile and theory, so the
# compiler is checked against the spec rather than against itself.
SPEC_MODES = {"aeolian": (0, 2, 3, 5, 7, 8, 10), "dorian": (0, 2, 3, 5, 7, 9, 10), "ionian": (0, 2, 4, 5, 7, 9, 11)}
SPEC_SAX = {"tenor": {"play": (44, 75), "comfort": (48, 72), "home": (50, 61)},     # section 3.2, Registers
            "alto": {"play": (49, 80), "comfort": (55, 77), "home": (57, 68)}}
SPEC_SWING = Fraction(11, 20)                                                       # groove.swing.default 0.55


def spec_sample(k: int, n: int, swing: Fraction = SPEC_SWING, offset: Fraction = Fraction(0),
                nudge: Fraction = Fraction(0)) -> int:
    """Section 3.1 for step k, rounded half up: t_k = k N/128 + [k odd] (s - 1/2) N/64 + offset + nudge."""
    t = Fraction(k * n, 128) + (k % 2) * (swing - Fraction(1, 2)) * Fraction(n, 64) + offset + nudge
    return math.floor(t + Fraction(1, 2))


def spec_sax_pitches(loop: Loop, tonic: int, mode: str, sax: str = "tenor") -> dict[int, int]:
    """Onset step -> sounding pitch of every sax note (section 3.2): T + S[d-1] + accidental + 12 octave, with T
    the tonic in the home band; per phrase the shift k in (0, -1, +1) with the fewest notes outside the comfort
    window (ties in that order); then each note outside the playable range folds by octaves into it."""
    voice = SPEC_SAX[sax]
    lo, hi = voice["home"]
    home = lo + (tonic - lo) % 12
    out = {}
    for start, items in ((0, loop.a), (64, loop.b)):
        notes, k = [], start
        for it in items:
            if not it.rest:
                notes.append((k, home + SPEC_MODES[mode][it.degree - 1] + it.accidental + 12 * it.octave))
            k += it.dur
        c_lo, c_hi = voice["comfort"]
        shift = min((0, -1, 1), key=lambda s: sum(not c_lo <= p + 12 * s <= c_hi for _, p in notes))
        p_lo, p_hi = voice["play"]
        for k, p in notes:
            q = p + 12 * shift
            while q < p_lo:
                q += 12
            while q > p_hi:
                q -= 12
            out[k] = q
    return out
