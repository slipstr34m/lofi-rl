"""Shared, cached inputs for the compiler tests: the dev palette, parsed fixtures and compiled loops.

Cached per test session (lru_cache), so each fixture is parsed and compiled once however many test
modules use it. Not a test module.
"""
from __future__ import annotations

from dataclasses import replace
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
