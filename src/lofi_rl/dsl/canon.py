"""Canonical LoopScript text: fixed statement order, single spaces, no comments.

Bar markers `|` are not stored in the AST; they are printed at every bar line (16, 32, 48)
that falls between two items, so the section 1 example is already canonical.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from .ast import STEPS, Item, Loop, edit_key, lane_key
from .vocab import STEPS_DUR

_ACCIDENTAL = {-1: "b", 0: "", 1: "#"}


def item_text(item: Item) -> str:
    dur = STEPS_DUR[item.dur]
    if item.rest:
        return dur + "r"
    marks = "'" * item.octave if item.octave > 0 else "," * -item.octave
    return f"{dur}{_ACCIDENTAL[item.accidental]}{item.degree}{marks}"


def phrase_text(items: Sequence[Item]) -> str:
    total = sum(item.dur for item in items)
    out, t = [], 0
    for item in items:
        out.append(item_text(item))
        t += item.dur
        if t % STEPS == 0 and t < total:
            out.append("|")
    return " ".join(out)


def canon(loop: Loop) -> str:
    first, second = loop.vamp.chords
    lines = [
        f"vamp: {first.text} {second.text} span {loop.vamp.span}",
        f"keys: {loop.keys.instrument} {loop.keys.voicing}",
        "comp: " + " ".join(loop.comp),
        "bass: " + (" ".join(loop.bass) if loop.bass else "lock"),
        f"drums: {loop.kit}",
    ]
    lines += [f"{lane.name}: " + " ".join(lane.grids) for lane in sorted(loop.lanes, key=lane_key)]
    lines += [f"@{e.bar} {e.target}: {e.grid}" for e in sorted(loop.edits, key=edit_key)]
    if loop.sax is not None:
        lines.append(f"sax: {loop.sax}")
    lines += [f"A: {phrase_text(loop.a)}", f"B: {phrase_text(loop.b)}"]
    return "\n".join(lines) + "\n"


def canon_sha(loop: Loop) -> str:
    return hashlib.sha256(canon(loop).encode("utf-8")).hexdigest()
