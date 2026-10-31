"""Repair log, gate result, and leniency steps 1-4 of instrument.md section 2.

Steps 2-4 (comments, whitespace, `=` for `:`) are silent: the grammar allows comments and the
section 5 leniency fixture expects only `strip_prose` in the log. Aliases (step 5) are logged
with `penalized=False`; the structural repairs of step 6 cost `PENALTY_EACH`, capped.

A keyword line (step 1) is an edit, or a `word:` line whose word the parser's `is_head`
accepts: the LoopScript heads, and also the forbidden words, so that `tempo: 90` before or
after the loop is a statement that reaches G5 rather than prose that is stripped for free.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

PENALTY_EACH = 0.02
PENALTY_CAP = 0.1

HeadTest = Callable[[str], bool]

# Digits are ASCII only: `\d` also matches other scripts' digits, which read `@٣` as bar 3.
_FENCE = re.compile(r"\s*```")
_KEYWORD = re.compile(r"\s*(?:@\s*[0-9]+\s*\S|([A-Za-z]+)\s*[:=])")
# comment = ("#" | "//") (SPACE | EOL) {any}: `#3 hat:`, `#notes` and `## Drums` are not comments.
_FULL_COMMENT = re.compile(r"\s*(?:#|//)(?=\s|$)")
_INLINE_COMMENT = re.compile(r"(?:^|\s)(?:#|//)(?=\s|$)")
_HEAD = re.compile(r"(@\s*([0-9]+)\s*)?([A-Za-z]+)\s*[:=]\s*")


@dataclass(frozen=True)
class Repair:
    code: str
    line: int                   # 1-based line in the text as given
    before: str
    after: str
    penalized: bool = False


@dataclass(frozen=True)
class Gate:
    ok: bool
    code: str | None = None     # "G1".."G6" when not ok
    detail: str = ""


class GateError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class Line:
    no: int
    text: str


def penalty(repairs: Iterable[Repair]) -> float:
    count = sum(1 for r in repairs if r.penalized)
    return round(min(PENALTY_CAP, PENALTY_EACH * count), 6)


def is_keyword_line(text: str, is_head: HeadTest) -> bool:
    """An edit (`@<digits> ...`), or a `word:` line whose word `is_head` accepts."""
    m = _KEYWORD.match(text)
    return bool(m) and (m.group(1) is None or is_head(m.group(1)))


def strip_prose(lines: list[Line], is_head: HeadTest) -> tuple[list[Line], list[Repair]]:
    """Step 1: drop code fences and anything outside the first and last keyword lines."""
    marks = [i for i, line in enumerate(lines) if is_keyword_line(line.text, is_head)]
    first, last = (marks[0], marks[-1]) if marks else (len(lines), -1)
    kept, dropped = [], []
    for i, line in enumerate(lines):
        inside = first <= i <= last and not _FENCE.match(line.text)
        (kept if inside else dropped).append(line)
    prose = [d for d in dropped if d.text.strip() and not _FULL_COMMENT.match(d.text)
             or _FENCE.match(d.text)]
    if not prose:
        return kept, []
    text = "\n".join(d.text for d in prose)
    return kept, [Repair("strip_prose", prose[0].no, text, "")]


def strip_comment(text: str) -> str:
    """Step 2: `#` or `//` followed by a space or the line end; a `#` inside a token is a sharp."""
    if _FULL_COMMENT.match(text):
        return ""
    m = _INLINE_COMMENT.search(text)
    return text[:m.start()] if m else text


def normalize_space(text: str) -> str:
    """Step 3: tabs and runs of spaces become one space; `=` after the head reads as `:`.

    An edit bar loses its leading zeros (`@03` -> `@3`) but stays text: the parser checks it
    against 1-8 as a string, because `int()` refuses strings of more than 4300 digits.
    """
    text = " ".join(text.split())
    m = _HEAD.match(text)
    if not m:
        return text
    head = f"@{m.group(2).lstrip('0') or '0'} {m.group(3)}" if m.group(1) else m.group(3)
    return f"{head}: {text[m.end():]}".rstrip()


def normalize(text: str, is_head: HeadTest) -> tuple[list[Line], list[Repair]]:
    """Steps 1-3. Case folding (step 4) happens per field, since grid cells keep their case."""
    lines = [Line(i + 1, t) for i, t in enumerate(text.removeprefix("﻿").splitlines())]
    kept, repairs = strip_prose(lines, is_head)
    out = []
    for line in kept:
        body = normalize_space(strip_comment(line.text))
        if body:
            out.append(Line(line.no, body))
    return out, repairs
