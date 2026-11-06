"""Human gates: a gate passes only on an explicit, dated approval line written by the human.

    from kit.gates import gate_approved
    gate_approved("gates/GATE_2.md")                          # any approver
    gate_approved("gates/GATE_6_publish.md", "Ada Lovelace")  # that approver only

    python -m kit.gates gates/GATE_2.md [--name "Ada Lovelace"]   # exit 0 approved, 1 not
    python -m kit.gates --how-to-approve --name "Ada Lovelace"    # print a gate file's How-to-approve section

An approval is a line of its own: `APPROVED by`, the name, and the date as year-month-day digits, for example
`APPROVED by Ada Lovelace 2026-12-09`. It stands at column 0 with nothing after the date, the date is a real calendar
date, and it lies outside code fences and HTML comments. The gate file's own instructions, quoted, indented, bulleted
or fenced examples, unfilled placeholders and "NOT APPROVED" never count. A substring check once accepted a publish
gate's own instructions as the human's go. The agent records a chat approval by appending such a line itself.

Showing the form: a placeholder gets copied. Shresth once wrote `APPROVED by Shresth Rana <2026-11-04>`, brackets
and all, after a How-to-approve line that read `<yyyy-mm-dd>`, and the refusal he got repeated that placeholder. So the
form is shown written out, with the approver's name and a real date, bare (`example_line`, `how_to_approve`), and a
line that starts as an approval but is not one is told what to change (`fixes`, `explain`, `attempts`): "write the
date bare: 2026-11-04, not <2026-11-04>". The accepted form (`APPROVAL_LINE`, `approvals`) is unchanged.

Another approver's line is never rewritten into the one asked for. An approval line by Ada Lovelace, checked for Grace
Hopper, is "an approval by Ada Lovelace, not by Grace Hopper: Grace Hopper's own line is needed", and so is a near miss
in another name (`fixes`); neither is shown as "the line that counts" in Grace Hopper's name (`corrected`), since an
agent that records approvals itself could append that line for someone who never wrote one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# the name may not start with a placeholder or markup character (_ < ` * [ #) and holds no < > ` * [ ]
APPROVAL_LINE = re.compile(r"^APPROVED by (?P<who>[^\s<>`*_\[\]#][^<>`*\[\]]*?) (?P<date>\d{4}-\d{2}-\d{2})[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
# a line of its own that starts as an approval, in any case: one, or a line `fixes` says how to mend. An indented
# line is an example (approvals never counts one), so it is not read as an attempt either
ATTEMPT = re.compile(r"^approved\b", re.I)
# how a line that starts as an approval is read when it is not one: what comes before the word, the word, then
# `by`, the name and the date it meant (digits, or a placeholder), with what wraps the date and what follows it
_LEAD = re.compile(r"^(?P<lead>[\s>*_`#+-]*)(?P<word>approved)\b(?P<rest>.*)$", re.I)
_BY = re.compile(r"^\s+by\b(?P<after>.*)$", re.I)
_DIGITS = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
_PLACEHOLDER = re.compile(r"(?i:y{2,4}[-/.]m{1,2}[-/.]d{1,2}|\bdate\b)")
_YMD = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
_WRAP_OPEN, _WRAP_CLOSE = "<[({\"'`*_", ">])}\"'`*_"
_NAME_BAD = re.compile(r"^[\s<>`*_\[\]#]|[<>`*\[\]]")      # what APPROVAL_LINE's name may not hold


def _readable(text: str) -> Iterator[tuple[int, str]]:
    """(line number, line) of every line an approval may stand on: outside code fences and HTML comments."""
    fence, in_comment = None, False
    for number, line in enumerate(text.splitlines(), 1):
        if in_comment:
            in_comment = "-->" not in line
            continue
        m = _FENCE.match(line)
        if m:
            mark = m.group(1)
            if fence is None:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        if "<!--" in line and "-->" not in line.split("<!--", 1)[1]:
            in_comment = True
            continue
        yield number, line


def _read(path: str | os.PathLike) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").lstrip("﻿")
    except (OSError, UnicodeDecodeError):
        return None


def _approval(line: str) -> tuple[str, str] | None:
    """(name, date) when the line is an approval, else None."""
    m = APPROVAL_LINE.match(line)
    if not m:
        return None
    try:
        dt.date.fromisoformat(m.group("date"))
    except ValueError:
        return None
    return " ".join(m.group("who").split()), m.group("date")


def approvals(path: str | os.PathLike) -> list[tuple[str, str]]:
    """Every valid (name, date) approval in the gate file, in file order; [] if it is missing."""
    text = _read(path)
    if text is None:
        return []
    return [hit for _, line in _readable(text) if (hit := _approval(line)) is not None]


def gate_approved(path: str | os.PathLike, name: str | None = None) -> bool:
    """True when the gate file holds an approval line (by `name`, when given)."""
    found = approvals(path)
    if name is None:
        return bool(found)
    want = " ".join(name.split())
    return any(who == want for who, _ in found)


# --------------------------------------------------------------------------------------------- showing the form
def _iso(day: dt.date | str | None) -> str:
    if day is None:
        return dt.date.today().isoformat()
    return day if isinstance(day, str) else day.isoformat()


def example_line(name: str, day: dt.date | str | None = None) -> str:
    """The approval line written out for `name`: `APPROVED by <name> <day>`, the day (today by default) as bare
    year-month-day digits, e.g. `APPROVED by Ada Lovelace 2026-12-09`. ValueError when no such line would count (a
    name with a placeholder or markup character, a date that is not real)."""
    line = f"APPROVED by {' '.join(name.split())} {_iso(day)}"
    if _approval(line) is None:
        raise ValueError(f"{line!r} would not count as an approval line")
    return line


def how_to_approve(name: str, day: dt.date | str | None = None, *, under: str | None = "## Approval") -> str:
    """A gate file's `## How to approve` section with the line written out: the approver's name and a real date
    (`day`, today by default), never a placeholder to copy. The example is indented, so neither `approvals` nor a
    reader of the `under` section ever counts it."""
    line = example_line(name, day)
    when = line.rsplit(" ", 1)[1]
    where = f" under `{under}`" if under else ""
    return ("## How to approve\n"
            f"Add one line of your own{where}, starting at its first character: APPROVED by {' '.join(name.split())}, "
            f"then the day you approve as year-month-day digits, with nothing around the date and nothing after it. "
            f"Approving on {when}, the line is:\n\n"
            f"    {line}\n")


@dataclass(frozen=True)
class _Parts:
    """A line that starts as an approval, read loosely (see `fixes`)."""
    lead: str                  # what comes before the word: spaces, list or quote marks, markup
    word: str                  # APPROVED as written
    by: bool                   # whether `by` follows it
    who: str | None            # the name between `by` and the date (None without `by`)
    date: str | None           # the date as written: digits, or a placeholder
    placeholder: bool          # the date is a placeholder (yyyy-mm-dd, <date>)
    wrap: tuple[str, str]      # what wraps the date: ("<", ">") in `<2026-11-04>`
    tail: str                  # what follows the date


def _parts(line: str) -> _Parts | None:
    m = _LEAD.match(line)
    if not m:
        return None
    by = _BY.match(m.group("rest"))
    after = by.group("after") if by else m.group("rest")
    hit = _DIGITS.search(after) or _PLACEHOLDER.search(after)
    if hit is None:
        who = after.strip() if by else None
        return _Parts(m.group("lead"), m.group("word"), bool(by), who, None, False, ("", ""), "")
    start, end = hit.span()
    while start > 0 and after[start - 1] in _WRAP_OPEN:
        start -= 1
    while end < len(after) and after[end] in _WRAP_CLOSE:
        end += 1
    wrap = (after[start:hit.start()], after[hit.end():end])
    who = after[:start].strip() if by else None
    return _Parts(m.group("lead"), m.group("word"), bool(by), who, hit.group(0), _PLACEHOLDER.fullmatch(hit.group(0))
                  is not None, wrap, after[end:].strip())


def _ymd(written: str) -> tuple[str | None, bool]:
    """(the date as year-month-day digits, whether that is a calendar date) for digits written in year-month-day
    order with -, / or . between them; (None, False) for any other order."""
    m = _YMD.fullmatch(written)
    if not m:
        return None, False
    iso = f"{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}"
    try:
        dt.date.fromisoformat(iso)
        return iso, True
    except ValueError:
        return iso, False


def _accepts(line: str, name: str | None) -> bool:
    hit = _approval(line)
    return hit is not None and (name is None or hit[0] == " ".join(name.split()))


def _other(line: str, name: str | None) -> str | None:
    """The name `line` gives when it is another approver's than `name`: an approval line by another name (as
    `gate_approved` compares them), or a line that starts as one and names, plainly, someone whose name differs from
    `name` in more than case and spacing. None without `name`, and for a near miss of `name`'s own line."""
    if not name:
        return None
    want, text = " ".join(name.split()), line.rstrip()
    hit = _approval(text)
    if hit is not None:
        return hit[0] if hit[0] != want else None
    p = _parts(text)
    if p is None or not p.who or _NAME_BAD.search(p.who):
        return None
    who = " ".join(p.who.split())
    return who if who.casefold() != want.casefold() else None


def fixes(line: str, name: str | None = None, *, today: dt.date | str | None = None) -> list[str]:
    """What to change, in plain words, so that `line` counts as an approval (by `name`, when given); [] when it does.
    Each item names its own fault, e.g. "write the date bare: 2026-11-04, not <2026-11-04>". A date the line has no
    use for (a placeholder, or none) is shown as `today` (default: today's date). Another approver's approval line
    needs nothing changed in it, only `name`'s own line ("Grace Hopper's own line is needed"), and a near miss in
    another name is told the same, beside its other faults."""
    text = line.rstrip()
    if _accepts(text, name):
        return []
    p = _parts(text)
    want = " ".join(name.split()) if name else None
    other = _other(text, name)
    if other is not None and _approval(text) is not None:
        return [f"{want}'s own line is needed"]
    if p is None:
        return [f"start the line with APPROVED by{' ' + want if want else ''}"]
    out = []
    if p.lead.strip():
        out.append(f"write the line plain, with nothing before APPROVED (here {p.lead.strip()!r})")
    elif p.lead:
        out.append("start the line at its first character: no spaces or tabs before APPROVED")
    if p.word != "APPROVED":
        out.append(f"write APPROVED in capitals, not {p.word}")
    if not p.by:
        out.append(f"write by and the name after APPROVED{': APPROVED by ' + want if want else ''}")
    elif not p.who:
        out.append(f"write the name after by{': APPROVED by ' + want if want else ''}")
    elif other is not None:
        out.append(f"it names {other}, not {want}: {want}'s own line is needed")
    elif want and " ".join(p.who.split()) != want:
        out.append(f"the name is {want}, not {p.who}")
    elif _NAME_BAD.search(p.who):
        out.append(f"write the name plainly: {p.who} holds a placeholder or markup character")
    if p.date is None:
        out.append(f"end the line with the date, written bare: {_iso(today)}")
    elif p.placeholder:
        out.append(f"write the day's date in place of {p.wrap[0]}{p.date}{p.wrap[1]}, bare: {_iso(today)}")
    else:
        iso, real = _ymd(p.date)
        if p.wrap != ("", ""):
            out.append(f"write the date bare: {p.date}, not {p.wrap[0]}{p.date}{p.wrap[1]}")
        if iso is None:
            out.append(f"write the date as year-month-day digits joined by hyphens, as {_iso(today)} is written, "
                       f"not {p.date}")
        elif iso != p.date:
            out.append(f"write the date as {iso}, not {p.date}")
        if iso is not None and not real:
            out.append(f"{iso} is not a date on the calendar")
    if p.tail:
        out.append(f"end the line with the date: nothing after it (here {p.tail!r})")
    return out or ["one space between the words, and nothing else on the line"]


def corrected(line: str, name: str | None = None, *, today: dt.date | str | None = None) -> str | None:
    """The line as it would count: `name` (or the line's own name, if that one is plain), and the line's date when it
    is a real one written in year-month-day order, else `today`. None when no name can be given, and when the line is
    another approver's (`_other`): it is never rewritten into `name`'s."""
    if _other(line, name) is not None:
        return None
    p = _parts(line.rstrip())
    who = " ".join(name.split()) if name else (p.who if p and p.who and not _NAME_BAD.search(p.who) else None)
    if not who:
        return None
    iso, real = _ymd(p.date) if p and p.date and not p.placeholder else (None, False)
    try:
        return example_line(who, iso if real else today)
    except ValueError:
        return None


def explain(line: str, name: str | None = None, *, today: dt.date | str | None = None) -> str:
    """One line for a refusal: the line, everything to change, and the line as it would count, e.g.
    "`APPROVED by Ada Lovelace <2026-12-09>` is not an approval line: write the date bare: 2026-12-09, not
    <2026-12-09> (the line that counts: APPROVED by Ada Lovelace 2026-12-09)". Another approver's approval line is
    named as theirs, with no line shown: "`APPROVED by Ada Lovelace 2026-12-09` is an approval by Ada Lovelace, not by
    Grace Hopper: Grace Hopper's own line is needed"."""
    todo, shown = fixes(line, name, today=today), line.rstrip()     # leading spaces kept: they may be the fault
    if not todo:
        return f"`{shown}` is an approval line"
    hit = _approval(shown)
    if hit is not None and _other(line, name) is not None:
        return f"`{shown}` is an approval by {hit[0]}, not by {' '.join(name.split())}: " + "; ".join(todo)
    fixed = corrected(line, name, today=today)
    return (f"`{shown}` is not an approval line: " + "; ".join(todo)
            + (f" (the line that counts: {fixed})" if fixed else ""))


def attempts(path: str | os.PathLike, name: str | None = None, *,
             today: dt.date | str | None = None) -> list[tuple[int, str, str]]:
    """(line number, line, `explain`) for each line of the gate file that starts as an approval, as a line of its own
    outside code fences and comments, but does not count (by `name`, when given); [] if the file is missing."""
    text = _read(path)
    if text is None:
        return []
    return [(number, line, explain(line, name, today=today)) for number, line in _readable(text)
            if ATTEMPT.match(line) and not _accepts(line, name)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="exit 0 when a gate file holds an approval line")
    ap.add_argument("gate", type=Path, nargs="?", help="the gate file to check")
    ap.add_argument("--name", default=None, help="require this approver (and, with --how-to-approve, show their line)")
    ap.add_argument("--how-to-approve", action="store_true",
                    help="print a gate file's How-to-approve section for --name, with the line written out")
    ap.add_argument("--date", default=None, help="the date the How-to-approve example shows (default today)")
    a = ap.parse_args(argv)
    if a.how_to_approve:
        if not a.name:
            ap.error("--how-to-approve needs --name, the approver whose line it shows")
        try:
            print(how_to_approve(a.name, a.date), end="")
        except ValueError as err:
            ap.error(str(err))
        return 0
    if a.gate is None:
        ap.error("give the gate file to check, or --how-to-approve")
    found = approvals(a.gate)
    ok = gate_approved(a.gate, a.name)
    seen = "; ".join(f"{who} {date}" for who, date in found)
    print(f"{a.gate}: {'APPROVED' if ok else 'NOT approved'}" + (f" (approval lines: {seen})" if seen else ""))
    if not ok:
        for number, _line, why in attempts(a.gate, a.name):
            print(f"  line {number}: {why}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
