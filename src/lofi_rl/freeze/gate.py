"""Reading gates/GATE_palette.md for the freeze (docs/FREEZE.md §4).

Every reader here sees the file through `_lines`, which skips exactly the lines kit.gates.approvals skips (LESSONS 52:
an approval counts only outside code fences and HTML comments), one entry per line so its indices are the file's own,
each skipped line blanked:
- a fence opens on a line of 0-3 spaces and 3 or more backticks or tildes, and closes only on a line of the same
  character, at least as many (so a ``` line inside a ~~~~ or ```` fence is text of that fence);
- a comment runs from a line holding a `<!--` with no `-->` after it through the next line holding `-->`.
A heading, a list item or an APPROVED line inside either is never read. That reading cannot follow a comment opened
after a `-->` on one line (`--> <!--`, or `<!-- a --> b <!-- c`), which a browser hides to its end, so the APPROVED
reader refuses a file holding one (`approval_lines`); the pick reader reads it as kit.gates does.

The gate file keeps changing after the freeze (the sign-off counts, the drawn round tables), so nothing here reads it
by line number:
- each section by its heading (`### Day 1: voted`, `### Day 2: voted`, `### Day 3: voted`, `## Kept drafts (keep3)`,
  `## Approval`): a heading line that starts with that text, the section running to the next heading of its level or
  above (`sections` finds a section by its whole heading line, as the seal and lofi_rl.freeze.tables find
  `## Approval`);
- in a day's section, Shresth's confirmation by its bold label `**Shresth's confirmation, <date>:**` and the quoted
  words on that line; each pick by the bold label of its item in the **Survivors** or **Results** list and every
  backticked id the item holds, its sub-bullets included, in order (a sub-bullet is also found by its first word);
- keep3's table by its header row (the `Brief` and `Kept` columns);
- the approval by its exact line under `## Approval`.

`picks_problems(text)` compares all of it with lofi_rl.freeze.PICKS and names the heading of each difference: a label,
its ids, a confirmation or a heading that is missing or differs. A line moved within its section, or a line added
anywhere outside these items, changes nothing. `approval_date(text)` gives the date of Shresth's line, as the seal
(lofi_rl.freeze.seal) and `check --sealed` read it.

Sound claims: none.
"""
from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lofi_rl.freeze import BED, GATE, PICKS

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")                 # as kit.gates.approvals: 0-3 spaces, then the mark
_BOLD_LINE = re.compile(r"^\*\*([^*]+?)\*\*\s*$")            # a line that is only a bold label: **Results**
_ITEM = re.compile(r"^- \*\*(.+?)\*\*")                       # an item of a list: - **Keys:** ...
_SUB = re.compile(r"^\s+- (.*)$")                             # a sub-bullet, indented
_TICKS = re.compile(r"`([^`]+)`")
_QUOTES = re.compile(r'"([^"]*)"')
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_CONFIRMATION = re.compile(r"^\*\*Shresth['’]s confirmation, (\d{4}-\d{2}-\d{2}):\*\*(.*)$")
_JOB = re.compile(r"\bjob (\d+)\b")
APPROVAL_HEADING = "## Approval"
APPROVED = re.compile(r"^APPROVED by Shresth Rana (\d{4}-\d{2}-\d{2})$")   # trailing whitespace aside
_ATTEMPT = re.compile(r"^\s*approved\b", re.I)          # a line that starts as an approval: the exact form or refused


class GateError(ValueError):
    pass


@dataclass(frozen=True)
class Item:
    """A bold-labelled list item: its label (the bold text, colon dropped), every backticked id in it (its own lines
    and its sub-bullets, in order), and each sub-bullet's first word with that sub-bullet's ids."""

    label: str
    ids: tuple[str, ...]
    subs: tuple[tuple[str | None, tuple[str, ...]], ...]

    def sub(self, word: str) -> tuple[str, ...] | None:
        hits = [ids for w, ids in self.subs if w == word]
        return hits[0] if len(hits) == 1 else None


def read(path: Path | str = GATE) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        raise GateError(f"{path}: {err}") from err


def _comment_open_after(line: str, inside: bool) -> bool:
    """Whether an HTML comment is open at the end of `line`, given whether one was open at its start: every `<!--` and
    `-->` on the line, in order."""
    pos = 0
    while True:
        mark = "-->" if inside else "<!--"
        k = line.find(mark, pos)
        if k < 0:
            return inside
        inside, pos = not inside, k + len(mark)


def _scan(text: str) -> tuple[list[str], list[int]]:
    """(the file's lines as every reader here sees them, the indices of the lines that open a comment after a `-->`).

    The lines: one entry per line of `text.splitlines()`, so the indices are the file's own, each line kit.gates
    .approvals skips blanked: the lines of a fenced code block, and of an HTML comment, its opening and closing lines
    included. A fence or comment that never closes blanks the rest of the file. A leading byte-order mark is dropped,
    as kit.gates does. The second list names each line on which, read in order, a `<!--` follows a `-->` and stays
    open, which this reading (kit.gates') does not follow (module doc)."""
    out: list[str] = []
    odd: list[int] = []
    fence: str | None = None             # the open fence's mark, e.g. "~~~~"
    comment = False
    for i, line in enumerate(text.lstrip("\ufeff").splitlines()):
        if comment:                      # the closing line is blanked too
            comment = "-->" not in line
            if not comment and _comment_open_after(line, True):
                odd.append(i)            # `... --> <!-- ...`: reopened on its closing line
            out.append("")
            continue
        m = _FENCE.match(line)
        if m:
            mark = m.group(1)
            if fence is None:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence):
                fence = None
            out.append("")
            continue
        if fence is not None:
            out.append("")
            continue
        if "<!--" in line and "-->" not in line.split("<!--", 1)[1]:
            comment = True
            out.append("")
            continue
        if _comment_open_after(line, False):
            odd.append(i)                # `<!-- a --> b <!-- c`: opened again after a close
        out.append(line)
    return out, odd


def _lines(text: str) -> list[str]:
    """The file's lines as every reader here sees them (`_scan`)."""
    return _scan(text)[0]


def _sections(lines: list[str], heading: str) -> list[tuple[int, int]]:
    heads = [(i, len(m.group(1)), ln.rstrip()) for i, ln in enumerate(lines) if (m := _HEADING.match(ln))]
    return [(i, next((j for j, lv, _ in heads[k + 1:] if lv <= level), len(lines)))
            for k, (i, level, line) in enumerate(heads) if line == heading]


def sections(text: str, heading: str) -> list[tuple[int, int]]:
    """(heading line, end line) indices of every section whose heading line is exactly `heading`, trailing space
    aside, read through `_lines`: the section runs to the next heading of its level or above, or to the end of the
    file. The indices are the file's own lines (`text.splitlines()`). The seal reads his line in the one `## Approval`
    section found here, and lofi_rl.freeze.tables puts its section where this one ends, so the two always agree."""
    return _sections(_lines(text), heading)


def section(text: str, prefix: str) -> tuple[str, list[str]]:
    """(heading line, body lines) of the one section whose heading starts with `prefix`. GateError if there is none,
    or more than one."""
    lines = _lines(text)
    hits = [i for i, ln in enumerate(lines) if _HEADING.match(ln) and ln.startswith(prefix)]
    if not hits:
        raise GateError(f"no heading {prefix!r}")
    if len(hits) > 1:
        raise GateError(f"heading {prefix!r} appears {len(hits)} times")
    i = hits[0]
    level = len(_HEADING.match(lines[i]).group(1))
    end = next((k for k in range(i + 1, len(lines))
                if (m := _HEADING.match(lines[k])) and len(m.group(1)) <= level), len(lines))
    return lines[i], lines[i + 1:end]


def _first_word(text: str) -> str | None:
    m = _WORD.match(text.strip())
    return m.group(0) if m else None


def _item(label: str, block: list[str]) -> Item:
    subs: list[list[str]] = []
    for line in block[1:]:
        m = _SUB.match(line)
        if m:
            subs.append([m.group(1)])
        elif subs:
            subs[-1].append(line.strip())
    return Item(label, tuple(_TICKS.findall("\n".join(block))),
                tuple((_first_word(s[0]), tuple(_TICKS.findall("\n".join(s)))) for s in subs))


def items(body: list[str], list_label: str, where: str) -> dict[str, Item]:
    """The items of the list under the bold line `**<list_label>**`, by label. The list runs to the next line that
    starts with `**` (the next bold paragraph) or the section's end; an item is its `- **label**` line and the indented
    lines under it."""
    starts = [i for i, ln in enumerate(body)
              if (m := _BOLD_LINE.match(ln)) and m.group(1).strip().rstrip(":") == list_label]
    if not starts:
        raise GateError(f"{where}: no **{list_label}** list")
    if len(starts) > 1:
        raise GateError(f"{where}: **{list_label}** appears {len(starts)} times")
    region = body[starts[0] + 1:]
    end = next((k for k, ln in enumerate(region) if ln.startswith("**")), len(region))
    region = region[:end]
    out: dict[str, Item] = {}
    k = 0
    while k < len(region):
        m = _ITEM.match(region[k])
        if not m:
            k += 1
            continue
        label = m.group(1).strip().rstrip(":").strip()
        block = [region[k]]
        k += 1
        while k < len(region) and region[k][:1].isspace() and region[k].strip():
            block.append(region[k])
            k += 1
        if label in out:
            raise GateError(f"{where}: **{label}** is listed twice under **{list_label}**")
        out[label] = _item(label, block)
    return out


def confirmation(body: list[str], where: str) -> tuple[str, tuple[str, ...]]:
    """(date, quoted words) of the section's one `**Shresth's confirmation, <date>:**` line."""
    hits = [m for ln in body if (m := _CONFIRMATION.match(ln))]
    if not hits:
        raise GateError(f"{where}: no **Shresth's confirmation, <date>:** line")
    if len(hits) > 1:
        raise GateError(f"{where}: {len(hits)} confirmation lines")
    return hits[0].group(1), tuple(_QUOTES.findall(hits[0].group(2)))


def _job(heading: str) -> int | None:
    m = _JOB.search(heading)
    return int(m.group(1)) if m else None


def kept_table(body: list[str], where: str) -> dict[str, str]:
    """keep3's table: {brief: the draft file kept}, read by its header row's `Brief` and `Kept` columns."""
    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    for i, line in enumerate(body):
        if not line.startswith("|"):
            continue
        head = cells(line)
        if "Brief" not in head or "Kept" not in head:
            continue
        ib, ik = head.index("Brief"), head.index("Kept")
        out: dict[str, str] = {}
        for row in body[i + 1:]:
            if not row.startswith("|"):
                break
            c = cells(row)
            if set("".join(c)) <= set("-: "):            # the |---|---| line
                continue
            if len(c) != len(head):
                raise GateError(f"{where}: a table row has {len(c)} cells, the header {len(head)}: {row.strip()}")
            files = _TICKS.findall(c[ik])
            if len(files) != 1:
                raise GateError(f"{where}: brief {c[ib]}'s Kept cell names {len(files)} files")
            if c[ib] in out:
                raise GateError(f"{where}: brief {c[ib]} is listed twice")
            out[c[ib]] = files[0]
        return out
    raise GateError(f"{where}: no table with Brief and Kept columns")


def _ids(ids: tuple[str, ...]) -> str:
    return ", ".join(f"`{i}`" for i in ids) or "no ids"


def picks_problems(text: str, picks: Mapping[str, Mapping[str, Any]] = PICKS) -> list[str]:
    """The gate file against `picks`: every heading, job, confirmation, labelled item (its ids and named sub-bullets)
    and keep3's kept table. Each problem names its heading. Empty when they agree."""
    out: list[str] = []
    for key, want in picks.items():
        where = want["heading"]
        try:
            heading, body = section(text, want["heading"])
        except GateError as err:
            out.append(str(err))
            continue
        if _job(heading) != want["job"]:
            out.append(f"{where}: the heading names job {_job(heading)}, not {want['job']}")
        if "confirmation" in want:
            try:
                date, words = confirmation(body, where)
                c = want["confirmation"]
                if (date, words) != (c["date"], tuple(c["words"])):
                    out.append(f"{where}: the confirmation is {date} {list(words)}; the pick's is {c['date']} "
                               f"{list(c['words'])}")
            except GateError as err:
                out.append(str(err))
        if "items" in want:
            try:
                got = items(body, want["list"], where)
            except GateError as err:
                out.append(str(err))
                continue
            for label, ids in want["items"].items():
                item = got.get(label)
                if item is None:
                    out.append(f"{where}: no **{label}** item under **{want['list']}**")
                    continue
                if item.ids != tuple(ids):
                    out.append(f"{where}: **{label}** holds {_ids(item.ids)}; the pick is {_ids(tuple(ids))}")
                for word, sub_ids in (want.get("subs") or {}).get(label, {}).items():
                    have = item.sub(word)
                    if have is None:
                        out.append(f"{where}: **{label}** has no one sub-bullet starting {word!r}")
                    elif have != tuple(sub_ids):
                        out.append(f"{where}: **{label}** {word} holds {_ids(have)}; the pick is "
                                   f"{_ids(tuple(sub_ids))}")
        if "kept" in want:
            try:
                got_kept = kept_table(body, where)
            except GateError as err:
                out.append(str(err))
                continue
            if got_kept != dict(want["kept"]):
                diff = sorted(set(got_kept.items()) ^ set(dict(want["kept"]).items()))
                out.append(f"{where}: the kept table differs from the kept 10: {diff}")
    return out


def constants_problems(picks: Mapping[str, Mapping[str, Any]] = PICKS) -> list[str]:
    """`picks` against what the palette is built from: the draft's names and options (lofi_rl.drafts.palette KEYS,
    KITS, SAX), its settings (SETTINGS) and the bed's two variants (BED). Empty when they agree."""
    from lofi_rl.drafts.palette import KEYS, KITS, SAX, SETTINGS
    out = []
    d2, d3 = picks["day2"], picks["day3"]
    it, subs = d2["items"], d2.get("subs", {}).get("Sax", {})
    checks = [
        ("Day 2 Keys", tuple(it["Keys"]), tuple(KEYS.values())),
        ("Day 2 Kits and their basses", tuple(it["Kits and their basses"]),
         tuple(x for pair in KITS.values() for x in pair)),
        ("Day 2 Sax", tuple(it["Sax"][:1]), tuple(SAX.values())),
        ("Day 2 settings", (it["Swing"][0], it["Humanize"][0], subs.get("Warmth", ("",))[0],
                            subs.get("Timing", ("",))[0]), tuple(SETTINGS)),
        ("Day 3 bed", (d3["items"]["Rain excerpt"][0], d3["items"]["Rain level"][0]),
         (BED["excerpt_from"], BED["delta_from"])),
    ]
    for what, pick, used in checks:
        if pick != used:
            out.append(f"{what}: the pick is {list(pick)}, the palette is built from {list(used)}")
    return out


def confirmations(text: str, picks: Mapping[str, Mapping[str, Any]] = PICKS) -> dict[str, str]:
    """{day: the date of Shresth's confirmation} for each section of `picks` that has one. GateError if missing."""
    out = {}
    for key, want in picks.items():
        if "confirmation" in want:
            _, body = section(text, want["heading"])
            out[key] = confirmation(body, want["heading"])[0]
    return out


def kept(text: str, picks: Mapping[str, Mapping[str, Any]] = PICKS) -> dict[str, str]:
    """keep3's table as the gate file has it: {brief: kept draft file}."""
    want = picks["keep3"]
    _, body = section(text, want["heading"])
    return kept_table(body, want["heading"])


def approval_lines(text: str) -> tuple[list[str], list[str]]:
    """Under the `## Approval` heading: (the dates of the lines that are exactly `APPROVED by Shresth Rana
    <yyyy-mm-dd>`, trailing whitespace aside, with a real date; every other line that starts with "approved", in any
    case or indent: a malformed approval). Other lines (a note that his line came from chat, say) are ignored.
    GateError without the heading, or with two of it. A line inside a code fence or an HTML comment is never read
    (`_lines`), so an example of the form there neither counts nor refuses. GateError too when a line opens an HTML
    comment after a `-->` (module doc): where that comment ends cannot be read, so no line after it is trusted."""
    lines, odd = _scan(text)
    if odd:
        raise GateError(f"line {odd[0] + 1} opens an HTML comment after a `-->`, so where it ends cannot be read: "
                        f"write each comment on lines of its own")
    hits = _sections(lines, APPROVAL_HEADING)
    if len(hits) != 1:
        raise GateError(f"{len(hits)} {APPROVAL_HEADING!r} headings; the gate file has one")
    head, end = hits[0]
    dates, other = [], []
    for line in lines[head + 1:end]:
        hit = APPROVED.match(line.rstrip())
        if hit:
            try:
                _dt.date.fromisoformat(hit.group(1))
                dates.append(hit.group(1))
                continue
            except ValueError:
                pass
        if _ATTEMPT.match(line):
            other.append(line.strip())
    return dates, other


def approval_date(text: str) -> str | None:
    """The date of Shresth's one APPROVED line under `## Approval`, or None when there is none. GateError for a second
    matching line, or a line that starts as an approval in another form (only the exact form counts)."""
    dates, other = approval_lines(text)
    if other:
        raise GateError(f"under {APPROVAL_HEADING}: {other} is not the form `APPROVED by Shresth Rana <yyyy-mm-dd>`")
    if len(dates) > 1:
        raise GateError(f"under {APPROVAL_HEADING}: {len(dates)} APPROVED lines ({dates}); there is one")
    return dates[0] if dates else None
