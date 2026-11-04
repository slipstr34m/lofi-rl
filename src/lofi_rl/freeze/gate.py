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
- keep3's table by its header row (the `Brief` and `Kept` columns), and each redo round's table the same way under its
  own heading (`## Kept drafts (keep3 redo<k>)`, lofi_rl.freeze.REDOS: the loops he kept in that round's batch);
- each sign-off listen by its heading (`## Sign-off listen...`, naming its batch and its job) and the backticked files
  of its `- **Tossed:**` item (lofi_rl.freeze.SIGNOFFS);
- the approval by its exact line under `## Approval`.

`picks_problems(text)` compares all of it with lofi_rl.freeze.PICKS, REDOS and SIGNOFFS and names the heading of each
difference: a label, its ids, a confirmation, a kept table, a tossed list or a heading that is missing or differs, and a
redo round or sign-off listen the gate file records that the constants do not (or the other way round). It also runs
`records_problems`, the constants' own rules (each redo round's key, folder, heading, batch, job and picks; each
sign-off's batch, job and tossed files). A line moved within its section, or a line added anywhere outside these
items, changes nothing. `kept_now_table(text)` is keep3's table overridden by each redo round's table in order, the
gate file's side of lofi_rl.freeze.kept_now(). `approval_date(text)` gives the date of Shresth's line, as the seal
(lofi_rl.freeze.seal) and `check --sealed` read it.

Sound claims: none.
"""
from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lofi_rl import freeze as F
from lofi_rl.drafts.personas import DRAFT_RE, PERSONAS
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
REDO_PREFIX = "## Kept drafts (keep3 redo"              # every redo round's heading starts so (REDOS `heading`)
SIGNOFF_PREFIX = "## Sign-off listen"                   # every sign-off listen's heading starts so (SIGNOFFS)
REDO_FIELDS = ("key", "folder", "batch", "job", "heading", "kept")
SIGNOFF_FIELDS = ("batch", "job", "tossed")
_REDO_KEY = re.compile(r"keep3_redo([1-9]\d*)")
_REDO_BATCH = re.compile(r"b\d{4}-keep3-redo\d*")      # lofi_rl.drafts.job.batch_name for a redraft
_SIGNOFF_BATCH = re.compile(r"b\d{4}-signoff(?:-\d+)?")  # lofi_rl.freeze.batch.NAME_RE


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


def sections_starting(text: str, prefix: str) -> list[tuple[str, list[str]]]:
    """(heading line, body lines) of every section whose heading starts with `prefix`, in file order, read through
    `_lines`: each runs to the next heading of its level or above."""
    lines = _lines(text)
    out = []
    for i, line in enumerate(lines):
        m = _HEADING.match(line)
        if not m or not line.startswith(prefix):
            continue
        level = len(m.group(1))
        end = next((k for k in range(i + 1, len(lines))
                    if (h := _HEADING.match(lines[k])) and len(h.group(1)) <= level), len(lines))
        out.append((line.rstrip(), lines[i + 1:end]))
    return out


def _names(heading: str, name: str) -> bool:
    """Whether `heading` names `name` as a whole token (so b1103-signoff is not b1103-signoff-2)."""
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", heading) is not None


def tossed_item(body: list[str], where: str) -> tuple[str, ...]:
    """The backticked files of a sign-off section's one `- **Tossed:**` item (its indented lines included); none for
    `- **Tossed:** none.`"""
    hits = []
    for k, line in enumerate(body):
        m = _ITEM.match(line)
        if not m or m.group(1).strip().rstrip(":").strip() != "Tossed":
            continue
        block, j = [line], k + 1
        while j < len(body) and body[j][:1].isspace() and body[j].strip():
            block.append(body[j])
            j += 1
        hits.append(tuple(_TICKS.findall("\n".join(block))))
    if not hits:
        raise GateError(f"{where}: no `- **Tossed:**` item (the kept files he tossed, or none)")
    if len(hits) > 1:
        raise GateError(f"{where}: {len(hits)} **Tossed:** items")
    return hits[0]


def records_problems(*, kept: Mapping[str, str] | None = None, redos: Sequence[Mapping[str, Any]] | None = None,
                     signoffs: Sequence[Mapping[str, Any]] | None = None,
                     jobs: Mapping[str, int] | None = None) -> list[str]:
    """lofi_rl.freeze.REDOS and SIGNOFFS against their own rules (each defaults to the constant, read at call time):
    - a redo round holds exactly REDO_FIELDS; its key is keep3_redo<k>, k rising from round to round; its folder is
      redo<k>, its heading `## Kept drafts (keep3 redo<k>)`, its batch b<MMDD>-keep3-redo[<n>] and its job JOBS[key];
      each pick is a brief of keep3's (KEPT) and a `<brief>_<persona>.ls` draft of that brief in the round's folder;
    - JOBS holds no keep3_redo<k> key without its round;
    - a sign-off listen holds exactly SIGNOFF_FIELDS; its batch is b<MMDD>-signoff[-<k>], its job a rising integer, and
      every file it tossed was kept by keep3 or by a redo round.
    Empty when they hold."""
    kept = F.KEPT if kept is None else kept
    redos = F.REDOS if redos is None else redos
    signoffs = F.SIGNOFFS if signoffs is None else signoffs
    jobs = F.JOBS if jobs is None else jobs
    out: list[str] = []
    ever, keys, last = set(kept.values()), set(), 0
    for i, r in enumerate(redos):
        if not isinstance(r, Mapping) or set(r) != set(REDO_FIELDS):
            out.append(f"REDOS[{i}]: a redo round holds exactly {list(REDO_FIELDS)}")
            continue
        where = f"REDOS[{i}] ({r['key']})"
        m = _REDO_KEY.fullmatch(str(r["key"]))
        if not m:
            out.append(f"{where}: its key is not keep3_redo<k>")
            continue
        k = int(m.group(1))
        if k <= last or r["key"] in keys:
            out.append(f"{where}: the rounds go in order, each once (redo{last} came before it)")
        last = max(last, k)
        keys.add(r["key"])
        if Path(str(r["folder"])).name != f"redo{k}":
            out.append(f"{where}: its folder {r['folder']} is not redo{k}")
        if r["heading"] != f"{REDO_PREFIX}{k})":
            out.append(f"{where}: its heading {r['heading']!r} is not {REDO_PREFIX + str(k) + ')'!r}")
        if not isinstance(r["batch"], str) or not _REDO_BATCH.fullmatch(r["batch"]):
            out.append(f"{where}: its batch {r['batch']!r} is not b<MMDD>-keep3-redo[<n>]")
        if isinstance(r["job"], bool) or not isinstance(r["job"], int) or jobs.get(r["key"]) != r["job"]:
            out.append(f"{where}: its job {r['job']!r} is not JOBS[{r['key']!r}] ({jobs.get(r['key'])!r})")
        if not isinstance(r["kept"], Mapping):
            out.append(f"{where}: its kept picks are a mapping of brief to file")
            continue
        for b, f in r["kept"].items():
            p = Path(str(f))
            name = DRAFT_RE.fullmatch(p.name)
            ok = name is not None and name["brief"] == b and name["persona"] in PERSONAS
            if b not in kept:
                out.append(f"{where}: brief {b} is not one of keep3's briefs")
            elif p.parent != Path(str(r["folder"])) or not ok:
                out.append(f"{where}: {b} keeps {f}, not a <brief>_<persona>.ls draft of {b} in {r['folder']}")
            ever.add(f)
    out += [f"JOBS[{k!r}]: no redo round of that key in REDOS" for k in jobs
            if _REDO_KEY.fullmatch(k) and k not in keys]
    batches, last = set(), 0
    for i, s in enumerate(signoffs):
        if not isinstance(s, Mapping) or set(s) != set(SIGNOFF_FIELDS):
            out.append(f"SIGNOFFS[{i}]: a sign-off listen holds exactly {list(SIGNOFF_FIELDS)}")
            continue
        where = f"SIGNOFFS[{i}] ({s['batch']})"
        if not isinstance(s["batch"], str) or not _SIGNOFF_BATCH.fullmatch(s["batch"]) or s["batch"] in batches:
            out.append(f"{where}: its batch is not a b<MMDD>-signoff[-<k>] name of its own")
        batches.add(s["batch"])
        if isinstance(s["job"], bool) or not isinstance(s["job"], int) or s["job"] <= last:
            out.append(f"{where}: its job {s['job']!r} is not a job id above the listen before it")
        else:
            last = s["job"]
        if not isinstance(s["tossed"], (tuple, list)) or len(set(s["tossed"])) != len(s["tossed"]):
            out.append(f"{where}: its tossed files are a list, each once")
            continue
        out += [f"{where}: it tossed {f}, which neither keep3 nor a redo round kept" for f in s["tossed"]
                if f not in ever]
    return out


def records_compare(text: str, *, redos: Sequence[Mapping[str, Any]] | None = None,
                    signoffs: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
    """The gate file against REDOS and SIGNOFFS (which records_problems must pass first): each redo round's section,
    its heading's job and batch and its kept table; each sign-off listen's section (the one `## Sign-off listen`
    heading naming its batch), its heading's job and its **Tossed:** files; and no redo round or sign-off listen in
    the gate file that the constants lack. Each problem names its heading."""
    redos = F.REDOS if redos is None else redos
    signoffs = F.SIGNOFFS if signoffs is None else signoffs
    out: list[str] = []
    for r in redos:
        where = r["heading"]
        try:
            heading, body = section(text, where)
        except GateError as err:
            out.append(str(err))
            continue
        if _job(heading) != r["job"]:
            out.append(f"{where}: the heading names job {_job(heading)}, not {r['job']}")
        if not _names(heading, r["batch"]):
            out.append(f"{where}: the heading does not name its batch {r['batch']}")
        try:
            got = kept_table(body, where)
        except GateError as err:
            out.append(str(err))
            continue
        if got != dict(r["kept"]):
            diff = sorted(set(got.items()) ^ set(dict(r["kept"]).items()))
            out.append(f"{where}: the kept table differs from REDOS {r['key']}'s picks: {diff}")
    out += [f"{heading}: a redo round the gate file records and lofi_rl.freeze.REDOS lacks"
            for heading, _ in sections_starting(text, REDO_PREFIX) if not any(heading.startswith(r["heading"])
                                                                              for r in redos)]
    found, used = sections_starting(text, SIGNOFF_PREFIX), set()
    for s in signoffs:
        where = f"{SIGNOFF_PREFIX} ({s['batch']})"
        hits = [i for i, (heading, _) in enumerate(found) if _names(heading, s["batch"])]
        if len(hits) != 1:
            out.append(f"{where}: {len(hits)} `{SIGNOFF_PREFIX}` headings name its batch, not one")
            continue
        used.add(hits[0])
        heading, body = found[hits[0]]
        if _job(heading) != s["job"]:
            out.append(f"{where}: the heading names job {_job(heading)}, not {s['job']}")
        try:
            tossed = tossed_item(body, where)
        except GateError as err:
            out.append(str(err))
            continue
        if sorted(tossed) != sorted(s["tossed"]):
            out.append(f"{where}: **Tossed:** holds {_ids(tossed)}; SIGNOFFS has {_ids(tuple(s['tossed']))}")
    out += [f"{heading}: a sign-off listen the gate file records and lofi_rl.freeze.SIGNOFFS lacks"
            for i, (heading, _) in enumerate(found) if i not in used]
    return out


def picks_problems(text: str, picks: Mapping[str, Mapping[str, Any]] = PICKS, *,
                   redos: Sequence[Mapping[str, Any]] | None = None,
                   signoffs: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
    """The gate file against `picks`: every heading, job, confirmation, labelled item (its ids and named sub-bullets)
    and keep3's kept table; then the redo rounds and sign-off listens (records_problems, then records_compare, with
    REDOS and SIGNOFFS unless given). Each problem names its heading. Empty when they agree."""
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
    records = records_problems(redos=redos, signoffs=signoffs)
    return out + (records or records_compare(text, redos=redos, signoffs=signoffs))


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


def kept_now_table(text: str, picks: Mapping[str, Mapping[str, Any]] = PICKS, *,
                   redos: Sequence[Mapping[str, Any]] | None = None) -> dict[str, str]:
    """The kept list as the gate file has it: keep3's table overridden by each redo round's table (REDOS, unless
    given), in order; {brief: kept draft file}, in keep3's brief order. GateError for a section or table that is not
    there."""
    now = kept(text, picks)
    for r in (F.REDOS if redos is None else redos):
        _, body = section(text, r["heading"])
        now.update(kept_table(body, r["heading"]))
    return now


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
