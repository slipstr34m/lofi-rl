"""LoopScript v1 parser with the leniency rules and hard gates G1-G6 of instrument.md section 2.

Gate precedence, when one text has several problems: G2 (truncated output) first, then the
first parse failure labelled G5 if any failing token is forbidden content, else G1, then G4,
G3 and G6. Statements are parsed independently, so a structural problem (G6) on one line never
hides a parse error on another. Within a statement body every token is checked before the line
fails, so a span or arity error never hides a forbidden token on the same line. `ast` is None
whenever the gate fails.

Readings of the spec that the tests pin down:
- an empty phrase is one with no notes (no items, or rests only);
- snare and rim: when both lanes have a base statement, the first in the text wins and every
  statement on the other lane, edits included, is dropped with a penalized `snare_and_rim`
  repair. An edit on the lane that has no base is an ordinary edit (the section 5 seam fixture);
- a `word:` line whose word is forbidden (`tempo: 90`) is a statement wherever it sits, so it
  reaches G5 instead of being stripped as prose;
- a keyword without its colon (`A q5 ...`, `BD x...`) is G1 in any case, never a note name;
- a misplaced `|` is logged with `penalized=False` (nothing is repaired);
- a bare numeral needs the brief's `mode`; the sax octave fold needs the tonic, so it belongs
  to the compiler.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, TypeVar

from .ast import (BARS, PHRASE_STEPS, REST_BAR, STEPS, Chord, Edit, Item, Keys, Lane, Loop, Vamp,
                  edit_key, expand)
from .canon import phrase_text
from .repair import Gate, GateError, Line, Repair, normalize, penalty
from .vocab import (BACKBEAT, DEFAULT_VOCAB, DUR_STEPS, EDIT_TARGETS, LANE_ALIASES, LANES,
                    MINOR_QUALITIES, STATEMENT_HEADS, STEPS_DUR, Vocab, canonical_quality,
                    diatonic_quality, forbidden_kind, scale_of)

REQUIRED = ("vamp", "keys", "comp", "bass", "drums", "kick", "A", "B")   # plus snare or rim
GRID_MIN, GRID_MAX = 12, 20
PHRASE_MAX_OFF = 16
_HEADS = STATEMENT_HEADS | frozenset(LANE_ALIASES)
_SIZES = sorted(STEPS_DUR, reverse=True)
_BAR_NUMBERS = frozenset(str(bar) for bar in range(1, BARS + 1))
_SPANS = {"1": 1, "2": 2}
_LINE_ORDER = ("G5", "G1", "G4", "G6")      # which of a statement's failures it reports
_CHORD = re.compile(r"([b#]?)([IViv]+)(.*)")
_ITEM = re.compile(r"([seqhw])(\.?)(?:(r)|([b#]?)([1-7])([',]*))")
_EDIT = re.compile(r"@([0-9]+) ([A-Za-z]+): ?(.*)")
_STMT = re.compile(r"([A-Za-z]+): ?(.*)")
T = TypeVar("T")


@dataclass(frozen=True)
class ParseResult:
    ast: Loop | None
    repairs: list[Repair]
    gate: Gate

    @property
    def penalty(self) -> float:
        """P_fix: 0.02 per penalized repair, capped at 0.1."""
        return penalty(self.repairs)


@dataclass(frozen=True)
class _Stmt:
    line: Line
    key: str                    # "vamp", "kick", "A", "@3 hat", ...
    lane: str                   # the head for statements, the target for edits
    body: str
    bar: int = 0                # 1..8 for edits


class _Failures:
    """Every failure on one statement, so a later check never masks a forbidden token."""

    def __init__(self) -> None:
        self.errors: list[GateError] = []

    def check(self, fn: Callable[..., T], *args: Any) -> T | None:
        try:
            return fn(*args)
        except GateError as err:
            self.errors.append(err)
            return None

    def add(self, err: GateError) -> None:
        self.errors.append(err)

    def raise_worst(self) -> None:
        """G5 if any failing token is forbidden content, else G1, G4, G6; the first of that code."""
        if self.errors:
            raise min(self.errors, key=lambda err: _LINE_ORDER.index(err.code))


def _present(values: Sequence[T | None]) -> list[T]:
    """The results of the checks that passed: all of them once `raise_worst` has returned."""
    return [v for v in values if v is not None]


def parse(text: str, vocab: Vocab = DEFAULT_VOCAB, finish_reason: str | None = None,
          *, mode: str | None = None) -> ParseResult:
    """Parse one model completion.

    `mode` (a mode or mood name) is only needed to repair a bare numeral into the mode's
    diatonic 7th chord; without it a bare numeral fails G4.
    """
    if mode is not None:
        scale_of(mode)
    if finish_reason == "length":
        return ParseResult(None, [], Gate(False, "G2", "finish_reason is 'length'"))
    return _Parser(vocab, mode).run(text)


class _Parser:
    def __init__(self, vocab: Vocab, mode: str | None) -> None:
        self.vocab = vocab
        self.mode = mode
        self.repairs: list[Repair] = []
        self.errors: dict[str, str] = {}

    # -- driver ---------------------------------------------------------------------------
    def run(self, text: str) -> ParseResult:
        lines, self.repairs = normalize(text, self._is_head)
        if not lines:
            return self._result("G1", "no LoopScript statements")
        heads = [self._guard(self._head, line) for line in lines]
        stmts = self._dedupe([s for s in heads if s is not None])
        parts: dict[str, Any] = {}
        for s in stmts:
            value = self._guard(self._body, s)
            if value is not None:
                parts[s.key] = value
        for code in ("G5", "G1", "G4"):
            if code in self.errors:
                return self._result(code, self.errors[code])
        missing = _missing({s.key for s in stmts})
        if missing:
            return self._result("G3", "missing: " + ", ".join(missing))
        if "G6" in self.errors:
            return self._result("G6", self.errors["G6"])
        loop = _build(parts)
        silent = _silence(loop)
        return self._result("G6", silent) if silent else self._result(None, "", loop)

    def _result(self, code: str | None, detail: str, loop: Loop | None = None) -> ParseResult:
        repairs = sorted(self.repairs, key=lambda r: r.line)
        return ParseResult(loop, repairs, Gate(code is None, code, detail))

    def _guard(self, fn: Callable[[Any], T], arg: Any) -> T | None:
        try:
            return fn(arg)
        except GateError as err:
            self.errors.setdefault(err.code, err.detail)
            return None

    def _log(self, code: str, line: int, before: str, after: str, penalized: bool = False) -> None:
        self.repairs.append(Repair(code, line, before, after, penalized))

    def _token_error(self, token: str, no: int, what: str) -> GateError:
        kind = forbidden_kind(token, self.vocab.forbidden)
        if kind:
            return GateError("G5", f"line {no}: {token!r} is {kind}")
        return GateError("G1", f"line {no}: {token!r} {what}")

    def _is_head(self, word: str) -> bool:
        """Keyword lines for `strip_prose`: LoopScript heads and forbidden words (`tempo:`)."""
        low = word.lower()
        return low in _HEADS or low in self.vocab.forbidden

    # -- statement heads ------------------------------------------------------------------
    def _head(self, line: Line) -> _Stmt:
        if line.text.startswith("@"):
            return self._edit_head(line)
        m = _STMT.fullmatch(line.text)
        if m is None:
            word = line.text.split()[0]
            if word.lower() in _HEADS:          # a keyword, never a note name (A, B, BD, SD)
                raise GateError("G1", f"line {line.no}: {word!r} is not followed by ':'")
            raise self._token_error(word, line.no, "does not start a statement")
        head = self._lane_alias(m.group(1), line.no)
        if head not in STATEMENT_HEADS:
            raise self._token_error(m.group(1), line.no, "is not a statement")
        key = head.upper() if head in ("a", "b") else head
        return _Stmt(line, key, head, m.group(2))

    def _edit_head(self, line: Line) -> _Stmt:
        m = _EDIT.fullmatch(line.text)
        if m is None:
            raise GateError("G1", f"line {line.no}: malformed edit {line.text!r}")
        digits, word, body = m.groups()
        target = self._lane_alias(word, line.no)
        if target in ("vamp", "sax"):
            raise GateError("G5", f"line {line.no}: an edit on {target} is not allowed")
        if target not in EDIT_TARGETS:
            if target in STATEMENT_HEADS:       # keys, drums, A, B: a keyword, not a note name
                raise GateError("G1", f"line {line.no}: {word!r} cannot be edited")
            raise self._token_error(word, line.no, "cannot be edited")
        if digits not in _BAR_NUMBERS:          # compared as text: it may have 5000 digits
            raise GateError("G1", f"line {line.no}: edit bar {digits} is outside 1-8")
        bar = int(digits)
        return _Stmt(line, f"@{bar} {target}", target, body, bar)

    def _lane_alias(self, word: str, no: int) -> str:
        low = word.lower()
        if low in LANE_ALIASES:
            self._log("alias_lane", no, word, LANE_ALIASES[low])
            return LANE_ALIASES[low]
        return low

    def _dedupe(self, stmts: list[_Stmt]) -> list[_Stmt]:
        """Duplicates: the first wins. Snare and rim: only when both have a base statement does
        the first base lane win, and then the other lane goes, its edits included."""
        bases = list(dict.fromkeys(s.lane for s in stmts if not s.bar and s.lane in BACKBEAT))
        loser = bases[1] if len(bases) > 1 else None
        kept, seen = [], set()
        for s in stmts:
            if s.lane == loser:
                self._log("snare_and_rim", s.line.no, s.line.text, "", penalized=True)
            elif s.key in seen:
                self._log("duplicate", s.line.no, s.line.text, "", penalized=True)
            else:
                seen.add(s.key)
                kept.append(s)
        return kept

    # -- statement bodies -----------------------------------------------------------------
    def _body(self, s: _Stmt) -> object:
        if s.bar:
            (grid,) = self._grids(s, drum=s.lane in LANES, most=1)
            return Edit(s.bar, s.lane, grid)
        if s.lane == "vamp":
            return self._vamp(s)
        if s.lane == "keys":
            inst, voicing = self._words(s, (self.vocab.keys, "keys instrument"),
                                        (self.vocab.voicings, "voicing"))
            return Keys(inst, voicing)
        if s.lane == "drums":
            return self._words(s, (self.vocab.kits, "drum kit"))[0]
        if s.lane == "sax":
            return self._words(s, (self.vocab.sax, "sax"))[0]
        if s.lane in ("a", "b"):
            return self._phrase(s)
        if s.lane == "bass" and s.body.lower() == "lock":
            return ()
        return self._grids(s, drum=s.lane in LANES, most=2)

    def _words(self, s: _Stmt, *slots: tuple[frozenset[str], str]) -> list[str]:
        """One name per slot; every token is checked before the arity."""
        no, toks, fails = s.line.no, s.body.split(), _Failures()
        names = [fails.check(self._name, tok, allowed, what, no)
                 for tok, (allowed, what) in zip(toks, slots)]
        for tok in toks[len(slots):]:
            fails.add(self._token_error(tok, no, "is unexpected here"))
        if len(toks) < len(slots):
            fails.add(GateError("G1", f"line {no}: {s.key} needs {len(slots)} word(s)"))
        fails.raise_worst()
        return _present(names)

    def _name(self, token: str, allowed: frozenset[str], what: str, no: int) -> str:
        """A legal name stays as written; an alias applies only where it lands on a legal name."""
        low = token.lower()
        if low in allowed:
            return low
        alias = self.vocab.name_aliases.get(low)
        if alias in allowed:
            self._log("alias_name", no, token, alias)
            return alias
        kind = forbidden_kind(token, self.vocab.forbidden)
        if kind:
            raise GateError("G5", f"line {no}: {token!r} is {kind}")
        raise GateError("G4", f"line {no}: {what} {token!r} is not in the palette")

    # -- grids ----------------------------------------------------------------------------
    def _grids(self, s: _Stmt, drum: bool, most: int) -> tuple[str, ...]:
        no, toks, fails = s.line.no, s.body.split(), _Failures()
        if not toks:
            raise GateError("G1", f"line {no}: {s.key} has no grid")
        cells = _present([fails.check(self._cells, t, drum, no) for t in toks])
        if len(toks) > most:
            fails.add(GateError("G1", f"line {no}: {s.key} takes at most {most} grid(s)"))
        fails.raise_worst()
        return tuple(self._fit_grid(c, no) for c in cells)

    def _cells(self, token: str, drum: bool, no: int) -> str:
        allowed = set("Xxg." if drum else "Xxg-.")
        if drum and "-" in token and set(token) <= allowed | {"-"}:
            fixed = token.replace("-", ".")
            self._log("alias_dash", no, token, fixed)
            token = fixed
        if set(token) - allowed:
            raise self._token_error(token, no, "is not a grid")
        return token

    def _fit_grid(self, grid: str, no: int) -> str:
        n = len(grid)
        if n == STEPS:
            return grid
        if GRID_MIN <= n < STEPS:
            code, fixed = "pad_grid", grid.ljust(STEPS, ".")
        elif STEPS < n <= GRID_MAX:
            code, fixed = "trunc_grid", grid[:STEPS]
        else:
            raise GateError("G6", f"line {no}: grid {grid!r} has {n} cells (needs 12-20)")
        self._log(code, no, grid, fixed, penalized=True)
        return fixed

    # -- vamp -----------------------------------------------------------------------------
    def _vamp(self, s: _Stmt) -> Vamp:
        no, toks, fails = s.line.no, s.body.split(), _Failures()
        lowered = [t.lower() for t in toks]
        cut = lowered.index("span") if "span" in lowered else len(toks)
        if cut == len(toks):
            self._log("default_span", no, s.line.text, s.line.text + " span 2")
        chords = _present([fails.check(self._chord, t, no) for t in toks[:cut]])
        span = self._span(toks[cut:], no, fails) if cut < len(toks) else 2
        fails.raise_worst()
        if len(chords) != 2 or chords[0] == chords[1]:
            got = " ".join(c.text for c in chords) or "none"
            raise GateError("G6", f"line {no}: the vamp needs 2 distinct chords, got {got}")
        return Vamp((chords[0], chords[1]), span)

    def _span(self, toks: list[str], no: int, fails: _Failures) -> int:
        """`toks` is the `span` keyword and what follows it."""
        value = toks[1] if len(toks) > 1 else ""
        if value not in _SPANS:
            fails.add(self._token_error(value or toks[0], no, "is not a span (1 or 2)"))
        for tok in toks[2:]:
            fails.add(self._token_error(tok, no, "is unexpected after the span"))
        return _SPANS.get(value, 2)

    def _chord(self, token: str, no: int) -> Chord:
        m = _CHORD.fullmatch(token)
        if m is None:
            raise self._token_error(token, no, "is not a chord (numeral + quality)")
        acc, roman, text = m.groups()
        if not (roman.isupper() or roman.islower()):
            raise GateError("G4", f"line {no}: numeral in {token!r} mixes upper and lower case")
        numeral = acc + roman.upper()
        if numeral not in self.vocab.numerals:
            raise GateError("G4", f"line {no}: numeral {acc + roman!r} is not in the palette")
        if not text:
            return self._bare(token, numeral, no)
        quality = canonical_quality(text, self.vocab.qualities)
        if roman.islower():
            return self._lowercase(token, numeral, text, quality, no)
        if quality is None:
            raise GateError("G4", f"line {no}: quality {text!r} is not in the palette")
        chord = Chord(numeral, quality)
        if chord.text != token:
            self._log("alias_quality", no, token, chord.text)
        return chord

    def _lowercase(self, token: str, numeral: str, text: str, quality: str | None,
                   no: int) -> Chord:
        """A lowercase numeral means minor (ii9 -> IIm9); if the quality says otherwise, it wins."""
        if quality not in MINOR_QUALITIES:
            minor = canonical_quality("m" + text, self.vocab.qualities)
            if minor in MINOR_QUALITIES:
                quality = minor
            elif quality is not None:
                chord = Chord(numeral, quality)
                self._log("case_conflict", no, token, chord.text, penalized=True)
                return chord
            else:
                raise GateError("G4", f"line {no}: quality {text!r} is not in the palette")
        chord = Chord(numeral, quality)
        self._log("alias_minor_numeral", no, token, chord.text)
        return chord

    def _bare(self, token: str, numeral: str, no: int) -> Chord:
        if self.mode is None:
            raise GateError("G4", f"line {no}: bare numeral {token!r} needs the brief's mode")
        quality = diatonic_quality(numeral, self.mode)
        if quality is None or quality not in self.vocab.qualities:
            raise GateError("G4", f"line {no}: {token!r} has no diatonic 7th chord in {self.mode}")
        chord = Chord(numeral, quality)
        self._log("bare_numeral", no, token, chord.text, penalized=True)
        return chord

    # -- phrases --------------------------------------------------------------------------
    def _phrase(self, s: _Stmt) -> tuple[Item, ...]:
        no, fails = s.line.no, _Failures()
        items: list[Item] = []
        misplaced, t = False, 0
        for tok in s.body.replace("|", " | ").split():
            if tok == "|":
                misplaced = misplaced or t % STEPS != 0
                continue
            item = fails.check(self._item, tok, no)
            if item is not None:
                items.append(item)
                t += item.dur
        fails.raise_worst()
        if all(item.rest for item in items):
            raise GateError("G6", f"line {no}: phrase {s.key} has no notes")
        fitted = self._fit_phrase(items, s)
        if all(item.rest for item in fitted):
            raise GateError("G6", f"line {no}: phrase {s.key} has no notes within 64 steps")
        if misplaced:
            self._log("misplaced_bar", no, s.body, phrase_text(fitted))
        return tuple(fitted)

    def _item(self, token: str, no: int) -> Item:
        m = _ITEM.fullmatch(token.lower())
        if m is None or (m.group(1) == "s" and m.group(2)):
            what = "is a dotted sixteenth, which does not exist" if m else "is not a phrase item"
            raise self._token_error(token, no, what)
        letter, dot, rest, acc, degree, marks = m.groups()
        dur = DUR_STEPS[letter] * 3 // 2 if dot else DUR_STEPS[letter]
        if rest:
            return Item(dur, rest=True)
        accidental = {"b": -1, "#": 1}.get(acc, 0)
        return Item(dur, False, int(degree), accidental, marks.count("'") - marks.count(","))

    def _fit_phrase(self, items: list[Item], s: _Stmt) -> list[Item]:
        total = sum(item.dur for item in items)
        off = total - PHRASE_STEPS
        if off == 0:
            return items
        if abs(off) > PHRASE_MAX_OFF:
            raise GateError("G6", f"line {s.line.no}: phrase {s.key} is {total} steps "
                                  f"(needs 64, at most 16 off)")
        fitted = items + _rests(-off) if off < 0 else _truncate(items, PHRASE_STEPS)
        code = "pad_phrase" if off < 0 else "trunc_phrase"
        self._log(code, s.line.no, s.body, phrase_text(fitted), penalized=True)
        return fitted


def _rests(steps: int) -> list[Item]:
    out = []
    for size in _SIZES:
        while steps >= size:
            out.append(Item(size, rest=True))
            steps -= size
    return out


def _truncate(items: list[Item], limit: int) -> list[Item]:
    """Cut at `limit` steps; a note across the cut is shortened, the rest of the gap is rest."""
    out, t = [], 0
    for item in items:
        room = limit - t
        if item.dur <= room:
            out.append(item)
            t += item.dur
            continue
        if room and not item.rest:
            size = max(s for s in _SIZES if s <= room)
            out.append(replace(item, dur=size))
            t += size
        out.extend(_rests(limit - t))
        break
    return out


def _missing(keys: set[str]) -> list[str]:
    missing = [k for k in REQUIRED if k not in keys]
    if not keys & set(BACKBEAT):
        missing.append("snare or rim")
    return missing


def _build(parts: dict[str, Any]) -> Loop:
    edits = sorted((v for k, v in parts.items() if k.startswith("@")), key=edit_key)
    return Loop(
        vamp=parts["vamp"], keys=parts["keys"], comp=parts["comp"], bass=parts["bass"],
        kit=parts["drums"], lanes=tuple(Lane(n, parts[n]) for n in LANES if n in parts),
        a=parts["A"], b=parts["B"], edits=tuple(edits), sax=parts.get("sax"),
    )


def _silence(loop: Loop) -> str | None:
    drums = expand(loop).drums
    if all(bar == REST_BAR for bar in drums["kick"]):
        return "kick is silent in all 8 bars"
    if all(bar == REST_BAR for lane in BACKBEAT for bar in drums[lane]):
        return "snare and rim are silent in all 8 bars"
    return None
