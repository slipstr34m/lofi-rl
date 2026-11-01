"""The LoopScript regex for grammar-constrained decoding (src/lofi_rl/grammar.py; docs/GRAMMAR.md; D-016).

- Acceptance: `re.fullmatch` takes the canonical text, without `sax:`, of every gate-passing fixture, of the 9 shootout
  reference loops, and of the 98 passing completions of the base-model test v0 (runs/junk-13632676, which is pulled,
  never committed: those tests skip where it is not on disk).
- Line ends: the whitespace layouts v0 wrote (two spaces before each newline, an empty line before A, no newline
  after B), which the parser reads silently: no repair and the same AST as the canonical text. Each v0 passing
  completion's canonical text, written back in that completion's own layout, is accepted, and so is every layout
  any of the 500 wrote, on the example.
- Rejection: each v0 failure mode (a per-bar edit given two grids, grids of 13, 14 or 20 cells, an unknown chord or
  name, a missing required statement), every v0 G1 completion in the parser's own normalized form, and the
  canonical-form rules, among them the line-end layouts past the bounds. The G6 cases a regex cannot see pass
  through to the parser.
- Soundness: a seeded random walk over the regex's own parse tree (its groups, alternations and repeats) builds 1,500
  strings; every one parses with none of the gates the regex guarantees (G1, G3, G4, G5), no repair but the phrase
  ones, and exactly as its canonical layout does (the line ends change nothing).
- Dialect: vLLM 0.32's structured-outputs backends (xgrammar by default) take no lookaround, backreference, lazy
  quantifier or named group. The walk's reader parses only a small common dialect and refuses everything else.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from lofi_rl import grammar, prompt
from lofi_rl.dsl import Vocab, canon, parse
from lofi_rl.dsl.repair import normalize
from lofi_rl.dsl.vocab import FORBIDDEN_WORDS, LANE_ALIASES, STATEMENT_HEADS
from lofi_rl.palette import load_palette

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "palettes" / "dev_gu.yaml"
FIX = ROOT / "tests" / "fixtures" / "loops"
SHOOTOUT = ROOT / "shootouts" / "loops"
PASSING_FIXTURES = ["example.ls", "example_leniency.ls", "leniency_lower_x.ls", "one_bar_base.ls", "shaker_edits.ls",
                    "span1_dotted.ls"]
V0 = Path("runs") / "junk-13632676" / "summary"
V0_COUNTS = {"rows": 500, "pass": 98, "G1": 389}         # runs/junk-13632676/summary/junk.json (D-019's outcome)
# v0's whitespace layouts, measured on all 500 completions.jsonl texts: every line ends "  \n", 37 texts hold one empty
# line, always right before A, and every text stops right after its last item (no tab, CR, leading or doubled space)
V0_LAYOUTS = {"line ends '  ', no empty line, end ''": 463, "line ends '  ', an empty line before A, end ''": 37}
V0_PASSING_LAYOUTS = {"line ends '  ', no empty line, end ''": 97, "line ends '  ', an empty line before A, end ''": 1}
WALK_SEED, WALK_N = 596, 1500
PHRASE_REPAIRS = {"misplaced_bar", "pad_phrase", "trunc_phrase"}
A_LINE = "A: q5 e6 e5 q3 e2 e1 | h3 er e4 e5 e6 | h.5 er e4 | q3 q2 hr"
B_LINE = "B: q1' e7 e1' q5 q3 | er e3 e4 eb5 h5 | qr e4 e3 q2 q1 | h.6 qr"


@pytest.fixture(scope="module")
def pal():
    return load_palette(DEV, verify=False)


@pytest.fixture(scope="module")
def rx(pal):
    return grammar.loopscript_regex(pal)


@pytest.fixture(scope="module")
def full(rx):
    return re.compile(rx).fullmatch


def canonical(text: str, pal, mode: str | None = None) -> str:
    """The canonical text of a gate-passing loop without its optional `sax:` line: the form the regex matches."""
    res = parse(text, vocab=pal.vocab(), mode=mode)
    assert res.gate.ok, res.gate
    return canon(replace(res.ast, sax=None))


@pytest.fixture(scope="module")
def base(pal):
    """tests/fixtures/loops/example.ls in dev_gu's names, without sax: the in-prompt example."""
    text = canonical((FIX / "example.ls").read_text(), pal)
    assert text == prompt.example_text(pal) and A_LINE in text and B_LINE in text
    return text


def swap(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new)


def drop_line(text: str, prefix: str) -> str:
    lines = text.splitlines(keepends=True)
    assert sum(line.startswith(prefix) for line in lines) == 1, prefix
    return "".join(line for line in lines if not line.startswith(prefix))


def swap_lines(text: str, p: str, q: str) -> str:
    lines = text.splitlines(keepends=True)
    i, j = (next(k for k, line in enumerate(lines) if line.startswith(x)) for x in (p, q))
    lines[i], lines[j] = lines[j], lines[i]
    return "".join(lines)


# ------------------------------------------------------------------------------------------- whitespace layouts
_KEY = re.compile(r"\s*(@\s*0*([0-9]+)\s*)?([A-Za-z]+)\s*[:=]")


def statement_key(line: str) -> str:
    """The statement a line holds, as the parser keys it (`kick`, `@4 hat`, `A`): lane aliases resolved, case folded,
    leading zeros dropped. A line with no head keys as its own stripped text."""
    m = _KEY.match(line)
    if m is None:
        return line.strip()
    word = LANE_ALIASES.get(m.group(3).lower(), m.group(3).lower())
    word = word.upper() if word in ("a", "b") else word
    return f"@{m.group(2)} {word}" if m.group(1) else word


def layout(text: str) -> dict:
    """A text's whitespace layout, which the parser drops: the spaces before each statement's newline (by statement,
    and the commonest of them), the blank lines right before each statement (their own spaces), and what follows the
    text's last non-space character."""
    eol: dict[str, str] = {}
    blank: dict[str, list[str]] = {}
    pending: list[str] = []
    lines = text.split("\n")
    for k, line in enumerate(lines):
        last = k == len(lines) - 1
        if not line.strip():
            if not last:
                pending.append(line)
            continue
        key = statement_key(line)
        if pending:
            blank.setdefault(key, []).extend(pending)
            pending = []
        if not last:
            eol.setdefault(key, line[len(line.rstrip()):])
    common = Counter(eol.values()).most_common(1)[0][0] if eol else ""
    return {"eol": eol, "eol_common": common, "blank": blank, "end": text[len(text.rstrip()):]}


def describe(lay: dict) -> str:
    ends = ", ".join(repr(e) for e in sorted(set(lay["eol"].values())))
    blanks = ", ".join(f"{'an empty' if b == [''] else repr(b)} line before {k}" for k, b in sorted(lay["blank"].items()))
    return f"line ends {ends}, {blanks or 'no empty line'}, end {lay['end']!r}"


def relayout(canonical_text: str, lay: dict) -> str:
    """A canonical text written in a layout: each line's own line end (by statement, else the layout's commonest), its
    blank lines before it, and the layout's end after the last line."""
    lines = canonical_text.splitlines()
    out = []
    for k, line in enumerate(lines):
        key = statement_key(line)
        out.extend(b + "\n" for b in lay["blank"].get(key, ()))
        out.append(line + (lay["end"] if k == len(lines) - 1 else lay["eol"].get(key, lay["eol_common"]) + "\n"))
    missing = set(lay["blank"]) - {statement_key(line) for line in lines}
    assert not missing, f"blank lines before statements the canonical text does not have: {missing}"
    return "".join(out)


def without_layout(text: str) -> str:
    """The canonical layout of an accepted text: no spaces before a newline, no empty line, one final newline."""
    lines = [line.rstrip(" ") for line in text.split("\n")]
    return "".join(line + "\n" for line in lines if line)


def silent(text: str, reference: str, pal, mode: str | None = None) -> bool:
    """The parser reads `text` exactly as `reference`: the same gate and AST, the same repairs (line numbers aside, as
    a blank line shifts them)."""
    a, b = (parse(t, vocab=pal.vocab(), mode=mode) for t in (text, reference))
    same_gate = (a.gate.code, _LINE_NO.sub("", a.gate.detail)) == (b.gate.code, _LINE_NO.sub("", b.gate.detail))
    reps = [Counter((r.code, r.before, r.after, r.penalized) for r in res.repairs) for res in (a, b)]
    return same_gate and a.ast == b.ast and reps[0] == reps[1]


_LINE_NO = re.compile(r"^line [0-9]+: ")


# ------------------------------------------------------------------------------------------- the dialect
class Dialect(ValueError):
    """A construct outside the dialect every structured-outputs backend of vLLM 0.32 reads."""


_ESCAPES = {"n": "\n", **{c: c for c in "\\.^$*+?()[]{}|"}}


class Reader:
    """Parses a regex of the common dialect into a tree of its component patterns:
    ("lit", char) | ("set", chars) | ("seq", [node]) | ("alt", [node]) | ("rep", node, lo, hi).

    The dialect: literals, the escapes \\n and \\<metacharacter>, classes of single characters and ranges, plain groups,
    alternation, and the greedy quantifiers ? * + {n} {n,m} {n,}. Everything else raises Dialect, naming it:
    lookarounds, named and non-capturing groups and inline flags (anything after "(?"), backreferences, lazy and
    possessive quantifiers, anchors, the wildcard, negated classes, a bare "-" in a class, and escapes like
    \\d \\w \\b."""

    def __init__(self, regex: str):
        self.rx, self.i = regex, 0

    def parse(self):
        node = self._alt()
        if self.i != len(self.rx):
            raise Dialect(f"unbalanced ')' at {self.i}")
        return node

    def _peek(self, k: int = 0) -> str:
        j = self.i + k
        return self.rx[j] if j < len(self.rx) else ""

    def _alt(self):
        branches = [self._seq()]
        while self._peek() == "|":
            self.i += 1
            branches.append(self._seq())
        return branches[0] if len(branches) == 1 else ("alt", branches)

    def _seq(self):
        items = []
        while self._peek() not in ("", "|", ")"):
            items.append(self._quantified(self._atom()))
        return ("seq", items)

    def _atom(self):
        c, at = self._peek(), self.i
        if c == "(":
            if self._peek(1) == "?":
                raise Dialect(f"'(?' at {at}: a lookaround, a named or non-capturing group, or an inline flag")
            self.i += 1
            node = self._alt()
            if self._peek() != ")":
                raise Dialect(f"unclosed '(' at {at}")
            self.i += 1
            return node
        if c == "[":
            return self._set()
        if c == "\\":
            return ("lit", self._escape())
        if c in "^$":
            raise Dialect(f"anchor {c!r} at {at}")
        if c == ".":
            raise Dialect(f"wildcard '.' at {at}")
        if c in "*+?{":
            raise Dialect(f"quantifier {c!r} at {at} repeats nothing")
        if c in "]}":
            raise Dialect(f"unbalanced {c!r} at {at}")
        self.i += 1
        return ("lit", c)

    def _escape(self) -> str:
        e, at = self._peek(1), self.i
        if e.isdigit():
            raise Dialect(f"backreference \\{e} at {at}")
        if e not in _ESCAPES:
            raise Dialect(f"escape \\{e} at {at}: class escapes and \\k<name> are refused")
        self.i += 2
        return _ESCAPES[e]

    def _set(self):
        at = self.i
        self.i += 1
        if self._peek() == "^":
            raise Dialect(f"negated class at {at}")
        chars: list[str] = []
        while self._peek() != "]":
            c = self._peek()
            if c in ("", "[", "-"):
                raise Dialect(f"unclosed class at {at}" if not c else f"{c!r} inside the class at {at}")
            if c == "\\":
                lo = self._escape()
            else:
                lo = c
                self.i += 1
            if self._peek() == "-":
                hi = self._peek(1)
                if hi in ("", "]", "\\", "[") or hi < lo:
                    raise Dialect(f"bad range in the class at {at}")
                self.i += 2
                chars.extend(chr(k) for k in range(ord(lo), ord(hi) + 1))
            else:
                chars.append(lo)
        self.i += 1
        if not chars:
            raise Dialect(f"empty class at {at}")
        return ("set", tuple(chars))

    def _quantified(self, node):
        c, at = self._peek(), self.i
        if c in ("?", "*", "+"):
            self.i += 1
            lo, hi = {"?": (0, 1), "*": (0, None), "+": (1, None)}[c]
        elif c == "{":
            end = self.rx.find("}", self.i)
            m = re.fullmatch(r"([0-9]+)(,([0-9]*))?", self.rx[self.i + 1:end]) if end > 0 else None
            if m is None:
                raise Dialect(f"malformed repeat at {at}")
            lo = int(m[1])
            hi = lo if m[2] is None else (int(m[3]) if m[3] else None)
            if hi is not None and hi < lo:
                raise Dialect(f"repeat {{{lo},{hi}}} at {at}")
            self.i = end + 1
        else:
            return node
        if self._peek() in ("?", "+"):
            raise Dialect(f"lazy or possessive quantifier at {self.i - 1}")
        if self._peek() in ("*", "{"):
            raise Dialect(f"stacked quantifier at {self.i}")
        return ("rep", node, lo, hi)


def nodes(node):
    """Every node of a Reader tree, depth first."""
    yield node
    if node[0] in ("seq", "alt"):
        for n in node[1]:
            yield from nodes(n)
    elif node[0] == "rep":
        yield from nodes(node[1])


def longest(node) -> int:
    """The length of the longest string the tree accepts (every repeat is bounded)."""
    kind = node[0]
    if kind in ("lit", "set"):
        return 1
    if kind == "seq":
        return sum(longest(n) for n in node[1])
    if kind == "alt":
        return max(longest(n) for n in node[1])
    return node[3] * longest(node[1])


def walk(node, rng: random.Random) -> str:
    """One string of the tree's language: a uniform branch at each alternation, a seeded count at each repeat."""
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "set":
        return rng.choice(node[1])
    if kind == "seq":
        return "".join(walk(n, rng) for n in node[1])
    if kind == "alt":
        return walk(rng.choice(node[1]), rng)
    _, sub, lo, hi = node
    return "".join(walk(sub, rng) for _ in range(_count(lo, hi, rng)))


def _count(lo: int, hi: int | None, rng: random.Random) -> int:
    """An optional group half the time; for a longer repeat both ends sometimes (10 % each), otherwise a short run
    past its minimum (mean about 4)."""
    hi = lo + 8 if hi is None else hi
    if hi - lo <= 1:
        return rng.choice((lo, hi))
    r = rng.random()
    if r < 0.1:
        return lo
    if r < 0.2:
        return hi
    return min(hi, lo + int(rng.expovariate(0.25)))


def walk_sample(rx: str, n: int = WALK_N, seed: int = WALK_SEED) -> list[str]:
    tree, rng = Reader(rx).parse(), random.Random(seed)
    return [walk(tree, rng) for _ in range(n)]


@pytest.mark.parametrize("bad, what", [
    ("a(?=b)", "(?"), ("a(?!b)", "(?"), ("(?<=a)b", "(?"), ("(?<!a)b", "(?"),        # lookarounds
    ("(a)\\1", "backreference"), ("(?P<x>a)(?P=x)", "(?"), ("(a)\\k<x>", "escape"),   # backreferences
    ("a*?", "lazy"), ("a+?", "lazy"), ("a??", "lazy"), ("a{2,3}?", "lazy"), ("a*+", "possessive"),
    ("(?P<x>a)", "(?"), ("(?<x>a)", "(?"),                                            # named groups
    ("(?:a)", "(?"), ("(?i)a", "(?"), ("^a", "anchor"), ("a$", "anchor"), ("a.b", "wildcard"),
    ("[^a]", "negated"), ("[-a]", "inside the class"), ("[a-]", "bad range"), ("[z-a]", "bad range"),
    ("\\d", "escape"), ("\\w+", "escape"), ("\\bx", "escape"),
    ("a**", "stacked"), ("a{2}{3}", "stacked"), ("(a", "unclosed"), ("a)", "unbalanced"),
])
def test_the_dialect_reader_refuses_what_the_backends_do_not_all_read(bad, what):
    with pytest.raises(Dialect, match=re.escape(what)):
        Reader(bad).parse()


def test_the_regex_is_in_the_common_dialect_with_every_repeat_bounded(rx):
    tree = Reader(rx).parse()                                          # raises Dialect on anything outside it
    for construct in ("(?", "*?", "+?", "??", "}?", "\\k<", "(?P"):    # spelled out, though the reader refuses them too
        assert construct not in rx, construct
    assert not re.search(r"\\[0-9]", rx), "no backreference"
    reps = [n for n in nodes(tree) if n[0] == "rep"]
    assert reps and all(n[3] is not None for n in reps), "no unbounded repeat: a completion cannot run on inside it"
    assert "\n" not in rx and "\\n" in rx, "newlines are written as the escape \\n"


# ------------------------------------------------------------------------------------------- identity
def test_the_sha_is_the_sha256_of_the_regex_text_and_stable(pal, rx):
    assert grammar.loopscript_regex(pal) == rx, "deterministic"
    assert grammar.regex_sha256(pal) == hashlib.sha256(rx.encode("utf-8")).hexdigest()
    narrow = Vocab(keys={"rhodes"}, kits={"jazz"}, numerals={"I", "IV"}, qualities={"m9", "13"})
    assert grammar.regex_sha256(narrow) != grammar.regex_sha256(pal)


def test_the_regex_is_assembled_from_its_named_components(pal, rx):
    c = grammar.components(pal)
    for name in ("dgrid", "sgrid", "chord", "keys", "voicing", "kit", "span", "phrase"):
        assert c[name] in rx, name
    assert re.fullmatch(c["dgrid"], "X.xg" * 4) and not re.fullmatch(c["dgrid"], "X-xg" * 4)
    assert re.fullmatch(c["sgrid"], "X-xg" * 4) and not re.fullmatch(c["sgrid"], "X-x" * 5)
    assert re.fullmatch(c["item"], "w.#7''") and re.fullmatch(c["item"], "e.r") and not re.fullmatch(c["item"], "s.1")


# ------------------------------------------------------------------------------------------- acceptance
def test_the_gate_passing_fixtures_are_the_six_this_file_names(pal):
    passing = sorted(f.name for f in FIX.glob("*.ls") if parse(f.read_text(), vocab=pal.vocab()).gate.ok)
    assert passing == PASSING_FIXTURES


def v0_layouts(text: str) -> list[str]:
    """A canonical text in v0's two layouts: two spaces before every newline and none after B, and the same with an
    empty line before A."""
    plain = text[:-1].replace("\n", "  \n")
    return [plain, swap(plain, "  \nA: ", "  \n\nA: ")]


@pytest.mark.parametrize("name", PASSING_FIXTURES)
def test_accepts_the_canonical_text_of_every_gate_passing_fixture(name, pal, full):
    text = canonical((FIX / name).read_text(), pal)
    assert full(text)
    assert all(full(t) for t in v0_layouts(text)), "and in v0's layouts"


def _shootout_moods() -> dict[str, str]:
    return {n: b["mood"] for n, b in yaml.safe_load((SHOOTOUT / "briefs.yaml").read_text())["loops"].items()}


def test_there_are_nine_shootout_loops():
    assert sorted(p.stem for p in SHOOTOUT.glob("*.ls")) == sorted(_shootout_moods()) and len(_shootout_moods()) == 9


@pytest.mark.parametrize("name", sorted(_shootout_moods()))
def test_accepts_the_canonical_text_of_every_shootout_loop(name, pal, full):
    text = canonical((SHOOTOUT / f"{name}.ls").read_text(), pal, mode=_shootout_moods()[name])
    assert full(text)
    assert all(full(t) for t in v0_layouts(text)), "and in v0's layouts"


def _v0_dir() -> Path:
    """runs/ is pulled, never committed (CLAUDE.md §7): this checkout's runs/, else, from an agent worktree
    (<checkout>/.claude/worktrees/<name>), the main checkout's."""
    cands = [ROOT / V0]
    if ROOT.parent.name == "worktrees" and ROOT.parent.parent.name == ".claude":
        cands.append(ROOT.parents[2] / V0)
    for d in cands:
        if (d / "scored.jsonl").is_file() and (d / "completions.jsonl").is_file():
            return d
    pytest.skip(f"{V0} is not on this machine (make pull brings it to the laptop)")


def _jsonl(path: Path) -> dict[int, dict]:
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {r["i"]: r for r in rows}


@pytest.fixture(scope="module")
def v0():
    d = _v0_dir()
    rows, scored = _jsonl(d / "completions.jsonl"), _jsonl(d / "scored.jsonl")
    assert len(rows) == len(scored) == V0_COUNTS["rows"]
    return rows, scored


def test_accepts_the_canonical_text_of_all_98_passing_v0_completions(v0, pal, full):
    rows, scored = v0
    ok = sorted(i for i, s in scored.items() if s["ok"])
    assert len(ok) == V0_COUNTS["pass"]
    texts = {}
    for i in ok:
        r = rows[i]
        res = parse(r["text"], vocab=pal.vocab(), finish_reason=r["finish_reason"], mode=r["brief"]["mood"])
        assert res.gate.ok, (i, res.gate)
        texts[i] = canon(replace(res.ast, sax=None))
    assert [i for i, t in texts.items() if not full(t)] == []


def test_accepts_every_v0_passing_completion_in_its_own_whitespace_layout(v0, pal, full):
    """F1: v0's canonical content, written back with the line ends, blank lines and end of the completion it came
    from, is in the language, and the parser reads it as the canonical text (no repair, the same AST)."""
    rows, scored = v0
    layouts = Counter()
    for i in sorted(i for i, s in scored.items() if s["ok"]):
        r = rows[i]
        mode = r["brief"]["mood"]
        canon_text = canonical(r["text"], pal, mode=mode)
        lay = layout(r["text"])
        layouts[describe(lay)] += 1
        text = relayout(canon_text, lay)
        assert full(text), (i, describe(lay), text)
        res = parse(text, vocab=pal.vocab(), mode=mode)
        assert res.gate.ok and not res.repairs, (i, res.gate, res.repairs)
        assert res.ast == parse(canon_text, vocab=pal.vocab(), mode=mode).ast, i
    assert layouts == V0_PASSING_LAYOUTS


def test_accepts_every_whitespace_layout_any_v0_completion_wrote_on_the_example(v0, pal, base, full):
    """All 500 completions, the 389 G1 ones included: each one's layout, applied to the in-prompt example."""
    rows, _ = v0
    layouts: dict[str, dict] = {}
    counts = Counter()
    for r in rows.values():
        lay = layout(r["text"])
        assert "\t" not in r["text"] and "\r" not in r["text"], r["i"]
        counts[describe(lay)] += 1
        layouts.setdefault(describe(lay), lay)
    assert counts == V0_LAYOUTS
    for name, lay in layouts.items():
        text = relayout(base, lay)
        assert full(text), name
        assert silent(text, base, pal), name


LAYOUTS = {                     # accepted line-end layouts: each one parses as the canonical text does
    "one trailing space": lambda b: swap(b, "drums: jazz\n", "drums: jazz \n"),
    "no final newline": lambda b: b[:-1],
    "two spaces before every newline": lambda b: b.replace("\n", "  \n"),
    "v0: two spaces before every newline, none after B": lambda b: b[:-1].replace("\n", "  \n"),
    "v0: the same, with an empty line before A": lambda b: swap(b[:-1].replace("\n", "  \n"), "  \nA: ", "  \n\nA: "),
    "an empty line before A": lambda b: swap(b, "\nA: ", "\n\nA: "),
    "two spaces before B's newline": lambda b: b[:-1] + "  \n",
    "two spaces after a one-grid lane": lambda b: swap(b, "kick: x......x..x..... x......x.x....x.\n",
                                                        "kick: x......x..x.....  \n"),
}


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_accepts_the_line_end_layouts_the_parser_reads_silently(name, base, full, pal):
    text = LAYOUTS[name](base)
    assert text != base and full(text)
    res = parse(text, vocab=pal.vocab())
    assert res.gate.ok and not res.repairs
    assert res.ast == parse(without_layout(text), vocab=pal.vocab()).ast
    assert canon(res.ast) == without_layout(text), "read exactly as written, line ends aside"


def normalized(text: str) -> str:
    """A completion as the parser reads it (dsl.repair.normalize: fences, prose outside the loop, comments, blank
    lines and extra spaces gone), without `sax:` lines, ending in one newline. What is still wrong is a real fault."""
    heads = STATEMENT_HEADS | frozenset(LANE_ALIASES) | FORBIDDEN_WORDS
    lines, _ = normalize(text, lambda word: word.lower() in heads)
    return "".join(line.text + "\n" for line in lines if not re.match(r"(?i)sax:", line.text))


def test_rejects_every_v0_g1_completion_in_the_parsers_own_normalized_form(v0, pal, full):
    rows, scored = v0
    g1 = sorted(i for i, s in scored.items() if s["gate"] == "G1")
    assert len(g1) == V0_COUNTS["G1"]
    for i in g1:
        text = normalized(rows[i]["text"])
        assert parse(text, vocab=pal.vocab(), mode=rows[i]["brief"]["mood"]).gate.code == "G1", (i, "the fault stays")
        assert not full(text), (i, text)


# ------------------------------------------------------------------------------------------- rejection
@pytest.mark.parametrize("old, new", [
    ("@4 kick: x......x..x...x.\n", "@4 kick: x......x..x...x. x......x..x...x.\n"),
    ("@3 hat: x.x.x.x.x.x.x.xx\n", "@3 hat: x.x.x.x.x.x.x.xx x.x.x.x.x.x.x.x.\n"),
    ("@5 comp: x-----.x--x-..x-\n", "@5 comp: x-----.x--x-..x- x-----.x----..x-\n"),
    ("@6 rim: ....X..g....X.g.\n", "@6 rim: ....X..g....X.g. ....X.......X...\n"),
], ids=["kick", "hat", "comp", "rim"])
def test_rejects_a_per_bar_edit_given_two_grids(old, new, base, full, pal):
    text = swap(base, old, new)
    assert parse(text, vocab=pal.vocab()).gate.code == "G1", "v0's commonest failure (@4 hat, @6 hat, @6 bass, ...)"
    assert not full(text)


GRIDS = {"kick": "x......x..x.....", "comp": "x-----.x----..x-",
         "@4 kick": "x......x..x...x.", "@5 comp": "x-----.x--x-..x-"}


@pytest.mark.parametrize("cells", [12, 13, 14, 15, 17, 20, 21, 32])
@pytest.mark.parametrize("where", sorted(GRIDS))
def test_rejects_a_grid_of_any_length_but_16(cells, where, base, full):
    grid = GRIDS[where]
    text = swap(base, f"{where}: {grid}", f"{where}: {(grid * 2)[:cells]}")
    assert not full(text)


@pytest.mark.parametrize("vamp", [
    "Im9 IV11", "Im9 IVm6", "Dm9 G13", "ii9 V13", "Im9 #IVm7b5", "Im9 IVMaj7", "Im9 IVΔ7", "Im9 IV", "I IV13",
    "Im9 IV13 bVII7",           # three chords, one of v0's G6 failures
    "Im9",                      # one chord
])
def test_rejects_an_unknown_or_non_canonical_chord(vamp, base, full):
    assert not full(swap(base, "vamp: Im9 IV13 span 2", f"vamp: {vamp} span 2"))


@pytest.mark.parametrize("old, new", [
    ("keys: rhodes rootless", "keys: clavinet rootless"),
    ("keys: rhodes rootless", "keys: wurli rootless"),          # an alias: canonical text has the resolved name
    ("keys: rhodes rootless", "keys: Rhodes rootless"),
    ("keys: rhodes rootless", "keys: rhodes drop2"),
    ("keys: rhodes rootless", "keys: rhodes"),
    ("drums: jazz", "drums: trap808"),
    ("drums: jazz", "drums: dusty"),                           # an alias
    ("vamp: Im9 IV13 span 2", "vamp: Im9 IV13 span 3"),
    ("vamp: Im9 IV13 span 2", "vamp: Im9 IV13"),               # canonical text always writes the span
])
def test_rejects_an_unknown_or_aliased_name(old, new, base, full):
    assert not full(swap(base, old, new))


@pytest.mark.parametrize("prefix", ["vamp:", "keys:", "comp:", "bass:", "drums:", "kick:", "rim:", "A:", "B:"])
def test_rejects_a_missing_required_statement(prefix, base, full, pal):
    text = drop_line(base, prefix)
    assert parse(text, vocab=pal.vocab()).gate.code == "G3"
    assert not full(text)


@pytest.mark.parametrize("prefix", ["hat:", "ohat:", "@3 hat:", "@4 kick:", "@5 comp:", "@6 rim:", "@7 ohat:",
                                    "@8 comp:"])
def test_optional_statements_may_be_left_out(prefix, base, full):
    assert full(drop_line(base, prefix))


NOT_CANONICAL = {
    "sax line": lambda b: swap(b, "A: q5", "sax: tenor\nA: q5"),
    "lanes out of order": lambda b: swap_lines(b, "kick:", "rim:"),
    "edits out of bar order": lambda b: swap_lines(b, "@3 hat:", "@4 kick:"),
    "edits out of lane order": lambda b: swap(b, "@4 kick: x......x..x...x.\n",
                                              "@4 kick: x......x..x...x.\n@4 bass: x-------..x-x---\n"),
    "duplicate edit": lambda b: swap(b, "@4 kick: x......x..x...x.\n", "@4 kick: x......x..x...x.\n" * 2),
    "snare and rim bases": lambda b: swap(b, "rim: ....X.......X... ",
                                          "snare: ....x.......x...\nrim: ....X.......X... "),
    "duplicate lane": lambda b: swap(b, "\nohat: ", "\nhat: x.x.x.x.x.x.x.x.\nohat: "),
    "bar 9": lambda b: swap(b, "@4 kick:", "@9 kick:"),
    "bar 0": lambda b: swap(b, "@4 kick:", "@0 kick:"),
    "bar 04": lambda b: swap(b, "@4 kick:", "@04 kick:"),
    "edit on keys": lambda b: swap(b, "@4 kick: x......x..x...x.", "@4 keys: rhodes shell"),
    "dash in a drum grid": lambda b: swap(b, "@4 kick: x......x..x...x.", "@4 kick: x------x--x---x-"),
    "cell outside the set": lambda b: swap(b, "@4 kick: x......x..x...x.", "@4 kick: x......o..x...x."),
    "three grids": lambda b: swap(b, "x-----.x-x------\n", "x-----.x-x------ x-------..x-x---\n"),
    "double space": lambda b: swap(b, "keys: rhodes", "keys:  rhodes"),
    "tab": lambda b: swap(b, "keys: rhodes", "keys:\trhodes"),
    "three trailing spaces": lambda b: swap(b, "drums: jazz\n", "drums: jazz   \n"),
    "a trailing tab": lambda b: swap(b, "drums: jazz\n", "drums: jazz\t\n"),
    "spaces after B with no newline": lambda b: b[:-1] + "  ",
    "comment": lambda b: swap(b, "drums: jazz\n", "drums: jazz # dusty\n"),
    "blank line": lambda b: swap(b, "drums: jazz\n", "drums: jazz\n\n"),
    "an empty line before B": lambda b: swap(b, "\nB: ", "\n\nB: "),
    "two empty lines before A": lambda b: swap(b, "\nA: ", "\n\n\nA: "),
    "a line of spaces before A": lambda b: swap(b, "\nA: ", "\n  \nA: "),
    "an empty line first": lambda b: "\n" + b,
    "a leading space": lambda b: swap(b, "\nkick:", "\n kick:"),
    "two final newlines": lambda b: b + "\n",
    "CRLF": lambda b: b.replace("\n", "\r\n"),
    "code fence": lambda b: "```\n" + b + "```\n",
    "prose": lambda b: "Here's a loop:\n" + b,
    "upper-case head": lambda b: swap(b, "\nkick:", "\nKICK:"),
    "lane alias": lambda b: swap(b, "\nkick:", "\nbd:"),
    "no space after a colon": lambda b: swap(b, "A: q5", "A:q5"),
    "dotted sixteenth": lambda b: swap(b, "A: q5 e6", "A: s.5 e6"),
    "degree 8": lambda b: swap(b, "A: q5 e6", "A: q8 e6"),
    "degree first": lambda b: swap(b, "A: q5 e6", "A: 5q e6"),
    "upper-case item": lambda b: swap(b, "A: q5 e6", "A: Q5 e6"),
    "note name": lambda b: swap(b, "A: q5 e6", "A: qD e6"),
    "three octave marks": lambda b: swap(b, "A: q5 e6", "A: q5''' e6"),
    "mixed octave marks": lambda b: swap(b, "A: q5 e6", "A: q5', e6"),
    "two accidentals": lambda b: swap(b, "A: q5 e6", "A: q#b5 e6"),
    "forbidden word": lambda b: swap(b, "A: q5 e6", "A: q5 tempo e6"),
    "leading bar mark": lambda b: swap(b, "A: q5 e6", "A: | q5 e6"),
    "trailing bar mark": lambda b: swap(b, "| q3 q2 hr\n", "| q3 q2 hr |\n"),
    "two bar marks in a row": lambda b: swap(b, "| h3 er", "| | h3 er"),
    "bar mark touching items": lambda b: swap(b, "e1 | h3", "e1|h3"),
    "five bar marks": lambda b: swap(b, "q3 q2 hr\n", "q3 | q2 | hr\n"),
    "empty phrase": lambda b: swap(b, B_LINE, "B: "),
}


@pytest.mark.parametrize("name", sorted(NOT_CANONICAL))
def test_rejects_text_that_is_not_canonical_loopscript(name, base, full):
    text = NOT_CANONICAL[name](base)
    assert text != base
    assert not full(text)


@pytest.mark.parametrize("old, new", [
    ("bass: x-------..x-x--- x-----.x-x------", "bass: lock"),
    ("vamp: Im9 IV13 span 2", "vamp: Im9 IV13 span 1"),
    ("rim: ....X.......X... ....X..g....X..g", "snare: ....X.......X... ....X..g....X..g"),
    ("rim: ....X.......X... ....X..g....X..g", "rim: ....X.......X..."),
    ("comp: x-----.x----..x- x-------..x-x---", "comp: x-----.x----..x-"),
    ("ohat: ................ ..............x.\n",
     "ohat: ................ ..............x.\nshaker: g.x.g.x.g.x.g.x.\n"),
    ("@8 comp: x-----.x----x-x-\n",
     "@8 comp: x-----.x----x-x-\n@8 bass: X-----x-..x-----\n@8 shaker: ................\n"),
    ("@3 hat: x.x.x.x.x.x.x.xx\n", "@1 comp: ----------------\n@3 snare: ........XXXXXXXX\n@3 hat: x.x.x.x.x.x.x.xx\n"),
    (A_LINE, "A: q.5 e6 q5' e.3 s2 | h.3, er e1 | q.5, e7, h1 | e.3 s#4 h5 qr"),
    (B_LINE, "B: q.1' e7 q5 e.3' s2' | w.3 e.4 s5 q6 | q.5 e4 hr"),          # a note across a bar line: 2 marks
    (B_LINE, "B: w.1'' h5,, w.3"),                                        # 56 steps: padded by the parser
])
def test_accepts_canonical_variants(old, new, base, full, pal):
    text = swap(base, old, new)
    assert full(text)
    assert parse(text, vocab=pal.vocab()).gate.ok


@pytest.mark.parametrize("edits, detail", [
    ([("vamp: Im9 IV13 span 2", "vamp: Im9 Im9 span 2")], "2 distinct chords"),
    ([(A_LINE, "A: wr wr wr wr")], "has no notes"),
    ([(A_LINE, "A: w5 w5 w5 w5 w5 w5")], "is 96 steps"),
    ([(A_LINE, "A: w5 w5")], "is 32 steps"),
    ([(A_LINE, "A: wr wr wr wr w5")], "no notes within 64 steps"),
    ([("kick: x......x..x..... x......x.x....x.", "kick: ................"), ("@4 kick: x......x..x...x.\n", "")],
     "kick is silent"),
    ([("rim: ....X.......X... ....X..g....X..g", "rim: ................"), ("@6 rim: ....X..g....X.g.\n", "")],
     "snare and rim are silent"),
])
def test_the_g6_checks_a_regex_cannot_make_stay_with_the_parser(edits, detail, base, full, pal):
    text = base
    for old, new in edits:
        text = swap(text, old, new)
    assert full(text), "the regex lets it through"
    gate = parse(text, vocab=pal.vocab()).gate
    assert gate.code == "G6" and detail in gate.detail, gate


# ------------------------------------------------------------------------------------------- vocabulary
def test_the_chords_are_every_numeral_x_quality_the_prompt_offers(pal, base, full):
    v = prompt.vocabulary(pal)
    assert grammar.vocabulary(pal) == v
    chords = [n + q for n in v["numerals"] for q in v["qualities"]]
    assert len(chords) == 11 * 14
    assert set().union(*prompt.mood_chords(pal).values()) < set(chords), "the mood lists are hints, not limits"
    for c in chords:
        other = "IV13" if c != "IV13" else "Im9"
        assert full(swap(base, "vamp: Im9 IV13 span 2", f"vamp: {c} {other} span 2")), c
        assert full(swap(base, "vamp: Im9 IV13 span 2", f"vamp: {other} {c} span 1")), c


def test_the_names_are_the_palettes_resolved_names(pal, base, full):
    v = prompt.vocabulary(pal)
    for k in v["keys"]:
        for s in v["voicings"]:
            assert full(swap(base, "keys: rhodes rootless", f"keys: {k} {s}"))
    for kit in v["kits"]:
        assert full(swap(base, "drums: jazz", f"drums: {kit}"))
    assert pal.aliases
    for alias in pal.aliases:
        assert not full(swap(base, "keys: rhodes rootless", f"keys: {alias} rootless"))
        assert not full(swap(base, "drums: jazz", f"drums: {alias}"))


def test_a_narrower_vocabulary_narrows_the_regex(base):
    v = Vocab(keys={"rhodes"}, kits={"jazz"}, numerals={"I", "IV"}, qualities={"m9", "13"}, voicings={"rootless"})
    full = re.compile(grammar.loopscript_regex(v)).fullmatch
    assert full(base)
    assert not full(swap(base, "vamp: Im9 IV13", "vamp: Im7 IV13"))
    assert not full(swap(base, "vamp: Im9 IV13", "vamp: Im9 V13"))
    assert not full(swap(base, "keys: rhodes rootless", "keys: rhodes shell"))


def test_a_name_that_cannot_be_one_token_is_refused():
    with pytest.raises(ValueError, match="token"):
        grammar.loopscript_regex(Vocab(keys={"fender rhodes"}, kits={"jazz"}))
    with pytest.raises(ValueError, match="no kits"):
        grammar.loopscript_regex(Vocab(kits=frozenset()))


# ------------------------------------------------------------------------------------------- soundness
def test_every_string_of_a_random_walk_over_the_regex_parses_with_no_guaranteed_gate(pal, rx, full):
    samples = walk_sample(rx)
    assert len(samples) == WALK_N >= 1500 and len(set(samples)) == WALK_N
    codes, shapes = Counter(), Counter()
    for k, s in enumerate(samples):
        assert full(s), (k, s)
        res = parse(s, vocab=pal.vocab())
        assert res.gate.code not in grammar.GUARANTEED, (k, res.gate, s)
        assert res.gate.code in (None, "G6"), (k, res.gate, s)
        assert {r.code for r in res.repairs} <= PHRASE_REPAIRS, (k, res.repairs)
        plain = without_layout(s)
        assert silent(s, plain, pal), (k, "the line ends change the reading", s)
        codes[res.gate.code] += 1
        shapes.update({"trailing spaces": " \n" in s, "an empty line before A": "\n\nA: " in s,
                       "no newline after B": not s.endswith("\n"), "canonical layout": s == plain})
    assert codes[None] > 0 and codes["G6"] > 0, codes
    assert all(shapes[k] > 0 for k in ("trailing spaces", "an empty line before A", "no newline after B")), shapes


def test_the_gate_table_covers_every_gate():
    assert set(grammar.GUARANTEED) == {"G1", "G3", "G4", "G5"}
    every = {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "compile"}
    assert set(grammar.GUARANTEED) | set(grammar.NOT_GUARANTEED) == every
    assert not set(grammar.GUARANTEED) & set(grammar.NOT_GUARANTEED)
    assert all(isinstance(v, str) and v for v in {**grammar.GUARANTEED, **grammar.NOT_GUARANTEED}.values())


def test_dev_gu_regex_is_the_one_d020_registered():
    """D-020 pre-registers this regex for both arms; a change to grammar.py or to dev_gu's lists needs an amendment."""
    regex = grammar.loopscript_regex(load_palette(DEV, verify=False))
    assert len(regex) == 3395
    assert hashlib.sha256(regex.encode("utf-8")).hexdigest() == (
        "1412f11f9ebdc520b50c263eda14be12829951357c17f43496ab2ddf5235e64e")
