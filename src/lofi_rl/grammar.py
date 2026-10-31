"""Grammar-constrained decoding for LoopScript: one regex for vLLM's structured outputs (docs/GRAMMAR.md; D-016).

    from lofi_rl.grammar import loopscript_regex, regex_sha256
    regex = loopscript_regex(palette)   # vLLM: SamplingParams(structured_outputs=StructuredOutputsParams(regex=regex))
    regex_sha256(palette)               # its identity: junk-rate's generate.json and junk.json, the GRPO run.json

The regex matches canonical LoopScript as `dsl.canon` prints it (docs/DSL.md §8), without the optional `sax:` line:
- the statements in canonical order, one per line, single spaces, a final newline: vamp, keys, comp, bass, drums,
  kick, snare or rim, then hat, ohat and shaker if present, the per-bar edits, A and B;
- every grid exactly 16 cells from its lane's own alphabet (drums `X x g .`, comp and bass also `-`);
- base lanes with 1 or 2 grids, `bass: lock`, and per-bar edits with exactly one grid, in canonical edit order (by
  bar, then comp, bass, kick, snare, rim, hat, ohat, shaker), each (bar, lane) at most once;
- the vocabulary `palette.vocab()` checks: every numeral x quality the prompt lists (in any mood, so a borrowed chord
  stays legal, as it is for G4), the keys names and voicing styles, the kits, the spans 1 and 2. Names are the
  resolved ones canonical text prints, never aliases;
- phrase items as DSL.md §3 defines them: a duration (s e e. q q. h h. w w.), then a rest `r` or a degree 1-7 with an
  optional `b` or `#` and up to two octave marks of one kind (canonical text prints the net count); `|` only between
  two items, at most 3 per phrase (canonical text marks the bar lines inside the phrase), a run of at most 64 items
  between marks.

Required statements are required by the regex, so G3 cannot fail. What a regex cannot express stays with the parser
(G6): the phrase durations summing to 64 steps (within 16), two distinct vamp chords, a note in each phrase, a kick
and a backbeat that sound after the edits. GUARANTEED and NOT_GUARANTEED below are the gate table.

The dialect is the one every structured-outputs backend of vLLM 0.32 reads (xgrammar by default; guidance,
outlines and lm-format-enforcer): literals, the escapes \\n \\. \\|, one class with a range ([1-7]), plain groups,
alternation, and the greedy, bounded quantifiers ? {n} {n,m}. No lookaround, backreference, lazy quantifier, named or
non-capturing group, anchor or wildcard (tests/test_grammar.py refuses them), and no unbounded repeat, so a
completion cannot run on inside the regex.

Standard library only (plus lofi_rl.dsl), so it imports in the policy env, the scorer env and on the laptop.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from lofi_rl.dsl.ast import BARS, PHRASE_STEPS, STEPS
from lofi_rl.dsl.vocab import BACKBEAT, EDIT_TARGETS, LANES, NUMERALS, QUALITIES, STEPS_DUR, VOICINGS, Vocab

DRUM_CELLS = "Xxg."                     # accent, hit, ghost, rest (DSL.md §3 dgrid)
STRIKE_CELLS = "Xxg-."                  # the same plus "-", which holds the strike before it (sgrid: comp, bass)
SPANS = ("1", "2")
ACCIDENTALS = ("b", "#")
OCTAVE_MARKS = ("'", ",")               # up, down; canonical text prints the net count, so one kind per note
MAX_OCTAVE_MARKS = 2                    # the parser takes any number; two move a note 24 of the tenor's 31 semitones
MAX_BAR_MARKS = PHRASE_STEPS // STEPS - 1           # 3: the bar lines inside a 4-bar phrase
MAX_RUN_ITEMS = PHRASE_STEPS            # 64: a 64-step phrase holds at most 64 items, each at least 1 step
OPTIONAL_LANES = tuple(lane for lane in LANES if lane != "kick" and lane not in BACKBEAT)   # hat ohat shaker
NEWLINE = "\\n"
_META = frozenset("\\.^$*+?()[]{}|")
if LANES[:3] != ("kick",) + BACKBEAT:   # the backbeat alternation sits where canonical text prints snare or rim
    raise RuntimeError(f"lofi_rl.grammar expects the lanes kick, snare, rim first, not {LANES}")

GUARANTEED: Mapping[str, str] = MappingProxyType({
    "G1": "every token comes from a closed list or a cell or item alphabet, one statement per line; the random walk "
          "in tests/test_grammar.py parses 600 accepted strings with no G1",
    "G3": "vamp, keys, comp, bass, drums, kick, snare or rim, A and B are not optional in the regex",
    "G4": "numerals, qualities, keys names, voicings and kits are the palette's resolved lists",
    "G5": "no free word: tempo, key, effect and bed words, note names and absolute chords cannot be written",
})
NOT_GUARANTEED: Mapping[str, str] = MappingProxyType({
    "G2": "max_tokens can still cut a completion: the regex's longest strings (64 edits, long phrases) run past 512 "
          "tokens",
    "G6": "two distinct vamp chords; each phrase within 16 steps of 64; a note in each phrase (within its 64 steps); "
          "a kick, and a snare or rim, that sound after the edits. (Exactly two chords and 16-cell grids are "
          "guaranteed.)",
    "G7": "the render (RENDER.md §3.5) is not text",
    "compile": "compile_loop can still refuse a loop that passes the gates (none did in v0)",
})


def loopscript_regex(palette: Any) -> str:
    """The regex for a palette (or a dsl Vocab): canonical LoopScript without `sax:`. Deterministic: same lists,
    same text, same sha."""
    c = components(palette)
    dlane = f"{c['dgrid']}( {c['dgrid']})?"
    slane = f"{c['sgrid']}( {c['sgrid']})?"
    lines = [
        f"vamp: {c['chord']} {c['chord']} span {c['span']}{NEWLINE}",
        f"keys: {c['keys']} {c['voicing']}{NEWLINE}",
        f"comp: {slane}{NEWLINE}",
        f"bass: (lock|{slane}){NEWLINE}",
        f"drums: {c['kit']}{NEWLINE}",
        f"kick: {dlane}{NEWLINE}",
        f"{_alt(BACKBEAT)}: {dlane}{NEWLINE}",
        *(f"({lane}: {dlane}{NEWLINE})?" for lane in OPTIONAL_LANES),
        *(f"(@{bar} {target}: {c['dgrid'] if target in LANES else c['sgrid']}{NEWLINE})?"
          for bar in range(1, BARS + 1) for target in EDIT_TARGETS),
        f"A: {c['phrase']}{NEWLINE}",
        f"B: {c['phrase']}{NEWLINE}",
    ]
    return "".join(lines)


def regex_sha256(palette: Any) -> str:
    """sha256 of the regex text (UTF-8): what runs record to name the grammar they decoded with."""
    return hashlib.sha256(loopscript_regex(palette).encode("utf-8")).hexdigest()


def components(palette: Any) -> dict[str, str]:
    """The named sub-patterns the regex is assembled from (docs/GRAMMAR.md lists them)."""
    v = vocabulary(palette)
    dur = _alt(STEPS_DUR[steps] for steps in sorted(STEPS_DUR))
    marks = _alt(mark * n for mark in OCTAVE_MARKS for n in range(1, MAX_OCTAVE_MARKS + 1))
    item = f"{dur}(r|{_alt(ACCIDENTALS)}?[1-7]{marks}?)"
    run = f"{item}( {item}){{0,{MAX_RUN_ITEMS - 1}}}"
    return {
        "dcell": _alt(DRUM_CELLS),
        "scell": _alt(STRIKE_CELLS),
        "dgrid": f"{_alt(DRUM_CELLS)}{{{STEPS}}}",
        "sgrid": f"{_alt(STRIKE_CELLS)}{{{STEPS}}}",
        "chord": _alt(v["numerals"]) + _alt(v["qualities"]),
        "span": _alt(SPANS),
        "keys": _alt(v["keys"]),
        "voicing": _alt(v["voicings"]),
        "kit": _alt(v["kits"]),
        "item": item,
        "run": run,
        "phrase": f"{run}( \\| {run}){{0,{MAX_BAR_MARKS}}}",
    }


def vocabulary(palette: Any) -> dict[str, tuple[str, ...]]:
    """The lists the regex allows, in the prompt's order (prompt.vocabulary): the grammar's order for numerals,
    qualities and voicings, alphabetical for the palette's names. A palette, or a dsl Vocab."""
    vocab: Vocab = palette.vocab() if hasattr(palette, "vocab") else palette
    lists = {"numerals": tuple(n for n in NUMERALS if n in vocab.numerals),
             "qualities": tuple(q for q in QUALITIES if q in vocab.qualities),
             "keys": tuple(sorted(vocab.keys)),
             "voicings": tuple(s for s in VOICINGS if s in vocab.voicings),
             "kits": tuple(sorted(vocab.kits))}
    empty = [k for k, items in lists.items() if not items]
    if empty:
        raise ValueError(f"the vocabulary has no {', '.join(empty)}: no loop could be written")
    return lists


def _alt(options: Iterable[str]) -> str:
    return "(" + "|".join(_lit(o) for o in options) + ")"


def _lit(text: str) -> str:
    """A literal: regex metacharacters escaped with a backslash, nothing else (escapes like \\# or \\- are not read
    the same way by every backend)."""
    if not text or any(ch.isspace() for ch in text):
        raise ValueError(f"{text!r} cannot be a LoopScript token: tokens are non-empty and have no spaces")
    return "".join("\\" + ch if ch in _META else ch for ch in text)
