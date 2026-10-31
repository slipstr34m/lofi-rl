"""Loop-text substitutions and the anchors' sour copies (docs/SHOOTOUTS.md sections 2.1, 2.5 and 5).

A round changes one thing only (section 2.1), so every substitution here rewrites one statement line and leaves
every other byte as it was:
- `swap_line`: the `vamp:` line of a vamp round (section 4.3), or any one statement;
- `spread_text`: K2's `keys: ep spread` (section 4.1);
- `raise_accidentals`: an anchor's sour copy, "the same loop with every sax note's accidental raised by one (a
  natural becomes #, a b becomes natural)" [CD-23]. A note already sharp cannot be raised in LoopScript, so it
  is an error; none of the nine reference loops has one.

`check_sour` is the builder's gate on a sour copy (section 7.6): G1-G6 in the loop's mood, and P_chrom >= 0.2.
P_chrom is instrument.md section 5's chromatic penalty, with section 0 amendment 1's free notes:
- a sax note is free if its pitch class is in the brief's mode, is a blue note (3, 6 or 10 semitones above the
  tonic), or is a tone of the chord sounding at its onset (the chord-tones table, named extensions included);
- a non-free note is bad unless it lasts 2 steps or fewer and the next note is a free note 1-2 semitones away;
  "the next note" is the next sounding note of the line read as a cycle (B's last note is followed by A's
  first), as the compiler reads it (COMPILER.md section 3.7), and its distance is in sounding MIDI notes after
  the compiler's per-phrase octave fit, so the lead's register is an input;
- P_chrom = min(.3, .1 x #bad).

Sound claims: none.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from lofi_rl.brief import Brief
from lofi_rl.compile.sax import sax_pitches
from lofi_rl.dsl import STEPS, Loop, chord_bars, parse
from lofi_rl.palette import SaxVoice
from lofi_rl.theory.chords import chord_pcs
from lofi_rl.theory.modes import BLUE_NOTES, scale_of

SOUR_MIN_P_CHROM = 0.2          # section 7.6: a sour copy must cost at least this much
P_CHROM_EACH, P_CHROM_CAP = 1, 3    # tenths: .1 per bad note, capped at .3 (instrument.md section 5)
RESOLVE_MAX_STEPS = 2
_PHRASE = re.compile(r"(A|B):(.*)")
_ITEM = re.compile(r"([seqhw]\.?)(?:(r)|([b#]?)([1-7])([',]*))")
_SEP = re.compile(r"(\s+|\|)")


# --------------------------------------------------------------------------------------------- substitutions
def _lines(text: str) -> tuple[list[str], bool]:
    return text.split("\n")[:-1] if text.endswith("\n") else text.split("\n"), text.endswith("\n")


def _join(lines: list[str], newline: bool) -> str:
    return "\n".join(lines) + ("\n" if newline else "")


def swap_line(text: str, head: str, body: str) -> str:
    """`text` with its one `<head>: ...` statement line replaced by `<head>: <body>`."""
    lines, nl = _lines(text)
    hits = [i for i, line in enumerate(lines) if re.match(rf"{re.escape(head)}:(\s|$)", line)]
    if len(hits) != 1:
        raise ValueError(f"{head}: must appear exactly once to be substituted, found {len(hits)}")
    lines[hits[0]] = f"{head}: {body}"
    return _join(lines, nl)


def spread_text(text: str) -> str:
    """K2's substitution: the loop's keys instrument with the spread voicing (`keys: ep spread`)."""
    lines, _ = _lines(text)
    keys = [line for line in lines if line.startswith("keys:")]
    words = keys[0].split()[1:] if len(keys) == 1 else []
    if len(words) != 2:
        raise ValueError("keys: must appear exactly once, as `keys: <instrument> <voicing>`")
    return swap_line(text, "keys", f"{words[0]} spread")


def raise_accidentals(text: str) -> str:
    """The sour copy: every sax note's accidental raised by one; every other byte unchanged."""
    lines, nl = _lines(text)
    found = set()
    for i, line in enumerate(lines):
        m = _PHRASE.fullmatch(line)
        if not m:
            continue
        found.add(m.group(1))
        out = []
        for tok in _SEP.split(m.group(2)):
            if not tok or _SEP.fullmatch(tok):
                out.append(tok)
                continue
            it = _ITEM.fullmatch(tok)
            if not it:
                raise ValueError(f"line {i + 1}: {tok!r} is not a canonical phrase item")
            dur, rest, acc, degree, octave = it.groups()
            if rest:
                out.append(tok)
            elif acc == "#":
                raise ValueError(f"line {i + 1}: {tok!r} is already sharp; LoopScript cannot raise it again")
            else:
                out.append(f"{dur}{'' if acc == 'b' else '#'}{degree}{octave}")
        lines[i] = f"{m.group(1)}:{''.join(out)}"
    if found != {"A", "B"}:
        raise ValueError(f"the loop needs an A: and a B: phrase line to sour, found {sorted(found)}")
    return _join(lines, nl)


# --------------------------------------------------------------------------------------------- P_chrom
def sax_voice(register: str, ranges: Mapping[str, tuple[int, int]]) -> SaxVoice:
    """A sax of `register` with the plan's sounding ranges (instrument.md section 3), for the octave fit."""
    return SaxVoice(register, "plan", 0, 0, tuple(ranges["play"]), tuple(ranges["comfort"]), tuple(ranges["home"]))


@dataclass(frozen=True)
class Chromatic:
    p_chrom: float
    bad: int            # non-free notes that do not resolve
    nonfree: int        # every non-free note


def chromatic(loop: Loop, brief: Brief, voice: SaxVoice) -> Chromatic:
    """instrument.md section 5's P_chrom of a parsed loop's sax line under `brief`, played by `voice`."""
    pitched, _ = sax_pitches(loop, brief.tonic, brief.mode, voice)        # (phrase, step, item, MIDI), in order
    scale = frozenset(scale_of(brief.mode))
    chords = chord_bars(loop.vamp)

    def free(step: int, pitch: int) -> bool:
        rel = (pitch - brief.tonic) % 12
        return rel in scale or rel in BLUE_NOTES or pitch % 12 in chord_pcs(chords[step // STEPS], brief.tonic)

    flags = [free(k, p) for _, k, _, p in pitched]
    bad = 0
    for i, (_, _, item, pitch) in enumerate(pitched):
        if flags[i]:
            continue
        j = (i + 1) % len(pitched)
        if not (item.dur <= RESOLVE_MAX_STEPS and flags[j] and 1 <= abs(pitched[j][3] - pitch) <= 2):
            bad += 1
    return Chromatic(min(P_CHROM_CAP, P_CHROM_EACH * bad) / 10, bad, flags.count(False))


@dataclass(frozen=True)
class SourCheck:
    text: str
    gate: str           # "pass", or the failing gate's code
    detail: str
    p_chrom: float
    bad: int
    nonfree: int

    @property
    def ok(self) -> bool:
        return self.gate == "pass" and self.p_chrom >= SOUR_MIN_P_CHROM

    def to_json(self) -> dict:
        return {"gate": self.gate, "detail": self.detail, "p_chrom": self.p_chrom, "bad_notes": self.bad,
                "nonfree_notes": self.nonfree, "ok": self.ok}


def check_sour(text: str, brief: Brief, voice: SaxVoice) -> SourCheck:
    """Gate a sour copy: G1-G6 in the brief's mood (the default vocabulary names `ep` and `dusty`), then
    P_chrom under `voice`. `ok` needs both."""
    res = parse(text, mode=brief.mood)
    if not res.gate.ok:
        return SourCheck(text, res.gate.code or "G1", res.gate.detail, 0.0, 0, 0)
    c = chromatic(res.ast, brief, voice)
    return SourCheck(text, "pass", "", c.p_chrom, c.bad, c.nonfree)
