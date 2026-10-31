"""Prompt template v1: the chat messages that ask the base model for one LoopScript per brief.

    from lofi_rl.prompt import PROMPT_VERSION, build_messages, template_sha256
    msgs = build_messages(brief, palette)    # [{"role": "system", ...}, {"role": "user", ...}]

The system message is the same for every brief of a palette, so vLLM's prefix cache computes it once. It holds:
1. a compact grammar card distilled from docs/DSL.md §3-4: statements, grids and phrase syntax;
2. the palette's vocabulary, every list `palette.vocab()` checks a chord or a name against: chord numerals and
   qualities, keys instruments, voicing styles, drum kits. Sax names are left out, because `sax:` is optional
   and ignored (DSL.md §4), so a loop that names no sax can never fail G4 on one; aliases are left out, because
   they are leniency for slips, not words to teach;
3. what the two knobs mean (TASTE.md, Energy and Briefs): energy is how busy the loop is, and the mood lives in
   the chords, with the chords of the palette whose notes all lie in each mood's notes;
4. three lines on what a good loop does, each from a TASTE.md row (drums and bass carry it, something new every
   bar, no fill at the seam, a few long sax notes, an answer that hangs open);
5. one example loop: tests/fixtures/loops/example.ls (instrument.md §1) in the palette's own names, without its
   optional `sax:` line, labelled with the brief it was written for (energy mid, mood wistful);
6. "Write only the loop".

The user message shows only the brief's scene and its energy and mood knobs. The model never sees the tempo, the
tonic, the mode or any pitch: it writes degrees, numerals and grids, and the compiler places them (TASTE.md; the
brief's knobs pick the mode and the energy). tests/test_prompt.py checks that no tempo, key, mode or forbidden word
and no note name reaches either message.

Identity. `template_sha256()` hashes the unfilled template (`template_text()`: the system, example and user
texts), so it names the version whatever the palette; `messages_sha256(build_messages(...))` names the exact text a
run sent. Size: under 1,500 tokens by the 4-characters-per-token rule (`estimate_tokens`). LoopScript runs denser
than that rule: tests/fixtures/loops/example.ls is 585 characters and 269 Qwen3-2507 tokens
(runs/smoke-13628361/summary/policy.json), about 2.2 characters per token, so tests/test_prompt.py also caps the
characters; the junk-rate run records vLLM's exact prompt token count.

Standard library only, so it imports in the policy env, the scorer env and on the laptop.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from math import ceil

from lofi_rl.brief import Brief
from lofi_rl.dsl import canon, parse
from lofi_rl.dsl.vocab import MODES, MOODS, NUMERAL_SEMITONES, NUMERALS, QUALITIES, VOICINGS, Vocab
from lofi_rl.palette import Palette, PaletteError
from lofi_rl.theory.chords import QUALITY

PROMPT_VERSION = "v1"
EXAMPLE_BRIEF = {"energy": "mid", "mood": "wistful"}   # the brief instrument.md §1 wrote the example for
EXAMPLE_NAMES = {"keys": "rhodes", "kits": "dusty"}     # the example's own names, resolved through the palette
CHARS_PER_TOKEN = 4                                      # the estimate's rule

# tests/fixtures/loops/example.ls without its `sax: tenor` line; {keys} and {kit} become the palette's names.
EXAMPLE_TEMPLATE = """\
vamp: Im9 IV13 span 2
keys: {keys} rootless
comp: x-----.x----..x- x-------..x-x---
bass: x-------..x-x--- x-----.x-x------
drums: {kit}
kick: x......x..x..... x......x.x....x.
rim: ....X.......X... ....X..g....X..g
hat: x.x.x.x.x.x.x.x. x.x.x.x.x.x.xxx.
ohat: ................ ..............x.
@3 hat: x.x.x.x.x.x.x.xx
@4 kick: x......x..x...x.
@5 comp: x-----.x--x-..x-
@6 rim: ....X..g....X.g.
@7 ohat: ......x.........
@8 comp: x-----.x----x-x-
A: q5 e6 e5 q3 e2 e1 | h3 er e4 e5 e6 | h.5 er e4 | q3 q2 hr
B: q1' e7 e1' q5 q3 | er e3 e4 eb5 h5 | qr e4 e3 q2 q1 | h.6 qr
"""

SYSTEM_TEMPLATE = """\
You write LoopScript: the text of one 8-bar jazzhop loop for a brief. The compiler plays it and sets everything \
the text leaves out: speed, sounds, mix, and the real notes behind each degree and chord. The loop repeats; \
each bar has 16 steps (sixteenth notes).

Grammar, one statement per line:
vamp: CHORD CHORD span 2   two different chords; span 2 plays each for 2 bars (1 1 2 2 1 1 2 2), span 1 alternates them
keys: INSTRUMENT VOICING   the keys player
comp: SGRID [SGRID]        keys rhythm: every strike plays the bar's chord
bass: SGRID [SGRID]        bass rhythm, always on the chord root; bass: lock follows the kick instead
drums: KIT
kick: DGRID [DGRID]        drum lanes: kick snare rim hat ohat shaker (ohat is the open hat)
@N LANE: GRID              bar N (1-8) only: replaces that bar of comp, bass or one drum lane
A: PHRASE                  the sax phrase, bars 1-4
B: PHRASE                  the sax answer, bars 5-8
Required: vamp, keys, comp, bass, drums, kick, snare or rim, A:, B:.

Each grid has 16 cells, one per step of a bar. One grid repeats in every bar; two grids alternate.
DGRID cells: X accent, x hit, g ghost, . rest. SGRID cells: the same, plus - to hold the strike before it.
Each PHRASE fills 64 steps (4 bars) with items, with | between bars. An item is a duration, then a degree, or r for a rest.
Durations in steps: s 1, e 2, q 4, h 8, w 16; a dot adds half: e. 3, q. 6, h. 12, w. 24 (no s.).
Degrees 1-7 count from home (1) in the mood's notes, as chord numerals do: I is home, IV is degree 4.
b or # before a degree lowers or raises it a half step (eb5); ' after it is an octave up, and , an octave down (q1', h5,).
So q5 is degree 5 for 4 steps, h.3 is degree 3 for 12 steps, and er is a 2-step rest.

Allowed names (nothing else is accepted):
CHORD = numeral + quality, like Im9 or IV13
numerals: {numerals}
qualities: {qualities}
keys instruments: {keys}
voicings: {voicings}
drum kits: {kits}

The brief gives a scene and two knobs.
Energy (low, mid, high) is how busy the loop is: more energy means more drum hits and X accents, and more comp and bass strikes.
Mood (melancholy, wistful, hopeful) lives in the chords. Chords whose notes all fit each mood:
melancholy: {melancholy}
wistful: {wistful}
hopeful: {hopeful}

Good loops:
- Kick on step 1, and snare or rim on steps 5 and 13, of every bar: drums and bass carry the groove.
- Something changes in every bar (use @N edits), and bar 8 keeps the groove, with no fill.
- The sax plays a few long notes (h or longer), with rests. A: and B: each start with a note on their first step. \
B: answers A:, starting like it and ending differently, not on degree 1.

Example, for energy mid and mood wistful (it shows the form; write your own loop):
{example}
Write only the loop, one statement per line: no title, no explanation."""

USER_TEMPLATE = """\
Scene: {scene}
Energy: {energy}
Mood: {mood}"""


class PromptError(ValueError):
    """The template cannot serve this palette (its example loop does not pass the palette's gates)."""


def template_text() -> str:
    """The template with no palette and no brief filled in: the system part, the example it embeds, the user part.
    The code that fills it (the vocabulary order, the mood chords) is not text; `messages_sha256` covers it."""
    return f"[system]\n{SYSTEM_TEMPLATE}\n[example]\n{EXAMPLE_TEMPLATE}[user]\n{USER_TEMPLATE}\n"


def template_sha256() -> str:
    """sha256 of `template_text()`: the identity of PROMPT_VERSION, whatever the palette."""
    return hashlib.sha256(template_text().encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Tokens by the 4-characters-per-token rule, rounded up (an estimate, not a tokenizer)."""
    return ceil(len(text) / CHARS_PER_TOKEN)


def messages_sha256(messages: list[dict]) -> str:
    """sha256 of the messages as canonical JSON (sorted keys, no spaces, UTF-8)."""
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def vocabulary(palette: Palette) -> dict[str, tuple[str, ...]]:
    """The lists the prompt shows, as `palette.vocab()` holds them: the grammar's own order for numerals,
    qualities and voicings, alphabetical order for the palette's names (sets have no order of their own)."""
    v = palette.vocab()
    return {"numerals": tuple(n for n in NUMERALS if n in v.numerals),
            "qualities": tuple(q for q in QUALITIES if q in v.qualities),
            "keys": tuple(sorted(v.keys)),
            "voicings": tuple(s for s in VOICINGS if s in v.voicings),
            "kits": tuple(sorted(v.kits))}


def mood_chords(palette: Palette) -> dict[str, tuple[str, ...]]:
    """For each mood, the palette's chords whose every tone (the chord-tones table) lies in the mood's mode, in
    the grammar's order. They fit the mood by construction; other chords stay allowed (borrowed chords,
    TASTE.md Harmony)."""
    return _fitting(vocabulary(palette))


def example_text(palette: Palette) -> str:
    """The in-prompt example: its keys instrument and kit as the palette resolves them, or the palette's first
    name (alphabetically) when it cannot; canonical text, so the junk-rate's near-copy check compares like with
    like. PromptError if the example does not pass the palette's gates with no repair."""
    return _system_parts(_key(palette))[1]


def system_message(palette: Palette) -> str:
    return _system_parts(_key(palette))[0]


def user_message(brief: Brief) -> str:
    return USER_TEMPLATE.format(scene=brief.scene, energy=brief.energy, mood=brief.mood)


def build_messages(brief: Brief, palette: Palette) -> list[dict]:
    """[{"role": "system", ...}, {"role": "user", ...}] for one brief: fresh dicts on every call."""
    return [{"role": "system", "content": system_message(palette)},
            {"role": "user", "content": user_message(brief)}]


# --------------------------------------------------------------------------------------------- internals
def _fitting(vocab: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    out = {}
    for mood, mode in MOODS.items():
        notes = set(MODES[mode])
        out[mood] = tuple(n + q for n in vocab["numerals"] for q in vocab["qualities"]
                          if all((NUMERAL_SEMITONES[n] + t) % 12 in notes for t in QUALITY[q].tones))
    return out


def _resolve(palette: Palette, kind: str) -> str:
    """The example's own name for `kind` through the palette's aliases, else the palette's first name."""
    try:
        return palette.resolve(kind, EXAMPLE_NAMES[kind])
    except PaletteError:
        names = sorted(getattr(palette, kind))
        if not names:
            raise PromptError(f"palette {palette.id} has no {kind} name for the example") from None
        return names[0]


def _key(palette: Palette) -> tuple:
    """Everything the system message depends on, as a hashable cache key (a Palette holds mappings)."""
    return (palette.id, tuple(vocabulary(palette).items()), tuple(sorted(palette.vocab().name_aliases.items())),
            _resolve(palette, "keys"), _resolve(palette, "kits"))


@lru_cache(maxsize=16)
def _system_parts(key: tuple) -> tuple[str, str]:
    """(system message, example text) for one cache key; the example must pass the gates with no repair."""
    palette_id, vocab_items, alias_items, keys_name, kit_name = key
    vocab = dict(vocab_items)
    v = Vocab(keys=frozenset(vocab["keys"]), kits=frozenset(vocab["kits"]), sax=frozenset(),
              voicings=frozenset(vocab["voicings"]), numerals=frozenset(vocab["numerals"]),
              qualities=frozenset(vocab["qualities"]), name_aliases=dict(alias_items))
    raw = EXAMPLE_TEMPLATE.format(keys=keys_name, kit=kit_name)
    res = parse(raw, vocab=v, mode=MOODS[EXAMPLE_BRIEF["mood"]])
    if not res.gate.ok:
        raise PromptError(f"the example loop does not pass the gates of palette {palette_id}: "
                          f"{res.gate.code} {res.gate.detail}")
    example = canon(res.ast)
    if res.repairs or example != raw:
        raise PromptError(f"the example loop is not canonical under palette {palette_id}: "
                          f"repairs {[r.code for r in res.repairs]}")
    fits = {mood: " ".join(chords) or "(none in this palette)" for mood, chords in _fitting(vocab).items()}
    system = SYSTEM_TEMPLATE.format(
        numerals=" ".join(vocab["numerals"]), qualities=" ".join(vocab["qualities"]),
        keys=" ".join(vocab["keys"]), voicings=" ".join(vocab["voicings"]), kits=" ".join(vocab["kits"]),
        example=example, **fits)
    return system, example
