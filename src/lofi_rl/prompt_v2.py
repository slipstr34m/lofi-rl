"""Prompt template v2: v1 (lofi_rl.prompt) with its one example loop chosen per brief. v1 stays the default.

    from lofi_rl import prompt_v2
    msgs = prompt_v2.build_messages(brief, palette)   # [{"role": "system", ...}, {"role": "user", ...}]
    prompt_v2.shown_examples(brief, palette)          # (("l4_wistful_low", text),): what the near-copy check reads

Why. The base-model test v0 (D-019; runs/junk-13632676/summary/junk.json) passed 98 of 500, and 78 of the 98 were
near-copies of v1's one example (character 5-gram Jaccard >= 0.8): the 56 mid/wistful briefs, the example's own
cell, gave 54 near-copies and 0 passes that were not copies. v2 shows each brief a loop written for another mood.

The rule (RULE): a brief of energy E and mood M is shown the loop written for energy E and the next mood,
melancholy -> wistful -> hopeful -> melancholy, labelled with that loop's own energy and mood. It reads only the two
knobs, never tempo or tonic. The example always shares the brief's energy and never its mood, and the 9 cells see 9
different loops:

    brief             example              its own cell
    low/melancholy    l4_wistful_low       low/wistful
    low/wistful       l7_hopeful_low       low/hopeful
    low/hopeful       l1_melancholy_low    low/melancholy
    mid/melancholy    example              mid/wistful      the control cell: v1's messages, byte for byte
    mid/wistful       l8_hopeful_mid       mid/hopeful
    mid/hopeful       l2_melancholy_mid    mid/melancholy
    high/melancholy   l6_wistful_high      high/wistful
    high/wistful      l9_hopeful_high      high/hopeful
    high/hopeful      l3_melancholy_high   high/melancholy

What changes from v1. SYSTEM_TEMPLATE is v1's with one line changed: the label "Example, for energy mid and mood
wistful (...)" names the example's own cell (V1_LABEL -> V2_LABEL). The grammar card, the vocabulary, the mood chords,
the three "Good loops" lines and the user message (USER_TEMPLATE) are v1's. l5_wistful_mid is not used: the mid/
wistful slot is v1's example, so mid/melancholy briefs get exactly v1's prompt.

The examples (EXAMPLES, keyed by their own cell in ENERGIES x MOOD_NAMES order) are embedded here, never read at run
time. Each is its source without the optional `sax:` line, with the keys instrument and the kit as {keys} and {kit}
(the shootout loops write `keys: ep` and `drums: dusty`, example.ls `keys: rhodes` and `drums: dusty`). Sources and
their sha256 when v2 was built (2026-10-31); tests/test_prompt_v2.py checks the templates against the files, so a
changed source needs a new prompt version:

    l1_melancholy_low    shootouts/loops/l1_melancholy_low.ls   9173fb9d3d9a41041f6dd52647cb62933737ef9518de7dfaebf380ae5f7703c2
    l2_melancholy_mid    shootouts/loops/l2_melancholy_mid.ls   81d554b78d06b3fafd63e97581e86f0db7adea232629283b2c858db2dc817224
    l3_melancholy_high   shootouts/loops/l3_melancholy_high.ls  6f2cb8405407685155bda197a38d68ff7d0b6cecc2126a69018cfe707f02c2f1
    l4_wistful_low       shootouts/loops/l4_wistful_low.ls      8208e5cf87d1fd9ff707a2fef8bb91c3a5ac8d29b726c0e9c23dfdf537d410d2
    example              tests/fixtures/loops/example.ls        232b798a30ebcbf5c498053d89b29856671459eadc2c33bc3dd72110ffeb7226
    l6_wistful_high      shootouts/loops/l6_wistful_high.ls     2985617db2ab0504458218649e1f258d5a4dba0f6696af3a15fd3a26cb024070
    l7_hopeful_low       shootouts/loops/l7_hopeful_low.ls      3eb1336cbf5c25c30bcc6a31b3f9c4acc4ef20a7f49d3ac035eb30e913822d07
    l8_hopeful_mid       shootouts/loops/l8_hopeful_mid.ls      b1126f66f09825faf497c234166bc6f81871b6ac5590670001e0a32d233c522f
    l9_hopeful_high      shootouts/loops/l9_hopeful_high.ls     0450383e912e4657ce89fd4bbd6c1240afea72f09f44fb8d0cbcf81e2ee1e5d5

{keys} and {kit} are filled as v1 fills its example: rhodes and dusty through the palette's aliases, else the
palette's first name (prompt._key). Each filled example is checked once per palette and cell: it parses under the
palette's vocabulary in its own mood with the gates passed and no repair at all, it is its own canonical text, and it
passes the gates (again with no repair) in the mood of the brief it is shown to. Any failure is a PromptError that
names the example and the palette. The examples need, from a palette: the qualities m9 maj7 maj9 9 13 m7, the
numerals I II III IV V bVI bVII, the voicing rootless, the lanes snare rim hat ohat shaker and `bass: lock`. A frozen
palette (D-017) without one of them makes its cell refuse: that calls for a v3, not an edit of v2.

Identity. `template_sha256()` hashes `template_text()`: the system template, the rule, the 9 unfilled examples with
their ids and cells, and the user template. tests/test_prompt_v2.py pins it as V2_SHA256 and still pins v1's.
`messages_sha256(build_messages(...))` names the exact text a run sent; junk-rate's generate.json records it per cell.

Size, dev_gu: 3,750-3,865 characters per prompt (system + user), under 1,500 tokens by the 4-characters rule. With the
Qwen3 BPE (the cached tokenizer.json) and the chat template: 1,322-1,408 tokens (system 1,295-1,381, user 14-16),
against v1's 1,389-1,391; the 9 prompts share their first 1,069 tokens.

The near-copy check (lofi_rl.rl.junk) compares each completion with the example its own prompt showed
(shown_examples), and `reference_texts` gives all 9 for the readout that compares with every example v2 can show.
GRPO still builds v1 (lofi_rl.rl.train): before a GRPO run uses v2, its pool-copy penalty (P_pool) must exist and
cover reference_texts(palette) as well as the seed pool.

Standard library only, like v1, so it imports in the policy env, the scorer env and on the laptop.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from types import MappingProxyType

from lofi_rl import prompt as v1
from lofi_rl.brief import ENERGIES, MOOD_NAMES, Brief
from lofi_rl.dsl import canon, parse
from lofi_rl.dsl.vocab import MOODS, Vocab
from lofi_rl.palette import Palette

PROMPT_VERSION = "v2"
V1_LABEL = "Example, for energy mid and mood wistful (it shows the form; write your own loop):"
V2_LABEL = "Example, for energy {example_energy} and mood {example_mood} (it shows the form; write your own loop):"
if v1.SYSTEM_TEMPLATE.count(V1_LABEL) != 1:
    raise AssertionError("prompt v1's example label must occur exactly once in its system template")
SYSTEM_TEMPLATE = v1.SYSTEM_TEMPLATE.replace(V1_LABEL, V2_LABEL)
USER_TEMPLATE = v1.USER_TEMPLATE
RULE = "example_cell(energy, mood) = (energy, MOOD_NAMES[(MOOD_NAMES.index(mood) + 1) % 3])"

PromptError = v1.PromptError
messages_sha256 = v1.messages_sha256
estimate_tokens = v1.estimate_tokens

_L1 = """\
vamp: Im9 bVImaj7 span 2
keys: {keys} rootless
comp: x-------------.. x---------x-----
bass: x-------..x----- x-----------....
drums: {kit}
kick: x.........x..... x.......x.......
rim: ....x.......x... ....x.......x...
hat: x.g.x.g.x.g.x.g. x.g.x.g.x.g.x.x.
@5 hat: x.g.x.g.x.g.x.gx
@6 kick: x.......x.x.....
@7 comp: x-------x-------
@8 comp: x---------x---x-
A: h5 q.4 e3 | q1 hr q3 | h.6 er e5 | q4 e3 e1 hr
B: h5 q.4 e3 | q1 e7, e1 hr | h.3 er eb5 | q4 e3 q.2 qr
"""

_L4 = """\
vamp: Im9 bVIImaj7 span 2
keys: {keys} rootless
comp: x--------------- x-----------x---
bass: lock
drums: {kit}
kick: x.......x....... x.........x.....
rim: ....x.......x... ....x.......x..g
hat: x...x.g.x...x.g. x...x.g.x...x.x.
@3 hat: x...x...x.g.x.g.
@4 kick: x.......x.......
@5 kick: x.....x.........
@6 hat: x...x.g.x...xxx.
@7 comp: x---------x-----
@8 comp: x-----x-----x---
A: h.5 e4 e3 | q1 qr q2 q3 | q4 e5 e6 h5 | q4 e3 e2 hr
B: h.5 e4 e3 | q1 qr e2 e3 q4 | h.6 er e7 | q5 q7 hr
"""

_L7 = """\
vamp: Imaj9 IVmaj9 span 2
keys: {keys} rootless
comp: x--------------- x-------x-------
bass: x------x-------- x---------------
drums: {kit}
kick: x......x........ x.........x.....
rim: ....x.......x... ....x.......x...
hat: x...x...x...x.x. x...x...x.x.x...
shaker: ..g...g...g...g. ..g...g...g...g.
@3 hat: x...x...x.x.x...
@4 kick: x.......x.......
@5 comp: x-------x-------
@6 kick: x......x..x.....
@7 hat: x...x.x.x...x.x.
@8 comp: x---------x-----
A: h3' q2' e1' e7 | q1' q5 hr | q.6 e5 h4 | q3 e2 eb3 q2 qr
B: h3' q2' e1' e7 | q1' e2' e3' q5' qr | h.4' e3' e2' | q1' eb7 e6 hr
"""

_L2 = """\
vamp: IVm9 bVII13 span 2
keys: {keys} rootless
comp: x-----x-----x--- x-------x-x-----
bass: x-----x---x----- x-------x----x--
drums: {kit}
kick: x.....x...x..... x.......x....x..
snare: ....x.......x... ....x..g....X..g
hat: x.x.x.x.x.x.x.x. x.x.x.x.x.x.x.xx
@4 hat: x.x.x.x.x.x.xxx.
@5 kick: x.....x...x...x.
@6 snare: ....x.g.....x..g
@7 ohat: ..........x.....
@8 comp: x-----x---x-----
A: h1' q7 e6 e5 | h.4 er e3 | q4 e5 e6 q5 qr | qr e4 e3 q1 qr
B: h1' q7 e6 e5 | q4 e3 e4 q5 qr | er e5 e6 eb5 h4 | q.2 e3 hr
"""

_EXAMPLE = """\
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

_L8 = """\
vamp: IIm9 V13 span 2
keys: {keys} rootless
comp: x-----.x----x--- x-------..x-x---
bass: lock
drums: {kit}
kick: X......x.x...... x.........x..x..
snare: ....x.......x... ....x..g....x...
hat: x.x.x.x.x.x.x.x. x.x.x.x.x.x.x.xx
@3 ohat: ......x.........
@4 kick: x.........x...x.
@5 kick: x......x.x....x.
@6 hat: x.x.x.x.x.xxx.x.
@7 snare: ....x..g....x..g
@8 comp: x-------..x--x--
A: h.2' e1' e7 | q6 e5 e6 q1' qr | h7 q6 e5 e4 | q3 e2 e1 hr
B: h.2' e1' e7 | q6 e5 eb5 q4 qr | q5 e6 e7 h1' | q.7 e2' hr
"""

_L3 = """\
vamp: bVImaj7 bVII13 span 2
keys: {keys} rootless
comp: x--x--x---x--x-- x-----x--x--x---
bass: x-----x-x-x----- x-------x-x---x-
drums: {kit}
kick: x.....x.x.x..... x.......x.x...x.
snare: ....X..g....X... ..g.X..g....X.g.
hat: xgxgxgxgxgxgxgxg xgxgxgxgxgxgxxXg
shaker: ..x...x...x...x. ..x...x...x...x.
@3 kick: x.....x.x.x...x.
@5 snare: ....X..g....X..g
@6 hat: xgxgxgxgxgxgxgxx
@7 comp: x--x--x---x-x---
@8 comp: x-----x--x--x-x-
A: h5 e6 e5 q3 | e1 e3 q4 q3 qr | q.7 e6 q5 e4 e3 | h1 hr
B: h5 e6 e5 q3 | e1 e3 q4 e3 e1 q7, | h.2 e3 eb5 | q4 qr hr
"""

_L6 = """\
vamp: Im9 IV9 span 1
keys: {keys} rootless
comp: x--x--x---x-x--- x-----x--x--x-x-
bass: x-----x---x-x--- x-------x-x---x-
drums: {kit}
kick: x.....x...x.x... x.......x.x...x.
snare: ....X..g....X..g ..g.X......gX...
hat: x.xxx.xxx.xxx.xx x.xxx.xxx.xxx.x.
shaker: ..x...x...x...x. ..x...x...x...x.
@3 kick: x.....x.x.x.x...
@4 hat: x.xxx.xxx.xxxxx.
@5 snare: ....X..g....X.g.
@6 ohat: ......x.........
@7 comp: x--x--x---x--x--
@8 comp: x-----x--x--x---
A: h5 q4 e3 e4 | q.6 e5 q4 qr | e3 e2 q1 h5, | q.4 e3 hr
B: h5 q4 e3 e4 | q.6 e7 q1' qr | h.7 e6 e5 | q4 e5 e4 hr
"""

_L9 = """\
vamp: IVmaj7 IIIm7 span 1
keys: {keys} rootless
comp: x--x---x--x---x- x-----x---x--x--
bass: x-----x-x-----x- x-x-------x-----
drums: {kit}
kick: x.....x.x.....x. x.x.......x.....
snare: ....X.......X..g ..g.X..g....X...
hat: xxx.xxx.xxx.xxx. xxx.xxx.xxx.x.x.
shaker: .x.x.x.x.x.x.x.x .x.x.x.x.x.x.x.x
@3 kick: x.....x.x.x...x.
@4 hat: xxx.xxx.xxx.xxxx
@5 snare: ....X..g....X.g.
@6 comp: x--x--x---x--x--
@7 ohat: ......x.........
@8 comp: x-----x---x-x---
A: h6 e5 e4 q3 | q.5 e3 q2 qr | e1 e2 q3 h4 | q3 e2 e1 hr
B: h6 e5 e4 q3 | q.5 e6 q7 qr | h.1' eb7 e6 | q5 e4 e3 hr
"""

# (energy, mood) of the example's own brief -> (id, template), in ENERGIES x MOOD_NAMES order
EXAMPLES = MappingProxyType({
    ("low", "melancholy"): ("l1_melancholy_low", _L1),
    ("low", "wistful"): ("l4_wistful_low", _L4),
    ("low", "hopeful"): ("l7_hopeful_low", _L7),
    ("mid", "melancholy"): ("l2_melancholy_mid", _L2),
    ("mid", "wistful"): ("example", _EXAMPLE),
    ("mid", "hopeful"): ("l8_hopeful_mid", _L8),
    ("high", "melancholy"): ("l3_melancholy_high", _L3),
    ("high", "wistful"): ("l6_wistful_high", _L6),
    ("high", "hopeful"): ("l9_hopeful_high", _L9),
})
if tuple(EXAMPLES) != tuple((e, m) for e in ENERGIES for m in MOOD_NAMES):
    raise AssertionError("EXAMPLES must hold one example per cell, in ENERGIES x MOOD_NAMES order")
if EXAMPLES[("mid", "wistful")][1] != v1.EXAMPLE_TEMPLATE:
    raise AssertionError("the mid/wistful example must be prompt v1's example template")


class ReferenceTexts(dict):
    """{example id: filled example text}, in EXAMPLES order; `refused` maps each id that was skipped (strict=False)
    to the PromptError that refused it."""

    def __init__(self) -> None:
        super().__init__()
        self.refused: dict[str, str] = {}


def _cell(energy: str, mood: str) -> tuple[str, str]:
    return energy, MOOD_NAMES[(MOOD_NAMES.index(mood) + 1) % len(MOOD_NAMES)]


def _brief_mood(example_mood: str) -> str:
    """The mood of the briefs an example of `example_mood` is shown to: the rule run backwards."""
    return MOOD_NAMES[(MOOD_NAMES.index(example_mood) - 1) % len(MOOD_NAMES)]


def example_cell(brief: Brief) -> tuple[str, str]:
    """(energy, mood) of the example a brief is shown: its own energy and the next mood (RULE)."""
    return _cell(brief.energy, brief.mood)


def example_id(brief: Brief) -> str:
    return EXAMPLES[example_cell(brief)][0]


def example_text(palette: Palette, brief: Brief) -> str:
    """The example this brief is shown, in the palette's names: canonical text with no `sax:` line. PromptError if
    it does not pass the palette's gates with no repair, in its own mood and in the brief's."""
    return _parts(v1._key(palette), example_cell(brief))[1]


def shown_examples(brief: Brief, palette: Palette) -> tuple[tuple[str, str], ...]:
    """Every complete loop this brief's prompt shows, as (id, text): one in v2."""
    return ((example_id(brief), example_text(palette, brief)),)


def reference_texts(palette: Palette, strict: bool = True) -> ReferenceTexts:
    """All 9 examples v2 can show, {id: text} in EXAMPLES order. strict=False skips the ones the palette refuses
    (PromptError) and lists them in `.refused`; strict=True raises the first."""
    key = v1._key(palette)
    out = ReferenceTexts()
    for cell, (ex_id, _) in EXAMPLES.items():
        try:
            out[ex_id] = _parts(key, cell)[1]
        except PromptError as err:
            if strict:
                raise
            out.refused[ex_id] = str(err)
    return out


def system_message(palette: Palette, brief: Brief) -> str:
    return _parts(v1._key(palette), example_cell(brief))[0]


def user_message(brief: Brief) -> str:
    return v1.user_message(brief)


def build_messages(brief: Brief, palette: Palette) -> list[dict]:
    """[{"role": "system", ...}, {"role": "user", ...}] for one brief: fresh dicts on every call."""
    return [{"role": "system", "content": system_message(palette, brief)},
            {"role": "user", "content": user_message(brief)}]


def template_text() -> str:
    """The template with no palette and no brief filled in: the system part, the rule, the 9 examples (id and own
    cell each) and the user part."""
    examples = "".join(f"[example {ex_id} {energy}/{mood}]\n{template}"
                       for (energy, mood), (ex_id, template) in EXAMPLES.items())
    return f"[system]\n{SYSTEM_TEMPLATE}\n[rule]\n{RULE}\n{examples}[user]\n{USER_TEMPLATE}\n"


def template_sha256() -> str:
    """sha256 of `template_text()`: the identity of PROMPT_VERSION, whatever the palette."""
    return hashlib.sha256(template_text().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------- internals
@lru_cache(maxsize=128)
def _parts(key: tuple, cell: tuple[str, str]) -> tuple[str, str]:
    """(system message, example text) for one palette key (prompt._key) and one example cell."""
    palette_id, vocab_items, alias_items, keys_name, kit_name = key
    ex_id, template = EXAMPLES[cell]
    energy, mood = cell
    where = f"example {ex_id} ({energy}/{mood})"
    vocab = dict(vocab_items)
    v = Vocab(keys=frozenset(vocab["keys"]), kits=frozenset(vocab["kits"]), sax=frozenset(),
              voicings=frozenset(vocab["voicings"]), numerals=frozenset(vocab["numerals"]),
              qualities=frozenset(vocab["qualities"]), name_aliases=dict(alias_items))
    raw = template.format(keys=keys_name, kit=kit_name)
    res = parse(raw, vocab=v, mode=MOODS[mood])
    if not res.gate.ok:
        raise PromptError(f"{where} does not pass the gates of palette {palette_id}: {res.gate.code} "
                          f"{res.gate.detail}")
    example = canon(res.ast)
    if res.repairs or example != raw:
        raise PromptError(f"{where} is not canonical under palette {palette_id}: "
                          f"repairs {[r.code for r in res.repairs]}")
    shown_to = _brief_mood(mood)
    again = parse(raw, vocab=v, mode=MOODS[shown_to])
    if not again.gate.ok or again.repairs:
        raise PromptError(f"{where} does not pass the gates of palette {palette_id} in the mood of the briefs it is "
                          f"shown to ({shown_to}): {again.gate.code} {again.gate.detail} "
                          f"repairs {[r.code for r in again.repairs]}")
    fits = {m: " ".join(chords) or "(none in this palette)" for m, chords in v1._fitting(vocab).items()}
    system = SYSTEM_TEMPLATE.format(
        numerals=" ".join(vocab["numerals"]), qualities=" ".join(vocab["qualities"]),
        keys=" ".join(vocab["keys"]), voicings=" ".join(vocab["voicings"]), kits=" ".join(vocab["kits"]),
        example_energy=energy, example_mood=mood, example=example, **fits)
    return system, example
