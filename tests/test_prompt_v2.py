"""Prompt template v2 (lofi_rl.prompt_v2): v1 with its one example loop chosen per brief by a fixed rule.

What these tests pin:
- identity: template_sha256() is pinned as V2_SHA256 and v1's pin (tests/test_prompt.py) still holds, so v1 stays
  byte-identical next to v2;
- the template: v1's system template with one line changed (the example label), v1's user template, the rule;
- the rule: the 9-cell table (the brief's energy, the next mood), deterministic, 9 different examples, and the
  mid/melancholy control cell, whose messages are v1's byte for byte;
- the examples: each is its source file without `sax:` in the palette's names, passes the gates with no repair in its
  own mood and in the mood of the briefs it is shown to, is canonical, is in the LoopScript regex's language, and
  sits in its own energy band and mood;
- no leak (tests/test_prompt.py's checks on all 9 cells), determinism across processes, the size caps, the
  vocabulary following the palette, the refusal of a palette that cannot play an example, and the trainer's
  message check.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from lofi_rl import grammar, prompt, prompt_v2
from lofi_rl.brief import ENERGIES, MOOD_NAMES, Brief
from lofi_rl.dsl import canon, parse
from lofi_rl.dsl.vocab import FORBIDDEN_WORDS, MOODS, forbidden_kind
from lofi_rl.palette import load_palette

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "palettes" / "dev_gu.yaml"
# The template's sha256. Changing the template text (system, rule, an example, user) changes this: make a v3.
V2_SHA256 = "127e3094fd5a2183fb952059c2221ad018bdef6d062f51a80b14367b8064466e"
V1_SHA256 = "7aceb2bc0d7339a7d3c9af7bf0d808ee156d4f7a8684e0393d923940fa5c1e68"     # tests/test_prompt.py's pin
# brief cell -> (example id, the example's own cell): the spec's table (the brief's energy, the next mood)
TABLE = {
    ("low", "melancholy"): ("l4_wistful_low", ("low", "wistful")),
    ("low", "wistful"): ("l7_hopeful_low", ("low", "hopeful")),
    ("low", "hopeful"): ("l1_melancholy_low", ("low", "melancholy")),
    ("mid", "melancholy"): ("example", ("mid", "wistful")),
    ("mid", "wistful"): ("l8_hopeful_mid", ("mid", "hopeful")),
    ("mid", "hopeful"): ("l2_melancholy_mid", ("mid", "melancholy")),
    ("high", "melancholy"): ("l6_wistful_high", ("high", "wistful")),
    ("high", "wistful"): ("l9_hopeful_high", ("high", "hopeful")),
    ("high", "hopeful"): ("l3_melancholy_high", ("high", "melancholy")),
}
SOURCES = {ex_id: (ROOT / "tests" / "fixtures" / "loops" / "example.ls" if ex_id == "example"
                   else ROOT / "shootouts" / "loops" / f"{ex_id}.ls") for ex_id, _ in TABLE.values()}
SOURCE_SHA256 = {   # at build time (2026-10-31); the module docstring records the same
    "l1_melancholy_low": "9173fb9d3d9a41041f6dd52647cb62933737ef9518de7dfaebf380ae5f7703c2",
    "l2_melancholy_mid": "81d554b78d06b3fafd63e97581e86f0db7adea232629283b2c858db2dc817224",
    "l3_melancholy_high": "6f2cb8405407685155bda197a38d68ff7d0b6cecc2126a69018cfe707f02c2f1",
    "l4_wistful_low": "8208e5cf87d1fd9ff707a2fef8bb91c3a5ac8d29b726c0e9c23dfdf537d410d2",
    "example": "232b798a30ebcbf5c498053d89b29856671459eadc2c33bc3dd72110ffeb7226",
    "l6_wistful_high": "2985617db2ab0504458218649e1f258d5a4dba0f6696af3a15fd3a26cb024070",
    "l7_hopeful_low": "3eb1336cbf5c25c30bcc6a31b3f9c4acc4ef20a7f49d3ac035eb30e913822d07",
    "l8_hopeful_mid": "b1126f66f09825faf497c234166bc6f81871b6ac5590670001e0a32d233c522f",
    "l9_hopeful_high": "0450383e912e4657ce89fd4bbd6c1240afea72f09f44fb8d0cbcf81e2ee1e5d5",
}
TAIL = "\nWrite only the loop, one statement per line: no title, no explanation."
VOCAB_HEADS = ("numerals: ", "qualities: ", "keys instruments: ", "voicings: ", "drum kits: ",
               "melancholy: ", "wistful: ", "hopeful: ")


@pytest.fixture(scope="module")
def pal():
    return load_palette(DEV, verify=False)


def _brief(energy: str, mood: str, bpm: int = 82, tonic: int = 2) -> Brief:
    return Brief(energy=energy, mood=mood, bpm=bpm, tonic=tonic)


def _cells() -> list[tuple[str, str]]:
    return [(e, m) for e in ENERGIES for m in MOOD_NAMES]


def _text(msgs: list[dict]) -> str:
    return "\n".join(m["content"] for m in msgs)


def _shown_mood(example_mood: str) -> str:
    """The mood of the briefs shown an example of `example_mood` (the rule run backwards)."""
    return next(m for (_, m), (_, cell) in TABLE.items() if cell[1] == example_mood)


# ------------------------------------------------------------------------------------------- identity and template
def test_the_version_and_the_template_sha_are_pinned_and_v1s_pin_holds():
    assert prompt_v2.PROMPT_VERSION == "v2" and prompt.PROMPT_VERSION == "v1"
    assert prompt_v2.template_sha256() == V2_SHA256, (
        "the v2 template changed (system, rule, an example or user): make it a new version, v3, and pin that; "
        "a run's generate.json names the sha it was sent")
    assert prompt.template_sha256() == V1_SHA256, "v1 stays byte-identical next to v2"
    assert prompt_v2.template_sha256() != prompt.template_sha256()
    assert prompt_v2.template_sha256() == hashlib.sha256(prompt_v2.template_text().encode("utf-8")).hexdigest()


def test_template_text_is_the_unfilled_system_rule_examples_and_user_parts():
    text = prompt_v2.template_text()
    want = ("[system]\n" + prompt_v2.SYSTEM_TEMPLATE + "\n[rule]\n" + prompt_v2.RULE + "\n"
            + "".join(f"[example {ex_id} {e}/{m}]\n{template}" for (e, m), (ex_id, template) in prompt_v2.EXAMPLES.items())
            + "[user]\n" + prompt_v2.USER_TEMPLATE + "\n")
    assert text == want
    for field in ("{scene}", "{energy}", "{mood}", "{numerals}", "{qualities}", "{keys}", "{kits}", "{voicings}",
                  "{example}", "{example_energy}", "{example_mood}", "{kit}", "{melancholy}", "{wistful}", "{hopeful}"):
        assert field in text, f"the unfilled template keeps {field}"
    assert "city at night" not in text, "no brief is filled in"
    assert all(template in text for _, template in prompt_v2.EXAMPLES.values()), "the sha covers all 9 examples"
    assert prompt_v2.RULE == "example_cell(energy, mood) = (energy, MOOD_NAMES[(MOOD_NAMES.index(mood) + 1) % 3])"


def test_the_system_template_is_v1s_with_only_the_label_line_changed():
    assert prompt.SYSTEM_TEMPLATE.count(prompt_v2.V1_LABEL) == 1
    assert prompt_v2.V1_LABEL == "Example, for energy mid and mood wistful (it shows the form; write your own loop):"
    assert prompt_v2.V2_LABEL == ("Example, for energy {example_energy} and mood {example_mood} "
                                  "(it shows the form; write your own loop):")
    old, new = prompt.SYSTEM_TEMPLATE.splitlines(), prompt_v2.SYSTEM_TEMPLATE.splitlines()
    assert len(old) == len(new)
    assert [(a, b) for a, b in zip(old, new) if a != b] == [(prompt_v2.V1_LABEL, prompt_v2.V2_LABEL)]
    assert prompt_v2.USER_TEMPLATE == prompt.USER_TEMPLATE


# ------------------------------------------------------------------------------------------- the rule
def test_the_rule_is_the_nine_cell_table_with_the_briefs_energy_and_the_next_mood():
    assert tuple(prompt_v2.EXAMPLES) == tuple(_cells()), "keyed by the example's own cell, ENERGIES x MOOD_NAMES"
    for (e, m), (ex_id, cell) in TABLE.items():
        b = _brief(e, m)
        assert prompt_v2.example_cell(b) == cell and prompt_v2.example_id(b) == ex_id, f"{e}/{m}"
        assert prompt_v2.EXAMPLES[cell][0] == ex_id
        assert cell == (e, MOOD_NAMES[(MOOD_NAMES.index(m) + 1) % 3]), "the RULE formula"
        assert cell[0] == e, "the energy is always kept"
        assert cell[1] != m, "the mood is never kept"
    ids = [ex_id for ex_id, _ in TABLE.values()]
    assert len(set(ids)) == 9, "a bijection: 9 cells, 9 examples"
    assert "l5_wistful_mid" not in ids and {ex_id for ex_id, _ in prompt_v2.EXAMPLES.values()} == set(ids)


def test_the_control_cell_mid_melancholy_is_v1s_prompt_byte_for_byte(pal):
    for bpm in (60, 75, 82, 95):
        for tonic in (0, 2, 7, 11):
            b = _brief("mid", "melancholy", bpm, tonic)
            assert prompt_v2.build_messages(b, pal) == prompt.build_messages(b, pal)
            assert prompt_v2.system_message(pal, b) == prompt.system_message(pal)
            assert prompt_v2.example_text(pal, b) == prompt.example_text(pal)
            assert prompt_v2.messages_sha256(prompt_v2.build_messages(b, pal)) == prompt.messages_sha256(
                prompt.build_messages(b, pal))


# ------------------------------------------------------------------------------------------- the examples
def test_each_example_is_its_source_without_sax_in_dev_gus_names(pal):
    refs = prompt_v2.reference_texts(pal)
    for (e, m), (ex_id, _) in prompt_v2.EXAMPLES.items():
        raw = SOURCES[ex_id].read_bytes()
        assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256[ex_id], (
            f"{SOURCES[ex_id].relative_to(ROOT)} changed since v2 was built: v2 embeds the old text, so a changed "
            "source needs a new prompt version")
        assert SOURCE_SHA256[ex_id] in prompt_v2.__doc__, "the docstring records the source's sha256"
        if ex_id == "example":
            want = prompt.example_text(pal)
        else:
            lines = [line for line in raw.decode("utf-8").splitlines() if not line.startswith("sax:")]
            lines = [line.replace("keys: ep ", "keys: rhodes ").replace("drums: dusty", "drums: jazz") for line in lines]
            want = "\n".join(lines) + "\n"
        assert refs[ex_id] == want, (f"{ex_id} is no longer its source in dev_gu's names: a changed source needs a "
                                     "new prompt version")
        assert prompt_v2.EXAMPLES[(e, m)][1].format(keys="rhodes", kit="jazz") == want
    assert prompt_v2.EXAMPLES[("mid", "wistful")][1] == prompt.EXAMPLE_TEMPLATE


def test_each_example_passes_the_gates_with_no_repair_is_canonical_and_in_the_regexs_language(pal):
    full = re.compile(grammar.loopscript_regex(pal)).fullmatch
    for (e, m), (ex_id, _) in prompt_v2.EXAMPLES.items():
        text = prompt_v2.reference_texts(pal)[ex_id]
        for mood in (m, _shown_mood(m)):                # its own mood, then the mood of the briefs it is shown to
            res = parse(text, vocab=pal.vocab(), mode=MOODS[mood])
            assert res.gate.ok and res.repairs == [], (ex_id, mood, res.gate, [r.code for r in res.repairs])
            assert canon(res.ast) == text, f"{ex_id} is canonical"
        assert full(text), f"{ex_id} is in the LoopScript regex's language (docs/GRAMMAR.md)"
        assert not re.search(r"^sax:", text, re.M)


def test_each_example_sits_in_its_own_energy_band_and_mood_and_not_in_the_briefs_mood(pal):
    from lofi_rl.verify.knobs import energy_band
    from lofi_rl.verify.scoring import score
    measured = {"low": (0.289, 0.297), "mid": (0.462, 0.503), "high": (0.700, 0.720)}      # dev_gu, rounded
    for (e, m), (ex_id, _) in prompt_v2.EXAMPLES.items():
        text = prompt_v2.reference_texts(pal)[ex_id]
        own = score(text, _brief(e, m), pal)
        shown_to = score(text, _brief(e, _shown_mood(m)), pal)
        assert own.gate_ok and shown_to.gate_ok
        energy = own.knob_parts["E"]
        assert energy_band(energy) == e, (ex_id, energy)
        lo, hi = measured[e]
        assert lo <= round(energy, 3) <= hi, (ex_id, energy)
        assert own.knob_parts["f_mood"] == 1.0, f"{ex_id} is in its own mood"
        assert shown_to.knob_parts["f_mood"] == 0.0, f"{ex_id} is not in the mood of the briefs it is shown to"


# ------------------------------------------------------------------------------------------- determinism and leaks
def test_tempo_and_tonic_never_reach_the_prompt_and_the_nine_cells_differ(pal):
    shas = {}
    for e, m in _cells():
        variants = {prompt_v2.messages_sha256(prompt_v2.build_messages(_brief(e, m, bpm, t), pal))
                    for bpm in (60, 75, 82, 95) for t in range(12)}
        assert len(variants) == 1, f"tempo or tonic reached the prompt for {e}/{m}"
        shas[(e, m)] = variants.pop()
    assert len(set(shas.values())) == 9


def test_no_tempo_key_mode_forbidden_word_note_name_or_sax_in_any_cell(pal):
    banned = FORBIDDEN_WORDS | {"tonic", "pitch", "pitches", "hz", "semitone", "semitones"}
    for e, m in _cells():
        for bpm, tonic in ((60, 0), (82, 2), (95, 11)):
            b = _brief(e, m, bpm, tonic)
            msgs = prompt_v2.build_messages(b, pal)
            for msg in msgs:
                text = msg["content"]
                words = set(re.findall(r"[a-z]+", text.lower()))
                assert not words & banned, (e, m, msg["role"], sorted(words & banned))
                assert str(b.bpm) not in re.findall(r"\d+", text), (e, m, msg["role"], b.bpm)
                assert not words & set(pal.sax), "no sax name"
                assert not words & set(pal.aliases), "no alias (dusty, wurli, ...)"
                assert not re.search(r"^sax:", text, re.M)
                for raw in text.split():
                    if raw in ("A:", "B:"):
                        continue
                    token = raw.strip(",.;()")
                    if token and token not in ("A:", "B:"):
                        assert forbidden_kind(token) != "a note name or absolute chord", (e, m, repr(raw))


def test_the_messages_are_the_same_in_fresh_processes_under_other_hash_seeds(pal):
    code = ("from lofi_rl import prompt_v2; from lofi_rl.brief import Brief; from lofi_rl.palette import load_palette; "
            f"p = load_palette({str(DEV)!r}, verify=False); "
            "print(*(prompt_v2.messages_sha256(prompt_v2.build_messages(Brief(energy=e, mood=m, bpm=70, tonic=4), p)) "
            "for e, m in (('high', 'hopeful'), ('mid', 'melancholy'))))")
    outs = set()
    for seed in ("0", "1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT / "src")}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
        assert out.returncode == 0, out.stderr
        outs.add(out.stdout.strip())
    here = " ".join(prompt_v2.messages_sha256(prompt_v2.build_messages(_brief(e, m, 70, 4), pal))
                    for e, m in (("high", "hopeful"), ("mid", "melancholy")))
    assert outs == {here}


def test_every_system_message_is_v1s_head_then_the_label_and_its_example_then_v1s_tail(pal):
    head = prompt.system_message(pal).split("Example, for energy ")[0] + "Example, for energy "
    assert len(head) == 3100, "dev_gu: v1's text through the label's first words"
    assert prompt.system_message(pal).endswith(TAIL)
    systems = set()
    for e, m in _cells():
        cell_systems = {prompt_v2.system_message(pal, _brief(e, m, bpm, t)) for bpm, t in ((60, 0), (82, 2), (95, 11))}
        assert len(cell_systems) == 1, f"one system message per cell ({e}/{m})"
        system = cell_systems.pop()
        ce, cm = prompt_v2.example_cell(_brief(e, m))
        example = prompt_v2.example_text(pal, _brief(e, m))
        assert system == head + f"{ce} and mood {cm} (it shows the form; write your own loop):\n{example}" + TAIL
        systems.add(system)
    assert len(systems) == 9


def test_every_cell_is_under_the_size_caps(pal):
    sizes = []
    for e, m in _cells():
        text = _text(prompt_v2.build_messages(_brief(e, m), pal))
        assert len(text) <= 4100, (e, m, len(text))      # tests/test_prompt.py's cap at LoopScript's density
        assert prompt_v2.estimate_tokens(text) < 1500, (e, m)
        sizes.append(len(text))
    assert (min(sizes), max(sizes)) == (3750, 3865), "dev_gu, measured 2026-10-31"
    assert prompt_v2.estimate_tokens("abcde") == prompt.estimate_tokens("abcde") == 2


# ------------------------------------------------------------------------------------------- the palette
def _narrow(pal):
    """tests/test_prompt.py's narrowed palette."""
    kit = pal.kits["jazz"]
    return replace(pal, qualities=("maj7", "m7", "m9", "7", "13"), numerals=("I", "II", "IV", "V", "bVII"),
                   keys=MappingProxyType({"rhodes": pal.keys["rhodes"], "wurli": pal.keys["rhodes"]}),
                   kits=MappingProxyType({"jazz": kit, "brushes": kit}),
                   aliases=MappingProxyType({"dusty": "jazz"}), sha256="0" * 64)


def _vocab_lines(system: str) -> list[str]:
    return [line for line in system.splitlines() if line.startswith(VOCAB_HEADS)]


def test_the_vocabulary_and_mood_lines_are_v1s_for_dev_gu_and_a_narrowed_palette(pal):
    narrow = _narrow(pal)
    for palette, buildable in ((pal, set(_cells())),
                               (narrow, {("low", "melancholy"), ("mid", "melancholy"), ("mid", "wistful"),
                                         ("mid", "hopeful")})):
        v1_system = prompt.system_message(palette)
        assert len(_vocab_lines(v1_system)) == len(VOCAB_HEADS)
        built = set()
        for e, m in _cells():
            try:
                system = prompt_v2.system_message(palette, _brief(e, m))
            except prompt_v2.PromptError:
                continue
            built.add((e, m))
            assert _vocab_lines(system) == _vocab_lines(v1_system), (palette.id, e, m)
            assert system.split("Example, for energy ")[0] == v1_system.split("Example, for energy ")[0]
        assert built == buildable, palette.sha256
    assert prompt_v2.system_message(narrow, _brief("mid", "melancholy")) == prompt.system_message(narrow)
    assert prompt_v2.system_message(narrow, _brief("mid", "wistful")) != prompt_v2.system_message(pal, _brief("mid", "wistful"))


def test_a_palette_without_maj9_refuses_only_the_cell_shown_l7(pal):
    no_maj9 = replace(pal, qualities=tuple(q for q in pal.qualities if q != "maj9"), sha256="3" * 64)
    with pytest.raises(prompt_v2.PromptError, match=r"l7_hopeful_low.*dev_gu|dev_gu.*l7_hopeful_low"):
        prompt_v2.build_messages(_brief("low", "wistful"), no_maj9)
    with pytest.raises(prompt_v2.PromptError, match="l7_hopeful_low"):
        prompt_v2.reference_texts(no_maj9)
    for e, m in _cells():
        if (e, m) != ("low", "wistful"):
            assert prompt_v2.build_messages(_brief(e, m), no_maj9)[0]["content"], (e, m)
    refs = prompt_v2.reference_texts(no_maj9, strict=False)
    assert list(refs.refused) == ["l7_hopeful_low"] and "maj9" in refs.refused["l7_hopeful_low"]
    assert len(refs) == 8 and "l7_hopeful_low" not in refs
    assert prompt_v2.reference_texts(pal, strict=False).refused == {}
    assert issubclass(prompt_v2.PromptError, ValueError) and prompt_v2.PromptError is prompt.PromptError


def test_the_examples_take_the_palettes_first_names_when_their_own_are_missing(pal):
    kit = pal.kits["jazz"]
    other = replace(pal, keys=MappingProxyType({"wurli": pal.keys["rhodes"], "piano": pal.keys["rhodes"]}),
                    kits=MappingProxyType({"swirly": kit, "virtuosity": kit}), aliases=MappingProxyType({}),
                    sha256="1" * 64)
    for e, m in _cells():
        b = _brief(e, m)
        ex = prompt_v2.example_text(other, b)
        assert "keys: piano rootless" in ex.splitlines() and "drums: swirly" in ex.splitlines(), (e, m)
        assert parse(ex, vocab=other.vocab(), mode=b.mode).gate.ok
        assert ex in prompt_v2.system_message(other, b)


def test_reference_texts_hold_all_nine_examples_and_contain_every_shown_one(pal):
    refs = prompt_v2.reference_texts(pal)
    assert list(refs) == [ex_id for ex_id, _ in prompt_v2.EXAMPLES.values()] and len(refs) == 9
    assert refs["example"] == prompt.example_text(pal)
    assert refs.refused == {}
    for e, m in _cells():
        shown = prompt_v2.shown_examples(_brief(e, m), pal)
        assert len(shown) == 1
        for ex_id, text in shown:
            assert refs[ex_id] == text == prompt_v2.example_text(pal, _brief(e, m))
            assert ex_id == prompt_v2.example_id(_brief(e, m))


def test_build_messages_is_deterministic_and_returns_fresh_lists(pal):
    b = _brief("high", "hopeful", 90, 7)
    a, c = prompt_v2.build_messages(b, pal), prompt_v2.build_messages(b, pal)
    assert a == c and a is not c and a[0] is not c[0]
    assert [m["role"] for m in a] == ["system", "user"]
    assert a[1]["content"] == "Scene: city at night\nEnergy: high\nMood: hopeful" == prompt_v2.user_message(b)
    a[0]["content"] = "changed"                                      # a caller's edit never reaches the cache
    assert prompt_v2.build_messages(b, pal) == c


def test_the_trainers_message_check_accepts_every_cell(pal):
    from lofi_rl.rl.train import _check_messages
    for e, m in _cells():
        msgs = prompt_v2.build_messages(_brief(e, m), pal)
        assert _check_messages(msgs, f"{e}/{m}") == msgs
