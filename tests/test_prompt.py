"""Prompt template v1 (lofi_rl.prompt): the text that teaches the base model to write one LoopScript from a brief.

What these tests pin:
- determinism: the same brief and palette give the same messages, in this process and in fresh processes with
  other hash seeds (the vocabulary is built from frozensets, so an unsorted walk would change the text);
- the template sha: template_sha256() is pinned, so any edit to the template has to bump PROMPT_VERSION;
- the vocabulary block equals palette.vocab() (numerals, qualities, keys instruments, voicings, kits), for the dev
  palette and for a narrowed copy, so the prompt follows the palette, not a hard-coded list;
- no leak: the messages depend only on the scene, energy and mood (never tempo or tonic), and carry no tempo,
  key or mode word, no forbidden word of the DSL, no note name or absolute chord;
- the in-prompt example is tests/fixtures/loops/example.ls in the palette's own names, without its optional
  `sax:` line, and passes the gates with no repair at all;
- size: under 1,500 tokens by the 4-characters-per-token rule.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from lofi_rl import prompt
from lofi_rl.brief import ENERGIES, MOOD_NAMES, SCENE, Brief, sample_briefs
from lofi_rl.dsl import canon, parse
from lofi_rl.dsl.vocab import FORBIDDEN_WORDS, NUMERAL_SEMITONES, QUALITIES, MODES, MOODS, forbidden_kind
from lofi_rl.palette import load_palette
from lofi_rl.theory.chords import QUALITY

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "palettes" / "dev_gu.yaml"
EXAMPLE_FILE = ROOT / "tests" / "fixtures" / "loops" / "example.ls"
# The template's sha256. Changing the template text changes this: bump PROMPT_VERSION, then re-pin.
V1_SHA256 = "7aceb2bc0d7339a7d3c9af7bf0d808ee156d4f7a8684e0393d923940fa5c1e68"


@pytest.fixture(scope="module")
def pal():
    return load_palette(DEV, verify=False)


def _text(msgs: list[dict]) -> str:
    return "\n".join(m["content"] for m in msgs)


def _briefs() -> list[Brief]:
    return [Brief(energy=e, mood=m, bpm=bpm, tonic=t) for e in ENERGIES for m in MOOD_NAMES
            for bpm, t in ((60, 0), (82, 2), (95, 11))]


# ------------------------------------------------------------------------------------------- shape and determinism
def test_messages_are_one_system_and_one_user_message(pal):
    msgs = prompt.build_messages(Brief(energy="mid", mood="wistful", bpm=82, tonic=2), pal)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert all(set(m) == {"role", "content"} and isinstance(m["content"], str) and m["content"] for m in msgs)
    assert msgs[1]["content"] == "Scene: city at night\nEnergy: mid\nMood: wistful"


def test_the_system_message_is_shared_by_every_brief(pal):
    systems = {prompt.build_messages(b, pal)[0]["content"] for b in _briefs()}
    assert systems == {prompt.system_message(pal)}


def test_build_messages_is_deterministic_and_returns_fresh_lists(pal):
    b = Brief(energy="high", mood="hopeful", bpm=90, tonic=7)
    a, c = prompt.build_messages(b, pal), prompt.build_messages(b, pal)
    assert a == c and a is not c and a[0] is not c[0]
    a[0]["content"] = "changed"                                      # a caller's edit never reaches the cache
    assert prompt.build_messages(b, pal) == c


def test_messages_are_the_same_in_fresh_processes_under_other_hash_seeds(pal):
    code = ("from lofi_rl import prompt; from lofi_rl.brief import Brief; from lofi_rl.palette import load_palette; "
            f"p = load_palette({str(DEV)!r}, verify=False); "
            "print(prompt.messages_sha256(prompt.build_messages(Brief(energy='low', mood='melancholy', bpm=70, tonic=4), p)))")
    shas = set()
    for seed in ("0", "1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT / "src")}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
        assert out.returncode == 0, out.stderr
        shas.add(out.stdout.strip())
    here = prompt.messages_sha256(prompt.build_messages(Brief(energy="low", mood="melancholy", bpm=70, tonic=4), pal))
    assert shas == {here}


def test_messages_sha256_is_the_sha_of_canonical_json(pal):
    msgs = prompt.build_messages(Brief(energy="mid", mood="wistful", bpm=82, tonic=2), pal)
    want = hashlib.sha256(json.dumps(msgs, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()
    assert prompt.messages_sha256(msgs) == want


# ------------------------------------------------------------------------------------------- version and sha
def test_template_sha_is_pinned_to_the_version():
    assert prompt.PROMPT_VERSION == "v1"
    assert prompt.template_sha256() == V1_SHA256, (
        "the template text changed: bump PROMPT_VERSION and re-pin V1_SHA256 (an old run's junk.json names the old sha)")


def test_template_sha_is_the_sha_of_the_unfilled_template_text():
    text = prompt.template_text()
    assert prompt.template_sha256() == hashlib.sha256(text.encode("utf-8")).hexdigest()
    for field in ("{scene}", "{energy}", "{mood}", "{numerals}", "{qualities}", "{keys}", "{kits}", "{voicings}",
                  "{example}", "{kit}", "{melancholy}", "{wistful}", "{hopeful}"):
        assert field in text, f"the unfilled template keeps {field}"
    assert "city at night" not in text, "no brief is filled in"
    assert prompt.EXAMPLE_TEMPLATE in text, "the sha covers the example too"


def test_the_filled_prompt_differs_from_the_template_only_where_fields_are_filled(pal):
    # every literal line of the template survives into the system message, in order
    filled = prompt.system_message(pal)
    for line in prompt.SYSTEM_TEMPLATE.splitlines():
        if "{" not in line:
            assert line in filled


# ------------------------------------------------------------------------------------------- vocabulary
def _vocab_lines(system: str) -> dict[str, set[str]]:
    labels = {"numerals": "numerals", "qualities": "qualities", "keys instruments": "keys",
              "voicings": "voicings", "drum kits": "kits"}
    got = {}
    for line in system.splitlines():
        head, sep, rest = line.partition(": ")
        if sep and head in labels:
            assert labels[head] not in got, f"{head} listed twice"
            got[labels[head]] = set(rest.split())
    return got


def _vocab_sets(pal) -> dict[str, set[str]]:
    v = pal.vocab()
    return {"numerals": set(v.numerals), "qualities": set(v.qualities), "keys": set(v.keys),
            "voicings": set(v.voicings), "kits": set(v.kits)}


def test_the_vocabulary_in_the_prompt_equals_the_palettes(pal):
    assert _vocab_lines(prompt.system_message(pal)) == _vocab_sets(pal)


def test_the_vocabulary_follows_a_narrowed_palette(pal):
    kit = pal.kits["jazz"]
    narrow = replace(pal, qualities=("maj7", "m7", "m9", "7", "13"), numerals=("I", "II", "IV", "V", "bVII"),
                     keys=MappingProxyType({"rhodes": pal.keys["rhodes"], "wurli": pal.keys["rhodes"]}),
                     kits=MappingProxyType({"jazz": kit, "brushes": kit}),
                     aliases=MappingProxyType({"dusty": "jazz"}), sha256="0" * 64)
    system = prompt.system_message(narrow)
    assert _vocab_lines(system) == _vocab_sets(narrow)
    assert system != prompt.system_message(pal)
    by_length = sorted(NUMERAL_SEMITONES, key=len, reverse=True)
    for mood in MOOD_NAMES:      # the mood lists use only the palette's chords
        chords = prompt.mood_chords(narrow)[mood]
        assert chords
        for c in chords:
            numeral = next(n for n in by_length if c.startswith(n))
            assert numeral in narrow.numerals and c[len(numeral):] in narrow.qualities, c


def test_lists_keep_the_grammars_order_and_names_are_sorted(pal):
    vocab = prompt.vocabulary(pal)
    assert vocab["numerals"] == tuple(NUMERAL_SEMITONES)
    assert vocab["qualities"] == QUALITIES
    assert vocab["keys"] == ("organ", "rhodes") and vocab["kits"] == ("jazz",)
    assert vocab["voicings"] == ("rootless", "shell", "spread", "quartal")


def test_sax_names_and_aliases_are_not_taught(pal):
    # sax: is optional and ignored (docs/DSL.md §4), and aliases are leniency for slips, not words to teach
    words = set(re.findall(r"[a-z]+", prompt.system_message(pal).lower()))
    assert not words & set(pal.sax), "no sax name"
    assert not words & set(pal.aliases), "no alias (dusty, wurli, ...)"
    assert not re.search(r"^sax:", prompt.system_message(pal), re.M)


def test_mood_chords_are_exactly_the_chords_whose_notes_lie_in_the_moods_notes(pal):
    got = prompt.mood_chords(pal)
    assert set(got) == set(MOOD_NAMES)
    for mood, chords in got.items():
        notes = set(MODES[MOODS[mood]])
        want = tuple(n + q for n in NUMERAL_SEMITONES for q in QUALITIES
                     if all((NUMERAL_SEMITONES[n] + t) % 12 in notes for t in QUALITY[q].tones))
        assert chords == want
        line = f"{mood}: " + " ".join(want)
        assert line in prompt.system_message(pal).splitlines()
    assert "IV13" in got["wistful"] and "IV13" not in got["melancholy"] and "IV13" not in got["hopeful"]
    assert "bVImaj7" in got["melancholy"] and "Imaj9" in got["hopeful"]


# ------------------------------------------------------------------------------------------- no leaks
def test_the_messages_depend_only_on_scene_energy_and_mood(pal):
    for e in ENERGIES:
        for m in MOOD_NAMES:
            variants = {prompt.messages_sha256(prompt.build_messages(Brief(energy=e, mood=m, bpm=bpm, tonic=t), pal))
                        for bpm in (60, 75, 82, 95) for t in range(12)}
            assert len(variants) == 1, f"tempo or tonic reached the prompt for {e}/{m}"
    nine = {prompt.messages_sha256(prompt.build_messages(Brief(energy=e, mood=m, bpm=82, tonic=2), pal))
            for e in ENERGIES for m in MOOD_NAMES}
    assert len(nine) == 9


def test_no_tempo_key_mode_or_other_forbidden_word(pal):
    banned = FORBIDDEN_WORDS | {"tonic", "pitch", "pitches", "hz", "semitone", "semitones"}
    for b in _briefs():
        words = set(re.findall(r"[a-z]+", _text(prompt.build_messages(b, pal)).lower()))
        assert not words & banned, sorted(words & banned)
        assert str(b.bpm) not in re.findall(r"\d+", _text(prompt.build_messages(b, pal)))


def test_no_note_name_or_absolute_chord(pal):
    # every whitespace token, with punctuation trimmed, is checked by the parser's own G5 patterns; the phrase
    # heads A: and B: keep their colon, which no note name has
    for b in (Brief(energy="mid", mood=m, bpm=82, tonic=2) for m in MOOD_NAMES):
        for raw in _text(prompt.build_messages(b, pal)).split():
            if raw in ("A:", "B:"):
                continue
            token = raw.strip(",.;()")
            if token and token not in ("A:", "B:"):
                assert forbidden_kind(token) != "a note name or absolute chord", repr(raw)


# ------------------------------------------------------------------------------------------- the example
def test_the_example_is_the_fixture_in_the_palettes_names_without_its_sax_line(pal):
    fixture = EXAMPLE_FILE.read_text().splitlines()
    want = [line for line in fixture if not line.startswith("sax:")]
    want = [line.replace("drums: dusty", "drums: jazz") for line in want]       # dev_gu: dusty is an alias of jazz
    assert prompt.example_text(pal).splitlines() == want
    assert prompt.example_text(pal) in prompt.system_message(pal)


def test_the_example_passes_every_gate_with_no_repair_and_is_canonical(pal):
    ex = prompt.example_text(pal)
    res = parse(ex, vocab=pal.vocab(), mode=Brief(energy="mid", mood="wistful", bpm=82, tonic=2).mode)
    assert res.gate.ok and res.repairs == []
    assert canon(res.ast) == ex


def test_the_example_takes_the_palettes_first_names_when_its_own_are_missing(pal):
    kit = pal.kits["jazz"]
    other = replace(pal, keys=MappingProxyType({"wurli": pal.keys["rhodes"], "piano": pal.keys["rhodes"]}),
                    kits=MappingProxyType({"swirly": kit, "virtuosity": kit}), aliases=MappingProxyType({}),
                    sha256="1" * 64)
    ex = prompt.example_text(other)
    assert "keys: piano rootless" in ex.splitlines() and "drums: swirly" in ex.splitlines()
    assert parse(ex, vocab=other.vocab(), mode="dorian").gate.ok


def test_a_palette_that_cannot_play_the_example_is_refused(pal):
    with pytest.raises(prompt.PromptError, match="IV13|13"):
        prompt.system_message(replace(pal, qualities=("m7", "m9", "maj7"), sha256="2" * 64))


# ------------------------------------------------------------------------------------------- size
def test_the_prompt_is_under_1500_tokens_by_the_4_character_rule(pal):
    for b in _briefs():
        n = prompt.estimate_tokens(_text(prompt.build_messages(b, pal)))
        assert n < 1500, n
    assert prompt.estimate_tokens("abcd") == 1 and prompt.estimate_tokens("abcde") == 2


def test_the_prompt_stays_under_1500_real_tokens_at_its_measured_density(pal):
    # The rule above undercounts LoopScript. Measured 2026-10-31 with a byte-level BPE over the cached Qwen3
    # tokenizer.json (it gives the smoke's 269 for example.ls): v1's dev_gu system message is 3,791 characters and
    # 1,358 tokens, 2.79 characters per token. At that density 4,100 characters is about 1,470 tokens.
    for b in _briefs():
        assert len(_text(prompt.build_messages(b, pal))) <= 4100


def test_sample_briefs_feed_build_messages(pal):
    msgs = [prompt.build_messages(b, pal) for b in sample_briefs(2026, 40)]
    assert all(m[1]["content"].startswith(f"Scene: {SCENE}\n") for m in msgs)
