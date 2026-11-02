"""The independent token mask (src/lofi_rl/grammar_mask.py; D-021 Part A: M2 and M3; docs/GRAMMAR.md §3.3).

- Synthetic: small regexes over tiny vocabularies. The allowed sets, stop ids only where the regex accepts,
  termination and the violations after it, ids with no token, memoization by NFA state set. Brute force: for seven
  regexes, every string up to one past the longest is checked against `re.fullmatch` (alive in the NFA exactly when
  it is a prefix of the language, accepting exactly when it matches), and every allowed set against the tokens that
  keep the text a prefix. random_walk: uniform draws from the allowed sets, the same walk for the same seed, every
  walk taken by walk() with no violation and ending on a stop id or at max_len (or where nothing is allowed).
- The real regex (dev_gu's 1412f11f) over a one-character vocabulary, always on the laptop: the in-prompt example
  and its prefixes, a seeded random walk over the regex's own tree and single-character mutations of it all agree
  with `re.fullmatch`; so do the texts of the mask's own random walks, there and over the Qwen vocabulary.
- The dialect lofi_rl.grammar emits, and the refusals outside it.
- xgrammar's view of a tokenizer.json, on synthetic byte-level tokenizers: the vocabulary test, the byte-level
  decode, added and padded ids never allowed, an added token in the regex's alphabet refused, ids past vocab_size
  dropped, shared or shadowed ids, add_prefix_space and other unmirrored settings refused.
- encode_bpe on a synthetic Qwen2-shaped tokenizer: the pre-tokenizer's pieces, merges by rank, the refusals.
- M3: the cached mlx-community/Qwen3-4B-4bit tokenizer.json (pinned by sha256; the test skips, naming it, when it is
  not cached), at GRAMMAR.md §3.3's four points of the in-prompt example as the prototype built them: 20 ids and no
  stop id at the end of A, the same 20 and every stop id at the end of B, 25 after kick's first grid, 8 after the
  last edit; the strings and ids GRAMMAR.md lists.
- M2: every completion of runs/junk-13640325 (arm B) and runs/grpo-smoke-13657200 (both launches), re-encoded with
  encode_bpe and walked with a stop id at the end: all in the language, 0 violations, every walk terminated. runs/
  is pulled, never committed: each test skips, naming the file, where it is not on disk. Also v0's re-encodings
  against vLLM's own token counts (GRAMMAR.md §3.3), which checks encode_bpe itself.
"""
from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from lofi_rl import grammar_mask as gm
from lofi_rl import prompt
from lofi_rl.grammar import loopscript_regex
from lofi_rl.grammar_mask import (BYTE_TO_CHAR, CHAR_TO_BYTE, QWEN2_PRETOKENIZE, GrammarMask, GrammarMaskError, Walk,
                                  encode_bpe, xgrammar_vocab_type)
from lofi_rl.palette import load_palette

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "palettes" / "dev_gu.yaml"
DEV_REGEX_SHA = "1412f11f9ebdc520b50c263eda14be12829951357c17f43496ab2ddf5235e64e"   # docs/GRAMMAR.md §2, D-020
QWEN_TOKENIZER = "models--mlx-community--Qwen3-4B-4bit/snapshots/*/tokenizer.json"
QWEN_TOKENIZER_SHA = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"  # its blob name, 2026-11-02
STOP_IDS = (151643, 151645)             # <|endoftext|>, <|im_end|>: the model's eos set (vLLM's all_stop_token_ids)
EOS = 151645                            # the tokenizer's eos and config.json's eos_token_id
VOCAB_SIZE = 151936                     # Qwen3-4B config.json vocab_size (the lm_head, the bitmask's size)

# M2: what each file holds and what the walk reads (measured 2026-11-02 by these tests; the n_tokens counts are
# D-021's "361 of the 500 arm-B completions, and ... 227 of 320 and 107 of 160 in the smoke's two launches")
M2 = {
    "runs/junk-13640325/summary/completions.jsonl":
        {"rows": 500, "in_language": 500, "tokens": 130347, "violations": 0, "terminated": 500, "n_tokens": 361},
    "runs/grpo-smoke-13657200/summary/a/rewards.jsonl":
        {"rows": 320, "in_language": 320, "tokens": 81359, "violations": 0, "terminated": 320, "n_tokens": 227},
    "runs/grpo-smoke-13657200/summary/b/rewards.jsonl":
        {"rows": 160, "in_language": 160, "tokens": 39233, "violations": 0, "terminated": 160, "n_tokens": 107},
}
M2_RECORDS = {    # where each run records the regex it decoded with
    "runs/junk-13640325/summary/completions.jsonl": ("runs/junk-13640325/summary/generate.json",
                                                     ("constrained", "regex_sha256")),
    "runs/grpo-smoke-13657200/summary/a/rewards.jsonl": ("runs/grpo-smoke-13657200/summary/a/run.json",
                                                         ("structured_outputs", "sha256")),
    "runs/grpo-smoke-13657200/summary/b/rewards.jsonl": ("runs/grpo-smoke-13657200/summary/b/run.json",
                                                         ("structured_outputs", "sha256")),
}
V0 = "runs/junk-13632676/summary/completions.jsonl"
# GRAMMAR.md §3.3, "Tokens allowed": its lists, and the ids it gives
END_OF_A = {"\n", " ", " e", " h", " q", " s", " w", " eb", " er", " hb", " hr", " qb", " qr", " sb", " sr", " wb",
            " wr", " \n", "  ", "  \n"}
AFTER_KICK_LISTED = {"\n", " ", " .", " X", " g", " x", " ................", " \n", "  ", "  \n"}
BEFORE_A = {"\n", " ", " \n", "  ", "  \n", "\n\n", " \n\n", "  \n\n"}
GRAMMAR_IDS = {" \n": 715, "  ": 256, "  \n": 2303, "\n\n": 271, " \n\n": 4710, "  \n\n": 18611}
EOL_TOKENS = {"\n", " \n", "  \n"}


# ------------------------------------------------------------------------------------------- synthetic
def mask(tokens, regex, *, vocab_size=20, stop_ids=(18, 19)):
    return GrammarMask.from_tokens(tokens, regex, vocab_size=vocab_size, stop_ids=stop_ids)


TINY = {0: "a", 1: "b", 2: "ab", 3: "c", 4: "d", 5: "bc", 6: "x", 7: "abc"}


def test_allowed_sets_and_stop_ids_only_where_the_regex_accepts():
    m = mask(TINY, "ab(c|d)?")
    assert m.allowed_after_text("") == (0, 2, 7)                     # a, ab, abc: not accepting, no stop id
    assert m.allowed_after_text("a") == (1, 5)                       # b, bc
    assert m.allowed_after_text("ab") == (3, 4, 18, 19)              # c, d, and every stop id: "ab" matches
    assert m.allowed_after_text("abc") == (18, 19)                   # complete, nothing can follow: stops only
    with pytest.raises(GrammarMaskError, match="leaves the regex's language at character 1"):
        m.allowed_after_text("ac")


def test_a_walk_records_the_set_before_each_token_and_terminates_on_a_stop_id():
    m = mask(TINY, "ab(c|d)?")
    w = m.walk([0, 1, 3, 18])
    assert w == Walk(((0, 2, 7), (1, 5), (3, 4, 18, 19), (18, 19)), None, True)
    assert m.walk([7, 19]) == Walk(((0, 2, 7), (18, 19)), None, True)            # either stop id terminates
    assert m.walk([2]) == Walk(((0, 2, 7),), None, False)                        # no stop yet: not terminated
    assert m.walk([]) == Walk((), None, False)


@pytest.mark.parametrize("ids, violation, terminated, last", [
    ([0, 1, 18, 3], 3, True, ()),               # anything after the stop id: the matcher has terminated
    ([2, 18, 19], 2, True, ()),                 # a second stop id too
    ([0, 18], 1, False, (1, 5)),                # a stop id where the regex does not accept
    ([0, 3], 1, False, (1, 5)),                 # a token the regex refuses
    ([6], 0, False, (0, 2, 7)),                 # a token outside the alphabet
    ([10], 0, False, (0, 2, 7)),                # an id with no token
    ([20], 0, False, (0, 2, 7)),                # an id at vocab_size
    ([-1], 0, False, (0, 2, 7)),
])
def test_a_violation_is_the_first_token_outside_its_set(ids, violation, terminated, last):
    w = mask(TINY, "ab(c|d)?").walk(ids)
    assert (w.violation, w.terminated, w.allowed[-1]) == (violation, terminated, last)
    assert len(w.allowed) == violation + 1, "one set per token walked, up to and including the violating one"


def test_allowed_sets_are_memoized_by_nfa_state_set():
    m = mask({0: "a", 1: "b", 2: "c"}, "(a|b)c")
    assert m.record()["memoized_state_sets"] == 0
    wa, wb = m.walk([0, 2, 18]), m.walk([1, 2, 18])
    assert wa.allowed == wb.allowed == ((0, 1), (2,), (18, 19))
    assert wa.allowed[1] is wb.allowed[1], "after a or b the NFA is in one state set: one memo entry"
    assert m.record()["memoized_state_sets"] == 3
    m.walk([0, 2, 18])
    assert m.record()["memoized_state_sets"] == 3, "a walk over known state sets computes nothing new"


def test_a_regex_that_accepts_the_empty_text_allows_a_stop_id_at_the_start():
    m = mask({0: "a"}, "a?")
    assert m.walk([18]) == Walk(((0, 18, 19),), None, True)
    assert m.allowed_after_text("a") == (18, 19)


def test_ids_are_ints_and_the_vocabulary_is_checked():
    m = mask(TINY, "ab(c|d)?")
    with pytest.raises(GrammarMaskError, match="must be an int"):
        m.walk(["a"])
    with pytest.raises(GrammarMaskError, match="must be an int"):
        m.walk([True])
    for kwargs, what in [({"stop_ids": ()}, "cannot be empty"), ({"stop_ids": (20,)}, "not all in"),
                         ({"stop_ids": (2,)}, "also a text token"), ({"vocab_size": 0}, "must be positive"),
                         ({"vocab_size": 7}, "not all in")]:
        with pytest.raises(GrammarMaskError, match=what):
            mask(TINY, "ab(c|d)?", **{"stop_ids": (18, 19), "vocab_size": 20, **kwargs})
    with pytest.raises(GrammarMaskError, match="not in"):
        mask({25: "a"}, "a")
    with pytest.raises(GrammarMaskError, match="must be a str"):
        mask({0: b"a"}, "a")
    assert mask({0: "", 1: "a"}, "a").record()["empty_tokens"] == 1, "an empty token is never allowed"


def test_a_random_walk_draws_uniformly_from_the_allowed_sets():
    m = mask(TINY, "ab(c|d)?")
    rng = random.Random(3)
    first = Counter(m.random_walk(rng, max_len=1)[0] for _ in range(3000))
    assert set(first) == {0, 2, 7} and all(900 <= k <= 1100 for k in first.values()), first   # a, ab, abc: 1/3 each
    walks = [m.random_walk(random.Random(s)) for s in range(4000)]
    after_ab = Counter(w[1] for w in walks if w[0] == 2)                   # c, d and the two stop ids: 1/4 each
    assert set(after_ab) == {3, 4, 18, 19} and min(after_ab.values()) > 0.8 * max(after_ab.values()), after_ab


WALK_REGEXES = [("ab(c|d)?", "abcd"), ("(a|b){2,3}c?", "abc"), ("[12]{0,2}y?", "12y"), ("(ab|a)(bc|c)?", "abc"),
                ("( {0,2}\\n)?A", " \nA")]


@pytest.mark.parametrize("regex, chars", WALK_REGEXES)
def test_every_random_walk_walks_with_no_violation_and_ends_on_a_stop_id_or_at_max_len(regex, chars):
    vocab = dict(enumerate("".join(p) for n in (1, 2) for p in itertools.product(chars, repeat=n)))
    stops = (len(vocab), len(vocab) + 1)
    m = GrammarMask.from_tokens(vocab, regex, vocab_size=len(vocab) + 2, stop_ids=stops)
    ends = Counter()
    for seed in range(200):
        for max_len in (1, 2, 8):
            ids = m.random_walk(random.Random(seed), max_len=max_len)
            w = m.walk(ids)
            assert w.violation is None and len(w.allowed) == len(ids), (seed, ids)
            assert all(t in a for t, a in zip(ids, w.allowed)), "each id is drawn from the set allowed before it"
            assert w.terminated == (bool(ids) and ids[-1] in stops), (seed, ids)
            assert w.terminated or len(ids) == max_len, "it stops on a stop id or at max_len (no dead end here)"
            assert all(t not in stops for t in ids[:-1]), "nothing after a stop id"
            if w.terminated:
                assert re.fullmatch(regex, "".join(vocab[t] for t in ids[:-1])), (seed, ids)
            ends["stop" if w.terminated else "max_len"] += 1
    assert ends["stop"] and ends["max_len"]


def test_a_random_walk_is_deterministic_for_a_seed_and_checks_max_len():
    m = mask(TINY, "ab(c|d)?")
    walks = [m.random_walk(random.Random(s), max_len=10) for s in range(30)]
    assert walks == [m.random_walk(random.Random(s), max_len=10) for s in range(30)], "the same seed, the same walk"
    assert len(set(map(tuple, walks))) > 3, "different seeds take different walks"
    assert m.random_walk(random.Random(0), max_len=0) == []
    for bad, what in ((-1, "max_len must be >= 0"), (True, "must be an int"), (2.5, "must be an int")):
        with pytest.raises(GrammarMaskError, match=what):
            m.random_walk(random.Random(0), max_len=bad)


def test_a_random_walk_stops_where_nothing_is_allowed():
    m = mask({0: "a"}, "ab")                  # no token writes "b": after "a" nothing is allowed
    assert m.random_walk(random.Random(0), max_len=50) == [0]
    w = m.walk([0])
    assert w.violation is None and not w.terminated


BRUTE = [  # (regex, the characters it writes): every string to one past the longest is checked
    ("ab(c|d)?", "abcd"),
    ("(a|b){2,3}c?", "abc"),
    ("[12]{0,2}y?", "12y"),
    ("(a(b|c)?){1,2}\\|", "abc|"),
    ("a\\.b\\n", "a.b\n"),
    ("(ab|a)(bc|c)?", "abc"),
    ("( {0,2}\\n)?A", " \nA"),
]


@pytest.mark.parametrize("regex, chars", BRUTE)
def test_the_nfa_agrees_with_re_on_every_short_string(regex, chars):
    sigma = chars + "z"                                       # and one character the regex never writes
    every = ["".join(p) for n in range(7) for p in itertools.product(sigma, repeat=n)]
    language = {s for s in every if re.fullmatch(regex, s)}
    longest = max(map(len, language))
    assert longest <= 5, "every string to one past the longest is among `every`"
    strings = [s for s in every if len(s) <= longest + 1]
    prefixes = {s[:k] for s in language for k in range(len(s) + 1)}
    vocab = dict(enumerate("".join(p) for n in (1, 2) for p in itertools.product(sigma, repeat=n)))
    m = GrammarMask.from_tokens(vocab, regex, vocab_size=len(vocab) + 2, stop_ids=(len(vocab), len(vocab) + 1))
    stops = set(m.stop_ids)
    for s in strings:
        if s not in prefixes:
            with pytest.raises(GrammarMaskError):
                m.allowed_after_text(s)
            continue
        allowed = set(m.allowed_after_text(s))
        assert allowed & stops == (stops if s in language else set()), s
        assert allowed - stops == {i for i, t in vocab.items() if s + t in prefixes}, s


# ------------------------------------------------------------------------------------------- the real regex
@pytest.fixture(scope="module")
def pal():
    return load_palette(DEV, verify=False)


@pytest.fixture(scope="module")
def rx(pal):
    regex = loopscript_regex(pal)
    assert hashlib.sha256(regex.encode()).hexdigest() == DEV_REGEX_SHA
    return regex


CHAR_STOP = 500


@pytest.fixture(scope="module")
def char_mask(rx):
    """dev_gu's regex over one token per printable ASCII character and the newline (id = its code point)."""
    tokens = {ord(c): c for c in map(chr, range(32, 127))} | {10: "\n"}
    return GrammarMask.from_tokens(tokens, rx, vocab_size=CHAR_STOP + 1, stop_ids=(CHAR_STOP,))


def nfa_accepts(m: GrammarMask, text: str) -> bool:
    w = m.walk([ord(c) for c in text] + [CHAR_STOP])
    return w.violation is None and w.terminated


def example_layouts(pal) -> dict[str, str]:
    ex = prompt.example_text(pal)
    lines = [line for line in ex.split("\n") if line and not line.startswith("sax:")]
    return {"canonical": ex, "v0": "  \n".join(lines),
            "v0, empty line before A": "  \n".join(lines).replace("  \nA: ", "  \n\nA: ")}


def test_the_real_regex_accepts_the_example_and_stops_only_where_re_matches(pal, rx, char_mask):
    for name, text in example_layouts(pal).items():
        assert re.fullmatch(rx, text) and nfa_accepts(char_mask, text), name
        for k in range(len(text) + 1):                     # every prefix is alive; a stop id exactly where re matches
            allowed = char_mask.allowed_after_text(text[:k])
            assert (CHAR_STOP in allowed) == bool(re.fullmatch(rx, text[:k])), (name, k)


def _sample(node, rng: random.Random) -> str:
    """One string of a grammar_mask._Reader tree: a uniform branch, a short or (sometimes) full repeat."""
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "set":
        return rng.choice(sorted(node[1]))
    if kind == "seq":
        return "".join(_sample(n, rng) for n in node[1])
    if kind == "alt":
        return _sample(rng.choice(node[1]), rng)
    _, sub, lo, hi = node
    n = hi if rng.random() < 0.05 else min(hi, lo + int(rng.expovariate(0.5)))
    return "".join(_sample(sub, rng) for _ in range(n))


def test_the_real_regex_agrees_with_re_on_a_random_walk_and_its_mutations(rx, char_mask):
    rng = random.Random(596)
    tree = gm._Reader(rx).parse()
    sample = [_sample(tree, rng) for _ in range(150)]
    alphabet = sorted(char_mask.alphabet) + ["z", "\t"]
    mutated = []
    for s in sample:
        k = rng.randrange(len(s))
        op = rng.choice(("delete", "insert", "replace"))
        mutated.append(s[:k] + s[k + 1:] if op == "delete" else
                       s[:k] + rng.choice(alphabet) + s[k + (op == "replace"):])
    assert all(re.fullmatch(rx, s) for s in sample)
    for s in sample + mutated:
        assert nfa_accepts(char_mask, s) == bool(re.fullmatch(rx, s)), s
    refused = sum(not re.fullmatch(rx, s) for s in mutated)
    assert refused > 50, f"the mutations should mostly leave the language ({refused} of {len(mutated)} did)"


def check_random_walks(m: GrammarMask, rx: str, n: int, seed: str) -> Counter:
    """n seeded random walks of up to 512 ids over the real regex: each walks with no violation, holds only text
    tokens before its end, and is terminated exactly when it ends on a stop id, whose text then fullmatches."""
    rng, ends = random.Random(seed), Counter()
    for _ in range(n):
        ids = m.random_walk(rng, max_len=512)
        w = m.walk(ids)
        assert w.violation is None and len(w.allowed) == len(ids)
        assert all(m.token_text(t) is not None for t in ids[:-1])
        assert w.terminated == (ids[-1] in m.stop_ids)
        if w.terminated:
            assert re.fullmatch(rx, "".join(m.token_text(t) for t in ids[:-1]))
        else:
            assert len(ids) == 512, "a live state of the real regex always allows a token"
        ends["stop" if w.terminated else "max_len"] += 1
    return ends


def test_random_walks_over_the_real_regex_stay_in_its_language(rx, char_mask):
    ends = check_random_walks(char_mask, rx, 100, "char")
    assert ends["stop"] > 0, ends


# ------------------------------------------------------------------------------------------- the dialect
@pytest.mark.parametrize("bad, what", [
    ("a(?=b)", "(?"), ("a(?!b)", "(?"), ("(?<=a)b", "(?"), ("(?<!a)b", "(?"),        # lookarounds
    ("(a)\\1", "backreference"), ("(?P<x>a)(?P=x)", "(?"), ("(a)\\k<x>", "escape"),   # backreferences
    ("a*?", "unbounded"), ("a+?", "unbounded"), ("a??", "lazy"), ("a{2,3}?", "lazy"), ("a?+", "possessive"),
    ("(?P<x>a)", "(?"), ("(?<x>a)", "(?"),                                            # named groups
    ("(?:a)", "(?"), ("(?i)a", "(?"), ("^a", "anchor"), ("a$", "anchor"), ("a.b", "wildcard"),
    ("[^a]", "negated"), ("[-a]", "inside the class"), ("[a-]", "bad range"), ("[z-a]", "bad range"),
    ("[]", "empty class"), ("[a", "unclosed class"),
    ("\\d", "escape"), ("\\w+", "escape"), ("\\bx", "escape"), ("\\t", "escape"), ("\\-", "escape"),
    ("a\\", "trailing backslash"),
    ("a**", "unbounded"), ("a?{3}", "stacked"), ("a{2}{3}", "stacked"), ("(a", "unclosed"), ("a)", "unbalanced"),
    ("a*", "unbounded"), ("a+", "unbounded"), ("a{2,}", "unbounded"), ("a{3,2}", "repeat {3,2}"),
    ("a{x}", "malformed"), ("a{2", "malformed"), ("{2}", "repeats nothing"), ("a]", "unbalanced"),
    ("a}", "unbalanced"), ("", "empty alternative"), ("a|", "empty alternative"), ("()", "empty alternative"),
    ("(a|)b", "empty alternative"), ("caf\u00e9", "non-ASCII"), ("a\nb", "control character"),
    ("a\tb", "control character"), ("[a\tb]", "control character"),
])
def test_the_dialect_refuses_what_lofi_rl_grammar_never_writes(bad, what):
    with pytest.raises(GrammarMaskError, match=re.escape(what)):
        GrammarMask.from_tokens({0: "a"}, bad, vocab_size=2, stop_ids=(1,))


def test_the_dialect_takes_what_lofi_rl_grammar_writes(rx):
    for regex, text in [("\\n\\.\\|\\(\\)\\[\\]\\{\\}\\?\\*\\+\\^\\$\\\\", "\n.|()[]{}?*+^$\\"),
                        ("[1-3a\\]]{2}", "a]"), ("(ab|c){0,2}d", "ababd"), ("x{0}y", "y"), ("#',-@: ", "#',-@: ")]:
        tokens = {i: c for i, c in enumerate(sorted(set(text)))}
        m = GrammarMask.from_tokens(tokens, regex, vocab_size=len(tokens) + 1, stop_ids=(len(tokens),))
        inverse = {c: i for i, c in tokens.items()}
        assert re.fullmatch(regex, text) and m.walk([inverse[c] for c in text] + [len(tokens)]).terminated, regex
    gm._Reader(rx).parse()                                  # dev_gu's regex is in the dialect


# ------------------------------------------------------------------------------------------- xgrammar's view
def test_the_byte_level_table_is_gpt2s_256_characters():
    assert len(BYTE_TO_CHAR) == 256 and len(set(BYTE_TO_CHAR.values())) == 256
    assert all(CHAR_TO_BYTE[c] == b for b, c in BYTE_TO_CHAR.items())
    assert BYTE_TO_CHAR[ord(" ")] == "\u0120" and BYTE_TO_CHAR[ord("\n")] == "\u010a"  # "Ġ", "Ċ"
    kept = {*range(ord("!"), ord("~") + 1), *range(0xA1, 0xAD), *range(0xAE, 0x100)}
    assert all((BYTE_TO_CHAR[b] == chr(b)) == (b in kept) for b in range(256))
    assert sorted(ord(BYTE_TO_CHAR[b]) for b in range(256) if b not in kept) == list(range(256, 256 + 68))


def byte_level_tokenizer(extra=(), added=(), *, merges=(), prefix_space=False) -> dict:
    """A tokenizer.json shaped like Qwen's: the 256 byte-level characters as ids 0-255, then `extra` (byte-level
    spelled), the Qwen2 pre-tokenizer, and `added` as (id, content) added tokens."""
    vocab = {BYTE_TO_CHAR[b]: b for b in range(256)}
    for token in extra:
        vocab.setdefault(token, len(vocab))
    level = {"type": "ByteLevel", "add_prefix_space": prefix_space, "trim_offsets": False, "use_regex": False}
    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": True} for i, c in added],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN2_PRETOKENIZE}, "behavior": "Isolated", "invert": False},
            dict(level)]},
        "post_processor": dict(level), "decoder": dict(level),
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": [list(m) for m in merges]},
    }


def spell(text: str) -> str:
    """A text's byte-level spelling, as tokenizer.json writes it."""
    return "".join(BYTE_TO_CHAR[b] for b in text.encode("utf-8"))


TIMES = spell("\u00d7")                      # a two-byte token (0xC3 0x97): never allowed by an ASCII regex
SYNTH_ADDED = [(300, "<|end|>"), (301, "<|im|>"), (302, "<think>")]


def write(tmp_path: Path, tj: dict) -> Path:
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(tj), encoding="utf-8")
    return path


def test_from_tokenizer_json_mirrors_xgrammars_vocabulary(tmp_path):
    tj = byte_level_tokenizer([spell(" a"), "ab", spell("  "), TIMES], SYNTH_ADDED)
    path = write(tmp_path, tj)
    m = GrammarMask.from_tokenizer_json(path, "a( a){0,2}b?", vocab_size=320, stop_ids=(300, 301))
    assert [m.token_text(i) for i in (256, 257, 258, 32, 10)] == [" a", "ab", "  ", " ", "\n"]
    assert m.token_text(259) == "\u00c3\u0097", "bytes as latin-1 characters"
    assert m.token_text(302) is None and m.token_text(310) is None
    space, a, b = ord(" "), ord("a"), ord("b")
    assert m.allowed_after_text("") == (a, 257)                                  # a, ab
    assert m.allowed_after_text("a") == (space, a + 1, 256, 300, 301)            # " ", b, " a", stops
    assert m.allowed_after_text("a a a") == (b, 300, 301)
    assert m.allowed_after_text("ab") == (300, 301)
    w = m.walk([a, 256, 256, 98, 300])
    assert w.violation is None and w.terminated
    for never in (302, 303, 319, 259, 260, 299):                     # an added token, padded ids, the 2-byte token
        assert all(never not in s for s in w.allowed)
        assert m.walk([never]).violation == 0
    rec = m.record()
    assert rec["tokenizer"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (rec["tokenizer"]["model_vocab"], rec["tokenizer"]["added_tokens"], rec["tokenizer"]["vocab_type"]) == (
        260, 3, "BYTE_LEVEL")
    assert rec["added_tokens_never_allowed"] == 1 and rec["never_allowed"] == 320 - 260 - 2
    assert rec["tokenizer"]["dropped_at_or_past_vocab_size"] == 0
    json.dumps(rec)


def test_ids_at_or_past_vocab_size_are_dropped_as_xgrammar_drops_them(tmp_path):
    tj = byte_level_tokenizer(["ab"], SYNTH_ADDED)
    tj["model"]["vocab"]["bb"] = 400
    tj["added_tokens"].append({"id": 401, "content": "<late>"})
    m = GrammarMask.from_tokenizer_json(write(tmp_path, tj), "ab?", vocab_size=320, stop_ids=(300,))
    assert m.record()["tokenizer"]["dropped_at_or_past_vocab_size"] == 2
    assert m.token_text(400) is None and m.walk([400]).violation == 0


@pytest.mark.parametrize("change, what", [
    (lambda tj: tj["added_tokens"].append({"id": 303, "content": "bab"}), "spelled in the regex's alphabet"),
    (lambda tj: tj["added_tokens"].append({"id": 303, "content": spell(" b")}), "spelled in the regex's alphabet"),
    (lambda tj: tj["added_tokens"].append({"id": 97, "content": "<y>"}), "share an id"),
    (lambda tj: tj["model"]["vocab"].update(zz=257), "share an id"),
    (lambda tj: tj["added_tokens"].append({"id": 303, "content": "ab"}), "also a model.vocab token with another id"),
    (lambda tj: tj["pre_tokenizer"]["pretokenizers"][1].update(add_prefix_space=True), "add_prefix_space"),
    (lambda tj: tj["decoder"].update(add_prefix_space=True), "add_prefix_space"),
    (lambda tj: tj.update(pre_tokenizer=None, decoder=None), "add_prefix_space is None"),
    (lambda tj: tj["model"]["vocab"].update({"\u2581a": 260}), "not spelled in the byte-level alphabet"),
    (lambda tj: tj["model"].update(vocab={f"<0x{b:02X}>": b for b in range(256)}), "BYTE_FALLBACK"),
    (lambda tj: tj["model"].update(vocab={"hello": 0, "world": 1}), "reads None"),
    (lambda tj: tj["model"].update(type="WordPiece"), "not a BPE"),
    (lambda tj: tj["added_tokens"].append({"id": None, "content": "<z>"}), "must be an int"),
])
def test_from_tokenizer_json_refuses_what_it_does_not_mirror(tmp_path, change, what):
    tj = byte_level_tokenizer([spell(" a"), "ab"], SYNTH_ADDED)
    change(tj)
    with pytest.raises(GrammarMaskError, match=re.escape(what)):
        GrammarMask.from_tokenizer_json(write(tmp_path, tj), "a( a)?b?", vocab_size=320, stop_ids=(300, 301))


def test_an_added_token_that_is_its_own_model_vocab_token_stays_a_text_token(tmp_path):
    tj = byte_level_tokenizer(["ab"], SYNTH_ADDED)
    tj["added_tokens"].append({"id": 256, "content": "ab"})                  # "ab" is model.vocab's 256 too
    m = GrammarMask.from_tokenizer_json(write(tmp_path, tj), "ab?", vocab_size=320, stop_ids=(300,))
    assert m.token_text(256) == "ab" and m.walk([256, 300]) == Walk(((97, 256), (300,)), None, True)


def test_a_stop_id_that_is_a_text_token_is_refused(tmp_path):
    path = write(tmp_path, byte_level_tokenizer(["ab"], SYNTH_ADDED))
    with pytest.raises(GrammarMaskError, match="also a text token"):
        GrammarMask.from_tokenizer_json(path, "ab", vocab_size=320, stop_ids=(ord("a"),))


def test_xgrammars_vocabulary_test():
    level = {BYTE_TO_CHAR[b]: b for b in range(256)}
    assert xgrammar_vocab_type(level | {spell(" a"): 256}) == "BYTE_LEVEL"
    assert xgrammar_vocab_type(level | {f"<{k}>\u2581": 256 + k for k in range(2)}) == "BYTE_LEVEL", "1 % may differ"
    assert xgrammar_vocab_type(level | {f"\u2581{k}": 256 + k for k in range(20)}) is None
    assert xgrammar_vocab_type({f"<0x{b:02X}>": b for b in range(256)}) == "BYTE_FALLBACK"
    assert xgrammar_vocab_type({c: i for i, c in enumerate("abc")}) is None


# ------------------------------------------------------------------------------------------- encode_bpe
# merges, by rank; each pre-tokenizer case below has one that would cross its piece boundary if the split were wrong
BPE_MERGES = [("b", "c"), (spell(" "), "a"), ("a", "b"), (spell(" a"), "b"), (spell(" "), spell(" ")),
              (spell("  "), spell("\n")), ("a", "a"), ("x", "'"), ("'", "s"), ("e", "5"), (spell(" "), "b"),
              (spell(" "), "|"), (".", "5"), (spell("\n"), spell("\n"))]
BPE_EXTRA = ["".join(m) for m in BPE_MERGES]


@pytest.fixture()
def bpe_tj():
    return byte_level_tokenizer(BPE_EXTRA, [(300, "<|end|>")], merges=BPE_MERGES)


@pytest.mark.parametrize("text, tokens", [
    ("ab", ["ab"]),
    (" ab", [" ab"]),                 # a space joins the letters after it: " a" (rank 1), then " ab" (rank 3)
    ("aaa", ["aa", "a"]),             # equal ranks merge leftmost first
    ("abc", ["a", "bc"]),             # the lowest rank merges first, wherever it is
    ("  \n", ["  \n"]),               # spaces before a newline are one piece
    ("a  b", ["a", " ", " b"]),       # a run of spaces leaves its last one to the word after it (not "  ", "b")
    ("e5", ["e", "5"]),               # a digit is a piece of its own (not "e5")
    ("x's", ["x", "'s"]),             # the contractions are pieces (not "x'", "s")
    ("q.5 |", ["q", ".", "5", " |"]),  # punctuation stops at a digit (not ".5"); a space joins the "|" after it
    ("a\n\nb", ["a", "\n\n", "b"]),   # a run of newlines is one piece
])
def test_encode_bpe_pre_tokenizes_and_merges_as_qwen2_does(bpe_tj, text, tokens):
    vocab = bpe_tj["model"]["vocab"]
    assert encode_bpe(bpe_tj, text) == [vocab[spell(t)] for t in tokens]


def test_encode_bpe_refuses_what_it_does_not_reproduce(bpe_tj):
    with pytest.raises(GrammarMaskError, match="ASCII"):
        encode_bpe(bpe_tj, "caf\u00e9")
    with pytest.raises(GrammarMaskError, match="added token"):
        encode_bpe(bpe_tj, "ab<|end|>")
    for change in (lambda tj: tj["pre_tokenizer"]["pretokenizers"][0].update(pattern={"Regex": "\\s+"}),
                   lambda tj: tj["model"].update(byte_fallback=True),
                   lambda tj: tj.update(normalizer={"type": "Lowercase"}),
                   lambda tj: tj["model"].update(merges=[["a", "b"], ["a", "b"]])):
        tj = json.loads(json.dumps(bpe_tj))
        change(tj)
        with pytest.raises(GrammarMaskError):
            encode_bpe(tj, "ab")


def test_the_ascii_pre_tokenizer_is_the_qwen2_pattern_read_on_ascii():
    """transformers' models/qwen2/tokenization_qwen2.py:33 and the tokenizer.json spell the pattern; on ASCII text
    \\p{L} is [A-Za-z], \\p{N} is [0-9], and re.ASCII's \\s is Unicode White_Space (U+0009-U+000D, U+0020)."""
    assert QWEN2_PRETOKENIZE == ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"
                                 "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
    ascii_form = (QWEN2_PRETOKENIZE.replace("[^\\r\\n\\p{L}\\p{N}]", "[^\\r\\nA-Za-z0-9]")
                  .replace("\\p{L}+", "[A-Za-z]+").replace("|\\p{N}|", "|[0-9]|")
                  .replace("[^\\s\\p{L}\\p{N}]", "[^\\sA-Za-z0-9]"))
    assert gm._QWEN2_ASCII.pattern == ascii_form and gm._QWEN2_ASCII.flags & re.ASCII
    assert [c for c in map(chr, range(128)) if re.fullmatch("\\s", c, re.ASCII)] == list("\t\n\x0b\x0c\r ")
    assert re.fullmatch("\\s", "\x1c") and not re.fullmatch("\\s", "\x1c", re.ASCII), "why the module uses re.ASCII"


# ------------------------------------------------------------------------------------------- M3 and M2
def _hub() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


@pytest.fixture(scope="module")
def qwen_tokenizer() -> Path:
    found = sorted(glob.glob(str(_hub() / QWEN_TOKENIZER)))
    if not found:
        pytest.skip(f"{_hub() / QWEN_TOKENIZER} is not on this machine (the cached mlx-community/Qwen3-4B-4bit "
                    f"snapshot GRAMMAR.md §3.3 used)")
    shas = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in found}
    pinned = [p for p, sha in shas.items() if sha == QWEN_TOKENIZER_SHA]
    assert pinned, f"no cached tokenizer.json has sha256 {QWEN_TOKENIZER_SHA[:8]}: {shas}"
    return Path(pinned[0])


@pytest.fixture(scope="module")
def qwen_mask(qwen_tokenizer, rx) -> GrammarMask:
    return GrammarMask.from_tokenizer_json(qwen_tokenizer, rx, vocab_size=VOCAB_SIZE, stop_ids=STOP_IDS)


@pytest.fixture(scope="module")
def qwen_json(qwen_tokenizer) -> dict:
    return json.loads(qwen_tokenizer.read_text(encoding="utf-8"))


def m3_points(pal) -> dict[str, str]:
    """GRAMMAR.md §3.3's four points of the in-prompt example, built as the D-021 prototype (tokmask_points.py) built
    them: v0's layout (two spaces before every newline, no final newline)."""
    text = example_layouts(pal)["v0"]
    i_a, i_b = text.index("\nA: "), text.index("\nB: ")
    kick_end = text.index("kick: ") + len("kick: ") + 16
    return {"end of A": text[:i_b].rstrip(), "end of B": text, "after kick's one grid": text[:kick_end],
            "after the last edit, before A": text[:len(text[:i_a].rstrip(" "))]}


def test_m3_the_allowed_sets_at_grammar_md_3_3s_four_points(pal, qwen_mask):
    points = m3_points(pal)
    sets = {name: qwen_mask.allowed_after_text(text) for name, text in points.items()}
    stops = set(STOP_IDS)
    texts = {name: {qwen_mask.token_text(i) for i in ids if i not in stops} for name, ids in sets.items()}
    sizes = {name: (len([i for i in ids if i not in stops]), sorted(set(ids) & stops)) for name, ids in sets.items()}
    assert sizes == {"end of A": (20, []), "end of B": (20, list(STOP_IDS)), "after kick's one grid": (25, []),
                     "after the last edit, before A": (8, [])}
    assert texts["end of A"] == END_OF_A
    assert [i for i in sets["end of B"] if i not in stops] == list(sets["end of A"]), "the same 20 at the end of B"
    assert texts["after the last edit, before A"] == BEFORE_A
    kick = texts["after kick's one grid"]
    assert AFTER_KICK_LISTED <= kick
    grids = kick - {" ", "  "} - EOL_TOKENS
    assert len(grids) == 20 and all(re.fullmatch("( )[Xxg.]{1,16}", t) for t in grids), "20 start a second grid"
    by_text = {qwen_mask.token_text(i): i for ids in sets.values() for i in ids if i not in stops}
    assert {t: by_text[t] for t in GRAMMAR_IDS} == GRAMMAR_IDS


def test_m3_the_mask_is_built_from_the_qwen_vocabulary_as_xgrammar_sees_it(qwen_mask):
    rec = qwen_mask.record()
    assert rec["regex_sha256"] == DEV_REGEX_SHA and rec["vocab_size"] == VOCAB_SIZE
    assert rec["stop_ids"] == list(STOP_IDS) and rec["tokenizer"]["sha256"] == QWEN_TOKENIZER_SHA
    t = rec["tokenizer"]
    assert (t["model_vocab"], t["added_tokens"], t["vocab_type"], t["dropped_at_or_past_vocab_size"]) == (
        151643, 26, "BYTE_LEVEL", 0)
    assert t["stop_ids_not_added_tokens"] == [], "both stop ids are added tokens (<|endoftext|>, <|im_end|>)"
    assert rec["added_tokens_never_allowed"] == 24, "the 26 added tokens but the two stop ids: each holds '<'"
    assert rec["never_allowed"] == VOCAB_SIZE - 151643 - 2, "24 added tokens and 151669-151935 (no token)"
    assert rec["text_tokens"] == 151643 and 0 < rec["tokens_in_alphabet"] < rec["text_tokens"]
    assert "|" in rec["alphabet"] and "<" not in rec["alphabet"]
    json.dumps(rec)


def test_random_walks_over_the_qwen_vocabulary_stay_in_the_language(rx, qwen_mask):
    """mask_preflight's NFA-driven draw (seed "0:nfa", 300 walks of up to 512 ids), over the laptop's cached
    tokenizer rather than the cluster snapshot's: every walk walks, and every one ended on a stop id (measured here,
    2026-11-02: 34,002 ids in all)."""
    assert check_random_walks(qwen_mask, rx, 300, "0:nfa") == Counter(stop=300)


def _runs_file(rel: str) -> Path:
    """runs/ is pulled, never committed (CLAUDE.md §7): this checkout's, else, from an agent worktree
    (<checkout>/.claude/worktrees/<name>), the main checkout's."""
    cands = [ROOT / rel]
    if ROOT.parent.name == "worktrees" and ROOT.parent.parent.name == ".claude":
        cands.append(ROOT.parents[2] / rel)
    for path in cands:
        if path.is_file():
            return path
    pytest.skip(f"{rel} is not on this machine (make pull brings it to the laptop)")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("rel", sorted(M2))
def test_m2_every_re_encoded_completion_walks_with_no_violation(rel, rx, qwen_mask, qwen_json):
    record_rel, keys = M2_RECORDS[rel]
    record = json.loads(_runs_file(record_rel).read_text(encoding="utf-8"))
    assert record[keys[0]][keys[1]] == DEV_REGEX_SHA, "the run decoded with dev_gu's regex"
    full = re.compile(rx).fullmatch
    counts = Counter()
    for row in _jsonl(_runs_file(rel)):
        text = row["text"]                                   # both files keep the completion's text in "text"
        if "text_sha256" in row:
            assert hashlib.sha256(text.encode("utf-8")).hexdigest() == row["text_sha256"]
        counts["rows"] += 1
        if not full(text):
            continue
        counts["in_language"] += 1
        ids = encode_bpe(qwen_json, text)
        counts["n_tokens"] += row["n_tokens"] == len(ids) + 1        # vLLM's count holds the EOS
        walk = qwen_mask.walk(ids + [EOS])
        counts["tokens"] += len(ids) + 1
        counts["violations"] += walk.violation is not None
        counts["terminated"] += walk.terminated
    assert dict(counts) == M2[rel]


@pytest.mark.parametrize("rel", sorted(M2))
def test_m2_stop_ids_are_allowed_exactly_where_the_text_so_far_matches(rel, rx, qwen_mask, qwen_json):
    """The first 16 completions of each file: at every token boundary, the stop ids are in the walk's set exactly
    when `re.fullmatch` takes the text decoded so far (the stop position included)."""
    full = re.compile(rx).fullmatch
    for row in _jsonl(_runs_file(rel))[:16]:
        ids = encode_bpe(qwen_json, row["text"])
        walk = qwen_mask.walk(ids + [EOS])
        text = ""
        for i, allowed in enumerate(walk.allowed):
            assert (set(STOP_IDS) <= set(allowed)) == bool(full(text)), (row.get("i"), i)
            assert not set(STOP_IDS) & set(allowed) or set(STOP_IDS) <= set(allowed)
            if i < len(ids):
                text += qwen_mask.token_text(ids[i])


def test_encode_bpe_gives_vllms_own_token_count_for_v0_as_grammar_md_says(qwen_json):
    """GRAMMAR.md §3.3: re-encoding v0's texts gives vLLM's own token count for 496 of the 500 (EOS aside), and the
    8,062 two-space line ends encode as 8,025 tokens 2303 plus 37 tokens 18611."""
    rows = _jsonl(_runs_file(V0))
    assert len(rows) == 500
    same, ids_all = 0, Counter()
    for row in rows:
        ids = encode_bpe(qwen_json, row["text"])
        same += len(ids) + (row["finish_reason"] == "stop") == row["n_tokens"]
        ids_all.update(ids)
    assert (same, ids_all[2303], ids_all[18611]) == (496, 8025, 37)
