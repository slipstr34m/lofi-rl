"""The independent check of xgrammar's token mask for the LoopScript regex (D-021 Part A; docs/GRAMMAR.md §3.3).

    from lofi_rl.grammar_mask import GrammarMask, encode_bpe
    mask = GrammarMask.from_tokenizer_json(snapshot / "tokenizer.json", regex, vocab_size=151936,
                                           stop_ids=(151643, 151645))
    w = mask.walk(completion_ids)      # w.allowed[i]: the ids allowed before token i; w.violation; w.terminated
    mask.random_walk(rng, max_len=512) # ids drawn uniformly from the allowed sets (mask_preflight's random walks)
    mask.allowed_after_text(prefix)    # the ids allowed right after a text (GRAMMAR.md §3.3's points: D-021's M3)
    mask.record()                      # what run.json keeps of it
    encode_bpe(tokenizer_json, text)   # the laptop's canonical re-encoding of a completion (D-021's M2)

**What it is.** A Thompson NFA of the regex, over the characters of the byte-level BPE vocabulary. The set allowed at
a point is every token whose text keeps the NFA alive from there, plus every stop id when the NFA is in its accepting
state; sets are memoized by NFA state set, so a step of 32 completions computes few. The trainer
(`lofi_rl.rl.masked`) replays xgrammar's own matcher along each completion; this is the second opinion D-021 names.
Built from the same tokenizer.json, vocabulary size and stop ids, it must give xgrammar's set at every position (M8's
`nfa_mismatch`); on the laptop it can only estimate that set (M2, M3, tests/test_grammar_mask.py). It is built
independently of `lofi_rl.grammar` and of tests/test_grammar.py's reader: it reads the regex text alone.

**The dialect** is the one `lofi_rl.grammar` emits, and only that (docs/GRAMMAR.md §5): ASCII literals, the escapes
`\\n` and backslash + metacharacter, classes of characters and ranges (`[1-7]`), plain groups, alternation and the
bounded greedy quantifiers `?`, `{n}` and `{n,m}`. Anything else raises GrammarMaskError naming it: a non-ASCII or
control character, `*`, `+` and `{n,}`, lazy, possessive and stacked quantifiers, `(?`, backreferences, anchors, the
wildcard, negated classes, class escapes, bare `]` or `}`, empty alternatives and empty classes. The language is
what is checked, so greedy or lazy does not matter; a regex outside the dialect is refused rather than guessed at.

**How xgrammar 0.2.8 sees the vocabulary, and what this mirrors.** Read on 2026-11-02 in the policy env's installed
Python sources (xgrammar 0.2.8, vLLM 0.32.0, transformers 5.19; read-only copies, never imported here). The token
decoder, the special-token rule and the mask itself are in xgrammar's C++ core, which those sources do not hold:
what this module infers about them is marked as such, and M8 on the cluster compares the two at every position.
1. vLLM builds it with `xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.vocab_size)`
   (vllm/v1/structured_output/backend_xgrammar.py:69-72), `vocab_size` being `model_config.get_vocab_size()`
   (structured_output/__init__.py:144), the model's lm_head size (Qwen3-4B's config.json: 151936). The bitmask is
   allocated over that size (backend_xgrammar.py:166-167).
2. `from_huggingface` takes `vocab_dict = tokenizer.get_vocab()` (xgrammar/tokenizer_info.py:293-300). For the
   Qwen tokenizer, a `TokenizersBackend` (= `PreTrainedTokenizerFast`, transformers
   tokenization_utils_tokenizers.py:1514; models/qwen2/tokenization_qwen2.py:36), that is `model.vocab` plus every
   added token, by content (tokenization_utils_tokenizers.py:709-710, `get_vocab(with_added_tokens=True)`; vLLM's
   cached tokenizer returns the same dict, vllm/tokenizers/hf.py:115,154-155).
3. `encoded_vocab = [""] * vocab_size`, filled by id for the ids below vocab_size (tokenizer_info.py:302-313): ids
   with no token stay "" (the cached Qwen3-4B tokenizer.json holds 151643 + 26 added tokens, so 151669-151935;
   tests/test_grammar_mask.py), ids at or past vocab_size are dropped. from_tokenizer_json drops them too, and
   records how many.
4. The vocabulary type: the fast-tokenizer branch (tokenizer_info.py:315-360) reads it from the serialized backend
   (C++, :334) unless the vocabulary's own test disagrees, which then wins (:342-352). That test
   (`_detect_vocab_type_from_vocab`, :171-198) is pure Python, and this module runs it on the same dict: byte-level
   when no 128 `<0xNN>` pieces exist, at least 128 of the 256 byte-level characters are one-character tokens and at
   least 99 % of the tokens are spelled in them. Anything but BYTE_LEVEL is refused, so the type is the one xgrammar
   uses whatever the backend string says. The 256-character alphabet is GPT-2's byte-to-unicode table
   (`_build_byte_level_charset`, :54-69; VocabType.BYTE_LEVEL, :43-51): bytes `!`-`~`, 0xA1-0xAC and 0xAE-0xFF as
   themselves, the other 68 shifted to chr(256 + n) in byte order. TokenizerInfo "always decodes the vocabulary to
   the original format" (:79-82), so a token's text is its bytes. Here: every `model.vocab` token is decoded through
   that table (one not spelled in it is refused) and walked byte by byte, each byte one character (latin-1). The
   regex is ASCII, so a token with any byte at 0x80 or above is never allowed, as it cannot be under xgrammar's
   byte matching. add_prefix_space (:200-215, :359) must be false in the tokenizer's ByteLevel parts, or refused.
5. Stop ids. vLLM builds TokenizerInfo without stop ids, so it takes `[tokenizer.eos_token_id]` (:325-327), and then
   gives each request's matcher `override_stop_tokens=list(all_stop_token_ids)` (backend_xgrammar.py:157-161;
   structured_output/__init__.py:199-203): the tokenizer's eos plus the generation config's (sampling_params.py:
   772-796), the set lofi_rl.rl.train's `_eos_ids` builds for the mask. The matcher's stop tokens are those
   (xgrammar/matcher.py:520-530). After the whole grammar matches it accepts a stop token, and after accepting one
   it terminates and "cannot accept any new token or generate a new token mask" (matcher.py:216-219); accept_token
   refuses a token after termination, an id out of range and a special token (:281-293); vLLM ignores tokens after
   termination (backend_xgrammar.py:190-212). Here: a stop id is allowed exactly where the NFA accepts, accepting
   one terminates the walk, and any token after it is a violation (its allowed set is empty). That a stop id's bit
   is set exactly where the grammar can end, and never by its text, is inferred (the C++ is not in the sources).
6. Special and added tokens. Special tokens are "control tokens, reserved tokens, padded tokens, etc.", detected
   from the vocabulary in C++ (tokenizer_info.py:469-472), and accept_token refuses them (matcher.py:290). Inferred:
   the padded ids, the "" of item 3, are special, and added tokens are not: xgrammar matches an added token by its
   text, which is why its structural tags must exclude `<think>` and `</think>` "from appearing ... as plain text"
   (builtin_structural_tag.py:202-207, 470-482). Their contents reach the core through get_vocab (item 2), and
   Qwen's are printable ASCII, which the byte-level table maps to itself. Here: an added token that is not a stop
   id is never allowed, and an id with no token is never allowed (D-021's rule). That equals xgrammar's text
   treatment only while no added token can be spelled in the regex's alphabet, so the build checks it: an added
   token whose text (decoded through the table, or as written) uses only the regex's characters raises
   GrammarMaskError. Qwen's all hold `<`, which the LoopScript regex never writes. Only tokenizer.json is read:
   a token transformers added from elsewhere would be unknown here, so never allowed (M8 would show it).

**encode_bpe** is the Qwen2 tokenizer's encoding, for ASCII text: the Split pre-tokenizer
(models/qwen2/tokenization_qwen2.py:33, the same pattern as the tokenizer.json, which is checked; \\p{L} and \\p{N}
are [A-Za-z] and [0-9] on ASCII, \\s is Unicode White_Space, = Python's under re.ASCII), "Isolated" pieces, the
byte-level map, then BPE merges by rank, lowest first and leftmost among equals (byte_fallback false, ignore_merges
false). NFC is the identity on ASCII. Text holding an added token's content (which the real tokenizer would split
out first) or a non-ASCII character is refused. It is canonical tokenization: vLLM samples ids, not this, so M2
checks the NFA on re-encodings and M8 checks the sampled ids on the cluster.

Standard library only: it imports on the laptop, in the policy env and in the scorer env.
"""
from __future__ import annotations

import hashlib
import json
import operator
import random
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

__all__ = ["GrammarMaskError", "Walk", "GrammarMask", "encode_bpe", "BYTE_TO_CHAR", "CHAR_TO_BYTE",
           "QWEN2_PRETOKENIZE", "xgrammar_vocab_type"]

META = frozenset("\\.^$*+?()[]{}|")     # what a backslash may escape; with \n, the only escapes in the dialect
ESCAPES: Mapping[str, str] = MappingProxyType({"n": "\n", **{c: c for c in META}})
MAX_NFA_STATES = 1_000_000              # a bound on a bounded repeat's copies (dev_gu's regex needs far fewer)
QWEN2_PRETOKENIZE = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                     r"|\s*[\r\n]+|\s+(?!\S)|\s+")
# The same pattern on ASCII: \p{L} is [A-Za-z] and \p{N} is [0-9] there, and re.ASCII's \s is [ \t\n\r\f\v], which
# is Unicode White_Space on ASCII (Python's default \s would also take \x1c-\x1f, which White_Space does not)
_QWEN2_ASCII = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+[\r\n]*"
                          r"|\s*[\r\n]+|\s+(?!\S)|\s+", re.ASCII)


class GrammarMaskError(ValueError):
    """A regex outside the dialect, a vocabulary or tokenizer this module does not mirror, or a text it cannot
    decide. It never means a token was refused: walk() reports that."""


@dataclass(frozen=True)
class Walk:
    """One completion's ids walked through the mask.

    allowed[i] is the sorted ids allowed before token i: one entry per token walked, up to and including the first
    violating token (an empty set for a token after a stop id). violation is that token's index, or None.
    terminated says the walk accepted a stop id (so it had to end there)."""
    allowed: tuple[tuple[int, ...], ...]
    violation: int | None
    terminated: bool


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte-to-unicode table (xgrammar/tokenizer_info.py:54-69 spells the same 256 characters)."""
    kept = [*range(ord("!"), ord("~") + 1), *range(0xA1, 0xAC + 1), *range(0xAE, 0xFF + 1)]
    table = {b: chr(b) for b in kept}
    shifted = 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + shifted)
            shifted += 1
    return dict(sorted(table.items()))


BYTE_TO_CHAR: Mapping[int, str] = MappingProxyType(_bytes_to_unicode())
CHAR_TO_BYTE: Mapping[str, int] = MappingProxyType({c: b for b, c in BYTE_TO_CHAR.items()})


def xgrammar_vocab_type(vocab: Mapping[str, int]) -> str | None:
    """xgrammar 0.2.8's own vocabulary test (tokenizer_info.py:171-198, `_detect_vocab_type_from_vocab`), on the
    dict get_vocab() returns: "BYTE_FALLBACK", "BYTE_LEVEL", or None when it matches neither."""
    if sum(f"<0x{b:02X}>" in vocab for b in range(256)) >= 128:
        return "BYTE_FALLBACK"
    if sum(c in vocab for c in CHAR_TO_BYTE) < 128:
        return None
    spelled = sum(all(c in CHAR_TO_BYTE for c in token) for token in vocab)
    return "BYTE_LEVEL" if spelled >= 0.99 * len(vocab) else None


def _as_id(value: Any, what: str = "a token id") -> int:
    if isinstance(value, bool):
        raise GrammarMaskError(f"{what} must be an int, not {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise GrammarMaskError(f"{what} must be an int, not {value!r}") from None


# ------------------------------------------------------------------------------------------- the dialect
class _Reader:
    """Parses a regex of the dialect into a tree: ("lit", char) | ("set", chars) | ("seq", nodes) | ("alt", nodes)
    | ("rep", node, lo, hi). Everything outside the dialect raises GrammarMaskError, naming the construct."""

    def __init__(self, regex: str):
        self.rx, self.i = regex, 0

    def parse(self) -> tuple:
        node = self._alt()
        if self.i != len(self.rx):
            raise GrammarMaskError(f"unbalanced ')' at {self.i}")
        return node

    def _peek(self, k: int = 0) -> str:
        j = self.i + k
        return self.rx[j] if j < len(self.rx) else ""

    def _alt(self) -> tuple:
        at = self.i
        branches = [self._seq()]
        while self._peek() == "|":
            self.i += 1
            branches.append(self._seq())
        if any(not b[1] for b in branches):
            raise GrammarMaskError(f"an empty alternative or group at {at}")
        return branches[0] if len(branches) == 1 else ("alt", tuple(branches))

    def _seq(self) -> tuple:
        items = []
        while self._peek() not in ("", "|", ")"):
            items.append(self._quantified(self._atom()))
        return ("seq", tuple(items))

    def _atom(self) -> tuple:
        c, at = self._peek(), self.i
        if c == "(":
            if self._peek(1) == "?":
                raise GrammarMaskError(f"'(?' at {at}: a lookaround, a named or non-capturing group, or an inline "
                                       f"flag")
            self.i += 1
            node = self._alt()
            if self._peek() != ")":
                raise GrammarMaskError(f"unclosed '(' at {at}")
            self.i += 1
            return node
        if c == "[":
            return self._set()
        if c == "\\":
            return ("lit", self._escape())
        if c in ("^", "$"):             # tuples, not strings: "" is in every string
            raise GrammarMaskError(f"anchor {c!r} at {at}")
        if c == ".":
            raise GrammarMaskError(f"wildcard '.' at {at}")
        if c in ("*", "+", "?", "{"):
            raise GrammarMaskError(f"quantifier {c!r} at {at} repeats nothing")
        if c in ("]", "}"):
            raise GrammarMaskError(f"unbalanced {c!r} at {at}")
        self._literal(c, at)
        self.i += 1
        return ("lit", c)

    @staticmethod
    def _literal(c: str, at: int) -> None:
        if ord(c) < 0x20 or ord(c) == 0x7F:
            raise GrammarMaskError(f"control character {c!r} at {at}: the dialect writes a newline as the escape \\n")

    def _escape(self) -> str:
        e, at = self._peek(1), self.i
        if not e:
            raise GrammarMaskError(f"a trailing backslash at {at}")
        if e.isdigit():
            raise GrammarMaskError(f"backreference \\{e} at {at}")
        if e not in ESCAPES:
            raise GrammarMaskError(f"escape \\{e} at {at}: only \\n and a backslash before a metacharacter")
        self.i += 2
        return ESCAPES[e]

    def _set(self) -> tuple:
        at = self.i
        self.i += 1
        if self._peek() == "^":
            raise GrammarMaskError(f"negated class at {at}")
        chars: set[str] = set()
        while self._peek() != "]":
            c = self._peek()
            if c in ("", "[", "-"):
                raise GrammarMaskError(f"unclosed class at {at}" if not c else f"{c!r} inside the class at {at}")
            if c == "\\":
                lo = self._escape()
            else:
                self._literal(c, self.i)
                lo = c
                self.i += 1
            if self._peek() == "-":
                hi = self._peek(1)
                if hi in ("", "]", "\\", "[") or hi < lo:
                    raise GrammarMaskError(f"bad range in the class at {at}")
                self._literal(hi, self.i + 1)
                self.i += 2
                chars.update(chr(k) for k in range(ord(lo), ord(hi) + 1))
            else:
                chars.add(lo)
        self.i += 1
        if not chars:
            raise GrammarMaskError(f"empty class at {at}")
        return ("set", frozenset(chars))

    def _quantified(self, node: tuple) -> tuple:
        c, at = self._peek(), self.i
        if c == "?":
            self.i += 1
            lo, hi = 0, 1
        elif c in ("*", "+"):
            raise GrammarMaskError(f"unbounded repeat {c!r} at {at}")
        elif c == "{":
            end = self.rx.find("}", self.i)
            m = re.fullmatch(r"([0-9]+)(,([0-9]*))?", self.rx[self.i + 1:end]) if end > 0 else None
            if m is None:
                raise GrammarMaskError(f"malformed repeat at {at}")
            if m[2] is not None and not m[3]:
                raise GrammarMaskError(f"unbounded repeat {{{m[1]},}} at {at}")
            lo = int(m[1])
            hi = lo if m[2] is None else int(m[3])
            if hi < lo:
                raise GrammarMaskError(f"repeat {{{lo},{hi}}} at {at}")
            self.i = end + 1
        else:
            return node
        if self._peek() in ("?", "+"):
            raise GrammarMaskError(f"lazy or possessive quantifier at {self.i}")
        if self._peek() in ("*", "{"):
            raise GrammarMaskError(f"stacked quantifier at {self.i}")
        return ("rep", node, lo, hi)


# ------------------------------------------------------------------------------------------- the NFA
class _NFA:
    """Thompson's construction. A state either reads one character class (label, to) or has only epsilon moves.
    A bounded repeat {lo,hi} is lo copies, then hi - lo optional copies that can each skip to the end: linear size."""

    def __init__(self, tree: tuple):
        self.label: list[frozenset[str] | None] = []
        self.to: list[int] = []
        self.eps: list[list[int]] = []
        self.start, self.accept = self._build(tree)

    def _new(self) -> int:
        if len(self.label) >= MAX_NFA_STATES:
            raise GrammarMaskError(f"the regex needs more than {MAX_NFA_STATES} NFA states")
        self.label.append(None)
        self.to.append(-1)
        self.eps.append([])
        return len(self.label) - 1

    def _build(self, node: tuple) -> tuple[int, int]:
        kind = node[0]
        if kind in ("lit", "set"):
            a, b = self._new(), self._new()
            self.label[a] = frozenset((node[1],)) if kind == "lit" else node[1]
            self.to[a] = b
            return a, b
        if kind == "seq":
            a = cur = self._new()
            for sub in node[1]:
                x, y = self._build(sub)
                self.eps[cur].append(x)
                cur = y
            return a, cur
        if kind == "alt":
            a, b = self._new(), self._new()
            for sub in node[1]:
                x, y = self._build(sub)
                self.eps[a].append(x)
                self.eps[y].append(b)
            return a, b
        _, sub, lo, hi = node
        a = cur = self._new()
        for _ in range(lo):
            x, y = self._build(sub)
            self.eps[cur].append(x)
            cur = y
        end = self._new()
        for _ in range(hi - lo):
            x, y = self._build(sub)
            self.eps[cur].extend((x, end))
            cur = y
        self.eps[cur].append(end)
        return a, end


# ------------------------------------------------------------------------------------------- the mask
class GrammarMask:
    """The token mask of one regex over one vocabulary. Build it with from_tokenizer_json (the cluster's snapshot,
    the laptop's cached Qwen3 tokenizer) or from_tokens (any id -> text map; the tests' tiny vocabularies).

    A text token is allowed where its whole text keeps the NFA alive; a stop id where the NFA accepts. Every other id
    below vocab_size (no token, an added token, a special one) is never allowed. State sets keep only the states
    that read a character, plus the accepting state, so two prefixes that can continue alike share one memo entry."""

    def __init__(self, *, tokens: Mapping[int, str], regex: str, vocab_size: int, stop_ids: Iterable[int],
                 untexted: Mapping[int, tuple[str, ...]] | None = None, source: Mapping[str, Any] | None = None,
                 started: float | None = None):
        """Internal: use from_tokenizer_json or from_tokens. `untexted` holds, per added token, the texts xgrammar
        could match it by (item 6 of the module docstring); `source` is the tokenizer file's record."""
        t0 = time.perf_counter() if started is None else started
        if not isinstance(regex, str):
            raise GrammarMaskError(f"the regex must be a str, not {type(regex).__name__}")
        if not regex.isascii():
            bad = next(c for c in regex if not c.isascii())
            raise GrammarMaskError(f"non-ASCII character {bad!r} at {regex.index(bad)}: the dialect is ASCII")
        self.vocab_size = _as_id(vocab_size, "vocab_size")
        if self.vocab_size <= 0:
            raise GrammarMaskError(f"vocab_size must be positive, not {self.vocab_size}")
        stops = sorted({_as_id(s, "a stop id") for s in stop_ids})
        if not stops:
            raise GrammarMaskError("stop_ids cannot be empty (xgrammar refuses an empty list too, "
                                   "tokenizer_info.py:290-291)")
        if stops[0] < 0 or stops[-1] >= self.vocab_size:
            raise GrammarMaskError(f"stop ids {stops} are not all in [0, {self.vocab_size})")
        self.stop_ids: tuple[int, ...] = tuple(stops)
        self._stop_set = frozenset(stops)
        self.regex = regex
        self.regex_sha256 = hashlib.sha256(regex.encode("utf-8")).hexdigest()

        self._nfa = _NFA(_Reader(regex).parse())
        self.alphabet = frozenset().union(*(lab for lab in self._nfa.label if lab is not None))

        self._text: dict[int, str] = {}
        empty = 0
        for raw_id, text in tokens.items():
            tid = _as_id(raw_id)
            if not 0 <= tid < self.vocab_size:
                raise GrammarMaskError(f"token id {tid} is not in [0, {self.vocab_size})")
            if tid in self._stop_set:
                raise GrammarMaskError(f"stop id {tid} is also a text token ({text!r}): a stop id is only a stop")
            if not isinstance(text, str):
                raise GrammarMaskError(f"token {tid}'s text must be a str, not {type(text).__name__}")
            if not text:
                empty += 1              # xgrammar's "" is a padded, special id (tokenizer_info.py:310, 469-472)
                continue
            self._text[tid] = text
        for tid, texts in (untexted or {}).items():
            for text in texts:
                if text and set(text) <= self.alphabet:
                    raise GrammarMaskError(
                        f"added token {tid} {text!r} is spelled in the regex's alphabet: xgrammar matches an added "
                        f"token by its text, and this NFA never allows one, so the two masks could differ")

        # the trie of the tokens spelled in the alphabet: no other token survives its first foreign character
        self._trie: dict = {}
        self._n_in_alphabet = 0
        for tid, text in self._text.items():
            if set(text) <= self.alphabet:
                self._n_in_alphabet += 1
                node = self._trie
                for c in text:
                    node = node.setdefault(c, {})
                node.setdefault(None, []).append(tid)

        self._closure: dict[int, frozenset[int]] = {}
        self._steps: dict[tuple[frozenset[int], str], frozenset[int]] = {}
        self._moves: dict[tuple[frozenset[int], int], frozenset[int]] = {}
        self._allowed_memo: dict[frozenset[int], tuple[tuple[int, ...], frozenset[int]]] = {}
        self._start = self._close(self._nfa.start)
        self._empty = empty
        self._source = dict(source) if source else None
        self._untexted = len(untexted or {})
        self.build_s = time.perf_counter() - t0

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_tokens(cls, tokens: Mapping[int, str], regex: str, *, vocab_size: int,
                    stop_ids: Iterable[int]) -> GrammarMask:
        """A mask over an id -> text map: each text is what the token writes, character by character. Ids below
        vocab_size that are not in `tokens` (and not stop ids) are never allowed."""
        return cls(tokens=tokens, regex=regex, vocab_size=vocab_size, stop_ids=stop_ids)

    @classmethod
    def from_tokenizer_json(cls, path: str | Path, regex: str, *, vocab_size: int,
                            stop_ids: Iterable[int]) -> GrammarMask:
        """A mask over a byte-level BPE tokenizer.json, as xgrammar 0.2.8 builds its TokenizerInfo from it through
        vLLM 0.32 (the module docstring, items 1-6). vocab_size is the model's (config.json), not the tokenizer's."""
        t0 = time.perf_counter()
        path = Path(path)
        raw = path.read_bytes()
        try:
            tj = json.loads(raw)
        except ValueError as e:
            raise GrammarMaskError(f"{path} is not JSON: {e}") from None
        model = tj.get("model") if isinstance(tj, Mapping) else None
        if not isinstance(model, Mapping) or model.get("type") != "BPE" or not isinstance(model.get("vocab"), Mapping):
            raise GrammarMaskError(f"{path}: not a BPE tokenizer.json (model.type and model.vocab)")
        vocab: Mapping[str, Any] = model["vocab"]
        added = list(tj.get("added_tokens") or ())
        if not all(isinstance(a, Mapping) and isinstance(a.get("content"), str) for a in added):
            raise GrammarMaskError(f"{path}: an added token without a string content")
        get_vocab = {str(t): _as_id(i) for t, i in vocab.items()}
        get_vocab.update({a["content"]: _as_id(a.get("id")) for a in added})
        vocab_type = xgrammar_vocab_type(get_vocab)
        if vocab_type != "BYTE_LEVEL":
            raise GrammarMaskError(f"{path}: xgrammar's vocabulary test reads {vocab_type}, not BYTE_LEVEL "
                                   f"(tokenizer_info.py:171-198); only a byte-level vocabulary is mirrored")
        prefix = _byte_level_prefix_space(tj)
        if prefix is not False:
            raise GrammarMaskError(f"{path}: the ByteLevel parts' add_prefix_space is {prefix}, not false; "
                                   f"only false is mirrored")
        size = _as_id(vocab_size, "vocab_size")
        stops = {_as_id(s, "a stop id") for s in stop_ids}
        # one string per id and one id per string, so xgrammar's by-id fill (tokenizer_info.py:311-313) does not
        # depend on the dict's order and no model.vocab id loses its string to an added token in get_vocab
        shadowed = [a["content"] for a in added
                    if a["content"] in vocab and _as_id(vocab[a["content"]]) != _as_id(a.get("id"))]
        if shadowed:
            raise GrammarMaskError(f"{path}: added token {shadowed[0]!r} is also a model.vocab token with another id")
        if len(set(get_vocab.values())) != len(get_vocab) or len({_as_id(i) for i in vocab.values()}) != len(vocab):
            raise GrammarMaskError(f"{path}: two tokens share an id")
        tokens: dict[int, str] = {}
        dropped = 0
        for token, raw_id in vocab.items():
            tid = _as_id(raw_id)
            if tid >= size:
                dropped += 1            # tokenizer_info.py:311-313 keeps only idx < vocab_size
                continue
            try:
                tokens[tid] = bytes(CHAR_TO_BYTE[c] for c in token).decode("latin-1")
            except KeyError:
                raise GrammarMaskError(f"{path}: model.vocab token {tid} {token!r} is not spelled in the byte-level "
                                       f"alphabet") from None
        untexted: dict[int, tuple[str, ...]] = {}
        added_ids = set()
        for a in added:
            tid, content = _as_id(a.get("id")), a["content"]
            added_ids.add(tid)
            if tid >= size:
                dropped += 1
                continue
            if tid in tokens:
                continue                # model.vocab's own string on its own id (the checks above): a text token
            if tid in stops:
                continue
            texts = [content]
            if all(c in CHAR_TO_BYTE for c in content):
                texts.append(bytes(CHAR_TO_BYTE[c] for c in content).decode("latin-1"))
            untexted[tid] = tuple(dict.fromkeys(texts))
        source = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "model_vocab": len(vocab),
                  "added_tokens": len(added), "vocab_type": vocab_type, "add_prefix_space": False,
                  "dropped_at_or_past_vocab_size": dropped,
                  "stop_ids_not_added_tokens": sorted(stops - added_ids)}
        return cls(tokens=tokens, regex=regex, vocab_size=size, stop_ids=stops, untexted=untexted, source=source,
                   started=t0)

    # ------------------------------------------------------------------ the NFA's moves
    def _close(self, q: int) -> frozenset[int]:
        """The states reachable from q by epsilon moves that read a character, plus the accepting state."""
        hit = self._closure.get(q)
        if hit is None:
            label, eps, accept = self._nfa.label, self._nfa.eps, self._nfa.accept
            seen, stack, out = {q}, [q], []
            while stack:
                s = stack.pop()
                if label[s] is not None or s == accept:
                    out.append(s)
                for t in eps[s]:
                    if t not in seen:
                        seen.add(t)
                        stack.append(t)
            hit = self._closure[q] = frozenset(out)
        return hit

    def _step(self, states: frozenset[int], c: str) -> frozenset[int]:
        key = (states, c)
        hit = self._steps.get(key)
        if hit is None:
            label, to = self._nfa.label, self._nfa.to
            nxt: set[int] = set()
            for q in states:
                lab = label[q]
                if lab is not None and c in lab:
                    nxt |= self._close(to[q])
            hit = self._steps[key] = frozenset(nxt)
        return hit

    def _advance(self, states: frozenset[int], text: str) -> frozenset[int]:
        for c in text:
            states = self._step(states, c)
            if not states:
                break
        return states

    def _allowed(self, states: frozenset[int]) -> tuple[tuple[int, ...], frozenset[int]]:
        """(sorted ids, the same as a set) allowed in a state set, memoized by the set."""
        hit = self._allowed_memo.get(states)
        if hit is None:
            ids: list[int] = []
            stack = [(self._trie, states)]
            while stack:
                node, here = stack.pop()
                for c, child in node.items():
                    if c is None:
                        ids.extend(child)
                        continue
                    there = self._step(here, c)
                    if there:
                        stack.append((child, there))
            if self._nfa.accept in states:
                ids.extend(self.stop_ids)
            ordered = tuple(sorted(ids))
            hit = self._allowed_memo[states] = (ordered, frozenset(ordered))
        return hit

    # ------------------------------------------------------------------ the API
    def walk(self, ids: Sequence[int]) -> Walk:
        """Walk a completion's ids from the start, as the matcher replays them: the allowed set before each token,
        then the token. Stops at the first token outside its set, or after termination."""
        states = self._start
        allowed: list[tuple[int, ...]] = []
        terminated = False
        for i, raw in enumerate(ids):
            tid = _as_id(raw)
            if terminated:
                allowed.append(())
                return Walk(tuple(allowed), i, True)
            ordered, members = self._allowed(states)
            allowed.append(ordered)
            if tid not in members:
                return Walk(tuple(allowed), i, False)
            if tid in self._stop_set:
                terminated = True
                continue
            key = (states, tid)
            nxt = self._moves.get(key)
            if nxt is None:
                nxt = self._moves[key] = self._advance(states, self._text[tid])
            states = nxt
        return Walk(tuple(allowed), None, terminated)

    def random_walk(self, rng: random.Random, max_len: int = 512) -> list[int]:
        """Ids drawn uniformly from the allowed set at each position, from the start: the mask's own walk through the
        grammar, so the preflight can compare it with xgrammar at token boundaries no tokenizer would choose
        (lofi_rl.rl.mask_preflight). It stops after the first stop id, at max_len ids, or where nothing is allowed (a
        vocabulary that cannot continue the text; one with a token for every ASCII character, as Qwen's has, always
        can, since the regex is ASCII).
        walk() takes every id it returns with no violation, terminated exactly when the last id is a stop id. The same
        seeded `rng` gives the same walk: the sets are sorted and the draws are rng.choice."""
        n = _as_id(max_len, "max_len")
        if n < 0:
            raise GrammarMaskError(f"max_len must be >= 0, not {n}")
        states = self._start
        ids: list[int] = []
        while len(ids) < n:
            ordered, _ = self._allowed(states)
            if not ordered:
                break
            tid = rng.choice(ordered)
            ids.append(tid)
            if tid in self._stop_set:
                break
            key = (states, tid)                 # walk()'s memoized move, so a later walk() of these ids reuses it
            nxt = self._moves.get(key)
            if nxt is None:
                nxt = self._moves[key] = self._advance(states, self._text[tid])
            states = nxt
        return ids

    def allowed_after_text(self, text: str) -> tuple[int, ...]:
        """The sorted ids allowed right after `text`, read character by character from the start. GrammarMaskError
        if the text leaves the regex's language."""
        states = self._start
        for k, c in enumerate(text):
            states = self._step(states, c)
            if not states:
                raise GrammarMaskError(f"the text leaves the regex's language at character {k} ({c!r})")
        return self._allowed(states)[0]

    def token_text(self, token_id: int) -> str | None:
        """What a text token writes (bytes as latin-1 characters, for a tokenizer.json); None for any other id."""
        return self._text.get(_as_id(token_id))

    def record(self) -> dict[str, Any]:
        """The mask's identity and size, for run.json."""
        return {
            "builder": "lofi_rl.grammar_mask",
            "regex_sha256": self.regex_sha256,
            "regex_chars": len(self.regex),
            "vocab_size": self.vocab_size,
            "stop_ids": list(self.stop_ids),
            "tokenizer": dict(self._source) if self._source else None,
            "text_tokens": len(self._text),
            "tokens_in_alphabet": self._n_in_alphabet,
            "empty_tokens": self._empty,
            "added_tokens_never_allowed": self._untexted,
            "never_allowed": self.vocab_size - len(self._text) - len(self.stop_ids),
            "alphabet": "".join(sorted(self.alphabet)),
            "nfa_states": len(self._nfa.label),
            "nfa_char_states": sum(lab is not None for lab in self._nfa.label),
            "memoized_state_sets": len(self._allowed_memo),
            "build_s": round(self.build_s, 4),
        }

    def __repr__(self) -> str:
        return (f"GrammarMask(regex {self.regex_sha256[:8]}, vocab_size {self.vocab_size}, stop_ids {self.stop_ids}, "
                f"{len(self._text)} text tokens)")


def _byte_level_prefix_space(tj: Mapping[str, Any]) -> bool | None:
    """add_prefix_space of the tokenizer.json's ByteLevel pre-tokenizer and decoder parts: True if any says true,
    False if they all say false, None when there is no ByteLevel part to read."""
    found: list[bool] = []

    def visit(part: Any) -> None:
        if isinstance(part, Mapping):
            if part.get("type") == "ByteLevel":
                found.append(bool(part.get("add_prefix_space", False)))
            for key in ("pretokenizers", "decoders"):
                for sub in part.get(key) or ():
                    visit(sub)

    visit(tj.get("pre_tokenizer"))
    visit(tj.get("decoder"))
    return None if not found else any(found)


# ------------------------------------------------------------------------------------------- re-encoding
class _BPE:
    """The Qwen2 tokenizer's encoding of ASCII text (encode_bpe), read once per tokenizer.json dict."""

    def __init__(self, tokenizer: Mapping[str, Any]):
        problems = _qwen2_problems(tokenizer)
        if problems:
            raise GrammarMaskError("encode_bpe reproduces only the Qwen2 byte-level BPE tokenizer: "
                                   + "; ".join(problems))
        model = tokenizer["model"]
        self.vocab: Mapping[str, int] = model["vocab"]
        self.ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(model["merges"]):
            pair = tuple(merge) if isinstance(merge, (list, tuple)) else tuple(str(merge).split(" "))
            if len(pair) != 2 or pair in self.ranks:
                raise GrammarMaskError(f"merge {rank} {merge!r} is malformed or repeated")
            self.ranks[pair] = rank
        self.added = tuple(a["content"] for a in tokenizer.get("added_tokens") or () if a.get("content"))
        self.cache: dict[str, list[int]] = {}

    def word(self, piece: str) -> list[int]:
        hit = self.cache.get(piece)
        if hit is None:
            sym = [BYTE_TO_CHAR[b] for b in piece.encode("ascii")]
            while len(sym) > 1:
                best, at = None, -1
                for k in range(len(sym) - 1):
                    rank = self.ranks.get((sym[k], sym[k + 1]))
                    if rank is not None and (best is None or rank < best):
                        best, at = rank, k
                if best is None:
                    break
                sym[at:at + 2] = [sym[at] + sym[at + 1]]
            try:
                hit = self.cache[piece] = [self.vocab[s] for s in sym]
            except KeyError as e:
                raise GrammarMaskError(f"BPE symbol {e.args[0]!r} of {piece!r} is not in model.vocab") from None
        return hit

    def encode(self, text: str) -> list[int]:
        if not text.isascii():
            raise GrammarMaskError("encode_bpe reads ASCII text only (the LoopScript regex is ASCII)")
        held = [a for a in self.added if a in text]
        if held:
            raise GrammarMaskError(f"the text holds the added token {held[0]!r}, which the tokenizer would split "
                                   f"out first; encode_bpe does not")
        ids: list[int] = []
        pos = 0
        for m in _QWEN2_ASCII.finditer(text):
            if m.start() > pos:                     # "Isolated": a gap is a piece too (the pattern leaves none)
                ids += self.word(text[pos:m.start()])
            ids += self.word(m.group())
            pos = m.end()
        if pos < len(text):
            ids += self.word(text[pos:])
        return ids


def _qwen2_problems(tj: Mapping[str, Any]) -> list[str]:
    """What keeps a tokenizer.json from being the Qwen2 one encode_bpe reproduces (empty when it is)."""
    model = tj.get("model") if isinstance(tj, Mapping) else None
    if not isinstance(model, Mapping):
        return ["no model"]
    problems = []
    if model.get("type") != "BPE" or not isinstance(model.get("vocab"), Mapping) or "merges" not in model:
        problems.append("model is not BPE with a vocab and merges")
    for key, want in (("byte_fallback", False), ("ignore_merges", False), ("fuse_unk", False)):
        if model.get(key, False) is not want:
            problems.append(f"model.{key} is {model.get(key)!r}")
    if model.get("dropout") not in (None, 0, 0.0):
        problems.append(f"model.dropout is {model.get('dropout')!r}")
    for key in ("continuing_subword_prefix", "end_of_word_suffix"):
        if model.get(key) not in (None, ""):
            problems.append(f"model.{key} is {model.get(key)!r}")
    norm = tj.get("normalizer")
    if norm is not None and (not isinstance(norm, Mapping) or norm.get("type") != "NFC"):
        problems.append(f"normalizer is {norm!r}, not NFC")
    pre = tj.get("pre_tokenizer")
    parts = pre.get("pretokenizers") if isinstance(pre, Mapping) and pre.get("type") == "Sequence" else None
    ok = (isinstance(parts, list) and len(parts) == 2
          and isinstance(parts[0], Mapping) and parts[0].get("type") == "Split"
          and parts[0].get("pattern") == {"Regex": QWEN2_PRETOKENIZE}
          and parts[0].get("behavior") == "Isolated" and parts[0].get("invert") is False
          and isinstance(parts[1], Mapping) and parts[1].get("type") == "ByteLevel"
          and parts[1].get("add_prefix_space") is False and parts[1].get("use_regex") is False)
    if not ok:
        problems.append("pre_tokenizer is not Sequence[Split(Qwen2 pattern, Isolated), ByteLevel(no prefix space, "
                        "no regex)]")
    return problems


_ENCODERS: dict[int, tuple[Mapping[str, Any], _BPE]] = {}


def encode_bpe(tokenizer: Mapping[str, Any], text: str) -> list[int]:
    """The canonical Qwen2 encoding of ASCII `text` with a parsed tokenizer.json (no special tokens added): the laptop
    re-encoding D-021's M2 walks. The merges are read once per tokenizer dict (keyed by the object, which is kept)."""
    hit = _ENCODERS.get(id(tokenizer))
    if hit is None or hit[0] is not tokenizer:
        if len(_ENCODERS) >= 4:
            _ENCODERS.clear()
        hit = _ENCODERS[id(tokenizer)] = (tokenizer, _BPE(tokenizer))
    return hit[1].encode(text)
