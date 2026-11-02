"""lofi_rl.rl.masked: the grammar-masked GRPO trainer (docs/DECISIONS.md D-021 Part A, M1; build spec §2 D).

torch, TRL and xgrammar are not on the laptop, so everything runs on plain lists:
- the alignment proof: TRL 1.17.1's support path, emulated stage by stage from grpo_trainer.py and trainer/utils.py,
  puts completion token c's allowed set at token c's label position. It agrees with D-021's registered logits form (the
  fill) on random logits, and each off-by-one variant of a stage does not;
- the pure helpers: the completion mask's valid length, the support (AND with TRL's candidates, -1 padding, forced and
  largest sets, violations, cache misses), both reference log-prob paths, -log Z, the IS-weight stats, the bitmask
  decoder, the walk comparison, the pass labels and PEFT's adapter state;
- make_trainer_class on a stand-in GRPOTrainer whose passes run that emulation with a stand-in causal model over a toy
  grammar (12 ids, 2 stop ids): the support handed to every pass, the adapter check, violations stopping the launch
  in _generate, the old pass's -log Z and self-check, path fill, the step record and TRL's metrics;
- XgrammarReplay against a stand-in xgrammar module: the calls vLLM 0.32's backend makes, the walk, and its record
  (xgrammar's version from its distribution's metadata, the matcher's own stop ids)."""
import contextlib
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from lofi_rl.rl import masked as M

ROOT = Path(__file__).resolve().parents[1]
V = 12                       # the toy vocabulary: ids 0-9 are symbols, 10 and 11 the stop ids
STOPS = (10, 11)
PAD = 11                     # as Qwen3's pad, <|endoftext|>, is also a stop id
REGEX = "toy grammar"        # MaskSetup checks the sha of whatever text names the grammar
SHA = M.regex_sha256(REGEX)
PROMPTS = [[3, 4], [7, 8, 9], [1], [6, 6, 2, 9]]
COMPLETIONS = [[0, 1, 4, 10], [0, 1, 4, 10],                 # prompt 0: one completion twice
               [2, 5, 6, 9, 11], [2, 3, 6, 11],              # prompt 1: 6 is forced after 5
               [1, 2, 5, 6, 7, 8, 9, 0], [1, 4, 5, 6, 10],   # prompt 2: the first is cut at 8 tokens (no stop id)
               [0, 3, 10], [0, 1, 2, 3, 4, 5, 6, 7]]         # prompt 3: the second is cut too
INPUTS = [{"prompt": p} for p in PROMPTS for _ in range(2)]
VALID = [4, 4, 5, 4, 0, 5, 3, 0]                             # mask_truncated_completions zeroes the cut ones
SPEC_KEYS = ("tokens", "rows", "rows_scored", "positions", "violations", "nfa_mismatch", "nfa_positions",
             "nfa_examples", "neglogz", "neglogz_tokens", "forced_share", "k_max", "cache_miss", "trl_sampling_mask",
             "seq_weights", "seq_weights_raw", "seq_valid", "path", "selfcheck", "timing")


# ------------------------------------------------------------------------------------------ the toy grammar and model
@dataclass(frozen=True)
class W:
    """A walk with lofi_rl.grammar_mask.Walk's fields (section A's interface)."""
    allowed: tuple
    violation: int | None
    terminated: bool


class ToyGrammar:
    """A closed grammar over 10 symbol ids: the first three may start; after symbol index 5 only index 6 (a forced
    token); after index x, x + 1 and x + 3 (mod 10), plus the stop ids once two tokens are in (an accepting state).
    `quirk` (prefix length, id) makes an NFA that also allows that id there. walk() is the replay/NFA interface."""

    def __init__(self, symbols=tuple(range(10)), stops=STOPS, vocab_size=V, regex=REGEX, quirk=None):
        self.symbols, self.stop_ids, self.vocab_size = tuple(symbols), tuple(stops), vocab_size
        self.regex_sha256 = M.regex_sha256(regex)
        self.quirk = quirk
        self.walks = 0

    def allowed_after(self, prefix):
        s = self.symbols
        if prefix and prefix[-1] in self.stop_ids:          # terminated: nothing more is allowed
            return ()
        if not prefix:
            out = {s[0], s[1], s[2]}
        else:
            x = s.index(prefix[-1])
            out = {s[6]} if x == 5 else {s[(x + 1) % 10], s[(x + 3) % 10]}
            if x != 5 and len(prefix) >= 2:
                out |= set(self.stop_ids)
        if self.quirk and len(prefix) == self.quirk[0]:
            out |= {self.quirk[1]}
        return tuple(sorted(out))

    def walk(self, ids):
        self.walks += 1
        allowed = []
        for i, t in enumerate(ids):
            if i and ids[i - 1] in self.stop_ids:
                allowed.append(())
                return W(tuple(allowed), i, True)
            allowed.append(self.allowed_after(list(ids[:i])))
            if t not in allowed[-1]:
                return W(tuple(allowed), i, False)
        return W(tuple(allowed), None, bool(ids) and ids[-1] in self.stop_ids)


def toy_logits(content, salt=0):
    seed = int.from_bytes(hashlib.sha256(repr((salt, tuple(content))).encode()).digest()[:8], "big")
    rng = random.Random(seed)
    return [rng.gauss(0.0, 2.0) for _ in range(V)]


def logits_at(row, j, salt=0):
    """The stand-in causal model: hidden state j's logits depend on the row's tokens up to j after its left padding (no
    prompt here starts with PAD), so a row scores the same in any batch, order or pass."""
    k = 0
    while k < len(row) and row[k] == PAD:
        k += 1
    return toy_logits(row[k:j + 1], salt)


def lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def vllm_logps(prompt, comp, grammar):
    """What vLLM reports under the grammar: the sampled token's log-prob after the bitmask (-inf outside the allowed
    set), written independently of the module's helpers."""
    out = []
    for t, tok in enumerate(comp):
        z = toy_logits(prompt + comp[:t])
        allowed = grammar.allowed_after(comp[:t]) or (tok,)   # (a token after a stop id only in the refusal tests)
        out.append(z[tok] - lse([z[i] for i in allowed]))
    return out


def layout(width=10):
    """The fixture's rows as TRL builds them: prompts left-padded with PAD, completions right-padded to `width`
    columns (wider than the longest completion, as pad_to_multiple_of makes them), the truncated ones masked."""
    p_len = max(map(len, PROMPTS))
    ids, att = [], []
    for i, comp in enumerate(COMPLETIONS):
        p = PROMPTS[i // 2]
        ids.append([PAD] * (p_len - len(p)) + p + comp + [PAD] * (width - len(comp)))
        on = comp[-1] in STOPS
        att.append([0] * (p_len - len(p)) + [1] * len(p) + [int(on)] * len(comp) + [0] * (width - len(comp)))
    g = ToyGrammar()
    cache = M.AllowedCache(g)
    for comp in COMPLETIONS:
        cache.put(comp, g.walk(comp).allowed)
    rows = [r[len(r) - width:] for r in ids]
    build = M.build_support(rows, [M.valid_length(a[len(a) - width:]) for a in att], cache)
    return ids, att, width, build, g


# ------------------------------------------------------------------------------------------ 1. the alignment proof
def test_trls_placement_and_shift_put_each_completion_tokens_allowed_set_at_its_own_label():
    ids, att, width, build, g = layout()
    assert list(build.valid) == VALID
    logps, trace = M.emulate_trl_logps(ids, att, width, build.support, logits_at)
    checked = 0
    for b, comp in enumerate(COMPLETIONS):
        n = len(ids[b])
        for c in range(width):
            if c < VALID[b]:
                pos, label, support = trace[b][c]
                assert pos == n - width + c, "scored at the completion token's own input column"
                assert label == ids[b][pos] == comp[c], "the label is the completion token"
                assert support == list(g.allowed_after(comp[:c])), "against the set it was drawn from"
                checked += 1
            else:
                assert trace[b][c] is None and logps[b][c] == 0.0
    assert checked == sum(VALID) == 25


@pytest.mark.parametrize("salt", range(4))
def test_the_support_path_equals_the_registered_logits_form_on_random_logits(salt):
    ids, att, width, build, _ = layout()
    model = lambda row, j: logits_at(row, j, salt)                     # noqa: E731
    trl, _ = M.emulate_trl_logps(ids, att, width, build.support, model)
    fill, unmasked, entropy = M.emulate_logits_form(ids, att, width, build.support, model)
    trl_unmasked, _ = M.emulate_trl_logps(ids, att, width, None, model)
    diffs = [abs(a - b) for ra, rb in zip(trl, fill) for a, b in zip(ra, rb)]
    assert max(diffs) < 1e-9
    assert max(abs(a - b) for ra, rb in zip(trl_unmasked, unmasked) for a, b in zip(ra, rb)) < 1e-9
    assert all(math.isfinite(x) for r in fill + entropy for x in r)
    assert all(m >= u - 1e-12 for rm, ru in zip(fill, unmasked) for m, u in zip(rm, ru)), "a subset raises a log-prob"


def _shift_layout(support):
    """Our layout one column early: support[b][c] holds the set of token c + 1."""
    return [row[1:] + [[-1] * len(row[0])] for row in support]


def _place_early(seq_len, logits_to_keep, rows, k):
    grid = M.trl_support_ids(seq_len, logits_to_keep, rows, k)
    return grid[1:] + [[-1] * k]                                       # the grid written one column earlier


def _no_support_shift(labels, support_ids=None):
    lab = list(labels[1:])                                             # the labels shift, the support does not
    return [(j, lab[j], None if support_ids is None else support_ids[j]) for j in range(len(lab))]


def _keep_one_early(row, logits_to_keep):
    return list(row[len(row) - logits_to_keep - 1:len(row) - 1])


@pytest.mark.parametrize("variant", ["our layout shifted", "placed one column early", "support not shifted",
                                     "output sliced one early"])
def test_every_off_by_one_variant_disagrees_with_the_logits_form(variant):
    ids, att, width, build, g = layout()
    support, kw = build.support, {}
    if variant == "our layout shifted":
        support = _shift_layout(build.support)
    elif variant == "placed one column early":
        kw = {"place": _place_early}
    elif variant == "support not shifted":
        kw = {"fused": _no_support_shift}
    else:
        kw = {"keep_last": _keep_one_early}
    trl, trace = M.emulate_trl_logps(ids, att, width, support, logits_at, **kw)
    fill, _, _ = M.emulate_logits_form(ids, att, width, build.support, logits_at)
    worst = max(abs(a - b) for ra, rb in zip(trl, fill) for a, b in zip(ra, rb))
    assert worst > 0.05, f"{variant}: the agreement test would not see it ({worst})"
    wrong = sum(1 for b, comp in enumerate(COMPLETIONS) for c in range(VALID[b])
                if trace[b][c] is None or trace[b][c][2] != list(g.allowed_after(comp[:c]))
                or trace[b][c][1] != comp[c])
    assert wrong > 0


# ------------------------------------------------------------------------------------------ 2. pure helpers
def test_the_valid_length_is_the_ones_prefix_and_anything_else_is_refused():
    assert M.valid_length([1, 1, 1, 0, 0]) == 3
    assert M.valid_length([0, 0, 0]) == 0 and M.valid_length([1, 1]) == 2 and M.valid_length([]) == 0
    assert M.valid_length([True, 1.0, False]) == 2
    for row, why in (([1, 0, 1], "after a 0"), ([0, 1], "after a 0"), ([1, 2, 0], "not 0 or 1"),
                     ([1, float("nan")], "not 0 or 1")):
        with pytest.raises(M.MaskViolation, match=why):
            M.valid_length(row)


def test_the_support_holds_each_valid_tokens_allowed_set_padded_with_minus_one():
    cache = M.AllowedCache(ToyGrammar())
    cache.put([5, 6], [(1, 5, 7), (6,)])
    cache.put([1, 2, 3], [(1, 2), (2,), (3, 4)])
    rows = [[5, 6, 11, 11], [1, 2, 3, 11], [9, 9, 9, 9]]
    b = M.build_support(rows, [2, 3, 0], cache)
    assert b.support == [[[1, 5, 7], [6, -1, -1], [-1] * 3, [-1] * 3],
                         [[1, 2, -1], [2, -1, -1], [3, 4, -1], [-1] * 3],
                         [[-1] * 3] * 4]
    assert (b.positions, b.forced, b.k_max, b.rows, b.rows_scored, b.violations, b.cache_miss) == (5, 2, 3, 3, 2, (), 0)
    assert b.valid == (2, 3, 0)


def test_trls_candidates_are_anded_into_the_support():
    cache = M.AllowedCache(ToyGrammar())
    cache.put([5, 6], [(1, 5, 7), (6,)])
    cache.put([1, 2, 3], [(1, 2), (2,), (3, 4)])
    rows = [[5, 6, 11, 11], [1, 2, 3, 11]]
    cands = [[[5, 7, 9], [6, -1, -1], [-1] * 3, [-1] * 3], [[1, 2, -1], [2, 3, -1], [3, -1, -1], [-1] * 3]]
    b = M.build_support(rows, [2, 3], cache, cands)
    assert b.support == [[[5, 7], [6, -1], [-1, -1], [-1, -1]], [[1, 2], [2, -1], [3, -1], [-1, -1]]]
    assert (b.k_max, b.forced, b.violations) == (2, 3, ())
    cands[1][1] = [3, -1, -1]                                    # TRL's set at (1, 1) drops the sampled token
    b = M.build_support(rows, [2, 3], cache, cands)
    assert b.violations == ((1, 1, 2),) and b.support[1][1:] == [[-1, -1]] * 3


def test_a_token_outside_its_set_is_a_violation_and_ends_its_row():
    cache = M.AllowedCache(ToyGrammar())
    cache.put([4, 4, 4], [(4,), (5,), (4,)])
    b = M.build_support([[4, 4, 4]], [3], cache)
    assert b.violations == ((0, 1, 4),) and b.support == [[[4], [-1], [-1]]]
    every_row_truncated = M.build_support([[1, 2], [3, 4]], [0, 0], cache)
    assert every_row_truncated.support == [[[-1], [-1]], [[-1], [-1]]] and every_row_truncated.k_max == 0
    with pytest.raises(M.MaskError, match="completion columns"):
        M.build_support([[1, 2], [3]], [0, 0], cache)


def test_a_cache_miss_is_walked_counted_and_kept_only_when_the_walk_holds():
    g = ToyGrammar()
    cache = M.AllowedCache(g)
    b = M.build_support([[0, 1, 4, 10, 11]], [4], cache)
    assert b.cache_miss == 1 and g.walks == 1 and b.violations == () and [0, 1, 4, 10] in cache
    again = M.build_support([[0, 1, 4, 10, 11]], [4], cache)
    assert again.cache_miss == 0 and g.walks == 1 and again.support == b.support
    bad = M.build_support([[0, 2, 11]], [2], cache)                   # 2 may not follow 0
    assert bad.cache_miss == 1 and bad.violations == ((0, 1, 2),) and [0, 2] not in cache
    after_stop = M.build_support([[0, 3, 10, 1]], [4], cache)         # a token after the stop id
    assert after_stop.violations == ((0, 3, 1),)
    cache.clear()
    assert len(cache) == 0


def test_equal_allowed_sets_are_interned_once():
    cache = M.AllowedCache(ToyGrammar())
    cache.put([0, 1], [(1, 3), [1, 3]])
    cache.put([2, 3], [[1, 3], (9,)])
    a, _ = cache.get([0, 1])
    b, _ = cache.get([2, 3])
    assert a[0] is a[1] is b[0]


def test_the_bitmask_decoder_reads_the_set_bits_of_int32_words():
    assert M.decode_bitmask([-1], 40) == tuple(range(32))
    assert M.decode_bitmask([0, -2147483648], 64) == (63,)            # bit 31 of word 1, stored as an int32
    assert M.decode_bitmask([0b1010, 0, 1], 70) == (1, 3, 64)
    assert M.decode_bitmask([-1, -1], 40) == tuple(range(40))         # ids past the vocabulary are dropped
    assert M.bits_to_ids([(0, 1 << 5), (3, 6)], 200) == (5, 97, 98)
    assert M.decode_bitmask([0, 0], 64) == ()


def test_both_reference_paths_agree_where_the_target_is_in_its_support_and_the_fill_stays_finite():
    rng = random.Random(7)
    for _ in range(400):
        z = [rng.gauss(0.0, 5.0) for _ in range(V)]
        support = rng.sample(range(V), rng.randint(1, V))
        target, t = rng.choice(support), rng.choice([1.0, 0.7, 1.3])
        a = M.support_logp(z, target, support + [-1, -1], t)
        b = M.fill_logp(z, target, support + [-1], t)
        assert math.isfinite(b) and abs(a - b) < 1e-9
    z = [0.3, -1.0, 2.0, 0.5]
    assert M.support_logp(z, 2, [2, -1]) == 0.0 and M.fill_logp(z, 2, [2]) == 0.0      # a forced token
    assert abs(M.support_logp(z, 1) - (z[1] - lse(z))) < 1e-12                         # no support: the vocabulary
    assert abs(M.fill_logp(z, 1, None) - (z[1] - lse(z))) < 1e-12
    outside = M.support_logp(z, 1, [0, 3])          # the kernel cannot tell: e^z1 / (e^z1 + e^z0 + e^z3)
    assert abs(outside - (z[1] - lse([z[1], z[0], z[3]]))) < 1e-12
    assert math.isfinite(M.fill_logp(z, 1, [0, 3])) and M.fill_logp(z, 1, [0, 3]) < -1e8
    assert math.isfinite(M.fill_logp(z, 1, [-1, -1]))                                   # all filled: still finite
    p = [math.exp(x - lse(z)) for x in z]
    assert abs(M.ref_entropy(z) + sum(q * math.log(q) for q in p)) < 1e-12


def test_the_support_reference_is_the_kernels_own_formula_with_its_max_over_the_whole_support():
    """trl/kernels/chunked_logprob.py:595-597, written out here: z = logit * (1 / T); M the largest z over the whole
    support, the target included; S the sum of e^(z - M) over the support's other tokens; -softplus((M - z_t) + log S).
    Every row's target is its support's largest logit, where a max over the other tokens alone rounds differently."""
    rng = random.Random(11)
    for _ in range(400):
        z = [rng.gauss(0.0, 4.0) for _ in range(V)]
        support = sorted(rng.sample(range(V), rng.randint(2, V)))
        target, t = max(support, key=lambda i: z[i]), rng.choice([1.0, 0.7, 1.3])
        zt = [x * (1.0 / t) for x in z]
        m = max(zt[i] for i in support)
        x = (m - zt[target]) + math.log(sum(math.exp(zt[i] - m) for i in support if i != target))
        softplus = x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))
        assert M.support_logp(z, target, support + [-1], t) == -softplus
    assert M.support_logp([0.0, -800.0, -900.0], 0) == 0.0, "the others underflow: log S = -inf, -softplus(-inf) = 0"


def test_neglogz_is_token_weighted_over_the_valid_positions():
    r = M.neglogz([[-0.1, 0.0, -5.0], [-1.0, 9.0, 9.0]], [[-0.6, -0.3, -9.0], [-1.2, 0.0, 0.0]], [2, 1])
    assert r["tokens"] == 3 and r["nonfinite"] == 0 and abs(r["mean"] - (0.5 + 0.3 + 0.2) / 3) < 1e-12
    r = M.neglogz([[float("nan"), -1.0]], [[-1.0, -2.0]], [2])
    assert (r["tokens"], r["nonfinite"], r["mean"]) == (1, 1, 1.0)
    assert M.neglogz([[0.0]], [[0.0]], [0])["mean"] is None


def test_the_raw_weights_follow_trls_sequence_ratio():
    w, logs = M.raw_seq_weights([[-1.0, -2.0, 0.0], [0.0, 0.0, 0.0]],
                                [[-1.5, -2.0, float("nan")], [-1.0, -1.0, -1.0]], [[1, 1, 0], [0, 0, 0]])
    assert logs == [0.5, 0.0] and w == [math.exp(0.5), 1.0]           # NaN x 0 counts 0; a masked row weighs 1
    w, logs = M.raw_seq_weights([[400.0, 400.0]], [[0.0, 0.0]], [[1, 1]])
    assert w == [math.inf] and logs == [800.0]                        # past fp32's range, as TRL's exp would be
    w, _ = M.raw_seq_weights([[0.0]], [[-math.inf]], [[1]])
    assert w == [math.inf]


def test_the_sequence_weight_stats_give_d021_m11s_mean_and_largest_share():
    s = M.seq_weight_stats([0.5, 1.5, 2.0])
    assert (s["n"], s["sum"], s["max"], s["max_share"]) == (3, 4.0, 2.0, 0.5) and abs(s["mean"] - 4 / 3) < 1e-12
    s = M.seq_weight_stats([0.5, 1.5, 2.0], [True, False, True])
    assert (s["n"], s["sum"], s["max_share"]) == (2, 2.5, 0.8)
    assert M.seq_weight_stats([0.0, 0.0])["max_share"] is None
    assert M.seq_weight_stats([])["mean"] is None and M.seq_weight_stats(None)["n"] == 0
    d = M.diff_stats([[1.0, 2.0, 9.0]], [[1.5, 1.0, 0.0]], [2])
    assert d == {"positions": 2, "mean_abs_diff": 0.75, "max_abs_diff": 1.0, "nonfinite": 0}


def test_walks_are_compared_set_by_set_and_by_how_they_end():
    a = W(((1, 2), (3,), (4, 5)), None, True)
    assert M.compare_walks(a, W(((2, 1), (3,), (4, 5)), None, True)) == (3, [])
    n, d = M.compare_walks(a, W(((1, 2), (3, 9), (4, 5)), None, True), ids=[1, 3, 4])
    assert n == 3 and d == [{"pos": 1, "token": 3, "n_xgrammar": 1, "n_nfa": 2, "only_xgrammar": [],
                             "only_nfa": [9]}]
    n, d = M.compare_walks(a, W(((1, 2), (3,)), 1, False))
    assert n == 2 and d == [{"pos": 2, "end": {"xgrammar": [3, None, True], "nfa": [2, 1, False]}}]


def test_trls_own_sampling_mask_is_summarized():
    assert M.summarize_trl_mask(None) is None
    assert M.summarize_trl_mask([[[1, 2], [3]], [[4, 5, 6]]]) == {"completions": 2, "positions": 3, "k_max": 3}


# ------------------------------------------------------------------------------------------ 3. passes and adapters
class LoraLayer:
    """A module with PEFT's BaseTunerLayer `disable_adapters` property (peft/tuners/tuners_utils.py:1973-1976)."""

    def __init__(self):
        self._disable_adapters = False

    @property
    def disable_adapters(self):
        return self._disable_adapters


class PeftModel:
    """The stand-in PEFT model: tuner layers, PeftModel's own flag, and disable_adapter() as PEFT's LoRA branch sets
    them (peft/peft_model.py:1060-1076); transformers' disable_adapters is a method, not a property."""

    def __init__(self, layers=3):
        self.layers = [LoraLayer() for _ in range(layers)]
        self._adapters_disabled = False
        self.training = True

    def disable_adapters(self):                      # transformers' PeftAdapterMixin method: never read as state
        raise AssertionError("adapter_state_of must not call it")

    def modules(self):
        return [self, *self.layers]

    @contextlib.contextmanager
    def disable_adapter(self):
        for layer in self.layers:
            layer._disable_adapters = True
        self._adapters_disabled = True
        try:
            yield
        finally:
            for layer in self.layers:
                layer._disable_adapters = False
            self._adapters_disabled = False


def test_the_pass_labels_follow_trls_call_order():
    assert [M.pass_label(False, i) for i in range(4)] == ["old", "ref", "loss", "loss"]
    assert M.pass_label(True, 0) == M.pass_label(True, 1) == "loss"


def test_peft_adapter_state_is_read_from_the_tuner_layers_and_the_peft_flag():
    m = PeftModel()
    assert M.adapter_state_of(m) == "enabled"
    with m.disable_adapter():
        assert M.adapter_state_of(m) == "disabled"
    m.layers[1]._disable_adapters = True
    assert M.adapter_state_of(m) == "irregular"
    m.layers[1]._disable_adapters = False
    m._adapters_disabled = True
    assert M.adapter_state_of(m) == "irregular"
    assert M.adapter_state_of(PeftModel(layers=0)) == "none" and M.adapter_state_of(object()) == "none"
    for label, state in (("old", "enabled"), ("ref", "disabled"), ("loss", "enabled")):
        M.check_adapters(label, state)
    for label, state in (("old", "disabled"), ("ref", "enabled"), ("loss", "irregular"), ("ref", "none")):
        with pytest.raises(M.MaskError, match=f"the {label} pass ran with the LoRA adapters {state}"):
            M.check_adapters(label, state)


# ------------------------------------------------------------------------------------------ 4. the trainer class
class GradMode:
    """torch's grad mode, shared by the stand-in trainer and the list ops."""

    def __init__(self):
        self.on = True

    @contextlib.contextmanager
    def off(self):
        prev, self.on = self.on, False
        try:
            yield
        finally:
            self.on = prev


class ListOps:
    """masked.TorchOps's operations on lists (a tensor is a nested list here)."""

    def __init__(self, grad):
        self.grad = grad
        self.fills = []

    def grad_enabled(self):
        return self.grad.on

    def adapter_state(self, trainer, model):
        return M.adapter_state_of(trainer.accelerator.unwrap_model(model))

    def completion_rows(self, ids, att, c):
        return [list(r[len(r) - c:]) for r in ids], [list(r[len(r) - c:]) for r in att]

    def ints(self, t):
        return None if t is None else copy.deepcopy(t)

    def floats(self, t):
        return None if t is None else [[float(x) for x in r] for r in t]

    def long_tensor(self, nested, like):
        return copy.deepcopy(nested)

    def take_rows(self, t, rows):
        return None if t is None else [t[i] for i in rows]

    def fill_logps(self, trainer, model, ids, att, c, support, *, batch_size, compute_entropy, want_unmasked):
        self.fills.append(len(ids))
        lp, um, ent = M.emulate_logits_form(ids, att, c, support, logits_at, trainer.temperature)
        return lp, (ent if compute_entropy else None), (um if want_unmasked else None)


class FakeTRLTrainer:
    """A stand-in for TRL 1.17.1's GRPOTrainer: the three methods the masked subclass overrides, in TRL's order and
    shapes, on lists. _generate returns TRL's 9-tuple of the fixture's completions with vLLM's masked log-probs;
    _generate_and_score_completions pads as TRL does, runs the old pass, the sequence_mask IS ratio and the reference
    pass with the adapters off; _get_per_token_logps_and_entropies runs masked.emulate_trl_logps (TRL's support path)
    in TRL's row chunks; loss_passes shuffles, splits and runs the loss pass with grad."""

    def __init__(self, *, grad, grammar, completions=COMPLETIONS, batch_size=2, clip_max=3.0, trl_mask=None,
                 ref_adapters_off=True, old_pass=True):
        self.grad, self.grammar, self.completions = grad, grammar, completions
        self.batch_size, self.clip_max, self.trl_mask = batch_size, clip_max, trl_mask
        self.ref_adapters_off, self.old_pass = ref_adapters_off, old_pass
        self.state = SimpleNamespace(global_step=0)
        self.model = PeftModel()
        self.temperature = 1.0
        self.accelerator = SimpleNamespace(unwrap_model=lambda m: m, autocast=contextlib.nullcontext)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.calls = []

    def _generate(self, prompts):
        prompt_ids = [list(PROMPTS[i // 2]) for i in range(len(self.completions))]
        comps = [list(c) for c in self.completions]
        logprobs = [vllm_logps(p, c, self.grammar) for p, c in zip(prompt_ids, comps)]
        return prompt_ids, comps, None, ["text"] * len(comps), logprobs, copy.deepcopy(self.trl_mask), {}, None, []

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None,
                                           compute_entropy=False, compute_aux_loss=False, pixel_values=None,
                                           sampling_mask=None):
        self.calls.append({"rows": len(input_ids), "support": copy.deepcopy(sampling_mask), "grad": self.grad.on})
        step = batch_size or len(input_ids)
        logps, ents = [], []
        for s in range(0, len(input_ids), step):                         # TRL's row chunks (:1508)
            ids, att = input_ids[s:s + step], attention_mask[s:s + step]
            sm = None if sampling_mask is None else sampling_mask[s:s + step]
            logps += M.emulate_trl_logps(ids, att, logits_to_keep, sm, logits_at)[0]
            if compute_entropy:
                ents += M.emulate_logits_form(ids, att, logits_to_keep, [[[-1]] * logits_to_keep] * len(ids),
                                              logits_at)[2]
        return logps, (ents if compute_entropy else None), None

    def _generate_and_score_completions(self, inputs):
        prompt_ids, comps, _, _, sampling, trl_mask, _, _, _ = self._generate([x["prompt"] for x in inputs])
        p_len, width = max(map(len, prompt_ids)), max(map(len, comps))
        prompt = [[PAD] * (p_len - len(p)) + p for p in prompt_ids]
        pmask = [[0] * (p_len - len(p)) + [1] * len(p) for p in prompt_ids]
        comp = [c + [PAD] * (width - len(c)) for c in comps]
        cmask = [[1] * len(c) + [0] * (width - len(c)) if c[-1] in STOPS else [0] * width for c in comps]
        ids = [p + c for p, c in zip(prompt, comp)]
        att = [p + m for p, m in zip(pmask, cmask)]
        samp = [lp + [0.0] * (width - len(lp)) for lp in sampling]
        smask = None
        if trl_mask is not None:                                          # TRL pads its own mask with -1 (:2585-2592)
            k = max(len(c) for row in trl_mask for c in row)
            smask = [[list(c) + [-1] * (k - len(c)) for c in row] + [[-1] * k] * (width - len(row)) for row in trl_mask]
        out = {"prompt_ids": prompt, "prompt_mask": pmask, "completion_ids": comp, "completion_mask": cmask,
               "sampling_per_token_logps": samp}
        with self.grad.off():
            if self.old_pass:
                old = self._get_per_token_logps_and_entropies(self.model, ids, att, width, batch_size=self.batch_size,
                                                              sampling_mask=smask)[0]
                ratio = []
                for o, s, m in zip(old, samp, cmask):                     # sequence_mask (:2779-2822)
                    w = math.exp(sum((a - b) * k for a, b, k in zip(o, s, m)))
                    ratio.append([w if w <= self.clip_max else 0.0])
                out.update(old_per_token_logps=old, importance_sampling_ratio=ratio)
            ctx = self.model.disable_adapter() if self.ref_adapters_off else contextlib.nullcontext()
            with ctx:
                out["ref_per_token_logps"] = self._get_per_token_logps_and_entropies(
                    self.model, ids, att, width, batch_size=self.batch_size, sampling_mask=smask)[0]
        if smask is not None:
            out["sampling_mask"] = smask
        return out

    def loss_passes(self, out, micro=2, seed=5):
        """_prepare_inputs' shuffle and split (:1612-1613), then _compute_loss's pass (:3151-3168) per micro-batch."""
        order = list(range(len(out["completion_ids"])))
        random.Random(seed).shuffle(order)
        width = len(out["completion_ids"][0])
        res = []
        for k in range(0, len(order), micro):
            rows = order[k:k + micro]
            ids = [out["prompt_ids"][r] + out["completion_ids"][r] for r in rows]
            att = [out["prompt_mask"][r] + out["completion_mask"][r] for r in rows]
            sm = [out["sampling_mask"][r] for r in rows] if "sampling_mask" in out else None
            lp, ent, _ = self._get_per_token_logps_and_entropies(self.model, ids, att, width, compute_entropy=True,
                                                                 sampling_mask=sm)
            res.append((rows, lp, ent))
        self.state.global_step += 1
        return res


def make_setup(mode="grammar_masked", nfa=True, quirk=None):
    return M.MaskSetup(mode=mode, regex=REGEX, regex_sha256=SHA, vocab_size=V, stop_ids=STOPS, replay=ToyGrammar(),
                       nfa=ToyGrammar(quirk=quirk) if nfa else None, stats={})


def trainer_for(setup, **kw):
    grad = GradMode()
    cls = M.make_trainer_class(FakeTRLTrainer, ops=ListOps(grad))
    return cls(grad=grad, grammar=setup.replay, mask_setup=setup, **kw)


def run_step(setup, micro=2, **kw):
    tr = trainer_for(setup, **kw)
    out = tr._generate_and_score_completions(INPUTS)
    return tr, out, tr.loss_passes(out, micro=micro)


def expected_neglogz():
    vals = []
    for i, comp in enumerate(COMPLETIONS):
        if comp[-1] in STOPS:
            for t in range(len(comp)):
                z = toy_logits(PROMPTS[i // 2] + comp[:t])
                vals.append(lse(z) - lse([z[k] for k in ToyGrammar().allowed_after(comp[:t])]))
    return sum(vals) / len(vals), len(vals)


def test_the_fixture_follows_the_toy_grammar():
    g = ToyGrammar()
    walks = [g.walk(c) for c in COMPLETIONS]
    assert all(w.violation is None for w in walks)
    assert [w.terminated for w in walks] == [v > 0 for v in VALID]


def test_a_masked_step_hands_every_pass_the_grammars_support_and_records_it():
    setup = make_setup()
    tr, out, losses = run_step(setup)
    g = ToyGrammar()
    width = len(out["completion_ids"][0])
    want = [[list(g.allowed_after(comp[:c])) if c < VALID[b] else [] for c in range(width)]
            for b, comp in enumerate(COMPLETIONS)]
    # 1. the calls TRL's method saw: old (masked), old (TRL's own mask: None), ref (masked), 4 loss micro-batches
    assert [(c["rows"], c["grad"], c["support"] is not None) for c in tr.calls] == (
        [(8, False, True), (8, False, False), (8, False, True)] + [(2, True, True)] * 4)
    for call in (tr.calls[0], tr.calls[2]):
        got = [[[x for x in s if x >= 0] for s in row] for row in call["support"]]
        assert got == want and len(call["support"][0][0]) == 4          # K = the largest set
    # 2. the old pass is vLLM's masked distribution: its log-probs equal the sampling ones, so every weight is 1
    for b in range(8):
        for c in range(VALID[b]):
            assert abs(out["old_per_token_logps"][b][c] - out["sampling_per_token_logps"][b][c]) < 1e-9
    for rows, lp, _ in losses:                                          # the loss pass scores the same rows alike
        for r, row in zip(rows, lp):
            assert all(abs(row[c] - out["old_per_token_logps"][r][c]) < 1e-12 for c in range(VALID[r]))
    rec = setup.stats[0]
    assert all(abs(w - 1.0) < 1e-9 for w in rec["seq_weights"] + rec["seq_weights_raw"])
    # 3. the record
    neg, n = expected_neglogz()
    assert rec["rows"] == {"old": 8, "ref": 8, "loss": 8} and rec["rows_scored"] == {"old": 6, "ref": 6, "loss": 6}
    assert (rec["completions"], rec["positions"], rec["walked"], rec["duplicates"], rec["terminated"]) == (
        8, 41, 37, 1, 6)
    assert (rec["violations"], rec["nfa_mismatch"], rec["nfa_positions"], rec["cache_miss"]) == (0, 0, 37, 0)
    assert (rec["tokens"], rec["forced"], rec["forced_share"], rec["k_max"]) == (25, 2, 0.08, 4)
    assert rec["neglogz_tokens"] == n == 25 and abs(rec["neglogz"] - neg) < 1e-9 and rec["neglogz"] > 0
    assert rec["trl_sampling_mask"] is None and rec["path"] == "support_ids" and rec["generations"] == 1
    assert rec["seq_valid"] == [v > 0 for v in VALID] and rec["is_level"] == "sequence"
    assert rec["nonfinite"] == {"old": 0, "ref": 0, "loss": 0}
    sc = rec["selfcheck"]
    assert (sc["rows"], sc["ran"], sc["against"], sc["error"]) == ([0, 1], "support_ids", "fill", None)
    assert sc["masked"]["positions"] == 8 and sc["masked"]["max_abs_diff"] < 1e-9
    assert sc["unmasked"]["max_abs_diff"] < 1e-9 and tr._lofi_mask.ops.fills == [2]
    assert [p["label"] for p in rec["passes"]] == ["old", "ref"] + ["loss"] * 4
    assert [p["adapters"] for p in rec["passes"]] == ["enabled", "disabled"] + ["enabled"] * 4
    assert all(k in rec for k in SPEC_KEYS) and all(v >= 0 for v in rec["timing"].values())
    json.dumps(rec, allow_nan=False)
    # 4. TRL's metrics for the step
    m = tr._metrics["train"]
    assert (m["mask/violations"], m["mask/forced_share"], m["mask/nfa_mismatch"]) == ([0.0], [0.08], [0.0])
    assert m["mask/neglogz"] == [rec["neglogz"]] and abs(m["is/seq_weight_mean"][0] - 1.0) < 1e-9
    assert abs(m["is/seq_weight_max_share"][0] - 1 / 8) < 1e-9


def test_without_the_mask_the_old_pass_is_the_whole_vocabularys_and_the_weights_fall():
    grad = GradMode()
    plain = FakeTRLTrainer(grad=grad, grammar=ToyGrammar())
    out = plain._generate_and_score_completions(INPUTS)
    neg, _ = expected_neglogz()
    raw, logs = M.raw_seq_weights(out["old_per_token_logps"], out["sampling_per_token_logps"], out["completion_mask"])
    assert all(w < 1.0 for w, v in zip(raw, VALID) if v) and all(w == 1.0 for w, v in zip(raw, VALID) if not v)
    assert abs(-sum(logs) / sum(VALID) - neg) < 1e-9, "the gap is the grammar's -log Z"


def test_a_sampled_token_outside_the_grammar_stops_the_launch_before_any_pass():
    for row, comp, example in ((3, [2, 3, 7, 11], {"pos": 2, "token": 7, "terminated": False,
                                                   "allowed": [4, 6, 10, 11]}),
                               (6, [0, 3, 10, 4], {"pos": 3, "token": 4, "terminated": True, "allowed": []})):
        setup = make_setup()
        comps = [list(c) for c in COMPLETIONS]
        comps[row] = comp
        tr = trainer_for(setup, completions=comps)
        with pytest.raises(M.MaskViolation, match="1 of 8 completions"):
            tr._generate_and_score_completions(INPUTS)
        assert tr.calls == [], "no pass saw the rows"
        rec = setup.stats[0]
        assert rec["violations"] == 1 and rec["violation_examples"] == [{"row": row, **example}]


def test_the_reference_pass_with_the_adapters_on_or_a_missing_old_pass_stops_the_launch():
    with pytest.raises(M.MaskError, match="the ref pass ran with the LoRA adapters enabled"):
        run_step(make_setup(), ref_adapters_off=False)
    with pytest.raises(M.MaskError, match="the old pass ran with the LoRA adapters disabled"):
        run_step(make_setup(), old_pass=False)       # the reference pass would be read as the old one


def test_trls_own_sampling_mask_is_anded_into_the_grammars_support():
    g = ToyGrammar()
    cands = []
    for comp in COMPLETIONS:          # vLLM's candidates: the allowed set without its largest unsampled id, plus id 8
        row = []
        for c in range(len(comp)):
            a = list(g.allowed_after(comp[:c]))
            drop = max((x for x in a if x != comp[c]), default=None)
            row.append(sorted({x for x in a if x != drop} | {8}))
        cands.append(row)
    setup = make_setup()
    tr, out, _ = run_step(setup, trl_mask=cands)
    old, own = tr.calls[0]["support"], tr.calls[1]["support"]
    for b, comp in enumerate(COMPLETIONS):
        for c in range(VALID[b]):
            assert [x for x in old[b][c] if x >= 0] == sorted(set(g.allowed_after(comp[:c])) & set(cands[b][c]))
    assert own == out["sampling_mask"], "the unmasked companion pass gets TRL's own mask"
    assert setup.stats[0]["trl_sampling_mask"] == {"completions": 8, "positions": 41, "k_max": 4}
    cands[0][1] = [3, 8]                                                  # drops the sampled token 1 at (0, 1)
    with pytest.raises(M.MaskViolation, match="the old pass: 1 valid tokens outside"):
        run_step(make_setup(), trl_mask=cands)


def test_rows_the_cache_lacks_are_walked_in_the_pass_and_counted():
    setup = make_setup()
    tr, out, _ = run_step(setup)
    tr._lofi_mask.begin(7)                                                # a new generation clears the cache
    ids = [out["prompt_ids"][2] + out["completion_ids"][2]]
    att = [out["prompt_mask"][2] + out["completion_mask"][2]]
    lp, ent, _ = tr._get_per_token_logps_and_entropies(tr.model, ids, att, len(out["completion_ids"][0]),
                                                       compute_entropy=True)
    rec = setup.stats[7]
    assert rec["cache_miss"] == 1 and rec["rows"] == {"old": 0, "ref": 0, "loss": 1}
    assert all(abs(lp[0][c] - out["old_per_token_logps"][2][c]) < 1e-12 for c in range(VALID[2]))


def test_path_fill_runs_the_registered_logits_form_and_checks_itself_against_the_support_path():
    support_setup, fill_setup = make_setup(), make_setup(mode="grammar_masked_fp32")
    s_tr, s_out, s_losses = run_step(support_setup)
    f_tr, f_out, f_losses = run_step(fill_setup)
    assert [(c["rows"], c["support"] is not None) for c in f_tr.calls] == [(2, True), (2, False)], (
        "TRL's method runs only for the self-check")
    assert f_tr._lofi_mask.ops.fills == [8, 8, 2, 2, 2, 2]                # old, ref, then the 4 loss micro-batches
    for key in ("old_per_token_logps", "ref_per_token_logps"):
        assert max(abs(a - b) for ra, rb in zip(f_out[key], s_out[key]) for a, b in zip(ra, rb)) < 1e-9
    for (rows_f, lp_f, ent_f), (rows_s, lp_s, ent_s) in zip(f_losses, s_losses):
        assert rows_f == rows_s
        assert max(abs(a - b) for ra, rb in zip(lp_f + ent_f, lp_s + ent_s) for a, b in zip(ra, rb)) < 1e-9
    rec, ref = fill_setup.stats[0], support_setup.stats[0]
    assert rec["path"] == "fill" and abs(rec["neglogz"] - ref["neglogz"]) < 1e-9
    sc = rec["selfcheck"]
    assert (sc["ran"], sc["against"], sc["error"]) == ("fill", "support_ids", None)
    assert sc["masked"]["max_abs_diff"] < 1e-9 and sc["unmasked"]["max_abs_diff"] < 1e-9
    with pytest.raises(M.MaskError, match="pixel_values"):
        f_tr._get_per_token_logps_and_entropies(f_tr.model, [f_out["prompt_ids"][0] + f_out["completion_ids"][0]],
                                                [f_out["prompt_mask"][0] + f_out["completion_mask"][0]],
                                                len(f_out["completion_ids"][0]), pixel_values=[1.0])


def test_an_nfa_that_disagrees_with_xgrammar_is_recorded_and_the_launch_goes_on():
    setup = make_setup(quirk=(1, 9))                  # the NFA also allows id 9 after one token
    tr, _, _ = run_step(setup)
    rec = setup.stats[0]
    assert (rec["nfa_mismatch"], rec["nfa_positions"], len(rec["nfa_examples"])) == (7, 37, 5)
    assert rec["nfa_examples"][0] == {"row": 0, "pos": 1, "token": 1, "n_xgrammar": 2, "n_nfa": 3,
                                      "only_xgrammar": [], "only_nfa": [9]}
    assert tr._metrics["train"]["mask/nfa_mismatch"] == [7.0]
    no_nfa = make_setup(nfa=False)
    tr, _, _ = run_step(no_nfa)
    assert no_nfa.stats[0]["nfa_mismatch"] is None and math.isnan(tr._metrics["train"]["mask/nfa_mismatch"][0])


def test_two_generations_in_one_step_add_up_and_an_eval_generation_keeps_its_own_record():
    setup = make_setup()
    tr = trainer_for(setup)
    tr._generate_and_score_completions(INPUTS)
    tr._generate_and_score_completions(INPUTS)
    rec = setup.stats[0]
    assert rec["generations"] == 2 and rec["rows"]["old"] == 16 and len(rec["seq_weights"]) == 16
    assert rec["tokens"] == rec["neglogz_tokens"] == 50 and rec["selfcheck"]["step"] == 0
    tr.model.training = False
    tr.state.global_step = 3
    tr._generate_and_score_completions(INPUTS)
    assert 3 not in setup.stats and tr._lofi_mask.eval_stats[3]["rows"]["ref"] == 8


def test_the_class_overrides_the_three_methods_and_refuses_what_would_train_unmasked():
    cls = M.make_trainer_class(FakeTRLTrainer, ops=ListOps(GradMode()))
    assert issubclass(cls, FakeTRLTrainer) and cls.__qualname__ == "LofiGRPOTrainer"
    assert all(getattr(cls, m) is not getattr(FakeTRLTrainer, m) for m in M.METHODS)
    with pytest.raises(M.MaskError, match="mask_setup"):
        cls(grad=GradMode(), grammar=ToyGrammar())

    class NoGenerate:
        def _generate_and_score_completions(self, inputs): ...
        def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep,
                                               sampling_mask=None): ...

    class NoSamplingMask(NoGenerate):
        def _generate(self, prompts): ...
        def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep): ...

    with pytest.raises(M.MaskError, match="_generate"):
        M.make_trainer_class(NoGenerate)
    with pytest.raises(M.MaskError, match="sampling_mask"):
        M.make_trainer_class(NoSamplingMask)
    tr = trainer_for(make_setup())
    tr._generate_and_score_completions(INPUTS)
    with pytest.raises(M.MaskError, match="does not fit"):            # sampling_mask given twice
        tr._get_per_token_logps_and_entropies(tr.model, [[1]], [[1]], 1, None, False, False, None, None)


def test_the_setup_refuses_a_mask_built_over_another_grammar_vocabulary_or_stop_ids():
    s = make_setup()
    assert s.path == "support_ids" and s.stop_ids == STOPS and M.path_for("grammar_masked_fp32") == "fill"
    rec = s.record()
    assert (rec["mode"], rec["path"], rec["regex_sha256"], rec["vocab_size"], rec["stop_ids"]) == (
        "grammar_masked", "support_ids", SHA, V, [10, 11])
    base = dict(mode="grammar_masked", regex=REGEX, regex_sha256=SHA, vocab_size=V, stop_ids=(11, 10, 11),
                replay=ToyGrammar(), nfa=ToyGrammar(), stats={})
    assert M.MaskSetup(**base).stop_ids == (10, 11)
    for change, why in ((dict(mode="masked"), "not one of"), (dict(regex_sha256="0" * 64), "not the regex's sha256"),
                        (dict(replay=ToyGrammar(vocab_size=13)), "replay was built with vocab_size 13"),
                        (dict(nfa=ToyGrammar(stops=(10,))), r"nfa was built with stop_ids \(10,\)"),
                        (dict(nfa=ToyGrammar(regex="other")), "nfa was built with regex_sha256"),
                        (dict(replay=object()), "replay has no walk"), (dict(stop_ids=()), "stop ids"),
                        (dict(stop_ids=(12,)), "outside the vocabulary"), (dict(vocab_size=True), "vocab_size"),
                        (dict(selfcheck_rows=-1), "selfcheck_rows")):
        with pytest.raises(M.MaskError, match=why):
            M.MaskSetup(**{**base, **change})


def test_section_as_grammar_mask_is_accepted_as_the_nfa_and_compared_position_by_position():
    from lofi_rl.grammar_mask import GrammarMask
    tokens, regex = {0: "a", 1: "b", 2: "c", 3: "ab", 4: "x"}, "a(b|c)c?"
    replay = GrammarMask.from_tokens(tokens, regex, vocab_size=8, stop_ids=(6, 7))
    nfa = GrammarMask.from_tokens(tokens, regex, vocab_size=8, stop_ids=(7, 6))
    base = dict(mode="grammar_masked", regex=regex, regex_sha256=M.regex_sha256(regex), vocab_size=8, stop_ids=(6, 7),
                replay=replay, stats={})
    setup = M.MaskSetup(nfa=nfa, **base)
    ctl = M.MaskedPasses(setup, ListOps(GradMode()))
    ctl.begin(0)
    ctl.check_generation(([[1], [1]], [[0, 1, 6], [3, 2, 7]], None, [], [], None, {}, None, []))
    rec = setup.stats[0]
    assert (rec["nfa_mismatch"], rec["nfa_positions"], rec["terminated"], rec["violations"]) == (0, 6, 2, 0)
    assert [0, 1, 6] in ctl.cache and [3, 2, 7] in ctl.cache
    with pytest.raises(M.MaskViolation, match="1 of 1 completions"):
        ctl.check_generation(([[1]], [[0, 4]], None, [], [], None, {}, None, []))
    with pytest.raises(M.MaskError, match="vocab_size 9"):
        M.MaskSetup(nfa=GrammarMask.from_tokens(tokens, regex, vocab_size=9, stop_ids=(6, 7)), **base)


def test_a_pass_before_any_generation_opens_the_trainers_step_and_walks_its_rows():
    setup = make_setup()
    tr = trainer_for(setup)
    tr.state.global_step = 2
    ids, att, width, _, _ = layout(width=8)
    lp, _, _ = tr._get_per_token_logps_and_entropies(tr.model, ids[:2], att[:2], width, compute_entropy=True)
    rec = setup.stats[2]
    assert (rec["generations"], rec["cache_miss"], rec["rows"]["loss"]) == (0, 1, 2), "two equal rows: one walk"
    assert rec["trl_sampling_mask"] == M.UNREAD, "no _generate ran for this step"
    with pytest.raises(M.MaskError, match="completion_mask"):
        tr._lofi_mask.after_score({})


def test_the_step_record_has_every_field_the_spec_lists():
    rec = M.new_step_record(4, "support_ids", nfa=True)
    assert all(k in rec for k in SPEC_KEYS) and rec["trl_sampling_mask"] == M.UNREAD
    assert rec["rows"] == {"old": 0, "ref": 0, "loss": 0} and rec["nfa_mismatch"] == 0
    assert M.new_step_record(4, "fill", nfa=False)["nfa_positions"] is None
    json.dumps(rec, allow_nan=False)


def test_the_module_imports_without_torch_trl_vllm_xgrammar_numpy_or_the_nfa():
    blocked = ("torch", "trl", "vllm", "transformers", "peft", "datasets", "accelerate", "xgrammar", "numpy",
               "lofi_rl.grammar_mask")
    code = ("import sys\n" + "".join(f"sys.modules[{m!r}] = None\n" for m in blocked)
            + "import lofi_rl.rl.masked as m\nprint('MASKED_OK', m.TRAINER_LOGPROBS)\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, cwd=ROOT)
    assert p.returncode == 0 and "MASKED_OK ('grammar_masked', 'grammar_masked_fp32')" in p.stdout, p.stderr[-2000:]


# ------------------------------------------------------------------------------------------ 5. the xgrammar replay
WIDE = (0, 1, 2, 31, 32, 33, 40, 63, 64, 95)           # symbols across four bitmask words; stop ids 65 and 99


class FakeBitmask:
    """xgrammar's (1, ceil(V / 32)) int32 bitmask (a torch tensor on the cluster): fill_ and numpy, as the replay
    uses them."""

    def __init__(self, batch, vocab):
        import numpy as np
        self.a = np.full((batch, (vocab + 31) // 32), -1, dtype=np.int32)

    def fill_(self, value):
        self.a.fill(value)
        return self

    def numpy(self):
        return self.a


def fake_xgrammar(grammar, calls, *, decoder_agrees=True):
    """A stand-in xgrammar module over a ToyGrammar, recording the calls the replay makes. Like xgrammar 0.2.8's own
    package (its __init__ never imports _version.py), it has no __version__: the replay reads the distribution's
    metadata. Its TokenizerInfo keeps the tokenizer's one default stop id, and the matcher the override it is given."""
    import numpy as np
    mod = types.ModuleType("xgrammar")

    class TokenizerInfo:
        def __init__(self, vocab_size):
            self.vocab_size, self.vocab_type, self.add_prefix_space = vocab_size, "VocabType.BYTE_LEVEL", False
            self.stop_token_ids, self.special_token_ids = [grammar.stop_ids[0]], [7, 8]

        @staticmethod
        def from_huggingface(tokenizer, *, vocab_size=None, stop_token_ids=None):
            calls.append(("from_huggingface", tokenizer, vocab_size, stop_token_ids))
            return TokenizerInfo(vocab_size)

    class GrammarCompiler:
        def __init__(self, info, *, max_threads=8, cache_enabled=True, cache_limit_bytes=-1):
            calls.append(("GrammarCompiler", max_threads, cache_enabled, cache_limit_bytes))

        def compile_regex(self, regex):
            calls.append(("compile_regex", regex))
            return ("compiled", regex)

    class GrammarMatcher:
        def __init__(self, compiled, *, override_stop_tokens=None, terminate_without_stop_token=False,
                     max_rollback_tokens=-1):
            calls.append(("GrammarMatcher", compiled, override_stop_tokens, max_rollback_tokens))
            self.stops = tuple(override_stop_tokens)
            self.reset()

        @property
        def stop_token_ids(self):
            return list(self.stops)

        def reset(self):
            self.prefix, self.done = [], False

        def fill_next_token_bitmask(self, bm, index=0):
            words = [0] * bm.a.shape[1]
            for t in grammar.allowed_after(self.prefix):
                words[t // 32] |= 1 << (t % 32)
            bm.a[index] = np.array([w - (1 << 32) if w >= 1 << 31 else w for w in words], dtype=np.int32)
            return True

        def accept_token(self, t):
            if self.done or t not in grammar.allowed_after(self.prefix):
                return False
            self.prefix.append(t)
            self.done = t in self.stops
            return True

        def is_terminated(self):
            return self.done

    def allocate_token_bitmask(batch, vocab):
        calls.append(("allocate_token_bitmask", batch, vocab))
        return FakeBitmask(batch, vocab)

    def masked_tokens(bm, vocab, index=0):     # testing._get_masked_tokens_from_bitmask: the refused ids, bit by bit
        row = [int(x) & 0xFFFFFFFF for x in bm.a[index]]
        refused = [i for i in range(vocab) if not (row[i // 32] >> (i % 32)) & 1]
        return refused if decoder_agrees else refused[1:]

    mod.TokenizerInfo, mod.GrammarCompiler, mod.GrammarMatcher = TokenizerInfo, GrammarCompiler, GrammarMatcher
    mod.allocate_token_bitmask = allocate_token_bitmask
    mod.testing = SimpleNamespace(_get_masked_tokens_from_bitmask=masked_tokens)
    return mod


@pytest.fixture
def walk_cls(monkeypatch):
    """lofi_rl.grammar_mask is section A's module, written alongside this one; XgrammarReplay.walk returns its Walk.
    Until the module exists a stand-in with the interface's fields (allowed, violation, terminated) takes its place."""
    if importlib.util.find_spec("lofi_rl.grammar_mask") is not None:
        from lofi_rl.grammar_mask import Walk
        return Walk
    mod = types.ModuleType("lofi_rl.grammar_mask")

    @dataclass(frozen=True)
    class Walk:
        allowed: tuple
        violation: int | None
        terminated: bool

    mod.Walk = Walk
    monkeypatch.setitem(sys.modules, "lofi_rl.grammar_mask", mod)
    return Walk


def _metadata(monkeypatch, xgrammar):
    """importlib.metadata.version with xgrammar's distribution at version `xgrammar` (None: not installed); every other
    name is looked up as before."""
    before = importlib.metadata.version

    def version(name):
        if name != "xgrammar":
            return before(name)
        if xgrammar is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return xgrammar

    monkeypatch.setattr(importlib.metadata, "version", version)


def test_the_replay_is_built_as_vllms_backend_builds_it_and_walks_as_the_grammar_does(monkeypatch, walk_cls):
    g = ToyGrammar(symbols=WIDE, stops=(65, 99), vocab_size=100, regex="wide")
    calls = []
    monkeypatch.setitem(sys.modules, "xgrammar", fake_xgrammar(g, calls))
    _metadata(monkeypatch, "0.2.8 (metadata stand-in)")
    tok = object()
    r = M.XgrammarReplay("wide", tok, vocab_size=100, stop_ids=[99, 65, 65])
    assert calls == [("from_huggingface", tok, 100, None), ("GrammarCompiler", 8, True, -1), ("compile_regex", "wide"),
                     ("GrammarMatcher", ("compiled", "wide"), [65, 99], 0), ("allocate_token_bitmask", 1, 100)]
    assert (r.vocab_size, r.stop_ids, r.regex_sha256) == (100, (65, 99), M.regex_sha256("wide"))
    assert r.record["decoder_check"] == {"position": 0, "allowed": 3, "agree": True}
    assert r.record["tokenizer_info"]["special_token_ids"] == 2
    assert r.record["xgrammar"] == "0.2.8 (metadata stand-in)", "the installed distribution's metadata gives it"
    assert (r.record["tokenizer_info"]["stop_token_ids"], r.record["matcher_stop_token_ids"]) == ([65], [65, 99]), \
        "the tokenizer's default stop id beside the override the matcher (and vLLM's) stops on"
    good = [0, 1, 32, 65]                    # indices 0, 1, 4 and a stop id
    w = r.walk(good)
    assert isinstance(w, walk_cls)
    assert (w.allowed, w.violation, w.terminated) == (((0, 1, 2), (1, 31), (2, 32, 65, 99), (33, 63, 65, 99)),
                                                      None, True)
    for ids in (good, [0, 2], [0, 1, 32, 65, 33], [2, 33, 40], [1, 2, 32]):    # refused: the 2nd, 3rd and 5th
        mine, theirs = r.walk(ids), g.walk(ids)
        assert (mine.allowed, mine.violation, mine.terminated) == (theirs.allowed, theirs.violation, theirs.terminated)
    assert r.walk(good) == w, "the matcher is reset for every walk"
    assert r.record["walks"] == 7 and r.record["refused"] == 3 and r.record["accept_refused"] == 0
    setup = M.MaskSetup(mode="grammar_masked", regex="wide", regex_sha256=M.regex_sha256("wide"), vocab_size=100,
                        stop_ids=(65, 99), replay=r, nfa=g, stats={})
    assert setup.record()["replay"]["walks"] == 7


def test_a_bitmask_decoder_that_disagrees_with_xgrammars_own_stops_the_build(monkeypatch, walk_cls):
    g = ToyGrammar(symbols=WIDE, stops=(65, 99), vocab_size=100)
    monkeypatch.setitem(sys.modules, "xgrammar", fake_xgrammar(g, [], decoder_agrees=False))
    with pytest.raises(M.MaskError, match="decoder"):
        M.XgrammarReplay("wide", object(), vocab_size=100, stop_ids=[65, 99])
    monkeypatch.setitem(sys.modules, "xgrammar", fake_xgrammar(g, []))
    with pytest.raises(M.MaskError, match="no stop ids"):
        M.XgrammarReplay("wide", object(), vocab_size=100, stop_ids=[])


def test_xgrammars_version_comes_from_its_metadata_and_the_matchers_stop_ids_are_a_record(monkeypatch, walk_cls):
    g = ToyGrammar(symbols=WIDE, stops=(65, 99), vocab_size=100)
    mod = fake_xgrammar(g, [])
    assert not hasattr(mod, "__version__"), "as xgrammar 0.2.8's package binds none"
    monkeypatch.setitem(sys.modules, "xgrammar", mod)
    build = lambda: M.XgrammarReplay("wide", object(), vocab_size=100, stop_ids=[65, 99])      # noqa: E731
    _metadata(monkeypatch, None)                 # no distribution metadata: the module's own __version__, if it has one
    assert build().record["xgrammar"] is None
    mod.__version__ = "0.2.8+local"
    assert build().record["xgrammar"] == "0.2.8+local"
    _metadata(monkeypatch, "0.2.8")
    assert build().record["xgrammar"] == "0.2.8", "the installed distribution's metadata comes first"

    def unreadable(self):
        raise RuntimeError("no binding")

    monkeypatch.setattr(mod.GrammarMatcher, "stop_token_ids", property(unreadable))
    r = build()
    assert r.record["matcher_stop_token_ids"] == "unavailable: RuntimeError"
    assert r.walk([0, 1, 32, 65]).terminated, "a record that cannot be read never stops the replay"
    json.dumps(r.record, allow_nan=False)
