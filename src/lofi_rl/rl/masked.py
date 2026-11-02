"""The grammar-masked GRPO trainer (docs/DECISIONS.md D-021 Part A; docs/GRPO.md §8).

**Why.** vLLM samples every completion under the LoopScript grammar: xgrammar's bitmask sets the logits outside the
allowed set to -inf before the sampler runs (vllm v1/worker/gpu_model_runner.py:4335, then `_sample` at :4340), so the
sampling log-probs TRL reads are normalized over the allowed set. TRL's trainer normalized its own over the whole
vocabulary, and D-020's IS gap was that difference (D-021, Why). Here the trainer's log-probs become those of the
masked distribution in all three passes (old, reference, loss), so GRPO is on-policy for the policy that is played.

**What TRL 1.17.1 does** (its installed sources, read 2026-11-02; the build spec's §1):
- `_get_per_token_logps_and_entropies` never forms logits. It runs the model with `fused_lm_head=True`
  (trl/trainer/grpo_trainer.py:1566-1567). Given a `sampling_mask` (B, C, K) of token ids padded with -1, it writes it
  into the completion columns of a (B, L, K) `support_ids` (:1559-1560). The fused forward shifts that with the labels
  (trl/trainer/utils.py:1655-1658), and the chunked kernel normalizes each label's log-prob over its row's support
  (trl/kernels/chunked_logprob.py:593-597). "The entropy stays that of the whole vocabulary" (:483, :598).
- The old, reference and loss passes all call that method with `sampling_mask=` (grpo_trainer.py:2762-2770,
  :2846-2854, :3151-3168). TRL's own sampling mask is None unless top_p < 1, top_k > 0 or min_p > 0 (:1152-1161).
So the grammar goes in as that sampling mask, ANDed with TRL's own when there is one: path `support_ids`, mode
`grammar_masked` (D-021: "If so, the grammar mask is ANDed into it instead"). D-021's registered logits form is kept as
path `fill`, mode `grammar_masked_fp32` (D-021's named fix for a measured miss, unused until an amendment says so): the
override runs the model's ordinary forward itself (`logits_to_keep + 1` rows, the last dropped), converts to fp32,
divides by the temperature, writes FILL outside the allowed set out of place, and takes the log-softmax at the label.

**Path fill is a check, not a change of numerics.** D-021 names that fix "the masked log-softmax computed in fp32".
Under TRL 1.17.1 both paths already compute it: each normalizes bf16 logits in fp32 (vLLM's sampler, too, takes its
log-probs from logits cast to fp32, v1/sample/sampler.py:97).
- `support_ids`: TRL calls the model under `self.accelerator.autocast()` (grpo_trainer.py:1566-1567). The kernel's
  logit tile has the autocast dtype, else the hidden states' (trl/kernels/chunked_logprob.py:510-512, :539, :544,
  :548): bf16 either way, as the configs set `bf16: true` and train.py loads the model in bf16. Each tile is upcast to
  fp32 before it is folded (:160 Triton, :353 torch), and the statistics and the log-probs are fp32 (:522-532,
  :592-597). `cast_lm_head_to_fp32`, which would run the head itself in fp32 (trainer/utils.py:1670-1672), defaults
  to False (grpo_config.py:462-463, passed at grpo_trainer.py:1118), and no config here sets it.
- `fill`: Qwen3's forward applies lm_head to the bf16 hidden states with no upcast
  (transformers models/qwen3/modeling_qwen3.py:480-482), so these logits are bf16 too; TorchOps.fill_logps converts
  them to fp32 only after.
So the fill form is an independent re-implementation of the same arithmetic: the two paths' log-probs differ only by
rounding (the batches' and GEMMs' shapes, the form and order of the fp32 operations). It is the self-check's other side
(3), and it can move the log-probs beyond rounding only where path `support_ids` computes them wrongly on the GPU.

1. **The mask is xgrammar's own** (XgrammarReplay): a GrammarMatcher compiled from the same regex, with its
   TokenizerInfo, compiler and stop tokens built the way vLLM 0.32's backend builds each request's
   (v1/structured_output/backend_xgrammar.py:69-78, :129-131, :157-161; v1/structured_output/__init__.py:144, :199-200).
   It is replayed along each completion's sampled ids, so the trainer applies the bitmask vLLM applied. At each position
   it fills the bitmask, decodes the set bits of the nonzero words to sorted ids (bits_to_ids), checks the sampled
   token, and accepts it. The decoder is checked once against xgrammar's own (testing.py:211) when the replay is built.
2. **The independent check** is lofi_rl.grammar_mask.GrammarMask (D-021's NFA), walked along the same ids. Its allowed
   set is compared with xgrammar's at every position; a difference is recorded (D-021 M8 gates on it), never applied.
3. **The passes** (make_trainer_class; MaskedPasses holds the logic):
   - `_generate`: TRL's own, then every completion is replayed, truncated ones too. The allowed sets are cached by the
     completion's ids, and TRL's own sampling mask is recorded (None at the registered settings). A sampled token
     outside the mask raises MaskViolation there, before any pass sees the rows (D-021 Part A).
   - `_generate_and_score_completions`: opens the step's record (keyed by the global step it generates at), runs TRL's,
     then records TRL's applied per-sequence IS weights, the raw exp(sum((old - sampling) * mask)) and the step's
     scalars as TRL metrics.
   - `_get_per_token_logps_and_entropies` (TRL's signature passed through): labels the pass, checks the label against
     PEFT's adapter state, and builds the support from the cache by each row's valid ids. Every valid token must be in
     its support. The support then goes to TRL (path support_ids) or to the fill form (path fill). The old pass also
     takes the unmasked log-probs for -log Z, and the launch's first old pass runs the other path on a few rows beside
     the one that ran: a self-check that is recorded and never stops the launch.
   The pass label is "loss" with grad enabled. Without grad, the first call of a generation is "old" (:2762) and the
   second "ref" (:2846; a later one is an evaluation loss pass). The old and loss passes must run with the adapters
   enabled and "ref" with them disabled (TRL's reference under PEFT is the adapter-off base:
   use_adapter(model, None) is model.disable_adapter(), trl/trainer/utils.py:1459-1460). A mismatch is a MaskError.
4. **The alignment.** support[b][c] is the set completion token c was drawn from, the layout TRL's own sampling mask
   has (grpo_trainer.py:2585-2592). emulate_trl_logps replays TRL's placement, labels, shift and slice stage by stage
   on plain lists, and the tests show it puts that set at token c's label position. emulate_logits_form is the fill
   form on the same lists. The two agree, and every off-by-one variant of a stage does not.
5. **The records** (MaskSetup.stats[step], which train.py's StepLog writes as each step's "mask"; new_step_record
   lists every field): rows and valid rows per pass, positions walked, violations, NFA mismatches with up to
   MAX_EXAMPLES examples, the old pass's token-weighted -log Z, forced share and largest support, cache misses,
   non-finite masked log-probs, TRL's sampling mask, the 32 sequence weights (applied and raw), the path, the
   self-check and the timings.
6. **Laptop and cluster.** Only the standard library is imported here. torch, xgrammar and numpy are imported inside
   TorchOps and XgrammarReplay, which run only on the cluster. lofi_rl.grammar_mask (section A's module) is imported
   inside XgrammarReplay.walk, so this module imports whether or not it exists. MaskedPasses works on plain lists
   through an `ops` adapter (TorchOps on the cluster), so the laptop tests drive the subclass's control flow with a
   list-backed stand-in (tests/test_rl_masked_logps.py).
"""
from __future__ import annotations

import bisect
import hashlib
import importlib.metadata
import inspect
import math
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

TRAINER_LOGPROBS = ("grammar_masked", "grammar_masked_fp32")     # config values (lofi_rl.rl.config imports them)
PATHS = MappingProxyType({"grammar_masked": "support_ids", "grammar_masked_fp32": "fill"})   # mode -> its path
PASSES = ("old", "ref", "loss")
ADAPTERS = {"old": "enabled", "ref": "disabled", "loss": "enabled"}    # the PEFT state each pass must run under
METHODS = ("_generate", "_generate_and_score_completions", "_get_per_token_logps_and_entropies")
LEAD = ("self", "model", "input_ids", "attention_mask", "logits_to_keep")    # TRL's first parameters (:1482-1488)
FILL = -1.0e9            # path fill: exp(FILL - max) is 0 in fp32, and FILL stays finite (no inf - inf, no NaN)
MAX_EXAMPLES = 5         # violation and NFA-mismatch examples kept per step
UNREAD = "unread"        # trl_sampling_mask before _generate's hook has read TRL's own mask for the step
FP32_MAX = 3.4028234663852886e38    # torch.nan_to_num's stand-in for +-inf in fp32 (TRL's IS difference, :2782)
INTERN_LIMIT = 100_000   # interned allowed sets kept across steps before the table is dropped (a cap, not a count)
# the arguments the fill form reads; any other set to something but None/False (images, MoE aux loss) is refused there
FILL_ARGS = frozenset(LEAD + ("batch_size", "compute_entropy", "sampling_mask"))


class MaskError(RuntimeError):
    """The masked trainer cannot apply the mask as registered (its setup, the pass order, the adapter state, a shape):
    the launch stops rather than train without the mask (fail closed)."""


class MaskViolation(MaskError):
    """A sampled token outside the grammar's allowed set, a token after the grammar terminated, or a completion mask
    that is not ones then zeros: the launch stops before any pass sees the rows (D-021 Part A)."""


def regex_sha256(regex: str) -> str:
    """sha256 of the regex text (UTF-8), as lofi_rl.grammar.regex_sha256 names a grammar."""
    return hashlib.sha256(regex.encode("utf-8")).hexdigest()


def path_for(mode: str) -> str:
    """The log-prob path a trainer_logprobs mode runs: `support_ids` or `fill`."""
    if mode not in PATHS:
        raise MaskError(f"trainer_logprobs {mode!r} is not one of {list(TRAINER_LOGPROBS)}")
    return PATHS[mode]


def _norm_stops(ids: Iterable[Any]) -> tuple[int, ...]:
    return tuple(sorted({int(s) for s in ids}))


def _record_of(obj: Any) -> Any:
    r = getattr(obj, "record", None)
    try:
        return r() if callable(r) else r
    except Exception as err:  # noqa: BLE001: a record of a record
        return f"unavailable: {type(err).__name__}: {err}"[:300]


# ------------------------------------------------------------------------------------------ the setup
@dataclass
class MaskSetup:
    """What the masked trainer needs, built by train.py before the model loads. `replay` and `nfa` must have been
    built over this regex, vocabulary size and stop ids (D-021 M8: "over one vocabulary and one set of stop ids"); each
    exposes regex_sha256, vocab_size and stop_ids, and walk(ids) -> a lofi_rl.grammar_mask.Walk-like object
    (allowed, violation, terminated). `stats` is filled per global step and shared with StepLog."""

    mode: str
    regex: str
    regex_sha256: str
    vocab_size: int
    stop_ids: tuple[int, ...]
    replay: Any                      # .walk(ids) -> Walk-like (XgrammarReplay on the cluster; a stand-in in tests)
    nfa: Any | None                  # lofi_rl.grammar_mask.GrammarMask (the independent check), or None
    stats: dict[int, dict[str, Any]]
    selfcheck_rows: int = 2

    def __post_init__(self) -> None:
        path_for(self.mode)
        if not isinstance(self.regex, str) or not self.regex:
            raise MaskError("MaskSetup: the regex is empty")
        sha = regex_sha256(self.regex)
        if self.regex_sha256 != sha:
            raise MaskError(f"MaskSetup: regex_sha256 {self.regex_sha256!r} is not the regex's sha256 {sha}")
        if isinstance(self.vocab_size, bool) or not isinstance(self.vocab_size, int) or self.vocab_size < 1:
            raise MaskError(f"MaskSetup: vocab_size {self.vocab_size!r} is not a positive integer")
        stops = _norm_stops(self.stop_ids)
        if not stops or not all(0 <= s < self.vocab_size for s in stops):
            raise MaskError(f"MaskSetup: stop ids {list(stops)} are empty or outside the vocabulary")
        self.stop_ids = stops
        if not isinstance(self.stats, dict):
            raise MaskError("MaskSetup: stats must be a dict (keyed by global step)")
        if isinstance(self.selfcheck_rows, bool) or not isinstance(self.selfcheck_rows, int) or self.selfcheck_rows < 0:
            raise MaskError(f"MaskSetup: selfcheck_rows {self.selfcheck_rows!r} is not a count")
        for name, obj in (("replay", self.replay), ("nfa", self.nfa)):
            if obj is None and name == "nfa":
                continue
            if not callable(getattr(obj, "walk", None)):
                raise MaskError(f"MaskSetup: the {name} has no walk(ids)")
            for attr, want in (("regex_sha256", sha), ("vocab_size", self.vocab_size), ("stop_ids", stops)):
                got = getattr(obj, attr, None)
                if attr == "stop_ids" and got is not None:
                    got = _norm_stops(got)
                if got != want:
                    raise MaskError(f"MaskSetup: the {name} was built with {attr} {got!r}, not the run's {want!r} "
                                    "(D-021 M8: one vocabulary and one set of stop ids)")

    @property
    def path(self) -> str:
        return PATHS[self.mode]

    def record(self) -> dict[str, Any]:
        """What run.json's `masked` can carry of the setup."""
        return {"mode": self.mode, "path": self.path, "regex_sha256": self.regex_sha256,
                "vocab_size": self.vocab_size, "stop_ids": list(self.stop_ids), "selfcheck_rows": self.selfcheck_rows,
                "replay": _record_of(self.replay), "nfa": None if self.nfa is None else _record_of(self.nfa)}


# ------------------------------------------------------------------------------------------ pure helpers: the support
def valid_length(row: Sequence[Any]) -> int:
    """The valid completion tokens of one completion-mask row. TRL's completion_mask is ones over each completion's
    ids, right-padded with zeros (grpo_trainer.py:2559, :2566), and all zeros for a truncated completion under
    mask_truncated_completions (:2604-2608). Any other row (a one after a zero, a value other than 0 or 1) is refused:
    the support is built from the valid prefix, so a hole would put a token's allowed set at another token."""
    n, ended = 0, False
    for i, v in enumerate(row):
        if v == 1:
            if ended:
                raise MaskViolation(f"completion mask {list(row)[:i + 1]}...: a 1 at column {i} after a 0 (TRL's "
                                    "completion mask is ones then zeros)")
            n += 1
        elif v == 0:
            ended = True
        else:
            raise MaskViolation(f"completion mask value {v!r} at column {i} is not 0 or 1")
    return n


class AllowedCache:
    """Each completion's allowed sets, keyed by its ids, as the replay walked them in _generate. A missing key is
    walked now and counted (`misses`). Allowed tuples are interned, so a step's cache holds each distinct set once.
    The cache is cleared at each generation: TRL scores a generation's rows only until the next one."""

    def __init__(self, replay: Any) -> None:
        self.replay = replay
        self._walks: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}
        self._intern: dict[tuple[int, ...], tuple[int, ...]] = {}
        self.misses = 0

    def __len__(self) -> int:
        return len(self._walks)

    def __contains__(self, ids: Any) -> bool:
        return tuple(ids) in self._walks

    def clear(self) -> None:
        self._walks.clear()
        if len(self._intern) > INTERN_LIMIT:
            self._intern.clear()

    def intern(self, allowed: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
        out = []
        for a in allowed:
            t = a if isinstance(a, tuple) else tuple(a)
            out.append(self._intern.setdefault(t, t))
        return tuple(out)

    def put(self, ids: Sequence[int], allowed: Iterable[Iterable[int]]) -> None:
        self._walks[tuple(int(t) for t in ids)] = self.intern(allowed)

    def get(self, ids: Sequence[int]) -> tuple[tuple[tuple[int, ...], ...], int | None]:
        """(allowed sets, the first violating position or None). A miss walks the replay and is stored only when the
        walk covers every id without a violation."""
        key = tuple(int(t) for t in ids)
        hit = self._walks.get(key)
        if hit is not None:
            return hit, None
        self.misses += 1
        walk = self.replay.walk(list(key))
        allowed = self.intern(walk.allowed)
        bad = walk.violation
        if bad is None and len(allowed) != len(key):
            bad = min(len(allowed), len(key))       # a walk that does not cover the ids is refused
        if bad is None:
            self._walks[key] = allowed
        return allowed, bad


@dataclass(frozen=True)
class SupportBuild:
    """One pass's support and its counts (build_support)."""

    support: list[list[list[int]]]       # (B, C, K) ids, -1 padded: support[b][c] is the set token c was drawn from
    valid: tuple[int, ...]               # each row's valid tokens
    positions: int                       # valid positions
    forced: int                          # valid positions whose support is one token
    k_max: int                           # the largest support (K is max(1, k_max))
    rows: int
    rows_scored: int                     # rows with at least one valid token
    violations: tuple[tuple[int, int, int], ...]     # (row, position, token): a valid token outside its support
    cache_miss: int                      # rows whose allowed sets were walked now


def build_support(rows: Sequence[Sequence[int]], valid: Sequence[int], cache: AllowedCache,
                  candidates: Sequence[Sequence[Sequence[int]]] | None = None) -> SupportBuild:
    """The (B, C, K) support of one pass. Row b's first valid[b] ids are looked up in the cache. support[b][c] is
    allowed[c], the set completion token c was sampled from, ANDed with TRL's own candidate set at (b, c) when
    `candidates` (TRL's sampling_mask, -1 padded) is given. Columns past the valid prefix are all -1, as TRL pads its
    own (grpo_trainer.py:2585-2592). A valid token outside its support is a violation; a row's first one ends that
    row, whose remaining columns are -1. The caller raises."""
    misses = cache.misses
    width = len(rows[0]) if rows else 0
    sets: list[list[tuple[int, ...]]] = []
    violations: list[tuple[int, int, int]] = []
    positions = forced = k_max = scored = 0
    for b, (row, n) in enumerate(zip(rows, valid, strict=True)):
        if len(row) != width:
            raise MaskError(f"row {b} has {len(row)} completion columns, not {width}")
        if not 0 <= n <= width:
            raise MaskError(f"row {b}: {n} valid tokens in {width} columns")
        mine: list[tuple[int, ...]] = []
        sets.append(mine)
        if n == 0:
            continue
        scored += 1
        positions += n
        ids = tuple(int(t) for t in row[:n])
        allowed, bad = cache.get(ids)               # bad: the walk's own refusal (a miss walked now), if any
        for c in range(n):
            s = allowed[c] if c < len(allowed) else ()
            if candidates is not None:
                keep = {int(t) for t in candidates[b][c] if t >= 0}
                s = tuple(t for t in s if t in keep)
            if c == bad or ids[c] not in s:
                violations.append((b, c, ids[c]))
                break
            mine.append(s)
            forced += len(s) == 1
            k_max = max(k_max, len(s))
    k = max(1, k_max)
    support = [[list(s) + [-1] * (k - len(s)) for s in mine] + [[-1] * k for _ in range(width - len(mine))]
               for mine in sets]
    return SupportBuild(support=support, valid=tuple(int(n) for n in valid), positions=positions, forced=forced,
                        k_max=k_max, rows=len(rows), rows_scored=scored, violations=tuple(violations),
                        cache_miss=cache.misses - misses)


def _contains(sorted_ids: Sequence[int], token: int) -> bool:
    """`token in sorted_ids` for a sorted id tuple (a bitmask row decodes to one)."""
    i = bisect.bisect_left(sorted_ids, token)
    return i < len(sorted_ids) and sorted_ids[i] == token


def bits_to_ids(words: Iterable[tuple[int, int]], vocab_size: int) -> tuple[int, ...]:
    """The allowed ids of one xgrammar bitmask row from its nonzero words, as (word index, int32 value) pairs in
    increasing index order: bit j of word w set means token 32 * w + j is allowed (xgrammar/matcher.py:15-21, int32
    words, ceil(vocab / 32) of them). Ids at or past vocab_size are dropped. Sorted."""
    out: list[int] = []
    for w, v in words:
        v = int(v) & 0xFFFFFFFF
        base = 32 * int(w)
        while v:
            low = v & -v
            i = base + low.bit_length() - 1
            if i < vocab_size:
                out.append(i)
            v ^= low
    return tuple(out)


def decode_bitmask(row: Sequence[int], vocab_size: int) -> tuple[int, ...]:
    """bits_to_ids over a whole bitmask row (every word)."""
    return bits_to_ids(((w, v) for w, v in enumerate(row) if v), vocab_size)


def compare_walks(a: Any, b: Any, ids: Sequence[int] = ()) -> tuple[int, list[dict[str, Any]]]:
    """(positions compared, differences) between two walks of the same ids: `a` xgrammar's replay, `b` the NFA. A
    difference is a position both walks reached whose allowed sets differ (with what only one of them allows), or one
    entry when the walks end differently (length, violation or termination)."""
    n = min(len(a.allowed), len(b.allowed))
    diffs: list[dict[str, Any]] = []
    for p in range(n):
        x, y = a.allowed[p], b.allowed[p]
        if x != y and set(x) != set(y):
            sx, sy = set(x), set(y)
            diffs.append({"pos": p, "token": ids[p] if p < len(ids) else None, "n_xgrammar": len(sx), "n_nfa": len(sy),
                          "only_xgrammar": sorted(sx - sy)[:10], "only_nfa": sorted(sy - sx)[:10]})
    ends = ([len(a.allowed), a.violation, bool(a.terminated)], [len(b.allowed), b.violation, bool(b.terminated)])
    if ends[0] != ends[1]:
        diffs.append({"pos": n, "end": {"xgrammar": ends[0], "nfa": ends[1]}})
    return n, diffs


def summarize_trl_mask(mask: Any) -> dict[str, Any] | None:
    """What the step record keeps of TRL's own sampling mask from _generate (a list per completion of candidate lists;
    None at top_p 1, top_k 0, min_p unset)."""
    if mask is None:
        return None
    if hasattr(mask, "tolist"):
        mask = mask.tolist()
    sizes = [len([t for t in cands if t >= 0]) for comp in mask for cands in comp]
    return {"completions": len(mask), "positions": len(sizes), "k_max": max(sizes, default=0)}


# ------------------------------------------------------------------------------------------ pure helpers: numbers
def neglogz(masked: Sequence[Sequence[float]], unmasked: Sequence[Sequence[float]],
            valid: Sequence[int]) -> dict[str, Any]:
    """The token-weighted -log Z over the valid positions: masked - unmasked log-prob of each sampled token is -log of
    the allowed set's mass under the unmasked distribution (>= 0). Non-finite differences are counted, not averaged."""
    total, n, bad = 0.0, 0, 0
    for b, k in enumerate(valid):
        for c in range(k):
            d = float(masked[b][c]) - float(unmasked[b][c])
            if math.isfinite(d):
                total += d
                n += 1
            else:
                bad += 1
    return {"sum": total, "tokens": n, "mean": total / n if n else None, "nonfinite": bad}


def raw_seq_weights(old: Sequence[Sequence[float]], sampling: Sequence[Sequence[float]],
                    mask: Sequence[Sequence[Any]]) -> tuple[list[float], list[float]]:
    """(weights, their logs): exp(sum_t (old - sampling) * mask) per sequence, as TRL forms its sequence-level ratio
    before the clip (grpo_trainer.py:2779-2791). A NaN product counts 0 and +-inf the fp32 extremes (torch.nan_to_num,
    :2782); a weight past fp32's range is inf, as fp32's exp would make it."""
    weights, logs = [], []
    for o_row, s_row, m_row in zip(old, sampling, mask, strict=True):
        s = 0.0
        for o, x, m in zip(o_row, s_row, m_row, strict=True):
            d = (float(o) - float(x)) * float(m)
            if d != d:
                d = 0.0
            elif d in (math.inf, -math.inf):
                d = math.copysign(FP32_MAX, d)
            s += d
        w = math.exp(s) if s < 709.0 else math.inf
        weights.append(math.inf if w > FP32_MAX else w)
        logs.append(s)
    return weights, logs


def seq_weight_stats(weights: Sequence[float] | None, keep: Sequence[bool] | None = None) -> dict[str, Any]:
    """n, sum, mean, max and max / sum of the per-sequence weights (only those with keep[i], when given): D-021 M11
    reads the mean (>= 0.3) and the largest's share of the sum (<= 0.25)."""
    w = [float(x) for i, x in enumerate(weights or ()) if keep is None or keep[i]]
    n = len(w)
    total = sum(w) if n else 0.0
    top = max(w) if n else None
    share = top / total if n and math.isfinite(total) and total > 0 else None
    return {"n": n, "sum": total, "mean": total / n if n else None, "max": top, "max_share": share}


def diff_stats(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], valid: Sequence[int]) -> dict[str, Any]:
    """Mean and max |a - b| over the valid positions (the self-check)."""
    d, bad = [], 0
    for i, k in enumerate(valid):
        for c in range(k):
            x = abs(float(a[i][c]) - float(b[i][c]))
            if math.isfinite(x):
                d.append(x)
            else:
                bad += 1
    return {"positions": len(d), "mean_abs_diff": sum(d) / len(d) if d else None,
            "max_abs_diff": max(d) if d else None, "nonfinite": bad}


def _nonfinite(rows: Sequence[Sequence[float]], valid: Sequence[int]) -> int:
    return sum(not math.isfinite(float(rows[b][c])) for b, k in enumerate(valid) for c in range(k))


# ------------------------------------------------------------------------------------------ reference log-probs
def _softplus(x: float) -> float:
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def support_logp(logits: Sequence[float], target: int, support: Iterable[int] | None = None,
                 temperature: float = 1.0) -> float:
    """TRL's chunked kernel on one row, by its own formula (trl/kernels/chunked_logprob.py:595-597), in float64 where
    the kernel's statistics are fp32. The support is the given ids without the -1 padding, or the whole vocabulary when
    None; z = logit * (1 / T), as the kernel scales (`inv_t`, :516; :101 Triton, :315 torch). Then
    - M = the largest z over the whole support, the target included when it is in it: the kernel's others_max (:596),
      which is support_max (:179 Triton, :371 torch), or running_max without a support (:170, :361);
    - S = the sum of e^(z - M) over the support's other tokens (all but the target): support_other_sum_exp (:183,
      :375), or other_sum_exp without a support (:174-175, :366-367);
    - the log-prob is -softplus((M - z_target) + log S) (:597).
    A forced token (no other token: S = 0, log S = -inf, -softplus(-inf) = 0) is exactly 0, as is a target whose others
    all underflow. A target outside the support gets log(e^z_t / (e^z_t + sum_support e^z)), a finite value: the kernel
    cannot tell, so build_support checks."""
    inv_t = 1.0 / temperature
    z = [float(x) * inv_t for x in logits]
    ids = range(len(z)) if support is None else sorted({int(s) for s in support if s >= 0})
    others = [z[i] for i in ids if i != target]
    if not others:
        return 0.0
    m = max(z[i] for i in ids)
    s = sum(math.exp(o - m) for o in others)
    if s == 0.0:                       # log S = -inf (math.log refuses 0): -softplus(-inf) is 0, as in the kernel
        return 0.0
    return -_softplus((m - z[target]) + math.log(s))


def fill_logp(logits: Sequence[float], target: int, support: Iterable[int] | None = None,
              temperature: float = 1.0, fill: float = FILL) -> float:
    """D-021's registered form on one row: the logits / T, `fill` outside the support (out of place; None: no fill),
    the log-softmax, the target's entry. Finite whatever the support."""
    z = [float(x) / temperature for x in logits]
    if support is not None:
        keep = {int(s) for s in support if s >= 0}
        z = [x if i in keep else fill for i, x in enumerate(z)]
    m = max(z)
    return z[target] - (m + math.log(sum(math.exp(x - m) for x in z)))


def ref_entropy(logits: Sequence[float], temperature: float = 1.0) -> float:
    """The whole vocabulary's entropy of softmax(logits / T), the one TRL's kernel returns (chunked_logprob.py:598)."""
    z = [float(x) / temperature for x in logits]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return m + math.log(s) - sum(ei * zi for ei, zi in zip(e, z)) / s


# ------------------------------------------------------------------------------------------ TRL's alignment, emulated
def trl_support_ids(seq_len: int, logits_to_keep: int, sampling_mask_row: Sequence[Sequence[int]],
                    k: int) -> list[list[int]]:
    """grpo_trainer.py:1559-1560 for one row: a (L, K) grid of -1 whose last logits_to_keep rows are the row's (C, K)
    sampling mask (`support_ids[:, -logits_to_keep:] = sampling_mask[start:end]`)."""
    if len(sampling_mask_row) != logits_to_keep:
        raise MaskError(f"a sampling-mask row has {len(sampling_mask_row)} columns, not logits_to_keep "
                        f"{logits_to_keep}")
    grid = [[-1] * k for _ in range(seq_len)]
    grid[seq_len - logits_to_keep:] = [list(r) + [-1] * (k - len(r)) for r in sampling_mask_row]
    return grid


def trl_labels(ids: Sequence[int], attention: Sequence[Any], logits_to_keep: int) -> list[int]:
    """grpo_trainer.py:1564-1565 for one row: the ids, -100 where attention is 0 and on every column before the last
    logits_to_keep."""
    labels = [int(t) if a else -100 for t, a in zip(ids, attention, strict=True)]
    cut = len(labels) - logits_to_keep
    labels[:cut] = [-100] * cut
    return labels


def trl_fused_positions(labels: Sequence[int],
                        support_ids: Sequence[Sequence[int]] | None = None) -> list[tuple[int, int, Any]]:
    """trl/trainer/utils.py:1655-1668 for one row: hidden[:, :-1], labels[:, 1:], support_ids[:, 1:]. Returns, for every
    hidden position j (0 .. L-2), (j, its label, its support): hidden state j is scored on input token j + 1, against
    support_ids[j + 1]. A label of -100 is not scored (the kernel sees only `labels != -100`, :1668)."""
    lab = list(labels[1:])
    sup = None if support_ids is None else list(support_ids[1:])
    return [(j, lab[j], None if sup is None else sup[j]) for j in range(len(lab))]


def trl_keep_last(row: Sequence[Any], logits_to_keep: int) -> list[Any]:
    """grpo_trainer.py:1568 for one row: `outputs.log_probs[:, -logits_to_keep:]` of the (L - 1)-long fused output."""
    return list(row[len(row) - logits_to_keep:])


def emulate_trl_logps(input_ids: Sequence[Sequence[int]], attention_mask: Sequence[Sequence[Any]],
                      logits_to_keep: int, sampling_mask: Sequence[Sequence[Sequence[int]]] | None,
                      logits_at: Callable[[Sequence[int], int], Sequence[float]], temperature: float = 1.0, *,
                      place: Callable[..., list[list[int]]] = trl_support_ids,
                      labels: Callable[..., list[int]] = trl_labels,
                      fused: Callable[..., list[tuple[int, int, Any]]] = trl_fused_positions,
                      keep_last: Callable[..., list[Any]] = trl_keep_last) -> tuple[list[list[float]], list[list[Any]]]:
    """TRL 1.17.1's support path on plain lists, one stage per function above (each cites its source lines), with a
    stand-in model: logits_at(row ids, j) is the logit row hidden state j yields, predicting input token j + 1. The
    stages are parameters so that a test can swap one for an off-by-one variant. Returns (log_probs, trace):
    log_probs[b][c] as TRL returns it (0 where the label is -100, utils.py:1529-1530), and trace[b][c] = (the input
    position scored, its label, its support) or None."""
    out, trace = [], []
    c_keep = int(logits_to_keep)
    for b, (ids, att) in enumerate(zip(input_ids, attention_mask, strict=True)):
        if not 0 < c_keep < len(ids):
            raise MaskError(f"row {b}: logits_to_keep {c_keep} with {len(ids)} columns (a prompt needs one)")
        grid = None
        if sampling_mask is not None:
            k = max((len(r) for r in sampling_mask[b]), default=0)
            grid = place(len(ids), c_keep, sampling_mask[b], k)
        row, tr = [], []
        for j, label, sup in fused(labels(ids, att, c_keep), grid):
            if label == -100:
                row.append(0.0)
                tr.append(None)
                continue
            s = None if sup is None else [int(x) for x in sup if x >= 0]
            row.append(support_logp(logits_at(ids, j), label, s, temperature))
            tr.append((j + 1, label, s))
        out.append(keep_last(row, c_keep))
        trace.append(keep_last(tr, c_keep))
    return out, trace


def emulate_logits_form(input_ids: Sequence[Sequence[int]], attention_mask: Sequence[Sequence[Any]],
                        logits_to_keep: int, support: Sequence[Sequence[Sequence[int]]],
                        logits_at: Callable[[Sequence[int], int], Sequence[float]], temperature: float = 1.0,
                        fill: float = FILL) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """D-021's registered logits form (path fill) on plain lists: the ordinary forward's last logits_to_keep + 1 rows
    with the last dropped, so row c is the hidden state before completion token c (Qwen3's forward slices
    hidden_states[:, -logits_to_keep:], transformers models/qwen3/modeling_qwen3.py:481-482); then the temperature, the
    fill outside support[b][c], the log-softmax and the label's entry. 0 where the completion mask is 0. Returns
    (masked, unmasked, entropy), each (B, C); the entropy is the whole vocabulary's, as TRL's kernel keeps it."""
    masked, unmasked, ent = [], [], []
    c_keep = int(logits_to_keep)
    for b, (ids, att) in enumerate(zip(input_ids, attention_mask, strict=True)):
        n = len(ids)
        if not 0 < c_keep < n:
            raise MaskError(f"row {b}: logits_to_keep {c_keep} with {n} columns (a prompt needs one)")
        rows = [logits_at(ids, j) for j in range(n - c_keep - 1, n)][:-1]
        lab = ids[n - c_keep:]
        on = [bool(a) for a in att[n - c_keep:]]
        masked.append([fill_logp(rows[c], lab[c], support[b][c], temperature, fill) if on[c] else 0.0
                       for c in range(c_keep)])
        unmasked.append([fill_logp(rows[c], lab[c], None, temperature) if on[c] else 0.0 for c in range(c_keep)])
        ent.append([ref_entropy(rows[c], temperature) if on[c] else 0.0 for c in range(c_keep)])
    return masked, unmasked, ent


# ------------------------------------------------------------------------------------------ passes and adapters
def pass_label(grad_enabled: bool, nograd_before: int) -> str:
    """Which pass a call is: with grad, the loss pass (_compute_loss, :3151); without, the first no-grad call of a
    generation is the old pass (:2762) and the second the reference pass (:2846). A later no-grad call is a loss pass
    under no_grad (prediction_step, :3416-3420, in evaluation)."""
    if grad_enabled:
        return "loss"
    return PASSES[nograd_before] if nograd_before < 2 else "loss"


def check_adapters(label: str, state: str) -> None:
    """The old and loss passes run with the LoRA adapters enabled, the reference pass with them disabled (TRL's
    reference under PEFT is the adapter-off base, grpo_trainer.py:2845; use_adapter(model, None) is
    model.disable_adapter(), trl/trainer/utils.py:1459-1460). Anything else stops the launch."""
    want = ADAPTERS[label]
    if state != want:
        raise MaskError(f"the {label} pass ran with the LoRA adapters {state}, not {want}: the pass order or the "
                        "adapter state is not the one the masked trainer was written for")


def adapter_state_of(model: Any) -> str:
    """"enabled", "disabled", "irregular" or "none": PEFT's adapter state, read from every module whose class defines
    PEFT's `disable_adapters` property (BaseTunerLayer, peft/tuners/tuners_utils.py:1973-1976, and
    AuxiliaryTrainingWrapper, peft/utils/other.py:362-365; transformers' method of that name is not a property) and from
    PeftModel's own `_adapters_disabled` (set by disable_adapter(), peft/peft_model.py:1069-1077). A model whose layers
    disagree, with each other or with that flag, is irregular."""
    flags: set[bool] = set()
    modules = getattr(model, "modules", None)
    for mod in (modules() if callable(modules) else ()):
        if isinstance(getattr(type(mod), "disable_adapters", None), property):
            flags.add(bool(mod.disable_adapters))
    own = getattr(model, "_adapters_disabled", None)
    if not flags:
        return "none"
    if len(flags) > 1 or (isinstance(own, bool) and {own} != flags):
        return "irregular"
    return "disabled" if True in flags else "enabled"


def new_step_record(step: int, path: str, nfa: bool) -> dict[str, Any]:
    """A step's empty record (module docstring, 5). `positions` counts the completion tokens _generate's replay checked
    (every completion, truncated and repeated ones too: the denominator of `violations`); `tokens` the old pass's valid
    positions (the denominator of `neglogz` and `forced_share`)."""
    return {"step": int(step), "path": path, "generations": 0,
            "completions": 0, "terminated": 0, "duplicates": 0, "positions": 0, "walked": 0,
            "violations": 0, "violation_examples": [],
            "nfa_mismatch": 0 if nfa else None, "nfa_positions": 0 if nfa else None, "nfa_examples": [],
            "trl_sampling_mask": UNREAD,
            "rows": {p: 0 for p in PASSES}, "rows_scored": {p: 0 for p in PASSES},
            "tokens": 0, "forced": 0, "forced_share": None, "k_max": 0, "cache_miss": 0,
            "nonfinite": {p: 0 for p in PASSES},
            "neglogz": None, "neglogz_tokens": 0, "neglogz_nonfinite": 0,
            "seq_weights": None, "seq_weights_raw": None, "seq_log_weights_raw": None, "seq_valid": None,
            "seq_weight_stats": None, "is_level": None,
            "selfcheck": None, "passes": [],
            "timing": {"walk_s": 0.0, "nfa_s": 0.0, "old_s": 0.0, "ref_s": 0.0, "loss_s": 0.0, "unmasked_s": 0.0,
                       "selfcheck_s": 0.0}}


def _add_time(rec: dict[str, Any], key: str, seconds: float) -> None:
    rec["timing"][key] = round(rec["timing"][key] + seconds, 4)


def _nan(x: Any) -> float:
    return math.nan if x is None else float(x)


# ------------------------------------------------------------------------------------------ the passes' logic
class MaskedPasses:
    """What the masked trainer does around TRL's calls (module docstring, 3), on plain lists through `ops`: TorchOps on
    the cluster, a list-backed stand-in in the tests. `ops` provides grad_enabled(), adapter_state(trainer, model),
    completion_rows(input_ids, attention_mask, C) -> (ids rows, mask rows), ints(t), floats(t),
    long_tensor(nested, like), take_rows(t, rows) and fill_logps(...) (TorchOps documents each)."""

    def __init__(self, setup: MaskSetup, ops: Any, clock: Callable[[], float] = time.perf_counter) -> None:
        self.setup, self.ops, self.clock = setup, ops, clock
        self.path = setup.path
        self.cache = AllowedCache(setup.replay)
        self.eval_stats: dict[int, dict[str, Any]] = {}
        self.step: int | None = None
        self.mode = "train"
        self.nograd = 0
        self.selfcheck_done = False

    # ---------------------------------------------------------------- records
    def _book(self) -> dict[int, dict[str, Any]]:
        return self.setup.stats if self.mode == "train" else self.eval_stats

    def current(self, trainer: Any = None) -> dict[str, Any]:
        """The record passes write to: the open step's, or (a pass before any generation of this launch) the
        trainer's global step's, opened without counting a generation."""
        if self.step is None:
            self.step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
        book = self._book()
        if self.step not in book:
            book[self.step] = new_step_record(self.step, self.path, self.setup.nfa is not None)
        return book[self.step]

    def begin(self, step: int, mode: str = "train") -> dict[str, Any]:
        """_generate_and_score_completions starts: open (or reopen) the step's record, reset the pass counter, and drop
        the last generation's allowed sets (TRL never scores its rows again)."""
        self.step, self.mode, self.nograd = int(step), mode, 0
        self.cache.clear()
        rec = self.current()
        rec["generations"] += 1
        return rec

    # ---------------------------------------------------------------- _generate
    def check_generation(self, result: Any, trainer: Any = None) -> None:
        """_generate's output (grpo_trainer.py:2416-2427; completion_ids are the sampled ids as vLLM returned them):
        replay every completion, compare the NFA, cache the allowed sets, record TRL's own sampling mask, and raise
        MaskViolation on any sampled token outside the mask."""
        if not isinstance(result, tuple) or len(result) != 9:
            raise MaskError(f"GRPOTrainer._generate returned {type(result).__name__} of "
                            f"{len(result) if isinstance(result, tuple) else '?'} items, not TRL 1.17.1's 9")
        completion_ids, trl_mask = result[1], result[5]
        rec = self.current(trainer)
        t0 = self.clock()
        rec["trl_sampling_mask"] = summarize_trl_mask(trl_mask)
        nfa = self.setup.nfa
        seen: dict[tuple[int, ...], tuple[bool, dict[str, Any] | None]] = {}
        bad: list[dict[str, Any]] = []
        for i, raw in enumerate(completion_ids):
            ids = tuple(int(t) for t in raw)
            rec["completions"] += 1
            rec["positions"] += len(ids)
            if ids in seen:                                 # an identical completion: the same walk
                rec["duplicates"] += 1
                terminated, why = seen[ids]
            else:
                walk = self.setup.replay.walk(list(ids))
                rec["walked"] += len(ids)
                terminated, v = bool(walk.terminated), walk.violation
                if v is None and len(walk.allowed) != len(ids):
                    v = min(len(walk.allowed), len(ids))
                why = None
                if v is not None:
                    why = {"pos": v, "token": ids[v] if v < len(ids) else None, "terminated": terminated,
                           "allowed": list(walk.allowed[v])[:12] if v < len(walk.allowed) else []}
                else:
                    self.cache.put(ids, walk.allowed)
                seen[ids] = (terminated, why)
                if nfa is not None:
                    t1 = self.clock()
                    compared, diffs = compare_walks(walk, nfa.walk(list(ids)), ids)
                    rec["nfa_positions"] += compared
                    rec["nfa_mismatch"] += len(diffs)
                    room = MAX_EXAMPLES - len(rec["nfa_examples"])
                    rec["nfa_examples"] += [{"row": i, **d} for d in diffs[:max(room, 0)]]
                    _add_time(rec, "nfa_s", self.clock() - t1)
            rec["terminated"] += terminated
            if why is not None:
                bad.append({"row": i, **why})
        _add_time(rec, "walk_s", self.clock() - t0)
        if bad:
            rec["violations"] += len(bad)
            rec["violation_examples"] = (rec["violation_examples"] + bad)[:MAX_EXAMPLES]
            raise MaskViolation(f"{len(bad)} of {len(completion_ids)} completions hold a sampled token outside the "
                                f"grammar's allowed set (xgrammar replay, step {rec['step']}): {bad[:3]}")

    # ---------------------------------------------------------------- _generate_and_score_completions
    def after_score(self, out: Mapping[str, Any]) -> dict[str, float]:
        """TRL's output dict (grpo_trainer.py:3017-3035): record the applied per-sequence weights
        (`importance_sampling_ratio`, (B, 1) under the sequence modes), the raw ones, and which rows have a valid
        token; return the step's scalars for TRL's metrics (NaN where there is no value: TRL's log drops NaN, :3433)."""
        rec = self.current()
        if "completion_mask" not in out:
            raise MaskError("TRL's _generate_and_score_completions returned no completion_mask")
        cm = self.ops.ints(out["completion_mask"])
        valid = [any(bool(x) for x in row) for row in cm]
        ratio = out.get("importance_sampling_ratio")
        seq = None
        if ratio is not None:
            r = self.ops.floats(ratio)
            if all(len(row) == 1 for row in r):
                seq, rec["is_level"] = [row[0] for row in r], "sequence"
            else:
                rec["is_level"] = "token"
        old, samp = out.get("old_per_token_logps"), out.get("sampling_per_token_logps")
        raw = logs = None
        if old is not None and samp is not None:
            raw, logs = raw_seq_weights(self.ops.floats(old), self.ops.floats(samp), cm)
        for key, vals in (("seq_weights", seq), ("seq_weights_raw", raw), ("seq_log_weights_raw", logs),
                          ("seq_valid", valid)):
            if vals is not None:
                rec[key] = (rec[key] or []) + list(vals)
        stats = {"all": seq_weight_stats(rec["seq_weights"]),
                 "valid": seq_weight_stats(rec["seq_weights"], rec["seq_valid"]) if rec["seq_weights"] else None}
        rec["seq_weight_stats"] = stats
        return {"mask/violations": float(rec["violations"]), "mask/neglogz": _nan(rec["neglogz"]),
                "mask/forced_share": _nan(rec["forced_share"]), "mask/nfa_mismatch": _nan(rec["nfa_mismatch"]),
                "is/seq_weight_mean": _nan(stats["all"]["mean"]),
                "is/seq_weight_max_share": _nan(stats["all"]["max_share"])}

    # ---------------------------------------------------------------- _get_per_token_logps_and_entropies
    def logps(self, trainer: Any, base_fn: Callable[..., Any], signature: inspect.Signature, model: Any,
              input_ids: Any, attention_mask: Any, logits_to_keep: Any, args: tuple[Any, ...],
              kwargs: dict[str, Any], trl_mask: Any) -> Any:
        """One pass, masked (module docstring, 3). Returns what TRL's method returns: (logps, entropies, aux_loss)."""
        t0 = self.clock()
        try:
            bound = signature.bind(trainer, model, input_ids, attention_mask, logits_to_keep, *args,
                                   sampling_mask=trl_mask, **kwargs)
        except TypeError as err:
            raise MaskError(f"a call that does not fit TRL's _get_per_token_logps_and_entropies{signature}: {err}") \
                from err
        bound.apply_defaults()
        arguments = bound.arguments
        rec = self.current(trainer)
        grad = bool(self.ops.grad_enabled())
        label = pass_label(grad, self.nograd)
        if not grad:
            self.nograd += 1
        state = self.ops.adapter_state(trainer, model)
        check_adapters(label, state)

        # 1. the support, from each row's valid ids (the completion columns: input_ids[:, -C:])
        width = int(logits_to_keep)
        ids_rows, mask_rows = self.ops.completion_rows(input_ids, attention_mask, width)
        valid = [valid_length(r) for r in mask_rows]
        cands = None if trl_mask is None else self.ops.ints(trl_mask)
        build = build_support(ids_rows, valid, self.cache, cands)
        if build.violations:
            rec["violations"] += len(build.violations)
            ex = [{"pass": label, "row": b, "pos": c, "token": t} for b, c, t in build.violations]
            rec["violation_examples"] = (rec["violation_examples"] + ex)[:MAX_EXAMPLES]
            raise MaskViolation(f"the {label} pass: {len(build.violations)} valid tokens outside their allowed set "
                                f"(step {rec['step']}): {ex[:3]}")
        support = self.ops.long_tensor(build.support, like=input_ids)

        # 2. the masked log-probs, by the mode's path; the old pass also takes the unmasked ones (-log Z)
        unmasked = None
        if self.path == "support_ids":
            result = base_fn(model, input_ids, attention_mask, logits_to_keep, *args, sampling_mask=support, **kwargs)
            if label == "old":
                t1 = self.clock()
                unmasked = base_fn(model, input_ids, attention_mask, logits_to_keep, *args, sampling_mask=trl_mask,
                                   **kwargs)[0]
                _add_time(rec, "unmasked_s", self.clock() - t1)
        else:
            refused = [k for k, v in arguments.items() if k not in FILL_ARGS and v is not None and v is not False
                       and not (isinstance(v, dict) and not v)]
            if refused:
                raise MaskError(f"path fill computes text-only log-probs; this call also sets {refused}")
            logps_t, ent_t, unmasked = self.ops.fill_logps(
                trainer, model, input_ids, attention_mask, width, support, batch_size=arguments.get("batch_size"),
                compute_entropy=bool(arguments.get("compute_entropy")), want_unmasked=label == "old")
            result = (logps_t, ent_t, None)
        masked_rows = self.ops.floats(result[0])
        bad = _nonfinite(masked_rows, valid)

        # 3. the old pass: -log Z, and at the launch's first one the self-check
        if label == "old":
            unmasked_rows = self.ops.floats(unmasked)
            nz = neglogz(masked_rows, unmasked_rows, valid)
            n0, n1 = rec["neglogz_tokens"], nz["tokens"]
            if n1:
                rec["neglogz"] = ((rec["neglogz"] or 0.0) * n0 + nz["sum"]) / (n0 + n1)
            rec["neglogz_tokens"] = n0 + n1
            rec["neglogz_nonfinite"] += nz["nonfinite"]
            if not self.selfcheck_done:
                self._selfcheck(rec, trainer, base_fn, model, input_ids, attention_mask, logits_to_keep, args, kwargs,
                                trl_mask, build, masked_rows, unmasked_rows)

        # 4. the pass's record
        seconds = self.clock() - t0
        rec["rows"][label] += build.rows
        rec["rows_scored"][label] += build.rows_scored
        rec["cache_miss"] += build.cache_miss
        rec["k_max"] = max(rec["k_max"], build.k_max)
        rec["nonfinite"][label] += bad
        if label == "old":
            rec["tokens"] += build.positions
            rec["forced"] += build.forced
            rec["forced_share"] = rec["forced"] / rec["tokens"] if rec["tokens"] else None
        _add_time(rec, f"{label}_s", seconds)
        rec["passes"].append({"label": label, "grad": grad, "adapters": state, "rows": build.rows,
                              "rows_scored": build.rows_scored, "positions": build.positions, "forced": build.forced,
                              "k_max": build.k_max, "cache_miss": build.cache_miss, "nonfinite": bad,
                              "seconds": round(seconds, 4)})
        return result

    def _selfcheck(self, rec: dict[str, Any], trainer: Any, base_fn: Callable[..., Any], model: Any, input_ids: Any,
                   attention_mask: Any, logits_to_keep: Any, args: tuple[Any, ...], kwargs: dict[str, Any],
                   trl_mask: Any, build: SupportBuild, masked_rows: list[list[float]],
                   unmasked_rows: list[list[float]]) -> None:
        """The other path on the first selfcheck_rows rows that have valid tokens, beside the path that ran: mean and
        max |difference| over their valid positions, masked and unmasked. A diagnostic: recorded, never raised (a
        launch whose first old pass has no valid row tries again at the next one)."""
        rows = [b for b, n in enumerate(build.valid) if n][:self.setup.selfcheck_rows]
        if not rows:
            return
        self.selfcheck_done = True
        t0 = self.clock()
        other = "fill" if self.path == "support_ids" else "support_ids"
        out: dict[str, Any] = {"step": rec["step"], "rows": rows, "ran": self.path, "against": other,
                               "masked": None, "unmasked": None, "error": None}
        try:
            ops = self.ops
            sub_ids, sub_att = ops.take_rows(input_ids, rows), ops.take_rows(attention_mask, rows)
            sub_support = ops.long_tensor([build.support[b] for b in rows], like=input_ids)
            width = int(logits_to_keep)
            if other == "fill":
                lp, _, um = ops.fill_logps(trainer, model, sub_ids, sub_att, width, sub_support, batch_size=None,
                                           compute_entropy=False, want_unmasked=True)
            else:
                sub_trl = None if trl_mask is None else ops.take_rows(trl_mask, rows)
                lp = base_fn(model, sub_ids, sub_att, logits_to_keep, *args, sampling_mask=sub_support, **kwargs)[0]
                um = base_fn(model, sub_ids, sub_att, logits_to_keep, *args, sampling_mask=sub_trl, **kwargs)[0]
            sub_valid = [build.valid[b] for b in rows]
            out["masked"] = diff_stats([masked_rows[b] for b in rows], ops.floats(lp), sub_valid)
            out["unmasked"] = diff_stats([unmasked_rows[b] for b in rows], ops.floats(um), sub_valid)
        except Exception as err:  # noqa: BLE001: a diagnostic; the mask itself was applied above
            out["error"] = f"{type(err).__name__}: {err}"[:500]
        seconds = self.clock() - t0
        out["seconds"] = round(seconds, 4)
        _add_time(rec, "selfcheck_s", seconds)
        rec["selfcheck"] = out


# ------------------------------------------------------------------------------------------ the trainer class
def make_trainer_class(base: type, *, ops: Any = None) -> type:
    """LofiGRPOTrainer: `base` (TRL's GRPOTrainer, as train.py imports it) with the three methods of the module
    docstring (3) overridden; build it with mask_setup=MaskSetup(...). `ops` replaces TorchOps (the tests' stand-in).
    A base without those methods, or whose _get_per_token_logps_and_entropies does not take TRL 1.17.1's leading
    parameters and `sampling_mask`, is refused: the mask would not reach its passes."""
    missing = [m for m in METHODS if not callable(getattr(base, m, None))]
    if missing:
        raise MaskError(f"{getattr(base, '__name__', base)} has no {missing}: the masked trainer overrides them")
    signature = inspect.signature(base._get_per_token_logps_and_entropies)
    names = list(signature.parameters)
    if tuple(names[:len(LEAD)]) != LEAD or "sampling_mask" not in names:
        raise MaskError(f"{base.__name__}._get_per_token_logps_and_entropies{signature} does not start with {LEAD} "
                        "and take sampling_mask, as TRL 1.17.1's does (grpo_trainer.py:1482-1501)")

    class LofiGRPOTrainer(base):  # type: ignore[misc, valid-type]
        """TRL's GRPOTrainer with the grammar's mask in its old, reference and loss passes (lofi_rl.rl.masked)."""

        def __init__(self, *a: Any, mask_setup: MaskSetup | None = None, **kw: Any) -> None:
            if not isinstance(mask_setup, MaskSetup):
                raise MaskError("LofiGRPOTrainer needs mask_setup=MaskSetup(...): without it nothing is masked")
            self._lofi_mask = MaskedPasses(mask_setup, ops if ops is not None else TorchOps())
            super().__init__(*a, **kw)

        def _generate(self, prompts: Any) -> Any:
            result = super()._generate(prompts)
            self._lofi_mask.check_generation(result, self)
            return result

        def _generate_and_score_completions(self, inputs: Any) -> Any:
            mode = "train" if self.model.training else "eval"
            self._lofi_mask.begin(int(self.state.global_step), mode)
            out = super()._generate_and_score_completions(inputs)
            for name, value in self._lofi_mask.after_score(out).items():
                self._metrics[mode][name].append(value)
            return out

        def _get_per_token_logps_and_entropies(self, model: Any, input_ids: Any, attention_mask: Any,
                                               logits_to_keep: Any, *args: Any, sampling_mask: Any = None,
                                               **kwargs: Any) -> Any:
            return self._lofi_mask.logps(self, super()._get_per_token_logps_and_entropies, signature, model,
                                         input_ids, attention_mask, logits_to_keep, args, kwargs, sampling_mask)

    LofiGRPOTrainer.__module__ = __name__
    LofiGRPOTrainer.__qualname__ = "LofiGRPOTrainer"
    return LofiGRPOTrainer


# ------------------------------------------------------------------------------------------ cluster only: torch
class TorchOps:
    """The tensor operations MaskedPasses needs (cluster only: torch is imported here). Each is small and explicit;
    everything else in this module works on plain lists."""

    def __init__(self) -> None:
        import torch
        self.torch = torch

    def grad_enabled(self) -> bool:
        return bool(self.torch.is_grad_enabled())

    def adapter_state(self, trainer: Any, model: Any) -> str:
        unwrap = getattr(getattr(trainer, "accelerator", None), "unwrap_model", None)
        return adapter_state_of(unwrap(model) if callable(unwrap) else model)

    def completion_rows(self, input_ids: Any, attention_mask: Any, logits_to_keep: int) -> tuple[list, list]:
        """The completion columns (the last logits_to_keep) of the ids and the attention mask, as lists."""
        start = input_ids.size(1) - int(logits_to_keep)
        return input_ids[:, start:].tolist(), attention_mask[:, start:].tolist()

    def ints(self, t: Any) -> Any:
        return None if t is None else t.detach().cpu().tolist()

    def floats(self, t: Any) -> Any:
        return None if t is None else t.detach().float().cpu().tolist()

    def long_tensor(self, nested: Any, like: Any) -> Any:
        """The (B, C, K) support as a long tensor on the ids' device, the dtype TRL's own sampling mask has."""
        return self.torch.tensor(nested, dtype=self.torch.long, device=like.device)

    def take_rows(self, t: Any, rows: Sequence[int]) -> Any:
        return None if t is None else t[list(rows)]

    def fill_logps(self, trainer: Any, model: Any, input_ids: Any, attention_mask: Any, logits_to_keep: int,
                   support: Any, *, batch_size: int | None, compute_entropy: bool, want_unmasked: bool,
                   fill: float = FILL) -> tuple[Any, Any, Any]:
        """Path fill, D-021's registered form: (masked log-probs, entropies or None, unmasked log-probs or None), each
        (B, C), 0 where the completion mask is 0, as TRL returns its own. With grad when the caller has it."""
        torch = self.torch

        def cat(xs: list[Any]) -> Any:
            return torch.cat(xs, dim=0) if xs else None

        c = int(logits_to_keep)
        n = input_ids.size(0)
        step = batch_size or n
        out_m, out_e, out_u = [], [], []
        for start in range(0, n, step):                          # 1. TRL's row chunks, in order (:1508)
            end = min(start + step, n)
            ids, att = input_ids[start:end], attention_mask[start:end]
            with trainer.accelerator.autocast():                 # 2. the ordinary forward, c + 1 rows
                logits = model(input_ids=ids, attention_mask=att, logits_to_keep=c + 1).logits
            # 3. drop the last row (row j predicts completion token j); fp32 before the temperature, as vLLM's sampler
            #    does (v1/sample/sampler.py:97, then :279)
            z = logits[:, :-1, :].float() / trainer.temperature
            labels = ids[:, ids.size(1) - c:].unsqueeze(-1)
            sup = support[start:end].to(z.device)
            keep = torch.zeros(z.shape, dtype=torch.int8, device=z.device)      # 4. the allowed set as an indicator,
            keep.scatter_add_(-1, sup.clamp(min=0), (sup >= 0).to(torch.int8))  #    built as TRL's _support_mask is
            masked = z.masked_fill(keep == 0, fill)                             # 5. the finite fill, out of place
            on = att[:, att.size(1) - c:].bool()                                # 6. 0 where TRL's labels are -100
            out_m.append(masked.log_softmax(-1).gather(-1, labels).squeeze(-1).masked_fill(~on, 0.0))
            if compute_entropy or want_unmasked:
                full = z.log_softmax(-1)                                        # the whole vocabulary, as TRL keeps it
                if want_unmasked:
                    out_u.append(full.gather(-1, labels).squeeze(-1).masked_fill(~on, 0.0))
                if compute_entropy:
                    out_e.append((-(full.exp() * full).sum(-1)).masked_fill(~on, 0.0))
        return cat(out_m), cat(out_e), cat(out_u)


# ------------------------------------------------------------------------------------------ cluster only: xgrammar
class XgrammarReplay:
    """xgrammar's GrammarMatcher for the run's regex, built as vLLM 0.32's xgrammar backend builds each request's
    (module docstring, 1), and replayed along sampled ids (cluster only: xgrammar and numpy are imported here).

    - TokenizerInfo.from_huggingface(tokenizer, vocab_size=V) with no stop ids (backend_xgrammar.py:69-72), V being
      the model config's vocab_size (structured_output/__init__.py:144);
    - GrammarCompiler(info, max_threads=8, cache_enabled=True) (:73-78; vLLM's cache_limit_bytes is left at
      xgrammar's default: one regex is compiled);
    - compile_regex(regex) (:129-131);
    - GrammarMatcher(compiled, override_stop_tokens=list(stop_ids), max_rollback_tokens=0) (:157-161, with the
      request's all_stop_token_ids, __init__.py:199-200, and no speculative tokens).

    `record` counts the walks, positions, refusals and time, and keeps the decoder check, the installed xgrammar's
    version and the stop ids the matcher was built with."""

    def __init__(self, regex: str, tokenizer: Any, *, vocab_size: int, stop_ids: Iterable[int]) -> None:
        import numpy as np
        import xgrammar as xgr

        t0 = time.perf_counter()
        self.regex_sha256 = regex_sha256(regex)
        self.vocab_size = int(vocab_size)
        self.stop_ids = _norm_stops(stop_ids)
        if not self.stop_ids:
            raise MaskError("XgrammarReplay: no stop ids (vLLM compiles each grammar with the request's stop ids)")
        self._np = np
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=self.vocab_size)
        compiler = xgr.GrammarCompiler(info, max_threads=8, cache_enabled=True)
        compiled = compiler.compile_regex(regex)
        with warnings.catch_warnings():      # 0.2.8 warns on any max_rollback_tokens but -1 (matcher.py:261-266)
            warnings.simplefilter("ignore", DeprecationWarning)
            self._matcher = xgr.GrammarMatcher(compiled, override_stop_tokens=list(self.stop_ids),
                                               max_rollback_tokens=0)
        self._bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)
        self.record: dict[str, Any] = {
            "xgrammar": self._version(xgr), "regex_sha256": self.regex_sha256,
            "vocab_size": self.vocab_size, "stop_ids": list(self.stop_ids),
            "built_as": "vLLM 0.32 backend_xgrammar: TokenizerInfo.from_huggingface(tokenizer, vocab_size), "
                        "GrammarCompiler(max_threads=8, cache_enabled=True), compile_regex, "
                        "GrammarMatcher(override_stop_tokens=stop_ids, max_rollback_tokens=0)",
            # TokenizerInfo's own stop ids are the tokenizer's default, its eos_token_id alone (tokenizer_info.py:254,
            # :325-327); the matcher's override is the set vLLM's matcher stops on (backend_xgrammar.py:157-161)
            "tokenizer_info": self._info_record(info), "matcher_stop_token_ids": self._matcher_stops(),
            "decoder_check": None,
            "walks": 0, "positions": 0, "refused": 0, "accept_refused": 0, "walk_s": 0.0}
        self._check_decoder(xgr)
        self.record["build_s"] = round(time.perf_counter() - t0, 3)

    @staticmethod
    def _version(xgr: Any) -> str | None:
        """The installed xgrammar's version, from its distribution's metadata, as train._versions reads the stack's.
        xgrammar 0.2.8's package never imports its _version.py (xgrammar/__init__.py; __version__ is set only at
        _version.py:21), so the module's own __version__ is only the fallback for a build without metadata."""
        try:
            return importlib.metadata.version("xgrammar")
        except importlib.metadata.PackageNotFoundError:
            return getattr(xgr, "__version__", None)

    def _matcher_stops(self) -> list[int] | str:
        """GrammarMatcher.stop_token_ids (xgrammar/matcher.py:520-530): the stop ids the matcher was built with, the
        override vLLM passes. A record: a failure is recorded, never raised."""
        try:
            return [int(t) for t in self._matcher.stop_token_ids]
        except Exception as err:  # noqa: BLE001: a record
            return f"unavailable: {type(err).__name__}"

    @staticmethod
    def _info_record(info: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in ("vocab_type", "vocab_size", "add_prefix_space", "stop_token_ids", "special_token_ids"):
            try:
                v = getattr(info, name)
                out[name] = len(v) if name == "special_token_ids" else (v if isinstance(v, (int, bool, list))
                                                                         else str(v))
            except Exception as err:  # noqa: BLE001: a record
                out[name] = f"unavailable: {type(err).__name__}"
        return out

    def _allowed_now(self) -> tuple[int, ...]:
        np, bm = self._np, self._bitmask
        bm.fill_(-1)                                   # the full mask, vLLM's reset value (__init__.py:224)
        self._matcher.fill_next_token_bitmask(bm, 0)   # backend_xgrammar.py:241-242
        words = bm.numpy()[0]
        nz = np.flatnonzero(words)
        return bits_to_ids(zip(nz.tolist(), words[nz].tolist()), self.vocab_size)

    def _check_decoder(self, xgr: Any) -> None:
        """The grammar's first bitmask decoded two ways: bits_to_ids against xgrammar's own reference decoder
        (testing._get_masked_tokens_from_bitmask, testing.py:211, which lists the refused ids). A disagreement stops
        the launch; a missing reference decoder is recorded."""
        self._matcher.reset()
        mine = self._allowed_now()
        ref = getattr(getattr(xgr, "testing", None), "_get_masked_tokens_from_bitmask", None)
        if not callable(ref):
            self.record["decoder_check"] = "unavailable: xgrammar.testing._get_masked_tokens_from_bitmask"
            return
        refused = {int(i) for i in ref(self._bitmask, self.vocab_size, 0)}
        theirs = tuple(i for i in range(self.vocab_size) if i not in refused)
        self.record["decoder_check"] = {"position": 0, "allowed": len(mine), "agree": mine == theirs}
        if mine != theirs:
            raise MaskError(f"XgrammarReplay: the bitmask decoder reads {len(mine)} allowed ids at the grammar's start "
                            f"and xgrammar's own {len(theirs)}")

    def walk(self, ids: Sequence[int]) -> Any:
        """The matcher along `ids` from its start: allowed[i] is the sorted set allowed before token i (up to and
        including the first refused token), violation the first token outside its set (a token after the matcher
        terminated has the empty set), terminated whether a stop id was accepted."""
        from lofi_rl.grammar_mask import Walk           # section A's walk shape, imported here (cluster)

        t0 = time.perf_counter()
        m = self._matcher
        m.reset()
        allowed: list[tuple[int, ...]] = []
        violation, terminated = None, False
        for i, raw in enumerate(ids):
            tok = int(raw)
            if terminated:                              # xgrammar accepts nothing after its stop token
                allowed.append(())
                violation = i
                break
            here = self._allowed_now()
            allowed.append(here)
            if not _contains(here, tok):
                violation = i
                break
            if not m.accept_token(tok):                 # the bitmask allowed it and the matcher refused it
                self.record["accept_refused"] += 1
                violation = i
                break
            terminated = bool(m.is_terminated())
        self.record["walks"] += 1
        self.record["positions"] += len(allowed)
        self.record["refused"] += violation is not None
        self.record["walk_s"] = round(self.record["walk_s"] + time.perf_counter() - t0, 4)
        return Walk(allowed=tuple(allowed), violation=violation, terminated=terminated)
