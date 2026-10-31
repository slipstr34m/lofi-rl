"""Penalties on one loop and within one prompt's group (instrument.md §5 Penalties; docs/design/taste_pipeline.md §5).

Per loop (Score.penalties):
- bars: P_rep = min(0.3, 0.1 max(0, m - 3)), m the largest class of identical full bars, sax included
  (variation.largest_identical_class). The 0.3 cap is the plan's. Each chord fills 4 bars, so m <= 4 and in
  practice P_rep is 0 or 0.1.
- len: P_len = min(0.3, 0.004 max(0, T - 360)), T the completion's tokens. Without a count T is estimated at 4
  characters a token, ceil(len(text) / 4) (Claude's default, as the brief asked). LoopScript runs denser: §1's
  example is 268 Qwen3 tokens in 584 characters, 2.18 a token, so the estimate undercounts by about 1.8x and P_len
  starts near 1,440 characters, past what max_completion_length (448-512 tokens) lets through. Pass n_tokens from
  the trainer's completion_ids to charge the real count.

Within a group (group_penalties; the plan's two levels, which replace §5's ramp):
- identical: two gate-passing completions with the same compiled digest are one loop; the first copy is free and
  each later copy costs 0.3 ("0.3 per extra copy").
- near-identical: a gate-passing completion whose §5 similarity to another distinct one in the group is >= NEAR
  costs 0.15 (both members of the pair pay; a copy already paying 0.3 pays nothing more).
- sim is §5's Jaccard of event sets: drum (lane, step, level), sax (step, semitones above the tonic, duration) and
  chord (bar, chord) events, steps on the 128-step clock. §5 leaves comp and bass out, so two loops that differ
  only there have sim 1 and count as near-identical, not identical.
- NEAR = 0.875 is provisional (Claude's default): the plan's threshold is p99.5 of same-brief Claude pairs, from
  the seed pool, which does not exist yet. 0.875 is where §5's ramp, 0.3 clip((sim - 0.8) / 0.15, 0, 1), reaches
  the plan's 0.15, and §5's cousins below 0.8 still cost nothing.
- A completion that failed a gate costs nothing here (its reward is 0) and is no one's copy.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from lofi_rl.dsl.vocab import LANES

from .view import STRIKES, View

REP_EACH, REP_FREE, REP_CAP = 0.1, 3, 0.3
LEN_FREE, LEN_EACH, LEN_CAP = 360, 0.004, 0.3
CHARS_PER_TOKEN = 4
P_IDENTICAL, P_NEAR = 0.3, 0.15
NEAR = 0.875


def p_rep(m: int) -> float:
    return min(REP_CAP, REP_EACH * max(0, m - REP_FREE))


def tokens(text: str, n_tokens: int | None = None) -> int:
    return n_tokens if n_tokens is not None else math.ceil(len(text) / CHARS_PER_TOKEN)


def p_len(text: str, n_tokens: int | None = None) -> float:
    return min(LEN_CAP, LEN_EACH * max(0, tokens(text, n_tokens) - LEN_FREE))


def events(view: View) -> frozenset[tuple]:
    """§5's event set of a loop: drum hits, sax notes and the chord of each bar."""
    out: set[tuple] = set()
    for lane in LANES:
        for k, c in enumerate("".join(view.bars.drums[lane])):
            if c in STRIKES:
                out.add(("drum", lane, k, c))
    out.update(("sax", n.step, n.rel, n.dur) for n in view.notes)
    out.update(("chord", b, c.text) for b, c in enumerate(view.bars.chords))
    return frozenset(out)


def similarity(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def group(keys: Sequence[object | None], event_sets: Sequence[frozenset | None]) -> list[float]:
    """keys[i] is a loop's identity (its compiled digest), None for a completion that failed a gate; event_sets[i]
    is its §5 event set, or None when unknown (then it can only be judged identical)."""
    out = [0.0] * len(keys)
    first: dict[object, int] = {}
    for i, key in enumerate(keys):
        if key is None:
            continue
        if key in first:
            out[i] = P_IDENTICAL
        else:
            first[key] = i
    reps = list(first.values())
    for i in reps:
        if event_sets[i] is None:
            continue
        for j in reps:
            if j != i and event_sets[j] is not None and similarity(event_sets[i], event_sets[j]) >= NEAR:
                out[i] = P_NEAR
                break
    return out
