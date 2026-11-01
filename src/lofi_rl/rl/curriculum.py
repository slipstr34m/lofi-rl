"""The reward curriculum (docs/design/taste_pipeline.md §5; CLAUDE.md §8; TASTE.md, "Guided, then faded").

    w_k(t) = (1 - s) * W0[k] + s * W1[k],   s = min(max(t, 0), 30) / 30

t is the optimizer step whose rollouts are being scored (TRL's trainer_state.global_step during generation:
0 for the first step). Theory falls linearly from 0.50 to 0.10 over steps 0-30 and taste rises from 0.20 to
0.60; knobs stay at 0.15 (CLAUDE.md §8: "knobs fixed at 0.15"). CE and theme keep §5's 0.05 and 0.10.

In the GRPO smoke the taste, CE and theme terms are 0, because no scorer server supplies them there
(lofi_rl.rl.reward): only their weights are logged. §5 also says "ce -> 0, knobs -> .20 unless CE validated";
that choice belongs to the scorer-server reward, after the CE check on Shresth's votes, and is not made here:
lofi_rl.rl.reward applies it when a scorer reports CE as not validated (docs/SCORER.md §10).
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

TERMS: tuple[str, ...] = ("theory", "taste", "knobs", "ce", "theme")
W0: Mapping[str, float] = MappingProxyType({"theory": 0.50, "taste": 0.20, "knobs": 0.15, "ce": 0.05, "theme": 0.10})
W1: Mapping[str, float] = MappingProxyType({"theory": 0.10, "taste": 0.60, "knobs": 0.15, "ce": 0.05, "theme": 0.10})
RAMP_STEPS = 30


def progress(step: int, ramp: int = RAMP_STEPS) -> float:
    """s in [0, 1]: how far step `step` is through the ramp. Steps before 0 count as 0, after `ramp` as 1."""
    if ramp < 1:
        raise ValueError("the ramp needs at least one step")
    return min(max(int(step), 0), ramp) / ramp


def weights(step: int, *, w0: Mapping[str, float] = W0, w1: Mapping[str, float] = W1,
            ramp: int = RAMP_STEPS) -> dict[str, float]:
    """Every term's weight at `step`, in TERMS order. The endpoints, and a weight that does not move (knobs),
    come back exactly, with no floating-point drift from the interpolation."""
    s = progress(step, ramp)
    if s == 0:
        return {k: float(w0[k]) for k in TERMS}
    if s == 1:
        return {k: float(w1[k]) for k in TERMS}
    return {k: float(w0[k]) if w0[k] == w1[k] else (1 - s) * w0[k] + s * w1[k] for k in TERMS}
