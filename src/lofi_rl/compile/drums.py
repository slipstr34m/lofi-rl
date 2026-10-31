"""Drum hits from the 8 expanded bars (instrument.md section 3, Velocities; Drum sampler).

- Every X/x/g cell of every lane, after edits, is one hit at its lane's step time.
- gain = drum_gain[level] (X 1.0, x 0.71, g 0.32) x the hat/shaker metric offset of its step x humanize.
- The layer is X hard, x med, g soft, else the kit's nearest non-empty layer.
- Round-robin: rr = h mod R, h = the hit's index within its lane and layer in the cycle (so it repeats
  every loop), R = the number of one-shots in that layer.
- Choke: a hit of a choked lane (ohat) is cut by its choking lane (hat) at the first hit on a strictly later
  grid step, read cyclically, that sounds after it. The grid step decides, not the sample time: swing,
  offsets and humanize nudges only move hits a few ms, so a closed hat on the same step never chokes. The
  sampler applies the fade from `choke_at`.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from lofi_rl.dsl.ast import Bars
from lofi_rl.dsl.vocab import LANES
from lofi_rl.palette import Groove, Kit

from .events import DrumHit
from .humanize import Humanize
from .timeline import BAR, STEPS, Clock


def metric_gain(groove: Groove, lane: str, step: int) -> float:
    if lane not in groove.metric_lanes:
        return 1.0
    return 10.0 ** (groove.metric_db[step % BAR] / 20.0)


def drum_hits(bars: Bars, kit: Kit, groove: Groove, clocks: Mapping[str, Clock], human: Humanize,
              n: int) -> tuple[DrumHit, ...]:
    hits = []
    for lane in LANES:
        seen: dict[str, int] = {}
        drum = kit.lanes[lane]
        for k, level in enumerate("".join(bars.drums[lane])):
            if level not in "Xxg":
                continue
            layer = drum.layer_for(level)
            h = seen.get(layer, 0)
            seen[layer] = h + 1
            gain = groove.drum_gain[level] * metric_gain(groove, lane, k) * float(human.gain(lane, k))
            hits.append(DrumHit(lane, level, layer, clocks[lane].sample(k) % n, gain,
                                h % len(drum.layers[layer]), k))
    hits = _choke(hits, kit, clocks)
    return tuple(sorted(hits, key=lambda d: (d.on, LANES.index(d.lane), d.step)))


def choke_at(hit: DrumHit, killer_steps: frozenset[int], victim: Clock, killer: Clock) -> int | None:
    """Where the choking lane cuts `hit`: the first killer hit on a strictly later grid step (the same step
    one cycle on comes last) that also sounds after `hit`, since a hit that strikes first cannot cut it.
    A sample in `hit`'s cycle, so it may pass n; None when the killer lane is empty."""
    start = victim.sample(hit.step)
    for k in range(hit.step + 1, hit.step + STEPS + 1):
        if k % STEPS in killer_steps:
            t = killer.sample(k)                # k >= 128 is the next cycle (Clock.time adds n)
            if t > start:
                return hit.on + (t - start)
    return None


def _choke(hits: list[DrumHit], kit: Kit, clocks: Mapping[str, Clock]) -> list[DrumHit]:
    steps = {lane: frozenset(h.step for h in hits if h.lane == lane) for lane in LANES}
    out = []
    for hit in hits:
        cuts = [choke_at(hit, steps[killer], clocks[victim], clocks[killer])
                for victim, killer in kit.choke if hit.lane == victim and steps[killer]]
        cuts = [c for c in cuts if c is not None]
        out.append(replace(hit, choke_at=min(cuts)) if cuts else hit)
    return out
