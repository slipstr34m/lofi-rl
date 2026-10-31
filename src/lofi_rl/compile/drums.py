"""Drum hits from the 8 expanded bars (instrument.md section 3, Velocities; Drum sampler).

- Every X/x/g cell of every lane, after edits, is one hit at its lane's step time.
- gain = drum_gain[level] (X 1.0, x 0.71, g 0.32) x the hat/shaker metric offset of its step x humanize.
- The layer is X hard, x med, g soft, else the kit's nearest non-empty layer.
- Round-robin: rr = h mod R, h = the hit's index within its lane and layer in the cycle (so it repeats
  every loop), R = the number of one-shots in that layer. The cycle repeats, so when the layer's count of
  hits per cycle is 1 mod R, its last hit and the next cycle's first would both play one-shot 0. For R >= 3
  that last hit plays one-shot 1 instead, unlike both its neighbours (R - 1 and 0); with R = 2 and an odd
  count one repeat is unavoidable, so h mod 2 stays (review T8).
- Choke: for each of the kit's (choked, choking) pairs, a hit of the choked lane is cut by the choking lane's
  first hit on a strictly later grid step, read cyclically (the same step one cycle on comes last), that sounds
  after it; with several pairs the earliest cut wins. The grid step decides, not the sample time: swing,
  offsets and humanize nudges only move hits a few ms, so a closed hat on the same step never chokes. A lane
  may choke itself: dev_gu's open hat is choked by hat and by ohat, since a hi-hat is one instrument (review
  T7), so with no other hat in the loop it rings until its own repeat one cycle on. The sampler applies the
  fade from `choke_at`.
"""
from __future__ import annotations

from collections import Counter
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


def round_robin(h: int, count: int, size: int) -> int:
    """The one-shot of hit h (in step order) of the `count` hits a lane-layer plays per cycle, from its `size`
    one-shots: h mod size, except that for size >= 3 the last hit of a count = 1 (mod size) cycle takes one-shot
    1, so it differs from the hit before it (size - 1) and from the next cycle's first (0). A lone hit (count 1)
    and an odd count of two one-shots cannot avoid one repeat, so they keep h mod size."""
    if size >= 3 and count > 1 and h == count - 1 and count % size == 1:
        return 1
    return h % size


def drum_hits(bars: Bars, kit: Kit, groove: Groove, clocks: Mapping[str, Clock], human: Humanize,
              n: int) -> tuple[DrumHit, ...]:
    hits = []
    for lane in LANES:
        drum = kit.lanes[lane]
        cells = [(k, level, drum.layer_for(level)) for k, level in enumerate("".join(bars.drums[lane]))
                 if level in "Xxg"]
        count = Counter(layer for _, _, layer in cells)
        seen: dict[str, int] = {}
        for k, level, layer in cells:
            h = seen.get(layer, 0)
            seen[layer] = h + 1
            gain = groove.drum_gain[level] * metric_gain(groove, lane, k) * float(human.gain(lane, k))
            hits.append(DrumHit(lane, level, layer, clocks[lane].sample(k) % n, gain,
                                round_robin(h, count[layer], len(drum.layers[layer])), k))
    hits = _choke(hits, kit, clocks)
    return tuple(sorted(hits, key=lambda d: (d.on, LANES.index(d.lane), d.step)))


def choke_at(hit: DrumHit, killer_steps: frozenset[int], victim: Clock, killer: Clock) -> int | None:
    """Where the choking lane cuts `hit`: the first killer hit on a strictly later grid step (the same step
    one cycle on comes last) that also sounds after `hit`, since a hit that strikes first cannot cut it.
    For a lane that chokes itself, that same step is `hit`'s own repeat. A sample in `hit`'s cycle, so it may
    pass n; None when the killer lane is empty."""
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
