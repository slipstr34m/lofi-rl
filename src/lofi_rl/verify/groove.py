"""Groove: G = .30 f_bb + .25 f_k1 + .25 f_lock + .20 f_tk (instrument.md §5; TASTE.md "Guided, then faded").

- f_bb (backbeat): mean over bars of [a non-ghost snare or rim hit on step 5 and on step 13]. The bracket needs
  both beats, from either lane at each.
- f_k1 (kick on 1): mean over bars of [a non-ghost kick on step 1].
- f_lock (kick-bass lock): min(1, (|K n B| / |B|) / 0.6) over (bar, step) onsets. Onsets are strikes (X x g)
  of the kick and of the bass grid after edits and `bass: lock`; a hold the compiler re-strikes at a chord change
  is not one. With no bass strike at all it is 0 (Claude's default: nothing locks).
- f_tk (timekeeping): mean over bars of [hat + shaker hits >= 4]; ghosts are hits, the open hat is not counted.
"""
from __future__ import annotations

from .view import BACKBEAT, NON_GHOST, STRIKES, View

BEAT_2, BEAT_4 = 4, 12          # steps 5 and 13 of the bar
LOCK_SHARE = 0.6
TIMEKEEPERS = ("hat", "shaker")
TIMEKEEPING_HITS = 4
WEIGHTS = {"f_bb": 0.30, "f_k1": 0.25, "f_lock": 0.25, "f_tk": 0.20}


def backbeat(view: View, bar: int) -> bool:
    return all(any(view.drum(lane, bar)[step] in NON_GHOST for lane in BACKBEAT) for step in (BEAT_2, BEAT_4))


def kick_on_1(view: View, bar: int) -> bool:
    return view.drum("kick", bar)[0] in NON_GHOST


def timekeeping_hits(view: View, bar: int) -> int:
    return sum(c in STRIKES for lane in TIMEKEEPERS for c in view.drum(lane, bar))


def onsets(bars: tuple[str, ...]) -> frozenset[tuple[int, int]]:
    return frozenset((b, s) for b, cells in enumerate(bars) for s, c in enumerate(cells) if c in STRIKES)


def f_lock(view: View) -> float:
    bass = onsets(view.bars.bass)
    if not bass:
        return 0.0
    kick = onsets(view.bars.drums["kick"])
    return min(1.0, len(kick & bass) / len(bass) / LOCK_SHARE)


def parts(view: View) -> dict[str, float]:
    bars = range(len(view.bars.chords))
    out = {
        "f_bb": sum(backbeat(view, b) for b in bars) / len(bars),
        "f_k1": sum(kick_on_1(view, b) for b in bars) / len(bars),
        "f_lock": f_lock(view),
        "f_tk": sum(timekeeping_hits(view, b) >= TIMEKEEPING_HITS for b in bars) / len(bars),
    }
    out["groove"] = sum(w * out[k] for k, w in WEIGHTS.items())
    return out
