"""Variation across bars (instrument.md §5): something new every bar, no fill at the seam, repeated bars.

- f_new = (1/7) sum over b = 2..8 of [sig(b) not in {sig(1..b-1)}], sig = (chord, the drum lanes, bass, comp).
  The sax is left out because it is new by construction. §5's "5 drum lanes" counts snare and rim as the one
  backbeat lane; sig holds all six lane bars, which is the same information. Comp and bass are the bars as heard
  (view.heard). The bar signature holds the chord and a vamp has 2 distinct chords, so f_new >= 1/7: §5's fixture
  "1-bar bases, no edits -> f_new = 0" cannot occur, and that loop scores 1/7 (CHANGELOG 2026-10-30).
- f_seam = 0 if any of these hold, else 1:
  1. bar 8 has 3 or more non-ghost snare or rim hits in steps 9-16;
  2. the Hamming distance between bar 8's drums and the bases' bar 8 (no edits) is over 6 cells, over all six
     lanes, a changed level counting as a changed cell;
  3. bar 8 lacks the backbeat (groove.backbeat), or bar 1 lacks the kick on 1;
  4. bar 8 has fewer hat + shaker hits than max(2, half the median over bars 1-7).
- The largest class of identical full bars (P_rep's m) uses sig plus the sax as it sounds in the bar: for each of
  its 16 steps, silence, or the sounding pitch and whether it starts there.
"""
from __future__ import annotations

from collections import Counter
from statistics import median

from lofi_rl.dsl.ast import BARS, STEPS
from lofi_rl.dsl.vocab import LANES

from .groove import backbeat, kick_on_1, timekeeping_hits
from .view import BACKBEAT, NON_GHOST, View

FILL_HITS = 3                   # rule 1: this many non-ghost backbeat hits in steps 9-16 of bar 8 is a fill
FILL_CELLS = 6                  # rule 2: more changed cells than this is a fill
SEAM_MIN_HITS = 2               # rule 4


def bar_sig(view: View, bar: int) -> tuple:
    return (view.bars.chords[bar], tuple(view.drum(lane, bar) for lane in LANES), view.bass[bar], view.comp[bar])


def f_new(view: View) -> float:
    seen, new = {bar_sig(view, 0)}, 0
    for bar in range(1, BARS):
        sig = bar_sig(view, bar)
        new += sig not in seen
        seen.add(sig)
    return new / (BARS - 1)


def seam_breaks(view: View) -> list[int]:
    """The seam rules (1-4) that bar 8 breaks; empty means no fill at the seam."""
    last = BARS - 1
    broken = []
    if sum(view.drum(lane, last)[s] in NON_GHOST for lane in BACKBEAT for s in range(8, STEPS)) >= FILL_HITS:
        broken.append(1)
    changed = sum(a != b for lane in LANES for a, b in zip(view.drum(lane, last), view.base[lane][last]))
    if changed > FILL_CELLS:
        broken.append(2)
    if not backbeat(view, last) or not kick_on_1(view, 0):
        broken.append(3)
    floor = max(SEAM_MIN_HITS, median(timekeeping_hits(view, b) for b in range(last)) / 2)
    if timekeeping_hits(view, last) < floor:
        broken.append(4)
    return broken


def f_seam(view: View) -> float:
    return 0.0 if seam_breaks(view) else 1.0


def sax_roll(view: View, bar: int) -> tuple:
    roll: list[tuple[int, bool] | None] = [None] * STEPS
    for item in view.notes:
        for k in range(item.step, item.step + item.dur):
            if k // STEPS == bar:
                roll[k % STEPS] = (item.pitch, k == item.step)
    return tuple(roll)


def largest_identical_class(view: View) -> int:
    """m: the most full bars, sax included, that are identical."""
    return max(Counter((bar_sig(view, b), sax_roll(view, b)) for b in range(BARS)).values())

