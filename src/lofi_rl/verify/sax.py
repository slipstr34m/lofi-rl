"""The sax line: its shape S, the chromatic penalty P_chrom (instrument.md §5) and the never-fading P_sax
(TASTE.md "Sax guardrail" and "Never fades").

S = .25 f_long + .25 f_land + .20 f_open + .15 f_ans + .15 f_breath:
- f_long: 1 if 2-4 notes last 8+ steps over the line; 0.5 if 1 or 5; otherwise 0.
- f_land: mean over A and B: 1 if the phrase's first step has an onset on a tone of the bar's chord (extensions
  included), 0.5 on another note, 0 on a rest.
- f_open: [the pitch class of B's last note is not the tonic].
- f_ans: 0 if B repeats A item for item; otherwise from J, the Jaccard of the two phrases' note onsets relative to
  the phrase start: 1 for 0.2 <= J <= 0.8, falling linearly to 0 at J = 0 and J = 1 (TASTE.md: the answer "starts
  like the call, ends differently").
- f_breath: mean over phrases of [0.12 <= rest share <= 0.6, and a rest of 2+ steps]; a rest is a run of
  consecutive rest items inside the phrase.

P_chrom = min(.3, .1 #bad). A note that is not free (view.is_free: mode, blue note, or a tone of the chord at its
onset) is bad unless it lasts 2 steps or fewer and the next item of the line, read as a cycle (B's last item runs
into A's first), is a free note 1-2 semitones away in sounding pitch. A rest next is no resolution.

P_sax never fades. Each of three statistics is kept within a band, and every unit outside it costs a slope, each
part capped at 0.1 (so P_sax <= 0.3):
- busy: notes per phrase, band [0, 16], 0.0125 per note over, summed over A and B;
- wide: the line's range in sounding semitones, band [0, 19], 0.02 per semitone over;
- chromatic: the share of notes that are not free, band [0, 0.2], 0.5 per unit over.
The bands are provisional (Claude's default): the real ones come from Shresth's Good-or-better loops. Every
hand-written loop in the repo but span1_dotted.ls keeps inside them (the shootout reference loops play 10-14 notes a
phrase over at most 17 semitones, with no note that is not free).
"""
from __future__ import annotations

from lofi_rl.dsl.ast import PHRASE_STEPS

from .view import SaxItem, View, chord_rel, is_free

LONG_STEPS = 8
SAX_WEIGHTS = {"f_long": 0.25, "f_land": 0.25, "f_open": 0.20, "f_ans": 0.15, "f_breath": 0.15}
ANSWER_BAND = (0.2, 0.8)
BREATH_SHARE = (0.12, 0.6)
BREATH_MIN_REST = 2
CHROM_EACH, CHROM_CAP = 0.1, 0.3
CHROM_MAX_STEPS = 2
CHROM_STEP = (1, 2)             # semitones to the resolving note
# P_sax: part -> (band, slope per unit outside it, cap). Provisional (Claude's default), see the docstring.
P_SAX_BANDS = {"busy": ((0, 16), 0.0125, 0.1), "wide": ((0, 19), 0.02, 0.1), "chromatic": ((0.0, 0.2), 0.5, 0.1)}


def f_long(view: View) -> float:
    n = sum(item.dur >= LONG_STEPS for item in view.notes)
    return 1.0 if 2 <= n <= 4 else 0.5 if n in (1, 5) else 0.0


def _land(view: View, phrase: str) -> float:
    first = view.phrase(phrase)[0]
    if first.rest:
        return 0.0
    return 1.0 if first.chord is not None and first.pc in chord_rel(first.chord) else 0.5


def f_land(view: View) -> float:
    return (_land(view, "A") + _land(view, "B")) / 2


def f_open(view: View) -> float:
    last = [i for i in view.phrase("B") if not i.rest][-1]
    return float(last.pc != 0)


def answer_from_jaccard(j: float) -> float:
    lo, hi = ANSWER_BAND
    if j < lo:
        return j / lo
    if j > hi:
        return (1.0 - j) / (1.0 - hi)
    return 1.0


def _onsets(view: View, phrase: str, start: int) -> frozenset[int]:
    return frozenset(i.step - start for i in view.phrase(phrase) if not i.rest)


def f_ans(view: View) -> float:
    if view.loop.a == view.loop.b:
        return 0.0
    a, b = _onsets(view, "A", 0), _onsets(view, "B", PHRASE_STEPS)
    return answer_from_jaccard(len(a & b) / len(a | b))


def _breathes(items: tuple[SaxItem, ...]) -> bool:
    share = sum(i.dur for i in items if i.rest) / PHRASE_STEPS
    run = longest = 0
    for item in items:
        run = run + item.dur if item.rest else 0
        longest = max(longest, run)
    return BREATH_SHARE[0] <= share <= BREATH_SHARE[1] and longest >= BREATH_MIN_REST


def f_breath(view: View) -> float:
    return (_breathes(view.phrase("A")) + _breathes(view.phrase("B"))) / 2


def bad_notes(view: View) -> int:
    scale, items, bad = view.scale, view.items, 0
    for i, item in enumerate(items):
        if item.rest or is_free(item, scale):
            continue
        nxt = items[(i + 1) % len(items)]
        resolves = (item.dur <= CHROM_MAX_STEPS and not nxt.rest and is_free(nxt, scale)
                    and CHROM_STEP[0] <= abs(nxt.pitch - item.pitch) <= CHROM_STEP[1])
        bad += not resolves
    return bad


def p_chrom(view: View) -> float:
    return min(CHROM_CAP, CHROM_EACH * bad_notes(view))


def parts(view: View) -> dict[str, float]:
    out = {"f_long": f_long(view), "f_land": f_land(view), "f_open": f_open(view), "f_ans": f_ans(view),
           "f_breath": f_breath(view)}
    out["sax"] = sum(w * out[k] for k, w in SAX_WEIGHTS.items())
    return out


def _outside(x: float, band: tuple[float, float]) -> float:
    lo, hi = band
    return max(lo - x, 0.0, x - hi)


def p_sax_parts(view: View) -> dict[str, float]:
    notes = view.notes
    pitches = [n.pitch for n in notes]
    stats = {
        "busy": [sum(not i.rest for i in view.phrase(p)) for p in ("A", "B")],
        "wide": [max(pitches) - min(pitches)],
        "chromatic": [sum(not is_free(n, view.scale) for n in notes) / len(notes)],
    }
    out = {}
    for name, values in stats.items():
        band, slope, cap = P_SAX_BANDS[name]
        out[name] = min(cap, slope * sum(_outside(v, band) for v in values))
    return out


def p_sax(view: View) -> float:
    return sum(p_sax_parts(view).values())
