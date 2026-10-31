"""The sax line: phrase degrees -> pitches, timing and velocity (instrument.md section 3, Pitch mapping,
Registers, Sax expression).

- Pitch = T + S[d-1] + accidental + 12 * octave, with T the brief's tonic in the sax's home band and S
  the mode (melancholy aeolian, wistful dorian, hopeful ionian).
- Octave fit, per phrase (A = bars 1-4, B = 5-8): the shift k in {-1, 0, +1} with the fewest notes outside
  the comfort window; ties go to 0, then to -1. Notes still outside the playable range fold by octaves
  into it; each folded note is a penalized `sax_fold` repair.
- Timing (palette sax_expr.timing): `grid` plays on the band's swung grid (the sax lane's swing and
  offset); `lag` adds lag_ms; `swung` replaces the 16th swing with 8th-note swing swing8.
- Note-off is gap_ms (15 ms) before a note that follows immediately, the phrase read as a cycle (B's last
  note runs into A's first); otherwise at the end of its last step.
- Velocity 92 base; +8 on step 1 of a bar, +4 on beat 3, -10 for `s` notes, +6 on each phrase's first
  note; clipped to 60-112. With expression off it is the base.
- Channels alternate 2, 3 around the line read as a cycle, so each note's bend and swell leave the release
  of the note before it alone, B's last note before A's first included. Two channels cannot alternate
  around an odd number of notes, so then B's last note takes channel 4.
"""
from __future__ import annotations

from dataclasses import dataclass

from lofi_rl.dsl.ast import PHRASE_STEPS, Item, Loop, timed
from lofi_rl.dsl.canon import item_text
from lofi_rl.dsl.repair import Repair
from lofi_rl.palette import Groove, SaxExpr, SaxVoice
from lofi_rl.theory.modes import degree_offset, home_tonic

from .timeline import BAR, Clock, ms, rnd, wrap

CHANNELS = (2, 3)               # sax notes alternate these in line order
SEAM_CHANNEL = 4                # B's last note when the line has an odd number of notes
SHIFT_PREFERENCE = (0, -1, 1)   # octave-fit ties: no shift first, then down


@dataclass(frozen=True)
class SaxNote:
    """A sax note with what the expression needs to know about its place in the line."""

    pitch: int
    step: int                   # onset step 0..127
    dur: int                    # steps
    on: int                     # samples, [0, n)
    off: int                    # samples, may pass n
    velocity: int
    channel: int
    phrase: str                 # A | B
    first: bool                 # first note of its phrase
    last: bool                  # last note of its phrase
    scoop: bool                 # follows a rest, or a leap of scoop.leap semitones or more


def sax_clock(n: int, groove: Groove, expr: SaxExpr) -> Clock:
    offset = groove.offset_for("sax") + (expr.lag_ms if expr.timing == "lag" else 0)
    if expr.timing == "swung":
        return Clock(n, offset=ms(offset), swing8=expr.swing8)
    return Clock(n, swing=groove.swing_for("sax"), offset=ms(offset))


def note_channels(count: int) -> tuple[int, ...]:
    """The MIDI channel of each of `count` sax notes in line order: 2, 3, 2, 3, ..., with B's last note on
    SEAM_CHANNEL when the count is odd, so no two notes that follow each other around the cycle share one."""
    chans = [CHANNELS[i % 2] for i in range(count)]
    if count % 2 and count > 1:
        chans[-1] = SEAM_CHANNEL
    return tuple(chans)


def fit_shift(pitches: list[int], comfort: tuple[int, int]) -> int:
    lo, hi = comfort
    outside = {k: sum(not lo <= p + 12 * k <= hi for p in pitches) for k in SHIFT_PREFERENCE}
    return min(SHIFT_PREFERENCE, key=lambda k: (outside[k], SHIFT_PREFERENCE.index(k)))


def fold(pitch: int, play: tuple[int, int]) -> int:
    lo, hi = play
    while pitch < lo:
        pitch += 12
    while pitch > hi:
        pitch -= 12
    return pitch


def sax_pitches(loop: Loop, tonic: int, mode: str, voice: SaxVoice
                ) -> tuple[list[tuple[str, int, Item, int]], list[Repair]]:
    """(phrase, onset step, item, sounding pitch) for every note, and the fold repairs."""
    home = home_tonic(tonic, voice.home)
    out, repairs = [], []
    for name, items, start in (("A", loop.a, 0), ("B", loop.b, PHRASE_STEPS)):
        notes = [(k, it) for k, it in timed(items, start) if not it.rest]
        raw = [home + degree_offset(mode, it.degree, it.accidental, it.octave) for _, it in notes]
        shift = 12 * fit_shift(raw, voice.comfort)
        for (k, it), p in zip(notes, raw):
            q = fold(p + shift, voice.play)
            if q != p + shift:
                where = f"{name} {item_text(it)} in bar {k // BAR + 1} step {k % BAR + 1}"
                repairs.append(Repair("sax_fold", 0, f"{where}: MIDI {p + shift}", f"MIDI {q}",
                                      penalized=True))
            out.append((name, k, it, q))
    return out, repairs


def sax_line(loop: Loop, tonic: int, mode: str, voice: SaxVoice, groove: Groove, expr: SaxExpr,
             n: int) -> tuple[tuple[SaxNote, ...], list[Repair]]:
    pitched, repairs = sax_pitches(loop, tonic, mode, voice)
    pitch_at = {k: p for _, k, _, p in pitched}
    seq = timed(loop.a, 0) + timed(loop.b, PHRASE_STEPS)        # the line as a cycle of items
    clock = sax_clock(n, groove, expr)
    gap = ms(expr.gap_ms)
    by_phrase = [[k for ph, k, _, _ in pitched if ph == name] for name in "AB"]
    if not all(by_phrase):
        raise ValueError("each phrase needs at least one note (gate G6)")
    firsts, lasts = {ks[0] for ks in by_phrase}, {ks[-1] for ks in by_phrase}
    channels = note_channels(sum(not item.rest for _, item in seq))
    notes = []
    for i, (k, item) in enumerate(seq):
        if item.rest:
            continue
        prev_k, prev = seq[i - 1]
        nxt = seq[(i + 1) % len(seq)][1]
        on = clock.sample(k)
        end = clock.time(k + item.dur)
        off = max(on + 1, rnd(end - gap) if not nxt.rest else rnd(end))
        on, off = wrap(on, off, n)
        scoop = prev.rest or abs(pitch_at[k] - pitch_at[prev_k]) >= expr.scoop.leap
        notes.append(SaxNote(pitch_at[k], k, item.dur, on, off, _velocity(k, item, k in firsts, expr),
                             channels[len(notes)], "A" if k < PHRASE_STEPS else "B", k in firsts,
                             k in lasts, scoop))
    return tuple(notes), repairs


def _velocity(k: int, item: Item, phrase_start: bool, expr: SaxExpr) -> int:
    v = expr.velocity
    if not expr.enabled:
        return v.base
    vel = (v.base + v.bar_start * (k % BAR == 0) + v.beat3 * (k % BAR == 8) + v.sixteenth * (item.dur == 1)
           + v.phrase_start * phrase_start)
    return min(v.max, max(v.min, vel))
