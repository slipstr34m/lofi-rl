"""Phase-0 adapter: a parsed LoopScript -> stub notes and drum hits, so the smoke renders a real LoopScript.
It uses the real grid and timing (swing, holds, phrases on the 128-step clock, the brief's tonic and mode)
but plain voicings and no expression. The Phase 1 compiler (docs/design/instrument.md §3) replaces it."""
from __future__ import annotations

from lofi_rl.compile import stub
from lofi_rl.dsl.ast import BARS, STEPS, Loop, expand, timed
from lofi_rl.dsl.vocab import NUMERAL_SEMITONES, scale_of

QUALITY_TONES = {                     # semitones above the root (instrument.md §3 table)
    "maj7": (0, 4, 7, 11), "maj9": (0, 4, 7, 11, 14), "maj7#11": (0, 4, 7, 11, 14, 18),
    "7": (0, 4, 7, 10), "9": (0, 4, 7, 10, 14), "13": (0, 4, 7, 10, 14, 21), "7sus": (0, 5, 7, 10, 14),
    "7b9": (0, 4, 7, 10, 13), "7#9": (0, 4, 7, 10, 15),
    "m7": (0, 3, 7, 10), "m9": (0, 3, 7, 10, 14), "m11": (0, 3, 7, 10, 14, 17), "m7b5": (0, 3, 6, 10), "dim7": (0, 3, 6, 9),
}
GM_KEYS = {"rhodes": 4, "ep": 4, "wurli": 4, "piano": 0, "guitar": 26, "organ": 16}   # GM programs (stub only)
GM_SAX = {"tenor": 66, "alto": 65}
VEL = {"keys": {"X": 96, "x": 80, "g": 52}, "bass": {"X": 104, "x": 88, "g": 64}}
DRUM_GAIN = {"X": 1.0, "x": 0.71, "g": 0.32}
DRUM_VOICE = {"kick": "kick", "snare": "snare", "rim": "snare", "hat": "hat", "ohat": "hat", "shaker": "hat"}
KEYS_BAND = (55, 72)                  # voicing tones folded into this window (rough stand-in for C3-C5 voice-leading)
BASS_LOW = 28                         # E1: roots land in E1..D#2
TENOR_HOME = 50                       # the tonic of the sax sits in the tenor home band 50-61


def _fold(p: int, lo: int, hi: int) -> int:
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return p


def _held_runs(cells: str) -> list[tuple[int, int, str]]:
    """(start step, length in steps, level) for each strike; '-' extends the previous strike."""
    runs: list[tuple[int, int, str]] = []
    for i, c in enumerate(cells):
        if c in "Xxg":
            runs.append((i, 1, c))
        elif c == "-" and runs and runs[-1][0] + runs[-1][1] == i:
            s, n, lv = runs[-1]
            runs[-1] = (s, n + 1, lv)
    return runs


def notes_from_loop(loop: Loop, tonic_pc: int, mode: str, n: int, swing: float = 0.55):
    """Return (notes, programs, hits) for compile.stub, on the 128-step clock of an n-sample loop."""
    bars = expand(loop)
    scale = scale_of(mode)
    step = lambda k: stub.step_sample(k % 128, n, swing) + (n if k >= 128 else 0)   # noqa: E731
    dur = lambda k0, k1: max(1, step(k1) - step(k0) - 200)                            # noqa: E731
    notes: list[stub.Note] = []
    for name, grid, ch in (("keys", bars.comp, 0), ("bass", bars.bass, 1)):
        for k0, length, lv in _held_runs("".join(grid)):
            chord = bars.chords[k0 // STEPS]
            root_pc = (tonic_pc + NUMERAL_SEMITONES[chord.numeral]) % 12
            if name == "keys":
                tones = QUALITY_TONES[chord.quality][1:]            # rootless: the bass carries the root
                pitches = sorted({_fold(60 + root_pc + t, *KEYS_BAND) for t in tones})
            else:
                pitches = [BASS_LOW + (root_pc - BASS_LOW) % 12]
            for p in pitches:
                notes.append(stub.Note(ch, p, step(k0), dur(k0, k0 + length), VEL[name][lv]))
    home = _fold(TENOR_HOME + tonic_pc - TENOR_HOME % 12, TENOR_HOME, TENOR_HOME + 11)
    for k0, item in timed(loop.a, 0) + timed(loop.b, BARS * STEPS // 2):
        if item.rest:
            continue
        p = home + scale[item.degree - 1] + item.accidental + 12 * item.octave
        notes.append(stub.Note(2, p, step(k0), dur(k0, k0 + item.dur), 92))
    hits = [(DRUM_VOICE[lane], step(b * STEPS + s), DRUM_GAIN[c])
            for lane, grids in bars.drums.items() for b, g in enumerate(grids) for s, c in enumerate(g) if c in DRUM_GAIN]
    programs = {0: GM_KEYS.get(loop.keys.instrument, 4), 1: 33, 2: GM_SAX.get(loop.sax or "tenor", 66)}
    return notes, programs, hits
