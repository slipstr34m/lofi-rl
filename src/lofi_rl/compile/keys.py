"""Keys comp (instrument.md section 3, Voicings; Chord changes).

Each strike of the expanded comp grid plays the whole voicing of the chord in effect at that step, at
the keys instrument's velocity for its level (X 96, x 80, g 52 by default); `-` holds it. Each chord
keeps one voicing for the whole loop (theory.voicing.voice_vamp), and a held voicing is cut at a chord
change and re-struck there when the grid shows `-`.
"""
from __future__ import annotations

from collections.abc import Mapping

from lofi_rl.dsl.ast import Bars, Chord

from .bass import held_notes
from .events import Note
from .humanize import Humanize
from .timeline import Clock

CHANNEL = 0


def keys_notes(bars: Bars, voicings: Mapping[Chord, tuple[int, ...]], clock: Clock,
               vel: Mapping[str, int], human: Humanize, n: int) -> tuple[Note, ...]:
    return held_notes("".join(bars.comp), bars.chords, voicings, "keys", clock, vel, human, n, CHANNEL)
