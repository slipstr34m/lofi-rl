"""LoopScript v1: parser, repairs, hard gates G1-G6 and canonical text (instrument.md s.1-2).

Standard library only: it is imported in both the policy env and the scorer env.
"""
from .ast import (BARS, PHRASE_STEPS, STEPS, Bars, Chord, Edit, Item, Keys, Lane, Loop, Vamp,
                  chord_bars, expand, lock_bass, timed)
from .canon import canon, canon_sha, item_text, phrase_text
from .parse import ParseResult, parse
from .repair import PENALTY_CAP, PENALTY_EACH, Gate, Repair
from .vocab import DEFAULT_VOCAB, Vocab

__all__ = [
    "BARS", "PHRASE_STEPS", "STEPS", "Bars", "Chord", "Edit", "Item", "Keys", "Lane", "Loop",
    "Vamp", "chord_bars", "expand", "lock_bass", "timed", "canon", "canon_sha", "item_text",
    "phrase_text", "ParseResult", "parse", "PENALTY_CAP", "PENALTY_EACH", "Gate", "Repair",
    "DEFAULT_VOCAB", "Vocab",
]
