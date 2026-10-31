"""Symbolic reward terms of a LoopScript completion (docs/design/instrument.md §5; docs/VERIFIERS.md).

    from lofi_rl.verify import score, group_penalties
    s = score(text, brief, palette)            # gates G1-G6 and G7-compile, theory, knobs, p_sax, penalties
    extra = group_penalties(texts, scores)     # duplicates within one prompt's group

No audio: everything comes from the parsed Loop and the CompiledLoop. Standard library plus the palette's PyYAML,
so it imports in the policy env. Sound claims: none.
"""
from .scoring import KNOB_KEYS, PENALTY_KEYS, THEORY_KEYS, Score, group_penalties, score

__all__ = ["KNOB_KEYS", "PENALTY_KEYS", "THEORY_KEYS", "Score", "group_penalties", "score"]
