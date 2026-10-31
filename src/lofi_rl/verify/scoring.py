"""score(): every symbolic reward term of one completion; group_penalties(): duplicates within one prompt's group.

The shared interface of the GRPO workflow (lofi_rl.prompt, lofi_rl.verify, lofi_rl.rl.reward). docs/VERIFIERS.md
holds every formula, constant and reading.
1. Gates. dsl.parse with the palette's vocabulary in the brief's mood runs G1-G6 (G2 when finish_reason is
   "length"); then compile_loop, where a CompileError is "G7-compile". G7's render part belongs to the renderer.
   A failed gate gives gate_ok False and 0 for every term and every penalty, p_fix included, so any combination of
   them gives the reward 0 that a failed gate sets (docs/DSL.md §7); `repairs` still counts what was logged.
2. theory = clip(.3 G + .3 S + .2 f_new + .2 f_seam - P_chrom, 0, 1), its parts in theory_parts (groove, sax,
   variation).
3. knobs = (f_energy + f_mood) / 2, with E, the band match and the three mood scores in knob_parts (knobs).
4. p_sax, the never-fading sax restraint (sax), and penalties {bars, len} (penalties).
5. repairs: every repair the parser and the compiler logged, free aliases included; p_fix = 0.02 per penalized
   one, capped at 0.1 (dsl.repair.penalty). digest: the CompiledLoop's sha256, None when a gate failed.
Standard library plus the palette's PyYAML: it imports in the policy env.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from lofi_rl.brief import Brief
from lofi_rl.compile.compiler import CompileError, compile_loop
from lofi_rl.dsl import parse
from lofi_rl.dsl.repair import penalty
from lofi_rl.palette import Palette

from . import groove, knobs, penalties, sax, variation
from .view import make_view

THEORY_KEYS = ("f_bb", "f_k1", "f_lock", "f_tk", "groove", "f_long", "f_land", "f_open", "f_ans", "f_breath",
               "sax", "f_new", "f_seam", "p_chrom")
KNOB_KEYS = ("E", "f_energy", "energy_match", "mood_aeolian", "mood_dorian", "mood_ionian", "mood_margin", "f_mood")
PENALTY_KEYS = ("bars", "len")
THEORY_WEIGHTS = {"groove": 0.3, "sax": 0.3, "f_new": 0.2, "f_seam": 0.2}
G7_COMPILE = "G7-compile"


@dataclass(frozen=True)
class Score:
    """Every symbolic term of one completion. The dict fields are plain dicts, so dataclasses.asdict(score) is a
    JSON-ready log row.

    score() also attaches the loop's §5 event set as the non-field attribute `_events`, which group_penalties reads
    for near-identity; asdict, == and repr ignore it. A Score rebuilt without it (dataclasses.replace, a JSON log)
    is still judged identical by its digest."""

    gate_ok: bool
    gate: str | None                                    # the first failing gate: G1..G6 or G7-compile; None if ok
    repairs: int                                        # repairs logged by the parser and the compiler
    p_fix: float                                        # 0.02 per penalized repair, capped at 0.1
    theory: float                                       # 0-1
    theory_parts: dict[str, float] = field(hash=False)  # THEORY_KEYS, each 0-1 (p_chrom is the penalty, <= 0.3)
    knobs: float                                        # 0-1
    knob_parts: dict[str, float] = field(hash=False)    # KNOB_KEYS (E, f_energy, energy_match, f_mood in 0-1)
    p_sax: float                                        # >= 0 (<= 0.3)
    penalties: dict[str, float] = field(hash=False)     # bars, len: each >= 0
    digest: str | None                                  # CompiledLoop.digest(), None when a gate failed


def _failed(gate: str, repairs: int) -> Score:
    return Score(False, gate, repairs, 0.0, 0.0, dict.fromkeys(THEORY_KEYS, 0.0), 0.0, dict.fromkeys(KNOB_KEYS, 0.0),
                 0.0, dict.fromkeys(PENALTY_KEYS, 0.0), None)


def score(text: str, brief: Brief, palette: Palette, *, finish_reason: str | None = None,
          n_tokens: int | None = None) -> Score:
    """Score one completion for its brief. finish_reason "length" is gate G2; n_tokens, when given, replaces the
    4-characters-a-token estimate in the length penalty."""
    res = parse(text, vocab=palette.vocab(), finish_reason=finish_reason, mode=brief.mood)
    if not res.gate.ok or res.ast is None:
        return _failed(res.gate.code or "G1", len(res.repairs))
    try:
        compiled = compile_loop(res.ast, brief, palette)
    except CompileError:
        return _failed(G7_COMPILE, len(res.repairs))
    view = make_view(res.ast, brief, compiled)

    theory_parts = {**groove.parts(view), **sax.parts(view), "f_new": variation.f_new(view),
                    "f_seam": variation.f_seam(view), "p_chrom": sax.p_chrom(view)}
    theory_parts = {k: float(theory_parts[k]) for k in THEORY_KEYS}
    raw = sum(w * theory_parts[k] for k, w in THEORY_WEIGHTS.items()) - theory_parts["p_chrom"]
    knob_parts = {k: float(v) for k, v in knobs.parts(view, brief.energy).items()}
    repairs = list(res.repairs) + list(compiled.repairs)
    out = Score(
        gate_ok=True, gate=None, repairs=len(repairs), p_fix=penalty(repairs),
        theory=min(1.0, max(0.0, raw)), theory_parts=theory_parts,
        knobs=knobs.combine(knob_parts), knob_parts={k: knob_parts[k] for k in KNOB_KEYS},
        p_sax=sax.p_sax(view),
        penalties={"bars": penalties.p_rep(variation.largest_identical_class(view)),
                   "len": penalties.p_len(text, n_tokens)},
        digest=compiled.digest(),
    )
    object.__setattr__(out, "_events", penalties.events(view))
    return out


def group_penalties(texts: Sequence[str], scores: Sequence[Score]) -> list[float]:
    """The duplicate and near-duplicate penalty of each completion of one prompt's group (penalties.group): 0.3 for
    each later copy of a loop, 0.15 for a loop near-identical to another in the group, 0 for a failed one. A loop is
    its compiled digest (the stripped text when a gate-passing Score has none)."""
    if len(texts) != len(scores):
        raise ValueError(f"{len(texts)} texts but {len(scores)} scores")
    keys = [None if not s.gate_ok else (s.digest or ("text", t.strip())) for t, s in zip(texts, scores)]
    return penalties.group(keys, [getattr(s, "_events", None) if s.gate_ok else None for s in scores])
