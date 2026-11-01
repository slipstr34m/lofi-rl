"""Stand-ins for the two modules the RL code calls but other builders write: lofi_rl.verify (score,
group_penalties) and lofi_rl.prompt (build_messages). They follow the shared interfaces exactly, and their
values are read from the synthetic completion itself, so a test states every sub-reward it expects:

    "theory=0.8 knobs=0.6 p_sax=0.05 bars=0.1 len=0.02 p_fix=0.04 repairs=2 | any tag"
    "gate=G4 p_fix=0.02"      a failed gate (gate_ok False)
    "raise"                   the verifier raises ValueError

Unset values default to theory 0.5, knobs 0.5 and 0 for the rest. group_penalties charges 0.3 for every
extra identical copy within the group it is given (taste_pipeline.md §5's identical-LoopScript row).
Not a test module.
"""
from __future__ import annotations

import hashlib
import types
from dataclasses import dataclass, field

PROMPT_SYSTEM = "You write 8-bar jazzhop loops in LoopScript."


@dataclass(frozen=True)
class FakeScore:
    gate_ok: bool
    gate: str | None
    repairs: int
    p_fix: float
    theory: float
    theory_parts: dict = field(hash=False)
    knobs: float = 0.0
    knob_parts: dict = field(default_factory=dict, hash=False)
    p_sax: float = 0.0
    penalties: dict = field(default_factory=dict, hash=False)
    digest: str | None = None


def _values(text: str) -> dict[str, str]:
    head = text.split("|", 1)[0]
    return dict(tok.split("=", 1) for tok in head.split() if "=" in tok)


def fake_verify() -> types.ModuleType:
    """A fresh module object with score(), group_penalties(), Score, and the calls it received."""
    mod = types.ModuleType("lofi_rl.verify")
    mod.Score = FakeScore
    mod.score_calls, mod.group_calls = [], []

    def score(text, brief, palette):
        mod.score_calls.append((text, brief, palette))
        if text.strip() == "raise":
            raise ValueError("fake verifier failure")
        v = _values(text)
        gate = v.get("gate")
        theory, knobs = float(v.get("theory", 0.5)), float(v.get("knobs", 0.5))
        return FakeScore(gate_ok=gate is None, gate=gate, repairs=int(v.get("repairs", 0)),
                         p_fix=float(v.get("p_fix", 0.0)), theory=theory,
                         theory_parts={"groove": theory, "sax": theory}, knobs=knobs,
                         knob_parts={"energy": knobs, "mood": knobs}, p_sax=float(v.get("p_sax", 0.0)),
                         penalties={"bars": float(v.get("bars", 0.0)), "len": float(v.get("len", 0.0))},
                         digest=None if gate else hashlib.sha256(text.encode()).hexdigest()[:16])

    def group_penalties(texts, scores):
        mod.group_calls.append((tuple(texts), tuple(scores)))
        seen, out = set(), []
        for t in texts:
            out.append(0.3 if t in seen else 0.0)
            seen.add(t)
        return out

    mod.score, mod.group_penalties = score, group_penalties
    return mod


def fake_prompt() -> types.ModuleType:
    """lofi_rl.prompt's interface: PROMPT_VERSION, build_messages(brief, palette), template_sha256()."""
    mod = types.ModuleType("lofi_rl.prompt")
    mod.PROMPT_VERSION = "v1"

    def build_messages(brief, palette):
        return [{"role": "system", "content": f"{PROMPT_SYSTEM} Instruments: {', '.join(sorted(palette.keys))}."},
                {"role": "user", "content": f"Scene: {brief.scene}. Energy: {brief.energy}. Mood: {brief.mood}."}]

    mod.build_messages = build_messages
    mod.template_sha256 = lambda: hashlib.sha256(PROMPT_SYSTEM.encode()).hexdigest()
    return mod


# A stand-in for the part of TRL's GRPOTrainer source that computes sampling/sampling_logp_difference/mean, shaped
# like TRL 0.2x's (lofi_rl.rl.trl_probe checks the installed one at run time; this is not TRL 1.17.1's text).
TRL_LIKE_SOURCE = '''
import torch


class GRPOTrainer:
    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        out = self._generate(prompts)
        prompt_ids_list, completion_ids_list, sampling_per_token_logps_list = out
        completion_mask = pad([torch.ones(len(ids)) for ids in completion_ids_list], padding_value=0)
        sampling_per_token_logps = pad([torch.tensor(lp) for lp in sampling_per_token_logps_list], padding_value=0.0)
        if self.mask_truncated_completions:
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list])
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
        with torch.no_grad():
            if self.use_vllm and self.vllm_importance_sampling_correction:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(self.model, ids, attention_mask)
            else:
                old_per_token_logps = None
        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = completion_mask.bool() if not self.tools else (completion_mask * tool_mask).bool()
            delta = delta[mask]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
'''


# The same stand-ins as source, for child interpreters (tests that block modules through sys.modules).
CHILD_FAKES = '''
import hashlib, importlib.util, sys, types
from dataclasses import dataclass, field
if importlib.util.find_spec("lofi_rl.prompt") is None:
    p = types.ModuleType("lofi_rl.prompt"); p.PROMPT_VERSION = "v1"
    p.build_messages = lambda b, pal: [{"role": "system", "content": "S"}, {"role": "user", "content": b.mood}]
    p.template_sha256 = lambda: "0" * 64
    sys.modules["lofi_rl.prompt"] = p
if importlib.util.find_spec("lofi_rl.verify") is None:
    @dataclass(frozen=True)
    class Score:
        gate_ok: bool; gate: object; repairs: int; p_fix: float; theory: float; theory_parts: dict
        knobs: float; knob_parts: dict; p_sax: float; penalties: dict; digest: object
    v = types.ModuleType("lofi_rl.verify"); v.Score = Score
    v.score = lambda t, b, pal: Score(True, None, 0, 0.0, 0.5, {}, 0.5, {}, 0.0, {"bars": 0.0, "len": 0.0}, None)
    v.group_penalties = lambda texts, scores: [0.0] * len(texts)
    sys.modules["lofi_rl.verify"] = v
'''
