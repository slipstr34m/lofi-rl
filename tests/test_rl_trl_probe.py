"""lofi_rl.rl.trl_probe: what a GRPO run reads from the installed TRL, vLLM and PEFT (D-020, Outcome; D-021 Part A).

TRL 1.17.1 is not on the laptop, so these tests run the checks on stand-in sources and classes:
- the IS-metric check says "match" for a source shaped like TRL's (delta = |old - sampling|, masked by the completion
  mask, a plain mean, appended to sampling/sampling_logp_difference/mean) and "not_matched", naming what is
  missing, for every way a source can compute something else; logp_metric_source reads the file that defines
  GRPOTrainer, records its sha256 and lines, and never raises;
- BackendPin asks every LLM built for xgrammar as junk.build_llm does (a TypeError on the keyword builds it again
  without it), keeps a caller's own choice, and restores the class; find_engine walks a trainer to the engine,
  wherever it sits; engine_record reads its backend through junk.read_backend;
- D-021's source checks (T1-T5 gating, V1-V2 recorded) say "match" for a synthetic TRL / vLLM / PEFT tree written
  here (shaped like the installed one but with other local names, so every check binds names by structure; no
  library text is copied), name the path the grammar mask takes (support_ids, or fill for the registered logits
  form), say "not_matched" with what is missing for each way a source or the args can differ, "not_found" for an
  absent package or file and "error" for one that does not parse, and never raise; lm_head_record checks a built
  model's output embedding on stand-in PEFT models."""
import hashlib
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from helpers_rl import TRL_LIKE_SOURCE as TRL_LIKE
from lofi_rl.rl import trl_probe as tp


def _variant(old: str, new: str) -> str:
    assert old in TRL_LIKE, old
    return TRL_LIKE.replace(old, new)


# ------------------------------------------------------------------------------------------ the IS metric's source
def test_a_trl_shaped_source_matches_d020s_definition_and_the_record_keeps_its_lines():
    r = tp.analyze_source(TRL_LIKE)
    assert r["verdict"] == "match" and r["missing"] == []
    (site,) = r["sites"]
    assert site["function"] == "_generate_and_score_completions" and site["value"] == "mean_delta"
    assert site["mean_of"] == "delta" and site["difference"] == ["old_per_token_logps", "sampling_per_token_logps"]
    assert (site["vllm_operand"], site["trainer_operand"], site["mask"]) == (
        "sampling_per_token_logps", "old_per_token_logps", "mask")
    assert r["excludes_truncated"] is True, "the source zeroes truncated completions out of completion_mask"
    lines = "\n".join(r["lines"].values())
    for piece in ("torch.abs(old_per_token_logps - sampling_per_token_logps)", "delta = delta[mask]",
                  "self._get_per_token_logps_and_entropies(self.model", "completion_mask.bool()",
                  "sampling/sampling_logp_difference/mean", "mask_truncated_completions"):
        assert piece in lines, piece


@pytest.mark.parametrize("old, new", [
    ("delta = delta[mask]\n", "delta = torch.masked_select(delta, mask)\n"),
    ("delta = torch.abs(old_per_token_logps - sampling_per_token_logps)\n",
     "delta = (old_per_token_logps - sampling_per_token_logps).abs()\n"),
    ("mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)",
     "mean_delta = delta.mean()"),
], ids=["masked_select", "abs-method", "tensor-mean"])
def test_the_spellings_of_the_same_computation_match(old, new):
    assert tp.analyze_source(_variant(old, new))["verdict"] == "match"


@pytest.mark.parametrize("old, new, why", [
    ("delta = torch.abs(old_per_token_logps - sampling_per_token_logps)",
     "delta = torch.exp(old_per_token_logps - sampling_per_token_logps)", "not |A - B|"),
    ("delta = torch.abs(old_per_token_logps - sampling_per_token_logps)",
     "delta = old_per_token_logps - sampling_per_token_logps", "not |A - B|"),
    ("            delta = delta[mask]\n", "", "not masked"),
    ("mask = completion_mask.bool() if not self.tools else (completion_mask * tool_mask).bool()",
     "mask = attention_mask.bool()", "not built from completion_mask"),
    ("mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)",
     "mean_delta = (delta * weights).sum() / weights.sum()", "not a plain mean"),
    ("            else:\n                old_per_token_logps = None",
     "            else:\n                old_per_token_logps = sampling_per_token_logps.detach()",
     "not computed by a call on self.model"),
    ("old_per_token_logps, _ = self._get_per_token_logps_and_entropies(self.model, ids, attention_mask)",
     "old_per_token_logps, _ = self._get_per_token_logps_and_entropies(self.ref_model, ids, attention_mask)",
     "not computed by a call on self.model"),
    ("self.accelerator.gather(mean_delta).mean().item()", "self.accelerator.gather(mean_delta).sum().item()",
     "goes through ['sum']"),
    ("self.accelerator.gather(mean_delta).mean().item()", "(mean_delta + max_delta).item()", "not one tensor"),
    ("delta = torch.abs(old_per_token_logps - sampling_per_token_logps)",
     "delta = torch.abs(old_per_token_logps - ref_per_token_logps)", "not one sampling log-prob tensor"),
    ("            delta = delta[mask]\n", "            delta = delta[mask]\n            delta = delta * 2\n",
     "reassigned after |A - B|"),
])
def test_every_other_computation_is_not_matched_and_says_what_differs(old, new, why):
    r = tp.analyze_source(_variant(old, new))
    assert r["verdict"] == "not_matched" and any(why in m for m in r["missing"]), r["missing"]


def test_a_source_that_never_appends_the_metric_is_not_matched_and_truncation_unseen_is_none():
    r = tp.analyze_source("def f(self):\n    name = 'sampling/sampling_logp_difference/mean'\n")
    assert r["verdict"] == "not_matched" and r["sites"] == [] and r["excludes_truncated"] is None
    no_trunc = TRL_LIKE.replace("if self.mask_truncated_completions:", "if self.something_else:")
    assert tp.analyze_source(no_trunc)["excludes_truncated"] is None, "not seen is not 'no'"


def _package(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = tmp_path / "trl" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))
    return tmp_path / "trl"


def test_the_installed_source_is_read_from_the_file_that_defines_grpotrainer(tmp_path):
    other = _variant("delta = torch.abs(old_per_token_logps - sampling_per_token_logps)",
                     "delta = torch.exp(old_per_token_logps - sampling_per_token_logps)")
    base = _package(tmp_path, {"__init__.py": "", "trainer/grpo_trainer.py": TRL_LIKE,
                               "trainer/rloo_trainer.py": other.replace("class GRPOTrainer", "class RLOOTrainer"),
                               "trainer/utils.py": "LLM(model, logprobs_mode='processed_logprobs')\n"})
    r = tp.logp_metric_source(base=base)
    assert r["verdict"] == "match" and r["file"] == "trainer/grpo_trainer.py" and len(r["sha256"]) == 64
    assert r["files_naming_metric"] == ["trainer/grpo_trainer.py", "trainer/rloo_trainer.py"]
    assert r["read"] == "the file that defines GRPOTrainer" and r["excludes_truncated"] is True
    assert r["logprobs_mode"] == ["trainer/utils.py:1: LLM(model, logprobs_mode='processed_logprobs')"]


def test_without_grpotrainers_file_every_file_naming_the_metric_must_match(tmp_path):
    bad = _variant("delta = torch.abs(old_per_token_logps - sampling_per_token_logps)",
                   "delta = torch.exp(old_per_token_logps - sampling_per_token_logps)")
    base = _package(tmp_path, {"a.py": TRL_LIKE.replace("class GRPOTrainer", "class Mixin"),
                               "b.py": bad.replace("class GRPOTrainer", "class Other")})
    r = tp.logp_metric_source(base=base)
    assert r["verdict"] == "not_matched" and r["read"] == "every file that names the metric"
    assert r["others"][0]["file"] == "b.py" and r["others"][0]["verdict"] == "not_matched"


@pytest.mark.parametrize("files, verdict", [
    ({"x.py": "print('nothing here')\n"}, "not_found"),
    ({"x.py": "KEY = 'sampling/sampling_logp_difference/mean'\n"}, "not_found"),
    ({"x.py": "def f(:\n    'sampling/sampling_logp_difference/mean'\n"}, "error"),
])
def test_a_source_without_the_computation_or_one_that_does_not_parse_is_a_record_not_an_exception(tmp_path, files,
                                                                                                  verdict):
    assert tp.logp_metric_source(base=_package(tmp_path, files))["verdict"] == verdict


def test_an_uninstalled_package_is_not_found():
    r = tp.logp_metric_source(package="trl_is_not_installed_here")
    assert r["verdict"] == "not_found" and "not installed" in r["error"] and r["version"] is None
    assert tp.vllm_logprobs_evidence(base=None)["checked"] is False


def test_the_vllm_evidence_lists_the_lines_it_finds_and_decides_nothing(tmp_path):
    runner = tmp_path / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("x = 1\napply_grammar_bitmask(scheduler_output, logits)\nout = self.sampler(logits)\n")
    ev = tp.vllm_logprobs_evidence(base=tmp_path / "vllm")
    assert ev["checked"] is False and ev["files"]["v1/sample/sampler.py"] is None
    assert ev["files"]["v1/worker/gpu_model_runner.py"] == ["2: apply_grammar_bitmask(scheduler_output, logits)",
                                                           "3: out = self.sampler(logits)"]


# ------------------------------------------------------------------------------------------ the engine's backend
def _engine(backend: str):
    return SimpleNamespace(vllm_config=SimpleNamespace(structured_outputs_config=SimpleNamespace(backend=backend)))


class FakeLLM:
    """vllm.LLM's shape for the pin: keyword arguments, and llm_engine.vllm_config.structured_outputs_config."""

    def __init__(self, model, **kw):
        self.model, self.kw = model, kw
        self.llm_engine = _engine((kw.get("structured_outputs_config") or {}).get("backend", "auto"))


class OldLLM:
    """An LLM() without the keyword: a TypeError on it."""

    def __init__(self, model, gpu_memory_utilization=0.3):
        self.model, self.llm_engine = model, _engine("auto")


def test_the_pin_asks_every_llm_for_xgrammar_and_restores_the_class():
    pin = tp.BackendPin(FakeLLM, "xgrammar")
    try:
        llm = FakeLLM("m", gpu_memory_utilization=0.3)
        assert llm.kw == {"gpu_memory_utilization": 0.3, "structured_outputs_config": {"backend": "xgrammar"}}
        assert pin.record["pin_accepted"] is True and pin.record["calls"] == 1
        assert pin.record["patched"].endswith("FakeLLM.__init__")
    finally:
        pin.restore()
    assert FakeLLM("m").kw == {}, "restored"


def test_an_llm_without_the_keyword_is_built_again_without_it_as_junk_does():
    pin = tp.BackendPin(OldLLM, "xgrammar")
    try:
        llm = OldLLM("m", gpu_memory_utilization=0.25)
        assert llm.model == "m" and pin.record["pin_accepted"] is False and "TypeError" in pin.record["pin_error"]
    finally:
        pin.restore()


def test_a_caller_that_passes_its_own_config_keeps_it_and_the_record_says_so():
    pin = tp.BackendPin(FakeLLM, "xgrammar")
    try:
        llm = FakeLLM("m", structured_outputs_config={"backend": "guidance"})
        assert llm.kw["structured_outputs_config"] == {"backend": "guidance"}
        assert pin.record["pin_accepted"] is None and "guidance" in pin.record["set_by_caller"]
        rec = tp.engine_record(SimpleNamespace(llm=llm), pin)
        assert (rec["backend"], rec["applied"], rec["set_by_caller"]) == ("guidance", False, "{'backend': 'guidance'}")
    finally:
        pin.restore()


def test_restoring_an_inherited_init_falls_back_to_the_parents():
    class Child(FakeLLM):
        pass
    pin = tp.BackendPin(Child, "xgrammar")
    assert "__init__" in vars(Child)
    pin.restore()
    assert "__init__" not in vars(Child) and Child.__init__ is FakeLLM.__init__


def test_no_class_or_no_vllm_is_a_record_never_an_exception():
    pin = tp.BackendPin(None, "xgrammar", error="vllm.LLM: ModuleNotFoundError")
    rec = tp.engine_record(SimpleNamespace(), pin)
    assert (rec["requested"], rec["pin_accepted"], rec["backend"], rec["applied"]) == ("xgrammar", None, "unknown", None)
    assert "ModuleNotFoundError" in rec["error"]
    pin.restore()
    assert tp.BackendPin.for_vllm("xgrammar").record["error"]          # no vLLM on the laptop: recorded


def test_the_engine_is_found_wherever_the_trainer_keeps_it_and_its_backend_read_back():
    pin = tp.BackendPin(FakeLLM, "xgrammar")
    try:
        llm = FakeLLM("m")
        trainer = SimpleNamespace(args=SimpleNamespace(seed=1), callbacks=[object()],
                                  vllm_generation=SimpleNamespace(cfg={"x": 1}, llm=llm))
        trainer.self_ref = trainer                                     # cycles are fine
        assert tp.find_engine(trainer, FakeLLM) == (llm, "trainer.vllm_generation.llm")
        rec = tp.engine_record(trainer, pin)
        assert rec == {"requested": "xgrammar", "pin_accepted": True, "backend": "xgrammar",
                       "source": "llm_engine.vllm_config.structured_outputs_config.backend", "applied": True,
                       "engine": "trainer.vllm_generation.llm", "calls": 1, "patched": pin.record["patched"]}
        listed = SimpleNamespace(engines=[None, llm])
        assert tp.find_engine(listed, FakeLLM) == (llm, "trainer.engines[1]")
        deep = SimpleNamespace(a=SimpleNamespace(b=SimpleNamespace(c=SimpleNamespace(llm=llm))))
        assert tp.find_engine(deep, FakeLLM) == (None, None), "beyond max_depth"
        assert tp.engine_record(SimpleNamespace(), pin)["backend"] == "unknown"
    finally:
        pin.restore()


def test_the_walk_never_enters_the_model_or_the_accelerator():
    class Model:
        pass
    Model.__module__ = "torch.nn.modules.module"
    model = Model()
    model.hidden = FakeLLM("m")
    assert tp.find_engine(SimpleNamespace(model=model), FakeLLM) == (None, None)


# ------------------------------------------------------------------------------------------ D-021's source checks
# A synthetic install, written for these tests: the statements each check looks for, in TRL 1.17.1's / vLLM 0.32's /
# PEFT 0.21's shapes, with other local names (chunk bounds `first`/`last`, `feed`, `allowed`, `targets`, `res`, ...).
GRPO = '''
import math

import torch

from .utils import add_fused_lm_head, use_adapter


class GRPOTrainer:
    def __init__(self, model, args):
        self.args = args
        self.model = model
        self.ref_model = None
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.beta = args.beta
        self.loss_type = args.loss_type
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_mode = args.vllm_importance_sampling_mode
        self.vllm_importance_sampling_clip_min = args.vllm_importance_sampling_clip_min
        self.vllm_importance_sampling_clip_max = args.vllm_importance_sampling_clip_max
        add_fused_lm_head(self.model.get_base_model(), temperature=self.temperature, outputs=("log_probs", "entropy"))
        if self.use_vllm:
            filtered_sampling = self.top_p < 1.0 or self.top_k > 0 or (self.min_p or 0.0) > 0.0
            return_sampling_mask = self.vllm_importance_sampling_correction and filtered_sampling
            self.vllm_generation = VLLMGeneration(self.model, return_sampling_mask=return_sampling_mask)

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None,
                                           compute_entropy=False, sampling_mask=None):
        batch_size = batch_size or input_ids.size(0)
        chunks, entropy_chunks = [], []
        for first in range(0, input_ids.size(0), batch_size):
            last = min(first + batch_size, input_ids.size(0))
            ids = input_ids[first:last]
            att = attention_mask[first:last]
            feed = {"input_ids": ids, "attention_mask": att}
            if sampling_mask is not None:
                allowed = sampling_mask.new_full((*ids.shape, sampling_mask.size(-1)), -1)
                allowed[:, -logits_to_keep:] = sampling_mask[first:last]
                feed["support_ids"] = allowed
            targets = ids.masked_fill(att == 0, -100)
            targets[:, :-logits_to_keep] = -100
            with torch.autocast("cuda"):
                res = model(**feed, labels=targets, fused_lm_head=True)
            chunks.append(res.log_probs[:, -logits_to_keep:])
            if compute_entropy:
                entropy_chunks.append(res.entropy[:, -logits_to_keep:])
        logps = torch.cat(chunks, dim=0)
        return logps, (torch.cat(entropy_chunks, dim=0) if compute_entropy else None), None

    def _generate_and_score_completions(self, inputs):
        mode = "train"
        prompt_ids, completion_ids, sampling_per_token_logps, sampling_mask = self._generate(inputs)
        completion_mask = (completion_ids != self.pad_id).long()
        ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.ones_like(ids)
        logits_to_keep = completion_ids.size(1)
        with torch.no_grad():
            if self.use_vllm and self.vllm_importance_sampling_correction:
                old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                    self.model, ids, attention_mask, logits_to_keep, sampling_mask=sampling_mask)
            else:
                old_per_token_logps = None
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = completion_mask
                gap = (old_per_token_logps - sampling_per_token_logps) * mask
                gap = torch.nan_to_num(gap, nan=0.0)
                per_sequence = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if per_sequence:
                    summed = gap.sum(dim=-1, keepdim=True)
                    log_weight = summed
                else:
                    log_weight = gap
                weight = torch.exp(log_weight)
                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    weight = torch.clamp(weight, min=self.vllm_importance_sampling_clip_min,
                                         max=self.vllm_importance_sampling_clip_max)
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    lo = (self.vllm_importance_sampling_clip_min
                          if self.vllm_importance_sampling_clip_min is not None else -math.inf)
                    hi = (self.vllm_importance_sampling_clip_max
                          if self.vllm_importance_sampling_clip_max is not None else math.inf)
                    weight = weight.masked_fill((weight < lo) | (weight > hi), value=0.0)
                else:
                    raise ValueError(self.vllm_importance_sampling_mode)
            if self.beta != 0.0:
                unwrapped = self.accelerator.unwrap_model(self.model)
                with use_adapter(unwrapped, adapter_name=None):
                    ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                        self.model, ids, attention_mask, logits_to_keep, sampling_mask=sampling_mask)
            else:
                ref_per_token_logps = None
        if self.use_vllm and self.vllm_importance_sampling_correction:
            if per_sequence:
                flat = weight.flatten()
            else:
                flat = weight[mask.bool()]
            mean_weight = torch.mean(flat) if flat.numel() > 0 else torch.tensor(0.0)
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_weight).nanmean().item())
        result = {"completion_ids": completion_ids, "completion_mask": completion_mask}
        if old_per_token_logps is not None:
            result["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            result["importance_sampling_ratio"] = weight
        if ref_per_token_logps is not None:
            result["ref_per_token_logps"] = ref_per_token_logps
        return result

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("no outputs here")
        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        logits_to_keep = inputs["completion_ids"].size(1)
        mask = inputs["completion_mask"]
        per_token_logps, entropies, _ = self._get_per_token_logps_and_entropies(
            model, ids, attention_mask, logits_to_keep, compute_entropy=True, sampling_mask=inputs.get("sampling_mask"))
        old_per_token_logps = inputs.get("old_per_token_logps")
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (torch.exp(ref_per_token_logps - per_token_logps)
                            - (ref_per_token_logps - per_token_logps) - 1)
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1
        per_token_loss = -coef_1 * inputs["advantages"].unsqueeze(1)
        if self.use_vllm and self.vllm_importance_sampling_correction and self.loss_type != "vespo":
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        return (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
'''

# D-021's registered logits form (the shape of TRL 0.2x's method), for path `fill`.
FILL_METHOD = '''
    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None,
                                           compute_entropy=False):
        batch_size = batch_size or input_ids.size(0)
        chunks = []
        for first in range(0, input_ids.size(0), batch_size):
            ids = input_ids[first:first + batch_size]
            att = attention_mask[first:first + batch_size]
            scores = model(input_ids=ids, attention_mask=att, logits_to_keep=logits_to_keep + 1).logits
            scores = scores[:, :-1, :]
            scores = scores / self.temperature
            chunks.append(selective_log_softmax(scores, ids[:, -logits_to_keep:]))
        return torch.cat(chunks, dim=0), None
'''

TRL_TREE = {
    "trainer/grpo_trainer.py": GRPO,
    "trainer/utils.py": '''
import types


def use_adapter(model, adapter_name):
    return model.disable_adapter() if adapter_name is None else model.adapter(adapter_name)


def add_fused_lm_head(model, temperature=1.0, cast_lm_head_to_fp32=False, outputs=("log_probs",)):
    plain_forward = model.forward

    def _fused_forward(self, *args, fused_lm_head=False, labels=None, shift_labels=None, support_ids=None, **kwargs):
        if not fused_lm_head:
            return plain_forward(*args, labels=labels, **kwargs)
        states = self.base_model(*args, **kwargs).last_hidden_state
        if shift_labels is None:
            states = states[:, :-1]
            labels = labels[:, 1:]
            if support_ids is not None:
                support_ids = support_ids[:, 1:]
        else:
            labels = shift_labels
        keep = labels != -100
        if support_ids is not None:
            support_ids = support_ids[keep]
        head = self.get_output_embeddings()
        return _ChunkedLogProbFunction.apply(states[keep], head.weight, None, labels[keep], temperature, outputs,
                                             support_ids)

    model.forward = types.MethodType(_fused_forward, model)
''',
    "trainer/base_config.py": '''
from dataclasses import dataclass, field


@dataclass
class _BaseConfig:
    use_liger_kernel: bool = field(default=False)
''',
    "trainer/grpo_config.py": '''
from dataclasses import dataclass, field

from .base_config import _BaseConfig


@dataclass
class GRPOConfig(_BaseConfig):
    temperature: float = field(default=1.0)
    top_p: float = field(default=1.0)
    top_k: int = field(default=0)
    min_p: float | None = field(default=None)
    beta: float = field(default=0.0)
    loss_type: str = "dapo"
    use_vllm: bool = field(default=False)
    vllm_mode: str = field(default="colocate")
    vllm_importance_sampling_correction: bool = field(default=True)
    vllm_importance_sampling_mode: str = field(default="sequence_mask")
    vllm_importance_sampling_clip_min: float | None = field(default=None)
    vllm_importance_sampling_clip_max: float | None = field(default=3.0)
    vllm_importance_sampling_cap: float | None = field(default=None)
    use_bias_correction_kl: bool = field(default=True)
''',
    "generation/vllm_generation.py": '''
class VLLMGeneration:
    def __init__(self, model, return_sampling_mask=False):
        self.return_sampling_mask = return_sampling_mask
        self.llm = LLM(model=model.name_or_path, logprobs_mode="processed_logprobs")

    def generate(self, prompts):
        outs = self.llm.generate(prompts)
        completion_ids = [o.token_ids for o in outs]
        all_sampling_mask = None
        if self.return_sampling_mask:
            all_sampling_mask = [o.sampling_mask for o in outs]
        sampling_mask = all_sampling_mask
        return completion_ids, sampling_mask
''',
    "kernels/chunked_logprob.py": '''
def normalizer(support_max, support_sum_exp):
    support_log_z = support_max + log(support_sum_exp)
    return support_log_z
''',
}

VLLM_TREE = {
    "v1/worker/gpu_model_runner.py": '''
class GPUModelRunner:
    def _sample(self, logits, spec_decode_metadata):
        if spec_decode_metadata is None:
            return self.sampler(logits=logits, sampling_metadata=self.input_batch.sampling_metadata)
        return self.rejection_sampler(spec_decode_metadata, logits)

    def sample_tokens(self, grammar_output):
        scheduler_output, logits, spec_decode_metadata = self.execute_model_state
        if grammar_output is not None:
            apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)
        out = self._sample(logits, spec_decode_metadata)
        return out
''',
    "v1/worker/gpu/model_runner.py": '''
class GPUModelRunner:
    def sample(self, hidden_states, input_batch, grammar_output):
        logits = self.model.compute_logits(hidden_states)
        if grammar_output is not None:
            self.structured_outputs_worker.apply_grammar_bitmask(logits, input_batch, grammar_output.grammar_bitmask)
        return self.sampler(logits, input_batch)
''',
    "v1/structured_output/utils.py": '''
def apply_grammar_bitmask(scheduler_output, grammar_output, input_batch, logits):
    bitmask = grammar_output.grammar_bitmask
    xgr.apply_token_bitmask_inplace(logits, bitmask, indices=None)
''',
    "v1/worker/gpu/structured_outputs.py": '''
class StructuredOutputsWorker:
    def apply_grammar_bitmask(self, logits, input_batch, grammar_bitmask):
        grid = (grammar_bitmask.shape[0],)
        _mask_kernel[grid](logits, grammar_bitmask)


@triton.jit
def _mask_kernel(logits_ptr, bitmask_ptr):
    tl.store(logits_ptr, -float("inf"), mask=tl.load(bitmask_ptr) == 0)
''',
    "v1/sample/sampler.py": '''
class Sampler:
    def sample(self, logits, sampling_metadata):
        logprobs_mode = self.logprobs_mode
        if sampling_metadata.all_greedy:
            processed_logprobs = None
            if logprobs_mode == "processed_logprobs":
                processed_logprobs = self.compute_logprobs(logits)
            return logits.argmax(dim=-1), processed_logprobs
        logits = logits / sampling_metadata.temperature
        sampled, processed_logprobs = self.topk_topp_sampler(logits, sampling_metadata.generators, None, None)
        return sampled, processed_logprobs

    @staticmethod
    def compute_logprobs(logits):
        return logits.log_softmax(dim=-1, dtype=torch.float32)
''',
    "v1/sample/ops/topk_topp_sampler.py": '''
class TopKTopPSampler:
    def forward_native(self, logits, generators, k, p):
        logits = apply_top_k_top_p(logits, k, p)
        returned = None
        if self.logprobs_mode == "processed_logprobs":
            returned = logits.log_softmax(dim=-1, dtype=torch.float32)
        return random_sample(logits.softmax(dim=-1), generators), returned
''',
    "v1/worker/gpu/sample/sampler.py": '''
class Sampler:
    def __call__(self, logits, input_batch):
        sampled, processed_logits = self.sample(logits, input_batch)
        if self.logprobs_mode in PROCESSED_LOGPROBS_MODES:
            logits = processed_logits
        return sampled, logits.log_softmax(dim=-1)
''',
    "config/model.py": 'PROCESSED_LOGPROBS_MODES: tuple = ("processed_logits", "processed_logprobs")\n',
    "config/vllm.py": '''
class VllmConfig:
    @property
    def use_v2_model_runner(self):
        chosen = envs.VLLM_USE_V2_MODEL_RUNNER
        if chosen is not None:
            return chosen
        if not HAS_TRITON:
            return False
        return True
''',
    "v1/structured_output/backend_xgrammar.py": '''
class XgrammarBackend:
    def __post_init__(self):
        info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.vocab_size)
        self.compiler = xgr.GrammarCompiler(info, max_threads=8, cache_enabled=True)
        self.num_speculative_tokens = 0

    def compile_grammar(self, request_type, grammar_spec, stop_token_ids=None):
        if request_type == StructuredOutputOptions.JSON:
            ctx = self.compiler.compile_json_schema(grammar_spec)
        elif request_type == StructuredOutputOptions.REGEX:
            ctx = self.compiler.compile_regex(grammar_spec)
        else:
            raise ValueError(request_type)
        matcher = xgr.GrammarMatcher(ctx, override_stop_tokens=list(stop_token_ids) if stop_token_ids else None,
                                     max_rollback_tokens=self.num_speculative_tokens)
        return Grammar(matcher=matcher, ctx=ctx)
''',
    "v1/structured_output/__init__.py": '''
class StructuredOutputManager:
    def grammar_init(self, request):
        if self.backend is None:
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            self.backend = XgrammarBackend(self.vllm_config, tokenizer=self.tokenizer, vocab_size=vocab_size)
        request.grammar = self._create_grammar(request)

    def _create_grammar(self, request):
        request_type, grammar_spec = request.structured_output_key
        stops = request.sampling_params.all_stop_token_ids if request.sampling_params is not None else None
        return self.backend.compile_grammar(request_type, grammar_spec, stop_token_ids=stops)
''',
    "sampling_params.py": '''
class SamplingParams:
    def update_from_generation_config(self, generation_config, eos_token_id=None):
        if eos_token_id is not None:
            self._all_stop_token_ids.add(eos_token_id)
''',
}

PEFT_TREE = {
    "tuners/tuners_utils.py": '''
from peft.utils import INCLUDE_LINEAR_LAYERS_SHORTHAND


def _maybe_include_all_linear_layers(peft_config, model):
    if not (isinstance(peft_config.target_modules, str)
            and peft_config.target_modules.lower() == INCLUDE_LINEAR_LAYERS_SHORTHAND):
        return peft_config
    names = {name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    excluded = set()
    if isinstance(model, PreTrainedModel):
        head = model.get_output_embeddings()
        if head is not None:
            head_name = next(name for name, module in model.named_modules() if module is head)
            excluded.add(head_name)
    names -= excluded
    peft_config.target_modules = names
    return peft_config


class BaseTuner:
    def inject_adapter(self, model, adapter_name):
        peft_config = _maybe_include_all_linear_layers(self.peft_config[adapter_name], model)
        return peft_config
''',
    "utils/constants.py": 'INCLUDE_LINEAR_LAYERS_SHORTHAND = "all-linear"\n',
}

TREES = {"trl": TRL_TREE, "vllm": VLLM_TREE, "peft": PEFT_TREE}
# The settings these checks read, as configs/grpo_smoke_v2_grammar.yaml resolves them (the rest fall to defaults).
ARGS = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "use_vllm": True, "vllm_mode": "colocate", "beta": 0.04,
        "loss_type": "dapo", "num_iterations": 1, "per_device_train_batch_size": 4}


def _install(tmp_path: Path, edits=(), drop=()) -> dict[str, Path]:
    """The synthetic tree under tmp_path, with `edits` [(package, file, old, new)] applied and `drop` [(package, file)]
    left out: {package: its directory}."""
    files = {(pkg, rel): text for pkg, tree in TREES.items() for rel, text in tree.items()}
    for pkg, rel, old, new in edits:
        assert old in files[(pkg, rel)], (pkg, rel, old)
        files[(pkg, rel)] = files[(pkg, rel)].replace(old, new)
    for (pkg, rel), text in files.items():
        if (pkg, rel) in drop:
            continue
        p = tmp_path / pkg / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text.lstrip("\n"))
    return {pkg: tmp_path / pkg for pkg in TREES}


def _checks(tmp_path: Path, edits=(), drop=(), args=ARGS):
    bases = _install(tmp_path, edits, drop)
    return tp.source_checks(trl_base=bases["trl"], vllm_base=bases["vllm"], peft_base=bases["peft"], args=args)


def _grpo(old: str, new: str) -> tuple[str, str, str, str]:
    return ("trl", "trainer/grpo_trainer.py", old, new)


def test_a_tree_shaped_like_the_installed_one_matches_every_check_and_names_the_support_path(tmp_path):
    res = _checks(tmp_path)
    assert {n: res[n]["verdict"] for n in tp.GATING + tp.RECORDED} == dict.fromkeys(tp.GATING + tp.RECORDED, "match")
    assert (res["verdict"], res["path"], res["failed"], res["gating"]) == ("match", "support_ids", [], list(tp.GATING))
    assert res["args_given"] is True and res["seconds"] >= 0
    for name in tp.GATING + tp.RECORDED:                 # the record schema, and the sha256 of the bytes read
        rec = res[name]
        assert {"check", "verdict", "package", "file", "sha256", "lines", "missing"} <= set(rec), name
        raw = (tmp_path / rec["package"] / rec["file"]).read_bytes()
        assert rec["sha256"] == hashlib.sha256(raw).hexdigest() and rec["missing"] == [], name
    t1, t2 = res["T1"], res["T2"]
    assert [(c["target"], c["model"]) for c in t1["calls"]] == [
        ("old_per_token_logps", "self.model"), ("ref_per_token_logps", "self.model"), ("per_token_logps", "model")]
    assert any("use_adapter" in w for w in t1["calls"][1]["with"]) and "torch.no_grad()" in t1["calls"][0]["with"]
    assert t1["reads"]["use_liger_kernel"] == {"value": False, "from": "default (trainer/base_config.py:6)"}
    assert t2["form"] == "support_ids" and t2["form_missing"]["support_ids"] == []
    assert t2["trl_sampling_mask"]["none_at_args"] is True and t2["trl_sampling_mask"]["filtered_sampling"] is False
    assert t2["trl_sampling_mask"]["reads"]["min_p"]["from"].startswith("default (trainer/grpo_config.py:")
    assert t2["ordinary_forward"]["found"] is True and t2["autocast"] == ["torch.autocast('cuda')"]
    assert "trl/trainer/utils.py" in t2["files"] and "support_log_z" in str(t2["kernel"]["lines"])
    lines = "\n".join(t2["lines"].values())
    for piece in ("allowed[:, -logits_to_keep:] = sampling_mask[first:last]", "targets[:, :-logits_to_keep] = -100",
                  "fused_lm_head=True", "chunks.append(res.log_probs[:, -logits_to_keep:])"):
        assert piece in lines, piece


def test_t3_and_t4_record_the_weight_the_bound_and_the_kl_with_where_each_setting_came_from(tmp_path):
    res = _checks(tmp_path)
    t3, t4 = res["T3"], res["T4"]
    assert (t3["level"], t3["mode"], t3["branch"], t3["ratio_mean_over"]) == ("sequence", "sequence_mask", "mask",
                                                                              "sequences")
    assert (t3["clip_min"], t3["clip_max"], t3["nan_to_zero"], t3["applied_in_loss"]) == (None, 3.0, True, True)
    assert t3["reads"]["vllm_importance_sampling_mode"]["from"].startswith("default (trainer/grpo_config.py:")
    assert t4["kl"] == "torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1"
    assert t4["bias_correction"]["times"] == "coef_1" and t4["kl_on_at_args"] is True
    assert t4["use_bias_correction_kl"]["value"] is True and t4["beta"] == {"value": 0.04, "from": "args"}
    off = _checks(tmp_path / "off", args={**ARGS, "use_bias_correction_kl": False})["T4"]
    assert off["verdict"] == "match" and off["use_bias_correction_kl"] == {"value": False, "from": "args"}
    trunc = _checks(tmp_path / "trunc", args={**ARGS, "vllm_importance_sampling_mode": "sequence_truncate"})["T3"]
    assert (trunc["verdict"], trunc["branch"], trunc["ratio_mean_over"]) == ("match", "truncate", "sequences")


def test_the_registered_logits_form_is_path_fill(tmp_path):
    start = GRPO.index("    def _get_per_token_logps_and_entropies")
    end = GRPO.index("    def _generate_and_score_completions")
    res = _checks(tmp_path, edits=[_grpo(GRPO[start:end], FILL_METHOD.lstrip("\n") + "\n")])
    assert (res["T2"]["verdict"], res["path"], res["verdict"]) == ("match", "fill", "match")
    assert res["T2"]["form_missing"]["fill"] == [] and res["T2"]["form_missing"]["support_ids"]
    no_gate = _checks(tmp_path / "old", edits=[            # an older TRL: no sampling-mask construct, read from args
        _grpo(GRPO[start:end], FILL_METHOD.lstrip("\n") + "\n"),
        _grpo("            return_sampling_mask = self.vllm_importance_sampling_correction and "
              "filtered_sampling\n", ""),
        _grpo(", return_sampling_mask=return_sampling_mask", "")])
    assert (no_gate["T2"]["verdict"], no_gate["path"]) == ("match", "fill")
    assert no_gate["T2"]["trl_sampling_mask"]["expression"] == tp.FILTERED_FROM_ARGS
    top_k = _checks(tmp_path / "topk", edits=[_grpo(GRPO[start:end], FILL_METHOD.lstrip("\n") + "\n")],
                    args={**ARGS, "top_k": 50})
    assert top_k["T2"]["verdict"] == "not_matched" and top_k["path"] is None


@pytest.mark.parametrize("old, new, why", [
    ("logits_to_keep=logits_to_keep + 1", "logits_to_keep=logits_to_keep", "logits_to_keep + 1"),
    ("            scores = scores[:, :-1, :]\n", "", "last row is not dropped"),
    ("            scores = scores / self.temperature\n", "", "not divided by self.temperature"),
    ("            scores = scores / self.temperature\n"
     "            chunks.append(selective_log_softmax(scores, ids[:, -logits_to_keep:]))",
     "            chunks.append(selective_log_softmax(scores, ids[:, -logits_to_keep:]))\n"
     "            scores = scores / self.temperature",
     "before the row drop or the temperature"),
], ids=["no-plus-one", "no-drop", "no-temperature", "softmax-first"])
def test_a_logits_form_that_misses_a_step_is_neither_path(tmp_path, old, new, why):
    start, end = GRPO.index("    def _get_per_token_logps_and_entropies"), GRPO.index("    def _generate_and_score")
    fill = FILL_METHOD.lstrip("\n") + "\n"
    assert old in fill
    res = _checks(tmp_path, edits=[_grpo(GRPO[start:end], fill.replace(old, new))])
    assert res["T2"]["verdict"] == "not_matched" and res["path"] is None and "T2" in res["failed"]
    assert any(m.startswith("fill form:") and why in m for m in res["T2"]["missing"]), res["T2"]["missing"]


GRPO_VARIANTS = [   # (check, old, new, what the missing list says)
    ("T1", "old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(",
     "old_per_token_logps, _, _ = self._other_logps(", "old_per_token_logps at line"),
    ("T1", "                    self.model, ids, attention_mask, logits_to_keep, sampling_mask=sampling_mask)\n"
           "            else:\n                old_per_token_logps = None",
     "                    self.ref_model, ids, attention_mask, logits_to_keep, sampling_mask=sampling_mask)\n"
     "            else:\n                old_per_token_logps = None", "computed on `self.ref_model`"),
    ("T1", "                ref_per_token_logps = None", "                ref_per_token_logps = old_per_token_logps",
     "ref_per_token_logps at line"),
    ("T1", "per_token_logps, entropies, _ = self._get_per_token_logps_and_entropies(",
     "per_token_logps, entropies, _ = self._fast_logps(", "per_token_logps is never the first value"),
    ("T1", '            result["ref_per_token_logps"] = ref_per_token_logps\n', "            pass\n",
     "never stores ref_per_token_logps"),
    ("T1", "        return self._compute_loss(model, inputs)",
     "        if self.use_liger_loss:\n            return self._liger_loss(model, inputs)\n"
     "        return self._compute_loss(model, inputs)", "not only self._compute_loss"),
    ("T2", "for first in range(0, input_ids.size(0), batch_size):",
     "for first in reversed(range(0, input_ids.size(0), batch_size)):", "row chunks in order"),
    ("T2", 'feed = {"input_ids": ids, "attention_mask": att}', 'feed = {"input_ids": input_ids, "attention_mask": att}',
     "does not read this chunk's rows"),
    ("T2", "targets = ids.masked_fill(att == 0, -100)", "targets = ids.clone()", "masked_fill(att == 0, -100)"),
    ("T2", "            targets[:, :-logits_to_keep] = -100\n", "", "targets[:, :-logits_to_keep] = -100"),
    ("T2", "allowed[:, -logits_to_keep:] = sampling_mask[first:last]",
     "allowed[:, -logits_to_keep - 1:-1] = sampling_mask[first:last]", "completion positions"),
    ("T2", "sampling_mask.size(-1)), -1)", "sampling_mask.size(-1)), 0)", "-1 off the completion"),
    ("T2", '                feed["support_ids"] = allowed\n', "", "support_ids is not passed"),
    ("T2", "res = model(**feed, labels=targets, fused_lm_head=True)", "res = model(**feed, labels=targets)",
     "fused_lm_head=True"),
    ("T2", "res = model(**feed, labels=targets, fused_lm_head=True)",
     "res = model(**feed, labels=targets, shift_labels=targets, fused_lm_head=True)", "shift_labels"),
    ("T2", "chunks.append(res.log_probs[:, -logits_to_keep:])", "chunks.append(res.log_probs[:, :logits_to_keep])",
     "append(res.log_probs[:, -logits_to_keep:])"),
    ("T2", "            if compute_entropy:\n                entropy_chunks.append(",
     "            res = model(**feed, labels=targets, fused_lm_head=True)\n"
     "            if compute_entropy:\n                entropy_chunks.append(", "one per micro-batch"),
    ("T2", "logps = torch.cat(chunks, dim=0)", "logps = torch.cat(chunks, dim=1)", "torch.cat(chunks, dim=0)"),
    ("T2", "add_fused_lm_head(self.model.get_base_model(), temperature=self.temperature,",
     "add_fused_lm_head(self.model.get_base_model(),", "temperature=self.temperature"),
    ("T3", "gap.sum(dim=-1, keepdim=True)", "gap.mean(dim=-1, keepdim=True)", "not the sum of `gap`"),
    ("T3", "                mask = completion_mask\n", "                mask = attention_mask\n",
     "not built from completion_mask"),
    ("T3", "                flat = weight.flatten()", "                flat = weight[mask.bool()]",
     "not the flattened (B, 1) weights"),
    ("T3", "weight = weight.masked_fill((weight < lo) | (weight > hi), value=0.0)",
     "weight = torch.clamp(weight, min=lo, max=hi)", "clamps the weight"),
    ("T3", "(weight > hi), value=0.0)", "(weight > hi), value=1.0)", "not set to 0"),
    ("T3", "(weight < lo) | (weight > hi)", "(weight > hi)", "below clip_min or above clip_max"),
    ("T3", '            result["importance_sampling_ratio"] = weight\n', "            pass\n",
     "does not carry `weight`"),
    ("T3", '            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]\n', "            pass\n",
     "does not multiply per_token_loss"),
    ("T4", "(torch.exp(ref_per_token_logps - per_token_logps)", "(torch.exp(per_token_logps - ref_per_token_logps)",
     "no single KL"),
    ("T4", 'ref_per_token_logps = inputs["ref_per_token_logps"]', "ref_per_token_logps = self._ref_logps(inputs)",
     "does not read inputs['ref_per_token_logps']"),
    ("T4", "                per_token_kl = per_token_kl * coef_1\n",
     "                per_token_kl = per_token_kl * coef_1\n            per_token_kl = per_token_kl.clamp(max=10.0)\n",
     "other than the bias correction"),
    ("T4", "            per_token_loss = per_token_loss + self.beta * per_token_kl\n", "            pass\n",
     "does not add self.beta * per_token_kl"),
]


@pytest.mark.parametrize("check, old, new, why", GRPO_VARIANTS,
                         ids=[f"{v[0]}-{i}" for i, v in enumerate(GRPO_VARIANTS)])
def test_every_other_trainer_source_fails_its_check_and_says_what_differs(tmp_path, check, old, new, why):
    res = _checks(tmp_path, edits=[_grpo(old, new)])
    rec = res[check]
    assert rec["verdict"] == "not_matched" and any(why in m for m in rec["missing"]), rec["missing"]
    assert check in res["failed"] and res["verdict"] == "not_matched"
    if check == "T2":
        assert res["path"] is None


OTHER_VARIANTS = [   # (check, package, file, old, new, what the missing list says)
    ("T2", "trl", "trainer/utils.py", "                support_ids = support_ids[:, 1:]\n",
     "                support_ids = support_ids\n", "does not shift support_ids"),
    ("T2", "trl", "trainer/utils.py", "temperature, outputs,\n" + " " * 45 + "support_ids)", "temperature, outputs)",
     "support_ids is not passed to _ChunkedLogProbFunction.apply"),
    ("T2", "trl", "trainer/utils.py", "    model.forward = types.MethodType(_fused_forward, model)\n", "",
     "does not bind _fused_forward"),
    ("T2", "trl", "generation/vllm_generation.py",
     "        all_sampling_mask = None\n        if self.return_sampling_mask:\n            all_sampling_mask = ",
     "        all_sampling_mask = None\n        if True:\n            all_sampling_mask = ",
     "outside `if self.return_sampling_mask:`"),
    ("T5", "peft", "tuners/tuners_utils.py", "    names -= excluded\n", "", "not subtracted"),
    ("T5", "peft", "utils/constants.py", '"all-linear"', '"all_linear"', "'all_linear', not 'all-linear'"),
    ("T5", "peft", "tuners/tuners_utils.py",
     "        peft_config = _maybe_include_all_linear_layers(self.peft_config[adapter_name], model)\n", "",
     "never called"),
    ("T5", "peft", "tuners/tuners_utils.py", "if module is head", "if module is not head",
     "module name is not found"),
    ("V1", "vllm", "v1/worker/gpu_model_runner.py",
     "            apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)\n"
     "        out = self._sample(logits, spec_decode_metadata)\n",
     "            pass\n        out = self._sample(logits, spec_decode_metadata)\n"
     "        apply_grammar_bitmask(scheduler_output, grammar_output, self.input_batch, logits)\n",
     "does not sample from `logits` after the bitmask"),
    ("V1", "vllm", "v1/worker/gpu/model_runner.py", "apply_grammar_bitmask(logits, input_batch,",
     "apply_grammar_bitmask(hidden_states, input_batch,", "not to `logits`"),
    ("V1", "vllm", "v1/sample/sampler.py", 'if logprobs_mode == "processed_logprobs":',
     'if logprobs_mode == "processed_logits":', "under the processed mode"),
    ("V1", "vllm", "v1/worker/gpu/structured_outputs.py", '-float("inf")', "0.0", "does not write -inf"),
    ("V1", "trl", "generation/vllm_generation.py", 'logprobs_mode="processed_logprobs"', 'logprobs_mode="raw_logprobs"',
     "not processed_logprobs"),
    ("V2", "vllm", "v1/structured_output/backend_xgrammar.py", "(self.tokenizer, vocab_size=self.vocab_size)",
     "(self.tokenizer)", "vocab_size=...) is not built"),
    ("V2", "vllm", "v1/structured_output/backend_xgrammar.py",
     "override_stop_tokens=list(stop_token_ids) if stop_token_ids else None,\n" + " " * 37, "",
     "override_stop_tokens"),
    ("V2", "vllm", "v1/structured_output/__init__.py", "stop_token_ids=stops)", "stop_token_ids=None)",
     "stop_token_ids = request.sampling_params.all_stop_token_ids"),
]


@pytest.mark.parametrize("check, pkg, rel, old, new, why", OTHER_VARIANTS,
                         ids=[f"{v[0]}-{i}" for i, v in enumerate(OTHER_VARIANTS)])
def test_every_other_library_source_fails_its_check_and_only_t1_to_t5_gate(tmp_path, check, pkg, rel, old, new, why):
    res = _checks(tmp_path, edits=[(pkg, rel, old, new)])
    rec = res[check]
    assert rec["verdict"] == "not_matched" and any(why in m for m in rec["missing"]), rec["missing"]
    if check in tp.GATING:
        assert check in res["failed"] and res["verdict"] == "not_matched"
    else:
        assert rec["gating"] is False and res["failed"] == [] and res["verdict"] == "match", "V1 and V2 are recorded"


@pytest.mark.parametrize("args, check, why", [
    ({**ARGS, "use_liger_kernel": True}, "T1", "use_liger_kernel is true"),
    ({**ARGS, "top_k": 50}, "T2", "TRL builds its own sampling_mask"),
    ({**ARGS, "top_p": 0.95}, "T2", "TRL builds its own sampling_mask"),
    ({**ARGS, "min_p": 0.05}, "T2", "TRL builds its own sampling_mask"),
    ({**ARGS, "vllm_importance_sampling_mode": "token_mask"}, "T3", "not a sequence mode"),
    ({**ARGS, "vllm_importance_sampling_correction": False}, "T3", "IS correction is off"),
    ({**ARGS, "vllm_importance_sampling_cap": 2.0}, "T3", "vllm_importance_sampling_cap is set"),
    ({**ARGS, "loss_type": "vespo"}, "T3", "does not apply the weight"),
    (None, "T1", "args not given"),
    (None, "T2", "args not given"),
    (None, "T3", "args not given"),
], ids=["liger", "top_k", "top_p", "min_p", "token-mode", "no-correction", "cap", "vespo", "no-args-T1",
        "no-args-T2", "no-args-T3"])
def test_the_run_time_halves_fail_closed_on_the_args(tmp_path, args, check, why):
    res = _checks(tmp_path, args=args)
    rec = res[check]
    assert rec["verdict"] == "not_matched" and any(why in m for m in rec["missing"]), rec["missing"]
    assert check in res["failed"] and res["verdict"] == "not_matched"
    if args is None:
        assert res["args_given"] is False and res["T4"]["verdict"] == "match", "T4 records the args, it does not gate"


def test_an_absent_package_or_file_is_not_found_and_a_file_that_does_not_parse_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "package_dir", lambda name: None)       # the laptop: none of the three is installed
    bare = tp.source_checks(args=ARGS)
    assert {n: bare[n]["verdict"] for n in tp.GATING + tp.RECORDED} == dict.fromkeys(tp.GATING + tp.RECORDED,
                                                                                     "not_found")
    assert (bare["verdict"], bare["path"], bare["failed"]) == ("not_matched", None, list(tp.GATING))
    assert "trl is not installed here" in bare["T1"]["error"] and bare["packages"]["trl"]["root"] is None
    gone = _checks(tmp_path / "gone", drop=[("trl", "trainer/utils.py")])
    assert gone["T2"]["verdict"] == "not_found" and "trainer/utils.py" in gone["T2"]["error"]
    assert gone["failed"] == ["T2"] and gone["T1"]["verdict"] == "match"
    broken = _checks(tmp_path / "broken", edits=[_grpo("class GRPOTrainer:", "class GRPOTrainer(:")])
    assert {n: broken[n]["verdict"] for n in ("T1", "T2", "T3", "T4")} == dict.fromkeys(("T1", "T2", "T3", "T4"),
                                                                                         "error")
    assert "SyntaxError" in broken["T1"]["error"] and broken["failed"] == ["T1", "T2", "T3", "T4"]
    assert broken["T5"]["verdict"] == "match" and broken["V1"]["verdict"] == "match"


def test_a_check_that_breaks_is_an_error_record_and_a_moved_file_is_found_by_what_it_defines(tmp_path, monkeypatch):
    def boom(c, args):
        raise RuntimeError("the check itself broke")
    what, main, _ = tp.CHECKS["T5"]
    monkeypatch.setitem(tp.CHECKS, "T5", (what, main, boom))
    res = _checks(tmp_path / "boom")
    assert res["T5"]["verdict"] == "error" and "the check itself broke" in res["T5"]["error"] and res["T5"]["trace"]
    assert res["failed"] == ["T5"] and res["verdict"] == "not_matched"
    monkeypatch.undo()
    bases = _install(tmp_path / "moved")
    (bases["trl"] / "trainer" / "grpo_trainer.py").rename(bases["trl"] / "trainer" / "grpo.py")
    moved = tp.source_checks(trl_base=bases["trl"], vllm_base=bases["vllm"], peft_base=bases["peft"], args=ARGS)
    assert moved["verdict"] == "match" and moved["T1"]["file"] == "trainer/grpo.py"


def test_v1_reads_both_runners_and_records_which_one_vllm_builds_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    v1 = _checks(tmp_path)["V1"]
    assert v1["verdict"] == "match" and v1["gating"] is False and v1["trl_logprobs_mode"] == ["processed_logprobs"]
    assert [m["method"] for m in v1["runners"]["v1"]["methods"]] == ["sample_tokens"]
    assert [m["method"] for m in v1["runners"]["v2"]["methods"]] == ["sample"]
    assert v1["runner_default"]["default"] == "v2" and v1["runner_default"]["env_VLLM_USE_V2_MODEL_RUNNER"] == "0"
    one = _checks(tmp_path / "one", drop=[("vllm", "v1/worker/gpu/model_runner.py")])["V1"]
    assert one["verdict"] == "match" and one["runners"]["v2"]["absent"] is True
    v2 = _checks(tmp_path / "v2")["V2"]
    assert v2["tokenizer_info"]["kwargs"] == {"vocab_size": "self.vocab_size"}
    assert v2["compiler"]["kwargs"] == {"max_threads": 8, "cache_enabled": True}
    assert v2["matcher"]["kwargs"]["max_rollback_tokens"] == "self.num_speculative_tokens"
    assert v2["compile_regex"] == "self.compiler.compile_regex(grammar_spec)"
    assert v2["vocab_size_from"] == "self.vllm_config.model_config.get_vocab_size()"
    assert v2["stop_ids_from"].startswith("request.sampling_params.all_stop_token_ids")


# ------------------------------------------------------------------------------------------ T5 at run time
class Tuner:
    """Stands in for peft.tuners.tuners_utils.BaseTunerLayer."""


class Plain:
    pass


class LoraLinear(Tuner):
    pass


class PeftLike:
    """The parts of a PEFT model lm_head_record reads."""

    def __init__(self, head=None, modules=None, params=None, targets=None):
        self.head = head if head is not None else Plain()
        self.modules = modules if modules is not None else [("base_model.model.model.layers.0.self_attn.q_proj",
                                                             LoraLinear()), ("base_model.model.lm_head", self.head)]
        self.params = params if params is not None else [
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
             SimpleNamespace(requires_grad=True)),
            ("base_model.model.model.embed_tokens.weight", SimpleNamespace(requires_grad=False))]
        self.peft_config = {"default": SimpleNamespace(target_modules=targets if targets is not None else
                                                       {"q_proj", "k_proj"}, modules_to_save=None)}

    def get_output_embeddings(self):
        return self.head

    def named_modules(self):
        return iter(self.modules)

    def named_parameters(self):
        return iter(self.params)


def test_lm_head_record_matches_a_peft_model_whose_output_embedding_is_untouched():
    rec = tp.lm_head_record(PeftLike(), tuner_layer=Tuner)
    assert rec["verdict"] == "match" and rec["missing"] == [] and rec["tuner_layers"] == 1
    assert rec["output_embedding_is_tuner_layer"] is False and rec["output_embedding"].endswith("Plain")
    assert rec["adapters"]["default"]["target_modules"] == {"count": 2, "lm_head": []}
    wrapped = tp.lm_head_record(SimpleNamespace(module=PeftLike()), tuner_layer=Tuner)
    assert wrapped["verdict"] == "match", "a wrapper's .module is read"


@pytest.mark.parametrize("model, why", [
    (PeftLike(head=LoraLinear()), "the output embedding is a tuner layer"),
    (PeftLike(modules=[("x.q_proj", LoraLinear()), ("base_model.model.lm_head", LoraLinear())]), "named *lm_head"),
    (PeftLike(params=[("base_model.model.lm_head.weight", SimpleNamespace(requires_grad=True))]),
     "trainable parameters named *lm_head*"),
    (PeftLike(targets={"q_proj", "lm_head"}), "target_modules names lm_head"),
    (PeftLike(modules=[("base_model.model.lm_head", Plain())]), "no tuner layer"),
], ids=["head-is-lora", "lora-named-lm_head", "trainable-lm_head", "targets-lm_head", "not-peft"])
def test_lm_head_record_fails_closed_on_a_lora_lm_head_or_a_model_that_is_not_peft(model, why):
    rec = tp.lm_head_record(model, tuner_layer=Tuner)
    assert rec["verdict"] == "not_matched" and any(why in m for m in rec["missing"]), rec["missing"]


def test_lm_head_record_is_an_error_without_peft_or_on_a_model_it_cannot_read():
    no_peft = tp.lm_head_record(PeftLike())                         # PEFT is not on the laptop
    assert no_peft["verdict"] == "error" and "peft" in no_peft["error"].lower()
    unreadable = tp.lm_head_record(object(), tuner_layer=Tuner)
    assert unreadable["verdict"] == "error" and "get_output_embeddings" in unreadable["error"]
