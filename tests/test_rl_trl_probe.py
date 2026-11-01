"""lofi_rl.rl.trl_probe: what a GRPO run reads from the installed TRL and vLLM (D-020, Outcome).

TRL 1.17.1 is not on the laptop, so these tests run the checks on stand-in sources and classes:
- the IS-metric check says "match" for a source shaped like TRL's (delta = |old - sampling|, masked by the completion
  mask, a plain mean, appended to sampling/sampling_logp_difference/mean) and "not_matched", naming what is
  missing, for every way a source can compute something else; logp_metric_source reads the file that defines
  GRPOTrainer, records its sha256 and lines, and never raises;
- BackendPin asks every LLM built for xgrammar as junk.build_llm does (a TypeError on the keyword builds it again
  without it), keeps a caller's own choice, and restores the class; find_engine walks a trainer to the engine,
  wherever it sits; engine_record reads its backend through junk.read_backend."""
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from lofi_rl.rl import trl_probe as tp

TRL_LIKE = '''
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
