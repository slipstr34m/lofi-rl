"""A laptop rehearsal of the GRPO smoke's data flow: lofi_rl.rl.train.train() runs against stand-ins for torch,
transformers, TRL, PEFT and datasets that call the reward the way TRL's GRPOTrainer does (4 prompts x 8
completions per step, trainer_state, callbacks, checkpoints every save_steps). Launch a, the relaunch from
checkpoint-5, and smoke_check's verdict on what they wrote. It proves the wiring (field fitting, prompts, the
reward's log, the step log, the resume step, run.json) but nothing about real TRL, which the cluster smoke tests."""
import dataclasses
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import kit.hf_cache
from helpers_rl import TRL_LIKE_SOURCE, fake_prompt, fake_verify
from lofi_rl.rl import config as rc
from lofi_rl.rl import smoke_check, train, trl_probe

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "grpo_smoke.yaml"
V2_GRAMMAR = ROOT / "configs" / "grpo_smoke_v2_grammar.yaml"
EOS = (151645, 151643)


class FakeLLM:
    """vllm.LLM's shape for the backend pin: keyword arguments in, llm_engine.vllm_config.structured_outputs_config
    out (what junk.read_backend reads)."""

    def __init__(self, model, **kw):
        self.model, self.kw = model, kw
        backend = (kw.get("structured_outputs_config") or {}).get("backend", "auto")
        self.llm_engine = SimpleNamespace(vllm_config=SimpleNamespace(
            structured_outputs_config=SimpleNamespace(backend=backend)))


def _plain_texts(k: int, batch: list[dict]) -> list[str]:
    texts = [f"theory=0.{i % 8 + 1} knobs=0.5 | {k}-{i}" for i in range(len(batch))]
    texts[1] = texts[0]                                         # a duplicate in group 0
    return texts


def _stack(missing: set[str], config: Path = SMOKE, texts=_plain_texts) -> dict[str, types.ModuleType]:
    cfg = rc.load_config(config)
    names = {k for p in cfg.profiles for k in rc.resolve(cfg, p)} | {"output_dir", "run_name"}
    GRPOConfig = dataclasses.make_dataclass("GRPOConfig", [(n, Any, None) for n in sorted(names - missing)],
                                            namespace={"to_dict": lambda self: dataclasses.asdict(self)})
    seen: dict[str, Any] = {"loads": 0, "trainers": []}

    class Tok:
        eos_token_id, padding_side, truncation_side = EOS[0], "right", "right"

        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            return " ".join(m["content"] for m in msgs) + " <assistant>"

        def __call__(self, text, add_special_tokens):
            return {"input_ids": list(range(len(text.split())))}

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True)

        @classmethod
        def from_pretrained(cls, path, dtype):
            seen["loads"] += 1
            seen["dtype"] = dtype
            return cls()

        def get_nb_trainable_parameters(self):
            return 33_030_144, 4_055_498_240

    class Trainer:
        def __init__(self, model, reward_funcs, args, train_dataset, processing_class, peft_config, callbacks):
            self.model, (self.reward,), self.args, self.rows = model, reward_funcs, args, list(train_dataset)
            self.tok, self.lora, self.callbacks = processing_class, peft_config, callbacks
            self.state = SimpleNamespace(global_step=0)
            # colocate mode: TRL builds its own vLLM engine, here on a helper object as a newer TRL might
            self.vllm_generation = SimpleNamespace(llm=sys.modules["vllm"].LLM(
                model="snapshot", gpu_memory_utilization=args.vllm_gpu_memory_utilization))
            seen["trainers"].append(self)

        def _each(self, hook, **kw):
            for cb in self.callbacks:
                getattr(cb, hook)(self.args, self.state, None, **kw)

        def train(self, resume_from_checkpoint=None):
            a = self.args
            if resume_from_checkpoint:
                self.state.global_step = json.loads((Path(resume_from_checkpoint) / "trainer_state.json").read_text())["global_step"]
            self._each("on_train_begin")
            prompts = a.per_device_train_batch_size * a.gradient_accumulation_steps // a.num_generations
            while self.state.global_step < a.max_steps:
                k = self.state.global_step
                batch = [r for r in self.rows[k * prompts:(k + 1) * prompts] for _ in range(a.num_generations)]
                ids = [[7] * (10 + k) + [EOS[0]] for _ in batch]          # lengths that differ by step
                ids[2] = [7] * a.max_completion_length                      # cut at the limit: G2
                columns = {c: [r[c] for r in batch] for c in batch[0] if c != "prompt"}   # TRL passes every column
                out = self.reward(prompts=[r["prompt"] for r in batch], completion_ids=ids, trainer_state=self.state,
                                  completions=[[{"role": "assistant", "content": t}] for t in texts(k, batch)],
                                  **columns)
                assert len(out) == len(batch) == 32
                self.state.global_step += 1
                self._each("on_step_end")
                self._each("on_log", logs={"loss": 0.0, "reward": sum(out) / len(out), "reward_std": 0.1,
                                           "kl": 0.001 * k, "completions/mean_length": 11.0,
                                           "sampling/sampling_logp_difference/mean": 0.002 * (k + 1)})
                if self.state.global_step % a.save_steps == 0:
                    d = Path(a.output_dir) / f"checkpoint-{self.state.global_step}"
                    d.mkdir(parents=True)
                    (d / "trainer_state.json").write_text(json.dumps({"global_step": self.state.global_step}))
                    for f in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                        (d / f).write_bytes(b"")
                    self._each("on_save")
            self._each("on_log", logs={"train_runtime": 1.0})
            return SimpleNamespace(metrics={"train_runtime": 1.0})

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        return m
    cuda = SimpleNamespace(is_available=lambda: True,
                           get_device_properties=lambda i: SimpleNamespace(name="FAKE A100-SXM4-80GB", total_memory=80 * 2**30))
    return {
        "torch": mod("torch", cuda=cuda, bfloat16="bf16"),
        "datasets": mod("datasets", Dataset=SimpleNamespace(from_list=list)),
        "peft": mod("peft", LoraConfig=lambda **kw: dict(kw)),
        "transformers": mod("transformers", AutoModelForCausalLM=Model, AutoTokenizer=Tok, TrainerCallback=object,
                            GenerationConfig=SimpleNamespace(from_pretrained=lambda p: SimpleNamespace(eos_token_id=list(EOS)))),
        "trl": mod("trl", GRPOConfig=GRPOConfig, GRPOTrainer=Trainer),
        "vllm": mod("vllm", LLM=type("LLM", (FakeLLM,), {})),       # a fresh class each time: the pin patches it
        "_seen": seen,
    }


@pytest.fixture
def stack(monkeypatch, tmp_path):
    def install(missing=frozenset({"vllm_max_model_length"}), config=SMOKE, texts=_plain_texts, real_prompts=False):
        mods = _stack(set(missing), config, texts)
        seen = mods.pop("_seen")
        for name, m in mods.items():
            monkeypatch.setitem(sys.modules, name, m)
        if not real_prompts:            # the v2 rehearsal reads the real templates (example ids, references)
            monkeypatch.setitem(sys.modules, "lofi_rl.prompt", fake_prompt())
        monkeypatch.setitem(sys.modules, "lofi_rl.verify", fake_verify())
        monkeypatch.setattr(kit.hf_cache, "resolve_pinned", lambda repo, rev, need: str(tmp_path / "snapshot"))
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        return seen
    return install


def test_the_smoke_data_flow_from_two_launches_to_the_verdict(stack, tmp_path):
    seen = stack()
    cfg, run = rc.load_config(SMOKE), tmp_path / "grpo-smoke-1"
    assert train.train(cfg, run_dir=run / "a", summary_dir=run / "summary" / "a") == 0
    assert train.train(cfg, run_dir=run / "b", summary_dir=run / "summary" / "b",
                       resume_from=run / "a" / "checkpoint-5") == 0

    a, b = seen["trainers"]
    assert a.args.output_dir == str(run / "a") and a.args.run_name == "grpo-smoke-local"
    assert (a.args.per_device_train_batch_size, a.args.gradient_accumulation_steps, a.args.beta) == (8, 4, 0.04)
    assert a.lora == {"r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": "all-linear",
                      "bias": "none", "task_type": "CAUSAL_LM"}
    assert a.tok.padding_side == a.tok.truncation_side == "left" and a.model.config.use_cache is False
    assert seen["dtype"] == "bf16" and len(a.rows) == 40

    man = json.loads((run / "summary" / "a" / "run.json").read_text())
    assert (man["status"], man["final_global_step"], man["start_step"], man["profile"]) == ("ok", 10, 0, "a100-80gb")
    assert man["dropped_optional_fields"] == ["vllm_max_model_length"] and "loss_type" in man["verified_trl_fields"]
    assert man["eos_token_ids"] == sorted(EOS) and man["prompt"]["rows"] == 40 and man["params"]["trainable"] > 0
    assert man["grpo_config"]["output_dir"] == str(run / "a") and man["startup_s"] >= 0
    man_b = json.loads((run / "summary" / "b" / "run.json").read_text())
    assert (man_b["start_step"], man_b["resume_from"], man_b["final_global_step"]) == (5, str(run / "a" / "checkpoint-5"), 10)

    recs = [json.loads(l) for l in (run / "summary" / "a" / "rewards.jsonl").read_text().splitlines()]
    assert len(recs) == 320 and sorted({r["step"] for r in recs}) == list(range(10))
    assert {r["step_source"] for r in recs} == {"trainer_state"}
    assert all(r["gate"] == "G2" and r["G"] == 0 for r in recs if r["i"] == 2)
    assert all(r["p_group"] == pytest.approx(0.3) for r in recs if r["i"] == 1)
    w = {r["step"]: r["w"]["theory"] for r in recs}
    assert w[0] == pytest.approx(.5) and w[9] == pytest.approx(.5 - .4 * 9 / 30)
    steps = [json.loads(l) for l in (run / "summary" / "a" / "steps.jsonl").read_text().splitlines()]
    assert [s["kind"] for s in steps].count("step") == 10 and [s["kind"] for s in steps].count("save") == 2
    assert steps[0]["lofi"]["step"] == 0 and steps[0]["lofi"]["truncated"] == 1 and steps[0]["trl"]["kl"] == 0.0

    rep = smoke_check.check(run, "final", cfg, {"a": "0", "b": "0"})
    assert rep["ok"], rep["failures"]
    assert rep["resumed"]["trained"] == [6, 7, 8, 9, 10] and rep["resumed"]["reward_steps"] == [5, 6, 7, 8, 9]
    assert any("vllm_max_model_length" in w for w in rep["warnings"])
    assert len(rep["steps"]) == 15 and all(s["step_s"] is not None for s in rep["steps"])


def test_a_required_field_trl_lacks_stops_the_run_before_the_model_loads(stack, tmp_path):
    seen = stack(missing={"loss_type"})
    with pytest.raises(rc.ConfigError, match="loss_type"):
        train.train(rc.load_config(SMOKE), run_dir=tmp_path / "a", summary_dir=tmp_path / "summary" / "a")
    man = json.loads((tmp_path / "summary" / "a" / "run.json").read_text())
    assert man["status"] == "failed" and "loss_type" in man["error"] and seen["loads"] == 0
    assert not (tmp_path / "a").exists()


# ------------------------------------------------------------------------------------------ D-020's run-1 settings
def _v2_texts(k: int, batch: list[dict]) -> list[str]:
    """Per group of 8: the shown example verbatim (J 1), a cut completion (index 2 of the batch, cut by its ids),
    and the shown example with new A and B phrases (in the ramp); every stop completion is in the regex's language."""
    from lofi_rl import prompt_v2
    from lofi_rl.palette import load_palette
    refs = prompt_v2.reference_texts(load_palette(ROOT / "palettes" / "dev_gu.yaml", verify=False))
    out = []
    for i, r in enumerate(batch):
        lines = refs[r["example_id"]].splitlines()
        if i % 8 == 0:
            out.append(refs[r["example_id"]])
        elif i == 2:
            out.append("cut at the limit, not a loop")
        else:
            out.append("\n".join(lines[:-2] + [f"A: h1 h3 | w5 | h.6 q5 | w{1 + i % 7}", "B: w1 | w3 | w5 | w1"]) + "\n")
    return out


def test_the_v2_grammar_smoke_pins_the_backend_counts_the_language_and_reads_the_is_metric(stack, tmp_path,
                                                                                            monkeypatch):
    seen = stack(config=V2_GRAMMAR, texts=_v2_texts, real_prompts=True)
    base = tmp_path / "site" / "trl"
    (base / "trainer").mkdir(parents=True)
    (base / "trainer" / "grpo_trainer.py").write_text(TRL_LIKE_SOURCE)
    real = trl_probe.logp_metric_source
    monkeypatch.setattr(trl_probe, "logp_metric_source", lambda *a, **k: real(base=base))    # a TRL-shaped install
    cfg, run = rc.load_config(V2_GRAMMAR), tmp_path / "grpo-smoke-2"
    llm_cls = sys.modules["vllm"].LLM
    assert train.train(cfg, run_dir=run / "a", summary_dir=run / "summary" / "a") == 0
    assert train.train(cfg, run_dir=run / "b", summary_dir=run / "summary" / "b",
                       resume_from=run / "a" / "checkpoint-5") == 0

    a, b = seen["trainers"]
    assert a.args.vllm_structured_outputs_regex == cfg.structured_outputs["regex"]
    assert a.vllm_generation.llm.kw["structured_outputs_config"] == {"backend": "xgrammar"}, "the pin reached LLM()"
    assert "__init__" not in vars(llm_cls), "and was taken off after the launch"
    from lofi_rl import prompt_v2
    from lofi_rl.brief import Brief
    assert [r["example_id"] for r in a.rows] == [prompt_v2.example_id(Brief(**json.loads(r["brief"]))) for r in a.rows]
    man = json.loads((run / "summary" / "a" / "run.json").read_text())
    so = man["structured_outputs"]
    assert so["sha256"] == cfg.structured_outputs["sha256"] and "regex" not in so
    assert {k: so["backend"][k] for k in ("requested", "pin_accepted", "backend", "applied", "engine", "calls")} == {
        "requested": "xgrammar", "pin_accepted": True, "backend": "xgrammar", "applied": True,
        "engine": "trainer.vllm_generation.llm", "calls": 1}
    assert so["backend"]["source"] == "llm_engine.vllm_config.structured_outputs_config.backend"
    p = man["prompt"]
    assert p["version"] == "v2" and p["template_sha256"] == "127e3094fd5a2183fb952059c2221ad018bdef6d062f51a80b14367b8064466e"
    assert p["example_ids"] == [r["example_id"] for r in a.rows] and p["tokens"]["max"] > 0
    assert man["copy_penalty"]["references"] == 9 and man["trl_logp_metric"]["verdict"] == "match"

    recs = [json.loads(l) for l in (run / "summary" / "a" / "rewards.jsonl").read_text().splitlines()]
    assert len(recs) == 320 and all({"p_copy", "copy_j", "finish", "in_language", "example_id"} <= set(r) for r in recs)
    assert {r["finish"] for r in recs if r["i"] == 2} == {"length"} and all(r["gate"] == "G2" for r in recs if r["i"] == 2)
    assert all(r["in_language"] for r in recs if r["finish"] == "stop")
    shown = [r for r in recs if r["i"] % 8 == 0]
    assert all(r["copy_j"] == 1.0 and r["p_copy"] == 0.3 and r["copy_nearest"] == r["example_id"] for r in shown)
    from lofi_rl.rl.copy_penalty import ramp
    edited = [r for r in recs if r["i"] % 8 != 0 and r["i"] != 2]
    assert all(r["p_copy"] == ramp(r["copy_j"]) for r in recs)
    assert any(0 < r["p_copy"] < 0.3 for r in edited) and any(r["p_copy"] == 0 for r in edited), \
        sorted({round(r["copy_j"], 3) for r in edited})

    rep = smoke_check.check(run, "final", cfg, {"a": "0", "b": "0"})
    assert rep["ok"], rep["failures"]
    assert rep["in_language"]["stop"] == {"n": 15 * 31, "in_language": 15 * 31, "out_of_language": 0}
    assert rep["in_language"]["length"]["n"] == 15 and rep["in_language"]["other"]["n"] == 0
    assert rep["backend"]["a"]["applied"] is True and rep["backend"]["b"]["applied"] is True
    penalized = [r for r in recs if r["p_copy"] > 0]
    assert rep["copy"]["a"]["penalized"] == len(penalized) and rep["copy"]["a"]["n"] == 320
    assert rep["copy"]["a"]["nearest_is_shown"] == sum(1 for r in penalized if r["copy_nearest"] == r["example_id"])
    isr = rep["is_ratio"]
    steps = list(range(1, 11)) + list(range(6, 11))
    tokens = [31 * (10 + s) for s in steps]                 # step s scores reward step s-1: 31 uncut of 11 + (s-1)
    values = [0.002 * s for s in steps]
    assert [r["value"] for r in isr["steps"]] == pytest.approx(values) and [r["tokens"] for r in isr["steps"]] == tokens
    assert isr["value_source"] == "trl:sampling/sampling_logp_difference/mean"
    assert isr["mean_of_steps"] == pytest.approx(sum(values) / 15)
    assert isr["token_weighted_mean"] == pytest.approx(sum(v * t for v, t in zip(values, tokens)) / sum(tokens))
    rule = isr["rule"]
    assert rule["applies"] is True and rule["reads"] == "token_weighted_mean" and rule["exceeds"] is False
    assert rule["value"] == isr["token_weighted_mean"] and "keeps the grammar on" in rule["says"]
    assert any("truncated completions" in c for c in rule["caveats"]), "each step had one cut completion"
    assert isr["grammar_shown"] is True and isr["vllm_logprobs_checked"] == {"a": False, "b": False}
    assert smoke_check.VLLM_CAVEAT in rule["caveats"], "train() records vllm_logprobs unchecked, and the rule says so"
    assert set(isr["by_launch"]) == {"a", "b"}


def test_a_trainer_that_fails_to_build_still_records_the_pin_and_restores_the_class(stack, tmp_path):
    stack(config=V2_GRAMMAR, texts=_v2_texts, real_prompts=True)
    base = sys.modules["trl"].GRPOTrainer

    class Failing(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)               # the engine is built, then the trainer fails
            raise RuntimeError("trainer init failed")
    sys.modules["trl"].GRPOTrainer = Failing
    llm_cls = sys.modules["vllm"].LLM
    with pytest.raises(RuntimeError, match="trainer init failed"):
        train.train(rc.load_config(V2_GRAMMAR), run_dir=tmp_path / "a", summary_dir=tmp_path / "summary" / "a")
    man = json.loads((tmp_path / "summary" / "a" / "run.json").read_text())
    b = man["structured_outputs"]["backend"]
    assert man["status"] == "failed" and (b["pin_accepted"], b["calls"], b["engine"]) == (True, 1, None)
    assert b["read"] == "no trainer: GRPOTrainer was not built" and "__init__" not in vars(llm_cls)


def test_without_a_checked_trl_source_the_is_readout_is_unavailable_and_says_why(stack, tmp_path):
    stack(config=V2_GRAMMAR, texts=_v2_texts, real_prompts=True)        # the laptop: no TRL installed to read
    cfg, run = rc.load_config(V2_GRAMMAR), tmp_path / "grpo-smoke-3"
    assert train.train(cfg, run_dir=run / "a", summary_dir=run / "summary" / "a") == 0
    assert train.train(cfg, run_dir=run / "b", summary_dir=run / "summary" / "b",
                       resume_from=run / "a" / "checkpoint-5") == 0
    rep = smoke_check.check(run, "final", cfg, {"a": "0", "b": "0"})
    assert rep["ok"], rep["failures"]
    isr = rep["is_ratio"]
    assert isr["value_source"] == "unavailable" and isr["rule"] is None and isr["mean_of_steps"] is None
    assert isr["verdicts"] == {"a": "not_found", "b": "not_found"} and "not checked" in isr["reason"]
    assert any(w.startswith("IS readout unavailable") for w in rep["warnings"])
    assert [r["value"] for r in isr["steps"]][:2] == pytest.approx([0.002, 0.004]), "TRL's values stay visible"
