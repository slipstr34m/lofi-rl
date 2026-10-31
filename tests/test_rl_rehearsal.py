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
from helpers_rl import fake_prompt, fake_verify
from lofi_rl.rl import config as rc
from lofi_rl.rl import smoke_check, train

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "grpo_smoke.yaml"
EOS = (151645, 151643)


def _stack(missing: set[str]) -> dict[str, types.ModuleType]:
    cfg = rc.load_config(SMOKE)
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
                texts = [f"theory=0.{i % 8 + 1} knobs=0.5 | {k}-{i}" for i in range(len(batch))]
                texts[1] = texts[0]                                         # a duplicate in group 0
                ids = [[7] * 10 + [EOS[0]] for _ in batch]
                ids[2] = [7] * a.max_completion_length                      # cut at the limit: G2
                out = self.reward(prompts=[r["prompt"] for r in batch], completion_ids=ids, trainer_state=self.state,
                                  completions=[[{"role": "assistant", "content": t}] for t in texts],
                                  brief=[r["brief"] for r in batch], brief_id=[r["brief_id"] for r in batch])
                assert len(out) == len(batch) == 32
                self.state.global_step += 1
                self._each("on_step_end")
                self._each("on_log", logs={"loss": 0.0, "reward": sum(out) / len(out), "reward_std": 0.1,
                                           "kl": 0.001 * k, "completions/mean_length": 11.0})
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
        "_seen": seen,
    }


@pytest.fixture
def stack(monkeypatch, tmp_path):
    def install(missing=frozenset({"vllm_max_model_length"})):
        mods = _stack(set(missing))
        seen = mods.pop("_seen")
        for name, m in mods.items():
            monkeypatch.setitem(sys.modules, name, m)
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
