"""The GRPO run config (configs/grpo_smoke.yaml, src/lofi_rl/rl/config.py) and the trainer's laptop half
(src/lofi_rl/rl/train.py): the §6 settings, validation, fitting the settings to TRL's GRPOConfig fields, the
prompts, `--dry-run` without torch/TRL/vLLM, and the per-step log. Training itself needs a GPU and the
policy env, so jobs/grpo_smoke.sbatch tests it on the cluster.

D-021 Part A (the masked trainer): configs/grpo_smoke_v2_grammar_masked.yaml pinned to the v2 grammar smoke plus one
key, `trainer_logprobs` and its refusals, the dry run's M4 readout, the refusal rules for the source checks, the step
log's mask records, and train()'s masked wiring against stand-ins for the ML stack and for sections A, C and D."""
import dataclasses
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import types
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from helpers_rl import CHILD_FAKES, fake_prompt, fake_verify
from lofi_rl import grammar
from lofi_rl.brief import Brief, sample_briefs
from lofi_rl.palette import load_palette
from lofi_rl.rl import config as rc
from lofi_rl.rl import train, trl_probe

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "grpo_smoke.yaml"
CONSTRAINED = ROOT / "configs" / "grpo_smoke_constrained.yaml"
REGEX_FIELD = "vllm_structured_outputs_regex"
HEAVY = ("torch", "trl", "vllm", "transformers", "peft", "datasets", "accelerate")


def _write(tmp_path, edit) -> Path:
    doc = yaml.safe_load(SMOKE.read_text())
    edit(doc)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


# ------------------------------------------------------------------------------------------ the §6 table
def test_the_smoke_config_is_the_section_6_table():
    cfg = rc.load_config(SMOKE)
    t = cfg.trainer
    assert (t["learning_rate"], t["lr_scheduler_type"], t["warmup_steps"]) == (1e-5, "constant_with_warmup", 3)
    assert (t["num_generations"], cfg.prompts_per_step, t["num_iterations"]) == (8, 4, 1)
    assert (t["max_completion_length"], t["mask_truncated_completions"], t["beta"]) == (512, True, 0.04)
    assert (t["loss_type"], t["scale_rewards"]) == ("dapo", "batch")
    assert (t["temperature"], t["top_p"], t["top_k"]) == (1.0, 1.0, 0)
    assert (t["use_vllm"], t["vllm_mode"]) == (True, "colocate")
    assert (t["max_steps"], t["save_steps"], cfg.resume_step) == (10, 5, 5)
    assert cfg.lora == {"r": 16, "alpha": 32, "dropout": 0.0, "target_modules": "all-linear"}
    a100, card48 = rc.resolve(cfg, "a100-80gb"), rc.resolve(cfg, "48gb")
    assert (a100["per_device_train_batch_size"], a100["gradient_accumulation_steps"]) == (8, 4)
    assert (card48["per_device_train_batch_size"], card48["gradient_accumulation_steps"]) == (4, 8)
    assert (a100["vllm_gpu_memory_utilization"], card48["vllm_gpu_memory_utilization"]) == (0.25, 0.30)
    assert (a100["vllm_enable_sleep_mode"], card48["vllm_enable_sleep_mode"]) == (False, True)


def test_the_model_is_the_pinned_snapshot_the_fetch_job_cached():
    from lofi_rl.smoke import QWEN
    cfg = rc.load_config(SMOKE)
    assert (cfg.model_repo, cfg.model_revision) == QWEN[:2] == ("Qwen/Qwen3-4B-Instruct-2507",
                                                               "cdbee75f17c01a7cc42f958dc650907174af0554")


@pytest.mark.parametrize("profile", ["a100-80gb", "48gb"])
def test_every_profile_trains_on_4_prompts_x_8_completions_per_step(profile):
    cfg = rc.load_config(SMOKE)
    kw = rc.resolve(cfg, profile)
    assert kw["per_device_train_batch_size"] * kw["gradient_accumulation_steps"] == 4 * 8
    assert "min_gib" not in kw and "output_dir" not in kw


# ------------------------------------------------------------------------------------------ validation
@pytest.mark.parametrize("edit, match", [
    (lambda d: d.update(extra=1), "unknown key"),
    (lambda d: d["trainer"].pop("beta"), "missing.*beta"),
    (lambda d: d["trainer"].update(learning_rate="1e-5"), "learning_rate"),
    (lambda d: d["trainer"].update(output_dir="/tmp/x"), "output_dir"),
    (lambda d: d["profiles"]["48gb"].update(gradient_accumulation_steps=4), "32"),
    (lambda d: d["trainer"].update(save_steps=4), "save_steps"),
    (lambda d: d["smoke"].update(resume_from_step=10), "resume"),
    (lambda d: d["briefs"].update(count=39), "count"),
    (lambda d: d["model"].update(revision="main"), "40-hex"),
    (lambda d: d["trainer"].update(scale_rewards="grouped"), "scale_rewards"),
    (lambda d: d["optional_fields"].append("beta"), "optional"),
    (lambda d: d.update(structured_outputs="json_schema"), "structured_outputs"),
    (lambda d: d.update(structured_outputs=True), "structured_outputs"),
    (lambda d: d["trainer"].update(vllm_structured_outputs_regex="vamp: .*"), "structured_outputs: loopscript_regex"),
    (lambda d: d["profiles"]["48gb"].update(vllm_structured_outputs_regex="x"), "structured_outputs: loopscript_regex"),
    (lambda d: d["optional_fields"].append(REGEX_FIELD) or d.update(structured_outputs="loopscript_regex"),
     "optional"),
    (lambda d: d.update(prompt="v3"), "prompt 'v3' is not one of"),
    (lambda d: d.update(prompt=2), "prompt 2 is not one of"),
    (lambda d: d.update(prompt="V2"), "prompt 'V2'"),
    (lambda d: d.update(copy_penalty=True), "copy_penalty needs exactly"),
    (lambda d: d.update(copy_penalty={}), "copy_penalty needs exactly"),
    (lambda d: d.update(copy_penalty={"pool": [], "j_full": 0.9}), "copy_penalty needs exactly"),
    (lambda d: d.update(copy_penalty={"pool": "drafts/v1"}), "not a list of paths"),
    (lambda d: d.update(copy_penalty={"pool": ["drafts/v1", "drafts/v1"]}), "twice"),
    (lambda d: d.update(copy_penalty={"pool": ["drafts/nowhere"]}), "does not exist"),
    (lambda d: d.update(copy_penalty={"pool": ["drafts/v1/briefs.yaml"]}), "not a .ls file"),
    (lambda d: d.update(copy_penalty={"pool": ["configs"]}), "holds no .ls file"),
])
def test_a_bad_config_is_refused_with_its_reason(tmp_path, edit, match):
    with pytest.raises(rc.ConfigError, match=match):
        rc.load_config(_write(tmp_path, edit))


# ------------------------------------------------------------------------------------------ constrained decoding
def test_the_constrained_smoke_is_the_smoke_plus_the_loopscript_regex_built_at_load_time():
    plain, constrained = yaml.safe_load(SMOKE.read_text()), yaml.safe_load(CONSTRAINED.read_text())
    assert constrained.pop("structured_outputs") == "loopscript_regex"
    assert constrained == plain, "identical to configs/grpo_smoke.yaml but for the one key"
    pal = load_palette(ROOT / plain["palette"], verify=False)
    regex = grammar.loopscript_regex(pal)
    assert regex not in CONSTRAINED.read_text() and f"{REGEX_FIELD}:" not in CONSTRAINED.read_text(), \
        "the regex comes from lofi_rl.grammar when the config loads; it is never pasted into the YAML"
    cfg, smoke = rc.load_config(CONSTRAINED), rc.load_config(SMOKE)
    assert cfg.structured_outputs["builder"] == "loopscript_regex" and cfg.structured_outputs["field"] == REGEX_FIELD
    assert cfg.structured_outputs["sha256"] == grammar.regex_sha256(pal)
    assert cfg.structured_outputs["chars"] == len(regex) and smoke.structured_outputs is None
    for profile in cfg.profiles:
        kw, base = rc.resolve(cfg, profile), rc.resolve(smoke, profile)
        assert kw.pop(REGEX_FIELD) == regex and kw == base, profile
        assert REGEX_FIELD not in base


def test_a_trl_without_the_regex_field_stops_the_constrained_run_before_the_model_loads():
    cfg = rc.load_config(CONSTRAINED)
    kw = rc.resolve(cfg, "48gb")
    known = [k for k in kw if k != REGEX_FIELD]
    without = dataclasses.make_dataclass("GRPOConfig", [(k, object, None) for k in known])
    with pytest.raises(rc.ConfigError, match=REGEX_FIELD):
        rc.fit_fields(kw, without, cfg.optional_fields)       # what train() runs first, before the model loads
    with_it = dataclasses.make_dataclass("GRPOConfig", [(k, object, None) for k in known + [REGEX_FIELD]])
    kept, dropped = rc.fit_fields(kw, with_it, cfg.optional_fields)
    assert kept[REGEX_FIELD] == grammar.loopscript_regex(load_palette(cfg.palette_path(), verify=False))
    assert dropped == []


def test_the_dry_run_of_the_constrained_smoke_names_its_regex(monkeypatch):
    monkeypatch.setitem(sys.modules, "lofi_rl.prompt", fake_prompt())
    monkeypatch.setattr(train, "_verify_module", lambda: None)
    buf = io.StringIO()
    report = train.dry_run(rc.load_config(CONSTRAINED), out=buf)
    pal = load_palette(ROOT / "palettes" / "dev_gu.yaml", verify=False)
    want = {"builder": "loopscript_regex", "field": REGEX_FIELD, "sha256": grammar.regex_sha256(pal),
            "chars": len(grammar.loopscript_regex(pal))}
    assert report["structured_outputs"] == want and json.loads(buf.getvalue())["structured_outputs"] == want
    assert REGEX_FIELD in report["unverified_trl_fields"], "the cluster's GRPOConfig check covers the name"
    assert train.dry_run(rc.load_config(SMOKE), out=io.StringIO())["structured_outputs"] is None


def test_settings_fit_trl_fields_dropping_only_the_optional_unknowns():
    @dataclasses.dataclass
    class FakeGRPOConfig:
        beta: float = 0.0
        max_steps: int = -1
    kept, dropped = rc.fit_fields({"beta": .04, "max_steps": 10, "vllm_max_model_length": 4096},
                                  FakeGRPOConfig, optional={"vllm_max_model_length"})
    assert kept == {"beta": .04, "max_steps": 10} and dropped == ["vllm_max_model_length"]
    with pytest.raises(rc.ConfigError, match="loss_type"):
        rc.fit_fields({"beta": .04, "loss_type": "dapo"}, FakeGRPOConfig, optional=set())


@pytest.mark.parametrize("gib, want", [(79.2, "a100-80gb"), (44.4, "48gb"), (47.5, "48gb")])
def test_the_gpu_profile_follows_the_card_memory(gib, want):
    assert rc.detect_profile(rc.load_config(SMOKE), gib) == want


def test_a_card_under_40_gib_has_no_profile():
    with pytest.raises(rc.ConfigError, match="no profile"):
        rc.detect_profile(rc.load_config(SMOKE), 23.6)


@pytest.mark.parametrize("path", [SMOKE, CONSTRAINED, ROOT / "configs" / "grpo_smoke_v2_grammar.yaml"],
                         ids=["smoke", "constrained", "v2_grammar"])
def test_trl_knows_every_setting_of_the_smoke(path):
    # Skipped on the laptop (no TRL). In the policy env it checks the names this file could not verify.
    trl = pytest.importorskip("trl")
    names = {f.name for f in dataclasses.fields(trl.GRPOConfig)}
    cfg = rc.load_config(path)
    for profile in cfg.profiles:
        unknown = sorted(k for k in rc.resolve(cfg, profile) if k not in names and k not in cfg.optional_fields)
        assert not unknown, f"{profile}: GRPOConfig of TRL {trl.__version__} lacks {unknown}"


# ------------------------------------------------------------------------------------------ prompts
def test_rows_are_the_sampled_briefs_in_order_with_their_messages():
    cfg = rc.load_config(SMOKE)
    pal = load_palette(ROOT / cfg.palette, verify=False)
    rows = train.build_rows(cfg, pal, fake_prompt())
    briefs = sample_briefs(cfg.brief_seed, cfg.brief_count, cfg.brief_start)
    assert len(rows) == cfg.brief_count == len(briefs)
    for i, (row, b) in enumerate(zip(rows, briefs)):
        assert set(row) == {"prompt", "brief", "brief_id"}
        assert Brief(**json.loads(row["brief"])) == b and row["brief_id"] == f"{cfg.brief_seed}:{cfg.brief_start + i}"
        assert [m["role"] for m in row["prompt"]] == ["system", "user"] and b.mood in row["prompt"][1]["content"]


def test_malformed_messages_from_the_prompt_builder_are_refused():
    cfg = rc.load_config(SMOKE)
    pal = load_palette(ROOT / cfg.palette, verify=False)
    bad = fake_prompt()
    bad.build_messages = lambda brief, palette: [{"role": "user", "content": 3}]
    with pytest.raises(rc.ConfigError, match="build_messages"):
        train.build_rows(cfg, pal, bad)


# ------------------------------------------------------------------------------------------ dry run
def test_dry_run_prints_the_resolved_settings_and_scores_a_fixture(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "lofi_rl.prompt", fake_prompt())
    monkeypatch.setitem(sys.modules, "lofi_rl.verify", fake_verify())
    assert train.main(["--config", str(SMOKE), "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    cfg = rc.load_config(SMOKE)
    assert out["dry_run"] is True and out["config"]["sha256"] == cfg.sha256
    assert out["model"] == {"repo": cfg.model_repo, "revision": cfg.model_revision}
    assert set(out["profiles"]) == {"a100-80gb", "48gb"}
    assert out["profiles"]["a100-80gb"]["trainer"]["per_device_train_batch_size"] == 8
    assert out["profiles"]["a100-80gb"]["completions_per_step"] == 32
    assert out["lora"]["target_modules"] == "all-linear" and out["palette"]["id"] == "dev_gu"
    assert out["prompts"]["rows"] == cfg.brief_count and out["prompts"]["version"] == "v1"
    assert sum(out["prompts"]["energy_x_mood"].values()) == cfg.brief_count
    assert out["unverified_trl_fields"], "the dry run says which TRL names the cluster must confirm"
    assert out["reward_check"]["reward"] == pytest.approx(.5 * .5 + .15 * .5)


def test_dry_run_never_contacts_a_scorer_named_by_the_environment(monkeypatch):
    # An exported $LOFI_SCORER_SOCK turns the scorer on for a run, not for the dry run's symbolic check: with the
    # old default the check waited up to 600 s for /health of a socket that does not exist.
    import time
    from lofi_rl.scoring.client import ENV_SOCK, ScorerClient
    monkeypatch.setenv(ENV_SOCK, "/nonexistent/x.sock")
    monkeypatch.setitem(sys.modules, "lofi_rl.prompt", fake_prompt())
    monkeypatch.setitem(sys.modules, "lofi_rl.verify", fake_verify())

    def contacted(self, *a, **k):
        raise AssertionError(f"the dry run contacted the scorer at {self.sock}")
    monkeypatch.setattr(ScorerClient, "wait_healthy", contacted)
    t = time.monotonic()
    report = train.dry_run(rc.load_config(SMOKE), out=io.StringIO())
    assert time.monotonic() - t < 30
    assert report["reward_check"]["reward"] == pytest.approx(.5 * .5 + .15 * .5)
    assert report["reward"]["scorer"] == {"env": ENV_SOCK, "sock": "/nonexistent/x.sock", "contacted": False}


def test_dry_run_runs_with_torch_trl_and_vllm_blocked():
    block = "import sys\n" + "".join(f"sys.modules[{m!r}] = None\n" for m in HEAVY)
    code = (block + CHILD_FAKES + "from lofi_rl.rl import train\n"
            f"rc = train.main(['--config', {str(SMOKE)!r}, '--dry-run'])\n"
            f"assert not [m for m in {HEAVY!r} if sys.modules.get(m) is not None]\nraise SystemExit(rc)\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=env, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-3000:]
    assert json.loads(p.stdout)["prompts"]["rows"] == rc.load_config(SMOKE).brief_count


def test_dry_run_refuses_a_bad_config(tmp_path, capsys):
    assert train.main(["--config", str(_write(tmp_path, lambda d: d.update(extra=1))), "--dry-run"]) == 2
    assert "unknown key" in capsys.readouterr().err


def test_a_real_run_without_the_ml_stack_says_so(tmp_path, monkeypatch, capsys):
    for m in ("torch", "trl"):
        monkeypatch.setitem(sys.modules, m, None)
    monkeypatch.setitem(sys.modules, "lofi_rl.prompt", fake_prompt())
    assert train.main(["--config", str(SMOKE), "--run-dir", str(tmp_path / "a")]) == 3
    assert "policy env" in capsys.readouterr().err
    assert not (tmp_path / "a").exists()


# ------------------------------------------------------------------------------------------ step log
def test_the_step_log_writes_one_line_per_optimizer_step_with_trl_and_reward_metrics(tmp_path):
    clock = iter([100.0, 125.0, 152.0, 153.0])
    stats = {0: {"reward_mean": .1, "gate_rate": .5}, 1: {"reward_mean": .2, "gate_rate": .75}}
    log = train.StepLog(tmp_path / "steps.jsonl", reward_stats=stats, clock=lambda: next(clock))
    log.train_begin(0)
    log.step_end(1); log.log(1, {"loss": 0.01, "reward": .1, "kl": 0.0})
    log.step_end(2); log.log(2, {"loss": 0.02, "reward": .2, "kl": 0.001})
    log.log(2, {"train_runtime": 53.0, "train_loss": 0.015})
    rows = [json.loads(l) for l in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [(r["kind"], r["step"]) for r in rows] == [("step", 1), ("step", 2), ("train_end", 2)]
    assert rows[0]["step_s"] == 25.0 and rows[1]["step_s"] == 27.0
    assert rows[0]["trl"]["kl"] == 0.0 and rows[1]["lofi"] == {"reward_mean": .2, "gate_rate": .75}
    assert rows[2]["trl"]["train_runtime"] == 53.0


def test_every_link_of_a_run_shares_one_scorer_pin(tmp_path):
    # docs/GRPO.md's layout: <run>/a, then <run>/b resumed from <run>/a/checkpoint-5, then <run>/c from b's.
    run = tmp_path / "run"
    a = train.scorer_pin_path(run / "a")
    assert a == run / "a" / "scorer_pin.json" and not a.exists(), "nothing is written before a scorer pins"
    a.parent.mkdir(parents=True)
    a.write_text('{"pinned": "by a"}')
    b = train.scorer_pin_path(run / "b", resume_from=run / "a" / "checkpoint-5")
    assert b == run / "b" / "scorer_pin.json" and b.read_text() == a.read_text(), "inherited from a"
    c = train.scorer_pin_path(run / "c", resume_from=run / "b" / "checkpoint-10")
    assert c.read_text() == a.read_text(), "and passed on"
    c.write_text('{"pinned": "by c"}')
    assert train.scorer_pin_path(run / "c", resume_from=run / "b" / "checkpoint-10").read_text() == '{"pinned": "by c"}'
    assert train.scorer_pin_path(run / "d", explicit=run / "pin.json") == run / "pin.json"
    assert not train.scorer_pin_path(tmp_path / "x", resume_from=tmp_path / "nowhere" / "checkpoint-1").exists()


def test_resume_reads_the_global_step_of_the_checkpoint(tmp_path):
    ck = tmp_path / "checkpoint-5"
    ck.mkdir()
    (ck / "trainer_state.json").write_text(json.dumps({"global_step": 5, "log_history": []}))
    assert train.checkpoint_step(ck) == 5
    with pytest.raises(rc.ConfigError, match="trainer_state.json"):
        train.checkpoint_step(tmp_path)


def test_dry_run_output_is_json_even_with_unicode_prompts(monkeypatch):
    p = fake_prompt()
    p.build_messages = lambda b, pal: [{"role": "system", "content": "Δ7 ♭9"}, {"role": "user", "content": b.mood}]
    monkeypatch.setitem(sys.modules, "lofi_rl.prompt", p)
    monkeypatch.delitem(sys.modules, "lofi_rl.verify", raising=False)
    monkeypatch.setattr(train, "_verify_module", lambda: None)
    buf = io.StringIO()
    report = train.dry_run(rc.load_config(SMOKE), out=buf)
    assert json.loads(buf.getvalue())["prompts"]["preview"][0]["content"] == "Δ7 ♭9"
    assert report["reward_check"].startswith("skipped")


# ------------------------------------------------------------------------------------------ D-020: prompt v2, P_copy
V2_GRAMMAR = ROOT / "configs" / "grpo_smoke_v2_grammar.yaml"
SMOKE_13632679_SHA = "d0ed170267ca1ee8e711d7603e5a1b5997b2fcfe9b90c10a49e02c5189caa04d"   # its smoke.json
V2_TEMPLATE = "127e3094fd5a2183fb952059c2221ad018bdef6d062f51a80b14367b8064466e"          # D-020, arm B
V1_TEMPLATE = "7aceb2bc0d7339a7d3c9af7bf0d808ee156d4f7a8684e0393d923940fa5c1e68"


def test_todays_smoke_config_is_byte_for_byte_the_one_smoke_13632679_ran():
    cfg = rc.load_config(SMOKE)
    assert cfg.sha256 == SMOKE_13632679_SHA, "a bare submit of jobs/grpo_smoke.sbatch runs exactly that smoke"
    assert (cfg.prompt, cfg.copy_penalty, cfg.structured_outputs) == ("v1", None, None)


def test_the_v2_grammar_smoke_is_the_smoke_plus_prompt_v2_the_grammar_and_the_copy_penalty():
    plain, v2 = yaml.safe_load(SMOKE.read_text()), yaml.safe_load(V2_GRAMMAR.read_text())
    added = {k: v2.pop(k) for k in ("prompt", "structured_outputs", "copy_penalty")}
    assert added == {"prompt": "v2", "structured_outputs": "loopscript_regex", "copy_penalty": {"pool": []}}
    assert v2 == plain, "every other key is the smoke's"
    cfg = rc.load_config(V2_GRAMMAR)
    assert cfg.prompt == "v2" and cfg.copy_penalty == {"pool": ()} and cfg.pool_files() == ()
    assert cfg.structured_outputs["sha256"] == "1412f11f9ebdc520b50c263eda14be12829951357c17f43496ab2ddf5235e64e"


def test_the_pool_lists_ls_files_and_directories_of_them(tmp_path):
    p = _write(tmp_path, lambda d: d.update(copy_penalty={"pool": ["drafts/v1", "shootouts/loops/l5_wistful_mid.ls"]}))
    cfg = rc.load_config(p)
    files = cfg.pool_files()
    assert len(files) == len(list((ROOT / "drafts" / "v1").glob("*.ls"))) + 1 == 31
    assert files[-1] == ROOT / "shootouts" / "loops" / "l5_wistful_mid.ls" and all(f.suffix == ".ls" for f in files)


def test_the_trainer_hands_the_configs_pool_to_the_copy_penalty(tmp_path):
    # train() and the dry run build P_copy with one function (train.copy_penalty): the config's pool files, keyed by
    # their repo path (root=ROOT), after every example prompt v2 shows (D-020: scored against the seed pool too)
    doc = yaml.safe_load(V2_GRAMMAR.read_text())
    doc["copy_penalty"] = {"pool": ["drafts/v1"]}
    p = tmp_path / "pool.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    cfg = rc.load_config(p)
    n = len(list((ROOT / "drafts" / "v1").glob("*.ls")))
    cp = train.dry_run(cfg, out=io.StringIO())["reward"]["copy_penalty"]
    assert cp["pool"]["files"] == n == 30 and cp["references"] == 9 + n
    assert len(cp["pool"]["ids"]) == 20 and all(i.startswith("drafts/v1/") for i in cp["pool"]["ids"])
    pal = load_palette(ROOT / cfg.palette, verify=False)
    copy = train.copy_penalty(cfg, pal, train.prompt_module("v2"))
    assert copy.pool == tuple(str(f.relative_to(ROOT)) for f in cfg.pool_files())


def test_v2_rows_record_their_shown_example_and_v1_rows_are_unchanged():
    from lofi_rl import prompt, prompt_v2
    v1, v2 = rc.load_config(SMOKE), rc.load_config(V2_GRAMMAR)
    pal = load_palette(ROOT / v1.palette, verify=False)
    rows1 = train.build_rows(v1, pal, train.prompt_module("v1"))
    rows2 = train.build_rows(v2, pal, train.prompt_module("v2"))
    briefs = sample_briefs(v2.brief_seed, v2.brief_count, v2.brief_start)
    assert all(set(r) == {"prompt", "brief", "brief_id"} for r in rows1), "no new column for v1"
    assert [r["prompt"] for r in rows1] == [prompt.build_messages(b, pal) for b in briefs]
    assert [r["prompt"] for r in rows2] == [prompt_v2.build_messages(b, pal) for b in briefs]
    assert [r["example_id"] for r in rows2] == [prompt_v2.example_id(b) for b in briefs]
    assert train.row_example_ids(rows1) == ["example"] * 40
    rec = train.prompt_record(v2, train.prompt_module("v2"), rows2)
    assert rec["version"] == "v2" and rec["module"] == "lofi_rl.prompt_v2" and rec["template_sha256"] == V2_TEMPLATE
    assert rec["example_ids"] == [r["example_id"] for r in rows2] and sum(rec["example_counts"].values()) == 40
    assert rec["messages_sha256"] == [prompt.messages_sha256(r["prompt"]) for r in rows2]
    assert train.prompt_record(v1, train.prompt_module("v1"), rows1)["template_sha256"] == V1_TEMPLATE


def test_a_prompt_module_of_another_version_is_refused(monkeypatch):
    monkeypatch.setitem(sys.modules, "lofi_rl.prompt_v2", fake_prompt())       # says PROMPT_VERSION "v1"
    with pytest.raises(rc.ConfigError, match="PROMPT_VERSION is 'v1', not the config's prompt 'v2'"):
        train.prompt_module("v2")
    with pytest.raises(rc.ConfigError, match="v9"):
        train.prompt_module("v9")


def test_the_dry_run_of_the_v2_smoke_reports_the_prompt_the_copy_penalty_and_the_language():
    from lofi_rl import prompt_v2
    buf = io.StringIO()
    report = train.dry_run(rc.load_config(V2_GRAMMAR), out=buf)
    assert json.loads(buf.getvalue()) == json.loads(json.dumps(report, default=str))
    p = report["prompts"]
    briefs = sample_briefs(596, 40, 0)
    assert (p["version"], p["module"], p["template_sha256"]) == ("v2", "lofi_rl.prompt_v2", V2_TEMPLATE)
    assert p["example_ids"] == [prompt_v2.example_id(b) for b in briefs] and len(p["messages_sha256"]) == 40
    assert p["distinct"] == len({(b.energy, b.mood) for b in briefs}), "one message pair per energy x mood cell"
    cp = report["reward"]["copy_penalty"]
    assert cp["prompt_version"] == "v2" and cp["references"] == 9 and cp["pool"]["files"] == 0
    assert (cp["j_free"], cp["j_full"], cp["p_max"], cp["ngram"]) == (0.6, 0.8, 0.3, 5)
    assert report["reward"]["in_language"]["regex_sha256"] == report["structured_outputs"]["sha256"]
    assert report["backend_pin"]["requested"] == "xgrammar"
    check = report["reward_check"]
    assert (check["copy_j"], check["copy_nearest"], check["p_copy"]) == (1.0, "example", 0.3), \
        "example.ls is v1's example with a sax line: a copy"
    assert check["penalty"] == pytest.approx(0.3) and check["example_id"] == prompt_v2.example_id(briefs[0])
    assert check["in_language"] is False, "example.ls has a sax: line, which the regex does not take"
    assert report["trl_logp_metric"]["verdict"] in ("not_found", "match", "not_matched")


def test_the_dry_run_of_todays_smoke_scores_the_example_as_before():
    report = train.dry_run(rc.load_config(SMOKE), out=io.StringIO())
    check = report["reward_check"]
    assert not {"p_copy", "copy_j", "in_language", "example_id"} & set(check)
    assert report["reward"]["copy_penalty"] is None and report["reward"]["in_language"] is None
    assert report["backend_pin"] is None and report["prompts"]["example_counts"] == {"example": 40}


def test_the_v2_dry_run_runs_with_torch_trl_and_vllm_blocked():
    block = "import sys\n" + "".join(f"sys.modules[{m!r}] = None\n" for m in HEAVY)
    code = (block + CHILD_FAKES + "from lofi_rl.rl import train\n"
            f"rc = train.main(['--config', {str(V2_GRAMMAR)!r}, '--dry-run'])\n"
            f"assert not [m for m in {HEAVY!r} if sys.modules.get(m) is not None]\nraise SystemExit(rc)\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=env, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-3000:]
    out = json.loads(p.stdout)
    assert out["prompts"]["version"] == "v2" and out["trl_logp_metric"]["verdict"] == "not_found"


# ------------------------------------------------------------------------------------------ D-021: the masked trainer
MASKED = ROOT / "configs" / "grpo_smoke_v2_grammar_masked.yaml"
REGEX_SHA = "1412f11f9ebdc520b50c263eda14be12829951357c17f43496ab2ddf5235e64e"     # dev_gu's (D-021, M4)
REWARD_CHECK = 0.2784511276657195                                                    # 13657200's dry run (M4)
SPEC_VALUES = ("grammar_masked", "grammar_masked_fp32")                              # the build spec's TRAINER_LOGPROBS
GATES = ("T1", "T2", "T3", "T4", "T5")
EOS_IDS = (151645, 151643)
MATCHED = {"verdict": "match", "path": "support_ids", "gating": list(GATES), "failed": [],
           **{t: {"verdict": "match"} for t in GATES + ("V1", "V2")}}


def _masked():
    """lofi_rl.rl.masked (section D's module), imported where a test needs it, so a broken module fails those tests
    and not this file's import."""
    return importlib.import_module("lofi_rl.rl.masked")


def _masked_doc(edit=lambda d: None) -> dict:
    doc = yaml.safe_load(MASKED.read_text())
    edit(doc)
    return doc


def _write_doc(tmp_path, doc, name="masked.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


def test_the_masked_smoke_is_the_v2_grammar_smoke_plus_trainer_logprobs():
    v2, masked = yaml.safe_load(V2_GRAMMAR.read_text()), yaml.safe_load(MASKED.read_text())
    assert masked.pop("trainer_logprobs") == "grammar_masked"
    assert masked == v2, "every other key is configs/grpo_smoke_v2_grammar.yaml's"
    cfg, base = rc.load_config(MASKED), rc.load_config(V2_GRAMMAR)
    assert (cfg.trainer_logprobs, cfg.mask_path, base.trainer_logprobs, base.mask_path) == \
        ("grammar_masked", "support_ids", None, None)
    assert cfg.structured_outputs["sha256"] == REGEX_SHA
    assert dict(cfg.structured_outputs) == dict(base.structured_outputs)
    for profile in cfg.profiles:
        assert rc.resolve(cfg, profile) == rc.resolve(base, profile), "trainer_logprobs is no GRPOConfig field"
    head = MASKED.read_text().split("\nrun:")[0]
    assert "D-021" in head and "1dded12d386c7fab" in head
    assert ("make submit JOB=jobs/grpo_smoke.sbatch OVR='--gres=gpu:l40s:1 --constraint=l40s' "
            "ARGS='configs/grpo_smoke_v2_grammar_masked.yaml'") in head, "D-021's L40S command"


def test_trainer_logprobs_takes_the_masked_modules_values_and_paths(tmp_path):
    m = _masked()
    assert tuple(m.TRAINER_LOGPROBS) == SPEC_VALUES, "config.py imports them; the build spec names these two"
    assert set(rc.MASK_T2_PATHS) == set(m.TRAINER_LOGPROBS) and "MASK_PATHS" not in vars(rc), "no copy of D's map"
    assert rc.MASK_T2_PATHS["grammar_masked"] == ("support_ids",), "it hands TRL the support: TRL's own form only"
    assert set(rc.MASK_T2_PATHS["grammar_masked_fp32"]) == {"support_ids", "fill"}, "it computes its own log-probs"
    for value, path in (("grammar_masked", "support_ids"), ("grammar_masked_fp32", "fill")):
        doc = _masked_doc(lambda d: d.update(trainer_logprobs=value))
        cfg = rc.load_config(_write_doc(tmp_path, doc, f"{value}.yaml"))
        assert cfg.trainer_logprobs == value and cfg.mask_path == m.path_for(value) == path


@pytest.mark.parametrize("edit, match", [
    (lambda d: d.update(trainer_logprobs="grammar"), "trainer_logprobs 'grammar' is not one of"),
    (lambda d: d.update(trainer_logprobs=True), "trainer_logprobs True is not one of"),
    (lambda d: d.update(trainer_logprobs=None), "trainer_logprobs None is not one of"),
    (lambda d: d.pop("structured_outputs"), "trainer_logprobs 'grammar_masked' needs structured_outputs"),
])
def test_a_bad_trainer_logprobs_is_refused(tmp_path, edit, match):
    with pytest.raises(rc.ConfigError, match=match):
        rc.load_config(_write_doc(tmp_path, _masked_doc(edit)))


def test_only_a_config_with_the_key_imports_the_masked_module_and_one_that_cannot_is_refused():
    code = ("import sys\nsys.modules['lofi_rl.rl.masked'] = None\n"        # an import of it raises ImportError
            "from lofi_rl.rl import config as rc\n"
            f"rc.load_config({str(V2_GRAMMAR)!r})\n"
            "try:\n"
            f"    rc.load_config({str(MASKED)!r})\n"
            "except rc.ConfigError as e:\n    print('REFUSED', e)\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-3000:]
    assert "REFUSED trainer_logprobs 'grammar_masked': lofi_rl.rl.masked" in p.stdout
    assert "cannot be imported" in p.stdout


# ------------------------------------------------------------------------------------------ D-021: the dry run (M4)
def test_the_masked_dry_run_reports_m4_and_what_a_launch_would_refuse_on():
    report = train.dry_run(rc.load_config(MASKED), out=io.StringIO())
    assert report["structured_outputs"]["sha256"] == REGEX_SHA
    assert report["prompts"]["template_sha256"] == V2_TEMPLATE and report["reward_check"]["reward"] == REWARD_CHECK
    assert report["trainer_logprobs"] == "grammar_masked"
    m = report["masked"]
    assert (m["mode"], m["path_expected"], m["t2_paths"]) == ("grammar_masked", "support_ids", ["support_ids"])
    checks = m["source_checks"]
    assert checks["verdict"] == "not_matched" and checks["path"] is None, "the laptop has no TRL, vLLM or PEFT to read"
    assert all(checks[t]["verdict"] == "not_found" for t in GATES + ("V1", "V2"))
    assert m["refusals"][-1] == "T2 path None: trainer_logprobs grammar_masked runs only under ['support_ids']"
    assert train.dry_run(rc.load_config(V2_GRAMMAR), out=io.StringIO())["masked"] is None


D020_DRY_RUN = ROOT / "runs" / "grpo-smoke-13657200" / "summary" / "dry_run.json"


def test_the_masked_dry_run_is_13657200s_dry_run_plus_the_masked_keys():
    # D-021 M4: regex 1412f11f..., template 127e3094... and reward check 0.2784511276657195, as 13657200 reported
    if not D020_DRY_RUN.is_file():
        pytest.skip(f"{D020_DRY_RUN.relative_to(ROOT)} is not on this machine (make pull)")
    old = json.loads(D020_DRY_RUN.read_text())
    new = json.loads(json.dumps(train.dry_run(rc.load_config(MASKED), out=io.StringIO()), default=str))
    assert new.pop("trainer_logprobs") == "grammar_masked" and new.pop("masked")["path_expected"] == "support_ids"
    for key in ("config", "trl_logp_metric"):       # another file; the TRL the cluster had and the laptop lacks
        old.pop(key), new.pop(key)
    assert new == old


def test_the_dry_run_hands_the_source_checks_the_settings_every_profile_shares(monkeypatch):
    calls = []

    def fake(*, args=None, **kw):
        calls.append(args)
        return deepcopy(MATCHED)
    monkeypatch.setattr(trl_probe, "source_checks", fake)
    cfg = rc.load_config(MASKED)
    m = train.dry_run(cfg, out=io.StringIO())["masked"]
    assert m["source_checks"] == MATCHED and m["refusals"] == []
    (args,) = calls
    assert (args["top_p"], args["top_k"], args["temperature"], args["beta"]) == (1.0, 0, 1.0, 0.04)
    assert args[REGEX_FIELD] == cfg.structured_outputs["regex"]
    own = {"per_device_train_batch_size", "gradient_accumulation_steps", "vllm_gpu_memory_utilization"}
    assert not own & set(args), "the profiles' own settings differ, so the dry run passes none of them"


def test_the_masked_dry_run_runs_with_torch_trl_vllm_and_xgrammar_blocked():
    heavy = HEAVY + ("xgrammar",)
    block = "import sys\n" + "".join(f"sys.modules[{m!r}] = None\n" for m in heavy)
    code = (block + CHILD_FAKES + "from lofi_rl.rl import train\n"
            f"rc = train.main(['--config', {str(MASKED)!r}, '--dry-run'])\n"
            f"assert not [m for m in {heavy!r} if sys.modules.get(m) is not None]\nraise SystemExit(rc)\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-3000:]
    out = json.loads(p.stdout)
    assert out["trainer_logprobs"] == "grammar_masked" and out["masked"]["source_checks"]["verdict"] == "not_matched"


# ------------------------------------------------------------------------------------------ D-021: refusing a launch
def _checks(**changes):
    rec = deepcopy(MATCHED)
    rec.update(changes)
    return rec


@pytest.mark.parametrize("mode, checks, says", [
    ("grammar_masked", _checks(), None),
    ("grammar_masked_fp32", _checks(), None),
    ("grammar_masked_fp32", _checks(path="fill"), None),
    ("grammar_masked", _checks(path="fill"),
     "T2 path 'fill': trainer_logprobs grammar_masked runs only under ['support_ids']"),
    ("grammar_masked",
     _checks(path=None, verdict="not_matched", T2={"verdict": "not_matched", "missing": ["no support"]}),
     "T2 not_matched (['no support'])"),
    ("grammar_masked", {k: v for k, v in MATCHED.items() if k != "T5"}, "T5 absent"),
    ("grammar_masked", _checks(T4={"verdict": "error", "error": "SyntaxError"}), "T4 error (SyntaxError)"),
    ("grammar_masked", _checks(verdict="not_matched"), "source_checks verdict 'not_matched'"),
    ("grammar_masked", _checks(gating=list(GATES) + ["T6"]), "T6 absent"),
])
def test_a_launch_trains_only_when_every_gating_check_matches_and_t2s_path_fits_its_mode(mode, checks, says):
    out = train.mask_refusals(mode, checks)
    assert (out == []) if says is None else any(says in r for r in out), out


def test_a_source_check_that_raises_or_is_missing_refuses_the_launch(monkeypatch):
    def boom(**kw):
        raise RuntimeError("no TRL here")
    monkeypatch.setattr(trl_probe, "source_checks", boom)
    rec = train.source_checks({"top_k": 0})
    assert rec["verdict"] == "not_matched" and "RuntimeError: no TRL here" in rec["error"] and rec["path"] is None
    assert any("no TRL here" in r for r in train.mask_refusals("grammar_masked", rec))
    monkeypatch.delattr(trl_probe, "source_checks")
    assert "AttributeError" in train.source_checks({})["error"]
    monkeypatch.setattr(trl_probe, "lm_head_record", lambda model: "fine")
    assert train.lm_head_check(object())["verdict"] == "error", "a record, or the launch is refused"
    assert issubclass(train.Refused, rc.ConfigError), "main() turns it into exit 2"


# ------------------------------------------------------------------------------------------ D-021: the step log
def test_the_step_log_writes_each_steps_mask_record_at_the_rewards_offset(tmp_path):
    clock = iter([100.0, 130.0, 161.0, 162.0])
    rewards, masks = {0: {"reward_mean": .1}, 1: {"reward_mean": .2}}, {0: {"violations": 0, "rows": {"old": 32}},
                                                                        1: {"violations": 0, "rows": {"old": 32}}}
    log = train.StepLog(tmp_path / "steps.jsonl", reward_stats=rewards, clock=lambda: next(clock), mask_stats=masks)
    log.train_begin(0)
    log.step_end(1); log.log(1, {"loss": 0.0})
    log.step_end(2); log.log(2, {"loss": 0.0})
    log.log(2, {"train_runtime": 2.0})
    rows = [json.loads(l) for l in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [(r["kind"], r.get("mask")) for r in rows] == [("step", masks[0]), ("step", masks[1]), ("train_end", None)]
    assert [r["lofi"] for r in rows[:2]] == [rewards[0], rewards[1]], "the same rollouts, the same offset"
    plain = train.StepLog(tmp_path / "plain.jsonl", reward_stats=rewards)
    plain.train_begin(0); plain.step_end(1); plain.log(1, {})
    assert "mask" not in json.loads((tmp_path / "plain.jsonl").read_text()), "an unmasked run's lines are unchanged"


def test_run_json_records_the_code_tree_and_whether_it_is_a_snapshot(tmp_path, monkeypatch):
    monkeypatch.delenv("LOFI_CODE", raising=False)
    rec = train.code_record(tmp_path)
    assert rec == {"lofi_code": None, "lofi_code_snapshot": None, "root": str(tmp_path), "root_is_lofi_code": None,
                   "root_snapshot": False}
    (tmp_path / ".lofi-snapshot").write_text("")
    monkeypatch.setenv("LOFI_CODE", str(tmp_path))
    rec = train.code_record(tmp_path)
    assert rec["lofi_code_snapshot"] is True and rec["root_is_lofi_code"] is True and rec["root_snapshot"] is True
    assert train.code_record(ROOT)["root_is_lofi_code"] is False


def test_run_json_versions_name_the_stack_accelerate_and_xgrammar_from_their_metadata(monkeypatch):
    asked = []

    def version(name):
        asked.append(name)
        if name == "peft":
            raise importlib.metadata.PackageNotFoundError(name)
        return f"{name} (metadata stand-in)"

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert "xgrammar" not in train.STACK, "main()'s environment check does not require xgrammar"
    got = train._versions()
    assert asked == [*train.STACK, "accelerate", "xgrammar"]
    assert got == {**{name: f"{name} (metadata stand-in)" for name in asked}, "peft": None}


# ------------------------------------------------------------------------------------------ D-021: train()'s wiring
class _FakeLLM:
    def __init__(self, model=None, **kw):
        self.kw = kw


class _Walker:
    """The replay's and the NFA's identity, as lofi_rl.rl.masked.MaskSetup checks it: one regex, vocabulary and set of
    stop ids (D-021 M8). Walking is section D's to test."""

    def __init__(self, regex: str, vocab_size: int, stop_ids) -> None:
        self.regex_sha256 = hashlib.sha256(regex.encode("utf-8")).hexdigest()
        self.vocab_size, self.stop_ids = vocab_size, tuple(sorted(stop_ids))

    def walk(self, ids):
        raise AssertionError("no completion is walked in train()'s wiring test")


def _mask_stack(monkeypatch, tmp_path, *, checks=None, head=None, trainer_class=None) -> dict[str, Any]:
    """train()'s masked path, with section D's module (lofi_rl.rl.masked: its values, paths and MaskSetup) and
    stand-ins only for what needs the cluster: torch, transformers, TRL, PEFT, datasets and vLLM as
    tests/test_rl_rehearsal.py fakes them; xgrammar's replay (masked.XgrammarReplay) and the torch-backed trainer class
    (masked.make_trainer_class); section A's GrammarMask over a tokenizer.json; and section C's source_checks and
    lm_head_record, which return `checks` and `head`. The fake GRPOTrainer's loop calls
    _generate_and_score_completions once a step, as TRL does, and the stand-in subclass records the step there."""
    seen: dict[str, Any] = {"loads": 0, "trainers": [], "replays": [], "nfas": [], "check_args": [], "heads": [],
                            "loads_at_check": None}
    cfg = rc.load_config(MASKED)
    names = {"output_dir", "run_name"} | {k for p in cfg.profiles for k in rc.resolve(cfg, p)}
    GRPOConfig = dataclasses.make_dataclass("GRPOConfig", [(n, Any, None) for n in sorted(names)],
                                            namespace={"to_dict": lambda self: dataclasses.asdict(self)})

    class Tok:
        eos_token_id, padding_side, truncation_side = EOS_IDS[0], "right", "right"

        @classmethod
        def from_pretrained(cls, path):
            return cls()

        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            return " ".join(m["content"] for m in msgs)

        def __call__(self, text, add_special_tokens):
            return {"input_ids": list(range(len(text.split())))}

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True)

        @classmethod
        def from_pretrained(cls, path, dtype):
            seen["loads"] += 1
            return cls()

        def get_nb_trainable_parameters(self):
            return 1, 2

    class GRPOTrainer:
        def __init__(self, model, reward_funcs, args, train_dataset, processing_class, peft_config, callbacks):
            self.model, self.args, self.tok, self.callbacks = model, args, processing_class, callbacks
            self.state = SimpleNamespace(global_step=0)
            self.llm = sys.modules["vllm"].LLM(model="snapshot")
            seen["trainers"].append(self)

        def _generate(self, prompts):
            return None

        def _generate_and_score_completions(self, inputs):
            return {}

        def _get_per_token_logps_and_entropies(self, *a, **k):
            return None

        def train(self, resume_from_checkpoint=None):
            a = self.args
            for cb in self.callbacks:
                cb.on_train_begin(a, self.state, None)
            while self.state.global_step < a.max_steps:
                self._generate_and_score_completions({})
                self.state.global_step += 1
                for cb in self.callbacks:
                    cb.on_step_end(a, self.state, None)
                    cb.on_log(a, self.state, None, logs={"loss": 0.0, "kl": 0.0, "entropy": 0.05})
            return SimpleNamespace(metrics={"train_runtime": 1.0})

    def make_trainer_class(base):
        if trainer_class is not None:
            return trainer_class(base)

        class LofiGRPOTrainer(base):
            def __init__(self, *a, mask_setup, **k):
                self.mask_setup = mask_setup
                super().__init__(*a, **k)

            def _generate(self, prompts):
                return super()._generate(prompts)

            def _generate_and_score_completions(self, inputs):
                self.mask_setup.stats[self.state.global_step] = {"generated_at": self.state.global_step,
                                                                 "path": self.mask_setup.path}
                return super()._generate_and_score_completions(inputs)

            def _get_per_token_logps_and_entropies(self, *a, **k):
                return super()._get_per_token_logps_and_entropies(*a, **k)
        return LofiGRPOTrainer

    class XgrammarReplay(_Walker):
        def __init__(self, regex, tokenizer, *, vocab_size, stop_ids):
            super().__init__(regex, vocab_size, stop_ids)
            self.tokenizer = tokenizer
            self.record = {"xgrammar": "stand-in", "regex_sha256": self.regex_sha256, "vocab_size": vocab_size,
                           "stop_ids": list(self.stop_ids)}
            seen["replays"].append(self)

    class GrammarMask(_Walker):
        @classmethod
        def from_tokenizer_json(cls, path, regex, *, vocab_size, stop_ids):
            nfa = cls(regex, vocab_size, stop_ids)
            nfa.path = path
            seen["nfas"].append(nfa)
            return nfa

        def record(self):
            return {"builder": "stand-in", "tokenizer_json": str(self.path), "regex_sha256": self.regex_sha256,
                    "vocab_size": self.vocab_size, "stop_ids": list(self.stop_ids)}

    real = _masked()
    masked = types.ModuleType("lofi_rl.rl.masked")
    masked.__dict__.update({k: v for k, v in vars(real).items() if not k.startswith("__")})
    masked.XgrammarReplay, masked.make_trainer_class = XgrammarReplay, make_trainer_class
    gm = types.ModuleType("lofi_rl.grammar_mask")
    gm.GrammarMask = GrammarMask

    def fake_checks(*, args=None, **kw):
        seen["check_args"].append(args)
        seen["loads_at_check"] = seen["loads"]
        return deepcopy(checks if checks is not None else MATCHED)

    def fake_head(model):
        seen["heads"].append(model)
        return dict(head if head is not None else {"verdict": "match", "missing": [], "error": None})

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        return m
    card = SimpleNamespace(name="FAKE L40S", total_memory=44.39 * 2**30)       # the 48gb profile, as 13657200's
    cuda = SimpleNamespace(is_available=lambda: True, get_device_properties=lambda i: card)
    for name, m in {"torch": mod("torch", cuda=cuda, bfloat16="bf16"),
                    "datasets": mod("datasets", Dataset=SimpleNamespace(from_list=list)),
                    "peft": mod("peft", LoraConfig=lambda **kw: dict(kw)),
                    "transformers": mod("transformers", AutoModelForCausalLM=Model, AutoTokenizer=Tok,
                                        TrainerCallback=object, GenerationConfig=SimpleNamespace(
                                            from_pretrained=lambda p: SimpleNamespace(eos_token_id=list(EOS_IDS)))),
                    "trl": mod("trl", GRPOConfig=GRPOConfig, GRPOTrainer=GRPOTrainer),
                    "vllm": mod("vllm", LLM=type("LLM", (_FakeLLM,), {})),
                    "lofi_rl.rl.masked": masked, "lofi_rl.grammar_mask": gm}.items():
        monkeypatch.setitem(sys.modules, name, m)
    monkeypatch.setitem(sys.modules, "lofi_rl.verify", fake_verify())
    monkeypatch.setattr(trl_probe, "source_checks", fake_checks)
    monkeypatch.setattr(trl_probe, "lm_head_record", fake_head)
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"vocab_size": 151936, "tie_word_embeddings": True}))
    (snap / "tokenizer.json").write_text("{}")
    import kit.hf_cache
    monkeypatch.setattr(kit.hf_cache, "resolve_pinned", lambda repo, rev, need: str(snap))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    seen.update(cfg=cfg, GRPOTrainer=GRPOTrainer, snapshot=snap, MaskSetup=real.MaskSetup)
    return seen


def _run_json(tmp_path) -> dict:
    return json.loads((tmp_path / "summary" / "run.json").read_text())


def test_a_masked_launch_checks_the_sources_builds_the_mask_and_trains_the_masked_subclass(monkeypatch, tmp_path):
    seen = _mask_stack(monkeypatch, tmp_path)
    cfg = seen["cfg"]
    assert train.train(cfg, run_dir=tmp_path / "a", summary_dir=tmp_path / "summary") == 0
    man = _run_json(tmp_path)
    assert man["status"] == "ok" and man["trainer_logprobs"] == "grammar_masked" and man["profile"] == "48gb"
    assert man["trl_source_checks"] == MATCHED and man["lm_head"]["verdict"] == "match"
    assert seen["loads_at_check"] == 0, "the source checks ran before the model loaded"
    (args,) = seen["check_args"]
    assert args["output_dir"] == str(tmp_path / "a") and (args["top_p"], args["top_k"]) == (1.0, 0) \
        and args["per_device_train_batch_size"] == 4, "the launch's own GRPOConfig, fitted to the installed TRL"
    (t,) = seen["trainers"]
    assert type(t).__name__ == "LofiGRPOTrainer" and isinstance(t, seen["GRPOTrainer"])
    s = t.mask_setup
    assert isinstance(s, seen["MaskSetup"]), "section D's MaskSetup, which checks the replay's and the NFA's identity"
    assert (s.mode, s.regex, s.regex_sha256, s.vocab_size, s.stop_ids, s.path) == (
        "grammar_masked", cfg.structured_outputs["regex"], REGEX_SHA, 151936, (151643, 151645), "support_ids")
    (replay,), (nfa,) = seen["replays"], seen["nfas"]
    assert s.replay is replay and s.nfa is nfa and replay.tokenizer is t.tok, "vLLM's tokenizer: the trainer's"
    assert nfa.path == seen["snapshot"] / "tokenizer.json" and seen["heads"] == [t.model]
    m = man["masked"]
    assert (m["mode"], m["path"], m["regex_sha256"], m["vocab_size"], m["stop_ids"]) == (
        "grammar_masked", "support_ids", REGEX_SHA, 151936, [151643, 151645])
    assert m["replay"] == replay.record and m["nfa"] == nfa.record() and m["selfcheck_rows"] == 2
    assert m["trainer_class"].endswith("LofiGRPOTrainer") and m["vocab_from"] == str(seen["snapshot"] / "config.json")
    assert m["stats_steps"] == list(range(10)) and m["last_stats"] == {"generated_at": 9, "path": "support_ids"}
    assert set(man["code"]) == {"lofi_code", "lofi_code_snapshot", "root", "root_is_lofi_code", "root_snapshot"}
    steps = [json.loads(l) for l in (tmp_path / "summary" / "steps.jsonl").read_text().splitlines()]
    logged = [(r["step"], r["mask"]["generated_at"]) for r in steps if r["kind"] == "step"]
    assert logged == [(k, k - 1) for k in range(1, 11)], "record k + 1 carries generation step k's"


def test_a_source_check_that_does_not_match_refuses_the_launch_before_the_model_loads(monkeypatch, tmp_path):
    seen = _mask_stack(monkeypatch, tmp_path, checks=_checks(verdict="not_matched", path=None,
                                                            T2={"verdict": "not_matched", "missing": ["no support"]}))
    with pytest.raises(train.Refused, match="T2 not_matched"):
        train.train(seen["cfg"], run_dir=tmp_path / "a", summary_dir=tmp_path / "summary")
    man = _run_json(tmp_path)
    assert man["status"] == "refused" and "T2 not_matched" in man["error"] and "T2 path None" in man["error"]
    assert man["trl_source_checks"]["T2"]["verdict"] == "not_matched" and "masked" not in man
    assert seen["loads"] == 0 and seen["replays"] == [] and seen["trainers"] == []


def test_a_trl_form_the_mode_cannot_run_under_refuses_it(monkeypatch, tmp_path):
    seen = _mask_stack(monkeypatch, tmp_path, checks=_checks(path="fill"))
    with pytest.raises(train.Refused, match="T2 path 'fill'"):
        train.train(seen["cfg"], run_dir=tmp_path / "a", summary_dir=tmp_path / "summary")
    assert _run_json(tmp_path)["status"] == "refused" and seen["loads"] == 0


def test_lm_head_as_a_lora_target_refuses_the_launch_once_the_trainer_is_built(monkeypatch, tmp_path):
    seen = _mask_stack(monkeypatch, tmp_path, head={"verdict": "not_matched", "error": None,
                                                    "missing": ["tuner layers named *lm_head: ['lm_head']"]})
    with pytest.raises(train.Refused, match=r"T5 at run time.*tuner layers named \*lm_head"):
        train.train(seen["cfg"], run_dir=tmp_path / "a", summary_dir=tmp_path / "summary")
    man = _run_json(tmp_path)
    assert man["status"] == "refused" and man["lm_head"]["verdict"] == "not_matched"
    assert len(seen["trainers"]) == 1 and man["masked"]["stats_steps"] == [], "refused before any step"


def test_a_trainer_class_that_does_not_override_the_mask_methods_is_refused_before_the_model_loads(monkeypatch,
                                                                                                    tmp_path):
    def lazy(base):
        class Lazy(base):
            def __init__(self, *a, mask_setup, **k):
                super().__init__(*a, **k)
        return Lazy
    seen = _mask_stack(monkeypatch, tmp_path, trainer_class=lazy)
    with pytest.raises(train.Refused, match="inherits .*_get_per_token_logps_and_entropies"):
        train.train(seen["cfg"], run_dir=tmp_path / "a", summary_dir=tmp_path / "summary")
    assert _run_json(tmp_path)["status"] == "refused" and seen["loads"] == 0
    with pytest.raises(train.Refused, match="not a subclass"):
        train.check_trainer_class(object, seen["GRPOTrainer"])


def test_a_refused_launch_exits_2(monkeypatch, capsys):
    def refused(cfg, **kw):
        raise train.Refused("the source checks refuse the masked trainer: T3 not_matched")
    monkeypatch.setattr(train, "train", refused)
    monkeypatch.setattr(train.importlib.util, "find_spec", lambda name: object())    # as if the policy env were here
    for k, v in (("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "1"), ("TOKENIZERS_PARALLELISM", "false")):
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(train.faulthandler, "enable", lambda *a, **k: None)     # pytest captures stderr
    monkeypatch.setattr(train.faulthandler, "register", lambda *a, **k: None)
    monkeypatch.setattr(train, "_HARD_EXIT", False)
    assert train.main(["--config", str(MASKED)]) == 2
    assert "train: the source checks refuse the masked trainer: T3 not_matched" in capsys.readouterr().err
