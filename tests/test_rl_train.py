"""The GRPO run config (configs/grpo_smoke.yaml, src/lofi_rl/rl/config.py) and the trainer's laptop half
(src/lofi_rl/rl/train.py): the §6 settings, validation, fitting the settings to TRL's GRPOConfig fields, the
prompts, `--dry-run` without torch/TRL/vLLM, and the per-step log. Training itself needs a GPU and the
policy env, so jobs/grpo_smoke.sbatch tests it on the cluster."""
import dataclasses
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from helpers_rl import CHILD_FAKES, fake_prompt, fake_verify
from lofi_rl import grammar
from lofi_rl.brief import Brief, sample_briefs
from lofi_rl.palette import load_palette
from lofi_rl.rl import config as rc
from lofi_rl.rl import train

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


@pytest.mark.parametrize("path", [SMOKE, CONSTRAINED], ids=["smoke", "constrained"])
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
