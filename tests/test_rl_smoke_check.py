"""The GRPO smoke's verdict (src/lofi_rl/rl/smoke_check.py), on synthetic run directories laid out like the
ones jobs/grpo_smoke.sbatch leaves: run a trains steps 1-10 and saves checkpoints 5 and 10; run b resumes
from a/checkpoint-5 and must train exactly steps 6-10 (taste_pipeline.md §6, smoke row).

The masked smoke (D-021 Part A) adds smoke.json's d021 block: a passing smoke, invalid ones (a sampled token outside
the mask, an NFA mismatch, a source check that does not match, and each other validity gate), measured misses (M9
over 0.05, M10's ratio, collapsed IS weights), gates that cannot be decided, M11's stand-in and M12. M5's
registration half reads a frozen tree of the tests' own (frozen_root): a changed, missing or malformed entry, a
missing list and a DECISIONS.md without D-021 each fail it. Also: a step whose old pass had no valid token weighs 0
in M10, the self-check's flags (warnings, never the verdict), M12's timing breakdown, and the masked IS caveat."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from lofi_rl.rl import smoke_check as sc

GROUP, PROMPTS = 8, 4


def _ckpt(run: Path, step: int, files=("adapter_model.safetensors", "optimizer.pt", "scheduler.pt")):
    d = run / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": 10}))
    for f in files:
        (d / f).write_bytes(b"x")


def _summary(out: Path, steps, *, resume_from=None, start_step=0, errors=0, status="ok"):
    out.mkdir(parents=True)
    lines = []
    for k in steps:
        lines.append({"kind": "step", "step": k, "t": 1000.0 + 30 * k, "step_s": 30.0 + k,
                      "trl": {"reward": 0.01 * k, "reward_std": 0.1, "kl": 0.001 * k, "completions/mean_length": 300.0},
                      "lofi": {"reward_mean": 0.01 * k, "reward_std": 0.1, "gate_rate": 0.5, "errors": 0,
                               "step_source": "trainer_state"}})
    lines.append({"kind": "train_end", "step": steps[-1], "trl": {"train_runtime": 300.0}})
    (out / "steps.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    rec = []
    for k in steps:
        for i in range(GROUP * PROMPTS):
            rec.append({"step": k - 1, "step_source": "trainer_state", "call": k - 1 - start_step, "i": i,
                        "group": i // GROUP, "reward": 0.1, "gate": None, "G": 1,
                        **({"error": "ValueError: x"} if errors and k == steps[0] and i < errors else {})})
    (out / "rewards.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rec))
    (out / "run.json").write_text(json.dumps({"status": status, "resume_from": resume_from, "start_step": start_step,
                                              "final_global_step": steps[-1], "startup_s": 150.0, "profile": "a100-80gb",
                                              "dropped_optional_fields": []}))


def _good(tmp_path: Path) -> Path:
    run = tmp_path / "grpo-smoke-1"
    _ckpt(run / "a", 5); _ckpt(run / "a", 10); _ckpt(run / "b", 10)
    _summary(run / "summary" / "a", list(range(1, 11)))
    _summary(run / "summary" / "b", list(range(6, 11)), resume_from=str(run / "a" / "checkpoint-5"), start_step=5)
    return run


def _final(run, capsys):
    rc = sc.main(["--run", str(run), "--stage", "final"])
    return rc, json.loads((run / "summary" / "smoke.json").read_text()), capsys.readouterr().out


def test_a_good_smoke_passes_and_reports_reward_and_time_per_step(tmp_path, capsys):
    run = _good(tmp_path)
    assert sc.main(["--run", str(run), "--stage", "first"]) == 0
    assert "FIRST_RUN_OK" in capsys.readouterr().out
    rc, rep, out = _final(run, capsys)
    assert rc == 0 and rep["ok"] is True and rep["failures"] == [] and out.strip().endswith("GRPO_SMOKE_OK")
    assert [(s["run"], s["step"]) for s in rep["steps"]] == [("a", k) for k in range(1, 11)] + [("b", k) for k in range(6, 11)]
    s3 = rep["steps"][2]
    assert s3["reward_mean"] == pytest.approx(0.03) and s3["step_s"] == 33.0 and s3["kl"] == pytest.approx(0.003)
    assert rep["resumed"]["first_step"] == 6 and rep["resumed"]["steps"] == 5 and rep["resumed"]["final_global_step"] == 10
    assert rep["resumed"]["reward_steps"] == [5, 6, 7, 8, 9] and rep["first"]["reward_steps"] == list(range(10))
    assert rep["first"]["completions"] == 320 and rep["timing"]["first"]["startup_s"] == 150.0


def test_a_missing_checkpoint_fails_both_stages(tmp_path, capsys):
    run = _good(tmp_path)
    for f in (run / "a" / "checkpoint-5").iterdir():
        f.unlink()
    (run / "a" / "checkpoint-5").rmdir()
    assert sc.main(["--run", str(run), "--stage", "first"]) == 1
    assert "checkpoint-5" in capsys.readouterr().out
    rc, rep, out = _final(run, capsys)
    assert rc == 1 and "GRPO_SMOKE_FAILED" in out and any("checkpoint-5" in f for f in rep["failures"])


def test_a_checkpoint_without_its_optimizer_cannot_resume(tmp_path, capsys):
    run = _good(tmp_path)
    (run / "a" / "checkpoint-5" / "optimizer.pt").unlink()
    rc, rep, _ = _final(run, capsys)
    assert rc == 1 and any("optimizer.pt" in f for f in rep["failures"])


def test_a_resumed_run_that_restarted_from_step_0_fails(tmp_path, capsys):
    run = _good(tmp_path)
    b = run / "summary" / "b"
    for f in b.iterdir():
        f.unlink()
    b.rmdir()
    _summary(b, list(range(1, 11)), resume_from=str(run / "a" / "checkpoint-5"), start_step=0)
    rc, rep, _ = _final(run, capsys)
    assert rc == 1 and any("resumed" in f for f in rep["failures"])


def test_reward_errors_fail_the_smoke(tmp_path, capsys):
    run = _good(tmp_path)
    a = run / "summary" / "a"
    for f in a.iterdir():
        f.unlink()
    a.rmdir()
    _summary(a, list(range(1, 11)), errors=3)
    assert sc.main(["--run", str(run), "--stage", "first", "--rc", "a=0"]) == 0, "the resume test still runs"
    assert "FIRST_RUN_OK" in capsys.readouterr().out
    rc, rep, _ = _final(run, capsys)
    assert rc == 1 and any("3 reward errors" in f for f in rep["failures"])


def test_a_launch_that_never_wrote_its_summary_is_reported_not_raised(tmp_path, capsys):
    run = tmp_path / "grpo-smoke-2"
    (run / "summary").mkdir(parents=True)
    rc, rep, out = _final(run, capsys)
    assert rc == 1 and rep["ok"] is False and "GRPO_SMOKE_FAILED" in out
    assert any("summary/a" in f for f in rep["failures"])


def test_a_launch_that_exited_non_zero_fails_the_smoke_even_when_its_files_look_complete(tmp_path, capsys):
    run = _good(tmp_path)
    rc = sc.main(["--run", str(run), "--stage", "final", "--rc", "a=0", "--rc", "b=124"])
    rep = json.loads((run / "summary" / "smoke.json").read_text())
    assert rc == 1 and rep["exit_codes"] == {"a": "0", "b": "124"}
    assert any("launch b exited 124 (timed out)" in f for f in rep["failures"])
    assert sc.main(["--run", str(run), "--stage", "final", "--rc", "a=0", "--rc", "b=0"]) == 0
    assert capsys.readouterr().out.strip().endswith("GRPO_SMOKE_OK")


# ------------------------------------------------------------------------------------------ D-020's readouts
ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "configs" / "grpo_smoke_v2_grammar.yaml"
MATCH = {"verdict": "match", "package": "trl", "version": "1.17.1", "file": "trainer/grpo_trainer.py",
         "sha256": "a" * 64, "sites": [], "missing": [], "excludes_truncated": True, "lines": {"1": "x"}}


def _cfg(path=V2):
    from lofi_rl.rl.config import load_config
    return load_config(path)


def _d020(out: Path, steps, *, start_step=0, resume_from=None, cfg=None, out_of_language=(), source=MATCH,
          value=lambda s: 0.01, tokens=lambda k, i: 100, truncated=lambda k, i: False, copy=True, sha=None):
    """A launch's summary with the D-020 fields the reward and the trainer write: finish, in_language, p_copy and
    copy_j per completion; TRL's IS metric per step; the regex sha, the backend record, the copy penalty's pool (as
    train.py records it from the config) and trl_logp_metric."""
    cfg = cfg or _cfg()
    out.mkdir(parents=True)
    lines = [{"kind": "step", "step": s, "t": 1000.0 + s, "step_s": 30.0,
              "trl": {"reward": 0.1, "kl": 0.0, "completions/mean_length": 250.0,
                      **({} if value(s) is None else {sc.IS_METRIC: value(s)})},
              "lofi": {"reward_mean": 0.1, "reward_std": 0.1, "gate_rate": 1.0}} for s in steps]
    (out / "steps.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    rec = []
    for s in steps:
        k = s - 1
        for i in range(GROUP * PROMPTS):
            cut = truncated(k, i)
            r = {"step": k, "step_source": "trainer_state", "i": i, "group": i // GROUP, "reward": 0.1, "gate": None,
                 "G": 1, "n_tokens": 512 if cut else tokens(k, i), "truncated": cut,
                 "finish": "length" if cut else "stop", "in_language": (k, i) not in out_of_language}
            if copy:
                r.update(p_copy=0.3 if i == 0 else 0.0, copy_j=1.0 if i == 0 else 0.3, copy_nearest="example",
                         example_id="example" if i < 8 else "l8_hopeful_mid")
            rec.append(r)
    (out / "rewards.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rec))
    so = None if cfg.structured_outputs is None else {
        "sha256": sha or cfg.structured_outputs["sha256"],
        "backend": {"requested": "xgrammar", "pin_accepted": True, "backend": "xgrammar", "applied": True,
                    "engine": "trainer.llm"}}
    pool = [str(p.relative_to(ROOT)) for p in cfg.pool_files()]
    cp = None if cfg.copy_penalty is None else {"pool": {"files": len(pool), "ids": pool[:20]},
                                                "references": 9 + len(pool)}
    (out / "run.json").write_text(json.dumps({"status": "ok", "resume_from": resume_from, "start_step": start_step,
                                              "final_global_step": steps[-1], "startup_s": 60.0,
                                              "structured_outputs": so, "copy_penalty": cp,
                                              "trl_logp_metric": source}))


def _d020_run(tmp_path: Path, **kw) -> Path:
    run = tmp_path / "grpo-smoke-9"
    _ckpt(run / "a", 5); _ckpt(run / "a", 10); _ckpt(run / "b", 10)
    _d020(run / "summary" / "a", list(range(1, 11)), **kw)
    _d020(run / "summary" / "b", list(range(6, 11)), start_step=5, resume_from=str(run / "a" / "checkpoint-5"), **kw)
    return run


def test_a_d020_smoke_reports_the_language_the_backend_the_copies_and_the_is_readout(tmp_path, capsys):
    run = _d020_run(tmp_path, value=lambda s: 0.001 * s, tokens=lambda k, i: 200 + 10 * k)
    rc = sc.main(["--run", str(run), "--stage", "final", "--config", str(V2), "--rc", "a=0", "--rc", "b=0"])
    rep, out = json.loads((run / "summary" / "smoke.json").read_text()), capsys.readouterr().out
    assert rc == 0 and rep["ok"], rep["failures"]
    assert rep["prompt"] == "v2" and rep["copy_penalty"] == {"pool": []} and "regex" not in rep["structured_outputs"]
    assert rep["in_language"]["stop"] == {"n": 480, "in_language": 480, "out_of_language": 0}
    assert rep["in_language"]["launches"]["a"]["by_step"]["0"] == {"stop": 32, "stop_out_of_language": 0}
    assert rep["backend"]["a"]["applied"] is True and "backend a: requested xgrammar" in out
    assert rep["copy"]["a"] == {"pool_files": 0, "n": 320, "p_copy_mean": 0.3 / 32, "penalized": 10, "at_least_0.8": 10,
                                "j_max": 1.0, "nearest": {"example": 10}, "nearest_is_shown": 10}
    isr = rep["is_ratio"]
    steps = list(range(1, 11)) + list(range(6, 11))
    values, tokens = [0.001 * s for s in steps], [32 * (200 + 10 * (s - 1)) for s in steps]
    assert [r["value"] for r in isr["steps"]] == pytest.approx(values) and [r["tokens"] for r in isr["steps"]] == tokens
    assert isr["mean_of_steps"] == pytest.approx(sum(values) / 15)
    assert isr["token_weighted_mean"] == pytest.approx(sum(v * t for v, t in zip(values, tokens)) / sum(tokens))
    assert isr["by_launch"]["a"]["mean_of_steps"] == pytest.approx(0.0055)
    langs = rep["in_language"]["launches"]
    assert langs["a"]["shows_grammar"] is True and langs["b"]["shows_grammar"] is True
    assert isr["configured"] is True and isr["grammar_shown"] is True and isr["applies"] is True
    assert isr["rule"]["applies"] is True and isr["rule"]["exceeds"] is False
    assert [c for c in isr["rule"]["caveats"] if c != sc.VLLM_CAVEAT] == [] and sc.VLLM_CAVEAT in isr["rule"]["caveats"]
    assert isr["vllm_logprobs_checked"] == {"a": None, "b": None}, "these run.json files have no vllm_logprobs"
    assert "IS readout (trl:sampling/sampling_logp_difference/mean)" in out and "in the regex's language" in out


def test_a_stop_completion_out_of_the_language_fails_the_smoke_at_the_final_stage(tmp_path, capsys):
    run = _d020_run(tmp_path, out_of_language={(3, 17)})
    assert sc.main(["--run", str(run), "--stage", "first", "--config", str(V2), "--rc", "a=0"]) == 0
    assert "judged at the final stage" in capsys.readouterr().out, "it does not block the resume test"
    rc = sc.main(["--run", str(run), "--stage", "final", "--config", str(V2), "--rc", "a=0", "--rc", "b=0"])
    rep = json.loads((run / "summary" / "smoke.json").read_text())
    assert rc == 1 and any("1 of 320 stop completions are not in the regex's language" in f for f in rep["failures"])
    assert rep["in_language"]["launches"]["a"]["first_out_of_language"] == [{"step": 3, "i": 17, "text_sha256": None}]
    assert "GRPO_SMOKE_FAILED" in capsys.readouterr().out


def test_a_smoke_whose_engine_did_not_apply_the_grammar_gives_no_grammar_verdict(tmp_path):
    # Every stop completion out of the language, at TRL's unconstrained level (smoke 13632679 logged 0.0044-0.0091):
    # the value is under 0.05, but by D-020's Validity a constrained run that fails the in-language check decides
    # nothing, so the rule must not say "keeps the grammar on".
    run = _d020_run(tmp_path, out_of_language={(k, i) for k in range(10) for i in range(32)}, value=lambda s: 0.007)
    rep = sc.check(run, "final", _cfg(), {"a": "0", "b": "0"})
    isr, rule = rep["is_ratio"], rep["is_ratio"]["rule"]
    assert not rep["ok"] and rule["exceeds"] is False and rule["value"] == pytest.approx(0.007)
    assert isr["configured"] is True and isr["grammar_shown"] is False
    assert rule["applies"] is False and isr["applies"] is False
    assert "decides nothing" in rule["says"] and "keeps the grammar on" not in rule["says"]
    assert rep["in_language"]["launches"]["a"]["shows_grammar"] is False
    assert any("0.05 rule decides nothing" in w for w in rep["warnings"])


def test_one_launch_that_names_another_regex_is_enough_to_withhold_the_verdict(tmp_path):
    run = _d020_run(tmp_path)
    meta = run / "summary" / "b" / "run.json"
    m = json.loads(meta.read_text())
    meta.write_text(json.dumps({**m, "structured_outputs": {**m["structured_outputs"], "sha256": "0" * 64}}))
    rep = sc.check(run, "final", _cfg(), {"a": "0", "b": "0"})
    langs = rep["in_language"]["launches"]
    assert (langs["a"]["shows_grammar"], langs["b"]["shows_grammar"]) == (True, False)
    assert any(f.startswith("launch b: run.json's regex sha256 0000000000000000") for f in rep["failures"])
    assert rep["is_ratio"]["grammar_shown"] is False and rep["is_ratio"]["rule"]["applies"] is False
    meta.unlink()                                   # no run.json: nothing names the regex the engine was given
    rep = sc.check(run, "final", _cfg(), {"a": "0", "b": "0"})
    assert any("summary/b/run.json: cannot read" in f for f in rep["failures"])
    assert rep["in_language"]["launches"]["b"]["shows_grammar"] is False and rep["is_ratio"]["grammar_shown"] is False


def test_a_length_completion_out_of_the_language_is_counted_apart_and_passes(tmp_path):
    run = _d020_run(tmp_path, truncated=lambda k, i: i == 5, out_of_language={(k, 5) for k in range(10)})
    rep = sc.check(run, "final", _cfg(), {"a": "0", "b": "0"})
    assert rep["ok"], rep["failures"]
    assert rep["in_language"]["length"] == {"n": 15, "in_language": 0, "out_of_language": 15}
    assert rep["is_ratio"]["rule"]["applies"] is True, "a cut completion out of the language still shows the grammar"
    assert any("truncated completions" in c for c in rep["is_ratio"]["rule"]["caveats"])
    assert rep["is_ratio"]["steps"][0]["tokens"] == 31 * 100, "TRL's mask leaves the cut completion out"


def test_a_constrained_run_whose_records_lack_the_language_or_name_another_regex_fails(tmp_path):
    run = _d020_run(tmp_path, sha="0" * 64)
    rep = sc.check(run, "final", _cfg(), {"a": "0", "b": "0"})
    assert any("is not the config's" in f for f in rep["failures"])
    plain = tmp_path / "plain"
    plain.mkdir()
    run2 = _d020_run(plain)
    p = run2 / "summary" / "a" / "rewards.jsonl"
    p.write_text("".join(json.dumps({k: v for k, v in json.loads(l).items() if k != "in_language"}) + "\n"
                         for l in p.read_text().splitlines()))
    rep2 = sc.check(run2, "final", _cfg(), {"a": "0", "b": "0"})
    assert any("320 of 320 completions have no in-language record" in f for f in rep2["failures"])


def _rewrite(run: Path, **fields):
    """Set `fields` on every completion record of both launches' rewards.jsonl."""
    for name in ("a", "b"):
        p = run / "summary" / name / "rewards.jsonl"
        p.write_text("".join(json.dumps({**json.loads(l), **fields}) + "\n" for l in p.read_text().splitlines()))


def test_completions_without_a_finish_reason_from_their_ids_fail_and_the_rule_says_it_reads_the_step_mean(tmp_path):
    # completion_ids or the limit unknown: finish "other", no token count, no truncation flag. The in-language check
    # judged nothing (D-020, Validity: other.n is 0), and the IS rule has no tokens to weight the steps by.
    run = _d020_run(tmp_path)
    _rewrite(run, finish="other", in_language=False, n_tokens=None, truncated=None)
    rep = sc.check(run, "final", _cfg(), {"a": "0", "b": "0"})
    assert not rep["ok"]
    for name, n in (("a", 320), ("b", 160)):
        assert any(f.startswith(f"launch {name}: {n} of {n} completions have no finish reason from the token ids")
                   for f in rep["failures"]), rep["failures"]
        assert f"launch {name}: no completion ended by stop, so nothing shows the grammar applied" in rep["failures"]
    assert rep["in_language"]["other"]["n"] == 480 and rep["in_language"]["stop"]["n"] == 0
    rule = rep["is_ratio"]["rule"]
    assert rule["reads"] == "mean_of_steps" and rule["value"] == pytest.approx(0.01)
    assert any(c.startswith("steps [('a', 1), ('a', 2)") and "mean of the per-step values" in c
               for c in rule["caveats"]), rule["caveats"]
    assert any("reads the mean of the per-step values" in w for w in rep["warnings"])


def test_a_launch_whose_completions_all_hit_the_limit_shows_no_grammar_and_fails(tmp_path):
    rep = sc.check(_d020_run(tmp_path, truncated=lambda k, i: True), "final", _cfg(), {"a": "0", "b": "0"})
    assert not rep["ok"] and rep["in_language"]["length"]["n"] == 480
    assert {f for f in rep["failures"] if "no completion ended by stop" in f} == {
        f"launch {n}: no completion ended by stop, so nothing shows the grammar applied" for n in ("a", "b")}
    rule = rep["is_ratio"]["rule"]
    assert rule["reads"] == "mean_of_steps", "TRL's mask leaves every token out, so nothing weights the steps"
    assert any(c.startswith("the steps with a value have no completion tokens") for c in rule["caveats"])


def test_a_copy_penalty_config_whose_reward_logged_none_fails(tmp_path):
    rep = sc.check(_d020_run(tmp_path, copy=False), "final", _cfg(), {"a": "0", "b": "0"})
    assert any("have no p_copy" in f for f in rep["failures"])


def test_a_launch_whose_copy_penalty_lacks_the_configs_pool_fails(tmp_path):
    # D-020 scores P_copy against the pool too; run.json's copy_penalty records how many pool files the trainer's
    # CopyPenalty holds, and the config says how many it should
    import yaml
    doc = yaml.safe_load(V2.read_text())
    doc["copy_penalty"] = {"pool": ["drafts/v1"]}
    p = tmp_path / "pool.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    cfg = _cfg(p)
    run = _d020_run(tmp_path / "run", cfg=cfg)
    rep = sc.check(run, "final", cfg, {"a": "0", "b": "0"})
    assert rep["ok"], rep["failures"]
    assert rep["copy"]["a"]["pool_files"] == rep["copy"]["b"]["pool_files"] == len(cfg.pool_files()) == 30
    meta = run / "summary" / "b" / "run.json"
    m = json.loads(meta.read_text())
    meta.write_text(json.dumps({**m, "copy_penalty": {**m["copy_penalty"], "pool": {"files": 0, "ids": []}}}))
    rep = sc.check(run, "final", cfg, {"a": "0", "b": "0"})
    assert any(f.startswith("launch b: run.json's copy penalty holds 0 pool files, not the config's 30")
               for f in rep["failures"]), rep["failures"]
    m.pop("copy_penalty")
    meta.write_text(json.dumps(m))
    rep = sc.check(run, "final", cfg, {"a": "0", "b": "0"})
    assert any(f.startswith("launch b: run.json's copy penalty holds None pool files") for f in rep["failures"])


@pytest.mark.parametrize("source", [None, {**MATCH, "verdict": "not_matched", "missing": ["no abs"]},
                                    {**MATCH, "verdict": "not_found"}])
def test_without_a_checked_source_the_is_readout_is_unavailable_and_claims_nothing(tmp_path, source):
    rep = sc.check(_d020_run(tmp_path, source=source), "final", _cfg(), {"a": "0", "b": "0"})
    isr = rep["is_ratio"]
    assert rep["ok"], "a readout, not a gate"
    assert isr["value_source"] == "unavailable" and isr["rule"] is None and isr["token_weighted_mean"] is None
    assert any(w.startswith("IS readout unavailable") for w in rep["warnings"])
    assert all(r["value"] == pytest.approx(0.01) for r in isr["steps"]), "TRL's numbers stay visible, unread"


def test_one_launch_without_a_checked_source_makes_the_readout_unavailable(tmp_path):
    run = _d020_run(tmp_path)
    meta = run / "summary" / "b" / "run.json"
    m = json.loads(meta.read_text())
    meta.write_text(json.dumps({**m, "trl_logp_metric": {**MATCH, "verdict": "error"}}))
    isr = sc.check(run, "final", _cfg(), {"a": "0", "b": "0"})["is_ratio"]
    assert isr["value_source"] == "unavailable" and isr["verdicts"] == {"a": "match", "b": "error"}


def _set_vllm_logprobs(run: Path, **checked):
    for name, value in checked.items():
        meta = run / "summary" / name / "run.json"
        m = json.loads(meta.read_text())
        meta.write_text(json.dumps({**m, "vllm_logprobs": {"package": "vllm", "version": "0.32.0", "checked": value,
                                                            "files": {}}}))


def test_the_rule_says_vllms_log_probs_were_not_checked_to_be_the_masked_samplers(tmp_path):
    # trl_logp_metric checks TRL's half (|old - sampling| over completion_mask); which log-probs vLLM reports under
    # the grammar is recorded by run.json's vllm_logprobs, whose checked is false in every run today
    run = _d020_run(tmp_path)
    _set_vllm_logprobs(run, a=False, b=False)
    isr = sc.check(run, "final", _cfg(), {"a": "0", "b": "0"})["is_ratio"]
    assert isr["vllm_logprobs_checked"] == {"a": False, "b": False}
    assert isr["rule"]["caveats"] == [sc.VLLM_CAVEAT] and isr["rule"]["applies"] is True
    assert sc.VLLM_CAVEAT.startswith("vLLM's reported log-probs were not checked to be the masked sampler's pi_vLLM")
    _set_vllm_logprobs(run, b=True)
    assert sc.VLLM_CAVEAT in sc.check(run, "final", _cfg(), {"a": "0", "b": "0"})["is_ratio"]["rule"]["caveats"], \
        "one launch unchecked keeps the caveat"
    _set_vllm_logprobs(run, a=True)
    isr = sc.check(run, "final", _cfg(), {"a": "0", "b": "0"})["is_ratio"]
    assert isr["vllm_logprobs_checked"] == {"a": True, "b": True} and isr["rule"]["caveats"] == []


def test_a_mean_over_0_05_turns_the_grammar_off_by_d020s_rule(tmp_path):
    isr = sc.check(_d020_run(tmp_path, value=lambda s: 0.08), "final", _cfg(), {"a": "0", "b": "0"})["is_ratio"]
    assert isr["rule"]["exceeds"] is True and "the grammar goes off" in isr["rule"]["says"]
    assert isr["rule"]["value"] == pytest.approx(0.08)


def test_the_two_means_on_two_sides_of_0_05_are_flagged(tmp_path):
    # steps 1-5 short and high, 6-10 long and low: the plain mean is over 0.05, the token-weighted mean under it
    run = _d020_run(tmp_path, value=lambda s: 0.12 if s <= 5 else 0.02, tokens=lambda k, i: 50 if k < 5 else 500)
    isr = sc.check(run, "final", _cfg(), {"a": "0", "b": "0"})["is_ratio"]
    assert isr["mean_of_steps"] > 0.05 > isr["token_weighted_mean"] and isr["rule"]["reads"] == "token_weighted_mean"
    assert isr["rule"]["exceeds"] is False and any("two sides of 0.05" in c for c in isr["rule"]["caveats"])


def test_steps_without_the_metric_are_warned_and_the_mean_reads_the_others(tmp_path):
    rep = sc.check(_d020_run(tmp_path, value=lambda s: None if s == 3 else 0.01), "final", _cfg(),
                   {"a": "0", "b": "0"})
    isr = rep["is_ratio"]
    assert isr["mean_of_steps"] == pytest.approx(0.01) and isr["token_weighted_mean"] == pytest.approx(0.01)
    assert any("1 of 15 steps have no" in w for w in rep["warnings"])


NOT_CONFIGURED = "not D-020's reading: it is read in a GRPO smoke with the grammar and prompt v2"


def test_on_todays_smoke_the_is_rule_is_not_d020s_reading(tmp_path):
    smoke = _cfg(ROOT / "configs" / "grpo_smoke.yaml")
    rep = sc.check(_d020_run(tmp_path, cfg=smoke, copy=False), "final", smoke, {"a": "0", "b": "0"})
    assert rep["ok"], rep["failures"]
    assert "in_language" not in rep and "backend" not in rep and "copy" not in rep
    assert rep["is_ratio"]["rule"]["applies"] is False and "not D-020's reading" in rep["is_ratio"]["rule"]["says"]


def test_the_grammar_on_prompt_v1_is_not_d020s_reading(tmp_path):
    # configs/grpo_smoke_constrained.yaml, which the job takes: the grammar is shown, but the prompt is v1
    cfg = _cfg(ROOT / "configs" / "grpo_smoke_constrained.yaml")
    rep = sc.check(_d020_run(tmp_path, cfg=cfg, copy=False), "final", cfg, {"a": "0", "b": "0"})
    assert rep["ok"], rep["failures"]
    isr = rep["is_ratio"]
    assert cfg.prompt == "v1" and isr["grammar_shown"] is True
    assert isr["configured"] is False and isr["rule"]["applies"] is False and isr["rule"]["says"] == NOT_CONFIGURED


def test_prompt_v2_without_the_grammar_is_not_d020s_reading(tmp_path):
    import yaml
    doc = yaml.safe_load(V2.read_text())
    del doc["structured_outputs"]                       # D-020's fallback: prompt v2, the grammar off
    p = tmp_path / "v2_no_grammar.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    cfg = _cfg(p)
    rep = sc.check(_d020_run(tmp_path / "run", cfg=cfg), "final", cfg, {"a": "0", "b": "0"})
    assert rep["ok"], rep["failures"]
    isr = rep["is_ratio"]
    assert cfg.prompt == "v2" and cfg.structured_outputs is None and "in_language" not in rep
    assert isr["configured"] is False and isr["rule"]["applies"] is False and isr["rule"]["says"] == NOT_CONFIGURED


# ------------------------------------------------------------------------------------------ D-021 Part A's gates
MASKED = ROOT / "configs" / "grpo_smoke_v2_grammar_masked.yaml"
GATES = ("T1", "T2", "T3", "T4", "T5")
CHECKS = {"verdict": "match", "path": "support_ids", "gating": list(GATES), "failed": [],
          **{t: {"verdict": "match"} for t in GATES + ("V1", "V2")},
          "T3": {"verdict": "match", "ratio_mean_over": "sequences", "mode": "sequence_mask", "clip_max": 3.0}}
STAMP = "commit=0123456789ab dirty=0 diff=- at=2026-11-03T01:02:03Z"
STOP_IDS = [151643, 151645]
WEIGHTS = [1.1 if i % 2 else 0.9 for i in range(32)]       # mean 1.0, the largest 1.1 of a sum of 32
ORDER = ["M5", "M6", "M7", "M8", "M9", "M10", "M10_ratio", "M11", "M12"]


FROZEN = {"src/lofi_rl/rl/masked.py": b"TRAINER_LOGPROBS = ('grammar_masked', 'grammar_masked_fp32')\n",
          "configs/grpo_smoke_v2_grammar_masked.yaml": b"trainer_logprobs: grammar_masked\n"}
D021_LINE = "## D-021 (2026-11-02): run 1's start, registered again"
DECISIONS_TEXT = f"# Decisions\n\n## D-020 (2026-10-31): the base-model test v1\n\n{D021_LINE}\n\n- the entry\n"


def _blob(data: bytes) -> str:
    """git's blob id of `data` (git hash-object's formula), written out here rather than taken from smoke_check."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _frozen_tree(root: Path, files=FROZEN, decisions=DECISIONS_TEXT) -> Path:
    """A tree laid out like the job's snapshot for M5's registration half: `files`, D-021's frozen path list naming
    them by git blob (with comment lines, as the D-020 lists have them), and docs/DECISIONS.md."""
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    lst = root / sc.PREREG_LIST
    lst.parent.mkdir(parents=True, exist_ok=True)
    lst.write_text("# D-021 masked smoke path (a test's own), git blob sha and path\n\n"
                   + "".join(f"{_blob(d)}  {rel}\n" for rel, d in files.items()))
    (root / sc.DECISIONS).write_text(decisions)
    return root


@pytest.fixture
def frozen_root(tmp_path_factory, monkeypatch):
    """M5's registration half reads smoke_check.ROOT (config.ROOT; on the cluster the job's snapshot). The working tree
    holds no D-021 list until the lead writes it, after the code is final, so the masked tests point ROOT at a tree of
    their own, which passes that half."""
    root = _frozen_tree(tmp_path_factory.mktemp("frozen"))
    monkeypatch.setattr(sc, "ROOT", root)
    return root


@pytest.fixture
def masked_cfg(frozen_root):
    """The masked smoke's config (its trainer_logprobs values come from lofi_rl.rl.masked), judged against a frozen
    tree that passes M5's registration half (frozen_root)."""
    return _cfg(MASKED)


CLOSE = {"positions": 200, "mean_abs_diff": 1e-6, "max_abs_diff": 4e-6, "nonfinite": 0}   # masked.diff_stats' keys


def _selfcheck(global_step: int, **kw) -> dict:
    """The record the masked trainer keeps for a launch's first self-check (masked.MaskedPasses._selfcheck): path fill
    on 2 rows beside support_ids, with |diff| over their valid positions, masked and unmasked; `kw` overrides."""
    return {"step": global_step, "rows": [0, 1], "ran": "support_ids", "against": "fill", "masked": dict(CLOSE),
            "unmasked": dict(CLOSE), "error": None, "seconds": 0.4, **kw}


def _mask_record(k: int, neglogz: float, weights, path: str = "support_ids") -> dict:
    """The record the masked trainer keeps for the rollouts it generated at global step k: section D's own empty
    record (masked.new_step_record), filled as a clean step would fill it. 32 completions of 100 tokens each, all
    stopped: the replay checked 3,200 tokens and the old pass had 3,200 valid positions, as the reward counts them."""
    from lofi_rl.rl.masked import new_step_record
    rec = new_step_record(k, path, nfa=True)
    rec.update(generations=1, completions=32, terminated=32, positions=3200, walked=3200, nfa_positions=3200,
               trl_sampling_mask=None, rows={"old": 32, "ref": 32, "loss": 32},
               rows_scored={"old": 32, "ref": 32, "loss": 32}, tokens=3200, forced=237, forced_share=237 / 3200,
               k_max=52, neglogz=neglogz, neglogz_tokens=3200, seq_weights=weights, seq_weights_raw=weights,
               seq_valid=[True] * 32, is_level="sequence")
    return rec


def _masked_run(tmp_path: Path, cfg, *, value=lambda s: 0.01, neglogz=lambda s: 0.15, weights=lambda s: WEIGHTS,
                mask=lambda n, s: {}, meta=lambda n: {}, trl=lambda n, s: {}, step_s=lambda n, s: 30.0,
                truncated=lambda k, i: False, selfcheck=lambda n, s: _selfcheck(s - 1)) -> Path:
    """A masked smoke's two launches: _d020's files, plus what train.py and the masked trainer add (D-021 Part A):
    run.json's stamp, code, trainer_logprobs, masked, trl_source_checks and lm_head (then `meta(launch)`'s changes),
    and per step TRL's loss and entropy and the mask record of the rollouts it scored (then `mask` and `trl`), with
    `selfcheck(launch, step)` (None: none) at each launch's first step."""
    run = _d020_run(tmp_path, cfg=cfg, value=value, truncated=truncated)
    sha = cfg.structured_outputs["sha256"]
    for name in ("a", "b"):
        out = run / "summary" / name
        m = json.loads((out / "run.json").read_text())
        m.update(stamp=STAMP, trainer_logprobs="grammar_masked",
                 code={"lofi_code": "/snap/x", "lofi_code_snapshot": True, "root": "/snap/x",
                       "root_is_lofi_code": True, "root_snapshot": True},
                 masked={"mode": "grammar_masked", "path": "support_ids", "regex_sha256": sha, "vocab_size": 151936,
                         "stop_ids": STOP_IDS, "replay": {"vocab_size": 151936, "stop_ids": STOP_IDS},
                         "nfa": {"regex_sha256": sha, "vocab_size": 151936, "stop_ids": STOP_IDS}},
                 trl_source_checks=copy.deepcopy(CHECKS), lm_head={"verdict": "match"})
        m.update(meta(name))
        (out / "run.json").write_text(json.dumps(m))
        lines = []
        for line in (out / "steps.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") == "step":
                s = r["step"]
                r["step_s"] = step_s(name, s)
                r["trl"].update({"loss": 0.01, "entropy": 0.05, **trl(name, s)})
                r["mask"] = {**_mask_record(s - 1, neglogz(s), weights(s)), **mask(name, s)}
                if s == (1 if name == "a" else 6) and selfcheck(name, s) is not None:
                    r["mask"]["selfcheck"] = selfcheck(name, s)
            lines.append(r)
        (out / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    return run


def _d021(run: Path, cfg) -> dict:
    return sc.check(run, "final", cfg, {"a": "0", "b": "0"})


def test_a_masked_smoke_that_meets_m5_to_m11_passes_and_says_so_just_before_the_last_line(tmp_path, capsys,
                                                                                           masked_cfg):
    run = _masked_run(tmp_path, masked_cfg)
    rc = sc.main(["--run", str(run), "--stage", "final", "--config", str(MASKED), "--rc", "a=0", "--rc", "b=0"])
    rep = json.loads((run / "summary" / "smoke.json").read_text())
    lines = capsys.readouterr().out.strip().splitlines()
    assert rc == 0 and rep["ok"], rep["failures"]
    assert lines[-1] == "GRPO_SMOKE_OK" and lines[-2] == "D021_PART_A pass"
    d = rep["d021"]
    assert (d["verdict"], d["not_passing"], d["undecided"], d["mode"], d["path"]) == (
        "pass", [], [], "grammar_masked", "support_ids")
    g = d["gates"]
    assert list(g) == ORDER and all(g[k]["pass"] is True for k in ORDER), {k: g[k]["reasons"] for k in ORDER}
    assert {k: g[k]["kind"] for k in ORDER} == {
        **dict.fromkeys(("M5", "M6", "M7", "M8", "M10"), "validity"),
        **dict.fromkeys(("M9", "M10_ratio", "M11"), "measured"), "M12": "estimate-only"}
    assert g["M5"]["value"] == {n: {"commit": "0123456789ab", "dirty": "0"} for n in ("a", "b")}
    assert g["M7"]["value"]["a"]["gating"] == dict.fromkeys(GATES, "match")
    assert g["M7"]["value"]["b"]["T5_run_time"] == "match"
    m8 = g["M8"]["value"]
    assert (m8["violations"], m8["nfa_mismatch"], m8["nfa_positions"], m8["steps_with_record"], m8["steps"]) == (
        0, 0, 15 * 3200, 15, 15)
    assert (m8["completions"], m8["positions"], m8["trl_sampling_mask"]) == (15 * 32, 15 * 3200, 0)
    assert m8["path"] == {"a": "support_ids", "b": "support_ids"} and m8["rows_per_pass"] == 32
    assert g["M9"]["value"] == pytest.approx(0.01) and g["M10"]["value"] == pytest.approx(0.15)
    assert g["M10"]["tokens"] == 15 * 3200 and g["M10_ratio"]["value"] == pytest.approx(0.01 / 0.15)
    m11 = g["M11"]["value"]
    assert m11["passing"] == m11["steps"] == 15 and m11["min_mean"] == pytest.approx(1.0)
    assert m11["max_share"] == pytest.approx(1.1 / 32) and m11["non_truncated"]["max_share"] == pytest.approx(1.1 / 32)
    assert g["M12"]["value"] == pytest.approx(30.0)
    diag = d["diagnostics"]
    assert diag["token_counts"]["steps_compared"] == diag["token_counts"]["agree"] == 15
    assert diag["replay_positions"]["steps_compared"] == diag["replay_positions"]["agree"] == 15
    assert diag["nonfinite_masked_logps"] == 0 and diag["forced_share_mean_of_steps"] == pytest.approx(237 / 3200)
    assert set(diag["selfcheck"]) == {"a", "b"} and diag["selfcheck"]["b"]["step"] == 6
    assert diag["selfcheck"]["b"]["global_step"] == 5, "the record's own step is the global step it ran at"
    assert d["selfcheck_flags"] == [] and not any("self-check" in w for w in rep["warnings"])
    assert g["M10"]["zero_weight"] == []
    assert g["M5"]["registration"]["matched"] == g["M5"]["registration"]["files"] == len(FROZEN)
    assert "supersedes D-020's IS-half verdict" in d["follows"]
    assert "D-021 Part A reads it as M9" in rep["is_ratio"]["rule"]["says"], "D-021, not D-020's rule, decides"
    assert any(l.startswith("[smoke] D-021 M11 (measured): pass: 15 of 15 steps pass") for l in lines)


def test_the_first_stage_and_an_unmasked_smoke_judge_no_d021_gate(tmp_path, capsys, masked_cfg):
    run = _masked_run(tmp_path, masked_cfg)
    assert sc.main(["--run", str(run), "--stage", "first", "--config", str(MASKED), "--rc", "a=0"]) == 0
    out = capsys.readouterr().out
    assert out.strip().endswith("FIRST_RUN_OK") and "D021_PART_A" not in out
    assert json.loads((run / "summary" / "smoke.json").read_text())["d021"] is None
    plain = tmp_path / "plain"
    plain.mkdir()
    assert "d021" not in sc.check(_d020_run(plain), "final", _cfg(), {"a": "0", "b": "0"})


def _drop(path: Path, keep) -> None:
    lines = [l for l in path.read_text().splitlines() if keep(json.loads(l))]
    path.write_text("".join(l + "\n" for l in lines))


def test_a_launch_stopped_by_a_mask_violation_is_invalid_and_fails_the_smoke(tmp_path, capsys, masked_cfg):
    # the masked trainer raises MaskViolation before any pass sees the rows: launch b stopped in step 8, unlogged
    err = "MaskViolation: step 7: completion 3 position 12: token 9 is not in its allowed set"
    run = _masked_run(tmp_path, masked_cfg, meta=lambda n: {"status": "failed", "error": err} if n == "b" else {})
    _drop(run / "summary" / "b" / "steps.jsonl", lambda r: r["kind"] != "step" or r["step"] < 8)
    _drop(run / "summary" / "b" / "rewards.jsonl", lambda r: r["step"] < 7)
    rc = sc.main(["--run", str(run), "--stage", "final", "--config", str(MASKED), "--rc", "a=0", "--rc", "b=1"])
    rep = json.loads((run / "summary" / "smoke.json").read_text())
    lines = capsys.readouterr().out.strip().splitlines()
    assert rc == 1 and lines[-1].startswith("GRPO_SMOKE_FAILED") and lines[-2].startswith("D021_PART_A invalid: M6")
    d = rep["d021"]
    assert d["verdict"] == "invalid" and {"M6", "M8"} <= set(d["not_passing"])
    m8 = d["gates"]["M8"]["reasons"]
    assert any(r.startswith("launch b stopped on a sampled token outside the mask: MaskViolation") for r in m8)
    assert {f"launch b step {s}: no mask record" for s in (8, 9, 10)} <= set(m8)
    assert "measured nothing" in d["follows"]


@pytest.mark.parametrize("case, kw, gate, says", [
    ("a sampled token outside the mask", dict(mask=lambda n, s: {"violations": 1, "violation_examples": [
        {"row": 3, "pos": 12, "token": 9}]} if (n, s) == ("b", 8) else {}),
     "M8", "launch b step 8: violations 1, not 0 (first [{'row': 3, 'pos': 12, 'token': 9}])"),
    ("violations as a list", dict(mask=lambda n, s: {"violations": [[3, 12, 9]]} if (n, s) == ("b", 8) else {}),
     "M8", "launch b step 8: violations [[3, 12, 9]], not 0"),
    ("a step the replay never checked", dict(mask=lambda n, s: {
        "completions": 0, "positions": 0, "walked": 0, "trl_sampling_mask": "unread"} if (n, s) == ("a", 5) else {}),
     "M8", "launch a step 5: the replay checked 0 completions (0 tokens), not the step's 32"),
    ("an NFA mismatch", dict(mask=lambda n, s: {"nfa_mismatch": 2, "nfa_examples": [{"row": 1, "pos": 40}]}
                             if (n, s) == ("a", 4) else {}), "M8", "launch a step 4: nfa_mismatch 2, not 0"),
    ("no NFA comparison", dict(mask=lambda n, s: {"nfa_positions": 0}), "M8", "the independent check did not run"),
    ("an NFA over another vocabulary", dict(meta=lambda n: {"masked": {
        "mode": "grammar_masked", "path": "support_ids", "vocab_size": 151936, "stop_ids": STOP_IDS,
        "replay": {"vocab_size": 151936}, "nfa": {"vocab_size": 151669, "stop_ids": STOP_IDS}}} if n == "a" else {}),
     "M8", "the nfa's vocab_size 151669 is not the run's 151936"),
    ("a T check not matching", dict(meta=lambda n: {"trl_source_checks": {
        **CHECKS, "verdict": "not_matched", "T3": {"verdict": "not_matched", "missing": ["no sequence sum"]}}}
        if n == "b" else {}), "M7", "launch b: T3 not_matched"),
    ("lm_head a LoRA target", dict(meta=lambda n: {"lm_head": {"verdict": "not_matched"}} if n == "a" else {}),
     "M7", "launch a: T5 at run time (lm_head) not_matched"),
    ("no source checks", dict(meta=lambda n: {"trl_source_checks": None} if n == "a" else {}), "M7",
     "launch a: run.json has no trl_source_checks"),
    ("a short loss pass", dict(mask=lambda n, s: {"rows": {"old": 32, "ref": 32, "loss": 28}} if (n, s) == ("a", 2)
                               else {}), "M8", "launch a step 2: masked rows {'old': 32, 'ref': 32, 'loss': 28}"),
    ("a NaN loss", dict(trl=lambda n, s: {"loss": float("nan")} if (n, s) == ("b", 9) else {}), "M8",
     "launch b step 9: logged loss not finite"),
    ("no KL logged", dict(trl=lambda n, s: {"kl": None} if (n, s) == ("a", 3) else {}), "M8",
     "launch a step 3: logged kl not finite"),
    ("another path ran", dict(mask=lambda n, s: {"path": "fill"} if (n, s) == ("a", 1) else {}), "M8",
     "launch a step 1: the step ran path 'fill', not 'support_ids'"),
    ("an unmasked launch", dict(meta=lambda n: {"trainer_logprobs": None} if n == "b" else {}), "M8",
     "launch b: run.json trainer_logprobs None, not 'grammar_masked'"),
    ("a dirty stamp", dict(meta=lambda n: {"stamp": STAMP.replace("dirty=0", "dirty=1")} if n == "b" else {}), "M5",
     "launch b: its stamp says dirty=1, not 0"),
    ("two commits", dict(meta=lambda n: {"stamp": STAMP.replace("0123456789ab", "ba9876543210")} if n == "b" else {}),
     "M5", "not one commit"),
    ("no stamp", dict(meta=lambda n: {"stamp": None} if n == "a" else {}), "M5",
     "launch a: run.json has no .sync-stamp"),
    ("the backend not applied", dict(meta=lambda n: {"structured_outputs": {
        "sha256": "1412f11f9ebdc520b50c263eda14be12829951357c17f43496ab2ddf5235e64e",
        "backend": {"requested": "xgrammar", "pin_accepted": False, "backend": "auto", "applied": False}}}
        if n == "b" else {}), "M6", "launch b: backend"),
    ("-log Z under 0.05", dict(neglogz=lambda s: 0.01), "M10", None),
])
def test_an_invalid_masked_smoke_names_the_validity_gate_it_fails(tmp_path, masked_cfg, case, kw, gate, says):
    rep = _d021(_masked_run(tmp_path, masked_cfg, **kw), masked_cfg)
    d = rep["d021"]
    assert d["verdict"] == "invalid" and d["gates"][gate]["pass"] is False, (case, d["not_passing"])
    assert d["not_passing"][0] == gate, (case, d["not_passing"])
    assert says is None or any(says in r for r in d["gates"][gate]["reasons"]), (case, d["gates"][gate]["reasons"])
    assert "measured nothing" in d["follows"]
    if gate != "M6":
        assert rep["ok"], "the D-021 gates never add to the smoke's own failures (M6 reads them)"


def test_m9_over_0_05_is_a_measured_miss_and_names_d021s_one_fix(tmp_path, masked_cfg):
    d = _d021(_masked_run(tmp_path, masked_cfg, value=lambda s: 0.12), masked_cfg)["d021"]
    assert d["verdict"] == "measured_miss" and d["not_passing"] == ["M9", "M10_ratio"]
    g = d["gates"]
    assert g["M9"]["value"] == pytest.approx(0.12) and g["M10_ratio"]["value"] == pytest.approx(0.12 / 0.15)
    assert g["M10"]["pass"] is True, "-log Z is the trainer's own: the mask ran"
    assert "grammar_masked_fp32" in d["follows"] and "D-020's verdict" in d["follows"]


def test_m10s_ratio_alone_catches_log_probs_taken_before_the_mask(tmp_path, masked_cfg):
    # M9 under 0.05 but no smaller than a quarter of -log Z: what vLLM's log-probs before the bitmask would look like
    d = _d021(_masked_run(tmp_path, masked_cfg, value=lambda s: 0.04, neglogz=lambda s: 0.06), masked_cfg)["d021"]
    assert d["verdict"] == "measured_miss" and d["not_passing"] == ["M10_ratio"]
    assert d["gates"]["M10_ratio"]["value"] == pytest.approx(0.04 / 0.06) and d["gates"]["M9"]["pass"] is True


def test_collapsed_is_weights_are_a_measured_miss_with_the_non_truncated_rows_beside_them(tmp_path, masked_cfg):
    # D-021's grammar smoke: one completion carried at least 90 % of a step's weight in 7 of its 15 steps
    collapsed = [0.001] * 31 + [0.5]
    valid = [i != 5 for i in range(32)]
    d = _d021(_masked_run(tmp_path, masked_cfg, weights=lambda s: collapsed if s == 4 else WEIGHTS,
                          mask=lambda n, s: {"seq_valid": valid}), masked_cfg)["d021"]
    assert d["verdict"] == "measured_miss" and d["not_passing"] == ["M11"]
    g = d["gates"]["M11"]
    bad = [r for r in g["steps"] if not r["pass"]]
    assert [(r["run"], r["step"]) for r in bad] == [("a", 4)] and g["value"]["passing"] == 14
    assert bad[0]["mean"] == pytest.approx(0.531 / 32) and bad[0]["max_share"] == pytest.approx(0.5 / 0.531)
    assert bad[0]["non_truncated"]["n"] == 31 and bad[0]["non_truncated"]["max_share"] == pytest.approx(0.5 / 0.530)
    assert g["value"]["max_share"] == pytest.approx(0.5 / 0.531)
    assert any(r.startswith("launch a step 4: mean 0.01659") for r in g["reasons"])


def test_weights_trl_set_to_zero_are_counted_and_an_all_zero_step_misses(tmp_path, masked_cfg):
    raw = [4.0] * 32                                   # every weight over clip_max 3.0: sequence_mask zeroes them
    d = _d021(_masked_run(tmp_path, masked_cfg, weights=lambda s: [0.0] * 32 if s == 2 else WEIGHTS,
                          mask=lambda n, s: {"seq_weights_raw": raw} if s == 2 else {}), masked_cfg)["d021"]
    row = next(r for r in d["gates"]["M11"]["steps"] if r["step"] == 2)
    assert row["zeroed"] == 32 and row["mean"] == 0 and row["max_share"] is None and row["pass"] is False
    assert d["verdict"] == "measured_miss" and d["gates"]["M11"]["value"]["zero_sum_steps"] == 1


def test_a_measured_gate_that_cannot_be_read_makes_the_launch_invalid(tmp_path, masked_cfg):
    # launch b's TRL metric source was not checked to match, so the IS readout (M9) and M10's ratio cannot be read:
    # nothing was measured, which D-021 calls an invalid launch, never a measured miss
    meta = lambda n: {"trl_logp_metric": {**MATCH, "verdict": "not_matched", "missing": ["x"]}} if n == "b" else {}
    d = _d021(_masked_run(tmp_path, masked_cfg, meta=meta), masked_cfg)["d021"]
    assert d["gates"]["M9"]["pass"] is None and d["gates"]["M10_ratio"]["pass"] is None
    assert d["verdict"] == "invalid" and d["undecided"] == d["not_passing"] == ["M9", "M10_ratio"]
    assert any("the IS readout is unavailable" in r for r in d["gates"]["M9"]["reasons"])
    # a step without its metric: M9 reads every step, so it is undecided too
    d2 = _d021(_masked_run(tmp_path / "2", masked_cfg, value=lambda s: None if s == 7 else 0.01), masked_cfg)["d021"]
    assert d2["gates"]["M9"]["pass"] is None and d2["verdict"] == "invalid"


def test_trls_ratio_metric_stands_in_for_seq_weights_only_when_t3_reads_it_over_sequences(tmp_path, masked_cfg):
    stand_in = dict(weights=lambda s: None, trl=lambda n, s: {sc.IS_WEIGHT_MEAN: 0.8, sc.IS_WEIGHT_MAX: 1.6})
    d = _d021(_masked_run(tmp_path, masked_cfg, **stand_in), masked_cfg)["d021"]
    g = d["gates"]["M11"]
    assert g["pass"] is True and d["verdict"] == "pass"
    assert all(r["source"].startswith("trl:sampling/importance_sampling_ratio/mean") for r in g["steps"])
    assert g["value"]["max_share"] == pytest.approx(1.6 / (32 * 0.8)), "D-021's table: max / (32 x mean)"
    over_tokens = {**CHECKS, "T3": {"verdict": "match", "ratio_mean_over": "tokens"}}
    d2 = _d021(_masked_run(tmp_path / "2", masked_cfg, meta=lambda n: {"trl_source_checks": over_tokens}, **stand_in),
               masked_cfg)["d021"]
    assert d2["gates"]["M11"]["pass"] is None and d2["verdict"] == "invalid" and d2["undecided"] == ["M11"]
    assert any("TRL's metric cannot stand in" in r for r in d2["gates"]["M11"]["reasons"])


def test_m12_over_its_bar_decides_only_the_run_1_estimate(tmp_path, masked_cfg, capsys):
    run = _masked_run(tmp_path, masked_cfg, step_s=lambda n, s: 50.0)
    assert sc.main(["--run", str(run), "--stage", "final", "--config", str(MASKED), "--rc", "a=0", "--rc", "b=0"]) == 0
    d = json.loads((run / "summary" / "smoke.json").read_text())["d021"]
    assert d["gates"]["M12"]["pass"] is False and d["gates"]["M12"]["value"] == pytest.approx(50.0)
    assert d["verdict"] == "pass" and d["not_passing"] == [] and d["estimate"] == {"M12": False}
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-2:] == ["D021_PART_A pass", "GRPO_SMOKE_OK"]
    assert "[smoke] D-021 M12 (estimate-only): miss: mean step 50 s over 15 steps; bar <= 43.3 s" in lines


def test_the_named_fixs_rerun_says_what_a_second_miss_means(tmp_path, masked_cfg):
    doc = yaml.safe_load(MASKED.read_text())
    doc["trainer_logprobs"] = "grammar_masked_fp32"
    p = tmp_path / "fp32.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    cfg = _cfg(p)
    fill = lambda n: {"trainer_logprobs": "grammar_masked_fp32", "masked": {
        "mode": "grammar_masked_fp32", "path": "fill", "vocab_size": 151936, "stop_ids": STOP_IDS,
        "replay": {}, "nfa": {"vocab_size": 151936, "stop_ids": STOP_IDS}}}
    d = _d021(_masked_run(tmp_path / "run", cfg, value=lambda s: 0.12, meta=fill,
                          mask=lambda n, s: {"path": "fill"}), cfg)["d021"]
    assert (d["mode"], d["path"], d["verdict"]) == ("grammar_masked_fp32", "fill", "measured_miss")
    assert d["gates"]["M8"]["pass"] is True and sc.FP32_RERUN in d["follows"]


# ------------------------------------------------------------------------------------------ M5's registration half
def test_git_blob_sha_is_git_hash_objects_id():
    # the ids `git hash-object --stdin` printed for these bytes (run 2026-11-02 on the laptop)
    assert sc.git_blob_sha(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert sc.git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert all(sc.git_blob_sha(d) == _blob(d) for d in FROZEN.values())


def test_m5_reads_the_frozen_path_and_d021_from_the_tree_that_ran(tmp_path, masked_cfg, frozen_root, capsys):
    run = _masked_run(tmp_path, masked_cfg)
    assert sc.main(["--run", str(run), "--stage", "final", "--config", str(MASKED), "--rc", "a=0", "--rc", "b=0"]) == 0
    m5 = json.loads((run / "summary" / "smoke.json").read_text())["d021"]["gates"]["M5"]
    reg = m5["registration"]
    assert m5["pass"] is True and m5["reasons"] == []
    assert (reg["root"], reg["snapshot"], reg["list"]) == (str(frozen_root), False, sc.PREREG_LIST)
    assert (reg["files"], reg["matched"], reg["n_changed"], reg["n_unreadable"], reg["n_malformed"]) == (2, 2, 0, 0, 0)
    assert reg["list_sha256"] == hashlib.sha256((frozen_root / sc.PREREG_LIST).read_bytes()).hexdigest()
    assert reg["d021_heading"] == D021_LINE
    assert m5["value"] == {n: {"commit": "0123456789ab", "dirty": "0"} for n in ("a", "b")}, "the stamps, as before"
    out = capsys.readouterr().out
    assert "[smoke] D-021 M5 (validity): pass: commit 0123456789ab, dirty 0/0; frozen path: 2 of 2 files at their " \
           f"blob (list sha256 {reg['list_sha256'][:12]}), D-021 in docs/DECISIONS.md" in out
    from lofi_rl.rl.train import SNAPSHOT_MARKER
    assert sc.SNAPSHOT_MARKER == SNAPSHOT_MARKER
    (frozen_root / SNAPSHOT_MARKER).write_text("")        # what scripts/snapshot.py leaves in a job's snapshot
    assert sc._m5({}, root=frozen_root)["registration"]["snapshot"] is True


def _append(path: Path, text: str) -> None:
    path.write_text(path.read_text() + text)


@pytest.mark.parametrize("case, breaks, says", [
    ("a changed file", lambda r: (r / "src/lofi_rl/rl/masked.py").write_bytes(b"TRAINER_LOGPROBS = ()\n"),
     "1 frozen files are not their frozen blob in {root} (first: src/lofi_rl/rl/masked.py: frozen "
     + _blob(FROZEN["src/lofi_rl/rl/masked.py"])[:12]),
    ("a missing file", lambda r: (r / "configs/grpo_smoke_v2_grammar_masked.yaml").unlink(),
     "1 frozen files cannot be read in {root} (first: configs/grpo_smoke_v2_grammar_masked.yaml"),
    ("a malformed line", lambda r: _append(r / sc.PREREG_LIST, "not-a-sha  configs/x.yaml\n"),
     "1 lines of docs/prereg/D-021_masked_smoke_path.txt are not '<git blob sha>  <path>' with a relative path "
     "(first: line 5: 'not-a-sha  configs/x.yaml')"),
    ("a sha without a path", lambda r: _append(r / sc.PREREG_LIST, _blob(b"") + "\n"), "1 lines of"),
    ("a path out of the tree", lambda r: _append(r / sc.PREREG_LIST, f"{_blob(b'')}  ../outside.txt\n"),
     "1 lines of"),
    ("an absolute path", lambda r: _append(r / sc.PREREG_LIST, f"{_blob(b'')}  /etc/hosts\n"), "1 lines of"),
    ("a missing list", lambda r: (r / sc.PREREG_LIST).unlink(),
     "cannot read the frozen path list docs/prereg/D-021_masked_smoke_path.txt"),
    ("a list of comments only", lambda r: (r / sc.PREREG_LIST).write_text("# nothing frozen\n\n"),
     "docs/prereg/D-021_masked_smoke_path.txt names no file"),
    ("a list that is not text", lambda r: (r / sc.PREREG_LIST).write_bytes(b"\xff\xfe\x00"), "is not UTF-8 text"),
    ("DECISIONS.md without D-021", lambda r: (r / sc.DECISIONS).write_text(DECISIONS_TEXT.replace("D-021", "D-022")),
     "docs/DECISIONS.md has no line starting '## D-021'"),
    ("no DECISIONS.md", lambda r: (r / sc.DECISIONS).unlink(), "cannot read docs/DECISIONS.md"),
])
def test_m5_fails_closed_on_the_frozen_path_or_the_entry(tmp_path, masked_cfg, frozen_root, case, breaks, says):
    breaks(frozen_root)
    rep = _d021(_masked_run(tmp_path, masked_cfg), masked_cfg)
    d = rep["d021"]
    m5 = d["gates"]["M5"]
    assert m5["pass"] is False and d["verdict"] == "invalid" and d["not_passing"] == ["M5"], (case, m5["reasons"])
    assert any(says.format(root=frozen_root) in r for r in m5["reasons"]), (case, m5["reasons"])
    assert rep["ok"], "the D-021 gates never add to the smoke's own failures"
    assert "measured nothing" in d["follows"]


def test_m5_counts_every_problem_and_keeps_the_first_examples(tmp_path):
    files = {f"src/f{i}.py": b"x\n" for i in range(12)}
    root = _frozen_tree(tmp_path, files)
    for i in range(12):
        (root / f"src/f{i}.py").write_bytes(b"y\n")
    m5 = sc._m5({}, root=root)
    reg = m5["registration"]
    assert m5["pass"] is False and (reg["files"], reg["matched"], reg["n_changed"]) == (12, 0, 12)
    assert len(reg["changed"]) == sc.MAX_LISTED and reg["changed"][0] == {
        "line": 3, "path": "src/f0.py", "frozen": _blob(b"x\n"), "now": _blob(b"y\n")}
    assert any(r.startswith("launch a: run.json has no .sync-stamp") for r in m5["reasons"]), "the stamps still count"


def test_m5_never_raises(tmp_path, masked_cfg, monkeypatch):
    def boom(root):
        raise RuntimeError("the tree cannot be walked")
    monkeypatch.setattr(sc, "_registration", boom)
    d = _d021(_masked_run(tmp_path, masked_cfg), masked_cfg)["d021"]
    m5 = d["gates"]["M5"]
    assert m5["pass"] is False and m5["value"] is None and d["verdict"] == "invalid" and d["not_passing"] == ["M5"]
    assert m5["reasons"] == ["M5 could not be read (RuntimeError: the tree cannot be walked)"]
    assert sc._m5({"a": {"meta": {"stamp": 7}}, "b": None}, root=tmp_path / "nowhere")["pass"] is False


# ------------------------------------------------------------------------------------------ M10: zero-weight steps
ALL_TRUNCATED = {"neglogz": None, "neglogz_tokens": 0, "tokens": 0, "neglogz_nonfinite": 0, "forced": 0,
                 "forced_share": None, "positions": 32 * 512, "walked": 32 * 512, "nfa_positions": 32 * 512,
                 "terminated": 0, "rows_scored": {"old": 0, "ref": 0, "loss": 0}, "seq_weights": [1.0] * 32,
                 "seq_weights_raw": [1.0] * 32, "seq_valid": [False] * 32}


def test_a_step_whose_completions_were_all_truncated_weighs_zero_in_m10(tmp_path, masked_cfg):
    # every completion of launch a's step 4 hit the limit: TRL's mask zeroed them all, its IS metric reads 0.0 over an
    # empty mask, the old pass had no valid position, and the trainer kept neglogz None over 0 tokens (masked.py)
    run = _masked_run(tmp_path, masked_cfg, value=lambda s: 0.0 if s == 4 else 0.01, truncated=lambda k, i: k == 3,
                      mask=lambda n, s: ALL_TRUNCATED if (n, s) == ("a", 4) else {})
    rep = _d021(run, masked_cfg)
    d, g = rep["d021"], rep["d021"]["gates"]
    assert rep["ok"], rep["failures"]
    assert d["verdict"] == "pass" and d["not_passing"] == [], {k: g[k]["reasons"] for k in g}
    assert g["M10"]["zero_weight"] == [("a", 4)] and g["M10"]["tokens"] == 14 * 3200
    assert g["M10"]["value"] == pytest.approx(0.15) and "1 zero-weight steps" in g["M10"]["reads"]
    assert g["M9"]["value"] == pytest.approx(0.01), "M9 weighs the same step 0"
    assert next(r for r in g["M11"]["steps"] if (r["run"], r["step"]) == ("a", 4))["pass"] is True


def _neglogz_data(*recs) -> tuple[dict, dict]:
    good = {"neglogz": 0.15, "neglogz_tokens": 3200, "tokens": 3200, "neglogz_nonfinite": 0}
    steps = [{"step": 1, "mask": good}] + [{"step": k + 2, "mask": r} for k, r in enumerate(recs)]
    return {"a": {"steps": steps}}, {"a": [s["step"] for s in steps]}


@pytest.mark.parametrize("rec", [
    {"neglogz": None, "neglogz_tokens": 0, "tokens": 3200, "neglogz_nonfinite": 3200},   # tokens, all non-finite
    {"neglogz": None, "neglogz_tokens": 0, "tokens": 0, "neglogz_nonfinite": 2},
    {"neglogz": None, "neglogz_tokens": False, "tokens": 0, "neglogz_nonfinite": 0},      # ill-typed counts
    {"neglogz": None, "neglogz_tokens": 0, "tokens": 0.0, "neglogz_nonfinite": 0},
    {"neglogz": None, "neglogz_tokens": 0, "tokens": None, "neglogz_nonfinite": 0},
    {"neglogz": None, "neglogz_tokens": 0, "tokens": 0, "neglogz_nonfinite": False},
    {"neglogz": None, "neglogz_tokens": 0},                                               # no token count
    {},
    None,                                                                                 # no mask record
])
def test_m10_keeps_a_step_missing_unless_its_old_pass_had_no_valid_token(rec):
    data, want = _neglogz_data(rec)
    first, ratio = sc._m10(data, want, {"pass": True, "value": 0.01})
    assert first["pass"] is None and first["zero_weight"] == [] and ratio["pass"] is None
    assert first["reasons"] == ["steps [('a', 2)] have no neglogz or neglogz_tokens"]


def test_m10_reads_zero_weight_steps_and_still_needs_one_valid_token():
    data, want = _neglogz_data({"neglogz": None, "neglogz_tokens": 0, "tokens": 0, "neglogz_nonfinite": 0},
                               {"neglogz": None, "neglogz_tokens": 0, "tokens": 0})          # no non-finite count
    first, ratio = sc._m10(data, want, {"pass": True, "value": 0.01})
    assert first["pass"] is True and first["zero_weight"] == [("a", 2), ("a", 3)] and first["tokens"] == 3200
    assert ratio["pass"] is True and ratio["value"] == pytest.approx(0.01 / 0.15)
    data, want = _neglogz_data()
    data["a"]["steps"][0]["mask"] = {"neglogz": None, "neglogz_tokens": 0, "tokens": 0, "neglogz_nonfinite": 0}
    first, _ = sc._m10(data, want, {"pass": True, "value": 0.01})
    assert first["pass"] is None and first["reasons"] == ["no valid completion token in any step's old pass"]


# ------------------------------------------------------------------------------------------ the self-check's flags
FAR = {**CLOSE, "mean_abs_diff": 1.7, "max_abs_diff": 9.3}


def test_a_far_apart_self_check_is_a_warning_and_leaves_the_verdict_alone(tmp_path, masked_cfg, capsys):
    run = _masked_run(tmp_path, masked_cfg, selfcheck=lambda n, s: _selfcheck(s - 1, masked=dict(FAR)) if n == "a"
                      else _selfcheck(s - 1))
    assert sc.main(["--run", str(run), "--stage", "final", "--config", str(MASKED), "--rc", "a=0", "--rc", "b=0"]) == 0
    rep = json.loads((run / "summary" / "smoke.json").read_text())
    d = rep["d021"]
    flag = "launch a step 1 (support_ids against fill): masked: mean |diff| 1.7 > 0.05; masked: max |diff| 9.3 > 1.0"
    assert d["selfcheck_flags"] == [flag] and d["verdict"] == "pass" and sc.SELFCHECK_NOTE not in d["follows"]
    assert f"D-021 self-check (a diagnostic, not a gate): {flag}" in rep["warnings"]
    lines = capsys.readouterr().out.strip().splitlines()
    assert f"[smoke] warning: D-021 self-check (a diagnostic, not a gate): {flag}" in lines
    assert lines[-2:] == ["D021_PART_A pass", "GRPO_SMOKE_OK"]


def test_a_close_self_check_raises_no_flag(tmp_path, masked_cfg):
    from lofi_rl.rl.masked import diff_stats
    assert set(CLOSE) == set(diff_stats([[0.0]], [[0.0]], [1])), "the fixture's stats have the trainer's keys"
    at_the_bars = {**CLOSE, "mean_abs_diff": sc.SELFCHECK_MEAN_ABS_DIFF, "max_abs_diff": sc.SELFCHECK_MAX_ABS_DIFF}
    checks = (lambda n, s: _selfcheck(s - 1), lambda n, s: _selfcheck(s - 1, masked=dict(at_the_bars)))
    for k, check in enumerate(checks):
        rep = _d021(_masked_run(tmp_path / str(k), masked_cfg, selfcheck=check), masked_cfg)
        assert rep["d021"]["selfcheck_flags"] == [] and not any("self-check" in w for w in rep["warnings"])


def test_a_measured_miss_with_a_flagged_self_check_says_the_fix_presupposes_a_numerical_miss(tmp_path, masked_cfg):
    far = lambda n, s: _selfcheck(s - 1, masked=dict(FAR))
    d = _d021(_masked_run(tmp_path / "far", masked_cfg, value=lambda s: 0.12, selfcheck=far), masked_cfg)["d021"]
    assert d["verdict"] == "measured_miss" and d["not_passing"] == ["M9", "M10_ratio"], "the verdict stays the gates'"
    assert len(d["selfcheck_flags"]) == 2 and d["follows"].endswith(sc.SELFCHECK_NOTE)
    assert d["follows"].startswith(sc.D021_FOLLOWS["measured_miss"])
    close = _d021(_masked_run(tmp_path / "close", masked_cfg, value=lambda s: 0.12), masked_cfg)["d021"]
    assert close["verdict"] == "measured_miss" and close["follows"] == sc.D021_FOLLOWS["measured_miss"]
    invalid = _d021(_masked_run(tmp_path / "invalid", masked_cfg, value=lambda s: 0.12, selfcheck=far,
                                meta=lambda n: {"stamp": None}), masked_cfg)["d021"]
    assert invalid["verdict"] == "invalid" and sc.SELFCHECK_NOTE not in invalid["follows"]


def test_the_self_check_flags_name_each_way_it_cannot_vouch_for_the_path():
    errored = _selfcheck(0, masked=None, unmasked=None, error="RuntimeError: CUDA error")
    flags = sc._selfcheck_flags({"a": {**errored, "step": 1}, "b": {"step": 6, "record": "x"}}, ["a", "b"])
    assert flags == ["launch a step 1 (support_ids against fill): it failed (RuntimeError: CUDA error); no masked "
                     "comparison; no unmasked comparison",
                     "launch b step 6 (None against None): no masked comparison; no unmasked comparison"]
    nothing = {**CLOSE, "positions": 0, "mean_abs_diff": None, "max_abs_diff": None}
    flags = sc._selfcheck_flags({"a": _selfcheck(1, masked={**CLOSE, "nonfinite": 3}, unmasked=nothing)}, ["a", "b"])
    assert flags == ["launch a step 1 (support_ids against fill): masked: non-finite differences 3, not 0; unmasked: "
                     "no finite difference compared (mean |diff| None, max None)",
                     "launch b: no self-check record (the launch's first old pass with a valid row runs it)"]
    assert sc._selfcheck_flags(None, ["a"]) == [
        "launch a: no self-check record (the launch's first old pass with a valid row runs it)"]


# ------------------------------------------------------------------------------------------ M12's breakdown
def test_m12_breaks_the_step_down_by_the_masks_timings_and_still_gates_on_step_s(tmp_path, masked_cfg):
    from lofi_rl.rl.masked import new_step_record
    assert sc.MASK_TIMING == tuple(new_step_record(0, "support_ids", True)["timing"])
    first = {("a", 1), ("b", 6)}                        # each launch's first old pass also runs the self-check

    def timing(n, s):
        check = 1.5 if (n, s) in first else 0.0
        return {"timing": {"walk_s": 0.4, "nfa_s": 0.1, "old_s": 6.0 + check, "ref_s": 3.0, "loss_s": 9.0,
                           "unmasked_s": 2.5, "selfcheck_s": check}}
    d = _d021(_masked_run(tmp_path, masked_cfg, mask=timing, step_s=lambda n, s: 45.0), masked_cfg)["d021"]
    m12 = d["gates"]["M12"]
    assert m12["pass"] is False and m12["value"] == pytest.approx(45.0), "M12 reads step_s, as registered"
    assert m12["reads"] == "mean step 45 s over 15 steps; bar <= 43.3 s"
    assert d["verdict"] == "pass" and d["estimate"] == {"M12": False}
    br = m12["breakdown"]
    means = {f: v["mean_s"] for f, v in br["mask_timing"].items()}
    assert means == pytest.approx({"walk_s": 0.4, "nfa_s": 0.1, "old_s": 6.2, "ref_s": 3.0, "loss_s": 9.0,
                                   "unmasked_s": 2.5, "selfcheck_s": 0.2})
    assert all(v["steps"] == 15 for v in br["mask_timing"].values())
    assert br["step_s_net"]["mean_s"] == pytest.approx((13 * 42.5 + 2 * 41.0) / 15) and br["step_s_net"]["steps"] == 15
    d2 = _d021(_masked_run(tmp_path / "2", masked_cfg, step_s=lambda n, s: 45.0,
                           mask=lambda n, s: {"timing": None} if (n, s) == ("b", 9) else timing(n, s)), masked_cfg)
    br2 = d2["d021"]["gates"]["M12"]["breakdown"]
    assert all(v["steps"] == 14 for v in br2["mask_timing"].values()) and br2["step_s_net"]["steps"] == 14


# ------------------------------------------------------------------------------------------ the IS readout's caveat
def test_the_vllm_caveat_reads_the_masked_trainers_way_under_a_masked_config(tmp_path, masked_cfg):
    rule = _d021(_masked_run(tmp_path / "m", masked_cfg), masked_cfg)["is_ratio"]["rule"]
    assert sc.VLLM_CAVEAT_MASKED in rule["caveats"] and sc.VLLM_CAVEAT not in rule["caveats"]
    assert "approach the token-weighted -log Z" in sc.VLLM_CAVEAT_MASKED and "M10's ratio" in sc.VLLM_CAVEAT_MASKED
    assert "can understate" not in sc.VLLM_CAVEAT_MASKED
    plain = sc.check(_d020_run(tmp_path / "p"), "final", _cfg(), {"a": "0", "b": "0"})["is_ratio"]["rule"]
    assert sc.VLLM_CAVEAT in plain["caveats"] and sc.VLLM_CAVEAT_MASKED not in plain["caveats"], "D-020's text stays"
    run = _masked_run(tmp_path / "c", masked_cfg)
    _set_vllm_logprobs(run, a=True, b=True)
    rule = _d021(run, masked_cfg)["is_ratio"]["rule"]
    assert sc.VLLM_CAVEAT_MASKED not in rule["caveats"] and sc.VLLM_CAVEAT not in rule["caveats"]
