"""The GRPO smoke's verdict (src/lofi_rl/rl/smoke_check.py), on synthetic run directories laid out like the
ones jobs/grpo_smoke.sbatch leaves: run a trains steps 1-10 and saves checkpoints 5 and 10; run b resumes
from a/checkpoint-5 and must train exactly steps 6-10 (taste_pipeline.md §6, smoke row)."""
import json
from pathlib import Path

import pytest

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
V2 = Path(__file__).resolve().parents[1] / "configs" / "grpo_smoke_v2_grammar.yaml"
MATCH = {"verdict": "match", "package": "trl", "version": "1.17.1", "file": "trainer/grpo_trainer.py",
         "sha256": "a" * 64, "sites": [], "missing": [], "excludes_truncated": True, "lines": {"1": "x"}}


def _cfg(path=V2):
    from lofi_rl.rl.config import load_config
    return load_config(path)


def _d020(out: Path, steps, *, start_step=0, resume_from=None, cfg=None, out_of_language=(), source=MATCH,
          value=lambda s: 0.01, tokens=lambda k, i: 100, truncated=lambda k, i: False, copy=True, sha=None):
    """A launch's summary with the D-020 fields the reward and the trainer write: finish, in_language, p_copy and
    copy_j per completion; TRL's IS metric per step; the regex sha, the backend record and trl_logp_metric."""
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
    (out / "run.json").write_text(json.dumps({"status": "ok", "resume_from": resume_from, "start_step": start_step,
                                              "final_global_step": steps[-1], "startup_s": 60.0,
                                              "structured_outputs": so, "trl_logp_metric": source}))


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
    assert rep["copy"]["a"] == {"n": 320, "p_copy_mean": 0.3 / 32, "penalized": 10, "at_least_0.8": 10, "j_max": 1.0,
                                "nearest": {"example": 10}, "nearest_is_shown": 10}
    isr = rep["is_ratio"]
    steps = list(range(1, 11)) + list(range(6, 11))
    values, tokens = [0.001 * s for s in steps], [32 * (200 + 10 * (s - 1)) for s in steps]
    assert [r["value"] for r in isr["steps"]] == pytest.approx(values) and [r["tokens"] for r in isr["steps"]] == tokens
    assert isr["mean_of_steps"] == pytest.approx(sum(values) / 15)
    assert isr["token_weighted_mean"] == pytest.approx(sum(v * t for v, t in zip(values, tokens)) / sum(tokens))
    assert isr["by_launch"]["a"]["mean_of_steps"] == pytest.approx(0.0055)
    assert isr["rule"]["applies"] is True and isr["rule"]["exceeds"] is False and isr["rule"]["caveats"] == []
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


def test_a_length_completion_out_of_the_language_is_counted_apart_and_passes(tmp_path):
    run = _d020_run(tmp_path, truncated=lambda k, i: i == 5, out_of_language={(k, 5) for k in range(10)})
    rep = sc.check(run, "final", _cfg(), {"a": "0", "b": "0"})
    assert rep["ok"], rep["failures"]
    assert rep["in_language"]["length"] == {"n": 15, "in_language": 0, "out_of_language": 15}
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


def test_on_todays_smoke_the_is_rule_is_not_d020s_reading(tmp_path):
    from lofi_rl.rl.config import ROOT
    smoke = _cfg(ROOT / "configs" / "grpo_smoke.yaml")
    rep = sc.check(_d020_run(tmp_path, cfg=smoke, copy=False), "final", smoke, {"a": "0", "b": "0"})
    assert rep["ok"], rep["failures"]
    assert "in_language" not in rep and "backend" not in rep and "copy" not in rep
    assert rep["is_ratio"]["rule"]["applies"] is False and "not D-020's reading" in rep["is_ratio"]["rule"]["says"]
