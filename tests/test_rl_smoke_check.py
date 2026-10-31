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
    rc, rep, _ = _final(run, capsys)
    assert rc == 1 and any("3 reward errors" in f for f in rep["failures"])


def test_a_launch_that_never_wrote_its_summary_is_reported_not_raised(tmp_path, capsys):
    run = tmp_path / "grpo-smoke-2"
    (run / "summary").mkdir(parents=True)
    rc, rep, out = _final(run, capsys)
    assert rc == 1 and rep["ok"] is False and "GRPO_SMOKE_FAILED" in out
    assert any("summary/a" in f for f in rep["failures"])
