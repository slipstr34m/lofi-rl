"""The GRPO smoke's verdict (docs/design/taste_pipeline.md §6, smoke row; jobs/grpo_smoke.sbatch; docs/GRPO.md §5).

    python -m lofi_rl.rl.smoke_check --run <RUN> --stage first|final [--config configs/grpo_smoke.yaml]

The job leaves, under <RUN>:
    a/checkpoint-5, a/checkpoint-10     launch 1: steps 1-10, saved every 5
    b/checkpoint-10                     launch 2: --resume-from a/checkpoint-5, steps 6-10
    summary/a/, summary/b/              each launch's run.json, steps.jsonl and rewards.jsonl (lofi_rl.rl.train)

--stage first (between the launches): launch 1 finished, and checkpoint-5 and checkpoint-10 exist with their
trainer state, adapter and optimizer, so a relaunch can resume. Prints FIRST_RUN_OK or FIRST_RUN_FAILED.
--stage final: also launch 1 trained steps 1-10; launch 2 started at step 5 and trained exactly steps 6-10 (the
resumed step count) and saved checkpoint-10; the reward scored max_steps x prompts x num_generations completions
in groups of num_generations, at the curriculum steps 0-9 and 5-9; no reward errors. Writes summary/smoke.json
with the per-step reward and time, and prints GRPO_SMOKE_OK or GRPO_SMOKE_FAILED <reasons> last.
Both stages write summary/smoke.json, never raise on a missing or broken file (each is a reported failure), and
read the steps, save points and batch size from the run config. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from lofi_rl.rl.config import ROOT, ConfigError, RunConfig, load_config

DEFAULT_CONFIG = ROOT / "configs" / "grpo_smoke.yaml"
ADAPTER = ("adapter_model.safetensors", "adapter_model.bin")
OPTIMIZER = ("optimizer.pt", "optimizer.bin", "optimizer.safetensors")
OPTIONAL_STATE = ("scheduler.pt", "rng_state.pth")


def _jsonl(path: Path, problems: list[str]) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except OSError as err:
        problems.append(f"{path}: cannot read ({err.strerror or err})")
        return []
    out = []
    for n, line in enumerate(lines, 1):
        try:
            out.append(json.loads(line))
        except ValueError:
            problems.append(f"{path}:{n}: not JSON")
    return out


def _json(path: Path, problems: list[str]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as err:
        problems.append(f"{path}: cannot read ({getattr(err, 'strerror', None) or err})")
        return {}


def _checkpoint(d: Path, step: int, problems: list[str], warnings: list[str]) -> dict[str, Any]:
    name = f"{d.parent.name}/{d.name}"
    if not d.is_dir():
        problems.append(f"{name} is missing")
        return {"exists": False}
    state = _json(d / "trainer_state.json", problems)
    got = state.get("global_step")
    if state and got != step:
        problems.append(f"{name}: trainer_state.json says global_step {got}, not {step}")
    files = {p.name for p in d.iterdir()}
    if not any(f in files for f in ADAPTER):
        problems.append(f"{name} has no adapter weights ({' or '.join(ADAPTER)})")
    if not any(f in files for f in OPTIMIZER):
        problems.append(f"{name} has no optimizer state ({OPTIMIZER[0]}): it cannot resume")
    for f in OPTIONAL_STATE:
        if f not in files:
            warnings.append(f"{name} has no {f}")
    return {"exists": True, "global_step": got, "files": sorted(files)}


def _launch(run: Path, name: str, cfg: RunConfig, steps_want: list[int], start_step: int,
            problems: list[str], warnings: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One launch's summary: its status, the steps it trained, the reward's steps and group sizes."""
    out = run / "summary" / name
    if not out.is_dir():
        problems.append(f"summary/{name} is missing: launch {name} wrote nothing")
        return {"ran": False}, []
    meta = _json(out / "run.json", problems)
    recs = _jsonl(out / "steps.jsonl", problems)
    rewards = _jsonl(out / "rewards.jsonl", problems)
    step_recs = [r for r in recs if r.get("kind") == "step"]
    trained = sorted({r.get("step") for r in step_recs if isinstance(r.get("step"), int)})
    if meta and meta.get("status") != "ok":
        problems.append(f"launch {name}: status {meta.get('status')!r}: {str(meta.get('error'))[:300]}")
    if len(step_recs) != len(trained):
        warnings.append(f"launch {name}: {len(step_recs)} step records for {len(trained)} steps")
    gens = cfg.trainer["num_generations"]
    by_step: dict[int, list[dict[str, Any]]] = {}
    for r in rewards:
        by_step.setdefault(r.get("step"), []).append(r)
    errors = sum(1 for r in rewards if r.get("error"))
    if errors:
        first = next(r["error"] for r in rewards if r.get("error"))
        problems.append(f"launch {name}: {errors} reward errors (first: {str(first)[:200]})")
    per_step = cfg.prompts_per_step * gens
    for s, rs in sorted(by_step.items(), key=lambda kv: (kv[0] is None, kv[0])):
        sizes = sorted(Counter(r.get("group") for r in rs).values())
        if len(rs) != per_step or sizes != [gens] * cfg.prompts_per_step:
            problems.append(f"launch {name}: reward step {s} scored {len(rs)} completions in groups {sizes}, "
                            f"not {cfg.prompts_per_step} groups of {gens}")
    sources = sorted({str(r.get("step_source")) for r in rewards})
    if rewards and sources != ["trainer_state"]:
        warnings.append(f"launch {name}: the curriculum step came from {sources}, not TRL's trainer_state")
    hooks = sorted({e for r in step_recs for e in ((r.get("lofi") or {}).get("hook_errors") or [])})
    if hooks:
        warnings.append(f"launch {name}: TRL's logging hooks failed: {hooks[:3]}")
    info = {"ran": True, "status": meta.get("status"), "resume_from": meta.get("resume_from"),
            "start_step": meta.get("start_step"), "final_global_step": meta.get("final_global_step"),
            "profile": meta.get("profile"), "dropped_optional_fields": meta.get("dropped_optional_fields"),
            "first_step": trained[0] if trained else None, "steps": len(trained), "trained": trained,
            "reward_steps": sorted(s for s in by_step if isinstance(s, int)), "completions": len(rewards),
            "reward_errors": errors, "step_sources": sources}
    if meta.get("dropped_optional_fields"):
        warnings.append(f"launch {name}: TRL lacks optional fields {meta['dropped_optional_fields']}")
    rows = []
    for r in step_recs:
        trl, lofi = r.get("trl") or {}, r.get("lofi") or {}
        rows.append({"run": name, "step": r.get("step"), "reward_mean": lofi.get("reward_mean"),
                     "reward_std": lofi.get("reward_std"), "trl_reward": trl.get("reward"),
                     "trl_reward_std": trl.get("reward_std"), "step_s": r.get("step_s"), "kl": trl.get("kl"),
                     "completion_mean_length": trl.get("completions/mean_length"),
                     "gate_rate": lofi.get("gate_rate"), "loss": trl.get("loss")})
    if trained != steps_want:
        what = "resumed run" if start_step else "first run"
        problems.append(f"launch {name} ({what}) trained steps {_span(trained)}, not {_span(steps_want)}")
    want_reward = [s - 1 for s in steps_want]
    if info["reward_steps"] != want_reward:
        problems.append(f"launch {name}: the reward scored curriculum steps {_span(info['reward_steps'])}, "
                        f"not {_span(want_reward)}")
    timing = {"startup_s": meta.get("startup_s"), "wall_s": meta.get("wall_s"),
              "train_s": round(sum(x["step_s"] for x in rows if isinstance(x["step_s"], (int, float))), 1)}
    info["timing"] = timing
    return info, rows


def _span(xs: list[int]) -> str:
    if not xs:
        return "none"
    return f"{xs[0]}-{xs[-1]}" if xs == list(range(xs[0], xs[-1] + 1)) else str(xs)


EXIT_MEANING = {"124": "timed out", "137": "killed after the timeout", "2": "config error", "3": "no policy env"}


def check(run: Path, stage: str, cfg: RunConfig, exit_codes: dict[str, str] | None = None) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []
    for name, code in sorted((exit_codes or {}).items()):
        if code == "skipped":
            problems.append(f"launch {name} did not run (an earlier step failed)")
        elif code != "0":
            why = EXIT_MEANING.get(code)
            problems.append(f"launch {name} exited {code}" + (f" ({why})" if why else ""))
    max_steps, save = cfg.trainer["max_steps"], cfg.trainer["save_steps"]
    resume = cfg.resume_step if cfg.resume_step is not None else save
    ckpts = {s: _checkpoint(run / "a" / f"checkpoint-{s}", s, problems, warnings)
             for s in range(save, max_steps + 1, save)}
    # The relaunch needs only launch a's exit code and checkpoints; its steps and rewards are judged at the end.
    later: list[str] = []
    first, rows_a = _launch(run, "a", cfg, list(range(1, max_steps + 1)), 0, later, warnings)
    if stage == "final":
        problems += later
    else:
        warnings += [f"judged at the final stage: {p}" for p in later]
    rep: dict[str, Any] = {"stage": stage, "run": str(run), "config": cfg.path, "config_sha256": cfg.sha256,
                           "max_steps": max_steps, "save_steps": save, "resume_from_step": resume,
                           "exit_codes": dict(exit_codes or {}), "first": {**first, "checkpoints": ckpts}}
    rows = rows_a
    if stage == "final":
        resumed, rows_b = _launch(run, "b", cfg, list(range(resume + 1, max_steps + 1)), resume, problems, warnings)
        if resumed.get("ran"):
            want = str(run / "a" / f"checkpoint-{resume}")
            got = resumed.get("resume_from")
            if not got or os.path.normpath(str(got)) != os.path.normpath(want):
                problems.append(f"launch b resumed from {got!r}, not {want}")
            if resumed.get("start_step") != resume:
                problems.append(f"launch b started at step {resumed.get('start_step')!r}, not {resume}")
            if resumed.get("final_global_step") != max_steps:
                problems.append(f"launch b ended at global step {resumed.get('final_global_step')!r}, not {max_steps}")
        resumed["checkpoint"] = _checkpoint(run / "b" / f"checkpoint-{max_steps}", max_steps, problems, warnings)
        rep["resumed"] = resumed
        rows = rows_a + rows_b
    rep["steps"] = rows
    rep["timing"] = {"first": first.get("timing")} | ({"resumed": rep["resumed"].get("timing")} if stage == "final" else {})
    rep["failures"], rep["warnings"], rep["ok"] = problems, warnings, not problems
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--stage", choices=("first", "final"), required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--rc", action="append", default=[], metavar="LAUNCH=CODE",
                    help="a launch's exit code (a=0, b=124, b=skipped); anything but 0 is a failure")
    a = ap.parse_args(argv)
    codes = dict(x.split("=", 1) for x in a.rc if "=" in x)
    try:
        cfg = load_config(a.config)
    except (OSError, ConfigError) as err:
        print(f"GRPO_SMOKE_FAILED: config {a.config}: {err}")
        return 1
    rep = check(a.run, a.stage, cfg, codes)
    out = a.run / "summary"
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "smoke.json.tmp"
    tmp.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    os.replace(tmp, out / "smoke.json")
    for row in rep["steps"]:
        print(f"[smoke] {row['run']} step {row['step']}: reward {row['reward_mean']} (std {row['reward_std']}) "
              f"kl {row['kl']} len {row['completion_mean_length']} {row['step_s']} s")
    for w in rep["warnings"]:
        print(f"[smoke] warning: {w}")
    print(f"[smoke] report: {out / 'smoke.json'}")
    tag = "FIRST_RUN" if a.stage == "first" else "GRPO_SMOKE"
    if rep["ok"]:
        print(f"{tag}_OK")
        return 0
    print(f"{tag}_FAILED: " + "; ".join(rep["failures"]))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
