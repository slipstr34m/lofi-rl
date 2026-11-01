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

What the D-020 settings add to the report (docs/GRPO.md §5):
- **in_language** (a constrained config): every completion's finish reason and whether it fullmatches the regex,
  as the reward logged them, per launch and per step. The smoke fails if any "stop" completion is out of the
  language (the engine did not apply the grammar as `re` reads it; D-020, Validity), if a completion has no such
  record, if any has the finish reason "other" (none from the token ids, so the check cannot judge it; D-020's
  Validity asks other.n to be 0), if a launch scored completions but none ended by stop (then nothing shows the
  grammar applied), or if a launch's run.json names another regex. A launch's failures here are judged at the final
  stage. Its shows_grammar is true only when the launch scored completions, named the config's regex and has none
  of these failures.
- **backend**: each launch's record of the structured-outputs pin on the engine TRL built (requested, pin_accepted,
  backend, source, applied, engine). It decides nothing, as in junk.json.
- **copy** (a config with copy_penalty): the pool files run.json records, P_copy's mean, the completions it penalized
  and the largest J, per launch; the smoke fails if the reward logged no P_copy, or if a launch's run.json records
  another number of pool files than the config's pool holds (the trainer must hand the pool to the copy penalty).
- **is_ratio**: D-020's IS readout, the mean over completion tokens of |log pi_vLLM - log pi_trainer|, per step.
  TRL's logged sampling/sampling_logp_difference/mean stands in for it only when every launch's run.json says the
  installed TRL's source was checked to compute mean |old_per_token_logps - sampling_per_token_logps| over
  completion_mask (trl_logp_metric.verdict "match"); otherwise the readout is unavailable, with the reason, and
  nothing is claimed. That check is TRL's half of D-020's definition: whether vLLM's reported log-probs include the
  grammar mask is unchecked (run.json vllm_logprobs), so with the grammar the rule carries that caveat unless every
  launch's vllm_logprobs.checked is true (is_ratio's vllm_logprobs_checked records them). Every step of both
  launches counts (the resumed steps 6-10 are steps the smoke ran); the mean of the per-step values, and the
  token-weighted mean that D-020's "over every completion token" reads (weights: each step's completion tokens from
  rewards.jsonl, without the truncated completions when TRL's mask drops them), are both reported, and the 0.05
  rule reads the token-weighted mean when every step has its tokens, else the mean of the per-step values, with a
  caveat and a warning that say so. The rule is D-020's reading (`applies`) only for a config with the grammar and
  prompt v2 (`configured`) whose two launches both showed the grammar applied (`grammar_shown`: each launch's
  in_language shows_grammar), since by D-020's Validity a constrained run that fails the in-language check decides
  nothing. An unavailable readout is a warning, not a failure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from lofi_rl.rl.config import ROOT, ConfigError, RunConfig, load_config

DEFAULT_CONFIG = ROOT / "configs" / "grpo_smoke.yaml"
ADAPTER = ("adapter_model.safetensors", "adapter_model.bin")
OPTIMIZER = ("optimizer.pt", "optimizer.bin", "optimizer.safetensors")
OPTIONAL_STATE = ("scheduler.pt", "rng_state.pth")
FINISH = ("stop", "length", "other")
IS_METRIC = "sampling/sampling_logp_difference/mean"        # TRL's metric (lofi_rl.rl.trl_probe.METRIC)
IS_THRESHOLD = 0.05                                         # D-020: the grammar goes off if the mean exceeds it
IS_DEFINITION = ("the mean, over every completion token of every step, of |log pi_vLLM(token) - "
                 "log pi_trainer(token)| (docs/DECISIONS.md D-020, Outcome)")
VLLM_CAVEAT = ("vLLM's reported log-probs were not checked to be the masked sampler's pi_vLLM (run.json "
               "vllm_logprobs.checked is not true): if vLLM takes them before the grammar bitmask, TRL's value leaves "
               "out the mask's mismatch with the trainer and can understate D-020's mean")


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
            problems: list[str], warnings: list[str],
            data: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One launch's summary: its status, the steps it trained, the reward's steps and group sizes. `data`, when
    given, receives the launch's run.json, step records and reward records for the readouts across launches."""
    out = run / "summary" / name
    if not out.is_dir():
        problems.append(f"summary/{name} is missing: launch {name} wrote nothing")
        return {"ran": False}, []
    meta = _json(out / "run.json", problems)
    recs = _jsonl(out / "steps.jsonl", problems)
    rewards = _jsonl(out / "rewards.jsonl", problems)
    if data is not None:
        data.update(meta=meta, steps=[r for r in recs if r.get("kind") == "step"], rewards=rewards)
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
            "profile": meta.get("profile"), "gpu": meta.get("gpu"), "versions": meta.get("versions"),
            "verified_trl_fields": meta.get("verified_trl_fields"),
            "dropped_optional_fields": meta.get("dropped_optional_fields"), "prompt_tokens": (meta.get("prompt") or {}).get("tokens"),
            "first_step": trained[0] if trained else None, "steps": len(trained), "trained": trained,
            "reward_steps": sorted(s for s in by_step if isinstance(s, int)), "completions": len(rewards),
            "reward_errors": errors, "step_sources": sources}
    if meta.get("dropped_optional_fields"):
        warnings.append(f"launch {name}: TRL lacks optional fields {meta['dropped_optional_fields']}")
    if cfg.structured_outputs is not None:
        info["in_language"] = _language(name, meta, rewards, cfg, problems)
        info["backend"] = (meta.get("structured_outputs") or {}).get("backend")
    if cfg.copy_penalty is not None:
        info["copy"] = _copy(name, meta, rewards, cfg, problems)
    rows = []
    for r in step_recs:
        trl, lofi = r.get("trl") or {}, r.get("lofi") or {}
        row = {"run": name, "step": r.get("step"), "reward_mean": lofi.get("reward_mean"),
               "reward_std": lofi.get("reward_std"), "trl_reward": trl.get("reward"),
               "trl_reward_std": trl.get("reward_std"), "step_s": r.get("step_s"), "kl": trl.get("kl"),
               "completion_mean_length": trl.get("completions/mean_length"),
               "gate_rate": lofi.get("gate_rate"), "loss": trl.get("loss"),
               "trl_logp_difference_mean": trl.get(IS_METRIC)}
        if cfg.structured_outputs is not None:
            stop = ((lofi.get("in_language") or {}).get("stop")) or {}
            row.update(stop_n=stop.get("n"), stop_in_language=stop.get("in_language"))
        if cfg.copy_penalty is not None:
            row["p_copy_mean"] = (lofi.get("copy") or {}).get("p_copy_mean")
        rows.append(row)
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


def _language(name: str, meta: dict[str, Any], rewards: list[dict[str, Any]], cfg: RunConfig,
              problems: list[str]) -> dict[str, Any]:
    """A constrained launch's completions in the regex's language, by finish reason (module docstring). Its
    shows_grammar is true only when the launch scored completions, its run.json names the config's regex and none of
    the failures below was found: the one predicate the IS rule's `applies` reads."""
    want = cfg.structured_outputs["sha256"]
    got = (meta.get("structured_outputs") or {}).get("sha256")
    mine: list[str] = []
    if meta and got != want:
        mine.append(f"launch {name}: run.json's regex sha256 {str(got)[:16]} is not the config's {want[:16]}")
    counts = {g: {"n": 0, "in_language": 0, "out_of_language": 0} for g in FINISH}
    by_step: dict[Any, dict[str, int]] = {}
    unrecorded, out = 0, []
    for r in rewards:
        finish = r.get("finish")
        if finish not in FINISH or not isinstance(r.get("in_language"), bool):
            unrecorded += 1
            continue
        c = counts[finish]
        c["n"] += 1
        c["in_language" if r["in_language"] else "out_of_language"] += 1
        s = by_step.setdefault(r.get("step"), {"stop": 0, "stop_out_of_language": 0})
        if finish == "stop":
            s["stop"] += 1
            if not r["in_language"]:
                s["stop_out_of_language"] += 1
                if len(out) < 5:
                    out.append({"step": r.get("step"), "i": r.get("i"), "text_sha256": r.get("text_sha256")})
    if unrecorded:
        mine.append(f"launch {name}: {unrecorded} of {len(rewards)} completions have no in-language record (a "
                    "constrained run's reward counts every one)")
    other = counts["other"]["n"]
    if other:
        mine.append(f"launch {name}: {other} of {len(rewards)} completions have no finish reason from the token ids "
                    "(completion_ids or the limit unknown), so the in-language check cannot judge them (D-020, "
                    "Validity: other.n is 0)")
    if rewards and counts["stop"]["n"] == 0:
        mine.append(f"launch {name}: no completion ended by stop, so nothing shows the grammar applied")
    bad = counts["stop"]["out_of_language"]
    if bad:
        mine.append(f"launch {name}: {bad} of {counts['stop']['n']} stop completions are not in the regex's "
                    "language: the engine did not apply the grammar as re reads it, or the text TRL decoded for "
                    f"the reward is not the text vLLM generated (D-020, Validity); first {out[:2]}")
    problems += mine
    return {"regex_sha256": want, **counts, "unrecorded": unrecorded, "first_out_of_language": out,
            "shows_grammar": bool(rewards) and got == want and not mine,
            "by_step": {str(k): v for k, v in sorted(by_step.items(), key=lambda kv: (kv[0] is None, kv[0]))}}


def _copy(name: str, meta: dict[str, Any], rewards: list[dict[str, Any]], cfg: RunConfig,
          problems: list[str]) -> dict[str, Any]:
    """A launch's P_copy: the pool files its run.json records (the config's, or a failure), its mean, the
    completions it penalized, the largest J and the nearest references."""
    recs = [r for r in rewards if "p_copy" in r]
    if rewards and len(recs) != len(rewards):
        problems.append(f"launch {name}: the config sets copy_penalty but {len(rewards) - len(recs)} of "
                        f"{len(rewards)} completions have no p_copy")
    want = len(cfg.pool_files())
    got = ((meta.get("copy_penalty") or {}).get("pool") or {}).get("files")
    if meta and got != want:
        problems.append(f"launch {name}: run.json's copy penalty holds {got} pool files, not the config's {want} "
                        "(D-020: P_copy is scored against the pool)")
    js = [r["copy_j"] for r in recs if isinstance(r.get("copy_j"), (int, float))]
    pen = [r for r in recs if r["p_copy"] > 0]
    return {"pool_files": got, "n": len(recs),
            "p_copy_mean": statistics.fmean(r["p_copy"] for r in recs) if recs else None,
            "penalized": len(pen), "at_least_0.8": sum(1 for j in js if j >= 0.8), "j_max": max(js, default=None),
            "nearest": dict(sorted(Counter(r.get("copy_nearest") for r in pen).items())),
            "nearest_is_shown": sum(1 for r in pen if r.get("example_id") is not None
                                    and r.get("copy_nearest") == r.get("example_id"))}


def _number(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _step_tokens(rewards: list[dict[str, Any]], drop_truncated: bool) -> dict[Any, tuple[int | None, int]]:
    """Per reward step: (completion tokens, None if any is unknown; truncated completions). With drop_truncated, the
    truncated completions' tokens are left out, as TRL's mask leaves them out under mask_truncated_completions."""
    out: dict[Any, list] = {}
    for r in rewards:
        e = out.setdefault(r.get("step"), [0, 0])
        n, trunc = r.get("n_tokens"), r.get("truncated")
        if trunc is True:
            e[1] += 1
        if e[0] is None:
            continue
        if not isinstance(n, int) or trunc is None:
            e[0] = None
        elif not (drop_truncated and trunc):
            e[0] += n
    return {k: (v[0], v[1]) for k, v in out.items()}


def _is_readout(launches: dict[str, dict[str, Any]], cfg: RunConfig, warnings: list[str],
                grammar_shown: bool = False) -> dict[str, Any]:
    """D-020's IS readout from TRL's per-step metric, read only when its source was checked (module docstring).
    grammar_shown: both launches' in_language showed the grammar applied (check() reads it); without it the rule
    is not D-020's reading, whatever the config."""
    sources = {n: (d.get("meta") or {}).get("trl_logp_metric") for n, d in launches.items()}
    verdicts = {n: (s or {}).get("verdict") for n, s in sources.items()}
    configured = cfg.structured_outputs is not None and cfg.prompt == "v2"
    applies = configured and grammar_shown
    # TRL's completion mask leaves truncated completions out when mask_truncated_completions is on (the weights do
    # the same); whether the source was seen to do it is in excludes_truncated
    excludes = cfg.trainer.get("mask_truncated_completions") is True
    seen = bool(sources) and all((s or {}).get("excludes_truncated") is True for s in sources.values())
    out: dict[str, Any] = {
        "definition": IS_DEFINITION, "metric": IS_METRIC, "threshold": IS_THRESHOLD, "applies": applies,
        "configured": configured, "grammar_shown": grammar_shown, "verdicts": verdicts,
        "source": {n: None if s is None else {k: s.get(k) for k in ("package", "version", "verdict", "file", "sha256",
                                                                     "sites", "missing", "excludes_truncated",
                                                                     "lines", "error")}
                   for n, s in sources.items()}}
    rows = []
    for n, d in launches.items():
        tokens = _step_tokens(d.get("rewards") or [], drop_truncated=excludes)
        for r in d.get("steps") or []:
            s = r.get("step")
            t, trunc = tokens.get(s - 1 if isinstance(s, int) else None, (None, 0))
            rows.append({"run": n, "step": s, "value": _number((r.get("trl") or {}).get(IS_METRIC)), "tokens": t,
                         "truncated": trunc})
    out["steps"] = rows
    # trl_logp_metric checks TRL's half of the definition; whether vLLM's sampling log-probs are the masked sampler's
    # (D-020's pi_vLLM under the grammar) is recorded by run.json's vllm_logprobs, never checked today
    vllm_checked = {n: ((d.get("meta") or {}).get("vllm_logprobs") or {}).get("checked") for n, d in launches.items()}
    out["vllm_logprobs_checked"] = vllm_checked
    usable = bool(launches) and all(v == "match" for v in verdicts.values())
    reason = None
    if not usable:
        reason = ("the installed TRL's source was not checked to compute D-020's definition (trl_logp_metric "
                  f"verdicts {verdicts})")
    elif not any(r["value"] is not None for r in rows):
        reason = f"TRL logged no {IS_METRIC} at any step"
    if reason is not None:
        out.update(value_source="unavailable", reason=reason, mean_of_steps=None, token_weighted_mean=None,
                   by_launch=None, rule=None)
        warnings.append(f"IS readout unavailable: {reason}")
        return out
    have = [r for r in rows if r["value"] is not None]
    if len(have) != len(rows):
        warnings.append(f"IS readout: {len(rows) - len(have)} of {len(rows)} steps have no {IS_METRIC}")

    def means(rs: list[dict[str, Any]]) -> tuple[float | None, float | None]:
        vals = [r["value"] for r in rs if r["value"] is not None]
        plain = statistics.fmean(vals) if vals else None
        known = [r for r in rs if r["value"] is not None and r["tokens"] is not None]
        tok = sum(r["tokens"] for r in known)
        weighted = (sum(r["value"] * r["tokens"] for r in known) / tok
                    if vals and len(known) == len(vals) and tok > 0 else None)
        return plain, weighted
    plain, weighted = means(rows)
    by_launch = {n: dict(zip(("mean_of_steps", "token_weighted_mean"), means([r for r in rows if r["run"] == n])))
                 for n in launches}
    reads = "token_weighted_mean" if weighted is not None else "mean_of_steps"
    value = weighted if weighted is not None else plain
    exceeds = value > IS_THRESHOLD
    caveats = []
    if weighted is None:
        miss = [(r["run"], r["step"]) for r in rows if r["value"] is not None and r["tokens"] is None]
        why = (f"steps {miss} have no completion-token count" if miss
               else "the steps with a value have no completion tokens to weight it by")
        caveats.append(f"{why}, so the rule reads the mean of the per-step values, not D-020's mean over every "
                       "completion token")
        warnings.append(f"IS readout: {why}, so the 0.05 rule reads the mean of the per-step values, not D-020's "
                        "token-weighted mean")
    cut = [(r["run"], r["step"]) for r in rows if r["truncated"]]
    if cut and excludes:
        how = ("its source drops them from completion_mask" if seen else
               "mask_truncated_completions is on, though the source was not seen to drop them")
        caveats.append(f"steps {cut} have truncated completions, and TRL's value leaves their tokens out ({how}); "
                       "D-020 reads every completion token, so on those steps the two differ")
    if plain is not None and weighted is not None and (plain > IS_THRESHOLD) != (weighted > IS_THRESHOLD):
        caveats.append(f"the mean of the per-step values ({plain:.6f}) and the token-weighted mean ({weighted:.6f}) "
                       "fall on two sides of 0.05")
    if cfg.structured_outputs is not None and not all(c is True for c in vllm_checked.values()):
        caveats.append(VLLM_CAVEAT)
    if not configured:
        says = "not D-020's reading: it is read in a GRPO smoke with the grammar and prompt v2"
    elif not grammar_shown:
        says = ("not D-020's reading: this smoke did not show the grammar applied in both launches (in_language "
                "failed), so it decides nothing (D-020, Validity)")
        warnings.append("IS readout: the in-language check did not show the grammar applied in both launches, so the "
                        "0.05 rule decides nothing")
    elif exceeds:
        says = "exceeds 0.05: by D-020 the grammar goes off for GRPO run 1, and run 1's start is registered again"
    else:
        says = ("at most 0.05: this reading keeps the grammar on (D-020's other reading, the step-0 parse failures "
                "without it, decides separately)")
    out.update(value_source=f"trl:{IS_METRIC}", reason=None, mean_of_steps=plain, token_weighted_mean=weighted,
               by_launch=by_launch,
               rule={"threshold": IS_THRESHOLD, "reads": reads, "value": value, "exceeds": exceeds,
                     "applies": applies, "says": says, "caveats": caveats})
    return out


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
    data: dict[str, dict[str, Any]] = {"a": {}}
    first, rows_a = _launch(run, "a", cfg, list(range(1, max_steps + 1)), 0, later, warnings, data["a"])
    if stage == "final":
        problems += later
    else:
        warnings += [f"judged at the final stage: {p}" for p in later]
    rep: dict[str, Any] = {"stage": stage, "run": str(run), "config": cfg.path, "config_sha256": cfg.sha256,
                           "prompt": cfg.prompt, "structured_outputs": None if cfg.structured_outputs is None
                           else {k: v for k, v in cfg.structured_outputs.items() if k != "regex"},
                           "copy_penalty": None if cfg.copy_penalty is None else {"pool": list(cfg.copy_penalty["pool"])},
                           "max_steps": max_steps, "save_steps": save, "resume_from_step": resume,
                           "exit_codes": dict(exit_codes or {}), "first": {**first, "checkpoints": ckpts}}
    rows = rows_a
    if stage == "final":
        data["b"] = {}
        resumed, rows_b = _launch(run, "b", cfg, list(range(resume + 1, max_steps + 1)), resume, problems, warnings,
                                  data["b"])
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
    ran = {n: d for n, d in data.items() if d}
    if cfg.structured_outputs is not None:
        langs = {n: info["in_language"] for n, info in (("a", first), ("b", rep.get("resumed") or {}))
                 if info.get("in_language") is not None}
        rep["in_language"] = {"regex_sha256": cfg.structured_outputs["sha256"], "launches": langs,
                              **{g: {k: sum(x[g][k] for x in langs.values())
                                     for k in ("n", "in_language", "out_of_language")} for g in FINISH}}
        rep["backend"] = {n: info.get("backend") for n, info in (("a", first), ("b", rep.get("resumed") or {}))
                          if info.get("ran")}
    if cfg.copy_penalty is not None:
        rep["copy"] = {n: info.get("copy") for n, info in (("a", first), ("b", rep.get("resumed") or {}))
                       if info.get("ran")}
    infos = [i for i in (first, rep.get("resumed") or {}) if i.get("ran")]
    shown = (cfg.structured_outputs is not None and len(infos) == 2
             and all((i.get("in_language") or {}).get("shows_grammar") is True for i in infos))
    rep["is_ratio"] = _is_readout(ran, cfg, warnings, grammar_shown=shown) if stage == "final" else None
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
    lang = rep.get("in_language")
    if lang:
        print(f"[smoke] in the regex's language ({lang['regex_sha256'][:16]}): stop {lang['stop']['in_language']}/"
              f"{lang['stop']['n']}, length {lang['length']['in_language']}/{lang['length']['n']}, other "
              f"{lang['other']['in_language']}/{lang['other']['n']}")
    for name, b in (rep.get("backend") or {}).items():
        b = b or {}
        print(f"[smoke] backend {name}: requested {b.get('requested')} pin_accepted {b.get('pin_accepted')} "
              f"backend {b.get('backend')} applied {b.get('applied')} engine {b.get('engine')}")
    for name, c in (rep.get("copy") or {}).items():
        c = c or {}
        print(f"[smoke] copy {name}: pool files {c.get('pool_files')} p_copy mean {c.get('p_copy_mean')} penalized "
              f"{c.get('penalized')}/{c.get('n')} J>=0.8 {c.get('at_least_0.8')} max J {c.get('j_max')}")
    ratio = rep.get("is_ratio")
    if ratio:
        rule = ratio.get("rule") or {}
        print(f"[smoke] IS readout ({ratio['value_source']}): mean of steps {ratio.get('mean_of_steps')}, "
              f"token-weighted {ratio.get('token_weighted_mean')}; rule: {rule.get('says', ratio.get('reason'))}")
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
