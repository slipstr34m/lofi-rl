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
  launch's vllm_logprobs.checked is true (is_ratio's vllm_logprobs_checked records them). Under the masked trainer
  the caveat reads the other way (VLLM_CAVEAT_MASKED): log-probs taken before the bitmask would push TRL's value
  toward -log Z, not under D-020's mean, and M10's ratio settles it. Every step of both
  launches counts (the resumed steps 6-10 are steps the smoke ran); the mean of the per-step values, and the
  token-weighted mean that D-020's "over every completion token" reads (weights: each step's completion tokens from
  rewards.jsonl, without the truncated completions when TRL's mask drops them), are both reported, and the 0.05
  rule reads the token-weighted mean when every step has its tokens, else the mean of the per-step values, with a
  caveat and a warning that say so. The rule is D-020's reading (`applies`) only for a config with the grammar and
  prompt v2 (`configured`) whose two launches both showed the grammar applied (`grammar_shown`: each launch's
  in_language shows_grammar), since by D-020's Validity a constrained run that fails the in-language check decides
  nothing. An unavailable readout is a warning, not a failure.

What a masked config (`trainer_logprobs`, D-021 Part A; docs/GRPO.md §8) adds at the final stage:
- **d021**: D-021's gates M5-M12 as it registers them, each with its kind, value, bar, pass and source, read from
  both launches' run.json (stamp, trl_source_checks, lm_head, masked) and steps.jsonl (TRL's metrics and each step's
  mask record), and from this report's in_language, backend and is_ratio. Validity: M5 (one commit, dirty=0; and, in
  the tree this module runs from, config.ROOT, which on the cluster is the job's read-only snapshot, every file of
  docs/prereg/D-021_masked_smoke_path.txt at its frozen git blob and D-021's heading in docs/DECISIONS.md), M6
  (GRPO_SMOKE_OK, the language counts, xgrammar pinned and applied), M7 (T1-T5 match, T5's run-time half too), M8
  (no sampled token outside the mask, 32 old/ref/loss rows a step, no NFA mismatch, finite loss/KL/entropy, the path
  named) and M10's first clause (token-weighted mask/neglogz >= 0.05; a step whose old pass had no valid token
  weighs 0 and is listed as zero_weight). Measured: M9 (the token-weighted IS readout <= 0.05, every step read),
  M10's ratio (M9 <= 0.25 x neglogz) and M11 (at every step the 32 IS weights' mean >= 0.3 and largest <= 0.25 of
  their sum). Estimate-only: M12 (mean step_s <= 43.3 s; beside it, ungated, the mean of each mask timing and of
  step_s net of the old pass's unmasked forward and the self-check).
- **selfcheck_flags** (diagnostics, never gates): a launch whose first self-check is absent, failed, non-finite, or
  puts the two log-prob paths more than 0.05 apart in mean |diff| or 1.0 at most. Each flag is printed as a warning,
  and on a measured miss the follow-up adds that D-021's fp32 fix presupposes a numerical miss. The verdict never
  reads them.
- **The verdict**: `invalid` when a validity gate does not pass, or a measured gate cannot be decided (a check that
  cannot decide counts as not passing, and a launch that could not measure measured nothing); `measured_miss` when
  every validity gate passes and a measured gate misses; `pass` otherwise. M12 never changes it. The log prints
  D021_PART_A <verdict> just before GRPO_SMOKE_OK or GRPO_SMOKE_FAILED, which stays last and keeps its meaning: the
  D-021 gates never add to the smoke's failures (M6 reads them).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from lofi_rl.rl.config import MASK_GATING, ROOT, ConfigError, RunConfig, load_config

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
# The same unchecked record read under the masked trainer (D-021 Part A), whose log-probs are normalized over the
# allowed set: unmasked vLLM log-probs would sit below them by each position's -log Z, so the gap would grow, not shrink
VLLM_CAVEAT_MASKED = ("vLLM's reported log-probs were not checked to be the masked sampler's pi_vLLM (run.json "
                      "vllm_logprobs.checked is not true): under the masked trainer, if vLLM takes them before the "
                      "grammar bitmask, the trainer's masked log-probs exceed them by each position's -log Z, so TRL's "
                      "value would approach the token-weighted -log Z (mask/neglogz) rather than understate D-020's "
                      "mean; M10's ratio (M9 <= 0.25 x neglogz, smoke.json d021) settles it")
# D-021 Part A's gates (docs/DECISIONS.md D-021, "Pre-registered gates for Part A"); M10 has two clauses of two kinds
D021_VALIDITY = ("M5", "M6", "M7", "M8", "M10")             # M10 here is its first clause
D021_MEASURED = ("M9", "M10_ratio", "M11")
D021_ESTIMATE = ("M12",)
NEGLOGZ_MIN = 0.05                  # M10, first clause: the trainer's token-weighted -log Z
RATIO_OF_NEGLOGZ = 0.25             # M10, its ratio: M9 <= 0.25 x neglogz
WEIGHT_MEAN_MIN = 0.3               # M11: the mean of a step's 32 IS weights
WEIGHT_SHARE_MAX = 0.25             # M11: the largest weight's share of their sum
STEP_S_MAX = 43.3                   # M12: 1.15 x 13657200's mean step_s, 37.63 s (D-021)
BACKEND = "xgrammar"                # M6: pinned and applied in both launches (lofi_rl.rl.junk.STRUCTURED_BACKEND)
FINITE_LOGS = ("loss", "kl", "entropy")                     # M8: TRL's logged loss, KL and entropy
IS_WEIGHT_MEAN = "sampling/importance_sampling_ratio/mean"  # M11's stand-in, when T3 reads it as over sequences
IS_WEIGHT_MAX = "sampling/importance_sampling_ratio/max"
# M5's registration half (D-021: "this entry and the frozen path are committed and synced before the submit"), read
# from the tree this module runs from (config.ROOT: on the cluster the job's read-only code snapshot)
PREREG_LIST = "docs/prereg/D-021_masked_smoke_path.txt"     # "<git blob sha>  <path>" lines, as check_frozen.sh reads
DECISIONS = "docs/DECISIONS.md"
D021_HEADING = "## D-021"                                   # the line that starts D-021's entry starts with it
BLOB_SHA = re.compile(r"[0-9a-f]{40}")                      # a blob id as git ls-tree and git hash-object print it
MAX_LISTED = 10                                             # M5: problem examples kept per kind
SNAPSHOT_MARKER = ".lofi-snapshot"                          # scripts/snapshot.py's marker (train.SNAPSHOT_MARKER)
# M12's breakdown, recorded and never gated: the mask record's timings (lofi_rl.rl.masked.new_step_record). They
# overlap: walk_s holds nfa_s, and old_s holds the old pass's unmasked_s and selfcheck_s, so they do not sum to step_s
MASK_TIMING = ("walk_s", "nfa_s", "old_s", "ref_s", "loss_s", "unmasked_s", "selfcheck_s")
# The self-check's flags are diagnostics, not D-021 gates: D-021 registers no bar for the self-check, and these never
# change its verdict. 0.05 nats is D-020's IS bar, for scale; 1.0 is a whole nat at one position.
SELFCHECK_MEAN_ABS_DIFF = 0.05
SELFCHECK_MAX_ABS_DIFF = 1.0


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
        # D-020's reading for TRL's own trainer; under the masked trainer the unchecked record cuts the other way
        caveats.append(VLLM_CAVEAT_MASKED if cfg.trainer_logprobs is not None else VLLM_CAVEAT)
    if not configured:
        says = "not D-020's reading: it is read in a GRPO smoke with the grammar and prompt v2"
    elif not grammar_shown:
        says = ("not D-020's reading: this smoke did not show the grammar applied in both launches (in_language "
                "failed), so it decides nothing (D-020, Validity)")
        warnings.append("IS readout: the in-language check did not show the grammar applied in both launches, so the "
                        "0.05 rule decides nothing")
    elif cfg.trainer_logprobs is not None:      # D-021 registered this reading as its M9; its verdict decides
        says = (f"{'exceeds' if exceeds else 'at most'} 0.05 under the masked trainer: D-021 Part A reads it as M9 "
                "(smoke.json d021), and D-021's verdict, not D-020's rule, says what follows")
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


# ------------------------------------------------------------------------------------------ D-021 Part A
D021_RULE = ("invalid: a validity gate (M5-M8, M10's first clause) did not pass, or a measured gate (M9, M10's ratio, "
             "M11) could not be decided, so the launch measured nothing; measured_miss: every validity gate passed "
             "and a measured gate missed; pass: every validity and measured gate passed. M12 is estimate-only: it "
             "decides the run-1 estimate, never this verdict (D-021, Two kinds of miss)")
D021_FOLLOWS = {
    "pass": ("M5-M11 all pass: D-021 supersedes D-020's IS-half verdict, and the grammar is on for run 1 with the "
             "masked trainer. Next, in D-021's order: D-020's official Phase-1 gate on the frozen palette"),
    "measured_miss": ("a measured miss on a valid launch gets at most one re-run, with the one fix D-021 names "
                      "(trainer_logprobs grammar_masked_fp32: the masked log-softmax in fp32 in all three passes), "
                      "which applies only when M8 holds, as it did here; a measured miss after that re-run leaves "
                      "D-020's verdict in force: the grammar is off for run 1, and Part B goes to Shresth"),
    "invalid": ("an invalid launch measured nothing: its defect is fixed, the fix is recorded in D-021, and the smoke "
                "runs again; a launch invalid twice goes to Shresth with the records"),
}
FP32_RERUN = ("this launch ran D-021's named fix (grammar_masked_fp32): if it is the one re-run after a measured "
              "miss, D-020's verdict stands, the grammar is off for run 1, and Part B goes to Shresth")
SELFCHECK_NOTE = ("the self-check disagrees with the path that ran, or could not compare it (d021.selfcheck_flags, "
                  "diagnostics.selfcheck); D-021's fp32 fix presupposes a numerical miss (D-021, Two kinds of miss: "
                  "\"It applies only when M8 holds, so the mask matched and the miss is numerical\"): read the "
                  "self-check before the re-run is spent")


def _g(x: Any) -> str:
    """A number as the log prints it (6 significant digits); anything else as it is."""
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def _gate(kind: str, ok: bool | None, value: Any, bar: str, source: str, reads: str,
          reasons: list[str] | None = None, **details: Any) -> dict[str, Any]:
    """One D-021 gate: ok is True (passes), False (misses) or None (cannot be decided: counts as not passing)."""
    reasons = list(reasons or [])
    more = len(reasons) - 30
    return {"kind": kind, "pass": ok, "value": value, "bar": bar, "source": source, "reads": reads,
            "reasons": reasons[:30] + ([f"... and {more} more"] if more > 0 else []), **details}


def _verdict(rec: Any) -> str | None:
    return rec.get("verdict") if isinstance(rec, dict) else None


def _stamp(text: Any) -> dict[str, str] | None:
    """make sync's .sync-stamp as run.json records it (`commit=… dirty=… diff=… at=…`), as a mapping."""
    if not isinstance(text, str) or "commit=" not in text:
        return None
    return dict(tok.split("=", 1) for tok in text.split() if "=" in tok)


def _count(v: Any) -> int | None:
    """A count a mask record holds as an int, or as the list of what it counts (violations as (row, pos, token))."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if v >= 0 else None
    return len(v) if isinstance(v, (list, tuple)) else None


def _meta(data: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    return (data.get(name) or {}).get("meta") or {}


def _iter_steps(data: dict[str, dict[str, Any]], want: dict[str, list[int]]):
    """(launch, step, its mask record or None, TRL's logs) for every step each launch should have trained."""
    for name, steps in want.items():
        by_step = {r.get("step"): r for r in (data.get(name) or {}).get("steps") or []}
        for s in steps:
            r = by_step.get(s) or {}
            mask = r.get("mask")
            yield name, s, mask if isinstance(mask, dict) else None, r.get("trl") or {}


def git_blob_sha(data: bytes) -> str:
    """The blob id `git hash-object` prints for a file's bytes: sha1 of b"blob <size>\\0" + the bytes. The repo has no
    .gitattributes, so no filter changes the bytes git hashes."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _registration(root: Path) -> tuple[dict[str, Any], list[str]]:
    """M5's registration half, read from `root`, the tree this module runs from (on the cluster the job's read-only
    code snapshot, which both launches imported): PREREG_LIST can be read and names at least one file; each of its
    lines but blanks and # comments (check_frozen.sh skips the same) is "<git blob sha>  <path>", the path relative and
    inside `root`, and the path's bytes hash to that blob; docs/DECISIONS.md has a line starting "## D-021". The
    stamps' dirty=0 says the synced tree was the commit; this says the tree that ran holds the frozen path and the
    entry. A file that cannot be read, a malformed line or another blob is a reason (fail closed)."""
    rec: dict[str, Any] = {"root": str(root), "snapshot": (root / SNAPSHOT_MARKER).is_file(), "list": PREREG_LIST,
                           "list_sha256": None, "files": 0, "matched": 0, "changed": [], "unreadable": [],
                           "malformed": [], "decisions": DECISIONS, "d021_heading": None}
    reasons: list[str] = []
    counts = {"changed": 0, "unreadable": 0, "malformed": 0}

    def note(kind: str, item: dict[str, Any]) -> None:
        counts[kind] += 1
        if len(rec[kind]) < MAX_LISTED:
            rec[kind].append(item)

    lines: list[str] | None = None
    try:
        raw = (root / PREREG_LIST).read_bytes()
        rec["list_sha256"] = hashlib.sha256(raw).hexdigest()
        lines = raw.decode("utf-8").splitlines()
    except OSError as err:
        reasons.append(f"cannot read the frozen path list {PREREG_LIST} ({err.strerror or err})")
    except UnicodeDecodeError as err:
        reasons.append(f"the frozen path list {PREREG_LIST} is not UTF-8 text ({err})")
    for n, line in enumerate(lines or [], 1):
        fields = line.split(None, 1)
        if not fields or fields[0].startswith("#"):
            continue
        path = fields[1].strip() if len(fields) == 2 else ""
        rel = PurePosixPath(path)
        if not BLOB_SHA.fullmatch(fields[0]) or not path or rel.is_absolute() or ".." in rel.parts:
            note("malformed", {"line": n, "text": line[:200]})
            continue
        rec["files"] += 1
        try:
            got = git_blob_sha((root / path).read_bytes())
        except OSError as err:
            note("unreadable", {"line": n, "path": path, "error": str(err.strerror or err)})
            continue
        if got == fields[0]:
            rec["matched"] += 1
        else:
            note("changed", {"line": n, "path": path, "frozen": fields[0], "now": got})
    rec.update({f"n_{k}": v for k, v in counts.items()})
    if counts["malformed"]:
        first = rec["malformed"][0]
        reasons.append(f"{counts['malformed']} lines of {PREREG_LIST} are not '<git blob sha>  <path>' with a "
                       f"relative path (first: line {first['line']}: {first['text']!r})")
    if lines is not None and rec["files"] == 0:
        reasons.append(f"{PREREG_LIST} names no file")
    if counts["unreadable"]:
        first = rec["unreadable"][0]
        reasons.append(f"{counts['unreadable']} frozen files cannot be read in {root} (first: {first['path']}: "
                       f"{first['error']})")
    if counts["changed"]:
        first = rec["changed"][0]
        reasons.append(f"{counts['changed']} frozen files are not their frozen blob in {root} (first: {first['path']}: "
                       f"frozen {first['frozen'][:12]}, now {first['now'][:12]})")
    try:
        text = (root / DECISIONS).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        reasons.append(f"cannot read {DECISIONS} ({getattr(err, 'strerror', None) or err})")
    else:
        rec["d021_heading"] = next((l[:200] for l in text.splitlines() if l.startswith(D021_HEADING)), None)
        if rec["d021_heading"] is None:
            reasons.append(f"{DECISIONS} has no line starting {D021_HEADING!r}: D-021 is not in the tree that ran")
    return rec, reasons


M5_BAR = (f"both launches' stamps: one commit, dirty=0; in the tree that ran, every file {PREREG_LIST} names at its "
          f"frozen blob, and {DECISIONS} holds D-021")
M5_SOURCE = f"run.json stamp (a, b); {PREREG_LIST} and {DECISIONS} under config.ROOT (the job's snapshot)"


def _m5(data: dict[str, dict[str, Any]], root: Path | None = None) -> dict[str, Any]:
    """M5 (D-021: "this entry and the frozen path are committed and synced before the submit. Both launches' stamps
    show that commit with dirty=0"): both launches' stamps show one commit with dirty=0 (the code tree each imported is
    recorded beside), and the registration half holds (_registration) in `root`, by default config.ROOT as this module
    reads it when it runs. It never raises (main() has no guard around check()): an error is a reason, and M5 fails."""
    try:
        stamps, reasons = {}, []
        for n in ("a", "b"):
            meta = _meta(data, n)
            s = _stamp(meta.get("stamp"))
            stamps[n] = None if s is None else {"commit": s.get("commit"), "dirty": s.get("dirty")}
            if s is None:
                reasons.append(f"launch {n}: run.json has no .sync-stamp ({meta.get('stamp')!r})")
            elif s.get("dirty") != "0":
                reasons.append(f"launch {n}: its stamp says dirty={s.get('dirty')}, not 0")
        commits = sorted({str(s["commit"]) for s in stamps.values()
                          if s and s.get("commit") not in (None, "", "none")})
        if len(commits) != 1 and not reasons:
            reasons.append(f"the launches' stamps name {commits or 'no commit'}, not one commit")
        code = {n: _meta(data, n).get("code") for n in ("a", "b")}
        reg, why = _registration(ROOT if root is None else root)
        reasons += why
        reads = (f"commit {commits[0] if len(commits) == 1 else commits}, dirty "
                 + "/".join(str((stamps[n] or {}).get("dirty")) for n in ("a", "b"))
                 + f"; frozen path: {reg['matched']} of {reg['files']} files at their blob (list sha256 "
                 + f"{str(reg['list_sha256'])[:12]}), D-021 {'in' if reg['d021_heading'] else 'NOT in'} {DECISIONS}")
        return _gate("validity", not reasons, stamps, M5_BAR, M5_SOURCE, reads, reasons, code=code, registration=reg)
    except Exception as err:  # noqa: BLE001: M5 fails closed rather than stop the report
        why = f"M5 could not be read ({type(err).__name__}: {err})"[:300]
        return _gate("validity", False, None, M5_BAR, M5_SOURCE, why, [why], code=None, registration=None)


def _m6(rep: dict[str, Any]) -> dict[str, Any]:
    """M6: GRPO_SMOKE_OK; stop out_of_language 0, other.n 0, a stop completion in each launch; xgrammar pinned and
    applied in both launches."""
    reasons = []
    if not rep.get("ok"):
        reasons.append(f"the smoke failed (GRPO_SMOKE_FAILED; first: {str((rep.get('failures') or ['?'])[0])[:200]})")
    lang = rep.get("in_language") or {}
    stop_out, other = (lang.get("stop") or {}).get("out_of_language"), (lang.get("other") or {}).get("n")
    if stop_out != 0:
        reasons.append(f"{stop_out} stop completions are out of the regex's language, not 0")
    if other != 0:
        reasons.append(f"other.n is {other}, not 0")
    stops = {n: (((lang.get("launches") or {}).get(n) or {}).get("stop") or {}).get("n") for n in ("a", "b")}
    for n, k in stops.items():
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            reasons.append(f"launch {n}: {k} stop completions, not at least 1")
    want = {"requested": BACKEND, "pin_accepted": True, "backend": BACKEND, "applied": True}
    backend = {n: {k: (((rep.get("backend") or {}).get(n)) or {}).get(k) for k in want} for n in ("a", "b")}
    for n, b in backend.items():
        if b != want:
            reasons.append(f"launch {n}: backend {b}, not {want}")
    value = {"smoke_ok": bool(rep.get("ok")), "stop_out_of_language": stop_out, "other_n": other, "stop_n": stops,
             "backend": backend}
    reads = (f"smoke {'ok' if rep.get('ok') else 'FAILED'}, stop out of language {stop_out}, other {other}, stop "
             f"completions a {stops['a']} b {stops['b']}, xgrammar applied a {backend['a']['applied']} "
             f"b {backend['b']['applied']}")
    return _gate("validity", not reasons, value, "GRPO_SMOKE_OK; stop out_of_language 0, other.n 0, at least one stop "
                 "completion a launch; xgrammar pinned and applied in both launches",
                 "this report: ok, in_language, backend", reads, reasons)


def _m7(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """M7: T1-T5 read "match" in both launches' run.json, T5's run-time half (lm_head) too; V1 and V2 recorded."""
    per: dict[str, Any] = {}
    reasons = []
    for n in ("a", "b"):
        meta = _meta(data, n)
        checks = meta.get("trl_source_checks")
        if not isinstance(checks, dict):
            per[n] = None
            reasons.append(f"launch {n}: run.json has no trl_source_checks")
            continue
        names = list(dict.fromkeys(list(MASK_GATING) + list(checks.get("gating") or [])))
        verdicts = {t: _verdict(checks.get(t)) for t in names}
        runtime = _verdict(meta.get("lm_head"))
        per[n] = {"gating": verdicts, "T5_run_time": runtime, "verdict": checks.get("verdict"),
                  "path": checks.get("path"), "recorded": {v: _verdict(checks.get(v)) for v in ("V1", "V2")}}
        bad = [f"{t} {v}" for t, v in verdicts.items() if v != "match"]
        if runtime != "match":
            bad.append(f"T5 at run time (lm_head) {runtime}")
        if checks.get("verdict") != "match" and not bad:
            bad.append(f"source_checks verdict {checks.get('verdict')!r}")
        if bad:
            reasons.append(f"launch {n}: " + ", ".join(bad))
    reads = "; ".join(f"{n}: " + ("no record" if p is None else
                                  " ".join(f"{t} {v}" for t, v in p["gating"].items()) + f" T5rt {p['T5_run_time']}")
                      for n, p in per.items())
    return _gate("validity", not reasons, per, "T1-T5 match in both launches (T5's run-time half too); V1, V2 recorded",
                 "run.json trl_source_checks and lm_head (a, b)", reads, reasons)


def _one_vocabulary(name: str, masked: dict[str, Any]) -> list[str]:
    """M8's "one vocabulary and one set of stop ids": the run's, and those the NFA's and the replay's records name
    (the NFA's must name them; the replay's are compared where it names them)."""
    want = {"vocab_size": masked.get("vocab_size"), "stop_ids": sorted(masked.get("stop_ids") or [])}
    if isinstance(want["vocab_size"], bool) or not isinstance(want["vocab_size"], int) or not want["stop_ids"]:
        return [f"launch {name}: run.json masked names no vocabulary size or stop ids ({want})"]
    out = []
    for part in ("nfa", "replay"):
        rec = masked.get(part)
        if not isinstance(rec, dict):
            out.append(f"launch {name}: run.json masked has no {part} record")
            continue
        for key, val in want.items():
            if key not in rec:
                if part == "nfa":
                    out.append(f"launch {name}: the NFA's record names no {key}")
                continue
            got = sorted(rec[key]) if key == "stop_ids" and isinstance(rec[key], (list, tuple)) else rec[key]
            if got != val:
                out.append(f"launch {name}: the {part}'s {key} {rec[key]!r} is not the run's {val!r}")
    return out


def _m8(data: dict[str, dict[str, Any]], want: dict[str, list[int]], cfg: RunConfig) -> dict[str, Any]:
    """M8: no sampled token outside the mask over every completion (the replay walked all 32 of each step), 32 masked
    rows in each of the old, ref and loss passes a step, no position where the NFA's allowed set differs from
    xgrammar's, finite logged loss/KL/entropy, and the path named."""
    from lofi_rl.rl.masked import UNREAD            # trl_sampling_mask before the replay's hook ran (a masked config
    mode, path = cfg.trainer_logprobs, cfg.mask_path  # loads only with lofi_rl.rl.masked importable)
    rows_want = cfg.prompts_per_step * cfg.trainer["num_generations"]
    reasons: list[str] = []
    tot = {"violations": 0, "nfa_mismatch": 0, "nfa_positions": 0, "completions": 0, "positions": 0,
           "steps_with_record": 0, "trl_sampling_mask": 0}
    paths, vocab = {}, {}
    for n in want:
        meta = _meta(data, n)
        err = str(meta.get("error") or "")
        if "MaskViolation" in err:
            reasons.append(f"launch {n} stopped on a sampled token outside the mask: {err[:300]}")
        if meta.get("trainer_logprobs") != mode:
            reasons.append(f"launch {n}: run.json trainer_logprobs {meta.get('trainer_logprobs')!r}, not {mode!r}")
        masked = meta.get("masked") if isinstance(meta.get("masked"), dict) else {}
        paths[n] = masked.get("path")
        if masked.get("path") != path:
            reasons.append(f"launch {n}: run.json masked.path {masked.get('path')!r}, not {path!r} (the path {mode} "
                           "runs)")
        vocab[n] = {"vocab_size": masked.get("vocab_size"), "stop_ids": masked.get("stop_ids")}
        reasons += _one_vocabulary(n, masked)
    if len({json.dumps(v, sort_keys=True) for v in vocab.values()}) > 1:
        reasons.append(f"the launches name two vocabularies or stop-id sets: {vocab}")
    for n, s, mk, trl in _iter_steps(data, want):
        bad = [k for k in FINITE_LOGS if _number(trl.get(k)) is None]
        if bad:
            reasons.append(f"launch {n} step {s}: logged {', '.join(bad)} not finite "
                           f"({ {k: trl.get(k) for k in bad} })")
        if mk is None:
            reasons.append(f"launch {n} step {s}: no mask record")
            continue
        tot["steps_with_record"] += 1
        v, nm, npos = _count(mk.get("violations")), _count(mk.get("nfa_mismatch")), _count(mk.get("nfa_positions"))
        done, walked = _count(mk.get("completions")), _count(mk.get("positions"))
        if done != rows_want or not walked or mk.get("trl_sampling_mask") == UNREAD:
            reasons.append(f"launch {n} step {s}: the replay checked {mk.get('completions')!r} completions "
                           f"({mk.get('positions')!r} tokens), not the step's {rows_want}")
        tot["completions"] += done or 0
        tot["positions"] += walked or 0
        if v != 0:
            reasons.append(f"launch {n} step {s}: violations {mk.get('violations')!r}, not 0 (first "
                           f"{(mk.get('violation_examples') or [None])[:1]})")
        if nm != 0:
            reasons.append(f"launch {n} step {s}: nfa_mismatch {mk.get('nfa_mismatch')!r}, not 0 (first "
                           f"{(mk.get('nfa_examples') or [None])[:1]})")
        if not npos:
            reasons.append(f"launch {n} step {s}: the NFA compared {mk.get('nfa_positions')!r} positions, so the "
                           "independent check did not run")
        tot["violations"] += v or 0
        tot["nfa_mismatch"] += nm or 0
        tot["nfa_positions"] += npos or 0
        rows = mk.get("rows") if isinstance(mk.get("rows"), dict) else {}
        got = {p: rows.get(p) for p in ("old", "ref", "loss")}
        if got != dict.fromkeys(got, rows_want):
            reasons.append(f"launch {n} step {s}: masked rows {got}, not {rows_want} in each pass")
        if mk.get("path") != path:
            reasons.append(f"launch {n} step {s}: the step ran path {mk.get('path')!r}, not {path!r}")
        if mk.get("trl_sampling_mask") not in (None, UNREAD):        # TRL's own mask, ANDed in (expected none)
            tot["trl_sampling_mask"] += 1
    steps = sum(len(v) for v in want.values())
    value = {**tot, "steps": steps, "rows_per_pass": rows_want, "path": paths, "vocabulary": vocab}
    reads = (f"{tot['violations']} sampled tokens outside the mask over {tot['completions']} completions "
             f"({tot['positions']} tokens), {tot['nfa_mismatch']} NFA mismatches over {tot['nfa_positions']} "
             f"positions, mask records at {tot['steps_with_record']} of {steps} steps, path "
             f"{'/'.join(str(paths.get(n)) for n in want)}")
    return _gate("validity", not reasons, value, f"0 violations over every completion ({rows_want} a step); "
                 f"{rows_want} old, {rows_want} ref and {rows_want} loss rows a step; 0 NFA mismatches over one "
                 "vocabulary and one stop-id set; finite loss, KL and entropy; run.json names the path",
                 "run.json masked (a, b); steps.jsonl mask and trl", reads, reasons)


def _m9(rep: dict[str, Any], steps: int) -> dict[str, Any]:
    """M9: the token-weighted mean, over every step of both launches, of TRL's IS metric (source verdict match) is at
    most 0.05: this report's is_ratio, read only when every step has its value and its tokens."""
    isr = rep.get("is_ratio") or {}
    rows = isr.get("steps") or []
    value = isr.get("token_weighted_mean")
    reasons = []
    if isr.get("value_source") != f"trl:{IS_METRIC}":
        reasons.append(f"the IS readout is unavailable ({isr.get('reason')})")
    unread = [(r.get("run"), r.get("step")) for r in rows if r.get("value") is None or r.get("tokens") is None]
    if unread:
        reasons.append(f"steps {unread} have no logged value or no token count, and M9 reads every step")
    if len(rows) != steps:
        reasons.append(f"the readout has {len(rows)} steps, not {steps}")
    if value is None and not reasons:
        reasons.append("no token-weighted mean")
    ok = None if reasons else value <= IS_THRESHOLD
    reads = (f"token-weighted {_g(value)} over {len(rows)} steps (mean of steps {_g(isr.get('mean_of_steps'))}); "
             f"bar <= {IS_THRESHOLD}")
    return _gate("measured", ok, value, f"<= {IS_THRESHOLD}", f"is_ratio.token_weighted_mean (TRL's {IS_METRIC}; "
                 f"source verdicts {isr.get('verdicts')})", reads, reasons, mean_of_steps=isr.get("mean_of_steps"),
                 by_launch=isr.get("by_launch"))


def _zero_weight(mk: dict[str, Any]) -> bool:
    """A step whose old pass had no valid position: every completion was truncated, so mask_truncated_completions
    zeroed them all, and the trainer left neglogz None and added 0 tokens (masked.MaskedPasses.logps). Its weight in
    the token-weighted -log Z is 0, as it is in M9. Only that exact record (ints where it counts, no non-finite
    difference) weighs 0; an absent or ill-typed one stays missing."""
    nonfinite = mk.get("neglogz_nonfinite")
    return (mk.get("neglogz") is None
            and type(mk.get("neglogz_tokens")) is int and mk["neglogz_tokens"] == 0
            and type(mk.get("tokens")) is int and mk["tokens"] == 0
            and (nonfinite is None or (type(nonfinite) is int and nonfinite == 0)))


def _m10(data: dict[str, dict[str, Any]], want: dict[str, list[int]],
         m9: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """M10, both clauses: the trainer's token-weighted -log Z (each step's mask/neglogz weighted by its
    neglogz_tokens) is at least 0.05 (validity), and M9 is at most 0.25 times it (measured). A step with no valid
    old-pass token weighs 0 and is listed (zero_weight); a step with tokens but no value is missing."""
    num = den = 0.0
    missing, zero = [], []
    for n, s, mk, _ in _iter_steps(data, want):
        mk = mk or {}
        if _zero_weight(mk):
            zero.append((n, s))
            continue
        z, t = _number(mk.get("neglogz")), mk.get("neglogz_tokens")
        if z is None or isinstance(t, bool) or not isinstance(t, int) or t < 0:
            missing.append((n, s))
            continue
        num += z * t
        den += t
    reasons = [f"steps {missing} have no neglogz or neglogz_tokens"] if missing else []
    if not missing and den <= 0:
        reasons.append("no valid completion token in any step's old pass")
    value = num / den if not reasons else None
    first = _gate("validity", None if value is None else value >= NEGLOGZ_MIN, value, f">= {NEGLOGZ_MIN}",
                  "steps.jsonl mask: neglogz weighted by neglogz_tokens, every step of both launches (a step with no "
                  "valid old-pass token weighs 0)",
                  f"token-weighted -log Z {_g(value)} over {int(den)} tokens"
                  + (f" ({len(zero)} zero-weight steps: no valid old-pass token)" if zero else "")
                  + f"; bar >= {NEGLOGZ_MIN}", reasons,
                  clause="first (validity)",
                  tokens=int(den), zero_weight=zero)
    m9v = m9["value"] if m9["pass"] is not None else None
    why = ([] if m9v is not None else ["M9 could not be decided"]) + ([] if value is not None else
                                                                        ["mask/neglogz could not be read"])
    bound = None if value is None else RATIO_OF_NEGLOGZ * value
    ratio = _gate("measured", None if why else m9v <= bound, None if why or value == 0 else m9v / value,
                  f"M9 <= {RATIO_OF_NEGLOGZ} x neglogz" + ("" if bound is None else f" = {bound:.6g}"),
                  "M9's value and M10's -log Z",
                  f"M9 {_g(m9v)} against {RATIO_OF_NEGLOGZ} x -log Z = {_g(bound)}", why, clause="the ratio (measured)",
                  m9=m9v, neglogz=value,
                  settles=("D-020's caveat (is_ratio's vLLM caveat): had vLLM taken its log-probs before the mask, "
                           "M9 would equal -log Z and this would fail (D-021, M10)"))
    return first, ratio


def _weights(w: Any, n: int) -> list[float] | None:
    if not isinstance(w, (list, tuple)) or len(w) != n:
        return None
    vals = [_number(x) for x in w]
    return None if any(v is None for v in vals) else vals


def _mean_share(vals: list[float]) -> tuple[float | None, float | None]:
    """(mean, largest / sum) of a step's IS weights; the share is None when they sum to 0."""
    if not vals:
        return None, None
    total = sum(vals)
    return total / len(vals), (max(vals) / total if total > 0 else None)


def _m11(data: dict[str, dict[str, Any]], want: dict[str, list[int]], cfg: RunConfig) -> dict[str, Any]:
    """M11: at every step the mean of the 32 per-sequence IS weights TRL applied (mask record seq_weights) is at
    least 0.3, and the largest is at most 0.25 of their sum. The non-truncated rows' values are reported beside it,
    not gated. TRL's logged ratio mean and max stand in for a step without seq_weights only when both launches' T3
    reads that mean as over sequences (share = max / (32 x mean), D-021's table)."""
    n_want = cfg.prompts_per_step * cfg.trainer["num_generations"]
    t3 = {n: (_meta(data, n).get("trl_source_checks") or {}).get("T3") for n in want}
    over_sequences = bool(t3) and all(isinstance(r, dict) and r.get("ratio_mean_over") == "sequences"
                                      for r in t3.values())
    rows, reasons = [], []
    for n, s, mk, trl in _iter_steps(data, want):
        row: dict[str, Any] = {"run": n, "step": s}
        w = _weights((mk or {}).get("seq_weights"), n_want)
        stand_mean, stand_max = _number(trl.get(IS_WEIGHT_MEAN)), _number(trl.get(IS_WEIGHT_MAX))
        if w is not None:
            mean, share = _mean_share(w)
            row.update(source="mask.seq_weights", n=len(w), mean=mean, max=max(w), sum=sum(w), max_share=share)
            valid = (mk or {}).get("seq_valid")
            if isinstance(valid, (list, tuple)) and len(valid) == len(w):
                vm, vs = _mean_share([x for x, v in zip(w, valid) if v is True])
                row["non_truncated"] = {"n": sum(1 for v in valid if v is True), "mean": vm, "max_share": vs}
            raw = _weights((mk or {}).get("seq_weights_raw"), n_want)
            if raw is not None:
                row["zeroed"] = sum(1 for a, b in zip(w, raw) if a == 0 and b != 0)   # TRL's mask set them to 0
        elif over_sequences and stand_mean is not None and stand_max is not None:
            row.update(source=f"trl:{IS_WEIGHT_MEAN} and max (T3: a mean over sequences)", n=n_want, mean=stand_mean,
                       max=stand_max, max_share=stand_max / (n_want * stand_mean) if stand_mean > 0 else None)
        else:
            row.update(source=None, mean=None, max_share=None, **{"pass": None})
            reasons.append(f"launch {n} step {s}: no {n_want} finite seq_weights in its mask record"
                           + ("" if over_sequences else ", and T3 does not read TRL's ratio mean as over sequences, "
                              "so TRL's metric cannot stand in"))
            rows.append(row)
            continue
        ok = row["mean"] >= WEIGHT_MEAN_MIN and row["max_share"] is not None and row["max_share"] <= WEIGHT_SHARE_MAX
        row["pass"] = ok
        if not ok:
            reasons.append(f"launch {n} step {s}: mean {row['mean']:.4g}, largest share {row['max_share']}")
        rows.append(row)
    decided = [r for r in rows if r["pass"] is not None]
    ok = None if not rows or len(decided) != len(rows) else all(r["pass"] for r in rows)
    shares = [r["max_share"] for r in decided if r["max_share"] is not None]
    nt = [r["non_truncated"] for r in rows if "non_truncated" in r]
    value = {"steps": len(rows), "passing": sum(1 for r in rows if r["pass"] is True),
             "min_mean": min((r["mean"] for r in decided), default=None), "max_share": max(shares, default=None),
             "zero_sum_steps": sum(1 for r in decided if r["max_share"] is None),
             "non_truncated": {"min_mean": min((x["mean"] for x in nt if x["mean"] is not None), default=None),
                               "max_share": max((x["max_share"] for x in nt if x["max_share"] is not None),
                                                default=None)}}
    reads = (f"{value['passing']} of {len(rows)} steps pass (bar: mean >= {WEIGHT_MEAN_MIN}, largest share <= "
             f"{WEIGHT_SHARE_MAX}); lowest mean {_g(value['min_mean'])}, largest share {_g(value['max_share'])}")
    return _gate("measured", ok, value, f"every step: mean of the {n_want} weights >= {WEIGHT_MEAN_MIN}, largest <= "
                 f"{WEIGHT_SHARE_MAX} x their sum", "steps.jsonl mask.seq_weights (TRL's applied per-sequence "
                 "weights)", reads, reasons, steps=rows)


def _timing_breakdown(rep: dict[str, Any], data: dict[str, dict[str, Any]],
                      want: dict[str, list[int]]) -> dict[str, Any]:
    """M12's breakdown, never gated: per mask timing field the mean over the steps that record it, and the mean of
    step_s - unmasked_s - selfcheck_s over the steps that record all three. The old pass's unmasked forward (for M10's
    -log Z) and the launch's self-check are smoke-only costs that D-021's "the mask adds under 1 s a step" did not
    count, so the net mean is the one the lofi-grpo estimate can be redone from."""
    step_s = {(r.get("run"), r.get("step")): _number(r.get("step_s")) for r in rep.get("steps") or []}
    per: dict[str, list[float]] = {f: [] for f in MASK_TIMING}
    net: list[float] = []
    for n, s, mk, _ in _iter_steps(data, want):
        t = (mk or {}).get("timing")
        got = {f: _number(t.get(f)) if isinstance(t, dict) else None for f in MASK_TIMING}
        for f, v in got.items():
            if v is not None:
                per[f].append(v)
        whole, unmasked, selfcheck = step_s.get((n, s)), got["unmasked_s"], got["selfcheck_s"]
        if whole is not None and unmasked is not None and selfcheck is not None:
            net.append(whole - unmasked - selfcheck)
    return {"what": "steps.jsonl mask.timing, the mean over steps of each field; they overlap (walk_s holds nfa_s, "
                    "old_s holds unmasked_s and selfcheck_s), so they do not sum to step_s",
            "mask_timing": {f: {"mean_s": statistics.fmean(v) if v else None, "steps": len(v)} for f, v in per.items()},
            "step_s_net": {"of": "step_s - unmasked_s - selfcheck_s", "mean_s": statistics.fmean(net) if net else None,
                           "steps": len(net)}}


def _m12(rep: dict[str, Any], data: dict[str, dict[str, Any]], want: dict[str, list[int]]) -> dict[str, Any]:
    """M12 (estimate-only): the mean step_s over every step of both launches is at most 43.3 s. The mask's timings
    and step_s net of the smoke-only costs are recorded beside it (breakdown) and decide nothing."""
    steps = sum(len(v) for v in want.values())
    vals = [r["step_s"] for r in rep.get("steps") or [] if _number(r.get("step_s")) is not None]
    reasons = [] if len(vals) == steps else [f"{len(vals)} of {steps} steps have a step time"]
    mean = statistics.fmean(vals) if vals else None
    ok = None if reasons or mean is None else mean <= STEP_S_MAX
    return _gate("estimate-only", ok, mean, f"<= {STEP_S_MAX} s (1.15 x smoke 13657200's 37.63 s)",
                 "steps.jsonl step_s (a, b)", f"mean step {_g(mean)} s over {len(vals)} steps; bar <= {STEP_S_MAX} s",
                 reasons,
                 decides=("only the run-1 estimate: over the bar, the lofi-grpo estimate is redone before the ask; "
                          "the job's SU is read from sacct"),
                 breakdown=_timing_breakdown(rep, data, want))


def _agreement(pairs: list[tuple[str, int, Any, Any]]) -> dict[str, Any]:
    """How many (launch, step, the reward's count, the trainer's count) pairs agree, with the first that do not."""
    known = [(n, s, a, b) for n, s, a, b in pairs if isinstance(a, int) and isinstance(b, int)]
    return {"steps_compared": len(known), "agree": sum(1 for *_, a, b in known if a == b),
            "differ": [{"run": n, "step": s, "reward": a, "trainer": b} for n, s, a, b in known if a != b][:10]}


def _diagnostics(rep: dict[str, Any], data: dict[str, dict[str, Any]], want: dict[str, list[int]]) -> dict[str, Any]:
    """Recorded beside the gates, never gated: per step the reward's token counts against the trainer's (the old
    pass's valid positions; the tokens the replay walked), the first self-check of each launch, and the mask's shape
    (forced share, largest allowed set, cache misses, non-finite masked log-probs)."""
    valid = {(r.get("run"), r.get("step")): r.get("tokens") for r in (rep.get("is_ratio") or {}).get("steps") or []}
    every = {(n, k + 1): t for n in want for k, (t, _) in
             _step_tokens((data.get(n) or {}).get("rewards") or [], drop_truncated=False).items() if isinstance(k, int)}
    pairs_valid, pairs_walked = [], []
    selfcheck: dict[str, Any] = {}
    forced, k_max, misses, nonfinite = [], [], 0, 0
    for n, s, mk, _ in _iter_steps(data, want):
        if mk is None:
            continue
        pairs_valid.append((n, s, valid.get((n, s)), mk.get("tokens")))
        pairs_walked.append((n, s, every.get((n, s)), mk.get("positions")))
        nf = mk.get("nonfinite")
        nonfinite += sum(_count(v) or 0 for v in nf.values()) if isinstance(nf, dict) else (_count(nf) or 0)
        if mk.get("selfcheck") is not None and n not in selfcheck:
            own = mk["selfcheck"] if isinstance(mk["selfcheck"], dict) else {"record": mk["selfcheck"]}
            # `step` as this report counts steps (steps.jsonl); the record's own is the global step it ran at
            selfcheck[n] = {**own, "step": s, "global_step": own.get("step")}
        if _number(mk.get("forced_share")) is not None:
            forced.append(mk["forced_share"])
        if _number(mk.get("k_max")) is not None:
            k_max.append(mk["k_max"])
        misses += _count(mk.get("cache_miss")) or 0
    return {"token_counts": {"what": "rewards.jsonl completion tokens (truncated ones out) against the old pass's "
                                     "valid positions (mask.tokens)", **_agreement(pairs_valid)},
            "replay_positions": {"what": "rewards.jsonl completion tokens (every completion) against the tokens the "
                                         "replay checked (mask.positions)", **_agreement(pairs_walked)},
            "selfcheck": selfcheck, "forced_share_mean_of_steps": statistics.fmean(forced) if forced else None,
            "k_max": max(k_max, default=None), "cache_miss": misses, "nonfinite_masked_logps": nonfinite}


def _selfcheck_flags(selfcheck: Any, want: Iterable[str]) -> list[str]:
    """One flag for each launch in `want` whose first self-check (diagnostics.selfcheck) is absent or failed, lacks
    its masked or unmasked comparison, has non-finite or no finite differences, or puts the path that ran and the
    other path further apart than SELFCHECK_MEAN_ABS_DIFF (mean |diff|) or SELFCHECK_MAX_ABS_DIFF (max |diff|).
    Never gated, and the verdict never reads it: the self-check is the one run-time comparison of the path that ran
    with an independent computation on the same rows, so a flag says a miss may not be the numerical one that D-021's
    fp32 fix presupposes."""
    out = []
    for n in want:
        rec = selfcheck.get(n) if isinstance(selfcheck, dict) else None
        if not isinstance(rec, dict):
            out.append(f"launch {n}: no self-check record (the launch's first old pass with a valid row runs it)")
            continue
        bad = []
        if rec.get("error"):
            bad.append(f"it failed ({str(rec['error'])[:200]})")
        for part in ("masked", "unmasked"):
            st = rec.get(part)
            if not isinstance(st, dict):
                bad.append(f"no {part} comparison")
                continue
            if _count(st.get("nonfinite")) != 0:
                bad.append(f"{part}: non-finite differences {st.get('nonfinite')!r}, not 0")
            mean, top = _number(st.get("mean_abs_diff")), _number(st.get("max_abs_diff"))
            if mean is None or top is None:
                bad.append(f"{part}: no finite difference compared (mean |diff| {st.get('mean_abs_diff')!r}, "
                           f"max {st.get('max_abs_diff')!r})")
                continue
            if mean > SELFCHECK_MEAN_ABS_DIFF:
                bad.append(f"{part}: mean |diff| {mean:.6g} > {SELFCHECK_MEAN_ABS_DIFF}")
            if top > SELFCHECK_MAX_ABS_DIFF:
                bad.append(f"{part}: max |diff| {top:.6g} > {SELFCHECK_MAX_ABS_DIFF}")
        if bad:
            out.append(f"launch {n} step {rec.get('step')} ({rec.get('ran')} against {rec.get('against')}): "
                       + "; ".join(bad))
    return out


def d021_block(rep: dict[str, Any], data: dict[str, dict[str, Any]], cfg: RunConfig,
               want: dict[str, list[int]]) -> dict[str, Any]:
    """smoke.json's `d021` (module docstring): every gate, the verdict and what D-021 says follows it, the diagnostics
    and the self-check's flags. `rep` is this report with its base verdict (ok) already set; `want` the steps each
    launch should have trained."""
    steps = sum(len(v) for v in want.values())
    gates = {"M5": _m5(data), "M6": _m6(rep), "M7": _m7(data), "M8": _m8(data, want, cfg), "M9": _m9(rep, steps)}
    gates["M10"], gates["M10_ratio"] = _m10(data, want, gates["M9"])
    gates["M11"] = _m11(data, want, cfg)
    gates["M12"] = _m12(rep, data, want)
    not_valid = [g for g in D021_VALIDITY if gates[g]["pass"] is not True]
    undecided = [g for g in D021_MEASURED if gates[g]["pass"] is None]
    missed = [g for g in D021_MEASURED if gates[g]["pass"] is False]
    verdict = "invalid" if not_valid or undecided else "measured_miss" if missed else "pass"
    order = [g for g in gates if g in D021_VALIDITY + D021_MEASURED]
    diagnostics = _diagnostics(rep, data, want)
    flags = _selfcheck_flags(diagnostics["selfcheck"], want)     # read after the verdict, which never sees them
    follows = D021_FOLLOWS[verdict]
    if verdict == "measured_miss" and cfg.trainer_logprobs == "grammar_masked_fp32":
        follows = f"{follows}; {FP32_RERUN}"
    if verdict == "measured_miss" and flags:
        follows = f"{follows}; {SELFCHECK_NOTE}"
    return {"entry": "docs/DECISIONS.md D-021, Part A: the pre-registered gates M5-M12", "mode": cfg.trainer_logprobs,
            "path": cfg.mask_path, "kinds": {"validity": list(D021_VALIDITY), "measured": list(D021_MEASURED),
                                             "estimate-only": list(D021_ESTIMATE)},
            "gates": gates, "verdict": verdict,
            "not_passing": [g for g in order if gates[g]["pass"] is not True],
            "undecided": [g for g in gates if gates[g]["pass"] is None],
            "estimate": {"M12": gates["M12"]["pass"]}, "rule": D021_RULE, "follows": follows,
            "selfcheck_flags": flags, "diagnostics": diagnostics}


EXIT_MEANING = {"124": "timed out", "137": "killed after the timeout", "3": "no policy env",
                "2": "config error, or a masked launch refused by a check: run.json's status says which"}


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
    if cfg.trainer_logprobs is not None:    # D-021's verdict, beside the smoke's own (M6 reads `ok`), never inside it
        want = {"a": list(range(1, max_steps + 1)), "b": list(range(resume + 1, max_steps + 1))}
        rep["d021"] = d021_block(rep, data, cfg, want) if stage == "final" else None
        for flag in (rep["d021"] or {}).get("selfcheck_flags") or []:      # rep["warnings"] is this list
            warnings.append(f"D-021 self-check (a diagnostic, not a gate): {flag}")
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
    d021 = rep.get("d021")
    for gid, g in ((d021 or {}).get("gates") or {}).items():
        word = {True: "pass", None: "undecided"}.get(g["pass"], "fail" if g["kind"] == "validity" else "miss")
        print(f"[smoke] D-021 {gid} ({g['kind']}): {word}: {g['reads']}"
              + (f"; {g['reasons'][0]}" if g["reasons"] and g["pass"] is not True else ""))
    if d021:
        print(f"[smoke] D-021 follows: {d021['follows']}")
    for w in rep["warnings"]:
        print(f"[smoke] warning: {w}")
    print(f"[smoke] report: {out / 'smoke.json'}")
    if d021:                                # just before the last line, which stays GRPO_SMOKE_OK or _FAILED
        print(f"D021_PART_A {d021['verdict']}" + (f": {', '.join(d021['not_passing'])}" if d021["not_passing"] else ""))
    tag = "FIRST_RUN" if a.stage == "first" else "GRPO_SMOKE"
    if rep["ok"]:
        print(f"{tag}_OK")
        return 0
    print(f"{tag}_FAILED: " + "; ".join(rep["failures"]))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
