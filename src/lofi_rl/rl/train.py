"""The GRPO trainer (docs/design/taste_pipeline.md §6; docs/GRPO.md §3-4).

    python -m lofi_rl.rl.train --config configs/grpo_smoke.yaml [--resume-from <ckpt>] [--dry-run]
        [--run-dir DIR] [--summary-dir DIR] [--gpu-profile a100-80gb|48gb] [--scorer-pin FILE]

A training run (policy env, one GPU) builds TRL 1.17.1's GRPOTrainer with:
- Qwen3-4B-Instruct-2507 at its pinned snapshot in the offline HF cache (kit.hf_cache.resolve_pinned), bf16;
- PEFT LoRA (lora: r 16, alpha 32, all-linear); with PEFT the KL reference is the adapter-off base;
- the trainer settings of the config, merged with the GPU profile its memory picks, and fitted to the installed
  GRPOConfig's fields before anything heavy loads (config.fit_fields);
- one dataset row per brief: brief.sample_briefs(seed, count, start) -> build_messages(brief, palette) of the
  config's prompt version (`prompt:` v1, the default, lofi_rl.prompt; v2 lofi_rl.prompt_v2, whose messages differ
  per energy x mood cell), with the brief (JSON) and a brief_id ("<seed>:<index>") as the columns the reward reads;
  a v2 row also carries example_id, the example its prompt shows;
- the interim symbolic reward (lofi_rl.rl.reward.make_reward), which logs every completion to
  <summary>/rewards.jsonl, and a callback that writes one line per optimizer step to <summary>/steps.jsonl
  (TRL's metrics: reward mean and std, completion lengths, KL; and the reward's sub-reward means). A config with
  `copy_penalty` adds P_copy (lofi_rl.rl.copy_penalty) against every example its prompt version can show and the
  pool; a constrained config (`structured_outputs`) counts the completions in the regex's language;
- a constrained config also pins the structured-outputs backend of the vLLM engine TRL builds in colocate mode
  (lofi_rl.rl.trl_probe.BackendPin: xgrammar, as junk.py pins its own) and reads it back from the engine it finds
  on the trainer; run.json records requested, pin_accepted, backend, source, applied under structured_outputs;
- run.json's trl_logp_metric: where the installed TRL computes sampling/sampling_logp_difference/mean and whether
  that is D-020's |log pi_vLLM - log pi_trainer| over completion tokens (lofi_rl.rl.trl_probe), so smoke_check
  reads TRL's per-step value as the IS readout only when the source was checked to match;
- checkpoints every save_steps in --run-dir; --resume-from continues from one (its global step restarts the
  curriculum at the right weights even if TRL passes no trainer_state);
- with a scorer ($LOFI_SCORER_SOCK), the run's scorer pin (scorer_pin_path): the versions and live terms its first
  link's scorer reported, which every later link must find again (lofi_rl.scoring.client.read_pin).
<summary>/run.json records the settings, versions, GPU, prompt token lengths, dropped optional fields and the
outcome. Before the GPU work starts the run checks that every prompt fits max_prompt_length and the vLLM
context, so nothing is silently cut.

--dry-run (laptop or cluster) validates the config, builds the prompts and prints the resolved settings of
every profile as JSON, and scores tests/fixtures/loops/example.ls through the reward when lofi_rl.verify
exists: the symbolic reward, with the copy penalty and the language count as the config sets them and the scorer
off whatever $LOFI_SCORER_SOCK says (the report records whether it is set; the scorer is never contacted). It also
records the prompt version, its template sha and every row's example id, and where the installed TRL (if any) defines
its IS metric, read as text. It never imports torch, TRL, vLLM, transformers, PEFT or datasets
(tests/test_rl_train.py).
"""
from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import signal
import socket
import sys
import tempfile
import time
import traceback
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Any

from lofi_rl.brief import sample_briefs
from lofi_rl.palette import Palette, load_palette
from lofi_rl.rl import curriculum, trl_probe
from lofi_rl.rl.config import (PROMPT_MODULES, ConfigError, RunConfig, detect_profile, fit_fields, load_config,
                               resolve, structured_summary)
from lofi_rl.rl.copy_penalty import CopyPenalty
from lofi_rl.rl.junk import STRUCTURED_BACKEND
from lofi_rl.rl.reward import SERVER_TERMS, make_reward
from lofi_rl.scoring.client import ENV_SOCK

T_PROCESS = time.time()
ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")
EXAMPLE = ROOT / "tests" / "fixtures" / "loops" / "example.ls"
STACK = ("torch", "transformers", "trl", "peft", "datasets", "vllm")      # the policy env's training stack
ROLES = ("system", "user", "assistant")
SCORER_PIN = "scorer_pin.json"
_HARD_EXIT = False


# ------------------------------------------------------------------------------------------ prompts
V1_EXAMPLE = "example"          # the id junk.py gives v1's one example (junk.example_id)


def prompt_module(version: str = "v1"):
    """The template module of a prompt version (config.PROMPT_MODULES: v1 lofi_rl.prompt, v2 lofi_rl.prompt_v2):
    PROMPT_VERSION, build_messages, template_sha256 (v2 also example_id). A module whose PROMPT_VERSION names another
    version is refused, so a row is never built from a template the config did not name."""
    if version not in PROMPT_MODULES:
        raise ConfigError(f"prompt version {version!r} is not one of {list(PROMPT_MODULES)}")
    name = PROMPT_MODULES[version]
    try:
        mod = importlib.import_module(name)
    except ImportError as err:
        raise ConfigError(f"{name} cannot be imported ({err}): the trainer builds the rows of prompt {version} with "
                          "its build_messages(brief, palette)") from err
    got = getattr(mod, "PROMPT_VERSION", version)
    if got != version:
        raise ConfigError(f"{name}.PROMPT_VERSION is {got!r}, not the config's prompt {version!r}")
    return mod


def _verify_module():
    try:
        return importlib.import_module("lofi_rl.verify")
    except ImportError:
        return None


def _check_messages(msgs: Any, where: str) -> list[dict[str, str]]:
    ok = (isinstance(msgs, list) and msgs and all(isinstance(m, Mapping) and m.get("role") in ROLES
                                                  and isinstance(m.get("content"), str) for m in msgs)
          and msgs[-1]["role"] == "user")
    if not ok:
        raise ConfigError(f"prompt.build_messages returned {str(msgs)[:300]!r} for {where}: it must be a list of "
                          f"{{role, content}} messages (roles {ROLES}, string content) ending with the user's")
    return [dict(m) for m in msgs]


def build_rows(cfg: RunConfig, palette: Palette, prompts: Any) -> list[dict[str, Any]]:
    """One dataset row per brief: the chat prompt, the brief as JSON, and its id '<seed>:<index>'. A prompt v2 row
    also has example_id, the example its prompt shows (prompt_v2.example_id); a v1 row has no such column (its one
    example is "example"), so v1's rows, and the reward's log of them, are what they were before v2."""
    shown = getattr(prompts, "example_id", None) if cfg.prompt != "v1" else None
    if cfg.prompt != "v1" and not callable(shown):
        raise ConfigError(f"{getattr(prompts, '__name__', prompts)} has no example_id(brief): a prompt {cfg.prompt} row "
                          "records the example its prompt shows")
    rows = []
    for i, b in enumerate(sample_briefs(cfg.brief_seed, cfg.brief_count, cfg.brief_start)):
        bid = f"{cfg.brief_seed}:{cfg.brief_start + i}"
        row = {"prompt": _check_messages(prompts.build_messages(b, palette), f"brief {bid}"),
               "brief": json.dumps(dataclasses.asdict(b), sort_keys=True), "brief_id": bid}
        if shown is not None:
            row["example_id"] = shown(b)
        rows.append(row)
    return rows


def row_example_ids(rows: list[dict[str, Any]]) -> list[str]:
    """The id of the example each row's prompt shows: its example_id column, else v1's one example."""
    return [r.get("example_id", V1_EXAMPLE) for r in rows]


def prompt_record(cfg: RunConfig, prompts: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What run.json and the dry run record of the prompt: the version, its module and template sha, and per row the
    brief id, the example its prompt shows and the sha256 of its messages (prompt.messages_sha256)."""
    template = getattr(prompts, "template_sha256", None)
    msg_sha = getattr(prompts, "messages_sha256", None)
    ids = row_example_ids(rows)
    return {"version": cfg.prompt, "module": getattr(prompts, "__name__", None),
            "module_version": getattr(prompts, "PROMPT_VERSION", None),
            "template_sha256": template() if callable(template) else None, "rows": len(rows),
            "example_ids": ids, "example_counts": dict(sorted(Counter(ids).items())),
            "brief_ids": [r["brief_id"] for r in rows],
            "messages_sha256": [msg_sha(r["prompt"]) for r in rows] if callable(msg_sha) else None}


def copy_penalty(cfg: RunConfig, palette: Palette, prompts: Any) -> CopyPenalty | None:
    """The run's P_copy when the config sets `copy_penalty`: every example the prompt module shows (v1 its
    example_text, v2 all 9 of its reference_texts) and the pool's .ls files; None otherwise."""
    if cfg.copy_penalty is None:
        return None
    return CopyPenalty.for_prompt(cfg.prompt, palette, cfg.pool_files(), root=ROOT, module=prompts)


def scorer_pin_path(run_dir: Path | str, resume_from: Path | str | None = None,
                    explicit: Path | str | None = None) -> Path:
    """The run's scorer pin (lofi_rl.rl.reward's scorer_pin): `explicit` (--scorer-pin: one path for every link of
    a chain), else <run-dir>/scorer_pin.json. A resumed link without one first copies the pin of the run it resumes
    from (<resume-from>/../scorer_pin.json), so with per-link run dirs (docs/GRPO.md: <run>/a, <run>/b) every link
    still pins what the first one pinned, and never the per-link summary dir. Nothing is read or written here
    without a pin to inherit; the reward writes the first one."""
    if explicit is not None:
        return Path(explicit)
    pin = Path(run_dir) / SCORER_PIN
    if resume_from is not None and not pin.exists():
        prev = Path(resume_from).parent / SCORER_PIN
        if prev.is_file() and prev.resolve() != pin.resolve():
            pin.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prev, pin)
    return pin


def checkpoint_step(ckpt: Path | str) -> int:
    """The global step a checkpoint was saved at, from its trainer_state.json."""
    p = Path(ckpt) / "trainer_state.json"
    try:
        step = json.loads(p.read_text())["global_step"]
    except (OSError, ValueError, KeyError) as err:
        raise ConfigError(f"{ckpt} is not a trainer checkpoint: no readable trainer_state.json ({err})") from err
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ConfigError(f"{p}: global_step {step!r} is not a step")
    return step


# ------------------------------------------------------------------------------------------ dry run
def _reward_check(cfg: RunConfig, row: Mapping[str, Any], copy: CopyPenalty | None = None) -> dict[str, Any] | str:
    """Score tests/fixtures/loops/example.ls under the first row's brief, through the real reward path. The
    symbolic path only: scorer=False, so an exported $LOFI_SCORER_SOCK never makes the quick check before any GPU
    work wait for a scorer (dry_run records whether it is set). The copy penalty and the language count are the
    run's: example.ls is v1's example with a `sax:` line, so with P_copy on it is a copy (J 1.0 with "example")."""
    verify = _verify_module()
    if verify is None:
        return "skipped: lofi_rl.verify is not importable here (another builder's module)"
    if not EXAMPLE.is_file():
        return f"skipped: {EXAMPLE} is missing"
    extra = {"example_id": [row["example_id"]]} if "example_id" in row else {}
    with tempfile.TemporaryDirectory() as d:
        reward = make_reward(cfg.palette_path(), Path(d) / "rewards.jsonl", verify=verify, strict=True, scorer=False,
                             copy=copy, language=cfg.structured_outputs)
        out = reward(prompts=[row["prompt"]], completions=[[{"role": "assistant", "content": EXAMPLE.read_text()}]],
                     brief=[row["brief"]], brief_id=[row["brief_id"]], **extra)
        rec = json.loads((Path(d) / "rewards.jsonl").read_text().splitlines()[0])
    keep = ("step", "gate", "G", "w", "r", "p_sax", "penalties", "p_group", "p_fix", "repairs", "positive", "penalty",
            "p_copy", "copy_j", "copy_nearest", "example_id", "finish", "in_language")
    return {"text": str(EXAMPLE.relative_to(ROOT)), "brief_id": row["brief_id"], "reward": out[0],
            **{k: rec[k] for k in keep if k in rec}}


def dry_run(cfg: RunConfig, out: IO[str] | None = None, prompts: Any = None) -> dict[str, Any]:
    """Validate, build the prompts, print every profile's resolved settings as JSON (module docstring)."""
    palette = load_palette(cfg.palette_path(), verify=False)
    prompts = prompts if prompts is not None else prompt_module(cfg.prompt)
    rows = build_rows(cfg, palette, prompts)
    copy = copy_penalty(cfg, palette, prompts)
    profiles = {}
    for name in cfg.profiles:
        kw = resolve(cfg, name)
        profiles[name] = {"min_gib": cfg.profiles[name]["min_gib"], "trainer": kw,
                          "completions_per_step": kw["per_device_train_batch_size"] * kw["gradient_accumulation_steps"]}
    chars = [sum(len(m["content"]) for m in r["prompt"]) for r in rows]
    mix = Counter(f"{json.loads(r['brief'])['energy']}/{json.loads(r['brief'])['mood']}" for r in rows)
    record = prompt_record(cfg, prompts, rows)
    report = {
        "dry_run": True,
        "config": {"path": cfg.path, "sha256": cfg.sha256, "run": cfg.run},
        "model": {"repo": cfg.model_repo, "revision": cfg.model_revision},
        "snapshot": "resolved on the cluster from the offline HF cache (kit.hf_cache.resolve_pinned)",
        "lora": dict(cfg.lora),
        "palette": {"path": cfg.palette, "id": palette.id, "sha256": palette.sha256},
        "briefs": {"seed": cfg.brief_seed, "count": cfg.brief_count, "start": cfg.brief_start},
        "prompts": {"rows": len(rows), "distinct": len({json.dumps(r["prompt"], sort_keys=True) for r in rows}),
                    **{k: record[k] for k in ("version", "module", "module_version", "template_sha256",
                                              "example_counts", "example_ids", "messages_sha256")},
                    "energy_x_mood": dict(sorted(mix.items())), "chars": {"min": min(chars), "max": max(chars)},
                    "preview": rows[0]["prompt"]},
        "profiles": profiles,
        "structured_outputs": structured_summary(cfg),
        "optional_fields": sorted(cfg.optional_fields),
        "unverified_trl_fields": sorted({k for p in profiles.values() for k in p["trainer"]}),
        "resume_from_step": cfg.resume_step,
        "reward": {"w0": dict(curriculum.W0), "w1": dict(curriculum.W1), "ramp_steps": curriculum.RAMP_STEPS,
                   "zero_in_smoke": list(SERVER_TERMS),
                   "scorer": {"env": ENV_SOCK, "sock": os.environ.get(ENV_SOCK) or None, "contacted": False},
                   "copy_penalty": None if copy is None else copy.summary(),
                   "in_language": None if cfg.structured_outputs is None
                   else {"regex_sha256": cfg.structured_outputs["sha256"], "by": "finish reason from the token ids"}},
        "backend_pin": None if cfg.structured_outputs is None
        else {"requested": STRUCTURED_BACKEND, "how": "lofi_rl.rl.trl_probe.BackendPin on vllm.LLM, at run time"},
        "trl_logp_metric": trl_probe.logp_metric_source(),
        "reward_check": _reward_check(cfg, rows[0], copy),
    }
    print(json.dumps(report, indent=1, ensure_ascii=False, default=str), file=out or sys.stdout)
    return report


# ------------------------------------------------------------------------------------------ step log
class StepLog:
    """What the trainer's callback writes: one JSONL record per optimizer step (kind "step": TRL's logged metrics
    under "trl", the reward's summary of that step's rollouts under "lofi", the step's wall time), one per
    checkpoint (kind "save") and TRL's end-of-training metrics (kind "train_end").

    TRL generates and scores a step's rollouts at global_step k and logs after the update, at k + 1, so the
    record for step k + 1 carries reward stats[k]."""

    def __init__(self, path: Path | str, reward_stats: Mapping[int, Any] | None = None,
                 clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stats = reward_stats if reward_stats is not None else {}
        self.clock = clock
        self.t_begin: float | None = None
        self._last: float | None = None
        self._ends: dict[int, tuple[float, float | None]] = {}

    def _write(self, rec: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def train_begin(self, global_step: int) -> None:
        self.t_begin = self._last = self.clock()

    def step_end(self, global_step: int) -> None:
        t = self.clock()
        self._ends[global_step] = (t, None if self._last is None else t - self._last)
        self._last = t

    def saved(self, global_step: int) -> None:
        self._write({"kind": "save", "step": global_step, "t": self.clock()})

    def log(self, global_step: int, logs: Mapping[str, Any] | None) -> None:
        logs = dict(logs or {})
        if "train_runtime" in logs:
            self._write({"kind": "train_end", "step": global_step, "t": self.clock(), "trl": logs})
            return
        t, dt = self._ends.get(global_step, (None, None))
        self._write({"kind": "step", "step": global_step, "t": t, "step_s": dt, "trl": logs,
                     "lofi": self.stats.get(global_step - 1)})


def _callback(base: type, log: StepLog):
    """A transformers TrainerCallback (`base`, imported by the training run) that feeds `log`."""
    class LofiStepLog(base):
        def on_train_begin(self, args, state, control, **kw):
            log.train_begin(state.global_step)

        def on_step_end(self, args, state, control, **kw):
            log.step_end(state.global_step)

        def on_log(self, args, state, control, logs=None, **kw):
            log.log(state.global_step, logs)

        def on_save(self, args, state, control, **kw):
            log.saved(state.global_step)
    return LofiStepLog()


# ------------------------------------------------------------------------------------------ the run
def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


def _versions() -> dict[str, str | None]:
    out = {}
    for name in STACK + ("accelerate",):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def default_run_dir(cfg: RunConfig) -> Path:
    base = Path(os.environ.get("REMOTE_SCR") or ROOT) / "runs"
    return base / f"{cfg.run}-{os.environ.get('SLURM_JOB_ID') or 'local'}"


def _prompt_tokens(tok: Any, msgs: list[dict[str, str]]) -> int:
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _check_lengths(lengths: list[int], kw: Mapping[str, Any]) -> None:
    longest = max(lengths)
    mpl, ctx = kw.get("max_prompt_length"), kw.get("vllm_max_model_length")
    if mpl is not None and longest > mpl:
        raise ConfigError(f"the longest prompt is {longest} tokens, over max_prompt_length {mpl}: TRL would cut it")
    if ctx is not None and longest + kw["max_completion_length"] > ctx:
        raise ConfigError(f"the longest prompt ({longest} tokens) + max_completion_length "
                          f"{kw['max_completion_length']} exceeds vllm_max_model_length {ctx}")


def _eos_ids(tok: Any, gen_cfg: Any) -> frozenset[int]:
    ids = set()
    if getattr(tok, "eos_token_id", None) is not None:
        ids.add(int(tok.eos_token_id))
    e = getattr(gen_cfg, "eos_token_id", None)
    if isinstance(e, int):
        ids.add(e)
    elif isinstance(e, (list, tuple)):
        ids.update(int(x) for x in e)
    return frozenset(ids)


def train(cfg: RunConfig, *, run_dir: Path, summary_dir: Path, resume_from: Path | None = None,
          profile: str | None = None, scorer_pin: Path | None = None) -> int:
    """One GRPO launch on one GPU (module docstring). Needs the policy env."""
    import datasets
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from kit.hf_cache import resolve_pinned

    summary_dir.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID")
    stamp = ROOT / ".sync-stamp"
    man: dict[str, Any] = {
        "status": "starting", "config": {"path": cfg.path, "sha256": cfg.sha256, "run": cfg.run},
        "model": {"repo": cfg.model_repo, "revision": cfg.model_revision}, "run_dir": str(run_dir),
        "summary_dir": str(summary_dir), "resume_from": str(resume_from) if resume_from else None,
        "job": job, "node": platform.node(), "stamp": stamp.read_text().strip() if stamp.is_file() else None,
        "versions": _versions(), "lora": dict(cfg.lora), "structured_outputs": structured_summary(cfg),
        "trl_logp_metric": trl_probe.logp_metric_source(), "vllm_logprobs": trl_probe.vllm_logprobs_evidence()}
    _write_json(summary_dir / "run.json", man)
    steps: StepLog | None = None
    pin: trl_probe.BackendPin | None = None
    trainer = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch sees no GPU")
        props = torch.cuda.get_device_properties(0)
        gib = props.total_memory / 2**30
        prof = profile or detect_profile(cfg, gib)
        kw, dropped = fit_fields(resolve(cfg, prof), GRPOConfig, cfg.optional_fields)
        named, unnamed = fit_fields({"run_name": f"{cfg.run}-{job or 'local'}"}, GRPOConfig, {"run_name"})
        dropped += unnamed
        start_step = checkpoint_step(resume_from) if resume_from else 0
        man.update(gpu={"name": props.name, "total_gib": round(gib, 2)}, profile=prof, trainer=kw,
                   verified_trl_fields=sorted(kw), dropped_optional_fields=dropped, start_step=start_step)
        _write_json(summary_dir / "run.json", man)

        snap = resolve_pinned(cfg.model_repo, cfg.model_revision, SNAPSHOT_FILES)
        palette = load_palette(cfg.palette_path(), verify=False)
        prompts = prompt_module(cfg.prompt)
        rows = build_rows(cfg, palette, prompts)
        copy = copy_penalty(cfg, palette, prompts)
        tok = AutoTokenizer.from_pretrained(snap)
        tok.padding_side = tok.truncation_side = "left"     # as TRL loads it; no prompt is cut (checked next)
        lengths = [_prompt_tokens(tok, r["prompt"]) for r in rows]
        _check_lengths(lengths, kw)
        eos = _eos_ids(tok, GenerationConfig.from_pretrained(snap))
        man.update(snapshot=snap, eos_token_ids=sorted(eos),
                   palette={"path": cfg.palette, "id": palette.id, "sha256": palette.sha256},
                   prompt={**prompt_record(cfg, prompts, rows),
                           "tokens": {"min": min(lengths), "mean": round(sum(lengths) / len(lengths), 1),
                                      "max": max(lengths)}},
                   copy_penalty=None if copy is None else copy.summary())
        _write_json(summary_dir / "run.json", man)

        args = GRPOConfig(output_dir=str(run_dir), **named, **kw)
        model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16)
        model.config.use_cache = False                      # gradient checkpointing; vLLM generates
        spin = scorer_pin_path(run_dir, resume_from, scorer_pin)
        man["scorer_pin"] = str(spin)
        reward = make_reward(cfg.palette_path(), summary_dir / "rewards.jsonl",
                             max_completion_length=kw["max_completion_length"], eos_token_ids=eos,
                             start_step=start_step, scorer_pin=spin, copy=copy, language=cfg.structured_outputs)
        steps = StepLog(summary_dir / "steps.jsonl", reward_stats=reward.stats)
        lora = LoraConfig(r=cfg.lora["r"], lora_alpha=cfg.lora["alpha"], lora_dropout=cfg.lora["dropout"],
                          target_modules=cfg.lora["target_modules"], bias="none", task_type="CAUSAL_LM")
        if cfg.structured_outputs is not None:              # the engine TRL builds asks for xgrammar, as junk's does
            pin = trl_probe.BackendPin.for_vllm(STRUCTURED_BACKEND)
        trainer = GRPOTrainer(model=model, reward_funcs=[reward], args=args,
                              train_dataset=datasets.Dataset.from_list(rows), processing_class=tok,
                              peft_config=lora, callbacks=[_callback(TrainerCallback, steps)])
        if pin is not None:
            man["structured_outputs"]["backend"] = trl_probe.engine_record(trainer, pin)
        try:
            trainable, total = trainer.model.get_nb_trainable_parameters()
            man["params"] = {"trainable": int(trainable), "total": int(total)}
        except Exception as e:  # noqa: BLE001: informational only
            man["params"] = f"unavailable: {type(e).__name__}: {e}"
        man.update(status="training", grpo_config=_jsonable(args.to_dict()), init_s=round(time.time() - T_PROCESS, 1))
        _write_json(summary_dir / "run.json", man)

        out = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
        man.update(status="ok", final_global_step=int(trainer.state.global_step), train_metrics=_jsonable(out.metrics))
    except BaseException as e:
        man.update(status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-8000:])
        raise
    finally:
        if pin is not None:
            backend = man["structured_outputs"].get("backend")
            if backend is None or backend.get("engine") is None:     # read again, or record the pin's state alone
                man["structured_outputs"]["backend"] = {
                    **trl_probe.engine_record(trainer, pin),
                    "read": ("after the launch: no engine on the trainer when it was built" if trainer is not None
                             else "no trainer: GRPOTrainer was not built")}
            else:                                   # the pin's own record goes on counting LLM() calls
                backend.update(calls=pin.record.get("calls"), pin_accepted=pin.record.get("pin_accepted"))
            pin.restore()
        if steps is not None and steps.t_begin is not None:
            man["startup_s"] = round(steps.t_begin - T_PROCESS, 1)
        man["wall_s"] = round(time.time() - T_PROCESS, 1)
        _write_json(summary_dir / "run.json", man)
    return 0


# ------------------------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    global _HARD_EXIT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume-from", type=Path, help="a checkpoint-<step> directory to continue from")
    ap.add_argument("--run-dir", type=Path, help="checkpoints (default $REMOTE_SCR/runs/<run>-<job>)")
    ap.add_argument("--summary-dir", type=Path, help="run.json, steps.jsonl, rewards.jsonl (default <run-dir>/summary)")
    ap.add_argument("--gpu-profile", help="a profile of the config (default: chosen by the GPU's memory)")
    ap.add_argument("--dry-run", action="store_true", help="validate, build the prompts, print the settings; no torch")
    ap.add_argument("--scorer-pin", type=Path, help="the scorer pin every link of the run shares (default "
                    "<run-dir>/scorer_pin.json, inherited from the run --resume-from names)")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.config)
        if a.gpu_profile is not None:
            resolve(cfg, a.gpu_profile)
        if a.dry_run:
            dry_run(cfg)
            return 0
        missing = [m for m in STACK if importlib.util.find_spec(m) is None]
        if missing:
            print(f"train: a training run needs the policy env (torch, TRL, vLLM, PEFT); {missing} cannot be "
                  "imported here. On the laptop, use --dry-run.", file=sys.stderr)
            return 3
        if a.resume_from is not None:
            checkpoint_step(a.resume_from)
        run_dir = a.run_dir or default_run_dir(cfg)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")    # vLLM colocate's process group: a free local port
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        faulthandler.enable(all_threads=True)
        faulthandler.register(signal.SIGTERM, all_threads=True, chain=True)   # `timeout` shows where it hung
        _HARD_EXIT = True
        return train(cfg, run_dir=run_dir, summary_dir=a.summary_dir or run_dir / "summary",
                     resume_from=a.resume_from, profile=a.gpu_profile, scorer_pin=a.scorer_pin)
    except ConfigError as e:
        print(f"train: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001: printed, then the hard exit below
        traceback.print_exc()
        code = 1
    if _HARD_EXIT:              # after a GPU run, skip native teardown (vLLM, NCCL), which can hang past the walltime
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    raise SystemExit(code)
