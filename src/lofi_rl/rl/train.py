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
  it was checked to compute mean |old_per_token_logps - sampling_per_token_logps| over completion_mask
  (lofi_rl.rl.trl_probe), so smoke_check reads TRL's per-step value as the IS readout only when the source was
  checked to match. That is TRL's half of D-020's definition: whether vLLM's reported log-probs include the grammar
  mask is UNVERIFIED (run.json's vllm_logprobs records the vLLM lines that say, checked false);
- checkpoints every save_steps in --run-dir; --resume-from continues from one (its global step restarts the
  curriculum at the right weights even if TRL passes no trainer_state);
- with a scorer ($LOFI_SCORER_SOCK), the run's scorer pin (scorer_pin_path): the versions and live terms its first
  link's scorer reported, which every later link must find again (lofi_rl.scoring.client.read_pin);
- a masked config (`trainer_logprobs`, D-021 Part A; docs/GRPO.md §8), in this order, all before the model loads:
  1. trl_probe.source_checks reads the installed TRL, vLLM and PEFT sources against the resolved GRPOConfig
     (run.json trl_source_checks). A gating check (T1-T5) that is not "match", or a T2 path the mode cannot run under
     (config.MASK_T2_PATHS), refuses the launch: status "refused", exit 2 (D-021: they fail closed with exit 2);
  2. the mask's two halves over one vocabulary and one set of stop ids: lofi_rl.rl.masked.XgrammarReplay (xgrammar,
     built as vLLM 0.32's backend builds it) and lofi_rl.grammar_mask.GrammarMask (the independent NFA), from the
     snapshot's tokenizer, its config.json vocab_size and the EOS set below (vLLM's all_stop_token_ids);
  3. the trainer class, masked.make_trainer_class(GRPOTrainer), which must override the three methods the mask
     needs; GRPOTrainer is then built from it with mask_setup=, and T5's run-time half (trl_probe.lm_head_record on
     the built PEFT model) refuses the launch the same way unless lm_head is not a LoRA target.
  run.json records `masked` (mode, path, regex sha, vocab size, stop ids, the replay's and the NFA's records) and
  `lm_head`; steps.jsonl carries each step's mask record under "mask" (StepLog).
<summary>/run.json records the settings, versions, GPU, prompt token lengths, dropped optional fields, the code tree
it imports (`code`: $LOFI_CODE and whether it is a snapshot) and the outcome. Before the GPU work starts the run checks
that every prompt fits max_prompt_length and the vLLM context, so nothing is silently cut.

--dry-run (laptop or cluster) validates the config, builds the prompts and prints the resolved settings of
every profile as JSON, and scores tests/fixtures/loops/example.ls through the reward when lofi_rl.verify
exists: the symbolic reward, with the copy penalty and the language count as the config sets them and the scorer
off whatever $LOFI_SCORER_SOCK says (the report records whether it is set; the scorer is never contacted). It also
records the prompt version, its template sha and every row's example id, and where the installed TRL (if any) defines
its IS metric, read as text. A masked config adds `trainer_logprobs` and `masked`: the path its mode runs and
trl_probe.source_checks against the settings every profile shares, with what a run would refuse on (on the laptop
nothing is installed to read, so nothing matches). It never imports torch, TRL, vLLM, transformers, PEFT or datasets
(tests/test_rl_train.py).
"""
from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import hashlib
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
from lofi_rl.rl.config import (MASK_GATING, MASK_T2_PATHS, PROMPT_MODULES, ConfigError, RunConfig, detect_profile,
                               fit_fields, load_config, resolve, structured_summary)
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
SNAPSHOT_MARKER = ".lofi-snapshot"      # scripts/snapshot.py marks a job's code snapshot with it
# the GRPOTrainer methods the masked subclass overrides (build spec §2 D): the replay check of every sampled id, the
# step's record, and the log-probs of the old, reference and loss passes
MASK_METHODS = ("_generate", "_generate_and_score_completions", "_get_per_token_logps_and_entropies")
_HARD_EXIT = False


class Refused(ConfigError):
    """A masked launch that may not train: a source check did not match, the trainer class lacks the mask, or lm_head
    is a LoRA target (D-021 Part A: they fail closed with exit 2). run.json says status "refused"."""


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


def code_record(root: Path = ROOT) -> dict[str, Any]:
    """The code tree this process imports (D-021 M5): $LOFI_CODE (scripts/job_env.sh: the job's code snapshot), this
    module's tree, and whether each holds scripts/snapshot.py's marker, so a run from the live mirror shows."""
    env = os.environ.get("LOFI_CODE") or None
    same = None
    if env is not None:
        try:
            same = Path(env).resolve() == root.resolve()
        except OSError:
            same = False
    return {"lofi_code": env, "lofi_code_snapshot": None if env is None else (Path(env) / SNAPSHOT_MARKER).is_file(),
            "root": str(root), "root_is_lofi_code": same, "root_snapshot": (root / SNAPSHOT_MARKER).is_file()}


# ------------------------------------------------------------------------------------------ the masked trainer
def source_checks(args: Mapping[str, Any]) -> dict[str, Any]:
    """trl_probe.source_checks(args=...) (D-021 M7: T1-T5 gate, V1 and V2 are recorded). It never raises by contract;
    if it raises anyway, or is missing, the result is a record that does not match, so the launch is refused."""
    try:
        rec = trl_probe.source_checks(args=dict(args))
        if not isinstance(rec, dict):
            raise TypeError(f"returned {type(rec).__name__}, not a record")
        return rec
    except Exception as err:  # noqa: BLE001: a record that refuses the launch, never one that lets it train
        return {"verdict": "not_matched", "path": None, "gating": list(MASK_GATING), "failed": ["source_checks"],
                "error": f"trl_probe.source_checks: {type(err).__name__}: {err}"[:500]}


def mask_refusals(mode: str, checks: Mapping[str, Any]) -> list[str]:
    """Why a launch of `mode` may not train under these source checks; empty when it may. Every gating check (D-021's
    T1-T5, plus any the record names) must read "match", the record's verdict too, and T2's path must be one the mode
    runs under (config.MASK_T2_PATHS)."""
    out: list[str] = []
    if checks.get("error"):
        out.append(str(checks["error"])[:300])
    for name in dict.fromkeys(list(MASK_GATING) + list(checks.get("gating") or [])):
        rec = checks.get(name)
        verdict = rec.get("verdict") if isinstance(rec, Mapping) else None
        if verdict != "match":
            why = (rec.get("missing") or rec.get("error")) if isinstance(rec, Mapping) else None
            out.append(f"{name} {verdict or 'absent'}" + (f" ({str(why)[:200]})" if why else ""))
    if checks.get("verdict") != "match" and not out:
        out.append(f"source_checks verdict {checks.get('verdict')!r}")
    path = checks.get("path")
    if path not in MASK_T2_PATHS[mode]:
        out.append(f"T2 path {path!r}: trainer_logprobs {mode} runs only under {list(MASK_T2_PATHS[mode])}")
    return out


def shared_settings(cfg: RunConfig) -> dict[str, Any]:
    """The resolved trainer settings every GPU profile shares: the sampling, IS and KL settings the source checks'
    run-time halves read. The dry run runs on no particular card; a run passes its own GRPOConfig."""
    resolved = [resolve(cfg, name) for name in cfg.profiles]
    return {k: v for k, v in resolved[0].items() if all(k in r and r[k] == v for r in resolved[1:])}


def lm_head_check(model: Any) -> dict[str, Any]:
    """T5's run-time half on the PEFT model GRPOTrainer built (trl_probe.lm_head_record), as a record even when the
    call raises or is missing (verdict "error", which refuses the launch)."""
    try:
        rec = trl_probe.lm_head_record(model)
        if not isinstance(rec, dict):
            raise TypeError(f"returned {type(rec).__name__}, not a record")
        return rec
    except Exception as err:  # noqa: BLE001
        return {"verdict": "error", "error": f"trl_probe.lm_head_record: {type(err).__name__}: {err}"[:500]}


def check_trainer_class(cls: Any, base: type) -> None:
    """masked.make_trainer_class's class must be a subclass of TRL's GRPOTrainer that overrides every method the mask
    needs (MASK_METHODS); otherwise the launch would train without the mask, so it is refused."""
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise Refused(f"masked.make_trainer_class returned {cls!r}, not a subclass of {base.__name__}")
    inherited = [m for m in MASK_METHODS if getattr(cls, m, None) is getattr(base, m, None)]
    if inherited:
        raise Refused(f"{cls.__qualname__} inherits {inherited} from {base.__name__} unchanged: it would train "
                      "without the grammar's mask")


def snapshot_vocab_size(snap: Path | str) -> int:
    """`vocab_size` of the snapshot's config.json: what vLLM 0.32 builds its xgrammar TokenizerInfo with
    (model_config.get_vocab_size(); build spec §1, V2)."""
    p = Path(snap) / "config.json"
    try:
        v = json.loads(p.read_text())["vocab_size"]
    except (OSError, ValueError, KeyError) as err:
        raise ConfigError(f"{p}: no readable vocab_size ({err}); the mask needs the model's vocabulary size") from err
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ConfigError(f"{p}: vocab_size {v!r} is not a positive integer")
    return v


def build_mask(cfg: RunConfig, kw: Mapping[str, Any], snap: Path | str, tok: Any, eos: frozenset[int],
               base: type) -> tuple[Any, type, dict[str, Any]]:
    """The masked trainer's parts, before the model loads (module docstring, masked config 2-3): (MaskSetup, the
    trainer class, run.json's `masked`). Cluster only: xgrammar is imported by masked.XgrammarReplay. MaskSetup
    refuses a replay or an NFA built over another regex, vocabulary or stop-id set (D-021 M8)."""
    from lofi_rl.grammar_mask import GrammarMask
    from lofi_rl.rl.masked import MaskSetup, XgrammarReplay, make_trainer_class

    so = cfg.structured_outputs
    regex = kw.get(so["field"])                         # the very string GRPOConfig hands vLLM
    sha = hashlib.sha256(str(regex).encode("utf-8")).hexdigest()
    if regex != so["regex"] or sha != so["sha256"]:
        raise ConfigError(f"the resolved {so['field']} is not the config's regex ({sha[:16]} vs {so['sha256'][:16]})")
    snap = Path(snap)
    vocab = snapshot_vocab_size(snap)
    stop = tuple(sorted(eos))
    if not stop:
        raise ConfigError("no EOS id from the tokenizer or the generation config: the grammar has no stop ids")
    replay = XgrammarReplay(regex, tok, vocab_size=vocab, stop_ids=stop)
    nfa = GrammarMask.from_tokenizer_json(snap / "tokenizer.json", regex, vocab_size=vocab, stop_ids=stop)
    cls = make_trainer_class(base)
    check_trainer_class(cls, base)
    setup = MaskSetup(mode=cfg.trainer_logprobs, regex=regex, regex_sha256=sha, vocab_size=vocab, stop_ids=stop,
                      replay=replay, nfa=nfa, stats={})
    rec = {**_jsonable(setup.record()), "vocab_from": str(snap / "config.json"),
           "stop_ids_from": "the tokenizer's eos_token_id and the generation config's eos_token_id (vLLM's "
                            "all_stop_token_ids)",
           "tokenizer_json": str(snap / "tokenizer.json"), "trainer_class": f"{cls.__module__}.{cls.__qualname__}"}
    return setup, cls, rec


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


def _masked_report(cfg: RunConfig) -> dict[str, Any]:
    """The dry run's `masked`: the mode, the path it runs, the TRL forms it can run under, and the source checks
    against the settings every profile shares, with what a launch would refuse on. Nothing here imports torch, TRL,
    vLLM, PEFT, xgrammar or lofi_rl.rl.masked: the checks read source text (trl_probe)."""
    checks = source_checks(shared_settings(cfg))
    return {"mode": cfg.trainer_logprobs, "path_expected": cfg.mask_path,
            "t2_paths": list(MASK_T2_PATHS[cfg.trainer_logprobs]), "source_checks": checks,
            "source_checks_args": "the resolved settings every profile shares (a launch passes its GRPOConfig)",
            "refusals": mask_refusals(cfg.trainer_logprobs, checks),
            "mask": "built at run time from the snapshot's tokenizer.json and config.json vocab_size, with the EOS "
                    "ids: lofi_rl.rl.masked.XgrammarReplay and lofi_rl.grammar_mask.GrammarMask"}


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
        "trainer_logprobs": cfg.trainer_logprobs,
        "masked": None if cfg.trainer_logprobs is None else _masked_report(cfg),
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
    record for step k + 1 carries reward stats[k]. With `mask_stats` (a masked config: the MaskSetup's stats, keyed
    by the global step the masked trainer generated at) each step record also has "mask": mask_stats[k], at the same
    offset; without it the records have no such key."""

    def __init__(self, path: Path | str, reward_stats: Mapping[int, Any] | None = None,
                 clock: Callable[[], float] = time.time, mask_stats: Mapping[int, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stats = reward_stats if reward_stats is not None else {}
        self.mask = mask_stats
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
        rec = {"kind": "step", "step": global_step, "t": t, "step_s": dt, "trl": logs,
               "lofi": self.stats.get(global_step - 1)}
        if self.mask is not None:
            rec["mask"] = self.mask.get(global_step - 1)
        self._write(rec)


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
    """run.json's `versions` (and the mask preflight's): the installed distributions of the training stack, accelerate
    and xgrammar, the masked trainer's replay (lofi_rl.rl.masked.XgrammarReplay). xgrammar stays out of STACK, which
    main() also checks the environment against."""
    out = {}
    for name in STACK + ("accelerate", "xgrammar"):
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
        "trl_logp_metric": trl_probe.logp_metric_source(), "vllm_logprobs": trl_probe.vllm_logprobs_evidence(),
        "trainer_logprobs": cfg.trainer_logprobs, "code": code_record()}
    _write_json(summary_dir / "run.json", man)
    steps: StepLog | None = None
    pin: trl_probe.BackendPin | None = None
    trainer = None
    setup = None                                        # the masked trainer's MaskSetup (a masked config)
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
        trainer_cls, extra = GRPOTrainer, {}
        if cfg.trainer_logprobs is not None:                # D-021 Part A, all before the model loads (docstring)
            settings = args.to_dict()
            man["grpo_config"] = _jsonable(settings)
            man["trl_source_checks"] = checks = source_checks(settings)
            _write_json(summary_dir / "run.json", man)
            refusals = mask_refusals(cfg.trainer_logprobs, checks)
            if refusals:
                raise Refused("the source checks refuse the masked trainer: " + "; ".join(refusals))
            setup, trainer_cls, man["masked"] = build_mask(cfg, kw, snap, tok, eos, GRPOTrainer)
            extra = {"mask_setup": setup}
            _write_json(summary_dir / "run.json", man)
        model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16)
        model.config.use_cache = False                      # gradient checkpointing; vLLM generates
        spin = scorer_pin_path(run_dir, resume_from, scorer_pin)
        man["scorer_pin"] = str(spin)
        reward = make_reward(cfg.palette_path(), summary_dir / "rewards.jsonl",
                             max_completion_length=kw["max_completion_length"], eos_token_ids=eos,
                             start_step=start_step, scorer_pin=spin, copy=copy, language=cfg.structured_outputs)
        steps = StepLog(summary_dir / "steps.jsonl", reward_stats=reward.stats,
                        mask_stats=None if setup is None else setup.stats)
        lora = LoraConfig(r=cfg.lora["r"], lora_alpha=cfg.lora["alpha"], lora_dropout=cfg.lora["dropout"],
                          target_modules=cfg.lora["target_modules"], bias="none", task_type="CAUSAL_LM")
        if cfg.structured_outputs is not None:              # the engine TRL builds asks for xgrammar, as junk's does
            pin = trl_probe.BackendPin.for_vllm(STRUCTURED_BACKEND)
        trainer = trainer_cls(model=model, reward_funcs=[reward], args=args,
                              train_dataset=datasets.Dataset.from_list(rows), processing_class=tok,
                              peft_config=lora, callbacks=[_callback(TrainerCallback, steps)], **extra)
        if pin is not None:
            man["structured_outputs"]["backend"] = trl_probe.engine_record(trainer, pin)
        if setup is not None:                               # T5's run-time half, on the PEFT model just built
            man["lm_head"] = head = lm_head_check(trainer.model)
            _write_json(summary_dir / "run.json", man)
            if head.get("verdict") != "match":
                why = head.get("error") or head.get("missing") or ""
                raise Refused(f"T5 at run time: trl_probe.lm_head_record reads {head.get('verdict')!r}"
                              + (f" ({str(why)[:200]})" if why else "") + ": lm_head must not be a LoRA target")
        try:
            trainable, total = trainer.model.get_nb_trainable_parameters()
            man["params"] = {"trainable": int(trainable), "total": int(total)}
        except Exception as e:  # noqa: BLE001: informational only
            man["params"] = f"unavailable: {type(e).__name__}: {e}"
        man.update(status="training", grpo_config=_jsonable(args.to_dict()), init_s=round(time.time() - T_PROCESS, 1))
        _write_json(summary_dir / "run.json", man)

        out = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
        man.update(status="ok", final_global_step=int(trainer.state.global_step), train_metrics=_jsonable(out.metrics))
    except Refused as e:                                    # a masked launch stopped by a check: exit 2 (main)
        man.update(status="refused", error=f"{type(e).__name__}: {e}")
        raise
    except BaseException as e:
        man.update(status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-8000:])
        raise
    finally:
        if setup is not None and isinstance(man.get("masked"), dict):
            # the steps the trainer recorded, and the last record (a launch stopped by a MaskViolation never logs it)
            man["masked"]["stats_steps"] = sorted(setup.stats)
            man["masked"]["last_stats"] = _jsonable(setup.stats[max(setup.stats)]) if setup.stats else None
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
