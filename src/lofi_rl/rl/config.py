"""GRPO run configs (configs/grpo_*.yaml; docs/design/taste_pipeline.md §6; docs/GRPO.md §3).

load_config() reads and validates one file strictly, the way palette.py reads a palette: unknown top-level keys
are errors, and the file's sha256 is recorded. The `trainer` section and every `profiles` entry hold TRL
GRPOConfig (and transformers TrainingArguments) field names, merged per GPU profile by resolve().

TRL is not on the laptop, so field names are checked against the real GRPOConfig only on the cluster:
fit_fields() refuses a name GRPOConfig lacks, before the model loads, unless the config lists it under
`optional_fields`, which are dropped and reported instead. Standard library and PyYAML only (a constrained config
also loads its palette and lofi_rl.grammar, both standard library and PyYAML).

Grammar-constrained decoding (D-016; docs/GRAMMAR.md): the top-level `structured_outputs: loopscript_regex` names a
regex builder of lofi_rl.grammar. load_config() builds the regex from the config's palette, and resolve() sets it as
the GRPOConfig field `vllm_structured_outputs_regex` in every profile, so the regex is never pasted into YAML and goes
through the same field check as every other name (the field exists in TRL 1.17.1's GRPOConfig:
runs/grpo-smoke-13632679/summary/a/run.json lists it in grpo_config, as None). Writing that field by hand in
`trainer` or a profile is refused.

The prompt version (D-020): the optional top-level `prompt: v1|v2` picks the template the trainer builds every row's
messages with, lofi_rl.prompt (v1, the default, so a config without the key is unchanged) or lofi_rl.prompt_v2 (each
brief shown the example of its energy and the next mood).

The copy penalty (taste_pipeline §5; lofi_rl.rl.copy_penalty): the optional top-level `copy_penalty: {pool: [...]}`
turns it on, against every example the run's prompt version can show plus the LoopScript files the pool lists (`.ls`
files, or directories of them; the seed pool, later). A config without the key has no copy penalty, so its rewards
are byte for byte what they were.

The masked trainer (D-021 Part A; docs/GRPO.md §8): the optional top-level `trainer_logprobs` names how the trainer
takes its log-probs under the grammar. Its values, and the log-prob path each runs (RunConfig.mask_path: support_ids,
the grammar's allowed set handed to TRL as its sampling_mask; fill, the override's own fp32 log-softmax with a finite
fill), are lofi_rl.rl.masked's TRAINER_LOGPROBS and path_for, imported (never copied) when a config sets the key, so a
config without it never imports that module; if the module cannot be imported, the config is refused. It is refused
without `structured_outputs`, since the mask is the grammar's. MASK_T2_PATHS names the forms of TRL's log-prob method
(trl_probe.source_checks' T2 path) each value can run under.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
TOP_REQUIRED = ("run", "model", "palette", "briefs", "prompts_per_step", "lora", "trainer", "profiles",
                "optional_fields")
TOP_OPTIONAL = ("smoke", "structured_outputs", "prompt", "copy_penalty", "trainer_logprobs")
STRUCTURED_BUILDERS = ("loopscript_regex",)                 # functions of lofi_rl.grammar: palette -> regex
# trainer_logprobs value -> the forms of TRL's _get_per_token_logps_and_entropies (source_checks' T2 path) it can run
# under (D-021 Part A; build spec §1). `grammar_masked` runs path support_ids, handing TRL the grammar's allowed set as
# its sampling_mask, so it needs TRL's own support form. `grammar_masked_fp32` (D-021's named fix for a measured miss)
# runs path fill, the override's own fp32 log-softmax, and calls TRL's method only unmasked: either matched form serves
MASK_T2_PATHS = MappingProxyType({"grammar_masked": ("support_ids",), "grammar_masked_fp32": ("support_ids", "fill")})
MASK_GATING = ("T1", "T2", "T3", "T4", "T5")    # D-021 M7: the source checks a masked launch needs "match" from
STRUCTURED_FIELD = "vllm_structured_outputs_regex"          # the TRL GRPOConfig field the regex goes into
PROMPT_MODULES = MappingProxyType({"v1": "lofi_rl.prompt", "v2": "lofi_rl.prompt_v2"})   # `prompt:` -> its template
DEFAULT_PROMPT = "v1"
COPY_KEYS = ("pool",)                                       # copy_penalty: {pool: [paths]}
POOL_SUFFIX = ".ls"
# The §6 settings every run must state (in `trainer` or in each profile), so none falls to a TRL default.
REQUIRED = ("learning_rate", "lr_scheduler_type", "warmup_steps", "max_steps", "num_generations", "num_iterations",
            "max_completion_length", "mask_truncated_completions", "beta", "loss_type", "scale_rewards",
            "temperature", "top_p", "top_k", "use_vllm", "vllm_mode", "vllm_gpu_memory_utilization",
            "per_device_train_batch_size", "gradient_accumulation_steps", "save_strategy", "save_steps",
            "logging_steps", "bf16", "gradient_checkpointing", "remove_unused_columns", "report_to", "seed")
FORBIDDEN = {"output_dir": "is set by the command line (--run-dir)",
             "resume_from_checkpoint": "is set by the command line (--resume-from)",
             "push_to_hub": "never: nothing leaves the cluster but pulled summaries",
             "hub_model_id": "never: nothing leaves the cluster but pulled summaries",
             "hub_token": "never: no credential on the cluster (CLAUDE.md §1)"}
# (kind, low, high, low is exclusive); a bound of None is open
NUMBERS = {
    "learning_rate": (float, 0, None, True), "beta": (float, 0, None, False), "temperature": (float, 0, None, True),
    "top_p": (float, 0, 1, True), "vllm_gpu_memory_utilization": (float, 0, 1, True),
    "warmup_steps": (int, 0, None, False), "max_steps": (int, 1, None, False), "num_generations": (int, 2, None, False),
    "num_iterations": (int, 1, None, False), "max_completion_length": (int, 1, None, False),
    "top_k": (int, -1, None, False), "per_device_train_batch_size": (int, 1, None, False),
    "gradient_accumulation_steps": (int, 1, None, False), "save_steps": (int, 1, None, False),
    "logging_steps": (int, 1, None, False), "seed": (int, 0, None, False), "max_prompt_length": (int, 1, None, False),
    "vllm_max_model_length": (int, 1, None, False),
}
BOOLS = ("mask_truncated_completions", "use_vllm", "bf16", "gradient_checkpointing", "remove_unused_columns",
         "save_only_model", "vllm_enable_sleep_mode", "log_completions")
CHOICES = {"scale_rewards": ("group", "batch", "none"), "vllm_mode": ("colocate", "server"),
           "save_strategy": ("steps",), "loss_type": ("grpo", "bnpo", "dr_grpo", "dapo"), "report_to": ("none",)}
_SHA = re.compile(r"[0-9a-f]{40}")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RunConfig:
    path: str
    sha256: str                                     # of the YAML bytes
    run: str
    model_repo: str
    model_revision: str
    palette: str                                    # repo-relative, or absolute
    brief_seed: int
    brief_count: int
    brief_start: int
    prompts_per_step: int
    lora: Mapping[str, Any]
    trainer: Mapping[str, Any]
    profiles: Mapping[str, Mapping[str, Any]]       # name -> {min_gib, trainer overrides}
    optional_fields: frozenset[str]
    resume_step: int | None                         # smoke.resume_from_step
    # builder, field, regex, sha256, chars when the config sets `structured_outputs`; None otherwise
    structured_outputs: Mapping[str, Any] | None = None
    prompt: str = DEFAULT_PROMPT                    # `prompt:`, a key of PROMPT_MODULES
    # {"pool": (paths as written, ...)} when the config sets `copy_penalty`; None (no copy penalty) otherwise
    copy_penalty: Mapping[str, Any] | None = None
    # `trainer_logprobs:`, a value of lofi_rl.rl.masked.TRAINER_LOGPROBS (the masked trainer); None: TRL's own
    trainer_logprobs: str | None = None

    @property
    def mask_path(self) -> str | None:
        """The log-prob path the masked trainer runs (lofi_rl.rl.masked.path_for), or None without `trainer_logprobs`.
        The module imported when the config loaded, since a config with the key loads only then."""
        if self.trainer_logprobs is None:
            return None
        from lofi_rl.rl.masked import path_for
        return path_for(self.trainer_logprobs)

    def palette_path(self, root: Path = ROOT) -> Path:
        p = Path(self.palette)
        return p if p.is_absolute() else root / p

    def pool_files(self, root: Path = ROOT) -> tuple[Path, ...]:
        """The copy penalty's pool as files: each listed .ls file, and the .ls files of each listed directory in
        name order. Empty without a copy penalty or with an empty pool."""
        out: list[Path] = []
        for entry in (self.copy_penalty or {}).get("pool", ()):
            p = Path(entry) if Path(entry).is_absolute() else root / entry
            out += sorted(p.glob(f"*{POOL_SUFFIX}")) if p.is_dir() else [p]
        return tuple(out)


def _int(v: Any, where: str, lo: int | None = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or (lo is not None and v < lo):
        raise ConfigError(f"{where} {v!r} is not an integer{'' if lo is None else f' >= {lo}'}")
    return v


def _check_value(k: str, v: Any, where: str) -> None:
    if k in NUMBERS:
        kind, lo, hi, open_lo = NUMBERS[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and not isinstance(v, int)):
            hint = " (YAML reads 1e-5 as text: write 1.0e-5)" if isinstance(v, str) and "e" in v.lower() else ""
            raise ConfigError(f"{where}.{k} {v!r} is not {'an integer' if kind is int else 'a number'}{hint}")
        if (lo is not None and (v <= lo if open_lo else v < lo)) or (hi is not None and v > hi):
            raise ConfigError(f"{where}.{k} {v!r} is outside {'(' if open_lo else '['}{lo}, {hi if hi is not None else 'inf'}]")
    elif k in BOOLS and not isinstance(v, bool):
        raise ConfigError(f"{where}.{k} {v!r} is not true or false")
    elif k in CHOICES and v not in CHOICES[k]:
        raise ConfigError(f"{where}.{k} {v!r} is not one of {CHOICES[k]}")


def load_config(path: Path | str) -> RunConfig:
    path = Path(path)
    data = path.read_bytes()
    try:
        doc = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as err:
        raise ConfigError(f"{path}: not a UTF-8 YAML document ({err})") from err
    if not isinstance(doc, dict):
        raise ConfigError(f"{path}: the top level must be a mapping")
    unknown = sorted(set(doc) - set(TOP_REQUIRED) - set(TOP_OPTIONAL))
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}")
    missing = [k for k in TOP_REQUIRED if k not in doc]
    if missing:
        raise ConfigError(f"{path}: missing {missing}")

    model = doc["model"]
    if not isinstance(model, dict) or set(model) != {"repo", "revision"}:
        raise ConfigError("model needs exactly repo and revision")
    if not isinstance(model["revision"], str) or not _SHA.fullmatch(model["revision"]):
        raise ConfigError(f"model.revision {model['revision']!r} is not a 40-hex commit sha (pin the snapshot)")
    briefs = doc["briefs"]
    if not isinstance(briefs, dict) or set(briefs) != {"seed", "count", "start"}:
        raise ConfigError("briefs needs exactly seed, count and start")
    for k in ("seed", "count", "start"):
        _int(briefs[k], f"briefs.{k}", 0 if k != "count" else 1)
    pps = _int(doc["prompts_per_step"], "prompts_per_step", 1)

    lora = doc["lora"]
    if not isinstance(lora, dict) or set(lora) != {"r", "alpha", "dropout", "target_modules"}:
        raise ConfigError("lora needs exactly r, alpha, dropout and target_modules")
    _int(lora["r"], "lora.r", 1)
    if isinstance(lora["alpha"], bool) or not isinstance(lora["alpha"], (int, float)) or lora["alpha"] <= 0:
        raise ConfigError(f"lora.alpha {lora['alpha']!r} is not a positive number")
    if isinstance(lora["dropout"], bool) or not isinstance(lora["dropout"], (int, float)) or not 0 <= lora["dropout"] < 1:
        raise ConfigError(f"lora.dropout {lora['dropout']!r} is not in [0, 1)")
    tm = lora["target_modules"]
    if not (tm == "all-linear" or isinstance(tm, list) and tm and all(isinstance(x, str) for x in tm)):
        raise ConfigError(f"lora.target_modules {tm!r} is not 'all-linear' or a list of module names")

    trainer, profiles = doc["trainer"], doc["profiles"]
    if not isinstance(trainer, dict):
        raise ConfigError("trainer must be a mapping of GRPOConfig fields")
    if not isinstance(profiles, dict) or not profiles:
        raise ConfigError("profiles needs at least one GPU profile")
    optional = doc["optional_fields"]
    if not isinstance(optional, list) or not all(isinstance(x, str) for x in optional):
        raise ConfigError("optional_fields must be a list of field names")
    clash = sorted(set(optional) & set(REQUIRED))
    if clash:
        raise ConfigError(f"optional_fields lists {clash}, which every run requires")
    builder = doc.get("structured_outputs")
    if builder is not None and (not isinstance(builder, str) or builder not in STRUCTURED_BUILDERS):
        raise ConfigError(f"structured_outputs {builder!r} is not one of {list(STRUCTURED_BUILDERS)} (a regex builder "
                          "of lofi_rl.grammar)")
    if STRUCTURED_FIELD in optional:
        raise ConfigError(f"optional_fields lists {STRUCTURED_FIELD}: a constrained run must not drop its regex "
                          "silently, and an unconstrained one never sets it")

    sections = [("trainer", trainer)] + [(f"profiles.{n}", p) for n, p in profiles.items() if isinstance(p, dict)]
    for k, why in FORBIDDEN.items():
        for where, sec in sections:
            if k in sec:
                raise ConfigError(f"{where}.{k} {why}")
    for where, sec in sections:
        if STRUCTURED_FIELD in sec:
            raise ConfigError(f"{where}.{STRUCTURED_FIELD}: write `structured_outputs: loopscript_regex` instead; the "
                              "regex is built from lofi_rl.grammar when the config loads, never pasted into it")
    for k, v in trainer.items():
        _check_value(k, v, "trainer")
    for name, prof in profiles.items():
        if not isinstance(prof, dict) or "min_gib" not in prof:
            raise ConfigError(f"profiles.{name} needs min_gib (the smallest GPU memory it runs on)")
        g = prof["min_gib"]
        if isinstance(g, bool) or not isinstance(g, (int, float)) or g <= 0:
            raise ConfigError(f"profiles.{name}.min_gib {g!r} is not a positive number")
        for k, v in prof.items():
            if k != "min_gib":
                _check_value(k, v, f"profiles.{name}")

    smoke = doc.get("smoke")
    resume = None
    if smoke is not None:
        if not isinstance(smoke, dict) or set(smoke) != {"resume_from_step"}:
            raise ConfigError("smoke needs exactly resume_from_step")
        resume = _int(smoke["resume_from_step"], "smoke.resume_from_step", 1)
    version = doc.get("prompt", DEFAULT_PROMPT)
    if not isinstance(version, str) or version not in PROMPT_MODULES:
        raise ConfigError(f"prompt {version!r} is not one of {list(PROMPT_MODULES)} (the template the rows are built "
                          "with: v1 lofi_rl.prompt, v2 lofi_rl.prompt_v2)")
    copy = _copy_penalty(doc["copy_penalty"]) if "copy_penalty" in doc else None
    logprobs = _trainer_logprobs(doc["trainer_logprobs"], builder) if "trainer_logprobs" in doc else None

    cfg = RunConfig(
        path=str(path), sha256=hashlib.sha256(data).hexdigest(), run=str(doc["run"]),
        model_repo=str(model["repo"]), model_revision=model["revision"], palette=str(doc["palette"]),
        brief_seed=briefs["seed"], brief_count=briefs["count"], brief_start=briefs["start"], prompts_per_step=pps,
        lora=MappingProxyType(dict(lora)), trainer=MappingProxyType(dict(trainer)),
        profiles=MappingProxyType({n: MappingProxyType(dict(p)) for n, p in profiles.items()}),
        optional_fields=frozenset(optional), resume_step=resume, prompt=version, copy_penalty=copy,
        trainer_logprobs=logprobs)
    if not cfg.palette_path().is_file():
        raise ConfigError(f"palette {cfg.palette} does not exist")
    _check_pool(cfg)
    if builder is not None:
        cfg = dataclasses.replace(cfg, structured_outputs=_structured(builder, cfg.palette_path()))
    for name in cfg.profiles:
        resolve(cfg, name)                          # every profile complete and consistent
    t = cfg.trainer
    if t["max_steps"] % t["save_steps"]:
        raise ConfigError(f"trainer.save_steps {t['save_steps']} does not divide max_steps {t['max_steps']}: "
                          "the last step would not be saved")
    if resume is not None and not (resume < t["max_steps"] and resume % t["save_steps"] == 0):
        raise ConfigError(f"smoke.resume_from_step {resume} must be a saved step (a multiple of save_steps "
                          f"{t['save_steps']}) before max_steps {t['max_steps']}")
    need = t["max_steps"] * pps
    if cfg.brief_count < need:
        raise ConfigError(f"briefs.count {cfg.brief_count} < max_steps {t['max_steps']} x prompts_per_step {pps} = "
                          f"{need}: the dataset would run out inside the run")
    return cfg


def _copy_penalty(block: Any) -> Mapping[str, Any]:
    """`copy_penalty: {pool: [...]}`: exactly the key pool, a list of paths (an empty list: the prompt's examples
    only). The thresholds are taste_pipeline §5's and live in lofi_rl.rl.copy_penalty, never in a config."""
    if not isinstance(block, dict) or set(block) != set(COPY_KEYS):
        raise ConfigError(f"copy_penalty needs exactly {list(COPY_KEYS)} (a list of LoopScript files or directories; "
                          "[] for the prompt's examples only); the thresholds are taste_pipeline §5's")
    pool = block["pool"]
    if not isinstance(pool, list) or not all(isinstance(p, str) and p for p in pool):
        raise ConfigError(f"copy_penalty.pool {pool!r} is not a list of paths")
    if len(set(pool)) != len(pool):
        raise ConfigError(f"copy_penalty.pool {pool!r} lists a path twice")
    return MappingProxyType({"pool": tuple(pool)})


def _trainer_logprobs(value: Any, builder: str | None) -> str:
    """`trainer_logprobs: <value>`: a value of lofi_rl.rl.masked.TRAINER_LOGPROBS, with `structured_outputs` set (the
    mask is the grammar's). The tuple is imported here, only for a config that sets the key; a module that cannot be
    imported refuses the config, so no run takes a mode nobody defined."""
    if builder is None:
        raise ConfigError(f"trainer_logprobs {value!r} needs structured_outputs: the masked trainer applies the "
                          "grammar's allowed sets, and an unconstrained config has none (D-021 Part A)")
    try:
        from lofi_rl.rl.masked import TRAINER_LOGPROBS, path_for
    except ImportError as err:
        raise ConfigError(f"trainer_logprobs {value!r}: lofi_rl.rl.masked, which defines its values "
                          f"(TRAINER_LOGPROBS) and their paths (path_for), cannot be imported ({err})") from err
    if not isinstance(value, str) or value not in TRAINER_LOGPROBS:
        raise ConfigError(f"trainer_logprobs {value!r} is not one of {list(TRAINER_LOGPROBS)} "
                          "(lofi_rl.rl.masked.TRAINER_LOGPROBS)")
    try:
        path_for(value)
    except Exception as err:  # noqa: BLE001: a value without a path never reaches a run
        raise ConfigError(f"trainer_logprobs {value!r}: lofi_rl.rl.masked.path_for names no path for it "
                          f"({err})") from err
    if value not in MASK_T2_PATHS:
        raise ConfigError(f"trainer_logprobs {value!r}: config.MASK_T2_PATHS names no TRL form it runs under "
                          f"(it has {sorted(MASK_T2_PATHS)}): say which before a run takes it")
    return value


def _check_pool(cfg: RunConfig, root: Path = ROOT) -> None:
    """Every pool entry is a .ls file or a directory, and a pool that lists anything holds at least one .ls file."""
    pool = (cfg.copy_penalty or {}).get("pool", ())
    for entry in pool:
        p = Path(entry) if Path(entry).is_absolute() else root / entry
        if not p.exists():
            raise ConfigError(f"copy_penalty.pool {entry!r} does not exist")
        if not p.is_dir() and p.suffix != POOL_SUFFIX:
            raise ConfigError(f"copy_penalty.pool {entry!r} is not a {POOL_SUFFIX} file or a directory")
    if pool and not cfg.pool_files(root):
        raise ConfigError(f"copy_penalty.pool {list(pool)} holds no {POOL_SUFFIX} file")


def _structured(builder: str, palette_path: Path) -> Mapping[str, Any]:
    """The regex a `structured_outputs` builder makes from the config's palette, with its identity."""
    from lofi_rl import grammar                     # only a constrained config needs the grammar and the palette
    from lofi_rl.palette import PaletteError, load_palette
    try:
        palette = load_palette(palette_path, verify=False)
    except PaletteError as err:
        raise ConfigError(f"structured_outputs: the palette {palette_path} does not load ({err})") from err
    regex = getattr(grammar, builder)(palette)
    return MappingProxyType({"builder": builder, "field": STRUCTURED_FIELD, "regex": regex,
                             "sha256": hashlib.sha256(regex.encode("utf-8")).hexdigest(), "chars": len(regex)})


def structured_summary(cfg: RunConfig) -> dict[str, Any] | None:
    """What run.json and the dry run record of the regex: everything but its text."""
    so = cfg.structured_outputs
    return None if so is None else {k: v for k, v in so.items() if k != "regex"}


def resolve(cfg: RunConfig, profile: str) -> dict[str, Any]:
    """The trainer settings of one GPU profile: `trainer`, then the profile's overrides (min_gib dropped), then the
    `structured_outputs` regex in its GRPOConfig field when the config asks for one."""
    if profile not in cfg.profiles:
        raise ConfigError(f"unknown GPU profile {profile!r}; the config has {sorted(cfg.profiles)}")
    kw = dict(cfg.trainer)
    kw.update({k: v for k, v in cfg.profiles[profile].items() if k != "min_gib"})
    if cfg.structured_outputs is not None:
        kw[cfg.structured_outputs["field"]] = cfg.structured_outputs["regex"]
    missing = [k for k in REQUIRED if k not in kw]
    if missing:
        raise ConfigError(f"profile {profile}: trainer settings missing {missing}")
    if kw["remove_unused_columns"]:
        raise ConfigError("trainer.remove_unused_columns must be false: the reward reads the brief columns")
    per_step = kw["per_device_train_batch_size"] * kw["gradient_accumulation_steps"]
    want = cfg.prompts_per_step * kw["num_generations"]
    if per_step != want:
        raise ConfigError(f"profile {profile}: per_device_train_batch_size {kw['per_device_train_batch_size']} x "
                          f"gradient_accumulation_steps {kw['gradient_accumulation_steps']} = {per_step} completions "
                          f"per step, but prompts_per_step {cfg.prompts_per_step} x num_generations "
                          f"{kw['num_generations']} = {want} (one GPU)")
    return kw


def detect_profile(cfg: RunConfig, total_gib: float) -> str:
    """The profile for a GPU with `total_gib` GiB: the one with the largest min_gib it meets."""
    fits = [(p["min_gib"], n) for n, p in cfg.profiles.items() if p["min_gib"] <= total_gib]
    if not fits:
        raise ConfigError(f"no profile for a GPU with {total_gib:.1f} GiB; the smallest needs "
                          f"{min(p['min_gib'] for p in cfg.profiles.values())} GiB")
    return max(fits)[1]


def fit_fields(kwargs: Mapping[str, Any], config_cls: type, optional: Iterable[str]) -> tuple[dict[str, Any], list[str]]:
    """Split settings into those `config_cls` (a dataclass, TRL's GRPOConfig) has, and the optional ones it lacks.
    A required name it lacks is a ConfigError naming every such field, raised before anything heavy loads."""
    names = {f.name for f in dataclasses.fields(config_cls)}
    optional = set(optional)
    unknown = [k for k in kwargs if k not in names]
    hard = sorted(k for k in unknown if k not in optional)
    if hard:
        raise ConfigError(f"{config_cls.__name__} has no field {hard}: fix the name in the config (or list it in "
                          "optional_fields if the run can go without it)")
    return {k: v for k, v in kwargs.items() if k in names}, sorted(unknown)
