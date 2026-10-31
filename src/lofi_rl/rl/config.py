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
through the same field check as every other name (UNVERIFIED for TRL 1.17.1 until the cluster: a wrong name stops
the run in about 2 minutes). Writing that field by hand in `trainer` or a profile is refused.
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
TOP_OPTIONAL = ("smoke", "structured_outputs")
STRUCTURED_BUILDERS = ("loopscript_regex",)                 # functions of lofi_rl.grammar: palette -> regex
STRUCTURED_FIELD = "vllm_structured_outputs_regex"          # the TRL GRPOConfig field the regex goes into
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

    def palette_path(self, root: Path = ROOT) -> Path:
        p = Path(self.palette)
        return p if p.is_absolute() else root / p


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

    cfg = RunConfig(
        path=str(path), sha256=hashlib.sha256(data).hexdigest(), run=str(doc["run"]),
        model_repo=str(model["repo"]), model_revision=model["revision"], palette=str(doc["palette"]),
        brief_seed=briefs["seed"], brief_count=briefs["count"], brief_start=briefs["start"], prompts_per_step=pps,
        lora=MappingProxyType(dict(lora)), trainer=MappingProxyType(dict(trainer)),
        profiles=MappingProxyType({n: MappingProxyType(dict(p)) for n, p in profiles.items()}),
        optional_fields=frozenset(optional), resume_step=resume)
    if not cfg.palette_path().is_file():
        raise ConfigError(f"palette {cfg.palette} does not exist")
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
