"""The interim GRPO reward of the smoke (docs/design/taste_pipeline.md §5-6; CLAUDE.md §8; docs/GRPO.md §2).

    R = G * sum_k w_k(t) * r_k  -  (P_sax + sum(penalties) + P_group + P_fix)

make_reward(palette_path, log_path) returns the function TRL's GRPOTrainer calls once per generation batch:
reward(prompts=..., completions=..., completion_ids=..., **columns) -> list[float], one float per completion.

- **G**, the hard gate: 1 if lofi_rl.verify.score(text, brief, palette).gate_ok, else 0. A completion cut off
  at max_completion_length (its last token is no EOS) is gate G2 whatever its text, because G2 comes first
  (DSL.md §7) and parses nothing (DSL.md §1): its P_sax, penalties and P_fix are 0 too.
- **r_k and w_k(t)**: theory and knobs come from verify.score; the weights come from lofi_rl.rl.curriculum at
  the step t that TRL passes (trainer_state.global_step), else at this function's own call counter, which a
  resumed run starts at the checkpoint's step (start_step).
- **taste, CE and theme are 0 in the smoke.** TODO(taste_pipeline.md §5, "Scorer server"): the scorer server
  (lofi_rl.scoring.server, POST /score over a node-local Unix socket, in the scorer env) will supply r_taste
  (the frozen taste RM), r_ce (Audiobox CE, only if it predicts Shresth's votes), r_theme (CLAP on the music
  stem against the contrast scenes) and the audio part of the gate (render, PQ floor, P_acc). Until it exists
  these three terms are 0, and only their curriculum weights are logged.
- **Penalties**: verify.score's P_sax and per-loop penalties (bars, len) and P_fix (repairs), plus
  verify.group_penalties over each prompt's group. A group is the completions of one dataset row (the trainer's
  `brief_id` column), or of one prompt text when that column is absent. TRL repeats each row num_generations
  times, so a group is one GRPO group however the batch is ordered.
- **Failures**: a verifier exception or a non-finite value scores 0 (gate "error") and is logged with its
  message, so a verifier bug cannot crash a GPU run; strict=True re-raises instead. The smoke counts them
  (lofi_rl.rl.smoke_check) and fails on any.

Every completion is one JSONL line in log_path with every sub-reward, its weights, the step and the text.
Standard library and the project's standard-library modules only, so it imports in the policy env and on the
laptop (tests/test_rl_imports.py).
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import math
import numbers
import statistics
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from lofi_rl.brief import Brief
from lofi_rl.palette import Palette, load_palette
from lofi_rl.rl import curriculum
from lofi_rl.rl.curriculum import TERMS

NAME = "lofi_reward"                         # TRL logs it as rewards/lofi_reward/{mean,std}
VERIFY_TERMS: tuple[str, ...] = ("theory", "knobs")         # symbolic, from lofi_rl.verify.score
SERVER_TERMS: tuple[str, ...] = ("taste", "ce", "theme")    # 0 until the scorer server exists (TODO above)
ERROR_GATE = "error"
_BRIEF_FIELDS = tuple(f.name for f in dataclasses.fields(Brief))


# ------------------------------------------------------------------------------------------ readings
def completion_text(c: Any) -> str:
    """The text of one TRL completion: a str (standard prompts) or [{"role": "assistant", "content": ...}]
    (conversational prompts; the last assistant message). Content given as parts keeps its text parts."""
    if isinstance(c, str):
        return c
    if isinstance(c, Mapping):
        return _content(c.get("content"))
    if isinstance(c, Sequence):
        msgs = [m for m in c if isinstance(m, Mapping)]
        if not msgs:
            return "".join(str(x) for x in c)
        mine = [m for m in msgs if m.get("role", "assistant") == "assistant"] or msgs
        return _content(mine[-1].get("content"))
    raise TypeError(f"cannot read a completion of type {type(c).__name__}")


def _content(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, Sequence):
        return "".join(str(p.get("text", "")) for p in x if isinstance(p, Mapping))
    return str(x)


def decode_brief(x: Any) -> Brief:
    """A brief column entry: the JSON text the trainer writes, a dict of Brief's fields, or a Brief."""
    if isinstance(x, Brief):
        return x
    if isinstance(x, str):
        x = json.loads(x)
    if isinstance(x, Mapping):
        return Brief(**{k: x[k] for k in _BRIEF_FIELDS if k in x})
    raise ValueError(f"cannot read a brief from {type(x).__name__} {x!r}")


def group_key(prompt: Any, brief_id: Any = None) -> str:
    """The GRPO group of a completion: its dataset row's brief_id, else a hash of its prompt."""
    if brief_id is not None:
        return str(brief_id)
    blob = prompt if isinstance(prompt, str) else json.dumps(prompt, sort_keys=True, ensure_ascii=False, default=str)
    return "p:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def is_truncated(ids: Iterable[int] | None, max_len: int | None, eos_ids: frozenset[int]) -> bool | None:
    """True when a completion hit the token limit: at least max_len tokens and no EOS as its last token (TRL cuts
    completion_ids after the first EOS). None when the ids or the limit are unknown: never guessed."""
    if ids is None or max_len is None:
        return None
    ids = list(ids)
    if len(ids) < max_len:
        return False
    return not ids or ids[-1] not in eos_ids


def _real(v: Any) -> bool:
    """A real number, numpy scalars included (numbers.Real), but not a bool."""
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and type(v).__name__ != "bool_"


def _finite(name: str, v: Any) -> float:
    if not _real(v) or not math.isfinite(float(v)):
        raise ValueError(f"verify.score returned {name}={v!r}, which is not finite")
    return float(v)


def _checked(s: Any) -> Any:
    """The score itself, after checking that every number entering R is finite (a NaN would poison TRL's group
    statistics) and that the gate is a bool (numpy's bool_ too)."""
    if not (isinstance(s.gate_ok, bool) or type(s.gate_ok).__name__ == "bool_"):
        raise ValueError(f"verify.score returned gate_ok={s.gate_ok!r}, not a bool")
    for name in ("theory", "knobs", "p_sax", "p_fix"):
        _finite(name, getattr(s, name))
    for k, v in dict(s.penalties).items():
        _finite(f"penalties[{k!r}]", v)
    return s


def _parts(d: Any) -> dict[str, Any]:
    out = {}
    for k, v in dict(d or {}).items():
        out[str(k)] = float(v) if _real(v) and math.isfinite(float(v)) else repr(v)
    return out


def _mean(xs: Sequence[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


# ------------------------------------------------------------------------------------------ the reward
class LofiReward:
    """The reward TRL calls. Build it with make_reward(). `stats[step]` holds the latest call's summary for
    that step (the trainer's step log merges it with TRL's metrics); every completion goes to `log_path`."""

    def __init__(self, palette: Palette, log_path: Path | str, verify: ModuleType, *,
                 max_completion_length: int | None = None, eos_token_ids: Iterable[int] = (),
                 start_step: int = 0, strict: bool = False):
        self.__name__ = NAME
        self.palette, self.verify, self.strict = palette, verify, strict
        self.log_path = Path(log_path)
        self.max_completion_length = max_completion_length
        self.eos_token_ids = frozenset(int(t) for t in eos_token_ids)
        self.start_step, self.calls = int(start_step), 0
        self.stats: dict[int, dict[str, Any]] = {}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------------------- inputs
    def _step(self, kwargs: Mapping[str, Any]) -> tuple[int, str]:
        gs = getattr(kwargs.get("trainer_state"), "global_step", None)
        if isinstance(gs, int) and not isinstance(gs, bool):
            return gs, "trainer_state"
        for name in ("global_step", "step"):
            v = kwargs.get(name)
            if isinstance(v, int) and not isinstance(v, bool):
                return v, name
        return self.start_step + self.calls, "counter"

    @staticmethod
    def _column(kwargs: Mapping[str, Any], name: str, n: int, required: bool) -> list[Any]:
        v = kwargs.get(name)
        if v is None:
            if required:
                raise ValueError(f"the reward needs a `{name}` column, one entry per completion (got none): "
                                 "the trainer's dataset carries it and TRL passes every extra column")
            return [None] * n
        if isinstance(v, (str, bytes)) or not isinstance(v, Sequence) or len(v) != n:
            got = len(v) if isinstance(v, Sequence) and not isinstance(v, (str, bytes)) else type(v).__name__
            raise ValueError(f"the `{name}` column has {got} entries for {n} completions")
        return list(v)

    # -------------------------------------------------------------------------------------- call
    def __call__(self, prompts: Sequence[Any] | None = None, completions: Sequence[Any] | None = None,
                 completion_ids: Sequence[Sequence[int]] | None = None, **kwargs: Any) -> list[float]:
        t0 = time.perf_counter()
        completions = list(completions or [])
        n = len(completions)
        prompts = list(prompts) if prompts is not None else [None] * n
        if len(prompts) != n:
            raise ValueError(f"{len(prompts)} prompts for {n} completions")
        step, source = self._step(kwargs)
        w = curriculum.weights(step)
        briefs = [decode_brief(b) for b in self._column(kwargs, "brief", n, required=True)]
        brief_ids = self._column(kwargs, "brief_id", n, required=False)
        ids = list(completion_ids) if completion_ids is not None else [None] * n
        if len(ids) != n:
            raise ValueError(f"{len(ids)} completion_ids for {n} completions")
        texts = [completion_text(c) for c in completions]
        trunc = [is_truncated(ids[i], self.max_completion_length, self.eos_token_ids) for i in range(n)]

        # groups, in order of first appearance
        index: dict[str, int] = {}
        group = [index.setdefault(group_key(prompts[i], brief_ids[i]), len(index)) for i in range(n)]
        members: list[list[int]] = [[] for _ in index]
        for i, g in enumerate(group):
            members[g].append(i)
        keys = {g: k for k, g in index.items()}

        scores: list[Any] = [None] * n
        errors: list[str | None] = [None] * n
        for i in range(n):
            try:
                scores[i] = _checked(self.verify.score(texts[i], briefs[i], self.palette))
            except Exception as e:  # noqa: BLE001: scored 0 and logged, never a crashed GPU run (strict re-raises)
                if self.strict:
                    raise
                errors[i] = f"{type(e).__name__}: {e}"

        p_group = [0.0] * n
        for live in ([i for i in m if scores[i] is not None] for m in members):
            if not live:
                continue
            try:
                got = list(self.verify.group_penalties([texts[i] for i in live],
                                                       [self._gated(scores[i], trunc[i]) for i in live]))
                if len(got) != len(live):
                    raise ValueError(f"group_penalties returned {len(got)} values for {len(live)} completions")
                vals = [_finite("group penalty", p) for p in got]
                if min(vals) < 0:
                    raise ValueError(f"group_penalties returned a negative penalty {min(vals)}")
                for i, p in zip(live, vals):
                    p_group[i] = p
            except Exception as e:  # noqa: BLE001: the whole group scores 0 rather than skip its penalty
                if self.strict:
                    raise
                for i in live:
                    errors[i] = errors[i] or f"group_penalties: {type(e).__name__}: {e}"

        records = [self._record(i, step, source, w, group[i], keys[group[i]], brief_ids[i], briefs[i], ids[i],
                                trunc[i], texts[i], scores[i], p_group[i], errors[i]) for i in range(n)]
        rewards = [float(r["reward"]) for r in records]
        with self.log_path.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        st = self._summary(step, source, n, len(members), w, records, rewards, trunc, errors)
        st["t_ms"] = round(1000 * (time.perf_counter() - t0), 2)
        st["hook_errors"] = self._hooks(kwargs, st, records)
        self.stats[step] = st
        self.calls += 1
        return rewards

    @staticmethod
    def _gated(score: Any, truncated: bool | None) -> Any:
        """The score group_penalties sees: a G2 completion shows its failed gate."""
        if truncated and dataclasses.is_dataclass(score) and not isinstance(score, type):
            return dataclasses.replace(score, gate_ok=False, gate="G2")
        return score

    def _record(self, i, step, source, w, g, key, brief_id, brief, ids, trunc, text, s, p_group, error) -> dict:
        rec: dict[str, Any] = {
            "step": step, "step_source": source, "call": self.calls, "i": i, "group": g, "group_key": key,
            "brief_id": brief_id, "brief": dataclasses.asdict(brief),
            "n_tokens": len(ids) if ids is not None else None, "truncated": trunc, "w": dict(w)}
        if error is not None:
            rec.update(gate_ok=False, gate=ERROR_GATE, G=0, r={k: 0.0 for k in TERMS}, theory_parts={}, knob_parts={},
                       p_sax=0.0, penalties={}, p_group=0.0, p_fix=0.0, repairs=0, positive=0.0, penalty=0.0,
                       reward=0.0, digest=None, error=error)
        else:
            g2 = trunc is True
            gate_ok = bool(s.gate_ok) and not g2
            G = 1 if gate_ok else 0
            gate = "G2" if g2 else (None if s.gate_ok else (s.gate or "unknown"))
            live = {"theory": float(s.theory), "knobs": float(s.knobs)}
            r = {k: live.get(k, 0.0) for k in TERMS}          # taste, ce, theme: 0 (module docstring, TODO)
            pens = {str(k): 0.0 if g2 else float(v) for k, v in dict(s.penalties).items()}
            p_sax = 0.0 if g2 else float(s.p_sax)
            p_fix = 0.0 if g2 else float(s.p_fix)
            positive = G * sum(w[k] * r[k] for k in TERMS)
            penalty = p_sax + sum(pens.values()) + p_group + p_fix
            rec.update(gate_ok=gate_ok, gate=gate, G=G, r=r,
                       theory_parts=_parts(s.theory_parts), knob_parts=_parts(s.knob_parts), p_sax=p_sax,
                       penalties=pens, p_group=p_group, p_fix=p_fix, repairs=0 if g2 else int(s.repairs),
                       positive=positive, penalty=penalty, reward=positive - penalty,
                       digest=None if g2 else s.digest)
            if g2:
                rec["raw_gate"] = None if s.gate_ok else (s.gate or "unknown")   # the verifier's view of the cut text
        rec["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rec["text"] = text
        return rec

    @staticmethod
    def _summary(step, source, n, groups, w, records, rewards, trunc, errors) -> dict[str, Any]:
        ok = [r for r in records if r["G"] == 1]
        pen_names = sorted({k for r in records for k in r["penalties"]})
        return {
            "step": step, "step_source": source, "n": n, "groups": groups,
            "reward_mean": _mean(rewards), "reward_std": statistics.pstdev(rewards) if rewards else None,
            "reward_min": min(rewards) if rewards else None, "reward_max": max(rewards) if rewards else None,
            "gate_rate": len(ok) / n if n else None,
            "gate_counts": dict(Counter(r["gate"] for r in records if r["gate"] is not None)),
            "truncated": sum(1 for t in trunc if t), "errors": sum(1 for e in errors if e),
            "w": dict(w), "r_mean": {k: _mean([r["r"][k] for r in ok]) for k in TERMS},
            "p_sax_mean": _mean([r["p_sax"] for r in records]),
            "penalties_mean": {k: _mean([r["penalties"].get(k, 0.0) for r in records]) for k in pen_names},
            "p_group_mean": _mean([r["p_group"] for r in records]),
            "p_fix_mean": _mean([r["p_fix"] for r in records]),
            "n_tokens_mean": _mean([r["n_tokens"] for r in records if r["n_tokens"] is not None]),
        }

    @staticmethod
    def _hooks(kwargs: Mapping[str, Any], st: Mapping[str, Any], records: list[dict]) -> list[str]:
        """TRL's optional per-batch logging hooks (taste_pipeline.md §5 names log_metric and log_extra; their
        TRL 1.17.1 signatures are unverified), used when passed and never allowed to crash the reward."""
        errs = []
        log_metric, log_extra = kwargs.get("log_metric"), kwargs.get("log_extra")
        if callable(log_metric):
            try:
                log_metric("lofi/reward", st["reward_mean"])
                log_metric("lofi/G", st["gate_rate"])
                for k in TERMS:
                    if st["r_mean"][k] is not None:
                        log_metric(f"lofi/r_{k}", st["r_mean"][k])
                log_metric("lofi/p_group", st["p_group_mean"])
            except Exception as e:  # noqa: BLE001
                errs.append(f"log_metric: {type(e).__name__}: {e}")
        if callable(log_extra):
            try:
                log_extra("lofi_gate", [r["gate"] or "ok" for r in records])
            except Exception as e:  # noqa: BLE001
                errs.append(f"log_extra: {type(e).__name__}: {e}")
        return errs


def make_reward(palette_path: Path | str | Palette, log_path: Path | str, *, verify: ModuleType | None = None,
                max_completion_length: int | None = None, eos_token_ids: Iterable[int] = (),
                start_step: int = 0, strict: bool = False) -> LofiReward:
    """The TRL reward function: (prompts, completions, **kwargs) -> list[float] (module docstring).

    palette_path: the palette the loops are scored against (loaded with verify=False: the reward is symbolic and
    reads no sound file). log_path: the per-completion JSONL, appended to. The keyword-only extras are optional:
    `verify` replaces lofi_rl.verify (tests); max_completion_length and eos_token_ids let the reward see a
    completion cut at the token limit (gate G2); start_step is the step of the first call when TRL passes no
    trainer_state (a resumed run starts at its checkpoint's step); strict re-raises verifier errors."""
    if verify is None:
        verify = importlib.import_module("lofi_rl.verify")
    palette = palette_path if isinstance(palette_path, Palette) else load_palette(palette_path, verify=False)
    return LofiReward(palette, log_path, verify, max_completion_length=max_completion_length,
                      eos_token_ids=eos_token_ids, start_step=start_step, strict=strict)
