"""The GRPO reward (docs/design/taste_pipeline.md §5-6; CLAUDE.md §8; docs/GRPO.md §2; docs/SCORER.md §2, §10).

    R = G * sum_k w_k(t) * r_k  -  (P_sax + sum(penalties) + P_group + P_fix)

make_reward(palette_path, log_path) returns the function TRL's GRPOTrainer calls once per generation batch:
reward(prompts=..., completions=..., completion_ids=..., **columns) -> list[float], one float per completion.

- **G**, the hard gate: 1 if lofi_rl.verify.score(text, brief, palette).gate_ok, else 0. A completion cut off
  at max_completion_length (its last token is no EOS) is gate G2 whatever its text, because G2 comes first
  (DSL.md §7) and parses nothing (DSL.md §1): its P_sax, penalties and P_fix are 0 too.
- **r_k and w_k(t)**: theory and knobs come from verify.score; the weights come from lofi_rl.rl.curriculum at
  the step t that TRL passes (trainer_state.global_step), else at this function's own call counter, which a
  resumed run starts at the checkpoint's step (start_step).
- **taste, CE and theme are 0 without a scorer** (the smoke): only their curriculum weights are logged.
- **With a scorer** (make_reward(..., scorer=...) or $LOFI_SCORER_SOCK; lofi_rl.scoring, docs/SCORER.md): the
  completions that passed the symbolic gate go to the scorer server in one POST /score, which renders them and
  returns the audio part:
      G = G_sym x hard x 1[PQ >= tau_PQ] x clip(P_acc / 0.5, 0.25, 1)          (taste_pipeline.md §5, "Gate")
  and r_ce, r_theme, r_taste. The weights still come from the curriculum, except that when the server reports
  CE as not validated the reward applies taste_pipeline.md §5's rule, written beside W0 there as
  "ce→0, knobs→.20 unless CE validated": w_ce = 0 and w_knobs = 0.20 at every step. Any scorer failure (a non-200,
  a timeout, a malformed answer) raises, whatever `strict` says: the trainer crashes and resumes from its
  checkpoint rather than train on a silently partial reward (§5).
- **The run's pin** (scorer_pin): a file every link of a chained run shares. The first link writes the versions
  and live terms its scorer reported; every later link seeds its client from the file before /health, so a
  server with another RM, references, calibration, features or formulas than the run started with is refused
  (taste_pipeline.md §4: "rm_ver is pinned per run"), and the CE rule's weights never change under a run.
- **Penalties**: verify.score's P_sax and per-loop penalties (bars, len) and P_fix (repairs), plus
  verify.group_penalties over each prompt's group. A group is the completions of one dataset row (the trainer's
  `brief_id` column), or of one prompt text when that column is absent. TRL repeats each row num_generations
  times, so a group is one GRPO group however the batch is ordered.
- **Failures**: a verifier exception or a non-finite value scores 0 (gate "error") and is logged with its
  message, so a verifier bug cannot crash a GPU run; strict=True re-raises instead. The smoke counts them
  (lofi_rl.rl.smoke_check) and fails on any.
- **P_copy** (opt-in: make_reward(..., copy=CopyPenalty); a config's `copy_penalty` key): taste_pipeline §5's copy
  penalty, 0 at J 0.6 rising linearly to 0.3 at J >= 0.8, with J the largest character 5-gram Jaccard of the
  completion with every example the prompt version can show and the pool (lofi_rl.rl.copy_penalty, junk.py's
  measure). It is one more term of the penalty, for every completion that is scored (a failed gate and a G2 keep it,
  as they keep P_group: it reads the text, not a parse); a verifier error still scores 0. Logged as p_copy, copy_j
  (the largest J) and copy_nearest (its reference).
- **In the language** (opt-in: make_reward(..., language=cfg.structured_outputs); a constrained config): whether each
  completion fullmatches the decoding regex under Python's `re`, by finish reason read from the token ids: "length"
  when it hit the limit (G2's test), "stop" when it ended before it or on an EOS, "other" when the ids or the limit
  are unknown. Under a working constraint every "stop" completion is in the language (D-020, Validity).
- **The shown example**: a dataset row of prompt v2 carries `example_id`, the example its prompt showed; the log
  copies it per completion (absent for v1 rows, whose one example is "example").

Every completion is one JSONL line in log_path with every sub-reward, its weights, the step and the text; with a
scorer also G_sym, the server's item (`audio`), the clip id, the CE rule (`w_rule`) and the server's five versions.
Without a scorer, P_copy and the language check, the rewards, the log and the per-step stats are byte-for-byte what
they were before the scorer hook (tests/test_scoring_reward_hook.py against tests/golden/reward_no_scorer.json).
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
import re
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
from lofi_rl.scoring import client as scoring_client
from lofi_rl.scoring import protocol as scoring_protocol

NAME = "lofi_reward"                         # TRL logs it as rewards/lofi_reward/{mean,std}
VERIFY_TERMS: tuple[str, ...] = ("theory", "knobs")         # symbolic, from lofi_rl.verify.score
SERVER_TERMS: tuple[str, ...] = ("taste", "ce", "theme")    # from the scorer server; 0 without one
ERROR_GATE = "error"
PQ_GATE = "PQ"                               # the audio gate's PQ floor (taste_pipeline.md §5)
KNOBS_WITHOUT_CE = 0.20                      # taste_pipeline.md §5, beside W0: "ce→0, knobs→.20 unless CE validated"
CE_RULE = "ce->0, knobs->0.20: the scorer reports CE not validated (taste_pipeline.md §5, W0)"
FINISH_GROUPS = ("stop", "length", "other")  # junk.FINISH_GROUPS: the in-language counts' finish reasons
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


def finish_reason(truncated: bool | None) -> str:
    """The finish reason the token ids show (is_truncated): "length" at the limit, "stop" before it or on an EOS,
    "other" when the ids or the limit are unknown."""
    return "length" if truncated is True else "stop" if truncated is False else "other"


# ------------------------------------------------------------------------------------------ the reward
class LofiReward:
    """The reward TRL calls. Build it with make_reward(). `stats[step]` holds the latest call's summary for
    that step (the trainer's step log merges it with TRL's metrics); every completion goes to `log_path`."""

    def __init__(self, palette: Palette, log_path: Path | str, verify: ModuleType, *,
                 max_completion_length: int | None = None, eos_token_ids: Iterable[int] = (),
                 start_step: int = 0, strict: bool = False, scorer: Any = None, scorer_wait_s: float = 600.0,
                 scorer_pin: Path | str | None = None, copy: Any = None, language: Mapping[str, Any] | None = None):
        self.__name__ = NAME
        self.palette, self.verify, self.strict = palette, verify, strict
        self.log_path = Path(log_path)
        self.max_completion_length = max_completion_length
        self.eos_token_ids = frozenset(int(t) for t in eos_token_ids)
        self.start_step, self.calls = int(start_step), 0
        self.stats: dict[int, dict[str, Any]] = {}
        self.scorer, self.scorer_wait_s, self._scorer_ready = scorer, float(scorer_wait_s), False
        self.scorer_pin = None if scorer_pin is None else Path(scorer_pin)
        self.copy = copy                            # a CopyPenalty (lofi_rl.rl.copy_penalty), or None: no P_copy
        self.language = None                        # (sha256, compiled regex) of the decoding regex, or None
        if language is not None:
            regex = language["regex"]
            sha = hashlib.sha256(regex.encode("utf-8")).hexdigest()
            if language.get("sha256") not in (None, sha):
                raise ValueError(f"the language regex has sha256 {sha}, not the {language['sha256']} it was given with")
            self.language = (sha, re.compile(regex))
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
        w_rule = None
        if self.scorer is not None:
            w, w_rule = self._scorer_weights(w)
        briefs = [decode_brief(b) for b in self._column(kwargs, "brief", n, required=True)]
        brief_ids = self._column(kwargs, "brief_id", n, required=False)
        shown = self._column(kwargs, "example_id", n, required=False) if "example_id" in kwargs else None
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
        copies: list[Any] = [None] * n
        if self.copy is not None:
            for i in range(n):
                try:
                    copies[i] = self.copy(texts[i], briefs[i], trunc[i])
                    if not (_real(copies[i].p_copy) and math.isfinite(copies[i].p_copy)):
                        raise ValueError(f"the copy penalty returned p_copy={copies[i].p_copy!r}, which is not finite")
                except Exception as e:  # noqa: BLE001: like a verifier error: scored 0 and logged (strict re-raises)
                    if self.strict:
                        raise
                    copies[i] = None
                    errors[i] = errors[i] or f"copy_penalty: {type(e).__name__}: {e}"
                    scores[i] = None

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
                                trunc[i], texts[i], scores[i], p_group[i], errors[i], copies[i],
                                None if shown is None else shown[i]) for i in range(n)]
        audio = None
        if self.scorer is not None:     # after the symbolic records, so a scorer error leaves no partial log
            audio = self._audio(step, texts, briefs, brief_ids, records)
            records = [self._with_audio(r, audio["items"].get(i), w, w_rule, audio["versions"])
                       for i, r in enumerate(records)]
        rewards = [float(r["reward"]) for r in records]
        with self.log_path.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        st = self._summary(step, source, n, len(members), w, records, rewards, trunc, errors)
        if self.copy is not None:
            st["copy"] = self._copy_summary(records)
        if self.language is not None:
            st["in_language"] = self._language_summary(records, self.language[0])
        if audio is not None:
            st["audio"] = self._audio_summary(records, audio, w_rule)
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

    def _record(self, i, step, source, w, g, key, brief_id, brief, ids, trunc, text, s, p_group, error,
                copy=None, shown=None) -> dict:
        rec: dict[str, Any] = {
            "step": step, "step_source": source, "call": self.calls, "i": i, "group": g, "group_key": key,
            "brief_id": brief_id, "brief": dataclasses.asdict(brief),
            "n_tokens": len(ids) if ids is not None else None, "truncated": trunc, "w": dict(w)}
        if error is not None:
            rec.update(gate_ok=False, gate=ERROR_GATE, G=0, r={k: 0.0 for k in TERMS}, theory_parts={}, knob_parts={},
                       p_sax=0.0, penalties={}, p_group=0.0, p_fix=0.0, repairs=0, positive=0.0, penalty=0.0,
                       reward=0.0, digest=None, error=error)
            if self.copy is not None:
                rec.update(p_copy=0.0, copy_j=None, copy_nearest=None)
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
            if copy is not None:                # opt-in, so a run without P_copy adds nothing, not even 0.0
                penalty += copy.p_copy
            rec.update(gate_ok=gate_ok, gate=gate, G=G, r=r,
                       theory_parts=_parts(s.theory_parts), knob_parts=_parts(s.knob_parts), p_sax=p_sax,
                       penalties=pens, p_group=p_group, p_fix=p_fix, repairs=0 if g2 else int(s.repairs),
                       positive=positive, penalty=penalty, reward=positive - penalty,
                       digest=None if g2 else s.digest)
            if copy is not None:
                rec.update(p_copy=copy.p_copy, copy_j=copy.j, copy_nearest=copy.nearest)
            if g2:
                rec["raw_gate"] = None if s.gate_ok else (s.gate or "unknown")   # the verifier's view of the cut text
        if shown is not None:
            rec["example_id"] = shown
        if self.language is not None:
            rec.update(finish=finish_reason(trunc), in_language=self.language[1].fullmatch(text) is not None)
        rec["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rec["text"] = text
        return rec

    # -------------------------------------------------------------------------------------- the scorer
    def _scorer_weights(self, w: dict[str, float]) -> tuple[dict[str, float], str | None]:
        """The curriculum's weights, with §5's CE rule when the server reports CE as not validated."""
        if not self._scorer_ready:
            self._pin_scorer()
            self._scorer_ready = True
        if self.scorer.ce_validated:
            return w, None
        return {**w, "ce": 0.0, "knobs": KNOBS_WITHOUT_CE}, CE_RULE

    def _pin_scorer(self) -> None:
        """wait_healthy under the run's pin (module docstring): seed the client from scorer_pin when the file
        exists (ScorerError if /health differs from it), else write what this first link pinned."""
        pin = None if self.scorer_pin is None else scoring_client.read_pin(self.scorer_pin)
        if pin is not None:
            self.scorer.pinned, self.scorer.pinned_terms = dict(pin["versions"]), dict(pin["terms"])
        self.scorer.wait_healthy(self.scorer_wait_s)
        if self.scorer_pin is not None and pin is None:
            scoring_client.write_pin(self.scorer_pin, self.scorer, palette_sha256=self.palette.sha256,
                                     sock=str(getattr(self.scorer, "sock", "")), first_step=self.start_step + self.calls)

    def _audio(self, step: int, texts: list[str], briefs: list[Brief], brief_ids: list[Any],
               records: list[dict]) -> dict[str, Any]:
        """One POST /score for the completions that passed the symbolic gate; raises on any scorer failure."""
        sent = [i for i, r in enumerate(records) if r["gate_ok"]]
        out: dict[str, Any] = {"sent": len(sent), "items": {}, "versions": dict(self.scorer.pinned or {}),
                               "terms": self.scorer.terms, "batch": None, "scorer_ms": 0.0}
        if not sent:
            return out
        t = time.perf_counter()
        resp = self.scorer.score([scoring_protocol.item_dict(i, texts[i], briefs[i], brief_ids[i]) for i in sent],
                                 palette_sha256=self.palette.sha256, step=step)
        out.update(items={it["i"]: it for it in resp["items"]}, batch=resp["batch"], terms=dict(resp["terms"]),
                   versions={k: resp[k] for k in scoring_protocol.VERSION_KEYS},
                   scorer_ms=round(1000 * (time.perf_counter() - t), 2))
        return out

    @staticmethod
    def _with_audio(rec: dict, item: Mapping[str, Any] | None, w: Mapping[str, float], w_rule: str | None,
                    versions: Mapping[str, str]) -> dict:
        """A symbolic record with the server's audio part merged in (module docstring)."""
        out = {k: v for k, v in rec.items() if k not in ("text_sha256", "text")}
        g_sym = int(rec["G"])
        r = dict(rec["r"])
        if item is None:                # not sent: the symbolic gate failed, so G is already 0
            G, gate = 0.0, rec["gate"]
        else:
            r.update({k: float(item["r"][k]) for k in SERVER_TERMS})
            G = float(g_sym * item["G"])
            gate = rec["gate"] or item["gate"] or (PQ_GATE if item["diag"].get("pq_factor") == 0 else None)
        positive = G * sum(w[k] * r[k] for k in TERMS) if G else 0.0
        out.update(G=G, G_sym=g_sym, gate_ok=G > 0, gate=None if G > 0 else (gate or "unknown"), r=r,
                   positive=positive, reward=positive - rec["penalty"], w_rule=w_rule,
                   clip_id=None if item is None else item["clip_id"], audio=None if item is None else dict(item),
                   scorer=dict(versions), text_sha256=rec["text_sha256"], text=rec["text"])
        return out

    @staticmethod
    def _audio_summary(records: list[dict], audio: Mapping[str, Any], w_rule: str | None) -> dict[str, Any]:
        sent = [r["audio"] for r in records if r.get("audio") is not None]
        ok = [r for r in records if r["gate_ok"]]
        sig = {k: _mean([a["sig"][k] for a in sent if a["sig"].get(k) is not None]) for k in scoring_protocol.SIG_KEYS}
        diag = {k: _mean([a["diag"][k] for a in sent if a["diag"].get(k) is not None]) for k in ("p_acc", "rm_std")}
        p_acc = [a["diag"]["p_acc"] for a in sent if a["diag"].get("p_acc") is not None]
        diag["p_acc_lt_half"] = _mean([1.0 if p < 0.5 else 0.0 for p in p_acc])     # §7: "rate of P_acc < 0.5"
        return {"sent": audio["sent"], "hard_rate": _mean([float(a["hard"]) for a in sent]),
                "G_mean": _mean([float(r["G"]) for r in records]),
                "r_mean": {k: _mean([r["r"][k] for r in ok]) for k in SERVER_TERMS}, "sig_mean": sig,
                "diag_mean": diag, "p_acc_below_half": sum(1 for p in p_acc if p < 0.5),
                "gates": dict(Counter(a["gate"] for a in sent if a["gate"] is not None)),
                "pq_fail": sum(1 for a in sent if a["hard"] and a["diag"].get("pq_factor") == 0),
                "render_errors": sum(1 for a in sent if a["gate"] in ("render-error", "render-timeout")),
                "batch": audio["batch"], "scorer_ms": audio["scorer_ms"], "versions": audio["versions"],
                "terms": audio["terms"], "w_rule": w_rule}

    @staticmethod
    def _copy_summary(records: list[dict]) -> dict[str, Any]:
        """One step's P_copy: its mean over every completion, and J over the scored ones (an error has none)."""
        js = [r["copy_j"] for r in records if r.get("copy_j") is not None]
        return {"p_copy_mean": _mean([r["p_copy"] for r in records]),
                "penalized": sum(1 for r in records if r["p_copy"] > 0),
                "j_mean": _mean(js), "j_max": max(js, default=None),
                "at_least": {"0.6": sum(1 for j in js if j >= 0.6), "0.8": sum(1 for j in js if j >= 0.8)},
                "nearest": dict(sorted(Counter(r["copy_nearest"] for r in records
                                               if r.get("copy_nearest") is not None and r["p_copy"] > 0).items()))}

    @staticmethod
    def _language_summary(records: list[dict], sha: str) -> dict[str, Any]:
        """junk.json's in_language block for one step: per finish reason n, in_language and out_of_language."""
        out: dict[str, Any] = {"regex_sha256": sha}
        for g in FINISH_GROUPS:
            group = [r for r in records if r["finish"] == g]
            k = sum(1 for r in group if r["in_language"])
            out[g] = {"n": len(group), "in_language": k, "out_of_language": len(group) - k}
        return out

    @staticmethod
    def _summary(step, source, n, groups, w, records, rewards, trunc, errors) -> dict[str, Any]:
        ok = [r for r in records if r["gate_ok"]]       # G == 1 without a scorer; G > 0 with one
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
                if "copy" in st and st["copy"]["p_copy_mean"] is not None:     # opt-in: absent, it logs nothing
                    log_metric("lofi/p_copy", st["copy"]["p_copy_mean"])
                if "in_language" in st:
                    log_metric("lofi/stop_out_of_language", st["in_language"]["stop"]["out_of_language"])
                if "audio" in st:               # with a scorer: taste_pipeline.md §5's hack/* signatures
                    a = st["audio"]
                    if a["G_mean"] is not None:
                        log_metric("lofi/G_mean", a["G_mean"])
                    for k, v in {**a["sig_mean"], **a["diag_mean"]}.items():
                        if v is not None:
                            log_metric(f"hack/{k}", v)
            except Exception as e:  # noqa: BLE001
                errs.append(f"log_metric: {type(e).__name__}: {e}")
        if callable(log_extra):
            try:
                log_extra("lofi_gate", [r["gate"] or "ok" for r in records])
                if "audio" in st:
                    log_extra("clip_id", [r["clip_id"] or "" for r in records])
            except Exception as e:  # noqa: BLE001
                errs.append(f"log_extra: {type(e).__name__}: {e}")
        return errs


def resolve_scorer(scorer: Any = None, environ: Mapping[str, str] | None = None) -> Any:
    """The scorer a reward uses: None reads $LOFI_SCORER_SOCK (unset or empty: no scorer); False is no scorer;
    a str or Path is a socket path; anything else is used as a client (ScorerClient or a stand-in with
    wait_healthy, ce_validated, terms, pinned and score; with a scorer_pin, also pinned_terms)."""
    if scorer is None:
        return scoring_client.from_env(environ)
    if scorer is False:
        return None
    if isinstance(scorer, (str, Path)):
        return scoring_client.ScorerClient(scorer)
    return scorer


def make_reward(palette_path: Path | str | Palette, log_path: Path | str, *, verify: ModuleType | None = None,
                max_completion_length: int | None = None, eos_token_ids: Iterable[int] = (),
                start_step: int = 0, strict: bool = False, scorer: Any = None,
                scorer_wait_s: float = 600.0, scorer_pin: Path | str | None = None, copy: Any = None,
                language: Mapping[str, Any] | None = None) -> LofiReward:
    """The TRL reward function: (prompts, completions, **kwargs) -> list[float] (module docstring).

    palette_path: the palette the loops are scored against (loaded with verify=False: the reward is symbolic and
    reads no sound file). log_path: the per-completion JSONL, appended to. The keyword-only extras are optional:
    `verify` replaces lofi_rl.verify (tests); max_completion_length and eos_token_ids let the reward see a
    completion cut at the token limit (gate G2); start_step is the step of the first call when TRL passes no
    trainer_state (a resumed run starts at its checkpoint's step); strict re-raises verifier errors.
    `scorer` (resolve_scorer: default $LOFI_SCORER_SOCK) adds the scorer server's audio part; the first call
    waits up to scorer_wait_s for its /health, under the run's pin when `scorer_pin` names one (a file shared by
    every link of the run, never a per-link summary directory). The GRPO config has no scorer field yet
    (lofi_rl.rl.config refuses unknown keys), so until it does a run turns the scorer on with the environment
    variable. `copy` (a lofi_rl.rl.copy_penalty.CopyPenalty) adds P_copy, and `language` (a mapping with the
    decoding regex under "regex" and optionally its "sha256": a config's structured_outputs) counts the completions
    in the regex's language; both are off by default, so a run without them logs what it did before."""
    if verify is None:
        verify = importlib.import_module("lofi_rl.verify")
    palette = palette_path if isinstance(palette_path, Palette) else load_palette(palette_path, verify=False)
    return LofiReward(palette, log_path, verify, max_completion_length=max_completion_length,
                      eos_token_ids=eos_token_ids, start_step=start_step, strict=strict,
                      scorer=resolve_scorer(scorer), scorer_wait_s=scorer_wait_s, scorer_pin=scorer_pin,
                      copy=copy, language=language)
