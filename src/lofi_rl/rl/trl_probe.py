"""What a GRPO run reads from the installed TRL and vLLM at run time (D-020, Outcome; docs/GRPO.md §7).

TRL 1.17.1 and vLLM 0.32.0 are not on the laptop, so nothing here assumes their internals. Each function looks,
records what it found, and fails closed: a record, never an exception.

1. **Where TRL's IS metric comes from** (logp_metric_source). D-020 reads "mean |log IS ratio|" as the mean, over
   every completion token of every step, of |log π_vLLM(token) − log π_trainer(token)|, and lets TRL's own metric
   stand in for it only if its definition in the TRL source is checked to be this one. This finds the installed
   package's directory without importing it (importlib.util.find_spec), parses each source file that names the metric
   `sampling/sampling_logp_difference/mean` (ast, never executing it), and checks the chain that computes the
   appended value, in the function that appends it:
   - the value is one name, aggregated only by gather/mean/item-like calls (one process: the identity);
   - that name is a plain mean (torch.mean(D), D.mean(), torch.nanmean(D)) of a tensor D;
   - D is the absolute difference of two per-token log-prob tensors, |A − B| (torch.abs, .abs() or abs);
   - D is masked by a mask built from `completion_mask` (D[M], or torch.masked_select(D, M)) before the mean;
   - one operand is vLLM's: its name says sampling and logp;
   - the other is the trainer's: every assignment to it that is not None calls something with `self.model`.
   The verdict is "match" only when every element is found, for every place that appends the metric; otherwise
   "not_matched" (with what is missing), "not_found" (no file names the metric, or TRL is not installed) or "error".
   The record keeps the file, its sha256, the version and every source line the check read, and says whether
   TRL's completion mask drops truncated completions (`mask_truncated_completions`), since D-020 reads every
   completion token: with that mask, TRL's value equals D-020's only on steps with no truncated completion.
   What the check cannot see, and records as evidence only: which log-probs vLLM reports (TRL's `logprobs_mode`, and
   whether vLLM applies the grammar bitmask to the logits before it computes them; vllm_logprobs_evidence).
2. **The colocate engine's structured-outputs backend** (BackendPin, find_engine, engine_record). junk.py asks the
   LLM it builds for structured_outputs_config={"backend": "xgrammar"} and reads back what it got (junk.build_llm,
   junk.read_backend). In colocate mode TRL builds the LLM itself, so BackendPin wraps vllm.LLM.__init__ for the run:
   every LLM built in this process is asked for the pin the same way (a TypeError on the keyword builds it again
   without it, pin_accepted false; a caller that passes its own structured_outputs_config keeps it). find_engine
   then finds the LLM on the trainer at run time (a bounded walk of instance attributes), and engine_record reads
   its backend through junk.read_backend: requested, pin_accepted, backend, source, applied, and where the engine
   was found.
3. **D-021's source checks** (source_checks, lm_head_record; docs/DECISIONS.md D-021 Part A). The grammar-masked
   trainer overrides one TRL method and relies on what TRL, vLLM and PEFT do around it, so before the model loads
   train.py reads the installed sources (package_dir, ast; never imported) and refuses the launch unless T1-T5 all
   read "match" (D-021: "They fail closed with exit 2"). Each check returns a record {verdict, package, file, sha256,
   lines, missing, ...}: "match" only when every element is found, "not_matched" with what is missing, "not_found"
   (the package or a file is absent; on the laptop TRL, vLLM and PEFT are) or "error" (a file does not parse, or the
   check broke). `args` is the resolved GRPOConfig kwargs (lofi_rl.rl.config.resolve); a field it lacks is read as
   the literal default of TRL's GRPOConfig (or its base config), and the record says which. Without args the
   run-time halves cannot be decided, so T1-T3 do not match.
   - T1: in GRPOTrainer, every binding of old_per_token_logps / ref_per_token_logps in
     _generate_and_score_completions is None or the first value of self._get_per_token_logps_and_entropies(...)
     (old on self.model; ref on self.model under the adapter switch, or self.ref_model), both reach the loss through
     the output dict under their own names, _compute_loss's per_token_logps is that method's first value (its old is
     inputs' or per_token_logps.detach()), compute_loss returns only self._compute_loss(...), and use_liger_kernel is
     false.
   - T2: the method's body, and the path the mask can take through it. **support_ids** (TRL 1.17.1's form): one
     forward per row chunk of range(0, input_ids.size(0), batch_size), labels -100 at attention 0 and on
     [:, :-logits_to_keep], TRL's sampling_mask placed at [:, -logits_to_keep:] of a -1-filled (B, L, K) tensor passed
     as support_ids with fused_lm_head=True (no shift_labels), and each chunk's .log_probs[:, -logits_to_keep:]
     concatenated in order and returned first; trainer/utils.py's _fused_forward shifts support_ids[:, 1:] with
     labels[:, 1:] and hands it to _ChunkedLogProbFunction.apply, which normalizes each token's log-prob over its
     support; __init__ adds that head to the policy with temperature=self.temperature. So a grammar support passed as
     sampling_mask normalizes the old, reference and loss log-probs over the allowed set. **fill** (D-021's registered
     logits form, as in TRL 0.2x): one forward per chunk asked for logits_to_keep + 1 rows, the last dropped, divided
     by self.temperature, then a log-softmax. Either way TRL's own sampling_mask must be None at args: it is built only
     when top_p < 1, top_k > 0 or min_p > 0 (__init__'s filtered_sampling and return_sampling_mask, and
     VLLMGeneration.generate). `path` is the form that matched (None otherwise); `ordinary_forward` records whether
     fused_lm_head=False still reaches the model's own forward (the override's fill path computes from it).
   - T3: in sequence modes TRL's weight is exp(sum over the completion of (old - sampling) * mask) per sequence
     (NaN -> 0), set to 0 outside [clip_min, clip_max] in sequence_mask (clamped in sequence_truncate), stored as
     importance_sampling_ratio and multiplied into the loss; sampling/importance_sampling_ratio/mean is the mean of the
     flattened (B, 1) weights, so ratio_mean_over is "sequences". The args' mode must be a sequence mode, with the
     correction on.
   - T4: per_token_kl is exp(ref - logp) - (ref - logp) - 1, ref from inputs["ref_per_token_logps"] and logp the
     loss pass's; use_bias_correction_kl (which multiplies it by coef_1) and beta are recorded at args.
   - T5: PEFT's all-linear resolution (tuners_utils._maybe_include_all_linear_layers) removes the output embedding
     from the targets; lm_head_record(trainer.model) is the run-time half, once the trainer has built the PEFT model.
   - V1 (recorded, not gating): in each model runner vLLM ships, the method that calls apply_grammar_bitmask( on
     `logits` (in place) samples from those logits after it, and the sampler's processed-logprobs branch exists; TRL
     asks for logprobs_mode="processed_logprobs". vLLM 0.32 has two runners: v1/worker/gpu_model_runner.py (the lines
     D-021 cites) and Model Runner V2, v1/worker/gpu/model_runner.py, which config/vllm.py's use_v2_model_runner
     selects by default when Triton is available, VLLM_USE_V2_MODEL_RUNNER is unset and no feature it lacks is asked
     for; both are read, and the selection is recorded.
   - V2 (recorded): backend_xgrammar.py builds TokenizerInfo.from_huggingface(tokenizer, vocab_size=...), a
     GrammarCompiler, compiles a regex with compile_regex and builds GrammarMatcher(..., override_stop_tokens=...);
     structured_output/__init__.py passes vocab_size = model_config.get_vocab_size() and stop_token_ids =
     request.sampling_params.all_stop_token_ids. The masked trainer's xgrammar replay is built the same way.

Standard library and lofi_rl.rl.junk only: vLLM is imported inside BackendPin.for_vllm and PEFT inside
lm_head_record, in the policy env.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import importlib.metadata
import importlib.util
import operator
import os
import re
import time
import traceback
import types
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from lofi_rl.rl import junk

METRIC = "sampling/sampling_logp_difference/mean"
DEFINITION = ("the mean, over every completion token of every step, of |log pi_vLLM(token) - log pi_trainer(token)| "
              "(docs/DECISIONS.md D-020, Outcome)")
AGGREGATORS = frozenset({"gather", "gather_for_metrics", "mean", "nanmean", "item", "float", "cpu", "detach",
                         "reduce", "tolist"})
MEANS = frozenset({"mean", "nanmean"})
MAX_LINES = 12                  # source lines kept per element of the chain
MAX_FILES = 4000                # a package bigger than this is not scanned whole
SKIP_PREFIXES = ("torch", "transformers", "peft", "datasets", "accelerate", "numpy", "safetensors", "tokenizers")


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_dir(name: str) -> Path | None:
    """The installed package's directory, found without importing it; None when it is not installed (or blocked)."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    locations = list(spec.submodule_search_locations or []) if spec is not None else []
    return Path(locations[0]) if locations else None


# ------------------------------------------------------------------------------------------ 1. the IS metric
def logp_metric_source(package: str = "trl", metric: str = METRIC, base: Path | None = None) -> dict[str, Any]:
    """The installed TRL's definition of `metric`, checked against D-020's (module docstring). Never raises."""
    rec: dict[str, Any] = {"package": package, "version": _version(package), "metric": metric,
                           "definition": DEFINITION, "verdict": None, "file": None, "sha256": None,
                           "sites": [], "missing": [], "excludes_truncated": None, "lines": {}, "error": None}
    try:
        base = base if base is not None else package_dir(package)
        if base is None:
            rec.update(verdict="not_found", error=f"{package} is not installed here")
            return rec
        files = sorted(base.rglob("*.py"))[:MAX_FILES]
        hits = [(p, t) for p, t in ((p, p.read_text(encoding="utf-8", errors="replace")) for p in files) if metric in t]
        if not hits:
            rec.update(verdict="not_found", error=f"no source file of {package} at {base} names {metric!r}")
            return rec
        # GRPOTrainer's own file when it names the metric (another trainer may log the same name); else every file
        grpo = [(p, t) for p, t in hits if "class GRPOTrainer" in t]
        rec["files_naming_metric"] = [str(p.relative_to(base)) for p, _ in hits]
        rec["read"] = "the file that defines GRPOTrainer" if grpo else "every file that names the metric"
        results = []
        for p, text in grpo or hits:
            res = analyze_source(text, metric)
            res["file"] = str(p.relative_to(base))
            res["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            results.append(res)
        found = [r for r in results if r["sites"]]
        if not found:
            rec.update(verdict="not_found", files=[r["file"] for r in results],
                       error=f"{metric!r} is named but never appended to a metric list")
            return rec
        if len(found) > 1:      # two files append it: both must match, and both are recorded
            rec["others"] = [{k: r[k] for k in ("file", "sha256", "verdict", "missing")} for r in found[1:]]
        main = found[0]
        verdict = "match" if all(r["verdict"] == "match" for r in found) else "not_matched"
        rec.update({k: main[k] for k in ("file", "sha256", "sites", "missing", "excludes_truncated", "lines")},
                   verdict=verdict, root=str(base))
        rec["logprobs_mode"] = _keyword_lines(base, "logprobs_mode")
    except Exception as err:  # noqa: BLE001: a record, never a reason to lose the run
        rec.update(verdict="error", error=f"{type(err).__name__}: {err}"[:500], trace=traceback.format_exc()[-2000:])
    return rec


def analyze_source(text: str, metric: str = METRIC) -> dict[str, Any]:
    """The check of logp_metric_source on one source text: {verdict, sites, missing, excludes_truncated, lines}."""
    tree = ast.parse(text)
    src = text.splitlines()
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    appends = sorted((n for n in ast.walk(tree) if _is_metric_append(n, metric)), key=lambda n: n.lineno)
    lines: dict[int, str] = {}
    sites, missing_all = [], []
    excludes: bool | None = None
    for node in appends:
        site, missing, keep, trunc = _check_site(node, parents, src)
        sites.append(site)
        missing_all += [f"line {node.lineno}: {m}" for m in missing]
        for ln in keep:
            lines[ln] = src[ln - 1]
        if trunc:
            excludes = True             # None: not seen in the function that appends it (it may happen elsewhere)
    if not appends:
        missing_all.append(f"no `...[{metric!r}].append(value)`")
    verdict = "match" if appends and not missing_all else "not_matched"
    return {"verdict": verdict, "sites": sites, "missing": missing_all, "excludes_truncated": excludes,
            "lines": {str(k): lines[k] for k in sorted(lines)}}


def _is_metric_append(node: ast.AST, metric: str) -> bool:
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "append"
            and len(node.args) == 1):
        return False
    target = node.func.value
    return (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
            and target.slice.value == metric)


def _span(node: ast.AST) -> list[int]:
    start, end = node.lineno, getattr(node, "end_lineno", None) or node.lineno
    return list(range(start, min(end, start + MAX_LINES - 1) + 1))


def _names(node: ast.AST) -> list[str]:
    return [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]


def _assignments(fn: ast.AST, name: str, before: int) -> list[ast.AST]:
    """Assignments to `name` (plain, annotated, augmented or tuple-unpacking) in `fn` above line `before`."""
    out = []
    for n in ast.walk(fn):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and n.lineno < before:
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Name) and t.id == name for tg in targets for t in ast.walk(tg)):
                out.append(n)
    return sorted(out, key=lambda n: n.lineno)


def _call_name(call: ast.Call) -> str | None:
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None


def _mean_operand(value: ast.AST) -> str | None:
    """D when `value` is torch.mean(D), D.mean() or torch.nanmean(D) (inside a conditional expression's body)."""
    if isinstance(value, ast.IfExp):
        value = value.body
    if not isinstance(value, ast.Call) or _call_name(value) not in MEANS:
        return None
    f = value.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id != "torch" and not value.args:
        return f.value.id                                       # D.mean()
    if value.args and isinstance(value.args[0], ast.Name) and len(value.args) == 1 and not value.keywords:
        return value.args[0].id                                 # torch.mean(D)
    return None


def _abs_diff(value: ast.AST) -> tuple[str, str] | None:
    """(A, B) when `value` is |A - B|: torch.abs(A - B), abs(A - B) or (A - B).abs(), A and B plain names."""
    inner = None
    if isinstance(value, ast.Call) and _call_name(value) == "abs":
        f = value.func
        if value.args and len(value.args) == 1:
            inner = value.args[0]                               # torch.abs(A - B), abs(A - B)
        elif isinstance(f, ast.Attribute) and not value.args:
            inner = f.value                                     # (A - B).abs()
    if (isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Sub) and isinstance(inner.left, ast.Name)
            and isinstance(inner.right, ast.Name)):
        return inner.left.id, inner.right.id
    return None


def _masking(value: ast.AST, name: str) -> ast.AST | None:
    """The mask M when `value` is name[M] or torch.masked_select(name, M)."""
    if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) and value.value.id == name:
        return value.slice
    if (isinstance(value, ast.Call) and _call_name(value) == "masked_select" and len(value.args) == 2
            and isinstance(value.args[0], ast.Name) and value.args[0].id == name):
        return value.args[1]
    return None


def _uses_self_model(node: ast.AST) -> bool:
    for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
        for arg in list(call.args) + [k.value for k in call.keywords]:
            if (isinstance(arg, ast.Attribute) and arg.attr == "model" and isinstance(arg.value, ast.Name)
                    and arg.value.id == "self"):
                return True
    return False


def _is_none(node: ast.AST) -> bool:
    value = getattr(node, "value", None)
    return isinstance(value, ast.Constant) and value.value is None


def _check_site(node: ast.Call, parents: dict, src: list[str]) -> tuple[dict, list[str], list[int], bool | None]:
    """One append of the metric: (the site's record, what is missing, the source lines read, excludes_truncated)."""
    keep, missing = _span(_statement(node, parents)), []
    fn = node
    while fn in parents and not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        fn = parents[fn]
    site: dict[str, Any] = {"line": node.lineno, "function": getattr(fn, "name", None)}
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return site, ["the append is not inside a function"], keep, None
    expr = node.args[0]
    calls = sorted({_call_name(c) for c in ast.walk(expr) if isinstance(c, ast.Call)} - {None})
    names = sorted(set(_names(expr)) - {"self", "torch"})
    site["aggregation"] = calls
    if set(calls) - AGGREGATORS:
        missing.append(f"the appended value goes through {sorted(set(calls) - AGGREGATORS)}, not only "
                       f"{sorted(AGGREGATORS)}")
    if len(names) != 1:
        missing.append(f"the appended value names {names}, not one tensor")
        return site, missing, keep, None
    value_name = names[0]
    site["value"] = value_name
    v_assigns = _assignments(fn, value_name, node.lineno)
    if not v_assigns:
        return site, missing + [f"no assignment to {value_name} before the append"], keep, None
    v_node = v_assigns[-1]
    keep += _span(v_node)
    d_name = _mean_operand(getattr(v_node, "value", None))
    if d_name is None:
        return site, missing + [f"{value_name} is not a plain mean of one tensor (line {v_node.lineno})"], keep, None
    site["mean_of"] = d_name
    d_assigns = _assignments(fn, d_name, v_node.lineno)
    keep += [ln for a in d_assigns for ln in _span(a)]
    diff, mask_node, mask_line = None, None, None
    for a in d_assigns:
        value = getattr(a, "value", None)
        if isinstance(value, ast.Subscript) and _abs_diff(value.value) is not None:
            diff, mask_node, mask_line = _abs_diff(value.value), value.slice, a.lineno     # D = |A - B|[M]
        elif _abs_diff(value) is not None:
            diff, mask_node = _abs_diff(value), None                         # D = |A - B|, then masked below
        elif diff is not None and _masking(value, d_name) is not None:
            mask_node, mask_line = _masking(value, d_name), a.lineno
        elif diff is not None:
            missing.append(f"{d_name} is reassigned after |A - B| by something other than a mask (line {a.lineno})")
    if diff is None:
        return site, missing + [f"{d_name} is not |A - B| of two tensors"], keep, None
    site["difference"] = list(diff)
    sampling = [x for x in diff if "sampling" in x and "logp" in x]
    trainer = [x for x in diff if x not in sampling]
    if len(sampling) != 1 or len(trainer) != 1:
        missing.append(f"the operands {list(diff)} are not one sampling log-prob tensor and one other")
    else:
        site["vllm_operand"], site["trainer_operand"] = sampling[0], trainer[0]
        t_assigns = _assignments(fn, trainer[0], node.lineno)
        s_assigns = _assignments(fn, sampling[0], node.lineno)
        keep += [ln for a in t_assigns + s_assigns for ln in _span(a)]
        real = [a for a in t_assigns if not _is_none(a)]
        if not real or not all(_uses_self_model(a) for a in real):
            missing.append(f"{trainer[0]} is not computed by a call on self.model at every assignment "
                           f"(lines {[a.lineno for a in t_assigns]})")
        if not s_assigns:
            missing.append(f"no assignment to {sampling[0]} in {site['function']}")
    if mask_node is None:
        missing.append(f"{d_name} is not masked before its mean")
    else:
        mask_names = set(_names(mask_node))
        for m in sorted(mask_names):            # a mask held in a name: the last assignment to it before the masking
            prior = _assignments(fn, m, mask_line)
            if prior:
                mask_names |= set(_names(getattr(prior[-1], "value", prior[-1])))
                keep += _span(prior[-1])
        site["mask"] = ast.unparse(mask_node)
        if not any("completion_mask" in m for m in mask_names):
            missing.append(f"the mask {site['mask']!r} is not built from completion_mask")
    trunc, trunc_lines = _truncation_mask(fn)
    keep += trunc_lines
    return site, missing, keep, trunc


def _statement(node: ast.AST, parents: dict) -> ast.AST:
    while node in parents and not isinstance(node, ast.stmt):
        node = parents[node]
    return node


def _truncation_mask(fn: ast.AST) -> tuple[bool | None, list[int]]:
    """True (and the lines) when `fn` changes completion_mask under `if ...mask_truncated_completions:`; None when it
    does not there (it may elsewhere: not seen is not "no")."""
    for n in ast.walk(fn):
        if isinstance(n, ast.If) and "mask_truncated_completions" in ast.unparse(n.test):
            for a in ast.walk(n):
                if isinstance(a, (ast.Assign, ast.AugAssign)) and "completion_mask" in ast.unparse(a):
                    return True, _span(n.test) + _span(a)
    return None, []


def _keyword_lines(base: Path, keyword: str, limit: int = 8) -> list[str]:
    """Up to `limit` "file:line: text" lines of the package that pass `keyword=` (evidence; nothing is decided)."""
    pat = re.compile(rf"\b{re.escape(keyword)}\s*=")
    out = []
    for p in sorted(base.rglob("*.py"))[:MAX_FILES]:
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if pat.search(line):
                out.append(f"{p.relative_to(base)}:{i}: {line.strip()}"[:300])
                if len(out) >= limit:
                    return out
    return out


VLLM_EVIDENCE = {
    "v1/worker/gpu_model_runner.py": re.compile(r"grammar_bitmask|self\.sampler\(|def _sample\(|logprobs_mode"),
    "v1/sample/sampler.py": re.compile(r"logprobs_mode|compute_logprobs\(|def forward\("),
}


def vllm_logprobs_evidence(base: Path | None = None, limit: int = 30) -> dict[str, Any]:
    """Lines of the installed vLLM that say where the grammar bitmask meets the logits and which log-probs the
    sampler returns: recorded for a reader, never checked (checked: false). Never raises."""
    out: dict[str, Any] = {"package": "vllm", "version": _version("vllm"), "checked": False, "files": {},
                           "note": "whether vLLM's reported log-probs include the grammar mask is not checked here"}
    try:
        base = base if base is not None else package_dir("vllm")
        if base is None:
            out["error"] = "vllm is not installed here"
            return out
        for rel, pat in VLLM_EVIDENCE.items():
            p = base / rel
            if not p.is_file():
                out["files"][rel] = None
                continue
            hits = [f"{i}: {line.strip()}"[:240] for i, line in
                    enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1) if pat.search(line)]
            out["files"][rel] = hits[:limit]
    except Exception as err:  # noqa: BLE001
        out["error"] = f"{type(err).__name__}: {err}"[:300]
    return out


# ------------------------------------------------------------------------------------------ 2. the engine's backend
class BackendPin:
    """vllm.LLM.__init__, wrapped so that every LLM built in this process asks for `backend` the way junk.build_llm
    does (module docstring). `record` is live: calls, pin_accepted (True, False after a TypeError on the keyword,
    None while no LLM was built or when the caller passed its own config), the error, and what was wrapped.
    Never raises; restore() puts the original back."""

    def __init__(self, llm_cls: type | None, backend: str, error: str | None = None) -> None:
        self.backend, self._cls, self._orig, self._own = backend, None, None, False
        self.record: dict[str, Any] = {"requested": backend, "pin_accepted": None, "calls": 0, "patched": None,
                                       "error": error}
        if llm_cls is None:
            return
        try:
            orig = llm_cls.__init__
            own = "__init__" in vars(llm_cls)
            rec = self.record

            @functools.wraps(orig)
            def __init__(inner: Any, *args: Any, **kwargs: Any) -> None:
                rec["calls"] += 1
                if "structured_outputs_config" in kwargs:       # the caller chose: its choice stands, and is recorded
                    rec["pin_accepted"] = None
                    rec["set_by_caller"] = repr(kwargs["structured_outputs_config"])[:300]
                    return orig(inner, *args, **kwargs)
                try:
                    orig(inner, *args, structured_outputs_config={"backend": backend}, **kwargs)
                except TypeError as err:                        # junk.build_llm: an LLM() without the keyword
                    rec["pin_accepted"] = False
                    rec["pin_error"] = f"TypeError: {err}"[:300]
                    return orig(inner, *args, **kwargs)
                rec["pin_accepted"] = True
                return None

            llm_cls.__init__ = __init__
            self._cls, self._orig, self._own = llm_cls, orig, own
            self.record["patched"] = f"{llm_cls.__module__}.{llm_cls.__qualname__}.__init__"
        except Exception as err:  # noqa: BLE001: no pin, recorded; the run goes on as it would without one
            self.record["error"] = f"{type(err).__name__}: {err}"[:300]

    @classmethod
    def for_vllm(cls, backend: str) -> BackendPin:
        """The pin on vllm.LLM, imported here (policy env); a vLLM that cannot be imported gives a record."""
        try:
            import vllm
            return cls(vllm.LLM, backend)
        except Exception as err:  # noqa: BLE001
            return cls(None, backend, error=f"vllm.LLM: {type(err).__name__}: {err}"[:300])

    @property
    def llm_cls(self) -> type | None:
        return self._cls

    def restore(self) -> None:
        if self._cls is None:
            return
        try:
            if self._own:
                self._cls.__init__ = self._orig
            else:
                del self._cls.__init__
        except Exception as err:  # noqa: BLE001
            self.record["restore_error"] = f"{type(err).__name__}: {err}"[:300]
        self._cls = None


_LEAVES = (str, bytes, int, float, complex, bool, type, ModuleType, types.FunctionType, types.MethodType,
           types.BuiltinFunctionType)


def _skip(value: Any) -> bool:
    if value is None or isinstance(value, _LEAVES):
        return True
    mod = type(value).__module__ or ""
    return mod.split(".")[0] in SKIP_PREFIXES


def find_engine(root: Any, cls: type, *, max_depth: int = 3, max_nodes: int = 5000,
                name: str = "trainer") -> tuple[Any, str | None]:
    """The first instance of `cls` reachable from `root` through instance attributes (and small lists, tuples and
    dicts), breadth first, at most max_depth links away: (object, its path), or (None, None). The model, the
    tokenizer, the accelerator and the datasets are not walked (SKIP_PREFIXES). Reads instance __dict__s only, so no
    property runs."""
    queue: list[tuple[Any, str, int]] = [(root, name, 0)]
    seen: set[int] = set()
    while queue and len(seen) < max_nodes:
        obj, path, depth = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, cls):
            return obj, path
        if depth >= max_depth:
            continue
        if isinstance(obj, (list, tuple)):
            items = [(f"{path}[{i}]", v) for i, v in enumerate(obj)] if len(obj) <= 16 else []
        elif isinstance(obj, dict):
            items = [(f"{path}[{k!r}]", v) for k, v in obj.items()] if len(obj) <= 64 else []
        else:
            try:
                items = [(f"{path}.{k}", v) for k, v in vars(obj).items()]
            except TypeError:
                items = []
        for p, v in items:
            if isinstance(v, (cls, list, tuple, dict)) or not _skip(v):
                queue.append((v, p, depth + 1))
    return None, None


def engine_record(root: Any, pin: BackendPin, cls: type | None = None) -> dict[str, Any]:
    """junk's backend record for the LLM TRL built: the pin (requested, pin_accepted, calls), the backend read back
    from the engine found on `root` (junk.read_backend: backend, source), `applied` (the backend asked for, or None
    when nothing could be read) and `engine`, the path it was found at. Never raises."""
    rec: dict[str, Any] = {"requested": pin.backend, "pin_accepted": pin.record.get("pin_accepted"),
                           "backend": "unknown", "source": None, "applied": None, "engine": None,
                           "calls": pin.record.get("calls"), "patched": pin.record.get("patched")}
    for k in ("pin_error", "set_by_caller", "error"):
        if pin.record.get(k):
            rec[k] = pin.record[k]
    cls = cls or pin.llm_cls
    try:
        if cls is not None:
            llm, path = find_engine(root, cls)
            if llm is not None:
                rec.update(junk.read_backend(llm), engine=path)
    except Exception as err:  # noqa: BLE001: a record, never a reason to lose the run
        rec["read_error"] = f"{type(err).__name__}: {err}"[:300]
    rec["applied"] = None if rec["backend"] == "unknown" else rec["backend"] == pin.backend
    return rec


# ------------------------------------------------------------------------------------------ 3. D-021's source checks
GATING = ("T1", "T2", "T3", "T4", "T5")
RECORDED = ("V1", "V2")
LOGPS = "_get_per_token_logps_and_entropies"
SIGNATURE = ("self", "model", "input_ids", "attention_mask", "logits_to_keep")
ROW_COUNTS = ("input_ids.size(0)", "input_ids.shape[0]", "len(input_ids)")
IS_RATIO_MEAN = "sampling/importance_sampling_ratio/mean"
FILTERED_FROM_ARGS = "args.top_p < 1.0 or args.top_k > 0 or (args.min_p or 0.0) > 0.0"   # TRL's own test, on args
TRL_GRPO, TRL_UTILS, TRL_VLLM = "trainer/grpo_trainer.py", "trainer/utils.py", "generation/vllm_generation.py"
TRL_CONFIGS = (("trainer/base_config.py", None), ("trainer/grpo_config.py", "GRPOConfig"))   # base first: GRPO wins
TRL_KERNEL = "kernels/chunked_logprob.py"
VLLM_RUNNERS = {"v1": "v1/worker/gpu_model_runner.py", "v2": "v1/worker/gpu/model_runner.py"}
VLLM_SAMPLERS = {"v1": "v1/sample/sampler.py", "v2": "v1/worker/gpu/sample/sampler.py"}
VLLM_INPLACE = {"v1": "v1/structured_output/utils.py", "v2": "v1/worker/gpu/structured_outputs.py"}
VLLM_TOPK, VLLM_CONFIG, VLLM_MODEL_CONFIG = ("v1/sample/ops/topk_topp_sampler.py", "config/vllm.py",
                                             "config/model.py")
VLLM_BACKEND, VLLM_MANAGER, VLLM_PARAMS = ("v1/structured_output/backend_xgrammar.py",
                                           "v1/structured_output/__init__.py", "sampling_params.py")
PEFT_TUNERS, PEFT_CONSTANTS = "tuners/tuners_utils.py", "utils/constants.py"
_MISSING = object()
_OPS = {ast.Lt: operator.lt, ast.Gt: operator.gt, ast.LtE: operator.le, ast.GtE: operator.ge, ast.Eq: operator.eq,
        ast.NotEq: operator.ne, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
        ast.Is: operator.is_, ast.IsNot: operator.is_not}
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


class _NotFound(Exception):
    """A package or a source file a check needs is absent: that check reads not_found."""


class _Undecidable(Exception):
    """A value a check needs cannot be read: no args, or an expression _evaluate does not handle."""


def _u(node: ast.AST | None) -> str:
    return "" if node is None else ast.unparse(node)


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return _MISSING
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return _MISSING


def _none(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _kwarg(call: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def _assigns(node: ast.AST) -> list[ast.Assign]:
    """The single-target assignments anywhere below `node`, in line order."""
    return sorted((n for n in ast.walk(node) if isinstance(n, ast.Assign) and len(n.targets) == 1),
                  key=lambda n: (n.lineno, n.col_offset))


def _target(a: ast.Assign) -> str:
    return _u(a.targets[0])


def _walk_list(stmts: list[ast.stmt]) -> list[ast.AST]:
    return [n for st in stmts for n in ast.walk(st)]


def _jsonable(v: Any) -> Any:
    return v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)[:200]


def _numbered(kept: Mapping[int, str]) -> dict[str, str]:
    return {str(k): kept[k] for k in sorted(kept)}


class _Src:
    """One source file a check reads: the sha256 of its bytes, its AST with parent links, and its lines."""

    def __init__(self, package: str, base: Path, rel: str) -> None:
        raw = (base / rel).read_bytes()
        self.package, self.rel = package, rel
        self.sha256 = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8", errors="replace")
        self.tree = ast.parse(text, filename=rel)
        self.lines = text.splitlines()
        self.parents = {c: n for n in ast.walk(self.tree) for c in ast.iter_child_nodes(n)}

    def statement(self, node: ast.AST) -> ast.AST:
        while node in self.parents and not isinstance(node, ast.stmt):
            node = self.parents[node]
        return node

    def ancestors(self, node: ast.AST) -> list[ast.AST]:
        out = []
        while node in self.parents:
            node = self.parents[node]
            out.append(node)
        return out

    def inside(self, node: ast.AST, outer: ast.AST) -> bool:
        return any(a is outer for a in self.ancestors(node))

    def cls(self, name: str) -> ast.ClassDef | None:
        return next((n for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef) and n.name == name), None)

    def func(self, name: str, within: ast.AST | None = None) -> ast.AST | None:
        """The function `name` defined directly in `within` (the module when None), else anywhere below it."""
        scope = within if within is not None else self.tree
        direct = [n for n in getattr(scope, "body", []) if isinstance(n, _FUNCS) and n.name == name]
        return direct[0] if direct else next((n for n in ast.walk(scope) if isinstance(n, _FUNCS) and n.name == name
                                              and n is not scope), None)

    def withs(self, node: ast.AST) -> list[str]:
        return [_u(i.context_expr) for a in reversed(self.ancestors(node)) if isinstance(a, (ast.With, ast.AsyncWith))
                for i in a.items]

    def guards(self, node: ast.AST) -> list[tuple[ast.If, bool]]:
        """The `if` statements around `node`, outermost first, each with True when node is in its body (False: in
        its else)."""
        out, child = [], node
        for a in self.ancestors(node):
            if isinstance(a, ast.If) and child is not a.test:
                out.append((a, any(child is s for s in a.body)))
            child = a
        return out[::-1]


class _Reader:
    """The three packages' source files, each read and parsed once per source_checks call (a parse error is
    remembered, so every check that reads that file reports it)."""

    def __init__(self, bases: Mapping[str, Path | None]) -> None:
        self.bases = dict(bases)
        self._files: dict[tuple[str, str], _Src | Exception] = {}
        self._moved: dict[tuple[str, str, str], str] = {}
        self._defaults: dict[str, tuple[Any, str]] | None = None

    def src(self, package: str, rel: str, needle: str | None = None) -> _Src:
        base = self.bases.get(package)
        if base is None:
            raise _NotFound(f"{package} is not installed here")
        if needle is not None and not (base / rel).is_file():    # moved: the first file that holds `needle`
            key = (package, rel, needle)
            if key not in self._moved:
                self._moved[key] = next((str(p.relative_to(base)) for p in sorted(base.rglob("*.py"))[:MAX_FILES]
                                         if needle in p.read_text(encoding="utf-8", errors="replace")), rel)
            rel = self._moved[key]
        key = (package, rel)
        if key not in self._files:
            try:
                if not (base / rel).is_file():
                    raise _NotFound(f"{package}/{rel} does not exist under {base}")
                self._files[key] = _Src(package, base, rel)
            except Exception as err:  # noqa: BLE001: remembered, raised to each check that reads the file
                self._files[key] = err
        got = self._files[key]
        if isinstance(got, Exception):
            raise got
        return got

    def defaults(self) -> dict[str, tuple[Any, str]]:
        """{field: (literal default, "file:line")} of TRL's GRPOConfig and its base config, read from the source."""
        if self._defaults is None:
            self._defaults = {}
            for rel, only in TRL_CONFIGS:
                try:
                    src = self.src("trl", rel)
                except Exception:  # noqa: BLE001: no default from that file; a read that needs one is undecidable
                    continue
                for cls in (n for n in src.tree.body if isinstance(n, ast.ClassDef) and only in (None, n.name)):
                    for st in cls.body:
                        if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
                            v = st.value
                            if isinstance(v, ast.Call) and _call_name(v) == "field":
                                v = _kwarg(v, "default")
                            val = _literal(v)
                            if val is not _MISSING:
                                self._defaults[st.target.id] = (val, f"{rel}:{st.lineno}")
        return self._defaults


class _Check:
    """One check's record in the making: the files it read, the lines it kept, what is missing, its fields."""

    def __init__(self, what: str, reader: _Reader, main: tuple[str, str]) -> None:
        self.what, self.reader, self.main = what, reader, main
        self.used: dict[tuple[str, str], _Src] = {}
        self.kept: dict[tuple[str, str], dict[int, str]] = {}
        self.missing: list[str] = []
        self.fields: dict[str, Any] = {}

    def use(self, package: str, rel: str, needle: str | None = None) -> _Src:
        s = self.reader.src(package, rel, needle)
        if (package, rel) == self.main:
            self.main = (package, s.rel)
        self.used.setdefault((package, s.rel), s)
        self.kept.setdefault((package, s.rel), {})
        return s

    def keep(self, src: _Src, *nodes: ast.AST | None) -> None:
        for n in nodes:
            if n is not None and hasattr(n, "lineno"):
                for ln in _span(n):
                    if 0 < ln <= len(src.lines):
                        self.kept[(src.package, src.rel)][ln] = src.lines[ln - 1]

    def need(self, ok: Any, msg: str) -> bool:
        if not ok:
            self.missing.append(msg)
        return bool(ok)

    def cls(self, src: _Src, name: str) -> ast.ClassDef | None:
        node = src.cls(name)
        if self.need(node is not None, f"{src.package}/{src.rel} defines no class {name}"):
            self.kept[(src.package, src.rel)][node.lineno] = src.lines[node.lineno - 1]
        return node

    def method(self, src: _Src, cls: ast.ClassDef | None, name: str) -> ast.AST | None:
        if cls is None:
            return None
        node = next((n for n in cls.body if isinstance(n, _FUNCS) and n.name == name), None)
        if self.need(node is not None, f"class {cls.name} defines no {name}"):
            self.kept[(src.package, src.rel)][node.lineno] = src.lines[node.lineno - 1]
        return node

    def record(self) -> dict[str, Any]:
        pkg, rel = self.main
        main = self.used.get(self.main)
        rec: dict[str, Any] = {"check": self.what, "verdict": "not_matched" if self.missing else "match",
                               "package": pkg, "file": rel, "sha256": main.sha256 if main else None,
                               "lines": _numbered(self.kept.get(self.main, {})), "missing": list(self.missing)}
        files = {f"{p}/{r}": {"sha256": s.sha256, "lines": _numbered(self.kept.get((p, r), {}))}
                 for (p, r), s in self.used.items() if (p, r) != self.main}
        if files:
            rec["files"] = files
        rec.update(self.fields)
        return rec


def _read_arg(name: str, args: Any, defaults: Mapping[str, tuple[Any, str]], reads: dict[str, Any]) -> Any:
    """`name` from the resolved kwargs, else TRL's literal default; `reads` records the value and where it came from."""
    if args is None:
        raise _Undecidable("args not given")
    value = args.get(name, _MISSING) if isinstance(args, Mapping) else getattr(args, name, _MISSING)
    if value is not _MISSING:
        reads[name] = {"value": _jsonable(value), "from": "args"}
        return value
    if name in defaults:
        value, where = defaults[name]
        reads[name] = {"value": _jsonable(value), "from": f"default ({where})"}
        return value
    raise _Undecidable(f"{name} is not in args and has no literal default in TRL's GRPOConfig source")


def _attr_map(init: ast.AST) -> dict[str, str]:
    """{attr: field} for every `self.attr = args.field` in GRPOTrainer.__init__."""
    return {a.targets[0].attr: a.value.attr for a in _assigns(init)
            if isinstance(a.targets[0], ast.Attribute) and _u(a.targets[0].value) == "self"
            and isinstance(a.value, ast.Attribute) and _u(a.value.value) == "args"}


def _evaluate(node: ast.AST, attrs: Mapping[str, str], args: Any, defaults: Mapping[str, tuple[Any, str]],
              reads: dict[str, Any]) -> Any:
    """A small expression of the trainer's settings, at `args`: constants, lists, not and unary minus, and/or (with
    Python's short circuit), one comparison, `self.X` for an attribute __init__ sets from `args.Y` (attrs),
    `self.args.Y` and `args.Y`. Anything else is _Undecidable."""
    def ev(n: ast.AST) -> Any:
        return _evaluate(n, attrs, args, defaults, reads)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [ev(e) for e in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
        v = ev(node.operand)
        try:
            return (not v) if isinstance(node.op, ast.Not) else -v
        except TypeError as err:
            raise _Undecidable(f"`{_u(node)}`: {err}") from err
    if isinstance(node, ast.BoolOp):
        v = None
        for part in node.values:
            v = ev(part)
            if (not v) if isinstance(node.op, ast.And) else v:
                return v
        return v
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _OPS:
        try:
            return _OPS[type(node.ops[0])](ev(node.left), ev(node.comparators[0]))
        except TypeError as err:
            raise _Undecidable(f"`{_u(node)}`: {err}") from err
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "self" and node.attr in attrs:
            return _read_arg(attrs[node.attr], args, defaults, reads)
        if node.value.id == "args":
            return _read_arg(node.attr, args, defaults, reads)
    if isinstance(node, ast.Attribute) and _u(node.value) == "self.args":
        return _read_arg(node.attr, args, defaults, reads)
    raise _Undecidable(f"`{_u(node)}` is not a setting read here")


def _holds(src: _Src, node: ast.AST, ev: Any) -> bool:
    """Whether, at args, every `if` around `node` takes the branch that holds it."""
    return all(bool(ev(i.test)) == in_body for i, in_body in src.guards(node))


def _bindings(src: _Src, fn: ast.AST, name: str) -> list[ast.AST]:
    """Every statement of `fn` that binds `name` (any assignment, for, with, walrus), and its parameter of that name."""
    stmts = {src.statement(n) for n in ast.walk(fn)
             if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)}
    params = [a for a in ast.walk(fn.args) if isinstance(a, ast.arg) and a.arg == name]
    return sorted(stmts, key=lambda n: n.lineno) + params


def _is_logps_call(v: ast.AST | None) -> bool:
    return isinstance(v, ast.Call) and _u(v.func) == f"self.{LOGPS}"


def _inputs_key(v: ast.AST | None) -> Any:
    """k for `inputs["k"]` or `inputs.get("k")`; None otherwise."""
    if isinstance(v, ast.Subscript) and _u(v.value) == "inputs" and isinstance(_literal(v.slice), str):
        return _literal(v.slice)
    if isinstance(v, ast.Call) and _u(v.func) == "inputs.get" and v.args and isinstance(_literal(v.args[0]), str):
        return _literal(v.args[0])
    return None


def _value_form(v: ast.AST, name: str) -> Any:
    if _none(v):
        return "None"
    if isinstance(v, ast.Subscript) and _is_logps_call(v.value) and _literal(v.slice) == 0:
        return "call[0]"
    if _inputs_key(v) is not None:
        return f"inputs[{_inputs_key(v)!r}]"
    if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "detach" and not v.args:
        return f"{_u(v.func.value)}.detach()"
    if isinstance(v, ast.Name) and v.id == name:
        return "itself"
    if isinstance(v, ast.IfExp):
        return ("either", _value_form(v.body, name), _value_form(v.orelse, name))
    return _u(v)[:160]


def _binding_form(st: ast.AST, name: str) -> Any:
    """What a binding of `name` is: "None", "call[0]" (the first value of self._get_per_token_logps_and_entropies),
    "inputs['k']", "x.detach()", ("either", a, b) for a conditional, or the statement's own text."""
    if isinstance(st, ast.arg):
        return "a parameter"
    if isinstance(st, ast.Assign) and len(st.targets) == 1:
        t = st.targets[0]
        if isinstance(t, ast.Name):
            return _value_form(st.value, name)
        if isinstance(t, (ast.Tuple, ast.List)) and _is_logps_call(st.value):
            ids = [e.id if isinstance(e, ast.Name) else None for e in t.elts]
            if ids.count(name) == 1:
                return "call[0]" if ids[0] == name else f"call[{ids.index(name)}]"
    return f"{type(st).__name__}: {_u(st)[:160]}"


def _allowed(form: Any, allowed: tuple[str, ...]) -> bool:
    if isinstance(form, tuple):
        parts = form[1:]
        return all(p == "itself" or _allowed(p, allowed) for p in parts) and any(p != "itself" for p in parts)
    return form in allowed


def _show(form: Any) -> str:
    return f"{_show(form[1])} or {_show(form[2])}" if isinstance(form, tuple) else str(form)


def _logps_call(st: ast.AST) -> ast.Call | None:
    v = getattr(st, "value", None)
    return v if _is_logps_call(v) else v.value if isinstance(v, ast.Subscript) and _is_logps_call(v.value) else None


def _stored_under(fn: ast.AST, key: str) -> list[ast.AST]:
    """`out["key"] = key` assignments and `{"key": key}` dict entries in `fn`."""
    out: list[ast.AST] = [a for a in _assigns(fn) if isinstance(a.targets[0], ast.Subscript)
                          and _literal(a.targets[0].slice) == key and _u(a.value) == key]
    out += [d for d in ast.walk(fn) if isinstance(d, ast.Dict)
            and any(k is not None and _literal(k) == key and _u(v) == key for k, v in zip(d.keys, d.values))]
    return out


def _first_arg_is(call: ast.Call, name: str) -> bool:
    return bool(call.args and _u(call.args[0]) == name) or _u(_kwarg(call, "logits")) == name


def _branch_values(src: _Src, fn: ast.AST, flag: str, name: str) -> tuple[ast.AST | None, ast.AST | None,
                                                                             list[ast.AST]]:
    """The values `name` takes when `flag` is true and when it is false, from `name = a if flag else b` or from the
    blocks of `if flag:` (one name hop inside a block), with the statements read."""
    for a in _assigns(fn):
        if _target(a) == name and isinstance(a.value, ast.IfExp) and _u(a.value.test) == flag:
            return a.value.body, a.value.orelse, [a]

    def block(stmts: list[ast.stmt]) -> tuple[ast.AST | None, list[ast.AST]]:
        vals = {_target(s): s for s in stmts if isinstance(s, ast.Assign) and len(s.targets) == 1}
        s = vals.get(name)
        if s is None:
            return None, []
        if isinstance(s.value, ast.Name) and s.value.id in vals:
            return vals[s.value.id].value, [s, vals[s.value.id]]
        return s.value, [s]
    for n in sorted((n for n in ast.walk(fn) if isinstance(n, ast.If) and _u(n.test) == flag), key=lambda n: n.lineno):
        (t, rt), (f, rf) = block(n.body), block(n.orelse)
        if t is not None or f is not None:
            return t, f, [n.test, *rt, *rf]
    return None, None, []


def _evidence(reader: _Reader, package: str, rel: str, pattern: str, limit: int = 8) -> dict[str, Any]:
    """Up to `limit` lines of one file that match `pattern`, with its sha256: recorded for a reader, never checked."""
    try:
        s = reader.src(package, rel)
    except Exception as err:  # noqa: BLE001
        return {"file": f"{package}/{rel}", "error": f"{type(err).__name__}: {err}"[:300]}
    pat = re.compile(pattern)
    hits = {i: line for i, line in enumerate(s.lines, 1) if pat.search(line)}
    return {"file": f"{package}/{rel}", "sha256": s.sha256,
            "lines": {str(i): hits[i].strip()[:240] for i in sorted(hits)[:limit]}}


# T1 ---------------------------------------------------------------------------------------------------------------
T1_PLAN = (  # (function, name, the forms its bindings may take, the models a call may run on, or None: no call)
    ("_generate_and_score_completions", "old_per_token_logps", ("None", "call[0]"), ("self.model",)),
    ("_generate_and_score_completions", "ref_per_token_logps", ("None", "call[0]"), ("self.model", "self.ref_model")),
    ("_compute_loss", "per_token_logps", ("call[0]",), ("model",)),
    ("_compute_loss", "old_per_token_logps", ("None", "inputs['old_per_token_logps']", "per_token_logps.detach()"),
     None),
)


def _t1(c: _Check, args: Any) -> None:
    src = c.use("trl", TRL_GRPO, "class GRPOTrainer")
    cls = c.cls(src, "GRPOTrainer")
    fns = {n: c.method(src, cls, n) for n in ("_generate_and_score_completions", "_compute_loss", "compute_loss")}
    if any(f is None for f in fns.values()):
        return
    calls, forms = [], {}
    for fname, name, allowed, models in T1_PLAN:
        seen = []
        for st in _bindings(src, fns[fname], name):
            form = _binding_form(st, name)
            seen.append({"line": st.lineno, "form": _show(form)})
            c.keep(src, st)
            c.need(_allowed(form, allowed),
                   f"{fname}: {name} at line {st.lineno} is `{_show(form)}`, not one of {list(allowed)}")
            call = _logps_call(st) if form == "call[0]" else None
            if call is not None and models is not None:
                model = _u(call.args[0]) if call.args else _u(_kwarg(call, "model"))
                calls.append({"function": fname, "line": st.lineno, "target": name, "model": model,
                              "sampling_mask": _u(_kwarg(call, "sampling_mask")) or None, "with": src.withs(st)})
                c.need(model in models, f"{fname}: {name} at line {st.lineno} is computed on `{model}`, not "
                                        f"{' or '.join(models)}")
        forms[f"{fname}.{name}"] = seen
        if models is not None:
            c.need(any(s["form"] == "call[0]" for s in seen),
                   f"{fname}: {name} is never the first value of self.{LOGPS}(...)")
    keys = {}
    for key in ("old_per_token_logps", "ref_per_token_logps"):      # how old and ref reach _compute_loss's inputs
        hits = _stored_under(fns["_generate_and_score_completions"], key)
        c.keep(src, *(src.statement(h) for h in hits))
        keys[key] = [h.lineno for h in hits]
        c.need(hits, f"_generate_and_score_completions never stores {key} in its output under its own name")
    rets = [r for r in ast.walk(fns["compute_loss"]) if isinstance(r, ast.Return)]
    c.keep(src, *rets)
    c.need(rets and all(isinstance(r.value, ast.Call) and _u(r.value.func) == "self._compute_loss" for r in rets),
           f"compute_loss returns {[_u(r.value) for r in rets]}, not only self._compute_loss(...)")
    reads: dict[str, Any] = {}
    try:
        c.need(not _read_arg("use_liger_kernel", args, c.reader.defaults(), reads), "use_liger_kernel is true at args")
    except _Undecidable as err:
        c.need(False, f"use_liger_kernel cannot be read: {err}")
    c.fields.update(calls=calls, assignments=forms, output_keys=keys,
                    compute_loss_returns=[_u(r.value) for r in rets], reads=reads)


# T2 ---------------------------------------------------------------------------------------------------------------
def _signature(fn: ast.AST, extra: tuple[str, ...]) -> list[str]:
    params = [a.arg for a in fn.args.posonlyargs + fn.args.args]
    named = set(params) | {a.arg for a in fn.args.kwonlyargs}
    miss = [] if params[:5] == list(SIGNATURE) else [f"its parameters start {params[:5]}, not {list(SIGNATURE)}"]
    return miss + [f"it has no parameter {p}" for p in extra if p not in named]


def _row_loop(c: _Check, src: _Src, fn: ast.AST) -> tuple[ast.For | None, dict[str, Any], list[str]]:
    """The one `for start in range(0, input_ids.size(0), batch_size)` loop (row chunks, in order) and its row slices
    of input_ids and attention_mask: (loop, {start, stops, ids, att}, missing)."""
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
             and isinstance(n.iter, ast.Call) and _u(n.iter.func) == "range" and len(n.iter.args) == 3
             and _literal(n.iter.args[0]) == 0 and _u(n.iter.args[1]) in ROW_COUNTS
             and _u(n.iter.args[2]) == "batch_size"]
    if len(loops) != 1:
        return None, {}, [f"{len(loops)} loops `for start in range(0, input_ids.size(0), batch_size)` (one is needed: "
                          "row chunks in order)"]
    loop = loops[0]
    s = loop.target.id
    c.keep(src, loop.iter)
    body = _assigns(loop)
    ends = [a for a in body if isinstance(a.targets[0], ast.Name) and _u(a.value).startswith(f"min({s} + batch_size, ")]
    stops = {f"{s} + batch_size"} | {a.targets[0].id for a in ends}
    sliced: dict[str, str] = {}
    for a in body:
        v = a.value
        if (isinstance(a.targets[0], ast.Name) and isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name)
                and v.value.id in ("input_ids", "attention_mask") and isinstance(v.slice, ast.Slice)
                and _u(v.slice.lower) == s and _u(v.slice.upper) in stops and v.slice.step is None):
            sliced.setdefault(v.value.id, a.targets[0].id)
            c.keep(src, a)
    c.keep(src, *ends)
    miss = [f"no `{k}[{s}:{s} + batch_size]` row slice in the loop" for k in ("input_ids", "attention_mask")
            if k not in sliced]
    return loop, {"start": s, "stops": stops, "ids": sliced.get("input_ids"), "att": sliced.get("attention_mask")}, miss


def _one_forward(c: _Check, src: _Src, fn: ast.AST, loop: ast.For, info: Mapping[str, Any]
                 ) -> tuple[ast.Call | None, list[str]]:
    """The loop's one `model(...)` call (none outside it), fed this chunk's rows: input_ids and attention_mask are
    the loop's slices, as keywords or in the dict the call unpacks."""
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and _u(n.func) == "model"]
    inner = [n for n in calls if src.inside(n, loop)]
    if len(inner) != 1 or len(calls) != 1:
        return None, [f"{len(inner)} forward(s) `model(...)` in the row loop and {len(calls) - len(inner)} outside it "
                      "(one per micro-batch is needed)"]
    call = inner[0]
    c.keep(src, src.statement(call))
    want = {"input_ids": info.get("ids"), "attention_mask": info.get("att")}
    star = [k.value for k in call.keywords if k.arg is None]
    inputs = star[0].id if len(star) == 1 and isinstance(star[0], ast.Name) else None
    fed = {k: _u(_kwarg(call, k)) for k in want if _kwarg(call, k) is not None}
    body = _assigns(loop)
    for a in body:
        if inputs and _target(a) == inputs and isinstance(a.value, ast.Dict):
            fed.update({_literal(k): _u(v) for k, v in zip(a.value.keys, a.value.values)
                        if k is not None and _literal(k) in want})
            c.keep(src, a)
    fed.update({k: _u(a.value) for a in body for k in want if inputs and _target(a) == f"{inputs}[{k!r}]"})
    if fed != want:
        return call, [f"the forward does not read this chunk's rows ({fed}, not {want})"]
    return call, []


def _concat_returned(c: _Check, src: _Src, fn: ast.AST, loop: ast.For, lst: str) -> list[str]:
    """`torch.cat(lst, dim=0)` after the loop is what the method returns first."""
    cats = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and _u(n.func) == "torch.cat" and n.args
            and _u(n.args[0]) == lst and not src.inside(n, loop)
            and ((_kwarg(n, "dim") is None and len(n.args) == 1) or _literal(_kwarg(n, "dim")) == 0
                 or (len(n.args) == 2 and _literal(n.args[1]) == 0))]
    if not cats:
        return [f"`torch.cat({lst}, dim=0)` after the loop is missing"]
    firsts_ok = {_u(n) for n in cats}
    for n in cats:
        st = src.statement(n)
        c.keep(src, st)
        if isinstance(st, ast.Assign) and st.value is n and isinstance(st.targets[0], ast.Name):
            firsts_ok.add(st.targets[0].id)
    firsts = []
    for r in (n for n in ast.walk(fn) if isinstance(n, ast.Return)):
        v = r.value
        firsts.append(_u(v.elts[0] if isinstance(v, ast.Tuple) and v.elts else v))
        c.keep(src, r)
    if not firsts or any(f not in firsts_ok for f in firsts):
        return [f"the method does not return the concatenation of {lst} first (it returns {firsts} first)"]
    return []


def _support_form(c: _Check, src: _Src, fn: ast.AST) -> list[str]:
    """What the method lacks of TRL 1.17.1's support_ids form (module docstring, 3); [] when it has it all."""
    miss = _signature(fn, ("batch_size", "sampling_mask"))
    loop, info, more = _row_loop(c, src, fn)
    miss += more
    if loop is None:
        return miss
    s, stops, ids_b, att_b = info["start"], info["stops"], info["ids"], info["att"]
    call, more = _one_forward(c, src, fn, loop, info)
    miss += more
    if call is None:
        return miss
    body = _assigns(loop)
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    star = [k.value for k in call.keywords if k.arg is None]
    inputs = star[0].id if len(star) == 1 and isinstance(star[0], ast.Name) else None
    if _literal(kw.get("fused_lm_head")) is not True:
        miss.append("the forward does not pass fused_lm_head=True")
    if "shift_labels" in kw or any(inputs and _target(a) == f"{inputs}['shift_labels']" for a in body):
        miss.append("the forward is given shift_labels, so the fused head would not shift support_ids")
    lab = kw.get("labels")
    labels = lab.id if isinstance(lab, ast.Name) else None
    if labels is None:
        miss.append("the forward's labels= is not one name")
    else:
        masked = [a for a in body if _target(a) == labels and a.lineno < call.lineno
                  and _u(a.value) == f"{ids_b}.masked_fill({att_b} == 0, -100)"]
        prompt = [a for a in body if _target(a) == f"{labels}[:, :-logits_to_keep]" and _literal(a.value) == -100
                  and a.lineno < call.lineno]
        stray = [a.lineno for a in body if (_target(a) == labels or _target(a).startswith(f"{labels}["))
                 and a not in masked + prompt]
        c.keep(src, *masked, *prompt)
        if not masked:
            miss.append(f"the labels are not `{ids_b}.masked_fill({att_b} == 0, -100)` before the forward")
        if not prompt:
            miss.append(f"`{labels}[:, :-logits_to_keep] = -100` is not set before the forward")
        if stray:
            miss.append(f"the labels are also set at lines {stray}")
    via_key = [a for a in body if inputs and _target(a) == f"{inputs}['support_ids']" and isinstance(a.value, ast.Name)]
    via_kw = kw.get("support_ids")
    sup = via_key[0].value.id if via_key else via_kw.id if isinstance(via_kw, ast.Name) else None
    if sup is None:
        miss.append("support_ids is not passed to the forward")
    else:
        new = [a for a in body if _target(a) == sup
               and _u(a.value) == f"sampling_mask.new_full((*{ids_b}.shape, sampling_mask.size(-1)), -1)"]
        placed = [a for a in body if _target(a) == f"{sup}[:, -logits_to_keep:]"
                  and _u(a.value) in {f"sampling_mask[{s}:{e}]" for e in stops}]
        stray = [a.lineno for a in body if (_target(a) == sup or _target(a).startswith(f"{sup}["))
                 and a not in new + placed]
        c.keep(src, *via_key, *new, *placed)
        if not new:
            miss.append(f"`{sup}` is not `sampling_mask.new_full((*{ids_b}.shape, sampling_mask.size(-1)), -1)` "
                        "(-1 off the completion)")
        if not placed:
            miss.append(f"`{sup}[:, -logits_to_keep:] = sampling_mask[{s}:...]` (this chunk's completion positions) "
                        "is missing")
        if stray:
            miss.append(f"the support is also set at lines {stray}")
        if not all(any(inside and _u(i.test) == "sampling_mask is not None" for i, inside in src.guards(a))
                   for a in via_key + new + placed):
            miss.append("the support is not built under `if sampling_mask is not None:`")
    st = src.statement(call)
    named = isinstance(st, ast.Assign) and st.value is call and isinstance(st.targets[0], ast.Name)
    out = st.targets[0].id if named else None
    c.fields.setdefault("autocast", src.withs(st))
    if out is None:
        return miss + ["the forward's output is not assigned to one name"]
    app = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "append" and len(n.args) == 1
           and _u(n.args[0]) == f"{out}.log_probs[:, -logits_to_keep:]"]
    if len(app) != 1:
        return miss + [f"`<list>.append({out}.log_probs[:, -logits_to_keep:])` is not in the loop once"]
    c.keep(src, src.statement(app[0]))
    return miss + _concat_returned(c, src, fn, loop, _u(app[0].func.value))


def _fill_form(c: _Check, src: _Src, fn: ast.AST) -> list[str]:
    """What the method lacks of D-021's registered logits form (module docstring, 3); [] when it has it all."""
    miss = _signature(fn, ("batch_size",))
    loop, info, more = _row_loop(c, src, fn)
    miss += more
    if loop is None:
        return miss
    call, more = _one_forward(c, src, fn, loop, info)
    miss += more
    if call is None:
        return miss
    body = _assigns(loop)
    star = [k.value for k in call.keywords if k.arg is None]
    inputs = star[0].id if len(star) == 1 and isinstance(star[0], ast.Name) else None
    plus = [a for a in body if inputs and _target(a) == f"{inputs}['logits_to_keep']"
            and _u(a.value) == "logits_to_keep + 1"]
    c.keep(src, *plus)
    if _u(_kwarg(call, "logits_to_keep")) != "logits_to_keep + 1" and not plus:
        miss.append("the forward is not asked for logits_to_keep + 1 rows")
    parent, st = src.parents.get(call), src.statement(call)
    if not (isinstance(parent, ast.Attribute) and parent.attr == "logits" and isinstance(st, ast.Assign)
            and st.value is parent and isinstance(st.targets[0], ast.Name)):
        return miss + ["the forward's `.logits` is not assigned to one name"]
    lg = st.targets[0].id
    drop = [a for a in body if a.lineno > st.lineno and _target(a) == lg and _u(a.value) in (f"{lg}[:, :-1, :]",
                                                                                              f"{lg}[:, :-1]")]
    temp = [a for a in ast.walk(loop) if isinstance(a, ast.stmt) and a.lineno > st.lineno and (
        (isinstance(a, ast.Assign) and _target(a) == lg
         and _u(a.value) in (f"{lg} / self.temperature", f"{lg}.div(self.temperature)"))
        or (isinstance(a, ast.AugAssign) and _u(a.target) == lg and isinstance(a.op, ast.Div)
            and _u(a.value) == "self.temperature"))]
    lsm = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and "log_softmax" in (_call_name(n) or "")
           and ((n.args and _u(n.args[0]) == lg) or (isinstance(n.func, ast.Attribute) and _u(n.func.value) == lg))]
    c.keep(src, *drop, *temp, *(src.statement(n) for n in lsm))
    if not drop:
        miss.append(f"the last row is not dropped (`{lg} = {lg}[:, :-1, :]`)")
    if not temp:
        miss.append(f"`{lg}` is not divided by self.temperature")
    if len(lsm) != 1:
        return miss + [f"{len(lsm)} log-softmax calls of `{lg}` in the loop (one is needed)"]
    if drop and temp and not (lsm[0].lineno > drop[0].lineno and lsm[0].lineno > temp[0].lineno):
        miss.append("the log-softmax comes before the row drop or the temperature")
    holder = src.statement(lsm[0])
    held = holder.targets[0].id if isinstance(holder, ast.Assign) and isinstance(holder.targets[0], ast.Name) else None
    app = [n for n in ast.walk(loop) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr == "append" and len(n.args) == 1
           and (any(x is lsm[0] for x in ast.walk(n.args[0])) or (held is not None and _u(n.args[0]) == held))]
    if len(app) != 1:
        return miss + ["the log-probs are not appended to one list in the loop"]
    c.keep(src, src.statement(app[0]))
    return miss + _concat_returned(c, src, fn, loop, _u(app[0].func.value))


def _t2_fused_head(c: _Check, src: _Src, init: ast.AST) -> None:
    recs = []
    for n in (n for n in ast.walk(init) if isinstance(n, ast.Call) and _call_name(n) == "add_fused_lm_head"):
        c.keep(src, src.statement(n))
        recs.append({"line": n.lineno, "model": _u(n.args[0]) if n.args else _u(_kwarg(n, "model")),
                     **{k: _u(_kwarg(n, k)) or None for k in ("temperature", "outputs", "cast_lm_head_to_fp32")}})
    c.fields["fused_lm_head"] = recs
    c.need(any("self.model" in r["model"] and r["temperature"] == "self.temperature" for r in recs),
           "GRPOTrainer.__init__ does not add the fused LM head to the policy with temperature=self.temperature")


def _t2_fused_forward(c: _Check) -> None:
    us = c.use("trl", TRL_UTILS, "def add_fused_lm_head")
    outer = us.func("add_fused_lm_head")
    if not c.need(outer is not None, f"trl/{us.rel} defines no add_fused_lm_head"):
        return
    inner = us.func("_fused_forward", within=outer)
    c.kept[("trl", us.rel)][outer.lineno] = us.lines[outer.lineno - 1]
    if not c.need(inner is not None and inner is not outer, "add_fused_lm_head defines no _fused_forward"):
        return
    c.keep(us, *inner.args.kwonlyargs)
    kwonly = {a.arg for a in inner.args.kwonlyargs}
    c.need({"fused_lm_head", "labels", "support_ids"} <= kwonly,
           f"_fused_forward's keyword-only parameters {sorted(kwonly)} lack fused_lm_head, labels or support_ids")
    head = [n for n in ast.walk(inner) if isinstance(n, ast.If) and _u(n.test) == "shift_labels is None"]
    nodes = _walk_list(head[0].body if head else inner.body)
    shifted = {k: [a for a in nodes if isinstance(a, ast.Assign) and len(a.targets) == 1 and _target(a) == k
                   and _u(a.value) == f"{k}[:, 1:]"] for k in ("labels", "support_ids")}
    hidden = [a for a in nodes if isinstance(a, ast.Assign) and len(a.targets) == 1
              and _u(a.value).endswith("[:, :-1]")]
    c.keep(us, *[h.test for h in head[:1]], *shifted["labels"], *shifted["support_ids"], *hidden)
    c.need(shifted["labels"], "_fused_forward does not shift the labels (`labels = labels[:, 1:]`)")
    c.need(shifted["support_ids"], "_fused_forward does not shift support_ids with the labels "
                                   "(`support_ids = support_ids[:, 1:]`)")
    c.need(hidden, "_fused_forward does not drop the last hidden state ([:, :-1])")
    applies = [n for n in ast.walk(inner) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "apply" and "ChunkedLogProb" in _u(n.func.value)]
    given = [n for n in applies if "support_ids" in {_u(a) for a in list(n.args) + [k.value for k in n.keywords]}]
    c.keep(us, *(us.statement(n) for n in applies))
    c.need(given, "support_ids is not passed to _ChunkedLogProbFunction.apply")
    c.need(any("temperature" in {_u(a) for a in list(n.args) + [k.value for k in n.keywords]} for n in given),
           "the temperature is not passed to _ChunkedLogProbFunction.apply")
    after = [a for a in _assigns(inner) if _target(a) == "support_ids" and a not in shifted["support_ids"]]
    patch = [a for a in _assigns(outer) if _target(a) == "model.forward"
             and _u(a.value) == f"types.MethodType({inner.name}, model)"]
    c.keep(us, *after, *patch)
    c.need(patch, "add_fused_lm_head does not bind _fused_forward as model.forward")
    originals = {a.targets[0].id for a in _assigns(outer) if isinstance(a.targets[0], ast.Name)
                 and _u(a.value) == "model.forward"}
    plain = [r for n in ast.walk(inner) if isinstance(n, ast.If) and _u(n.test) == "not fused_lm_head"
             for r in _walk_list(n.body) if isinstance(r, ast.Return) and isinstance(r.value, ast.Call)
             and _u(r.value.func) in originals]
    c.keep(us, *plain)
    c.fields["fused_forward"] = {"pre_shifted": [_u(s) for s in head[0].orelse][:2] if head else None,
                                 "support_after_shift": [_u(a)[:200] for a in after],
                                 "kernel": [n.lineno for n in given]}
    c.fields["ordinary_forward"] = {"found": bool(plain), "lines": [r.lineno for r in plain],
                                    "note": "fused_lm_head=False returns the model's own forward (path fill reads "
                                            "its logits)"}
    c.fields["kernel"] = _evidence(c.reader, "trl", TRL_KERNEL,
                                   r"\"inv_t\":|support_log_z =|logprobs = -|The entropy stays", 6)


def _t2_generation_mask(c: _Check) -> bool:
    g = c.use("trl", TRL_VLLM, "class VLLMGeneration")
    gen = c.method(g, c.cls(g, "VLLMGeneration"), "generate")
    if gen is None:
        return False
    alls = [a for a in _assigns(gen) if _target(a) == "all_sampling_mask"]
    sms = [a for a in _assigns(gen) if _target(a) == "sampling_mask"]
    c.keep(g, *alls, *sms)
    guarded = [i.test for a in alls for i, inside in g.guards(a)
               if inside and _u(i.test) == "self.return_sampling_mask"]
    c.keep(g, *guarded)
    loose = [a.lineno for a in alls if not _none(a.value) and not any(
        inside and _u(i.test) == "self.return_sampling_mask" for i, inside in g.guards(a))]
    loose += [a.lineno for a in sms if not _none(a.value) and "all_sampling_mask" not in _names(a.value)]
    if not sms:
        return c.need(False, "VLLMGeneration.generate never sets sampling_mask")
    return c.need(not loose, f"VLLMGeneration.generate builds sampling_mask outside `if self.return_sampling_mask:` "
                             f"(lines {loose})")


def _t2_trl_mask(c: _Check, src: _Src, init: ast.AST, args: Any, from_source: bool) -> None:
    """TRL's own sampling_mask is None at args: top_p, top_k and min_p filter nothing."""
    reads: dict[str, Any] = {}
    rec: dict[str, Any] = {"reads": reads, "filtered_sampling": None, "none_at_args": False}
    c.fields["trl_sampling_mask"] = rec
    n_missing = len(c.missing)
    body = _assigns(init)
    fs = [a for a in body if _target(a) == "filtered_sampling"]
    rsm = [a for a in body if _target(a) == "return_sampling_mask"]
    passed = [n for n in ast.walk(init) if isinstance(n, ast.Call)
              and _u(_kwarg(n, "return_sampling_mask")) == "return_sampling_mask"]
    c.keep(src, *fs, *rsm, *(_kwarg(n, "return_sampling_mask") for n in passed))
    filtered = None
    if fs and rsm and passed:
        rec.update(expression=_u(fs[0].value), return_sampling_mask=_u(rsm[0].value))
        c.need(isinstance(rsm[0].value, ast.BoolOp) and isinstance(rsm[0].value.op, ast.And)
               and any(_u(v) == "filtered_sampling" for v in rsm[0].value.values),
               "return_sampling_mask is not an `and` that includes filtered_sampling")
        try:
            filtered = _evaluate(fs[0].value, _attr_map(init), args, c.reader.defaults(), reads)
        except _Undecidable as err:
            c.need(False, f"filtered_sampling at args: {err}")
    elif from_source:
        c.need(False, "GRPOTrainer.__init__ has no `filtered_sampling = ...`, `return_sampling_mask = ...` and "
                      "VLLMGeneration(..., return_sampling_mask=return_sampling_mask)")
    else:                                           # the logits form (an older TRL): vLLM's candidate set from args
        rec["expression"] = FILTERED_FROM_ARGS
        try:
            filtered = _evaluate(ast.parse(FILTERED_FROM_ARGS, mode="eval").body, {}, args, c.reader.defaults(), reads)
        except _Undecidable as err:
            c.need(False, f"top_p/top_k/min_p at args: {err}")
    if filtered is not None:
        rec["filtered_sampling"] = bool(filtered)
        c.need(not filtered, f"top_p/top_k/min_p at args filter the sampled set ({reads}): TRL builds its own "
                             "sampling_mask")
    if from_source:
        _t2_generation_mask(c)
    rec["none_at_args"] = filtered is not None and not filtered and len(c.missing) == n_missing


def _t2(c: _Check, args: Any) -> None:
    src = c.use("trl", TRL_GRPO, "class GRPOTrainer")
    cls = c.cls(src, "GRPOTrainer")
    fn, init = c.method(src, cls, LOGPS), c.method(src, cls, "__init__")
    if fn is None or init is None:
        return
    support, fill = _support_form(c, src, fn), _fill_form(c, src, fn)
    form = "support_ids" if not support else "fill" if not fill else None
    c.fields.update(form=form, form_missing={"support_ids": support, "fill": fill})
    if form is None:
        c.missing += [f"support_ids form: {m}" for m in support] + [f"fill form: {m}" for m in fill]
    elif form == "support_ids":
        _t2_fused_head(c, src, init)
        _t2_fused_forward(c)
    _t2_trl_mask(c, src, init, args, from_source=form != "fill")
    c.fields["path"] = form if not c.missing else None


# T3 ---------------------------------------------------------------------------------------------------------------
def _bound_is(bounds: Mapping[str, ast.AST], node: ast.AST, which: str) -> bool:
    """`node` is self.vllm_importance_sampling_<which>, or a name set to it (with an infinite fallback when None)."""
    want = f"self.vllm_importance_sampling_{which}"
    v = bounds.get(node.id, node) if isinstance(node, ast.Name) else node
    return _u(v) == want or (isinstance(v, ast.IfExp) and _u(v.body) == want)


def _t3_bound(c: _Check, src: _Src, stmts: list[ast.stmt], r: str) -> str | None:
    """How the taken branch bounds the weight `r`: "mask" (0 outside [clip_min, clip_max]) or "truncate" (clamped)."""
    nodes = _walk_list(stmts)
    assigns = [n for n in nodes if isinstance(n, ast.Assign) and len(n.targets) == 1]
    c.keep(src, *assigns)
    if any(isinstance(n, ast.Raise) for n in nodes):
        c.need(False, "TRL raises at the args' mode")
        return None
    calls = [a for a in assigns if _target(a) == r and isinstance(a.value, ast.Call)]
    fill = [a for a in calls if _u(a.value.func) == f"{r}.masked_fill"]
    clamp = [a for a in calls if _u(a.value.func) == "torch.clamp" and a.value.args and _u(a.value.args[0]) == r]
    bounds = {a.targets[0].id: a.value for a in assigns if isinstance(a.targets[0], ast.Name)}
    if fill:
        call = fill[0].value
        cond = call.args[0] if call.args else None
        zero = _kwarg(call, "value")
        if zero is None and len(call.args) > 1:
            zero = call.args[1]
        c.need(_literal(zero) == 0, "outside the bounds the weight is not set to 0")
        inv = bounds.get(cond.id) if isinstance(cond, ast.Name) else cond
        ok = (isinstance(inv, ast.BinOp) and isinstance(inv.op, ast.BitOr) and isinstance(inv.left, ast.Compare)
              and isinstance(inv.right, ast.Compare) and _u(inv.left.left) == r and isinstance(inv.left.ops[0], ast.Lt)
              and _bound_is(bounds, inv.left.comparators[0], "clip_min") and _u(inv.right.left) == r
              and isinstance(inv.right.ops[0], ast.Gt) and _bound_is(bounds, inv.right.comparators[0], "clip_max"))
        c.need(ok, f"the weights set to 0 are not those below clip_min or above clip_max (`{_u(inv)}`)")
        return "mask"
    if clamp:
        call = clamp[0].value
        c.need(_u(_kwarg(call, "min")) == "self.vllm_importance_sampling_clip_min"
               and _u(_kwarg(call, "max")) == "self.vllm_importance_sampling_clip_max",
               "the weight is not clamped to [clip_min, clip_max]")
        return "truncate"
    c.need(False, f"at the args' mode the weight `{r}` is neither set to 0 outside its bounds nor clamped")
    return None


def _t3(c: _Check, args: Any) -> None:
    src = c.use("trl", TRL_GRPO, "class GRPOTrainer")
    cls = c.cls(src, "GRPOTrainer")
    init, gen, loss = (c.method(src, cls, n) for n in ("__init__", "_generate_and_score_completions", "_compute_loss"))
    if init is None or gen is None or loss is None:
        return
    reads: dict[str, Any] = {}
    c.fields["reads"] = reads
    attrs, defaults = _attr_map(init), c.reader.defaults()

    def ev(node: ast.AST) -> Any:
        return _evaluate(node, attrs, args, defaults, reads)
    body = _assigns(gen)
    diffs = [a for a in body if isinstance(a.targets[0], ast.Name) and isinstance(a.value, ast.BinOp)
             and isinstance(a.value.op, ast.Mult) and isinstance(a.value.right, ast.Name)
             and _u(a.value.left) == "old_per_token_logps - sampling_per_token_logps"]
    if not c.need(len(diffs) == 1, "no single `d = (old_per_token_logps - sampling_per_token_logps) * mask`"):
        return
    da = diffs[0]
    d, m = da.targets[0].id, da.value.right.id
    mask_def = [a for a in body if _target(a) == m and a.lineno < da.lineno][-1:]
    nan = [a for a in body if _target(a) == d and _u(a.value) == f"torch.nan_to_num({d}, nan=0.0)"]
    stray = [a.lineno for a in body if _target(a) == d and a is not da and a not in nan]
    c.keep(src, da, *mask_def, *nan)
    c.need(mask_def and "completion_mask" in _u(mask_def[0].value), f"the mask `{m}` is not built from completion_mask")
    c.need(not stray, f"`{d}` is changed again at lines {stray}")
    levels = [a for a in body if isinstance(a.targets[0], ast.Name) and isinstance(a.value, ast.Compare)
              and _u(a.value.left) == "self.vllm_importance_sampling_mode" and len(a.value.ops) == 1
              and isinstance(a.value.ops[0], ast.In)]
    if not c.need(len(levels) == 1, "no single `level = self.vllm_importance_sampling_mode in [...]`"):
        return
    la = levels[0]
    flag, seq_modes = la.targets[0].id, _literal(la.value.comparators[0])
    c.keep(src, la)
    per_sequence = None
    try:
        per_sequence = bool(ev(la.value))
    except _Undecidable as err:
        c.need(False, f"the IS level at args: {err}")
    exps = [a for a in body if isinstance(a.targets[0], ast.Name) and isinstance(a.value, ast.Call)
            and _u(a.value.func) == "torch.exp" and len(a.value.args) == 1 and isinstance(a.value.args[0], ast.Name)
            and a.lineno > la.lineno]
    if not c.need(len(exps) == 1, "no single `ratio = torch.exp(log_ratio)` after the level"):
        return
    ea = exps[0]
    r, x = ea.targets[0].id, ea.value.args[0].id
    on_seq, on_tok, read = _branch_values(src, gen, flag, x)
    c.keep(src, ea, *read)
    c.need(_u(on_seq) in (f"{d}.sum(dim=-1, keepdim=True)", f"{d}.sum(-1, keepdim=True)"),
           f"per sequence, `{x}` is `{_u(on_seq)}`, not the sum of `{d}` over the completion (keepdim)")
    heads = [n for n in ast.walk(gen) if isinstance(n, ast.If) and "self.vllm_importance_sampling_mode" in _u(n.test)
             and n.lineno > ea.lineno]
    nested = {id(o) for h in heads for o in h.orelse}
    heads = sorted((h for h in heads if id(h) not in nested), key=lambda n: n.lineno)
    kind = None
    if c.need(heads, "no `if self.vllm_importance_sampling_mode in [...]` bounds the weight"):
        node, taken = heads[0], None
        try:
            while taken is None:
                c.keep(src, node.test)
                if ev(node.test):
                    taken = node.body
                elif len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                    node = node.orelse[0]
                else:
                    taken = node.orelse
        except _Undecidable as err:
            c.need(False, f"the bound at args: {err}")
        if taken is not None:
            kind = _t3_bound(c, src, taken, r)
    over = None
    apps = [n for n in ast.walk(gen) if _is_metric_append(n, IS_RATIO_MEAN)]
    if c.need(len(apps) == 1, f"{IS_RATIO_MEAN} is not appended once"):
        c.keep(src, src.statement(apps[0]))
        names = set(_names(apps[0].args[0])) - {"self", "torch"}
        means = [a for a in body if isinstance(a.targets[0], ast.Name) and a.targets[0].id in names
                 and a.lineno < apps[0].lineno]
        mean_of = _mean_operand(means[-1].value) if means else None
        if c.need(mean_of is not None, f"{IS_RATIO_MEAN} is not a plain mean of one tensor"):
            fl_seq, fl_tok, read = _branch_values(src, gen, flag, mean_of)
            c.keep(src, means[-1], *read)
            flat = _u(fl_seq) in (f"{r}.flatten()", f"{r}.view(-1)", f"{r}.reshape(-1)")
            c.need(flat, f"per sequence, the metric's mean is over `{_u(fl_seq)}`, not the flattened (B, 1) weights")
            over = ("sequences" if flat else None) if per_sequence else "tokens" if per_sequence is False else None
    out = [a for a in body if isinstance(a.targets[0], ast.Subscript)
           and _literal(a.targets[0].slice) == "importance_sampling_ratio" and _u(a.value) == r]
    applied = [a for a in _assigns(loss) if _target(a) == "per_token_loss" and isinstance(a.value, ast.BinOp)
               and isinstance(a.value.op, ast.Mult) and _u(a.value.left) == "per_token_loss"
               and _inputs_key(a.value.right) == "importance_sampling_ratio"]
    c.keep(src, *out, *applied)
    c.need(out, f"the output does not carry `{r}` as importance_sampling_ratio")
    at_args = None
    if c.need(applied, "_compute_loss does not multiply per_token_loss by inputs['importance_sampling_ratio']"):
        c.keep(src, *(i.test for i, _ in src.guards(applied[0])))
        try:
            at_args = _holds(src, applied[0], ev)
            c.need(at_args, "at args, _compute_loss does not apply the weight")
        except _Undecidable as err:
            c.need(False, f"whether the loss applies the weight at args: {err}")
    settings: dict[str, Any] = {}
    for name in ("use_vllm", "vllm_importance_sampling_correction", "vllm_importance_sampling_mode",
                 "vllm_importance_sampling_clip_min", "vllm_importance_sampling_clip_max",
                 "vllm_importance_sampling_cap"):
        try:
            settings[name] = _read_arg(name, args, defaults, reads)
        except _Undecidable as err:
            settings[name] = _MISSING
            if name != "vllm_importance_sampling_cap":      # an older GRPOConfig has no cap: nothing to override
                c.need(False, f"{name} cannot be read: {err}")
    mode = settings["vllm_importance_sampling_mode"]
    c.need(settings["use_vllm"] is True, "use_vllm is not true at args")
    c.need(settings["vllm_importance_sampling_correction"] is True, "TRL's IS correction is off at args")
    c.need(settings["vllm_importance_sampling_cap"] in (None, _MISSING),
           "vllm_importance_sampling_cap is set: GRPOConfig.__post_init__ would replace clip_max with it")
    c.need(per_sequence is not False, f"the mode at args ({mode!r}) is not a sequence mode {seq_modes}")
    if kind is not None and isinstance(mode, str):
        c.need(kind == ("mask" if mode.endswith("_mask") else "truncate"),
               f"at {mode!r} the branch taken {'sets the weight to 0' if kind == 'mask' else 'clamps the weight'}, "
               "not what that mode names")
    c.fields.update(weight=f"exp(sum over the completion of {_u(da.value)})", nan_to_zero=bool(nan),
                    level="sequence" if per_sequence else "token" if per_sequence is False else None,
                    sequence_modes=seq_modes, mode=_jsonable(mode) if mode is not _MISSING else None,
                    clip_min=_jsonable(settings["vllm_importance_sampling_clip_min"])
                    if settings["vllm_importance_sampling_clip_min"] is not _MISSING else None,
                    clip_max=_jsonable(settings["vllm_importance_sampling_clip_max"])
                    if settings["vllm_importance_sampling_clip_max"] is not _MISSING else None,
                    branch=kind, ratio_mean_over=over, applied_in_loss=at_args)


# T4 ---------------------------------------------------------------------------------------------------------------
def _t4(c: _Check, args: Any) -> None:
    src = c.use("trl", TRL_GRPO, "class GRPOTrainer")
    cls = c.cls(src, "GRPOTrainer")
    init, loss = c.method(src, cls, "__init__"), c.method(src, cls, "_compute_loss")
    if init is None or loss is None:
        return
    reads: dict[str, Any] = {}
    c.fields["reads"] = reads
    attrs, defaults = _attr_map(init), c.reader.defaults()
    body = _assigns(loss)
    lps = sorted({a.targets[0].elts[0].id for a in body if isinstance(a.targets[0], ast.Tuple) and a.targets[0].elts
                  and isinstance(a.targets[0].elts[0], ast.Name) and _is_logps_call(a.value)})
    refs = [a for a in body if isinstance(a.targets[0], ast.Name) and _inputs_key(a.value) == "ref_per_token_logps"]
    if not (c.need(len(lps) == 1, f"_compute_loss's log-probs are not one name from self.{LOGPS}(...)")
            & c.need(refs, "_compute_loss does not read inputs['ref_per_token_logps']")):
        return
    lp, ref = lps[0], refs[0].targets[0].id
    other = [st.lineno for st in _bindings(src, loss, ref)
             if not (isinstance(st, ast.Assign) and _inputs_key(st.value) == "ref_per_token_logps")]
    c.need(not other, f"`{ref}` is also set at lines {other}")
    want = f"torch.exp({ref} - {lp}) - ({ref} - {lp}) - 1"
    kls = [a for a in body if isinstance(a.targets[0], ast.Name) and _u(a.value) == want]
    c.keep(src, *refs, *kls)
    if not c.need(len(kls) == 1, f"no single KL `{want}`"):
        return
    kl, bias = kls[0].targets[0].id, None
    for st in _bindings(src, loss, kl):
        if st is kls[0]:
            continue
        c.keep(src, st)
        times = (isinstance(st, ast.Assign) and isinstance(st.value, ast.BinOp) and isinstance(st.value.op, ast.Mult)
                 and _u(st.value.left) == kl and isinstance(st.value.right, ast.Name))
        under = [i for i, inside in src.guards(st) if inside and "use_bias_correction_kl" in _u(i.test)]
        if times and under and bias is None:
            defs = [a for a in body if _target(a) == st.value.right.id and a.lineno < st.lineno]
            c.keep(src, under[-1].test, *defs[-1:])
            bias = {"line": st.lineno, "times": st.value.right.id, "under": _u(under[-1].test),
                    "defined": _u(defs[-1].value) if defs else None}
        else:
            c.need(False, f"`{kl}` is changed at line {st.lineno} by something other than the bias correction")
    in_loss = [a for a in body if _target(a) == "per_token_loss"
               and _u(a.value) == f"per_token_loss + self.beta * {kl}"]
    c.keep(src, *in_loss)
    c.need(in_loss, f"the loss does not add self.beta * {kl}")
    for name in ("use_bias_correction_kl", "beta"):                     # recorded, not gating
        try:
            _read_arg(name, args, defaults, reads)
        except _Undecidable as err:
            reads[name] = {"value": None, "from": str(err)}
    try:
        kl_on = _holds(src, kls[0], lambda node: _evaluate(node, attrs, args, defaults, reads))
    except _Undecidable:
        kl_on = None
    c.fields.update(kl=want, ref_from="inputs['ref_per_token_logps']", logp_from=f"{lp}, the loss pass's (T1)",
                    bias_correction=bias, use_bias_correction_kl=reads.get("use_bias_correction_kl"),
                    beta=reads.get("beta"), kl_on_at_args=kl_on)


# T5 ---------------------------------------------------------------------------------------------------------------
def _t5(c: _Check, args: Any) -> None:
    pt = c.use("peft", PEFT_TUNERS, "def _maybe_include_all_linear_layers")
    fn = pt.func("_maybe_include_all_linear_layers")
    if not c.need(fn is not None, f"peft/{pt.rel} defines no _maybe_include_all_linear_layers"):
        return
    c.kept[("peft", pt.rel)][fn.lineno] = pt.lines[fn.lineno - 1]
    short = None
    for n in (n for n in ast.walk(fn) if isinstance(n, ast.Compare) and len(n.ops) == 1
              and isinstance(n.ops[0], ast.Eq) and _u(n.left).endswith("target_modules.lower()")):
        c.keep(pt, n)
        short = _literal(n.comparators[0])
        if short is _MISSING and isinstance(n.comparators[0], ast.Name):
            const = c.use("peft", PEFT_CONSTANTS)
            hits = [a for a in _assigns(const.tree) if _target(a) == n.comparators[0].id]
            c.keep(const, *hits[-1:])
            short = _literal(hits[-1].value) if hits else None
    c.need(short == "all-linear", f"the all-linear shorthand compared is {short!r}, not 'all-linear'")
    body = _assigns(fn)
    emb = [a for a in body if isinstance(a.targets[0], ast.Name) and _u(a.value) == "model.get_output_embeddings()"]
    if not c.need(emb, "the output embedding (model.get_output_embeddings()) is never read"):
        return
    e = emb[0].targets[0].id
    named = [a for a in body if isinstance(a.targets[0], ast.Name) and isinstance(a.value, ast.Call)
             and _u(a.value.func) == "next" and any(isinstance(x, ast.Compare) and isinstance(x.ops[0], ast.Is)
                                                    and _u(x.comparators[0]) == e for x in ast.walk(a.value))]
    nm = named[0].targets[0].id if named else None
    adds = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "add" and nm and n.args and _u(n.args[0]) == nm]
    excl = _u(adds[0].func.value) if adds else None

    def subtracts(n: ast.AST) -> bool:      # `lin -= excl` or `lin = lin - excl`
        if isinstance(n, ast.AugAssign):
            return isinstance(n.op, ast.Sub) and _u(n.value) == excl
        return (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.value, ast.BinOp)
                and isinstance(n.value.op, ast.Sub) and _u(n.value.right) == excl
                and _u(n.value.left) == _target(n))
    subs = sorted((n for n in ast.walk(fn) if excl and subtracts(n)), key=lambda n: n.lineno)
    lin = _u(subs[0].target if isinstance(subs[0], ast.AugAssign) else subs[0].targets[0]) if subs else None
    sets = [a for a in body if lin and _target(a).endswith(".target_modules") and _u(a.value) == lin
            and a.lineno > subs[0].lineno]
    c.keep(pt, emb[0], *named, *(pt.statement(n) for n in adds), *subs, *sets)
    c.need(nm, f"the output embedding's module name is not found (`next(... if module is {e})`)")
    c.need(excl, "the output embedding's name is not added to an exclusion set")
    c.need(subs, "the exclusion set is not subtracted from the linear layers")
    c.need(sets, "the targets left after the subtraction are not set as target_modules")
    guards = [_u(i.test) for i, inside in pt.guards(emb[0]) if inside]
    c.keep(pt, *(i.test for i, inside in pt.guards(emb[0]) if inside))
    called = [n for n in ast.walk(pt.tree) if isinstance(n, ast.Call) and _call_name(n) == fn.name
              and not pt.inside(n, fn)]
    callers = []
    for n in called:
        c.keep(pt, pt.statement(n))
        outer = next((a for a in pt.ancestors(n) if isinstance(a, _FUNCS)), None)
        callers.append(f"{getattr(outer, 'name', '<module>')}:{n.lineno}")
    c.need(called, f"{fn.name} is never called in peft/{pt.rel}")
    c.fields.update(function=fn.name, shorthand=short, excludes="model.get_output_embeddings()", guard=guards,
                    called_from=callers, run_time="lofi_rl.rl.trl_probe.lm_head_record(trainer.model)")


# V1 ---------------------------------------------------------------------------------------------------------------
def _v1_runner(c: _Check, label: str, src: _Src) -> dict[str, Any]:
    """In this runner: the method that calls apply_grammar_bitmask( on `logits` samples from them after it."""
    fns = [f for f in ast.walk(src.tree) if isinstance(f, _FUNCS)
           and any(isinstance(n, ast.Call) and _call_name(n) == "apply_grammar_bitmask" for n in ast.walk(f))]
    fns = [f for f in fns if not any(g is not f and src.inside(g, f) for g in fns)]
    rec: dict[str, Any] = {"file": f"vllm/{src.rel}", "methods": []}
    if not c.need(fns, f"runner {label}: no method calls apply_grammar_bitmask(...)"):
        return rec
    for f in fns:
        for bm in (n for n in ast.walk(f) if isinstance(n, ast.Call) and _call_name(n) == "apply_grammar_bitmask"):
            on = "logits" in {_u(a) for a in list(bm.args) + [k.value for k in bm.keywords]}
            samples = sorted((n for n in ast.walk(f) if isinstance(n, ast.Call)
                              and _u(n.func) in ("self._sample", "self.sampler") and _first_arg_is(n, "logits")),
                             key=lambda n: n.lineno)
            after = [n for n in samples if n.lineno > bm.lineno]
            rebound = [n.lineno for n in ast.walk(f) if isinstance(n, ast.Name) and n.id == "logits"
                       and isinstance(n.ctx, ast.Store) and after and bm.lineno < n.lineno < after[0].lineno]
            c.keep(src, bm, *after[:1])
            rec["methods"].append({"method": f.name, "bitmask_line": bm.lineno, "on_logits": on,
                                   "sampled_at": [n.lineno for n in samples]})
            c.need(on, f"runner {label}: {f.name} applies the bitmask at line {bm.lineno}, not to `logits`")
            c.need(after, f"runner {label}: {f.name} does not sample from `logits` after the bitmask")
            c.need(not [n for n in samples if n.lineno < bm.lineno],
                   f"runner {label}: {f.name} samples before the bitmask")
            c.need(not rebound, f"runner {label}: `logits` is rebound at {rebound} between the bitmask and sampling")
            if after and _u(after[0].func) == "self._sample":       # v1: _sample hands the logits to the sampler
                owner = src.parents.get(f)
                smp = src.func("_sample", within=owner) if isinstance(owner, ast.ClassDef) else None
                hand = [n for n in ast.walk(smp) if isinstance(n, ast.Call) and _u(n.func) == "self.sampler"
                        and _first_arg_is(n, "logits")] if smp is not None else []
                c.keep(src, *hand[:1])
                c.need(hand, f"runner {label}: _sample does not call self.sampler(logits=logits, ...)")
    return rec


def _v1_inplace(c: _Check, label: str) -> None:
    src = c.use("vllm", VLLM_INPLACE[label])
    if label == "v1":
        fn = src.func("apply_grammar_bitmask")
        hits = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and _call_name(n) == "apply_token_bitmask_inplace"
                and n.args and _u(n.args[0]) == "logits"] if fn is not None else []
        c.keep(src, *hits)
        c.need(hits, f"runner {label}: apply_grammar_bitmask does not call apply_token_bitmask_inplace(logits, ...)")
        return
    worker = src.cls("StructuredOutputsWorker")
    fn = src.func("apply_grammar_bitmask", within=worker) if worker is not None else None
    launches = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)
                and n.args and _u(n.args[0]) == "logits"] if fn is not None else []
    kernels = [src.func(_u(n.func.value)) for n in launches]
    stores = [n for k in kernels if k is not None for n in ast.walk(k) if isinstance(n, ast.Call)
              and _u(n.func) == "tl.store" and "-float('inf')" in {_u(a) for a in n.args}]
    c.keep(src, *launches[:1], *stores[:1])
    c.need(stores, f"runner {label}: StructuredOutputsWorker.apply_grammar_bitmask does not write -inf into `logits`")


def _v1_samplers(c: _Check, label: str) -> None:
    src = c.use("vllm", VLLM_SAMPLERS[label])
    cls = c.cls(src, "Sampler")
    if label == "v1":
        sample, logprobs = c.method(src, cls, "sample"), c.method(src, cls, "compute_logprobs")
        if sample is None or logprobs is None:
            return
        greedy = [a for n in ast.walk(sample) if isinstance(n, ast.If) and _is_mode_test(n.test, "processed_logprobs")
                  for a in _walk_list(n.body) if isinstance(a, ast.Assign) and len(a.targets) == 1
                  and _target(a) == "processed_logprobs" and _u(a.value) == "self.compute_logprobs(logits)"]
        rand = [a for a in _assigns(sample) if isinstance(a.targets[0], ast.Tuple)
                and "processed_logprobs" in {_u(e) for e in a.targets[0].elts}
                and isinstance(a.value, ast.Call) and _u(a.value.func) == "self.topk_topp_sampler"]
        soft = [r for r in ast.walk(logprobs) if isinstance(r, ast.Return)
                and _u(r.value) == "logits.log_softmax(dim=-1, dtype=torch.float32)"]
        c.keep(src, *greedy, *rand, *soft)
        c.need(greedy, "sampler v1: no `processed_logprobs = self.compute_logprobs(logits)` under the processed mode")
        c.need(rand, "sampler v1: the random branch does not take processed_logprobs from self.topk_topp_sampler")
        c.need(soft, "sampler v1: compute_logprobs is not logits.log_softmax(dim=-1, dtype=torch.float32)")
        tk = c.use("vllm", VLLM_TOPK)
        native = c.method(tk, c.cls(tk, "TopKTopPSampler"), "forward_native")
        if native is not None:
            branch = [a for n in ast.walk(native) if isinstance(n, ast.If)
                      and _is_mode_test(n.test, "processed_logprobs")
                      for a in _walk_list(n.body) if isinstance(a, ast.Assign) and len(a.targets) == 1
                      and _u(a.value) == "logits.log_softmax(dim=-1, dtype=torch.float32)"]
            c.keep(tk, *branch)
            c.need(branch, "top-k/top-p sampler: forward_native has no processed-logprobs log_softmax")
        return
    call = c.method(src, cls, "__call__")
    if call is None:
        return
    took = [a for a in _assigns(call) if isinstance(a.targets[0], ast.Tuple) and isinstance(a.value, ast.Call)
            and _u(a.value.func) == "self.sample" and _first_arg_is(a.value, "logits")
            and "processed_logits" in {_u(e) for e in a.targets[0].elts}]
    swap = [a for n in ast.walk(call) if isinstance(n, ast.If)
            and _u(n.test) == "self.logprobs_mode in PROCESSED_LOGPROBS_MODES"
            for a in _walk_list(n.body) if isinstance(a, ast.Assign) and len(a.targets) == 1 and _target(a) == "logits"
            and _u(a.value) == "processed_logits"]
    c.keep(src, *took, *swap)
    c.need(took, "sampler v2: __call__ does not take processed_logits from self.sample(logits, ...)")
    c.need(swap, "sampler v2: __call__ does not score the processed logits in the processed modes")
    mc = c.use("vllm", VLLM_MODEL_CONFIG)
    modes = [a for a in ast.walk(mc.tree) if isinstance(a, (ast.Assign, ast.AnnAssign))
             and _u(a.targets[0] if isinstance(a, ast.Assign) else a.target) == "PROCESSED_LOGPROBS_MODES"]
    held = _literal(modes[0].value) if modes else _MISSING
    c.keep(mc, *modes)
    c.need(isinstance(held, (tuple, list, set)) and "processed_logprobs" in held,
           "PROCESSED_LOGPROBS_MODES does not hold processed_logprobs")


def _is_mode_test(test: ast.AST, mode: str) -> bool:
    return (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
            and _u(test.left).endswith("logprobs_mode") and _literal(test.comparators[0]) == mode)


def _v1_default(c: _Check) -> dict[str, Any]:
    """Which runner vLLM builds by default: config/vllm.py's use_v2_model_runner (recorded)."""
    out: dict[str, Any] = {"env_VLLM_USE_V2_MODEL_RUNNER": os.environ.get("VLLM_USE_V2_MODEL_RUNNER")}
    try:
        cfg = c.use("vllm", VLLM_CONFIG)
    except _NotFound as err:
        return {**out, "error": str(err)}
    fn = next((n for n in ast.walk(cfg.tree) if isinstance(n, _FUNCS) and n.name == "use_v2_model_runner"), None)
    if fn is None:
        return {**out, "default": "v1", "note": "config/vllm.py has no use_v2_model_runner"}
    rets = [r for r in ast.walk(fn) if isinstance(r, ast.Return)]
    env = [n for n in ast.walk(fn) if isinstance(n, ast.Attribute) and n.attr == "VLLM_USE_V2_MODEL_RUNNER"]
    c.keep(cfg, *rets, *(cfg.statement(n) for n in env[:1]))
    last = fn.body[-1] if fn.body else None
    out["default"] = "v2" if isinstance(last, ast.Return) and _literal(last.value) is True else None
    out["rule"] = ("v2 unless VLLM_USE_V2_MODEL_RUNNER says otherwise, Triton is absent or a feature v2 lacks is asked "
                   "for (config/vllm.py, use_v2_model_runner)")
    return out


def _v1(c: _Check, args: Any) -> None:
    runners: dict[str, Any] = {}
    for label, rel in VLLM_RUNNERS.items():
        try:
            src = c.use("vllm", rel)
        except _NotFound:
            if label == "v1":
                raise
            runners[label] = {"file": f"vllm/{rel}", "absent": True}
            continue
        runners[label] = _v1_runner(c, label, src)
        _v1_inplace(c, label)
        _v1_samplers(c, label)
    g = c.use("trl", TRL_VLLM, "class VLLMGeneration")
    llms = [n for n in ast.walk(g.tree) if isinstance(n, ast.Call) and _u(n.func) == "LLM"]
    modes = [_literal(_kwarg(n, "logprobs_mode")) for n in llms]
    c.keep(g, *(_kwarg(n, "logprobs_mode") for n in llms))
    c.need("processed_logprobs" in modes, f"TRL builds LLM(...) with logprobs_mode {modes}, not processed_logprobs")
    c.fields.update(gating=False, runners=runners, runner_default=_v1_default(c),
                    trl_logprobs_mode=[m if m is not _MISSING else None for m in modes])


# V2 ---------------------------------------------------------------------------------------------------------------
def _call_record(call: ast.Call) -> dict[str, Any]:
    return {"call": _u(call.func), "args": [_u(a) for a in call.args],
            "kwargs": {k.arg: (_literal(k.value) if _literal(k.value) is not _MISSING else _u(k.value))
                       for k in call.keywords if k.arg}}


def _v2(c: _Check, args: Any) -> None:
    be = c.use("vllm", VLLM_BACKEND)
    cls = c.cls(be, "XgrammarBackend")
    post, comp = c.method(be, cls, "__post_init__"), c.method(be, cls, "compile_grammar")
    if post is None or comp is None:
        return
    info = [n for n in ast.walk(post) if isinstance(n, ast.Call)
            and _u(n.func).endswith("TokenizerInfo.from_huggingface")]
    compiler = [n for n in ast.walk(post) if isinstance(n, ast.Call) and _u(n.func).endswith("GrammarCompiler")]
    regex = [n for r in ast.walk(comp) if isinstance(r, ast.If) and _u(r.test).endswith(".REGEX")
             for n in _walk_list(r.body) if isinstance(n, ast.Call)
             and "compile_regex" in {_u(n.func).rsplit(".", 1)[-1]} | {_u(a).rsplit(".", 1)[-1] for a in n.args}]
    matcher = [n for n in ast.walk(comp) if isinstance(n, ast.Call) and _u(n.func).endswith("GrammarMatcher")]
    c.keep(be, *(be.statement(n) for n in info + compiler + regex[:1]), *matcher)
    c.need(info and _kwarg(info[0], "vocab_size") is not None,
           "TokenizerInfo.from_huggingface(tokenizer, vocab_size=...) is not built")
    c.need(compiler, "no GrammarCompiler is built")
    c.need(regex, "the REGEX branch of compile_grammar does not compile with compile_regex")
    c.need(matcher and _kwarg(matcher[0], "override_stop_tokens") is not None,
           "GrammarMatcher(..., override_stop_tokens=...) is not built")
    mgr = c.use("vllm", VLLM_MANAGER)
    body = _assigns(mgr.tree)
    vocab = {a.targets[0].id for a in body if isinstance(a.targets[0], ast.Name)
             and _u(a.value) == "self.vllm_config.model_config.get_vocab_size()"}
    built = [n for n in ast.walk(mgr.tree) if isinstance(n, ast.Call) and _u(n.func) == "XgrammarBackend"
             and _u(_kwarg(n, "vocab_size")) in vocab]
    stops = [a for a in body if isinstance(a.targets[0], ast.Name)
             and "request.sampling_params.all_stop_token_ids" in _u(a.value)]
    passed = [n for n in ast.walk(mgr.tree) if isinstance(n, ast.Call) and _u(n.func).endswith(".compile_grammar")
              and stops and _u(_kwarg(n, "stop_token_ids")) in {a.targets[0].id for a in stops}]
    c.keep(mgr, *(a for a in body if isinstance(a.targets[0], ast.Name) and a.targets[0].id in vocab),
           *(mgr.statement(n) for n in built), *stops, *(mgr.statement(n) for n in passed))
    c.need(built, "XgrammarBackend is not built with vocab_size = self.vllm_config.model_config.get_vocab_size()")
    c.need(passed, "compile_grammar is not given stop_token_ids = request.sampling_params.all_stop_token_ids")
    c.fields.update(gating=False,
                    tokenizer_info=_call_record(info[0]) if info else None,
                    compiler=_call_record(compiler[0]) if compiler else None,
                    compile_regex=_u(regex[0]) if regex else None,
                    matcher=_call_record(matcher[0]) if matcher else None,
                    vocab_size_from="self.vllm_config.model_config.get_vocab_size()" if built else None,
                    stop_ids_from=_u(stops[0].value) if stops else None,
                    stop_ids_evidence=_evidence(c.reader, "vllm", VLLM_PARAMS,
                                                r"_all_stop_token_ids\.(add|update)|def all_stop_token_ids", 6))


CHECKS: dict[str, tuple[str, tuple[str, str], Any]] = {
    "T1": ("the old, reference and loss log-probs come only from self._get_per_token_logps_and_entropies, and "
           "use_liger_kernel is false", ("trl", TRL_GRPO), _t1),
    "T2": ("that method's body, and the path the grammar mask takes through it (support_ids or fill); TRL's own "
           "sampling_mask is None at args", ("trl", TRL_GRPO), _t2),
    "T3": ("the per-sequence IS weight, its sequence_mask bound, and what sampling/importance_sampling_ratio/mean "
           "averages", ("trl", TRL_GRPO), _t3),
    "T4": ("the KL is exp(ref - logp) - (ref - logp) - 1 of the reference and loss log-probs; use_bias_correction_kl "
           "recorded", ("trl", TRL_GRPO), _t4),
    "T5": ("PEFT's all-linear targets exclude the output embedding (lm_head_record checks the built model)",
           ("peft", PEFT_TUNERS), _t5),
    "V1": ("vLLM applies the grammar bitmask to the logits before it samples and takes processed log-probs from them "
           "(recorded)", ("vllm", VLLM_RUNNERS["v1"]), _v1),
    "V2": ("how vLLM's xgrammar backend builds its TokenizerInfo, compiler and matcher, with which vocab size and stop "
           "ids (recorded)", ("vllm", VLLM_BACKEND), _v2),
}


def _run(name: str, reader: _Reader, args: Any) -> dict[str, Any]:
    what, main, fn = CHECKS[name]
    c = _Check(what, reader, main)
    try:
        fn(c, args)
        return c.record()
    except _NotFound as err:
        verdict, extra = "not_found", {"error": str(err)[:500]}
    except Exception as err:  # noqa: BLE001: a record, never a reason to lose the run
        verdict = "error"
        extra = {"error": f"{type(err).__name__}: {err}"[:500], "trace": traceback.format_exc()[-2000:]}
    try:
        rec = c.record()
    except Exception:  # noqa: BLE001
        rec = {"check": what, "package": main[0], "file": main[1], "sha256": None, "lines": {}, "missing": []}
    rec.update(verdict=verdict, **extra)
    return rec


def source_checks(*, trl_base: Path | None = None, vllm_base: Path | None = None, peft_base: Path | None = None,
                  args: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """D-021 Part A's source checks of the installed TRL, vLLM and PEFT (module docstring, 3), read where package_dir
    finds them unless a base is given; `args` = the resolved GRPOConfig kwargs. Never raises:
    {"T1": rec, ..., "T5": rec, "V1": rec, "V2": rec, "path": "support_ids" | "fill" | None,
     "gating": ["T1", ..., "T5"], "verdict": "match" | "not_matched", "failed": [the gating checks not "match"],
     "packages": {name: {root, version}}, "args_given", "seconds"}."""
    t0 = time.perf_counter()
    out: dict[str, Any] = {}
    try:
        bases = {"trl": trl_base if trl_base is not None else package_dir("trl"),
                 "vllm": vllm_base if vllm_base is not None else package_dir("vllm"),
                 "peft": peft_base if peft_base is not None else package_dir("peft")}
        out["packages"] = {k: {"root": str(v) if v is not None else None, "version": _version(k)}
                           for k, v in bases.items()}
        reader = _Reader(bases)
        for name in GATING + RECORDED:
            out[name] = _run(name, reader, args)
    except Exception as err:  # noqa: BLE001: nothing above should raise; if it does, nothing passes
        out["error"] = f"{type(err).__name__}: {err}"[:500]
        for name in GATING + RECORDED:
            out.setdefault(name, {"check": CHECKS[name][0], "verdict": "error", "error": out["error"], "missing": []})
    t2 = out.get("T2", {})
    out["path"] = t2.get("path") if t2.get("verdict") == "match" else None
    out["gating"] = list(GATING)
    out["failed"] = [n for n in GATING if out.get(n, {}).get("verdict") != "match"]
    out["verdict"] = "not_matched" if out["failed"] else "match"
    out["args_given"] = args is not None
    out["seconds"] = round(time.perf_counter() - t0, 3)
    return out


def lm_head_record(model: Any, *, tuner_layer: type | None = None) -> dict[str, Any]:
    """T5's run-time half (D-021 Part A), on the model the trainer built (trainer.model, the PEFT model): its output
    embedding is not a PEFT tuner layer (peft.tuners.tuners_utils.BaseTunerLayer), no tuner layer's name ends in
    lm_head, no trainable parameter's name holds lm_head, no adapter targets or saves a module ending in lm_head, and
    the model holds at least one tuner layer (so it is the PEFT model). "match", "not_matched" (with what is
    missing) or "error"; never raises. `tuner_layer` stands in for BaseTunerLayer in tests (PEFT is imported here,
    in the policy env, only when it is None)."""
    rec: dict[str, Any] = {"check": "T5 (run time): the built model's output embedding is not a LoRA target",
                           "verdict": None, "output_embedding": None, "output_embedding_is_tuner_layer": None,
                           "tuner_layers": None, "lm_head_tuner_layers": [], "trainable_lm_head_parameters": [],
                           "adapters": {}, "missing": [], "error": None}
    missing = rec["missing"]
    try:
        if tuner_layer is None:
            from peft.tuners.tuners_utils import BaseTunerLayer as tuner_layer
        m = model.module if not hasattr(model, "get_output_embeddings") and hasattr(model, "module") else model
        out = m.get_output_embeddings()
        if out is None:
            missing.append("the model has no output embedding")
        else:
            rec["output_embedding"] = f"{type(out).__module__}.{type(out).__qualname__}"
            rec["output_embedding_is_tuner_layer"] = isinstance(out, tuner_layer)
            if isinstance(out, tuner_layer):
                missing.append(f"the output embedding is a tuner layer ({rec['output_embedding']})")
        tuned = [n for n, mod in m.named_modules() if isinstance(mod, tuner_layer)]
        rec["tuner_layers"] = len(tuned)
        rec["lm_head_tuner_layers"] = [n for n in tuned if n.endswith("lm_head")][:20]
        rec["trainable_lm_head_parameters"] = [n for n, p in m.named_parameters()
                                               if getattr(p, "requires_grad", False) and "lm_head" in n][:20]
        if not tuned:
            missing.append("the model holds no tuner layer: it is not the PEFT model")
        if rec["lm_head_tuner_layers"]:
            missing.append(f"tuner layers named *lm_head: {rec['lm_head_tuner_layers']}")
        if rec["trainable_lm_head_parameters"]:
            missing.append(f"trainable parameters named *lm_head*: {rec['trainable_lm_head_parameters']}")
        for adapter, cfg in dict(getattr(m, "peft_config", None) or {}).items():
            entry: dict[str, Any] = {}
            for key in ("target_modules", "modules_to_save"):
                v = getattr(cfg, key, None)
                names = sorted(map(str, v)) if isinstance(v, (set, frozenset, list, tuple)) else v
                hits = ([t for t in names if t.endswith("lm_head")] if isinstance(names, list)
                        else [names] if isinstance(names, str) and "lm_head" in names else [])
                entry[key] = {"count": len(names), "lm_head": hits} if isinstance(names, list) else \
                    {"value": _jsonable(names), "lm_head": hits}
                if hits:
                    missing.append(f"adapter {adapter!r} {key} names lm_head: {hits}")
            rec["adapters"][str(adapter)] = entry
        rec["verdict"] = "not_matched" if missing else "match"
    except Exception as err:  # noqa: BLE001: a record; train.py stops the launch on anything but match
        rec.update(verdict="error", error=f"{type(err).__name__}: {err}"[:500])
    return rec
