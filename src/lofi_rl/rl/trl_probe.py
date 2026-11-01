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

Standard library and lofi_rl.rl.junk only: vLLM is imported inside BackendPin.for_vllm, in the policy env.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import importlib.metadata
import importlib.util
import re
import traceback
import types
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
