"""The step-0 package in the policy env (docs/design/step0_samples.md §2.2, §6.1), and the junk names it reuses.

- Every module of lofi_rl.step0 imports in a child interpreter with tests/test_rl_imports.py's blocked list (the audio
  stack the policy env lacks and the ML stack the laptop lacks) and numpy blocked: the policy env's build imports every
  lofi_rl module outside its scorer-side list (jobs/env_policy.sbatch), step0.render among them, so vLLM is imported
  inside functions (through junk) and render's audio code too. The standard-library modules (the package, requests,
  select, path, cli, __main__) import with yaml blocked as well.
- step0 reuses junk by import, public names only (§2.2), and junk stays byte-identical; the names it reads exist with
  the values it expects (SAMPLING, STRUCTURED_BACKEND "xgrammar", DEFAULT_CHUNK 128, HANG_MARGIN_S 90, RENDER_JOBS 2,
  the signatures of load_engine, render_clip, score_row and summarize, Completion's fields).
Check and prereg, the laptop builder's modules, are checked once they exist.
"""
from __future__ import annotations

import ast
import inspect
import os
import pkgutil
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import lofi_rl.step0 as S
from lofi_rl import prompt_v2
from lofi_rl.rl import junk as J

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "lofi_rl" / "step0"
BLOCKED = ("mido", "soundfile", "scipy", "pedalboard", "torch", "trl", "vllm", "transformers", "peft", "datasets",
           "accelerate", "numpy", "torchaudio", "laion_clap", "audiobox_aesthetics", "pyloudnorm", "pretty_midi")
STDLIB = ("lofi_rl.step0", "lofi_rl.step0.requests", "lofi_rl.step0.select", "lofi_rl.step0.path",
          "lofi_rl.step0.cli", "lofi_rl.step0.__main__")
# §2.2's public junk names, plus two more public ones: cell_brief (a cell's stand-in brief, whose messages every brief
# of the cell shares) and RENDER_JOBS (the renders at a time junk.render_lanes splits --workers between, D-022)
JUNK_NAMES = {"SAMPLING", "STRUCTURED_BACKEND", "PROMPTS", "split_model", "load_engine", "VllmEngine",
              "structured_outputs_params", "constrained_block", "backend_record", "prompt_record", "example_id",
              "brief_dict", "sha256_text", "read_jsonl", "Completion", "CopyCheck", "score_row", "summarize",
              "render_clip", "render_lanes", "in_order", "grams", "jaccard", "HANG_MARGIN_S", "DEFAULT_CHUNK",
              "cell_brief", "RENDER_JOBS"}
AUDIO_AT_TOP = ("numpy", "scipy", "soundfile", "pedalboard", "mido", "vllm", "torch", "lofi_rl.render",
                "lofi_rl.scoring.features", "lofi_rl.scoring.render_pool", "lofi_rl.shootout")


def modules() -> list[str]:
    return ["lofi_rl.step0"] + sorted(f"lofi_rl.step0.{m.name}" for m in pkgutil.iter_modules([str(PKG)]))


def child(mods, blocked) -> subprocess.CompletedProcess:
    # a finder that refuses the blocked packages (tests/test_scoring_imports.py's way)
    code = ("import importlib, importlib.abc, sys\n"
            f"BLOCKED = {tuple(blocked)!r}\n"
            "class Block(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path, target=None):\n"
            "        if name.split('.')[0] in BLOCKED:\n"
            "            raise ImportError(f'blocked in this test: {name}')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            + f"for m in {tuple(mods)!r}:\n    importlib.import_module(m)\n"
            + "leaked = [m for m in BLOCKED if m in sys.modules]\n"
            + "assert not leaked, leaked\nprint('IMPORTS_OK', len(sys.modules))\n")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, cwd=ROOT)


def test_the_walk_finds_every_step0_module():
    names = {m.rsplit(".", 1)[-1] for m in modules()}
    assert names >= {"step0", "requests", "select", "path", "cli", "__main__", "generate", "score", "render"}


def test_every_step0_module_imports_with_the_policy_envs_absent_modules_and_numpy_blocked():
    p = child(modules(), BLOCKED)
    assert p.returncode == 0 and "IMPORTS_OK" in p.stdout, p.stderr[-3000:]


def test_the_standard_library_modules_import_with_yaml_blocked_too():
    p = child(STDLIB, (*BLOCKED, "yaml"))
    assert p.returncode == 0 and "IMPORTS_OK" in p.stdout, p.stderr[-3000:]


def _imports(path: Path) -> tuple[set[str], set[str]]:
    """(the modules a file imports at module level, those it imports inside functions)."""
    tree = ast.parse(path.read_text())
    top, inner = set(), set()

    def names(node) -> list[str]:
        if isinstance(node, ast.Import):
            return [a.name for a in node.names]
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            return [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        return []
    for node in tree.body:
        top.update(names(node))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                inner.update(names(sub))
    return top, inner


def test_audio_and_vllm_are_imported_only_inside_functions():
    for name in ("generate", "score", "render"):
        top, _ = _imports(PKG / f"{name}.py")
        bad = sorted(m for m in top if m.startswith(AUDIO_AT_TOP))
        assert not bad, f"{name}.py imports {bad} at module level"
    _, inner = _imports(PKG / "render.py")
    assert {"lofi_rl.scoring.features", "lofi_rl.scoring.render_pool"} <= inner, "render's audio code, inside"
    for name in ("generate", "score", "render"):
        top, inner = _imports(PKG / f"{name}.py")
        assert not [m for m in top | inner if m.split(".")[0] == "vllm"], f"{name}.py reaches vLLM only through junk"


def _junk_names(path: Path) -> set[str]:
    """Every attribute a step0 module reads from junk (imported `as J`)."""
    tree = ast.parse(path.read_text())
    alias = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "lofi_rl.rl"
             for a in n.names if a.name == "junk"}
    return {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id in alias}


def test_step0_reads_only_junks_public_names_and_each_exists():
    used = set().union(*(_junk_names(p) for p in PKG.glob("*.py")))
    assert used, "step0 reuses junk"
    assert not [n for n in used if n.startswith("_")], "no private junk helper (spec §2.2: written again in step0)"
    assert used <= JUNK_NAMES, sorted(used - JUNK_NAMES)
    assert all(hasattr(J, n) for n in used | JUNK_NAMES)


def test_the_junk_names_step0_uses_hold_the_values_it_expects():
    assert J.SAMPLING == dict(S.V0_SAMPLING) and list(J.SAMPLING) == list(S.V0_SAMPLING)
    assert J.STRUCTURED_BACKEND == "xgrammar" == S.BACKEND
    assert J.DEFAULT_CHUNK == 128 == S.CHUNK and J.HANG_MARGIN_S == 90 and J.RENDER_JOBS == 2
    assert J.PROMPTS["v2"] is prompt_v2 and J.PROMPTS["v2"].PROMPT_VERSION == S.PROMPT_VERSION
    assert J.render_lanes(J.RENDER_JOBS, S.RENDER_WORKERS) == (2, 2)
    sig = inspect.signature(J.render_clip)
    assert list(sig.parameters) == ["row", "palette", "clips_dir", "cache", "workers"]
    assert [p.kind for p in sig.parameters.values()][3:] == [inspect.Parameter.KEYWORD_ONLY] * 2
    sig = inspect.signature(J.load_engine)
    assert list(sig.parameters) == ["model", "seed", "gpu_mem", "max_model_len", "backend"]
    assert all(sig.parameters[k].kind is inspect.Parameter.KEYWORD_ONLY for k in ("seed", "gpu_mem", "max_model_len"))
    assert list(inspect.signature(J.score_row).parameters) == ["row", "palette", "check"]
    assert list(inspect.signature(J.summarize).parameters) == ["rows", "scored", "meta", "palette", "check",
                                                               "unreadable", "run", "clips"]
    assert [f.name for f in fields(J.Completion)] == ["text", "finish_reason", "n_tokens", "prompt_tokens"]
    assert callable(J.VllmEngine.sampling) and callable(J.VllmEngine.chat)
    assert list(J.split_model(f"{S.MODEL}@{S.MODEL_REVISION}")) == [S.MODEL, S.MODEL_REVISION]
    assert J.constrained_block("x")["builder"] == "lofi_rl.grammar.loopscript_regex"
    assert J.cell_brief("low", "hopeful").bpm == J.CELL_BRIEF["bpm"]
    assert list(J.in_order(lambda i: i * 2, [3, 1, 2], 2)) == [6, 2, 4], "in the items' order"
