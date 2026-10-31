"""The modules the GRPO reward imports must import in the cluster's policy env (requirements/policy.lock.txt:
numpy 2.3.5, no mido, soundfile, scipy or pedalboard). A fresh interpreter blocks those four through sys.modules
and imports every module of the shared interface; the policy lock has no torch-free stand-in on the laptop, so a
module that needs torch, vLLM or TRL (in the lock, not in the laptop venv) is skipped here, and a module not built
yet (lofi_rl.prompt, lofi_rl.rl.reward while their builders work) is skipped too.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = ("mido", "soundfile", "scipy", "pedalboard")
SHARED = ("lofi_rl.dsl", "lofi_rl.compile", "lofi_rl.compile.compiler", "lofi_rl.theory", "lofi_rl.brief",
          "lofi_rl.palette", "lofi_rl.prompt", "lofi_rl.verify", "lofi_rl.rl.reward")
WALK = ("lofi_rl.dsl", "lofi_rl.theory", "lofi_rl.compile", "lofi_rl.verify")    # every submodule of these too
SCORER_ONLY = {"lofi_rl.compile.midi", "lofi_rl.compile.stub", "lofi_rl.compile.stub_from_loop"}   # mido: COMPILER.md §1

PROBE = r"""
import importlib, importlib.util, json, pkgutil, sys
blocked, shared, walk = json.loads(sys.argv[1])
for name in blocked:
    sys.modules[name] = None                  # `import mido` now raises ModuleNotFoundError(name="mido")
names = list(shared)
for pkg in walk:
    try:
        mod = importlib.import_module(pkg)
    except ImportError:
        continue
    names += [m.name for m in pkgutil.walk_packages(mod.__path__, pkg + ".")]
out = {}
for name in dict.fromkeys(names):
    try:
        if importlib.util.find_spec(name) is None:
            raise ModuleNotFoundError(name=name)
    except ModuleNotFoundError as err:
        if err.name and (name == err.name or name.startswith(err.name + ".")):
            out[name] = ["absent", err.name]
            continue
    try:
        importlib.import_module(name)
        out[name] = ["ok", None]
    except ImportError as err:
        out[name] = ["missing", err.name]
    except Exception as err:                  # noqa: BLE001: any import-time error is a failure to report
        out[name] = ["error", f"{type(err).__name__}: {err}"]
print(json.dumps(out))
"""

VERIFY_CLOSURE = r"""
import sys
import yaml                                   # the palette's parser, with its compiled extension's runtime modules
before = set(sys.modules)
import lofi_rl.verify
new = {m.split(".")[0] for m in set(sys.modules) - before}
print(sorted(m for m in new if m not in sys.stdlib_module_names and m != "lofi_rl"))
"""


def _run(code: str, *args: str) -> str:
    r = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=60,
                       cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout.strip().splitlines()[-1]


@lru_cache(maxsize=None)
def probe() -> dict[str, list]:
    return json.loads(_run(PROBE, json.dumps([BLOCKED, SHARED, WALK])))


@lru_cache(maxsize=None)
def policy_packages() -> frozenset[str]:
    """Import names of the policy lock's packages, as far as a name maps (huggingface-hub -> huggingface_hub)."""
    names = re.findall(r"^([A-Za-z0-9_.\-]+)==", (ROOT / "requirements" / "policy.lock.txt").read_text(), re.M)
    return frozenset(n.lower().replace("-", "_") for n in names)


def test_the_policy_lock_has_none_of_the_blocked_modules():
    assert not set(BLOCKED) & policy_packages()
    assert {"numpy", "pyyaml", "torch", "trl", "vllm"} <= policy_packages()


@pytest.mark.parametrize("name", sorted(set(SHARED) | set(WALK)))
def test_shared_modules_import_without_the_scorer_only_modules(name):
    status, detail = probe()[name]
    if status == "absent":
        pytest.skip(f"{name} is not built in this tree yet")
    if status == "missing" and detail not in BLOCKED and detail.split(".")[0].lower() in policy_packages():
        pytest.skip(f"{name} needs {detail}, which the policy env has and the laptop venv lacks")
    assert status == "ok", (name, status, detail)


def test_every_submodule_imports_but_the_scorer_side_midi_writers():
    results = probe()
    walked = {n for n in results if n.startswith(tuple(p + "." for p in WALK))}
    assert SCORER_ONLY <= walked
    failing = {n for n in walked if results[n][0] != "ok"}
    assert failing == SCORER_ONLY, {n: results[n] for n in failing ^ SCORER_ONLY}
    assert all(results[n][1] == "mido" for n in SCORER_ONLY)


def test_verify_needs_only_the_standard_library_and_pyyaml():
    """Once PyYAML is in, lofi_rl.verify adds no module outside the standard library, so numpy's version (2.3.5 in
    the policy env, 1.26.4 here) cannot change a reward."""
    assert _run(VERIFY_CLOSURE) == "[]"
