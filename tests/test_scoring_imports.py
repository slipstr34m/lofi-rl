"""Every module of lofi_rl.scoring imports on the laptop, which has no torch (CLAUDE.md §1): each is imported in a
child interpreter with the scorer env's ML stack blocked through sys.modules, so torch, transformers, laion_clap
and audiobox_aesthetics stay inside the functions that load models (as in lofi_rl.smoke). The modules the
policy env imports (the client and the protocol, through lofi_rl.rl.reward) also import with numpy and the audio
stack blocked: they are standard library only (requirements/policy.lock.txt has no scipy, soundfile, pedalboard
or mido). Modules are found by walking the package, so a new module is checked without editing this file."""
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ML = ("torch", "torchaudio", "torchvision", "transformers", "laion_clap", "audiobox_aesthetics", "vllm", "trl")
AUDIO = ("numpy", "scipy", "soundfile", "pedalboard", "mido", "pyloudnorm", "pretty_midi")
POLICY_SIDE = ("lofi_rl.scoring.client", "lofi_rl.scoring.protocol", "lofi_rl.rl.reward")


def _modules() -> list[str]:
    import lofi_rl.scoring as pkg
    return ["lofi_rl.scoring"] + sorted(m.name for m in pkgutil.walk_packages(pkg.__path__, "lofi_rl.scoring."))


def _child(modules, blocked):
    # A finder that refuses the blocked packages, rather than sys.modules[name] = None: scipy's array-API helpers
    # treat any "torch" key in sys.modules as an imported torch and fail on the None (scipy 1.17, laptop).
    code = ("import importlib, importlib.abc, sys\n"
            f"BLOCKED = {tuple(blocked)!r}\n"
            "class Block(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, name, path, target=None):\n"
            "        if name.split('.')[0] in BLOCKED:\n"
            "            raise ImportError(f'blocked in this test: {name}')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            + f"for m in {tuple(modules)!r}:\n    importlib.import_module(m)\n"
            + "leaked = [m for m in BLOCKED if m in sys.modules]\n"
            + "assert not leaked, leaked\nprint('IMPORTS_OK', len(sys.modules))\n")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, cwd=ROOT)


def test_the_walk_finds_every_scoring_module():
    want = {"protocol", "client", "calib", "rm_api", "theme", "cache", "render_pool", "features", "core", "server",
            "fakes", "smoke"}
    assert {m.rsplit(".", 1)[-1] for m in _modules()} >= want


def test_every_scoring_module_imports_without_the_ml_stack():
    p = _child(_modules(), ML)
    assert p.returncode == 0 and "IMPORTS_OK" in p.stdout, p.stderr[-3000:]


def test_the_policy_side_modules_import_without_numpy_or_the_audio_stack():
    p = _child(POLICY_SIDE, ML + AUDIO)
    assert p.returncode == 0 and "IMPORTS_OK" in p.stdout, p.stderr[-3000:]
