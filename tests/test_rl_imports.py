"""The modules the GRPO trainer imports in the cluster's POLICY env, which has numpy 2.3.5 but no mido,
soundfile, scipy or pedalboard (requirements/policy.lock.txt). Each is imported in a child interpreter with
those modules blocked through sys.modules, and with torch, TRL, vLLM, transformers, PEFT, datasets and
accelerate blocked too: the laptop has none of them, so the RL modules import them only inside the functions
that train. lofi_rl.prompt and lofi_rl.verify are other builders' modules; they are checked once they exist."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = ("mido", "soundfile", "scipy", "pedalboard",
           "torch", "trl", "vllm", "transformers", "peft", "datasets", "accelerate")
MODULES = ("lofi_rl.dsl", "lofi_rl.compile", "lofi_rl.compile.compiler", "lofi_rl.theory", "lofi_rl.brief",
           "lofi_rl.palette", "lofi_rl.grammar", "lofi_rl.rl.curriculum", "lofi_rl.rl.reward", "lofi_rl.rl.config",
           "lofi_rl.rl.train", "lofi_rl.rl.smoke_check", "lofi_rl.rl.copy_penalty", "lofi_rl.rl.trl_probe",
           "lofi_rl.prompt_v2", "lofi_rl.rl.masked", "lofi_rl.grammar_mask", "lofi_rl.brief_streams",
           "lofi_rl.rl.mask_preflight")
OTHER_BUILDERS = ("lofi_rl.prompt", "lofi_rl.verify")


def test_policy_side_modules_import_without_the_audio_or_ml_stack():
    present = [m for m in OTHER_BUILDERS if importlib.util.find_spec(m) is not None]
    code = ("import importlib, sys\n"
            + "".join(f"sys.modules[{m!r}] = None\n" for m in BLOCKED)
            + f"for m in {MODULES + tuple(present)!r}:\n    importlib.import_module(m)\n"
            + f"leaked = [m for m in {BLOCKED!r} if sys.modules.get(m) is not None]\n"
            + "assert not leaked, leaked\nprint('IMPORTS_OK', len(sys.modules))\n")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, cwd=ROOT)
    assert p.returncode == 0 and "IMPORTS_OK" in p.stdout, p.stderr[-3000:]
