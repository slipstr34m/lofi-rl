"""Where a render ran: the platform facts that a render's bits depend on (instrument.md §4 Outputs, §7).

The manifest records the full platform, including the CPU model, because cluster renders are canonical
(docs/kit/LESSONS.md 38) and the cluster's determinism job compares two CPU models. Golden hashes are keyed
by `golden_key()`: platform, FluidSynth and numpy versions.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from lofi_rl.render import fluid

PACKAGES = ("numpy", "scipy", "pedalboard", "pyloudnorm", "soundfile", "mido")


@lru_cache(maxsize=None)
def cpu_model() -> str:
    """The CPU's marketing name: sysctl on macOS, /proc/cpuinfo on Linux, else platform.processor()."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=5).stdout.strip()
            if out:
                return out
        else:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


@lru_cache(maxsize=None)
def fluidsynth_version() -> str:
    return fluid.version()


def _version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "missing"


@lru_cache(maxsize=None)
def platform_info() -> dict[str, str]:
    """Everything the manifest records about the machine and its audio stack (JSON-ready, no wall clock)."""
    import soundfile
    info = {"system": platform.system(), "machine": platform.machine(), "cpu": cpu_model(),
            "python": platform.python_version(), "fluidsynth": fluidsynth_version(),
            "libsndfile": soundfile.__libsndfile_version__}
    info.update({pkg: _version(pkg) for pkg in PACKAGES})
    return info


def machine_key() -> str:
    """system-machine/cpu: what a cached one-shot is tied to, so two CPU models never share renders."""
    return f"{platform.system()}-{platform.machine()}/{cpu_model()}"


def golden_key() -> str:
    """The key of tests/golden/render_hashes.json: platform, FluidSynth and numpy versions."""
    return f"{platform.system()}-{platform.machine()}/fluidsynth-{fluidsynth_version()}/numpy-{np.__version__}"
