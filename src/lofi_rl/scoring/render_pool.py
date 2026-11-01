"""The render pool: lofi_rl.render.pipeline.render in spawn-context worker processes (taste_pipeline.md §5,
"Render pool"; docs/RENDER.md; docs/SCORER.md §7).

- Each render is the existing pipeline path: parse and gate the canonical text, compile, one fresh FluidSynth
  process per stem (cpu-cores=1, docs/RENDER.md §3.1), the sampler, the bus chains, the bed, loudness and G7.
  It returns the finished mix, the music-only stem (the same gain and limiter, no bed), the manifest, and the mix's
  spectral centroid (a hack-watch signature, taste_pipeline.md §7).
- Workers are spawned (no fork of a process that may hold CUDA) and load the palette once each. Each worker
  leads its own process group, so killing a worker (the batch deadline, restart(), close()) signals the group and
  ends its FluidSynth children with it; and a worker whose server is gone (SIGKILLed: no close()) ends its own
  group within a second, since a group kill from outside (`timeout -k`) no longer reaches it. A render uses
  `render_workers` (default 1) FluidSynth processes and pedalboard threads at a time, so 8 workers keep about 8
  cores busy; the output never depends on either number (docs/RENDER.md §5).
- A render that fails a gate (G1-G6 from the parser, G7 out of spec, a compile error) or an infrastructure fault
  (FluidSynthError, a file error, a crashed worker, a render past the deadline) becomes that item's outcome
  with the error recorded; the batch never fails because of one render. A crashed worker breaks the pool for
  every render in flight, so those are retried one at a time in a fresh pool; a batch past `timeout_s` gets
  render-timeout for what is left, and its workers are killed and replaced.
"""
from __future__ import annotations

import multiprocessing
import os
import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lofi_rl.brief import Brief
from lofi_rl.palette import ROOT, load_palette
from lofi_rl.render import pipeline
from lofi_rl.render.mix import BUFFER

from . import protocol

RENDER_TIMEOUT_S = 150.0          # under the client's 300 s, with room for the features
_STATE: dict[str, Any] = {}       # the worker's palette and settings, set by _init_worker


def render_config() -> dict[str, Any]:
    """taste_pipeline.md §2's render config, as lofi_rl.shootout.render.clip_id reads it from a manifest."""
    chain = {"cycles": pipeline.CYCLES, "fluid_preroll_cycles": pipeline.PREROLL,
             "midi_cycles": pipeline.PREROLL + pipeline.CYCLES, "pedalboard_buffer": BUFFER,
             "limiter_tile": pipeline.LIMITER_TILE}
    return {"schema": pipeline.SCHEMA, "chain": chain, "flac": protocol.FLAC_FORMAT}


@dataclass(frozen=True)
class RenderJob:
    clip_id: str
    canon: str                  # canonical LoopScript text
    brief: Mapping[str, Any]    # the Brief's fields


def spectral_centroid(x: np.ndarray, sr: int = pipeline.SR) -> float:
    """The magnitude-weighted mean frequency (Hz) of the whole cycle's mono spectrum; 0 for silence."""
    mono = np.asarray(x, dtype=np.float64).mean(axis=0)
    mag = np.abs(np.fft.rfft(mono))
    total = float(mag.sum())
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    return float((np.fft.rfftfreq(mono.size, 1.0 / sr) * mag).sum() / total)


def _exit_with_parent(parent: int, poll_s: float = 1.0) -> None:
    """In a worker: end its whole process group (itself and any FluidSynth) once the process that spawned it is
    gone. The worker leads its own group (_init_worker), so a SIGKILLed server leaves no orphans rendering."""
    while os.getppid() == parent:
        time.sleep(poll_s)
    os.killpg(os.getpgrp(), signal.SIGKILL)


def _init_worker(palette_path: str, root: str, render_workers: int) -> None:
    os.setpgrp()            # the worker's FluidSynth children join its group: RenderPool._kill signals the group
    threading.Thread(target=_exit_with_parent, args=(os.getppid(),), name="lofi-render-parent", daemon=True).start()
    _STATE.update(palette=load_palette(palette_path, verify=False, root=Path(root)), root=Path(root),
                  workers=int(render_workers))


def _ping() -> int:
    return os.getpid()


def failure(gate: str, error: str, *, transient: bool, t0: float | None = None) -> dict[str, Any]:
    return {"ok": False, "gate": gate, "error": error[:2000], "transient": transient,
            "render_ms": None if t0 is None else round(1000 * (time.perf_counter() - t0), 1)}


def render_job(canon: str, brief: Mapping[str, Any]) -> dict[str, Any]:
    """One render in a worker (module docstring). Never raises for a loop's or the machine's fault."""
    from lofi_rl.compile.compiler import CompileError
    t0 = time.perf_counter()
    try:
        r = pipeline.render(canon, Brief(**brief), _STATE["palette"], root=_STATE["root"], workers=_STATE["workers"])
    except pipeline.RenderError as err:
        return failure(err.code, err.detail, transient=False, t0=t0)
    except CompileError as err:
        return failure("compile", str(err), transient=False, t0=t0)
    except Exception as err:  # noqa: BLE001: FluidSynth, files, the machine: infrastructure, never a gate
        return failure("render-error", f"{type(err).__name__}: {err}", transient=True, t0=t0)
    m = r.manifest
    loud = m["loudness"]
    info = {"n": int(m["n"]), "sr": int(m["sr"]), "seconds": float(m["seconds"]), "bpm_eff": float(m["bpm_eff"]),
            "lufs": float(loud["lufs"]), "true_peak_dbtp": float(loud["true_peak_dbtp"]),
            "music_lufs": float(loud["music_only"]["lufs"]), "music_true_peak_dbtp": float(loud["music_only"]["true_peak_dbtp"]),
            "iterations": int(loud["iterations"]), "bed": m["bed"] is not None, "audio_sha256": m["audio_sha256"],
            "music_sha256": m["music_sha256"], "canon_sha256": m["loop"]["canon_sha256"],
            "compiled_sha256": m["loop"]["compiled_sha256"], "centroid_hz": spectral_centroid(r.audio)}
    info["render_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return {"ok": True, "gate": None, "error": None, "transient": False, "info": info, "manifest": m,
            "mix": r.audio, "music": r.music, "render_ms": info["render_ms"]}


def _signal_group(p: Any, sig: int) -> None:
    """Signal worker p's process group (it leads one: _init_worker), so its FluidSynth children end with it. A
    worker without a group of its own (it is still starting) gets the signal alone; a group that is gone, nothing.
    A group outlives a dead leader while a member lives, so a worker that died mid-render still has its FluidSynth
    signalled."""
    try:
        os.killpg(p.pid, sig)
    except ProcessLookupError:
        if p.is_alive():
            p.terminate() if sig == signal.SIGTERM else p.kill()


class RenderPool:
    """The pool (module docstring). render_many(jobs, on_done) calls on_done(k, outcome) for every job, in the
    order the renders finish, so the caller can featurize one clip while the others render."""

    name = "real"

    def __init__(self, palette_path: Path | str, *, workers: int = 8, render_workers: int = 1, root: Path = ROOT,
                 timeout_s: float = RENDER_TIMEOUT_S) -> None:
        if workers < 1:
            raise ValueError("the render pool needs at least one worker")
        self.palette_path, self.root = str(palette_path), Path(root)
        self.workers, self.render_workers, self.timeout_s = int(workers), int(render_workers), float(timeout_s)
        self.rebuilds = 0
        self._ctx = multiprocessing.get_context("spawn")
        self._ex = self._new()

    def _new(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(max_workers=self.workers, mp_context=self._ctx, initializer=_init_worker,
                                   initargs=(self.palette_path, str(self.root), self.render_workers))

    def warm(self, timeout_s: float = 300.0) -> list[int]:
        """Start every worker (imports and the palette) before the first request; their pids."""
        futures = [self._ex.submit(_ping) for _ in range(self.workers)]
        return sorted({f.result(timeout=timeout_s) for f in futures})

    def _kill(self) -> None:
        """Shut the executor down and end its workers and their FluidSynth children, a stuck render included:
        SIGTERM to each worker's process group, SIGKILL to the groups still alive 5 s later (CPython 3.12 has no
        public way to stop a running worker, so this reads the executor's process table)."""
        procs = list((getattr(self._ex, "_processes", None) or {}).values())
        self._ex.shutdown(wait=False, cancel_futures=True)
        for p in procs:
            _signal_group(p, signal.SIGTERM)
        for p in procs:
            p.join(5)
            if p.is_alive():
                _signal_group(p, signal.SIGKILL)
                p.join(5)

    def restart(self) -> None:
        self._kill()
        self._ex = self._new()
        self.rebuilds += 1

    def close(self) -> None:
        self._kill()

    def render_many(self, jobs: Sequence[RenderJob], on_done: Callable[[int, dict[str, Any]], None]) -> None:
        t0 = time.perf_counter()
        if not jobs:
            return
        try:
            futures = {self._ex.submit(render_job, j.canon, dict(j.brief)): k for k, j in enumerate(jobs)}
        except BrokenProcessPool:           # a worker died since the last batch: start this one on a fresh pool
            self.restart()
            futures = {self._ex.submit(render_job, j.canon, dict(j.brief)): k for k, j in enumerate(jobs)}
        broken: list[int] = []
        done: set[int] = set()
        pending = as_completed(futures, timeout=self.timeout_s)
        try:
            while True:
                try:
                    fut = next(pending)
                except StopIteration:
                    break
                except TimeoutError:                # the batch deadline; an on_done error propagates as itself
                    late = [k for k in futures.values() if k not in done and k not in broken]
                    self.restart()
                    for k in late:
                        done.add(k)
                        on_done(k, failure("render-timeout", f"not rendered within the batch's {self.timeout_s:g} s",
                                           transient=True, t0=t0))
                    break
                k = futures[fut]
                try:
                    out = fut.result()
                except BrokenProcessPool:
                    broken.append(k)
                    continue
                except Exception as err:  # noqa: BLE001: pickling and the like: this item's infrastructure fault
                    out = failure("render-error", f"{type(err).__name__}: {err}", transient=True)
                done.add(k)
                on_done(k, out)
        finally:
            for fut in futures:
                fut.cancel()
        if broken:
            self.restart()
            for k in sorted(broken):            # one at a time, so a loop that kills its worker takes only itself
                left = max(1.0, self.timeout_s - (time.perf_counter() - t0))
                fut = self._ex.submit(render_job, jobs[k].canon, dict(jobs[k].brief))
                try:
                    out = fut.result(timeout=left)
                except BrokenProcessPool:
                    self.restart()
                    out = failure("render-error", "the render worker died (BrokenProcessPool), twice", transient=True)
                except TimeoutError:
                    self.restart()
                    out = failure("render-timeout", f"retry not rendered within {left:.0f} s", transient=True)
                except Exception as err:  # noqa: BLE001
                    out = failure("render-error", f"{type(err).__name__}: {err}", transient=True)
                on_done(k, out)
