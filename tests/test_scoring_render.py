"""The render pool (src/lofi_rl/scoring/render_pool.py; docs/SCORER.md §7): spawned workers on the pipeline path.
One real render (example.ls, dev_gu, D dorian 82 BPM) goes through the pool inside a Scorer: the mix and the
music-only stem reach the featurizer, their pcm_sha256 match the render's manifest and this machine's golden
hashes (tests/golden/render_hashes.json), and a text with the same canonical form is not rendered again. The
failure paths need no FluidSynth: a gate failure in the worker, a batch past its deadline, a broken pool."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from helpers_scoring import BRIEF_A, PALETTE, ROOT, fixture_text, make_scorer, pid_alive, start_sleeper, wait_gone
from lofi_rl.palette import load_palette
from lofi_rl.scoring import fakes, features, protocol
from lofi_rl.scoring.render_pool import RenderJob, RenderPool, RenderPoolDrift, render_config, spectral_centroid

PAL = load_palette(PALETTE, verify=False)
needs_fluidsynth = pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth not on PATH")


class RecordingFeaturizer(fakes.FakeFeaturizer):
    """The fake features, plus what the pool handed over for each clip."""

    def __init__(self):
        super().__init__(render="real")
        self.seen = []

    def featurize(self, clip_id, canon, mix, music):
        from lofi_rl.render.wavio import pcm_sha256
        self.seen.append({"clip_id": clip_id, "mix": (mix.shape, mix.dtype.name, pcm_sha256(mix)),
                          "music": (music.shape, music.dtype.name, pcm_sha256(music)),
                          "finite": bool(np.isfinite(mix).all() and np.isfinite(music).all())})
        return super().featurize(clip_id, canon, mix, music)


def _req(texts):
    return protocol.parse_score_request(protocol.score_request(
        [protocol.item_dict(i, t, BRIEF_A, "596:0") for i, t in enumerate(texts)], palette_sha256=PAL.sha256))


@needs_fluidsynth
def test_one_real_render_through_the_pool_reaches_the_featurizer_once(tmp_path):
    """The suite's three real scorer renders: one /score (this clip), one /render of the same clip to FLAC, and
    the server's canary (the prompt's example loop), which must render in spec on the pool, with the workers'
    hashes and the manifest's clip id accepted (else the server would never bind)."""
    import soundfile as sf
    from lofi_rl.render.machine import golden_key
    feat = RecordingFeaturizer()
    pool = RenderPool(PALETTE, workers=2, timeout_s=120)
    try:
        scorer = make_scorer(tmp_path, renderer=pool, featurizer=feat)
        out = scorer.score(_req([fixture_text("example.ls"), fixture_text("example_leniency.ls"), "vamp: no"]))
        files = tmp_path / "out"
        files.mkdir()
        rendered = scorer.render(protocol.parse_render_request(protocol.render_request(
            [protocol.item_dict(0, fixture_text("example.ls"), BRIEF_A)], palette_sha256=PAL.sha256,
            out_dir=str(files), flac=True)))["items"][0]
        canary = scorer.check_renderer("test")
    finally:
        pool.close()
    assert canary["ok"] and canary["gate"] is None and len(canary["audio_sha256"]) == 64
    man = json.loads(Path(rendered["files"]["manifest"]).read_text())
    pcm, sr = sf.read(rendered["files"]["flac"], dtype="int16")
    assert rendered["ok"] and man["audio_sha256"] == rendered["render"]["audio_sha256"] == \
        out["items"][0]["diag"]["render"]["audio_sha256"], "the same clip renders to the same PCM"
    assert (sr, pcm.shape) == (44100, (man["n"], 2)) and Path(rendered["files"]["flac"]).name == \
        f"{out['items'][0]['clip_id']}.flac"
    from lofi_rl.shootout.render import clip_id as shootout_clip_id
    assert shootout_clip_id(man) == out["items"][0]["clip_id"], "the id votes are kept under (taste_pipeline.md §2)"
    assert render_config() == {"schema": man["schema"], "chain": man["chain"], "flac": protocol.FLAC_FORMAT}
    a, b, junk = out["items"]
    assert (a["hard"], a["diag"]["cache"], b["diag"]["cache"], b["clip_id"]) == (1, "miss", "dup", a["clip_id"])
    assert (junk["hard"], junk["gate"]) == (0, "G1") and out["batch"]["rendered"] == 1
    assert len(feat.seen) == 1, "rendered and featurized once"
    info, seen = a["diag"]["render"], feat.seen[0]
    n = info["n"]
    assert seen["mix"][:2] == ((2, n), "float32") and seen["music"][:2] == ((2, n), "float32") and seen["finite"]
    assert seen["mix"][2] == info["audio_sha256"] and seen["music"][2] == info["music_sha256"]
    assert abs(info["lufs"] + 14) <= 0.1 and info["true_peak_dbtp"] <= -1.0 and info["bed"] is False
    assert 50.0 < info["centroid_hz"] < 22050.0 and a["sig"]["centroid_hz"] == info["centroid_hz"]
    table = json.loads((ROOT / "tests" / "golden" / "render_hashes.json").read_text())["keys"].get(golden_key(), {})
    want = table.get("example.ls|wistful|mid|82|2")
    if want is None:
        pytest.skip(f"no golden hashes for {golden_key()}")
    assert {"audio_sha256": info["audio_sha256"], "music_sha256": info["music_sha256"], "n": n} == want


def _outcomes(pool, jobs):
    got = {}
    try:
        pool.render_many(jobs, lambda k, out: got.__setitem__(k, out))
    finally:
        pool.close()
    return got


def test_a_gate_failure_in_the_worker_is_that_items_outcome():
    brief = protocol.brief_dict(BRIEF_A)
    got = _outcomes(RenderPool(PALETTE, workers=1, timeout_s=120), [RenderJob("c" + "0" * 16, "vamp: Im9", brief)])
    assert got[0]["ok"] is False and got[0]["transient"] is False
    assert got[0]["gate"] == "G3" and "B" in got[0]["error"], "a vamp alone lacks the required parts"


def test_a_batch_past_its_deadline_becomes_render_timeouts_and_a_fresh_pool():
    pool = RenderPool(PALETTE, workers=1, timeout_s=0.01)       # a cold spawn takes far longer than 10 ms
    brief = protocol.brief_dict(BRIEF_A)
    got = {}
    try:
        pool.render_many([RenderJob("c1", "vamp: Im9", brief), RenderJob("c2", "x", brief)],
                         lambda k, out: got.__setitem__(k, out))
        assert pool.rebuilds == 1
    finally:
        pool.close()
    assert {k: (v["gate"], v["transient"]) for k, v in got.items()} == {0: ("render-timeout", True),
                                                                         1: ("render-timeout", True)}


def test_a_broken_pool_retries_each_render_alone_then_records_an_infrastructure_fault(tmp_path):
    pool = RenderPool(tmp_path / "missing.yaml", workers=1, timeout_s=60)     # every worker's initializer fails
    got = _outcomes(pool, [RenderJob("c1", "vamp: Im9", protocol.brief_dict(BRIEF_A))])
    assert got[0]["gate"] == "render-error" and got[0]["transient"] is True and "BrokenProcessPool" in got[0]["error"]


def test_a_pool_found_broken_at_submit_is_restarted_before_the_batch():
    # A worker that died between batches breaks the pool before anything is in flight: submit itself raises
    # BrokenProcessPool, which render_many answers with a fresh pool instead of a 500 that crashes the trainer.
    pool = RenderPool(PALETTE, workers=1, timeout_s=120)
    got = {}
    try:
        [pid] = pool.warm()
        os.kill(pid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while not pool._ex._broken and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pool._ex._broken, "the executor noticed its dead worker"
        pool.render_many([RenderJob("c1", "vamp: Im9", protocol.brief_dict(BRIEF_A))],
                         lambda k, out: got.__setitem__(k, out))
        assert pool.rebuilds == 1
    finally:
        pool.close()
    assert (got[0]["gate"], got[0]["transient"]) == ("G3", False), "rendered (and gated) on the fresh pool"


def test_killing_the_pool_ends_a_workers_child_processes_too(tmp_path):
    # A FluidSynth that a stuck render started must not outlive the batch deadline's kill: workers lead their own
    # process group, and _kill signals the group. The child here is `sleep 60`, started by a task in a worker.
    pool = RenderPool(PALETTE, workers=1, timeout_s=120)
    pid_file = tmp_path / "child.pid"
    child = None
    try:
        pool._ex.submit(start_sleeper, str(pid_file))
        deadline = time.monotonic() + 60
        while not (pid_file.is_file() and pid_file.read_text()) and time.monotonic() < deadline:
            time.sleep(0.05)
        child = int(pid_file.read_text())
        assert pid_alive(child)
        pool._kill()
        assert wait_gone(child, 10), "the worker's child survived the kill"
    finally:
        pool.close()
        if child is not None and pid_alive(child):
            os.kill(child, signal.SIGKILL)


def test_a_worker_whose_server_was_sigkilled_ends_itself(tmp_path):
    # With its own process group a worker no longer gets the group kill of `timeout -k`; when the server dies
    # without close(), the worker sees its parent change and ends its group within about a second.
    code = (f"import time\nfrom lofi_rl.scoring.render_pool import RenderPool\n"
            f"pool = RenderPool({str(PALETTE)!r}, workers=1, timeout_s=60)\n"
            f"print(pool.warm()[0], flush=True)\ntime.sleep(120)\n")
    parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, cwd=ROOT,
                              env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    worker = None
    try:
        worker = int(parent.stdout.readline())
        assert pid_alive(worker) and os.getpgid(worker) == worker, "the worker leads its own process group"
        parent.kill()
        parent.wait(10)
        assert wait_gone(worker, 10), "the orphaned worker kept running"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if worker is not None and pid_alive(worker):     # never leave an orphan behind, even when this fails
            try:
                os.killpg(worker, signal.SIGKILL)
            except ProcessLookupError:
                os.kill(worker, signal.SIGKILL)


@pytest.mark.parametrize("key", ["src_sha8", "palette_sha256"])
def test_a_worker_that_loads_other_code_or_another_palette_is_refused_at_warm(key):
    want = {"src_sha8": features.source_sha8(), "palette_sha256": PAL.sha256}
    pool = RenderPool(PALETTE, workers=1, timeout_s=60, expect={**want, key: "0" * len(want[key])})
    try:
        with pytest.raises(RenderPoolDrift, match=key):
            pool.warm()
    finally:
        pool.close()


def test_respawned_workers_are_checked_against_the_code_the_pool_started_with(tmp_path):
    # `make sync` under a running server: a restart() after a timeout or a crash spawns workers on the new code.
    for p in sorted({p for g in features.SOURCE_GLOBS for p in ROOT.glob(g) if p.is_file()}):
        dst = tmp_path / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    pool = RenderPool(PALETTE, workers=1, timeout_s=60, root=tmp_path)
    try:
        assert len(pool.warm()) == 1 and pool.expect["src_sha8"] == features.source_sha8(tmp_path)
        with (tmp_path / "src" / "lofi_rl" / "render" / "mix.py").open("ab") as f:
            f.write(b"\n")
        with pytest.raises(RenderPoolDrift, match="src_sha8"):
            pool.restart()
    finally:
        pool.close()


def test_the_spectral_centroid_of_a_tone_and_of_silence():
    t = np.arange(44100) / 44100
    tone = np.stack([np.sin(2 * np.pi * 1000 * t)] * 2).astype(np.float32)
    assert spectral_centroid(tone) == pytest.approx(1000.0, rel=0.02)
    assert spectral_centroid(np.zeros((2, 1000), np.float32)) == 0.0
