"""Determinism (instrument.md §7 test_determinism; CLAUDE.md §6). The same LoopScript, brief and palette give
bit-identical PCM (pcm_sha256 of the float32 samples, never the file bytes) in one process, in A-B-A render
order, with any worker count, and in a fresh process with another hash seed. The whole manifest is equal too.

Golden values are in tests/golden/render_hashes.json under machine.golden_key() (platform, FluidSynth and
numpy versions); other keys skip. Cluster renders are canonical (docs/kit/LESSONS.md 38) and the cluster's
determinism job records its own keys (docs/RENDER.md). To record this machine's values instead of checking
them, point LOFI_RECORD_GOLDEN at the file:
    LOFI_RECORD_GOLDEN=tests/golden/render_hashes.json .venv/bin/python -m pytest -m "cluster or not cluster" \
        tests/test_determinism.py
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf

from helpers_compile import DEV, EXAMPLE_BRIEF, FIX, ROOT, passing_fixtures
from helpers_render import OTHER_BRIEF, needs_fluidsynth, palette, rendered
from lofi_rl.brief import Brief
from lofi_rl.render.formats import dither_seed, to_int16
from lofi_rl.render.machine import golden_key
from lofi_rl.render.pipeline import manifest_json, render
from lofi_rl.render.wavio import pcm_sha256, read_float

GOLDEN = ROOT / "tests" / "golden" / "render_hashes.json"


def case_id(name: str, brief: Brief) -> str:
    return f"{name}|{brief.mood}|{brief.energy}|{brief.bpm}|{brief.tonic}"


def _golden_check(name: str, brief: Brief) -> None:
    r = rendered(name, brief)
    got = {"audio_sha256": r.manifest["audio_sha256"], "music_sha256": r.manifest["music_sha256"],
           "n": r.compiled.n}
    record = os.environ.get("LOFI_RECORD_GOLDEN")
    if record:
        path = Path(record)
        doc = json.loads(path.read_text()) if path.is_file() else {"keys": {}}
        doc.setdefault("keys", {}).setdefault(golden_key(), {})[case_id(name, brief)] = got
        doc["keys"] = {k: dict(sorted(v.items())) for k, v in sorted(doc["keys"].items())}
        path.write_text(json.dumps(doc, indent=2) + "\n")
        return
    table = json.loads(GOLDEN.read_text())["keys"].get(golden_key())
    if table is None or case_id(name, brief) not in table:
        pytest.skip(f"no golden render hashes for {golden_key()} {case_id(name, brief)} (cluster: canonical, "
                    f"recorded by the determinism job)")
    assert got == table[case_id(name, brief)], (
        "the render changed; if that is intended, re-record (see this module's docstring) and log it in CHANGELOG.md")


@pytest.fixture(scope="module")
def fresh_cli(tmp_path_factory):
    """The CLI on example.ls in a fresh process, with 1 worker and another hash seed. It starts when first
    requested and runs alongside this module's in-process renders, which keeps the suite's wall time down
    (instrument.md §7: under 60 s); a render must not depend on what else the machine is doing anyway."""
    if shutil.which("fluidsynth") is None:
        pytest.skip("fluidsynth not on PATH")
    out = tmp_path_factory.mktemp("fresh_cli") / "x.wav"
    env = dict(os.environ, PYTHONHASHSEED="4093",
               PYTHONPATH=os.pathsep.join([str(ROOT / "src")] + [p for p in [os.environ.get("PYTHONPATH")] if p]))
    cmd = [sys.executable, "-m", "lofi_rl.render.pipeline", str(FIX / "example.ls"), "--mood", "wistful",
           "--energy", "mid", "--bpm", "82", "--tonic", "2", "--palette", str(DEV), "--out", str(out), "--flac",
           "--workers", "1"]
    with (out.parent / "stdout.txt").open("w") as stdout, (out.parent / "stderr.txt").open("w") as stderr:
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env, cwd=out.parent)   # files: no pipe to fill
    yield proc, out
    if proc.poll() is None:
        proc.kill()
        proc.wait()


@needs_fluidsynth
def test_same_process_and_a_b_a_order_give_the_same_bits(fresh_cli):     # fresh_cli: start it now, check it below
    a1 = rendered("example.ls")
    b = render((FIX / "shaker_edits.ls").read_text(), OTHER_BRIEF, palette())
    a2 = render((FIX / "example.ls").read_text(), EXAMPLE_BRIEF, palette())
    assert pcm_sha256(a2.audio) == pcm_sha256(a1.audio) == a1.manifest["audio_sha256"]
    assert pcm_sha256(a2.music) == pcm_sha256(a1.music)
    assert manifest_json(a2.manifest) == manifest_json(a1.manifest), "every stem and one-shot hash too"
    assert b.manifest["audio_sha256"] != a1.manifest["audio_sha256"] and b.compiled.n != a1.compiled.n


@needs_fluidsynth
def test_a_fresh_process_with_one_worker_renders_the_same_bits_through_the_cli(fresh_cli):
    a = rendered("example.ls")                 # rendered here with 4 workers
    proc, out = fresh_cli
    proc.wait(timeout=300)
    stdout, stderr = ((out.parent / f"{name}.txt").read_text() for name in ("stdout", "stderr"))
    assert proc.returncode == 0, stderr
    first, rest = stdout.split("\n", 1)
    m = a.manifest
    assert first == (f"N={m['n']} LUFS={m['loudness']['lufs']:.3f} true_peak={m['loudness']['true_peak_dbtp']:.3f} "
                     f"dBTP pcm_sha256={m['audio_sha256']}")
    assert rest == manifest_json(m), "the whole manifest is equal across processes"
    assert out.with_suffix(".manifest.json").read_text() == manifest_json(m)
    x, _ = read_float(out)
    music, _ = read_float(out.with_suffix(".music.wav"))
    assert pcm_sha256(x) == m["audio_sha256"] and pcm_sha256(music) == m["music_sha256"]
    flac, sr = sf.read(str(out.with_suffix(".flac")), dtype="int16", always_2d=True)
    assert sr == 44100 and (flac.T == to_int16(a.audio, dither_seed(m["audio_sha256"]))).all()


@needs_fluidsynth
def test_golden_hashes_for_this_platform():
    _golden_check("example.ls", EXAMPLE_BRIEF)


CASES = [(f, EXAMPLE_BRIEF) for f in passing_fixtures()] + [("shaker_edits.ls", OTHER_BRIEF)]


@pytest.mark.cluster
@needs_fluidsynth
@pytest.mark.parametrize("name,brief", CASES, ids=[case_id(*c) for c in CASES])
def test_golden_hashes_for_every_fixture(name, brief):
    _golden_check(name, brief)
