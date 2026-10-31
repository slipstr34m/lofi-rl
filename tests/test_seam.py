"""Seamless loops (instrument.md §3 Seamless, §0 amendment 3, §7 test_seam).

A 3-cycle render through FluidSynth, the sampler, the bus chains and the master chain must repeat:
cycle 2 equals cycle 3 to within 1e-6. Production keeps the last of 2 cycles, which is exactly cycle 2
of the 3-cycle render, because every stage is causal. The bed's seam and the 3x-tile limiter are periodic
by construction (tests/test_bed.py, render/loudness.py)."""
from dataclasses import replace

import numpy as np
import pytest

from helpers_compile import EXAMPLE_BRIEF, loop_of, loop_text
from helpers_render import needs_fluidsynth, noise, palette, rendered
from lofi_rl.brief import Brief
from lofi_rl.compile.compiler import compile_loop
from lofi_rl.render import pipeline
from lofi_rl.render.loudness import _limiter, limit_loop, limiter_latency
from lofi_rl.render.pipeline import music_cycles

TOL = 1e-6
# A random valid loop from the compiler's fuzz run (docs/COMPILER.md §4.3 caveat): without the pre-roll cycle,
# FluidSynth's 2nd and 3rd sax cycles differ by 4.7e-5 in the first block, where the first note starts while
# the previous cycle's last note still sounds. Its 3rd and later cycles are identical.
PREROLL_CASE = """shaker: xx....g...xgX.XX
@7 bass: .---x.-g.-x.XXXg
@6 shaker: XXxggxx.x.gxxx.X
@5 bass: .--x.X.Xxg-ggxxX
kick: x.....g...g..Xx.
B: er e.2,, h#3 w#3'' w.r q#6'' q.b2 s2
rim: ....xx..x.X.....
drums: jazz
keys: rhodes shell
bass: .--x-X..--X--X..
@8 comp: -g---x-XxX-.XXgx
vamp: bVI13 IIm9 span 1
@3 bass: xgXx-.Xgx-.X..Xg
comp: ---XX..--.X--x..
A: q2'' e#6' eb3, q.b2, q6'' w.r s3, e4 q1 sr q3'' q6,, sr q7 sr
sax: tenor
@1 hat: ..xgg.g..XX.xX..
"""
PREROLL_BRIEF = Brief(energy="high", mood="melancholy", bpm=79, tonic=1)        # brief.sample_brief(99, 0)


def _cycles(name: str, brief: Brief, count: int = 3):
    c = compile_loop(loop_of(name, brief.mode), brief, palette())
    music = music_cycles(c, palette(), count)
    return c.n, [music.cycles[:, i * c.n:(i + 1) * c.n] for i in range(count)]


@needs_fluidsynth
def test_cycle_2_equals_cycle_3_and_is_what_the_render_keeps():
    n, (c1, c2, c3) = _cycles("example.ls", EXAMPLE_BRIEF)
    assert np.abs(c2).max() > 0.01
    assert np.abs(c2 - c3).max() < TOL
    assert np.abs(c1 - c2).max() > 1e-4, "cycle 1 lacks the previous cycle's tails, so the check can fail"
    assert np.array_equal(rendered("example.ls").pre_limiter, c2), "the 2-cycle render keeps exactly cycle 2"


@needs_fluidsynth
def test_the_fluidsynth_pre_roll_cycle_covers_a_history_sensitive_loop(monkeypatch):
    # The render drops MIDI cycle 1, so the bus chains warm up on cycle 2 and the kept cycle comes from cycle 3.
    assert pipeline.PREROLL == 1 and pipeline.CYCLES == 2
    c = compile_loop(loop_text(PREROLL_CASE, PREROLL_BRIEF.mode), PREROLL_BRIEF, palette())
    sax_only = replace(c, keys=replace(c.keys, notes=()), bass=replace(c.bass, notes=()))
    n = c.n
    monkeypatch.setattr(pipeline, "PREROLL", 0)
    sax = pipeline.fluid_stems(sax_only, palette(), 4)["sax"]          # MIDI cycles 1-4 as rendered
    cyc = [sax[:, i * n:(i + 1) * n] for i in range(4)]
    assert np.abs(cyc[1] - cyc[2]).max() > 1e-5, "FluidSynth's 2nd cycle is not yet its steady state"
    assert np.abs(cyc[2] - cyc[3]).max() < TOL, "its 3rd cycle is"


def test_the_limiter_on_the_3x_tile_keeps_the_steady_state_cycle():
    # loudness.limit_loop keeps the middle of a 3x tile; a 5x tile's middle is further from any edge.
    n, sr = 64 * 2000, 44100
    x = 2.0 * noise(2, n, 21)
    x[:, :64], x[:, -64:] = 1.5, -1.5                     # peaks right at the seam
    lat = limiter_latency(sr)
    five = _limiter().process(np.tile(x, 5), sr)[:, 2 * n + lat:3 * n + lat]
    assert np.abs(limit_loop(x, sr) - five).max() < TOL


@pytest.mark.cluster
@needs_fluidsynth
@pytest.mark.parametrize("name", ["example.ls", "leniency_lower_x.ls", "one_bar_base.ls", "shaker_edits.ls",
                                  "span1_dotted.ls"])
@pytest.mark.parametrize("bpm", [61, 71, 87, 95])
def test_every_fixture_repeats_at_other_tempos(name, bpm):
    brief = Brief(energy="mid", mood="melancholy", bpm=bpm, tonic=bpm % 12)
    _, (_, c2, c3) = _cycles(name, brief)
    assert np.abs(c2 - c3).max() < TOL
