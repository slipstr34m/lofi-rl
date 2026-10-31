"""compile_loop: LoopScript AST + brief + palette -> CompiledLoop (docs/COMPILER.md).

Deterministic and symbolic: no audio, no wall clock, no unseeded randomness, and no dependence on dict or
set order or threads. Times are exact fractions until one round-half-up per sample position. The same
(loop, brief, palette) always gives the same CompiledLoop, so `digest()` is equal across calls,
processes and orders. Stages:
1. names: keys instrument and kit through the palette's aliases; the sax is the palette's lead;
2. timeline: N, BPM_eff, one clock per lane (swing, offset, humanize nudges);
3. bars: dsl.expand (bases tiled, edits applied, `bass: lock` resolved);
4. theory: chord roots, the keys voicing pair, the bass octave pair;
5. events: drum hits, keys and bass notes, the sax line, then its bend and CC11 curves.
"""
from __future__ import annotations

from lofi_rl.brief import Brief
from lofi_rl.dsl.ast import Loop, expand
from lofi_rl.dsl.vocab import LANES
from lofi_rl.palette import Palette, PaletteError
from lofi_rl.theory.chords import root_pc
from lofi_rl.theory.voicing import voice_vamp

from .bass import CHANNEL as BASS_CHANNEL
from .bass import bass_notes, bass_octaves
from .drums import drum_hits
from .events import CompiledLoop, Control, DrumHit, Note, Stem
from .expression import sax_controls
from .humanize import Humanize, loop_seed
from .keys import CHANNEL as KEYS_CHANNEL
from .keys import keys_notes
from .sax import sax_line
from .timeline import Clock, bpm_eff, loop_samples, ms

__all__ = ["CompileError", "CompiledLoop", "Control", "DrumHit", "Note", "Stem", "compile_loop"]


class CompileError(ValueError):
    pass


def lane_clock(lane: str, n: int, palette: Palette, human: Humanize) -> Clock:
    g = palette.groove
    return Clock(n, swing=g.swing_for(lane), offset=ms(g.offset_for(lane)), nudge=human.nudges(lane))


def compile_loop(loop: Loop, brief: Brief, palette: Palette) -> CompiledLoop:
    """Compile a parsed loop (a ParseResult's `ast`) for one brief with one palette."""
    try:
        keys_name = palette.resolve("keys", loop.keys.instrument)
        kit_name = palette.resolve("kits", loop.kit)
    except PaletteError as err:
        raise CompileError(str(err)) from err
    style = loop.keys.voicing
    if style not in palette.voicings:
        raise CompileError(f"voicing {style!r} is not in palette {palette.id}")
    for chord in loop.vamp.chords:
        if chord.numeral not in palette.numerals or chord.quality not in palette.qualities:
            raise CompileError(f"chord {chord.text!r} is not in palette {palette.id}")
    first, second = loop.vamp.chords
    if first == second:
        raise CompileError("the vamp needs 2 distinct chords (gate G6)")

    n = loop_samples(brief.bpm)
    seed = loop_seed(loop)
    human = Humanize.from_groove(palette.groove, seed)
    clocks = {lane: lane_clock(lane, n, palette, human) for lane in LANES + ("keys", "bass")}
    bars = expand(loop)

    voicing_a, voicing_b = voice_vamp(first, second, brief.tonic, style)
    bass_a, bass_b = bass_octaves(root_pc(first.numeral, brief.tonic), root_pc(second.numeral, brief.tonic))
    kit = palette.kits[kit_name]
    keys_voice = palette.keys[keys_name]
    sax_voice = palette.sax[palette.sax_lead]
    expr = palette.sax_expr

    try:
        line, repairs = sax_line(loop, brief.tonic, brief.mode, sax_voice, palette.groove, expr, n)
    except ValueError as err:
        raise CompileError(str(err)) from err
    bend, cc11 = sax_controls(line, expr, n)
    sax_notes = tuple(Note(s.pitch, s.on, s.off, s.velocity, s.channel, s.step) for s in line)
    return CompiledLoop(
        n=n, bpm=brief.bpm, bpm_eff=bpm_eff(n), tonic=brief.tonic, mode=brief.mode,
        palette_id=palette.id, palette_sha256=palette.sha256, kit=kit_name, keys_name=keys_name,
        sax_name=sax_voice.name, voicings=(voicing_a, voicing_b), bass_pitches=(bass_a, bass_b),
        keys=Stem("keys", keys_voice.src, keys_voice.bank, keys_voice.prog, (KEYS_CHANNEL,),
                  keys_notes(bars, {first: voicing_a, second: voicing_b}, clocks["keys"], keys_voice.vel,
                             human, n)),
        bass=Stem("bass", kit.bass.src, kit.bass.bank, kit.bass.prog, (BASS_CHANNEL,),
                  bass_notes(bars, {first: bass_a, second: bass_b}, clocks["bass"], kit.bass.vel, human, n)),
        sax=Stem("sax", sax_voice.src, sax_voice.bank, sax_voice.prog, tuple(sorted({s.channel for s in line})),
                 tuple(sorted(sax_notes, key=lambda x: (x.on, x.channel, x.pitch))), bend, cc11,
                 expr.bend_range),
        drums=drum_hits(bars, kit, palette.groove, clocks, human, n),
        seed=seed, repairs=tuple(repairs),
    )
