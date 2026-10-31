# COMPILER.md: LoopScript to CompiledLoop

The compiler turns a parsed LoopScript, a brief and a palette into a **CompiledLoop**: every note, drum
hit and controller point of one loop cycle, at exact sample positions. It makes no audio. The renderer
(`src/lofi_rl/render/`) turns a CompiledLoop into a WAV.

- **Spec:** `docs/design/instrument.md` §3 (binding formulas and tables), §6 (palette), §7 (layout and
  tests), and `docs/TASTE.md`. Section references below are to instrument.md unless noted.
- **Code:** `src/lofi_rl/compile/`, `src/lofi_rl/theory/`, and `src/lofi_rl/{palette,prng,brief}.py`.
- **Sound claims:** none. Every audible value here and in `palettes/dev_gu.yaml` is a starting point for
  an ear check or a shootout (instrument.md header).

## 1. Entry point

```python
from lofi_rl.brief import Brief
from lofi_rl.compile.compiler import compile_loop
from lofi_rl.compile.midi import write_stem_midi
from lofi_rl.dsl import parse
from lofi_rl.palette import load_palette

palette = load_palette("palettes/dev_gu.yaml")            # verify=True hashes every source file
brief = Brief(energy="mid", mood="wistful", bpm=82, tonic=2)
res = parse(text, vocab=palette.vocab(), mode=brief.mode, finish_reason=finish)
compiled = compile_loop(res.ast, brief, palette)           # only when res.gate.ok
write_stem_midi(compiled, "sax", path, cycles=2)            # one file per FluidSynth stem
```

- **Inputs.**
  - `loop`: a `ParseResult.ast` that passed gates G1–G6 (the language: `docs/DSL.md`). Parse it with
    `palette.vocab()`, so names are checked against the palette and its aliases. A loop parsed with another
    vocabulary still compiles when its names resolve through the palette.
  - `brief`: `lofi_rl.brief.Brief`: energy (`low`/`mid`/`high`), mood (`melancholy`/`wistful`/`hopeful`,
    which picks the mode), integer BPM in 60–95, tonic pitch class 0–11, scene `city at night`.
    The compiler reads only BPM, tonic and mode. Energy changes nothing it does (TASTE.md: more energy is
    the model's writing, and the sax gets no busier).
  - `palette`: `lofi_rl.palette.Palette` from `load_palette`.
- **Errors.** `CompileError` (a `ValueError`) for a name the palette lacks, a chord, quality or voicing
  outside the palette, a vamp of two equal chords, or a phrase with no notes.
- **Cost.** About 10 ms per loop on the laptop: 240 parses plus compiles of the fixtures took 2.57 s (2026-10-31; best of 3 runs after voicing rules V2, against 2.68 s before them the same day). The review fixes T7, T8 and SR-7 (§7) cost nothing measurable: run back to back the same day, the code before them took 2.69 and 2.68 s, and after them 2.67 and 2.69 s (load average about 2.2).
- **Environments.** Everything but `midi.py` needs only the standard library and PyYAML, so it imports in all
  three envs. `midi.py` needs mido, which the scorer env and the laptop have.

## 2. Stages

| # | Stage | Module | What it does | Spec |
|---|---|---|---|---|
| 1 | Names | `compiler.py`, `palette.py` | Keys instrument and kit resolved through the palette's aliases. The sax is the palette's `sax_lead`. | §6; TASTE.md, Lead |
| 2 | Timeline | `timeline.py`, `humanize.py` | N, BPM_eff, one clock per lane (swing, offset, humanize nudges) | §3 Timeline and swing, Humanize |
| 3 | Bars | `dsl.ast.expand` | Bases tiled to 8 bars, edits applied, `bass: lock` resolved | §1, §3 |
| 4 | Theory | `theory/{modes,chords,voicing}.py`, `bass.py` | Chord roots and tones, one keys voicing per chord, one bass octave per chord | §3 Pitch mapping, Voicings, Registers |
| 5a | Drums | `drums.py` | One hit per X/x/g cell: gain, layer, round-robin index, choke | §3 Velocities, Drum sampler |
| 5b | Keys, bass | `keys.py`, `bass.py`, `timeline.segments` | Strikes, holds, cuts and re-strikes at chord changes | §3 Chord changes |
| 5c | Sax | `sax.py` | Degrees to pitches, octave fit and folds, timing, note-offs, velocity, channels | §3 Pitch mapping, Registers, Sax expression |
| 5d | Expression | `expression.py` | Pitch bend (scoop, vibrato) and CC11 points | §3 Sax expression |
| 6 | MIDI | `midi.py` | Per-stem MIDI files for FluidSynth, written by the renderer's call | §3 Timeline, §4 |

## 3. Formulas

All sample positions are computed as exact fractions and rounded once, **half up** (`timeline.rnd`). Rates are
44.1 kHz, so 1 ms = 44.1 samples (`timeline.ms`).

### 3.1 Timeline (§3, Timeline and swing)
- **Loop length.** N = 64·⌊1,323,000/BPM + ½⌋ samples, a multiple of 64. BPM_eff = 84,672,000/N (`CompiledLoop.bpm_eff`).
- **Steps.** Step k ∈ [0, 128), with 16 steps to a bar. t_k = k·N/128 + [k odd]·(s − 0.5)·N/64 + offset_lane + nudge_lane(k).
  - s is the lane's MPC swing: `groove.swing.default` (0.55), or a per-lane override.
  - offset_lane is `groove.offset_ms[lane]`.
  - nudge is humanize (§3.6), zero by default.
  - The clock is periodic: t_{k+128} = t_k + N.
- **Lanes with a clock.** kick, snare, rim, hat, ohat, shaker, keys, bass, sax.
- **8th-note swing.** Used only by the sax's `swung` feel. Inside each quarter, at position p = k mod 4, the
  time is (4q + w(p))·N/128, with w(0) = 0, w(1) = 2s₈, w(2) = 4s₈ and w(3) = 2 + 2s₈. So the off-8th gets
  share s₈ of the beat, and the 16ths in between are placed linearly.

### 3.2 Pitch (§3, Pitch mapping and Registers)
- **Modes.** melancholy = aeolian [0,2,3,5,7,8,10], wistful = dorian [0,2,3,5,7,9,10], hopeful = ionian [0,2,4,5,7,9,11] (`theory/modes.py`, tables shared with `dsl/vocab.py`).
- **Chord root.** tonic + I 0, bII 1, II 2, bIII 3, III 4, IV 5, V 7, bVI 8, VI 9, bVII 10, VII 11 (`theory.chords.root_pc`).
- **Chord tones.** The §3 table, stored by role as third, fifth, seventh and extensions (`theory.chords.QUALITY`):

  | quality | tones | quality | tones |
  |---|---|---|---|
  | maj7 | 0 4 7 11 | 7sus | 0 5 7 10 14 |
  | maj9 | 0 4 7 11 14 | 7b9 | 0 4 7 10 13 |
  | maj7#11 | 0 4 7 11 14 18 | 7#9 | 0 4 7 10 15 |
  | 7 | 0 4 7 10 | m7 | 0 3 7 10 |
  | 9 | 0 4 7 10 14 | m9 | 0 3 7 10 14 |
  | 13 | 0 4 7 10 14 21 | m11 | 0 3 7 10 14 17 |
  | m7b5 | 0 3 6 10 17 | dim7 | 0 3 6 9 |

- **Sax pitch.** T + S[d−1] + accidental + 12·octave.
  - T is the tonic placed in the sax's 12-note home band: tenor 50–61, alto 57–68.
  - S is the mode. Accidentals are ±1; each `'` or `,` is ±12.
- **Sax octave fit, per phrase** (A = steps 0–63, B = steps 64–127):
  - Choose the shift k ∈ {−1, 0, +1} with the fewest notes outside the comfort window (tenor 48–72, alto 55–77).
  - Ties go to 0, then to −1.
  - Each note still outside the playable range (tenor 44–75, alto 49–80) folds by octaves into it. Each fold is one `Repair("sax_fold", 0, before, after, penalized=True)` in `CompiledLoop.repairs`. Line 0 means no source line. P_fix covers both lists: `dsl.repair.penalty(parse_result.repairs + compiled.repairs)`, at 0.02 each, capped at 0.1 (§2).
- **Bass pitch.** The root of the chord in effect at each strike, one octave per vamp chord for the whole loop. Choose (pA, pB) in 28–43 (E1–G2) to minimize |pA−36| + |pB−36| + 0.5·|pA−pB|. Ties go to the lower pair: lower pA + pB, then lower pA (`bass.bass_octaves`).

### 3.3 Keys voicings (§3, Voicings, as amended by voicing rules V2, §9)
Every voicing holds the chord's 3rd (the 4th for 7sus) and its 7th, and its colour: the top named extension, plus the ♭5 of m7b5 and dim7. The styles, in semitones above the root (`theory/voicing.py`; the template table, the rules and why are in §9):
- **rootless:** no root. The third, the seventh and the named extensions (at most two, the top one always), plus m7b5's ♭5. With no extension: third, fifth and seventh.
- **shell:** the third (the fourth for sus), the seventh and the top extension. m7b5 and dim7 take ♭3, ♭5 and their seventh.
- **spread:** drop-2 of each inversion of the rootless voicing. The second-highest note goes down an octave.
- **quartal:** every stack of 3 or 4 distinct tones drawn from the chord's named tones and the brief's mode's tensions that holds the 3rd, the 7th and the colour, and whose adjacent intervals are all 5 or 6 semitones, except at most one of 3 or 4.
  - The tensions are the 9ths, 11ths and 13ths (♭9 9 ♯9 11 ♯11 ♭13 13) that the brief's mode holds above the chord root (`voicing.chord_scale`) and the symbol does not name. A stack adds none a semitone above a named tone it holds (no natural 11 over a major 3rd).
  - So the stacks depend on the mode as well as the chord, and `voice_vamp` takes the brief's mode. Stacks are placed at every octave and never inverted.
  - A chord with no quartal candidate takes its rootless candidates: m7b5 always, and m9 on B, C and C♯ (§9).

The choice:
1. **Candidates.** Inversion j of a template raises its lowest j tones by an octave, for j = 0 … len−1. Each inversion, or each quartal stack, is placed at every octave where all of its notes lie in 48–72.
2. **Rejections.** A candidate is rejected if a ♯9 sounds under a major 3rd (R5: in 7♯9, rootless and shell inversions 1 and 2), or if two adjacent notes are 2 semitones apart or less and the lower one is below 55. It is also rejected if any two notes are 13 semitones apart, unless the chord is 7b9.
3. **Cost.** The pair (A, B) minimizes VL(A→B) + VL(B→A) + 0.25·(|μA − 60| + |μB − 60|), where μ is the mean pitch and VL is the summed movement under sorted pairing. With unequal sizes, voice i of A pairs with voice round(i·(|B|−1)/(|A|−1)) of B, halves up. So the outer voices pair bottom with bottom and top with top.
4. **Ties.** Ties go to the lexicographically smallest (A, B). Costs are exact: `choose` compares the cost times 4·lcm(the voicing sizes) as integers, one factor for every pair, so the order and the ties are the cost's. Each chord keeps its voicing for the whole loop.

`tests/test_voicing.py` checks all of this against an oracle that never reads `voicing.py` (`tests/helpers_voicing.py`), which rebuilds the tone sets from the chord-tones table and the mode tables:
- every candidate set, for every mode × numeral (borrowed ones included) × quality × style at every tonic, equals the oracle's, and every candidate keeps R1–R6;
- R7 in rootless, shell and spread, and the count of quartal coincidences;
- the quartal oracle itself against every pitch set in 48–72, for every quality on I and the sample vamps' chords;
- the choice against a brute force over the oracle's candidates, for 21 vamp × mode cases (the 12 shootout vamps of §6 in their moods, and IIm7b5 V7b9, Idim7 bIIImaj7#11 and I7sus V7#9 in every mode) × 12 tonics × 4 styles;
- the 12 shootout vamps at tonic 2 in every style, pinned as note names (§9).

### 3.4 Strikes, holds and chord changes (§3, Chord changes and `bass: lock`)
- **Grids.** The comp and bass grids are read as one 128-cell cycle (`timeline.segments`).
  - A strike (X/x/g) holds over the `-` cells that follow it, across the seam too (bar 8 into bar 1).
  - A `-` with no strike before it is silent.
- **Chord changes.** A note held across a chord change is cut at the change. Because the grid shows `-` there, it is re-struck with the new chord at the original strike's level.
- **Keys.** Every strike plays the whole voicing of its chord.
- **Bass.** Every strike plays the chord's bass pitch.
- **Note length.** A note runs from t_start to t_end on its lane's clock, so legato strikes meet exactly.
- **`bass: lock`** is resolved by `dsl.ast.lock_bass` before compiling:
  - the onsets are the non-ghost kicks after edits (level copied), plus an `x` at every chord change, the seam included;
  - each note holds to the next onset, at most 8 steps;
  - `@N bass:` edits then replace their bar.

### 3.5 Velocities (§3, Velocities)

| | X | x | g |
|---|---|---|---|
| drum gain (`groove.drum_gain`) | 1.0 | 0.71 | 0.32 |
| keys MIDI (`keys.<name>.vel`) | 96 | 80 | 52 |
| bass MIDI (`kits.<name>.bass.vel`) | 104 | 88 | 64 |

- **Hat and shaker metric offsets** (`groove.metric_db`), by step 1–16 of the bar:
  - steps 1 and 9: 0 dB;
  - steps 5 and 13: −1 dB;
  - the other odd steps: −2 dB;
  - even steps: −4 dB.
- **Drum hit gain.** `DrumHit.gain` = drum_gain[level] × 10^(metric_dB/20) × the humanize factor.
- **Sax velocity.** 92, plus 8 on step 1 of a bar, plus 4 on beat 3 (step 9), minus 10 for `s` notes, plus 6 on the first note of A and of B. Clipped to 60–112. With expression off it is a flat 92.

### 3.6 Humanize (§3, Humanize; shootout only)
- **Modes.** `groove.humanize` is `off` (the default), `bar` (H1) or `loop` (H2).
- **Lanes.** `groove.humanize_lanes`: the drum lanes, bass and keys. Never the sax.
- **Seed.** seed = the first 8 bytes, big-endian, of SHA256 of the canonical text (`dsl.canon(loop)`) without its
  `keys:`, `drums:` and `sax:` lines (`humanize.seed_text`, `CompiledLoop.seed`). What is left is the vamp, the
  grids, the edits and the phrases, so the groove depends only on them: another keys instrument, voicing style, kit
  or sax name, or no `sax:` line, draws the same nudges (SR-7, §7).
- **Key.** key = seed ⊕ (draw ≪ 24) ⊕ (lane ≪ 16) ⊕ (bar ≪ 8) ⊕ step-in-bar.
  - draw is 1 for timing and 2 for velocity;
  - lane runs from kick = 1 to sax = 9;
  - bar is 0 in H1 and the bar (0–7) in H2.
- **Draws.** h = splitmix64(key), and u = (h ≫ 11)·2⁻⁵³.
  - Timing nudge: (2u − 1)·3 ms, applied to every step of the lane, so note-offs move with the step they end on.
  - Velocity factor: 1 + (2u − 1)·0.08.
- **PRNG.** SplitMix64 (`prng.py`), with the test vector splitmix64(0) = 0xE220A8397B1DCDAF.

### 3.7 Sax timing and note-offs (§3, Sax expression; §8 issue 2)
- **Feel** (`sax_expr.timing`):
  - `grid` (the default): the band's swung grid, through the sax lane's swing and offset;
  - `lag`: also delays every sax event by `lag_ms` (25);
  - `swung`: 8th-note swing `swing8` (0.6) replaces the 16th swing.
- **Note-off.** `gap_ms` (15 ms) before a note that follows immediately. The line is read as a cycle, so B's last note runs into A's first. Otherwise the note-off is at the end of the note's last step.
- **Channels.** Sax notes alternate MIDI channels 2 and 3 in line order. Bend and CC11 are channel-wide, so this keeps each note's curves off the release of the note before it. The line is a cycle, so B's last note comes before A's first. Two channels cannot alternate around an odd number of notes, so then B's last note plays on channel 4 (`sax.note_channels`). `Stem.channels` lists the channels that play: (2, 3), or (2, 3, 4) for an odd line.

### 3.8 Sax expression curves (§3, Sax expression)
- **Setup.** The pitch-bend range is ±2 semitones, set by RPN 0 at t = 0 on every sax channel.
- **When points are written.** For each note, from its onset (τ = 0) to its note-off (τ = d samples):
  - a pitch-bend point every 10 ms (τ = round(j·441));
  - a CC11 point every 20 ms, plus one at τ = d.
- **Scoop.** A note of 4+ steps that follows a rest, or a leap of 3+ semitones from the previous note, bends from −60 cents to 0 over 80 ms. The shape is cents = −60·(1 + cos(πτ/80 ms))/2.
- **Vibrato.** On notes of 8+ steps it starts at 0.35·d and fades in linearly over 300 ms, with depth ±18 cents.
  - The phase is sin(2π·c·((on + τ) mod N)/N), with c = round(5.5·L) whole cycles per loop of L = N/44,100 s (`expression.vibrato_cycles`).
  - The rate is f_v = c/L.
- **Bend value.** round(cents/(100·range)·8192), clipped to −8192 … 8191.
- **CC11.**
  - Notes under 8 steps: 110.
  - Notes of 8+ steps: 100 + 27·sin(πτ/d).
  - The last note of A and of B crossfades linearly from that curve to 80 over its final 40%.
- **Expression off** (`sax_expr.enabled: false`, the shootout control): no bend points, CC11 110 at each onset, and velocity 92.
- **Periodicity.** Every value depends only on the note list and the time mod N, so the curves repeat every cycle.

### 3.9 Drum sampler fields (§3, Drum sampler)
- **Layer.** X plays hard, x med and g soft. If the kit's lane has no one-shots in that layer, the nearest non-empty layer plays, and the softer one wins a tie (`DrumHit.layer`).
- **Round-robin.** rr = h mod R. h is the hit's index, in step order, within its lane and layer in the cycle, and R is the number of one-shots in that layer (`DrumHit.rr`, `drums.round_robin`).
  - **Across the seam** (review T8, §7). The cycle repeats, so when a lane-layer plays n ≡ 1 (mod R) hits a cycle, h mod R gives its last hit and the next cycle's first hit both one-shot 0. For R ≥ 3 that last hit plays one-shot 1 instead, which differs from both of its neighbours (R − 1 before it, 0 after it). Every other hit keeps h mod R, so no two hits in a row play the same one-shot.
  - **Where a repeat is unavoidable.** With R = 2 and an odd n, two one-shots cannot alternate around an odd cycle, so one pair of hits in a row repeats wherever it is put. The assignment stays h mod 2, and the repeat stays at the seam (the last hit and the next cycle's first both play one-shot 0). A lone hit (n = 1) plays one-shot 0 in every cycle.
  - `dev_gu` has one one-shot per layer, so its rr is always 0. `tests/test_compile_drums.py` checks palette copies with 2, 3 and 4 one-shots per layer.
- **Choke.** For each kit pair (choked, choking), the cut is the onset of the choking-lane hit on the first strictly later grid step, read cyclically (the same step one cycle on comes last), that also sounds after the choked hit (`drums.choke_at`). `choke_at` is the earliest cut over the kit's pairs. It is a sample in the choked hit's cycle, so it may be ≥ N, which means the next cycle. With no choking hit in the loop it is `None`.
  - **The dev kit's pairs are (ohat, hat) and (ohat, ohat)** (review T7, §7). A hi-hat is one instrument, so an open hat ends at the next hat hit of either kind, closed or open. A lane may choke itself; `palette.py` refuses only a pair listed twice.
  - **So** consecutive open hats cut each other, and an open hat with a closed hat before the next open one still rings to that closed hat. An open hat with no other hat in the loop rings for the whole cycle: its own step one cycle on comes last, as for a closed hat, and that is its own repeat. The sampler leaves a one-shot that ends before its choke alone (`render/sampler.choke`). GU's open hat lasts 925,888 samples (soft) or 925,952 (med, hard), more than N from 92 BPM up. There its own repeat cuts the end of its tail: for a loop whose only hat is one accented open hat in bar 7, against the same loop without the (ohat, ohat) pair, 5,566 samples of the drum stem differ at 92 BPM and 34,468 at 95 BPM, by at most 4.4e-8 and 6.5e-8, with stem peaks of 0.023 and 0.024; at 60, 82 and 91 BPM none differ (measured 2026-10-31 with the cached one-shots).
  - **The earliest cut wins.** Extreme lane offsets and swing (within the palette's ranges) can make an open hat sound before a closed hat on an earlier step; the open hat then cuts the ring, since it strikes first (`test_the_earliest_cut_of_either_hat_wins`).
  - **The grid step decides, not the sample time.** Swing, lane offsets and humanize nudges move hits by a few ms, so a closed hat on the same step never chokes, whichever of the two lands first. With H1 humanize, each of `example.ls`'s five open hats shares its step with a closed hat that lands 40 samples after it; with H2, those at steps 102 and 126 do, 153 and 43 samples after. Every one plays until the closed hat 2 steps on (`tests/test_compile_drums.py`, which also checks that such a late closed hat exists).
  - **A hit that sounds first cannot cut.** Extreme offsets and swing (within the palette's ranges) can put the next step's closed hat ahead of the open hat in time; then the closed hat after it chokes.

## 4. The interface the renderer consumes

### 4.1 Conventions
- **Times.** Every time is a sample index at 44.1 kHz in one cycle of `n` samples.
- **Periodic.** Every event repeats in every cycle: an event at `on` plays at on + c·n in cycle c.
- **Ranges.** Onsets and control points lie in [0, n). A note's `off` may pass n, and then the note is held into the next cycle. That is how a lane offset or the sax lag crosses the seam.
- **Immutable.** All types are frozen dataclasses built from tuples and ints. `drums[*].gain` and `bpm_eff` are floats.

### 4.2 Types (`src/lofi_rl/compile/events.py`, re-exported by `compiler.py`)

```python
@dataclass(frozen=True)
class Note:
    pitch: int          # sounding MIDI note number
    on: int             # onset sample in [0, n)
    off: int            # note-off sample, on < off; may pass n
    velocity: int       # MIDI 1..127
    channel: int        # keys 0, bass 1, sax 2 and 3 in turn (4: B's last note of an odd line)
    step: int           # grid step 0..127 of the onset

@dataclass(frozen=True)
class Control:
    at: int             # sample in [0, n), written in every cycle
    channel: int
    value: int          # pitch bend -8192..8191 (0 = centre), or CC11 0..127

@dataclass(frozen=True)
class Stem:
    name: str                        # keys | bass | sax
    src: str                         # palette source id of the SoundFont
    bank: int
    program: int
    channels: tuple[int, ...]        # each gets bank, program (and bend range) at t = 0
    notes: tuple[Note, ...]          # sorted by (on, channel, pitch)
    bend: tuple[Control, ...] = ()   # sorted by (at, channel)
    cc11: tuple[Control, ...] = ()   # sorted by (at, channel)
    bend_range: int = 0              # semitones, RPN 0 at t = 0; 0 = no bend

@dataclass(frozen=True)
class DrumHit:
    lane: str                        # kick | snare | rim | hat | ohat | shaker
    level: str                       # X | x | g, as written
    layer: str                       # hard | med | soft, after the nearest-layer fallback
    on: int                          # onset sample in [0, n)
    gain: float                      # linear sampler gain (§3.5)
    rr: int                          # index into palette.kits[kit].lanes[lane].layers[layer]
    step: int                        # grid step 0..127
    choke_at: int | None = None      # sample (may pass n) where the choke fade starts

@dataclass(frozen=True)
class CompiledLoop:
    n: int                           # samples per cycle, a multiple of 64
    bpm: int                         # the brief's tempo
    bpm_eff: float                   # 84,672,000 / n
    tonic: int                       # pitch class, C = 0
    mode: str                        # aeolian | dorian | ionian
    palette_id: str
    palette_sha256: str              # sha256 of the palette YAML bytes
    kit: str                         # palette kit name, after aliases
    keys_name: str                   # palette keys name, after aliases
    sax_name: str                    # the palette's lead sax
    voicings: tuple[tuple[int, ...], tuple[int, ...]]   # keys voicing of vamp chord 1, chord 2
    bass_pitches: tuple[int, int]                       # bass pitch of vamp chord 1, chord 2
    keys: Stem
    bass: Stem
    sax: Stem
    drums: tuple[DrumHit, ...]       # sorted by (on, lane order, step)
    seed: int                        # humanize seed (§3.6)
    repairs: tuple[Repair, ...] = () # sax octave folds (penalized, like the parser's repairs)
    # .stems -> (keys, bass, sax); .to_json() canonical JSON; .digest() its sha256
```

### 4.3 Pitched stems: MIDI and FluidSynth
- **Files.** `midi.write_stem_midi(compiled, name, path, cycles=2)` writes one type-0 file per stem.
  - Tempo is 120 BPM at PPQ 22,050.
  - At t = 0 each channel gets bank select CC0 = bank, CC32 = 0, and the program. The sax channels also get the RPN pitch-bend range, then RPN null.
  - Then come `cycles` copies of the stem's events.
  - The end of track is at cycles·n + 4,410 ticks, or later if a note-off needs it.
  - At equal ticks the order is note-offs, then controllers, then note-ons. A note lasts at least one 64-sample block.
  - The SoundFont is `palette.source_path(stem.src)`.
  - Render each file in a fresh FluidSynth process (`render/fluid.py`) and keep [n, 2n).
- **Ticks are not samples** (a deviation from §3's "1 tick = 1 sample"; `fluid=False` gives the literal mapping). Measured on the laptop with FluidSynth 2.6.1 on 2026-10-31 (`tests/test_compile_midi.py` holds the table):
  - **Dispatch.** The file player renders in 64-sample blocks. Before block b + 1 it sends every event with tick ≤ cur(b) = round(44.1·⌊64,000·b/44,100⌋). The event then takes effect from sample 64·(b + 1). A note-on at tick 0 starts at sample 64.
  - **Why plain ticks fail.** With tick = sample, the same event lands in different blocks in different cycles unless N is a multiple of 28,224 samples (640 ms). In 60–95 BPM only 60 and 75 are. At every other tempo tried (61, 82, 87, 90 and 95 BPM), the 2nd and 3rd cycles of a 3-cycle render differed by 0.043–0.095, against peaks of 0.042–0.074.
  - **The fix.** `midi.fluid_tick(t)` picks the middle of the tick range that this player sends into the block nearest t, which is block round(t/64), at least block 1.
  - **Result.** The 2nd and 3rd cycles of 3-cycle renders were bit-identical (max difference 0.0) for every stem of `example.ls`, `shaker_edits.ls`, `span1_dotted.ls` and `one_bar_base.ls`, at 82, 90, 71 and 95 BPM.
  - **Cost.** Pitched notes start on a block boundary within ±32 samples (0.73 ms) of their sample time. Drum hits stay sample-exact.
- **Caveat for the renderer's seam test: 2 cycles are not always enough.** FluidSynth's output for a note depends slightly on the history of the voice it is given. One isolated note differed by 8.6e-5 between its first cycle and its later ones. So the 2nd cycle can differ from the steady state even with the ticks above.
  - In 24 fuzzed stems (8 random valid loops × 3 stems, across 5 palette variants), the 2nd and 3rd cycles were identical for 23. For one sax stem (79 BPM, 17 notes) they differed by 4.7e-5, against a peak of 0.033. The difference lay only in samples 2–63, the cycle's first block, where its first note starts while the previous cycle's last note is still releasing. It stayed at 3.7e-5 with expression off. Its 3rd, 4th and 5th cycles were identical.
  - In a synthetic stress case, 0.68 s notes overlapped the seam heavily. The 2nd cycle differed from the 3rd by up to 1.1e-4, and the 3rd and 4th were identical.
  - So a 1e-6 `test_seam` on "keep the 2nd of 2 cycles" can fail on some loops. Rendering 3 cycles (`cycles=3`) and keeping the 3rd gave identical cycles in every case tried. That choice is the renderer's (§3, Seamless says 2).
  - **The renderer's choice:** it writes 1 + C cycles and drops the first as a pre-roll, so "keep [n, 2n)" above becomes the last cycle of 3 in production (C = 2; `docs/RENDER.md` §4).
- **Empty stems.** A comp or bass grid may be all rests. That stem then has no notes, and its MIDI file holds only the setup events.

### 4.4 Drums
- **One-shot.** Each `DrumHit` plays `palette.kits[compiled.kit].lanes[hit.lane].layers[hit.layer][hit.rr]` at sample `hit.on`, scaled by `hit.gain`.
- **Lane mix.** The lane's `pan` (−1 … +1) and `db` come from the same palette lane.
- **Choke.** If `choke_at` is set, the hit fades out over 8 ms from that sample, which may be in the next cycle.
- **Seam.** Tails past n wrap to the start of the cycle.
- **Dev one-shots.** In `dev_gu` a one-shot is a GeneralUser GS drum note `{src, bank: 128, prog: 32, note, vel}`: kick 36, snare 38, rim 37, hat 42, ohat 46, shaker 82, at vel 48/80/112 for soft/med/hard. The renderer renders each one once and caches it. A one-shot can instead be a file in a source (`{src, file}`).

### 4.5 What else the renderer reads from the palette
The compiler does not use these, so `palette.py` only validates their shape:
- **`mix`:** `bus_gain_db`, `bus_target_lu` (the §4 targets: sax 0, drums −1.5, bass −3, keys −6 LU), the `sax_warm` flag, and `buses` (sax, sax_warm, keys, bass, drums) plus `master`. Each chain is a list of `Plugin(name, params)` using pedalboard 0.9.26 class and parameter names, in order. `sax_warm` replaces the sax reverb when the flag is on. The §4 values are written out; parameters §4 leaves open are at pedalboard's defaults.
- **`bed`:** `{src, start_s, xfade_s, delta_lu}`, or `None`. `dev_gu` has no rain source, so it is `None`.
- **`loudness`:** −14 LUFS, tolerance 0.1, 4 iterations, ceiling −1 dBTP.

## 5. Determinism
- **What the compiler avoids.** A pure function of (loop, brief, palette):
  - no wall clock, and no randomness beyond the keyed SplitMix64;
  - no `hash()`, and no dict or set order in any output: every set is sorted, and palette mappings are built in sorted key order;
  - no threads.
- **Numbers.** Sample positions use exact `Fraction`s, including palette decimals: 0.55 is read as 11/20. Floats appear only in drum gains, `bpm_eff` and the sine/cosine curves. The curves are rounded to MIDI integers before they leave the compiler; drum gains and `bpm_eff` stay floats.
- **Tests** (`tests/test_compile.py`): equal objects and digests in one process, in A-B-A order, and in two fresh processes with different `PYTHONHASHSEED`s.
- **Across machines.** A libm difference in one ulp could in principle move a rounded curve value. Cluster renders are canonical (LESSONS 38).

## 6. Choices the spec left open
Each is pinned by a test and can be changed by Shresth.

- **§3 Voicings.**
  - "Every inversion" of a template wider than an octave raises its lowest j tones by an octave.
  - Spread applies drop-2 to each rootless inversion; with 3 notes, the middle note drops.
  - The quartal templates (and §3's "+ext" and "18|19") are gone: voicing rules V2 replace them with stacks (§9, which also lists the readings V2 leaves open).
  - VL with unequal sizes pairs the outer voices (§3.3).
  - Rule 2's "≤ 2 semitones" is between adjacent notes, which covers every pair; "13-semitone pairs" is between any two notes.
- **§3 Registers.**
  - The sax octave fit breaks a tie between −1 and +1 toward −1.
  - The bass's "ties go lower" means the lower sum, then the lower pA.
- **§3 Chord changes.** Holds read the grid as a cycle; a lone `-` is silent; a re-strike keeps its strike's level.
- **§3 Timeline.** One rounding, half up, on exact fractions.
- **§3 Sax expression.**
  - "Phrase starts" and "last note of a phrase" mean the first and last notes of A and of B.
  - "Long notes" for CC11 means 8+ steps.
  - d is the sounding length from onset to note-off.
  - The vibrato fade and the last-note decay are linear.
  - Bend and CC11 points start at each onset, plus a CC11 point at the note-off. Nothing is written during rests.
  - The line is cyclic for "follows a rest or leap" and "follows immediately".
  - Expression off means no bend, CC11 110, and velocity 92.
  - Sax notes alternate channels 2 and 3 around the cycle; B's last note takes channel 4 when the count is odd.
- **§3 Humanize** (`tests/test_compile.py` recomputes every drum, keys and bass nudge and velocity factor of three
  loops from the §3.6 key, H1 and H2).
  - The key packs each field into its own byte, and timing and velocity are separate draws.
  - It applies to the drum lanes, bass and keys, never the sax: the sax stem is identical with humanize on and off.
  - A nudge moves note-offs too, so legato strikes stay joined (`test_humanize_keeps_legato_keys_and_bass_strikes_joined`).
  - The seed is read big-endian.
  - "SHA256(canonical text)" leaves out the `keys:`, `drums:` and `sax:` lines (SR-7, §7), so naming another sound
    never moves the groove (`test_naming_another_keys_instrument_voicing_or_kit_keeps_every_nudge`).
- **§3 Drum sampler.**
  - The compiler resolves the layer (the softer one wins a tie) and the round-robin index, which avoids a repeat across the seam where it can (review T8, §3.9).
  - "The next closed hat" became the next hat hit of either kind (review T7, §7): for each choking lane, its hit on the next strictly later grid step that sounds after the open hat, and the earliest of those. The compiler computes `choke_at`.
- **§8 issues** (settled by TASTE.md and the plan):
  - One sax plays every loop: the palette's `sax_lead`. A loop's `sax:` line is optional and does not choose (not a G3 requirement; `docs/DSL.md` §4). When present it still has to pass G4. Shootouts swap `sax_lead`.
  - The sax feel is a palette option. `grid` is the default; `lag` (25 ms) and `swung` (0.6) are starting points for the sax shootout.
  - Energy does not change the sax.
  - The bass holds the roots.
  - Drop-outs mid-loop compile like any other bar.
- **§6 Palette.**
  - Every key is required.
  - Added keys: `sax_lead`, `aliases`, `kits.<k>.bass.vel`, `groove.swing` per timing lane, `humanize_ms`, `humanize_vel`, `humanize_lanes`, `metric_db: {lanes, db[16]}`, the `sax_expr` fields, `mix.bus_target_lu`, and `bed: null`.
  - One-shots take the SoundFont-note or file form.
  - Licences (D-012): mode `commit` is an allow-list of CC0, public domain, permissive and CC BY licences, with no restriction named (non-commercial, ShareAlike, NoDerivatives, all rights reserved, personal use, no redistribution). Any other licence, an unrecognized spelling included, must be mode `fetch`. A CC-BY source must name its attribution, however the licence is spelled ("CC BY 4.0", "Creative Commons Attribution 3.0 Unported"). `palette.licence_problem` is the check. `tests/test_licenses.py` applies the same allow-list to `assets/MANIFEST.yaml`:
    - a mode-commit entry needs a licence the allow-list accepts;
    - an entry whose licence it refuses is mode fetch under `assets/big/`;
    - the one carve-out is a licence document (`LICENSE.txt`, licence "the license text itself") committed in the folder of an asset the allow-list accepts.
  - `load_palette(verify=True)` also reads each SoundFont's preset headers (`palette.sf2_presets`) and refuses a (bank, program) the file lacks, which FluidSynth would otherwise replace with another preset.
- **Briefs** (`brief.sample_brief(seed, index)`).
  - BPM is an integer.
  - Each tempo in 75–90 has 4× the weight of one outside it, so P(75–90) = 64/84.
  - Energy, mood and tonic are uniform.
  - Each field is its own keyed draw, so a brief's index fixes it, however many briefs are drawn.
  - **The key is chained SplitMix64:** h = splitmix64(splitmix64(splitmix64(seed mod 2⁶⁴) ⊕ index) ⊕ field).
    - It hashes the seed, then mixes in the index, then the field. The fields are energy 1, mood 2, tempo 3 and
      tonic 4, and the index runs from 0 to 2⁶⁴ − 1.
    - The value is `prng.below(h, k)` = ((h ≫ 11)·k) ≫ 53, for k choices. The tempo takes it over the 84 weight
      units.
  - **Streams from different seeds are independent** (`tests/test_brief.py`). The old key, seed ⊕ (index ≪ 8) ⊕
    field, had two flaws:
    - seeds a multiple of 256 apart replayed each other's briefs in another order:
      `sample_brief(256, 0) == sample_brief(0, 1)`, and brief i of seed 256k was brief i ⊕ k of seed 0;
    - adjacent seeds coupled fields: the energy of seed 2026 was the mood draw of seed 2025.
  - **The fields of one brief are independent too.** `tests/test_brief.py` checks 900 briefs (seeds 0, 2026 and
    2⁶⁴ − 1) against the key above, written out in the test with its own SplitMix64, so equal field ids or a changed
    chain fail there. It also checks 6,000 briefs of seed 2026: all 9 energy × mood pairs occur, and a chi-square
    on every pair of fields stays under z = 6. The marginals alone could not see a shared draw: with mood drawn
    from the energy hash, only 3 of the 9 pairs occur and every marginal is still uniform.
  - **No golden file pins a sampled brief.** Golden renders and the seam tests use explicit `Brief(...)`s.
    The comment on `tests/test_seam.py`'s `PREROLL_BRIEF` still names `sample_brief(99, 0)`, which is where the
    brief first came from, under the old key.

## 7. Spec issues found (not changed)
- **§3 rootless on m7b5 (settled 2026-10-31 by voicing rules V2, §9).** The literal rule ("the 5th is dropped when there is an extension") dropped m7b5's ♭5, so rootless m7b5 was [3, 10, 17] with no ♭5. Under V2 only a perfect 5th goes: rootless m7b5 is [3, 6, 10, 17] and its shell [3, 6, 10].
- **§3 rootless "first extension" (settled 2026-10-31 by V2, §9).** It made rootless 13 = rootless 9 ([4, 10, 14]), rootless m11 = rootless m9, and rootless maj7#11 = rootless maj9. Under V2 rootless holds every named extension (at most two): 13 is [4, 10, 14, 21], m11 [3, 10, 14, 17] and maj7#11 [4, 11, 14, 18].
- **§1 vs the parser on `sax:` (settled 2026-10-31: the parser is right).** §1 lists `sax:` as required, but TASTE.md (one sax plays in every loop) and the plan (`sax:` is dropped once one sax survives) make the line choose nothing. So it is optional, never a G3 requirement, and the compiler plays `sax_lead` whatever it says (`docs/DSL.md` §4 and §10). `tests/test_compile.py::test_a_loop_without_a_sax_statement_passes_the_gates` pins the gate.
  - **The humanize seed no longer reads the line (settled 2026-10-31, SR-7).** The seed hashed all of `dsl.canon(loop)`, which prints the `sax:` line, the keys instrument and voicing style, and the kit. So adding `sax: tenor`, or swapping rhodes for organ or rootless for spread, gave another seed, and under `bar` or `loop` every drum, keys and bass nudge moved. `compile/humanize.loop_seed` now hashes the canonical text without its `keys:`, `drums:` and `sax:` lines (§3.6), so the groove depends only on the vamp, the grids and the phrases.
    - `test_a_loop_without_a_sax_statement_compiles_to_the_same_loop_as_with_sax_tenor` passes, and its strict xfail mark is gone. `test_naming_another_keys_instrument_voicing_or_kit_keeps_every_nudge` checks, under H1 and H2, that organ for rhodes and spread for rootless leave every drum hit and bass note, and every keys strike's times and velocity, unchanged, and that the kit names dusty, brushes and jazz compile to one loop. `test_humanize_off_puts_every_hit_on_its_clock` pins the new formula.
    - Every loop's seed changed. Of 126 compiles (the 6 gate-passing fixtures × 7 briefs × humanize off, bar and loop), the 42 with humanize off changed only in `seed`; the 84 with `bar` or `loop` changed in their drum, keys and bass events, never in the sax. No golden render changed: they run with humanize off.
    - With the new seed, `example.ls` again has closed hats that share a step with an open hat and land after it (§3.9), so `test_a_closed_hat_on_the_same_step_never_chokes_under_humanize` keeps its case; it now asserts that the case exists. For the same reason the loop of `test_per_lane_swing_or_humanize_never_lets_a_same_step_hat_choke` now ends B with `w4 | w3 | w2 | h1 hr`: with its old B, no closed hat landed after its open hat under H1.
    - **Dropping the line from canon instead** (`dsl/canon.py`) would break 13 tests in `tests/test_dsl_canon.py`, which pin that canon prints the line and round-trips it.
- **§3 Drum sampler, "the open hat is choked by the next closed hat" (settled 2026-10-31, review T7).** A hi-hat is one instrument, so an open hat ends at the next hat hit of either kind, closed or open. `palette.py` now accepts a lane that chokes itself (it refused `[ohat, ohat]` as a lane listed twice) and refuses a pair listed twice instead; `dev_gu` lists `[[ohat, hat], [ohat, ohat]]`, and `choke_at` is the earliest cut over the pairs (§3.9). No fixture compiles differently but for its `palette_sha256`: in each, a closed hat comes before the next open hat. The edit moved `dev_gu.yaml`'s sha256 from `f2632ffb…` (the value in the cluster render-check manifests under `runs/`) to `b2e4a340…`.
- **§3 Drum sampler, "rr = h mod R" (settled 2026-10-31, review T8).** With n ≡ 1 (mod R) hits a cycle in a lane-layer, the last hit and the next cycle's first both played one-shot 0. For R ≥ 3 the last hit now plays one-shot 1; with R = 2 and an odd n one repeat is unavoidable, and today's assignment stays (§3.9). `dev_gu` has one one-shot per layer, so nothing it compiles or renders changed.
- **§6 naming.** "GU Drawbar Organ 16" is the GM name for program 16. GeneralUser GS 2.0.3 calls that preset "Tonewheel Organ".

## 8. Tests
The instrument.md §7 rows, plus the interface. `tests/helpers_compile.py` holds the spec oracles: §3.1 step times and §3.2 sax pitches written out from the tables, independently of `src/`. `tests/helpers_voicing.py` is the voicing oracle (§3.3, §9). Together the 12 files below hold 377 passing tests and no xfail, which ran in 6.59 s on the laptop (2026-10-31, after the review fixes T7, T8 and SR-7 of §7; 7.18 s in the run before it). After voicing rules V2 they held 360 and 1 strict xfail, the `sax:` seed issue that SR-7 settled, and ran in 5.81 s (`tests/test_voicing.py` alone, 27 tests, 0.91 s).

| §7 row | file |
|---|---|
| `test_modes_chords` | `tests/test_modes_chords.py`: modes, every degree of every mode through `degree_offset`, home bands, the chord-tones table |
| `test_voicing` | `tests/test_voicing.py`: voicing rules V2 against the oracle in `tests/helpers_voicing.py` (R1–R6 for every mode × numeral × quality × style at every tonic, R7 and the quartal coincidence count, the quartal fallbacks), the quartal oracle against every pitch set, the brute-force optimum, the shootout vamps pinned at tonic 2, inversions, drop-2, rule 2, ties |
| `test_timeline` | `tests/test_timeline.py`: N, swing positions at every tempo 60–95 (ten put the steps on half samples), the 8th warp, segments |
| `test_bass_lock` | `tests/test_bass_lock.py`: octave pairs, lock, re-strikes, keys and bass velocities for X, x and g |
| `test_sax_register` | `tests/test_sax_register.py`: pitches against the spec tables in every mood on both saxes, the mood's mode, octave fit (comfort, not playable), folds (playable, not comfort), note-offs, velocity, feels, a per-lane sax swing, one sax |
| `test_expression` | `tests/test_expression.py`: whole vibrato cycles, bend range, scoops read from the written line (across the seam too), vibrato, CC11, RPN |
| `test_prng` | `tests/test_prng.py`: the test vector and the reference sequence |
| `test_palette` | `tests/test_palette.py`: schema, licences, sha256, the dev palette's values, choke pairs (a lane may choke itself; no pair twice), pedalboard chains, fetch-mode sources under `$LOFI_ASSETS` (unset, empty, absolute, relative) |
| compile | `tests/test_compile.py`: every gate-passing fixture in five briefs against the spec (the §3.3 voicing pair for the brief's tonic and mode and the loop's style, chord tones plus quartal's in-mode tensions and no root outside quartal, §3.2 sax pitches, §3.1 step times, every open hat choked), the example's claims, a loop without `sax:`, humanize against the §3.6 key (drums, keys and bass; the sax untouched; legato joins; the seed formula; another keys instrument, voicing or kit name keeps every nudge), determinism |
| | `tests/test_compile_drums.py` (gains, layers, round robin across the seam with 2, 3 and 4 one-shots, the choke by either hat: consecutive open hats, a lone open hat, no other hat, the earliest cut, same-step hats under swing and humanize), `tests/test_compile_midi.py` (the dispatch model, file layout, seamless FluidSynth cycles), `tests/test_brief.py` (the sampler, its distribution, the documented key, independent fields and streams) |

## 9. Amendment V2 to instrument.md §3 Voicings
Voicing rules V2 are a Claude's default (2026-10-31). They replace instrument.md §3's voicing templates and keep its choice, steps 1–4. Like every default in TASTE.md, Shresth can override them; the keys and vamp shootouts still prune by ear. Sound claims: none.

**Why.** The second review (workflow `wf_ddc78253-994`, theory lens T1–T6) found that the §3 templates break TASTE.md's harmony row: "every chord at least a 7th; 9ths common", "chords that give a feeling", and the dealbreaker "sour notes". The code followed §3 literally, so each finding was rejected as a code bug, and the rules changed instead. The numbers in T1–T6 are the review's.
- **T1.** Rootless and spread took the first extension, so 13 = 9, m11 = m9 and maj7#11 = maj9. In 2 of the 4 styles the model's choice among those symbols changed nothing, and the 13 chords of the shootout vamps played as 9 chords.
- **T2.** Quartal dominants added an unnamed natural 13, outside the mode in both minor moods. Over every symbol, quartal held a tone outside both mode and chord in 126 of 462 (mode, numeral, quality) cases.
- **T3.** Quartal major chords had no 7th. Quartal could not tell maj7 from maj9, m7 from m9 or m11, or 9 from 13.
- **T4.** m7b5 never sounded its ♭5, in any style. The m7b5 shell equalled the m11 shell, and the dim7 shell had no ♭5.
- **T5.** Quartal candidates were inversions, so most chosen quartal voicings were not stacks of 4ths: 27 of 288 chosen voicings of the 12 shootout vamps at 12 tonics had every adjacent interval at 5–6 semitones.
- **T6.** The ♯9 of 7♯9 mostly sat a semitone under the 3rd: in 1,213 rootless, 1,275 shell and 1,013 quartal voicings of 1,836 per style.

Checked against the V2 oracle (`tests/helpers_voicing.py`), the §3 templates broke:
- R1 for 432 of 2,016 quartal chords (root × quality × mode above the root);
- R2 for 288 of those, and for 48, 24 and 48 of the 168 rootless, shell and spread chords (root × quality);
- R3 for 573 quartal chords, and R5 for 144 quartal, 12 rootless and 12 shell chords;
- R7 in rootless and spread (9 = 13, m9 = m11, maj9 = maj7#11) and in shell (m11 = m7b5).

Under V2, 264 of T5's 288 chosen quartal voicings are stacks; the other 24 are m9's rootless fallback on B, C and C♯.

**Rules.** Tones are pitch classes above the chord root. "Named" means in the quality's entry of the chord-tones table (§3.2, `theory.chords.QUALITY`), where 11 and 13 imply the 9.
- **R1 Guide tones.** Every voicing of every style holds the chord's 3rd (the 4th for 7sus) and its 7th (𝄫7 for dim7).
- **R2 Colour.** Every voicing holds the quality's top named extension: 13 for 13; 11 for m11 and m7b5; ♯11 for maj7#11; 9 for maj9, 9, m9 and 7sus; ♭9 for 7b9; ♯9 for 7#9. m7b5 and dim7 also hold their ♭5. The one exception: a shell is three notes, so the m7b5 shell is ♭3-♭5-♭7, since its ♭5 names it, and the 11 goes.
- **R3 No foreign tones.** A voicing holds only the quality's tones and named extensions, plus, in quartal, added tensions. An added tension is in the brief's mode and does not sit a semitone, in pitch class, above a chord tone the voicing also holds (the avoid-note rule: no natural 11 over a major 3rd). Nothing is outside chord + mode, ever: a borrowed chord keeps its own tones, and its added tensions must still be in the mode.
- **R4 Styles.**
  - rootless: no root. With extensions: the 3rd, the 7th and the named extensions (at most two, the top one always), plus m7b5's ♭5. Without: 3rd-5th-7th, as before.
  - shell: 3rd + 7th + the top extension; ♭3-♭5-♭7 for m7b5 and ♭3-♭5-𝄫7 for dim7.
  - spread: drop-2 of each inversion of the new rootless voicing (the rule itself is unchanged).
  - quartal: every stack of 3 or 4 tones from R3's tone set that holds R1 and R2, whose adjacent intervals are all 5 or 6 semitones, except at most one of 3 or 4. A chord with no stack is voiced rootless, and nothing is logged: it is a voicing, not a repair.
- **R5 ♯9 above the 3rd.** In a 7#9 voicing the ♯9 sounds above the 3rd, never a semitone under it.
- **R6 Unchanged.** The register C3–C5 (48–72); no interval of 2 semitones or less below 55; no 13-semitone pair outside 7b9; the voice-leading cost and its tie-break; the two vamp chords are voiced jointly.
- **R7 Distinct symbols.** In rootless, shell and spread, two qualities whose top named extensions differ never compile to the same voicing for the same root and mode (13 vs 9, m11 vs m9, maj7#11 vs maj9, m7b5 vs m11). Quartal stacks may coincide, because an added tension can be another quality's named one.

**Templates** (semitones above the root; **bold** where V2 changed §3, with §3's value after the arrow). Every inversion is a candidate, and spread is drop-2 of each rootless inversion.

| quality | named tones | every voicing holds (R1, R2) | rootless | shell |
|---|---|---|---|---|
| maj7 | 0 4 7 11 | 3, 7 | 4 7 11 | 4 11 |
| maj9 | 0 4 7 11 14 | 3, 7, 9 | 4 11 14 | 4 11 14 |
| maj7#11 | 0 4 7 11 14 18 | 3, 7, ♯11 | **4 11 14 18** ← 4 11 14 | 4 11 18 |
| 7 | 0 4 7 10 | 3, ♭7 | 4 7 10 | 4 10 |
| 9 | 0 4 7 10 14 | 3, ♭7, 9 | 4 10 14 | 4 10 14 |
| 13 | 0 4 7 10 14 21 | 3, ♭7, 13 | **4 10 14 21** ← 4 10 14 | 4 10 21 |
| 7sus | 0 5 7 10 14 | 4, ♭7, 9 | 5 10 14 | 5 10 14 |
| 7b9 | 0 4 7 10 13 | 3, ♭7, ♭9 | 4 10 13 | 4 10 13 |
| 7#9 | 0 4 7 10 15 | 3, ♭7, ♯9 above the 3 | 4 10 15 (inversion 0 only) | 4 10 15 (inversion 0 only) |
| m7 | 0 3 7 10 | ♭3, ♭7 | 3 7 10 | 3 10 |
| m9 | 0 3 7 10 14 | ♭3, ♭7, 9 | 3 10 14 | 3 10 14 |
| m11 | 0 3 7 10 14 17 | ♭3, ♭7, 11 | **3 10 14 17** ← 3 10 14 | 3 10 17 |
| m7b5 | 0 3 6 10 17 | ♭3, ♭5, ♭7, 11 (the shell: ♭3, ♭5, ♭7) | **3 6 10 17** ← 3 10 17 | **3 6 10** ← 3 10 17 |
| dim7 | 0 3 6 9 | ♭3, ♭5, 𝄫7 | 3 6 9 | **3 6 9** ← 3 9 |

Quartal has no fixed template, because its stacks depend on the mode above the root. Some of them:
- Imaj7 in ionian: 7-3-13-9 [11, 16, 21, 26], 9-5-7-3 [2, 7, 11, 16], 7-3-5 [11, 16, 19] and 5 more, against §3's 3-13-9-5 [4, 9, 14, 19], which had no 7th;
- V13 in ionian: 5-♭7-3-13, ♭7-3-13, ♭7-3-13-R and ♭7-3-13-9;
- Im9 in dorian has one stack, ♭7-♭3-13-9 [10, 15, 21, 26], and in aeolian one, ♭7-♭3-♭13-9 [10, 15, 20, 26];
- m7b5 has one stack in every mode, 11-♭7-♭3-♭5 [5, 10, 15, 18], and R6 rejects it (its 11 and ♭5 are 13 semitones apart).

**Readings** (where V2's words leave a choice). `tests/test_voicing.py` pins each one through the oracle. The counts are the oracle's, over every mode × numeral × quality: 3,549 quartal stacks in all, before R5 and R6.
- **The root.** R3 allows the quality's tones, and the root is one, so a quartal stack may hold it: 666 of the 3,549 stacks do. Rootless, shell and spread never hold it.
- **Added tensions** are the 9ths, 11ths and 13ths, ♭9 9 ♯9 11 ♯11 ♭13 13, that the symbol does not name. So quartal never adds a ♭7 to a major 7th chord, a 3rd to 7sus or a major 7th to dim7, even where the mode holds them. Reading "any tone of the mode" instead would add 194 stacks, every one of them holding one of those three tones.
- **"A chord tone the voicing also holds"** is a named tone, extensions included, held by the keys voicing itself. The bass is not counted, so a stack that leaves out the root may add a ♭9: 450 stacks do, all on chords other than 7b9.
- **A stack** holds each tone once.
- **"If no stack exists"** is read per root, after R5 and R6: a chord whose stacks all break R6, or none of which fits C3–C5 on its root, takes its rootless candidates. That happens in 495 of the 5,544 mode × numeral × quality × tonic cases: every m7b5 (396), and m9 on B, C and C♯ (99), whose stacks span 16 semitones.
- **R5** is checked for every quality with a major 3rd. Only 7#9 can hold both a ♯9 and a major 3rd, so R5 rejects rootless and shell inversions 1 and 2 of 7#9, and spread's drop-2 of inversion 1, which R6 rejects anyway.
- **R7.** In rootless, shell and spread no two qualities share a voicing on any root, which covers R7's pairs and the qualities whose top extensions are equal too. In quartal, 2,430 of the 36,036 (mode, numeral, tonic, quality pair) cases share at least one candidate. The 13 pairs are 7/9, m7/m11 and maj7/maj9 (240 each), 7/13 and 7/7#9 (228), 7/7b9 and maj7/maj7#11 (216), maj9/maj7#11 (198), m7/m9 (180), 9/13 (171), 7b9/7#9 (143), 13/7b9 (70) and 13/7#9 (60). No two qualities share every stack in any mode × numeral.

**What V2 leaves open: the two chords of a vamp can get the same pitches.** V2 constrains each chord, and the cost then prefers the least movement. With quartal's added tensions, the two chords sometimes meet: in 29 of the 576 (shootout vamp, style, tonic) cases, all quartal, both chords play the same notes and only the bass moves. These are Royal-half on 10 tonics, Step on 10 and Rise on 9 (at tonic 2: Royal-half F#3 B3 E4 A4, Step D3 G3 C4 F4), and `span1_dotted.ls`. Under §3 it was 0 of 576. Each of the 29 holds one chord's root; with the root left out of quartal stacks it would be 0 of the 144 quartal cases.

**The 12 shootout vamps at tonic 2** (§6 gives no tonic; its /1 and /2 are spans), each in its mood, as the compiler voices them. `tests/test_voicing.py::test_the_shootout_vamps_at_tonic_2_are_pinned` holds these.

| vamp | mood | rootless | shell | spread | quartal |
|---|---|---|---|---|---|
| Aruarian IIm9 V13 | hopeful | G3 D4 F#4 \| G3 B3 C#4 F#4 | G3 D4 F#4 \| G3 C#4 F#4 | D3 G3 F#4 \| C#3 G3 B3 F#4 | D3 G3 C#4 F#4 \| E3 G3 C#4 F#4 |
| Royal-half IVmaj7 IIIm7 | hopeful | B3 D4 F#4 \| A3 C#4 E4 | B3 F#4 \| A3 E4 | F#3 D4 B4 \| E3 C#4 A4 | F#3 B3 E4 A4 \| F#3 B3 E4 A4 |
| Two-of-us IVmaj7 III7 | hopeful | B3 D4 F#4 \| Bb3 C#4 E4 | B3 F#4 \| Bb3 E4 | F#3 D4 B4 \| E3 C#4 Bb4 | F#3 B3 E4 \| F#3 Bb3 E4 |
| Lift Imaj9 IVmaj9 | hopeful | F#3 C#4 E4 \| A3 B3 F#4 | F#3 C#4 E4 \| A3 B3 F#4 | E3 C#4 F#4 \| F#3 B3 A4 | E3 A3 C#4 F#4 \| F#3 B3 E4 A4 |
| i–IV Im9 IV13 | wistful | F3 C4 E4 \| F3 A3 B3 E4 | F3 C4 E4 \| F3 B3 E4 | F3 E4 C5 \| F3 B3 E4 A4 | C3 F3 B3 E4 \| D3 F3 B3 E4 |
| Pulse Im9 IV9 | wistful | F3 C4 E4 \| A3 B3 F4 | F3 C4 E4 \| A3 B3 F4 | F3 E4 C5 \| A3 F4 B4 | C3 F3 B3 E4 \| F3 B3 E4 A4 |
| Step Im11 IIm7 | wistful | E4 F4 G4 C5 \| D4 G4 B4 | C4 F4 G4 \| D4 G4 | G3 E4 F4 C5 \| G3 D4 B4 | D3 G3 C4 F4 \| D3 G3 C4 F4 |
| Float Im9 bVIImaj7 | wistful | F3 C4 E4 \| G3 B3 E4 | C4 E4 F4 \| B3 E4 | F3 E4 C5 \| G3 E4 B4 | C3 F3 B3 E4 \| D3 G3 B3 E4 |
| Drift Im9 bVImaj7 | melancholy | F3 C4 E4 \| F3 A3 D4 | F3 C4 E4 \| A3 D4 | F3 E4 C5 \| F3 D4 A4 | C3 F3 Bb3 E4 \| C3 F3 A3 D4 |
| Plagal Im9 IVm9 | melancholy | F3 C4 E4 \| A3 Bb3 F4 | F3 C4 E4 \| A3 Bb3 F4 | F3 E4 C5 \| F3 Bb3 A4 | C3 F3 Bb3 E4 \| F3 Bb3 E4 A4 |
| Backdoor IVm9 bVII13 | melancholy | Bb3 F4 A4 \| Bb3 D4 E4 A4 | Bb3 F4 A4 \| Bb3 E4 A4 | F3 Bb3 A4 \| E3 Bb3 D4 A4 | F3 Bb3 E4 A4 \| G3 Bb3 E4 A4 |
| Rise bVImaj7 bVII13 | melancholy | F3 A3 D4 \| E3 A3 Bb3 D4 | D3 A3 \| E3 A3 Bb3 | F3 D4 A4 \| E3 Bb3 D4 A4 | A3 D4 G4 \| Bb3 E4 A4 |

Spelling: one fixed name per pitch class (C C# D Eb E F F# G Ab A Bb B), C4 = 60.

**Code and renders.** `voice_vamp(first, second, tonic, mode, style)` takes the brief's mode; `voicing.chord_scale(numeral, mode)` gives the mode above a chord's root. Of the 8 golden render cases on the laptop key, 7 changed, and only through their keys stem (`tests/golden/render_hashes.json`; the bass, sax and drum stems hash the same). `one_bar_base.ls` (IIm9 V13, shell) is unchanged, since its shells did not change.
