# CONVERTER.md: the six SFZ shootout banks as SoundFont 2.04

FluidSynth 2.6.1 cannot load SFZ, and six shootout candidates are SFZ instruments: Weresax, MTG Solo Saxophones
(tenor and alto, two banks), jRhodes3d, the VCSL Upright Piano Knight, the VSCO 2 contrabass pizzicato and the
Greg Sullivan Wurlitzer. `docs/SHOOTOUT_ASSETS.md` explains why Polyphone cannot do the conversion. This converter
is the default instead. It is a small, deterministic SFZ → SF2 converter in Python with no dependency beyond the
scorer env's numpy and soundfile, so it runs on the laptop and on a compute node.

- **Code:** `src/lofi_rl/convert/`:
  - `sfz.py`: parsing, ARIA evaluation at the controller defaults and region selection;
  - `mapping.py`: regions to zones and the fits;
  - `sf2.py`: the byte writer and lossless sample reading;
  - `banks.py`: the bank table and the pipeline;
  - `__main__.py`: the CLI.
- **Tests:**
  - `tests/test_convert.py`, synthetic, in the default suite (64 tests, about 1.5 s, three of them through
    FluidSynth). 38 of them are the writer's refusals (§6) and the edges it must accept;
  - `tests/test_convert_banks.py`, the real banks, marked `cluster` (17 tests, 14–17 s): run them with
    `LOFI_ASSETS_BIG=assets/big pytest -m cluster tests/test_convert_banks.py -s`. They also parse all 23 `.sfz`
    files in the six banks' folders and check that no opcode of a region kept at the defaults goes unexplained,
    and they check the loops the Wurlitzer's and MTG's FLACs carry (§6);
  - `tests/helpers_sf2.py`, shared by both: an SF2 reader, a FLAC writer and a FLAC smpl reader of the tests' own.
- **Sound claims:** none. Everything below is a number read from a file, a render or a test. Whether a converted
  bank is any good is the shootout's question, by ear.

## 1. Use

```
python -m lofi_rl.convert <bank-id> --big assets/big --out <file.sf2> [--json report.json]
python -m lofi_rl.convert all --big assets/big --out-dir <folder>          # writes <bank-id>.sf2 for each
```

**What the CLI prints, per bank:**
- the output's sha256 and size;
- regions read, kept and dropped (by reason);
- the key range and the velocity layers;
- the sample count, rates, bit depths and channels;
- how many zones play a loop, and where each source file's loop comes from (`loop_start`/`loop_end`, its smpl
  chunk, or none);
- the velocity-curve fit and its error, and the envelope-time fits and their errors, each that of the values
  written;
- the attenuation shift;
- every opcode dropped from a kept region, and why;
- the presets as `palette.sf2_presets` reads them back from the written file.

`--json` writes the same report.

**The jRhodes3d conversion is CC BY-NC 4.0, like its samples.** It never goes into the repo; write it under
`assets/big/` or a scratch folder. The real-bank tests write every conversion to `tmp_path`.

Bank ids: `weresax-alto`, `mtg-tenor`, `mtg-alto`, `jrhodes3d-mono`, `vcsl-upright-knight`,
`vsco2-contrabass-pizz` and `gsullivan-wurlitzer`. Each becomes **one preset at bank 0, program 0**, named after
the bank. `vsco2-contrabass-pizz` is the exception: its preset is `vsco2-cbass-pizz`, because SF2 names hold 19
characters plus a terminator. INAM holds the full id.

## 2. Pipeline

1. **Parse** (`sfz.parse_sfz`).
   - **Headers and inheritance.** It reads `<control> <global> <master> <group> <region> <curve>`. A region
     inherits global < master < group < region opcodes. A new `<global>` clears master and group; a new
     `<master>` clears group.
   - **`#define $VAR value`.** The value is substituted textually in every later line: opcode names, values and
     include paths. The longest defined name wins, so `$VELTRACK` beats `$VEL`.
   - **`#include "file"`.** The path is looked up beside the including file first, then beside the root SFZ.
     MTG needs the second: its `Data/group_ten.txt` includes `"Data/ten_p_rr1.txt"`.
   - **Comments and values.** `//` and `/* */` are comments. A value runs to the next `name=` or `<header>`, so
     sample paths may hold spaces. Either slash works.
   - **`default_path`.** It is a `<control>` opcode, applied as it stands at each region.
   - **Keys.** Note names put `c4` at 60, with `octave_offset` and `note_offset`. `key=` sets `lokey`, `hikey`
     and `pitch_keycenter`.
   - **Sample paths** are matched component by component to the names on disk: the exact name if present, else
     the one name equal ignoring case. So the laptop (case-insensitive) and the cluster (case-sensitive) read
     the same file under the same spelling.
2. **Evaluate at the controller defaults** (`sfz.Evaluator`). Every controller reads its `set_ccN` (0–127) or
   `set_hdccN` (0–1) value, or 0.
   - **CC modulations.** `X_onccN` and the SFZ v1 `XccN` add depth × curve(CC/127) to X. The curve is
     `X_curveccN`: ARIA's predefined curves 0 linear, 1 bipolar, 2 inverted, 3 inverted bipolar, 4 x², 5 √x,
     6 √(1−x), or the file's own `<curve>` (points linear in between, 0 at 0 and 1 at 127 unless given).
   - **Amplitude** multiplies instead: amplitude% × Π depth% × curve.
   - **CC crossfades** (`xfin_/xfout_loccN/hiccN`) give a gain, with the power curve by default.
   - **Conditions.** `loccN`/`hiccN` enable or disable the region.
   - **Per-note sources.** ARIA extended CC 131 (note-on velocity) and 133 (key number), and the v1 `ampeg_vel2*`
     opcodes, are not defaults. They stay velocity and key terms for the mapping (§4). The other extended CCs
     (pitch bend, pressures, randoms, alternate) read 0.
   - **Checked against the banks.** The curve and extended-CC readings are ARIA's as recalled, not re-checked
     against its documentation in this work. The kept regions use curves 1, 2 and 4 and CCs 131 and 133, and each
     reading agrees with the bank's own labels:
     - MTG's "Veltrack" knob, `amp_veltrack_oncc99=-100` through curve 2 at `set_hdcc99=1`, gives 100 %;
     - its "Pan" knob, curve 1 at 0.5, gives the centre;
     - its CC131 attack shortens as velocity rises;
     - the Wurlitzer's CC133 hold and decay shorten up the keyboard.
3. **Select** (`sfz.select`).
   - **Kept:** round robin `seq_position` 1; the `lorand`/`hirand` region that covers random value 0; the
     default keyswitch (`sw_default`, matched against `sw_last`); `trigger=attack` and `first`.
   - **Dropped** (counted by reason):
     - `trigger=release`, `release_key` and `legato`;
     - CC-triggered regions (`on_loccN`/`on_hiccN`);
     - generated samples (`sample=*noise` and the like);
     - regions whose gain at the defaults is 0, a check made before the sample is looked up, so unfetched silent
       layers do not matter;
     - regions failing a CC condition;
     - each bank's own rule (MTG: `master_label=noises`).
   - **Expectations.** Each bank also states what its reduction must keep (sample pattern and count); the
     conversion stops if it does not.
4. **Map and write** (`mapping.zone_spec`, `banks.convert_sfz`, `sf2.sf2_bytes`): §3–§6.

## 3. Opcodes to generators

| SFZ (at the defaults) | SF2 2.04 | Notes |
|---|---|---|
| `lokey`/`hikey`, `lovel`/`hivel` | keyRange (43), velRange (44) | always written, first in the zone |
| `pitch_keycenter` (default 60; `sample` = the smpl chunk's unity note, from a WAV or a FLAC) | overridingRootKey (58) | always written; the sample header holds the first zone's root |
| `tune` + 100 × `transpose` (cents) | coarseTune (51) + fineTune (52) | fine is kept within ±99; MTG's and VCSL's per-region `tune=` survive as fineTune |
| `pitch_keytrack` | scaleTuning (56) | written when not 100 |
| `volume` + 20 log10(`amplitude` × crossfade gain) | initialAttenuation (48) | negated, shifted bank-wide, and written as cB / 0.4 (§5) |
| `pan` | pan (17), × 5 | a stereo sample becomes a left/right zone pair at −500/+500 (plus the pan) |
| `ampeg_delay/attack/hold/decay/release` (s) | delay/attack/hold/decay/releaseVolEnv (33–36, 38), timecents | 1200 log2(s), inside SF2 2.04 §8.1.3's ranges: −12000..5000 for delay and hold (18.0 s at most), −12000..8000 for the others (101.6 s). A longer SFZ time is written at the ceiling and listed as an approximation; 0 s keeps the default −12000 |
| `ampeg_sustain` (%) | sustainVolEnv (37) | −200 log10(s/100) / 0.96 cB; 0 % → 1000 (§5) |
| `ampeg_hold`/`ampeg_decay` key terms (CC 133) | holdVolEnv/decayVolEnv + keynumToVolEnvHold/Decay (39, 40) | fitted per zone over its keys (§4) |
| velocity terms on envelope times (CC 131, `ampeg_vel2*`) | velocity → *VolEnv modulators | fitted per zone over its velocities (§4) |
| `amp_veltrack`, `amp_velcurve_N` | velocity → initialAttenuation modulators | fitted per zone (§4); none when it is FluidSynth's default curve |
| `loop_mode`, `loop_start`/`loop_end`, the sample's smpl loop (a WAV's own chunk, or the one a FLAC carries in a `riff` block, §6) | sampleModes (54: 1 continuous, 3 sustain), sample loop, loop offsets (2, 3, 45, 50) | SFZ plays a sample's own loop unless `loop_mode` says otherwise; `loop_end` is inclusive, SF2's end exclusive |
| `offset`, `end` | startAddrsOffset (+ coarse), endAddrsOffset (+ coarse) | |
| `sample` | sampleID (53) | last in the zone; one SF2 sample per channel per file, shared by every zone that uses the file |

**Dropped from kept regions.** Each is listed per bank in the CLI report, with its value at the defaults where
that shows whether anything audible was lost:
- **LFOs** (`lfoNN_*`, `pitchlfo_*`, `amplfo_*`): ARIA's LFOs do not map onto SF2's two fixed ones.
- **EQ and ARIA variable modulators** (`eqN_*`, `varNN_*`).
- **Filters and filter envelopes.** No kept region of the six banks uses one.
- **Pitch envelope, `pitch_veltrack`, `pitch_random`.**
- **`bend_up`/`bend_down`.** The compiler sets the bend range by RPN 0 (`sax_expr.bend_range`).
- **Voice groups and polyphony** (`group`, `off_by`, `off_mode`, `off_time`, `off_shape`, `polyphony`,
  `note_polyphony`), not mapped (exclusiveClass is not used):
  - FluidSynth already releases a voice when its key is struck again. Measured: a noise voice struck twice with
    no note-off stayed at one voice's level, −30.88 dB against −30.92 dB, not +3 dB;
  - the compiler's sax line is monophonic, with note-offs before the next note.
- **`sustain_sw`/`sostenuto_sw`.** FluidSynth always honours CC64; the compiler sends none.
- **`ampeg_dynamic`.** SF2 envelopes are fixed at note-on.
- **Velocity and key crossfades.** SF2 zones switch; they cannot fade.
- **Opcodes inside `<control>`** that no region inherits (jRhodes3d's half-pedal pair).

## 4. Velocity and envelope fits

**What FluidSynth 2.6.1 does.** The model was measured on its renders of synthetic SF2s on 2026-10-31: 127
notes, one per velocity, RMS over whole periods of a looped sine. The measured levels come in whole centibels.
- **Negative linear velocity source:** (127 − v)/128. Absolute levels put it at 0 at v = 127.
- **Negative concave source:** the 128-point table c[127 − v], where c[127 − i] = −(400/960) log10(i/127). With
  amount 960, the default modulator, this is exactly SFZ's 100 % `amp_veltrack` curve, 40 log10(127/v) dB.
- **Products.** A modulator with an amount source multiplies the two (linear × linear matched within 1 cB at
  amount 1440).
- **Saturation.** Concave and convex sources saturate at 127/128, so the bases below avoid them near v = 127.
- **Rounding.** The measured levels step in whole centibels (0.1 dB) of total attenuation, consistent with the
  total being rounded after FluidSynth's own CC7 = 100 default (41.5 cB) is added.
- **No velocity filter.** A noise sample at v = 64 and v = 100 matched v = 127 in the 200 Hz–2 kHz, 8–12 kHz and
  14–19 kHz bands to 0.01 dB. FluidSynth does not apply SF2's default velocity → filter modulator, so the
  converter writes no override for it.

**The SFZ target.** For each region the target is c(v) = (v/127)², or the `amp_velcurve_N` points (linear, 0 at 0
and 1 at 127 unless given). `amp_veltrack` t blends it in amplitude: g(v) = 1 − t(1 − c(v)) for t ≥ 0. The target
attenuation is −20 log10 g(v).
- **Source of the formula.** It is sfizz's reading as recalled, not re-checked against sfizz's source in this
  work. Weresax sets both `amp_veltrack=77` and `amp_velcurve_63=1` on one layer, which only makes sense if the
  two combine.
- **If ARIA instead ignores `amp_veltrack` under a curve,** only Weresax's p layer would change: v/63 instead of
  1 − 0.77(1 − v/63).

**The fit.** Each zone gets a constant plus four instrument-zone modulators to initialAttenuation:
- concave (its source, destination and empty amount source replace FluidSynth's default velocity modulator);
- linear;
- linear × linear;
- concave × linear.

The amounts are a minimax fit (Lawson's reweighted least squares, 120 iterations, pure Python floats) over **that
zone's velocities only**, rounded to integers, and the constant is refitted on the residual. A zone at 100 % with
no curve gets no modulators at all: the default already is the curve. A flat zone (t = 0) gets the overriding
concave modulator at amount 0.

**Velocity-curve error against the SFZ curve** (fit, before FluidSynth's 0.1 dB rounding):

| bank | curves | max error |
|---|---|---|
| weresax-alto | f 64–127, t 66 % | 0.021 dB (amounts −237, 159, 41, 5) |
| weresax-alto | p 0–63, t 77 % with `amp_velcurve_63=1` | 0.058 dB (amounts −982, 250, −59, 976; constant −47 cB) |
| the other six | t 100 % (MTG: `amp_veltrack_oncc99=-100` through curve 2 at hdcc 1; Wurlitzer: 0 + `amp_veltrack_oncc99=100` at hdcc 1) | 0 (FluidSynth's default) |

**Measured through FluidSynth.** On every converted bank, every velocity of each zone at one key was rendered and
compared within the zone (same sample), relative to the zone's top velocity. This was done on the first conversions;
the later fixes (§6 loops, §4 attack) left every attenuation generator and velocity modulator as it was (a
structural diff of the old and new files of the three banks they changed):
- the largest deviation from the SFZ curve was 0.101 dB (Wurlitzer), with 0.083 dB on Weresax;
- VSCO 2 measured 0.06 dB with notes 4 s apart; at 0.1 s apart, its 3 s release overlapped the next note;
- on a synthetic bank with Weresax's two curves, the measured maximum over v = 1–127 was 0.10 dB.

`test_fluidsynth_levels_follow_the_sfz` checks a 66 % curve, a −12 dB region and a 50 % sustain through
FluidSynth, each within 0.15 dB. It fails, as it should, if either level correction in §5 is removed (−4.8 and
−5.7 dB measured).

**Envelope times.**
- **The base is a generator,** so it stays inside its stage's §8.1.3 range: −12000..8000 tc for attack, decay and
  release, −12000..5000 for delay and hold. When the free fit puts it outside, it is fixed at the nearer bound and
  only the key or velocity terms are refitted.
- **FluidSynth clamps the sum,** generator plus modulators, not the generator alone, and the error model clamps
  the same sum. Measured on the synthetic copy below: with its base patched from −12000 to −12039 tc, every attack
  at v = 1, 10, 22, 25 and 50 played one 64-sample block (1.45 ms) shorter.
- **Errors.** Every error reported is that of the integer values written, at every key or velocity of the zone.
- **Key terms** (Wurlitzer: `ampeg_hold=5 ampeg_holdcc133=-5`, `ampeg_decay=25 ampeg_decaycc133=-25`). SF2 gives
  tc(k) = base + keynum × (60 − k). This is fitted minimax in timecents per zone over the zone's keys. The largest
  relative error over the Wurlitzer's keys is 0.27 % (hold) and 0.29 % (decay); no Wurlitzer base reaches a bound.
- **Velocity terms** (MTG: `ampeg_attack_oncc131=0.1` through the bank's curve 8, so 0.1 × max(0, 1 − v/100) s).
  These become modulators on attackVolEnv, base + a·lin + b·lin², fitted minimax in seconds (Lawson reweighting by
  the seconds error).
  - **MTG's p layer.** The SFZ attack reaches 0 s at v = 100, so the free fit's base is −12039 tc, below the floor.
    The first conversion clamped it after the fit: the file then sat up to 7.52 ms from the SFZ (85.5 ms against
    78.0 at v = 22) while the report said 5.66 ms. With the base fixed at −12000, the refitted amounts are 16199
    (linear) and −8299 (linear²), and the written values are at most 5.68 ms from the SFZ, at v = 100, where the
    SFZ asks for 0 s.
  - **The f layer** (101–127) has attack 0 and needs no modulator; it gets the −12000 tc floor, 0.98 ms.
  - **Measured through FluidSynth** on a synthetic copy (a looped sine with MTG's opcodes, each velocity 1–100,
    the linear ramp read from the envelope): within 1.40 ms of the file's generators (FluidSynth runs the attack in
    64-sample blocks) and within 6.70 ms of the SFZ attack.

## 5. Levels

**Static level.** volume, amplitude and crossfades give a static level in dB per region, and the velocity fit adds
its constant.
- **Why a shift.** SF2 attenuation cannot be negative, while SFZ volume can be positive: VCSL's kept regions run
  from +3.7 to +25.2 dB, and VSCO 2's from +5 to +24 dB.
- **The shift.** The whole bank moves by one whole number of cB: the smallest shift that keeps every generator ≥ 0
  and every note's total attenuation ≥ 0 at every velocity it plays. The loudest note then sits at 0–1 cB.
- **Uniform terms** cancel in the shift, such as `amplitude_oncc7` at `set_cc7=100`. FluidSynth's own
  CC7/CC10/CC11 default modulators then act on the live controllers, as they do for every SF2. MTG's
  `amplitude_oncc11` with curve 4 is the same x² law as FluidSynth's default CC11.

**FluidSynth multiplies every initialAttenuation generator by 0.4.** EMU hardware did the same
(EMU_ATTENUATION_FACTOR); modulators are not scaled.
- **Measured:** with the same modulators on a zone, generator values 20, 47 and 100 added 8, 19 and 40 cB.
- **So the converter writes cB / 0.4.** The generator's 0–1440 range then holds at most 57.6 dB between the
  bank's loudest and quietest zone. VCSL needs 21.6 dB, the most of the six.

**Sustain.** FluidSynth maps a sustainVolEnv of 1000 cB onto its 96 dB envelope range, so the converter writes
cB / 0.96. Measured: 50 %, 25 % and 10 % sustain gave −6.0, −12.0 and −19.9 dB, against targets of −6.02, −12.04
and −20.0.

Times map 1:1. SF2's decay and release run at 96 dB per stated time in FluidSynth, while SFZ players use their own
curves.

## 6. Samples and bytes

- **Decoding.** WAV and FLAC are decoded by libsndfile as int32, which arrive as 16-bit << 16 or 24-bit << 8, and
  shifted back to 24-bit points. 32-bit and float sources are refused, not quantized.
- **Storage.** smpl holds the high 16 bits. When any source is 24-bit, sm24 holds the low byte, so a 24-bit point
  is (smpl << 8) | sm24, lossless. The tests read every point back with a reader of their own.
- **Padding.** Each sample is followed by at least 46 zero points. The total is kept even, so sm24 is exactly
  half of smpl and needs no pad byte.
- **Rates.** Each sample keeps its source rate in its header (MTG 48 kHz); FluidSynth resamples. The pitch of a
  48 kHz, 24-bit sine with `tune=-30` rendered within 0.05 cents at keys 60–79.
- **Stereo.** A stereo file becomes a left (type 4) and a right (type 2) sample linked both ways, with zones at
  pan −500/+500.
- **Loops.**
  - The source is `loop_start`/`loop_end` when given, else the first forward loop of the sample's smpl chunk.
  - **Where the smpl chunk is.** A WAV holds its own. A FLAC made by `flac --keep-foreign-metadata` keeps each of
    its WAV's chunks except the points (the RIFF header, fmt, the data chunk's 8-byte header, smpl) in an
    APPLICATION metadata block with the id `riff`. libsndfile does not read these blocks, so `sf2.flac_smpl` does.
  - **The banks.** Weresax's 64 kept WAVs, the Wurlitzer's 42 FLACs and MTG's 132 FLACs each carry one forward
    loop; jRhodes3d's FLACs, VCSL's WAVs and VSCO 2's WAVs carry none.
  - **SFZ plays a sample's own loop by default.** So the Weresax and Wurlitzer zones loop (sampleModes 1), while
    MTG's NL presets say `loop_mode=no_loop`, and their sample headers hold the FLAC loops unplayed.
  - **The first conversion read smpl chunks from WAVs only.** The Wurlitzer's zones were then unlooped, and a
    held note stopped when its sample ran out: key 74 at velocity 20 has 0.34 s of points, against an SFZ hold of
    2.09 s.
  - A loop that ends within 8 points of the sample's end gets guard points that continue the loop, so 8 points
    follow it (SF2 2.04 §7.10). Six Wurlitzer samples need them (1, 3 and 8 points).
  - Unlooped samples get a valid loop that is never played.
- **What the writer refuses** (`sf2.SF2Error`; each case is a test). FluidSynth 2.6.1 discards some of these while
  loading, with a warning, and applies others differently, so a file holding them would not play as written:
  - in a preset zone, the instrument-only generators of §8.1.2 (sample addresses and loops, keynum, velocity,
    sampleModes, exclusiveClass, overridingRootKey). Measured: a preset-zone overridingRootKey was discarded with
    the warning "Some invalid generators were discarded";
  - in an instrument zone, a value outside its §8.1.3 range (pan ±500, attenuation and sustainVolEnv 0..1440,
    fineTune ±99, coarseTune ±120, timecents up to 5000 for delay and hold and 8000 for the others, and so on),
    and sampleModes 2, which is unused;
  - a modulator source that §8.2.1 does not allow: a general controller other than 0, 2, 3, 10, 13, 14 and 16,
    CC 0, 6, 32, 38, 98–101 or 120–127, or a curve type above 3. Links are not written. Measured: an amount
    source of general controller 5 was removed with the warning "Invalid modulator";
  - a transform other than 0 or 2 (§8.3). Measured: transform 1 left the level where no modulator puts it
    (−32.82 dB, against −35.12 with transform 0), and FluidSynth logged nothing;
  - a modulator to keyRange, velRange, instrument, sampleID or an unknown generator;
  - a sample under 48 points, and a loop without 8 points before it, 32 in it and 8 after it (§7.10);
  - **FluidSynth's rule, not the spec's:** a modulator whose primary source is "no controller". The spec reads it
    as 1, but FluidSynth 2.6.1 warns about it; the constant belongs in the generator.

  A preset zone's values are offsets added to the instrument's, so they are not range-checked. None of the seven
  conversions changed when these checks were added: switching off the two fixes of this section and §4
  reproduces the earlier sha256s exactly.
- **Names** are file stems: ASCII, at most 19 characters with the L/R suffix. A prefix shared by every stem is
  dropped when stems are too long (`BKCtbss_Pizz_A#0_v1_rr1` → `A#0_v1_rr1`).
- **Chunks:**
  - `RIFF sfbk`: INFO (ifil 2.04, isng EMU8000, INAM, ICOP with the source licence, ICMT, ISFT), no ICRD date;
  - `sdta`: smpl [sm24];
  - `pdta`: phdr pbag pmod pgen inst ibag imod igen shdr, each with its terminal record.
  - Generators are written by id, after keyRange and velRange and before sampleID or instrument. Modulators are
    sorted; two with the same source, destination and amount source in one zone are refused.

## 7. Determinism

**The guarantee.** The bytes are a function of the SFZ text and the sample files alone. Nothing reads the clock,
the environment, a dict or set order, or an absolute path: every path written is bank-relative, and every list is
sorted.

**Tested:**
- two conversions in one process give the same bytes, and two fresh processes with `PYTHONHASHSEED` 0 and 4242
  give the same sha256 (`test_bytes_are_identical_across_runs_processes_and_hash_seeds`);
- the CLI's sha256s matched the real-bank test's in-process conversions.

**Across machines.** The fits use plain IEEE floats, with no BLAS, and `math.log10`. They can differ between
machines only if libm's log10 differs in the last bit at a point where a rounded amount sits on a half.
`test_velocity_fits_are_pinned_for_the_weresax_curves` pins the amounts, so such a machine fails a test before a
bank changes. The cluster's first conversion should still be compared with the laptop's sha256 (§9).

**Cost** (laptop, `/usr/bin/time -l`): the seven conversions in one process (`all`) took 4.2 s and 4.8 s in two runs,
peaking at 1.6 and 1.1 GB of resident memory. VCSL alone took 1.0 s and peaked at 1.45 GB: the decoded samples
plus the 365 MB file.

## 8. Per bank (converted 2026-10-31, laptop)

| bank | SFZ | regions read / kept | keys | velocity layers | SF2 samples / zones | bytes | sha256 |
|---|---|---|---|---|---|---|---|
| weresax-alto | `Programs/Sax.sfz` | 257 / 64 | 49–80 | 0–63, 64–127 | 64 / 64 (24-bit, 44.1 k) | 49,835,920 | `d4f79749…` |
| mtg-tenor | `MTG Tenor Sax (NL).sfz` | 493 / 66 | 44–76 | 0–100, 101–127 | 66 / 66 (24-bit, 48 k) | 39,541,982 | `38fcf328…` |
| mtg-alto | `MTG Alto Sax (NL).sfz` | 500 / 66 | 49–81 | 0–100, 101–127 | 66 / 66 (24-bit, 48 k) | 40,870,876 | `2b338b89…` |
| jrhodes3d-mono | `jRhodes3d-mono-no-xfade.sfz` | 75 / 75 | 21–108 | 1–47, 48–72, 73–95, 96–111, 112–127 | 65 / 75 (16-bit) | 78,655,288 | `d662cb23…` |
| vcsl-upright-knight | `Upright Piano, Knight.sfz` | 143 / 90 | 21–109 | 0–83, 84–127 | 180 / 180 (24-bit stereo) | 364,927,584 | `151ea259…` |
| vsco2-contrabass-pizz | `ContrabassPizz.sfz` | 40 / 20 | 24–60 | 0–62, 63–127, 0–127 | 40 / 40 (16-bit stereo) | 10,164,334 | `27f341ab…` |
| gsullivan-wurlitzer | `Wurlitzer EP200.sfz` | 48 / 48 | 33–96 | 1–37, 38–65, 66–89, 90–127 | 42 / 48 (16-bit) | 8,874,716 | `aab58d9c…` |

The full sha256s are in the CLI output and go into `assets/MANIFEST.yaml` (§9). Three changed after the first
conversion: MTG's two (the attack refit of §4, and the FLAC loops in their sample headers, §6), from `4a187983…`
and `e33ce5b4…`, and the Wurlitzer's (its FLAC loops, now played, §6), from `71ffdb1c…`. The other four are
byte-identical to the first conversion.

- **weresax-alto.**
  - **Program choice.** `Sax.sfz` is the bank's sax instrument; the readme's other two, saxcordion and sxnth,
    build an accordion and a synth on the same samples. `Sax1.sfz` differs only by `off_mode=fast`, which is
    dropped, and it converts to the same sha256.
  - **Mic Blend.** CC120 has no `set_cc`, so it reads 0, the clean condenser end. The condenser groups'
    `xfout_locc120=0` gives gain 1 there, and the dynamic groups' `xfin_hicc120=127` gives 0: 64 regions are
    dropped as silent (their samples were never fetched).
  - **Other drops:** rr2 (128 regions) and the generated wind noise (1).
  - **CC-driven vibrato and humanize.** Every LFO and pitch-EG depth reads 0 at the defaults and is dropped.
    eq1_gain is 0.001 dB.
  - **Envelope at the defaults:** attack 0.001 s, decay 0.504 s, sustain 100 %, release 0.316 s. Pan is −100 +
    200 × 64/127 = +0.79.
- **mtg-tenor, mtg-alto.**
  - **Kept:** the NL presets; round robin 1, whose samples are each key's own (rr2 and rr3 repitch the
    neighbours); the p (≤ 100) and f (≥ 101) layers.
  - **Dropped:**
    - 210 rr2/rr3 regions;
    - the 198-region legato master (`trigger=legato`, and `locc64=1` fails at CC64 0);
    - the noise regions: 2 by the bank rule, the rest by round robin or random selection;
    - the pitch LFO (depth 0 at CC1 0) and the pedal switches.
  - **Kept as written:** `tune=` as fineTune (tenor −40 to +3 cents, alto −40 to +11); release 0.001 + 1 × 0.3
    = 0.301 s; the CC131 attack as velocity modulators (§4).
  - **Loops.** Each FLAC's own loop goes into its sample header, unplayed: the NL presets set
    `loop_mode=no_loop`, so every zone keeps sampleModes 0.
- **jrhodes3d-mono.**
  - **Which file.** The root-level `jRhodes3d-mono-no-xfade.sfz` and `jRhodes3d-mono/_jRhodes3d-mono-no-xfade-flac.sfz`
    have the same 75 regions over the same 65 FLACs, all present. They differ only in the outer key ranges, 10 of
    the 75 regions: the root file covers keys 21–108 (A0–C8), the folder's copy 24–101 (C1–F7). The root file is
    used, since its regions cover the 88 keys.
  - **Tremolo and autopan.** `lfo01_volume` = 0 + 6 × CC1 (0) and `lfo02_pan` = 0 + 100 × CC95 (0) are depth 0 at
    the defaults; they are dropped.
  - **Half-pedal.** `ampeg_dynamic` and `ampeg_releasecc64` sit in `<control>`, where no region reads them.
- **vcsl-upright-knight.** Kept: the sustains, both layers, stereo. Dropped: the 45-region release-trigger group
  and the 8 CC64 pedal-noise regions; neither group's samples were fetched.
- **vsco2-contrabass-pizz.** Random value 0 selects the first `lorand`/`hirand` group, whose 20 regions are the
  rr1 samples. The other 20 are dropped. `ampeg_dynamic` is dropped.
- **gsullivan-wurlitzer.** All 48 regions. The key-scaled hold and decay are fitted (§4). `note_polyphony=1` and
  `off_time` are dropped.
  - **Loops.** The SFZ sets no `loop_mode`, so every zone plays its FLAC's own loop (sampleModes 1) under the
    hold (5 − 5k/127 s) and the decay (25 − 25k/127 s, to sustain 0). The 42 samples last 0.358–5.583 s at their
    own pitch.
  - **Before and after.** The first conversion left them unlooped. Key 74 at velocity 20 (db5pp) then ran out
    of points at 0.339 s, against an SFZ hold of 2.09 s. Now one loop period at 1.5 s measures −62.94 dB, the
    same as the loop's first pass at 0.292 s. Key 71 at velocity 52 (b4mp: points run out at 0.781 s, hold
    2.20 s) measures −53.11 dB at both.

**Pitch at velocity 100, every key** (`tests/test_convert_banks.py`, 2026-10-31).
- **Fidelity.** The render is cross-correlated against the source sample moved by `tune=` and the key offset. It
  agrees within 0.09–0.29 cents for every bank.
- **Against equal temperament** (centroid of the strongest of harmonics 1–4):

| bank | min | median | max | keys beyond 15 cents |
|---|---|---|---|---|
| weresax-alto | −0.6 | −0.1 | +1.1 | none |
| mtg-tenor | −6.2 | −0.2 | +8.3 | none |
| mtg-alto | −1.2 | +2.0 | +9.2 | none |
| jrhodes3d-mono | −0.4 | +3.7 | +4.9 | none |
| vcsl-upright-knight | −36.7 | −4.5 | +20.4 | 23–28, 37–40, 43, 44, 101, 102, 107 |
| vsco2-contrabass-pizz | −15.9 | −8.2 | +18.0 | 29, 31, 45, 46 |
| gsullivan-wurlitzer | −6.7 | −2.8 | +3.5 | none |

The VCSL and VSCO 2 keys beyond 15 cents are where the sources play that far off; the conversion reproduces the
SFZ.
- **VCSL.** The VCSL SFZ's own `tune=` values cause them. Its B-1 sample measures within 4 cents of B, yet the
  region has `tune=-52`; C#0 has −33.
- **VSCO 2.** The SFZ has no `tune=`, and samples such as G0 and A1 measure about 46 cents flat and 40 cents sharp
  (peak estimator).
- **The test** checks those keys for fidelity only (`SOURCE_OUT_OF_TUNE`).
- **Retuning** them from measured pitch would change the bank's tuning. That is Shresth's call after an ear
  check, not the converter's.

## 9. Pinning a converted bank

Claude writes the pins later; the converter does not edit them.

1. **Convert on the laptop** into assets/big, for example
   `python -m lofi_rl.convert all --big assets/big --out-dir assets/big/converted`. assets/big is gitignored, so
   nothing reaches git, the jRhodes3d file included.
2. **Record the output** in `assets/MANIFEST.yaml`, in the SFZ source's fetch-mode entry, the way archives list
   their SF2 under `extracts`:

   ```yaml
   extracts:
     - {path: "assets/big/converted/<bank-id>.sf2", bytes: N, sha256: <the CLI's sha256>,
        from: "<folder>/<sfz>", by: "lofi_rl.convert at <commit>"}
   ```

   The entry keeps the source's licence; the SF2 is an adaptation of it (ATTRIBUTION.md).
3. **Point the palette's `sources` entry at it:** `path` relative to `assets_root`, its `sha256`, and
   `converted_from` naming the SFZ. `load_palette(verify=True)` then checks the hash and that preset (0, 0)
   exists, through `palette.sf2_presets`.
4. **On the cluster,** either rsync the converted files with the banks into `$REMOTE_KEEP/assets/`, or convert
   there in a job (4 s of CPU, about 1.5 GB of memory). Either way, compare the sha256 with the laptop's before
   the first render. A mismatch means a machine difference (§7), not a different bank.
5. **Re-pin after a converter change.** Any change to the converter that changes a bank's bytes changes its
   sha256: re-pin it, and record it in the CHANGELOG with the reason.
