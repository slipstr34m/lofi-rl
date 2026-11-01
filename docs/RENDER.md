# RENDER.md: CompiledLoop to a seamless loop at −14 LUFS

The renderer turns a LoopScript, a brief and a palette into one cycle of stereo float32 audio at 44.1 kHz,
exactly N samples long, that repeats without a seam. It also writes the same render without the bed (for
the scorers), a manifest, and optionally a dithered 16-bit FLAC listening copy.

- **Spec:** `docs/design/instrument.md` §0 (amendment 3), §2 (gate G7), §3 (Drum sampler, Seamless), §4
  (binding), §6 (palette) and §7 (tests). Section references below are to instrument.md unless noted.
- **Input:** the compiler's `CompiledLoop` (`docs/COMPILER.md` §4).
- **Code:** `src/lofi_rl/render/`: `pipeline.py` (stages and CLI), `sampler.py` (drums and one-shots),
  `mix.py` (bus chains), `bed.py`, `formats.py`, `machine.py` (platform facts), plus the existing `fluid.py`
  (now isolated from FluidSynth's config files and checked for faults, §3.1), `loudness.py` and `wavio.py`.
- **Sound claims:** none. Every audible value here and in `palettes/dev_gu.yaml` is a starting point for an
  ear check or a shootout (instrument.md header).

## 1. Entry points

```python
from lofi_rl.brief import Brief
from lofi_rl.palette import load_palette
from lofi_rl.render.pipeline import render

palette = load_palette("palettes/dev_gu.yaml")             # verify=True hashes every source file
r = render(text, Brief(energy="mid", mood="wistful", bpm=82, tonic=2), palette)
r.audio, r.music          # (2, N) float32: render.wav and music.wav (the same without the bed)
r.manifest                # JSON-ready dict (§3.7); r.manifest["audio_sha256"] is the render's identity
r.pre_limiter, r.compiled # the kept cycle after the master chain, and the CompiledLoop
```

- `render(loop, brief, palette, *, cycles=2, cache=None, root=ROOT, finish_reason=None, workers=4)`.
  `loop` is LoopScript text, parsed with `palette.vocab()` and gated (G1–G6), or a parsed `Loop`.
- **Errors.** `RenderError(code, detail)` for a gate: G1–G6 from the parser, G7 from the render.
  `CompileError` for a loop the palette cannot play. Infrastructure faults are never gates, because they are
  not the policy's:
  - `fluid.FluidSynthError` when FluidSynth does not render what it was asked to (§3.1): a non-zero exit, an
    error in its log, a preset it substituted, an output shorter than its MIDI or at another rate, or a stem
    with notes that comes back silent in a kept cycle, or when `/etc/fluidsynth.conf` holds commands;
  - a missing file propagates unchanged.
- **Stages for tests and tools:** `music_cycles(compiled, palette, cycles)` returns stages 2–4 over all
  cycles; `finish(music_cycle, palette)` runs stages 5–7.
- **Where the palette's files are** (`Palette.source_path`, `Palette.assets_dir`). A mode-commit source is
  repo-relative. A mode-fetch source lies under the palette's `assets_root`, `$LOFI_ASSETS` in every palette:
  - unset or empty, `$LOFI_ASSETS` is `<repo>/assets/big`, where the laptop fetches the shootout assets
    (docs/SHOOTOUT_ASSETS.md, Route), so a fetch-mode palette loads on the laptop with nothing exported;
  - a relative value is relative to the repo root (`root=`), never to the current directory;
  - any other unset or empty variable in `assets_root` is a `PaletteError`, which the CLI reports as exit 2.
  On the cluster, `<repo>/assets/big` is inside the code mirror, which never holds assets, so a fetch-mode
  palette would fail `verify` there ("missing", exit 2), so `scripts/job_env.sh` exports
  `LOFI_ASSETS=${LOFI_ASSETS:-$REMOTE_KEEP/assets}`, where `make push-assets` copies `assets/big/`.
- **`workers`** sets how many FluidSynth processes and pedalboard threads run at once. The output never
  depends on it (`tests/test_determinism.py`, `tests/test_mix.py`).

**CLI**
```
python -m lofi_rl.render.pipeline file.ls --mood wistful --energy mid --bpm 82 --tonic 2 \
    --palette palettes/dev_gu.yaml --out x.wav [--flac] [--cycles 2] [--cache DIR] [--workers 4] [--no-verify]
```
- **Writes** `x.wav` (render), `x.music.wav`, `x.manifest.json` and, with `--flac`, `x.flac`.
- **Prints** `N=… LUFS=… true_peak=… dBTP pcm_sha256=…`, then the manifest JSON. The list of files written
  goes to stderr.
- **Exit codes:** 0 on success; 2 for a bad argument, checked before anything renders (`--cycles` below 2,
  `--bpm` outside 60–95, `--tonic` outside 0–11, a loop or palette file that is missing or unreadable, a
  palette that does not load or verify); 3 for a gate or compile failure (with the code and detail on stderr).
  An infrastructure fault (`FluidSynthError`, a file error) ends in a traceback, exit 1.

## 2. The chain

| # | Stage | Code | What it does | Spec |
|---|---|---|---|---|
| 1 | Parse, compile | `dsl.parse`, `compile.compile_loop` | Gates G1–G6, then the CompiledLoop | §1–3 |
| 2 | Pitched stems | `pipeline.fluid_stems` | keys, bass, sax: one fresh FluidSynth process each, on a MIDI file of 1 + C cycles. The first cycle is a pre-roll and is dropped. | §3 Seamless, §4 |
| 3 | Drums | `sampler.kit_shots`, `sampler.render_drums` | One-shots built or loaded from the cache; every hit summed into one periodic cycle, then tiled C times | §3 Drum sampler |
| 4 | Mix | `mix.mix_buses` | Bus chains, faders and the master chain over all C cycles; the last cycle is kept | §0 amendment 3, §4 |
| 5 | Bed | `bed.bed_loop`, `bed.place` | The bed's seamless cycle at LUFS_music − Δ, added to the music | §4 Rain bed |
| 6 | Loudness | `loudness.normalize_loop` | Gain into the true-peak limiter on the 3× tile, −14 LUFS ± 0.1 in at most 4 iterations | §4 Loudness |
| 7 | Gate G7, manifest | `pipeline.finish`, `pipeline.render` | Length, finiteness, loudness, true peak on the repeating loop; the manifest | §2, §4 Outputs |

C = `cycles` = 2 in production ("two cycles, keep the second"). The seam test uses C = 3 (§4).

## 3. Numbers, with their sources

### 3.1 FluidSynth stems (§4)
- **Version.** FluidSynth 2.6.1. `render/fluid.py` raises on any other version. `fluidsynth --version` starts a
  whole FluidSynth (about 50 ms), so a pass is remembered per binary file (path, size, mtime, inode) for the
  life of the process; a replaced binary is checked again.
- **Flags** (`render/fluid.py`): `-q -n -i -R 0 -C 0 -g 0.2 -r 44100 -o synth.cpu-cores=1 -T wav -O float -F out.wav <sf2> stem.mid`.
  Interpolation stays at the default (4th order). §4 also lists `-o synth.limiter.active=0`, but FluidSynth 2.6.1
  has no such setting: it prints "Setting parameter 'synth.limiter.active' not found" and exits 255
  (`tests/test_fluid_cli.py`). So the flags leave it out.
- **No config files.** FluidSynth 2.6.1 sources `/etc/fluidsynth.conf` and `$HOME/.fluidsynth` on every start,
  whatever the flags: `-f` adds a file, it does not replace them (its `-v` log, 2026-10-31). In a one-note
  render, a user config of `gain 0.6` and `interp 0` changed the PCM, with or without `-f`, and
  `set synth.sample-rate 48000` made the output 48 kHz (measured 2026-10-31). So every render runs with `HOME`
  set to a new empty directory (the rest of the environment is kept, for the cluster's wrapper), and no render
  starts while `/etc/fluidsynth.conf` holds commands (blank and `#` lines change nothing, and only root could
  remove the file). Unsetting `HOME` would not help: FluidSynth then finds the account's home directory itself.
  `tests/test_fluid_cli.py` pins the two files FluidSynth tries, so a build that tries more fails there.
- **Faults** (`fluid.FluidSynthError`, never a gate). FluidSynth exits 0 on several faults, so its log is read:
  - an `error` or `panic` line (for example a SoundFont it could not load), or a non-zero exit;
  - "Instrument not found on channel …, substituted …": a (bank, program) the SoundFont lacks, which FluidSynth
    replaces with another preset (a drum kit with kit 0, a melodic bank with bank 0). `load_palette(verify=True)`
    refuses such a palette first, from the SoundFont's preset headers (`palette.sf2_presets`);
  - a missing preset with no stand-in leaves its channel silent, which the checks below catch;
  - a render that has not finished after 120 s (`fluid.TIMEOUT_S`, under the scorer's 150 s batch deadline):
    `subprocess.run` kills it.
- **One process per stem.** A stem with no notes (a comp or bass grid of rests) is silence and runs no process.
- **MIDI.** `compile.midi.write_stem_midi(compiled, name, path, cycles=1 + C)`, with ticks from `midi.fluid_tick`
  (docs/COMPILER.md §4.3). The output must hold at least (1 + C)·N samples at 44.1 kHz; it is cut to that, and
  samples [N, (1 + C)·N) are kept. Every stem MIDI ends 0.1 s after its last cycle, so a shorter output can only
  be a fault (a full disk, say), and a stem with notes must sound in every kept cycle. Either fault is a
  `FluidSynthError`; neither is padded with zeros or passed on.
- **Pan.** The GU keys, bass and tenor renders of `example.ls` have identical left and right channels
  (measured 2026-10-31), so the sax is centred as §4 asks with no extra step.

### 3.2 Drum sampler and one-shots (§3 Drum sampler; docs/COMPILER.md §3.9, §4.4)
- **A hit** plays `palette.kits[kit].lanes[lane].layers[layer][rr]` from sample `on`. Its gain is
  `hit.gain × 10^(lane.db/20)`, and it is panned by the lane's `pan`. The compiler has already chosen the layer
  (nearest-layer fallback) and `rr`.
- **Pan law:** gL = √((1 − pan)/2), gR = √((1 + pan)/2). It is constant power and exact at −1, 0 and +1.
- **Mono one-shots.** A stereo one-shot is downmixed to (L + R)/2 before the pan, so the palette's lane pan alone
  places it. The GU Jazz kit itself pans the rim, hats and shaker (its right channel is up to about 3× the left
  in RMS for the shaker, measured). With every `dev_gu` pan at 0, the dev drums are centred.
- **Choke.** A hit with `choke_at` fades out linearly over 8 ms = 353 samples (352.8, rounded half up), starting
  at `choke_at`, which may lie in the next cycle (§3: "an 8 ms fade").
- **Seam.** Hits are summed in float64 in (on, lane, step) order into a buffer as long as the longest reach, and
  everything past N folds back onto [0, N), as many times as needed.
- **SoundFont one-shots** (`{src, bank, prog, note, vel}`, the `dev_gu` form):
  - The MIDI file has one note at tick 0, a note-off 4 s later and an end of track 0.1 s after that.
    Bank 128 plays on channel 9, where FluidSynth ignores bank select and takes the program from bank 128;
    other banks play on channel 0 with CC0 set to the bank.
  - The note sounds from sample 64 (block 1, docs/COMPILER.md §4.3). The one-shot is the render from sample 64
    to its last non-zero sample; FluidSynth renders on past the end of track until its voices are silent.
    The builder raises if anything sounds before sample 64, or if the note renders silent, and FluidSynth's
    faults raise as for the stems (§3.1): a palette naming GU drum kit 33, which GU lacks, no longer renders
    kit 0 ("Standard") under the Jazz kit's name.
  - The 4 s hold: the GU open hat renders identically with a 3 s or a 10 s hold (measured). One-shot lengths
    at velocity 112 (the cache's records): kick 28,012 samples, snare 16,894, rim 7,500, hat 9,060, shaker
    7,091, open hat 925,952 (21.0 s; its tail falls below −80 dB re its peak about 14 s after the onset).
  - `dev_gu`: kick 36, snare 38, rim 37, hat 42, ohat 46, shaker 82, at velocities 48/80/112 for soft/med/hard,
    from GU bank 128 preset 32 "Jazz" (the SF2's preset header). Program 32 renders differently from program 0
    (`tests/test_sampler.py`).
- **File one-shots** (`{src, file}`): a WAV or FLAC member of a directory, zip or tar source, at 44.1 kHz.

### 3.3 Bus chains (§4 table; `palettes/dev_gu.yaml` `mix`)
- The chains are the palette's, built as pedalboard 0.9.26 plugins in order. Each bus is followed by its fader
  `bus_gain_db`, then the buses are summed in float64 in the order sax, keys, bass, drums, and the master chain
  runs on the sum.
- **`sax_warm: true`** splices the `sax_warm` chain (HighShelfFilter 4 kHz −5 dB, LowpassFilter 9 kHz, Reverb
  room .35 damp .7 wet .12) in place of the sax chain's Reverb.
- **Block size.** pedalboard always gets 8,192-sample blocks (`mix.BUFFER`, pedalboard's default) and a reset
  state. Every `dev_gu` chain and the master give identical bits at 64, 1,000, 4,160 and 8,192 samples per block
  (`tests/test_mix.py`), so cycle boundaries need not fall on block boundaries.
- **Faders.** `dev_gu` has every `bus_gain_db` at 0 (uncalibrated). §4 calibrates them on the hand loops so that
  the median bus loudness is sax 0, drums −1.5, bass −3 and keys −6 LU. The manifest records each bus's
  loudness before its fader, and `mix.calibrated_gains(rows, targets)` turns a list of them into gains, with the
  sax fixed at 0 dB. On the five distinct gate-passing fixtures (`example_leniency.ls` repeats `example.ls`),
  rendered with `dev_gu` at 82 BPM in D dorian, the medians relative to the sax are bass −4.95, drums −15.42 and
  keys +1.74 LU. The proposed gains are keys −7.74, bass +1.95 and drums +13.92 dB. They are **not applied**:
  the spec's calibration set is the hand loops.

### 3.4 Bed (§4 Rain bed; §6 `bed`)
- **Source:** any mono or stereo file at 44.1 kHz named by `bed.src`, a palette source. A mono source plays in
  both channels. It is read from `start_s` through HighpassFilter 200 Hz (§4). The filter warms up on up to
  0.5 s of the file before `start_s`.
- **Seam.** An equal-power crossfade over X = `xfade_s` = 1 s = 44,100 samples. o[k] = e[X + k] for k < N − X,
  and o[N − X + j] = cos θⱼ·e[N + j] + sin θⱼ·e[j], with θⱼ = (π/2)·j/X. The fade starts on e[N], right after
  e[N − 1], and it ends where the loop wraps to o[0] = e[X], the sample after e[X − 1]. The source needs
  `start_s` + N + X samples. At 60 BPM, N is 32 s, so the file needs `start_s` + 33 s.
- **Level.** LUFS_bed = LUFS_music − Δ, with Δ = `delta_lu` (24 by default) and the music measured after the
  master chain and before the limiter. On `example.ls` with a synthetic noise bed, the bed (render minus
  music.wav) sat 17.998, 23.997 and 29.996 LU below music.wav for Δ = 18, 24 and 30 (measured 2026-10-31).
- **`bed: null`** (as in `dev_gu`) renders without a bed, and music.wav equals render.wav.

### 3.5 Loudness and gate G7 (§2, §4 Loudness)
- **Normalization.** `loudness.normalize_loop` with the palette's `loudness`: target −14 LUFS, tolerance 0.1 LU,
  at most 4 iterations. Each iteration tiles the loop 3 times, runs BrickwallLimiter (ceiling −1.5 dB, release
  100 ms, lookahead 5 ms, true peak; `LIMIT_DB` sits under the −1 dBTP ceiling) and keeps [N + λ, 2N + λ),
  where λ is the limiter's latency. Not converging is gate G7. Every fixture converges in 1 iteration (§7); a
  synthetic loop in `tests/test_render.py`, a tone with strong 1 kHz bursts, takes 3, and fails G7 with `iters: 1`
  (its first pass lands at −14.78 LUFS). The palette's `ceiling_dbtp` must be −1, the
  value `loudness.py` is built for; any other value raises.
- **music.wav** gets the same gain and the same limiter as the render, without the bed.
- **True peak** is measured on the loop as it repeats (`pipeline.loop_true_peak_dbtp`): 4× `resample_poly` over
  the cycle with 64 samples of the other end on each side, keeping only the cycle's own samples. Edge padding
  therefore neither hides the inter-sample peak at the seam nor adds ringing.
- **Gate G7** (§2) fails on:
  - NaN or infinity;
  - music below −50 LUFS (measured before the gain);
  - a length other than N;
  - final loudness more than 0.5 LU from the target;
  - true peak above −0.9 dBTP.
  In practice the 0.1 LU convergence check fires first.

### 3.6 Outputs (§4 Outputs)
- **render.wav and music.wav:** IEEE float32 WAV, stereo, 44.1 kHz, exactly N samples, with no PEAK chunk
  (`wavio.write_float`), so the same audio always gives the same bytes.
- **Identity:** `audio_sha256` = `wavio.pcm_sha256`, the sha256 of the interleaved float32 little-endian samples.
- **FLAC listening copy** (`formats.write_flac`): 16-bit, TPDF dither d = u(2m) − u(2m+1), where u is
  `prng.stream` (SplitMix64) keyed by the first 8 bytes of the render's pcm_sha256 and m is the interleaved
  sample index. The sample is rint(32767·x + d), clipped to int16. Two writes give identical bytes, also 1.5 s
  apart (measured), and the copy decodes to exactly the dithered samples (`tests/test_formats.py`).

### 3.7 The manifest
- **Contents:**
  - `loop`: canonical text, its sha256, the compiled digest, parser plus compiler repairs, and P_fix;
  - `brief` and `palette` (id, sha256, frozen);
  - `platform`: system, machine, CPU model, Python, FluidSynth, libsndfile and the audio packages' versions;
  - `n`, `sr`, `bpm_eff`, `seconds`;
  - `chain`: cycles, pre-roll, MIDI cycles, pedalboard block size, limiter tile;
  - `instruments`;
  - `stems`: the pcm_sha256 of each bus input's kept cycle;
  - `oneshots`: the pcm_sha256 of every one-shot;
  - `buses`: each bus's LUFS before its fader, and its gain;
  - `music_lufs` (before the limiter), `bed` (Δ, gain, levels, or null);
  - `loudness`: LUFS, gain, iterations, true peak, and the same two for music.wav;
  - `gates`, `audio_sha256` and `music_sha256`.
- **No wall clock, paths or cache state.** The manifest is as deterministic as the audio: equal across
  processes and render orders (`tests/test_determinism.py`).

## 4. Seamless: why the MIDI holds 3 cycles
- **Bus effects.** §0 amendment 3 puts the bus effects inside "two cycles, keep the second". The reverbs and
  compressors run over both cycles, and the second is kept. For `example.ls` at 82 BPM, the master output's
  cycle 1 differs from cycle 2 by up to 0.0217, against a cycle-2 peak of 0.123: the warm-up. Cycles 2 and 3 of
  a 3-cycle render are bit-identical.
- **FluidSynth's history.** FluidSynth's output for a note depends slightly on what its voice played before
  (docs/COMPILER.md §4.3 caveat). In a 79 BPM loop from the compiler's fuzz run (`PREROLL_CASE` in
  `tests/test_seam.py`), the sax stem's MIDI cycles 2 and 3 differ by 4.7e-5, and the whole chain's cycles 2 and
  3 by 8.64e-5. Both are 0 with one pre-roll cycle, and 0 with two.
- **Pre-roll.** So the renderer adds one FluidSynth pre-roll cycle (`pipeline.PREROLL = 1`). The MIDI holds
  3 cycles; the bus chains run on MIDI cycles 2 and 3, and the kept cycle comes from MIDI cycle 3. Cost: about
  0.04 s of FluidSynth time per stem (measured on the `example.ls` stems: 2 cycles 0.22–0.24 s, 3 cycles
  0.25–0.28 s).
- **Seam checks.** In 24 random valid loops (5 palette variants, random briefs), the whole chain's cycles 2 and
  3 were bit-identical both with and without the pre-roll (fuzz run, 2026-10-31). With the pre-roll, cycles 2 and
  3 are bit-identical on the laptop for 5 gate-passing fixtures × 4 tempos (61, 71, 87, 95 BPM;
  `test_seam.py -m cluster`).
- **Linux.** On the cluster (Linux x86_64; epyc-7513, epyc-7542 and xeon-4116 alike) every FluidSynth stem is
  still bit-identical from cycle to cycle, but the pedalboard bus chains repeat only to float32 rounding:
  - fed an exactly periodic input, their cycles 2 and 3 differ by 5.7e-7 to 1.4e-6 over 21 fixture × tempo cases
    (`lofi-seam-diag` 13630343, `runs/seam-diag-13630343/summary/seam.json`);
  - the differences are scattered through the cycle, are at most 9.1e-7 in its first and last second, and do not
    grow over a 4th cycle.
  So `test_seam` holds the whole chain to `SEAM_TOL` = 1e-5 (−100 dBFS, a third of the 16-bit listening copy's
  LSB) and FluidSynth stems and the limiter tile to 1e-6. Renders stay deterministic: the same inputs give the
  same bits on all three CPU models (`lofi-render-check` 13630298–13630300).
- **Causality.** Every stage is causal, so the production render's kept cycle (C = 2) is bit-identical to
  cycle 2 of the 3-cycle render that `test_seam` checks.
- **Limiter.** The 3× tile keeps the steady-state cycle: on a 2.9 s synthetic loop with peaks at the seam, the
  3× tile's middle matches a 5× tile's to within 1e-6 (`tests/test_seam.py`).

## 5. Determinism (CLAUDE.md §6)
- **Equal results.** The same LoopScript, brief and palette give bit-identical PCM and an equal manifest:
  - within one process;
  - in A-B-A render order;
  - with 1 or 4 workers;
  - in a fresh process with another `PYTHONHASHSEED`, through the CLI (`tests/test_determinism.py`).
- **What the code avoids:**
  - no wall clock, and no randomness except keyed SplitMix64;
  - no dict or set order: buses, lanes and layers iterate fixed tuples, and hits are sorted;
  - no dependence on thread count: each FluidSynth process is single-core, and every parallel task is
    independent and collected in a fixed order;
  - no dependence on the account: FluidSynth runs with an empty `HOME`, and refuses to while
    `/etc/fluidsynth.conf` holds commands (§3.1);
  - pedalboard always gets the same block size and a reset state.
- **Cross-CPU hygiene.**
  - No numpy transcendental runs on an audio array: numpy's SIMD sin/cos/exp can differ in the last bits
    between AVX2 and AVX-512 machines.
  - The pan law uses `math.sqrt`, the choke a division, and the bed crossfade one `math.cos`/`math.sin` call per
    weight. The dither is integer arithmetic.
  - The remaining float paths are pyloudnorm's LUFS (numpy `log10` and `sum` per 400 ms block), libm `pow` for
    dB gains, pedalboard and FluidSynth. A last-bit change in a LUFS value moves the linear gain by under 1e-15
    relative, and `normalize_loop` rounds that gain to float32 (steps of about 6e-8) before applying it, so it
    almost never changes a sample. That is still a risk, and the cluster's determinism job measures it.
- **Golden hashes.** `tests/golden/render_hashes.json` holds pcm_sha256 values keyed by `machine.golden_key()`:
  platform, FluidSynth and numpy versions. This laptop's key is `Darwin-arm64/fluidsynth-2.6.1/numpy-1.26.4`,
  with all six gate-passing fixtures at the `example.ls` brief, `shaker_edits.ls` at G ionian, 95 BPM, and
  `example.ls` at A aeolian, 87 BPM (a mood no other case has, at a tempo whose steps fall on half samples).
  The key also holds `oneshots|dev_gu|jazz`, the pcm_sha256 of each of the kit's 18 one-shots (§6). Other keys
  skip. The default suite checks `example.ls` and the one-shots; `-m cluster` checks every case.

## 6. The one-shot cache
- **Location:** `$LOFI_CACHE`, else `~/.cache/lofi-rl`, or `--cache` / `cache=`. Entries live in `oneshots/`:
  `<key>.wav` (float32, as rendered, stereo) and `<key>.json` (settings, pcm_sha256, length).
- **Key:** sha256 of the canonical JSON of the settings:
  - the builder version and the SoundFont's sha256 from the palette. The builder is `fluid-note/2` since
    FluidSynth runs in an empty home, so no entry built while a `~/.fluidsynth` may have shaped it is reused;
  - bank, program, note, velocity and channel;
  - onset, hold, tail, sample rate and FluidSynth gain;
  - the FluidSynth runtime version;
  - the machine: system, architecture and CPU model, so two CPU models never share renders.
- **Reuse:** an entry is used only if its samples still hash to the recorded pcm_sha256; otherwise it is
  rebuilt and replaced. Writes go through a temporary file and a rename, so parallel workers never see a
  partial entry.
- **Cost:** the GU Jazz kit is 18 one-shots, 36 files and about 23 MB, most of it the three open-hat layers.
  Building them into an empty cache took 1.65 s on the laptop with 4 workers (1.56 s with 8), once per cache.
- **A builder change shows only in a cold cache.** Every render reads its one-shots through the cache, and an entry
  is reused while it matches its own recorded hash, so a change to the builder that does not bump `BUILDER` passes
  every render test on a machine with a warm cache. So `tests/test_determinism.py` builds the kit into an empty
  cache in the default suite (about 1 s) and compares each one-shot with this machine's golden entry (§5) and
  with the warm cache's, which the render manifests record.
- **On the cluster:** `scripts/job_env.sh` sets `LOFI_CACHE=$REMOTE_SCR/cache/lofi-rl`, so the small home
  directory is never used. Scratch purges only cost a rebuild.

## 7. Cost (laptop, Apple M4, 2026-10-31)
- **One render of `example.ls`** (N = 1,032,576, 23.4 s), in process with a warm cache: 1.26 s with 4 workers,
  1.72 s with 1. With the FluidSynth version check remembered per binary (§3.1), the best of 3 renders was
  1.14 s with 4 workers and 1.54 s with 1 (re-measured 2026-10-31, load average about 1.7).
  - FluidSynth stems: 0.5 s with 3 in parallel; 0.79 s one after another (re-measured: 0.37 s and 0.64 s).
  - Bus chains plus master: 0.16 s.
  - Normalization (1 iteration) and checks: about 0.5 s.
- **The CLI** takes about 2.2 s wall per fixture, interpreter start included.
- All six fixtures converged in 1 iteration (§8).

## 8. Fixture renders (CLI, `dev_gu`, wistful, mid, 82 BPM, tonic 2; this laptop, 2026-10-31, after voicing rules V2.1)

| fixture | N | LUFS | true peak (dBTP) | pcm_sha256 |
|---|---|---|---|---|
| example.ls | 1,032,576 | −14.006 | −1.496 | `08e696500fbf4f1083666f2b4da277a61dbe150f12eb199542a991b85c883e90` |
| example_leniency.ls | 1,032,576 | −14.006 | −1.496 | `08e696500fbf4f1083666f2b4da277a61dbe150f12eb199542a991b85c883e90` |
| leniency_lower_x.ls | 1,032,576 | −14.007 | −1.496 | `013093004a1dd7d4c1fda80925dacbcd10417c004ff4ebaea40c8f3f89858b87` |
| one_bar_base.ls | 1,032,576 | −14.028 | −1.494 | `31fdfbaa4f5fd6dfd2e6af69e26972dc14d5e0c0ea43c98d6a0e309f7705cb6f` |
| shaker_edits.ls | 1,032,576 | −14.038 | −1.494 | `93a0baa3067fa0b1cbb738f58c2f2f4512dd8847d9c9d15f5a6853747f845a19` |
| span1_dotted.ls | 1,032,576 | −14.008 | −1.495 | `71fec47349295cad00a5c98874673d51780433396c33d8dc0fb20c6850abfc54` |

`example_leniency.ls` parses to the same AST as `example.ls`, so their renders are identical. These are laptop
hashes; the cluster's are canonical (docs/kit/LESSONS.md 38). None of these clips has had an ear check.
- **Voicing rules V2** (docs/COMPILER.md §9) changed every row but `one_bar_base.ls`, whose shells did not change.
  Each change is in the keys stem alone: the bass, sax and drum stem hashes are the same as before. Before V2,
  `example.ls` rendered to `6061a4a3…`, `leniency_lower_x.ls` to `ec884f97…`, `shaker_edits.ls` to `bf8b4f1e…` and
  `span1_dotted.ls` to `703079f8…`.
- **Voicing rules V2.1** (docs/COMPILER.md §9) changed only `span1_dotted.ls`, the one quartal fixture, through its
  keys stem alone: under V2 it rendered to `86d6b1fd…` at −14.028 LUFS and −1.495 dBTP.

## 9. Choices the spec left open
Each is pinned by a test and can be changed.

- **§3 Seamless.** A FluidSynth pre-roll cycle is added (§4 above).
- **§3 Drum sampler.**
  - The pan law is the square-root constant-power law.
  - One-shots are mono, downmixed (L + R)/2.
  - The choke fade is linear.
  - Tails fold back as often as they need to.
- **§4 one-shots.**
  - A one-shot runs from the onset block to the last non-zero sample.
  - The note-off comes at 4 s.
  - Bank 128 plays on channel 9.
  - The cache is keyed by every setting and the machine, and verified on reuse.
  - File one-shots come from directory, zip or tar sources.
- **§4 chains.**
  - The block size is 8,192 for every call.
  - The fader comes after the bus chain.
  - The buses sum in float64 in the order sax, keys, bass, drums.
  - `sax_warm` splices its chain in place of the sax Reverb.
- **§4 bed.**
  - θⱼ = (π/2)·j/X, so the fade starts exactly on e[N].
  - The high-pass warms up on up to 0.5 s of the file before `start_s`.
  - A mono bed plays in both channels.
  - A file at another sample rate, or too short, is an error, never resampled or looped.
- **§4 loudness.**
  - music.wav takes the render's gain and its own limiter pass.
  - True peak is measured with wrap-around context.
  - The G7 true-peak limit is −0.9 dBTP (§2), while the tests hold renders to −1.0.
- **§4 outputs.**
  - The FLAC uses full scale 32767 with rint rounding, and the dither key is the first 8 bytes of
    pcm_sha256, big-endian.
  - The CLI writes `x.music.wav` and `x.manifest.json` next to `x.wav`.

## 10. Spec issues found
- **§4 FluidSynth flags.** `-o synth.limiter.active=0` is not a FluidSynth 2.6.1 setting; the command exits 255.
- **§3 Seamless.** "The MIDI holds 2 cycles" is not always enough for FluidSynth (§4 above).
- **§2 vs §4 on loudness.** G7 fails at 0.5 LU, but §4 step 4 sends any loop that is not within 0.1 LU after
  4 iterations to G7. The renderer follows step 4, so the 0.5 LU check never decides anything.
- **§4 Bus balance.** The hand loops do not exist yet, so `dev_gu`'s faders are uncalibrated (§3.3 has the
  measured offsets and the proposed gains).
- **§6 palette.**
  - `bed` has no field for the 200 Hz high-pass, so it is a constant in `bed.py`.
  - `palette.verify_sources` hashes files only, so a directory source for file one-shots cannot pass
    `verify=True`. A zip or tar source can. (`verify=True` also checks every SoundFont preset the palette
    names against the file's preset headers, §3.1.)
  - `.rsyncignore` drops `*.wav` everywhere, so loose one-shot WAVs would never reach the cluster mirror
    (CHANGELOG, 2026-10-31). An archive source avoids that.
- **§4 Assets: two places for fetched assets.** §4 puts `$LOFI_ASSETS` at
  `/project2/anakano_429/shresthr/lofi-rl-assets`; CLAUDE.md §1, docs/SHOOTOUT_ASSETS.md (Route) and
  `scripts/cluster_setup.sh init` put fetched assets in `$REMOTE_KEEP/assets`. The renderer follows CLAUDE.md:
  `scripts/job_env.sh` exports `LOFI_ASSETS=$REMOTE_KEEP/assets`; §4's path is not defined anywhere else.

## 11. What the cluster still has to do
- **Canonical hashes.** The determinism job this item asked for is `jobs/render_check.sbatch` (next item). Run it
  once per CPU model (`OVR='--constraint=<model>'`). Each run records `{golden_key: {case: hashes}}` into
  `$REMOTE_SCR/runs/render-check-<job>/summary/render_hashes.json` with `LOFI_RECORD_GOLDEN`, in its own
  `LOFI_CACHE`. After `make pull`, the hashes are merged into `tests/golden/render_hashes.json`. The golden key
  carries no CPU model, which holds while every model gives the same hashes.
- **Done 2026-10-31: canonical hashes and the seam sweep.** `jobs/render_check.sbatch` (`lofi-render-check`) ran
  the node suite, recorded the golden hashes and rendered every fixture on epyc-7513, epyc-7542 and xeon-4116
  (13630298–13630300). All three gave the same hashes, so one Linux key serves the partition. Its seam failures
  were the bus-chain rounding of §4 (Linux), measured by `lofi-seam-diag` 13630343. The Linux key goes into
  `tests/golden/render_hashes.json` after a rerun records every case, the two below included.
- **Golden entries added 2026-10-31.** The determinism job should also record `example.ls|melancholy|low|87|9`
  and `oneshots|dev_gu|jazz` for each cluster key (the one-shot test runs in the default suite and records under
  `LOFI_RECORD_GOLDEN` like the renders).
- **Re-record after voicing rules V2.1.** The Linux key came from `lofi-render-check` 13632135, under V2. V2.1
  (docs/COMPILER.md §9) changed `span1_dotted.ls|wistful|mid|82|2` on the laptop key, so that Linux entry is stale
  until the next render check records it. The other 7 render cases compile to the same events as under V2 (equal
  `CompiledLoop` digests), so their entries stand.
- **Fetch-mode sources** (every shootout palette; §1, Where the palette's files are). Two pieces are outside the
  renderer and not done:
  - done: `scripts/job_env.sh` exports `LOFI_ASSETS=${LOFI_ASSETS:-$REMOTE_KEEP/assets}`;
  - done: `make push-assets` copies `assets/big/` to `$REMOTE_KEEP/assets/` without `.rsyncignore` (which drops
    `assets/big/`, `*.wav` and `*.flac`) and without `--delete`, owner-only. The 14 shootout sources went up on
    2026-10-31, checked by rsync checksums (1,960 files).
- **Smoke.** `src/lofi_rl/smoke.py` still renders through the Phase-0 stub chain (`compile/stub.py`). Moving
  the smoke's `loop_render` to `render.pipeline.render` is a separate change; smoke.py was out of scope here.
- **Spectral fingerprints.** `tests/golden/fingerprints.json` (§7: laptop vs cluster within a tolerance) is
  not written yet.
- **Calibration.** Calibrate `bus_gain_db` on the hand loops with `mix.calibrated_gains` once they exist.

## 12. Tests

| §7 row | file | what it checks |
|---|---|---|
| `test_sampler` | `tests/test_sampler.py` | layers and fallback, round-robin from compiled hits, the 8 ms choke (also into the next cycle), folding at the seam, the pan law, order independence, file one-shots, the GU kit build, cache reuse and rebuild, the drum-channel program, a missing drum kit as an error, a user config that never reaches the cache, each lane's own pan and dB through `kit_shots` and `render_drums` |
| `test_render` | `tests/test_render.py` | length N, −14 ± 0.1 LUFS, true peak ≤ −1 dBTP on the repeating loop, the manifest against the audio, the bed at −Δ ± 0.5 LU, gates G1–G6 and G7. On synthetic loops (no FluidSynth): a loop that needs 2–4 loudness iterations (and fails G7 with `iters: 1`), the bed at Δ = 18, 24 and 30 with the reported level, the G7 music floor at −55 and −45 LUFS, the G7 true-peak limit at −0.69 and −0.94 dBTP, an overshoot at the seam. With the audio stages stubbed: parser plus compiler repairs and P_fix in the manifest, and a bare numeral parsed in the brief's mode |
| `test_seam` | `tests/test_seam.py` | 3-cycle render, cycle 2 vs cycle 3 < 1e-5 for the whole chain (§4, Linux); the render keeps exactly cycle 2; the pre-roll case; the 3× limiter tile; `-m cluster`: 5 fixtures × 4 tempos |
| `test_determinism` | `tests/test_determinism.py` | the same process in A-B-A order; a fresh CLI process with 1 worker against the in-process render with 4 (WAV, FLAC and manifest), run alongside the A-B-A renders to save wall time; golden hashes; the kit's one-shots built into an empty cache against their golden hashes and the warm cache's |
| `test_fluid_cli` | `tests/test_fluid_cli.py` | version 2.6.1, the exact flags, the version check and its memo, the rejected limiter flag, one process per non-empty stem with the pre-roll; an empty home for every render, a user config that cannot reach the PCM, the two config files FluidSynth tries, a system config that stops the render, substituted presets, errors in the log, a render past its 120 s, short, silent and wrong-rate output as `FluidSynthError` |
| (CLI) | `tests/test_render.py` | exit 2 on an out-of-range `--bpm` or `--tonic`, a missing loop or palette file, or a palette that does not load, before anything renders |
| (outputs) | `tests/test_formats.py` | SplitMix64 vectorized = `prng.stream`, the TPDF distribution, dither keyed by the hash, FLAC decode and bytes, no PEAK chunk |
| (mix, bed) | `tests/test_mix.py`, `tests/test_bed.py` | block-size invariance, sax_warm, faders and sum order, calibration; the crossfade formula, the wrap, equal power, the excerpt, the level |

Shared renders are cached per session in `tests/helpers_render.py`. Tests that call FluidSynth skip when it is
not on PATH (`tests/test_render_basic.py` convention). The 8 files hold 88 default tests (28 more are
`-m cluster`), which ran in 11.11 s on the laptop (2026-10-31).
