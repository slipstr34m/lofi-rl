# KITS.md: SF2 coverage, the 48 → 44.1 kHz resampler and the one-shot kit importer

Three pieces of the shootout tooling (docs/SHOOTOUTS.md §7.5, §7.2 and §7.4), built 2026-10-31 on branch `wt/kits`.
Job 1 needs them before its levels pass: the coverage check refuses a palette whose voice would play a note silent,
and the importer turns the Virtuosity and Swirly banks into archives a palette can name.

- **Code:** `src/lofi_rl/palette.py` (`sf2_coverage`, `Palette.note_cells`, `Palette.verify_coverage`),
  `src/lofi_rl/convert/resample.py`, `src/lofi_rl/convert/kit.py`; the kit specs `shootouts/kits/kit_virtuosity.yaml`
  and `shootouts/kits/kit_swirly.yaml`.
- **Tests:** `tests/test_sf2_coverage.py` (10 default, 17 cluster), `tests/test_resample.py` (6 default),
  `tests/test_kit_import.py` (21 default, 3 cluster). The 37 default tests ran in 3.07 s on the laptop (2026-10-31),
  two of them through FluidSynth and two in a fresh Python process. The cluster tests read `assets/big` and skip
  unless `LOFI_ASSETS_BIG` (or `LOFI_ASSETS`) names it:
  `LOFI_ASSETS_BIG=assets/big pytest -m cluster tests/test_sf2_coverage.py tests/test_kit_import.py -s`.
- **Sound claims:** none. Everything below is a count, a hash or a level read from a file or a run.

## 1. SF2 coverage at palette load (§7.5)

`palette.sf2_coverage(path, bank, prog) -> frozenset[(key, velocity)]` is where preset (bank, prog) sounds a sample:
a cell is covered when one of the preset's zones and one of that zone's instrument zones both hold it. Zones are read
as FluidSynth 2.6.1 reads them (its sffile.c and defsfont.c), which three cases decide:

| Case | Rule | FluidSynth 2.6.1, measured 2026-10-31 |
|---|---|---|
| Global zones | the first zone, when it names no instrument (no sample), is global; every later zone starts from its key and velocity ranges and overrides them with its own; a later zone with no instrument or sample is dropped | an instrument global zone with keyRange 0–59 silenced key 60; a preset global zone with velRange 0–63 silenced velocity 100 |
| Generator order | a keyRange counts only as its zone's first generator, a velRange only after nothing or a keyRange (SoundFont 2.04 §8.1.2); generators after the instrument or sample are dropped | a keyRange 0–59 written after a pan was ignored: key 60 sounded |
| Ranges | key and velocity ranges intersect between the preset zone and the instrument zone | a preset zone with keyRange 0–59 silenced key 60 |

`test_the_coverage_reader_agrees_with_fluidsynth` renders these cases and checks that each note sounds exactly where
`sf2_coverage` says it is covered. A missing preset, or a file that is not a SoundFont or whose records are out of
range, raises `PaletteError`.

**What `load_palette(verify=True)` checks** (`Palette.note_cells`, then `verify_coverage`; after the sha256 and preset
checks pass): every note the compiler may send each SoundFont voice.

| Voice | Keys | Velocities |
|---|---|---|
| each sax | its `play` range (the octave fit folds every note into it) | `sax_expr.velocity` min–max (60–112 in dev_gu) and its flat `base` (expression off) |
| each keys voice | 48–72 (`theory.voicing.KEYS_RANGE`) | its `vel` values |
| each kit bass | 28–43 (`compile.bass.BASS_RANGE`) | its `vel` values |
| each SoundFont one-shot | its note | its `vel` |

With humanize on for the keys or bass lane, each `vel` value v widens to rnd(v (1 − h))…rnd(v (1 + h)), h =
`groove.humanize_vel`, because a humanized strike plays v times its velocity factor (COMPILER.md §3.6): an X of 96
reaches 104 at h = 0.08. The error names the voice and lists the missing cells as key runs:
`sax.tenor: source syn bank 0 prog 0 has no zone for keys 44-47 at velocity 60-112; keys 73-75 at velocity 60-112`.

**Cost:** `verify_coverage` on dev_gu took 1.86 ms per call, against 12.73 ms for hashing its sources (laptop, mean of
10). It reads only each SoundFont's pdta chunk (223 kB of GU's 32 MB).

**The real banks** (cluster tests, 2026-10-31). Keys sounding at every velocity 1–127, as §7.5 read them: FreePats tenor
43–89, FM Piano #1 27–98 in each of presets 0–3, Upright KW 21–108, FSBS jazz guitar 35–86, Finger Bass YR 26–45. Each
of those five and the seven converted banks loads with `verify=True` in its shootout role (keys sounding at some
velocity in brackets): the converted tenor and altos as saxes (MTG tenor 44–76, MTG alto 49–81, Weresax 49–80),
jRhodes3d (21–108), the VCSL Knight (21–109) and the Wurlitzer (33–96) as keys, the VSCO 2 pizz (24–60) as a bass. GU presets 4, 16, 32, 65 and 66 cover keys 0–108 at
every velocity 1–127, and the Jazz kit 32 sounds its six notes at 48, 80 and 112.

## 2. The 48 → 44.1 kHz resampler (§7.2)

`lofi_rl.convert.resample.to_44k(x, sr)` is `scipy.signal.resample_poly(x, 147, 160, axis=0)` in float64 on the whole
signal (scipy 1.17.1's default Kaiser window, β = 5, zero padding), time on axis 0, giving ⌈147 n / 160⌉ samples. Any
rate but 48 kHz is refused, 44.1 kHz included.

```
python -m lofi_rl.convert.resample <in> <out.wav>
```

reads anything libsndfile reads, writes a float32 WAV with no PEAK chunk (`wavio.write_float`) through a temporary
file and a rename, and prints the output's sha256, its pcm_sha256 and the frame counts. Exit 2, with nothing written,
for an input that is missing, not 48 kHz or the output itself.

**Measured** (2026-10-31): a 1 s 1 kHz sine at 48 kHz came out as 44,100 samples with its FFT peak at 1000.0 Hz and
its RMS 0.0045 dB above the input's (samples 2,000 to 42,100). One real file,
`virtuosity-drums/Samples/oh/kick/oh_kick_snon_vl4_rr1.flac` (48 kHz 24-bit stereo, 113,255 frames), became 104,054
frames with sha256 `fea58039f2fdc6fb230843e950f2e72971389cf6d73718371766797951504fc4` in two processes with
`PYTHONHASHSEED` 0 and 4242. The rain recording is not here yet; the same CLI converts it when it is (§4.5), and the
excerpt picker of §7.2 is not built.

## 3. The one-shot kit importer (§7.4)

```
python -m lofi_rl.convert.kit shootouts/kits/<kit>.yaml --big assets/big --out assets/big/derived/kits/<kit>.tar \
    [--json report.json] [--reference palettes/dev_gu.yaml] [--reference-kit jazz] [--cache DIR] [--locks DIR]
```

It writes the archive and prints one YAML document: the report in comment lines, then `sources` (the archive's
fetch-mode entry: path under `--big`, sha256, licence, `converted_from` naming the spec and its sha256), `choke`
(the reference kit's, dev_gu's `[[ohat, hat], [ohat, ohat]]`) and `lanes` (one `{src, file}` per layer and round robin,
`pan: 0.0`, `db: 0.0` on the kit's own lanes and the offset on a borrowed one). `--json` adds every one-shot's input
files with their sha256, rate, channels and frames, its trim and its pcm_sha256. Exit 2 for a bad spec or input.

**The spec** (`shootouts/kits/<kit>.yaml`, paths relative to `--big`): `id` (the palette source id), `license`,
`attribution`, an optional kit-wide `mics` premix, and all six lanes. A lane has a `path` template, optional `mics`
(name → dB, or name → {db, variables} for a file-name label) and `borrowed`, and either `vl: {soft, med, hard}` with
`rr: [...]` or `rr: neighbours`, or `all_layers: [{variables}, ...]`. A spec error, an unknown template variable or two
layers naming one member with different files stops `load_spec`.

**Per one-shot** [CD-35]: each mic's file, stereo as (L + R)/2; the mics summed with their dB gains with every file's
sample 0 aligned and shorter files padded at the end; a 48 kHz sum resampled once by `to_44k`; then, on the float32
samples written, the first sample within −30 dB of the peak (−20 dB on the shaker lane) is placed at sample 44 (1 ms,
rounded half up) under a raised-cosine fade-in w[k] = (1 − cos(πk/44))/2 over samples 0–43, and trailing exact zeros
are dropped. A one-shot whose threshold sample comes within 44 samples of its start is padded with zeros, so it too
sits at 44. Inputs are checked against `assets/big/_locks/<folder>.tsv` when the folder has one.

**The archive:** an uncompressed USTAR tar, members sorted, mtime 0, uid and gid 0, empty owner names, mode 0644, no
pax headers; each member a mono float32 WAV at 44.1 kHz with no PEAK chunk, named `<lane>/<shot>.wav` (`kick/vl4_rr1`,
`snare/vl22`, `shaker/D_rr2`), so layers that share a one-shot share a member. The sampler reads a member by seeking
past the others; an archive that would also read as a zip, which the sampler tries first, is refused.

**A borrowed lane** [CD-30] (only a shaker; both kits' shakers are borrowed): o = (ref shaker − ref snare) − (kit
shaker − kit snare), rounded to 0.01 dB. Each term is the integrated loudness (`render.loudness.lufs`) of that lane's
med one-shots alone playing x at `drum_gain["x"]` (0.71) on every 8th note of 2 bars at 85 BPM: a periodic loop of
249,040 samples (N(85)/4), a hit every 15,565 samples, through `render.sampler.mix_hits`, round robins dealt by the
compiler's `round_robin`, pan 0, no metric offsets. The reference shots are dev_gu's Jazz kit 32 rendered by FluidSynth
through the one-shot cache; a spec with no borrowed lane needs no FluidSynth.

### 3.1 Choices the plan left open

Each is pinned by a test.

- **Mic timing:** sample 0 of every mic file aligned, shorter files padded with zeros at the end; one rate per
  one-shot (mixed rates are refused).
- **`± 1 as rr`:** the round robins of a set without them are vl − 1, vl, vl + 1, in that order.
- **Shakers in all layers:** the same list in soft, med and hard (`all_layers`); Virtuosity's alternates D and U from
  round robin 1 to 7 (D1 U1 D2 U2 … U7); Swirly's is the plan's D2 U1 D3 U4.
- **The 1 ms:** 44 samples (44.1 rounded half up, as every sample position); the threshold sample itself is unfaded;
  sample 0 is 0.
- **Too little lead-in:** padded with zeros, so every member's threshold sample is at 44 (58 of Swirly's 64 one-shots).
- **Digital silence:** exact float32 zeros after the last non-zero sample.
- **Member names:** `<lane>/<shot variables>.wav`; distinct one-shots only.
- **The grid** of CD-30: no metric offsets, no swing (8th notes sit on even steps), x gain 0.71, med layer, the loop
  periodic so tails fold.
- **Locks:** files are checked against `assets/big/_locks/` when present; the report says which folders were.
- **Printing:** the source entry and the blocks go to stdout as one YAML document with the report as comments; the
  source path is relative to `--big` when the archive lies under it.
- **One archive per kit:** Swirly's VCSL shaker lane is inside `kit_swirly.tar`, so each kit is one palette source.
- **The reference kit's choke list** is the one every kit takes, read from `--reference` (dev_gu).

### 3.2 The two kits (laptop, 2026-10-31, from `assets/big`)

| | `kit_virtuosity.tar` | `kit_swirly.tar` |
|---|---|---|
| sha256 | `5ca78cde9f96a1e42e1fd558195125529d40508b53cea1aeb2a34735ca66cd9e` | `539f8ca94b189fe9e107859d46caa010e538263844a560a74dec4c01004b0ba5` |
| bytes | 30,986,240 | 18,145,280 |
| one-shots / input files | 65 / 181 (lock: 181 match) | 64 / 100 (locks: 96 Swirly + 4 VCSL match) |
| soft / med / hard per lane | kick 4/4/4, snare 3/3/3, rim 3/3/3, hat 4/4/4, ohat 3/3/3, shaker 14/14/14 | 4/4/4 in every lane |
| resampled | every one-shot but the 14 shakers (48 kHz premix) | none (44.1 kHz) |
| lead-in trimmed, samples at 44.1 kHz (min / median / max) | 11 / 156 / 1,545; none padded | 0 / 0 / 2,457; 58 padded (by up to 44) |
| per lane (min / median / max) | kick 114/141/182, snare 159/211/239, rim 17/154/173, hat 108/139.5/166, ohat 96/136/155, shaker 11/998.5/1,545 | kick, hat, ohat 0/0/0; snare 0/0/9; rim 0/0/11; shaker 1,834/2,167/2,457 |
| shaker offset | +2.47 dB = (−51.623 − −52.242) − (−36.253 − −34.402) LUFS | +6.0 dB = (−51.623 − −52.242) − (−45.218 − −39.838) LUFS |
| shaker − snare on the grid, after the offset | 0.619 LU (GU: 0.619) | 0.620 LU (GU: 0.619) |

- **Determinism:** each archive was byte-identical across two imports in one process, and across two CLI runs in
  fresh processes with `PYTHONHASHSEED` 0 and 4242, the second with a cold one-shot cache (`cmp`, sha256 above).
- **The files** hold what §7.4 read from them: Virtuosity's kickmic and snaremic files mono and its oh files stereo,
  all 48 kHz 24-bit; its perc shakers 44.1 kHz 16-bit (close mono, oh stereo); Swirly mono 16-bit 44.1 kHz; the VCSL
  shakers stereo 16-bit 44.1 kHz. Each lane's velocity picks equal CD-33's from the layers on disk (N = 4, 36, 16, 4,
  4 for Virtuosity; 22, 12, 10, 12, 14 for Swirly).
- **Reading:** with each archive pasted into dev_gu as kit `jazz`, `load_palette(verify=True)` took 49 and 45 ms and
  `render.sampler.kit_shots` read every one-shot in 106 and 75 ms (1 worker; 230 and 163 ms with 8).
- **Ear checks:** none yet. The archives are inputs to the levels pass (§3) and R1.

## 4. Not done here

- **Pinning.** The archives were written to a scratch folder, not `assets/big/derived/kits/`. Claude copies them there
  (or re-runs the importer there), records each in `assets/MANIFEST.yaml` with its sha256 (as CONVERTER.md §9 does for
  the SF2s) and runs `make push-assets`. The sha256s above are the laptop's; a cluster run of the importer should be
  compared with them before the first render.
- **The rain:** the excerpt picker (§7.2) and the converted rain file, due before job 3.
- **The palette generator** (§7.8) pastes the printed blocks into the shootout palettes and adds the kit trim (§3) to
  every lane's `db`.
