# Phase-1 shootout assets

**Status: downloaded and verified on 2026-10-31**, after Shresth's OK ("Yes, if we need it"; every source below is a planned shootout candidate).
- **Size:** 944,155,471 bytes in 13 folders under `assets/big/` (the VCSL piano and shakers share one).
- **Records:** `assets/MANIFEST.yaml` (one fetch-only entry each) and `assets/ATTRIBUTION.md`.
- **Two files beyond the table:** VCSL's LICENSE from master `c1ea7bcc` (its sfz branch has none) and Virtuosity's LICENSE, 7,048 bytes each. jRhodes3d's third SFZ is the root-level `jRhodes3d-mono-no-xfade.sfz`.
- **Archive SF2s** (bank, program): tenor sax, Upright KW, FSBS jazz and Finger Bass YR hold one preset each, (0, 0). FM Piano #1 holds (0, 0) to (0, 3).
- **Sample rates:** 44.1 kHz everywhere except the MTG sax samples (132, 48 kHz 24-bit) and the Virtuosity kit (720, 48 kHz 24-bit); its perc shakers are 44.1 kHz.
- **Still Shresth's to download:** the Freesound rain file (below).

Every size is a byte count from a HEAD request or the GitHub API. A second agent re-checked each one, and each license, independently (workflow `wf_57c1b524-c49`, 2026-10-31), and two sizes plus five license texts were spot-checked by hand. Already in the repo: GeneralUser GS (tenor 66, alto 65, Tine EP 4, Drawbar Organ 16, Acoustic Bass 32, Jazz kit 32).

## Route
- Everything is fetched on the laptop into `assets/big/` (gitignored) and checked against a pinned hash: the sha256 of a release archive, or the git blob SHA-1 of each GitHub file at a pinned commit.
- It reaches the cluster by rsync into `$REMOTE_KEEP/assets/`, outside the code mirror.
- Git gets only `assets/MANIFEST.yaml` (URL, hash, license) and `ATTRIBUTION.md`, so the public repo stays small and every asset is reproducible from its URL.
- **jRhodes3d is non-commercial.** It, and anything converted from it, never enters git and is never shared. Music made with it is CC0, by its license.

## The 14 sources (approved 2026-10-31)
Requested as 944,168,732 bytes; 944,155,471 were fetched. The request counted MTG's four shared files (LICENSE, README and two Data includes: 20,309 bytes) once per sax, and it left out Virtuosity's 7,048-byte LICENSE.

| # | For | Source | What is fetched | Bytes | License | Pinned at |
|---|---|---|---|---|---|---|
| 1 | sax | FreePats Tenor Saxophone, full ("best quality") SF2 | `TenorSaxophone-SF2-20200717.tar.bz2`, freepats.zenvoid.org/Reed/TenorSaxophone/ | 25,757,059 | CC0 1.0 | version 2020-07-17 |
| 2 | sax | Karoryfer Weresax (alto) v1.003: clean-mic samples and SFZ programs | 143 files (128 WAV), github.com/sfzinstruments/karoryfer.weresax | 101,315,389 | CC0 1.0 (from v1.002 on; older copies are CC BY 4.0) | commit `a4d756b` |
| 3 | sax | MTG Solo Saxophones: tenor and alto notes only | 132 FLAC plus the SFZ and text files, github.com/sfzinstruments/MTG.SoloSax | ≈ 47,660,000 | CC BY 4.0 (mapping by kinwie) and CC BY 3.0 (samples, MTG via Freesound): attribution | commit `b494d25` |
| 4 | keys | jRhodes3d (1977 Rhodes Mk I), mono variant | 65 FLAC, 3 SFZ, LICENSE, README, github.com/sfzinstruments/jlearman.jRhodes3d | 22,915,733 | samples CC BY-NC 4.0: never committed or shared | commit `aea5b8d` |
| 5 | keys | FreePats FM Synthesized Piano #1 (modelled on the DX7 E.Piano 1 patch; 3 velocity layers) | `FM-Piano1-SF2-20190916.7z`, GitHub release freepats/fm-piano1 | 13,727,542 | CC0 1.0 | tag 2019-09-16 |
| 6 | keys | FreePats Upright Piano KW, full 2022 SF2 | `UprightPianoKW-SF2-20220221.7z`, freepats.zenvoid.org/Piano/UprightPianoKW/ | 28,832,466 | CC0 1.0 | version 2022-02-21 |
| 7 | keys | VCSL Upright Piano, Knight: sustain samples and SFZ | 90 WAV, 1 SFZ and LICENSE, github.com/sgossner/VCSL | 364,980,766 | CC0 1.0 | commit `dfcf4a4` (sfz branch) |
| 8 | keys | FreePats FSBS Electric Guitar Clean #2 (jazz), bridge pickup, small SF2 | `EGuitarFSBS-jazz-bridge-small-SF2-20260807.7z`, GitHub release freepats/electric-guitar-FSBS-jazz | 5,232,868 | CC0 1.0 | tag 2026-08-07 |
| 9 | keys | Greg Sullivan Wurlitzer EP200 | 42 FLAC, SFZ, LICENSE, README, github.com/sfzinstruments/GregSullivan.E-Pianos | 2,417,317 | CC BY 3.0: attribution | commit `8c3e581` |
| 10 | drums | Virtuosity Drums v0.925: kick, snare (center, rimshot, cross-stick), closed and open hat, ride, shaker; all 6 mics | 776 files, github.com/sfzinstruments/virtuosity_drums | 213,439,142 | CC0 1.0 | commit `9f04cf9` |
| 11 | drums | Karoryfer Swirly Drums v1.104 (brushes), dry only: marching kick, snare main/edge/stir, hats closed/tight/foot/open | 584 WAV and the license, github.com/sfzinstruments/karoryfer.swirly-drums | 91,219,492 | CC0 1.0 (from v1.104 on) | commit `c40dafe` |
| 12 | drums | VCSL Shaker, Small and Large | 22 WAV, github.com/sgossner/VCSL | 3,248,476 | CC0 1.0 | commit `dfcf4a4` |
| 13 | bass | VSCO 2 CE Solo Contrabass, pizzicato | 40 WAV and 1 SFZ, github.com/sgossner/VSCO-2-CE | 20,804,952 | CC0 1.0 (its readme asks for credit; we give it) | commit `6dd651d` |
| 14 | bass | FreePats Finger Bass YR SF2 | `FingerBassYR-SF2-20190930.7z`, GitHub release freepats/electric-bass-YR | 2,613,000 | CC0 1.0 | tag 2019-09-30 |

## Shresth downloads this one (Freesound needs a login)
- **"Soft rain on roof window" by Metadex**: https://freesound.org/people/Metadex/sounds/235827/. The file is `235827__metadex__soft-rain-on-roof-window.wav`, about 56.4 MB (the page says 53.8 MB, which is MiB), 48 kHz, 24-bit, stereo, 3:16, CC0.
- Save it as `assets/big/rain/235827__metadex__soft-rain-on-roof-window.wav`.
- The page says the recording also has a suburban hum, a few birds and distant thunder. Which stretch the bed uses needs your ear check.

## Left out, and why
- **FreePats tenor small SF2** (6.8 MB, the design's pick): replaced by the full build of the same bank, 19 MB more.
- **FreePats FM Piano #2** (4.8 MB): one FM electric piano is enough; #1 has 3 velocity layers and #2 has 1.
- **FreePats Upright KW small** (6.1 MB): it is the 2019 build, while the full download is the 2022 one.
- **VCSL Upright Yamaha** (249 MB): the design has one VCSL upright. The Knight is sampled every whole tone, the Yamaha only on C and G.
- **Not needed for these shootouts:**
  - VCSL Knight release and pedal samples (55 MB): SF2 has no release triggers;
  - the Virtuosity full zip (1.23 GB) and the Swirly wet rooms (1.48 GB);
  - the Weresax dynamic-mic set (101 MB);
  - the MTG soprano, baritone and noise samples.

## The SFZ → SF2 converter
Six candidates are SFZ instruments, which FluidSynth cannot load: Weresax, MTG, jRhodes3d, the VCSL Knight, the VSCO 2 pizz and the Wurlitzer.

instrument.md §4 planned to convert them with the Polyphone CLI in a cluster job. That cannot work:
- Polyphone needs a GUI even from its command line;
- its Homebrew cask was disabled on 2026-09-01;
- its macOS app is only ad-hoc signed, so installing it means overriding Gatekeeper in System Settings, which is Shresth's call;
- it drops the ARIA features these banks use: round robins, random layers and CC conditions.

**Default:** a small, deterministic SFZ → SF2 converter in Python, built and tested like the parser. It runs on both machines and needs no download. Each bank gets a reduced mapping that keeps only what SF2 can express.

**Fallback, only if Shresth prefers it:** Polyphone 2.6.0 (`Polyphone_2.6.0-macOS_12.dmg`, 31,438,016 bytes, GitHub release davy7125/polyphone, GPL-3.0-or-later), installed and allowed by Shresth himself.

## What the research corrected in the design
- **Swirly** has no rim click and no shaker, and its kick is a marching kick. Its rim lane uses `snare_edge`, and the shaker comes from VCSL, as §6 already planned.
- **FreePats tenor sax** is built from VCSL samples, not recorded by FreePats.
- **FreePats finger bass** covers MIDI 26–45. That holds the design's bass register (28–43, E1–G2), though notes above 39 are pitched up from the D#2 sample.
- **MTG presets** depend on ARIA features. Use its non-looped presets with round robin 1 only, the p and f layers, and no noise regions; its tuning corrections (`tune=`) must survive conversion.
- **Weresax's vibrato and humanize** are CC-driven ARIA features and are dropped. The compiler's deterministic sax expression replaces them.
- **jRhodes3d:** convert the no-crossfade mono mapping, with its tremolo and autopan LFOs off.
- **Sizes in `docs/research/sound_sources.md`** were compressed or unpacked sizes. The table above gives what is actually fetched.
- **Unverified until fetched:** the sample rate and bit depth of every source, and the inner SF2 file names of the FreePats archives. MTG is 48 kHz per its README, and the rain is 48 kHz per its page; the palette runs at 44.1 kHz.
- **GitHub limits** its anonymous API to 60 requests an hour per IP. The file lists are built on the laptop with the authenticated `gh` (read-only).
