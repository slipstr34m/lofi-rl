# Lo-fi sound research for lofi-rl: conventions, redistributable sources, render-chain caveats (as of 2026-10-30)

The repo at /Users/shresthrana/csci596-project currently holds only /Users/shresthrana/csci596-project/CLAUDE.md. `palettes/` and `docs/` do not exist yet. Citations use [keys] defined in the reference list at the end. "Weak" means a practitioner blog or SEO guide. UNVERIFIED means I found no primary source.

---

## Part A: Lo-fi production conventions

### A1. By subgenre

**Jazzhop (Nujabes / J Dilla lineage)**
- **Tempo:** most guides say 70–90 BPM [m-jh]. A Nujabes-style guide says 90–100 [soundcy, weak].
  - Data points: Hooktheory lists Aruarian Dance at 100 BPM [ht-ad] and Feather at 181 BPM, which is double-time of about 90 [ht-fe]. songbpm lists After Hanabi at 83 [sbpm-n] and Dilla tracks at 87–97 [sbpm-d]. songbpm uses automated detection, so treat its numbers as UNVERIFIED.
- **Feel:**
  - Dilla time means several feels at once: some layers straight, some swung, some ahead of or behind the grid [hein].
  - Charnas lists three Dilla techniques [iaspm]: quantize off; slowing sample sources to magnify human error; and, most important, using the MPC timing functions to set elements in conflict. He says the hi-hat and snare treatments recur as formulae.
  - Dilla's offsets were repeated identically, many times per song. They were not random [hein].
- **Drums:** brushed or lightly compressed; soft kick; dusty snare; lightly swung hats [m-jh].
- **Harmony:**
  - Extended 7th/9th/11th/13th chords, Dorian or Lydian colour, add9 and #11 [m-jh].
  - Two-chord ii9–V13 vamp (Aruarian Dance) [ht-ad].
  - Royal Road progression IVmaj7–V7–iii7–vi, and the III7 variant [royal].
  - "Just the Two of Us" / Marusa progression IVmaj7–III7–vi7 (then v7–I7) [flat-jp].
- **Instruments:**
  - Rhodes or mellow piano, soft electric or upright bass, guitar comping [m-jh].
  - Optional sax, trumpet or vibraphone [m-jh].
  - Wikipedia's infobox lists flute among Nujabes's instruments [wiki-nuj].
- **Texture:** low-level crackle, room tone and hiss, plus tape or console saturation [m-jh].

**Study beats (Lofi Girl / Chillhop)**
- **Tempo:** 70–90 [serato]; chillhop 70–92 [m-ch].
- **Drums:**
  - Boom-bap-derived, with soft rounded kicks and snappy but not harsh snares [m-ch].
  - Ghost notes, shakers and rimshots, and soft sidechain swells [m-ch].
  - Common pattern: kick on 1, a ghost kick before 2, an optional double kick before 3, snare on 2 and 4 [ldp].
  - Open hats are choked by the next closed hat (mute group) [attack-chill].
- **Harmony:**
  - 7ths through 13ths, minor keys, ii–V–I, and borrowed iv or bVII [m-ch].
  - Jazzed-up I–vi–IV–V, I–vi–ii–V and ii–V–I, with each chord held a bar or more [flat-lofi].
- **Lead:** sparse melodies on guitar, vibraphone, sax or synth. Bass is electric, upright or sine [m-ch].
- **Chillhop vs lo-fi:** chillhop is cleaner and more jazz-centred; lo-fi foregrounds hiss, crackle and bit artifacts [m-ch]. Wikipedia treats chillhop as an alternate name for the genre [wiki-lofi].
- **Lineage:** Roland SP-303/SP-404 lo-fi effects shaped the sound [wiki-lofi].

**Sleepy / ambient lo-fi**
- **Tempo:** 55–80 BPM [m-sl], or 60–75 [beatkey, weak].
- **Drums:** very soft (brushed snare, gentle rim, muted kick) or none. No abrupt fills or sharp transients [m-sl].
- **Harmony:** maj7, min9 and 7(9,13) in low-to-mid registers; 2–4 slow-moving chords; understated melody [m-sl].
- **Instruments:** Rhodes or soft piano, nylon or clean guitar, soft pads, upright or sine bass [m-sl].
- **Texture:** a stable low-level noise floor of crackle, pink or brown noise, or rain/night ambience [m-sl].

**Dusty 90s boom-bap**
- **Tempo:** 85–96 BPM [m-bb]; about 90 is the centre [routenote-bb, weak].
- **Drums:**
  - Hard kick on 1, snare on 2 and 4 [m-bb].
  - Hats either straight 8ths or swung 16ths, with occasional open-hat off-beats [m-bb].
  - Short room or plate reverb on the snare [m-bb].
- **Attack magazine boom-bap tutorial [attack-bb]:**
  - Kick and snare grooved with MPC3000 8th-note swing 57; hats with MPC3000 16th-note swing 74.
  - Clap nudged a few ms late.
  - Kick EQ: small boost around 60 Hz, large dip around 150 Hz.
  - Sine-like bass; sample transposed down 5 semitones.
  - One workflow guide suggests MPC 58% [audeobox, weak].
- **Sampler character:**
  - SP-1200: 26.04 kHz, 12-bit, SSM2044 filters [sp1200].
  - MPC60: 40 kHz with 12-bit storage (16-bit converters) [mpc60, forum].
- **Bass and samples:** sparse bass that doubles the sample's roots; low-pass filtering to carve space; 12-bit-style grit [m-bb].

### A2. Cross-cutting

**Swing (always ask in MPC terms)**
- In MPC terms, swing is the share of each 8th-note pair given to the first 16th [mpc-man, linn]:
  - 50% = straight; 66% ≈ triplet; maximum 75%.
  - It is implemented by delaying every even-numbered 16th.
- Linn on feel [linn]:
  - 54% loosens a straight beat without sounding swung.
  - At 90 BPM, 62% feels looser than a perfect 66%.
- Other conventions differ [attack-daw]:
  - FL Studio / Cubase 0% = MPC 50%.
  - FL 100% = MPC 66.6%.
  - Guide figures like "30–60%" or "8–15% on hats" use the FL/Cubase scale.
- Delay of each even 16th = (s − 0.5) × 30/BPM seconds:

| Swing | 70 BPM | 80 BPM | 90 BPM |
|---|---|---|---|
| 54% | 17 ms | 15 ms | 13 ms |
| 58% | 34 ms | 30 ms | 27 ms |
| 62% | 51 ms | 45 ms | 40 ms |
| 66% | 69 ms | 60 ms | 53 ms |

- MPC3000 resolution is 96 PPQN [mpc-ppq], so one tick ≈ 6.9 ms at 90 BPM.
- Jazz drummers' swing ratio falls as tempo rises [friberg]:
  - Up to 3.5:1 at slow tempos, about 2:1 near 200 BPM, 1:1 when fast.
  - Drummers always swing harder than the soloist.
  - Soloists lag the beat, by up to about 100 ms at medium tempo. That figure is from a press interview (UNVERIFIED in the paper).
  - To convert a ratio r:1 to MPC %: r/(r+1). So 1.5:1 = 60%, 2:1 = 66.7%.

**"Drunk" / per-instrument microtiming**
- Dilla examples [hein]: claps slightly early and some hats late (Get Dis Money); in E=mc2, almost nothing is on the grid.
- Pudding's interactive demo keeps kick and hat straight and shifts only the snare earlier [pudding].
- One review describes Dilla's nudges as a few milliseconds [4col].
- Hosken's computational analysis of *Another Batch* found the snare backbeats anchor 2 and 4 [iaspm].
- Drummer heuristic reported by Charnas: Nate Smith puts kick and snare one septuplet behind on-beat hats [iaspm]. At 90 BPM that is about 95 ms (my arithmetic), so read it as a feel concept.
- The perceived onset (P-centre) depends on the sound [iaspm]. The same ms shift feels different on a soft bass than on a sharp snare, so offsets should be set per instrument.
- Splice suggests about ±10 ticks of humanize as a starting point [splice-h].
- Design consequence: use fixed per-layer swing and offsets that repeat every bar, not per-hit random jitter.

**Drums**
- Velocity accent pattern [splice-h]:
  - Beat 1 hardest, then beat 3, then 2 and 4; off-beat hats softer.
  - Vary hats most; keep kick and snare consistent.
  - The main fake-sounding causes are flat velocities, copy-pasted hats, the same velocity pattern every bar, and more than two kit pieces at once (other than kick plus hat pedal).
- Ghost notes [bonedo]:
  - Very soft; drop them right before accents.
  - One repeated snare sample sounds machine-like; use a sidestick or other sound.
  - Round robins and velocity layers fix the "machine-gun" effect [kvr-rr].
- Hats: use gentle closed hats; avoid crashes and full open hats [rn-min].
- Rim vs snare:
  - Chillhop uses rimshots and shakers [m-ch].
  - Sleepy uses gentle rim or brushes [m-sl].
- Fills and drop-outs [lm-8bar]:
  - A half-bar or one-bar drum variation at the end of a section.
  - Drop elements (for example hats) for a bar before a change.
  - Lo-fi "drops" are gentle.
- Specific ghost-note MIDI velocities (around 25–45): UNVERIFIED.

**Harmony and voicing**
- Chord qualities: maj7 (warm), m7 (mellow), dom7 (bluesy), 9ths [flat-lofi]; 11ths and 13ths [splice-ch, m-jh].
- Loop families:
  - 2-chord: ii9–V13; IVmaj7–iii7; i–IV Dorian.
  - 4-chord: I–vi–ii–V; Royal Road; Just the Two of Us.
  - Borrowed iv and bVII [m-ch].
- Half-diminished and altered dominants, tritone subs and chromatic approach chords are standard jazz practice. How often they appear in lo-fi is UNVERIFIED.
- Rootless voicings: Dm9 = F–A–C–E, G13 = F–A–B–E, Cmaj9 = E–G–B–D. The root must sound elsewhere, i.e. in the bass [rootless].
- Spread voicings across two octaves [flat-lofi].
- Low-interval limits [sweet-lil]:
  - A major third goes muddy below about B1–D#2.
  - Close voicings below C3 tend to be muddy; this is a rule of thumb.
  - Suggested compiler registers: bass about E1–G2, chords about C3–C5, lead about C4–C6 (my derivation, UNVERIFIED).

**Bass**
- Lines [lofiweekly]:
  - Chord tones on strong beats; the ear eventually wants the root.
  - Chromatic approach notes that surround the target.
  - Mostly descending; stay out of the upper register.
- Timbre by subgenre:
  - Sleepy: upright or sine [m-sl].
  - Chillhop: electric, upright or sine [m-ch].
  - Jazzhop: soft electric or upright [m-jh].
  - Boom-bap: sine-like [attack-bb], or sparse doubled roots [m-bb].

**Melody / lead**
- Sparse and understated [m-sl, m-ch]; call-and-response (question, then answer) [soundfly-cr].
- Wikipedia quotes Pitchfork: slow tempos, sweet piano or guitar melodies, and fake vinyl hiss are hallmarks [wiki-lofi].
- Candidate leads: Rhodes, piano, nylon or jazz guitar, vibraphone, sax, muted trumpet, flute, kalimba or music box.

**8-bar arrangement**
- Loops are 4–8 bars, often with A/B alternation and small variations [lm-8bar].
- For LoopScript (my mapping, standard practice but UNVERIFIED as lo-fi-specific):
  - Bars 1–4 = A, bars 5–8 = A′.
  - Bar 8 is the turnaround (V7alt, bII7, or ii–V back to bar 1) and/or a drum variation or drop-out.
- A seamless loop needs three things (my derivation):
  - Tails wrapped around: render two cycles and keep the second.
  - All LFOs periodic in the loop length L = 1920/BPM seconds. That is 24 s at 80 BPM and 21.3 s at 90; 20–30 s clips mean about 64–96 BPM.
  - Noise beds crossfaded at the seam.

**Master bus and texture level**
- Low-pass: about 12–16 kHz gentle [mpw, weak]; a cassette preset sits around 8 kHz [dusk].
- High-shelf cut: −3 to −6 dB above 8 kHz, or −6 to −10 dB above 6 kHz [acoustica].
- Wow and flutter:
  - Wow about 0.3–0.8 Hz, flutter about 3–7 Hz [dusk].
  - Measurement standards put wow below about 4–6 Hz and flutter at about 6–100/200 Hz [virtins].
  - Goodhertz Wow Control defaults to 0.55 Hz [goodhertz].
  - Real consumer cassette decks measure about 0.05–0.10% WRMS [hk200, k903]; lo-fi plugins exaggerate this.
- Bit/rate reduction: 12-bit at 26.04 kHz (SP-1200) or 40 kHz (MPC60) [sp1200]; 14–16 bits for subtle [mpw, weak].
- Sidechain to the kick, sometimes the whole mix [landr].
- Focus the midrange, roll off the extreme highs and lows; use vintage spring, plate, tape or BBD reverb [landr]. MusicRadar's headline warns against cathedral-sized decays [mr-10].
- Texture bed: practitioners put crackle and rain about 18–30 dB below the main elements, so it is felt rather than heard [violet-crackle, beatsync, weak]. Heavy, deliberate texture goes up to −12 to −8 dB.

**Why MIDI-rendered lo-fi sounds fake, and the usual fixes**

| Cause | Fix | Source |
|---|---|---|
| Flat velocities, hard quantize | Accent pattern plus per-layer swing and offsets | [splice-h] |
| Identical sample repeats | Round robin / velocity layers / alternate articulation | [kvr-rr, bonedo] |
| Dated GM timbres (Sound Canvas lineage) | Multisampled instruments | [gmsynths] |
| Low-mid mud at 200–500 Hz | Subtractive cut, often 300–500 Hz | [hornet] |
| Harsh highs | Shelf or LPF from 6–8 kHz | [acoustica] |
| Busy or bright hats | Gentle closed hats | [rn-min] |
| Cheesy GM square/saw leads and bright brass | Soft leads, sparse lines | UNVERIFIED |

### A3. Multiple-choice axes for the taste interview

- **Tempo band:** 55–65 / 66–75 / 76–85 / 86–96.
- **Swing (MPC %):** 50 / 54 / 58 / 62 / 66, asked separately for hats and for kick/snare.
- **Snare offset:** −10 / 0 / +10 / +20 ms.
- **Snare voice:** snare / rim / cross-stick / brush / clap.
- **Hat grid:** 8ths / 16ths / shaker-led.
- **Chord vocabulary:** 7ths only / +9 / +11 and 13 / +altered and half-diminished.
- **Progression family:** ii–V vamp / I–vi–ii–V / Royal Road / Just the Two of Us / borrowed iv.
- **Voicing:** 3-note rootless / 4-note / spread.
- **Bass:** upright pizz / electric finger / sine sub; root-hold / root-5th-approach / walking.
- **Lead density:** none / 1 motif per 2 bars / call-and-response.
- **Bar 8:** none / fill / drop-out / turnaround chord.
- **LPF:** 6 / 8 / 10 / 12+ kHz.
- **Wow peak depth:** 0 / 0.1 / 0.3 / 1%.
- **Bit reduction:** none / 14 / 12-bit at 26 kHz.
- **Crackle:** off / subtle / featured.
- **Rain:** −30 / −24 / −18 / −12 dB below the music.

---

## Part B: Sound sources

Format: source | roles | license | redistributable? | URL. Files over 100 MiB cannot be pushed to GitHub; 50 MiB gives a warning; total repo ideally under 1 GB [gh-large].

**General-MIDI SoundFonts**
- GeneralUser GS v2.0.3 (2026-02-22), about 30 MB sf2 | Full GM. Includes:
  - Tine EP and FM EP; acoustic, finger and pick bass; nylon guitar.
  - Vibraphone (with and without tremolo), flute, muted trumpet, saxes.
  - Warm and Halo pads, celeste, music box.
  - Drum programs: 8 Room, 24 Electronic, 25 808/909, 32 Jazz, 40 Brush [gu-readme].
  
  | Custom permissive license v2.0 [gu-lic], ScanCode category "permissive" [scancode-gu]. Use is "without restriction for your own music creation, private or commercial"; use in software and modification are allowed. The author asks people not to hotlink his download files, so host your own copy. No attribution is required. The author says he cannot be fully sure where every sample came from. | YES, commit with LICENSE.txt | [gu-repo]
- FluidR3_GM (Frank Wen), 141 MB sf2 | Full GM | MIT, © 2000–2002, 2008 Frank Wen; partly built from public-domain samples [fluid-deb, ms-handbook] | License yes; too big to commit, so fetch script with SHA256. FluidR3Mono_GM.sf3 (13.8 MB) is committable [ms-handbook] | [fluid-deb]
- MuseScore_General | GM | MIT with PD/CC0 parts: drumline samples CC0, Splendid Grand PD, VCSL cymbals CC0, VSCO strings CC0. Acknowledgements must be included in any derivative [msg-deb] | sf3 (35.9 MB) commit with license file; sf2 (208 MB) fetch [ms-handbook] | ftp://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/
- TimGM6mb, 5.7 MB | GM, low quality | GPL-2 | Commit is allowed with the GPL text | [ms-handbook]
- FreePats GM set (2022-10-26) | Incomplete: 45 melodic programs | GPL-3+ with exception | 227 MiB, so fetch | [fp-gm]
- Arachno 1.0 (148 MB) | GM | No license stated; site-wide notice forbids reproduction without authorization; includes sounds from commercial synths | AVOID | [arachno]
- Timbres of Heaven, Magic Sound Font, SGM, Musyng Kite | License unclear (UNVERIFIED) | AVOID | [ms-handbook]

**Per-role candidates.** SFZ entries need conversion to SF2 (Polyphone CLI) or sfizz; see B5.

*Electric piano*
- FreePats FM Synthesized Piano #1/#2 (DX7 EP-style), SF2 | EP | CC0 | YES | [fp-ep]
- VCSL TX81Z "FM Piano" / "Piano 1", WAV/SFZ | FM EP | CC0 | YES | [vcsl]
- Greg Sullivan E-Pianos: Wurlitzer EP200, Yamaha CP80, Hohner Pianet T. SFZ + FLAC, 21.5 MB | Wurlitzer | CC-BY 3.0, attribution requested | YES with ATTRIBUTION file | [gsullivan]
- jRhodes3d / 3c (1977 Rhodes Mk I; 3d is 92 MB, up to 5 velocity layers) | Rhodes | Samples CC-BY-NC-4.0 (3d) or NC-SA (3c); the SFZ files are CC0; music made with it is CC0 | NC, so FETCH-ONLY | [jrhodes]
- No CC0 sampled Rhodes found. GeneralUser's Tine EP is the permissive fallback.

*Piano*
- FreePats Upright Piano KW (2022-02-21; SF2 27 MiB, or 5.8 MiB small; 2 velocity layers) | Upright / felt-ish | CC0 | YES | [fp-piano]
- VCSL Upright Piano Knight and Yamaha; Grand Kawai and Steinway B | CC0 | YES (check size) | [vcsl]
- VSCO2 CE Upright Piano | CC0 | YES | [vsco]
- YDP Grand, SF2 36 MiB | CC-BY 3.0 | YES with attribution | [fp-piano]
- Salamander V3, SF2 296 MiB | CC-BY 3.0 [salamander]. Robopianist claims public domain since 2022-04-03, which conflicts (UNVERIFIED) [robopianist] | FETCH | [fp-piano]
- Splendid Grand | Public domain | YES (size UNVERIFIED) | [sfz-pianos]
- Felt piano: no redistributable option found. Pianobook samples cannot be redistributed [pianobook].

*Bass*
- FreePats Clean Electric Bass YR (pick / finger), SF2 2.2–2.5 MiB | Electric bass | CC0 | YES | [fp-bass]
- VSCO2 CE Solo Contrabass, articulations Pizz / Spic / SusNV / SusVib / Trem | Upright pizz | CC0 | YES | [vsco]
- Karoryfer Black And Blue Basses, Growlybass, Pastabass, Swagbass, Fashionbass (SFZ) | Electric bass | CC0 | YES | [sfz-basses]

*Guitar*
- FreePats Spanish classical guitar, SF2 9.5 MiB | Nylon | CC0 | YES | [fp-nylon]
- FreePats FSBS Electric Guitar Clean #2 (Jazz) and #1 (2026-08-07; SF2 small 5–6 MiB, full 66–129 MiB) | Clean jazz electric | CC0 | YES (small) | [fp-eg]
- Karoryfer Emilyguitar, Shinyguitar, Black_And_Green | CC0 SFZ | YES | [sfz-gtr]

*Mallets, celesta, music box*
- VCSL Vibraphone | CC0 | YES | [vcsl]
- MSLP Vibes | CC-BY 3.0 | YES with attribution | [sfz-mperc]
- VCSL Glockenspiel, Marimba, kalimbas (Kenya, Tanzania), mbiras | Music-box / celesta substitutes | CC0 | YES | [vcsl]
- FreePats Glass (water glasses A4–G6), SF2 9.8 MiB | CC0 | YES | [fp-glass]
- No CC0 standalone celesta or music box found; these exist only inside GM banks.

*Winds and brass*
- VSCO2 CE Flute (expressive vibrato, staccato, sustain with/without vibrato) | Flute | CC0 | YES | [vsco]
- Ixox Flute | CC-BY 4.0 | YES with attribution | [sfz-ww]
- FreePats Tenor Saxophone, SF2 25 MiB or 6.5 MiB | Sax | CC0 | YES | [fp-sax]
- Karoryfer Weresax (alto), Bear Sax (baritone) | CC0 | YES | [sfz-ww]
- MTG Solo Sax | CC-BY 4.0 | YES with attribution | [sfz-ww]
- Muted trumpet: no free sampled Harmon trumpet found; VG's is commercial [sfz-brass]. Use the GeneralUser preset.

*Pads and strings*
- FreePats Synth Pad Choir (GM 92), Sweep (96), New Age (89), Bowed (93); SF2 7–21 MiB | CC0 | YES | [fp-pad]
- VSCO2 CE string sections, solo violin, harp | CC0 | YES | [vsco]

**Drum one-shots.** Extract the chosen hits, process them for lo-fi character, and freeze them as palette WAVs.
- Virtuosity Drums (Versilian; jazz kit). Kick with snares on/off; 11 snare articulations including rimshot and cross-stick; 6 hi-hat articulations; ride and flat ride; about 30 auxiliary pieces including shaker; 1.1 GB | CC0 | YES (commit extracted hits) | [virt]
- Karoryfer Swirly Drums (brushes, including snare "stirs"; 4,500+ WAV, 1.7 GB) | CC0 | YES | [kvr-swirly, sfz-drums]
- Karoryfer Gogodze Phu Vol II (lo-fi-to-hi-fi kit; 1,701 WAV) | CC0 per [sfz-drums], but KVR says GPL-3 [kvr-gogodze] | Check the bundled license first | [rekkerd-gogodze]
- Karoryfer Big Rusty, Unruly, Frankensnare | CC0 | YES | [sfz-drums]
- SM Drums | Public domain | YES | [sfz-drums]
- VCSL percussion: small and large shakers, hi-hat, claps, cabasa, tambourines, modern and rope-tension snares, bongos, conga, woodblock | CC0 | YES | [vcsl]
- MuldjordKit (SF2 53 MiB), DRS Kit, Naked Drums | CC-BY 4.0 | YES with attribution | [fp-drum, sfz-drums]
- Salamander Drumkit | CC-BY-SA 3.0 | YES with attribution | [sfz-drums]
- Do not commit: 99Sounds (no redistribution even for free) [99s]; Pianobook [pianobook]; Looperman and commercial packs (royalty-free is not CC0).

**Textures (Freesound CC0)**
- Rain:
  - allrealsound 161705, "Rain from inside": 37.6 s, 44.1 kHz, mono content, includes hits on an air-conditioner unit | [fs-161705]
  - Metadex 235827, "Soft rain on roof window": 3:16, 48 kHz / 24-bit stereo; contains birdsong and distant thunder, so excerpt carefully | [fs-235827]
  - bastipictures 243781: 65 s, MP3 320 kbps (lossy source) | [fs-243781]
- Crackle:
  - yfjesse 471007, "Vinyl Crackle 33RPM": 5:04, stereo 16-bit 44.1 kHz, recorded from a silent LP | [fs-471007]
  - OwlStorm 371583: 5.1 s loopable, 48 kHz | [fs-371583]
  - DanielsonIII 68400 "Vinyl Hiss" (99 s): CC0 per Freesound's search filter; page not checked individually.
- Tape hiss:
  - foad 443338, "TAPE HISS Sonab c500 A": 63 s, 88.2 kHz / 16-bit mono | [fs-443338]
  - D.S.G. 241415, ondrosik 171097, Littleboot 150005: CC0 per filter; pages not checked individually.
- Freesound's API only serves original files with OAuth2; token access gives lossy previews (MP3 about 128 kbps, OGG about 192 kbps) [fs-api]. Committing the CC0 originals with a manifest (URL, author, license, SHA256) is simpler than a fetch script.

### B5. FluidSynth limits and determinism

**Formats**
- Supports SF2, SF3 (Vorbis-compressed SF2; needs libsndfile) and limited DLS Level 1/2 [fs-sf3, vcpkg-fs]. I found no SFZ loader.
- SFZ options:
  - sfizz (BSD-2-Clause, has a `sfizz_render` CLI) [sfizz, sfizz-man]. The repo was archived on 2026-06-21; last release 1.2.3 [sfizz-rel].
  - Polyphone CLI converts SFZ to SF2/SF3 [polyphone].

**Version: pin 2.6.1 on both machines.** Homebrew and conda-forge are both at 2.6.1 today [brew-fs, conda-fs]. 2.6.0 (2026-08-11) changed four things [fs-260]:
- New output limiter / gain control, `synth.limiter.active`, default off [fs-settings].
- Default reverb is now Dattorro.
- 7th-order sinc interpolation was dropped.
- A sample fine-tune fix that the release notes list as musically breaking.

**Defaults that matter**
- Chorus and reverb are on by default (`synth.chorus.active=1`, `synth.reverb.active=1`) [fs-settings]. Turn both off and do all effects in pedalboard.
- Other defaults [fs-settings]: interpolation is 4th-order [fs-interp]; `synth.cpu-cores=1`; `synth.min-note-length=10` ms; `synth.gain=0.2`; sample rate 44100, fixed when the synth is created.

**Timing**
- Events are processed on 64-sample blocks, about 1.45 ms at 44.1 kHz [fs-timing]. The numpy drum sampler is sample-accurate, so the two paths can be offset by up to about 1.5 ms.

**Output paths**
- `pyfluidsynth.get_samples()` uses `fluid_synth_write_s16` [pyfs]. That path dithers [fd-2016]:
  - The dither index carries over across calls and resets only when a new synth is created [fs-writeint].
  - So a reused synth's output depends on what it rendered before.
  - The dither table is random values computed once; it used `rand()` historically [fd-2007]. The current generator is UNVERIFIED.
- The CLI fast-render path with libsndfile uses `fluid_synth_write_float` and then libsndfile; FluidSynth applies no dither there [fs-filerender].
- Fix: render float, create a fresh synth per render (or system-reset with effects off), keep `cpu-cores=1`, and parallelise across processes. Whether multi-core rendering is deterministic is UNVERIFIED.

**Other**
- pyfluidsynth has no interpolation setter, so use ctypes [pyfs].
- Exact reproduction also depends on the SoundFont having no free-running modulators [tensi].
- Converting SF3 to SF2 once and pinning the file hash would take Vorbis decoding out of the render path (my inference).
- Expect bit-identity only within the same binary and architecture. macOS arm64 and Linux x86_64 will likely differ in the least significant bits (UNVERIFIED).

---

## Part C: pedalboard, numpy DSP, pyloudnorm

**Current version and built-ins**
- pedalboard 0.9.26 (2026-10-07) [pb-pypi]. `BrickwallLimiter` was added in 0.9.25 [pb-rel].
- Built-in plugins, with constructor defaults [pb-pyi, pb-gh]:
  - Bitcrush(bit_depth=8)
  - BrickwallLimiter(ceiling_db=−1, release_ms=100, lookahead_ms=5, true_peak=False)
  - Chorus(rate_hz=1, depth=.25, centre_delay_ms=7, feedback=0, mix=.5)
  - Clipping(threshold_db=−6)
  - Compressor(threshold_db, ratio, attack_ms, release_ms)
  - Convolution(impulse_response_filename, mix, sample_rate)
  - Delay(delay_seconds, feedback, mix)
  - Distortion(drive_db=25)
  - Gain
  - GSMFullRateCompressor(quality)
  - High/LowShelfFilter and PeakFilter(cutoff_frequency_hz, gain_db, q)
  - HighpassFilter / LowpassFilter(cutoff_frequency_hz)
  - Invert
  - LadderFilter(mode LPF12/HPF12/BPF12/LPF24/HPF24/BPF24, cutoff_hz, resonance, drive)
  - Limiter(threshold_db=−10, release_ms)
  - MP3Compressor(vbr_quality=2)
  - NoiseGate
  - Phaser(rate_hz, depth, centre_frequency_hz, feedback, mix)
  - PitchShift(semitones)
  - Resample(target_sample_rate, quality)
  - Reverb(room_size=.5, damping=.5, wet_level=.33, dry_level=.4, width=1, freeze_mode=0)
- Containers: Pedalboard, Chain, Mix. Also VST3Plugin, AudioUnitPlugin (macOS only) and `time_stretch()`.

**Lo-fi behaviour and gotchas** [pb-ref]
- LowpassFilter and HighpassFilter are first-order (6 dB/octave, −3 dB at the cutoff). For real band-limiting use LadderFilter LPF24, cascades, or scipy biquads.
- Resample downsamples and then upsamples back. Quality options run from ZeroOrderHold to WindowedSinc256; ZeroOrderHold aliases, which suits SP-1200 emulation (e.g. target 26040 Hz). Default is WindowedSinc32.
- Distortion is tanh waveshaping. Its 25 dB default drive is very hot.
- Limiter always applies makeup gain and hard-clips at 0 dBFS. BrickwallLimiter offers true-peak mode.
- MP3Compressor is LAME, VBR only, and accepts only certain sample rates (8 to 48 kHz list).
- GSMFullRateCompressor resamples to 8 kHz.
- Reverb is FreeVerb-based (JUCE) [juce-rev].

**Determinism**
- `process(reset=True)` is the default. `reset()` clears reverb tails, delay lines and LFO state [pb-ref].
- Plugins that buffer internally can return fewer samples than they were given, so check output length and alignment, especially for MP3Compressor and lookahead limiters. Chunked processing changes behaviour [pb-276]. Keep `buffer_size` fixed.
- No built-in plugin exposes a seed or random parameter [pb-pyi].
- The README claims strong thread-safety and releases the GIL [pb-gh].
- x86_64 wheels need AVX; 0.9.26 targets an AVX baseline [pb-rel].
- Bit-identity across CPU architectures is UNVERIFIED; test it on the cluster's node type.

**Wow/flutter in numpy**
- Method: a fractional delay read y[n] = x(n − τ[n]·sr) using cubic (Lagrange or Catmull-Rom) interpolation. Plain linear interpolation (`np.interp`) also works and is slightly duller.
- τ(t) = τ0 + Σ Aₖ sin(2π fₖ t + φₖ).
- For a peak pitch deviation p at rate f, the delay amplitude is A = p/(2πf):
  - Wow 0.5 Hz at p = 0.3% → about 0.95 ms (42 samples at 44.1 kHz).
  - Flutter 6 Hz at p = 0.1% → about 27 µs.
  - In cents, deviation ≈ 1731·p, so 0.3% ≈ 5 cents.
- Choose every fₖ = kₖ/L (integer kₖ) so the loop is seamless, and read with circular indexing.
- Replace random "drift" with a few incommensurate integer-k sinusoids, or with raw-bit PRNG noise.

**Vinyl and hiss in numpy**
- Clicks are short impulses from dust and scratches, lasting from under 20 µs to 4 ms. Real records carry about 2,000 per second, mostly inaudible; dense clusters read as crackle [moliner].
- Thumps: an onset under 2 ms plus a decaying low-frequency tail longer than 50 ms. Hiss: EQ-filtered noise. Rumble: low-passed noise [moliner].
- Generate events as a Poisson process (exponential spacing) [vcv-dust], with heavy-tailed amplitudes.
- A secondary page gives the crackle band as about 0.8–8 kHz and pops as a 200 Hz low-pass with about 10 ms decay (UNVERIFIED; the page is behind a login).
- At 33⅓ rpm a revolution takes 1.8 s; pick an integer number of revolutions per loop (my arithmetic).
- Alternative: crossfade-loop the CC0 crackle recording (yfjesse 471007).

**RNG and SIMD caveats**
- numpy's `Generator` streams are only promised on the same numpy build, environment and machine. BitGenerator raw bits are more stable across versions; RandomState is frozen. All three can differ across machines [np-rng].
- numpy picks SIMD code paths at runtime, and this changes results bitwise (e.g. sin/cos on AVX-512 machines) [np-sincos]. `NPY_DISABLE_CPU_FEATURES="X86_V4"` can switch that path off [np-simd].
- Recommendations: use an integer-hash PRNG (e.g. SplitMix64 in Python ints) for humanize timing and noise, pin numpy, and render on one node type.

**pyloudnorm caveats** (v0.2.0, 2026-01-04, implements ITU-R BS.1770-4 [pyln-pypi])
- Measurement [pyln-meter]:
  - 400 ms blocks with 75% overlap; absolute gate −70 LUFS; relative gate −10 LU.
  - Filters are computed when the Meter is created; changing `.rate` afterwards leaves them stale.
  - The "DeMan" filter class is the one described as fully compliant [pyln-pypi].
- Input [pyln-util]: must be float, shaped (samples, channels), at most 5 channels, at least one block long. pedalboard arrays are (channels, samples), so transpose.
- Output [pyln-norm]: `normalize.loudness` is a pure gain. It only warns about clipped samples; it does no limiting and no true-peak handling. A true-peak meter is open issue #61, BS.1770-5 is issue #62 [pyln-issues].
- Procedure:
  1. Measure the music stem.
  2. Set the bed at music LUFS − Δ.
  3. Sum, gain to −14 LUFS, then BrickwallLimiter(true_peak=True, ceiling about −1 dB).
  4. Re-measure and iterate at most 3 times to within ±0.1 LU.
  5. Write float32, or int16 with explicit rounding.
- A mono file measures 3 dB lower than the same signal duplicated to stereo.
- Silence returns −inf, so guard it with the gate.

---

## The 10 facts most likely to change design decisions

1. **Pin FluidSynth 2.6.1 on both machines.** 2.6.0 changed tuning (musically breaking), the default reverb and the interpolation options [fs-260]. Assert bit-identity only on the cluster build. Laptop listening clips should be the cluster renders, pulled back.
2. **The obvious pyfluidsynth path breaks render-order independence.** `get_samples()` dithers, and the dither index persists inside a synth [pyfs, fs-writeint]. FluidSynth's chorus and reverb are also on by default [fs-settings]. Fix: float output, effects off, a new or reset synth per render, `cpu-cores=1`.
3. **Use a hash PRNG, not numpy RNG or SIMD-sensitive functions, for anything stochastic.** numpy random streams and SIMD dispatch are not guaranteed across builds or CPUs [np-rng, np-sincos]. That covers humanize timing, crackle and drift. Render on a single node type.
4. **Make microtiming a fixed per-layer pattern, not random.** Dilla's offsets repeat identically [hein]. Offsets feel different on different sounds [iaspm]. FluidSynth quantises to about 1.45 ms [fs-timing]. Expose per-instrument swing % and ms offsets as palette or brief knobs.
5. **Always speak MPC swing %.** 50 is straight, 66 is triplet, 75 is the maximum. Linn's 54 and 62 are the useful reference points [linn, mpc-man]. Guide numbers like "30–60%" use the FL scale [attack-daw].
6. **The weak licensing spot is Rhodes and Wurlitzer.**
   - Most roles have CC0 SF2s: FreePats upright piano, nylon guitar, jazz clean guitar, electric bass, tenor sax, pads, FM EP [fp-*]; VCSL / VSCO2 CE vibraphone, contrabass pizz, flute (WAV/SFZ, needs conversion) [vcsl, vsco].
   - No CC0 sampled Rhodes, felt piano, muted trumpet, celesta or music box was found.
   - Choose between GeneralUser GS's Tine EP (permissive), Greg Sullivan's Wurlitzer (CC-BY 3.0), or jRhodes3 (NC, fetch-only).
7. **Use SF2 throughout and convert SFZ offline with the Polyphone CLI.** sfizz was archived on 2026-06-21 [sfizz-rel, polyphone].
8. **GitHub's 100 MiB limit decides commit vs fetch.**
   - Fetch: FluidR3_GM sf2 (141 MB), MuseScore_General sf2 (208 MB), Salamander (296 MiB).
   - Commit: GeneralUser GS (about 30 MB), MuseScore_General sf3 (35.9 MB), FreePats small SF2s [gh-large, ms-handbook].
   - Freesound originals need OAuth2 [fs-api], so commit CC0 textures directly.
9. **pedalboard's simple filters are 6 dB/octave, and Limiter adds makeup gain.** For band-limiting use LadderFilter or scipy biquads. Use BrickwallLimiter (true-peak, 0.9.25+) and verify output length for buffering plugins [pb-ref, pb-pyi].
10. **Normalisation will move the rain level unless you fix the ratio first.** pyloudnorm is gain-only. If the bed is set before normalisation in absolute terms, sparse loops get louder rain. Set the bed relative to the music's loudness, then normalise, limit and re-measure [pyln-norm]. Practitioner texture levels sit 18–30 dB under the music [violet-crackle].

---

## References

- [linn] https://attackmagazine.com/features/interview/roger-linn-swing-groove-magic-mpc-timing/
- [attack-daw] https://attackmagazine.com/technique/passing-notes/daw-drum-machine-swing/
- [mpc-man] https://www.manualslib.com/manual/207365/Akai-Mpc-3000.html?page=63
- [mpc-ppq] https://www.manualslib.com/manual/207365/Akai-Mpc-3000.html?page=62
- [attack-bb] https://www.attackmagazine.com/technique/beat-dissected/90s-boom-bap-hip-hop/
- [attack-chill] https://www.attackmagazine.com/technique/beat-dissected/chillout-dusty-hip-hop/
- [iaspm] https://iaspmjournal.net/index.php/IASPM_Journal/article/download/1503/2037/10531
- [hein] https://ethanhein.substack.com/p/dilla-time
- [pudding] https://pudding.cool/2023/06/groove/
- [4col] https://4columns.org/frere-jones-sasha/dilla-time
- [friberg] https://acoustics.org/pressroom/httpdocs/137th/friberg.html
- [ht-ad] https://www.hooktheory.com/theorytab/view/nujabes/aruarian-dance
- [ht-fe] https://www.hooktheory.com/theorytab/view/nujabes/feather
- [sbpm-n] https://songbpm.com/@nujabes
- [sbpm-d] https://songbpm.com/@j-dilla
- [royal] https://en.wikipedia.org/wiki/Royal_road_progression
- [flat-jp] https://blog.flat.io/jpop-anime-chord-progressions/
- [flat-lofi] https://blog.flat.io/lofi-chord-progressions/
- [splice-ch] https://splice.com/blog/?p=40762
- [rootless] https://ledutu.com/en/learn/jazz/rootless-voicings/
- [sweet-lil] https://www.sweetwater.com/insync/low-interval-limit/
- [m-jh] https://www.melodigging.com/genre/lo-fi-jazzhop
- [m-ch] https://www.melodigging.com/genre/chillhop
- [m-sl] https://www.melodigging.com/genre/lo-fi-sleep
- [m-bb] https://www.melodigging.com/genre/boom-bap
- [serato] https://serato.com/studio/tutorials/how-to-make-a-lo-fi-beat
- [soundcy] https://soundcy.com/article/how-to-sound-like-nujabes
- [beatkey] https://beatkey.pages.dev/how-to-make-lo-fi-music
- [routenote-bb] https://create.routenote.com/blog/how-to-make-boom-bap-hip-hop/
- [audeobox] https://www.audeobox.com/learn/mpc-software/classic-hip-hop-workflow/
- [sp1200] https://wiki.riteme.site/wiki/SP-1200
- [mpc60] https://www.mpc-forums.com/viewtopic.php?p=127962
- [wiki-lofi] https://en.wikipedia.org/wiki/Lo-fi_hip_hop
- [wiki-nuj] https://en.wikipedia.org/wiki/Nujabes
- [ldp] https://lofi-drum-patterns.pages.dev/
- [splice-h] https://splice.com/blog/humanize-your-drums/
- [bonedo] https://www.bonedo.de/artikel/einzelansicht/drum-programming-tutorial-6.html
- [kvr-rr] https://www.kvraudio.com/forum/viewtopic.php?p=8380202
- [rn-min] https://create.routenote.com/blog/how-to-make-minimal-lofi-drums/
- [lm-8bar] https://www.loopmasters.com/cloud/blog/4463-Finishing-Tracks-7-Ways-to-Turn-an-Eight-Bar-Loop-into-a-Full-Arrangement
- [lofiweekly] https://www.lofiweekly.com/2021/12/08/crafting-jazzy-basslines-for-lofi-hip-hop
- [soundfly-cr] https://flypaper.soundfly.com/?p=22529
- [landr] https://blog.landr.com/lo-fi-mixing
- [mr-10] https://musicradar.com/tutorials/music-production-tutorials/dont-bathe-every-sound-in-cathedral-sized-decay-10-ways-to-make-lo-fi-sounds
- [dusk] https://dusk-audio.github.io/assets/manuals/tapemachine-manual.pdf (numbers taken from a search snippet)
- [goodhertz] https://goodhertz.com/wow-ctrl/manual/?v=3.6
- [virtins] https://www.virtins.com/doc/Wow-and-Flutter-Measurement-using-Multi-Instrument.pdf
- [hk200] https://support.harmankardon.com/us/en/home-audio/tape-players-recorders/tape-players-recorders-cassette/HK200XM-.html
- [k903] https://www.manualslib.com/manual/196068/Yamaha-K-903.html?page=17
- [acoustica] https://discourse.acoustica.com/?p=19830
- [hornet] https://www.hornetplugins.com/the-low-mids-problem-why-your-mixes-sound-muddy-and-how-to-fix-it/
- [violet-crackle] https://violetrecording.com/best-free-vinyl-crackle-samples/
- [beatsync] https://beatsyncpro.ai/blog/free-vinyl-crackle-samples.html
- [mpw] https://musicproductionwiki.com/articles/how-to-make-a-lo-fi-sample-pack
- [gmsynths] https://gmsynths.blogspot.com/2014
- [moliner] https://ar5iv.labs.arxiv.org/html/2206.06259
- [vcv-dust] https://community.vcvrack.com/t/audible-instruments-macro-oscillator-2-what-is-dust-noise/11307
- [gu-repo] https://github.com/mrbumpy409/GeneralUser-GS
- [gu-lic] https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/documentation/LICENSE.txt
- [gu-readme] https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/documentation/README.md
- [scancode-gu] https://scancode-licensedb.aboutcode.org/generaluser-gs-2.0.yml
- [fluid-deb] https://sources.debian.org/copyright/license/fluid-soundfont/3.1-5.3/
- [ms-handbook] https://musescore.org/en/handbook/soundfont-0
- [msg-deb] https://people.debian.org/~tg/sf/MuseScore_General_Full/copyright
- [fp-gm] https://freepats.zenvoid.org/SoundSets/general-midi.html
- [arachno] http://www.arachnosoft.com/main/soundfont.php
- [fp-piano] https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html
- [fp-ep] https://freepats.zenvoid.org/ElectricPiano/synthesized-piano.html
- [fp-nylon] https://freepats.zenvoid.org/Guitar/acoustic-guitar.html
- [fp-eg] https://freepats.zenvoid.org/ElectricGuitar/clean-electric-guitar.html
- [fp-bass] https://freepats.zenvoid.org/ElectricGuitar/clean-electric-bass.html
- [fp-sax] https://freepats.zenvoid.org/Reed/saxophone.html
- [fp-pad] https://freepats.zenvoid.org/Synthesizer/synth-pad.html
- [fp-drum] https://freepats.zenvoid.org/Percussion/acoustic-drum-kit.html
- [fp-glass] https://freepats.zenvoid.org/ChromaticPercussion/glass.html
- [fp-*] the FreePats pages above
- [vcsl] https://github.com/sgossner/VCSL (subfolders: Idiophones/Struck Idiophones, Idiophones/Plucked Idiophones, Chordophones/Zithers, Electrophones/TX81Z, Membranophones/Struck Membranophones)
- [vsco] https://github.com/sgossner/VSCO-2-CE (Strings/Solo Contrabass, Woodwinds/Flute, Keys, Percussion)
- [sfz-pianos] https://sfzinstruments.github.io/pianos
- [jrhodes] https://sfzinstruments.github.io/pianos/jrhodes3d
- [gsullivan] https://github.com/sfzinstruments/GregSullivan.E-Pianos
- [sfz-basses] https://sfzinstruments.github.io/basses
- [sfz-gtr] https://sfzinstruments.github.io/guitars
- [sfz-mperc] https://sfzinstruments.github.io/melodic-percussion
- [sfz-ww] https://sfzinstruments.github.io/woodwinds
- [sfz-brass] https://sfzinstruments.github.io/brass
- [sfz-drums] https://sfzinstruments.github.io/drums
- [salamander] https://github.com/sfzinstruments/SalamanderGrandPiano
- [robopianist] https://github.com/google-research/robopianist/blob/main/docs/soundfonts.md
- [virt] https://versilian-studios.com/virtuosity-drums/
- [kvr-swirly] https://www.kvraudio.com/news-print.php?id=57708
- [rekkerd-gogodze] https://rekkerd.org/karoryfer-releases-gogodze-phu-vol-ii-free-lo-fi-drums-for-sforzando/
- [kvr-gogodze] https://www.kvraudio.com/product/gogodze-phu-vol-ii-by-karoryfer
- [pianobook] https://lists.linuxaudio.org/archives/linux-audio-user/2021-May/114358.html
- [99s] https://99sounds.org/about/
- [fs-161705] https://freesound.org/people/allrealsound/sounds/161705/
- [fs-235827] https://freesound.org/people/Metadex/sounds/235827/
- [fs-243781] https://freesound.org/people/bastipictures/sounds/243781/
- [fs-471007] https://freesound.org/people/yfjesse/sounds/471007/
- [fs-371583] https://freesound.org/people/OwlStorm/sounds/371583/
- [fs-443338] https://freesound.org/people/foad/sounds/443338/
- [fs-api] https://freesound.org/docs/api/resources_apiv2.html
- [gh-large] https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- [fs-settings] https://www.fluidsynth.org/api/settings_synth.html
- [fs-interp] https://www.fluidsynth.org/api/group__synthesis__params.html
- [fs-260] https://github.com/FluidSynth/fluidsynth/releases/tag/v2.6.0
- [fs-timing] https://www.github.com/FluidSynth/fluidsynth/discussions/1260
- [fs-filerender] https://raw.githubusercontent.com/FluidSynth/fluidsynth/master/src/bindings/fluid_filerenderer.c
- [fs-writeint] https://raw.githubusercontent.com/FluidSynth/fluidsynth/master/src/synth/fluid_synth_write_int.cpp
- [fd-2016] https://mail.gnu.org/archive/html/fluid-dev/2016-01/msg00013.html
- [fd-2007] https://lists.gnu.org/archive/html/fluid-dev/2007-05/msg00009.html
- [pyfs] https://raw.githubusercontent.com/nwhitehead/pyfluidsynth/master/fluidsynth.py
- [fs-sf3] https://www.github.com/FluidSynth/fluidsynth/wiki/SoundFont3Format
- [vcpkg-fs] https://vcpkg.io/en/package/fluidsynth
- [brew-fs] https://formulae.brew.sh/formula/fluid-synth
- [conda-fs] https://anaconda.org/conda-forge/fluidsynth
- [tensi] https://www.tensi.eu/thomas/programming/music_utilities/fluidsynthplugin/FluidSynthPlugin-documentation.pdf
- [sfizz] https://github.com/sfztools/sfizz
- [sfizz-rel] https://www.github.com/sfztools/sfizz/releases
- [sfizz-man] https://man.archlinux.org/man/sfizz_render.1.en
- [polyphone] https://linuxmanpag.es/man/1/polyphone
- [pb-ref] https://spotify.github.io/pedalboard/reference/pedalboard.html
- [pb-pyi] https://raw.githubusercontent.com/spotify/pedalboard/master/pedalboard_native/__init__.pyi
- [pb-gh] https://github.com/spotify/pedalboard
- [pb-rel] https://github.com/spotify/pedalboard/releases
- [pb-pypi] https://pypi.org/project/pedalboard/
- [pb-276] https://www.github.com/spotify/pedalboard/issues/276
- [juce-rev] https://git.kx.studio/DISTRHO/JUCE/raw/commit/0b67a40bb9afb9f3ea71143c935db40ab01d1bdd/modules/juce_audio_basics/effects/juce_Reverb.h
- [np-rng] https://numpy.org/doc/stable/reference/random/compatibility.html
- [np-simd] https://numpy.org/doc/stable/reference/simd/build-options.html
- [np-sincos] https://mail.python.org/archives/list/numpy-discussion@python.org/thread/C6EYZZSR4EWGVKHAZXLE7IBILRMNVK7L
- [pyln-meter] https://raw.githubusercontent.com/csteinmetz1/pyloudnorm/master/pyloudnorm/meter.py
- [pyln-norm] https://raw.githubusercontent.com/csteinmetz1/pyloudnorm/master/pyloudnorm/normalize.py
- [pyln-util] https://raw.githubusercontent.com/csteinmetz1/pyloudnorm/master/pyloudnorm/util.py
- [pyln-pypi] https://pypi.org/project/pyloudnorm/
- [pyln-issues] https://github.com/csteinmetz1/pyloudnorm/issues