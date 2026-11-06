# Instrument subsystem: LoopScript v1, compiler, renderer, palette, verifiers

Repo root: `/Users/shresthrana/csci596-project`. Paths below are relative to it. Every audible value here is a starting point for an ear check or a shootout. None of it is a claim about how anything sounds.

## 0. Claude's default calls
All defaults are applied, with three amendments and one mild dissent.

**Amendments**
1. **Free sax notes are defined by pitch class, not by spelling.** Sax degrees are relative to the mode, so a plain degree is always in the mode.
   - A note is free if its pitch class is in the mode, is a blue note (3, 6 or 10 semitones above the tonic), or is a tone of the chord sounding at that moment.
   - So b3 and b7 only need writing in hopeful; b5 needs writing in every mood.
   - A held blue note is never penalized. The taste row says "long-note color is left to votes".
2. **`lofi_rl.dsl` uses only the standard library and runs in the policy env.** Gates G1–G6 therefore zero out junk before any request reaches the scorer server.
3. **"Two cycles, keep the second" must also cover the bus effects.** The limiter runs on the loop tiled three times (a 3× tile).

**Dissent (mild).** MPC 16th swing leaves 8th-note sax lines straight. Keep 55% as the default, but audition the sax's feel (Open issue 2).

## 1. LoopScript v1
```ebnf
loop    = { [stmt] [comment] EOL } ;              (* one statement per line, any order *)
stmt    = vamp | keys | comp | bass | drums | lane | edit | sax | phrase ;
vamp    = "vamp:" chord chord ["span" ("1"|"2")] ;  (* bars per chord, default 2: ABAB.. or AABB.. *)
chord   = numeral quality ;
numeral = "I"|"bII"|"II"|"bIII"|"III"|"IV"|"V"|"bVI"|"VI"|"bVII"|"VII" ;  (* re tonic, major-scale reference *)
quality = "maj7"|"maj9"|"maj7#11"|"7"|"9"|"13"|"7sus"|"7b9"|"7#9"|"m7"|"m9"|"m11"|"m7b5"|"dim7" ;
keys    = "keys:" KEYS ("rootless"|"shell"|"spread"|"quartal") ;
comp    = "comp:" sgrid [sgrid] ;                  (* 1- or 2-bar base, tiled *)
bass    = "bass:" ("lock" | sgrid [sgrid]) ;       (* rhythm only; pitch = chord root *)
drums   = "drums:" KIT ;
lane    = dlane ":" dgrid [dgrid] ;  dlane = "kick"|"snare"|"rim"|"hat"|"ohat"|"shaker" ;
edit    = "@" ("1".."8") (dlane ":" dgrid | ("comp"|"bass") ":" sgrid) ;  (* replaces that lane, that bar only *)
dgrid   = 16*("X"|"x"|"g"|".") ;                   (* accent/normal/ghost/rest *)
sgrid   = 16*("X"|"x"|"g"|"-"|".") ;               (* "-" holds the previous strike *)
sax     = "sax:" SAX ;
phrase  = ("A"|"B") ":" item {item | "|"} ;        (* A = bars 1-4, B = 5-8, 64 steps each *)
item    = dur ("r" | ["b"|"#"] ("1".."7") {"'"|","}) ;   (* mode degree; ' up / , down an octave *)
dur     = ("s"|"e"|"q"|"h"|"w") ["."] ;            (* 1,2,4,8,16 sixteenths; dot x1.5; no "s." *)
comment = ("#"|"//") (SPACE|EOL) {any} ;           (* "#" inside a token is a sharp *)
```
- **Required statements:** vamp, keys, comp, bass, drums, kick, one of snare/rim, sax, A, B.
- **Names:** KEYS, KIT and SAX come from the palette lists.
- **Never written by the model:** pitches or note names, tempo, key or mode, swing, effects, levels, the bed.

**Example.** Brief: city at night, wistful (D dorian), energy mid, 82 BPM. The model sees neither key nor mode. The names are `DEFAULT_VOCAB`'s (`tests/fixtures/loops/example.ls`). The frozen palette's names are rhodes, wurli and piano, studio and jazz, and alto (DSL.md §5), and under it `drums: dusty` and `sax: tenor` are G4.
```
vamp: Im9 IV13 span 2
keys: rhodes rootless
comp: x-----.x----..x- x-------..x-x---
bass: x-------..x-x--- x-----.x-x------
drums: dusty
kick: x......x..x..... x......x.x....x.
rim: ....X.......X... ....X..g....X..g
hat: x.x.x.x.x.x.x.x. x.x.x.x.x.x.xxx.
ohat: ................ ..............x.
@3 hat: x.x.x.x.x.x.x.xx
@4 kick: x......x..x...x.
@5 comp: x-----.x--x-..x-
@6 rim: ....X..g....X.g.
@7 ohat: ......x.........
@8 comp: x-----.x----x-x-
sax: tenor
A: q5 e6 e5 q3 e2 e1 | h3 er e4 e5 e6 | h.5 er e4 | q3 q2 hr
B: q1' e7 e1' q5 q3 | er e3 e4 eb5 h5 | qr e4 e3 q2 q1 | h.6 qr
```
How the example obeys the profile:
- Bar 1 lands on A3, the 5th of Dm9; bar 5 lands on D4, its root.
- 4 notes last 8 steps or more.
- Ab is the blue b5.
- The answer ends on B, off the tonic.
- Bars 1–8 are all different in the rhythm section.
- Bar 8 keeps the base drums.
- Bass and kick coincide on 19 of 24 bass onsets.
- Sax range D3–D4.
- It scores 1.0 on every prior in section 5.

**Token count: 268** (584 characters).
- **How measured:** Qwen3 BPE, using `tokenizers` on `~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/*/tokenizer.json` (vocab 151,669), this session.
- **Still to do:** re-measure with the 2507 tokenizer in `smoke_env` and log the number in CHANGELOG.
- **Bigger variants:** adding a shaker lane and a 7th edit gives 294 (measured). A busy loop (24 notes per phrase, 5 lanes × 2 bars, 7 edits) is estimated at about 340.

Why it stays under 400:
- Runs of rests merge: 16 dots are 1 token, and `x......x..x.....` is 6.
- Writing the duration first costs 2 tokens per note (`q5`). Degree-first `5q` costs 3: 51 vs 37 tokens per phrase line, measured.
- A rest is 1 token (`qr`).

Settings: soft penalty above 360 tokens; `max_completion_length=448`.

## 2. Hard gates and leniency
**Gates** (each sets reward to 0 and logs a reason code):

| Gate | Fails when |
|---|---|
| G1 | parse error after normalization |
| G2 | `finish_reason == "length"` |
| G3 | a required statement is missing |
| G4 | a name, quality or numeral is not in the palette after aliasing |
| G5 | the failing token is forbidden content: a note name or absolute chord (`Dm9`, `C4`), a tempo/bpm/key/mode word, an effect or bed word, or an edit on vamp/sax. G5 is a labelled subset of G1, for the junk report. |
| G6 | the vamp is not 2 distinct chords; a grid has fewer than 12 or more than 20 cells; a phrase is off by more than 16 steps; a phrase is empty; kick, or both snare and rim, are silent in all bars |
| G7 (render) | failure or NaN; music below −50 LUFS; length ≠ N; final LUFS not within ±0.5 LU; true peak above −0.9 dBTP |

**Leniency.** Steps run in this order. Each step appends `Repair(code, line, before, after)` (LESSONS 32), and the counts appear in `make junk-rate`.
1. Strip code fences, and any prose outside the first and last keyword lines (`strip_prose`).
2. Strip comments.
3. Normalize whitespace: tabs, CRLF, runs of spaces, blank lines, and `=` used as `:`.
4. Case-fold keywords, names and durations. Never case-fold grid cells: `X` and `x` differ.
5. Aliases, logged with no penalty:
   - lane names `bd`, `sd`, `hh`, `oh`, `shk`;
   - `-` read as `.` in drum grids;
   - qualities: `M7`/`Δ7` → maj7, `-7`/`min7` → m7, `ø`/`ø7` → m7b5, `°7` → dim7, `sus`/`9sus4` → 7sus;
   - a lowercase numeral means minor (`ii9` → `IIm9`);
   - a missing `span` becomes 2.
6. Repairs, penalized 0.02 each with a cap of 0.1:

| Problem | Repair |
|---|---|
| grid of 12–15 cells | pad with `.` |
| grid of 17–20 cells | truncate |
| phrase off by 16 steps or fewer | pad with a rest, or truncate |
| bare numeral | the mode's diatonic 7th chord |
| numeral case contradicts the quality | the quality wins |
| duplicate statement, or both snare and rim | the first wins |
| misplaced `|` | logged only |
| sax note out of range | octave fold |

Because aliases are free, the Phase 1 gate measures structure, not spelling.

## 3. Compiler (`src/lofi_rl/compile/`)
**Pitch mapping**
- Modes:
  - melancholy = aeolian [0,2,3,5,7,8,10]
  - wistful = dorian [0,2,3,5,7,9,10]
  - hopeful = ionian [0,2,4,5,7,9,11]
- Sax pitch = T + S[d−1] + accidental + 12·octave. T is the brief's tonic placed inside the instrument's home band.
- Chord root = tonic + I 0, bII 1, II 2, bIII 3, III 4, IV 5, V 7, bVI 8, VI 9, bVII 10, VII 11.
- Chord tones:

| Quality | Semitones |
|---|---|
| maj7 / maj9 / maj7#11 | [0,4,7,11] / +14 / +14,18 |
| 7 / 9 / 13 | [0,4,7,10] / +14 / +14,21 |
| 7sus | [0,5,7,10,14] |
| 7b9 / 7#9 | [0,4,7,10] +13 / +15 |
| m7 / m9 / m11 | [0,3,7,10] / +14 / +14,17 |
| m7b5 | [0,3,6,10,17] |
| dim7 | [0,3,6,9] |

**Registers** (sounding MIDI)

| Instrument | Playable | Comfort | Home tonic band |
|---|---|---|---|
| Tenor sax | 44–75 | 48–72 | 50–61 |
| Alto sax | 49–80 | 55–77 | 57–68 |
| Keys | 48–72 (C3–C5) | | |
| Bass | 28–43 (E1–G2) | | |

- **Sax octave fit:** per phrase, pick the shift k ∈ {−1, 0, +1} that minimizes notes outside the comfort window (ties go to 0). Fold any remaining outliers ±12 into the playable range; each fold is a logged repair.
- **Bass octave:** choose (pA, pB) to minimize |pA−36| + |pB−36| + 0.5·|pA−pB|; ties go lower.

**Voicings**
- **shell:** 3rd (or 4th for sus) + 7th + the top extension.
- **rootless:** the core tones minus the root, plus the first extension; the 5th is dropped when there is an extension.
- **spread:** drop-2 of the rootless voicing.
- **quartal:** maj [4,9,14,18|19], min [5,10,15,19], dom [10,16,21,+ext], sus [5,10,14,19]. m7b5 and dim7 fall back to rootless.

How a voicing is chosen:
1. Candidates are every inversion at every octave that fits in 48–72.
2. Reject any interval of 2 semitones or less whose lower note is below 55. Reject 13-semitone pairs unless the chord is 7b9.
3. Choose the pair that minimizes VL(A→B) + VL(B→A) + 0.25·(|μA−60| + |μB−60|), where μ is mean pitch and VL is the summed movement under sorted pairing.
4. Ties break lexicographically. Each chord keeps one voicing for the whole loop.

**Chord changes and `bass: lock`**
- A keys or bass note held across a chord change is cut at the change. It is re-struck if the grid shows `-` at that step.
- `bass: lock` takes its onsets from the non-ghost kicks after edits. Each note holds to the next onset, at most 8 steps. Every chord change forces an onset.

**Timeline and swing**
- N = 64·⌊1,323,000/BPM + ½⌋ samples. N is a multiple of 64, so FluidSynth's 64-sample blocks line up in every cycle. BPM_eff = 84,672,000/N is logged.
- Step k ∈ [0,128): t_k = k·N/128 + [k odd]·(s − 0.5)·N/64 + offset_lane, rounded to a sample. This is sound.md's (s − 0.5)·30/BPM.
- s = 0.55 per lane by default (palette).
- MIDI is written at PPQ 22050 with a fixed 120 BPM tempo, so 1 tick = 1 sample.

**Velocities**

| | X | x | g |
|---|---|---|---|
| Drum sampler gain | 1.0 | 0.71 | 0.32 |
| Keys (MIDI) | 96 | 80 | 52 |
| Bass (MIDI) | 104 | 88 | 64 |

Hat and shaker metric offsets: steps 1 and 9 at 0 dB; 5 and 13 at −1 dB; other odd steps −2 dB; even steps −4 dB.

**Sax expression.** Pitch-bend range is ±2 semitones (set by RPN at t = 0). Pitch bend is written every 10 ms and CC11 every 20 ms. Everything is a function of the note list and t mod L, so it is periodic in L:
- **Scoop:** notes of 4+ steps that follow a rest or a leap of 3+ semitones bend from −60 cents to 0 over 80 ms (raised cosine).
- **Vibrato:** notes of 8+ steps start vibrato after 0.35 × duration, fade in over 300 ms, depth ±18 cents. Rate f_v = round(5.5·L)/L, a whole number of cycles per loop, phase-locked to the loop clock.
- **CC11:** short notes 110; long notes 100 + 27·sin(πτ/d); the last note of a phrase decays to 80 over its final 40%.
- **Velocity:** 92 base; +8 on step 1, +4 on beat 3, −10 for `s` notes, +6 on phrase starts; clipped to 60–112.
- **Note-off:** 15 ms before a note that follows immediately.

**Humanize** (`compile/humanize.py`, shootout only)
- PRNG: SplitMix64. Test vector: splitmix64(0) = 0xE220A8397B1DCDAF. Uniform draw u = (h>>11)·2⁻⁵³.
- Amounts: timing ±3 ms, velocity ±8%, uniform.
- Key for H1 (same nudge every bar): seed⊕lane⊕step. Key for H2 (varies by bar, repeats every loop): also ⊕ bar mod 8.
- seed = SHA256(canonical text)[:8].

**Drum sampler** (`render/sampler.py`)
- Velocity layer by level (soft/med/hard), falling back to the nearest layer.
- Round-robin: rr = h mod R, where h is the hit's index within its lane and layer in the cycle (periodic).
- The open hat is choked by the next closed hat with an 8 ms fade.
- Hits are summed in float64 in sorted order, with constant-power pan.

**Seamless.** The MIDI holds 2 cycles (end-of-track at 2N + 0.1 s). Bus chains run on both cycles, then [N, 2N) is kept.

## 4. Renderer (`src/lofi_rl/render/`)
**FluidSynth 2.6.1.** One fresh process per stem (keys, bass, sax):
```
fluidsynth -q -n -i -R 0 -C 0 -o synth.cpu-cores=1 -o synth.limiter.active=0 -r 44100 -T wav -O float -F stem.wav <sf2> stem.mid
```
- Interpolation stays at the default (4th order).
- If `fluidsynth --version` does not report 2.6.1, the renderer raises.

**Assets**
- SF3 and SFZ sources are converted to SF2 once, in `jobs/lofi-assets.sbatch` (Polyphone CLI). The SHA256 is pinned in the palette and checked at load.
- Fetched, NC or large assets live in `$LOFI_ASSETS` (`/project2/anakano_429/shresthr/lofi-rl-assets`; verify with myquota per LESSONS 6), outside the rsync `--delete` mirror.
- Small CC0 and permissive assets are committed under `assets/`, listed in `assets/MANIFEST.yaml`.

**pedalboard 0.9.26 chains.** Starting values, to be tuned on the 10 Phase 1 hand-written loops before the freeze:

| bus | chain |
|---|---|
| sax | HighpassFilter 140 → PeakFilter 450 Hz −2 dB q1 → Compressor −20 dB 3:1 8/120 ms → PeakFilter 2.8 kHz +2 dB q.8 (presence) → Reverb room .22 damp .55 wet .07 dry 1 |
| sax_warm (flag; replaces that reverb) | HighShelfFilter 4 kHz −5 dB → LowpassFilter 9 kHz → Reverb room .35 damp .7 wet .12 dry 1 |
| keys | HighpassFilter 120 → PeakFilter 400 Hz −3 dB → Compressor −18 2:1 20/150 → Reverb room .3 wet .10 dry 1 |
| bass | HighpassFilter 30 → PeakFilter 250 Hz −2 dB → Compressor −20 4:1 15/120 |
| drums | HighpassFilter 30 → PeakFilter 350 Hz −2.5 dB → Compressor −16 3:1 10/80 |
| master | HighpassFilter 25 → PeakFilter 300 Hz −1.5 dB q.7 (low-mid cut) → Compressor −12 1.5:1 30/200 |

- **Bus balance** uses static `bus_gain_db`, calibrated so the median bus loudness on the hand loops is sax 0, drums −1.5, bass −3, keys −6 LU. The sax is panned center. Per-loop bus normalization was rejected because it would flatten the energy knob.
- **Excluded by the profile:** wow, crackle, bitcrush.

**Rain bed**
- Source: a CC0 rain excerpt, with HighpassFilter 200.
- Seam: an equal-power crossfade over X = 1 s:
  - o[n] = e[X+n] for n < N−X;
  - over the last X samples, o[n] = cos θ·e[X+n] + sin θ·e[n−N+X].
- Level: LUFS_bed = LUFS_music − Δ, with the music measured after the master chain and before the limiter.

**Default Δ = 24 LU**
- sound.md gives 18–30 dB below the music as the practitioner range for texture that is felt rather than heard, and −12 to −8 dB for featured texture.
- "Barely there" takes the middle-to-quiet side of that range.
- The rain is stationary, so it should mostly surface when the band thins. That is a hypothesis, and the rain sanity check decides it.

**Loudness**
1. Tile the loop 3 times and render y = BrickwallLimiter(ceiling −1, true_peak, 100 ms, 5 ms)(tile₃(x·10^(g/20))).
2. Keep [N+λ, 2N+λ), where λ is the limiter latency measured once with an impulse.
3. Measure LUFS L with pyloudnorm (input shaped samples × channels) and update g += −14 − L.
4. Stop when |L + 14| ≤ 0.1, at most 4 iterations; otherwise gate G7.
5. Check true peak separately with 4× `resample_poly`.

**Outputs**
- **Canonical, on the cluster:** `$REMOTE_SCR/renders/<run>/<id>/`
  - `render.wav`: float32, 44.1 kHz, stereo, exactly N samples.
  - `music.wav`: the same without the bed, for the scorers.
  - `manifest.json`: canonical text and its SHA, brief, palette SHA, platform key including CPU model, N, BPM_eff, LUFS, true peak, Δ, gain, repairs, gates, and `audio_sha256` (SHA256 of the float32 little-endian PCM).
- **Listening copies:** `$REMOTE_SCR/inbox/<date>/<opaque>.flac`, 16-bit with TPDF dither from SplitMix64, brought to the laptop by `make pull`. The map from opaque id to loop stays in `runs/`.

## 5. Verifiers (`src/lofi_rl/verify/`; bars b = 1..8, steps 1..16)
**Groove:** G = .30·f_bb + .25·f_k1 + .25·f_lock + .20·f_tk
- f_bb (backbeat) = mean over bars of [non-ghost snare/rim on steps 5 and 13].
- f_k1 (kick on 1) = mean over bars of [non-ghost kick on step 1].
- f_lock (kick-bass lock) = min(1, (|K∩B|/|B|)/0.6), over (bar, step) onsets.
- f_tk (timekeeping) = mean over bars of [hat + shaker hits ≥ 4].

**Sax:** S = .25·f_long + .25·f_land + .20·f_open + .15·f_ans + .15·f_breath
- f_long: 1 if 2–4 notes last 8+ steps; 0.5 if 1 or 5; otherwise 0.
- f_land: mean over A and B. Each scores 1 if its first step has an onset on a chord tone, 0.5 if the onset is a non-chord tone, 0 if it is a rest.
- f_open: [the last pitch class of B ≠ tonic].
- f_ans: 0 if B repeats A exactly. Otherwise take J = Jaccard of the two phrases' relative onset steps: 1 for 0.2 ≤ J ≤ 0.8, falling linearly to 0 at J = 0 and J = 1.
- f_breath: mean over phrases of [0.12 ≤ rest share ≤ 0.6, and at least one rest of 2+ steps].

**Chromatic notes:** P_chrom = min(.3, .1·#bad). A non-free note is bad unless it lasts 2 steps or fewer and the next note is a free note 1–2 semitones away.

**Something new every bar:** f_new = (1/7)·Σ_{b=2..8} [sig(b) ∉ {sig(1..b−1)}], where sig = (chord, 5 drum lanes, bass, comp). The sax is excluded because it is new by construction.

**No fill at the seam:** f_seam = 0 if any of these hold:
- bar 8 has 3 or more non-ghost snare/rim hits in steps 9–16;
- Hamming(bar-8 drums, base pattern for bar 8) > 6 cells;
- bar 8 lacks the backbeat, or bar 1 lacks the kick on 1;
- bar 8 has fewer hat + shaker hits than max(2, ½ × the median of bars 1–7).

**Theory score:** theory = clip(.3·G + .3·S + .2·f_new + .2·f_seam − P_chrom, 0, 1).

**Energy**
- E = .35·min(1, H/192) + .15·(n_X/n_non-ghost) + .25·min(1, n_keys/48) + .25·min(1, n_bass/48).
- H counts drum hits over 8 bars, with X and x weighted 1 and g weighted 0.5.
- Bands are the tertiles of E over the seed pool, frozen in `config/energy_bands.yaml` with a DECISIONS.md entry.
- f_energy = exp(−d/0.04), where d is the distance outside the target band. Obedience is reported as a band match.

**Mood**
- score_m = Σ_chords bars_c · Σ_tones w·s_m(pc).
- w = 1 for root, 3rd/4th and 7th; 0.5 otherwise.
- s = +1 if the tone is in the mode, 0 if it is in Alt_m, −1 otherwise. Alt: aeolian {11, 1}, dorian {11}, ionian {8, 10}.
- f_mood = [the target mode wins, by a margin of at least 2]. The example scores dorian 34, aeolian 26.

**Penalties**
- **Near-duplicate in group:** P_dup = .3·clip((max_j sim − .8)/.15, 0, 1).
  - sim is the Jaccard of event sets: drum (lane, bar, step, level); sax (bar, step, relative pitch, duration); chords.
  - Same-brief cousins below 0.8 cost nothing.
- **Repeated bars:** P_rep = .1·max(0, m − 3), where m is the largest count of identical full bars, sax included.
- **Over-length:** P_len = min(.3, .004·max(0, T − 360)).
- **Copy of a pool item:** P_pool = .3·clip((max sim to the pool ∪ prompt examples − .75)/.15, 0, 1).
- **Repairs:** P_fix, as in section 2.

**Fixtures** (`tests/fixtures/loops/`; the pass fixture is `example.ls` unless noted)

| verifier | fail fixture → expected |
|---|---|
| gates | `tempo: 90` → G5; `vamp: Im9` → G6; 9-cell kick → G6; `drums: trap808` → G4; no `B:` → G3; finish = length → G2; `kick: four on the floor` → G1 |
| leniency | pass: the example in caps, with tabs, `# notes`, a ``` fence and "Here's a loop:" → same AST plus [strip_prose]. fail: a lowercase `x` written where `X` was meant changes the AST (case is kept) |
| groove | snare `x.......x.......`, kick never on 1, bass only on offbeats → G < .3 |
| sax | A opens with `qr`, no note of 8+ steps, B = A ending on `1` → S ≤ .3 |
| chromatic | pass `e#4 q5`; fail `h#4 q1` → P = .1 |
| new every bar | 1-bar bases, no edits → f_new = 0 |
| seam | `@8 snare: ........XXXXXXXX` → 0; bar 8 hat and kick all `.` → 0 |
| energy | quarter hats, `x---…` comp and 2 bass hits classify low; 16th accented hats, 6 stabs and 6 bass hits classify high (test bands .30/.50); E rises strictly as hits are added |
| mood | `Im9 IV13` passes wistful and fails melancholy; `Im7 Vm7` ties aeolian and dorian → fails both |
| dup / rep / pool / length | one hat cell changed → P_dup .3, while 3 changed edits → 0; static rhythm with sax only in bars 1 and 5 → m = 4; example in pool → P_pool .3; padded to 420 tokens → P_len .24 |

## 6. Palette and shootouts
`palettes/lofi_rain_v1.yaml`, validated by `src/lofi_rl/palette.py`:
```yaml
id: lofi_rain_v1; frozen: null; sr: 44100; assets_root: $LOFI_ASSETS
sources: {<id>: {path, sha256, license, mode: commit|fetch, url, converted_from, attribution}}
sax:  {tenor: {src, bank, prog, play: [44,75], comfort: [48,72], home: [50,61]}, alto: {...}}
keys: {<name>: {src, bank, prog, vel: {X: 96, x: 80, g: 52}}}
kits: {<name>: {bass: {src, bank, prog}, choke: [[ohat, hat]],
       lanes: {kick: {soft: [wav], med: [wav], hard: [wav], pan: 0, db: 0}, snare: ..., rim: ..., hat: ..., ohat: ..., shaker: ...}}}
chords: {numerals: [...], qualities: [...]}; voicings: [rootless, shell, spread, quartal]
groove: {swing: {default: 0.55}, offset_ms: {}, humanize: off|bar|loop, drum_gain: {...}, metric_db: {...}}
sax_expr: {...}; mix: {bus_gain_db: {...}, buses: {...}, sax_warm: false, master: [...]}
bed: {src, start_s, xfade_s: 1.0, delta_lu: 24}; loudness: {lufs: -14, tol: 0.1, iters: 4, ceiling_dbtp: -1}
```
Each kit carries its paired bass, because the two are auditioned together. Names the model sees are plain words.

**Shootouts.** Run by `jobs/lofi-shootout.sbatch` into `inbox/`; best and worst of 4, with opaque clip ids.

| shootout | candidates (license) | variants | clips |
|---|---|---|---|
| sax (2 hand loops, full band) | FreePats Tenor SF2 (CC0, commit 6.5 MiB); Karoryfer Weresax alto, SFZ→SF2 (CC0); MTG Solo Sax (CC-BY-4.0, attribution); GeneralUser GS 2.0.3 tenor 66 / alto 65 (permissive) | warm off/on; best source with expression off as control | 11 |
| keys (fixed groove, rootless) | jRhodes3d (CC-BY-NC-4.0, fetch-only); GU Tine EP 4; FreePats FM EP (CC0); FreePats Upright KW (CC0); VCSL Upright (CC0); FreePats FSBS Jazz Guitar #2 (CC0); Greg Sullivan Wurlitzer EP200 (CC-BY-3.0); GU Drawbar Organ 16 | top 3 re-run with spread voicing | 11 |
| drums+bass | kits: Virtuosity (CC0), Karoryfer Swirly brushes (CC0), GU Jazz kit 32 rendered to one-shots, plus VCSL shakers (CC0). Basses: VSCO2 contrabass pizz (CC0), FreePats E-bass finger (CC0), GU Acoustic Bass 32 | R1: kits × 3 patterns at 55%, no humanize. R2: 3 basses × the top 2 kits. R3: the winner × swing {50, 55, 58, hats 60 with the rest at 54} × humanize {off, H1, H2} | 9 + 6 + 12 |
| vamps (one groove, no sax) | hopeful: Aruarian IIm9 V13 /2, Royal-half IVmaj7 IIIm7 /1, Two-of-us IVmaj7 III7 /2, Lift Imaj9 IVmaj9 /2. Wistful: i–IV Im9 IV13 /2, Pulse Im9 IV9 /1, Step Im11 IIm7 /2, Float Im9 bVIImaj7 /2. Melancholy: Drift Im9 bVImaj7 /2, Plagal Im9 IVm9 /2, Backdoor IVm9 bVII13 /2, Rise bVImaj7 bVII13 /2 | all 12 pass their own mood check; the minor ii–V IIm7b5 V7b9 was excluded because it ties aeolian and ionian | 12 |
| rain | Metadex 235827 (CC0; converted 48 kHz → 44.1 kHz once; birdsong and thunder excerpted out) | Δ ∈ {none, 30, 24, 18} on one hand loop | 4 |

## 7. Layout and tests
```
src/lofi_rl/dsl/{ast,parse,repair,canon,vocab}.py        stdlib only (both envs)
src/lofi_rl/theory/{modes,chords,voicing}.py
src/lofi_rl/compile/{timeline,drums,bass,keys,sax,expression,humanize,midi,compiler}.py
src/lofi_rl/render/{fluid,sampler,mix,loudness,bed,pipeline,formats}.py   scorer env
src/lofi_rl/verify/{gates,groove,sax,variation,knobs,penalties,score}.py  score() -> every term
src/lofi_rl/{palette,prng,brief}.py ; docs/{DSL,COMPILER,RENDER,VERIFIERS,SHOOTOUTS}.md
jobs/lofi-{assets,shootout,determinism,render}.sbatch    logs: /scratch1/shresthr/lofi-rl/logs/%x_%j.out
tests/golden/{render_hashes,fingerprints}.json
```
`pytest -q` should finish in under 60 s on the laptop. Render tests use the committed GeneralUser GS and the CC0 one-shots.

| Area | Tests |
|---|---|
| DSL | `test_parse`; `test_leniency`; `test_canon` (parse then print returns the original); `test_gates` |
| Theory | `test_modes_chords`; `test_voicing` (matches the brute-force optimum; rejects low-interval violations) |
| Compiler | `test_timeline` (N % 64 == 0; swing sample positions match the formula); `test_bass_lock`; `test_sax_register`; `test_expression` (whole vibrato cycles per L; pitch bend in range); `test_prng` (test vector) |
| Render | `test_sampler` (round-robin, layers, choke); `test_render` (length N, −14 ± 0.1 LUFS, true peak ≤ −1, bed at −Δ ± 0.5 LU on its synthetic fixture; with the frozen palette's rain, placed at −Δ before the limiter, the bed as heard sits 22.64–23.93 LU under the music on the kept 10, job 13706355; docs/RENDER.md §3.4, D-022); `test_seam` (3-cycle render, cycle 2 vs cycle 3 max abs diff < 1e-6) |
| Determinism | `test_determinism` (below) |
| Verifiers and assets | `test_verifiers` (the section 5 table); `test_palette` (every source has a license, NC sources are fetch-only, SHA matches); `test_fluid_cli` (version 2.6.1, flags) |

`test_determinism` covers:
- bit identity across the same process, a fresh process, and A-B-A render order;
- golden `audio_sha256` values keyed by platform, skipped on other platforms.

Per LESSONS 38 the cluster hashes are canonical. `jobs/lofi-determinism.sbatch` writes them and runs on two main-partition CPU models, recording whether they agree. Laptop and cluster renders are compared only by spectral fingerprint, within a tolerance.

## 8. Open issues for Shresth (plain words; the design's current default is in brackets)
1. If tenor and alto both pass, should every loop use the same sax? (a) one sax for all loops, as the project's signature (b) the model picks per loop (c) tenor for sad loops, alto for hopeful ones [a]
2. Sax timing: (a) on the band's grid (b) a touch behind the beat (c) lightly swung, like a jazz soloist (d) audition all three [d]
3. How should the answer (bars 5–8) relate to the call? (a) same rhythm, new notes (b) a new idea (c) starts like the call, ends differently (d) no preference, votes decide [c]
4. Should more energy also make the sax busier? (a) no (b) a little (c) clearly [a]
5. Bass: (a) roots only (b) roots plus occasional octave jumps (c) roots plus a lead-in note before each chord change (d) try (b) and (c) in the drums+bass shootout [a]
6. One bar where something drops out (say the hats vanish in bar 4): (a) welcome as "something new" (b) only rarely (c) it breaks the groove [a]
7. Rain: (a) steady level (b) dips under the band and comes up in the gaps (c) audition both [a]
8. Going back into bar 1, the sax should: (a) breathe (b) play a short lead-in (c) either is fine [c]
9. Ear-check clips: (a) one pass (b) two passes, so you hear the seam (c) one pass plus a short seam clip (bars 7–8 into 1–2) [b]
10. Extra reason taps in instrument shootouts: (a) none (b) "plastic", "too bright" (c) "plastic", "too bright", "too dull", "too loud" [a]

### Critical Files for Implementation
- /Users/shresthrana/csci596-project/src/lofi_rl/dsl/parse.py
- /Users/shresthrana/csci596-project/src/lofi_rl/compile/compiler.py
- /Users/shresthrana/csci596-project/src/lofi_rl/render/pipeline.py
- /Users/shresthrana/csci596-project/src/lofi_rl/verify/score.py
- /Users/shresthrana/csci596-project/palettes/lofi_rain_v1.yaml