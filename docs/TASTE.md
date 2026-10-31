# TASTE.md: Shresth's taste profile (the spec)

These are Shresth's answers from the 27-round multiple-choice interview on **2026-10-30**. He is a
listener, not a producer, so the questions were worded as listening experiences. Every design choice
in this repo traces to a row here.

- **Claude's defaults.** Rows marked "Claude's default" are Claude's call where he answered
  "I don't know", or where the research forced a detail he didn't pick. He can override any of them.
- **Changes.** Edit this file only with his words, and add the date.

Rules for using this file:
- **"Shootout prunes, votes decide."** When a choice can't be named in words, a blind shootout cuts
  it to 2–3 options and his votes settle the rest.
- **No sound claims.** Claude never states how a clip sounds; it queues the clip for his ear (CLAUDE.md §6).
- **Ear approvals** are lines of the form `APPROVED by Shresth Rana <yyyy-mm-dd>`.

**Sound**

| | |
|---|---|
| Lineage | Jazzhop (Nujabes/Dilla). The reference is "San Francisco \| Jazzhop Mix 🎷 Chill LoFi & Jazzy Hip Hop Beats" (Fantastic Music). Learn its horn lines, drum pocket and keys/voicings; its dust is not a target. |
| Job | Head-nod: drums and bass carry it. Success means "I'd play it while working". |
| Listener | No theory or production. Taste comes mostly from votes; tags and questions use plain words. |
| Tempo / feel | Set per brief within 60–95 BPM, centered on 75–90. The compiler applies light swing (about 55% MPC). |
| Harmony | Every chord at least a 7th; 9ths common. A 2-chord vamp. Borrowed chords are allowed (Claude's default); the vamp shootout can remove them. |
| Mix | Clean and modern, with no wobble, crackle or bitcrush. Rain barely there. Loops are seamless, because he plays a few repeats, then the next. |
| Great / dealbreakers | Great: can't stop nodding, a sax line he'd replay, chords that give a feeling. Dealbreakers: fake/cheap, no groove, sour notes, boring. |

**Lead and band**

| | |
|---|---|
| Sax | **One sax** (the shootout winner) plays in every loop; it's the signature. A few long notes. Phrase in bars 1–4, answer in bars 5–8; each lands on a downbeat, and the answer hangs open. Up front like a vocal. Blue notes are welcome; long-note color is left to votes. |
| Sax realism | Compiler expression: scoops, vibrato on long notes, breath swells. If it still reads fake, warm the sax only. Fallback lead: Rhodes. No purchases. |
| Sax guardrail | Soft penalties on busy, wide or chromatic lines that **never fade**. |
| Drums | He couldn't describe a preference, so a drums+bass shootout prunes kits and patterns to 2–3, and votes decide. No fill at the seam: keep the groove. |
| Energy | More energy means busier drums, harder accents, busier keys and a busier bass rhythm. |
| Bass | Holds the roots. Its sound comes from the drums+bass shootout. |
| Keys | Shootout among Rhodes, piano, jazz guitar and Wurli/organ, keeping 2–3. What the keys do under a held sax note is left to votes. |
| Humanize | Decided by the shootout. |

**Variation and briefs**

| | |
|---|---|
| Variation | Something new every bar. One signature sound. Loops from the same brief may be close cousins; different briefs should give different songs. |
| Briefs | A scene plus knobs. The only scene is city at night. Knobs: energy (3 levels) and mood (3 levels: melancholy = aeolian, wistful = dorian, hopeful = ionian). Mood picks the mode. |

**Feedback process**

| | |
|---|---|
| Unnameables | A shootout prunes each to 2–3 options; votes decide among them. |
| Vote unit | **Best and worst of 4** clips from one brief. Clips auto-play in sequence and he can pick early at any time; pairs with unheard clips are dropped. Escape hatches: "no clear best/worst" and "all four bad". One optional tag: fake, no groove, sour, boring / nodding, sax line, chords. The page shows only the scene. RM predictions are never shown while voting. |
| Listening | 1 h a day in micro-sessions while working, on the laptop with AirPods. Ear checks come back the same day, and push notifications announce them. |
| Tiers | Toss = a dealbreaker fired · Fine = not embarrassing · Good = I'd play it while working · Great = can't stop nodding. |
| Labeling | Triage covers a random 300 seeds. Claude drafts 30 loops and he keeps 10; the rejects become labeled negatives. A vamp shootout of about 12 vamps takes about 15 min. |
| Playlist mode | A background mode plays loops 2–4 times each; he taps ♥ or ✗ only when something stands out. |
| Hidden controls | About 10% of rounds are repeats, which measure his self-consistency; about 5% are broken anchors. A session that misses its anchors counts at half weight (Claude's default). |
| Drift and storage | All votes weigh the same. Votes are private: never in git, backed up to the cluster KEEP dir and to the private `slipstr34m/lofi-rl-votes` repo. |

**Evaluation (pre-registered)**

| | |
|---|---|
| RM gate | Held-out agreement must reach at least 0.9 × his self-consistency ceiling q, measured from hidden repeats. Absolute agreement is reported against 70%. The gate runs Nov 15 on whatever votes exist and is re-checked nightly through Nov 17. |
| RM refresh | Frozen within a run and refreshed between runs, with a daily hack-watch. |
| No-collapse | Measured across briefs: step N must keep at least 70% of step 0's distance. Same-brief cousins are fine. |
| Final A/B | A dedicated blind sitting Nov 30 – Dec 2: 100 held-out briefs, **forced choice**. Step N is the checkpoint his votes favor on separate dev briefs, fixed before Nov 30. |
| Secondary | "Would play while working": playlist ♥ rate for step N vs step 0, interleaved blind. |
| Ablations | Pairwise vs absolute labels; curriculum on/off; "taste RM on vs off", which replaces "palette vs free-form" (Claude's default). |
| GPU scaling | Whichever configuration is faster and has the shorter queue at the time. |

**Process**

| | |
|---|---|
| Seeds | Written by a Claude Code workflow, with no API key. Half the writers see TASTE.md and half don't. |
| Policy | About 4B parameters. If syntax failures sink the Phase 1 gate, constrained decoding is the first lever. |
| Assets | The repo is public: commit only CC0/PD/permissive/CC-BY assets, with an ATTRIBUTION file. Non-commercial assets (e.g. jRhodes3d) are fetched, never committed. The palette never changes after the freeze. |
| Git | Claude may push after `pytest -q`, with no AI co-author trailer. Repos: public `slipstr34m/lofi-rl` and private `slipstr34m/lofi-rl-votes`. |
| SU | Job types are approved once. Claude registers types whose max is ≤ 500 SU itself; Shresth approves larger ones once. |
| Autonomy | Keep running on its own via a scheduled local session. Shresth says ssh needs no Duo; Phase 0 verifies this, since the Oct 30 test failed at DNS. |
| Calendar | Shresth is around through Thanksgiving. He books the Nakano meeting for about Nov 19. |

## Where each bias lives in the system

| Layer | Mechanism | Biases encoded |
|---|---|---|
| **Fixed by construction** | DSL grammar, palette, compiler | 4/4 time and 8 bars. A 2-chord vamp. A 7th-floor chord vocabulary. Bass = chord roots; the model writes only its rhythm. One sax line, phrase plus answer. Blue-note accidentals. Swing from the palette. Deterministic sax expression. A clean mix with the sax up front, rain at −Δ LU, −14 LUFS, and a seamless render. Tempo, tonic and mode come from the brief. Instruments are limited to shootout survivors. |
| **Guided, then faded** (theory weight 0.5→0.1 over 30 steps) | soft verifiers | Backbeat on 2 and 4. Kick on beat 1. Kick–bass lock. Something new every bar. No fill at the seam. Phrases land on downbeats; the answer is open. Non-blue chromatic notes are short and resolve by step. |
| **Never fades** | P_sax | Notes per phrase, range and chromatic share, each kept within bands taken from his Good-or-better loops. |
| **Obeyed** (full weight) | knob verifiers | Energy band and mood signature, which back the 90% obedience claim. |
| **Hard gates** (reward × 0) | parser, compiler, renderer | It parses, uses vocabulary only, is not truncated, and contains every required part. Minimum content. A render in spec. |
| **Learned from his ear** | shootouts → palette; votes, triage and ♥/✗ → taste RM | Kits, pocket, keys instrument and voicing, sax source and warmth, vamps, what the keys do under the sax, long-note color, borrowed chords, everything else. |
| **Generic, small** | Audiobox and CLAP | Audiobox: PQ as a junk floor; CE only if it predicts his votes; never PC. CLAP: city-at-night prompts scored against contrast scenes, on the music-only stem. |

## Claude's defaults (all overridable)

- **Drums:** 16-step grids with X/x/g levels, base plus per-bar edits, no nudges.
- **Instruments:**
  - the model picks the keys voicing style;
  - free sax notes are mode tones, blue notes and the current chord's tones; a held blue note is never penalized.
- **Brief and arrangement:**
  - 12 tonics, uniform;
  - Δ = 24 LU;
  - swing 0.55;
  - the answer "starts like the call, ends differently";
  - drop-outs are welcome mid-loop but not in bar 8.
- **Listening:**
  - ear checks play two passes, so the seam is audible;
  - votes are a single pass;
  - "heard" means 4 bars or more.
- **If the RM gate still fails on Nov 17:** run with the taste weight capped at 0.3, labeled exploratory, and keep voting.
- **CLAP contrast scenes:** sunny morning, beach, forest, café afternoon, snowy cabin.
- **Alternate base model:** Granite 4.1 8B, tested only at the gate.
- **Ablation swap:** taste RM on vs off.
- **Process:** at most 2 pushes a day; a 16-clip hack-watch.
