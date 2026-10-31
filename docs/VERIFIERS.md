# VERIFIERS.md: the symbolic reward terms

`lofi_rl.verify` scores one LoopScript completion with the symbolic terms of `docs/design/instrument.md` §5, plus
the penalties `docs/design/taste_pipeline.md` §5 defines. Everything comes from the parsed `Loop` and the
`CompiledLoop`; nothing is rendered.

- **Spec:** instrument.md §5 (verifiers, penalties, fixtures) and §0 (free notes), taste_pipeline.md §5
  (penalties), `docs/TASTE.md` (the layers: "Guided, then faded", "Never fades", "Obeyed", "Hard gates").
- **Code:** `src/lofi_rl/verify/`, standard library plus the palette's PyYAML, so it imports in the policy env
  (`tests/test_verify_policy_env.py`).
  - `scoring.py`: `Score`, `score`, `group_penalties`;
  - `view.py`: the shared symbolic reading of a loop;
  - `groove.py`, `sax.py`, `variation.py`, `knobs.py`, `penalties.py`: the terms.
- **Sound claims:** none. A score says what a loop's text and events do, never how it sounds.

## 1. Entry point

```python
from lofi_rl.brief import Brief
from lofi_rl.palette import load_palette
from lofi_rl.verify import group_penalties, score

palette = load_palette("palettes/dev_gu.yaml")
s = score(text, brief, palette)                          # optional: finish_reason="length" (G2), n_tokens=...
extra = group_penalties(texts, [score(t, brief, palette) for t in texts])   # one prompt's group
```

`Score` is a frozen dataclass. Its dict fields are plain dicts, so `dataclasses.asdict(s)` is a JSON log row.

| field | type | meaning |
|---|---|---|
| `gate_ok` | bool | G1–G6 and the compile passed |
| `gate` | str or None | the first failing gate: `G1`…`G6`, or `G7-compile` |
| `repairs` | int | every repair the parser and the compiler logged, free aliases included |
| `p_fix` | float | 0.02 per penalized repair, capped at 0.1 (`dsl.repair.penalty`) |
| `theory` | 0–1 | clip(.3 G + .3 S + .2 f_new + .2 f_seam − P_chrom, 0, 1) |
| `theory_parts` | dict | `f_bb f_k1 f_lock f_tk groove f_long f_land f_open f_ans f_breath sax f_new f_seam p_chrom`, each 0–1 |
| `knobs` | 0–1 | (f_energy + f_mood) / 2 |
| `knob_parts` | dict | `E f_energy energy_match mood_aeolian mood_dorian mood_ionian mood_margin f_mood` |
| `p_sax` | ≥ 0 | the never-fading sax restraint, at most 0.3 |
| `penalties` | dict | `bars` (P_rep) and `len` (P_len), each ≥ 0, at most 0.3 |
| `digest` | str or None | `CompiledLoop.digest()`, None when a gate failed |

- **A failed gate** gives `gate_ok` False and 0 for every term and every penalty, `p_fix` included. Any
  combination of them then gives the reward of 0 that a failed gate sets (`docs/DSL.md` §7). `repairs` still
  counts what the parser logged.
- **`score` also attaches** the loop's §5 event set as the non-field attribute `_events`, for `group_penalties`.
  `asdict`, `==`, `repr` and pickling all leave the fields as they are; pickling keeps the attribute.
- **Cost:** 14.5 ms for `example.ls` on the laptop, almost all of it compiling (about 10 ms) and the digest (3 ms).
  A completion that fails its gate costs 0.13 ms. Measured 2026-10-31, best of 3 × 50 calls, load average 5.7.

## 2. Gates

1. `dsl.parse(text, vocab=palette.vocab(), finish_reason=..., mode=brief.mood)` runs G1–G6 (`docs/DSL.md` §7). A
   bare numeral is repaired in the brief's mode.
2. `compile_loop(loop, brief, palette)`. A `CompileError` is `G7-compile`. G7's render part (RENDER.md §3.5) is the
   renderer's.

## 3. Theory (TASTE.md: guided, then faded)

Bars b = 1…8, steps 1…16. "Non-ghost" is `X` or `x`; a strike or hit is `X`, `x` or `g`.

**Groove** G = .30 f_bb + .25 f_k1 + .25 f_lock + .20 f_tk.

| term | definition |
|---|---|
| f_bb | mean over bars of [a non-ghost snare or rim hit on step 5 and on step 13] |
| f_k1 | mean over bars of [a non-ghost kick on step 1] |
| f_lock | min(1, (\|K ∩ B\| / \|B\|) / 0.6), K and B the kick and bass strikes as (bar, step) |
| f_tk | mean over bars of [hat + shaker hits ≥ 4] |

**Sax** S = .25 f_long + .25 f_land + .20 f_open + .15 f_ans + .15 f_breath.

| term | definition |
|---|---|
| f_long | 1 if 2–4 notes of the line last 8+ steps; 0.5 if 1 or 5; else 0 |
| f_land | mean over A and B of 1 if the phrase opens on a tone of its bar's chord, 0.5 on another note, 0 on a rest |
| f_open | [the pitch class of B's last note is not the tonic] |
| f_ans | 0 if B repeats A; else 1 for 0.2 ≤ J ≤ 0.8, linear to 0 at J = 0 and J = 1 (J: Jaccard of the phrases' note onsets) |
| f_breath | mean over phrases of [0.12 ≤ rest share ≤ 0.6, and a rest of 2+ steps] |

**f_new** = (1/7) Σ_{b=2..8} [sig(b) ∉ {sig(1..b−1)}], sig = (chord, the six drum lanes, bass, comp).

**f_seam** = 0 if any of these hold, else 1:
1. bar 8 has 3 or more non-ghost snare or rim hits in steps 9–16;
2. bar 8's drums differ from the bases' bar 8 (no edits) in more than 6 cells;
3. bar 8 lacks the backbeat, or bar 1 lacks the kick on 1;
4. bar 8 has fewer hat + shaker hits than max(2, half the median of bars 1–7).

**P_chrom** = min(.3, .1 · #bad). A note is free if its pitch class is in the mode, is a blue note (3, 6 or 10
semitones above the tonic), or is a tone of the chord at its onset (instrument.md §0; TASTE.md, Claude's defaults).
A note that is not free is bad unless it lasts 2 steps or fewer and the next item is a free note 1–2 semitones away.

## 4. Knobs (TASTE.md: obeyed)

**Energy.** E = .35 min(1, H/192) + .15 n_X/n_nonghost + .25 min(1, n_keys/48) + .25 min(1, n_bass/48).
- H counts the drum hits of all six lanes over 8 bars: X and x weigh 1, g weighs 0.5.
- n_keys and n_bass count the strikes of the comp and bass grids.
- f_energy = exp(−d / 0.04), d = the distance from E to the brief's band. `energy_match` = [E's band is the
  brief's], the obedience measure.

**Mood.** score_m = Σ_chords bars_c · Σ_tones w · s_m(pc), w = 1 for the root, 3rd (the 4th of 7sus) and 7th, 0.5
otherwise; s_m = +1 in the mode, 0 in Alt_m (aeolian {11, 1}, dorian {11}, ionian {8, 10}), −1 otherwise.
f_mood = [the brief's mode wins by 2 or more]. `example.ls` scores dorian 34, aeolian 26, ionian 14.

## 5. P_sax (TASTE.md: never fades)

Three statistics of the sax line, each kept within a band. Every unit outside the band costs a slope, and each part
is capped at 0.1.

| part | statistic | band | slope | cap |
|---|---|---|---|---|
| busy | notes per phrase, summed over A and B | 0–16 | 0.0125 a note | 0.1 |
| wide | the line's range in sounding semitones | 0–19 | 0.02 a semitone | 0.1 |
| chromatic | the share of notes that are not free | 0–0.2 | 0.5 a unit | 0.1 |

**The bands are provisional.** The real ones come from Shresth's Good-or-better loops, and none are rated yet. In
the meantime, the 9 shootout reference loops play 10–14 notes a phrase over 9–17 semitones, with no note that is not
free, and every fixture but `span1_dotted.ls` stays inside the bands. That fixture was built to stress octave marks:
its A spans Ab2–G4, 28 semitones, so its wide part is capped at 0.1 (measured 2026-10-31;
`tests/test_verify_fixtures.py` pins it).

## 6. Penalties

**Per loop** (`Score.penalties`):
- **bars:** P_rep = min(0.3, 0.1 · max(0, m − 3)). m is the largest class of identical full bars, the sax as it
  sounds in each bar included. The 0.3 cap is taste_pipeline's.
- **len:** P_len = min(0.3, 0.004 · max(0, T − 360)). T is `n_tokens` when given, else ceil(len(text) / 4).

**Within one prompt's group** (`group_penalties`; taste_pipeline §5's two levels):
- **identical:** completions with the same compiled digest are one loop. The first copy is free, and each later
  copy costs 0.3.
- **near-identical:** a completion whose similarity to another distinct loop of the group is ≥ 0.875 costs 0.15.
  Both members of the pair pay, and a copy already paying 0.3 pays nothing more.
- **sim** is §5's Jaccard of event sets: drum (lane, step, level), sax (step, semitones above the tonic, duration)
  and chord (bar, chord), with steps on the 128-step clock.
- **Failed loops** cost nothing here and are no one's copy.
- **Without `_events`** (a Score rebuilt by `dataclasses.replace` or read back from a JSON log), copies are still
  found by digest, but near-identity is not judged.

## 7. Choices the spec left open (Claude's defaults; each pinned by a test)

1. **f_bb** needs both beats in the bar, with no half credit, from either lane at each beat (`snare` on 2 and
   `rim` on 4 count).
2. **f_lock**
   - Onsets are grid strikes after edits and `bass: lock`, ghosts included.
   - A hold the compiler re-strikes at a chord change is not an onset.
   - With no bass strike at all, f_lock is 0.
3. **f_tk** counts the `hat` and `shaker` lanes, ghosts included. The open hat is not counted.
4. **f_land:** a chord tone is any tone of the bar's chord, extensions included.
5. **f_breath:** a rest is a run of consecutive rest items inside the phrase, so `sr sr` is a 2-step rest.
6. **P_chrom**
   - The chord is the one at the note's onset.
   - "The next note" is the next item of the line read as a cycle, so B's last note resolves into A's first; a
     rest there means no resolution.
   - Distances are in sounding semitones, after the octave fit.
7. **f_new and P_rep**
   - Comp and bass are compared as heard: a `-` after a rest is silence, and a hold cut at a chord change is its
     re-strike (`view.heard`, which uses the compiler's own `timeline.segments`). So a bar whose written comp
     repeats an earlier bar's is still new if a chord holds into it where the earlier bar was silent.
   - §5's "5 drum lanes" are kick, the snare/rim backbeat pair, hat, open hat and shaker. All six lanes carry the
     same information.
8. **f_seam rule 2** counts a changed level (x to X) as a changed cell.
9. **Energy**
   - n_X / n_nonghost is 0 when no hit is non-ghost.
   - A band edge belongs to the band above it.
   - **Provisional bands (0.38, 0.59).** §5 takes them from the seed pool's tertiles, which do not exist yet. The 9
     shootout reference loops have three loops in each energy cell, at E 0.289–0.297 (low), 0.462–0.478 (mid) and
     0.700–0.720 (high; docs/SHOOTOUTS.md §5, reproduced by `tests/test_verify_fixtures.py`). A pool's tertiles
     would fall in the two gaps between the cells, and each edge is its gap's midpoint (0.3795 and 0.589), rounded.
   - Replace the bands with the frozen tertiles (`config/energy_bands.yaml`, plus a DECISIONS.md entry) once the
     pool exists.
10. **knobs** = the mean of f_energy and f_mood, since §5 gives no combination.
11. **P_sax:** its three parts, bands, slopes and caps (§5 above) are all provisional.
12. **P_len**
    - The 4-characters-a-token estimate is as the brief asked.
    - An optional `n_tokens` argument takes the trainer's real count.
13. **Group penalties**
    - Identity is the compiled digest, so whitespace, comments, statement order and free aliases do not hide a copy.
    - The first copy of a loop is free.
    - **The near-identity threshold 0.875 is provisional.** The plan's threshold is the p99.5 of same-brief Claude
      pairs, which needs the seed pool. 0.875 is where §5's ramp, 0.3 · clip((sim − 0.8)/0.15, 0, 1), reaches the
      plan's 0.15.
14. **On a failed gate**, every term and penalty is 0, `p_fix` included. `repairs` still counts the parser's log.

## 8. Spec issues found (not changed)

1. **§5's chromatic fixture contradicts §0 and TASTE.md.** The fixture says "fail `h#4 q1` → P = .1". But `#4` is 6
   semitones above the tonic in every mood, the blue ♭5, so it is free (§0 amendment 1), and TASTE.md says "a held
   blue note is never penalized". The tests use `#1` instead, which is not free over this vamp in dorian, and pin
   `h#4` at 0.
2. **§5's "1-bar bases, no edits → f_new = 0" cannot occur** (CHANGELOG 2026-10-30). The signature holds the chord,
   and the vamp has two distinct chords, so f_new ≥ 1/7, and that loop scores exactly 1/7.
3. **§5's duplicate fixtures assume its ramp.**
   - Under the plan's two levels, one changed hat cell is near-identical (0.15), not 0.3.
   - "3 changed edits → 0" needs sim < 0.8 under §5's own ramp, and small edit changes cannot get there. Three
     edits of the example with 4 changed events each leave sim at 155/167 = 0.928, which §5's ramp charges 0.26.
     The test's cousin rewrites three edits more fully (sim 148/174 = 0.851), so it costs 0 under 0.875.
4. **§5's energy test bands (.30 / .50) call instrument.md §1's mid example high** (E 0.503). §5's fixture counts
   ("2 bass hits", "6 stabs", "6 bass hits") are read per bar. Read per loop, the high fixture would score 0.474,
   which is mid.
5. **P_len cannot fire on the estimate.** LoopScript tokenizes densely: §1's example is 268 Qwen3 tokens in 584
   characters, 2.18 a token. At 4 characters a token, P_len starts at 1,441 characters, about 660 real tokens,
   which is past `max_completion_length` (448–512). Truncated completions are G2 anyway. The reward should pass
   `n_tokens`.
6. **P_rep is 0 or 0.1 in practice.** Each chord fills 4 bars, so m ≤ 4, and the 0.3 cap never binds.
7. **§5's event sets leave out comp and bass.** Two loops that differ only there have sim 1 and are judged
   near-identical, not identical (their digests differ).
8. **The rule "f_ans = 0 if B repeats A" is implied by J.** An exact repeat has J = 1, which scores 0 anyway.
9. **A gate-passing loop can score below a failed one.** The penalties can add up past the positive terms
   (P_sax 0.3 + P_rep 0.1 + P_len 0.3 + P_fix 0.1 + group 0.3, against at most 1). The rule that a failed gate
   gives 0 then ranks some junk above a valid loop. How the terms combine is the reward's choice (`lofi_rl.rl.reward`).

## 9. Tests

| file | what it checks |
|---|---|
| `tests/test_verify.py` | Every term on hand-built loops: one that scores 1, one that scores 0 (or the lowest a term can reach), and the edges. Also the gates (the G1–G6 fixtures, G2, G7-compile, the mood's bare numerals), the interface, theory's combination, energy (the §5 fixtures, strict rise, the bands' edges), mood (hand-worked score tables), P_sax, P_rep, P_len, group penalties, a 30-case junk fuzz, pickling, and determinism across processes. |
| `tests/test_verify_fixtures.py` | A score table for the 6 gate-passing fixtures, derived by hand. One mutation of `example.ls` per theory term, which moves that term and no other. The knob and penalty mutations. The 9 shootout reference loops against SHOOTOUTS.md §5. The provenance of the energy bands. |
| `tests/test_verify_policy_env.py` | A fresh interpreter blocks mido, soundfile, scipy and pedalboard (absent from `requirements/policy.lock.txt`), then imports `lofi_rl.{dsl, compile, compile.compiler, theory, brief, palette, prompt, verify, rl.reward}` and every submodule of dsl, theory, compile and verify. Only compile's `midi`, `stub` and `stub_from_loop` may fail, and they fail on mido. A module that is not built yet, or that needs a policy-env package the laptop lacks (torch, TRL, vLLM), is skipped. Once PyYAML is loaded, verify imports nothing else outside the standard library. |

On 2026-10-31 the three files held 167 passing tests and 2 skips (`lofi_rl.prompt` and `lofi_rl.rl.reward`, built in
parallel). They ran in 2.32 s on the laptop, and in 2.1–4.1 s across the day's runs while other builders loaded the
machine (load average up to 7.6). The full suite then gave 1385 passed, 2 skipped, 45 deselected in 72.62 s.

Two more checks ran from scratch scripts and are not in the suite:
- **Mutation check.** A run of 60 source mutations killed 58. The other 2 are equivalent: deleting the exact-repeat
  rule (issue 8), and dropping "the next item is not a rest" (a rest's pitch is 0, never 1–2 semitones from a note).
- **Fuzz.** A fuzz of 2,000 mutated completions raised nothing, kept every value in range, and scored each the
  same twice (438 passed the gates).
