# GRAMMAR.md: grammar-constrained decoding for LoopScript

The base-model test v0 failed mostly on syntax: G1 389 of 500 (`runs/junk-13632676/summary/junk.json`). The commonest
fault was a per-bar edit given two grids, the format of the base lines, and grids of the wrong length were common too
(`pad_grid` in 316 loops, `trunc_grid` in 146). D-016 makes grammar-constrained decoding the first lever, and D-019's
outcome applies it. This file describes the one regex that does it.

- **Code:** `src/lofi_rl/grammar.py`, standard library plus `lofi_rl.dsl`, so it imports in the policy env, the scorer
  env and on the laptop.
- **Tests:** `tests/test_grammar.py` covers acceptance, rejection, a random walk and the dialect (§6). Also
  `tests/test_junk.py`, `tests/test_junk_job.py` and `tests/test_rl_train.py` for the two places it is used.
- **Spec:** `docs/DSL.md` §3 (grammar), §7 (gates) and §8 (canonical text).
- **Sound claims:** none.

## 1. Entry points

```python
from lofi_rl.grammar import loopscript_regex, regex_sha256
regex = loopscript_regex(palette)   # a Palette, or a dsl Vocab
regex_sha256(palette)               # sha256 of the regex text (UTF-8): the identity runs record
```

- **Junk rate.** `python -m lofi_rl.rl.junk generate --run DIR --constrained` sends
  `SamplingParams(..., structured_outputs=StructuredOutputsParams(regex=regex))`. v0's `generate.json` shows that
  field as `structured_outputs=None`.
  - **The import.** `vllm.sampling_params.StructuredOutputsParams` is imported inside the run, before the model
    loads. A vLLM without it stops in seconds, and `generate.json` `stopped` names the import that failed.
  - **The record.** `generate.json` and `junk.json` hold a `constrained` block: `enabled`, `regex_sha256`,
    `regex_chars` and `builder`.
  - **Unchanged:** the sampling settings (`sampling` is v0's) and the score step.
  - **The job:** `make submit JOB=jobs/junk_rate.sbatch ARGS='500 2026 1'`. The third argument, CONSTRAINED, is 0
    (the default) or 1. Anything else stops the job before any work.
- **GRPO.** `configs/grpo_smoke_constrained.yaml` is `configs/grpo_smoke.yaml` plus one top-level key,
  `structured_outputs: loopscript_regex`.
  - **Load time.** `lofi_rl.rl.config.load_config` builds the regex from the config's palette.
  - **The field.** `resolve` puts the regex in TRL's `vllm_structured_outputs_regex` for every profile. The regex is
    never written in YAML, and writing that field by hand is refused.
  - **The record.** The dry run and `run.json` give its builder, field, sha256 and length under
    `structured_outputs`.
  - **UNVERIFIED:** the field name, for TRL 1.17.1. It goes through the trainer's GRPOConfig field check before the
    model loads, so a wrong name stops the run in about 2 minutes.
  - **Not yet runnable:** `jobs/grpo_smoke.sbatch` runs only `configs/grpo_smoke.yaml`.

## 2. Identity

| palette | palette sha256 | regex characters | regex sha256 |
|---|---|---|---|
| `palettes/dev_gu.yaml` | `b2e4a340…` | 2,931 | `bc3381e8c5dddcd16c9c8fff00976f5555c0b559af355e80c932b6ef63afbded` |

- **The source:** `loopscript_regex(load_palette("palettes/dev_gu.yaml", verify=False))` on the laptop, 2026-10-31.
  The constrained dry run (`python -m lofi_rl.rl.train --config configs/grpo_smoke_constrained.yaml --dry-run`)
  reports the same sha256 and length.
- **When it changes:** the regex follows the palette's lists, so `palettes/lofi_rain_v1.yaml` will have its own sha.
  Any change to the lists or to `grammar.py` changes it too.

## 3. What the regex enforces

It matches canonical LoopScript as `dsl.canon` prints it (DSL.md §8), without the optional `sax:` line (DSL.md §4).
The skeleton, with the components of §3.1:

```
vamp: CHORD CHORD span SPAN\n
keys: KEYS VOICING\n
comp: SGRID( SGRID)?\n
bass: (lock|SGRID( SGRID)?)\n
drums: KIT\n
kick: DGRID( DGRID)?\n
(snare|rim): DGRID( DGRID)?\n
(hat: DGRID( DGRID)?\n)?(ohat: DGRID( DGRID)?\n)?(shaker: DGRID( DGRID)?\n)?
(@1 comp: SGRID\n)?(@1 bass: SGRID\n)?(@1 kick: DGRID\n)? … (@8 ohat: DGRID\n)?(@8 shaker: DGRID\n)?
A: PHRASE\n
B: PHRASE\n
```

- **Statements** come in canonical order, one per line, with single spaces and a final newline. The order is vamp,
  keys, comp, bass, drums, kick, the backbeat lane, hat, ohat, shaker, the edits, then A and B.
- **Required statements:** vamp, keys, comp, bass, drums, kick, exactly one of snare or rim, A and B are not optional.
  Canonical text never holds both a snare and a rim base, because the parser drops the second.
- **Edits.** There are 64 optional slots, one per (bar, lane), in canonical edit order: by bar, then comp, bass, kick,
  snare, rim, hat, ohat, shaker. Each slot takes exactly one grid, so the v0 fault (`@4 hat takes at most 1 grid(s)`)
  cannot be written. An edit on a lane with no base is legal, as it is for the parser.
- **Grids** are exactly 16 cells from the lane's own alphabet: drum lanes `X x g .`, and comp and bass also `-`.
- **Vocabulary.** It is what `palette.vocab()` checks, as the names canonical text prints after aliasing. The lists
  are the prompt's (`prompt.vocabulary`): every numeral × quality, in any mood.
  - The per-mood chord lists in the prompt are hints. 8 of the 98 v0 passes use a chord outside every mood's list
    (Im7b5 and bVIImaj7#11). So the regex takes the full product, as G4 does: 11 × 14 = 154 chords for dev_gu.
  - Also the keys names, voicing styles and kits, and spans 1 and 2. Aliases (`ep`, `dusty`, …) are not
    canonical, so they are not in the regex.
- **Phrase items.** Each is a duration (`s e e. q q. h h. w w.`), then a rest `r`, or a degree `1`–`7` with an
  optional `b` or `#` and up to two octave marks of one kind.
- **Bar marks.** A `|` stands only between two items, with a space on each side, at most 3 per phrase. So a phrase
  is up to 4 runs, and each run holds at most 64 items.

### 3.1 Components (dev_gu)

| name | pattern |
|---|---|
| DGRID | `(X\|x\|g\|\.){16}` |
| SGRID | `(X\|x\|g\|-\|\.){16}` |
| CHORD | `(I\|bII\|II\|bIII\|III\|IV\|V\|bVI\|VI\|bVII\|VII)(maj7\|maj9\|maj7#11\|7\|9\|13\|7sus\|7b9\|7#9\|m7\|m9\|m11\|m7b5\|dim7)` |
| SPAN | `(1\|2)` |
| KEYS, VOICING, KIT | `(organ\|rhodes)`, `(rootless\|shell\|spread\|quartal)`, `(jazz)` |
| ITEM | `(s\|e\|e\.\|q\|q\.\|h\|h\.\|w\|w\.)(r\|(b\|#)?[1-7]('\|''\|,\|,,)?)` |
| RUN | `ITEM( ITEM){0,63}` |
| PHRASE | `RUN( \| RUN){0,3}` |

`grammar.components(palette)` returns these patterns, and `loopscript_regex` is assembled from them.

### 3.2 What the parser takes and the regex does not

Everything below is leniency or a repair in the parser (DSL.md §6). Canonical text never contains it, so the model
cannot write it either:
- **Text around the loop:** prose, fences and comments.
- **Spacing:** tabs and extra spaces, blank lines, CRLF line ends, and `=` for `:`.
- **Spelling:** case-folded heads and names, lane and name aliases, quality respellings, lowercase or bare numerals,
  and a missing `span`.
- **Edit bars:** leading zeros.
- **Statements:** any statement order, duplicates, and both a snare and a rim base.
- **Grids:** 12–15 or 17–20 cells (padded or cut), and `-` in a drum grid.
- **Bar marks:** a `|` touching an item, or leading, trailing or doubled.

Two narrowings are new:
1. **No `sax:` line.** It is optional and ignored (DSL.md §4).
2. **At most two octave marks per note.** The parser takes any number. Two marks already move a note 24 semitones,
   most of the tenor's playable range (44–75, 31 semitones; palettes/dev_gu.yaml). The bound also keeps a completion
   from looping on marks.

The bar-mark and run bounds are exact for canonical text: a 64-step phrase has 3 inner bar lines, and at most 64
items.

## 4. Gate table

| gate | guaranteed by the regex | what stays with the parser |
|---|---|---|
| G1 parse | **yes** | Every token comes from a closed list or an alphabet, one statement per line (§6, the walk) |
| G2 truncation | no | `max_tokens` can still cut a completion; the longest accepted string is 5,627 characters |
| G3 missing | **yes** | The required statements are not optional |
| G4 vocabulary | **yes** | Numerals, qualities, keys, voicings and kits are the palette's resolved lists |
| G5 forbidden | **yes** | No free word: tempo, key, effect and bed words, note names and absolute chords cannot be written |
| G6 structure | partly | **Guaranteed:** exactly 2 chords, grids of 16 cells. **Left to the parser:** two distinct chords, each phrase within 16 steps of 64, a note in each phrase within its 64 steps, a kick and a backbeat that sound after the edits |
| G7 render | no | The renderer's (RENDER.md §3.5) |
| compile | no | `compile_loop` can still refuse a loop that passes the gates (none did in v0) |

- **The source of the table:** `grammar.GUARANTEED` and `grammar.NOT_GUARANTEED`, which the random walk checks.
- **Repairs.** On accepted text, the parser can log only `misplaced_bar` (free), `pad_phrase` and `trunc_phrase`
  (0.02 each). No grid, alias, duplicate, snare-and-rim, span or numeral repair is possible.
- **G2.** 64 edits and four 64-item runs per phrase make 5,627 characters, from the walk's tree in
  `tests/test_grammar.py::longest`. LoopScript runs about 2.2 characters per token (`prompt.py`), so that is far past
  512 tokens. v0's completions had a p99 of 406.1 tokens (`junk.json`).

## 5. Dialect

vLLM 0.32's structured-outputs backends are xgrammar by default, then guidance, outlines and lm-format-enforcer
(`requirements/policy.lock.txt`: xgrammar 0.2.8, llguidance 1.7.6, outlines-core 0.2.14, lm-format-enforcer 0.11.3).
The regex uses only constructs they all read:
- literals, and the escapes `\n`, `\.` and `\|`;
- one class with a range, `[1-7]`;
- plain groups `( )` and alternation `|`;
- the greedy, bounded quantifiers `?`, `{16}`, `{0,3}` and `{0,63}`.

It has no lookaround, backreference, lazy or possessive quantifier, named or non-capturing group, inline flag,
anchor, wildcard, negated class or class escape (`\d`, `\w`, `\s`). It also has no unbounded `*` or `+`, so a
completion cannot run on inside it.
- **The laptop check:** `tests/test_grammar.py::Reader` parses only this dialect and refuses each of those
  constructs, which a test lists.
- **UNVERIFIED until the cluster:** no backend is installed on the laptop, so xgrammar's own compile of the regex is
  checked first by the constrained junk-rate run.

## 6. Evidence (`tests/test_grammar.py`, 2026-10-31)

- **Acceptance.** `re.fullmatch` takes the canonical text, without `sax:`, of:
  - the 6 gate-passing fixtures (of 12 in `tests/fixtures/loops`);
  - the 9 shootout loops;
  - all 98 passing v0 completions, canonicalized with `dsl.canon`.
- **Rejection.**
  - **v0's G1 completions:** all 389 are rejected in the parser's own normalized form (`dsl.repair.normalize`, with
    `sax:` lines dropped). Each still parses as G1 in that form, so the fault survives normalization.
  - **v0's failure modes, applied to the canonical example:**
    - an edit with two grids (kick, hat, comp, rim);
    - grids of 12, 13, 14, 15, 17, 20, 21 and 32 cells, on base lanes and edits;
    - unknown, aliased or non-canonical chords and names;
    - each of the 9 required statements missing.
  - **42 more non-canonical forms**, from a `sax:` line to a misplaced `|`.
- **Random walk.**
  - **The sample:** 600 strings, all distinct, from seed 596. Each is a uniform branch at every alternation and a
    seeded count at every repeat, over the tree `Reader` parses from the regex itself.
  - **G1–G5:** none of the 600.
  - **G6:** 591. 548 have a phrase more than 16 steps off 64, 38 a phrase with no note in its 64 steps, and 5 two equal
    vamp chords.
  - **Passing:** 9 pass every gate, and all 9 compile (`compile_loop`, brief mid / wistful / 82 BPM / D).
  - **Repairs:** `misplaced_bar` 115, `pad_phrase` 93, `trunc_phrase` 75, and nothing else.

## 7. Open

- **Phrase length.** This is the G6 a regex cannot count without blowing up: the durations fill one 16-step bar in
  24,098 orders.
  - **An EBNF grammar** (xgrammar's) could count steps with about 17 rules per bar.
  - **But GRPO takes a regex:** the TRL option this project found is `vllm_structured_outputs_regex`
    (`docs/research/taste_scorers_rl.md`), so the regex is the lever both paths share.
- **The IS-ratio caveat.** `docs/design/taste_pipeline.md` §6 gates the lever for GRPO:
  - on only if step-0 parse failures exceed 20 %;
  - off if the mean |log IS ratio| exceeds 0.05 (UNVERIFIED).
- **Near-copies.** The in-prompt example is in the regex's language, so constrained decoding does nothing about
  copying it. That is D-019's second cause.
