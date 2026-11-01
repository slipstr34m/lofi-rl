# GRAMMAR.md: grammar-constrained decoding for LoopScript

The base-model test v0 failed mostly on syntax: G1 389 of 500 (`runs/junk-13632676/summary/junk.json`). The commonest
fault was a per-bar edit given two grids, the format of the base lines, and grids of the wrong length were common too
(`pad_grid` in 316 loops, `trunc_grid` in 146). D-016 makes grammar-constrained decoding the first lever, and D-019's
outcome applies it. This file describes the one regex that does it.

- **Code:** `src/lofi_rl/grammar.py`, standard library plus `lofi_rl.dsl`, so it imports in the policy env, the scorer
  env and on the laptop.
- **Tests:** `tests/test_grammar.py` covers acceptance, line ends, rejection, a random walk and the dialect (§6). Also
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
  - **The backend.** `LLM()` is asked for `structured_outputs_config={"backend": "xgrammar"}`
    (`junk.STRUCTURED_BACKEND`), so a regex xgrammar refuses stops the run with the error, where vLLM's default
    `auto` would fall back to guidance without a word. If `LLM()` raises TypeError on that keyword (a vLLM without
    it), it is built again without it, and the record says `pin_accepted: false` and keeps the error. An
    unconstrained run asks for nothing.
  - **The record.** `generate.json` and `junk.json` hold a `constrained` block: `enabled`, `regex_sha256`,
    `regex_chars`, `builder` and `backend`. `backend` has:
    - `requested` and `pin_accepted`;
    - `backend`, read from the built `LLM` through `junk.BACKEND_PATHS` (VllmConfig's
      `structured_outputs_config.backend` first, then the older `decoding_config` names), or `unknown` when none
      resolves;
    - `source`, the path it came from;
    - `applied`, whether that backend is the one asked for (null when nothing could be read). A vLLM that takes
      the keyword but keeps `auto` shows `pin_accepted: true, applied: false`.
  - **In the language.** `score` adds `in_language` to `junk.json`'s block. It rebuilds the regex from the palette,
    checks its sha256 against the recorded one, and counts the completions that fullmatch it, apart for
    `finish_reason` `stop`, `length` and anything else. Each count has `n`, `in_language` and `out_of_language`.
    - Under a working constraint every `stop` completion is in the language. One out of it means the engine did
      not apply the regex.
    - A `length` completion is cut, so it is counted apart.
    - If the sha256 differs (another palette, or a changed `grammar.py`), nothing is counted and
      `matches_generate` is false. The headline line prints the counts.
  - **UNVERIFIED:** vLLM 0.32.0 is not on the laptop. Whether its `LLM()` takes `structured_outputs_config` and
    where it keeps the backend are first checked by the constrained run's own record.
  - **Unchanged:** the sampling settings (`sampling` is v0's) and the gates.
  - **The job:** `make submit JOB=jobs/junk_rate.sbatch ARGS='500 2026 1'`. The third argument, CONSTRAINED, is 0
    (the default) or 1. Anything else stops the job before any work. A fourth, PROMPT, is v1 (the default) or v2
    (`lofi_rl.prompt_v2`): `ARGS='500 2026 1 v2'` is the constrained run on prompt v2.
- **GRPO.** `configs/grpo_smoke_constrained.yaml` is `configs/grpo_smoke.yaml` plus one top-level key,
  `structured_outputs: loopscript_regex`. `configs/grpo_smoke_v2_grammar.yaml` adds `prompt: v2` and the copy
  penalty too: D-020's run-1 settings (docs/GRPO.md §3).
  - **Load time.** `lofi_rl.rl.config.load_config` builds the regex from the config's palette.
  - **The field.** `resolve` puts the regex in TRL's `vllm_structured_outputs_regex` for every profile. The regex is
    never written in YAML, and writing that field by hand is refused.
  - **The field exists** in TRL 1.17.1's GRPOConfig: `runs/grpo-smoke-13632679/summary/a/run.json` lists it in
    `grpo_config` (as null, in that unconstrained run). The trainer's field check still runs before the model loads.
  - **The backend, as junk does it.** In colocate mode TRL builds the vLLM engine, so the trainer wraps
    `vllm.LLM.__init__` for the launch (`lofi_rl.rl.trl_probe.BackendPin`).
    - **The pin:** every LLM built asks for `structured_outputs_config={"backend": "xgrammar"}`; a TypeError on the
      keyword builds it again without it.
    - **The read-back:** the engine is found on the trainer and read through `junk.read_backend`.
    - **The record:** run.json `structured_outputs.backend` has `requested`, `pin_accepted`, `backend`, `source`,
      `applied` and `engine`, as junk.json's block does. It decides nothing.
  - **In the language.** The reward counts, per step and by finish reason (`stop`, `length`, `other`, read from
    the token ids), the completions that fullmatch the regex under `re`. `smoke_check` fails the smoke if any `stop`
    completion is out of it (docs/GRPO.md §2, §5).
  - **The record.** The dry run and `run.json` give its builder, field, sha256 and length under
    `structured_outputs`.
  - **The job:** `make submit JOB=jobs/grpo_smoke.sbatch ARGS='configs/grpo_smoke_v2_grammar.yaml'` (or
    `configs/grpo_smoke_constrained.yaml`). A bare submit still runs `configs/grpo_smoke.yaml`.

## 2. Identity

| palette | palette sha256 | regex characters | regex sha256 |
|---|---|---|---|
| `palettes/dev_gu.yaml` | `b2e4a340…` | 3,395 | `1412f11f9ebdc520b50c263eda14be12829951357c17f43496ab2ddf5235e64e` |

- **The source:** `loopscript_regex(load_palette("palettes/dev_gu.yaml", verify=False))` on the laptop, 2026-10-31.
  The constrained dry run (`python -m lofi_rl.rl.train --config configs/grpo_smoke_constrained.yaml --dry-run`)
  reports the same sha256 and length.
- **Superseded:** the first build, 2,931 characters with sha256 `bc3381e8…`, took only bare newlines and a final
  one. The line ends of §3.3 replaced it the same day. No run had used it: the constrained base-model test is still
  to come (CHANGELOG, 2026-10-31).
- **When it changes:** the regex follows the palette's lists, so `palettes/lofi_rain_v1.yaml` will have its own sha.
  Any change to the lists or to `grammar.py` changes it too.

## 3. What the regex enforces

It matches canonical LoopScript as `dsl.canon` prints it (DSL.md §8), without the optional `sax:` line (DSL.md §4),
in the line-end layouts the base model writes and the parser reads silently (§3.3). The skeleton, with the components
of §3.1 (`<EOL>` is ` {0,2}\n`, up to two spaces and then the newline):

```
vamp: CHORD CHORD span SPAN<EOL>
keys: KEYS VOICING<EOL>
comp: SGRID( SGRID)?<EOL>
bass: (lock|SGRID( SGRID)?)<EOL>
drums: KIT<EOL>
kick: DGRID( DGRID)?<EOL>
(snare|rim): DGRID( DGRID)?<EOL>
(hat: DGRID( DGRID)?<EOL>)?(ohat: DGRID( DGRID)?<EOL>)?(shaker: DGRID( DGRID)?<EOL>)?
(@1 comp: SGRID<EOL>)?(@1 bass: SGRID<EOL>)?(@1 kick: DGRID<EOL>)? … (@8 shaker: DGRID<EOL>)?
(\n)?A: PHRASE<EOL>
B: PHRASE(<EOL>)?
```

- **Statements** come in canonical order, one per line, with single spaces between tokens. The order is vamp, keys,
  comp, bass, drums, kick, the backbeat lane, hat, ohat, shaker, the edits, then A and B.
- **Line ends** (§3.3): up to two spaces before each newline, one empty line allowed right before `A:`, and B's line
  end optional, so the model can stop right after B's last item. Canonical text, with bare newlines and a final one,
  is in the language too.
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
| EOL | ` {0,2}\n` (`grammar.EOL`), after every statement but B |
| the empty line | `(\n)?` (`grammar.BLANK_BEFORE_A`), right before `A:` |
| B's line end | `( {0,2}\n)?` (`grammar.LAST_EOL`) |

`grammar.components(palette)` returns the first ten patterns, `loopscript_regex` is assembled from them, and the
three line ends are module constants.

### 3.2 What the parser takes and the regex does not

Everything below is leniency or a repair in the parser (DSL.md §6). Canonical text never contains it, so the model
cannot write it either:
- **Text around the loop:** prose, fences and comments.
- **Spacing:** tabs; extra spaces between tokens or at a line's start; three or more spaces before a newline, or any
  after B with no newline; blank lines other than one empty line before `A:` (after B too); CRLF line ends; and `=`
  for `:`. The line ends the regex does take are in §3.3.
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

### 3.3 Line ends (review finding F1)

**What v0 writes.** These counts are over all 500 completions in `runs/junk-13632676/summary/completions.jsonl`,
measured on the laptop on 2026-10-31.

| whitespace | v0 |
|---|---|
| two spaces before the newline (Qwen3 token 2303 is `"  \n"`) | 8,062 of the 8,099 newlines, on every kind of statement, in all 500 completions |
| an empty line | the other 37 newlines, always right before `A:` (25 after an `@N bass` edit, 12 after an `@N comp`); one is in a passing completion |
| the end of the text | none of the 500 ends with a newline or a space. Each stops right after its last item: 498 on B with `stop`, and two cut by `length` (one on B, one on A) |
| tabs, CR, other whitespace, leading spaces, two spaces inside a line | none |
| the head's colon | all 8,562 heads are written `head: ` |

So v0 has two layouts. 463 completions end every line with two spaces and stop right after their last item; the
other 37 do the same with an empty line before A.

**Why it mattered.** The first build took only `\n` and needed a final newline.
- At every line end the model's top token, `"  \n"`, was masked.
- At the ends of A and B, and after a one-grid base lane, the tokens left included `" "` and ones that start
  another item or a second grid, so some of that token's probability could go to lengthening the phrase or the
  lane.
- At the end of B, EOS was refused.

**The fix.** The parser drops trailing spaces, empty lines and a missing final newline without a repair, and the AST
is the canonical text's (§6). So the regex now takes:
- up to two spaces before every newline (`EOL`);
- one empty line right before A (`BLANK_BEFORE_A`);
- B's line end, or none (`LAST_EOL`), so EOS is allowed right after B's last item.

The bounds are v0's. The regex refuses what v0 never wrote: three spaces or a tab before a newline, spaces after B
with no newline, an empty line anywhere else or two in a row, a line of spaces, leading or doubled spaces, and CRLF.

**First refusals.** Each of v0's 500 completions, as written, walked character by character through an NFA of each
regex (built independently of `grammar.py` and of the test suite's reader):

| regex | first refused at whitespace | first refused at content |
|---|---|---|
| first build (`bc3381e8`) | 452, all at line 1's two trailing spaces (all 98 passes among them) | 48, all on line 1: 45 a third vamp chord where `span` goes, 3 a word outside the lists |
| now (`1412f11f`) | **0** | 500 (below) |

Where the 500 are first refused now:
- **Grid lengths, 352:** comp grids over 16 cells (140), `@N ohat` under 16 (93), `@N hat` under 16 (54), comp under
  16 (16), `@N rim` under 16 (15), and 34 more on other lanes.
- **A second grid on an edit, or a third on a base lane, 95:** `@N bass` 47, `@N hat` 41, comp 5, `@N rim` 2.
- **The vamp, 48:** the same 48 as before.
- **A rim base after a snare base, 5.**

The 98 passes are now first refused at: a 15-cell `@N ohat` (81), a comp grid over 16 cells (10), a short `@N hat`
(4), a short snare grid (2) and both backbeat bases (1). The parser repairs each of these (`pad_grid`, `trunc_grid`,
`snare_and_rim`); under the regex the grids must be written at 16 cells, and only one backbeat base. None of the 500
raw texts is in the language, as none was in the first build's.

**Tokens allowed.** These come from the Qwen3 BPE: `tokenizer.json` of the cached `mlx-community/Qwen3-4B-4bit`.
Re-encoding v0's texts with it gives vLLM's own token count for 496 of the 500 (EOS aside), and the 8,062 line ends
encode as 8,025 tokens 2303 plus the 37 tokens 18611 (`"  

"`) of the empty lines. Each token is walked through the
regex from that point of the in-prompt example.

| where | first build | now |
|---|---|---|
| the end of A (after `hr`) | 17, EOS refused: `\n`, `" "`, and 15 that start another item (`" e"`, `" h"`, `" q"`, `" s"`, `" w"`, `" eb"`, `" er"`, `" hb"`, `" hr"`, `" qb"`, `" qr"`, `" sb"`, `" sr"`, `" wb"`, `" wr"`) | 20, EOS refused: those 17, plus `" \n"` (715), `"  "` (256) and `"  \n"` (2303) |
| the end of B | the same 17, EOS refused | the same 20, **EOS allowed** |
| after a one-grid base lane (kick) | 22: `\n`, `" "`, and 20 that start a second grid (`" ."`, `" X"`, `" g"`, `" x"`, … `" ................"`) | 25: those 22, plus 715, 256 and 2303 |
| after the last edit, before A | 1: `\n` | 8: `\n`, `" "`, 715, 256, 2303, `"\n\n"` (271), `" \n\n"` (4710), `"  \n\n"` (18611) |

## 4. Gate table

| gate | guaranteed by the regex | what stays with the parser |
|---|---|---|
| G1 parse | **yes** | Every token comes from a closed list or an alphabet, one statement per line, and the line ends are ones the parser drops silently (§6, the walk) |
| G2 truncation | no | `max_tokens` can still cut a completion; the longest accepted string is 5,780 characters |
| G3 missing | **yes** | The required statements are not optional |
| G4 vocabulary | **yes** | Numerals, qualities, keys, voicings and kits are the palette's resolved lists |
| G5 forbidden | **yes** | No free word: tempo, key, effect and bed words, note names and absolute chords cannot be written |
| G6 structure | partly | **Guaranteed:** exactly 2 chords, grids of 16 cells. **Left to the parser:** two distinct chords, each phrase within 16 steps of 64, a note in each phrase within its 64 steps, a kick and a backbeat that sound after the edits |
| G7 render | no | The renderer's (RENDER.md §3.5) |
| compile | no | `compile_loop` can still refuse a loop that passes the gates (none did in v0) |

- **The source of the table:** `grammar.GUARANTEED` and `grammar.NOT_GUARANTEED`, which the random walk checks.
- **Repairs.** On accepted text, the parser can log only `misplaced_bar` (free), `pad_phrase` and `trunc_phrase`
  (0.02 each). No grid, alias, duplicate, snare-and-rim, span or numeral repair is possible.
- **G2.** 64 edits and four 64-item runs per phrase, with two spaces before every newline and the empty line
  before A, make 5,780 characters (5,627 without the line ends), from the walk's tree in
  `tests/test_grammar.py::longest`. LoopScript runs about 2.2 characters per token (`prompt.py`), so that is far past
  512 tokens. v0's completions had a p99 of 406.1 tokens (`junk.json`).

## 5. Dialect

vLLM 0.32's structured-outputs backends are xgrammar by default, then guidance, outlines and lm-format-enforcer
(`requirements/policy.lock.txt`: xgrammar 0.2.8, llguidance 1.7.6, outlines-core 0.2.14, lm-format-enforcer 0.11.3).
The regex uses only constructs they all read:
- literals, and the escapes `\n`, `\.` and `\|`;
- one class with a range, `[1-7]`;
- plain groups `( )` and alternation `|`;
- the greedy, bounded quantifiers `?`, `{16}`, `{0,2}` (a space before a newline), `{0,3}` and `{0,63}`.

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
- **Line ends** (§3.3).
  - **v0's passes in their own layout.** Each of the 98 canonical texts, written back with the line ends, empty
    lines and end of the completion it came from, is accepted: 97 in the plain v0 layout and 1 with the empty line
    before A. Each parses with no repair to its canonical text's AST.
  - **Every layout v0 wrote.** All 500 completions hold 2 layouts (463 and 37). Each, applied to the in-prompt
    example, is accepted and parses exactly as the example does.
  - **8 named layouts:** one trailing space, no final newline, two spaces before every newline, v0's two, an empty
    line before A, two spaces before B's newline, and two spaces after a one-grid lane. Each is accepted, parses with
    no repair, and prints back (`canon`) as its canonical layout.
  - **The fixtures and the shootout loops** are accepted in v0's two layouts too (6 of 6 and 9 of 9).
- **Rejection.**
  - **v0's G1 completions:** all 389 are rejected in the parser's own normalized form (`dsl.repair.normalize`, with
    `sax:` lines dropped). Each still parses as G1 in that form, so the fault survives normalization.
  - **v0's failure modes, applied to the canonical example:**
    - an edit with two grids (kick, hat, comp, rim);
    - grids of 12, 13, 14, 15, 17, 20, 21 and 32 cells, on base lanes and edits;
    - unknown, aliased or non-canonical chords and names;
    - each of the 9 required statements missing.
  - **48 more non-canonical forms**, from a `sax:` line to a misplaced `|`. Among them are the line ends past the
    bounds: three spaces or a tab before a newline, spaces after B with no newline, an empty line after drums, before
    B, after B or first, two empty lines before A, a line of spaces, a leading space, a double space, a tab, CRLF.
- **Random walk.**
  - **The sample:** 1,500 strings, all distinct, from seed 596. Each is a uniform branch at every alternation and a
    seeded count at every repeat, over the tree `Reader` parses from the regex itself.
  - **G1–G5:** none of the 1,500.
  - **G6:** 1,482. 1,367 have a phrase more than 16 steps off 64, 107 a phrase with no note in its 64 steps, and 8
    two equal vamp chords.
  - **Passing:** 18 pass every gate, and all 18 compile (`compile_loop`, brief mid / wistful / 82 BPM / D).
  - **Repairs:** `misplaced_bar` 264, `pad_phrase` 193, `trunc_phrase` 168, and nothing else.
  - **The line ends change nothing.** Each string parses exactly as its canonical layout does: the same gate, AST
    and repairs, line numbers aside. The walk covers the layouts: all 1,500 have spaces before some newline, 761 the
    empty line before A, and 755 no newline after B.

## 7. Open

- **Phrase length.** This is the G6 a regex cannot count without blowing up: the durations fill one 16-step bar in
  24,098 orders.
  - **An EBNF grammar** (xgrammar's) could count steps with about 17 rules per bar.
  - **But GRPO takes a regex:** the TRL option this project found is `vllm_structured_outputs_regex`
    (`docs/research/taste_scorers_rl.md`), so the regex is the lever both paths share.
- **The IS-ratio caveat.** `docs/design/taste_pipeline.md` §6 gates the lever for GRPO:
  - on only if step-0 parse failures exceed 20 %;
  - off if the mean |log IS ratio| exceeds 0.05 (UNVERIFIED).

  D-020's outcome fixes both readings for v2. The IS one is the v2 grammar smoke's `smoke.json` `is_ratio`
  (docs/GRPO.md §5). It is TRL's `sampling/sampling_logp_difference/mean`, read only when the installed TRL's
  source is checked at run time to compute mean |old_per_token_logps − sampling_per_token_logps| over
  `completion_mask`; otherwise it is unavailable. That check is TRL's half of D-020's definition. Whether vLLM's
  reported log-probs include the grammar mask is unchecked (UNVERIFIED; run.json `vllm_logprobs`), and the rule says
  so in its caveats.
- **Near-copies.** The in-prompt example is in the regex's language, so constrained decoding does nothing about
  copying it. That is D-019's second cause. Prompt v2 (`lofi_rl.prompt_v2`) is the lever for it: each brief is shown
  the example of its energy and the next mood, and junk-rate's near-copy check compares each completion with the
  example its own prompt showed. All 9 of v2's examples are in the regex's language (`tests/test_prompt_v2.py`).
- **Line ends past v0's.** The bounds of §3.3 are what v0 wrote. A model that wants three spaces before a newline,
  or an empty line elsewhere, is masked there, though `"\n"` and `"  \n"` stay open at every line end. How often
  that happens is not measured: a constrained run's completions are all in the language, so they cannot show it.
