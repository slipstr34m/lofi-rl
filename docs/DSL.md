# DSL.md: LoopScript v1

LoopScript is the text the policy writes: one 8-bar loop in about 20 lines. A line names the vamp, the keys,
the comp and bass rhythms, the drum kit and its lanes, per-bar edits, and the sax phrase and its answer. The model
writes scale degrees, step grids, chord symbols and instrument names. It never writes pitches, tempo, key, mode,
swing, effects, levels or the bed: the brief and the palette supply those.

- **This file** describes LoopScript as `src/lofi_rl/dsl` parses it today. Where the parser and
  `docs/design/instrument.md` §1 disagree, this file follows the parser, and §10 lists each difference.
- **Spec:** instrument.md §1 (grammar) and §2 (gates and leniency); `docs/TASTE.md` and the plan for `sax:` (§4).
- **Code:** `src/lofi_rl/dsl/`, standard library only, so it runs in the policy env and the scorer env:
  - `parse.py`: statements, gates G1–G6;
  - `repair.py`: normalization, the repair log, P_fix;
  - `vocab.py`: closed lists, aliases, forbidden content;
  - `ast.py`: the AST and the 8-bar expansion;
  - `canon.py`: canonical text.
- **Tests:** `tests/test_dsl_{parse,leniency,gates,canon,expand,review}.py`, plus `tests/test_compile.py` for `sax:`.
- **After parsing:** the compiler turns the loop into events (`docs/COMPILER.md`). The renderer turns those into
  audio and owns gate G7 (`docs/RENDER.md`).
- **Sound claims:** none.

## 1. Entry point

```python
from lofi_rl.dsl import canon, expand, parse
from lofi_rl.palette import load_palette

palette = load_palette("palettes/dev_gu.yaml")
res = parse(text, vocab=palette.vocab(), finish_reason=finish, mode=brief.mode)
res.gate        # Gate(ok, code, detail): code is "G1".."G6", or None when ok
res.ast         # the Loop, or None whenever the gate fails
res.repairs     # [Repair(code, line, before, after, penalized)], sorted by line
res.penalty     # P_fix of these repairs
canon(res.ast)  # canonical text (§8); expand(res.ast) gives the 8 concrete bars (§9)
```

- **`vocab`** sets the names, numerals, qualities and voicings a loop may use (§5). The default, `DEFAULT_VOCAB`,
  holds placeholder names; `render.pipeline.render` passes `palette.vocab()`.
- **`finish_reason`**: `"length"` (a truncated completion) is gate G2, before anything is parsed.
- **`mode`** is a mode or mood name (`dorian` or `wistful`). It is needed only to repair a bare numeral (§6);
  without it, a bare numeral is G4. An unknown mode raises `ValueError`, because it is the caller's error.
- **Repairs** keep the text's line numbers, counting from 1. They are returned when a gate fails too, except G2,
  which parses nothing.

## 2. An example

This is instrument.md §1's example, which is `tests/fixtures/loops/example.ls`. Its brief is wistful (D dorian),
energy mid, 82 BPM, though the loop itself names none of them.

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

- **Default vocabulary:** it parses with no repairs.
- **`dev_gu` vocabulary:** it logs one free alias, `dusty` → `jazz`.
- **Text is canonical:** `canon` prints this text back unchanged (`tests/test_dsl_canon.py`).
- **The examples below** change or add a line or two of it.

## 3. Grammar

The EBNF of the text after normalization (§6). It is instrument.md §1's, with the parser's differences marked
`(* parser: ... *)` and listed in §10.

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
edit    = "@" bar (dlane ":" dgrid | ("comp"|"bass") ":" sgrid) ;  (* replaces that lane, that bar only *)
bar     = {"0"} ("1".."8") ;                       (* parser: leading zeros are allowed *)
dgrid   = 16*("X"|"x"|"g"|".") ;                   (* accent/normal/ghost/rest *)
sgrid   = 16*("X"|"x"|"g"|"-"|".") ;               (* "-" holds the previous strike *)
sax     = "sax:" SAX ;                             (* parser: optional, and the compiler ignores it *)
phrase  = ("A"|"B") ":" {item | "|"} ;             (* A = bars 1-4, B = 5-8, 64 steps each *)
                                                   (* parser: "|" may come first; the phrase needs a note *)
item    = dur ("r" | ["b"|"#"] ("1".."7") {"'"|","}) ;   (* mode degree; ' up / , down an octave *)
dur     = ("s"|"e"|"q"|"h"|"w") ["."] ;            (* 1,2,4,8,16 sixteenths; dot x1.5; no "s." *)
comment = ("#"|"//") (SPACE|EOL) {any} ;           (* "#" inside a token is a sharp *)
```

- **Names.** KEYS, KIT and SAX come from the palette (§5).
- **Tokens** are separated by spaces. A `|` may touch the items around it (`e1|h3`).
- **Required statements (G3):** vamp, keys, comp, bass, drums, kick, A, B, and a `snare:` or `rim:` base. An edit
  does not count as the base. `sax:` is not required (§4).

## 4. Statements

| statement | form | required | notes |
|---|---|---|---|
| `vamp:` | two chords, then `span 1` or `span 2` | yes | `span 2` (the default) plays AABB AABB, `span 1` plays ABAB ABAB. The two chords must differ (G6). |
| `keys:` | instrument, voicing | yes | Exactly two words. The voicing is one of the palette's `voicings`. |
| `comp:` | 1 or 2 sgrids | yes | The keys rhythm. Every strike plays the whole voicing of its bar's chord. |
| `bass:` | `lock`, or 1 or 2 sgrids | yes | The bass rhythm; the pitch is always the chord root. `lock` follows the kick (§9). |
| `drums:` | kit | yes | Exactly one word. |
| `kick:` | 1 or 2 dgrids | yes | It must sound somewhere after edits (G6). |
| `snare:` or `rim:` | 1 or 2 dgrids | one of them | The backbeat lane. Snare and rim cannot both be silent in all 8 bars (G6). With both bases, the first wins (§6). |
| `hat:`, `ohat:`, `shaker:` | 1 or 2 dgrids | no | A lane that is never written is silent. |
| `@N lane:` | exactly 1 grid | no | Replaces that lane in bar N (1–8). Lanes: comp, bass and the six drum lanes. `@N vamp:` and `@N sax:` are G5; `@N keys:`, `@N drums:`, `@N A:` and `@N B:` are G1. |
| `sax:` | sax name | **no** | Optional and ignored (below). |
| `A:` | items, about 64 steps | yes | The phrase, bars 1–4. It needs at least one note within 64 steps (G6). |
| `B:` | items, about 64 steps | yes | The answer, bars 5–8, with the same rules. |

**`sax:` is optional and ignored.** TASTE.md (Lead): one sax, the shootout winner, plays in every loop. The
project plan (§3): "Once one sax survives, `sax:` is dropped." The compiler plays the palette's `sax_lead`
whatever the line says (`docs/COMPILER.md` §2, stage 1, and §6).
- **The gate:** a loop may leave the line out, and G3 does not ask for it (`tests/test_dsl_gates.py::test_sax_is_not_required`
  and `tests/test_compile.py::test_a_loop_without_a_sax_statement_passes_the_gates`).
- **When present,** it is still checked:
  - it must be one sax name from the palette (`sax: soprano` is G4, `sax: tenor alto` G1, `sax: 90bpm` G5);
  - it cannot be edited (`@4 sax: alto` is G5);
  - it is kept in the AST (`Loop.sax`) and in the canonical text (§8).
- **It does not reach the humanize seed either** (settled 2026-10-31, SR-7). The seed hashes the canonical text
  without its `keys:`, `drums:` and `sax:` lines (COMPILER.md §3.6 and §7). So the same loop with `sax: tenor`,
  with `sax: alto` or with no `sax:` line compiles to the same `CompiledLoop`, digest included, with humanize off,
  `bar` or `loop` (`tests/test_compile.py::test_a_loop_without_a_sax_statement_compiles_to_the_same_loop_as_with_sax_tenor`).
  Naming another keys instrument, voicing style or kit keeps the humanize nudges too.
- **The Phase-0 smoke adapter** (`compile/stub_from_loop.py`) still reads the line to pick a General MIDI program
  (tenor 66, alto 65). The Phase 1 compiler does not read it.

## 5. Vocabulary

Every list a parse checks against lives in a `Vocab`. `Palette.vocab()` builds one from the palette, so the palette
decides what a loop may name.

**Fixed by the grammar** (`dsl/vocab.py`). A palette may narrow the first three lists (`chords.numerals`,
`chords.qualities`, `voicings`), and the parser then refuses what it leaves out (G4).

| list | values |
|---|---|
| numerals | I bII II bIII III IV V bVI VI bVII VII (re the tonic, major-scale reference) |
| qualities | maj7 maj9 maj7#11 7 9 13 7sus 7b9 7#9 m7 m9 m11 m7b5 dim7 (7th floor) |
| voicings | rootless shell spread quartal |
| drum lanes | kick snare rim hat ohat shaker |
| durations | s e q h w = 1 2 4 8 16 sixteenths; a dot ×1.5: e. 3, q. 6, h. 12, w. 24; no s. |
| degrees | 1–7 of the brief's mode, with an optional `b` or `#` before the degree, and any number of `'` (up) or `,` (down) after it |

**Names, from the palette.** These are the keys instruments, the kits and the saxes, plus the palette's `aliases`.

| vocabulary | keys | kits | sax | aliases |
|---|---|---|---|---|
| `palettes/dev_gu.yaml` (the development palette) | rhodes, organ | jazz | tenor, alto (lead: tenor) | ep, wurli, piano, guitar → rhodes; dusty, brushes → jazz |
| `DEFAULT_VOCAB` (no palette) | rhodes, piano, guitar, organ, wurli, ep | dusty, brushes, jazz | tenor, alto | none |

- **Placeholders.** These are not the project's names. `palettes/lofi_rain_v1.yaml`, frozen at the end of Phase 1
  from the shootouts, will set those (instrument.md §6; docs/DECISIONS.md D-017).
- **Case.** Names are case-folded.
- **Aliases** follow §6.

**Forbidden content (G5)** is checked on a token that fails to parse:
- **Note names:**
  - an uppercase note letter with an optional accidental and octave (`D`, `Eb4`, `C-1`), alone or after a
    duration (`qD`);
  - a lowercase `a`, `c`, `d` or `f` with an octave digit (`c4`, `f#3`). Lowercase `e`, `b` and `g` would clash
    with durations, flats and ghost cells, so they never count as notes.
- **Absolute chords:** `Dm9`, `G13`, `CMaj7`, `G7sus`, `C9sus4`, `E7alt`, `C/E`, `Dm7/G`, `C6/9`, `Cm(maj7)`.
- **Forbidden words.** A token is forbidden when it is one of the words below, or when one of its runs of letters is
  a word of 3+ letters (`82bpm`, `lowpass0.3`). A trailing `:` or `=` is dropped first.
  - tempo: tempo, bpm, swing;
  - key: key, mode, scale, major, minor, and the seven mode names (ionian, dorian, phrygian, lydian, mixolydian,
    aeolian, locrian);
  - effects and levels: reverb, delay, echo, chorus, flanger, phaser, compressor, compression, limiter, eq, filter,
    lowpass, highpass, distortion, saturation, bitcrush, bitcrusher, wow, flutter, sidechain, fx, effect, effects,
    volume, gain, level, levels, db, velocity, pan;
  - the bed: bed, rain, vinyl, crackle, noise, texture, ambience, hiss, tape.
- **An edit on `vamp` or `sax`.**

## 6. Leniency and repairs

The parser accepts near-misses so that the Phase 1 gate measures structure, not spelling (instrument.md §2).

**Normalization** runs first, in this order. Only step 2 logs a repair.
1. A byte-order mark is dropped. Lines may end in LF or CRLF.
2. **Prose.** Code-fence lines are dropped, inside the loop too, and so is every line before the first keyword
   line and after the last.
   - A keyword line is an edit (`@` and digits), or a `word:` or `word=` line whose word is a statement head (lane
     aliases included) or a forbidden word.
   - The dropped text is logged once as `strip_prose`. Blank and comment lines are not prose.
   - Prose between two statements is G1 (`Here's the groove:`).
3. **Comments.** A line that starts with `#` or `//` followed by a space or the line end is dropped. The same mark
   after a space, and followed by a space or the line end, ends the line early (`hat: x.x. … # busier`). `#3 hat:`,
   `#notes` and `## Drums` are not comments: inside the loop each is G1.
4. **Whitespace and colons.**
   - Tabs and runs of spaces become one space, and blank lines go.
   - `kick = …` and `kick : …` read as `kick: …`; only the `=` right after the head changes.
   - An edit bar may be written `@03 hat:`, `@3hat:` or `@ 3 hat:`.
5. **Case.** These are case-folded: statement heads, `span`, `lock`, names, voicings and phrase items (`Q5`,
   `EB5`). Grid cells never are, since `X` and `x` differ. A chord keeps its case:
   - a lowercase numeral means minor (`ii9`);
   - in a quality only the one-letter prefix is case-sensitive: `M` major, `m` minor (`IM9` is I maj9).

**Free repairs** are logged with `penalized=False` and cost nothing.

| code | when | example (line of §2's loop → result) |
|---|---|---|
| `strip_prose` | prose outside the loop, or a code fence anywhere | `Here's a loop:` and a fence before it, a fence and `Enjoy!` after → the loop |
| `alias_lane` | `bd` `sd` `hh` `oh` `shk` for kick snare hat ohat shaker, in edits too | `bd: x......x..x..... x......x.x....x.` → `kick:` |
| `alias_dash` | `-` in a drum grid, when every other cell is legal | `kick: x------x--x----- x......x.x....x.` → `x......x..x.....` |
| `alias_quality` | a respelled quality (list below) | `vamp: Im9 IVΔ7 span 2` → `IVmaj7` |
| `alias_minor_numeral` | a lowercase numeral (minor) | `vamp: ii9 V13 span 2` → `IIm9` |
| `alias_name` | a palette alias that lands on a name of that slot | `keys: wurli rootless` → `rhodes` (dev_gu) |
| `default_span` | no `span` | `vamp: Im9 IV13` → `vamp: Im9 IV13 span 2` |
| `misplaced_bar` | a `\|` that is not on a bar line (logged only; nothing moves) | `A: q5 e6 \| e5 q3 e2 e1 h3 er e4 e5 e6 h.5 er e4 q3 q2 hr` |

- **Quality respellings:**
  - `M7` `Δ7` `Δ` → maj7; `-7` `min7` → m7; `ø` `ø7` → m7b5; `°7` → dim7; `sus` `sus4` `9sus4` `7sus4` → 7sus;
  - prefixes: `maj…` and `min…` in any case, `M…` and `Δ…` for maj, `-…` for m (`IVMaj7`, `IVM9`, `II-9`,
    `II-7b5`, `V7SUS`).
- **Palette aliases.** An alias applies only where it lands on a legal name of that slot. With
  `Vocab(name_aliases={"jazz": "guitar"})`, `keys: jazz rootless` gives guitar, but `drums: jazz` stays the jazz kit,
  and `sax: jazz` is G4.

**Penalized repairs** cost 0.02 each, and P_fix is capped at 0.1 (`dsl.repair.penalty`; `PENALTY_EACH`,
`PENALTY_CAP`).

| code | when | example (line of §2's loop → result) |
|---|---|---|
| `pad_grid` | a grid of 12–15 cells: rests pad it to 16 | `kick: x......x..x... x......x.x....x.` → `x......x..x.....` |
| `trunc_grid` | a grid of 17–20 cells: cut to 16 | `kick: x......x..x.....x. x......x.x....x.` → `x......x..x.....` |
| `pad_phrase` | a phrase 1–16 steps short: rests, longest first | `A: q5 e6 e5 q3 e2 e1 \| h3 er e4 e5 e6 \| h.5 er e4 \| q3 q2` → `… q3 q2 hr` |
| `trunc_phrase` | a phrase 1–16 steps long: cut at 64; a note across the cut is shortened to the longest duration that fits | `A: q5 e6 e5 q3 e2 e1 \| h3 er e4 e5 e6 \| h.5 er e4 \| q3 q2 hr q5` → `… q3 q2 hr` |
| `bare_numeral` | a numeral with no quality: the mode's diatonic 7th chord (needs `mode`) | `vamp: I IV span 2` in dorian → `Im7 IV7` (two repairs) |
| `case_conflict` | a lowercase numeral whose quality has no minor form: the quality wins | `vamp: Im9 ii13 span 2` → `II13` |
| `duplicate` | a statement given twice, edits included: the first wins | a second `kick:` line → dropped |
| `snare_and_rim` | base `snare:` and `rim:` both: the first base wins, and every line of the other lane goes, edits included | `snare: ....x.......x...` after §2's loop → dropped |
| `sax_fold` (compiler) | a sax note outside the playable range: folded by octaves | in `CompiledLoop.repairs`, line 0 (COMPILER.md §3.2) |

- **The snare and rim rule has a limit.** An edit on the lane that has no base is an ordinary edit:
  `@8 snare: ........XXXXXXXX` in §2's rim loop is kept, with no repair.
- **The full P_fix** is `penalty(parse.repairs + compiled.repairs)`.

## 7. Gates

A failed gate sets the reward to 0, and its code and detail are logged (instrument.md §2).

| gate | fails when | failing example (one line of §2's loop) | gate detail |
|---|---|---|---|
| G1 | a parse error after normalization | `kick: four on the floor` | `line 6: 'four' is not a grid` |
| G2 | `finish_reason == "length"` | §2's loop, cut off by the token limit | `finish_reason is 'length'` |
| G3 | a required statement is missing | no `B:` line | `missing: B` |
| G4 | a name, quality or numeral is not in the palette after aliasing | `drums: trap808` | `line 5: drum kit 'trap808' is not in the palette` |
| G5 | the failing token is forbidden content (§5); a labelled subset of G1 | `tempo: 90`, added after `drums:` | `line 6: 'tempo' is a forbidden word (tempo)` |
| G6 | structure | `vamp: Im9` | `line 1: the vamp needs 2 distinct chords, got Im9` |
| G7 | the render (RENDER.md §3.5) | no one-line example (below) | `music at -inf LUFS is below -50.0` |

**What else each gate covers:**
- **G1:**
  - a keyword without its colon (`A q5 …`, `BD x…`), in any case;
  - an unknown head (`foo: bar`);
  - a degree outside 1–7 (`q8`), degree first (`5q`), or `s.`;
  - a grid cell outside its set (`x...o…`);
  - too many grids (3 for a lane, 2 for an edit), or too many or too few words (`keys: rhodes`, `sax: tenor alto`);
  - `span 3`, or a bare `span`;
  - an edit bar outside 1–8 (`@9`, `@0`, `@٣`);
  - a malformed edit;
  - an empty text.
- **G3:** each of vamp, keys, comp, bass, drums, kick, A and B, and a snare or rim base. The detail lists everything
  missing (`missing: keys, B`).
- **G4:**
  - an unknown instrument (`keys: clavinet rootless`), voicing, kit or sax;
  - a quality outside the list (`IV11`);
  - a numeral outside the list (`#IVm7b5`);
  - a numeral that mixes cases (`Iv13`);
  - a bare numeral with no `mode`, or with no diatonic 7th chord in that mode (`bII` in dorian).
- **G6:**
  - a vamp that is not two distinct chords: one chord, three chords, `Im9 Im9`, or `IIm9 iim9`, which are equal
    after aliasing;
  - a grid of fewer than 12 or more than 20 cells (`kick: x......x. …` has 9);
  - a phrase more than 16 steps off 64 (`… h.5 er s4` is 47);
  - a phrase with no notes, or none within its 64 steps (`B: wr wr wr wr`);
  - after edits, a kick that is silent in all 8 bars, or a snare and rim that both are. A ghost is not silence.
- **G7** belongs to the renderer:
  - it fails on a render failure or NaN, music below −50 LUFS, a length other than N, a final loudness more than
    0.5 LU from −14 LUFS (in practice, not reaching ±0.1 LU in 4 iterations), or a true peak above −0.9 dBTP;
  - a loop that passes G1–G6 always has a kick, a backbeat and sax notes, so its music is never silent, and the
    other checks test the render chain, so no one-line change to §2's loop fails it;
  - the test case is an all-zero music cycle (`tests/test_render.py::test_gate_g7_on_silent_or_broken_music`).

**Precedence**, when a text has several problems:
1. G2 comes first.
2. Then, over the whole text: G5 if any failing token is forbidden content, else G1, then G4, then G3, then G6. The
   structure checks come before the silence checks.
3. Statements are parsed independently, so one line's G6 never hides another line's parse error.
4. Within a line, every token is checked, and the line reports its worst failure in the order G5, G1, G4, G6.
5. The detail names the first failure found with the winning code. Heads are checked before bodies.

## 8. Canonical text

`canon(loop)` prints a loop in one fixed form. Two texts that differ only in statement order, spacing, comments,
the case of keywords and names, or free aliases print the same. `canon_sha(loop)` is the sha256 of that text.
- **Statement order:** vamp, keys, comp, bass, drums, then the lanes in the order kick, snare, rim, hat, ohat,
  shaker.
- **Edits:** sorted by bar, then in the order comp, bass, kick, snare, rim, hat, ohat, shaker.
- **`sax:`** when the loop has one, then `A:` and `B:`.
- **Words:** `span` is always written, names are as resolved (`drums: dusty` prints as `drums: jazz` under
  `dev_gu`), and repairs are applied. One space separates tokens, with no comments.
- **Phrases:** `|` stands at every bar line between two items. The octave marks of a note print as their net count.
- **Round trip:** `parse(canon(x))` gives `x` with no repairs. §2's loop is already canonical
  (`tests/test_dsl_canon.py`).

## 9. What a loop means

The 8-bar reading that the compiler and the verifiers share is `dsl.expand(loop)`:
- **Chords per bar:**
  - `span 2` gives chord 1, 1, 2, 2, 1, 1, 2, 2;
  - `span 1` gives 1, 2, 1, 2, ….
- **Bases:** a 1-bar base repeats in every bar; a 2-bar base alternates. A lane with no base statement is rests.
- **Edits** then replace their lane's bar.
- **Holds.** In comp and bass grids, a strike (`X`, `x`, `g`) holds over the `-` cells after it, across bar lines
  and across the seam. A `-` that follows a rest is silent. At a chord change, a held note is cut and struck again
  with the new chord (COMPILER.md §3.4).
- **`bass: lock`:**
  - the onsets are the non-ghost kicks after edits (their levels copied), plus an `x` at every chord change, the
    seam into bar 1 included;
  - each note holds to the next onset, at most 8 steps;
  - `@N bass:` edits then replace their bar.
- **Phrases.** A covers steps 0–63 (bars 1–4) and B covers 64–127 (bars 5–8), 16 steps to a bar.
  - A note's pitch is the brief's tonic plus the degree in the brief's mode, plus its accidental and octave marks
    (COMPILER.md §3.2).
  - A rest is `r` after a duration (`qr`).
- **Levels.** `X`, `x` and `g` are the only levels a loop can write. The palette maps each one to a drum gain and
  to a keys and a bass velocity (COMPILER.md §3.5).

## 10. Differences from instrument.md

Where the parser and instrument.md disagree, this file follows the parser.

**§1, the grammar:**
1. **`sax:` is optional and ignored.** §1 lists it among the required statements; the parser does not require it.
   TASTE.md (one sax plays in every loop) and the plan (`sax:` is dropped once one sax survives) settle it the
   parser's way (§4; COMPILER.md §7).
2. **Edit bars.** §1 allows one digit, 1–8. The parser also takes leading zeros (`@03 hat:` is bar 3) and spaces
   around the bar number (`@ 3 hat:`, `@3hat:`).
3. **Phrases.** §1 starts a phrase with an item. The parser takes a leading `|` (`A: | q5 …`) and a `|` that touches
   an item (`e1|h3`). A phrase of bar marks only (`A: | | |`) is G6 (no notes), not G1.
4. **Names.** §1 takes KEYS, KIT and SAX from the palette lists. The parser also takes the palette's `aliases`, a
   key the implementation adds to instrument.md §6's schema (COMPILER.md §6). They land only on a legal name of
   their slot (§6 here).

**§2, gates and leniency:**
5. **More quality aliases.** Beyond §2's list, the parser accepts bare `Δ`, `sus4`, `7sus4`, and the prefix
   respellings `maj…` and `min…` in any case, plus `M…`, `Δ…` and `-…` (§6).
6. **Forbidden statements outside the loop.** §2 strips prose outside the first and last keyword lines. The parser
   counts a `word:` line with a forbidden word as a keyword line, so `tempo: 90` placed before or after the loop is
   G5, not stripped prose.
7. **The sax octave fold.** §2 lists it among the parser's repairs, but the fold needs the brief's tonic, so the
   compiler makes it (`sax_fold`, line 0, the same 0.02).

**Readings where §2 is silent** (`parse.py` docstring; pinned by the DSL tests):
- **Snare and rim:** only two base statements conflict. The first wins, and the losing lane's edits go with it. An
  edit on the lane with no base is an ordinary edit.
- **G3:** the backbeat lane needs a base statement; an edit is not enough.
- **Empty phrase:** no notes, or no notes within 64 steps after truncation.
- **Silence:** a ghost is not silence.
- **Bare numerals:** they need the brief's `mode`.
- **Bar marks:** a misplaced `|` is logged and nothing moves.
- **Precedence:** G2, G5, G1, G4, G3, G6 (§7).
