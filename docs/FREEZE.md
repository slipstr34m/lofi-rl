# FREEZE.md: the frozen palette `palettes/lofi_rain_v1.yaml`

The end of Phase 1: the procedure of docs/SHOOTOUTS.md §1 ("From votes to the freeze", step 4). Claude assembles the
palette from the shootout survivors, their trims and faders (SHOOTOUTS.md §3 "Frozen values") and the rain pick, under
the plain-word names the model sees (instrument.md §6). Shresth signs it off with `APPROVED by Shresth Rana
<yyyy-mm-dd>` in gates/GATE_palette.md; then `frozen:` takes that date, and after that **the file never changes**
(D-017, docs/DECISIONS.md).

- **Code:** `src/lofi_rl/freeze/` (`__init__` holds the constants and the kept-list readers; `palette`: build, check,
  vocab; `gate`: reading the gate file; `checks`, `signoff`, `batch`: the sign-off, §5; `seal`, `tables`), tests
  `tests/test_freeze_*.py`.
- **Files:** `palettes/lofi_rain_v1.yaml`, written by `python -m lofi_rl.freeze build`, never by hand;
  `drafts/v1/keep3_kept10.json` (§5.3).
- **Status (2026-11-03):** built with `frozen: null`. The first sign-off listen (b1103-signoff) did not pass: 4 Toss,
  and Shresth's words were "The loops themselves", so the palette does not change and briefs d01, d06, d07 and d10
  reopen through the drafts redo path (§5).
- **Sound claims:** none. Every number here is read from a repo file or from the build's output; the file names where
  it came from.

## 1. What it holds

Every block is the draft palette's (`palettes/draft_day2.yaml`, the Day-2 picks, which keep3 played; docs/DRAFTS.md §3),
except for the eight changes of §2. The picks are Shresth's confirmed results (gates/GATE_palette.md): Day 1 and Day 2
"Yes, use these"; Day 3 "Yes, use these" with "24 below (Recommended)" for the rain level; keep3, the 10 drafts he kept.

| name the model sees | instrument (shootout option) | trim_db | fader_db | rule |
|---|---|---|---|---|
| sax `alto` (`sax_lead`) | `sax_mtg_alto`, warmth off, expression on, timing lag | 6.63 | 2.55 | median F; lag keeps the instrument's (CD-26) |
| keys `rhodes` | `gu_tine_ep` (GU, bank 0, prog 4) | 0.0 | −5.94 | CD-28: mean of K1 rootless −6.04 and K2 spread −5.84 |
| keys `wurli` | `gs_wurli_ep200` | −3.22 | −6.435 | CD-28: mean of −6.65 and −6.22 |
| keys `piano` | `vcsl_upright_knight` | 21.86 | −6.06 | CD-28: mean of −6.06 and −6.06 |
| kit `studio` | `kit_virtuosity` (the whole drum stem) | −14.94 | 15.26 | median F |
| its bass | `bass_vsco2_pizz` | 34.97 | 5.09 | median F |
| kit `jazz` | `kit_gu_jazz` (GU Jazz kit 32 one-shots) | 0.0 | 15.21 | median F |
| its bass | `bass_fp_finger_yr` | −2.79 | 5.13 | median F |

- **Levels** are the fields of SHOOTOUTS.md §7.1 (`trim_db`, `fader_db`; docs/RENDER.md §3.3). Every instrument carries
  both, even at 0.0. Each is t and median F of lofi-shootout 13651610 (Day 2), equal in 13635646 (Day 1). Day 3
  (13656048) measured `gu_tine_ep`, `kit_virtuosity`, `bass_vsco2_pizz`, `kit_gu_jazz` and `sax_mtg_alto` again, with
  the same t and F; S\* is −31.924651028769418 in all three.
- **The keys faders are CD-28's mean, exact** (`drafts.palette.mean2`): −6.435 for the wurli, not rounded to 0.01 dB
  as SHOOTOUTS.md §3 step 5 rounds each F, because keep3 played −6.435 and the freeze keeps the faders he heard
  (CD-29). D-022 records it as a departure.
- **Kits:** the studio lanes are the Virtuosity importer's, all at pan 0.0 and db 0.0 except the borrowed shaker at
  2.47 (CD-30); the kit's trim is its field, so the renderer's sum (lane db + kit trim) gives the draft's lane db to the
  bit (2.47 + (−14.94) = −12.469999999999999). The jazz lanes are the GU one-shots at db 0.0. Both kits choke
  `[[ohat, hat], [ohat, ohat]]`.
- **Settings:** swing 0.55 on every lane (`swing_55`), humanize off (`humanize_off`), `mix.sax_warm` false
  (`sax_plain`), `sax_expr.enabled` true and `timing: lag` at 25 ms (`timing_lag`); dev_gu's sax expression otherwise.
- **Chords:** the grammar's full lists, unchanged (the borrowed-chord condition was not met); voicings rootless,
  shell, spread, quartal; `aliases: {}`.
- **Mix:** `bus_gain_db` 0.0 on every bus (each fader is its instrument's `fader_db`); `bus_target_lu` sax 0, drums
  −1.5, bass −3, keys −6; dev_gu's bus and master chains (the draft's without their chain-head trim slots).
- **Bed:** `{src: rain, start_s: 44.3, xfade_s: 1.0, delta_lu: 24.0}`: excerpt `e2` (the W1 pick `excerpt_e2`; its
  stretch's pcm_sha256 `e6f3a632…` in shootouts/rain_excerpts.json) at Δ 24 (the W2 pick `rain_d24`). §3.3 says why
  only the level comes from `rain_d24`.
- **Loudness:** −14 LUFS ± 0.1, at most 4 iterations, ceiling −1 dBTP.
- **Sources:** GeneralUser GS is the one committed source; every other is mode `fetch` under `$LOFI_ASSETS` (D-012),
  the rain file included (`derived/rain/235827__metadex__soft-rain-on-roof-window_44k.wav`, sha256 `e6884875cba5…`,
  CC0 1.0, built as `lofi_rl.shootout.palettes.read_rain` builds it).

**The bed as placed and as heard.** The renderer scales the bed to the music's loudness minus Δ, measured after the
master chain and before the limiter (RENDER.md §3.4); the manifest's `bed.lufs` is that target. After the limiter the
bed comes out closer to the music than Δ. On Day 3's four W2 renders at Δ 24, the ones he judged, the bed sat 21.99 to
23.98 LU under the music (`bed.bed_lu` in runs/shootout-13656048/summary/renders.json). E2 at Δ 24 was never played as
a pair (W1 played e2 at Δ 18, W2 played its levels on e1), so the sign-off listen is its first hearing.

## 2. What differs from `draft_day2.yaml`

| # | draft_day2.yaml | lofi_rain_v1.yaml | why |
|---|---|---|---|
| 1 | `id: draft_day2` | `id: lofi_rain_v1` | instrument.md §6 |
| 2 | `#\| draft_levels:` block | `#\| freeze_provenance:` block (§3.3) | the numbers move into fields; the block keeps the trace |
| 3 | no per-instrument fields | `trim_db` and `fader_db` on all 8 instruments | SHOOTOUTS.md §7.1 replaces CD-51's carriage |
| 4 | studio lanes at db −14.94, shaker −12.469999999999999 | the importer's 0.0 and 2.47; `kits.studio.trim_db: -14.94` | the trim is a field |
| 5 | chain-head `[Gain, …]` slots on sax 6.63, keys 0.0, bass 34.97 | none: dev_gu's chains | the trims are fields |
| 6 | `bus_gain_db` sax 2.55, drums 15.26, bass 5.09, keys −5.94 | all 0.0 | the faders are fields |
| 7 | `bed: null`, no `rain` source | the Day-3 bed and the `rain` source | Day 3, his word |
| 8 | `frozen: null` | null, then the date of his APPROVED line | SHOOTOUTS.md §1 step 4 |

**Equivalence:** folded back into the draft's carriage (each trim in its chain-head slot or on its kit's lane db, each
fader in `bus_gain_db`, the bed dropped), every one of the 6 (keys, kit) bands is the draft's render palette for that
band (`drafts.palette.render_doc`) apart from its `id` (and, once sealed, `frozen`, which no render reads). So the
music of the frozen palette is the music keep3 played, by construction; the bed is the one addition. The check tests it
(§3.6 f); the sign-off render job renders the kept 10 with the frozen palette and with a no-bed twin, which must equal
keep3's renders.

## 3. The generator

### 3.1 Where it lives
A package of its own, `lofi_rl.freeze`, so the draft module (`lofi_rl.drafts.palette`, whose build must keep writing
`draft_day2.yaml` byte for byte) stays as it is. It imports the draft module's helpers (`load`, `render_doc`,
`untrimmed_lanes`, `yaml_text`, `mean2`, the sha helpers) and constants (`KEYS`, `KITS`, `SAX`, `SETTINGS`,
`TRIM_BUSES`, `BUS_ORDER`), so nothing is copied. Its own constants (`lofi_rl.freeze`): `FROZEN`, `PICKS` (the gate
file's picks, §4), `KEPT` (keep3's table), `REDOS` and `SIGNOFFS` (the redo rounds and the sign-off listens, §5),
`BED`, `JOBS` (13635646, 13651610, 13656048, keep3's 13656049, then each redo round's job under its key), `CHANGES`,
and the frozen grammar's `REGEX_CHARS`, `REGEX_SHA256` and `V2_REFERENCES`; and three readers of them, `kept_rounds`,
`kept_now` and `tossed_kept` (§5.1).

### 3.2 CLI
```
python -m lofi_rl.freeze build [--draft palettes/draft_day2.yaml]
    --levels-day3 runs/shootout-13656048/summary/levels.json
    [--plan shootouts/plan.yaml] [--rain shootouts/rain_excerpts.json]
    [--frozen null|YYYY-MM-DD] [--out palettes/lofi_rain_v1.yaml]
python -m lofi_rl.freeze check [--palette palettes/lofi_rain_v1.yaml] [--gate gates/GATE_palette.md]
    [--levels-day1 L1 --levels-day2 L2 --levels-day3 L3] [--verify] [--sealed]
python -m lofi_rl.freeze vocab [--palette palettes/lofi_rain_v1.yaml] [--json]
```
Exit codes as `drafts.palette`'s: 0 ok, 2 bad input (a file that does not load, a refused build), 3 problems found.
`check` prints one `ok:` line naming the sha256 and each check, or `problem:` lines on stderr.

### 3.3 Inputs and the transform
The inputs: `palettes/draft_day2.yaml` (loaded with `drafts.palette.load`; it must pass its own check),
`shootouts/plan.yaml` (read by `lofi_rl.shootout.plan.load_plan`), `shootouts/rain_excerpts.json`, and Day 3's
`levels.json`, the one input that is not committed. The build reads no asset file, so it reruns anywhere.

1. Copy the draft document; set `id` and `frozen`.
2. Each instrument's `trim_db` and `fader_db` from the draft's `#| ` record; each kit's lanes with its trim taken off
   (`untrimmed_lanes`).
3. Drop the chain-head `Gain` slot of the sax, keys and bass chains (any other head is refused, and so is a `Gain`
   left anywhere in a chain); `bus_gain_db` 0.0.
4. The bed: the excerpt that the W1 variant `excerpt_e2` sets, and **only** the `delta_lu` of the W2 variant
   `rain_d24`. `rain_d24` also sets `bed.excerpt: e1`, since W2 played its levels on e1, so applying the two variants
   in order would freeze e1. The `rain` source comes from the excerpt record.
5. The provenance block `#| freeze_provenance:` (schema `lofi-rl/freeze-provenance/1`): the inputs and their sha256s;
   the draft's `draft_levels` record verbatim; `day3` (job, levels sha256, S\*, `moved`, and t and F of each band
   instrument Day 3 measured); `bed` (excerpt, the two variants, start, crossfade, Δ, the excerpt's and the file's
   pcm sha256s and the file's sha256); `picks` (the three confirmation dates); `rules`; and one record per change of
   Shresth's (§3.5). It holds nothing that changes at the seal.
6. A fixed header, then the document in palette.py's key order.
7. The text must parse (`parse_palette` and the provenance block), or nothing is written.

**Day 3 must agree with Day 2.** For every band instrument Day 3 measured, its t, F and instrument (source sha256,
bank and program, or kit map) must equal Day 2's; S\* and the targets must be equal, `moved` empty, Day 3's prior job
1's levels file, its base palette Day 2's, and its plan the committed `shootouts/plan.yaml` (so the bed's variants
are the ones Day 3 played). Day 3 must have measured the band it played (the draft's leaders, `rhodes` and `studio`,
and the sax). Any move stops the build.

### 3.4 Byte-identical rebuilds
The YAML is written by the pure-Python SafeDumper with the draft's options, so the bytes never depend on libyaml (a
test dumps the same document with libyaml and compares); floats are written by repr; there are no timestamps,
absolute paths or environment values. Day 3's levels file is recorded under its pulled path
`runs/shootout-13656048/summary/levels.json` and pinned by its sha256 wherever it was read. Offline, the check rebuilds
it from the provenance block (`day3_from_provenance`: the fields the build reads), so the rebuild needs committed
files alone; `tests/test_freeze_palette.py` also rebuilds from that synthesized file, as tests/test_drafts_palette.py
does for its inputs.

### 3.5 Shresth's changes (`CHANGES`)
Nothing changes without his words. An entry of `lofi_rl.freeze.CHANGES` has a `rule`, `by: Shresth`, a `date` and his
`words` verbatim; the build refuses an entry without them, and an entry whose rule it does not know.

**One rule exists: `keys_offset_db`** (Q1: new keys levels). `db` gives a number for each keys name, added to that
keys fader as the exact sum of the decimals as written, and recorded in the provenance as
`rules.keys_offset_db {by, date, words, db}`. Checks (e) and (f) trace it; tests/test_freeze_palette.py builds one
with synthetic words.

**Any other change is new code before it is a rebuild.** After a sign-off that does not pass, his words may change the
bed, a level or a pick. Every such change except a keys fader offset first needs its own `CHANGES` rule in
`lofi_rl.freeze.palette`, with the check rule that traces it and tests; only then does `freeze build` run with his
entry. Editing a constant is not that path: with `BED` set to another of Day 3's variants the build writes that bed,
and checks (b) and (e) pass, but (c) and (d) refuse the file (the gate file's Day-3 picks and `PICKED_BED` are
`rain_d24`'s), and its provenance would hold none of his words. A rule for the bed touches:
1. `freeze/palette.py`:
   - `CHANGE_RULES` and `changes_record`: a rule `{by, date, words, excerpt_from, delta_from}`, recorded verbatim
     under `rules` with a name of its own (`rules.bed` is already the build's rule text);
   - `_bed`: it refuses a W2 variant without `bed.delta_lu`, and `rain_none` sets `bed: null`, so a palette without a
     bed cannot be built today. For none, the rule writes `bed: null` and leaves out the `rain` source;
   - the header, which names the Day-3 rain;
   - `picks_problems` (`PICKED_BED`) and `numbers_problems` (which compares with `BED`, expects the `rain` source and
     refuses a source that plays nothing): they compare with the recorded change instead.
2. `freeze/gate.py` `constants_problems`: it keeps comparing the gate file's Day-3 picks with `BED`, since the Day-3
   section records his confirmed picks and is not rewritten; the change is checked on its own.
3. `freeze/signoff.py` `expected_bed` and `bed_problems`: the bed in effect, with none allowed (then there is no bed
   stem to check), and no "new to your ears" claim for a pair Day 3 played.
4. `freeze/batch.py` `note_text` and `push_text`: words for the bed change (today they name the rain he picked in Day 3,
   and their change words name only the keys).

Its tests build `rain_d30` and `rain_none` through the rule. A level other than a keys fader also touches
`build_text`, `numbers_problems` and `fold_problems`, which apply only keys offsets today, and `batch.CHANGE_WORDS`,
which names the keys.

**`CHANGES` is empty.** Q1 asked him whether the keys of the kept 10, which measure off their target, stay as he heard
them; his answer (2026-11-03) was "Keep as heard (Recommended)", so the keys faders are the ones keep3 played.

### 3.6 The check
**Offline** (committed files only; `tests/test_freeze_palette.py` runs it on every commit):
- (a) the file parses, and its provenance block parses with its schema;
- (b) a fresh build from the inputs is byte-identical (Day 3's levels from the provenance), and each input still has
  the sha256 the provenance records;
- (d) the names, options and settings are the picks' (the assertions of `drafts.palette.picks_problems`), plus the bed
  `{rain, 44.3, 1.0, 24.0}` from excerpt e2, and GU the one committed source;
- (e) every number is traced: each field equals its record by its rule (t; the median F; CD-28's mean; a recorded
  change as the exact decimal sum); Day 1's measures (all 8 instruments) and Day 3's (the 5 it took, which
  `day3.measured` lists) equal Day 2's; the kit lanes are the draft's without the trim and hash to the kit map job 2
  measured; the chains are dev_gu's; groove, sax expression, targets, warmth, chords, voicings, aliases and loudness
  are the draft's; the bed is the record's excerpt at `rain_d24`'s level; the sources are the draft's and the rain,
  each one played;
- (f) the fold of every (keys, kit) band is the draft's render palette (§2), and with a recorded keys offset differs
  from it by exactly that offset;
- (g) the LoopScript regex is 3,408 characters, sha256 `5c081828ac50…`: the draft's, since the regex reads only the
  vocabulary (dev_gu's is 3,395 characters, `1412f11f…`);
- (h) prompt v2 gives its 9 references under `strict=True`; they are the same texts as under dev_gu and draft_day2;
- (i) `frozen` is null, or with `--sealed` the date of the APPROVED line under the gate file's `## Approval`.

**With `--gate gates/GATE_palette.md`:** (c) the gate file's picks equal `PICKS` (§4), and `PICKS` names what the
palette is built from (the draft's `KEYS`, `KITS`, `SAX`, `SETTINGS` and `BED`); each redo round's section equals its
`REDOS` entry and each sign-off listen's section its `SIGNOFFS` entry (§5.2), and both constants keep their own rules.
It runs in the laptop checks, in job B and at the seal.

**Permanent tests read the real gate file,** which keeps changing after the freeze:
- tests/test_freeze_checks.py: its keep3 table, and `checks report --gate`, which runs this check on the committed
  palette;
- tests/test_freeze_seal.py: the confirmation dates, and copies given an APPROVED line under `## Approval`, which
  the seal checks;
- tests/test_freeze_tables.py: a copy, with the drawn tables written into it.

They find what they read by heading, bold label and backticked id (§4's reader, and `tables.sections` for where
`## Approval` ends), never by line number. So what the file gains after the freeze (his APPROVED line, the sign-off
counts, the drawn tables) does not break them. An edit to what the reader reads (a pick, a confirmation, a keep3 row,
a heading written twice) fails them, as it fails the check.

**With `--levels-day1 L1 --levels-day2 L2 --levels-day3 L3`:** each file's sha256 is the one recorded (L1
`9cf78680…`, L2 `8f4f6e9f…`, L3 `b313ff9a…`); every provenance number equals its value there (each instrument's t and
F in each job, job 2's instrument, the K2 spread F, S\*, the targets, what Day 3 measured); `drafts.palette build` from
L1 and L2 writes the draft as committed; and a build from the real L3 is the file. That draft build reads the studio
kit's archive (`derived/kits/kit_virtuosity.tar` and its blocks file) from the fetched assets, so this check, like
`--verify`, needs them. Where runs/ holds the three files, tests/test_freeze_palette.py runs it on them, and edits each
of those provenance numbers in memory: each edit names its own problem.

**With `--verify`:** `load_palette(…, verify=True)`: every source hashed against its sha256, every preset read, and
the SF2 coverage of every note the compiler may send (palette.py `note_cells`). It needs the fetched assets.

## 4. Reading the gate file (`lofi_rl.freeze.gate`)

The gate file keeps changing after the freeze (the sign-off counts, the drawn round tables), so nothing reads it by
line number:
- a section by its heading (`### Day 1: voted`, `### Day 2: voted`, `### Day 3: voted`, `## Kept drafts (keep3)`, and
  each redo round's `## Kept drafts (keep3 redo<k>)`), running to the next heading of its level or above; its heading
  must name its job (a redo round's heading its batch too);
- in a day's section, the line `**Shresth's confirmation, <date>:**` and the quoted words on it (Day 3: "Yes, use
  these" and "24 below (Recommended)");
- each pick by the bold label of its item in the **Survivors** (Day 1) or **Results** (Days 2 and 3) list, and every
  backticked id the item holds, its sub-bullets included, in order; Day 2's **Sax** also by its sub-bullets Warmth,
  Expression and Timing;
- keep3's table, and each redo round's, by its header row's `Brief` and `Kept` columns (a redo round's table lists only
  the briefs he kept a loop for);
- each sign-off listen by its `## Sign-off listen` heading that names its batch (`batch b1103-signoff`, a whole word,
  so `b1103-signoff-2` is another listen) and its job, and the backticked files of its `- **Tossed:**` item (`- **Tossed:**
  none.` for a listen that passed).

`picks_problems` names the heading of every difference: a heading, job, batch, confirmation, label, id, kept file or
tossed file that is missing or differs, and a redo round or sign-off listen the gate file records that `REDOS` or
`SIGNOFFS` lacks (or the other way round). A line moved within its section, or added outside these items, changes
nothing. `kept_now_table` is keep3's table with each redo round's table over it, in order: the gate file's side of
`kept_now()`.
`approval_date` reads the APPROVED line: under `## Approval`, exactly one line `APPROVED by Shresth Rana <yyyy-mm-dd>`
with a real date (trailing whitespace aside). A note beside it is ignored; a second such line, or a line that starts
as an approval in another form, is refused. `confirmations` and `kept` give the seal and the checks what they read.

## 5. The sign-off loop and the kept list

The kept loops render again with the frozen palette in job B, the sign-off mode of `lofi-drafts` (docs/DRAFTS.md
§7.1), and Shresth hears them in one triage batch: the sign-off listen. The palette is sealed (`frozen:` set,
`lofi_rl.freeze.seal`) only after a listen with no Toss and his APPROVED line.

### 5.1 The loop
1. **The listen.** Job B serves `b<MMDD>-signoff`, then `b<MMDD>-signoff-<k>`. His marks go into the gate file under a
   `## Sign-off listen` heading that names its batch and its job, with a `- **Tossed:**` item, and into `SIGNOFFS`.
2. **A Toss.** He is asked what fired: the rain, a part, or the loops themselves.
   - **The palette:** a change in his words goes through §3.5 (its rule first, then `freeze build` with his entry),
     then job B again.
   - **The loops:** the palette does not change. The tossed loops' briefs reopen through the drafts redo path
     (docs/DRAFTS.md §6): new drafts in `drafts/v1/redo<k>/`, a normal `lofi-drafts` run of them into a keep3 batch
     `b<MMDD>-keep3-redo[<n>]`, where he keeps one per brief or says "none acceptable" (that brief goes to the next
     round).
   - **His picks there** enter the kept list only as his words recorded them: the round's own gate section
     `## Kept drafts (keep3 redo<k>)` (naming its batch and its job, with a Brief/Kept table of the briefs he kept a
     loop for) and its `REDOS` entry with the same picks, its job in `JOBS` under its key. Then `checks expect` writes
     keep3_kept10.json again (§5.3), and job B runs again on the new kept list.
3. **No Toss.** The listen passed: `SIGNOFFS` records it with no tossed file, then his APPROVED line, then the seal.

On 2026-11-03, b1103-signoff (job 13701026) tossed d01_chords, d06_horn, d07_chords and d10_groove. His words were "The
loops themselves", so d01, d06, d07 and d10 reopen. `REDOS` stays empty until he picks in their redo batch.

### 5.2 The constants
- **`KEPT`:** keep3's table.
- **`REDOS`:** the redo rounds, in order. An entry is added once he has picked in its batch:
  `{key: keep3_redo<k>, folder: drafts/v1/redo<k>, batch: b<MMDD>-keep3-redo[<n>], job: <its lofi-drafts job>,
  heading: "## Kept drafts (keep3 redo<k>)", kept: {brief: drafts/v1/redo<k>/<brief>_<persona>.ls}}`. `kept` holds
  only the briefs he kept a loop for. The rules (`gate.records_problems`):
  - k rises from round to round and names the folder and the heading;
  - the batch has the redraft form, and the job is `JOBS[key]`;
  - each pick is a draft of its own brief, a brief of keep3's, in the round's folder.
- **`SIGNOFFS`:** every sign-off listen, in order, as `{batch, job, tossed}`. Today there is one: b1103-signoff, job
  13701026, the four files above. A listen that passes has `tossed: ()`. The jobs rise, and every tossed file was kept
  by keep3 or by a redo round.
- **`kept_rounds()`:** {brief: (file, the key of the round that picked it)}: `KEPT` with each `REDOS` round's picks
  applied in order, in brief order. `kept_now()` gives the files alone.
- **`tossed_kept()`:** the files of `kept_now()` that some listen tossed, each with the batches that tossed it. A
  tossed file stays tossed until a redo pick of another file replaces it; a later pick of the same file keeps it
  tossed.

`freeze check --gate` compares `REDOS` and `SIGNOFFS` with the gate file (§3.6 (c), §4).

### 5.3 `drafts/v1/keep3_kept10.json`, schema `lofi-rl/keep3-kept10/2`
Written by `python -m lofi_rl.freeze.checks expect`. Job B reads it from its code snapshot: runs/ is never committed or
synced, and a scratch purge cannot remove it.
- **`sources`:** `[{key, path, job, sha256}]`, keep3 first, then each redo round that supplies a kept loop, in round
  order. Each is the summary/renders.json of its round's job `JOBS[key]`, and its path must name `drafts-<job>`.
- **`draft_palette` and `s_star`:** S\* of `palettes/draft_day2.yaml`. Every source's bus records were computed with it,
  since keep3 and every redo round render with the draft palette.
- **`kept`:** `kept_now()` as a list, in brief order.
- **`loops`**, keyed by the kept file's stem (one loop per brief, so no two share a stem): its `file`, its `source`, and
  schema 1's fields (draft_sha256, brief, persona, keys, kit, render palette, n, pcm_sha256, music_sha256, LUFS, true
  peak, master gain, each bus's record). They come from exactly one ok record of that file, under its stem, in its
  own source's renders.json, and the committed text must still be the one rendered.
- **`allow_tossed` and `tossed`:** `expect` refuses while a kept file is tossed, naming the brief, the file and the
  batch. With `--allow-tossed` it writes the file and lists each such loop under `tossed`.

Every kept draft is read by its file path, never by its stem: `drafts/v1/redo1/d01_horn.ls` shares its stem with
`drafts/v1/d01_horn.ls`.

**The committed file (2026-11-03)** was regenerated with `--allow-tossed` from keep3's renders.json
(runs/drafts-13656049/summary/renders.json, sha256 `94ee7e1e…`). No redo round is recorded, so the kept list is still
keep3's 10, and the four tossed loops are listed under `tossed`.
- Every per-loop value equals schema 1's (sha256 `9df59929…`, main bd33ce2): all 16 fields of each of the 10 loops.
  Each loop gains `source: keep3`, and the old `keep3` record becomes `sources[0]`.
- The file still records keep3's values, and nothing takes the tossed loops from it: `checks report` fails on exactly
  those four (its check (i)), and job B stops at its check stage, until a redo round's picks replace them and
  `expect` runs again without the flag.

### 5.4 The commands
```
python -m lofi_rl.freeze.checks expect --renders keep3=runs/drafts-13656049/summary/renders.json \
    [--renders keep3_redo1=runs/drafts-<job>/summary/renders.json ...] [--allow-tossed]
python -m lofi_rl.freeze.checks report --gate gates/GATE_palette.md \
    --renders keep3=runs/drafts-13656049/summary/renders.json [--renders keep3_redo1=...] --verify
make submit JOB=jobs/drafts.sbatch ARGS='--drafts drafts/v1/redo1 --briefs d01+d06+d07+d10 --against drafts/v1'
make submit JOB=jobs/drafts.sbatch ARGS='--drafts drafts/v1 --signoff palettes/lofi_rain_v1.yaml'
python -m lofi_rl.freeze.seal [--dry-run]
```
`--keep3 PATH` is `--renders keep3=PATH`. Job B's arguments stay `--drafts drafts/v1 --signoff ...`: it finds the redo
folders under drafts/v1 (`REDOS`'s and every `redo<k>/` present), so the job script and its registry sha
(`85a1ae8e085fd255`) do not change. The checks and job B's stages are docs/DRAFTS.md §7.1's.

### 5.5 The seal, the tables and the note after a redo round
- **The seal** (`lofi_rl.freeze.seal`, step 2) takes the latest sign-off job under runs/, the highest job id in sign-off
  mode. It must be the latest `SIGNOFFS` entry (its job and batch: his marks of it are recorded first), and that listen
  tossed nothing. Before it reads that entry, `REDOS` and `SIGNOFFS` must keep their own rules (`gate.records_problems`,
  §5.2), so a malformed entry refuses by name. Today the latest entry, b1103-signoff, tossed 4, so the seal refuses.
- **The drawn tables** (`lofi_rl.freeze.tables`) publish each redo round's job as they publish keep3's
  (`Drawn, keep3 redo<k>`), its job `JOBS[key]`.
- **A later sign-off batch's note and push** say "again": the last listen played this rain at this level, and redo
  picks may be among the kept loops, so neither "new to your ears" nor "the same loops" holds.

## 6. Run on the laptop, 2026-11-03

`python -m lofi_rl.freeze build --levels-day3 runs/shootout-13656048/summary/levels.json` wrote
`palettes/lofi_rain_v1.yaml`, 496 lines, sha256 `d3590c8da2e186de2b5a113a585f17646dfec9390f1eb9de569b1bacaa47bc8a`.
Then

```
python -m lofi_rl.freeze check --gate gates/GATE_palette.md --levels-day1 runs/shootout-13635646/summary/levels.json
    --levels-day2 runs/shootout-13651610/summary/levels.json --levels-day3 runs/shootout-13656048/summary/levels.json
    --verify
```
printed `ok: palettes/lofi_rain_v1.yaml (lofi_rain_v1, sha256 d3590c8d…)` with every check of §3.6, the regex
(3,408 characters, sha256 `5c081828ac50718edc4db82c2a855ea52210debb375280ba01e8442de60f8558`), prompt v2 at 9 of 9
(strict) and "sources, presets and coverage verified" (the 8 source files, 561,512,304 bytes, hashed against their
sha256s).

This is the second build. The first (494 lines, sha256 `bf9be2c3fb3c…`; the laptop runs in docs/DRAFTS.md name it)
carried `trim_db` and `fader_db` rule texts that said every value was equal in Day 1 and Day 3, though Day 3 measured
only 5 of the 8 instruments (not `gs_wurli_ep200`, `vcsl_upright_knight` or `bass_fp_finger_yr`). The rebuild changed
only the provenance block's `rules:` lines (7 lines in place of 5); every other line is byte for byte the first
build's, so nothing a render reads changed.

## 7. Claude's defaults here
Each is overridable.
- The drafts' names (`rhodes`, `wurli`, `piano`, `studio`, `jazz`, `alto`) and `aliases: {}`: the kept 10 parse with
  no repair only under them, and prompt v2's references stay byte-identical.
- Every level in a per-instrument field, `bus_gain_db` 0.0, no chain-head slots; each instrument carries both fields
  even at 0.0, in a fixed key order.
- CD-28's keys mean kept exact (−6.435), not rounded to 0.01 dB.
- The values from Day 2, with Day 1 = Day 2 asserted for all 8 instruments and Day 3 = Day 2 for the 5 Day 3
  measured; any move stops the build.
- The provenance in a `#| ` block inside the file, pinned by its sha256 as the draft's is.
- `frozen` is the date of his APPROVED line, written quoted (`frozen: '<date>'`); only that line differs between the
  signed-off file and the frozen one.
- The gate-file reader finds headings, labels and backticked ids, never line numbers. It runs in the checks, in job B
  and at the seal, and the permanent tests that read the real gate file read it only this way (§3.6), so its later
  additions (his line, the sign-off counts, the drawn tables) leave them as they were.
- The redo rounds and the sign-off listens are constants beside `PICKS` (`REDOS`, `SIGNOFFS`), each traced to its own
  gate section and compared with it by `freeze check --gate`, so the kept list in effect never comes from a file that
  the gate file does not confirm. A tossed file stays tossed for good (a later pick of it again is refused), and a
  `## Sign-off listen` section always carries a `- **Tossed:**` item (`none.` when it passed), so a missing record
  fails closed.
- keep3_kept10.json is regenerated with `--allow-tossed` while the tossed loops are still kept, so it keeps recording
  their renders, and the flag is written into the file; the report and job B refuse those loops from the constants,
  never from the file.
- In job B a drafts/v1 draft's render id is its stem and a redo folder's draft's id is `<folder>_<stem>`
  (`redo1_d01_horn`): ids stay unique across the folders, and drafts/v1's ids, the ones the first sign-off job used,
  do not change.
