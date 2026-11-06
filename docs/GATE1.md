# GATE1.md: the Phase-1 gate on the frozen palette

D-021's next step ("What follows", item 1): D-020's official Phase-1 gate, re-run on the frozen palette
`palettes/lofi_rain_v1.yaml` (sha256 `09f80a64…`, D-022) with the settings D-020 chose, prompt v2 under the grammar.
The job is `lofi-junk-rate` with `ARGS='500 2026 1 v2'` on `-p gpu` (job type `5a423302eb84dd91`, at most 504 SU,
registered by Shresth on 2026-11-06; D-022 "Consequences"). This file is the tool that reads its results, builds
Shresth's ear check, counts his votes and writes `gates/GATE_phase1.md`.

- **Code:** `src/lofi_rl/gate1/` (`__init__`: the registered values and `load_run`; `check`, `batch`, `tally`,
  `gatefile`; `cli`), run on the laptop from the repo root as `PYTHONPATH=src .venv/bin/python -m lofi_rl.gate1
  <check|batch|tally|gate>` (written `python -m lofi_rl.gate1` below). Tests: `tests/test_gate1_*.py`, on synthetic
  runs, FLACs and vote logs (`tests/helpers_gate1.py`).
- **Reads:** a run folder `runs/junk-<job>/` with `summary/` as `make pull` brings it (`junk.json`, `scored.jsonl`,
  `completions.jsonl`, `clips.json`) and, for the batch, the run's 50 one-pass FLACs copied from scratch into a
  folder of the laptop (`runs/junk-<job>/clips/`; runs/ is never committed).
- **Writes:** the batch under `inbox/<date>/` and its records `summary/draw.json` (the draw's seed) and
  `summary/batch.json` into the run's summary (all three never committed); the tally's JSON beside the votes, in
  `data/votes/audit/` (never outside `data/` inside the repo); `gates/GATE_phase1.md`, which is committed.
- **Sound claims:** none. Every number the tool prints is read from the run's files or the vote log.

## 1. The bar

D-020's bar, unchanged, as D-021 restates it and CLAUDE.md §9 sets it (60 % of 500, 30 % of 50):

1. `pass_not_copy.count` ≥ 300 of 500 in a valid run (D-019: the 60 % reads `pass_not_copy`, so copying the in-prompt
   example cannot carry it);
2. at least 15 of its 50 clips "fine" by his ear: Fine, Good or Great (TASTE.md, Tiers: Toss = a dealbreaker fired,
   Fine = not embarrassing, Good, Great);
3. his own line under `## Approval` in `gates/GATE_phase1.md`: `APPROVED by Shresth Rana` and the day he approves,
   written bare (approving on 2026-11-09, `APPROVED by Shresth Rana 2026-11-09`; kit.gates).

Beside the bar, before he signs, the gate file reports three readouts that decide nothing (D-021): the share of the
passing non-copies whose loop carries the shown example's A or B line verbatim, the same share among the 50 clips, and
his "fine" count split by whether a clip carries such a line.

## 2. `check RUN_DIR [--json]`

**Validity** is D-020's amended list ("an arm counts only if its `summary/junk.json` shows all of the following"),
with D-022's two swaps and its one dropped clause. Each of the 27 conditions is reported by name (its junk.json
field) with what junk.json shows:

| group | conditions |
|---|---|
| Complete | `n` 500, `requested` 500, `partial` false, `unreadable_lines` 0, `generate.stopped` null |
| As registered | `generate.model.revision` `cdbee75f…`; `briefs.seed` 2026; `sampling` equal to runs/junk-13632676's; `palette.sha256` `09f80a64…` (D-022) and `palette.matches_generate` true; `prompt.version` v2, `prompt.template_sha256` `127e3094…`, `prompt.template_matches_generate` true, `prompt.rows_matching` 500, `prompt.example_mismatch` 0 |
| Constrained | `constrained.enabled` true; `constrained.regex_sha256` and `constrained.in_language.regex_sha256` `5c081828…` (D-022); `constrained.in_language.matches_generate` true; `constrained.in_language.stop.out_of_language` 0; `constrained.in_language.other.n` 0 |
| What the regex guarantees | `failures.G1`, `G3`, `G4`, `G5` and `crash` 0 |
| One stamp | `generate.host.stamp` and `score_stamp` show the same `commit=… dirty=0` |

- **The swaps** (D-022): the frozen file's sha256 B in place of dev_gu's `b2e4a340…`, and the frozen grammar's
  `5c081828…` in place of `1412f11f…`. **The dropped clause:** D-020's second amendment tied the stamp's commit to the
  60 frozen files of `docs/prereg/D-020_junk_path.txt`; that list recorded its arms' code and does not bind the gate.
- **Types are the registered ones:** `partial` 0 is not false, `n` 500.0 is not 500, and a field that is absent is
  "missing", never null.
- **Shown, never judged** (D-020): `constrained.backend`, the render step and G7.
- **The bar's first half:** `pass_not_copy.count` against 300, with `rate` and `ci95` (Wilson) as junk.json gives
  them.
- **The readouts:** from scored.jsonl's `verbatim` (junk.py: the vamp, keys, A and B lines equal to the shown
  example's), over the passing non-copies, and over the clips that rendered (clips.json records with `ok` true), with
  the render step's whole sample beside it when some did not render. Without clips.json, the clips' readout is
  reported as absent and validity is still checked.
- **Bad input** (`lofi_rl.gate1.load_run`, exit 2): a file missing or unreadable, or files that are not one run's:
  scored.jsonl's rows and counts against junk.json's (`n`, `pass.count`, `pass_not_copy.count`) and against
  completions.jsonl's, and clips.json against the registered draw from that scored.jsonl: `junk.sample_indices` over
  the passing non-copies with k 50 and the run's brief seed (`briefs.seed`), the render step's defaults, with the
  same palette. A clips.json of another k or seed (a hand re-render with `--k 100`, say) is another draw, so it is
  refused and never batched. junk.json's `render` block, when there, must be the one clips.json gives
  (`junk.render_block`); the render step merges it at its end, so a step cut short leaves none, and its clips.json is
  read as far as it got. Every problem is listed.
- **Exit:** 0 valid and the first half met, 3 otherwise, 2 bad input. The last line is `GATE1_CHECK <verdict>`.

**On arm B** (runs/junk-13640325: v2 with the grammar, on dev_gu; 2026-11-06, laptop): `INVALID, 24 of 27
conditions hold; failing: palette.sha256, constrained.regex_sha256, constrained.in_language.regex_sha256`, exactly the
swapped values; the first half met at 446 of 500 (ci95 [0.8617, 0.9163]); a verbatim A or B line in 367 of the 446
passing non-copies (339 both) and in 38 of its 50 clips (35 both); exit 3. tests/test_gate1_check.py pins this.

## 3. `batch RUN_DIR --clips DIR [--inbox DIR] [--date YYYY-MM-DD]`

One triage batch of the run's clips for his ear check.

- **Only for a run that passes check** (exit 3 otherwise, nothing written): an invalid run decides nothing (D-020),
  and under 300 the gate fails whatever the ear says.
- **The clips:** clips.json's records that rendered, in the sample's order. One that did not (G7, another gate, an
  infrastructure error, a render-deadline skip, a sample the render step never reached) is left out and named in
  batch.json and on the terminal.
- **Each FLAC must be what clips.json records.** clips.json keeps no sha256 of the FLAC file: junk.render_clip records
  the float render's pcm_sha256 (`audio_sha256`) and the FLAC's size (`flac_bytes`), and the 16-bit copy is dithered,
  so its samples never hash to the float render's. So `<i>.flac` is held to every field clips.json does record of it:
  a plain file of exactly `flac_bytes`; a STREAMINFO of one pass (44,100 Hz, 2 channels, `n` frames, `n` =
  `loop_samples(bpm)`); and its samples, decoded at render.formats' full scale (32767) and measured with the render's
  own functions, within 0.002 of clips.json's `lufs` and `true_peak_dbtp` (kept to 3 decimals). Should clips.json ever
  record a `flac_sha256`, that sha256 must match exactly. On three laptop renders through junk.render_clip on the
  frozen palette (2026-11-06, scratch; laptop renders are not canonical, so only the copy against its own render is
  used) the copy measured within 1e-6 LU and 3.2e-4 dB of its render.
- **The render manifest** (`<i>.manifest.json`, which the render step writes beside each FLAC) is read when it was
  copied, and must then agree with clips.json and the run (pcm sha256, n, the canonical loop's sha, the palette, the
  brief, LUFS and true peak, its gates passed). The real clip id (taste_pipeline.md §2) hashes its schema and render
  chain; for a clip copied without its manifest they come from `lofi_rl.render.pipeline`, as junk renders (cycles 2),
  and batch.json lists those clips (`without_manifest`).
- **The brief** of every clip is the measurement stream's brief i, `brief.sample_brief(2026, i)` (D-021 item 1), in
  clips.json and completions.jsonl alike.
- **Triage mode** (docs/RATING.md): one loop a round, A Toss, S Fine, D Good, F Great; the page plays the 8 bars,
  then bars 1–2 again.
- **The order** is drawn from a fresh 64-bit seed with no two consecutive rounds in one energy × mood cell, as the
  sign-off batch draws it (lofi_rl.freeze.batch): the first drawn order with the fewest such neighbours, and the draw
  says whether the rule held. The seed is only in the run's `summary/draw.json`.
- **Ids** are `r` + 8 hex per round and `k` + 10 hex per clip, drawn again while one holds a word of a hidden value or
  a 10-hex stretch of a real clip id or any digest the hidden manifest keeps (lofi_rl.drafts.batch's leak rule).
- **Public manifest:** the scene, a plain note ("Phase-1 gate: 50 loops, one a round. Toss, Fine, Good or Great?"),
  and each round's bpm and clip (opaque id, file, sha256); nothing else.
- **Hidden manifest:** category `gate1`, every round of kind `none` (no hidden controls, as keep3 and the sign-off);
  per clip the real clip id, the sample index and its brief, the loop text's sha256 and the canonical sha, the
  verbatim flags and `verbatim_line` (A or B), the copy Jaccard and its reference, the render's pcm_sha256, LUFS and
  true peak, the FLAC's sha256, the run, its job id and the palette label.
- **Files:** written into `<inbox>/.staging-*`, checked by `python -m lofi_rl.ui.manifest check` in a fresh
  interpreter, then moved into place. An existing batch is never overwritten, and neither are a run's draw.json and
  batch.json: a second batch from one run refuses until they are moved aside.
- **Name:** `b<MMDD>-gate1`, then `b<MMDD>-gate1-<k>` for the inbox's k-th gate1 batch over every date.
- **Exit:** 0 built (it prints the push text, "Needs ear check: …"), 3 the run does not pass check, 2 bad input or a
  refused clip, with every problem listed and nothing written.

## 4. `tally BATCH_DIR --votes FILE [--json OUT]`

- **His standing vote per round** (`ui.votes.active_votes`): a vote he took back no longer counts, and the one after
  it does.
- **Heard** (D-013 as amended): once 1 bar has played, over every attempt of the round, undone votes and abandoned
  attempts included (`ui.audit.heard_over_attempts`). A tier given to a clip that never played is left out of the
  counts and reported apart; every clip under 1 bar is listed.
- **The counts:** per tier over the heard, voted rounds; the "fine" count (Fine, Good or Great) against 15. The bar is
  a count, so a batch with fewer than 50 rounds (a clip that did not render) still needs 15.
- **The split:** the same counts for the clips with a verbatim A or B line (the hidden `verbatim_line`) and without.
- **Bad input** (exit 2): not a gate1 triage batch (or one of more rounds than the 50 clips of the registered
  sample), a missing log, or a log that is not this batch's (another batch's event, another mode's vote, a tier
  outside the four, a round it does not hold, a clip that is not the round's).
- **Output:** the text on the terminal, and the JSON in the votes folder's `audit/<batch>.gate1-tally.json`
  (`data/votes/audit/`, where the shootout tally writes its audit files), or `--json OUT` (`-` prints it in place of
  the text). Its per-round rows pair his tiers with sample indices, which are private (D-010), so inside the repo it
  is written only under `data/`.
- **Exit:** 0 every round voted and the bar met, 3 otherwise.

## 5. `gate --run RUN_DIR [--batch BATCH_DIR --votes FILE] --write gates/GATE_phase1.md`

The gate file, in gates/GATE_palette.md's style: the bar and where it comes from; the run (job, node and GPU, the
stamp generate and score ran under, the palette label, the prompt, the grammar and its backend, the briefs, the render
step); the validity table; `pass_not_copy` with its CI; the readouts; his ear check once tallied (counts only: no
round's tier, sample or id); where the gate stands; `## How to approve` in kit.gates' wording, with the line written
out for a bare date (`--date`, default today in Pacific time), indented so that no reader counts it; and an empty
`## Approval`.

- **Rewriting** keeps everything under `## Approval`, from the line after its heading to the end of the file, byte
  for byte, and writes every line above it again. The heading is found as the seal finds it
  (`lofi_rl.freeze.gate.sections`). It refuses (exit 2, nothing written):
  - a file it did not make: another first line than the title, no `## Approval` or two, or no `Written by` line
    above it. Another gate's file (`gates/GATE_palette.md`, say) is never written over, and its approval never
    stands under this gate's numbers;
  - a file with a line that starts as an approval above that heading, which a rewrite would drop.
- `--batch` and `--votes` go together, and the batch must be the run's (its clips' job and palette label, and exactly
  the run's rendered clips).
- **Exit:** 0 written or unchanged, 2 bad input. It prints whether the file holds his line (`kit.gates`).

## 6. On the day

1. `make pull`, then `python -m lofi_rl.gate1 check runs/junk-13783877`.
2. With the clips copied into `runs/junk-13783877/clips/`: `python -m lofi_rl.gate1 batch runs/junk-13783877 --clips
   runs/junk-13783877/clips`, then the push to Shresth with the line it prints ("needs ear check").
3. `python -m lofi_rl.gate1 gate --run runs/junk-13783877 --write gates/GATE_phase1.md`, committed, so he can read the
   run before he listens.
4. He votes (`make rate BATCH=inbox/<date>/<batch>`); then `python -m lofi_rl.gate1 tally inbox/<date>/<batch> --votes
   data/votes/<batch>.jsonl`.
5. `python -m lofi_rl.gate1 gate --run runs/junk-13783877 --batch inbox/<date>/<batch> --votes
   data/votes/<batch>.jsonl --write gates/GATE_phase1.md`, committed; the votes are backed up (`make backup-votes`).
6. His line under `## Approval` is his; `python -m kit.gates gates/GATE_phase1.md --name "Shresth Rana"` reads it.

If the run is invalid, it is reported and decides nothing (D-020's rules for an invalid arm apply). If either half of
the bar fails, the gate goes to Shresth with CLAUDE.md §9's next fallback, D-021's Part B.

## 7. Claude's defaults here

Each is a reading of the decisions above where they do not say, overridable by Shresth:

- **The FLAC check** (§3): clips.json records no FLAC sha256, so the FLAC is held to its size, its one-pass header
  and its re-measured loudness and true peak, and its own sha256 is pinned from the batch on.
- **The render manifests are optional** (§3): the batch's inputs are clips.json and the FLAC folder; the manifest is
  verified when it is there.
- **The batch refuses a run that does not pass check** (exit 3): its ear check would decide nothing.
- **The ear check's sample is the registered draw** (§2): 50 clips, drawn with the run's brief seed, as the render
  step draws by default; clips.json of another k or seed is bad input.
- **The ear check's bar is 15, a count**, whatever number of the 50 rendered; the gate file says how many rendered.
- **The clips' readout** covers the clips that rendered, the ones he hears, with the whole sample beside it.
- **A tally is complete** when every round has a standing vote; a vote on a clip that never played is left out.
- **clips.json is optional for check and gate**, since validity reads junk.json alone; the batch needs it.
- **The note** names the batch and the four tiers only.
