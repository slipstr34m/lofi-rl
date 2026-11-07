# Gate: Phase 1, the base model on the frozen palette

Phase 1 passes when:
1. a valid `lofi-junk-rate` run of prompt v2 under the grammar on `palettes/lofi_rain_v1.yaml` (frozen 2026-11-04, sha256 `09f80a64…`) has `pass_not_copy.count` at least 300 of 500;
2. at least 15 of its 50 clips are "fine" by Shresth's ear: Fine, Good or Great (TASTE.md, Tiers);
3. Shresth Rana has written the approval line below.

**Where the bar comes from:** CLAUDE.md §9 (at least 60 % of 500 base-model samples pass the gates, and at least 30 % of a 50-clip ear check is "fine"); D-019 (the 60 % reads `pass_not_copy`, so copying the in-prompt example cannot carry it); D-020 as amended (300 of 500 in a valid run, and the validity list); D-021 "What follows" item 1 (this gate, on the frozen palette, and the three readouts beside it); D-022 "Consequences" (the frozen palette's two values in the list). The bar is D-020's, unchanged.

Written by `python -m lofi_rl.gate1 gate` (docs/GATE1.md). Writing it again keeps everything under `## Approval` exactly as it is. Once a line there starts as an approval, a rewrite that would change the numbers above it is refused, so his line always stands under the numbers he read.

## The run: `lofi-junk-rate` 13788483

- **Job:** 13788483 on b11-13.hpc.usc.edu (NVIDIA A40, 46068 MiB).
- **Code:** generate and score both ran under the stamp `commit=90d8a910e7ee dirty=0 diff=- at=2026-11-06T22:27:18Z` (the job's `.sync-stamp`; a job runs from its code snapshot, CLAUDE.md §4).
- **Palette:** `lofi_rain_v1@09f80a64` (sha256 `09f80a64…`).
- **Prompt:** v2, template `127e3094…`.
- **Grammar:** enabled true; regex sha256 `5c081828…` (3,408 characters); backend `xgrammar` (pin taken: true; applied: true), recorded and not part of validity (D-020).
- **Briefs:** 500 of `lofi_rl.brief.sample_briefs(2026, 500)`, one completion each.
- **Render step** (recorded, not part of validity, D-020): attempted 50, rendered 50, G7 0, other_gates 0, errors 0, skipped 0 (k 50, seed 2026).
- **Files:** `runs/junk-13788483/summary/` `junk.json`, `scored.jsonl`, `completions.jsonl` and `clips.json`.

## Validity (D-020's list as amended, with D-022's two swaps)

Every condition reads `summary/junk.json`. D-022 swaps in the frozen palette's sha256 and its grammar's, and drops the One-mirror clause's 60 frozen files; the stamp condition is the one it keeps.

| group | condition | junk.json shows | needs | holds |
|---|---|---|---|---|
| Complete | `n` | `500` | `500` | yes |
| Complete | `requested` | `500` | `500` | yes |
| Complete | `partial` | `false` | `false` | yes |
| Complete | `unreadable_lines` | `0` | `0` | yes |
| Complete | `generate.stopped` | `null` | `null` | yes |
| As registered | `generate.model.revision` | `cdbee75f17c01a7cc42f958dc650907174af0554` | `cdbee75f17c01a7cc42f958dc650907174af0554` | yes |
| As registered | `briefs.seed` | `2026` | `2026` | yes |
| As registered | `sampling` | `temperature=1.0 top_p=1.0 top_k=0 min_p=0.0 repetition_penalty=1.0 presence_penalty=0.0 frequency_penalty=0.0 max_tokens=512` | `runs/junk-13632676's: temperature=1.0 top_p=1.0 top_k=0 min_p=0.0 repetition_penalty=1.0 presence_penalty=0.0 frequency_penalty=0.0 max_tokens=512` | yes |
| As registered | `palette.sha256` (D-022) | `09f80a64…` | `09f80a64…` | yes |
| As registered | `palette.matches_generate` | `true` | `true` | yes |
| As registered | `prompt.version` | `v2` | `v2` | yes |
| As registered | `prompt.template_sha256` | `127e3094…` | `127e3094…` | yes |
| As registered | `prompt.template_matches_generate` | `true` | `true` | yes |
| As registered | `prompt.rows_matching` | `500` | `500` | yes |
| As registered | `prompt.example_mismatch` | `0` | `0` | yes |
| Constrained | `constrained.enabled` | `true` | `true` | yes |
| Constrained | `constrained.regex_sha256` (D-022) | `5c081828…` | `5c081828…` | yes |
| Constrained | `constrained.in_language.regex_sha256` (D-022) | `5c081828…` | `5c081828…` | yes |
| Constrained | `constrained.in_language.matches_generate` | `true` | `true` | yes |
| Constrained | `constrained.in_language.stop.out_of_language` | `0` | `0` | yes |
| Constrained | `constrained.in_language.other.n` | `0` | `0` | yes |
| What the regex guarantees | `failures.G1` | `0` | `0` | yes |
| What the regex guarantees | `failures.G3` | `0` | `0` | yes |
| What the regex guarantees | `failures.G4` | `0` | `0` | yes |
| What the regex guarantees | `failures.G5` | `0` | `0` | yes |
| What the regex guarantees | `failures.crash` | `0` | `0` | yes |
| One stamp | `generate.host.stamp = score_stamp` (D-020; D-022) | `commit=90d8a910e7ee dirty=0 diff=- at=2026-11-06T22:27:18Z (both)` | the same commit=… dirty=0 in both | yes |

**Valid:** all 27 conditions hold.

## The bar, first half: pass_not_copy

- `pass_not_copy.count` **477** of 500 (rate 0.954; Wilson 95 % CI 0.9319–0.9692, as junk.json gives it). It needs at least 300: **met**.

## Beside the bar (D-021: reported, never gated)

A verbatim line is the A or B line of the example the loop's own prompt showed, word for word (scored.jsonl `verbatim`).
- **Among the passing non-copies:** 391 of 477 carry a verbatim A or B line (82.0 %); 357 carry both.
- **Among the clips he hears:** 40 of 50 (80.0 %); 39 carry both.
- **His "fine" count by verbatim line:** in his ear check, below.

## His ear check: batch b1106-gate1

The run's one ear check, the batch its `summary/batch.json` records. Triage mode: one clip a round, played through its seam; Toss, Fine, Good or Great (TASTE.md, Tiers). A clip counts as heard once 1 bar has played, over every attempt of its round (D-013 as amended).
- **Counts:** 50 of 50 heard; 50 voted and heard. Great 8, Good 7, Fine 13, Toss 22.
- **"Fine" (Fine, Good or Great):** 28. It needs at least 15 of 50: **met**.
- **Split by verbatim line** (D-021):
  - with a verbatim A or B line: "fine" 25 of 40 (Great 7, Good 6, Fine 12, Toss 15).
  - with neither: "fine" 3 of 10 (Great 1, Good 1, Fine 1, Toss 7).
- **Not heard (under 1 bar):** none.
- **Not voted:** none.
- **Audit file** (private): `data/votes/b1106-gate1.jsonl`.

## Where it stands

- Both halves of the bar are met. The gate passes with his line under `## Approval`.

## How to approve
Add one line of your own under `## Approval`, starting at its first character: APPROVED by Shresth Rana, then the day you approve as year-month-day digits, with nothing around the date and nothing after it. Approving on 2026-11-06, the line is:

    APPROVED by Shresth Rana 2026-11-06

## Approval
