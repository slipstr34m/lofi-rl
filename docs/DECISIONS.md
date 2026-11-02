# DECISIONS.md: irreversible and pre-registered choices

Each entry gives a date, the decision, who decided, and why. Entries are never edited; a later entry
supersedes an earlier one. Sources: `docs/TASTE.md` (Shresth's interview), `docs/research/`, `docs/design/`.

## D-001 (2026-10-30): code moves laptop → cluster by rsync mirror, never git
**Decided by:** Shresth (CLAUDE.md), with Claude adapting the kit.
- `make sync` mirrors the repo to `$REMOTE_PROJ` with `--delete` and writes `.sync-stamp` (commit, dirty flag, diff hash).
- `make submit` refuses unless the stamp matches the laptop's committed code.
- The kit's git-pull transport, and every `git` call in its jobs, were removed (no git on the cluster).

## D-002 (2026-10-30): cluster paths
**Decided by:** Claude, after checking with `myquota`.
- `/project/anakano_429` does not exist; `/project2/anakano_429` does (1 TiB, shared by the class; 88 GiB used on 2026-10-30).
- `REMOTE_PROJ=/project2/anakano_429/shresthr/lofi-rl` is the mirror.
- `REMOTE_KEEP=/project2/anakano_429/shresthr/lofi-rl-keep` sits outside the mirror: envs, tools, votes backup, checkpoints, assets, guard log.
- `REMOTE_SCR=/scratch1/shresthr/lofi-rl` holds logs, runs, renders, inbox, caches, HF weights.
- The per-user dir is mode 700, since votes are private.

## D-003 (2026-10-30): SU budget and the job-type registry
**Decided by:** Shresth.
- The SU target is 1,000,000 and the hard stop 1,200,000 (`config/budget.yaml`). Past the target the guard needs `PAST_TARGET_OK`.
- Every job type is registered in `jobs/APPROVED.yaml` with the sha of its `#SBATCH` and non-comment lines and its `max_su`.
- Shresth approves each type once. Claude registers types ≤ 500 SU itself. A changed type, or a run over its estimate, asks again (guard exit 4).
- **Amended 2026-10-31 by Shresth** ("Yes, up to 500 SU", asked whether Claude may re-register changed job types of 500 SU or less itself): Claude re-registers a changed type of 500 SU or less itself (`BY=Claude`) and records it in the CHANGELOG. A type over 500 SU, new or changed, and a run over its estimate still need his yes in chat.

## D-004 (2026-10-30): two environments
**Decided by:** Claude; the research forced it.
- policy: vLLM 0.32.0, TRL 1.17.1, transformers 5.19.0, torch 2.13.0 (`requirements/policy.lock.txt`, 211 pins).
- scorer: torch 2.8.0, transformers 4.43.4, laion-clap 1.1.7, audiobox-aesthetics 0.0.4, pedalboard 0.9.26, numpy 1.26.4 (`requirements/scorer.lock.txt`, 111 pins).
- Why two: MERT's remote code breaks on transformers ≥ 4.44, while vLLM 0.32 needs 5.16–5.19.
- Locks are compiled on the laptop for `x86_64-manylinux_2_28` (Rocky 8.10).

## D-005 (2026-10-30): FluidSynth 2.6.1 everywhere; cluster renders are canonical
**Decided by:** Claude.
- FluidSynth 2.6.0 changed tuning and interpolation, so 2.6.1 is pinned on both machines (brew pin; conda-forge in `jobs/tools.sbatch`).
- Rendering uses the CLI float path with chorus and reverb off, `cpu-cores=1`, and a fresh process per render.
- Bit-identity is asserted only on the same platform (LESSONS 38). Every clip Shresth hears is a cluster render.

## D-006 (2026-10-30): base model
**Decided by:** Shresth ("4B: faster, more runs").
- Qwen3-4B-Instruct-2507 (Apache-2.0, dense, non-thinking), pinned revision recorded at fetch time.
- Granite 4.1 8B is measured only at the Phase 1 junk-rate gate, as a fallback.

## D-007 (2026-10-30): taste RM refresh
**Decided by:** Claude, accepted with the plan.
- The RM is frozen within a run and refreshed between runs.
- A daily hack-watch ear check (16 blind clips) can stop a run, add a penalty, refresh the RM, and resume from the last clean checkpoint.
- All votes weigh the same, whenever they were cast.

## D-008 (2026-10-30): the Phase 2 RM gate is relative to Shresth's own consistency (pre-registered)
**Decided by:** Shresth.
- **Pass if:** held-out agreement ≥ 0.9·q, where q = (1 + √(2p − 1))/2 and p is his agreement on hidden repeats.
- **And:** the lower 95% bound of agreement must be above 0.5.
- **Reporting:** absolute agreement is reported against 70%.
- **Timing:** the gate runs on Nov 15 on the votes that exist, and is re-checked nightly through Nov 17.
- **If it still fails:** run with the taste weight capped at 0.3, labeled exploratory.

## D-009 (2026-10-30): no-collapse is measured across briefs (pre-registered)
**Decided by:** Shresth ("cousins within a brief").
- Mean embedding distance across 100 held-out briefs (one loop each) at step N must be ≥ 70% of step 0.
- Same-brief loops may be close cousins; only near-identical ones are penalized.
- Within-group distance is reported, and used as a hack-watch alarm.

## D-010 (2026-10-30): votes are private
**Decided by:** Shresth.
- Votes are never committed to the public repo.
- `data/votes/` is a nested git repo whose remote is the private `slipstr34m/lofi-rl-votes`; `make backup-votes` also copies to `$REMOTE_KEEP/votes/`.

## D-011 (2026-10-30): Claude may push
**Decided by:** Shresth.
- Claude may push after `pytest -q` passes.
- No AI co-author or attribution trailer, ever.
- Never force-push.

## D-012 (2026-10-30): asset licensing
**Decided by:** Shresth (the repo is public; "NC is OK for a course project"; no purchases).
- Committed: CC0, public-domain, permissive and CC-BY assets (with an ATTRIBUTION file), each ≤ 50 MiB.
- Fetched only, never committed: non-commercial assets (for example jRhodes3d) and anything large.
- No purchases. If no free sax passes the shootout, the lead becomes Rhodes.

## D-013 (2026-10-30): vote protocol (pre-registered)
**Decided by:** Shresth.
- **Rounds:** best and worst of 4 clips from one brief, auto-played in order. Early pick at any time; pairs with unheard clips (< 4 bars) are dropped.
- **Escape hatches:** "no clear best/worst" and "all four bad". One optional reason tag.
- **Display:** the page shows the scene only. RM predictions are never shown while voting.
- **Hidden controls:** 10% repeats, at least 2 days later with positions reversed; 5% broken anchors. A session that misses its anchors counts at half weight and never enters the held-out set.
- **Tiers:** Toss = a dealbreaker fired; Fine = not embarrassing; Good = I'd play it while working; Great = can't stop nodding.
- **Amended 2026-10-31 by Shresth.** He said "Imagine it sounded so bad that I did not even let it play more than 4 bars.", then "Yes, use these" to the proposal that followed.
  - A clip counts as heard once 1 bar has played, over every attempt of its round: undone votes and abandoned attempts included.
  - Only pairs with clips that never played are dropped.
  - This holds in every mode, for the shootouts and the RM votes alike. It comes before the first RM vote.
  - First use: the Day-1 shootout tally. Before this change, the count kept only the final attempt's bars, so clips heard before an undo or a reload counted as unheard.

## D-014 (2026-10-30): final evaluation (pre-registered)
**Decided by:** Shresth.
- A blind, forced-choice A/B on 100 held-out briefs, Nov 30 – Dec 2, in 2–3 sittings.
- Step N is the checkpoint his votes favor on separate dev briefs, fixed before Nov 30.
- Secondary measure, not gated: the playlist-mode ♥ rate for step N vs step 0.

## D-015 (2026-10-30): the "palette vs free-form" ablation becomes "taste RM on vs off"
**Decided by:** Claude's default (Shresth: "I don't know"). Recorded before any results exist.
- Why: the free-form grammar is a large extra build, and RM on vs off tests the project's central claim directly.

## D-016 (2026-10-30): the first lever at the Phase 1 gate is constrained decoding
**Decided by:** Shresth.
- If syntax failures sink the junk-rate gate, grammar-constrained decoding (vLLM regex) comes first.
- Then, in order: SFT warm-start, edit-mode rollouts, stop.

## D-017 (2026-10-30): the palette never changes after the freeze
**Decided by:** Shresth ("never change it").
- Once `palettes/lofi_rain_v1.yaml` is frozen (Phase 1, after his shootouts and the kept 10), it stays as is for the rest of the project.

## D-018 (2026-10-30): seed loops are written by Claude Code workflows, not a paid API
**Decided by:** Shresth.
- There are no API keys and no `.env` for seeds.
- Half the writing agents see `docs/TASTE.md`; half see only the DSL spec and the brief.
- Seed generation runs in batches, within his Claude plan's usage limits.

## D-019 (2026-10-31): the base-model test (junk-rate) v0, pre-registered before its first run
**Decided by:** Claude. Shresth delegated the order of work ("Feel free to decide how to proceed next") after asking why
no GRPO had run.
- **Why now:** the plan dates this test Nov 7–8. It runs early, on the dev palette (palettes/dev_gu.yaml), as an
  exploratory v0. The official Phase 1 gate (≥ 60 % pass and ≥ 30 % "fine" by ear) is re-run on the frozen palette
  (D-017) with the same settings. v0 also decides whether GRPO run 1 starts from the plain base model. The GRPO
  smoke (`jobs/grpo_smoke.sbatch`) runs in parallel on the plain model. It only tests the training stack, so it
  decides nothing here.
- **Prompt:** `lofi_rl.prompt` v1, template sha256 `7aceb2bc0d7339a7d3c9af7bf0d808ee156d4f7a8684e0393d923940fa5c1e68`
  (tests/test_prompt.py pins it). It holds:
  - a grammar card from docs/DSL.md and the palette's full vocabulary;
  - per mood, the chords all in its mode, and one example loop (example.ls without `sax:`);
  - the brief as scene + energy + mood only, never tempo, tonic or mode.
- **Run settings:**
  - model: Qwen3-4B-Instruct-2507 @cdbee75f;
  - briefs: `brief.sample_briefs(2026, 500)`, one completion each;
  - sampling: T 1.0, top_p 1, top_k 0, max_tokens 512 (taste_pipeline §6);
  - vLLM: bf16, eager.
- **Readout** (summary/junk.json):
  - **pass:** gates G1–G6 plus compile, with its Wilson 95 % CI;
  - **pass_not_copy:** pass, counting a near-copy of the in-prompt example as a failure (character 5-gram Jaccard
    ≥ 0.8). The ≥ 60 % bar reads pass_not_copy, so copying the example cannot carry it;
  - also: failures per gate, repairs and P_fix, token lengths and truncations.
- **What follows from it:**
  - **pass_not_copy ≥ 60 %:** GRPO run 1 starts from the base model.
  - **Below that, with syntax failures (G1, G2, G5) the main cause:** grammar-constrained decoding comes first (D-016).
  - **Below that, with other causes:** the prompt gets fixed as v2 and re-registered here before a re-run.
- **Outcome, 2026-10-31** (`lofi-junk-rate` 13632676, A40 node b11-13, mirror `63b208a`, 86.4 SU; `runs/junk-13632676/summary/junk.json`):
  - pass 98/500 = 19.6 % (Wilson 95 % CI 16.4–23.3 %); **pass_not_copy 20/500 = 4.0 %** (2.6–6.1 %), far under the 60 % bar.
  - Failures: G1 389, G6 11, G2 2; G3–G5, compile and crashes 0. The commonest G1 details are a per-bar edit given two grids
    (`@4 hat takes at most 1 grid(s)` 100, `@6 hat` 78, `@6 bass` 56, …). Grids of the wrong length are common too
    (pad_grid in 316 loops, trunc_grid in 146).
  - 78 of the 98 passing loops are near-copies of the in-prompt example. By mood, wistful (the example's) passes 83/183,
    melancholy 8/152 and hopeful 7/165.
  - **Per this entry's rule, syntax failures are the main cause, so grammar-constrained decoding comes first (D-016).** The
    copying is a second, separate cause. If it still sinks pass_not_copy under constraints, the prompt becomes v2,
    re-registered here before its run.

## D-020 (2026-10-31): the base-model test v1, with grammar-constrained decoding, in two prompt arms, pre-registered before either run
**Decided by:** Claude, under D-019's delegation, following D-019's rule for its outcome.
- **Why:**
  - v0 failed mostly on syntax (G1 389 of 500), so D-019's rule puts grammar-constrained decoding first (D-016).
  - Copying is the second cause: 78 of v0's 98 passes were near-copies of the in-prompt example. Prompt v2 is the fix D-019 named for it.
  - Both arms run now, from one mirror, so one round trip answers both questions. Arm A measures the grammar alone; arm B adds v2.
- **Arms:** both `--constrained`, both on palettes/dev_gu.yaml.
  - **A:** prompt v1 (`lofi_rl.prompt`, template sha256 `7aceb2bc0d7339a7d3c9af7bf0d808ee156d4f7a8684e0393d923940fa5c1e68`, as in D-019).
  - **B:** prompt v2 (`lofi_rl.prompt_v2`, template sha256 `127e3094fd5a2183fb952059c2221ad018bdef6d062f51a80b14367b8064466e`).
    - Each brief is shown one complete example: the loop written for the brief's own energy and the next mood (melancholy → wistful → hopeful → melancholy), labelled with that loop's own energy and mood.
    - mid/melancholy briefs get exactly v1's messages, which makes them a control cell inside arm B.
- **Grammar:** `lofi_rl.grammar.loopscript_regex(dev_gu)`. On the laptop it is 3,395 characters with sha256 `1412f11f9ebdc520b50c263eda14be12829951357c17f43496ab2ddf5235e64e` (docs/GRAMMAR.md). Each run records:
  - the sha it built;
  - the backend: xgrammar is requested, and the record says whether LLM() took the pin and what the engine reports.
- **Run settings,** D-019's, unchanged:
  - Qwen3-4B-Instruct-2507 @cdbee75f;
  - `brief.sample_briefs(2026, 500)`, one completion each;
  - T 1.0, top_p 1, top_k 0, max_tokens 512;
  - vLLM in bf16, eager mode.
- **Jobs:** `make submit JOB=jobs/junk_rate.sbatch ARGS='500 2026 1 v1'` (A) and `ARGS='500 2026 1 v2'` (B). Job type sha `4a9d8685231fd573`, at most 450 SU each.
- **Gate, unchanged from D-019:** pass_not_copy ≥ 60 % in an arm. A near-copy is character 5-gram Jaccard ≥ 0.8 with the example that row's own prompt showed; for arm A that is v1's one example, as in D-019.
- **Reported, not gated** (junk.json `copy` and `by_cell`):
  - quasi-copies (containment ≥ 0.85 in the shown example) and pass_not_quasi_copy;
  - `copy.any`: J ≥ 0.8 with any of the 9 examples;
  - the per-cell table, the knob readouts, the verbatim lines and the grid share.

  The judge that chose v2 predicted from a v0 proxy that both arms would fail if quasi-copies counted. Adding them to the bar after that prediction would move it, so they stay out. Copying under the 0.8 line is the job of the reward's copy penalty (taste_pipeline §5: 0 at J 0.6, rising linearly to 0.3 at ≥ 0.8), which is added before GRPO run 1.
- **Validity:** a constrained arm counts only if every completion that ended by stop is in the regex's language (junk.json `constrained.in_language`, with a matching sha). A stop completion outside it means the engine did not apply the grammar. That arm's numbers are then reported but decide nothing, and it is re-run after a fix.
- **What follows:**
  - **Both arms pass:** GRPO run 1 starts from the base model with the arm with the higher pass_not_copy. If the arms' Wilson 95 % CIs overlap, the higher pass_not_quasi_copy decides. If those CIs overlap too, v2 wins, because each brief sees an example at its own energy (knob obedience is a pre-registered claim, CLAUDE.md §9).
  - **One arm passes:** GRPO run 1 uses that arm.
  - **Neither passes:** the largest of three counts, taken over both arms together, sets the next step. That step is registered here before its run.
    - **Passes that are near-copies:** prompt v3. Each brief is shown two examples at its own energy, from the two other moods.
    - **G6 failures** (the regex does not count phrase lengths): prompt v3 with one more card line, "The items between two | marks add up to 16 steps: h.5 er e4 is 12 + 2 + 2."
    - **All other failures** (G1–G5, compile, crash): the SFT warm-start, CLAUDE.md §9's next fallback. It is drafted and taken to Shresth before any job.
  - **Either way:** the official Phase 1 gate still re-runs on the frozen palette (D-017) with the chosen settings, together with the ≥ 30 % "fine" ear check.
  - **Before GRPO run 1:** a prompt-version key in the GRPO config, the copy penalty, and the backend pin and in-language count on the GRPO path. Today that path records only the regex's builder, field, sha and length.
- **Amended 2026-10-31, before either run.** These are the findings of two independent checkers (workflow `wf_ce8fc507-f15`), applied under D-019's delegation.
  - **What changes:** these bullets replace the original Decided-by, Jobs, Gate, Validity, What-follows and Before-GRPO bullets, and the sentence about the judge's proxy prediction. The arms, prompts, regex and run settings are unchanged.
  - **Decided by:** Claude, under D-019's delegation.
    - This supersedes D-019's last sentence. v2 runs now, beside v1 and with both under the grammar, instead of only after copying is shown to sink pass_not_copy under constraints.
    - Arm A alone is the test D-019's rule asked for. Arm B can win only by the rule below.
  - **Sources for the Why numbers:** runs/junk-13632676/summary/junk.json: `failures.G1` is 389 of 500, and `near_copy.count_passing` is 78 of `pass.count` 98.
  - **Control cell:** mid/melancholy holds 57 of the 500 briefs. It is a readout only: its `by_cell` row is reported for both arms side by side and changes no decision.
  - **Jobs:** both arms run from one `make sync` of the commit that holds this amendment. They run on an A40 in the `gpu` partition, as v0 did, with `OVR='-p gpu'` and the job's own `#SBATCH` lines (sha `4a9d8685231fd573`, at most 450 SU each).
    - Arm A: `make submit JOB=jobs/junk_rate.sbatch OVR='-p gpu' ARGS='500 2026 1 v1'`.
    - Arm B: `make submit JOB=jobs/junk_rate.sbatch OVR='-p gpu' ARGS='500 2026 1 v2'`.
    - No `make sync`, from any tree, runs from the first submission until both jobs have ended. A job reads the mirror when it starts, and score and render start new processes later in the job.
  - **Gate:** `pass_not_copy.count` ≥ 300, which is 60 % of 500, in a valid arm.
  - **Validity:** an arm counts only if its `summary/junk.json` shows all of the following.
    - **Complete:** `n` = `requested` = 500, `partial` false, `unreadable_lines` 0, and `generate.stopped` null. score divides by the rows it read, and a run stopped by its deadline still ends `JUNK_RATE_OK`.
    - **As registered:**
      - `generate.model.revision` is `cdbee75f17c01a7cc42f958dc650907174af0554`, `briefs.seed` is 2026, and `sampling` equals runs/junk-13632676's;
      - `palette.sha256` is `b2e4a340b9387e49f896e235c0f1f7f3c9339308ddc307b343c2d96e9eb8fab9`, with `palette.matches_generate` true;
      - `prompt.version` and its template sha are the arm's, `prompt.template_matches_generate` is true, `prompt.rows_matching` is 500, and `prompt.example_mismatch` is 0.
    - **Constrained:**
      - `constrained.enabled` is true;
      - `constrained.regex_sha256` and `constrained.in_language.regex_sha256` are both `1412f11f9ebdc520b50c263eda14be12829951357c17f43496ab2ddf5235e64e` (tests/test_grammar.py pins it);
      - `constrained.in_language.matches_generate` is true;
      - `constrained.in_language.stop.out_of_language` is 0, and `constrained.in_language.other.n` is 0.

      A stop completion that does not fullmatch the regex under Python's `re` means the engine did not apply this regex as `re` reads it: no grammar, another regex, or a dialect or detokenization difference.
    - **What the regex guarantees:** `failures.G1`, `G3`, `G4`, `G5` and `crash` are all 0 (docs/GRAMMAR.md §4). A non-zero count is a defect in the regex, the parser or the checker, not a model failure.
    - **One mirror:** in both arms, `generate.host.stamp` and `score_stamp` show the same `commit=… dirty=0`, which is the one sync's commit. Each job log's stamp lines show it too.
    - **Not part of validity:** `constrained.backend` (the pin, whether `LLM()` took it, and what the engine reports) is recorded and decides nothing. The render step and G7 do not affect validity either.
  - **An invalid arm** is reported and decides nothing.
    - **A scoring defect** (a crash, or the checker): it is fixed, and the arm is re-scored from its own `completions.jsonl` with no new generation.
    - **Anything else:** it is fixed, and the arm is re-run once. The fix, and any sha it changes, is recorded here as an amendment before the run. A fix that changes the regex, either prompt, the palette or junk.py's scoring re-runs both arms.
    - **Invalid twice:** the arm goes to Shresth.
    - **When the rule applies:** the rule below is read only once both arms are valid. A re-run or a re-score stands in place of its arm.
  - **What follows,** with both arms valid. Every count is read from the arm's junk.json.
    - **Overlap:** two `ci95` intervals overlap when the larger lower bound is at most the smaller upper bound, as junk.json prints them (4 decimals). Equal counts always overlap.
    - **Both arms pass:** GRPO run 1 starts from the base model with the grammar and the winning arm's prompt. The winner is decided in this order:
      - if the `pass_not_copy.ci95` intervals do not overlap, the higher `pass_not_copy.count` wins;
      - otherwise, if the `copy.pass_not_quasi_copy.ci95` intervals do not overlap, the higher `copy.pass_not_quasi_copy.count` wins;
      - otherwise v2 wins.

      The last step is a judgment, not a measurement. v2's example always shares the brief's energy but never its mood; v1's example shares the energy only for the 166 mid briefs, and the mood for the 183 wistful briefs.
    - **One arm passes:** GRPO run 1 starts from the base model with that arm's prompt and the grammar.
    - **Neither passes:** the lead arm is the one with the higher `pass_not_copy.count`, or v2 on a tie. Three disjoint counts are read from the lead arm alone. Together with `pass_not_copy.count` they sum to 500, and G7 is in none of them.
      - (a) `near_copy.count_passing`, passes that are near-copies: next is prompt v3a, the shared card with each brief shown two examples at its own energy, from the two other moods. junk.py records one example per row today, so that changes first.
      - (b) `failures.G6` + `failures.compile`, structure the regex does not check: next is prompt v3b, the lead arm's prompt plus one card line aimed at the commonest detail in `top_details.G6` and `top_details.compile`. For phrase lengths ("phrase A is N steps (needs 64, at most 16 off)"), the line is "Each bar of a phrase adds up to 16 steps, before the first |, between two | and after the last |: h.5 er e4 is 12 + 2 + 2."
      - (c) `failures.G2`, truncation, the one D-019 syntax class a valid arm can still have: next is the SFT warm-start (D-016, CLAUDE.md §9). It is drafted and taken to Shresth before any job.
      - **Choosing:** the largest count sets the next step, which is registered here before its run. A tie goes to the earlier of (a), (b) and (c).
      - **Limit:** prompt v3 is the last prompt-only step. If it also fails the bar, the SFT warm-start comes next.
    - **Whichever settings pass, here or after the next step:** the official Phase 1 gate re-runs them on the frozen palette (D-017), with the ≥ 30 % "fine" ear check.
      - The regex is rebuilt from that palette and gets its own sha.
      - If that palette refuses a v2 example, that run's prompt is registered here first.
      - The 50 ear-check clips come from loops that pass and are not near-copies. junk.py's render step changes with this amendment; 41 of v0's 50 clips were copies (runs/junk-13632676/summary/clips.json).
  - **Quasi-copies** stay out of the bar, which stays D-019's: changing it after the v2 design and its estimates were made would move it. This replaces the sentence about the judge's proxy prediction, which no repo file records.
  - **Before GRPO run 1:**
    - a prompt-version key in the GRPO config;
    - the copy penalty (taste_pipeline §5), scored against the seed pool and every example the run's prompt version can show. For v1 that is its one example; for v2 it is all 9 of `prompt_v2.reference_texts`, since across its prompts the policy sees all 9;
    - the backend pin and the in-language count on the GRPO path, which today records only the regex's builder, field, sha and length;
    - taste_pipeline §6's rule for the grammar lever, applied to the chosen prompt. The grammar is on only if that prompt's step-0 parse failures without it exceed 20 % (v1: G1 389 of 500 in D-019; v2: measured by neither arm), and off if the mean |log IS ratio| exceeds 0.05. If it ends up off, this entry's constrained result does not carry over, and the start of run 1 is registered here again;
    - a GRPO smoke with the grammar and the chosen prompt.
  - **Code changed with this amendment,** before the one sync:
    - submit.sh checks the mirror's own `.sync-stamp` on the cluster;
    - generate's deadline forecast reads the chunks after the first, which also pays the grammar compile;
    - junk.json records `score_stamp`;
    - render samples only passing loops that are not near-copies;
    - tests/test_grammar.py pins the dev_gu regex.
- **Second amendment, 2026-11-01, before either run.**
  - **Why:** both jobs (13640324, 13640325) have been PENDING (Priority) since 20:06 PDT on Oct 31. At 07:32 PST on Nov 1, 28 A40 jobs ranked above them. The no-sync rule was blocking every other cluster job, so it is replaced by a code-path rule with the same purpose: both arms run the registered code.
  - **The frozen path:** 60 files, listed with their git blob shas in `docs/prereg/D-020_junk_path.txt` (taken from `d497684`).
    - The import closure of junk's generate, score and render steps: 51 modules.
    - `scripts/job_env.sh`, `scripts/_kit_env.sh` and `kit.env`.
    - `palettes/dev_gu.yaml` and `tests/fixtures/loops/example.ls`.
    - The GeneralUser GS SoundFont.
  - **What changed on that path since `49e2f5a`:** two files, both reviewed here. Neither changes what generate or score computes.
    - `src/lofi_rl/smoke.py` adds the `AUDIOBOX` constant. junk reads only `QWEN`, which is unchanged.
    - `src/lofi_rl/render/fluid.py` kills a FluidSynth stem render after 120 s. A normal stem takes 1–3 s, and the render step and G7 decide nothing here.
  - **Syncs** are allowed while the jobs wait or run, but only if every frozen file is unchanged. `make sync` runs `scripts/check_frozen.sh` first whenever `.sync-freeze` names the list. `.sync-freeze` is deleted when both jobs have ended.
  - **The One-mirror clause is replaced:**
    - in each arm, `generate.host.stamp` and `score_stamp` show the same `commit=… dirty=0`;
    - that commit's frozen files equal the list, which is checked on the laptop with git;
    - the two arms may run from different commits.
  - **Unchanged:** the hardware (an A40 in the `gpu` partition) and every other rule above.
- **Outcome, 2026-11-01.** Arm A: `lofi-junk-rate` 13640324, A40 b09-09, 93.0 SU. Arm B: 13640325, A40 b10-11, 99.0 SU. Results: `runs/junk-1364032{4,5}/summary/junk.json`.
  - **Validity:** both arms meet every condition of the amended entry: complete, as registered, constrained (500/500 stop completions in the language), the regex's guaranteed gates at 0, and one mirror. An independent checker (workflow `wf_5586783e-d6b`) reproduced both from the raw files with its own copy check and its own Wilson intervals.
  - **The sync:** both arms ran from one sync, of `e636485` (07:38 PST on Nov 1), a commit that holds both amendments.
    - The second amendment's code-path rule replaced the first amendment's single sync of `49e2f5a`.
    - The 60 frozen files at `e636485` equal the list. They differ from `49e2f5a` only in the two changes the second amendment reviewed.
  - **Arm A (v1):** pass 494/500; pass_not_copy 355/500, ci95 [0.6687, 0.7481]; G6 6.
  - **Arm B (v2):** pass 500/500; pass_not_copy 446/500, ci95 [0.8617, 0.9163]; no gate failures.
  - **Verdict:** both arms pass. Their pass_not_copy intervals do not overlap (0.8617 > 0.7481), so arm B wins on its count. **GRPO run 1 starts from the base model with prompt v2 and the grammar**, once the Before-GRPO-run-1 list is done. The quasi-copy tier and the closing judgment were not needed.
  - **Readings fixed now, before any further measurement:**
    - **The official Phase 1 gate** re-runs the chosen settings, v2 with the grammar, on the frozen palette.
    - **The grammar lever** (taste_pipeline §6) for v2:
      - "Step-0 parse failures without the grammar" means G1 + G2 + G5, D-019's syntax class. They are counted in a 500-brief junk-rate run of v2 without the grammar, on D-019's briefs, seed and sampling (`ARGS='500 2026 0 v2'`, `-p gpu`, an A40).
      - The grammar stays on for GRPO only if that count exceeds 100, which is 20 %.
      - "Mean |log IS ratio|" means the mean, over every completion token of every step, of |log π_vLLM(token) − log π_trainer(token)|, in a GRPO smoke with the grammar and v2. TRL's own metric may stand in for it only if its definition in the TRL 1.17.1 source is checked to be this one. The grammar goes off if the mean exceeds 0.05.
      - If either reading turns the grammar off, the start of run 1 is registered here again.
- **Grammar lever, the parse-failure half, 2026-11-01:** `lofi-junk-rate` 13656050 (`500 2026 0 v2`, A40 b09-11, 80.7 SU; `runs/junk-13656050/summary/junk.json`). It ran complete, as registered and without the grammar, from mirror `7f1328d`, whose 60 frozen files equal the list.
  - **G1 + G2 + G5 = 397 of 500 (79.4 %)** (G1 397, G2 0, G5 0), above the 100 the rule names. **The grammar stays on for GRPO** on this half.
  - Without the grammar, v2's pass is 78/500 and pass_not_copy 64/500. The commonest G1 detail is a per-bar edit given two grids (`@3 rim` 96).
  - **The other half:** the mean |log IS ratio| ≤ 0.05 is read from GRPO smoke 13657200 (grammar + v2). Its code path is frozen in `docs/prereg/D-020_grpo_smoke_path.txt` until it ends.
- **Grammar lever, the IS-ratio half, 2026-11-01:** `lofi-grpo-smoke` 13657200 (`configs/grpo_smoke_v2_grammar.yaml`: prompt v2, the grammar, the copy penalty), L40S b18-16, 243.0 SU, `GRPO_SMOKE_OK`. Results: `runs/grpo-smoke-13657200/summary/smoke.json`.
  - **The grammar was applied:** 480 of 480 stop completions are in the regex's language. xgrammar was pinned and read back on TRL's engine (`trainer.vllm_generation.llm`).
  - **TRL 1.17.1's `sampling/sampling_logp_difference/mean`** was checked in its installed source (`trainer/grpo_trainer.py`, sha256 `28f815d8…`, lines 2980–2989). It is the mean over completion-masked tokens of |old_per_token_logps − sampling_per_token_logps|, which is D-020's definition on the trainer's side.
  - **Mean |log IS ratio| = 0.1538** (token-weighted over 15 steps of both launches; per step 0.105–0.257). That is above 0.05. Smoke 13632679, v1 without the grammar, had 0.0044–0.0091 per step.
  - **Caveat** (the smoke's own): vLLM's log-probs were not shown to be taken after the grammar's mask. If they are taken before it, the true value is higher, not lower.
  - **Verdict:** by this entry's rule, **the grammar goes off for GRPO run 1**. The parse-failure half (397 > 100) would keep it on, but the IS half turns it off. So this entry's constrained result does not carry over, and **run 1's start is registered again**, in a later entry, before any run-1 job.

## D-021 (2026-11-02): run 1's start, registered again: keep the grammar by training the grammar-masked policy, decided by one smoke; the SFT warm-start only if that fails

**Decided by:** Shresth, 2026-11-02; proposed by Claude (drafted 2026-11-01, workflow `wf_76128e23-f1d`). Asked "Which start should run 1 take?" after a plain-words overview of A (make the trainer apply the grammar's rules, keep the grammar, one smoke on the registered smoke type at most 500 SU, Part B only if it fails) and B (the SFT warm-start first), with what a yes accepts (step 0 copies the example's sax lines in 421 of 500; held-out briefs are fresh samples of the same 9 prompts), he answered: "A: fix first".
- Nothing here runs before his answer. The entry is committed, with his words, before its first job.
- D-020's IS verdict sent run 1's start to a later entry. The CHANGELOG (2026-11-01, 19:30) promised him the SFT warm-start draft before any job.
- This entry puts one cheaper step in front of that draft. It keeps the draft as the fallback (Part B), and keeps D-016's order after it.
- **What this entry supersedes, once he says yes.** Entries are never edited; a later one supersedes.
  - D-020's IS-half verdict ("the grammar goes off for GRPO run 1"), but only if Part A passes. If Part A misses, that verdict stands.
  - D-020's Before-GRPO clause that scores the copy penalty "against the seed pool". Item 2 narrows the pool.
  - taste_pipeline §3's held-out seed (596) and its `data/briefs/briefs_v1.jsonl` split. Item 1 replaces both. `data/` is never committed, so that file could not hold a pre-registration.

- **Why**
  - **D-020's two readings for v2.**
    - Without the grammar, G1 + G2 + G5 = 397 of 500 (`runs/junk-13656050/summary/junk.json`). That keeps the grammar on.
    - With it, the GRPO smoke's mean |log IS ratio| was 0.1538 (`runs/grpo-smoke-13657200/summary/smoke.json`), above 0.05. That turned it off.
  - **The gap comes from the mask alone.**
    - Step 1's rollouts come before any update, and `learning_rate` is 0.0 at step 1 (`a/steps.jsonl`). So vLLM and the trainer held the same weights.
    - The gap there was 0.1146. At step 1 of the unconstrained smoke 13632679, it was 0.0087.
    - So vLLM's log-probs carry the grammar's mask, and the trainer's do not. This conclusion rests on those two step-1 gaps.
    - 13657200's `run.json` `vllm_logprobs` only lists lines. `apply_grammar_bitmask` is at `gpu_model_runner.py:4335`. `_sample` is at 3437, with its sampler call at 3448. `processed_logprobs` is at `sampler.py:270–273`, and TRL asks for it at `vllm_generation.py:370`.
    - The record says `checked: false` and does not show the call order. V1 and M10 check it.
  - **What that did to training.** TRL 1.17.1 weights each sequence by exp(Σ old − sampling) and masks weights above 3.0 (`a/run.json`: `vllm_importance_sampling_correction` true, `sequence_mask`, `clip_max` 3.0). The exact form is UNVERIFIED until check T3. From `{a,b}/steps.jsonl` of both smokes:

    | per step | with the grammar (13657200) | without (13632679) |
    |---|---|---|
    | mean IS weight | 0.0010–0.0814 | 0.698–1.027 |
    | max IS weight ÷ (32 × mean) | 0.319–1.000, and ≥ 0.9 in 7 of 15 steps | 0.061–0.112 |
    | grad norm | 0.00088–0.062 | 0.059–0.221 |
    | IS gap | 0.105–0.257 | 0.0044–0.0091 |

    - The second row is the largest weight's share of the step's 32, if TRL's logged mean is per sequence. T3 checks this; no value exceeds 1.000.
    - Read that way, one completion carried at least 90 % of the weight in 7 of the 15 grammar steps, and 32–87 % in the other 8.
  - **The cause can be removed.**
    - If the trainer masks its own logits with the same grammar, π_trainer is the masked distribution vLLM samples from. GRPO is then on-policy for the policy that is played.
    - taste_pipeline §6's rule exists to catch exactly this ("the masked sampler mismatches the trainer").
    - This entry changes the trainer, not the rule:
      - D-020's definition and its 0.05 threshold are kept;
      - the reading is registered before it is taken;
      - a measured miss leaves D-020's verdict in force.
    - What differs from 13657200 (CLAUDE.md §6): the trainer masks its logits in all three passes.
  - **The order (D-016):** grammar first, then the SFT warm-start, then edit-mode rollouts, then stop.
    - This entry stays on lever 1, registers lever 2 as the fallback, and names lever 3 after it.
    - Trying lever 1 first costs the SFT path no time.
      - An SFT checkpoint that can start run 1 must be built with the frozen palette's names. So none can exist before the freeze, which is planned for Nov 6 (docs/SHOOTOUTS.md).
      - The 30 drafts fullmatch `draft_day2`'s regex 30 of 30, and dev_gu's only 3 of 30 (laptop count).
    - The masked smoke can run now on dev_gu.
  - **Why not an SFT first.**
    - **A self-distilled corpus would teach copying at the part level.**
      - Arm B of D-020 has 445 candidates: passing, no sax fold, J_any < 0.8.
      - Of these, 367 carry the shown example's A or B line verbatim (339 both), and 86.6 % of their grids are the example's (laptop count over `runs/junk-13640325/summary/scored.jsonl`).
      - Only 78 of the 500 copy neither line and sit under J_any 0.8, and 50 of those are in one cell, mid/wistful.
    - **A Claude-written corpus avoids that.** But it needs about 180 new loops under global constraints, and the model risks memorizing about 21 targets per prompt.
    - **Both redefine step 0** for D-009 and D-014, which he registered when step 0 was the base model.
  - **What this step 0 is.** These are laptop counts over arm B (`runs/junk-13640325/summary/`). He reads them before he answers.
    - **It copies the shown example's sax lines.**
      - A or B is verbatim in 421 of 500 loops, and both in 392.
      - 7,287 of 8,319 grids are the example's.
      - pass_not_copy (446) counts only whole-loop near-copies, so these loops pass.
    - **It is already near-collapsed inside a cell.** The nearest same-cell neighbour sits at J5 median 0.9377, with 448 of 500 at 0.8 or more.
    - **A cell is one prompt.**
      - Every brief of a cell gets the same messages. So "different briefs should give different songs" (TASTE.md, Variation) rests on sampling alone.
      - Nothing in the reward sees two briefs of one cell together: P_group sees one brief's 8, and item 2 keeps model output out of the pool.
    - **D-009 cannot see a collapse inside a cell.**
      - Suppose every cell's loops were replaced by that cell's step-0 medoid. The mean 1 − J5 across 100 random arm-B briefs would still keep 0.938–0.963 of its step-0 value (200 draws).
      - Only 11.1 % of the pairs share a cell, and D-009's bar is 0.70.
      - This is a symbolic proxy for D-009's embedding distance (an estimate). Audio also differs by tempo and tonic, which would hide a collapse further.
    - **What this entry does about it:** items 3–8 add the readouts and alarms this calls for. D-009, D-014 and the obedience claim are his, and stay as registered.
  - **Data** (laptop counts, 2026-11-01):
    - `runs/junk-13640325/summary` (arm B: v2 + grammar, dev_gu), `runs/junk-13656050/summary` (v2 without the grammar);
    - `runs/grpo-smoke-13657200/summary` and `runs/grpo-smoke-13632679/summary`;
    - `drafts/v1/`.
    - Raw completions in the regex's language: 26 of 500 without the grammar, 500 of 500 with it.

- **Part A: the masked trainer** (`src/lofi_rl/rl/`; docs/GRPO.md gains a section)
  - **The config key:** `trainer_logprobs: grammar_masked`. It is refused without `structured_outputs`, and the dry run and `run.json` record it.
  - **The override:** `LofiGRPOTrainer(GRPOTrainer)` overrides one method, `_get_per_token_logps_and_entropies`.
    - In every forward of the policy, the logits at completion positions get a large finite negative outside the grammar's allowed set, written out of place. That covers three passes:
      - the old pass, which feeds the IS weight and D-020's metric;
      - the reference pass, with the adapter off;
      - the loss pass.
    - Prompt rows, rows after the first EOS or pad, and the extra last row are left alone.
    - The fill is finite, not −inf, so TRL's entropy (Σ p·log p) stays finite.
    - **T2's alternative.** T2 may show that TRL's own `sampling_mask` is a keep-mask applied before the log-softmax in all three passes. If so, the grammar mask is ANDed into it instead.
      - `run.json` names the path that ran.
      - M1's tests cover both paths.
    - It also logs the step's 32 per-sequence IS weights, from the old and sampling log-probs (`is/seq_weights`), for M11.
  - **The mask is xgrammar's own** (0.2.8, `requirements/policy.lock.txt`).
    - A `GrammarMatcher` is compiled from the same regex string. Its `TokenizerInfo` is built the way vLLM 0.32's backend builds it: vocabulary size and stop ids, read at run time from the installed `v1/structured_output/backend_xgrammar.py`.
    - The matcher is replayed along each completion's sampled ids, so the trainer applies the bitmask vLLM applied, by construction.
    - The API names are UNVERIFIED until that run-time read.
  - **An independent check:** `src/lofi_rl/grammar_mask.py`, standard library only. It is a Thompson NFA of the regex over the byte-level vocabulary, with allowed sets memoized by state set.
    - **On the cluster**, it is built from the pinned snapshot's `tokenizer.json`, with the vocabulary size and stop ids that V2 reads. So it and xgrammar see one vocabulary.
    - **On the laptop** (2026-11-01, with the cached `mlx-community/Qwen3-4B-4bit` `tokenizer.json`, as in GRAMMAR.md §3.3), it accepts every token of the canonical re-encodings of all 980 completions of arm B and smoke 13657200. There are 0 violations.
    - These laptop numbers estimate xgrammar's bitmask; they are not it.
      - At most 52 tokens are allowed at any position (median 20–21), and 7.3–7.5 % of positions are forced.
      - A cold 32-completion step takes 0.13 s on the laptop, and a warm one 0.013 s.
    - **Canonical re-encoding** gives vLLM's token count for only 361 of the 500 arm-B completions, and for 227 of 320 and 107 of 160 in the smoke's two launches. So the sampled ids themselves are checked on the cluster (M8).
  - **A sampled token outside the mask** stops the launch before the loss sees it.
  - **Source checks before the model loads** (`trl_probe`, AST reads). They fail closed with exit 2, after about 2 minutes.
    - T1: the old, reference and loss log-probs come only from `self._get_per_token_logps_and_entropies` (`use_liger_kernel` false).
    - T2: that method's body. It runs one forward per micro-batch, keeps rows in order, takes `logits_to_keep + 1` and drops the last row, then applies the temperature and the log-softmax. T2 also reads what `sampling_mask` is, and checks that it is None at top_k 0 and top_p 1.
    - T3: the sequence_mask weight, and whether `sampling/importance_sampling_ratio/mean` is a mean over sequences or over tokens.
    - T4: the KL comes from the reference and current log-probs; `use_bias_correction_kl` is recorded.
    - T5: lm_head is not a LoRA target.
    - V1: in the method that holds `gpu_model_runner.py:4335`, `apply_grammar_bitmask` changes the logits before `self._sample` runs. Also, `_sample` reaches the `processed_logprobs` branch at `sampler.py:270–273`, which TRL asks for. V1 is recorded, and M10 is its test.
    - V2: how `backend_xgrammar.py` builds its `TokenizerInfo` (vocabulary size and stop ids) and compiles the regex.
  - **Per-step records:**
    - `mask/violations` and `mask/rows`;
    - `mask/neglogz`, the token-weighted −log Z in the old pass;
    - `mask/forced_share` and `mask/nfa_mismatch`;
    - `is/seq_weights`.
  - **The smoke:** `configs/grpo_smoke_v2_grammar_masked.yaml`, which is `configs/grpo_smoke_v2_grammar.yaml` plus that one key (`tests/test_rl_train.py` pins it).
    - It runs on the unchanged `jobs/grpo_smoke.sbatch`. That job takes any `configs/grpo_smoke*.yaml`, so its registered sha `1dded12d386c7fab` stands.
    - It runs on an L40S, as 13657200 did, so the numerics compare.
    - Its code path is frozen in `docs/prereg/D-021_masked_smoke_path.txt`, with `.sync-freeze` set until it ends.

- **Settings:** D-020's run-1 settings, plus `trainer_logprobs: grammar_masked`.
  - Model: Qwen3-4B-Instruct-2507 @`cdbee75f`; LoRA r 16, α 32, dropout 0, all-linear.
  - Optimizer: lr 1e-5, constant after 3 warmup steps.
  - Batch: 4 prompts × 8 completions; `num_iterations` 1; `max_completion_length` 512 with `mask_truncated_completions`.
  - Loss: β 0.04 to the adapter-off base, under the same mask; dapo; `scale_rewards` batch.
  - Sampling: T 1, top_p 1, top_k 0; TRL's IS correction at its default.
  - Prompt: v2 (template `127e3094…`).
    - D-020's rule holds: if the frozen palette refuses a v2 example, the prompt is registered first.
    - `draft_day2` refuses none of the 9 (laptop check).
  - Grammar: `loopscript_regex(palette)` with xgrammar pinned. The smoke uses dev_gu's `1412f11f…`; run 1 uses the frozen `lofi_rain_v1`'s own sha.
  - P_copy: against the 9 v2 references, plus the pool in item 2.

- **For whichever part runs** (grafted from the two SFT drafts)
  1. **Brief streams, fixed now** in `docs/prereg/brief_streams_v1.json`, committed with this entry before any step-0 sample is drawn.
     - **Training:** seed 596 from index 0. Run 1 takes [0, 400).
     - **Measurement:** seed 2026, [0, 500) (junk-rate).
     - **Evaluation:**
       - RM held-out: 40 briefs from seed 5101;
       - dev: 40 from seed 5102;
       - the final A/B: 100 from seed 5103;
       - Part B's dev set: 270 from seed 5104.
       - Seeds 5101–5104 appear nowhere in the repo today (laptop grep).
     - **Stratified by cell.** Each stream takes the first briefs of each cell in stream order, up to per-cell counts the file fixes: 4 or 5 a cell for 40, 11 or 12 for 100, and 30 for 270.
     - **Draw seeds.** The file also fixes the vLLM seed of the k-th draw for brief i of each stream. That covers the step-0 samples, the checkpoint quads and the A/B's ≤ 4 draws.
     - **What "held-out" means.** The model sees only the cell, so all 9 prompts are the same in every stream. Held-out briefs are kept apart only by fresh samples, rendered at tempi and tonics the model never reads.
     - **Twins.** In stream order, some evaluation briefs have the same energy, mood, bpm and tonic as a brief in seed 596's [0, 400) (laptop count):
       - 8 of seed 5101's first 40;
       - 4 of 5102's first 40;
       - 15 of 5103's first 100.
       - The file lists them for the RM's report and excludes none. The policy never reads bpm or tonic, and an exclusion would thin the 75–90 BPM centre.
  2. **The copy penalty's pool holds only loops the policy did not write:** the 9 v2 references, the 30 drafts, and the seed pool's Claude-written loops, mutations and broken copies.
     - It never holds the 660 step-0 samples or any other model output.
     - Why: at step 0, the nearest same-cell neighbour has J at median 0.9377, and 448 of 500 sit at J ≥ 0.8 (laptop count over arm B). A pool holding step-0 samples would put P_copy at its 0.3 cap on nearly every completion.
     - This supersedes D-020's "scored against the seed pool", and narrows taste_pipeline §5's "vs pool".
  3. **Part-level copy readouts, per step,** on the GRPO path: the share of completions whose A or B line equals that line of the shown example, and the share of grids that are the example's (junk.py's `verbatim` and `grid_share`).
     - At step 0 under the grammar (arm B): A or B verbatim in 421 of 500, and 7,287 of 8,319 grids (`scored.jsonl`, `junk.json`).
     - Whole-loop J cannot see a copied sax line over new drums, and that path is already in step 0: 8 of the 52 loops at J5 ≤ 0.6 (P_copy 0) carry a verbatim A or B line (laptop count).
     - **Alarm** (hack-watch, taste_pipeline §7). It fires when, for 3 consecutive 5-step windows, both hold:
       - the verbatim A-or-B share stays at or above its step-0 value;
       - the mean J5 with the shown example falls at least 0.1 below its step-0 value.
       - That is P_copy evaded with the lines kept. For scale, arm B's values are 0.842 and 0.7115.
  4. **Diversity readouts,** in junk.json, in smoke.json, and per 5-step window in run 1. There are two copy forms, named once here and used everywhere in this entry. Arm B's values (laptop count):
     - **junk's form** (canonical text without `sax:`):
       - cell-balanced within-cell mean 1 − J5: 0.2263;
       - pair-weighted mean J5: 0.7743;
       - extra exact duplicates within cells: 34 (466 distinct of 500).
     - **the K/D form** (DRAFTS.md §5: keys and kit written K and D):
       - cell-balanced within-cell mean 1 − J5: 0.2158;
       - pair-weighted mean J5: 0.7839;
       - extra exact duplicates within cells: 44 (456 distinct, 0.912).
     - The K/D form ignores instrument names, so it is the one compared across palettes: in B4, and in run 1 against step 0.
     - In run 1, a window's value compares completions of different briefs of one cell.
  5. **Step-0 values.** Items 3, 4 and 6 read their step-0 values from the official Phase-1 gate run on the frozen palette (Part A) or from B2's run (Part B), with the same seed and settings. Arm B's dev_gu values are for scale only.
  6. **Within-cell collapse alarm** (hack-watch, taste_pipeline §7): the window's within-cell mean J5 across briefs (K/D form) stays above its step-0 value for 3 consecutive 5-step windows.
  7. **Run-1 STOP rules, and what follows a STOP.**
     - **The rules:** any mask violation; a per-step gap above 0.05; an IS-weight mean below 0.3 for 3 steps in a row.
     - **A STOP** ends run 1 at its last checkpoint (one every 25 steps). Nothing resumes until an amendment here names the cause and the fix.
     - **Any of these rules marks a trainer defect.**
       - The masked smoke runs on the fix (at most 500 SU) and must meet M5–M11.
       - Run 1 then resumes from the last checkpoint before the first bad step.
       - If the smoke misses, D-020's verdict applies to run 1, and Part B goes to Shresth.
     - **An alarm of items 3 or 6** follows taste_pipeline §7's On-alarm list.
  8. **Readouts beside the registered gates and claims.** These are reported and never gated. D-008, D-009, D-014 and the obedience claim stay as he registered them.
     - **Beside D-008's gate:**
       - each held-out tuple's pairwise J5 and verbatim-line sharing;
       - agreement on the pairs whose two clips differ in A or B;
       - under Part B, agreement without the clips at J5 ≥ 0.8 to a training target.
       - Under this step 0, about 56 % of held-out tuples would carry both of the shown example's sax lines in all four clips (20,000 random within-cell 4-tuples of arm B; laptop count, an estimate).
     - **Beside D-009's number:** the within-cell mean 1 − J5 (K/D form) and the distinct forms per cell, at step N and at step 0, on the 100 final briefs.
     - **Beside D-014's A/B:**
       - each trial's copy readouts for both arms: J5 to the shown example, to the pool and to any training target, and verbatim A/B;
       - the win rate on the trials whose step-0 clip carries no verbatim line of its shown example;
       - a trial order in which no two consecutive trials share a cell.
     - **Beside the obedience claim:** energy and mood obedience separately, with step 0's values on the same briefs. For scale, arm B on dev_gu matched the energy band in 498 of 500 loops and the mood signature in 333 of 500.

- **Pre-registered gates for Part A**
  - **Two kinds of miss.**
    - **An invalid launch** (M5–M8, or M10's first clause) measured nothing. Its defect is fixed, the fix is recorded here, and the smoke runs again. A launch invalid twice goes to Shresth, as D-020's invalid arms did.
    - **A measured miss** (M9, M10's ratio or M11) on a valid launch gets at most one re-run, with the one fix named now.
      - That fix is the masked log-softmax computed in fp32 inside the override, in all three passes. vLLM's sampler takes its log-probs in fp32.
      - It applies only when M8 holds, so the mask matched and the miss is numerical.
    - **Not named:** `cast_lm_head_to_fp32`. TRL documents it for untied embeddings only (UNVERIFIED here: there is no TRL source on the laptop), and the model's config has `tie_word_embeddings: true` (the cached configs).
  - **M1 (laptop):** `pytest -q` passes on the commit that holds this entry. That includes the new `tests/test_grammar_mask.py` and `tests/test_rl_masked_logps.py`, which cover both T2 paths.
  - **M2 (laptop):** the NFA accepts every token of the 980 re-encoded completions above, with EOS at each end: 0 violations. This checks the NFA on re-encodings only; M8 checks the sampled ids.
  - **M3 (laptop):** at GRAMMAR.md §3.3's four points, the allowed sets are 20 with EOS refused, 20 with EOS allowed, 25, and 8.
  - **M4 (laptop):** the dry run reports regex `1412f11f…`, template `127e3094…` and reward check 0.2784511276657195, as `runs/grpo-smoke-13657200/summary/dry_run.json` does. It also reports `trainer_logprobs: grammar_masked`.
  - **M5 (registration, validity):** this entry and the frozen path are committed and synced before the submit. Both launches' stamps show that commit with dirty=0.
  - **M6 (validity,** `smoke.json`):
    - GRPO_SMOKE_OK;
    - in `in_language`: stop out_of_language 0, other.n 0, and at least one stop completion per launch;
    - xgrammar pinned and applied in both launches.
  - **M7 (validity, sources):** T1–T5 read `match` in both launches' `run.json`; otherwise the launch stops before the model loads. V1 and V2 are recorded.
  - **M8 (validity, fidelity):**
    - 0 sampled tokens outside the trainer's mask, over every completion token of both launches;
    - 96 masked rows per step: 32 old, 32 reference, 32 loss;
    - 0 positions where the NFA's allowed set differs from xgrammar's bitmask, over one vocabulary and one set of stop ids;
    - every logged loss, KL and entropy finite;
    - `run.json` names the T2 path that ran.
  - **M9 (measured): D-020's quantity, with its threshold unchanged.**
    - The token-weighted mean, over every step of both launches, of TRL's `sampling/sampling_logp_difference/mean` (source verdict `match`) is at most 0.05.
    - For reference, 13657200 read 0.1538, and 13632679 read 0.0044–0.0091 per step.
  - **M10: removed rather than reduced.**
    - **Validity:** the trainer's own token-weighted −log Z (`mask/neglogz`) is at least 0.05. 13657200's gap, 0.1538, implies a −log Z of about that size, so a value under 0.05 means the records disagree with that run.
    - **Measured:** M9's value is at most 0.25 times `mask/neglogz`. If vLLM's log-probs came before the mask, the gap would equal −log Z and this would fail. So M10 also settles D-020's caveat.
  - **M11 (measured): the gradient survives TRL's correction** (from `is/seq_weights`).
    - At every step, the mean of the 32 weights is at least 0.3, and the largest is at most 0.25 of their sum.
    - TRL's `sampling/importance_sampling_ratio/{mean,max}` stands in only if T3 shows its mean is over sequences.
  - **M12 (cost; it decides only the run-1 estimate):** mean `step_s` is at most 43.3 s (1.15 × 13657200's 37.63 s). Otherwise the lofi-grpo estimate is redone before the ask. The job's SU is read from `sacct`.

- **What follows**
  - **M5–M11 all pass:** this entry supersedes D-020's IS-half verdict. The grammar is on for run 1 with the masked trainer, and D-020's Outcome (v2 with the grammar) is run 1's setting again. Then, in order:
    1. **D-020's official Phase-1 gate** on the frozen palette: `lofi-junk-rate` with `ARGS='500 2026 1 v2'`.
       - It needs `pass_not_copy.count` ≥ 300 of 500 in a valid run, and at least 15 of 50 clips "fine" by his ear, with his APPROVED line in `gates/GATE_phase1.md`. The bar is D-020's and unchanged.
       - Before he signs, the gate file reports three things beside the bar:
         - the verbatim A-or-B share among the passing non-copies;
         - the same share among the 50 clips;
         - his "fine" count, split by whether a clip carries a verbatim line.
       - This run's readouts become the step-0 values of items 3, 4 and 6.
    2. **The step-0 samples,** all drawn under the grammar, with item 1's streams and draw seeds: the RM held-out tuples, PAR's Ref_b and the seed pool's 660.
    3. **Phase 2's GRPO smoke** (CLAUDE.md §9) meets M5–M11 under the same two-kind rule. It runs with the scorer on, the frozen palette's grammar and run 1's pool.
    4. **Run 1's code path** is frozen as the smokes' paths were, or snapshotted per job (CHANGELOG Status). Then `lofi-grpo` goes to him for his yes.
  - **An invalid launch, twice:** it goes to Shresth with the records. Nothing else on this path runs until he answers.
  - **A measured miss after its one re-run:** D-020's verdict stands, the grammar is off for run 1, and Part B goes to Shresth.
  - **The Phase-1 gate fails on the frozen palette:** it goes to Shresth with CLAUDE.md §9's next fallback, Part B.
  - **Part B fails:** see Part B.

- **Part B: the SFT warm-start** (D-016's lever 2). It runs only if Part A or the Phase-1 gate fails, or if he declines Part A. It is registered in full as an amendment here before any of its jobs.
  - **The corpus is Claude-written. It never holds model output, nor a loop he voted down.**
    - It holds the 10 kept drafts (d01, d02, d03, d07, d09, d04, d06, d08, d05 and d10; CHANGELOG 2026-11-01 13:38), plus 180 new loops, 20 per cell.
    - The 20 rejected drafts stay out of it. They are labeled negatives (TASTE.md, Labeling; taste_pipeline §8: each kept loop ≻ its rejects). They stay in the copy pool and in the RM's labels.
    - The new loops are written as the first seed-pool shards, which Phase 2 needs either way (taste_pipeline §8, D-018; half the writers see TASTE.md).
    - Admission rules:
      - zero repairs;
      - J5 < 0.6 against the 10 references (the 9 v2 examples plus L5, DRAFTS.md §5) and against every other target;
      - no A or B line equal to a reference's or another target's;
      - written in the model's own line-end layout (GRAMMAR.md §3.3).
    - At least 21 targets per cell (22 in low/wistful, which has two draft briefs).
  - **Training:**
    - LoRA r 16, α 32, all-linear, on the exact v2 messages of each target's cell. The loss is on the completion only, up to the end-of-turn token.
    - lr 1e-4, constant after 5 warmup steps; up to 3 epochs, with the adapter saved after each.
    - **The dev bars** are read inside `lofi-sft`, with vLLM and each saved adapter, on 270 completions without the grammar from seed 5104:
      - G1 + G2 + G5 ≤ 27;
      - pass_not_copy ≥ 189;
      - near-copies of a training target ≤ 27.
    - The earliest save that passes the dev bars is merged: an fp32 merge, one bf16 copy and a sha256 manifest. If no save passes, nothing is merged and Part B fails.
  - **Gates on the merged model** (seed 2026 × 500, no grammar, the frozen palette):
    - B1: G1 + G2 + G5 ≤ 100, so the grammar stays off.
    - B2: pass_not_copy ≥ 300, the Phase-1 bar on run 1's own settings.
    - B3, copying:
      - near-copies of any target ≤ 50;
      - completions with an A or B line equal to a target's or a reference's ≤ 125;
      - mean P_copy ≤ 0.0877, half of arm B's 0.1754.
    - B4: diversity no worse than arm B's in the K/D form (item 4): within-cell pair-weighted mean J5 ≤ 0.7839, and distinct forms ÷ passing ≥ 0.912.
    - B5: a GRPO smoke without the grammar meets M9 and M11.
    - B6: at least 15 of 50 clips "fine". The 50 are drawn from loops that pass and are near-copies (J5 ≥ 0.8) of neither the shown example nor any training target.
  - **A miss:** each job gets at most one re-run, after a fix recorded here. Part B fails if no save passes the dev bars, or if any of B1–B6 still fails.
  - **Part B fails:** D-016's lever 3, edit-mode rollouts, is drafted and taken to Shresth before any job. If that fails too, D-016's last step, stop, goes to him.
  - **Step 0** becomes the merged checkpoint, sampled without the grammar. That needs his word at that point (D-009, D-014).
  - **Its jobs** are each at most 500 SU, so Claude registers them:
    - `lofi-sft`: A100, 4 CPU, 32 GB, 25 min (20 SU/min, at most 500 SU), covering training, the dev bars and the merge;
    - `lofi-junk-rate` with a MODEL argument, re-registered at most 450;
    - the registered smoke.

- **Step 0 under Part A**
  - It is the base model (Qwen3-4B-Instruct-2507 @`cdbee75f`, no adapter) with prompt v2. vLLM decodes it under the frozen palette's grammar at T 1, top_p 1, top_k 0 and 512 tokens.
  - That is D-020's own Outcome. Arm B measured it on dev_gu: 500 of 500 pass, and 446 of 500 pass_not_copy.
  - What it copies, and what that means for D-009 and D-014, is under "What this step 0 is".
  - Under the mask, it is also the KL reference.
  - Every checkpoint is decoded through that same grammar. That covers:
    - the dev briefs that pick step N;
    - the 100 final briefs;
    - the obedience check;
    - the across-brief no-collapse distance;
    - the ♥ secondary.
    - Nothing unmasked is ever played to him.

- **Jobs and SU** (CLAUDE.md §4 rates; 1,481.9 SU of the 1,000,000 target used so far, CHANGELOG Status)
  - **The masked smoke:** `lofi-grpo-smoke`, already registered (sha `1dded12d386c7fab`, max 500).
    - Command: `make submit JOB=jobs/grpo_smoke.sbatch OVR='--gres=gpu:l40s:1 --constraint=l40s' ARGS='configs/grpo_smoke_v2_grammar_masked.yaml'`.
    - Shape: L40S, 4 CPU, 32 GB, 25 min, at 20 SU/min, so at most 500 SU a launch.
    - Expected: about 250. 13657200 cost 243.0 SU in 12:09, and the mask adds under 1 s a step (an estimate; the laptop NFA takes 0.13 s for a cold step).
    - At most three launches: the first, one after an invalid launch, and one after a measured miss. That is 1,500 SU at worst.
  - **The Phase-1 gate:** `lofi-junk-rate`, registered (sha `4a9d8685231fd573`, max 450). Its measured runs cost 80.7–99.0 SU, and D-020 already requires it.
  - **Phase 2's GRPO smoke,** with the scorer on: taste_pipeline §6 puts it at about 550 SU on an A100, or 670 on an A40 (UNVERIFIED). That is over 500, so its shape goes to him with the lofi-grpo ask.
  - **PAR's Ref_b** for run 1's 400 briefs: 1,600 step-0 samples under the grammar, rendered and featurized. No job type or estimate exists yet; it goes into the lofi-grpo ask.
  - **Run 1:** `lofi-grpo`, a new type over 500 SU, for his yes later.
    - taste_pipeline §6's estimates (UNVERIFIED):
      - A100: about 2,300 ± 35 %, worst case 3,780 at 1:45;
      - L40S: about 3,200 ± 35 %, worst case 5,400 at 2:30.
    - **From measured parts** (estimates):
      - 13657200's L40S step was 37.63 s, with the symbolic reward.
      - The CHANGELOG (2026-11-01, 15:05) sizes the scorer for 32 passing completions at about 12 s of rendering on 7 workers, plus about 7 s of features. §6 budgets 9 s.
      - That makes about 57 s a step. At 36 SU/min, 100 steps plus 7 min of startup come to about 3,700 SU on an L40S, and about 2,900 on an A100 by the same arithmetic.
      - Both are inside §6's worst cases. M12 and Phase 2's smoke replace these numbers before the ask.
  - **Part B, only if needed:** at most 1,450 SU, or 2,900 with one re-run of each job; about 590–720 expected (an estimate).
  - **New SU this entry adds before run 1 under Part A:** about 250 expected, 1,500 at most.

- **What Shresth decides**
  - **Now, one yes or no:** Part A first, with Part B only if it fails. Part A means the masked smoke, the grammar kept, and step 0 the base model under the grammar.
  - **What a yes accepts.** He can override each item in his own words.
    - **Step 0 copies the example's lines.** It copies the shown example's sax lines in most loops (A or B in 421 of 500, both in 392; arm B).
      - So the final A/B compares step N with loops that mostly carry Claude's example lines, and the two arms may be told apart by ear.
      - Item 8 reports this. D-014 is unchanged.
    - **"Held-out" means fresh samples.** In D-009, D-014 and the obedience claim, it means fresh samples of the same 9 prompts.
    - **D-009 cannot see a collapse inside a cell.**
      - A total per-cell collapse would keep 0.94–0.96 of step 0's across-brief distance (the proxy above).
      - So items 4, 6 and 8 watch it. Making a within-cell readout a claim would need his word.
    - **The defaults this entry sets:**
      - the narrowed copy pool;
      - the brief streams;
      - the alarms and readouts;
      - Part B's corpus, without the drafts he rejected.
  - **If no:** Part B is registered in full as the next entry and comes to him. Nothing runs.
  - **Later:**
    - `lofi-grpo`'s SU, which is over 500, together with Phase 2's smoke shape and Ref_b's job;
    - only if Part B runs, step 0 as the SFT checkpoint.

- **Amended 2026-11-02, before its first job: how Part A applies to the installed TRL, and what the build settled.**
  Claude read the policy env's installed sources on 2026-11-02, copied read-only to the laptop:
  - TRL 1.17.1: `trainer/grpo_trainer.py` (sha256 `28f815d8…`, the file D-020's smoke checked), `trainer/utils.py`
    (`51e0b430…`), `kernels/chunked_logprob.py` (`7f4d852a…`) and `generation/vllm_generation.py` (`123eab02…`);
  - vLLM 0.32.0: `v1/structured_output/backend_xgrammar.py` (`1f9a431b…`), `v1/structured_output/__init__.py`
    (`cfd42513…`), `v1/worker/gpu_model_runner.py` (`4cf95efe…`), `v1/worker/gpu/model_runner.py` (`5240fe8d…`) and
    `v1/sample/sampler.py` (`a67286eb…`);
  - xgrammar 0.2.8 and PEFT.

  The gates, their bars, the two kinds of miss, the named fix, Part B and the order are unchanged.
  - **T2's alternative is the path.**
    - `_get_per_token_logps_and_entropies` never forms logits. It calls the model with `fused_lm_head=True`, and TRL's
      chunked kernel normalizes each token's log-prob over `support_ids`, which the method builds from `sampling_mask`
      (`grpo_trainer.py:1557-1561`; `add_fused_lm_head` in `trainer/utils.py`).
    - The old, reference and loss passes all hand `sampling_mask` to it (`:2770`, `:2854`, `:3168`).
    - So the trainer passes the grammar's allowed set at each completion position as that support, ANDed with TRL's
      own sampling mask, which is None at top_k 0 and top_p 1 (`:1152-1161`). run.json names the path `support_ids`.
  - **T2's checks for that path.** The registered list describes the logits form, which this TRL does not have. For the
    support path T2 checks that the statements are present:
    - the rows in chunk order, and labels −100 outside the completion;
    - the support placed at `[:, -logits_to_keep:]` and shifted with the labels in the fused forward;
    - `fused_lm_head=True`, and the trainer's temperature given to `add_fused_lm_head`;
    - TRL's own sampling mask None at the run's settings.

    A source in the registered logits form reads as path `fill`. Neither form is not_matched, and the launch stops
    (M7). Mode `grammar_masked` needs path `support_ids`. The fix's mode, `grammar_masked_fp32`, runs under either,
    because the override computes its form itself. T2 does not trace the data flow: the alignment is checked
    numerically, by the preflight below on the installed TRL, and at run time by M9, M10 and the self-check.
  - **The overrides.** The entry says the subclass "overrides one method". The mask is applied in that one, and two
    more are overridden to check and record; both return TRL's results unchanged:
    - `_get_per_token_logps_and_entropies` applies the mask;
    - `_generate` replays xgrammar along every completion's sampled ids;
    - `_generate_and_score_completions` opens each step's record and reads TRL's per-sequence weights.
  - **Entropy and KL.** TRL's logged `kl` is between the masked policy and the masked reference. Its `entropy` stays
    the whole vocabulary's ("The entropy stays that of the whole vocabulary", `kernels/chunked_logprob.py:483`), so it
    is finite without a fill, and M8 reads it as logged. The fill form logs that same quantity.
  - **The fill form stays,** computed by the override from the model's ordinary forward: `logits_to_keep + 1` with
    the last row dropped, fp32, the temperature, a large finite fill, then the log-softmax.
    - At each launch's first old pass it runs on two rows beside the support path, as a recorded self-check. It is a
      diagnostic and never stops a launch. smoke.json flags a self-check with a mean difference over 0.05 or a max
      over 1.0 (`selfcheck_flags`, never a gate).
    - It is the named fix for a measured miss (`trainer_logprobs: grammar_masked_fp32`), used only after an amendment
      here says so.
    - **A fact about that fix.** Under TRL 1.17.1 the support path is already an fp32 masked log-softmax of bf16
      logits: the bf16 tile is at `chunked_logprob.py:510-512` and `:548`, the fp32 statistics at `:160`, `:353` and
      `:589-597`. The fill form takes bf16 logits too, so it changes the numerics only if the GPU's support path is
      wrong. After a measured miss, the first self-check is read before the re-run is spent, and the reading goes to
      Shresth with the miss.
  - **`cast_lm_head_to_fp32`:** the UNVERIFIED note is corrected. TRL 1.17.1 handles tied embeddings
    (`grpo_trainer.py:1059-1094`). It stays not named.
  - **The mask's source, as vLLM builds it.**
    - vLLM compiles each request's grammar with `stop_token_ids` = the request's `all_stop_token_ids`
      (`v1/structured_output/__init__.py:195-206`). For TRL's requests these are the tokenizer's eos plus the
      generation config's eos ids (`sampling_params.py:772-800`): 151643 and 151645.
    - Its matcher takes them as `override_stop_tokens`, and its TokenizerInfo is
      `from_huggingface(tokenizer, vocab_size=model_config.get_vocab_size())` (`backend_xgrammar.py:69-72`, `:157-161`).
    - The trainer's matcher is built the same way, with the same stop ids. Its bitmask decoder is checked against
      xgrammar's own when it is built.
  - **V1 reads both of vLLM's model runners.** vLLM 0.32 builds Model Runner V2 by default (`v1/worker/gpu/`), not the
    file whose lines the entry cites. Both apply the bitmask to the logits in place before the processed log-probs. V1
    stays recorded, and M10 stays its test.
  - **Where the matcher runs.** TRL's `_generate` returns every completion's sampled ids. The hook replays the matcher
    along each completion once per generation, truncated ones too. The replay:
    - checks every token;
    - compares the NFA's allowed set with xgrammar's at every position (M8);
    - caches the allowed sets that the three passes read.

    A violation stops the launch there, before any pass sees the rows.
  - **M10's −log Z** needs the old pass's unmasked log-probs as well. That takes one more no-grad forward of the old
    pass's 32 rows each step, through TRL's own path without a support. Its cost is of the order of the old pass (an
    estimate). D-021's "under 1 s a step" did not count it, and M12 now includes it. Run 1's STOP rules do not read
    −log Z, so whether run 1 keeps the forward is decided when run 1's code path is registered.
  - **M11's weights** are TRL's applied per-sequence weights (`importance_sampling_ratio` under `sequence_mask`).
  - **Record names.** Each steps.jsonl line's `mask` holds the per-step records. `mask/rows` is `mask.rows`
    ({old, ref, loss}); `is/seq_weights` is `mask.seq_weights`, with the raw exp(Σ(old − sampling)) as
    `mask.seq_weights_raw`. TRL's logs carry scalar summaries of them.
  - **How smoke_check reads the gates** (docs/GRPO.md §8.3-8.4).
    - M5's registration half is checked in the job's own snapshot. Every file `docs/prereg/D-021_masked_smoke_path.txt`
      lists must hash to its blob, and `docs/DECISIONS.md` must hold this entry.
    - M10's two clauses are two gates: `M10` (validity) and `M10_ratio` (measured). A step whose old pass had no valid
      token (every completion truncated) weighs 0 in M10's token-weighted mean.
    - M9 needs a value and a token count at every step of both launches. Otherwise it is undecided.
    - A measured gate that cannot be decided makes the launch invalid, not a measured miss: it measured nothing.
    - The D-021 gates never change GRPO_SMOKE_OK. That line keeps its meaning (the smoke ran end to end), and the
      verdict prints before it as `D021_PART_A <verdict>`.
  - **Items 3 and 4's readouts** start where item 5's step-0 values exist: the official Phase-1 gate run's junk.json
    and Phase 2's GRPO smoke.json. The masked smoke's values, if wanted for scale, are computed afterwards from its
    rewards.jsonl, by a script the CHANGELOG names. No gate reads them.
  - **A preflight, deciding nothing here.** Before the smoke, `lofi-mask-preflight` runs on a CPU node of the policy
    env (`main`, 8 CPU, 32 GB, 20 min: at most 320 SU; registered by Claude). It exercises what the laptop cannot:
    - the source checks;
    - the replay and the NFA, as a launch builds them, walked along D-020's 980 completions and prompt v2's references
      (tokenizer ids and one character per token) and along 400 seeded random walks;
    - TRL's own support path on a tiny random Qwen3 across three vocabulary tiles, compact packing included, against
      the fill form and a direct masked log-softmax;
    - the trainer's glue on a real PEFT model.

    A failure there is fixed before the smoke is submitted.
  - **New SU before run 1 under Part A:** about 250 for the smoke, plus the preflight's few minutes (an estimate). At
    most 1,820: 1,500 for the smoke and 320 for one preflight.
  - **Item 1, as built** (`docs/prereg/brief_streams_v1.json`, by `python -m lofi_rl.brief_streams`):
    - The rule for the extra brief: the 9 cells in ENERGIES × MOOD_NAMES order, of which the first n mod 9 take
      ⌈n/9⌉ (Claude's default).
    - Each brief of train and the evaluation streams has 8 draw seeds (k = 0..7), not 4. A use that needs m passing
      samples takes the first m passing draws in k order, and a brief still short is reported short.
    - The measurement stream has none, because junk-rate seeds its engine as D-020 registered.
  - **Re-read after his yes.** The draft said its laptop counts would be re-read before the entry was decided; they
    were re-read the next morning. None is a gate.
    - **Twins:** 8 of 5101's first 40, 4 of 5102's first 40 and 15 of 5103's first 100 all hold. The stratified
      selections the file registers have 8 of 40, 4 of 40, 13 of 100 and 40 of 270.
    - **"Seeds 5101–5104 appear nowhere in the repo":** this was true before the draft. Since then they appear only in
      the documents that register them and in `lofi_rl.brief_streams`.
    - **M2:** 980 of 980 completions, 250,939 tokens walked, 0 violations (`tests/test_grammar_mask.py`).
    - **M3:** 20 with no stop id, the same 20 with both stop ids, 25, and 8.
    - The prototype's "median 20–21" allowed tokens is 21, from an alphabet that now includes the bar mark. A cold
      step takes 0.043–0.046 s and a warm one 0.006–0.008 s (laptop, the committed module).
