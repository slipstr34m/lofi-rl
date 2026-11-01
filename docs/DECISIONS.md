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
