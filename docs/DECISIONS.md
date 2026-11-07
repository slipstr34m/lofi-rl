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
- **Amended 2026-11-06 by Shresth** ("increase your limit for job to 1000 SU now", in chat, while approving the Phase-1 gate's 28 minutes): Claude registers a new or changed type of 1,000 SU or less itself (`BY=Claude`) and records it in the CHANGELOG; `config/budget.yaml` `self_approve_max_su: 1000`. A type over 1,000 SU, new or changed, and a run over its estimate still need his yes in chat. Entries he approved before stay `BY=Shresth`.

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
- **Outcome, 2026-11-03: Part A passes, and the grammar is on for run 1 with the masked trainer.**
  - **The smoke:** `lofi-grpo-smoke` 13673180 ran on an L40S (b17-15) from snapshot `ae7d3bf6e1eb-068f2ec9aad0`. It COMPLETED in 00:13:57 for 279.0 SU, with `GRPO_SMOKE_OK` and `D021_PART_A pass`. Every number below is read from `runs/grpo-smoke-13673180/summary/smoke.json` (`d021`).
  - **Validity, M5-M8: all pass.**
    - M5: both launches' stamps read `ae7d3bf6e1eb`, dirty=0. In the snapshot, all 73 frozen files sat at their blobs, and this entry was there.
    - M6: 480 of 480 stop completions in the language (320 and 160), other.n 0. xgrammar was pinned and applied in both launches.
    - M7: T1-T5 `match` in both launches, T5's run-time half too, path `support_ids`; V1 and V2 `match`.
    - M8: 0 violations and 0 NFA mismatches over 120,958 positions of 480 completions; 32 old, 32 reference and 32 loss rows at each of the 15 steps; every loss, KL and entropy finite.
  - **Measured, M9-M11: all pass.**
    - M9: D-020's quantity is 0.00665, token-weighted (per step 0.0048-0.0082). The bar is 0.05; 13657200 read 0.1538.
    - M10: the token-weighted −log Z is 0.1492, at least 0.05 (validity). M9 is 0.0446 of it, at most 0.25 (measured).
    - M11: at every step the mean of the 32 weights is 0.713-0.980 (at least 0.3), and the largest weight is 0.067-0.103 of their sum (at most 0.25). 13657200's per-step means were 0.0010-0.0814.
  - **M12 (estimate only) misses:** the mean step_s is 43.60, over the 43.3 bar. Net of the −log Z forward (5.40 s a step) and the self-check, it is 38.16. So, as this entry says, the lofi-grpo estimate is redone before the ask.
  - **The self-check (a diagnostic):** fill against support, the masked mean |difference| is 2.0e-8 in launch a (step 1, adapter still zero) and 0.0091 in launch b (step 6). Neither raised a flag.
  - **What follows, by this entry's rule (M5-M11 all pass):**
    - This entry supersedes D-020's IS-half verdict. The grammar is on for run 1 with the masked trainer, and D-020's Outcome (prompt v2 with the grammar) is run 1's setting again.
    - Next, in order: D-020's official Phase-1 gate on the frozen palette (with the 50-clip ear check and his APPROVED line in `gates/GATE_phase1.md`); the step-0 samples; Phase 2's GRPO smoke; then `lofi-grpo` goes to him for his yes.

## D-022 (2026-11-04): `palettes/lofi_rain_v1.yaml` frozen
**Decided by:** Shresth, with his line under `## Approval` in gates/GATE_palette.md, `APPROVED by Shresth Rana 2026-11-04`, which he wrote there after the second sign-off listen passed. Claude assembled the palette from his picks (docs/FREEZE.md) and sealed it. "The freeze spec" below is the design of workflow `wf_831448d6-864`, which docs/FREEZE.md describes as built. The spec itself is not in the repo: it was `/private/tmp/claude-501/-Users-shresthrana-csci596-project/1fee16e5-4162-472d-ab8f-ddc709802fb2/scratchpad/freeze/SPEC.md`, in a session scratchpad that is not kept, so this entry restates each of its rules that it applies (§5.3, §5.5, §5.6 and §6.2's F22). The laptop evidence it cites is committed in docs/evidence/D-022/.
- **His line and the seal** (CHANGELOG 2026-11-04 10:50; main `074b46c`).
  - His first save put the date in angle brackets (`<2026-11-04>`), as the How-to-approve placeholder showed it. The seal refused that line, and he fixed it in the file. The refusals now show the line with a bare date (`80596d9`, `9aca73c`; the docs `313b865`), and name a line in another name as that person's, never rewritten into his (`5acc9a5`); the accepted form is unchanged.
  - `python -m lofi_rl.freeze.seal` read his line and checked that the palette (sha256 A) is what the latest sign-off job, 13706355, rendered for b1103-signoff-2, and that `freeze check --gate` passes ("Days 1-3, keep3, 1 redo rounds, 2 sign-off listens"). It rebuilt the file with `frozen: '2026-11-04'`, one line apart from A, and `freeze check --sealed --verify` passed on it.
  - Then `python -m lofi_rl.freeze.tables --write gates/GATE_palette.md --private` and tests/test_palette_frozen.py: steps 1–4 of the freeze spec's order on the day (§5.6). This entry is step 5.
- **His confirmations** (gates/GATE_palette.md, each day's confirmation line): Day 1 (2026-10-31) and Day 2 (2026-11-01), "Yes, use these"; Day 3 (2026-11-03), "Yes, use these", with "24 below (Recommended)" as his word on the rain level.
- **keep3** (2026-11-01; batch b1101-keep3, job 13656049): he kept 10 of the 30 drafts, one per brief.
- **Q1** (2026-11-03), before any sign-off render: on keep3's kept 10 the keys sat 1.22 LU under their target (wurli −2.42, rhodes −1.41, piano +0.62; CHANGELOG 2026-11-03 15:17). His answer: "Keep as heard (Recommended)". He gave no numbers, so `lofi_rl.freeze.CHANGES` stayed empty.
- **The first sign-off listen** (2026-11-03; batch b1103-signoff, job 13701026): 10 of 10 heard; Great 2, Good 1, Fine 3, Toss 4.
  - Tossed: `drafts/v1/d01_chords.ls`, `drafts/v1/d06_horn.ls`, `drafts/v1/d07_chords.ls` and `drafts/v1/d10_groove.ls`.
  - Asked what made him toss them (the rain, a part, or the loops themselves), he answered "The loops themselves".
  - His words named the loops, not the palette. So the palette did not change, nothing was sealed, and briefs d01, d06, d07 and d10 reopened through the drafts redo path (the freeze spec §4.5, step 3; docs/FREEZE.md §5.1).
- **The order** (2026-11-03): asked whether to redraft the four briefs before the sign-off or to approve the palette first, he answered "Redraft first".
- **His redo picks** (2026-11-03; batch b1103-keep3-redo, job 13703830), one per round, with no round marked "none acceptable": `drafts/v1/redo1/d01_groove.ls`, `drafts/v1/redo1/d06_chords.ls`, `drafts/v1/redo1/d07_groove.ls` and `drafts/v1/redo1/d10_chords.ls` (gates/GATE_palette.md, `## Kept drafts (keep3 redo1)`).
- **Q2** (2026-11-03), on the new kept 10: the drums sat +1.149 LU and the keys −1.040 LU off their targets (median). His answer: "Keep as heard (Recommended)". Nothing changed, and A stayed `d3590c8d…`.
- **The second sign-off listen** (2026-11-04; batch b1103-signoff-2, job 13706355): 10 of 10 heard; Great 3, Good 4, Fine 3, Toss 0. It passed, and his line followed.

- **What froze**
  - **The file:** `palettes/lofi_rain_v1.yaml`, 496 lines, sha256 B `09f80a644476abe4d14517d0ec4129d69e1a4bc91eb0935c9ffd152411872183`, `frozen: '2026-11-04'`, written by `python -m lofi_rl.freeze build` and the seal, never by hand.
    - The signed-off file, sha256 A `d3590c8da2e186de2b5a113a585f17646dfec9390f1eb9de569b1bacaa47bc8a`, is B with `frozen: null`; exactly one line differs (tests/test_palette_frozen.py).
    - No render reads `frozen:`, so A ≡ B in audio. Both sign-off jobs rendered A: `palette_sha256` in each `summary/args.json`, and the label `lofi_rain_v1@d3590c8d` in each `summary/renders.json`. Their votes are private and record real clip ids (src/lofi_rl/ui/votes.py), and each real clip id hashes that label in (`clip_id`, src/lofi_rl/drafts/render.py).
  - **The picks, under the drafts' names** (F1), each with its trim t and fader F in dB (docs/FREEZE.md §1):

    | name | instrument (shootout option) | trim_db | fader_db |
    |---|---|---|---|
    | sax `alto` | `sax_mtg_alto`, warmth off, expression on, timing lag | 6.63 | 2.55 |
    | keys `rhodes` | `gu_tine_ep` (GU, bank 0, prog 4) | 0.0 | −5.94 |
    | keys `wurli` | `gs_wurli_ep200` | −3.22 | −6.435 |
    | keys `piano` | `vcsl_upright_knight` | 21.86 | −6.06 |
    | kit `studio` | `kit_virtuosity` (the whole drum stem) | −14.94 | 15.26 |
    | its bass | `bass_vsco2_pizz` | 34.97 | 5.09 |
    | kit `jazz` | `kit_gu_jazz` (GU Jazz kit 32 one-shots) | 0.0 | 15.21 |
    | its bass | `bass_fp_finger_yr` | −2.79 | 5.13 |

    - Each t and median F is Day 2's (lofi-shootout 13651610), equal in Day 1 (13635646) for all 8 instruments and in Day 3 (13656048) for the 5 it measured; S\* is −31.924651028769418 in all three.
    - The keys faders are CD-28's mean of the K1 rootless and K2 spread F: rhodes −6.04 and −5.84, wurli −6.65 and −6.22, piano −6.06 and −6.06 (the file's `#| freeze_provenance:` block).
  - **The bed:** `{src: rain, start_s: 44.3, xfade_s: 1.0, delta_lu: 24.0}`, placed before the limiter. That is excerpt e2 (the W1 pick `excerpt_e2`) at Δ 24, the level alone taken from the W2 pick `rain_d24`, which W2 played on e1.
    - So e2 at Δ 24 was first played in the first sign-off listen; W1 played e2 at Δ 18 (docs/FREEZE.md §1).
    - The e2 stretch's pcm sha256 is `e6f3a632…` (shootouts/rain_excerpts.json). Both sign-off jobs hashed it on the cluster to that value (each job's summary/palettes/index.json).
  - **The settings:** swing 0.55 (`swing_55`), humanize off (`humanize_off`), warmth off (`sax_plain`), expression on, and timing lag at 25 ms (`timing_lag`).
  - **The mix and loudness:** dev_gu's bus and master chains (the draft's without their chain-head trim slots); `bus_gain_db` 0.0 on every bus; bus targets sax 0, drums −1.5, bass −3 and keys −6 LU; −14 LUFS ± 0.1 in at most 4 iterations, ceiling −1 dBTP.
  - **The vocabulary:** the chord numerals and qualities unchanged (Day 2's borrowed-chord condition was not met); voicings rootless, shell, spread and quartal; `aliases: {}` (F2). Its LoopScript regex is 3,408 characters, sha256 `5c081828ac50718edc4db82c2a855ea52210debb375280ba01e8442de60f8558`, the same as draft_day2's. dev_gu's is 3,395 characters, `1412f11f…`.
  - **The sources,** each with its sha256 and licence mode (D-012; F14):
    - committed: GeneralUser GS `9575028c…`, the one committed source;
    - fetched under `$LOFI_ASSETS`: the gsullivan Wurlitzer EP200 `aab58d9c…` (CC BY 3.0), the VCSL upright piano `151ea259…` (CC0 1.0), the VSCO 2 contrabass pizzicato `27f341ab…` (CC0 1.0), the FreePats finger bass `bd133269…` (CC0 1.0), the MTG alto `2b338b89…` (CC BY 4.0 and CC BY 3.0), the Virtuosity kit `5ca78cde…` (CC0 1.0), and the rain, Freesound 235827 by Metadex converted to 44.1 kHz, `e6884875…` (CC0 1.0).
  - **The provenance** (F13): the file's `#| freeze_provenance:` block, pinned by B. It names the inputs with their sha256s (palettes/draft_day2.yaml `58aeb924…`, shootouts/plan.yaml `4f99b9f3…`, shootouts/rain_excerpts.json `ec52348a…`, Day 3's levels.json `b313ff9a…`), the three confirmation dates, and the rule behind every number.

- **Departures, each with its reason**
  - **The bed as heard is not −Δ ± 0.5 LU.**
    - Δ is applied before the limiter (src/lofi_rl/render/bed.py, "Level"; instrument.md §4, Rain bed), and the limiter then works on the music and the bed together. The docs' "bed at −Δ ± 0.5 LU" holds on tests/test_render.py's synthetic noise fixture, not with the rain.
    - As heard (`bed_lu`: the final render minus the music, against the music):
      - the final kept 10: 22.64–23.93 LU under the music (job 13706355, runs/drafts-13706355/summary/verify.json `checks.bed_heard.kept_below`);
      - keep3's kept 10 at the first sign-off: 22.64–23.99 LU (job 13701026, the same field);
      - Day 3's four W2 renders at Δ 24, the ones he judged, on excerpt e1: 21.99–23.98 LU (runs/shootout-13656048/summary/renders.json `bed.bed_lu`);
      - two drafts outside the kept 10 sat more than 2.015 LU closer than −24 in both jobs (d04_chords by 2.556 LU, d04_groove by 2.074 LU), flagged as information.
    - So the bed is checked as placed, and its level as heard is recorded, not gated (F8). As placed it is 24 LU under the music within 0.01 LU; over the 51 renders with the bed in job 13706355, the largest miss was 2.1e-9 LU. The W1 and W2 renders he judged placed the bed the same way.
    - RENDER.md §3.4 and §12 and instrument.md §7 now say so (`9abc2cd`, `01cf5a2`).
  - **CD-28's keys mean kept exact:** wurli −6.435 dB, not rounded to 0.01 dB as SHOOTOUTS.md §3 rounds trims and faders, because keep3 played −6.435 and the freeze keeps the faders he heard (CD-29; F6).
  - **Loudness iterations: "at most 4" kept** (instrument.md §4, Loudness, step 4). All 61 renders of job 13706355 and all 49 of job 13701026 converged in 1–3 iterations and passed G7 (their summary/renders.json; verify.json `checks.iterations`). No render forced a higher cap.
  - **The published drawn tables hold the design rounds only** (Claude's default from review finding S4-6; CHANGELOG 2026-11-03 15:17).
    - The plan promised the drawn tables in the gate file after the freeze; SHOOTOUTS.md §2.3 ("Published after the freeze") now names the design rounds only. The gate file's `## Drawn round tables (published after the freeze)` covers Days 1–3, keep3 and keep3 redo1, and leaves out the 8 hidden-check rounds: Phase 2's votes and the final A/B reuse those checks, and the rater is never told which rounds they are (RATING.md, "Checks you won't see").
    - The full tables are private, in `data/votes/audit/drawn_round_tables.md` (sha256 `7c6c679c…`; CHANGELOG 2026-11-04 10:50). The seeds, the opaque and real ids and the audio hashes are in neither (F11; CD-21).
  - **Job A did not pass as the freeze spec §4.3 defines it.** `lofi-render-check` 13701025 (2026-11-03, 105.6 SU) ended with 3 failed and 3229 passed (runs/render-check-13701025/summary/pytest.txt). None of the three came from the freeze build (CHANGELOG 2026-11-03 15:45):
    - span1_dotted's Linux golden key was stale since voicing V2.1, and it was re-recorded from this run (`950e3800…`);
    - `test_junk.py::test_render_writes_one_pass_flacs_for_real` rendered the in-prompt example, a near-copy that D-020 keeps out of the render pool, so it got 1 clip of 2. Since `8da39fd` it renders two non-copies, and it passes on the laptop (`-m cluster`, 2026-11-04);
    - `test_rain_excerpts.py::test_the_real_file_gives_the_committed_record_when_it_is_here`: the excerpt picker, re-run on the cluster's rain file, does not give the laptop-made record byte for byte. It is still to be investigated. The bed itself is the record's: both sign-off jobs hashed the e2 stretch to `e6f3a632…`.
    - Job A was there to run S1's test 1, the golden render hashes, on Linux (the freeze spec §4.3). Job B's twin checks S1 on the kept loops: twin0, the frozen document with `bed: null` at the levels keep3 played, gave each kept loop its own source render's audio on the cluster, 10 of 10 in both jobs.

- **Evidence**
  - **The second sign-off job,** `lofi-drafts --signoff` 13706355 (main, a02-09; snapshot `e5111e2e2e05-0237b644dee3`): COMPLETED in 00:07:03, 84.6 SU, `SIGNOFF_OK` (CHANGELOG 2026-11-04 10:30). Its stages (summary/stages.json): check 5.3 s, palettes 5.7 s, render 405.0 s, verify 4.7 s, build 0.2 s. Every check of summary/verify.json passed:
    - N: each kept loop's n equals its source's and `loop_samples(bpm)` (940,800–1,282,880 samples);
    - 61 of 61 renders ok (the kept 10, their 10 twins, and 41 others: the 32 drafts outside the kept 10 and prompt v2's 9 references): −14.092 … −13.990 LUFS, true peak at most −1.456 dBTP, G1–G7 with no repair, 1–3 loudness iterations;
    - the bed placed 24 LU under the music (above), and as heard 22.64–23.93 LU under it on the kept 10;
    - each frozen kept render equals its fresh-process repeat (audio, music and the whole manifest), 10 of 10, in 4.09–5.20 s a process;
    - twin0's audio is each kept loop's source render (keep3's for 6, keep3 redo1's for 4), the music under the bed is twin0's, and every bus's pre-fader loudness, trim and fader are its source's;
    - the batch b1103-signoff-2: 10 triage rounds, 301.6 s of audio, no two consecutive rounds in one cell (summary/batch.json, draw.json).
  - **The first sign-off job,** 13701026 (main, a01-13): COMPLETED in 00:04:26, 53.2 SU, `SIGNOFF_OK`. 49 of 49 renders ok at −14.093 … −13.990 LUFS, true peak at most −1.456 dBTP, in 1–3 iterations; twin0 = keep3, and each repeat = its render, 10 of 10 (runs/drafts-13701026/summary/verify.json, renders.json).
  - **The redo render,** `lofi-drafts` 13703830 (00:00:27, 5.4 SU, `DRAFTS_OK`), with `palettes/draft_day2.yaml` as keep3: 12 of 12 at −14.082 … −13.994 LUFS, true peak at most −1.456 dBTP (runs/drafts-13703830/summary/renders.json).
  - **SU:** the four jobs of the freeze took 248.8 SU (105.6 + 53.2 + 5.4 + 84.6; CHANGELOG, from `session_report.py`).
  - **The kept 10's bus levels** (the freeze spec §4.1 e), as `python -m lofi_rl.freeze.checks report --gate gates/GATE_palette.md --renders keep3=runs/drafts-13656049/summary/renders.json --renders keep3_redo1=runs/drafts-13703830/summary/renders.json --verify` reads them from drafts/v1/keep3_kept10.json (sha256 `72a739ec…`). It ran on 2026-11-04 on the sealed file, from a worktree, with the two renders.json given by absolute path. Each bus after its fader, relative to S\* and its target:

    | bus | median off target (LU) | per-loop range | over 1.0? |
    |---|---|---|---|
    | sax | +0.480 | −0.456 … +1.037 | no |
    | keys | −1.040 | −2.422 … +1.486 | yes |
    | bass | −0.012 | −1.130 … +0.806 | no |
    | drums | +1.149 | −4.043 … +4.390 | yes |

    - The keys per instrument: wurli (d01_groove, d03_chords, d04_horn, d10_chords) median −1.685; rhodes (d02_chords, d07_groove) −1.232; piano (d05_groove, d06_chords, d08_horn, d09_chords) +0.091. Against each loop's own sax, the keys' median is −1.172.
    - The two buses over 1.0 went to him as Q2. On keep3's kept 10 the keys median was −1.220 and the drums +0.499 (CHANGELOG 2026-11-03 19:15).
    - The kept 10 play the jazz kit in all 10, and keys piano in 4, wurli in 4 and rhodes in 2. The studio kit and its bass are in none of them. In the sign-off listens they played only in b1103-signoff's d01_chords and d10_groove, both tossed (runs/drafts-13701026/summary/renders.json).
    - The same report: (a) the 10 kept files are the texts their sources rendered, and pass gate, compile, energy and mood under the frozen palette; 42 of 42 drafts pass in drafts/v1 and drafts/v1/redo1. (c) Every draft fullmatches the frozen regex, 42 of 42. (d) and (g): `freeze check --verify --sealed` and `freeze check --gate --sealed` exit 0. (i) 10 kept now (keep3 6, keep3 redo1 4), and no kept file tossed. (h) The fresh `expect` differs from the committed file only in its two sources' path labels, which are absolute in a worktree that has no runs/; the run of 2026-11-03 19:15 found it equal.
  - **His triage counts:** above (gates/GATE_palette.md; the votes are private, votes commits `9dec1cc` and `a1eb3c0`).
  - **Prompt v2:** 9 of 9 references under `strict=True` with the frozen palette, identical to dev_gu's (the report's (b)). So D-020's "If that palette refuses a v2 example, that run's prompt is registered here first" asks for nothing, and the template stays `127e3094…`.
  - **The frozen regex:** 3,408 characters, sha256 `5c081828…` (the report's (c)); tests/test_palette_frozen.py pins both.
  - **The Phase-1 gate's render step** (the freeze spec §5.5 and F22: it changes only if 50 × s exceeds 300 s, where s is job B's slowest one-at-a-time render, and 300 s is the 360 s deadline less 60 s kept for an outlier).
    - **On the cluster.** Job B is the latest sign-off job, 13706355. It rendered its 41 others one at a time with 4 workers, in an 8-CPU job, with palette A, whose audio is B's. Their render_s: min 2.90 s, median 6.09 s, mean 7.707 s, max 17.39 s (d07_chords), sum 315.99 s (runs/drafts-13706355/summary/verify.json `checks.render_time`; renders.json `others`). So 50 × 17.39 = 869.5 s, against 300 s.
    - The first job B, 13701026, rendered the same palette sha: 29 renders, mean 6.019 s, median 4.49 s, max 30.68 s, a projection of 1,534.0 s (runs/drafts-13701026/summary/verify.json, renders.json). Its 30.68 s render was ref_l1_melancholy_low, an isolated outlier: the same render took 3.63 s in 13706355, and 13701026's next slowest took 10.29 s.
    - **On the laptop** (Apple M4, 10 CPUs): the frozen palette B through `junk.render_clip` with a warm cache, by docs/evidence/D-022/render_orders.py. Laptop renders are not canonical (LESSONS 38), so only the ratio is used.
      - The 42 drafts and 9 references, with job B's briefs (docs/evidence/D-022/drafts_orders.json): one at a time with 4 workers, 113.03 s and 122.81 s; two at a time with 2 workers each, 67.39 s and 70.28 s; speedup 117.92 / 68.84 = 1.713.
      - 50 of arm B's loops (docs/evidence/D-022/armb_orders.json): 115.83 s and 114.91 s against 66.89 s and 63.85 s, speedup 115.37 / 65.37 = 1.765.
      - The projection uses the drafts' lower 1.713 for what they cover. Of arm B's 446 passing non-copies on dev_gu (runs/junk-13640325), 170 (38 %) fail G4 under the frozen vocabulary, since keys 'organ' is not in it, and 276 compile (docs/evidence/D-022/armb_compile.json). The 276 and the 50 sampled from them all play rhodes and the jazz kit (their `keys:` and `drums:` lines in runs/junk-13640325/summary/completions.jsonl; armb_orders.json `meta.keys_kits`), so they exercise neither wurli, piano nor the studio kit. Re-scored on the frozen palette, 312 of 500 pass and 276 are passing non-copies. The drafts and references play all three keys and both kits.
    - **The same clips either way.**
      - In all 4 runs of each laptop set, every clip's audio_sha256, FLAC sha256 and manifest sha256 was the same: 51 of 51 drafts and references, 50 of 50 arm B loops.
      - Through the CLI, `python -m lofi_rl.rl.junk render` with jobs 1 and 4 workers against jobs 2 and 2 workers, on arm B's re-scored run, gave the same order and 50 of 50 identical clips: records less render_s, FLAC bytes and manifest bytes (docs/evidence/D-022/armb_cli_compare.json).
      - In the suite, `tests/test_junk.py::test_render_two_at_a_time_gives_one_at_a_times_clips_with_the_frozen_palette` (`f07d579`; its drafts reordered in `b92acfa`) renders three drafts on the frozen palette two at a time on a cold one-shot cache, then one at a time, and finds them identical. The two renders that start together both play the jazz kit, so both fill its one-shots at once: in a probe of that run, all 72 cache writes came while both ran (docs/evidence/D-022/concurrent_fill_probe.json). It passes with LOFI_ASSETS and skips where the fetch-only sources are missing.
      - A render never depends on its workers (render.pipeline.render; tests/test_determinism.py).
    - **The freeze spec's order** (§5.5), under two rules on the job's timeouts.
      - tests/test_junk_job.py requires the three timeouts and their kill grace to sit at least 60 s under the walltime. Generate takes 840 + 15 s and score 90 + 10 s, so the render timeout gets the rest: 475 s at 25 min, 535 s at 26, 595 s at 27 and 655 s at 28.
      - The render deadline stays 60 s under its timeout, as the job's 360 s and 420 s were, so that a render that starts just before the deadline ends before the timeout kills the step: about 20.3 s on a lane at the worst projection below, about 35.8 s for an outlier like 13701026's. That gives deadlines of 415, 475, 535 and 595 s. With the test alone, (iii) with (ii) at 26 min could take a deadline near its 535 s timeout and hold the 507.6 s projection (468 SU); it fails below only under this rule.
      - **The test applied to each option:** its worst-case projection (50 × s, divided by the laptop speedup for (iii)) fits under its deadline. The spec states a test only for no change, 50 × s ≤ 300 s, which keeps 60 s for an outlier below the 360 s deadline. The test applied here keeps no such 60 s at the end; it keeps only the gap between the last render's start and the deadline (below). That is Claude's reading of the spec, and a departure from the no-change rule's form.
      - No change: 869.5 > 300.
      - (i) at 25 min: the deadline can rise to at most 415 s (timeout 475 s), so the spec's +90 s does not keep the minute spare; and 869.5 > 415.
      - (ii) at 26 min (deadline 475 s, timeout 535 s) or 27 min (535 s, 595 s): 869.5 > 535.
      - (iii) alone, the sbatch untouched: 869.5 / 1.713 = 507.6 > 360. With (i): 507.6 > 415. With (ii) at 26 min: 507.6 > 475.
      - **(iii) with (ii) at 27 min: 507.6 ≤ 535, with 27.4 s spare. Chosen,** under the test above: the smallest change within Claude's 500 SU, until his answer (28 min, below).
      - Under the no-change rule's own form (the projection at most the deadline less 60 s), 27 min fails (507.6 > 475), and the first option that passes is 28 min (507.6 ≤ 595 − 60 = 535). That is (iv), 504 SU, Shresth's call.
      - Without (iii), the deadline would need 869.5 s: 33 min and 594 SU, so (iv), Shresth's call.
    - **The projection after the change:** at worst 507.6 s against the 535 s deadline, about 20.3 s a render on each of the two lanes, with the last renders starting at about 487.3 s, 47.7 s before the deadline. At job B's mean: 50 × 7.707 / 1.713 = 225.0 s.
    - **Sensitivity:**
      - the laptop runs, paired one by one, give speedups of 1.6083 (113.03 / 70.28) to 1.8224 (122.81 / 67.39), so 540.6 s to 477.1 s: the most pessimistic pairing is 5.6 s over the deadline, and its last renders would start at about 519 s, before it;
      - taking the first job's 30.68 s as s gives 895.5 s, which would need about 34 min and 612 SU;
      - one isolated outlier of that size (about 35.8 s on a lane) moves the last start to about 502.8 s, still under 535 s, which is the case the spec's outlier margin is for.
    - **Decided by Shresth, 2026-11-06: "28 minutes".** He was asked whether the gate job runs with a 27- or 28-minute limit, after a plain-words overview: 27 min as registered (486 SU; worst case 507.6 s against the 535 s deadline, 27.4 s spare), or 28 min (504 SU; a 595 s deadline, so the worst case keeps the no-change rule's 60 s), with the cost billed by the minutes used either way. 28 minutes or more is his decision (the freeze spec §5.5 (iv)).
      - So the job runs at 28 min, with the render step under `timeout -k 10 655 … --deadline-s 595`: the three timeouts with their kill grace come to 855 + 100 + 665 = 1,620 s, a minute under 1,680 s, and the 507.6 s worst case sits 87.4 s under the deadline.
      - Registered on his yes: `make approve JOB=jobs/junk_rate.sbatch MAX=504 BY=Shresth`, so jobs/APPROVED.yaml reads `{sha: 5a423302eb84dd91, max_su: 504, by: Shresth, date: 2026-11-06}`.

- **Consequences**
  - **D-017 holds:** the file never changes. tests/test_palette_frozen.py pins B, with the message "D-017: the palette never changes after the freeze", and also A (B with `frozen: null`), the date, the regex (3,408 characters, `5c081828…`), prompt v2's 9 references, and his one line under `## Approval`.
  - **`palettes/draft_day2.yaml`** (sha256 `58aeb924…`) is a rebuild input of B's provenance, so it stays byte-identical too.
  - **The Phase-1 gate's validity** on the frozen palette is D-020's list (its amended Validity bullets), with two values swapped and one condition dropped. The swapped values:
    - `palette.sha256` is B, `09f80a644476abe4d14517d0ec4129d69e1a4bc91eb0935c9ffd152411872183`, instead of dev_gu's `b2e4a340…`, with `palette.matches_generate` true;
    - `constrained.regex_sha256` and `constrained.in_language.regex_sha256` are both `5c081828ac50718edc4db82c2a855ea52210debb375280ba01e8442de60f8558`, instead of `1412f11f…`.

    The dropped condition: D-020's second amendment made its One-mirror clause require that the stamp's commit hold the 60 frozen files of docs/prereg/D-020_junk_path.txt. That list recorded its arms' code and does not bind the gate (the freeze spec §5.3: "one stamp, now the job snapshot's"). At `c9c97d6` 53 of the 60 match and 7 differ: kit.env, scripts/job_env.sh, src/lofi_rl/palette.py, src/lofi_rl/render/mix.py, render/pipeline.py, render/sampler.py, and src/lofi_rl/rl/junk.py, the one this entry's work changed (`git rev-parse <commit>:<path>` against the list; the first 6 already differed at main `074b46c`). So no reader checks the gate against that list.

    Everything else is unchanged:
    - complete: `n` = `requested` = 500, `partial` false, `unreadable_lines` 0 and `generate.stopped` null;
    - as registered: the model revision `cdbee75f17c01a7cc42f958dc650907174af0554`, `briefs.seed` 2026, and `sampling` equal to runs/junk-13632676's; prompt v2 with template `127e3094…`, `prompt.template_matches_generate` true, `prompt.rows_matching` 500 and `prompt.example_mismatch` 0;
    - constrained: `constrained.enabled` true, `constrained.in_language.matches_generate` true, `stop.out_of_language` 0 and `other.n` 0;
    - what the regex guarantees: `failures.G1`, `G3`, `G4`, `G5` and `crash` all 0;
    - one stamp: `generate.host.stamp` and `score_stamp` show the same `commit=… dirty=0`, now the job snapshot's.

    tests/test_junk.py runs generate (`--prompt v2 --constrained`) and score with no `--palette`, as the job does, through a fake engine, and checks the swapped values and the v2 and grammar conditions above (`0f02428`). The render step and G7 stay outside validity. The bar is D-020's (CLAUDE.md §9), as D-021 restates it, and unchanged: `pass_not_copy.count` ≥ 300 of 500 in a valid run, and at least 15 of 50 clips "fine" by his ear, with his APPROVED line in `gates/GATE_phase1.md`.
  - **Beside that list: the gate's job type and its render step changed** (the freeze spec §5.5 (iii) with (ii); the evidence is above).
    - D-020's arms ran under job sha `4a9d8685231fd573` (max 450 SU), and D-021 named that sha for the gate. The gate runs under `5a423302eb84dd91`, max 504 SU, registered on his yes (`make approve JOB=jobs/junk_rate.sbatch MAX=504 BY=Shresth`, 2026-11-06). Until his answer it was `a7c7ea3502cf395c`, max 486 SU, re-registered by Claude under D-003's amendment (`make approve JOB=jobs/junk_rate.sbatch MAX=486 BY=Claude`, 2026-11-04). `scripts/approve_job.py --check` lists no stale entry.
    - `src/lofi_rl/rl/junk.py`: `RENDER_JOBS = 2`, so the job's `--workers 4` renders two clips at a time with 2 workers each (`8da39fd`). The clips, their manifests and their order are those of one-at-a-time rendering; only render_s differs. No render starts after the deadline, on either lane.
    - `jobs/junk_rate.sbatch`: `--time 00:28:00`, and the render step under `timeout -k 10 655 … --deadline-s 595`, on his answer. From `2198836` until then it was 27 min, a 535 s deadline and a 595 s timeout, and before 2026-11-04 25 min, 360 s and 420 s. The three timeouts with their kill grace come to 855 + 100 + 665 = 1,620 s, a minute under 1,680 s.
    - Cost: 4 CPU + 6 (24 GB) + 8 (the GPU) = 18 SU/min × 28 min = 504 SU at most.
    - The render step's change is a code change on the gate path that leaves the clips and their order unchanged. Neither it nor the walltime touches validity: D-020 says "The render step and G7 do not affect validity either".
    - The submit is `make submit JOB=jobs/junk_rate.sbatch OVR='-p gpu' ARGS='500 2026 1 v2'`, with no `--palette`: junk's default is now the frozen file (`eb5bf11`).
  - **The other switches of the freeze spec §5.5** (`eb5bf11`, `9abc2cd`):
    - junk-rate's generate, score and render read `palettes/lofi_rain_v1.yaml` by default. D-019's and D-020's runs stay on dev_gu, as their junk.json records, and a re-score of one passes `--palette palettes/dev_gu.yaml`;
    - the scorer server's default stays dev_gu (src/lofi_rl/scoring/server.py), with the reason now in its comment: jobs/scorer_smoke.sbatch starts its servers without `--palette`, and 5 of the 6 gate-passing fixtures it scores fail G4 under the frozen palette, which has no aliases. Phase-2 jobs pass `--palette palettes/lofi_rain_v1.yaml`, and a request made for another palette is refused with a 409;
    - the docs: DSL.md's names, GRAMMAR.md's regex row, RENDER.md §3.4 and §12 and instrument.md §7 (the bed as heard), SHOOTOUTS.md's status, taste_pipeline.md's sign-off vote rules, and the rain's credit in assets/ATTRIBUTION.md;
    - no change: the copy penalty's references (prompt v2's texts are the same under both palettes); the masked trainer's NFA, checked on dev_gu's regex, since Phase 2's smoke re-checks M8 at run time (D-021);
    - new configs, for Phase 2's GRPO smoke and run 1, name `palette: palettes/lofi_rain_v1.yaml` and decode under `5c081828…` (D-021's What follows, step 3, for the Phase-2 smoke, "the frozen palette's grammar"; D-021's Settings, Grammar, for run 1; those Settings name dev_gu's regex only for D-021's own smoke).
  - **Not settled here.** D-021 was written before the redo round. Its Part B corpus names keep3's 10 kept drafts, and its item 2 pool names "the 30 drafts". Neither has run. Which drafts they hold now (the kept 10 above, the four loops he tossed at b1103-signoff, the 12 redrafts) is registered with Part B if it runs, and in run 1's config (`copy_penalty.pool`) and code path.
  - **What follows** (the freeze spec §5.6, steps 7–10): the CHANGELOG entry, with the re-registration line D-003's amendment asks for (jobs/junk_rate.sbatch `4a9d8685231fd573`, 450 SU, 2026-10-31 → `a7c7ea3502cf395c`, 486 SU, `BY=Claude`, 2026-11-04, by `make approve JOB=jobs/junk_rate.sbatch MAX=486 BY=Claude`, for the freeze spec §5.5 (ii) with (iii)), and the second, on his answer: `a7c7ea3502cf395c`, 486 SU → `5a423302eb84dd91`, 504 SU, `BY=Shresth`, 2026-11-06; `pytest -q`, commit and push; `make sync`; then D-021's next step, the Phase-1 gate on the frozen palette, a separate unit.

- **Records, so no reader takes them for pending work**
  - **Historical comments left unedited.** Four GRPO smoke configs keep `palette: palettes/dev_gu.yaml` with the comment `# until palettes/lofi_rain_v1.yaml is frozen (D-017)`: configs/grpo_smoke.yaml:15, grpo_smoke_constrained.yaml:13, grpo_smoke_v2_grammar.yaml:18 and grpo_smoke_v2_grammar_masked.yaml:23.
    - They record the smoke configurations made on dev_gu. Two are pinned by blob: grpo_smoke_v2_grammar.yaml in docs/prereg/D-020_grpo_smoke_path.txt and grpo_smoke_v2_grammar_masked.yaml in docs/prereg/D-021_masked_smoke_path.txt.
    - So is src/lofi_rl/dsl/vocab.py (blob `b27b49a9…`, in all three docs/prereg lists), whose docstring still calls the names placeholders until the palette freeze and names the frozen palette.
  - **junk.py's old blob** (`f545f5bc…`) stays in the three docs/prereg lists, as a record of runs that have ended. No `.sync-freeze` is active, and the gate job's snapshot records the new blob.
  - **The sign-off votes** are triage tiers, each loop's tier an acceptability label at weight 1 (taste_pipeline.md, Storage). A sign-off batch whose palette did not freeze never trains the RM (F9). Both batches, b1103-signoff and b1103-signoff-2, played `lofi_rain_v1@d3590c8d`, the palette that froze, so neither is excluded.
  - **The four loops he tossed** stay in drafts/v1 as committed, and a tossed file stays tossed (`lofi_rl.freeze.tossed_kept`; docs/FREEZE.md §5.2).
  - **The redo round** (docs/DRAFTS.md §6; CHANGELOG 2026-11-03 17:35): workflow `wf_7f517813-71f` wrote 12 redrafts into drafts/v1/redo1/, three per reopened brief, and the validator passed 12 of 12.
    - Claude's default for their keys and kits (docs/DRAFTS.md §8): in each brief's round one `studio` draft and two `jazz`, and the three keys instruments, none in the slot its old draft played.
    - Its reason: the 30 drafts used each kit 15 times; he kept 8 `jazz` and 2 `studio` loops in keep3, and both `studio` loops were among the 4 he tossed. One `studio` draft per round keeps that kit in his rounds while his picks decide.
    - His four picks are all `jazz` drafts (runs/drafts-13703830/summary/renders.json).
  - **The freeze tools after the spec** (docs/FREEZE.md §5; workflow `wf_ad4dd7fe-b31`, merge `79c327c`):
    - `REDOS` and `SIGNOFFS` record the redo rounds and the sign-off listens beside `PICKS`, each compared with its own gate section by `freeze check --gate`. `REDOS` holds keep3_redo1 (job 13703830), and `SIGNOFFS` holds b1103-signoff (13701026, 4 tossed) and b1103-signoff-2 (13706355, none);
    - `kept_now()` is keep3's table with each redo round's picks over it;
    - `drafts/v1/keep3_kept10.json` is schema `lofi-rl/keep3-kept10/2`: a source per kept loop (keep3's renders.json `94ee7e1e…` for 6, keep3 redo1's `d00ec18f…` for 4), and every kept draft read by its file path, since a redo file shares its stem with a drafts/v1 file;
    - `checks expect`, `checks report` and job B refuse a tossed loop that is still kept, and the seal takes only a passing latest sign-off listen.
  - **The build** (CHANGELOG 2026-11-03 15:17): the spec by workflow `wf_831448d6-864`, S1–S4 built by `wf_a13612f3-74d`, reviewed by `wf_81d8287d-967` and fixed by `wf_981d2247-1ed`.

- **Claude's defaults applied** (the freeze spec §6.2; each is overridable):
  - F1 the drafts' names; F2 `aliases: {}`; F3 every level in a per-instrument field, `bus_gain_db` 0 and no chain-head slots; F4 a FluidSynth trim as a prepended `Gain`, only when t ≠ 0; F5 a kit trim added to each lane's dB in `kit_shots`;
  - F6 CD-28's mean kept exact (a departure, above); F7 Day 2's values, with Day 1 = Day 2 asserted for all 8 instruments and Day 3 = Day 2 for the 5 Day 3 measured; F8 the bed checked as placed and recorded as heard, with an information flag above 2.015 LU (a departure, above);
  - F9 the sign-off in triage mode, no hidden controls, the votes private, counts only in the gate file, and a batch whose palette did not freeze kept out of the RM; F10 `lofi-drafts --signoff`, no new job type (sha `85a1ae8e085fd255`, max 180 SU; the two sign-off jobs took 53.2 and 84.6 SU);
  - F11 drawn tables for the 9 shootout batches, keep3 and its redo round, without seeds or ids (narrowed by S4-6, above); F12 `frozen` is the date of his line, quoted, one line apart from A; F13 the provenance block inside the file; F14 the source modes unchanged, the rain added to assets/ATTRIBUTION.md's CC0 list;
  - F15 the manifest schema stays `lofi-rl/render/1`; F16 trims and faders written on every instrument, even at 0.0, in a fixed key order; F17 junk-rate switched, the scorer server's default kept with its reason, the smoke configs left; F18 the package `lofi_rl.freeze`;
  - F19 the kept loops' expected values committed as `drafts/v1/keep3_kept10.json` (schema 2 since the redo round); F20 job B also renders the drafts outside the kept 10 and the 9 references one at a time with 4 workers (41 in 13706355); F21 the gate-file reader by headings, labels and backticked ids, in the checks and at the seal; F22 the render step changed only because 869.5 > 300, by (iii) with (ii) within 500 SU, on the test that each option's worst-case projection fits under its deadline (a departure from the no-change rule's form, above), then at 28 min on his answer, which keeps that rule's 60 s.
  - Since the spec: S4-6, the published tables' design rounds only (above); the redo round's keys and kits (above); docs/FREEZE.md §7's for the redo rounds: `REDOS` and `SIGNOFFS` as constants traced to the gate file, a tossed file tossed for good, a `- **Tossed:**` item in every sign-off section, keep3_kept10.json written with `--allow-tossed` while a tossed loop was still kept, and a redo draft's render id `<folder>_<stem>`; and, for the render step, (iii) with (ii) at 27 min as the smallest change within Claude's 500 SU under that test, the render deadline kept 60 s under its timeout, and his answer to the 27-or-28-minute question: "28 minutes".

- **Amended 2026-11-06: the gate job writes its run on the node's own disk** (Claude, under D-020's "Anything else: it is fixed, and the arm is re-run once", and D-003's amendment of 2026-11-06 for the registration).
  - **The first gate launch is invalid.** `lofi-junk-rate` 13783877 (a02-15, snapshot `ac4acfb0388f-ec11f881579c`, 450.9 SU) generated 384 of 500 before its 720 s deadline (`runs/junk-13783877/summary/generate.json`: `stopped: deadline`), so the run is not complete (D-020). Its render step was then killed at its 655 s timeout after 45 of 50 clips. Score read 383 of 384 passing and 370 passing non-copies; none of it counts.
  - **The cause was the file system, not the palette.** Its three generate chunks took 13.58, 12.8 and 12.46 s of model time at 2,581 tok/s, and the log printed 128/500 at 333 s, 256 at 543 s and 384 at 709 s. Between chunks junk.py only appends 128 JSON lines, fsyncs them and rewrites generate.json on scratch. The D-020 runs 13640324, 13640325 and 13656050 printed their chunks 12–15 s apart. Its renders took up to 126.67 s, against job B's 17.39 s at most.
  - **`lofi-io-probe` 13787408** (a01-12, 2026-11-06, `runs/io-probe-13787408/summary/io.json`) timed junk's write round of 128 lines plus an fsynced JSON rename: 2.209, 132.108 and 71.906 s on scratch, 0.0 s on the node's disk, and 0.006–0.007 s on KEEP. A 2.6 MB clip took 0.911–171.923 s on scratch and 0.001 s on the node's disk.
  - **So the job now writes its run on the node's own disk** (`RUN=${TMPDIR:-/tmp}/junk-<job>`) and copies it to `$REMOTE_KEEP/runs/junk-<job>/`, its records first and then its clips, each under `timeout -k 2 10`. The copy runs after the render, and from an EXIT trap when a step stops the script early or Slurm's TERM ends it (`trap 'exit 143' TERM`), so the records outlive a kill. Its one-shot cache is `$REMOTE_KEEP/cache/lofi-rl`. `make pull` brings summaries back from KEEP too, and `make init` makes KEEP's `runs/` and `cache/`. The steps, their arguments and their deadlines are unchanged, so the files they write are the same; only where they are written moves. The Slurm log stays on scratch: 13783877's file times put its stall in the fsyncs and the rewrite of generate.json, not in the log's prints.
  - **The walltime stays his 28 minutes.** Score's timeout drops from 90 to 60 s (it took under 29 s in 13640325 and 13783877), so generate 840 + 15, score 60 + 10, render 655 + 10 and the two copies 2 × (10 + 2) come to 1,614 s, 66 s under 1,680 s. At 18 SU/min that is 504 SU at most. The changed type is re-registered by Claude within D-003's 1,000 SU (`make approve JOB=jobs/junk_rate.sbatch MAX=504 BY=Claude`: `{sha: 0eb12128013576e3, max_su: 504, by: Claude, date: 2026-11-06}`).
  - **Validity is unchanged:** D-020's list with this entry's two swaps. A second invalid launch goes to Shresth with the records (D-021, "An invalid launch, twice").

## D-023 (2026-11-07): the step-0 samples, drawn by `lofi-step0` under rule `step0-select/1`, pre-registered before its run

**Decided by:** Shresth, 2026-11-07, with his answers to the build spec's §7 below, in his words; proposed by Claude (docs/design/step0_samples.md, workflow `wf_72e78bac-bc8`, 2026-11-06, revised after an independent review, and built on 2026-11-07 with the deviations its §10 lists). Asked whether to take the recommended answer to all 15 open questions (this spec's 8 and the seed-pool spec's 7, docs/design/seed_pool.md §9) or to name the ones to change, he answered: "Defaults OK". So each answer below is the spec's (a). The rules he did not answer are Claude's defaults (the spec's §9), each overridable.
- Nothing here runs before his answers. This entry is committed with them, beside the code and before the job, as D-021 was (`DECISIONS.md:306`).
- It records the rules D-021 "What follows" item 2 leaves open ("The step-0 samples, all drawn under the grammar, with item 1's streams and draw seeds: the RM held-out tuples, PAR's Ref_b and the seed pool's 660", `DECISIONS.md:537`), and the registered text they depart from, under "What this entry supersedes".
- From now on this entry is the record, not the spec. Amendments go here, never into docs/design/step0_samples.md. A fix to the step-0 code before a redo is one such amendment (below, "An invalid run, and recovery").

- **His answers** (the spec's §7; each sets one named constant in `src/lofi_rl/step0/__init__.py`, under the comment that names its question):
  - **Q1, the seed pool's 220 training briefs:** (a) ("Defaults OK"). Sets `POOL_BRIEFS` ("band" for (a): energy × mood × tempo band, the bands 60–74, 75–90 and 91–95 BPM in the sampler's shares, a short band giving its room to 75–90; "cell" for (b): `stratified_indices(596, 220)`), with `POOL_BANDS` and `POOL_FILL_BAND`.
  - **Q2, may a seed-pool loop also be one of its brief's 4 yardstick loops:** (a) ("Defaults OK"). Sets `POOL_SHARES_REFB` (False for (a): the pool takes the 3 draws after Ref_b's 4 whose texts differ from them, and no clip that is any train brief's Ref_b).
  - **Q3, which base-model loops count as passing:** (a) ("Defaults OK"). Sets `PASSING` ("hard" for (a): parse, compile and render in spec, no PQ floor; "hard_not_copy" (b); "hard_no_repair" (c)).
  - **Q4, two identical loops from one brief:** (a) ("Defaults OK"). Sets `DISTINCT` ({"heldout": True, "refb": False} for (a)).
  - **Q5, a held-out loop's exact LoopScript turning up elsewhere:** (a) ("Defaults OK"). Sets `POOL_EXCLUDES` ("text" for (a), the texts compared across all cells, a short held-out brief's clips included; "clip" for (b) and (c)).
  - **Q6, the yardstick's 1,600 loops, now or with the run-1 request:** (a) ("Defaults OK"). Sets `STREAMS` ((("rm_heldout", "all"), ("train", "all")) for (a), 3,520 requests; (("rm_heldout", "all"), ("train", "pool")) for (b), 2,080), and with it `MARK`.
  - **Q7, what "committed before the first vote" means now:** (a) ("Defaults OK"). Sets `COMMITTED_BEFORE` ((a): "before the first Phase-2 vote (the triage of 300 or the first best/worst round), and before any round with a held-out loop is built").
  - **Q8, a brief with too few usable loops in its 8 draws:** (a) ("Defaults OK"). Sets `SHORT_RULE` ("report" for (a); "ask" for (b): render exits 1 and prereg refuses while a held-out or Ref_b brief is short, until this entry holds his answer for the run as the line `D-023 short answer: run step0-<job>; as (a)`, alone on its line in this entry, at column 0 and starting a block (one list bullet before it at most; in backticks, bold or italics, or with a closing period), outside fences and comments (a sentence, a note or a refusal that quotes it, a wrapped sentence's next line, and an indented, nested or fenced line are no answer, so an example names `step0-<job>`), added after the run with no redo, which handles those briefs as (a) does and is recorded in both prereg files' `rule.answers.Q8`; any other answer for the run, a note after the line included, keeps the refusal until the run's short-answer lines are replaced with that one line, alone (his words and the date go on a line of their own, never after it); prereg reads the line in the checkout it runs in, so once main has moved (below, "At the run's commit") the line goes into this entry on main and, by hand and uncommitted, into the docs/DECISIONS.md of the git worktree of the run's commit that prereg runs in, and the two files it writes there are committed on main; a short pool brief is reported under either answer).
  - **The answers line**, which `lofi_rl.step0.requests.answers_line()` writes from those constants, and which this entry holds verbatim so that an answer recorded here and a constant left at another value cannot both stand (`tests/test_step0_docs.py`):

    ```
    D-023 answers: Q1 POOL_BRIEFS band, POOL_BANDS 60-74/75-90/91-95, POOL_FILL_BAND 75-90; Q2 POOL_SHARES_REFB False; Q3 PASSING hard; Q4 DISTINCT heldout True, refb False; Q5 POOL_EXCLUDES text; Q8 SHORT_RULE report
    ```

- **Rule `step0-select/1`** (the spec's §1.3–1.8 as answered, with its §10; `src/lofi_rl/step0/select.py`, a pure function of complete records):
  - **The draws.** Draw k of a brief (k = 0..7) is one vLLM request with n = 1 and SamplingParams(seed=draw_seeds[k]) from `docs/prereg/brief_streams_v1.json` (sha256 `bb2daee5…`), read only through `lofi_rl.brief_streams.load()`. Every draw of every drawn brief is made (Claude's default, §1.7): rm_heldout 40 × 8 = 320 and train 400 × 8 = 3,200, so 3,520 requests (Q6). Step 0 is Qwen3-4B-Instruct-2507 @`cdbee75f`, no adapter, prompt v2 (template `127e3094…`), the frozen palette's grammar (regex `5c081828…`, xgrammar pinned), T 1, top_p 1, top_k 0, 512 tokens: junk's `SAMPLING`.
  - **Passing (Q3).** A draw passes its records when it passes G1–G6 and compile (junk's score), and passes when it then passes G7 at its set's render. The PQ floor is not applied: τ_PQ comes from the triage of 300, which draws its base-model clips from these 660 after the held-out set must be committed. Near-copies and repaired loops pass, and are reported per set. G7 is applied lazily, in k order: a draw is rendered only while the draws before it in its set have not filled m, and a draw whose text equals one still to render counts as its repeat until that one's render decides.
  - **The RM held-out tuples** (rm_heldout, seed 5101, the 40 registered briefs): m = 4. Walk k = 0..7 and take a passing draw whose canonical text (canon_sha256) differs from every draw already taken (Q4); stop at 4. This run's render step decides G7. The set is 160 clips, 40 tuples of 4, if no brief is short. Twins stay, listed for the RM's report.
  - **PAR's Ref_b** (train, seed 596, run 1's [0, 400)): m = 4. The brief's first 4 draws that pass G1–G6 and compile, in k order, repeats kept (Q4), so 1,600 in all. Ref_b's own render decides its G7 later: a reference that fails it is replaced by the brief's first spare that passes, in k order. The spares are its other passing draws, less the pool's draws, any draw with a pool draw's text, and any draw that is another pool brief's pick (Q2).
  - **The seed pool's 660** (train, 220 × m = 3, the briefs by Q1 within [0, 400), which stand in for the planner's): each brief walks its draws after its 4 Ref_b draws, k ascending, and takes a draw that passes G1–G6 and compile, whose canonical text is unlike the brief's Ref_b texts, its earlier picks' and (Q5) every held-out clip's, and that is no other train brief's Ref_b clip (Q2) and no earlier pool brief's pick; stop at 3. The clip id has no brief id, so two train briefs of one cell, bpm and tonic give one clip for one text: every Ref_b by its records comes first, then the pool briefs in stream order. Its own render decides G7 later, and a pool draw that fails it is dropped, never replaced. 660 is a planning count.
  - **Short briefs (Q8).** A brief still short after its 8 draws is reported short (sets.json `short`, with each draw not taken and why: a gate code, `near_copy` or `repaired` under Q3(b) or (c), `repeat_of:<k>`, `refb`, `pool`, `heldout_text`, `heldout_clip`, `refb_clip` or `pool_clip`), never drawn again under other seeds; a brief whose walk meets a draw with no decided render is `undetermined`. A held-out tuple whose clip later fails G7 under newer render code is dropped the same way and reported. A Ref_b brief left short leaves run 1's [0, 400) and goes to Shresth with the lofi-grpo ask.
  - **Which code decides each set's G7:** held-out, this run's render step; Ref_b and the pool, the render jobs that render them (the lofi-grpo ask; `lofi-render-pool`), each under its own recorded render-code key (its stamp and `features.source_sha8()`). No later outcome re-selects anything else. The render path is not frozen (Claude's default, §3).

- **The mark** (the spec's §2.3). Before any work, generate checks that the job's code tree holds this line in its docs/DECISIONS.md, read from its head to the next backtick or double quote or the line's end (`lofi_rl.step0.requests.find_marks`, which check and the docs test read it with too), and that it equals `lofi_rl.step0.MARK`, which `requests.mark_line(STREAMS, RULE)` builds. So no job draws under rules or streams not registered first. The precedent is M5 (`DECISIONS.md:713-714`).

  ```
  D-023 mark: rule step0-select/1; streams rm_heldout (seed 5101, all 40 briefs) and train (seed 596, all 400 briefs); 8 draws a brief, 3520 requests
  ```

- **The job:** `lofi-step0`, `jobs/step0.sbatch`, with no argument (any argument stops it with exit 2).
  - One A40 on `gpu`: 1 node, 1 task, 4 CPU, 24 GB, `--gres=gpu:a40:1`, `--time=00:45:00`, `--no-requeue`, the log at `/scratch1/%u/lofi-rl/logs/%x_%j.out`.
  - **Cost:** 4 CPU + 6 (24 GB) + 8 (1 GPU) = 18 SU/min × 45 min, so at most 810 SU. That is within D-003's amendment of 2026-11-06 (`DECISIONS.md:26`), so Claude registers it: `make approve JOB=jobs/step0.sbatch MAX=810 BY=Claude`, registry sha `deba472bcc2f4f67`. About 301 SU are expected under Q6(a) (an estimate at the gate run's rates, the spec's §2.9).
  - **Steps:** generate (policy env) under `timeout -k 15 1080`, `--deadline-s 960`, with a stack dump at 1,050 s; score (policy env) under `timeout -k 10 420`; render (scorer env, the held-out set only) under `timeout -k 10 960`, `--deadline-s 900`; the copy-out of `summary/` and `clips/` under `timeout -k 2 30` and `timeout -k 2 60`, and, when that copy fails after the render, one more copy of the records from the EXIT trap under `timeout -k 2 20`. Together 2,611 s, 89 s under the 2,700 s walltime.
  - **The seed probe**, before any draw: 3 identical requests at each of seeds 1 and 2 (neither a draw seed), the low/hopeful cell's messages, in one call. It passes when at each seed at least 2 texts are identical and the two seeds' texts differ; a miss stops generate (exit 2) with nothing drawn.
  - **Where it writes:** the run on the node's own disk (`${TMPDIR:-/tmp}/step0-<job>`), copied to `$REMOTE_KEEP/runs/step0-<job>/` once, records first and then the clips: after the render, or from an EXIT trap (with `trap 'exit 143' TERM`) if a step stops the script early or the job is killed. When the copy after the render fails, the EXIT trap copies the records once more, under `timeout -k 2 20`, and no more; a copy the EXIT trap makes itself is not retried. `LOFI_CACHE` is on KEEP. generate.json says the draws are complete (`partial` false) before the replay, so a kill there leaves a complete run. A used OUT is refused before the traps are set. Nothing goes to the inbox and no ear check is asked.
  - **Submit:** `make submit JOB=jobs/step0.sbatch`, on USC's VPN, after `make sync`, with `pytest -q` (whose pre-flight checks the path file and this mark in the checkout) passing at the commit.

- **The path and the stamp rule** (the spec's §4.1): `docs/prereg/D-023_step0_path.txt`, written last at the code commit by `python -m lofi_rl.step0 path --write`, lists "<git blob sha>  <path>" for every file the job runs or reads: the import closure of `lofi_rl.step0` (static, imports inside functions included, check.py and prereg.py among them), `jobs/step0.sbatch`, `scripts/job_env.sh`, `scripts/_kit_env.sh`, `kit.env`, and the committed data files (the frozen palette and its mode-commit SoundFont, prompt v2's 9 example sources, the streams file). Generate checks every blob in its snapshot before any work; the laptop checks them at the stamp's commit with git.

- **Validity** (the spec's §4.1; `python -m lofi_rl.step0 check RUN_DIR [--redo REDO_DIR] [--json] [--rescore]`, on the laptop after `make pull`; exit 0 valid, 3 invalid, 2 bad input; the last line `STEP0_CHECK <verdict>`). D-020's list with D-022's values, adapted to per-request seeds. A run counts only if its files show all of the following, each value of the registered JSON type (`partial` 0 is not false, `n` 3520.0 is not 3520). That is 74 conditions under either Q6 answer; a redo pair has 75: its stamp condition is `stamps of the redo pair`, and `the path at the redo commit` is added.
  - **Complete:** generate.json `stopped` null and `partial` false; step0.json `n` and `requested`, and generate.json `requested`, all 3,520; `unreadable_lines` 0; over completions.jsonl against the registered requests, `requests.missing`, `requests.duplicate` and `requests.foreign` 0.
  - **Seeded:** `requests.seed_mismatch` 0 (each row's `draw_seed` and `sp_seed` are `brief_streams.draw_seed(seed, index, k)`); generate.json `probe.passed` true; `requests.prompt_tokens_mismatch` 0 (each cell's rows at runs/junk-13788483's `prompt_tokens` for that cell, 1,325–1,411).
  - **As registered:** generate.json `model.revision` `cdbee75f…`, `streams_file.sha256` `bb2daee5…` and `fresh` true, `streams` the mark's, `palette.sha256` `09f80a64…` and `sampling` equal to runs/junk-13632676's; `requests.brief_mismatch` 0 (each row's brief is `sample_brief(seed, index)`); per stream, step0.json `prompt.version` v2, `template_sha256` `127e3094…`, `template_matches_generate` true, `rows_matching` the stream's requests (320, 3,200), `example_mismatch` 0, and `palette.sha256` `09f80a64…` with `matches_generate` true; clips.json `palette.sha256` `09f80a64…`.
  - **Registered first:** generate.json `registration.found` true and `registration.mark` equal to `MARK`; and, with git, the generate stamp's commit holds the mark, read as generate reads it, in its docs/DECISIONS.md, whose sha256 is `registration.decisions_sha256`.
  - **Constrained, per stream:** `constrained.enabled` true; `regex_sha256` and `in_language.regex_sha256` `5c081828…`; `in_language.matches_generate` true; `stop.out_of_language` 0; `other.n` 0.
  - **What the regex guarantees, per stream:** `failures.G1`, `G3`, `G4`, `G5` and `crash` all 0 (`DECISIONS.md:221`).
  - **One stamp, one path:** generate.json `host.stamp`, step0.json `score_stamp` and `render_stamp` show the same `commit=… dirty=0`; generate.json `path_file.blobs_ok` true; and, with git, at that commit the path file is the one generate checked (its sha256 is `path_file.sha256`) and every file it lists has its listed blob. A redo pair follows the recovery rule below instead.
  - **Held-out render:** `render.errors_final` 0, `render.skipped` 0, `render.other_gates` 0; no held-out brief undetermined. A render error retried once that then renders is `render.errors_retried`, shown and never judged.
  - **Sets:** every heldout row is stream rm_heldout, and every refb and pool row train (the entries' streams and their rows'); the pool takes at least one draw; under Q5(a) no pool pick has `heldout_text` true, and under (b) or (c) none is a held-out clip.
  - **Reproduced on the laptop:** sets.json equals `lofi_rl.step0.select` over scored.jsonl and clips.json; every clip id equals the formula (`taste_pipeline.md:76`) over its record's canon_sha256 (score's for its row), the brief, clips.json's palette label and the record's `render_cfg_sha8`, one config for the run; `render_code.source_sha8` equals `features.source_sha8` over the render stamp's commit, read with git.
  - **At the run's commit:** check and prereg reproduce and judge the run with the laptop's code, so they refuse (exit 2) a checkout where a file that can change a verdict, or a byte prereg writes, differs from its blob at the run's latest stamp: the import closure of `lofi_rl.step0.check` and `lofi_rl.step0.prereg` (the rule, the requests and the answer constants, the DSL, junk, the features' `SOURCE_GLOBS` and the palette code among it), less the checker's own machinery, and the streams file and the palette. Check and prereg run at the run's commit, in a git worktree of it (`git worktree add --detach <dir> <commit>`, the commit the refusal names; the run's folder stays in main's runs/) once main has changed one of those files, the two prereg files then committed on main from there; a fix to the machinery (check.py, prereg.py, path.py, cli.py, __main__.py) runs at any, and docs/DECISIONS.md, which this entry's later lines change, is never compared.
  - **Bad input, which decides nothing** (exit 2): a summary file missing or unreadable, a schema that is not the step's, or files that are not one run's own (scored.jsonl against completions.jsonl, row by row and text_sha256 by text; step0.json's `n` and `render` block against scored.jsonl and clips.json; one render-code key in step0.json, sets.json and clips.json), as `lofi_rl.gate1.load_run` refuses them. A scored.jsonl or step0.json missing beside generate's records is named as the scoring defect the recovery below redoes. Three files may be absent. Without replay.jsonl no condition fails: the replay is shown, never judged, and only its readout is absent. Without sets.json the 5 Sets conditions, `held-out briefs undetermined` and `sets.json = step0.select` fail, and item 8's readout is absent. Without clips.json `clips.palette.sha256`, `clip ids` and `sets.json = step0.select` (select reads its records) fail; beside a step0.json with a render block it is bad input, since render writes clips.json before that block.
  - **A run stopped before its first draw** (generate.json alone, `generated` 0 and no completions.jsonl: `stopped` set, as a pre-check, the probe or a failed engine load leaves it, or `stopped` null with `partial` true, as a kill during the engine's load or the probe's call leaves it, the watchdog's exit or timeout's TERM recording no stop) is read with no rows: it reads invalid, naming `generate.stopped`, or for a kill `generate.partial`, shown as killed before its first draw, and `generate.probe.passed` when the probe missed or never ran, and is drawn again whole. generate.json alone in any other state is bad input.
  - **Shown, never judged:** `constrained.backend` (as D-020 treats it); the probe's seeds, cell, requests and distinct texts (its `passed` is judged, under Seeded); the engine and `per_request_seed`; the replay readout; the timings and the GPU; `render.errors_retried`; the near-copy, repair, verbatim, same-clip and `heldout_text` counts; item 8's readouts; `check --rescore`, which re-scores completions.jsonl with the laptop's code and lists any differences from scored.jsonl.

- **An invalid run, and recovery** (the spec's §4.2; D-020's rule, `DECISIONS.md:224-228`). An invalid run decides nothing: it is reported, and its rows never enter any set, pool or file.
  - **A scoring or selection defect, or a held-out render cut short or failed by infrastructure:** the run is redone from its own records, with no new generation, by `python -m lofi_rl.step0 render --from OUT --run REDO`, after `score --from OUT --run REDO` when the scores are redone, in `lofi-step0-redo` (a CPU job on `main`, a new type of at most 1,000 SU that Claude builds and registers only if it is needed). A score redo is always followed by its render redo in the same redo: new scores need new sets, which only render writes, and a pair holding score's records alone reads invalid. The redo writes `$REMOTE_KEEP/runs/step0-redo-<job>/`, reads OUT's summary read-only and never writes under OUT.
  - **A score that left no records:** when generate's records are complete but score or render left no scored.jsonl or step0.json (check: bad input), the run is a scoring defect, D-020's "a crash", redone from its records the same way.
  - **The fix comes first:** any code fix, and every file on the path it changes, is recorded in an amendment of this entry before the redo, with one line per redo, verbatim, bare repo paths comma-separated, on a line of its own in this entry, bulleted, fenced or indented, outside comments (check also reads a path in backticks, a closing period, a bold line or a note after a path as this form; a bold head is not the form and is not read, and a sentence or note that quotes the line names no file):

    ```
    D-023 redo amendment: run step0-<job>; changed <path>, <path>
    ```

  - **Its check:** `python -m lofi_rl.step0 check OUT --redo REDO` reads generate's records (generate.json, completions.jsonl, replay.jsonl) from OUT, and score's and render's (scored.jsonl, step0.json, sets.json, clips.json) from REDO, each from OUT when REDO does not hold it. Every stamp shows dirty=0; the generate stamp's commit holds every listed blob; and each later stamp's commit holds every blob of that same path file except the files that the amendment lines for that run, in that commit's docs/DECISIONS.md, name (D-020's second amendment's rule, `DECISIONS.md:261-276`, with this list). A named file that did not change is fine; one off the path is shown.
  - **Anything else** (incomplete, a run stopped before its first draw, a setting, the seeds, the grammar, the stamp, the registration, the streams): fixed, with the fix recorded here before the re-run. Then the run is drawn again whole, once, into its own folder, and never merged with the invalid rows.
  - **Invalid twice:** it goes to Shresth with the records (`DECISIONS.md:540`).
  - **The first valid run, or valid pair, fixes the sets.** No later run replaces them. A new set is a new file version, with its own DECISIONS entry.

- **Fixing the outputs** (the spec's §4.4). Once check reads valid, `python -m lofi_rl.step0 prereg runs/step0-<job> [--redo runs/step0-redo-<job>] --write`, at the run's commit, writes two files, neither holding a LoopScript text: `docs/prereg/heldout_v1.json` (`lofi-rl/heldout/1`, the file `taste_pipeline.md:128` names: the rule in sentences built from the answers; the settings, `render_cfg_sha8` among them; per tuple its brief and shown example, per clip its hashes, clip id, `audio_sha256`, LUFS, true peak, P_fix, near-copy readouts and verbatim A and B; the short briefs under `short`, whose clips are held-out data in no tuple; the texts' pin; and item 8's readouts with each pair's A/B-differ flags) and `docs/prereg/step0_v1.json` (`lofi-rl/step0-run/1`: every summary file's path, size and sha256, the counts, the verdict, the probe and the replay with its stop as a code, Ref_b per brief with its spares, the pool's picks, every train draw's `heldout_text` flag with the train rows that share each held-out clip's text, passing or not, the readouts, the short briefs and heldout_v1.json's sha256). Both are ASCII JSON, one record a line, byte-identical when written again at the same code; the writer refuses an invalid run, and under Q8(b) a short held-out or Ref_b brief until this entry's short-answer line for the run, never replaces a different file and has no `--force`. Both are committed before the first Phase-2 vote (the triage of 300 or the first best/worst round), and before any round with a held-out loop is built (Q7), and this entry gains its Outcome with their sha256s. The KEEP run folder is kept.

- **Separation** (the spec's §5): every rm_heldout row, taken or not, in completions.jsonl, scored.jsonl and replay.jsonl, is held-out data, used only through `docs/prereg/heldout_v1.json` and never as a base-model, triage, control, Ref_b or pool clip; a short held-out brief's clips (heldout_v1.json `short`) are held-out clips too. Held-out clips enter no seed pool, triage or round but their own held-out rounds and those rounds' hidden repeats. Ref_b is never voted on, served or put in the seed pool, and the 660 are never Ref_b's or in the copy pool. No committed file (the two prereg files, the path file, this entry and its Outcome, the CHANGELOG) holds a step-0 text: the texts stay in `$REMOTE_KEEP/runs/step0-<job>/` and the laptop's runs/.

- **What this entry supersedes** (each with the question that settles it):
  1. `docs/prereg/brief_streams_v1.json:40`, "takes the first m passing draws in k order", for these three sets. The file itself stays unchanged ("do not edit this file", `:4`). For these uses:
     - the held-out set takes the first 4 passing draws whose canonical texts differ (Q4);
     - "passing" is the hard gates without the PQ floor (Q3);
     - Ref_b's G7 replacement skips the pool's draws and clips, and a pool draw that fails G7 is not replaced (Q2);
     - the pool's draws start after Ref_b's 4, with texts unlike Ref_b's (Q2), and, under Q5(a), unlike every held-out clip's; none is another train brief's Ref_b clip (Q2) or an earlier pool pick.
  2. `taste_pipeline.md:127`, "its first 4 step-0 samples that pass the gate": as item 1 (Q3, Q4).
  3. `taste_pipeline.md:233-235`, the gate's PQ factor, for the three sets (Q3).
  4. `taste_pipeline.md:128`, "committed before the first vote" (Q7).
  5. `taste_pipeline.md:329`, the planner's 220 briefs. This unit's 220, by Q1's rule, stand in for them, and a later `lofi_rl.seeds.plan` takes them.
  6. `taste_pipeline.md:349`, "from `lofi-gen`": `lofi-step0` draws the 660.
  7. `DECISIONS.md:603`, Ref_b's 1,600 samples in the lofi-grpo ask: their draws move here, in full or for the pool's 220 (Q6). Their render, features and refs.json stay with the ask (`:603`, `:633`).
  8. `DECISIONS.md:433` and `:436`, run 1's [0, 400) and the 40 held-out briefs: a short brief, or a held-out tuple whose clip later fails G7, leaves them, reported (Q8).

- **Readings, which supersede nothing:**
  - "Never trained on, selected on, or tuned on" (`taste_pipeline.md:129`) holds at the text level under Q5(a) or (b). Near-identical texts remain, since held-out means "fresh samples" of the same 9 prompts (`DECISIONS.md:443`).
  - D-021 item 2's "the seed pool's 660" is a planning count. Short briefs make it fewer, and are reported.

- **Asked of later units:**
  - **The RM loader** (`lofi_rl.taste`, not built): labels on held-out clips go only to the gate (acc from first showings, p from repeats, `taste_pipeline.md:183-184`); under Q5(a) or (b), every label on a clip whose canon_sha256 equals that of any clip heldout_v1.json lists, in `tuples` or in `short`, in any cell, is dropped from training, whatever its source, and the count is reported beside D-008.
  - **The queue job** (`taste_pipeline.md:105-108`, not built): held-out clips appear only in their held-out rounds and those rounds' repeats; a re-render is compared on `audio_sha256` (under the same `source_sha8`, palette, `render_cfg_sha8` and audio stack, the manifest's `platform` less `cpu`, a mismatch is a determinism violation; otherwise it is logged as render drift), and a clip that fails G7 there drops its tuple (Q8).
  - **The pool's and Ref_b's renders** (`lofi-render-pool`; the lofi-grpo ask): G7 by Q2's rules, under their own recorded render-code key, on the picks and spares step0_v1.json records. They give `lofi_rl.step0.select` every scored row and this run's clips.json records with their own (it refuses otherwise: its Q5 exclusion reads the final held-out set), a recomputed pick that differs from step0_v1.json's stops them, and their outcome re-selects nothing else.
  - **The seed pool's render, the triage of 300 and the queue job:** outside the held-out set's own rounds and their repeats, base-model clips come only from step0_v1.json's pool entries (by `i` and text_sha256), and Ref_b only from its refb entries and spares; a recomputed pick that differs stops them. Mutations and broken copies are never made from a step-0 text.
  - **Run 1's copy-pool builder:** it refuses any entry whose canon_sha256 is one of this run's step-0 rows' (scored.jsonl, pinned by step0_v1.json), unless the entry is one of prompt v2's 9 reference texts. A mutation or broken copy made from a step-0 text counts as model output (`DECISIONS.md:449-450`).
  - **`lofi_rl.rl.config`:** it refuses a `briefs.seed` equal to an evaluation stream's seed (5101–5104, `brief_streams.EVALUATION`), with a test, built with Phase 2's GRPO smoke config before that smoke runs.

- **Claude's defaults** (the spec's §9 and §10; each overridable): every drawn brief's 8 draws are made; G7 lazily, in k order; Ref_b by the passing rule with repeats kept and spares for G7; under Q1(a), the bands, their shares and the fill-from-centre rule; twins stay; a pool draw that is a held-out clip is always skipped, and so, under Q2(a), is one that is another train brief's Ref_b clip, and, under any answer, an earlier pool pick, the spares leaving out every pool pick; the yardstick is never played; no committed file holds a step-0 text; the seed probe on low/hopeful at seeds 1 and 2 (3 requests at each in one call; at each seed at least 2 identical texts, the two seeds' differing) and the replay readout (the 320 held-out requests again, in reverse, in chunks of 160, never scored, rendered or selected); engine seed 0, chunks of 128, an A40, 45 min and at most 810 SU; no argument, `--no-requeue`, a used OUT refused, and one bounded retry of the records' copy; the redo job built only if needed; only the held-out set rendered here; the render code recorded with every G7 outcome, the render path not frozen; both prereg files holding hashes only and pinning the summary files; a new entry, D-023, rather than an amendment inside D-021.
