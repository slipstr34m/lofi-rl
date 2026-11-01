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
