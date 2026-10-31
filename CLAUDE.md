# lofi-rl — CLAUDE.md

An RL-trained small LLM that writes 8-bar jazzhop loops in a constrained DSL (LoopScript).
Loops are compiled and rendered deterministically and rewarded by music-theory verifiers, knob checks,
a small aesthetics term, and a reward model trained on Shresth's best/worst votes. Final project for
USC CSCI 596 (Nakano), Fall 2026. Repo due Dec 11. Proposal meeting with Nakano around Nov 19 (≤ Nov 20).

Read this file, then **docs/TASTE.md** (Shresth's taste profile: it is the spec), then CHANGELOG.md, at the
start of every session.

## 1. Topology — who does what, where

| Where | Role | Allowed | Never |
|---|---|---|---|
| **Laptop (this machine)** | source of truth | edit code, git, GitHub, `.env` secrets, rating UI, listening, Claude Code workflows that write seed loops (no API key), `pytest -q` and preview renders | training, render farms, anything that takes a GPU |
| **Discovery login node** (`scripts/remote.sh`) | dispatcher | the budget guard (via `make submit`), `squeue`, `sacct`, `sinfo`, `scontrol show`, `sprio`, `myaccount`, `myquota`, `module avail/spider`, `rsync`, `ls`, `tail`, `cat`, `mkdir` (`make init`), `python3 scripts/session_report.py` | compute > 1 min CPU, GPU work, `pip install`, builds, `git` |
| **Discovery compute nodes** (Slurm jobs only) | execution target | env builds, renders, scoring, GRPO, scaling runs | `git`, secrets, anything not launched from `jobs/` |

The cluster holds a **mirror** of this repo at `$REMOTE_PROJ`, pushed by rsync. It has no git history and
no credentials. `make sync` also writes `.sync-stamp` (commit, dirty flag, diff hash), which every remote
command and job prints. Code flows laptop → cluster via `make sync`; results flow cluster → laptop via
`make pull`. If a step appears to need a secret or a `git` command on the cluster, stop and ask.

Paths on Discovery (all in `kit.env`). Verified with `myquota` on 2026-10-30: `/project` does not exist for
this account; `/project2` does.

- `REMOTE_PROJ=/project2/anakano_429/shresthr/lofi-rl`: the code mirror. Sync runs with `--delete`, so never put state here.
- `REMOTE_KEEP=/project2/anakano_429/shresthr/lofi-rl-keep`: persistent state outside the mirror. It holds the envs, the tools (uv, FluidSynth), the votes backup, checkpoints, fetched assets and the guard's `submissions.log`.
- `REMOTE_SCR=/scratch1/shresthr/lofi-rl`: logs, runs, renders, inbox, caches, `HF_HOME`. Scratch is purged periodically, so anything worth keeping is pulled or copied to `$REMOTE_KEEP`.
- Home (`/home1/shresthr`) is small: nothing large goes there. `/project2/anakano_429` is 1 TB shared by the whole class.

## 2. Access

The laptop's real `~/.ssh/config`:

```
Host discovery discovery.usc.edu
  HostName discovery.usc.edu
  User shresthr
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  ServerAliveInterval 60
  ServerAliveCountMax 5
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 4h
```

- **Duo.** Key-based BatchMode ssh logs in without a Duo prompt (verified 2026-10-30), so unattended work can reach the cluster.
- **DNS fallback.** The macOS resolver sometimes cannot resolve `discovery.usc.edu` on the USC network. `scripts/_cluster_ssh.sh` then connects by IP with the host key pinned (`HostKeyAlias`, LESSONS 30).
- **If ssh ever prompts (Duo) or hangs:** stop and ask Shresth. BatchMode makes it fail fast instead.
- **Remote commands** always go through `scripts/remote.sh '<cmd>'`: a login shell, BatchMode, running in the mirror and printing its stamp.
- **Long work** never runs inside the ssh call; it is submitted with `make submit` and polled.

## 3. Sync

- `make sync` writes `.sync-stamp`, then dry-runs `rsync --delete --exclude-from=.rsyncignore` through `scripts/check_deletes.sh`. The check refuses protected deletions or more than 200. Then it syncs to `$REMOTE_PROJ`. It needs GNU rsync 3.x (`/opt/homebrew/bin/rsync`); macOS `/usr/bin/rsync` is openrsync.
- `make pull` brings back `runs/<run>/summary/**`, keeping its structure (never flattened), and `inbox/` from `$REMOTE_SCR`. It never deletes on the laptop.

`.rsyncignore`, kept in sync with `tests/test_privacy.py`: `.git/ .env* *.key *.pem .venv/ node_modules/ __pycache__/ .pytest_cache/ .DS_Store data/ inbox/ runs/ ckpt/ assets/big/ *.wav *.flac *.mp3 *.pt *.bin *.safetensors jobs/submissions.log`

Rules:
- Sync before every submit; `make submit` refuses otherwise.
- Pull before reading any result.
- Never edit files on the cluster; the next sync overwrites them.
- Large artifacts (WAVs, checkpoints) stay on the cluster. Only summaries, metrics, plots and FLAC listening copies for ear checks are pulled.

## 4. Slurm conventions

- **Account and partitions.** Account `anakano_429`.
  - `debug`: ≤ 1 h tests on A40s; a single node, often drained.
  - `gpu`: A100-80GB via `--gres=gpu:a100:1 --constraint=a100-80gb` (2 per node); L40S via `--gres=gpu:l40s:1` (3 per node).
  - `main`: CPU nodes with 64 threads = 32 physical cores. Report scaling against cores (LESSONS 37).
- **Walltime.** The cap on `gpu`/`main` is 48 h. Every long job checkpoints and resubmits itself from inside the running link, through the guard (`afterany` plus DONE/STOP files, `src/kit/chain.py`). Never as a pre-queued `afterok` job.
- **Job scripts.** `tests/test_job_scripts.py` enforces that every job script:
  - lives in `jobs/` and is named `lofi-<purpose>`;
  - logs to `/scratch1/%u/lofi-rl/logs/%x_%j.out`;
  - sources `scripts/job_env.sh` with `MODE=policy|scorer|tools`;
  - has a header stating partition, resources, walltime and **worst-case SU** (GPU 8 SU/min each; CPU 1 SU/core-min; memory 1 SU per 4 GB-min).
- **Submission.** Only `make submit JOB=jobs/x.sbatch`, which runs `scripts/submit.sh` (the mirror must match the committed laptop code), which runs `scripts/sbatch_guard.sh` on the cluster.
- **SU budget.** Target **1,000,000 SU**, hard stop **1,200,000**, in `config/budget.yaml`. The guard counts used + reserved + projected SU and fails closed. Past the target it needs `PAST_TARGET_OK=<date of Shresth's OK>`. After each job, log the actual SU in the CHANGELOG (`sacct -j <id> --format=JobID,Elapsed,AllocTRES`, or `python3 scripts/session_report.py`). This is the class's shared allocation: be frugal, and say so at the Nakano meeting.
- **Job-type registry** (`jobs/APPROVED.yaml`).
  - The guard refuses (exit 4, `NEEDS_APPROVAL`) any job type that is unregistered, has changed since approval (sha of its `#SBATCH` and non-comment lines), or is projected above its `max_su`.
  - Claude registers types whose max is ≤ 500 SU itself (`make approve JOB=... MAX=... BY=Claude`).
  - Larger types need Shresth's yes in chat, once (`BY=Shresth`). Reruns within the approved estimate never ask again.
- **Polling.** `make status`, `scripts/watch_job.sh <id>` in the background, or `tail -n 50` of the log. Check, report, move on; no foreground sleep loops.

## 5. Environment (built on compute nodes, never on the login node)

- **Two uv venvs** in `$REMOTE_KEEP/envs/`, each synced from a lock compiled on the laptop (`uv pip compile requirements/<env>.in --python-version 3.12 --python-platform x86_64-manylinux_2_28`):
  - `policy` (`jobs/env_policy.sbatch`): vLLM 0.32.0, TRL 1.17.1, transformers 5.19, and torch as vLLM pins it.
  - `scorer` (`jobs/env_scorer.sbatch`): torch 2.8, transformers 4.43.4 (MERT's remote code breaks on ≥ 4.44), laion_clap 1.1.7, audiobox_aesthetics 0.0.4, pedalboard 0.9.26, pyloudnorm 0.2.0, mido, pretty_midi.
  - The project itself is never installed; jobs get `PYTHONPATH=$REMOTE_PROJ/src`.
- **Tools.** `jobs/tools.sbatch` builds `$REMOTE_KEEP/opt/tools` from conda-forge: python 3.12, uv, and **FluidSynth 2.6.1**. Discovery has no module for it, and the laptop pins the same version (`brew pin fluid-synth`). Jobs call `$REMOTE_KEEP/opt/bin/fluidsynth`, a wrapper that drops `LD_LIBRARY_PATH` (LESSONS 8).
- **Builds** run on `main`, with a `timeout` on every step (LESSONS 4), and end with an import walk of every module (LESSONS 43).
- **Models** are downloaded anonymously into `$HF_HOME` by `jobs/fetch.sbatch`: ungated repos only, at pinned revisions. Compute nodes have outbound network (LESSONS 11).
- **`jobs/smoke_env.sbatch`** (debug A40) must pass before anything else:
  - imports and a visible GPU in both envs;
  - FluidSynth renders a test MIDI twice with the same hash;
  - one LoopScript renders to a WAV of exactly N samples at −14 ± 0.1 LUFS, with true peak ≤ −1 dBTP;
  - vLLM generates one completion;
  - the scorers run, with timings recorded.
- **The laptop venv** (`.venv`, from `requirements/laptop.lock.txt`, no torch) runs `pytest -q`, preview renders and the rating UI.

## 6. Memory and verification

- **docs/TASTE.md is the spec.** It holds Shresth's taste profile from the 2026-10-30 interview. Every design choice traces to a row there, or to a "Claude's default" it lists. Change it only with his words.
- **CHANGELOG.md is the lab notebook.** After every unit of work append: date, what was done, numbers (read from files), failed approaches and why, SU spent, next step. Read it first every session. Never retry something it says failed without stating what is different this time.
- **docs/DECISIONS.md** records irreversible and pre-registered choices (palette freeze, base model, reward weights, gates) with date and rationale.
- **Test oracle:** `pytest -q` (`make test`) runs before every commit. It covers:
  - the parser and compiler;
  - determinism: the same LoopScript gives bit-identical PCM (`pcm_sha256`) on the same machine. The cluster render is canonical (LESSONS 38). Hash the samples, not the file: libsndfile writes a timestamp into float WAV headers (`src/lofi_rl/render/wavio.py`);
  - verifiers on hand-built cases;
  - the guard and sync safety.

  `make junk-rate` is the progress oracle for Phases 1–2: 500 samples → gate pass %, verifier distributions, and 50 clips to `inbox/`.
- **Metrics.** Never report a metric that wasn't read from a file in this session.
- **Sound.** Never state how something *sounds*. Queue clips to `inbox/<date>/<batch>/` with a manifest, send Shresth a push notification, and write "needs ear check". Ear checks are his. Approvals are recorded as a line `APPROVED by Shresth Rana <yyyy-mm-dd>` (for example in `gates/GATE_palette.md`).
- Nakano's rule applies: keep checking and verifying. Agents fabricate; don't.

## 7. Git

- Laptop only. Small commits after each unit of work, with messages that say what changed and what the numbers were. Commit density matters for the course.
- Claude may commit and push to `origin main` once `pytest -q` passes. Never force-push.
- Never add an AI co-author or attribution trailer to any commit; this overrides tool defaults.
- Never commit `.env`, WAVs/FLACs, checkpoints, votes (`data/`), non-commercial assets, or anything under `runs/`.
- Votes live in the separate private repo `slipstr34m/lofi-rl-votes`, nested at `data/votes/` (`make backup-votes`).

## 8. What we are building (summary; details live in docs/)

- **Policy:** Qwen3-4B-Instruct-2507 + LoRA (r 16), trained with GRPO via TRL using `use_vllm`. A `reward_funcs` callable in the policy env calls a scorer server running in the scorer env.
- **Artifact:** LoopScript (`docs/DSL.md`) → compile (`docs/COMPILER.md`) → render → features → reward. Rendering uses FluidSynth for pitched instruments, a numpy one-shot sampler for drums, a fixed pedalboard chain, a rain bed at −Δ LU, −14 LUFS, and a seamless two-cycle render.
- **Briefs:** a city-at-night scene plus two knobs.
  - energy: 3 levels;
  - mood: melancholy = aeolian, wistful = dorian, hopeful = ionian (mood picks the mode);
  - tempo per brief within 60–95 BPM, centered on 75–90; 12 tonics.
- **Reward:** `gate × Σ w(t)·[theory, taste RM, knobs, Audiobox CE, CLAP theme] − penalties`.
  - Curriculum: theory 0.5→0.1 and taste 0.2→0.6 over the first 30 steps; knobs fixed at 0.15.
  - Audiobox: PQ is a junk floor; CE earns 0.05 only if it predicts Shresth's held-out votes; PC never.
  - CLAP theme is scored on the music stem against contrast scenes.
  - Penalties: sax restraint (never fades), identical or near-identical loops within a group (cousins pass), > 3 identical bars, over-length code, near-copies of pool items, repairs.
- **Taste RM:** best+worst-of-4 votes → MaxDiff / Bradley-Terry heads over MERT-95M, CLAP and symbolic features (chosen by cross-validation), with an ensemble and a PAR-bounded reward. It is frozen within a run, refreshed between runs, and watched daily for hacks.
- **Palette:** `palettes/lofi_rain_v1.yaml`, frozen at the end of Phase 1 from Shresth's shootouts (DECISIONS.md D-017); it never changes after the freeze. The model writes scale degrees, drum step grids, chord symbols from the vocabulary, and instrument names from the list. It never writes pitches, tempo, key, effects, or the texture bed.

## 9. Phases and pre-registered gates

| Phase | Dates | Done when |
|---|---|---|
| 0 Setup | Oct 30 – Nov 2 | ssh/rsync/Makefile work; `smoke_env` passes on `debug`; skeleton committed and pushed |
| 1 Instrument | Nov 2 – 8 | DSL + compiler + renderer deterministic; shootouts done; 10 of 30 drafted loops kept by ear; palette_v1 frozen. **Gate:** ≥ 60 % of 500 base-model samples pass gates and ≥ 30 % are "fine" (= not embarrassing) in a 50-clip ear check. Fallbacks, in order: grammar-constrained decoding if syntax sinks it; SFT warm-start; edit-mode rollouts; stop. |
| 2 Taste | Nov 9 – 15 | Seed pool of ≈2,000 loops rendered. Triage of 300 plus best/worst votes; ≈1,500 pairs is the aim, but the gate runs on what exists Nov 15 and is re-checked nightly through Nov 17. **Gate: RM held-out agreement ≥ 0.9 × Shresth's self-consistency ceiling q** (q from hidden repeats; absolute agreement reported against 70 %). Audiobox, CLAP and verifiers wired. GRPO smoke (10 steps, 1 GPU) runs end to end. |
| 3 First run | Nov 16 – 22 | 100 steps; sub-reward logs; hacks catalogued and gated; **Nakano meeting around Nov 19** (deck by Nov 17) with the curve, the step-0 vs step-100 pair, an architecture diagram and the SU ask |
| 4 Runs + science | Nov 23 – 29 | Long runs with nightly votes. Ablations: pairwise vs absolute; taste RM on vs off; curriculum on/off. Scaling: render throughput vs cores; step time vs GPUs, on whichever config is faster with the shorter queue. |
| 5 Show | Nov 30 – Dec 6 | Final blind A/B Nov 30 – Dec 2; embedding-map trajectory, spectrogram-per-step strip, knob-obedience curves, progression video, README with who-did-what |
| 6 Buffer | Dec 7 – 11 | polish, commits, submit |

Final pre-registered claims:
- **Preference:** in a blind, forced-choice A/B on 100 held-out briefs, Shresth picks step N over step 0 ≥ 70 % of the time. Step N is the checkpoint his votes favor on separate dev briefs, fixed before Nov 30.
- **Obedience:** knob obedience ≥ 90 % on held-out briefs.
- **No collapse:** mean embedding distance **across** held-out briefs at step N stays ≥ 70 % of its step-0 value. Same-brief cousins are allowed.
- **Scaling:** efficiency is reported as measured, whatever it is.
- **Secondary, not gated:** the "would play while working" ♥ rate, step N vs step 0, in blind playlist mode.

## 10. Session rituals

- **Start:** read CLAUDE.md → docs/TASTE.md → CHANGELOG.md → `make pull` → `make status` → state the next unit of work in one line.
- **End:** CHANGELOG entry → `pytest -q` → commit (and push) → `make sync` if code changed → list anything waiting on an ear check or an approval (with a push notification if Shresth is away).
- **Autonomy:** a scheduled local session runs this ritual every few hours and advances the next unit of work.
  - It stops and pings Shresth when blocked: an ssh failure, `NEEDS_APPROVAL`, a pending ear check, or failing tests.
  - It never self-approves a job type over 500 SU, never deletes `runs/`, never touches a frozen palette, and never commits votes or secrets.

## 11. Hard rules

1. No git, no secrets, no heavy compute on the login node. Ever.
2. No deleting anything under `runs/` on the cluster unless asked.
3. No palette change after the freeze. Ever (Shresth, 2026-10-30).
4. No claims about how anything sounds; queue clips.
5. No metric without the file path it was read from.
6. Every job type is registered in `jobs/APPROVED.yaml`. Ask Shresth first for anything over 500 SU, any changed type, and any run over its estimate.
7. If ssh prompts for Duo or hangs: stop and ask.
8. No non-commercial asset is ever committed (fetch-only); votes live only in `data/votes/` and its private backups.
