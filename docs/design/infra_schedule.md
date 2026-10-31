# lofi-rl infrastructure and schedule design (Phase 0 to Nov 22)

**Sources.**
- The kit at HEAD 718def1, cited as `file:line`.
- The kit, sound and taste reports in the scratchpad.
- The HF and PyPI APIs, read 2026-10-30.
- `man rsync` and `make --version` run on this laptop.

Nothing below has run yet, and every estimate is labelled as one.

**Default calls.** All are applied with no disagreement. There are two refinements:
- The scorer env gets its own torch 2.8, because audiobox needs torchaudio and nothing in the scorer needs vLLM's torch.
- The scorer server talks over an AF_UNIX socket, so there are no ports to collide (LESSONS 17).

## 1. Phase 0 bootstrap

```bash
B=$(mktemp -d)/lofi-rl && ~/carc-workflow-kit/bootstrap.sh "$B" lofi-rl   # target must be empty (bootstrap.sh:25)
tar -C "$B" -cf - <keep+adapt paths> | tar -C ~/csci596-project -xf -      # project CLAUDE.md untouched
make test && git init -q -b main && git add -A && git commit -qm "chore: skeleton from carc-workflow-kit 718def1 (rsync mirror)"
```

**Keep:**
- `scripts/{_cluster_ssh.sh,watch_job.sh,scaling_study.py}`
- `src/kit/{chain,gates,hf_cache,bench}.py`
- `jobs/bench_scaling{,_2node}.sbatch`, plus their tests
- LICENSE
- `docs/kit/{PLAYBOOK,LESSONS}.md`, as reference only; CLAUDE.md wins any conflict

**Adapt:**
- `remote.sh` and `job_env.sh`, shown below.
- `sbatch_guard.sh`: add the registry and target checks from §3, and set `GUARD_LOG=$REMOTE_KEEP/submissions.log`.
- `pull_artifacts.sh`: keep the `path` and `gate` modes, rooted at `$REMOTE_SCR`.
- `cluster_setup.sh`:
  - `facts` adds `myquota`, `ls -ld /project/anakano_429 /project2/anakano_429`, `module avail fluidsynth uv` and `python3 -V`.
  - `init` becomes mkdir only.
- `session_report.py`: stdlib only and safe on Python 3.6.
- Renames:
  - build_serve_env becomes `jobs/env_policy.sbatch`.
  - build_cpu_env becomes `env_scorer`.
  - fetch_weights gains a `#files=` mode.
  - chained_template becomes `grpo` in Phase 2.
- Tests:
  - `test_job_scripts` uses prefix `lofi-` and requires `MODE=policy` for GPU jobs.
  - `test_laptop_scripts` drops git pull.
  - `test_sbatch_guard` adds registry and target cases.
- Config:
  - `config/budget.yaml`.
  - `kit.env` becomes `config/cluster.env`.
  - `.gitignore`.
  - `pyproject.toml` requires Python ≥ 3.12 and adds markers `laptop`, `cluster` and `gpu`, with `addopts -m "not cluster and not gpu"`.
  - The CHANGELOG Status block shows the mirror stamp instead of the clone commit.

**Drop:**
- The git transport (remote.sh:22-30) and the clone step in `init`.
- `netcheck`, which runs an srun the guard never sees.
- environment.yml and micromamba on the laptop.
- `serve_vllm*`, `vllm_client`, `bench_vllm`, `gpu_vllm_template`, `publish.py`, `smoke_cpu`, `artifacts/` and the templates.

**`config/cluster.env`** holds literal values only, so both bash and make can read it:
- `REMOTE_ROOT=/project2/anakano_429/shresthr` (to be confirmed with myquota, LESSONS 6)
- `REMOTE_PROJ=/project2/anakano_429/shresthr/lofi-rl` is the mirror.
- `REMOTE_KEEP=/project2/anakano_429/shresthr/lofi-rl-keep` holds what can't be rebuilt: votes/, ckpt/ and submissions.log. It sits outside the `--delete` mirror.
- `REMOTE_SCR=/scratch1/shresthr/lofi-rl` holds logs/, runs/, inbox/, hf/ and cache/.

**`scripts/remote.sh`** drops git and keeps the kit's quoting:
```bash
REMOTE_CMD="cd '$REMOTE_PROJ' && echo \"[remote] mirror=\$(cat .sync-stamp) laptop=$($REPO/scripts/stamp.sh|cut -d' ' -f1-3)\" && {
$*
}"
exec ssh -o BatchMode=yes -o ConnectTimeout=30 discovery "bash -lc $(printf '%q' "$REMOTE_CMD")"
```

**`scripts/stamp.sh`** prints `commit=<sha12> dirty=<0|1> diff=<sha256(git diff HEAD)[:12]|-> at=<iso>`. `make sync` writes it to `.sync-stamp`, which is gitignored but synced.

**`scripts/job_env.sh`** replaces `jobs/env.sh`, never calls git, and is sourced with `MODE=policy|scorer|tools`:
- It runs `. config/cluster.env; cd $REMOTE_PROJ`. It never trusts SLURM_SUBMIT_DIR, because a chained link's submit dir is a dead tmpfs.
- It snapshots the code with `rsync -a --exclude-from=.rsyncignore ./ $TMPDIR/code/; cd $TMPDIR/code`, so a later sync can't change code under a running job.
- It exports `PYTHONPATH=$TMPDIR/code/src`, `HF_HOME=$REMOTE_SCR/hf`, `HF_HUB_OFFLINE=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `OMP_NUM_THREADS=1`, `UV_LINK_MODE=copy` and `PATH=$REMOTE_PROJ/opt/bin:$PATH`.
- It sets `PY=$REMOTE_PROJ/.venv-$MODE/bin/python` and echoes node, CPU model, `$PY` and the stamp.
- Every run copies the stamp to `summary/stamp.txt`. A chain link whose mirror stamp has changed writes STOP, unless `ALLOW_CODE_CHANGE=1`.

## 2. Makefile, .rsyncignore, .gitignore

**GNU rsync is required** (`brew install rsync`). `/usr/bin/rsync` is openrsync, which speaks only protocol 29. Its man page documents no `/./` anchor, no `--mkpath` and no `--protect-args`, and the LESSONS 47 fixes assume GNU filter semantics. GNU rsync also matches what runs on Discovery.

macOS make is 3.81, which ignores `.SHELLFLAGS`, so the one piped line sets pipefail itself.

```make
SHELL:=/bin/bash
include config/cluster.env
RSYNC:=/opt/homebrew/bin/rsync
RS:=$(RSYNC) -az -e 'ssh -o BatchMode=yes -o ConnectTimeout=30'
SYNC:=$(RS) --delete --max-delete=200 --exclude-from=.rsyncignore
sync:
	$(RSYNC) --version | grep -q 'version 3\.'
	scripts/stamp.sh > .sync-stamp
	set -o pipefail; $(SYNC) -n -i ./ discovery:$(REMOTE_PROJ)/ | scripts/check_deletes.sh  # own list: .venv-*/ opt/ ckpt/ logs/ hf/ assets/big/
	$(SYNC) ./ discovery:$(REMOTE_PROJ)/
pull:
	mkdir -p runs inbox
	$(RS) -m --include='/*/' --include='/*/summary/***' --exclude='*' discovery:$(REMOTE_SCR)/runs/ runs/
	$(RS) discovery:$(REMOTE_SCR)/inbox/ inbox/
test:
	.venv/bin/python -m pytest -q
junk-rate: sync        # MODEL=<repo> [N=500]
	scripts/submit.sh jobs/junk_rate.sbatch $(MODEL) $(or $(N),500)
submit: sync           # JOB=jobs/x.sbatch [OVR='-t 0:20:00'] [ARGS=...]
	scripts/submit.sh $(OVR) $(JOB) $(ARGS)
status:
	scripts/remote.sh 'squeue -u $$USER -o "%.10i %.24j %.8T %.9M %.9l %R %S"; tail -n3 $(REMOTE_KEEP)/submissions.log; python3 scripts/session_report.py | tail -n1; myquota'
backup-votes:
	$(RS) data/votes/ discovery:$(REMOTE_KEEP)/votes/
	$(RSYNC) -a data/votes/ ~/lofi-rl-votes/votes/
	cd ~/lofi-rl-votes && git add -A && { git diff --cached --quiet || git commit -qm "votes $$(date +%F_%H%M)"; } && git push -q
```

- **pull** keeps the `runs/<run>/summary/**` structure, which fixes the flattening in CLAUDE.md §3. It never deletes anything on the laptop.
- **submit.sh** refuses unless `.sync-stamp` matches the current `stamp.sh` output on fields 1 to 3 and shows `dirty=0`. `ALLOW_DIRTY=1` overrides the dirty check. It then runs `scripts/remote.sh "scripts/sbatch_guard.sh$(printf ' %q' "$@")"`.
- **Inbox batches** are FLAC files plus a `MANIFEST.json` with one sha256 per clip, written last with `os.replace`. Readers verify the hashes before using a clip.
- **Votes** are append-only JSONL, and backups never use `--delete`.

`.rsyncignore`, one pattern per line:
```
.git/  .env  .env.*  *.key  *.pem  data/votes/  inbox/  runs/  .venv/  node_modules/  __pycache__/  .pytest_cache/  .DS_Store
*.wav  *.flac  *.pt  *.bin  *.safetensors  assets/big/
/.venv-policy/  /.venv-scorer/  /opt/  /ckpt/  /logs/  /hf/  /jobs/submissions.log
```
Line 3 is cluster state. Because those paths are excluded, `--delete` also leaves them alone.

`.gitignore`:
```
.env .env.* *.key *.pem data/votes/ inbox/ runs/ ckpt/ assets/big/ *.wav *.flac *.mp3 *.pt *.bin *.safetensors .sync-stamp jobs/submissions.log .venv/ __pycache__/ *.py[cod] *.egg-info/ .pytest_cache/ .DS_Store .idea/ .vscode/
```

Fetched assets (anything non-commercial like jRhodes3d, or over 50 MiB) live only in `assets/big/`:
- The committed `assets/FETCH.yaml` lists url, sha256, license and whether each one is redistributable.
- `jobs/fetch_assets.sbatch` downloads them on the cluster.
- `tests/test_licenses.py` fails if any tracked asset carries an NC license, or if any tracked path is under `data/votes/`.

## 3. Budget and job-type approvals

**`config/budget.yaml`:**
- `max_project_sus: 1200000` is the hard stop, with no override.
- `target_sus: 1000000`: past it, the guard exits 4 unless `PAST_TARGET_OK=<date of Shresth's OK in chat>` is set.
- `warn_at_fraction: 0.5`
- `max_concurrent_jobs: 8`
- `max_gpu_nodes_per_job: 2`
- `max_job_hours: 47.5`
- `project_start_date: "2026-10-31"`
- `account: "anakano_429"`
- `job_name_prefix: "lofi-"`

The formula is the kit's (sbatch_guard.sh:80-92, :312) and fails closed:

> SU = max(billing, CPU + GB/4 + 8·GPU) × minutes × tasks

**`jobs/APPROVED.yaml`** has one flow map per line, so the guard can parse it with awk on the login node:
```yaml
# sha = sha256[:16] over #SBATCH + non-comment lines (editing comments keeps the approval)
jobs/smoke_env.sbatch: {sha: <16hex>, max_su: 720, by: Shresth, date: 2026-10-31}
jobs/grpo.sbatch: {sha: <16hex>, max_su: 13440, max_run_su: 30000, by: Shresth, date: 2026-11-16}
```

**What the guard checks before calling sbatch:**
- If the entry is missing, the sha doesn't match, or the projected SU exceeds max_su, it prints `NEEDS_APPROVAL <new|changed|over-estimate> <job> projected=<n>`, exits 4 and submits nothing.
- Chain links are named `lofi-grpo-<run>`. The guard sums sacct for that name and refuses once the run passes max_run_su.

**On exit 4:**
1. Claude asks in chat.
2. After a yes, Claude appends the line and commits `approve <job> max_su=<n> (Shresth, chat <date>)`.

A test requires `by: Shresth` on every entry over 500 SU (see Open issue 7).

## 4. Environments on Discovery

Every build runs on `main`, wraps each step in `timeout`, and ends with an import walk of `lofi_rl` (LESSONS 4, 43).

**Lock files are compiled on the laptop:**
```
uv pip compile requirements/<env>.in --python-version 3.12 --python-platform x86_64-manylinux_2_28
```
That platform tag matches Rocky 8.10 with glibc 2.28. On the cluster each build runs `uv venv --python 3.12 $REMOTE_PROJ/.venv-<env> && uv pip sync <lock>` and logs its pins.

**`lofi-tools`** installs into `opt/`:
- uv 0.12.8 from the GitHub release tarball, sha256-checked, then `uv python install 3.12`.
- The static micromamba 2.10.0 binary, sha256-checked, then `micromamba create -p opt/fluidsynth -c conda-forge --override-channels fluidsynth=2.6.1`.
- A wrapper `opt/bin/fluidsynth` containing `exec env -u LD_LIBRARY_PATH $REMOTE_PROJ/opt/fluidsynth/bin/fluidsynth "$@"`.
  - Conda's RPATH finds its own libraries, so nothing conda-related leaks into the job (LESSONS 8).
  - Renders use the CLI float path (`-ni -R 0 -C 0 -r 44100 -O float -F`), with a fresh process per render.
  - That leaves no dither state carried between renders and keeps conda's libstdc++ out of Python.

**`lofi-env-policy`:**
- `vllm==0.32.0`, `trl==1.17.1`, `transformers>=5.19,<5.20`, `peft>=0.17`, `accelerate>=1.11`, `datasets>=4.7`.
- torch is whatever vllm pins; the kit recorded 2.13.0+cu130 on driver 580 (build_serve_env.sbatch:12).
- flashinfer-cubin is installed only on an exact version match.

**`lofi-env-scorer`:**
- Pins: `torch==2.8.0` and `torchaudio==2.8.0` (PyPI CUDA 12.x build), `transformers==4.43.4`, `numpy==1.26.4`, `laion_clap==1.1.7`, `audiobox_aesthetics==0.0.4`, `pedalboard==0.9.26`, `pyloudnorm==0.2.0`, plus `pretty_midi`, `mido`, `soundfile`, `scipy` and `pytest`, with `WANDB_MODE=disabled`.
- transformers is pinned to 4.43.4 because `modeling_MERT.py` imports `HubertPositionalConvEmbedding`, which reads `conv_pos_batch_norm` from 4.44 on. The fallback is the latest 4.x with that config field patched.
- numpy is pinned below 2 because laion_clap 1.1.7 declares `numpy<2.0.0,>=1.23.5` on PyPI.

**`lofi-fetch`** downloads anonymously through a throwaway uv venv with huggingface_hub. All four repos are ungated, and the revisions are from the HF API:

| Repo @ revision | License | Notes |
|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554` | Apache-2.0 | about 8 GB of BF16 |
| `m-a-p/MERT-v1-95M@12af15fef9d0ac838c3f475bfbbf26d2060dd4f5` | CC-BY-NC | json, py and bin files; the fairseq .pt is skipped |
| `lukewys/laion_clap@b3708341862f581175dba5c356a4ebf74a9b6651#files=music_audioset_epoch_15_esc_90.14.pt` | CC0 | needs the new `#files=` mode, since the kit skips `*.pt` |
| `facebook/audiobox-aesthetics@9b1dd8e5df9af7216e836a98974fe3b82c56ded6#files=config.json,model.safetensors,checkpoint.pt` | CC-BY-4.0 | |

After the fetch, `lofi_rl.scorer.patch_clap` writes a tensors-only state_dict without `text_branch.embeddings.position_ids` (CLAP issue #127) and logs its sha256. The Granite 4.1 8B alternate is fetched on Nov 6.

**Paths.** Run `facts` before `init`, and use whichever of /project2 or /project myquota shows as writable. Record the choice in cluster.env, the CHANGELOG and D-002. The envs take about 20 GB of the 1 TB quota the whole class shares.

## 5. smoke_env and SU estimates

`jobs/smoke_env.sbatch` runs as `lofi-smoke-env` on debug with `-c 8 --mem=32G --gres=gpu:a40:1 -t 0:30:00`. If the debug A40 is drained, it falls back to `OVR='-p gpu --gres=gpu:1 --constraint=a40|a100|l40s'`. Each check runs as its own process and writes to `runs/smoke-<id>/summary/{smoke,timings}.json`, `golden.sha256` and `pins-*.txt`.

1. Node facts (CPU model, GPU, driver, glibc), the stamp, and whether HF is reachable.
2. Imports and a visible bf16 GPU in both envs, plus the import walk.
3. `pytest -q -m "not laptop"` in the scorer env.
4. `fluidsynth --version` reports 2.6.1. `tests/fixtures/scale.mid` is rendered twice and the two sha256s must match, with RMS above −60 dBFS.
5. `tests/fixtures/hello.loop` goes through the Phase-0 stub compiler (rerun in Phase 1 with the real one). Checks:
   - sample count equals round(1920/BPM × 44100);
   - integrated loudness is −14.0 ± 0.1 LUFS and true peak ≤ −1 dBTP;
   - two processes produce the same sha256, which becomes the cluster golden hash.

   The clip is queued to inbox marked "needs ear check".
6. vLLM loads `LLM(snapshot, bfloat16, gpu_memory_utilization=0.45, max_model_len=4096, enforce_eager=True)` and produces one completion at temperature 0. It logs load time, tok/s and finish_reason.
7. Scorer: the server runs in the scorer env, listening on an AF_UNIX socket in `$TMPDIR` with a random authkey. The policy env sends it 8 LoopScripts. It records:
   - render seconds per loop with 1 worker, and wall time for the group with 7 workers;
   - MERT, CLAP and Audiobox ms per clip;
   - CLAP run twice over fixed bar-aligned windows of ≤ 10 s, which must give max|Δ| = 0;
   - round-trip latency.

   The job ends with `SMOKE_OK`.

**SU estimates** use the kit formula. The assumed timings are t_loop = 3 s, t_embed = 0.3 s per clip and t_step = 1.5 min; the smoke and the GRPO smoke replace them with measurements.

| Job (`lofi-…`) | Shape | SU/min | Limit → max | Expected |
|---|---|---|---|---|
| tools | main 4c/8G | 6 | 0:30 → 180 | 60 |
| env-policy | main 8c/32G | 16 | 1:00 → 960 | 320 |
| env-scorer | main 8c/32G | 16 | 1:00 → 960 | 240 |
| fetch | main 4c/32G | 12 | 1:00 → 720 | 180 |
| smoke-env | debug 8c/32G/A40 | 24 | 0:30 → 720 | 360 |
| render per 1k loops | main 32c/64G | 48 | 0:30 → 1,440 | 48·(1+1000·t_loop/1440) ≈ 150 |
| embed per 1k clips | gpu 8c/32G/1 | 24 | 0:30 → 720 | 24·(3+1000·t_embed/60) ≈ 190 |
| junk-rate (500) | debug or gpu 16c/64G/1 | 40 | 0:30 → 1,200 | 480 |
| rm-train | main 8c/16G | 12 | 0:20 → 240 | 60 |
| grpo-smoke (10 steps) | debug 32c/64G/A40 | 56 | 0:45 → 2,520 | 56·(10+10·t_step) ≈ 1,400 |
| grpo (100 steps) | gpu 32c/64G/1 | 56 | 4:00 per link → 13,440 | 56·(10+100·t_step) ≈ 9,000 |

Projected spend through Nov 22 is about 35k SU, roughly 3.5% of the 1.0M target.

## 6. Laptop setup

- `brew install fluid-synth rsync && brew pin fluid-synth`, so both machines stay on 2.6.1 (2.6.0 changed tuning).
- `uv venv --python 3.12 .venv && uv pip sync requirements/laptop.lock.txt`.
  - The lock holds numpy==1.26.4, scipy, pretty_midi, mido, soundfile, pedalboard==0.9.26, pyloudnorm==0.2.0, pyyaml, pytest and the rating-UI deps.
  - There is no torch.
- Laptop tests check determinism on this machine, exact length and LUFS.
- Golden-hash tests are marked `@pytest.mark.cluster`.
- Every clip Shresth hears is a cluster render (LESSONS 38).

## 7. CLAUDE.md edits and DECISIONS.md

**§1**
- Laptop row: "frontier-model API calls" becomes "seed loops written by Claude Code workflows (no API key)".
- Login-node row adds the read-only `sinfo`, `scontrol show`, `sprio` and `module avail`, plus the guard and session_report (python3 stdlib, under 5 s).
- Paths: use `/project2` once myquota confirms it, add `REMOTE_KEEP`, and point to `config/cluster.env`.

**§2.** Every remote command goes through `scripts/remote.sh`. The ssh block becomes the laptop's real one (`ControlPersist 4h`, `cm-%r@%h:%p`).

**§3.** The Makefile recipes and `.rsyncignore` above, using GNU rsync.

**§4**
- Logs go to `--output=/scratch1/%u/lofi-rl/logs/%x-%j.out`.
- `scripts/job_env.sh` takes `MODE`, and jobs are submitted only with `make submit`.
- New rule: every job script needs an APPROVED.yaml entry. A new type, a changed type, or a run projected over its max_su needs Shresth's yes in chat.
- Chains resubmit from inside the running link through the guard (`afterany` plus DONE/STOP files), never as a pre-queued `afterok` job.

**§5.** Two venvs, built on `main` from laptop-compiled locks; the FluidSynth wrapper; the fetch list; and the smoke checks from §5 above.

**§6 and §10**
- Session start becomes CLAUDE.md, then **docs/TASTE.md (required)**, then CHANGELOG, then `make pull`, then `make status`.
- An ear check is an `inbox/<date>/<batch>/` folder with a manifest, plus a push notification.

**§7.** "Claude may commit and push to origin main after `pytest -q` passes. No AI co-author or attribution trailer; this overrides tool defaults. Never force-push. Votes are never committed."

**§8**
- A brief is a scene plus knobs, and the only scene is city at night.
- energy is low, mid or high. mood maps melancholy to aeolian, wistful to dorian, hopeful to ionian.
- Tempo is set per brief within 60–95, centred on 75–90.

**§9**
- The RM gate becomes: held-out agreement ≥ 0.90·q̂, where q̂ = (1+√(2p̂−1))/2 and p̂ is the agreement on hidden repeats from Nov 9–15. Absolute agreement is reported alongside, against 70%.
- The no-collapse claim moves across briefs: mean pairwise embedding distance across 100 held-out briefs (one loop each) at step N must be ≥ 70% of step 0. Loops from the same brief are allowed to be close cousins.
- Secondary measure, reported but not gated: the share of clips marked "would play while working" in the blind sitting, step N vs step 0, which runs Nov 30 – Dec 2.

**§11.** Rule 6 becomes the registry rule. Add "no non-commercial asset is ever committed" and "votes live only in data/votes/ and the private backups".

**Initial DECISIONS.md** (all dated 2026-10-31):

| ID | Decision |
|---|---|
| D-001 | rsync mirror plus stamp; no git off the laptop |
| D-002 | REMOTE_ROOT, from myquota |
| D-003 | 1.0M target and 1.2M hard stop, `lofi-` prefix, the job registry |
| D-004 | the env pins in §4 |
| D-005 | FluidSynth 2.6.1 rendered through the CLI float path; cluster renders are canonical |
| D-006 | Qwen3-4B-Instruct-2507 @cdbee75f, with Granite 4.1 8B as an alternate used only at the junk-rate gate |
| D-007 | the taste RM is frozen within a run, refreshed between runs, with a daily hack-watch |
| D-008 | the relative RM gate |
| D-009 | no-collapse measured across briefs |
| D-010 | votes stay private (KEEP plus a private repo) |
| D-011 | Claude may push, with no AI trailer |
| D-012 | NC assets are fetch-only; committed files are ≤ 50 MiB |

## 8. Schedule, Oct 31 to Nov 22

- **S** is what Shresth does, with listening minutes in brackets; his total stays at or under 60 a day.
- Every day he opens Duo once around 9:00 (2 min) and again whenever Claude sends a push. Ear checks are answered the same day.
- **C** is what Claude works on in parallel while waiting.
- **Pairs** is the running total of implied BWS-4 pairs, assuming about 200 per listening hour: 50 tuples × 5 pairs, minus 20% for hidden repeats, anchors and escape hatches.

| Date | S | C | Milestone · pairs |
|---|---|---|---|
| Sat 10-31 | Duo; approve first job types, brew installs, both repos (15) | bootstrap, copy, git init; facts, path check, init, sync; tools then env-policy and env-scorer (afterok), plus fetch; compile locks | skeleton committed |
| Sun 11-01 | Duo | stub render chain, scorer server and client; submit smoke | smoke #1 |
| Mon 11-02 | hello.wav (optional) | re-smoke, record actual SU; start DSL, parser, compiler; fetch-assets | **P0 gate** |
| Tue 11-03 | keys shootout and drums+bass shootout (30) | renderer v1, determinism tests; shootout renders to inbox | |
| Wed 11-04 | vamp and sax shootouts (25) | sax expression, humanize A/B; draft 30 loops | |
| Thu 11-05 | keep 10 of 30 (30) | verifiers, gates, knob checks, junk-rate pipeline; the 20 rejects become labelled negatives | 10 loops approved |
| Fri 11-06 | sign off palette (10) | freeze palette_v1; junk-rate dry run of 50; seed workflow, render about 2k | palette frozen |
| Sat 11-07 | 50-clip junk ear check (25), triage first half (35) | junk-rate 500 on both Qwen and Granite; embed the pool; draft the SFT fallback | |
| Sun 11-08 | triage second half (35), votes (25) | **P1 gate** from `runs/junk-*/summary/junk.json` plus his labels; build the vote queue | **P1 gate** · 80 |
| Mon 11-09 | votes, 4×15 | BT/RM code and learning curves; wire the reward functions | 280 |
| Tue 11-10 | votes | nightly rm-train builds the next day's queue; GRPO trainer | 480 |
| Wed 11-11 | votes | GRPO smoke #1 (RM v0) | 680 |
| Thu 11-12 | votes; book Nakano for Nov 19 (5) | GRPO smoke #2; estimate p̂ | 880 · **GRPO smoke** |
| Fri 11-13 | votes | hack signatures, plots, architecture diagram | 1,080 |
| Sat 11-14 | votes | learning curves at 200, 400 and 800 | 1,280 |
| Sun 11-15 | votes | **RM gate** (1,280 train, 200 held out) | 1,480 |
| Mon 11-16 | votes | if it passed: freeze RM v1 and submit 100 steps; otherwise re-check that night | 1,680 |
| Tue 11-17 | hack-watch (5), step 0 vs step 100 (15), votes (40) | sub-reward plots; catalogue hacks and gate them | run 1 done |
| Wed 11-18 | hack-watch (5), votes (55) | rerun with hack gates if needed; deck with curve, the step-0/step-100 pair, diagram, SU ask | deck ready |
| Thu 11-19 | **Nakano meeting**; review the deck (20) | apply feedback | **meeting** |
| Fri 11-20 | votes (fallback meeting date) | refresh to RM v2 | |
| Sat 11-21 | hack-watch (5), votes (55) | run 2 with RM v2; prepare the render-scaling benchmark | |
| Sun 11-22 | votes | CHANGELOG, plots | Phase 3 done |

## 9. Risk register

| # | Risk | Mitigation |
|---|---|---|
| 1 | GPU queue waits from age priority | Short, accurately timed jobs; accept any bf16 GPU; use debug for anything ≤ 45 min; keep renders off the GPU; submit before nights and weekends; read `squeue --start` |
| 2 | Version conflicts (MERT vs transformers 4.44, numpy<2, torchaudio, vLLM's transformers window) | Two venvs; manylinux lock files; import checks in CPU jobs; MERT config patch as fallback |
| 3 | Fewer than 200 pairs per hour, or q̂ < 0.78, which puts 70% out of reach | BWS-4 with early pick; the relative gate; learning curves; Open issue 5 |
| 4 | Phase 1 gate fails | The alternate is measured the same day; SFT warm-start pre-drafted for Nov 9–10; then edit-mode rollouts; then stop |
| 5 | `--delete` wipes cluster state | KEEP sits outside the mirror; excluded paths are protected; a separate dry-run check; `--max-delete` |
| 6 | Renders differ across machines (arm64 vs x86, SIMD, dither) | Cluster renders are canonical; float CLI with effects off; hash-based PRNG; log CPU model and add `--constraint` if hashes split |
| 7 | The Duo master drops while Shresth is away | BatchMode fails fast and triggers a push; laptop-only work is queued; long jobs go in inside the Duo window; chains resubmit on the cluster |
| 8 | Things change under running jobs (a sync, scratch purge, quota) | Code snapshot plus stamp; STOP on a stamp change; pull daily; copy checkpoints to KEEP; check myquota weekly |

## 10. Open issues

1. **Duo on the Phase 0 weekend.**
   - (a) Saturday and Sunday mornings, as planned.
   - (b) Sunday only, which moves the smoke to Nov 2 and P0 to Nov 3.
   - (c) Neither, which starts cluster work Nov 2 and puts Phase 1 two days late.
2. **Daily Duo routine.**
   - (a) Once each morning, re-opened when pushed (recommended).
   - (b) At fixed times, 9:00 and 18:00.
   - (c) Only when pushed.
3. **How to split the 1 h listening day.**
   - (a) 4×15 min, about 12 tuples each (recommended; BS.1116 suggests sessions of 20–30 min or less).
   - (b) 3×20 min.
   - (c) 2×30 min.
4. **Ear-check pushes.**
   - (a) At most 2 batched pushes a day, each 15 min or less (recommended).
   - (b) One push per batch.
   - (c) One evening digest.
5. **If there are fewer than about 1,500 pairs by Nov 15.**
   - (a) Move the RM gate and run 1 back 1–2 days, keeping the meeting on Nov 19 (recommended).
   - (b) Add an extra hour on Nov 14–15.
   - (c) Run on the current RM with the taste weight held at 0.2, labelled exploratory.
6. **Daily hack-watch.**
   - (a) Top 6 clips by reward, about 3 min (recommended).
   - (b) Top 4 plus bottom 4.
   - (c) Top 12.
7. **Job-type approvals.**
   - (a) Claude registers types up to 500 SU itself, and you approve larger ones once (recommended).
   - (b) You approve every type.
   - (c) Claude registers types up to 2,000 SU itself.
8. **RM refits that choose the next day's pairs.**
   - (a) A nightly cluster job (recommended).
   - (b) A CPU refit on the laptop that takes seconds, which needs a CLAUDE.md exception.
   - (c) Random pairs.
9. **Repos.**
   - (a) Public `slipstr34m/lofi-rl` plus private `slipstr34m/lofi-rl-votes` (recommended).
   - (b) Votes only on KEEP and Time Machine.
   - (c) A different GitHub account.
10. **Nakano meeting date.**
    - (a) Thu Nov 19 (recommended).
    - (b) Fri Nov 20.
    - (c) Tue Nov 17, showing step 0 and the GRPO smoke only.

### Critical Files for Implementation
- /Users/shresthrana/carc-workflow-kit/scripts/sbatch_guard.sh
- /Users/shresthrana/carc-workflow-kit/scripts/remote.sh
- /Users/shresthrana/carc-workflow-kit/scripts/job_env.sh
- /Users/shresthrana/carc-workflow-kit/jobs/build_serve_env.sbatch
- /Users/shresthrana/csci596-project/CLAUDE.md