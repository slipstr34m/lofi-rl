# Playbook — laptop-driven projects on USC CARC Discovery

How a coding agent (or you) runs a research project where the laptop holds the code and every
credential, and the cluster does the heavy lifting. Read `LESSONS.md` for why each rule exists.

## 1. The operating model

```
 laptop (single writer)            GitHub (public hub)              Discovery (read-only mirror)
 ──────────────────────            ───────────────────              ────────────────────────────
 edit · test · commit · push  ───▶ main  ───(anonymous pull)───▶   clone at $KIT_CLUSTER_PATH
 gh / hf logins, API keys                                          envs, weights, scratch, jobs
 ratings, gates, publishing   ◀─── rsync pull (scripts/pull_artifacts.sh) ◀── $KIT_CLUSTER_ARTIFACTS
```

* The laptop is the only writer of the repository. The cluster clone never commits (push URL
  `no_push`); results come back as files and are committed from the laptop.
* The cluster holds **no credentials of any kind**: anonymous clone, ungated model weights, no tokens.
* Every remote command goes through `scripts/remote.sh` (pulls `main` first, prints the commit);
  every job through `scripts/sbatch_guard.sh` (budget, concurrency, size, time limits); every result
  through `scripts/pull_artifacts.sh`.
* `CHANGELOG.md` is the agent's memory between sessions; `CLAUDE.md` is its plan and rules.

## 2. One-time laptop setup

```bash
brew install micromamba gh                      # micromamba: keep envs OUT of the Homebrew Cellar
echo 'export MAMBA_ROOT_PREFIX=~/micromamba' >> ~/.zshrc   # every micromamba call needs it; agent shells are fresh
gh auth login                                   # repo scope; delete_repo only if you ever need it
```

`~/.ssh/config` (ControlMaster keeps one Duo-authenticated connection alive for hours):

```
Host discovery discovery.usc.edu
  HostName discovery.usc.edu
  User <cluster-user>
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  ServerAliveInterval 60
  ServerAliveCountMax 5
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 4h
```

If a script fails with an auth error: run `ssh -fN discovery` (Duo) and retry. The agent must stop
remote work and ask you to do this rather than hang.

## 3. Starting a project

```bash
~/carc-workflow-kit/bootstrap.sh ~/my-idea my-idea          # copies the kit, fills placeholders, first commit
cd ~/my-idea
# fill the <FILL: ...> sections of CLAUDE.md
export MAMBA_ROOT_PREFIX=~/micromamba             # (already in ~/.zshrc after §2)
micromamba create -y -n my-idea-dev -f environment.yml
micromamba run -n my-idea-dev pytest -q
gh repo create <github-user>/my-idea --public --source=. --remote=origin --push
scripts/cluster_setup.sh facts       # verify partitions, GPU types, storage, SUs; paste into CHANGELOG.md
scripts/cluster_setup.sh netcheck    # compute-node network
scripts/cluster_setup.sh init        # dirs + anonymous clone + no_push
scripts/remote.sh 'scripts/sbatch_guard.sh jobs/build_cpu_env.sbatch'
scripts/remote.sh 'scripts/sbatch_guard.sh jobs/smoke_cpu.sbatch'          # Phase-0 oracle: tests pass on a node
# GPU work only (the serve env is this project's own; rebuild it after adding a dependency):
scripts/remote.sh 'scripts/sbatch_guard.sh jobs/build_serve_env.sbatch 0.32.0'
scripts/remote.sh 'scripts/sbatch_guard.sh jobs/fetch_weights.sbatch Qwen/Qwen3-VL-8B-Instruct'
```

Nothing from the first project survives on the cluster (it was wiped on 2026-10-26), so a new
project builds its envs and fetches its weights again; §11 lists the versions known to work.

Prove the guard refuses before trusting it (nothing can be submitted: `sbatch` is replaced):

```bash
scripts/remote.sh 'SBATCH_BIN=/bin/false scripts/sbatch_guard.sh -p gpu --gres=gpu:a100:2 -c 64 --mem=0 -t 47:30:00 jobs/smoke_cpu.sbatch'
# -> REFUSED ... would exceed max_project_sus     (a full A100 node for 47.5 h = 410,400 SU)
```

The guard also refuses a job whose name does not start with `<project>-` (the SU ledger counts only
those), an array whose tasks together exceed the cap or the job limit (every task is a job; a `%N`
throttle changes neither), and a job whose `--output` directory cannot be created. `--exclusive` is
costed as the whole node.

## 4. The daily loop

1. Edit and test on the laptop: `pytest -q` (with `set -o pipefail` in any `pytest | … && git commit`).
2. Commit and push (no AI attribution trailers).
3. Submit: `scripts/remote.sh 'VAR=x scripts/sbatch_guard.sh [overrides] jobs/<job>.sbatch args'`.
4. Watch in the background: `scripts/watch_job.sh <jobid>`; do laptop work meanwhile. Pending lines show
   SLURM's estimated start; exit status 0 means the job left the queue, 1 still queued, 2 the polls
   failed (ssh or squeue), which is never reported as completion.
5. Pull: `scripts/pull_artifacts.sh all` (curated), `path <scratch-subdir>` (one run's outputs) or
   `gate <n>` (the paths in `gates/GATE_<n>.files`, structure kept, all-or-nothing).
6. Write `CHANGELOG.md`: what was done (paths), every number with its log path, failed approaches
   and why, decisions, pins, open questions, SUs used (`python scripts/session_report.py` via remote.sh).
7. If something fails twice the same way, write the failed approach down before the third attempt.

## 5. Sizing jobs

| Need | Partition and shape | SU/min |
|---|---|---|
| quick checks, smoke tests | `debug`: ≤ 1 h, a 2×A40 node and CPU nodes (often drained) | 6 for 4 CPU / 8 GB |
| CPU work, env builds, downloads | `main`: ≤ 48 h, 64-thread EPYC nodes | 1 per CPU + 1 per 4 GB |
| one 8B model | `gpu`, `--gres=gpu:1 --constraint="a40\|a100\|l40s"`, 8 CPU, 48 GB | 28 |
| one 32B model (bf16, 66 GB) | `gpu`, `--gres=gpu:2` + `TP=2`, or one A100-80GB (`--constraint=a100-80gb`) | 56 |
| a full A100 node | `--gres=gpu:a100:2 -c 64 --mem=0` | 144 |
| long CPU runs | `oneweek`: ≤ 7 days, no GPUs (raise `max_job_hours` with the human first: the guard caps every job at 47.5 h) | — |
| a sweep of many small tasks | one job with a worker pool (`kit.bench`-style), not an array: every array task counts as a job | — |

Short jobs start sooner on a busy partition: other jobs' age priority holds the queue, and short,
accurately timed jobs fit the backfill gaps (`squeue --start -j <id>` shows SLURM's estimate).

Long runs chain (`jobs/chained_template.sbatch` + `kit.chain.run_units`): the controller works in
units (a generation, an epoch, a batch) under `$KIT_CLUSTER_SCRATCH/runs/<name>/`, writes each unit's
completeness marker last and then moves an atomic `latest` pointer, resumes only from the newest
complete unit, and starts a unit only if max(1.25 × the last unit's wall time, 120 s) fits before
SLURM's end time minus a reserve for teardown and the resubmission (never a fixed margin). The job
checks `DONE`/`STOP` before serving a model, so a successor exits in seconds, and resubmits itself
through the guard with `--dependency=afterany:$SLURM_JOB_ID`, SLURM tools run under
`env -u LD_LIBRARY_PATH`. `max_units` is required and writes `DONE` when reached, so a smoke run is a
small `max_units` and never chains; a crash never resubmits; a link that cannot fit one unit writes
`STOP`. The guard sees one link at a time, so size `max_units` for the whole chain's cost, and copy
what matters from the run directory to `$KIT_CLUSTER_ARTIFACTS` (scratch is purged). Try the
machinery first with the demo controller: `MODULE=kit.chain` (see the template's header).

## 6. Serving models with vLLM

```bash
MODE=serve source scripts/job_env.sh            # this project's vLLM env, offline HF cache, $PY
TP=2 source scripts/serve_vllm.sh Qwen/Qwen3-VL-32B-Instruct     # or serve_vllm_multi.sh <model> <n_gpus>
"$PY" -m mypkg.run --server "$VLLM_URL" --model Qwen/Qwen3-VL-32B-Instruct
stop_vllm
```

In Python: `from kit.vllm_client import make_client; c = make_client(url_or_urls, model)` then
`text, finish = c.chat_full(messages, max_tokens=..., temperature=..., seed=...)` from a thread pool
(32–64 requests in flight per GPU). Record `finish_reason == "length"` as truncation.

GPU jobs run only in the serve env, so everything they import must be installed there:
`build_serve_env.sbatch` installs the project with its `pyproject.toml` dependencies (held to vLLM's
versions), imports every module of the project package and checks every `KIT_SERVE_BINARIES`
executable; binaries such as ffmpeg can come from the CPU env, whose `bin/` `job_env.sh` appends to
`PATH` in serve mode. A failure there costs a CPU job, not a GPU queue wait. To look inside a running
job: `srun --jobid <id> --overlap --pty bash`.

## 7. Human gates and the CHANGELOG

* A gate = files pulled into `artifacts/review/<gate>/`, a `gates/GATE_<n>.md` (what to look at, the
  question, how to approve), a commit, a message to the human — then keep working on anything the
  gate does not block. Approval is an explicit "approve" in chat (recorded in the file by the agent)
  or an approval line of its own, the name and the date written bare, e.g. `APPROVED by Ada Lovelace
  2026-12-09` (`kit.gates.gate_approved`; a plain grep also matches the file's own "how to approve"
  line). The gate's How-to-approve section comes from `python -m kit.gates --how-to-approve --name
  "Ada Lovelace"`: the line written out with a real date, never a placeholder, which gets copied
  brackets and all (LESSONS 55). `python -m kit.gates gates/GATE_<n>.md --name "Ada Lovelace"` names
  what to change in a line that starts as an approval but is not one. Praise is feedback, not approval.
* Publishing always needs an explicit go, every time: `python -m kit.publish` is a dry run unless given
  `--go` and an approved gate file.
* Never change a scoring rule, threshold or prompt silently: a CHANGELOG row with before/after evidence.

## 8. Quality control for agent-written code

After a burst of code and before the cluster spends SUs on it: run a multi-agent review (finders per
area → three independent verifiers per finding with different lenses → majority) and add one
regression test per confirmed finding. Treat a finding whose verifiers failed as unverified, not
rejected.

## 9. CSCI 596 "concepts applied"

Measure, don't assert: strong scaling of the parallel stage (`jobs/bench_scaling.sbatch` + `scripts/
scaling_study.py`: workers 1…64 on one node; `jobs/bench_scaling_2node.sbatch` for the two-node
point), throughput vs GPUs for any model serving (`jobs/bench_vllm.sbatch`), wall time per pipeline
stage, and the job-chaining/checkpoint story. Each claim in the README cites a CSV and the job that
produced it.

```bash
scripts/remote.sh 'TASK=<module:function> scripts/sbatch_guard.sh jobs/bench_scaling_2node.sbatch 2000'
scripts/remote.sh 'scripts/sbatch_guard.sh --constraint=a40 --gres=gpu:2 -c 16 --mem=96G jobs/bench_vllm.sbatch 400'
scripts/pull_artifacts.sh all
python scripts/scaling_study.py --glob 'artifacts/scaling/bench_<job>_w*.json' --x nodes --tag nodes_v1
python scripts/scaling_study.py --glob 'artifacts/scaling/vllm_<job>_g*.json' --x gpus --metric calls_per_min --tag gpu_v1
```

Each job measures its whole series in one allocation (the two-node job runs its own one-node
reference first; the GPU job loads the model once and measures 1…G replicas), so every point of a
series shares hardware.

## 10. Ending a project

1. Cancel your queued jobs (`squeue -u $USER`, then `scancel <ids>` — your own ids only).
2. Final `CHANGELOG.md` status entry (what exists, why it stopped), push.
3. GitHub: archive (`gh repo archive`), make private, or delete (`gh auth refresh -s delete_repo` first).
4. Cluster: delete the clone, the project's envs (CPU and serve: both are per project), artifacts and
   scratch. The weight cache `$KIT_CLUSTER_HF_HOME` sits outside every project directory and can be
   kept for the next project. Ask before deleting.
5. Laptop: remove the project's micromamba env and directory once nothing in it is needed.

The first project ended this way on 2026-10-26: its GitHub repo was made private, and everything on
the cluster was deleted at the human's choice, including the 84 GB weight cache and the vLLM env.

## 11. Verified cluster facts (2026-10-25/26; re-verify with `cluster_setup.sh facts`)

* Partitions: debug (1 h; per user 48 CPUs, 4 GPUs, 5 jobs), main (48 h), gpu (48 h; per user 36 GPUs,
  400 CPUs, 36 running and 100 queued jobs), oneweek (168 h, no GPUs), largemem (168 h, 1.5 TB),
  htcondor.
* SU rules: https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs and
  https://www.carc.usc.edu/user-guides/advanced-hpc-programming/gpu-programming (other guessed CARC
  URLs 404). sacct is the ledger; `myaccount` drifts, `sshare` decays, `sreport` lags (LESSONS #42).
* GPU nodes: A100 ×2 (64 CPUs, 256 GB; features `a100-40gb` or `a100-80gb`), A40 ×2 (32–64 CPUs,
  256 GB), L40S ×3–4 (64 CPUs, 768 GB), V100 ×2, P100 ×2. Billing weights: CPU 1, Mem 0.25/GB,
  A100/A40/L40S 8, V100/P100 4.
* Storage: `/home1/$USER` 100 GB; `/project2/<account>` 1 TB shared by the class; `/scratch1/$USER`
  ~10 TB (purged). Node-local `$TMPDIR` in jobs is a 126 GB tmpfs.
* Modules: `conda/25.11.0` (miniforge), cuda up to 13.0, ffmpeg up to 7.1; system git 2.43.
* Compute nodes: Rocky Linux 8.10, glibc 2.28, NVIDIA driver 580 (CUDA 13), outbound internet, no proxy.
* Allocation `anakano_429`: 30,000,000 SU shared by the class; fair share ~0.4 for one student, which
  is not what makes GPU jobs wait: other jobs' age is (PriorityMaxAge 7 days, weight 7,000; LESSONS #3).
* Known-good versions (first project, 2026-10-25/26): vllm 0.32.0, torch 2.13.0+cu130, huggingface_hub
  2.3.0, python 3.12 (serve env); models, all ungated: Qwen/Qwen3-VL-8B-Instruct
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (17 GB), Qwen/Qwen3-VL-32B-Instruct
  `0cfaf48183f594c314753d30a4c4974bc75f3ccb` (66.7 GB, Apache-2.0), yuvalkirstain/PickScore_v1
  `a4e4367c6dfa7288a00c550414478f865b875800` (3.9 GB; the card declares no licence),
  laion/CLIP-ViT-H-14-laion2B-s32B-b79K `1c2b8495b28150b8a4922ee1c8edee224c284c0c` (MIT).
* Added on 2026-10-26 after the cluster was wiped, so not yet run on Discovery: the `sprio`,
  `squeue --start`, `sacctmgr` QOS and `sinfo -p main` lines of `cluster_setup.sh facts`. Run them in
  session 1 and record what they print.
