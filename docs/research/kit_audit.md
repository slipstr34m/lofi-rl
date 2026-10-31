Audit of the carc-workflow-kit for reuse in lofi-rl (Phase 0)

Paths: K = /Users/shresthrana/carc-workflow-kit (git HEAD 718def1, clean tree, 3 commits, origin git@github.com:slipstr34m/carc-workflow-kit.git). P = /Users/shresthrana/csci596-project (only CLAUDE.md, 128 lines, not a git repo, no CHANGELOG.md yet). A reference like "bootstrap.sh:25" means K/bootstrap.sh line 25; "CLAUDE.md:NN" means P/CLAUDE.md.

I read these in full: README.md, README.project.md, PLAYBOOK.md, LESSONS.md, CLAUDE.template.md, CHANGELOG.template.md, bootstrap.sh, kit.env.example, environment.yml, pyproject.toml, .gitignore, config/budget.yaml, all 9 jobs/*.sbatch, all 14 scripts/* files, all 8 src/kit/*.py files. From tests I read test_kit_bootstrap.py, test_job_scripts.py, test_laptop_scripts.py and test_session_report.py in full, plus the harness sections of the other test files. The kit has no Makefile, no .rsyncignore and no requirements.txt.

== 1. bootstrap.sh <dir> <name> [--github-user U] [--cluster-user U] [--account A] [--human "Full Name"] ==

Steps, in order:
- :9 runs with `set -euo pipefail`. :10 sets KIT to the script's own directory. :11-12 TARGET and NAME are required. :15-23 parse the options; an unknown option exits 2.
- :14 defaults: `GH_USER=slipstr34m; CL_USER=shresthr; ACCOUNT=anakano_429; HUMAN="Shresth Rana"`.
- :24 the name must match [a-z0-9-] and not start with "-". "lofi-rl" passes.
- :25-27 the precondition check: if TARGET exists and `ls -A` is non-empty, it prints "exists and is not empty; refusing to overwrite anything" and exits 1. This runs before anything is created.
- :28 PKG is the name with "-" turned into "_" (lofi_rl). :29 `mkdir -p TARGET`.
- :30-31 local copy: `rsync -a --exclude .git --exclude __pycache__ --exclude .pytest_cache --exclude '*.egg-info' --exclude bootstrap.sh --exclude README.md --exclude tests/test_kit_bootstrap.py`. This copies .gitignore, config/, jobs/, scripts/, src/kit/, tests/, environment.yml, pyproject.toml and the templates.
- :33-36 renames: CLAUDE.template.md to CLAUDE.md, CHANGELOG.template.md to CHANGELOG.md, README.project.md to README.md, kit.env.example to kit.env.
- :37 `mkdir -p docs src/$PKG artifacts gates`. :38 moves PLAYBOOK.md and LESSONS.md into docs/.
- :39 writes src/$PKG/__init__.py. :40 writes artifacts/MANIFEST.json as `{"artifacts": []}`.
- :41 perl sets the pyproject name line to `name = "__PROJECT__"`.
- :42-64 writes an MIT LICENSE, "Copyright (c) <year> $HUMAN".
- :65-71 exports TODAY=$(date +%F). It then runs `grep -rlI` and `perl -pi` in place over every file, substituting __PROJECT__, __PKG__, __GITHUB_USER__, __CLUSTER_USER__, __ACCOUNT__, __HUMAN_NAME__ and __START_DATE__. Files affected: CLAUDE.md, CHANGELOG.md, README.md, kit.env, environment.yml, pyproject.toml, config/budget.yaml (start date, account, `job_name_prefix: "lofi-rl-"`), every job's --job-name and --output, and test files (test_job_scripts.py:81, and the dict keys in the bench tests).
- :72 `chmod +x scripts/*.sh`. :73 `git init -q -b main`. :74 `git add -A`. :75 `git -c commit.gpgsign=false commit -q -m "chore: bootstrap $NAME from carc-workflow-kit"`, using your own git identity and no AI trailer.
- :76-86 prints next steps only: fill the <FILL> sections; `export MAMBA_ROOT_PREFIX=~/micromamba`; micromamba create `$NAME-dev`; `gh repo create $GH_USER/$NAME --public --source=. --remote=origin --push`; cluster_setup.sh facts, netcheck and init; build_cpu_env then smoke_cpu.

What it does not do: no network access, no ssh, no env creation, no gh call.

Preconditions: TARGET is missing or empty (hidden files count, so a lone .DS_Store blocks it); bash, rsync, perl, git, grep and date are on PATH; a git identity is set (it is set on this laptop). If the commit fails, the target is left half-built.

On P: it refuses with exit 1 and touches nothing, because CLAUDE.md is present. It neither overwrites nor merges, and tests/test_kit_bootstrap.py:52-53 asserts exactly this. Workarounds:
- Bootstrap into a fresh directory and copy pieces across by hand.
- Or move CLAUDE.md aside, bootstrap P, then replace the generated template CLAUDE.md (which has <FILL> sections and conflicting rules) by hand.

== 2. Kit equivalents ==

make sync: none. The kit moves code with git:
- PLAYBOOK §1:8-21 laptop pushes to a public GitHub repo.
- cluster_setup.sh:40-41 an anonymous `git clone https://github.com/$KIT_GITHUB_REPO.git` with `git remote set-url --push origin no_push`.
- remote.sh:30 runs `cd '$KIT_CLUSTER_PATH' && git pull -q --ff-only origin $KIT_BRANCH && echo "[remote.sh] cluster HEAD=..." && { <cmd> }` before every remote command.
- remote.sh:22-28 exits 3 with "push first" if the laptop is ahead of origin.
- remote.sh:35 does `exec ssh $CLUSTER_SSH_OPTS host "bash -lc $(printf '%q' ...)"`.
- _cluster_ssh.sh:9 sets `-o BatchMode=yes -o ConnectTimeout=30`, so it fails fast instead of prompting for Duo. :10-16 fall back to the IP address with HostKeyAlias when the resolver fails.

make pull: scripts/pull_artifacts.sh `all | path <remote> [dst] | gate <n>`.
- :23 `RSYNC=(rsync -avzL --partial -e "ssh $CLUSTER_SSH_OPTS")`.
- all (:27-30): $KIT_CLUSTER_ARTIFACTS/ to artifacts/.
- path (:31-37): a path relative to $KIT_CLUSTER_SCRATCH, pulled to artifacts/review/<dst>/.
- gate (:38-69): gates/GATE_<n>.files, keeps directory structure, pulls nothing if any path is missing.

.rsyncignore: none. The closest is .gitignore:
- :2-5 `.env .env.* *.key *.pem`
- :8-17 `artifacts/review/ *.mp4 *.gif *.safetensors *.bin *.pt *.parquet checkpoints/ frames/ .cache/`
- :20 `jobs/submissions.log`
- :23-29 python entries, including `.venv/`
- It has no runs/, inbox/, *.wav, logs/ or assets/big/.

jobs/env.sh equals scripts/job_env.sh (`MODE=cpu|serve source scripts/job_env.sh`), which sources scripts/_kit_env.sh. _kit_env.sh:3-13 loads kit.env with `set -a` and refuses __PLACEHOLDERS__. job_env.sh then:
- :15 `PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}`
- :16 `HF_HOME=$KIT_CLUSTER_HF_HOME HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_HUB_DISABLE_TELEMETRY=1`
- :17 conda_pkgs and pip_cache on scratch
- :18 `module purge; module load usc >/dev/null 2>&1 || true; module load "$KIT_CONDA_MODULE"`
- :19-22 conda hook, activates the CPU or serve env, sets `PY="$CONDA_PREFIX/bin/python"`
- :25 serve mode appends the CPU env's bin/ to PATH
- :27 echoes `repo=$(git rev-parse --short HEAD)`, which runs git on the compute node
- :28-30 helpers model_rev, stop_vllm and `slurm() { env -u LD_LIBRARY_PATH "$@"; }`

build-env jobs:
- jobs/build_cpu_env.sbatch: main, 4 CPU, 16G, 0:45. :23-27 `timeout 40m conda env create|update -p $KIT_CLUSTER_CPU_ENV -f environment.yml`. :29 writes pins to $KIT_CLUSTER_SCRATCH/logs/pins-cpu-<job>.txt. :22 also runs git rev-parse. Ends with CPU_ENV_OK.
- jobs/build_serve_env.sbatch: main, 8 CPU, 32G, 1:30, argument `[vllm_version] [extra pip pkgs...]`, default 0.32.0 (:26).
  - :34 `conda create python=3.12`.
  - :37 `pip install vllm==$V "huggingface_hub[hf_xet]" pyyaml "$@"`.
  - :40-41 installs flashinfer-cubin only at an exact version match.
  - :44-46 `pip install -e . -c constraints` (constraints are the versions vLLM installed).
  - :50-66 imports every module of the project and kit, and checks every $KIT_SERVE_BINARIES is on PATH.
  - Ends with SERVE_ENV_OK.

smoke-env: jobs/smoke_cpu.sbatch (debug, 4 CPU, 8G, 0:15). :17 runs a urllib probe to huggingface.co. :18 runs `"$PY" -m pytest tests/ -x -q -p no:cacheprovider`. Ends with SMOKE_OK. It has no GPU, FluidSynth, vLLM or LUFS check. Related pieces:
- cluster_setup.sh netcheck (:34-36) runs a blocking `srun -p debug -n1 -c1 --mem=2G -t 00:05:00` curl probe from the login node. It bypasses the guard and is not named for the ledger.
- jobs/gpu_vllm_template.sbatch is the kit's GPU smoke: serve a model, then run MODULE.
- jobs/fetch_weights.sbatch: anonymous, ungated only (:26, :32-33), writes refs/main (:42-44). Its allow-list (:36-39) is `*.json *.txt *.model *.jinja *.py` plus *.safetensors, or *.bin if there are no safetensors. It skips *.pt and *.ckpt.

Header convention, verbatim (smoke_cpu.sbatch:1-15):
```
#!/bin/bash
#SBATCH --job-name=__PROJECT__-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch1/%u/__PROJECT__/logs/%x-%j.out
# Phase-0 oracle: ...
# Cost: 4 CPU + 2 (8 GB) = 6 SU/min -> at most 90 SU.
#   scripts/remote.sh 'scripts/sbatch_guard.sh jobs/smoke_cpu.sbatch'
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
MODE=cpu source scripts/job_env.sh
```

tests/test_job_scripts.py enforces these rules:
- :18 starts with #!/bin/bash.
- :19 output matches `/scratch1/%u/<proj>/logs/%x-%j.out`, and :67-75 it must equal the path implied by kit.env.
- :20 contains `set -euo pipefail`.
- :21 mentions sbatch_guard.sh.
- :22 matches `Cost: .*SU`.
- :23 sources job_env.sh or _kit_env.sh.
- :24-26 never writes artifacts/ outside $KIT_CLUSTER_ARTIFACTS.
- :27 no HF_TOKEN or API-key names.
- :28-29 any job with --gres=gpu must contain `MODE=serve source scripts/job_env.sh`.
- :32-41 no bare `sbatch` outside the guard.
- :78-82 job name is `--job-name=<KIT_PROJECT>-...`.

Cost lines, verbatim:
- smoke_cpu:11 "4 CPU + 2 (8 GB) = 6 SU/min -> at most 90 SU."
- build_cpu_env:12 "4 CPU + 4 (16 GB) = 8 SU/min -> at most 360 SU."
- build_serve_env:21 "8 CPU + 8 (32 GB) = 16 SU/min -> at most 1,440 SU."
- fetch_weights:14 "4 CPU + 8 (32 GB) = 12 SU/min -> at most 720 SU."
- gpu_vllm_template:19 "8 CPU + 12 (48 GB) + 8 (1 GPU) = 28 SU/min -> at most 560 SU for 20 minutes."
- chained_template:29-30 "28 SU/min -> at most 6,720 SU per 4-hour link, and EVERY link costs that again"
- bench_scaling:14 "64 CPU + 16 (64 GB) = 80 SU/min -> at most 1,600 SU."
- bench_scaling_2node:20 "2 nodes x (64 CPU + 16 (64 GB)) = 160 SU/min -> at most 1,600 SU for 10 minutes."
- bench_vllm:23-24 "28 SU/min -> at most 560 SU ... 2 GPUs ... 56 SU/min -> at most 1,120 SU."

Budget guard: scripts/sbatch_guard.sh is the only submit path.
- Formula (:80-92, :312-313): `max(billing, cpu + GB*0.25 + 8*GPU[a100/a40/l40s], 4 for v100/p100) x minutes`, times array tasks.
- It counts sacct used SUs (:180, `sacct -u $USER -A $ACCOUNT -S $START -X -n -P --format=AllocTRES,ElapsedRaw,JobName`), plus the remaining reservation of queued and running jobs (:186-194), plus the new job projected at its full time limit.
- It fails closed if sacct or squeue errors.
- It refuses: over max_project_sus (:329), over max_concurrent_jobs where every array task counts (:328), more than max_gpu_nodes_per_job (:327), over max_job_hours (:326), a name without the prefix (:321-325), an --output directory it cannot create (:332-339, absolute paths only; relative paths are skipped), and --wrap jobs (:223).
- It adds --account itself (:342) and appends to `$GUARD_LOG`, default `jobs/submissions.log` (:44, :345-347).
- Defaults: partition main (:238), time 1:00:00 (:239).
- Pitfall: a GPU job without -c is costed at 64 CPUs, and without --mem at 2 GB/CPU (:276-279, :298). One GPU with no -c/--mem is costed 104 SU/min, so always set -c and --mem.
- config/budget.yaml: `max_project_sus: 250000` (:6), `max_concurrent_jobs: 8` (:7), `max_gpu_nodes_per_job: 2` (:8), `max_job_hours: 47.5` (:9), `warn_at_fraction: 0.7` (:10), `job_name_prefix: "__PROJECT__-"` (:13).

Actual-SU logging:
- scripts/session_report.py runs on the cluster: `scripts/remote.sh python scripts/session_report.py [--since D | --all]` (:4-6). It uses `sacct ... --format=JobID,JobName,Partition,State,ElapsedRaw,AllocTRES,NodeList` (:47-48) and prints a per-job table plus "TOTAL x SU since D for jobs named <prefix>* (cap, %)" (:59-60). It imports yaml (:21).
- watch_job.sh:40 prints `sacct -j $JOB -X -n -P --format=JobID,JobName,State,ElapsedRaw,AllocTRES,NodeList,ExitCode` when a job finishes.

Polling: scripts/watch_job.sh `<jobid> [poll_seconds=60] [max_polls=30] [tail_lines=40]` (:9, :13).
- It is a sleep loop, documented "Run it in the background" (:10). `watch_job.sh <id> 0 1` gives a single check.
- Exit codes: 0 left the queue, 1 still queued, 2 polls failed 3 times (:11). Only empty output or "Invalid job id" means the job left (:26-33).
- :21 Duo hint `ssh -fN discovery`. :42 tails `$KIT_CLUSTER_SCRATCH/logs/<jobname>-<jobid>.out`.

Self-chaining: jobs/chained_template.sbatch with src/kit/chain.py.
- :42-43 checks DONE/STOP before loading the model.
- :48-49 serves vLLM, then runs `$PY -m $MODULE --run-dir --server --model`.
- :51-58 the job resubmits itself at its end: `slurm scripts/sbatch_guard.sh --dependency=afterany:"$SLURM_JOB_ID" ${TL:+--time="$TL"} "$SELF" ...`. A crash never resubmits.
- chain.run_units (:176-229): requires max_units, writes DONE at max_units, writes STOP on no progress, writes COMPLETE.json last, moves an atomic `latest` symlink, reads the deadline from SLURM_JOB_END_TIME or scontrol, uses RESERVE_S=600 and needs max(1.25 x last unit, 120 s). There is a demo controller, `MODULE=kit.chain`.

Scaling: kit.bench (multiprocessing Pool over module:function(i); it times items and discards outputs), bench_scaling.sbatch (WORKERS "64 32 16 8 4 2 1"), bench_scaling_2node.sbatch (node count from SLURM_JOB_NUM_NODES, so --nodes=4 works), bench_vllm.sbatch, and scripts/scaling_study.py (CSV and PNG into docs/scaling/).

== 3. PLAYBOOK §11 verbatim (PLAYBOOK.md:198-223) ==

## 11. Verified cluster facts (2026-10-25/26; re-verify with `cluster_setup.sh facts`)
* Partitions: debug (1 h; per user 48 CPUs, 4 GPUs, 5 jobs), main (48 h), gpu (48 h; per user 36 GPUs, 400 CPUs, 36 running and 100 queued jobs), oneweek (168 h, no GPUs), largemem (168 h, 1.5 TB), htcondor.
* SU rules: https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs and https://www.carc.usc.edu/user-guides/advanced-hpc-programming/gpu-programming (other guessed CARC URLs 404). sacct is the ledger; `myaccount` drifts, `sshare` decays, `sreport` lags (LESSONS #42).
* GPU nodes: A100 ×2 (64 CPUs, 256 GB; features `a100-40gb` or `a100-80gb`), A40 ×2 (32–64 CPUs, 256 GB), L40S ×3–4 (64 CPUs, 768 GB), V100 ×2, P100 ×2. Billing weights: CPU 1, Mem 0.25/GB, A100/A40/L40S 8, V100/P100 4.
* Storage: `/home1/$USER` 100 GB; `/project2/<account>` 1 TB shared by the class; `/scratch1/$USER` ~10 TB (purged). Node-local `$TMPDIR` in jobs is a 126 GB tmpfs.
* Modules: `conda/25.11.0` (miniforge), cuda up to 13.0, ffmpeg up to 7.1; system git 2.43.
* Compute nodes: Rocky Linux 8.10, glibc 2.28, NVIDIA driver 580 (CUDA 13), outbound internet, no proxy.
* Allocation `anakano_429`: 30,000,000 SU shared by the class; fair share ~0.4 for one student, which is not what makes GPU jobs wait: other jobs' age is (PriorityMaxAge 7 days, weight 7,000; LESSONS #3).
* Known-good versions (first project, 2026-10-25/26): vllm 0.32.0, torch 2.13.0+cu130, huggingface_hub 2.3.0, python 3.12 (serve env); models, all ungated: Qwen/Qwen3-VL-8B-Instruct `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (17 GB), Qwen/Qwen3-VL-32B-Instruct `0cfaf48183f594c314753d30a4c4974bc75f3ccb` (66.7 GB, Apache-2.0), yuvalkirstain/PickScore_v1 `a4e4367c6dfa7288a00c550414478f865b875800` (3.9 GB; the card declares no licence), laion/CLIP-ViT-H-14-laion2B-s32B-b79K `1c2b8495b28150b8a4922ee1c8edee224c284c0c` (MIT).
* Added on 2026-10-26 after the cluster was wiped, so not yet run on Discovery: the `sprio`, `squeue --start`, `sacctmgr` QOS and `sinfo -p main` lines of `cluster_setup.sh facts`. Run them in session 1 and record what they print.

Other version and module mentions:
- build_serve_env.sbatch:10-13: "Known-good on 2026-10-25: vllm 0.32.0, torch 2.13.0+cu130, driver 580 (Discovery A40/A100)". The wheels bundle the CUDA runtime, so no cuda module is needed. That build plus one 17 GB model fetch took 1,317 s, 351 SU (job 13455457).
- kit.env.example:15 `KIT_CONDA_MODULE=conda/25.11.0`. job_env.sh:18 `module load usc`. cluster_setup.sh:31 runs module spider only for conda, cuda, ffmpeg and git.
- CPU env: environment.yml:10 `python=3.11`, unpinned pip deps pytest, pyyaml, numpy, matplotlib, huggingface_hub and -e . (:12-18). pyproject.toml:11-12 `>=3.11`, deps pyyaml and numpy.
- The kit has zero mentions of trl, transformers, peft, accelerate, deepspeed, GRPO or LoRA (grep-confirmed), so no versions for them.
- No text-only instruct model has been verified on Discovery; the known-good models are vision-language models.
- Nothing from the first project survives on the cluster (README.md:19-20, PLAYBOOK.md:70-71, :195-196).

== 4. LESSONS.md entries relevant to lofi-rl (number: rule) ==

Accounting and queue:
- 1 (:12-15): SUs are billed per minute of reserved resources; the guard computes max(billing, cpu+GB/4+8·GPU) x minutes. A full A100 node is 144 SU/min.
- 2 (:16-18): fail closed if accounting can't be read; count the remaining reservations of queued and running jobs.
- 3 (:19-29): GPU waits come from other jobs' age priority. Keep GPU jobs short and accurately timed (backfill); allow `--gres=gpu:1 --constraint="a40|a100|l40s"`; use debug (1 h, one 2xA40 node) when it is up; submit early; read `squeue --start`.
- 4 (:30-33): nodes get drained (a conda install hung). Build envs and do downloads in dedicated jobs with `timeout` on every step.
- 5 (:34-36): sbatch snapshots the script at submission. After editing a job script, cancel and resubmit pending jobs.
- 40 (:37-43): an array costs every task; each task counts toward max_concurrent_jobs; %N throttles neither; --exclusive costs a whole node. A big sweep should be one job with a worker pool (kit.bench). This matters for the render farm.
- 41 (:44-48): name every job `<project>-...`; the ledger and guard count only the prefix.
- 42 (:49-55): only sacct is a ledger (not myaccount, sshare or sreport). Per-user limits: debug 1 h, 48 CPU, 4 GPU, 5 jobs; gpu 48 h, 36 GPU, 400 CPU, 36 running and 100 queued.

Cluster shell and environment:
- 6 (:59-60): project storage is /project2/<account>, not /project; check with myquota. /project2 is a 1 TB quota shared by the class; scratch is ~10 TB and purged. No purge interval is given anywhere.
- 7 (:61-62): run remote commands under `bash -lc`; filter "module: command not found".
- 8 (:63-66): `module load conda` breaks system binaries through LD_LIBRARY_PATH. Probe in Python; run system and SLURM tools via `env -u LD_LIBRARY_PATH`. This applies to conda-forge FluidSynth.
- 9 (:67-68): call the env's interpreter explicitly ($PY) and print it in every log.
- 10 (:69-70): no heavy Python imports on login nodes; inspect packages by reading source.
- 11 (:71-72): compute nodes have outbound network (HF, GitHub, PyPI, conda; no proxy). Downloads still belong in a job.
- 43 (:73-78): the GPU env must hold the project with its dependencies, import-check every module and check binaries in a CPU build job. Rebuild after adding a dependency.
- 44 (:79-82): one env per project; editable installs shadow each other. Only the weight cache is shared.
- 45 (:83-85): inspect a live job with `srun --jobid <id> --overlap --pty bash`.

SLURM hygiene:
- 12 (:89-92): jobs never write into the code clone; write to the artifacts dir or scratch.
- 13 (:93-96): the --output directory must exist before submission; #SBATCH lines cannot expand variables.
- 14 (:97-100): remote commands run inside a `{ }` group whose closing brace is on its own line (handles heredocs, comments, `&`).
- 15 (:101-103): laptop env vars don't reach jobs; put them inside the quoted remote command.
- 16 (:104-105): set `PYTHONUNBUFFERED=1` and `OPENBLAS_NUM_THREADS=4`.
- 17 (:106-108): vLLM ports in blocks of 8 per job id, plus a random per-job --api-key.
- 46 (:109-112): a failed squeue is not completion; only empty output with exit 0, or "Invalid job id specified", means the job left.
- 47 (:113-119): rsync file lists silently merged same-named files and skipped missing ones. Keep structure, pull all-or-nothing, pass rsync arguments as an array.

vLLM (0.32):
- 18 (:123): `--disable-log-requests` no longer exists.
- 19 (:124-126): offline load by repo id needs refs/main; serve the snapshot path with --served-model-name.
- 20 (:127-128): `VLLM_USE_FLASHINFER_SAMPLER=0` (no nvcc on compute nodes).
- 21 (:129-130): flashinfer-cubin must exactly match flashinfer-python, otherwise go without it.
- 22 (:131-133): an 8B VLM did about 1,025 calls/min per A40, x1.99 on 2 GPUs.
- 23 (:134-135): the round-robin client locks per call.

Hugging Face:
- 24 (:139): ungated repos only, downloaded anonymously; get_token() must be None on the cluster.
- 25 (:140-141): the xet downloader needs 32 GB of memory and max_workers=4 for big models.
- 26 (:142-143): du understates the HF cache (blobs are content-addressed).
- 27 (:144-145): check every licence and flag gaps for the human. This matters for SoundFonts, samples and audio models.

Laptop:
- 28 (:149-151): `set -o pipefail` in every `pytest | ... && git commit` chain.
- 29 (:152): macOS has no `timeout`.
- 48 (:153-156): `export MAMBA_ROOT_PREFIX=~/micromamba` must be in ~/.zshrc.
- 30 (:157-160): on the USC VPN the resolver can fail; use the IP fallback with HostKeyAlias.
- 31 (:161-165): no AI attribution trailers in commits; never force-push.

Chained runs:
- 49 (:169-175): read the job's end time from SLURM, keep a reserve, and start a unit only if max(1.25 x last unit, 120 s) fits.
- 50 (:176-181): resume only from complete checkpoints: marker written last, atomic `latest`, partial units moved aside.
- 51 (:182-187): every chain is bounded by max_units; check DONE/STOP before loading the model; a crash never resubmits.

Gates:
- 52 (:191-195): an approval is a line of its own, `^APPROVED by <Name> <YYYY-MM-DD>$`. This could serve for ear-check approvals.

LLM content:
- 32 (:203-207): small models ignore format constraints. Use a deterministic repair stage before validation and log every repair; max_tokens about equal to output characters; record finish_reason; set sampling parameters explicitly (default sampling gave near-identical samples). Directly relevant to LoopScript.
- 33 (:208-210): measure quality automatically on a small sample before asking a human to rate. Relevant to triage before votes.

Review:
- 34-36 (:214-222): adversarial multi-agent review after bursts of code; treat a finding whose verifiers died as unverified; one regression test per finding.

Measurement:
- 37 (:226-229): Discovery "CPUs" are hardware threads. An EPYC 7542 is 32 cores / 64 threads; efficiency was 96/94/89/74/49 % at 4/8/16/32/64 workers. Report efficiency against physical cores.
- 38 (:230-232): byte-identical outputs need explicit flags (ffmpeg bitexact), and even then files differed between arm64 laptops and x86 nodes; make the cluster's output canonical. This bears directly on the bit-identical WAV rule.
- 39 (:233-234): every number cites its log.
- 54 (:235-244): measure a multi-node point in one allocation against a same-allocation reference, and record each node's CPU model (a pinned CPU constraint was not honoured).

Audio: no lessons mention FluidSynth, SoundFonts or audio; 38 is the only one involving ffmpeg.

== 5. Kit conventions vs CLAUDE.md (kit first, then CLAUDE.md) ==

Code transport
- Kit: GitHub hub with an anonymous `git clone` and `git pull --ff-only` on the login node before every command (remote.sh:30, cluster_setup.sh:40-41). It requires a public repo (cluster_setup.sh:12).
- CLAUDE.md: a `make sync` rsync mirror with --delete, no git history, and git never on login or compute nodes (CLAUDE.md:15-21, 52, 122). The repo may stay private.

Git on compute nodes
- Kit: job_env.sh:27, build_cpu_env.sbatch:22 and build_serve_env.sbatch:32 run `git rev-parse` on the compute node.
- CLAUDE.md:16 forbids git there. In a mirror without .git this prints a fatal error but isn't fatal (it's inside echo), and you lose the commit stamp.

Who pushes
- Kit: the agent commits and pushes (PLAYBOOK.md:88; CLAUDE.template.md:107-111), and remote.sh refuses if the laptop is ahead of origin.
- CLAUDE.md:87-89: "Shresth pushes unless he says Claude may."

Attribution
- Kit: no AI trailers (README.md:50; LESSONS 31; CLAUDE.template.md:110-111; bootstrap.sh:8).
- CLAUDE.md §7 says nothing. Needs an explicit decision.

Paths
- Kit (kit.env.example:18-29):
  - project root /project2/anakano_429/shresthr
  - clone /project2/.../lofi-rl
  - artifacts /project2/anakano_429/shresthr/artifacts/lofi-rl
  - envs /project2/.../envs/lofi-rl and lofi-rl-serve
  - scratch /scratch1/shresthr/lofi-rl
  - HF_HOME /scratch1/shresthr/hf_home, shared across projects
  - caches /scratch1/shresthr/{conda_pkgs,pip_cache}
- CLAUDE.md:25-27, 65:
  - REMOTE_PROJ=/project/anakano_429/shresthr/lofi-rl ("/project", which LESSONS 6 says is wrong; CLAUDE.md itself says verify with myquota)
  - REMOTE_SCR=/scratch1/shresthr/lofi-rl
  - HF_HOME=$REMOTE_SCR/hf
  - .venv at $REMOTE_PROJ/.venv
  - FluidSynth build at $REMOTE_PROJ/opt (CLAUDE.md:73)
- Variable names: KIT_* vs REMOTE_PROJ and REMOTE_SCR.

Env tool
- Kit: conda module on the cluster plus micromamba on the laptop; one environment.yml (py3.11) for the CPU env plus a separate conda py3.12 serve env with pip-installed vLLM (build_serve_env.sbatch:34-37); unpinned files, with pins recorded via pip freeze logs.
- CLAUDE.md:71: `uv venv .venv && uv pip install -r requirements.txt`, one pinned venv. uv never appears in the kit.

Env build partition
- Kit: main (build_cpu_env.sbatch:3, build_serve_env.sbatch:3).
- CLAUDE.md:71: a debug job, `jobs/build_env.sbatch`. Debug's 1 h cap, and LESSONS 4 (drained nodes), apply.

Smoke job
- Kit: smoke_cpu.sbatch runs pytest only, on CPU.
- CLAUDE.md:75: smoke_env.sbatch on debug with a GPU checks imports, a visible GPU, a FluidSynth MIDI render, one vLLM completion and a LoopScript WAV at -14 LUFS.

Job naming
- Kit: `--job-name=lofi-rl-<x>`; the guard refuses names without budget.yaml's prefix (sbatch_guard.sh:321-325); test_job_scripts.py:78-82 checks it.
- CLAUDE.md:64: `lofi-<purpose>`. Names like `lofi-smoke` would be refused unless the prefix is set to "lofi-".

Logs
- Kit: `--output=/scratch1/%u/lofi-rl/logs/%x-%j.out`, absolute, hyphen, test-enforced; watch_job.sh:42 reads it.
- CLAUDE.md:64, 67: `logs/%x_%j.out`, relative to the submit dir ($REMOTE_PROJ/logs), underscore. The guard does not create relative log dirs (sbatch_guard.sh:335).

rsync --delete hazards that the kit's lessons expose in CLAUDE.md's own design
- CLAUDE.md:55's .rsyncignore minimum doesn't list logs/, opt/, jobs/submissions.log (the guard's log, sbatch_guard.sh:44) or any checkpoint dir. `make sync --delete` would delete those on the cluster.
- LoRA checkpoint files like *.json, *.bin and *.pth are not covered by `*.pt *.safetensors`. LESSONS 12 is the root rule.
- GUARD_LOG can be pointed at scratch.

make pull
- Kit: pull_artifacts.sh all, path or gate, keeping structure and pulling all-or-nothing.
- CLAUDE.md:53: `rsync -az discovery:$REMOTE_SCR/runs/*/summary/ runs/` flattens every run's summary/ into runs/, so same-named files collide. This is the LESSONS 47 class of bug.

Makefile
- Kit: none.
- CLAUDE.md: make sync, make pull, make junk-rate (CLAUDE.md:52-53, 81, 102).

ssh config
- Kit (PLAYBOOK.md:35-45, and the laptop's actual ~/.ssh/config): `Host discovery discovery.usc.edu`, IdentityFile id_ed25519, IdentitiesOnly yes, ServerAliveCountMax 5, `ControlPath ~/.ssh/cm-%r@%h:%p`, `ControlPersist 4h`.
- CLAUDE.md:34-40: `Host discovery`, `ControlPath ~/.ssh/cm-%r@%h-%p`, `ControlPersist 8h`.
- Opening the master: kit `ssh -fN discovery` (PLAYBOOK.md:47) vs CLAUDE.md:44 `ssh discovery true`.
- The kit's BatchMode=yes (fail fast, no Duo prompt) is absent from CLAUDE.md's raw `ssh discovery 'bash -lc "<cmd>"'`, whose nested quotes also break on commands containing double quotes. The kit's printf %q quoting (remote.sh:35) avoids that.

Login-node commands
- Kit: cluster_setup.sh facts uses sinfo, scontrol, sacctmgr, sshare, sprio and module spider; netcheck uses a blocking srun; session_report.py and the guard use python3.
- CLAUDE.md:15: allowed only sbatch, squeue, sacct, myaccount, myquota, rsync, ls, tail, cat. CLAUDE.md:47-48 says no long work inside ssh.

Partitions and GPUs
- Kit: `--gres=gpu:1 --constraint=a40|a100|l40s` (gpu_vllm_template.sbatch:8-9). Debug is described as one 2xA40 node, often drained. Main nodes are "64-thread EPYC" (PLAYBOOK.md:104).
- CLAUDE.md:63: A100-80GB `--gres=gpu:a100:1 --constraint=a100-80gb`, L40S `--gres=gpu:l40s:1`, main described as "64-core EPYC". The kit says 64 threads (32 cores), which affects the 16 to 256 "cores" scaling claims.
- The guard parses both GPU forms correctly (sbatch_guard.sh:262-273).

Budget
- Kit: max_project_sus 250000 (budget.yaml:6), no per-job approval threshold, max 8 concurrent jobs.
- CLAUDE.md:66, 127: 1,000,000 SU target, hard stop 1,200,000, and "Anything estimated above 500 SU: ask first".
- Kit jobs whose worst case is over 500 SU: build_serve_env 1,440, fetch_weights 720, gpu_vllm_template 560, a chain link 6,720, bench_scaling 1,600, bench_scaling_2node 1,600, bench_vllm 560/1,120.
- Consistent: the SU formula (CLAUDE.md:64 matches the kit) and the 48 h cap vs max_job_hours 47.5.

SU logging
- Kit: session_report.py totals in the CHANGELOG Status block.
- CLAUDE.md:66: per job, `sacct -j <id> --format=JobID,Elapsed,AllocTRES` into the CHANGELOG.

Chaining
- Kit: the job resubmits itself at its end with `--dependency=afterany` through the guard (chained_template.sbatch:57). The template serves an OpenAI-style vLLM server.
- CLAUDE.md:63: `sbatch --dependency=afterok:$JOBID`. A pre-queued successor is counted by the guard as a full reservation and as a concurrent job (sbatch_guard.sh:11-13). TRL's use_vllm doesn't fit the template's serve-then-MODULE shape without adaptation.

HF offline
- Kit: jobs default to HF_HUB_OFFLINE=1, with weights pre-fetched by fetch_weights.sbatch (anonymous, ungated).
- CLAUDE.md:74: download from a compute job if the network allows, else from the laptop via hpc-transfer1.usc.edu. The kit never mentions the transfer node.

Pull destinations and git hygiene
- Kit: artifacts/ (small curated files committed, plus MANIFEST.json) and artifacts/review/ (gitignored).
- CLAUDE.md:53, 82: runs/ and inbox/<date>/ with a manifest; never commit runs/ or WAVs. The kit's .gitignore lacks runs/, inbox/ and *.wav.

Memory files
- Kit: CHANGELOG with Status (SUs used/cap, cluster clone commit, NEEDS HUMAN), then Done, Metrics, Failed approaches, Decisions, Pins, Open questions, plus Known limitations and a scoring change-log table (CHANGELOG.template.md:3-30; CLAUDE.template.md:141-145). No DECISIONS.md.
- CLAUDE.md:79-80: entries with date, work, numbers, failed approaches, SU spent and next step, plus docs/DECISIONS.md.

Rituals
- Kit: start by reading CLAUDE.md, CHANGELOG.md, PLAYBOOK and LESSONS in session 1; report SUs every session; `pytest tests/ -x -q` under pipefail; commit and push every unit (CLAUDE.template.md:3-6, 106-119).
- CLAUDE.md:117-118: start with CLAUDE.md, CHANGELOG, `make pull`, `ssh discovery 'squeue -u shresthr'`, then state the next unit. End with CHANGELOG, `pytest -q`, commit, `make sync` if code changed, and list items waiting on an ear check or a Duo re-open.

Human approval
- Kit: gates/GATE_<n>.md with an `APPROVED by <Name> <YYYY-MM-DD>` line (src/kit/gates.py:24).
- CLAUDE.md:82, 103: ear checks are Shresth's, through clips queued in inbox/; no approval-recording format is defined.

Same on both sides: account anakano_429, cluster user shresthr, no secrets on the cluster, every number cites its source.

== 6. What the kit does not cover (all grep-confirmed absent unless noted) ==

- FluidSynth: nothing (no module check, no conda package, no source build). cluster_setup.sh:31 doesn't spider fluidsynth. The reusable piece is the KIT_SERVE_BINARIES binary check (build_serve_env.sbatch:50-65).
- SoundFonts and one-shot samples: no asset handling, no licence tracking for audio (only LESSONS 27 for models), no big-file transfer path. CLAUDE.md:55 also excludes assets/big/ from sync, so how assets reach the cluster is undefined in both documents.
- pedalboard, pyloudnorm, soundfile/libsndfile, pretty_midi, mido, scipy: none.
- laion-clap, MERT, audiobox-aesthetics: none. fetch_weights skips *.pt and *.ckpt (:36-39), and offline loading needs local paths (kit.hf_cache.resolve).
- TRL, GRPO, PEFT, transformers: none. There is no training job, no colocate or server-mode handling, and no LoRA checkpoint policy. The vLLM scripts are OpenAI-server and vision-oriented (serve_vllm.sh:18 `--limit-mm-per-prompt '{"image":2}'`, gpu-util 0.90).
- CPU render-farm job arrays: the guard supports and costs arrays, but the kit advises against them (PLAYBOOK.md:109). There is no production render driver; kit.bench only times items. Note OPENBLAS_NUM_THREADS defaults to 4 per worker (job_env.sh:15), and OMP threads are not set.
- Local rating UI and pairwise vote storage or Bradley-Terry: none.
- Frontier-API seed generation: none (publish.py only scans for key patterns).
- Also missing: a Makefile, uv, requirements.txt, a per-job 500 SU approval threshold, the transfer node, a determinism (bit-identical WAV) harness, a DECISIONS.md, inbox clip sampling with a manifest, and any make junk-rate equivalent.

== 7. How the kit's own tests run ==

- Command (README.md:42-46): `export MAMBA_ROOT_PREFIX=~/micromamba`, then `micromamba create -y -n kit-dev -f environment.yml`, then `micromamba run -n kit-dev pytest -q`.
- That env is conda-forge python 3.11 with pip pytest, pyyaml, numpy, matplotlib, huggingface_hub and -e . (environment.yml:10-18). Network to conda-forge and PyPI is needed once.
- pytest config: pyproject.toml:20-23 sets testpaths tests, -ra, pythonpath src.
- Needs on PATH: bash, git (temporary repos), rsync and perl (the real bootstrap), and python3. The guard defaults to python3 (sbatch_guard.sh:48), and the remote.sh heredoc test and _cluster_ssh.sh:10 also use it.
- Needs in the env: PyYAML (session_report), matplotlib (test_bench_scaling.py:39), and optionally huggingface_hub (importorskip at test_gates_publish.py:130, 268).
- Free localhost ports for the fake vLLM servers; test_bench_vllm.py:196 skips if none are free.
- No cluster and no external network: ssh, sbatch, sacct, squeue, srun, scontrol, module, conda and lscpu are all PATH stubs. Everything is written under tmp_path.
- Count: README.md:38 claims 210 tests. I counted 123 test functions plus 14 parametrize decorators statically; I did not collect them.
- Laptop state:
  - micromamba 2.10.0, uv 0.12.8 and gh are installed; make is GNU 3.81; /usr/bin/python3 is 3.9.6.
  - /usr/bin/rsync is openrsync (protocol 29).
  - ~/micromamba/envs exists but is empty (no kit-dev env).
  - MAMBA_ROOT_PREFIX is unset and not in ~/.zshrc, ~/.zshenv or ~/.zprofile.
  - tests/__pycache__ holds cpython-311-pytest-9.1.1 .pyc files, so the suite was run at least once; the result is unknown.
  - I did not run the suite (read-only task).
- In a bootstrapped project: test_kit_bootstrap.py is dropped (bootstrap.sh:31), and smoke_cpu runs the suite on a compute node (smoke_cpu.sbatch:18). That includes git calls (test_laptop_scripts.py:78-98), which CLAUDE.md:16 forbids there.
- test_job_scripts.py checks every jobs/*.sbatch against kit conventions. Jobs written the CLAUDE.md way (relative logs/%x_%j.out, lofi- names, jobs/env.sh, a uv venv, GPU jobs without `MODE=serve source scripts/job_env.sh`, a bare `sbatch --dependency`) would fail :19, :23, :28-29, :32-41, :67-75 and :78-82 unless the tests are adapted.

== 8. Not verified ==

- That /project2 rather than /project is right for shresthr (needs myquota).
- The scratch purge interval.
- The current partitions and debug GPUs (the kit's facts are from 2026-10-25/26; the `sprio`, `squeue --start`, `sacctmgr` QOS and `sinfo -p main` lines were never run on Discovery).
- Whether uv or fluidsynth modules exist on Discovery.
- Whether a `python` with PyYAML exists on the login node (session_report.py:21).
- Whether openrsync supports -z, --partial, -L and --exclude-from against Discovery's rsync, and whether excluded files are protected under --delete.
- Whether chained_template.sbatch's self-resubmission has ever run on Discovery.
- Whether the Audiobox, CLAP and MERT repos are ungated, what licences they carry, and what file types they ship.
- laion-clap's dependency compatibility with the vLLM/torch pins.
- How vLLM handles --limit-mm-per-prompt for a text-only model.
- Whether $TMPDIR tmpfs usage counts against job memory.
- That slipstr34m is the intended GitHub account for lofi-rl (it is the kit's default and the kit's own origin).
- FluidSynth and pedalboard bit-determinism between arm64 macOS and x86 Rocky 8.