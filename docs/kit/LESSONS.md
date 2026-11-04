# Lessons learned (symptom → cause → rule)

Every entry cost real time or SUs on USC CARC Discovery between 2026-10-25 and 2026-10-26 (the first
project built this way, "One Line"; its CHANGELOG and history are in the private repo
`slipstr34m/one-line`), or was found by the review of this kit on 2026-10-26 and reproduced with
stubs. The kit already encodes each rule; this file says why, so nobody "simplifies" a fix away.
Numbers are stable references (code comments cite them), so later lessons are appended within their
section rather than renumbered.

## Accounting, budget and the queue

1. **SUs are billed per MINUTE, not per hour.** A first guard draft multiplied SLURM's `billing` TRES
   by hours and under-counted 60×. CARC: CPU 1, 4 GB RAM 1, A100/A40/L40S GPU 8, V100/P100 GPU 4 SU per
   minute of *reserved* resources. A full A100 node is 144 SU/min = 8,640 SU/h; one 47.5-hour job on it
   costs 410,400 SU. Rule: `sbatch_guard.sh` computes `max(billing, cpu + GB/4 + 8·GPU) × minutes`.
2. **A guard that cannot read accounting must refuse.** The first guard treated a failing `sacct` as
   "0 SU used" (fail-open). Rule: fail closed; also count the remaining reservation of every queued and
   running job, not only finished ones.
3. **GPU queue waits come from other jobs' age, not from our fair share.** 20-minute 1-GPU jobs were
   estimated about 10 hours out. `sprio` showed why (2026-10-26 12:05): 222 of 775 pending `gpu` jobs
   outranked ours on **age** (PriorityMaxAge 7 days at weight 7,000, about 42 points per hour waited;
   the leaders had waited about 5 days), while our fair-share term (user factor 0.40, 1,612 points) was
   above most pending jobs'. Nearly every GPU node was `mixed`, held for those jobs, so ours started
   only in backfill gaps. An earlier entry blamed fair share; the first project's commit `1e7f3ca`
   corrected it. Rule: keep GPU jobs short and accurately timed (backfill fits them), allow any bf16
   GPU type (`--gres=gpu:1 --constraint="a40|a100|l40s"`), use `debug` (1 h, one 2×A40 node) when it
   is up, submit batches early (a pending job gains priority as it waits, so a cancel-and-resubmit
   costs that), and read SLURM's estimate (`squeue --start`, shown by `watch_job.sh` and
   `cluster_setup.sh facts`) instead of guessing.
4. **Nodes get drained.** The debug A40 node was drained by someone else's job; a CPU debug node was
   drained by ours (a job stuck in `conda install` that SLURM could not kill). Rule: environment
   building and downloads happen in dedicated jobs with `timeout` on every step, never mixed into
   compute jobs.
5. **`sbatch` snapshots the job script at submission.** Jobs queued before a script fix still ran the
   old script. Rule: after changing a job script, cancel and resubmit anything still pending (the new
   job starts its age priority from zero, #3, so batch script fixes before submitting).
40. **Job arrays and `--exclusive` slipped past the guard.** The first guard ignored `--array`: ten
    47-hour 1-GPU tasks of `gpu_vllm_template.sbatch` were costed as one (78,960 SU instead of about
    790,000, over 3× the project cap), a queued range `123_[2-99]` counted as one job, and `--exclusive -c 8` was costed at
    8 CPUs although SLURM allocates (and bills) the whole node. Rule: an array costs every task (a `%N`
    throttle caps tasks running at once, not tasks run), each task counts toward
    `max_concurrent_jobs`, pending ranges are expanded, and `--exclusive` costs the largest node it
    could land on. A sweep larger than the job limit is one job with a worker pool (`kit.bench`).
41. **"This project's SUs" needs a scope.** `sacct -u $USER -A <account>` counts every project of the
    user on the class account: a project bootstrapped the day the previous one ran would have billed
    the old jobs to itself. Rule: every job is named `<project>-...`; the guard and
    `session_report.py` count only `job_name_prefix` jobs (`budget.yaml`), and the guard refuses a job
    whose name lacks the prefix, since the ledger would never see it.
42. **Only `sacct` is a ledger.** `myaccount`'s Usage drifted by tens of SU between calls (283,113 →
    283,062 → 282,882 in 20 minutes with about 39 SU of our jobs); `sshare` RawUsage is half-life
    decayed (2,721 billing-seconds, about 45 SU, against sacct's 39.2); `sreport` returned nothing for
    a same-day window. CARC's SU rules are on two live pages (linked in `sbatch_guard.sh`); four other
    plausible CARC URLs 404. Per-user limits from the same guide: debug 1 h, 48 CPUs, 4 GPUs, 5 jobs;
    gpu 48 h, 36 GPUs, 400 CPUs, 36 running and 100 queued jobs. Rule: check the guard against
    `session_report.py`, never against `myaccount` or `sshare`.

## The cluster shell and environment

6. **Project storage is `/project2/<account>`, not `/project`.** Check with `myquota`; `/project2` is a
   shared 1 TB quota for the whole class. Scratch (`/scratch1/$USER`, ~10 TB) is large but purged.
7. **`module: command not found` in non-interactive ssh shells.** The cluster `~/.bashrc` calls
   `module` before it exists. Rule: run remote commands under `bash -lc`; filter that line.
8. **`module load conda` breaks system binaries.** It prepends conda's `lib/` to `LD_LIBRARY_PATH`;
   `/usr/bin/curl` then loads conda's libcurl and exits 4 on a plain http URL — a health loop waited 30
   minutes on a server that was up. Rule: probes in Python; system and SLURM tools via
   `env -u LD_LIBRARY_PATH` (`slurm` helper in `job_env.sh`).
9. **A job once ran the miniforge *base* interpreter** although `conda activate` had run. Rule: call
   `$CONDA_PREFIX/bin/python` explicitly (`$PY`) and print it at the top of every log.
10. **Login nodes cannot import heavy Python stacks** (vLLM, numpy with many BLAS threads segfault on
    thread limits). Rule: inspect installed packages by reading their source files, not by importing.
11. **Compute nodes have outbound network** (huggingface.co, github.com, pypi.org and conda all
    answered from a debug node; no proxy). Downloads still belong in a job, not on a login node.
43. **A GPU job died on a missing package after its queue wait.** Job 13456023: vLLM healthy after
    90 s, then `ModuleNotFoundError: svgpathtools`, because the serving env had the project installed
    with `--no-deps` and GPU jobs run only in that env. Rule: `build_serve_env.sbatch` installs the
    project WITH its declared dependencies (held to the versions vLLM installed), then imports every
    module of the project package and finds every `$KIT_SERVE_BINARIES` executable, so the CPU build
    job fails instead of the GPU job. Rebuild it after adding a dependency.
44. **One serve env per project.** Two projects editable-installed into one shared env both put their
    `src/` on `sys.path`; from project B's clone `import kit` silently resolved to project A's `kit`
    (the `__editable__` `.pth` file that sorts first wins). Rule: `$KIT_CLUSTER_SERVE_ENV` is
    `envs/<project>-serve`; only the weight cache is shared.
45. **Inspect a live job from inside its allocation.** `srun --jobid <id> --overlap --pty bash` (or a
    one-off command instead of `--pty bash`) runs on the job's node, in its cgroup, while it runs; that
    found the conda-libcurl cause of #8 in minutes.

## SLURM job hygiene

12. **Jobs must never write into the git clone.** A benchmark wrote its stats into the clone; after the
    same files were committed from the laptop, `git pull --ff-only` on the cluster refused. Rule: jobs
    write to `$KIT_CLUSTER_ARTIFACTS` (curated, pulled) or `$KIT_CLUSTER_SCRATCH` (raw), never under
    `$KIT_CLUSTER_PATH`.
13. **The `--output` log directory must exist before submission**, or SLURM cannot open the log and the
    job dies silently. `cluster_setup.sh init` creates it and `sbatch_guard.sh` creates it (or refuses)
    before every submission. `#SBATCH` lines cannot expand variables, so each job spells the directory
    out; a test checks it agrees with `kit.env`.
14. **`remote.sh` ran commands after a failed pull** when they were joined with `;` (the `&&` chain only
    guarded the first). Then `{ $* ; }` broke any command ending in a heredoc, a `# comment` or a `&`
    (`here-document delimited by end-of-file`). Rule: the command runs inside a `{ }` group whose
    closing brace is on its own line.
15. **Environment variables set on the laptop do not reach the job.** Put them inside the quoted remote
    command: `scripts/remote.sh 'MODEL=x TP=2 scripts/sbatch_guard.sh jobs/y.sbatch'` (sbatch exports the
    submitting environment by default).
16. **Unbuffered output and bounded BLAS threads** (`PYTHONUNBUFFERED=1`, `OPENBLAS_NUM_THREADS=4`):
    otherwise logs arrive at exit and each worker spawns a thread per CPU.
17. **Port collisions between our own jobs.** Two jobs on the same debug node both used port 8000; one
    job's health check found the other's server, which was then cancelled under it. Rule: ports in
    blocks of 8 per job id, plus a random per-job `--api-key` that the health probe must present.
46. **A failed `squeue` looked like a finished job.** `watch_job.sh` discarded squeue's stderr and read
    empty output as "left the queue", so a slurmctld timeout would end a watch while the job ran.
    Rule: only an empty answer with exit 0, or `Invalid job id specified`, means the job left; any
    other error is a failed poll (three in a row stop the watch), and the exit status says which.
47. **Gate pulls lost files without an error.** `rsync --files-from` with `--no-relative` and no `-r`
    merged same-named files from different directories into one, pulled listed directories empty,
    and skipped missing paths, all with exit 0; relative entries pointed into the clone, where jobs
    never write. A second bug: `eval` of an rsync string split a laptop path containing a space. Rule:
    `pull_artifacts.sh gate` keeps each path (relative to `$KIT_CLUSTER_ARTIFACTS`, or below a `/./`
    marker), copies directories with their contents, pulls nothing if any listed path is missing, and
    passes rsync its arguments as an array.

## vLLM serving (0.32 on A40/A100, driver 580)

18. `--disable-log-requests` no longer exists — the server refuses to start.
19. **Offline loading by repo id fails** when weights were fetched by commit sha (no `refs/main`).
    Rule: `fetch_weights.sbatch` writes `refs/main`, and the launcher serves the local snapshot path
    with `--served-model-name <repo id>`.
20. **The first sampling call crashed**: vLLM's default sampler JIT-compiles a FlashInfer kernel and
    compute nodes have no `nvcc`. Rule: `VLLM_USE_FLASHINFER_SAMPLER=0`.
21. **`flashinfer-cubin` must equal `flashinfer-python`'s version exactly** or vLLM refuses to start;
    PyPI had no matching 0.7.x cubin. Rule: install only an exact match, else run without it.
22. Throughput measured with an 8B vision-language model, 32 concurrent requests: about 1,025
    one-image calls/min and 186 two-image comparisons/min per A40, scaling ×1.99 / ×2.01 on two GPUs
    with one instance per GPU. Rendering was never the bottleneck; GPU calls were.
23. **Thread-safe clients.** A "round-robin" client that mutated its `base_url` from 64 threads was not
    round-robin. Rule: `kit.vllm_client.MultiClient` picks the server under a lock per call.

## Hugging Face weights

24. Ungated repos only, downloaded anonymously (`get_token()` must be `None` on the cluster).
25. **The xet downloader needs memory**: an 8 GB job was OOM-killed on a 66 GB model. 32 GB and
    `max_workers=4` worked; partial downloads resume.
26. **`du` lies about the cache**: huggingface_hub 2.x keeps large files in a shared content-addressed
    `hub/blobs/<xx>/` and links each repo to it, so a `models--*` directory shows megabytes.
27. **Check every licence before use.** One preference model's card declared no licence (its code
    repository was MIT) — record such gaps as a question for the human.

## The laptop and git

28. **zsh `pytest … | tail && git commit` commits failing code** — the pipeline's status is `tail`'s.
    Rule: `set -o pipefail` in every commit chain; check `git status` for files a blocked chain left
    untracked.
29. **macOS has no `timeout`**; use the agent tool's own timeout.
48. **`VAR=x cmd1 && cmd2` gives VAR only to cmd1.** The printed setup step
    `MAMBA_ROOT_PREFIX=~/micromamba micromamba create … && micromamba run …` failed in a fresh shell
    (`The given prefix does not exist: /opt/homebrew/Cellar/micromamba/…`), and every agent Bash call
    starts a fresh shell. Rule: `export MAMBA_ROOT_PREFIX=~/micromamba` in `~/.zshrc`.
30. **macOS's system resolver sometimes stops resolving the cluster on the USC VPN** while `host`
    still works; the agent's own API calls failed the same way during a network switch. Rule:
    `_cluster_ssh.sh` falls back to the address with `HostKeyAlias` (host key still verified); retry
    failed agent work after the network settles.
31. **No AI attribution in commits.** The human asked for `Co-Authored-By` trailers to be removed;
    60 commits were rewritten once (bundle backup → `git filter-branch --msg-filter` → verify trees,
    authors and dates unchanged → `git push --force-with-lease` → re-sync the cluster clone with
    `git fetch && git reset --hard` → commit an old→new hash map, because job logs cite old hashes).
    Rule: never add the trailer; never force-push otherwise.

## Chained runs

49. **A fixed stop margin lets SLURM kill a long unit mid-way.** The first project stopped 30 minutes
    before the limit whatever a generation took, so a 40-minute generation could start with 35
    minutes left and lose its GPU time (a high-severity review finding). Rule: read the job's end from
    SLURM (`SLURM_JOB_END_TIME`, else `scontrol show job` EndTime, run without conda's
    `LD_LIBRARY_PATH`), keep a reserve for teardown and the resubmission, and start a unit only if
    max(1.25 × the last unit's wall time, 120 s) still fits; a resumed link reads that time from the
    newest unit's marker (`kit.chain.should_stop`).
50. **Resume must trust only complete checkpoints.** A job killed while writing a unit leaves a
    directory that looks like a checkpoint, and a `latest` link rewritten with unlink + symlink is
    briefly missing. Rule: write a completeness marker LAST (tmp + fsync + `os.replace`), resume after
    the newest marked unit found by scanning the directories (never via the pointer), move a partial
    unit aside rather than reuse or delete it, and move `latest` with a temporary symlink +
    `os.replace` (`kit.chain.mark_complete`, `latest_complete`, `publish_latest`).
51. **A chain must be bounded, and must check DONE/STOP before it loads the model.** The first
    project's 5-generation smoke run never wrote DONE and would have chained 23.5-hour GPU jobs to the
    full run's length; a successor queued before a STOP spent minutes loading a model to do nothing.
    Rule: `run_units` requires `max_units` and writes DONE when it is reached; the template checks
    DONE/STOP before the serve env loads; a crash never resubmits; a link that cannot fit one unit
    writes STOP instead of looping (`jobs/chained_template.sbatch`).

## Gates and publishing

52. **A gate was approved by its own instructions.** The publish gate searched for the substring
    "APPROVED by", which the gate's "How to approve" line contains. Rule: an approval is a line of its
    own matching `^APPROVED by <Name> <YYYY-MM-DD>$` with a real date, outside code fences and
    comments; placeholders, quoted, indented or bulleted examples and "NOT APPROVED" never count
    (`kit.gates.gate_approved`, `python -m kit.gates`).
53. **huggingface_hub 2.3.0 has no `upload_large_folder`.** The first project's CLAUDE.md prescribed
    it, so the upload would have crashed at publish time. Rule: `HfApi.upload_folder` (unchanged files
    dedupe server-side), a dry run by default, an upload only with `--go` plus an approval line, then
    an anonymous read-back (`token=False`) that checks every file arrived (`python -m kit.publish`).
55. **A placeholder was copied, brackets and all.** A gate's How-to-approve line read `APPROVED by
    Shresth Rana <yyyy-mm-dd>`. His first save was `APPROVED by Shresth Rana <2026-11-04>`, and the
    refusal repeated the placeholder form, so it never said what was wrong. Rule: show the line written
    out with the approver's name and a real date, bare (`kit.gates.how_to_approve`, `example_line`;
    `python -m kit.gates --how-to-approve --name "Ada Lovelace"`), and have a refusal name each fault
    and show the line as it would count, e.g. "write the date bare: 2026-11-04, not <2026-11-04>"
    (`kit.gates.explain`). The accepted form stays as strict as #52 says.

## Building with LLM-generated content

32. **Small local models ignore format constraints**: coordinates outside the stated bounds, extra
    pen-up segments, replies cut off by `max_tokens`. Rule: a deterministic repair/normalisation stage
    before validation, every repair recorded; `max_tokens` ≈ output characters (digits tokenise about
    one per token); record `finish_reason`; explicit sampling parameters (default sampling made fresh
    samples near-identical).
33. **Measure quality before asking a human to rate anything.** An 8B model's 1,000 seed drawings were
    recognised by the vision model 0.26 % of the time; a quick contact sheet of 10 per category showed
    it in a minute. Compare candidate generators on a small sample with an automatic score first.

## Multi-agent review

34. **Adversarial review pays.** Two review workflows (finders per dimension, three verifier lenses per
    finding, majority vote) found 34 real defects in a day of fast code, 15 of them severe (a smoke
    run that would have chained 23-hour GPU jobs, a guard that failed open, a publish gate satisfied by
    its own instructions). A third review, of this kit after extraction, upheld 20 more (one high: job
    arrays past the guard, #40), mostly know-how lost in the extraction. Rule: after every burst of
    code, and after moving code between repositories, run one before using it on the cluster.
35. **Verifiers that die count as "rejected"** if the script only checks vote counts. Rule: classify a
    finding with fewer than the full set of votes as *unverified*, and re-run it.
36. **Keep fixes and their regression tests together**: one test per confirmed finding, named after it.

## Measurements for a class report

37. **CPUs are hardware threads on Discovery.** Strong scaling of a CPU-bound render on an EPYC 7542
    (32 cores / 64 threads): efficiency 96 % at 4 workers, 94 % at 8, 89 % at 16, 74 % at 32, 49 % at
    64; two nodes delivered 1.82× one node, but against a one-node run on a different CPU model
    (#54). Report efficiency against the physical core count.
38. **Byte-identical outputs need explicit flags** (ffmpeg `-fflags +bitexact -flags:v +bitexact
    -map_metadata -1 -threads 1`), and even then video files differ between arm64 laptops and x86
    nodes. Make the cluster's outputs canonical.
39. **Every number in a report cites its log.** Never write a percentage before the table that produces
    it exists (one README line had to be corrected after guessing).
54. **Measure a multi-node point as one run, against a reference from the same allocation.** The first
    project's two-node point (job 13455976, two EPYC 7513 nodes, 2,970.3 drawings/min) was divided by
    a one-node run on an EPYC 7542 (job 13455727, 1,634.1/min), so its "1.82×" compared two CPU
    models; a job submitted from a script pinned to `--constraint=epyc-7513` had logged that EPYC
    7542, so record each node's CPU model rather than trust the pin. The kit's first
    `scaling_study.py` also silently dropped the per-node files of a run that started `kit.bench` on
    each node, because they collided with the one-node point of the same worker count. Rule:
    `jobs/bench_scaling_2node.sbatch` measures the one-node reference and the two-node split in one
    allocation, times the whole `srun` step, records each node's CPU model and writes one file with
    `nodes`; `scaling_study.py` keeps every node count as its own point and names every file it ignores.
