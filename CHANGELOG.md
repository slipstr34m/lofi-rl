# CHANGELOG — lofi-rl lab notebook

Newest entry first. Every number cites the file or command output it was read from (CLAUDE.md §6).

## Status
- **SU used:** 0 of the 1,000,000 target (no jobs yet).
- **Phase:** 0 (setup), started 2026-10-30.
- **Waiting on Shresth:** nothing yet. Phase 1 shootouts start once the renderer exists (about Nov 3).

---

## 2026-10-30: planning interview, research, design, Phase 0 skeleton

**What was done**
- **Taste interview.** A 27-round multiple-choice interview with Shresth produced the taste profile in `docs/TASTE.md` and the pre-registered choices in `docs/DECISIONS.md` (D-001 to D-018). Highlights:
  - jazzhop, head-nod, a 2-chord vamp with 7ths and 9ths;
  - one featured sax playing a few long notes, phrase plus answer;
  - a clean mix with the rain barely there;
  - best+worst-of-4 voting, 1 h/day in micro-sessions on AirPods while working;
  - an RM gate relative to his own consistency;
  - forced-choice final A/B, Nov 30 – Dec 2.
- **Research and design.** Six read-only agents (kit audit; sound sources and licenses; taste elicitation / scorers / GRPO; instrument, taste-pipeline and infra designs) produced the reports now in `docs/research/` and `docs/design/`. `docs/research/decision_bank.json` lists the 103 decisions the completeness critic raised; the interview or the Claude defaults in TASTE.md cover them.
- **Bootstrap.** Bootstrapped `~/carc-workflow-kit` (HEAD 718def1) into a scratch dir, because it refuses non-empty targets, then copied in what lofi-rl keeps:
  - **Kept as is:** `_cluster_ssh.sh`, `watch_job.sh`, `scaling_study.py`, `session_report.py`; `src/kit/{chain,gates,bench,hf_cache}.py`; PLAYBOOK and LESSONS (now in `docs/kit/`).
  - **Adapted:**
    - `remote.sh`: no git pull; prints the mirror stamp.
    - `job_env.sh`: uv venvs, `MODE=policy|scorer|tools`, no git.
    - `sbatch_guard.sh`: adds the job-type registry and the soft target.
    - `cluster_setup.sh`: init is mkdir only.
    - `session_report.py`: stdlib only.
  - **New:**
    - `Makefile`: sync, pull, test, status, submit, facts, init, approve, backup-votes.
    - Scripts: `stamp.sh`, `submit.sh`, `check_deletes.sh`, `ssh_cluster.sh`, `approve_job.py`.
    - Jobs: `jobs/{tools,env_policy,env_scorer,fetch}.sbatch` and `jobs/APPROVED.yaml`.
- **CLAUDE.md** updated to match the decisions: rsync mirror plus stamp, `/project2` paths, registry rule, two envs, push rule, briefs, reward and gates.

**Numbers**
- **ssh.** The macOS resolver could not resolve `discovery.usc.edu`; `host` returned 10.72.0.13 and 10.72.0.14. BatchMode ssh by IP with `HostKeyAlias` logged in as `shresthr` on `discovery2.hpc.usc.edu` with no Duo prompt (command output, this session).
- **`myquota`** on the login node:
  - `/project2/anakano_429`: 88.00 GiB used of 1.00 TiB
  - `/scratch1/shresthr`: 33 KiB of 9.766 TiB
  - `/home1/shresthr`: 4.76 MiB of 100 GiB
- **Paths and modules.** `ls -ld /project/anakano_429` → no such directory; `/project2/anakano_429/shresthr` does not exist yet. There is no fluidsynth module; the conda modules are 23.10.0, 24.3.0 and 25.11.0. The login node has Python 3.11.9 and rsync 3.1.3.
- **Laptop tools.**
  - Homebrew installed rsync 3.5.1 (protocol 33) and fluid-synth 2.6.1, which is pinned.
  - `.venv` (Python 3.12.13) from `requirements/laptop.lock.txt` (27 pins): numpy 1.26.4, pedalboard 0.9.26.
- **Locks** (`requirements/*.lock.txt`, compiled for `x86_64-manylinux_2_28`):

  | Lock | Pins | Versions |
  |---|---|---|
  | policy | 211 | vllm 0.32.0, torch 2.13.0, transformers 5.19.0, trl 1.17.1, peft 0.21.2, numpy 2.3.5 |
  | scorer | 111 | torch 2.8.0, torchaudio 2.8.0, transformers 4.43.4, numpy 1.26.4, laion-clap 1.1.7, audiobox-aesthetics 0.0.4, pedalboard 0.9.26 |

- **Tests.** `pytest -q` on the infra suite: 174 passed, 10 skipped. The skips were the git-dependent privacy tests, run before `git init`; the result after `git init` is in the commit message.

**Failed approaches and why**
- **Plain `ssh discovery`** fails at DNS from the laptop (LESSONS 30). Fix: every script goes through `scripts/_cluster_ssh.sh`, which falls back to the IP with the pinned host key.
- **Bootstrap in place** refused because CLAUDE.md already existed. Fix: bootstrap into a scratch dir and copy.
- **Removed kit pieces:**
  - the kit's chained-job template, its tests and the bench jobs: they assume the kit's conda job env. They return with `jobs/grpo.sbatch` (Phase 3) and the render-scaling jobs (Phase 4).
  - `kit.publish` and its tests: not needed.
- **First run of the new guard test** wrongly passed with used SU at 0, because the stub sacct rows had no job name and the prefix filter dropped them. The test now names its rows.

**SU spent:** 0.

**Next**
1. Commit, create `slipstr34m/lofi-rl` (public) and `slipstr34m/lofi-rl-votes` (private), push.
2. `make init` (cluster dirs), `make sync`, then submit `tools` → `env-policy` and `env-scorer` → `fetch`.
3. The LoopScript v1 parser (a background workflow: implement, adversarial review, fix) lands as its own commit.
4. Then `jobs/smoke_env.sbatch` with a stub render. It needs a SoundFont: GeneralUser GS (permissive, ~30 MB), to be downloaded with Shresth's OK.
