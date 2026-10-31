# lofi-rl

A small open LLM (Qwen3-4B + LoRA, GRPO) learns to write 8-bar jazzhop loops in a constrained text DSL,
**LoopScript**. Its reward comes from one listener's ear: a taste model trained on Shresth's best/worst
votes, plus music-theory verifiers and knob checks. Final project for USC CSCI 596 (Fall 2026).

- **The spec is a taste profile:** [docs/TASTE.md](docs/TASTE.md), from a 27-round interview.
- **Irreversible and pre-registered choices:** [docs/DECISIONS.md](docs/DECISIONS.md).
- **Lab notebook:** [CHANGELOG.md](CHANGELOG.md).
- **Research and designs** behind the plan: [docs/research/](docs/research/) and [docs/design/](docs/design/).

## Layout

| Path | What |
|---|---|
| `src/lofi_rl/` | the project: `dsl/` (LoopScript parser), then compiler, renderer, verifiers, taste model, rating UI, RL |
| `src/kit/`, `scripts/` | laptop-to-cluster workflow from carc-workflow-kit, adapted to an rsync mirror (no git on the cluster) |
| `jobs/` | Slurm jobs; each is registered with its SU estimate in `jobs/APPROVED.yaml` |
| `requirements/` | pinned locks for the laptop, policy and scorer envs |
| `palettes/` | the frozen sound palette, from Phase 1 |
| `gates/` | human approvals (`APPROVED by Shresth Rana <date>`) |

## Running it

```bash
make test        # pytest -q on the laptop (.venv from requirements/laptop.lock.txt)
make sync        # mirror the code to Discovery (GNU rsync, guarded --delete)
make submit JOB=jobs/tools.sbatch
make status      # queue, guard log, SU total
make pull        # run summaries and ear-check clips back to the laptop
```

Who did what: written up in Phase 5 (CLAUDE.md §9).
