"""The keep3 drafts: 30 hand-drafted loops (3 personas x 10 briefs), of which Shresth keeps 10 (docs/DRAFTS.md).

docs/design/taste_pipeline.md §8 ("30 drafts -> keep 10"), docs/TASTE.md (Labeling: "Claude drafts 30 loops and he keeps
10; the rejects become labeled negatives") and docs/SHOOTOUTS.md §1 step 3 (the drafts play the Day-2 picks).

- `briefs`: the 10 briefs, drawn by a recorded rule from a recorded seed (drafts/v1/briefs.yaml);
- `personas`: the three writer ids (their cards are in docs/DRAFTS.md);
- `palette`: the draft palette (palettes/draft_day2.yaml): the Day-2 picks, their trims and faders, and the one palette
  per (keys, kit) each render plays;
- `check`: the validator, `python -m lofi_rl.drafts.check <dir>`;
- `render`, `batch`, `job`: the cluster job (`jobs/drafts.sbatch`, `python -m lofi_rl.drafts.job`) that renders the
  drafts and builds the keep3 batch.

Sound claims: none. A draft that passes every check is only a draft; Shresth's ear decides.
"""
