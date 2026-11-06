# D-022's laptop evidence: the Phase-1 gate's render step

The laptop measurements behind D-022's render-step change (docs/DECISIONS.md D-022, "The Phase-1 gate's render
step"), copied byte for byte from the session scratchpad where they were made on 2026-11-04, so that they outlive it.
Each copy's sha256 equals its original's (`shasum -a 256`, 2026-11-04).

- **Not canonical.** Laptop renders are not canonical (docs/kit/LESSONS.md 38), so D-022 uses only the speedup
  ratios and the identity checks from them. The cluster's numbers are job B's, `runs/drafts-13706355/summary/`
  (`verify.json` `checks.render_time`, `renders.json`), which stay uncommitted with the rest of `runs/` (CLAUDE.md §7).
- **Paths.** The absolute paths inside the files and the scripts are that session's worktree
  (`.claude/worktrees/wf_93640418-d43-1`) and scratchpad. The `runs/` files they read are not committed.
- **Machine.** Apple M4, 10 CPUs (`machine` in both orders files). The frozen palette B, sha256 `09f80a64…`.

| file | sha256 | what it holds |
|---|---|---|
| `drafts_orders.json` | `c22c815e…` | The 42 drafts (drafts/v1 and drafts/v1/redo1) and prompt v2's 9 references, with job B's briefs, through `junk.render_clip` on a warm cache: one at a time with 4 workers (`wall_by_order` `1x4` 113.03 and 122.81 s) and two at a time with 2 workers each (`2x2` 67.39 and 70.28 s); `speedup_2x2` 1.713 (the means, 117.92 / 68.84). `identical_to_first_1x4`: 51 of 51 the same in all four runs (audio, FLAC and manifest sha256). |
| `armb_orders.json` | `7b6da6ce…` | 50 of D-020 arm B's loops (runs/junk-13640325), re-scored on the frozen palette (`meta`: pass 312, pass_not_copy 276) and sampled as `cmd_render` samples them: `1x4` 115.83 and 114.91 s, `2x2` 66.89 and 63.85 s, `speedup_2x2` 1.765; 50 of 50 the same in all four runs. `meta.keys_kits` is `[["rhodes", "jazz"]]`. |
| `armb_cli_compare.json` | `7e86821b…` | Two `python -m lofi_rl.rl.junk render` runs of that re-scored run, recorded as jobs 1 with 4 workers and jobs 2 with 2 workers: `same_order` true, `clips_same` 50 (records less render_s, FLAC bytes, manifest bytes). |
| `armb_compile.json` | `52d915d8…` | Arm B's 446 passing non-copies parsed and compiled under dev_gu (446 ok) and under the frozen palette (276 ok, 170 G4: "keys instrument 'organ' is not in the palette"). |
| `concurrent_fill_probe.json` | `413b7ce0…` | tests/test_junk.py's frozen-palette two-lane render as reordered after review T1: renders 0 and 2 start at 0.017 and 0.018 s; all 72 one-shot cache writes (36 files) come while both run; 36 files are left. |
| `render_orders.py` | `af1dc3e8…` | Made both orders files: `python render_orders.py <worktree> drafts|armb <out>`. |
| `armb_compile.py` | `6f27bb23…` | Made `armb_compile.json`: `python armb_compile.py <worktree> <run>`, its stdout. |
| `compare_cli.py` | `4b2156a2…` | Made `armb_cli_compare.json`: `python compare_cli.py <one-at-a-time run> <two-at-a-time run> <out>`. |
| `concurrent_fill_probe.py` | `1377d825…` | Made `concurrent_fill_probe.json`: `python concurrent_fill_probe.py <worktree> <out>`, with `LOFI_ASSETS` set. |
