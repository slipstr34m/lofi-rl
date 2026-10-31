"""Phase-1 shootout tooling (docs/SHOOTOUTS.md §7).

- `plan`: shootouts/plan.yaml, the plan in symbols (§4 and §6), its strict loader, `substitute` and `apply_cuts`;
- `palettes`: the palette generator (§7.8), measuring palettes and one palette per drawn band;
- `levels`: the level-matching tool (§3, §7.1);
- `sour`: the anchors' sour copies and their gate (§2.5);
- `draw`: the job-time draw of option maps, round order and hidden controls (§2.3-§2.5);
- `batch`: the batch builder CLI, `draw` and `build` (§7.6);
- `render`: the job's clip renders (§7.7);
- `job`: the `lofi-shootout` job's stages in order (§7.7).

Sound claims: none.
"""
