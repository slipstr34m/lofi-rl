"""The scorer server: the audio part of the GRPO reward (docs/SCORER.md; docs/design/taste_pipeline.md §4-5, §7).

  protocol     the wire schema: versions, requests, responses, validation, clip ids (stdlib)
  client       ScorerClient: HTTP/1.1 + JSON over a Unix socket; the policy env imports it (stdlib)
  calib        tau_PQ, the theme median/MAD and the CE validation result, read from files (stdlib)
  rm_api       the reward-model protocol, the null RM, the RM loader and PAR (stdlib)
  theme        configs/clap_theme_v1.yaml and the CLAP theme margin m and r_theme
  cache        the feature cache: <cache>/feat/<feat_ver>/hw-<sha8>/<clip_id>.npz plus an in-memory LRU
  render_pool  a spawn-context process pool around lofi_rl.render.pipeline.render
  features     MERT-v1-95M, CLAP music and Audiobox aesthetics (torch etc. imported inside functions)
  core         Scorer: dedup, cache, render, features, theme, RM, the audio gate; backends injected
  server       python -m lofi_rl.scoring.server --sock PATH ...: the HTTP server on a 0600 Unix socket
  fakes        deterministic stand-in backends for the laptop tests and rehearsals (no audio claims)
  smoke        the steps of jobs/scorer_smoke.sbatch: setup, pass, verdict

The symbolic reward (gates G1-G6, theory, knobs, P_sax, penalties, P_fix) stays in lofi_rl.rl.reward. Nothing
here imports torch, transformers, laion_clap or audiobox_aesthetics at module level, so every module imports on
the laptop (tests/test_scoring_imports.py), and client and protocol import in the policy env.
"""
