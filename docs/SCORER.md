# SCORER.md: the scorer server, the audio part of the GRPO reward

The scorer server renders the completions that passed the symbolic gate, measures them with MERT, CLAP and
Audiobox, and returns the audio part of the reward: the render and PQ gate, the P_acc factor, r_ce, r_theme,
r_taste and the audio hack signatures. It runs in the scorer env, next to the trainer, on a node-local Unix socket.

- **Spec:** `docs/design/taste_pipeline.md` §4 (reward model, PAR), §5 (reward architecture: scorer server,
  render pool, caching, gate, Audiobox, CLAP theme, penalties), §6 (GRPO) and §7 (hack-watch signatures);
  `docs/TASTE.md` (Audiobox and CLAP rows, the contrast scenes); `CLAUDE.md` §8.
- **Code:** `src/lofi_rl/scoring/` (the table in its `__init__.py`); the hook in `src/lofi_rl/rl/reward.py`;
  the prompts in `configs/clap_theme_v1.yaml`. **Job:** `jobs/scorer_smoke.sbatch` (`lofi-scorer-smoke`,
  not registered or submitted yet).
- **Tests:** `tests/test_scoring_*.py`, with helpers in `tests/helpers_scoring.py` (§13).
- **Sound claims:** none. Every number here is a measurement or a formula; none says how a clip sounds.

## 1. Entry points

```bash
python -m lofi_rl.scoring.server --sock $TMPDIR/lofi-scorer.sock [--workers 8] [--cache $REMOTE_SCR/cache] \
    [--calib DIR] [--rm DIR] [--gpu-mem-gb 6] [--device cuda|cpu] [--palette palettes/dev_gu.yaml] \
    [--theme configs/clap_theme_v1.yaml] [--backend real|fake] [--render auto|real|fake] [--render-workers 1] \
    [--render-timeout-s 150] [--lru 4096]
LOFI_SCORER_SOCK=$TMPDIR/lofi-scorer.sock python -m lofi_rl.rl.train ...     # the reward hook (§10)
make submit JOB=jobs/scorer_smoke.sbatch                                     # once registered (§11)
```

```python
from lofi_rl.scoring.client import ScorerClient
c = ScorerClient("/tmp/lofi-scorer.sock")
c.wait_healthy(600)                                   # pins reward_ver, feat_ver, rm_ver, render_ver
out = c.score([{"i": 0, "text": loop, "brief": {...}, "brief_id": "596:0"}], palette_sha256=pal.sha256, step=3)
```

- The server prints one JSON line `{"event": "ready", "sock", "pid", "versions", "terms", "backend"}` on stdout
  when it answers, and one log line per request on stderr.
- **Exit codes:** 0 after SIGTERM or SIGINT (the socket removed); 2 for a bad argument, calibration, RM
  directory, theme config or socket path (an unreadable file, a YAML syntax error and an RM factory module
  that fails to import included), before the models load where possible; 1 for anything else (a
  traceback). SIGUSR1 prints every thread's stack.

## 2. Division of labour (decided): a deliberate narrowing of §5's item schema

§5 has the server return the whole reward per item: `{clip_id, hard, G, r{theory,knobs,ce,theme,taste},
pen{dup,bars,len,copy}, sig{…}, diag{rm_mean,rm_std,p_acc,pq,pc,cu}, t_ms}`. The split chosen instead keeps the
symbolic part where it already runs and is tested:

| part | where | why |
|---|---|---|
| gates G1–G6, G7-compile, G2 (truncation) | `lofi_rl.rl.reward` + `lofi_rl.verify` (policy env) | stdlib, already in the smoke |
| r_theory, r_knobs, P_sax, pen.bars, pen.len, P_fix, P_group (dup, near-dup) | policy env | the same |
| pen.copy | **not built here** (another branch owns the example-copy penalty) | |
| render (parse, compile, render in spec), PQ floor, P_acc factor | **server** | needs the renderer and the models |
| r_ce (Audiobox CE), r_theme (CLAP on the music stem), r_taste (RM, PAR) | **server** | the same |
| audio hack signatures: PQ, PC, CE, CU, CLAP margin, spectral centroid | **server** | taste_pipeline.md §7 |

So the server's item keeps §5's names where they survive and drops the rest:

    {i, clip_id, hard, G, gate, r{ce, theme, taste}, sig{pq, pc, ce, cu, clap_margin, centroid_hz},
     diag{rm_mean, rm_std, p_acc, pq, pc, cu, ...}, t_ms}

- `hard` is the audio part of §5's hard (parse ∧ compile ∧ render in spec, as the server re-runs them); the reward
  ANDs it with the symbolic gate, which holds the hard theory checks.
- `G` is the audio part of §5's gate: `hard × 1[PQ ≥ τ_PQ] × clip(P_acc / 0.5, 0.25, 1)`. The reward's G is
  `G_sym × G`.
- `r` has only the three audio terms; `pen` is gone. Only completions that passed the symbolic gate are sent:
  the others have G = 0 already, so rendering them would buy nothing.

## 3. Protocol (`lofi_rl.scoring.protocol`, standard library)

### 3.1 Transport and versions
- HTTP/1.1 + JSON over `AF_UNIX` (`client.UnixHTTPConnection`, an `http.client.HTTPConnection` subclass), one
  connection per request. Bodies are UTF-8 JSON with no NaN or infinity, at most 16 MiB.
- `protocol: "lofi-scorer/1"` in every request and response.
- Every response and `/health` carry four versions. `ScorerClient.wait_healthy` pins them; a later answer with
  another value raises, and every `/score` request sends them as `expect`, so the server answers 409 if it was
  restarted with another RM, other features or other formulas.

| version | value | changes when |
|---|---|---|
| `reward_ver` | `audio-reward/1` (`protocol.REWARD_VER`) | the gate, r_ce, r_theme or PAR formulas change (a manual bump) |
| `feat_ver` | `feat1.mert-12af15fe.clap-b3708341.abx-0.0.4.w4.src-<sha8>` | a model pin, the window count, or the render or feature code (`features.source_sha8`: dsl, compile, theory, render, brief, palette, prng, smoke.py, features.py, render_pool.py) |
| `rm_ver` | the RM's (`rm.json`), `null` without one | another RM |
| `render_ver` | `render-cfg-<sha8>` of §2's render config | the render schema or chain constants |

### 3.2 Requests
- `POST /score`: `{protocol, palette_sha256, step, items: [{i, text, brief: {energy, mood, bpm, tonic, scene},
  brief_id}], expect?}`. `i` is the completion's index in the reward's batch, unique; at most 1,024 items of at
  most 32,768 characters. The brief is validated as a `lofi_rl.brief.Brief`. Unknown keys are refused.
  `expect` is null (nothing pinned) or an object of pinned versions: `[]`, `0`, `""` or `false` is a 400,
  never read as nothing pinned.
- `POST /render`: `{protocol, palette_sha256, items, out_dir?, flac?}`: render only (no features, no cache). With
  an absolute `out_dir`, it writes `<clip_id>.manifest.json` and, with `flac`, `<clip_id>.flac` (the 16-bit
  dithered listening copy) there, for hack-watch and ear checks.
- `GET /health`, `GET /stats` (§4).

### 3.3 The response item (`protocol.check_item`)

| field | type | meaning |
|---|---|---|
| `i` | int | the request item's index, in request order |
| `clip_id` | `c` + 16 hex | §3.4 |
| `hard` | 0 or 1 | parse, compile and render in spec |
| `G` | 0–1 | hard × PQ factor × P_acc factor (§5.1); 0 when hard is 0 |
| `gate` | null or str | why hard is 0: `G1`…`G6` (the server's parse), `G7` (render out of spec), `compile`, `render-error` (an infrastructure fault), `render-timeout`, `parse-error` (a parser crash) |
| `r` | {ce, theme, taste}, each 0–1 | 0 when disabled or hard is 0 |
| `sig` | {pq, pc, ce, cu, clap_margin, centroid_hz} | the audio hack signatures; null when hard is 0 |
| `diag` | {rm_mean, rm_std, p_acc, pq, pc, cu, pq_factor, pacc_factor, tau_pq, u, theme, disabled, cache, dup_of, error, render} | `rm_mean`/`rm_std`: mean and std of PAR's centred utilities; `u`: each member's utility; `theme`: pos_mean, contrast_max, nearest_contrast and each scene's mean cosine; `disabled`: the terms off (§5.4); `cache`: `miss` (rendered now), `mem`, `disk`, `dup` (an earlier item's clip), `none` (did not parse); `render`: n, sr, seconds, bpm_eff, LUFS and true peak of the mix and the music stem, iterations, bed, pcm_sha256 of both, canon and compiled sha256, centroid_hz, render_ms |
| `t_ms` | {render, features, score} | this item's work in this request (0 for a cache hit) |

The response also has the versions, `terms` (which are live), `disabled`, `step`, and `batch` (n, unique,
rendered, cache counts, failing gates, parse/records/total ms).

### 3.4 Clip ids and dedup
- `clip_id = "c" + sha256(sha256(canonical LoopScript) | bpm,key,mode,energy | palette_id@sha8 | render_cfg_sha8)[:16]`,
  taste_pipeline.md §2's formula, equal to `lofi_rl.shootout.render.clip_id(manifest)`
  (`tests/test_scoring_protocol.py`), so the scorer's ids are the ids votes are kept under. The render config is
  `{schema, chain, flac}` of the render manifest, built from the pipeline's constants before rendering.
- The canonical form comes from parsing the text with the palette's vocabulary in the brief's mode. Texts with
  one canonical form, brief and palette are one clip, rendered and featurized once per request (`example.ls` and
  `example_leniency.ls` are one clip), and scored once per (clip, brief_id, scene): PAR's references are keyed by
  brief_id, so one clip under two brief ids is scored twice, and the later items say `cache: dup` with `dup_of`.
  A text that does not parse has no canonical form: its id hashes `"raw:" + sha256(text)` in the first field.
- The scene is not in the id: it changes no audio. It enters only r_theme, which is computed per item.

### 3.5 Status codes
400 a malformed request (bad JSON, schema, brief); 404 an unknown path; 405 another method; 409 another palette
or pinned version; 413 a body over 16 MiB; 422 a brief without PAR references while taste is live, or an unusable
`out_dir`; 500 a backend exception (a featurizer or RM failure, a non-finite feature), with `error`, `type` and the
end of the traceback. **The client raises `ScorerError` on anything but 200**, on a timeout (300 s), on an
unreachable socket and on a response that fails `check_score_response`.

## 4. The server (`lofi_rl.scoring.server`)
- **The socket** is created under umask 0177 and then chmod 0600: owner only, no window with wider permissions.
  It is node-local, so it cannot clash with another job's ports (taste_pipeline.md §5; LESSONS 17).
- **The path rules** (`check_sock_path`), checked before the models load and again at bind:
  - free: fine; longer than the platform's `sun_path` (103 bytes on macOS, 107 on Linux) or in a missing
    directory: refused;
  - a socket of this user that refuses connections (stale): removed;
  - a live socket (another server), a socket of another user, or anything that is not a socket (a file, a
    symlink): refused, exit 2, and never deleted.
- **Startup order:** arguments, palette (`verify=True` when it renders), theme config, calibrations and RM
  directory (cheap, exit 2 on a bad one), then the models (the featurizer), the render pool (warmed: every worker
  spawned and its palette loaded), the theme's text embeddings, and only then the socket. So the socket exists
  only when `/health` can answer ok.
- **Requests** run on threads (`ThreadingMixIn`), so `/health` and `/stats` answer while a batch is scored;
  `/score` and `/render` take one lock, since the models and the pool are shared.
- **GPU memory:** `torch.cuda.set_per_process_memory_fraction(gpu_mem_gb / total)` before any model loads
  (default 6 GB, §5). It caps the caching allocator of this process; the CUDA context comes on top
  (`nvidia-smi` in the smoke records the whole process).
- **Shutdown:** SIGTERM or SIGINT stops the loop from a helper thread; the socket is removed only if it is
  still the inode the server created; the render workers are terminated; exit 0. In-flight requests are
  abandoned (daemon threads): the trainer is gone by then.
- **`/health`:** ok, protocol, backend, versions, terms, disabled, palette (id, sha256, path), calibration
  flags (§5.4 and the RM, §6), theme config (path, sha256, version), renderer, featurizer (models, device, load
  time), GPU (cap, peaks), cache counters, pid, uptime.
- **`/stats`:** counts (requests, items, unique, dup, rendered, render failures by gate), per-stage timings
  (n, sum, mean, max ms: request, parse, render, featurize, mert, clap, audiobox, score), cache counters (memory
  and disk hits, misses, writes, unreadable files), GPU peaks, pool size and rebuilds.

## 5. The audio reward (`lofi_rl.scoring.core`, `calib`, `theme`)

### 5.1 The gate
`G = hard × 1[PQ ≥ τ_PQ] × clip(P_acc / 0.5, 0.25, 1)` (`core.gate_factor`), §5 verbatim. PQ at exactly τ passes;
P_acc above 0.5 earns nothing extra; P_acc below 0.125 is floored at 0.25.

### 5.2 CLAP theme (`configs/clap_theme_v1.yaml`, `theme.py`)
- **Prompts:** 8 city-at-night prompts plus the brief's scene against the 5 contrast scenes of TASTE.md (sunny
  morning, beach, forest, cafe afternoon, snowy cabin) × 4 prompts, all plain scene descriptions with no genre,
  instrument or mood words (`tests/test_scoring_core.py` checks the counts, the scenes and the words).
- **Audio:** the music-only stem (`Render.music`, before the rain bed), mono, 48 kHz, four 2-bar windows from
  `smoke.clap_windows` (each under laion_clap's 10 s, so `rand_trunc` never fires); e = the normalized mean of
  the four unit window embeddings.
- **Margin:** m = mean_j cos(e, p_j) − max_k mean_i cos(e, c_k,i), p the 9 positives and c_k scene k's prompts.
  It is logged as `sig.clap_margin` whether or not the term is live, so the pool's med and MAD can be computed.
- **r_theme = σ((m − med) / (1.4826·MAD))** with med and MAD from `theme.json`.

### 5.3 Audiobox
On the mix as heard (with the bed, as votes hear it). **PQ** is only the floor (§5.1). **CE** earns its weight
only when validated (§5.4); then r_ce = σ((CE − med) / (1.4826·MAD)) over the pool's CE. **PC is never
rewarded**; PC and CU are logged as signatures (`sig`, `diag`).

### 5.4 Calibration files (`--calib DIR`, `calib.py`)

| file | keys | missing means |
|---|---|---|
| `pq.json` | `tau_pq` (the 5th percentile of PQ over Good-or-better clips) | PQ factor 1 |
| `theme.json` | `med`, `mad` (> 0), `theme_sha256` (the prompt file's sha256) | r_theme 0 |
| `ce.json` | `agree`, `lower95`, `n_pairs`; `med`, `mad` when validated; optional `validated` | r_ce 0 |

- Any file may also carry `n`, `source`, `date`, `note`, `feat_ver`; any other key is an error.
- **CE validated** ⇔ agree ≥ 0.55 and lower95 > 0.50 (§5: "≥ 55 % of strict pairs … and the bootstrap lower bound
  is > 50 %"). The server decides it from the numbers; a file whose `validated` disagrees is an error.
- A theme calibration made for another prompt file (sha256) is an error: the margin's distribution depends on
  the prompts.
- **Missing disables, broken stops:** a missing file (or `--calib` not given) turns its term or factor off, and
  `/health` (`calibration.<name>.status: missing` with the effect), `terms`, the response's `disabled` and every
  item's `diag.disabled` say so. A file that exists but is malformed is never read as missing: the server exits 2.
  So does a `--calib` or `--rm` directory that is given but does not exist: a mistyped or purged path would
  otherwise run a whole run with every calibration, or taste, off. An existing directory without a file is
  still "missing" for that file.

## 6. The RM interface and PAR (`rm_api.py`)
- **Protocol** (`RewardModel`): `rm_ver`, `members` (M), `utilities(feats) -> [u_1 … u_M]`,
  `p_acc(feats) -> P(tier ≥ Fine)`, evaluated on a clip's cached features (`cache.ClipFeatures`: MERT, CLAP mix
  and music windows, Audiobox, and the canonical LoopScript for a symbolic backbone). `NullRM` is no RM.
- **`--rm DIR`:** `rm.json` = `{rm_ver, members, factory: "package.module:callable"}`; the callable gets the
  directory and returns the RM, whose `rm_ver` and `members` must match. `refs.json` = `{rm_ver, members, refs:
  {brief_id: M lists of 4 utilities}}`: each member's utilities of the brief's 4 step-0 reference samples. The
  real RM (`lofi_rl.taste`) does not exist yet; its loader will be such a factory.
- **PAR, exactly as §4:** d_m = u_m(x) − mean of the brief's 4 reference utilities of member m;
  **r_taste = σ(mean_m d_m − 1.0 · std_m d_m)**, std the population standard deviation over members.
  `diag.rm_mean` and `diag.rm_std` are those mean and std.
- **What is live:** no RM (no `--rm`, or a directory without `rm.json`; a given directory that does not exist
  is exit 2) → r_taste 0 and P_acc factor 1; an RM without `refs.json` → r_taste 0 and P_acc live;
  both → both live. A request with a `brief_id` that `refs.json` lacks (or none) while taste is live is a 422,
  checked before anything renders: the trainer crashes rather than train with taste silently off for some
  briefs. A broken `rm.json` or `refs.json` (unknown keys, mismatched versions or member counts, rows that are
  not 4 finite numbers, a factory that does not import, raises or returns something else) stops the server.

## 7. Render pool, features and caching
- **Render pool** (`render_pool.py`): a `ProcessPoolExecutor` with the spawn context; each worker loads the
  palette once. Each render is `lofi_rl.render.pipeline.render` on the canonical text: parse and gates, compile, a
  fresh FluidSynth process per stem (`synth.cpu-cores=1`), the sampler, the chains, the bed, loudness and G7;
  `render_workers` (default 1) FluidSynth processes and pedalboard threads at a time per render. It returns the
  finished mix, the music-only stem, the manifest and the mix's spectral centroid (magnitude-weighted mean
  frequency of the whole cycle's mono spectrum).
  - A gate failure (G1–G6, G7 out of spec, a compile error) is that item's outcome and is cached (deterministic).
  - An infrastructure fault (FluidSynthError, a file error) is that item's `render-error`, not cached.
  - A crashed worker breaks the pool for every render in flight: those are retried one at a time in a fresh pool,
    and a second break is that item's `render-error`.
  - A batch past `--render-timeout-s` (150 s, under the client's 300 s) gets `render-timeout` for what is left;
    the stuck workers are killed (CPython 3.12 has no public call for it, so through the executor's process
    table) and the pool is rebuilt.
  - The server featurizes each clip as its render finishes, so the GPU works while other clips render.
- **Features** (`features.py`, real backend):

| model | input | kept |
|---|---|---|
| MERT-v1-95M @12af15fe | the mix, mono, 24 kHz, four equal 2-bar windows in one batch | per-layer means over windows and frames, (13, 768) |
| CLAP music @b3708341 (HTSAT-base) | the mix and the music stem, mono, 48 kHz, four 2-bar windows each (`smoke.clap_windows`) | unit window embeddings, (4, 512) each |
| Audiobox aesthetics 0.0.4 | the mix, written to a float WAV under the server's temp dir (the call the Phase-0 smoke verified) | PQ, CE, PC, CU |

  Eval mode, `torch.no_grad`, float32 with TF32 off, cuDNN deterministic without benchmark autotuning,
  `use_deterministic_algorithms(warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, one clip per batch: a clip's
  features never depend on the rest of the request. A non-finite feature is a 500.
- **Cache** (`cache.py`): `<cache>/feat/<feat_ver>/<clip_id>.npz` (arrays `mert`, `clap_mix`, `clap_music` plus a
  JSON `meta`: the render numbers, Audiobox, the canonical text, the device) and an in-memory LRU of 4,096
  records. A record is a function of the clip id and feat_ver alone, so a resumed run or a restarted server
  reuses it. Writes go to a temporary file and a rename; an unreadable file, or one for another clip, schema or
  feat_ver, is a miss and is rewritten. Without `--cache` only the LRU is used.
- §5's `cache/score/<rm_ver>/<reward_ver>/` is not built: a score is a few dot products and an RM call on cached
  features, and it depends on the brief id (PAR) and the calibrations, so caching it would save nothing.

## 8. Choices the design left open (each pinned by a test)
1. **The split** of §2, and only symbolic-gate passes are sent.
2. **Clip id** = §2's formula as the shootout computes it, not a new one, so votes and scores share ids.
3. **feat_ver hashes the render and feature code**, since the clip id does not change when the renderer's
   output does (voicing V2 and V2.1 changed renders under the same schema): a code change starts a new cache
   directory instead of serving features of other audio. Cost: any edit to those files re-renders once.
4. **Features on the mix**, theme on the music stem (§5), Audiobox on the mix (votes hear the mix, and τ_PQ comes
   from voted clips), the centroid on the mix.
5. **r_ce** uses the same robust sigmoid as r_theme, with the pool's CE median and MAD in `ce.json` (§5 gives CE
   no scale).
6. **PAR's std** is the population std over members; references are exactly 4 per brief and member.
7. **Missing vs broken:** a missing calibration or RM disables its term (§5.4, §6); a malformed one stops the
   server. A brief without references while taste is live is a 422, not a silent 0.
8. **Theme embedding:** the normalized mean of the window embeddings (cosines need a unit vector); the brief's
   scene is the 9th positive and is embedded with the config's prompts in one batch at start.
9. **Server-side gate codes** add `render-error`, `render-timeout` and `parse-error` to the gates, so
   infrastructure faults are visible but zero only their own item (the task's rule), and are never cached.
10. **One lock** for `/score` and `/render`; threads for the rest.
11. **Fakes ship in the package** (`fakes.py`, `--backend fake`), so the trainer can rehearse against a real
    server on the laptop; `/health` says `"backend": "fake"`, and a client refuses one unless `allow_fake=True`.
12. **`--palette`, `--theme`, `--backend`, `--render`, `--render-workers`, `--render-timeout-s`, `--lru`** were
    added to the CLI the task named; every request carries the palette's sha256, and a mismatch is a 409.
13. **No score cache** (§7) and no recheck endpoint; the 1 % re-render determinism check of §2 stays with the
    queue job.

## 9. Fakes (`fakes.py`) and the laptop
`FakeRenderer` (no FluidSynth: records from the canonical text's sha256; injectable gate and infrastructure
failures), `FakeFeaturizer` (features of the real shapes drawn from a seeded generator; Audiobox overrides;
injectable exceptions), `TableRM` (hand-built utilities and P_acc), `FakeRM` and `load_fake_rm` (utilities from
the features, for the cluster smoke). They never render or hear audio. `python -m lofi_rl.scoring.server
--backend fake` serves them; `--backend fake --render real` renders for real with fake features.

## 10. The reward hook (`lofi_rl.rl.reward`)
- **Switch:** `make_reward(..., scorer=...)`: None reads `$LOFI_SCORER_SOCK` (unset or empty: no scorer), False
  is none, a path is a socket, anything else is a client. The GRPO config has no scorer field yet, because
  `lofi_rl.rl.config` refuses unknown keys and another branch owns it; until then a run turns the scorer on with
  the environment variable. The trainer's `--dry-run` passes `scorer=False`: it checks the symbolic path and
  records whether the variable is set (`reward.scorer`), without contacting the scorer.
- **Without a scorer** the rewards, `rewards.jsonl` and the per-step stats are byte-for-byte what they were
  (`tests/golden/reward_no_scorer.json`, recorded from main 96bd121 before the hook existed).
- **With one:** the first call waits up to 600 s for `/health`. After the symbolic records, the completions with
  `gate_ok` go out in one `POST /score`; any scorer failure raises whatever `strict` says, and nothing is logged
  for that call. Then, per completion:
  - `G = G_sym × server G`; `gate` is the symbolic gate, else the server's, else `PQ`; `gate_ok` ⇔ G > 0;
  - `r` gets the server's ce, theme and taste; `positive = G × Σ w_k r_k`; `reward = positive − penalty`;
  - **the CE rule:** when the server reports CE not validated, w_ce = 0 and w_knobs = 0.20 at every step
    (taste_pipeline.md §5, written beside W0: "ce→0, knobs→.20 unless CE validated"); `w` records the weights
    used and `w_rule` the rule. With CE validated the weights are the curriculum's.
  - **New fields:** `G_sym`, `audio` (the server's whole item), `clip_id`, `w_rule`, `scorer` (the four versions).
- **Per-step stats** gain `audio`: sent, hard_rate, G_mean, r means over gate_ok, signature means, P_acc and
  rm_std means, the rate of P_acc < 0.5, failing gates, PQ failures, render errors, the server's batch numbers,
  the call's ms, versions, terms and the rule. TRL's hooks get `lofi/G_mean`, `hack/<signature>` and
  `log_extra("clip_id", …)`.

## 11. The smoke (`jobs/scorer_smoke.sbatch`, `lofi_rl.scoring.smoke`)
- **Shape:** gpu partition, one L40S, 8 CPUs, 32 GB, 20 min: 8 + 32/4 + 8 = 24 SU/min, at most 480 SU.
  Not registered: Claude may register it itself (≤ 500 SU, CLAUDE.md §4) when it is to run.
- **Steps:** `setup` writes the batch (4 briefs × 8 from `tests/fixtures/loops`: 24 distinct clips, 20 renders,
  4 G1 items), smoke-only calibrations (τ_PQ 1.0, a theme med/MAD, a CE "validation") and the fake RM with zero
  references, so every term runs live on real features; server **a** (cold cache) scores the batch twice;
  server **b** (fresh process, cold cache) once; server **c** (fresh process, a's disk cache) once; each is
  stopped with SIGTERM. The sockets live under `$TMPDIR` (`/tmp` when that path would pass 90 characters), mode
  0600, in a 0700 directory that is removed at the end.
- **Verdict** (`summary/scorer_smoke.json`, `SCORER_SMOKE_OK` or `SCORER_SMOKE_FAILED`): the expected gates and no
  infrastructure failure in any pass; a's second pass all memory hits and identical; b within `smoke.TOL` of a on
  every audio term (G, r, sig, P_acc, rm_mean, rm_std; the max abs differences recorded); c all disk hits and
  identical; feature arrays of a's and b's caches compared (max abs per array, pcm mismatches); every pass's
  per-stage timings, cache counters and GPU peaks; nvidia-smi after pass a; exit codes 0 and sockets removed.
- **Timeouts:** every step runs under `timeout`, sized from the job's remaining time (`kit.chain.end_time`) so
  the verdict always runs.

## 12. Unverified until the cluster smoke
- laion_clap 1.1.7: `get_audio_embedding_from_data` on two batches of 4 windows per clip, and
  `get_text_embedding` on 29 texts in one batch (the Phase-0 smoke used 4 windows and 2 texts).
- MERT: `Wav2Vec2FeatureExtractor` on a list of 4 equal windows (the smoke used one), and its normalization.
- Audiobox: `initialize_predictor()` on the GPU under the memory cap; one clip per call.
- Determinism across processes on one GPU (b vs a), the 6 GB cap with all three models, and every timing:
  model loads, renders with 7 workers on 8 CPUs, MERT, CLAP, Audiobox, a whole 32-item batch.
- §5's sizing (n* ≈ 7 render workers on an L40S) and its "reward 9 s" step estimate.

## 13. Tests

| file | what it checks |
|---|---|
| `test_scoring_protocol.py` | clip id = the shootout formula, its parts, canonical-form sharing; request round trips; 25 malformed requests; NaN; response and health checks; pinned versions |
| `test_scoring_server.py` | the real server on a real socket with fakes: health, score (order, fields, dedup, memory hits, stats), render, 400/404/405/413, 500 with its trace, 409, render gate and infrastructure failures, mode 0600, the path rules, close() leaving a socket that replaced its own, the client's errors, a real client refusing a fake backend, SIGTERM and SIGINT removing the socket (a subprocess), exit 2 on a foreign path |
| `test_scoring_core.py` | G on 11 hand-built cases, PAR on hand-built utilities, the RM with and without references, the 422, the theme margin on hand-built embeddings, the prompt file, every term off without calibration and on with it, CE that fails its check, the disk cache read by a fresh scorer, an unreadable file, LRU eviction, the fakes |
| `test_scoring_calib_rm.py` | the calibration and RM loaders: missing, valid, every malformed case, CE thresholds, exit 2 before binding |
| `test_scoring_render.py` | two real renders through the pool (example.ls, dev_gu): `/score`, where the mix and music stem reach the featurizer once and their hashes match the manifest and the laptop's golden hashes, and `/render` to a FLAC and manifest named by clip id with the same PCM; a gate failure in a worker, a batch past its deadline, a broken pool, the centroid |
| `test_scoring_reward_hook.py` | the no-scorer golden byte for byte; the env var; G, r, gate and the new fields with a stand-in client; the CE rule; failures raise; nothing sent without a pass; TRL hooks; the real client and server end to end |
| `test_scoring_job.py` | the job's size, cost line, env, timeouts, steps and outputs; the smoke's batch; the steps rehearsed on fakes to SCORER_SMOKE_OK; the verdict failing on drift, a missing pass, a bad exit and a left socket |
| `test_scoring_imports.py` | every scoring module imports without the ML stack; client, protocol and the reward without numpy or the audio stack |
