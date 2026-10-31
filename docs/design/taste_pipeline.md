# Taste pipeline and reward design (lofi-rl, 2026-10-30)

**Defaults:** I applied all of them. I have no disagreements, and I refine two:
- **Render pool:** sized by SU cost (§5), not half the node.
- **Data location:** votes, RM, features and both venvs live outside the `--delete` mirror, at `$REMOTE_DATA=/project2/anakano_429/shresthr/lofi-rl-data` and `…/envs/`. The reason is that openrsync's protection of excluded paths under `--delete` is unverified (kit.md §8). `data/` is listed in both `.gitignore` and `.rsyncignore`.

## 1. Rating UI

**Server and page**
- Served at `127.0.0.1:8596` via `make rate`.
- Server: `src/lofi_rl/ui/serve.py` (stdlib `ThreadingHTTPServer`, append-only writer, loads no model).
- Page: `src/lofi_rl/ui/static/{index.html,app.js,audio.js}` (vanilla JS, no npm).

**Audio**
- All clips in a round share one brief, so they share one tempo and one sample length.
- All sources start at one `AudioContext` time with `loop=true`.
- Switching only moves gains: 5 ms raised-cosine ramps, no crossfade (BS.1534), so the bar position is kept.
- The page shows the scene text, tiles 1–k with heard-bar shading, and a bar ruler. It never shows the RM.

**Round mechanics**
- Auto-play runs clip 1→k, 8 bars each.
- An early pick is allowed at any time. Marking the clip that is playing skips to the next unheard clip.
- A bar counts as heard if the clip's gain was up for ≥75 % of it. A clip counts as heard at ≥4 distinct bars (`configs/rating.yaml`, see Q2).
- Pairs with an unheard clip are dropped at fit time. A triage Toss needs only ≥2 bars.

| Mode | Unit → output |
|---|---|
| `shootout` | ≤4 blind variants of one LoopScript (drum kits use drums+bass clips; the 12 vamps go in 9 incomplete blocks) → best−worst counts → keep 2–3, plus an `APPROVED by` line in `gates/GATE_palette.md` |
| `keep3` | 3 drafts → keep 1, or "none acceptable" |
| `bws4` | 4 clips from one brief → best, worst, escape hatches, one tag |
| `triage` | 1 clip → Toss/Fine/Good/Great. Also used for the 50-clip check ("fine" = ≥Fine) and for hack-watch |
| `ab` | 2 clips → forced pick (final A/B) |
| `audit` | after voting: counts, heard and anchor rates, position-of-best χ², repeat p with Wilson CI, hours vs budget; replay; append-only `retract` |

```
1 2 3 4 | 5  listen k (bar-synced)   | 5 replay from bar 1
Q W E R | T  best = k (ab: Q/W)      | T no clear best
A S D F | G  Toss/Fine/Good/Great    | G all four bad / none acceptable
Z X C V | B  worst = k               | B no clear worst
J fake  K no groove  L sour  ; boring  |  U nodding  I sax line  O chords
(negative tag → worst, positive → best) · Space play · Enter commit · Bksp undo · Esc end
```

**Hidden controls** (listed in `manifest.hidden.json`, which is never served; file names are opaque per batch)
- **Repeats:** 10 % of rounds, shown ≥2 days later with positions reversed. Held-out tuples are repeated first.
- **Broken anchors:** 5 % of rounds. One clip is replaced by a broken copy of another clip in the round. The expected answer is worst (or Toss in triage).

**Logging:** order, position, last-heard clip, switches, the `heard_bars` bitmask, `heard_ms`, time to each mark, and device. A `devicechange` pauses playback.

**Resume**
- Each event is fsynced, and `data/votes/state.json` holds the cursor.
- A gap under 10 min continues the micro-session.
- An uncommitted round is logged as `abandon` and re-queued at the end.
- A break nudge appears at 25 min (BS.1116).

**Push notification** (needs a live Claude session; with Remote Control it reaches his phone)
1. A backgrounded until-loop sees `batch_done`.
2. Claude runs `make votes-backup queue`, then `scripts/watch_job.sh <id> 120 60` in the background.
3. On exit 0: `make pull inbox-check`, then `PushNotification` with "inbox 11-10b: 45 rounds + 16 checks, ~65 min, 127.0.0.1:8596".
4. On exit 2: push "Duo re-open needed".

## 2. Votes, ids, transport

`data/votes/YYYY-MM-DD.jsonl`, schema in `src/lofi_rl/taste/schema.py` (v1):
```json
{"v":1,"type":"bws4","event_id":"…","ts":"2026-11-10T21:14:03-08:00","session":"s1110a","micro":2,
 "batch":"b1110-01","round":"r…","brief_id":"…","scene":"…",
 "clips":[{"clip":"c…","pos":1,"heard_bars":"11110000","heard_ms":11800}],
 "best":"c…","worst":"c…","no_best":false,"no_worst":false,"all_bad":false,"tag":{"name":"sax_line","clip":"c…"},
 "last_heard":"c…","switches":3,"t_ms":{"best":41000,"worst":52000,"commit":55000},"device":"airpods","ui":"<sha>",
 "hidden":{"kind":"repeat|anchor|heldout|none","of":"r…"}}
```
Other event types: `triage{tier}`, `keep3`, `shootout{category,options}`, `ab{left,right,pick}`, `retract{target}`, `session`, `abandon`.

**Ids**
- `clip_id = "c" + sha256(sha256(canonical LoopScript) | bpm,key,mode,energy | palette_id@sha8 | render_cfg_sha8)[:16]`.
- `$REMOTE_DATA/clips/registry.jsonl` adds `pcm_sha`, the sha256 of the canonical int16 PCM.
- A mismatch is logged as `determinism_violation`. The queue job re-renders 1 % of clips to check.

**Storage**
- `data/votes/` is a nested git repo whose remote is a private `lofi-rl-votes` (Shresth creates it).
- `make votes-backup` commits and pushes it, then copies to `$REMOTE_DATA/votes/` with `rsync -a` (no `--delete`).
- `tests/test_privacy.py` fails if anything under `data/votes` is tracked.
- Shootout votes cast before the palette freeze never train the RM.

**Clip transport**
- Canonical WAVs stay in `$REMOTE_SCR/renders/`. Voted clips persist as LoopScript plus features, since renders are deterministic.
- Listening copies are FLAC at PCM16 (≈2.3 MB per clip):
  - Lossless, so he hears the canonical PCM.
  - No encoder priming, so looping and switching stay sample-exact.
- Copies go to `$REMOTE_SCR/inbox/<date>/<batch>/` with a `manifest.json` and are pulled by `make pull`. `make inbox-check` verifies count, sha256 and loop length.
- Phase 0 checks that Safari's `decodeAudioData` accepts FLAC; the fallback is WAV.

## 3. Pair and round selection

`jobs/lofi-queue.sbatch` (main partition, 8 CPU, 16 GB, ≤360 SU) runs on the cluster. It:
1. refits the RM on CPU for selection only;
2. re-renders the picked clips from their logged LoopScripts;
3. writes the manifests and `queue_report.json`.

| Slot | Phase 2 (seed pool, by brief) | Phase 3+ (one on-policy group of 8) |
|---|---|---|
| random | 35 % | 35 % |
| active (BALD) | 34 % (random until 200 votes) | 40 % |
| held-out (40 tuples) | 16 % (all done by Nov 15) | – |
| checkpoint quads (dev briefs, steps 0/25/50/100) | – | 10 % |
| repeats / anchors | 10 / 5 % | 10 / 5 % |

**Active selection**
- BALD score: I(S) = H[mean_m P_m(b,w|S)] − mean_m H[P_m(b,w|S)]. It targets disagreement between ensemble members, not near-ties.
- Per batch: ≤2 quads per brief, ≤2 appearances per clip, and clips with P_acc < 0.2 are skipped.
- Positions are balanced with a Latin square.

**Brief splits** (`data/briefs/briefs_v1.jsonl`): train 400 / RM held-out 40 / dev 40 / final 100.

**Held-out set**
- 40 briefs, stratified by energy×mood.
- Each brief contributes its first 4 step-0 samples that pass the gate (seed 596).
- Stored in `docs/prereg/heldout_v1.json`, with its sha recorded in DECISIONS.md, and committed before the first vote.
- Never trained on, selected on, or tuned on.

**Final A/B** (`docs/prereg/final_ab_v1.json`)
- Each arm uses its first gate-passing sample out of ≤4 seeded draws; an arm with none forfeits that brief.
- Step-N is the checkpoint-quad winner, fixed before Nov 30.
- 5 anchors and 10 repeats are added and excluded from the statistic.

## 4. Reward model

Code in `src/lofi_rl/taste/{features,rm,fit,gate}.py`; the fit runs on CPU in `jobs/lofi-rmfit.sbatch`. Features are computed on the mix as heard and cached.

**Backbones**
- **MERT-95M** at 24 kHz, four 2-bar windows. Per-layer means (13×768) go through per-layer PCA, then z = Σ softmax(w)_l z_l.
- **CLAP music** at 48 kHz, four 2-bar chunks ≤8 s with repeatpad, so `rand_trunc` never fires. Mean of the normalized embeddings, then PCA.
- **Symbolic (SYM):** about 80 LoopScript features (lane density, velocity mix, syncopation, sax shape, per-bar edits, vamp and voicing).

**Backbone selection**
- Runs at 400 strict votes and is re-checked at 600.
- 5-fold GroupKFold by brief, scored on pairwise log-loss.
- Grid: d ∈ {8,16,32,64}, L2 ∈ 10^{−3…1}, and MaxDiff vs batched implied pairs.
- Every backbone within 1 SE of the best joins the ensemble. The choice is recorded in DECISIONS.md.

**Likelihood**
- Computed over the heard set H.
- Utility: v_i = u_i + β_pos + γ·[last heard]. The β and γ terms are dropped at inference.
- Best b and worst w: P = e^{v_b−v_w} / Σ_{i≠j∈H} e^{v_i−v_j}.
- "No clear best": P = e^{−v_w} / Σ e^{−v_i}; "no clear worst" is the mirror case.
- Both hatches: Rao-Kupper ties, P = 1 − σ(Δ − ln θ) − σ(−Δ − ln θ), with θ = 1 + e^η.
- `ab` and `keep3` use plain BT. Triage tiers add within-brief pairs at weight 0.5.

**Heads**
- A shared W (8×d) feeds three heads:
  - utility u = vᵀWz;
  - ordinal acceptability P(tier ≥ k) = σ(eᵀWz − c_k);
  - 7 tag logits, used only for diagnostics and never rewarded.
- Loss: NLL + 0.5·NLL_ord + 0.2·ΣBCE + λ‖·‖².
- Acceptability labels:

| Source | Label | Weight |
|---|---|---|
| triage tiers | tier | 1 |
| "all four bad" | Toss ×4 | 0.5 |
| kept loops | ≥Fine | 1 |
| "none acceptable" | Toss | 1 |
| broken copies | Toss | 0.25 |

**Ensemble and PAR**
- M = 5 bootstrap members, assigned round-robin over the selected backbones.
- d_m = u_m(x) − mean_{Ref_b} u_m, where Ref_b is 4 step-0 samples of brief b.
- **r_taste = σ(mean_m d_m − 1.0·std_m d_m).** This is a pessimistic P(he prefers x over a typical step-0 loop).
- The group baseline cancels the centering, so what PAR adds is the bound.
- `rm_ver` is pinned per run.

**Gate** (pre-registered, `gate.py`)
- acc = the share of held-out strict implied pairs that the ensemble mean orders correctly (first presentations only).
- p = agreement on pairs implied by both presentations of a repeat.
- **q = (1 + √max(0, 2p−1)) / 2; pass iff acc ≥ 0.9q.**
- 95 % CIs come from a cluster bootstrap (10⁴ resamples). Absolute acc vs 0.70 is reported alongside.
- Learning curves at 200 / 400 / 800 votes, 5 seeds each, fitted to a − b·n^{−c}.

## 5. Reward architecture

**Environments**
- Scorer env: transformers 4.43.x plus MERT, CLAP, audiobox, FluidSynth 2.6.1, pedalboard, pyloudnorm.
- Policy env: vLLM 0.32, transformers 5.19, TRL 1.17.

**Scorer server**
- Started by `jobs/lofi-grpo.sbatch` as `python -m lofi_rl.scoring.server --sock $TMPDIR/lofi-scorer.sock`. The trainer waits for `/health`, and a trap kills the server on exit.
- Transport is HTTP/1.1 + JSON over a node-local Unix socket (mode 0600), so it cannot clash with ports of other jobs on the node.
- `POST /score` takes the whole generation batch, so group penalties see all 8 completions of a prompt. Each item returns `{clip_id, hard, G, r{theory,knobs,ce,theme,taste}, pen{dup,bars,len,copy}, sig{…}, diag{rm_mean,rm_std,p_acc,pq,pc,cu}, t_ms}`.
- Other endpoints: `/health`, `/render`, `/stats`.
- The client times out at 300 s. Any error crashes the trainer, which then resumes from its checkpoint.
- The scorer caps itself at 6 GB of GPU memory.

**Render pool**
- A spawn-based `ProcessPoolExecutor`, with a fresh synth and `cpu-cores=1` per render.
- Optimal size n* = √(A·B / T0):
  - A = 32·t_clip / 0.9 = 71 worker-seconds, assuming t_clip = 2 s (UNVERIFIED);
  - B = 8 (GPU) + 16 (64 GB RAM) + 4 = 28 SU/min;
  - T0 = 25 s on A100, 40 s on L40S.
- That gives n* ≈ 9 (A100) or 7 (L40S), so request `-c 12` and run 8 workers.
- Each CPU thread bills 1 SU/min against 8 SU/min for the GPU, so half a node would cost more than the GPU.

**Caching**
- `$REMOTE_SCR/cache/feat/<feat_ver>/<clip_id>.npz` and `cache/score/<rm_ver>/<reward_ver>/`, plus an in-memory LRU of 4,096.
- Duplicate LoopScripts are rendered once.

**Reward function:** `src/lofi_rl/rl/reward.py` (stdlib only; check the `log_*` signatures against the TRL 1.17.1 source during the smoke):
```python
W0 = dict(theory=.50, taste=.20, knobs=.15, ce=.05, theme=.10)   # ce→0, knobs→.20 unless CE validated
W1 = dict(theory=.10, taste=.60, knobs=.15, ce=.05, theme=.10)
def lofi_reward(prompts, completions, completion_ids, trainer_state, log_extra, log_metric, brief, **_):
    s = min(trainer_state.global_step / 30, 1.0)
    w = {k: (1 - s) * W0[k] + s * W1[k] for k in W0}
    res = SCORER.score([c[0]["content"] for c in completions], brief,
                       step=trainer_state.global_step, n_tok=[len(t) for t in completion_ids])
    R = [o["G"] * sum(w[k] * o["r"][k] for k in w) - sum(o["pen"].values()) for o in res]
    ok = [o for o in res if o["hard"]]
    for k in w: log_metric(f"lofi/r_{k}", mean([o["r"][k] for o in ok]))
    for k in SIG: log_metric(f"hack/{k}", mean([o["sig"][k] for o in ok]))
    log_metric("lofi/G", mean([o["G"] for o in res])); log_extra("clip_id", [o["clip_id"] for o in res])
    return R
```

**Gate:** G = hard × 1[PQ ≥ τ_PQ] × clip(P_acc / 0.5, 0.25, 1)
- hard = parse ∧ compile ∧ render ∧ the hard theory checks.
- τ_PQ = the 5th percentile of PQ over Good-or-better clips.
- P_acc above 0.5 earns nothing extra.

**Audiobox**
- PQ is only a junk filter, and PC is never used.
- CE gets weight 0.05 only if the sign of ΔCE agrees with ≥55 % of strict pairs in the first 400–600 votes and the bootstrap lower bound is >50 %.

**CLAP theme** (`configs/clap_theme_v1.yaml`), computed on the music stem before the rain bed is added:
- Prompts: 8 city-at-night prompts plus the brief's scene, against 5 contrast scenes × 4 prompts.
- m = mean cos⁺ − max_k mean cos_k.
- r_theme = σ((m − med) / (1.4826·MAD)), with median and MAD taken over the pool.

**Penalties**

| Penalty | Amount |
|---|---|
| Identical LoopScript in a group | 0.3 per extra copy |
| Near-identical (symbolic cosine ≥ p99.5 of same-brief Claude pairs, so close cousins pass) | 0.15 |
| Largest identical-bar class m | 0.1·max(0, m−3), cap 0.3 |
| Length | 0.002·max(0, n_tok − p95), cap 0.2 |
| Copy (MinHash 5-gram Jaccard vs pool and prompt examples) | 0 at 0.6, linear to 0.3 at ≥0.8 |

## 6. GRPO (`configs/grpo_{smoke,run1}.yaml`)

| Setting | Value |
|---|---|
| Model | Qwen3-4B-Instruct-2507, pinned sha |
| LoRA | r 16, α 32, all-linear |
| Learning rate | 1e-5, constant, 3 warmup steps (taste.md C4) |
| Batch | 4 prompts × `num_generations` 8 = 32 completions per step; LoRA wants ≤~32. Per-device × grad-accum: 8×4 on A100, 4×8 on L40S/A40. `num_iterations` 1 |
| `max_completion_length` | 512, then 1.3× the pool's p99 |
| `mask_truncated_completions` | on |
| β | 0.04 (see below) |
| `loss_type` | dapo |
| `scale_rewards` | batch |
| Sampling | T 1.0, top_p 1, top_k 0 (no IS bias, taste.md C2); `make junk-rate` uses the same settings |
| vLLM | colocate, memory 0.25 on A100-80 or 0.30 on 48 GB cards; sleep mode on 48 GB cards |
| Smoke | 10 steps, saved every 5, then relaunched from checkpoint-5 to test resume |
| Run 1 | 100 steps, saved every 25 |

- **β = 0.04:** SMART's long β=0 run collapsed diversity (taste.md A7), and no-collapse is one of our pre-registered claims. With PEFT the reference is the adapter-off base, so the cost is one extra forward pass (about +4 s/step on A100) and no memory.
- **`scale_rewards` batch:** group-std scaling amplifies RM noise in groups whose completions are nearly identical.
- **Server mode** is used only for the multi-GPU runs in Phase 4.
- **`vllm_structured_outputs_regex`** is a lever:
  - on only if step-0 parse failures exceed 20 %;
  - off if the mean |log IS ratio| exceeds 0.05 (the masked sampler mismatches the trainer; UNVERIFIED).

| Estimate (UNVERIFIED) | A100-80GB | L40S |
|---|---|---|
| Generation + LoRA sync | 8 s | 12 s |
| Reward | 9 s | 9 s |
| Train (≈49k tok, 1.6e15 FLOP at 35 % MFU) | 14 s | 25 s |
| Overhead | 3 s | 3 s |
| **Step** | **34 s** | **49 s** |
| 100 steps + 7 min startup | 64 min | 89 min |
| SU at 36/min (GPU 8 + 12 CPU + 64 GB) | **≈2,300 ±35 %** | **≈3,200 ±35 %** |

- The smoke costs about 550 SU on an A100 or 670 SU on an A40 (debug partition).
- Every one of these is over 500 SU, so ask once for `lofi-grpo`.
- `--time`: 0:40 for the smoke, 1:45 on A100, 2:30 on L40S.

## 7. Hack-watch

**Daily set:** 16 blind triage clips, shuffled, about 5 min:
- the top 8 by R, at most one per brief;
- the top 4 by r_taste;
- 4 random gate-passing controls.

**Alarm** if any of these hold:
- ≥3 of the top 12 are Toss;
- the top 12's mean tier is ≤ the controls' mean tier;
- a signature stays outside its band for 3 consecutive 5-step windows (band = [p2.5, p97.5] of Good-or-better clips, frozen in `runs/<run>/bands.json`);
- within-group MERT distance falls below 0.7× its step-0 value.

**On alarm**
1. Write `STOP`.
2. Add an entry to `docs/HACKS.md`: signature, clip ids, steps.
3. Add a verifier or penalty.
4. Add the labels to the next RM.
5. Resume from the last clean checkpoint.

**Signatures logged** (`hack/*`):
- **Sax:** notes per bar, range, and rest / chromatic / blue fractions.
- **Keys:** hits per bar, polyphony.
- **Drums:** hits per bar per lane, ghost and accent fractions.
- **Bass:** notes per bar, kick-lock.
- **Arrangement:** empty-beat fraction, largest identical-bar class, bar edits.
- **Output:** token count, distinct completions per group.
- **Audio scores:** PQ, PC, CE, CU, CLAP margin, spectral centroid.
- **RM:** rm_std, rate of P_acc < 0.5, Spearman(r_taste, sax density).
- **RL:** KL, entropy, clipped ratio, frac_reward_zero_std, IS ratio.

## 8. Seed pool

**Plan:** `python -m lofi_rl.seeds.plan --seed 596` produces 220 briefs stratified by energy × mood × tempo, with 5 style cards each, split into 22 shards. The cards come from a Latin hypercube over:
- drum pocket: straight, late snare, half-time, ghost shuffle;
- bass: kick-lock, syncopated, sparse, push;
- voicing;
- sax: long tones, blues riff, motif + sequence, call/response with rests;
- 12 vamps.

**Workflow:** `workflows/seed_pool_v1.js`. Args are the shard paths, and personas vary by shard index. Each shard runs a pipeline:
1. **Write** 50 loops.
2. **Check** with `python -m lofi_rl.dsl.check --repair --log seeds/claude/v1/repairs.jsonl`.
   - Deterministic repairs only: case and whitespace, instrument aliases, grids padded or truncated to 16 steps, chord symbols at edit distance 1.
   - Anything else is rejected and rewritten, at most twice.
   - If more than 20 % of a shard needed repair, the shard re-runs with its repair log.
3. **Diversity critic:** the 5 loops for a brief must differ on ≥2 axes and use ≥3 vamps.

After all shards, a barrier runs `seeds.dedupe` and a coverage report, then backfills until the counts are met. Agents only parse. Gate-passing loops go to `jobs/lofi-render-pool.sbatch` (main, 32 CPU, ≈400 SU), then to `lofi-embed` (≈250 SU).

| Source | Count |
|---|---|
| Claude-written (220×5) | 1,100 |
| Base-model step-0 samples that pass the gate (220×3, from `lofi-gen` at T 1.0) | 660 |
| Mutations of the kept 10 (15 each) | 150 |
| Broken copies (sour, no groove, boring, fake; 25 each) | 100 |
| Rejects / kept | 20 / 10 |
| **Total** | **2,040**, plus 160 held-out |

**30 drafts → keep 10**
- Three personas each draft one loop for each of 10 briefs, and Shresth rates them in `keep3`.
- Kept loops are labeled ≥Fine, and each kept loop ≻ its rejects.
- "None acceptable" labels all 3 as Toss and triggers a redraft.

**Triage of 300:** stratified as 160 Claude-written, 100 base-model, 25 mutations, 15 broken. It takes about 1.2 h and feeds the acceptability head, per-source quality, τ_PQ and the bands.

**Listening budget:** a round takes about 4·(1920/BPM)·0.75 + 8 ≈ 78 s, so about 45 rounds per hour.

| Phase (1 h/day) | Use | Yield |
|---|---|---|
| 1 (audio from about Nov 5) | keep3 0.3 h; shootouts 1.1 h (keys 6, drums+bass 12, vamps 9, sax realism 6 rounds; humanize 8 A/B); 50-clip check 0.3 h; slack 1 h | palette, gate |
| 2 | triage 1.2 h; bws4 5.5 h ≈ 250 rounds | ≈800 train + ≤200 held-out votes |
| 3 | hack-watch 0.6 h; bws4 6 h | ≈ +1,000 |
| 4 | bws4, hack-watch, 3×30 ablation A/Bs | ≈ +800 |
| 5 | final 115 A/B trials ≈ 1.8 h over 3–4 sittings | the claim |

## 9. Open issues ([rec] = Claude's pick)

1. **Votes by Nov 15 come to about 800 training votes, not ~1,500.** (a) Accept; on-policy votes pass 1,500 by about Nov 20 [rec] (b) Auto-play only bars 1–4 (1.6× throughput) (c) 1.5 h/day from Nov 9 to 15 (d) Use 30 held-out tuples.
2. **What counts as "heard"?** (a) 4 bars, one sax phrase [rec] (b) 2 bars (c) All 8 (d) 2 bars for worst, 4 for best.
3. **Auto-play shape.** (a) Each clip's 8 bars in turn [rec] (b) Bars 1–4 of all four, then bars 5–8 (c) Bars 1–4 of each, then free play.
4. **The seam from bar 8 back to bar 1 is never heard in rounds.** (a) Triage and hack-watch wrap into bars 1–2 [rec] (b) Rounds loop each clip once more (+25 % time) (c) Trust the construction.
5. **Rain bed while voting.** (a) Full mix [rec] (b) Music only, with rain in the final A/B only.
6. **Tier meaning.** (a) Absolute: Toss = dealbreaker, Fine = not embarrassing, Good = I'd play it while working, Great = can't stop nodding [rec] (b) Relative to the day's batch.
7. **Which clip the tag applies to.** (a) Negative → worst, positive → best [rec] (b) One tag per extreme (c) Any clip, via an extra key.
8. **A micro-session catches fewer than half of its ≥2 anchors.** (a) Quarantine it and offer a re-listen [rec] (b) Keep its votes at full weight (c) Half weight.
9. **RM on the audit page.** (a) Aggregate only [rec] (b) Per-clip, after a batch is frozen into a run (c) Never.
10. **Denominator for the 50-clip check.** (a) 50 random samples that pass the gate [rec] (b) 50 random from all 500, with unparseable counted as not fine.
11. **Final A/B response.** (a) Forced choice [rec] (b) Tie key counted as 0.5 (c) Ties excluded.
12. **Floor for the relative gate.** (a) 0.9q and the lower 95 % bound on acc > 0.5 [rec] (b) 0.9q only (c) (acc − 0.5) ≥ 0.8·(q − 0.5).
13. **The gate fails on Nov 15.** (a) Run anyway with taste weight capped at 0.3 and keep voting [rec] (b) Hold GRPO (c) Use a symbolic-only RM.
14. **CLAP contrast scenes.** (a) Other lo-fi scenes: sunny morning, beach, forest, café afternoon, snowy cabin [rec] (b) Other genres (c) Both.
15. **Close-cousin mutations of the kept 10.** (a) 150 (7 %) [rec] (b) 400 (20 %) (c) None.
16. **Spacing of hidden repeats.** (a) ≥2 days [rec] (b) ≥5 days (c) Same day is fine.

### Critical Files for Implementation
- /Users/shresthrana/csci596-project/src/lofi_rl/ui/serve.py (with src/lofi_rl/ui/static/audio.js)
- /Users/shresthrana/csci596-project/src/lofi_rl/taste/rm.py
- /Users/shresthrana/csci596-project/src/lofi_rl/scoring/server.py
- /Users/shresthrana/csci596-project/src/lofi_rl/rl/reward.py
- /Users/shresthrana/csci596-project/src/lofi_rl/taste/select.py