"""The step-0 samples (docs/DECISIONS.md D-021 "What follows" item 2; the build spec docs/design/step0_samples.md; the
rules D-023 registers): the RM held-out tuples, PAR's Ref_b and the seed pool's 660, drawn from the base model under
the grammar with D-021 item 1's streams and draw seeds.

    python -m lofi_rl.step0 generate --run DIR [--deadline-s 960] [--chunk 128] [--hang-s S]     policy env (vLLM)
    python -m lofi_rl.step0 score --run DIR | --from OUT --run REDO                             policy env
    python -m lofi_rl.step0 render --run DIR [--workers 4] [--deadline-s 900] [--cache DIR]     scorer env
                                   | --from OUT --run REDO
    python -m lofi_rl.step0 check RUN_DIR [--redo REDO_DIR] [--json] [--rescore]                laptop
    python -m lofi_rl.step0 prereg RUN_DIR [--redo REDO_DIR] [--write]                          laptop
    python -m lofi_rl.step0 path [--write | --check | --commit REV] [--root DIR]                any; --commit uses git

jobs/step0.sbatch (`lofi-step0`) runs generate, score and render on one A40 (spec §2). Each step is its own module
(generate, score, render, check, prereg, path), the selection rule is lofi_rl.step0.select and the requests are
lofi_rl.step0.requests.

This module holds what the steps share:
- the registered values (spec §1.1, §2.2): the model, prompt v2's template, the frozen palette and its grammar, the
  streams file, each with the full sha256 the tests that pin it give (tests/test_prompt_v2.py,
  tests/test_palette_frozen.py) or, for the streams file, its own sha256; m per set, the rule's name, the probe's and
  the engine's seeds, and MARK, the line D-023 holds and generate checks before any work;
- the answers to the spec's §7, Q1 to Q8: every answer-dependent choice is one named constant under the comment that
  names its question, set to the recommended (a) of each until D-023 records Shresth's, so another answer is an edit
  there (and a new MARK when it changes what is drawn);
- the values the validity list compares with a past run (spec §4.1), each read once from that run's pulled files on
  2026-11-06 and written out here with its source: nothing at check time reads any run but the one checked;
- the schemas and, as data, the fields each summary file holds (spec §2.6), so the steps write and read exactly these.

Standard library only, so every env imports it. Sound claims: none.
"""
from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

# --------------------------------------------------------------------------------------------- the registered values
MODEL = "Qwen/Qwen3-4B-Instruct-2507"                                                # D-019: step 0 has no adapter
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"                          # D-019 (smoke.QWEN's commit)
PROMPT_VERSION = "v2"                                                                # D-020 arm B; D-021 Settings
TEMPLATE_SHA256 = "127e3094fd5a2183fb952059c2221ad018bdef6d062f51a80b14367b8064466e"  # tests/test_prompt_v2.py
PALETTE = "palettes/lofi_rain_v1.yaml"                                               # frozen 2026-11-04 (D-017, D-022)
PALETTE_SHA256 = "09f80a644476abe4d14517d0ec4129d69e1a4bc91eb0935c9ffd152411872183"   # tests/test_palette_frozen.py
REGEX_SHA256 = "5c081828ac50718edc4db82c2a855ea52210debb375280ba01e8442de60f8558"    # the same file: its grammar
STREAMS_FILE = "docs/prereg/brief_streams_v1.json"                                   # D-021 item 1
STREAMS_SHA256 = "bb2daee53e22974c0b09dabede906efa31395a946e0278a42f14872cb158dff7"   # shasum -a 256, 2026-11-06
BACKEND = "xgrammar"                    # junk.STRUCTURED_BACKEND: pinned and read back (spec §2.3)
RULE = "step0-select/1"                 # spec §1.3-1.8 as answered (D-023)
DRAWS = 8                               # draws a brief, k = 0..7 (brief_streams.DRAWS); every one is made (§1.7)
M = MappingProxyType({"heldout": 4, "refb": 4, "pool": 3})   # m per set: taste_pipeline.md:127, :177, :349
SETS: tuple[str, ...] = ("heldout", "refb", "pool")
SET_STREAM = MappingProxyType({"heldout": "rm_heldout", "refb": "train", "pool": "train"})
RUN1 = (0, 400)                         # run 1's train briefs, 596:[0, 400) (DECISIONS.md:433): Ref_b's and the pool's
POOL_N = 220                            # the seed pool's briefs (taste_pipeline.md:329, :349): 220 x 3 = 660
PROBE_SEED = 1                          # the seed probe's seed (spec §2.3; not among the file's 6,800 draw seeds)
PROBE_CELL = ("low", "hopeful")         # its cell: no identical canonical text in 1,326 same-cell pairs of the gate run
PROBE_N = 4                             # identical requests in one call
PROBE_SAME = 2                          # it passes when at least this many of its texts are identical
ENGINE_SEED = 0                         # LLM(seed=...): the smoke's (smoke.py:296); every request carries its own
GPU_MEM, MAX_MODEL_LEN = 0.85, 4096     # the gate run's engine settings (generate.json `engine`)
CHUNK = 128                             # requests per engine call (junk.DEFAULT_CHUNK)
REPLAY_CHUNK = 160                      # the replay's chunks (spec §2.3, item 6)
GENERATE_DEADLINE_S = 960.0             # spec §2.8: no chunk starts that would end after it
RENDER_DEADLINE_S = 900.0               # spec §2.8: no render starts after it
RENDER_WORKERS = 4                      # the job's CPUs, two renders at a time with 2 workers each (junk.render_lanes)
JOB = "jobs/step0.sbatch"
JOB_NAME = "lofi-step0"
DECISIONS = "docs/DECISIONS.md"
PATH_FILE = "docs/prereg/D-023_step0_path.txt"           # spec §4.1, written last by `path --write`
HELDOUT_FILE = "docs/prereg/heldout_v1.json"              # spec §4.4: taste_pipeline.md:128's file
RUN_FILE = "docs/prereg/step0_v1.json"                    # spec §4.4
# D-023's mark, one line naming the rule and the streams drawn: generate checks that the code tree's DECISIONS.md holds
# it and that it is requests.mark_line(STREAMS, RULE), so no job draws under rules or streams not registered first
MARK = ("D-023 mark: rule step0-select/1; streams rm_heldout (seed 5101, all 40 briefs) and train (seed 596, all 400 "
        "briefs); 8 draws a brief, 3520 requests")

# --------------------------------------------------------------------------------------------- the answers (spec §7)
# The recommended answer (a) of each question, until D-023 records Shresth's. Each choice below is the one place its
# answer lives; select.py and requests.py read these names at call time.
#
# Q1. The seed pool's 220 briefs, within run 1's [0, 400): "band" (a), each cell's 25 or 24
#     (brief_streams.per_cell_counts) split by tempo band in the sampler's own shares, by largest remainder, a band
#     [0, 400) cannot fill giving its room to the cell's POOL_FILL_BAND; "cell" (b),
#     brief_streams.stratified_indices(596, 220). (c), a new stream, is not built.
POOL_BRIEFS = "band"
POOL_BANDS = ((60, 74), (75, 90), (91, 95))     # (a): BPM bands around TASTE.md's centre, Claude's default
POOL_FILL_BAND = (75, 90)                       # (a): the band that takes a short band's room
# Q2. May a pool loop also be one of its brief's 4 Ref_b loops? (a) False: the pool walks the draws after Ref_b's 4 and
#     skips Ref_b's texts; Ref_b's spares skip the pool's draws and texts; a pool draw that fails G7 is not replaced.
#     (b) True: the pool takes the brief's first passing draws in k order, Ref_b's among them, and spares skip nothing.
POOL_SHARES_REFB = False
# Q3. Which draws pass: (a) "hard": G1-G6 and compile (junk.score_row's `ok`), then G7 at the set's render, no PQ floor;
#     near-copies and repaired loops pass. (b) "hard_not_copy": a near-copy of the shown example does not.
#     (c) "hard_no_repair": a repaired loop does not.
PASSING = "hard"
# Q4. Two draws of one brief with one canonical text: per set, whether a repeat is skipped. (a) the held-out set skips
#     it, Ref_b keeps it; (b) both skip; (c) neither. The pool always skips one (§1.6).
DISTINCT = MappingProxyType({"heldout": True, "refb": False})
# Q5. A held-out clip's text elsewhere: (a) "text": the pool skips every draw whose canonical text a held-out clip has;
#     (b) and (c) "clip": the pool skips only a draw that is a held-out clip (its twin brief's draw of its text).
POOL_EXCLUDES = "text"
# Q6. What is drawn now: (a) every train brief of [0, 400), (("rm_heldout", "all"), ("train", "all")); (b) only the
#     pool's 220, (("rm_heldout", "all"), ("train", "pool")). A change here needs its MARK (requests.mark_line).
STREAMS: tuple[tuple[str, str], ...] = (("rm_heldout", "all"), ("train", "all"))
# Q7. "Committed before the first vote" (taste_pipeline.md:128): (a) as below; (b) another point he names.
COMMITTED_BEFORE = ("before the first Phase-2 vote (the triage of 300 or the first best/worst round), and before any "
                    "round with a held-out loop is built")
# Q8. A brief short of its m after its 8 draws: (a) "report": it is reported and its set holds one brief fewer (a short
#     Ref_b brief leaves run 1's [0, 400) and goes to Shresth with the lofi-grpo ask); (b) "ask": stop and ask him.
SHORT_RULE = "report"

# --------------------------------------------------------------------------------------------- read from past runs
# D-020: `sampling` equals runs/junk-13632676's. Read from runs/junk-13632676/summary/generate.json `sampling` on the
# laptop, 2026-11-06, in that file's key order; it equals junk.SAMPLING (tests/test_step0_requests.py).
V0_RUN = "runs/junk-13632676"
V0_SAMPLING = MappingProxyType({"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
                                "presence_penalty": 0.0, "frequency_penalty": 0.0, "max_tokens": 512})
# Each cell's prompt length as vLLM counted it, one value per cell: read from runs/junk-13788483/summary/
# completions.jsonl `prompt_tokens` on the laptop, 2026-11-06 (the Phase-1 gate run: prompt v2 on the frozen palette,
# the messages step 0 sends; 500 rows, 9 values, 1,325-1,411). A row with another cell's count is a misaligned output.
GATE_RUN = "runs/junk-13788483"
PROMPT_TOKENS = MappingProxyType({
    "low/melancholy": 1343, "low/wistful": 1382, "low/hopeful": 1325,
    "mid/melancholy": 1393, "mid/wistful": 1369, "mid/hopeful": 1367,
    "high/melancholy": 1401, "high/wistful": 1399, "high/hopeful": 1411})

# --------------------------------------------------------------------------------------------- schemas and fields
GENERATE_SCHEMA = "lofi-rl/step0-generate/1"
STEP0_SCHEMA = "lofi-rl/step0/1"
SETS_SCHEMA = "lofi-rl/step0-sets/1"
CLIPS_SCHEMA = "lofi-rl/step0-clips/1"
HELDOUT_SCHEMA = "lofi-rl/heldout/1"                    # docs/prereg/heldout_v1.json
RUN_SCHEMA = "lofi-rl/step0-run/1"                      # docs/prereg/step0_v1.json
SCHEMAS = MappingProxyType({"generate.json": GENERATE_SCHEMA, "step0.json": STEP0_SCHEMA, "sets.json": SETS_SCHEMA,
                            "clips.json": CLIPS_SCHEMA})
SUMMARY_FILES = ("generate.json", "completions.jsonl", "replay.jsonl", "scored.jsonl", "step0.json", "sets.json",
                 "clips.json")
# junk.score_row's row, in the order it writes it (junk.py:722-750); canon_sha256 only for a loop that parsed
SCORE_ROW_FIELDS = ("i", "energy", "mood", "n_tokens", "finish_reason", "truncated", "canon_sha256", "ok", "gate",
                    "detail", "repairs", "penalized_codes", "p_fix", "copy_jaccard", "near_copy", "copy_reference",
                    "copy_any_jaccard", "copy_any_source", "near_copy_any", "containment", "quasi_copy", "verbatim",
                    "grid_share", "forms", "E", "energy_match", "f_mood")
# junk.render_clip's record by outcome (junk.py:1047-1060), and what the render step adds to each
RENDER_CLIP_FIELDS = MappingProxyType({
    "ok": ("i", "ok", "gate", "clip", "n", "seconds", "lufs", "true_peak_dbtp", "audio_sha256", "canon_sha256", "p_fix",
           "render_s", "flac_bytes"),
    "gate": ("i", "ok", "gate", "detail"),          # a RenderError (G7) or a CompileError ("compile")
    "error": ("i", "ok", "gate", "error"),          # infrastructure, never a gate
    "skipped": ("i", "ok", "gate", "skipped")})     # the render deadline (the step writes it, as junk's render does)
CLIP_ADDED = ("brief_id", "k", "round", "final", "clip_id", "render_cfg_sha8")
# Each summary file's fields (a JSON file's top-level keys; a JSONL file's row keys), in the order the steps write them
FIELDS = MappingProxyType({
    "generate.json": ("schema", "partial", "requested", "generated", "stopped", "streams", "streams_file",
                      "registration", "path_file", "model", "prompt", "palette", "sampling", "per_request_seed",
                      "constrained", "engine", "sampling_params", "probe", "replay", "timing", "host"),
    "completions.jsonl": ("i", "stream", "seed", "index", "brief_id", "k", "draw_seed", "sp_seed", "brief",
                          "prompt_sha256", "prompt_version", "example", "text", "finish_reason", "n_tokens",
                          "prompt_tokens"),
    "replay.jsonl": ("i", "text", "finish_reason", "n_tokens", "same"),
    "scored.jsonl": SCORE_ROW_FIELDS + ("stream", "brief_id", "k", "text_sha256"),
    "step0.json": ("schema", "run", "n", "requested", "partial", "unreadable_lines", "by_stream", "candidates",
                   "render", "render_code", "sets", "score_stamp", "render_stamp"),
    "sets.json": ("schema", "rule", "mark", "render_code", "heldout", "refb", "pool", "heldout_text"),
    "clips.json": ("schema", "render_code", "palette", "jobs", "workers", "clips"),
})
OPTIONAL = MappingProxyType({"generate.json": ("trace",),          # on an error, as junk's generate writes it
                             "scored.jsonl": ("canon_sha256",)})   # a loop that did not parse has none
# The blocks the spec names field by field: "<file>:<dotted key>" ("*" for each key of a mapping) -> its fields.
# generate.json's prompt, constrained and engine are junk's (prompt_record, constrained_block with backend_record,
# VllmEngine.info), step0.json's by_stream is junk.summarize's per stream, and score writes step0.json with
# render, render_code, sets and render_stamp null, which render then fills.
BLOCKS = MappingProxyType({
    "generate.json:streams.*": ("seed", "briefs", "draws"),
    "generate.json:streams_file": ("path", "sha256", "fresh"),
    "generate.json:registration": ("found", "mark", "decisions_sha256"),
    "generate.json:path_file": ("sha256", "blobs_ok"),
    "generate.json:model": ("repo", "revision"),
    "generate.json:palette": ("id", "sha256", "path"),
    "generate.json:probe": ("seed", "cell", "n", "texts_sha256", "distinct", "passed"),
    "generate.json:replay": ("n", "same", "differ", "by_chunk", "stopped"),
    "generate.json:timing": ("load_s", "probe_s", "chunks", "gen_s", "tokens", "tok_per_s", "replay_s", "total_s"),
    "generate.json:host": ("node", "job", "stamp", "gpu"),
    "step0.json:candidates": ("passing", "refb", "heldout_order"),
    "step0.json:render": ("attempted", "rendered", "G7", "other_gates", "errors_final", "errors_retried", "skipped",
                          "rounds"),
    "step0.json:render_code": ("stamp", "source_sha8"),
    "step0.json:sets": ("heldout", "refb", "pool", "heldout_text"),
    "step0.json:sets.heldout": ("briefs", "taken", "short", "undetermined", "near_copy", "repaired", "verbatim"),
    "step0.json:sets.refb": ("briefs", "taken", "short", "undetermined", "near_copy", "repaired", "verbatim"),
    "step0.json:sets.pool": ("briefs", "taken", "short", "undetermined", "near_copy", "repaired", "verbatim"),
    "step0.json:sets.heldout_text.*": ("draws", "refb", "pool"),
    "sets.json:render_code": ("stamp", "source_sha8"),
    "clips.json:render_code": ("stamp", "source_sha8"),
    "clips.json:palette": ("id", "sha256"),
})
# sets.json's entries: one per brief of the set, in stream order. `undetermined`: a draw the walk needs has no decided
# render (an error twice, a deadline skip, or not rendered); before their own renders Ref_b and the pool never are
ENTRY_FIELDS = MappingProxyType({
    "heldout": ("brief_id", "stream", "cell", "bpm", "tonic", "twin_of", "taken", "not_taken", "short", "undetermined"),
    "refb": ("brief_id", "stream", "cell", "bpm", "tonic", "twin_of", "taken", "not_taken", "spares", "short",
             "undetermined"),
    "pool": ("brief_id", "stream", "cell", "bpm", "tonic", "twin_of", "taken", "not_taken", "short", "undetermined")})
TAKEN_FIELDS = ("k", "i", "draw_seed", "text_sha256", "canon_sha256", "clip_id", "audio_sha256")
NOT_TAKEN_FIELDS = ("k", "why")
HELDOUT_TEXT_FIELDS = ("i", "brief_id", "k", "heldout_text")      # sets.json heldout_text: one per train draw
VERBATIM_FIELDS = ("A", "B", "AB")                                 # step0.json sets.<set>.verbatim
# Why a draw was not taken (sets.json not_taken): a gate code (G1-G7, compile, crash), Q3's near_copy or repaired, or
REPEAT = "repeat_of:"               # + k: its canonical text is that of draw k, which the set or a rule before it took
REFB = "refb"                       # (the pool) one of the brief's 4 Ref_b draws
POOL = "pool"                       # (Ref_b's spares) one of the pool's draws
HELDOUT_TEXT = "heldout_text"       # (the pool, Q5 "text") its canonical text is a held-out clip's
HELDOUT_CLIP = "heldout_clip"       # (the pool) it is a held-out clip: a twin brief's draw of that clip's text


class Step0Error(ValueError):
    """Bad input or a refused setting: nothing was drawn, decided or written. `problems` lists every one found."""

    def __init__(self, problems: Iterable[str] | str):
        problems = [problems] if isinstance(problems, str) else [str(p) for p in problems]
        self.problems = problems or ["bad input"]
        super().__init__("\n".join(f"- {p}" for p in self.problems))


def short(sha: str | None, n: int = 8) -> str:
    """A sha as the reports print it (`09f80a64…`); anything else as it is."""
    return f"{sha[:n]}…" if isinstance(sha, str) and len(sha) == 64 else str(sha)
