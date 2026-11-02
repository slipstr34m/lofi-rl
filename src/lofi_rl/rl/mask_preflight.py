"""The masked trainer's preflight, on a CPU node of the policy env (D-021 Part A; docs/GRPO.md §8).

    python -m lofi_rl.rl.mask_preflight --config configs/grpo_smoke_v2_grammar_masked.yaml --out DIR [--texts FILE]...

Before the GPU smoke spends a launch, this runs the parts of the masked trainer the laptop cannot run (torch, TRL,
PEFT and xgrammar are not on the laptop), against the installed packages, and reads four things:
1. **The source checks** train.py makes before the model loads: train.source_checks with the config's shared settings
   (train.shared_settings), and train.mask_refusals for the config's mode.
2. **The mask's two builders**, exactly as train.build_mask builds them for a launch: xgrammar's replay and the NFA,
   over the pinned snapshot's tokenizer, vocabulary size and stop ids. Both are walked along the same ids; at every
   position the two allowed sets must agree (D-021 M8's comparison, offline), and neither may refuse the ids or leave
   the walk unterminated. vLLM samples ids, and a sampled completion is often not the tokenizer's own encoding of its
   text, so the walks cross token boundaries three ways:
   - the real tokenizer's ids (plus the EOS) of every text that fullmatches the regex: prompt v2's 9 references and
     the completions in each --texts file (JSONL rows with a `text` field). The job passes D-020's three (junk
     13640325's completions.jsonl, smoke 13657200's a and b rewards.jsonl): every file must be read, and at least
     MIN_TEXTS of their texts must be in the language;
   - the same texts one character per token (`char_level`): each character's one-character byte-level token, plus
     the EOS, so that every character boundary is a token boundary;
   - seeded random walks (`random_walks`), each drawn uniformly from one builder's allowed sets until a stop id or
     WALK_MAX_LEN ids: RANDOM_WALKS["nfa"] from the NFA's (GrammarMask.random_walk), RANDOM_WALKS["xgrammar"] from
     xgrammar's own matcher. A walk cut at WALK_MAX_LEN is counted apart, not as unterminated.
3. **TRL's support path on a tiny random Qwen3** (CPU, fp32): TRL's own `_get_per_token_logps_and_entropies` on a
   PEFT-wrapped model with `add_fused_lm_head` on its base model, as GRPOTrainer adds it (grpo_trainer.py:1113-1120),
   given the support masked.build_support makes. The vocabulary (70000) is wider than TRL's 32768-wide vocabulary tile
   (_CHUNKED_LOGPROB_CHUNK_SIZE, trl/trainer/utils.py:1523), so the kernel folds three tiles, and the supports reach
   into the last one (its top two ids stand in for the stop ids). The masked log-probs must agree, to TOL, with
   masked.TorchOps's fill form and with a float64 masked log-softmax of the model's own logits; the unmasked ones with
   the fill form's; the gradient of the masked log-probs' sum with the fill form's; and a support shifted one column
   must not agree (the alignment, on the installed TRL). The CPU runs the kernel's torch fallback and TRL's
   non-compact packing; the GPU runs its Triton kernel and the compact packing (`support_ids[mask]`, utils.py:1683-
   1697). One more call, with trl.trainer.utils._use_triton patched to return True, runs the compact packing on the
   CPU. That name is the one utils.py imports (:61) and reads only for `compact` (:1683); the kernel's fold choice
   (trl/kernels/chunked_logprob.py:563 forward, :700 backward) reads the kernel module's own _use_triton, so it stays
   the torch fallback. The compact call's log-probs, entropies and gradient must equal the non-compact call's.
4. **The masked trainer's own glue on real objects** (`glue`): masked.adapter_state_of on the PEFT model, outside and
   inside its disable_adapter(); trl_probe.lm_head_record (T5's run-time half) on it; and one step of
   masked.MaskedPasses with masked.TorchOps through TRL's real method, called as the subclass calls it: begin,
   check_generation on a 9-tuple shaped like _generate's, the old pass under no_grad, the reference pass under TRL's
   use_adapter(model, None), two loss micro-batches with grad, then after_score on TRL's output fields. The pass
   labels and adapter states, the rows, -log Z, the log-probs (against the float64 masked log-softmax, adapters on
   and off), the self-check and the IS records must be the ones the trainer is written to produce.
It writes <out>/preflight.json and prints MASK_PREFLIGHT_OK, or MASK_PREFLIGHT_FAILED <reasons>, last. It decides
nothing D-021 registered: the gates are read from the smoke. Exit 0 iff OK.
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import random
import re
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

from lofi_rl.grammar_mask import BYTE_TO_CHAR, Walk
from lofi_rl.palette import load_palette
from lofi_rl.rl import train
from lofi_rl.rl.config import ConfigError, load_config, resolve

TOL = 1e-4                  # fp32 on CPU: TRL's kernel, the fill form and float64 agree far closer than this
SHIFTED_MIN = 0.05          # a support one column off must miss by more than this (D-021's 0.05 bar, for scale)
MAX_EXAMPLES = 5
MIN_TEXTS = 980             # D-020's completions: junk 13640325's 500 and smoke 13657200's 320 + 160, all of them in
                            # the language (tests/test_grammar_mask.py's M2 walks them on the laptop)
RANDOM_WALKS = MappingProxyType({"nfa": 300, "xgrammar": 100})   # seeded random walks, by the builder that draws them
WALK_MAX_LEN = 512
WALK_SEED = 0               # the builder d draws with random.Random(f"{WALK_SEED}:{d}")
TINY = {"vocab_size": 70000, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 256,
        "tie_word_embeddings": True}          # Qwen3-4B ties its embeddings too (config.json)
TINY_STOPS = (TINY["vocab_size"] - 2, TINY["vocab_size"] - 1)    # the top two ids (the last tile): the stop ids' part
TRL_VOCAB_TILE = 32768      # TRL 1.17.1's _CHUNKED_LOGPROB_CHUNK_SIZE (utils.py:1523); the check reads the installed
                            # one and records it
# The tiny generation (_batch): prompt and completion lengths, which completions end on a stop id (row 2 is truncated
# at C, so its completion mask is all zeros), its one forced position, and the loss pass's micro-batches (TRL
# shuffles a generation's rows and splits them, grpo_trainer.py:1612-1613)
P_LENS, C_LENS, STOPS_AT_END = (6, 3, 5, 4), (9, 5, 9, 7), (True, True, False, True)
FORCED_AT = (1, 2)
LOSS_MICRO = ((3, 0), (2, 1))
GLUE_REGEX = "x"            # the glue's replay is a stand-in over the batch's own sets: MaskSetup reads only its sha
# what TRL passes besides the leading arguments: the old and reference passes (grpo_trainer.py:2762-2772, :2846-2856;
# batch_size is the per-device batch, here 2 so the rows go in two chunks), and _compute_loss (:3151-3168), text only
OLD_KWARGS = MappingProxyType({"batch_size": 2, "num_images": None, "num_tiles": None})
LOSS_KWARGS = MappingProxyType({"compute_entropy": True, "compute_aux_loss": False, "pixel_values": None,
                                "image_grid_thw": None, "num_images": None, "pixel_attention_mask": None,
                                "spatial_shapes": None, "num_tiles": None, "image_sizes": None, "token_type_ids": None,
                                "mm_token_type_ids": None, "image_position_ids": None})
AFTER_SCORE_KEYS = ("mask/violations", "mask/neglogz", "mask/forced_share", "mask/nfa_mismatch", "is/seq_weight_mean",
                    "is/seq_weight_max_share")


# ------------------------------------------------------------------------------------------ 2. the two builders
def read_texts(paths: list[Path], problems: list[str],
               files: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    """(source, text) for every JSONL row of `paths` with a non-empty string `text`; a source is "<path>:<line>".
    Blank lines are skipped. `files` gets each path's record: the lines read, the texts kept, and the error that kept
    the file from being read (also added to `problems`)."""
    out = []
    for p in paths:
        rec: dict[str, Any] = {"lines": 0, "texts": 0, "error": None}
        if files is not None:
            files[str(p)] = rec
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except (OSError, ValueError) as err:          # ValueError: not UTF-8
            rec["error"] = f"{type(err).__name__}: {err}"[:300]
            problems.append(f"{p}: {err}")
            continue
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            rec["lines"] += 1
            try:
                row = json.loads(line)
            except ValueError:
                problems.append(f"{p}:{n}: not JSON")
                continue
            text = row.get("text") if isinstance(row, dict) else None
            if isinstance(text, str) and text:
                out.append((f"{p}:{n}", text))
                rec["texts"] += 1
    return out


def one_char_ids(vocab: Mapping[str, int]) -> dict[str, int]:
    """Each ASCII character's one-character token in a byte-level vocabulary (get_vocab(): token -> id): the token
    spelled by the character's byte in GPT-2's table (grammar_mask.BYTE_TO_CHAR, the one xgrammar reads the vocabulary
    with). A character with no such token is absent. The regex is ASCII, so an in-language text is too."""
    out: dict[str, int] = {}
    for b in range(128):
        tid = vocab.get(BYTE_TO_CHAR[b])
        if isinstance(tid, int) and not isinstance(tid, bool):
            out[chr(b)] = tid
    return out


def char_ids(text: str, one_char: Mapping[str, int]) -> list[int] | None:
    """The text one character per token (one_char_ids), or None when a character has no one-character token."""
    ids = [one_char.get(c) for c in text]
    return None if None in ids else ids                     # type: ignore[return-value]


def _tally() -> dict[str, Any]:
    return {"positions": 0, "mismatch": 0, "examples": [], "refused": {"xgrammar": 0, "nfa": 0},
            "not_terminated": {"xgrammar": 0, "nfa": 0}, "refused_examples": [],
            "walk_s": {"xgrammar": 0.0, "nfa": 0.0}}


def _finish(t: dict[str, Any]) -> dict[str, Any]:
    t["walk_s"] = {k: round(v, 3) for k, v in t["walk_s"].items()}
    if t["positions"]:
        t["us_per_position"] = {k: round(1e6 * v / t["positions"], 2) for k, v in t["walk_s"].items()}
    return t


def _walk_pair(setup: Any, ids: list[int], src: str, t: dict[str, Any], cut_at: int | None = None) -> None:
    """Walk `ids` through xgrammar's replay and the NFA, and count what masked.compare_walks finds into the tally `t`:
    the positions compared, the differences (MAX_EXAMPLES of them kept), and per builder the walks it refused or left
    unterminated. With cut_at, a walk of cut_at ids that a builder neither refused nor terminated goes to
    t["cut_at_max_len"] instead: a random walk stopped at its bound."""
    from lofi_rl.rl import masked

    t0 = time.perf_counter()
    a = setup.replay.walk(ids)
    t1 = time.perf_counter()
    b = setup.nfa.walk(ids)
    t["walk_s"]["xgrammar"] += t1 - t0
    t["walk_s"]["nfa"] += time.perf_counter() - t1
    n, diffs = masked.compare_walks(a, b, ids)
    t["positions"] += n
    t["mismatch"] += len(diffs)
    t["examples"] += [{"text": src, **d} for d in diffs][:max(MAX_EXAMPLES - len(t["examples"]), 0)]
    for name, w in (("xgrammar", a), ("nfa", b)):
        if w.violation is not None:
            t["refused"][name] += 1
            if len(t["refused_examples"]) < MAX_EXAMPLES:
                t["refused_examples"].append({"text": src, "by": name, "pos": w.violation,
                                              "token": ids[w.violation] if w.violation < len(ids) else None})
        elif not w.terminated:
            cut = cut_at is not None and len(ids) >= cut_at
            t["cut_at_max_len" if cut else "not_terminated"][name] += 1


def _xgrammar_draw(replay: Any, rng: random.Random, max_len: int) -> tuple[list[int], str]:
    """One random walk drawn from xgrammar's own allowed sets: the replay's matcher from its start, a uniform draw from
    the bitmask's set at each position, then accept_token, until the matcher terminates, refuses a token its own
    bitmask allowed, allows nothing, or max_len ids are drawn. Returns (ids, how it ended). This reads the replay's
    private `_matcher` and `_allowed_now()`, the bitmask decode walk() uses: acceptable in this diagnostic only (the
    trainer replays sampled ids through walk(), which resets the matcher itself)."""
    m = replay._matcher
    m.reset()
    ids: list[int] = []
    end = "max_len"
    while len(ids) < max_len:
        allowed = replay._allowed_now()
        if not allowed:
            end = "dead_end"
            break
        tok = rng.choice(allowed)
        ids.append(tok)
        if not m.accept_token(tok):
            end = "accept_refused"
            break
        if m.is_terminated():
            end = "stop"
            break
    m.reset()
    return ids, end


def random_walks(setup: Any, walks: Mapping[str, int] = RANDOM_WALKS, max_len: int = WALK_MAX_LEN,
                 seed: int = WALK_SEED) -> dict[str, Any]:
    """walks[d] seeded random walks drawn from builder d's allowed sets (module docstring, 2), each then walked
    through both builders and compared like a text. `drawn` says, per builder, how many walks it drew, their ids, how
    they ended (stop, max_len, dead_end, or accept_refused: xgrammar refused a token its own bitmask allowed) and the
    seconds; a walk that reached max_len unterminated is in `cut_at_max_len`, not in `not_terminated`."""
    out: dict[str, Any] = {"walks": dict(walks), "max_len": max_len, "seed": seed, "drawn": {}, **_tally(),
                           "cut_at_max_len": {"xgrammar": 0, "nfa": 0}}
    stops = set(setup.stop_ids)
    for driver, n in walks.items():
        rng = random.Random(f"{seed}:{driver}")
        ends: Counter[str] = Counter()
        drawn = 0
        t0 = time.perf_counter()
        for i in range(n):
            if driver == "nfa":
                ids = setup.nfa.random_walk(rng, max_len=max_len)
                end = "stop" if ids and ids[-1] in stops else "max_len" if len(ids) >= max_len else "dead_end"
            elif driver == "xgrammar":
                ids, end = _xgrammar_draw(setup.replay, rng, max_len)
            else:
                raise ValueError(f"no random-walk builder {driver!r} (nfa, xgrammar)")
            ends[end] += 1
            drawn += len(ids)
            _walk_pair(setup, ids, f"random:{driver}:{i}", out, cut_at=max_len)
        out["drawn"][driver] = {"walks": n, "ids": drawn, "ends": dict(ends), "s": round(time.perf_counter() - t0, 3)}
    return _finish(out)


def check_masks(setup: Any, tok: Any, texts: list[tuple[str, str]], *, walks: Mapping[str, int] = RANDOM_WALKS,
                max_len: int = WALK_MAX_LEN, seed: int = WALK_SEED) -> dict[str, Any]:
    """xgrammar's replay and the NFA walked along the same ids and compared position by position (module docstring,
    2): each in-language text's tokenizer ids + EOS (the top-level counts), its characters one token each + EOS
    (`char_level`), and the random walks (`random_walks`; an error there is recorded, and fails the verdict). A text's
    source is "<file or prompt_v2>:<line or id>"; `in_language_by_source` counts each source's in-language texts."""
    rx = re.compile(setup.regex)
    eos = int(tok.eos_token_id)
    one_char = one_char_ids(tok.get_vocab())
    out: dict[str, Any] = {"texts": len(texts), "in_language": 0, "in_language_by_source": {}, "eos": eos,
                           **_tally()}
    char: dict[str, Any] = {"texts": 0, "one_char_tokens": len(one_char), "no_token": 0, "no_token_examples": [],
                            **_tally()}
    for src, text in texts:
        if not rx.fullmatch(text):
            continue
        out["in_language"] += 1
        source = src.rsplit(":", 1)[0]
        out["in_language_by_source"][source] = out["in_language_by_source"].get(source, 0) + 1
        _walk_pair(setup, list(tok(text, add_special_tokens=False)["input_ids"]) + [eos], src, out)
        ids = char_ids(text, one_char)
        if ids is None:
            char["no_token"] += 1
            if len(char["no_token_examples"]) < MAX_EXAMPLES:
                char["no_token_examples"].append({"text": src, "char": next(c for c in text if c not in one_char)})
            continue
        char["texts"] += 1
        _walk_pair(setup, ids + [eos], src, char)
    out["char_level"] = _finish(char)
    try:
        out["random_walks"] = random_walks(setup, walks, max_len, seed)
    except Exception as err:  # noqa: BLE001: recorded; the verdict fails on it
        out["random_walks"] = {"error": f"{type(err).__name__}: {err}"[:800], "trace": traceback.format_exc()[-3000:]}
    return _finish(out)


# ------------------------------------------------------------------------------------------ 3-4. the tiny model
def _batch(rng: random.Random, vocab: int, stops: tuple[int, ...]) -> dict[str, Any]:
    """One generation in TRL's layout (grpo_trainer.py:2546-2617), as plain lists: prompts left-padded, completions
    right-padded with 0, the completion mask ones over each completion's ids, and all zeros for a truncated one, whose
    ids stay in place (mask_truncated_completions, :2603-2611). Each sampled token gets a random allowed set holding
    it, drawn from the whole vocabulary so that the supports span the kernel's vocabulary tiles. A completion that
    stops ends on one of `stops` (the top ids: the last tile); the stop ids are allowed together at its end and at
    every column c % 3 == 1 (where a grammar might accept). FORCED_AT's set is its token alone.
    Returns prompts, sampled (each completion's ids, as _generate returns them), allowed (a set per sampled id), ids
    and att (the (B, P + C) rows), comp (the completion columns), valid (each row's valid tokens), P and C."""
    P, C = max(P_LENS), max(C_LENS)
    first_stop = min(stops)
    out: dict[str, Any] = {"prompts": [], "sampled": [], "allowed": [], "ids": [], "att": [], "comp": [], "valid": [],
                           "P": P, "C": C}
    for b, (pl, cl, ends) in enumerate(zip(P_LENS, C_LENS, STOPS_AT_END, strict=True)):
        prompt = [rng.randrange(1, vocab) for _ in range(pl)]
        done = [rng.randrange(1, first_stop) for _ in range(cl - ends)] + ([rng.choice(stops)] if ends else [])
        sets = []
        for c, t in enumerate(done):
            s = {t}
            if (b, c) != FORCED_AT:
                s |= {rng.randrange(1, vocab) for _ in range(rng.randrange(1, 12))}
                if c % 3 == 1 or (ends and c == cl - 1):
                    s |= set(stops)
            sets.append(tuple(sorted(s)))
        n = cl if ends else 0
        out["prompts"].append(prompt)
        out["sampled"].append(done)
        out["allowed"].append(sets)
        out["ids"].append([0] * (P - pl) + prompt + done + [0] * (C - cl))
        out["att"].append([0] * (P - pl) + [1] * pl + [1] * n + [0] * (C - n))
        out["comp"].append(done + [0] * (C - cl))
        out["valid"].append(n)
    return out


def tile_coverage(batch: Mapping[str, Any], tile: int, vocab: int) -> dict[str, int]:
    """Which of the kernel's vocabulary tiles the batch's valid positions reach: how many tiles, the positions whose
    support holds an id of the last tile, the targets in it, the supports spanning more than one tile (the fold across
    tiles), and the forced positions (a support of one id: the kernel's branch with no other token)."""
    tiles = -(-vocab // tile)
    on = [(b, c) for b, n in enumerate(batch["valid"]) for c in range(n)]
    spans = [{t // tile for t in batch["allowed"][b][c]} for b, c in on]
    return {"vocab_tile": tile, "vocab_tiles": tiles,
            "last_tile_support_positions": sum(tiles - 1 in s for s in spans),
            "last_tile_targets": sum(batch["sampled"][b][c] // tile == tiles - 1 for b, c in on),
            "multi_tile_positions": sum(len(s) > 1 for s in spans),
            "forced_positions": sum(len(batch["allowed"][b][c]) == 1 for b, c in on)}


class SetsReplay:
    """The glue's stand-in for masked.XgrammarReplay: walk(ids) returns the allowed sets the tiny batch drew for those
    sampled ids (a grammar_mask.Walk, terminated when they end on a stop id), and refuses at position 0 any ids it was
    not given, so a lookup the trainer should not make shows as a violation rather than as a guess."""

    def __init__(self, regex: str, vocab_size: int, stop_ids: Sequence[int],
                 walks: Mapping[tuple[int, ...], Sequence[Sequence[int]]]) -> None:
        from lofi_rl.rl import masked

        self.regex_sha256 = masked.regex_sha256(regex)
        self.vocab_size = int(vocab_size)
        self.stop_ids = tuple(sorted({int(s) for s in stop_ids}))
        self._walks = {tuple(int(t) for t in ids): tuple(tuple(int(t) for t in a) for a in sets)
                       for ids, sets in walks.items()}
        self.record: dict[str, Any] = {"stand_in": "the tiny batch's allowed sets", "walks": 0, "unknown": 0}

    def walk(self, ids: Sequence[int]) -> Walk:
        key = tuple(int(t) for t in ids)
        self.record["walks"] += 1
        sets = self._walks.get(key)
        if sets is None or len(sets) != len(key):
            self.record["unknown"] += 1
            return Walk(allowed=((),), violation=0, terminated=False)
        return Walk(allowed=sets, violation=None, terminated=bool(key) and key[-1] in self.stop_ids)


def glue_setup(batch: Mapping[str, Any]) -> Any:
    """The glue's masked.MaskSetup (mode grammar_masked: path support_ids) over the batch's stand-in replay."""
    from lofi_rl.rl import masked

    replay = SetsReplay(GLUE_REGEX, TINY["vocab_size"], TINY_STOPS,
                        {tuple(s): a for s, a in zip(batch["sampled"], batch["allowed"], strict=True)})
    return masked.MaskSetup(mode="grammar_masked", regex=GLUE_REGEX, regex_sha256=masked.regex_sha256(GLUE_REGEX),
                            vocab_size=TINY["vocab_size"], stop_ids=TINY_STOPS, replay=replay, nfa=None, stats={})


def generation_result(batch: Mapping[str, Any], sampling: Sequence[Sequence[float]]) -> tuple:
    """A 9-tuple shaped like TRL 1.17.1's _generate result (grpo_trainer.py:2416-2427): prompt ids, the sampled
    completion ids (a stop completion ends with its stop id, a truncated one does not), no tool mask, the texts,
    vLLM's log-probs, TRL's own sampling mask (None at the registered settings), extra fields, images, tool images."""
    return ([list(p) for p in batch["prompts"]], [list(s) for s in batch["sampled"]], None,
            [""] * len(batch["sampled"]), [list(x) for x in sampling], None, {}, None, [])


def tiny_model(seed: int = 0) -> SimpleNamespace:
    """What checks 3 and 4 share: the tiny PEFT-wrapped Qwen3 with TRL's fused head, its generation (_batch), the
    support masked.build_support makes for it, TRL's method past its profiling wrapper, a trainer stand-in, and float64
    references from the model's own logits at each valid position: the masked log-softmax with the adapters on
    (`direct`) and off (`direct_ref`), and the token-weighted -log Z."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from trl import GRPOTrainer
    from trl.trainer.utils import add_fused_lm_head

    from lofi_rl.rl import masked

    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY)).float()
    peft = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear",
                                            bias="none", task_type="CAUSAL_LM"))
    with torch.no_grad():                       # LoRA's B starts at zero: move it, so the adapter path counts
        for name, p in peft.named_parameters():
            if "lora_B" in name:
                p.normal_(0.0, 0.02)
    add_fused_lm_head(peft.get_base_model(), temperature=1.0, outputs=("log_probs", "entropy"))
    peft.eval()
    batch = _batch(rng, TINY["vocab_size"], TINY_STOPS)
    P, C = batch["P"], batch["C"]
    valid = [masked.valid_length(r) for r in (row[P:] for row in batch["att"])]
    if valid != batch["valid"]:
        raise RuntimeError(f"masked.valid_length reads {valid} valid tokens, the batch was built with {batch['valid']}")
    cache = masked.AllowedCache(replay=None)
    for row, n, sets in zip(batch["comp"], valid, batch["allowed"], strict=True):
        if n:
            cache.put(row[:n], sets[:n])
    build = masked.build_support([row[P:] for row in batch["ids"]], valid, cache)
    if build.violations or build.cache_miss:
        raise RuntimeError(f"build_support: violations {build.violations}, misses {build.cache_miss}")
    fn = GRPOTrainer._get_per_token_logps_and_entropies
    fn = getattr(fn, "__wrapped__", fn)          # past TRL's profiling_decorator (extras/profiling.py:205)
    trainer = SimpleNamespace(accelerator=SimpleNamespace(autocast=contextlib.nullcontext, is_main_process=True,
                                                          unwrap_model=lambda m: m),
                              temperature=1.0, state=SimpleNamespace(global_step=0))
    ids, att = torch.tensor(batch["ids"]), torch.tensor(batch["att"])
    on = [(b, c) for b, n in enumerate(valid) for c in range(n)]
    with torch.no_grad():
        logits = peft(input_ids=ids, attention_mask=att).logits.double()   # column P + c - 1 predicts token c
        with peft.disable_adapter():
            logits_ref = peft(input_ids=ids, attention_mask=att).logits.double()
    direct = torch.zeros(len(valid), C, dtype=torch.float64)
    direct_ref = torch.zeros(len(valid), C, dtype=torch.float64)
    neglogz = []
    for b, c in on:
        keep = torch.tensor(batch["allowed"][b][c])
        for out, z in ((direct, logits[b, P + c - 1]), (direct_ref, logits_ref[b, P + c - 1])):
            out[b, c] = z[batch["comp"][b][c]] - torch.logsumexp(z[keep], dim=0)
        z = logits[b, P + c - 1]
        neglogz.append(float(torch.logsumexp(z, dim=0) - torch.logsumexp(z[keep], dim=0)))
    return SimpleNamespace(torch=torch, peft=peft, batch=batch, P=P, C=C, valid=valid, on=on, build=build,
                           support=torch.tensor(build.support, dtype=torch.long), fn=fn, trainer=trainer, ids=ids,
                           att=att, direct=direct, direct_ref=direct_ref, neglogz_direct=sum(neglogz) / len(neglogz))


def check_support_path(tiny: SimpleNamespace) -> dict[str, Any]:
    """Module docstring, 3: TRL's support path against the fill form and float64, its gradient, a shifted support,
    the vocabulary tiles the batch reaches, and the compact packing (an error in the compact call is recorded under
    compact_error, and fails the verdict)."""
    from unittest import mock

    from trl.kernels import chunked_logprob as trl_kernel
    from trl.trainer import utils as trl_utils

    from lofi_rl.rl import masked

    torch, peft, fn, trainer = tiny.torch, tiny.peft, tiny.fn, tiny.trainer
    ids, att, C, on, support, direct = tiny.ids, tiny.att, tiny.C, tiny.on, tiny.support, tiny.direct
    shifted = torch.roll(support, shifts=1, dims=1)
    ops = masked.TorchOps()

    def gap(x: Any, y: Any) -> float:
        return max(abs(float(x[b, c]) - float(y[b, c])) for b, c in on)

    with torch.no_grad():
        lp_s, ent_s, _ = fn(trainer, peft, ids, att, C, batch_size=2, compute_entropy=True, sampling_mask=support)
        lp_u, _, _ = fn(trainer, peft, ids, att, C, batch_size=2, sampling_mask=None)
        lp_x, _, _ = fn(trainer, peft, ids, att, C, batch_size=2, sampling_mask=shifted)
        lp_f, ent_f, lp_fu = ops.fill_logps(trainer, peft, ids, att, C, support, batch_size=2, compute_entropy=True,
                                            want_unmasked=True)

    def grads(path: str) -> list[Any]:
        peft.zero_grad(set_to_none=True)
        if path == "support_ids":
            lp = fn(trainer, peft, ids, att, C, batch_size=2, compute_entropy=False, sampling_mask=support)[0]
        else:
            lp = ops.fill_logps(trainer, peft, ids, att, C, support, batch_size=2, compute_entropy=False,
                                want_unmasked=False)[0]
        sum(lp[b, c] for b, c in on).backward()
        return [p.grad.detach().clone() for n, p in peft.named_parameters() if p.requires_grad and p.grad is not None]

    def grad_gap(xs: list[Any], ys: list[Any]) -> float | None:
        return max((float((a - b).abs().max()) for a, b in zip(xs, ys)), default=None) if len(xs) == len(ys) else None

    with torch.enable_grad():
        g_s, g_f = grads("support_ids"), grads("fill")
    chunk = getattr(trl_utils, "_CHUNKED_LOGPROB_CHUNK_SIZE", None)         # the tile width the fused forward passes
    tiles = (tile_coverage(tiny.batch, chunk, TINY["vocab_size"]) if isinstance(chunk, int)
             else {"vocab_tile": chunk, "vocab_tiles": None})
    out = {"tiny": TINY, "stops": list(TINY_STOPS), "rows": len(tiny.valid), "P": tiny.P, "C": C, "valid": tiny.valid,
           "positions": len(on), "k_max": tiny.build.k_max, "forced": tiny.build.forced, **tiles,
           "masked_support_vs_direct": gap(lp_s, direct), "masked_fill_vs_direct": gap(lp_f, direct),
           "masked_support_vs_fill": gap(lp_s, lp_f), "unmasked_support_vs_fill": gap(lp_u, lp_fu),
           "entropy_support_vs_fill": gap(ent_s, ent_f), "shifted_vs_direct": gap(lp_x, direct),
           "grad_tensors": len(g_s), "grad_tensors_fill": len(g_f), "grad_max_abs_diff": grad_gap(g_s, g_f),
           "grad_max_abs": max((float(a.abs().max()) for a in g_s), default=None),
           "kernel": "torch fallback (no Triton on CPU): chunked_logprob.py:563 and :700 read the kernel module's own "
                     "_use_triton",
           "packing": "non-compact (the else branches of utils.py:1684-1697); compact_*: the compact packing (their "
                      "`[mask]` branches), with trl.trainer.utils._use_triton patched to return True",
           "tol": TOL, "shifted_min": SHIFTED_MIN}
    out["invalid_rows_zero"] = all(float(lp_s[b].abs().max()) == 0.0 for b, n in enumerate(tiny.valid) if n == 0)

    # TRL's compact packing, the one the GPU takes: utils.py's own `compact = _use_triton(device)` made True
    calls: list[str] = []

    def use_triton(device: Any) -> bool:
        calls.append(str(device))
        return True

    out.update(compact_vs_noncompact=None, compact_grad_tensors=None, compact_grad_max_abs_diff=None,
               compact_kernel_triton=None, compact_error=None)
    try:
        with mock.patch.object(trl_utils, "_use_triton", use_triton):
            out["compact_kernel_triton"] = bool(trl_kernel._use_triton(ids.device))   # the fold's own choice
            with torch.no_grad():
                lp_c, ent_c, _ = fn(trainer, peft, ids, att, C, batch_size=2, compute_entropy=True,
                                    sampling_mask=support)
            with torch.enable_grad():
                g_c = grads("support_ids")
        # every column, the invalid rows' zeros too: a misplaced scatter shows anywhere
        out["compact_vs_noncompact"] = max(float((lp_c - lp_s).abs().max()), float((ent_c - ent_s).abs().max()))
        out["compact_grad_tensors"] = len(g_c)
        out["compact_grad_max_abs_diff"] = grad_gap(g_c, g_s)
    except Exception as err:  # noqa: BLE001: recorded; the verdict fails on it
        out["compact_error"] = f"{type(err).__name__}: {err}"[:800]
        out["compact_trace"] = traceback.format_exc()[-3000:]
    out["compact_calls"] = len(calls)
    return out


def check_glue(tiny: SimpleNamespace) -> dict[str, Any]:
    """Module docstring, 4: records what each piece of the masked trainer's glue did on the real objects (verdict()
    holds what it must be). An exception in the step is recorded under `error`, with its trace, and fails."""
    from lofi_rl.rl import masked, trl_probe

    peft = tiny.peft
    out: dict[str, Any] = {"error": None}
    before = masked.adapter_state_of(peft)
    with peft.disable_adapter():
        inside = masked.adapter_state_of(peft)
    out["adapter_state"] = {"before": before, "inside_disable_adapter": inside, "after": masked.adapter_state_of(peft)}
    lm = trl_probe.lm_head_record(peft)
    out["lm_head"] = {k: lm.get(k) for k in ("verdict", "output_embedding", "output_embedding_is_tuner_layer",
                                             "tuner_layers", "missing", "error")}
    try:
        out.update(_glue_step(tiny))
    except Exception as err:  # noqa: BLE001: recorded; the verdict fails on it
        out["error"] = f"{type(err).__name__}: {err}"[:800]
        out["trace"] = traceback.format_exc()[-6000:]
    return out


def _glue_step(tiny: SimpleNamespace) -> dict[str, Any]:
    """One generation's cycle of masked.MaskedPasses with masked.TorchOps, as LofiGRPOTrainer drives it
    (masked.make_trainer_class), through TRL's real _get_per_token_logps_and_entropies."""
    import functools

    from trl import GRPOTrainer
    from trl.trainer.utils import use_adapter

    from lofi_rl.rl import masked

    torch, peft, trainer, batch = tiny.torch, tiny.peft, tiny.trainer, tiny.batch
    ids, att, P, C, valid, direct = tiny.ids, tiny.att, tiny.P, tiny.C, tiny.valid, tiny.direct
    rows = len(valid)
    setup = glue_setup(batch)
    ctl = masked.MaskedPasses(setup, masked.TorchOps())
    signature = inspect.signature(GRPOTrainer._get_per_token_logps_and_entropies)   # as make_trainer_class reads it
    base = functools.partial(tiny.fn, trainer)    # the subclass hands logps super()'s bound method; here TRL's
                                                  # function past its profiling wrapper, bound to the stand-in
    # vLLM's sampling log-probs: here the float64 masked ones (the truncated row's are -1, under its zero mask)
    sampling = [[float(direct[r, c]) if c < valid[r] else -1.0 for c in range(len(batch["sampled"][r]))]
                for r in range(rows)]
    ctl.begin(0)
    ctl.check_generation(generation_result(batch, sampling), trainer)
    with torch.no_grad():
        old = ctl.logps(trainer, base, signature, peft, ids, att, C, (), dict(OLD_KWARGS), None)
        with use_adapter(peft, adapter_name="ref" if "ref" in peft.peft_config else None):   # :2840-2856
            ref = ctl.logps(trainer, base, signature, peft, ids, att, C, (), dict(OLD_KWARGS), None)
    peft.zero_grad(set_to_none=True)
    losses = []
    with torch.enable_grad():
        for micro in LOSS_MICRO:
            r = list(micro)
            losses.append((r, ctl.logps(trainer, base, signature, peft, ids[r], att[r], C, (), dict(LOSS_KWARGS),
                                        None)))
        sum(lp[i, c] for r, (lp, _, _) in losses for i, row in enumerate(r) for c in range(valid[row])).backward()
    trainable = [p for p in peft.parameters() if p.requires_grad]
    # TRL's output fields after the passes (:3017-3035), the sequence-level ratio formed as TRL forms it (:2776-2791)
    cm = att[:, P:]
    samp = torch.tensor([s + [0.0] * (C - len(s)) for s in sampling], dtype=torch.float32)
    ratio = torch.exp(torch.nan_to_num((old[0] - samp) * cm, nan=0.0).sum(dim=-1, keepdim=True))
    scalars = ctl.after_score({"completion_mask": cm, "importance_sampling_ratio": ratio,
                               "old_per_token_logps": old[0], "sampling_per_token_logps": samp})
    rec = setup.stats[0]
    ratio_l = [float(x) for x in ratio[:, 0]]

    def gap(calls: Sequence[tuple[Sequence[int], Any]], want: Any) -> float | None:
        """max |log-prob - float64| over the valid positions of (rows, log-probs) calls: row i of a call is batch
        row rows[i]."""
        return max((abs(float(lp[i, c]) - float(want[row, c])) for which, lp in calls for i, row in enumerate(which)
                    for c in range(valid[row])), default=None)

    weights, raw = rec["seq_weights"] or [], rec["seq_weights_raw"] or []
    return {
        "rows_in_batch": rows, "valid": valid, "loss_micro_batches": [list(m) for m in LOSS_MICRO],
        "completions": rec["completions"], "terminated": rec["terminated"], "violations": rec["violations"],
        "trl_sampling_mask": rec["trl_sampling_mask"], "cache_miss": rec["cache_miss"],
        "replay_unknown": setup.replay.record["unknown"],
        "passes": [{k: p[k] for k in ("label", "adapters", "grad", "rows", "rows_scored", "positions")}
                   for p in rec["passes"]],
        "rows": dict(rec["rows"]), "rows_scored": dict(rec["rows_scored"]), "tokens": rec["tokens"],
        "neglogz": rec["neglogz"], "neglogz_tokens": rec["neglogz_tokens"],
        "neglogz_nonfinite": rec["neglogz_nonfinite"], "neglogz_direct": tiny.neglogz_direct,
        "nonfinite": dict(rec["nonfinite"]),
        "old_vs_direct": gap([(range(rows), old[0])], direct),
        "ref_vs_direct": gap([(range(rows), ref[0])], tiny.direct_ref),
        "loss_vs_direct": gap([(r, lp) for r, (lp, _, _) in losses], direct),
        "invalid_rows_zero": all(float(old[0][row].abs().max()) == 0.0 for row in range(rows) if not valid[row]),
        "loss_requires_grad": all(lp.requires_grad for _, (lp, _, _) in losses),
        "loss_entropy": all(ent is not None for _, (_, ent, _) in losses),
        "trainable_tensors": len(trainable),
        "loss_grad_tensors": sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trainable),
        "selfcheck": rec["selfcheck"],
        "is_level": rec["is_level"], "seq_valid": rec["seq_valid"],
        "ratio": ratio_l,
        "seq_weights_vs_ratio": (max(abs(float(w) - x) for w, x in zip(weights, ratio_l))
                                 if len(weights) == rows else None),
        "raw_vs_ratio": (max(abs(float(w) - x) / max(1.0, abs(x)) for w, x in zip(raw, ratio_l))
                         if len(raw) == rows else None),
        "truncated_raw_weights": [raw[row] for row in range(rows) if not valid[row]] if len(raw) == rows else None,
        "after_score": {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in scalars.items()},
    }


# ------------------------------------------------------------------------------------------ the verdict
def _small(x: Any, bar: float = TOL) -> bool:
    return isinstance(x, float) and math.isfinite(x) and x <= bar


def _text_failures(report: Mapping[str, Any]) -> list[str]:
    """Every --texts file read and holding texts, and at least MIN_TEXTS of their texts in the language."""
    files = list(report.get("texts_files") or [])
    read = report.get("texts_read") or {}
    bad = []
    for p in files:
        r = read.get(p)
        if not isinstance(r, dict) or r.get("error") or not r.get("texts"):
            why = r.get("error") if isinstance(r, dict) else None
            bad.append(f"--texts {p} was not read ({why or 'no text rows'})")
    by_source = (report.get("masks") or {}).get("in_language_by_source") or {}
    counts = [by_source.get(p, 0) for p in files]
    n = sum(c for c in counts if isinstance(c, int) and not isinstance(c, bool))
    if n < MIN_TEXTS:
        bad.append(f"{n} in-language texts in {len(files)} --texts files: D-020's {MIN_TEXTS} are required")
    return bad


def _walk_failures(name: str, t: Any) -> list[str]:
    """A walk tally's failures: missing or errored, nothing compared, or a difference, a refusal or an unterminated
    walk (a random walk cut at its bound is none of these)."""
    if not isinstance(t, dict):
        return [f"{name}: did not run"]
    if t.get("error"):
        return [f"{name}: {t['error']}"]
    bad = []
    if not (isinstance(t.get("positions"), int) and t["positions"] > 0):
        bad.append(f"{name}: no position was compared")
    if t.get("mismatch") != 0:
        bad.append(f"{name}: {t.get('mismatch')!r} positions where the NFA and xgrammar disagree")
    for k in ("xgrammar", "nfa"):
        r, u = (t.get("refused") or {}).get(k), (t.get("not_terminated") or {}).get(k)
        if r != 0 or u != 0:
            bad.append(f"{name}: {k} refused {r!r} walks and left {u!r} unterminated")
    return bad


def _mask_failures(m: Any) -> list[str]:
    if not isinstance(m, dict):
        return ["the mask builders did not run"]
    bad = []
    if not m.get("in_language"):
        bad.append("no in-language text was walked")
    bad += _walk_failures("texts", m)
    c = m.get("char_level")
    bad += _walk_failures("char_level", c)
    if isinstance(c, dict) and not c.get("error"):
        if c.get("no_token") != 0:
            bad.append(f"char_level: {c.get('no_token')!r} texts hold a character with no one-character token")
        if c.get("texts") != m.get("in_language"):
            bad.append(f"char_level walked {c.get('texts')!r} of the {m.get('in_language')!r} in-language texts")
    w = m.get("random_walks")
    bad += _walk_failures("random_walks", w)
    if isinstance(w, dict) and not w.get("error"):
        drawn = {d: r.get("walks") if isinstance(r, dict) else None for d, r in (w.get("drawn") or {}).items()}
        if drawn != dict(RANDOM_WALKS):
            bad.append(f"random_walks drew {drawn}, not {dict(RANDOM_WALKS)}")
    return bad


def _support_failures(t: Any) -> list[str]:
    if not isinstance(t, dict):
        return ["the support-path check did not run"]
    bad = []
    for k in ("masked_support_vs_direct", "masked_fill_vs_direct", "masked_support_vs_fill",
              "unmasked_support_vs_fill", "entropy_support_vs_fill"):
        if not _small(t.get(k)):
            bad.append(f"{k} {t.get(k)!r} > {TOL}")
    if not (isinstance(t.get("shifted_vs_direct"), float) and t["shifted_vs_direct"] > SHIFTED_MIN):
        bad.append(f"a support one column off agrees with the direct masked log-softmax "
                   f"({t.get('shifted_vs_direct')!r}): the check cannot see the alignment")
    gd, gs = t.get("grad_max_abs_diff"), t.get("grad_max_abs")
    if not (t.get("grad_tensors") and t.get("grad_tensors") == t.get("grad_tensors_fill")
            and isinstance(gd, float) and isinstance(gs, float) and gd <= TOL * max(1.0, gs)):
        bad.append(f"gradients differ: {gd!r} (scale {gs!r}, {t.get('grad_tensors')} tensors)")
    if not t.get("invalid_rows_zero"):
        bad.append("a row with no valid token got non-zero log-probs")
    tiles = t.get("vocab_tiles")
    reach = {k: t.get(k) for k in ("last_tile_support_positions", "last_tile_targets", "multi_tile_positions",
                                   "forced_positions")}
    if not (isinstance(tiles, int) and tiles >= 2 and all(isinstance(v, int) and v > 0 for v in reach.values())):
        bad.append(f"the support check does not reach the kernel's tiles: {tiles!r} tiles, {reach}")
    if t.get("compact_error"):
        bad.append(f"the compact packing: {t['compact_error']}")
    if not (isinstance(t.get("compact_calls"), int) and t["compact_calls"] > 0):
        bad.append(f"the compact call never read the patched _use_triton ({t.get('compact_calls')!r} reads): TRL's "
                   "fused forward no longer chooses its packing there")
    if t.get("compact_kernel_triton") is not False:
        bad.append(f"the kernel's own fold choice read {t.get('compact_kernel_triton')!r} under the patch, not the "
                   "torch fallback")
    if not _small(t.get("compact_vs_noncompact")):
        bad.append(f"compact_vs_noncompact {t.get('compact_vs_noncompact')!r} > {TOL}")
    cg = t.get("compact_grad_max_abs_diff")
    if not (t.get("compact_grad_tensors") == t.get("grad_tensors") and isinstance(cg, float) and isinstance(gs, float)
            and cg <= TOL * max(1.0, gs)):
        bad.append(f"the compact packing's gradient differs: {cg!r} ({t.get('compact_grad_tensors')!r} tensors)")
    return bad


def _glue_failures(g: Any) -> list[str]:
    """What the masked trainer's glue must have done on the real objects (module docstring, 4)."""
    from lofi_rl.rl import masked

    if not isinstance(g, dict):
        return ["the glue check did not run"]
    bad = []
    want_state = {"before": "enabled", "inside_disable_adapter": "disabled", "after": "enabled"}
    if g.get("adapter_state") != want_state:
        bad.append(f"glue: masked.adapter_state_of read {g.get('adapter_state')!r}, not {want_state}")
    lm = g.get("lm_head") or {}
    if lm.get("verdict") != "match":
        bad.append(f"glue: lm_head_record {lm.get('verdict')!r} ({lm.get('missing') or lm.get('error')})")
    if g.get("error"):
        return bad + [f"glue: {g['error']}"]
    valid = g.get("valid")
    if not (isinstance(valid, list) and valid and all(isinstance(v, int) for v in valid)
            and g.get("rows_in_batch") == len(valid)):
        return bad + [f"glue: no batch was recorded ({valid!r})"]
    n, scored, tokens = len(valid), sum(1 for v in valid if v), sum(valid)
    labels = ["old", "ref"] + ["loss"] * len(LOSS_MICRO)
    passes = [p if isinstance(p, dict) else {} for p in g.get("passes") or []]
    seen = {k: [p.get(k) for p in passes] for k in ("label", "adapters", "grad")}
    want = {"label": labels, "adapters": [masked.ADAPTERS[x] for x in labels], "grad": [x == "loss" for x in labels]}
    for k in want:
        if seen[k] != want[k]:
            bad.append(f"glue: the passes' {k} {seen[k]}, not {want[k]}")
    for k, v in (("rows", n), ("rows_scored", scored)):
        if g.get(k) != {p: v for p in masked.PASSES}:
            bad.append(f"glue: {k} {g.get(k)!r}, not {v} in each pass")
    for k, v in (("completions", n), ("terminated", sum(STOPS_AT_END)), ("violations", 0), ("cache_miss", 0),
                 ("replay_unknown", 0), ("tokens", tokens), ("neglogz_tokens", tokens), ("neglogz_nonfinite", 0),
                 ("nonfinite", {p: 0 for p in masked.PASSES}), ("trl_sampling_mask", None),
                 ("invalid_rows_zero", True), ("loss_requires_grad", True), ("loss_entropy", True),
                 ("is_level", "sequence"), ("seq_valid", [v > 0 for v in valid])):
        if g.get(k) != v:
            bad.append(f"glue: {k} {g.get(k)!r}, not {v!r}")
    nz, nzd = g.get("neglogz"), g.get("neglogz_direct")
    if not (isinstance(nz, float) and isinstance(nzd, float) and math.isfinite(nz) and abs(nz - nzd) <= TOL):
        bad.append(f"glue: -log Z {nz!r}, the float64 one {nzd!r}")
    for k in ("old_vs_direct", "ref_vs_direct", "loss_vs_direct", "seq_weights_vs_ratio", "raw_vs_ratio"):
        if not _small(g.get(k)):
            bad.append(f"glue: {k} {g.get(k)!r} > {TOL}")
    if not (isinstance(g.get("trainable_tensors"), int) and g["trainable_tensors"] > 0
            and g.get("loss_grad_tensors") == g["trainable_tensors"]):
        bad.append(f"glue: the loss passes' gradient reached {g.get('loss_grad_tensors')!r} of "
                   f"{g.get('trainable_tensors')!r} adapter tensors")
    trunc = g.get("truncated_raw_weights")
    if not (isinstance(trunc, list) and len(trunc) == n - scored
            and all(isinstance(w, float) and _small(abs(w - 1.0)) for w in trunc)):
        bad.append(f"glue: a truncated row's raw weight {trunc!r} is not exp(0): after_score did not apply its mask")
    sc = g.get("selfcheck")
    if not (isinstance(sc, dict) and sc.get("error") is None and sc.get("against") == "fill"
            and all(isinstance(sc.get(k), dict) and _small(sc[k].get("max_abs_diff")) and sc[k].get("nonfinite") == 0
                    for k in ("masked", "unmasked"))):
        bad.append(f"glue: the self-check {sc!r}")
    a, ratio = g.get("after_score"), g.get("ratio")
    mean = a.get("is/seq_weight_mean") if isinstance(a, dict) else None
    ratio_mean = (sum(ratio) / n if isinstance(ratio, list) and len(ratio) == n
                  and all(isinstance(x, float) for x in ratio) else None)
    if not (isinstance(a, dict) and set(a) == set(AFTER_SCORE_KEYS) and a.get("mask/violations") == 0.0
            and isinstance(nz, float) and a.get("mask/neglogz") == nz and a.get("mask/nfa_mismatch") is None
            and isinstance(mean, float) and isinstance(ratio_mean, float) and _small(abs(mean - ratio_mean))):
        bad.append(f"glue: after_score returned {a!r} (the ratio's mean is {ratio_mean!r})")
    return bad


def verdict(report: dict[str, Any]) -> list[str]:
    """The reasons the preflight fails (empty: MASK_PREFLIGHT_OK)."""
    bad = list(report.get("problems") or [])
    sc = report.get("source_checks") or {}
    if sc.get("verdict") != "match":
        bad.append(f"source checks {sc.get('verdict')!r} (failed {sc.get('failed')})")
    if report.get("mask_refusals"):
        bad.append(f"mask refusals {report['mask_refusals']}")
    bad += _text_failures(report)
    bad += _mask_failures(report.get("masks"))
    bad += _support_failures(report.get("support_path"))
    bad += _glue_failures(report.get("glue"))
    return bad


# ------------------------------------------------------------------------------------------ the run
def run(cfg_path: Path, out_dir: Path, texts: list[Path]) -> int:
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"config": str(cfg_path), "texts_files": [str(p) for p in texts], "texts_read": {},
                              "problems": [], "versions": train._versions(), "code": train.code_record()}
    rows = read_texts(texts, report["problems"], report["texts_read"])
    try:
        cfg = load_config(cfg_path)
        if cfg.trainer_logprobs is None or cfg.structured_outputs is None:
            raise ConfigError(f"{cfg_path} is not a masked config (trainer_logprobs and structured_outputs)")
        report["trainer_logprobs"] = cfg.trainer_logprobs
        checks = train.source_checks(train.shared_settings(cfg))
        report["source_checks"] = checks
        report["mask_refusals"] = train.mask_refusals(cfg.trainer_logprobs, checks)

        from transformers import AutoTokenizer, GenerationConfig
        from trl import GRPOTrainer

        from kit.hf_cache import resolve_pinned
        from lofi_rl import prompt_v2

        snap = Path(resolve_pinned(cfg.model_repo, cfg.model_revision, train.SNAPSHOT_FILES))
        tok = AutoTokenizer.from_pretrained(snap)
        eos = train._eos_ids(tok, GenerationConfig.from_pretrained(snap))
        kw = resolve(cfg, next(iter(cfg.profiles)))
        t0 = time.perf_counter()
        setup, _cls, rec = train.build_mask(cfg, kw, snap, tok, eos, GRPOTrainer)
        report["build_s"] = round(time.perf_counter() - t0, 3)
        report["setup"] = rec
        refs = prompt_v2.reference_texts(load_palette(cfg.palette_path(), verify=False), strict=False)
        t0 = time.perf_counter()
        report["masks"] = check_masks(setup, tok, [(f"prompt_v2:{k}", v) for k, v in refs.items()] + rows)
        report["masks"]["seconds"] = round(time.perf_counter() - t0, 1)
        report["replay"] = getattr(setup.replay, "record", None)
    except Exception as err:  # noqa: BLE001: a reported failure, with its trace
        report["problems"].append(f"{type(err).__name__}: {err}"[:800])
        report["trace"] = traceback.format_exc()[-6000:]
    tiny = None
    try:
        tiny = tiny_model()
    except Exception as err:  # noqa: BLE001
        report["problems"].append(f"tiny model: {type(err).__name__}: {err}"[:800])
        report["tiny_trace"] = traceback.format_exc()[-6000:]
    if tiny is not None:
        for key, check in (("support_path", check_support_path), ("glue", check_glue)):
            try:
                t0 = time.perf_counter()
                report[key] = check(tiny)
                report[key]["seconds"] = round(time.perf_counter() - t0, 1)
            except Exception as err:  # noqa: BLE001
                report["problems"].append(f"{key}: {type(err).__name__}: {err}"[:800])
                report[f"{key}_trace"] = traceback.format_exc()[-6000:]
    try:
        bad = verdict(report)
    except Exception as err:  # noqa: BLE001: a record the verdict cannot read fails, and the report is still written
        bad = [f"the verdict could not read the report: {type(err).__name__}: {err}"[:800]]
    report.update(ok=not bad, failures=bad, wall_s=round(time.time() - t_start, 1))
    (out_dir / "preflight.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n")
    print("MASK_PREFLIGHT_OK" if not bad else "MASK_PREFLIGHT_FAILED " + "; ".join(bad)[:2000])
    return 0 if not bad else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--texts", type=Path, action="append", default=[], help="JSONL with a `text` field (repeatable)")
    a = ap.parse_args(argv)
    return run(a.config, a.out, a.texts)


if __name__ == "__main__":
    sys.exit(main())
