"""The masked trainer's preflight (lofi_rl.rl.mask_preflight): what the laptop can check of it. Its checks need the
policy env (torch, TRL, PEFT, xgrammar), so here:
- the verdict fails closed on every part that did not run or did not agree: the --texts files and D-020's 980 texts,
  the three walks (tokenizer ids, one character per token, random walks; a walk cut at its bound is not a failure),
  the support path with its vocabulary tiles and compact packing, and the glue on real objects;
- the walks' own logic, end to end, with GrammarMask on both sides (a stand-in replay exposes the matcher calls the
  xgrammar-driven walks make), and with a replay that disagrees;
- the tiny generation's TRL layout and the tiles it reaches, and the glue's construction through
  masked.MaskedPasses.check_generation;
- run() itself, which without the policy env writes its report and fails closed, naming each part that did not run;
- the job passes D-020's three files and stops, with the verdict's prefix, when one is missing; and those files, as
  pulled to the laptop, hold the 980 in-language texts the verdict requires.
"""
from __future__ import annotations

import copy
import glob
import inspect
import json
import os
import random
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from lofi_rl.grammar_mask import BYTE_TO_CHAR, GrammarMask
from lofi_rl.rl import mask_preflight as M
from lofi_rl.rl import masked

ROOT = Path(__file__).resolve().parents[1]
FILES = ["/scr/runs/junk-13640325/summary/completions.jsonl", "/scr/runs/grpo-smoke-13657200/summary/a/rewards.jsonl",
         "/scr/runs/grpo-smoke-13657200/summary/b/rewards.jsonl"]


def tally(**kw) -> dict:
    return {"positions": 5000, "mismatch": 0, "examples": [], "refused": {"xgrammar": 0, "nfa": 0},
            "not_terminated": {"xgrammar": 0, "nfa": 0}, **kw}


def good_glue() -> dict:
    valid = [9, 5, 0, 7]
    labels = ["old", "ref", "loss", "loss"]
    return {
        "error": None, "adapter_state": {"before": "enabled", "inside_disable_adapter": "disabled", "after": "enabled"},
        "lm_head": {"verdict": "match", "missing": []},
        "rows_in_batch": 4, "valid": valid, "completions": 4, "terminated": 3, "violations": 0, "cache_miss": 0,
        "replay_unknown": 0, "trl_sampling_mask": None,
        "passes": [{"label": x, "adapters": masked.ADAPTERS[x], "grad": x == "loss"} for x in labels],
        "rows": {"old": 4, "ref": 4, "loss": 4}, "rows_scored": {"old": 3, "ref": 3, "loss": 3}, "tokens": 21,
        "neglogz": 1.83, "neglogz_tokens": 21, "neglogz_nonfinite": 0, "neglogz_direct": 1.83 + 1e-7,
        "nonfinite": {"old": 0, "ref": 0, "loss": 0},
        "old_vs_direct": 1e-7, "ref_vs_direct": 1e-7, "loss_vs_direct": 1e-7, "invalid_rows_zero": True,
        "loss_requires_grad": True, "loss_entropy": True, "trainable_tensors": 28, "loss_grad_tensors": 28,
        "selfcheck": {"against": "fill", "error": None, "masked": {"max_abs_diff": 1e-7, "nonfinite": 0},
                      "unmasked": {"max_abs_diff": 1e-7, "nonfinite": 0}},
        "is_level": "sequence", "seq_valid": [True, True, False, True], "ratio": [1.0, 1.0, 1.0, 1.0],
        "seq_weights_vs_ratio": 0.0, "raw_vs_ratio": 1e-7, "truncated_raw_weights": [1.0],
        "after_score": {"mask/violations": 0.0, "mask/neglogz": 1.83, "mask/forced_share": 1 / 21,
                        "mask/nfa_mismatch": None, "is/seq_weight_mean": 1.0, "is/seq_weight_max_share": 0.25},
    }


def good_report() -> dict:
    return {
        "problems": [],
        "texts_files": list(FILES),
        "texts_read": {p: {"lines": n, "texts": n, "error": None} for p, n in zip(FILES, (500, 320, 160))},
        "source_checks": {"verdict": "match", "failed": []},
        "mask_refusals": [],
        "masks": {"in_language": 989, "in_language_by_source": {"prompt_v2": 9, FILES[0]: 500, FILES[1]: 320,
                                                                 FILES[2]: 160},
                  **tally(),
                  "char_level": tally(texts=989, no_token=0),
                  "random_walks": tally(drawn={"nfa": {"walks": 300}, "xgrammar": {"walks": 100}},
                                        cut_at_max_len={"xgrammar": 2, "nfa": 2})},
        "support_path": {"masked_support_vs_direct": 1e-7, "masked_fill_vs_direct": 1e-7,
                         "masked_support_vs_fill": 1e-7, "unmasked_support_vs_fill": 1e-7,
                         "entropy_support_vs_fill": 1e-7, "shifted_vs_direct": 3.0, "grad_tensors": 20,
                         "grad_tensors_fill": 20, "grad_max_abs_diff": 1e-8, "grad_max_abs": 0.5,
                         "invalid_rows_zero": True, "vocab_tiles": 3, "last_tile_support_positions": 15,
                         "last_tile_targets": 4, "multi_tile_positions": 19, "forced_positions": 1,
                         "compact_vs_noncompact": 1e-7, "compact_grad_tensors": 20, "compact_grad_max_abs_diff": 1e-8,
                         "compact_calls": 4, "compact_kernel_triton": False, "compact_error": None},
        "glue": good_glue(),
    }


def test_a_clean_report_passes():
    assert M.verdict(good_report()) == []


def broken(edit) -> list[str]:
    r = copy.deepcopy(good_report())
    edit(r)
    return M.verdict(r)


def test_every_part_fails_closed():
    assert broken(lambda r: r["problems"].append("ImportError: no xgrammar"))
    assert broken(lambda r: r["source_checks"].update(verdict="not_matched", failed=["T2"]))
    assert broken(lambda r: r.pop("source_checks"))
    assert broken(lambda r: r.update(mask_refusals=["T2 path fill"]))
    assert broken(lambda r: r.pop("masks"))
    assert broken(lambda r: r["masks"].update(in_language=0))
    assert broken(lambda r: r["masks"].update(mismatch=1))
    assert broken(lambda r: r["masks"].pop("mismatch"))                  # absent is not zero
    assert broken(lambda r: r["masks"].update(positions=0))
    assert broken(lambda r: r["masks"]["refused"].update(nfa=1))
    assert broken(lambda r: r["masks"]["not_terminated"].update(xgrammar=1))
    assert broken(lambda r: r.pop("support_path"))
    assert broken(lambda r: r["support_path"].update(masked_support_vs_fill=1e-3))
    assert broken(lambda r: r["support_path"].update(entropy_support_vs_fill=None))
    assert broken(lambda r: r["support_path"].update(shifted_vs_direct=0.01))      # blind to the alignment
    assert broken(lambda r: r["support_path"].update(grad_max_abs_diff=1e-2))
    assert broken(lambda r: r["support_path"].update(grad_tensors=0, grad_tensors_fill=0, compact_grad_tensors=0))
    assert broken(lambda r: r["support_path"].update(grad_tensors_fill=19))
    assert broken(lambda r: r["support_path"].update(invalid_rows_zero=False))


def test_the_texts_files_must_all_be_read_and_hold_d020s_980_texts():
    assert broken(lambda r: r["texts_read"].pop(FILES[1])) == [f"--texts {FILES[1]} was not read (no text rows)"]
    got = broken(lambda r: r["texts_read"][FILES[0]].update(error="FileNotFoundError: gone", texts=0))
    assert got == [f"--texts {FILES[0]} was not read (FileNotFoundError: gone)"]
    assert broken(lambda r: r["texts_read"][FILES[2]].update(texts=0))
    # 979 of the files' texts in the language: the 9 references do not make up the difference
    assert broken(lambda r: r["masks"]["in_language_by_source"].update({FILES[2]: 159})) == [
        "979 in-language texts in 3 --texts files: D-020's 980 are required"]
    assert broken(lambda r: r.update(texts_files=[], texts_read={}))     # the job ran without them
    assert broken(lambda r: r["masks"]["in_language_by_source"].update({FILES[0]: True}))   # not a count


@pytest.mark.parametrize("key", ["char_level", "random_walks"])
def test_each_new_walk_fails_on_a_difference_a_refusal_or_an_unterminated_walk(key):
    assert broken(lambda r: r["masks"].pop(key)) == [f"{key}: did not run"]
    assert broken(lambda r: r["masks"][key].update(error="AttributeError: _matcher")) == [
        f"{key}: AttributeError: _matcher"]
    assert broken(lambda r: r["masks"][key].update(mismatch=2))
    assert broken(lambda r: r["masks"][key].update(positions=0))
    for k in ("refused", "not_terminated"):
        for builder in ("xgrammar", "nfa"):
            assert broken(lambda r: r["masks"][key][k].update({builder: 1}))


def test_char_level_must_cover_every_in_language_text_with_one_character_tokens():
    assert broken(lambda r: r["masks"]["char_level"].update(no_token=1))
    assert broken(lambda r: r["masks"]["char_level"].update(texts=988))


def test_random_walks_must_draw_the_registered_counts_and_may_be_cut_at_their_bound():
    r = good_report()
    r["masks"]["random_walks"]["cut_at_max_len"] = {"xgrammar": 100, "nfa": 100}
    assert M.verdict(r) == [], "a walk cut at max_len is counted apart, not as unterminated"
    assert broken(lambda r: r["masks"]["random_walks"]["drawn"]["xgrammar"].update(walks=99))
    assert broken(lambda r: r["masks"]["random_walks"]["drawn"].pop("nfa"))


def test_the_support_path_must_reach_the_kernels_tiles_and_agree_with_the_compact_packing():
    for k in ("vocab_tiles", "last_tile_support_positions", "last_tile_targets", "multi_tile_positions",
              "forced_positions"):
        assert broken(lambda r: r["support_path"].update({k: 0})), k
        assert broken(lambda r: r["support_path"].pop(k)), k
    assert broken(lambda r: r["support_path"].update(vocab_tiles=1))      # a vocabulary inside one tile
    assert broken(lambda r: r["support_path"].update(compact_vs_noncompact=1e-3))
    assert broken(lambda r: r["support_path"].update(compact_vs_noncompact=None))
    assert broken(lambda r: r["support_path"].update(compact_grad_max_abs_diff=1e-2))
    assert broken(lambda r: r["support_path"].update(compact_grad_tensors=19))
    assert broken(lambda r: r["support_path"].update(compact_calls=0))    # the patch never reached the packing
    assert broken(lambda r: r["support_path"].update(compact_kernel_triton=True))
    assert broken(lambda r: r["support_path"].update(compact_kernel_triton=None))
    assert broken(lambda r: r["support_path"].update(compact_error="AttributeError: _use_triton"))


GLUE_BREAKS = [
    lambda g: g.update(error="MaskError: the ref pass ran with the LoRA adapters enabled"),
    lambda g: g["adapter_state"].update(inside_disable_adapter="enabled"),
    lambda g: g["adapter_state"].update(after="irregular"),
    lambda g: g["lm_head"].update(verdict="not_matched"),
    lambda g: g.update(valid=[]),
    lambda g: g["passes"].pop(),
    lambda g: g["passes"][1].update(label="loss"),
    lambda g: g["passes"][1].update(adapters="enabled"),
    lambda g: g["passes"][2].update(grad=False),
    lambda g: g["rows"].update(ref=3),
    lambda g: g["rows_scored"].update(loss=4),
    lambda g: g.update(completions=3),
    lambda g: g.update(terminated=4),
    lambda g: g.update(violations=1),
    lambda g: g.update(cache_miss=1),
    lambda g: g.update(replay_unknown=1),
    lambda g: g.update(trl_sampling_mask={"completions": 4}),
    lambda g: g.update(tokens=20),
    lambda g: g.update(neglogz_tokens=0),
    lambda g: g.update(neglogz=None),
    lambda g: g.update(neglogz=float("nan")),
    lambda g: g.update(neglogz=1.9),                                     # not the float64 -log Z
    lambda g: g["nonfinite"].update(loss=1),
    lambda g: g.update(old_vs_direct=1e-3),
    lambda g: g.update(ref_vs_direct=None),                              # the reference pass never compared
    lambda g: g.update(loss_vs_direct=float("inf")),
    lambda g: g.update(invalid_rows_zero=False),
    lambda g: g.update(loss_requires_grad=False),
    lambda g: g.update(loss_entropy=False),
    lambda g: g.update(loss_grad_tensors=27),
    lambda g: g.update(trainable_tensors=0, loss_grad_tensors=0),
    lambda g: g["selfcheck"].update(error="RuntimeError: shape"),
    lambda g: g["selfcheck"]["masked"].update(max_abs_diff=0.8),
    lambda g: g["selfcheck"]["unmasked"].update(nonfinite=1),
    lambda g: g.update(selfcheck=None),
    lambda g: g.update(is_level="token"),
    lambda g: g.update(seq_valid=[True, True, True, True]),
    lambda g: g.update(seq_weights_vs_ratio=0.5),
    lambda g: g.update(raw_vs_ratio=None),
    lambda g: g.update(truncated_raw_weights=[8103.08]),                 # exp(9): the zero mask was not applied
    lambda g: g.update(truncated_raw_weights=[]),
    lambda g: g["after_score"].pop("mask/neglogz"),
    lambda g: g["after_score"].update({"mask/violations": 1.0}),
    lambda g: g["after_score"].update({"mask/nfa_mismatch": 0.0}),       # this setup has no NFA
    lambda g: g["after_score"].update({"is/seq_weight_mean": 0.5}),
    lambda g: g["after_score"].update({"is/seq_weight_mean": None}),
]


@pytest.mark.parametrize("edit", GLUE_BREAKS, ids=range(len(GLUE_BREAKS)))
def test_the_glue_fails_on_anything_the_trainer_is_not_written_to_do(edit):
    assert broken(lambda r: edit(r["glue"]))


def test_the_glue_must_have_run():
    assert broken(lambda r: r.pop("glue")) == ["the glue check did not run"]


def test_a_record_the_verdict_cannot_read_still_fails_without_raising():
    assert broken(lambda r: r["glue"].update(valid=[None, 1], truncated_raw_weights=["x"], ratio=["y"]))
    assert broken(lambda r: r["glue"].update(passes=[None, 3]))
    assert broken(lambda r: r["masks"]["random_walks"].update(drawn={"nfa": 300, "xgrammar": None}))


def test_without_the_policy_env_the_run_writes_its_report_and_fails_closed(tmp_path):
    import importlib.util

    if importlib.util.find_spec("torch") or importlib.util.find_spec("transformers"):
        pytest.skip("this env has torch or transformers: the run would go further than the laptop's")
    texts = tmp_path / "rewards.jsonl"
    texts.write_text(json.dumps({"text": "vamp: x"}) + "\n")
    code = M.main(["--config", str(ROOT / "configs" / "grpo_smoke_v2_grammar_masked.yaml"), "--out",
                   str(tmp_path / "out"), "--texts", str(texts)])
    report = json.loads((tmp_path / "out" / "preflight.json").read_text())
    assert code == 1 and report["ok"] is False
    assert report["texts_read"] == {str(texts): {"lines": 1, "texts": 1, "error": None}}
    for part in ("the mask builders did not run", "the support-path check did not run", "the glue check did not run",
                 "0 in-language texts in 1 --texts files: D-020's 980 are required"):
        assert part in report["failures"], part


# ------------------------------------------------------------------------------------------ reading the texts
def test_read_texts_keeps_rows_with_text_and_reports_the_rest(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join([json.dumps({"text": "vamp: x"}), "not json", json.dumps({"text": ""}),
                            json.dumps({"other": 1}), "", json.dumps({"text": "kick: y"})]) + "\n")
    bad = tmp_path / "latin.jsonl"
    bad.write_bytes(b'{"text": "caf\xe9"}\n')
    problems: list[str] = []
    files: dict = {}
    got = M.read_texts([p, tmp_path / "missing.jsonl", bad], problems, files)
    assert got == [(f"{p}:1", "vamp: x"), (f"{p}:6", "kick: y")]
    assert any("rows.jsonl:2: not JSON" in x for x in problems) and any("missing.jsonl" in x for x in problems)
    assert files[str(p)] == {"lines": 5, "texts": 2, "error": None}, "the blank line is skipped, not counted"
    assert files[str(tmp_path / "missing.jsonl")]["error"].startswith("FileNotFoundError")
    assert files[str(bad)]["error"].startswith("UnicodeDecodeError") and len(problems) == 3


# ------------------------------------------------------------------------------------------ the walks
def spell(text: str) -> str:
    return "".join(BYTE_TO_CHAR[b] for b in text.encode("utf-8"))


def test_one_char_ids_reads_each_ascii_characters_byte_level_token():
    vocab = {spell(chr(b)): 1000 + b for b in range(128)} | {spell(" a"): 7, "<|im_end|>": 9}
    one = M.one_char_ids(vocab)
    assert len(one) == 128 and one[" "] == 1000 + 32 and one["\n"] == 1000 + 10 and one["a"] == 1000 + 97
    del vocab[spell("c")]
    one = M.one_char_ids(vocab)
    assert "c" not in one and M.char_ids("ab c", one) is None and M.char_ids("ab a", one) == [1097, 1098, 1032, 1097]
    assert M.char_ids("café", M.one_char_ids({spell(chr(b)): b for b in range(256)})) is None, "ASCII only"


def test_one_char_ids_over_the_cached_qwen_vocabulary():
    pattern = "models--mlx-community--Qwen3-4B-4bit/snapshots/*/tokenizer.json"     # tests/test_grammar_mask.py's
    hub = (Path(os.environ["HF_HUB_CACHE"]) if os.environ.get("HF_HUB_CACHE")
           else Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    found = sorted(glob.glob(str(hub / pattern)))
    if not found:
        pytest.skip(f"{hub / pattern} is not on this machine")
    tj = json.loads(Path(found[0]).read_text(encoding="utf-8"))
    vocab = dict(tj["model"]["vocab"]) | {a["content"]: a["id"] for a in tj["added_tokens"]}     # get_vocab()
    one = M.one_char_ids(vocab)
    assert len(one) == 128, "every ASCII character is a token of Qwen's byte-level vocabulary"
    assert (one[" "], one["\n"]) == (220, 198)                           # "Ġ", "Ċ"


TOKENS = {i: chr(i) for i in range(128)} | {128: "ab", 129: " c", 130: "ba"}
STOPS = (200, 201)
REGEX = "(ab|ba){1,3}( c)?"


class MaskReplay:
    """masked.XgrammarReplay's surface over a GrammarMask (no xgrammar on the laptop): walk() is the mask's, and
    _matcher (reset, accept_token, is_terminated) and _allowed_now() replay it token by token, as the preflight's
    xgrammar-driven walks call them."""

    def __init__(self, mask: GrammarMask):
        self.mask, self._matcher, self.ids = mask, self, []

    def walk(self, ids):
        return self.mask.walk(ids)

    def reset(self):
        self.ids = []

    def _allowed_now(self):
        return self.mask.walk(self.ids + [-1]).allowed[-1]

    def accept_token(self, tok):
        if tok not in self._allowed_now():
            return False
        self.ids.append(tok)
        return True

    def is_terminated(self):
        return bool(self.ids) and self.ids[-1] in self.mask.stop_ids


class GreedyTok:
    """A tokenizer stand-in: get_vocab() spells each token as tokenizer.json does (byte level), and a call encodes
    longest token first (an encoding of the text; the preflight walks whatever the tokenizer gives)."""

    def __init__(self, tokens):
        self.tokens, self.eos_token_id = tokens, STOPS[1]
        self.by_text = {t: i for i, t in tokens.items()}

    def get_vocab(self):
        return {spell(t): i for i, t in self.tokens.items()}

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            n = next(n for n in (2, 1) if text[i:i + n] in self.by_text)
            ids.append(self.by_text[text[i:i + n]])
            i += n
        return {"input_ids": ids}


def setup_for(replay_tokens=TOKENS):
    nfa = GrammarMask.from_tokens(TOKENS, REGEX, vocab_size=202, stop_ids=STOPS)
    replay = MaskReplay(GrammarMask.from_tokens(replay_tokens, REGEX, vocab_size=202, stop_ids=STOPS))
    return SimpleNamespace(regex=REGEX, replay=replay, nfa=nfa, stop_ids=STOPS)


TEXTS = [("/f/a.jsonl:1", "abab c"), ("/f/a.jsonl:2", "zz"), ("/f/b.jsonl:1", "ba"), ("prompt_v2:E1", "abba c")]


def test_check_masks_walks_three_ways_and_agrees_when_the_builders_do():
    out = M.check_masks(setup_for(), GreedyTok(TOKENS), TEXTS)
    assert (out["texts"], out["in_language"]) == (4, 3)
    assert out["in_language_by_source"] == {"/f/a.jsonl": 1, "/f/b.jsonl": 1, "prompt_v2": 1}
    # tokenizer ids + EOS: [128, 128, 129, 201], [130, 201], [128, 130, 129, 201]
    assert out["positions"] == 4 + 2 + 4 and out["eos"] == 201
    char = out["char_level"]
    assert (char["texts"], char["positions"], char["no_token"], char["one_char_tokens"]) == (3, 7 + 3 + 7, 0, 128)
    walks = out["random_walks"]
    assert walks["walks"] == dict(M.RANDOM_WALKS) and walks["max_len"] == M.WALK_MAX_LEN
    assert {d: r["walks"] for d, r in walks["drawn"].items()} == dict(M.RANDOM_WALKS)
    assert {d: r["ends"] for d, r in walks["drawn"].items()} == {"nfa": {"stop": 300}, "xgrammar": {"stop": 100}}
    assert walks["positions"] == sum(r["ids"] for r in walks["drawn"].values())
    assert M._mask_failures(out) == []
    again = M.check_masks(setup_for(), GreedyTok(TOKENS), TEXTS)
    assert again["random_walks"]["drawn"]["nfa"]["ids"] == walks["drawn"]["nfa"]["ids"], "seeded"


def test_a_walk_cut_at_its_bound_is_counted_apart():
    out = M.random_walks(setup_for(), {"nfa": 40, "xgrammar": 40}, max_len=2)
    assert out["drawn"]["nfa"]["ends"]["max_len"] > 0 and out["drawn"]["xgrammar"]["ends"]["max_len"] > 0
    cut = out["drawn"]["nfa"]["ends"]["max_len"] + out["drawn"]["xgrammar"]["ends"]["max_len"]
    assert out["cut_at_max_len"] == {"xgrammar": cut, "nfa": cut}
    assert out["not_terminated"] == {"xgrammar": 0, "nfa": 0} and M._walk_failures("random_walks", out) == []


def test_check_masks_finds_a_replay_that_disagrees_and_a_character_with_no_token():
    without_ab = {i: t for i, t in TOKENS.items() if t != "ab"}           # this "xgrammar" never allows "ab"
    out = M.check_masks(setup_for(without_ab), GreedyTok(TOKENS), TEXTS)
    assert out["mismatch"] > 0 and out["refused"] == {"xgrammar": 2, "nfa": 0}   # "abab c", "abba c" use "ab"
    assert out["char_level"]["mismatch"] > 0 and out["char_level"]["refused"] == {"xgrammar": 0, "nfa": 0}
    assert out["random_walks"]["mismatch"] > 0 and out["random_walks"]["refused"]["xgrammar"] > 0
    fails = M._mask_failures(out)
    assert any(f.startswith("texts:") for f in fails) and any(f.startswith("char_level:") for f in fails)
    assert any(f.startswith("random_walks:") for f in fails)
    no_c = GreedyTok({i: t for i, t in TOKENS.items() if t != "c"})        # " c" stays a token, "c" alone does not
    out = M.check_masks(setup_for(), no_c, TEXTS)
    assert out["char_level"]["no_token"] == 2 and out["char_level"]["no_token_examples"][0]["char"] == "c"
    assert out["mismatch"] == 0 and any("no one-character token" in f for f in M._mask_failures(out))


def test_an_error_in_the_random_walks_is_recorded_and_fails():
    s = setup_for()
    s.replay = SimpleNamespace(walk=s.replay.walk)                         # a replay without the matcher's calls
    out = M.check_masks(s, GreedyTok(TOKENS), TEXTS)
    assert out["random_walks"]["error"].startswith("AttributeError") and out["char_level"]["mismatch"] == 0
    assert any(f.startswith("random_walks: AttributeError") for f in M._mask_failures(out))


# ------------------------------------------------------------------------------------------ the tiny generation
def batch():
    return M._batch(random.Random(0), M.TINY["vocab_size"], M.TINY_STOPS)


def test_the_tiny_generation_is_in_trls_layout():
    b = batch()
    assert b == batch(), "seeded"
    P, C = b["P"], b["C"]
    assert (P, C, b["valid"]) == (6, 9, [9, 5, 0, 7])
    assert all(len(r) == P + C for r in b["ids"] + b["att"])
    assert [masked.valid_length(r[P:]) for r in b["att"]] == b["valid"]
    assert [sum(r[:P]) for r in b["att"]] == list(M.P_LENS), "prompts are left-padded"
    assert b["ids"][2][P:] == b["sampled"][2] and not any(b["att"][2][P:]), "a truncated row keeps its ids, mask 0"
    stops = set(M.TINY_STOPS)
    for sampled, sets, ends in zip(b["sampled"], b["allowed"], M.STOPS_AT_END):
        assert all(t in s for t, s in zip(sampled, sets, strict=True))
        assert (sampled[-1] in stops) == ends and not stops & set(sampled[:-1])
        if ends:
            assert stops <= set(sets[-1]), "both stop ids are allowed where a completion stops"
    cache = masked.AllowedCache(replay=None)
    for row, n, sets in zip(b["comp"], b["valid"], b["allowed"]):
        if n:
            cache.put(row[:n], sets[:n])
    build = masked.build_support([r[P:] for r in b["ids"]], b["valid"], cache)
    assert (build.violations, build.cache_miss, build.positions, build.forced) == ((), 0, 21, 1)


def test_the_tiny_vocabulary_spans_three_of_trls_tiles_and_the_supports_reach_the_last():
    cov = M.tile_coverage(batch(), M.TRL_VOCAB_TILE, M.TINY["vocab_size"])
    assert cov["vocab_tiles"] == 3 and M.TINY_STOPS[0] // M.TRL_VOCAB_TILE == 2
    assert cov["last_tile_support_positions"] > 0 and cov["last_tile_targets"] >= 3 and cov["multi_tile_positions"] > 0
    assert cov["forced_positions"] == 1
    one = M.tile_coverage(batch(), 1 << 17, M.TINY["vocab_size"])      # a tile wider than the vocabulary
    assert (one["vocab_tiles"], one["multi_tile_positions"]) == (1, 0), "what the old 512-id vocabulary gave"


def test_the_glue_takes_the_tiny_generation_through_check_generation():
    b = batch()
    setup = M.glue_setup(b)
    assert setup.path == "support_ids" and setup.stop_ids == M.TINY_STOPS and setup.nfa is None
    ctl = masked.MaskedPasses(setup, ops=None)
    ctl.begin(0)
    ctl.check_generation(M.generation_result(b, [[0.0] * len(s) for s in b["sampled"]]))
    rec = setup.stats[0]
    assert (rec["completions"], rec["terminated"], rec["violations"], rec["trl_sampling_mask"]) == (4, 3, 0, None)
    assert len(ctl.cache) == 4 and setup.replay.record["unknown"] == 0
    build = masked.build_support([r[b["P"]:] for r in b["ids"]], b["valid"], ctl.cache)
    assert (build.violations, build.cache_miss, build.positions) == ((), 0, 21)
    assert setup.replay.walk([1, 2, 3]).violation == 0, "ids the batch never drew are refused"
    labels = [masked.pass_label(False, 0), masked.pass_label(False, 1)] + [masked.pass_label(True, 2)] * len(
        M.LOSS_MICRO)
    assert labels == ["old", "ref", "loss", "loss"] == [p["label"] for p in good_glue()["passes"]]
    assert sorted(r for m in M.LOSS_MICRO for r in m) == [0, 1, 2, 3], "the loss micro-batches hold every row once"


def trl_method(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False,
               compute_aux_loss=False, pixel_values=None, image_grid_thw=None, num_images=None,
               pixel_attention_mask=None, spatial_shapes=None, num_tiles=None, image_sizes=None, token_type_ids=None,
               mm_token_type_ids=None, image_position_ids=None, sampling_mask=None):
    """TRL 1.17.1's _get_per_token_logps_and_entropies signature (grpo_trainer.py:1481-1501), for the binding."""


def test_the_glues_keyword_arguments_bind_to_trls_signature():
    sig = inspect.signature(trl_method)
    for kw in (M.OLD_KWARGS, M.LOSS_KWARGS):
        sig.bind(None, "model", "ids", "att", 9, sampling_mask=None, **kw)
    assert set(M.LOSS_KWARGS) == set(sig.parameters) - set(masked.LEAD) - {"batch_size", "sampling_mask"}, (
        "_compute_loss passes every other parameter")


# ------------------------------------------------------------------------------------------ the job
JOB = ROOT / "jobs" / "mask_preflight.sbatch"
D020 = ["runs/junk-13640325/summary/completions.jsonl", "runs/grpo-smoke-13657200/summary/a/rewards.jsonl",
        "runs/grpo-smoke-13657200/summary/b/rewards.jsonl"]


def test_the_job_runs_the_masked_config_and_ends_on_the_verdict():
    job = JOB.read_text()
    assert "-m lofi_rl.rl.mask_preflight --config configs/grpo_smoke_v2_grammar_masked.yaml" in job
    assert (ROOT / "configs" / "grpo_smoke_v2_grammar_masked.yaml").is_file()
    assert "MODE=policy source scripts/job_env.sh" in job and "--partition=main" in job
    assert '--out "$OUT" "${TEXTS[@]}"' in job and "skipped" not in job
    assert subprocess.run(["bash", "-n", str(JOB)], capture_output=True).returncode == 0


def test_d020s_files_hold_the_980_in_language_texts_the_verdict_requires():
    """The job's three files as pulled to the laptop, read as the preflight reads them, against the masked config's
    own regex (measured here 2026-11-02: 980 rows, all in the language, none refused)."""
    import re

    from lofi_rl.rl.config import load_config

    files = [ROOT / rel for rel in D020]
    for p in files:
        if not p.is_file():
            pytest.skip(f"{p.relative_to(ROOT)} is not on this machine (make pull brings it to the laptop)")
    rx = re.compile(load_config(ROOT / "configs" / "grpo_smoke_v2_grammar_masked.yaml").structured_outputs["regex"])
    problems: list[str] = []
    read: dict = {}
    rows = M.read_texts(files, problems, read)
    assert problems == [] and [r["texts"] for r in read.values()] == [500, 320, 160]
    by_source: dict[str, int] = {}
    for src, text in rows:
        if rx.fullmatch(text):
            by_source[src.rsplit(":", 1)[0]] = by_source.get(src.rsplit(":", 1)[0], 0) + 1
    assert M._text_failures({"texts_files": [str(p) for p in files], "texts_read": read,
                             "masks": {"in_language_by_source": by_source}}) == []
    assert sum(by_source.values()) == M.MIN_TEXTS


def file_check(scr: Path) -> subprocess.CompletedProcess:
    """The job's own lines that gather the --texts files, run against a scratch directory."""
    job = JOB.read_text()
    block = job[job.index("TEXTS=()"):job.index("\ndone\n") + len("\ndone\n")]
    script = f"set -euo pipefail\nREMOTE_SCR={shlex.quote(str(scr))}\n{block}printf '%s\\n' \"${{TEXTS[@]}}\"\n"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_the_job_passes_d020s_three_files_and_stops_without_one(tmp_path):
    for rel in D020:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("{}\n")
    ok = file_check(tmp_path)
    assert ok.returncode == 0 and ok.stdout.split("\n")[:-1] == [x for rel in D020 for x in ("--texts",
                                                                                             str(tmp_path / rel))]
    (tmp_path / D020[1]).unlink()
    gone = file_check(tmp_path)
    assert gone.returncode == 1
    assert gone.stdout.strip() == (f"MASK_PREFLIGHT_FAILED {tmp_path / D020[1]} is not on scratch: the preflight "
                                   "walks all of D-020's 980 completions")
