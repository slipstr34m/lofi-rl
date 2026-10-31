"""The interim GRPO reward (src/lofi_rl/rl/reward.py) and its curriculum (src/lofi_rl/rl/curriculum.py),
against the fake verifier in tests/helpers_rl.py (lofi_rl.verify is another builder's module):

    R = G x sum_k w_k(t) r_k - (P_sax + penalties + group penalties + P_fix)

taste_pipeline.md §5 (weights, penalties), DSL.md §7 (G2 parses nothing), the task's interim rules
(theory .50 -> .10 over steps 0-30, knobs .15, taste/CE/theme 0 in the smoke)."""
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from helpers_rl import fake_verify
from lofi_rl.brief import Brief
from lofi_rl.rl import curriculum
from lofi_rl.rl.reward import NAME, SERVER_TERMS, make_reward

ROOT = Path(__file__).resolve().parents[1]
PALETTE = ROOT / "palettes" / "dev_gu.yaml"
BRIEF = Brief(energy="mid", mood="wistful", bpm=82, tonic=2)
BRIEF_JSON = json.dumps(dataclasses.asdict(BRIEF))
PROMPT_A = [{"role": "system", "content": "S"}, {"role": "user", "content": "Energy: mid. Mood: wistful."}]
PROMPT_B = [{"role": "system", "content": "S"}, {"role": "user", "content": "Energy: low. Mood: hopeful."}]
FULL = "theory=0.8 knobs=0.6 p_sax=0.05 bars=0.1 len=0.02 p_fix=0.04 repairs=2"   # penalties sum to 0.21
EOS = 151645


def _reward(tmp_path, **kw):
    fake = fake_verify()
    return make_reward(PALETTE, tmp_path / "rewards.jsonl", verify=fake, **kw), fake


def _call(reward, texts, prompts=None, as_messages=True, **kw):
    prompts = prompts or [PROMPT_A] * len(texts)
    comps = [[{"role": "assistant", "content": t}] for t in texts] if as_messages else list(texts)
    kw.setdefault("brief", [BRIEF_JSON] * len(texts))
    return reward(prompts=prompts, completions=comps, **kw)


def _lines(tmp_path):
    return [json.loads(l) for l in (tmp_path / "rewards.jsonl").read_text().splitlines()]


# ------------------------------------------------------------------------------------------ curriculum
def test_curriculum_moves_theory_from_half_to_a_tenth_over_30_steps_and_holds_knobs():
    w0, w15, w30 = curriculum.weights(0), curriculum.weights(15), curriculum.weights(30)
    assert w0 == pytest.approx({"theory": .50, "taste": .20, "knobs": .15, "ce": .05, "theme": .10})
    assert w30 == pytest.approx({"theory": .10, "taste": .60, "knobs": .15, "ce": .05, "theme": .10})
    assert w15["theory"] == pytest.approx(.30) and w15["taste"] == pytest.approx(.40)
    assert curriculum.weights(-3) == w0 and curriculum.weights(31) == w30 == curriculum.weights(10_000)
    th = [curriculum.weights(t)["theory"] for t in range(0, 31)]
    assert all(a > b for a, b in zip(th, th[1:])), "theory falls strictly, step by step, until step 30"
    assert all(b - a == pytest.approx(-0.4 / 30) for a, b in zip(th, th[1:])), "linearly"
    assert {curriculum.weights(t)["knobs"] for t in range(0, 40)} == {0.15}
    assert tuple(w0) == curriculum.TERMS == ("theory", "taste", "knobs", "ce", "theme")


# ------------------------------------------------------------------------------------------ the formula
@pytest.mark.parametrize("step, theory_w", [(0, .50), (15, .30), (30, .10), (45, .10)])
def test_reward_is_gate_times_weighted_terms_minus_every_penalty(tmp_path, step, theory_w):
    reward, _ = _reward(tmp_path)
    out = _call(reward, [FULL], trainer_state=SimpleNamespace(global_step=step))
    assert out == pytest.approx([theory_w * .8 + .15 * .6 - (.05 + .1 + .02 + 0 + .04)])
    rec = _lines(tmp_path)[0]
    assert rec["G"] == 1 and rec["gate_ok"] is True and rec["gate"] is None
    assert rec["positive"] == pytest.approx(theory_w * .8 + .15 * .6) and rec["penalty"] == pytest.approx(.21)


def test_a_failed_gate_zeroes_the_weighted_terms_but_keeps_the_penalties(tmp_path):
    reward, _ = _reward(tmp_path)
    out = _call(reward, ["gate=G4 theory=0.9 knobs=0.9 p_fix=0.02 bars=0.1"])
    assert out == pytest.approx([-(0.02 + 0.1)])
    rec = _lines(tmp_path)[0]
    assert (rec["G"], rec["gate_ok"], rec["gate"], rec["positive"]) == (0, False, "G4", 0.0)


def test_a_completion_cut_at_the_token_limit_is_gate_g2_and_parses_nothing(tmp_path):
    reward, _ = _reward(tmp_path, max_completion_length=8, eos_token_ids=[EOS])
    texts = [FULL + " | cut", FULL + " | ends", FULL + " | short"]
    ids = [[1] * 8, [1] * 7 + [EOS], [1] * 3]          # cut at the limit; EOS at the limit; short, no EOS
    out = _call(reward, texts, completion_ids=ids)
    full = .5 * .8 + .15 * .6 - .21
    assert out == pytest.approx([0.0, full, full])
    cut, ends, short = _lines(tmp_path)
    assert (cut["gate"], cut["G"], cut["truncated"], cut["n_tokens"]) == ("G2", 0, True, 8)
    assert cut["raw_gate"] is None and cut["p_fix"] == cut["p_sax"] == 0 and set(cut["penalties"].values()) == {0.0}
    assert ends["truncated"] is False and short["truncated"] is False and ends["gate"] is None
    unknown, _ = _reward(tmp_path / "u")          # no limit given: truncation is unknown, never guessed
    assert _call(unknown, [FULL], completion_ids=[[1] * 8]) == pytest.approx([full])


def test_taste_ce_and_theme_are_zero_in_the_smoke_while_their_weights_are_logged(tmp_path):
    reward, _ = _reward(tmp_path)
    _call(reward, [FULL])
    rec = _lines(tmp_path)[0]
    assert set(SERVER_TERMS) == {"taste", "ce", "theme"}
    assert all(rec["r"][k] == 0.0 for k in SERVER_TERMS)
    assert rec["w"] == pytest.approx(curriculum.weights(0)) and rec["w"]["taste"] == pytest.approx(.2)
    assert rec["r"]["theory"] == .8 and rec["r"]["knobs"] == .6


# ------------------------------------------------------------------------------------------ groups
def test_group_penalties_follow_each_prompt_even_when_the_batch_interleaves_them(tmp_path):
    reward, fake = _reward(tmp_path)
    texts = ["x | 1", "x | 1", "x | 1", "x | 3", "x | 2", "x | 3"]      # A B A B A B
    prompts = [PROMPT_A, PROMPT_B] * 3
    out = _call(reward, texts, prompts=prompts)
    base = .5 * .5 + .15 * .5
    assert out == pytest.approx([base, base, base - .3, base, base, base - .3])
    assert [g[0] for g in fake.group_calls] == [("x | 1", "x | 1", "x | 2"), ("x | 1", "x | 3", "x | 3")]
    recs = _lines(tmp_path)
    assert [r["group"] for r in recs] == [0, 1, 0, 1, 0, 1]
    assert [r["p_group"] for r in recs] == [0, 0, .3, 0, 0, .3]


def test_a_brief_id_column_keeps_two_rows_with_the_same_prompt_text_apart(tmp_path):
    reward, fake = _reward(tmp_path)
    out = _call(reward, ["x | 1"] * 4, brief_id=["596:0", "596:0", "596:1", "596:1"])
    base = .5 * .5 + .15 * .5
    assert out == pytest.approx([base, base - .3, base, base - .3])
    assert [len(g[0]) for g in fake.group_calls] == [2, 2]


# ------------------------------------------------------------------------------------------ the step
def test_the_step_comes_from_trainer_state_else_a_counter_that_resumes(tmp_path):
    reward, _ = _reward(tmp_path)
    for _ in range(3):
        _call(reward, [FULL])
    _call(reward, [FULL], trainer_state=SimpleNamespace(global_step=12))
    _call(reward, [FULL], global_step=20)
    recs = _lines(tmp_path)
    assert [(r["step"], r["step_source"]) for r in recs] == [
        (0, "counter"), (1, "counter"), (2, "counter"), (12, "trainer_state"), (20, "global_step")]
    assert recs[3]["w"]["theory"] == pytest.approx(.5 - .4 * 12 / 30)
    resumed, _ = _reward(tmp_path / "r", start_step=5)
    _call(resumed, [FULL]); _call(resumed, [FULL])
    assert [r["step"] for r in _lines(tmp_path / "r")] == [5, 6]
    assert sorted(reward.stats) == [0, 1, 2, 12, 20]


# ------------------------------------------------------------------------------------------ the log
def test_the_log_has_one_line_per_completion_with_every_sub_reward(tmp_path):
    reward, _ = _reward(tmp_path)
    texts = [FULL, "gate=G3 p_fix=0.02", "theory=1 knobs=0 | b", "theory=0 knobs=1 | c"]
    out = _call(reward, texts, trainer_state=SimpleNamespace(global_step=4), brief_id=["596:7"] * 4,
                completion_ids=[[1, EOS]] * 4)
    recs = _lines(tmp_path)
    assert len(recs) == len(texts) == len(out)
    need = {"step", "step_source", "call", "i", "group", "group_key", "brief_id", "brief", "n_tokens", "truncated",
            "gate_ok", "gate", "G", "w", "r", "theory_parts", "knob_parts", "p_sax", "penalties", "p_group", "p_fix",
            "repairs", "positive", "penalty", "reward", "digest", "text_sha256", "text"}
    for i, (rec, r, t) in enumerate(zip(recs, out, texts)):
        assert need <= set(rec), need - set(rec)
        assert rec["i"] == i and rec["text"] == t and rec["reward"] == pytest.approx(r)
        assert set(rec["w"]) == set(rec["r"]) == set(curriculum.TERMS)
        assert rec["brief"] == dataclasses.asdict(BRIEF) and rec["brief_id"] == "596:7" and rec["n_tokens"] == 2
        assert rec["reward"] == pytest.approx(rec["positive"] - rec["penalty"])
        assert rec["penalty"] == pytest.approx(rec["p_sax"] + sum(rec["penalties"].values()) + rec["p_group"] + rec["p_fix"])
    assert recs[0]["theory_parts"] == {"groove": .8, "sax": .8} and recs[0]["repairs"] == 2
    st = reward.stats[4]
    assert st["n"] == 4 and st["groups"] == 1 and st["gate_rate"] == .75 and st["gate_counts"] == {"G3": 1}
    assert st["reward_mean"] == pytest.approx(sum(out) / 4) and st["errors"] == 0 and st["truncated"] == 0
    assert st["r_mean"]["theory"] == pytest.approx((.8 + 1 + 0) / 3), "means over gate-passing completions"


def test_plain_string_completions_score_like_chat_messages(tmp_path):
    a, _ = _reward(tmp_path / "a")
    b, _ = _reward(tmp_path / "b")
    texts = [FULL, "gate=G1", "x | y"]
    assert _call(a, texts) == _call(b, texts, prompts=["prompt text"] * 3, as_messages=False)


@pytest.mark.parametrize("form", ["json", "dict", "brief"])
def test_the_brief_column_reaches_the_verifier_as_a_brief(tmp_path, form):
    reward, fake = _reward(tmp_path)
    value = {"json": BRIEF_JSON, "dict": dataclasses.asdict(BRIEF), "brief": BRIEF}[form]
    _call(reward, [FULL], brief=[value])
    text, brief, palette = fake.score_calls[0]
    assert text == FULL and brief == BRIEF and palette.id == "dev_gu"


def test_without_a_brief_per_completion_the_reward_refuses(tmp_path):
    reward, _ = _reward(tmp_path)
    with pytest.raises(ValueError, match="brief"):
        reward(prompts=[PROMPT_A], completions=[FULL])
    with pytest.raises(ValueError, match="brief"):
        _call(reward, [FULL, FULL], brief=[BRIEF_JSON])


# ------------------------------------------------------------------------------------------ failures
def test_a_verifier_error_scores_zero_and_is_logged_unless_strict(tmp_path):
    reward, _ = _reward(tmp_path)
    out = _call(reward, [FULL, "raise", "theory=nan"])
    assert out[0] == pytest.approx(.5 * .8 + .15 * .6 - .21) and out[1:] == [0.0, 0.0]
    recs = _lines(tmp_path)
    assert "ValueError: fake verifier failure" in recs[1]["error"] and recs[1]["gate"] == "error"
    assert "not finite" in recs[2]["error"] and reward.stats[0]["errors"] == 2
    strict, _ = _reward(tmp_path / "s", strict=True)
    with pytest.raises(ValueError, match="fake verifier failure"):
        _call(strict, ["raise"])


def test_the_reward_carries_the_name_trl_logs_it_under(tmp_path):
    reward, _ = _reward(tmp_path)
    assert reward.__name__ == NAME == "lofi_reward"


def test_trl_metric_hooks_are_used_when_passed_and_never_crash_the_reward(tmp_path):
    reward, _ = _reward(tmp_path)
    seen = []
    _call(reward, [FULL, "gate=G2"], log_metric=lambda name, value: seen.append((name, value)))
    names = dict(seen)
    assert names["lofi/G"] == .5 and names["lofi/r_theory"] == pytest.approx(.8) and "lofi/reward" in names

    def broken(*_a, **_k):
        raise TypeError("signature changed")
    out = _call(reward, [FULL], log_metric=broken, log_extra=broken)
    assert len(out) == 1 and "TypeError" in reward.stats[1]["hook_errors"][0]
