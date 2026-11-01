"""The copy penalty P_copy (src/lofi_rl/rl/copy_penalty.py; docs/design/taste_pipeline.md §5; D-020, Before GRPO
run 1) and the reward's opt-in use of it (src/lofi_rl/rl/reward.py):
- the ramp: 0 at J <= 0.6, linear to 0.3 at J >= 0.8;
- J is lofi_rl.rl.junk's measure: equal to junk.score_row's copy_any_jaccard (v2, all 9 examples) and copy_jaccard
  (v1, its one example) on every loop the repo holds and on edited copies of the examples;
- the references are every example the prompt version can show, plus the pool;
- in the reward, P_copy is one more penalty term, logged per completion and per step, kept by a failed gate and a
  G2, and absent (not even 0.0) when the reward is built without it."""
import dataclasses
import json
from pathlib import Path

import pytest

from helpers_rl import fake_verify
from lofi_rl import prompt, prompt_v2
from lofi_rl.brief import ENERGIES, MOOD_NAMES, Brief
from lofi_rl.palette import load_palette
from lofi_rl.rl import junk
from lofi_rl.rl.copy_penalty import (J_FREE, J_FULL, P_MAX, CopyPenalty, copy_form, prompt_references, ramp)
from lofi_rl.rl.reward import make_reward

ROOT = Path(__file__).resolve().parents[1]
PALETTE_PATH = ROOT / "palettes" / "dev_gu.yaml"
EOS = 151645


@pytest.fixture(scope="module")
def pal():
    return load_palette(PALETTE_PATH, verify=False)


def _repo_loops() -> list[tuple[str, str]]:
    files = sorted((ROOT / "tests" / "fixtures" / "loops").glob("*.ls")) + sorted((ROOT / "shootouts" / "loops").glob("*.ls")) \
        + sorted((ROOT / "drafts" / "v1").glob("*.ls"))
    return [(str(p.relative_to(ROOT)), p.read_text()) for p in files]


def _edited(pal) -> list[tuple[str, str]]:
    """Copies of the examples with a line or two changed, dropped, or written in v0's layout (two spaces before each
    newline, no final one) or fenced, so J falls on every part of the ramp."""
    out = []
    for ex_id, text in prompt_v2.reference_texts(pal).items():
        lines = text.splitlines()
        out.append((f"{ex_id}/v0-layout", "  \n".join(lines)))
        out.append((f"{ex_id}/fenced", "```\n" + text + "```\n"))
        out.append((f"{ex_id}/no-edits", "\n".join(l for l in lines if not l.startswith("@")) + "\n"))
        out.append((f"{ex_id}/new-B", "\n".join(lines[:-1] + ["B: w1 | w3 | w5 | w1"]) + "\n"))
        out.append((f"{ex_id}/new-A-B", "\n".join(lines[:-2] + ["A: h1 h3 | w5 | h.6 q5 | w1", "B: w1 | w3 | w5 | w1"]) + "\n"))
        out.append((f"{ex_id}/half", "\n".join(lines[: len(lines) // 2]) + "\n"))
    return out


# ------------------------------------------------------------------------------------------ the ramp
@pytest.mark.parametrize("j, want", [(0.0, 0.0), (0.5, 0.0), (0.6, 0.0), (0.65, 0.075), (0.7, 0.15), (0.75, 0.225),
                                     (0.8, 0.3), (0.85, 0.3), (1.0, 0.3)])
def test_the_ramp_is_zero_at_0_6_and_rises_linearly_to_0_3_at_0_8(j, want):
    assert ramp(j) == pytest.approx(want, abs=1e-12)
    assert (J_FREE, J_FULL, P_MAX) == (0.6, 0.8, 0.3), "taste_pipeline §5's numbers"


def test_the_ramps_endpoints_are_exact_and_it_never_falls():
    assert ramp(0.6) == 0.0 and ramp(0.8) == 0.3 and ramp(0.8 - 1e-15) <= 0.3
    js = [i / 1000 for i in range(1001)]
    assert all(ramp(a) <= ramp(b) for a, b in zip(js, js[1:]))


# ------------------------------------------------------------------------------------------ junk's measure
def _row(i: int, text: str, brief: Brief, finish: str = "stop") -> dict:
    return {"i": i, "brief": junk.brief_dict(brief), "text": text, "finish_reason": finish, "n_tokens": 300}


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_j_is_junks_copy_measure_on_every_repo_loop_and_edited_copies(pal, version):
    check = junk.CopyCheck(version, pal)
    copy = CopyPenalty.for_prompt(version, pal)
    texts = _repo_loops() + _edited(pal) + [(f"ref/{k}", t) for k, t in prompt_v2.reference_texts(pal).items()]
    assert len(texts) >= 100
    seen = set()
    for i, (name, text) in enumerate(texts):
        brief = Brief(energy=ENERGIES[i % 3], mood=MOOD_NAMES[(i // 3) % 3], bpm=82, tonic=i % 12)
        s = junk.score_row(_row(i, text, brief), pal, check)
        got = copy(text, brief, truncated=False)
        want = s["copy_any_jaccard"] if version == "v2" else s["copy_jaccard"]
        assert round(got.j, 4) == want, name
        assert got.nearest == (s["copy_any_source"] if version == "v2" else s["copy_reference"]), name
        assert got.p_copy == ramp(got.j)
        seen.add("over" if got.j >= 0.8 else "ramp" if got.j > 0.6 else "free")
    assert seen == {"over", "ramp", "free"}, "the texts reach every part of the ramp"


def test_a_cut_completion_is_compared_by_its_plain_text_as_junk_compares_a_length_row(pal):
    check, copy = junk.CopyCheck("v2", pal), CopyPenalty.for_prompt("v2", pal)
    text = prompt_v2.reference_texts(pal)["l8_hopeful_mid"]
    brief = Brief(energy="mid", mood="wistful", bpm=80, tonic=3)
    s = junk.score_row(_row(0, text, brief, finish="length"), pal, check)
    got = copy(text, brief, truncated=True)
    assert round(got.j, 4) == s["copy_any_jaccard"] and copy_form(text, brief.mode, pal, True) == junk.plain_text(text)
    assert copy_form(text, brief.mode, pal, False) == text, "a passing loop: its canonical text without sax"


def test_a_parser_crash_is_a_failed_gate_compared_by_its_plain_text(pal, monkeypatch):
    from lofi_rl.rl import copy_penalty as cp

    def crash(*a, **k):
        raise RuntimeError("parser bug")
    monkeypatch.setattr(cp, "parse", crash)
    assert copy_form("vamp: Im9 IV13 span 2  \n", "dorian", pal) == junk.plain_text("vamp: Im9 IV13 span 2  \n")


# ------------------------------------------------------------------------------------------ the references
def test_v1_compares_with_its_one_example_and_v2_with_all_nine(pal):
    assert prompt_references("v1", pal) == {"example": prompt.example_text(pal)}
    v2 = prompt_references("v2", pal)
    assert list(v2) == list(prompt_v2.reference_texts(pal)) and len(v2) == 9
    every_shown = {prompt_v2.example_id(Brief(energy=e, mood=m, bpm=82, tonic=0)) for e in ENERGIES for m in MOOD_NAMES}
    assert every_shown == set(v2), "across its prompts the policy sees all 9"
    assert v2["example"] == prompt.example_text(pal), "the mid/wistful slot is v1's example"
    with pytest.raises(ValueError, match="v3"):
        prompt_references("v3", pal)


def test_the_pool_adds_one_reference_per_file_in_its_copy_form(pal, tmp_path):
    draft = ROOT / "drafts" / "v1" / "d01_groove.ls"
    (tmp_path / "pool").mkdir()
    (tmp_path / "pool" / "d01.ls").write_text(draft.read_text())
    copy = CopyPenalty.for_prompt("v2", pal, [tmp_path / "pool" / "d01.ls"], root=tmp_path)
    assert list(copy.texts)[-1] == "pool/d01.ls" and len(copy.texts) == 10 and copy.pool == ("pool/d01.ls",)
    brief = Brief(energy="low", mood="melancholy", bpm=70, tonic=5)
    got = copy(draft.read_text(), brief)
    assert got.j == 1.0 and got.nearest == "pool/d01.ls" and got.p_copy == 0.3
    s = copy.summary()
    assert s["pool"] == {"files": 1, "ids": ["pool/d01.ls"]} and s["references"] == 10 and len(s["sha256"]) == 9


# ------------------------------------------------------------------------------------------ in the reward
BRIEF = Brief(energy="mid", mood="wistful", bpm=82, tonic=2)


def _reward(tmp_path, pal, **kw):
    fake = fake_verify()
    return make_reward(pal, tmp_path / "rewards.jsonl", verify=fake, max_completion_length=8, eos_token_ids=[EOS],
                       **kw)


def _lines(tmp_path):
    return [json.loads(l) for l in (tmp_path / "rewards.jsonl").read_text().splitlines()]


def test_p_copy_is_one_more_penalty_term_logged_per_completion_and_per_step(tmp_path, pal):
    copy = CopyPenalty.for_prompt("v2", pal)
    reward = _reward(tmp_path, pal, copy=copy)
    example = prompt_v2.reference_texts(pal)["l2_melancholy_mid"]
    first13 = "\n".join(example.splitlines()[:13]) + "\n"          # through @8 comp: no A, no B
    texts = [example, first13, "theory=0.8 knobs=0.6 | not a loop", "gate=G4 p_fix=0.02 | " + example]
    out = reward(prompts=[[{"role": "user", "content": "x"}]] * 4,
                 completions=[[{"role": "assistant", "content": t}] for t in texts],
                 brief=[json.dumps(dataclasses.asdict(BRIEF))] * 4, completion_ids=[[1, EOS]] * 4)
    recs = _lines(tmp_path)
    js = [copy(t, BRIEF, False).j for t in texts]
    assert js[0] == 1.0 and 0.6 < js[1] < 0.8 and js[2] == 0.0 and js[3] >= 0.8, js
    assert recs[1]["p_copy"] == pytest.approx(0.3 * (js[1] - 0.6) / 0.2) and 0 < recs[1]["p_copy"] < 0.3
    for rec, t, r, j in zip(recs, texts, out, js):
        assert rec["copy_j"] == j and rec["p_copy"] == ramp(j) and rec["reward"] == pytest.approx(r)
        assert rec["penalty"] == pytest.approx(rec["p_sax"] + sum(rec["penalties"].values()) + rec["p_group"]
                                               + rec["p_fix"] + rec["p_copy"])
    assert recs[0]["copy_nearest"] == "l2_melancholy_mid" and recs[0]["p_copy"] == 0.3
    assert recs[3]["gate"] == "G4" and recs[3]["p_copy"] == ramp(js[3]), "a failed gate keeps P_copy"
    st = reward.stats[0]["copy"]
    assert st["p_copy_mean"] == pytest.approx(sum(ramp(j) for j in js) / 4) and st["j_max"] == 1.0
    assert st["at_least"]["0.8"] == sum(1 for j in js if j >= 0.8) and st["nearest"]["l2_melancholy_mid"] >= 1


def test_a_g2_completion_keeps_p_copy_while_its_parsed_penalties_are_zero(tmp_path, pal):
    reward = _reward(tmp_path, pal, copy=CopyPenalty.for_prompt("v1", pal))
    text = prompt.example_text(pal) + "@9 x"
    reward(prompts=[[{"role": "user", "content": "x"}]], completions=[[{"role": "assistant", "content": text}]],
           brief=[json.dumps(dataclasses.asdict(BRIEF))], completion_ids=[[1] * 8])
    rec = _lines(tmp_path)[0]
    assert rec["gate"] == "G2" and rec["truncated"] is True and rec["p_fix"] == rec["p_sax"] == 0.0
    j = CopyPenalty.for_prompt("v1", pal)(text, BRIEF, truncated=True).j
    assert rec["copy_j"] == j and rec["p_copy"] == ramp(j) and rec["penalty"] == pytest.approx(rec["p_copy"])


def test_a_verifier_error_scores_zero_with_no_copy_reading(tmp_path, pal):
    reward = _reward(tmp_path, pal, copy=CopyPenalty.for_prompt("v1", pal))
    out = reward(prompts=[None], completions=["raise"], brief=[json.dumps(dataclasses.asdict(BRIEF))])
    rec = _lines(tmp_path)[0]
    assert out == [0.0] and rec["gate"] == "error" and (rec["p_copy"], rec["copy_j"], rec["copy_nearest"]) == (0.0, None, None)


def test_a_copy_penalty_failure_is_an_error_unless_strict(tmp_path, pal):
    class Broken:
        def __call__(self, text, brief, truncated=None):
            raise RuntimeError("copy bug")
    reward = _reward(tmp_path, pal, copy=Broken())
    out = reward(prompts=[None], completions=["theory=0.9 | x"], brief=[json.dumps(dataclasses.asdict(BRIEF))])
    assert out == [0.0] and "copy_penalty: RuntimeError: copy bug" in _lines(tmp_path)[0]["error"]
    strict = _reward(tmp_path / "s", pal, copy=Broken(), strict=True)
    with pytest.raises(RuntimeError, match="copy bug"):
        strict(prompts=[None], completions=["x"], brief=[json.dumps(dataclasses.asdict(BRIEF))])


def test_without_the_copy_penalty_the_log_has_no_copy_field(tmp_path, pal):
    reward = _reward(tmp_path, pal)
    reward(prompts=[None], completions=[prompt.example_text(pal)], brief=[json.dumps(dataclasses.asdict(BRIEF))])
    rec = _lines(tmp_path)[0]
    assert not {"p_copy", "copy_j", "copy_nearest", "finish", "in_language", "example_id"} & set(rec)
    assert "copy" not in reward.stats[0] and "in_language" not in reward.stats[0]


def test_trls_hook_gets_p_copy_only_when_it_is_on(tmp_path, pal):
    seen = []
    reward = _reward(tmp_path, pal, copy=CopyPenalty.for_prompt("v1", pal))
    reward(prompts=[None], completions=[prompt.example_text(pal)], brief=[json.dumps(dataclasses.asdict(BRIEF))],
           log_metric=lambda name, value: seen.append((name, value)))
    assert ("lofi/p_copy", 0.3) in seen
