"""The reward hook (src/lofi_rl/rl/reward.py with lofi_rl.scoring; docs/SCORER.md §10):
- without a scorer the rewards, the log and the per-step stats are byte-for-byte the golden recorded before the
  hook existed (tests/golden/reward_no_scorer.json, main 96bd121);
- with a scorer: G = G_sym x the server's G (hard x [PQ >= tau] x clip(P_acc / 0.5, 0.25, 1)), the server's
  r_ce, r_theme, r_taste in r, every new field in rewards.jsonl, §5's CE rule only when CE is not validated, and
  scorer failures raise whatever `strict` says;
- end to end: the real ScorerClient against the real server (fake backends) over a Unix socket.
The fake verifier (helpers_rl) reads each completion's sub-rewards from its own text."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from helpers_rl import fake_verify
from helpers_scoring import GOLDEN, PALETTE, run_fixed_batch, short_sock_dir
from lofi_rl.rl import curriculum
from lofi_rl.rl.reward import CE_RULE, KNOBS_WITHOUT_CE, make_reward, resolve_scorer
from lofi_rl.scoring import protocol
from lofi_rl.scoring.client import ENV_SOCK, ScorerClient, ScorerError
from lofi_rl.palette import load_palette

FULL = "theory=0.8 knobs=0.6 p_sax=0.05 bars=0.1 len=0.02 p_fix=0.04 repairs=2"
BRIEF = {"energy": "mid", "mood": "wistful", "bpm": 82, "tonic": 2, "scene": "city at night"}
VERS = {"reward_ver": protocol.REWARD_VER, "feat_ver": "fake1.w4.src-0", "rm_ver": "null", "render_ver": "render-cfg-x"}


@pytest.fixture(autouse=True)
def _no_env_scorer(monkeypatch):
    monkeypatch.delenv(ENV_SOCK, raising=False)


# ------------------------------------------------------------------------------------------ no scorer
def test_without_a_scorer_rewards_log_and_stats_are_byte_for_byte_the_golden(tmp_path):
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    got = run_fixed_batch(make_reward, tmp_path / "rewards.jsonl")
    assert got["rewards"] == want["rewards"]
    assert got["log"] == want["log"], "rewards.jsonl changed byte for byte"
    assert got["stats"] == want["stats"] and got["metrics"] == want["metrics"]


def test_an_empty_env_var_and_scorer_false_mean_no_scorer(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_SOCK, "")
    assert resolve_scorer(None) is None and resolve_scorer(False) is None
    got = run_fixed_batch(make_reward, tmp_path / "a.jsonl")
    assert got["log"] == json.loads(GOLDEN.read_text(encoding="utf-8"))["log"]
    monkeypatch.setenv(ENV_SOCK, "/tmp/x.sock")
    c = resolve_scorer(None)
    assert isinstance(c, ScorerClient) and c.sock == "/tmp/x.sock" and not c.allow_fake
    assert resolve_scorer(False) is None and resolve_scorer("/tmp/y.sock").sock == "/tmp/y.sock"


# ------------------------------------------------------------------------------------------ a stand-in client
class StubScorer:
    """The client interface the reward uses, answering from a table: text -> server item fields."""

    def __init__(self, answers, ce=False, fail=None):
        self.answers, self.ce, self.fail = answers, ce, fail
        self.pinned = dict(VERS)
        self.waits, self.calls = 0, []
        self.terms = {"pq_gate": True, "pacc_gate": True, "ce": ce, "theme": True, "taste": True}

    def wait_healthy(self, timeout):
        self.waits += 1
        return {}

    @property
    def ce_validated(self):
        return self.ce

    def score(self, items, *, palette_sha256, step=None):
        self.calls.append((list(items), palette_sha256, step))
        if self.fail:
            raise self.fail
        out = []
        for it in items:
            a = self.answers[it["text"]]
            hard = a.get("hard", 1)
            out.append({"i": it["i"], "clip_id": a.get("clip_id", "c" + "0" * 16), "hard": hard, "G": a["G"],
                        "gate": None if hard else a.get("gate", "G7"),
                        "r": {"ce": a.get("ce", 0.0), "theme": a.get("theme", 0.0), "taste": a.get("taste", 0.0)},
                        "sig": {k: (None if not hard else 1.0) for k in protocol.SIG_KEYS},
                        "diag": {**{k: None for k in protocol.DIAG_KEYS}, "p_acc": a.get("p_acc"),
                                 "pq_factor": a.get("pq_factor", 1.0), "disabled": [], "cache": "miss"},
                        "t_ms": {}})
        return {"protocol": protocol.PROTOCOL, **VERS, "terms": self.terms, "items": out,
                "batch": {"n": len(out), "unique": len(out), "rendered": len(out), "cache": {}, "gates": {},
                          "t_ms": {}}}


def _call(reward, texts, step=0, **kw):
    return reward(prompts=[[{"role": "user", "content": "p"}]] * len(texts),
                  completions=[[{"role": "assistant", "content": t}] for t in texts],
                  brief=[json.dumps(BRIEF)] * len(texts), brief_id=["596:0"] * len(texts),
                  trainer_state=SimpleNamespace(global_step=step), **kw)


def _lines(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def test_g_is_the_symbolic_gate_times_the_servers_gate_and_audio_terms_fill_r(tmp_path):
    answers = {FULL: {"G": 0.8, "ce": 0.3, "theme": 0.7, "taste": 0.6, "p_acc": 0.4, "clip_id": "c" + "a" * 16},
               "theory=0.5 knobs=0.5 | pq": {"G": 0.0, "pq_factor": 0.0, "theme": 0.9},
               "theory=0.5 knobs=0.5 | g7": {"G": 0.0, "hard": 0, "gate": "G7"}}
    stub = StubScorer(answers, ce=True)
    reward = make_reward(PALETTE, tmp_path / "r.jsonl", verify=fake_verify(), scorer=stub)
    texts = [FULL, "theory=0.5 knobs=0.5 | pq", "theory=0.5 knobs=0.5 | g7", "gate=G4 p_fix=0.02"]
    out = _call(reward, texts, step=0)
    w = curriculum.weights(0)                                   # CE validated: the curriculum's weights
    full = 0.8 * (w["theory"] * .8 + w["knobs"] * .6 + w["ce"] * .3 + w["theme"] * .7 + w["taste"] * .6)
    assert out == pytest.approx([full - .21, 0.0, 0.0, -0.02])
    a, pq, g7, g4 = _lines(tmp_path / "r.jsonl")
    assert (a["G"], a["G_sym"], a["gate_ok"], a["gate"]) == (pytest.approx(0.8), 1, True, None)
    assert a["r"] == pytest.approx({"theory": .8, "taste": .6, "knobs": .6, "ce": .3, "theme": .7})
    assert a["clip_id"] == "c" + "a" * 16 and a["audio"]["diag"]["p_acc"] == 0.4 and a["w_rule"] is None
    assert a["scorer"] == VERS and a["w"] == pytest.approx(w)
    assert (pq["G"], pq["gate"], pq["gate_ok"], pq["r"]["theme"]) == (0.0, "PQ", False, 0.9)
    assert (g7["G"], g7["gate"], g7["audio"]["hard"]) == (0.0, "G7", 0)
    assert (g4["G"], g4["G_sym"], g4["gate"], g4["audio"], g4["clip_id"]) == (0.0, 0, "G4", None, None)
    sent = stub.calls[0][0]
    assert [x["i"] for x in sent] == [0, 1, 2], "only symbolic-gate passes are sent"
    assert sent[0]["brief"] == BRIEF and sent[0]["brief_id"] == "596:0" and stub.calls[0][2] == 0
    assert stub.calls[0][1] == load_palette(PALETTE, verify=False).sha256
    st = reward.stats[0]
    assert st["audio"]["sent"] == 3 and st["audio"]["G_mean"] == pytest.approx(0.8 / 4)
    assert st["audio"]["pq_fail"] == 1 and st["audio"]["gates"] == {"G7": 1}
    assert st["gate_rate"] == 0.25 and st["r_mean"]["theme"] == pytest.approx(0.7), "means over G > 0"
    assert stub.waits == 1
    _call(reward, [FULL], step=1)
    assert stub.waits == 1, "wait_healthy only before the first call"


def test_the_ce_rule_applies_only_when_the_server_reports_ce_not_validated(tmp_path):
    answers = {FULL: {"G": 1.0, "ce": 0.9}}
    off = make_reward(PALETTE, tmp_path / "off.jsonl", verify=fake_verify(), scorer=StubScorer(answers, ce=False))
    on = make_reward(PALETTE, tmp_path / "on.jsonl", verify=fake_verify(), scorer=StubScorer(answers, ce=True))
    for step in (0, 15, 40):
        _call(off, [FULL], step=step)
        _call(on, [FULL], step=step)
    for rec in _lines(tmp_path / "off.jsonl"):
        w = curriculum.weights(rec["step"])
        assert rec["w"] == pytest.approx({**w, "ce": 0.0, "knobs": KNOBS_WITHOUT_CE}) and rec["w_rule"] == CE_RULE
        assert sum(rec["w"].values()) == pytest.approx(1.0)
        assert list(rec["w"]) == list(curriculum.TERMS)
    for rec in _lines(tmp_path / "on.jsonl"):
        assert rec["w"] == pytest.approx(curriculum.weights(rec["step"])) and rec["w_rule"] is None
    assert off.stats[0]["audio"]["w_rule"] == CE_RULE and off.stats[0]["w"]["ce"] == 0.0


@pytest.mark.parametrize("strict", [False, True])
def test_a_scorer_failure_raises_whatever_strict_says_and_logs_nothing(tmp_path, strict):
    stub = StubScorer({FULL: {"G": 1.0}}, fail=ScorerError("POST /score answered 500: boom", status=500))
    reward = make_reward(PALETTE, tmp_path / "r.jsonl", verify=fake_verify(), scorer=stub, strict=strict)
    with pytest.raises(ScorerError, match="500"):
        _call(reward, [FULL])
    assert not (tmp_path / "r.jsonl").exists() or (tmp_path / "r.jsonl").read_text() == ""
    assert reward.stats == {}


def test_a_batch_with_no_symbolic_pass_never_calls_the_scorer(tmp_path):
    stub = StubScorer({})
    reward = make_reward(PALETTE, tmp_path / "r.jsonl", verify=fake_verify(), scorer=stub)
    out = _call(reward, ["gate=G3 p_fix=0.02", "raise"])
    assert out == pytest.approx([-0.02, 0.0]) and stub.calls == []
    assert [r["audio"] for r in _lines(tmp_path / "r.jsonl")] == [None, None]
    assert reward.stats[0]["audio"]["sent"] == 0


def test_hack_metrics_and_clip_ids_reach_trls_hooks(tmp_path):
    seen, extra = [], []
    stub = StubScorer({FULL: {"G": 1.0, "p_acc": 0.7}, "theory=0.4 | low": {"G": 0.25, "p_acc": 0.0}})
    reward = make_reward(PALETTE, tmp_path / "r.jsonl", verify=fake_verify(), scorer=stub)
    _call(reward, [FULL, "gate=G1", "theory=0.4 | low"], log_metric=lambda k, v: seen.append(k),
          log_extra=lambda k, v: extra.append((k, v)))
    assert {"lofi/G_mean", "hack/pq", "hack/clap_margin", "hack/centroid_hz", "hack/p_acc",
            "hack/p_acc_lt_half"} <= set(seen)
    assert ("clip_id", ["c" + "0" * 16, "", "c" + "0" * 16]) in extra and reward.stats[0]["hook_errors"] == []
    a = reward.stats[0]["audio"]
    assert a["p_acc_below_half"] == 1 and a["diag_mean"]["p_acc_lt_half"] == 0.5, "P_acc 0.0 counts as below half"


# ------------------------------------------------------------------------------------------ end to end
def test_the_reward_scores_through_the_real_client_and_server_with_fake_backends(tmp_path):
    from helpers_scoring import fixture_text, make_scorer
    from lofi_rl.scoring.server import ScorerServer
    scorer = make_scorer(tmp_path)
    with short_sock_dir() as d:
        srv = ScorerServer(scorer, d / "s.sock", logf=lambda m: None)
        srv.bind()
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            client = ScorerClient(d / "s.sock", allow_fake=True)
            reward = make_reward(PALETTE, tmp_path / "r.jsonl", verify=fake_verify(), scorer=client)
            loop = fixture_text("example.ls")
            fake_ok = FULL + "\n"      # the fake verifier passes it; the real parser on the server does not
            out = _call(reward, [loop, loop, fake_ok], step=2)
        finally:
            srv.shutdown()
            t.join(5)
            srv.close()
    recs = _lines(tmp_path / "r.jsonl")
    assert len(out) == 3 and all(r["scorer"]["reward_ver"] == protocol.REWARD_VER for r in recs)
    a, b, c = recs
    assert a["clip_id"] == b["clip_id"] and a["audio"]["hard"] == 1 and b["audio"]["diag"]["cache"] == "dup"
    assert c["audio"]["hard"] == 0 and c["gate"] == "G1" and c["G"] == 0.0
    assert a["w_rule"] == CE_RULE, "no ce.json: CE not validated"
    assert a["r"]["theme"] == 0.0 and a["r"]["taste"] == 0.0 and a["r"]["ce"] == 0.0, "no calibration: all off"
    assert a["G"] == pytest.approx(1.0), "no tau_PQ and no RM: the gate is the render alone"
    assert set(a["audio"]["diag"]["disabled"]) == {"pq_gate", "pacc_gate", "ce", "theme", "taste"}
