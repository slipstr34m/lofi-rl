"""Shared inputs for the scorer-server tests (src/lofi_rl/scoring/, docs/SCORER.md). Not a test module.

- `FIXED_CALLS` and `run_fixed_batch`: the fixed batch of the reward-hook golden. tests/golden/reward_no_scorer.json
  holds what lofi_rl.rl.reward wrote for it before the scorer hook existed (commit 96bd121, fake verifier from
  helpers_rl), so test_scoring_reward_hook can check that without a scorer the rewards, the log and the per-step
  stats are byte-for-byte unchanged.
- `short_sock_dir`: a temp dir short enough for an AF_UNIX path (macOS allows 104 bytes; pytest's tmp_path on
  this laptop is longer than that).
- `pid_alive`, `wait_gone`: whether a process this test did not start still runs (a zombie does not), and a
  bounded wait for it to end; `start_sleeper`: a render-pool task that starts a child process and waits on it.
- `serving`: a ScorerServer on a socket for the length of a block (two blocks on one socket: a restart).
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from helpers_rl import fake_verify
from lofi_rl.brief import Brief

ROOT = Path(__file__).resolve().parents[1]
PALETTE = ROOT / "palettes" / "dev_gu.yaml"
GOLDEN = ROOT / "tests" / "golden" / "reward_no_scorer.json"
EOS = 151645
BRIEF_A = Brief(energy="mid", mood="wistful", bpm=82, tonic=2)
BRIEF_B = Brief(energy="low", mood="hopeful", bpm=75, tonic=9)
JA, JB = json.dumps(dataclasses.asdict(BRIEF_A)), json.dumps(dataclasses.asdict(BRIEF_B))
PA = [{"role": "system", "content": "S"}, {"role": "user", "content": "Energy: mid. Mood: wistful."}]
PB = [{"role": "system", "content": "S"}, {"role": "user", "content": "Energy: low. Mood: hopeful."}]
FULL = "theory=0.8 knobs=0.6 p_sax=0.05 bars=0.1 len=0.02 p_fix=0.04 repairs=2"

# Three calls: TRL's trainer_state at step 3 with two brief_id groups, G2, a failed gate, an error and a copy;
# the reward's own counter with plain-string completions grouped by prompt hash; global_step 31 with log hooks.
FIXED_CALLS = (
    dict(texts=[FULL, FULL, "gate=G4 p_fix=0.02 bars=0.1", "theory=1 knobs=0 | b",
                "theory=0.2 knobs=0.9 p_sax=0.1 | c", FULL + " | cut", "raise", "theory=nan"],
         prompts=[PA] * 4 + [PB] * 4, briefs=[JA] * 4 + [JB] * 4, brief_id=["596:0"] * 4 + ["596:1"] * 4,
         ids=[[1, EOS]] * 5 + [[1] * 8] + [[1, 2, EOS]] * 2, trainer_state=3),
    dict(texts=["x | 1", "x | 1", "x | 2", "theory=0.7 knobs=0.7 len=0.05"], prompts=["p one", "p one", "p two", "p two"],
         briefs=[JA, JA, JB, JB], brief_id=None, ids=None, trainer_state=None, plain=True),
    dict(texts=[FULL, "gate=G2 p_fix=0.02", "theory=0.9 knobs=0.9 | z"], prompts=[PA] * 3, briefs=[JA] * 3,
         brief_id=["596:9"] * 3, ids=[[1, EOS]] * 3, trainer_state=None, global_step=31, hooks=True),
)


def run_fixed_batch(make_reward, log_path: Path, **extra) -> dict:
    """Run FIXED_CALLS through make_reward(PALETTE, log_path, verify=fake, max_completion_length=8, ...): the
    rewards (repr of each float), the log's bytes and the per-step stats without their wall-clock t_ms."""
    fake = fake_verify()
    reward = make_reward(PALETTE, log_path, verify=fake, max_completion_length=8, eos_token_ids=[EOS], **extra)
    rewards, metrics = [], []
    for c in FIXED_CALLS:
        comps = list(c["texts"]) if c.get("plain") else [[{"role": "assistant", "content": t}] for t in c["texts"]]
        kw = {"brief": list(c["briefs"])}
        if c["brief_id"] is not None:
            kw["brief_id"] = list(c["brief_id"])
        if c["ids"] is not None:
            kw["completion_ids"] = [list(x) for x in c["ids"]]
        if c["trainer_state"] is not None:
            kw["trainer_state"] = SimpleNamespace(global_step=c["trainer_state"])
        if c.get("global_step") is not None:
            kw["global_step"] = c["global_step"]
        if c.get("hooks"):
            kw["log_metric"] = lambda name, value: metrics.append([name, value])
            kw["log_extra"] = lambda name, values: metrics.append([name, list(values)])
        rewards.append([repr(x) for x in reward(prompts=list(c["prompts"]), completions=comps, **kw)])
    stats = {str(k): {kk: vv for kk, vv in v.items() if kk != "t_ms"} for k, v in sorted(reward.stats.items())}
    return {"rewards": rewards, "log": log_path.read_text(encoding="utf-8"), "metrics": metrics,
            "stats": json.loads(json.dumps(stats, sort_keys=True, default=str))}


class short_sock_dir:
    """A context manager: a fresh temp dir with a short path, removed on exit."""

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="lofi-s-"))
        return self.path

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


@contextlib.contextmanager
def serving(scorer, sock):
    """`scorer` on `sock` until the block ends: a server stopped and another started on the same socket is a
    restart, as far as a client can tell."""
    import threading
    from lofi_rl.scoring.server import ScorerServer
    srv = ScorerServer(scorer, sock, logf=lambda m: None)
    srv.bind()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        t.join(5)
        srv.close()


def pid_alive(pid: int) -> bool:
    """Whether process `pid` still runs: False once it has exited, a zombie included (`ps` says Z)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


def wait_gone(pid: int, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while pid_alive(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def start_sleeper(pid_file: str) -> int:
    """A render-pool task: start `sleep 60` as this worker's child, write its pid to pid_file, wait on it."""
    child = subprocess.Popen(["sleep", "60"])
    Path(pid_file).write_text(str(child.pid))
    return child.wait()


# ------------------------------------------------------------------------------------------ scorer fixtures
FIX = ROOT / "tests" / "fixtures" / "loops"
PASSING = ("example.ls", "leniency_lower_x.ls", "one_bar_base.ls", "shaker_edits.ls", "span1_dotted.ls")


def fixture_text(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def canon_sha_of(text: str, brief: Brief = BRIEF_A) -> str:
    """sha256 of a text's canonical form: the key the fakes (lofi_rl.scoring.fakes) take behaviours by."""
    from lofi_rl.dsl import canon, parse
    from lofi_rl.palette import load_palette
    res = parse(text, vocab=load_palette(PALETTE, verify=False).vocab(), mode=brief.mode)
    assert res.gate.ok, res.gate
    import hashlib
    return hashlib.sha256(canon(res.ast).encode("utf-8")).hexdigest()


def variant(text: str, k: int) -> str:
    """A distinct valid loop: the k-th single-cell change of the first `kick:` grid (k < 16)."""
    lines = text.splitlines()
    for n, line in enumerate(lines):
        if line.startswith("kick:"):
            head, grids = line.split(":", 1)
            cells = list(grids.split()[0])
            pos = 1 + (k % 15)
            cells[pos] = "." if cells[pos] != "." else "x"
            lines[n] = f"{head}: {''.join(cells)} " + " ".join(grids.split()[1:])
            return "\n".join(lines) + "\n"
    raise ValueError("no kick line")


def make_scorer(tmp_path: Path, *, calib_dir: Path | None = None, rm=None, rm_dir: Path | None = None,
                renderer=None, featurizer=None, cache_root: Path | None = None, lru: int = 4096):
    """A Scorer on the dev palette with fake backends (lofi_rl.scoring.fakes) unless given others."""
    from lofi_rl.palette import load_palette
    from lofi_rl.scoring import calib, fakes, rm_api, theme
    from lofi_rl.scoring.cache import FeatureCache
    from lofi_rl.scoring.core import Scorer
    from lofi_rl.scoring.render_pool import render_config
    renderer = renderer or fakes.FakeRenderer()
    featurizer = featurizer or fakes.FakeFeaturizer(render=renderer.name)
    cfg = theme.load_config()
    return Scorer(palette=load_palette(PALETTE, verify=False), renderer=renderer,
                  featurizer=featurizer, theme=theme.Theme(cfg, featurizer.encode_text),
                  calib=calib.load(calib_dir, theme_sha256=cfg.sha256),
                  rm=rm if rm is not None else rm_api.load_rm(rm_dir),
                  cache=FeatureCache(cache_root, featurizer.feat_ver, lru), render_cfg=render_config(),
                  backend="fake")
