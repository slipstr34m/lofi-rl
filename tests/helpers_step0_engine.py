"""Stand-ins for the step-0 generate step's world, for tests/test_step0_{generate,render}.py: a code tree its
pre-checks pass in, and a fake engine in junk.VllmEngine's interface (the laptop has no vLLM).

- `code_tree(dst)`: a copy of what the path file lists (src/, the data files with a small stand-in for the SoundFont,
  the env scripts, kit.env, a stand-in jobs/step0.sbatch) plus a docs/DECISIONS.md that holds D-023's mark, and the
  path file lofi_rl.step0.path writes for that tree. The repo's own D-023 and docs/prereg/D-023_step0_path.txt are
  written at the code commit, with Shresth's answers, so no test reads or writes them. The tree is only read by the
  pre-checks: the code that runs is this checkout's.
- `FakeEngine`: sampling(**kw) makes one FakeSP per call (n 1, as VllmEngine.sampling makes a SamplingParams);
  chat(convs, sps) takes one list of them per call and answers each request with text_of(cell, key), the key being
  the request's seed (seeds honoured) or a running count (seeds ignored).
- `draw_only(monkeypatch, briefs)`: generate draws those briefs' requests only, each keeping its place i in the full
  order, as tests/helpers_step0.py's `briefs=` writes them.

Sound claims: none.
"""
from __future__ import annotations

import shutil
from types import SimpleNamespace

import helpers_step0 as H
import lofi_rl.step0 as S
from lofi_rl import prompt_v2
from lofi_rl.brief import ENERGIES, MOOD_NAMES
from lofi_rl.brief_streams import ROOT
from lofi_rl.prompt import messages_sha256
from lofi_rl.rl import junk as J
from lofi_rl.step0 import path as P
from lofi_rl.step0 import requests as R

SF2 = "assets/soundfonts/generaluser-gs/GeneralUser-GS.sf2"
JOB_TEXT = "#!/bin/bash\n#SBATCH --job-name=lofi-step0\n# a stand-in for the job script, for these tests\n"
DECISIONS_TEXT = ("# Decisions (a stand-in for these tests)\n\n## D-023 The step-0 samples\n"
                  f"- **The mark** generate checks: `{S.MARK}`.\n")
# what VllmEngine.info["structured_outputs"] holds when vLLM takes the xgrammar pin (tests/test_junk.py's PINNED)
PINNED = {"requested": "xgrammar", "pin_accepted": True, "backend": "xgrammar",
          "source": "llm_engine.vllm_config.structured_outputs_config.backend", "applied": True}
INFO = {"engine": "fake", "seed": S.ENGINE_SEED, "structured_outputs": PINNED}
CELL_OF = {messages_sha256(prompt_v2.build_messages(J.cell_brief(e, m), H.palette())): f"{e}/{m}"
           for e in ENERGIES for m in MOOD_NAMES}         # a conversation's cell, by its messages' sha256


def code_tree(dst, decisions: str = DECISIONS_TEXT):
    """A code tree the pre-checks pass in (module doc)."""
    shutil.copytree(ROOT / "src", dst / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for rel in [*P.data_files(ROOT), "scripts/job_env.sh", "scripts/_kit_env.sh", "kit.env"]:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        if rel == SF2:
            (dst / rel).write_bytes(b"a stand-in for the SoundFont\n")
        else:
            shutil.copy(ROOT / rel, dst / rel)
    (dst / "jobs").mkdir()
    (dst / S.JOB).write_text(JOB_TEXT)
    (dst / S.DECISIONS).write_text(decisions)
    P.write(dst)
    return dst


class FakeSP(SimpleNamespace):
    """A stand-in SamplingParams: the keywords it was made with, n 1 as VllmEngine.sampling makes it."""

    def __repr__(self) -> str:
        return f"FakeSP(n={self.n}, seed={self.seed})"


def plain_text(cell: str, key) -> tuple[str, str, int, int]:
    return f"loop {cell} {key}\n", "stop", 100, S.PROMPT_TOKENS[cell]


class FakeEngine:
    """The engine interface step0.generate uses (junk.VllmEngine's): info, load_s, sampling(**kw), chat(convs, sps).
    A completion is text_of(cell, key): key is the request's seed, or, with seeds ignored, a running count. `clock`
    moves on `per_request_s` for each request of a call; call number `fail_at` raises."""

    def __init__(self, text_of=None, *, honour_seeds=True, clock=None, per_request_s=0.0, info=None, fail_at=None):
        self.text_of = text_of or plain_text
        self.honour, self.clock, self.per_request_s, self.fail_at = honour_seeds, clock, per_request_s, fail_at
        self.info, self.load_s = dict(INFO if info is None else info), 1.5
        self.made, self.calls, self.count = [], [], 0

    def sampling(self, **kw):
        sp = FakeSP(n=1, **kw)
        self.made.append(sp)
        self.info["sampling_params"] = repr(sp)         # as VllmEngine.sampling does, on every call
        return sp

    def chat(self, convs, sps):
        assert isinstance(sps, list) and len(sps) == len(convs), "one SamplingParams per request, as a list"
        self.calls.append((convs, sps))
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("the fake engine died")
        if self.clock is not None:
            self.clock[0] += self.per_request_s * len(convs)
        out = []
        for conv, sp in zip(convs, sps):
            self.count += 1
            text, finish, n_tok, p_tok = self.text_of(CELL_OF[messages_sha256(conv)],
                                                      sp.seed if self.honour else self.count)
            out.append(J.Completion(text=text, finish_reason=finish, n_tokens=n_tok, prompt_tokens=p_tok))
        return out


def draw_only(monkeypatch, briefs) -> tuple:
    """generate draws only these briefs' requests (module doc); gives them."""
    keep = set(briefs)
    H.all_requests()                                    # the helper's cache holds the full list first
    real = R.requests

    def some(*a, **kw):
        return tuple(r for r in real(*a, **kw) if r.brief_id in keep)
    monkeypatch.setattr(R, "requests", some)
    return tuple(r for r in real(streams=H.streams()) if r.brief_id in keep)
