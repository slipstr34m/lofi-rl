"""Shared test setup. No test may reach a live scorer: an exported $LOFI_SCORER_SOCK turns the reward hook on for
every make_reward a test builds (lofi_rl.rl.reward; docs/SCORER.md), so it is removed for each test. A test that
needs it sets it itself with monkeypatch, which runs after this fixture."""
import pytest


@pytest.fixture(autouse=True)
def _no_live_scorer(monkeypatch):
    monkeypatch.delenv("LOFI_SCORER_SOCK", raising=False)
