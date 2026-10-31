"""Audit aggregates over a finished batch (src/lofi_rl/ui/audit.py): statistics against known values, anchors,
repeats, position of best, mode tallies, and no hidden label or real clip id in the per-round rows."""
import json
import math
import random
from datetime import date, datetime, timedelta, timezone

import pytest

from lofi_rl.ui.audit import aggregate, agreement, chi2_sf, chi2_uniform, implied_pairs, wilson
from lofi_rl.ui.manifest import build_demo, load_batch, parse_hidden, parse_public
from lofi_rl.ui.votes import Ledger, read_events

FAST = dict(bpm=240.0, sr=8000)


def test_chi2_and_wilson_match_known_values():
    for x, df, p in [(3.841458820694124, 1, 0.05), (5.991464547107979, 2, 0.05), (7.814727903251178, 3, 0.05),
                     (11.344866730144373, 3, 0.01), (0.0, 3, 1.0)]:
        assert math.isclose(chi2_sf(x, df), p, rel_tol=1e-6, abs_tol=1e-12), (x, df)
    closed = {1: lambda x: math.erfc(math.sqrt(x / 2)), 2: lambda x: math.exp(-x / 2),
              3: lambda x: math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)}
    for df, f in closed.items():                                  # series and continued-fraction branches
        for x in (0.3, 1.0, 2.5, 4.0, 9.0, 40.0):
            assert math.isclose(chi2_sf(x, df), f(x), rel_tol=1e-9), (x, df)
    assert chi2_uniform([5, 5, 5, 5]) == (0.0, 1.0)
    x2, p = chi2_uniform([10, 0, 0, 0])
    assert math.isclose(x2, 30.0) and p < 1e-5
    lo, hi = wilson(8, 10)
    assert math.isclose(lo, 0.4901, abs_tol=1e-4) and math.isclose(hi, 0.9433, abs_tol=1e-4)
    assert wilson(0, 0) == (None, None)


def test_implied_pairs_skip_unheard_clips():
    v = {"clips": [{"clip": c, "pos": i + 1, "heard_bars": "11110000" if c != "d" else "11000000"}
                   for i, c in enumerate("abcd")], "best": "a", "worst": "c"}
    pairs = implied_pairs("bws4", v, 4)
    assert pairs == {frozenset("ab"): "a", frozenset("ac"): "a", frozenset("bc"): "b"}   # d heard 2 bars only
    w = dict(v, best="b", worst="c")
    assert agreement("bws4", v, w, 4) == (2, 3)                     # they disagree on a vs b only


@pytest.fixture
def finished(tmp_path):
    """A bws4 demo batch voted to the end: the anchor caught, the repeat answered the same way."""
    d = build_demo(tmp_path / "inbox" / "2026-11-10" / "demo-bws4", "bws4", **FAST)
    m, h = load_batch(d)
    votes = tmp_path / "votes"
    votes.mkdir()
    led = Ledger(votes, m, h, ui="t")
    picks = {}
    for r in m.rounds:
        hr = h.rounds[r.id]
        ids = list(r.ids)
        if hr.kind == "anchor":
            worst = hr.anchor_clip
            best = next(c for c in ids if c != worst)
        elif hr.kind == "repeat":                                     # same real clips as round 1, reversed
            first = h.rounds[hr.of]
            real_to_oid = {hr.real(o): o for o in ids}
            best, worst = (real_to_oid[first.real(picks[hr.of][k])] for k in (0, 1))
        else:
            best, worst = ids[0], ids[3]
        picks[r.id] = (best, worst)
        led.submit({"type": "bws4", "round": r.id, "best": best, "worst": worst, "tag": "sour" if r.number == 1 else None,
                    "clips": [{"clip": c, "pos": i + 1, "heard_bars": "11111111", "heard_ms": 8000}
                              for i, c in enumerate(ids)], "t_ms": {"commit": 30000}})
    events = led.snapshot()
    led.close()
    return m, h, events


def test_aggregates_of_a_finished_batch(finished):
    m, h, events = finished
    a = aggregate(m, h, events, today=date(2026, 11, 10))
    assert (a["rounds"], a["voted"], a["retracted"], a["abandoned"]) == (4, 4, 0, 0)
    assert a["heard"] == {"clips": 16, "heard": 16, "rate": 1.0, "min_bars": 4}
    assert a["anchors"]["rounds"] == 1 and a["anchors"]["caught"] == 1
    assert a["repeats"]["found"] == 1 and a["repeats"]["compared"] == 5 and a["repeats"]["p"] == 1.0
    assert a["position_of_best"][0]["clips"] == 4 and sum(a["position_of_best"][0]["counts"]) == 4
    assert a["bws4"]["tags"] == {"sour": 1}
    assert a["time"]["minutes"] == 2.0 and a["time"]["today_minutes"] == 0.0     # demo timestamps are "now"
    assert [row["number"] for row in a["list"]] == [1, 2, 3, 4]


def test_per_round_rows_carry_no_hidden_labels(finished):
    m, h, events = finished
    a = aggregate(m, h, events)
    rows = json.dumps(a["list"])
    reals = {info["clip"] for hr in h.rounds.values() for info in hr.clips.values()}
    assert not any(rc in json.dumps(a) for rc in reals)           # no real clip id anywhere in the audit
    for word in ("anchor", "repeat", "hidden", "broken", "brief"):
        assert word not in rows
    assert a["list"][0]["picks"] == {"best": 1, "worst": 4, "no_best": False, "no_worst": False,
                                     "all_bad": False, "tag": "sour"}


def test_repeat_original_can_live_in_another_batch(finished):
    m, h, events = finished
    rep = next(e for e in events if (e.get("hidden") or {}).get("kind") == "repeat")
    others = {"older-batch": [e for e in events if e["round"] == rep["hidden"]["of"]]}
    mine = [e for e in events if e["round"] != rep["hidden"]["of"]]
    a = aggregate(m, h, mine, all_events=others)
    assert a["repeats"]["found"] == 1 and a["repeats"]["p"] == 1.0


def test_mode_tallies(tmp_path):
    out = {}
    for mode, pick in [("triage", {"tier": "good"}), ("playlist", {"mark": "heart", "plays": 2}),
                       ("shootout", None), ("ab", None), ("keep3", None)]:
        d = build_demo(tmp_path / mode / f"demo-{mode}", mode, **FAST)
        m, h = load_batch(d)
        (tmp_path / mode / "votes").mkdir()
        led = Ledger(tmp_path / mode / "votes", m, h)
        for r in m.rounds:
            ids = list(r.ids)
            if pick:
                p = dict(pick)
            elif mode == "shootout":
                p = {"best": ids[0], "worst": ids[1]}
            elif mode == "ab":
                p = {"pick": ids[1]}
            else:
                p = {"best": ids[2]}
            led.submit({"type": mode, "round": r.id, **p,
                        "clips": [{"clip": c, "pos": i + 1, "heard_bars": "11111111"} for i, c in enumerate(ids)]})
        out[mode] = aggregate(m, h, read_events(led.path)[0])
        led.close()
    assert out["triage"]["triage"] == {"tiers": {"toss": 0, "fine": 0, "good": 6, "great": 0}, "fine_or_better": 1.0}
    assert out["triage"]["anchors"]["caught"] == 0                 # the broken copy was not tossed
    assert out["playlist"]["playlist"] == {"played": 5, "hearts": 5, "crosses": 0, "heart_rate": 1.0}
    assert out["ab"]["ab"] == {"picks": [0, 4]}
    assert out["keep3"]["keep3"] == {"kept": 3, "none_acceptable": 0}
    opts = {o["option"]: o for o in out["shootout"]["shootout"]["options"]}
    assert out["shootout"]["shootout"]["category"] == "keys"
    assert set(opts) <= {"rhodes", "piano", "guitar", "organ"}
    assert sum(o["best"] for o in opts.values()) + sum(o["worst"] for o in opts.values()) <= 8
    assert all(o["net"] == o["best"] - o["worst"] for o in opts.values())


# ----------------------------------------------------------------------------- batches on paper (no audio needed)

class _Clock:
    def __init__(self):
        self.t = datetime(2026, 11, 10, 21, 0, 0, tzinfo=timezone(timedelta(hours=-8)))

    def __call__(self):
        return self.t


def _paper_batch(tmp_path, name, mode, rounds, *, category=None):
    """Both manifests of a batch, validated, without audio files. `rounds` lists, per round, (kind, of, clips)
    where `of` is the index of the round a repeat repeats and clips are (real id, option, broken) tuples."""
    rng = random.Random(name)
    hexs = lambda n: "".join(rng.choice("0123456789abcdef") for _ in range(n))     # noqa: E731
    pub, hid, rids = [], {}, []
    for kind, of, clips in rounds:
        rid = "r" + hexs(8)
        rids.append(rid)
        oids = ["k" + hexs(10) for _ in clips]
        pub.append({"id": rid, "clips": [{"id": o, "file": f"{o}.flac"} for o in oids]})
        entry = {"kind": kind, "brief_id": "brief-" + hexs(4),
                 "clips": {o: {"clip": real, **({"option": opt} if opt else {})} for o, (real, opt, _) in zip(oids, clips)}}
        if kind == "repeat":
            entry["of"] = rids[of]
        if kind == "anchor":
            entry["anchor_clip"] = next(o for o, (_, _, broken) in zip(oids, clips) if broken)
        hid[rid] = entry
    d = tmp_path / name
    m = parse_public({"v": 1, "batch": name, "mode": mode, "scene": "paper batch", "bpm": 80, "bars": 8,
                      "rounds": pub}, d, check_files=False)
    h = parse_hidden({"v": 1, "batch": name, "category": category, "rounds": hid}, m)
    return m, h


def _real(n):
    return "c" + f"{n:016x}"


def _listened(r, bars="11111111"):
    return [{"clip": c, "pos": i + 1, "heard_bars": bars, "heard_ms": 8000} for i, c in enumerate(r.ids)]


def test_shootout_tally_counts_each_block_once(tmp_path):
    """Incomplete blocks {A,B}, {B,C}, {A,C} plus a hidden repeat of {A,B} with positions reversed, answered
    A > B > C throughout. The repeat measures self-consistency; it is not counted again (spec-6)."""
    a, b, c = (_real(1), "rhodes", False), (_real(2), "piano", False), (_real(3), "organ", False)
    m, h = _paper_batch(tmp_path, "paper-shootout", "shootout",
                        [("none", None, [a, b]), ("none", None, [b, c]), ("none", None, [a, c]),
                         ("repeat", 0, [b, a])], category="keys")
    votes = tmp_path / "votes"
    votes.mkdir()
    led = Ledger(votes, m, h)
    rank = {_real(1): 0, _real(2): 1, _real(3): 2}
    try:
        for r in m.rounds:
            hr = h.rounds[r.id]
            order = sorted(r.ids, key=lambda o: rank[hr.real(o)])
            led.submit({"type": "shootout", "round": r.id, "best": order[0], "worst": order[-1],
                        "clips": _listened(r)})
        events = led.snapshot()
    finally:
        led.close()
    out = aggregate(m, h, events)
    tally = {o["option"]: (o["best"], o["worst"], o["net"], o["shown"]) for o in out["shootout"]["options"]}
    assert tally == {"rhodes": (2, 0, 2, 2), "piano": (1, 1, 0, 2), "organ": (0, 2, -2, 2)}
    assert [o["option"] for o in out["shootout"]["options"]] == ["rhodes", "piano", "organ"]
    assert out["repeats"]["found"] == 1 and out["repeats"]["p"] == 1.0     # the repeat still counts here


def test_a_triage_toss_needs_only_toss_min_bars(tmp_path):
    """taste_pipeline §1: a Toss is heard at 2 bars; every other answer needs 4 (spec-7 A2)."""
    m, h = _paper_batch(tmp_path, "paper-triage", "triage",
                        [("none", None, [(_real(i), None, False)]) for i in range(1, 5)])
    votes = tmp_path / "votes"
    votes.mkdir()
    led = Ledger(votes, m, h)
    try:
        for r, tier, bars in zip(m.rounds, ("toss", "fine", "toss", "good"),
                                 ("11000000", "11000000", "10000000", "11110000")):
            led.submit({"type": "triage", "round": r.id, "tier": tier, "clips": _listened(r, bars)})
        events = led.snapshot()
    finally:
        led.close()
    assert aggregate(m, h, events, heard_min_bars=4, toss_min_bars=2)["heard"] == {
        "clips": 4, "heard": 2, "rate": 0.5, "min_bars": 4}


def test_a_micro_session_that_misses_most_of_its_anchors_is_flagged(tmp_path):
    """§9 Q8: a micro-session that catches fewer than half of its 2 or more anchors is set aside (spec-7 A3)."""
    rows, n = [], 1
    for _ in range(5):
        clips = [(_real(n + i), None, i == 3) for i in range(4)]
        n += 4
        rows.append(("anchor", None, clips))
    m, h = _paper_batch(tmp_path, "paper-bws4", "bws4", rows)
    votes = tmp_path / "votes"
    votes.mkdir()
    clock = _Clock()
    led = Ledger(votes, m, h, clock=clock)
    try:
        # micro 1: 2 anchors, none caught; micro 2: 2 anchors, 1 caught (half); micro 3: 1 anchor, missed
        plan = [(0, False), (0, False), (11, True), (0, False), (11, False)]
        for r, (gap_min, caught) in zip(m.rounds, plan):
            clock.t += timedelta(minutes=gap_min, seconds=30)
            broken = h.rounds[r.id].anchor_clip
            others = [o for o in r.ids if o != broken]
            led.submit({"type": "bws4", "round": r.id, "best": others[0], "worst": broken if caught else others[1],
                        "clips": _listened(r)})
        events = led.snapshot()
    finally:
        led.close()
    micro = aggregate(m, h, events)["anchors"]["micro"]
    assert [(g["micro"], g["anchors"], g["caught"], g["flag"]) for g in micro] == [
        (1, 2, 0, True), (2, 2, 1, False), (3, 1, 0, False)]


def test_today_counts_the_minutes_of_every_batch_voted_today(finished, tmp_path):
    """'Today so far' adds up today's votes and abandons across all logs, against the daily budget (spec-7 A1)."""
    m, h, _ = finished
    clock = _Clock()
    votes = tmp_path / "votes-today"
    votes.mkdir()
    led = Ledger(votes, m, h, clock=clock)
    try:
        for r in m.rounds:
            clock.t += timedelta(minutes=1)
            led.submit({"type": "bws4", "round": r.id, "best": r.ids[0], "worst": r.ids[3],
                        "clips": _listened(r), "t_ms": {"commit": 30000}})
        events = led.snapshot()
    finally:
        led.close()
    older = [{"type": "triage", "ts": "2026-11-10T08:00:00-08:00", "t_ms": {"commit": 60000}},
             {"type": "abandon", "ts": "2026-11-10T08:05:00-08:00", "t_ms": {"abandon": 30000}},
             {"type": "session", "ts": "2026-11-10T08:06:00-08:00", "t_ms": {"commit": 600000}},
             {"type": "triage", "ts": "2026-11-09T23:59:00-08:00", "t_ms": {"commit": 60000}}]
    a = aggregate(m, h, events, all_events={"older": older}, today=date(2026, 11, 10), daily_budget_min=60)
    assert a["time"] == {"minutes": 2.0, "est_min": m.est_min, "today_minutes": 3.5, "daily_budget_minutes": 60}
    assert aggregate(m, h, events, all_events={"older": older}, today=date(2026, 11, 11))["time"]["today_minutes"] == 0.0
