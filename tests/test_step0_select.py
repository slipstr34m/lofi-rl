"""The selection rule step0-select/1 (lofi_rl.step0.select; docs/design/step0_samples.md §1.3-1.8, §6.1) on hand-built
tables, and the synthetic-run writer the other step-0 tests use (tests/helpers_step0.py).

A table is brief id -> its 8 draws in k order, each a text label (it passes its records; one label is one canonical
text) or a gate code (it fails; G2 and G6 are the ones the grammar leaves possible). Render outcomes are given per
draw and applied round by round through select.pending, as the render step applies them.

What these tests pin (§6.1): everything passing; a gate failure; a repeat skipped by the held-out set and kept by
Ref_b; a G7 failure taking the next draw, lazily; the pool after Ref_b, skipping Ref_b's texts, a held-out text (Q5(a))
and a twin's clip (Q5(b)); Ref_b's spares skipping the pool's draws and texts, with §1.5's {k0, k1, k3, k7} example; a
pool G7 failure not replaced; short briefs and their reasons; the same result whatever the row order; both Q1 walks'
counts (220 each; 35/169/16 within [0, 380) and 30/176/14 within [0, 260); 206 shared); the other answers of Q2-Q5
as the local edits they are; and the refusals of rows that are not one run's.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

import pytest

import helpers_step0 as H
import lofi_rl.step0 as S
from lofi_rl import brief_streams
from lofi_rl.step0 import Step0Error
from lofi_rl.step0 import requests as R
from lofi_rl.step0 import select as X

GATE_CODES = ("G1", "G2", "G3", "G4", "G5", "G6", "compile", "crash")
TWIN = ("5101:0", "596:177")            # a held-out brief and its train twin (brief_streams_v1.json, twins)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture(scope="module")
def streams():
    return R.load()


@pytest.fixture(scope="module")
def order(streams):
    return {(r.brief_id, r.k): r.i for r in R.requests(streams=streams)}


@pytest.fixture(scope="module")
def pool(streams):
    return X.pool_briefs(streams["train"])


def scored_rows(table: dict, order: dict, streams) -> list[dict]:
    """scored.jsonl rows for a hand-built table (module doc); a label may carry +nc (a near-copy) or +fix (repaired)."""
    briefs = {f"{st.spec.seed}:{p.index}": (name, p.brief) for name, st in streams.items() for p in st.picks}
    out = []
    for bid, spec in table.items():
        stream, b = briefs[bid]
        for k, token in enumerate(spec.split()):
            label, _, flag = token.partition("+")
            gate = label if label in GATE_CODES else None
            out.append({"i": order[(bid, k)], "stream": stream, "brief_id": bid, "k": k, "energy": b.energy,
                        "mood": b.mood, "ok": gate is None, "gate": gate,
                        "canon_sha256": None if gate else sha(f"canon {label}"), "text_sha256": sha(f"text {bid} {k}"),
                        "near_copy": flag == "nc", "repairs": {"pad_grid": 1} if flag == "fix" else {},
                        "verbatim": {"vamp": False, "keys": False, "A": k % 2 == 0, "B": k % 3 == 0}})
    return out


def record(row: dict, rnd: int, outcome: str, final: bool = True) -> dict:
    """A render record of one draw, shaped as clips.json's (only what select reads, and the clip id)."""
    rec = {"i": row["i"], "brief_id": row["brief_id"], "k": row["k"], "round": rnd, "final": final}
    if outcome == "ok":
        return {**rec, "ok": True, "gate": None, "clip_id": "c" + sha(f"clip {row['i']}")[:16],
                "audio_sha256": sha(f"pcm {row['i']}")}
    if outcome == "error":
        return {**rec, "ok": False, "gate": None, "error": "OSError: synthetic", "clip_id": None}
    if outcome == "skipped":
        return {**rec, "ok": False, "gate": None, "skipped": "render deadline", "clip_id": None}
    return {**rec, "ok": False, "gate": outcome, "detail": "synthetic", "clip_id": None}


def rounds(rows: list[dict], streams, outcomes: dict | None = None, sets=("heldout",)) -> tuple[list[dict], list]:
    """Render what select.pending asks, round by round: outcomes[(brief id, k)] is "ok" (the default), a gate code,
    "error" (once, then ok), "error2" (twice) or "skipped". Returns (records, each round's rows as (brief id, k))."""
    outcomes = outcomes or {}
    by_i = {r["i"]: r for r in rows}
    recs, plan, tried = [], [], Counter()
    while True:
        todo = X.pending(rows, recs, streams=streams, sets=sets)
        if not todo:
            return recs, plan
        plan.append([(by_i[i]["brief_id"], by_i[i]["k"]) for i in todo])
        for i in todo:
            key = (by_i[i]["brief_id"], by_i[i]["k"])
            o = outcomes.get(key, "ok")
            tried[key] += 1
            if o in ("error", "error2") and tried[key] == 1:
                recs.append(record(by_i[i], len(plan), "error", final=False))
            elif o == "error2":
                recs.append(record(by_i[i], len(plan), "error"))
            else:
                recs.append(record(by_i[i], len(plan), "ok" if o == "error" else o))


def ks(entry: dict) -> list[int]:
    return [t["k"] for t in entry["taken"]]


def why(entry: dict) -> dict[int, str]:
    return {n["k"]: n["why"] for n in entry["not_taken"]}


def one(doc: dict, name: str, bid: str) -> dict:
    (e,) = [e for e in doc[name] if e["brief_id"] == bid]
    return e


def run(table, order, streams, outcomes=None, train_outcomes=None, **kw):
    rows = scored_rows(table, order, streams)
    recs, plan = rounds(rows, streams, outcomes)
    if train_outcomes:
        by = {(r["brief_id"], r["k"]): r for r in rows}
        recs += [record(by[key], 1, o) for key, o in train_outcomes.items()]
    return X.select(rows, recs, streams=streams, **kw), rows, recs, plan


# --------------------------------------------------------------------------------------------- the rule
def test_everything_passes(order, streams, pool):
    assert "596:0" in {f"596:{j}" for j in pool}
    doc, rows, recs, plan = run({"5101:0": "a b c d e f g h", "596:0": "p q r s t u v w"}, order, streams)
    assert plan == [[("5101:0", k) for k in range(4)]], "one round: the first 4 distinct candidates, k order"
    h = one(doc, "heldout", "5101:0")
    assert ks(h) == [0, 1, 2, 3] and h["not_taken"] == [] and h["short"] is False and h["undetermined"] is False
    assert [t["clip_id"] for t in h["taken"]] == [r["clip_id"] for r in recs], "the clip ids are the renders'"
    assert h["taken"][0] == {"k": 0, "i": 0, "draw_seed": 1702876545, "text_sha256": sha("text 5101:0 0"),
                             "canon_sha256": sha("canon a"), "clip_id": recs[0]["clip_id"],
                             "audio_sha256": sha("pcm 0")}
    assert h["cell"] == "high/melancholy" and (h["bpm"], h["tonic"]) == (76, 0) and h["twin_of"] == ["596:177"]
    b = one(doc, "refb", "596:0")
    assert ks(b) == [0, 1, 2, 3] and b["spares"] == [7] and b["short"] is False
    assert all(t["clip_id"] is None and t["audio_sha256"] is None for t in b["taken"]), "Ref_b renders later (§3)"
    p = one(doc, "pool", "596:0")
    assert ks(p) == [4, 5, 6] and why(p) == {0: S.REFB, 1: S.REFB, 2: S.REFB, 3: S.REFB}
    assert [d["heldout_text"] for d in doc["heldout_text"]] == [False] * 8
    assert [(d["brief_id"], d["k"]) for d in doc["heldout_text"]] == [("596:0", k) for k in range(8)]
    assert list(h) == list(S.ENTRY_FIELDS["heldout"]) and list(b) == list(S.ENTRY_FIELDS["refb"])
    assert list(p) == list(S.ENTRY_FIELDS["pool"]) and list(h["taken"][0]) == list(S.TAKEN_FIELDS)


def test_a_gate_failure_is_skipped_and_the_next_draw_taken(order, streams):
    doc, *_, plan = run({"5101:0": "a G2 b c d e f g", "596:0": "p G6 q r s t u v"}, order, streams)
    assert plan == [[("5101:0", k) for k in (0, 2, 3, 4)]]
    h = one(doc, "heldout", "5101:0")
    assert ks(h) == [0, 2, 3, 4] and why(h) == {1: "G2"} and h["short"] is False
    b = one(doc, "refb", "596:0")
    assert ks(b) == [0, 2, 3, 4] and why(b) == {1: "G6"} and b["spares"] == []
    assert ks(one(doc, "pool", "596:0")) == [5, 6, 7]


def test_a_repeat_is_skipped_by_the_held_out_set_and_kept_by_refb(order, streams):
    doc, *_, plan = run({"5101:0": "a b a c d e f g", "596:0": "p q p r s t u v", "596:1": "p q r s p t u v"},
                        order, streams)
    assert plan == [[("5101:0", k) for k in (0, 1, 3, 4)]], "a repeat is never rendered for the held-out set"
    h = one(doc, "heldout", "5101:0")
    assert ks(h) == [0, 1, 3, 4] and why(h) == {2: f"{S.REPEAT}0"}
    b = one(doc, "refb", "596:0")
    assert ks(b) == [0, 1, 2, 3] and b["not_taken"] == [], "Q4(a): Ref_b keeps the repeat"
    assert ks(one(doc, "pool", "596:0")) == [4, 5, 6] and one(doc, "refb", "596:0")["spares"] == [7]
    p1 = one(doc, "pool", "596:1")
    assert ks(p1) == [5, 6, 7] and why(p1)[4] == f"{S.REPEAT}0", "the pool skips a Ref_b text"
    assert one(doc, "refb", "596:1")["spares"] == [4], "a draw with a Ref_b text is still a spare (repeats kept)"


def test_a_g7_failure_takes_the_next_draw_lazily(order, streams):
    table = {"5101:0": "a b c d e f g h", "5101:1": "a b c d e f g h"}
    doc, rows, recs, plan = run(table, order, streams, outcomes={("5101:0", 1): "G7", ("5101:0", 4): "G7"})
    assert plan == [[("5101:0", k) for k in range(4)] + [("5101:1", k) for k in range(4)],
                    [("5101:0", 4)], [("5101:0", 5)]], "a draw renders only once the draws before it fall short"
    h = one(doc, "heldout", "5101:0")
    assert ks(h) == [0, 2, 3, 5] and why(h) == {1: "G7", 4: "G7"} and h["short"] is False
    assert ks(one(doc, "heldout", "5101:1")) == [0, 1, 2, 3]
    assert X.render_block(recs) == {"attempted": 10, "rendered": 8, "G7": 2, "other_gates": 0, "errors_final": 0,
                                    "errors_retried": 0, "skipped": 0, "rounds": 3}
    assert X.pending(rows, recs, streams=streams) == [], "the set is decided"


def test_an_error_is_retried_once_and_a_second_leaves_the_brief_undetermined(order, streams):
    table = {"5101:0": "a b c d e f g h", "5101:1": "a b c d e f g h"}
    doc, rows, recs, plan = run(table, order, streams, outcomes={("5101:0", 2): "error", ("5101:1", 0): "error2"})
    assert plan[1:] == [[("5101:0", 2), ("5101:1", 0)]], "round 2 retries both"
    assert ks(one(doc, "heldout", "5101:0")) == [0, 1, 2, 3], "a retry that renders counts"
    u = one(doc, "heldout", "5101:1")
    assert u["undetermined"] is True and u["short"] is False and u["taken"] == [], "an error twice decides nothing"
    blk = X.render_block(recs)
    assert (blk["errors_retried"], blk["errors_final"], blk["rounds"]) == (1, 1, 2)
    skipped = [record(r, 1, "skipped") if (r["brief_id"], r["k"]) == ("5101:1", 1) else record(r, 1, "ok")
               for r in rows if r["k"] < 4]
    assert one(X.select(rows, skipped, streams=streams), "heldout", "5101:1")["undetermined"] is True
    never = X.select(rows, [], streams=streams)
    assert all(e["undetermined"] for e in never["heldout"]), "a held-out draw never rendered leaves its brief open"


def test_the_pool_walks_after_refb_and_skips_refb_texts_held_out_texts_and_held_out_clips(order, streams, monkeypatch):
    table = {TWIN[0]: "a b c d e f g h", TWIN[1]: "p q r s a t u v", "596:2": "p q r s a t u v",
             "596:3": "p q r s q t u v"}
    assert brief_streams.sample_brief(596, 2).energy == "mid" != "high"
    doc, *_ = run(table, order, streams)
    twin = one(doc, "pool", TWIN[1])
    assert ks(twin) == [5, 6, 7] and why(twin)[4] == S.HELDOUT_TEXT, "Q5(a): a held-out clip's text"
    assert ks(one(doc, "pool", "596:2")) == [5, 6, 7], "Q5(a) skips the text in any brief: one label is one text here"
    assert why(one(doc, "pool", "596:3"))[4] == f"{S.REPEAT}1" and ks(one(doc, "pool", "596:3")) == [5, 6, 7]
    flags = {(d["brief_id"], d["k"]): d["heldout_text"] for d in doc["heldout_text"]}
    assert [k for (b, k), f in flags.items() if f] == [4, 4], "596:177 k4 and 596:2 k4"
    assert one(doc, "refb", TWIN[1])["spares"] == [4], "Ref_b may use a held-out text: reported only (§5)"
    monkeypatch.setattr(S, "POOL_EXCLUDES", "clip")             # Q5(b) or (c)
    doc, *_ = run(table, order, streams)
    twin = one(doc, "pool", TWIN[1])
    assert ks(twin) == [5, 6, 7] and why(twin)[4] == S.HELDOUT_CLIP, "a held-out clip is always skipped"
    assert ks(one(doc, "pool", "596:2")) == [4, 5, 6], "another brief's draw of that text is another clip"
    assert sum(d["heldout_text"] for d in doc["heldout_text"]) == 2, "the flags are recorded under every answer"


def test_refb_spares_skip_the_pools_draws_and_texts_and_section_1_5s_example(order, streams):
    table = {"596:0": "p q r s t u v t", "596:1": "p q r s t u v w"}
    doc, *_ = run(table, order, streams)
    assert one(doc, "refb", "596:0")["spares"] == [], "k7 has the text of pool draw k4"
    assert one(doc, "refb", "596:1")["spares"] == [7]
    # Ref_b k0-3, pool k4-6, k2 fails G7 at Ref_b's render: the rule gives {k0, k1, k3, k7}; the words {k0, k1, k3, k4}
    renders = {("596:1", k): "ok" for k in (0, 1, 3, 7)} | {("596:1", 2): "G7"}
    doc, *_ = run(table, order, streams, train_outcomes=renders)
    b = one(doc, "refb", "596:1")
    assert ks(b) == [0, 1, 3, 7] and why(b) == {2: "G7", 4: S.POOL, 5: S.POOL, 6: S.POOL} and b["short"] is False
    assert ks(one(doc, "pool", "596:1")) == [4, 5, 6], "the pool is not re-selected"
    b0 = one(run(table, order, streams, train_outcomes={("596:0", 2): "G7"})[0], "refb", "596:0")
    assert ks(b0) == [0, 1, 3] and b0["short"] is True and why(b0)[7] == f"{S.REPEAT}4", "no spare: short"


def test_the_lazy_rule_renders_refb_and_pool_draws_only_when_needed(order, streams):
    rows = scored_rows({"596:1": "p q r s t u v w"}, order, streams)
    by = {(r["brief_id"], r["k"]): r for r in rows}
    first = X.pending(rows, [], streams=streams, sets=("refb", "pool"))
    assert first == [by[("596:1", k)]["i"] for k in range(7)], "Ref_b's 4 and the pool's 3"
    recs = [record(by[("596:1", k)], 1, "G7" if k == 2 else "ok") for k in range(7)]
    assert X.pending(rows, recs, streams=streams, sets=("refb",)) == [by[("596:1", 7)]["i"]], "then the spare"
    assert X.pending(rows, recs, streams=streams, sets=("pool",)) == [], "a pool failure is never replaced"
    doc = X.select(rows, recs, streams=streams, pending_train=True)
    assert one(doc, "refb", "596:1")["undetermined"] is True, "its spare is still to render"


def test_a_pool_g7_failure_is_not_replaced(order, streams):
    doc, *_ = run({"596:1": "p q r s t u v w"}, order, streams, train_outcomes={("596:1", 5): "G7"})
    p = one(doc, "pool", "596:1")
    assert ks(p) == [4, 6] and why(p)[5] == "G7" and p["short"] is True
    assert one(doc, "refb", "596:1")["spares"] == [7] and ks(one(doc, "refb", "596:1")) == [0, 1, 2, 3]


def test_short_briefs_list_every_draw_not_taken_and_why(order, streams):
    table = {"5101:0": "a a a G2 b b G6 c", TWIN[1]: "p q r s a b c G6", "596:0": "G2 G2 G2 G2 G6 p q r"}
    doc, *_ = run(table, order, streams)
    h = one(doc, "heldout", "5101:0")
    assert ks(h) == [0, 4, 7] and h["short"] is True and h["undetermined"] is False
    assert why(h) == {1: f"{S.REPEAT}0", 2: f"{S.REPEAT}0", 3: "G2", 5: f"{S.REPEAT}4", 6: "G6"}
    twin = one(doc, "pool", TWIN[1])
    assert ks(twin) == [] and twin["short"] is True
    assert why(twin) == {0: S.REFB, 1: S.REFB, 2: S.REFB, 3: S.REFB, 4: S.HELDOUT_TEXT, 5: S.HELDOUT_TEXT,
                         6: S.HELDOUT_TEXT, 7: "G6"}
    b = one(doc, "refb", "596:0")
    assert ks(b) == [5, 6, 7] and b["short"] is True and why(b) == {0: "G2", 1: "G2", 2: "G2", 3: "G2", 4: "G6"}
    p = one(doc, "pool", "596:0")
    assert ks(p) == [] and p["short"] is True and set(why(p).values()) == {"G2", "G6", S.REFB}
    rows = scored_rows(table, order, streams)
    out = X.readouts(X.sets_json(rows, rounds(rows, streams)[0], streams=streams, render_code=None), rows)
    assert (out["heldout"]["short"], out["refb"]["short"], out["pool"]["short"]) == (1, 1, 2)


def test_the_result_is_the_same_whatever_the_row_order(order, streams):
    table = {"5101:0": "a b a G2 c d e f", "5101:1": "a b c d e f g h", TWIN[1]: "p q r s a t u v",
             "596:0": "p q p G6 r s t s"}
    rows = scored_rows(table, order, streams)
    recs, _ = rounds(rows, streams, {("5101:0", 1): "G7", ("5101:1", 3): "error"})
    want = json.dumps(X.sets_json(rows, recs, streams=streams, render_code={"stamp": "s", "source_sha8": "x"}))
    rng = random.Random(596)
    for _ in range(5):
        r2, c2 = rows[:], recs[:]
        rng.shuffle(r2)
        rng.shuffle(c2)
        assert json.dumps(X.sets_json(r2, c2, streams=streams, render_code={"stamp": "s", "source_sha8": "x"})) == want
        assert X.candidates_block(r2, streams=streams) == X.candidates_block(rows, streams=streams)
        assert X.render_block(c2) == X.render_block(recs)


# --------------------------------------------------------------------------------------------- Q1: the pool's briefs
def test_both_q1_walks_reproduce_this_sessions_counts(streams):
    train = streams["train"]
    a, b = X.pool_briefs(train, "band"), X.pool_briefs(train, "cell")
    assert a == X.pool_briefs(train) and S.POOL_BRIEFS == "band", "Q1(a), the recommended answer"
    assert len(a) == len(set(a)) == 220 == len(b) == len(set(b))
    assert max(a) == 379 and max(b) == 259, "(a) within [0, 380); (b) within [0, 260)"
    assert list(a) == sorted(a) and list(b) == sorted(b), "stream order"
    bands = lambda idx: [sum(1 for i in idx if X.band_of(train.picks[i].brief.bpm) == j) for j in range(3)]
    assert bands(a) == [35, 169, 16] and bands(b) == [30, 176, 14]
    assert len(set(a) & set(b)) == 206
    cells = lambda idx: Counter(brief_streams.cell_name(train.picks[i].cell) for i in idx)
    want = {brief_streams.cell_name(c): n for c, n in brief_streams.per_cell_counts(220).items()}
    assert cells(a) == cells(b) == want and sorted(set(want.values())) == [24, 25]
    twins = {j for _, js in brief_streams.twins(streams)["rm_heldout"] for j in js}
    assert len(twins) == 9 and sorted(twins & set(a)) == sorted(twins & set(b)) == [13, 79, 81, 95, 137, 177, 225]
    assert set(range(40)) <= set(a) and set(range(40)) <= set(b), "596:[0, 40), the smokes' briefs"


def test_q1a_splits_each_cell_by_band_in_the_samplers_shares(streams):
    assert X.band_weights() == (15, 64, 5), "brief.TEMPO_WEIGHTS: 15, 64 and 5 of 84"
    assert X.band_quotas(25, (15, 64, 5)) == (4, 19, 2) and X.band_quotas(24, (15, 64, 5)) == (4, 18, 2)
    rooms = X.band_rooms(streams["train"])
    moved = {c: (rooms[(c, 0)], rooms[(c, 1)], rooms[(c, 2)]) for c in ("low/melancholy", "high/wistful",
                                                                       "high/melancholy", "low/wistful")}
    assert moved == {"low/melancholy": (4, 20, 1), "high/wistful": (4, 19, 1), "high/melancholy": (3, 19, 2),
                     "low/wistful": (4, 19, 2)}, "a band [0, 400) cannot fill gives its room to 75-90"
    assert sum(rooms.values()) == 220
    with pytest.raises(Step0Error, match="tile"):
        X.band_weights(((60, 74), (76, 95)))
    with pytest.raises(Step0Error, match="not one of"):
        X.pool_briefs(streams["train"], "tempo")


# --------------------------------------------------------------------------------------------- the other answers
def test_q2b_lets_the_pool_reuse_refbs_draws(order, streams, monkeypatch):
    monkeypatch.setattr(S, "POOL_SHARES_REFB", True)
    doc, *_ = run({"596:1": "p q r s t u v w"}, order, streams)
    assert ks(one(doc, "pool", "596:1")) == [0, 1, 2] and one(doc, "refb", "596:1")["spares"] == [4, 5, 6, 7]
    doc, *_ = run({"596:1": "p q r s t u v w"}, order, streams, train_outcomes={("596:1", 2): "G7"})
    assert ks(one(doc, "refb", "596:1")) == [0, 1, 3, 4], "the registered words: the next passing draw"


@pytest.mark.parametrize("rule, flag, why_", [("hard_not_copy", "nc", "near_copy"), ("hard_no_repair", "fix",
                                                                                      "repaired")])
def test_q3b_and_q3c_leave_out_near_copies_or_repaired_loops(order, streams, monkeypatch, rule, flag, why_):
    table = {"5101:0": f"a b+{flag} c d e f g h"}
    assert ks(one(run(table, order, streams)[0], "heldout", "5101:0")) == [0, 1, 2, 3], "Q3(a): they pass"
    monkeypatch.setattr(S, "PASSING", rule)
    h = one(run(table, order, streams)[0], "heldout", "5101:0")
    assert ks(h) == [0, 2, 3, 4] and why(h) == {1: why_}


def test_q4b_and_q4c_change_which_sets_skip_a_repeat(order, streams, monkeypatch):
    table = {"5101:0": "a a b c d e f g", "596:0": "p p q r s t u v"}
    monkeypatch.setattr(S, "DISTINCT", {"heldout": True, "refb": True})            # Q4(b)
    doc, *_ = run(table, order, streams)
    assert ks(one(doc, "refb", "596:0")) == [0, 2, 3, 4] and why(one(doc, "refb", "596:0")) == {1: f"{S.REPEAT}0"}
    assert why(one(doc, "pool", "596:0"))[1] == f"{S.REPEAT}0"
    monkeypatch.setattr(S, "DISTINCT", {"heldout": False, "refb": False})          # Q4(c)
    doc, *_ = run(table, order, streams)
    assert ks(one(doc, "heldout", "5101:0")) == [0, 1, 2, 3], "a tuple with a repeat (it could not be played)"


def test_the_answer_constants_are_the_recommended_ones():
    assert S.POOL_BRIEFS == "band" and S.POOL_BANDS == ((60, 74), (75, 90), (91, 95)) and S.POOL_FILL_BAND == (75, 90)
    assert S.POOL_SHARES_REFB is False and S.PASSING == "hard" and dict(S.DISTINCT) == {"heldout": True, "refb": False}
    assert S.POOL_EXCLUDES == "text" and S.STREAMS == (("rm_heldout", "all"), ("train", "all"))
    assert S.SHORT_RULE == "report" and S.COMMITTED_BEFORE.startswith("before the first Phase-2 vote")
    assert S.PASSING in X.PASSING_RULES and S.POOL_BRIEFS in X.POOL_RULES and S.POOL_EXCLUDES in X.EXCLUDES


# --------------------------------------------------------------------------------------------- refusals
def test_rows_that_are_not_one_runs_are_refused(order, streams):
    rows = scored_rows({"5101:0": "a b c d e f g h", "596:0": "p q r s t u v w"}, order, streams)
    with pytest.raises(Step0Error, match="holds draws"):
        X.select(rows[1:], streams=streams)
    with pytest.raises(Step0Error, match="repeated"):
        X.select(rows + [rows[3]], streams=streams)
    with pytest.raises(Step0Error, match="not a draw of rm_heldout or train"):
        X.select(rows + [{**rows[0], "stream": "dev"}], streams=streams)
    with pytest.raises(Step0Error, match="not a brief of the rm_heldout stream"):
        X.select([{**r, "brief_id": "5101:12"} for r in rows[:8]], streams=streams)       # 12 is not one of the 40
    with pytest.raises(Step0Error, match="is not a brief of the train stream"):
        X.select([{**r, "brief_id": "596:400"} for r in rows[8:]], streams=streams)       # past run 1's [0, 400)
    with pytest.raises(Step0Error, match="canon_sha256"):
        X.select([{**r, "canon_sha256": None} if r["k"] == 0 else r for r in rows], streams=streams)
    with pytest.raises(Step0Error, match="the row says"):
        X.select([{**r, "mood": "hopeful"} if r["k"] == 0 else r for r in rows], streams=streams)
    recs = [record(rows[0], 1, "ok"), record(rows[0], 2, "ok")]
    with pytest.raises(Step0Error, match="2 final render records"):
        X.select(rows, recs, streams=streams)
    with pytest.raises(Step0Error, match="not a draw of the rows"):
        X.select(rows, [{**record(rows[0], 1, "ok"), "i": 999}], streams=streams)


def test_an_answer_constant_out_of_its_choices_is_refused(order, streams, monkeypatch):
    rows = scored_rows({"5101:0": "a b c d e f g h", "596:0": "p q r s t u v w"}, order, streams)
    for name, bad in (("PASSING", "soft"), ("POOL_EXCLUDES", "nothing"), ("POOL_SHARES_REFB", "no"),
                      ("DISTINCT", {"heldout": "yes", "refb": False}), ("POOL_BRIEFS", "tempo")):
        with monkeypatch.context() as m:
            m.setattr(S, name, bad)
            with pytest.raises(Step0Error):
                X.select(rows, streams=streams)


# --------------------------------------------------------------------------------------------- step0.json's blocks
def test_candidates_render_and_readout_blocks(order, streams):
    table = {"5101:0": "a G2 a b c d e f", TWIN[1]: "p q r s a t u v"}
    rows = scored_rows(table, order, streams)
    recs, _ = rounds(rows, streams, {("5101:0", 3): "G7"})
    cand = X.candidates_block(rows, streams=streams)
    assert list(cand) == list(S.BLOCKS["step0.json:candidates"])
    assert cand["passing"] == {"5101:0": [0, 2, 3, 4, 5, 6, 7], TWIN[1]: list(range(8))}
    assert cand["refb"] == {TWIN[1]: [0, 1, 2, 3]}
    assert cand["heldout_order"] == [order[("5101:0", k)] for k in (0, 3, 4, 5)], "round 1: 4 distinct candidates"
    doc = X.sets_json(rows, recs, streams=streams, render_code={"stamp": "s", "source_sha8": "x"})
    assert list(doc) == list(S.FIELDS["sets.json"]) and doc["mark"] == S.MARK and doc["rule"] == S.RULE
    assert ks(one(doc, "heldout", "5101:0")) == [0, 4, 5, 6]
    out = X.readouts(doc, rows)
    assert list(out) == list(S.BLOCKS["step0.json:sets"])
    assert out["heldout"] == {"briefs": 1, "taken": 4, "short": 0, "undetermined": 0, "near_copy": 0, "repaired": 0,
                              "verbatim": {"A": 3, "B": 2, "AB": 2}}
    assert out["pool"]["taken"] == 3 and out["heldout_text"]["high/melancholy"] == {"draws": 1, "refb": 0, "pool": 0}
    assert sum(c["draws"] for c in out["heldout_text"].values()) == 1 and len(out["heldout_text"]) == 9
    assert list(X.render_block(recs)) == list(S.BLOCKS["step0.json:render"])


# --------------------------------------------------------------------------------------------- the synthetic run
def _fields_ok(name: str, keys) -> bool:
    want = S.FIELDS[name]
    return [k for k in want if k in keys] == list(keys) and set(want) - set(keys) <= set(S.OPTIONAL.get(name, ()))


def test_the_synthetic_run_is_one_runs_files_with_the_registered_fields(tmp_path, streams):
    table = {"5101:0": "a b a/G7 c d/E e f g", TWIN[1]: "t0 t1 t2 t3 a b c G6", "596:0": "t0 G2 t1 t2 t3 t4 t5 t6"}
    run = H.make_run(tmp_path, table, briefs=["5101:1"])
    files = {p.name for p in run.summary.iterdir()}
    assert files == set(S.SUMMARY_FILES)
    for name in S.SUMMARY_FILES:
        text = (run.summary / name).read_text(encoding="utf-8")
        if name.endswith(".jsonl"):
            rows = [json.loads(line) for line in text.splitlines()]
            assert rows and all(_fields_ok(name, r) for r in rows), name
        else:
            doc = json.loads(text)
            assert list(doc) == list(S.FIELDS[name]), name
            if name in S.SCHEMAS:
                assert doc["schema"] == S.SCHEMAS[name]
    gen, step0 = run.doc("generate.json"), run.doc("step0.json")
    for key, fields in S.BLOCKS.items():
        name, _, path = key.partition(":")
        node = run.doc(name)
        for part in path.split("."):
            node = node[part] if part != "*" else next(iter(node.values()))
        assert list(node) == list(fields), key
    assert [r["i"] for r in run.rows] == sorted(r["i"] for r in run.rows), "each row keeps its place in the order"
    assert gen["requested"] == gen["generated"] == len(run.rows) == 32 and step0["n"] == 32
    recs = json.loads((run.summary / "clips.json").read_text())["clips"]
    assert all(list(r) == [*S.RENDER_CLIP_FIELDS[kind], *S.CLIP_ADDED] for r in recs
               for kind in ["ok" if r["ok"] else "gate" if r["gate"] else "error" if "error" in r else "skipped"])
    scored = [json.loads(line) for line in (run.summary / "scored.jsonl").read_text().splitlines()]
    want = X.sets_json(scored, recs, streams=streams, render_code=step0["render_code"])
    assert json.loads((run.summary / "sets.json").read_text()) == want, "sets.json is select over the run's own files"
    h = one(want, "heldout", "5101:0")
    assert ks(h) == [0, 1, 3, 4] and why(h) == {2: f"{S.REPEAT}0"}
    assert ks(one(want, "pool", TWIN[1])) == [] and one(want, "pool", TWIN[1])["short"] is True
    assert why(one(want, "refb", "596:0")) == {1: "G2"}
    assert step0["render"]["errors_retried"] == 1 and step0["candidates"] == X.candidates_block(scored, streams=streams)
    assert {r["i"]: r["canon_sha256"] for r in recs if r["ok"]} == {
        s["i"]: s["canon_sha256"] for s in scored if s["i"] in {r["i"] for r in recs if r["ok"]}}
    assert all(s["ok"] for s in scored if s["brief_id"] == "5101:1"), "the default draws all pass"


def test_the_default_synthetic_run_is_complete_and_clean(tmp_path):
    run = H.make_run(tmp_path)
    step0, sets = run.doc("step0.json"), run.doc("sets.json")
    assert len(run.rows) == 3520 and run.doc("generate.json")["requested"] == 3520
    assert [(len(sets[n]), sum(len(e["taken"]) for e in sets[n])) for n in S.SETS] == [(40, 160), (400, 1600),
                                                                                       (220, 660)]
    assert not any(e["short"] or e["undetermined"] for n in S.SETS for e in sets[n])
    assert not any(d["heldout_text"] for d in sets["heldout_text"]) and len(sets["heldout_text"]) == 3200
    assert step0["render"] == {"attempted": 160, "rendered": 160, "G7": 0, "other_gates": 0, "errors_final": 0,
                               "errors_retried": 0, "skipped": 0, "rounds": 1}
    for name, n in (("rm_heldout", 320), ("train", 3200)):
        b = step0["by_stream"][name]
        assert b["n"] == n and b["pass"]["count"] == n and b["prompt"]["rows_matching"] == n
        assert b["constrained"]["in_language"]["stop"] == {"n": n, "in_language": n, "out_of_language": 0}
        assert b["palette"]["matches_generate"] is True and b["prompt"]["template_matches_generate"] is True
    assert {r["prompt_tokens"] for r in run.rows} == set(S.PROMPT_TOKENS.values())
    assert all(r["sp_seed"] == r["draw_seed"] for r in run.rows)
