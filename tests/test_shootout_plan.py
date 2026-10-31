"""shootouts/plan.yaml, the shootout plan in symbols (docs/SHOOTOUTS.md sections 4, 6 and 7.6 test 2).

The plan is public, so it places only symbols on tiles: which option a symbol stands for is drawn at job time
(section 2.3). These tests also pin the designs' balance as section 2.4 states it (tile sums, Latin squares,
the alto-above rounds), so a transcription slip in the YAML fails here rather than in a listening session."""
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import yaml

from lofi_rl.shootout.plan import PLAN, PlanError, apply_cuts, load_plan
from lofi_rl.ui.manifest import BATCH_RE

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def plan():
    return load_plan()


def _sums(slots):
    """symbol -> sum of the tile numbers (1-based) it sits on, over the given slots."""
    out = Counter()
    for slot in slots:
        for pos, sym in enumerate(slot.tiles, 1):
            out[sym] += pos
    return dict(out)


def _tiles(slots):
    out = defaultdict(list)
    for slot in slots:
        for pos, sym in enumerate(slot.tiles, 1):
            out[sym].append(pos)
    return out


def test_the_plan_loads_with_every_stage_batch_and_loop(plan):
    assert set(plan.stages) == {"K1", "R1", "R2", "S1", "R3a", "R3b", "S1f", "S2a", "S2b", "K2", "V", "W1", "W2"}
    assert [b.name for b in plan.batches] == [
        "b1103-1-keys", "b1103-2-groove", "b1103-3-sax", "b1104-1-groove", "b1104-2-sax", "b1104-3-keys",
        "b1104-4-vamps", "b1106-1-mix", "b1106-2-mix"]
    assert sorted(plan.loops) == [f"L{i}" for i in range(1, 10)]
    l5 = plan.loops["L5"]
    assert (l5.name, l5.brief.mood, l5.brief.energy, l5.brief.bpm, l5.brief.tonic) == ("l5_wistful_mid", "wistful",
                                                                                        "mid", 83, 2)
    assert plan.path == PLAN and re.fullmatch(r"[0-9a-f]{64}", plan.sha256)


def test_tiles_hold_only_symbols_never_an_option(plan):
    # section 7.6 test 2, on the parsed plan and on the raw YAML (cut tile overrides included)
    options = {o for lst in plan.candidates.values() for o in lst}
    for st in plan.stages.values():
        declared = st.symbols
        for slots in [st.slots, *st.slots_by_count.values()]:
            for slot in slots:
                assert set(slot.tiles) <= declared, (st.id, slot.id)
                assert not set(slot.tiles) & options, (st.id, slot.id)
    raw = yaml.safe_load(PLAN.read_text())
    tile_lists = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "tiles":
                    tile_lists.extend(v.values() if isinstance(v, dict) else [v])
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(raw)
    assert len(tile_lists) > 50
    assert all(not set(t) & options for t in tile_lists)


def test_k1_is_the_tournament_of_section_4_1(plan):
    k1 = plan.stages["K1"]
    sums, tiles = _sums(k1.slots), _tiles(k1.slots)
    assert all(6 <= s <= 9 for s in sums.values()) and len(sums) == 8
    for a, b in k1.groups[0].pairs:
        assert sums[a] + sums[b] == 15, (a, b)                  # each family's two members add to 15 [CD-8]
    assert all(len(set(t)) == 3 for t in tiles.values())       # each candidate on 3 different tiles
    loops = defaultdict(set)
    for slot in k1.slots:
        for sym in slot.tiles:
            loops[sym].add(slot.loop)
    assert all(v == {"L1", "L5", "L9"} for v in loops.values())
    meet = Counter(frozenset(p) for slot in k1.slots for p in
                   [(x, y) for i, x in enumerate(slot.tiles) for y in slot.tiles[i + 1:]])
    twins = {frozenset(p) for p in k1.groups[0].pairs}
    assert all(meet[t] == 3 for t in twins) and all(n == 1 for p, n in meet.items() if p not in twins)


@pytest.mark.parametrize("sid", ["R1", "R2", "R3b", "S2a", "S2b", "W1", "W2"])
def test_latin_squares_put_each_option_on_every_tile_once(plan, sid):
    st = plan.stages[sid]
    k = len(st.slots[0].tiles)
    assert len(st.slots) == k
    assert all(sorted(t) == list(range(1, k + 1)) for t in _tiles(st.slots).values())


def test_r3a_plays_its_second_round_reversed(plan):
    a, b = plan.stages["R3a"].slots
    assert b.tiles == tuple(reversed(a.tiles)) and set(_sums([a, b]).values()) == {5}


def test_s1_puts_each_source_on_every_tile_once_and_each_alto_once_above_the_tenor(plan):
    s1 = plan.stages["S1"]
    assert all(sorted(t) == [1, 2, 3, 4] for t in _tiles(s1.slots).values())
    above = [s for s in s1.slots if s.loop in plan.alto_above]
    altos = s1.groups[1].symbols
    assert all(sum(a in s.tiles for s in above) == 1 for a in altos)
    pairs = Counter(frozenset((x, y)) for s in s1.slots for i, x in enumerate(s.tiles) for y in s.tiles[i + 1:])
    same = [p for p in pairs if len({q[0] for q in p}) == 1]
    assert all(pairs[p] == 3 for p in same) and all(n == 2 for p, n in pairs.items() if p not in same)
    assert {plan.loops[s.loop].brief.mood for s in s1.slots} == {"melancholy", "wistful", "hopeful"}


def test_final_and_k2_tile_sums_by_option_count(plan):
    assert sorted(_sums(plan.stages["S1f"].slots_for(2)).values()) == [4, 5]
    assert all(sorted(t) == [1, 2, 3] for t in _tiles(plan.stages["S1f"].slots_for(3)).values())
    assert sorted(_sums(plan.stages["K2"].slots_for(3)).values()) == [3, 4, 5]
    assert set(_sums(plan.stages["K2"].slots_for(4)).values()) == {5}
    with pytest.raises(PlanError):
        plan.stages["K2"].slots_for(5)


def test_vamp_rows_give_tile_sums_6_to_9_with_the_pinned_vamps_central(plan):
    v = plan.stages["V"]
    for g in v.groups:
        slots = [s for s in v.slots if s.id in g.slots]
        moods = {plan.loops[s.loop].brief.mood for s in slots}
        assert moods == {g.where["mood"]}
        assert _sums(slots) == {"v1": 6, "v2": 7, "v3": 8, "v4": 9}
        for opt, allowed in g.pin.items():
            assert plan.candidates["vamps"][opt]["mood"] == g.where["mood"]
            assert {_sums(slots)[s] for s in allowed} == {7, 8}
    assert {o for g in v.groups for o in g.pin} == {"plagal", "float"}


def test_every_symbol_is_heard_in_two_loops_or_more_within_its_stage(plan):
    for st in plan.stages.values():
        for slots in [st.slots, *st.slots_by_count.values()]:
            loops = defaultdict(set)
            for s in slots:
                for sym in s.tiles:
                    loops[sym].add(s.loop)
            assert all(len(v) >= 2 for v in loops.values()), st.id


def test_consecutive_slot_loops_can_be_ordered_and_batches_sort_in_serving_order(plan):
    names = [b.name for b in plan.batches]
    assert names == sorted(names) and all(BATCH_RE.fullmatch(n) for n in names)
    assert [b.date for b in plan.batches] == sorted(b.date for b in plan.batches)
    homes = Counter()
    for b in plan.batches:
        for sid in b.stages:
            homes[(sid, b.split or b.name)] += 1
    owners = Counter(sid for sid, _ in homes)
    assert all(n == 1 for n in owners.values()) and set(owners) == set(plan.stages)
    assert [b.name for b in plan.batches if b.anchor] == ["b1103-3-sax", "b1104-2-sax", "b1106-1-mix",
                                                          "b1106-2-mix"]


def test_band_slots_and_stage_fills_name_real_candidates(plan):
    assert plan.band["day1"].fixed["keys"] == "gu_tine_ep" and plan.band["day1"].leaders == {}
    assert set(plan.band["day2"].leaders) == {"keys", "kit", "bass", "sax"}
    assert plan.band["day3"].leaders["swing"] == "swing_55" and plan.band["day3"].leaders["keys"] is None
    assert plan.stages["W1"].band == {"rain": "rain_d18"} and plan.stages["W2"].band == {"bed": "excerpt_e1"}
    assert plan.stages["V"].fills is None and plan.stages["V"].sax_muted and not plan.stages["V"].anchor
    assert plan.stages["S1"].anchor and plan.stages["K2"].sub == "spread"


def test_cuts_apply_as_section_2_7_lists_them(plan):
    cut = apply_cuts(plan, ["vamps-high", "latin3", "final"])
    assert "S1f" not in cut.stages and "S1f" not in {s for b in cut.batches for s in b.stages}
    v = cut.stages["V"]
    assert [s.id for s in v.slots] == ["V.1", "V.2", "V.4", "V.5", "V.7", "V.8"]
    for g in v.groups:
        assert set(_sums([s for s in v.slots if s.id in g.slots]).values()) == {5}
    r3b = cut.stages["R3b"]
    assert [s.id for s in r3b.slots] == ["R3b.1", "R3b.2"]
    sums = _sums(r3b.slots)
    assert r3b.groups[0].pin == {"humanize_off": ("n1",)} and sorted(sums.values())[1] == sums["n1"]
    w = apply_cuts(plan, ["rain-w1", "rain-w2"])
    assert len(w.stages["W1"].slots) == 2 and w.stages["W1"].groups[0].pin == {"excerpt_e1": ("z1",)}
    a, b = w.stages["W2"].slots
    assert b.tiles == tuple(reversed(a.tiles))
    assert load_plan().stages["R3b"].slots == plan.stages["R3b"].slots       # the plan itself is unchanged
    with pytest.raises(PlanError, match="unknown cut"):
        apply_cuts(plan, ["no-such-cut"])


def _write(tmp_path, mutate):
    doc = yaml.safe_load(PLAN.read_text())
    doc["loops_dir"] = str(ROOT / "shootouts" / "loops")
    mutate(doc)
    p = tmp_path / "plan.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["stages"]["R1"]["slots"][0].update(tiles=["kit_swirly", "k2", "k3"]), "not a symbol"),
    (lambda d: d["stages"]["R1"]["slots"][0].update(tiles=["k1", "k9", "k3"]), "not a symbol"),
    (lambda d: d["stages"]["R1"]["slots"][1].update(id="R1.1"), "twice"),
    (lambda d: d["stages"]["R1"].update(colour="red"), "unknown"),
    (lambda d: d["stages"]["R1"]["slots"][0].update(loop="L10"), "loop"),
    (lambda d: d["stages"]["R1"]["draw"][0].update(options="drums"), "candidate list"),
    (lambda d: d["stages"]["R1"]["draw"][0].update(symbols=["k1", "k2"]), "options"),
    (lambda d: d["batches"][0].update(name="b1103 keys"), "batch name"),
    (lambda d: d["batches"][0].update(stages=["K1", "Z9"]), "stage"),
    (lambda d: d["band"]["day1"]["fixed"].update(keys="clavinet"), "clavinet"),
    (lambda d: d["candidates"]["kits"].update(gu_tine_ep={}), "twice"),
])
def test_a_broken_plan_is_refused(tmp_path, mutate, match):
    with pytest.raises(PlanError, match=match):
        load_plan(_write(tmp_path, mutate))
