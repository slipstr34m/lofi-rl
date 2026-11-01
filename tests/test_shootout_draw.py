"""The job-time draw (docs/SHOOTOUTS.md sections 2.3-2.5; section 7.6 tests 3, 4 and 6).

The draw maps each stage's symbols to options within the stage's constraints, adds the day's anchor, splits the
day-3 pool, and orders every batch. These tests run it from fixed seeds, the way the job runs it from a fresh
one, and check what section 2.4 promises: K1's family sums of 15, one alto-above S1 round per alto, plagal and
float on the central tile sums, the Latin squares, R3a reversed, no two consecutive rounds on one loop, and a
design round first. No audio is involved."""
import json
from collections import defaultdict

import pytest

from lofi_rl.prng import TEST_VECTOR, stream
from lofi_rl.shootout.draw import (DrawError, Rng, draw, est_minutes, order_rounds, parse_leaders,
                                   round_seconds)
from lofi_rl.shootout.plan import load_plan
from lofi_rl.shootout.sour import raise_accidentals

PLAN = load_plan()
DAY2 = parse_leaders("keys=gu_tine_ep+fp_fm_ep1+gu_organ/kit=kit_swirly/bass=bass_fp_finger_yr/"
                     "sax=sax_mtg_tenor+sax_gu_alto")
DAY3 = parse_leaders("keys=fp_fm_ep1/kit=kit_swirly/bass=bass_fp_finger_yr/sax=sax_mtg_tenor/swing=swing_58")
ARGS = {"day1": {}, "day2": {"leaders": DAY2}, "day3": {"leaders": DAY3}}
SEEDS = range(1, 13)
SEED_A, SEED_B = 0x5EED0001, 0x5EED000A      # two seeds whose maps differ in every stage of every day


def _draw(day, seed, **kw):
    return draw(PLAN, day, seed=seed, **{**ARGS[day], **kw})


@pytest.fixture(scope="module")
def draws():
    return {(day, s): _draw(day, s) for day in ARGS for s in SEEDS}


def _rounds(doc, stage, kind="none"):
    return [r for b in doc["batches"] for r in b["rounds"] if r["stage"] == stage and r["kind"] == kind]


def _opts(r):
    return [c["option"] for c in r["clips"]]


def _positions(rounds):
    out = defaultdict(list)
    for r in rounds:
        for pos, o in enumerate(_opts(r), 1):
            out[o].append(pos)
    return out


def test_the_rng_is_the_projects_splitmix64_stream():
    assert Rng(TEST_VECTOR[0]).next64() == TEST_VECTOR[1]
    r = Rng(2026)
    assert tuple(r.next64() for _ in range(5)) == stream(2026, 5)
    a, b = Rng.keyed(7, "map", "K1"), Rng.keyed(7, "map", "R1")
    assert a.next64() != b.next64() and Rng.keyed(7, "map", "K1").next64() == Rng.keyed(7, "map", "K1").next64()
    xs = Rng(1).shuffle(list(range(10)))
    assert sorted(xs) == list(range(10)) and xs != list(range(10))
    assert all(len(Rng(s).hex(10)) == 10 for s in range(3))


@pytest.mark.parametrize("day", list(ARGS))
def test_one_seed_gives_one_draw_and_two_seeds_differ_in_every_stage(day):
    a = json.dumps(_draw(day, SEED_A), sort_keys=True)
    assert a == json.dumps(_draw(day, SEED_A), sort_keys=True)
    da, db = json.loads(a), _draw(day, SEED_B)
    assert da["seed"] == f"{SEED_A:016x}" and set(da["maps"]) == {s.id for s in PLAN.stages_of(day)}
    for sid in da["maps"]:
        assert da["maps"][sid] != db["maps"][sid], sid
        assert [_opts(r) for r in _rounds(da, sid)] != [_opts(r) for r in _rounds(db, sid)], sid


def test_the_maps_vary_across_seeds(draws):
    for day in ARGS:
        for sid in {s.id for s in PLAN.stages_of(day)}:
            seen = {json.dumps(draws[(day, s)]["maps"][sid], sort_keys=True) for s in SEEDS}
            assert len(seen) >= 2, (day, sid)
    k1 = {json.dumps(draws[("day1", s)]["maps"]["K1"]) for s in SEEDS}
    assert len(k1) == len(SEEDS)                       # 384 maps: 12 seeds give 12 different ones here


def test_k1_families_sum_to_15_and_each_keys_option_is_heard_on_three_loops_and_tiles(draws):
    fams = defaultdict(list)
    for o in PLAN.stages["K1"].options:
        fams[PLAN.candidates[o].family].append(o)
    for s in SEEDS:
        rounds = _rounds(draws[("day1", s)], "K1")
        pos = _positions(rounds)
        assert len(rounds) == 6 and len(pos) == 8
        assert all(sum(pos[a]) + sum(pos[b]) == 15 for a, b in fams.values())
        assert all(len(set(p)) == 3 and 6 <= sum(p) <= 9 for p in pos.values())
        loops = defaultdict(set)
        for r in rounds:
            for o in _opts(r):
                loops[o].add(r["loop"])
        assert all(len(v) == 3 for v in loops.values())


def test_s1_puts_every_alto_once_above_the_tenor_and_every_source_on_every_tile(draws):
    above = {PLAN.loops[lid].name for lid in PLAN.alto_above}
    altos = {o for o in PLAN.stages["S1"].options if PLAN.candidates[o].register == "alto"}
    for s in SEEDS:
        rounds = _rounds(draws[("day1", s)], "S1")
        assert all(sorted(p) == [1, 2, 3, 4] for p in _positions(rounds).values())
        for a in altos:
            assert sum(a in _opts(r) for r in rounds if r["loop"] in above) == 1


def test_the_latin_squares_r3a_reversed_and_r2_on_each_kit(draws):
    for s in SEEDS:
        for day, sids in (("day1", ["R1", "R2"]), ("day2", ["R3b", "S2a", "S2b"]), ("day3", ["W1", "W2"])):
            for sid in sids:
                rounds = _rounds(draws[(day, s)], sid)
                k = len(rounds[0]["clips"])
                assert len(rounds) == k and all(sorted(p) == list(range(1, k + 1))
                                                for p in _positions(rounds).values()), (day, sid)
        a, b = sorted(_rounds(draws[("day2", s)], "R3a"), key=lambda r: r["slot"])
        assert _opts(b) == _opts(a)[::-1] and set(_opts(a)) == set(PLAN.stages["R3a"].options)
        d1 = draws[("day1", s)]
        r1 = d1["maps"]["R1"][0]
        kits = {r["slot"]: {c["kit"] for c in r["clips"]} for r in _rounds(d1, "R2")}
        assert kits == {"R2.1": {r1["k1"]}, "R2.2": {r1["k2"]}, "R2.3": {r1["k3"]}}


def test_plagal_and_float_always_sit_on_the_central_tile_sums(draws):
    for s in SEEDS:
        rounds = _rounds(draws[("day2", s)], "V")
        assert len(rounds) == 9
        pos = _positions(rounds)
        assert sum(pos["plagal"]) in (7, 8) and sum(pos["float"]) in (7, 8)
        by_mood = defaultdict(set)
        for r in rounds:
            by_mood[PLAN.loop_named(r["loop"]).brief.mood] |= set(_opts(r))
        for mood, opts in by_mood.items():
            assert opts == {v for v, x in PLAN.vamps.items() if x.mood == mood}
        assert sorted(sum(p) for p in pos.values()) == [6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]


def test_every_batch_opens_with_a_design_round_and_never_repeats_a_loop_back_to_back(draws):
    for doc in draws.values():
        for b in doc["batches"]:
            rs = b["rounds"]
            assert rs[0]["kind"] == "none", b["name"]
            assert all(x["loop"] != y["loop"] for x, y in zip(rs, rs[1:])), b["name"]
            assert all(len({c["option"] for c in r["clips"]}) == len(r["clips"]) for r in rs)


def test_the_anchor_imitates_a_sax_round_with_one_sour_clip(draws):
    for (day, s), doc in draws.items():
        anchors = [(b["name"], r) for b in doc["batches"] for r in b["rounds"] if r["kind"] == "anchor"]
        assert len(anchors) == 1, (day, s)
        name, a = anchors[0]
        assert name in {"day1": ["b1103-3-sax"], "day2": ["b1104-2-sax"],
                        "day3": ["b1106-1-mix", "b1106-2-mix"]}[day]
        src = [r for b in doc["batches"] for r in b["rounds"] if r["kind"] == "none" and r["slot"] == a["imitates"]]
        assert len(src) == 1 and src[0]["loop"] == a["loop"] and sorted(_opts(src[0])) == sorted(_opts(a))
        assert PLAN.stages[a["stage"]].anchor
        sour = [c for c in a["clips"] if c["render"].endswith(".sour")]
        assert len(sour) == 1 and sour[0]["what"] == "sour copy" and a["clips"].index(sour[0]) + 1 == a["anchor_pos"]
        rec = doc["sour"][sour[0]["render"]]
        assert rec["ok"] and rec["gate"] == "pass" and rec["p_chrom"] >= 0.2
        ren = doc["renders"][sour[0]["render"]]
        normal = doc["renders"][next(c["render"] for c in src[0]["clips"] if c["option"] == sour[0]["option"])]
        assert ren["text"] == raise_accidentals(normal["text"])
        sp, np_ = doc["palettes"][ren["palette"]], doc["palettes"][normal["palette"]]
        if "level" in np_:          # CD-52: the sour copy is its own clip, matched on its own text
            assert sp["band"] == np_["band"] and sp["base"] == np_["base"] and ren["palette"] == normal["palette"] + "_sour"
            assert sp["level"] == {**np_["level"], "case": np_["level"]["case"] + ".sour"}
        else:                       # a stage whose option is not an instrument keeps the clip's palette
            assert ren["palette"] == normal["palette"]
        others = [c["render"] for c in a["clips"] if c is not sour[0]]
        assert set(others) <= {c["render"] for c in src[0]["clips"]}       # the other clips reuse their renders


def test_render_and_palette_counts_match_section_7_7(draws):
    d1, d2, d3 = draws[("day1", 1)], draws[("day2", 1)], draws[("day3", 1)]
    assert (len(d1["renders"]), len(d2["renders"]), len(d3["renders"])) == (67, 84, 26)
    assert len(d1["palettes"]) == 67 and len({p["base"] for p in d1["palettes"].values()}) == 8 + 3 + 9 + 6
    assert [b["est_min"] for b in d1["batches"][:2]] == [7.8, 6.0] and 9.0 <= d1["batches"][2]["est_min"] <= 9.3
    assert d2["batches"][0]["est_min"] == 5.6
    rerun = _draw("day3", 1, rerun_s2="sax_gu_alto")
    assert len(rerun["renders"]) == 26 + 18
    assert {p["band"]["sax"] for k, p in rerun["palettes"].items() if k.startswith("s2")} == {"sax_gu_alto"}


MATCHED = ("keys", "kit", "bass", "sax")       # band slots whose option is an instrument (CD-52)


def test_every_clip_of_an_instrument_stage_plays_its_own_palette_named_for_its_loop(draws):
    # CD-52: a clip's varied-bus fader comes from its candidate's loudness on the round's own loop, so each render
    # of a stage whose option is an instrument plays its own palette: the stage palette `base` (the band), whose
    # bus fader `level` names (the bus, the option, and the case it is measured on: the loop, or loop.<variant>
    # for a changed text such as an anchor's sour copy). Other stages keep one palette per option.
    for (day, s), doc in draws.items():
        seen = []
        for rid, ren in doc["renders"].items():
            spec = doc["palettes"][ren["palette"]]
            st = PLAN.stages[ren["stage"]]
            if st.fills in MATCHED:
                lp = PLAN.loop_named(ren["loop"])
                case = ren["loop"] if ren["variant"] == "base" else f"{ren['loop']}.{ren['variant']}"
                assert spec["level"] == {"bus": st.bus, "variant": ren["option"], "case": case}, (day, rid)
                tail = [lp.id.lower()] + ([] if ren["variant"] == "base" else [ren["variant"]])
                assert ren["palette"] == "_".join([spec["base"], *tail]), (day, rid)
                assert spec["band"][st.fills] == ren["option"]
                seen.append(ren["palette"])
            else:
                assert "level" not in spec and "base" not in spec, (day, rid)
        assert len(seen) == len(set(seen)), (day, s, "one palette per clip")
        assert bool(seen) == (day != "day3")


def test_palettes_carry_the_day_band_and_stage_texts_their_substitution(draws):
    d1, d2 = draws[("day1", 3)], draws[("day2", 3)]
    dev = {"keys": "gu_tine_ep", "kit": "kit_gu_jazz", "bass": "bass_gu_acoustic", "sax": "sax_gu_tenor"}
    for key, p in d1["palettes"].items():
        fixed = {k: v for k, v in dev.items() if k != p["fills"] and not (p["stage"] == "R2" and k == "kit")}
        assert {k: p["band"][k] for k in fixed} == fixed, key
    lead = {"keys": "gu_tine_ep", "kit": "kit_swirly", "bass": "bass_fp_finger_yr", "sax": "sax_mtg_tenor"}
    r3a = d2["palettes"]["r3a_swing_58"]["band"]
    assert {k: r3a[k] for k in lead} == lead and r3a["swing"] == "swing_58" and r3a["humanize"] == "humanize_off"
    assert d2["palettes"]["v"]["sax_muted"] and set(_opts(_rounds(d2, "S1f")[0])) == {"sax_mtg_tenor", "sax_gu_alto"}
    assert {o for r in _rounds(d2, "K2") for o in _opts(r)} == {"gu_tine_ep", "fp_fm_ep1", "gu_organ"}
    for rid, ren in d2["renders"].items():
        base = PLAN.loop_named(ren["loop"]).text
        if ren["stage"] == "K2":
            assert ren["text"] == base.replace("keys: ep rootless", "keys: ep spread")
        elif ren["stage"] == "V":
            line = PLAN.vamps[ren["option"]].line
            assert ren["text"].splitlines()[0] == f"vamp: {line}"
            assert ren["text"].splitlines()[1:] == base.splitlines()[1:]
        elif ren["variant"] == "base":
            assert ren["text"] == base
    w = draws[("day3", 3)]["palettes"]
    assert w["w1_excerpt_e2"]["band"]["rain"] == "rain_d18" and w["w2_rain_d30"]["band"]["bed"] == "excerpt_e1"
    assert w["w1_excerpt_e2"]["band"]["swing"] == "swing_58" and w["w1_excerpt_e2"]["band"]["keys"] == "fp_fm_ep1"


def test_the_day3_pool_splits_into_two_mix_batches_of_about_equal_length(draws):
    for s in SEEDS:
        b1, b2 = draws[("day3", s)]["batches"]
        assert (b1["name"], b2["name"]) == ("b1106-1-mix", "b1106-2-mix")
        assert len(b1["rounds"]) + len(b2["rounds"]) == 3 + 4 + 1
        assert abs(b1["est_min"] - b2["est_min"]) <= 1.5
    rr = _draw("day3", 5, rerun_s2="sax_gu_alto")
    assert sum(len(b["rounds"]) for b in rr["batches"]) == 3 + 4 + 1 + 6
    assert max(b["est_min"] for b in rr["batches"]) <= 11.6                 # section 2.4's bound with the re-run


def test_cuts_shrink_day2_and_leave_the_other_stages_draws_alone():
    full = _draw("day2", 9)
    cut = draw(PLAN, "day2", seed=9, cuts=["vamps-high", "latin3", "final"],
               leaders={**DAY2, "sax": ("sax_mtg_tenor",)})
    assert "S1f" not in cut["maps"] and len(_rounds(cut, "V")) == 6 and len(_rounds(cut, "R3b")) == 2
    for sid in ("R3b", "S2a", "S2b"):
        default = PLAN.stages[sid].default
        pos = _positions(_rounds(cut, sid))
        sums = sorted(sum(p) for p in pos.values())
        assert sum(pos[default]) == sums[1], sid                 # the default on the middle tile sum
    assert cut["maps"]["R3a"] == full["maps"]["R3a"] and cut["maps"]["K2"] == full["maps"]["K2"]
    assert cut["plan"]["cuts"] == ["vamps-high", "latin3", "final"]
    with pytest.raises(DrawError, match="takes"):
        draw(PLAN, "day2", seed=9, cuts=["final"], leaders=DAY2)          # no final: one sax leader
    with pytest.raises(DrawError, match="belongs to day3"):
        draw(PLAN, "day2", seed=9, cuts=["w1-two"], leaders=DAY2)


@pytest.mark.parametrize("day,kw,match", [
    ("day1", {"leaders": {"keys": ("gu_organ",)}}, "does not take"),
    ("day2", {}, "needs --leaders"),
    ("day2", {"leaders": {**DAY2, "keys": ("gu_tine_ep", "gu_organ")}}, "takes"),
    ("day2", {"leaders": {**DAY2, "kit": ("kit_marching",)}}, "not one of the kits"),
    ("day2", {"leaders": DAY2, "rerun_s2": "sax_gu_alto"}, "re-run comes only on day3"),
    ("day1", {"repeats": ["K1"]}, "repeats come only on day3"),
    ("day3", {"leaders": DAY3, "repeats": ["K1"]}, "need the --prior"),
    ("day3", {"leaders": DAY3, "rerun_s2": "sax_tuba"}, "not one of the saxes"),
])
def test_bad_job_arguments_are_refused_before_anything_is_written(day, kw, match):
    with pytest.raises(DrawError, match=match):
        draw(PLAN, day, seed=1, **kw)


def test_parse_leaders_reads_the_job_argument():
    assert parse_leaders("keys=a+b+c/kit=x/bass=y/sax=s") == {"keys": ("a", "b", "c"), "kit": ("x",),
                                                             "bass": ("y",), "sax": ("s",)}
    assert parse_leaders(None) == {} and parse_leaders("") == {}
    for bad in ("keys", "drums=x", "keys=a+a", "kit=x/kit=y", "keys=a++b"):
        with pytest.raises(DrawError):
            parse_leaders(bad)


def test_order_rounds_and_the_listening_estimate():
    r = [{"kind": "none", "loop": "a", "bpm": 90, "clips": [1, 2]}, {"kind": "anchor", "loop": "a", "bpm": 90,
                                                                     "clips": [1, 2]}]
    with pytest.raises(DrawError, match="no order"):
        order_rounds(r, Rng(1))
    ok = order_rounds(r + [{"kind": "none", "loop": "b", "bpm": 90, "clips": [1, 2]}], Rng(1))
    assert [x["loop"] for x in ok] == ["a", "b", "a"] and ok[0]["kind"] == "none"
    assert round(round_seconds(4, 76), 1) == 83.8 and round(round_seconds(2, 90), 1) == 40.0
    assert est_minutes(ok) == round(3 * round_seconds(2, 90) / 60, 1)
