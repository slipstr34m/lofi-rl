"""shootouts/plan.yaml against docs/SHOOTOUTS.md §2, §4 and §6 (§7.8 "Plan").

The round tables, the candidate tables, the vamp table and the batch table are written out here a second time,
from the doc, independently of plan.yaml, so a slip in either transcription fails. The design properties the doc
states in words (twins meet 3 times, Latin squares, tile sums, each candidate heard in 2+ loops on distinct tiles)
are recomputed from the plan's slots. The plan is public (§2.3): its tiles hold symbols only, never an option."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pytest
import yaml

from lofi_rl.dsl import expand, parse
from lofi_rl.dsl.vocab import LANES
from lofi_rl.palette import load_palette
from lofi_rl.shootout.plan import PlanError, load_plan, parse_plan, substitute

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "shootouts" / "plan.yaml"
DEV = ROOT / "palettes" / "dev_gu.yaml"
PLAN = load_plan(PLAN_PATH)
DOC = yaml.safe_load(PLAN_PATH.read_text())

# ------------------------------------------------------------------ the doc's tables, written out again
# (loop, tiles 1-4) per round slot: §4.1 K1, K2; §4.2 R1, R2, R3a, R3b; §4.3 the vamp rows; §4.4 S1, S1f, S2a,
# S2b; §4.5 W1, W2.
DOC_SLOTS = {
    "K1.1": ("L5", "P1a P1b P2a P2b"), "K1.2": ("L1", "P1b P3a P3b P1a"), "K1.3": ("L9", "P4a P4b P1a P1b"),
    "K1.4": ("L5", "P3a P3b P4a P4b"), "K1.5": ("L1", "P2b P4a P4b P2a"), "K1.6": ("L9", "P2a P2b P3a P3b"),
    "K2.1": ("L3", "u1 u2 u3"), "K2.2": ("L8", "u2 u3 u1"),
    "R1.1": ("L3", "k1 k2 k3"), "R1.2": ("L4", "k2 k3 k1"), "R1.3": ("L8", "k3 k1 k2"),
    "R2.1": ("L6", "b1 b2 b3"), "R2.2": ("L7", "b2 b3 b1"), "R2.3": ("L2", "b3 b1 b2"),
    "R3a.1": ("L9", "w1 w2 w3 w4"), "R3a.2": ("L4", "w4 w3 w2 w1"),
    "R3b.1": ("L3", "n1 n2 n3"), "R3b.2": ("L7", "n2 n3 n1"), "R3b.3": ("L5", "n3 n1 n2"),
    "V_melancholy.1": ("L1", "v1 v2 v3 v4"), "V_melancholy.2": ("L2", "v2 v1 v4 v3"),
    "V_melancholy.3": ("L3", "v3 v4 v1 v2"),
    "V_wistful.1": ("L4", "v1 v2 v3 v4"), "V_wistful.2": ("L5", "v2 v1 v4 v3"), "V_wistful.3": ("L6", "v3 v4 v1 v2"),
    "V_hopeful.1": ("L7", "v1 v2 v3 v4"), "V_hopeful.2": ("L8", "v2 v1 v4 v3"), "V_hopeful.3": ("L9", "v3 v4 v1 v2"),
    "S1.1": ("L4", "t3 a1 a2 a3"), "S1.2": ("L3", "a3 t1 t2 t3"), "S1.3": ("L1", "t1 a2 a3 a1"),
    "S1.4": ("L5", "a1 t2 t3 t1"), "S1.5": ("L9", "t2 a3 a1 a2"), "S1.6": ("L7", "a2 t3 t1 t2"),
    "S1f.1": ("L2", "f1 f2"), "S1f.2": ("L6", "f2 f1"), "S1f.3": ("L8", "f1 f2"),
    "S2a.1": ("L1", "m1 m2 m3"), "S2a.2": ("L4", "m2 m3 m1"), "S2a.3": ("L9", "m3 m1 m2"),
    "S2b.1": ("L3", "g1 g2 g3"), "S2b.2": ("L5", "g2 g3 g1"), "S2b.3": ("L7", "g3 g1 g2"),
    "W1.1": ("L1", "z1 z2 z3"), "W1.2": ("L7", "z2 z3 z1"), "W1.3": ("L5", "z3 z1 z2"),
    "W2.1": ("L2", "y1 y2 y3 y4"), "W2.2": ("L4", "y2 y3 y4 y1"), "W2.3": ("L6", "y3 y4 y1 y2"),
    "W2.4": ("L8", "y4 y1 y2 y3"),
}
R2_KITS = {"R2.1": "k1", "R2.2": "k2", "R2.3": "k3"}
# The round tables' loop labels ("L5 wistful mid 83") against shootouts/loops/briefs.yaml.
DOC_LOOPS = {"L1": ("melancholy", "low", 76), "L2": ("melancholy", "mid", 81), "L3": ("melancholy", "high", 87),
             "L4": ("wistful", "low", 78), "L5": ("wistful", "mid", 83), "L6": ("wistful", "high", 89),
             "L7": ("hopeful", "low", 79), "L8": ("hopeful", "mid", 85), "L9": ("hopeful", "high", 90)}
DAYS = {1: {"K1", "R1", "R2", "S1"},
        2: {"R3a", "R3b", "S1f", "S2a", "S2b", "K2", "V_melancholy", "V_wistful", "V_hopeful"},
        3: {"W1", "W2"}}
# §4.1, §4.2, §4.4: option -> (bus, bank, prog, family or register, the source file under assets/big, or gu).
GU = "gu"
DOC_CANDIDATES = {
    "jrhodes3d": ("keys", 0, 0, "rhodes", "converted/jrhodes3d-mono.sf2"),
    "gu_tine_ep": ("keys", 0, 4, "rhodes", GU),
    "fp_fm_ep1": ("keys", 0, 0, "other_ep", "freepats-fm-piano-1/x/FM-Piano1 SF2-20190916/FM-Piano1 20190916.sf2"),
    "gs_wurli_ep200": ("keys", 0, 0, "other_ep", "converted/gsullivan-wurlitzer.sf2"),
    "fp_upright_kw": ("keys", 0, 0, "piano",
                      "freepats-upright-piano-kw/x/UprightPianoKW-SF2-20220221/UprightPianoKW-20220221.sf2"),
    "vcsl_upright_knight": ("keys", 0, 0, "piano", "converted/vcsl-upright-knight.sf2"),
    "fp_fsbs_jazz_guitar": ("keys", 0, 0, "guitar_organ", "freepats-fsbs-jazz-guitar/x/EGuitarFSBS-jazz bridge small "
                            "SF2-20260807/EGuitarFSBS-jazz bridge small 20260807.sf2"),
    "gu_organ": ("keys", 0, 16, "guitar_organ", GU),
    "bass_vsco2_pizz": ("bass", 0, 0, None, "converted/vsco2-contrabass-pizz.sf2"),
    "bass_fp_finger_yr": ("bass", 0, 0, None, "freepats-finger-bass-yr/x/FingerBassYR SF2-20190930/FingerBassYR "
                          "20190930.sf2"),
    "bass_gu_acoustic": ("bass", 0, 32, None, GU),
    "sax_fp_tenor": ("sax", 0, 0, "tenor",
                     "freepats-tenor-sax/x/TenorSaxophone-SF2-20200717/TenorSaxophone-20200717.sf2"),
    "sax_weresax_alto": ("sax", 0, 0, "alto", "converted/weresax-alto.sf2"),
    "sax_mtg_tenor": ("sax", 0, 0, "tenor", "converted/mtg-tenor.sf2"),
    "sax_mtg_alto": ("sax", 0, 0, "alto", "converted/mtg-alto.sf2"),
    "sax_gu_tenor": ("sax", 0, 66, "tenor", GU),
    "sax_gu_alto": ("sax", 0, 65, "alto", GU),
}
DOC_KITS = ("kit_virtuosity", "kit_swirly", "kit_gu_jazz")
# §4.3: vamp -> (mood, chords, span, borrowed)
DOC_VAMPS = {
    "aruarian": ("hopeful", "IIm9 V13", 2, False), "royal_half": ("hopeful", "IVmaj7 IIIm7", 1, False),
    "two_of_us": ("hopeful", "IVmaj7 III7", 2, False), "lift": ("hopeful", "Imaj9 IVmaj9", 2, False),
    "i_iv": ("wistful", "Im9 IV13", 2, False), "pulse": ("wistful", "Im9 IV9", 1, False),
    "step": ("wistful", "Im11 IIm7", 2, False), "float": ("wistful", "Im9 bVIImaj7", 2, True),
    "drift": ("melancholy", "Im9 bVImaj7", 2, True), "plagal": ("melancholy", "Im9 IVm9", 2, False),
    "backdoor": ("melancholy", "IVm9 bVII13", 2, True), "rise": ("melancholy", "bVImaj7 bVII13", 2, True),
}
# §6: (date, names, rounds per stage, anchors, category, note, est_min plan / full)
DOC_BATCHES = [
    ("2026-11-03", ("b1103-1-keys",), {"K1": 6}, 0, "keys", "Which keys do you like best?", (7.8, 9.3)),
    ("2026-11-03", ("b1103-2-groove",), {"R1": 3, "R2": 3}, 0, "per_round", "Which drums and bass do you like best?",
     (6.0, 6.9)),
    ("2026-11-03", ("b1103-3-sax",), {"S1": 6}, 1, "sax", "Which sax do you like best?", (9.2, 11.1)),
    ("2026-11-04", ("b1104-1-groove",), {"R3a": 2, "R3b": 3}, 0, "per_round", "Which groove do you like best?",
     (5.6, 6.5)),
    ("2026-11-04", ("b1104-2-sax",), {"S1f": 3, "S2a": 3, "S2b": 3}, 1, "per_round", "Which sax do you like best?",
     (9.2, 10.5)),
    ("2026-11-04", ("b1104-3-keys",), {"K2": 2}, 0, "keys", "Which keys do you like best?", (1.9, 2.2)),
    ("2026-11-04", ("b1104-4-vamps",), {"V_melancholy": 3, "V_wistful": 3, "V_hopeful": 3}, 0, "vamps",
     "Which chords do you like best? There is no sax in this batch.", (11.6, 13.9)),
    ("2026-11-06", ("b1106-1-mix", "b1106-2-mix"), {"W1": 3, "W2": 4}, 1, "per_round",
     "Pick the best and the worst in each round.", (7.7, 9.1)),
]


def slots_of(stage: str):
    return PLAN.stages[stage].slots


def tile_sums(stage: str) -> dict[str, int]:
    sums: Counter = Counter()
    for slot in slots_of(stage):
        for pos, sym in enumerate(slot.tiles, 1):
            sums[sym] += pos
    return dict(sums)


# ------------------------------------------------------------------------------------- the tables
def test_every_round_slot_is_the_docs():
    got = {slot.id: (slot.loop, " ".join(slot.tiles)) for st in PLAN.stages.values() for slot in st.slots}
    assert got == DOC_SLOTS
    assert {s.id: s.kit for s in slots_of("R2")} == R2_KITS, "R2 plays each R1 kit symbol in one round (§4.2)"
    assert all(s.kit is None for st in PLAN.stages.values() if st.id != "R2" for s in st.slots)


def test_each_stage_has_its_day_and_the_round_count_is_58_with_the_controls():
    assert {d: {s.id for s in PLAN.stages.values() if s.day == d} for d in (1, 2, 3)} == DAYS
    assert {d: set(PLAN.days[d].stages) for d in (1, 2, 3)} == DAYS
    design = sum(len(st.slots) for st in PLAN.stages.values())
    assert design == 50
    assert design + PLAN.controls["anchors"]["count"] + PLAN.controls["repeats"]["count"] == 58   # §2.5, §2.7
    assert [len(slots_of(s)) for s in ("K1", "K2", "R1", "R2", "R3a", "R3b", "S1", "S1f", "S2a", "S2b", "W1", "W2")] \
        == [6, 2, 3, 3, 2, 3, 6, 3, 3, 3, 3, 4]


def test_loop_names_map_to_the_reference_loops_and_their_briefs():
    briefs = PLAN.briefs()
    assert set(PLAN.loop_names) == set(DOC_LOOPS)
    for name, (mood, energy, bpm) in DOC_LOOPS.items():
        loop = PLAN.loop_names[name]
        assert loop.startswith(name.lower() + "_") and (PLAN.loops_dir / f"{loop}.ls").is_file()
        b = briefs[loop]
        assert (b.mood, b.energy, b.bpm) == (mood, energy, bpm), name
        assert PLAN.loop_text(name) == (PLAN.loops_dir / f"{loop}.ls").read_text()


def test_no_option_is_placed_on_a_tile():
    # §2.3: the public plan holds symbols only; which option sits on which tile is drawn at job time.
    options = set(PLAN.candidates) | set(PLAN.variants) | set(PLAN.vamps)
    for st in PLAN.stages.values():
        for slot in st.slots:
            assert set(slot.tiles) <= set(st.symbols), slot.id
            assert not set(slot.tiles) & options, slot.id
            assert len(set(slot.tiles)) == len(slot.tiles), f"{slot.id} shows a symbol twice"
    for st in DOC["stages"].values():
        for slot_doc in st["slots"]:
            assert not set(slot_doc["tiles"]) & options
        for tiles in (st.get("tie") or {}).get("tiles", []):
            assert not set(tiles) & options


# ------------------------------------------------------------------------- design properties (§2.4, §4)
def test_k1_is_the_tournament_of_section_4_1():
    k1 = PLAN.stages["K1"]
    pairs = k1.draw["pairs"]
    assert pairs == {"P1": ["P1a", "P1b"], "P2": ["P2a", "P2b"], "P3": ["P3a", "P3b"], "P4": ["P4a", "P4b"]}
    meet = Counter(frozenset(p) for s in k1.slots for p in combinations(s.tiles, 2))
    twins = {frozenset(v) for v in pairs.values()}
    for pair, count in meet.items():
        assert count == (3 if pair in twins else 1), sorted(pair)
    assert len(meet) == 4 + 24, "every pair of the 8 meets: twins 3 times, every other pair once"
    for sym in k1.symbols:
        seen = [(s.loop, s.tiles.index(sym) + 1) for s in k1.slots if sym in s.tiles]
        assert sorted(loop for loop, _ in seen) == ["L1", "L5", "L9"], sym
        assert len({t for _, t in seen}) == 3, f"{sym} is heard on three different tiles"
    sums = tile_sums("K1")
    assert set(sums.values()) <= {6, 7, 8, 9}
    assert all(sums[a] + sums[b] == 15 for a, b in pairs.values()), "each family's members add to 15 (CD-8)"
    families = Counter(PLAN.candidates[o].family for o in k1.options)
    assert families == {"rhodes": 2, "other_ep": 2, "piano": 2, "guitar_organ": 2}


@pytest.mark.parametrize("stage", ["R1", "R2", "R3b", "S2a", "S2b", "W1", "W2"])
def test_latin_squares_put_each_option_on_every_tile_once(stage):
    st = PLAN.stages[stage]
    for sym in st.symbols:
        tiles = [s.tiles.index(sym) for s in st.slots]
        assert sorted(tiles) == list(range(len(st.symbols))), (stage, sym)
    assert len(st.slots) == len(st.symbols)


def test_r3a_plays_its_second_round_reversed_and_k2_has_sums_3_4_5():
    a, b = slots_of("R3a")
    assert b.tiles == tuple(reversed(a.tiles)) and set(tile_sums("R3a").values()) == {5}
    assert sorted(tile_sums("K2").values()) == [3, 4, 5]
    tie = PLAN.stages["K2"].tie                       # a tie at third place takes up to 4 (CD-41)
    assert tie["symbols"] == ["u1", "u2", "u3", "u4"]
    assert [tuple(t) for t in tie["tiles"]] == [("u1", "u2", "u3", "u4"), ("u4", "u3", "u2", "u1")]
    tie = PLAN.stages["S1f"].tie                      # three tied for two places: a 3x3 Latin square (§4.4)
    assert tie["symbols"] == ["f1", "f2", "f3"]
    assert [tuple(t) for t in tie["tiles"]] == [("f1", "f2", "f3"), ("f2", "f3", "f1"), ("f3", "f1", "f2")]
    assert PLAN.stages["K2"].sub == {"keys": "ep spread"} and PLAN.stages["K2"].ranks == {"of": "K1", "top": 3}
    assert PLAN.stages["S1f"].ranks == {"of": "S1", "top": 2}


def test_s1_follows_section_4_4():
    s1 = PLAN.stages["S1"]
    groups = s1.draw["groups"]
    assert groups == {"tenor": ["t1", "t2", "t3"], "alto": ["a1", "a2", "a3"]}
    reg = {sym: r for r, syms in groups.items() for sym in syms}
    for slot in s1.slots:
        out = set(s1.symbols) - set(slot.tiles)
        assert len(out) == 2 and len({reg[s] for s in out}) == 1, f"{slot.id} leaves out two of one register"
    for sym in s1.symbols:
        assert sorted(s.tiles.index(sym) for s in s1.slots if sym in s.tiles) == [0, 1, 2, 3], sym
    meet = Counter(frozenset(p) for s in s1.slots for p in combinations(s.tiles, 2))
    for a, b in combinations(s1.symbols, 2):
        assert meet[frozenset((a, b))] == (3 if reg[a] == reg[b] else 2), (a, b)
    all_altos = [s.loop for s in s1.slots if set(groups["alto"]) <= set(s.tiles)]
    assert sorted(all_altos) == ["L1", "L4", "L9"], "the all-alto rounds hold the alto-above loop L4"
    final = {s.loop for s in slots_of("S1f")}
    assert final == set(DOC_LOOPS) - {s.loop for s in s1.slots} == {"L2", "L6", "L8"}
    assert sorted(tile_sums("S1f").values()) == [4, 5]
    moods = {sym: {PLAN.briefs()[PLAN.loop_names[s.loop]].mood for s in s1.slots if sym in s.tiles}
             for sym in s1.symbols}
    assert all(len(m) == 3 for m in moods.values()), "each source covers all three moods"
    registers = Counter(PLAN.candidates[o].register for o in s1.options)
    assert registers == {"tenor": 3, "alto": 3}


def test_vamp_rounds_are_three_rows_of_a_latin_square_per_mood():
    for mood, loops in (("melancholy", ["L1", "L2", "L3"]), ("wistful", ["L4", "L5", "L6"]),
                        ("hopeful", ["L7", "L8", "L9"])):
        st = PLAN.stages[f"V_{mood}"]
        assert [s.loop for s in st.slots] == loops
        assert [PLAN.briefs()[PLAN.loop_names[s.loop]].energy for s in st.slots] == ["low", "mid", "high"]
        assert tile_sums(st.id) == {"v1": 6, "v2": 7, "v3": 8, "v4": 9}
        assert sorted(st.options) == sorted(v for v, d in DOC_VAMPS.items() if d[0] == mood)
        assert st.category == "vamps" and st.sub == {"vamp": "option"}
    central = {m: PLAN.stages[f"V_{m}"].draw for m in ("melancholy", "wistful", "hopeful")}
    assert central["melancholy"] == {"rule": "central", "options": ["plagal"], "symbols": ["v2", "v3"]}
    assert central["wistful"] == {"rule": "central", "options": ["float"], "symbols": ["v2", "v3"]}
    assert central["hopeful"] == {"rule": "any"}
    assert tile_sums("V_melancholy")["v2"] in (7, 8) and tile_sums("V_melancholy")["v3"] in (7, 8)


def test_every_candidate_is_heard_in_two_loops_on_distinct_tiles_within_its_stage():
    # §2.4 and §7.8: whatever the draw, a symbol's appearances are its option's.
    for st in PLAN.stages.values():
        for sym in st.symbols:
            seen = [(s.loop, s.tiles.index(sym)) for s in st.slots if sym in s.tiles]
            assert len({loop for loop, _ in seen}) >= 2, (st.id, sym)
            assert len({t for _, t in seen}) >= 2, (st.id, sym)


# ---------------------------------------------------------------------------------- candidates
def test_the_candidates_are_section_4s():
    keys_bass_sax = {c.id: c for c in PLAN.candidates.values() if c.bus != "drums"}
    assert set(keys_bass_sax) == set(DOC_CANDIDATES)
    base = load_palette(ROOT / PLAN.base, verify=False)
    for cid, (bus, bank, prog, extra, path) in DOC_CANDIDATES.items():
        c = PLAN.candidates[cid]
        assert (c.bus, c.bank, c.prog) == (bus, bank, prog), cid
        assert (c.family if bus == "keys" else c.register if bus == "sax" else None) == extra, cid
        src = base.sources[c.source]
        assert src.path == ("assets/soundfonts/generaluser-gs/GeneralUser-GS.sf2" if path == GU else path), cid
        assert src.mode == ("commit" if path == GU else "fetch"), cid
    kits = {c.id: c for c in PLAN.candidates.values() if c.bus == "drums"}
    assert set(kits) == set(DOC_KITS)
    assert kits["kit_gu_jazz"].archive is None, "the GU kit is the base palette's own one-shots"
    for kid in ("kit_virtuosity", "kit_swirly"):
        a = kits[kid].archive
        assert a["path"] == f"derived/kits/{kid}.tar" and a["blocks"] == f"derived/kits/{kid}.yaml", kid
        assert a["kit_map"] == f"shootouts/kits/{kid}.yaml" and "CC0" in a["license"]
    assert PLAN.stages["K1"].options == tuple(c for c in DOC_CANDIDATES if DOC_CANDIDATES[c][0] == "keys")
    assert PLAN.stages["R1"].options == DOC_KITS
    assert PLAN.stages["R2"].options == ("bass_vsco2_pizz", "bass_fp_finger_yr", "bass_gu_acoustic")
    assert set(PLAN.stages["S1"].options) == {c for c, d in DOC_CANDIDATES.items() if d[0] == "sax"}


def test_the_fixed_band_and_level_anchors_are_the_gu_instruments():
    assert PLAN.days[1].band == {"keys": "gu_tine_ep", "drums": "kit_gu_jazz", "bass": "bass_gu_acoustic",
                                 "sax": "sax_gu_tenor"}
    lv = PLAN.levels
    assert lv.anchors == PLAN.days[1].band and lv.sax_target == "sax_gu_tenor" and lv.zero_trim_source == "gu"
    assert (lv.linear_drop_db, lv.linear_tol_db, lv.median_tol_lu, lv.loop_tol_lu) == (6.02, 0.1, 0.05, 0.1)
    dev = load_palette(DEV, verify=False)
    for reg in ("tenor", "alto"):
        r, d = PLAN.registers[reg], dev.sax[reg]
        assert (tuple(r["play"]), tuple(r["comfort"]), tuple(r["home"])) == (d.play, d.comfort, d.home), reg


def test_the_vamp_table_and_each_loops_own_vamp():
    assert {v.id: (v.mood, v.chords, v.span, v.borrowed) for v in PLAN.vamps.values()} == DOC_VAMPS
    for vid, v in PLAN.vamps.items():
        assert v.borrowed == any(c.startswith(("bVI", "bVII")) for c in v.chords.split()), vid
    briefs = yaml.safe_load((PLAN.loops_dir / "briefs.yaml").read_text())["loops"]
    for name, loop in PLAN.loop_names.items():
        own = PLAN.vamps[briefs[loop]["vamp"]]
        assert own.mood == briefs[loop]["mood"]
        assert f"vamp: {own.chords} span {own.span}\n" in PLAN.loop_text(name), name


# ---------------------------------------------------------------------------- substitutions (§5, §7.8)
def f_new(text: str, mode: str) -> float:
    """instrument.md §5 'something new every bar': the share of bars 2-8 whose (chord, drum lanes, bass, comp)
    signature no earlier bar has."""
    res = parse(text, mode=mode)
    bars = expand(res.ast)
    sig = [(bars.chords[b], tuple(bars.drums[lane][b] for lane in LANES), bars.bass[b], bars.comp[b])
           for b in range(8)]
    return sum(sig[b] not in sig[:b] for b in range(1, 8)) / 7


def substituted_texts():
    """Every loop text a substitution produces: K2's spread voicing, and each vamp of the mood on a vamp loop."""
    for st in PLAN.stages.values():
        for slot in st.slots:
            text = PLAN.loop_text(slot.loop)
            if st.sub == {"keys": "ep spread"}:
                yield slot.id, None, substitute(text, keys="ep spread")
            elif st.sub == {"vamp": "option"}:
                for vid in st.options:
                    v = PLAN.vamps[vid]
                    yield slot.id, vid, substitute(text, vamp=f"{v.chords} span {v.span}")


def test_every_substitution_keeps_g1_to_g6_and_something_new_every_bar():
    vocab = load_palette(ROOT / PLAN.base, verify=False).vocab()
    seen = 0
    for slot_id, vid, text in substituted_texts():
        stage = PLAN.stages[slot_id.split(".")[0]]
        slot = next(s for s in stage.slots if s.id == slot_id)
        mood = PLAN.briefs()[PLAN.loop_names[slot.loop]].mood
        res = parse(text, vocab=vocab, mode=mood)
        assert res.gate.ok and not res.repairs, (slot_id, vid, res.gate, res.repairs)
        assert f_new(text, mood) == 1.0, (slot_id, vid)
        seen += 1
    assert seen == 2 + 36, "K2's two rounds and the 9 vamp loops under the 4 vamps of their mood"


def test_substitute_replaces_exactly_one_line():
    text = PLAN.loop_text("L1")
    out = substitute(text, keys="ep spread", vamp="Im9 IVm9 span 2")
    diff = [(a, b) for a, b in zip(text.splitlines(), out.splitlines()) if a != b]
    assert diff == [("vamp: Im9 bVImaj7 span 2", "vamp: Im9 IVm9 span 2"), ("keys: ep rootless", "keys: ep spread")]
    with pytest.raises(PlanError, match="no 'sax:' line"):
        substitute(text, sax="tenor")


# ----------------------------------------------------------------------------------- batches (§6)
def _words(value: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", value.lower()) if len(w) >= 3}


def test_the_batches_are_section_6s():
    got = [(b.date, b.names, dict(b.rounds), b.anchors, b.category, b.note, (b.est_min_plan, b.est_min_full))
           for b in PLAN.batches]
    assert got == DOC_BATCHES
    names = [n for b in PLAN.batches for n in b.names]
    assert names == sorted(names), "batch names sort in serving order (CD-49)"
    for b in PLAN.batches:
        for stage, count in b.rounds.items():
            assert len(slots_of(stage)) == count
            assert PLAN.stages[stage].day == b.day
        options = {w for st in b.rounds for o in PLAN.stages[st].options for w in _words(o)}
        category = _words(b.category) | {w for st in b.rounds for w in _words(PLAN.stages[st].category)}
        assert not (_words(b.note) & options) - category, (b.names, "the note must not name an option")
    assert {s: PLAN.stages[s].category for s in ("K1", "R1", "R2", "S1", "S2a", "S2b", "R3a", "R3b")} == {
        "K1": "keys", "R1": "kits", "R2": "bass", "S1": "sax", "S2a": "sax_warmth", "S2b": "sax_timing",
        "R3a": "swing", "R3b": "humanize"}
    # D-013 anchors: one per day, only in batches that have a sax (§2.5).
    assert sum(b.anchors for b in PLAN.batches) == PLAN.controls["anchors"]["count"] == 3
    assert sorted(b.day for b in PLAN.batches if b.anchors) == [1, 2, 3]
    assert all("sax" in n or "mix" in n for b in PLAN.batches if b.anchors for n in b.names)
    assert PLAN.controls["repeats"]["groups"] == {
        "keys": ["K1", "K2"], "groove": ["R1", "R2", "R3a", "R3b"],
        "vamps": ["V_melancholy", "V_wistful", "V_hopeful"], "sax_sources": ["S1", "S1f"],
        "sax_variants": ["S2a", "S2b"]}


def test_the_cut_options_are_section_2_7s():
    cuts = {c.id: c for c in PLAN.cuts}
    assert [c.id for c in sorted(PLAN.cuts, key=lambda c: (c.day, c.order))] == \
        ["vamps-high", "latin3", "final", "w1-two", "w2-two"]
    assert cuts["vamps-high"].drop == ("V_melancholy.3", "V_wistful.3", "V_hopeful.3")
    assert cuts["vamps-high"].retile == {"V_melancholy.2": ["v4", "v3", "v2", "v1"],
                                         "V_wistful.2": ["v4", "v3", "v2", "v1"],
                                         "V_hopeful.2": ["v4", "v3", "v2", "v1"]}
    assert cuts["latin3"].drop == ("R3b.3", "S2a.3", "S2b.3")
    assert cuts["latin3"].default_on == {"R3b": ["n1", "humanize_off"], "S2a": ["m1", "sax_plain"],
                                         "S2b": ["g1", "timing_grid"]}
    assert cuts["final"].drop == tuple(s.id for s in slots_of("S1f"))
    assert cuts["w1-two"].drop == ("W1.3",) and cuts["w1-two"].default_on == {"W1": ["z1", "excerpt_e1"]}
    assert cuts["w2-two"].drop == ("W2.3", "W2.4") and cuts["w2-two"].retile == {"W2.2": ["y4", "y3", "y2", "y1"]}
    # The default sits on the middle tile sum once the third round goes (§2.7).
    for stage, (sym, _) in cuts["latin3"].default_on.items():
        kept = [s for s in slots_of(stage) if s.id not in cuts["latin3"].drop]
        sums = Counter()
        for s in kept:
            for pos, t in enumerate(s.tiles, 1):
                sums[t] += pos
        assert sums[sym] == sorted(sums.values())[1], stage


# --------------------------------------------------------------------------------------- the loader
@pytest.mark.parametrize("edit, message", [
    (lambda d: d["stages"]["K1"]["slots"][0].update(tiles=["P1a", "P1b", "P2a", "jrhodes3d"]), "not a symbol"),
    (lambda d: d["stages"]["K1"]["slots"][0].update(loop="L10"), "L10"),
    (lambda d: d["stages"]["K1"]["options"].append("nonesuch"), "nonesuch"),
    (lambda d: d["stages"]["K1"].update(colour="blue"), "unknown key"),
    (lambda d: d["days"][1]["band"].update(keys="nonesuch"), "nonesuch"),
    (lambda d: d["batches"][0]["rounds"].update(K9=1), "K9"),
])
def test_the_loader_refuses_a_broken_plan(edit, message):
    import copy
    doc = copy.deepcopy(DOC)
    edit(doc)
    with pytest.raises(PlanError, match=message):
        parse_plan(doc, PLAN_PATH)
