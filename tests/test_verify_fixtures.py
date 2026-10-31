"""lofi_rl.verify on the repo's own loops: a pinned score table for the 6 gate-passing fixtures, one mutation per
term that moves that term and nothing else, and the 9 shootout reference loops against docs/SHOOTOUTS.md §5.

Every pinned number is derived by hand from instrument.md §5 (the arithmetic is in the comments), not read back
from the code. Sound claims: none.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

from helpers_compile import FIX, ROOT, dev_palette, passing_fixtures
from lofi_rl.brief import Brief
from lofi_rl.verify import Score, score
from lofi_rl.verify import knobs as vk

EXAMPLE = (FIX / "example.ls").read_text()
WISTFUL_D = Brief(energy="mid", mood="wistful", bpm=82, tonic=2)


@lru_cache(maxsize=None)
def sc(src: str, brief: Brief = WISTFUL_D) -> Score:
    return score(src, brief, dev_palette())


def _fit(e: float, target: str) -> float:
    lo, hi = {"low": (0.0, 0.38), "mid": (0.38, 0.59), "high": (0.59, 1.0)}[target]
    return math.exp(-max(lo - e, 0.0, e - hi) / 0.04)


# E = .35 min(1, H/192) + .15 n_X/n_nonghost + .25 min(1, n_keys/48) + .25 min(1, n_bass/48)
E_EXAMPLE = 0.35 * 122 / 192 + 0.15 * 16 / 118 + 0.25 * 26 / 48 + 0.25 * 24 / 48
E_ONE_BAR = 0.35 * 104 / 192 + 0.0 + 0.25 * 24 / 48 + 0.25 * 32 / 48
E_SHAKER = 0.35 * 122 / 192 + 0.15 * 8 / 104 + 0.25 * 14 / 48 + 0.25 * 23 / 48
E_SPAN1 = 0.35 * 140.5 / 192 + 0.0 + 0.25 * 28 / 48 + 0.25 * 20 / 48
ALL_ONE = dict(f_bb=1, f_k1=1, f_lock=1, f_tk=1, groove=1, f_long=1, f_land=1, f_open=1, f_ans=1, f_breath=1, sax=1,
               f_new=1, f_seam=1, p_chrom=0)

# fixture -> (brief, theory parts that differ from ALL_ONE, E, (aeolian, dorian, ionian), p_sax, repairs)
TABLE = {
    # instrument.md §1: "It scores 1.0 on every prior in section 5." H = kick 28 + rim 16 + 8 ghosts/2 + hat 69 +
    # ohat 5 = 122; 16 of 118 non-ghost hits are accents (the rim's X); comp 26 strikes, bass 24.
    "example.ls": (WISTFUL_D, {}, E_EXAMPLE, (26, 34, 14), 0.0, 1),
    "example_leniency.ls": (WISTFUL_D, {}, E_EXAMPLE, (26, 34, 14), 0.0, 2),
    # the base rim in lowercase: only @6's two X remain accents
    "leniency_lower_x.ls": (WISTFUL_D, {}, 0.35 * 122 / 192 + 0.15 * 2 / 118 + 0.25 * 26 / 48 + 0.25 * 24 / 48,
                            (26, 34, 14), 0.0, 1),
    # IIm9 V13 in G ionian. 6 notes of 8+ steps (f_long 0); B ends on the tonic (f_open 0); B keeps A's rhythm
    # (J = 1, f_ans 0); A rests 4/64 (0), B 12/64 (1); 1-bar bases (f_new 1/7). Kick 24 + snare 16 + hat 64;
    # comp 3 and bass 4 strikes a bar.
    "one_bar_base.ls": (Brief(energy="low", mood="hopeful", bpm=76, tonic=7),
                        dict(f_long=0, f_open=0, f_ans=0, f_breath=0.5, sax=0.25 + 0.15 * 0.5, f_new=1 / 7),
                        E_ONE_BAR, (14, 22, 34), 0.0, 2),
    # bVImaj7 bVII13 in A aeolian, bass: lock. 8 long notes (0); same rhythm in A and B (0); rests 6/64 in each
    # (0); bar 5 repeats bar 1 (f_new 6/7). H = kick 24.5 + rim 18 + hat 36 + shaker 43.5; 8 kick X of 104
    # non-ghost; comp 14 strikes, bass 23 (lock plus @2 and @7).
    "shaker_edits.ls": (Brief(energy="high", mood="melancholy", bpm=88, tonic=9),
                        dict(f_long=0, f_ans=0, f_breath=0, sax=0.25 + 0.20, f_new=6 / 7),
                        E_SHAKER, (32, 16, 16), 0.0, 1),
    # Im11 IIm7 span 1 in F dorian. A rests 6/64 (0), B 8/64 (1); bars 3, 4, 6, 7, 8 repeat bars 1-2, bar 5 has
    # its snare edit (f_new 2/7). Its A spans Ab2-G4 (MIDI 44-72): 28 semitones, 9 over the band, so P_sax's
    # wide part is capped at 0.1. H = kick 24 + snare 18.5 + hat 94 + ohat 4; comp 28, bass 20.
    "span1_dotted.ls": (Brief(energy="mid", mood="wistful", bpm=70, tonic=5),
                        dict(f_breath=0.5, sax=1 - 0.15 * 0.5, f_new=2 / 7),
                        E_SPAN1, (28, 32, 20), 0.1, 2),
}


def test_the_table_covers_every_gate_passing_fixture():
    assert sorted(TABLE) == passing_fixtures()


@pytest.mark.parametrize("name", sorted(TABLE))
def test_pinned_score_table(name):
    brief, changed, e, (aeo, dor, ion), p_sax, repairs = TABLE[name]
    s = sc((FIX / name).read_text(), brief)
    assert s.gate_ok and s.gate is None and s.repairs == repairs and s.p_fix == 0.0
    want = {**ALL_ONE, **changed}
    for key, value in want.items():
        assert s.theory_parts[key] == pytest.approx(value), key
    p = s.theory_parts
    theory = 0.3 * p["groove"] + 0.3 * p["sax"] + 0.2 * p["f_new"] + 0.2 * p["f_seam"] - p["p_chrom"]
    assert s.theory == pytest.approx(min(1.0, max(0.0, theory)))
    k = s.knob_parts
    assert k["E"] == pytest.approx(e)
    assert k["f_energy"] == pytest.approx(_fit(e, brief.energy))
    assert k["energy_match"] == float(vk.energy_band(e) == brief.energy)
    assert (k["mood_aeolian"], k["mood_dorian"], k["mood_ionian"]) == (aeo, dor, ion)
    target = {"aeolian": aeo, "dorian": dor, "ionian": ion}[brief.mode]
    margin = target - max(v for m, v in (("aeolian", aeo), ("dorian", dor), ("ionian", ion)) if m != brief.mode)
    assert k["mood_margin"] == margin and k["f_mood"] == float(margin >= 2)
    assert s.knobs == pytest.approx((k["f_energy"] + k["f_mood"]) / 2)
    assert s.p_sax == pytest.approx(p_sax)
    assert s.penalties == {"bars": 0.0, "len": 0.0}


def test_the_leniency_fixture_scores_as_the_example():
    """Same loop after normalization: only the repair count and the length differ."""
    a, b = sc(EXAMPLE), sc((FIX / "example_leniency.ls").read_text())
    assert (a.theory_parts, a.knob_parts, a.p_sax, a.digest) == (b.theory_parts, b.knob_parts, b.p_sax, b.digest)


# ------------------------------------------------------------------------------------------------ mutations
def _edit(*pairs: tuple[str, str]) -> str:
    out = EXAMPLE
    for old, new in pairs:
        assert out.count(old) == 1, old
        out = out.replace(old, new)
    return out


A_LINE = "A: q5 e6 e5 q3 e2 e1 | h3 er e4 e5 e6 | h.5 er e4 | q3 q2 hr\n"
B_LINE = "B: q1' e7 e1' q5 q3 | er e3 e4 eb5 h5 | qr e4 e3 q2 q1 | h.6 qr\n"
EDITS = ("@3 hat: x.x.x.x.x.x.x.xx\n", "@4 kick: x......x..x...x.\n", "@5 comp: x-----.x--x-..x-\n",
         "@6 rim: ....X..g....X.g.\n", "@7 ohat: ......x.........\n", "@8 comp: x-----.x----x-x-\n")
MUTATIONS = {
    # term: (the example with one change, the term's value after it). Every other theory part stays at its
    # example value (1, or p_chrom 0); the composites groove and sax follow their terms.
    "f_bb": (_edit(("rim: ....X.......X... ", "rim: ....X.......g... ")), 0.5),         # bars 1 3 5 7: no beat 4
    "f_k1": (_edit(("kick: x......x..x..... x......x.x....x.", "kick: x......x..x..... .......x.x....x.")),
             5 / 8),                                                                    # bars 2 6 8; lock 16/24
    "f_lock": (_edit(("bass: x-------..x-x--- x-----.x-x------", "bass: ..x.....x....x.. ..x...x....x...x")),
               0.0),                                                                    # no bass on a kick
    "f_tk": (_edit(("hat: x.x.x.x.x.x.x.x. ", "hat: x...x...x....... ")), 5 / 8),        # 3 hats in bars 1 5 7
    "f_long": (_edit(("| q3 q2 hr\n", "| h3 hr\n")), 0.5),                              # 5 long notes
    "f_land": (_edit(("A: q5 e6", "A: q4 e6")), 0.75),                                  # G over Dm9
    "f_open": (_edit(("h.6 qr", "h.1 qr")), 0.0),                                       # the answer ends on D
    "f_ans": (_edit((B_LINE, "B" + A_LINE[1:])), 0.0),                                  # B repeats A
    "f_breath": (_edit((A_LINE, "A: q5 e6 e5 q3 e2 e1 | h3 e3 e4 e5 e6 | h.5 e5 e4 | q3 q2 q3 q2\n")), 0.5),
    "f_new": (_edit(*((e, "") for e in EDITS)), 3 / 7),                                 # 2-bar bases, no edits
    "f_seam": (EXAMPLE + "@8 snare: ........XXXXXXXX\n", 0.0),                          # §5's fill fixture
    "p_chrom": (_edit(("| q3 q2 hr\n", "| q3 q#1 hr\n")), 0.1),                         # Eb over G13, then a rest
}
COMPOSITE = {"groove": ("f_bb", "f_k1", "f_lock", "f_tk"), "sax": ("f_long", "f_land", "f_open", "f_ans", "f_breath")}


@pytest.mark.parametrize("term", sorted(MUTATIONS))
def test_each_mutation_moves_only_its_term(term):
    src, value = MUTATIONS[term]
    base, s = sc(EXAMPLE), sc(src)
    assert s.gate_ok, s.gate
    assert s.theory_parts[term] == pytest.approx(value)
    moved = {term} | {c for c, parts in COMPOSITE.items() if term in parts}
    for key, before in base.theory_parts.items():
        if key not in moved:
            assert s.theory_parts[key] == before, key
    assert s.theory < base.theory == 1.0


def test_knob_mutations_follow_the_brief():
    """The example's E (0.503) sits in the provisional mid band: a low or high brief costs exp(-d / 0.04), and a
    melancholy brief fails the mood check (aeolian 26 against dorian 34)."""
    for energy in ("low", "high"):
        s = sc(EXAMPLE, Brief(energy=energy, mood="wistful", bpm=82, tonic=2))
        assert s.knob_parts["f_energy"] == pytest.approx(_fit(E_EXAMPLE, energy)) and s.knob_parts["f_energy"] < 0.2
        assert s.knob_parts["energy_match"] == 0.0 and s.theory == 1.0
    s = sc(EXAMPLE, Brief(energy="mid", mood="melancholy", bpm=82, tonic=2))
    assert s.knob_parts["f_mood"] == 0.0 and s.knobs == 0.5 and s.knob_parts["mood_margin"] == 26 - 34


def test_penalty_mutations():
    busy = _edit((A_LINE, "A: e5 e6 e5 e3 e2 e1 e2 e3 | e5 e6 e5 e3 e2 e1 e2 e3 | q5 q6 q5 q3 | h2 hr\n"))
    assert sc(busy).p_sax == pytest.approx(0.0125 * (21 - 16))
    long = EXAMPLE + "# " + "x" * (1680 - len(EXAMPLE) - 3) + "\n"
    assert len(long) == 1680 and sc(long).penalties["len"] == pytest.approx(0.24)


# ------------------------------------------------------------------------------------------------ reference loops
SHOOTOUT = ROOT / "shootouts" / "loops"
# docs/SHOOTOUTS.md §5, column E, computed there from the text by the shootout workflow's own code
SHOOTOUT_E = {"l1_melancholy_low": 0.289, "l2_melancholy_mid": 0.462, "l3_melancholy_high": 0.720,
              "l4_wistful_low": 0.297, "l5_wistful_mid": 0.478, "l6_wistful_high": 0.717,
              "l7_hopeful_low": 0.294, "l8_hopeful_mid": 0.465, "l9_hopeful_high": 0.700}


def _shootout_briefs() -> dict[str, Brief]:
    doc = yaml.safe_load((SHOOTOUT / "briefs.yaml").read_text())["loops"]
    return {name: Brief(energy=b["energy"], mood=b["mood"], bpm=b["bpm"], tonic=b["tonic"]) for name, b in doc.items()}


@pytest.mark.parametrize("name", sorted(SHOOTOUT_E))
def test_the_shootout_reference_loops_score_as_shootouts_md_says(name):
    """SHOOTOUTS.md §5 "Own vamp": groove 1, S 1, f_new 1, f_seam 1, P_chrom 0, theory 1 in every loop; E as its
    table gives it. Each loop's E also lands in its own energy cell under the provisional bands, which are the
    midpoints of the gaps between these cells, and its sax line keeps inside P_sax's provisional bands."""
    brief = _shootout_briefs()[name]
    s = sc((SHOOTOUT / f"{name}.ls").read_text(), brief)
    assert s.gate_ok and s.theory == 1.0
    assert all(s.theory_parts[k] == v for k, v in ALL_ONE.items())
    assert round(s.knob_parts["E"], 3) == SHOOTOUT_E[name]
    assert s.knob_parts["energy_match"] == 1.0 and s.knob_parts["f_mood"] == 1.0 and s.knobs == 1.0
    assert s.p_sax == 0.0 and s.penalties == {"bars": 0.0, "len": 0.0}


def test_provisional_energy_bands_are_the_midpoints_between_the_reference_cells():
    by_energy: dict[str, list[float]] = {"low": [], "mid": [], "high": []}
    for name, brief in _shootout_briefs().items():
        by_energy[brief.energy].append(SHOOTOUT_E[name])
    lo = (max(by_energy["low"]) + min(by_energy["mid"])) / 2
    hi = (max(by_energy["mid"]) + min(by_energy["high"])) / 2
    assert vk.ENERGY_BANDS == (round(lo, 2), round(hi, 2))
    assert vk.energy_band(E_EXAMPLE) == "mid"                      # instrument.md §1's loop is energy mid


def test_every_fixture_and_reference_loop_is_within_the_sax_bands_but_span1():
    """P_sax's provisional bands come from the hand-written loops (Shresth's Good-or-better loops do not exist
    yet); only span1_dotted.ls, built to stress octave marks, leaves them (its A spans 28 semitones)."""
    loops = [(FIX / n, TABLE[n][0]) for n in TABLE] + [(SHOOTOUT / f"{n}.ls", b) for n, b in _shootout_briefs().items()]
    charged = sorted(Path(p).name for p, b in loops if sc(Path(p).read_text(), b).p_sax > 0)
    assert charged == ["span1_dotted.ls"]
