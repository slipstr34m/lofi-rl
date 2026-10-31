"""Modes, sax degrees and chord tones (instrument.md section 3, Pitch mapping and the chord tones table)."""
import pytest

from lofi_rl.dsl.ast import Chord
from lofi_rl.dsl.vocab import NUMERALS, QUALITIES, diatonic_quality
from lofi_rl.theory.chords import QUALITY, chord_pcs, quality, root_pc
from lofi_rl.theory.modes import BLUE_NOTES, MODES, degree_offset, home_tonic, mode_of

# The section 3 tables, copied here so the code is checked against the spec rather than against itself.
SPEC_MODES = {"aeolian": [0, 2, 3, 5, 7, 8, 10], "dorian": [0, 2, 3, 5, 7, 9, 10], "ionian": [0, 2, 4, 5, 7, 9, 11]}
SPEC_ROOTS = {"I": 0, "bII": 1, "II": 2, "bIII": 3, "III": 4, "IV": 5, "V": 7, "bVI": 8, "VI": 9, "bVII": 10,
              "VII": 11}
SPEC_TONES = {
    "maj7": [0, 4, 7, 11], "maj9": [0, 4, 7, 11, 14], "maj7#11": [0, 4, 7, 11, 14, 18],
    "7": [0, 4, 7, 10], "9": [0, 4, 7, 10, 14], "13": [0, 4, 7, 10, 14, 21],
    "7sus": [0, 5, 7, 10, 14], "7b9": [0, 4, 7, 10, 13], "7#9": [0, 4, 7, 10, 15],
    "m7": [0, 3, 7, 10], "m9": [0, 3, 7, 10, 14], "m11": [0, 3, 7, 10, 14, 17],
    "m7b5": [0, 3, 6, 10, 17], "dim7": [0, 3, 6, 9],
}


def test_modes_and_moods():
    assert {m: list(s) for m, s in MODES.items()} == SPEC_MODES
    assert [mode_of(m) for m in ("melancholy", "wistful", "hopeful")] == ["aeolian", "dorian", "ionian"]
    with pytest.raises(ValueError):
        mode_of("dorian")
    assert BLUE_NOTES == (3, 6, 10)


def test_degree_offsets_with_accidentals_and_octaves():
    assert [degree_offset("dorian", d) for d in range(1, 8)] == SPEC_MODES["dorian"]
    assert degree_offset("dorian", 5, -1) == 6                      # eb5: the blue b5
    assert degree_offset("ionian", 3, -1) == 3 and degree_offset("ionian", 7, -1) == 10
    assert degree_offset("aeolian", 1, 0, 1) == 12 and degree_offset("aeolian", 3, 0, -1) == -9
    assert degree_offset("wistful", 6) == 9
    with pytest.raises(ValueError):
        degree_offset("dorian", 8)


@pytest.mark.parametrize("home, tonic, expected", [
    ((50, 61), 2, 50), ((50, 61), 0, 60), ((50, 61), 11, 59), ((50, 61), 1, 61),     # tenor: D3 .. C#4
    ((57, 68), 9, 57), ((57, 68), 2, 62), ((57, 68), 8, 68),                         # alto: A3 .. G#4
])
def test_home_tonic_places_the_tonic_in_the_band(home, tonic, expected):
    assert home_tonic(tonic, home) == expected
    assert all(home[0] <= home_tonic(t, home) <= home[1] for t in range(12))
    assert len({home_tonic(t, home) for t in range(12)}) == 12


def test_chord_tones_match_the_table():
    assert tuple(QUALITY) == QUALITIES
    assert {q: list(QUALITY[q].tones) for q in QUALITIES} == SPEC_TONES
    families = {q: QUALITY[q].family for q in QUALITIES}
    assert families == {"maj7": "maj", "maj9": "maj", "maj7#11": "maj", "7": "dom", "9": "dom", "13": "dom",
                        "7sus": "sus", "7b9": "dom", "7#9": "dom", "m7": "min", "m9": "min", "m11": "min",
                        "m7b5": "half", "dim7": "dim"}
    with pytest.raises(ValueError):
        quality("6")


def test_roots_and_pitch_classes():
    for tonic in range(12):
        assert {n: root_pc(n, tonic) for n in NUMERALS} == {n: (tonic + s) % 12 for n, s in SPEC_ROOTS.items()}
    assert chord_pcs(Chord("I", "m9"), 2) == {2, 5, 9, 0, 4}                 # Dm9: D F A C E
    assert chord_pcs(Chord("IV", "13"), 2) == {7, 11, 2, 5, 9, 4}            # G13: G B D F A E
    assert chord_pcs(Chord("bVII", "maj7"), 0) == {10, 2, 5, 9}              # Bbmaj7 in C


@pytest.mark.parametrize("mode", ["aeolian", "dorian", "ionian"])
def test_diatonic_sevenths_lie_in_their_mode(mode):
    # The parser's bare-numeral repair (dsl.vocab.diatonic_quality) and the compiler's tones agree.
    scale = set(SPEC_MODES[mode])
    found = 0
    for numeral in NUMERALS:
        q = diatonic_quality(numeral, mode)
        if q is None:
            continue
        found += 1
        root = SPEC_ROOTS[numeral]
        assert {(root + t) % 12 for t in QUALITY[q].tones[:4]} <= scale, (mode, numeral, q)
    assert found == 7
