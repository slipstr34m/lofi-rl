"""Briefs: a scene plus two knobs, and the tempo and tonic the compiler plays them in (docs/TASTE.md, Briefs).

The model sees only the scene and the knobs; tempo, tonic and mode are applied by the compiler
(instrument.md section 1: "Never written by the model"). Mood picks the mode.
"""
from __future__ import annotations

from dataclasses import dataclass

from lofi_rl.dsl.vocab import MOODS
from lofi_rl.prng import MASK, below, splitmix64

SCENE = "city at night"                     # the only scene (TASTE.md, Briefs)
ENERGIES: tuple[str, ...] = ("low", "mid", "high")
MOOD_NAMES: tuple[str, ...] = ("melancholy", "wistful", "hopeful")
BPM_RANGE = (60, 95)
BPM_CENTER = (75, 90)
CENTER_WEIGHT = 4       # each tempo in 75-90 is 4x as likely as one outside it: P(75-90) = 64/84
PITCH_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

# Field ids for the sampler's keys: one draw per (brief index, field).
_ENERGY, _MOOD, _BPM, _TONIC = 1, 2, 3, 4


@dataclass(frozen=True)
class Brief:
    energy: str                 # low | mid | high
    mood: str                   # melancholy | wistful | hopeful
    bpm: int                    # 60..95
    tonic: int                  # pitch class of the key centre, C = 0
    scene: str = SCENE

    def __post_init__(self) -> None:
        if self.energy not in ENERGIES:
            raise ValueError(f"energy {self.energy!r} is not one of {ENERGIES}")
        if self.mood not in MOOD_NAMES:
            raise ValueError(f"mood {self.mood!r} is not one of {MOOD_NAMES}")
        if isinstance(self.bpm, bool) or not isinstance(self.bpm, int) \
                or not BPM_RANGE[0] <= self.bpm <= BPM_RANGE[1]:
            raise ValueError(f"bpm {self.bpm!r} is not an integer in {BPM_RANGE[0]}-{BPM_RANGE[1]}")
        if isinstance(self.tonic, bool) or not isinstance(self.tonic, int) or not 0 <= self.tonic <= 11:
            raise ValueError(f"tonic {self.tonic!r} is not a pitch class 0-11")
        if self.scene != SCENE:
            raise ValueError(f"scene {self.scene!r}: the only scene is {SCENE!r}")

    @property
    def mode(self) -> str:
        """aeolian, dorian or ionian (melancholy, wistful, hopeful)."""
        return MOODS[self.mood]

    @property
    def key_name(self) -> str:
        """For logs and manifests only, e.g. 'D dorian'. The model never sees it."""
        return f"{PITCH_NAMES[self.tonic]} {self.mode}"


def _tempi() -> tuple[tuple[int, int], ...]:
    lo, hi = BPM_RANGE
    return tuple((bpm, CENTER_WEIGHT if BPM_CENTER[0] <= bpm <= BPM_CENTER[1] else 1)
                 for bpm in range(lo, hi + 1))


TEMPO_WEIGHTS = _tempi()


def _draw(seed: int, index: int, field: int) -> int:
    """The 64-bit hash behind one field of brief `index` of stream `seed`, by chained SplitMix64.

    h = splitmix64(splitmix64(splitmix64(seed mod 2**64) ^ index) ^ field): hash the seed, then mix in the
    index, then the field, each followed by a full SplitMix64 round, so streams from different seeds share no
    structure. The old key, seed ^ (index << 8) ^ field, made seeds a multiple of 256 apart replay each other's
    briefs in another order (brief i of seed 256k was brief i ^ k of seed 0) and coupled the fields of adjacent
    seeds (the energy of seed 2026 was drawn from the mood hash of seed 2025).
    """
    if not 0 <= index <= MASK:
        raise ValueError(f"brief index {index} is outside 0..2**64 - 1")
    return splitmix64(splitmix64(splitmix64(seed & MASK) ^ index) ^ field)


def _weighted(h: int, weights: tuple[tuple[int, int], ...]) -> int:
    x = below(h, sum(w for _, w in weights))
    for value, w in weights:
        if x < w:
            return value
        x -= w
    raise AssertionError("unreachable")


def sample_brief(seed: int, index: int) -> Brief:
    """Brief number `index` of the stream `seed`: energy, mood and tonic uniform, tempo centred on 75-90.

    Each field is its own keyed draw (`_draw`), so brief i is the same however many briefs are drawn, and
    streams from different seeds are independent. The seed is read modulo 2**64; the index is 0..2**64 - 1.
    """
    return Brief(
        energy=ENERGIES[below(_draw(seed, index, _ENERGY), len(ENERGIES))],
        mood=MOOD_NAMES[below(_draw(seed, index, _MOOD), len(MOOD_NAMES))],
        bpm=_weighted(_draw(seed, index, _BPM), TEMPO_WEIGHTS),
        tonic=below(_draw(seed, index, _TONIC), 12),
    )


def sample_briefs(seed: int, count: int, start: int = 0) -> tuple[Brief, ...]:
    return tuple(sample_brief(seed, i) for i in range(start, start + count))
