"""Humanize: hash-keyed timing and velocity nudges, for shootouts only (instrument.md section 3, Humanize).

Off by default (palette groove.humanize). Modes:
- bar (H1): key = seed ^ lane ^ step-in-bar, so every bar gets the same nudges;
- loop (H2): the key also takes the bar (mod 8), so bars differ but every loop repeats exactly.
The key XORs each field into its own byte of the 64-bit seed (draw bits 24-31, lane 16-23, bar 8-15,
step 0-7), so no two fields collide, and h = splitmix64(key). u = (h >> 11) 2^-53; timing = (2u - 1) *
humanize_ms and velocity x (1 + (2u - 1) * humanize_vel), both uniform. seed = the first 8 bytes of
SHA256(the canonical text without its keys:, drums: and sax: lines), so the groove depends only on the vamp,
the grids and the phrases, never on which keys instrument, voicing style, kit or sax a loop names (SR-7).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction

from lofi_rl.dsl.ast import Loop
from lofi_rl.dsl.canon import canon
from lofi_rl.palette import TIMING_LANES, Groove
from lofi_rl.prng import MASK, splitmix64, uniform

from .timeline import BAR, STEPS, ms

TIMING, VELOCITY = 1, 2
LANE_ID = {lane: i + 1 for i, lane in enumerate(TIMING_LANES)}     # kick 1 .. sax 9
NAME_LINES = ("keys:", "drums:", "sax:")    # canonical lines that only name sounds; the seed leaves them out


def seed_text(loop: Loop) -> str:
    """The canonical text without its keys:, drums: and sax: lines (the keys instrument and style, the kit, the
    sax): the vamp, the grids, the edits and the phrases, in canonical order."""
    return "".join(line for line in canon(loop).splitlines(keepends=True) if not line.startswith(NAME_LINES))


def loop_seed(loop: Loop) -> int:
    """seed = SHA256(seed_text)[:8], big-endian: the same vamp, grids and phrases always draw the same nudges,
    whatever keys instrument, voicing style, kit or sax the loop names."""
    return int.from_bytes(hashlib.sha256(seed_text(loop).encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class Humanize:
    mode: str                   # off | bar | loop
    seed: int
    timing_ms: Fraction
    velocity: Fraction
    lanes: frozenset[str]

    @classmethod
    def from_groove(cls, groove: Groove, seed: int) -> Humanize:
        return cls(groove.humanize, seed, groove.humanize_ms, groove.humanize_vel, groove.humanize_lanes)

    def active(self, lane: str) -> bool:
        return self.mode != "off" and lane in self.lanes

    def _u(self, draw: int, lane: str, k: int) -> Fraction:
        r = k % STEPS
        bar = r // BAR if self.mode == "loop" else 0
        key = (self.seed ^ (draw << 24) ^ (LANE_ID[lane] << 16) ^ (bar << 8) ^ (r % BAR)) & MASK
        return uniform(splitmix64(key))

    def nudge(self, lane: str, k: int) -> Fraction:
        """Timing offset of step k in samples, uniform in +-timing_ms (0 when off for this lane)."""
        if not self.active(lane):
            return Fraction(0)
        return (2 * self._u(TIMING, lane, k) - 1) * ms(self.timing_ms)

    def nudges(self, lane: str) -> tuple[Fraction, ...]:
        """One nudge per step 0..127, or () when off for this lane."""
        return tuple(self.nudge(lane, k) for k in range(STEPS)) if self.active(lane) else ()

    def gain(self, lane: str, k: int) -> Fraction:
        """Velocity factor of step k, uniform in 1 +- velocity (1 when off for this lane)."""
        if not self.active(lane):
            return Fraction(1)
        return 1 + (2 * self._u(VELOCITY, lane, k) - 1) * self.velocity
