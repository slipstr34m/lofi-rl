"""SFZ regions (evaluated at the controller defaults) to SoundFont 2.04 zones (docs/CONVERTER.md).

Opcodes to generators:
- lokey/hikey, lovel/hivel -> keyRange, velRange; pitch_keycenter -> overridingRootKey;
- tune + 100 x transpose -> coarseTune + fineTune (|fine| <= 99); pitch_keytrack -> scaleTuning;
- volume, amplitude and CC crossfades -> initialAttenuation, shifted bank-wide so the loudest note is at 0 cB
  (SF2 attenuation cannot be negative, and SFZ volume can be), and written as cB / 0.4: FluidSynth multiplies
  every initialAttenuation generator by 0.4, as EMU hardware did (EMU_ATTENUATION_FACTOR; measured 2026-10-31),
  while modulator outputs are applied as they are;
- pan -> pan (x 5); a stereo sample becomes two linked zones at -500 and +500;
- ampeg_delay/attack/hold/decay/release -> *VolEnv in timecents (1200 log2 s), inside SoundFont 2.04's ranges
  (-12000..5000 for delay and hold, -12000..8000 for the others); ampeg_sustain -> sustainVolEnv in centibels,
  divided by 0.96 because FluidSynth maps 1000 cB of sustain onto its 96 dB envelope range;
- key-number terms (ARIA CC 133) on hold and decay -> keynumToVolEnvHold/Decay, fitted per zone;
- velocity terms (ARIA CC 131, ampeg_vel2*) on envelope times -> velocity modulators, fitted per zone. A fitted
  base stays inside its generator's range (the slopes are refitted when it binds), and each fit's error is that of
  the values written;
- loop_mode and loop points -> sampleModes and the sample's loop; offset/end -> address offsets.

Velocity: SFZ's amp_veltrack/amp_velcurve response is fitted per zone, over that zone's velocities, by four
instrument-zone modulators to initialAttenuation. The first has the source, destination and amount source of
FluidSynth's default velocity modulator, so it replaces it. The source curves are FluidSynth 2.6.1's, as measured
on its renders on 2026-10-31:
- negative linear: (127 - v)/128;
- negative concave: the 128-point table c[127 - v], where c[127 - i] = -(400/960) log10(i/127);
- a modulator with an amount source multiplies the two.
A zone at 100 % veltrack with no curve gets no modulators, since FluidSynth's default is that curve exactly.

The fits are minimax (Lawson's iteration) in pure Python floats. The same inputs give the same amounts in any
process, with nothing depending on BLAS or dict order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

from lofi_rl.convert import sf2
from lofi_rl.convert.sf2 import GEN, Mod, NO_SOURCE, VELOCITY, src_oper
from lofi_rl.convert.sfz import Param, Resolved, velocity_gain

# ----------------------------------------------------------------------------- FluidSynth source curves
_CONCAVE = [0.0] * 128
_CONCAVE[127] = 1.0
for _i in range(1, 127):
    _CONCAVE[127 - _i] = -(400.0 / 960.0) * math.log10(_i / 127.0)

LIN_NEG = src_oper(VELOCITY, negative=True)
CONCAVE_NEG = src_oper(VELOCITY, negative=True, curve="concave")   # FluidSynth's default modulator's source
# (source, amount source) of the velocity bases, in the order of velocity_basis()
VEL_BASES: tuple[tuple[int, int], ...] = ((CONCAVE_NEG, NO_SOURCE), (LIN_NEG, NO_SOURCE), (LIN_NEG, LIN_NEG),
                                          (CONCAVE_NEG, LIN_NEG))


def velocity_basis(v: int) -> tuple[float, float, float, float]:
    """FluidSynth's values of the four velocity bases at velocity v (1..127): concave, linear, linear x linear,
    concave x linear, all negative unipolar."""
    c, lin = _CONCAVE[127 - v], (127 - v) / 128.0
    return c, lin, lin * lin, c * lin


# --------------------------------------------------------------------------------------- the fitting
def _wls(rows: Sequence[Sequence[float]], y: Sequence[float], w: Sequence[float]) -> list[float]:
    """Weighted least squares by modified Gram-Schmidt on sqrt(w)-scaled rows, in plain floats."""
    n = len(rows[0])
    s = [math.sqrt(x) for x in w]
    cols = [[s[i] * rows[i][j] for i in range(len(rows))] for j in range(n)]
    b = [s[i] * y[i] for i in range(len(rows))]
    q, r = [], [[0.0] * n for _ in range(n)]
    for j in range(n):
        vcol = cols[j][:]
        for k in range(len(q)):
            r[k][j] = sum(a * c for a, c in zip(q[k], vcol))
            vcol = [a - r[k][j] * c for a, c in zip(vcol, q[k])]
        norm = math.sqrt(sum(a * a for a in vcol))
        r[j][j] = norm
        q.append([a / norm for a in vcol] if norm > 1e-300 else [0.0] * len(vcol))
    qb = [sum(a * c for a, c in zip(q[k], b)) for k in range(n)]
    x = [0.0] * n
    for j in range(n - 1, -1, -1):
        acc = qb[j] - sum(r[j][k] * x[k] for k in range(j + 1, n))
        x[j] = acc / r[j][j] if abs(r[j][j]) > 1e-300 else 0.0
    return x


def lawson(rows: Sequence[Sequence[float]], y: Sequence[float], error: Callable[[list[float]], list[float]],
           iterations: int = 120) -> list[float]:
    """Minimax-style fit: weighted least squares reweighted by each point's error (Lawson 1961). `error` maps a
    solution to the per-point errors in the metric being minimized. Returns the best solution seen."""
    m = len(rows)
    w = [1.0 / m] * m
    best, best_err = None, math.inf
    for _ in range(iterations):
        x = _wls(rows, y, w)
        err = error(x)
        top = max(err)
        if top < best_err:
            best, best_err = x, top
        total = sum(wi * ei for wi, ei in zip(w, err))
        if total <= 0.0:
            break
        w = [max(wi * ei / total, 1e-15) for wi, ei in zip(w, err)]
    return best


def _round(x: float) -> int:
    return int(math.floor(x + 0.5))


@dataclass(frozen=True)
class VelocityFit:
    constant: int                      # cB added to the zone's attenuation
    amounts: tuple[int, int, int, int]  # per VEL_BASES
    max_error_db: float                # over the zone's velocities, before FluidSynth's whole-cB rounding

    def modulators(self) -> tuple[Mod, ...]:
        if self.amounts == (960, 0, 0, 0):
            return ()                              # FluidSynth's default already is this curve
        mods = [Mod(CONCAVE_NEG, GEN["initialAttenuation"], self.amounts[0])]      # replaces the default
        for (src, amt_src), a in zip(VEL_BASES[1:], self.amounts[1:]):
            if a:
                mods.append(Mod(src, GEN["initialAttenuation"], a, amt_src))
        return tuple(mods)

    def model(self, v: int) -> float:
        return self.constant + sum(a * b for a, b in zip(self.amounts, velocity_basis(v)))


def fit_velocity(veltrack: float, velcurve: tuple[tuple[int, float], ...], lovel: int, hivel: int) -> VelocityFit:
    """Fit the SFZ velocity response (attenuation, cB, relative to its value at 127) over velocities
    max(1, lovel)..hivel. The constant absorbs the zone's offset; the bank-wide shift comes later."""
    vels = list(range(max(1, lovel), max(1, hivel) + 1))
    target = [-200.0 * math.log10(max(velocity_gain(veltrack, velcurve, v), 1e-12)) for v in vels]
    if veltrack == 100.0 and not velcurve:
        return VelocityFit(0, (960, 0, 0, 0), 0.0)
    use = 4 if len(vels) >= 8 else (2 if len(vels) >= 3 else 0)
    rows = [(1.0,) + velocity_basis(v)[:use] for v in vels]

    def err(x):
        return [abs(sum(a * b for a, b in zip(x, row)) - t) for row, t in zip(rows, target)]
    x = lawson(rows, target, err) if use else [sum(target) / len(target)]
    amounts = [_round(a) for a in x[1:]] + [0] * (4 - use)
    resid = [t - sum(a * b for a, b in zip(amounts, velocity_basis(v))) for v, t in zip(vels, target)]
    constant = _round((max(resid) + min(resid)) / 2.0)
    worst = max(abs(r - constant) for r in resid)
    return VelocityFit(constant, tuple(amounts), worst / 10.0)


@dataclass(frozen=True)
class TimeFit:
    """An envelope time as SF2 generators: tc at the zone (base), plus per-key or per-velocity terms. The base is
    a generator value, so it lies inside its stage's SoundFont 2.04 range; max_error is that of these values as
    written, against the SFZ's time at every key or velocity of the zone."""

    base: int
    keynum: int = 0                      # keynumToVolEnv* (tc per key below 60)
    vel_mods: tuple[tuple[int, int, int], ...] = ()   # (source, amount source, amount)
    max_error: float = 0.0               # seconds (velocity fits) or relative (key fits)


# SoundFont 2.04 section 8.1.3: attackVolEnv, decayVolEnv and releaseVolEnv run -12000..8000 tc, delayVolEnv and
# holdVolEnv -12000..5000 (18.0 s at most).
ATTACK_RANGE = sf2.GEN_RANGE[GEN["attackVolEnv"]]


def seconds_to_tc(seconds: float, hi: int = 8000) -> int | None:
    """Timecents for a time in seconds, at most `hi` (the stage's ceiling); None for 0 (the generator stays at its
    default, -12000)."""
    if seconds <= 0.0:
        return None
    return max(-12000, min(hi, _round(1200.0 * math.log2(seconds))))


def _tc_seconds(tc: float, lo: float = -12000.0, hi: float = 8000.0) -> float:
    """The seconds FluidSynth plays for a stage's summed tc (generator plus modulators), clamped to its range. The
    clamp of the sum, not of the generator alone, was measured for the attack (2026-10-31): a generator of -12039
    tc with modulators plays one 64-sample block shorter than -12000."""
    return 2.0 ** (max(lo, min(hi, tc)) / 1200.0)


def fit_time_key(p: Param, lokey: int, hikey: int, curves, tc_range: tuple[int, int] = ATTACK_RANGE) -> TimeFit:
    """A key-dependent time (hold or decay): SF2 gives tc(k) = base + keynum x (60 - k). Fit over the zone's
    keys, minimax in timecents (relative time error). The base is the generator, so when the free fit puts it
    outside `tc_range` it is fixed at the nearer bound and keynum alone is refitted."""
    lo, hi = tc_range
    keys = list(range(lokey, hikey + 1))
    secs = [max(p.at(0, k, curves), 0.001) for k in keys]
    tcs = [1200.0 * math.log2(s) for s in secs]
    rows = [(1.0, 60.0 - k) for k in keys]

    def worst(base: int, keynum: int) -> float:
        return max(abs(_tc_seconds(base + keynum * r[1], lo, hi) / s - 1.0) for r, s in zip(rows, secs))
    if len(keys) == 1:
        base = max(lo, min(hi, _round(tcs[0])))
        return TimeFit(base, 0, (), worst(base, 0))

    def err(x):
        return [abs(x[0] + x[1] * r[1] - t) for r, t in zip(rows, tcs)]
    x = lawson(rows, tcs, err)
    keynum = max(-1200, min(1200, _round(x[1])))
    resid = [t - keynum * r[1] for r, t in zip(rows, tcs)]
    base = _round((max(resid) + min(resid)) / 2.0)
    if not lo <= base <= hi:
        base = max(lo, min(hi, base))
        y = [t - base for t in tcs]
        z = lawson([(r[1],) for r in rows], y, lambda z: [abs(z[0] * r[1] - t) for r, t in zip(rows, y)])
        keynum = max(-1200, min(1200, _round(z[0])))
    return TimeFit(base, keynum, (), worst(base, keynum))


_TIME_BASES = ((LIN_NEG, NO_SOURCE), (LIN_NEG, LIN_NEG))


def fit_time_velocity(p: Param, lovel: int, hivel: int, key: int, curves,
                      tc_range: tuple[int, int] = ATTACK_RANGE) -> TimeFit:
    """A velocity-dependent time (ARIA CC 131): tc(v) = base + a1 lin(v) + a2 lin(v)², minimax in seconds over
    the zone's velocities, with FluidSynth's clamp of the summed tc to `tc_range` in the model. The base is the
    generator, so when the free fit puts it outside the range (an SFZ time that reaches 0 inside the zone does, as
    MTG's p layer at v = 100) it is fixed at the nearer bound and only the two amounts are refitted."""
    lo, hi = tc_range
    vels = list(range(max(1, lovel), max(1, hivel) + 1))
    secs = [max(p.at(v, key, curves), 0.0) for v in vels]
    rows = []
    for v in vels:
        lin = (127 - v) / 128.0
        rows.append((1.0, lin, lin * lin))

    def err(x):
        return [abs(_tc_seconds(sum(a * b for a, b in zip(x, r)), lo, hi) - s) for r, s in zip(rows, secs)]
    if max(secs) - min(secs) < 1e-4:
        tc = seconds_to_tc(secs[0], hi)
        base = -12000 if tc is None else tc
        return TimeFit(base, 0, (), max(err([base, 0, 0])))
    tcs = [1200.0 * math.log2(max(s, 0.001)) for s in secs]
    x = lawson(rows, tcs, err)
    base = _round(x[0])
    if not lo <= base <= hi:
        base = max(lo, min(hi, base))
        x = [base] + lawson([r[1:] for r in rows], [t - base for t in tcs], lambda z: err([base, *z]))
    a = [base] + [max(-32768, min(32767, _round(c))) for c in x[1:]]
    mods = tuple((src, amt, amount) for (src, amt), amount in zip(_TIME_BASES, a[1:]) if amount)
    return TimeFit(base, 0, mods, max(err(a)))


# --------------------------------------------------------------------------------------- zones
_STAGES = (("delay", "delayVolEnv", None), ("attack", "attackVolEnv", None),
           ("hold", "holdVolEnv", "keynumToVolEnvHold"), ("decay", "decayVolEnv", "keynumToVolEnvDecay"),
           ("release", "releaseVolEnv", None))


@dataclass
class ZoneSpec:
    """One region's zone(s) before the bank-wide attenuation shift."""

    region: Resolved
    gens: dict[int, int | tuple[int, int]]
    mods: list[Mod]
    velocity: VelocityFit
    level_cb: float                     # attenuation in cB (-10 x static dB + the fit constant), before the shift
    notes: dict[str, str] = field(default_factory=dict)       # approximations, opcode -> what was done
    time_errors: dict[str, float] = field(default_factory=dict)


def _tune_gens(cents: float) -> dict[int, int]:
    total = _round(cents)
    coarse = int(total / 100)                  # toward zero, so |fine| <= 99
    fine = total - 100 * coarse
    out = {}
    if coarse:
        out[GEN["coarseTune"]] = coarse
    if fine:
        out[GEN["fineTune"]] = fine
    return out


def _split_offset(offset: int, fine_gen: str, coarse_gen: str) -> dict[int, int]:
    """An address offset in points as fine (< 32768 in magnitude) + coarse x 32768, same sign."""
    if offset == 0:
        return {}
    sign = -1 if offset < 0 else 1
    coarse, fine = divmod(abs(offset), 32768)
    out = {}
    if fine:
        out[GEN[fine_gen]] = sign * fine
    if coarse:
        out[GEN[coarse_gen]] = sign * coarse
    return out


def zone_spec(res: Resolved, *, frames: int, stored: int | None = None, loop: tuple[int, int] | None,
              header_loop: tuple[int, int] | None, root: int, curves) -> ZoneSpec:
    """The generators and modulators of one kept region (the sample-independent part of its zones). `frames` is
    the source's length and `stored` the SF2 sample's (longer by any loop guard points), which end= counts from."""
    stored = frames if stored is None else stored
    g: dict[int, int | tuple[int, int]] = {GEN["keyRange"]: (res.lokey, res.hikey),
                                          GEN["velRange"]: (res.lovel, res.hivel)}
    notes: dict[str, str] = {}
    errors: dict[str, float] = {}
    mods: list[Mod] = []
    g[GEN["overridingRootKey"]] = root
    g.update(_tune_gens(res.tune))
    if res.keytrack != 100.0:
        g[GEN["scaleTuning"]] = max(0, min(1200, _round(res.keytrack)))
    if res.offset:
        if not 0 < res.offset < frames - 8:
            raise sf2.SF2Error(f"{res.where}: offset {res.offset} is outside the sample's {frames} frames")
        g.update(_split_offset(res.offset, "startAddrsOffset", "startAddrsCoarseOffset"))
    if res.end is not None and res.end + 1 < frames:
        g.update(_split_offset(res.end + 1 - stored, "endAddrsOffset", "endAddrsCoarseOffset"))
    # sample mode: SFZ plays loops when the sample has them unless loop_mode says otherwise
    mode = res.loop_mode or ("loop_continuous" if loop is not None else "no_loop")
    if mode in ("loop_continuous", "loop_sustain"):
        if loop is None:
            raise sf2.SF2Error(f"{res.where}: loop_mode={mode} but the sample has no loop points")
        g[GEN["sampleModes"]] = 1 if mode == "loop_continuous" else 3
        if header_loop is not None and loop != header_loop:
            g.update(_split_offset(loop[0] - header_loop[0], "startloopAddrsOffset", "startloopAddrsCoarseOffset"))
            g.update(_split_offset(loop[1] - header_loop[1], "endloopAddrsOffset", "endloopAddrsCoarseOffset"))
    elif mode == "one_shot":
        notes["loop_mode=one_shot"] = "played unlooped; SF2 releases at note-off where one_shot would not"
    mid_key = (res.lokey + res.hikey) // 2
    for stage, gen, key_gen in _STAGES:
        p = res.env[stage]
        tc_range = sf2.GEN_RANGE[GEN[gen]]
        if p.vel and p.key:
            raise sf2.SF2Error(f"{res.where}: ampeg_{stage} has both velocity and key terms; not mapped")
        if p.key and key_gen is not None:
            fit = fit_time_key(p, res.lokey, res.hikey, curves, tc_range)
            g[GEN[gen]] = fit.base
            if fit.keynum:
                g[GEN[key_gen]] = fit.keynum
            errors[f"ampeg_{stage} (per key, relative)"] = fit.max_error
        elif p.vel:
            fit = fit_time_velocity(p, res.lovel, res.hivel, mid_key, curves, tc_range)
            if fit.base != -12000 or fit.vel_mods:
                g[GEN[gen]] = fit.base
            mods += [Mod(src, GEN[gen], amount, amt) for src, amt, amount in fit.vel_mods]
            errors[f"ampeg_{stage} (per velocity, s)"] = fit.max_error
        else:
            if p.key:
                notes[f"ampeg_{stage} key terms"] = f"evaluated at the zone's middle key {mid_key}"
            seconds = p.at(0, mid_key, curves)
            tc = seconds_to_tc(seconds, tc_range[1])
            if tc is not None and tc != -12000:
                g[GEN[gen]] = tc
            if tc == tc_range[1] and _round(1200.0 * math.log2(seconds)) > tc_range[1]:
                notes[f"ampeg_{stage}={seconds:g} s"] = (f"longer than SF2's {tc_range[1]} tc ceiling "
                                                         f"({_tc_seconds(tc_range[1]):.1f} s); written at the ceiling")
    sus = res.env["sustain"]
    if not sus.static:
        notes["ampeg_sustain velocity/key terms"] = f"evaluated at velocity 100 and key {mid_key}"
    level = max(0.0, sus.at(100, mid_key, curves))
    sus_cb = 1000 if level <= 0.0 else min(1000, _round(-200.0 * math.log10(min(level, 100.0) / 100.0) / 0.96))
    if sus_cb:
        g[GEN["sustainVolEnv"]] = sus_cb
    vfit = fit_velocity(res.veltrack, res.velcurve, res.lovel, res.hivel)
    mods += list(vfit.modulators())
    level_cb = -10.0 * res.volume_db + vfit.constant
    return ZoneSpec(res, g, mods, vfit, level_cb, notes, errors)


EMU_ATTENUATION_FACTOR = 0.4      # FluidSynth scales initialAttenuation generators by this (not modulators)


def bank_shift(specs: Sequence[ZoneSpec]) -> int:
    """The cB added to every zone's attenuation: the smallest whole shift that keeps every generator at or above
    0 and every note's total attenuation (generator plus velocity modulators, at every velocity the zone plays)
    at or above 0, where FluidSynth would clamp it. The loudest note of the bank then sits at (or just above)
    0 cB."""
    lowest, lowest_gen = math.inf, math.inf
    for s in specs:
        lowest_gen = min(lowest_gen, s.level_cb)
        for v in range(max(1, s.region.lovel), max(1, s.region.hivel) + 1):
            lowest = min(lowest, s.level_cb - s.velocity.constant + s.velocity.model(v))
    return int(math.ceil(max(-lowest, -lowest_gen) - 1e-9))


def attenuation_generator(cb: float) -> int:
    """The initialAttenuation generator that FluidSynth plays as `cb` centibels (it applies x 0.4)."""
    gen = _round(cb / EMU_ATTENUATION_FACTOR)
    if not 0 <= gen <= 1440:
        raise sf2.SF2Error(f"attenuation {cb:.1f} cB needs generator {gen}, outside 0..1440 (576 cB at most)")
    return gen
