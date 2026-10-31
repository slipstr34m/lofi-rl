"""A byte-deterministic SoundFont 2.04 writer, and the sample side of the converter (docs/CONVERTER.md).

The file is a function of the arguments alone. Nothing reads the clock, the environment or a dict or set order:
- generators are written by id, after keyRange and velRange and before sampleID or instrument;
- modulators are sorted by their fields;
- samples, instruments and presets keep the order the caller gives, and the converter gives sorted ones.
So the same inputs give the same bytes in any process and with any PYTHONHASHSEED.

Layout (SoundFont 2.04, sections 4-7):
- RIFF sfbk with INFO (ifil 2.04, isng, INAM, optional ICOP and ICMT, ISFT; no ICRD date);
- sdta: smpl (16-bit words) and, when any source is 24-bit, sm24 (the low bytes), each sample followed by 46 zero
  sample points. A 24-bit point is (smpl << 8) | sm24, so 24-bit sources are stored losslessly. The total point
  count is kept even, so sm24 needs no pad byte and its size is exactly half of smpl's;
- pdta: phdr, pbag, pmod, pgen, inst, ibag, imod, igen and shdr, each with its terminal record.

What SoundFont 2.04 forbids is refused rather than written (SF2Error lists what): FluidSynth 2.6.1 discards some of
it while loading and zeroes some, so such a file would not play as written.

Sources keep their own sample rate in the sample header (FluidSynth resamples); nothing here resamples. A source's
loop and unity note come from its smpl chunk: a WAV's own, or the one a FLAC carries in an APPLICATION 'riff' block.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

# SoundFont 2.04 section 8.1.2: generator ids.
GEN: dict[str, int] = {
    "startAddrsOffset": 0, "endAddrsOffset": 1, "startloopAddrsOffset": 2, "endloopAddrsOffset": 3,
    "startAddrsCoarseOffset": 4, "modLfoToPitch": 5, "vibLfoToPitch": 6, "modEnvToPitch": 7,
    "initialFilterFc": 8, "initialFilterQ": 9, "modLfoToFilterFc": 10, "modEnvToFilterFc": 11,
    "endAddrsCoarseOffset": 12, "modLfoToVolume": 13, "chorusEffectsSend": 15, "reverbEffectsSend": 16,
    "pan": 17, "delayModLFO": 21, "freqModLFO": 22, "delayVibLFO": 23, "freqVibLFO": 24,
    "delayModEnv": 25, "attackModEnv": 26, "holdModEnv": 27, "decayModEnv": 28, "sustainModEnv": 29,
    "releaseModEnv": 30, "keynumToModEnvHold": 31, "keynumToModEnvDecay": 32, "delayVolEnv": 33,
    "attackVolEnv": 34, "holdVolEnv": 35, "decayVolEnv": 36, "sustainVolEnv": 37, "releaseVolEnv": 38,
    "keynumToVolEnvHold": 39, "keynumToVolEnvDecay": 40, "instrument": 41, "keyRange": 43, "velRange": 44,
    "startloopAddrsCoarseOffset": 45, "keynum": 46, "velocity": 47, "initialAttenuation": 48,
    "endloopAddrsCoarseOffset": 50, "coarseTune": 51, "fineTune": 52, "sampleID": 53, "sampleModes": 54,
    "scaleTuning": 56, "exclusiveClass": 57, "overridingRootKey": 58,
}
GEN_NAME: dict[int, str] = {v: k for k, v in GEN.items()}
_RANGES = (GEN["keyRange"], GEN["velRange"])
_INDEX = (GEN["instrument"], GEN["sampleID"])          # unsigned indices, always last in their zone
PAD_POINTS = 46                                        # zero sample points after every sample (section 6.1)
MONO, RIGHT, LEFT = 1, 2, 4                            # sfSampleType (section 7.10)
_CURVES = {"linear": 0, "concave": 1, "convex": 2, "switch": 3}
NO_SOURCE, VELOCITY, KEY = 0, 2, 3                     # general controller indices (section 8.2.1)

# Section 8.1.3: the range of each generator's value where it is absolute, in an instrument zone (global or local).
# The address offsets have no fixed range. A preset zone's values are offsets added to the instrument's, so only
# the 16-bit word bounds them there.
GEN_RANGE: dict[int, tuple[int, int]] = {GEN[k]: v for k, v in {
    "modLfoToPitch": (-12000, 12000), "vibLfoToPitch": (-12000, 12000), "modEnvToPitch": (-12000, 12000),
    "initialFilterFc": (1500, 13500), "initialFilterQ": (0, 960), "modLfoToFilterFc": (-12000, 12000),
    "modEnvToFilterFc": (-12000, 12000), "modLfoToVolume": (-960, 960), "chorusEffectsSend": (0, 1000),
    "reverbEffectsSend": (0, 1000), "pan": (-500, 500), "delayModLFO": (-12000, 5000), "freqModLFO": (-16000, 4500),
    "delayVibLFO": (-12000, 5000), "freqVibLFO": (-16000, 4500), "delayModEnv": (-12000, 5000),
    "attackModEnv": (-12000, 8000), "holdModEnv": (-12000, 5000), "decayModEnv": (-12000, 8000),
    "sustainModEnv": (0, 1000), "releaseModEnv": (-12000, 8000), "keynumToModEnvHold": (-1200, 1200),
    "keynumToModEnvDecay": (-1200, 1200), "delayVolEnv": (-12000, 5000), "attackVolEnv": (-12000, 8000),
    "holdVolEnv": (-12000, 5000), "decayVolEnv": (-12000, 8000), "sustainVolEnv": (0, 1440),
    "releaseVolEnv": (-12000, 8000), "keynumToVolEnvHold": (-1200, 1200), "keynumToVolEnvDecay": (-1200, 1200),
    "keynum": (0, 127), "velocity": (0, 127), "initialAttenuation": (0, 1440), "coarseTune": (-120, 120),
    "fineTune": (-99, 99), "sampleModes": (0, 3), "scaleTuning": (0, 1200), "exclusiveClass": (0, 127),
    "overridingRootKey": (0, 127)}.items()}
# Section 8.1.2: generators an instrument zone may hold but a preset zone may not (sample addressing and loops, and
# the per-voice substitutions). FluidSynth 2.6.1 discards them from a preset zone, with a warning.
INSTRUMENT_ONLY = frozenset(GEN[k] for k in (
    "startAddrsOffset", "endAddrsOffset", "startloopAddrsOffset", "endloopAddrsOffset", "startAddrsCoarseOffset",
    "endAddrsCoarseOffset", "startloopAddrsCoarseOffset", "keynum", "velocity", "endloopAddrsCoarseOffset",
    "sampleModes", "exclusiveClass", "overridingRootKey"))
# Section 8.2.1: the general controllers a modulator may read (no controller, velocity, key, poly pressure, channel
# pressure, pitch wheel, pitch wheel sensitivity; 127, a link, is not written here), and the MIDI CCs it may not
# (bank select, data entry, NRPN/RPN and the channel mode messages).
_GC_SOURCES = frozenset({NO_SOURCE, VELOCITY, KEY, 10, 13, 14, 16})
_CC_NOT_SOURCES = frozenset({0, 6, 32, 38, 98, 99, 100, 101, *range(120, 128)})
_NOT_MOD_DEST = frozenset(_RANGES + _INDEX)
MIN_POINTS = 48                     # section 7.10: 8 points before a loop, 32 in it, 8 after it
LOOP_GUARD = 8                      # points a loop needs before its start and after its end (section 7.10)


def src_oper(index: int, *, cc: bool = False, negative: bool = False, bipolar: bool = False,
             curve: str = "linear") -> int:
    """A modulator source operator (section 8.2): index, CC flag, direction, polarity and curve type."""
    if not 0 <= index < 128:
        raise ValueError(f"modulator source index {index} is outside 0..127")
    return (_CURVES[curve] << 10) | (int(bipolar) << 9) | (int(negative) << 8) | (int(cc) << 7) | index


class SF2Error(ValueError):
    """The arguments cannot be written as a valid SoundFont 2.04 file, or as one FluidSynth 2.6.1 loads as written.

    sf2_bytes refuses, besides malformed structure: a generator a preset zone may not hold (section 8.1.2), an
    instrument-zone value outside its section 8.1.3 range (sampleModes 2, "unused", included), a modulator whose
    source is not one section 8.2.1 allows or whose transform is not 0 or 2 (section 8.3), a modulator to a
    structural or unknown generator or a linked one, a sample under 48 points, and a loop without 8 points before
    it, 32 in it and 8 after it (section 7.10). One rule is FluidSynth's, not the spec's: a modulator whose primary
    source is "no controller" (the spec reads it as 1) is refused, because FluidSynth 2.6.1 warns about it; a
    constant belongs in the generator."""


@dataclass(frozen=True, order=True)
class Mod:
    """One modulator record (section 7.4/7.8): amount x f(src) x f(amt_src), added to generator `dest`."""

    src: int
    dest: int
    amount: int
    amt_src: int = NO_SOURCE
    transform: int = 0


@dataclass(frozen=True)
class Zone:
    """An instrument zone (sample set) or a preset zone (instrument set); neither for a global zone.

    gens maps generator ids to values: a (lo, hi) pair for keyRange and velRange, a signed int otherwise. The
    sampleID or instrument generator is added by the writer from `sample` or `instrument`."""

    gens: tuple[tuple[int, int | tuple[int, int]], ...] = ()
    mods: tuple[Mod, ...] = ()
    sample: str | None = None
    instrument: str | None = None


@dataclass(frozen=True)
class Instrument:
    name: str
    zones: tuple[Zone, ...]
    global_zone: Zone | None = None


@dataclass(frozen=True)
class Preset:
    name: str
    bank: int
    program: int
    zones: tuple[Zone, ...]
    global_zone: Zone | None = None


@dataclass(frozen=True, eq=False)
class Sample:
    """One mono sample: 24-bit integer points (int32 array), its rate and header fields.

    loop is [start, end) in frames from the first point (end exclusive, SoundFont's dwEndloop), or None.
    kind is "mono", "left" or "right"; a left or right sample names its partner in `link`."""

    name: str
    pcm: np.ndarray
    rate: int
    bits: int = 16
    root: int = 60
    correction: int = 0
    loop: tuple[int, int] | None = None
    kind: str = "mono"
    link: str | None = None


# ------------------------------------------------------------------------------------------------ bytes
def _chunk(cid: bytes, *parts: bytes) -> list[bytes]:
    """A RIFF chunk as a list of byte strings (joined once, at the end, so large sample data is not copied
    chunk by chunk)."""
    size = sum(len(x) for x in parts)
    return [cid + struct.pack("<I", size), *parts] + ([b"\0"] if size & 1 else [])


def _list(form: bytes, chunks: Sequence[list[bytes]]) -> list[bytes]:
    return _chunk(b"LIST", form, *(x for c in chunks for x in c))


def _zstr(text: str, limit: int) -> bytes:
    """An INFO string: ASCII, zero-terminated, padded to an even length (section 5.1)."""
    raw = _ascii(text, "INFO string")
    if len(raw) + 1 > limit:
        raise SF2Error(f"INFO string {text[:40]!r} is longer than {limit - 1} bytes")
    raw += b"\0"
    return raw + (b"\0" if len(raw) & 1 else b"")


def _ascii(text: str, what: str) -> bytes:
    try:
        return text.encode("ascii")
    except UnicodeEncodeError as err:
        raise SF2Error(f"{what} {text!r} is not ASCII") from err


def _name20(text: str, what: str) -> bytes:
    """A 20-byte name field: at most 19 ASCII characters, so it always ends in a zero byte."""
    raw = _ascii(text, what)
    if not raw or len(raw) > 19 or b"\0" in raw:
        raise SF2Error(f"{what} {text!r} must be 1 to 19 ASCII characters")
    return raw.ljust(20, b"\0")


def _gen_amount(gid: int, value) -> bytes:
    if gid in _RANGES:
        lo, hi = value
        if not (0 <= lo <= hi <= 127):
            raise SF2Error(f"{GEN_NAME[gid]} ({lo}, {hi}) is not 0 <= lo <= hi <= 127")
        return bytes((lo, hi))
    if gid in _INDEX:
        return struct.pack("<H", value)
    if not -32768 <= value <= 32767:
        raise SF2Error(f"generator {GEN_NAME.get(gid, gid)} value {value} does not fit a signed 16-bit amount")
    return struct.pack("<h", value)


def _check_source(op: int, what: str, primary: bool) -> None:
    """A modulator source operator (section 8.2.1): a general controller the spec defines or an allowed MIDI CC,
    with a curve type 0-3. A primary source of "no controller" is refused too (FluidSynth 2.6.1 warns about it)."""
    if not 0 <= op <= 0xFFFF:
        raise SF2Error(f"{what} {op} does not fit 16 bits")
    index, cc, curve = op & 0x7F, op >> 7 & 1, op >> 10
    if curve > 3:
        raise SF2Error(f"{what} {op:#06x}: curve type {curve} is not linear, concave, convex or switch (0-3)")
    if cc and index in _CC_NOT_SOURCES:
        raise SF2Error(f"{what} {op:#06x}: MIDI CC {index} cannot drive a modulator (section 8.2.1)")
    if not cc and index not in _GC_SOURCES:
        raise SF2Error(f"{what} {op:#06x}: general controller {index} is not a modulator source (0, 2, 3, 10, 13, "
                       f"14 or 16; links are not written)")
    if primary and not cc and index == NO_SOURCE:
        raise SF2Error(f"{what} {op:#06x} is 'no controller': FluidSynth 2.6.1 warns about such a modulator (the "
                       f"spec reads it as 1); put the constant in the generator")


def _zone_records(zone: Zone, index_gen: int | None, index: int | None,
                  preset: bool = False) -> tuple[list[bytes], list[bytes]]:
    """The gen and mod records of one zone, in canonical order. A preset zone (`preset`) holds offsets to the
    instrument's values, so it may not hold the instrument-only generators and its values are not range-checked;
    an instrument zone's values are absolute and must lie in their section 8.1.3 ranges."""
    gens = dict(zone.gens)
    if len(gens) != len(zone.gens):
        raise SF2Error("a zone sets the same generator twice")
    for gid, value in gens.items():
        if gid in _INDEX or gid not in GEN_NAME:
            raise SF2Error(f"generator id {gid} cannot be set directly in a zone")
        if preset and gid in INSTRUMENT_ONLY:
            raise SF2Error(f"generator {GEN_NAME[gid]} may appear only in an instrument zone (section 8.1.2)")
        if not preset and gid in GEN_RANGE:
            lo, hi = GEN_RANGE[gid]
            if not lo <= value <= hi:
                raise SF2Error(f"generator {GEN_NAME[gid]} = {value} is outside its range {lo}..{hi} (section 8.1.3)")
            if gid == GEN["sampleModes"] and value == 2:
                raise SF2Error("sampleModes 2 is unused (section 8.1.2); 0 is an unlooped sample")
    order = [g for g in _RANGES if g in gens] + sorted(g for g in gens if g not in _RANGES)
    recs = [struct.pack("<H", g) + _gen_amount(g, gens[g]) for g in order]
    if index_gen is not None:
        recs.append(struct.pack("<H", index_gen) + _gen_amount(index_gen, index))
    seen = set()
    mods = []
    for m in sorted(zone.mods):
        key = (m.src, m.dest, m.amt_src)
        if key in seen:      # section 7.8: a second identical modulator would silently replace the first
            raise SF2Error(f"two modulators share source {m.src:#06x}, destination {m.dest} and amount "
                           f"source {m.amt_src:#06x} in one zone")
        seen.add(key)
        _check_source(m.src, "modulator source", primary=True)
        _check_source(m.amt_src, "modulator amount source", primary=False)
        if m.dest & 0x8000:
            raise SF2Error(f"modulator destination {m.dest:#06x} is a link; linked modulators are not written")
        if m.dest not in GEN_NAME or m.dest in _NOT_MOD_DEST:
            raise SF2Error(f"modulator destination {m.dest} is not a generator a modulator can drive")
        if m.transform not in (0, 2):
            raise SF2Error(f"modulator transform {m.transform} is neither linear (0) nor absolute value (2) "
                           f"(section 8.3)")
        if not -32768 <= m.amount <= 32767:
            raise SF2Error(f"modulator amount {m.amount} does not fit a signed 16-bit amount")
        mods.append(struct.pack("<HHhHH", m.src, m.dest, m.amount, m.amt_src, m.transform))
    return recs, mods


def _hydra(items, index_of: dict[str, int], index_gen: int, what: str):
    """bags, gens and mods of a preset or instrument list: (bag start per item, bag, gen, mod records)."""
    bag_start, bags, gens, mods = [], [], [], []
    preset = index_gen == GEN["instrument"]
    for item in items:
        bag_start.append(len(bags))
        zones = ([(item.global_zone, None, True)] if item.global_zone is not None else []) + \
                [(z, z.sample if index_gen == GEN["sampleID"] else z.instrument, False) for z in item.zones]
        if not item.zones:
            raise SF2Error(f"{what} {item.name!r} has no zones")
        for zone, ref, is_global in zones:
            if is_global:
                if zone.sample is not None or zone.instrument is not None:
                    raise SF2Error(f"the global zone of {what} {item.name!r} names a sample or instrument")
                g, m = _zone_records(zone, None, None, preset)
            else:
                if ref is None or ref not in index_of:
                    raise SF2Error(f"a zone of {what} {item.name!r} refers to {ref!r}, which is not in the file")
                g, m = _zone_records(zone, index_gen, index_of[ref], preset)
            bags.append(struct.pack("<HH", len(gens), len(mods)))
            gens += g
            mods += m
    if len(gens) > 0xFFFF or len(mods) > 0xFFFF or len(bags) > 0xFFFF:
        raise SF2Error(f"too many {what} zones, generators or modulators for 16-bit bag indices")
    return bag_start, bags, gens, mods


def sf2_bytes(*, name: str, presets: Sequence[Preset], instruments: Sequence[Instrument],
              samples: Sequence[Sample], software: str = "lofi-rl convert", copyright: str | None = None,
              comment: str | None = None) -> bytes:
    """The complete SoundFont 2.04 file as bytes."""
    if not presets or not instruments or not samples:
        raise SF2Error("a SoundFont needs at least one preset, instrument and sample")
    for what, items in (("sample", samples), ("instrument", instruments), ("preset", presets)):
        names = [x.name for x in items]
        if len(set(names)) != len(names):
            raise SF2Error(f"{what} names are not unique")
    keys = [(p.bank, p.program) for p in presets]
    if len(set(keys)) != len(keys):
        raise SF2Error("two presets share a bank and program")

    # ---- INFO (section 5)
    info = [_chunk(b"ifil", struct.pack("<HH", 2, 4)), _chunk(b"isng", _zstr("EMU8000", 256)),
            _chunk(b"INAM", _zstr(name, 256))]
    if copyright:
        info.append(_chunk(b"ICOP", _zstr(copyright, 256)))
    if comment:
        info.append(_chunk(b"ICMT", _zstr(comment, 65536)))
    info.append(_chunk(b"ISFT", _zstr(software, 256)))

    # ---- sdta (section 6)
    sample_index = {s.name: i for i, s in enumerate(samples)}
    want24 = any(s.bits > 16 for s in samples)
    hi_parts, lo_parts, headers = [], [], []
    pos = 0
    for i, s in enumerate(samples):
        pcm = np.asarray(s.pcm)
        if pcm.ndim != 1 or pcm.dtype.kind not in "iu":
            raise SF2Error(f"sample {s.name!r}: points must be a 1-D integer array")
        n = int(pcm.shape[0])
        if n < MIN_POINTS:
            raise SF2Error(f"sample {s.name!r} has {n} points; a SoundFont sample needs at least {MIN_POINTS} "
                           f"(section 7.10)")
        p = pcm.astype(np.int32, copy=False)
        if p.min(initial=0) < -(1 << 23) or p.max(initial=0) >= (1 << 23):
            raise SF2Error(f"sample {s.name!r} has points outside the 24-bit range")
        pad = PAD_POINTS + (1 if i == len(samples) - 1 and (pos + n + PAD_POINTS) & 1 else 0)
        hi_parts += [(p >> 8).astype("<i2").tobytes(), bytes(2 * pad)]
        lo_parts += [(p & 0xFF).astype(np.uint8).tobytes(), bytes(pad)]
        start, end = pos, pos + n
        if s.loop is not None:
            ls, le = s.loop
            if not (LOOP_GUARD <= ls and le - ls >= 32 and le <= n - LOOP_GUARD):
                raise SF2Error(f"sample {s.name!r}: loop [{ls}, {le}) of its {n} points needs {LOOP_GUARD} points "
                               f"before it, 32 in it and {LOOP_GUARD} after it (section 7.10)")
            loop = (start + ls, start + le)
        else:          # never played (sampleModes 0) but kept valid: 8 points from each end, at least 32 in it
            loop = (start + LOOP_GUARD, end - LOOP_GUARD)
        if s.kind == "mono":
            stype, link = MONO, 0
            if s.link is not None:
                raise SF2Error(f"mono sample {s.name!r} names a link")
        elif s.kind in ("left", "right"):
            stype = LEFT if s.kind == "left" else RIGHT
            if s.link not in sample_index:
                raise SF2Error(f"{s.kind} sample {s.name!r} links to {s.link!r}, which is not in the file")
            partner = samples[sample_index[s.link]]
            if partner.link != s.name or {partner.kind, s.kind} != {"left", "right"}:
                raise SF2Error(f"samples {s.name!r} and {s.link!r} are not a left/right pair linked both ways")
            if partner.pcm.shape != s.pcm.shape or partner.rate != s.rate:
                raise SF2Error(f"stereo pair {s.name!r}/{s.link!r} differs in length or rate")
            link = sample_index[s.link]
        else:
            raise SF2Error(f"sample {s.name!r}: kind {s.kind!r} is not mono, left or right")
        if not (0 <= s.root <= 127 and -128 <= s.correction <= 127 and 400 <= s.rate <= 0xFFFFFFFF):
            raise SF2Error(f"sample {s.name!r}: root {s.root}, correction {s.correction} or rate {s.rate} "
                           f"is out of range")
        headers.append(_name20(s.name, "sample name") + struct.pack(
            "<IIIIIBbHH", start, end, loop[0], loop[1], s.rate, s.root, s.correction, link, stype))
        pos = end + pad
    if pos & 1:
        raise AssertionError("sample point count is odd")     # kept even above
    if pos > 0xFFFFFFFF // 2:
        raise SF2Error("too much sample data for one SoundFont")
    sdta = [_chunk(b"smpl", *hi_parts)]
    if want24:
        sdta.append(_chunk(b"sm24", *lo_parts))

    # ---- pdta (section 7)
    inst_index = {x.name: i for i, x in enumerate(instruments)}
    order = sorted(range(len(presets)), key=lambda i: (presets[i].bank, presets[i].program))
    ordered = [presets[i] for i in order]
    p_start, pbag, pgen, pmod = _hydra(ordered, inst_index, GEN["instrument"], "preset")
    i_start, ibag, igen, imod = _hydra(instruments, sample_index, GEN["sampleID"], "instrument")
    phdr = [_name20(p.name, "preset name") + struct.pack("<HHHIII", p.program, p.bank, p_start[k], 0, 0, 0)
            for k, p in enumerate(ordered)]
    phdr.append(b"EOP".ljust(20, b"\0") + struct.pack("<HHHIII", 0, 0, len(pbag), 0, 0, 0))
    pbag.append(struct.pack("<HH", len(pgen), len(pmod)))
    inst = [_name20(x.name, "instrument name") + struct.pack("<H", i_start[k]) for k, x in enumerate(instruments)]
    inst.append(b"EOI".ljust(20, b"\0") + struct.pack("<H", len(ibag)))
    ibag.append(struct.pack("<HH", len(igen), len(imod)))
    headers.append(b"EOS".ljust(20, b"\0") + bytes(26))
    pdta = [_chunk(b"phdr", *phdr), _chunk(b"pbag", *pbag), _chunk(b"pmod", *pmod, bytes(10)),
            _chunk(b"pgen", *pgen, bytes(4)), _chunk(b"inst", *inst), _chunk(b"ibag", *ibag),
            _chunk(b"imod", *imod, bytes(10)), _chunk(b"igen", *igen, bytes(4)), _chunk(b"shdr", *headers)]
    body = [b"sfbk", *_list(b"INFO", info), *_list(b"sdta", sdta), *_list(b"pdta", pdta)]
    size = sum(len(x) for x in body)
    if size > 0xFFFFFFFF:
        raise SF2Error("the SoundFont would exceed the 4 GiB RIFF limit")
    return b"".join([b"RIFF", struct.pack("<I", size), *body])


# ------------------------------------------------------------------------------------------- sources
@dataclass(frozen=True, eq=False)
class Source:
    """A decoded sample file: one int32 array of 24-bit points per channel, its rate and bit depth, and the
    loops of its smpl chunk ([start, end) frames, end exclusive) if it has one: a WAV's own chunk, or the one a
    FLAC carries from its WAV in an APPLICATION 'riff' block."""

    path: Path
    channels: tuple[np.ndarray, ...]
    rate: int
    bits: int
    smpl_loops: tuple[tuple[int, int], ...] = ()
    smpl_root: int | None = None          # the smpl chunk's MIDI unity note, for pitch_keycenter=sample

    @property
    def frames(self) -> int:
        return int(self.channels[0].shape[0])


_SUBTYPE_BITS = {"PCM_16": 16, "PCM_24": 24, "PCM_S8": 8, "PCM_U8": 8}


def read_source(path: Path) -> Source:
    """Decode a WAV or FLAC file losslessly into 24-bit integer points (libsndfile scales integer reads to
    int32, so a 16-bit point arrives as x << 16 and a 24-bit one as x << 8; both shift back exactly).
    32-bit and float sources are refused rather than quantized."""
    path = Path(path)
    info = sf.info(str(path))
    bits = _SUBTYPE_BITS.get(info.subtype)
    if bits is None:
        raise SF2Error(f"{path.name}: {info.subtype} samples cannot be stored losslessly in SoundFont 2.04 "
                       f"(16 or 24 bits)")
    data, rate = sf.read(str(path), dtype="int32", always_2d=True)
    if data.shape[0] < 8:
        raise SF2Error(f"{path.name} has {data.shape[0]} frames")
    pts = np.right_shift(data, 8)                                          # 24-bit points
    chans = tuple(np.ascontiguousarray(pts[:, c]) for c in range(pts.shape[1]))
    suffix = path.suffix.lower()
    root, loops = wav_smpl(path) if suffix == ".wav" else flac_smpl(path) if suffix == ".flac" else (None, ())
    return Source(path, chans, int(rate), bits, loops, root)


def _smpl(body: bytes) -> tuple[int | None, list[tuple[int, int]]]:
    """A smpl chunk's body: its MIDI unity note (None above 127) and its forward loops as [start, end) frames. The
    chunk stores the last point of a loop (inclusive), so end = dwEnd + 1. Ping-pong and reverse loops are not
    SoundFont loops and are skipped."""
    unity = struct.unpack_from("<I", body, 12)[0]
    count = struct.unpack_from("<I", body, 28)[0]
    out = []
    for i in range(count):
        if 36 + 24 * (i + 1) > len(body):
            break
        _cue, ltype, start, end, _frac, _plays = struct.unpack_from("<6I", body, 36 + 24 * i)
        if ltype == 0:
            out.append((start, end + 1))
    return (unity if unity <= 127 else None), out


def wav_smpl(path: Path) -> tuple[int | None, tuple[tuple[int, int], ...]]:
    """A WAV file's smpl chunk: its MIDI unity note and its forward loops as [start, end) frames (see _smpl).
    (None, ()) without a smpl chunk."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None, ()
    pos, root, out = 12, None, []
    while pos + 8 <= len(data):
        cid, size = struct.unpack_from("<4sI", data, pos)
        if cid == b"smpl" and size >= 36:
            root, loops = _smpl(data[pos + 8:pos + 8 + size])
            out += loops
        pos += 8 + size + (size & 1)
    return root, tuple(out)


def flac_smpl(path: Path) -> tuple[int | None, tuple[tuple[int, int], ...]]:
    """The smpl chunk a FLAC file carries from the WAV it was encoded from, as wav_smpl reads it. `flac
    --keep-foreign-metadata` stores each of the WAV's chunks but the points (the RIFF header, fmt, the data chunk's
    8-byte header, smpl, ...) in an APPLICATION metadata block (type 2) of its own, whose body is the application
    id 'riff' and then the chunk as the WAV held it. libsndfile does not read them. (None, ()) without one."""
    root, out = None, []
    with Path(path).open("rb") as f:
        if f.read(4) != b"fLaC":
            return None, ()
        while True:
            head = f.read(4)
            if len(head) < 4:
                raise SF2Error(f"{Path(path).name}: FLAC metadata is cut short")
            size = int.from_bytes(head[1:], "big")
            if head[0] & 0x7F == 2:
                body = f.read(size)
                if body[:8] == b"riffsmpl" and len(body) >= 12:
                    csize = struct.unpack_from("<I", body, 8)[0]
                    chunk = body[12:12 + csize]
                    if csize >= 36 and len(chunk) == csize:
                        root, loops = _smpl(chunk)
                        out += loops
            else:
                f.seek(size, 1)
            if head[0] & 0x80:                 # the last metadata block; audio frames follow
                return root, tuple(out)
