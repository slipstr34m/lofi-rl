"""SFZ parsing, ARIA evaluation at the controller defaults, and deterministic region selection (docs/CONVERTER.md).

Parsing (parse_sfz):
- headers <control>, <global>, <master>, <group>, <region> and <curve>; a region inherits global, then master,
  then group opcodes, and a later level overrides an earlier one. A new <global> clears master and group; a new
  <master> clears group;
- #define $VAR value, with $VAR substituted in every later line (opcode names, values, #include paths);
- #include "file", relative to the including file; default_path (a <control> opcode, current at each region);
- // and /* */ comments; values run to the next opcode or header, so sample paths may hold spaces; either slash;
- note names (c4 = 60, with octave_offset and note_offset from <control>); key= sets lokey, hikey and
  pitch_keycenter.

Evaluation (evaluate): every ARIA modulation is read at the controller defaults that set_cc / set_hdcc give
(0 otherwise): *_onccN and *ccN depths through their *_curveccN curves (ARIA's predefined curves 0-6 or the
file's <curve>s), locc/hicc conditions, CC crossfades. Extended CC 131 (note-on velocity) and 133 (key number)
are per-note sources, kept as velocity and key terms for the SF2 mapping to express; the random sources read 0.

Selection (select): round robin keeps seq_position 1, lorand/hirand keeps the region that covers random value 0,
keyswitches keep the default (sw_default), and regions are dropped when they are release, legato or CC
triggered, generated (sample=*noise and the like), silent at the defaults, or fail a CC condition.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

HEADERS = ("control", "global", "master", "group", "region", "curve", "effect", "midi", "sample")
_HEADER_OR_OPCODE = re.compile(r"<\s*([A-Za-z_]+)\s*>|(?:(?<=[\s>])|^)([A-Za-z0-9_]+)=")
_DOLLAR = re.compile(r"\$([A-Za-z0-9_]+)")
_DEFINE = re.compile(r"^#define\s+\$([A-Za-z0-9_]+)\s+(.*?)\s*$")
_INCLUDE = re.compile(r'^#include\s+"([^"]+)"\s*$')
_NOTE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
_NOTE_BASE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
VELOCITY_CC, KEY_CC = 131, 133          # ARIA extended CCs: note-on velocity and key number (per-note sources)
_ZERO_CCS = frozenset({128, 129, 130, 132, 134, 135, 136, 137, 140, 141})   # bend, pressures, randoms...: read 0


class SfzError(ValueError):
    """The SFZ cannot be parsed or evaluated as the converter reads it."""


@dataclass(frozen=True)
class Header:
    kind: str
    opcodes: tuple[tuple[str, str], ...]
    where: str                                    # file:line of the header


@dataclass(frozen=True)
class Region:
    """One <region> with its inherited opcodes (global < master < group < region) and the <control> state
    (default_path, set_cc, octave and note offsets) current when it was read."""

    index: int                                    # order in the file, from 0
    opcodes: Mapping[str, str] = field(hash=False)
    control: Mapping[str, str] = field(hash=False)
    where: str = ""


@dataclass(frozen=True)
class SfzFile:
    path: Path
    headers: tuple[Header, ...]
    regions: tuple[Region, ...]
    control: Mapping[str, str] = field(hash=False)          # every <control> opcode, later ones winning
    curves: Mapping[int, tuple[tuple[int, float], ...]] = field(hash=False)
    files: tuple[Path, ...] = ()                              # the SFZ file and every file it included


# ------------------------------------------------------------------------------------------------ parsing
def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _strip_comments(text: str) -> str:
    """Remove // line comments and /* */ block comments, keeping every newline (so line numbers hold)."""
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            block = text[i:n if j < 0 else j + 2]
            out.append("\n" * block.count("\n"))
            i = n if j < 0 else j + 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _substitute(line: str, defines: Mapping[str, str]) -> str:
    """Replace $NAME by its #define value. The longest defined name that starts the token wins."""
    def repl(m: re.Match) -> str:
        token = m.group(1)
        for end in range(len(token), 0, -1):
            if token[:end] in defines:
                return defines[token[:end]] + token[end:]
        return m.group(0)
    return _DOLLAR.sub(repl, line)


def _lines(path: Path, defines: dict[str, str], files: list[Path], root: Path, depth: int = 0):
    """(text, file, line number) of every non-directive line, with #include expanded in place. An #include
    path is looked up beside the including file first, then beside the root SFZ file (the folder ARIA and sfizz
    resolve it against: MTG's Data/group_ten.txt includes "Data/ten_p_rr1.txt")."""
    if depth > 32:
        raise SfzError(f"{path}: #include nested more than 32 deep")
    files.append(path)
    for no, line in enumerate(_strip_comments(_read_text(path)).splitlines(), 1):
        text = line.strip()
        if text.startswith("#define"):
            m = _DEFINE.match(text)
            if not m:
                raise SfzError(f"{path.name}:{no}: malformed #define {text!r}")
            defines[m.group(1)] = _substitute(m.group(2), defines)
            continue
        if text.startswith("#include"):
            m = _INCLUDE.match(_substitute(text, defines))
            if not m:
                raise SfzError(f"{path.name}:{no}: malformed #include {text!r}")
            target = _resolve(path.parent, m.group(1)) or _resolve(root.parent, m.group(1))
            if target is None:
                raise SfzError(f"{path.name}:{no}: #include {m.group(1)!r} not found")
            yield from _lines(target, defines, files, root, depth + 1)
            continue
        if text.startswith("#"):
            raise SfzError(f"{path.name}:{no}: unknown directive {text.split()[0]!r}")
        if text:
            yield _substitute(text, defines), path, no


def parse_sfz(path: Path | str) -> SfzFile:
    """Parse an SFZ file and its includes into headers and inheriting regions (no evaluation yet)."""
    path = Path(path)
    _DIR_CACHE.clear()          # folder listings are cached for one parse and the selection that follows it
    defines: dict[str, str] = {}
    files: list[Path] = []
    headers: list[tuple[str, list[tuple[str, str]], str]] = []
    for text, src, no in _lines(path, defines, files, path):
        marks = list(_HEADER_OR_OPCODE.finditer(text))
        if not marks:
            raise SfzError(f"{src.name}:{no}: no header or opcode in {text[:60]!r}")
        if marks[0].start() > 0 and text[:marks[0].start()].strip():
            raise SfzError(f"{src.name}:{no}: text before the first opcode: {text[:marks[0].start()]!r}")
        for k, m in enumerate(marks):
            end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
            if m.group(1) is not None:
                kind = m.group(1).lower()
                if kind not in HEADERS:
                    raise SfzError(f"{src.name}:{no}: unknown header <{kind}>")
                if text[m.end():end].strip():
                    raise SfzError(f"{src.name}:{no}: text after <{kind}>: {text[m.end():end]!r}")
                headers.append((kind, [], f"{src.name}:{no}"))
            else:
                if not headers:
                    headers.append(("global", [], f"{src.name}:{no}"))   # opcodes before any header
                name, value = m.group(2), text[m.end():end].strip()
                if name == "key":                                         # key= is lokey, hikey and keycenter
                    headers[-1][1].extend([("lokey", value), ("hikey", value), ("pitch_keycenter", value)])
                else:
                    headers[-1][1].append((name, value))
    return _assemble(path, [Header(k, tuple(ops), w) for k, ops, w in headers], files)


def _assemble(path: Path, headers: list[Header], files: list[Path]) -> SfzFile:
    control: dict[str, str] = {}
    levels: dict[str, dict[str, str]] = {"global": {}, "master": {}, "group": {}}
    regions: list[Region] = []
    curves: dict[int, tuple[tuple[int, float], ...]] = {}
    for h in headers:
        ops = dict(h.opcodes)
        if h.kind == "control":
            control.update(ops)
        elif h.kind == "global":
            levels = {"global": ops, "master": {}, "group": {}}
        elif h.kind == "master":
            levels["master"], levels["group"] = ops, {}
        elif h.kind == "group":
            levels["group"] = ops
        elif h.kind == "region":
            merged = {**levels["global"], **levels["master"], **levels["group"], **ops}
            regions.append(Region(len(regions), MappingProxyType(merged), MappingProxyType(dict(control)), h.where))
        elif h.kind == "curve":
            index = _int(ops.get("curve_index", ""), f"{h.where} curve_index")
            pts = []
            for k, v in ops.items():
                m = re.fullmatch(r"v(\d{1,3})", k)
                if m:
                    pts.append((int(m.group(1)), _float(v, f"{h.where} {k}")))
                elif k != "curve_index":
                    raise SfzError(f"{h.where}: unknown <curve> opcode {k}")
            curves[index] = tuple(sorted(pts))
        # <effect>, <midi> and <sample> carry nothing a region plays
    return SfzFile(path, tuple(headers), tuple(regions), MappingProxyType(control), MappingProxyType(curves),
                   tuple(files))


_DIR_CACHE: dict[Path, tuple[str, ...]] = {}


def _listdir(folder: Path) -> tuple[str, ...]:
    if folder not in _DIR_CACHE:
        try:
            _DIR_CACHE[folder] = tuple(sorted(p.name for p in folder.iterdir()))
        except OSError:
            _DIR_CACHE[folder] = ()
    return _DIR_CACHE[folder]


def _resolve(base: Path, rel: str) -> Path | None:
    """`rel` (either slash) under `base`, matched component by component to the names on disk: the exact name
    if present, else the one name equal ignoring case. So a path resolves to the same file and the same spelling
    on a case-insensitive laptop and a case-sensitive cluster. None when a component is missing."""
    cur = base
    for part in rel.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            cur = cur.parent
            continue
        names = _listdir(cur)
        if part in names:
            cur = cur / part
            continue
        hits = [n for n in names if n.lower() == part.lower()]
        if len(hits) != 1:
            return None
        cur = cur / hits[0]
    return cur if cur.exists() else None


# --------------------------------------------------------------------------------------------- numbers
def _float(text: str, where: str) -> float:
    try:
        value = float(text)
    except ValueError as err:
        raise SfzError(f"{where}: {text!r} is not a number") from err
    if not math.isfinite(value):
        raise SfzError(f"{where}: {text!r} is not finite")
    return value


def _int(text: str, where: str) -> int:
    value = _float(text, where)
    if value != int(value):
        raise SfzError(f"{where}: {text!r} is not a whole number")
    return int(value)


def note_number(text: str, where: str = "", octave_offset: int = 0, note_offset: int = 0) -> int:
    """A key as a MIDI number: digits, or a note name where c4 = 60 (c#4 = db4 = 61), then the offsets."""
    m = _NOTE.match(text.strip())
    if m:
        value = 12 * (int(m.group(3)) + 1) + _NOTE_BASE[m.group(1).lower()] + {"#": 1, "b": -1, "": 0}[m.group(2)]
    else:
        value = _int(text, where)
    return value + 12 * octave_offset + note_offset


# ------------------------------------------------------------------------------------- ARIA evaluation
_PREDEFINED = {0: lambda x: x, 1: lambda x: 2 * x - 1, 2: lambda x: 1 - x, 3: lambda x: 1 - 2 * x,
               4: lambda x: x * x, 5: lambda x: math.sqrt(x), 6: lambda x: math.sqrt(1 - x)}


def curve_value(index: int, x: float, curves: Mapping[int, tuple[tuple[int, float], ...]]) -> float:
    """ARIA curve `index` at x in [0, 1]: a file <curve> (points at 0..127, linear between, 0 and 1 at the ends
    unless given), else predefined curves 0 (linear), 1 (bipolar), 2 (inverted), 3 (inverted bipolar), 4 (x²),
    5 (√x), 6 (√(1−x))."""
    x = min(1.0, max(0.0, x))
    if index in curves:
        pts = dict(curves[index])
        pts.setdefault(0, 0.0)
        pts.setdefault(127, 1.0)
        keys = sorted(pts)
        pos = 127.0 * x
        for a, b in zip(keys, keys[1:]):
            if a <= pos <= b:
                return pts[a] if b == a else pts[a] + (pts[b] - pts[a]) * (pos - a) / (b - a)
        return pts[keys[-1]]
    if index in _PREDEFINED:
        return _PREDEFINED[index](x)
    raise SfzError(f"curve {index} is not defined")


def cc_defaults(control: Mapping[str, str]) -> dict[int, float]:
    """Controller defaults from set_ccN (0..127) and set_hdccN (0..1), normalized to [0, 1]."""
    out: dict[int, float] = {}
    for k, v in control.items():
        m = re.fullmatch(r"set_(hd|real)?cc(\d+)", k)
        if m:
            value = _float(v, k)
            out[int(m.group(2))] = value if m.group(1) else value / 127.0
    return out


@dataclass(frozen=True)
class Param:
    """A parameter evaluated at the controller defaults: base + Σ depth·curve(v/127) over its velocity terms
    + Σ depth·curve(k/127) over its key terms (ARIA extended CCs 131 and 133, or v1 *_vel2* opcodes)."""

    base: float
    vel: tuple[tuple[float, int], ...] = ()
    key: tuple[tuple[float, int], ...] = ()

    def at(self, velocity: int, key: int, curves=MappingProxyType({})) -> float:
        return (self.base + sum(d * curve_value(c, velocity / 127, curves) for d, c in self.vel)
                + sum(d * curve_value(c, key / 127, curves) for d, c in self.key))

    @property
    def static(self) -> bool:
        return not self.vel and not self.key


class Evaluator:
    """Reads one region's opcodes at the controller defaults, recording which opcodes it consumed."""

    def __init__(self, region: Region, curves: Mapping[int, tuple[tuple[int, float], ...]]):
        self.ops = region.opcodes
        self.where = region.where
        self.curves = curves
        self.cc = cc_defaults(region.control)
        self.used: set[str] = set()
        self.octave_offset = _int(region.control.get("octave_offset", "0"), "octave_offset")
        self.note_offset = _int(region.control.get("note_offset", "0"), "note_offset")

    def has(self, name: str) -> bool:
        return name in self.ops

    def text(self, name: str, default: str | None = None) -> str | None:
        if name in self.ops:
            self.used.add(name)
            return self.ops[name]
        return default

    def num(self, name: str, default: float) -> float:
        t = self.text(name)
        return default if t is None else _float(t, f"{self.where} {name}")

    def key(self, name: str, default: int) -> int:
        t = self.text(name)
        return default if t is None else note_number(t, f"{self.where} {name}", self.octave_offset, self.note_offset)

    def ccval(self, n: int) -> float:
        if n in _ZERO_CCS:
            return 0.0
        return self.cc.get(n, 0.0)

    def mod_terms(self, name: str, aliases: tuple[str, ...] = ()) -> list[tuple[int, float, int]]:
        """(cc, depth, curve) of every CC modulation of `name`: name_onccN, nameccN (SFZ v1), aliases."""
        out = []
        for base in (name,) + aliases:
            pat = re.compile(rf"{re.escape(base)}(?:_on)?cc(\d+)")
            for k in sorted(self.ops):
                m = pat.fullmatch(k)
                if m:
                    n = int(m.group(1))
                    depth = _float(self.ops[k], f"{self.where} {k}")
                    self.used.add(k)
                    ck = f"{base}_curvecc{n}"
                    curve = 0
                    if ck in self.ops:
                        curve = _int(self.ops[ck], f"{self.where} {ck}")
                        self.used.add(ck)
                    for extra in (f"{base}_smoothcc{n}", f"{base}_stepcc{n}"):
                        if extra in self.ops:
                            self.used.add(extra)     # smoothing and stepping only shape live changes
                    out.append((n, depth, curve))
        return out

    def param(self, name: str, default: float, *, aliases: tuple[str, ...] = (), vel2: str | None = None) -> Param:
        """An additive parameter: base value plus every CC term read at the defaults; velocity and key terms
        (CC 131 / 133, and the v1 `vel2` opcode, a linear velocity term) are kept for the mapping."""
        base = self.num(name, default)
        vel, key = [], []
        for n, depth, curve in self.mod_terms(name, aliases):
            if n == VELOCITY_CC:
                vel.append((depth, curve))
            elif n == KEY_CC:
                key.append((depth, curve))
            else:
                base += depth * curve_value(curve, self.ccval(n), self.curves)
        if vel2 is not None and self.has(vel2):
            vel.append((self.num(vel2, 0.0), 0))
        return Param(base, tuple(vel), tuple(key))

    def amplitude(self) -> Param:
        """amplitude (percent) times each amplitude_onccN factor (depth% x curve), as ARIA multiplies them. A
        velocity or key factor is not expressible here and is reported by the caller."""
        value = self.num("amplitude", 100.0) / 100.0
        vel = []
        for n, depth, curve in self.mod_terms("amplitude"):
            if n in (VELOCITY_CC, KEY_CC):
                vel.append((depth, curve))
            else:
                value *= depth / 100.0 * curve_value(curve, self.ccval(n), self.curves)
        return Param(value, tuple(vel))

    def crossfade_gain(self) -> float:
        """The product of every CC crossfade (xfin_/xfout_loccN/hiccN) at the defaults, power curve by default
        (xf_cccurve=power: √ of the linear fade) or gain."""
        shape = self.text("xf_cccurve", "power")
        gain = 1.0
        ccs = sorted({int(m.group(2)) for k in self.ops
                      for m in [re.fullmatch(r"xf(in|out)_(?:lo|hi)cc(\d+)", k)] if m})
        for n in ccs:
            x = 127.0 * self.ccval(n)
            lo_in, hi_in = self.num(f"xfin_locc{n}", 0.0), self.num(f"xfin_hicc{n}", 0.0)
            lo_out, hi_out = self.num(f"xfout_locc{n}", 127.0), self.num(f"xfout_hicc{n}", 127.0)
            if x >= hi_in:
                fin = 1.0
            elif x <= lo_in:
                fin = 0.0
            else:
                fin = (x - lo_in) / (hi_in - lo_in)
            if x <= lo_out:
                fout = 1.0
            elif x >= hi_out:
                fout = 0.0
            else:
                fout = (hi_out - x) / (hi_out - lo_out)
            for f in (fin, fout):
                gain *= math.sqrt(f) if shape == "power" else f
        return gain

    def cc_conditions(self) -> list[str]:
        """The locc/hicc conditions that fail at the defaults (empty: the region is enabled)."""
        failed = []
        for n in sorted({int(m.group(2)) for k in self.ops for m in [re.fullmatch(r"(lo|hi)cc(\d+)", k)] if m}):
            lo = self.num(f"locc{n}", 0.0)
            hi = self.num(f"hicc{n}", 127.0)
            x = 127.0 * self.ccval(n)
            if not lo <= x <= hi:
                failed.append(f"CC{n} = {x:g} is outside locc/hicc {lo:g}..{hi:g}")
        return failed

    def lfo_depth(self, name: str) -> float:
        """An LFO depth opcode (lfoNN_pitch, pitchlfo_depth, ...) with its CC terms, at the defaults. Read for the
        report only: the opcodes stay unconsumed, so they are listed as dropped."""
        used = set(self.used)
        try:
            return self.param(name, 0.0).base
        finally:
            self.used = used


# --------------------------------------------------------------------------------- resolved regions
@dataclass(frozen=True)
class Resolved:
    """A kept region evaluated at the controller defaults, in SFZ units, with the opcodes left unexpressed."""

    index: int
    where: str
    sample: Path
    lokey: int
    hikey: int
    lovel: int
    hivel: int
    keycenter: int | None              # None: pitch_keycenter=sample (the file's own root)
    tune: float                        # cents, transpose included
    keytrack: float                    # cents per key (pitch_keytrack)
    volume_db: float                   # volume + amplitude + crossfades, static part
    pan: float                         # -100..100
    width: float                       # stereo width, percent
    veltrack: float                    # amp_veltrack, percent
    velcurve: tuple[tuple[int, float], ...]
    env: Mapping[str, Param] = field(hash=False)     # ampeg_delay attack hold decay sustain release
    loop_mode: str | None = None       # no_loop, one_shot, loop_continuous, loop_sustain or None (sample decides)
    loop: tuple[int, int] | None = None              # [start, end) frames from loop_start/loop_end
    offset: int = 0
    end: int | None = None             # last frame to play (inclusive), or None
    unexpressed: Mapping[str, str] = field(default=MappingProxyType({}), hash=False)  # opcode -> why


@dataclass(frozen=True)
class Dropped:
    index: int
    where: str
    reason: str
    sample: str


_GENERATED = re.compile(r"^\*")
# LFO and envelope depth opcodes: dropped, but their value at the defaults says whether that changes anything.
_LFO_DEPTH = re.compile(r"(lfo\d+_(pitch|volume|amplitude|pan|width|cutoff\d?|resonance\d?|eq\d(gain|freq|bw))|"
                        r"(pitch|amp|fil)lfo_depth|pitcheg_depth|fileg_depth|lfo\d+_freq_lfo\d+|eq\d+_gain)"
                        r"((?:_on)?cc\d+)?")
_META = re.compile(r"(region|group|master|global)_label|label_cc\d+|label_key\d+|set_(hd|real)?cc\d+|"
                   r"default_path|octave_offset|note_offset|curve_index|sw_label|sw_default|hint_.*|image")
# Why an opcode the SF2 cannot express is dropped. Families match by prefix; the first match wins.
_WHY = (
    (re.compile(r"(lfo\d+|amplfo|pitchlfo|fillfo)_"), "LFO (ARIA LFOs do not map onto SF2's two fixed ones)"),
    (re.compile(r"(eq\d+|var\d+)_"), "EQ or ARIA variable modulator (no SF2 equivalent)"),
    (re.compile(r"(cutoff|resonance|fil\d*_|fileg_|fil_)"), "filter (ARIA filters are not mapped onto SF2's "
     "low-pass)"),
    (re.compile(r"bend_(up|down|step)"), "bend range (the compiler sets it by RPN 0)"),
    (re.compile(r"pitcheg_|pitch_veltrack|pitch_random"), "pitch envelope, velocity or random pitch (not mapped)"),
    (re.compile(r"(group|off_by|off_mode|off_time|off_shape|polyphony|note_polyphony|note_selfmask)$"),
     "voice group or polyphony limit (not mapped: FluidSynth releases a voice when its key is struck again, and "
     "the compiler's sax line is monophonic)"),
    (re.compile(r"(sustain_sw|sostenuto_sw|sustain_cc|sustain_lo|sostenuto_cc)"),
     "pedal switches: FluidSynth always honours CC64; the compiler sends none"),
    (re.compile(r"ampeg_dynamic|ampeg_.*_curvecc|ampeg_.*smooth"), "live envelope updates: SF2 envelopes are "
     "fixed at note-on"),
    (re.compile(r"(xfin|xfout)_(lo|hi)(key|vel)|xf_(key|vel)curve"),
     "key or velocity crossfades: SF2 zones switch, they cannot fade"),
    (re.compile(r"(amp_random|amp_keytrack|amp_keycenter|delay|delay_random|delay_cc\d+|offset_random|rt_decay|"
                r"rt_dead|trigger|width|position|phase|direction|count|sync_.*|seq_length)"),
     "no SF2 equivalent at the defaults"),
)


def why_dropped(opcode: str) -> str:
    for pat, why in _WHY:
        if pat.match(opcode):
            return why
    return "no SF2 equivalent"


def select(sfz: SfzFile, *, keep=None, need_samples: bool = True) -> tuple[list[Resolved], list[Dropped]]:
    """Evaluate every region at the controller defaults and keep the ones a note plays there.

    `keep(region, evaluator)` is an optional bank rule returning a drop reason (str) or None. With
    need_samples=False a kept region's missing sample is not an error (its path is kept as written): for
    auditing programs whose samples were not fetched."""
    kept, dropped = [], []
    sw_default = sfz.control.get("sw_default")
    for region in sfz.regions:
        ev = Evaluator(region, sfz.curves)
        sample_text = region.opcodes.get("sample", "")
        reason = _drop_reason(ev, sw_default)
        if reason is None and keep is not None:
            reason = keep(region, ev)
        if reason is None:
            res, reason = _resolve_region(sfz, region, ev, need_samples)
            if res is not None:
                kept.append(res)
                continue
        dropped.append(Dropped(region.index, region.where, reason, sample_text))
    return kept, dropped


def _drop_reason(ev: Evaluator, sw_default: str | None) -> str | None:
    sample = ev.ops.get("sample")
    if sample is None or not sample.strip():
        return "no sample"
    if _GENERATED.match(sample.strip()):
        return f"generated sample {sample.strip()} (noise/oscillator, not a recording)"
    trigger = ev.ops.get("trigger", "attack")
    if trigger in ("release", "release_key"):
        return "release trigger (SF2 has none)"
    if trigger == "legato":
        return "legato trigger (plays only under a held note)"
    if trigger not in ("attack", "first"):
        return f"trigger={trigger}"
    if any(re.fullmatch(r"on_(lo|hi)cc\d+", k) for k in ev.ops):
        return "CC-triggered (on_locc/on_hicc), never by a note"
    seq = _int(ev.ops.get("seq_position", "1"), f"{ev.where} seq_position")
    if seq != 1:
        return f"round robin seq_position {seq} (only 1 is kept)"
    lo = _float(ev.ops.get("lorand", "0"), f"{ev.where} lorand")
    hi = _float(ev.ops.get("hirand", "1"), f"{ev.where} hirand")
    if not lo <= 0.0 < hi:
        return f"random layer lorand={lo:g} hirand={hi:g} does not cover 0"
    if "sw_last" in ev.ops:
        want = None if sw_default is None else note_number(sw_default, "sw_default", ev.octave_offset,
                                                          ev.note_offset)
        if want is None or note_number(ev.ops["sw_last"], "sw_last", ev.octave_offset, ev.note_offset) != want:
            return f"keyswitch sw_last={ev.ops['sw_last']} is not the default"
    for k in ("sw_down", "sw_up", "sw_previous"):
        if k in ev.ops:
            return f"keyswitch {k} (needs a held key)"
    failed = ev.cc_conditions()
    if failed:
        return "; ".join(failed)
    lokey = note_number(ev.ops.get("lokey", "0"), "lokey", ev.octave_offset, ev.note_offset)
    hikey = note_number(ev.ops.get("hikey", "127"), "hikey", ev.octave_offset, ev.note_offset)
    lovel = _int(ev.ops.get("lovel", "0"), "lovel")
    hivel = _int(ev.ops.get("hivel", "127"), "hivel")
    if lokey > hikey or lovel > hivel or hikey < 0 or lokey > 127 or hivel < 1:
        return f"empty key or velocity range (keys {lokey}..{hikey}, velocities {lovel}..{hivel})"
    return None


_ENV = ("delay", "attack", "hold", "decay", "sustain", "release")
_ENV_DEFAULT = {"delay": 0.0, "attack": 0.0, "hold": 0.0, "decay": 0.0, "sustain": 100.0, "release": 0.001}


def _resolve_region(sfz: SfzFile, region: Region, ev: Evaluator, need_samples: bool = True
                    ) -> tuple[Resolved | None, str]:
    amplitude = ev.amplitude()
    xf = ev.crossfade_gain()
    gain = amplitude.base * xf
    if gain <= 0.0:          # checked before the sample is looked up: silent layers may not have been fetched
        return None, "silent at the controller defaults (amplitude or crossfade gain 0)"
    default_path = region.control.get("default_path", "")
    rel = (default_path + ev.text("sample")).replace("\\", "/")
    path = _resolve(sfz.path.parent, rel)
    if path is None:
        if need_samples:
            raise SfzError(f"{region.where}: sample {rel!r} not found under {sfz.path.parent}")
        path = sfz.path.parent / rel

    for k in ("trigger", "seq_position", "seq_length", "lorand", "hirand", "sw_last", "sw_lokey", "sw_hikey",
              "sw_default"):
        ev.text(k)
    for k in list(ev.ops):
        if re.fullmatch(r"(lo|hi)cc\d+", k):
            ev.used.add(k)
    lokey = max(0, ev.key("lokey", 0))
    hikey = min(127, ev.key("hikey", 127))
    lovel = max(0, int(ev.num("lovel", 0)))
    hivel = min(127, int(ev.num("hivel", 127)))
    kc_text = ev.text("pitch_keycenter")
    keycenter = None if kc_text == "sample" else (
        60 if kc_text is None else note_number(kc_text, f"{region.where} pitch_keycenter", ev.octave_offset,
                                               ev.note_offset))
    tune = ev.param("tune", 0.0)
    transpose = ev.num("transpose", 0.0)
    keytrack = ev.param("pitch_keytrack", 100.0)
    volume = ev.param("volume", 0.0, aliases=("gain",))
    pan = ev.param("pan", 0.0)
    width = ev.param("width", 100.0)
    veltrack = ev.param("amp_veltrack", 100.0)
    unexpressed: dict[str, str] = {}
    for name, p in (("tune", tune), ("pitch_keytrack", keytrack), ("volume", volume), ("pan", pan),
                    ("width", width), ("amp_veltrack", veltrack)):
        if not p.static:
            unexpressed[f"{name} (velocity/key terms)"] = "per-note modulation of this parameter is not mapped"
    if not amplitude.static:
        unexpressed["amplitude (velocity/key terms)"] = "per-note modulation of amplitude is not mapped"
    velcurve = []
    for k in sorted(ev.ops):
        m = re.fullmatch(r"amp_velcurve_(\d+)", k)
        if m:
            velcurve.append((int(m.group(1)), ev.num(k, 0.0)))
    env = {}
    for stage in _ENV:
        env[stage] = ev.param(f"ampeg_{stage}", _ENV_DEFAULT[stage], vel2=f"ampeg_vel2{stage}")
    loop_mode = ev.text("loop_mode") or ev.text("loopmode")
    if loop_mode not in (None, "no_loop", "one_shot", "loop_continuous", "loop_sustain"):
        raise SfzError(f"{region.where}: loop_mode={loop_mode!r}")
    ls = ev.text("loop_start") or ev.text("loopstart")
    le = ev.text("loop_end") or ev.text("loopend")
    loop = None
    if ls is not None or le is not None:
        if ls is None or le is None:
            raise SfzError(f"{region.where}: loop_start and loop_end must come together")
        loop = (_int(ls, "loop_start"), _int(le, "loop_end") + 1)     # SFZ loop_end is the last loop frame
    offset = ev.param("offset", 0.0)
    if not offset.static:
        unexpressed["offset (velocity/key terms)"] = "per-note sample offset is not mapped"
    end_text = ev.text("end")
    end = None if end_text is None else _int(end_text, f"{region.where} end")
    if end is not None and end < 0:
        return None, "end=-1: the region never plays"
    for k in ev.ops:
        if k not in ev.used and _META.fullmatch(k):
            ev.used.add(k)
    depths: dict[str, float] = {}                    # a depth and its CC terms, read at the defaults
    for k in sorted(ev.ops):
        m = _LFO_DEPTH.fullmatch(k)
        if k not in ev.used and m:
            target = m.group(1)
            if target not in depths:
                depths[target] = ev.lfo_depth(target)
            unexpressed[k] = f"{why_dropped(k)}; {target} at the controller defaults = {depths[target]:g}"
        elif k not in ev.used:
            unexpressed[k] = why_dropped(k)
    if width.base != 100.0:
        unexpressed.setdefault("width", "stereo width other than 100 % is not mapped")
    res = Resolved(
        index=region.index, where=region.where, sample=path, lokey=lokey, hikey=hikey, lovel=lovel, hivel=hivel,
        keycenter=keycenter, tune=tune.base + 100.0 * transpose, keytrack=keytrack.base,
        volume_db=volume.base + 20.0 * math.log10(gain), pan=max(-100.0, min(100.0, pan.base)),
        width=width.base, veltrack=max(-100.0, min(100.0, veltrack.base)), velcurve=tuple(velcurve),
        env=MappingProxyType(env), loop_mode=loop_mode, loop=loop, offset=int(round(offset.base)), end=end,
        unexpressed=MappingProxyType(unexpressed))
    return res, ""


def velocity_gain(veltrack: float, velcurve: tuple[tuple[int, float], ...], v: int) -> float:
    """The SFZ amplitude factor at velocity v: the curve c(v) is (v/127)² or the amp_velcurve_N points (linear
    between them, 0 at 0 and 1 at 127 unless given), and amp_veltrack t blends it in amplitude: 1 − |t|(1 − c) for
    t ≥ 0 and |t|(1 − c) for t < 0. This is sfizz's reading as the converter's author knows it (not re-checked
    against its source); Weresax setting both amp_veltrack=77 and amp_velcurve_63=1 on one layer only makes sense
    if the two combine."""
    if velcurve:
        pts = dict(velcurve)
        pts.setdefault(0, 0.0)
        pts.setdefault(127, 1.0)
        keys = sorted(pts)
        c = pts[keys[-1]]
        for a, b in zip(keys, keys[1:]):
            if a <= v <= b:
                c = pts[a] if a == b else pts[a] + (pts[b] - pts[a]) * (v - a) / (b - a)
                break
    else:
        c = (v / 127.0) ** 2
    t = abs(veltrack) / 100.0
    return 1.0 - t * (1.0 - c) if veltrack >= 0 else t * (1.0 - c)
