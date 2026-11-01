"""Palette loader and validator (docs/design/instrument.md section 6).

A palette names every sound the renderer may use and every audible value the compiler applies. Loading
is strict: every key is required and unknown keys are refused, so the file's sha256 pins everything a
render depends on (the manifest records it). Licences (docs/DECISIONS.md D-012): every source has a sha256
and a licence; mode `commit` takes only CC0, public domain, permissive and CC BY licences, so a
non-commercial, restricted or unrecognized one must be mode `fetch`; a CC-BY source names its attribution.
`load_palette(..., verify=True)` also checks each source file against its sha256, every SoundFont
preset the palette names against the file's preset headers, and that each SoundFont voice has a zone for every
note the compiler may send it (`sf2_coverage`, docs/SHOOTOUTS.md §7.5). A mode-commit path is repo-relative; a
mode-fetch path lies under `assets_root` (`Palette.assets_dir`: `$LOFI_ASSETS`, or <root>/assets/big while it is
unset).

Numbers that feed sample positions (swing, offsets, lags) are kept as exact `Fraction`s of the decimal
written in the file, so the timeline never depends on binary floating point.
"""
from __future__ import annotations

import hashlib
import os
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from lofi_rl.dsl.vocab import LANES, NUMERALS, QUALITIES, VOICINGS, Vocab

ROOT = Path(__file__).resolve().parents[2]
SR = 44100
LEVELS: tuple[str, ...] = ("X", "x", "g")
LAYERS: tuple[str, ...] = ("soft", "med", "hard")          # soft to hard: "nearest" walks this order
LAYER_OF_LEVEL: Mapping[str, str] = MappingProxyType({"X": "hard", "x": "med", "g": "soft"})
TIMING_LANES: tuple[str, ...] = LANES + ("keys", "bass", "sax")
STEMS: tuple[str, ...] = ("keys", "bass", "sax")
HUMANIZE_MODES: tuple[str, ...] = ("off", "bar", "loop")
SAX_TIMINGS: tuple[str, ...] = ("grid", "lag", "swung")
BUSES: tuple[str, ...] = ("sax", "sax_warm", "keys", "bass", "drums")
BUS_GAINS: tuple[str, ...] = ("sax", "drums", "bass", "keys")
PLUGINS: Mapping[str, frozenset[str]] = MappingProxyType({    # pedalboard 0.9.26 parameter names
    "HighpassFilter": frozenset({"cutoff_frequency_hz"}),
    "LowpassFilter": frozenset({"cutoff_frequency_hz"}),
    "PeakFilter": frozenset({"cutoff_frequency_hz", "gain_db", "q"}),
    "HighShelfFilter": frozenset({"cutoff_frequency_hz", "gain_db", "q"}),
    "LowShelfFilter": frozenset({"cutoff_frequency_hz", "gain_db", "q"}),
    "Compressor": frozenset({"threshold_db", "ratio", "attack_ms", "release_ms"}),
    "Reverb": frozenset({"room_size", "damping", "wet_level", "dry_level", "width", "freeze_mode"}),
    "Gain": frozenset({"gain_db"}),
})
# Licences (docs/DECISIONS.md D-012). Mode commit is an allow-list: CC0, public domain, permissive and CC BY.
# Anything else, including an unrecognized licence, must be mode fetch, so a misspelling fails closed.
_NC = re.compile(r"\bNC\b|non[-\s_]*commercial", re.I)
_NO_COMMIT = re.compile(r"all\s+rights\s+reserved|proprietary|(?:personal|private)\s+use|no\s+redistribution|"
                        r"not\s+(?:to\s+)?(?:be\s+)?(?:for\s+)?redistribut|non[-\s_]*redistributable|"
                        r"\b(?:ND|SA)\b|no[-\s_]*deriv|share[-\s_]*alike", re.I)
_COMMIT_OK = re.compile(r"\bCC[-\s_]?0\b|creative\s+commons\s+zero|public[-\s_]+domain|\bunlicense\b|"
                        r"\bCC[-\s_]*BY\b|creative\s+commons\s+attribution|^\s*attribution\s+\d|"
                        r"\bMIT\b|\bBSD\b|\bApache\b|\bISC\b|\bzlib\b|\bpermissive\b", re.I)
_CC_BY = re.compile(r"\bCC[-\s_]*BY\b|creative\s+commons\s+attribution|^\s*attribution\b", re.I)
_ID = re.compile(r"[a-z][a-z0-9_]*")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_VAR = re.compile(r"\$(?:\{(\w+)\}|(\w+))")
# Mode-fetch sources resolve under assets_root, `$LOFI_ASSETS` in every palette. Unset or empty, it is the
# laptop's fetch folder <root>/assets/big (docs/SHOOTOUT_ASSETS.md, Route). Cluster jobs must export it as
# $REMOTE_KEEP/assets (CLAUDE.md section 1); docs/RENDER.md sections 10 and 11 say what is still missing there.
ASSETS_VAR = "LOFI_ASSETS"
ASSETS_DEFAULT = Path("assets") / "big"


class PaletteError(ValueError):
    pass


# ------------------------------------------------------------------------------------------- schema
@dataclass(frozen=True)
class Source:
    id: str
    path: str                   # repo-relative for mode commit, relative to assets_root for mode fetch
    sha256: str
    license: str
    mode: str                   # commit | fetch
    url: str | None
    converted_from: str | None
    attribution: str | None


@dataclass(frozen=True)
class SaxVoice:
    name: str
    src: str
    bank: int
    prog: int
    play: tuple[int, int]       # sounding MIDI, inclusive
    comfort: tuple[int, int]
    home: tuple[int, int]       # 12 notes: the brief's tonic is placed here


@dataclass(frozen=True)
class KeysVoice:
    name: str
    src: str
    bank: int
    prog: int
    vel: Mapping[str, int] = field(hash=False)      # level -> MIDI velocity


@dataclass(frozen=True)
class BassVoice:
    src: str
    bank: int
    prog: int
    vel: Mapping[str, int] = field(hash=False)


@dataclass(frozen=True)
class OneShot:
    """A drum one-shot: a SoundFont note rendered once (src, bank, prog, note, vel), or a file in a source."""

    src: str
    bank: int | None = None
    prog: int | None = None
    note: int | None = None
    vel: int | None = None
    file: str | None = None


@dataclass(frozen=True)
class DrumLane:
    layers: Mapping[str, tuple[OneShot, ...]] = field(hash=False)   # soft/med/hard; a layer may be empty
    pan: float = 0.0            # -1 left .. +1 right
    db: float = 0.0

    def layer_for(self, level: str) -> str:
        """The layer a level plays: X hard, x med, g soft, else the nearest non-empty layer.

        Nearest walks soft-med-hard; at equal distance the softer layer wins.
        """
        want = LAYERS.index(LAYER_OF_LEVEL[level])
        order = sorted(range(len(LAYERS)), key=lambda i: (abs(i - want), i))
        return next(LAYERS[i] for i in order if self.layers[LAYERS[i]])


@dataclass(frozen=True)
class Kit:
    name: str
    bass: BassVoice
    choke: tuple[tuple[str, str], ...]      # (choked lane, choking lane); a lane may choke itself
    lanes: Mapping[str, DrumLane] = field(hash=False)


@dataclass(frozen=True)
class Groove:
    swing: Fraction                                         # default MPC 16th swing
    swing_lanes: Mapping[str, Fraction] = field(hash=False)  # per timing lane, overriding the default
    offset_ms: Mapping[str, Fraction] = field(hash=False)   # fixed per-lane nudge
    humanize: str = field()                                 # off | bar (H1) | loop (H2)
    humanize_ms: Fraction = field()
    humanize_vel: Fraction = field()
    humanize_lanes: frozenset[str] = field()
    drum_gain: Mapping[str, float] = field(hash=False)      # level -> linear sampler gain
    metric_lanes: frozenset[str] = field()
    metric_db: tuple[float, ...] = field()                  # dB per step 0..15 of the bar

    def swing_for(self, lane: str) -> Fraction:
        return self.swing_lanes.get(lane, self.swing)

    def offset_for(self, lane: str) -> Fraction:
        return self.offset_ms.get(lane, Fraction(0))


@dataclass(frozen=True)
class Scoop:
    min_steps: int
    leap: int
    cents: Fraction
    ms: Fraction


@dataclass(frozen=True)
class Vibrato:
    min_steps: int
    start: Fraction             # share of the note's duration before vibrato starts
    fade_ms: Fraction
    cents: Fraction
    rate_hz: Fraction


@dataclass(frozen=True)
class Swell:
    short: int
    long_steps: int
    long_base: int
    long_swell: int
    last: int
    last_share: Fraction


@dataclass(frozen=True)
class SaxVelocity:
    base: int
    bar_start: int
    beat3: int
    sixteenth: int
    phrase_start: int
    min: int
    max: int


@dataclass(frozen=True)
class SaxExpr:
    enabled: bool
    timing: str                 # grid | lag | swung
    lag_ms: Fraction
    swing8: Fraction
    bend_range: int             # semitones
    bend_every_ms: Fraction
    cc11_every_ms: Fraction
    gap_ms: Fraction
    scoop: Scoop
    vibrato: Vibrato
    cc11: Swell
    velocity: SaxVelocity


@dataclass(frozen=True)
class Plugin:
    name: str                   # a pedalboard class
    params: Mapping[str, float] = field(hash=False)


@dataclass(frozen=True)
class Mix:
    bus_gain_db: Mapping[str, float] = field(hash=False)
    bus_target_lu: Mapping[str, float] = field(hash=False)
    sax_warm: bool = field()
    buses: Mapping[str, tuple[Plugin, ...]] = field(hash=False)
    master: tuple[Plugin, ...] = field()


@dataclass(frozen=True)
class Bed:
    src: str
    start_s: float
    xfade_s: float
    delta_lu: float


@dataclass(frozen=True)
class Loudness:
    lufs: float
    tol: float
    iters: int
    ceiling_dbtp: float


@dataclass(frozen=True)
class Palette:
    id: str
    frozen: str | None          # the freeze date, or None
    sr: int
    assets_root: str
    sources: Mapping[str, Source] = field(hash=False)
    sax: Mapping[str, SaxVoice] = field(hash=False)
    sax_lead: str = field()                                  # the one sax every loop plays
    keys: Mapping[str, KeysVoice] = field(hash=False)
    kits: Mapping[str, Kit] = field(hash=False)
    aliases: Mapping[str, str] = field(hash=False)          # alias -> keys, kit or sax name
    numerals: tuple[str, ...] = field()
    qualities: tuple[str, ...] = field()
    voicings: tuple[str, ...] = field()
    groove: Groove = field()
    sax_expr: SaxExpr = field()
    mix: Mix = field()
    bed: Bed | None = field()                               # None: render without a bed
    loudness: Loudness = field()
    path: str = field()
    sha256: str = field()                                   # of the YAML bytes: the palette identity

    def vocab(self) -> Vocab:
        """What parse() checks names against for loops rendered with this palette."""
        return Vocab(keys=frozenset(self.keys), kits=frozenset(self.kits), sax=frozenset(self.sax),
                     voicings=frozenset(self.voicings), numerals=frozenset(self.numerals),
                     qualities=frozenset(self.qualities), name_aliases=dict(self.aliases))

    def resolve(self, kind: str, name: str) -> str:
        """A keys, kits or sax name as the palette lists it, after its aliases; PaletteError if unknown."""
        table = {"keys": self.keys, "kits": self.kits, "sax": self.sax}[kind]
        low = name.strip().lower()
        hit = low if low in table else self.aliases.get(low)
        if hit not in table:
            raise PaletteError(f"{kind} name {name!r} is not in palette {self.id}")
        return hit

    def assets_dir(self, root: Path = ROOT) -> Path:
        """Where mode-fetch sources live: `assets_root` with its `$VARS` and `~` expanded.

        $LOFI_ASSETS, when unset or empty, is <root>/assets/big, where the laptop fetches them
        (docs/SHOOTOUT_ASSETS.md, Route). Any other unset or empty variable is an error. A relative result is
        relative to `root`, like a commit-mode path, never to the current directory."""
        def value(m: re.Match) -> str:
            name = m.group(1) or m.group(2)
            got = os.environ.get(name, "")
            if got:
                return got
            if name == ASSETS_VAR:
                return str(root / ASSETS_DEFAULT)
            raise PaletteError(f"assets_root {self.assets_root!r}: ${name} is unset or empty")
        if "$" in _VAR.sub("", self.assets_root):
            raise PaletteError(f"assets_root {self.assets_root!r} has a '$' that names no variable")
        base = Path(os.path.expanduser(_VAR.sub(value, self.assets_root)))
        return base if base.is_absolute() else root / base

    def source_path(self, src: str, root: Path = ROOT) -> Path:
        s = self.sources[src]
        if s.mode == "commit":
            return root / s.path
        try:
            return self.assets_dir(root) / s.path
        except PaletteError as err:
            raise PaletteError(f"{err} (source {src} is mode fetch)") from err

    def verify_sources(self, root: Path = ROOT) -> list[str]:
        """Problems with the source files: missing, or not matching their sha256. Empty when all good."""
        problems = []
        for src in sorted(self.sources):
            try:
                path = self.source_path(src, root)
            except PaletteError as err:
                problems.append(str(err))
                continue
            if not path.is_file():
                problems.append(f"source {src}: {path} is missing")
            elif _file_sha256(path) != self.sources[src].sha256:
                problems.append(f"source {src}: {path} does not match its sha256")
        return problems

    def presets(self) -> list[tuple[str, str, int, int]]:
        """(where, source, bank, prog) of every SoundFont preset the palette plays, in a fixed order."""
        out = [(f"sax.{k}", v.src, v.bank, v.prog) for k, v in self.sax.items()]
        out += [(f"keys.{k}", v.src, v.bank, v.prog) for k, v in self.keys.items()]
        for name, kit in self.kits.items():
            out.append((f"kits.{name}.bass", kit.bass.src, kit.bass.bank, kit.bass.prog))
            for lane in LANES:
                for layer in LAYERS:
                    out += [(f"kits.{name}.lanes.{lane}.{layer}[{i}]", s.src, s.bank, s.prog)
                            for i, s in enumerate(kit.lanes[lane].layers[layer]) if s.file is None]
        return out

    def verify_presets(self, root: Path = ROOT) -> list[str]:
        """Presets the palette names that its SoundFonts lack. FluidSynth would substitute another preset
        (a drum kit falls back to kit 0, a melodic bank to bank 0) and play a preset the palette never named.
        Empty when all good; run it on files whose sha256 already matched."""
        problems, found = [], {}
        for where, src, bank, prog in self.presets():
            if src not in found:
                try:
                    found[src] = sf2_presets(self.source_path(src, root))
                except (OSError, PaletteError) as err:
                    found[src] = frozenset()
                    problems.append(f"source {src}: {err}")
                    continue
            if found[src] and (bank, prog) not in found[src]:
                problems.append(f"{where}: bank {bank} prog {prog} is not a preset of source {src}")
        return problems

    def note_cells(self) -> list[tuple[str, str, int, int, frozenset[tuple[int, int]]]]:
        """(where, source, bank, prog, cells) of every SoundFont voice, in a fixed order: the (key, velocity)
        cells the compiler may send it (docs/SHOOTOUTS.md §7.5).

        - each sax: its `play` range (the octave fit folds every note into it) at every velocity of its clip,
          sax_expr.velocity min..max, and the flat base velocity it plays with expression off (COMPILER.md §3.5);
        - each keys voice: keys 48-72, where every voicing lies (theory.voicing.KEYS_RANGE), at its `vel` values;
        - each kit bass: keys 28-43 (compile.bass.BASS_RANGE) at its `vel` values;
        - each SoundFont one-shot: its note at its `vel`.
        With humanize on for the keys or bass lane, each `vel` value v widens to rnd(v (1 - h))..rnd(v (1 + h)),
        h = groove.humanize_vel, since a humanized strike plays v times its velocity factor (COMPILER.md §3.6)."""
        # Imported here: compile.bass imports compile.humanize, which imports this module.
        from lofi_rl.compile.bass import BASS_RANGE
        from lofi_rl.theory.voicing import KEYS_RANGE
        v = self.sax_expr.velocity
        sax_vels = set(range(v.min, v.max + 1)) | {v.base}
        out = [(f"sax.{k}", s.src, s.bank, s.prog, _grid(range(s.play[0], s.play[1] + 1), sax_vels))
               for k, s in self.sax.items()]
        keys = range(KEYS_RANGE[0], KEYS_RANGE[1] + 1)
        out += [(f"keys.{k}", s.src, s.bank, s.prog, _grid(keys, self._velocities("keys", s.vel)))
                for k, s in self.keys.items()]
        for name, kit in self.kits.items():
            b = kit.bass
            out.append((f"kits.{name}.bass", b.src, b.bank, b.prog,
                        _grid(range(BASS_RANGE[0], BASS_RANGE[1] + 1), self._velocities("bass", b.vel))))
            for lane in LANES:
                for layer in LAYERS:
                    out += [(f"kits.{name}.lanes.{lane}.{layer}[{i}]", s.src, s.bank, s.prog,
                             frozenset({(s.note, s.vel)}))
                            for i, s in enumerate(kit.lanes[lane].layers[layer]) if s.file is None]
        return out

    def _velocities(self, lane: str, vel: Mapping[str, int]) -> set[int]:
        g = self.groove
        if g.humanize == "off" or lane not in g.humanize_lanes:
            return set(vel.values())
        out: set[int] = set()
        for v in vel.values():
            lo, hi = _round_half_up(v * (1 - g.humanize_vel)), _round_half_up(v * (1 + g.humanize_vel))
            out.update(range(max(1, lo), min(127, hi) + 1))
        return out

    def verify_coverage(self, root: Path = ROOT) -> list[str]:
        """Notes the compiler may send (`note_cells`) that a voice's SoundFont preset has no zone for: FluidSynth
        would play them silent. Each problem lists the missing (key, velocity) cells. Empty when all good; run it
        on files whose sha256 and presets already passed."""
        problems: list[str] = []
        masks: dict[tuple[str, int, int], tuple[int, ...] | None] = {}
        for where, src, bank, prog, want in self.note_cells():
            key = (src, bank, prog)
            if key not in masks:
                try:
                    masks[key] = _coverage_masks(self.source_path(src, root), bank, prog)
                except (OSError, PaletteError) as err:
                    masks[key] = None
                    problems.append(f"{where}: {err}")
                    continue
            got = masks[key]
            if got is None:
                continue
            missing = sorted(c for c in want if not got[c[0]] >> c[1] & 1)
            if missing:
                problems.append(f"{where}: source {src} bank {bank} prog {prog} has no zone for {_cells_text(missing)}")
        return problems


def load_palette(path: Path | str, *, verify: bool = True, root: Path = ROOT) -> Palette:
    """Load and validate a palette file. `verify` also hashes every source file (missing = error), checks every
    SoundFont preset it names, and checks that each voice can sound every note the compiler may send it."""
    path = Path(path)
    data = path.read_bytes()
    try:
        doc = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as err:
        raise PaletteError(f"{path}: not a UTF-8 YAML document ({err})") from err
    pal = parse_palette(doc, str(path), hashlib.sha256(data).hexdigest())
    if verify:
        problems = pal.verify_sources(root) or pal.verify_presets(root) or pal.verify_coverage(root)
        if problems:
            raise PaletteError("; ".join(problems))
    return pal


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pdta(path: Path, wanted: tuple[bytes, ...]) -> dict[bytes, bytes]:
    """The `wanted` sub-chunks of a SoundFont 2 file's pdta list (SoundFont 2.04 §7), by id. Walks the RIFF chunks
    and seeks past the sample data, so it reads a few hundred kB of any bank. A file that is not one, is cut short
    or lacks one of them raises PaletteError."""
    found: dict[bytes, bytes] = {}
    with Path(path).open("rb") as f:
        try:
            riff, size, form = struct.unpack("<4sI4s", f.read(12))
            if riff != b"RIFF" or form != b"sfbk":
                raise PaletteError(f"{path} is not a SoundFont 2 file (no RIFF sfbk header)")
            end = 8 + size
            while f.tell() + 8 <= end and len(found) < len(wanted):
                cid, csize = struct.unpack("<4sI", f.read(8))
                body = f.tell()
                if cid == b"LIST" and f.read(4) == b"pdta":
                    while f.tell() + 8 <= body + csize:
                        sid, ssize = struct.unpack("<4sI", f.read(8))
                        if sid in wanted and sid not in found:
                            data = f.read(ssize)
                            if len(data) != ssize:
                                raise PaletteError(f"{path} is cut short: its {sid.decode()} chunk has "
                                                   f"{len(data)} of {ssize} bytes")
                            found[sid] = data
                            f.seek(ssize & 1, 1)
                        else:
                            f.seek(ssize + (ssize & 1), 1)
                f.seek(body + csize + (csize & 1))
        except struct.error as err:
            raise PaletteError(f"{path} is cut short: {err}") from err
    missing = [w.decode() for w in wanted if w not in found]
    if missing:
        raise PaletteError(f"{path} has no pdta/{', pdta/'.join(missing)} chunk")
    return found


def _count(data: bytes, size: int, what: str, path: Path) -> int:
    """Records in a pdta sub-chunk of `size`-byte records, its terminal record included."""
    if len(data) % size or len(data) < size:
        raise PaletteError(f"{path}: malformed {what} chunk of {len(data)} bytes")
    return len(data) // size


def sf2_presets(path: Path) -> frozenset[tuple[int, int]]:
    """(bank, program) of every preset in a SoundFont 2 file, from its pdta/phdr chunk (SoundFont 2.04 §7.2).

    Bank 128 holds the drum kits. A file that is not one, or is cut short, raises PaletteError."""
    data = _pdta(path, (b"phdr",))[b"phdr"]
    if len(data) % 38 or len(data) < 76:
        raise PaletteError(f"{path}: malformed phdr chunk of {len(data)} bytes")
    # 38-byte records: name[20], preset, bank, bag index, 3 reserved dwords; the last is the EOP terminator.
    return frozenset((bank, prog) for _, prog, bank, *_ in struct.iter_unpack("<20sHHHIII", data[:-38]))


_KEY_RANGE, _VEL_RANGE, _INSTRUMENT, _SAMPLE_ID = 43, 44, 41, 53     # SoundFont 2.04 §8.1.2 generator ids


def _zone_ranges(path: Path, bags: range, bag: bytes, gen: bytes, terminal: int, limit: int,
                 what: str) -> list[tuple[int, int, int, int, int]]:
    """(key lo, key hi, velocity lo, velocity hi, index) of each zone of one preset or instrument that names an
    instrument or a sample (its `terminal` generator), read as FluidSynth 2.6.1 reads them (sffile.c, defsfont.c):
    - a keyRange counts only as its zone's first generator, a velRange only after nothing or a keyRange (SoundFont
      2.04 §8.1.2); generators after the terminal one are dropped;
    - the first zone, when it has no terminal generator, is the global zone: every later zone starts from its
      ranges (0-127 without one) and overrides them with its own; a later zone with no terminal is discarded.
    Measured with FluidSynth 2.6.1 on 2026-10-31: an instrument global zone's keyRange and a preset global zone's
    velRange silence notes outside them, and a keyRange written after a pan is ignored."""
    n_bags, n_gens = len(bag) // 4, len(gen) // 4
    default = (0, 127, 0, 127)
    out = []
    for z, b in enumerate(bags):
        if b + 1 >= n_bags:
            raise PaletteError(f"{path}: a {what} names zone {b} of {n_bags - 1}")
        g0, g1 = struct.unpack_from("<H", bag, 4 * b)[0], struct.unpack_from("<H", bag, 4 * b + 4)[0]
        if not g0 <= g1 <= n_gens:
            raise PaletteError(f"{path}: a {what} zone's generators {g0}..{g1} are outside its {n_gens}")
        klo, khi, vlo, vhi = default
        level, index = 0, None
        for g in range(g0, g1):
            oper, amount = struct.unpack_from("<HH", gen, 4 * g)
            if oper == _KEY_RANGE:
                if level == 0:
                    klo, khi, level = amount & 0xFF, amount >> 8, 1
            elif oper == _VEL_RANGE:
                if level <= 1:
                    vlo, vhi, level = amount & 0xFF, amount >> 8, 2
            elif oper == terminal:
                index = amount
                break
            else:
                level = 2
        if index is None:
            if z == 0:
                default = (klo, khi, vlo, vhi)
            continue
        if index >= limit:
            raise PaletteError(f"{path}: a {what} zone names {'instrument' if terminal == _INSTRUMENT else 'sample'} "
                               f"{index} of {limit}")
        out.append((klo, khi, vlo, vhi, index))
    return out


def _coverage_masks(path: Path, bank: int, prog: int) -> tuple[int, ...]:
    """Per key 0-127, a bit mask of the velocities 0-127 at which preset (bank, prog) sounds a zone."""
    ch = _pdta(path, (b"phdr", b"pbag", b"pgen", b"inst", b"ibag", b"igen", b"shdr"))
    phdr, inst = ch[b"phdr"], ch[b"inst"]
    n_presets = _count(phdr, 38, "phdr", path) - 1
    n_inst = _count(inst, 22, "inst", path) - 1
    n_samples = _count(ch[b"shdr"], 46, "shdr", path) - 1
    for c, size in ((b"pbag", 4), (b"pgen", 4), (b"ibag", 4), (b"igen", 4)):
        _count(ch[c], size, c.decode(), path)
    for i in range(n_presets):
        p, b, bag0 = struct.unpack_from("<HHH", phdr, 38 * i + 20)
        if (b, p) == (bank, prog):
            bag1 = struct.unpack_from("<H", phdr, 38 * (i + 1) + 24)[0]
            break
    else:
        raise PaletteError(f"{path}: bank {bank} prog {prog} is not a preset of this SoundFont")
    masks = [0] * 128
    for klo, khi, vlo, vhi, j in _zone_ranges(path, range(bag0, bag1), ch[b"pbag"], ch[b"pgen"], _INSTRUMENT,
                                              n_inst, "preset"):
        ib0, ib1 = struct.unpack_from("<H", inst, 22 * j + 20)[0], struct.unpack_from("<H", inst, 22 * j + 42)[0]
        for ilo, ihi, jlo, jhi, _ in _zone_ranges(path, range(ib0, ib1), ch[b"ibag"], ch[b"igen"], _SAMPLE_ID,
                                                  n_samples, "instrument"):
            k0, k1 = max(klo, ilo), min(khi, ihi, 127)
            v0, v1 = max(vlo, jlo), min(vhi, jhi, 127)
            if k0 <= k1 and v0 <= v1:
                bits = (1 << (v1 + 1)) - (1 << v0)
                for k in range(k0, k1 + 1):
                    masks[k] |= bits
    return tuple(masks)


def sf2_coverage(path: Path | str, bank: int, prog: int) -> frozenset[tuple[int, int]]:
    """The (key, velocity) cells, 0-127 each, at which preset (bank, prog) of a SoundFont 2 file sounds a sample:
    where one of its preset zones and one of that zone's instrument zones both hold, with global zones read as
    FluidSynth 2.6.1 reads them (`_zone_ranges`). A missing preset, or a file that is not a SoundFont or is
    malformed, raises PaletteError."""
    masks = _coverage_masks(Path(path), bank, prog)
    return frozenset((k, v) for k, m in enumerate(masks) if m for v in range(128) if m >> v & 1)


def _grid(keys: range, vels: set[int]) -> frozenset[tuple[int, int]]:
    return frozenset((k, v) for k in keys for v in vels)


def _round_half_up(x: Fraction) -> int:
    return int((x + Fraction(1, 2)) // 1)


def _runs(values: list[int]) -> str:
    """Sorted integers as runs: '52, 80, 96', '60-112' or '48-49, 97-104'."""
    runs: list[list[int]] = []
    for v in values:
        if runs and runs[-1][1] == v - 1:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _cells_text(cells: list[tuple[int, int]]) -> str:
    """Sorted (key, velocity) cells as key runs that miss the same velocities: 'keys 44-47 at velocity 60-112'."""
    by_key: dict[int, list[int]] = {}
    for k, v in cells:
        by_key.setdefault(k, []).append(v)
    groups: list[list] = []
    for k, vels in by_key.items():
        if groups and groups[-1][1] == k - 1 and groups[-1][2] == vels:
            groups[-1][1] = k
        else:
            groups.append([k, k, vels])
    return "; ".join(f"{f'key {a}' if a == b else f'keys {a}-{b}'} at velocity {_runs(vels)}" for a, b, vels in groups)


# ----------------------------------------------------------------------------------------- checking
def _fail(where: str, what: str) -> PaletteError:
    return PaletteError(f"{where}: {what}")


def licence_problem(licence: str, mode: str, attribution: str | None = None) -> tuple[str, str] | None:
    """(field, why) when a source with this licence cannot be used in this mode (D-012), else None.

    Mode commit takes only the allow-list (CC0, public domain, permissive, CC BY) with no restriction
    named; a non-commercial or unrecognized licence must be mode fetch. A CC BY source (BY-NC too) names
    its attribution in either mode."""
    if mode == "commit":
        if _NC.search(licence):
            return "mode", "a non-commercial source must be mode fetch (never committed)"
        if _NO_COMMIT.search(licence) or not _COMMIT_OK.search(licence):
            return "mode", (f"licence {licence!r} is not one D-012 lets the repo commit (CC0, public domain, "
                            f"permissive, CC BY): make the source mode fetch")
    if _CC_BY.search(licence) and not attribution:
        return "attribution", "a CC-BY source needs its attribution"
    return None


def _map(node: Any, where: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict:
    if not isinstance(node, dict):
        raise _fail(where, f"expected a mapping, got {type(node).__name__}")
    keys = {str(k) for k in node}
    missing = [k for k in required if k not in keys]
    unknown = sorted(keys - set(required) - set(optional))
    if missing:
        raise _fail(where, "missing " + ", ".join(missing))
    if unknown:
        raise _fail(where, "unknown key(s) " + ", ".join(unknown))
    return {str(k): v for k, v in node.items()}


def _str(node: Any, where: str, pattern: re.Pattern | None = None) -> str:
    if not isinstance(node, str) or not node.strip():
        raise _fail(where, "expected a non-empty string")
    if pattern is not None and not pattern.fullmatch(node):
        raise _fail(where, f"{node!r} does not match {pattern.pattern}")
    return node


def _opt_str(node: Any, where: str) -> str | None:
    return None if node is None else _str(node, where)


def _int(node: Any, where: str, lo: int, hi: int) -> int:
    if isinstance(node, bool) or not isinstance(node, int) or not lo <= node <= hi:
        raise _fail(where, f"expected an integer in {lo}..{hi}, got {node!r}")
    return node


def _num(node: Any, where: str, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)) or node != node:
        raise _fail(where, f"expected a number, got {node!r}")
    if (lo is not None and node < lo) or (hi is not None and node > hi):
        raise _fail(where, f"{node!r} is outside {lo}..{hi}")
    return float(node)


def _dec(node: Any, where: str, lo: float | None = None, hi: float | None = None) -> Fraction:
    """A number as the exact decimal written in the file (0.55 -> 11/20)."""
    return Fraction(repr(_num(node, where, lo, hi)))


def _bool(node: Any, where: str) -> bool:
    if not isinstance(node, bool):
        raise _fail(where, f"expected true or false, got {node!r}")
    return node


def _range(node: Any, where: str) -> tuple[int, int]:
    if not isinstance(node, list) or len(node) != 2:
        raise _fail(where, "expected [low, high]")
    lo, hi = (_int(v, f"{where}[{i}]", 0, 127) for i, v in enumerate(node))
    if lo > hi:
        raise _fail(where, f"low {lo} is above high {hi}")
    return lo, hi


def _names(node: Any, where: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(node, list) or not node:
        raise _fail(where, "expected a non-empty list")
    out = []
    for i, v in enumerate(node):
        name = _str(str(v) if isinstance(v, int) and not isinstance(v, bool) else v, f"{where}[{i}]")
        if name not in allowed:
            raise _fail(f"{where}[{i}]", f"{name!r} is not one of {list(allowed)}")
        if name in out:
            raise _fail(f"{where}[{i}]", f"{name!r} is listed twice")
        out.append(name)
    return tuple(out)


def _vel(node: Any, where: str) -> Mapping[str, int]:
    m = _map(node, where, LEVELS)
    return MappingProxyType({lv: _int(m[lv], f"{where}.{lv}", 1, 127) for lv in LEVELS})


def _src(node: Any, where: str, sources: Mapping[str, Source]) -> str:
    src = _str(node, where)
    if src not in sources:
        raise _fail(where, f"source {src!r} is not in sources")
    return src



# ------------------------------------------------------------------------------------------ parsing
_TOP = ("id", "frozen", "sr", "assets_root", "sources", "sax", "sax_lead", "keys", "kits", "aliases",
        "chords", "voicings", "groove", "sax_expr", "mix", "bed", "loudness")


def parse_palette(doc: Any, path: str = "", sha256: str = "") -> Palette:
    """Validate a palette document already read from YAML."""
    top = _map(doc, "palette", _TOP)
    pid = _str(top["id"], "id", _ID)
    frozen = top["frozen"]
    if frozen is not None:
        frozen = str(frozen)
        if not _DATE.fullmatch(frozen):
            raise _fail("frozen", f"expected null or a date YYYY-MM-DD, got {frozen!r}")
    if top["sr"] != SR or isinstance(top["sr"], bool):
        raise _fail("sr", f"must be {SR} (the timeline formula is fixed to 44.1 kHz)")
    sources = _sources(top["sources"])
    sax = _sax(top["sax"], sources)
    sax_lead = _str(top["sax_lead"], "sax_lead")
    if sax_lead not in sax:
        raise _fail("sax_lead", f"{sax_lead!r} is not a sax in this palette")
    keys = _keys(top["keys"], sources)
    kits = _kits(top["kits"], sources)
    aliases = _aliases(top["aliases"], keys, kits, sax)
    chords = _map(top["chords"], "chords", ("numerals", "qualities"))
    return Palette(
        id=pid, frozen=frozen, sr=SR, assets_root=_str(top["assets_root"], "assets_root"),
        sources=sources, sax=sax, sax_lead=sax_lead, keys=keys, kits=kits, aliases=aliases,
        numerals=_names(chords["numerals"], "chords.numerals", NUMERALS),
        qualities=_names(chords["qualities"], "chords.qualities", QUALITIES),
        voicings=_names(top["voicings"], "voicings", VOICINGS),
        groove=_groove(top["groove"]), sax_expr=_sax_expr(top["sax_expr"]), mix=_mix(top["mix"]),
        bed=None if top["bed"] is None else _bed(top["bed"], sources),
        loudness=_loudness(top["loudness"]), path=path, sha256=sha256,
    )


def _table(node: Any, where: str) -> dict:
    if not isinstance(node, dict) or not node:
        raise _fail(where, "expected a non-empty mapping")
    for k in node:
        _str(k, f"{where} key", _ID)
    return node


def _sources(node: Any) -> Mapping[str, Source]:
    out = {}
    for sid, raw in sorted(_table(node, "sources").items()):
        w = f"sources.{sid}"
        m = _map(raw, w, ("path", "sha256", "license", "mode"), ("url", "converted_from", "attribution"))
        sha = _str(m["sha256"], f"{w}.sha256", re.compile(r"[0-9a-f]{64}"))
        lic = _str(m["license"], f"{w}.license")
        mode = _str(m["mode"], f"{w}.mode")
        if mode not in ("commit", "fetch"):
            raise _fail(f"{w}.mode", f"{mode!r} is not commit or fetch")
        attribution = _opt_str(m.get("attribution"), f"{w}.attribution")
        problem = licence_problem(lic, mode, attribution)
        if problem:
            raise _fail(f"{w}.{problem[0]}", problem[1])
        out[sid] = Source(sid, _str(m["path"], f"{w}.path"), sha, lic, mode,
                          _opt_str(m.get("url"), f"{w}.url"),
                          _opt_str(m.get("converted_from"), f"{w}.converted_from"), attribution)
    return MappingProxyType(out)


def _sax(node: Any, sources: Mapping[str, Source]) -> Mapping[str, SaxVoice]:
    out = {}
    for name, raw in sorted(_table(node, "sax").items()):
        w = f"sax.{name}"
        m = _map(raw, w, ("src", "bank", "prog", "play", "comfort", "home"))
        play, comfort, home = (_range(m[k], f"{w}.{k}") for k in ("play", "comfort", "home"))
        if not play[0] <= comfort[0] <= comfort[1] <= play[1]:
            raise _fail(w, "comfort must lie inside play")
        if not (comfort[0] <= home[0] and home[1] <= comfort[1]):
            raise _fail(w, "home must lie inside comfort")
        if home[1] - home[0] != 11:
            raise _fail(f"{w}.home", "must span exactly 12 notes, so each tonic has one place")
        if play[1] - play[0] < 11:
            raise _fail(f"{w}.play", "must span at least 12 notes, so an octave fold always lands")
        out[name] = SaxVoice(name, _src(m["src"], f"{w}.src", sources), _int(m["bank"], f"{w}.bank", 0, 127),
                             _int(m["prog"], f"{w}.prog", 0, 127), play, comfort, home)
    return MappingProxyType(out)


def _keys(node: Any, sources: Mapping[str, Source]) -> Mapping[str, KeysVoice]:
    out = {}
    for name, raw in sorted(_table(node, "keys").items()):
        w = f"keys.{name}"
        m = _map(raw, w, ("src", "bank", "prog", "vel"))
        out[name] = KeysVoice(name, _src(m["src"], f"{w}.src", sources), _int(m["bank"], f"{w}.bank", 0, 127),
                              _int(m["prog"], f"{w}.prog", 0, 127), _vel(m["vel"], f"{w}.vel"))
    return MappingProxyType(out)


def _one_shot(node: Any, where: str, sources: Mapping[str, Source]) -> OneShot:
    if isinstance(node, dict) and "file" in node:
        m = _map(node, where, ("src", "file"))
        return OneShot(src=_src(m["src"], f"{where}.src", sources), file=_str(m["file"], f"{where}.file"))
    m = _map(node, where, ("src", "bank", "prog", "note", "vel"))
    return OneShot(src=_src(m["src"], f"{where}.src", sources),
                   bank=_int(m["bank"], f"{where}.bank", 0, 128),
                   prog=_int(m["prog"], f"{where}.prog", 0, 127),
                   note=_int(m["note"], f"{where}.note", 0, 127),
                   vel=_int(m["vel"], f"{where}.vel", 1, 127))


def _choke_pairs(node: Any, where: str) -> list[tuple[str, str]]:
    """A kit's [choked lane, choking lane] pairs, in order. A lane may choke itself ([ohat, ohat]: the next open
    hat cuts the one before, review T7); a pair may not be listed twice."""
    if not isinstance(node, list):
        raise _fail(where, "expected a list of [choked lane, choking lane]")
    out: list[tuple[str, str]] = []
    for i, pair in enumerate(node):
        if not isinstance(pair, list) or len(pair) != 2:
            raise _fail(f"{where}[{i}]", "expected [choked lane, choking lane]")
        for j, lane in enumerate(pair):
            if _str(lane, f"{where}[{i}][{j}]") not in LANES:
                raise _fail(f"{where}[{i}][{j}]", f"{lane!r} is not one of {list(LANES)}")
        victim, killer = pair
        if (victim, killer) in out:
            raise _fail(f"{where}[{i}]", f"[{victim}, {killer}] is listed twice")
        out.append((victim, killer))
    return out


def _kits(node: Any, sources: Mapping[str, Source]) -> Mapping[str, Kit]:
    out = {}
    for name, raw in sorted(_table(node, "kits").items()):
        w = f"kits.{name}"
        m = _map(raw, w, ("bass", "choke", "lanes"))
        b = _map(m["bass"], f"{w}.bass", ("src", "bank", "prog", "vel"))
        bass = BassVoice(_src(b["src"], f"{w}.bass.src", sources), _int(b["bank"], f"{w}.bass.bank", 0, 127),
                         _int(b["prog"], f"{w}.bass.prog", 0, 127), _vel(b["vel"], f"{w}.bass.vel"))
        choke = _choke_pairs(m["choke"], f"{w}.choke")
        lanes_raw = _map(m["lanes"], f"{w}.lanes", LANES)
        lanes = {}
        for lane in LANES:
            lw = f"{w}.lanes.{lane}"
            lm = _map(lanes_raw[lane], lw, LAYERS + ("pan", "db"))
            layers = {}
            for layer in LAYERS:
                shots = lm[layer]
                if not isinstance(shots, list):
                    raise _fail(f"{lw}.{layer}", "expected a list of one-shots")
                layers[layer] = tuple(_one_shot(s, f"{lw}.{layer}[{i}]", sources)
                                      for i, s in enumerate(shots))
            if not any(layers.values()):
                raise _fail(lw, "every layer is empty")
            lanes[lane] = DrumLane(MappingProxyType(layers), _num(lm["pan"], f"{lw}.pan", -1, 1),
                                   _num(lm["db"], f"{lw}.db", -60, 24))
        out[name] = Kit(name, bass, tuple(choke), MappingProxyType(lanes))
    return MappingProxyType(out)


def _aliases(node: Any, keys: Mapping, kits: Mapping, sax: Mapping) -> Mapping[str, str]:
    if not isinstance(node, dict):
        raise _fail("aliases", "expected a mapping (use {} for none)")
    names = set(keys) | set(kits) | set(sax)
    out = {}
    for alias, target in sorted(node.items()):
        a, t = _str(alias, "aliases key", _ID), _str(target, f"aliases.{alias}")
        if a in names:
            raise _fail(f"aliases.{a}", "an alias cannot also be a palette name")
        if t not in names:
            raise _fail(f"aliases.{a}", f"{t!r} is not a keys, kit or sax name")
        out[a] = t
    return MappingProxyType(out)


def _lane_values(node: Any, where: str, allowed: tuple[str, ...], lo: float, hi: float) -> dict:
    if not isinstance(node, dict):
        raise _fail(where, "expected a mapping")
    out = {}
    for lane, v in node.items():
        if lane not in allowed:
            raise _fail(f"{where}.{lane}", f"not one of {list(allowed)}")
        out[lane] = _dec(v, f"{where}.{lane}", lo, hi)
    return out


def _groove(node: Any) -> Groove:
    m = _map(node, "groove", ("swing", "offset_ms", "humanize", "humanize_ms", "humanize_vel",
                              "humanize_lanes", "drum_gain", "metric_db"))
    swing = _lane_values(m["swing"], "groove.swing", ("default",) + TIMING_LANES, 0.5, 0.75)
    if "default" not in swing:
        raise _fail("groove.swing", "missing default")
    humanize = m["humanize"]
    if humanize is False:
        raise _fail("groove.humanize", 'YAML read off as false: write "off"')
    if humanize not in HUMANIZE_MODES:
        raise _fail("groove.humanize", f"{humanize!r} is not one of {list(HUMANIZE_MODES)}")
    gain = _map(m["drum_gain"], "groove.drum_gain", LEVELS)
    metric = _map(m["metric_db"], "groove.metric_db", ("lanes", "db"))
    db = metric["db"]
    if not isinstance(db, list) or len(db) != 16:
        raise _fail("groove.metric_db.db", "expected 16 numbers, one per step of the bar")
    return Groove(
        swing=swing.pop("default"), swing_lanes=MappingProxyType(swing),
        offset_ms=MappingProxyType(_lane_values(m["offset_ms"], "groove.offset_ms", TIMING_LANES, -50, 50)),
        humanize=humanize, humanize_ms=_dec(m["humanize_ms"], "groove.humanize_ms", 0, 20),
        humanize_vel=_dec(m["humanize_vel"], "groove.humanize_vel", 0, 0.5),
        humanize_lanes=frozenset(_names(m["humanize_lanes"], "groove.humanize_lanes", TIMING_LANES[:-1])),
        drum_gain=MappingProxyType({lv: _num(gain[lv], f"groove.drum_gain.{lv}", 0, 4) for lv in LEVELS}),
        metric_lanes=frozenset(_names(metric["lanes"], "groove.metric_db.lanes", LANES)),
        metric_db=tuple(_num(v, f"groove.metric_db.db[{i}]", -60, 12) for i, v in enumerate(db)),
    )


def _sax_expr(node: Any) -> SaxExpr:
    w = "sax_expr"
    m = _map(node, w, ("enabled", "timing", "lag_ms", "swing8", "bend_range", "bend_every_ms",
                       "cc11_every_ms", "gap_ms", "scoop", "vibrato", "cc11", "velocity"))
    timing = _str(m["timing"], f"{w}.timing")
    if timing not in SAX_TIMINGS:
        raise _fail(f"{w}.timing", f"{timing!r} is not one of {list(SAX_TIMINGS)}")
    s = _map(m["scoop"], f"{w}.scoop", ("min_steps", "leap", "cents", "ms"))
    v = _map(m["vibrato"], f"{w}.vibrato", ("min_steps", "start", "fade_ms", "cents", "rate_hz"))
    c = _map(m["cc11"], f"{w}.cc11", ("short", "long_steps", "long_base", "long_swell", "last", "last_share"))
    vel = _map(m["velocity"], f"{w}.velocity",
               ("base", "bar_start", "beat3", "sixteenth", "phrase_start", "min", "max"))
    bend_range = _int(m["bend_range"], f"{w}.bend_range", 1, 24)
    cc11 = Swell(*(_int(c[k], f"{w}.cc11.{k}", 0, 127) for k in ("short", "long_steps", "long_base",
                                                                 "long_swell", "last")),
                 last_share=_dec(c["last_share"], f"{w}.cc11.last_share", 0, 1))
    if cc11.long_base + cc11.long_swell > 127:
        raise _fail(f"{w}.cc11", "long_base + long_swell must stay within 127")
    velocity = SaxVelocity(*(_int(vel[k], f"{w}.velocity.{k}", -127, 127) for k in
                             ("base", "bar_start", "beat3", "sixteenth", "phrase_start", "min", "max")))
    if not 1 <= velocity.min <= velocity.max <= 127:
        raise _fail(f"{w}.velocity", "needs 1 <= min <= max <= 127")
    scoop = Scoop(_int(s["min_steps"], f"{w}.scoop.min_steps", 1, 128),
                  _int(s["leap"], f"{w}.scoop.leap", 0, 48),
                  _dec(s["cents"], f"{w}.scoop.cents", -100 * bend_range, 100 * bend_range),
                  _dec(s["ms"], f"{w}.scoop.ms", 0, 1000))
    vibrato = Vibrato(_int(v["min_steps"], f"{w}.vibrato.min_steps", 1, 128),
                      _dec(v["start"], f"{w}.vibrato.start", 0, 1),
                      _dec(v["fade_ms"], f"{w}.vibrato.fade_ms", 0, 5000),
                      _dec(v["cents"], f"{w}.vibrato.cents", 0, 100),
                      _dec(v["rate_hz"], f"{w}.vibrato.rate_hz", 0.5, 20))
    if abs(scoop.cents) + vibrato.cents > 100 * bend_range:
        raise _fail(w, "scoop and vibrato depths together exceed the pitch-bend range")
    return SaxExpr(
        enabled=_bool(m["enabled"], f"{w}.enabled"), timing=timing,
        lag_ms=_dec(m["lag_ms"], f"{w}.lag_ms", -50, 100), swing8=_dec(m["swing8"], f"{w}.swing8", 0.5, 0.75),
        bend_range=bend_range, bend_every_ms=_dec(m["bend_every_ms"], f"{w}.bend_every_ms", 1, 100),
        cc11_every_ms=_dec(m["cc11_every_ms"], f"{w}.cc11_every_ms", 1, 100),
        gap_ms=_dec(m["gap_ms"], f"{w}.gap_ms", 0, 100), scoop=scoop, vibrato=vibrato, cc11=cc11,
        velocity=velocity,
    )


def _chain(node: Any, where: str) -> tuple[Plugin, ...]:
    if not isinstance(node, list):
        raise _fail(where, "expected a list of [plugin, parameters]")
    out = []
    for i, item in enumerate(node):
        w = f"{where}[{i}]"
        if not isinstance(item, list) or len(item) != 2:
            raise _fail(w, "expected [plugin, parameters]")
        name = _str(item[0], f"{w}[0]")
        if name not in PLUGINS:
            raise _fail(f"{w}[0]", f"{name!r} is not one of {sorted(PLUGINS)}")
        params = _map(item[1], f"{w}[1]", (), tuple(PLUGINS[name]))
        values = {k: _num(v, f"{w}[1].{k}") for k, v in sorted(params.items())}
        out.append(Plugin(name, MappingProxyType(values)))
    return tuple(out)


def _mix(node: Any) -> Mix:
    m = _map(node, "mix", ("bus_gain_db", "bus_target_lu", "sax_warm", "buses", "master"))
    gains = _map(m["bus_gain_db"], "mix.bus_gain_db", BUS_GAINS)
    targets = _map(m["bus_target_lu"], "mix.bus_target_lu", BUS_GAINS)
    buses = _map(m["buses"], "mix.buses", BUSES)
    return Mix(
        bus_gain_db=MappingProxyType({k: _num(gains[k], f"mix.bus_gain_db.{k}", -60, 24) for k in BUS_GAINS}),
        bus_target_lu=MappingProxyType({k: _num(targets[k], f"mix.bus_target_lu.{k}", -60, 24)
                                        for k in BUS_GAINS}),
        sax_warm=_bool(m["sax_warm"], "mix.sax_warm"),
        buses=MappingProxyType({k: _chain(buses[k], f"mix.buses.{k}") for k in BUSES}),
        master=_chain(m["master"], "mix.master"),
    )


def _bed(node: Any, sources: Mapping[str, Source]) -> Bed:
    m = _map(node, "bed", ("src", "start_s", "xfade_s", "delta_lu"))
    return Bed(_src(m["src"], "bed.src", sources), _num(m["start_s"], "bed.start_s", 0),
               _num(m["xfade_s"], "bed.xfade_s", 0, 10), _num(m["delta_lu"], "bed.delta_lu", 0, 60))


def _loudness(node: Any) -> Loudness:
    m = _map(node, "loudness", ("lufs", "tol", "iters", "ceiling_dbtp"))
    return Loudness(_num(m["lufs"], "loudness.lufs", -40, 0), _num(m["tol"], "loudness.tol", 0, 3),
                    _int(m["iters"], "loudness.iters", 1, 16),
                    _num(m["ceiling_dbtp"], "loudness.ceiling_dbtp", -20, 0))
