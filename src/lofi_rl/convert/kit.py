"""The one-shot kit importer (docs/SHOOTOUTS.md §7.4; CD-30 to CD-35).

    python -m lofi_rl.convert.kit shootouts/kits/<kit>.yaml --big assets/big --out assets/big/derived/kits/<kit>.tar
        [--json report.json] [--reference palettes/dev_gu.yaml] [--reference-kit jazz] [--cache DIR] [--locks DIR]

reads a kit spec, builds every one-shot it names, writes them as a tar of 44.1 kHz mono float32 WAVs and prints the
palette's source entry, `choke:` list and `lanes:` block (one {src, file} per layer and round robin), with a report
in comment lines, as one YAML document. `--json` writes the full report: every one-shot's input files and their
sha256, its trim and its pcm_sha256. Exit 2 for a bad spec or input, with nothing written.

**The spec** (shootouts/kits/<kit>.yaml; paths are relative to --big):
- `id` (the palette source id), `license`, optional `attribution`, optional `mics` (the default premix of every lane
  that names none: mic name -> dB, or mic name -> {db, <variable>: value} for extra path variables);
- `lanes`: all six (kick snare rim hat ohat shaker), each with a `path` template, optional `mics` and `borrowed`,
  and either `vl: {soft, med, hard}` with `rr: [1, 2, ...]` (each layer plays those round robins) or
  `rr: neighbours` (each layer plays vl - 1, vl, vl + 1), or `all_layers: [{variables}, ...]` (one list in all three
  layers). A template takes {mic}, the mic's own variables and the shot's (vl, rr, or the names all_layers uses). A
  lane with no mics, in a kit with none, reads one file per one-shot.
- A one-shot's tar member is <lane>/<its shot variables>.wav (vl4_rr1, vl22, D_rr2), so layers that name the same
  one-shot share one member.

**Per one-shot** [CD-35]: each mic's file is read and a stereo file becomes (L + R) / 2, as the sampler downmixes
anyway (RENDER.md §3.2); the mics are summed with their dB gains in their recorded timing (sample 0 of every file
aligned, shorter files padded at the end); a 48 kHz sum is resampled once (`resample.to_44k`, SHOOTOUTS.md §7.2) and
any rate but 48 and 44.1 kHz is refused; the lead-in is trimmed at the first sample within -30 dB of the one-shot's
peak (-20 dB on a shaker lane), keeping PRE = 44 samples (1 ms, rounded half up) before it under a raised-cosine
fade-in w[k] = (1 - cos(pi k / 44)) / 2, k = 0..43, so that sample sits at 44 in every one-shot (a one-shot whose
threshold sample lies within 1 ms of its start is padded with zeros to keep it there); trailing digital silence
(exact float32 zeros) is dropped. The threshold, the peak and the fade are computed on the float32 samples written.

**Lane levels**: the kit's own lanes keep their recorded balance (`db: 0.0`; the shootout's kit trim, §3, is added
by the palette generator). A borrowed lane (only a shaker may be one) gets o = (ref shaker - ref snare) - (kit
shaker - kit snare), rounded to 0.01 dB [CD-30]: each term the integrated loudness of that lane's med one-shots
alone playing x (gain drum_gain["x"]) on every 8th note of 2 bars at 85 BPM, as a periodic loop through the sampler
(`render.sampler.mix_hits`), round robins dealt by the compiler's rule, pan 0. The reference is a palette kit,
dev_gu's GU Jazz kit by default, whose one-shots FluidSynth renders through the one-shot cache; every kit takes its
choke list ([[ohat, hat], [ohat, ohat]] in dev_gu).

**Inputs** are checked against assets/big/_locks/<folder>.tsv (the sha256 pinned at fetch time) when the folder
has one, and their sha256 is in the report either way.

**The archive** is byte-deterministic: members sorted, USTAR headers with mtime 0, uid/gid 0, empty owner names,
mode 0644 and no pax headers, each WAV written as `wavio.write_float` writes (scipy, no PEAK chunk), uncompressed,
so the sampler reads any member by seeking past the others (never tar.gz or tar.bz2, which it would decompress
for every member). An archive that would also read as a zip, which the sampler tries first, is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import statistics
import string
import sys
import tarfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from scipy.io import wavfile

from lofi_rl.compile.drums import round_robin
from lofi_rl.compile.events import DrumHit
from lofi_rl.compile.timeline import loop_samples, ms, rnd
from lofi_rl.convert.resample import SR_IN, to_44k
from lofi_rl.dsl.vocab import LANES
from lofi_rl.palette import LAYERS, ROOT, PaletteError, load_palette
from lofi_rl.render import sampler
from lofi_rl.render.loudness import lufs
from lofi_rl.render.wavio import pcm_sha256

SR = 44100
PRE = rnd(ms(1))                    # 44 samples kept before the threshold sample
DEFAULT_TRIM_DB = -30.0
TRIM_DB = {"shaker": -20.0}         # a shake builds up over tens of ms (CD-35)
GRID_BPM, GRID_BARS = 85, 2         # the borrowed-lane grid (CD-30)
LANE_DB = (-60.0, 24.0)             # the palette's lane db range
SOFTWARE = "lofi_rl.convert.kit 1"
_ID = re.compile(r"[a-z][a-z0-9_]*")
_VAR = re.compile(r"[a-z][a-z0-9_]*")
_MEMBER = re.compile(r"[A-Za-z0-9_.+-]+")
_SHARES = (Fraction(1, 4), Fraction(3, 5), Fraction(9, 10))       # CD-33, soft / med / hard


class KitError(ValueError):
    pass


# --------------------------------------------------------------------------------------------- the spec
@dataclass(frozen=True)
class Mic:
    name: str                                   # "" for a lane that reads one file per one-shot
    db: float
    vars: tuple[tuple[str, str], ...] = ()      # extra path variables, such as a file-name label


@dataclass(frozen=True)
class Shot:
    member: str                                 # <lane>/<shot variables>.wav
    files: tuple[tuple[str, float], ...]        # (path under --big, dB), one per mic, in mic order


@dataclass(frozen=True)
class LaneSpec:
    name: str
    path: str
    mics: tuple[Mic, ...]
    vl: Mapping[str, int] | None
    rr: tuple[int, ...] | str | None            # round robins, or "neighbours"
    all_layers: tuple[tuple[tuple[str, Any], ...], ...] | None
    borrowed: bool


@dataclass(frozen=True)
class KitSpec:
    id: str
    license: str
    attribution: str | None
    lanes: Mapping[str, LaneSpec]
    path: str
    sha256: str


def _keys(node: Any, where: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict:
    if not isinstance(node, dict):
        raise KitError(f"{where}: expected a mapping")
    keys = {str(k) for k in node}
    missing = [k for k in required if k not in keys]
    unknown = sorted(keys - set(required) - set(optional))
    if missing:
        raise KitError(f"{where}: missing {', '.join(missing)}")
    if unknown:
        raise KitError(f"{where}: unknown key(s) {', '.join(unknown)}")
    return {str(k): v for k, v in node.items()}


def _number(node: Any, where: str) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)) or node != node:
        raise KitError(f"{where}: expected a number, got {node!r}")
    return float(node)


def _positive(node: Any, where: str) -> int:
    if isinstance(node, bool) or not isinstance(node, int) or node < 1:
        raise KitError(f"{where}: expected a whole number of 1 or more, got {node!r}")
    return node


def _mics(node: Any, where: str) -> tuple[Mic, ...]:
    if not isinstance(node, dict) or not node:
        raise KitError(f"{where}: expected a mapping of mic name -> dB or {{db, variables}}")
    out = []
    for name, value in node.items():
        if not isinstance(name, str) or not _VAR.fullmatch(name):
            raise KitError(f"{where}: mic name {name!r} must match {_VAR.pattern}")
        if isinstance(value, dict):
            m = _keys(value, f"{where}.{name}", ("db",), tuple(k for k in map(str, value) if k != "db"))
            extra = []
            for k, v in m.items():
                if k == "db":
                    continue
                if not _VAR.fullmatch(k) or k in ("mic", "vl", "rr"):
                    raise KitError(f"{where}.{name}: variable {k!r} must match {_VAR.pattern} and not be mic, vl or rr")
                if not isinstance(v, (str, int)) or isinstance(v, bool):
                    raise KitError(f"{where}.{name}.{k}: expected text or a whole number")
                extra.append((k, str(v)))
            out.append(Mic(name, _number(m["db"], f"{where}.{name}.db"), tuple(extra)))
        else:
            out.append(Mic(name, _number(value, f"{where}.{name}")))
    return tuple(out)


def _lane(name: str, node: Any, kit_mics: tuple[Mic, ...] | None) -> LaneSpec:
    w = f"lanes.{name}"
    m = _keys(node, w, ("path",), ("mics", "vl", "rr", "all_layers", "borrowed"))
    path = m["path"]
    if not isinstance(path, str) or not path.strip() or path.startswith("/") or ".." in Path(path).parts:
        raise KitError(f"{w}.path: expected a template relative to --big")
    mics = _mics(m["mics"], f"{w}.mics") if "mics" in m else kit_mics or (Mic("", 0.0),)
    borrowed = m.get("borrowed", False)
    if not isinstance(borrowed, bool):
        raise KitError(f"{w}.borrowed: expected true or false")
    if borrowed and name != "shaker":
        raise KitError(f"{w}.borrowed: only a shaker lane is set by the reference kit's shaker-to-snare difference")
    vl = rr = layers = None
    if "all_layers" in m:
        if "vl" in m or "rr" in m:
            raise KitError(f"{w}: give either vl with rr, or all_layers")
        raw = m["all_layers"]
        if not isinstance(raw, list) or not raw:
            raise KitError(f"{w}.all_layers: expected a non-empty list of {{variable: value}}")
        shots = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict) or not item:
                raise KitError(f"{w}.all_layers[{i}]: expected a mapping of variable -> value")
            for k, v in item.items():
                if not isinstance(k, str) or not _VAR.fullmatch(k) or k == "mic":
                    raise KitError(f"{w}.all_layers[{i}]: variable {k!r} must match {_VAR.pattern} and not be mic")
                if not isinstance(v, (str, int)) or isinstance(v, bool):
                    raise KitError(f"{w}.all_layers[{i}].{k}: expected text or a whole number")
            shots.append(tuple(item.items()))
        layers = tuple(shots)
    else:
        if "vl" not in m or "rr" not in m:
            raise KitError(f"{w}: give vl and rr (a list of round robins or neighbours), or all_layers")
        v = _keys(m["vl"], f"{w}.vl", LAYERS)
        vl = {layer: _positive(v[layer], f"{w}.vl.{layer}") for layer in LAYERS}
        if m["rr"] == "neighbours":
            rr = "neighbours"
        elif isinstance(m["rr"], list) and m["rr"]:
            rr = tuple(_positive(x, f"{w}.rr[{i}]") for i, x in enumerate(m["rr"]))
            if len(set(rr)) != len(rr):
                raise KitError(f"{w}.rr: a round robin is listed twice")
        else:
            raise KitError(f"{w}.rr: expected a list of round robins or neighbours, got {m['rr']!r}")
    return LaneSpec(name, path, mics, vl, rr, layers, borrowed)


def load_spec(path: Path | str) -> KitSpec:
    """Read and check a kit spec; every one-shot's file paths are expanded once, so a template error shows here."""
    path = Path(path)
    data = path.read_bytes()
    try:
        doc = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as err:
        raise KitError(f"{path}: not a UTF-8 YAML document ({err})") from err
    top = _keys(doc, "kit", ("id", "license", "lanes"), ("attribution", "mics"))
    if not isinstance(top["id"], str) or not _ID.fullmatch(top["id"]):
        raise KitError(f"id: {top['id']!r} must match {_ID.pattern} (it is the palette source id)")
    if not isinstance(top["license"], str) or not top["license"].strip():
        raise KitError("license: expected a non-empty string")
    attribution = top.get("attribution")
    if attribution is not None and (not isinstance(attribution, str) or not attribution.strip()):
        raise KitError("attribution: expected null or a non-empty string")
    kit_mics = _mics(top["mics"], "mics") if top.get("mics") is not None else None
    lanes_raw = _keys(top["lanes"], "lanes", LANES)
    lanes = {lane: _lane(lane, lanes_raw[lane], kit_mics) for lane in LANES}
    try:
        shown = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(path)
    spec = KitSpec(top["id"], top["license"], attribution, lanes, shown, hashlib.sha256(data).hexdigest())
    members: dict[str, Shot] = {}
    for lane in LANES:
        for shots in expand(lanes[lane]).values():
            for s in shots:
                if members.setdefault(s.member, s) != s:
                    raise KitError(f"lanes.{lane}: one-shot {s.member} names two different file sets")
    return spec


def _shot_id(variables: Sequence[tuple[str, Any]]) -> str:
    return "_".join(f"{k}{v}" if k in ("vl", "rr") else str(v) for k, v in variables)


def expand(lane: LaneSpec) -> dict[str, tuple[Shot, ...]]:
    """Each layer's one-shots, in round-robin order, with the files every mic reads for them."""
    out = {}
    for layer in LAYERS:
        if lane.all_layers is not None:
            variables = list(lane.all_layers)
        elif lane.rr == "neighbours":
            v = lane.vl[layer]
            variables = [(("vl", v - 1),), (("vl", v),), (("vl", v + 1),)]
        else:
            variables = [(("vl", lane.vl[layer]), ("rr", r)) for r in lane.rr]
        shots = []
        for shot_vars in variables:
            sid = _shot_id(shot_vars)
            if not _MEMBER.fullmatch(sid):
                raise KitError(f"lanes.{lane.name}: one-shot name {sid!r} must match {_MEMBER.pattern}")
            files = []
            for mic in lane.mics:
                values = {"mic": mic.name, **dict(mic.vars), **{k: str(v) for k, v in shot_vars}}
                try:
                    rel = lane.path.format_map(values)
                except KeyError as err:
                    raise KitError(f"lanes.{lane.name}.path: {lane.path!r} names {{{err.args[0]}}}, which neither "
                                   f"the mic nor the one-shot gives") from err
                except (ValueError, IndexError) as err:
                    raise KitError(f"lanes.{lane.name}.path: {lane.path!r} is not a template ({err})") from err
                files.append((rel, mic.db))
            if len({f for f, _ in files}) != len(files):
                raise KitError(f"lanes.{lane.name}: two mics read the same file {files[0][0]!r}; name {{mic}} or a "
                               f"mic variable in the path")
            shots.append(Shot(f"{lane.name}/{sid}.wav", tuple(files)))
        out[layer] = tuple(shots)
    return out


def cd33_layers(n: int) -> dict[str, int]:
    """CD-33: of a set's N velocity layers, soft / med / hard take floor(0.25 N + 1/2), floor(0.6 N + 1/2) and
    floor(0.9 N + 1/2)."""
    return {layer: int(share * n + Fraction(1, 2)) for layer, share in zip(LAYERS, _SHARES)}


def layer_count(lane: LaneSpec, big: Path) -> int:
    """N, the velocity layers of a vl lane's set on disk: the {vl} values of the files in its first one-shot's
    folder that its first mic's file-name template matches (1..N, none missing)."""
    if lane.vl is None:
        raise KitError(f"lanes.{lane.name} has no velocity layers")
    mic = lane.mics[0]
    known = {"mic": mic.name, **dict(mic.vars)}
    rx = ""
    for literal, field, _, _ in string.Formatter().parse(lane.path.rsplit("/", 1)[-1]):
        rx += re.escape(literal)
        if field is not None:
            rx += r"(?P<vl>\d+)" if field == "vl" else re.escape(known[field]) if field in known else r"[^/]+"
    folder = (Path(big) / expand(lane)["soft"][0].files[0][0]).parent
    found = {int(m.group("vl")) for p in folder.iterdir() if (m := re.fullmatch(rx, p.name))}
    if not found or found != set(range(1, max(found) + 1)):
        raise KitError(f"lanes.{lane.name}: velocity layers {sorted(found)} in {folder} are not 1..N")
    return max(found)


# ---------------------------------------------------------------------------------------- one one-shot
def _read(path: Path) -> tuple[np.ndarray, int, dict]:
    try:
        info = sf.info(str(path))
        x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    except RuntimeError as err:                         # soundfile's LibsndfileError
        raise KitError(f"{path}: {err}") from err
    if x.shape[1] not in (1, 2) or x.shape[0] == 0:
        raise KitError(f"{path} has {x.shape[1]} channel(s) and {x.shape[0]} frames; a one-shot file is mono or "
                       f"stereo audio")
    m = np.ascontiguousarray(x[:, 0]) if x.shape[1] == 1 else (x[:, 0] + x[:, 1]) * 0.5
    peak = float(np.abs(m).max())
    first = int(np.flatnonzero(np.abs(m) >= peak * 0.1)[0]) if peak > 0 else None
    return m, int(sr), {"rate": int(sr), "channels": int(x.shape[1]), "frames": int(x.shape[0]),
                        "subtype": info.subtype, "first_within_20db": first}


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    """A file as float64 mono, a stereo file as (L + R) / 2 (RENDER.md §3.2), and its rate."""
    m, sr, _ = _read(Path(path))
    return m, sr


def premix(parts: Sequence[tuple[np.ndarray, int, float]]) -> tuple[np.ndarray, int]:
    """The mics of one one-shot, (mono signal, rate, dB) each, summed in float64 in their recorded timing: sample
    0 of every mic aligned, a shorter one padded with zeros at its end."""
    rates = sorted({sr for _, sr, _ in parts})
    if len(rates) != 1:
        raise KitError(f"the mics of one one-shot are at {' and '.join(map(str, rates))} Hz")
    out = np.zeros(max(len(x) for x, _, _ in parts), dtype=np.float64)
    for x, _, db in parts:
        out[:len(x)] += x * (10.0 ** (db / 20.0))
    return out, rates[0]


def trim(y: np.ndarray, threshold_db: float) -> tuple[np.ndarray, dict]:
    """The one-shot from PRE samples before its first sample within `threshold_db` of its peak, faded in over those
    PRE samples by a raised cosine, without trailing digital silence: (float32 samples, what was done)."""
    y = np.asarray(y, dtype=np.float32)
    mag = np.abs(y).astype(np.float64)
    peak = float(mag.max(initial=0.0))
    if peak == 0.0:
        raise KitError("the one-shot is silent")
    crossing = int(np.flatnonzero(mag >= peak * 10.0 ** (threshold_db / 20.0))[0])
    start = crossing - PRE
    body = y[max(0, start):].astype(np.float64)
    padded = max(0, -start)
    if padded:
        body = np.concatenate([np.zeros(padded), body])
    body[:PRE] *= np.array([0.5 * (1.0 - math.cos(math.pi * k / PRE)) for k in range(PRE)])
    out = body.astype(np.float32)
    out = np.ascontiguousarray(out[:int(np.flatnonzero(out)[-1]) + 1])
    return out, {"crossing": crossing, "trimmed": max(0, start), "padded": padded, "samples": int(len(out)),
                 "peak_dbfs": round(20.0 * math.log10(peak), 3), "threshold_db": threshold_db}


def one_shot(files: Sequence[tuple[Path, float]], lane: str) -> tuple[np.ndarray, dict]:
    """One one-shot of `lane` from its mics' files, (path, dB) each: mono float32 at 44.1 kHz, and its report."""
    parts, inputs = [], []
    for path, db in files:
        m, sr, info = _read(Path(path))
        parts.append((m, sr, db))
        inputs.append(info)
    mixed, sr = premix(parts)
    if sr == SR_IN:
        mixed = to_44k(mixed, sr)
    elif sr != SR:
        raise KitError(f"{files[0][0]} is {sr} Hz; one-shots are 44.1 kHz, or 48 kHz converted once "
                       f"(SHOOTOUTS.md §7.2)")
    y, info = trim(mixed.astype(np.float32), TRIM_DB.get(lane, DEFAULT_TRIM_DB))
    info.update(rate_in=sr, resampled=sr == SR_IN, premix_samples=int(len(mixed)), inputs=inputs)
    return y, info


# ----------------------------------------------------------------------------------- borrowed lane level
@dataclass(frozen=True, eq=False)
class Reference:
    """The kit a borrowed lane's level is set against (CD-30), and the choke list every kit takes."""

    name: str
    choke: tuple[tuple[str, str], ...]
    shaker: tuple[np.ndarray, ...]              # mono one-shots its x hits play (med layer), by round robin
    snare: tuple[np.ndarray, ...]
    gain: float                                 # groove.drum_gain["x"]


def reference_from_palette(path: Path | str, kit: str = "jazz", *, cache: Path | str | None = None,
                           shots: bool = True, root: Path = ROOT) -> Reference:
    """A palette kit as the reference. With `shots`, the palette is verified and the one-shots an x hit plays on
    its shaker and snare lanes are loaded through the sampler (FluidSynth and the one-shot cache for SoundFont
    one-shots); without, only its choke list and x gain are read."""
    pal = load_palette(path, verify=shots, root=root)
    if kit not in pal.kits:
        raise KitError(f"reference kit {kit!r} is not in palette {pal.id}")
    k = pal.kits[kit]

    def x_hits(lane: str) -> tuple[np.ndarray, ...]:
        drum = k.lanes[lane]
        return tuple(sampler.mono(sampler.load_shot(s, pal, cache=cache, root=root)[0])
                     for s in drum.layers[drum.layer_for("x")])
    return Reference(f"{pal.id}.{kit}", k.choke, x_hits("shaker") if shots else (),
                     x_hits("snare") if shots else (), pal.groove.drum_gain["x"])


def grid_hits(lane: str, count: int, gain: float) -> tuple[int, list[DrumHit]]:
    """x on every 8th note of GRID_BARS bars at GRID_BPM, a periodic loop of n samples, the lane-layer's `count`
    one-shots dealt by the compiler's round robin: (n, hits)."""
    n = loop_samples(GRID_BPM) * GRID_BARS // 8
    eighths = GRID_BARS * 8
    step = Fraction(n, eighths)
    return n, [DrumHit(lane, "x", "med", rnd(h * step), gain, round_robin(h, eighths, count), 2 * h)
               for h in range(eighths)]


def grid_lufs(shots: Sequence[np.ndarray], gain: float, db: float = 0.0, lane: str = "snare") -> float:
    """Integrated loudness of `shots` on the grid, through the sampler, at lane level `db`."""
    if not shots:
        raise KitError(f"no one-shots for the {lane} grid")
    n, hits = grid_hits(lane, len(shots), gain)
    y = sampler.mix_hits(hits, {(lane, "med"): tuple(shots)}, {lane: (0.0, db)}, n)
    return lufs(y, SR)


def borrowed_offset(kit_shaker: Sequence[np.ndarray], kit_snare: Sequence[np.ndarray],
                    ref: Reference) -> tuple[float, dict]:
    """o = (ref shaker - ref snare) - (kit shaker - kit snare) on the grid, rounded to 0.01 dB [CD-30]."""
    if not ref.shaker or not ref.snare:
        raise KitError(f"reference {ref.name} carries no shaker and snare one-shots")
    terms = {"ref_shaker_lufs": grid_lufs(ref.shaker, ref.gain, lane="shaker"),
             "ref_snare_lufs": grid_lufs(ref.snare, ref.gain),
             "kit_shaker_lufs": grid_lufs(kit_shaker, ref.gain, lane="shaker"),
             "kit_snare_lufs": grid_lufs(kit_snare, ref.gain)}
    o = round((terms["ref_shaker_lufs"] - terms["ref_snare_lufs"])
              - (terms["kit_shaker_lufs"] - terms["kit_snare_lufs"]), 2) + 0.0
    if not LANE_DB[0] <= o <= LANE_DB[1]:
        raise KitError(f"the borrowed lane's offset {o} dB is outside the palette's lane range {LANE_DB}")
    return o, dict(terms, offset_db=o, grid=f"x on every 8th note, {GRID_BARS} bars at {GRID_BPM} BPM, "
                                          f"gain {ref.gain}, med layer")


# ------------------------------------------------------------------------------------------- the archive
def wav_bytes(y: np.ndarray) -> bytes:
    """A mono float32 WAV with no PEAK chunk, as wavio.write_float writes."""
    buf = io.BytesIO()
    wavfile.write(buf, SR, np.ascontiguousarray(y, dtype=np.float32))
    return buf.getvalue()


def tar_bytes(members: Mapping[str, bytes]) -> bytes:
    """An uncompressed USTAR archive of `members`, sorted, with mtime 0, uid/gid 0, no owner names, mode 0644."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as t:
        for name in sorted(members):
            data = members[name]
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode, info.uid, info.gid = len(data), 0, 0o644, 0, 0
            info.uname = info.gname = ""
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _lock(path: Path) -> dict[str, str] | None:
    """path -> sha256 of a bank's lock file (assets/big/_locks/<folder>.tsv), or None when there is none."""
    if not path.is_file():
        return None
    rows = path.read_text(encoding="utf-8").splitlines()
    head = rows[0].split("\t") if rows else []
    if "path" not in head or "sha256" not in head:
        raise KitError(f"{path}: expected a header with path and sha256")
    i, j = head.index("path"), head.index("sha256")
    return {cols[i]: cols[j] for cols in (r.split("\t") for r in rows[1:] if r.strip())}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class KitImport:
    spec: KitSpec
    data: bytes                                 # the tar
    members: dict[str, np.ndarray]              # member -> mono float32 one-shot
    lanes: dict[str, dict[str, list[str]]]      # lane -> layer -> members, in round-robin order
    offsets: dict[str, float]                   # borrowed lane -> dB
    choke: tuple[tuple[str, str], ...]
    report: dict

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def blocks(self, source_path: str) -> dict:
        """The palette's source entry for the archive at `source_path` (under assets_root), and the kit's choke
        and lanes blocks."""
        folders = sorted({f.split("/", 1)[0] for s in self.report["shots"].values() for f in
                          (i["path"] for i in s["files"])})
        source = {"path": source_path, "sha256": self.sha256, "license": self.spec.license, "mode": "fetch",
                  "url": None, "converted_from": f"{self.spec.path} (sha256 {self.spec.sha256[:12]}) by {SOFTWARE} "
                                                f"from {', '.join(f + '/' for f in folders)}",
                  "attribution": self.spec.attribution}
        lanes = {}
        for lane in LANES:
            entry: dict[str, Any] = {layer: [{"src": self.spec.id, "file": m} for m in self.lanes[lane][layer]]
                                     for layer in LAYERS}
            entry["pan"] = 0.0
            entry["db"] = self.offsets.get(lane, 0.0)
            lanes[lane] = entry
        return {"sources": {self.spec.id: source}, "choke": [list(p) for p in self.choke], "lanes": lanes}


def _stats(values: list[int]) -> dict:
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def import_kit(spec: KitSpec, big: Path | str, *, reference: Reference, locks: Path | str | None = None) -> KitImport:
    """Build every one-shot of `spec` from the sample folders under `big` and the archive that holds them."""
    big = Path(big)
    locks = big / "_locks" if locks is None else Path(locks)
    plan = {lane: expand(spec.lanes[lane]) for lane in LANES}
    shots = {s.member: s for lane in LANES for layer in LAYERS for s in plan[lane][layer]}
    files: dict[str, dict] = {}
    tables: dict[str, dict[str, str] | None] = {}
    checked: dict[str, int] = {}
    for member in sorted(shots):
        for rel, _ in shots[member].files:
            if rel in files:
                continue
            p = big / rel
            if not p.is_file():
                raise KitError(f"{member}: {rel} is missing (under {big})")
            digest = _sha256(p)
            folder, _, inner = rel.partition("/")
            if folder not in tables:
                tables[folder] = _lock(locks / f"{folder}.tsv")
            table = tables[folder]
            if table is not None:
                if table.get(inner) != digest:
                    raise KitError(f"{member}: {rel} does not match its sha256 in {locks / (folder + '.tsv')}")
                checked[folder] = checked.get(folder, 0) + 1
            files[rel] = {"sha256": digest}
    members, info = {}, {}
    for member in sorted(shots):
        s = shots[member]
        try:
            y, report = one_shot([(big / rel, db) for rel, db in s.files], member.split("/", 1)[0])
        except KitError as err:
            raise KitError(f"{member}: {err}") from err
        inputs = report.pop("inputs")
        report["files"] = [dict(path=rel, db=db, **files[rel], **i) for (rel, db), i in zip(s.files, inputs)]
        report["pcm_sha256"] = pcm_sha256(y[None, :])
        members[member], info[member] = y, report
    lanes = {lane: {layer: [s.member for s in plan[lane][layer]] for layer in LAYERS} for lane in LANES}
    offsets, borrowed = {}, {}
    for lane in LANES:
        if spec.lanes[lane].borrowed:
            o, terms = borrowed_offset([members[m] for m in lanes[lane]["med"]],
                                       [members[m] for m in lanes["snare"]["med"]], reference)
            offsets[lane], borrowed[lane] = o, terms
    data = tar_bytes({m: wav_bytes(y) for m, y in members.items()})
    if zipfile.is_zipfile(io.BytesIO(data)):
        raise KitError("the archive also reads as a zip, which the sampler tries first; it cannot be used")
    trims = [info[m]["trimmed"] for m in sorted(info)]
    report = {
        "id": spec.id, "spec": spec.path, "spec_sha256": spec.sha256, "software": SOFTWARE,
        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "members": len(members),
        "files": len(files), "reference": reference.name, "choke": [list(p) for p in reference.choke],
        "lanes": {lane: {"rr": {layer: len(lanes[lane][layer]) for layer in LAYERS},
                         "one_shots": len({m for layer in LAYERS for m in lanes[lane][layer]}),
                         "threshold_db": TRIM_DB.get(lane, DEFAULT_TRIM_DB),
                         "mics": [{"name": mic.name, "db": mic.db} for mic in spec.lanes[lane].mics],
                         "db": offsets.get(lane, 0.0),
                         "lead_in": _stats([info[m]["trimmed"] for m in
                                            sorted({m for layer in LAYERS for m in lanes[lane][layer]})])}
                  for lane in LANES},
        "lead_in": dict(_stats(trims), unit="samples at 44.1 kHz removed before the 44 kept",
                        padded=sum(1 for m in info if info[m]["padded"])),
        "borrowed": borrowed,
        "locks": {folder: (f"{checked.get(folder, 0)} files match" if tables[folder] is not None else "no lock")
                  for folder in sorted(tables)},
        "shots": info,
    }
    return KitImport(spec, data, members, lanes, offsets, tuple(tuple(p) for p in reference.choke), report)


def source_path(out: Path | str, big: Path | str) -> str:
    """The archive's path as the palette's fetch-mode source gives it: relative to assets_root (--big) when it
    lies under it, else as given."""
    try:
        return Path(out).resolve().relative_to(Path(big).resolve()).as_posix()
    except ValueError:
        return str(out)


def blocks_text(imp: KitImport, blocks: dict) -> str:
    """The blocks as one YAML document, the report in comment lines above them."""
    r = imp.report
    li = r["lead_in"]
    lines = [f"# {imp.spec.id}: sha256 {imp.sha256} ({len(imp.data):,} bytes), {r['members']} one-shots from "
             f"{r['files']} files; spec {r['spec']} (sha256 {r['spec_sha256'][:12]}); {SOFTWARE}",
             "# one-shots per layer (soft/med/hard): " + "; ".join(
                 f"{lane} {'/'.join(str(r['lanes'][lane]['rr'][layer]) for layer in LAYERS)}" for lane in LANES),
             f"# lead-in trimmed before the 44 samples kept (44.1 kHz; min/median/max): {li['min']}/{li['median']}/"
             f"{li['max']}; {li['padded']} one-shot(s) padded to sample 44",
             f"# reference {r['reference']}; locks: " + ", ".join(f"{k} {v}" for k, v in r["locks"].items())]
    for lane, t in r["borrowed"].items():
        lines.append(f"# borrowed {lane}: db {t['offset_db']} = (ref shaker {t['ref_shaker_lufs']:.3f} - ref snare "
                     f"{t['ref_snare_lufs']:.3f}) - (kit shaker {t['kit_shaker_lufs']:.3f} - kit snare "
                     f"{t['kit_snare_lufs']:.3f}) LUFS")
    lines.append("# sources goes into the palette's sources; choke and lanes into kits.<name> beside its bass.")
    head = yaml.safe_dump({"sources": blocks["sources"]}, sort_keys=False, default_flow_style=False, width=1000)
    rest = yaml.safe_dump({k: v for k, v in blocks.items() if k != "sources"}, sort_keys=False,
                          default_flow_style=None, width=120)
    return "\n".join(lines) + "\n" + head + rest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lofi_rl.convert.kit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path, help="a kit spec, shootouts/kits/<kit>.yaml")
    ap.add_argument("--big", type=Path, required=True, help="the folder the spec's paths are relative to")
    ap.add_argument("--out", type=Path, required=True, help="the .tar to write")
    ap.add_argument("--json", type=Path, help="also write the full report as JSON here")
    ap.add_argument("--reference", type=Path, default=ROOT / "palettes" / "dev_gu.yaml",
                    help="the palette whose kit sets a borrowed lane's level and the choke list (dev_gu)")
    ap.add_argument("--reference-kit", default="jazz", help="that palette's kit (jazz)")
    ap.add_argument("--cache", type=Path, help="the one-shot cache for the reference's SoundFont one-shots")
    ap.add_argument("--locks", type=Path, help="the lock folder (default <big>/_locks)")
    a = ap.parse_args(argv)
    try:
        spec = load_spec(a.spec)
        borrowed = any(lane.borrowed for lane in spec.lanes.values())
        ref = reference_from_palette(a.reference, a.reference_kit, cache=a.cache, shots=borrowed)
        imp = import_kit(spec, a.big, reference=ref, locks=a.locks)
    except (KitError, PaletteError, OSError) as err:
        print(f"kit: {err}", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".part")
    tmp.write_bytes(imp.data)
    tmp.replace(a.out)
    blocks = imp.blocks(source_path(a.out, a.big))
    if a.json:
        a.json.write_text(json.dumps(dict(imp.report, out=str(a.out), blocks=blocks), indent=1) + "\n")
    print(blocks_text(imp, blocks), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
