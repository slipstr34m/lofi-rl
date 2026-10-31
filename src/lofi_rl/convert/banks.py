"""The six shootout SFZ banks, each reduced to what SF2 can express and written as one preset at bank 0, program 0
(docs/CONVERTER.md, docs/SHOOTOUT_ASSETS.md "What the research corrected in the design").

convert_sfz() is the whole pipeline: parse, evaluate at the controller defaults, select, read the samples
losslessly, map the regions to zones, shift the attenuation bank-wide and write the bytes. convert_bank() runs it
on one of BANKS from an assets/big folder. Paths inside the file are bank-relative, so a bank converts to the
same bytes wherever assets/big lives.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from lofi_rl.convert import sf2
from lofi_rl.convert.mapping import ZoneSpec, attenuation_generator, bank_shift, zone_spec
from lofi_rl.convert.sfz import Dropped, Region, Resolved, SfzError, parse_sfz, select

SOFTWARE = "lofi-rl convert 1 (SFZ to SF2, docs/CONVERTER.md)"
GUARD = 8                     # points a loop keeps after its end (SoundFont 2.04 section 7.10)


@dataclass(frozen=True)
class Bank:
    id: str
    folder: str                         # under assets/big
    sfz: str                            # the program, relative to the folder
    preset: str                         # preset and instrument name, at most 19 characters
    copyright: str                      # INFO ICOP
    why: str                            # the reduction, in one paragraph (printed by the CLI)
    keep: Callable[[Region, object], str | None] | None = None      # extra drop rule: a reason or None
    expect: Callable[[list[Resolved]], list[str]] | None = None     # checks on the kept regions


def _drop_mtg_noise(region: Region, ev) -> str | None:
    if region.opcodes.get("master_label") == "noises":
        return "noise region (breath and key noise; the bank keeps notes only)"
    return None


def _samples_match(pattern: str, count: int):
    rx = re.compile(pattern)

    def check(kept: list[Resolved]) -> list[str]:
        bad = sorted({r.sample.name for r in kept if not rx.search(r.sample.as_posix())})
        problems = [f"kept sample {b} does not match {pattern}" for b in bad[:5]]
        if len(kept) != count:
            problems.append(f"{len(kept)} regions kept, {count} expected")
        return problems
    return check


BANKS: dict[str, Bank] = {b.id: b for b in (
    Bank("weresax-alto", "karoryfer-weresax", "Programs/Sax.sfz", "weresax-alto",
         "CC0 1.0: Weresax v1.003 by Karoryfer Samples",
         "Programs/Sax.sfz is the bank's sax instrument (the readme's other two, saxcordion and sxnth, are an "
         "accordion and a synth built on the same samples); Sax1.sfz is the same file without off_mode=fast, which "
         "the reduction drops, so both give the same SF2. Mic Blend CC120 has no set_cc, so it reads 0, the clean "
         "condenser end: the dynamic-mic layers fade to gain 0 there and drop out. Kept: the p (1-63) and f "
         "(64-127) condenser layers, round robin 1. Dropped: rr2, the dynamic mic, the generated wind noise, and "
         "the CC-driven vibrato, humanize, EQ and bend range.",
         expect=_samples_match(r"_[pf]_rr1_cnd\.wav$", 64)),
    Bank("mtg-tenor", "mtg-solo-sax", "MTG Solo Saxophones/MTG Tenor Sax (NL).sfz", "mtg-tenor",
         "CC BY 4.0 (SFZ mapping by kinwie) and CC BY 3.0 (samples by MTG, Universitat Pompeu Fabra, via Freesound)",
         "The non-looped (NL) preset, round robin 1 (seq_position 1, each key's own sample), the p (1-100) and f "
         "(101-127) layers. Dropped: rr2 and rr3 (neighbouring samples repitched), the legato master (CC64 "
         "condition, legato trigger), the noise regions, the CC vibrato and the pedal switches. tune= is kept per "
         "zone as fineTune; the velocity-dependent attack (CC131) becomes velocity modulators.",
         keep=_drop_mtg_noise, expect=_samples_match(r"/ten_[pf]_\d\d\.flac$", 66)),
    Bank("mtg-alto", "mtg-solo-sax", "MTG Solo Saxophones/MTG Alto Sax (NL).sfz", "mtg-alto",
         "CC BY 4.0 (SFZ mapping by kinwie) and CC BY 3.0 (samples by MTG, Universitat Pompeu Fabra, via Freesound)",
         "As mtg-tenor, on the alto notes.",
         keep=_drop_mtg_noise, expect=_samples_match(r"/alt_[pf]_\d\d\.flac$", 66)),
    Bank("jrhodes3d-mono", "jrhodes3d", "jRhodes3d-mono-no-xfade.sfz", "jrhodes3d-mono",
         "CC BY-NC 4.0 (samples by Jeff Learman): non-commercial, never committed or shared",
         "The root-level jRhodes3d-mono-no-xfade.sfz: its 75 regions map the same 65 samples as "
         "jRhodes3d-mono/_jRhodes3d-mono-no-xfade-flac.sfz, but cover keys 21-108 (A0-C8) where the folder's copy "
         "stops at 24-101 (C1-F7). Its tremolo (lfo01, CC1) and autopan (lfo02, CC95) default to depth 0 and are "
         "dropped; the half-pedal opcodes sit in <control>, where no region reads them.",
         expect=_samples_match(r"/A_\d{3}__[A-G]b?\d_[1-5]\.flac$", 75)),
    Bank("vcsl-upright-knight", "vcsl", "Chordophones/Zithers/Upright Piano, Knight.sfz", "vcsl-upright-knight",
         "CC0 1.0: VCSL Upright Piano, Knight (Simon Dalzell, Ivy Audio)",
         "The sustain samples, both velocity layers (0-83, 84-127), stereo. Dropped: the release-trigger group "
         "and the CC64 pedal-noise groups (their samples were not fetched).",
         expect=_samples_match(r"/Sustains/Player_vl[12]_rr1_", 90)),
    Bank("vsco2-contrabass-pizz", "vsco2-ce", "ContrabassPizz.sfz", "vsco2-cbass-pizz",
         "CC0 1.0: VSCO 2 Community Edition, Versilian Studios / Sam Gossner and Ivy Audio / Simon Dalzell",
         "Random value 0 selects the first lorand/hirand group (0-0.5), whose regions are the rr1 samples. "
         "Stereo. The preset name is shortened to fit SoundFont's 20-byte field.",
         expect=_samples_match(r"_rr1\.wav$", 20)),
    Bank("gsullivan-wurlitzer", "gsullivan-epianos", "Wurlitzer EP200/Wurlitzer EP200.sfz", "gsullivan-wurlitzer",
         "CC BY 3.0: Wurlitzer EP200 samples by Greg Sullivan, SFZ mapping by kinwie",
         "All four layers (pp, mp, f, ff). The key-scaled hold and decay (CC133) become keynumToVolEnvHold and "
         "keynumToVolEnvDecay, fitted per zone; note_polyphony=1 is FluidSynth's own same-key release.",
         expect=_samples_match(r"/Samples/[a-g]b?\d(pp|mp|f|ff)\.flac$", 48)),
)}


@dataclass
class Conversion:
    data: bytes
    report: dict = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def _sample_names(rels: list[str], stereo: set[str]) -> dict[str, str]:
    """A unique ASCII name for each bank-relative sample path: its file stem, at most 19 characters with the
    L/R suffix of a stereo pair. When stems are too long, the prefix they all share is dropped first
    (BKCtbss_Pizz_A#0_v1_rr1 -> A#0_v1_rr1); a name still too long is cut and numbered (~2, ~3, ...)."""
    stems = {rel: re.sub(r"[^A-Za-z0-9_.#+-]", "_", Path(rel).stem) or "sample" for rel in rels}
    room = {rel: 18 if rel in stereo else 19 for rel in rels}
    if any(len(stems[r]) > room[r] for r in rels) and len(rels) > 1:
        values = sorted(stems.values())
        common = 0
        for a, b in zip(values[0], values[-1]):
            if a != b:
                break
            common += 1
        cut = max((i + 1 for i in range(common) if values[0][i] in "_-. "), default=0)
        if cut:
            stems = {r: s[cut:] for r, s in stems.items()}
    out, used = {}, set()
    for rel in sorted(rels):
        stem, limit = stems[rel], room[rel]
        name, k = stem[:limit], 1
        while name in used:
            k += 1
            tail = f"~{k}"
            name = stem[:limit - len(tail)] + tail
        used.add(name)
        out[rel] = name
    return out


def _with_guard(pcm: np.ndarray, loop: tuple[int, int]) -> np.ndarray:
    """Append points after a loop that ends within GUARD points of the sample end, continuing the loop (the
    point after the loop's last one is the loop's first), so a looped sample keeps GUARD points after its loop."""
    start, end = loop
    need = GUARD - (pcm.shape[0] - end)
    if need <= 0:
        return pcm
    have = pcm.shape[0] - end
    extra = np.array([pcm[start + (have + i) % (end - start)] for i in range(need)], dtype=pcm.dtype)
    return np.concatenate([pcm, extra])


def convert_sfz(path: Path | str, *, title: str, preset: str, root: Path | None = None,
                keep=None, expect=None, copyright: str | None = None, comment: str | None = None) -> Conversion:
    """Convert one SFZ program to a one-preset SF2 (bank 0, program 0). `root` is the folder that sample and
    SFZ paths are reported relative to (default: the SFZ's own folder)."""
    path = Path(path)
    root = Path(root) if root is not None else path.parent
    parsed = parse_sfz(path)
    kept, dropped = select(parsed, keep=keep)
    if not kept:
        raise SfzError(f"{path.name}: no region is kept")
    if expect is not None:
        problems = expect(kept)
        if problems:
            raise SfzError(f"{path.name}: the reduction is not what this bank expects: " + "; ".join(problems))

    def rel(p: Path) -> str:
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            return p.name

    # ---- sources: decoded once each, in sorted bank-relative order
    paths = sorted({r.sample for r in kept}, key=rel)
    sources = {p: sf2.read_source(p) for p in paths}
    first_loop: dict[Path, tuple[int, int]] = {}
    for r in sorted(kept, key=lambda r: r.index):
        if r.loop is not None and r.sample not in first_loop:
            first_loop[r.sample] = r.loop
    header_loop: dict[Path, tuple[int, int] | None] = {}
    for p in paths:
        src = sources[p]
        loop = first_loop.get(p) or (src.smpl_loops[0] if src.smpl_loops else None)
        if loop is not None and not (0 < loop[0] < loop[1] <= src.frames):
            raise sf2.SF2Error(f"{rel(p)}: loop {loop} is outside its {src.frames} frames")
        header_loop[p] = loop
    names = _sample_names([rel(p) for p in paths], {rel(p) for p in paths if len(sources[p].channels) == 2})

    # ---- zones
    specs: list[ZoneSpec] = []
    roots: dict[Path, int] = {}
    for r in sorted(kept, key=lambda r: (r.lokey, r.lovel, r.hikey, r.hivel, r.index)):
        src = sources[r.sample]
        loop = r.loop if r.loop is not None else (src.smpl_loops[0] if src.smpl_loops else None)
        root_key = r.keycenter if r.keycenter is not None else (src.smpl_root if src.smpl_root is not None else 60)
        roots.setdefault(r.sample, root_key)
        guard = 0 if header_loop[r.sample] is None else max(0, GUARD - (src.frames - header_loop[r.sample][1]))
        specs.append(zone_spec(r, frames=src.frames, stored=src.frames + guard, loop=loop,
                               header_loop=header_loop[r.sample], root=root_key, curves=parsed.curves))
    shift = bank_shift(specs)

    # ---- samples (mono, or linked left/right) and instrument zones
    samples: list[sf2.Sample] = []
    sample_ids: dict[Path, tuple[str, ...]] = {}
    for p in paths:
        src = sources[p]
        loop = header_loop[p]
        chans = [(_with_guard(c, loop) if loop is not None else c) for c in src.channels]
        base = names[rel(p)]
        if len(chans) == 1:
            samples.append(sf2.Sample(base, chans[0], src.rate, src.bits, roots[p], 0, loop))
            sample_ids[p] = (base,)
        elif len(chans) == 2:
            left, right = base + "L", base + "R"
            samples.append(sf2.Sample(left, chans[0], src.rate, src.bits, roots[p], 0, loop, "left", right))
            samples.append(sf2.Sample(right, chans[1], src.rate, src.bits, roots[p], 0, loop, "right", left))
            sample_ids[p] = (left, right)
        else:
            raise sf2.SF2Error(f"{rel(p)} has {len(chans)} channels; only mono and stereo map to SF2")
    zones = []
    for s in specs:
        ids = sample_ids[s.region.sample]
        try:
            att = attenuation_generator(s.level_cb + shift)
        except sf2.SF2Error as err:
            raise sf2.SF2Error(f"{s.region.where}: {err}") from err
        for i, sid in enumerate(ids):
            g = dict(s.gens)
            if att:
                g[sf2.GEN["initialAttenuation"]] = att
            pan = round(5 * s.region.pan)
            if len(ids) == 2:
                pan = max(-500, min(500, (-500 if i == 0 else 500) + pan))
            if pan:
                g[sf2.GEN["pan"]] = pan
            zones.append(sf2.Zone(gens=tuple(g.items()), mods=tuple(s.mods), sample=sid))
    instrument = sf2.Instrument(preset, tuple(zones))
    data = sf2.sf2_bytes(name=title, presets=[sf2.Preset(preset, 0, 0, (sf2.Zone(instrument=preset),))],
                         instruments=[instrument], samples=samples, software=SOFTWARE, copyright=copyright,
                         comment=comment or f"{title}: converted from {rel(path)}")
    report = _report(path, rel, parsed, kept, dropped, specs, sources, shift)
    looped = sum(dict(z.gens).get(sf2.GEN["sampleModes"], 0) in (1, 3) for z in zones)
    loop_from = Counter("loop_start/loop_end" if p in first_loop else "smpl chunk" if sources[p].smpl_loops
                        else "none" for p in paths)
    report.update(title=title, preset=preset, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                  sf2_samples=len(samples), zones=len(zones), looped_zones=looped,
                  sample_loops=dict(sorted(loop_from.items())))
    return Conversion(data, report)


def _report(path, rel, parsed, kept: list[Resolved], dropped: list[Dropped], specs: list[ZoneSpec], sources,
            shift: int) -> dict:
    reasons = Counter(_reason_key(d.reason) for d in dropped)
    opcodes: dict[str, str] = {}
    for r in kept:
        for k, why in r.unexpressed.items():
            opcodes.setdefault(k, why)
    control_known = re.compile(r"set_(hd|real)?cc\d+|label_cc\d+|label_key\d+|default_path|octave_offset|"
                               r"note_offset|sw_default|hint_.*|image|define")
    for k in sorted(parsed.control):
        if not control_known.fullmatch(k):
            opcodes.setdefault(f"{k} (in <control>)", "a region opcode in <control>: no region inherits it")
    layers = sorted({(r.lovel, r.hivel) for r in kept})
    notes: dict[str, str] = {}
    time_err: dict[str, float] = {}
    for s in specs:
        notes.update(s.notes)
        for k, e in s.time_errors.items():
            time_err[k] = max(time_err.get(k, 0.0), e)
    vel_err = max(s.velocity.max_error_db for s in specs)
    rates = Counter(src.rate for src in sources.values())
    bits = Counter(src.bits for src in sources.values())
    chans = Counter(len(src.channels) for src in sources.values())
    return {
        "sfz": rel(path),
        "files": [rel(Path(f)) for f in parsed.files],
        "regions": len(parsed.regions),
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_by_reason": dict(sorted(reasons.items())),
        "dropped_opcodes": dict(sorted(opcodes.items())),
        "approximations": dict(sorted(notes.items())),
        "keys": [min(r.lokey for r in kept), max(r.hikey for r in kept)],
        "velocity_layers": [list(x) for x in layers],
        "samples": len(sources),
        "sample_rates": {str(k): v for k, v in sorted(rates.items())},
        "sample_bits": {str(k): v for k, v in sorted(bits.items())},
        "channels": {str(k): v for k, v in sorted(chans.items())},
        "velocity_curve_max_error_db": round(vel_err, 4),
        "velocity_fits": sorted({(s.region.veltrack, s.region.velcurve, s.velocity.amounts, s.velocity.constant,
                                  round(s.velocity.max_error_db, 4)) for s in specs}, key=repr),
        "time_fit_max_errors": {k: round(v, 5) for k, v in sorted(time_err.items())},
        "attenuation_shift_cb": shift,
        "attenuation_range_cb": [round(min(s.level_cb + shift for s in specs), 1),
                                 round(max(s.level_cb + shift for s in specs), 1)],
    }


def _reason_key(reason: str) -> str:
    """Group drop reasons: random-layer and round-robin reasons by kind, so counts stay readable."""
    if reason.startswith("random layer"):
        return "random layer that does not cover random value 0"
    if reason.startswith("round robin"):
        return "round robin seq_position other than 1"
    return reason


def convert_bank(bank_id: str, big: Path | str) -> Conversion:
    """Convert one of BANKS from an assets/big folder."""
    if bank_id not in BANKS:
        raise KeyError(f"unknown bank {bank_id!r}; banks: {', '.join(BANKS)}")
    b = BANKS[bank_id]
    folder = Path(big) / b.folder
    sfz_path = folder / b.sfz
    if not sfz_path.is_file():
        raise FileNotFoundError(f"{sfz_path} is missing (assets/big holds the fetched banks)")
    conv = convert_sfz(sfz_path, title=b.id, preset=b.preset, root=folder, keep=b.keep, expect=b.expect,
                       copyright=b.copyright, comment=f"{b.id}: {b.sfz} reduced by lofi_rl.convert (docs/CONVERTER.md)")
    conv.report.update(bank=b.id, folder=b.folder, why=b.why)
    return conv
