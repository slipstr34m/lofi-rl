"""The SFZ -> SF2 converter on synthetic banks (docs/CONVERTER.md): parser semantics, ARIA evaluation and region
selection, the writer read back by an independent RIFF reader (tests/helpers_sf2.py), what the writer refuses,
sm24 losslessness, stereo links, the loops FLAC sources carry, byte determinism across runs, processes and hash
seeds, the velocity and envelope-time fits, and palette.sf2_presets on the output. Every SFZ, WAV and FLAC is
generated in tmp_path; nothing downloaded is read. Three tests render through FluidSynth and skip without it."""
from __future__ import annotations

import hashlib
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import mido
import numpy as np
import pytest

from helpers_render import needs_fluidsynth
from helpers_sf2 import name, read_sf2, signed, write_flac, zones
from lofi_rl.convert import mapping, sf2
from lofi_rl.convert.banks import BANKS, convert_sfz
from lofi_rl.convert.sfz import note_number, parse_sfz, select, velocity_gain
from lofi_rl.palette import sf2_presets
from lofi_rl.render import fluid
from lofi_rl.render.wavio import read_float

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------- WAV fixtures
def write_wav(path: Path, frames: np.ndarray, rate: int, bits: int, loop: tuple[int, int] | None = None,
              unity: int = 60) -> Path:
    """PCM WAV from integer frames (n,) or (n, channels); loop = (start, end exclusive) goes in a smpl chunk."""
    x = np.asarray(frames, dtype=np.int64)
    x = x[:, None] if x.ndim == 1 else x
    ch = x.shape[1]
    if bits == 16:
        data = x.astype("<i2").tobytes()
    else:
        raw = (x & 0xFFFFFF).astype("<u4").tobytes()
        data = b"".join(raw[i:i + 3] for i in range(0, len(raw), 4))
    body = b"fmt " + struct.pack("<IHHIIHH", 16, 1, ch, rate, rate * ch * bits // 8, ch * bits // 8, bits)
    body += b"data" + struct.pack("<I", len(data)) + data + (b"\0" if len(data) & 1 else b"")
    if loop is not None:
        smpl = struct.pack("<9I", 0, 0, 1_000_000_000 // rate, unity, 0, 0, 0, 1, 0)
        smpl += struct.pack("<6I", 0, 0, loop[0], loop[1] - 1, 0, 0)
        body += b"smpl" + struct.pack("<I", len(smpl)) + smpl
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
    return path


def noise(n: int, bits: int, channels: int = 1, seed: int = 1) -> np.ndarray:
    top = 1 << (bits - 1)
    x = np.random.default_rng(seed).integers(-top, top, size=(n, channels))
    return x[:, 0] if channels == 1 else x


# ----------------------------------------------------------------------------------------- parser
def sfz(folder: Path, text: str, file: str = "t.sfz", samples: tuple[str, ...] = ("x.wav",)) -> Path:
    for s in samples:
        if not (folder / s).exists():
            write_wav(folder / s, noise(64, 16), 44100, 16)
    p = folder / file
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_headers_inherit_global_master_group_region_and_reset_below(tmp_path):
    p = sfz(tmp_path, "<global> volume=-1 pan=10\n<master> tune=5\n<group> lovel=10\n"
                      "<region> sample=x.wav\n<region> sample=x.wav pan=0\n"
                      "<master> transpose=2\n<region> sample=x.wav\n"
                      "<global> volume=-3\n<region> sample=x.wav\n")
    r = [dict(x.opcodes) for x in parse_sfz(p).regions]
    assert r[0] == {"volume": "-1", "pan": "10", "tune": "5", "lovel": "10", "sample": "x.wav"}
    assert r[1]["pan"] == "0" and r[1]["lovel"] == "10"
    assert r[2] == {"volume": "-1", "pan": "10", "transpose": "2", "sample": "x.wav"}, "a new <master> clears group"
    assert r[3] == {"volume": "-3", "sample": "x.wav"}, "a new <global> clears master and group"


def test_define_substitutes_everywhere_and_the_longest_name_wins(tmp_path):
    p = sfz(tmp_path, "#define $VEL 30\n#define $VELTRACK 99\n#define $EXT wav\n"
                      "<control> set_cc$VELTRACK=127 label_cc$VEL=Thirty\n"
                      "<region> sample=x.$EXT lovel=$VEL amp_veltrack_oncc$VELTRACK=50\n")
    s = parse_sfz(p)
    assert dict(s.control) == {"set_cc99": "127", "label_cc30": "Thirty"}
    assert dict(s.regions[0].opcodes) == {"sample": "x.wav", "lovel": "30", "amp_veltrack_oncc99": "50"}


def test_include_resolves_beside_the_including_file_then_the_root(tmp_path):
    # Data/a.txt includes b.txt (beside it) and Data/c.txt (resolvable only from the root, as MTG's files do)
    (tmp_path / "Data").mkdir()
    (tmp_path / "Data" / "a.txt").write_text('<group> lovel=1\n#include "b.txt"\n#include "Data/c.txt"\n')
    (tmp_path / "Data" / "b.txt").write_text("<region> sample=x.wav key=60\n")
    (tmp_path / "Data" / "c.txt").write_text("<region> sample=x.wav key=62\n")
    p = sfz(tmp_path, '#include "Data/a.txt"\n')
    s = parse_sfz(p)
    assert [r.opcodes["lokey"] for r in s.regions] == ["60", "62"]
    assert [f.name for f in s.files] == ["t.sfz", "a.txt", "b.txt", "c.txt"]


def test_comments_default_path_backslashes_and_spaces_in_sample_paths(tmp_path):
    write_wav(tmp_path / "My Samples" / "Sub dir" / "a b.wav", noise(64, 16), 44100, 16)
    p = sfz(tmp_path, "/* a block\ncomment */ <control> default_path=My Samples\\Sub dir\\   // trailing\n"
                      "<region> sample=a b.wav lokey=c4 hikey=c#4 // note names\n", samples=())
    kept, dropped = select(parse_sfz(p))
    assert not dropped and kept[0].sample == tmp_path / "My Samples" / "Sub dir" / "a b.wav"
    assert (kept[0].lokey, kept[0].hikey) == (60, 61)


def test_note_names_and_key():
    assert [note_number(n) for n in ("c4", "C4", "c#4", "db4", "a0", "c-1", "g9", "60")] == [60, 60, 61, 61, 21, 0,
                                                                                            127, 60]
    assert note_number("c4", octave_offset=1, note_offset=-2) == 70


def test_cc_defaults_curves_amplitude_crossfades_and_conditions(tmp_path):
    p = sfz(tmp_path, "<control> set_cc7=100 set_hdcc11=0.5\n<curve> curve_index=8 v000=1 v100=0 v127=0\n"
                      "<region> sample=x.wav amplitude_oncc7=100 pan=-100 pan_oncc11=200 ampeg_release=0.1 "
                      "ampeg_release_oncc11=1 ampeg_release_curvecc11=4 ampeg_attack_oncc72=1 "
                      "tune_oncc11=10 tune_curvecc11=8 xfout_locc1=0\n"
                      "<region> sample=x.wav xfin_locc1=0 xfin_hicc1=127\n"     # fades in from CC1 = 0: silent
                      "<region> sample=x.wav locc64=64\n"                       # CC64 = 0 fails locc64=64
                      "<region> sample=x.wav hicc11=60\n")                      # CC11 = 63.5 fails hicc11=60
    kept, dropped = select(parse_sfz(p))
    assert len(kept) == 1
    r = kept[0]
    assert r.volume_db == pytest.approx(20 * math.log10(100 / 127))      # amplitude multiplies by depth x CC
    assert r.pan == pytest.approx(0.0)                                     # -100 + 200 x 0.5
    assert r.env["release"].base == pytest.approx(0.1 + 0.25)             # curve 4 is x²
    assert r.env["attack"].base == 0.0                                     # CC72 has no set_cc: 0
    assert r.tune == pytest.approx(10 * (1 - 63.5 / 100))                 # the file's curve 8 at 63.5
    assert [d.reason.split(" (")[0] for d in dropped] == [
        "silent at the controller defaults", "CC64 = 0 is outside locc/hicc 64..127",
        "CC11 = 63.5 is outside locc/hicc 0..60"]


def test_selection_round_robin_random_triggers_generated_and_cc_triggered(tmp_path):
    p = sfz(tmp_path, "<group> seq_length=3\n<region> sample=x.wav seq_position=1 key=60\n"
                      "<region> sample=x.wav seq_position=2 key=60\n"
                      "<group>\n<region> sample=x.wav lorand=0 hirand=0.5 key=61\n"
                      "<region> sample=x.wav lorand=0.5 hirand=1 key=61\n"
                      "<region> sample=x.wav trigger=release key=62\n<region> sample=x.wav trigger=legato key=62\n"
                      "<region> sample=x.wav trigger=first key=63\n<region> sample=*noise key=64\n"
                      "<region> sample=x.wav on_locc64=1 on_hicc64=127\n")
    kept, dropped = select(parse_sfz(p))
    assert [r.lokey for r in kept] == [60, 61, 63]
    assert [d.reason.split(" (")[0].split(" lorand")[0] for d in dropped] == [
        "round robin seq_position 2", "random layer", "release trigger", "legato trigger",
        "generated sample *noise", "CC-triggered"]


def test_keyswitches_keep_the_default(tmp_path):
    p = sfz(tmp_path, "<control> sw_default=c1\n<global> sw_lokey=c1 sw_hikey=d1\n"
                      "<region> sample=x.wav sw_last=c1 key=60\n<region> sample=x.wav sw_last=d1 key=61\n"
                      "<region> sample=x.wav sw_down=d1 key=62\n<region> sample=x.wav key=63\n")
    kept, dropped = select(parse_sfz(p))
    assert [r.lokey for r in kept] == [60, 63]
    assert len(dropped) == 2


def test_dropped_opcodes_are_listed_with_their_value_at_the_defaults(tmp_path):
    p = sfz(tmp_path, "<control> set_cc1=127\n<region> sample=x.wav lfo01_pitch_oncc1=50 lfo02_volume=0 "
                      "lfo02_volume_oncc2=6 eq1_gain=3 group=1 off_by=1 bend_up=500 region_label=r1\n")
    kept, _ = select(parse_sfz(p))
    un = kept[0].unexpressed
    assert set(un) == {"lfo01_pitch_oncc1", "lfo02_volume", "lfo02_volume_oncc2", "eq1_gain", "group", "off_by",
                       "bend_up"}, "labels are metadata, not dropped opcodes"
    assert un["lfo01_pitch_oncc1"].endswith("lfo01_pitch at the controller defaults = 50")
    assert un["lfo02_volume"].endswith("lfo02_volume at the controller defaults = 0")
    assert un["eq1_gain"].endswith("eq1_gain at the controller defaults = 3")


# -------------------------------------------------------------------------------- writer round trip
def _bank(folder: Path) -> tuple[Path, dict[str, np.ndarray]]:
    src = {
        "a16.wav": (noise(1000, 16, 1, 1), 44100, 16, None),
        "b24.wav": (noise(1200, 24, 1, 2), 48000, 24, (200, 1200)),      # the loop ends on the last frame
        "c24s.wav": (noise(900, 24, 2, 3), 44100, 24, None),
        "d16s.wav": (noise(800, 16, 2, 4), 48000, 16, None),
    }
    for f, (x, rate, bits, loop) in src.items():
        write_wav(folder / "samples" / f, x, rate, bits, loop)
    p = folder / "bank.sfz"
    p.write_text("<control> default_path=samples\\\n<global> ampeg_release=0.5\n"
                 "<group> lovel=1 hivel=63\n"
                 "<region> sample=a16.wav lokey=36 hikey=47 pitch_keycenter=40 tune=-17 volume=-6\n"
                 "<region> sample=b24.wav lokey=48 hikey=59 pitch_keycenter=52 transpose=1 tune=25 pan=-50\n"
                 "<group> lovel=64\n"
                 "<region> sample=c24s.wav lokey=36 hikey=47 pitch_keycenter=41 offset=10 end=849\n"
                 "<region> sample=d16s.wav lokey=48 hikey=59 pitch_keycenter=53 loop_mode=loop_continuous "
                 "loop_start=100 loop_end=699\n")
    return p, {k: v[0] for k, v in src.items()}


def _gens(z: dict) -> dict[int, int]:
    return {g: v for g, v in z["gens"]}


def test_the_writer_reads_back_through_an_independent_riff_reader(tmp_path):
    p, src = _bank(tmp_path)
    data = convert_sfz(p, title="synthetic bank", preset="synth").data
    sf = read_sf2(data)
    assert sf["order"] == [b"INFO", b"sdta", b"pdta"]
    info = sf["info"]
    assert list(info) == ["ifil", "isng", "INAM", "ICMT", "ISFT"] and "ICRD" not in info, "no date is written"
    assert struct.unpack("<HH", info["ifil"]) == (2, 4)
    assert info["isng"] == b"EMU8000\0" and info["INAM"].rstrip(b"\0") == b"synthetic bank"
    assert all(len(v) % 2 == 0 and v.endswith(b"\0") for k, v in info.items() if k != "ifil")
    # presets and instruments, with their terminal records
    assert [(name(r[0]), r[1], r[2]) for r in sf["phdr"]] == [("synth", 0, 0), ("EOP", 0, 0)]
    assert sf["phdr"][1][3] == len(sf["pbag"]) - 1
    assert [name(r[0]) for r in sf["inst"]] == ["synth", "EOI"] and sf["inst"][1][1] == len(sf["ibag"]) - 1
    assert sf["pgen"][-1] == (0, 0) and sf["igen"][-1] == (0, 0) and sf["imod"][-1] == (0, 0, 0, 0, 0)
    assert sf["pgen"][:-1] == [(41, 0)], "one preset zone: instrument 0"
    assert name(sf["shdr"][-1][0]) == "EOS" and sf["shdr"][-1][1:] == (0,) * 9
    # samples: every point exactly, 46+ zeros after each, rates, loops, links
    shdr = {name(r[0]): r for r in sf["shdr"][:-1]}
    assert sorted(shdr) == ["a16", "b24", "c24sL", "c24sR", "d16sL", "d16sR"]
    full = sf["smpl"].astype(np.int64) * 256 + sf["sm24"].astype(np.int64)
    ends = sorted((r[1], r[2]) for r in sf["shdr"][:-1])
    for (_, e), (s2, _) in zip(ends, ends[1:] + [(len(full), 0)]):
        assert s2 - e >= 46 and not full[e:s2].any(), "46 zero sample points follow every sample"
    for nm, x in (("a16", src["a16.wav"] * 256), ("b24", src["b24.wav"]), ("c24sL", src["c24s.wav"][:, 0]),
                  ("c24sR", src["c24s.wav"][:, 1]), ("d16sL", src["d16s.wav"][:, 0] * 256),
                  ("d16sR", src["d16s.wav"][:, 1] * 256)):
        r = shdr[nm]
        assert np.array_equal(full[r[1]:r[1] + len(x)], x), f"{nm} is stored point for point"
    rate = {nm: r[5] for nm, r in shdr.items()}
    assert rate == {"a16": 44100, "b24": 48000, "c24sL": 44100, "c24sR": 44100, "d16sL": 48000, "d16sR": 48000}
    b = shdr["b24"]
    assert (b[3] - b[1], b[4] - b[1], b[2] - b[1]) == (200, 1200, 1208), "smpl loop, then 8 guard points"
    loop = src["b24.wav"][200:1200]
    assert np.array_equal(full[b[1] + 1200:b[2]], loop[:8]), "the guard points continue the loop"
    d = shdr["d16sL"]
    assert (d[3] - d[1], d[4] - d[1]) == (100, 700), "SFZ loop_end is inclusive; SF2's end is exclusive"
    # zones: ranges first, sampleID last, generators as mapped
    zs = zones(sf)
    assert len(zs) == 6
    for z in zs:
        assert [g for g, _ in z["gens"][:2]] == [43, 44] and z["gens"][-1][0] == 53
    by_sample = {name(sf["shdr"][_gens(z)[53]][0]): _gens(z) for z in zs}
    a, bz, cl, cr, dl = (by_sample[k] for k in ("a16", "b24", "c24sL", "c24sR", "d16sL"))
    assert (a[43], a[44]) == (36 | 47 << 8, 1 | 63 << 8) and (cl[44]) == (64 | 127 << 8)
    assert a[58] == 40 and signed(a[52]) == -17 and 51 not in a
    assert bz[51] == 1 and bz[52] == 25 and signed(bz[17]) == -250
    assert signed(a[38]) == round(1200 * math.log2(0.5)) == -1200
    assert a[48] == round(60 / 0.4) and 48 not in bz, "volume -6 dB -> 60 cB, written / 0.4 for FluidSynth"
    assert cl[0] == 10 and signed(cl[1]) == 850 - 900 and signed(cl[17]) == -500 and cr[17] == 500
    assert dl[54] == 1 and 54 not in a and bz[54] == 1, "a smpl loop plays (loop_continuous) unless told not to"
    assert all(not z["mods"] for z in zs), "100 % veltrack is FluidSynth's default modulator: none written"
    assert sf2_presets(_write(tmp_path / "bank.sf2", data)) == frozenset({(0, 0)})


def test_loop_mode_no_loop_and_pitch_keycenter_sample_read_the_smpl_chunk(tmp_path):
    write_wav(tmp_path / "x.wav", noise(400, 16), 44100, 16, loop=(100, 300), unity=57)
    p = sfz(tmp_path, "<region> sample=x.wav lokey=40 hikey=50 pitch_keycenter=sample loop_mode=no_loop\n"
                      "<region> sample=x.wav lokey=51 hikey=60 pitch_keycenter=sample\n", samples=())
    sf = read_sf2(convert_sfz(p, title="t", preset="t").data)
    a, b = (_gens(z) for z in zones(sf))
    assert a[58] == b[58] == 57 and sf["shdr"][0][6] == 57, "the smpl chunk's unity note"
    assert 54 not in a and b[54] == 1, "no_loop plays the looped sample straight; the default loops it"
    r = sf["shdr"][0]
    assert (r[3] - r[1], r[4] - r[1]) == (100, 300)


def test_flac_sources_read_the_smpl_loop_their_riff_blocks_carry(tmp_path):
    # `flac --keep-foreign-metadata` keeps a WAV's smpl chunk in an APPLICATION 'riff' block (the Wurlitzer's and
    # MTG's FLACs carry their loops this way): a FLAC source has the same loop and unity note as its WAV would
    x = noise(3000, 24, seed=5)
    src = sf2.read_source(write_flac(tmp_path / "s.flac", x, 48000, 24, loop=(1000, 2900), unity=57))
    assert np.array_equal(src.channels[0], x) and (src.rate, src.bits) == (48000, 24), "still lossless"
    assert src.smpl_loops == ((1000, 2900),) and src.smpl_root == 57
    plain = sf2.read_source(write_flac(tmp_path / "plain.flac", x, 48000, 24))      # riff blocks, no smpl
    assert plain.smpl_loops == () and plain.smpl_root is None
    # SFZ plays the sample's own loop unless loop_mode says otherwise, as for a WAV
    p = sfz(tmp_path, "<region> sample=s.flac lokey=40 hikey=50 pitch_keycenter=sample\n"
                      "<region> sample=s.flac lokey=51 hikey=60 loop_mode=no_loop\n", samples=())
    conv = convert_sfz(p, title="t", preset="t")
    sf = read_sf2(conv.data)
    a, b = (_gens(z) for z in zones(sf))
    assert a[54] == 1 and 54 not in b and a[58] == 57 and sf["shdr"][0][6] == 57
    r = sf["shdr"][0]
    assert (r[3] - r[1], r[4] - r[1]) == (1000, 2900)
    assert (conv.report["looped_zones"], conv.report["sample_loops"]) == (1, {"smpl chunk": 1}), "the CLI shows it"


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_sm24_is_lossless_and_absent_without_24_bit_sources(tmp_path):
    x = np.array([-(1 << 23), (1 << 23) - 1, -1, 0, 1, 255, -256, 0x123456] * 16, dtype=np.int64)
    write_wav(tmp_path / "x.wav", x, 44100, 24)
    data = convert_sfz(sfz(tmp_path, "<region> sample=x.wav\n", samples=()), title="t", preset="t").data
    sf = read_sf2(data)
    assert len(sf["sm24"]) == len(sf["smpl"]), "one low byte per point; the even point count needs no pad"
    r = sf["shdr"][0]
    back = sf["smpl"][r[1]:r[2]].astype(np.int64) * 256 + sf["sm24"][r[1]:r[2]]
    assert np.array_equal(back, x)
    write_wav(tmp_path / "y" / "x.wav", noise(100, 16), 44100, 16)
    data16 = convert_sfz(sfz(tmp_path / "y", "<region> sample=x.wav\n", samples=()), title="t", preset="t").data
    assert "sm24" not in str(read_sf2(data16)["chunks"][b"sdta"])


def test_stereo_sources_become_linked_left_right_samples(tmp_path):
    p, _ = _bank(tmp_path)
    sf = read_sf2(convert_sfz(p, title="t", preset="t").data)
    names = [name(r[0]) for r in sf["shdr"][:-1]]
    for left in ("c24sL", "d16sL"):
        li, ri = names.index(left), names.index(left[:-1] + "R")
        lrec, rrec = sf["shdr"][li], sf["shdr"][ri]
        assert (lrec[9], lrec[8]) == (4, ri) and (rrec[9], rrec[8]) == (2, li), "left 4 -> right, right 2 -> left"
        assert lrec[2] - lrec[1] == rrec[2] - rrec[1] and lrec[5] == rrec[5]
    assert [sf["shdr"][names.index(n)][9] for n in ("a16", "b24")] == [1, 1], "mono samples"


# ------------------------------------------------------------------------- what the writer refuses
G = sf2.GEN
VEL = sf2.src_oper(sf2.VELOCITY, negative=True)


def _one_zone(*, preset_gens=(), inst_gens=(), mods=(), loop=(40, 160), n=200) -> bytes:
    """sf2_bytes on one sample, one instrument zone and one preset zone."""
    pcm = np.round(1e6 * np.sin(2 * np.pi * np.arange(n) / 40)).astype(np.int32)
    s = sf2.Sample("s", pcm, 44100, 24, 60, 0, loop)
    inst = sf2.Instrument("i", (sf2.Zone(gens=tuple(inst_gens), mods=tuple(mods), sample="s"),))
    pre = sf2.Preset("p", 0, 0, (sf2.Zone(gens=tuple(preset_gens), instrument="i"),))
    return sf2.sf2_bytes(name="t", presets=[pre], instruments=[inst], samples=[s])


# Each of these went through sf2_bytes, and FluidSynth 2.6.1 then discarded, zeroed or warned about it (or the
# spec forbids it): SoundFont 2.04 sections 7.10 (samples and loops), 8.1.2/8.1.3 (generators), 8.2.1 and 8.3
# (modulator sources and transforms).
_FORBIDDEN = {
    "preset zone: sampleModes": dict(preset_gens=[(G["sampleModes"], 1)]),
    "preset zone: overridingRootKey": dict(preset_gens=[(G["overridingRootKey"], 72)]),
    "preset zone: startAddrsOffset": dict(preset_gens=[(G["startAddrsOffset"], 100)]),
    "preset zone: endloopAddrsCoarseOffset": dict(preset_gens=[(G["endloopAddrsCoarseOffset"], 1)]),
    "preset zone: keynum": dict(preset_gens=[(G["keynum"], 60)]),
    "preset zone: exclusiveClass": dict(preset_gens=[(G["exclusiveClass"], 1)]),
    "pan 900": dict(inst_gens=[(G["pan"], 900)]),
    "sustainVolEnv 2000": dict(inst_gens=[(G["sustainVolEnv"], 2000)]),
    "initialAttenuation -1": dict(inst_gens=[(G["initialAttenuation"], -1)]),
    "fineTune 150": dict(inst_gens=[(G["fineTune"], 150)]),
    "coarseTune -121": dict(inst_gens=[(G["coarseTune"], -121)]),
    "delayVolEnv 6000": dict(inst_gens=[(G["delayVolEnv"], 6000)]),
    "holdVolEnv 5001": dict(inst_gens=[(G["holdVolEnv"], 5001)]),
    "attackVolEnv -12001": dict(inst_gens=[(G["attackVolEnv"], -12001)]),
    "keynumToVolEnvDecay 1201": dict(inst_gens=[(G["keynumToVolEnvDecay"], 1201)]),
    "sampleModes 2": dict(inst_gens=[(G["sampleModes"], 2)]),
    "source: general controller 5": dict(mods=[sf2.Mod(sf2.src_oper(5), 48, 100)]),
    "source: CC 6 (data entry)": dict(mods=[sf2.Mod(sf2.src_oper(6, cc=True), 48, 100)]),
    "source: CC 120": dict(mods=[sf2.Mod(sf2.src_oper(120, cc=True), 48, 100)]),
    "source: link (127)": dict(mods=[sf2.Mod(sf2.src_oper(127), 48, 100)]),
    "source: no controller": dict(mods=[sf2.Mod(sf2.src_oper(0), 48, 100)]),
    "source: curve type 5": dict(mods=[sf2.Mod((5 << 10) | VEL, 48, 100)]),
    "amount source: general controller 5": dict(mods=[sf2.Mod(VEL, 48, 100, sf2.src_oper(5))]),
    "amount source: CC 99 (NRPN)": dict(mods=[sf2.Mod(VEL, 48, 100, sf2.src_oper(99, cc=True))]),
    "transform 1": dict(mods=[sf2.Mod(VEL, 48, 100, 0, 1)]),
    "transform 3": dict(mods=[sf2.Mod(VEL, 48, 100, 0, 3)]),
    "destination 14 (unused)": dict(mods=[sf2.Mod(VEL, 14, 100)]),
    "destination 99": dict(mods=[sf2.Mod(VEL, 99, 100)]),
    "destination sampleID": dict(mods=[sf2.Mod(VEL, G["sampleID"], 100)]),
    "destination: a linked modulator": dict(mods=[sf2.Mod(VEL, 0x8000, 100)]),
    "loop starting at point 1": dict(loop=(1, 160)),
    "loop with 7 points before it": dict(loop=(7, 160)),
    "loop ending on the sample's end": dict(loop=(40, 200)),
    "loop with 7 points after it": dict(loop=(40, 193)),
    "loop of 31 points": dict(loop=(40, 71)),
    "20-point sample": dict(n=20, loop=None),
    "47-point sample": dict(n=47, loop=None),
}


@pytest.mark.parametrize("case", list(_FORBIDDEN))
def test_the_writer_refuses_what_soundfont_2_04_or_fluidsynth_would_drop(case):
    _one_zone()                                         # the same file without the case writes
    with pytest.raises(sf2.SF2Error):
        _one_zone(**_FORBIDDEN[case])


def test_the_writer_accepts_the_edges_of_every_range():
    _one_zone(inst_gens=[(G["pan"], -500), (G["sustainVolEnv"], 1440), (G["initialAttenuation"], 1440),
                         (G["fineTune"], -99), (G["coarseTune"], 120), (G["delayVolEnv"], 5000),
                         (G["holdVolEnv"], -12000), (G["attackVolEnv"], 8000), (G["keynumToVolEnvHold"], -1200),
                         (G["sampleModes"], 3), (G["overridingRootKey"], 127), (G["scaleTuning"], 0)],
              preset_gens=[(G["pan"], 250), (G["coarseTune"], -12)],
              mods=[sf2.Mod(VEL, 48, 960), sf2.Mod(sf2.src_oper(sf2.KEY, curve="switch"), 48, 10, VEL, 2),
                    sf2.Mod(sf2.src_oper(7, cc=True, curve="convex"), 48, -10),
                    sf2.Mod(sf2.src_oper(16), 52, 1, sf2.src_oper(14, bipolar=True))])
    _one_zone(n=48, loop=(8, 40))                       # 8 points, a 32-point loop, 8 points
    _one_zone(n=48, loop=None)


def test_bytes_are_identical_across_runs_processes_and_hash_seeds(tmp_path):
    p, _ = _bank(tmp_path)
    first = convert_sfz(p, title="t", preset="t").data
    assert convert_sfz(p, title="t", preset="t").data == first
    code = ("import hashlib, sys; from lofi_rl.convert.banks import convert_sfz; "
            "print(hashlib.sha256(convert_sfz(sys.argv[1], title='t', preset='t').data).hexdigest())")
    seen = set()
    for seed in ("0", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(ROOT / "src"))
        out = subprocess.run([sys.executable, "-c", code, str(p)], capture_output=True, text=True, env=env,
                             check=True)
        seen.add(out.stdout.strip())
    assert seen == {hashlib.sha256(first).hexdigest()}


# ----------------------------------------------------------------------------------- the fits
def test_fluidsynth_default_velocity_curve_is_sfz_full_veltrack():
    for v in range(1, 128):
        assert 960 * mapping.velocity_basis(v)[0] == pytest.approx(400 * math.log10(127 / v), abs=1e-9)


def test_velocity_fits_are_pinned_for_the_weresax_curves():
    # Weresax's condenser layers: f (64-127) at amp_veltrack=66, p (1-63) at 77 with amp_velcurve_63=1. Pinned, so
    # a machine whose floats differ would show here before it changed a converted bank's bytes.
    forte = mapping.fit_velocity(66.0, (), 64, 127)
    piano = mapping.fit_velocity(77.0, ((63, 1.0),), 0, 63)
    assert (forte.amounts, forte.constant) == ((-237, 159, 41, 5), 0)
    assert (piano.amounts, piano.constant) == ((-982, 250, -59, 976), -47)
    assert forte.max_error_db < 0.03 and piano.max_error_db < 0.07
    for fit, t, curve, vels in ((forte, 66.0, (), range(64, 128)), (piano, 77.0, ((63, 1.0),), range(1, 64))):
        for v in vels:
            target = -200 * math.log10(velocity_gain(t, curve, v))
            assert abs(fit.model(v) - target) <= 10 * fit.max_error_db + 1e-9
    assert mapping.fit_velocity(100.0, (), 0, 127).modulators() == ()
    flat = mapping.fit_velocity(0.0, (), 0, 127)
    assert flat.modulators() == (mapping.Mod(mapping.CONCAVE_NEG, 48, 0),), "the default, overridden at amount 0"


def test_attenuation_generators_compensate_fluidsynths_emu_factor():
    assert mapping.attenuation_generator(60.0) == 150 and mapping.attenuation_generator(0.0) == 0
    with pytest.raises(Exception, match="outside 0..1440"):
        mapping.attenuation_generator(600.0)


def test_key_and_velocity_terms_on_envelope_times(tmp_path):
    # Wurlitzer-style key-scaled hold (CC133) and MTG-style velocity-scaled attack (CC131 through curve 8)
    p = sfz(tmp_path, "<curve> curve_index=8 v000=1 v100=0 v127=0\n"
                      "<region> sample=x.wav lokey=33 hikey=41 ampeg_hold=5 ampeg_holdcc133=-5\n"
                      "<region> sample=x.wav lokey=60 hikey=60 hivel=100 ampeg_attack_oncc131=0.1 "
                      "ampeg_attack_curvecc131=8\n")
    sf = read_sf2(convert_sfz(p, title="t", preset="t").data)
    z = zones(sf)
    hold = _gens(z[0])
    k = hold[39]
    for key in (33, 37, 41):
        want = 5 * (1 - key / 127)
        got = 2 ** ((signed(hold[35]) + k * (60 - key)) / 1200)
        assert got == pytest.approx(want, rel=0.01)
    mods = z[1]["mods"]
    assert {(m[0], m[1], m[3]) for m in mods} == {(mapping.LIN_NEG, 34, 0), (mapping.LIN_NEG, 34, mapping.LIN_NEG)}
    base = signed(_gens(z[1])[34])
    for v in (1, 40, 80, 99):
        lin = (127 - v) / 128
        tc = base + sum(m[2] * (lin if m[3] == 0 else lin * lin) for m in mods)
        got = 2 ** (max(-12000, tc) / 1200)
        assert abs(got - 0.1 * max(0.0, 1 - v / 100)) < 0.008


def _played_seconds(tc: float, lo: int, hi: int) -> float:
    """A volume-envelope time as FluidSynth 2.6.1 plays it: the generator plus its modulators, the sum clamped."""
    return 2 ** (max(lo, min(hi, tc)) / 1200)


def test_velocity_time_fit_keeps_its_base_in_range_and_reports_what_is_written(tmp_path):
    # MTG's p layer: attack 0.1 x max(0, 1 - v/100) s at velocities 0-100, through its curve 8. The SFZ time reaches 0
    # at v = 100, so a free fit puts the base below attackVolEnv's -12000 tc floor (SoundFont 2.04 section 8.1.3).
    p = sfz(tmp_path, "<curve> curve_index=8 v000=1 v100=0 v127=0\n"
                      "<region> sample=x.wav lokey=60 hikey=60 hivel=100 ampeg_attack_oncc131=0.1 "
                      "ampeg_attack_curvecc131=8\n")
    conv = convert_sfz(p, title="t", preset="t")
    (z,) = zones(read_sf2(conv.data))
    base = signed(_gens(z)[34])
    mods = [m for m in z["mods"] if m[1] == 34]
    assert -12000 <= base <= 8000 and len(mods) == 2

    def played(v):
        lin = (127 - v) / 128
        return _played_seconds(base + sum(m[2] * (lin if m[3] == 0 else lin * lin) for m in mods), -12000, 8000)
    written = max(abs(played(v) - 0.1 * max(0.0, 1 - v / 100)) for v in range(1, 101))
    reported = conv.report["time_fit_max_errors"]["ampeg_attack (per velocity, s)"]
    assert reported == pytest.approx(written, abs=1e-5), "the report gives the error of what the file holds"
    assert written < 0.006, f"{written * 1000:.2f} ms"


def test_key_time_fit_keeps_its_base_in_range_and_reports_what_is_written(tmp_path):
    # hold 115 - k s: 15 s at key 100 down to 5 s at key 110. SF2's hold is base + keynum x (60 - k), so a free fit
    # puts the base (key 60) near 12300 tc, beyond holdVolEnv's 5000 tc ceiling (section 8.1.3)
    p = sfz(tmp_path, "<region> sample=x.wav lokey=100 hikey=110 ampeg_hold=115 ampeg_holdcc133=-127\n")
    conv = convert_sfz(p, title="t", preset="t")
    g = _gens(zones(read_sf2(conv.data))[0])
    base, keynum = signed(g[35]), signed(g.get(39, 0))
    assert -12000 <= base <= 5000 and -1200 <= keynum <= 1200
    written = max(abs(_played_seconds(base + keynum * (60 - k), -12000, 5000) / (115 - k) - 1) for k in range(100, 111))
    reported = conv.report["time_fit_max_errors"]["ampeg_hold (per key, relative)"]
    assert reported == pytest.approx(written, abs=1e-5), "the report gives the error of what the file holds"


def test_hold_and_delay_stop_at_their_5000_timecent_ceiling(tmp_path):
    # SoundFont 2.04 section 8.1.3: delayVolEnv and holdVolEnv run -12000..5000 tc (18.0 s at most), decayVolEnv
    # -12000..8000; a longer SFZ time is written at the ceiling and reported as an approximation
    p = sfz(tmp_path, "<region> sample=x.wav ampeg_delay=30 ampeg_hold=30 ampeg_decay=30\n")
    conv = convert_sfz(p, title="t", preset="t")
    g = _gens(zones(read_sf2(conv.data))[0])
    assert (signed(g[33]), signed(g[35]), signed(g[36])) == (5000, 5000, round(1200 * math.log2(30)))
    approx = conv.report["approximations"]
    assert any(k.startswith("ampeg_delay") for k in approx) and any(k.startswith("ampeg_hold") for k in approx)
    assert not any(k.startswith("ampeg_decay") for k in approx)


def test_the_six_banks_are_registered_with_sf2_names():
    assert list(BANKS) == ["weresax-alto", "mtg-tenor", "mtg-alto", "jrhodes3d-mono", "vcsl-upright-knight",
                           "vsco2-contrabass-pizz", "gsullivan-wurlitzer"]
    assert all(0 < len(b.preset) <= 19 and b.preset.isascii() for b in BANKS.values())


# --------------------------------------------------------------------------------------- FluidSynth
def _sine(freq: float, rate: int, seconds: float, periods: int) -> tuple[np.ndarray, tuple[int, int]]:
    n = int(rate * seconds)
    x = np.round(0.5 * (2 ** 23 - 1) * np.sin(2 * np.pi * freq * np.arange(n) / rate)).astype(np.int64)
    length = int(round(rate / freq * periods))
    return x, (n - length - 64, n - 64)


def _peak_hz(x: np.ndarray, sr: int, near: float) -> float:
    n = len(x)
    mag = np.abs(np.fft.rfft(x * np.hanning(n), 8 * n))
    f = np.fft.rfftfreq(8 * n, 1 / sr)
    band = np.nonzero((f > near * 0.97) & (f < near * 1.03))[0]
    k = band[np.argmax(mag[band])]
    a, b, c = np.log(mag[k - 1:k + 2])
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * sr / (8 * n)


_RENDER: dict = {}


def _render(tmp_path_factory) -> tuple[np.ndarray, list]:
    """One synthetic bank and one render, shared by the two FluidSynth tests: a 48 kHz 24-bit sine 30 cents
    sharp (tune=-30 corrects it) on keys 60-80, and a 44.1 kHz one at amp_veltrack=66 on keys 90-100, again at
    volume=-12 on 101-105 and with ampeg_sustain=50 on 106-110 (each played at its own pitch)."""
    if "y" not in _RENDER:
        d = tmp_path_factory.mktemp("convert-fs")
        x48, loop48 = _sine(440 * 2 ** (30 / 1200), 48000, 1.0, 200)
        x44, loop44 = _sine(441.0, 44100, 1.0, 100)
        write_wav(d / "s48.wav", x48, 48000, 24, loop48)
        write_wav(d / "s44.wav", x44, 44100, 24, loop44)
        (d / "b.sfz").write_text("<region> sample=s48.wav lokey=60 hikey=80 pitch_keycenter=69 tune=-30\n"
                                 "<region> sample=s44.wav lokey=90 hikey=100 pitch_keycenter=93 amp_veltrack=66\n"
                                 "<region> sample=s44.wav lokey=101 hikey=105 pitch_keycenter=103 amp_veltrack=66 "
                                 "volume=-12\n<region> sample=s44.wav lokey=106 hikey=110 pitch_keycenter=108 "
                                 "amp_veltrack=66 ampeg_decay=0.05 ampeg_sustain=50\n")
        (d / "b.sf2").write_bytes(convert_sfz(d / "b.sfz", title="t", preset="t").data)
        notes = [(69, 100), (76, 100), (93, 20), (93, 64), (93, 100), (93, 127), (103, 127), (108, 127)]
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message("program_change", program=0))
        for i, (key, vel) in enumerate(notes):           # 120 bpm: 0.6 s notes, 0.1 s apart
            track.append(mido.Message("note_on", note=key, velocity=vel, time=0 if i == 0 else 96))
            track.append(mido.Message("note_off", note=key, velocity=0, time=576))
        track.append(mido.MetaMessage("end_of_track", time=480))
        mid.save(d / "n.mid")
        y, sr = read_float(fluid.render_midi(d / "b.sf2", d / "n.mid", d / "n.wav"))
        assert sr == 44100
        _RENDER.update(y=y, notes=notes)
    return _RENDER["y"], _RENDER["notes"]


def _note(y: np.ndarray, i: int) -> np.ndarray:
    a = int(round((i * 0.7 + 0.15) * 44100))
    return y[0, a:a + 17640].astype(np.float64)           # 0.4 s inside the note


@needs_fluidsynth
def test_fluidsynth_plays_a_converted_sine_bank_at_its_pitch(tmp_path_factory):
    y, notes = _render(tmp_path_factory)                    # render_midi raises on any FluidSynth fault line
    for i in (0, 1):
        key = notes[i][0]
        want = 440 * 2 ** ((key - 69) / 12)
        cents = 1200 * math.log2(_peak_hz(_note(y, i), 44100, want) / want)
        assert abs(cents) < 3, f"key {key}: {cents:+.2f} cents"


@needs_fluidsynth
def test_fluidsynth_levels_follow_the_sfz(tmp_path_factory):
    # the fitted velocity modulators, volume through initialAttenuation (FluidSynth applies x 0.4 to it, so the
    # converter writes cB / 0.4), and ampeg_sustain through sustainVolEnv (written / 0.96)
    y, notes = _render(tmp_path_factory)
    level = {notes[i]: 20 * math.log10(np.sqrt(np.mean(_note(y, i)[:17600] ** 2))) for i in range(2, 8)}
    for v in (20, 64, 100):
        want = 20 * math.log10(velocity_gain(66.0, (), v))
        assert level[93, v] - level[93, 127] == pytest.approx(want, abs=0.15), f"velocity {v}"
    assert level[103, 127] - level[93, 127] == pytest.approx(-12.0, abs=0.15)
    assert level[108, 127] - level[93, 127] == pytest.approx(20 * math.log10(0.5), abs=0.15)


@needs_fluidsynth
def test_fluidsynth_holds_a_flac_note_on_the_loop_its_riff_block_carries(tmp_path):
    # 0.25 s of a 441 Hz sine as a 24-bit FLAC whose riff block holds a 10-period loop near its end, at the SFZ's
    # default envelope (sustain 100 %): held 1.5 s, the note keeps its level on the loop after the points run out
    x, loop = _sine(441.0, 44100, 0.25, 10)
    write_flac(tmp_path / "s.flac", x, 44100, 24, loop=loop, unity=69)
    (tmp_path / "b.sfz").write_text("<region> sample=s.flac key=69\n")
    (tmp_path / "b.sf2").write_bytes(convert_sfz(tmp_path / "b.sfz", title="t", preset="t").data)
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("program_change", program=0))
    track.append(mido.Message("note_on", note=69, velocity=100, time=0))
    track.append(mido.Message("note_off", note=69, velocity=0, time=1440))       # 1.5 s at 120 bpm
    track.append(mido.MetaMessage("end_of_track", time=480))
    mid.save(tmp_path / "n.mid")
    y, sr = read_float(fluid.render_midi(tmp_path / "b.sf2", tmp_path / "n.mid", tmp_path / "n.wav"))

    def db(a: float, b: float) -> float:
        seg = y[0, int(a * sr):int(b * sr)].astype(np.float64)
        return 20 * math.log10(np.sqrt(np.mean(seg ** 2)) + 1e-30)
    assert db(1.0, 1.4) == pytest.approx(db(0.05, 0.2), abs=0.5), f"{db(1.0, 1.4):.1f} dB against {db(0.05, 0.2):.1f}"
