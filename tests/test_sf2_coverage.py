"""SoundFont coverage at palette load (docs/SHOOTOUTS.md §7.5).

`palette.sf2_coverage(path, bank, prog)` is the set of (key, velocity) cells a preset can sound, read from its
preset and instrument zones the way FluidSynth 2.6.1 reads them: a zone's key and velocity ranges default to its
global zone's, a keyRange counts only as its zone's first generator (a velRange only after nothing or a keyRange),
and a preset zone plays an instrument zone where both ranges hold. `load_palette(verify=True)` then refuses a voice
that cannot sound a note the compiler may send: keys 48-72 and kit basses 28-43 at their `vel` values (widened by the
humanize velocity range when humanize is on for the lane), each sax's `play` range at every velocity of its clip, and
each SoundFont one-shot's note at its `vel`; the error lists the missing cells.

Synthetic banks are written with the converter's SoundFont writer (lofi_rl.convert.sf2). One test renders the
tricky cases through FluidSynth, so the reader is checked against the synthesizer, not only against the spec. The
archive SF2s and the converted banks are checked by the cluster-marked tests, which skip unless LOFI_ASSETS_BIG
(or LOFI_ASSETS) names the assets/big folder."""
from __future__ import annotations

import copy
import hashlib
import os
import re
import struct
import tempfile
from pathlib import Path

import mido
import numpy as np
import pytest
import yaml

from helpers_compile import DEV, ROOT
from helpers_render import needs_fluidsynth
from lofi_rl.convert.sf2 import GEN, Instrument, Preset, Sample, Zone, sf2_bytes
from lofi_rl.palette import PaletteError, load_palette, sf2_coverage
from lofi_rl.render import fluid
from lofi_rl.render.wavio import read_float

DOC = yaml.safe_load(DEV.read_text())
GU = ROOT / "assets" / "soundfonts" / "generaluser-gs" / "GeneralUser-GS.sf2"
ALL = range(128)


def cells(keys, vels) -> set[tuple[int, int]]:
    return {(k, v) for k in keys for v in vels}


def span(lo: int, hi: int) -> range:
    return range(lo, hi + 1)


def zone(keys=None, vels=None, *, gens=(), **ref) -> Zone:
    """A zone with optional key and velocity ranges (written first, as the writer always does) and other gens."""
    head = ([(GEN["keyRange"], keys)] if keys else []) + ([(GEN["velRange"], vels)] if vels else [])
    return Zone(gens=tuple(head) + tuple(gens), **ref)


def tone() -> Sample:
    """One looped sine at its root key 60, 0.25 s, at half scale (Sample points are 24-bit; smpl keeps 16 bits)."""
    t = np.arange(11025)
    pcm = np.round(np.sin(2 * np.pi * 261.6256 * t / 44100) * 2 ** 22).astype(np.int32)
    return Sample("tone", pcm, 44100, bits=16, root=60, loop=(1000, 1000 + 169 * 50))


LOOPED = ((GEN["sampleModes"], 1),)


def write_sf2(path: Path, presets, instruments) -> Path:
    path.write_bytes(sf2_bytes(name="coverage", presets=presets, instruments=instruments, samples=[tone()]))
    return path


def one_preset(path: Path, *inst_zones: Zone, preset_zone: Zone | None = None, global_inst: Zone | None = None,
               global_preset: Zone | None = None) -> Path:
    """A bank with one preset (0, 0) playing one instrument."""
    inst = Instrument("inst", tuple(inst_zones), global_zone=global_inst)
    pz = preset_zone or Zone()
    preset = Preset("preset", 0, 0, (Zone(gens=pz.gens, instrument="inst"),), global_zone=global_preset)
    return write_sf2(path, [preset], [inst])


def misplace_key_range(path: Path) -> None:
    """Move the instrument zone's keyRange after its pan: sf2_bytes always writes it first, as SF2 2.04 asks."""
    data = bytearray(path.read_bytes())
    pos = data.find(b"igen")
    count = struct.unpack_from("<I", data, pos + 4)[0] // 4
    recs = [struct.unpack_from("<HH", data, pos + 8 + 4 * i) for i in range(count)]
    i = next(i for i in range(count - 1) if recs[i][0] == GEN["keyRange"] and recs[i + 1][0] == GEN["pan"])
    struct.pack_into("<HHHH", data, pos + 8 + 4 * i, *recs[i + 1], *recs[i])
    path.write_bytes(bytes(data))


# ------------------------------------------------------------------------------------------- the reader
def test_coverage_is_where_a_preset_zone_and_an_instrument_zone_both_hold(tmp_path):
    sf2 = one_preset(tmp_path / "a.sf2", zone((0, 59), sample="tone"), zone((61, 127), (0, 63), sample="tone"),
                     preset_zone=zone((40, 100)))
    assert sf2_coverage(sf2, 0, 0) == frozenset(cells(span(40, 59), ALL) | cells(span(61, 100), span(0, 63)))
    sf2 = one_preset(tmp_path / "b.sf2", zone(sample="tone"))
    assert sf2_coverage(sf2, 0, 0) == frozenset(cells(ALL, ALL))
    with pytest.raises(PaletteError, match="bank 0 prog 1 is not a preset"):
        sf2_coverage(sf2, 0, 1)
    with pytest.raises(PaletteError, match="not a SoundFont"):
        sf2_coverage(DEV, 0, 0)


def test_global_zones_set_the_default_ranges_of_their_local_zones(tmp_path):
    # FluidSynth 2.6.1 starts every zone from its global zone's ranges; a zone's own range overrides them.
    sf2 = one_preset(tmp_path / "a.sf2", zone(sample="tone"), zone((70, 80), sample="tone"),
                     global_inst=zone((0, 59)), global_preset=zone(vels=(0, 63)))
    assert sf2_coverage(sf2, 0, 0) == frozenset(cells(list(span(0, 59)) + list(span(70, 80)), span(0, 63)))


def test_a_range_that_is_not_its_zones_first_generator_is_ignored(tmp_path):
    sf2 = one_preset(tmp_path / "a.sf2", zone((0, 59), gens=((GEN["pan"], 0),), sample="tone"))
    assert sf2_coverage(sf2, 0, 0) == frozenset(cells(span(0, 59), ALL))
    misplace_key_range(sf2)
    assert sf2_coverage(sf2, 0, 0) == frozenset(cells(ALL, ALL))


@needs_fluidsynth
def test_the_coverage_reader_agrees_with_fluidsynth(tmp_path):
    # One bank with the three tricky cases as presets 0-2, one note per channel and second: each note sounds in
    # FluidSynth exactly when sf2_coverage says its cell is covered.
    insts = [Instrument("global", (zone(gens=LOOPED, sample="tone"),), global_zone=zone((0, 59))),
             Instrument("plain", (zone(gens=LOOPED, sample="tone"),)),
             Instrument("late", (zone((0, 59), gens=((GEN["pan"], 0),) + LOOPED, sample="tone"),))]
    presets = [Preset("p0", 0, 0, (Zone(instrument="global"),)),
               Preset("p1", 0, 1, (Zone(instrument="plain"),), global_zone=zone(vels=(0, 63))),
               Preset("p2", 0, 2, (Zone(instrument="late"),))]
    sf2 = write_sf2(tmp_path / "t.sf2", presets, insts)
    misplace_key_range(sf2)
    notes = [(0, 50, 100), (0, 60, 100), (1, 60, 40), (1, 60, 100), (2, 50, 100), (2, 60, 100)]
    mid = mido.MidiFile(type=0, ticks_per_beat=500)          # 120 BPM: 1 tick = 1 ms
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for ch, (prog, _, _) in enumerate(notes):
        track += [mido.Message("control_change", channel=ch, control=0, value=0),
                  mido.Message("program_change", channel=ch, program=prog)]
    for ch, (_, key, vel) in enumerate(notes):
        track += [mido.Message("note_on", channel=ch, note=key, velocity=vel, time=0 if ch == 0 else 700),
                  mido.Message("note_off", channel=ch, note=key, velocity=0, time=300)]
    track.append(mido.MetaMessage("end_of_track", time=100))
    mid.save(str(tmp_path / "t.mid"))
    x, sr = read_float(fluid.render_midi(sf2, tmp_path / "t.mid", tmp_path / "t.wav"))
    for i, (prog, key, vel) in enumerate(notes):
        window = x[:, int((i + 0.05) * sr):int((i + 0.25) * sr)]
        sounds = float(np.abs(window).max()) > 1e-4
        assert sounds == ((key, vel) in sf2_coverage(sf2, 0, prog)), (prog, key, vel)


def test_the_gu_presets_cover_every_dev_voice():
    # docs/SHOOTOUTS.md §7.5: "GU presets 4, 16, 32, 65 and 66 cover 0-108 or more"; the Jazz kit sounds its notes.
    for prog in (4, 16, 32, 65, 66):
        assert cells(span(0, 108), span(1, 127)) <= sf2_coverage(GU, 0, prog), prog
    kit = sf2_coverage(GU, 128, 32)
    assert cells((36, 37, 38, 42, 46, 82), (48, 80, 112)) <= kit
    pal = load_palette(DEV, verify=True)
    assert pal.verify_coverage() == []


# ----------------------------------------------------------------------------------- refusals at load
def with_source(tmp_path: Path, sf2: Path, **paths) -> Path:
    """The dev palette with a source `syn` (the synthetic bank) and dotted paths replaced (None deletes)."""
    doc = copy.deepcopy(DOC)
    doc["sources"]["syn"] = {"path": str(sf2), "sha256": hashlib.sha256(sf2.read_bytes()).hexdigest(),
                             "license": "CC0-1.0", "mode": "commit"}
    for path, value in paths.items():
        *head, last = path.split("__")
        node = doc
        for key in head:
            node = node[int(key)] if isinstance(node, list) else node[key]
        node[last] = value
    out = tmp_path / "pal.yaml"
    out.write_text(yaml.safe_dump(doc))
    return out


def test_a_keys_voice_with_no_zone_at_key_60_is_refused_naming_60(tmp_path):
    sf2 = one_preset(tmp_path / "a.sf2", zone((0, 59), sample="tone"), zone((61, 127), sample="tone"))
    pal = with_source(tmp_path, sf2, keys__rhodes__src="syn", keys__rhodes__prog=0)
    assert load_palette(pal, verify=False)
    with pytest.raises(PaletteError, match=r"keys\.rhodes: source syn bank 0 prog 0 has no zone for key 60 at "
                                           r"velocity 52, 80, 96$"):
        load_palette(pal, verify=True)


def test_a_bass_whose_zones_stop_at_velocity_63_fails_at_104(tmp_path):
    sf2 = one_preset(tmp_path / "a.sf2", zone(vels=(0, 63), sample="tone"))
    pal = with_source(tmp_path, sf2, kits__jazz__bass__src="syn", kits__jazz__bass__prog=0)
    with pytest.raises(PaletteError, match=r"kits\.jazz\.bass: source syn bank 0 prog 0 has no zone for keys "
                                           r"28-43 at velocity 64, 88, 104$"):
        load_palette(pal, verify=True)


def test_a_sax_covering_48_72_fails_for_play_44_75_listing_44_47_and_73_75(tmp_path):
    sf2 = one_preset(tmp_path / "a.sf2", zone((48, 72), sample="tone"))
    pal = with_source(tmp_path, sf2, sax__tenor__src="syn", sax__tenor__prog=0)
    with pytest.raises(PaletteError) as err:
        load_palette(pal, verify=True)
    assert str(err.value) == ("sax.tenor: source syn bank 0 prog 0 has no zone for keys 44-47 at velocity 60-112; "
                              "keys 73-75 at velocity 60-112")
    # The check reads the play range and the velocity clip from the palette: a narrower sax passes.
    pal = with_source(tmp_path, sf2, sax__tenor__src="syn", sax__tenor__prog=0, sax__tenor__play=[48, 72])
    assert load_palette(pal, verify=True).sax["tenor"].play == (48, 72)


def test_a_soundfont_one_shot_needs_its_note_at_its_velocity(tmp_path):
    sf2 = one_preset(tmp_path / "a.sf2", zone((40, 127), sample="tone"), zone((36, 36), (0, 100), sample="tone"))
    shot = {"src": "syn", "bank": 0, "prog": 0, "note": 36, "vel": 112}
    pal = with_source(tmp_path, sf2, kits__jazz__lanes__kick__hard=[shot])
    with pytest.raises(PaletteError, match=r"^kits\.jazz\.lanes\.kick\.hard\[0\]: source syn bank 0 prog 0 has no "
                                           r"zone for key 36 at velocity 112$"):
        load_palette(pal, verify=True)
    pal = with_source(tmp_path, sf2, kits__jazz__lanes__kick__hard=[dict(shot, vel=100)])
    assert load_palette(pal, verify=True)


def test_humanize_widens_the_keys_and_bass_velocities_checked(tmp_path):
    # COMPILER.md §3.6: with humanize on, a keys or bass strike plays rnd(vel x (1 +- 0.08)), so an X of 96 can
    # reach 104 and a g of 52 can fall to 48.
    sf2 = one_preset(tmp_path / "a.sf2", zone(vels=(50, 96), sample="tone"))
    assert load_palette(with_source(tmp_path, sf2, keys__rhodes__src="syn", keys__rhodes__prog=0), verify=True)
    pal = with_source(tmp_path, sf2, keys__rhodes__src="syn", keys__rhodes__prog=0, groove__humanize="bar")
    with pytest.raises(PaletteError, match=r"keys\.rhodes: .* keys 48-72 at velocity 48-49, 97-104$"):
        load_palette(pal, verify=True)
    pal = with_source(tmp_path, sf2, keys__rhodes__src="syn", keys__rhodes__prog=0, groove__humanize="bar",
                      groove__humanize_lanes=["kick", "snare", "bass"])
    assert load_palette(pal, verify=True)


# ------------------------------------------------------------------------------------ real banks (cluster)
def big_or_skip() -> Path:
    for var in ("LOFI_ASSETS_BIG", "LOFI_ASSETS"):
        value = os.environ.get(var)
        if value and Path(value).is_dir():
            return Path(value)
    pytest.skip("LOFI_ASSETS_BIG (or LOFI_ASSETS) does not name the assets/big folder")


MANIFEST = {a["path"]: a for a in yaml.safe_load((ROOT / "assets" / "MANIFEST.yaml").read_text())["assets"]}
ARCHIVES = {   # the archive's MANIFEST path -> (presets, keys sounding at every velocity 1-127: SHOOTOUTS.md §7.5)
    "assets/big/freepats-tenor-sax/TenorSaxophone-SF2-20200717.tar.bz2": ([(0, 0)], (43, 89)),
    "assets/big/freepats-fm-piano-1/FM-Piano1-SF2-20190916.7z": ([(0, 0), (0, 1), (0, 2), (0, 3)], (27, 98)),
    "assets/big/freepats-upright-piano-kw/UprightPianoKW-SF2-20220221.7z": ([(0, 0)], (21, 108)),
    "assets/big/freepats-fsbs-jazz-guitar/EGuitarFSBS-jazz-bridge-small-SF2-20260807.7z": ([(0, 0)], (35, 86)),
    "assets/big/freepats-finger-bass-yr/FingerBassYR-SF2-20190930.7z": ([(0, 0)], (26, 45)),
}


def _local(big: Path, manifest_path: str) -> Path:
    return big / Path(manifest_path).relative_to("assets/big")


def _keys_at_every_velocity(cover: frozenset) -> tuple[int, int]:
    keys = sorted(k for k in ALL if all((k, v) in cover for v in span(1, 127)))
    assert keys == list(span(keys[0], keys[-1])), "one unbroken key span"
    return keys[0], keys[-1]


@pytest.mark.cluster
@pytest.mark.parametrize("archive", sorted(ARCHIVES))
def test_the_archive_soundfonts_cover_what_the_plan_read_from_them(archive):
    big = big_or_skip()
    (extract,) = MANIFEST[archive]["extracts"]
    sf2 = _local(big, extract["path"])
    presets, keys = ARCHIVES[archive]
    for bank, prog in presets:
        cover = sf2_coverage(sf2, bank, prog)
        assert _keys_at_every_velocity(cover) == keys, (archive, prog)
        print(f"{sf2.name} ({bank}, {prog}): keys {keys[0]}-{keys[1]} at every velocity 1-127; {len(cover)} cells")


# Each SoundFont in the role its shootout gives it (SHOOTOUTS.md §4): a palette with it there loads with verify=True.
ROLES = {
    "assets/big/freepats-tenor-sax/TenorSaxophone-SF2-20200717.tar.bz2": "tenor",
    "assets/big/freepats-fm-piano-1/FM-Piano1-SF2-20190916.7z": "keys",
    "assets/big/freepats-upright-piano-kw/UprightPianoKW-SF2-20220221.7z": "keys",
    "assets/big/freepats-fsbs-jazz-guitar/EGuitarFSBS-jazz-bridge-small-SF2-20260807.7z": "keys",
    "assets/big/freepats-finger-bass-yr/FingerBassYR-SF2-20190930.7z": "bass",
    "assets/big/converted/weresax-alto.sf2": "alto",
    "assets/big/converted/mtg-tenor.sf2": "tenor",
    "assets/big/converted/mtg-alto.sf2": "alto",
    "assets/big/converted/jrhodes3d-mono.sf2": "keys",
    "assets/big/converted/vcsl-upright-knight.sf2": "keys",
    "assets/big/converted/vsco2-contrabass-pizz.sf2": "bass",
    "assets/big/converted/gsullivan-wurlitzer.sf2": "keys",
}


@pytest.mark.cluster
@pytest.mark.parametrize("asset", sorted(ROLES))
def test_every_shootout_soundfont_passes_in_its_role(asset):
    big = big_or_skip()
    entry = MANIFEST[asset]
    pinned = entry["extracts"][0] if "extracts" in entry else entry
    sf2 = _local(big, pinned["path"])
    if not sf2.is_file():
        pytest.skip(f"{sf2} is not there")
    doc = copy.deepcopy(DOC)
    doc["sources"]["cand"] = {"path": str(sf2), "sha256": pinned["sha256"], "license": entry["license"],
                              "mode": "fetch", "attribution": "assets/ATTRIBUTION.md"}
    role = ROLES[asset]
    voice = {"src": "cand", "bank": 0, "prog": 0}
    if role in ("tenor", "alto"):
        doc["sax"][role].update(voice)
    elif role == "keys":
        doc["keys"]["rhodes"].update(voice)
    else:
        doc["kits"]["jazz"]["bass"].update(voice)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "pal.yaml"
        path.write_text(yaml.safe_dump(doc))
        pal = load_palette(path, verify=True)
    cover = sf2_coverage(sf2, 0, 0)
    keys = sorted({k for k, v in cover if v >= 1})
    print(f"{sf2.name} as {role}: keys {keys[0]}-{keys[-1]} at some velocity; {len(cover)} cells")
    assert pal.verify_coverage() == []
