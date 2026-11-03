"""Per-instrument `trim_db` and `fader_db` (docs/SHOOTOUTS.md §7.1, acceptance tests 1-3 and 8; tests 4-7 are the level
tool's, tests/test_shootout_levels.py): the palette fields, where the renderer applies them, and what the render
manifest and the drafts and shootout records report.

- Test 1: with every field at 0 the golden hashes stand (tests/test_determinism.py checks them, never re-recorded);
  here dev_gu with an explicit 0.0 on every instrument renders exactly as dev_gu does.
- Test 2: a trim is the old carriage (CD-51). A FluidSynth voice's trim is bit-equal (stronger than the doc's 1e-7)
  to a [Gain, {gain_db: t}] at the head of its bus chain, sax_warm off and on; a kit's equals t added to every
  lane's db.
- Test 3: `fader_db: 6` on the keys voice raises only the keys bus after its chain, by 6 dB; the keys bus before its
  fader and every stem are unchanged.
- Test 8: an out-of-range or misplaced field is a PaletteError that names its path.
- Beyond the doc's list: (a) the manifest's new bus fields name the playing instrument; (b) `Palette.bus_levels` for
  every keys x kit pair; (c) the drafts and shootout records report effective trims and faders (slot + field,
  bus_gain_db + field); (d) a frozen-style palette (fields, no slots, bus_gain_db 0) renders like its folded twin
  (slots, lane dbs, bus_gain_db).

Render-level cases render dev_gu's GU-only banks (the committed SoundFont), all at once in one module fixture, and
skip without FluidSynth. Sound claims: none.
"""
import copy
import hashlib
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import yaml

from helpers_compile import DEV, EXAMPLE_BRIEF, FIX, ROOT, compiled, dev_palette
from helpers_render import needs_fluidsynth, noise, rendered
from lofi_rl.drafts import render as drafts_render
from lofi_rl.palette import (BassVoice, BusLevel, Kit, KeysVoice, PaletteError, Plugin, SaxVoice, load_palette,
                             parse_palette)
from lofi_rl.render.loudness import lufs
from lofi_rl.render.machine import golden_key
from lofi_rl.render.mix import BUSES, MixError, bus_chain, mix_buses, process
from lofi_rl.render.pipeline import SCHEMA, render
from lofi_rl.render.sampler import kit_shots, render_drums
from lofi_rl.shootout import render as shootout_render

SR = 44100
DOC = yaml.safe_load(DEV.read_text())
PLAYING = {"sax": "sax.tenor", "keys": "keys.rhodes", "bass": "kits.jazz.bass", "drums": "kits.jazz"}  # example.ls
# Test values, carried as the frozen palette carries its levels (the freeze spec, §2): every instrument with its own
# trim and fader, the two that never play here (sax.alto, keys.organ) with values of their own; no chain-head slots;
# bus_gain_db 0, as in dev_gu.
FROZEN_STYLE = {"sax.tenor": (6.63, 0.89), "sax.alto": (21.86, -1.0), "keys.rhodes": (-3.22, -6.435),
                "keys.organ": (1.5, -2.0), "kits.jazz": (-14.94, 15.21), "kits.jazz.bass": (-2.79, 5.18)}
SHAKER_DB = 2.47                # a lane db of its own under the kit trim, as the frozen studio kit's shaker


# --------------------------------------------------------------------------------------------- documents
def put(doc: dict, path: str, value) -> dict:
    """A deep copy of `doc` with the dotted `path` set to `value` (the last key may be new)."""
    out = copy.deepcopy(doc)
    *head, last = path.split(".")
    node = out
    for key in head:
        node = node[key]
    node[last] = value
    return out


def with_levels(doc: dict, levels: dict[str, tuple[float, float]]) -> dict:
    """`doc` with (trim_db, fader_db) written on each instrument named by its palette path."""
    for path, (t, f) in levels.items():
        doc = put(put(doc, f"{path}.trim_db", t), f"{path}.fader_db", f)
    return doc


def instruments(doc: dict) -> list[str]:
    """The palette path of every instrument: each sax and keys voice, each kit and each kit's bass."""
    return ([f"sax.{k}" for k in doc["sax"]] + [f"keys.{k}" for k in doc["keys"]]
            + [p for k in doc["kits"] for p in (f"kits.{k}", f"kits.{k}.bass")])


def gain(t: float) -> list:
    return ["Gain", {"gain_db": t}]


def folded(doc: dict, keys: str = "rhodes", kit: str = "jazz") -> dict:
    """The playing band's fields put back where the shootout and draft palettes carry them (CD-51; the freeze spec's
    fold): a FluidSynth voice's trim as a chain-head Gain slot, the kit's added to every lane's db, every fader added
    to bus_gain_db. Every field is dropped."""
    out = copy.deepcopy(doc)
    band = {"sax": out["sax"][out["sax_lead"]], "keys": out["keys"][keys], "bass": out["kits"][kit]["bass"],
            "drums": out["kits"][kit]}
    for lane in out["kits"][kit]["lanes"].values():
        lane["db"] = lane["db"] + band["drums"].get("trim_db", 0.0)
    for bus in ("sax", "keys", "bass"):
        out["mix"]["buses"][bus] = [gain(band[bus].get("trim_db", 0.0))] + out["mix"]["buses"][bus]
    out["mix"]["bus_gain_db"] = {b: g + band[b].get("fader_db", 0.0) for b, g in out["mix"]["bus_gain_db"].items()}
    for node in [*out["sax"].values(), *out["keys"].values(), *out["kits"].values(),
                 *(k["bass"] for k in out["kits"].values())]:
        node.pop("trim_db", None)
        node.pop("fader_db", None)
    return out


def palette_of(doc: dict, name: str):
    """The palette a file holding `doc` would load to, its id `fields_<name>`."""
    doc = put(doc, "id", f"fields_{name}")
    text = yaml.safe_dump(doc, sort_keys=False)
    return parse_palette(yaml.safe_load(text), f"{name}.yaml", hashlib.sha256(text.encode()).hexdigest())


# --------------------------------------------------------------------------------------------- test 8: the schema
@pytest.mark.parametrize("path, literal, message", [
    ("keys.rhodes.trim_db", "40.01", r"^keys\.rhodes\.trim_db: 40\.01 is outside -40\.\.40$"),
    ("sax.tenor.fader_db", "-40.5", r"^sax\.tenor\.fader_db: -40\.5 is outside -40\.\.40$"),
    ("kits.jazz.bass.trim_db", ".nan", r"^kits\.jazz\.bass\.trim_db: expected a number, got nan$"),
    ("kits.jazz.fader_db", "true", r"^kits\.jazz\.fader_db: expected a number, got True$"),
    # beyond the doc's list: the infinities, a quoted number and a null, on the other fields
    ("kits.jazz.trim_db", ".inf", r"^kits\.jazz\.trim_db: inf is outside -40\.\.40$"),
    ("sax.alto.trim_db", "-.inf", r"^sax\.alto\.trim_db: -inf is outside -40\.\.40$"),
    ("keys.organ.fader_db", "'6'", r"^keys\.organ\.fader_db: expected a number, got '6'$"),
    ("kits.jazz.bass.fader_db", "null", r"^kits\.jazz\.bass\.fader_db: expected a number, got None$"),
])
def test_an_out_of_range_field_is_a_palette_error_naming_its_path(path, literal, message):
    with pytest.raises(PaletteError, match=message):
        parse_palette(put(DOC, path, yaml.safe_load(literal)))


BED = {"src": "gu", "start_s": 0.0, "xfade_s": 1.0, "delta_lu": 24.0}


@pytest.mark.parametrize("path, value, where, field", [
    ("kits.jazz.lanes.kick.trim_db", 1, "kits.jazz.lanes.kick", "trim_db"),
    ("mix.fader_db", 0, "mix", "fader_db"),
    ("groove.trim_db", 0, "groove", "trim_db"),
    ("trim_db", 0, "palette", "trim_db"),
    # beyond the doc's list
    ("mix.bus_gain_db.fader_db", 0, "mix.bus_gain_db", "fader_db"),
    ("keys.rhodes.vel.trim_db", 0, "keys.rhodes.vel", "trim_db"),
    ("sources.gu.fader_db", 0, "sources.gu", "fader_db"),
    ("sax_expr.trim_db", 0, "sax_expr", "trim_db"),
    ("bed", {**BED, "fader_db": 0}, "bed", "fader_db"),
])
def test_a_field_anywhere_else_is_an_unknown_key(path, value, where, field):
    with pytest.raises(PaletteError, match=rf"^{re.escape(where)}: unknown key\(s\) {field}$"):
        parse_palette(put(DOC, path, value))


def test_the_fields_are_numbers_on_exactly_the_four_instrument_mappings():
    doc = with_levels(DOC, {"sax.tenor": (40, -40), "keys.organ": (-3.22, 6), "kits.jazz": (-14.94, 15.26),
                            "kits.jazz.bass": (34.97, 0)})
    pal = parse_palette(doc)
    got = {"sax.tenor": pal.sax["tenor"], "keys.organ": pal.keys["organ"], "kits.jazz": pal.kits["jazz"],
           "kits.jazz.bass": pal.kits["jazz"].bass}
    assert {p: (v.trim_db, v.fader_db) for p, v in got.items()} == {
        "sax.tenor": (40.0, -40.0), "keys.organ": (-3.22, 6.0), "kits.jazz": (-14.94, 15.26),
        "kits.jazz.bass": (34.97, 0.0)}
    assert all(type(x) is float for v in got.values() for x in (v.trim_db, v.fader_db))
    assert (pal.sax["alto"].trim_db, pal.sax["alto"].fader_db, pal.keys["rhodes"].trim_db) == (0.0, 0.0, 0.0)
    # every constructor call written before the fields stays valid (palette.py, shootout/sour.py): they come last
    assert SaxVoice("t", "gu", 0, 66, (44, 75), (48, 72), (50, 61)).trim_db == 0.0
    assert KeysVoice("r", "gu", 0, 4, {"X": 96}).fader_db == 0.0
    bass = BassVoice("gu", 0, 32, {"X": 104})
    assert (bass.trim_db, Kit("k", bass, (), {}).trim_db, Kit("k", bass, (), {}).fader_db) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("rel", ["palettes/dev_gu.yaml", "palettes/draft_day2.yaml", "shootouts/palettes/base.yaml"])
def test_a_palette_without_the_fields_keeps_its_identity_and_plays_every_level_at_zero(rel):
    path = ROOT / rel
    pal = load_palette(path, verify=False)
    assert pal.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    voices = [*pal.sax.values(), *pal.keys.values(), *pal.kits.values(), *(k.bass for k in pal.kits.values())]
    assert all((v.trim_db, v.fader_db) == (0.0, 0.0) for v in voices)
    assert all((lv.trim_db, lv.fader_db) == (0.0, 0.0)
               for keys in pal.keys for kit in pal.kits for lv in pal.bus_levels(keys, kit).values())


# --------------------------------------------------------------------------------------------- (b) bus_levels
def test_bus_levels_name_each_buss_playing_instrument_for_every_keys_and_kit_pair():
    doc = put(DOC, "kits.studio", copy.deepcopy(DOC["kits"]["jazz"]))
    values = {p: (round(-10.0 + 1.25 * i, 2), round(5.5 - 0.75 * i, 2)) for i, p in enumerate(instruments(doc))}
    assert len(set(values.values())) == len(values) == 8
    pal = parse_palette(with_levels(doc, values))
    pairs = [(k, d) for k in pal.keys for d in pal.kits]
    assert sorted(pairs) == [("organ", "jazz"), ("organ", "studio"), ("rhodes", "jazz"), ("rhodes", "studio")]
    for keys, kit in pairs:
        got = pal.bus_levels(keys, kit)
        assert list(got) == list(BUSES), "render/mix.py's bus order"
        want = {"sax": "sax.tenor", "keys": f"keys.{keys}", "bass": f"kits.{kit}.bass", "drums": f"kits.{kit}"}
        assert got == {b: BusLevel(p, *values[p]) for b, p in want.items()}, (keys, kit)
    assert pal.bus_levels("rhodes", "jazz", "alto")["sax"] == BusLevel("sax.alto", *values["sax.alto"])
    for args, kind in ((("piano", "jazz"), "keys"), (("rhodes", "dusty"), "kits"),     # dusty: an alias, not a name
                       (("rhodes", "jazz", "soprano"), "sax")):
        with pytest.raises(PaletteError, match=rf"^{kind} name '\w+' is not in palette dev_gu$"):
            pal.bus_levels(*args)


# --------------------------------------------------------------------------------------------- tests 2, 3: the mix
def _burst(seed: int, seconds: float = 1.0) -> np.ndarray:
    """Decaying noise bursts every 0.25 s: enough level and gaps to move the compressors and reverbs."""
    x = noise(2, int(seconds * SR), seed).astype(np.float64)
    t = np.arange(x.shape[1]) % (SR // 4)
    return (x * np.exp(-t / 2000.0) * 1.5).astype(np.float32)


STEMS = {b: _burst(11 + i) for i, b in enumerate(BUSES)}


def _with_head(mix, bus: str, t: float):
    """`mix` with a [Gain, {gain_db: t}] slot at the head of `bus`'s chain, as the shootout and draft palettes carry a
    trim (CD-51)."""
    head = Plugin("Gain", MappingProxyType({"gain_db": t}))
    return replace(mix, buses=MappingProxyType({**mix.buses, bus: (head,) + mix.buses[bus]}))


@pytest.mark.parametrize("bus, t, warm", [("keys", 6.63, False), ("sax", 6.63, False), ("sax", 6.63, True),
                                          ("bass", 34.97, False), ("keys", -3.22, False)])
def test_a_voice_trim_is_bit_equal_to_the_old_chain_head_gain(bus, t, warm):
    mix = replace(dev_palette().mix, sax_warm=warm)
    got, want = mix_buses(STEMS, mix, trims={bus: t}), mix_buses(STEMS, _with_head(mix, bus, t))
    assert np.array_equal(got.music, want.music)
    assert all(np.array_equal(got.buses[b], want.buses[b]) for b in BUSES)
    assert bus_chain(mix, bus, t) == bus_chain(_with_head(mix, bus, t), bus), "the same plugins, the trim first"
    assert not np.array_equal(got.buses[bus], mix_buses(STEMS, mix).buses[bus]), "the trim moves its bus"


def test_zero_trims_and_faders_run_every_bus_as_before_and_a_drums_trim_is_refused():
    mix = dev_palette().mix
    base = mix_buses(STEMS, mix)
    zero = mix_buses(STEMS, mix, trims={b: 0.0 for b in ("sax", "keys", "bass")}, faders={b: 0.0 for b in BUSES})
    assert np.array_equal(zero.music, base.music) and all(np.array_equal(zero.buses[b], base.buses[b]) for b in BUSES)
    assert bus_chain(mix, "keys", 0.0) is mix.buses["keys"] and bus_chain(mix, "drums") is mix.buses["drums"]
    with pytest.raises(MixError, match="drums trim is not a Gain on the drum bus"):
        mix_buses(STEMS, mix, trims={"drums": -14.94})
    with pytest.raises(MixError, match="lane's dB"):
        bus_chain(mix, "drums", -14.94)
    for what, bad in (("trims", {"sax_warm": 1.0}), ("faders", {"bed": 1.0})):
        with pytest.raises(MixError, match=f"^{what}: '\\w+' is not one of"):
            mix_buses(STEMS, mix, **{what: bad})
    for v in (float("nan"), float("inf"), True, "6"):
        with pytest.raises(MixError, match="must be a finite number of dB"):
            mix_buses(STEMS, mix, faders={"keys": v})


def _flat(gains: dict[str, float]):
    """A mix with every chain empty, the master too: its music is the master chain's input itself."""
    empty = MappingProxyType({b: () for b in ("sax", "sax_warm", "keys", "bass", "drums")})
    mix = dev_palette().mix
    return replace(mix, bus_gain_db=MappingProxyType(gains), buses=empty, master=())


def test_a_keys_fader_of_6_db_raises_only_the_keys_bus_after_its_chain_by_6_db():
    gains = {"sax": 0.89, "keys": -6.04, "bass": 5.18, "drums": 15.21}
    up = {b: 6.0 if b == "keys" else 0.0 for b in BUSES}
    out = mix_buses(STEMS, _flat(gains), faders={"keys": 6.0})
    want = sum(STEMS[b].astype(np.float64) * 10.0 ** ((gains[b] + up[b]) / 20.0) for b in BUSES)
    assert np.array_equal(out.music, want.astype(np.float32)), "the master input, bit for bit"
    mix = dev_palette().mix                                       # dev_gu's chains and master
    base, louder = mix_buses(STEMS, mix), mix_buses(STEMS, mix, faders={"keys": 6.0})
    assert all(np.array_equal(louder.buses[b], base.buses[b]) for b in BUSES), "every bus before its fader"
    g = mix.bus_gain_db
    total = sum(base.buses[b].astype(np.float64) * 10.0 ** ((g[b] + up[b]) / 20.0) for b in BUSES)
    assert np.array_equal(louder.music, process(mix.master, total.astype(np.float32)))
    keys = base.buses["keys"].astype(np.float64)
    rise = lufs(keys * 10.0 ** ((g["keys"] + 6.0) / 20.0), SR) - lufs(keys * 10.0 ** (g["keys"] / 20.0), SR)
    assert f"{rise:.3f}" == "6.000"


# --------------------------------------------------------------------------------------------- test 2: the kit
@needs_fluidsynth
def test_a_kit_trim_is_every_lanes_db_plus_the_trim():
    lanes = put(DOC, "kits.jazz.lanes.shaker.db", SHAKER_DB)
    trimmed = parse_palette(with_levels(lanes, {"kits.jazz": (-14.94, 0.0)}))
    twin = copy.deepcopy(lanes)
    for lane in twin["kits"]["jazz"]["lanes"].values():
        lane["db"] = lane["db"] + -14.94
    a, b = kit_shots(trimmed, "jazz"), kit_shots(parse_palette(twin), "jazz")
    assert a.lanes == b.lanes and a.pcm == b.pcm and a.lanes["kick"] == (0.0, -14.94)
    draft = yaml.safe_load((ROOT / "palettes" / "draft_day2.yaml").read_text(encoding="utf-8"))
    assert a.lanes["shaker"][1] == draft["kits"]["studio"]["lanes"]["shaker"]["db"] == -12.469999999999999, \
        "the float sum the draft palette's studio shaker holds (CD-51)"
    loop = compiled("shaker_edits.ls")                    # a loop with shaker hits
    assert any(h.lane == "shaker" for h in loop.drums)
    assert np.array_equal(render_drums(loop, a), render_drums(loop, b))
    assert not np.array_equal(render_drums(loop, a), render_drums(loop, kit_shots(parse_palette(lanes), "jazz")))


# --------------------------------------------------------------------------------------------- (c) the records
def _fake_render(pal, buses: dict[str, dict]):
    """render.pipeline.render's result shape, with a manifest holding `buses` and no audio."""
    manifest = {"schema": SCHEMA, "chain": {"cycles": 2}, "n": 1032576, "sr": SR, "buses": buses,
                "loop": {"canon_sha256": "a" * 64, "compiled_sha256": "b" * 64},
                "brief": {"bpm": 82, "tonic": 2, "mode": "dorian", "energy": "mid"},
                "palette": {"id": pal.id, "sha256": pal.sha256}, "bed": None,
                "loudness": {"lufs": -14.0, "true_peak_dbtp": -1.5, "gain_db": 3.0},
                "gates": {"G1-G6": "pass", "G7": "pass"}, "audio_sha256": "c" * 64, "music_sha256": "c" * 64}
    return SimpleNamespace(manifest=manifest, compiled=SimpleNamespace(kit="jazz", sax=SimpleNamespace(notes=())))


TASK = {"render": "x", "file": "x.ls", "text": "x", "brief_id": "d00", "persona": "test", "keys": "rhodes",
        "kit": "jazz", "palette": "x.yaml", "s_star": -30.0,
        "brief": {"energy": "mid", "mood": "wistful", "bpm": 82, "tonic": 2}}
ENTRY = {"loop": "l0", "brief": TASK["brief"], "stage": "test", "option": "o", "what": "w", "bus": "keys",
         "variant": "v", "palette": "p"}


def records(pal, r) -> tuple[dict, dict]:
    """Each bus of the drafts record and of the shootout record of one render."""
    return (drafts_render._record(TASK, "ok", None, pal, r)["buses"],
            shootout_render._record("x", ENTRY, pal, "ok", None, r)["buses"])


def test_the_records_add_each_field_to_its_slot_and_to_bus_gain_db():
    doc = with_levels(DOC, {"sax.tenor": (2.0, 1.5), "keys.rhodes": (-1.0, 3.0), "kits.jazz": (-3.0, 4.0),
                            "kits.jazz.bass": (0.5, -2.5)})
    doc = put(doc, "mix.buses.sax", [gain(-0.25)] + doc["mix"]["buses"]["sax"])     # the old slots, beside the fields
    doc = put(doc, "mix.buses.keys", [gain(1.5)] + doc["mix"]["buses"]["keys"])
    doc = put(doc, "mix.buses.bass", [gain(10.0)] + doc["mix"]["buses"]["bass"])
    doc = put(doc, "kits.jazz.lanes.kick.db", -2.0)
    doc = put(doc, "mix.bus_gain_db", {"sax": 0.5, "drums": -1.0, "bass": 2.0, "keys": -6.0})
    pal = palette_of(doc, "slots_and_fields")
    lv = pal.bus_levels("rhodes", "jazz")
    pre = {"sax": -30.0, "keys": -35.5, "bass": -40.25, "drums": -45.0}
    new = {b: {"lufs": pre[b], "gain_db": pal.mix.bus_gain_db[b], "instrument": lv[b].instrument,
               "trim_db": lv[b].trim_db, "fader_db": lv[b].fader_db} for b in BUSES}
    trim = {"sax": 1.75, "keys": 0.5, "bass": 10.5, "drums": -5.0}               # slot + field
    fader = {"sax": 2.0, "keys": -3.0, "bass": -0.5, "drums": 3.0}               # bus_gain_db + field
    targets = dict(pal.mix.bus_target_lu)
    d, s = records(pal, _fake_render(pal, new))
    assert d == {b: {"lufs": pre[b], "fader_db": fader[b], "trim_db": trim[b], "post_lufs": pre[b] + fader[b],
                     "lu": pre[b] + fader[b] + 30.0, "target_lu": targets[b],
                     "off_lu": pre[b] + fader[b] + 30.0 - targets[b]} for b in BUSES}
    assert s == {b: {"lufs": pre[b], "fader_db": fader[b], "trim_db": trim[b], "post_lufs": pre[b] + fader[b]}
                 for b in BUSES}
    # a manifest written before the fields: the slot and bus_gain_db alone, exactly
    old = {b: {"lufs": pre[b], "gain_db": pal.mix.bus_gain_db[b]} for b in BUSES}
    slot = {"sax": -0.25, "keys": 1.5, "bass": 10.0, "drums": -2.0}
    for buses in records(pal, _fake_render(pal, old)):
        assert {b: (v["trim_db"], v["fader_db"]) for b, v in buses.items()} == {
            b: (slot[b], pal.mix.bus_gain_db[b]) for b in BUSES}


# --------------------------------------------------------------------------------------------- the renders
EXAMPLE = (FIX / "example.ls").read_text()
SHAKER = (FIX / "shaker_edits.ls").read_text()        # keys: wurli and drums: jazz (dev_gu: rhodes, jazz), sax: alto
FROZEN_DOC = with_levels(put(DOC, "kits.jazz.lanes.shaker.db", SHAKER_DB), FROZEN_STYLE)
CASES = {                       # name: (palette document, loop text), each rendered at the example brief
    "zeros": (with_levels(DOC, {p: (0.0, 0.0) for p in instruments(DOC)}), EXAMPLE),
    "trim_field": (put(DOC, "keys.rhodes.trim_db", 6.63), EXAMPLE),
    "trim_slot": (put(DOC, "mix.buses.keys", [gain(6.63)] + DOC["mix"]["buses"]["keys"]), EXAMPLE),
    "fader6": (put(DOC, "keys.rhodes.fader_db", 6), EXAMPLE),
    "frozen": (FROZEN_DOC, SHAKER),
    "folded": (folded(FROZEN_DOC), SHAKER),
}


@pytest.fixture(scope="module")
def renders():
    """Every case's render, and dev_gu's own of example.ls (helpers_render's cached one), made at the same time: a
    render never depends on what else runs (docs/RENDER.md §5), which keeps this module's wall time near one
    render's."""
    if shutil.which("fluidsynth") is None:
        pytest.skip("fluidsynth not on PATH")
    pals = {name: palette_of(doc, name) for name, (doc, _) in CASES.items()}
    with ThreadPoolExecutor(max_workers=len(CASES) + 1) as ex:
        dev = ex.submit(rendered, "example.ls")
        jobs = {name: ex.submit(render, CASES[name][1], EXAMPLE_BRIEF, pal, workers=1) for name, pal in pals.items()}
        out = {name: f.result() for name, f in jobs.items()}
        out["dev"] = dev.result()
    return SimpleNamespace(r=out, pal=pals)


def bus(m: dict, key: str) -> dict:
    return {b: v[key] for b, v in m["buses"].items()}


@needs_fluidsynth
def test_explicit_zero_fields_render_exactly_as_dev_gu(renders):
    dev, zeros = renders.r["dev"].manifest, renders.r["zeros"].manifest
    for key in ("audio_sha256", "music_sha256", "stems", "oneshots", "n", "loudness"):
        assert zeros[key] == dev[key], key
    assert bus(zeros, "lufs") == bus(dev, "lufs")
    golden = json.loads((ROOT / "tests" / "golden" / "render_hashes.json").read_text())["keys"].get(golden_key(), {})
    case = golden.get("example.ls|wistful|mid|82|2")
    if case is not None:                                  # this machine's golden (tests/test_determinism.py)
        assert (zeros["audio_sha256"], zeros["music_sha256"]) == (case["audio_sha256"], case["music_sha256"])


@needs_fluidsynth
def test_the_manifest_names_each_buss_playing_instrument_and_its_fields(renders):
    for name, values in (("dev", {}), ("zeros", {}), ("frozen", FROZEN_STYLE)):
        m = renders.r[name].manifest
        assert m["schema"] == SCHEMA == "lofi-rl/render/1", "additive keys: every clip id keeps its render cfg"
        assert all(list(v) == ["lufs", "gain_db", "instrument", "trim_db", "fader_db"] for v in m["buses"].values())
        assert bus(m, "instrument") == PLAYING, name       # the lead, and the palette's names after aliases
        assert {b: (v["trim_db"], v["fader_db"]) for b, v in m["buses"].items()} == {
            b: values.get(p, (0.0, 0.0)) for b, p in PLAYING.items()}, name
        assert bus(m, "gain_db") == dict(renders.pal.get(name, dev_palette()).mix.bus_gain_db), name
    frozen = renders.r["frozen"].manifest              # shaker_edits.ls writes keys: wurli and sax: alto
    assert frozen["instruments"] == {"keys": "rhodes", "kit": "jazz", "sax": "tenor", "sax_warm": False}


@needs_fluidsynth
def test_a_keys_trim_field_renders_as_the_chain_head_gain_slot(renders):
    field, slot = renders.r["trim_field"].manifest, renders.r["trim_slot"].manifest
    assert field["audio_sha256"] == slot["audio_sha256"]
    assert field["buses"]["keys"]["lufs"] == slot["buses"]["keys"]["lufs"]
    assert field["music_sha256"] == slot["music_sha256"] and field["stems"] == slot["stems"]
    assert bus(field, "lufs") == bus(slot, "lufs")
    assert field["audio_sha256"] != renders.r["dev"].manifest["audio_sha256"], "the trim reaches the render"
    assert (field["buses"]["keys"]["trim_db"], slot["buses"]["keys"]["trim_db"]) == (6.63, 0.0)
    for f, s in zip(records(renders.pal["trim_field"], renders.r["trim_field"]),
                    records(renders.pal["trim_slot"], renders.r["trim_slot"])):
        assert f == s and f["keys"]["trim_db"] == 6.63, "the same effective trim: field or slot"


@needs_fluidsynth
def test_a_keys_fader_of_6_db_leaves_the_keys_bus_before_its_fader_and_every_stem(renders):
    up, dev = renders.r["fader6"].manifest, renders.r["dev"].manifest
    assert abs(up["buses"]["keys"]["lufs"] - dev["buses"]["keys"]["lufs"]) <= 1e-9
    assert bus(up, "lufs") == bus(dev, "lufs") and up["stems"] == dev["stems"] and up["oneshots"] == dev["oneshots"]
    assert (up["buses"]["keys"]["gain_db"], up["buses"]["keys"]["fader_db"]) == (0.0, 6.0)
    assert up["audio_sha256"] != dev["audio_sha256"]
    for buses in records(renders.pal["fader6"], renders.r["fader6"]):
        assert buses["keys"]["fader_db"] == 6.0 and buses["keys"]["post_lufs"] == dev["buses"]["keys"]["lufs"] + 6.0
        assert all(buses[b]["fader_db"] == 0.0 for b in BUSES if b != "keys")


@needs_fluidsynth
def test_a_frozen_style_palette_renders_as_its_folded_twin(renders):
    # The freeze spec's §2 equivalence, end to end on GU banks: the fields, or the shootout carriage (CD-51).
    frozen, twin = renders.r["frozen"], renders.r["folded"]
    for key in ("audio_sha256", "music_sha256", "stems", "oneshots", "n", "music_lufs", "loudness"):
        assert frozen.manifest[key] == twin.manifest[key], key
    assert bus(frozen.manifest, "lufs") == bus(twin.manifest, "lufs")
    assert np.array_equal(frozen.pre_limiter, twin.pre_limiter)
    assert bus(frozen.manifest, "gain_db") == {b: 0.0 for b in BUSES}
    assert bus(twin.manifest, "gain_db") == {b: FROZEN_STYLE[p][1] for b, p in PLAYING.items()}
    assert renders.pal["folded"].kits["jazz"].lanes["shaker"].db == SHAKER_DB + -14.94
    want = {b: FROZEN_STYLE[p] for b, p in PLAYING.items()}             # (effective trim, effective fader)
    for f, t in zip(records(renders.pal["frozen"], frozen), records(renders.pal["folded"], twin)):
        assert f == t, "the records read the same effective levels"
        assert {b: (v["trim_db"], v["fader_db"]) for b, v in f.items()} == want
