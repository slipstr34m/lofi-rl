"""Palette schema and licences (instrument.md sections 3, 4 and 6; docs/DECISIONS.md D-012).

Every source has a sha256 and a licence, non-commercial sources are fetch-only, the sha256 matches the
file, and the dev palette carries the spec's starting values."""
import copy
import hashlib
import re
from pathlib import Path

import pytest
import yaml

from helpers_compile import DEV, FIX, ROOT, dev_palette
from lofi_rl.dsl import parse
from lofi_rl.dsl.vocab import LANES, NUMERALS, QUALITIES, VOICINGS
from lofi_rl.palette import PLUGINS, PaletteError, licence_problem, load_palette, parse_palette, sf2_presets

DOC = yaml.safe_load(DEV.read_text())


def edited(**paths):
    """A copy of the dev palette document with dotted paths replaced (None deletes the key)."""
    doc = copy.deepcopy(DOC)
    for path, value in paths.items():
        *head, last = path.split("__")
        node = doc
        for key in head:
            node = node[int(key)] if isinstance(node, list) else node[key]
        if value is None:
            del node[last]
        else:
            node[last] = value
    return doc


def test_dev_palette_loads_and_its_source_matches_its_sha256():
    pal = load_palette(DEV, verify=True)
    assert pal.id == "dev_gu" and pal.frozen is None and pal.sr == 44100
    assert pal.verify_sources() == []
    assert pal.sha256 == hashlib.sha256(DEV.read_bytes()).hexdigest()
    src = pal.sources["gu"]
    assert src.mode == "commit" and src.license and len(src.sha256) == 64
    manifest = {a["path"]: a for a in yaml.safe_load((ROOT / "assets" / "MANIFEST.yaml").read_text())["assets"]}
    assert manifest[src.path]["sha256"] == src.sha256, "the palette and assets/MANIFEST.yaml pin the same file"


def test_every_palette_in_the_repo_validates():
    for path in sorted((ROOT / "palettes").glob("*.yaml")):
        pal = load_palette(path, verify=False)
        for sid, src in pal.sources.items():
            assert src.license and len(src.sha256) == 64, (path.name, sid)
            if src.mode == "commit":              # fetch-mode files are not on the laptop
                data = pal.source_path(sid).read_bytes()
                assert hashlib.sha256(data).hexdigest() == src.sha256, (path.name, sid)


def test_a_missing_or_changed_source_file_is_reported(tmp_path):
    doc = edited(sources__gu__path=str(tmp_path / "missing.sf2"))
    pal = parse_palette(doc)
    assert any("missing" in p for p in pal.verify_sources(root=Path("/")))
    other = tmp_path / "other.sf2"
    other.write_bytes(b"not the soundfont")
    pal = parse_palette(edited(sources__gu__path=str(other)))
    assert any("sha256" in p for p in pal.verify_sources(root=Path("/")))
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(edited(sources__gu__path=str(other))))
    with pytest.raises(PaletteError, match="sha256"):
        load_palette(bad, verify=True, root=Path("/"))


@pytest.mark.parametrize("license_text", ["CC-BY-NC-4.0", "CC BY-NC-SA 3.0", "free for non-commercial use"])
def test_non_commercial_sources_must_be_fetch_only(license_text):
    doc = edited(sources__gu__license=license_text, sources__gu__attribution="Someone, jRhodes3d")
    with pytest.raises(PaletteError, match="non-commercial"):
        parse_palette(doc)
    doc["sources"]["gu"]["mode"] = "fetch"
    assert parse_palette(doc).sources["gu"].mode == "fetch"


def test_cc_by_sources_need_their_attribution():
    with pytest.raises(PaletteError, match="attribution"):
        parse_palette(edited(sources__gu__license="CC-BY-4.0"))
    assert parse_palette(edited(sources__gu__license="CC-BY-4.0", sources__gu__attribution="MTG, Solo Sax"))
    for spelled_out in ("Creative Commons Attribution 3.0 Unported", "Attribution 4.0 International"):
        with pytest.raises(PaletteError, match="attribution"):
            parse_palette(edited(sources__gu__license=spelled_out))


@pytest.mark.parametrize("license_text", [
    "free for non commercial use", "Creative Commons Attribution Non Commercial 4.0", "Non Commercial use only",
    "noncommercial", "All rights reserved", "personal use only", "proprietary; do not redistribute",
    "free to use, not for redistribution", "CC BY-SA 4.0", "CC BY-ND 4.0", "GPL-3.0-or-later", "unknown"])
def test_only_redistributable_licences_may_be_committed(license_text):
    # docs/DECISIONS.md D-012: commit only CC0, public domain, permissive and CC-BY; everything else is fetched.
    doc = edited(sources__gu__license=license_text, sources__gu__attribution="Someone, Some Bank")
    with pytest.raises(PaletteError, match="mode fetch"):
        parse_palette(doc)
    doc["sources"]["gu"]["mode"] = "fetch"
    assert parse_palette(doc).sources["gu"].mode == "fetch"


@pytest.mark.parametrize("license_text, attribution", [
    ("CC0 1.0", None), ("CC0-1.0 (synthetic test noise)", None), ("public domain", None), ("MIT", None),
    ("BSD-3-Clause", None), ("Apache-2.0", None), (DOC["sources"]["gu"]["license"], None),
    ("CC BY 4.0", "MTG, Solo Sax"), ("CC-BY-3.0", "Greg Sullivan, Wurlitzer EP200"),
    ("Creative Commons Attribution 3.0 Unported", "Greg Sullivan, Wurlitzer EP200")])
def test_redistributable_licences_may_be_committed(license_text, attribution):
    src = parse_palette(edited(sources__gu__license=license_text, sources__gu__attribution=attribution)).sources["gu"]
    assert src.mode == "commit" and licence_problem(license_text, "commit", attribution) is None


def test_every_committed_palette_source_has_a_committable_licence_in_the_asset_manifest():
    manifest = {a["path"]: a for a in yaml.safe_load((ROOT / "assets" / "MANIFEST.yaml").read_text())["assets"]}
    for path in sorted((ROOT / "palettes").glob("*.yaml")):
        for sid, src in load_palette(path, verify=False).sources.items():
            if src.mode == "commit":
                entry = manifest[src.path]
                assert entry["mode"] == "commit", (path.name, sid)
                assert licence_problem(entry["license"], "commit", src.attribution) is None, (path.name, sid)


@pytest.mark.parametrize("paths, message", [
    ({"sources__gu__sha256": None}, "missing sha256"),
    ({"sources__gu__license": None}, "missing license"),
    ({"sources__gu__sha256": "abc"}, "does not match"),
    ({"sources__gu__mode": "copy"}, "commit or fetch"),
    ({"keys__rhodes__src": "nowhere"}, "not in sources"),
    ({"sax_expr__vibrato__cents": None}, "missing cents"),
    ({"groove__swing": {"hat": 0.6}}, "missing default"),
    ({"groove__swing": {"default": 0.8}}, "outside"),
    ({"groove__humanize": False}, 'write "off"'),
    ({"groove__metric_db__db": [0.0] * 15}, "16 numbers"),
    ({"sax__tenor__home": [50, 62]}, "12 notes"),
    ({"sax__tenor__comfort": [40, 72]}, "inside play"),
    ({"sax_lead": "soprano"}, "not a sax"),
    ({"kits__jazz__lanes__shaker": None}, "missing shaker"),
    ({"kits__jazz__choke": [["ohat", "cowbell"]]}, "not one of"),
    ({"chords__qualities": ["m9", "6/9"]}, "not one of"),
    ({"voicings": ["rootless", "drop3"]}, "not one of"),
    ({"aliases": {"rhodes": "organ"}}, "also be a palette name"),
    ({"aliases": {"clav": "harpsichord"}}, "not a keys, kit or sax name"),
    ({"mix__buses__sax": [["Chorus", {"rate_hz": 1.0}]]}, "not one of"),
    ({"mix__master": [["Compressor", {"knee_db": 3.0}]]}, "unknown key"),
    ({"bed": {"src": "gu", "start_s": 0.0, "xfade_s": 1.0}}, "missing delta_lu"),
    ({"sr": 48000}, "44.1 kHz"),
    ({"frozen": "tomorrow"}, "YYYY-MM-DD"),
    ({"extra": 1}, "unknown key"),
])
def test_schema_errors_name_the_path(paths, message):
    with pytest.raises(PaletteError, match=message):
        parse_palette(edited(**paths))


def test_dev_palette_carries_the_spec_starting_values():
    pal = dev_palette()
    tenor, alto = pal.sax["tenor"], pal.sax["alto"]                         # section 3, Registers
    assert (tenor.play, tenor.comfort, tenor.home, tenor.prog) == ((44, 75), (48, 72), (50, 61), 66)
    assert (alto.play, alto.comfort, alto.home, alto.prog) == ((49, 80), (55, 77), (57, 68), 65)
    assert pal.sax_lead == "tenor"
    assert {k: (v.bank, v.prog) for k, v in pal.keys.items()} == {"rhodes": (0, 4), "organ": (0, 16)}
    assert all(dict(v.vel) == {"X": 96, "x": 80, "g": 52} for v in pal.keys.values())   # section 3, Velocities
    kit = pal.kits["jazz"]
    assert (kit.bass.bank, kit.bass.prog, dict(kit.bass.vel)) == (0, 32, {"X": 104, "x": 88, "g": 64})
    assert kit.choke == (("ohat", "hat"),)
    notes = {lane: {s.note for layer in kit.lanes[lane].layers.values() for s in layer} for lane in LANES}
    assert notes == {"kick": {36}, "snare": {38}, "rim": {37}, "hat": {42}, "ohat": {46}, "shaker": {82}}
    assert all((s.bank, s.prog) == (128, 32) for d in kit.lanes.values() for layer in d.layers.values() for s in layer)
    g = pal.groove
    assert g.swing == pytest.approx(0.55) and str(g.swing) == "11/20" and not g.swing_lanes and not g.offset_ms
    assert g.humanize == "off" and float(g.humanize_ms) == 3.0 and float(g.humanize_vel) == 0.08
    assert dict(g.drum_gain) == {"X": 1.0, "x": 0.71, "g": 0.32}
    assert g.metric_lanes == {"hat", "shaker"}
    assert g.metric_db == (0, -4, -2, -4, -1, -4, -2, -4, 0, -4, -2, -4, -1, -4, -2, -4)
    e = pal.sax_expr
    assert e.enabled and e.timing == "grid" and e.bend_range == 2
    assert (e.bend_every_ms, e.cc11_every_ms, e.gap_ms) == (10, 20, 15)
    assert (e.scoop.min_steps, e.scoop.leap, e.scoop.cents, e.scoop.ms) == (4, 3, -60, 80)
    v = e.vibrato
    assert (v.min_steps, float(v.start), v.fade_ms, v.cents, float(v.rate_hz)) == (8, 0.35, 300, 18, 5.5)
    c = e.cc11
    assert (c.short, c.long_steps, c.long_base, c.long_swell, c.last, float(c.last_share)) == (110, 8, 100, 27, 80, 0.4)
    vel = e.velocity
    assert (vel.base, vel.bar_start, vel.beat3, vel.sixteenth, vel.phrase_start, vel.min, vel.max) == \
        (92, 8, 4, -10, 6, 60, 112)
    lo = pal.loudness
    assert (lo.lufs, lo.tol, lo.iters, lo.ceiling_dbtp) == (-14.0, 0.1, 4, -1.0)
    assert pal.bed is None
    assert dict(pal.mix.bus_target_lu) == {"sax": 0.0, "drums": -1.5, "bass": -3.0, "keys": -6.0}   # section 4
    assert pal.numerals == NUMERALS and pal.qualities == QUALITIES and pal.voicings == VOICINGS


def test_mix_chains_build_in_pedalboard():
    pedalboard = pytest.importorskip("pedalboard")
    pal = dev_palette()
    chains = list(pal.mix.buses.values()) + [pal.mix.master]
    assert [p.name for p in pal.mix.buses["sax"]] == ["HighpassFilter", "PeakFilter", "Compressor", "PeakFilter",
                                                      "Reverb"]
    for chain in chains:
        for plugin in chain:
            assert set(plugin.params) <= PLUGINS[plugin.name]
            built = getattr(pedalboard, plugin.name)(**plugin.params)
            for k, v in plugin.params.items():
                assert getattr(built, k) == pytest.approx(v, rel=1e-6), (plugin.name, k)


def test_vocab_and_aliases_accept_every_default_vocab_fixture():
    pal = dev_palette()
    for p in sorted(FIX.glob("*.ls")):
        default = parse(p.read_text(), mode="wistful")
        dev = parse(p.read_text(), vocab=pal.vocab(), mode="wistful")
        assert default.gate.ok == dev.gate.ok, p.name
        if dev.gate.ok:
            assert dev.ast.keys.instrument in pal.keys and dev.ast.kit in pal.kits
    assert pal.resolve("keys", "Wurli") == "rhodes" and pal.resolve("kits", "dusty") == "jazz"
    assert pal.resolve("sax", "alto") == "alto"
    with pytest.raises(PaletteError):
        pal.resolve("keys", "dusty")
    with pytest.raises(PaletteError):
        pal.resolve("kits", "trap808")


def test_the_soundfont_preset_headers_are_read_exactly(tmp_path):
    gu = dev_palette().source_path("gu")
    presets = sf2_presets(gu)
    assert len(presets) == 287                                   # 288 phdr records, the last is EOP
    assert sorted(p for b, p in presets if b == 128) == [0, 1, 2, 8, 16, 24, 25, 26, 32, 40, 48, 56, 127]
    assert {(0, 66), (0, 65), (0, 4), (0, 16), (0, 32), (128, 32)} <= presets and (7, 4) not in presets
    with pytest.raises(PaletteError, match="not a SoundFont"):
        sf2_presets(DEV)
    cut = tmp_path / "cut.sf2"
    with gu.open("rb") as f:
        cut.write_bytes(f.read(4096))                            # header intact, sample data and presets gone
    with pytest.raises(PaletteError, match="cut short"):
        sf2_presets(cut)
    doc = edited(sources__gu__path=str(cut), sources__gu__sha256=hashlib.sha256(cut.read_bytes()).hexdigest())
    (tmp_path / "cut.yaml").write_text(yaml.safe_dump(doc))
    with pytest.raises(PaletteError, match="source gu: .*cut short"):
        load_palette(tmp_path / "cut.yaml", verify=True, root=Path("/"))


@pytest.mark.parametrize("paths, message", [
    ({f"kits__jazz__lanes__snare__{layer}__0__prog": 33 for layer in ("soft", "med", "hard")},
     "kits.jazz.lanes.snare.hard[0]: bank 128 prog 33"),
    ({"keys__rhodes__bank": 7}, "keys.rhodes: bank 7 prog 4"),
    ({"sax__alto__prog": 127, "sax__alto__bank": 9}, "sax.alto: bank 9 prog 127"),
    ({"kits__jazz__bass__prog": 99, "kits__jazz__bass__bank": 1}, "kits.jazz.bass: bank 1 prog 99")])
def test_a_preset_missing_from_its_soundfont_is_refused_at_load(tmp_path, paths, message):
    # FluidSynth would play another preset in its place (GU drum kit 33 -> kit 0, "Standard"), so the palette
    # would render a sound it does not name.
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(edited(**paths)))
    assert load_palette(path, verify=False)
    with pytest.raises(PaletteError, match=re.escape(message) + ".* not a preset of source gu"):
        load_palette(path, verify=True)


def test_layer_fallback_is_nearest_then_softer():
    doc = edited(kits__jazz__lanes__hat__med=[], kits__jazz__lanes__ohat__soft=[], kits__jazz__lanes__ohat__med=[])
    kit = parse_palette(doc).kits["jazz"]
    hat, ohat = kit.lanes["hat"], kit.lanes["ohat"]
    assert [hat.layer_for(lv) for lv in "Xxg"] == ["hard", "soft", "soft"]
    assert [ohat.layer_for(lv) for lv in "Xxg"] == ["hard", "hard", "hard"]
    with pytest.raises(PaletteError, match="every layer is empty"):
        parse_palette(edited(kits__jazz__lanes__hat__med=[], kits__jazz__lanes__hat__soft=[],
                             kits__jazz__lanes__hat__hard=[]))


# ------------------------------------------------------------------------------- fetch-mode sources
FETCHED = b"a fetch-mode file"


def _fetch_doc(assets_root="$LOFI_ASSETS"):
    """The dev palette with its SoundFont at an absolute path (so any `root` finds it) and one mode-fetch
    source, `kit/one.zip`, resolved through `assets_root`."""
    doc = edited(assets_root=assets_root, sources__gu__path=str(dev_palette().source_path("gu")))
    doc["sources"]["kit"] = {"path": "kit/one.zip", "sha256": hashlib.sha256(FETCHED).hexdigest(),
                             "license": "CC0-1.0", "mode": "fetch"}
    return doc


def _place(folder: Path) -> Path:
    (folder / "kit").mkdir(parents=True, exist_ok=True)
    (folder / "kit" / "one.zip").write_bytes(FETCHED)
    return folder / "kit" / "one.zip"


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_fetch_sources_resolve_under_assets_big_while_lofi_assets_is_unset_or_empty(tmp_path, monkeypatch, value):
    # docs/SHOOTOUT_ASSETS.md, Route: the laptop fetches into <repo>/assets/big/. Nothing sets $LOFI_ASSETS on the
    # laptop, and an empty value once gave the bare relative path kit/one.zip, read from the current directory.
    if value is None:
        monkeypatch.delenv("LOFI_ASSETS", raising=False)
    else:
        monkeypatch.setenv("LOFI_ASSETS", value)
    root, elsewhere = tmp_path / "repo", tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    want = _place(root / "assets" / "big")
    path = tmp_path / "pal.yaml"
    path.write_text(yaml.safe_dump(_fetch_doc()))
    pal = load_palette(path, verify=True, root=root)            # the file is found and matches its sha256
    assert pal.source_path("kit", root) == want
    assert pal.source_path("kit") == ROOT / "assets" / "big" / "kit" / "one.zip"


def test_fetch_sources_follow_lofi_assets_and_a_relative_value_is_relative_to_the_root(tmp_path, monkeypatch):
    root, elsewhere = tmp_path / "repo", tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    pal = parse_palette(_fetch_doc())
    monkeypatch.setenv("LOFI_ASSETS", str(tmp_path / "keep" / "assets"))
    assert pal.source_path("kit", root) == _place(tmp_path / "keep" / "assets")
    assert pal.verify_sources(root) == []
    monkeypatch.setenv("LOFI_ASSETS", "shared/assets")
    assert pal.source_path("kit", root) == root / "shared" / "assets" / "kit" / "one.zip", "never the cwd"
    assert any("missing" in p for p in pal.verify_sources(root))
    assert parse_palette(_fetch_doc("fetched")).source_path("kit", root) == root / "fetched" / "kit" / "one.zip"


def test_an_unset_variable_other_than_lofi_assets_is_an_error(monkeypatch):
    monkeypatch.delenv("LOFI_NOWHERE", raising=False)
    pal = parse_palette(_fetch_doc("$LOFI_NOWHERE/assets"))
    with pytest.raises(PaletteError, match=r"\$LOFI_NOWHERE is unset or empty.*source kit is mode fetch"):
        pal.source_path("kit")
    assert pal.source_path("gu") == dev_palette().source_path("gu"), "mode commit never reads assets_root"
    assert any("LOFI_NOWHERE" in p for p in pal.verify_sources())
