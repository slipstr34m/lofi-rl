"""The frozen palette's generator and check (lofi_rl.freeze.palette; docs/FREEZE.md §3).

Offline and fast, and true before and after the seal: the committed palettes/lofi_rain_v1.yaml parses, a fresh build
from the committed inputs is byte-identical (twice; libyaml on and off; Day 3's levels synthesized from the provenance,
as tests/test_drafts_palette.py synthesizes its inputs), every number traces to its record, all 6 (keys, kit) bands
fold back to the draft's render palettes (and with a synthetic keys offset differ by exactly that offset), the regex is
the frozen grammar's and prompt v2 gives its 9 references. The build refuses another excerpt, the excerpt of
rain_d24, a Day-3 move, a stray slot and a change without Shresth's words; the check refuses an edited file.
Where runs/ holds the three levels files (the laptop, after make pull), `check --levels-day1/2/3` passes on them, and
each provenance number edited in memory (a levels sha256, S*, the targets, `moved`, a t or F of job 1 or job 2, job 2's
instrument, the K2 spread F, what Day 3 measured, the draft it rebuilds) raises its own problem and no other; the cases
past the sha256s rebuild the draft, so they also need the fetched kit archive. No test here reads
gates/GATE_palette.md (tests/test_freeze_gate.py uses synthetic gate texts)."""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from lofi_rl import palette as palette_module
from lofi_rl import prompt_v2
from lofi_rl.drafts import palette as DP
from lofi_rl.freeze import BED, CHANGES, FROZEN, JOBS, LEVELS_DAY3, PALETTE_ID, REGEX_CHARS, REGEX_SHA256
from lofi_rl.freeze import palette as F
from lofi_rl.grammar import loopscript_regex, regex_sha256
from lofi_rl.palette import LEVEL_FIELDS

ROOT = Path(__file__).resolve().parents[1]
L3 = ROOT / LEVELS_DAY3
L1, L2 = (ROOT / "runs" / f"shootout-{JOBS[day]}" / "summary" / "levels.json" for day in ("day1", "day2"))
needs_levels = pytest.mark.skipif(not (L1.is_file() and L2.is_file() and L3.is_file()),
                                  reason="runs/ is pulled to the laptop only (make pull)")
DATE = "2026-11-05"                      # a stand-in date for the seal's one-line change; never his
# offsets whose binary sums miss the exact decimal ones (-5.94 + 0.1 is -5.840000000000001), so the rule shows
OFFSET = {"rule": "keys_offset_db", "by": "Shresth", "date": "2026-11-03", "words": "synthetic words for a test",
          "db": {"rhodes": 0.1, "wurli": 0.2, "piano": 1.1}}


@pytest.fixture(scope="module")
def fp():
    return F.load(FROZEN)


@pytest.fixture(scope="module")
def dp():
    return DP.load(DP.DRAFT_PALETTE)


@pytest.fixture(scope="module")
def inp(fp):
    return F.read_inputs(F.day3_from_provenance(fp.provenance))


def _with(fp, text: str):
    return F.parse_frozen(text, fp.path)


# ------------------------------------------------------------------------------------------- the committed file
def test_the_committed_palette_passes_its_offline_check(fp):
    assert fp.palette.id == PALETTE_ID and fp.provenance["schema"] == "lofi-rl/freeze-provenance/1"
    assert F.offline_problems(fp) == []
    if fp.palette.frozen is None:                # before the seal; after it tests/test_palette_frozen.py pins the date
        assert F.frozen_problems(fp) == []
        assert F.main(["check"]) == 0


def test_two_builds_are_byte_identical_and_are_the_file(fp, inp):
    a = F.build_text(inp, frozen=fp.palette.frozen)
    b = F.build_text(F.read_inputs(F.day3_from_provenance(fp.provenance)), frozen=fp.palette.frozen)
    assert a == b == fp.text == F.rebuild_text(fp)


def test_a_build_from_a_synthesized_day3_levels_file_reproduces_the_file(fp, tmp_path):
    doc, real_sha = F.day3_from_provenance(fp.provenance)
    path = tmp_path / "levels.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    text = F.build(path, frozen=fp.palette.frozen)
    assert text.replace(DP.file_sha256(path), real_sha) == fp.text


def test_libyaml_and_the_pure_python_dumper_write_the_same_bytes(fp):
    body = "".join(line + "\n" for line in fp.text.splitlines() if not line.startswith("#"))
    block = {"freeze_provenance": fp.provenance}
    block_text = "".join(line[len(F.BLOCK):] + "\n" for line in fp.text.splitlines() if line.startswith(F.BLOCK))
    assert yaml.dump(fp.doc, Dumper=yaml.SafeDumper, **DP._DUMP) == body
    assert [ln.rstrip() for ln in yaml.dump(block, Dumper=yaml.SafeDumper, **DP._DUMP).splitlines()] == \
        block_text.splitlines()
    if hasattr(yaml, "CSafeDumper"):
        assert yaml.dump(fp.doc, Dumper=yaml.CSafeDumper, **DP._DUMP) == body
        assert [ln.rstrip() for ln in yaml.dump(block, Dumper=yaml.CSafeDumper, **DP._DUMP).splitlines()] == \
            block_text.splitlines()


@pytest.mark.skipif(not L3.is_file(), reason="runs/ is pulled to the laptop only (make pull)")
def test_the_synthesized_day3_file_is_the_real_one_where_the_build_reads_it(fp):
    real = json.loads(L3.read_text(encoding="utf-8"))
    syn, sha = F.day3_from_provenance(fp.provenance)
    assert DP.file_sha256(L3) == sha
    for k in ("stage", "plan_sha256", "base_sha256", "targets", "S_star", "moved"):
        assert syn[k] == real[k], k
    assert syn["prior"]["sha256"] == real["prior"]["sha256"]
    band = {o for _, o in F.band_options()}
    assert set(syn["variants"]) == band & set(real["variants"])
    for option, v in syn["variants"].items():
        assert {k: real["variants"][option].get(k) for k in v} == v, option
    assert F.rebuild_text(fp, levels_day3=L3) == fp.text


def test_the_seal_changes_exactly_the_frozen_line(fp, inp):
    open_text, sealed = F.build_text(inp, frozen=None), F.build_text(inp, frozen=DATE)
    a, b = open_text.splitlines(), sealed.splitlines()
    assert len(a) == len(b)
    assert [(x, y) for x, y in zip(a, b) if x != y] == [("frozen: null", f"frozen: '{DATE}'")]
    assert F.parse_frozen(sealed).palette.frozen == DATE
    for bad in ("2026-11-31", "2026-11-5", "tomorrow"):
        with pytest.raises(F.FreezeError, match="frozen"):
            F.build_text(inp, frozen=bad)


def test_the_top_level_order_is_palette_py_s():
    assert F.ORDER == palette_module._TOP


# ------------------------------------------------------------------------------------------- content
def test_every_instrument_carries_its_levels_and_no_slot_or_fader_is_left(fp):
    doc = fp.doc
    voices = list(doc["sax"].values()) + list(doc["keys"].values())
    voices += [k for k in doc["kits"].values()] + [k["bass"] for k in doc["kits"].values()]
    assert len(voices) == 8 and all(list(v)[-2:] == list(LEVEL_FIELDS) or list(v)[:2] == list(LEVEL_FIELDS)
                                    for v in voices)
    assert all(list(k)[:2] == ["trim_db", "fader_db"] for k in doc["kits"].values())
    assert doc["mix"]["bus_gain_db"] == {"sax": 0.0, "drums": 0.0, "bass": 0.0, "keys": 0.0}
    assert not any(p[0] == "Gain" for chain in doc["mix"]["buses"].values() for p in chain)
    assert doc["bed"] == {"src": "rain", "start_s": 44.3, "xfade_s": 1.0, "delta_lu": 24.0}
    assert fp.provenance["bed"]["excerpt"] == "e2" and fp.provenance["bed"]["delta_from"] == BED["delta_from"]
    assert fp.provenance["day3"]["job"] == JOBS["day3"] and fp.provenance["day3"]["moved"] == []


def test_the_kit_trim_is_a_field_and_its_lanes_are_the_importer_s(fp, dp):
    studio = fp.doc["kits"]["studio"]
    assert studio["trim_db"] == -14.94 and studio["fader_db"] == 15.26
    assert {lane: v["db"] for lane, v in studio["lanes"].items()} == {
        "kick": 0.0, "snare": 0.0, "rim": 0.0, "hat": 0.0, "ohat": 0.0, "shaker": 2.47}
    # the renderer's sum (sampler.kit_shots: lane db + kit trim) is the draft's own lane db, to the bit
    for lane, v in studio["lanes"].items():
        assert v["db"] + studio["trim_db"] == dp.doc["kits"]["studio"]["lanes"][lane]["db"], lane
    assert DP.canonical_sha256({"lanes": studio["lanes"], "choke": studio["choke"]}) == \
        dp.levels["kits"]["studio"]["measured"]["as"]["kit_map_sha256"]


def test_all_six_bands_fold_back_to_the_draft_render_palettes(fp, dp):
    pairs = [(k, kit) for k in fp.doc["keys"] for kit in fp.doc["kits"]]
    assert len(pairs) == 6
    for keys, kit in pairs:
        want = DP.render_doc(dp, keys, kit)
        got = F.fold(fp.doc, keys, kit)
        assert got.pop("id") == PALETTE_ID and want.pop("id") == f"draft_day2_{keys}_{kit}"
        assert got.pop("frozen") == fp.palette.frozen and want.pop("frozen") is None    # no render reads it
        assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True), (keys, kit)
    assert F.fold_problems(fp, dp) == []


def test_a_sealed_file_passes_the_offline_check(fp, inp):
    # the seal changes only `frozen:`; every offline check holds on the sealed text as on the signed-off one
    sealed = _with(fp, F.build_text(inp, frozen=DATE))
    assert sealed.palette.frozen == DATE and F.offline_problems(sealed) == []


def test_a_synthetic_keys_offset_moves_each_keys_fader_by_exactly_that_offset(fp, dp, inp):
    text = F.build_text(inp, changes=(OFFSET,))
    moved = _with(fp, text)
    assert moved.provenance["rules"]["keys_offset_db"] == {k: v for k, v in OFFSET.items() if k != "rule"}
    assert F.fold_problems(moved, dp) == [] and F.numbers_problems(moved, dp, _dev()) == []
    assert {k: v["fader_db"] for k, v in moved.doc["keys"].items()} == {"rhodes": -5.84, "wurli": -6.235,
                                                                          "piano": -4.96}
    for keys in fp.doc["keys"]:
        for kit in fp.doc["kits"]:
            got, want = F.fold(moved.doc, keys, kit), DP.render_doc(dp, keys, kit)
            assert got["mix"]["bus_gain_db"]["keys"] == F.dec_add(want["mix"]["bus_gain_db"]["keys"],
                                                                  OFFSET["db"][keys])
            got["mix"]["bus_gain_db"]["keys"] = want["mix"]["bus_gain_db"]["keys"]
            got["id"] = want["id"]
            assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True), (keys, kit)
    # the committed file is not that build: the check sees the change it lacks, and the change the file lacks
    assert F.rebuild_text(fp, changes=(OFFSET,)) != fp.text
    assert F.rebuild_text(moved, changes=CHANGES) != moved.text


def _dev() -> dict:
    return yaml.safe_load((ROOT / "palettes" / "dev_gu.yaml").read_text(encoding="utf-8"))


def test_the_regex_and_prompt_v2_are_the_frozen_grammar_s(fp):
    rx = loopscript_regex(fp.palette)
    assert len(rx) == REGEX_CHARS == 3408 and regex_sha256(fp.palette) == REGEX_SHA256
    assert REGEX_SHA256.startswith("5c081828")
    assert len(prompt_v2.reference_texts(fp.palette, strict=True)) == 9
    assert F.grammar_problems(fp.palette) == []


def test_the_rain_source_is_the_one_read_rain_builds(tmp_path):
    from lofi_rl.shootout.palettes import read_rain
    record = json.loads(F.RAIN.read_text(encoding="utf-8"))
    stand_in = tmp_path / "assets" / record["source"]["asset_path"]
    stand_in.parent.mkdir(parents=True)
    stand_in.write_bytes(b"a stand-in for the rain file, never the real one\n")
    record["source"]["sha256"] = DP.file_sha256(stand_in)
    path = tmp_path / "rain_excerpts.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert read_rain(path, tmp_path / "assets")["palette_source"] == F.rain_source(record)


def test_vocab_names_the_picks(fp):
    v = F.vocabulary(fp)
    assert v["keys"] == {"rhodes": "gu_tine_ep", "wurli": "gs_wurli_ep200", "piano": "vcsl_upright_knight"}
    assert v["kits"] == {"studio": {"kit": "kit_virtuosity", "bass": "bass_vsco2_pizz"},
                         "jazz": {"kit": "kit_gu_jazz", "bass": "bass_fp_finger_yr"}}
    assert v["sax"] == {"alto": "sax_mtg_alto"} and v["voicings"] == ["rootless", "shell", "spread", "quartal"]


# ------------------------------------------------------------------------------------------- the build refuses
@pytest.mark.parametrize("bed_from, match", [
    ({"excerpt_from": "rain_d24", "delta_from": "rain_d24"}, "W1"),       # the pitfall: rain_d24 sets e1 too
    ({"excerpt_from": "excerpt_e2", "delta_from": "excerpt_e2"}, "W2"),
    ({"excerpt_from": "excerpt_e9", "delta_from": "rain_d24"}, "W1"),
])
def test_the_bed_takes_the_excerpt_from_w1_and_only_the_level_from_w2(inp, bed_from, match):
    with pytest.raises(F.FreezeError, match=match):
        F.build_text(inp, bed_from=bed_from)


def test_another_excerpt_builds_but_fails_the_check(fp, inp, dp):
    other = _with(fp, F.build_text(inp, bed_from={"excerpt_from": "excerpt_e3", "delta_from": "rain_d24"}))
    assert other.doc["bed"]["start_s"] == 77.8
    assert any("bed" in p for p in F.picks_problems(other))
    assert any("bed comes from" in p for p in F.numbers_problems(other, dp, _dev()))
    assert F.rebuild_text(other) != other.text


def test_the_bed_applied_variant_by_variant_freezes_e1_and_the_check_refuses_it(fp, dp):
    # applying excerpt_e2 then rain_d24 in order (the shootout's bed_block) would play e1 at 24
    text = fp.text.replace("bed: {src: rain, start_s: 44.3, xfade_s: 1.0, delta_lu: 24.0}",
                           "bed: {src: rain, start_s: 150.4, xfade_s: 1.0, delta_lu: 24.0}")
    wrong = _with(fp, text)
    assert wrong.doc["bed"]["start_s"] == 150.4
    assert any("bed" in p for p in F.picks_problems(wrong))
    assert any("bed" in p for p in F.numbers_problems(wrong, dp, _dev()))
    assert F.offline_problems(wrong) != []


def _day3(inp, **edit):
    doc = copy.deepcopy(inp.day3)
    for path, value in edit.items():
        node, keys = doc, path.split("__")
        for k in keys[:-1]:
            node = node[k]
        if value is None:
            del node[keys[-1]]
        else:
            node[keys[-1]] = value
    return dataclasses.replace(inp, day3=doc)


@pytest.mark.parametrize("edit, match", [
    ({"variants__gu_tine_ep__F": -6.05}, "gu_tine_ep moved"),
    ({"variants__kit_virtuosity__t": -14.93}, "kit_virtuosity moved"),
    ({"variants__kit_gu_jazz__F": 15.2}, "kit_gu_jazz moved"),
    ({"moved": ["sax_mtg_alto"]}, "moved"),
    ({"S_star": -31.9}, "S\\* moved"),
    ({"variants__sax_mtg_alto__source_sha256": "0" * 64}, "another instrument"),
    ({"variants__kit_virtuosity__kit_map_sha256": "0" * 64}, "another instrument"),
    ({"variants__bass_vsco2_pizz": None}, "did not measure"),
    ({"stage": "day2"}, "stage day3"),
    ({"prior__sha256": "0" * 64}, "prior"),
])
def test_a_day3_move_stops_the_build(inp, edit, match):
    with pytest.raises(F.FreezeError, match=match):
        F.build_text(_day3(inp, **edit))


def test_a_day3_file_from_another_plan_is_refused(fp):
    doc, sha = F.day3_from_provenance(fp.provenance)
    doc["plan_sha256"] = "0" * 64
    with pytest.raises(F.FreezeError, match="plan"):
        F.read_inputs((doc, sha))


def _draft(inp, old: str, new: str):
    assert inp.draft.text.count(old) == 1, old
    return dataclasses.replace(inp, draft=DP.parse_draft(inp.draft.text.replace(old, new), inp.draft.path))


def test_a_stray_slot_stops_the_build(inp):
    drums = "    drums:\n    - - HighpassFilter\n      - {cutoff_frequency_hz: 30.0}\n"
    stray = _draft(inp, drums, "    drums:\n    - - Gain\n      - {gain_db: 0.0}\n    - - HighpassFilter\n"
                               "      - {cutoff_frequency_hz: 30.0}\n")
    with pytest.raises(F.FreezeError, match="drums: a Gain is left"):
        F.build_text(stray)
    second = _draft(inp, "    sax_warm:\n    - - HighShelfFilter\n",
                    "    sax_warm:\n    - - Gain\n      - {gain_db: 1.0}\n    - - HighShelfFilter\n")
    with pytest.raises(F.FreezeError, match="sax_warm: a Gain is left"):
        F.build_text(second)


def test_a_stray_slot_or_a_fader_left_in_the_frozen_file_fails_the_check(fp, dp):
    head = "    keys:\n    - - HighpassFilter\n      - {cutoff_frequency_hz: 120.0}\n"
    assert fp.text.count(head) == 1
    slot = _with(fp, fp.text.replace(head, "    keys:\n    - - Gain\n      - {gain_db: 0.0}\n    - - HighpassFilter\n"
                                            "      - {cutoff_frequency_hz: 120.0}\n"))
    assert any("mix.buses.keys is not dev_gu's chain" in p for p in F.numbers_problems(slot, dp, _dev()))
    gains = "bus_gain_db: {sax: 0.0, drums: 0.0, bass: 0.0, keys: 0.0}"
    fader = _with(fp, fp.text.replace(gains, gains.replace("keys: 0.0", "keys: 0.5")))
    assert any("bus_gain_db" in p for p in F.numbers_problems(fader, dp, _dev()))
    assert any("fold rhodes/studio" in p for p in F.fold_problems(fader, dp))
    for edited in (slot, fader):
        assert F.offline_problems(edited) != []


def test_an_edited_number_fails_the_check(fp, dp):
    wurli = "    trim_db: -3.22\n    fader_db: -6.435\n"
    assert fp.text.count(wurli) == 1
    rounded = _with(fp, fp.text.replace(wurli, "    trim_db: -3.22\n    fader_db: -6.44\n"))
    assert any(p.startswith("keys.wurli") for p in F.numbers_problems(rounded, dp, _dev()))
    assert any("fold wurli" in p for p in F.fold_problems(rounded, dp))
    day3 = F.BLOCK + "      kit_gu_jazz: {t: 0.0, F: 15.21}\n"          # provenance: day3.measured
    assert fp.text.count(day3) == 1
    told = _with(fp, fp.text.replace(day3, day3.replace("15.21", "15.2")))
    assert any("Day 3 measured" in p for p in F.numbers_problems(told, dp, _dev()))
    with pytest.raises(F.FreezeError, match="kit_gu_jazz moved"):          # its Day 3, rebuilt, is a move
        F.rebuild_text(told)
    assert any(p.startswith("rebuild: kit_gu_jazz moved") for p in F.offline_problems(told))


@pytest.mark.parametrize("change, match", [
    ({"rule": "keys_offset_db", "db": {"rhodes": 1.0, "wurli": 1.0, "piano": 1.0}}, "by, date, words"),
    ({**OFFSET, "by": None}, "by"),
    ({**OFFSET, "words": ""}, "words"),
    ({**OFFSET, "date": None}, "date"),
    ({**OFFSET, "by": "Claude"}, "only Shresth"),
    ({**OFFSET, "date": "2026-13-01"}, "date"),
    ({**OFFSET, "rule": "sax_offset_db"}, "not one the build knows"),
    ({**OFFSET, "db": {"rhodes": 1.0}}, "each keys name"),
    ({**OFFSET, "db": {"rhodes": 1.0, "wurli": True, "piano": 0.0}}, "wurli"),
    ({**OFFSET, "db": {"rhodes": 41.0, "wurli": 0.0, "piano": 0.0}}, "rhodes"),
])
def test_a_change_needs_shresths_words_and_a_known_rule(inp, change, match):
    with pytest.raises(F.FreezeError, match=match):
        F.build_text(inp, changes=(change,))


def test_a_draft_that_fails_its_own_check_is_refused(inp):
    bad = _draft(inp, "#|       fader_db: -6.435\n", "#|       fader_db: -6.43\n")
    with pytest.raises(F.FreezeError, match="fails its own check"):
        F.build_text(bad)


def test_frozen_must_be_null_or_with_sealed_the_approved_date(fp, inp):
    sealed = _with(fp, F.build_text(inp, frozen=DATE))
    assert F.frozen_problems(sealed) and F.frozen_problems(_with(fp, F.build_text(inp))) == []
    gate = "## Approval\n\nEntered from Shresth's chat line, verbatim.\nAPPROVED by Shresth Rana " + DATE + "\n"
    assert F.frozen_problems(sealed, sealed=True, gate_text=gate) == []
    assert F.frozen_problems(sealed, sealed=True, gate_text=gate.replace(DATE, "2026-11-06"))
    assert F.frozen_problems(sealed, sealed=True, gate_text="## Approval\n")


# ------------------------------------------------------------------------------------------- with the levels files
B = F.BLOCK
REBUILT = f"{F.label(FROZEN)} is not what build writes with --levels-day3 {LEVELS_DAY3}"
# what levels_problems prints for a levels file whose S* or targets are not the record's: the file's own values
S_STAR = ("S* -31.924651028769418 or targets {'sax': 0.0, 'keys': -6.0, 'bass': -3.0, 'drums': -1.5} are not the "
          "record's")


@pytest.fixture(scope="module")
def kit_archives(fp):
    """The levels check rebuilds the draft (drafts.palette build), which reads each §7.4 kit archive and its blocks
    from the fetched assets ($LOFI_ASSETS, or assets/big): skip where they are not fetched."""
    from lofi_rl.shootout.plan import load_plan
    plan, assets = load_plan(DP.PLAN), fp.palette.assets_dir(ROOT)
    need = [assets / c.archive[k] for kit, _bass in DP.KITS.values() if not (c := plan.candidates[kit]).base
            for k in ("path", "blocks")]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        pytest.skip(f"the draft rebuild reads {missing}, which are not fetched here")


def test_the_levels_files_go_together(capsys):
    capsys.readouterr()
    assert F.main(["check", "--levels-day1", "x.json", "--levels-day3", "y.json"]) == 2
    assert capsys.readouterr().err == "error: --levels-day1, --levels-day2 and --levels-day3 go together\n"


@needs_levels
def test_the_levels_check_passes_on_the_real_levels_files(fp, inp, kit_archives, tmp_path, capsys):
    assert F.levels_problems(fp, L1, L2, L3) == []
    palette = FROZEN
    if fp.palette.frozen is not None:            # sealed: check its signed-off text (only `frozen:` differs)
        palette = tmp_path / FROZEN.name
        palette.write_text(F.build_text(inp, frozen=None), encoding="utf-8")
    capsys.readouterr()
    assert F.main(["check", "--palette", str(palette), "--levels-day1", str(L1), "--levels-day2", str(L2),
                   "--levels-day3", str(L3)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("ok: ") and (f"levels {F.label(L1)}, {F.label(L2)}, {LEVELS_DAY3} equal; draft rebuild "
                                       "byte-identical; rebuild with --levels-day3 byte-identical") in out


@needs_levels
def test_the_levels_check_names_a_swapped_levels_file(capsys):
    capsys.readouterr()
    assert F.main(["check", "--levels-day1", str(L2), "--levels-day2", str(L1), "--levels-day3", str(L3)]) == 3
    err = capsys.readouterr().err
    sha1, sha2 = DP.file_sha256(L1)[:12], DP.file_sha256(L2)[:12]
    assert f"problem: --levels-day1 {F.label(L2)} has sha256 {sha2}..., the provenance {sha1}...\n" in err
    assert f"problem: --levels-day2 {F.label(L1)} has sha256 {sha1}..., the provenance {sha2}...\n" in err


# One provenance number edited at a time, so that it contradicts the real levels files; each edit raises exactly its own
# problems. A recorded sha256 that is not the file's stops the comparison there, so that problem comes alone.
@needs_levels
@pytest.mark.parametrize("old, new, want", [
    pytest.param("day1: {job: 13635646, sha256: 9cf7", "day1: {job: 13635646, sha256: 0cf7",
                 f"--levels-day1 {F.label(L1)} has sha256 9cf78680b3d0..., the provenance 0cf78680b3d0...",
                 id="day1-sha256"),
    pytest.param("levels_sha256: b313", "levels_sha256: 0313",
                 f"--levels-day3 {LEVELS_DAY3} has sha256 b313ff9ab98f..., the provenance 0313ff9ab98f...",
                 id="day3-sha256"),
])
def test_the_levels_check_stops_at_a_levels_file_that_is_not_the_recorded_one(fp, old, new, want):
    assert fp.text.count(old) == 1, old
    assert F.levels_problems(_with(fp, fp.text.replace(old, new)), L1, L2, L3) == [want]


# Any other edit also raises REBUILT, since the build never writes an edited provenance.
@needs_levels
@pytest.mark.parametrize("old, new, want", [
    pytest.param(f"{B}    schema: lofi-rl/draft-levels/1\n{B}    S_star: -31.924651028769418\n",
                 f"{B}    schema: lofi-rl/draft-levels/1\n{B}    S_star: -31.92465102876942\n",
                 [f"day1: {S_STAR}", f"day2: {S_STAR}", f"day3: {S_STAR}", REBUILT], id="S*"),
    pytest.param(f"{B}    targets: {{sax: 0.0, keys: -6.0,", f"{B}    targets: {{sax: 0.0, keys: -6.5,",
                 [f"day1: {S_STAR}", f"day2: {S_STAR}", f"day3: {S_STAR}", REBUILT], id="targets"),
    pytest.param(f"{B}    moved: []\n", f"{B}    moved: [x]\n", ["day3: moved [], the provenance ['x']", REBUILT],
                 id="moved"),
    pytest.param(f"{B}          day1: {{t: -3.22, F: -6.65}}\n", f"{B}          day1: {{t: -3.22, F: -6.66}}\n",
                 ["keys.wurli: the record's day1 {'t': -3.22, 'F': -6.66} is not day1's t -3.22 F -6.65", REBUILT],
                 id="wurli-day1-F"),
    pytest.param(f"{B}            day2: {{t: 34.97, F: 5.09}}\n", f"{B}            day2: {{t: 34.96, F: 5.09}}\n",
                 ["kits.studio.bass: the record's day2 {'t': 34.96, 'F': 5.09} is not day2's t 34.97 F 5.09", REBUILT],
                 id="studio-bass-day2-t"),
    pytest.param("source_sha256: aab58d9c", "source_sha256: 0ab58d9c",
                 ["keys.wurli: the record's instrument {'source': 'gsullivan_wurlitzer', 'source_sha256': "
                  "'0ab58d9cc609c23e72b2719e7b7b7cec6a7d7dab0264639d362c97fc8a7cac82', 'bank': 0, 'prog': 0} is not "
                  "what job 2 measured", REBUILT], id="wurli-instrument"),
    pytest.param(f"{B}        spread_F: -6.22\n", f"{B}        spread_F: -6.20\n",
                 ["keys.wurli: spread_F -6.2 is not job 2's K2 median -6.22", REBUILT], id="wurli-K2-spread-F"),
    pytest.param(f"{B}      gu_tine_ep: {{t: 0.0, F: -6.04}}\n", f"{B}      gu_tine_ep: {{t: 0.0, F: -6.05}}\n",
                 ["keys.rhodes: the provenance's Day-3 measure {'t': 0.0, 'F': -6.05} is not day3's "
                  "{'t': 0.0, 'F': -6.04}", REBUILT], id="day3-F"),
    pytest.param(f"{B}      kit_virtuosity: {{t: -14.94, F: 15.26}}\n",
                 f"{B}      kit_virtuosity: {{t: -14.93, F: 15.26}}\n",
                 ["kits.studio: the provenance's Day-3 measure {'t': -14.93, 'F': 15.26} is not day3's "
                  "{'t': -14.94, 'F': 15.26}", REBUILT], id="day3-t"),
    pytest.param(f"{B}      sax_mtg_alto: {{t: 6.63, F: 2.55}}\n", "",
                 ["sax.alto: the provenance's Day-3 measure None is not day3's {'t': 6.63, 'F': 2.55}", REBUILT],
                 id="a-day3-measure-left-out"),
    # Day 3 measured 5 of the 8 instruments: the provenance cannot claim a Day-3 measure of the wurli
    pytest.param(f"{B}      sax_mtg_alto: {{t: 6.63, F: 2.55}}\n",
                 f"{B}      sax_mtg_alto: {{t: 6.63, F: 2.55}}\n{B}      gs_wurli_ep200: {{t: -3.22, F: -6.65}}\n",
                 ["keys.wurli: the provenance's Day-3 measure {'t': -3.22, 'F': -6.65} is not day3's None", REBUILT],
                 id="a-day3-measure-day3-never-took"),
])
def test_the_levels_check_names_each_provenance_number_the_levels_files_contradict(fp, kit_archives, old, new, want):
    assert fp.text.count(old) == 1, old
    assert F.levels_problems(_with(fp, fp.text.replace(old, new)), L1, L2, L3) == want


@needs_levels
def test_the_levels_check_rebuilds_the_draft_from_jobs_1_and_2(fp, kit_archives, tmp_path):
    draft = tmp_path / "draft_day2.yaml"
    draft.write_text(DP.DRAFT_PALETTE.read_text(encoding="utf-8") + "# a line drafts.palette build never writes\n",
                     encoding="utf-8")
    old = "draft: {path: palettes/draft_day2.yaml,"
    assert fp.text.count(old) == 1
    other = _with(fp, fp.text.replace(old, f"draft: {{path: {json.dumps(str(draft))},"))
    assert F.levels_problems(other, L1, L2, L3) == [
        f"{draft} is not what drafts.palette build writes from these files", REBUILT]
