"""The scorer's wire protocol (src/lofi_rl/scoring/protocol.py; docs/SCORER.md §3): clip ids as taste_pipeline.md §2
and lofi_rl.shootout.render.clip_id define them, request round trips and validation, response checks, versions."""
import copy
import dataclasses
import json
import math

import pytest

from helpers_scoring import BRIEF_A, BRIEF_B, PALETTE, fixture_text
from lofi_rl.brief import Brief
from lofi_rl.dsl import canon, canon_sha, parse
from lofi_rl.palette import load_palette
from lofi_rl.scoring import protocol
from lofi_rl.scoring.protocol import ProtocolError
from lofi_rl.scoring.render_pool import render_config

PAL = load_palette(PALETTE, verify=False)
CFG = render_config()


def _items(n=3):
    return [protocol.item_dict(i, fixture_text("example.ls"), BRIEF_A if i % 2 else BRIEF_B, f"596:{i}") for i in range(n)]


def _body(**over):
    body = protocol.score_request(_items(), palette_sha256=PAL.sha256, step=4)
    body.update(over)
    return body


# ------------------------------------------------------------------------------------------ clip ids
def test_clip_id_is_the_shootout_formula_of_taste_pipeline_section_2():
    from lofi_rl.shootout.render import clip_id as shootout_clip_id
    ast = parse(fixture_text("example.ls"), vocab=PAL.vocab(), mode=BRIEF_A.mode).ast
    manifest = {"schema": CFG["schema"], "chain": CFG["chain"], "loop": {"canon_sha256": canon_sha(ast)},
                "brief": {"bpm": BRIEF_A.bpm, "tonic": BRIEF_A.tonic, "mode": BRIEF_A.mode, "energy": BRIEF_A.energy},
                "palette": {"id": PAL.id, "sha256": PAL.sha256}}
    cid = protocol.clip_id(canon_sha(ast), BRIEF_A, PAL.id, PAL.sha256, CFG)
    assert cid == shootout_clip_id(manifest)
    assert len(cid) == 17 and cid[0] == "c" and int(cid[1:], 16) >= 0


def test_the_render_config_is_the_one_a_real_manifest_carries():
    from lofi_rl.render import pipeline
    assert CFG["schema"] == pipeline.SCHEMA and CFG["chain"]["cycles"] == pipeline.CYCLES
    assert CFG["chain"]["midi_cycles"] == pipeline.PREROLL + pipeline.CYCLES
    assert protocol.render_version(CFG) == "render-cfg-" + protocol.render_cfg_sha8(CFG)


def test_every_part_of_the_clip_id_matters_and_the_scene_does_not():
    sha = canon_sha(parse(fixture_text("example.ls"), vocab=PAL.vocab()).ast)
    base = protocol.clip_id(sha, BRIEF_A, PAL.id, PAL.sha256, CFG)
    others = [protocol.clip_id("0" * 64, BRIEF_A, PAL.id, PAL.sha256, CFG),
              protocol.clip_id(sha, dataclasses.replace(BRIEF_A, bpm=83), PAL.id, PAL.sha256, CFG),
              protocol.clip_id(sha, dataclasses.replace(BRIEF_A, tonic=3), PAL.id, PAL.sha256, CFG),
              protocol.clip_id(sha, dataclasses.replace(BRIEF_A, mood="hopeful"), PAL.id, PAL.sha256, CFG),
              protocol.clip_id(sha, dataclasses.replace(BRIEF_A, energy="high"), PAL.id, PAL.sha256, CFG),
              protocol.clip_id(sha, BRIEF_A, "other", PAL.sha256, CFG),
              protocol.clip_id(sha, BRIEF_A, PAL.id, "f" * 64, CFG),
              protocol.clip_id(sha, BRIEF_A, PAL.id, PAL.sha256, {**CFG, "schema": "lofi-rl/render/2"}),
              protocol.raw_clip_id(fixture_text("example.ls"), BRIEF_A, PAL.id, PAL.sha256, CFG)]
    assert len({base, *others}) == len(others) + 1
    assert protocol.clip_id(sha, BRIEF_A, PAL.id, PAL.sha256, CFG) == base, "a pure function"


def test_texts_with_one_canonical_form_share_a_clip_id():
    a, b = fixture_text("example.ls"), fixture_text("example_leniency.ls")
    assert a != b
    sa, sb = (canon_sha(parse(t, vocab=PAL.vocab(), mode="dorian").ast) for t in (a, b))
    assert sa == sb and protocol.clip_id(sa, BRIEF_A, PAL.id, PAL.sha256, CFG) == \
        protocol.clip_id(sb, BRIEF_A, PAL.id, PAL.sha256, CFG)
    assert canon(parse(a, vocab=PAL.vocab()).ast) == canon(parse(b, vocab=PAL.vocab()).ast)


# ------------------------------------------------------------------------------------------ requests
def test_a_score_request_round_trips_through_json():
    body = json.loads(json.dumps(_body(expect={"rm_ver": "null"})))
    req = protocol.parse_score_request(body)
    assert req.palette_sha256 == PAL.sha256 and req.step == 4 and dict(req.expect) == {"rm_ver": "null"}
    assert [it.i for it in req.items] == [0, 1, 2] and req.items[1].brief == BRIEF_A
    assert req.items[0].brief == BRIEF_B and req.items[2].brief_id == "596:2"
    assert req.items[0].text == fixture_text("example.ls")
    assert dict(protocol.parse_score_request({**body, "expect": None}).expect) == {}, "null: nothing pinned"
    r = protocol.parse_render_request(json.loads(json.dumps(
        protocol.render_request(_items(1), palette_sha256=PAL.sha256, out_dir="/tmp/x", flac=True))))
    assert r.out_dir == "/tmp/x" and r.flac is True and len(r.items) == 1


def _broken(path, value):
    body = copy.deepcopy(_body())
    *head, last = path
    node = body
    for k in head:
        node = node[k]
    if value is KeyError:
        del node[last]
    else:
        node[last] = value
    return body


@pytest.mark.parametrize("path, value, match", [
    (("protocol",), "lofi-scorer/0", "speaks"),
    (("protocol",), KeyError, "lacks"),
    (("palette_sha256",), "abc", "64 lowercase hex"),
    (("items",), "nope", "must be a list"),
    (("step",), True, "integer"),
    (("expect",), {"rm_ver": 3}, "version strings"),
    (("expect",), {"colour": "x"}, "unknown keys"),
    (("expect",), [], "must be a JSON object"),          # falsy, but not null: never read as "nothing pinned"
    (("expect",), 0, "must be a JSON object"),
    (("expect",), "", "must be a JSON object"),
    (("expect",), False, "must be a JSON object"),
    (("surprise",), 1, "unknown keys"),
    (("items", 0, "i"), -1, ">= 0"),
    (("items", 0, "i"), 1, "repeated"),
    (("items", 0, "i"), False, "integer"),
    (("items", 0, "text"), 7, "string"),
    (("items", 0, "text"), "x" * (protocol.MAX_TEXT_CHARS + 1), "characters"),
    (("items", 0, "extra"), 1, "unknown keys"),
    (("items", 0, "brief_id"), 12, "brief_id"),
    (("items", 0, "brief_id"), "", "brief_id"),
    (("items", 0, "brief", "mood"), "angry", "mood"),
    (("items", 0, "brief", "bpm"), 120, "bpm"),
    (("items", 0, "brief", "scene"), "beach", "scene"),
    (("items", 0, "brief", "tonic"), KeyError, "lacks"),
    (("items", 0, "brief", "key"), "D", "unknown keys"),
])
def test_a_malformed_score_request_is_a_protocol_error(path, value, match):
    with pytest.raises(ProtocolError, match=match):
        protocol.parse_score_request(_broken(path, value))


def test_too_many_items_and_bad_render_options_are_refused():
    many = [protocol.item_dict(i, "x", BRIEF_A) for i in range(protocol.MAX_ITEMS + 1)]
    with pytest.raises(ProtocolError, match="at most"):
        protocol.parse_score_request({"protocol": protocol.PROTOCOL, "palette_sha256": PAL.sha256, "items": many})
    with pytest.raises(ProtocolError, match="absolute"):
        protocol.render_request(_items(1), palette_sha256=PAL.sha256, out_dir="relative/dir")
    assert protocol.parse_score_request({"protocol": protocol.PROTOCOL, "palette_sha256": PAL.sha256,
                                         "items": []}).items == ()


def test_bodies_with_nan_or_bad_bytes_are_refused_and_nan_is_never_sent():
    with pytest.raises(ProtocolError, match="NaN"):
        protocol.decode_json(b'{"a": NaN}')
    with pytest.raises(ProtocolError, match="UTF-8"):
        protocol.decode_json(b"\xff\xfe")
    with pytest.raises(ValueError):
        protocol.encode_json({"G": math.nan})
    assert protocol.decode_json(protocol.encode_json({"x": [1, 2.5, "é"]})) == {"x": [1, 2.5, "é"]}


# ------------------------------------------------------------------------------------------ responses
VERS = {"reward_ver": protocol.REWARD_VER, "feat_ver": "f", "rm_ver": "null", "render_ver": "r", "calib_ver": "c"}
TERMS = {k: False for k in protocol.TERM_FLAGS}


def _item(i, **over):
    it = {"i": i, "clip_id": "c" + "0" * 16, "hard": 1, "G": 0.5, "gate": None,
          "r": {"ce": 0.0, "theme": 0.4, "taste": 0.0}, "sig": {k: 1.0 for k in protocol.SIG_KEYS},
          "diag": {**{k: None for k in protocol.DIAG_KEYS}, "disabled": ["ce"], "cache": "miss"}, "t_ms": {}}
    it.update(over)
    return it


def _resp(items):
    return {"protocol": protocol.PROTOCOL, **VERS, "terms": TERMS, "items": items}


def test_a_valid_response_passes_and_each_defect_is_caught():
    sent = [{"i": 3}, {"i": 7}]
    ok = _resp([_item(3), _item(7, hard=0, G=0.0, gate="G7", sig={k: None for k in protocol.SIG_KEYS})])
    assert protocol.check_score_response(ok, sent, VERS)["items"][1]["gate"] == "G7"
    bad = [
        (_resp([_item(3)]), "1 items for 2"),
        (_resp([_item(7), _item(3)]), "was sent there"),
        (_resp([_item(3, G=1.5), _item(7)]), "outside"),
        (_resp([_item(3, hard=0, gate="G7"), _item(7)]), "hard = 0"),
        (_resp([_item(3, gate="G7"), _item(7)]), "gate"),
        (_resp([_item(3, hard=True), _item(7)]), "0 or 1"),
        (_resp([_item(3, r={"ce": 0.0, "theme": 0.4}), _item(7)]), "exactly"),
        (_resp([_item(3, sig={**_item(3)["sig"], "pq": math.inf}), _item(7)]), "finite"),
        (_resp([_item(3, diag={k: None for k in protocol.DIAG_KEYS}), _item(7)]), "disabled"),
        (_resp([_item(3, clip_id="x1"), _item(7)]), "16 hex"),
        ({**_resp([_item(3), _item(7)]), "rm_ver": "rm-2"}, "pinned"),
        ({**_resp([_item(3), _item(7)]), "terms": {"ce": True}}, "terms"),
        ({**_resp([_item(3), _item(7)]), "protocol": "x"}, "speaks"),
    ]
    for doc, match in bad:
        with pytest.raises(ProtocolError, match=match):
            protocol.check_score_response(doc, sent, VERS)


def test_health_needs_ok_versions_terms_and_the_palette():
    h = {"protocol": protocol.PROTOCOL, "ok": True, "versions": VERS, "terms": TERMS, "palette_sha256": PAL.sha256}
    assert protocol.check_health(h) == h
    for over, match in (({"ok": False}, "ok"), ({"versions": {}}, "version"), ({"palette_sha256": None}, "palette"),
                        ({"protocol": "x"}, "speaks")):
        with pytest.raises(ProtocolError, match=match):
            protocol.check_health({**h, **over})


def test_versions_are_checked_against_the_pinned_ones():
    assert protocol.check_versions(VERS, None, "x") == VERS
    assert protocol.check_versions(VERS, {"rm_ver": "null"}, "x") == VERS
    with pytest.raises(ProtocolError, match="feat_ver"):
        protocol.check_versions(VERS, {"feat_ver": "g"}, "x")
    with pytest.raises(ProtocolError, match="lacks"):
        protocol.check_versions({k: v for k, v in VERS.items() if k != "rm_ver"}, None, "x")


def test_brief_dict_round_trips():
    b = Brief(energy="high", mood="melancholy", bpm=61, tonic=11)
    assert protocol._brief(protocol.brief_dict(b), "b") == b
