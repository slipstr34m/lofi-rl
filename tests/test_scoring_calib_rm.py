"""Calibration files and the RM directory (src/lofi_rl/scoring/calib.py, rm_api.py; docs/SCORER.md §5-6): a missing
file disables its term, a malformed or mismatched one is an error that stops the server (exit 2) before it binds."""
import json

import pytest

from helpers_scoring import short_sock_dir
from lofi_rl.scoring import calib, rm_api, server, theme

SHA = theme.load_config().sha256


def _w(d, name, doc):
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------ calibration
def test_no_directory_or_an_empty_one_disables_everything(tmp_path):
    for d in (None, tmp_path / "absent", tmp_path):
        c = calib.load(d, theme_sha256=SHA)
        assert (c.pq, c.theme, c.ce, c.ce_validated) == (None, None, None, False)
        f = c.flags()
        assert {k: f[k]["status"] for k in ("pq", "theme", "ce")} == {"pq": "missing", "theme": "missing",
                                                                       "ce": "missing"}
        assert f["pq"]["effect"] == "PQ factor 1" and f["theme"]["effect"] == "r_theme 0"


def test_valid_files_load_with_their_hashes(tmp_path):
    _w(tmp_path, "pq.json", {"tau_pq": 6.25, "n": 41, "source": "triage of 300", "date": "2026-11-12"})
    _w(tmp_path, "theme.json", {"med": -0.012, "mad": 0.018, "theme_sha256": SHA, "feat_ver": "feat1..."})
    _w(tmp_path, "ce.json", {"agree": 0.58, "lower95": 0.52, "n_pairs": 512, "med": 5.9, "mad": 0.35,
                             "validated": True})
    c = calib.load(tmp_path, theme_sha256=SHA)
    assert c.pq.tau_pq == 6.25 and c.pq.info == {"date": "2026-11-12", "n": 41, "source": "triage of 300"}
    assert (c.theme.med, c.theme.mad) == (-0.012, 0.018) and len(c.theme.sha256) == 64
    assert c.ce_validated and (c.ce.med, c.ce.mad) == (5.9, 0.35)


@pytest.mark.parametrize("agree, lower, want", [(0.55, 0.5001, True), (0.549, 0.6, False), (0.7, 0.50, False)])
def test_ce_is_validated_by_its_numbers(tmp_path, agree, lower, want):
    doc = {"agree": agree, "lower95": lower, "n_pairs": 400, **({"med": 5.0, "mad": 1.0} if want else {})}
    _w(tmp_path, "ce.json", doc)
    assert calib.load(tmp_path, theme_sha256=SHA).ce_validated is want


@pytest.mark.parametrize("name, doc, match", [
    ("pq.json", "{not json", "JSON"),
    ("pq.json", [1, 2], "object"),
    ("pq.json", {}, "lacks"),
    ("pq.json", {"tau_pq": 6.0, "colour": "red"}, "unknown"),
    ("pq.json", {"tau_pq": "6"}, "finite"),
    ("pq.json", {"tau_pq": 11.0}, "range"),
    ("theme.json", {"med": 0.0, "mad": 0.0, "theme_sha256": SHA}, "range"),
    ("theme.json", {"med": 0.0, "mad": 0.1, "theme_sha256": "0" * 64}, "prompt file"),
    ("ce.json", {"agree": 0.6, "lower95": 0.55, "n_pairs": 400}, "needs med and mad"),
    ("ce.json", {"agree": 0.5, "lower95": 0.45, "n_pairs": 400, "validated": True}, "give False"),
    ("ce.json", {"agree": 0.6, "lower95": 0.55, "n_pairs": -1}, "count"),
])
def test_a_malformed_calibration_is_an_error_not_a_missing_one(tmp_path, name, doc, match):
    _w(tmp_path, name, doc)
    with pytest.raises(calib.CalibError, match=match):
        calib.load(tmp_path, theme_sha256=SHA)


def test_a_calibration_path_that_is_a_file_is_an_error(tmp_path):
    f = _w(tmp_path, "x", "{}")
    with pytest.raises(calib.CalibError, match="not a directory"):
        calib.load(f, theme_sha256=SHA)


# ------------------------------------------------------------------------------------------ the RM
FACTORY = "lofi_rl.scoring.fakes:load_fake_rm"


def test_no_rm_directory_or_no_rm_json_is_the_null_rm(tmp_path):
    for d in (None, tmp_path / "absent", tmp_path):
        s = rm_api.load_rm(d)
        assert (s.model, s.refs, s.status, s.rm_ver, s.taste_live, s.pacc_live) == (None, None, "missing", "null",
                                                                                   False, False)
    assert isinstance(rm_api.NullRM(), rm_api.RewardModel)


def test_an_rm_loads_through_its_factory_with_or_without_references(tmp_path):
    _w(tmp_path, "rm.json", {"rm_ver": "fake-rm/3", "members": 3, "factory": FACTORY, "note": "smoke only"})
    s = rm_api.load_rm(tmp_path)
    assert s.status == "no_refs" and s.pacc_live and not s.taste_live and s.model.members == 3
    _w(tmp_path, "refs.json", {"rm_ver": "fake-rm/3", "members": 3,
                               "refs": {"596:0": [[0, 1, 2, 3], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]]}})
    s = rm_api.load_rm(tmp_path)
    assert s.status == "live" and s.taste_live and s.refs.means("596:0") == pytest.approx((1.5, 1.0, 0.5))
    assert s.flags()["ref_briefs"] == 1 and s.flags()["members"] == 3


@pytest.mark.parametrize("rm, refs, match", [
    ({"rm_ver": "null", "members": 3, "factory": FACTORY}, None, "other than"),
    ({"rm_ver": "x", "members": 0, "factory": FACTORY}, None, "positive count"),
    ({"rm_ver": "x", "members": 3, "factory": "no_colon"}, None, "package.module:callable"),
    ({"rm_ver": "x", "members": 3, "factory": "lofi_rl.nope:f"}, None, "cannot load"),
    ({"rm_ver": "x", "members": 3, "factory": "pathlib:Path"}, None, "lacks the RewardModel"),
    ({"rm_ver": "x", "members": 3, "factory": "json:dumps"}, None, "raised TypeError"),
    ({"rm_ver": "x", "members": 3, "factory": FACTORY, "extra": 1}, None, "unknown keys"),
    ({"rm_ver": "x", "members": 3, "factory": FACTORY}, {"rm_ver": "y", "members": 3, "refs": {"b": []}}, "rm_ver"),
    ({"rm_ver": "x", "members": 3, "factory": FACTORY}, {"rm_ver": "x", "members": 2, "refs": {"b": []}}, "members"),
    ({"rm_ver": "x", "members": 3, "factory": FACTORY},
     {"rm_ver": "x", "members": 3, "refs": {"b": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]}}, "lists of 4"),
    ({"rm_ver": "x", "members": 3, "factory": FACTORY},
     {"rm_ver": "x", "members": 3, "refs": {"b": [[0, 0, 0, float("nan")]] * 3}}, "finite"),
])
def test_a_broken_rm_directory_is_an_error(tmp_path, rm, refs, match):
    _w(tmp_path, "rm.json", rm)
    if refs is not None:
        _w(tmp_path, "refs.json", json.dumps(refs))
    with pytest.raises(rm_api.RMError, match=match):
        rm_api.load_rm(tmp_path)


def test_the_server_exits_2_on_a_bad_calibration_before_it_binds(tmp_path, capsys):
    _w(tmp_path / "calib", "theme.json", {"med": 0.0, "mad": 0.1, "theme_sha256": "0" * 64})
    with short_sock_dir() as d:
        rc = server.main(["--sock", str(d / "s.sock"), "--backend", "fake", "--calib", str(tmp_path / "calib")])
        assert rc == 2 and not (d / "s.sock").exists()
    assert "prompt file" in capsys.readouterr().err
    _w(tmp_path / "rm", "rm.json", {"rm_ver": "x", "members": 3, "factory": "nope"})
    with short_sock_dir() as d:
        assert server.main(["--sock", str(d / "s.sock"), "--backend", "fake", "--rm", str(tmp_path / "rm")]) == 2
