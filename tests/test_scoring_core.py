"""The scorer's arithmetic and state (src/lofi_rl/scoring/core.py, theme.py, rm_api.py, cache.py; docs/SCORER.md §5-7):
the gate G = hard x 1[PQ >= tau_PQ] x clip(P_acc / 0.5, 0.25, 1) and PAR on hand-built numbers, the theme margin on
hand-built embeddings, every term off when its calibration is missing and on when present, the 422 for a brief
without references, and the feature cache on disk and in memory."""
import hashlib
import importlib.metadata
import json
import math
import shutil
import statistics
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from helpers_scoring import BRIEF_A, PALETTE, ROOT, canon_sha_of, fixture_text, make_scorer, variant
from lofi_rl.palette import load_palette
from lofi_rl.scoring import features, fakes, protocol, rm_api, theme
from lofi_rl.scoring.cache import FeatureCache, hw_key
from lofi_rl.smoke import AUDIOBOX
from lofi_rl.scoring.calib import MAD_SCALE, robust_sigmoid, sigmoid
from lofi_rl.scoring.core import Unprocessable, gate_factor

PAL = load_palette(PALETTE, verify=False)
EX, SHAKER, SPAN = fixture_text("example.ls"), fixture_text("shaker_edits.ls"), fixture_text("span1_dotted.ls")
CFG = theme.load_config()


def _req(texts, brief_id="596:0"):
    return protocol.parse_score_request(protocol.score_request(
        [protocol.item_dict(i, t, BRIEF_A, brief_id) for i, t in enumerate(texts)], palette_sha256=PAL.sha256))


def _write(d, name, doc):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(doc), encoding="utf-8")
    return d


# ------------------------------------------------------------------------------------------ the gate
@pytest.mark.parametrize("hard, pq, tau, p_acc, want", [
    (0, 9.0, None, None, (0.0, 1.0, 1.0)),          # the render failed: nothing else matters
    (1, None, None, None, (1.0, 1.0, 1.0)),         # no tau_PQ, no RM: the render alone
    (1, 5.0, 5.0, None, (1.0, 1.0, 1.0)),           # PQ >= tau passes at equality
    (1, 4.99, 5.0, None, (0.0, 0.0, 1.0)),
    (1, 7.0, 5.0, 0.5, (1.0, 1.0, 1.0)),            # P_acc above 0.5 earns nothing extra
    (1, 7.0, 5.0, 0.9, (1.0, 1.0, 1.0)),
    (1, 7.0, 5.0, 0.3, (0.6, 1.0, 0.6)),
    (1, 7.0, 5.0, 0.25, (0.5, 1.0, 0.5)),
    (1, 7.0, 5.0, 0.1, (0.25, 1.0, 0.25)),          # floored at 0.25
    (1, 7.0, 5.0, 0.0, (0.25, 1.0, 0.25)),
    (1, 3.0, 5.0, 0.3, (0.0, 0.0, 0.6)),
])
def test_gate_formula_on_hand_built_cases(hard, pq, tau, p_acc, want):
    assert gate_factor(hard, pq, tau, p_acc) == pytest.approx(want)


# ------------------------------------------------------------------------------------------ PAR
def test_par_is_sigma_of_mean_minus_one_std_of_the_centred_utilities():
    u = [1.0, 2.0, 3.0, 4.0, 5.0]
    refs = [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 2.0, 4.0, 6.0], [-1.0, -1.0, 1.0, 1.0], [5.0] * 4]
    means = [statistics.fmean(r) for r in refs]                 # 0, 1, 3, 0, 5
    d = [1.0, 1.0, 0.0, 4.0, 0.0]
    assert [a - b for a, b in zip(u, means)] == d
    mean, std = statistics.fmean(d), statistics.pstdev(d)       # 1.2, population std
    r, m, s = rm_api.par(u, means)
    assert (m, s) == pytest.approx((mean, std)) and r == pytest.approx(1 / (1 + math.exp(-(mean - std))))
    assert rm_api.par([2.0, 2.0], [1.0, 1.0])[0] == pytest.approx(sigmoid(1.0)), "agreeing members: no penalty"
    assert rm_api.par([3.0, -1.0], [0.0, 0.0])[0] == pytest.approx(sigmoid(1.0 - 2.0)), "disagreement is penalized"
    with pytest.raises(ValueError):
        rm_api.par([1.0, 2.0], [0.0])
    refs_obj = rm_api.Refs("r", 5, {"b": tuple(tuple(x) for x in refs)}, "p", "s")
    assert refs_obj.means("b") == pytest.approx(tuple(means))


def _rm_state(tmp_path, table, *, refs=None, members=5, rm_ver="table-rm/1"):
    model = fakes.TableRM(table, rm_ver=rm_ver, members=members)
    r = None if refs is None else rm_api.Refs(rm_ver, members, {k: tuple(tuple(x) for x in v) for k, v in refs.items()},
                                              str(tmp_path / "refs.json"), "0" * 64)
    return rm_api.RMState(model, r, str(tmp_path), "live" if r else "no_refs", rm_ver, "1" * 64)


def test_r_taste_is_par_against_the_briefs_references_and_p_acc_scales_g(tmp_path):
    table = {canon_sha_of(EX): ([1.0, 2.0, 3.0, 4.0, 5.0], 0.3), canon_sha_of(SHAKER): ([0.5] * 5, 0.8)}
    refs = {"596:0": [[0.0] * 4, [1.0] * 4, [0.0, 2.0, 4.0, 6.0], [-1.0, -1.0, 1.0, 1.0], [5.0] * 4]}
    scorer = make_scorer(tmp_path, rm=_rm_state(tmp_path, table, refs=refs))
    ex, sh = scorer.score(_req([EX, SHAKER]))["items"]
    d = [1.0, 1.0, 0.0, 4.0, 0.0]
    assert ex["r"]["taste"] == pytest.approx(sigmoid(statistics.fmean(d) - statistics.pstdev(d)))
    assert (ex["diag"]["rm_mean"], ex["diag"]["rm_std"]) == pytest.approx((1.2, statistics.pstdev(d)))
    assert ex["diag"]["u"] == [1.0, 2.0, 3.0, 4.0, 5.0] and ex["diag"]["p_acc"] == 0.3
    assert ex["G"] == pytest.approx(0.6) and ex["diag"]["pacc_factor"] == pytest.approx(0.6)
    d2 = [0.5 - m for m in (0.0, 1.0, 3.0, 0.0, 5.0)]
    assert sh["r"]["taste"] == pytest.approx(sigmoid(statistics.fmean(d2) - statistics.pstdev(d2)))
    assert sh["G"] == 1.0
    assert scorer.terms() == {"pq_gate": False, "pacc_gate": True, "ce": False, "theme": False, "taste": True}


def test_identical_items_are_scored_once_and_another_brief_id_is_scored_again(tmp_path):
    class Counting(fakes.TableRM):
        calls = 0

        def utilities(self, feats):
            Counting.calls += 1
            return super().utilities(feats)

    refs = {"596:0": [[0.0] * 4] * 5, "596:1": [[1.0] * 4] * 5}
    rm = rm_api.RMState(Counting({canon_sha_of(EX): ([2.0] * 5, 0.9)}), rm_api.Refs(
        "table-rm/1", 5, {k: tuple(tuple(x) for x in v) for k, v in refs.items()}, "p", "s"), None, "live",
        "table-rm/1")
    scorer = make_scorer(tmp_path, rm=rm)
    items = [protocol.item_dict(i, EX, BRIEF_A, "596:0") for i in range(3)] + \
        [protocol.item_dict(3, fixture_text("example_leniency.ls"), BRIEF_A, "596:1")]
    out = scorer.score(protocol.parse_score_request(protocol.score_request(items, palette_sha256=PAL.sha256)))["items"]
    assert Counting.calls == 2, "one RM call per (clip, brief_id)"
    a, b, c, d = out
    assert b["r"] == c["r"] == a["r"] and (b["diag"]["dup_of"], c["diag"]["dup_of"]) == (0, 0)
    assert b["t_ms"] == {"render": 0.0, "features": {}, "score": 0.0} and b["i"] == 1 and c["i"] == 2
    assert d["clip_id"] == a["clip_id"] and d["diag"]["cache"] == "dup"
    assert a["r"]["taste"] == pytest.approx(sigmoid(2.0)) and d["r"]["taste"] == pytest.approx(sigmoid(1.0))


def test_an_rm_without_references_keeps_p_acc_but_not_taste(tmp_path):
    table = {canon_sha_of(EX): ([1.0] * 5, 0.2)}
    scorer = make_scorer(tmp_path, rm=_rm_state(tmp_path, table))
    it = scorer.score(_req([EX], brief_id=None))["items"][0]
    assert it["r"]["taste"] == 0.0 and it["G"] == pytest.approx(0.4) and it["diag"]["rm_mean"] is None
    assert "taste" in it["diag"]["disabled"] and "pacc_gate" not in it["diag"]["disabled"]


def test_a_brief_without_references_is_unprocessable_while_taste_is_live(tmp_path):
    scorer = make_scorer(tmp_path, rm=_rm_state(tmp_path, {}, refs={"596:0": [[0.0] * 4] * 5}))
    with pytest.raises(Unprocessable, match="596:9"):
        scorer.score(_req([EX], brief_id="596:9"))
    with pytest.raises(Unprocessable, match="None"):
        scorer.score(_req([EX], brief_id=None))


# ------------------------------------------------------------------------------------------ the theme
def test_the_theme_margin_on_hand_built_embeddings():
    e = np.eye(4)
    vec = {**{p: e[0] for p in CFG.positive}, "city at night": e[0]}
    for scene, prompts in CFG.contrast:
        for j, p in enumerate(prompts):
            vec[p] = e[1] if scene == "beach" else (e[2] if j % 2 else -e[2])
    th = theme.Theme(CFG, lambda texts: np.stack([vec[t] for t in texts]))
    windows = np.stack([e[0] * 0.8 + e[1] * 0.6] * 4)               # cos with e0 0.8, with e1 0.6
    got = th.margin(windows)
    assert got["pos_mean"] == pytest.approx(0.8) and got["contrast"]["beach"] == pytest.approx(0.6)
    assert got["contrast"]["forest"] == pytest.approx(0.0) and got["nearest_contrast"] == "beach"
    assert got["m"] == pytest.approx(0.8 - 0.6)
    cal = type("C", (), {"med": 0.1, "mad": 0.05})
    assert theme.Theme.r_theme(got["m"], cal) == pytest.approx(1 / (1 + math.exp(-(0.2 - 0.1) / (MAD_SCALE * 0.05))))
    assert theme.Theme.r_theme(got["m"], None) == 0.0
    tilted = np.stack([e[0], e[0], e[1], e[1]])                     # the mean is renormalized before the cosines
    assert th.margin(tilted)["pos_mean"] == pytest.approx(1 / math.sqrt(2))


def test_the_prompt_file_has_8_positives_and_tastes_5_contrast_scenes_of_4():
    assert len(CFG.positive) == 8 and CFG.include_brief_scene
    assert [s for s, _ in CFG.contrast] == ["sunny morning", "beach", "forest", "cafe afternoon", "snowy cabin"]
    assert all(len(ps) == 4 for _, ps in CFG.contrast)
    words = " ".join(CFG.texts()).lower()
    assert not any(w in words for w in ("music", "lofi", "lo-fi", "jazz", "beat", "song", "sax")), "scenes only"


@pytest.mark.parametrize("edit, match", [
    (lambda d: d["positive"].pop(), "8 non-empty"),
    (lambda d: d["contrast"].pop("beach"), "5 scenes"),
    (lambda d: d["contrast"]["forest"].append("a fifth"), "4 non-empty"),
    (lambda d: d.update(extra=1), "exactly the keys"),
    (lambda d: d.update(include_brief_scene=False), "include_brief_scene"),
    (lambda d: d["contrast"]["beach"].__setitem__(0, d["positive"][0]), "twice"),
])
def test_a_bad_prompt_file_is_refused(tmp_path, edit, match):
    import yaml
    doc = yaml.safe_load(theme.DEFAULT_CONFIG.read_text())
    edit(doc)
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(theme.ThemeError, match=match):
        theme.load_config(p)


# ------------------------------------------------------------------------------------------ calibrations
def test_missing_calibrations_disable_their_terms_and_present_ones_apply(tmp_path):
    off = make_scorer(tmp_path).score(_req([EX, SHAKER]))["items"]
    for it in off:
        assert it["r"] == {"ce": 0.0, "theme": 0.0, "taste": 0.0} and it["G"] == 1.0
        assert sorted(it["diag"]["disabled"]) == sorted(protocol.TERM_FLAGS)
        assert it["diag"]["pq_factor"] == 1.0 and it["diag"]["pacc_factor"] == 1.0 and it["diag"]["tau_pq"] is None
        assert it["sig"]["ce"] is not None and it["sig"]["clap_margin"] is not None, "signatures are always measured"
    cal = tmp_path / "calib"
    _write(cal, "pq.json", {"tau_pq": 6.0, "n": 40, "note": "hand-built"})
    _write(cal, "theme.json", {"med": 0.01, "mad": 0.02, "theme_sha256": CFG.sha256})
    _write(cal, "ce.json", {"agree": 0.6, "lower95": 0.53, "n_pairs": 500, "med": 5.5, "mad": 0.4})
    feat = fakes.FakeFeaturizer(audiobox={canon_sha_of(EX): {"PQ": 7.5, "CE": 6.1},
                                          canon_sha_of(SHAKER): {"PQ": 5.9, "CE": 4.0}})
    scorer = make_scorer(tmp_path, calib_dir=cal, featurizer=feat)
    ex, sh = scorer.score(_req([EX, SHAKER]))["items"]
    assert scorer.terms() == {"pq_gate": True, "pacc_gate": False, "ce": True, "theme": True, "taste": False}
    assert (ex["G"], sh["G"], sh["hard"], sh["gate"]) == (1.0, 0.0, 1, None), "PQ 5.9 < tau 6.0"
    assert ex["r"]["ce"] == pytest.approx(robust_sigmoid(6.1, 5.5, 0.4))
    assert sh["r"]["ce"] == pytest.approx(robust_sigmoid(4.0, 5.5, 0.4))
    for it in (ex, sh):
        assert it["r"]["theme"] == pytest.approx(robust_sigmoid(it["sig"]["clap_margin"], 0.01, 0.02))
        assert sorted(it["diag"]["disabled"]) == ["pacc_gate", "taste"]
    flags = scorer.health()["calibration"]
    assert flags["pq"]["tau_pq"] == 6.0 and flags["pq"]["note"] == "hand-built" and flags["ce"]["status"] == "validated"


def test_ce_that_fails_its_check_stays_off(tmp_path):
    cal = _write(tmp_path / "c", "ce.json", {"agree": 0.54, "lower95": 0.51, "n_pairs": 450})
    scorer = make_scorer(tmp_path, calib_dir=cal)
    assert scorer.terms()["ce"] is False and scorer.health()["calibration"]["ce"]["status"] == "not_validated"
    assert scorer.score(_req([EX]))["items"][0]["r"]["ce"] == 0.0


# ------------------------------------------------------------------------------------------ the cache
def test_features_are_cached_on_disk_by_feat_ver_and_clip_id_and_read_back_by_a_fresh_scorer(tmp_path):
    root = tmp_path / "cache"
    r1 = fakes.FakeRenderer(fail={canon_sha_of(SPAN): ("G7", "true peak above -0.9 dBTP")})
    a = make_scorer(tmp_path, cache_root=root, renderer=r1)
    first = a.score(_req([EX, SHAKER, SPAN]))["items"]
    files = sorted(p.name for p in (root / "feat" / a.featurizer.feat_ver).glob("*.npz"))
    assert files == sorted(f"{it['clip_id']}.npz" for it in first), "gate failures are cached too"
    r2 = fakes.FakeRenderer()
    b = make_scorer(tmp_path, cache_root=root, renderer=r2)
    second = b.score(_req([EX, SHAKER, SPAN]))["items"]
    assert r2.calls == [] and [it["diag"]["cache"] for it in second] == ["disk"] * 3
    assert a.canary is None, "a gate failure (G7) is the loop's, not the renderer's: no canary"
    for x, y in zip(first, second):
        assert (x["clip_id"], x["hard"], x["G"], x["gate"], x["r"], x["sig"]) == \
            (y["clip_id"], y["hard"], y["G"], y["gate"], y["r"], y["sig"])
    assert second[2]["gate"] == "G7" and "dBTP" in second[2]["diag"]["error"]
    bad = root / "feat" / a.featurizer.feat_ver / f"{first[0]['clip_id']}.npz"
    bad.write_bytes(b"not an npz")
    c = make_scorer(tmp_path, cache_root=root, renderer=fakes.FakeRenderer())
    again = c.score(_req([EX]))["items"][0]
    assert again["diag"]["cache"] == "miss" and c.cache.counters()["bad"] == 1 and again["r"] == first[0]["r"]
    assert np.load(bad, allow_pickle=False)["mert"].shape == (13, 768), "rewritten"


def test_feat_ver_follows_the_render_and_feature_source_files(tmp_path):
    # SCORER.md §8.3: an edit to the code a clip's audio and features come from starts a new cache directory.
    files = sorted({p for g in features.SOURCE_GLOBS for p in ROOT.glob(g) if p.is_file()})
    for p in files:
        dst = tmp_path / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    assert features.source_sha8(tmp_path) == features.source_sha8(tmp_path) == features.source_sha8(ROOT)
    sha, ver = features.source_sha8(tmp_path), features.feat_version("fake", root=tmp_path)
    with (tmp_path / "src" / "lofi_rl" / "render" / "mix.py").open("ab") as f:
        f.write(b"\n")
    assert features.source_sha8(tmp_path) != sha and features.feat_version("fake", root=tmp_path) != ver


STACK_KEYS = ("numpy", "scipy", "pedalboard", "pyloudnorm", "soundfile", "mido", "fluidsynth", "libsndfile", "python",
              *features.ML_PACKAGES, "cuda")


def test_feat_ver_names_the_audiobox_revision_and_every_library_of_the_stack():
    stack = {k: "1.0" for k in STACK_KEYS}                     # synthetic versions: only their changes matter
    v = features.feat_version("real", src="0" * 8, stack=stack)
    assert f".abx-{AUDIOBOX[1][:8]}." in v and v == features.feat_version("real", src="0" * 8, stack=dict(stack))
    changed = {features.feat_version("real", src="0" * 8, stack={**stack, k: "1.1"}) for k in STACK_KEYS}
    assert len(changed | {v}) == len(STACK_KEYS) + 1, "each library's version is in feat_ver"


@pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth not on PATH")
def test_the_stack_is_the_render_stack_without_the_cpu_plus_the_ml_stack(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda="cuda-x")))
    s = features.stack_versions()
    assert "cpu" not in s and set(STACK_KEYS) <= set(s), "the CPU model keys the cache directory instead"
    assert s["cuda"] == "cuda-x" and s["numpy"] == np.__version__


def test_the_audiobox_checkpoint_must_be_the_pinned_revision_in_refs_main(tmp_path):
    repo, rev, need = AUDIOBOX
    base = tmp_path / "hub" / ("models--" + repo.replace("/", "--"))
    with pytest.raises(FileNotFoundError, match="refs/main"):
        features.check_audiobox_ref(str(tmp_path))
    (base / "refs").mkdir(parents=True)
    (base / "refs" / "main").write_text("f" * 40)            # what a fetch without @rev would have written
    with pytest.raises(RuntimeError, match=f"pinned at {rev[:12]}"):
        features.check_audiobox_ref(str(tmp_path))
    (base / "refs" / "main").write_text(rev + "\n")
    with pytest.raises(FileNotFoundError, match="no cached snapshot"):
        features.check_audiobox_ref(str(tmp_path))
    (base / "snapshots" / rev).mkdir(parents=True)
    for f in need:
        (base / "snapshots" / rev / f).write_text("{}")
    assert features.check_audiobox_ref(str(tmp_path)) == rev


def test_the_installed_audiobox_package_must_be_the_one_feat_ver_names():
    assert features.check_audiobox_package(lambda name: features.AUDIOBOX_PKG) == "0.0.4"
    with pytest.raises(RuntimeError, match="0.0.5 is installed"):
        features.check_audiobox_package(lambda name: "0.0.5")

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)
    with pytest.raises(RuntimeError, match="None is installed"):
        features.check_audiobox_package(missing)


def test_a_record_for_another_feat_ver_clip_or_schema_is_a_miss(tmp_path):
    root = tmp_path / "cache"
    feat_a = fakes.FakeFeaturizer()
    feat_a.feat_ver = "A"
    a = make_scorer(tmp_path, cache_root=root, featurizer=feat_a)
    cid = a.score(_req([EX]))["items"][0]["clip_id"]
    rec = root / "feat" / "A" / f"{cid}.npz"
    assert rec.is_file() and FeatureCache(root, "B").get(cid) is None
    shutil.copyfile(rec, root / "feat" / "B" / f"{cid}.npz")         # A's record under B's name: still a miss
    b_cache = FeatureCache(root, "B")
    assert b_cache.get(cid) is None and b_cache.counters()["bad"] == 1
    feat_b, r = fakes.FakeFeaturizer(), fakes.FakeRenderer()
    feat_b.feat_ver = "B"
    b = make_scorer(tmp_path, cache_root=root, featurizer=feat_b, renderer=r)
    it = b.score(_req([EX]))["items"][0]
    assert it["diag"]["cache"] == "miss" and len(r.calls) == 1 and b.cache.counters()["bad"] == 1, "rendered again"
    other = "c" + "f" * 16                                            # A's record under another clip's name
    shutil.copyfile(rec, root / "feat" / "A" / f"{other}.npz")
    a_cache = FeatureCache(root, "A")
    assert a_cache.get(other) is None and a_cache.counters()["bad"] == 1
    with np.load(rec, allow_pickle=False) as z:                       # another schema
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays["meta"]))
    arrays["meta"] = np.array(json.dumps({**meta, "schema": "lofi-scorer/clip/0"}))
    with rec.open("wb") as f:
        np.savez(f, **arrays)
    assert a_cache.get(cid) is None and a_cache.counters()["bad"] == 2


def test_the_servers_cache_is_keyed_by_its_machine_and_gpu_and_every_record_names_them(tmp_path):
    # The project's render rule (render/machine.py: two CPU models never share renders), applied to features: an
    # L40S node and an A100 node, or two CPU models, never serve each other's records from one --cache.
    from lofi_rl.scoring import server
    a = server.parse_args(["--sock", str(tmp_path / "unused.sock"), "--backend", "fake", "--cache", str(tmp_path / "c")])
    scorer = server.build_scorer(a)
    it = scorer.score(_req([EX]))["items"][0]
    feat = scorer.featurizer
    assert feat.hw == hw_key(feat.machine, "fake") and it["diag"]["hw"] == {"machine": feat.machine, "gpu": "fake"}
    rec = (tmp_path / "c" / "feat" / feat.feat_ver / f"hw-{hashlib.sha256(feat.hw.encode()).hexdigest()[:8]}" /
           f"{it['clip_id']}.npz")
    with np.load(rec, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    assert (meta["hw"], meta["machine"], meta["gpu"], meta["device"]) == (feat.hw, feat.machine, "fake", "fake")
    assert scorer.health()["cache"]["hw"] == feat.hw
    other = FeatureCache(tmp_path / "c", feat.feat_ver, hw=hw_key(feat.machine, "another GPU"))
    assert other.dir != rec.parent and other.get(it["clip_id"]) is None, "another GPU: another directory"
    shutil.copyfile(rec, other.dir / rec.name)
    assert other.get(it["clip_id"]) is None and other.counters()["bad"] == 1, "a record of other hardware is a miss"


def test_fake_features_with_another_renderer_are_another_cache(tmp_path):
    # --backend fake with --render fake and with --render real on one --cache: neither serves the other's records
    # (a fake render's "ok", exactly -14.0 LUFS, would hide a real G7).
    class StandIn(fakes.FakeRenderer):                  # in place of the real pool
        name = "real"
    root = tmp_path / "cache"
    a = make_scorer(tmp_path, cache_root=root)
    assert a.score(_req([EX]))["items"][0]["diag"]["cache"] == "miss"
    r = StandIn()
    b = make_scorer(tmp_path, cache_root=root, renderer=r)
    assert b.score(_req([EX]))["items"][0]["diag"]["cache"] == "miss" and len(r.calls) == 1
    assert "render-fake" in a.featurizer.feat_ver and "render-real" in b.featurizer.feat_ver
    assert make_scorer(tmp_path, cache_root=root).score(_req([EX]))["items"][0]["diag"]["cache"] == "disk"
    with pytest.raises(ValueError, match="renders with 'real'"):
        make_scorer(tmp_path, renderer=StandIn(), featurizer=fakes.FakeFeaturizer())


def test_the_memory_lru_evicts_the_oldest_clip(tmp_path):
    renderer = fakes.FakeRenderer()
    scorer = make_scorer(tmp_path, renderer=renderer, lru=2)
    loops = [variant(EX, k) for k in range(3)]
    scorer.score(_req(loops))
    assert len(renderer.calls) == 3 and len(scorer.cache) == 2
    kinds = [it["diag"]["cache"] for it in scorer.score(_req(loops))["items"]]
    assert kinds[0] == "miss", "evicted, so rendered again"
    assert len(renderer.calls) == 4


def test_the_fakes_are_deterministic_and_shaped_like_the_real_features():
    f = fakes.FakeFeaturizer()
    a = f.featurize("c1", "x", None, None)
    b = fakes.FakeFeaturizer().featurize("c1", "x", None, None)
    assert a.mert.shape == (13, 768) and a.clap_mix.shape == a.clap_music.shape == (4, 512)
    assert np.array_equal(a.mert, b.mert) and np.array_equal(a.clap_music, b.clap_music)
    assert np.allclose(np.linalg.norm(a.clap_mix, axis=1), 1.0, atol=1e-6)
    t = f.encode_text(["a", "b", "a"])
    assert np.array_equal(t[0], t[2]) and np.allclose(np.linalg.norm(t, axis=1), 1.0)
