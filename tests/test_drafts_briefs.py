"""The 10 draft briefs (lofi_rl.drafts.briefs; docs/DRAFTS.md §1): the committed file is the rule's output for its
seed, every cell is covered, tempos stay in TASTE.md's range and follow its weights stratum by stratum, tonics are
distinct, and the rule is the documented hash chain (re-derived here with its own SplitMix64)."""
from pathlib import Path

import pytest
import yaml

from lofi_rl.brief import BPM_RANGE, ENERGIES, MOOD_NAMES, SCENE, TEMPO_WEIGHTS
from lofi_rl.drafts import briefs as B

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "drafts" / "v1" / "briefs.yaml"
MASK = (1 << 64) - 1


def sm64(x: int) -> int:                       # Vigna's reference SplitMix64, written out apart from lofi_rl.prng
    z = (x + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def test_the_committed_file_is_the_rules_output_for_its_seed():
    assert FILE.read_text(encoding="utf-8") == B.text(B.SEED, B.COUNT)
    assert B.main(["--check", str(FILE)]) == 0
    doc = yaml.safe_load(FILE.read_text(encoding="utf-8"))
    assert doc["seed"] == 1105 and doc["count"] == 10 and doc["scene"] == SCENE == "city at night"


def test_ten_briefs_cover_every_cell_once_plus_one_drawn_cell():
    got = B.load(FILE)
    assert list(got) == [f"d{i:02d}" for i in range(1, 11)]
    cells = [b.cell for b in got.values()]
    assert cells[:9] == [(e, m) for e in ENERGIES for m in MOOD_NAMES]
    assert cells[9] in cells[:9]
    want = B.CELLS[(((sm64(sm64(sm64(1105) ^ 1) ^ 0)) >> 11) * 9) >> 53]
    assert cells[9] == want


def test_tempos_and_tonics():
    got = list(B.load(FILE).values())
    tempos = [b.brief.bpm for b in got]
    assert all(BPM_RANGE[0] <= t <= BPM_RANGE[1] for t in tempos)
    assert sum(75 <= t <= 90 for t in tempos) == 8          # TASTE: centered on 75-90 (the strata give 7 to 9)
    assert len({b.brief.tonic for b in got}) == 10            # ten different keys
    assert sorted(tempos) == B.stratified_tempos(B.SEED)


def test_the_strata_partition_the_weights_and_each_tempo_sits_in_its_own():
    total = sum(w for _, w in TEMPO_WEIGHTS)
    assert total == 84
    units = []
    for k in range(10):
        lo, hi = total * k // 10, total * (k + 1) // 10
        units.append((lo, hi))
    assert units[0][0] == 0 and units[-1][1] == 84 and all(a[1] == b[0] for a, b in zip(units, units[1:]))
    cum, edges = 0, {}
    for bpm, w in TEMPO_WEIGHTS:
        edges[bpm] = (cum, cum + w)
        cum += w
    for k, bpm in enumerate(B.stratified_tempos(B.SEED)):
        lo, hi = units[k]
        a, b = edges[bpm]
        assert a < hi and b > lo, (k, bpm)                     # the tempo's units meet stratum k


def test_the_rule_is_the_documented_hash_chain():
    seed = 1105
    h = lambda f, i: sm64(sm64(sm64(seed) ^ f) ^ i)            # noqa: E731
    assert all(B.h(seed, f, i) == h(f, i) for f in (1, 2, 3, 4) for i in range(12))
    order = sorted(range(12), key=lambda pc: (h(4, pc), pc))[:10]
    assert [b.brief.tonic for b in B.draw(seed)] == order
    perm = sorted(range(10), key=lambda k: (h(3, k), k))
    strata = B.stratified_tempos(seed)
    assert [b.brief.bpm for b in B.draw(seed)] == [strata[k] for k in perm]


def test_other_seeds_still_cover_the_cells_with_distinct_tonics_and_seven_central_tempos():
    for seed in (0, 1, 596, 2026, MASK):
        got = B.draw(seed)
        assert [b.cell for b in got][:9] == list(B.CELLS)
        assert len({b.brief.tonic for b in got}) == 10
        strata = B.stratified_tempos(seed)
        assert strata == sorted(strata)                         # stratum order is tempo order
        assert all(75 <= t <= 90 for t in strata[2:9])          # strata 2-8 lie inside 75-90 for every seed
        assert sorted(b.brief.bpm for b in got) == strata


def test_load_refuses_a_bad_file(tmp_path):
    bad = tmp_path / "briefs.yaml"
    doc = yaml.safe_load(FILE.read_text(encoding="utf-8"))
    doc["briefs"]["d01"]["key"] = "C ionian"
    bad.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(B.BriefsError, match="key"):
        B.load(bad)
    doc["briefs"]["d01"].update(key=B.load(FILE)["d01"].brief.key_name, bpm=120)
    bad.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(B.BriefsError, match="bpm"):
        B.load(bad)
    assert B.main(["--check", str(bad)]) == 1


def test_a_redraft_folder_reads_its_parents_briefs(tmp_path):
    (tmp_path / "v1").mkdir()
    (tmp_path / "v1" / "briefs.yaml").write_text(FILE.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "v1" / "redo1").mkdir()
    assert B.briefs_file_for(tmp_path / "v1" / "redo1") == tmp_path / "v1" / "briefs.yaml"
    with pytest.raises(B.BriefsError):
        B.briefs_file_for(tmp_path)
