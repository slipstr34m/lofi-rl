"""The drawn round tables (lofi_rl.freeze.tables; the freeze spec §5.2; SHOOTOUTS.md §2.3 "Published after the freeze").

On a synthetic world: three shootout jobs and a keep3 job, written as the batch builders write their summaries and
the inbox manifests. It has design rounds, an anchor, a repeat of an earlier job's round, and a stage whose maps share
symbols. The tests check that the tables come out deterministic, cross-checked against the manifests, without a seed
or an id, and only after the APPROVED line. The section published in the gate file holds the design rounds only, with
no serving position, no hidden-check row and no word of how one is built (Phase 2 and the final A/B use the same
checks); the full tables go only to a private file under data/, which every test here moves to a temporary folder.
The section goes where lofi_rl.freeze.gate sees the Approval section end. Then the same runs on copies of the real
runs, inbox and gate in a temporary folder (skipped where `make pull` has not brought them)."""
import copy
import hashlib
import json
import re
import shutil
from datetime import date
from pathlib import Path

import pytest

from lofi_rl.freeze import JOBS as REAL_JOBS
from lofi_rl.freeze import PICKS
from lofi_rl.freeze import gate as G
from lofi_rl.freeze import tables as T

ROOT = Path(__file__).resolve().parents[1]
REAL_PRIVATE_ROOT = T.PRIVATE_ROOT                  # read at import, before the fixture below moves it
JOBS = {"day1": 101, "day2": 102, "day3": 103, "keep3": 104}
DATES = {"day1": "2026-01-01", "day2": "2026-01-02", "day3": "2026-01-03", "keep3": "2026-01-04"}
SQUARE = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
LINE = "APPROVED by Shresth Rana 2026-11-05"
D = date(2026, 11, 5)
# the words that would say how a hidden check is built, or mark one
RECIPE = re.compile(r"\b(repeat\w*|anchor\w*|revers\w*|sour|broken|first shown|shown again|imitat\w*)\b", re.I)
SHOOTOUT_ROW = re.compile(r"^\| [A-Za-z][A-Za-z0-9]*\.\d+ \| ", re.M)          # a published row: | <slot> | ...
PRIVATE_ROW = re.compile(r"^\| \d+ \| [A-Za-z][A-Za-z0-9]*\.\d+ \| ", re.M)     # a private row: | <pos> | <slot> |


@pytest.fixture(autouse=True)
def private_root(tmp_path_factory, monkeypatch):
    """lofi_rl.freeze.tables.PRIVATE_ROOT (the repo's data/) moved to a temporary `data/` holding `votes/audit/`, so
    no test reads or writes the real one."""
    root = tmp_path_factory.mktemp("private") / "data"
    (root / "votes" / "audit").mkdir(parents=True)
    monkeypatch.setattr(T, "PRIVATE_ROOT", root)
    return root


def private_file(root: Path) -> Path:
    return root / "votes" / "audit" / "drawn_round_tables.md"


def argv(root: Path, *extra: str) -> list[str]:
    return ["--runs", str(root / "runs"), "--inbox", str(root / "inbox"), *extra]


def _h(label: str, n: int) -> str:
    return hashlib.sha256(label.encode()).hexdigest()[:n]


def rid(day: str, batch: str, index: int) -> str:
    return "r" + _h(f"{day}|{batch}|{index}|round", 8)


def oid(day: str, batch: str, index: int, pos: int) -> str:
    return "k" + _h(f"{day}|{batch}|{index}|{pos}|opaque", 10)


def R(stage, slot, loop, bpm, tiles, category=None, **extra):
    return {"stage": stage, "slot": slot, "loop": loop, "bpm": bpm, "tiles": tiles, "category": category or stage,
            **extra}


DESIGN = {
    "day1": ({"K1": [{"P1a": "keys_a", "P1b": "keys_b", "P2a": "keys_c", "P2b": "keys_d"}],
              "S1": [{"t1": "sax_t1", "t2": "sax_t2"}, {"a1": "sax_a1", "a2": "sax_a2"}]},
             [("b0101-1-keys", "keys", [
                 R("K1", "K1.1", "l1_melancholy_low", 76, [("keys_a", "rootless"), ("keys_b", "rootless"),
                                                           ("keys_c", "rootless")]),
                 R("K1", "K1.2", "l5_wistful_mid", 84, [("keys_d", "rootless"), ("keys_a", "rootless"),
                                                        ("keys_b", "rootless")])]),
              ("b0101-2-sax", "sax", [
                  R("S1", "S1.1", "l1_melancholy_low", 76, [("sax_t1", "sax source"), ("sax_a1", "sax source"),
                                                            ("sax_a2", "sax source")]),
                  R("S1", "S1.2", "l9_hopeful_high", 90, [("sax_t2", "sax source"), ("sax_a1", "sour copy"),
                                                          ("sax_t1", "sax source")], kind="anchor", broken=2)])]),
    "day2": ({"V": [{"v1": "rise", "v2": "plagal"}, {"v1": "float", "v2": "step"}]},
             [("b0102-1-vamps", "vamps", [
                 R("V", "V.1", "l1_melancholy_low", 76, [("rise", "sax muted"), ("plagal", "sax muted")]),
                 R("V", "V.4", "l4_wistful_low", 80, [("float", "sax muted"), ("step", "sax muted")])])]),
    "day3": ({"W1": [{"z1": "excerpt_e1", "z2": "excerpt_e2"}]},
             [("b0103-1-mix", "per_round", [
                 R("W1", "W1.1", "l1_melancholy_low", 76, [("excerpt_e1", "rain excerpt at delta 18"),
                                                           ("excerpt_e2", "rain excerpt at delta 18")],
                   category="rain_excerpt"),
                 {"repeat": ("day1", "b0101-1-keys", 2)},
                 R("W1", "W1.1", "l1_melancholy_low", 76, [("excerpt_e2", "sour copy"),
                                                           ("excerpt_e1", "rain excerpt at delta 18")],
                   category="rain_excerpt", kind="anchor", broken=1)])]),
}
KEEP3_ROUNDS = [{"brief": "d02", "energy": "low", "mood": "wistful", "bpm": 81, "row": 1},
                {"brief": "d01", "energy": "low", "mood": "melancholy", "bpm": 79, "row": 0}]
SYMBOLS = ["chords", "horn", "groove"]


def world(jobs: dict[str, int] = JOBS, design: dict = DESIGN) -> dict[str, dict]:
    """Every file of the synthetic world: repo-relative path -> JSON document."""
    docs: dict[str, dict] = {}
    rounds_of: dict[tuple, dict] = {}                  # (day, batch, index) -> its batches.json round
    for day in ("day1", "day2", "day3"):
        job, maps, batches = jobs[day], *design[day]
        draw = {"schema": "lofi-rl/shootout-draw/1", "seed": _h(f"{day}|seed", 16), "stage": day, "v": 1,
                "maps": maps, "batches": [], "band": {}, "leaders": {}, "palettes": {}, "plan": {}, "renders": {},
                "repeats": {}, "rerun_s2": None, "sour": {}}
        out = {"schema": "lofi-rl/shootout-batches/1", "run": f"shootout-{job}", "seed": _h(f"{day}|seed", 16),
               "stage": day, "v": 1, "batches": {}}
        for name, category, spec in batches:
            d = DATES[day]
            draw_rounds, b_rounds, pub_rounds, hid_rounds = [], [], [], {}
            for index, r in enumerate(spec, start=1):
                if "repeat" in r:
                    fday, fname, findex = r["repeat"]
                    first = rounds_of[(fday, fname, findex)]
                    base = {k: first[k] for k in ("stage", "slot", "loop", "category")}
                    bpm = first["_bpm"]
                    tiles = [(c["option"], c["what"], c["render"]) for c in reversed(first["clips"])]
                    kind, broken = "repeat", None
                else:
                    base = {"stage": r["stage"], "slot": r["slot"], "loop": r["loop"], "category": r["category"]}
                    bpm, kind, broken = r["bpm"], r.get("kind", "none"), r.get("broken")
                    tiles = [(o, w, f"{r['stage']}.{r['loop']}.{o}" + (".sour" if pos == broken else ""))
                             for pos, (o, w) in enumerate(r["tiles"], start=1)]
                rnd_id = rid(day, name, index)
                clips = [{"clip": "c" + _h(render, 16), "flac_sha256": _h(render + "|flac", 64),
                          "id": oid(day, name, index, pos), "option": o, "pcm_sha256": _h(render + "|pcm", 64),
                          "pos": pos, "render": render, "what": w}
                         for pos, (o, w, render) in enumerate(tiles, start=1)]
                anchor = clips[broken - 1]["id"] if kind == "anchor" else None
                of = rid(*r["repeat"]) if kind == "repeat" else None
                b_round = {"anchor_clip": anchor, "category": base["category"], "clips": clips, "id": rnd_id,
                           "index": index, "kind": kind, "loop": base["loop"], "of": of, "slot": base["slot"],
                           "stage": base["stage"]}
                b_rounds.append(b_round)
                rounds_of[(day, name, index)] = {**b_round, "_bpm": bpm}
                dr = {"bpm": bpm, "category": base["category"], "kind": kind, "loop": base["loop"],
                      "slot": base["slot"], "stage": base["stage"],
                      "clips": [{"option": c["option"], "render": c["render"], "what": c["what"]} for c in clips]}
                if kind == "anchor":
                    dr.update(anchor_pos=broken, imitates=base["slot"])
                if kind == "repeat":
                    fday, fname, _ = r["repeat"]
                    dr.update(of=of, of_batch=fname, of_run=f"shootout-{jobs[fday]}", group="keys")
                draw_rounds.append(dr)
                pub_rounds.append({"id": rnd_id, "bpm": bpm, "clips": [{"id": c["id"], "file": f"{c['id']}.flac",
                                                                       "sha256": c["flac_sha256"]} for c in clips]})
                hid_rounds[rnd_id] = {"kind": kind, "of": of, "anchor_clip": anchor, "brief_id": base["loop"],
                                      "category": base["category"],
                                      "clips": {c["id"]: {"clip": c["clip"], "option": c["option"], "what": c["what"],
                                                          "stage": base["stage"], "slot": base["slot"],
                                                          "loop": base["loop"], "palette": f"p_{c['option']}@12345678",
                                                          "render": c["render"], "pcm_sha256": c["pcm_sha256"]}
                                                for c in clips}}
            draw["batches"].append({"name": name, "date": d, "day": day, "category": category, "est_min": 1.0,
                                    "note": "Which do you like best?", "rounds": draw_rounds})
            out["batches"][name] = {"category": category, "date": d, "dir": f"{d}/{name}", "est_min": 1.0,
                                    "rounds": b_rounds}
            docs[f"inbox/{d}/{name}/manifest.json"] = {"v": 1, "batch": name, "mode": "shootout",
                                                       "scene": "city at night", "note": "Which?", "bars": 8,
                                                       "sr": 44100, "est_min": 1.0, "rounds": pub_rounds}
            docs[f"inbox/{d}/{name}/manifest.hidden.json"] = {"v": 1, "batch": name, "category": category,
                                                              "rounds": hid_rounds}
        docs[f"runs/shootout-{job}/summary/draw.json"] = draw
        docs[f"runs/shootout-{job}/summary/batches.json"] = out
    job, d, name = jobs["keep3"], DATES["keep3"], "b0104-keep3"
    draw = {"schema": "lofi-rl/draft-draw/1", "seed": _h("keep3|seed", 16), "square": SQUARE, "symbols": SYMBOLS,
            "rounds": [], "order_constraint_met": True, "same_cell_neighbours": 0}
    b_rounds, pub_rounds, hid_rounds = [], [], {}
    for index, r in enumerate(KEEP3_ROUNDS, start=1):
        personas = [SYMBOLS[i] for i in SQUARE[r["row"]]]
        draw["rounds"].append({"brief": r["brief"], "row": r["row"], "tiles": personas})
        rnd_id = rid("keep3", name, index)
        clips = [{"id": oid("keep3", name, index, pos), "pos": pos, "persona": p,
                  "draft": f"drafts/v1/{r['brief']}_{p}.ls", "render": f"{r['brief']}_{p}",
                  "clip": "c" + _h(f"{r['brief']}_{p}", 16), "pcm_sha256": _h(f"{r['brief']}_{p}|pcm", 64),
                  "flac_sha256": _h(f"{r['brief']}_{p}|flac", 64)} for pos, p in enumerate(personas, start=1)]
        b_rounds.append({"id": rnd_id, "index": index, "brief": r["brief"], "row": r["row"], "clips": clips})
        pub_rounds.append({"id": rnd_id, "bpm": r["bpm"], "clips": [{"id": c["id"], "file": f"{c['id']}.flac",
                                                                    "sha256": c["flac_sha256"]} for c in clips]})
        hid_rounds[rnd_id] = {"kind": "none", "of": None, "anchor_clip": None, "brief_id": r["brief"],
                              "category": "drafts",
                              "clips": {c["id"]: {"clip": c["clip"], "persona": c["persona"], "draft": c["draft"],
                                                  "draft_sha256": _h(c["draft"], 64), "brief": r["brief"],
                                                  "energy": r["energy"], "mood": r["mood"], "bpm": r["bpm"],
                                                  "render": c["render"], "pcm_sha256": c["pcm_sha256"]}
                                        for c in clips}}
    docs[f"runs/drafts-{job}/summary/draw.json"] = draw
    docs[f"runs/drafts-{job}/summary/batch.json"] = {"schema": "lofi-rl/draft-batch/1", "batch": name, "date": d,
                                                     "dir": f"{d}/{name}", "est_min": 1.0, "tile_counts": {},
                                                     "rounds": b_rounds}
    docs[f"inbox/{d}/{name}/manifest.json"] = {"v": 1, "batch": name, "mode": "keep3", "scene": "city at night",
                                               "note": "Keep one.", "bars": 8, "sr": 44100, "est_min": 1.0,
                                               "rounds": pub_rounds}
    docs[f"inbox/{d}/{name}/manifest.hidden.json"] = {"v": 1, "batch": name, "category": "drafts",
                                                      "rounds": hid_rounds}
    return copy.deepcopy(docs)                          # no test's change reaches DESIGN or another test


def write(root: Path, docs: dict[str, dict]) -> Path:
    for rel, doc in docs.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return root


GATE_HEAD = ("# Gate: palette_v1 freeze (Phase 1)\n\n## Shootout results\n\n"
             "### Day 1: voted 2026-10-31 (batches b0101-1-keys, b0101-2-sax; job 101)\n\n"
             "**Shresth's confirmation, 2026-10-31:** \"Yes, use these\".\n\n"
             "## How to approve\nAdd one line of your own, exactly in this form:\n\n"
             "    APPROVED by Shresth Rana <yyyy-mm-dd>\n\n## Approval\n")


def gate_file(root: Path, approval: str = LINE + "\n", after: str = "") -> Path:
    p = root / "gates" / "GATE_palette.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(GATE_HEAD + approval + after, encoding="utf-8")
    return p


@pytest.fixture()
def jobs(monkeypatch):
    """The CLI reads lofi_rl.freeze.JOBS: here the synthetic world's."""
    monkeypatch.setattr(T, "JOBS", JOBS)
    return JOBS


@pytest.fixture()
def synth(tmp_path, jobs):
    return write(tmp_path, world())


def collect(root: Path, jobs=JOBS):
    return T.collect(root / "runs", root / "inbox", jobs)


def secrets_of(docs: dict[str, dict]) -> set[str]:
    """Every seed, opaque id, real clip id and audio hash the inputs hold."""
    out: set[str] = set()

    def walk(x, key=None):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and re.fullmatch(r"[rk][0-9a-f]{8,10}", k):
                    out.add(k)
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, str) and key in ("seed", "id", "of", "anchor_clip", "clip", "pcm_sha256", "flac_sha256",
                                            "sha256", "file"):
            out.add(x.removesuffix(".flac"))
    for doc in docs.values():
        walk(doc)
    return out


# --------------------------------------------------------------------------------------------- the tables
def test_every_round_in_serving_order_with_its_kind_and_tiles(synth):
    tabs = collect(synth)
    assert [t.key for t in tabs] == ["day1", "day2", "day3", "keep3"]
    d1, d2, d3, k3 = tabs
    keys, sax = d1.batches
    assert [(r.index, r.slot, r.kind) for r in keys.rows] == [(1, "K1.1", "design"), (2, "K1.2", "design")]
    assert [(t.option, t.what, t.broken) for t in keys.rows[0].tiles] == [
        ("keys_a", "rootless", False), ("keys_b", "rootless", False), ("keys_c", "rootless", False)]
    anchor = sax.rows[1]
    assert anchor.kind == "anchor of S1.2" and [t.broken for t in anchor.tiles] == [False, True, False]
    assert anchor.tiles[1] == T.Tile("sax_a1", "sour copy", True)
    w1, rep, anc = d3.batches[0].rows
    assert w1.kind == "design" and anc.kind == "anchor of W1.1" and [t.broken for t in anc.tiles] == [True, False]
    assert rep.kind == "repeat of K1.2, first shown b0101-1-keys #2" and rep.slot == "K1.2"
    assert [t.option for t in rep.tiles] == ["keys_b", "keys_a", "keys_d"]                  # tiles reversed
    assert d1.maps[:4] == (("K1", "P1a", "keys_a"), ("K1", "P1b", "keys_b"), ("K1", "P2a", "keys_c"),
                           ("K1", "P2b", "keys_d"))
    assert ("S1", "a1", "sax_a1") in d1.maps                                              # disjoint symbols: no slots
    assert d2.maps == (("V (V.1)", "v1", "rise"), ("V (V.1)", "v2", "plagal"), ("V (V.4)", "v1", "float"),
                       ("V (V.4)", "v2", "step"))                                          # shared symbols: their slots
    assert [t.public_maps for t in (d1, d2, d3)] == [t.maps for t in (d1, d2, d3)]        # no anchor-only slot here
    assert k3.square == (("chords", "horn", "groove"), ("horn", "groove", "chords"), ("groove", "chords", "horn"))
    assert [(r.slot, r.loop, r.row, [t.option for t in r.tiles]) for r in k3.batches[0].rows] == [
        ("d02", "low, wistful", 1, ["horn", "groove", "chords"]),
        ("d01", "low, melancholy", 0, ["chords", "horn", "groove"])]
    assert k3.batches[0].rows[0].tiles[0].what == "drafts/v1/d02_horn.ls"
    assert dict(d1.files) == {
        "runs/shootout-101/summary/draw.json": hashlib.sha256(
            (synth / "runs/shootout-101/summary/draw.json").read_bytes()).hexdigest(),
        "runs/shootout-101/summary/batches.json": hashlib.sha256(
            (synth / "runs/shootout-101/summary/batches.json").read_bytes()).hexdigest()}


def _table_rows(text: str, heading: str) -> list[str]:
    """The rows of the table under the one line `heading` of `text`, its header row and rule left out."""
    lines = text.splitlines()
    assert lines.count(heading) == 1, heading
    out = []
    for x in lines[lines.index(heading) + 1:]:
        if x.startswith("#"):
            break
        if x.startswith("|") and not x.startswith("|---"):
            out.append(x)
    return out[1:]


def _cells_match_headers(lines: list[str]) -> int:
    """Every table row has its header's number of cells; returns the number of tables."""
    tables = 0
    for i, x in enumerate(lines):
        if x.startswith("|---"):
            tables += 1
            width = lines[i - 1].count("|")
            assert x.count("|") == width, lines[i - 1]
            for r in lines[i + 1:]:
                if not r.startswith("|"):
                    break
                assert r.count("|") - r.count("\\|") == width, (lines[i - 1], r)
    return tables


def test_the_section_lists_the_design_rounds_by_slot(synth):
    text = T.render(collect(synth), D)
    lines = text.splitlines()
    assert lines[0] == T.SECTION and text.endswith("|\n") and not text.endswith("\n\n")
    assert "his approval line above is dated 2026-11-05" in text
    assert _table_rows(text, "#### b0101-1-keys (`inbox/2026-01-01/b0101-1-keys/`)") == [
        "| K1.1 | l1_melancholy_low | `keys_a` (rootless) | `keys_b` (rootless) | `keys_c` (rootless) |",
        "| K1.2 | l5_wistful_mid | `keys_d` (rootless) | `keys_a` (rootless) | `keys_b` (rootless) |"]
    assert _table_rows(text, "#### b0101-2-sax (`inbox/2026-01-01/b0101-2-sax/`)") == [
        "| S1.1 | l1_melancholy_low | `sax_t1` (sax source) | `sax_a1` (sax source) | `sax_a2` (sax source) |"]
    assert _table_rows(text, "#### b0103-1-mix (`inbox/2026-01-03/b0103-1-mix/`)") == [  # only its design round's tiles
        "| W1.1 | l1_melancholy_low | `excerpt_e1` (rain excerpt at delta 18) | "
        "`excerpt_e2` (rain excerpt at delta 18) |"]
    assert _table_rows(text, "#### b0104-keep3 (`inbox/2026-01-04/b0104-keep3/`)") == [   # keep3: as served
        "| 1 | d02 | low, wistful | 1 | `horn` (`drafts/v1/d02_horn.ls`) | `groove` (`drafts/v1/d02_groove.ls`) | "
        "`chords` (`drafts/v1/d02_chords.ls`) |",
        "| 2 | d01 | low, melancholy | 0 | `chords` (`drafts/v1/d01_chords.ls`) | `horn` (`drafts/v1/d01_horn.ls`) | "
        "`groove` (`drafts/v1/d01_groove.ls`) |"]
    assert "| slot | loop | tile 1 | tile 2 | tile 3 |" in lines and "| V (V.4) | v1 | `float` |" in lines
    assert [x for x in lines if x.startswith("### ")] == [
        "### Drawn, Day 1 (job 101)", "### Drawn, Day 2 (job 102)", "### Drawn, Day 3 (job 103)",
        "### Drawn, keep3 (job 104)"]
    assert [x for x in lines if x.startswith("#### ")] == [
        "#### b0101-1-keys (`inbox/2026-01-01/b0101-1-keys/`)", "#### b0101-2-sax (`inbox/2026-01-01/b0101-2-sax/`)",
        "#### b0102-1-vamps (`inbox/2026-01-02/b0102-1-vamps/`)", "#### b0103-1-mix (`inbox/2026-01-03/b0103-1-mix/`)",
        "#### b0104-keep3 (`inbox/2026-01-04/b0104-keep3/`)"]
    assert _cells_match_headers(lines) == 3 + 1 + 5         # 3 map tables, keep3's square, then 5 batches


def test_the_section_holds_no_hidden_check_and_no_word_of_how_one_is_built(synth):
    tabs = collect(synth)
    text = T.render(tabs, D)
    assert T.hidden_checks(tabs) == 3
    assert RECIPE.findall(text) == [] and "**broken**" not in text and "sour copy" not in text
    assert [x for x in text.splitlines() if x.startswith("- Left out")] == [
        "- Left out: 3 rounds, the hidden checks (RATING.md, \"Checks you won't see\"). Phase 2's votes and the final "
        "A/B use the same checks, and the rater is never told which rounds those are. The full tables are kept "
        "privately, never committed."]
    hidden = [r for t in tabs for b in t.batches for r in b.rows if r.kind != "design"]
    assert len(hidden) == 3
    for r in hidden:                                        # no hidden round's row: its slot, loop and tiles
        assert f"| {r.slot} | {r.loop} | " + " | ".join(T._tile(x, False) for x in r.tiles) not in text, r
    assert not [r for r in _table_rows(text, "#### b0101-2-sax (`inbox/2026-01-01/b0101-2-sax/`)") if "S1.2" in r]
    assert not [r for r in _table_rows(text, "#### b0103-1-mix (`inbox/2026-01-03/b0103-1-mix/`)") if "K1.2" in r]
    assert len(SHOOTOUT_ROW.findall(text)) == 6 and not PRIVATE_ROW.findall(text)       # no serving position


def test_the_section_tells_nothing_of_the_serving_order(tmp_path, jobs):
    """Two worlds with the same design rounds, served in other orders and with other hidden checks (the repeat shows
    another slot), publish the same section, apart from the sha256 of the files they came from."""
    design = copy.deepcopy(DESIGN)
    _maps, keys_batches = design["day1"]
    name, category, rounds = keys_batches[0]
    keys_batches[0] = (name, category, [rounds[1], rounds[0]])          # K1.2 served first: the repeat now shows K1.1
    _maps, mix_batches = design["day3"]
    name, category, rounds = mix_batches[0]
    mix_batches[0] = (name, category, [rounds[2], rounds[0], rounds[1]])  # the anchor first, the repeat last
    a, b = collect(write(tmp_path / "a", world())), collect(write(tmp_path / "b", world(design=design)))

    def body(text: str) -> list[str]:
        return [x for x in text.splitlines() if " sha256 `" not in x]
    assert body(T.render(a, D)) == body(T.render(b, D))
    assert T.render(a, D) != T.render(b, D)                             # (the files differ)
    assert body(T.render_private(a, D)) != body(T.render_private(b, D))
    assert "repeat of K1.1, first shown b0101-1-keys #2" in T.render_private(b, D)


def test_a_batch_of_hidden_checks_alone_publishes_no_row(tmp_path, jobs):
    design = copy.deepcopy(DESIGN)
    _maps, mix_batches = design["day3"]
    name, category, rounds = mix_batches[0]
    mix_batches[0] = (name, category, rounds[1:])                       # the repeat and the anchor, no design round
    tabs = collect(write(tmp_path, world(design=design)))
    lines = T.render(tabs, D).splitlines()
    i = lines.index("#### b0103-1-mix (`inbox/2026-01-03/b0103-1-mix/`)")
    assert lines[i + 1:i + 4] == ["", "No design round.", ""] and lines[i + 4].startswith("### Drawn, keep3")
    assert RECIPE.findall("\n".join(lines)) == [] and T.hidden_checks(tabs) == 3
    assert len(PRIVATE_ROW.findall(T.render_private(tabs, D))) == 5 + 3


def test_a_map_of_shared_symbols_names_its_design_slots_only_in_the_section(tmp_path, jobs):
    design = copy.deepcopy(DESIGN)
    design["day2"][1][0][2].append(R("V", "V.7", "l7_melancholy_high", 88, [("rise", "sax muted"),
                                                                           ("plagal", "sour copy")],
                                     kind="anchor", broken=2))          # an anchor of a slot with no design round
    tabs = collect(write(tmp_path, world(design=design)))
    assert [m[0] for m in tabs[1].maps] == ["V (V.1, V.7)"] * 2 + ["V (V.4)"] * 2
    assert [m[0] for m in tabs[1].public_maps] == ["V (V.1)"] * 2 + ["V (V.4)"] * 2
    text = T.render(tabs, D)
    assert "| V (V.1) | v1 | `rise` |" in text.splitlines() and "V.7" not in text and T.hidden_checks(tabs) == 4
    assert "| V (V.1, V.7) | v1 | `rise` |" in T.render_private(tabs, D).splitlines()


def test_the_private_file_has_every_round_as_served_with_its_kind(synth):
    text = T.render_private(collect(synth), D)
    lines = text.splitlines()
    assert lines[0] == T.PRIVATE_TITLE and text.endswith("|\n") and not text.endswith("\n\n")
    assert "His approval line is dated 2026-11-05" in text and T.SECTION in text
    assert "| 2 | S1.2 | l9_hopeful_high | anchor of S1.2 | `sax_t2` (sax source) | `sax_a1` (sour copy) **broken** | " \
           "`sax_t1` (sax source) |" in lines
    assert "| 2 | K1.2 | l5_wistful_mid | repeat of K1.2, first shown b0101-1-keys #2 | `keys_b` (rootless) | " \
           "`keys_a` (rootless) | `keys_d` (rootless) |" in lines
    assert "| 1 | d02 | low, wistful | design | 1 | `horn` (`drafts/v1/d02_horn.ls`) | " \
           "`groove` (`drafts/v1/d02_groove.ls`) | `chords` (`drafts/v1/d02_chords.ls`) |" in lines
    assert "| V (V.4) | v1 | `float` |" in lines
    assert [x for x in lines if x.startswith("## ")] == [
        "## Drawn, Day 1 (job 101)", "## Drawn, Day 2 (job 102)", "## Drawn, Day 3 (job 103)",
        "## Drawn, keep3 (job 104)"]
    assert [x for x in lines if x.startswith("### ")] == [
        "### b0101-1-keys (`inbox/2026-01-01/b0101-1-keys/`, 2 rounds)",
        "### b0101-2-sax (`inbox/2026-01-01/b0101-2-sax/`, 2 rounds)",
        "### b0102-1-vamps (`inbox/2026-01-02/b0102-1-vamps/`, 2 rounds)",
        "### b0103-1-mix (`inbox/2026-01-03/b0103-1-mix/`, 3 rounds)",
        "### b0104-keep3 (`inbox/2026-01-04/b0104-keep3/`, 2 rounds)"]
    assert _cells_match_headers(lines) == 3 + 1 + 5
    assert "| W1.1 | l1_melancholy_low | design | `excerpt_e1` (rain excerpt at delta 18) | " \
           "`excerpt_e2` (rain excerpt at delta 18) |  |" in text                        # a shorter round's empty cell
    assert len(PRIVATE_ROW.findall(text)) == 6 + 3


def test_the_output_is_deterministic_and_independent_of_where_the_files_are(synth, tmp_path_factory):
    a, a_full = T.render(collect(synth), D), T.render_private(collect(synth), D)
    assert (a, a_full) == (T.render(collect(synth), D), T.render_private(collect(synth), D))
    docs = world()
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    for rel, doc in docs.items():                           # the same documents, keys reordered and re-indented
        p = elsewhere / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith("manifest.hidden.json"):
            p.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
        else:
            p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    assert (T.render(collect(elsewhere), D), T.render_private(collect(elsewhere), D)) == (a, a_full)
    assert not [p for p in (synth, elsewhere) for x in (a, a_full) if str(p) in x]


def test_no_seed_or_id_reaches_the_output(synth):
    docs = world()
    secrets = secrets_of(docs)
    assert len(secrets) > 60
    for text in (T.render(collect(synth), D), T.render_private(collect(synth), D)):
        assert [s for s in sorted(secrets) if s in text] == []
        assert not re.search(r"\b[0-9a-f]{16}\b", text)                                  # no seed
        assert not re.search(r"\b[rk][0-9a-f]{8,10}\b|\bc[0-9a-f]{16}\b", text)           # no opaque or real id
        assert set(re.findall(r"\b[0-9a-f]{64}\b", text)) == {
            hashlib.sha256((synth / p).read_bytes()).hexdigest()
            for p in docs if p.startswith("runs/")}                                       # only the files' hashes


def test_the_section_leaves_the_gate_files_readers_alone(synth):
    for text in (T.render(collect(synth), D), T.render_private(collect(synth), D)):
        heads = [p["heading"] for p in PICKS.values()] + [G.APPROVAL_HEADING]
        assert not [x for x in text.splitlines() if any(x.startswith(h) for h in heads)]
        assert not [x for x in text.splitlines() if G.APPROVED.match(x.rstrip()) or re.match(r"\s*approved\b", x, re.I)]


# --------------------------------------------------------------------------------------------- cross-checks
def _hidden(docs, day, batch):
    return docs[f"inbox/{DATES[day]}/{batch}/manifest.hidden.json"]


def _public(docs, day, batch):
    return docs[f"inbox/{DATES[day]}/{batch}/manifest.json"]


def _batches(docs, day):
    return docs[f"runs/shootout-{JOBS[day]}/summary/batches.json"]


def _draw(docs, day):
    return docs[f"runs/shootout-{JOBS[day]}/summary/draw.json"]


def _hclip(docs, day, batch, index, pos):
    return _hidden(docs, day, batch)["rounds"][rid(day, batch, index)]["clips"][oid(day, batch, index, pos)]


def _swap(xs, i, j):
    xs[i], xs[j] = xs[j], xs[i]


MUTATIONS = {
    "hidden option": (lambda d: _hclip(d, "day1", "b0101-1-keys", 1, 2).update(option="keys_x"),
                      "b0101-1-keys #1 (K1.1) tile 2: hidden ['option'] differ"),
    "hidden real id": (lambda d: _hclip(d, "day2", "b0102-1-vamps", 2, 1).update(clip="c" + "0" * 16),
                       "b0102-1-vamps #2 (V.4) tile 1: hidden ['clip'] differ"),
    "hidden pcm": (lambda d: _hclip(d, "day3", "b0103-1-mix", 1, 1).update(pcm_sha256="0" * 64),
                   "b0103-1-mix #1 (W1.1) tile 1: hidden ['pcm_sha256'] differ"),
    "hidden kind": (lambda d: _hidden(d, "day1", "b0101-2-sax")["rounds"][rid("day1", "b0101-2-sax", 1)]
                    .update(kind="repeat"), "b0101-2-sax #1 (S1.1): hidden kind 'repeat'"),
    "hidden round missing": (lambda d: _hidden(d, "day1", "b0101-2-sax")["rounds"].pop(rid("day1", "b0101-2-sax", 2)),
                             "manifest.hidden.json holds 1 rounds, not the batch's 2"),
    "hidden category": (lambda d: _hidden(d, "day2", "b0102-1-vamps").update(category="keys"),
                        "b0102-1-vamps: hidden category 'keys'"),
    "public order": (lambda d: _swap(_public(d, "day1", "b0101-1-keys")["rounds"], 0, 1),
                     "b0101-1-keys: manifest.json serves its rounds in another order"),
    "public tiles": (lambda d: _swap(_public(d, "day3", "b0103-1-mix")["rounds"][1]["clips"], 0, 2),
                     "b0103-1-mix #2: manifest.json plays other clips or another tile order"),
    "public flac": (lambda d: _public(d, "day1", "b0101-2-sax")["rounds"][0]["clips"][0].update(sha256="0" * 64),
                    "b0101-2-sax #1: a tile's FLAC sha256 is not the batch's"),
    "public bpm": (lambda d: _public(d, "day2", "b0102-1-vamps")["rounds"][0].update(bpm=77),
                   "b0102-1-vamps #1 (V.1): manifest.json bpm 77"),
    "draw tile": (lambda d: _draw(d, "day1")["batches"][0]["rounds"][1]["clips"][0].update(option="keys_x"),
                  "b0101-1-keys #2 (K1.2): the draw's tiles are not the batches file's"),
    "draw slot": (lambda d: _draw(d, "day2")["batches"][0]["rounds"][1].update(slot="V.5"),
                  "b0102-1-vamps #2 (V.4): the draw's slot 'V.5'"),
    "anchor tile": (lambda d: _draw(d, "day1")["batches"][1]["rounds"][1].update(anchor_pos=3),
                    "b0101-2-sax #2 (S1.2): the anchor's broken tile [2] is not the draw's anchor_pos 3"),
    "anchor without clip": (lambda d: _batches(d, "day3")["batches"]["b0103-1-mix"]["rounds"][2].update(anchor_clip=None),
                            "b0103-1-mix #3 (W1.1): kind 'anchor' with anchor clip None"),
    "design with an anchor clip": (lambda d: _batches(d, "day1")["batches"]["b0101-1-keys"]["rounds"][0]
                                   .update(anchor_clip=oid("day1", "b0101-1-keys", 1, 1)),
                                   "b0101-1-keys #1 (K1.1): kind 'none' with anchor clip"),
    "repeat of nothing": (lambda d: [x.update(of="r00000000") for x in (
        _batches(d, "day3")["batches"]["b0103-1-mix"]["rounds"][1], _draw(d, "day3")["batches"][0]["rounds"][1],
        _hidden(d, "day3", "b0103-1-mix")["rounds"][rid("day3", "b0103-1-mix", 2)])],
        "b0103-1-mix #2 (K1.2): its first showing does not resolve"),
    "repeat elsewhere": (lambda d: _draw(d, "day3")["batches"][0]["rounds"][1].update(of_batch="b0101-2-sax"),
                         "its first showing is b0101-1-keys of shootout-101, the draw says 'b0101-2-sax'"),
    "repeat not reversed": (lambda d: [x.update(clip="c" + "1" * 16) for x in (
        _batches(d, "day3")["batches"]["b0103-1-mix"]["rounds"][1]["clips"][0], _hclip(d, "day3", "b0103-1-mix", 2, 1))],
        "b0103-1-mix #2 (K1.2): its clips are not its first showing's in reverse order"),
    "option not in the maps": (lambda d: _draw(d, "day1")["maps"]["K1"][0].pop("P2b"),
                               "Day 1 (job 101) K1.2: options ['keys_d'] are in none of K1's maps"),
    "shared symbols fit two maps": (lambda d: _draw(d, "day2")["maps"]["V"].append({"v1": "float", "v2": "step"}),
                                    "Day 2 (job 102) V.4: its options fit 2 of V's maps, not one"),
    "run name": (lambda d: _batches(d, "day2").update(run="shootout-999"),
                 "Day 2 (job 102): draw stage 'day2', batches stage 'day2' and run 'shootout-999'"),
    "draw stage": (lambda d: _draw(d, "day3").update(stage="day2"), "Day 3 (job 103): draw stage 'day2'"),
    "batch names": (lambda d: _draw(d, "day1")["batches"].pop(),
                    "Day 1 (job 101): the draw holds batches ['b0101-1-keys'], the batches file"),
    "keep3 square": (lambda d: d["runs/drafts-104/summary/draw.json"].update(square=[[0, 1, 2], [2, 1, 0], [2, 0, 1]]),
                     "b0104-keep3 #1 (d02): its tiles ['horn', 'groove', 'chords'] are not row 1 of the square"),
    "keep3 draw tiles": (lambda d: d["runs/drafts-104/summary/draw.json"]["rounds"][1].update(row=2),
                         "b0104-keep3 #2 (d01): the draw's brief, row and tiles are not the batch file's"),
    "keep3 hidden persona": (lambda d: _hidden(d, "keep3", "b0104-keep3")["rounds"][rid("keep3", "b0104-keep3", 1)]
                             ["clips"][oid("keep3", "b0104-keep3", 1, 3)].update(persona="horn"),
                             "b0104-keep3 #1 (d02) tile 3: hidden ['persona'] differ"),
    "keep3 hidden cell": (lambda d: _hidden(d, "keep3", "b0104-keep3")["rounds"][rid("keep3", "b0104-keep3", 2)]
                          ["clips"][oid("keep3", "b0104-keep3", 2, 1)].update(mood="hopeful"),
                          "b0104-keep3 #2 (d01): its clips name 2 energy and mood cells"),
}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_a_disagreement_between_the_files_refuses_naming_where(tmp_path, name):
    docs = world()
    mutate, want = MUTATIONS[name]
    mutate(docs)
    write(tmp_path, docs)
    with pytest.raises(T.TablesError) as err:
        collect(tmp_path)
    assert not isinstance(err.value, T.BadInput)
    assert any(want in p for p in err.value.problems), err.value.problems


def test_a_repeat_must_come_from_an_earlier_job(tmp_path):
    later = {**JOBS, "day1": 301}                          # Day 1's job id above Day 3's: not an earlier showing
    write(tmp_path, world(later))
    with pytest.raises(T.TablesError) as err:
        collect(tmp_path, later)
    assert err.value.problems == ["Day 3 (job 103) b0103-1-mix #2 (K1.2): its first showing is b0101-1-keys of "
                                  "shootout-301, the draw says 'b0101-1-keys' of 'shootout-301'"]


def test_every_problem_is_listed_at_once(tmp_path):
    docs = world()
    for name in ("hidden option", "public bpm", "keep3 hidden persona"):
        MUTATIONS[name][0](docs)
    write(tmp_path, docs)
    with pytest.raises(T.TablesError) as err:
        collect(tmp_path)
    assert len(err.value.problems) == 3


@pytest.mark.parametrize("gone", ["inbox/2026-01-02/b0102-1-vamps/manifest.hidden.json",
                                  "inbox/2026-01-04/b0104-keep3/manifest.json",
                                  "runs/shootout-103/summary/batches.json", "runs/drafts-104/summary/draw.json"])
def test_a_missing_file_is_bad_input(synth, gone):
    (synth / gone).unlink()
    with pytest.raises(T.BadInput) as err:
        collect(synth)
    assert gone.split("/")[-1] in str(err.value)
    gate_file(synth)
    assert T.main(["--runs", str(synth / "runs"), "--inbox", str(synth / "inbox"),
                   "--gate", str(synth / "gates" / "GATE_palette.md")]) == 2


def test_a_file_of_another_shape_is_bad_input(synth):
    p = synth / "runs/shootout-102/summary/batches.json"
    doc = json.loads(p.read_text())
    del doc["batches"]["b0102-1-vamps"]["rounds"][0]["clips"][0]["pos"]
    p.write_text(json.dumps(doc))
    with pytest.raises(T.BadInput, match="not the shootout job's shape"):
        collect(synth)


# --------------------------------------------------------------------------------------------- the APPROVED line
@pytest.mark.parametrize("approval", ["", "\n", "APPROVED by Shresth Rana <yyyy-mm-dd>\n",
                                      f"{LINE}\n{LINE}\n", "approved by shresth rana 2026-11-05\n",
                                      "APPROVED by Shresth Rana 2026-02-30\n", f"<!--\n{LINE}\n-->\n",
                                      f"~~~~\n```\n{LINE}\n```\n~~~~\n"])
def test_it_refuses_before_his_approved_line(synth, capsys, private_root, approval):
    gate = gate_file(synth, approval)
    before = gate.read_bytes()
    for args in (["--gate", str(gate)], ["--write", str(gate)], ["--gate", str(gate), "--private"],
                 ["--write", str(gate), "--private"]):
        assert T.main(argv(synth, *args)) == 3
        out = capsys.readouterr()
        assert out.out == "" and "published after the freeze" in out.err
    assert gate.read_bytes() == before and not private_file(private_root).exists()


def test_his_line_in_the_how_to_section_does_not_count(synth):
    p = synth / "gates" / "GATE_palette.md"
    p.parent.mkdir(parents=True)
    p.write_text(GATE_HEAD.replace("## How to approve\n", f"## How to approve\n{LINE}\n"), encoding="utf-8")
    assert T.main(argv(synth, "--write", str(p), "--private")) == 3


def test_it_prints_the_section_after_his_line(synth, capsys, private_root):
    gate = gate_file(synth, f"His line, given in chat on 2026-11-05, entered verbatim by Claude.\n{LINE}\n")
    assert T.main(argv(synth, "--gate", str(gate))) == 0
    assert capsys.readouterr().out == T.render(collect(synth), D)
    assert not private_file(private_root).exists()                         # printing writes nothing


# --------------------------------------------------------------------------------------------- --private
def test_private_writes_the_full_tables_under_data_and_never_prints_them(synth, capsys, private_root):
    gate = gate_file(synth)
    before = gate.read_bytes()
    full = T.render_private(collect(synth), D)
    assert T.main(argv(synth, "--gate", str(gate), "--private")) == 0
    out = private_file(private_root)
    assert out.read_text(encoding="utf-8") == full
    printed = capsys.readouterr()
    assert printed.out.startswith("wrote the full tables (3 hidden-check rounds included) to ")
    assert hashlib.sha256(full.encode()).hexdigest() in printed.out
    assert not [x for x in full.splitlines() if x.startswith("|") and x in printed.out + printed.err]   # no row shown
    assert gate.read_bytes() == before                                       # --private alone leaves the gate alone
    assert [p.name for p in out.parent.iterdir()] == [out.name]              # no temporary file left
    assert out.stat().st_mode & 0o777 == 0o600                               # as private as the votes
    assert T.main(argv(synth, "--gate", str(gate), "--private")) == 0 and out.read_text(encoding="utf-8") == full
    assert "unchanged" in capsys.readouterr().out
    named = private_root / "elsewhere.md"                                    # any file under data/
    assert T.main(argv(synth, "--gate", str(gate), "--private", str(named))) == 0
    assert named.read_text(encoding="utf-8") == full


@pytest.mark.parametrize("where", ["outside", "dotdot", "symlink-out", "the-data-folder", "no-folder", "a-folder"])
def test_private_refuses_a_path_outside_data_or_without_its_folder(synth, capsys, private_root, tmp_path_factory,
                                                                  where):
    gate = gate_file(synth)
    before = gate.read_bytes()
    outside = tmp_path_factory.mktemp("outside")
    path = {"outside": outside / "t.md", "dotdot": private_root / ".." / "t.md",
            "symlink-out": private_root / "votes" / "audit" / "link.md", "the-data-folder": private_root,
            "no-folder": private_root / "votes" / "nowhere" / "t.md",
            "a-folder": private_root / "votes" / "audit"}[where]
    if where == "symlink-out":
        path.symlink_to(outside / "target.md")
    for args in (["--gate", str(gate)], ["--write", str(gate)]):
        assert T.main(argv(synth, *args, "--private", str(path))) == 2
        assert "error: --private" in capsys.readouterr().err
    assert gate.read_bytes() == before and list(outside.iterdir()) == []
    assert not (private_root.parent / "t.md").exists() and not (private_root / "votes" / "nowhere").exists()
    assert sorted(p.name for p in (private_root / "votes" / "audit").iterdir()) == (
        ["link.md"] if where == "symlink-out" else [])


def test_the_private_record_is_never_rewritten(synth, capsys, private_root):
    gate = gate_file(synth)
    before = gate.read_bytes()
    out = private_file(private_root)
    out.write_text("other tables\n", encoding="utf-8")
    for args in (["--gate", str(gate), "--private"], ["--write", str(gate), "--private"], ["--write", str(gate)]):
        assert T.main(argv(synth, *args)) == 3
        assert "already holds other full tables" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "other tables\n" and gate.read_bytes() == before   # neither written


def test_write_needs_the_full_tables_recorded_privately_first(synth, capsys, private_root):
    gate = gate_file(synth)
    before = gate.read_bytes()
    assert T.main(argv(synth, "--write", str(gate))) == 3
    assert "give --private too" in capsys.readouterr().err
    assert gate.read_bytes() == before and not private_file(private_root).exists()
    assert T.main(argv(synth, "--gate", str(gate), "--private")) == 0        # the record first
    assert T.main(argv(synth, "--write", str(gate))) == 0                    # then the section on its own
    assert T.render(collect(synth), D) in gate.read_text(encoding="utf-8")


def test_the_private_file_lives_in_the_data_folder_git_and_the_sync_ignore(monkeypatch):
    assert REAL_PRIVATE_ROOT == ROOT / "data"
    monkeypatch.setattr(T, "PRIVATE_ROOT", REAL_PRIVATE_ROOT)
    assert T.private_default() == ROOT / "data" / "votes" / "audit" / "drawn_round_tables.md"
    for name in (".gitignore", ".rsyncignore"):
        assert "data/" in (ROOT / name).read_text(encoding="utf-8").split(), name


# --------------------------------------------------------------------------------------------- --write
def test_write_puts_the_section_right_after_the_approval_section(synth, capsys, private_root):
    gate = gate_file(synth)
    old = gate.read_text()
    args = argv(synth, "--write", str(gate), "--private")
    assert T.main(args) == 0
    section = T.render(collect(synth), D)
    new = gate.read_text()
    assert new == old + "\n" + section
    assert G.approval_date(new) == "2026-11-05"
    assert private_file(private_root).read_text(encoding="utf-8") == T.render_private(collect(synth), D)
    printed = capsys.readouterr().out
    assert "wrote the full tables" in printed and f"wrote `{T.SECTION}`" in printed
    assert T.main(args) == 0 and gate.read_text() == new                   # the same tables again: unchanged
    assert capsys.readouterr().out.count("unchanged") == 2


def test_write_keeps_a_later_section_after_the_tables(synth):
    tail = "\n## Sign-off\n\nHeard 10 of 10.\n"
    gate = gate_file(synth, LINE + "\n", after=tail)
    assert T.main(argv(synth, "--write", str(gate), "--private")) == 0
    section = T.render(collect(synth), D)
    new = gate.read_text()
    assert new == GATE_HEAD + LINE + "\n\n" + section + "\n## Sign-off\n\nHeard 10 of 10.\n"   # the blank line kept
    assert G.approval_date(new) == "2026-11-05"
    assert T.insert(new, section) == (new, False)


def test_the_section_goes_where_the_gate_reader_ends_the_approval_section():
    text = ("# G\n\n## Approval\n\n" + LINE + "\n<!--\n## Not a heading\n-->\n```\n## Nor this\n```\n\n"
            "## Later\nx\n")
    assert T.sections(text, G.APPROVAL_HEADING) == G.sections(text, G.APPROVAL_HEADING) == [(2, 12)]
    assert T.sections(text, "## Not a heading") == T.sections(text, "## Nor this") == []
    new, changed = T.insert(text, T.SECTION + "\n\nbody\n")
    assert changed and new == text.replace("\n## Later\n", "\n" + T.SECTION + "\n\nbody\n\n## Later\n")
    assert G.approval_date(new) == "2026-11-05" and T.sections(new, T.SECTION) == [(12, 16)]
    commented = "## Approval\n" + LINE + "\n<!--\n" + T.SECTION + "\n\nold\n-->\n"     # a section in a comment is none
    assert T.insert(commented, T.SECTION + "\n\nbody\n") == (commented + "\n" + T.SECTION + "\n\nbody\n", True)


def test_insert_keeps_every_other_byte():
    text = "# G\n\n## Approval\n\n" + LINE + "\n\n\n## Later\nx\n"
    new, changed = T.insert(text, "## Drawn round tables (published after the freeze)\n\nbody\n")
    assert changed and new == ("# G\n\n## Approval\n\n" + LINE + "\n\n\n"
                               "## Drawn round tables (published after the freeze)\n\nbody\n\n## Later\nx\n")
    no_newline = "## Approval\n" + LINE
    assert T.insert(no_newline, T.SECTION + "\n")[0] == no_newline + "\n\n" + T.SECTION + "\n"


def test_write_refuses_other_tables_already_in_the_file(synth, private_root):
    gate = gate_file(synth, LINE + "\n\n" + T.SECTION + "\n\nsomething else\n")
    before = gate.read_bytes()
    assert T.main(argv(synth, "--write", str(gate), "--private")) == 3
    assert gate.read_bytes() == before and not private_file(private_root).exists()   # a refusal writes neither


def test_write_and_gate_must_name_one_file(synth, tmp_path_factory):
    gate = gate_file(synth)
    other = tmp_path_factory.mktemp("other") / "GATE_palette.md"
    other.write_text(gate.read_text())
    assert T.main(argv(synth, "--gate", str(gate), "--write", str(other), "--private")) == 2
    assert other.read_text() == gate.read_text()


def test_a_cross_check_failure_writes_nothing(tmp_path, jobs, private_root):
    docs = world()
    MUTATIONS["hidden option"][0](docs)
    write(tmp_path, docs)
    gate = gate_file(tmp_path)
    before = gate.read_bytes()
    assert T.main(argv(tmp_path, "--write", str(gate), "--private")) == 3
    assert gate.read_bytes() == before and not private_file(private_root).exists()


# --------------------------------------------------------------------------------------------- the real files
def real_copy(tmp: Path) -> tuple[Path, list[Path]]:
    """Copies of the real jobs' summaries, their batches' manifests and the gate file under `tmp` (skip when `make
    pull` has not brought them to this machine). Returns (root, the batch folders)."""
    files = [f"runs/shootout-{REAL_JOBS[d]}/summary/{f}" for d in ("day1", "day2", "day3")
             for f in ("draw.json", "batches.json")]
    files += [f"runs/drafts-{REAL_JOBS['keep3']}/summary/{f}" for f in ("draw.json", "batch.json")]
    for rel in files:
        if not (ROOT / rel).is_file():
            pytest.skip(f"{rel} is not on this machine (make pull brings it to the laptop)")
    folders = []
    for rel in files:
        doc = json.loads((ROOT / rel).read_text())
        if rel.endswith("batches.json"):
            folders += [b["dir"] for b in doc["batches"].values()]
        elif rel.endswith("batch.json"):
            folders.append(doc["dir"])
    for f in folders:
        for name in ("manifest.json", "manifest.hidden.json"):
            if not (ROOT / "inbox" / f / name).is_file():
                pytest.skip(f"inbox/{f}/{name} is not on this machine (make pull brings it to the laptop)")
            files.append(f"inbox/{f}/{name}")
    files.append("gates/GATE_palette.md")
    for rel in files:
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp / rel)
    return tmp, [tmp / "inbox" / f for f in folders]


def test_on_copies_of_the_real_runs_inbox_and_gate(tmp_path, capsys, private_root):
    root, folders = real_copy(tmp_path)
    gate = root / "gates" / "GATE_palette.md"
    args = argv(root, "--write", str(gate), "--private")
    real_gate = gate.read_text()
    if G.approval_date(real_gate) is None:                 # before the freeze: it refuses and writes nothing
        assert T.main(args) == 3 and gate.read_text() == real_gate and not private_file(private_root).exists()
        # his line goes inside the Approval section, wherever that section sits and whatever follows it
        (_head, end), keep = T.sections(real_gate, G.APPROVAL_HEADING)[0], real_gate.splitlines(keepends=True)
        rest = "".join(keep[end:])
        gate.write_text("".join(keep[:end]).rstrip("\n") + "\n\n" + LINE + "\n" + ("\n" + rest if rest else ""),
                        encoding="utf-8")
    old = gate.read_text()
    capsys.readouterr()
    assert T.main(args) == 0, capsys.readouterr().err
    new = gate.read_text()
    tabs = T.collect(root / "runs", root / "inbox", REAL_JOBS)
    when = date.fromisoformat(G.approval_date(old))
    section = T.render(tabs, when)
    assert new == T.insert(old, section)[0] and section in new          # (after the freeze: already there, unchanged)
    assert T.sections(new, T.SECTION)[0][0] == T.sections(new, G.APPROVAL_HEADING)[0][1]   # right after Approval
    if T.SECTION not in old:
        assert new.replace(section, "", 1).split() == old.split()      # and nothing else moved
    assert G.approval_date(new) == G.approval_date(old)
    assert G.picks_problems(new) == G.picks_problems(old)  # the gate file's pick reader sees what it saw
    # the section: the design rounds only; the private file: every round
    design = sum(1 for t in tabs if t.key != "keep3" for b in t.batches for r in b.rows if r.kind == "design")
    assert RECIPE.findall(section) == [] and len(SHOOTOUT_ROW.findall(section)) == design
    assert not PRIVATE_ROW.findall(section) and T.hidden_checks(tabs) > 0
    assert f"- Left out: {T.hidden_checks(tabs)} rounds, the hidden checks" in section
    assert not [r for t in tabs for b in t.batches for r in b.rows if r.kind != "design"
                and f"| {r.slot} | {r.loop} | " + " | ".join(T._tile(x, False) for x in r.tiles) in section]
    full = private_file(private_root).read_text(encoding="utf-8")
    assert full == T.render_private(tabs, when)
    assert len(PRIVATE_ROW.findall(full)) == design + T.hidden_checks(tabs)
    assert len(section.splitlines()) > 100
    assert [t.job for t in tabs] == [REAL_JOBS[k] for k in ("day1", "day2", "day3", "keep3")]
    assert sum(len(b.rows) for t in tabs for b in t.batches) == sum(
        len(json.loads((f / "manifest.json").read_text())["rounds"]) for f in folders)
    assert [x[5:].split(" (")[0] for x in section.splitlines() if x.startswith("#### ")] == [f.name for f in folders]
    docs = {str(p.relative_to(root)): json.loads(p.read_text()) for p in root.rglob("*.json")}
    secrets = secrets_of(docs)
    assert len(secrets) > 500 and [s for s in secrets if s in section] == [] and [s for s in secrets if s in full] == []
    assert T.main(args) == 0 and gate.read_text() == new                   # deterministic: a second write changes nothing
    assert private_file(private_root).read_text(encoding="utf-8") == full


def test_on_real_copies_a_changed_hidden_manifest_refuses(tmp_path):
    root, folders = real_copy(tmp_path)
    hidden = folders[0] / "manifest.hidden.json"
    doc = json.loads(hidden.read_text())
    first_round = next(iter(doc["rounds"].values()))
    first_clip = next(iter(first_round["clips"].values()))
    first_clip["option"] = first_clip["option"] + "_x"
    hidden.write_text(json.dumps(doc))
    with pytest.raises(T.TablesError) as err:
        T.collect(root / "runs", root / "inbox", REAL_JOBS)
    assert len(err.value.problems) == 1 and "hidden ['option'] differ" in err.value.problems[0]
    assert folders[0].name in err.value.problems[0]


def test_the_cli_defaults_are_the_repo_paths(monkeypatch):
    seen = {}

    def fake_collect(runs, inbox, jobs):
        seen.update(runs=runs, inbox=inbox, jobs=jobs)
        raise T.TablesError(["stop here"])
    monkeypatch.setattr(T, "collect", fake_collect)
    monkeypatch.setattr(T, "approval", lambda text: date(2026, 11, 5))
    assert T.main([]) == 3
    assert seen == {"runs": ROOT / "runs", "inbox": ROOT / "inbox", "jobs": REAL_JOBS} and seen["jobs"] is REAL_JOBS
