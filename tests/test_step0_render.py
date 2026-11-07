"""The step-0 render step (lofi_rl.step0.render; docs/design/step0_samples.md §2.5, §3, §6.1) with junk.render_clip
stubbed, as tests/test_junk.py renders: the stub writes what the real one writes (a FLAC stand-in and a manifest with
the render config, brief, palette and canonical sha the clip id reads) and returns its record, with the outcomes
tests/helpers_step0.py's table gives (G7, an error once, an error twice).

What these tests pin (§6.1): round 1's 160; a G7 failure leading to a next round; an error retried once and rendered
(errors_retried 1, errors_final 0) and an error twice (undetermined, exit 1); the deadline; a partial run refused;
render_code recorded; the pool's selection with the held-out-text exclusion; --from writing only under --run; the two
lanes keeping the round's order; and, end to end, generate, score and render on fakes giving the synthetic-run
helper's files. With FluidSynth and the frozen palette's sources (LOFI_ASSETS), two real renders whose clip ids are
lofi_rl.shootout.render.clip_id's and lofi_rl.scoring.protocol.clip_id's; skipped otherwise.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import threading
from collections import Counter
from pathlib import Path

import pytest

import helpers_step0 as H
import helpers_step0_engine as E
import lofi_rl.step0 as S
from lofi_rl.brief import Brief
from lofi_rl.compile.timeline import loop_samples
from lofi_rl.palette import load_palette
from lofi_rl.rl import junk as J
from lofi_rl.scoring import protocol
from lofi_rl.step0 import cli
from lofi_rl.step0 import generate as G
from lofi_rl.step0 import render as Rd
from lofi_rl.step0 import score as Sc
from lofi_rl.step0 import select as X

NAME = f"step0-{H.JOB}"
TWIN = ("5101:0", "596:177")            # a held-out brief and its train twin (brief_streams_v1.json, twins)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def doc_of(path: Path):
    return json.loads(path.read_text())


def rows_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def hashes(folder: Path) -> dict[str, str]:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def render_cfg() -> dict:
    from lofi_rl.scoring.render_pool import render_config
    return render_config()


def outcomes_of(run: H.Run, table: dict) -> dict[int, str]:
    """i -> its render outcome in the helper's table ("G7", "E", "EE"; "" renders)."""
    order = {(r["brief_id"], r["k"]): r["i"] for r in run.rows}
    out = {}
    for bid, specs in table.items():
        for k, spec in enumerate(specs.split()):
            out[order[(bid, k)]] = spec.partition("/")[2]
    return out


class Stub:
    """junk.render_clip as tests/helpers_step0.py's records have it: an ok render writes clips/<i>.flac (a stand-in) and
    a manifest, and returns the helper's numbers; G7, an error once (E) and an error twice (EE) as the table says."""

    def __init__(self, scored: list[dict], outcomes: dict[int, str] | None = None, *, on_call=None):
        self.scored = {s["i"]: s for s in scored}
        self.outcomes, self.on_call = outcomes or {}, on_call
        self.tried, self.calls, self.lock = Counter(), [], threading.Lock()

    def __call__(self, row, palette, clips_dir, *, cache, workers):
        i = row["i"]
        with self.lock:
            self.tried[i] += 1
            self.calls.append({"i": i, "workers": workers, "cache": cache})
            n_try = self.tried[i]
        if self.on_call is not None:
            self.on_call(row)
        spec = self.outcomes.get(i, "")
        if spec == "G7":
            return {"i": i, "ok": False, "gate": "G7", "detail": "synthetic: the render is out of spec"}
        if spec == "compile":
            return {"i": i, "ok": False, "gate": "compile", "detail": "synthetic: the palette cannot play it"}
        if spec == "raise" and n_try == 1:
            raise OSError("synthetic: the disk is full")
        if (spec == "E" and n_try == 1) or spec == "EE":
            return {"i": i, "ok": False, "gate": None, "error": "OSError: synthetic: fluidsynth died"}
        b, s = Brief(**row["brief"]), self.scored[i]
        cfg = render_cfg()
        manifest = {"schema": cfg["schema"], "chain": cfg["chain"],
                    "loop": {"canon_sha256": s["canon_sha256"], "p_fix": s["p_fix"]},
                    "brief": {"scene": b.scene, "energy": b.energy, "mood": b.mood, "mode": b.mode, "bpm": b.bpm,
                              "tonic": b.tonic, "key": b.key_name},
                    "palette": {"id": palette.id, "sha256": palette.sha256, "frozen": True},
                    "audio_sha256": sha(f"pcm {i}")}
        (clips_dir / f"{i:04d}.flac").write_bytes(b"fLaC a stand-in")
        (clips_dir / f"{i:04d}.manifest.json").write_text(json.dumps(manifest))
        n = loop_samples(b.bpm)
        return {"i": i, "ok": True, "gate": None, "clip": f"clips/{i:04d}.flac", "n": n, "seconds": round(n / 44100, 3),
                "lufs": -14.0, "true_peak_dbtp": -1.5, "audio_sha256": sha(f"pcm {i}"),
                "canon_sha256": s["canon_sha256"], "p_fix": s["p_fix"], "render_s": 2.08, "flac_bytes": 2562058}


@pytest.fixture
def stubbed(monkeypatch):
    """The render step with its audio stubbed: the palette read without hashing its fetch-only sources, the render-code
    key the helper's, and junk.render_clip a Stub (installed by `use`)."""
    real = load_palette
    monkeypatch.setattr(Rd, "load_palette", lambda path, verify=True: real(path, verify=False))
    monkeypatch.setattr(Rd, "source_sha8", lambda: H.SOURCE_SHA8)
    monkeypatch.setattr(Rd, "sync_stamp", lambda: H.STAMP)

    def use(stub: Stub) -> Stub:
        monkeypatch.setattr(J, "render_clip", stub)
        return stub
    return use


def make(tmp_path, table: dict, name: str = NAME) -> H.Run:
    """The helper's run of `table` (those briefs only), its clips.json and sets.json taken away."""
    run = H.make_run(tmp_path / name, table, briefs=list(table))
    (run.summary / "clips.json").unlink()
    (run.summary / "sets.json").unlink()
    return run


def render(run_dir: Path, *args: str) -> int:
    return cli.main(["render", "--run", str(run_dir), *args])


def one(doc: dict, name: str, bid: str) -> dict:
    (e,) = [e for e in doc[name] if e["brief_id"] == bid]
    return e


def ks(entry: dict) -> list[int]:
    return [t["k"] for t in entry["taken"]]


# --------------------------------------------------------------------------------------------- the rounds
def test_round_one_renders_each_held_out_briefs_first_four_distinct_candidates(tmp_path, stubbed):
    run = H.make_run(tmp_path / NAME)                   # every request: 3,520 rows, nothing short
    want = {n: doc_of(run.summary / n) for n in ("clips.json", "sets.json", "step0.json")}
    stub = stubbed(Stub(run.scored))
    assert render(run.dir) == 0
    assert len(stub.calls) == 160 and {c["workers"] for c in stub.calls} == {2} and {c["cache"] for c in stub.calls} \
        == {None}, "two at a time with 2 workers each on the job's 4 CPUs"
    clips = doc_of(run.summary / "clips.json")
    assert clips == want["clips.json"], "the helper's records, round by round"
    assert (clips["jobs"], clips["workers"]) == (2, 2) and {r["round"] for r in clips["clips"]} == {1}
    assert doc_of(run.summary / "sets.json") == want["sets.json"]
    step0 = doc_of(run.summary / "step0.json")
    assert step0 == want["step0.json"], "render, render_code, sets and render_stamp as the helper fills them"
    assert step0["render"] == {"attempted": 160, "rendered": 160, "G7": 0, "other_gates": 0, "errors_final": 0,
                               "errors_retried": 0, "skipped": 0, "rounds": 1}
    sets = doc_of(run.summary / "sets.json")
    assert [(len(sets[n]), sum(len(e["taken"]) for e in sets[n])) for n in S.SETS] == [(40, 160), (400, 1600),
                                                                                       (220, 660)]
    assert sorted(p.name for p in (run.dir / "clips").glob("*.flac")) == [f"{i:04d}.flac" for i in sorted(
        r["i"] for r in clips["clips"])]


def test_a_g7_failure_leads_to_a_next_round(tmp_path, stubbed):
    table = {"5101:0": "a b c d/G7 e f g h", "5101:1": "h0 h1 h2 h3 h4 h5 h6 h7"}
    run = make(tmp_path, table)
    stub = stubbed(Stub(run.scored, outcomes_of(run, table)))
    assert render(run.dir) == 0
    clips = doc_of(run.summary / "clips.json")["clips"]
    assert [(r["brief_id"], r["k"], r["round"]) for r in clips] == [("5101:0", k, 1) for k in range(4)] + [
        ("5101:1", k, 1) for k in range(4)] + [("5101:0", 4, 2)]
    assert [r["final"] for r in clips] == [True] * 9 and clips[3]["gate"] == "G7" and clips[3]["clip_id"] is None
    sets = doc_of(run.summary / "sets.json")
    h = one(sets, "heldout", "5101:0")
    assert ks(h) == [0, 1, 2, 4] and h["not_taken"] == [{"k": 3, "why": "G7"}] and h["short"] is False
    blk = doc_of(run.summary / "step0.json")["render"]
    assert (blk["G7"], blk["rounds"], blk["rendered"], blk["attempted"]) == (1, 2, 8, 9)
    assert len(stub.calls) == 9


def test_an_error_is_retried_once_and_rendered(tmp_path, stubbed):
    table = {"5101:0": "a b/E c d e f g h"}
    run = make(tmp_path, table)
    stubbed(Stub(run.scored, outcomes_of(run, table)))
    assert render(run.dir) == 0
    clips = doc_of(run.summary / "clips.json")["clips"]
    assert [(r["k"], r["round"], r["final"], r["ok"]) for r in clips] == [
        (0, 1, True, True), (1, 1, False, False), (2, 1, True, True), (3, 1, True, True), (1, 2, True, True)]
    blk = doc_of(run.summary / "step0.json")["render"]
    assert (blk["errors_retried"], blk["errors_final"], blk["rounds"]) == (1, 0, 2)
    assert ks(one(doc_of(run.summary / "sets.json"), "heldout", "5101:0")) == [0, 1, 2, 3], "the retry counts"


def test_an_error_twice_leaves_the_brief_undetermined_and_exits_1(tmp_path, stubbed, capsys):
    table = {"5101:0": "a b/EE c d e f g h", "5101:1": "h0 h1 h2 h3 h4 h5 h6 h7"}
    run = make(tmp_path, table)
    stubbed(Stub(run.scored, outcomes_of(run, table)))
    assert render(run.dir) == 1
    assert "1 held-out briefs are undetermined (first 5101:0)" in capsys.readouterr().err
    sets = doc_of(run.summary / "sets.json")
    u = one(sets, "heldout", "5101:0")
    assert u["undetermined"] is True and u["short"] is False and ks(u) == [0], "an error twice decides nothing past it"
    assert ks(one(sets, "heldout", "5101:1")) == [0, 1, 2, 3]
    blk = doc_of(run.summary / "step0.json")["render"]
    assert (blk["errors_retried"], blk["errors_final"]) == (0, 1)
    assert doc_of(run.summary / "step0.json")["sets"]["heldout"]["undetermined"] == 1


def test_an_exception_out_of_render_clip_is_an_infrastructure_error(tmp_path, stubbed):
    table = {"5101:0": "a b c d e f g h"}
    run = make(tmp_path, table)
    order = {r["k"]: r["i"] for r in run.rows}
    stubbed(Stub(run.scored, {order[2]: "raise"}))
    assert render(run.dir) == 0
    clips = doc_of(run.summary / "clips.json")["clips"]
    first = next(r for r in clips if r["k"] == 2)
    assert first == {"i": order[2], "ok": False, "gate": None, "error": "OSError: synthetic: the disk is full",
                     "brief_id": "5101:0", "k": 2, "round": 1, "final": False, "clip_id": None,
                     "render_cfg_sha8": protocol.render_cfg_sha8(render_cfg())}
    assert doc_of(run.summary / "step0.json")["render"]["errors_retried"] == 1


def test_a_render_that_fails_another_gate_is_not_taken_and_exits_1(tmp_path, stubbed):
    table = {"5101:0": "a b c d e f g h"}
    run = make(tmp_path, table)
    order = {r["k"]: r["i"] for r in run.rows}
    stubbed(Stub(run.scored, {order[1]: "compile"}))
    assert render(run.dir) == 1, "score passed it: a defect between score and render"
    assert ks(one(doc_of(run.summary / "sets.json"), "heldout", "5101:0")) == [0, 2, 3, 4]
    assert doc_of(run.summary / "step0.json")["render"]["other_gates"] == 1


def test_no_render_starts_after_the_deadline(tmp_path, stubbed, monkeypatch):
    """One at a time (--workers 1), each render moving the clock 100 s on from the step's start: against a 250-s
    deadline the first three render and the rest of round 1 is skipped, final, leaving both briefs undetermined."""
    table = {"5101:0": "a b c d e f g h", "5101:1": "h0 h1 h2 h3 h4 h5 h6 h7"}
    run = make(tmp_path, table)
    clock = [0.0]
    monkeypatch.setattr(Rd, "_now", lambda: clock[0])

    def tick(row):
        clock[0] += 100.0
    stub = stubbed(Stub(run.scored, on_call=tick))
    assert render(run.dir, "--workers", "1", "--deadline-s", "250") == 1
    clips = doc_of(run.summary / "clips.json")["clips"]
    assert len(stub.calls) == 3 and [bool(r["ok"]) for r in clips] == [True] * 3 + [False] * 5
    assert all(r.get("skipped") == "render deadline" and r["final"] for r in clips[3:])
    assert [list(r) for r in clips[3:]] == [[*S.RENDER_CLIP_FIELDS["skipped"], *S.CLIP_ADDED]] * 5
    sets = doc_of(run.summary / "sets.json")
    assert all(e["undetermined"] for e in sets["heldout"]) and doc_of(run.summary / "clips.json")["jobs"] == 1
    blk = doc_of(run.summary / "step0.json")["render"]
    assert (blk["attempted"], blk["rendered"], blk["skipped"], blk["rounds"]) == (3, 3, 5, 1)


def test_two_lanes_keep_the_rounds_order(tmp_path, stubbed):
    """The round's first render holds until its last has started, so the other lane finishes the rest first: the
    records still come in the round's order, two at a time with 2 workers each."""
    table = {"5101:0": "a b c d e f g h", "5101:1": "h0 h1 h2 h3 h4 h5 h6 h7"}
    run = make(tmp_path, table)
    order = [r["i"] for r in run.rows if r["k"] < 4]
    release, active, peak, lock = threading.Event(), [0], [0], threading.Lock()

    def hold(row):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        if row["i"] == order[0]:
            assert release.wait(timeout=30), "the other lane never reached the round's last render"
        elif row["i"] == order[-1]:
            release.set()
        with lock:
            active[0] -= 1
    stub = stubbed(Stub(run.scored, on_call=hold))
    assert render(run.dir) == 0
    clips = doc_of(run.summary / "clips.json")
    assert [r["i"] for r in clips["clips"]] == order and sorted(c["i"] for c in stub.calls) == sorted(order)
    assert (clips["jobs"], clips["workers"]) == (2, 2) and {c["workers"] for c in stub.calls} == {2} and peak[0] == 2


# --------------------------------------------------------------------------------------------- the sets
def test_the_pool_skips_texts_the_final_held_out_set_holds(tmp_path, stubbed):
    """596:177, a pool brief and 5101:0's twin: its draws 4-6 carry texts the held-out set took (one label in one
    cell is one text), so the pool takes only draw 7; draw 3's text (d) was rendered and failed G7, so it is no
    held-out clip's text and Ref_b keeps it."""
    table = {TWIN[0]: "a b c d/G7 e f g h", TWIN[1]: "t0 t1 t2 d a b c t4"}
    run = make(tmp_path, table)
    stubbed(Stub(run.scored, outcomes_of(run, table)))
    assert render(run.dir) == 0
    sets = doc_of(run.summary / "sets.json")
    assert ks(one(sets, "heldout", TWIN[0])) == [0, 1, 2, 4]
    p = one(sets, "pool", TWIN[1])
    assert ks(p) == [7] and p["short"] is True
    assert {n["k"]: n["why"] for n in p["not_taken"]} == {0: S.REFB, 1: S.REFB, 2: S.REFB, 3: S.REFB,
                                                           4: S.HELDOUT_TEXT, 5: S.HELDOUT_TEXT, 6: S.HELDOUT_TEXT}
    flags = {(d["brief_id"], d["k"]): d["heldout_text"] for d in sets["heldout_text"]}
    assert [k for (b, k), f in flags.items() if f] == [4, 5, 6], "draw 3's text is not a held-out clip's"
    assert ks(one(sets, "refb", TWIN[1])) == [0, 1, 2, 3]
    readout = doc_of(run.summary / "step0.json")["sets"]["heldout_text"]["high/melancholy"]
    assert readout == {"draws": 3, "refb": 0, "pool": 0}


def test_render_code_is_the_steps_stamp_and_source_sha8(tmp_path, monkeypatch):
    """Unstubbed key: this checkout's .sync-stamp (none in a worktree) and lofi_rl.scoring.features.source_sha8()."""
    from lofi_rl.scoring.features import source_sha8
    table = {"5101:0": "a b c d e f g h"}
    run = make(tmp_path, table)
    real = load_palette
    monkeypatch.setattr(Rd, "load_palette", lambda path, verify=True: real(path, verify=False))
    monkeypatch.setattr(J, "render_clip", Stub(run.scored))
    assert render(run.dir) == 0
    want = {"stamp": Sc.sync_stamp(), "source_sha8": source_sha8()}
    assert len(want["source_sha8"]) == 8
    assert doc_of(run.summary / "clips.json")["render_code"] == want
    assert doc_of(run.summary / "sets.json")["render_code"] == want
    step0 = doc_of(run.summary / "step0.json")
    assert step0["render_code"] == want and step0["render_stamp"] == Sc.sync_stamp()
    assert doc_of(run.summary / "clips.json")["palette"] == {"id": "lofi_rain_v1", "sha256": S.PALETTE_SHA256}


def test_clip_of_gives_the_id_shootout_and_the_protocol_give_a_manifest():
    from lofi_rl.shootout.render import clip_id as shootout_clip_id
    pal = H.palette()
    b = Brief("high", "melancholy", 76, 0)
    cfg = render_cfg()
    manifest = {"schema": cfg["schema"], "chain": cfg["chain"], "loop": {"canon_sha256": sha("canon")},
                "brief": {"scene": b.scene, "energy": b.energy, "mood": b.mood, "mode": b.mode, "bpm": b.bpm,
                          "tonic": b.tonic, "key": b.key_name},
                "palette": {"id": pal.id, "sha256": pal.sha256}}
    clip, cfg8 = Rd.clip_of(manifest)
    assert clip == shootout_clip_id(manifest) == protocol.clip_id(sha("canon"), b, pal.id, pal.sha256, cfg)
    assert cfg8 == protocol.render_cfg_sha8(cfg) == Rd.pool_cfg_sha8()
    other = {**manifest, "chain": {**cfg["chain"], "cycles": 3}}
    assert Rd.clip_of(other)[0] == shootout_clip_id(other) != clip, "another render config is another clip"


# --------------------------------------------------------------------------------------------- refusals and --from
@pytest.mark.parametrize("tamper, needle", [
    ("stopped", "generate stopped: deadline"), ("generated", "generate drew 24 of 32 requests"),
    ("score_partial", "score marked the run partial"), ("row", "31 scored rows"),
    ("brief", "1 scored rows are not their completion's draw (first i="), ("cut", "unreadable lines")])
def test_a_partial_run_is_refused_with_nothing_written(tmp_path, stubbed, tamper, needle, capsys):
    table = {"5101:0": "a b c d e f g h", "5101:1": "h0 h1 h2 h3 h4 h5 h6 h7", "596:0": "t0 t1 t2 t3 t4 t5 t6 t7",
             "596:1": "t0 t1 t2 t3 t4 t5 t6 t7"}
    run = make(tmp_path, table)
    gen, step0 = doc_of(run.summary / "generate.json"), doc_of(run.summary / "step0.json")
    scored = rows_of(run.summary / "scored.jsonl")
    if tamper == "stopped":
        gen["stopped"] = "deadline"
    elif tamper == "generated":
        gen["generated"] = 24
    elif tamper == "score_partial":
        step0["partial"] = True
    elif tamper == "row":
        scored = scored[:-1]
    elif tamper == "brief":
        scored[5] = {**scored[5], "brief_id": "5101:1"}
    (run.summary / "generate.json").write_text(json.dumps(gen))
    (run.summary / "step0.json").write_text(json.dumps(step0))
    text = "".join(json.dumps(s) + "\n" for s in scored) + ('{"i": 9' if tamper == "cut" else "")
    (run.summary / "scored.jsonl").write_text(text)
    before = hashes(run.dir)
    stub = stubbed(Stub(run.scored))
    assert render(run.dir) == 2
    err = capsys.readouterr().err
    assert "a partial run is not rendered" in err and needle in err, err
    assert stub.calls == [] and hashes(run.dir) == before and not (run.dir / "clips").exists()


def test_missing_records_are_refused(tmp_path, stubbed, capsys):
    run = make(tmp_path, {"5101:0": "a b c d e f g h"})
    (run.summary / "scored.jsonl").unlink()
    assert render(run.dir) == 2 and "scored.jsonl is missing" in capsys.readouterr().err
    assert not (run.dir / "clips").exists() and not (run.summary / "clips.json").exists()


def test_a_render_code_key_that_cannot_be_read_is_refused_before_any_write(tmp_path, stubbed, monkeypatch, capsys):
    run = make(tmp_path, {"5101:0": "a b c d e f g h"})

    def no_numpy():
        raise ImportError("No module named 'numpy'")
    monkeypatch.setattr(Rd, "source_sha8", no_numpy)
    stub = stubbed(Stub(run.scored))
    before = hashes(run.dir)
    assert render(run.dir) == 2 and "ImportError: No module named 'numpy'" in capsys.readouterr().err
    assert stub.calls == [] and hashes(run.dir) == before and not (run.dir / "clips").exists()


def test_from_renders_outs_records_into_redo_and_never_writes_under_out(tmp_path, stubbed):
    table = {"5101:0": "a b c/G7 d e f g h", "596:177": "t0 t1 t2 t3 a b c t4"}
    out = make(tmp_path, table)
    before = hashes(out.dir)
    stubbed(Stub(out.scored, outcomes_of(out, table)))
    redo = tmp_path / "step0-redo-13800002"
    assert cli.main(["render", "--from", str(out.dir), "--run", str(redo)]) == 0
    assert hashes(out.dir) == before, "OUT is read only"
    files = sorted(p.relative_to(redo).as_posix() for p in redo.rglob("*") if p.is_file())
    assert [f for f in files if f.startswith("summary/")] == ["summary/clips.json", "summary/scored.jsonl",
                                                              "summary/sets.json", "summary/step0.json"]
    assert (redo / "summary" / "scored.jsonl").read_bytes() == (out.summary / "scored.jsonl").read_bytes()
    assert {f for f in files if f.startswith("clips/")} == {f"clips/{r['i']:04d}.{x}" for r in doc_of(
        redo / "summary" / "clips.json")["clips"] if r["ok"] for x in ("flac", "manifest.json")}
    a, b = doc_of(redo / "summary" / "step0.json"), doc_of(out.summary / "step0.json")
    assert {k: v for k, v in a.items() if k not in ("render", "render_code", "sets", "render_stamp")} == {
        k: v for k, v in b.items() if k not in ("render", "render_code", "sets", "render_stamp")}, "score's, unchanged"
    assert a["render"]["G7"] == 1 and a["run"] == NAME


def test_from_reads_a_score_recoverys_records_from_redo(tmp_path, stubbed):
    table = {"5101:0": "a b c d e f g h"}
    out = make(tmp_path, table)
    redo = tmp_path / "step0-redo-13800002"
    assert cli.main(["score", "--from", str(out.dir), "--run", str(redo)]) == 0
    scored = (redo / "summary" / "scored.jsonl").read_bytes()
    stubbed(Stub(out.scored))
    assert cli.main(["render", "--from", str(out.dir), "--run", str(redo)]) == 0
    assert doc_of(redo / "summary" / "step0.json")["run"] == "step0-redo-13800002", "the score recovery's step0.json"
    assert (redo / "summary" / "scored.jsonl").read_bytes() == scored


def test_from_and_run_that_overlap_are_refused(tmp_path, stubbed, capsys):
    run = make(tmp_path, {"5101:0": "a b c d e f g h"})
    stubbed(Stub(run.scored))
    for target in (run.dir, run.dir / "redo", run.dir.parent):
        assert cli.main(["render", "--from", str(run.dir), "--run", str(target)]) == 2
        assert "overlap" in capsys.readouterr().err
    assert not (run.dir / "redo").exists() and not (run.dir / "clips").exists()


# --------------------------------------------------------------------------------------------- end to end
def test_generate_score_and_render_on_fakes_give_the_synthetic_run(tmp_path, monkeypatch):
    """The three steps through the command line, on a fake engine answering each seed with the helper's text and a
    render stub with its outcomes, write the files tests/helpers_step0.py writes for the same table: the steps and the
    helper the other step-0 tests read agree field for field. What differs is what the helper marks as stand-ins:
    the engine's timings, the host, the shas of the code tree's DECISIONS.md and path file, the SamplingParams repr."""
    table = {TWIN[0]: "a b a/G7 c d/E e f g", "5101:1": "h0 G2 h1/G7 h2 h3 h4 h5 h6", TWIN[1]: "t0 t1 t2 t3 a b c G6",
             "596:0": "t0 G2 t1 t2 t3 t4 t5 t6"}         # 5101:0's G7 is on a repeat, never rendered; 5101:1's is
    helper = H.make_run(tmp_path / "helper" / NAME, table, briefs=list(table))
    by_seed = {r["draw_seed"]: r for r in helper.rows}

    def text_of(cell, seed):
        r = by_seed.get(seed)
        if r is None:                                   # the probe: the helper's probe texts
            t = H.variant_text(cell, "a")
            return t, "stop", H.n_tokens(t, "stop"), S.PROMPT_TOKENS[cell]
        return r["text"], r["finish_reason"], r["n_tokens"], r["prompt_tokens"]
    E.draw_only(monkeypatch, table)
    tree = E.code_tree(tmp_path / "tree")
    untouched = hashes(tree)
    monkeypatch.setattr(G, "ROOT", tree)
    monkeypatch.setattr(J, "load_engine", lambda model, **kw: E.FakeEngine(text_of, info=H.ENGINE_INFO))
    monkeypatch.setattr(J, "structured_outputs_params", lambda regex: {"regex_sha256": sha(regex)})
    monkeypatch.setattr(G, "_host", lambda root: dict(helper.doc("generate.json")["host"]))
    monkeypatch.setattr(Sc, "sync_stamp", lambda root=None: H.STAMP)
    monkeypatch.setattr(J, "_sync_stamp", lambda: H.STAMP)          # junk.summarize's score_stamp
    mine = tmp_path / "mine" / NAME
    assert cli.main(["generate", "--run", str(mine), "--hang-s", "0"]) == 0
    assert cli.main(["score", "--run", str(mine)]) == 0
    real = load_palette
    monkeypatch.setattr(Rd, "load_palette", lambda path, verify=True: real(path, verify=False))
    monkeypatch.setattr(Rd, "source_sha8", lambda: H.SOURCE_SHA8)
    monkeypatch.setattr(Rd, "sync_stamp", lambda: H.STAMP)
    scored = rows_of(mine / "summary" / "scored.jsonl")
    monkeypatch.setattr(J, "render_clip", Stub(scored, outcomes_of(helper, table)))
    assert cli.main(["render", "--run", str(mine), "--workers", "4"]) == 0
    got = {n: (rows_of if n.endswith(".jsonl") else doc_of)(mine / "summary" / n) for n in S.SUMMARY_FILES}
    want = {n: (rows_of if n.endswith(".jsonl") else doc_of)(helper.summary / n) for n in S.SUMMARY_FILES}
    for name in ("completions.jsonl", "replay.jsonl", "scored.jsonl", "sets.json", "clips.json"):
        assert got[name] == want[name], name
    stand_ins = {"timing", "registration", "path_file", "sampling_params"}
    g, w = got["generate.json"], want["generate.json"]
    assert list(g) == list(w) and {k: v for k, v in g.items() if k not in stand_ins} == {
        k: v for k, v in w.items() if k not in stand_ins}
    assert g["registration"]["found"] is True and g["path_file"]["blobs_ok"] is True
    assert list(g["timing"]) == list(w["timing"]) and g["timing"]["chunks"][0]["count"] == 32

    def steady(doc: dict) -> dict:
        blocks = {s: {**b, "generate": {**b["generate"], "timing": None}} for s, b in doc["by_stream"].items()}
        return {**doc, "by_stream": blocks}
    assert steady(got["step0.json"]) == steady(want["step0.json"])
    assert (got["step0.json"]["render"]["errors_retried"], got["step0.json"]["render"]["G7"]) == (1, 1)
    assert hashes(tree) == untouched, "the pre-checks only read the code tree"
    assert not list(mine.rglob("*.ls")), "no step writes a LoopScript file: none can reach a copy pool (spec §5)"
    assert {p.parent.name for p in mine.rglob("*") if p.is_file()} == {"summary", "clips"}
    assert [t["k"] for t in one(got["sets.json"], "heldout", "5101:1")["taken"]] == [0, 3, 4, 5]


# --------------------------------------------------------------------------------------------- real renders
def _sources_missing() -> list[str]:
    """The frozen palette's source files not on this machine (the fetch-only ones live under $LOFI_ASSETS)."""
    pal = load_palette(S.PALETTE if Path(S.PALETTE).is_absolute() else Rd.ROOT / S.PALETTE, verify=False)
    missing = []
    for src in sorted(pal.sources):
        try:
            path = pal.source_path(src)
        except Exception as err:  # noqa: BLE001: an unset assets variable, say
            return [f"{src}: {err}"]
        if not path.is_file():
            missing.append(str(path))
    return missing


needs_audio = pytest.mark.skipif(
    shutil.which("fluidsynth") is None or any(importlib.util.find_spec(m) is None for m in ("pedalboard", "soundfile")),
    reason="renders for real: needs fluidsynth on PATH, pedalboard and soundfile")


@needs_audio
def test_two_real_renders_carry_the_clip_ids_shootout_and_the_protocol_give(tmp_path):
    """Two held-out draws pass, the other six fail G2, so the step renders two clips for real with the frozen palette:
    each record's clip id is lofi_rl.shootout.render.clip_id of its manifest and lofi_rl.scoring.protocol.clip_id of
    its canonical sha, brief, palette and the render pool's config. Skipped where the palette's fetch-only sources
    are not here (LOFI_ASSETS)."""
    missing = _sources_missing()
    if missing:
        pytest.skip(f"the frozen palette's fetch-only sources are not here ({len(missing)}, e.g. {missing[0]})")
    from lofi_rl.shootout.render import clip_id as shootout_clip_id
    sf = pytest.importorskip("soundfile")
    run = make(tmp_path, {"5101:0": "h0 h1 G2 G2 G2 G2 G2 G2"})
    assert render(run.dir, "--workers", "4", "--cache", str(tmp_path / "cache")) == 0
    clips = doc_of(run.summary / "clips.json")["clips"]
    assert [(r["k"], r["ok"]) for r in clips] == [(0, True), (1, True)], clips
    scored = {s["i"]: s for s in run.scored}
    pal = H.palette()
    for r in clips:
        manifest = doc_of(run.dir / r["clip"].replace(".flac", ".manifest.json"))
        b = Brief(**next(c for c in run.rows if c["i"] == r["i"])["brief"])
        assert r["clip_id"] == shootout_clip_id(manifest) == protocol.clip_id(
            scored[r["i"]]["canon_sha256"], b, pal.id, pal.sha256, render_cfg())
        assert r["render_cfg_sha8"] == protocol.render_cfg_sha8(render_cfg()) and r["final"] is True
        assert r["audio_sha256"] == manifest["audio_sha256"] and r["canon_sha256"] == scored[r["i"]]["canon_sha256"]
        x, sr = sf.read(str(run.dir / r["clip"]), dtype="int16")
        assert sr == 44100 and x.shape == (r["n"], 2), "one pass of the loop"
    sets = doc_of(run.summary / "sets.json")
    h = one(sets, "heldout", "5101:0")
    assert ks(h) == [0, 1] and h["short"] is True
    assert [t["clip_id"] for t in h["taken"]] == [r["clip_id"] for r in clips]
