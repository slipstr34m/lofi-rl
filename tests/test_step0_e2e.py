"""The step-0 unit end to end on the laptop (docs/design/step0_samples.md §2.3-2.7, §4.1-4.4, §6.2 steps 4-9): the
files the steps themselves write, not tests/helpers_step0.py's synthetic ones, are what check and prereg accept.

tests/test_step0_check.py and tests/test_step0_prereg.py read synthetic runs whose stand-ins (the code tree's
DECISIONS.md and path-file sha256s, the stamps, the render-code key) are what check's "Registered first" and "One stamp,
one path" conditions read. Here generate, score and render write them, as jobs/step0.sbatch runs them:

- **The laptop's repo:** a temporary git repo holding a copy of the code tree (src/, jobs/step0.sbatch and its env
  scripts, kit.env and the data files the path file lists, the SoundFont as a small stand-in, whose blob the path
  file then lists: the palette is read here without its sources), docs/DECISIONS.md with a D-023 stand-in holding
  lofi_rl.step0.MARK on a line of its own, in a fenced block under its "The mark" bullet as the D-023 draft places
  it, and the path file `python -m lofi_rl.step0 path --write` writes for that tree, all committed: the code commit.
- **A snapshot** (scripts/snapshot.py's role): that commit's tree (git archive) and the .sync-stamp make sync writes
  (scripts/stamp.sh's line, commit=<sha12> dirty=0 diff=- at=...). Each step's code tree, stamp, palette and streams
  file are the snapshot's: the ROOT each step module reads points at it, as PYTHONPATH=<snapshot>/src makes it in the
  job. The code that runs is this checkout's, which the snapshot's src/ copies byte for byte.
- **A job:** SLURM_JOB_ID set; generate, score and render through the command line with the job's arguments, each run
  whatever the one before it returned, into the node's RUN folder; then the copy-out to KEEP (summary/, then clips/)
  and make pull (summary/ only) into the laptop's runs/, which check and prereg read with the laptop repo's git.

The stand-ins, and nothing else: the engine (tests/helpers_step0_engine.FakeEngine, each request answered by its seed),
vLLM's StructuredOutputsParams, the watchdog (--hang-s 0: its alarm would end the test process), the render's audio
(junk.render_clip without FluidSynth and the bus chains: the real parse, canonical sha, compile and P_fix in a real
render manifest, with a stand-in FLAC, audio hash and loudness), the hashing of the palette's fetch-only sources at
render, and a memo on junk.score_row keyed on its whole input, so a later job over the same completions reuses the
first job's scores (each distinct input is scored for real once). The texts are tests/helpers_step0.py's family (each
cell's shown example with its first kick grid set from a label) with a short sax phrase and answer, which
junk.score_row scores in about 5 ms a row instead of 23 on the laptop (the sax expression's compile is most of it).

A run whose generate stopped before its first draw reads invalid with the stop named (finding E1, in four cases).
Sound claims: none.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import pytest

import helpers_step0 as H
import helpers_step0_engine as E
import helpers_step0_laptop as L
import lofi_rl.step0 as S
from lofi_rl.brief import Brief
from lofi_rl.palette import load_palette
from lofi_rl.rl import junk as J
from lofi_rl.step0 import check as C
from lofi_rl.step0 import cli
from lofi_rl.step0 import generate as G
from lofi_rl.step0 import path as P
from lofi_rl.step0 import render as Rd
from lofi_rl.step0 import requests as R
from lofi_rl.step0 import score as Sc

pytestmark = pytest.mark.laptop   # git reads the stamps' commits: never on a compute node (CLAUDE.md §1)

ROOT = P.ROOT
SF2 = "assets/soundfonts/generaluser-gs/GeneralUser-GS.sf2"
SAX = {"A": "e5 er qr hr | hr hr | hr hr | hr hr", "B": "e3 er qr hr | hr hr | hr hr | hr hr"}
JOB, REPLAY_JOB, DRIFT_JOB, CUT_JOB = "13800001", "13800003", "13800005", "13800007"
# The main run's draws, as tests/helpers_step0.py's table writes them; every other brief has its default draws
TABLE = {
    "5101:0": "a b a/G7 c d/E e f g",       # 596:177's twin: k2 repeats k0 (never rendered); k4 errors once
    "5101:1": "h0 G2 h1/G7 h2 h3 h4 h5 h6",  # k2 fails G7 at its render, so round 2 renders k5
    "5101:2": "a a a b b c G2 G6",           # three distinct passing texts: a short tuple, reported (Q8(a))
    "596:177": "t0 t1 t2 t3 a b c G6",       # a pool brief: its draws 4-6 carry texts the held-out set took (Q5(a))
    "596:0": "t0 G2 t1 t2 t3 t4 t5 t6",      # Ref_b k0, k2-4; the pool k5-7
    "596:1": "t0 t0 t1 t2 t3 t4 t5 t6",      # Ref_b keeps its repeat (Q4(a)); the pool k4-6
}
SPECS = {bid: H.parse_specs(bid, spec, bid.startswith("5101:")) for bid, spec in TABLE.items()}
OUTCOMES = {(bid, k): render for bid, specs in SPECS.items() for k, (_, render) in enumerate(specs) if render}
DRAWS = {r.draw_seed: r for r in H.all_requests()}            # each of the 3,520 requests by its (distinct) seed
D023 = """
## D-023 (a stand-in for tests/test_step0_e2e.py): the step-0 samples, drawn by `lofi-step0` under rule `step0-select/1`

- **The mark** (the spec's §2.3). Before any work, generate checks that the job's code tree holds this line in its
  docs/DECISIONS.md, and that it equals `lofi_rl.step0.MARK`, which `requests.mark_line(STREAMS, RULE)` builds.

  ```
  {mark}
  ```

- **The fix comes first:** any code fix, and every file on the path it changes, is recorded in an amendment of this
  entry before the redo, with one line per redo, verbatim:

    ```
    D-023 redo amendment: run step0-<job>; changed <path>, <path>
    ```
"""
FIX = "src/lofi_rl/step0/render.py"     # a redo's fix: a file the path file lists, outside features.SOURCE_GLOBS
DRIFT = "src/lofi_rl/step0/select.py"   # a file that changes after the path file is written
REAL_SCORE_ROW = J.score_row
_SCORED: dict[str, str] = {}


# --------------------------------------------------------------------------------------------- the texts
@lru_cache(maxsize=None)
def text(cell: str, label: str) -> str:
    """The helper's text of a label (H.variant_text), with SAX's phrase and answer: it passes, in the grammar's
    language, one canonical text per label and cell."""
    lines = [f"{x[0]}: {SAX[x[0]]}" if x[:3] in ("A: ", "B: ") else x for x in H.variant_text(cell, label).splitlines()]
    return "\n".join(lines) + "\n"


@lru_cache(maxsize=None)
def gate_text(cell: str, gate: str) -> tuple[str, str]:
    """(text, finish_reason) failing `gate`, as H.gate_text makes it: G2 cut at 60 % at the cap, G6 its vamp's first
    chord twice (inside the language)."""
    base = text(cell, "h0")
    if gate == "G2":
        return base[:int(len(base) * 0.6)], "length"
    lines = base.splitlines()
    j = next(j for j, line in enumerate(lines) if line.startswith("vamp: "))
    words = lines[j].split()
    lines[j] = " ".join([words[0], words[1], words[1], *words[3:]])
    return "\n".join(lines) + "\n", "stop"


def draw_text(r) -> tuple[str, str]:
    """A request's (text, finish_reason): TABLE's label for its draw, or the helper's default (h<k> held-out, t<k>
    train, u<k> or v<k> for a train brief's twins: tests/helpers_step0.default_labels)."""
    label = SPECS[r.brief_id][r.k][0] if r.brief_id in SPECS else H.default_labels(r.stream, r.brief_id)[r.k]
    return gate_text(r.cell, label) if label in H.GATES else (text(r.cell, label), "stop")


def answers(differ_on_replay=frozenset()):
    """The fake engine's text_of(cell, key): a draw's text by its seed; each probe seed (or, with seeds ignored, a
    running count) another passing text; a seed in `differ_on_replay` answers its second request with another text."""
    seen: Counter = Counter()

    def text_of(cell, key):
        r = DRAWS.get(key)
        if r is None:
            t = text(cell, H.PROBE_LABELS[S.PROBE_SEEDS.index(key)] if key in S.PROBE_SEEDS else
                     H.LABELS[key % len(H.LABELS)])
            finish = "stop"
        else:
            assert cell == r.cell, "the engine got another cell's messages for this seed"
            seen[key] += 1
            t, finish = (text(cell, "z"), "stop") if seen[key] > 1 and key in differ_on_replay else draw_text(r)
        return t, finish, H.n_tokens(t, finish), S.PROMPT_TOKENS[cell]
    return text_of


def score_row(row, palette, check):
    """junk.score_row, memoized on everything it reads (the row's brief, text, finish_reason and n_tokens, the palette
    and the copy check's prompt version). A row whose knob readout failed is never kept, so CopyCheck counts it each
    time, as the real call does."""
    key = json.dumps([palette.sha256, check.version, row.get("brief"), row.get("text"), row.get("finish_reason"),
                      row.get("n_tokens")], sort_keys=True)
    if key in _SCORED:
        out = json.loads(_SCORED[key])
        out["i"] = row["i"]
        return out
    errors = sum(check.knob_errors.values())
    out = REAL_SCORE_ROW(row, palette, check)
    if sum(check.knob_errors.values()) == errors:
        _SCORED[key] = json.dumps(out)
    return out


# --------------------------------------------------------------------------------------------- the render's stand-in
class Render:
    """junk.render_clip without its audio: the real parse (a gate failure is the record's gate), canonical sha,
    compile (CompileError is "compile") and P_fix, in a render manifest of the pipeline's schema and chain, with a
    stand-in FLAC, audio_sha256 and loudness; OUTCOMES's G7 and errors by (brief id, k). `clock` moves on `step_s` a
    render (the render step's deadline reads it)."""

    def __init__(self, outcomes=None, clock=None, step_s: float = 0.0):
        self.outcomes, self.clock, self.step_s = dict(OUTCOMES if outcomes is None else outcomes), clock, step_s
        self.tried, self.calls, self.lock = Counter(), [], threading.Lock()

    def __call__(self, row, palette, clips_dir, *, cache, workers):
        from lofi_rl.compile.compiler import CompileError, compile_loop
        from lofi_rl.dsl import canon, canon_sha, parse
        from lofi_rl.dsl.repair import penalty
        from lofi_rl.render import pipeline
        key, i, t0 = (row["brief_id"], row["k"]), row["i"], time.perf_counter()
        with self.lock:
            self.tried[key] += 1
            n_try = self.tried[key]
            self.calls.append({"i": i, "workers": workers, "cache": cache})
            if self.clock is not None:
                self.clock[0] += self.step_s
        spec = self.outcomes.get(key, "")
        if spec == "G7":
            return {"i": i, "ok": False, "gate": "G7", "detail": "stand-in: the render is out of spec"}
        if (spec == "E" and n_try == 1) or spec == "EE":
            return {"i": i, "ok": False, "gate": None, "error": "OSError: stand-in: fluidsynth died"}
        b = Brief(**row["brief"])
        res = parse(row["text"], vocab=palette.vocab(), finish_reason=row.get("finish_reason"), mode=b.mode)
        if not res.gate.ok:
            return {"i": i, "ok": False, "gate": res.gate.code or "G1", "detail": res.gate.detail[:300]}
        try:
            compiled = compile_loop(res.ast, b, palette)
        except CompileError as err:
            return {"i": i, "ok": False, "gate": "compile", "detail": str(err)[:300]}
        repairs = list(res.repairs) + list(compiled.repairs)
        loop = {"canon": canon(res.ast), "canon_sha256": canon_sha(res.ast),
                "repairs": [asdict(r) for r in repairs], "p_fix": penalty(repairs)}
        brief = {"scene": b.scene, "energy": b.energy, "mood": b.mood, "mode": b.mode, "bpm": b.bpm, "tonic": b.tonic,
                 "key": b.key_name}
        audio = hashlib.sha256(f"stand-in pcm|{loop['canon_sha256']}|{json.dumps(brief)}".encode()).hexdigest()
        m = {"schema": pipeline.SCHEMA, "loop": loop, "brief": brief,
             "palette": {"id": palette.id, "sha256": palette.sha256, "frozen": palette.frozen},
             "n": compiled.n, "sr": pipeline.SR, "seconds": compiled.n / pipeline.SR,
             "chain": {"cycles": pipeline.CYCLES, "fluid_preroll_cycles": pipeline.PREROLL,
                       "midi_cycles": pipeline.PREROLL + pipeline.CYCLES, "pedalboard_buffer": pipeline.BUFFER,
                       "limiter_tile": pipeline.LIMITER_TILE},
             "loudness": {"lufs": -14.0, "true_peak_dbtp": -1.5}, "audio_sha256": audio}
        name = f"{i:04d}"
        flac = clips_dir / f"{name}.flac"
        flac.write_bytes(b"fLaC: a stand-in, no audio is rendered here\n")
        (clips_dir / f"{name}.manifest.json").write_text(pipeline.manifest_json(m), encoding="utf-8")
        return {"i": i, "ok": True, "gate": None, "clip": f"clips/{name}.flac", "n": m["n"],
                "seconds": round(m["seconds"], 3), "lufs": round(m["loudness"]["lufs"], 3),
                "true_peak_dbtp": round(m["loudness"]["true_peak_dbtp"], 3), "audio_sha256": m["audio_sha256"],
                "canon_sha256": loop["canon_sha256"], "p_fix": loop["p_fix"],
                "render_s": round(time.perf_counter() - t0, 2), "flac_bytes": flac.stat().st_size}


# --------------------------------------------------------------------------------------------- the world
def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha_bytes(p.read_bytes())
            for p in sorted(Path(root).rglob("*")) if p.is_file() and ".git" not in p.relative_to(root).parts}


def decisions(mark: str | None = S.MARK, extra: str = "") -> str:
    """docs/DECISIONS.md as committed (this checkout's) with the D-023 stand-in: `mark` in its fenced block (None:
    the block left out), then `extra` (an amendment)."""
    d023 = D023.format(mark=mark)
    if mark is None:
        d023 = d023.replace("\n  ```\n  None\n  ```\n", "\n")
    return (ROOT / S.DECISIONS).read_text(encoding="utf-8").rstrip("\n") + "\n" + d023 + extra


def amendment(run: str, files: list[str]) -> str:
    return ("\n- **Amended (a redo of the held-out render of " + run + "):** the fix and the files on the path it "
            f"changes.\n\n  ```\n  D-023 redo amendment: run {run}; changed {', '.join(files)}\n  ```\n")


def commit(repo: Path, message: str, edits: dict[str, str]) -> str:
    """A later commit of the repo: each path's new text, UTF-8. Its full sha."""
    for rel, body in edits.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(body, encoding="utf-8")
    L.git(repo, "add", "-A")
    L.git(repo, "commit", "-qm", message)
    return L.git(repo, "rev-parse", "HEAD").strip()


def code_repo(dest: Path) -> tuple[Path, str]:
    """The laptop's repo (module doc), committed once: (its folder, the code commit's full sha)."""
    root = Path(dest)
    shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for rel in [*P.data_files(ROOT), *P.RUN_FILES]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        if rel == SF2:
            (root / rel).write_bytes(b"a stand-in for the SoundFont (tests/test_step0_e2e.py)\n")
        else:
            shutil.copy2(ROOT / rel, root / rel)
    (root / S.DECISIONS).write_text(decisions(), encoding="utf-8")
    rc, out, err = call(["path", "--write", "--root", str(root)])          # the path file, last, for this tree
    assert rc == 0 and out.startswith("wrote "), err
    L.git(root, "init", "-q", "-b", "main")
    L.git(root, "config", "commit.gpgsign", "false")
    L.git(root, "add", "-A")
    L.git(root, "commit", "-qm", "the code commit: D-023's mark and the path file")
    return root, L.git(root, "rev-parse", "HEAD").strip()


def snapshot(repo: Path, commit_sha: str, dest: Path, stamp: str | None = None) -> Path:
    """A job's code tree: the commit's tree (git archive) and its .sync-stamp (default: the commit's, dirty=0)."""
    data = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", commit_sha], capture_output=True,
                          timeout=120, check=True).stdout
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")
    (dest / ".sync-stamp").write_text((L.stamp(commit_sha) if stamp is None else stamp) + "\n")
    (dest / ".lofi-snapshot").write_text(f"{commit_sha[:12]} (a stand-in for scripts/snapshot.py's marker)\n")
    return dest


def call(argv: list[str]) -> tuple[int, str, str]:
    """python -m lofi_rl.step0 <argv> in this process: (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


@dataclass
class Job:
    id: str
    run: Path                       # the node's RUN folder
    out: Path                       # KEEP's OUT, the copy-out
    pulled: Path                    # the laptop's runs/<run>, make pull's copy of OUT/summary
    steps: dict                     # step -> (exit code, stdout, stderr)
    loads: list                     # the engine loads (junk.load_engine's calls)
    engine: object
    render: Render
    code_untouched: bool            # the snapshot is as it was before the job
    redo_of: Path | None = None
    extra: dict = field(default_factory=dict)

    def rc(self) -> dict[str, int]:
        return {k: v[0] for k, v in self.steps.items()}

    def doc(self, name: str, where: str = "pulled"):
        p = getattr(self, where) / "summary" / name
        return [json.loads(x) for x in p.read_text().splitlines()] if name.endswith(".jsonl") else json.loads(
            p.read_text())


def _on_cluster(mp: pytest.MonkeyPatch, code: Path, job_id: str, *, engine=None, loads=None, render=None,
                clock=None, gen_clock=None) -> None:
    """A step in the job's snapshot `code` (module doc), with the stand-ins; `clock` and `gen_clock` are the render's
    and generate's clocks when a test moves them."""
    from lofi_rl.scoring import features
    mp.setenv("SLURM_JOB_ID", job_id)
    for mod in (G, Sc, Rd, R, J):
        mp.setattr(mod, "ROOT", code)
    mp.setattr(Rd, "source_sha8", lambda: features.source_sha8(code))
    mp.setattr(J, "score_row", score_row)
    mp.setattr(J, "load_engine", lambda model, **kw: loads.append((model, kw)) or engine)
    mp.setattr(J, "structured_outputs_params", lambda regex: {"regex_sha256": sha_bytes(regex.encode())})
    real = load_palette
    mp.setattr(Rd, "load_palette", lambda path, verify=True: real(path, verify=False))
    mp.setattr(J, "render_clip", render)
    if clock is not None:
        mp.setattr(Rd, "_now", lambda: clock[0])
    if gen_clock is not None:
        mp.setattr(G, "_now", lambda: gen_clock[0])


def copy_out(run: Path, out: Path) -> None:
    shutil.copytree(run / "summary", out / "summary")                          # cp -a "$RUN/summary" "$OUT/"
    if (run / "clips").is_dir():
        shutil.copytree(run / "clips", out / "clips")                          # cp -a "$RUN/clips" "$OUT/"


def pull(out: Path, laptop: Path) -> Path:
    shutil.copytree(out / "summary", laptop / "summary")                       # make pull: runs/<run>/summary/**
    return laptop


def job(base: Path, job_id: str, code: Path, *, engine=None, render: Render | None = None, clock=None,
        gen_clock=None) -> Job:
    """jobs/step0.sbatch for job `job_id` from the snapshot `code` (module doc)."""
    engine = E.FakeEngine(answers(), info=H.ENGINE_INFO) if engine is None else engine
    render = Render(clock=clock) if render is None else render
    run, out = base / "node" / f"step0-{job_id}", base / "keep" / "runs" / f"step0-{job_id}"
    loads: list = []
    before = tree_hashes(code)
    with pytest.MonkeyPatch.context() as mp:
        _on_cluster(mp, code, job_id, engine=engine, loads=loads, render=render, clock=clock, gen_clock=gen_clock)
        (run / "summary").mkdir(parents=True)                                  # mkdir -p "$RUN/summary" "$OUT"
        out.mkdir(parents=True)
        steps = {"generate": call(["generate", "--run", str(run), "--deadline-s", "960", "--hang-s", "0"]),
                 "score": call(["score", "--run", str(run)]),
                 "render": call(["render", "--run", str(run), "--workers", "4", "--deadline-s", "900"])}
    copy_out(run, out)
    return Job(job_id, run, out, pull(out, base / "laptop" / "runs" / f"step0-{job_id}"), steps, loads, engine,
               render, tree_hashes(code) == before)


def redo_job(base: Path, job_id: str, code: Path, of: Job, steps_: tuple[str, ...] = ("render",)) -> Job:
    """A recovery (§4.2) from the snapshot `code`: each of `steps_` with --from OUT --run REDO, on KEEP, then make
    pull."""
    redo = base / "keep" / "runs" / f"step0-redo-{job_id}"
    render, before, out_before = Render(), tree_hashes(code), tree_hashes(of.out)
    args = {"score": [], "render": ["--workers", "4", "--deadline-s", "900"]}
    with pytest.MonkeyPatch.context() as mp:
        _on_cluster(mp, code, job_id, render=render)
        steps = {s: call([s, "--from", str(of.out), "--run", str(redo), *args[s]]) for s in steps_}
    return Job(job_id, redo, redo, pull(redo, base / "laptop" / "runs" / f"step0-redo-{job_id}"), steps, [], None,
               render, tree_hashes(code) == before, of.out, {"out_untouched": tree_hashes(of.out) == out_before})


def laptop(repo: Path, argv: list[str]) -> tuple[int, str, str]:
    """python -m lofi_rl.step0 <argv> on the laptop, whose checkout (and git) is `repo`."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "ROOT", repo)
        mp.setattr(R, "ROOT", repo)
        return call(argv)


def check(repo: Path, run: Path, redo: Path | None = None) -> tuple[int, dict]:
    """check --json: (exit code, the report)."""
    rc, out, err = laptop(repo, ["check", str(run), *(["--redo", str(redo)] if redo else []), "--json"])
    assert out.strip(), err
    return rc, json.loads(out)


def cond(doc: dict, name: str) -> dict:
    (c,) = [c for c in doc["validity"]["conditions"] if c["name"] == name]
    return c


def copy_repo(src: Path, dest: Path) -> Path:
    shutil.copytree(src, dest)
    return dest


# --------------------------------------------------------------------------------------------- fixtures
@dataclass
class World:
    base: Path
    repo: Path              # the laptop's repo
    c1: str                 # the code commit
    snap: Path              # its snapshot
    main: Job               # the main job, 13800001


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> World:
    base = tmp_path_factory.mktemp("step0-e2e")
    repo, c1 = code_repo(base / "laptop" / "repo")
    snap = snapshot(repo, c1, base / "keep" / "snap" / c1[:12])
    return World(base, repo, c1, snap, job(base, JOB, snap))


@dataclass
class Later:
    repo: Path                                      # the laptop's repo with the later commits
    commits: dict[str, str]                         # what each later commit does -> its sha
    snaps: dict[str, Path]


@pytest.fixture(scope="module")
def later(world) -> Later:
    """Commits after the code commit, in a copy of the repo: D-023 without its mark, D-023 with another mark, and a
    listed file that changes after the path file was written."""
    repo = copy_repo(world.repo, world.base / "laptop-later" / "repo")
    other = S.MARK.replace("step0-select/1", "step0-select/2")
    commits = {"no mark": commit(repo, "D-023 without its mark", {S.DECISIONS: decisions(None)}),
               "another mark": commit(repo, "D-023 with another mark", {S.DECISIONS: decisions(other)})}
    commits["a listed file changed"] = commit(repo, f"{DRIFT} changes after the path file", {
        S.DECISIONS: decisions(), DRIFT: (repo / DRIFT).read_text() + "# an edit after the path file\n"})
    snaps = {k: snapshot(repo, c, world.base / "keep" / "snap" / c[:12]) for k, c in commits.items()}
    return Later(repo, commits, snaps)


STOPPED = ("no mark", "another mark", "a listed file changed", "the probe missed")


@pytest.fixture(scope="module")
def stopped(world, later) -> dict[str, tuple[Job, Path]]:
    """The jobs whose generate stops before its first draw, each with the laptop repo that holds its commit."""
    out = {}
    for n, why in enumerate(STOPPED):
        if why == "the probe missed":
            engine = E.FakeEngine(answers(), honour_seeds=False, info=H.ENGINE_INFO)
            out[why] = job(world.base / f"stopped{n}", f"1380002{n}", world.snap, engine=engine), world.repo
        else:
            out[why] = job(world.base / f"stopped{n}", f"1380002{n}", later.snaps[why]), later.repo
    return out


@pytest.fixture(scope="module")
def cut(world) -> Job:
    """A run whose held-out render the deadline cut: each render moves the render step's clock on 15 s, so about 60
    render before the job's 900 s and the rest are skipped."""
    clock = [0.0]
    return job(world.base / "cut", CUT_JOB, world.snap, render=Render(clock=clock, step_s=15.0), clock=clock)


# --------------------------------------------------------------------------------------------- the main case
def test_the_code_commit_holds_the_mark_and_its_path_file_checks_out_with_and_without_git(world):
    marks = G.find_marks((world.repo / S.DECISIONS).read_text(encoding="utf-8"))
    assert marks == [S.MARK], "one mark, the draft's line, and the amendment template is no amendment"
    assert C.amendment_files((world.repo / S.DECISIONS).read_text(encoding="utf-8"), f"step0-{JOB}") == []
    rc, out, _ = call(["path", "--commit", world.c1, "--root", str(world.repo)])
    assert rc == 0 and out.startswith("STEP0_PATH ok"), out
    rc, out, _ = call(["path", "--check", "--root", str(world.snap)])
    assert rc == 0 and out.startswith("STEP0_PATH ok"), "the snapshot, without git, as generate checks it"
    listed = {p for _, p in P.parse((world.repo / S.PATH_FILE).read_text())[0]}
    assert S.JOB in listed and S.DECISIONS not in listed and S.PATH_FILE not in listed
    assert (world.repo / S.JOB).read_bytes() == (ROOT / S.JOB).read_bytes(), "the real job script"


def test_the_job_runs_every_step_in_its_snapshot_and_writes_its_own_stamps(world):
    j, c1 = world.main, world.c1
    assert j.rc() == {"generate": 0, "score": 0, "render": 0}, {k: v[2][-2000:] for k, v in j.steps.items()}
    assert j.code_untouched, "the steps only read the snapshot"
    assert len(DRAWS) == 3520 and len(j.engine.calls) == 1 + 28 + 2, "the probe, 28 chunks of draws, the replay's 2"
    assert j.loads == [(f"{S.MODEL}@{S.MODEL_REVISION}", {"seed": 0, "gpu_mem": 0.85, "max_model_len": 4096,
                                                          "backend": "xgrammar"})]
    gen, step0 = j.doc("generate.json"), j.doc("step0.json")
    st = L.stamp(c1)
    assert gen["host"]["stamp"] == step0["score_stamp"] == step0["render_stamp"] == st and gen["host"]["job"] == JOB
    assert gen["registration"] == {"found": True, "mark": S.MARK,
                                   "decisions_sha256": sha_bytes(L.blob(world.repo, c1, S.DECISIONS))}
    assert gen["path_file"] == {"sha256": sha_bytes(L.blob(world.repo, c1, S.PATH_FILE)), "blobs_ok": True}
    assert (gen["requested"], gen["generated"], gen["stopped"], gen["partial"]) == (3520, 3520, None, False)
    assert gen["probe"]["passed"] is True and gen["replay"]["same"] == gen["replay"]["n"] == 320
    from lofi_rl.scoring.features import source_sha8
    code = {"stamp": st, "source_sha8": source_sha8(world.snap)}
    assert step0["render_code"] == code == j.doc("sets.json")["render_code"] == j.doc("clips.json")["render_code"]
    assert {b["score_stamp"] for b in step0["by_stream"].values()} == {st}, "junk.summarize read the snapshot's stamp"
    assert all(b["copy"]["knobs_error"] is None for b in step0["by_stream"].values()), "the memo kept every row"
    assert step0["render"] == {"attempted": 161, "rendered": 159, "G7": 1, "other_gates": 0, "errors_final": 0,
                               "errors_retried": 1, "skipped": 0, "rounds": 2}
    files = sorted(p.relative_to(j.run).as_posix() for p in j.run.rglob("*") if p.is_file())
    assert {f.split("/")[0] for f in files} == {"summary", "clips"} and not [f for f in files if f.endswith(".ls")]
    assert sorted(p.name for p in (j.pulled / "summary").iterdir()) == sorted(S.SUMMARY_FILES)
    assert not (j.pulled / "clips").exists(), "make pull brings summary/ only"
    for name in S.SUMMARY_FILES:                    # each file holds the registered fields, in their order
        want, optional = S.FIELDS[name], set(S.OPTIONAL.get(name, ()))
        for keys in ([list(r) for r in j.doc(name)] if name.endswith(".jsonl") else [list(j.doc(name))]):
            assert keys == [f for f in want if f in keys] and set(want) - set(keys) <= optional, (name, keys)


def test_check_reads_the_steps_own_files_valid_condition_by_condition(world):
    j, c1 = world.main, world.c1
    rc, out, err = laptop(world.repo, ["check", str(j.pulled)])
    assert rc == 0, out[-3000:] + err[-2000:]
    assert out.rstrip().splitlines()[-1] == "STEP0_CHECK valid"
    assert "validity: VALID, 74 of 74 conditions hold" in out
    rc, doc = check(world.repo, j.pulled)
    v = doc["validity"]
    assert rc == 0 and v["failed"] == [] and v["held"] == v["of"] == 74
    assert all(c["ok"] for c in v["conditions"])
    n_listed = len(P.parse((world.repo / S.PATH_FILE).read_text())[0])
    assert cond(doc, "the path at the generate commit")["shows"] == (
        f"at {c1[:12]}: {n_listed} of {n_listed} listed files have their blob; the path file's sha256 "
        f"{S.short(sha_bytes(L.blob(world.repo, c1, S.PATH_FILE)))} is generate's")
    assert cond(doc, "docs/DECISIONS.md at the generate commit")["shows"] == (
        f"at {c1[:12]}: holds MARK; sha256 {S.short(sha_bytes(L.blob(world.repo, c1, S.DECISIONS)))} is generate's")
    assert cond(doc, C.STAMP_NAME)["shows"] == f"{L.stamp(c1)} (all three)"
    sha8 = j.doc("step0.json")["render_code"]["source_sha8"]
    assert cond(doc, "render_code.source_sha8")["shows"] == f"at {c1[:12]}: {sha8}; render_code.source_sha8 {sha8}"
    assert cond(doc, "clip ids")["shows"].startswith("159 clip ids equal the formula (lofi_rain_v1@09f80a64")
    assert cond(doc, "sets.pool taken")["shows"] == "657 picks of 220 briefs"
    assert doc["requests"]["rows"] == doc["requests"]["requested"] == 3520
    sets = j.doc("sets.json")
    by = {name: {e["brief_id"]: e for e in sets[name]} for name in S.SETS}
    assert [t["k"] for t in by["heldout"]["5101:0"]["taken"]] == [0, 1, 3, 4]
    assert [t["k"] for t in by["heldout"]["5101:1"]["taken"]] == [0, 3, 4, 5]
    assert by["heldout"]["5101:2"]["short"] is True and len(by["heldout"]["5101:2"]["taken"]) == 3
    assert [t["k"] for t in by["refb"]["596:1"]["taken"]] == [0, 1, 2, 3], "Ref_b keeps its repeat"
    assert by["pool"]["596:177"]["taken"] == [] and {n["why"] for n in by["pool"]["596:177"]["not_taken"]} == {
        S.REFB, S.HELDOUT_TEXT, "G6"}
    assert [t["k"] for t in by["pool"]["596:0"]["taken"]] == [5, 6, 7]


def test_prereg_writes_both_files_into_the_repo_and_again_byte_identical_with_hashes_only(world):
    j = world.main
    repo = copy_repo(world.repo, world.base / "laptop-prereg" / "repo")
    files = (S.HELDOUT_FILE, S.RUN_FILE)
    assert not any((repo / rel).exists() for rel in files)
    rc, out, err = laptop(repo, ["prereg", str(j.pulled), "--write"])
    assert rc == 0 and f"wrote {S.HELDOUT_FILE}, {S.RUN_FILE}" in out, err
    first = {rel: (repo / rel).read_bytes() for rel in files}
    rc, out, err = laptop(repo, ["prereg", str(j.pulled), "--write"])
    assert rc == 0 and "unchanged: both files already hold this text" in out, err
    assert {rel: (repo / rel).read_bytes() for rel in files} == first
    held, run = json.loads(first[S.HELDOUT_FILE]), json.loads(first[S.RUN_FILE])
    assert held["counts"] == {"briefs": 40, "tuples": 39, "clips": 156, "short": 1, "twins": 8}
    assert [s["brief_id"] for s in held["short"]] == ["5101:2"]
    assert held["run"]["stamp"] == {"generate": L.stamp(world.c1), "score": L.stamp(world.c1),
                                    "render": L.stamp(world.c1)} and held["run"]["validity"]["verdict"] == "valid"
    comp = (j.pulled / "summary" / "completions.jsonl").read_bytes()
    assert held["texts"] == {"file": f"runs/step0-{JOB}/summary/completions.jsonl", "sha256": sha_bytes(comp),
                             "by": "text_sha256: sha256 of a row's `text`, UTF-8"} == run["texts"]
    assert run["heldout"]["sha256"] == sha_bytes(first[S.HELDOUT_FILE])
    assert {f["path"]: f["sha256"] for f in run["files"]} == {
        f"runs/step0-{JOB}/summary/{n}": sha_bytes((j.pulled / "summary" / n).read_bytes()) for n in S.SUMMARY_FILES}
    assert (run["pool"]["briefs"], run["pool"]["picks"], len(run["refb"])) == (220, 657, 400)
    rows = j.doc("completions.jsonl")
    lines = {x.strip() for r in rows for x in r["text"].splitlines() if x.strip()}
    assert len(lines) > 100
    for rel, body in first.items():
        body = body.decode("ascii")
        assert not [x for x in lines if x in body], f"{rel} holds a LoopScript line"
        assert not [r["i"] for r in rows if json.dumps(r["text"])[1:-1] in body], f"{rel} holds a LoopScript text"
        for head in ("vamp: ", "keys: ", "kick: ", "A: ", "B: ", "drums: "):
            assert f'"{head}' not in body and f"\\n{head}" not in body, (rel, head)
    shas = {sha_bytes(r["text"].encode()) for r in rows}
    clips = [c for t in held["tuples"] for c in t["clips"]]
    assert len(clips) == 156 and {c["text_sha256"] for c in clips} <= shas, "each clip's text is found by its sha"
    new = {p for p in tree_hashes(repo)} - {p for p in tree_hashes(world.repo)}
    assert new == set(files), "the two files and nothing else"


# --------------------------------------------------------------------------------------------- 1, 2, 4: stopped early
def test_a_run_stopped_before_its_first_draw_draws_nothing_and_decides_nothing(stopped, later, world):
    """No mark, another mark, a listed file changed after the path file (each in its commit's snapshot), and a probe
    that misses: generate stops with exit 2 and nothing drawn (before the engine loads, but for the probe), score and
    render (the job runs them all) refuse with exit 2 and write nothing, the copy-out carries generate.json alone, and
    on the laptop check reads nothing valid and prereg writes nothing."""
    other = S.MARK.replace("step0-select/1", "step0-select/2")
    for why, (j, repo) in stopped.items():
        assert j.rc() == {"generate": 2, "score": 2, "render": 2}, (why, {k: v[2][-800:] for k, v in j.steps.items()})
        gen = j.doc("generate.json", "run")
        assert (gen["generated"], gen["partial"], gen["requested"]) == (0, False, 3520), why
        assert sorted(p.name for p in (j.run / "summary").iterdir()) == ["generate.json"], why
        assert not (j.run / "clips").exists() and j.code_untouched, why
        assert "completions.jsonl is missing (run generate first)" in j.steps["score"][2], why
        assert "a partial run" not in j.steps["render"][2] and "is missing" in j.steps["render"][2], why
        assert sorted(p.name for p in (j.pulled / "summary").iterdir()) == ["generate.json"], why
        assert j.steps["generate"][1].rstrip().splitlines()[-1].startswith("GENERATE_STOPPED 0/3520"), why
        if why == "the probe missed":
            assert gen["stopped"] == "seed probe" and len(j.loads) == 1, "the engine loads, the probe runs, no draw"
            assert [len(c) for c, _ in j.engine.calls] == [S.PROBE_N * len(S.PROBE_SEEDS)]
            assert gen["probe"]["passed"] is False and gen["probe"]["distinct"] == 6
        else:
            assert j.loads == [] and gen["stopped"].startswith("pre-check: "), why
            assert gen["probe"]["passed"] is None and gen["engine"] == {}, why
        rc, out, _ = laptop(repo, ["check", str(j.pulled)])
        assert rc in (2, 3) and out.rstrip().splitlines()[-1] != "STEP0_CHECK valid", why
        before = tree_hashes(repo)
        rc, out, err = laptop(repo, ["prereg", str(j.pulled), "--write"])
        assert rc in (2, 3) and tree_hashes(repo) == before, why
    reg = {why: stopped[why][0].doc("generate.json")["registration"] for why in ("no mark", "another mark")}
    assert reg["no mark"]["found"] is False and reg["no mark"]["mark"] is None
    assert reg["another mark"] == {"found": False, "mark": other, "decisions_sha256": sha_bytes(
        L.blob(later.repo, later.commits["another mark"], S.DECISIONS))}
    assert "holds no 'D-023 mark:' line" in stopped["no mark"][0].doc("generate.json")["stopped"]
    assert f"D-023 mark is {other!r}, not MARK" in stopped["another mark"][0].doc("generate.json")["stopped"]
    drift = stopped["a listed file changed"][0].doc("generate.json")
    assert f"1 differ (first {DRIFT})" in drift["stopped"] and drift["path_file"]["blobs_ok"] is False
    assert drift["registration"]["found"] is True, "the mark is there; the path is not"


@pytest.mark.parametrize("why", STOPPED)
def test_a_run_stopped_before_its_first_draw_reads_invalid_naming_why(stopped, why):
    """Finding E1: generate.json alone, stopped with nothing drawn, is read with no rows, so check reads it invalid
    with generate.stopped named (and generate.probe.passed for the probe), not bad input: §4.2's re-draw follows."""
    j, repo = stopped[why]
    rc, out, _ = laptop(repo, ["check", str(j.pulled)])
    assert rc == 3 and out.rstrip().splitlines()[-1] == "STEP0_CHECK invalid", out[-600:]
    rc, doc = check(repo, j.pulled)
    failed = doc["validity"]["failed"]
    assert "generate.stopped" in failed and doc["validity"]["of"] == 74 and "requests.missing" in failed
    assert cond(doc, "generate.stopped")["shows"] == j.doc("generate.json")["stopped"]
    assert ("generate.probe.passed" in failed) is True, "a pre-check stop never ran the probe; the probe missed"
    assert cond(doc, "generate.probe.passed")["shows"] == ("false" if why == "the probe missed" else "null")


def test_draws_cut_by_generates_deadline_read_invalid_with_generate_stopped_named(world):
    """Like a run stopped before its first draw (finding E1), a run with chunks drawn reads invalid, naming the stop.
    Each request takes 0.5 s on generate's clock: the probe's 6 end at 3 s, chunk k at 3 + 64k s, and the 15th,
    forecast to end at 899 + 1.25 x 64 = 979 s, is not started against the job's 960 s: 14 x 128 = 1,792 drawn."""
    clock = [0.0]
    engine = E.FakeEngine(answers(), clock=clock, per_request_s=0.5, info=H.ENGINE_INFO)
    j = job(world.base / "deadline", "13800004", world.snap, engine=engine, gen_clock=clock)
    assert j.rc() == {"generate": 1, "score": 0, "render": 2}
    gen, step0 = j.doc("generate.json"), j.doc("step0.json")
    assert (gen["stopped"], gen["partial"], gen["generated"], gen["requested"]) == ("deadline", False, 1792, 3520)
    assert gen["replay"]["stopped"] == "not run: the draws are not complete"
    assert (step0["n"], step0["requested"], step0["partial"], step0["render"]) == (1792, 3520, True, None)
    assert "a partial run is not rendered" in j.steps["render"][2] and not (j.run / "clips").exists()
    rc, out, _ = laptop(world.repo, ["check", str(j.pulled)])
    assert rc == 3 and out.rstrip().splitlines()[-1] == "STEP0_CHECK invalid"
    rc, doc = check(world.repo, j.pulled)
    failed = doc["validity"]["failed"]
    assert {"generate.stopped", "n", "requests.missing"} <= set(failed) and "generate.partial" not in failed
    assert cond(doc, "generate.stopped")["shows"] == "deadline" and cond(doc, "n")["shows"] == "1792"
    before = tree_hashes(world.repo)
    rc, _, err = laptop(world.repo, ["prereg", str(j.pulled), "--write"])
    assert rc == 3 and "is not a valid run" in err and tree_hashes(world.repo) == before


def test_a_run_stamped_at_a_commit_where_a_listed_blob_differs_reads_invalid(world, later):
    """The steps run in a tree that passes generate's pre-check (the code commit's) under a stamp that names the later
    commit, where a listed file differs: check's at-commit path condition, and only it, fails."""
    c3 = later.commits["a listed file changed"]
    code = snapshot(world.repo, world.c1, world.base / "keep" / "snap" / f"{world.c1[:12]}-as-{c3[:12]}",
                    stamp=L.stamp(c3))
    j = job(world.base / "drift", DRIFT_JOB, code)
    assert j.rc() == {"generate": 0, "score": 0, "render": 0}
    assert j.doc("generate.json")["path_file"]["blobs_ok"] is True and j.doc("step0.json")["render_stamp"] == \
        L.stamp(c3)
    rc, doc = check(later.repo, j.pulled)
    assert rc == 3 and doc["validity"]["failed"] == ["the path at the generate commit"]
    assert f"at {c3[:12]}: " in cond(doc, "the path at the generate commit")["shows"]
    assert f"1 differ ({DRIFT})" in cond(doc, "the path at the generate commit")["shows"]
    rc, out, _ = laptop(later.repo, ["prereg", str(j.pulled), "--write"])
    assert rc == 3 and not (later.repo / S.HELDOUT_FILE).exists()


# --------------------------------------------------------------------------------------------- 3: the replay
def test_a_replay_that_differs_for_some_rows_is_shown_and_never_judged(world):
    held = [r for r in H.all_requests() if r.stream == "rm_heldout"]
    differ = frozenset(r.draw_seed for r in held[::10])
    engine = E.FakeEngine(answers(differ), info=H.ENGINE_INFO)
    j = job(world.base / "replay", REPLAY_JOB, world.snap, engine=engine)
    assert j.rc() == {"generate": 0, "score": 0, "render": 0}
    rp = j.doc("generate.json")["replay"]
    assert (rp["n"], rp["same"], rp["differ"], rp["stopped"]) == (320, 320 - len(differ), len(differ), None)
    rows = j.doc("replay.jsonl")
    assert sorted(r["i"] for r in rows if not r["same"]) == sorted(r.i for r in held[::10])
    for name in ("completions.jsonl", "scored.jsonl"):
        assert (j.pulled / "summary" / name).read_bytes() == (world.main.pulled / "summary" / name).read_bytes(), name
    assert j.doc("sets.json") == world.main.doc("sets.json"), "the replay never enters the sets"
    rc, doc = check(world.repo, j.pulled)
    assert rc == 0 and doc["validity"]["failed"] == [] and doc["verdict"] == "valid"
    assert doc["shown"]["replay"]["differ"] == len(differ) and doc["shown"]["replay_file"] == {
        "rows": 320, "same": 320 - len(differ)}
    rc, out, _ = laptop(world.repo, ["check", str(j.pulled)])
    assert rc == 0 and f"replay: n 320, same {320 - len(differ)}, differ {len(differ)}, stopped null" in out


# --------------------------------------------------------------------------------------------- Q6(b)
def test_under_q6b_the_job_draws_2080_and_its_files_read_valid(world):
    """Q6(b) (spec §7), as its code commit would set it: src/lofi_rl/step0/__init__.py's STREAMS draws the pool's 220
    train briefs and MARK is their line, the path file is written again, and D-023 holds that mark. The job draws
    2,080 requests and check and prereg accept its files (this process runs this checkout's code with the same two
    constants, which the snapshot's edited __init__.py is shown to hold)."""
    import runpy
    q6b = (("rm_heldout", "all"), ("train", "pool"))
    mark = R.mark_line(q6b)
    repo = copy_repo(world.repo, world.base / "laptop-q6b" / "repo")
    init = repo / "src/lofi_rl/step0/__init__.py"
    old = {'STREAMS: tuple[tuple[str, str], ...] = (("rm_heldout", "all"), ("train", "all"))':
           f"STREAMS: tuple[tuple[str, str], ...] = {q6b!r}",
           'MARK = ("D-023 mark: rule step0-select/1; streams rm_heldout (seed 5101, all 40 briefs) and train '
           '(seed 596, all 400 "\n        "briefs); 8 draws a brief, 3520 requests")': f"MARK = {mark!r}"}
    body = init.read_text()
    for a, b in old.items():
        assert body.count(a) == 1, a
        body = body.replace(a, b)
    init.write_text(body)
    (repo / S.DECISIONS).write_text(decisions(mark), encoding="utf-8")
    assert call(["path", "--write", "--root", str(repo)])[0] == 0           # the path file again, last
    L.git(repo, "add", "-A")
    L.git(repo, "commit", "-qm", "Q6(b): STREAMS and MARK, D-023's mark, the path file")
    cb = L.git(repo, "rev-parse", "HEAD").strip()
    code = snapshot(repo, cb, world.base / "keep" / "snap" / cb[:12])
    snap_init = runpy.run_path(str(code / "src/lofi_rl/step0/__init__.py"))
    assert (snap_init["STREAMS"], snap_init["MARK"]) == (q6b, mark)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(S, "STREAMS", q6b)
        mp.setattr(S, "MARK", mark)
        j = job(world.base / "q6b", "13800011", code)
        rc, out, err = laptop(repo, ["check", str(j.pulled)])
        rc_doc, doc = check(repo, j.pulled)
        rc_p, out_p, err_p = laptop(repo, ["prereg", str(j.pulled), "--write"])
    assert j.rc() == {"generate": 0, "score": 0, "render": 0}, {k: v[2][-1500:] for k, v in j.steps.items()}
    gen = j.doc("generate.json")
    assert (gen["requested"], gen["generated"], gen["registration"]["mark"]) == (2080, 2080, mark)
    assert gen["streams"] == {"rm_heldout": {"seed": 5101, "briefs": 40, "draws": 8},
                              "train": {"seed": 596, "briefs": 220, "draws": 8}}
    assert rc == rc_doc == 0 and out.rstrip().splitlines()[-1] == "STEP0_CHECK valid", out[-3000:] + err[-1000:]
    assert doc["validity"]["failed"] == [] and doc["validity"]["of"] == 74 and cond(doc, "n")["shows"] == "2080"
    sets = j.doc("sets.json")
    assert (len(sets["heldout"]), len(sets["refb"]), len(sets["pool"])) == (40, 220, 220)
    assert sum(len(e["taken"]) for e in sets["pool"]) == 657 and len(sets["heldout_text"]) == 1760
    assert sets["heldout"] == world.main.doc("sets.json")["heldout"], "the held-out set is Q6's either way"
    assert rc_p == 0 and f"wrote {S.HELDOUT_FILE}, {S.RUN_FILE}" in out_p, err_p
    st = json.loads((repo / S.RUN_FILE).read_text())
    assert st["rule"]["answers"]["Q6"] == {"streams": [list(p) for p in q6b]} and len(st["refb"]) == 220


# --------------------------------------------------------------------------------------------- 5: a tampered record
@pytest.mark.parametrize("field_, value, failed", [
    ("clip_id", "c0000000000000000", ["sets.json = step0.select", "clip ids"]),
    ("canon_sha256", "0" * 64, ["clip ids"])])
def test_a_render_record_tampered_after_the_fact_reads_invalid(world, tmp_path, field_, value, failed):
    run = L.copy_run(world.main.pulled, tmp_path / f"step0-{JOB}")
    first = next(r for r in world.main.doc("clips.json")["clips"] if r["ok"])
    L.edit_json(run / "summary" / "clips.json", lambda d: next(r for r in d["clips"] if r["i"] == first["i"]).update(
        {field_: value}))
    rc, doc = check(world.repo, run)
    assert rc == 3 and doc["validity"]["failed"] == failed
    assert cond(doc, "clip ids")["shows"].startswith(f"1 wrong: i={first['i']}: ")
    rc, _, err = laptop(world.repo, ["prereg", str(run)])
    assert rc == 3 and "is not a valid run" in err


# --------------------------------------------------------------------------------------------- 6: the redo pair
def test_a_score_only_redo_never_reads_valid_without_its_render(world):
    """§4.2 runs `score --from OUT --run REDO` "and/or" `render --from OUT --run REDO`. Score alone writes step0.json
    with its render block and render stamp null, and no sets.json or clips.json, so check reads REDO's step0.json
    beside OUT's sets and clips: invalid, even with its amendment and an OUT that is valid itself (the main run here,
    so every failure is the redo's). A score redo needs its render redo (finding E3)."""
    repo = copy_repo(world.repo, world.base / "laptop-score-redo" / "repo")
    c2 = commit(repo, "a scoring fix", {FIX: (repo / FIX).read_text() + "# a scoring fix\n",
                                        S.DECISIONS: decisions(extra=amendment(f"step0-{JOB}", [FIX]))})
    code = snapshot(repo, c2, world.base / "keep" / "snap" / f"{c2[:12]}-score")
    redo = redo_job(world.base / "score-redo", "13800012", code, world.main, ("score",))
    assert redo.rc() == {"score": 0} and redo.extra["out_untouched"]
    assert sorted(p.name for p in (redo.pulled / "summary").iterdir()) == ["scored.jsonl", "step0.json"]
    assert (redo.doc("step0.json")["render"], redo.doc("step0.json")["render_stamp"]) == (None, None)
    rc, doc = check(repo, world.main.pulled, redo.pulled)
    assert rc == 3 and doc["validity"]["failed"] == [C.REDO_STAMP_NAME, "render.errors_final", "render.skipped",
                                                     "render.other_gates"]
    assert "a score redo without its render redo" in cond(doc, C.REDO_STAMP_NAME)["shows"], "check says why"



REDOS = [(("render",), True, "13800008"), (("render",), False, "13800009"), (("score", "render"), True, "13800010")]


@pytest.mark.parametrize("steps, amended, redo_id", REDOS, ids=["render, with its amendment", "render, without it",
                                                                "score and render, with its amendment"])
def test_a_redo_of_a_cut_render_reads_valid_only_with_its_amendment(world, cut, steps, amended, redo_id):
    """§4.2: OUT's held-out render was cut by its deadline (render exit 1, briefs undetermined), so OUT is invalid. The
    redo renders again (or scores and renders again) from OUT's records at a later commit that changes a listed file:
    check OUT --redo REDO is valid when that commit's DECISIONS.md names the file in a redo amendment for OUT, and
    invalid without it."""
    assert cut.rc() == {"generate": 0, "score": 0, "render": 1}
    blk = cut.doc("step0.json")["render"]
    assert blk["skipped"] > 0 and blk["rendered"] < 159
    rc, doc = check(world.repo, cut.pulled)
    assert rc == 3 and doc["validity"]["failed"] == ["render.skipped", "held-out briefs undetermined"]
    run_name = f"step0-{CUT_JOB}"
    repo = copy_repo(world.repo, world.base / f"laptop-redo-{redo_id}" / "repo")
    extra = amendment(run_name, [FIX]) if amended else ""
    c2 = commit(repo, "the redo's fix", {FIX: (repo / FIX).read_text() + "# the redo's fix\n",
                                         S.DECISIONS: decisions(extra=extra)})
    code = snapshot(repo, c2, world.base / "keep" / "snap" / f"{c2[:12]}-{redo_id}")
    redo = redo_job(world.base / f"redo-{redo_id}", redo_id, code, cut, steps)
    assert redo.rc() == dict.fromkeys(steps, 0) and redo.extra["out_untouched"] and redo.code_untouched
    step0 = redo.doc("step0.json")
    assert step0["score_stamp"] == L.stamp(c2 if "score" in steps else world.c1)
    assert step0["render_stamp"] == L.stamp(c2) and step0["render"]["skipped"] == 0
    assert step0["render"] == world.main.doc("step0.json")["render"], "the redo renders the main run's 159"
    assert redo.doc("sets.json")["heldout"] == world.main.doc("sets.json")["heldout"]
    rc, out, err = laptop(repo, ["check", str(cut.pulled), "--redo", str(redo.pulled)])
    rc_json, doc = check(repo, cut.pulled, redo.pulled)
    if amended:
        assert rc == rc_json == 0 and out.rstrip().splitlines()[-1] == "STEP0_CHECK valid", out[-3000:]
        assert doc["validity"]["failed"] == [] and C.REDO_STAMP_NAME in [c["name"] for c in doc["validity"][
            "conditions"]]
        shows = cond(doc, "the path at the redo commit")["shows"]
        assert f"at {c2[:12]}:" in shows and f"1 differ ({FIX})" in shows and f"names {FIX}" in shows
        before = tree_hashes(repo)
        rc, out, err = laptop(repo, ["prereg", str(cut.pulled), "--redo", str(redo.pulled), "--write"])
        assert rc == 0 and f"with its redo step0-redo-{redo_id}" in out, err
        assert set(tree_hashes(repo)) - set(before) == {S.HELDOUT_FILE, S.RUN_FILE}
        st = json.loads((repo / S.RUN_FILE).read_text())
        assert st["run"]["redo"] == f"step0-redo-{redo_id}" and st["run"]["stamp"]["render"] == L.stamp(c2)
    else:
        assert rc == rc_json == 3 and doc["validity"]["failed"] == ["the path at the redo commit"]
        assert f"not named: {FIX}" in cond(doc, "the path at the redo commit")["shows"]
