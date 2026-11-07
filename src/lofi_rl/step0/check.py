"""`python -m lofi_rl.step0 check RUN_DIR [--redo REDO_DIR] [--json] [--rescore]`: is a step-0 run valid?
(docs/design/step0_samples.md §4.1-4.2; D-023 registers this list.)

It runs on the laptop after `make pull`, over RUN_DIR/summary/ (§2.6's seven files), the registered requests
(lofi_rl.step0.requests, from the streams file) and, with git, the commits the run's stamps name. **Validity** is
D-020's list (its amended Validity bullets) with D-022's values, adapted to per-request seeds, in ten groups. Each
condition is reported by name, the file and field it reads (generate.json's prefixed `generate.`, clips.json's
`clips.`, step0.json's bare, as junk.json's are in lofi_rl.gate1.check) or what the laptop reproduces, with what the
files show:
- **Complete:** generate.json `stopped` null and `partial` false; step0.json `n` and `requested` and generate.json
  `requested` all the marked streams' requests (3,520 under Q6(a)); `unreadable_lines` 0; and over completions.jsonl
  against those requests, `requests.missing`, `requests.duplicate` and `requests.foreign` 0;
- **Seeded:** `requests.seed_mismatch` 0 (each row's draw_seed and sp_seed are brief_streams.draw_seed(seed, index,
  k)); generate.json `probe.passed` true; `requests.prompt_tokens_mismatch` 0 (each row's prompt_tokens is its
  cell's in the Phase-1 gate run, lofi_rl.step0.PROMPT_TOKENS);
- **As registered:** generate.json `model.revision`, `streams_file.sha256` and `fresh`, `streams` (the mark's),
  `palette.sha256` and `sampling` (runs/junk-13632676's); `requests.brief_mismatch` 0 (each row's brief is
  sample_brief(seed, index)); per stream, step0.json `by_stream.<stream>.prompt` (v2, its template, matching
  generate's, `rows_matching` the stream's requests, `example_mismatch` 0) and `palette` (the frozen file,
  `matches_generate` true); clips.json `palette.sha256` (render's);
- **Registered first:** generate.json `registration.found` true and `registration.mark` MARK; and, read with git, the
  generate stamp's commit holds MARK in its docs/DECISIONS.md as generate reads it (requests.find_marks), the file's
  sha256 being `registration.decisions_sha256`;
- **Constrained,** per stream: `constrained.enabled` true, `regex_sha256` and `in_language.regex_sha256` the frozen
  grammar's, `in_language.matches_generate` true, `stop.out_of_language` 0 and `other.n` 0;
- **What the regex guarantees,** per stream: `failures.G1`, `G3`, `G4`, `G5` and `crash` 0;
- **One stamp, one path:** generate.json `host.stamp`, step0.json `score_stamp` and `render_stamp` show one
  `commit=… dirty=0`; generate.json `path_file.blobs_ok` true; and, with git, at that commit the path file
  (docs/prereg/D-023_step0_path.txt) is the one generate checked (its sha256 is `path_file.sha256`) and every file it
  lists has its listed blob. A redo pair reads §4.2's rule instead (below);
- **Held-out render:** step0.json `render.errors_final`, `render.skipped` and `render.other_gates` 0, and no held-out
  brief undetermined (sets.json);
- **Sets:** every heldout row is stream rm_heldout, and every refb and pool row train (the entries' streams and their
  taken rows' in scored.jsonl); the pool takes at least one draw; under Q5(a) no pool pick carries a held-out clip's
  canonical text (sets.json `heldout_text`), under (b) or (c) none is a held-out clip;
- **Reproduced on the laptop:** sets.json equals lofi_rl.step0.select over scored.jsonl and clips.json; every clip id
  equals the formula (taste_pipeline.md:76) over its record's canon_sha256 (score's for its row), the row's brief,
  clips.json's palette label and the record's render_cfg_sha8, one config for the run; and render_code.source_sha8
  equals lofi_rl.scoring.features.source_sha8 over the render stamp's commit, read with git.
A value must have the registered JSON type too, as lofi_rl.gate1.check reads it: `partial` 0 is not false, and `n`
3520.0 or true is not 3520.

**A redo pair** (§4.2: `check OUT --redo REDO`): generate's records (generate.json, completions.jsonl, replay.jsonl)
come from OUT, and score's and render's (scored.jsonl, step0.json, sets.json, clips.json) from REDO, each from OUT
when REDO does not hold it (a render-only redo reads OUT's scored.jsonl). A score redo is always followed by its render
redo: new scores need new sets, which only render writes, so a pair whose REDO holds score's records alone reads
invalid, the stamp condition saying why (finding E3). The stamp condition then reads both runs'
stamps: each shows dirty=0, generate's commit holds every listed blob as above, and each later commit holds every
listed blob of that same path file except the files D-023's amendment for the run names, on a line of that commit's
docs/DECISIONS.md:

    D-023 redo amendment: run step0-<job>; changed <path>, <path>

That form is read leniently (amendment_files): `path` in backticks, a closing period, a bold line or a note after a path
read as the bare form, and "changed nothing" names no file; a named file the path file does not list is shown.

**Bad input** (exit 2, nothing decided): a summary/ folder or a file missing or unreadable (replay.jsonl, sets.json
and clips.json may be absent: their conditions fail instead), a schema that is not the step's, or files that are not
one run's own, as lofi_rl.gate1.load_run refuses them: scored.jsonl against completions.jsonl (the same rows, each
text_sha256 its row's text's), step0.json's `n` against scored.jsonl, its `render` block against clips.json
(select.render_block), and one render-code key in step0.json, sets.json and clips.json. A scored.jsonl or step0.json
missing beside generate's records is named as what it is, a scoring defect that §4.2 redoes from the records.

**A run stopped before its first draw** (generate.json alone: `stopped` set, `generated` 0, no completions.jsonl, as a
pre-check, the probe or the engine's load leaves it) is read with no rows, not as bad input: it reads invalid, naming
generate.stopped (and generate.probe.passed when the probe missed), and §4.2 draws it again whole.

**Shown, never judged** (§4.1): the backend (as D-020 treats it), the probe, the replay, the timings and the GPU,
`render.errors_retried`, the per-set near-copy, repair and verbatim counts and the held-out-text counts, item 8's
readouts on the held-out tuples (D-021 item 8: each tuple's pairwise J5 in junk's copy form and the K/D form, its
verbatim-line sharing, and per pair whether the two clips differ in A or B; `item8`, which prereg writes), and with
`--rescore` the differences between scored.jsonl and junk.score_row over completions.jsonl with this checkout's code
(meaningful at the run's commit).

Exit 0 valid, 3 invalid, 2 bad input; the last line printed is `STEP0_CHECK <verdict>` (`--json` prints the report
alone, with its verdict). Laptop only: git reads the commits (CLAUDE.md §1). Sound claims: none.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import lofi_rl.step0 as S
from lofi_rl import brief_streams
from lofi_rl.brief import PITCH_NAMES, Brief, sample_brief
from lofi_rl.brief_streams import ROOT
from lofi_rl.step0 import Step0Error, short
from lofi_rl.step0 import path as P
from lofi_rl.step0 import requests as R
from lofi_rl.step0 import select as X

SCHEMA = "lofi-rl/step0-check/1"
GROUPS = ("Complete", "Seeded", "As registered", "Registered first", "Constrained", "What the regex guarantees",
          "One stamp, one path", "Held-out render", "Sets", "Reproduced on the laptop")
GENERATE_FILES = ("generate.json", "completions.jsonl", "replay.jsonl")      # a redo pair reads these from OUT
LATER_FILES = ("scored.jsonl", "step0.json", "sets.json", "clips.json")      # ... and these from REDO, when it has them
MAY_BE_ABSENT = ("replay.jsonl", "sets.json", "clips.json")     # their conditions fail, or their readouts are absent
REGEX_GATES = ("G1", "G3", "G4", "G5", "crash")      # D-020: what the regex guarantees (DECISIONS.md:221)
STAMP_NAME = "generate.host.stamp = score_stamp = render_stamp"
REDO_STAMP_NAME = "stamps of the redo pair"
AMENDMENT = re.compile(r"D-023 redo amendment: run (step0-[\w-]+); changed ([^\n]*)")
AMENDED_ITEMS = re.compile(r",|\s+and\s+")         # an amendment's files: comma-separated, or "a.py and b.py"
NOTHING = "nothing"                                  # `changed nothing`: a redo that changed no listed file
MAX_LISTED = 10                         # examples kept per kind of problem
TIMINGS = ("load_s", "probe_s", "gen_s", "tok_per_s", "replay_s", "total_s")
_STAMP = re.compile(r"^commit=(\S+) dirty=(\S+)")
_ID = re.compile(r"(\d+):(\d+)")
MISSING = object()


# --------------------------------------------------------------------------------------------- values
def field(doc, path: str):
    """The value at a dotted path of a JSON document, or MISSING."""
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur


def shown(v) -> str:
    """A file's value as the reports print it: JSON's spelling (null, true, 1.0), a mapping as k=v pairs."""
    if v is MISSING:
        return "missing"
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return " ".join(f"{k}={shown(x)}" for k, x in v.items())
    return json.dumps(v)


def _whole(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _int(v, want: int) -> bool:
    return _whole(v) and v == want


def _str(v, want: str) -> bool:
    return isinstance(v, str) and v == want


def _dumps(v) -> str:
    return json.dumps(v, sort_keys=True)


def _same_json(v, want) -> bool:
    """Equal as JSON text, so 0 and false, or 1 and 1.0, differ (dict order aside)."""
    return v is not MISSING and _dumps(v) == _dumps(want)


def stamp_fields(stamp) -> tuple[str, str] | None:
    """(commit, dirty) of a `.sync-stamp` line (`commit=… dirty=… diff=… at=…`), or None."""
    m = _STAMP.match(stamp) if isinstance(stamp, str) else None
    return (m.group(1), m.group(2)) if m else None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def clip_id(canon_sha256: str, brief: Brief, palette_label: str, render_cfg_sha8: str) -> str:
    """taste_pipeline.md:76's clip id from recorded inputs: "c" + sha256(canon_sha256 | bpm,key,mode,energy |
    palette_id@sha8 | render_cfg_sha8)[:16], the key lofi_rl.scoring.protocol.clip_id hashes for a render config
    whose render_cfg_sha8 is the recorded one."""
    key = (f"{canon_sha256}|{brief.bpm},{PITCH_NAMES[brief.tonic]},{brief.mode},{brief.energy}|{palette_label}|"
           f"{render_cfg_sha8}")
    return "c" + hashlib.sha256(key.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------------- reading a run
@dataclass(frozen=True)
class Run:
    """One step-0 run's summary files, or a redo pair's (`load_run`)."""

    dir: Path                        # the run folder (OUT)
    redo: Path | None                # a recovery's folder (REDO), or None
    files: Mapping[str, Path]        # summary file name -> the path it was read from
    generate: dict
    rows: list[dict]                 # completions.jsonl, as written
    unreadable: int                  # completions.jsonl lines that are not a JSON row with a whole-number i
    replay: list[dict] | None
    scored: list[dict]
    step0: dict
    sets: dict | None
    clips: dict | None

    @property
    def name(self) -> str:
        return self.dir.resolve().name

    @property
    def host(self) -> dict:
        h = self.generate.get("host")
        return h if isinstance(h, dict) else {}

    @property
    def job(self) -> str | None:
        job = self.host.get("job")
        return None if job is None else str(job)

    @property
    def run_name(self) -> str:
        """step0-<job>: the name D-023's amendments give the run (the folder's name when no job is recorded)."""
        return f"step0-{self.job}" if self.job else self.name

    @property
    def records(self) -> list[dict]:
        return list((self.clips or {}).get("clips") or [])

    @property
    def render_code(self) -> dict | None:
        """The render-code key {stamp, source_sha8}: one in step0.json, sets.json and clips.json (load_run)."""
        for doc in (self.step0, self.clips, self.sets):
            code = (doc or {}).get("render_code")
            if isinstance(code, dict):
                return code
        return None

    def entries(self, name: str) -> list[dict]:
        return list((self.sets or {}).get(name) or [])

    def text_rows(self) -> dict[int, dict]:
        """i -> its completions.jsonl row (the first, when one is repeated)."""
        out: dict[int, dict] = {}
        for r in self.rows:
            out.setdefault(r["i"], r)
        return out

    def scored_rows(self) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for s in self.scored:
            out.setdefault(s["i"], s)
        return out


def _read_json(path: Path, problems: list[str], *, need: bool = True) -> dict | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if need:
            problems.append(f"{path} is missing")
        return None
    except (OSError, UnicodeDecodeError, ValueError) as err:
        problems.append(f"{path} is not readable JSON ({err})")
        return None
    if not isinstance(doc, dict):
        problems.append(f"{path} is not a JSON object")
        return None
    return doc


def _read_rows(path: Path, problems: list[str], *, strict: bool, need: bool = True) -> tuple[list[dict], int] | None:
    """The rows of a JSONL file that are objects with a whole-number `i`, and how many lines are not (junk.read_jsonl's
    rule). `strict`: such a line is a problem (scored.jsonl is written whole by score); otherwise it is counted
    (completions.jsonl's are step0.json's `unreadable_lines`)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if need:
            problems.append(f"{path} is missing")
        return None
    except (OSError, UnicodeDecodeError) as err:
        problems.append(f"{path} is not readable ({err})")
        return None
    rows, bad = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(row, dict) and _whole(row.get("i")):
            rows.append(row)
        else:
            bad += 1
    if strict and bad:
        problems.append(f"{path} has {bad} unreadable line(s); score writes it whole")
    return rows, bad


def _sets_problems(sets: dict) -> list[str]:
    """sets.json's shape: what the conditions read must be there to be read."""
    problems = []
    for name in S.SETS:
        entries = sets.get(name)
        if not isinstance(entries, list) or not all(
                isinstance(e, dict) and isinstance(e.get("brief_id"), str) and isinstance(e.get("taken"), list)
                and all(isinstance(t, dict) for t in e["taken"]) for e in entries):
            problems.append(f"sets.json: `{name}` must be a list of entries, each with its brief_id and `taken` list")
    flags = sets.get("heldout_text")
    if not isinstance(flags, list) or not all(isinstance(d, dict) for d in flags):
        problems.append("sets.json: `heldout_text` must be a list of records")
    return problems


def stopped_before_drawing(gen: dict | None, files: Mapping[str, Path]) -> bool:
    """generate stopped before its first draw (a pre-check, the probe, the engine's load or its first call), as
    generate leaves such a run: generate.json alone, `stopped` set and `generated` 0, and no completions.jsonl (opened
    only when the draws start); score and render then refuse and write nothing."""
    return (isinstance(gen, dict) and gen.get("schema") == S.GENERATE_SCHEMA and gen.get("stopped") is not None
            and _int(gen.get("generated"), 0) and not files["completions.jsonl"].exists())


SCORE_LEFT_NOTHING = ("score left no complete records (it writes scored.jsonl, then step0.json, at its end, so its "
                      "timeout or a crash leaves them absent): with generate's records complete this is a scoring "
                      "defect, D-020's 'a crash', redone from the run's records with score and render --from OUT "
                      "--run REDO (spec section 4.2)")


def load_run(run_dir: Path | str, redo_dir: Path | str | None = None) -> Run:
    """The run's summary files, or a redo pair's, cross-checked (module doc: bad input). Step0Error lists every
    problem; none is a validity verdict. A run whose generate stopped before its first draw (stopped_before_drawing)
    is read with no rows, so that its conditions say why it decides nothing (finding E1)."""
    out = Path(run_dir)
    redo = None if redo_dir is None else Path(redo_dir)
    problems = [f"{d / 'summary'} is not a folder: a run folder holds summary/ (make pull brings it)"
                for d in (out, redo) if d is not None and not (d / "summary").is_dir()]
    if problems:
        raise Step0Error(problems)
    if redo is not None and redo.resolve() == out.resolve():
        raise Step0Error([f"--redo {redo} is the run folder itself: a redo writes its own folder (§4.2)"])
    files: dict[str, Path] = {}
    for name in S.SUMMARY_FILES:
        here = out
        if redo is not None and name not in GENERATE_FILES and (redo / "summary" / name).exists():
            here = redo
        files[name] = here / "summary" / name
    gen = _read_json(files["generate.json"], problems)
    if not problems and stopped_before_drawing(gen, files):
        extra = [str(files[name]) for name in S.SUMMARY_FILES if name != "generate.json" and files[name].exists()]
        if extra:
            raise Step0Error([f"{', '.join(extra)} beside a generate.json that drew nothing (stopped "
                              f"{shown(gen.get('stopped'))[:120]}): not one run's files"])
        return Run(dir=out, redo=redo, files=files, generate=gen, rows=[], unreadable=0, replay=None, scored=[],
                   step0={}, sets=None, clips=None)
    comp = _read_rows(files["completions.jsonl"], problems, strict=False)
    rep = _read_rows(files["replay.jsonl"], problems, strict=False, need="replay.jsonl" not in MAY_BE_ABSENT)
    sco = _read_rows(files["scored.jsonl"], problems, strict=True)
    step0 = _read_json(files["step0.json"], problems)
    sets = _read_json(files["sets.json"], problems, need="sets.json" not in MAY_BE_ABSENT)
    clips = _read_json(files["clips.json"], problems, need="clips.json" not in MAY_BE_ABSENT)
    for name, doc in (("generate.json", gen), ("step0.json", step0), ("sets.json", sets), ("clips.json", clips)):
        if doc is not None and doc.get("schema") != S.SCHEMAS[name]:
            problems.append(f"{files[name]}: schema {doc.get('schema')!r}, not {S.SCHEMAS[name]!r}")
    if comp is not None and not (files["scored.jsonl"].exists() and files["step0.json"].exists()):
        problems.append(SCORE_LEFT_NOTHING)                   # finding R2-7: the route §4.2 gives it
    if problems:
        raise Step0Error(problems)
    rows, unreadable = comp
    scored = sco[0]

    # one run's own files (lofi_rl.gate1.load_run's rule): score read this completions.jsonl ...
    row_i, sc_i = {r["i"] for r in rows}, {s["i"] for s in scored}
    if sc_i - row_i:
        problems.append(f"scored.jsonl rows {sorted(sc_i - row_i)[:MAX_LISTED]} have no completions.jsonl row: "
                        "not one run")
    if row_i - sc_i:
        problems.append(f"completions.jsonl rows {sorted(row_i - sc_i)[:MAX_LISTED]} are not in scored.jsonl: "
                        "score did not read this completions.jsonl")
    texts: dict[int, object] = {}
    for r in rows:
        texts.setdefault(r["i"], r.get("text"))
    off = sorted({s["i"] for s in scored if s["i"] in texts and not (
        isinstance(texts[s["i"]], str) and s.get("text_sha256") == sha256_text(texts[s["i"]]))})
    if off:
        problems.append(f"scored.jsonl rows {off[:MAX_LISTED]} carry a text_sha256 that is not their "
                        "completions.jsonl text's: not a score of these completions")
    n = step0.get("n")
    # numbers compared as numbers here: the validity list holds `n` to the registered JSON type (conditions)
    if not (isinstance(n, (int, float)) and not isinstance(n, bool) and n == len(scored)):
        problems.append(f"scored.jsonl holds {len(scored)} rows, step0.json `n` is {n!r}: not one score")
    # ... and one render step wrote step0.json's render block, sets.json and clips.json
    recs = None
    if clips is not None:
        recs = clips.get("clips")
        if not isinstance(recs, list) or not all(isinstance(r, dict) and _whole(r.get("i")) for r in recs):
            problems.append("clips.json: `clips` must be a list of records, each with its row `i`")
            recs = None
    if sets is not None:
        problems += _sets_problems(sets)
    block = step0.get("render")
    if block is not None and recs is not None and _dumps(block) != _dumps(X.render_block(recs)):
        problems.append(f"step0.json's `render` block {_dumps(block)} is not the one clips.json gives, "
                        f"{_dumps(X.render_block(recs))} (select.render_block): not one render step's files")
    if block is not None and clips is None:
        problems.append("step0.json has a `render` block but clips.json is absent: not one render step's files")
    codes = {name: doc.get("render_code") for name, doc in (("step0.json", step0), ("sets.json", sets),
                                                            ("clips.json", clips))
             if doc is not None and doc.get("render_code") is not None}
    if len({_dumps(c) for c in codes.values()}) > 1:
        problems.append(f"the render-code key differs in {', '.join(codes)} "
                        f"({'; '.join(f'{k} {_dumps(v)}' for k, v in codes.items())}): not one render step's files")
    code, rstamp = step0.get("render_code"), step0.get("render_stamp")
    if isinstance(code, dict) and rstamp is not None and code.get("stamp") != rstamp:
        problems.append(f"step0.json: render_stamp {rstamp!r} is not render_code.stamp {code.get('stamp')!r}: not "
                        "one render step's records")
    if problems:
        raise Step0Error(problems)
    return Run(dir=out, redo=redo, files=files, generate=gen, rows=rows, unreadable=unreadable,
               replay=None if rep is None else rep[0], scored=scored, step0=step0, sets=sets, clips=clips)


# --------------------------------------------------------------------------------------------- the requests
def requests_block(rows: Sequence[Mapping], reqs: Sequence[R.Request]) -> dict:
    """completions.jsonl against the registered requests (requests.requests): rows written, and the rows that are
    foreign (not the request their `i` names: its stream, seed, index, brief id and k), repeat a request, carry
    another draw_seed or sp_seed than brief_streams.draw_seed(seed, index, k), another brief than
    sample_brief(seed, index), or another prompt_tokens than their cell's (PROMPT_TOKENS); and the requests with no
    row. Each count with its first rows."""
    by_i = {r.i: r for r in reqs}
    seen: set[int] = set()
    found: dict[str, list[int]] = {k: [] for k in ("foreign", "duplicate", "seed_mismatch", "brief_mismatch",
                                                     "prompt_tokens_mismatch")}
    tokens: dict[str, Counter] = {}
    for row in rows:
        i = row["i"]
        req = by_i.get(i)
        if req is None or not (_str(row.get("stream"), req.stream) and _int(row.get("seed"), req.seed)
                               and _int(row.get("index"), req.index) and _str(row.get("brief_id"), req.brief_id)
                               and _int(row.get("k"), req.k)):
            found["foreign"].append(i)
            continue
        if i in seen:
            found["duplicate"].append(i)
            continue
        seen.add(i)
        ds = brief_streams.draw_seed(req.seed, req.index, req.k)
        if not (_int(row.get("draw_seed"), ds) and _int(row.get("sp_seed"), ds)):
            found["seed_mismatch"].append(i)
        if not _same_json(row.get("brief"), asdict(sample_brief(req.seed, req.index))):
            found["brief_mismatch"].append(i)
        pt = row.get("prompt_tokens")
        tokens.setdefault(req.cell, Counter())[pt if _whole(pt) else shown(pt)] += 1
        if not _int(pt, S.PROMPT_TOKENS[req.cell]):
            found["prompt_tokens_mismatch"].append(i)
    missing = [r.i for r in reqs if r.i not in seen]
    out: dict = {"requested": len(reqs), "rows": len(rows), "missing": len(missing),
                 "first_missing": missing[:MAX_LISTED]}
    for kind, ids in found.items():
        out[kind] = len(ids)
        out[f"first_{kind}"] = ids[:MAX_LISTED]
    out["prompt_tokens"] = {cell: {str(k): n for k, n in c.items()} for cell, c in tokens.items()}
    return out


# --------------------------------------------------------------------------------------------- git (the laptop)
def _git(repo: Path, *args: str) -> bytes:
    """`git -C repo args`' output (laptop only, CLAUDE.md §1); Step0Error when git cannot answer."""
    try:
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as err:
        raise Step0Error([f"git {' '.join(args[:2])} failed in {repo} ({err})"]) from err
    if p.returncode != 0:
        raise Step0Error([f"git {' '.join(args[:3])} in {repo}: "
                          f"{p.stderr.decode('utf-8', errors='replace').strip()[:300] or f'exit {p.returncode}'}"])
    return p.stdout


def resolve(repo: Path, commit: str) -> str:
    """The full sha of a stamp's commit (`git rev-parse`); Step0Error when the repo has no such commit."""
    if not re.fullmatch(r"[0-9a-f]{7,40}", commit or ""):
        raise Step0Error([f"{commit!r} is not a commit sha"])
    try:
        return _git(repo, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}").decode().strip()
    except Step0Error as err:
        raise Step0Error([f"commit {commit} is not in {repo} (git rev-parse)"]) from err


def blob_at(repo: Path, commit: str, path: str) -> bytes:
    """A file's bytes at a commit (`git cat-file blob <commit>:<path>`); Step0Error when it has none."""
    return _git(repo, "cat-file", "blob", f"{commit}:{path}")


@lru_cache(maxsize=16)
def _source_sha8_at(repo: str, commit: str) -> str:
    from lofi_rl.scoring.features import SOURCE_GLOBS, source_sha8
    tops = sorted({g.split("/", 1)[0] for g in SOURCE_GLOBS})
    data = _git(Path(repo), "archive", "--format=tar", commit, "--", *tops)
    with tempfile.TemporaryDirectory(prefix="step0-check-") as tmp:
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            tar.extractall(tmp, filter="data")
        return source_sha8(Path(tmp), SOURCE_GLOBS)


def source_sha8_at(repo: Path, commit: str) -> str:
    """lofi_rl.scoring.features.source_sha8 over the tree of `commit` (a full sha): its source files, from git
    archive, in a temporary folder."""
    return _source_sha8_at(str(Path(repo).resolve()), commit)


def _amended(item: str) -> str | None:
    """One file an amendment line names: the item's first word, less backticks and asterisks and a closing period,
    comma, semicolon or colon; None for "nothing" or an empty item."""
    words = item.replace("`", " ").replace("*", " ").split()
    path = words[0].rstrip(".,;:") if words else ""
    return None if not path or path == NOTHING else path


def amendment_files(text: str, run: str | None) -> list[str]:
    """The files D-023's redo amendments for `run` name (every run's, for None), every such line together, sorted.
    The form is one line, `D-023 redo amendment: run <run>; changed <path>, <path>` (module doc), read as house style
    writes it: each item (commas, or "and") is its first word, less backticks, asterisks and a closing period,
    comma, semicolon or colon, so `path`, a closing period, a bold line or a note after a path read as the bare form
    does, and "nothing" names no file. A bold head (**D-023 redo amendment:**) is not the form and is not read."""
    out: set[str] = set()
    for m in AMENDMENT.finditer(text):
        if run is None or m.group(1) == run:
            out |= {p for p in map(_amended, AMENDED_ITEMS.split(m.group(2))) if p}
    return sorted(out)


def _path_summary(rec: dict) -> str:
    parts = [f"{rec['matched']} of {rec['files']} listed files have their blob"]
    if rec["n_changed"]:
        parts.append(f"{rec['n_changed']} differ ({', '.join(c['path'] for c in rec['changed'])})")
    if rec["n_missing"]:
        parts.append(f"{rec['n_missing']} missing ({', '.join(rec['missing'])})")
    if rec["n_malformed"]:
        parts.append(f"{rec['n_malformed']} malformed lines")
    return "; ".join(parts)


# --------------------------------------------------------------------------------------------- the conditions
@dataclass(frozen=True)
class Condition:
    group: str            # one of GROUPS
    name: str             # the field it reads, or what the laptop reproduces
    needs: str            # what the list asks, as text
    shows: str            # what the files show, as text
    ok: bool


class _List:
    """The conditions as they are added, in order."""

    def __init__(self) -> None:
        self.items: list[Condition] = []

    def add(self, group: str, name: str, needs: str, ok, shows: str) -> None:
        self.items.append(Condition(group, name, needs, shows, bool(ok)))

    def value(self, group: str, doc, prefix: str, path: str, needs: str, ok_fn) -> None:
        v = field(doc, path)
        self.add(group, prefix + path, needs, v is not MISSING and ok_fn(v), shown(v))


def _stream_names() -> list[str]:
    return [name for name, _ in S.STREAMS]


def _attempt(fn, why: str | None) -> tuple[bool, str]:
    """fn() -> (ok, shows), or (False, why) when the commit it reads could not be read, or what git said."""
    if why is not None:
        return False, why
    try:
        return fn()
    except Step0Error as err:
        return False, err.problems[0]


def _git_conditions(run: Run, c: _List, repo: Path, parsed: Mapping[str, tuple[str, str] | None]) -> dict:
    """The conditions that read the repo with git (module doc), each added to its group; what they read."""
    gen = run.generate
    read: dict = {"repo": str(repo)}
    gcommit = (parsed.get("generate.host.stamp") or (None, None))[0]
    full = why = None
    try:
        if gcommit is None:
            raise Step0Error(["generate.json's host.stamp names no commit"])
        full = read["generate_commit"] = resolve(repo, gcommit)
    except Step0Error as err:
        why = err.problems[0]

    def decisions() -> tuple[bool, str]:
        dec = blob_at(repo, full, S.DECISIONS)
        holds, sha = S.MARK in R.find_marks(dec.decode("utf-8", errors="replace")), sha256_bytes(dec)
        want = field(gen, "registration.decisions_sha256")
        return holds and sha == want, (f"at {gcommit}: {'holds' if holds else 'does not hold'} MARK; sha256 "
                                       f"{short(sha)} " + ("is generate's" if sha == want else
                                                           f"is not generate's {shown(want)}"))
    c.add("Registered first", "docs/DECISIONS.md at the generate commit",
          "it holds MARK, and its sha256 is registration.decisions_sha256 (git)", *_attempt(decisions, why))

    body: list[str] = []

    def path_file() -> tuple[bool, str]:
        raw = blob_at(repo, full, S.PATH_FILE)
        try:
            body.append(raw.decode("ascii"))
        except UnicodeDecodeError as err:
            raise Step0Error([f"{S.PATH_FILE} at {gcommit} is not ASCII"]) from err
        sha, want = sha256_bytes(raw), field(gen, "path_file.sha256")
        rec = read["path"] = P.check_commit(repo, full, body[0])
        return rec["ok"] and sha == want, (f"at {gcommit}: {_path_summary(rec)}; the path file's sha256 {short(sha)} "
                                           + ("is generate's" if sha == want else f"is not generate's {shown(want)}"))
    c.add("One stamp, one path", "the path at the generate commit",
          f"{S.PATH_FILE} is the one generate checked (path_file.sha256), every file it lists at its blob (git)",
          *_attempt(path_file, why))

    if run.redo is not None:
        later = sorted({p[0] for k, p in parsed.items() if k != "generate.host.stamp" and p is not None
                        and p[0] != gcommit})
        oks, parts = [], []
        for commit in later:
            def at_redo(commit: str = commit) -> tuple[bool, str]:
                if not body:
                    raise Step0Error([f"no path file at the generate commit {gcommit} to compare with"])
                lfull = resolve(repo, commit)
                named = amendment_files(blob_at(repo, lfull, S.DECISIONS).decode("utf-8", errors="replace"),
                                        run.run_name)
                rec = P.check_commit(repo, lfull, body[0], excepted=named)
                read.setdefault("redo_commits", {})[commit] = {"commit": lfull, "amendment": named, "path": rec}
                changed = [x["path"] for x in rec["changed"]] + rec["missing"]
                listed = {p for _, p in P.parse(body[0])[0]}
                unlisted = [p for p in named if p not in listed]       # shown, so a typo is seen: it excepts nothing
                return rec["ok"], (f"at {commit}: {_path_summary(rec)}; D-023's amendment for {run.run_name} names "
                                   + (", ".join(named) if named else "no file")
                                   + (f"; named but not on the path: {', '.join(unlisted)}" if unlisted else "")
                                   + ("" if rec["ok"] else
                                      f"; not named: {', '.join(p for p in changed if p not in named)}"))
            ok, shows = _attempt(at_redo, None)
            oks.append(ok)
            parts.append(shows)
        c.add("One stamp, one path", "the path at the redo commit",
              "every file the generate commit's path file lists at its blob, but the files D-023's amendment for the "
              "run names (git)", all(oks), "; ".join(parts) or f"the redo ran at the generate commit {gcommit}")

    code = run.render_code or {}
    rc = stamp_fields(code.get("stamp"))

    def source() -> tuple[bool, str]:
        if rc is None:
            raise Step0Error([f"render_code.stamp {shown(code.get('stamp', MISSING))} names no commit"])
        rfull = resolve(repo, rc[0])
        got = source_sha8_at(repo, rfull)
        read["source_sha8"] = {"commit": rfull, "source_sha8": got}
        return got == code.get("source_sha8"), (f"at {rc[0]}: {got}; render_code.source_sha8 "
                                                f"{shown(code.get('source_sha8', MISSING))}")
    c.add("Reproduced on the laptop", "render_code.source_sha8",
          "features.source_sha8 over the render stamp's commit (git)", *_attempt(source, None))
    return read


def _sets_conditions(run: Run, c: _List) -> None:
    by_i = run.scored_rows()
    if run.sets is None:
        for name in S.SETS:
            c.add("Sets", f"sets.{name} streams", f"every {name} row stream {S.SET_STREAM[name]}", False,
                  "sets.json is missing")
        c.add("Sets", "sets.pool taken", "at least one pool pick", False, "sets.json is missing")
        c.add("Sets", "sets.pool held-out", "no pool pick of a held-out clip's text or clip (Q5)", False,
              "sets.json is missing")
        return
    for name in S.SETS:
        want, bad, n = S.SET_STREAM[name], [], 0
        for e in run.entries(name):
            if e.get("stream") != want:
                bad.append(f"{e['brief_id']} (entry stream {shown(e.get('stream', MISSING))})")
            for t in e["taken"]:
                n += 1
                s = by_i.get(t.get("i"))
                if s is None or s.get("stream") != want:
                    bad.append(f"{e['brief_id']} k{t.get('k')} (row i={t.get('i')} is "
                               f"{'no scored row' if s is None else 'stream ' + shown(s.get('stream', MISSING))})")
        c.add("Sets", f"sets.{name} streams", f"every {name} row stream {want}", not bad,
              f"{len(run.entries(name))} briefs, {n} taken rows, all {want}" if not bad else
              f"{len(bad)} not {want}: {'; '.join(bad[:3])}")
    picks = [(e, t) for e in run.entries("pool") for t in e["taken"]]
    c.add("Sets", "sets.pool taken", "at least one pool pick", bool(picks),
          f"{len(picks)} picks of {len(run.entries('pool'))} briefs")
    held = [(e, t) for e in run.entries("heldout") for t in e["taken"]]
    if S.POOL_EXCLUDES == "text":
        texts = {t.get("canon_sha256") for _, t in held}
        flags = {(d.get("brief_id"), d.get("k")): d.get("heldout_text") for d in run.sets["heldout_text"]}
        bad = [f"{e['brief_id']} k{t.get('k')}" for e, t in picks
               if t.get("canon_sha256") in texts or flags.get((e["brief_id"], t.get("k"))) is not False]
        c.add("Sets", "sets.pool held-out", "no pool pick has heldout_text true (Q5(a): text)", not bad,
              f"{len(bad)} of {len(picks)} picks" + (f" ({', '.join(bad[:3])})" if bad else ""))
    else:
        clips = {(e.get("cell"), e.get("bpm"), e.get("tonic"), t.get("canon_sha256")) for e, t in held}
        bad = [f"{e['brief_id']} k{t.get('k')}" for e, t in picks
               if (e.get("cell"), e.get("bpm"), e.get("tonic"), t.get("canon_sha256")) in clips]
        c.add("Sets", "sets.pool held-out", "no pool pick is a held-out clip (Q5(b), (c): clip)", not bad,
              f"{len(bad)} of {len(picks)} picks" + (f" ({', '.join(bad[:3])})" if bad else ""))


def _difference(got: Mapping, want: Mapping) -> str:
    """Where sets.json first differs from the reproduction, in words."""
    for key in want:
        if _dumps(got.get(key, None)) == _dumps(want[key]):
            continue
        a, b = got.get(key), want[key]
        if isinstance(a, list) and isinstance(b, list):
            for j, (x, y) in enumerate(zip(a, b)):
                if _dumps(x) != _dumps(y):
                    where = x.get("brief_id", j) if isinstance(x, dict) else j
                    keys = [k for k in (y if isinstance(y, dict) else {}) if _dumps((x or {}).get(k)) != _dumps(y[k])]
                    return f"`{key}` differs first at {where}" + (f" ({', '.join(keys)})" if keys else "")
            return f"`{key}` holds {len(a)} entries, the reproduction {len(b)}"
        return f"`{key}` differs ({shown(a if a is not None else MISSING)[:80]})"
    extra = sorted(set(got) - set(want))
    return f"sets.json has keys the reproduction has not: {', '.join(extra)}" if extra else "equal"


def _reproduced(run: Run, c: _List, reqs: Sequence[R.Request], streams) -> None:
    if run.sets is None:
        c.add("Reproduced on the laptop", "sets.json = step0.select", "sets.json equals select over the run's files",
              False, "sets.json is missing")
    else:
        try:
            want = X.sets_json(run.scored, run.records, streams=streams, render_code=run.render_code)
        except Step0Error as err:
            c.add("Reproduced on the laptop", "sets.json = step0.select",
                  "sets.json equals select over the run's files", False,
                  f"select refuses the files: {err.problems[0]}"
                  + (f" (and {len(err.problems) - 1} more)" if len(err.problems) > 1 else ""))
        else:
            same = _dumps(run.sets) == _dumps(want)
            briefs = ", ".join(f"{n} {len(want[n])} briefs" for n in S.SETS)
            c.add("Reproduced on the laptop", "sets.json = step0.select",
                  "sets.json equals select over the run's files", same,
                  f"equal: {briefs}, {len(want['heldout_text'])} train draws' heldout_text" if same else
                  _difference(run.sets, want))
    if run.clips is None:
        c.add("Reproduced on the laptop", "clip ids", "every clip id is the formula over its recorded inputs", False,
              "clips.json is missing")
        return
    by_req = {r.i: r for r in reqs}
    by_i = run.scored_rows()
    pal = run.clips.get("palette") if isinstance(run.clips.get("palette"), dict) else {}
    label = f"{pal.get('id')}@{str(pal.get('sha256'))[:8]}"
    bad, cfgs, n = [], Counter(), 0
    for rec in run.records:
        if rec.get("clip_id") is None and rec.get("ok") is not True:
            continue
        n += 1
        req, s, canon = by_req.get(rec["i"]), by_i.get(rec["i"]), rec.get("canon_sha256")
        cfg = rec.get("render_cfg_sha8")
        cfgs[shown(cfg if cfg is not None else MISSING)] += 1
        if req is None or s is None:
            bad.append(f"i={rec['i']}: not a request with a scored row")
        elif not isinstance(canon, str) or canon != s.get("canon_sha256"):
            bad.append(f"i={rec['i']}: canon_sha256 {short(canon)} is not score's {short(s.get('canon_sha256'))}")
        elif not isinstance(cfg, str) or rec.get("clip_id") != clip_id(canon, req.brief, label, cfg):
            bad.append(f"i={rec['i']}: clip id {shown(rec.get('clip_id', MISSING))}, the formula gives "
                       f"{clip_id(canon, req.brief, label, str(cfg))}")
    if len(cfgs) > 1:
        bad.append(f"{len(cfgs)} render_cfg_sha8 values in one run ({', '.join(sorted(cfgs))})")
    c.add("Reproduced on the laptop", "clip ids",
          "every clip id is the formula over its record's canon_sha256 (score's), the brief, the palette label and "
          "the run's one render_cfg_sha8", not bad,
          f"{n} clip ids equal the formula ({label}, render_cfg_sha8 {', '.join(sorted(cfgs)) or '-'})" if not bad
          else f"{len(bad)} wrong: {'; '.join(bad[:3])}")


def conditions(run: Run, *, streams=None, repo: Path | None = None) -> tuple[list[Condition], dict]:
    """Every condition of the validity list (module doc), in its groups' order; and what the check read beside the
    files: the requests block and the git records."""
    repo = Path(ROOT if repo is None else repo)
    streams = R.load() if streams is None else streams
    reqs = R.requests(streams=streams)
    want_n, names = len(reqs), _stream_names()
    per_stream = Counter(r.stream for r in reqs)
    block = requests_block(run.rows, reqs)
    gen, s0, c = run.generate, run.step0, _List()

    c.value("Complete", gen, "generate.", "stopped", "null", lambda v: v is None)
    c.value("Complete", gen, "generate.", "partial", "false", lambda v: v is False)
    c.value("Complete", s0, "", "n", str(want_n), lambda v: _int(v, want_n))
    c.value("Complete", s0, "", "requested", str(want_n), lambda v: _int(v, want_n))
    c.value("Complete", gen, "generate.", "requested", str(want_n), lambda v: _int(v, want_n))
    c.value("Complete", s0, "", "unreadable_lines", "0", lambda v: _int(v, 0))
    for kind in ("missing", "duplicate", "foreign"):
        first = block[f"first_{kind}"]
        c.add("Complete", f"requests.{kind}", "0", block[kind] == 0,
              str(block[kind]) + (f" (first i {', '.join(map(str, first))})" if first else ""))

    first = block["first_seed_mismatch"]
    c.add("Seeded", "requests.seed_mismatch", "0: draw_seed and sp_seed are brief_streams.draw_seed(seed, index, k)",
          block["seed_mismatch"] == 0,
          str(block["seed_mismatch"]) + (f" (first i {', '.join(map(str, first))})" if first else ""))
    c.value("Seeded", gen, "generate.", "probe.passed", "true", lambda v: v is True)
    first = block["first_prompt_tokens_mismatch"]
    cells = "; ".join(f"{cell} {'/'.join(sorted(t))}" for cell, t in block["prompt_tokens"].items())
    c.add("Seeded", "requests.prompt_tokens_mismatch",
          f"0: each cell's rows at {S.GATE_RUN}'s prompt_tokens (1,325-1,411)", block["prompt_tokens_mismatch"] == 0,
          str(block["prompt_tokens_mismatch"]) + (f" (first i {', '.join(map(str, first))})" if first else "")
          + (f"; {cells}" if block["prompt_tokens_mismatch"] else ""))

    ar = "As registered"
    c.value(ar, gen, "generate.", "model.revision", S.MODEL_REVISION, lambda v: _str(v, S.MODEL_REVISION))
    c.value(ar, gen, "generate.", "streams_file.sha256", S.STREAMS_SHA256, lambda v: _str(v, S.STREAMS_SHA256))
    c.value(ar, gen, "generate.", "streams_file.fresh", "true", lambda v: v is True)
    marked = R.stream_counts(reqs)
    c.value(ar, gen, "generate.", "streams", f"the mark's: {shown(marked)}", lambda v: _same_json(v, marked))
    first = block["first_brief_mismatch"]
    c.add(ar, "requests.brief_mismatch", "0: each row's brief is sample_brief(seed, index)",
          block["brief_mismatch"] == 0,
          str(block["brief_mismatch"]) + (f" (first i {', '.join(map(str, first))})" if first else ""))
    by = {name: field(s0, f"by_stream.{name}") for name in names}          # summarize's blocks, per stream
    for name in names:
        p, sb, n_s = f"by_stream.{name}.", by[name], per_stream[name]
        c.value(ar, sb, p, "prompt.version", S.PROMPT_VERSION, lambda v: _str(v, S.PROMPT_VERSION))
        c.value(ar, sb, p, "prompt.template_sha256", S.TEMPLATE_SHA256, lambda v: _str(v, S.TEMPLATE_SHA256))
        c.value(ar, sb, p, "prompt.template_matches_generate", "true", lambda v: v is True)
        c.value(ar, sb, p, "prompt.rows_matching", str(n_s), lambda v, n_s=n_s: _int(v, n_s))
        c.value(ar, sb, p, "prompt.example_mismatch", "0", lambda v: _int(v, 0))
    c.value(ar, gen, "generate.", "palette.sha256", S.PALETTE_SHA256, lambda v: _str(v, S.PALETTE_SHA256))
    for name in names:
        p, sb = f"by_stream.{name}.", by[name]
        c.value(ar, sb, p, "palette.sha256", S.PALETTE_SHA256, lambda v: _str(v, S.PALETTE_SHA256))
        c.value(ar, sb, p, "palette.matches_generate", "true", lambda v: v is True)
    c.value(ar, run.clips or {}, "clips.", "palette.sha256", S.PALETTE_SHA256, lambda v: _str(v, S.PALETTE_SHA256))
    c.value(ar, gen, "generate.", "sampling", f"{S.V0_RUN}'s: {shown(dict(S.V0_SAMPLING))}",
            lambda v: _same_json(v, dict(S.V0_SAMPLING)))

    c.value("Registered first", gen, "generate.", "registration.found", "true", lambda v: v is True)
    c.value("Registered first", gen, "generate.", "registration.mark", S.MARK, lambda v: _str(v, S.MARK))

    for name in names:
        p, sb = f"by_stream.{name}.", by[name]
        c.value("Constrained", sb, p, "constrained.enabled", "true", lambda v: v is True)
        c.value("Constrained", sb, p, "constrained.regex_sha256", S.REGEX_SHA256, lambda v: _str(v, S.REGEX_SHA256))
        c.value("Constrained", sb, p, "constrained.in_language.regex_sha256", S.REGEX_SHA256,
                lambda v: _str(v, S.REGEX_SHA256))
        c.value("Constrained", sb, p, "constrained.in_language.matches_generate", "true", lambda v: v is True)
        c.value("Constrained", sb, p, "constrained.in_language.stop.out_of_language", "0", lambda v: _int(v, 0))
        c.value("Constrained", sb, p, "constrained.in_language.other.n", "0", lambda v: _int(v, 0))
    for name in names:
        for g in REGEX_GATES:
            c.value("What the regex guarantees", by[name], f"by_stream.{name}.", f"failures.{g}", "0",
                    lambda v: _int(v, 0))

    stamps = {"generate.host.stamp": field(gen, "host.stamp"), "score_stamp": field(s0, "score_stamp"),
              "render_stamp": field(s0, "render_stamp")}
    parsed = {k: stamp_fields(v) for k, v in stamps.items()}
    every = "; ".join(f"{k} {shown(v)}" for k, v in stamps.items())
    if run.redo is None:
        one = len(set(parsed.values())) == 1 and None not in parsed.values()
        same = len({shown(v) for v in stamps.values()}) == 1
        c.add("One stamp, one path", STAMP_NAME, "the same commit=… dirty=0 in all three",
              one and next(iter(parsed.values()))[1] == "0",
              f"{shown(stamps['generate.host.stamp'])} (all three)" if same and one else every)
    else:
        score_only = run.files["step0.json"].parent.parent == run.redo and s0.get("render") is None
        c.add("One stamp, one path", REDO_STAMP_NAME,
              "dirty=0 in each: generate's from the run, score's and render's from the redo (§4.2)",
              all(p is not None and p[1] == "0" for p in parsed.values()),
              every + ("; the redo's step0.json has no render block: a score redo without its render redo, which "
                       "follows it in the same redo (render --from OUT --run REDO, spec section 4.2)"
                       if score_only else ""))
    c.value("One stamp, one path", gen, "generate.", "path_file.blobs_ok", "true", lambda v: v is True)

    blk = s0.get("render")
    for key in ("errors_final", "skipped", "other_gates"):
        c.value("Held-out render", s0, "", f"render.{key}", "0", lambda v: _int(v, 0))
    und = [e["brief_id"] for e in run.entries("heldout") if e.get("undetermined") is not False]
    if run.sets is None:
        shows = "sets.json is missing"
    else:
        shows = f"{len(und)} of {len(run.entries('heldout'))}" + (f" ({', '.join(und[:5])})" if und else "")
    c.add("Held-out render", "held-out briefs undetermined", "0 (sets.json heldout)", run.sets is not None and not und,
          shows + ("" if blk is not None else "; step0.json has no render block"))

    _sets_conditions(run, c)
    _reproduced(run, c, reqs, streams)
    git = _git_conditions(run, c, repo, parsed)
    order = {g: j for j, g in enumerate(GROUPS)}
    items = sorted(c.items, key=lambda x: order[x.group])          # stable: each group keeps its own order
    return items, {"requests": block, "git": git}


# --------------------------------------------------------------------------------------------- item 8's readouts
def copy_forms(text: str, brief: Brief, palette, finish_reason: str | None = "stop") -> tuple[str, str]:
    """(junk's copy form, the K/D form) of a loop: its canonical text without `sax:` (junk.score_row's), and that text
    with the keys instrument and the kit written K and D (DRAFTS.md §5, lofi_rl.drafts.check.copy_form).
    Step0Error for a text that does not pass the gates (no taken draw is one)."""
    from lofi_rl.dsl import canon, parse
    res = parse(text, vocab=palette.vocab(), finish_reason=finish_reason, mode=brief.mode)
    if not res.gate.ok:
        raise Step0Error([f"a held-out clip's text fails {res.gate.code}: {res.gate.detail[:120]}"])
    loop = replace(res.ast, sax=None)
    return canon(loop), canon(replace(loop, keys=replace(loop.keys, instrument="K"), kit="D"))


def tuple_readout(clips: Sequence[tuple[int, str, Mapping]], brief: Brief, palette) -> dict:
    """D-021 item 8's readouts of one held-out tuple, reported beside D-008 and never gated. `clips` is (k, text,
    scored row) of each clip in k order. Per pair of clips, in k order: their 5-gram Jaccard (junk.grams,
    junk.jaccard) in junk's copy form (`j5`) and in the K/D form (`j5_kd`), and whether their A lines, or their B
    lines, differ (`differ`: either); and how many of the tuple's clips carry the shown example's A, B, and both
    lines verbatim (score's `verbatim`)."""
    from lofi_rl.rl import junk as J
    forms = [copy_forms(text, brief, palette, row.get("finish_reason")) for _, text, row in clips]
    grams = [(J.grams(a), J.grams(b)) for a, b in forms]
    lines = [J.head_lines(a) for a, _ in forms]
    pairs = []
    for x, y in combinations(range(len(clips)), 2):
        da, db = lines[x].get("A") != lines[y].get("A"), lines[x].get("B") != lines[y].get("B")
        pairs.append({"k": [clips[x][0], clips[y][0]], "j5": round(J.jaccard(grams[x][0], grams[y][0]), 4),
                      "j5_kd": round(J.jaccard(grams[x][1], grams[y][1]), 4), "A": da, "B": db,
                      "differ": da or db})
    verb = [row.get("verbatim") or {} for _, _, row in clips]
    return {"verbatim": {"A": sum(1 for v in verb if v.get("A") is True),
                         "B": sum(1 for v in verb if v.get("B") is True),
                         "AB": sum(1 for v in verb if v.get("A") is True and v.get("B") is True)},
            "pairs": pairs}


def _brief_of(bid: str) -> Brief:
    m = _ID.fullmatch(bid)
    if not m:
        raise Step0Error([f"{bid!r} is not a brief id"])
    return sample_brief(int(m.group(1)), int(m.group(2)))


def item8(run: Run, palette=None) -> dict:
    """item 8's readouts (tuple_readout) on every held-out tuple of sets.json, a brief with its m clips, in stream
    order, and their totals."""
    if palette is None:
        from lofi_rl.palette import ROOT as CODE, load_palette
        palette = load_palette(CODE / S.PALETTE, verify=False)
    rows, scored = run.text_rows(), run.scored_rows()
    tuples = []
    for e in run.entries("heldout"):
        if len(e["taken"]) != S.M["heldout"]:
            continue
        clips = []
        for t in e["taken"]:
            row, s = rows.get(t.get("i")), scored.get(t.get("i"))
            if row is None or s is None or not isinstance(row.get("text"), str):
                raise Step0Error([f"{e['brief_id']} k{t.get('k')}: no completions.jsonl text for row {t.get('i')}"])
            clips.append((t["k"], row["text"], s))
        tuples.append({"brief_id": e["brief_id"], **tuple_readout(clips, _brief_of(e["brief_id"]), palette)})
    pairs = [p for t in tuples for p in t["pairs"]]
    m = S.M["heldout"]
    return {"tuples": tuples, "summary": {
        "tuples": len(tuples), "pairs": len(pairs), "differ": sum(1 for p in pairs if p["differ"]),
        "j5_mean": round(sum(p["j5"] for p in pairs) / len(pairs), 4) if pairs else None,
        "j5_kd_mean": round(sum(p["j5_kd"] for p in pairs) / len(pairs), 4) if pairs else None,
        "all_carry": {h: sum(1 for t in tuples if t["verbatim"][h] == m) for h in ("A", "B", "AB")}}}


# --------------------------------------------------------------------------------------------- shown, never judged
def rescore(run: Run) -> dict:
    """junk.score_row over completions.jsonl with this checkout's code (and CopyCheck v2 on the frozen palette), plus
    stream, brief_id, k and text_sha256, against scored.jsonl: the rows and fields that differ. Shown, never judged."""
    from lofi_rl.palette import ROOT as CODE, load_palette
    from lofi_rl.rl import junk as J
    pal = load_palette(CODE / S.PALETTE, verify=False)
    copy = J.CopyCheck(S.PROMPT_VERSION, pal)
    old = run.scored_rows()
    diffs, fields, n = [], Counter(), 0
    for i, row in sorted(run.text_rows().items()):
        n += 1
        try:
            new = {**J.score_row(row, pal, copy), "stream": row.get("stream"), "brief_id": row.get("brief_id"),
                   "k": row.get("k"), "text_sha256": J.sha256_text(row["text"])}
        except Exception as err:  # noqa: BLE001: a readout, never a reason to stop the check
            diffs.append({"i": i, "fields": [f"score_row raised {type(err).__name__}: {err}"[:200]]})
            continue
        was = old.get(i, {})
        keys = sorted(k for k in set(new) | set(was)
                      if (k in new) != (k in was) or _dumps(new.get(k)) != _dumps(was.get(k)))
        if keys:
            fields.update(keys)
            diffs.append({"i": i, "fields": keys})
    return {"rows": n, "differ": len(diffs), "fields": dict(sorted(fields.items())), "first": diffs[:MAX_LISTED],
            "knob_errors": dict(copy.knob_errors)}


def shown_block(run: Run) -> dict:
    """What the check shows and never judges (module doc), from the files."""
    g, s0 = run.generate, run.step0
    c = g.get("constrained") if isinstance(g.get("constrained"), dict) else {}
    timing = g.get("timing") if isinstance(g.get("timing"), dict) else {}
    replay = g.get("replay") if isinstance(g.get("replay"), dict) else None
    out = {"backend": c.get("backend"), "per_request_seed": g.get("per_request_seed"),
           "engine": {k: (g.get("engine") or {}).get(k) for k in ("vllm", "seed", "dtype", "enforce_eager")}
           if isinstance(g.get("engine"), dict) else None,
           "probe": g.get("probe"), "replay": replay,
           "replay_file": None if run.replay is None else {
               "rows": len(run.replay), "same": sum(1 for r in run.replay if r.get("same") is True)},
           "timing": {k: timing.get(k) for k in TIMINGS},
           "host": {k: run.host.get(k) for k in ("node", "job", "gpu")},
           "render": s0.get("render"), "sets": s0.get("sets")}
    try:
        out["item8"] = item8(run)["summary"] if run.sets is not None else None
    except Step0Error as err:
        out["item8"] = {"error": err.problems[0]}
    return out


# --------------------------------------------------------------------------------------------- the report
def run_record(run: Run) -> dict:
    """What the run is: its folders, job, node and GPU, its stamps and the files read."""
    return {"run": run.name, "job": run.job, "node": run.host.get("node"), "gpu": run.host.get("gpu"),
            "redo": None if run.redo is None else run.redo.resolve().name,
            "stamp": {"generate": run.host.get("stamp"), "score": run.step0.get("score_stamp"),
                      "render": run.step0.get("render_stamp")},
            "render_code": run.render_code,
            "files": {name: ("redo" if run.redo is not None and p.parent.parent == run.redo else "run")
                      for name, p in run.files.items() if p.exists()}}


def evaluate(run: Run, *, repo: Path | None = None, streams=None, rescore_rows: bool = False) -> dict:
    """The check's whole report (module doc)."""
    conds, read = conditions(run, streams=streams, repo=repo)
    failed = [c.name for c in conds if not c.ok]
    valid = not failed
    return {"schema": SCHEMA, **run_record(run), "rule": S.RULE, "mark": S.MARK,
            "validity": {"valid": valid, "held": len(conds) - len(failed), "of": len(conds), "failed": failed,
                         "conditions": [asdict(c) for c in conds]},
            "requests": read["requests"], "git": read["git"], "shown": shown_block(run),
            "rescore": rescore(run) if rescore_rows else None,
            "verdict": "valid" if valid else "invalid", "exit": 0 if valid else 3}


def _counts(block: Mapping | None, keys: Sequence[str]) -> str:
    if not isinstance(block, dict):
        return "absent"
    return ", ".join(f"{k} {shown(block.get(k, MISSING))}" for k in keys)


def format_text(doc: dict) -> str:
    v, rq, sh = doc["validity"], doc["requests"], doc["shown"]
    lines = [f"step0 check: {doc['run']} (job {doc['job']}, {doc['node']}, {doc['gpu']})"
             + (f", redo {doc['redo']} (from it: {', '.join(k for k, w in doc['files'].items() if w == 'redo')})"
                if doc["redo"] else ""),
             f"  stamps: generate {doc['stamp']['generate']}; score {doc['stamp']['score']}; render "
             f"{doc['stamp']['render']}",
             f"  requests: {rq['requested']} registered ({S.MARK}); {rq['rows']} rows written",
             "validity (D-023's list, docs/design/step0_samples.md section 4.1):"]
    group = None
    width = max(len(x["name"]) for x in v["conditions"])
    for x in v["conditions"]:
        if x["group"] != group:
            group = x["group"]
            lines.append(f"  {group}:")
        lines.append(f"    {'ok  ' if x['ok'] else 'FAIL'}  {x['name']:<{width}}  {x['shows']}"
                     + ("" if x["ok"] else f"   (needs {x['needs']})"))
    lines.append(f"validity: {'VALID' if v['valid'] else 'INVALID'}, {v['held']} of {v['of']} conditions hold"
                 + (f"; failing: {', '.join(v['failed'])}" if v["failed"] else ""))
    b = sh.get("backend") if isinstance(sh.get("backend"), dict) else {}
    pr = sh.get("probe") if isinstance(sh.get("probe"), dict) else {}
    rp = sh.get("replay") if isinstance(sh.get("replay"), dict) else {}
    t, h = sh["timing"], sh["host"]
    lines += ["shown, never judged:",
              f"  backend {shown(b.get('backend', MISSING))} (pin {shown(b.get('pin_accepted', MISSING))}, applied "
              f"{shown(b.get('applied', MISSING))}); engine {shown(sh.get('engine') or MISSING)}",
              f"  probe: seeds {shown(pr.get('seeds', MISSING))}, {shown(pr.get('cell', MISSING))}, "
              f"{shown(pr.get('n', MISSING))} requests a seed, {shown(pr.get('distinct', MISSING))} distinct text(s)",
              f"  replay: {_counts(rp, ('n', 'same', 'differ', 'stopped'))}; replay.jsonl "
              f"{_counts(sh.get('replay_file'), ('rows', 'same'))}",
              "  timing: " + ", ".join(f"{k} {shown(t.get(k))}" for k in TIMINGS) + f"; GPU {h.get('gpu')}",
              f"  render: {_counts(sh.get('render'), ('attempted', 'rendered', 'G7', 'errors_retried', 'rounds'))}"]
    sets = sh.get("sets") if isinstance(sh.get("sets"), dict) else {}
    for name in S.SETS:
        lines.append(f"  {name}: {_counts(sets.get(name), ('briefs', 'taken', 'short', 'near_copy', 'repaired'))}, "
                     f"verbatim {shown((sets.get(name) or {}).get('verbatim', MISSING))}, same clip as another "
                     f"brief's {shown((sets.get(name) or {}).get('same_clip', MISSING))}")
    ht = sets.get("heldout_text") if isinstance(sets.get("heldout_text"), dict) else {}
    lines.append("  heldout_text: " + (", ".join(f"{cell} {_counts(x, ('draws', 'refb', 'pool'))}"
                                                 for cell, x in ht.items() if (x or {}).get("draws"))
                                       or "no train draw carries a held-out clip's text"))
    i8 = sh.get("item8")
    if isinstance(i8, dict) and "tuples" in i8:
        lines.append(f"  item 8: {i8['tuples']} tuples, {i8['pairs']} pairs; mean J5 {shown(i8['j5_mean'])} (junk's "
                     f"form), {shown(i8['j5_kd_mean'])} (K/D form); A or B differ in {i8['differ']} pairs; all "
                     f"{S.M['heldout']} carry A in {i8['all_carry']['A']} tuples, B in {i8['all_carry']['B']}, both "
                     f"in {i8['all_carry']['AB']}")
    else:
        lines.append(f"  item 8: {shown(i8 if i8 is not None else 'sets.json is absent')}")
    rs = doc.get("rescore")
    if rs is None:
        lines.append("  rescore: not run (--rescore)")
    else:
        lines.append(f"  rescore with this checkout: {rs['differ']} of {rs['rows']} rows differ from scored.jsonl"
                     + (f" (fields {shown(rs['fields'])}; first i {', '.join(str(d['i']) for d in rs['first'])})"
                        if rs["differ"] else ""))
    lines.append(f"STEP0_CHECK {doc['verdict']}")
    return "\n".join(lines) + "\n"


def main(a: argparse.Namespace) -> int:
    """The `check` subcommand (lofi_rl.step0.cli)."""
    try:
        doc = evaluate(load_run(a.run, a.redo), rescore_rows=a.rescore)
    except Step0Error as err:
        print(f"error: bad input, nothing decided:\n{err}", file=sys.stderr)
        print("STEP0_CHECK bad input")
        return 2
    print(json.dumps(doc, indent=1) if a.json else format_text(doc).rstrip("\n"))
    return doc["exit"]
