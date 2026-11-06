"""The Phase-1 gate on the frozen palette (docs/GATE1.md; CLAUDE.md §9; docs/DECISIONS.md D-020 as amended, D-021
"What follows" item 1, D-022 "Consequences").

    python -m lofi_rl.gate1 check RUN_DIR [--json]
    python -m lofi_rl.gate1 batch RUN_DIR --clips DIR [--inbox DIR] [--date YYYY-MM-DD]
    python -m lofi_rl.gate1 tally BATCH_DIR --votes FILE [--json OUT]
    python -m lofi_rl.gate1 gate --run RUN_DIR [--batch BATCH_DIR --votes FILE] --write gates/GATE_phase1.md

**The bar** (D-020's, unchanged, as D-021 restates it): `pass_not_copy.count` >= 300 of 500 in a valid
`lofi-junk-rate` run of prompt v2 under the grammar on `palettes/lofi_rain_v1.yaml`, and at least 15 of its 50 clips
"fine" by Shresth's ear (Fine, Good or Great: TASTE.md, Tiers), with his APPROVED line in gates/GATE_phase1.md.
Beside the bar the gate file reports three readouts that decide nothing (D-021): the verbatim A-or-B share among the
passing non-copies, the same share among the 50 clips, and his "fine" count split by whether a clip carries a
verbatim A or B line of the example its prompt showed.

The four steps, each its own module: `check` (the validity list, the bar's first half, the two verbatim readouts),
`batch` (his triage batch of the clips), `tally` (his standing votes), `gatefile` (gates/GATE_phase1.md).

This module holds what they share:
- the registered values, each written out as the decision states it (D-019, D-020, D-022). They are the gate's, not
  the code's: tests/test_gate1_check.py compares each with the code that makes it today (the frozen file, the
  grammar, prompt v2's template, smoke.QWEN, junk.SAMPLING), so a drift shows up as a failing test, never as a
  silently moved bar;
- `load_run`, the reader of a run's summary/ (junk.json, scored.jsonl, completions.jsonl and clips.json). It refuses,
  with every problem listed, files that are not one run's own: scored.jsonl's rows and counts against junk.json's and
  completions.jsonl's, and clips.json against the sample junk's render step draws from that scored.jsonl
  (`junk.sample_indices` over the passing loops that are not near-copies, D-020). That refusal is bad input (exit 2),
  never a validity verdict. clips.json may be absent where a step reads no clip (`need_clips=False`): validity reads
  junk.json alone, and the render step decides nothing (D-020).

Stdlib only (plus lofi_rl's own stdlib modules), so the policy env's import walk (jobs/env_policy.sbatch) imports it.
Sound claims: none.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from lofi_rl.palette import ROOT
from lofi_rl.rl import junk as J

# --------------------------------------------------------------------------------------------- the registered values
FROZEN_SHA256 = "09f80a644476abe4d14517d0ec4129d69e1a4bc91eb0935c9ffd152411872183"   # D-022: sha256 B, frozen 2026-11-04
REGEX_SHA256 = "5c081828ac50718edc4db82c2a855ea52210debb375280ba01e8442de60f8558"    # D-022: the frozen palette's regex
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"                          # D-019: Qwen3-4B-Instruct-2507
PROMPT_VERSION = "v2"                                                                # D-020 arm B; D-021 Settings
TEMPLATE_SHA256 = "127e3094fd5a2183fb952059c2221ad018bdef6d062f51a80b14367b8064466e"  # D-020: prompt v2's template
BRIEF_SEED = 2026                                                                    # D-019: brief.sample_briefs(2026, 500)
N = 500
V0_RUN = "runs/junk-13632676"
# D-020: `sampling` equals runs/junk-13632676's. Read from runs/junk-13632676/summary/junk.json `sampling` on the
# laptop, 2026-11-06, in that file's key order.
V0_SAMPLING = MappingProxyType({"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
                                "presence_penalty": 0.0, "frequency_penalty": 0.0, "max_tokens": 512})
PASS_NOT_COPY_BAR = 300          # D-020 as amended: 60 % of 500
CLIPS = 50                       # D-020: the 50 ear-check clips (junk.RENDER_K)
FINE_BAR = 15                    # D-021: at least 15 of 50 "fine" (CLAUDE.md §9: 30 %)

APPROVER = "Shresth Rana"
CATEGORY = "gate1"               # the hidden manifest's category, and the batch name's tag
GATE_FILE = ROOT / "gates" / "GATE_phase1.md"
TIERS = ("toss", "fine", "good", "great")          # ui.votes.TIERS: TASTE.md, Tiers
FINE_OR_BETTER = ("fine", "good", "great")         # "fine" means Fine or better (D-021)
HEARD_MIN_BARS = 1               # D-013 as amended (Shresth, 2026-10-31): heard once 1 bar has played, over every attempt
VERBATIM_LINES = ("A", "B")      # "a verbatim line": the A or B line of the example the row's prompt showed (D-021)


class Gate1Error(ValueError):
    """Bad input: nothing was decided or written. `problems` lists every one found."""

    def __init__(self, problems: Iterable[str]):
        self.problems = [str(p) for p in problems] or ["bad input"]
        super().__init__("\n".join(f"- {p}" for p in self.problems))


def short(sha: str | None, n: int = 8) -> str:
    """A sha as the gate files print it (`09f80a64…`); anything else as it is."""
    return f"{sha[:n]}…" if isinstance(sha, str) and len(sha) == 64 else str(sha)


def verbatim_line(row: dict) -> bool:
    """Whether a scored row's loop carries the shown example's A or B line verbatim (junk.verbatim)."""
    v = row.get("verbatim") or {}
    return any(v.get(h) is True for h in VERBATIM_LINES)


def _whole(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


# --------------------------------------------------------------------------------------------- reading a run
@dataclass(frozen=True)
class Run:
    """One junk-rate run's summary, read and cross-checked (`load_run`)."""

    dir: Path
    junk: dict                       # summary/junk.json
    scored: dict                     # i -> summary/scored.jsonl row
    rows: dict                       # i -> summary/completions.jsonl row
    clips: dict | None               # summary/clips.json, or None when it is absent (need_clips=False)
    sample: tuple                    # the render step's whole draw: junk.sample_indices(pool, k, seed); () without clips

    @property
    def name(self) -> str:
        return self.dir.resolve().name

    @property
    def host(self) -> dict:
        g = self.junk.get("generate")
        h = g.get("host") if isinstance(g, dict) else None
        return h if isinstance(h, dict) else {}

    @property
    def job(self) -> str | None:
        job = self.host.get("job")
        return None if job is None else str(job)

    @property
    def palette(self) -> dict:
        p = self.junk.get("palette")
        return p if isinstance(p, dict) else {}

    @property
    def palette_label(self) -> str:
        """`<id>@<sha8>`, as the drafts' renders label a palette (lofi_rl.drafts.render)."""
        return f"{self.palette.get('id')}@{str(self.palette.get('sha256'))[:8]}"

    @property
    def pool(self) -> list[int]:
        """The passing loops that are not near-copies, in index order: the ear check's pool (D-020)."""
        return sorted(i for i, s in self.scored.items() if s.get("ok") is True and s.get("near_copy") is False)

    @property
    def clip_records(self) -> list[dict]:
        return list((self.clips or {}).get("clips") or [])

    @property
    def rendered(self) -> list[dict]:
        """clips.json's records that rendered (a FLAC was written), in the sample's order."""
        return [r for r in self.clip_records if r.get("ok") is True]


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


def _read_rows(path: Path, problems: list[str], *, strict: bool) -> dict[int, dict] | None:
    """i -> row of a JSONL file (junk.read_jsonl). `strict`: an unreadable line is a problem (scored.jsonl is written
    whole by score); otherwise it is left to the validity list (`unreadable_lines` counts completions.jsonl's)."""
    try:
        rows, bad = J.read_jsonl(path)
    except FileNotFoundError:
        problems.append(f"{path} is missing")
        return None
    except (OSError, UnicodeDecodeError) as err:
        problems.append(f"{path} is not readable ({err})")
        return None
    if strict and bad:
        problems.append(f"{path} has {bad} unreadable line(s); score writes it whole")
    out: dict[int, dict] = {}
    dup: set[int] = set()
    for r in rows:
        if r["i"] in out:
            dup.add(r["i"])
        out.setdefault(r["i"], r)
    if dup:
        problems.append(f"{path} repeats sample index(es) {sorted(dup)[:10]}")
    return out


def _count(doc: dict, key: str) -> int | None:
    block = doc.get(key)
    v = block.get("count") if isinstance(block, dict) else None
    return v if _whole(v) else None


def load_run(run_dir: Path | str, *, need_clips: bool = True) -> Run:
    """The run's summary/ files, cross-checked (module doc). Gate1Error lists every problem; none is a verdict.
    `need_clips`: clips.json must be there (the batch); otherwise its absence leaves `clips` None (check, gate)."""
    d = Path(run_dir)
    summ = d / "summary"
    if not summ.is_dir():
        raise Gate1Error([f"{summ} is not a folder: a run folder holds summary/ (make pull brings it)"])
    problems: list[str] = []
    junk = _read_json(summ / "junk.json", problems)
    if junk is not None and junk.get("schema") != J.SCHEMA:
        problems.append(f"{summ / 'junk.json'}: schema {junk.get('schema')!r}, not {J.SCHEMA!r}")
        junk = None
    scored = _read_rows(summ / "scored.jsonl", problems, strict=True)
    rows = _read_rows(summ / "completions.jsonl", problems, strict=False)
    clips = _read_json(summ / "clips.json", problems, need=need_clips)
    if clips is not None and clips.get("schema") != J.CLIPS_SCHEMA:
        problems.append(f"{summ / 'clips.json'}: schema {clips.get('schema')!r}, not {J.CLIPS_SCHEMA!r}")
        clips = None
    if problems:
        raise Gate1Error(problems)

    n = junk.get("n")
    for key in ("pass", "pass_not_copy"):
        if _count(junk, key) is None:
            problems.append(f"junk.json has no whole-number `{key}.count`")
    # numbers compared as numbers here: the validity list holds `n` to the registered JSON type (check.conditions)
    if not (isinstance(n, (int, float)) and not isinstance(n, bool) and len(scored) == n):
        problems.append(f"scored.jsonl holds {len(scored)} rows, junk.json `n` is {n!r}: not one score")
    ok = [s for s in scored.values() if s.get("ok") is True]
    pnc = [s for s in ok if s.get("near_copy") is False]
    if _count(junk, "pass") is not None and len(ok) != _count(junk, "pass"):
        problems.append(f"scored.jsonl has {len(ok)} passing rows, junk.json `pass.count` is {_count(junk, 'pass')}")
    if _count(junk, "pass_not_copy") is not None and len(pnc) != _count(junk, "pass_not_copy"):
        problems.append(f"scored.jsonl has {len(pnc)} passing non-copies, junk.json `pass_not_copy.count` is "
                        f"{_count(junk, 'pass_not_copy')}")
    bad_copy = sorted(s["i"] for s in ok if not isinstance(s.get("near_copy"), bool))
    if bad_copy:
        problems.append(f"scored.jsonl's passing rows {bad_copy[:10]} have no true/false `near_copy`")
    bad_verbatim = sorted(s["i"] for s in ok if not isinstance(s.get("verbatim"), dict))
    if bad_verbatim:
        problems.append(f"scored.jsonl's passing rows {bad_verbatim[:10]} have no `verbatim` readout")
    missing = sorted(set(scored) - set(rows))
    if missing:
        problems.append(f"scored.jsonl rows {missing[:10]} have no completions.jsonl row: not one run")
    unscored = sorted(set(rows) - set(scored))
    if unscored:
        problems.append(f"completions.jsonl rows {unscored[:10]} are not in scored.jsonl: score did not read this "
                        "completions.jsonl")

    sample: tuple = ()
    if clips is not None:
        k, seed, recs = clips.get("k"), clips.get("seed"), clips.get("clips")
        if not (_whole(k) and _whole(seed)):
            problems.append(f"clips.json: `k` {k!r} and `seed` {seed!r} must be whole numbers")
            k = None
        if not isinstance(recs, list) or not all(isinstance(r, dict) and _whole(r.get("i")) for r in recs):
            problems.append("clips.json: `clips` must be a list of records, each with its sample index `i`")
            recs = None
        cp = (clips.get("palette") or {}).get("sha256") if isinstance(clips.get("palette"), dict) else None
        jp = (junk.get("palette") or {}).get("sha256") if isinstance(junk.get("palette"), dict) else None
        if cp != jp:
            problems.append(f"clips.json was rendered with palette {short(cp)}, junk.json scored with {short(jp)}")
        if k is not None and recs is not None:
            pool = [i for i in sorted(scored) if scored[i].get("ok") is True and scored[i].get("near_copy") is False
                    and i in rows]
            sample = tuple(J.sample_indices(pool, k, seed))
            got = [r["i"] for r in recs]
            if got != list(sample[:len(got)]):
                problems.append(f"clips.json's {len(got)} clips are not junk's sample of {k} (seed {seed}) from the "
                                f"{len(pool)} passing non-copies of scored.jsonl, in its order")
            if clips.get("passing") not in (None, len(pool)):
                problems.append(f"clips.json counts {clips.get('passing')} passing non-copies; scored.jsonl has "
                                f"{len(pool)}")
    if problems:
        raise Gate1Error(problems)
    return Run(dir=d, junk=junk, scored=scored, rows=rows, clips=clips, sample=sample)
