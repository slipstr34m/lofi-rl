"""The step-0 requests (docs/design/step0_samples.md §1.1, §1.2, §2.2): one vLLM request a draw, with the draw's
registered seed.

`requests()` reads docs/prereg/brief_streams_v1.json through `lofi_rl.brief_streams.load()`, which refuses a file that
is not exactly a fresh build, and lists the draws of the streams `lofi_rl.step0.STREAMS` names (Q6): in stream order
(rm_heldout, then train), then the stream's own brief order, then k = 0..7. Request i is row i of completions.jsonl.
Each request's seed is `brief_streams.draw_seed(seed, index, k)`, and it must be the file's: "Draw k of a brief
(k = 0..7) is one vLLM request with n = 1 and SamplingParams(seed=draw_seeds[k])" (brief_streams_v1.json, draw_seeds).

STREAMS must be the streams MARK names: `mark_line(STREAMS)` builds D-023's mark from them, and a STREAMS whose mark
is not MARK is refused, so no draw is made under streams not registered first. A brief id is `<seed>:<index>`, as
lofi_rl.rl.train gives a dataset row.

Standard library only (plus lofi_rl's own stdlib modules). Sound claims: none.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import lofi_rl.step0 as S
from lofi_rl import brief_streams
from lofi_rl.brief import Brief
from lofi_rl.brief_streams import ROOT, BriefStreamsError, Pick, Stream
from lofi_rl.step0 import Step0Error

STREAM_ORDER = ("rm_heldout", "train")          # the streams this unit may draw, in the order they are drawn
WHICH = ("all", "pool")                         # all of a stream's briefs, or (train only) the seed pool's (Q1)
MARK_HEAD = "D-023 mark:"                       # the mark's first words (MARK, mark_line)
ANSWERS_HEAD = "D-023 answers:"                 # the answers line's (answers_line)
MARK_END = " \t.*_\"'"                          # what may close such a line in DECISIONS.md: markdown, punctuation


@dataclass(frozen=True)
class Request:
    i: int                  # its row in completions.jsonl: its place in stream, brief and k order
    stream: str
    seed: int               # the stream's seed
    index: int              # the brief's index in that seed
    brief_id: str           # "<seed>:<index>"
    k: int                  # the draw, 0..7
    draw_seed: int          # brief_streams.draw_seed(seed, index, k): SamplingParams(seed=...)
    brief: Brief

    @property
    def cell(self) -> str:
        return brief_streams.cell_name((self.brief.energy, self.brief.mood))


def _spec(name: str) -> brief_streams.Spec:
    return next(s for s in brief_streams.SPECS if s.name == name)


def streams_problems(drawn) -> list[str]:
    """What is wrong with a STREAMS value: it must be (stream, which) pairs, each stream of STREAM_ORDER once and in
    that order, `which` "all" or, for train, "pool"."""
    try:
        pairs = [tuple(p) for p in drawn]
    except TypeError:
        return [f"STREAMS {drawn!r} is not a sequence of (stream, which) pairs"]
    problems = []
    for p in pairs:
        if len(p) != 2 or p[0] not in STREAM_ORDER or p[1] not in WHICH:
            problems.append(f"STREAMS entry {p!r}: needs a stream of {STREAM_ORDER} and one of {WHICH}")
        elif p[1] == "pool" and p[0] != brief_streams.TRAIN:
            problems.append(f"STREAMS entry {p!r}: only {brief_streams.TRAIN} holds the seed pool's briefs")
    names = [p[0] for p in pairs]
    if not pairs:
        problems.append("STREAMS names no stream")
    elif not problems and names != [n for n in STREAM_ORDER if n in names]:
        problems.append(f"STREAMS {pairs!r}: each stream once, in the order {STREAM_ORDER}")
    return problems


def mark_line(drawn=None, rule: str | None = None) -> str:
    """D-023's mark for the streams `drawn` (default STREAMS) and `rule` (default RULE): one line naming both, with the
    draws a brief and the requests in all. MARK must be exactly this line for STREAMS."""
    drawn = S.STREAMS if drawn is None else drawn
    problems = streams_problems(drawn)
    if problems:
        raise Step0Error(problems)
    parts, n = [], 0
    for name, which in drawn:
        spec = _spec(name)
        count = spec.count if which == "all" else S.POOL_N
        parts.append(f"{name} (seed {spec.seed}, " + (f"all {count} briefs)" if which == "all" else
                                                     f"the seed pool's {count} briefs by Q1 {S.POOL_BRIEFS})"))
        n += count * S.DRAWS
    return (f"D-023 mark: rule {S.RULE if rule is None else rule}; streams {' and '.join(parts)}; "
            f"{S.DRAWS} draws a brief, {n} requests")


def answers_line() -> str:
    """The selection answers (the spec's §7: Q1-Q5 and Q8; Q6 is MARK's streams, Q7 a sentence) as one line, built
    from the constants that hold them. D-023 holds it verbatim beside his answers in words (tests/test_step0_docs.py),
    so an answer D-023 records and a constant left at another value cannot both stand (finding R1-3)."""
    def pairs(m) -> str:
        return ", ".join(f"{k} {v}" for k, v in dict(m).items())
    bands = "/".join(f"{lo}-{hi}" for lo, hi in S.POOL_BANDS)
    fill = "-".join(str(x) for x in S.POOL_FILL_BAND)
    return (f"{ANSWERS_HEAD} Q1 POOL_BRIEFS {S.POOL_BRIEFS}, POOL_BANDS {bands}, POOL_FILL_BAND {fill}; "
            f"Q2 POOL_SHARES_REFB {S.POOL_SHARES_REFB}; Q3 PASSING {S.PASSING}; Q4 DISTINCT {pairs(S.DISTINCT)}; "
            f"Q5 POOL_EXCLUDES {S.POOL_EXCLUDES}; Q8 SHORT_RULE {S.SHORT_RULE}")


def find_marks(text: str, head: str = MARK_HEAD) -> list[str]:
    """Every D-023 line of `head` (the mark, or the answers line) a DECISIONS.md text holds, in order: from the head to
    the next backtick or double quote, or the end of its line, less what may close it there (MARK_END). Generate's
    pre-check, check's git condition and tests/test_step0_docs.py all read D-023 this one way, so a mark wrapped
    across lines, or with text after it on its line, is no mark to any of them."""
    out = []
    for line in text.splitlines():
        at = line.find(head)
        while at >= 0:
            body = line[at:].split("`", 1)[0].split('"', 1)[0]
            out.append(body.rstrip(MARK_END))
            at = line.find(head, at + len(head))
    return out


def check_mark(drawn=None, mark: str | None = None) -> None:
    """Refuse (Step0Error) a STREAMS whose mark is not MARK: the streams drawn must be the ones D-023 registers."""
    drawn = S.STREAMS if drawn is None else drawn
    mark = S.MARK if mark is None else mark
    want = mark_line(drawn)
    if want != mark:
        raise Step0Error([f"STREAMS {tuple(tuple(p) for p in drawn)!r} gives the mark {want!r}, but MARK is "
                          f"{mark!r}: draw only the streams D-023's mark names (register them first)"])


def load(path: Path | str | None = None) -> dict[str, Stream]:
    """The registered streams (brief_streams.load of the streams file), its refusal as a Step0Error."""
    try:
        return brief_streams.load(ROOT / S.STREAMS_FILE if path is None else path)
    except BriefStreamsError as err:
        raise Step0Error([f"the streams file is refused: {err}"]) from err


def drawn_picks(streams: Mapping[str, Stream], drawn=None) -> list[tuple[Stream, list[Pick]]]:
    """Each drawn stream with the briefs it draws, in STREAMS' order and the stream's own."""
    from lofi_rl.step0 import select
    drawn = S.STREAMS if drawn is None else drawn
    out = []
    for name, which in drawn:
        st = streams[name]
        if st.spec.draws != S.DRAWS:
            raise Step0Error([f"stream {name} has {st.spec.draws} draws a brief; step 0 draws {S.DRAWS}"])
        picks = list(st.picks)
        if which == "pool":
            keep = set(select.pool_briefs(st))
            picks = [p for p in picks if p.index in keep]
        out.append((st, picks))
    return out


def requests(path: Path | str | None = None, *, drawn=None, mark: str | None = None,
             streams: Mapping[str, Stream] | None = None) -> tuple[Request, ...]:
    """Every step-0 request, in order (module doc). `drawn` and `mark` default to STREAMS and MARK, read at call time;
    `streams`, when given, is used instead of reading the file. Step0Error on an edited streams file or a STREAMS whose
    mark is not MARK."""
    drawn = S.STREAMS if drawn is None else drawn
    check_mark(drawn, mark)
    loaded = load(path) if streams is None else streams
    out: list[Request] = []
    for st, picks in drawn_picks(loaded, drawn):
        seed = st.spec.seed
        for p in picks:
            bid = brief_streams.brief_id(seed, p.index)
            for k in range(S.DRAWS):
                ds = brief_streams.draw_seed(seed, p.index, k)
                if ds != p.draw_seeds[k]:
                    raise Step0Error([f"{bid} draw {k}: draw_seed gives {ds}, the streams file {p.draw_seeds[k]}"])
                out.append(Request(len(out), st.spec.name, seed, p.index, bid, k, ds, p.brief))
    return tuple(out)


def stream_counts(reqs: Iterable[Request]) -> dict[str, dict[str, int]]:
    """generate.json's `streams`: per stream drawn, its seed, its briefs and its draws a brief."""
    by: dict[str, Counter] = {}
    seeds: dict[str, int] = {}
    for r in reqs:
        by.setdefault(r.stream, Counter())[r.brief_id] += 1
        seeds[r.stream] = r.seed
    return {name: {"seed": seeds[name], "briefs": len(c), "draws": max(c.values())} for name, c in by.items()}
