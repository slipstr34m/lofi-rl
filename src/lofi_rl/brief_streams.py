"""The brief streams of D-021 item 1, fixed in docs/prereg/brief_streams_v1.json before any step-0 sample is drawn.

    PYTHONPATH=src python -m lofi_rl.brief_streams [--out FILE] [--force]   # writes it; never over a different file
    PYTHONPATH=src python -m lofi_rl.brief_streams --check [FILE]           # exit 0 iff FILE is exactly a fresh build

A brief is `lofi_rl.brief.sample_brief(seed, index)`, so a stream is a seed and a set of its indices (D-021 item 1):
train is seed 596 from index 0, and run 1 takes [0, 400); measurement is seed 2026, [0, 500) (junk-rate); the
evaluation streams are rm_heldout (40 briefs from seed 5101), dev (40 from 5102), final_ab (100 from 5103) and
partb_dev (Part B's dev set, 270 from 5104).

**Stratified by cell.** An evaluation stream walks its seed from index 0 and takes each brief whose energy x mood cell
still has room, until every cell is full. The file lists the chosen indices in stream order, their briefs, and how far
the seed was read. D-021 fixes the sizes ("4 or 5 a cell for 40, 11 or 12 for 100, and 30 for 270"); which cells take
the extra brief is one rule, Claude's default: the 9 cells in ENERGIES x MOOD_NAMES order, of which the first n mod 9
take ceil(n / 9) and the others floor(n / 9).

**Draw seeds.** The vLLM seed of draw k (k = 0..7) of brief `index` of the stream `seed`:

    key = splitmix64(splitmix64(splitmix64(seed mod 2**64) ^ index) ^ FIELD_DRAW)
    draw_seed = splitmix64(key ^ k) >> 33                 # = prng.below(h, 2**31): the top 31 bits

- The key is brief.py's chain (docs/COMPILER.md §6) with a field of its own, b"draw", beside brief.py's fields 1-4.
  SplitMix64 is a bijection on 64-bit words, so a draw key is never the hash behind a field of its own brief.
- A seed is non-negative, since vLLM 0.32 reads -1 as no seed (`vllm/sampling_params.py:567-568`), and fits a signed
  32-bit int.
- A draw is one request with n = 1. For n > 1, vLLM gives child i the seed `seed + i`
  (`vllm/v1/engine/parallel_sampling.py:81`), which is not the table's.
- Eight draws a brief, not four: a use that needs m samples passing the gates (taste_pipeline.md takes each RM
  held-out brief's first 4 step-0 samples that pass) takes the first m passing draws in k order, and a brief still
  short after its 8 draws is reported short, never drawn again under other seeds.
- The measurement stream has no draws: junk-rate seeds its engine as D-020 registered (junk.py), one request a
  brief, so a per-brief seed would go unused. The stream fixes its briefs only.
- Every brief of a cell gets the same prompt (the model sees only the cell), so one seed drawn twice in a cell could
  replay a sample. build() therefore refuses a seed repeated anywhere in the file: across streams, briefs and draws.

**Twins** are evaluation briefs whose (energy, mood, bpm, tonic) equal a brief of train [0, 400). The file lists them
per stream, with the train briefs they equal, and excludes none (D-021: the policy never reads bpm or tonic, and an
exclusion would thin the 75-90 BPM centre). It also lists the twins among each evaluation seed's first n briefs in
stream order, unstratified: the reading behind D-021's laptop count.

**Held-out** means fresh samples of the same 9 prompts (D-021). The file says so, and that nothing is excluded.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lofi_rl.brief import ENERGIES, MOOD_NAMES, Brief, sample_brief
from lofi_rl.prng import MASK, TEST_VECTOR, below, splitmix64

ROOT = Path(__file__).resolve().parents[2]
STREAMS_FILE = ROOT / "docs" / "prereg" / "brief_streams_v1.json"
SCHEMA = "lofi-rl/brief-streams/1"
CELLS: tuple[tuple[str, str], ...] = tuple((e, m) for e in ENERGIES for m in MOOD_NAMES)
DRAWS = 8                                     # k = 0..7: D-021's step-0 samples, checkpoint quads, the A/B's <= 4 draws,
                                              # with spares for the uses that keep the first passing draws
FIELD_DRAW = int.from_bytes(b"draw", "big")   # 0x64726177; brief.py's fields are energy 1, mood 2, tempo 3, tonic 4
SEED_BITS = 31
READ_LIMIT = 100_000                          # a stratified walk with a cell still open here fails closed
TWIN_FIELDS = ("energy", "mood", "bpm", "tonic")
WIDTH = 220                                   # layout: a flat value that fits in WIDTH columns is written on one line
# Test vectors in the file, (stream seed, brief index, k): the formula at zero, then draws the table also holds.
VECTORS = ((0, 0, 0), (596, 0, 0), (596, 0, 1), (596, 0, 2), (596, 0, 3), (596, 399, 7), (5101, 0, 0), (5104, 334, 7))


class BriefStreamsError(ValueError):
    pass


@dataclass(frozen=True)
class Spec:
    name: str
    seed: int
    count: int                  # the briefs the stream holds
    stratified: bool            # True: the first briefs of each cell (per_cell_counts); False: indices [0, count)
    purpose: str                # D-021 item 1's words
    draws: int = DRAWS          # draw seeds per brief (0: the stream fixes its briefs only)


SPECS: tuple[Spec, ...] = (
    Spec("train", 596, 400, False, "Training: seed 596 from index 0. Run 1 takes [0, 400)."),
    Spec("measurement", 2026, 500, False, "Measurement: seed 2026, [0, 500) (junk-rate).", draws=0),
    Spec("rm_heldout", 5101, 40, True, "RM held-out: 40 briefs from seed 5101."),
    Spec("dev", 5102, 40, True, "dev: 40 from seed 5102."),
    Spec("final_ab", 5103, 100, True, "the final A/B: 100 from seed 5103."),
    Spec("partb_dev", 5104, 270, True, "Part B's dev set: 270 from seed 5104."),
)
TRAIN = "train"
EVALUATION: tuple[str, ...] = tuple(s.name for s in SPECS if s.stratified)


@dataclass(frozen=True)
class Pick:
    index: int                  # the brief's index in its stream's seed
    brief: Brief
    draw_seeds: tuple[int, ...]  # draw_seeds[k]: the vLLM seed of draw k

    @property
    def cell(self) -> tuple[str, str]:
        return self.brief.energy, self.brief.mood


@dataclass(frozen=True)
class Stream:
    spec: Spec
    picks: tuple[Pick, ...]     # in stream order
    read: int                   # the seed's indices [0, read) were read to choose them

    def brief_ids(self) -> list[str]:
        return [brief_id(self.spec.seed, p.index) for p in self.picks]


def brief_id(seed: int, index: int) -> str:
    """'<seed>:<index>', the brief_id that lofi_rl.rl.train gives a dataset row."""
    return f"{seed}:{index}"


def cell_name(cell: tuple[str, str]) -> str:
    return f"{cell[0]}/{cell[1]}"


def _count(n: Any) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise BriefStreamsError(f"a stream size must be an integer >= 0, got {n!r}")
    return n


def _word(x: Any, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= MASK:
        raise BriefStreamsError(f"{what} {x!r} is outside 0..2**64 - 1")
    return x


def per_cell_counts(n: int) -> dict[tuple[str, str], int]:
    """The rule: in CELLS order, the first n mod 9 cells take ceil(n / 9) briefs and the others floor(n / 9)."""
    q, r = divmod(_count(n), len(CELLS))
    return {c: q + (j < r) for j, c in enumerate(CELLS)}


def stratified_indices(seed: int, n: int, *, limit: int = READ_LIMIT) -> tuple[tuple[int, ...], int]:
    """(chosen indices in stream order, indices read): the first briefs of each cell of `seed`, per_cell_counts(n)."""
    room = per_cell_counts(n)
    left, chosen, i = n, [], 0
    while left:
        if i >= limit:
            open_cells = ", ".join(f"{cell_name(c)} {k}" for c, k in room.items() if k)
            raise BriefStreamsError(f"seed {seed}: after {limit} briefs these cells still have room: {open_cells}")
        b = sample_brief(seed, i)
        c = (b.energy, b.mood)
        if room[c]:
            room[c] -= 1
            left -= 1
            chosen.append(i)
        i += 1
    return tuple(chosen), i


def draw_key(seed: int, index: int) -> int:
    """splitmix64(splitmix64(splitmix64(seed mod 2**64) ^ index) ^ FIELD_DRAW): brief.py's chain, field b"draw"."""
    return splitmix64(splitmix64(splitmix64(seed & MASK) ^ _word(index, "brief index")) ^ FIELD_DRAW)


def draw_seed(seed: int, index: int, k: int) -> int:
    """The vLLM seed of draw k of brief `index` of stream `seed`: the top 31 bits of splitmix64(key ^ k)."""
    return below(splitmix64(draw_key(seed, index) ^ _word(k, "draw")), 1 << SEED_BITS)


def build_stream(spec: Spec) -> Stream:
    if spec.stratified:
        indices, read = stratified_indices(spec.seed, spec.count)
    else:
        indices, read = tuple(range(_count(spec.count))), spec.count
    picks = tuple(Pick(i, sample_brief(spec.seed, i), tuple(draw_seed(spec.seed, i, k) for k in range(spec.draws)))
                  for i in indices)
    return Stream(spec, picks, read)


def _refuse_repeated_draw_seeds(streams: Mapping[str, Stream]) -> None:
    seen: dict[int, str] = {}
    for name, st in streams.items():
        for p in st.picks:
            for k, s in enumerate(p.draw_seeds):
                here = f"{name} {brief_id(st.spec.seed, p.index)} draw {k}"
                if s in seen:
                    raise BriefStreamsError(f"draw seed {s} repeats: {seen[s]} and {here}")
                seen[s] = here


def build() -> dict[str, Stream]:
    """Every stream of SPECS, fresh. Fails closed on a shared stream seed or a repeated draw seed."""
    if len({s.seed for s in SPECS}) != len(SPECS) or len({s.name for s in SPECS}) != len(SPECS):
        raise BriefStreamsError("two streams share a seed or a name")
    streams = {s.name: build_stream(s) for s in SPECS}
    _refuse_repeated_draw_seeds(streams)
    return streams


def twin_key(b: Brief) -> tuple[str, str, int, int]:
    return b.energy, b.mood, b.bpm, b.tonic


def _by_key(picks: Iterable[Pick]) -> dict[tuple[str, str, int, int], tuple[int, ...]]:
    out: dict[tuple[str, str, int, int], list[int]] = {}
    for p in picks:
        out.setdefault(twin_key(p.brief), []).append(p.index)
    return {k: tuple(v) for k, v in out.items()}


def twins(streams: Mapping[str, Stream]) -> dict[str, list[tuple[int, tuple[int, ...]]]]:
    """Per evaluation stream, (index, the train indices it equals) for each chosen brief with a twin in train."""
    by = _by_key(streams[TRAIN].picks)
    return {name: [(p.index, by[twin_key(p.brief)]) for p in streams[name].picks if twin_key(p.brief) in by]
            for name in EVALUATION}


def twins_in_stream_order(streams: Mapping[str, Stream]) -> dict[str, list[int]]:
    """Per evaluation seed, the indices among its first n briefs (unstratified) that have a twin in train."""
    by = _by_key(streams[TRAIN].picks)
    out = {}
    for name in EVALUATION:
        sp = streams[name].spec
        out[name] = [i for i in range(sp.count) if twin_key(sample_brief(sp.seed, i)) in by]
    return out


def _row(p: Pick) -> dict[str, Any]:
    b = p.brief
    return {"index": p.index, "energy": b.energy, "mood": b.mood, "bpm": b.bpm, "tonic": b.tonic,
            "draw_seeds": list(p.draw_seeds)}


def _stream_doc(st: Stream) -> dict[str, Any]:
    sp = st.spec
    doc: dict[str, Any] = {"seed": sp.seed, "purpose": sp.purpose, "count": sp.count, "draws": sp.draws}
    if sp.stratified:
        doc |= {"selection": "stratified", "per_cell": {cell_name(c): k for c, k in per_cell_counts(sp.count).items()}}
    else:
        got = Counter(p.cell for p in st.picks)
        doc |= {"selection": "range", "start": 0, "stop": sp.count,
                "cell_counts": {cell_name(c): got[c] for c in CELLS}}
    doc |= {"read": st.read, "briefs": [_row(p) for p in st.picks]}
    return doc


def _rule_examples() -> str:
    return ". ".join(f"{n}: {' '.join(str(k) for k in per_cell_counts(n).values())}" for n in (40, 100, 270)) + "."


def document(streams: Mapping[str, Stream] | None = None) -> dict[str, Any]:
    s = build() if streams is None else streams
    train = s[TRAIN].spec
    tw, first = twins(s), twins_in_stream_order(s)
    return {
        "schema": SCHEMA,
        "registered": [
            "docs/DECISIONS.md D-021 item 1, decided by Shresth 2026-11-02 (\"A: fix first\").",
            "Committed with that entry before any step-0 sample is drawn."],
        "written_by": [
            "python -m lofi_rl.brief_streams (PYTHONPATH=src on the laptop); do not edit this file.",
            "python -m lofi_rl.brief_streams --check exits 0 iff this file is exactly a fresh build."],
        "sampler": [
            "A brief is lofi_rl.brief.sample_brief(seed, index) (docs/COMPILER.md section 6).",
            "Each field is its own keyed draw h = splitmix64(splitmix64(splitmix64(seed mod 2**64) ^ index) ^ field),"
            " fields energy 1, mood 2, tempo 3, tonic 4,",
            "so brief i of a seed is the same however many briefs are drawn, and a stream is a seed and its indices."],
        "held_out": [
            "Held-out means fresh samples of the same 9 prompts (D-021: in D-009, D-014 and the obedience claim).",
            "The model sees only the cell, so all 9 prompts are the same in every stream.",
            "Held-out briefs are kept apart only by fresh samples, rendered at tempi and tonics the model never reads.",
            "No draw seed repeats anywhere in this file: across streams, briefs and draws."],
        "excluded": [
            "None: every stream keeps every brief it selects, twins included.",
            "The policy never reads bpm or tonic, and an exclusion would thin the 75-90 BPM centre (D-021 item 1)."],
        "cells": [list(c) for c in CELLS],
        "selection": {
            "range": ["The seed's indices [start, stop), in order."],
            "stratified": [
                "Walk the seed from index 0 and take each brief whose energy x mood cell still has room;"
                " stop when every cell is full.",
                "The chosen briefs are listed in stream order. read is how many indices were read, [0, read),"
                " so the last brief listed is index read - 1."],
            "per_cell": [
                "The 9 cells in ENERGIES x MOOD_NAMES order (cells above): for n briefs the first n mod 9 cells take"
                " ceil(n / 9) and the others floor(n / 9).",
                _rule_examples(),
                "Claude's default for the one rule D-021 item 1 leaves to the file (4 or 5 a cell for 40, 11 or 12 for"
                " 100, 30 for 270)."],
        },
        "draw_seeds": {
            "formula": [
                f"key(seed, index) = splitmix64(splitmix64(splitmix64(seed mod 2**64) ^ index) ^ field),"
                f" field = 0x{FIELD_DRAW:08x} (the ASCII bytes of \"draw\").",
                "draw_seed(seed, index, k) = splitmix64(key(seed, index) ^ k) >> 33, the top 31 bits:"
                " 0 <= draw_seed < 2**31.",
                f"splitmix64 is lofi_rl.prng.splitmix64 (Vigna's SplitMix64, splitmix64(0) ="
                f" 0x{TEST_VECTOR[1]:016x}); ^ is XOR and >> a right shift, on 64-bit words.",
                "key is the brief sampler's chain with a field of its own (the sampler's are 1-4); SplitMix64 is a"
                " bijection on 64-bit words,",
                "so key is never the hash behind a field of its own brief."],
            "use": [
                f"Draw k of a brief (k = 0..{DRAWS - 1}) is one vLLM request with n = 1 and"
                " SamplingParams(seed=draw_seeds[k]).",
                "A use that needs m samples passing the gates takes the first m passing draws in k order; a brief"
                " still short after its draws is reported short, never drawn again under other seeds.",
                "The measurement stream has no draws (draws 0): junk-rate seeds its engine as D-020 registered"
                " (junk.py), one request a brief, so the stream fixes its briefs only.",
                "The seed depends only on the stream's seed, the brief's index and k, not on the checkpoint:"
                " draw k of a brief has one seed under every checkpoint, step 0 included.",
                "With n > 1, vLLM 0.32 gives child i of a request the seed seed + i"
                " (vllm/v1/engine/parallel_sampling.py:81), not this table's.",
                "D-021 item 1: these cover the step-0 samples, the checkpoint quads and the final A/B's <= 4 draws."],
            "field": FIELD_DRAW,
            "bits": SEED_BITS,
            "draws": {name: st.spec.draws for name, st in s.items()},
            "test_vectors": [{"seed": sd, "index": i, "k": k, "key": f"0x{draw_key(sd, i):016x}",
                              "draw_seed": draw_seed(sd, i, k)} for sd, i, k in VECTORS],
        },
        "streams": {name: _stream_doc(st) for name, st in s.items()},
        "twins": {
            "rule": [
                f"An evaluation brief is a twin when its ({', '.join(TWIN_FIELDS)}) equals that of a brief of"
                f" {TRAIN} [0, {train.count}) (seed {train.seed}).",
                "Each is listed by brief id <seed>:<index> (lofi_rl.rl.train's brief_id) with every train brief it"
                " equals."],
            "why": [
                "Listed for the RM's report and excluded from nothing (D-021 item 1): the policy never reads bpm or"
                " tonic, and an exclusion would thin the 75-90 BPM centre."],
            "against": {"stream": TRAIN, "seed": train.seed, "start": 0, "stop": train.count},
            "streams": {name: {"count": len(tw[name]), "of": s[name].spec.count,
                               "briefs": [{"id": brief_id(s[name].spec.seed, i),
                                           "train": [brief_id(train.seed, j) for j in js]} for i, js in tw[name]]}
                        for name in EVALUATION},
            "in_stream_order": {
                "note": [
                    "The twins among each evaluation seed's first n briefs in stream order, unstratified:"
                    " the reading behind D-021 item 1's laptop count."],
                "streams": {name: {"first": s[name].spec.count, "count": len(first[name]),
                                   "ids": [brief_id(s[name].spec.seed, i) for i in first[name]]}
                            for name in EVALUATION},
            },
        },
    }


def _scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int))       # bool is an int; the file holds no floats


def _flat(v: Any) -> bool:
    """A list or dict whose items are scalars or lists of scalars: what may go on one line."""
    items = list(v.values()) if isinstance(v, dict) else v
    return all(_scalar(x) or (isinstance(x, list) and all(_scalar(y) for y in x)) for x in items)


def _json(v: Any, indent: int = 0, column: int = 0) -> str:
    """JSON text of `v`: a flat value on one line when it fits in WIDTH columns from `column`, else one item a line."""
    one = json.dumps(v, separators=(", ", ": "))
    if _scalar(v) or not v or (_flat(v) and column + len(one) <= WIDTH):
        return one
    pad = " " * (indent + 2)
    if isinstance(v, list):
        body = [pad + _json(x, indent + 2, indent + 2) for x in v]
        return "[\n" + ",\n".join(body) + "\n" + " " * indent + "]"
    body = []
    for key, x in v.items():
        head = f"{pad}{json.dumps(key)}: "
        body.append(head + _json(x, indent + 2, len(head)))
    return "{\n" + ",\n".join(body) + "\n" + " " * indent + "}"


def text(streams: Mapping[str, Stream] | None = None) -> str:
    """The file's exact text: ASCII JSON, one brief a line, ending in a newline."""
    return _json(document(streams)) + "\n"


def load(path: Path | str = STREAMS_FILE) -> dict[str, Stream]:
    """The registered streams, after checking that `path` is exactly a fresh build (else BriefStreamsError)."""
    p = Path(path)
    try:
        got = p.read_bytes()
    except OSError as err:
        raise BriefStreamsError(f"{p}: {err}") from err
    streams = build()
    if got != text(streams).encode("ascii"):
        raise BriefStreamsError(f"{p}: not a fresh build of lofi_rl.brief_streams ({_difference(got, text(streams))})")
    return streams


def _difference(got: bytes, want: str) -> str:
    a, b = got.decode("utf-8", errors="replace").splitlines(), want.splitlines()
    for n, (x, y) in enumerate(zip(a, b), 1):
        if x != y:
            return f"line {n} differs"
    return f"{len(a)} lines where the build has {len(b)}" if len(a) != len(b) else "line endings differ"


def summary(streams: Mapping[str, Stream]) -> str:
    tw = twins(streams)
    parts = []
    for name, st in streams.items():
        if st.spec.stratified:
            parts.append(f"{name} {len(st.picks)} of [0, {st.read}), {len(tw[name])} twins")
        else:
            parts.append(f"{name} [0, {st.read})")
    return "; ".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lofi_rl.brief_streams", description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=STREAMS_FILE)
    p.add_argument("--force", action="store_true", help="replace an existing --out that differs from the build")
    p.add_argument("--check", nargs="?", const=str(STREAMS_FILE), default=None, metavar="FILE",
                   help="exit 0 iff FILE is exactly a fresh build (default: the registered file)")
    a = p.parse_args(argv)
    try:
        streams = build()
        want = text(streams)
    except BriefStreamsError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if a.check is not None:
        path = Path(a.check)
        try:
            got = path.read_bytes()
        except OSError as err:
            print(f"error: {path}: {err}", file=sys.stderr)
            return 2
        if got != want.encode("ascii"):
            print(f"{path}: not a fresh build of lofi_rl.brief_streams ({_difference(got, want)})", file=sys.stderr)
            return 1
        print(f"{path}: ok, a fresh build: {summary(streams)}")
        return 0
    out: Path = a.out
    if out.exists():
        if out.read_bytes() == want.encode("ascii"):
            print(f"{out}: unchanged, a fresh build: {summary(streams)}")
            return 0
        if not a.force:
            print(f"{out}: differs from the build ({_difference(out.read_bytes(), want)}); the file is registered "
                  f"(D-021 item 1), so it is replaced only with --force", file=sys.stderr)
            return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(want.encode("ascii"))
    print(f"wrote {out}: {summary(streams)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
