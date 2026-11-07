"""The selection rule step0-select/1 (docs/design/step0_samples.md §1.3-1.8; D-023): which step-0 draws make the RM
held-out tuples, PAR's Ref_b and the seed pool, as a pure function of complete records.

    doc = sets_json(scored, records, streams=requests.load(), render_code=...)      # sets.json

Inputs: scored.jsonl's rows (junk.score_row's fields plus stream, brief_id and k), render records shaped as
clips.json's (this run's held-out renders; for Ref_b and the pool, their own later renders), and the registered
streams (each brief's bpm and tonic, its draw seeds, its twins). The result never depends on the order of the rows or
the records. Every brief the rows hold must hold its 8 draws; whether every requested brief is there is the check's
(§4.1, Complete).

**Passing (Q3).** A draw passes its records when it passes G1-G6 and compile (junk.score_row's `ok`; PASSING), and
passes when it then passes G7 at its set's render. G7 is applied lazily, in k order: a draw is rendered only while the
draws before it in its set have not filled m. A draw's G7 is its final render record's (`outcome`).

**Held-out** (rm_heldout, m 4, Q4(a)): walk k = 0..7; take a passing draw whose canonical text differs from every draw
already taken (DISTINCT["heldout"]); stop at 4. Its G7 is this run's render step. A draw not rendered yet is pending
and counts as passing until it is: `pending()` lists the next round's renders.

**Ref_b** (train, m 4): the brief's first 4 draws that pass their records, repeats kept (DISTINCT["refb"]). At Ref_b's
own render (§3, a later unit) a reference that fails G7 is replaced by the brief's first spare that passes it, in k
order (Q2(a)). The spares are its other passing draws, less the pool's draws and any draw with a pool draw's text.

**The pool** (train, m 3): the 220 briefs of Q1 (`pool_briefs`), each walking its draws after its 4 Ref_b draws
(Q2(a)) and taking a passing draw whose canonical text differs from Ref_b's and the earlier picks', and (Q5(a)) from
every held-out clip's, or (Q5(b), (c)) that is not a held-out clip. Stop at 3. A pick that fails G7 at the pool's own
render is dropped and never replaced, so the spares are fixed by the records.

**Before their renders** (this unit), a train draw that has no render record stands by its records: Ref_b's and the
pool's picks are their records' (§4.4), and their G7 comes later (§3). `pending_train=True` makes such a draw pending
instead, for the later units that render them; those calls (select with pending_train, pending with "refb" or "pool")
refuse to run before the held-out set is decided by the records they are given, since the pool's Q5 exclusion reads
the final held-out set: a later unit passes every scored row and this run's clips.json records with its own.

**Short and undetermined (§1.8).** A brief whose walk ends with fewer than m is short, and every draw it walked and did
not take is listed with why: a gate code (G7 at a render), Q3's near_copy or repaired, repeat_of:<k>, refb, pool,
heldout_text or heldout_clip. A brief whose walk meets a draw that was rendered with an error twice or skipped by the
deadline, or that still has a draw to render, is undetermined (the render step then exits 1).

Standard library only (plus lofi_rl's own stdlib modules), and no I/O. Sound claims: none.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import lofi_rl.step0 as S
from lofi_rl import brief_streams
from lofi_rl.brief import BPM_RANGE, TEMPO_WEIGHTS, Brief
from lofi_rl.brief_streams import CELLS, Stream, cell_name, per_cell_counts, stratified_indices
from lofi_rl.step0 import HELDOUT_CLIP, HELDOUT_TEXT, POOL, REFB, REPEAT, Step0Error

PASS = "pass"                       # an outcome: passed G7 at its render
UNDETERMINED = "undetermined"       # an outcome: rendered with an error twice, or skipped by the deadline
PASSING_RULES = ("hard", "hard_not_copy", "hard_no_repair")       # Q3 (a), (b), (c)
POOL_RULES = ("band", "cell")                                     # Q1 (a), (b)
EXCLUDES = ("text", "clip")                                       # Q5 (a); (b) and (c)
_ID = re.compile(r"(\d+):(\d+)")
_HEX64 = re.compile(r"[0-9a-f]{64}")


# --------------------------------------------------------------------------------------------- records
@dataclass(frozen=True)
class Draw:
    i: int
    k: int
    draw_seed: int
    why: str | None             # None: it passes its records (Q3); else why not
    canon: str | None           # canon_sha256: the canonical text's sha256 (a loop that parsed)
    text_sha256: str | None
    row: Mapping = field(compare=False)     # its scored.jsonl row, for the readouts


@dataclass(frozen=True)
class Item:
    """One brief the rows hold, with its 8 draws in k order."""
    brief_id: str
    stream: str
    index: int
    brief: Brief
    twin_of: tuple[str, ...]    # held-out: the train briefs it equals; train: the held-out briefs that equal it
    draws: tuple[Draw, ...]

    @property
    def cell(self) -> str:
        return cell_name((self.brief.energy, self.brief.mood))

    @property
    def twin_key(self) -> tuple[str, str, int, int]:
        return brief_streams.twin_key(self.brief)


def _whole(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def why_not(row: Mapping) -> str | None:
    """Why a scored row does not pass its records under Q3 (PASSING), or None when it does."""
    if S.PASSING not in PASSING_RULES:
        raise Step0Error([f"PASSING {S.PASSING!r} is not one of {PASSING_RULES} (Q3)"])
    if row.get("ok") is not True:
        return str(row.get("gate"))
    if S.PASSING == "hard_not_copy" and row.get("near_copy") is True:
        return "near_copy"
    if S.PASSING == "hard_no_repair" and row.get("repairs"):
        return "repaired"
    return None


def _twins(streams: Mapping[str, Stream]) -> dict[str, tuple[str, ...]]:
    """brief id -> its twins across rm_heldout and train (brief_streams.twins), both ways."""
    held, train = streams["rm_heldout"].spec.seed, streams[brief_streams.TRAIN].spec.seed
    out: dict[str, list[str]] = {}
    for i, js in brief_streams.twins(streams)["rm_heldout"]:
        out.setdefault(brief_streams.brief_id(held, i), []).extend(brief_streams.brief_id(train, j) for j in js)
        for j in js:
            out.setdefault(brief_streams.brief_id(train, j), []).append(brief_streams.brief_id(held, i))
    return {k: tuple(v) for k, v in out.items()}


def items(scored: Iterable[Mapping], streams: Mapping[str, Stream]) -> dict[str, list[Item]]:
    """Per set stream (rm_heldout, train), the briefs the rows hold in stream order, each with its 8 draws. Step0Error
    lists every row that is not a registered draw of those streams or repeats one, and every brief short of a draw."""
    problems: list[str] = []
    picks = {name: {p.index: p for p in streams[name].picks} for name in ("rm_heldout", brief_streams.TRAIN)}
    seeds = {name: streams[name].spec.seed for name in picks}
    by: dict[str, dict[int, Draw]] = {}
    seen_i: set[int] = set()
    for r in scored:
        i, name, bid, k = r.get("i"), r.get("stream"), r.get("brief_id"), r.get("k")
        m = _ID.fullmatch(bid) if isinstance(bid, str) else None
        if not (_whole(i) and name in picks and m and _whole(k) and 0 <= k < S.DRAWS):
            problems.append(f"row i={i!r} stream={name!r} brief_id={bid!r} k={k!r}: not a draw of rm_heldout or train")
            continue
        seed, index = int(m.group(1)), int(m.group(2))
        if seed != seeds[name] or index not in picks[name]:
            problems.append(f"row {i}: {bid} is not a brief of the {name} stream (seed {seeds[name]})")
            continue
        b = picks[name][index].brief
        if r.get("energy", b.energy) != b.energy or r.get("mood", b.mood) != b.mood:
            problems.append(f"row {i}: {bid} is {b.energy}/{b.mood}, the row says {r.get('energy')}/{r.get('mood')}")
        ok = r.get("ok")
        if not isinstance(ok, bool) or (ok and not _HEX64.fullmatch(str(r.get("canon_sha256")))) \
                or (not ok and not (isinstance(r.get("gate"), str) and r["gate"])):
            problems.append(f"row {i}: needs `ok` true with its canon_sha256, or `ok` false with its gate")
            continue
        if i in seen_i or k in by.get(bid, {}):
            problems.append(f"row {i}: {bid} draw {k} or row index {i} is repeated")
            continue
        seen_i.add(i)
        by.setdefault(bid, {})[k] = Draw(i, k, picks[name][index].draw_seeds[k], why_not(r), r.get("canon_sha256"),
                                         r.get("text_sha256"), r)
    for bid, draws in by.items():
        if len(draws) != S.DRAWS:
            problems.append(f"{bid} holds draws {sorted(draws)}; every brief holds k = 0..{S.DRAWS - 1}")
    if problems:
        raise Step0Error(problems)
    twins = _twins(streams)
    out: dict[str, list[Item]] = {}
    for name in picks:
        rows = []
        for p in streams[name].picks:
            bid = brief_streams.brief_id(seeds[name], p.index)
            if bid in by:
                rows.append(Item(bid, name, p.index, p.brief, twins.get(bid, ()),
                                 tuple(by[bid][k] for k in range(S.DRAWS))))
        out[name] = rows
    return out


def _final(records: Iterable[Mapping], drawn: Mapping[tuple[str, int], Draw]) -> dict[tuple[str, int], list[Mapping]]:
    """(brief id, k) -> its render records, each checked against its draw; Step0Error lists every one that is not."""
    problems: list[str] = []
    out: dict[tuple[str, int], list[Mapping]] = {}
    for rec in records:
        key = (rec.get("brief_id"), rec.get("k"))
        d = drawn.get(key) if isinstance(key[0], str) and _whole(key[1]) else None
        if d is None or rec.get("i") != d.i or not isinstance(rec.get("final"), bool):
            problems.append(f"render record i={rec.get('i')!r} {key[0]!r} k={key[1]!r}: not a draw of the rows, or "
                            "no true/false `final`")
            continue
        out.setdefault(key, []).append(rec)
    for key, recs in out.items():
        if sum(1 for r in recs if r["final"]) > 1:
            problems.append(f"{key[0]} draw {key[1]} has {sum(1 for r in recs if r['final'])} final render records")
    if problems:
        raise Step0Error(problems)
    return out


def outcome(recs: Sequence[Mapping]) -> str | None:
    """A draw's G7 from its render records: PASS, the gate code its final render failed (G7, or another gate),
    UNDETERMINED (an error twice, or a deadline skip), or None when no final record decides it yet (never rendered,
    or an error the next round retries)."""
    final = next((r for r in recs if r.get("final") is True), None)
    if final is None:
        return None
    if final.get("ok") is True:
        return PASS
    if final.get("gate"):
        return str(final["gate"])
    return UNDETERMINED


def render_record(recs: Sequence[Mapping]) -> Mapping | None:
    """The final record of a draw that rendered (its clip id and audio hash), or None."""
    return next((r for r in recs if r.get("final") is True and r.get("ok") is True), None)


# --------------------------------------------------------------------------------------------- the walk
@dataclass
class Walk:
    """One brief's walk through its draws for one set."""
    taken: list[Draw] = field(default_factory=list)
    not_taken: list[tuple[int, str]] = field(default_factory=list)
    pending: list[Draw] = field(default_factory=list)       # passing so far; its G7 still to come (to render)
    undetermined: bool = False

    def first_with(self, canon: str | None) -> int | None:
        """The k of the first draw taken or pending with this canonical text."""
        ks = [d.k for d in self.taken + self.pending if canon is not None and d.canon == canon]
        return min(ks) if ks else None


def walk(draws: Sequence[Draw], m: int, skip: Callable[[Draw, Walk], str | None],
         g7: Callable[[Draw], str | None]) -> Walk:
    """k order until m are taken or pending: a draw `skip` gives a reason for is not taken; otherwise its G7 decides:
    PASS taken, None pending (counted toward m as if it passed: the lazy rule), a gate code not taken, UNDETERMINED
    ends the walk undetermined."""
    w = Walk()
    for d in draws:
        if len(w.taken) + len(w.pending) >= m:
            break
        why = skip(d, w)
        if why is not None:
            w.not_taken.append((d.k, why))
            continue
        o = g7(d)
        if o == PASS:
            w.taken.append(d)
        elif o is None:
            w.pending.append(d)
        elif o == UNDETERMINED:
            w.undetermined = True
            break
        else:
            w.not_taken.append((d.k, o))
    return w


def _distinct(name: str) -> bool:
    v = S.DISTINCT.get(name)
    if not isinstance(v, bool):
        raise Step0Error([f"DISTINCT[{name!r}] is {v!r}, not true or false (Q4)"])
    return v


def _skip(name: str) -> Callable[[Draw, Walk], str | None]:
    """The held-out set's and Ref_b's own reasons: not passing, or (DISTINCT) a repeat of a draw taken."""
    distinct = _distinct(name)

    def skip(d: Draw, w: Walk) -> str | None:
        if d.why is not None:
            return d.why
        k = w.first_with(d.canon) if distinct else None
        return None if k is None else f"{REPEAT}{k}"
    return skip


@dataclass(frozen=True)
class Train:
    """One train brief's selection: Ref_b by its records, the pool's picks by theirs (when it is a pool brief), the
    spares, and the walks with the G7 outcomes known."""
    refs: tuple[Draw, ...]              # Ref_b by its records: the first 4 passing draws (Q4)
    picks: tuple[Draw, ...]             # the pool's draws by their records ((), not a pool brief)
    spares: tuple[Draw, ...]
    refb: Walk
    pool: Walk | None


def _train(item: Item, pool_brief: bool, held_texts: set[str], held_clips: set[tuple],
           g7: Callable[[Draw], str | None]) -> Train:
    shares = S.POOL_SHARES_REFB
    if not isinstance(shares, bool) or S.POOL_EXCLUDES not in EXCLUDES:
        raise Step0Error([f"POOL_SHARES_REFB {shares!r} must be true or false (Q2) and POOL_EXCLUDES "
                          f"{S.POOL_EXCLUDES!r} one of {EXCLUDES} (Q5)"])
    m_refb = S.M["refb"]
    refs = walk(item.draws, m_refb, _skip("refb"), lambda d: PASS).taken
    after = refs[-1].k if len(refs) == m_refb else S.DRAWS        # the pool walks the draws after Ref_b's 4 (Q2(a))
    ref_ks = {d.k for d in refs}

    pool, picks = None, ()
    if pool_brief:
        def skip_pool(d: Draw, w: Walk) -> str | None:
            if not shares and d.k <= after:
                if d.k in ref_ks:
                    return REFB
                if d.why is None:                                 # a repeat Ref_b skipped (DISTINCT["refb"])
                    return f"{REPEAT}{min(r.k for r in refs if r.canon == d.canon)}"
            if d.why is not None:
                return d.why
            if not shares and d.canon in {r.canon for r in refs}:
                return f"{REPEAT}{min(r.k for r in refs if r.canon == d.canon)}"
            k = w.first_with(d.canon)
            if k is not None:
                return f"{REPEAT}{k}"
            if S.POOL_EXCLUDES == "text" and d.canon in held_texts:
                return HELDOUT_TEXT
            if (item.twin_key, d.canon) in held_clips:            # a held-out clip is always skipped (§9)
                return HELDOUT_CLIP
            return None
        by_records = walk(item.draws, S.M["pool"], skip_pool, lambda d: PASS)
        picks = tuple(by_records.taken)
        pool = Walk(not_taken=list(by_records.not_taken))
        for d in picks:                                       # G7 at the pool's render: a failure is not replaced
            o = g7(d)
            if o == PASS:
                pool.taken.append(d)
            elif o is None:
                pool.pending.append(d)
            elif o == UNDETERMINED:
                pool.undetermined = True
            else:
                pool.not_taken.append((d.k, o))
        pool.not_taken.sort()

    pick_ks, pick_texts = {d.k for d in picks}, {d.canon for d in picks}
    if shares:
        spares = tuple(d for d in item.draws if d.k > after and d.why is None)
    else:
        spares = tuple(d for d in item.draws if d.k > after and d.why is None and d.k not in pick_ks
                       and d.canon not in pick_texts)
    own = _skip("refb")

    def skip_refb(d: Draw, w: Walk) -> str | None:
        if not shares and d.k > after:
            if d.k in pick_ks:
                return POOL
            if d.why is None and d.canon in pick_texts:
                return f"{REPEAT}{min(p.k for p in picks if p.canon == d.canon)}"
        return own(d, w)
    return Train(tuple(refs), picks, spares, walk(item.draws, m_refb, skip_refb, g7), pool)


# --------------------------------------------------------------------------------------------- the pool's briefs (Q1)
def band_quotas(n: int, weights: Sequence[int]) -> tuple[int, ...]:
    """n split in proportion to `weights` by largest remainder (ties to the earlier band)."""
    total = sum(weights)
    exact = [n * w / total for w in weights]
    out = [int(x) for x in exact]
    for j in sorted(range(len(out)), key=lambda j: (-(exact[j] - out[j]), j))[:n - sum(out)]:
        out[j] += 1
    return tuple(out)


def band_of(bpm: int, bands: Sequence[tuple[int, int]] | None = None) -> int:
    bands = S.POOL_BANDS if bands is None else bands
    for j, (lo, hi) in enumerate(bands):
        if lo <= bpm <= hi:
            return j
    raise Step0Error([f"bpm {bpm} is in none of the bands {tuple(bands)} (Q1)"])


def band_weights(bands: Sequence[tuple[int, int]] | None = None) -> tuple[int, ...]:
    """The sampler's own share of each band (brief.TEMPO_WEIGHTS): 15, 64 and 5 of 84 for 60-74, 75-90 and 91-95."""
    bands = S.POOL_BANDS if bands is None else bands
    ends = [BPM_RANGE[0] - 1] + [hi for _, hi in bands]
    if not bands or any(lo != ends[j] + 1 or hi < lo for j, (lo, hi) in enumerate(bands)) or ends[-1] != BPM_RANGE[1]:
        raise Step0Error([f"the bands {tuple(bands)} must tile {BPM_RANGE[0]}-{BPM_RANGE[1]} BPM in order (Q1)"])
    return tuple(sum(w for bpm, w in TEMPO_WEIGHTS if lo <= bpm <= hi) for lo, hi in bands)


def band_rooms(train: Stream) -> dict[tuple[str, int], int]:
    """Q1(a)'s room per (cell, band): the cell's per_cell_counts(220) by band_quotas in the sampler's shares, a band
    run 1's [0, 400) cannot fill giving its room to the cell's POOL_FILL_BAND."""
    lo, hi = S.RUN1
    weights = band_weights()
    if tuple(S.POOL_FILL_BAND) not in [tuple(b) for b in S.POOL_BANDS]:
        raise Step0Error([f"POOL_FILL_BAND {S.POOL_FILL_BAND!r} is not one of the bands {S.POOL_BANDS!r} (Q1)"])
    fill = [tuple(b) for b in S.POOL_BANDS].index(tuple(S.POOL_FILL_BAND))
    supply = Counter((cell_name(p.cell), band_of(p.brief.bpm)) for p in train.picks if lo <= p.index < hi)
    rooms: dict[tuple[str, int], int] = {}
    for c, n in per_cell_counts(S.POOL_N).items():
        cell = cell_name(c)
        q = list(band_quotas(n, weights))
        for j in range(len(q)):
            if j != fill and supply[(cell, j)] < q[j]:
                q[fill] += q[j] - supply[(cell, j)]
                q[j] = supply[(cell, j)]
        if supply[(cell, fill)] < q[fill]:
            raise Step0Error([f"{cell}: [{lo}, {hi}) holds {supply[(cell, fill)]} briefs in "
                              f"{S.POOL_BANDS[fill][0]}-{S.POOL_BANDS[fill][1]} BPM, the band needs {q[fill]}"])
        rooms.update({(cell, j): q[j] for j in range(len(q))})
    return rooms


def pool_briefs(train: Stream, rule: str | None = None) -> tuple[int, ...]:
    """The seed pool's 220 train indices in stream order (Q1): "band" walks run 1's [0, 400) taking each brief whose
    cell and band have room (band_rooms); "cell" is brief_streams.stratified_indices(596, 220). Both lie in [0, 400)."""
    rule = S.POOL_BRIEFS if rule is None else rule
    lo, hi = S.RUN1
    if rule == "cell":
        chosen, read = stratified_indices(train.spec.seed, S.POOL_N)
        if read > hi:
            raise Step0Error([f"stratified_indices({train.spec.seed}, {S.POOL_N}) reads [0, {read}), past run 1's "
                              f"[{lo}, {hi})"])
        return tuple(chosen)
    if rule != "band":
        raise Step0Error([f"POOL_BRIEFS {rule!r} is not one of {POOL_RULES} (Q1)"])
    room = band_rooms(train)
    chosen = []
    for p in train.picks:
        key = (cell_name(p.cell), band_of(p.brief.bpm))
        if lo <= p.index < hi and room[key]:
            room[key] -= 1
            chosen.append(p.index)
    if len(chosen) != S.POOL_N:
        raise Step0Error([f"the band walk took {len(chosen)} briefs of [{lo}, {hi}), not {S.POOL_N}"])
    return tuple(chosen)


# --------------------------------------------------------------------------------------------- the sets
@dataclass(frozen=True)
class Selection:
    heldout: tuple[tuple[Item, Walk], ...]
    train: tuple[tuple[Item, Train], ...]
    pool_ids: frozenset[str]
    held_texts: frozenset[str]
    records: Mapping[tuple[str, int], list[Mapping]]


def selection(scored: Iterable[Mapping], records: Iterable[Mapping] = (), *, streams: Mapping[str, Stream],
              pending_train: bool = False) -> Selection:
    """The three sets' walks over the rows and the render records (module doc)."""
    by = items(scored, streams)
    drawn = {(it.brief_id, d.k): d for rows in by.values() for it in rows for d in it.draws}
    recs = _final(records, drawn)

    def g7_of(it: Item, unrendered: str | None) -> Callable[[Draw], str | None]:
        return lambda d: outcome(recs[(it.brief_id, d.k)]) if (it.brief_id, d.k) in recs else unrendered

    heldout = tuple((it, walk(it.draws, S.M["heldout"], _skip("heldout"), g7_of(it, None)))
                    for it in by["rm_heldout"])
    held_texts = frozenset(d.canon for _, w in heldout for d in w.taken)
    held_clips = {(it.twin_key, d.canon) for it, w in heldout for d in w.taken}
    train_stream = streams[brief_streams.TRAIN]
    pool = frozenset(brief_streams.brief_id(train_stream.spec.seed, j) for j in pool_briefs(train_stream))
    train = tuple((it, _train(it, it.brief_id in pool, set(held_texts), held_clips,
                              g7_of(it, None if pending_train else PASS)))
                  for it in by[brief_streams.TRAIN])
    return Selection(heldout, train, pool, held_texts, recs)


def held_out_decided(sel: Selection) -> None:
    """Refuse (Step0Error) a train selection that would run before the held-out set is decided. The pool's held-out
    exclusion (Q5) reads the final held-out set, which only this run's render records give: a later render of Ref_b
    or the pool passes every scored row (rm_heldout's too) and this run's clips.json records with its own. Without
    them every held-out draw is pending, no held-out text is excluded, and the pool and Ref_b's spares come out
    otherwise than the run fixed them (finding R3-1)."""
    open_ = [it.brief_id for it, w in sel.heldout if w.pending or w.undetermined]
    if open_:
        raise Step0Error([f"{len(open_)} held-out briefs are not decided (first {open_[0]}): the pool's held-out "
                          "exclusion reads the final held-out set, so pass this run's clips.json records with your "
                          "own"])


def _taken(d: Draw, rec: Mapping | None) -> dict:
    return {"k": d.k, "i": d.i, "draw_seed": d.draw_seed, "text_sha256": d.text_sha256, "canon_sha256": d.canon,
            "clip_id": rec.get("clip_id") if rec else None, "audio_sha256": rec.get("audio_sha256") if rec else None}


def _entry(name: str, it: Item, w: Walk, recs: Mapping, *, spares: Sequence[Draw] = ()) -> dict:
    out = {"brief_id": it.brief_id, "stream": it.stream, "cell": it.cell, "bpm": it.brief.bpm, "tonic": it.brief.tonic,
           "twin_of": list(it.twin_of),
           "taken": [_taken(d, render_record(recs.get((it.brief_id, d.k), ()))) for d in w.taken],
           "not_taken": [{"k": k, "why": why} for k, why in sorted(w.not_taken)]}
    undetermined = w.undetermined or bool(w.pending)
    if name == "refb":
        out["spares"] = [d.k for d in spares]
    out["short"] = not undetermined and len(w.taken) < S.M[name]
    out["undetermined"] = undetermined
    return {k: out[k] for k in S.ENTRY_FIELDS[name]}


def select(scored: Iterable[Mapping], records: Iterable[Mapping] = (), *, streams: Mapping[str, Stream],
           pending_train: bool = False) -> dict:
    """sets.json's heldout, refb, pool and heldout_text (module doc), each in stream order. With pending_train (a
    later unit's render of Ref_b and the pool) the held-out set must be decided by the records given
    (held_out_decided)."""
    sel = selection(scored, records, streams=streams, pending_train=pending_train)
    if pending_train:
        held_out_decided(sel)
    return {
        "heldout": [_entry("heldout", it, w, sel.records) for it, w in sel.heldout],
        "refb": [_entry("refb", it, t.refb, sel.records, spares=t.spares) for it, t in sel.train],
        "pool": [_entry("pool", it, t.pool, sel.records) for it, t in sel.train if t.pool is not None],
        "heldout_text": [{"i": d.i, "brief_id": it.brief_id, "k": d.k, "heldout_text": d.canon in sel.held_texts}
                         for it, _ in sel.train for d in it.draws],
    }


def sets_json(scored: Iterable[Mapping], records: Iterable[Mapping] = (), *, streams: Mapping[str, Stream],
              render_code: Mapping | None, pending_train: bool = False) -> dict:
    """The whole of sets.json: its schema, the rule, MARK and the render-code key, then select()'s sets."""
    return {"schema": S.SETS_SCHEMA, "rule": S.RULE, "mark": S.MARK,
            "render_code": None if render_code is None else dict(render_code),
            **select(scored, records, streams=streams, pending_train=pending_train)}


def pending(scored: Iterable[Mapping], records: Iterable[Mapping] = (), *, streams: Mapping[str, Stream],
            sets: Sequence[str] = ("heldout",)) -> list[int]:
    """The rows (i) the next render round renders, in stream order then k: each brief's draws the lazy rule needs and
    no final record decides yet (an error the round before is retried once). Round 1 of the held-out set is each
    brief's first 4 distinct candidates; ("refb", "pool") gives a later unit's renders of Ref_b and the pool, which
    need the held-out set decided by the records given: this run's clips.json records with their own
    (held_out_decided)."""
    sel = selection(scored, records, streams=streams, pending_train=True)
    if set(sets) & {"refb", "pool"}:
        held_out_decided(sel)
    out = []
    if "heldout" in sets:
        out += [d.i for _, w in sel.heldout for d in w.pending]
    for it, t in sel.train:
        ks = set()
        if "refb" in sets:
            ks |= {d.k for d in t.refb.pending}
        if "pool" in sets and t.pool is not None:
            ks |= {d.k for d in t.pool.pending}
        out += [it.draws[k].i for k in sorted(ks)]
    return out


# --------------------------------------------------------------------------------------------- step0.json's blocks
def candidates_block(scored: Iterable[Mapping], *, streams: Mapping[str, Stream]) -> dict:
    """step0.json's `candidates` (spec §2.4): each brief's draws that pass their records, in k order; Ref_b's 4 by
    their records; and the held-out render order, round 1's rows."""
    sel = selection(scored, (), streams=streams, pending_train=True)
    briefs = [it for it, _ in sel.heldout] + [it for it, _ in sel.train]
    return {"passing": {it.brief_id: [d.k for d in it.draws if d.why is None] for it in briefs},
            "refb": {it.brief_id: [d.k for d in t.refs] for it, t in sel.train},
            "heldout_order": [d.i for _, w in sel.heldout for d in w.pending]}


def render_block(records: Iterable[Mapping]) -> dict:
    """step0.json's `render` (spec §2.5) from clips.json's records: attempts, renders, final G7 failures and other
    gates, errors twice (errors_final), errors retried once that then rendered or failed a gate (errors_retried),
    deadline skips, rounds."""
    recs = list(records)
    final = [r for r in recs if r.get("final") is True]
    by: dict[tuple, list[Mapping]] = {}
    for r in recs:
        by.setdefault((r.get("brief_id"), r.get("k")), []).append(r)
    retried = sum(1 for rs in by.values()
                  if any(r.get("error") and r.get("final") is False for r in rs)
                  and any(r.get("final") is True and not r.get("error") and not r.get("skipped") for r in rs))
    return {"attempted": sum(1 for r in recs if not r.get("skipped")),
            "rendered": sum(1 for r in recs if r.get("ok") is True),
            "G7": sum(1 for r in final if r.get("gate") == "G7"),
            "other_gates": sum(1 for r in final if r.get("gate") not in (None, "G7")),
            "errors_final": sum(1 for r in final if r.get("error")),
            "errors_retried": retried,
            "skipped": sum(1 for r in final if r.get("skipped")),
            "rounds": max((r.get("round") for r in recs if _whole(r.get("round"))), default=0)}


def readouts(doc: Mapping, scored: Iterable[Mapping]) -> dict:
    """step0.json's `sets` (spec §2.6, per-set readouts, never gates): per set its briefs, taken draws, short and
    undetermined briefs, and how many taken draws are near-copies, repaired, or carry the shown example's A, B or both
    lines verbatim; and per cell the train draws whose canonical text is a held-out clip's, with those Ref_b and the
    pool took."""
    by_i = {r["i"]: r for r in scored}
    out: dict = {}
    for name in S.SETS:
        entries = doc[name]
        rows = [by_i[t["i"]] for e in entries for t in e["taken"]]
        verb = [r.get("verbatim") or {} for r in rows]
        out[name] = {"briefs": len(entries), "taken": len(rows), "short": sum(1 for e in entries if e["short"]),
                     "undetermined": sum(1 for e in entries if e["undetermined"]),
                     "near_copy": sum(1 for r in rows if r.get("near_copy") is True),
                     "repaired": sum(1 for r in rows if r.get("repairs")),
                     "verbatim": {"A": sum(1 for v in verb if v.get("A") is True),
                                  "B": sum(1 for v in verb if v.get("B") is True),
                                  "AB": sum(1 for v in verb if v.get("A") is True and v.get("B") is True)}}
    flagged = {d["i"] for d in doc["heldout_text"] if d["heldout_text"]}
    taken = {name: {t["i"] for e in doc[name] for t in e["taken"]} for name in ("refb", "pool")}
    cells: dict[str, Counter] = {cell_name(c): Counter() for c in CELLS}
    for i in flagged:
        cell = f"{by_i[i]['energy']}/{by_i[i]['mood']}"
        cells[cell]["draws"] += 1
        for name in ("refb", "pool"):
            cells[cell][name] += i in taken[name]
    out["heldout_text"] = {c: {"draws": n["draws"], "refb": n["refb"], "pool": n["pool"]} for c, n in cells.items()}
    return out
